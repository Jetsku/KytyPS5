#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/passes/ExecSelectElimination.h"
#include "graphics/shader/recompiler/ir/passes/ReadLaneElimination.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/SsaRewrite.h"
#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler {

namespace {

const char* GetDumpLabel(const CompileOptions& options) {
	return options.dump_label != nullptr ? options.dump_label : "ShaderRecompiler";
}

std::string MakeIrDump(std::string_view cfg, const IR::Program& ir) {
	std::string dump = "CFG:\n";
	dump += cfg;
	dump += "\nIR:\n";
	dump += fmt::format("mode={} scratch_dwords={}\n",
	                    ir.dispatcher_fallback ? "dispatcher" : "structured", ir.scratch_dwords);
	dump += IR::ProgramToString(ir);
	return dump;
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Compute: return "CS";
		case ShaderType::Vertex: return "VS";
		case ShaderType::Local: return "LS";
		case ShaderType::TessellationControl: return "HS";
		case ShaderType::TessellationEvaluation: return "TES";
		case ShaderType::Mesh: return "MS";
		case ShaderType::Pixel: return "PS";
		default: return "unknown";
	}
}

void LogDispatcherFallback(const CompileOptions& options, const CFG::Graph& cfg, const char* phase) {
	const auto* block        = cfg.FindBlock(cfg.failure_block);
	const auto  start        = block != nullptr ? block->start_pc : UINT32_MAX;
	const auto  end          = block != nullptr ? block->end_pc : UINT32_MAX;
	const auto  predecessors = block != nullptr ? block->predecessors.size() : 0u;
	const auto  successors   = block != nullptr ? block->successors.size() : 0u;
	LOGF("%s CFG dispatcher fallback: stage=%s hash=0x%016" PRIx64
	     " phase=%s failure=%s block=%" PRIu32 " pc=0x%08" PRIx32 "..0x%08" PRIx32 " preds=%" PRIu64
	     " succs=%" PRIu64 " blocks=%" PRIu64 " loops=%" PRIu64 " back_edges=%" PRIu64
	     " reason=%s\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash, phase,
	     CFG::FailureKindToString(cfg.failure_kind).c_str(), cfg.failure_block, start, end,
	     static_cast<uint64_t>(predecessors), static_cast<uint64_t>(successors),
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), cfg.unsupported_reason.c_str());
	// The dispatcher is rare: the console names each guest shader that takes it (the first 64, then
	// every 64th), so a title's log shows whether KYTY_DISPATCHER_CAP can apply to it.
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	size_t                              count = 0;
	{
		std::scoped_lock lock(mutex);
		if (!seen.insert(options.shader_hash).second) {
			return;
		}
		count = seen.size();
	}
	if (count <= 64u || count % 64u == 0u) {
		const auto cap = GetCodegenOptions().dispatcher_cap;
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader: CFG dispatcher #{}: {} 0x{:016x} ({}; {})\n", count, StageName(options.stage),
		    options.shader_hash, CFG::FailureKindToString(cfg.failure_kind),
		    cap != 0 ? fmt::format("an invocation leaves it after {} block transitions, "
		                           "KYTY_DISPATCHER_CAP",
		                           cap)
		             : std::string("no transition cap, KYTY_DISPATCHER_CAP=0")));
	}
}

enum class EmbeddedFetchValueType {
	Unknown,
	Constant,
	AttribTable,
	Attrib,
	BufferTable,
	Buffer
};

struct EmbeddedFetchSgprInfo {
	EmbeddedFetchValueType type      = EmbeddedFetchValueType::Unknown;
	int                    attrib_id = 0;
	uint32_t               value     = 0;
};

using EmbeddedFetchVectorLanes = std::map<uint64_t, EmbeddedFetchSgprInfo>;

uint64_t EmbeddedFetchVectorLaneKey(uint32_t reg, uint32_t lane) {
	return (static_cast<uint64_t>(reg) << 32u) | lane;
}

uint32_t EmbeddedFetchLane(uint32_t lane, uint32_t wave_size) {
	return wave_size == 32 || wave_size == 64 ? lane % wave_size : lane;
}

void ClearEmbeddedFetchVectorLanes(EmbeddedFetchVectorLanes* lanes, uint32_t reg) {
	const auto first = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg, 0));
	const auto last  = lanes->lower_bound(EmbeddedFetchVectorLaneKey(reg + 1u, 0));
	lanes->erase(first, last);
}

bool IsDecodedSgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Sgpr || op.kind == Decoder::OperandKind::VccLo ||
	       op.kind == Decoder::OperandKind::VccHi;
}

uint32_t DecodedSgprReg(const Decoder::Operand& op) {
	switch (op.kind) {
		case Decoder::OperandKind::VccLo: return 106u;
		case Decoder::OperandKind::VccHi: return 107u;
		default: return op.reg;
	}
}

bool IsDecodedVgpr(const Decoder::Operand& op) {
	return op.kind == Decoder::OperandKind::Vgpr;
}

uint32_t DecodedDstSize(const Decoder::Instruction& inst) {
	return std::max(inst.data_dwords, 1u);
}

uint32_t EmbeddedFetchDstSize(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::V_MAD_U64_U32 ? 2u : DecodedDstSize(inst);
}

void ClearEmbeddedFetchSgprs(std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                             const Decoder::Operand& dst, uint32_t size) {
	if (!IsDecodedSgpr(dst)) {
		return;
	}
	const auto register_id = DecodedSgprReg(dst);
	for (uint32_t i = 0; i < size && register_id + i < sgprs.size(); i++) {
		sgprs[register_id + i] = {};
	}
}

bool TryDecodedOperandConstant(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                               const Decoder::Operand& op, uint32_t& value) {
	switch (op.kind) {
		case Decoder::OperandKind::LiteralConstant:
		case Decoder::OperandKind::IntegerInlineConstant:
		case Decoder::OperandKind::FloatInlineConstant: value = op.value; return true;
		case Decoder::OperandKind::Null: value = 0; return true;
		default: break;
	}
	if (IsDecodedSgpr(op) && DecodedSgprReg(op) < sgprs.size() &&
	    sgprs[DecodedSgprReg(op)].type == EmbeddedFetchValueType::Constant) {
		value = sgprs[DecodedSgprReg(op)].value;
		return true;
	}
	return false;
}

bool TryDecodedSmemOffset(const std::array<EmbeddedFetchSgprInfo, 108>& sgprs,
                          const Decoder::Instruction& inst, uint32_t& raw_offset) {
	uint32_t base = 0;
	if (!TryDecodedOperandConstant(sgprs, inst.src1, base)) {
		return false;
	}
	const auto value = static_cast<uint64_t>(base) + inst.offset;
	if (value > 0xffffffffull) {
		return false;
	}
	raw_offset = static_cast<uint32_t>(value);
	return true;
}

bool IsEmbeddedFetchSLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_LOAD_DWORD:
		case Decoder::Opcode::S_LOAD_DWORDX2:
		case Decoder::Opcode::S_LOAD_DWORDX4:
		case Decoder::Opcode::S_LOAD_DWORDX8:
		case Decoder::Opcode::S_LOAD_DWORDX16: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchBufferLoad(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_X:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XY:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZ:
		case Decoder::Opcode::BUFFER_LOAD_FORMAT_XYZW: return true;
		default: return false;
	}
}

bool IsEmbeddedFetchAttribPropagationAlu(const Decoder::Instruction& inst) {
	switch (inst.opcode) {
		case Decoder::Opcode::S_BFE_U32:
		case Decoder::Opcode::S_AND_B32:
		case Decoder::Opcode::S_ADD_I32:
		case Decoder::Opcode::S_ADD_U32:
		case Decoder::Opcode::S_LSHL_B32: return true;
		default: return false;
	}
}

int BufferTableAttribFromOffset(uint32_t raw_offset, int dword) {
	return static_cast<int>((raw_offset + static_cast<uint32_t>(dword) * 4u) / 16u);
}

Frontend::EmbeddedFetchPlan DetectEmbeddedVertexFetch(
    const Decoder::Program& decoded, const ShaderVertexInputInfo* input_info,
    uint32_t user_data_base, uint32_t user_data_count, uint32_t wave_size) {
	const uint32_t    vertex_index_reg   = input_info->logical_stage == ShaderType::Local ? 2u : 5u;
	const uint32_t    instance_index_reg = input_info->logical_stage == ShaderType::Local ? 5u : 8u;
	Frontend::EmbeddedFetchPlan data;
	data.loads.reserve(input_info->resources_num);
	int32_t vertex_offset_candidate   = -1;
	int32_t instance_offset_candidate = -1;
	bool    vertex_offset_conflict    = false;
	bool    instance_offset_conflict  = false;

	const int shift_regs = 8;
	const int attrib_reg = input_info->fetch_attrib_reg + shift_regs;
	const int buffer_reg = input_info->fetch_buffer_reg + shift_regs;

	std::array<EmbeddedFetchSgprInfo, 108> sgprs {};
	std::array<bool, 256>                 vgpr_is_index {};
	EmbeddedFetchVectorLanes               vector_lanes;
	const bool                             track_vector_lanes =
	    std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
	                 [](const auto& inst) {
		                 return Decoder::IsDirectBranch(inst.opcode) ||
		                        inst.opcode == Decoder::Opcode::S_SETPC_B64;
	                 });

	if (attrib_reg >= 0 && attrib_reg < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg].type = EmbeddedFetchValueType::AttribTable;
	}
	if (attrib_reg + 1 >= 0 && attrib_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[attrib_reg + 1].type = EmbeddedFetchValueType::AttribTable;
	}
	if (buffer_reg >= 0 && buffer_reg < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg].type = EmbeddedFetchValueType::BufferTable;
	}
	if (buffer_reg + 1 >= 0 && buffer_reg + 1 < static_cast<int>(sgprs.size())) {
		sgprs[buffer_reg + 1].type = EmbeddedFetchValueType::BufferTable;
	}

	for (const auto& inst: decoded.instructions) {
		// Fetch shaders accumulate the draw's vertex offset in v0. The PS5 NGG ABI
		// seeds S_NGG_VERTEX_INDEX in v5 and S_NGG_INSTANCE_INDEX in v8, then applies the
		// corresponding direct-draw offsets before fetching.
		const bool vertex_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == 0 || (user_data_base == 8 && inst.dst.reg == vertex_index_reg));
		const bool instance_index_accumulator =
		    IsDecodedVgpr(inst.dst) &&
		    (inst.dst.reg == (user_data_base == 8 ? instance_index_reg : 3u));
		uint32_t   sad_zero = 0;
		const bool index_offset_add =
		    (vertex_index_accumulator || instance_index_accumulator) && IsDecodedSgpr(inst.src0) &&
		    ((inst.opcode == Decoder::Opcode::V_ADD_I32 && IsDecodedVgpr(inst.src1) &&
		      inst.src1.reg == inst.dst.reg) ||
		     (user_data_base == 8 &&
		      (inst.dst.reg == vertex_index_reg || inst.dst.reg == instance_index_reg) &&
		      inst.opcode == Decoder::Opcode::V_SAD_U32 && IsDecodedVgpr(inst.src2) &&
		      inst.src2.reg == inst.dst.reg &&
		      TryDecodedOperandConstant(sgprs, inst.src1, sad_zero) && sad_zero == 0));
		if (data.loads.empty() && index_offset_add) {
			const auto reg = DecodedSgprReg(inst.src0);
			if (reg >= user_data_base && reg - user_data_base < user_data_count) {
				auto& candidate = vertex_index_accumulator ? vertex_offset_candidate
				                                           : instance_offset_candidate;
				auto& conflict  = vertex_index_accumulator ? vertex_offset_conflict
				                                           : instance_offset_conflict;
				if (candidate >= 0 && candidate != static_cast<int32_t>(reg)) {
					conflict = true;
				} else {
					candidate = static_cast<int32_t>(reg);
				}
			}
		}
		switch (inst.opcode) {
			case Decoder::Opcode::V_WRITELANE_B32: {
				uint32_t lane = 0;
				if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
					vgpr_is_index[inst.dst.reg] = false;
				}
				if (track_vector_lanes && IsDecodedVgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size() &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					vector_lanes[EmbeddedFetchVectorLaneKey(inst.dst.reg,
					                                        EmbeddedFetchLane(lane, wave_size))] =
					    sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedVgpr(inst.dst)) {
					ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
				}
				break;
			}
			case Decoder::Opcode::V_READLANE_B32: {
				uint32_t lane = 0;
				if (track_vector_lanes && IsDecodedSgpr(inst.dst) &&
				    DecodedSgprReg(inst.dst) < sgprs.size() && IsDecodedVgpr(inst.src0) &&
				    TryDecodedOperandConstant(sgprs, inst.src1, lane)) {
					const auto found = vector_lanes.find(EmbeddedFetchVectorLaneKey(
					    inst.src0.reg, EmbeddedFetchLane(lane, wave_size)));
					sgprs[DecodedSgprReg(inst.dst)] =
					    found != vector_lanes.end() ? found->second : EmbeddedFetchSgprInfo {};
				} else if (IsDecodedSgpr(inst.dst)) {
					ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
				}
				break;
			}
			case Decoder::Opcode::S_MOV_B32:
				if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
				    DecodedSgprReg(inst.src0) < sgprs.size()) {
					sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
				} else if (IsDecodedSgpr(inst.dst)) {
					uint32_t value = 0;
					if (TryDecodedOperandConstant(sgprs, inst.src0, value)) {
						auto& dst = sgprs[DecodedSgprReg(inst.dst)];
						dst.type  = EmbeddedFetchValueType::Constant;
						dst.value = value;
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
					}
				}
				break;
			case Decoder::Opcode::S_MOVK_I32:
				if (IsDecodedSgpr(inst.dst)) {
					auto& dst = sgprs[DecodedSgprReg(inst.dst)];
					dst.type  = EmbeddedFetchValueType::Constant;
					dst.value = inst.src0.value;
				}
				break;
			default:
				if (IsEmbeddedFetchSLoad(inst)) {
					if (IsDecodedSgpr(inst.src0) && DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type ==
					        EmbeddedFetchValueType::AttribTable) {
						uint32_t raw_offset = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							const auto register_id = DecodedSgprReg(inst.dst);
							const int  index       = static_cast<int>(raw_offset / 4u);
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Attrib;
								dst.attrib_id    = index + static_cast<int>(i);
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else if (IsDecodedSgpr(inst.src0) &&
					           DecodedSgprReg(inst.src0) < sgprs.size() &&
					           sgprs[DecodedSgprReg(inst.src0)].type ==
					               EmbeddedFetchValueType::BufferTable) {
						const auto register_id = DecodedSgprReg(inst.dst);
						uint32_t   raw_offset  = 0;
						if (TryDecodedSmemOffset(sgprs, inst, raw_offset)) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst = sgprs[register_id + i];
								dst.type  = EmbeddedFetchValueType::Buffer;
								dst.attrib_id =
								    BufferTableAttribFromOffset(raw_offset, static_cast<int>(i));
							}
						} else if (!GetCodegenOptions().embedded_fetch_no_sgpr_offset &&
						           IsDecodedSgpr(inst.src1) &&
						           DecodedSgprReg(inst.src1) < sgprs.size() &&
						           sgprs[DecodedSgprReg(inst.src1)].type ==
						               EmbeddedFetchValueType::Attrib &&
						           (inst.offset & 0x3u) == 0) {
							for (uint32_t i = 0;
							     i < DecodedDstSize(inst) && register_id + i < sgprs.size(); i++) {
								auto& dst        = sgprs[register_id + i];
								dst.type         = EmbeddedFetchValueType::Buffer;
								dst.attrib_id    = sgprs[DecodedSgprReg(inst.src1)].attrib_id;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
						}
					} else {
						ClearEmbeddedFetchSgprs(sgprs, inst.dst, DecodedDstSize(inst));
					}
				} else if (inst.opcode == Decoder::Opcode::V_CNDMASK_B32) {
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size()) {
						ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg);
					}
					if (IsDecodedVgpr(inst.dst) && inst.dst.reg < vgpr_is_index.size() &&
					    IsDecodedVgpr(inst.src0) && inst.src0.reg == instance_index_reg &&
					    IsDecodedVgpr(inst.src1) && inst.src1.reg == vertex_index_reg) {
						vgpr_is_index[inst.dst.reg] = true;
					}
				} else if (IsEmbeddedFetchAttribPropagationAlu(inst)) {
					if (IsDecodedSgpr(inst.dst) && IsDecodedSgpr(inst.src0) &&
					    DecodedSgprReg(inst.src0) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src0)].type == EmbeddedFetchValueType::Attrib) {
						sgprs[DecodedSgprReg(inst.dst)] = sgprs[DecodedSgprReg(inst.src0)];
					} else if (IsDecodedSgpr(inst.dst)) {
						uint32_t src0 = 0;
						uint32_t src1 = 0;
						if (TryDecodedOperandConstant(sgprs, inst.src0, src0) &&
						    TryDecodedOperandConstant(sgprs, inst.src1, src1)) {
							auto& dst = sgprs[DecodedSgprReg(inst.dst)];
							dst.type  = EmbeddedFetchValueType::Constant;
							switch (inst.opcode) {
								case Decoder::Opcode::S_AND_B32: dst.value = src0 & src1; break;
								case Decoder::Opcode::S_LSHL_B32:
									dst.value = src0 << (src1 & 31u);
									break;
								case Decoder::Opcode::S_BFE_U32:
									dst.value = src0 >> (src1 & 31u);
									break;
								default: dst.value = src0 + src1; break;
							}
						} else {
							ClearEmbeddedFetchSgprs(sgprs, inst.dst, 1);
						}
					}
				} else if (IsEmbeddedFetchBufferLoad(inst)) {
					if (IsDecodedVgpr(inst.src0) && inst.src0.reg < vgpr_is_index.size() &&
					    vgpr_is_index[inst.src0.reg] &&
					    IsDecodedSgpr(inst.src1) && DecodedSgprReg(inst.src1) < sgprs.size() &&
					    sgprs[DecodedSgprReg(inst.src1)].type == EmbeddedFetchValueType::Buffer) {
						const auto& buffer = sgprs[DecodedSgprReg(inst.src1)];
						if (data.loads.empty()) {
							if (!vertex_offset_conflict) {
								data.vertex_offset_sgpr = vertex_offset_candidate;
							}
							if (!instance_offset_conflict) {
								data.instance_offset_sgpr = instance_offset_candidate;
							}
						}
						auto& load        = data.loads.emplace_back();
						load.pc           = inst.pc;
						load.attrib_id    = buffer.attrib_id;
						load.components   = DecodedDstSize(inst);
					}
				}
				break;
		}
		if (inst.opcode == Decoder::Opcode::V_MOVRELD_B32) {
			vector_lanes.clear();
		} else if (inst.opcode != Decoder::Opcode::V_WRITELANE_B32 && IsDecodedVgpr(inst.dst)) {
			for (uint32_t i = 0;
			     i < EmbeddedFetchDstSize(inst) && inst.dst.reg + i < vgpr_is_index.size();
			     i++) {
				ClearEmbeddedFetchVectorLanes(&vector_lanes, inst.dst.reg + i);
			}
		}
	}

	return data;
}

// Marks every BVH instruction the decoder produced as unsupported when no BVH mode is on (fused
// and front programs decode all instructions), and logs each BVH shader once.
void NoteBvhInstructions(const CompileOptions& options, Decoder::Program& decoded,
                         bool decode_bvh) {
	uint32_t count    = 0;
	uint32_t first_pc = UINT32_MAX;
	for (auto& inst: decoded.instructions) {
		if (!Decoder::IsBvhIntersect(inst)) {
			continue;
		}
		decoded.has_bvh = true;
		count++;
		first_pc = std::min(first_pc, inst.pc);
		if (!decode_bvh && inst.opcode != Decoder::Opcode::UNSUPPORTED) {
			Decoder::SetUnsupported(
			    inst, Decoder::Family::MIMG, inst.opcode_id,
			    "BVH ray intersection is disabled (KYTY_RT_SOFTWARE=1 or KYTY_RT_STUB=1 enables it)");
		}
	}
	if (count == 0 || !decode_bvh) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> logged;
	{
		std::scoped_lock lock(mutex);
		if (!logged.insert(options.shader_hash ^ static_cast<uint64_t>(options.stage)).second) {
			return;
		}
	}
	const bool software = GetCodegenOptions().rt_software;
	Log::WriteToConsoleAndLog(fmt::format(
	    "{}: {} shader 0x{:016x} has {} BVH intersection instruction(s), first at pc=0x{:08x}{}.\n",
	    software ? "KYTY_RT_SOFTWARE" : "KYTY_RT_STUB", StageName(options.stage),
	    options.shader_hash, count, first_pc, software ? "" : "; every ray misses"));
}

Decoder::Program DecodeFusedProgram(std::span<const uint32_t> front, std::span<const uint32_t> back,
                                    std::vector<uint32_t>& joined_code, bool decode_bvh) {
	EXIT_IF(back.empty());
	auto       result      = Decoder::DecodeFrontProgram(front);
	const auto front_words = static_cast<uint32_t>(result.code.size());
	joined_code.assign(result.code.begin(), result.code.end());
	joined_code.insert(joined_code.end(), back.begin(), back.end());
	// The merged-stage ABI passes the back shader in s[6:7]. Give that handoff an
	// ordinary CFG edge, retaining both bodies in one register and LDS lifetime.
	joined_code[front_words - 1u] = 0xbf820000u; // s_branch to the following instruction
	result.instructions.back()    = {};
	Decoder::DecodeInstruction(joined_code, front_words - 1u, result.instructions.back());
	Decoder::Program back_program;
	Decoder::DecodeProgram(back, back_program, decode_bvh);
	const auto back_pc = front_words * sizeof(uint32_t);
	for (auto& inst: back_program.instructions) {
		// A back-stage PC-relative data reference requires its guest code address.
		EXIT_NOT_IMPLEMENTED(inst.opcode == Decoder::Opcode::S_GETPC_B64);
		inst.pc += back_pc;
		inst.branch_target += back_pc;
		result.instructions.push_back(std::move(inst));
	}
	result.code = joined_code;
	return result;
}

} // namespace

// KYTY_SRT_VARIANT_READS: logs each program the switch skips instead of exiting, once per shader:
// a descriptor it computes at runtime that has no BDA path (see TrackResources; without the switch
// its flat SRT slots fail to evaluate and its dispatches or draws are dropped just the same), or an
// S_SWAPPC_B64 call (without the switch, the CFG build exits).
static void NoteSkippedProgram(const CompileOptions& options, uint32_t pc, std::string_view reason) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> logged;
	{
		std::scoped_lock lock(mutex);
		if (!logged.insert(options.shader_hash ^ static_cast<uint64_t>(options.stage)).second) {
			return;
		}
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "KYTY_SRT_VARIANT_READS: {} shader 0x{:016x} {} (pc=0x{:08x}); its dispatches and draws are "
	    "skipped.\n",
	    StageName(options.stage), options.shader_hash, reason, pc));
}

// S_SWAPPC_B64 with a destination: a call through a function pointer (the NULL-destination jump
// decodes as S_SETPC_B64). Psr's shader-mesh BVH builders fetch vertices through such callbacks.
static bool IsUnsupportedCall(const Decoder::Instruction& inst) {
	return inst.opcode == Decoder::Opcode::UNSUPPORTED && inst.family == Decoder::Family::SOP1 &&
	       inst.opcode_id == 0x21u;
}

// The SGPRs an instruction writes: its destination's width follows from the opcode (64-bit
// scalar ops and lane masks write a pair, SMEM loads their dword count); a second destination
// (carry-out) counts as a pair. Unsupported instructions other than calls write nothing known;
// calls write their return address to their SDST pair.
// `wave32` (KYTY_CALL_TRACE_WAVE32, a wave32 program): a lane mask (V_CMP result, carry-out) is
// one SGPR, not a pair. PPSA03671's material kernels keep their SRT pointer in s12..s13 and write
// V_CMP masks to s11, which the pair width took for writes of s12.
static bool WritesSgpr(const Decoder::Instruction& inst, uint32_t reg, bool wave32 = false) {
	if (inst.opcode == Decoder::Opcode::UNSUPPORTED) {
		const auto sdst = (inst.raw[0] >> 16u) & 0x7fu;
		return IsUnsupportedCall(inst) && (sdst == reg || sdst + 1u == reg);
	}
	const auto name  = magic_enum::enum_name(inst.opcode);
	const auto width = [&]() -> uint32_t {
		for (const auto& [suffix, dwords]: {std::pair {std::string_view("DWORDX16"), 16u},
		                                    std::pair {std::string_view("DWORDX8"), 8u},
		                                    std::pair {std::string_view("DWORDX4"), 4u},
		                                    std::pair {std::string_view("DWORDX3"), 3u},
		                                    std::pair {std::string_view("DWORDX2"), 2u}}) {
			if (name.ends_with(suffix)) return dwords;
		}
		// A 32-bit result of a 64-bit source (S_FF1_I32_B64, S_BCNT1_I32_B64, S_FLBIT_I32_B64): one.
		if (name.ends_with("32_B64")) return 1u;
		if (wave32 && name.starts_with("V_CMP")) return 1u;
		return name.ends_with("_B64") || name.ends_with("_U64") || name.ends_with("_I64") ||
		               name.ends_with("_F64") || name.starts_with("V_CMP")
		           ? 2u
		           : 1u;
	}();
	const auto covers = [&](const Decoder::Operand& operand, uint32_t count) {
		return operand.kind == Decoder::OperandKind::Sgpr && reg >= operand.reg &&
		       reg < operand.reg + count;
	};
	return covers(inst.dst, width) || covers(inst.dst2, wave32 ? 1u : 2u);
}

// Diagnostics for a program skipped for S_SWAPPC_B64: the functions its calls jump to, as the
// dispatch's user data gives them. Psr's shader-mesh builders copy a callback record from user
// SGPRs into their link registers before each call (s_mov_b32 s14, s8; s_mov_b32 s15, s9;
// ...; s_swappc_b64 s[14:15], s[14:15]) and use those registers for other values in between. So
// each call's target pair is traced backwards through the straight-line code before it, up to the
// last write of each half: an S_MOV_B32 per half or an S_MOV_B64 from user SGPRs that the program
// never writes, or the user SGPRs themselves when nothing before the call writes them. Anything
// else (another write, a branch or branch target on the way, an odd or non-user pair) leaves the
// call `unresolved`: its callee is unknown.
// `per_call`, when given, receives (call pc, user SGPR of the target pair) for every resolved call.
static std::vector<CallTarget>
FindCallTargets(const Decoder::Program& decoded, const CompileOptions& options, uint32_t& unresolved,
                std::vector<std::pair<uint32_t, uint32_t>>* per_call = nullptr) {
	unresolved             = 0;
	const auto& insts      = decoded.instructions;
	const auto  user_sgpr  = [&](uint32_t reg) {
        return reg >= options.user_data_base &&
               reg - options.user_data_base < options.user_data.size();
	};
	const bool wave32      = GetCodegenOptions().call_trace_wave32 && options.wave_size == 32u;
	const auto writes_sgpr = [wave32](const Decoder::Instruction& inst, uint32_t reg) {
		return WritesSgpr(inst, reg, wave32);
	};
	std::set<uint32_t> labels;
	for (const auto& inst: insts) {
		if (Decoder::IsDirectBranch(inst.opcode)) {
			labels.insert(inst.branch_target);
		}
	}
	// Whether nothing before instruction `end` (in program order) writes `reg`. Writes after the
	// copy do not matter: the builders reuse their callback registers only after the last call.
	const auto unwritten_before = [&](uint32_t reg, size_t end) {
		return std::none_of(insts.begin(), insts.begin() + static_cast<std::ptrdiff_t>(end),
		                    [&](const auto& inst) { return writes_sgpr(inst, reg); });
	};
	// The entry block: instructions before the first branch, SETPC or branch target. A write there
	// dominates every later instruction.
	size_t entry_end = insts.size();
	for (size_t i = 0; i < insts.size(); i++) {
		if (i != 0 && labels.contains(insts[i].pc)) {
			entry_end = i;
			break;
		}
		if (Decoder::IsDirectBranch(insts[i].opcode) || insts[i].opcode == Decoder::Opcode::S_SETPC_B64) {
			entry_end = i + 1;
			break;
		}
	}
	// A copy (S_MOV_B32, or one half of S_MOV_B64) of a user SGPR into `reg`: its source, or
	// UINT32_MAX.
	const auto copy_source = [&](const Decoder::Instruction& inst, uint32_t reg) -> uint32_t {
		const bool b64 = inst.opcode == Decoder::Opcode::S_MOV_B64;
		if ((inst.opcode != Decoder::Opcode::S_MOV_B32 && !b64) ||
		    inst.dst.kind != Decoder::OperandKind::Sgpr ||
		    inst.src0.kind != Decoder::OperandKind::Sgpr || reg < inst.dst.reg ||
		    reg - inst.dst.reg >= (b64 ? 2u : 1u)) {
			return UINT32_MAX;
		}
		return inst.src0.reg + (reg - inst.dst.reg);
	};
	std::vector<CallTarget> targets;
	for (size_t call = 0; call < insts.size(); call++) {
		if (!IsUnsupportedCall(insts[call])) {
			continue;
		}
		const auto target      = insts[call].raw[0] & 0xffu;          // SSRC0
		const auto return_sgpr = (insts[call].raw[0] >> 16u) & 0x7fu; // SDST
		uint32_t   halves[2]   = {UINT32_MAX, UINT32_MAX};            // user SGPR per half
		size_t     copies[2]   = {call, call};                        // where each half is read
		bool       ok          = target < 104u && (target & 1u) == 0u;
		bool       reached_top = true;
		bool crossed = false; // straight-line code ended before both halves were found
		for (size_t i = call; ok && i-- > 0;) {
			const auto& inst = insts[i];
			// Earlier calls count as writing only their return pair: the builders reload their
			// link registers from the same user SGPRs after each call, so the callees keep those.
			if (labels.contains(insts[i + 1].pc) || Decoder::IsDirectBranch(inst.opcode) ||
			    inst.opcode == Decoder::Opcode::S_SETPC_B64) {
				crossed     = true;
				reached_top = false;
				break;
			}
			for (uint32_t half = 0; half < 2; half++) {
				if (halves[half] != UINT32_MAX || !writes_sgpr(inst, target + half)) {
					continue;
				}
				const auto source = copy_source(inst, target + half);
				if (source != UINT32_MAX) {
					halves[half] = source;
					copies[half] = i;
				} else {
					ok = false;
				}
			}
			if (halves[0] != UINT32_MAX && halves[1] != UINT32_MAX) {
				reached_top = false;
				break;
			}
		}
		for (uint32_t half = 0; ok && crossed && half < 2; half++) {
			if (halves[half] != UINT32_MAX) {
				continue;
			}
			// Across control flow: every write of the half before the call must copy the same
			// source, and one of them must be in the entry block (so it reaches every path).
			uint32_t source    = UINT32_MAX;
			size_t   last      = call; // the latest copy: the source must be intact up to it
			bool     dominates = false;
			for (size_t i = 0; ok && i < call; i++) {
				if (!writes_sgpr(insts[i], target + half)) {
					continue;
				}
				const auto copied = copy_source(insts[i], target + half);
				ok        = copied != UINT32_MAX && (source == UINT32_MAX || copied == source);
				source    = copied;
				last      = i;
				dominates = dominates || i < entry_end;
			}
			if (ok && source == UINT32_MAX) {
				// Nothing before the call writes the half: it holds its entry value.
				source    = target + half;
				dominates = true;
			}
			ok           = ok && dominates;
			halves[half] = source;
			copies[half] = last;
		}
		if (ok && reached_top) {
			// Nothing before the call writes the open halves: they hold their entry values.
			for (uint32_t half = 0; half < 2; half++) {
				if (halves[half] == UINT32_MAX) {
					halves[half] = target + half;
				}
			}
		}
		ok = ok && halves[0] != UINT32_MAX && halves[1] == halves[0] + 1u && user_sgpr(halves[0]) &&
		     user_sgpr(halves[1]) && unwritten_before(halves[0], copies[0]) &&
		     unwritten_before(halves[1], copies[1]);
		if (!ok) {
			unresolved++;
			continue;
		}
		if (per_call != nullptr) {
			per_call->emplace_back(insts[call].pc, halves[0]);
		}
		const auto low  = static_cast<uint64_t>(options.user_data[halves[0] - options.user_data_base]);
		const auto high = static_cast<uint64_t>(options.user_data[halves[1] - options.user_data_base]);
		const auto address = (low | (high << 32u)) & 0x0000ffffffffffffull;
		if (std::ranges::none_of(targets, [&](const CallTarget& known) {
			    return known.user_sgpr == halves[0] && known.return_sgpr == return_sgpr;
		    })) {
			targets.push_back({address, halves[0], return_sgpr});
		}
	}
	return targets;
}

static std::string DescribeCallTargets(std::span<const CallTarget> targets, uint32_t unresolved) {
	std::string text;
	for (const auto& target: targets) {
		text += fmt::format("{}s[{}:{}]=0x{:x}", text.empty() ? "" : ", ", target.user_sgpr,
		                    target.user_sgpr + 1u, target.address);
	}
	if (unresolved != 0) {
		text += fmt::format("{}{} call(s) with a target not copied from user data",
		                    text.empty() ? "" : ", ", unresolved);
	}
	return text.empty() ? "no callee found" : text;
}

// Diagnostics for S_SWAPPC_B64 calls whose target pair is read from a table (CallTable): Miles
// Morales' ray-tracing shaders pick a hit callback per lane in a waterfall loop:
//   s_lshl_b32 vcc_lo, s8, 4
//   s_buffer_load_dwordx4 s[8:11], s[72:75], vcc_lo   ; 16-byte record: function, data pointer
//   s_mov_b32 s14, s8 ; s_mov_b32 s15, s9 ; ... ; s_swappc_b64 s[14:15], s[14:15]
// with the table V# s[72:75] loaded blocks earlier through the SRT (user s12 -> +544 -> +64). Each
// register range is traced to its definition: the last write in the straight-line code before the
// use, else the only instruction anywhere in the program that writes it (so it reaches every
// path), else the user SGPRs when nothing writes them. Only scalar loads with an immediate offset
// are followed; anything else leaves the call out.
static std::vector<CallTable> FindCallTables(const Decoder::Program& decoded,
                                             const CompileOptions& options, std::string& failures) {
	const auto& insts     = decoded.instructions;
	const bool  wave32    = GetCodegenOptions().call_trace_wave32 && options.wave_size == 32u;
	const auto  WritesSgpr32 = [wave32](const Decoder::Instruction& inst, uint32_t reg) {
		return WritesSgpr(inst, reg, wave32);
	};
	const auto  user_sgpr = [&](uint32_t reg) {
        return reg >= options.user_data_base &&
               reg - options.user_data_base < options.user_data.size();
	};
	std::set<uint32_t> labels;
	for (const auto& inst: insts) {
		if (Decoder::IsDirectBranch(inst.opcode)) {
			labels.insert(inst.branch_target);
		}
	}
	constexpr ptrdiff_t Top     = -1; // nothing writes the register: its entry value
	constexpr ptrdiff_t Unknown = -2;
	// The straight-line writer of `reg` before instruction `at`, Top at the program start, or
	// Unknown when a branch or branch target comes first.
	const auto straight_writer = [&](uint32_t reg, size_t at) -> ptrdiff_t {
		for (size_t i = at; i-- > 0;) {
			if (labels.contains(insts[i + 1].pc) || Decoder::IsDirectBranch(insts[i].opcode) ||
			    insts[i].opcode == Decoder::Opcode::S_SETPC_B64) {
				return Unknown;
			}
			if (WritesSgpr32(insts[i], reg)) {
				return static_cast<ptrdiff_t>(i);
			}
		}
		return Top;
	};
	// KYTY_CALL_TRACE_REACH: instruction-level successors (fallthrough and direct branch edges),
	// built on first use; empty when an indirect branch makes them unknown.
	std::vector<std::vector<size_t>> successors;
	bool                             successors_built = false;
	const auto                       build_successors = [&]() {
		if (successors_built) {
			return !successors.empty();
		}
		successors_built = true;
		std::unordered_map<uint32_t, size_t> index_of_pc;
		for (size_t i = 0; i < insts.size(); i++) {
			if (insts[i].opcode == Decoder::Opcode::S_SETPC_B64) {
				return false;
			}
			index_of_pc.emplace(insts[i].pc, i);
		}
		successors.resize(insts.size());
		for (size_t i = 0; i < insts.size(); i++) {
			const auto& inst = insts[i];
			if (inst.opcode == Decoder::Opcode::S_ENDPGM) {
				continue;
			}
			if (Decoder::IsDirectBranch(inst.opcode)) {
				const auto target = index_of_pc.find(inst.branch_target);
				if (target == index_of_pc.end()) {
					successors.clear();
					return false;
				}
				successors[i].push_back(target->second);
				if (inst.opcode == Decoder::Opcode::S_BRANCH) {
					continue;
				}
			}
			if (i + 1 < insts.size()) {
				successors[i].push_back(i + 1);
			}
		}
		return true;
	};
	// Whether `to` is reachable from the successors of `from`, or (from == SIZE_MAX) from the
	// entry, never passing `avoid`.
	const auto reaches = [&](size_t from, size_t to, size_t avoid) {
		std::vector<uint8_t> seen(insts.size(), 0u);
		std::vector<size_t>  work;
		if (from == SIZE_MAX) {
			work.push_back(0);
		} else {
			work = successors[from];
		}
		while (!work.empty()) {
			const auto node = work.back();
			work.pop_back();
			if (node == avoid || seen[node] != 0u) {
				continue;
			}
			if (node == to) {
				return true;
			}
			seen[node] = 1u;
			work.insert(work.end(), successors[node].begin(), successors[node].end());
		}
		return false;
	};
	const auto reach_definition = [&](uint32_t reg, uint32_t count, size_t at) -> ptrdiff_t {
		if (insts.empty() || !build_successors()) {
			return Unknown;
		}
		ptrdiff_t writer = Unknown;
		for (size_t i = 0; i < insts.size(); i++) {
			bool writes = IsUnsupportedCall(insts[i]);
			for (uint32_t r = reg; r < reg + count && !writes; r++) {
				writes = WritesSgpr32(insts[i], r);
			}
			if (!writes || !reaches(i, at, SIZE_MAX)) {
				continue;
			}
			if (writer != Unknown) {
				return Unknown; // two writers reach `at`
			}
			writer = static_cast<ptrdiff_t>(i);
		}
		if (writer == Unknown) {
			return Top; // nothing that writes the range reaches `at`: its entry value
		}
		const auto& inst = insts[static_cast<size_t>(writer)];
		if (IsUnsupportedCall(inst)) {
			return Unknown;
		}
		for (uint32_t r = reg; r < reg + count; r++) {
			if (!WritesSgpr32(inst, r)) {
				return Unknown;
			}
		}
		// Every path from the entry to `at` passes the writer.
		return reaches(SIZE_MAX, at, static_cast<size_t>(writer)) ? Unknown : writer;
	};
	// `also`: further identical writers when the definition crossed control flow.
	const auto definition = [&](uint32_t reg, uint32_t count, size_t at,
	                            std::vector<size_t>& also) -> ptrdiff_t {
		also.clear();
		ptrdiff_t found   = Unknown;
		bool      crossed = false;
		for (uint32_t r = reg; r < reg + count; r++) {
			const auto writer = straight_writer(r, at);
			if (writer == Unknown) {
				crossed = true;
				break;
			}
			if (found != Unknown && found != writer) {
				return Unknown;
			}
			found = writer;
		}
		if (!crossed) {
			return found;
		}
		const auto writes_range = [&](size_t i) {
			for (uint32_t r = reg; r < reg + count; r++) {
				if (WritesSgpr32(insts[i], r)) return true;
			}
			return false;
		};
		// Across control flow, the reaching definition: the last write before `at` in program
		// order, one instruction that writes the whole range, provided no branch from before it
		// lands between it and `at` (it dominates `at`), and no later write can loop back to `at`
		// (no backward branch at or after such a write targets `at` or earlier).
		{
			ptrdiff_t candidate = Unknown;
			for (size_t i = at; i-- > 0;) {
				if (writes_range(i)) {
					candidate = static_cast<ptrdiff_t>(i);
					break;
				}
			}
			bool ok = candidate >= 0;
			for (uint32_t r = reg; ok && r < reg + count; r++) {
				ok = WritesSgpr32(insts[static_cast<size_t>(candidate)], r);
			}
			const auto cand_pc = ok ? insts[static_cast<size_t>(candidate)].pc : 0u;
			for (size_t i = 0; ok && i < static_cast<size_t>(candidate); i++) {
				ok = !Decoder::IsDirectBranch(insts[i].opcode) ||
				     !(insts[i].branch_target > cand_pc && insts[i].branch_target <= insts[at].pc);
			}
			size_t first_later = insts.size();
			for (size_t i = at; ok && i < insts.size(); i++) {
				if (first_later == insts.size() && writes_range(i)) {
					first_later = i;
				}
				ok = first_later == insts.size() || !Decoder::IsDirectBranch(insts[i].opcode) ||
				     insts[i].branch_target > insts[at].pc;
			}
			for (const auto& inst: insts) {
				ok = ok && inst.opcode != Decoder::Opcode::S_SETPC_B64;
			}
			if (ok) {
				return candidate;
			}
			// KYTY_CALL_TRACE_WAVE32: nothing before `at` writes the range and no later write can
			// loop back to `at`: the range holds its entry value (PPSA03671's material kernels
			// reload s[8:15] from s12 only after their callback loop).
			if (candidate == Unknown && GetCodegenOptions().call_trace_wave32) {
				bool entry = true;
				size_t later = insts.size();
				for (size_t i = at; entry && i < insts.size(); i++) {
					if (later == insts.size() && writes_range(i)) {
						later = i;
					}
					entry = later == insts.size() || !Decoder::IsDirectBranch(insts[i].opcode) ||
					        insts[i].branch_target > insts[at].pc;
				}
				for (const auto& inst: insts) {
					entry = entry && inst.opcode != Decoder::Opcode::S_SETPC_B64;
				}
				if (entry) {
					return Top;
				}
			}
		}
		// KYTY_CALL_TRACE_REACH (adapted from usa-ratt/KytyPS5-GT7 a6cd035f0): the instruction-level
		// control flow graph decides. The writers from which `at` can be reached: none, and the
		// range holds its entry value; one that writes the whole range and that every path from
		// the entry to `at` passes, and it is the definition. A call (its callee is not known
		// here) counts as a writer, so a definition never crosses one that can reach `at`.
		if (GetCodegenOptions().call_trace_reach) {
			if (const auto reached = reach_definition(reg, count, at); reached != Unknown) {
				return reached;
			}
		}
		// Else every write of the range anywhere in the program must be the same
		// load (a compiler reload of one descriptor), at least one of them before `at`.
		found = Top;
		for (size_t i = 0; i < insts.size(); i++) {
			bool writes = false;
			for (uint32_t r = reg; r < reg + count && !writes; r++) {
				writes = WritesSgpr32(insts[i], r);
			}
			if (!writes) {
				continue;
			}
			if (found == Top) {
				found = static_cast<ptrdiff_t>(i);
				continue;
			}
			const auto& first = insts[static_cast<size_t>(found)];
			if (first.word_count != insts[i].word_count ||
			    !std::equal(first.raw, first.raw + first.word_count, insts[i].raw)) {
				return Unknown;
			}
			also.push_back(i);
		}
		return found != Top && static_cast<size_t>(found) >= at ? Unknown : found;
	};
	const auto smem_load = [](const Decoder::Instruction& inst, bool& buffer) {
		if (inst.family != Decoder::Family::SMEM || inst.dst.kind != Decoder::OperandKind::Sgpr) {
			return false;
		}
		const auto name = magic_enum::enum_name(inst.opcode);
		buffer          = name.starts_with("S_BUFFER_LOAD_DWORD");
		return buffer || name.starts_with("S_LOAD_DWORD");
	};
	// Traces s[reg:reg+count-1] before instruction `at`; `origin` is the first SGPR of the value
	// that holds them (the user SGPRs or the last step's destination).
	std::string resolve_why;
	std::function<bool(uint32_t, uint32_t, size_t, uint32_t, CallTable&, uint32_t&)> resolve;
	resolve = [&](uint32_t reg, uint32_t count, size_t at, uint32_t depth, CallTable& table,
	              uint32_t& origin) {
		if (depth > 8) {
			return false;
		}
		std::vector<size_t> also;
		const auto          def = definition(reg, count, at, also);
		if (def == Top) {
			if (!user_sgpr(reg) || !user_sgpr(reg + count - 1u)) {
				resolve_why = fmt::format("s{}..s{} unwritten but not user data", reg, reg + count - 1u);
				return false;
			}
			table.user_sgpr = reg;
			origin          = reg;
			return true;
		}
		if (def == Unknown) {
			std::string writers;
			for (size_t i = 0; i < insts.size(); i++) {
				for (uint32_t r = reg; r < reg + count; r++) {
					if (WritesSgpr32(insts[i], r)) {
						writers += fmt::format(" 0x{:x}:{}:s{}", insts[i].pc,
						                       magic_enum::enum_name(insts[i].opcode), r);
						break;
					}
				}
			}
			resolve_why = fmt::format("s{}..s{} before pc 0x{:x} has no unique definition; writers{}",
			                          reg, reg + count - 1u, insts[at].pc, writers);
			return false;
		}
		const auto& inst   = insts[static_cast<size_t>(def)];
		bool        buffer = false;
		if (!smem_load(inst, buffer) || inst.src1.kind != Decoder::OperandKind::Null ||
		    reg < inst.dst.reg || reg + count > inst.dst.reg + inst.data_dwords) {
			resolve_why = fmt::format("s{} defined at pc 0x{:x} by {}", reg, inst.pc,
			                          magic_enum::enum_name(inst.opcode));
			return false;
		}
		uint32_t base_origin = 0;
		// Each identical reload must see the same chain.
		for (const auto other: also) {
			CallTable copy  = table;
			uint32_t  other_origin = 0;
			CallTable first = table;
			uint32_t  first_origin = 0;
			if (!resolve(inst.src0.reg, buffer ? 4u : 2u, other, depth + 1u, copy, other_origin) ||
			    !resolve(inst.src0.reg, buffer ? 4u : 2u, static_cast<size_t>(def), depth + 1u, first,
			             first_origin) ||
			    other_origin - inst.src0.reg != first_origin - inst.src0.reg ||
			    copy.user_sgpr != first.user_sgpr || copy.steps.size() != first.steps.size() ||
			    !std::equal(copy.steps.begin(), copy.steps.end(), first.steps.begin(),
			                [](const CallTableStep& a, const CallTableStep& b) {
				                return a.buffer == b.buffer && a.base_index == b.base_index &&
				                       a.offset == b.offset && a.dwords == b.dwords;
			                })) {
				resolve_why = fmt::format("reloads of s{} at pc 0x{:x} and 0x{:x} differ", reg,
				                          inst.pc, insts[other].pc);
				return false;
			}
		}
		if (!resolve(inst.src0.reg, buffer ? 4u : 2u, static_cast<size_t>(def), depth + 1u, table,
		             base_origin)) {
			return false;
		}
		table.steps.push_back({buffer, inst.src0.reg - base_origin,
		                       static_cast<int32_t>(inst.offset), inst.data_dwords});
		origin = inst.dst.reg;
		return true;
	};
	std::vector<CallTable> tables;
	for (size_t call = 0; call < insts.size(); call++) {
		if (!IsUnsupportedCall(insts[call])) {
			continue;
		}
		CallTable table;
		table.call_pc     = insts[call].pc;
		table.return_sgpr = (insts[call].raw[0] >> 16u) & 0x7fu;
		const auto target = insts[call].raw[0] & 0xffu;
		const auto fail = [&](std::string_view why) {
			failures += fmt::format("{}pc 0x{:x}: {}", failures.empty() ? "" : "; ", table.call_pc, why);
		};
		if (target >= 104u || (target & 1u) != 0u) {
			fail("odd target pair");
			continue;
		}
		// The target pair: two S_MOV_B32 copies of a consecutive pair.
		uint32_t sources[2] = {};
		size_t   copies[2]  = {};
		bool     ok         = true;
		for (uint32_t half = 0; ok && half < 2; half++) {
			const auto writer = straight_writer(target + half, call);
			ok = writer >= 0 && insts[static_cast<size_t>(writer)].opcode == Decoder::Opcode::S_MOV_B32 &&
			     insts[static_cast<size_t>(writer)].src0.kind == Decoder::OperandKind::Sgpr;
			if (ok) {
				copies[half]  = static_cast<size_t>(writer);
				sources[half] = insts[copies[half]].src0.reg;
			}
		}
		if (!ok || sources[1] != sources[0] + 1u) {
			fail("target pair not two S_MOV_B32 copies of a pair");
			continue;
		}
		// The record load: both halves from one S_BUFFER_LOAD with a register offset.
		const auto load = straight_writer(sources[0], std::min(copies[0], copies[1]));
		bool       buffer = false;
		if (load < 0 || load != straight_writer(sources[1], std::min(copies[0], copies[1]))) {
			fail(fmt::format("copied pair s{} has no single straight-line writer", sources[0]));
			continue;
		}
		const auto& record = insts[static_cast<size_t>(load)];
		if (!smem_load(record, buffer) || !buffer || record.src1.kind == Decoder::OperandKind::Null ||
		    sources[0] < record.dst.reg || sources[1] >= record.dst.reg + record.data_dwords) {
			fail(fmt::format("pair writer at pc 0x{:x} is not a register-offset S_BUFFER_LOAD",
			                 record.pc));
			continue;
		}
		table.target_word   = sources[0] - record.dst.reg;
		table.record_offset = record.offset;
		// The record index shift: S_LSHL_B32 into the offset register right before.
		bool shift_found = false;
		for (size_t i = static_cast<size_t>(load); i-- > 0;) {
			const auto& inst = insts[i];
			if (labels.contains(insts[i + 1].pc) || Decoder::IsDirectBranch(inst.opcode)) {
				break;
			}
			if (inst.dst.kind != record.src1.kind ||
			    (inst.dst.kind == Decoder::OperandKind::Sgpr && inst.dst.reg != record.src1.reg)) {
				continue;
			}
			shift_found = inst.opcode == Decoder::Opcode::S_LSHL_B32 &&
			              inst.src1.kind == Decoder::OperandKind::IntegerInlineConstant;
			table.record_shift = shift_found ? inst.src1.value : 0u;
			break;
		}
		uint32_t origin = 0;
		if (!shift_found) {
			fail("no S_LSHL_B32 record index");
			continue;
		}
		// KYTY_CALL_TRACE_WAVE32: a table V# held in user data itself (no load step; PPSA03671's
		// 0xfc0537b7278db60c keeps it in s[8:11]) is evaluated from the user data as it is.
		if (!resolve(record.src0.reg, 4u, static_cast<size_t>(load), 0u, table, origin) ||
		    (table.steps.empty() && !GetCodegenOptions().call_trace_wave32)) {
			fail(fmt::format("table descriptor s{} not traced ({})", record.src0.reg,
			                 resolve_why.empty() ? std::string("held in user data") : resolve_why));
			continue;
		}
		table.table_index = record.src0.reg - origin;
		tables.push_back(std::move(table));
	}
	return tables;
}

ShaderCallAnalysis AnalyzeShaderCalls(std::span<const uint32_t> code, const CompileOptions& options) {
	ShaderCallAnalysis analysis;
	if (code.empty() || !options.back_code.empty() || options.stage == ShaderType::Local) {
		analysis.failures = "not a single-stage program";
		return analysis;
	}
	Decoder::Program decoded;
	Decoder::DecodeProgram(code, decoded, true);
	std::vector<std::pair<uint32_t, uint32_t>> user_calls;
	uint32_t                                   unresolved = 0;
	(void)FindCallTargets(decoded, options, unresolved, &user_calls);
	std::string table_failures;
	auto        tables = FindCallTables(decoded, options, table_failures);
	// KYTY_RAY_HIT_CALLS_RETURN=1 (default off; usa-ratt/KytyPS5-GT7's RayHitCallsReturn): in a
	// program that traces rays (IMAGE_BVH_INTERSECT_RAY), a call whose target cannot be traced is
	// a hit function (GT7's traversal calls one for a candidate on non-opaque geometry, with v7 = 1
	// "accept" set before it): it returns at once, so every candidate counts as opaque, instead of
	// the whole program being refused and its dispatches skipped (GT7 CS 0x2218a07ab01d4f46).
	static const bool ray_hit_return = [] {
		const char* value = std::getenv("KYTY_RAY_HIT_CALLS_RETURN");
		return value != nullptr && value[0] == '1';
	}();
	const bool traces_rays =
	    ray_hit_return && std::ranges::any_of(decoded.instructions, [](const Decoder::Instruction& inst) {
		    return inst.opcode == Decoder::Opcode::IMAGE_BVH_INTERSECT_RAY;
	    });
	for (const auto& inst: decoded.instructions) {
		if (!IsUnsupportedCall(inst)) {
			continue;
		}
		ShaderCall call;
		call.pc         = inst.pc;
		call.source_reg = inst.raw[0] & 0xffu;
		call.return_reg = (inst.raw[0] >> 16u) & 0x7fu;
		if (const auto user = std::ranges::find(user_calls, inst.pc, &std::pair<uint32_t, uint32_t>::first);
		    user != user_calls.end()) {
			call.user_sgpr = user->second;
		} else if (const auto table = std::ranges::find(tables, inst.pc, &CallTable::call_pc);
		           table != tables.end()) {
			call.table = *table;
		} else if (traces_rays) {
			call.null_callee = true;
		} else {
			analysis.failures += fmt::format("{}pc 0x{:x}: target not traced", analysis.failures.empty() ? "" : "; ",
			                                 inst.pc);
		}
		analysis.calls.push_back(std::move(call));
	}
	if (!analysis.failures.empty() && !table_failures.empty()) {
		analysis.failures += fmt::format(" (table trace: {})", table_failures);
	}
	return analysis;
}

// KYTY_DIAG_CODE_PATCH=<hash>:<byte offset>=<word>[,<byte offset>=<word>...][;<hash>:...] (diagnostic,
// all hex): the listed program's guest code words are replaced before decoding (an instruction turned
// into S_NOPs, a constant changed) to bisect a translation in game. Pair it with KYTY_SHADER_DUMP for
// that hash so the persistent program cache does not serve the unpatched translation.
static bool DiagCodePatch(uint64_t hash, std::span<const uint32_t> code, std::vector<uint32_t>& patched) {
	static const auto patches = [] {
		std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> parsed;
		const char* value = std::getenv("KYTY_DIAG_CODE_PATCH");
		if (value == nullptr) return parsed;
		std::string_view text(value);
		while (!text.empty()) {
			const auto end   = text.find(';');
			auto       group = text.substr(0, end);
			text             = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
			const auto colon = group.find(':');
			if (colon == std::string_view::npos) continue;
			const auto program = std::strtoull(std::string(group.substr(0, colon)).c_str(), nullptr, 16);
			group              = group.substr(colon + 1);
			while (!group.empty()) {
				const auto comma = group.find(',');
				const auto item  = std::string(group.substr(0, comma));
				group            = comma == std::string_view::npos ? std::string_view() : group.substr(comma + 1);
				const auto equal = item.find('=');
				if (equal == std::string::npos) continue;
				parsed.emplace_back(program,
				                    static_cast<uint32_t>(std::strtoul(item.substr(0, equal).c_str(), nullptr, 16)),
				                    static_cast<uint32_t>(std::strtoul(item.substr(equal + 1).c_str(), nullptr, 16)));
			}
		}
		return parsed;
	}();
	bool applied = false;
	for (const auto& [program, offset, word]: patches) {
		if (program != hash || offset % 4u != 0u || offset / 4u >= code.size()) continue;
		if (!applied) patched.assign(code.begin(), code.end());
		applied              = true;
		patched[offset / 4u] = word;
		std::fprintf(stderr, "DIAG_CODE_PATCH hash=0x%016llx offset=0x%x word=0x%08x (was 0x%08x)\n",
		             static_cast<unsigned long long>(hash), offset, word, code[offset / 4u]);
	}
	return applied;
}

// KYTY_CALL_LOOP_CAP: the loop headers inside the callee bodies the function expansion bracketed
// (Decoder::ShaderCallLoopBodyBegin / End) draw on the invocation's call-loop budget (the emitter
// counts their rounds and takes their exits once it is spent). An empty header the structurizer
// adds is judged by the block it enters.
static void MarkCallLoopHeaders(const Decoder::Program& decoded, IR::Program& ir) {
	if (GetCodegenOptions().call_loop_cap == 0u) return;
	std::vector<std::pair<uint32_t, uint32_t>> bracketed;
	for (const auto& inst: decoded.instructions) {
		if (inst.opcode != Decoder::Opcode::S_NOP) continue;
		if (inst.raw[0] == Decoder::ShaderCallLoopBodyBegin) {
			bracketed.emplace_back(inst.pc, UINT32_MAX);
		} else if (inst.raw[0] == Decoder::ShaderCallLoopBodyEnd && !bracketed.empty() &&
		           bracketed.back().second == UINT32_MAX) {
			bracketed.back().second = inst.pc;
		}
	}
	if (bracketed.empty()) return;
	const auto find = [&](uint32_t id) -> const IR::BlockInfo* {
		const auto found = std::ranges::find_if(ir.block_info, [&](const IR::BlockInfo& info) { return info.id == id; });
		return found != ir.block_info.end() ? &*found : nullptr;
	};
	for (auto& info: ir.block_info) {
		if (!info.terminator.loop_header) continue;
		const auto* first = &info;
		for (uint32_t step = 0; first != nullptr && first->start_pc == first->end_pc &&
		                        first->terminator.kind == CFG::TerminatorKind::Branch && step < 8;
		     step++) {
			first = find(first->terminator.true_block);
		}
		if (first == nullptr || first->start_pc == first->end_pc) continue;
		info.terminator.call_loop = std::ranges::any_of(bracketed, [&](const auto& range) {
			return first->start_pc >= range.first && first->start_pc < range.second;
		});
	}
}

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	if (code.empty()) {
		EXIT("shader recompiler input is empty\n");
	}
	std::vector<uint32_t> diag_patched;
	if (DiagCodePatch(options.shader_hash, code, diag_patched)) {
		code = diag_patched;
	}

	const auto compile_begin = std::chrono::steady_clock::now();
	const auto phase_ms      = [&compile_begin]() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                 std::chrono::steady_clock::now() - compile_begin)
		                                 .count());
	};

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " code_words=%" PRIu64 " decode\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(code.size()));

	// IMAGE_BVH*_INTERSECT_RAY is translated only when a BVH mode is on (KYTY_RT_SOFTWARE or
	// KYTY_RT_STUB).
	const bool decode_bvh = GetCodegenOptions().rt_software || GetCodegenOptions().rt_stub;
	Decoder::Program decoded;
	std::vector<uint32_t> joined_code;
	if (!options.back_code.empty()) {
		decoded = DecodeFusedProgram(code, options.back_code, joined_code, decode_bvh);
	} else if (options.stage == ShaderType::Local) {
		decoded = Decoder::DecodeFrontProgram(code);
		// The separately compiled hull half runs in the next Vulkan stage.
		auto& handoff     = decoded.instructions.back();
		handoff.opcode    = Decoder::Opcode::S_ENDPGM;
		handoff.src_count = 0;
	} else {
		Decoder::DecodeProgram(code, decoded, decode_bvh);
		// KYTY_RT_HW: sce::Psr traversal loops trace with host ray queries (RayTracingPatch.h).
		if (options.stage == ShaderType::Compute && decode_bvh && GetCodegenOptions().rt_hw) {
			Decoder::RayTracingPatch patch;
			const bool found   = Decoder::DeriveRayTracingPatch(decoded, patch);
			const bool applied = found && Decoder::ApplyRayTracingPatch(patch, decoded, GetCodegenOptions().rt_hw_verify);
			if (found || patch.unmatched != 0) {
				LOGF("%s native ray tracing patch: hash=0x%016" PRIx64 " sites=%" PRIu64 " unmatched=%u applied=%d%s\n",
				     GetDumpLabel(options), options.shader_hash, static_cast<uint64_t>(patch.sites.size()),
				     patch.unmatched, applied ? 1 : 0, patch.report.c_str());
			}
		}
	}
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " decode instructions=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(decoded.instructions.size()), phase_ms());
	NoteBvhInstructions(options, decoded, decode_bvh);

	// Without a BVH mode, compute dispatches of ray-tracing shaders are skipped (games may compile
	// them before the player can select a mode without ray tracing); other stages stop at the
	// unsupported instruction when the CFG is built.
	if (options.stage == ShaderType::Compute && decoded.has_bvh && !decode_bvh) {
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			const auto& bvh = decoded.instructions.back();
			Log::WriteToConsoleAndLog(fmt::format(
			    "Warning: ray tracing is not implemented; skipping compute dispatches containing "
			    "BVH intersection instructions (shader=0x{:016x}, pc=0x{:08x}, opcode=0x{:02x}). "
			    "KYTY_RT_STUB=1 runs them with every ray missing.\n",
			    options.shader_hash, bvh.pc, bvh.opcode_id));
		}
		TranslateResult skipped;
		skipped.skip_dispatch = true;
		skipped.skip_pc       = decoded.instructions.back().pc;
		skipped.skip_reason   = "BVH intersection without a BVH mode";
		if (options.dump_ir) {
			skipped.decoded_dump = Decoder::ProgramToString(decoded);
		}
		return skipped;
	}

	// KYTY_CALL_PARTITIONS: a partition program (ShaderCalls.h) drops lanes at its markers.
	struct PartitionScope {
		~PartitionScope() { IR::SetPartitionDropLanes(false); }
	} partition_scope;
	IR::SetPartitionDropLanes(std::ranges::any_of(decoded.instructions, [](const auto& inst) {
		return inst.opcode == Decoder::Opcode::S_KYTY_CALL_STATE &&
		       inst.offset == static_cast<uint32_t>(Decoder::CallStateOp::PartitionDrop);
	}));

	// Call-split programs (CallSplit.h) start with their header marker.
	std::optional<CallSplit::Translation> split;
	struct SplitScope {
		~SplitScope() { CallSplit::SetCurrent(nullptr); }
	} split_scope;
	if (!decoded.instructions.empty() &&
	    decoded.instructions.front().opcode == Decoder::Opcode::S_KYTY_CALL_STATE &&
	    decoded.instructions.front().offset == static_cast<uint32_t>(Decoder::CallStateOp::Header)) {
		const auto& first = decoded.instructions.front();
		split.emplace();
		const bool parsed = CallSplit::DecodeHeader(
		    std::span(code).subspan(first.secondary_offset, first.word_count - 1u), split->header);
		// CallSplitPlan BuildCode: header, then for a resume program (entry site k) one S_BRANCH, then
		// the canonical guest code. Callee programs carry no S_GETPC_B64 (ExpandShaderCalls turns
		// each into moves of its guest value), so the offset only matters for the others.
		split->guest_pc_offset =
		    (first.word_count + (split->header.entry_site != 0u ? 1u : 0u)) * static_cast<uint32_t>(sizeof(uint32_t));
		for (const auto& inst: decoded.instructions) {
			if (inst.opcode != Decoder::Opcode::S_KYTY_CALL_STATE) continue;
			const auto op = static_cast<Decoder::CallStateOp>(inst.offset);
			if ((op == Decoder::CallStateOp::Save || op == Decoder::CallStateOp::Probe) &&
			    inst.word_count == 2u) {
				split->site_at_pc[inst.pc] = code[inst.secondary_offset];
			}
		}
		const bool planning =
		    (split->header.flags & (CallSplit::FlagLiveness | CallSplit::FlagProbe)) != 0;
		if ((split->header.flags & CallSplit::FlagProbe) == 0) {
			split->recipe = CallSplit::FindRecipe(split->header.digest);
		}
		if (!parsed || options.stage != ShaderType::Compute || options.wave_size != 64u ||
		    (!planning && split->recipe == nullptr)) {
			NoteSkippedProgram(options, 0, parsed ? "is a call-split program without its plan"
			                                      : "has a damaged call-split header");
			TranslateResult skipped;
			skipped.skip_dispatch = true;
			return skipped;
		}
		CallSplit::SetCurrent(&*split);
	}

	std::string decoded_dump;
	if (options.dump_ir) {
		decoded_dump = Decoder::ProgramToString(decoded);
		if (options.early_dump) {
			LOGF("%s decoded RDNA2 (early):\n%s", GetDumpLabel(options), decoded_dump.c_str());
		}
	}
	// KYTY_SRT_VARIANT_READS: a program that calls a function through S_SWAPPC_B64 is skipped with
	// one log line instead of exiting when the CFG is built.
	if (GetCodegenOptions().srt_variant_reads) {
		const auto call = std::ranges::find_if(decoded.instructions, IsUnsupportedCall);
		if (call != decoded.instructions.end()) {
			TranslateResult skipped;
			uint32_t        unresolved = 0;
			skipped.call_targets       = FindCallTargets(decoded, options, unresolved);
			std::string table_failures;
			skipped.call_tables = FindCallTables(decoded, options, table_failures);
			NoteSkippedProgram(
			    options, call->pc,
			    fmt::format("calls a function through S_SWAPPC_B64, which is not supported "
			                "(callees from user data: {}; calls through a traced table: {}{}{})",
			                DescribeCallTargets(skipped.call_targets, unresolved),
			                skipped.call_tables.size(), table_failures.empty() ? "" : "; untraced: ",
			                table_failures));
			skipped.skip_dispatch = true;
			skipped.skip_pc       = call->pc;
			skipped.skip_reason   = "S_SWAPPC_B64 call";
			skipped.decoded_dump  = std::move(decoded_dump);
			return skipped;
		}
	}

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph\n", GetDumpLabel(options),
	     StageName(options.stage), options.shader_hash);
	auto cfg = CFG::BuildGraph(decoded);
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG BuildGraph blocks=%" PRIu64
	     " loops=%" PRIu64 " back_edges=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(cfg.blocks.size()), static_cast<uint64_t>(cfg.natural_loops.size()),
	     static_cast<uint64_t>(cfg.back_edges.size()), phase_ms());
	if (cfg.irreducible) {
		LogDispatcherFallback(options, cfg, "build");
	} else {
		LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " CFG Structurize\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
		if (!CFG::Structurize(cfg)) {
			LogDispatcherFallback(options, cfg, "structurize");
		} else {
			LOGF("%s structured CFG success: blocks=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(cfg.blocks.size()));
		}
		LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " CFG Structurize blocks=%" PRIu64
		     " loops=%" PRIu64 " elapsed_ms=%" PRIu64 "\n",
		     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
		     static_cast<uint64_t>(cfg.blocks.size()),
		     static_cast<uint64_t>(cfg.natural_loops.size()), phase_ms());
	}

	Frontend::EmbeddedFetchPlan embedded_fetch;
	if ((options.stage == ShaderType::Vertex || options.stage == ShaderType::Local) &&
	    options.input_info.vertex != nullptr && options.input_info.vertex->fetch_embedded) {
		embedded_fetch = DetectEmbeddedVertexFetch(
		    decoded, options.input_info.vertex, options.user_data_base,
		    static_cast<uint32_t>(options.user_data.size()), options.wave_size);
		if (!embedded_fetch.loads.empty()) {
			LOGF("%s embedded vertex fetch plan: detected=%" PRIu64 "\n", GetDumpLabel(options),
			     static_cast<uint64_t>(embedded_fetch.loads.size()));
		}
	}
	Frontend::TranslateOptions translate_options {
	    .stage            = options.stage,
	    .wave_size        = options.wave_size,
	    .shader_hash      = options.shader_hash,
	    .user_data_base   = options.user_data_base,
	    .user_data_count  = static_cast<uint32_t>(options.user_data.size()),
	    .input_info       = options.input_info,
	    .embedded_fetch   = embedded_fetch.loads.empty() ? nullptr : &embedded_fetch,
	};
	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash);
	auto ir = Frontend::TranslateProgram(decoded, cfg, translate_options);
	MarkCallLoopHeaders(decoded, ir);
	// The TLAS of the native ray tracing patch (NativeRayTracing::Prepare reads it).
	if (decoded.ray_tracing_source.valid && ir.info.ray_query_sites != 0) {
		ir.info.ray_query_user_sgpr     = decoded.ray_tracing_source.user_sgpr;
		ir.info.ray_query_vsharp_offset = decoded.ray_tracing_source.vsharp_offset;
		ir.info.ray_query_header_offset = decoded.ray_tracing_source.header_offset;
		ir.info.ray_query_root          = decoded.ray_tracing_source.root;
	}
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " IR TranslateProgram blocks=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(options.stage), options.shader_hash,
	     static_cast<uint64_t>(ir.blocks.size()), phase_ms());
	IR::RewriteToSsa(ir.blocks);
	IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
	IR::ResolveControlFlowIdentities(ir);
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);
	// KYTY_FOLD_LANE_MASKS (default off): mask reads that fold can make select conditions and phis
	// identical, so fold until stable; before read-lane elimination, which then sees the folded
	// EXEC of a lane reduction (x || !x is every lane).
	for (int round = 0; round < 4 && IR::FoldLaneMasks(ir) != 0; round++) {
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::ResolveControlFlowIdentities(ir);
		IR::RemoveIdentities(ir.blocks);
		IR::EliminateDeadCode(ir.blocks);
	}
	// KYTY_WAVE64_HIGH_LANES: a wave64 VS/PS on 32-wide host subgroups (no KYTY_WAVE32_CLUSTERS
	// wide stage) has no lanes 32-63 to shuffle from.
	const bool unreachable_high_lanes = [&] {
		if (!GetCodegenOptions().wave64_high_lanes || ir.wave_size != 64u ||
		    ShaderWorkgroupInput(ir.stage, options.input_info) != nullptr) {
			return false;
		}
		const auto wide = Spirv::GetHostWaveClusters();
		return ir.stage == ShaderType::Pixel ? !wide.pixel : !wide.vertex;
	}();
	const auto read_lane_stats = IR::EliminateReadLane(ir, ir.wave_size, unreachable_high_lanes);
	if (read_lane_stats.rewritten_reads != 0) {
		LOGF("%s read-lane elimination: reads=%" PRIu32 "\n", GetDumpLabel(options),
		     read_lane_stats.rewritten_reads);
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::ResolveControlFlowIdentities(ir);
		IR::RemoveIdentities(ir.blocks);
		IR::EliminateDeadCode(ir.blocks);
	}
	if (const auto* reason = LowerTessellationMemory(ir, options, &decoded); reason != nullptr) {
		// KYTY_TESS_INDEXED: skipped like the draws of a program that gives up.
		NoteSkippedProgram(options, 0, reason);
		TranslateResult skipped;
		skipped.skip_dispatch = true;
		skipped.skip_reason   = reason;
		return skipped;
	}
	// KYTY_DUMP_STDOUT=1 with the early dump: the decoded ISA and the IR that resource tracking is
	// about to see go to stdout, so a shader the tracker rejects can still be inspected offline.
	if (options.dump_ir && options.early_dump && std::getenv("KYTY_DUMP_STDOUT") != nullptr) {
		std::fputs(decoded_dump.c_str(), stdout);
		std::fputs(MakeIrDump(CFG::GraphToString(cfg), ir).c_str(), stdout);
		std::fflush(stdout);
	}
	if (split && (split->header.flags & (CallSplit::FlagLiveness | CallSplit::FlagProbe)) != 0) {
		auto report = std::make_shared<CallSplit::Report>();
		CallSplit::CollectReport(*split, ir, *report);
		TranslateResult planned;
		planned.skip_dispatch = true;
		planned.call_split    = std::move(report);
		return planned;
	}
	const bool variant_reads = GetCodegenOptions().srt_variant_reads;
	IR::BuildSrtPlan(ir, variant_reads);
	IR::EliminateDeadCode(ir.blocks);
	if (const auto unresolved_pc =
	        IR::TrackResources(ir, variant_reads, GetCodegenOptions().bda_writes,
	                           ShaderWorkgroupInput(options.stage, options.input_info));
	    unresolved_pc != UINT32_MAX) {
		NoteSkippedProgram(options, unresolved_pc,
		                   "computes a descriptor at runtime that has no BDA path");
		TranslateResult skipped;
		skipped.skip_dispatch = true;
		skipped.skip_pc       = unresolved_pc;
		skipped.skip_reason   = "descriptor computed at runtime without a BDA path";
		return skipped;
	}
	// Resource tracking can take a plan-table value's last IR use: a stand-in or bindless sampler
	// plan rewires the S# handle, so the scalar reads of the sampler heap are stranded, and the
	// heap's buffer source (collected in the walk) names V# words nothing else uses. Erasing them
	// left the plan naming freed instructions, which ExtractResourcePlan then read (and the
	// draw-prep thread crashed once another compile had reused that memory), so the elimination
	// keeps every value the plan tables name. The walk's list of dynamic reads is no plan value:
	// entries for stranded reads are cleared instead.
	IR::EliminateDeadCode(ir.blocks, IR::PlanTableInstructions(ir));
	(void)IR::DropDeadPlanReferences(ir);
	// A subgroup larger than a host mesh workgroup runs in passes, cut at its barriers.
	if (options.stage == ShaderType::Mesh && options.input_info.vertex != nullptr &&
	    options.input_info.vertex->mesh.passes > 1u) {
		if (const auto* reason = Spirv::MeshPassesUnsupported(ir); reason != nullptr) {
			// Skipped like the draws of a program that gives up, logged once per shader.
			NoteSkippedProgram(options, 0, reason);
			TranslateResult skipped;
			skipped.skip_dispatch = true;
			skipped.skip_reason   = "mesh passes cannot cut the program at its barriers";
			return skipped;
		}
	}
	TranslateResult result;
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = CFG::GraphToString(cfg);
	}
	return result;
}

// A shader with BVH instructions: its module size on the log (one line per shader) and, with
// KYTY_RT_DUMP_DIR=<dir>, the module itself (rt_<stage>_<hash>.spv) for spirv-val and spirv-opt.
static void NoteBvhKernelModule(const IR::Program& ir, const std::vector<uint32_t>& spirv,
                                const ShaderStageInputInfo& input_info) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		std::scoped_lock lock(mutex);
		if (!noted.insert(ir.shader_hash ^ (static_cast<uint64_t>(ir.stage) << 56) ^ spirv.size()).second) {
			return;
		}
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "KYTY_RT_SOFTWARE: {} shader 0x{:016x} SPIR-V module is {} words (KYTY_RT_FUNCTION={}).\n",
	    StageName(ir.stage), ir.shader_hash, spirv.size(), GetCodegenOptions().rt_function));
	const auto* dump_dir = std::getenv("KYTY_RT_DUMP_DIR");
	if (dump_dir == nullptr || dump_dir[0] == '\0') {
		return;
	}
	const auto path = std::string(dump_dir) +
	                  fmt::format("/rt_{}_{:016x}_{}.spv", StageName(ir.stage), ir.shader_hash,
	                              spirv.size());
	if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
		std::fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), file);
		std::fclose(file);
	}
	if (RtFunctionMode() == 0) {
		return;
	}
	// The same module with the node test inlined at each instruction (KYTY_RT_FUNCTION=0), for
	// comparisons: rt_<stage>_<hash>_<words>.inline.spv.
	SetThreadRtFunctionOverride(0);
	auto inlined = Spirv::EmitProgram(ir, input_info);
	SetThreadRtFunctionOverride(-1);
	if (std::FILE* file = std::fopen((path + ".inline.spv").c_str(), "wb"); file != nullptr) {
		std::fwrite(inlined.data(), sizeof(uint32_t), inlined.size(), file);
		std::fclose(file);
	}
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	EXIT_IF(translated.skip_dispatch);
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	IR::ApplyResourceSpecialization(ir, specialization);
	// RemoveIdentities erases identities that the plan tables may still name (with live values).
	(void)IR::ResolvePlanIdentities(ir);
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);
	// What specialization made dead may still be named by the plan tables (a zero-stride buffer
	// read's descriptor): clear those references before anything reads them. Exec-select
	// elimination walks every plan value; it read freed instructions here, which crashed when
	// another compile thread had reused their memory (the parallel precompile replay).
	(void)IR::DropDeadPlanReferences(ir);

	if (GetCodegenOptions().exec_selects) {
		// One lane per host invocation: the emitter branches per invocation (see BranchCondition).
		const bool per_invocation =
		    ir.lane_local || ShaderLanesPerInvocation(ir.stage, ir.wave_size, options.input_info) == 1u;
		(void)IR::EliminateExecSelects(ir, per_invocation);
	}

	// Copy the export at its original control-flow/EXEC site, not an interface placeholder.
	// Iterate only original instructions: aliases are never recursively copied.
	if (ir.stage == ShaderType::Vertex && options.input_info.vertex != nullptr) {
		const auto& vertex = *options.input_info.vertex;
		if (vertex.param_alias_mask != 0) {
			for (auto* block: ir.blocks) {
				for (auto it = block->begin(); it != block->end();) {
					auto& inst = *it++;
					if (inst.GetOpcode() != IR::ValueOpcode::SetAttribute) continue;
					const auto flags = inst.Flags<IR::ExportFlags>();
					const auto source = ir.export_info[flags.index];
					if (source.kind != IR::ExportTargetKind::Parameter) continue;
					for (uint32_t destination = 0; destination < 32u; ++destination) {
						if ((vertex.param_alias_mask & (1u << destination)) == 0 ||
						    vertex.param_alias_source[destination] != source.index) continue;
						auto alias = source;
						alias.index = destination;
						alias.target = 0x20u + destination;
						auto alias_flags = flags;
						alias_flags.index = static_cast<uint32_t>(ir.export_info.size());
						ir.export_info.push_back(alias);
						auto copy = block->PrependNewInst(it, IR::ValueOpcode::SetAttribute,
						                                 {inst.Arg(0), inst.Arg(1)});
						copy->SetFlags(alias_flags);
					}
					// Spare destinations are not sources of any active PS input.
					// Unused guest exports must not overwrite the alias.
					if ((vertex.param_alias_mask & (1u << source.index)) != 0) {
						inst.Invalidate();
					}
				}
			}
		}
	}
	IR::CollectShaderInfo(ir, options.input_info);
	IR::AllocateBindings(ir, push_data_start_dword,
	                     ir.stage == ShaderType::Compute && options.input_info.compute != nullptr &&
	                         options.input_info.compute->lds_storage);
	IR::AnalyzeBufferWriteRanges(ir, options.input_info);
	std::string ir_dump;
	if (options.dump_ir) {
		ir_dump = MakeIrDump(translated.cfg_dump, ir);
		if (options.early_dump) {
			LOGF("%s native IR and bindings (early):\n%s", GetDumpLabel(options), ir_dump.c_str());
		}
	}

	LOGF("%s phase begin: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash);
	auto spirv = Spirv::EmitProgram(ir, options.input_info);
	LOGF("%s phase end: stage=%s hash=0x%016" PRIx64 " SPIR-V EmitProgram words=%" PRIu64
	     " elapsed_ms=%" PRIu64 "\n",
	     GetDumpLabel(options), StageName(ir.stage), ir.shader_hash,
	     static_cast<uint64_t>(spirv.size()),
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - emit_begin)
	                               .count()));
	if (ir.info.uses_bvh) {
		NoteBvhKernelModule(ir, spirv, options.input_info);
	}
	CompileResult result;
	if (options.plain_mip_stats_variant && IR::UsesMipStats(ir)) {
		result.spirv_plain = Spirv::EmitProgram(ir, options.input_info, false);
	}
	result.spirv   = std::move(spirv);
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(translated.decoded_dump);
		result.ir_dump      = std::move(ir_dump);
	}
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler
