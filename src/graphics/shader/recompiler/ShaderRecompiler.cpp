#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
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
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <string_view>
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
						} else if (IsDecodedSgpr(inst.src1) &&
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

// Diagnostics for a program skipped for S_SWAPPC_B64: the functions its calls jump to, as the
// dispatch's user data gives them. Psr's shader-mesh builders copy a callback record from user
// SGPRs into their link registers before each call (s_mov_b32 s14, s8; s_mov_b32 s15, s9;
// ...; s_swappc_b64 s[14:15], s[14:15]) and use those registers for other values in between. So
// each call's target pair is traced backwards through the straight-line code before it, up to the
// last write of each half: an S_MOV_B32 per half or an S_MOV_B64 from user SGPRs that the program
// never writes, or the user SGPRs themselves when nothing before the call writes them. Anything
// else (another write, a branch or branch target on the way, an odd or non-user pair) leaves the
// call `unresolved`: its callee is unknown.
static std::vector<CallTarget> FindCallTargets(const Decoder::Program& decoded,
                                               const CompileOptions& options, uint32_t& unresolved) {
	unresolved             = 0;
	const auto& insts      = decoded.instructions;
	const auto  user_sgpr  = [&](uint32_t reg) {
        return reg >= options.user_data_base &&
               reg - options.user_data_base < options.user_data.size();
	};
	// The SGPRs an instruction writes: its destination's width follows from the opcode (64-bit
	// scalar ops and lane masks write a pair, SMEM loads their dword count); a second destination
	// (carry-out) counts as a pair. Unsupported instructions other than calls write nothing known;
	// calls write their return address to their SDST pair.
	const auto writes_sgpr = [](const Decoder::Instruction& inst, uint32_t reg) {
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
			return name.ends_with("_B64") || name.ends_with("_U64") || name.ends_with("_I64") ||
			               name.ends_with("_F64") || name.starts_with("V_CMP")
			           ? 2u
			           : 1u;
		}();
		const auto covers = [&](const Decoder::Operand& operand, uint32_t count) {
			return operand.kind == Decoder::OperandKind::Sgpr && reg >= operand.reg &&
			       reg < operand.reg + count;
		};
		return covers(inst.dst, width) || covers(inst.dst2, 2u);
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

TranslateResult TranslateProgram(std::span<const uint32_t> code, const CompileOptions& options) {
	if (code.empty()) {
		EXIT("shader recompiler input is empty\n");
	}
	if (options.stage != ShaderType::Compute && options.stage != ShaderType::Vertex &&
	    options.stage != ShaderType::Pixel && options.stage != ShaderType::Mesh &&
	    options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		EXIT("shader recompiler received unsupported stage %u\n",
		     static_cast<unsigned>(options.stage));
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
		if (options.dump_ir) {
			skipped.decoded_dump = Decoder::ProgramToString(decoded);
		}
		return skipped;
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
			NoteSkippedProgram(
			    options, call->pc,
			    fmt::format("calls a function through S_SWAPPC_B64, which is not supported "
			                "(callees from user data: {})",
			                DescribeCallTargets(skipped.call_targets, unresolved)));
			skipped.skip_dispatch = true;
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
	const auto read_lane_stats = IR::EliminateReadLane(ir, ir.wave_size);
	if (read_lane_stats.rewritten_reads != 0) {
		LOGF("%s read-lane elimination: reads=%" PRIu32 "\n", GetDumpLabel(options),
		     read_lane_stats.rewritten_reads);
		IR::ConstantPropagationPass(ir.blocks, ir.wave_size);
		IR::ResolveControlFlowIdentities(ir);
		IR::RemoveIdentities(ir.blocks);
		IR::EliminateDeadCode(ir.blocks);
	}
	LowerTessellationMemory(ir, options);
	// KYTY_DUMP_STDOUT=1 with the early dump: the decoded ISA and the IR that resource tracking is
	// about to see go to stdout, so a shader the tracker rejects can still be inspected offline.
	if (options.dump_ir && options.early_dump && std::getenv("KYTY_DUMP_STDOUT") != nullptr) {
		std::fputs(decoded_dump.c_str(), stdout);
		std::fputs(MakeIrDump(CFG::GraphToString(cfg), ir).c_str(), stdout);
		std::fflush(stdout);
	}
	const bool variant_reads = GetCodegenOptions().srt_variant_reads;
	IR::BuildSrtPlan(ir, variant_reads);
	IR::EliminateDeadCode(ir.blocks);
	if (const auto unresolved_pc =
	        IR::TrackResources(ir, variant_reads, GetCodegenOptions().bda_writes);
	    unresolved_pc != UINT32_MAX) {
		NoteSkippedProgram(options, unresolved_pc,
		                   "computes a descriptor at runtime that has no BDA path");
		TranslateResult skipped;
		skipped.skip_dispatch = true;
		return skipped;
	}
	IR::EliminateDeadCode(ir.blocks);
	TranslateResult result;
	result.program = std::move(ir);
	if (options.dump_ir) {
		result.decoded_dump = std::move(decoded_dump);
		result.cfg_dump     = CFG::GraphToString(cfg);
	}
	return result;
}

CompileResult CompileProgram(TranslateResult translated, const CompileOptions& options,
                             const IR::ResourceSpecialization& specialization,
                             uint32_t push_data_start_dword) {
	EXIT_IF(translated.skip_dispatch);
	const auto emit_begin = std::chrono::steady_clock::now();
	auto& ir = translated.program;
	IR::ApplyResourceSpecialization(ir, specialization);
	IR::RemoveIdentities(ir.blocks);
	IR::EliminateDeadCode(ir.blocks);

	if (GetCodegenOptions().exec_selects) {
		// One lane per host invocation: the emitter branches per invocation (see BranchCondition).
		const bool per_invocation =
		    ShaderLanesPerInvocation(ir.stage, ir.wave_size, options.input_info) == 1u;
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
	IR::AllocateBindings(ir, push_data_start_dword);
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
