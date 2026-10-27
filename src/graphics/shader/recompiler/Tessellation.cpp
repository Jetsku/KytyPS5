#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/skipInvalid.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;
};

bool operator==(const TessellationAddress& lhs, const TessellationAddress& rhs) {
	return lhs.kind == rhs.kind &&
	       (lhs.kind != TessellationAddress::Kind::Affine ||
	        (lhs.coefficient == rhs.coefficient && lhs.constant == rhs.constant));
}

using TessellationRegisters = std::array<TessellationAddress, IR::NumVectorRegs>;

// The register state on entry to a branch target. Unreached targets do not
// constrain a join; a reached one keeps a register only if every edge agrees.
struct TessellationJoin {
	bool                  reached = false;
	TessellationRegisters registers {};
};

bool MergeTessellationJoin(TessellationJoin& join, const TessellationRegisters& registers) {
	if (!join.reached) {
		join = {true, registers};
		return true;
	}
	bool changed = false;
	for (size_t index = 0; index < registers.size(); index++) {
		if (join.registers[index].kind != TessellationAddress::Kind::Unknown &&
		    !(join.registers[index] == registers[index])) {
			join.registers[index] = {};
			changed               = true;
		}
	}
	return changed;
}

uint32_t ReflectTessellationStride(const Decoder::Program& program, bool local,
                                   uint32_t control_points, uint32_t input_stride, uint64_t hash,
                                   std::unordered_map<uint32_t, uint32_t>* constant_stores = nullptr,
                                   std::string* rejection = nullptr) {
	using namespace Decoder;
	using Address = TessellationAddress;
	using Kind    = Address::Kind;
	const char*           stage_name = local ? "LS" : "HS";
	// KYTY_TESS_DUMP=1 logs a failing program's disassembly (local diagnosis only).
	const auto dump = [&] {
		if (const char* env = std::getenv("KYTY_TESS_DUMP"); env != nullptr && env[0] == '1')
			LOGF("%s\n", ProgramToString(program).c_str());
	};
	// GT7 (usa-ratt/KytyPS5-GT7 round 138): a program this analysis cannot model ended the
	// emulator. With KYTY_SKIP_INVALID (default on) the set is rejected and its patch draws are
	// skipped instead; the first failure is kept for the caller's log.
	std::string failure;
	const auto  fail = [&](std::string why) {
		if (!Common::SkipInvalidEnabled()) EXIT("%s\n", why.c_str());
		if (failure.empty()) failure = std::move(why);
	};
	TessellationRegisters entry {};
	entry[local ? 3u : 1u] =
	    local ? Address {Kind::Affine, 1u, 0u} : Address {Kind::PackedControlPoint};
	TessellationRegisters registers {};
	const auto constant = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const auto read     = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		}
		// The native packed HS ID's low byte is the patch ordinal. The logical
		// interface addresses one patch, as in the frontend's v1 initialization.
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return constant(0u);
		return operand.sdwa_sel == 6u ? value : Address {};
	};
	const auto affine = [](uint64_t coefficient, uint64_t offset) {
		return coefficient <= UINT32_MAX && offset <= UINT32_MAX
		           ? Address {Kind::Affine, static_cast<uint32_t>(coefficient),
		                      static_cast<uint32_t>(offset)}
		           : Address {};
	};
	const auto add = [&](Address lhs, Address rhs) {
		return lhs.kind == Kind::Affine && rhs.kind == Kind::Affine
		           ? affine(uint64_t {lhs.coefficient} + rhs.coefficient,
		                    uint64_t {lhs.constant} + rhs.constant)
		           : Address {};
	};
	const auto multiply = [&](Address lhs, Address rhs) {
		if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0u) ||
		    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0u))
			return constant(0u);
		if (lhs.kind != Kind::Affine || rhs.kind != Kind::Affine) return Address {};
		if (rhs.coefficient != 0u) std::swap(lhs, rhs);
		return rhs.coefficient == 0u ? affine(uint64_t {lhs.coefficient} * rhs.constant,
		                                      uint64_t {lhs.constant} * rhs.constant)
		                             : Address {};
	};
	const auto low24 = [&](Address value) {
		if (value.kind == Kind::Affine && value.coefficient == 0u)
			return constant(value.constant & 0xffffffu);
		return value.kind == Kind::Affine &&
		               uint64_t {value.coefficient} * (control_points - 1u) + value.constant <=
		                   0xffffffu
		           ? value
		           : Address {};
	};
	// Forward dataflow over the instruction list. Branch targets are joins whose
	// state merges every incoming edge; back edges are resolved by iterating
	// until no join changes. Memory accesses are checked only on the final,
	// stable pass, where every register reflects all paths reaching it.
	std::unordered_map<uint32_t, TessellationJoin> joins;
	uint32_t                                       stride = 0;
	const auto                                     pass   = [&](bool check) {
		bool changed   = false;
		bool reachable = true;
		registers      = entry;
		stride         = 0;
		for (const auto& inst: program.instructions) {
			if (const auto join = joins.find(inst.pc); join != joins.end()) {
				if (reachable) changed |= MergeTessellationJoin(join->second, registers);
				reachable = join->second.reached;
				if (reachable) registers = join->second.registers;
			}
			if (!reachable) continue;
			if (IsDirectBranch(inst.opcode)) {
				changed |= MergeTessellationJoin(joins[inst.branch_target], registers);
				if (inst.opcode == Opcode::S_BRANCH) reachable = false;
				continue;
			}
			if (inst.opcode == Opcode::S_ENDPGM || inst.opcode == Opcode::S_SETPC_B64) {
				reachable = false;
				continue;
			}
			const bool local_store =
			    inst.opcode == Opcode::DS_WRITE_B32 || inst.opcode == Opcode::DS_WRITE2_B32 ||
			    inst.opcode == Opcode::DS_WRITE_B64 || inst.opcode == Opcode::DS_WRITE2_B64 ||
			    inst.opcode == Opcode::DS_WRITE_B96 || inst.opcode == Opcode::DS_WRITE_B128;
			const bool buffer_store = inst.opcode == Opcode::BUFFER_STORE_DWORD ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX2 ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX3 ||
			                          inst.opcode == Opcode::BUFFER_STORE_DWORDX4;
			const bool control_store = !local && buffer_store &&
			                           inst.src2.kind == OperandKind::Sgpr && inst.src2.reg == 2u;
			const bool control_read =
			    !local && (inst.opcode == Opcode::DS_READ_B32 || inst.opcode == Opcode::DS_READ2_B32 ||
			               inst.opcode == Opcode::DS_READ_B64 || inst.opcode == Opcode::DS_READ2_B64 ||
			               inst.opcode == Opcode::DS_READ_B96 || inst.opcode == Opcode::DS_READ_B128);
			// Any other HS buffer store (the tessellation-factor ring): a constant address is
			// recorded for the lowering, whatever IR spelled it (Ratchet & Clank:
			// (v1 & 0xff) << 4 addresses patch 0's factors).
			if (check && !local && buffer_store && !control_store && constant_stores != nullptr) {
				const auto address = read(inst.src0);
				if (address.kind == Kind::Affine && address.coefficient == 0u)
					(*constant_stores)[inst.pc] = address.constant;
			}
			if (check && ((local && local_store) || control_store || control_read)) {
				const auto address = read(inst.src0);
				if (address.kind != Kind::Affine) {
					dump();
					fail(fmt::format("{} tessellation address is not affine at pc 0x{:08x} (shader "
					                 "0x{:016x})",
					                 stage_name, inst.pc, hash));
					return false;
				}
				if (address.coefficient != 0u) {
					const auto expected = control_read ? input_stride : stride;
					if ((address.coefficient & 3u) != 0u ||
					    (expected != 0u && expected != address.coefficient)) {
						dump();
						fail(fmt::format("{} tessellation stride {} conflicts with {} at pc 0x{:08x} "
						                 "(shader 0x{:016x})",
						                 stage_name, address.coefficient, expected, inst.pc, hash));
						return false;
					}
					if (!control_read) stride = address.coefficient;
				} else if (control_store && constant_stores != nullptr) {
					(*constant_stores)[inst.pc] = address.constant;
				}
			}
			// Store vdata is a source, despite occupying the decoder's dst field.
			if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) continue;
			const auto lhs   = read(inst.src0);
			const auto rhs   = read(inst.src1);
			const auto third = read(inst.src2);
			Address    value;
			switch (inst.opcode) {
				case Opcode::V_BFE_U32:
					if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
					    rhs.coefficient == 0u && third.kind == Kind::Affine &&
					    third.coefficient == 0u) {
						if (rhs.constant == 8u && third.constant == 5u)
							value = {Kind::Affine, 1u, 0u};
						if (rhs.constant == 0u && third.constant == 8u) value = constant(0u);
					}
					break;
				case Opcode::V_AND_B32:
					// The packed HS ID's low byte is the patch ordinal (Ratchet & Clank:
					// v23 = v1 & 0xff addresses the patch block, 0x114 * patch + 0x5c * cp).
					if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0xffu &&
					     rhs.kind == Kind::PackedControlPoint) ||
					    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0xffu &&
					     lhs.kind == Kind::PackedControlPoint))
						value = constant(0u);
					break;
				case Opcode::V_MOV_B32: value = lhs; break;
				case Opcode::V_ADD_NC_U32: value = add(lhs, rhs); break;
				case Opcode::V_ADD3_U32: value = add(add(lhs, rhs), third); break;
				case Opcode::V_MUL_LO_U32: value = multiply(lhs, rhs); break;
				case Opcode::V_ADD_LSHL_U32:
					if (third.kind == Kind::Affine && third.coefficient == 0u && third.constant < 32u)
						value = multiply(add(lhs, rhs), constant(1u << third.constant));
					break;
				case Opcode::V_MUL_U32_U24: value = multiply(low24(lhs), low24(rhs)); break;
				case Opcode::V_MAD_U32_U24:
					value = add(multiply(low24(lhs), low24(rhs)), third);
					break;
				case Opcode::V_LSHL_ADD_U32:
					if (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant < 32u)
						value = add(multiply(lhs, constant(1u << rhs.constant)), third);
					break;
				case Opcode::V_LSHLREV_B32:
					if (lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant < 32u)
						value = multiply(rhs, constant(1u << lhs.constant));
					break;
				case Opcode::V_SUB_NC_U32:
					if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
					    lhs.coefficient >= rhs.coefficient && lhs.constant >= rhs.constant)
						value =
						    affine(lhs.coefficient - rhs.coefficient, lhs.constant - rhs.constant);
					break;
				default: break;
			}
			if (inst.dst.sdwa_sel != 6u || inst.dst.op_sel || inst.dst.omod || inst.dst.clamp ||
			    inst.dst.dpp)
				value = {};
			for (uint32_t index = 0; index < std::max(inst.data_dwords, 1u) &&
			                         inst.dst.reg + index < registers.size();
			     index++) {
				registers[inst.dst.reg + index] = {};
			}
			registers.at(inst.dst.reg) = value;
		}
		return changed;
	};
	// A join register only moves from unreached to known to Unknown, so this ends.
	while (pass(false)) {}
	pass(true);
	if (failure.empty() && stride == 0u) {
		dump();
		fail(fmt::format("{} tessellation stride was not found (shader 0x{:016x})", stage_name, hash));
	}
	if (!failure.empty()) {
		if (rejection != nullptr) *rejection = failure;
		return 0u;
	}
	return stride;
}

const IR::Inst* TessellationBufferBase(const IR::Inst& inst) {
	if (IR::BufferAccessOf(inst.GetOpcode()) == IR::BufferAccess::None || inst.NumArgs() <= 3u)
		return nullptr;
	const auto* base = inst.Arg(3).Resolve().TryInstruction();
	return base != nullptr && base->GetOpcode() == IR::ValueOpcode::TessellationBase ? base
	                                                                                 : nullptr;
}

IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		return source->Arg(0).Resolve() == predicate.Resolve()
		           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
		           : value;
	}
	// Integer address arithmetic. Miles Morales' HS computes its patch-constant address as
	// 0x8000 - 96 * bfe(v6 = 1 + patch, 0, 24) with v6 written under the CP0 predicate.
	switch (source->GetOpcode()) {
		case ValueOpcode::IAdd32:
		case ValueOpcode::ISub32:
		case ValueOpcode::IMul32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitFieldUExtract: break;
		default: return value;
	}
	const auto      args = source->NumArgs();
	std::array<Value, 3> active {};
	bool            changed = false;
	for (size_t index = 0; index < args; index++) {
		active[index] = ActiveAddress(block, before, source->Arg(index), predicate, resolved);
		changed |= active[index] != source->Arg(index).Resolve();
	}
	if (changed) {
		auto copy = args == 3u ? block.PrependNewInst(before, source->GetOpcode(),
		                                              {active[0], active[1], active[2]},
		                                              source->Flags<uint64_t>())
		                       : block.PrependNewInst(before, source->GetOpcode(),
		                                              {active[0], active[1]},
		                                              source->Flags<uint64_t>());
		value     = Value(&*copy);
	}
	resolved.emplace(source, value);
	return value;
}

} // namespace

bool AnalyzeTessellationPrograms(std::span<const uint32_t> local, std::span<const uint32_t> control,
                                 ShaderTessellationInputInfo& info, uint64_t local_hash,
                                 uint64_t control_hash, std::span<const uint32_t> evaluation,
                                 std::string* rejection) {
	// Every tessellated draw prepares its programs; the interface depends only on
	// the two programs and the control point counts.
	struct Key {
		uint64_t local_hash, control_hash;
		uint64_t sizes;
		uint32_t control_points;
		bool     operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const {
			return std::hash<uint64_t> {}(key.local_hash ^ (key.control_hash * 0x9e3779b97f4a7c15ull) ^
			                              key.sizes ^ key.control_points);
		}
	};
	static std::mutex                                                      mutex;
	static std::unordered_map<Key, std::pair<uint32_t, uint32_t>, KeyHash> strides;
	const bool cacheable = local_hash != 0u && control_hash != 0u;
	const Key  key {local_hash, control_hash, (uint64_t {local.size()} << 32u) | control.size(),
	                (info.input_control_points << 8u) | info.output_control_points};
	if (cacheable) {
		std::lock_guard lock(mutex);
		if (const auto found = strides.find(key); found != strides.end()) {
			info.ls_stride = found->second.first;
			info.hs_stride = found->second.second;
			// A rejected set is cached as zero strides (its reason was logged once).
			if (info.ls_stride == 0u || info.hs_stride == 0u) {
				if (rejection != nullptr) *rejection = "rejected before";
				return false;
			}
			return true;
		}
	}
	const auto       local_program = Decoder::DecodeFrontProgram(local);
	Decoder::Program control_program;
	Decoder::DecodeProgram(control, control_program);
	std::string why;
	info.ls_stride = ReflectTessellationStride(local_program, true, info.input_control_points, 0u,
	                                           local_hash, nullptr, &why);
	info.hs_stride = info.ls_stride == 0u
	                     ? 0u
	                     : ReflectTessellationStride(control_program, false,
	                                                 info.output_control_points, info.ls_stride,
	                                                 control_hash, nullptr, &why);
	if (info.ls_stride == 0u || info.hs_stride == 0u) {
		LOGF("Tessellation: LS 0x%016" PRIx64 " / HS 0x%016" PRIx64
		     " not supported, its patch draws are skipped: %s\n",
		     local_hash, control_hash, why.c_str());
		std::printf("Warning: tessellation set LS 0x%016" PRIx64 " / HS 0x%016" PRIx64
		            " not supported (%s); its patch draws are skipped (KYTY_SKIP_INVALID)\n",
		            local_hash, control_hash, why.c_str());
		if (cacheable) {
			std::lock_guard lock(mutex);
			strides.emplace(key, std::pair {0u, 0u});
		}
		if (rejection != nullptr) *rejection = why;
		return false;
	}
	LOGF("Tessellation interface: input_cp=%u output_cp=%u ls_stride=%u hs_stride=%u domain=%u "
	     "partitioning=%u topology=%u levels=%g..%g (LS 0x%016" PRIx64 ", HS 0x%016" PRIx64 ")\n",
	     info.input_control_points, info.output_control_points, info.ls_stride, info.hs_stride,
	     info.domain, info.partitioning, info.output_topology,
	     static_cast<double>(std::bit_cast<float>(info.min_tess_level)),
	     static_cast<double>(std::bit_cast<float>(info.max_tess_level)), local_hash, control_hash);
	if (const char* env = std::getenv("KYTY_TESS_DUMP"); env != nullptr && env[0] == '2') {
		// KYTY_TESS_DUMP=2: every new LS/HS/TES set (local diagnosis only).
		Decoder::Program evaluation_program;
		if (!evaluation.empty()) Decoder::DecodeProgram(evaluation, evaluation_program);
		LOGF("Tessellation LS:\n%s\nTessellation HS:\n%s\nTessellation TES:\n%s\n",
		     ProgramToString(local_program).c_str(), ProgramToString(control_program).c_str(),
		     evaluation.empty() ? "" : ProgramToString(evaluation_program).c_str());
	}
	if (cacheable) {
		std::lock_guard lock(mutex);
		strides.emplace(key, std::pair {info.ls_stride, info.hs_stride});
	}
	return true;
}

void LowerTessellationMemory(IR::Program& program, const CompileOptions& options,
                             const Decoder::Program* decoded) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return;
	}
	const auto& tess = options.input_info.vertex->tess;
	// The guest-ISA analysis knows which HS ring stores have a constant address: those at or past
	// the per-vertex area are patch constants, however the IR spells their address.
	std::unordered_map<uint32_t, uint32_t> constant_stores;
	if (options.stage == ShaderType::TessellationControl && decoded != nullptr) {
		(void)ReflectTessellationStride(*decoded, false, tess.output_control_points, tess.ls_stride,
		                                options.shader_hash, &constant_stores);
	}
	uint32_t forced_patch = 0;
	// Ring addresses can reuse data VGPRs. Their inactive values are irrelevant to
	// a store guarded by the same EXEC predicate, but must remain intact elsewhere.
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			if (TessellationBufferBase(*it) == nullptr) continue;
			std::unordered_map<Inst*, Value> resolved;
			it->SetArg(
			    2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u), resolved));
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	uint32_t reads = 0, writes = 0, factors = 0;
	for (auto* block: program.blocks) {
		for (auto it = block->begin(); it != block->end(); ++it) {
			auto&      inst   = *it;
			const auto shared = SharedAccessOf(inst.GetOpcode());
			const auto buffer = BufferAccessOf(inst.GetOpcode());
			if (shared == SharedAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto&           memory = program.memory_info.at(inst.Flags<MemoryFlags>().index);
			bool                  write  = false;
			uint32_t              components = 0;
			TessellationAttribute kind;
			Value                 address, predicate;
			if (shared != SharedAccess::None && memory.kind == ResourceKind::Lds) {
				write = shared == SharedAccess::Write;
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u ||
				                     (options.stage == ShaderType::Local
				                          ? !write
				                          : options.stage != ShaderType::TessellationControl ||
				                                shared != SharedAccess::Read));
				kind       = write ? TessellationAttribute::LocalOutput
				                   : TessellationAttribute::ControlInput;
				components = SharedComponentCount(inst.GetOpcode());
				address    = inst.Arg(0);
				predicate  = inst.Arg(inst.NumArgs() - 1u);
			} else if (buffer != BufferAccess::None && memory.kind == ResourceKind::Buffer) {
				const auto* base = TessellationBufferBase(inst);
				if (base == nullptr) {
					continue;
				}
				EXIT_NOT_IMPLEMENTED(memory.data_bits != 32u || memory.formatted || memory.idxen ||
				                     !memory.offen || buffer == BufferAccess::Atomic);
				write      = buffer == BufferAccess::Write;
				components = BufferComponentCount(inst.GetOpcode());
				address    = inst.Arg(2).Resolve();
				predicate  = inst.Arg(inst.NumArgs() - 1u);
				if (base->Arg(0).U32() == 1u) {
					EXIT_NOT_IMPLEMENTED(!write ||
					                     options.stage != ShaderType::TessellationControl);
					kind = TessellationAttribute::Factor;
					factors += components;
					if (!address.IsImmediate()) {
						if (const auto found = constant_stores.find(inst.Flags<MemoryFlags>().pc);
						    found != constant_stores.end())
							address = Value(found->second);
					}
				} else {
					kind = write ? TessellationAttribute::ControlOutput
					             : TessellationAttribute::EvaluationInput;
					EXIT_NOT_IMPLEMENTED(write
					                         ? options.stage != ShaderType::TessellationControl
					                         : options.stage != ShaderType::TessellationEvaluation);
					if (write && !address.IsImmediate()) {
						if (const auto found = constant_stores.find(inst.Flags<MemoryFlags>().pc);
						    found != constant_stores.end() &&
						    found->second >= tess.hs_stride * tess.output_control_points) {
							address = Value(found->second);
							forced_patch++;
						}
					}
					if (address.IsImmediate() &&
					    address.U32() >= tess.hs_stride * tess.output_control_points) {
						// A TES reads the patch constants the HS stored (PatchOutput, as an input).
						EXIT_NOT_IMPLEMENTED(!write && options.stage != ShaderType::TessellationEvaluation);
						kind = TessellationAttribute::PatchOutput;
					}
				}
			} else {
				continue;
			}
			const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
				return Value(&*block->PrependNewInst(it, opcode, args));
			};
			std::array<Value, 4> values;
			for (uint32_t component = 0; component < components; component++) {
				const auto offset = memory.offset + 4u * component;
				const auto byte_address =
				    offset == 0u ? address : emit(ValueOpcode::IAdd32, {address, Value(offset)});
				if (write) {
					Value data = shared != SharedAccess::None ? inst.Arg(component + 1u)
					             : components == 1u
					                 ? inst.Arg(4)
					                 : emit(components == 2u   ? ValueOpcode::CompositeExtractU32x2
					                        : components == 3u ? ValueOpcode::CompositeExtractU32x3
					                                           : ValueOpcode::CompositeExtractU32x4,
					                        {inst.Arg(4), Value(component)});
					emit(ValueOpcode::SetTessellationAttribute,
					     {Value(static_cast<uint32_t>(kind)), byte_address, data, predicate});
					writes++;
				} else {
					values[component] =
					    emit(ValueOpcode::GetTessellationAttribute,
					         {Value(static_cast<uint32_t>(kind)), byte_address, predicate});
					reads++;
				}
			}
			if (!write) {
				Value replacement = values[0];
				if (components == 2u)
					replacement =
					    emit(ValueOpcode::CompositeConstructU32x2, {values[0], values[1]});
				if (components == 3u)
					replacement = emit(ValueOpcode::CompositeConstructU32x3,
					                   {values[0], values[1], values[2]});
				if (components == 4u)
					replacement = emit(ValueOpcode::CompositeConstructU32x4,
					                   {values[0], values[1], values[2], values[3]});
				inst.ReplaceUsesWith(replacement);
			}
			inst.Invalidate();
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	RemoveIdentities(program.blocks);
	EliminateDeadCode(program.blocks);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u isa_patch_addresses=%u (0x%016" PRIx64
	     ")\n",
	     options.stage == ShaderType::Local                 ? "LS"
	     : options.stage == ShaderType::TessellationControl ? "HS"
	                                                       : "TES",
	     reads, writes, factors, forced_patch, options.shader_hash);
}

} // namespace Libs::Graphics::ShaderRecompiler
