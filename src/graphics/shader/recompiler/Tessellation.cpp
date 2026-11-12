#include "graphics/shader/recompiler/Tessellation.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/skipInvalid.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

// An address as coefficient * index + constant. Affine: the index is the LS vertex or HS control
// point of the invocation. Indexed (KYTY_TESS_INDEXED; usa-ratt/KytyPS5-GT7 70341d775): a value
// the program loaded or computed (a control point a stencil table names), only usable where the
// stride is checked, never where it is derived. origin names the definition an Unknown, and so an
// Indexed, value came from: values of one origin share their unknown index (0: none).
struct TessellationAddress {
	enum class Kind { Unknown, Affine, PackedControlPoint, Indexed };
	Kind     kind        = Kind::Unknown;
	uint32_t coefficient = 0;
	uint32_t constant    = 0;
	uint32_t origin      = 0;
};

bool operator==(const TessellationAddress& lhs, const TessellationAddress& rhs) {
	using Kind = TessellationAddress::Kind;
	return lhs.kind == rhs.kind && lhs.origin == rhs.origin &&
	       ((lhs.kind != Kind::Affine && lhs.kind != Kind::Indexed) ||
	        (lhs.coefficient == rhs.coefficient && lhs.constant == rhs.constant));
}

using TessellationRegisters = std::array<TessellationAddress, IR::NumVectorRegs>;
// Scalar registers holding a constant (KYTY_TESS_INDEXED): s0..s105, then VCC as 106:107.
using TessellationScalars = std::array<TessellationAddress, 108>;

// The register state on entry to a branch target. Unreached targets do not
// constrain a join; a reached one keeps a register only if every edge agrees.
struct TessellationJoin {
	bool                  reached = false;
	TessellationRegisters registers {};
	TessellationScalars   scalars {};
};

template <typename Array>
bool MergeTessellationArray(Array& join, const Array& incoming) {
	bool changed = false;
	for (size_t index = 0; index < incoming.size(); index++) {
		if (!(join[index] == TessellationAddress {}) && !(join[index] == incoming[index])) {
			join[index] = {};
			changed     = true;
		}
	}
	return changed;
}

bool MergeTessellationJoin(TessellationJoin& join, const TessellationRegisters& registers,
                           const TessellationScalars& scalars) {
	if (!join.reached) {
		join = {true, registers, scalars};
		return true;
	}
	const bool vector = MergeTessellationArray(join.registers, registers);
	const bool scalar = MergeTessellationArray(join.scalars, scalars);
	return vector || scalar;
}

// The byte ranges an HS stores to its off-chip ring at constant addresses, as [begin, end).
using TessellationConstantRanges = std::vector<std::pair<uint32_t, uint32_t>>;

uint32_t ReflectTessellationStride(const Decoder::Program& program, bool local,
                                   uint32_t control_points, uint32_t input_stride, uint64_t hash,
                                   std::unordered_map<uint32_t, uint32_t>* constant_stores = nullptr,
                                   std::string* rejection = nullptr,
                                   TessellationConstantRanges* constant_ranges = nullptr) {
	using namespace Decoder;
	using Address = TessellationAddress;
	using Kind    = Address::Kind;
	const char*           stage_name = local ? "LS" : "HS";
	const bool            indexed    = GetCodegenOptions().tess_indexed;
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
	TessellationScalars   scalars {};
	// Scalar constants are tracked in every mode (upstream 65cfffe88: SDWA takes no literal, so
	// S_MOVK_I32 stages factors in SGPRs); KYTY_TESS_INDEXED adds the folds that use them further.
	const auto scalar_slot = [&](const Operand& operand) -> std::optional<uint32_t> {
		if (operand.kind == OperandKind::Sgpr && operand.reg < 106u) return operand.reg;
		if (operand.kind == OperandKind::VccLo) return 106u;
		if (operand.kind == OperandKind::VccHi) return 107u;
		return std::nullopt;
	};
	const auto constant    = [](uint32_t value) { return Address {Kind::Affine, 0u, value}; };
	const auto is_constant = [](const Address& value) {
		return value.kind == Kind::Affine && value.coefficient == 0u;
	};
	// The packed HS ID holds the patch ordinal in bits 7:0 and the control point ID in bits
	// 12:8. Addresses vary with the control point ID at patch 0, as the frontend initializes v1,
	// or (KYTY_TESS_SINGLE_CP, upstream 3b3e2475d) with the patch ordinal when one output control
	// point makes each HS lane one patch.
	const bool single_cp = GetCodegenOptions().tess_single_cp;
	const bool single    = single_cp && !local && control_points == 1u;
	const auto patch     = single ? Address {Kind::Affine, 1u, 0u} : constant(0u);
	const auto point     = single ? constant(0u) : Address {Kind::Affine, 1u, 0u};
	const auto read      = [&](const Operand& operand) -> Address {
		if (operand.absolute || operand.negate || operand.sdwa_sext || operand.op_sel ||
		    operand.op_sel_hi || operand.negate_hi || operand.dpp)
			return {};
		Address value;
		if (operand.kind == OperandKind::IntegerInlineConstant ||
		    operand.kind == OperandKind::LiteralConstant) {
			value = constant(operand.value);
		} else if (operand.kind == OperandKind::Vgpr) {
			value = registers.at(operand.reg);
		} else if (const auto slot = scalar_slot(operand); slot && is_constant(scalars[*slot])) {
			value = scalars[*slot];
		}
		if (value.kind == Kind::PackedControlPoint && operand.sdwa_sel == 0u) return patch;
		return operand.sdwa_sel == 6u ? value : Address {};
	};
	const auto affine = [](uint64_t coefficient, uint64_t offset) {
		return coefficient <= UINT32_MAX && offset <= UINT32_MAX
		           ? Address {Kind::Affine, static_cast<uint32_t>(coefficient),
		                      static_cast<uint32_t>(offset)}
		           : Address {};
	};
	// An Indexed result keeps its origin; a zero coefficient makes it a plain constant.
	const auto as_indexed = [](Address value, uint32_t origin) {
		if (value.kind == Kind::Affine && value.coefficient != 0u) {
			value.kind   = Kind::Indexed;
			value.origin = origin;
		}
		return value;
	};
	const auto add = [&](Address lhs, Address rhs) {
		if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine)
			return affine(uint64_t {lhs.coefficient} + rhs.coefficient,
			              uint64_t {lhs.constant} + rhs.constant);
		if (!indexed) return Address {};
		if (lhs.kind == Kind::Indexed && rhs.kind == Kind::Indexed && lhs.origin == rhs.origin)
			return as_indexed(affine(uint64_t {lhs.coefficient} + rhs.coefficient,
			                         uint64_t {lhs.constant} + rhs.constant),
			                  lhs.origin);
		// An indexed value plus a constant stays indexed in the same unknown index.
		if (is_constant(rhs)) std::swap(lhs, rhs);
		if (is_constant(lhs) && rhs.kind == Kind::Indexed)
			return as_indexed(affine(rhs.coefficient, uint64_t {lhs.constant} + rhs.constant),
			                  rhs.origin);
		return Address {};
	};
	const auto multiply = [&](Address lhs, Address rhs) {
		if ((is_constant(lhs) && lhs.constant == 0u) || (is_constant(rhs) && rhs.constant == 0u))
			return constant(0u);
		if (!indexed) {
			if (lhs.kind != Kind::Affine || rhs.kind != Kind::Affine) return Address {};
			if (rhs.coefficient != 0u) std::swap(lhs, rhs);
			return rhs.coefficient == 0u ? affine(uint64_t {lhs.coefficient} * rhs.constant,
			                                      uint64_t {lhs.constant} * rhs.constant)
			                             : Address {};
		}
		if (is_constant(lhs)) std::swap(lhs, rhs);
		if (!is_constant(rhs)) return Address {};
		// A value the program loaded, scaled: an index of unknown origin.
		if (lhs.kind == Kind::Unknown)
			return lhs.origin == 0u ? Address {} : as_indexed(affine(rhs.constant, 0u), lhs.origin);
		if (lhs.kind != Kind::Affine && lhs.kind != Kind::Indexed) return Address {};
		const auto product = affine(uint64_t {lhs.coefficient} * rhs.constant,
		                            uint64_t {lhs.constant} * rhs.constant);
		return lhs.kind == Kind::Indexed ? as_indexed(product, lhs.origin) : product;
	};
	const auto low24 = [&](Address value) {
		if (value.kind == Kind::Affine && value.coefficient == 0u)
			return constant(value.constant & 0xffffffu);
		// A loaded index is a control point number, far below 2^24.
		if (indexed && (value.kind == Kind::Unknown || value.kind == Kind::Indexed)) return value;
		// The largest index: the last control point, or the last patch ordinal of a one-point HS.
		const uint64_t maximum = single ? 0xffu : control_points - 1u;
		return value.kind == Kind::Affine &&
		               uint64_t {value.coefficient} * maximum + value.constant <= 0xffffffu
		           ? value
		           : Address {};
	};
	// KYTY_TESS_INDEXED: a mask of a constant, or of the packed HS ID (the control point in bits
	// 8-12, the patch ordinal in the low byte).
	const auto mask = [&](Address value, Address bits) -> Address {
		if (is_constant(value)) std::swap(value, bits);
		if (!is_constant(bits)) return {};
		if (is_constant(value)) return constant(value.constant & bits.constant);
		if (value.kind == Kind::PackedControlPoint) {
			if ((bits.constant & ~0xffu) == 0u)
				return !single ? constant(0u) : bits.constant == 0xffu ? patch : Address {};
			if (bits.constant == 0x1f00u) return multiply(point, constant(0x100u));
		}
		return {};
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
		scalars        = {};
		stride         = 0;
		for (const auto& inst: program.instructions) {
			if (const auto join = joins.find(inst.pc); join != joins.end()) {
				if (reachable) changed |= MergeTessellationJoin(join->second, registers, scalars);
				reachable = join->second.reached;
				if (reachable) {
					registers = join->second.registers;
					scalars   = join->second.scalars;
				}
			}
			if (!reachable) continue;
			if (IsDirectBranch(inst.opcode)) {
				changed |= MergeTessellationJoin(joins[inst.branch_target], registers, scalars);
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
				// KYTY_TESS_INDEXED: a read may name its control point by a loaded index; a
				// store's stride is derived from its address, so its index must be its own.
				if (address.kind != Kind::Affine &&
				    !(indexed && control_read && address.kind == Kind::Indexed)) {
					dump();
					fail(fmt::format("{} tessellation address is not affine at pc 0x{:08x} (shader "
					                 "0x{:016x})",
					                 stage_name, inst.pc, hash));
					return false;
				}
				// A coefficient below zero (KYTY_TESS_SINGLE_CP: later patches sit lower in the ring)
				// marks a patch constant, as a zero one does (upstream 3b3e2475d).
				if (static_cast<int32_t>(address.coefficient) > 0) {
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
				} else if (control_store) {
					if (constant_stores != nullptr) (*constant_stores)[inst.pc] = address.constant;
					if (constant_ranges != nullptr) {
						const uint64_t begin = uint64_t {address.constant} + inst.offset;
						const uint64_t end   = begin + 4ull * std::max(inst.data_dwords, 1u);
						if (end <= UINT32_MAX)
							constant_ranges->emplace_back(static_cast<uint32_t>(begin),
							                              static_cast<uint32_t>(end));
					}
				}
			}
			// Scalar constants (s_mov / s_movk); any other scalar write forgets what it may cover (a
			// load's dwords, a 64-bit pair, a carry-out).
			if (const auto slot = scalar_slot(inst.dst2); slot) {
				for (uint32_t index = 0; index < 2u && *slot + index < scalars.size(); index++)
					scalars[*slot + index] = {};
			}
			if (const auto slot = scalar_slot(inst.dst); slot) {
				const bool move =
				    inst.opcode == Opcode::S_MOV_B32 || inst.opcode == Opcode::S_MOVK_I32;
				const auto     value   = move ? read(inst.src0) : Address {};
				const uint32_t covered = move ? 1u : std::max(inst.data_dwords, 2u);
				for (uint32_t index = 0; index < covered && *slot + index < scalars.size(); index++)
					scalars[*slot + index] = {};
				if (is_constant(value)) scalars[*slot] = value;
				continue;
			}
			// Store vdata is a source, despite occupying the decoder's dst field.
			if (local_store || buffer_store || inst.dst.kind != OperandKind::Vgpr) continue;
			if (indexed && inst.opcode == Opcode::V_MOVRELD_B32) {
				// Writes v[dst + M0]: any register from dst up.
				for (uint32_t reg = inst.dst.reg; reg < registers.size(); reg++) registers[reg] = {};
				continue;
			}
			const auto lhs   = read(inst.src0);
			const auto rhs   = read(inst.src1);
			const auto third = read(inst.src2);
			Address    value;
			switch (inst.opcode) {
				case Opcode::V_BFE_U32:
					if (lhs.kind == Kind::PackedControlPoint && rhs.kind == Kind::Affine &&
					    rhs.coefficient == 0u && third.kind == Kind::Affine &&
					    third.coefficient == 0u) {
						if (rhs.constant == 8u && third.constant == 5u) value = point;
						if (rhs.constant == 0u && third.constant == 8u) value = patch;
					}
					break;
				case Opcode::V_AND_B32:
					if (indexed) {
						value = mask(lhs, rhs);
						break;
					}
					// The packed HS ID's low byte is the patch ordinal (Ratchet & Clank:
					// v23 = v1 & 0xff addresses the patch block, 0x114 * patch + 0x5c * cp).
					if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0xffu &&
					     rhs.kind == Kind::PackedControlPoint) ||
					    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0xffu &&
					     lhs.kind == Kind::PackedControlPoint))
						value = patch;
					// Bits 8..12 hold the control point (upstream 77458260f): & 0x1f00 is 256 * cp.
					if ((lhs.kind == Kind::Affine && lhs.coefficient == 0u && lhs.constant == 0x1f00u &&
					     rhs.kind == Kind::PackedControlPoint) ||
					    (rhs.kind == Kind::Affine && rhs.coefficient == 0u && rhs.constant == 0x1f00u &&
					     lhs.kind == Kind::PackedControlPoint))
						value = multiply(point, constant(256u));
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
					// KYTY_TESS_SINGLE_CP (upstream 3b3e2475d): the coefficient may fall below zero.
					if (lhs.kind == Kind::Affine && rhs.kind == Kind::Affine &&
					    lhs.constant >= rhs.constant &&
					    (single_cp || lhs.coefficient >= rhs.coefficient))
						value = {Kind::Affine, lhs.coefficient - rhs.coefficient,
						         lhs.constant - rhs.constant};
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
				// KYTY_TESS_INDEXED: each definition is its own unknown index, the same on every
				// pass so that joins converge.
				if (indexed) registers[inst.dst.reg + index].origin = (inst.pc + 4u) * 16u + index;
			}
			if (value.kind == Kind::Unknown) value.origin = registers.at(inst.dst.reg).origin;
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

// KYTY_TESS_MERGED_CARRY (based on Erlou93/KytyPS5-Yotei): the VGPRs the HS part of a split merged
// LS+HS program reads before writing them (ShaderTessellationInputInfo::carried_vgprs). On the
// hardware one wave runs the LS part and then the HS part, so such a register still holds what the
// LS part left in the lane: Ghost of Yotei 2.x's particle HS takes the particle's index, flags and
// position that way instead of reading them back from LDS as the 1.x HS halves do.
// The set may only err on the large side: a read counts every register its operand could cover; a
// write counts only when it certainly replaces the whole register in every lane that later reads
// it, i.e. a full-width write in the straight-line code after the entry sequence (EXEC = all,
// V_CMPX lane < HS threads, S_CBRANCH_EXECZ) and before the first branch or other EXEC change (a
// write under a narrower EXEC keeps the LS value in the other lanes). An extra register only costs
// one carried u32.
std::array<uint32_t, 8> MergedHullCarriedVgprs(const Decoder::Program& control) {
	using namespace Decoder;
	std::bitset<IR::NumVectorRegs> defined, live;
	defined.set(0); // the back half's own V_MBCNT lane index (SplitMergedHullProgram)
	defined.set(1); // the packed HS IDs the frontend sets
	bool entry    = true;
	bool definite = true;
	for (const auto& inst: control.instructions) {
		const auto text   = InstructionToString(inst);
		const auto colon  = text.find(": ");
		const auto start  = colon == std::string::npos ? 0u : colon + 2u;
		const auto name   = text.substr(start, text.find(' ', start) - start);
		const auto has    = [&](const char* part) { return name.find(part) != std::string::npos; };
		const bool memory = inst.family == Family::MUBUF || inst.family == Family::MTBUF ||
		                    inst.family == Family::FLAT || inst.family == Family::DS ||
		                    inst.family == Family::MIMG;
		const bool load = memory && !has("D16") &&
		                  (has("LOAD") || name.starts_with("DS_READ") || has("SAMPLE") || has("GATHER"));
		const bool valu = inst.family == Family::VOP1 || inst.family == Family::VOP2 ||
		                  inst.family == Family::VOP3 || inst.family == Family::VOP3P;
		// Destinations that keep part of the old value (or read it): accumulators, lane writes,
		// sub-dword writes.
		const bool partial = has("MAC") || has("DOT2C") || has("DOT4C") || has("DOT8C") ||
		                     has("WRITELANE") || has("MOVREL") || has("D16") || inst.dst.sdwa_sel != 6u ||
		                     inst.dst.op_sel || inst.dst.dpp;
		const auto read = [&](const Operand& operand, uint32_t width) {
			if (operand.kind != OperandKind::Vgpr) return;
			for (uint32_t index = 0; index < width && operand.reg + index < IR::NumVectorRegs; index++) {
				if (!defined[operand.reg + index]) live.set(operand.reg + index);
			}
		};
		const uint32_t source_width = memory ? std::max(inst.data_dwords, 4u) : has("64") ? 2u : 1u;
		for (const auto* operand: {&inst.src0, &inst.src1, &inst.src2, &inst.src3}) read(*operand, source_width);
		if (inst.family == Family::MIMG) {
			read(inst.src0, std::max(inst.image_address_components, 4u));
			for (uint32_t index = 0; index < inst.image_nsa_dwords && index < MaxImageNsaAddressComponents;
			     index++) {
				const Operand address {.kind = OperandKind::Vgpr, .reg = inst.image_nsa_addr[index]};
				read(address, 1u);
			}
		}
		if ((memory && !load) || ((valu || memory) && partial)) {
			read(inst.dst, std::max(inst.data_dwords, 2u)); // store data, atomics, partial writes
		}
		if (entry) {
			const bool entry_inst =
			    inst.opcode == Opcode::V_MBCNT_LO_U32_B32 || inst.opcode == Opcode::V_MBCNT_HI_U32_B32 ||
			    inst.opcode == Opcode::S_MOV_B64 || inst.opcode == Opcode::V_CMPX_LT_U32 ||
			    inst.opcode == Opcode::S_BARRIER || inst.opcode == Opcode::S_INST_PREFETCH ||
			    inst.opcode == Opcode::S_WAITCNT || inst.opcode == Opcode::S_NOP;
			if (inst.opcode == Opcode::S_CBRANCH_EXECZ || !entry_inst) entry = false;
			if (entry || inst.opcode == Opcode::S_CBRANCH_EXECZ) continue;
		}
		if (IsDirectBranch(inst.opcode) || inst.opcode == Opcode::S_SETPC_B64 ||
		    inst.opcode == Opcode::S_ENDPGM || name.starts_with("S_SWAPPC") ||
		    text.find("exec") != std::string::npos || text.find("EXEC") != std::string::npos) {
			definite = false;
		}
		if (!definite || inst.dst.kind != OperandKind::Vgpr || partial) continue;
		if (valu) {
			defined.set(inst.dst.reg);
		} else if (load &&
		           (inst.family == Family::DS || inst.family == Family::MUBUF || inst.family == Family::MTBUF)) {
			for (uint32_t index = 0;
			     index < std::max(inst.data_dwords, 1u) && inst.dst.reg + index < IR::NumVectorRegs; index++) {
				defined.set(inst.dst.reg + index);
			}
		} else if (load) {
			defined.set(inst.dst.reg);
		}
	}
	live.reset(0);
	live.reset(1);
	std::array<uint32_t, 8> mask {};
	for (uint32_t reg = 0; reg < IR::NumVectorRegs; reg++) {
		if (live[reg]) mask[reg / 32u] |= 1u << (reg % 32u);
	}
	return mask;
}

// KYTY_TESS_MERGED_CARRY: the carried set of a split merged pair; empty when the HS part reads no
// LS-part VGPR or the set cannot be carried (logged).
std::array<uint32_t, 8> CarriedVgprs(const Decoder::Program& control_program,
                                     const ShaderTessellationInputInfo& info, uint64_t control_hash) {
	const auto  mask = MergedHullCarriedVgprs(control_program);
	std::string list;
	uint32_t    count = 0;
	for (uint32_t reg = 0; reg < IR::NumVectorRegs; reg++) {
		if (((mask[reg / 32u] >> (reg % 32u)) & 1u) == 0u) continue;
		list += fmt::format(" v{}", reg);
		count++;
	}
	if (count == 0u) return {};
	// HS thread i of a patch ran the LS part for the patch's vertex i: the same lane only when
	// the counts agree.
	if (info.input_control_points != info.output_control_points) {
		LOGF("Tessellation: merged HS 0x%016" PRIx64 " reads %u LS-part VGPRs but input_cp=%u != "
		     "output_cp=%u, not carried:%s\n",
		     control_hash, count, info.input_control_points, info.output_control_points, list.c_str());
		std::printf("KYTY_TESS_MERGED_CARRY: HS 0x%016" PRIx64 " NOT carried (control points %u/%u):%s\n",
		            control_hash, info.input_control_points, info.output_control_points, list.c_str());
		return {};
	}
	// The carried registers follow the LS outputs in the LS -> HS interface (Vulkan guarantees 64
	// components per vertex there; the GPUs this runs on offer 128).
	const auto components = (info.ls_stride + 15u) / 16u * 4u + (count + 3u) / 4u * 4u;
	if (components > 128u) {
		LOGF("Tessellation: merged HS 0x%016" PRIx64 " carries %u VGPRs: %u interface components "
		     "(max 128), not carried:%s\n",
		     control_hash, count, components, list.c_str());
		std::printf("KYTY_TESS_MERGED_CARRY: HS 0x%016" PRIx64 " NOT carried (%u components):%s\n",
		            control_hash, components, list.c_str());
		return {};
	}
	LOGF("Tessellation interface: the merged HS 0x%016" PRIx64 " carries %u VGPRs from its LS part:%s\n",
	     control_hash, count, list.c_str());
	std::printf("KYTY_TESS_MERGED_CARRY: HS 0x%016" PRIx64 " carries %u VGPRs:%s\n", control_hash, count,
	            list.c_str());
	return mask;
}

// Whether predicate p implies q: equal, or a conjunction one of whose terms does.
bool ImpliesPredicate(IR::Value p, IR::Value q, uint32_t depth = 0) {
	p = p.Resolve();
	q = q.Resolve();
	if (p == q) return true;
	const auto* inst = p.TryInstruction();
	if (inst == nullptr || depth > 16u || inst->GetOpcode() != IR::ValueOpcode::LogicalAnd)
		return false;
	return ImpliesPredicate(inst->Arg(0), q, depth + 1u) ||
	       ImpliesPredicate(inst->Arg(1), q, depth + 1u);
}

// The address an access guarded by predicate computes on its active lanes: a register the guest
// wrote under EXEC holds Select(exec, written, previous). With KYTY_TESS_INDEXED (indexed), any
// EXEC the access's own implies selects the written value (usa-ratt/KytyPS5-GT7 70341d775), and
// the patch ordinal (the low bits of InvocationId << 8, the frontend's v1) folds to 0.
IR::Value ActiveAddress(IR::Block& block, IR::Block::iterator before, IR::Value value,
                        IR::Value predicate, std::unordered_map<IR::Inst*, IR::Value>& resolved,
                        bool indexed = false, uint32_t depth = 0) {
	using namespace IR;
	value        = value.Resolve();
	auto* source = value.TryInstruction();
	if (source == nullptr || (indexed && depth > 64u)) return value;
	if (const auto found = resolved.find(source); found != resolved.end()) return found->second;
	if (source->GetOpcode() == ValueOpcode::SelectU32) {
		if (!indexed) {
			return source->Arg(0).Resolve() == predicate.Resolve()
			           ? ActiveAddress(block, before, source->Arg(1), predicate, resolved)
			           : value;
		}
		const auto result = ImpliesPredicate(predicate, source->Arg(0))
		                        ? ActiveAddress(block, before, source->Arg(1), predicate,
		                                        resolved, indexed, depth + 1u)
		                        : value;
		resolved.emplace(source, result);
		return result;
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
		active[index] =
		    ActiveAddress(block, before, source->Arg(index), predicate, resolved, indexed, depth + 1u);
		changed |= active[index] != source->Arg(index).Resolve();
	}
	if (indexed) {
		// The bits below a shift are zero: the HS's packed ID is InvocationId << 8, so its patch
		// ordinal, the low byte, is 0 in the logical interface.
		const auto shifted_by = [](const Value& operand) -> std::optional<uint32_t> {
			const auto* shift = operand.Resolve().TryInstruction();
			if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
			    !shift->Arg(1).Resolve().IsImmediate())
				return std::nullopt;
			return shift->Arg(1).Resolve().U32();
		};
		if (source->GetOpcode() == ValueOpcode::BitwiseAnd32) {
			for (size_t index = 0; index < 2u; index++) {
				const auto shift = shifted_by(active[index]);
				const auto bits  = active[1u - index].Resolve();
				if (shift && *shift < 32u && bits.IsImmediate() && bits.U32() < (1u << *shift)) {
					resolved.emplace(source, Value(0u));
					return Value(0u);
				}
			}
		}
		if (source->GetOpcode() == ValueOpcode::BitFieldUExtract && args == 3u &&
		    active[1].Resolve().IsImmediate() && active[2].Resolve().IsImmediate()) {
			const auto shift = shifted_by(active[0]);
			if (shift &&
			    uint64_t {active[1].Resolve().U32()} + active[2].Resolve().U32() <= *shift) {
				resolved.emplace(source, Value(0u));
				return Value(0u);
			}
		}
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
                                 std::string* rejection, bool merged) {
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
	// ls_stride, hs_stride, patch_base, patch_size, then carried_vgprs.
	static std::unordered_map<Key, std::array<uint32_t, 12>, KeyHash> strides;
	const bool cacheable = local_hash != 0u && control_hash != 0u;
	const Key  key {local_hash, control_hash, (uint64_t {local.size()} << 32u) | control.size(),
	                (info.input_control_points << 8u) | info.output_control_points};
	if (cacheable) {
		std::lock_guard lock(mutex);
		if (const auto found = strides.find(key); found != strides.end()) {
			info.ls_stride  = found->second[0];
			info.hs_stride  = found->second[1];
			info.patch_base = found->second[2];
			info.patch_size = found->second[3];
			std::copy_n(found->second.begin() + 4, 8, info.carried_vgprs.begin());
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
	// KYTY_TESS_INDEXED: the HS's constant-address ring stores, for the patch-constant range.
	const bool                 indexed = GetCodegenOptions().tess_indexed;
	TessellationConstantRanges constant_ranges;
	info.ls_stride = ReflectTessellationStride(local_program, true, info.input_control_points, 0u,
	                                           local_hash, nullptr, &why);
	info.hs_stride = info.ls_stride == 0u
	                     ? 0u
	                     : ReflectTessellationStride(control_program, false,
	                                                 info.output_control_points, info.ls_stride,
	                                                 control_hash, nullptr, &why,
	                                                 &constant_ranges);
	info.patch_base    = 0;
	info.patch_size    = 0;
	info.carried_vgprs = {};
	// KYTY_TESS_MERGED_CARRY: a carried pair declares the HS's patch-constant range as well. Without it each
	// stage bases its patch interface at the lowest address it touches itself: the Yotei 2.x particle HS
	// stores from 0x7f30 (the factor copies first), its TES reads from 0x7f90, so the TES read the factor
	// copy (1.0f) where it expects the particle's material index (0x7fb0).
	if (merged && info.hs_stride != 0u && info.ls_stride != 0u) {
		info.carried_vgprs = CarriedVgprs(control_program, info, control_hash);
	}
	const bool carried = std::ranges::any_of(info.carried_vgprs, [](uint32_t word) { return word != 0u; });
	if ((indexed || carried) && info.hs_stride != 0u) {
		// The patch constants: what the HS stores at constant addresses past its control points.
		// HS and TES declare this range, so their patch interfaces agree whatever each touches.
		const uint32_t control_end = info.hs_stride * info.output_control_points;
		uint32_t       patch_begin = UINT32_MAX, patch_end = 0;
		for (const auto& [begin, end]: constant_ranges) {
			if (begin >= control_end) {
				patch_begin = std::min(patch_begin, begin);
				patch_end   = std::max(patch_end, end);
			}
		}
		if (patch_begin < patch_end) {
			info.patch_base = patch_begin & ~15u;
			info.patch_size = ((patch_end + 15u) & ~15u) - info.patch_base;
		}
	}
	if (info.ls_stride == 0u || info.hs_stride == 0u) {
		LOGF("Tessellation: LS 0x%016" PRIx64 " / HS 0x%016" PRIx64
		     " not supported, its patch draws are skipped: %s\n",
		     local_hash, control_hash, why.c_str());
		std::printf("Warning: tessellation set LS 0x%016" PRIx64 " / HS 0x%016" PRIx64
		            " not supported (%s); its patch draws are skipped (KYTY_SKIP_INVALID)\n",
		            local_hash, control_hash, why.c_str());
		if (cacheable) {
			std::lock_guard lock(mutex);
			strides.emplace(key, std::array<uint32_t, 12> {});
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
	if (info.patch_size != 0u)
		LOGF("Tessellation patch constants: 0x%x+0x%x (HS 0x%016" PRIx64 ")\n", info.patch_base,
		     info.patch_size, control_hash);
	if (const char* env = std::getenv("KYTY_TESS_DUMP"); env != nullptr && env[0] == '2') {
		// KYTY_TESS_DUMP=2: every new LS/HS/TES set (local diagnosis only).
		Decoder::Program evaluation_program;
		if (!evaluation.empty()) Decoder::DecodeProgram(evaluation, evaluation_program);
		LOGF("Tessellation LS:\n%s\nTessellation HS:\n%s\nTessellation TES:\n%s\n",
		     ProgramToString(local_program).c_str(), ProgramToString(control_program).c_str(),
		     evaluation.empty() ? "" : ProgramToString(evaluation_program).c_str());
	}
	if (cacheable) {
		std::array<uint32_t, 12> entry {info.ls_stride, info.hs_stride, info.patch_base, info.patch_size};
		std::copy_n(info.carried_vgprs.begin(), 8, entry.begin() + 4);
		std::lock_guard lock(mutex);
		strides.emplace(key, entry);
	}
	return true;
}

const char* LowerTessellationMemory(IR::Program& program, const CompileOptions& options,
                                    const Decoder::Program* decoded) {
	using namespace IR;
	if (options.stage != ShaderType::Local && options.stage != ShaderType::TessellationControl &&
	    options.stage != ShaderType::TessellationEvaluation) {
		return nullptr;
	}
	const auto& tess    = options.input_info.vertex->tess;
	const bool  indexed = GetCodegenOptions().tess_indexed;
	const char* stage_name = options.stage == ShaderType::Local                 ? "LS"
	                         : options.stage == ShaderType::TessellationControl ? "HS"
	                                                                            : "TES";
	uint32_t    current_pc = 0;
	const auto  unsupported = [&](const char* what) {
		LOGF("%s tessellation lowering does not model %s (pc 0x%08x, shader 0x%016" PRIx64
		     "): its draws are skipped\n",
		     stage_name, what, current_pc, options.shader_hash);
		return what;
	};
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
			it->SetArg(2, ActiveAddress(*block, it, it->Arg(2), it->Arg(it->NumArgs() - 1u),
			                            resolved, indexed));
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
			current_pc                   = inst.Flags<MemoryFlags>().pc;
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
					// KYTY_TESS_INDEXED: the HS also reads back what its invocations wrote (other
					// control points' outputs and patch constants, behind a barrier inserted below).
					const bool control = options.stage == ShaderType::TessellationControl;
					kind = write || (indexed && control) ? TessellationAttribute::ControlOutput
					                                     : TessellationAttribute::EvaluationInput;
					EXIT_NOT_IMPLEMENTED(write ? !control
					                           : !(indexed && control) &&
					                                 options.stage != ShaderType::TessellationEvaluation);
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
						EXIT_NOT_IMPLEMENTED(!write && !(indexed && control) &&
						                     options.stage != ShaderType::TessellationEvaluation);
						kind = TessellationAttribute::PatchOutput;
						// KYTY_TESS_INDEXED or a carried pair: within the range the analysis found
						// the HS storing, which both stages declare.
						const uint64_t first = uint64_t {address.U32()} + memory.offset;
						if (tess.patch_size != 0u &&
						    (first < tess.patch_base ||
						     first + 4u * BufferComponentCount(inst.GetOpcode()) >
						         uint64_t {tess.patch_base} + tess.patch_size)) {
							return unsupported("a patch-constant access outside the HS's");
						}
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
	// KYTY_TESS_INDEXED: an HS invocation reading what another wrote (the console's wave runs a
	// patch's control points together) needs a barrier after the writes, where every invocation
	// still runs. GT7's subdivision-surface HSs have control point 0 read all four outputs back
	// to compute the patch's factors.
	uint32_t barriers = 0;
	if (indexed && options.stage == ShaderType::TessellationControl) {
		const auto attribute_of = [](const Inst& inst) {
			return static_cast<TessellationAttribute>(inst.Arg(0).U32());
		};
		Block*          last_block = nullptr;
		Block::iterator last_write;
		for (auto* block: program.blocks) {
			for (auto it = block->begin(); it != block->end(); ++it) {
				const auto opcode = it->GetOpcode();
				if (opcode == ValueOpcode::Barrier) {
					last_block = nullptr;
				} else if (opcode == ValueOpcode::SetTessellationAttribute &&
				           (attribute_of(*it) == TessellationAttribute::ControlOutput ||
				            attribute_of(*it) == TessellationAttribute::PatchOutput)) {
					last_block = block;
					last_write = it;
				} else if (opcode == ValueOpcode::GetTessellationAttribute &&
				           (attribute_of(*it) == TessellationAttribute::ControlOutput ||
				            attribute_of(*it) == TessellationAttribute::PatchOutput) &&
				           last_block != nullptr) {
					last_block->PrependNewInst(std::next(last_write), ValueOpcode::Barrier, {});
					last_block = nullptr;
					barriers++;
				}
			}
		}
	}
	ConstantPropagationPass(program.blocks, program.wave_size);
	RemoveIdentities(program.blocks);
	EliminateDeadCode(program.blocks);
	LOGF("%s tessellation lowering: reads=%u writes=%u factors=%u isa_patch_addresses=%u (0x%016" PRIx64
	     ")\n",
	     stage_name, reads, writes, factors, forced_patch, options.shader_hash);
	if (!indexed) return nullptr;
	if (barriers != 0u)
		LOGF("HS tessellation lowering: %u barrier(s) before control-point outputs are read "
		     "(0x%016" PRIx64 ")\n",
		     barriers, options.shader_hash);
	// The emitter indexes factors and patch constants by constant addresses only: a stage whose
	// addresses did not fold is skipped instead of ending in the emitter.
	const uint32_t factor_count = tess.domain == 0u ? 2u : tess.domain == 1u ? 4u : 6u;
	for (auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::GetTessellationAttribute &&
			    inst.GetOpcode() != ValueOpcode::SetTessellationAttribute)
				continue;
			const auto kind = static_cast<TessellationAttribute>(inst.Arg(0).U32());
			if (kind != TessellationAttribute::Factor && kind != TessellationAttribute::PatchOutput)
				continue;
			current_pc = 0;
			if (!inst.Arg(1).IsImmediate()) {
				return unsupported(kind == TessellationAttribute::Factor
				                       ? "a tessellation factor at a computed address"
				                       : "a patch constant at a computed address");
			}
			if (kind == TessellationAttribute::Factor && inst.Arg(1).U32() / 4u >= factor_count) {
				return unsupported("a tessellation factor past the domain's");
			}
		}
	}
	return nullptr;
}

} // namespace Libs::Graphics::ShaderRecompiler
