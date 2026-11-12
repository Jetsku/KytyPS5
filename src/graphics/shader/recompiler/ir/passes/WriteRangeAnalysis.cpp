#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"

#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <bit>
#include <fmt/format.h>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

using Op = WriteRangeNode::Op;

constexpr uint32_t MaxNodes = 4096;
constexpr uint32_t MaxDepth = 96;
constexpr uint64_t U32Max   = UINT32_MAX;
// Origin tags of unknown Range leaves (WriteRangeNode::a): 0 = untagged, opcode + 1 = the value
// came from an unsupported instruction, UnknownLoopOrDepth = loop-carried or too deep.
constexpr uint32_t UnknownLoopOrDepth = UINT32_MAX;

bool IsBuffer64BitAtomic(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::BufferAtomicSwap64:
		case ValueOpcode::BufferAtomicOr64:
		case ValueOpcode::BufferAtomicCmpSwap64:
		case ValueOpcode::BufferAtomicIAdd64:
		case ValueOpcode::BufferAtomicISub64:
		case ValueOpcode::BufferAtomicSMin64:
		case ValueOpcode::BufferAtomicUMin64:
		case ValueOpcode::BufferAtomicSMax64:
		case ValueOpcode::BufferAtomicUMax64:
		case ValueOpcode::BufferAtomicAnd64:
		case ValueOpcode::BufferAtomicXor64: return true;
		default: return false;
	}
}

// Bytes past the computed start address that one operation may touch, rounded to whole dwords:
// the emitter writes dword (or 64-bit) elements, subword stores read-modify-write their dword,
// raw vectors store one dword per component at +4k and formatted/typed stores place their
// components within one element of at most 16 bytes.
uint32_t AccessExtent(ValueOpcode opcode, const MemoryInfo& memory) {
	if (memory.formatted || memory.typed) {
		return 16u;
	}
	if (IsBuffer64BitAtomic(opcode)) {
		return 8u;
	}
	switch (opcode) {
		case ValueOpcode::StoreBufferU32x2: return 8u;
		case ValueOpcode::StoreBufferU32x3: return 12u;
		case ValueOpcode::StoreBufferU32x4: return 16u;
		default: return 4u;
	}
}

class Builder {
public:
	Builder(const Program& program, ShaderStageInputInfo input, WriteRangeProgram& out)
	    : m_program(program), m_out(out), m_workgroup(ShaderWorkgroupInput(program.stage, input)) {
		m_full = Leaf(0, UINT32_MAX);
	}

	[[nodiscard]] bool Overflowed() const { return m_overflow; }

	uint32_t Visit(Value value, uint32_t depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			switch (value.GetType()) {
				case Type::U1: return Constant(value.U1() ? 1u : 0u);
				case Type::U8: return Constant(value.U8());
				case Type::U16: return Constant(value.U16());
				case Type::U32: return Constant(value.U32());
				default: return m_full;
			}
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return m_full;
		}
		if (const auto found = m_memo.find(inst); found != m_memo.end()) {
			return found->second;
		}
		// A value reached again while it is being visited is loop-carried: unknown. The partial
		// result memoized above it stays sound because unknown contains every value.
		if (depth >= MaxDepth || !m_active.insert(inst).second) {
			return Unknown(UnknownLoopOrDepth);
		}
		const auto node = Translate(*inst, depth + 1u);
		m_active.erase(inst);
		m_memo.emplace(inst, node);
		return node;
	}

private:
	uint32_t Push(WriteRangeNode node) {
		if (m_out.nodes.size() >= MaxNodes) {
			m_overflow = true;
			return m_full;
		}
		m_out.nodes.push_back(node);
		return static_cast<uint32_t>(m_out.nodes.size() - 1u);
	}

	uint32_t Leaf(uint32_t lo, uint32_t hi) {
		return Push({.op = Op::Range, .lo = lo, .hi = hi});
	}

	// An unknown value, tagged with its origin for diagnostics (see UnknownLoopOrDepth).
	uint32_t Unknown(uint32_t origin) {
		if (const auto found = m_unknown.find(origin); found != m_unknown.end()) {
			return found->second;
		}
		const auto node = Push({.op = Op::Range, .a = origin, .lo = 0, .hi = UINT32_MAX});
		m_unknown.emplace(origin, node);
		return node;
	}

	uint32_t Unknown(ValueOpcode opcode) {
		return Unknown(static_cast<uint32_t>(opcode) + 1u);
	}

	uint32_t Constant(uint32_t value) {
		if (const auto found = m_constants.find(value); found != m_constants.end()) {
			return found->second;
		}
		const auto node = Leaf(value, value);
		m_constants.emplace(value, node);
		return node;
	}

	uint32_t Node(Op op, uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t hi = UINT32_MAX) {
		return Push({.op = op, .a = a, .b = b, .c = c, .lo = 0, .hi = hi});
	}

	uint32_t Arg(const Inst& inst, size_t index, uint32_t depth) {
		return index < inst.NumArgs() ? Visit(inst.Arg(index), depth) : m_full;
	}

	uint32_t Binary(Op op, const Inst& inst, uint32_t depth) {
		const auto a = Arg(inst, 0, depth);
		const auto b = Arg(inst, 1, depth);
		return Node(op, a, b);
	}

	[[nodiscard]] uint32_t LocalSize(uint32_t axis) const {
		// Compute-derivative layouts can widen an unset axis to two invocations.
		return std::max(m_workgroup->threads_num[axis], 2u);
	}

	uint32_t Builtin(const Inst& inst) {
		if (inst.NumArgs() != 2 || !inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate()) {
			return m_full;
		}
		const auto kind      = static_cast<StageInputKind>(inst.Arg(0).U32());
		const auto component = inst.Arg(1).U32();
		const bool compute   = m_program.stage == ShaderType::Compute;
		switch (kind) {
			case StageInputKind::LocalInvocationId:
				if (m_workgroup == nullptr || component >= 3u) return m_full;
				return Leaf(0, LocalSize(component) - 1u);
			case StageInputKind::LocalInvocationIndex: {
				if (m_workgroup == nullptr) return m_full;
				// Split wave64 layouts number guest lanes up to the next multiple of 64.
				const uint64_t threads =
				    static_cast<uint64_t>(LocalSize(0)) * LocalSize(1) * LocalSize(2);
				const uint64_t padded = (threads + 63u) / 64u * 64u;
				return padded - 1u > U32Max ? m_full : Leaf(0, static_cast<uint32_t>(padded - 1u));
			}
			case StageInputKind::WorkgroupId:
				if (!compute || m_workgroup == nullptr || component >= 3u) return m_full;
				return Node(Op::WorkgroupId, component);
			case StageInputKind::GlobalInvocationId:
				if (!compute || m_workgroup == nullptr || component >= 3u) return m_full;
				return Node(Op::GlobalInvocationId, component, 0, 0, LocalSize(component));
			default: return Unknown(ValueOpcode::GetBuiltin);
		}
	}

	uint32_t CompositeExtract(const Inst& inst, uint32_t depth) {
		if (inst.NumArgs() != 2 || !inst.Arg(1).IsImmediate()) {
			return m_full;
		}
		const auto  component = inst.Arg(1).U32();
		const auto* source    = inst.Arg(0).Resolve().TryInstruction();
		if (source == nullptr) {
			return m_full;
		}
		switch (source->GetOpcode()) {
			case ValueOpcode::IAddCarry32:
				// Component 0 is the wrapped sum, component 1 the carry.
				if (component == 0u) return Binary(Op::Add, *source, depth);
				return component == 1u ? Leaf(0, 1) : m_full;
			case ValueOpcode::CompositeConstructU32x2:
			case ValueOpcode::CompositeConstructU32x3:
			case ValueOpcode::CompositeConstructU32x4:
				return component < source->NumArgs() ? Visit(source->Arg(component), depth) : m_full;
			default: return m_full;
		}
	}

	uint32_t Translate(const Inst& inst, uint32_t depth) {
		switch (inst.GetOpcode()) {
			case ValueOpcode::Identity: return Arg(inst, 0, depth);
			case ValueOpcode::Phi: {
				if (inst.NumArgs() == 0) return m_full;
				auto result = Arg(inst, 0, depth);
				for (size_t index = 1; index < inst.NumArgs(); index++) {
					result = Node(Op::Union, result, Arg(inst, index, depth));
				}
				return result;
			}
			case ValueOpcode::GetUserData: {
				if (inst.NumArgs() != 1 || inst.Arg(0).GetType() != Type::ScalarReg) return m_full;
				const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
				if (reg < m_program.user_data_base) return m_full;
				return Node(Op::UserData, reg - m_program.user_data_base);
			}
			case ValueOpcode::ReadConst:
				if (inst.NumArgs() != 2 || !inst.Arg(1).IsImmediate()) {
					return Unknown(ValueOpcode::ReadConst);
				}
				return Node(Op::FlatSrt, inst.Arg(1).U32());
			case ValueOpcode::GetBuiltin: return Builtin(inst);
			case ValueOpcode::LaneId:
				// Tessellation-control lanes are control-point invocations.
				return m_program.stage == ShaderType::TessellationControl ? m_full : Leaf(0, 63);
			case ValueOpcode::SelectU32:
				return Node(Op::Union, Arg(inst, 1, depth), Arg(inst, 2, depth));
			case ValueOpcode::IAdd32: return Binary(Op::Add, inst, depth);
			case ValueOpcode::ISub32: return Binary(Op::Sub, inst, depth);
			case ValueOpcode::IMul32: return Binary(Op::Mul, inst, depth);
			case ValueOpcode::UDiv32: return Binary(Op::UDiv, inst, depth);
			case ValueOpcode::UMulHi: return Binary(Op::UMulHi, inst, depth);
			case ValueOpcode::ShiftLeftLogical32: return Binary(Op::Shl, inst, depth);
			case ValueOpcode::ShiftRightLogical32: return Binary(Op::Shr, inst, depth);
			case ValueOpcode::ShiftRightArithmetic32: return Binary(Op::Sar, inst, depth);
			case ValueOpcode::BitwiseAnd32: return Binary(Op::And, inst, depth);
			case ValueOpcode::BitwiseOr32: return Binary(Op::Or, inst, depth);
			case ValueOpcode::BitwiseXor32: return Binary(Op::Xor, inst, depth);
			case ValueOpcode::UMin32: return Binary(Op::UMin, inst, depth);
			case ValueOpcode::UMax32: return Binary(Op::UMax, inst, depth);
			case ValueOpcode::SMin32: return Binary(Op::SMin, inst, depth);
			case ValueOpcode::SMax32: return Binary(Op::SMax, inst, depth);
			case ValueOpcode::UMinTri32:
				return Node(Op::UMin, Node(Op::UMin, Arg(inst, 0, depth), Arg(inst, 1, depth)),
				            Arg(inst, 2, depth));
			case ValueOpcode::UMaxTri32:
				return Node(Op::UMax, Node(Op::UMax, Arg(inst, 0, depth), Arg(inst, 1, depth)),
				            Arg(inst, 2, depth));
			case ValueOpcode::SMinTri32:
				return Node(Op::SMin, Node(Op::SMin, Arg(inst, 0, depth), Arg(inst, 1, depth)),
				            Arg(inst, 2, depth));
			case ValueOpcode::SMaxTri32:
				return Node(Op::SMax, Node(Op::SMax, Arg(inst, 0, depth), Arg(inst, 1, depth)),
				            Arg(inst, 2, depth));
			case ValueOpcode::UMedTri32:
				// The median is one of the operands.
				return Node(Op::Union, Node(Op::Union, Arg(inst, 0, depth), Arg(inst, 1, depth)),
				            Arg(inst, 2, depth));
			case ValueOpcode::SMedTri32: {
				// med3 = max(min(a, b), min(max(a, b), c)), all signed.
				const auto a = Arg(inst, 0, depth);
				const auto b = Arg(inst, 1, depth);
				const auto c = Arg(inst, 2, depth);
				return Node(Op::SMax, Node(Op::SMin, a, b),
				            Node(Op::SMin, Node(Op::SMax, a, b), c));
			}
			case ValueOpcode::BitFieldUExtract:
				return Node(Op::BitExtract, Arg(inst, 0, depth), Arg(inst, 1, depth),
				            Arg(inst, 2, depth));
			case ValueOpcode::BitCount32: return Leaf(0, 32);
			case ValueOpcode::ConvertU32U16:
			case ValueOpcode::ConvertU16U32:
				return Node(Op::And, Arg(inst, 0, depth), Constant(0xffffu));
			case ValueOpcode::ConvertU32U8:
			case ValueOpcode::ConvertU8U32:
				return Node(Op::And, Arg(inst, 0, depth), Constant(0xffu));
			case ValueOpcode::CompositeExtractU32x2:
			case ValueOpcode::CompositeExtractU32x3:
			case ValueOpcode::CompositeExtractU32x4: return CompositeExtract(inst, depth);
			default:
				// Memory and image loads, lane exchanges (a shuffle from an inactive lane is
				// undefined), float conversions, 64-bit values, undefined values: unknown.
				return Unknown(inst.GetOpcode());
		}
	}

	const Program&                          m_program;
	WriteRangeProgram&                      m_out;
	const ShaderWorkgroupInputInfo*         m_workgroup = nullptr;
	std::unordered_map<const Inst*, uint32_t> m_memo;
	std::unordered_set<const Inst*>         m_active;
	std::unordered_map<uint32_t, uint32_t>  m_constants;
	std::unordered_map<uint32_t, uint32_t>  m_unknown;
	uint32_t                                m_full     = 0;
	bool                                    m_overflow = false;
};

// The bits a 32-bit value can have set (UINT32_MAX: unknown). Based on TheCruZ/KytyPS5-GTA
// (ResourceTracking.cpp PossibleU32Bits), with BitFieldUExtract added.
uint32_t PossibleBits(Value value, std::unordered_map<const Inst*, uint32_t>& known) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return UINT32_MAX;
	}
	// A cycle stays unknown; results are shared across the SSA graph.
	const auto [entry, inserted] = known.try_emplace(inst, UINT32_MAX);
	if (!inserted) {
		return entry->second;
	}
	uint32_t   bits = UINT32_MAX;
	const auto arg  = [&](uint32_t index) {
		return index < inst->NumArgs() ? PossibleBits(inst->Arg(index), known) : UINT32_MAX;
	};
	const auto immediate = [&](uint32_t index, uint32_t& out) {
		if (index >= inst->NumArgs()) return false;
		const auto at = inst->Arg(index).Resolve();
		if (!at.IsImmediate() || at.GetType() != Type::U32) return false;
		out = at.U32();
		return true;
	};
	switch (inst->GetOpcode()) {
		case ValueOpcode::Identity: bits = arg(0); break;
		case ValueOpcode::BitwiseAnd32: bits = arg(0) & arg(1); break;
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::UMax32: bits = arg(0) | arg(1); break;
		case ValueOpcode::SelectU32: bits = arg(1) | arg(2); break;
		case ValueOpcode::Phi: {
			uint32_t merged = 0;
			for (uint32_t index = 0; index < inst->NumArgs(); ++index) merged |= arg(index);
			if (inst->NumArgs() != 0u) bits = merged;
			break;
		}
		case ValueOpcode::ShiftLeftLogical32: {
			uint32_t shift = 0;
			if (immediate(1, shift)) bits = arg(0) << (shift & 31u);
			break;
		}
		case ValueOpcode::ShiftRightLogical32: {
			uint32_t shift = 0;
			if (immediate(1, shift)) bits = arg(0) >> (shift & 31u);
			break;
		}
		case ValueOpcode::BitFieldUExtract: {
			uint32_t offset = 0;
			uint32_t count  = 0;
			if (immediate(1, offset) && immediate(2, count) && offset < 32u && count <= 32u - offset) {
				const auto mask = count == 32u ? UINT32_MAX : (1u << count) - 1u;
				bits            = (arg(0) >> offset) & mask;
			}
			break;
		}
		case ValueOpcode::BitFieldInsert: {
			uint32_t offset = 0;
			uint32_t count  = 0;
			if (immediate(2, offset) && immediate(3, count) && offset < 32u && count <= 32u - offset) {
				const auto field = (count == 32u ? UINT32_MAX : (1u << count) - 1u) << offset;
				bits             = (arg(0) & ~field) | ((arg(1) << offset) & field);
			}
			break;
		}
		default: break;
	}
	known[inst] = bits;
	return bits;
}

// A LoadAddressU32 scalar address read of one dword (ResourceKind::ScalarAddress) that the
// emitter loads: its base pair handle, offset value and dword-aligned immediate.
bool ScalarAddressRead(const Program& program, const Inst* read, const Inst*& handle, Value& offset,
                       int32_t& immediate) {
	if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadAddressU32 || read->NumArgs() != 4u) {
		return false;
	}
	const auto index = read->Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) {
		return false;
	}
	const auto& memory = program.memory_info[index];
	const auto  high   = read->Arg(2).Resolve();
	const auto  active = read->Arg(3).Resolve();
	if (memory.kind != ResourceKind::ScalarAddress || memory.address_is_full || memory.planning_only ||
	    !high.IsImmediate() || high.GetType() != Type::U32 || high.U32() != 0u || !active.IsImmediate() ||
	    active.GetType() != Type::U1 || !active.U1()) {
		return false;
	}
	handle = read->Arg(0).Resolve().TryInstruction();
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
	    handle->NumArgs() != 2u) {
		return false;
	}
	offset    = read->Arg(1).Resolve();
	immediate = static_cast<int32_t>(memory.offset & ~3u);
	return true;
}

bool SameAddressBase(const Inst& left, const Inst& right) {
	return &left == &right ||
	       (left.Arg(0).Resolve() == right.Arg(0).Resolve() && left.Arg(1).Resolve() == right.Arg(1).Resolve());
}

// Collects DmaReadReach for the program (see ShaderIR.h). Every opcode the SPIR-V emitter turns
// into a page-table walk (get_bda_pointer) reaches here through the checks below: address loads,
// loads through a computed V#, and the BVH / ray-tracing walks, which are unbounded.
template <typename Visitor>
void CollectDmaReads(const Program& program, Visitor&& visit, DmaReadReach& dma) {
	constexpr size_t MaxAccesses = 256;
	dma = {};
	if (!program.info.uses_dma) {
		return;
	}
	const auto fail = [&](DmaUnbounded why) {
		if (dma.reason == 0u) {
			dma.reason = static_cast<uint32_t>(why);
		}
	};
	if (program.has_address_writes || program.info.bda_writes) {
		fail(DmaUnbounded::AddressWrites);
	}
	if (program.info.uses_bvh || program.info.ray_query_sites != 0u) {
		fail(DmaUnbounded::RayTracing);
	}
	std::unordered_map<const Inst*, uint32_t> known_bits;
	std::vector<DmaReadAccess>                accesses;
	const auto add = [&](DmaReadAccess access) {
		// Accesses that differ only in their immediate share one entry over the immediates' hull.
		for (auto& other: accesses) {
			auto a = other;
			auto b = access;
			a.immediate = b.immediate = 0;
			a.extent = b.extent = 0;
			if (!(a == b)) {
				continue;
			}
			const int64_t begin = std::min<int64_t>(other.immediate, access.immediate);
			const int64_t end   = std::max<int64_t>(int64_t {other.immediate} + other.extent,
			                                        int64_t {access.immediate} + access.extent);
			if (end - begin > INT32_MAX) {
				break;
			}
			other.immediate = static_cast<int32_t>(begin);
			other.extent    = static_cast<uint32_t>(end - begin);
			return;
		}
		accesses.push_back(access);
	};
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			if (dma.reason != 0u) {
				break;
			}
			const auto op      = inst.GetOpcode();
			const auto address = AddressOpcodeInfoOf(op);
			const auto buffer  = BufferAccessOf(op);
			if (address.access == AddressAccess::None && buffer == BufferAccess::None) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size() || inst.NumArgs() == 0u) {
				fail(DmaUnbounded::OtherAccess);
				break;
			}
			const auto& memory = program.memory_info[index];
			if (address.access != AddressAccess::None) {
				if (memory.kind == ResourceKind::Scratch || memory.kind == ResourceKind::Lds ||
				    memory.kind == ResourceKind::Gds) {
					continue;
				}
				if (op == ValueOpcode::LoadAddressU32 && memory.planning_only) {
					continue; // not emitted
				}
				if (address.access != AddressAccess::Read) {
					fail(DmaUnbounded::AddressWrites);
					break;
				}
				if (op != ValueOpcode::LoadAddressU8 && op != ValueOpcode::LoadAddressU16 &&
				    op != ValueOpcode::LoadAddressU32) {
					fail(DmaUnbounded::RayTracing);
					break;
				}
				if (memory.address_is_full) {
					fail(DmaUnbounded::FullAddress);
					break;
				}
				const auto* handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
				    handle->NumArgs() != 2u || inst.NumArgs() < 2u) {
					fail(DmaUnbounded::AddressHandle);
					break;
				}
				// The mesh prologue's index fetch (Translate.cpp): GetAddressResource(draw dword 4 & ~3,
				// draw dword 5). No guest code yields MeshDrawParameter.
				const auto mesh_dword = [](Value value, uint32_t dword) {
					const auto* source = value.Resolve().TryInstruction();
					return source != nullptr && source->GetOpcode() == ValueOpcode::MeshDrawParameter &&
					       source->NumArgs() == 1u && source->Arg(0).IsImmediate() && source->Arg(0).U32() == dword;
				};
				const auto* low_and = handle->Arg(0).Resolve().TryInstruction();
				if (mesh_dword(handle->Arg(1), 5u) && low_and != nullptr &&
				    low_and->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
				    (mesh_dword(low_and->Arg(0), 4u) || mesh_dword(low_and->Arg(1), 4u))) {
					DmaReadAccess mesh;
					mesh.kind = DmaReadAccess::Kind::MeshIndex;
					add(mesh);
					continue;
				}
				DmaReadAccess access;
				access.kind      = DmaReadAccess::Kind::Window;
				access.scalar    = memory.kind == ResourceKind::ScalarAddress;
				access.base_low  = visit(handle->Arg(0));
				access.base_high = visit(handle->Arg(1));
				access.offset    = visit(inst.Arg(1));
				access.immediate = static_cast<int32_t>(access.scalar ? memory.offset & ~3u : memory.offset);
				// Unaligned loads read the dword(s) holding their bytes: at most 8 bytes past the
				// aligned address; 16 leaves room for every load shape.
				access.extent = 16;
				add(access);
				continue;
			}
			if (memory.kind != ResourceKind::IndirectBuffer) {
				continue; // a bound descriptor, not a device address
			}
			if ((op == ValueOpcode::ReadConstBuffer || op == ValueOpcode::LoadAddressU32) &&
			    memory.planning_only) {
				continue; // not emitted
			}
			if (buffer != BufferAccess::Read) {
				fail(DmaUnbounded::AddressWrites);
				break;
			}
			const auto* handle = inst.Arg(0).Resolve().TryInstruction();
			if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetBufferResource ||
			    handle->NumArgs() != 4u) {
				fail(DmaUnbounded::IndirectHandle);
				break;
			}
			DmaReadAccess access;
			const bool    scalar = op == ValueOpcode::ReadConstBuffer;
			access.kind = scalar ? DmaReadAccess::Kind::IndirectScalar : DmaReadAccess::Kind::IndirectVector;
			for (uint32_t word = 0; word < 4u; ++word) {
				access.words[word] = visit(handle->Arg(word));
			}
			// The V# from one table entry: dwords k = 0..3 read at the same base and offset, the
			// immediate 4k past dword 0's.
			const Inst* table_handle = nullptr;
			Value       table_offset;
			int32_t     table_immediate = 0;
			if (ScalarAddressRead(program, handle->Arg(0).Resolve().TryInstruction(), table_handle,
			                      table_offset, table_immediate)) {
				access.table_words = 1u;
				for (uint32_t word = 1; word < 4u; ++word) {
					const Inst* other_handle = nullptr;
					Value       other_offset;
					int32_t     other_immediate = 0;
					if (ScalarAddressRead(program, handle->Arg(word).Resolve().TryInstruction(), other_handle,
					                      other_offset, other_immediate) &&
					    SameAddressBase(*other_handle, *table_handle) && other_offset == table_offset &&
					    int64_t {other_immediate} == int64_t {table_immediate} + 4 * word) {
						access.table_words |= static_cast<uint8_t>(1u << word);
					}
				}
				access.base_low        = visit(table_handle->Arg(0));
				access.base_high       = visit(table_handle->Arg(1));
				access.table_offset    = visit(table_offset);
				access.table_bits      = PossibleBits(table_offset, known_bits) & ~3u;
				access.table_immediate = table_immediate;
			}
			if (!scalar) {
				if (inst.NumArgs() < 4u) {
					fail(DmaUnbounded::IndirectHandle);
					break;
				}
				access.index     = visit(inst.Arg(1));
				access.offset    = visit(inst.Arg(2));
				access.soffset   = visit(inst.Arg(3));
				access.immediate = static_cast<int32_t>(memory.offset);
				access.extent    = 16;
			}
			add(access);
		}
	}
	if (dma.reason == 0u && accesses.size() > MaxAccesses) {
		fail(DmaUnbounded::TooManyAccesses);
	}
	if (dma.reason == 0u) {
		dma.bounded  = true;
		dma.accesses = std::move(accesses);
	}
}

// Interval arithmetic over unsigned 32-bit values. Every result contains all values the native
// operation (with 32-bit wrap-around) can produce for operands inside the operand intervals.
struct Range {
	uint64_t lo = 0;
	uint64_t hi = U32Max;
};

constexpr Range Full() {
	return {0, U32Max};
}

constexpr Range Checked(uint64_t lo, uint64_t hi) {
	// A bound past 32 bits means the native operation may wrap to any value.
	return hi > U32Max || lo > hi ? Full() : Range {lo, hi};
}

constexpr bool NonNegative(Range value) {
	return value.hi <= INT32_MAX;
}

uint64_t BitCeilMask(uint64_t value) {
	// Smallest 2^n - 1 >= value: an upper bound of OR/XOR of values not above `value`.
	uint64_t mask = 0;
	while (mask < value) {
		mask = mask * 2u + 1u;
	}
	return mask;
}

} // namespace

void AnalyzeBufferWriteRanges(Program& program, ShaderStageInputInfo input_info) {
	WriteRangeProgram result;
	const auto&       buffers = program.info.buffers;
	std::vector<bool> failed(buffers.size(), false);
	std::vector<std::vector<WriteRangeAccess>> accesses(buffers.size());
	bool              global_failure = false;

	Builder builder(program, input_info, result);
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			const auto access = BufferAccessOf(inst.GetOpcode());
			if (access != BufferAccess::Write && access != BufferAccess::Atomic) {
				continue;
			}
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= program.memory_info.size() || inst.NumArgs() < 4) {
				global_failure = true;
				continue;
			}
			const auto& memory = program.memory_info[index];
			if (memory.kind != ResourceKind::Buffer || memory.resource >= buffers.size()) {
				// Not attributable to a binding: no binding may be narrowed.
				global_failure = true;
				continue;
			}
			const auto resource = memory.resource;
			WriteRangeAccess entry;
			entry.index     = builder.Visit(inst.Arg(1));
			entry.offset    = builder.Visit(inst.Arg(2));
			entry.soffset   = builder.Visit(inst.Arg(3));
			entry.immediate = memory.offset;
			entry.extent    = AccessExtent(inst.GetOpcode(), memory);
			if (builder.Overflowed()) {
				failed[resource] = true;
			}
			accesses[resource].push_back(entry);
		}
	}
	for (uint32_t index = 0; index < buffers.size(); index++) {
		if (!buffers[index].written) {
			continue;
		}
		BufferWriteRange entry;
		entry.buffer        = index;
		// (KYTY_BUFFER_OOB_MODES' flag is not part of the descriptor's PackedStride.)
		entry.packed_stride = buffers[index].packed_stride & ~BufferOobOffsetCheck;
		entry.bounded = !global_failure && !failed[index] && !accesses[index].empty() &&
		                !builder.Overflowed();
		if (entry.bounded) {
			entry.accesses = std::move(accesses[index]);
		}
		result.buffers.push_back(std::move(entry));
	}
	// KYTY_BDA_REACH: the guest memory the device-address reads can reach (on the same tape).
	CollectDmaReads(program, [&](Value value) { return builder.Visit(value); }, result.dma);
	if (result.dma.bounded && builder.Overflowed()) {
		result.dma          = {};
		result.dma.reason   = static_cast<uint32_t>(DmaUnbounded::NodeOverflow);
	}
	if (!result.dma.bounded &&
	    (result.buffers.empty() ||
	     std::none_of(result.buffers.begin(), result.buffers.end(),
	                  [](const BufferWriteRange& entry) { return entry.bounded; }))) {
		result.nodes.clear();
		result.nodes.shrink_to_fit();
	}
	program.write_ranges = std::move(result);
}

void WriteRangeEvaluator::Evaluate(const WriteRangeProgram& program,
                                   const WriteRangeInputs& inputs) {
	m_program = &program;
	m_values.resize(program.nodes.size());
	const auto value_of = [&](uint32_t node) -> Range {
		if (node >= m_values.size()) return Full();
		return {m_values[node].lo, m_values[node].hi};
	};
	for (size_t index = 0; index < program.nodes.size(); index++) {
		const auto& node = program.nodes[index];
		// Operands always precede their users; anything else is malformed and unknown.
		const bool  ordered = node.op == Op::Range || node.op == Op::UserData ||
		                     node.op == Op::FlatSrt || node.op == Op::WorkgroupId ||
		                     node.op == Op::GlobalInvocationId ||
		                     (node.a < index && node.b < index && node.c < index);
		Range       r       = Full();
		const Range a       = ordered ? value_of(node.a) : Full();
		const Range b       = ordered ? value_of(node.b) : Full();
		switch (node.op) {
			case Op::Range: r = Checked(node.lo, node.hi); break;
			case Op::UserData:
				if (node.a < inputs.user_data.size()) {
					r = {inputs.user_data[node.a], inputs.user_data[node.a]};
				}
				break;
			case Op::FlatSrt:
				if (node.a < inputs.flattened_srt.size()) {
					r = {inputs.flattened_srt[node.a], inputs.flattened_srt[node.a]};
				}
				break;
			case Op::WorkgroupId:
				if (inputs.has_groups && node.a < 3u && inputs.groups[node.a] != 0u) {
					r = {0, inputs.groups[node.a] - 1u};
				}
				break;
			case Op::GlobalInvocationId:
				if (inputs.has_groups && node.a < 3u && inputs.groups[node.a] != 0u &&
				    node.hi != 0u) {
					r = Checked(0, static_cast<uint64_t>(inputs.groups[node.a]) * node.hi - 1u);
				}
				break;
			case Op::Add: r = Checked(a.lo + b.lo, a.hi + b.hi); break;
			case Op::Sub:
				if (a.lo >= b.hi) r = {a.lo - b.hi, a.hi - b.lo};
				break;
			case Op::Mul: r = Checked(a.lo * b.lo, a.hi * b.hi); break;
			case Op::Shl:
				if (b.hi < 32u) r = Checked(a.lo << b.lo, a.hi << b.hi);
				break;
			case Op::Shr:
				if (b.hi < 32u) r = {a.lo >> b.hi, a.hi >> b.lo};
				break;
			case Op::Sar:
				if (b.hi < 32u && NonNegative(a)) r = {a.lo >> b.hi, a.hi >> b.lo};
				break;
			case Op::And: r = {0, std::min(a.hi, b.hi)}; break;
			case Op::Or: r = {std::max(a.lo, b.lo), BitCeilMask(std::max(a.hi, b.hi))}; break;
			case Op::Xor: r = {0, BitCeilMask(std::max(a.hi, b.hi))}; break;
			case Op::UMin: r = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)}; break;
			case Op::UMax: r = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
			case Op::SMin:
				if (NonNegative(a) && NonNegative(b)) {
					r = {std::min(a.lo, b.lo), std::min(a.hi, b.hi)};
				}
				break;
			case Op::SMax:
				if (NonNegative(a) && NonNegative(b)) {
					r = {std::max(a.lo, b.lo), std::max(a.hi, b.hi)};
				}
				break;
			case Op::UDiv:
				if (b.lo != 0u) r = {a.lo / b.hi, a.hi / b.lo};
				break;
			case Op::UMulHi: r = {(a.lo * b.lo) >> 32u, (a.hi * b.hi) >> 32u}; break;
			case Op::BitExtract: {
				const Range count = ordered ? value_of(node.c) : Full();
				if (b.hi + count.hi <= 32u) {
					const uint64_t mask = count.hi >= 32u ? U32Max : (uint64_t {1} << count.hi) - 1u;
					r = {0, std::min(a.hi >> b.lo, mask)};
				}
				break;
			}
			case Op::Union: r = {std::min(a.lo, b.lo), std::max(a.hi, b.hi)}; break;
		}
		m_values[index] = {r.lo, r.hi};
	}
}

bool WriteRangeEvaluator::AccessSpan(const BufferWriteRange& buffer, const WriteRangeAccess& access,
                                     WriteRangeSpan& span) const {
	const auto value_of = [&](uint32_t node) -> Range {
		if (node >= m_values.size()) return Full();
		return {m_values[node].lo, m_values[node].hi};
	};
	const uint32_t packed       = buffer.packed_stride;
	const uint64_t stride       = packed & 0x3fffu;
	const bool     swizzle      = stride != 0u && (packed & (1u << 14u)) != 0u;
	const uint32_t index_stride = (packed >> 16u) & 3u;
	const bool     add_tid      = (packed & (1u << 20u)) != 0u;
	if (access.extent < 4u || access.extent > 16u) {
		return false;
	}

	Range index = value_of(access.index);
	if (add_tid) {
		index = Checked(index.lo, index.hi + 63u);
	}
	// Component k of the access adds k to the immediate before the address is formed.
	const Range offset = Checked(value_of(access.offset).lo + access.immediate,
	                             value_of(access.offset).hi + access.immediate + access.extent - 4u);
	const Range soffset = value_of(access.soffset);
	if (index.hi > U32Max || offset.hi > U32Max || soffset.hi > U32Max ||
	    (index.lo == 0 && index.hi == U32Max && stride != 0u) ||
	    (offset.lo == 0 && offset.hi == U32Max) || (soffset.lo == 0 && soffset.hi == U32Max)) {
		return false;
	}
	Range address;
	if (swizzle) {
		// (index_msb * stride + offset_msb) * indices + index_lsb * 4 + offset_lsb; every term is
		// monotonic, so the bound of each term comes from the operand upper bounds.
		const uint64_t indices   = uint64_t {1} << (index_stride + 3u);
		const uint64_t index_msb = index.hi >> (index_stride + 3u);
		const uint64_t index_lsb = std::min<uint64_t>(index.hi, indices - 1u);
		const uint64_t hi = (index_msb * stride + (offset.hi & ~uint64_t {3})) * indices +
		                    index_lsb * 4u + 3u;
		address = Checked(0, hi);
	} else {
		address = Checked(index.lo * stride + offset.lo, index.hi * stride + offset.hi);
	}
	const Range byte = Checked(address.lo + soffset.lo, address.hi + soffset.hi);
	// The 64-bit element path adds the binding's sub-alignment (< 256 bytes) in 32 bits; stay
	// clear of any wrap-around there too.
	if (byte.hi > U32Max - 512u || (byte.lo == 0 && byte.hi == U32Max)) {
		return false;
	}
	// Each dword write covers [a & ~3, (a & ~3) + 4) and each 64-bit element starts at most 7
	// bytes below its address, so [lo - 8, hi + 12) holds every byte written.
	span.begin = byte.lo >= 8u ? byte.lo - 8u : 0u;
	span.end   = byte.hi + 12u;
	return true;
}

bool WriteRangeEvaluator::Spans(const WriteRangeProgram& program, uint32_t buffer, uint64_t size,
                                std::vector<WriteRangeSpan>& spans) const {
	spans.clear();
	const auto* entry = program.Find(buffer);
	if (entry == nullptr || !entry->bounded || m_program != &program ||
	    m_values.size() != program.nodes.size()) {
		return false;
	}
	for (const auto& access: entry->accesses) {
		WriteRangeSpan span;
		if (!AccessSpan(*entry, access, span)) {
			spans.clear();
			return false;
		}
		span.end = std::min(span.end, size);
		if (span.begin < span.end) {
			spans.push_back(span);
		}
	}
	std::sort(spans.begin(), spans.end(),
	          [](const WriteRangeSpan& lhs, const WriteRangeSpan& rhs) {
		          return lhs.begin < rhs.begin;
	          });
	// Merge overlapping or nearly adjacent spans (tracking is page-granular anyway).
	constexpr uint64_t MergeGap = 4096;
	size_t             out      = 0;
	for (size_t index = 0; index < spans.size(); index++) {
		if (out != 0 && spans[index].begin <= spans[out - 1].end + MergeGap) {
			spans[out - 1].end = std::max(spans[out - 1].end, spans[index].end);
		} else {
			spans[out++] = spans[index];
		}
	}
	spans.resize(out);
	if (spans.size() > MaxSpans) {
		const WriteRangeSpan hull {spans.front().begin, spans.back().end};
		spans.assign(1, hull);
	}
	return true;
}

void NormalizeDmaSpans(std::vector<std::pair<uint64_t, uint64_t>>& spans, uint64_t page) {
	std::erase_if(spans, [](const auto& span) { return span.second <= span.first; });
	for (auto& [begin, end]: spans) {
		begin &= ~(page - 1u);
		end = end > UINT64_MAX - (page - 1u) ? UINT64_MAX : (end + page - 1u) & ~(page - 1u);
	}
	std::sort(spans.begin(), spans.end());
	size_t out = 0;
	for (size_t index = 0; index < spans.size(); index++) {
		if (spans[index].second <= spans[index].first) {
			continue;
		}
		if (out != 0 && spans[index].first <= spans[out - 1].second) {
			spans[out - 1].second = std::max(spans[out - 1].second, spans[index].second);
		} else {
			spans[out++] = spans[index];
		}
	}
	spans.resize(out);
}

bool WriteRangeEvaluator::DmaSpans(const WriteRangeProgram& program, const DmaReachInputs& inputs,
                                   std::vector<std::pair<uint64_t, uint64_t>>& spans,
                                   DmaReachFailure& why) const {
	why = DmaReachFailure::None;
	if (!program.dma.bounded || m_program != &program || m_values.size() != program.nodes.size()) {
		why = DmaReachFailure::NotBounded;
		return false;
	}
	const auto value_of = [&](uint32_t node) -> Range {
		if (node >= m_values.size()) return Full();
		return {m_values[node].lo, m_values[node].hi};
	};
	const auto exact = [&](uint32_t node, uint32_t& value) {
		const auto range = value_of(node);
		if (range.lo != range.hi) return false;
		value = static_cast<uint32_t>(range.lo);
		return true;
	};
	const auto fail = [&](DmaReachFailure reason) {
		why = reason;
		return false;
	};
	// [base + lo, base + hi) with lo, hi signed byte offsets; clamped to the 64-bit space.
	const auto add_span = [&](uint64_t base, int64_t lo, int64_t hi) {
		const uint64_t begin = lo < 0 ? (base > static_cast<uint64_t>(-lo) ? base + lo : 0)
		                              : base + static_cast<uint64_t>(lo);
		const uint64_t end   = base + static_cast<uint64_t>(std::max<int64_t>(hi, 0));
		if (end <= begin) return true;
		if (end - begin > inputs.max_span_bytes) return false;
		spans.emplace_back(begin, end);
		return true;
	};
	for (const auto& access: program.dma.accesses) {
		m_dma_failed = static_cast<size_t>(&access - program.dma.accesses.data());
		if (access.kind == DmaReadAccess::Kind::MeshIndex) {
			// Dword-aligned loads of the index bytes: [begin & ~3, end + 4).
			if (!inputs.mesh_index_known) {
				return fail(DmaReachFailure::Base);
			}
			if (inputs.mesh_index_end > inputs.mesh_index_begin &&
			    !add_span(inputs.mesh_index_begin & ~uint64_t {3}, 0,
			              static_cast<int64_t>(inputs.mesh_index_end - (inputs.mesh_index_begin & ~uint64_t {3})) + 4)) {
				return fail(DmaReachFailure::Size);
			}
			continue;
		}
		if (access.kind == DmaReadAccess::Kind::Window) {
			uint32_t low  = 0;
			uint32_t high = 0;
			if (!exact(access.base_low, low) || !exact(access.base_high, high)) {
				return fail(DmaReachFailure::Base);
			}
			// GuestAddress(): base + zext(offset) + sext(immediate); a scalar read clears bits 1:0 of
			// the base, the offset and the immediate (already cleared in `immediate`).
			uint64_t base = low | (uint64_t {high} << 32u);
			auto     offset = value_of(access.offset);
			if (access.scalar) {
				base &= ~uint64_t {3};
				offset.lo &= ~uint64_t {3};
			}
			// LoadBda reads the aligned dword(s) holding the bytes: 4 below covers the alignment.
			if (!add_span(base, static_cast<int64_t>(offset.lo) + access.immediate - 4,
			              static_cast<int64_t>(offset.hi) + access.immediate + access.extent)) {
				return fail(DmaReachFailure::Size);
			}
			continue;
		}
		const bool vector = access.kind == DmaReadAccess::Kind::IndirectVector;
		// The words a V# needs: 0..2 for a scalar load (RDNA2 ISA 7.2.1 uses base, stride and
		// records), 0..3 for a vector load (word 3 optional: unknown means the widest rules).
		std::array<uint32_t, 4> fixed {};
		std::array<bool, 4>     known {};
		for (uint32_t word = 0; word < 4u; ++word) {
			if ((access.table_words & (1u << word)) == 0u) {
				known[word] = exact(access.words[word], fixed[word]);
			}
		}
		const auto needed_known = [&](uint32_t word) {
			return (access.table_words & (1u << word)) != 0u || known[word];
		};
		if (!needed_known(0) || !needed_known(1) || !needed_known(2)) {
			return fail(DmaReachFailure::Base);
		}
		const auto reach_of = [&](const std::array<uint32_t, 4>& words, bool word3_known) {
			const uint64_t base    = words[0] | (uint64_t {words[1] & 0xffffu} << 32u);
			const uint64_t stride  = (words[1] >> 16u) & 0x3fffu;
			const bool     swizzle = stride != 0u && (words[1] >> 31u) != 0u;
			const uint64_t records = words[2];
			if (!vector) {
				// LoadIndirectScalarBuffer: (offset & ~3) + 4 <= size, address (base + offset) & ~3.
				const uint64_t size = stride == 0u ? records : stride * records;
				return add_span(base & ~uint64_t {3}, 0, static_cast<int64_t>((base & 3u) + size))
				           ? DmaReachFailure::None
				           : DmaReachFailure::Size;
			}
			const uint32_t format = word3_known ? (words[3] >> 12u) & 0x7fu : 1u;
			if (format == 0u) {
				return DmaReachFailure::None; // FORMAT 0 (unbound): every load reads 0
			}
			const bool     add_tid      = !word3_known || ((words[3] >> 23u) & 1u) != 0u;
			const uint32_t mode         = word3_known ? (words[3] >> 28u) & 3u : 2u; // 2: records only
			if (swizzle && !word3_known) {
				return DmaReachFailure::Swizzle;
			}
			const uint32_t index_stride = word3_known ? (words[3] >> 21u) & 3u : 0u;
			// Every in-range access (IndirectBufferInBounds) satisfies the bounds of its OOB_SELECT
			// mode; out-of-range lanes read nothing. byte = index * stride + offset + soffset.
			auto index = value_of(access.index);
			if (add_tid) index.hi += 63u;
			uint64_t index_hi   = index.hi;
			uint64_t offset_hi  = value_of(access.offset).hi + static_cast<uint64_t>(access.immediate < 0 ? 0 : access.immediate);
			if (access.immediate < 0) offset_hi = U32Max; // a wrapping immediate: unknown
			uint64_t soffset_hi = value_of(access.soffset).hi;
			if (mode != 2u && records == 0u) {
				return DmaReachFailure::None; // nothing is in range
			}
			// Modes 0 and 1 check index < records; mode 3 checks index < records - soffset only
			// when swizzled.
			if (mode == 0u || mode == 1u || (mode == 3u && swizzle)) {
				index_hi = std::min(index_hi, records - 1u);
			}
			if (mode == 0u) offset_hi = std::min(offset_hi, stride);
			if (mode == 3u) {
				soffset_hi = std::min(soffset_hi, records);
				offset_hi  = std::min(offset_hi, swizzle ? stride : records);
			}
			uint64_t byte_hi = 0;
			if (swizzle) {
				const uint64_t indices = uint64_t {1} << (index_stride + 3u);
				byte_hi = ((index_hi >> (index_stride + 3u)) * stride + (offset_hi & ~uint64_t {3})) * indices +
				          std::min<uint64_t>(index_hi, indices - 1u) * 4u + 3u + soffset_hi;
			} else if (mode == 3u) {
				// offset + soffset + bytes <= records.
				byte_hi = (stride == 0u ? 0u : index_hi * stride) + records;
			} else {
				byte_hi = (stride == 0u ? 0u : index_hi * stride) + offset_hi + soffset_hi;
			}
			// A sum that can pass 32 bits wraps in the shader: any byte of the 4 GiB past base.
			if ((stride != 0u && index_hi >= U32Max) || byte_hi + access.extent > U32Max) {
				return DmaReachFailure::Offset;
			}
			return add_span(base, -4, static_cast<int64_t>(byte_hi + access.extent + 4u))
			           ? DmaReachFailure::None
			           : DmaReachFailure::Size;
		};
		if (access.table_words == 0u) {
			const auto result = reach_of(fixed, known[3]);
			if (result != DmaReachFailure::None) return fail(result);
			continue;
		}
		uint32_t table_low  = 0;
		uint32_t table_high = 0;
		if (!exact(access.base_low, table_low) || !exact(access.base_high, table_high) ||
		    inputs.read == nullptr) {
			return fail(DmaReachFailure::Base);
		}
		const uint64_t table = (table_low | (uint64_t {table_high} << 32u)) & ~uint64_t {3};
		const auto     range = value_of(access.table_offset);
		const uint64_t lo    = range.lo & ~uint64_t {3};
		const uint64_t hi    = range.hi;
		const uint32_t bits  = access.table_bits;
		uint32_t       entries = 0;
		const auto visit_entry = [&](uint64_t offset) {
			const int64_t  at      = static_cast<int64_t>(offset) + access.table_immediate;
			const uint64_t address = at < 0 ? table - static_cast<uint64_t>(-at) : table + static_cast<uint64_t>(at);
			if (at < 0 && table < static_cast<uint64_t>(-at)) return DmaReachFailure::TableRead;
			std::array<uint32_t, 4> words = fixed;
			uint32_t first = 0;
			uint32_t last  = 0;
			for (uint32_t word = 0; word < 4u; ++word) {
				if ((access.table_words & (1u << word)) != 0u) {
					last = word;
				}
			}
			while ((access.table_words & (1u << first)) == 0u) ++first;
			std::array<uint32_t, 4> read {};
			if (!inputs.read(inputs.user, address + first * 4u,
			                 std::span<uint32_t>(read.data(), last - first + 1u))) {
				return DmaReachFailure::TableRead;
			}
			for (uint32_t word = first; word <= last; ++word) {
				if ((access.table_words & (1u << word)) != 0u) words[word] = read[word - first];
			}
			return reach_of(words, needed_known(3));
		};
		// Offsets in [lo, hi] (dword-aligned) whose set bits are possible.
		if (std::popcount(bits) <= 16) {
			for (uint32_t sub = bits;; sub = (sub - 1u) & bits) {
				if (sub >= lo && sub <= hi) {
					if (++entries > inputs.max_table_entries) return fail(DmaReachFailure::TableEntries);
					const auto result = visit_entry(sub);
					if (result != DmaReachFailure::None) return fail(result);
				}
				if (sub == 0u) break;
			}
		} else {
			if (hi >= U32Max || (hi - lo) / 4u + 1u > inputs.max_table_entries) {
				return fail(DmaReachFailure::TableEntries);
			}
			for (uint64_t offset = lo; offset <= hi; offset += 4u) {
				if ((offset & ~uint64_t {bits}) != 0u) continue;
				const auto result = visit_entry(offset);
				if (result != DmaReachFailure::None) return fail(result);
			}
		}
	}
	return true;
}

std::string WriteRangeEvaluator::DescribeDmaFailure(const WriteRangeProgram& program) const {
	if (m_dma_failed >= program.dma.accesses.size()) {
		return "-";
	}
	const auto& access = program.dma.accesses[m_dma_failed];
	const auto  node   = [&](uint32_t index) {
		if (index >= program.nodes.size()) return std::string("?");
		const auto& item  = program.nodes[index];
		const auto  value = index < m_values.size() ? m_values[index] : Interval {};
		std::string what;
		switch (item.op) {
			case Op::Range:
				what = item.a == 0u ? "range"
				       : item.a == UnknownLoopOrDepth
				           ? "loop"
				           : std::string(ValueOpcodeName(static_cast<ValueOpcode>(item.a - 1u)));
				break;
			case Op::UserData: what = fmt::format("ud{}", item.a); break;
			case Op::FlatSrt: what = fmt::format("flat{}", item.a); break;
			default: what = fmt::format("op{}", static_cast<uint32_t>(item.op)); break;
		}
		return fmt::format("{}[{:x}..{:x}]", what, value.lo, value.hi);
	};
	static constexpr const char* kinds[] = {"window", "indirect-scalar", "indirect-vector", "mesh-index"};
	std::string text = fmt::format("access {}/{} {}{} base {}:{} offset {}", m_dma_failed, program.dma.accesses.size(),
	                               kinds[static_cast<uint32_t>(access.kind) % 4u], access.scalar ? " scalar" : "",
	                               node(access.base_low), node(access.base_high), node(access.offset));
	if (access.kind != DmaReadAccess::Kind::Window) {
		text += fmt::format(" table_words {:x} table_offset {} words {} {} {} {}", access.table_words,
		                    node(access.table_offset), node(access.words[0]), node(access.words[1]),
		                    node(access.words[2]), node(access.words[3]));
	}
	return text;
}

std::string WriteRangeEvaluator::Describe(const WriteRangeProgram& program, uint32_t buffer) const {
	const auto* entry = program.Find(buffer);
	if (entry == nullptr) {
		return "not-written";
	}
	if (!entry->bounded) {
		return "unprovable";
	}
	const auto value_of = [&](uint32_t node) -> Range {
		if (m_program != &program || node >= m_values.size()) return Full();
		return {m_values[node].lo, m_values[node].hi};
	};
	// Names the unsupported sources an access expression depends on.
	const auto origins = [&](std::initializer_list<uint32_t> roots) {
		std::vector<uint32_t> stack(roots);
		std::vector<bool>     seen(program.nodes.size(), false);
		std::vector<uint32_t> tags;
		while (!stack.empty()) {
			const auto node = stack.back();
			stack.pop_back();
			if (node >= program.nodes.size() || seen[node]) continue;
			seen[node]       = true;
			const auto& item = program.nodes[node];
			switch (item.op) {
				case Op::Range:
					if (item.a != 0u && std::ranges::find(tags, item.a) == tags.end()) {
						tags.push_back(item.a);
					}
					break;
				case Op::UserData:
				case Op::FlatSrt:
				case Op::WorkgroupId:
				case Op::GlobalInvocationId: break;
				default:
					stack.push_back(item.a);
					stack.push_back(item.b);
					if (item.op == Op::BitExtract) stack.push_back(item.c);
					break;
			}
		}
		std::string names;
		for (const auto tag: tags) {
			if (!names.empty()) names += ',';
			names += tag == UnknownLoopOrDepth
			             ? std::string("loop")
			             : std::string(ValueOpcodeName(static_cast<ValueOpcode>(tag - 1u)));
		}
		return names;
	};
	std::string text = fmt::format("packed_stride=0x{:x}", entry->packed_stride);
	for (const auto& access: entry->accesses) {
		const auto     index   = value_of(access.index);
		const auto     offset  = value_of(access.offset);
		const auto     soffset = value_of(access.soffset);
		WriteRangeSpan span;
		const bool     bounded = m_program == &program && AccessSpan(*entry, access, span);
		const auto     unknown = origins({access.index, access.offset, access.soffset});
		text += fmt::format(" [idx={:x}..{:x} off={:x}..{:x} soff={:x}..{:x} imm={} ext={} -> {}{}{}]",
		                    index.lo, index.hi, offset.lo, offset.hi, soffset.lo, soffset.hi,
		                    access.immediate, access.extent,
		                    bounded ? fmt::format("{:x}..{:x}", span.begin, span.end)
		                            : std::string("unbounded"),
		                    unknown.empty() ? "" : " unknown=", unknown);
	}
	return text;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
