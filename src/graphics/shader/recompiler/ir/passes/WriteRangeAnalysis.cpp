#include "graphics/shader/recompiler/ir/passes/WriteRangeAnalysis.h"

#include "graphics/shader/shader.h"

#include <algorithm>
#include <fmt/format.h>
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
	if (result.buffers.empty() ||
	    std::none_of(result.buffers.begin(), result.buffers.end(),
	                 [](const BufferWriteRange& entry) { return entry.bounded; })) {
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
