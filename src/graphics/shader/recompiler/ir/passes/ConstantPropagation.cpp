#include "graphics/shader/recompiler/ir/passes/ConstantPropagation.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

// Finite value sets for U32 SSA values: the exact set of values an instruction can produce when
// it is small (bit-field extracts of a few bits, masks, and arithmetic on such values), or
// unknown. A compare whose outcome is the same for every pair of possible operands folds to a
// constant. This is exact: nothing is assumed about unknown values. It exists for V_MOVRELS /
// V_MOVRELD, whose select chains compare M0 against every VGPR offset above the base, while the
// guest computes M0 from a 3-bit field (times a small stride), so most compares are always false.
class ValueSetAnalysis {
public:
	using Set = std::optional<std::vector<uint32_t>>;

	static constexpr size_t MaxValues  = 64;
	static constexpr size_t MaxProduct = 4096;
	static constexpr int    MaxDepth   = 24;

	// KYTY_MOVREL_KNOWN_ZEROS: equality compares also fold by known zero bits (KnownZeros).
	bool known_zeros = false;

	Set Of(Value value, int depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			if (value.GetType() == Type::U32) {
				return std::vector<uint32_t> {value.U32()};
			}
			return std::nullopt;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetType() != Type::U32 || depth > MaxDepth) {
			return std::nullopt;
		}
		if (const auto it = m_memo.find(inst); it != m_memo.end()) {
			return it->second;
		}
		if (!m_active.insert(inst).second) {
			return std::nullopt; // A cycle through phis: give up on this path.
		}
		auto result = Compute(*inst, depth + 1);
		m_active.erase(inst);
		m_memo.emplace(inst, result);
		return result;
	}

	// Folds a U32 compare when both operand sets are known and every pair gives the same answer.
	template <typename Predicate>
	std::optional<bool> Compare(Value lhs, Value rhs, Predicate predicate) {
		const auto a = Of(lhs);
		if (!a.has_value()) {
			return std::nullopt;
		}
		const auto b = Of(rhs);
		if (!b.has_value() || a->size() * b->size() > MaxProduct) {
			return std::nullopt;
		}
		bool any_true  = false;
		bool any_false = false;
		for (const auto x: *a) {
			for (const auto y: *b) {
				(predicate(x, y) ? any_true : any_false) = true;
				if (any_true && any_false) {
					return std::nullopt;
				}
			}
		}
		return any_true;
	}

	// KYTY_MOVREL_KNOWN_ZEROS: bits that are zero in every value a U32 can take, for values whose
	// set is unknown or too large. A loop counter times 4 (V_MOVRELS with M0 = i << 2) still has
	// two low zero bits. Under-approximated: a value reached again through a phi cycle, or deeper
	// than MaxDepth, counts as having none.
	uint32_t KnownZeros(Value value, int depth = 0) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return value.GetType() == Type::U32 ? ~value.U32() : 0u;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetType() != Type::U32 || depth > MaxDepth) {
			return 0u;
		}
		if (const auto it = m_zeros.find(inst); it != m_zeros.end()) {
			return it->second;
		}
		if (!m_zeros_active.insert(inst).second) {
			return 0u; // A cycle through phis: nothing is known along it.
		}
		const auto zeros = ComputeZeros(*inst, depth + 1);
		m_zeros_active.erase(inst);
		m_zeros.emplace(inst, zeros);
		return zeros;
	}

private:
	static constexpr uint32_t LowMask(uint32_t bits) {
		return bits >= 32u ? UINT32_MAX : (uint32_t {1} << bits) - 1u;
	}

	uint32_t ComputeZeros(const Inst& inst, int depth) {
		const auto zeros = [&](size_t index) { return KnownZeros(inst.Arg(index), depth); };
		// Low bits known zero.
		const auto trailing = [&](size_t index) {
			return static_cast<uint32_t>(std::countr_one(zeros(index)));
		};
		switch (inst.GetOpcode()) {
			case ValueOpcode::BitwiseAnd32: return zeros(0) | zeros(1);
			case ValueOpcode::BitwiseOr32:
			case ValueOpcode::BitwiseXor32: return zeros(0) & zeros(1);
			case ValueOpcode::ShiftLeftLogical32:
			case ValueOpcode::ShiftRightLogical32: {
				const auto shift = inst.Arg(1).Resolve();
				if (!shift.IsImmediate() || shift.GetType() != Type::U32) {
					return 0u;
				}
				const auto amount = shift.U32() & 31u;
				const auto source = zeros(0);
				return inst.GetOpcode() == ValueOpcode::ShiftLeftLogical32
				           ? (source << amount) | LowMask(amount)
				           : (source >> amount) | ~(UINT32_MAX >> amount);
			}
			// Modulo 2^32 a product has at least the sum of its factors' low zero bits, and a sum or
			// difference at least the fewer of its operands'.
			case ValueOpcode::IMul32: return LowMask(std::min(32u, trailing(0) + trailing(1)));
			case ValueOpcode::IAdd32:
			case ValueOpcode::ISub32: return LowMask(std::min(trailing(0), trailing(1)));
			case ValueOpcode::BitFieldUExtract: {
				const auto offset = inst.Arg(1).Resolve();
				const auto count  = inst.Arg(2).Resolve();
				if (!offset.IsImmediate() || !count.IsImmediate() || offset.GetType() != Type::U32 ||
				    count.GetType() != Type::U32 || offset.U32() > 32u ||
				    count.U32() > 32u - offset.U32()) {
					return 0u;
				}
				const auto field  = LowMask(count.U32());
				const auto source = offset.U32() == 32u ? UINT32_MAX : zeros(0) >> offset.U32();
				return ~field | (source & field);
			}
			// Zero extensions.
			case ValueOpcode::ConvertU32U16: return 0xffff0000u;
			case ValueOpcode::ConvertU32U8: return 0xffffff00u;
			// One lane's value of the source (a waterfall loop's key): the source's bits hold for it.
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane: return zeros(0);
			// One of the operands.
			case ValueOpcode::UMin32:
			case ValueOpcode::UMax32: return zeros(0) & zeros(1);
			case ValueOpcode::SelectU32: return zeros(1) & zeros(2);
			case ValueOpcode::Phi: {
				uint32_t result = UINT32_MAX;
				for (size_t index = 0; index < inst.NumArgs() && result != 0u; index++) {
					if (inst.Arg(index).Resolve().TryInstruction() == &inst) {
						continue;
					}
					result &= zeros(index);
				}
				return result;
			}
			default: return 0u;
		}
	}

	static Set Normalize(std::vector<uint32_t> values) {
		std::sort(values.begin(), values.end());
		values.erase(std::unique(values.begin(), values.end()), values.end());
		if (values.size() > MaxValues) {
			return std::nullopt;
		}
		return values;
	}

	template <typename Function>
	static Set Cross(const Set& a, const Set& b, Function function) {
		if (!a.has_value() || !b.has_value() || a->size() * b->size() > MaxProduct) {
			return std::nullopt;
		}
		std::vector<uint32_t> values;
		values.reserve(a->size() * b->size());
		for (const auto x: *a) {
			for (const auto y: *b) {
				values.push_back(function(x, y));
			}
		}
		return Normalize(std::move(values));
	}

	// Every value v with (v & ~mask) == 0, when the mask has few bits.
	static Set Submasks(uint32_t mask) {
		if (std::popcount(mask) > 6) {
			return std::nullopt;
		}
		std::vector<uint32_t> values;
		for (uint32_t subset = mask;; subset = (subset - 1u) & mask) {
			values.push_back(subset);
			if (subset == 0u) {
				break;
			}
		}
		return Normalize(std::move(values));
	}

	static uint32_t Bits(const std::vector<uint32_t>& values) {
		uint32_t bits = 0;
		for (const auto value: values) {
			bits |= value;
		}
		return bits;
	}

	Set Compute(const Inst& inst, int depth) {
		const auto arg = [&](size_t index) { return Of(inst.Arg(index), depth); };
		switch (inst.GetOpcode()) {
			case ValueOpcode::BitFieldUExtract: {
				const auto offset = inst.Arg(1).Resolve();
				const auto count  = inst.Arg(2).Resolve();
				if (!offset.IsImmediate() || !count.IsImmediate() ||
				    offset.GetType() != Type::U32 || count.GetType() != Type::U32 ||
				    offset.U32() > 32u || count.U32() > 32u - offset.U32()) {
					return std::nullopt;
				}
				const auto shift = offset.U32();
				const auto mask =
				    count.U32() == 32u ? UINT32_MAX : (uint32_t {1} << count.U32()) - 1u;
				if (count.U32() == 0u) {
					return std::vector<uint32_t> {0u};
				}
				if (const auto source = arg(0); source.has_value()) {
					std::vector<uint32_t> values;
					for (const auto value: *source) {
						values.push_back(shift == 32u ? 0u : (value >> shift) & mask);
					}
					return Normalize(std::move(values));
				}
				return Submasks(mask);
			}
			case ValueOpcode::BitwiseAnd32: {
				const auto a = arg(0);
				const auto b = arg(1);
				if (a.has_value() && b.has_value()) {
					return Cross(a, b, [](uint32_t x, uint32_t y) { return x & y; });
				}
				// x & y only keeps bits the known side can have.
				if (a.has_value()) {
					return Submasks(Bits(*a));
				}
				if (b.has_value()) {
					return Submasks(Bits(*b));
				}
				return std::nullopt;
			}
			case ValueOpcode::BitwiseOr32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x | y; });
			case ValueOpcode::BitwiseXor32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x ^ y; });
			case ValueOpcode::IAdd32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x + y; });
			case ValueOpcode::ISub32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x - y; });
			case ValueOpcode::IMul32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x * y; });
			case ValueOpcode::ShiftLeftLogical32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x << (y & 31u); });
			case ValueOpcode::ShiftRightLogical32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return x >> (y & 31u); });
			case ValueOpcode::UMin32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return std::min(x, y); });
			case ValueOpcode::UMax32:
				return Cross(arg(0), arg(1), [](uint32_t x, uint32_t y) { return std::max(x, y); });
			case ValueOpcode::SelectU32: {
				auto a = arg(1);
				auto b = arg(2);
				if (!a.has_value() || !b.has_value()) {
					return std::nullopt;
				}
				a->insert(a->end(), b->begin(), b->end());
				return Normalize(std::move(*a));
			}
			case ValueOpcode::Phi: {
				std::vector<uint32_t> values;
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					if (inst.Arg(index).Resolve().TryInstruction() == &inst) {
						continue;
					}
					const auto incoming = arg(index);
					if (!incoming.has_value()) {
						return std::nullopt;
					}
					values.insert(values.end(), incoming->begin(), incoming->end());
				}
				if (values.empty()) {
					return std::nullopt;
				}
				return Normalize(std::move(values));
			}
			default: return std::nullopt;
		}
	}

	std::unordered_map<const Inst*, Set>      m_memo;
	std::unordered_set<const Inst*>           m_active;
	std::unordered_map<const Inst*, uint32_t> m_zeros;
	std::unordered_set<const Inst*>           m_zeros_active;
};

Value Arg(const Inst& inst, size_t index) {
	return inst.Arg(index).Resolve();
}

bool IsImmediate(Value value, Type type) {
	return value.IsImmediate() && value.GetType() == type;
}

void Replace(Inst& inst, Value value) {
	inst.ReplaceUsesWith(value.Resolve());
}

template <typename Function>
bool FoldU32(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint32_t>(function(lhs.U32(), rhs.U32()))));
	return true;
}

template <typename Function>
bool FoldU64(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(lhs.U64(), rhs.U64()))));
	return true;
}

template <typename Function>
bool FoldU64Shift(Inst& inst, Function function) {
	const auto value = Arg(inst, 0);
	const auto shift = Arg(inst, 1);
	if (!IsImmediate(value, Type::U64) || !IsImmediate(shift, Type::U32)) {
		return false;
	}
	Replace(inst, Value(static_cast<uint64_t>(function(value.U64(), shift.U32()))));
	return true;
}

template <typename Function>
bool FoldU32Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U32) || !IsImmediate(rhs, Type::U32)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U32(), rhs.U32())));
	return true;
}

template <typename Function>
bool FoldU64Compare(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U64) || !IsImmediate(rhs, Type::U64)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U64(), rhs.U64())));
	return true;
}

template <typename Function>
bool FoldLogical(Inst& inst, Function function) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (!IsImmediate(lhs, Type::U1) || !IsImmediate(rhs, Type::U1)) {
		return false;
	}
	Replace(inst, Value(function(lhs.U1(), rhs.U1())));
	return true;
}

bool ReplaceBinaryIdentity(Inst& inst, Type type, uint64_t identity) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (IsImmediate(lhs, type) &&
	    (type == Type::U32 ? lhs.U32() == identity : lhs.U64() == identity)) {
		Replace(inst, rhs);
		return true;
	}
	if (IsImmediate(rhs, type) &&
	    (type == Type::U32 ? rhs.U32() == identity : rhs.U64() == identity)) {
		Replace(inst, lhs);
		return true;
	}
	return false;
}

// Whether one boolean is the LogicalNot of the other (KYTY_FOLD_LANE_MASKS: x && !x, x || !x).
bool IsComplement(Value lhs, Value rhs) {
	const auto negates = [](Value negation, Value value) {
		const auto* inst = negation.TryInstruction();
		return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalNot &&
		       Arg(*inst, 0) == value;
	};
	return negates(lhs, rhs) || negates(rhs, lhs);
}

bool FoldSelect(Inst& inst) {
	const auto condition   = Arg(inst, 0);
	const auto true_value  = Arg(inst, 1);
	const auto false_value = Arg(inst, 2);
	if (IsImmediate(condition, Type::U1)) {
		Replace(inst, condition.U1() ? true_value : false_value);
		return true;
	}
	if (true_value == false_value) {
		Replace(inst, true_value);
		return true;
	}
	return false;
}

bool FoldPhi(Inst& inst) {
	Value same;
	for (size_t index = 0; index < inst.NumArgs(); index++) {
		const auto value = Arg(inst, index);
		if (value.TryInstruction() == &inst) {
			continue;
		}
		if (same.IsEmpty()) {
			same = value;
		} else if (same != value) {
			return false;
		}
	}
	if (same.IsEmpty()) {
		return false;
	}
	Replace(inst, same);
	return true;
}

bool FoldBitCast(Inst& inst, ValueOpcode reverse) {
	const auto value = Arg(inst, 0);
	if (IsImmediate(value, Type::F32) && inst.GetOpcode() == ValueOpcode::BitCastU32F32) {
		Replace(inst, Value(std::bit_cast<uint32_t>(value.F32Value())));
		return true;
	}
	if (IsImmediate(value, Type::U32) && inst.GetOpcode() == ValueOpcode::BitCastF32U32) {
		Replace(inst, Value::F32(std::bit_cast<float>(value.U32())));
		return true;
	}
	if (IsImmediate(value, Type::F16) && inst.GetOpcode() == ValueOpcode::BitCastU16F16) {
		Replace(inst, Value(value.F16Bits()));
		return true;
	}
	if (IsImmediate(value, Type::U16) && inst.GetOpcode() == ValueOpcode::BitCastF16U16) {
		Replace(inst, Value::F16(value.U16()));
		return true;
	}
	if (auto* producer = value.TryInstruction();
	    producer != nullptr && producer->GetOpcode() == reverse) {
		Replace(inst, producer->Arg(0));
		return true;
	}
	return false;
}

bool FoldCompositeExtract(Inst& inst, ValueOpcode construct, size_t components) {
	const auto composite = Arg(inst, 0);
	const auto index     = Arg(inst, 1);
	if (!IsImmediate(index, Type::U32) || index.U32() >= components) {
		return false;
	}
	const auto  component = index.U32();
	const auto* producer  = composite.TryInstruction();
	if (IsImmediate(composite, Type::U64)) {
		Replace(inst, Value(static_cast<uint32_t>(composite.U64() >> (component * 32u))));
		return true;
	}
	if (producer != nullptr && producer->GetOpcode() == construct) {
		Replace(inst, producer->Arg(component));
		return true;
	}
	if (component < 2u && producer != nullptr &&
	    producer->GetOpcode() == ValueOpcode::IAddCarry32) {
		const auto lhs = producer->Arg(0).Resolve();
		const auto rhs = producer->Arg(1).Resolve();
		if (IsImmediate(lhs, Type::U32) && IsImmediate(rhs, Type::U32)) {
			const auto sum = static_cast<uint64_t>(lhs.U32()) + rhs.U32();
			Replace(inst, Value(component == 0u ? static_cast<uint32_t>(sum)
			                                    : static_cast<uint32_t>(sum >> 32u)));
			return true;
		}
	}
	return false;
}

template <typename Predicate>
void FoldU32CompareBySets(Inst& inst, ValueSetAnalysis* sets, Predicate predicate) {
	if (FoldU32Compare(inst, predicate) || sets == nullptr) {
		return;
	}
	if (const auto result = sets->Compare(Arg(inst, 0), Arg(inst, 1), predicate)) {
		Replace(inst, Value(*result));
	}
}

// KYTY_MOVREL_KNOWN_ZEROS: x == C (x != C) folds to false (true) when C sets a bit that x never
// has. Tried after the value sets, which fold more when x's set is known.
void FoldEqualityByKnownZeros(Inst& inst, ValueSetAnalysis* sets, ValueOpcode opcode) {
	if (sets == nullptr || !sets->known_zeros || inst.GetOpcode() != opcode) {
		return; // Disabled, or already folded (the instruction became an identity).
	}
	auto value    = Arg(inst, 0);
	auto constant = Arg(inst, 1);
	if (IsImmediate(value, Type::U32)) {
		std::swap(value, constant);
	}
	if (!IsImmediate(constant, Type::U32) || IsImmediate(value, Type::U32)) {
		return;
	}
	if ((constant.U32() & sets->KnownZeros(value)) != 0u) {
		Replace(inst, Value(opcode == ValueOpcode::INotEqual32));
	}
}

void FoldInstruction(Block& block, Block::iterator instruction,
                      std::unordered_set<Inst*>& lowered_ancillary, ValueSetAnalysis* sets) {
	auto& inst = *instruction;
	switch (inst.GetOpcode()) {
		case ValueOpcode::StoreBufferU32: {
			const auto data = ResolveActiveU32(Arg(inst, 4), Arg(inst, 5));
			if (!data.IsEmpty() && data != inst.Arg(4)) inst.SetArg(4, data);
			return;
		}
		case ValueOpcode::GetAttributeWithBary: {
			// The J operand of a V_INTERP_P2: when it is the second component of a hardware I/J
			// pair, the read gets that pair's interpolation; anything else (a computed J) keeps
			// the shader-wide interpolation.
			auto        mode = InterpolationMode::Unknown;
			const auto* j    = Arg(inst, 2).TryInstruction();
			if (j != nullptr && j->GetOpcode() == ValueOpcode::GetBuiltin &&
			    j->Arg(1).Resolve() == Value(1u)) {
				switch (static_cast<StageInputKind>(j->Arg(0).Resolve().U32())) {
					case StageInputKind::BaryCoordSmooth: mode = InterpolationMode::PerspectiveCenter; break;
					case StageInputKind::BaryCoordSmoothCentroid:
						mode = InterpolationMode::PerspectiveCentroid;
						break;
					case StageInputKind::BaryCoordSmoothSample:
						mode = InterpolationMode::PerspectiveSample;
						break;
					case StageInputKind::BaryCoordNoPerspective:
						mode = InterpolationMode::LinearCenter;
						break;
					case StageInputKind::BaryCoordNoPerspectiveCentroid:
						mode = InterpolationMode::LinearCentroid;
						break;
					case StageInputKind::BaryCoordNoPerspectiveSample:
						mode = InterpolationMode::LinearSample;
						break;
					default: break;
				}
			}
			auto read = block.PrependNewInst(instruction, ValueOpcode::GetAttribute,
			                                 {Arg(inst, 0), Arg(inst, 1)});
			read->SetFlags(static_cast<uint32_t>(mode));
			Replace(inst, Value(&*read));
			return;
		}
		case ValueOpcode::Phi: FoldPhi(inst); return;
		case ValueOpcode::SelectU1:
			if (!FoldSelect(inst) && IsImmediate(Arg(inst, 2), Type::U1) &&
			    !Arg(inst, 2).U1()) {
				auto result = block.PrependNewInst(instruction, ValueOpcode::LogicalAnd,
				                                   {Arg(inst, 0), Arg(inst, 1)});
				Replace(inst, Value(&*result));
			}
			return;
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32: FoldSelect(inst); return;
		case ValueOpcode::BitFieldInsert: {
			const auto base   = Arg(inst, 0);
			const auto insert = Arg(inst, 1);
			const auto offset = Arg(inst, 2);
			const auto count  = Arg(inst, 3);
			if (IsImmediate(base, Type::U32) && IsImmediate(insert, Type::U32) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) &&
			    offset.U32() <= 32u && count.U32() <= 32u - offset.U32()) {
				if (count.U32() == 0u) {
					Replace(inst, base);
					return;
				}
				const auto mask = count.U32() == 32u
				                      ? UINT32_MAX
				                      : ((uint32_t {1} << count.U32()) - 1u) << offset.U32();
				Replace(inst,
				        Value((base.U32() & ~mask) | ((insert.U32() << offset.U32()) & mask)));
			}
			return;
		}
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract: {
			const auto value  = Arg(inst, 0);
			const auto offset = Arg(inst, 1);
			const auto count  = Arg(inst, 2);
			auto* source = value.TryInstruction();
			if (source != nullptr && source->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32)) {
				const auto shift = Arg(*source, 1);
				if (IsImmediate(shift, Type::U32) && shift.U32() < 32u &&
				    offset.U32() <= shift.U32() && count.U32() <= shift.U32() - offset.U32()) {
					Replace(inst, Value(0u));
					return;
				}
			}
			if (source != nullptr && source->GetOpcode() == ValueOpcode::GetBuiltin &&
			    source->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::PackedAncillary)) &&
			    IsImmediate(offset, Type::U32) && IsImmediate(count, Type::U32) && count.U32() != 0u) {
				constexpr struct {
					uint32_t       start;
					uint32_t       end;
					StageInputKind kind;
				} fields[] = {{8u, 12u, StageInputKind::SampleId}, {16u, 27u, StageInputKind::Layer}};
				for (const auto& field: fields) {
					if (offset.U32() >= field.start && offset.U32() < field.end &&
					    count.U32() <= field.end - offset.U32()) {
						// Preserve extraction and sign extension while exposing only the used field.
						const auto input = block.PrependNewInst(
						    instruction, ValueOpcode::GetBuiltin,
						    {Value(static_cast<uint32_t>(field.kind)), Value(0u)});
						inst.SetArg(0, Value(&*input));
						inst.SetArg(1, Value(offset.U32() - field.start));
						lowered_ancillary.insert(source);
						return;
					}
				}
			}
			if (!IsImmediate(value, Type::U32) || !IsImmediate(offset, Type::U32) ||
			    !IsImmediate(count, Type::U32) || offset.U32() > 32u ||
			    count.U32() > 32u - offset.U32()) {
				return;
			}
			if (count.U32() == 0u) {
				Replace(inst, Value(0u));
			} else if (inst.GetOpcode() == ValueOpcode::BitFieldUExtract) {
				const auto mask =
				    count.U32() == 32u ? UINT32_MAX : (uint32_t {1} << count.U32()) - 1u;
				Replace(inst, Value((value.U32() >> offset.U32()) & mask));
			} else {
				const auto left = 32u - offset.U32() - count.U32();
				const auto bits = value.U32() << left;
				Replace(inst, Value(static_cast<uint32_t>(std::bit_cast<int32_t>(bits) >>
				                                          (left + offset.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCastU16F16: FoldBitCast(inst, ValueOpcode::BitCastF16U16); return;
		case ValueOpcode::BitCastF16U16: FoldBitCast(inst, ValueOpcode::BitCastU16F16); return;
		case ValueOpcode::BitCastU32F32: FoldBitCast(inst, ValueOpcode::BitCastF32U32); return;
		case ValueOpcode::BitCastF32U32: FoldBitCast(inst, ValueOpcode::BitCastU32F32); return;
		case ValueOpcode::ConvertU16U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint16_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U16: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U16)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U16())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU16U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::ConvertU8U32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint8_t>(value.U32())));
			}
			return;
		}
		case ValueOpcode::ConvertU32U8: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U8)) {
				Replace(inst, Value(static_cast<uint32_t>(value.U8())));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::ConvertU8U32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::CompositeExtractU64:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU64, 2);
			return;
		case ValueOpcode::CompositeExtractU32x2:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x2, 2);
			return;
		case ValueOpcode::CompositeExtractU32x3:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x3, 3);
			return;
		case ValueOpcode::CompositeExtractU32x4:
			FoldCompositeExtract(inst, ValueOpcode::CompositeConstructU32x4, 4);
			return;
		case ValueOpcode::CompositeConstructU64: {
			const auto low  = Arg(inst, 0);
			const auto high = Arg(inst, 1);
			if (IsImmediate(low, Type::U32) && IsImmediate(high, Type::U32)) {
				Replace(inst, Value(static_cast<uint64_t>(low.U32()) |
				                    (static_cast<uint64_t>(high.U32()) << 32u)));
			} else {
				const auto* low_extract  = low.TryInstruction();
				const auto* high_extract = high.TryInstruction();
				if (low_extract != nullptr && high_extract != nullptr &&
				    low_extract->GetOpcode() == ValueOpcode::CompositeExtractU64 &&
				    high_extract->GetOpcode() == ValueOpcode::CompositeExtractU64 &&
				    Arg(*low_extract, 1) == Value(0u) && Arg(*high_extract, 1) == Value(1u) &&
				    Arg(*low_extract, 0) == Arg(*high_extract, 0)) {
					// Guest register pairs need not unpack and repack between wide operations.
					Replace(inst, Arg(*low_extract, 0));
				}
			}
			return;
		}
		case ValueOpcode::IAdd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::IAdd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a + b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 0u);
			}
			return;
		case ValueOpcode::ISub32: {
			if (FoldU32(inst, [](uint32_t a, uint32_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::ISub64: {
			if (FoldU64(inst, [](uint64_t a, uint64_t b) { return a - b; })) {
				return;
			}
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U64) && rhs.U64() == 0u) {
				Replace(inst, Arg(inst, 0));
			}
			return;
		}
		case ValueOpcode::IMul32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 1u);
			}
			return;
		case ValueOpcode::IMul64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a * b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, 1u);
			}
			return;
		case ValueOpcode::WqmU64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				const auto expand = [](uint32_t word) {
					auto quads = word | (word >> 1u);
					quads |= quads >> 2u;
					return (quads & 0x11111111u) * 0x0fu;
				};
				const auto low  = expand(static_cast<uint32_t>(value.U64()));
				const auto high = expand(static_cast<uint32_t>(value.U64() >> 32u));
				Replace(inst,
				        Value(static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u)));
			}
			return;
		}
		case ValueOpcode::UDiv32: {
			const auto rhs = Arg(inst, 1);
			if (IsImmediate(rhs, Type::U32) && rhs.U32() == 1u) {
				Replace(inst, Arg(inst, 0));
			} else if (IsImmediate(rhs, Type::U32) && rhs.U32() != 0u) {
				FoldU32(inst, [](uint32_t a, uint32_t b) { return a / b; });
			}
			return;
		}
		case ValueOpcode::SMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				const auto product = static_cast<int64_t>(std::bit_cast<int32_t>(a)) *
				                     static_cast<int64_t>(std::bit_cast<int32_t>(b));
				return static_cast<uint32_t>(static_cast<uint64_t>(product) >> 32u);
			});
			return;
		case ValueOpcode::UMulHi:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32u);
			});
			return;
		case ValueOpcode::IAbs32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst,
				        Value((value.U32() & 0x80000000u) != 0u ? 0u - value.U32() : value.U32()));
			}
			return;
		}
		case ValueOpcode::ShiftLeftLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a << (b & 31u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a >> (b & 31u); })) {
				if (IsImmediate(Arg(inst, 0), Type::U32) && Arg(inst, 0).U32() == 0u) {
					Replace(inst, Value(0u));
					return;
				}
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 31u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic32:
			FoldU32(inst, [](uint32_t a, uint32_t b) {
				return static_cast<uint32_t>(std::bit_cast<int32_t>(a) >> (b & 31u));
			});
			return;
		case ValueOpcode::ShiftLeftLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a << (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightLogical64:
			if (!FoldU64Shift(inst, [](uint64_t a, uint32_t b) { return a >> (b & 63u); })) {
				const auto shift = Arg(inst, 1);
				if (IsImmediate(shift, Type::U32) && (shift.U32() & 63u) == 0u) {
					Replace(inst, Arg(inst, 0));
				}
			}
			return;
		case ValueOpcode::ShiftRightArithmetic64:
			FoldU64Shift(inst, [](uint64_t a, uint32_t b) {
				return static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
			});
			return;
		case ValueOpcode::BitwiseAnd32:
			if (!FoldU32(inst, [](uint32_t a, uint32_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0xffffffffu);
			}
			return;
		case ValueOpcode::BitwiseAnd64:
			if (!FoldU64(inst, [](uint64_t a, uint64_t b) { return a & b; })) {
				ReplaceBinaryIdentity(inst, Type::U64, UINT64_MAX);
			}
			return;
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
			if (!FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				    return opcode == ValueOpcode::BitwiseOr32 ? a | b : a ^ b;
			    })) {
				ReplaceBinaryIdentity(inst, Type::U32, 0u);
			}
			return;
		case ValueOpcode::BitwiseNot32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(~value.U32()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::BitwiseNot32) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		case ValueOpcode::BitReverse32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				uint32_t source = value.U32(), reversed = 0;
				for (uint32_t bit = 0; bit < 32u; ++bit, source >>= 1u) {
					reversed = (reversed << 1u) | (source & 1u);
				}
				Replace(inst, Value(reversed));
			}
			return;
		}
		case ValueOpcode::BitCount32: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U32)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U32()))));
			}
			return;
		}
		case ValueOpcode::BitCount64: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U64)) {
				Replace(inst, Value(static_cast<uint32_t>(std::popcount(value.U64()))));
			}
			return;
		}
		case ValueOpcode::SMin32:
		case ValueOpcode::SMax32:
			FoldU32(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				return opcode == ValueOpcode::SMin32 ? (lhs < rhs ? a : b) : (lhs > rhs ? a : b);
			});
			return;
		case ValueOpcode::UMin32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::min(a, b); });
			return;
		case ValueOpcode::UMax32:
			FoldU32(inst, [](uint32_t a, uint32_t b) { return std::max(a, b); });
			return;
		case ValueOpcode::IEqual32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a == b; });
			FoldEqualityByKnownZeros(inst, sets, ValueOpcode::IEqual32);
			return;
		case ValueOpcode::INotEqual32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a != b; });
			FoldEqualityByKnownZeros(inst, sets, ValueOpcode::INotEqual32);
			return;
		case ValueOpcode::ULessThan32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a < b; });
			return;
		case ValueOpcode::ULessThanEqual32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a <= b; });
			return;
		case ValueOpcode::UGreaterThan32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a > b; });
			return;
		case ValueOpcode::UGreaterThanEqual32:
			FoldU32CompareBySets(inst, sets, [](uint32_t a, uint32_t b) { return a >= b; });
			return;
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
			FoldU32Compare(inst, [opcode = inst.GetOpcode()](uint32_t a, uint32_t b) {
				const auto lhs = std::bit_cast<int32_t>(a);
				const auto rhs = std::bit_cast<int32_t>(b);
				switch (opcode) {
					case ValueOpcode::SLessThan32: return lhs < rhs;
					case ValueOpcode::SLessThanEqual32: return lhs <= rhs;
					case ValueOpcode::SGreaterThan32: return lhs > rhs;
					default: return lhs >= rhs;
				}
			});
			return;
		case ValueOpcode::IEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a == b; });
			return;
		case ValueOpcode::INotEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a != b; });
			return;
		case ValueOpcode::ULessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a < b; });
			return;
		case ValueOpcode::UGreaterThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a > b; });
			return;
		case ValueOpcode::ULessThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a <= b; });
			return;
		case ValueOpcode::UGreaterThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) { return a >= b; });
			return;
		case ValueOpcode::SLessThanEqual64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) {
				return std::bit_cast<int64_t>(a) <= std::bit_cast<int64_t>(b);
			});
			return;
		case ValueOpcode::SLessThan64:
			FoldU64Compare(inst, [](uint64_t a, uint64_t b) {
				return std::bit_cast<int64_t>(a) < std::bit_cast<int64_t>(b);
			});
			return;
		case ValueOpcode::LogicalAnd:
			if (!FoldLogical(inst, [](bool a, bool b) { return a && b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (GetCodegenOptions().fold_lane_masks && IsComplement(lhs, rhs)) {
					Replace(inst, Value(false));
					return;
				}
				const auto simplify = [&](Value assumption, Value expression) {
					const auto* disjunction = expression.TryInstruction();
					if (disjunction == nullptr || disjunction->GetOpcode() != ValueOpcode::LogicalOr) {
						return false;
					}
					for (uint32_t i = 0; i < 2u; ++i) {
						const auto* inverse = disjunction->Arg(i).Resolve().TryInstruction();
						if (inverse != nullptr && inverse->GetOpcode() == ValueOpcode::LogicalNot &&
						    inverse->Arg(0).Resolve() == assumption) {
							inst.SetArg(assumption == lhs ? 1u : 0u,
							            disjunction->Arg(i ^ 1u));
							return true;
						}
					}
					return false;
				};
				if (simplify(lhs, rhs) || simplify(rhs, lhs)) return;
				if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? rhs : lhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? lhs : rhs);
				}
			}
			return;
		case ValueOpcode::LogicalOr:
			if (!FoldLogical(inst, [](bool a, bool b) { return a || b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (GetCodegenOptions().fold_lane_masks && IsComplement(lhs, rhs)) {
					Replace(inst, Value(true));
				} else if (IsImmediate(lhs, Type::U1)) {
					Replace(inst, lhs.U1() ? lhs : rhs);
				} else if (IsImmediate(rhs, Type::U1)) {
					Replace(inst, rhs.U1() ? rhs : lhs);
				}
			}
			return;
		case ValueOpcode::LogicalXor:
			if (!FoldLogical(inst, [](bool a, bool b) { return a != b; })) {
				const auto lhs = Arg(inst, 0);
				const auto rhs = Arg(inst, 1);
				if (IsImmediate(lhs, Type::U1) && !lhs.U1()) {
					Replace(inst, rhs);
				} else if (IsImmediate(rhs, Type::U1) && !rhs.U1()) {
					Replace(inst, lhs);
				}
			}
			return;
		case ValueOpcode::LogicalNot: {
			const auto value = Arg(inst, 0);
			if (IsImmediate(value, Type::U1)) {
				Replace(inst, Value(!value.U1()));
			} else if (auto* producer = value.TryInstruction();
			           producer != nullptr && producer->GetOpcode() == ValueOpcode::LogicalNot) {
				Replace(inst, producer->Arg(0));
			}
			return;
		}
		default: return;
	}
}

// In wave32, extracting this invocation's bit from a ballot recovers its predicate.
// Keep that identity through scalar EXEC operations and their loop-carried mask Phis.
class LaneMaskProjection {
public:
	explicit LaneMaskProjection(std::unordered_set<Inst*>& lowered_ancillary)
	    : m_lowered_ancillary(lowered_ancillary) {}

	void Fold(Inst& inst) {
		if (inst.GetOpcode() != ValueOpcode::INotEqual32 || !Immediate(Arg(inst, 1), 0u)) return;
		const auto* bit = Arg(inst, 0).TryInstruction();
		if (bit == nullptr || bit->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
		    !Immediate(Arg(*bit, 1), 1u)) return;
		const auto* shift = Arg(*bit, 0).TryInstruction();
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftRightLogical32) return;
		const auto* index = Arg(*shift, 1).TryInstruction();
		if (index == nullptr || index->GetOpcode() != ValueOpcode::BitwiseAnd32 ||
		    !Immediate(Arg(*index, 1), 31u)) return;
		const auto* lane = Arg(*index, 0).TryInstruction();
		if (lane == nullptr || lane->GetOpcode() != ValueOpcode::LaneId) return;
		m_visited.clear();
		m_grounded = false;
		if (CanProject(Arg(*shift, 0)) && m_grounded) Replace(inst, Project(Arg(*shift, 0)));
	}

private:
	static bool Immediate(Value value, uint32_t expected) {
		return IsImmediate(value, Type::U32) && value.U32() == expected;
	}

	static Value BallotPredicate(const Inst& inst) {
		if (inst.GetOpcode() != ValueOpcode::CompositeExtractU32x4 ||
		    !Immediate(Arg(inst, 1), 0u)) return {};
		const auto* source = Arg(inst, 0).TryInstruction();
		return source != nullptr && source->GetOpcode() == ValueOpcode::Ballot
		           ? Arg(*source, 0) : Value {};
	}

	bool CanProject(Value value) {
		if (value.IsImmediate()) {
			const bool supported = Immediate(value, 0u) || Immediate(value, UINT32_MAX);
			m_grounded |= supported;
			return supported;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (m_values.contains(inst) || !BallotPredicate(*inst).IsEmpty()) {
			m_grounded = true;
			return true;
		}
		if (!m_visited.insert(inst).second) return true;
		const auto op = inst->GetOpcode();
		if (op != ValueOpcode::BitwiseAnd32 && op != ValueOpcode::BitwiseOr32 &&
		    op != ValueOpcode::BitwiseNot32 && op != ValueOpcode::SelectU32 && op != ValueOpcode::Phi)
			return false;
		for (size_t arg = op == ValueOpcode::SelectU32 ? 1u : 0u; arg < inst->NumArgs(); ++arg) {
			if (!CanProject(Arg(*inst, arg))) return false;
		}
		return inst->NumArgs() != 0;
	}

	Value Project(Value value) {
		if (value.IsImmediate()) return Value(value.U32() != 0u);
		auto* source = value.TryInstruction();
		if (const auto found = m_values.find(source); found != m_values.end()) return found->second.Resolve();
		if (const auto predicate = BallotPredicate(*source); !predicate.IsEmpty()) return predicate;
		auto* block = source->Parent();
		const auto where = std::find_if(block->begin(), block->end(),
		                               [source](const Inst& candidate) { return &candidate == source; });
		const auto op = source->GetOpcode() == ValueOpcode::BitwiseAnd32 ? ValueOpcode::LogicalAnd
		              : source->GetOpcode() == ValueOpcode::BitwiseOr32 ? ValueOpcode::LogicalOr
		              : source->GetOpcode() == ValueOpcode::BitwiseNot32 ? ValueOpcode::LogicalNot
		              : source->GetOpcode() == ValueOpcode::SelectU32 ? ValueOpcode::SelectU1
		                                                               : ValueOpcode::Phi;
		auto result = op == ValueOpcode::Phi ? block->PrependNewInst(where, op)
		            : op == ValueOpcode::LogicalNot
		                ? block->PrependNewInst(where, op, {Value(false)})
		            : op == ValueOpcode::SelectU1
		                ? block->PrependNewInst(where, op, {Arg(*source, 0), Value(false), Value(false)})
		                : block->PrependNewInst(where, op, {Value(false), Value(false)});
		m_values.emplace(source, Value(&*result));
		if (op == ValueOpcode::Phi) result->SetFlags(Type::U1);
		for (size_t arg = op == ValueOpcode::SelectU1 ? 1u : 0u; arg < source->NumArgs(); ++arg) {
			const auto projected = Project(Arg(*source, arg));
			if (op == ValueOpcode::Phi) result->AddPhiOperand(source->PhiBlock(arg), projected);
			else result->SetArg(arg, projected);
		}
		FoldInstruction(*block, result, m_lowered_ancillary, nullptr);
		return Value(&*result).Resolve();
	}

	std::unordered_set<Inst*>& m_lowered_ancillary;
	std::unordered_set<const Inst*> m_visited;
	std::unordered_map<const Inst*, Value> m_values;
	bool m_grounded = false;
};

// KYTY_FOLD_LANE_MASKS (FoldLaneMasks, Senaxx 5189ea360). Translator::ThreadBit reads the current
// lane's bit of a wave mask as
//   INotEqual32(BitwiseAnd32(ShiftRightLogical32(word, BitwiseAnd32(LaneId, 31)), 1), 0)
// where a wave64 word is SelectU32(ULessThan32(LaneId, 32), low, high).
struct LaneBitRead {
	Value low;
	Value high;
	Value high_half; // the ULessThan32 of a wave64 read; empty for wave32
};

bool IsOpcode(Value value, ValueOpcode opcode) {
	const auto* inst = value.TryInstruction();
	return inst != nullptr && inst->GetOpcode() == opcode;
}

// The operand of a commutative instruction that is not the immediate `constant`.
std::optional<Value> OtherThan(const Inst& inst, uint32_t constant) {
	const auto lhs = Arg(inst, 0);
	const auto rhs = Arg(inst, 1);
	if (IsImmediate(rhs, Type::U32) && rhs.U32() == constant) {
		return lhs;
	}
	if (IsImmediate(lhs, Type::U32) && lhs.U32() == constant) {
		return rhs;
	}
	return std::nullopt;
}

bool IsLaneBitIndex(Value value) {
	const auto* inst = value.TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseAnd32) {
		return false;
	}
	const auto lane = OtherThan(*inst, 31u);
	return lane && IsOpcode(*lane, ValueOpcode::LaneId);
}

std::optional<LaneBitRead> MatchLaneBit(const Inst& inst) {
	const auto masked = OtherThan(inst, 0u);
	if (!masked || !IsOpcode(*masked, ValueOpcode::BitwiseAnd32)) {
		return std::nullopt;
	}
	const auto shifted = OtherThan(*masked->TryInstruction(), 1u);
	if (!shifted || !IsOpcode(*shifted, ValueOpcode::ShiftRightLogical32)) {
		return std::nullopt;
	}
	const auto& shift = *shifted->TryInstruction();
	if (!IsLaneBitIndex(Arg(shift, 1))) {
		return std::nullopt;
	}
	const auto word = Arg(shift, 0);
	if (const auto* select = word.TryInstruction();
	    select != nullptr && select->GetOpcode() == ValueOpcode::SelectU32) {
		const auto  condition = Arg(*select, 0);
		const auto* compare   = condition.TryInstruction();
		if (compare != nullptr && compare->GetOpcode() == ValueOpcode::ULessThan32 &&
		    IsOpcode(Arg(*compare, 0), ValueOpcode::LaneId) &&
		    IsImmediate(Arg(*compare, 1), Type::U32) && Arg(*compare, 1).U32() == 32u) {
			return LaneBitRead {Arg(*select, 1), Arg(*select, 2), condition};
		}
	}
	return LaneBitRead {word, Value {}, Value {}};
}

// Word `part` of a U64 built by CompositeConstructU64, or of a U64 constant.
std::optional<Value> U64Word(Value value, uint32_t part) {
	if (IsImmediate(value, Type::U64)) {
		return Value(static_cast<uint32_t>(value.U64() >> (part * 32u)));
	}
	const auto* inst = value.TryInstruction();
	if (inst != nullptr && inst->GetOpcode() == ValueOpcode::CompositeConstructU64) {
		return Arg(*inst, part);
	}
	return std::nullopt;
}

// What the fold needs to know about the program. In a pixel shader, helper invocations may be
// left out of ballots (NVIDIA leaves them out), so there bit LaneId of Ballot(c) is c only for a
// lane that takes part in ballots: c AND Participating(). Lanes keep that status for the whole
// shader (a discard ends the invocation, OpKill). Elsewhere every running lane takes part.
struct LaneFold {
	Program& program;
	bool     pixel = false;
	Value    participating; // made on first use, at the top of the entry block
	// Lane bits already emitted in the current block, by (word, part): the same word gives the
	// same value, so x || !x stays recognisable (IsEveryLane compares instructions).
	std::unordered_map<const Inst*, std::array<Value, 2>> emitted;
	const Block*                                           emitted_block = nullptr;

	Value Participating() {
		if (!participating.IsEmpty()) {
			return participating;
		}
		auto&      entry = *program.blocks.front();
		const auto at    = entry.begin();
		const auto add   = [&](ValueOpcode opcode, std::initializer_list<Value> args) {
			return Value(&*entry.PrependNewInst(at, opcode, args));
		};
		const auto ballot = add(ValueOpcode::Ballot, {Value(true)});
		const auto lane   = add(ValueOpcode::LaneId, {});
		auto       word   = add(ValueOpcode::CompositeExtractU32x4, {ballot, Value(0u)});
		if (program.wave_size == 64u) {
			const auto high     = add(ValueOpcode::CompositeExtractU32x4, {ballot, Value(1u)});
			const auto low_half = add(ValueOpcode::ULessThan32, {lane, Value(32u)});
			word                = add(ValueOpcode::SelectU32, {low_half, word, high});
		}
		const auto bit     = add(ValueOpcode::BitwiseAnd32, {lane, Value(31u)});
		const auto shifted = add(ValueOpcode::ShiftRightLogical32, {word, bit});
		const auto masked  = add(ValueOpcode::BitwiseAnd32, {shifted, Value(1u)});
		// IEqual32(.., 1), not ThreadBit's INotEqual32(.., 0): FoldLaneMasks must not fold it.
		participating = add(ValueOpcode::IEqual32, {masked, Value(1u)});
		return participating;
	}
};

// Whether the current lane's bit of `word` is known to be set: an all-one constant, a ballot of
// true (outside pixel shaders; or under WQM, as a helper shares its quad with a lane that takes
// part), or logic or WQM of those. WQM only adds lanes, so it keeps a set bit set.
bool LaneBitSet(const LaneFold& fold, Value word, uint32_t part, int depth, bool under_wqm) {
	if (depth > 16) {
		return false;
	}
	if (IsImmediate(word, Type::U32)) {
		return word.U32() == UINT32_MAX;
	}
	const auto* inst = word.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU32x4: {
			const auto  index  = Arg(*inst, 1);
			const auto* ballot = Arg(*inst, 0).TryInstruction();
			return (!fold.pixel || under_wqm) && IsImmediate(index, Type::U32) &&
			       index.U32() == part && ballot != nullptr &&
			       ballot->GetOpcode() == ValueOpcode::Ballot &&
			       IsImmediate(Arg(*ballot, 0), Type::U1) && Arg(*ballot, 0).U1();
		}
		case ValueOpcode::CompositeExtractU64: {
			const auto index = Arg(*inst, 1);
			if (!IsImmediate(index, Type::U32) || index.U32() != part) {
				return false;
			}
			auto source = Arg(*inst, 0);
			bool wqm    = under_wqm;
			if (const auto* producer = source.TryInstruction();
			    producer != nullptr && producer->GetOpcode() == ValueOpcode::WqmU64) {
				source = Arg(*producer, 0);
				wqm    = true;
			}
			const auto component = U64Word(source, part);
			return component && LaneBitSet(fold, *component, part, depth + 1, wqm);
		}
		case ValueOpcode::BitwiseAnd32:
			return LaneBitSet(fold, Arg(*inst, 0), part, depth + 1, under_wqm) &&
			       LaneBitSet(fold, Arg(*inst, 1), part, depth + 1, under_wqm);
		case ValueOpcode::BitwiseOr32:
			return LaneBitSet(fold, Arg(*inst, 0), part, depth + 1, under_wqm) ||
			       LaneBitSet(fold, Arg(*inst, 1), part, depth + 1, under_wqm);
		default: return false;
	}
}

// Whether the current lane's bit of `word` (part 0: lanes 0-31, part 1: lanes 32-63) is known
// without the word: a ballot of a predicate, an all-zero or all-one constant, a bit known to be
// set (LaneBitSet), or bitwise and/or/xor/not of those.
bool LaneBitKnown(const LaneFold& fold, Value word, uint32_t part, int depth) {
	if (depth > 16) {
		return false;
	}
	if (IsImmediate(word, Type::U32)) {
		return word.U32() == 0u || word.U32() == UINT32_MAX;
	}
	if (LaneBitSet(fold, word, part, depth, false)) {
		return true;
	}
	const auto* inst = word.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU64: {
			const auto index = Arg(*inst, 1);
			if (!IsImmediate(index, Type::U32) || index.U32() != part) {
				return false;
			}
			const auto component = U64Word(Arg(*inst, 0), part);
			return component && LaneBitKnown(fold, *component, part, depth + 1);
		}
		case ValueOpcode::CompositeExtractU32x4: {
			const auto index = Arg(*inst, 1);
			return IsImmediate(index, Type::U32) && index.U32() == part &&
			       IsOpcode(Arg(*inst, 0), ValueOpcode::Ballot);
		}
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
			return LaneBitKnown(fold, Arg(*inst, 0), part, depth + 1) &&
			       LaneBitKnown(fold, Arg(*inst, 1), part, depth + 1);
		case ValueOpcode::BitwiseNot32: return LaneBitKnown(fold, Arg(*inst, 0), part, depth + 1);
		default: return false;
	}
}

Value EmitLaneBitUncached(LaneFold& fold, Value word, uint32_t part, Block& block,
                          Block::iterator at);

// Emits the current lane's bit of `word` before `at`; LaneBitKnown(word, part) must hold.
Value EmitLaneBit(LaneFold& fold, Value word, uint32_t part, Block& block, Block::iterator at) {
	if (IsImmediate(word, Type::U32)) {
		return Value(word.U32() != 0u);
	}
	if (fold.emitted_block != &block) {
		fold.emitted.clear();
		fold.emitted_block = &block;
	}
	const auto* key = word.TryInstruction();
	if (key != nullptr) {
		if (const auto found = fold.emitted.find(key);
		    found != fold.emitted.end() && !found->second[part].IsEmpty()) {
			return found->second[part];
		}
	}
	const auto value = EmitLaneBitUncached(fold, word, part, block, at);
	if (key != nullptr) {
		fold.emitted[key][part] = value;
	}
	return value;
}

Value EmitLaneBitUncached(LaneFold& fold, Value word, uint32_t part, Block& block,
                          Block::iterator at) {
	if (LaneBitSet(fold, word, part, 0, false)) {
		return Value(true);
	}
	const auto* inst   = word.TryInstruction();
	const auto  binary = [&](ValueOpcode opcode) {
		const auto lhs = EmitLaneBit(fold, Arg(*inst, 0), part, block, at);
		const auto rhs = EmitLaneBit(fold, Arg(*inst, 1), part, block, at);
		return Value(&*block.PrependNewInst(at, opcode, {lhs, rhs}));
	};
	switch (inst->GetOpcode()) {
		case ValueOpcode::CompositeExtractU64:
			return EmitLaneBit(fold, *U64Word(Arg(*inst, 0), part), part, block, at);
		case ValueOpcode::CompositeExtractU32x4: {
			// Bit LaneId of a ballot is the current lane's own predicate, if the lane takes part.
			const auto predicate = Arg(*Arg(*inst, 0).TryInstruction(), 0);
			if (!fold.pixel) {
				return predicate;
			}
			return Value(&*block.PrependNewInst(at, ValueOpcode::LogicalAnd,
			                                    {predicate, fold.Participating()}));
		}
		case ValueOpcode::BitwiseAnd32: return binary(ValueOpcode::LogicalAnd);
		case ValueOpcode::BitwiseOr32: return binary(ValueOpcode::LogicalOr);
		case ValueOpcode::BitwiseXor32: return binary(ValueOpcode::LogicalXor);
		default: {
			const auto value = EmitLaneBit(fold, Arg(*inst, 0), part, block, at);
			return Value(&*block.PrependNewInst(at, ValueOpcode::LogicalNot, {value}));
		}
	}
}

} // namespace

// KYTY_FOLD_LANE_MASKS=1 (CodegenOptions::fold_lane_masks, default off).
uint32_t FoldLaneMasks(Program& program) {
	// A hull shader's LaneId is its invocation id, not its subgroup lane.
	if (!GetCodegenOptions().fold_lane_masks || program.stage == ShaderType::TessellationControl ||
	    program.blocks.empty()) {
		return 0;
	}
	LaneFold fold {program, program.stage == ShaderType::Pixel, Value {}};
	uint32_t folded = 0;
	for (auto* block: program.blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			if (inst->GetOpcode() != ValueOpcode::INotEqual32) {
				continue;
			}
			const auto read = MatchLaneBit(*inst);
			if (!read || !LaneBitKnown(fold, read->low, 0u, 0) ||
			    (!read->high_half.IsEmpty() && !LaneBitKnown(fold, read->high, 1u, 0))) {
				continue;
			}
			auto value = EmitLaneBit(fold, read->low, 0u, *block, inst);
			if (!read->high_half.IsEmpty()) {
				const auto high = EmitLaneBit(fold, read->high, 1u, *block, inst);
				if (high != value) {
					value = Value(&*block->PrependNewInst(inst, ValueOpcode::SelectU1,
					                                      {read->high_half, value, high}));
				}
			}
			Replace(*inst, value);
			folded++;
		}
	}
	return folded;
}

void ConstantPropagationPass(const BlockList& blocks, uint32_t wave_size) {
	std::unordered_set<Inst*> lowered_ancillary;
	ValueSetAnalysis          value_sets;
	value_sets.known_zeros = GetCodegenOptions().movrel_known_zeros;
	auto* sets = GetCodegenOptions().movrel_range ? &value_sets : nullptr;
	LaneMaskProjection mask_projection(lowered_ancillary);
	for (auto* block: blocks) {
		for (auto inst = block->begin(); inst != block->end(); ++inst) {
			if (wave_size == 32u) mask_projection.Fold(*inst);
			FoldInstruction(*block, inst, lowered_ancillary, sets);
		}
	}
	// Normalize retained PHI/select values only after every supported field read has
	// been lowered; direct raw consumers remain unsupported.
	for (auto* source: lowered_ancillary) {
		const bool retained_only = std::ranges::all_of(source->Uses(), [](const Use& use) {
			const auto& user = *use.user;
			if (!user.HasUses() && !user.MayHaveSideEffects()) {
				return true;
			}
			return user.GetOpcode() == ValueOpcode::Phi ||
			       (user.GetOpcode() == ValueOpcode::SelectU32 && use.operand == 2u);
		});
		if (retained_only) {
			Replace(*source, Value(0u));
		}
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
