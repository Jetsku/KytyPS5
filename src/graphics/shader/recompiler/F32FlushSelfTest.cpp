#include "graphics/shader/recompiler/F32FlushSelfTest.h"

#include <cinttypes>
#include <cstdio>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest {

namespace {

// Integers ordered as the f32 values they encode (non-NaN bits only; +0 and -0 equal).
int64_t OrderKey(uint32_t bits) {
	const auto magnitude = static_cast<int64_t>(bits & 0x7fffffffu);
	return (bits & 0x80000000u) != 0u ? -magnitude : magnitude;
}

std::string Hex(uint32_t bits) {
	char text[16];
	std::snprintf(text, sizeof(text), "0x%08" PRIx32, bits);
	return text;
}

} // namespace

const char* UnaryName(UnaryOp op) {
	switch (op) {
		case UnaryOp::Rcp: return "rcp";
		case UnaryOp::Rsq: return "rsq";
		case UnaryOp::Sqrt: return "sqrt";
		case UnaryOp::Exp: return "exp";
		case UnaryOp::Log: return "log";
	}
	return "?";
}

const char* CompareName(CompareOp op) {
	static constexpr const char* names[CompareOps] = {
	    "OpFOrdEqual",           "OpFUnordEqual",           "OpFOrdNotEqual",
	    "OpFUnordNotEqual",      "OpFOrdLessThan",          "OpFUnordLessThan",
	    "OpFOrdGreaterThan",     "OpFUnordGreaterThan",     "OpFOrdLessThanEqual",
	    "OpFUnordLessThanEqual", "OpFOrdGreaterThanEqual", "OpFUnordGreaterThanEqual",
	};
	const auto index = static_cast<uint32_t>(op);
	return index < CompareOps ? names[index] : "?";
}

std::vector<uint32_t> Inputs() {
	std::vector<uint32_t>        inputs;
	std::unordered_set<uint32_t> seen;
	inputs.reserve(InputCount);
	const auto add = [&](uint32_t magnitude) {
		magnitude &= 0x007fffffu;
		if (magnitude == 0u || inputs.size() >= InputCount || !seen.insert(magnitude).second) {
			return;
		}
		inputs.push_back(magnitude);
		inputs.push_back(magnitude | 0x80000000u);
	};
	add(0x00000001u);
	add(0x007fffffu);
	add(0x007ffffeu);
	add(0x00000002u);
	add(0x00000003u);
	add(0x00400000u);
	add(0x00400001u);
	add(0x003fffffu);
	for (uint32_t k = 0; k < 23; k++) {
		add(1u << k);
		add((1u << k) + 1u);
		add((2u << k) - 1u);
	}
	// Log-uniform: a magnitude with its top bit at a uniform position.
	uint64_t state = 0x2545f4914f6cdd1dull;
	while (inputs.size() < InputCount) {
		state             = state * 6364136223846793005ull + 1442695040888963407ull;
		const auto random = static_cast<uint32_t>(state >> 32u);
		const auto top    = random % 23u;
		const auto low    = (random >> 5u) & ((1u << top) - 1u);
		add((1u << top) | low);
	}
	return inputs;
}

std::array<uint32_t, OperandCount> Operands() {
	return {
	    0x00000000u, 0x80000000u, // +-0
	    0x00000001u, 0x80000001u, // smallest denormals
	    0x00000002u, 0x80000002u, //
	    0x00400000u, 0x80400000u, //
	    0x007fffffu, 0x807fffffu, // largest denormals
	    0x00123456u, 0x80654321u, //
	    0x00800000u, 0x80800000u, // smallest normals
	    0x00800001u, 0x3f800000u, // a normal, 1.0
	    0xbf800000u, 0x7f7fffffu, // -1.0, FLT_MAX
	    0xff7fffffu, 0x7f800000u, // -FLT_MAX, +inf
	    0xff800000u, 0x7fc00000u, // -inf, quiet NaN
	    0xffc00000u, 0x7f800001u, // negative quiet NaN, signaling NaN
	};
}

std::vector<uint32_t> InitialBuffer() {
	std::vector<uint32_t> buffer(BufferWords, 0u);
	const auto            inputs = Inputs();
	for (uint32_t i = 0; i < InputCount; i++) {
		buffer[InputBase + i] = inputs[i];
	}
	const auto operands = Operands();
	for (uint32_t j = 0; j < OperandCount; j++) {
		buffer[OperandBase + j] = operands[j];
	}
	return buffer;
}

uint32_t ExpectedUnary(UnaryOp op, uint32_t input_bits) {
	const bool negative = (FlushF32(input_bits) & 0x80000000u) != 0u;
	switch (op) {
		case UnaryOp::Rcp:
		case UnaryOp::Rsq: return negative ? 0xff800000u : 0x7f800000u; // 1/+-0, 1/sqrt(+-0)
		case UnaryOp::Sqrt: return negative ? 0x80000000u : 0x00000000u;
		case UnaryOp::Exp: return 0x3f800000u;
		case UnaryOp::Log: return 0xff800000u;
	}
	return 0u;
}

bool ExpectedCompare(CompareOp op, uint32_t lhs_bits, uint32_t rhs_bits) {
	const auto lhs       = FlushF32(lhs_bits);
	const auto rhs       = FlushF32(rhs_bits);
	const bool unordered = IsNanF32(lhs) || IsNanF32(rhs);
	const auto a         = OrderKey(lhs);
	const auto b         = OrderKey(rhs);
	const auto ordered   = [&](bool relation) { return !unordered && relation; };
	const auto either    = [&](bool relation) { return unordered || relation; };
	switch (op) {
		case CompareOp::OrdEqual: return ordered(a == b);
		case CompareOp::UnordEqual: return either(a == b);
		case CompareOp::OrdNotEqual: return ordered(a != b);
		case CompareOp::UnordNotEqual: return either(a != b);
		case CompareOp::OrdLessThan: return ordered(a < b);
		case CompareOp::UnordLessThan: return either(a < b);
		case CompareOp::OrdGreaterThan: return ordered(a > b);
		case CompareOp::UnordGreaterThan: return either(a > b);
		case CompareOp::OrdLessThanEqual: return ordered(a <= b);
		case CompareOp::UnordLessThanEqual: return either(a <= b);
		case CompareOp::OrdGreaterThanEqual: return ordered(a >= b);
		case CompareOp::UnordGreaterThanEqual: return either(a >= b);
	}
	return false;
}

namespace {

// Compares one result plane per op against `expected` (a plane or the model).
template <typename ExpectedUnaryFn, typename ExpectedMaskFn>
Verdict ClassifyAgainst(std::span<const uint32_t> buffer, bool actual_side, ExpectedUnaryFn expected_unary,
                        ExpectedMaskFn expected_mask) {
	Verdict verdict;
	if (buffer.size() < BufferWords) {
		verdict.mismatches = 1;
		verdict.first      = "read-back buffer too small";
		return verdict;
	}
	// Only the first mismatch is described.
	const auto note = [&](uint32_t op, const auto& describe) {
		verdict.by_op[op]++;
		if (verdict.mismatches++ == 0) {
			verdict.first = describe();
		}
	};
	for (uint32_t op = 0; op < UnaryOps; op++) {
		const auto unary = static_cast<UnaryOp>(op);
		for (uint32_t i = 0; i < InputCount; i++) {
			const auto input    = buffer[InputBase + i];
			const auto got      = buffer[PlaneWord(actual_side ? UnaryActualPlane(unary) : UnaryExpectedPlane(unary), i)];
			const auto expected = expected_unary(unary, i);
			verdict.checked++;
			if (!ResultsEqual(got, expected)) {
				note(op, [&] {
					return std::string(UnaryName(unary)) + "(" + Hex(input) + ") = " + Hex(got) + ", expected " +
					       Hex(expected);
				});
			}
		}
	}
	for (uint32_t op = 0; op < CompareOps; op++) {
		const auto compare = static_cast<CompareOp>(op);
		for (uint32_t order = 0; order < 2; order++) {
			for (uint32_t i = 0; i < InputCount; i++) {
				const auto got      = buffer[PlaneWord(ComparePlane(compare, order, !actual_side), i)];
				const auto expected = expected_mask(compare, order, i);
				verdict.checked += OperandCount;
				auto differ = (got ^ expected) & ((OperandCount == 32u ? 0u : (1u << OperandCount)) - 1u);
				if (differ == 0u) {
					continue;
				}
				for (uint32_t j = 0; j < OperandCount; j++) {
					if ((differ & (1u << j)) == 0u) {
						continue;
					}
					const auto x       = buffer[InputBase + i];
					const auto y       = buffer[OperandBase + j];
					const auto lhs     = order == 0 ? x : y;
					const auto rhs     = order == 0 ? y : x;
					const bool result  = (got & (1u << j)) != 0u;
					note(UnaryOps + op, [&] {
						return std::string(CompareName(compare)) + "(" + Hex(lhs) + ", " + Hex(rhs) + ") = " +
						       (result ? "true" : "false") + ", expected " + (result ? "false" : "true");
					});
				}
			}
		}
	}
	return verdict;
}

} // namespace

std::string Verdict::ByOpText() const {
	std::string text;
	for (uint32_t op = 0; op < UnaryOps + CompareOps; op++) {
		if (by_op[op] == 0u) {
			continue;
		}
		text += text.empty() ? "" : ", ";
		text += op < UnaryOps ? UnaryName(static_cast<UnaryOp>(op))
		                      : CompareName(static_cast<CompareOp>(op - UnaryOps));
		text += " " + std::to_string(by_op[op]);
	}
	return text;
}

Verdict Classify(std::span<const uint32_t> buffer) {
	return ClassifyAgainst(
	    buffer, true,
	    [&](UnaryOp op, uint32_t i) { return buffer[PlaneWord(UnaryExpectedPlane(op), i)]; },
	    [&](CompareOp op, uint32_t order, uint32_t i) {
		    return buffer[PlaneWord(ComparePlane(op, order, true), i)];
	    });
}

Verdict ClassifyReference(std::span<const uint32_t> buffer) {
	return ClassifyAgainst(
	    buffer, false, [&](UnaryOp op, uint32_t i) { return ExpectedUnary(op, buffer[InputBase + i]); },
	    [&](CompareOp op, uint32_t order, uint32_t i) {
		    uint32_t   mask = 0;
		    const auto x    = buffer[InputBase + i];
		    for (uint32_t j = 0; j < OperandCount; j++) {
			    const auto y = buffer[OperandBase + j];
			    if (order == 0 ? ExpectedCompare(op, x, y) : ExpectedCompare(op, y, x)) {
				    mask |= 1u << j;
			    }
		    }
		    return mask;
	    });
}

void ForceMismatch(std::span<uint32_t> buffer) {
	// Input 1 is -0x00000001 (Inputs()).
	if (buffer.size() >= BufferWords) {
		buffer[PlaneWord(UnaryActualPlane(UnaryOp::Rsq), 1)] = 0x7fc00000u;
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest
