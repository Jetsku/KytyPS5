#include "graphics/shader/recompiler/F32FlushSelfTest.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
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

uint8_t PassingClasses(const Verdict& verdict) {
	if (verdict.checked == 0) {
		return 0;
	}
	uint8_t classes = 0;
	for (uint32_t op = 0; op < UnaryOps; op++) {
		classes |= verdict.by_op[op] == 0u ? static_cast<uint8_t>(1u << op) : 0u;
	}
	bool compares = true;
	for (uint32_t op = UnaryOps; op < UnaryOps + CompareOps; op++) {
		compares = compares && verdict.by_op[op] == 0u;
	}
	return static_cast<uint8_t>(classes | (compares ? ClassCompare : 0u));
}

std::string ClassesText(uint8_t classes) {
	if ((classes & ClassAll) == 0u) {
		return "none";
	}
	if ((classes & ClassAll) == ClassAll) {
		return "all";
	}
	std::string text;
	for (uint32_t op = 0; op < UnaryOps; op++) {
		if ((classes & (1u << op)) != 0u) {
			text += text.empty() ? "" : ", ";
			text += UnaryName(static_cast<UnaryOp>(op));
		}
	}
	if ((classes & ClassCompare) != 0u) {
		text += text.empty() ? "compares" : ", compares";
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

uint32_t F16ToF32Bits(uint32_t half) {
	const uint32_t sign     = (half & 0x8000u) << 16u;
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	uint32_t       mantissa = half & 0x3ffu;
	if (exponent == 0x1fu) {
		return sign | 0x7f800000u | (mantissa << 13u);
	}
	if (exponent != 0u) {
		return sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
	}
	if (mantissa == 0u) {
		return sign;
	}
	// Denormal m * 2^-24: normalize.
	uint32_t e = 113u; // f32 exponent of 2^-14
	while ((mantissa & 0x400u) == 0u) {
		mantissa <<= 1u;
		e--;
	}
	return sign | (e << 23u) | ((mantissa & 0x3ffu) << 13u);
}

uint32_t F32ToF16BitsRne(uint32_t bits) {
	const uint32_t sign = (bits >> 16u) & 0x8000u;
	const uint32_t abs  = bits & 0x7fffffffu;
	if (abs >= 0x7f800000u) {
		return sign | (abs > 0x7f800000u ? (0x7e00u | ((abs >> 13u) & 0x3ffu)) : 0x7c00u);
	}
	const uint32_t exponent = abs >> 23u;
	if (exponent < 113u) {
		// f16 denormal (or zero): (1.m) * 2^(e-127) = q * 2^-24, q = mantissa24 >> (126 - e).
		if (exponent < 102u) {
			return sign; // below half the smallest denormal (f32 denormals included)
		}
		const uint32_t mantissa = (abs & 0x7fffffu) | 0x800000u;
		const uint32_t shift    = 126u - exponent;
		uint32_t       q        = mantissa >> shift;
		const uint32_t rest     = mantissa & ((1u << shift) - 1u);
		const uint32_t half     = 1u << (shift - 1u);
		if (rest > half || (rest == half && (q & 1u) != 0u)) {
			q++;
		}
		return sign | q; // q == 0x400 is the smallest normal
	}
	uint32_t       result = ((exponent - 112u) << 10u) | ((abs >> 13u) & 0x3ffu);
	const uint32_t rest   = abs & 0x1fffu;
	if (rest > 0x1000u || (rest == 0x1000u && (result & 1u) != 0u)) {
		result++;
	}
	return sign | (result >= 0x7c00u ? 0x7c00u : result);
}

std::vector<uint32_t> F16InitialBuffer() {
	std::vector<uint32_t> buffer(F16BufferWords, 0u);
	// Unpack: word k holds denormal k+1 of one sign in the low half and of the other in the high.
	for (uint32_t k = 0; k < F16Words; k++) {
		const uint32_t m    = (k % 1023u) + 1u;
		const uint32_t flip = k < 1023u ? 0u : 0x8000u;
		buffer[F16UnpackBase + k] = (m | flip) | ((m | (flip ^ 0x8000u)) << 16u);
	}
	// Pack: exact f16 denormals and the ties above them, both signs, then f32 denormals.
	uint32_t n   = 0;
	const auto f32_of = [](double v) {
		const auto f = static_cast<float>(v);
		uint32_t   b = 0;
		std::memcpy(&b, &f, sizeof(b));
		return b;
	};
	for (uint32_t m = 1; m < 1024u && n + 4u <= F16PackValues; m++) {
		const auto exact = f32_of(static_cast<double>(m) * 0x1p-24);
		const auto tie   = f32_of((static_cast<double>(m) + 0.5) * 0x1p-24);
		buffer[F16PackBase + n++] = exact;
		buffer[F16PackBase + n++] = exact | 0x80000000u;
		buffer[F16PackBase + n++] = tie;
		buffer[F16PackBase + n++] = tie | 0x80000000u;
	}
	const uint32_t extra[] = {0x33000000u, 0xb3000000u, 0x00000001u, 0x80400000u}; // 2^-25 ties to 0
	for (uint32_t i = 0; n < F16PackValues; i++) {
		buffer[F16PackBase + n++] = extra[i % 4u];
	}
	return buffer;
}

Verdict ClassifyF16(std::span<const uint32_t> buffer) {
	Verdict verdict;
	if (buffer.size() < F16BufferWords) {
		verdict.mismatches = 1;
		verdict.first      = "read-back buffer too small";
		return verdict;
	}
	const auto note = [&](uint32_t op, const auto& describe) {
		verdict.by_op[op]++;
		if (verdict.mismatches++ == 0) {
			verdict.first = describe();
		}
	};
	for (uint32_t k = 0; k < F16Words; k++) {
		const auto word = buffer[F16UnpackBase + k];
		for (uint32_t h = 0; h < 2; h++) {
			const auto half     = (word >> (16u * h)) & 0xffffu;
			const auto got      = buffer[F16UnpackOutBase + 2u * k + h];
			const auto expected = F16ToF32Bits(half);
			verdict.checked++;
			if (got != expected) {
				note(0, [&] { return "UnpackHalf2x16(" + Hex(half) + ") = " + Hex(got) + ", expected " + Hex(expected); });
			}
		}
	}
	for (uint32_t k = 0; k < F16Words; k++) {
		const auto x        = buffer[F16PackBase + 2u * k];
		const auto y        = buffer[F16PackBase + 2u * k + 1u];
		const auto got      = buffer[F16PackOutBase + k];
		const auto expected = F32ToF16BitsRne(x) | (F32ToF16BitsRne(y) << 16u);
		verdict.checked += 2;
		if (got != expected) {
			note(1, [&] {
				return "PackHalf2x16(" + Hex(x) + ", " + Hex(y) + ") = " + Hex(got) + ", expected " + Hex(expected);
			});
		}
	}
	return verdict;
}

bool StageBWanted(const DecisionInput& in) {
	if (in.force == Force::StageB) {
		return true;
	}
	return in.force != Force::StageA && (in.stage_a_classes & ClassAll) != ClassAll && in.preserve16_declared &&
	       !in.native_f16_active;
}

Outcome Decide(const DecisionInput& in) {
	const bool a_all = (in.stage_a_classes & ClassAll) == ClassAll;
	if (in.force == Force::StageB) {
		return Outcome::NoPreserve16;
	}
	if (in.force == Force::StageA || a_all) {
		return a_all ? Outcome::AllDeclared : Outcome::PerClass;
	}
	if (StageBWanted(in) && in.stage_b_ran && (in.stage_b_classes & ClassAll) == ClassAll && in.f16_passed) {
		return Outcome::NoPreserve16;
	}
	return Outcome::PerClass;
}

uint8_t OutcomeClasses(Outcome outcome, uint8_t stage_a_classes) {
	switch (outcome) {
		case Outcome::AllDeclared: return ClassAll;
		case Outcome::PerClass: return static_cast<uint8_t>(stage_a_classes & ClassAll);
		case Outcome::NoPreserve16: return static_cast<uint8_t>(ClassAll | ClassNoPreserve16);
	}
	return 0;
}

} // namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest
