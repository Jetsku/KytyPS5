// The pure parts of the KYTY_F32_FLUSH_HOST start-up self-test (recompiler/F32FlushSelfTest.h):
// inputs, operands, buffer layout, the expected-value model, the NaN rule and the classification.
// The GPU run of the job is CodegenF32FlushSelfTest (shader_recompiler_compute_tests --codegen-only).
#include "graphics/shader/recompiler/F32FlushSelfTest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

namespace {

namespace T = Libs::Graphics::ShaderRecompiler::FlushSelfTest;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "F32FlushSelfTestTests: failed: %s\n", message);
		std::abort();
	}
}

void TestInputsAndOperands() {
	const auto inputs = T::Inputs();
	Check(inputs.size() == T::InputCount, "input count");
	std::set<uint32_t> distinct(inputs.begin(), inputs.end());
	Check(distinct.size() == inputs.size(), "inputs repeat");
	uint32_t negative = 0;
	for (uint32_t i = 0; i < inputs.size(); i++) {
		Check(T::IsDenormalF32(inputs[i]), "an input is not a denormal");
		negative += (inputs[i] & 0x80000000u) != 0u ? 1u : 0u;
		if (i % 2u == 1u) {
			Check(inputs[i] == (inputs[i - 1] | 0x80000000u), "an input pair is not one magnitude with both signs");
		}
	}
	Check(negative == T::InputCount / 2u, "signs unbalanced");
	Check(inputs[0] == 0x00000001u && inputs[1] == 0x80000001u, "smallest denormals first");
	Check(inputs[2] == 0x007fffffu && inputs[3] == 0x807fffffu, "largest denormals second");
	// A spread over every magnitude bit position.
	std::set<uint32_t> tops;
	for (const auto bits: inputs) {
		uint32_t top = 0;
		for (uint32_t m = bits & 0x007fffffu; m > 1u; m >>= 1u) {
			top++;
		}
		tops.insert(top);
	}
	Check(tops.size() == 23u, "inputs miss a magnitude range");
	const auto again = T::Inputs();
	Check(again == inputs, "inputs not deterministic");

	const auto operands = T::Operands();
	const auto has      = [&](uint32_t bits) {
		for (const auto value: operands) {
			if (value == bits) {
				return true;
			}
		}
		return false;
	};
	Check(has(0x00000000u) && has(0x80000000u), "operands: zeros of both signs");
	Check(has(0x00000001u) && has(0x807fffffu), "operands: denormals");
	Check(has(0x00800000u) && has(0x3f800000u), "operands: normals");
	Check(has(0x7f800000u) && has(0xff800000u), "operands: infinities");
	Check(has(0x7fc00000u) && has(0x7f800001u), "operands: NaNs");

	const auto buffer = T::InitialBuffer();
	Check(buffer.size() == T::BufferWords, "buffer size");
	Check(buffer[T::InputBase + 5] == inputs[5] && buffer[T::OperandBase + 3] == operands[3], "buffer layout");
	Check(T::PlaneWord(T::Planes - 1, T::InputCount - 1) == T::BufferWords - 1, "plane layout");
	Check(T::ComparePlane(T::CompareOp::UnordGreaterThanEqual, 1, true) == T::Planes - 1, "compare planes");
}

void TestModel() {
	using U = T::UnaryOp;
	Check(T::FlushF32(0x80000001u) == 0x80000000u && T::FlushF32(0x007fffffu) == 0u, "flush keeps the sign");
	Check(T::FlushF32(0x00800000u) == 0x00800000u && T::FlushF32(0x7fc00000u) == 0x7fc00000u,
	      "flush leaves normals and NaNs");
	Check(T::ExpectedUnary(U::Rcp, 0x00000001u) == 0x7f800000u, "rcp(+denormal) = +inf");
	Check(T::ExpectedUnary(U::Rcp, 0x80000001u) == 0xff800000u, "rcp(-denormal) = -inf");
	Check(T::ExpectedUnary(U::Rsq, 0x807fffffu) == 0xff800000u, "rsq(-denormal) = -inf");
	Check(T::ExpectedUnary(U::Sqrt, 0x807fffffu) == 0x80000000u, "sqrt(-denormal) = -0");
	Check(T::ExpectedUnary(U::Sqrt, 0x00000001u) == 0x00000000u, "sqrt(+denormal) = +0");
	Check(T::ExpectedUnary(U::Exp, 0x80400000u) == 0x3f800000u, "exp(denormal) = 1");
	Check(T::ExpectedUnary(U::Log, 0x80400000u) == 0xff800000u, "log(-denormal) = -inf");

	using C = T::CompareOp;
	Check(!T::ExpectedCompare(C::OrdLessThan, 0x00000001u, 0x00000002u), "1 < 2 denormals flush to 0 < 0");
	Check(T::ExpectedCompare(C::OrdEqual, 0x80000001u, 0x00000000u), "-denormal == +0");
	Check(T::ExpectedCompare(C::OrdEqual, 0x00400000u, 0x80000000u), "+denormal == -0");
	Check(!T::ExpectedCompare(C::OrdNotEqual, 0x00400000u, 0x807fffffu), "denormals of both signs are equal");
	Check(T::ExpectedCompare(C::OrdLessThan, 0x807fffffu, 0x00800000u), "-denormal < smallest normal");
	Check(T::ExpectedCompare(C::OrdGreaterThan, 0x00000001u, 0x80800000u), "+denormal > -smallest normal");
	Check(!T::ExpectedCompare(C::OrdEqual, 0x7fc00000u, 0x7fc00000u), "NaN is unordered");
	Check(T::ExpectedCompare(C::UnordEqual, 0x00000001u, 0x7f800001u), "unordered equal with a NaN");
	Check(T::ExpectedCompare(C::UnordNotEqual, 0x00000001u, 0xffc00000u), "unordered not equal with a NaN");
	Check(!T::ExpectedCompare(C::OrdNotEqual, 0x00000001u, 0xffc00000u), "ordered not equal with a NaN");
	Check(T::ExpectedCompare(C::OrdLessThanEqual, 0xff800000u, 0x80000001u), "-inf <= -denormal");
	Check(T::ExpectedCompare(C::UnordGreaterThanEqual, 0x7f7fffffu, 0x007fffffu), "FLT_MAX >= denormal");
	Check(!T::ExpectedCompare(C::UnordLessThan, 0x7f800000u, 0x00000001u), "+inf < denormal is false");

	Check(T::ResultsEqual(0x7fc00000u, 0xffc00001u), "any NaN equals any NaN");
	Check(!T::ResultsEqual(0x7fc00000u, 0xff800000u), "a NaN is not -inf");
	Check(!T::ResultsEqual(0x80000000u, 0x00000000u), "-0 is not +0");
}

// A read-back as a host that flushes would give it: both sides from the model.
std::vector<uint32_t> PerfectBuffer() {
	auto       buffer   = T::InitialBuffer();
	const auto operands = T::Operands();
	for (uint32_t i = 0; i < T::InputCount; i++) {
		const auto x = buffer[T::InputBase + i];
		for (uint32_t op = 0; op < T::UnaryOps; op++) {
			const auto unary = static_cast<T::UnaryOp>(op);
			const auto value = T::ExpectedUnary(unary, x);
			buffer[T::PlaneWord(T::UnaryActualPlane(unary), i)]   = value;
			buffer[T::PlaneWord(T::UnaryExpectedPlane(unary), i)] = value;
		}
		for (uint32_t op = 0; op < T::CompareOps; op++) {
			const auto compare = static_cast<T::CompareOp>(op);
			for (uint32_t order = 0; order < 2; order++) {
				uint32_t mask = 0;
				for (uint32_t j = 0; j < T::OperandCount; j++) {
					const bool result = order == 0 ? T::ExpectedCompare(compare, x, operands[j])
					                               : T::ExpectedCompare(compare, operands[j], x);
					mask |= result ? 1u << j : 0u;
				}
				buffer[T::PlaneWord(T::ComparePlane(compare, order, false), i)] = mask;
				buffer[T::PlaneWord(T::ComparePlane(compare, order, true), i)]  = mask;
			}
		}
	}
	return buffer;
}

void TestClassification() {
	constexpr uint64_t total = uint64_t {T::UnaryOps} * T::InputCount +
	                           uint64_t {T::CompareOps} * 2u * T::OperandCount * T::InputCount;
	auto buffer = PerfectBuffer();
	auto pass   = T::Classify(buffer);
	Check(pass.Passed() && pass.checked == total && pass.first.empty(), "a flushing host passes");
	Check(T::ClassifyReference(buffer).Passed(), "the model reference passes");

	// What a host that does not flush gives for rsq(-denormal) and a NaN with another payload.
	auto nan = buffer;
	nan[T::PlaneWord(T::UnaryActualPlane(T::UnaryOp::Rsq), 2)] = 0xffc00000u;
	nan[T::PlaneWord(T::UnaryExpectedPlane(T::UnaryOp::Rsq), 2)] = 0x7fc00001u;
	Check(T::Classify(nan).Passed(), "NaN against NaN matches");

	auto forced = buffer;
	T::ForceMismatch(forced);
	const auto fail = T::Classify(forced);
	Check(!fail.Passed() && fail.mismatches == 1u, "the forced mismatch is one result");
	Check(fail.first == "rsq(0x80000001) = 0x7fc00000, expected 0xff800000", "the forced mismatch text");
	Check(fail.by_op[static_cast<uint32_t>(T::UnaryOp::Rsq)] == 1u && fail.ByOpText() == "rsq 1",
	      "mismatches by op");

	// An unflushed compare: x = +smallest denormal (input 0) < operand 4 (0x00000002) is true.
	auto compare = buffer;
	compare[T::PlaneWord(T::ComparePlane(T::CompareOp::OrdLessThan, 0, false), 0)] |= 1u << 4u;
	const auto cmp = T::Classify(compare);
	Check(cmp.ByOpText() == "OpFOrdLessThan 1", "compare mismatches by op");
	Check(T::PassingClasses(pass) == T::ClassAll && T::ClassesText(T::ClassAll) == "all", "all classes pass");
	Check(T::PassingClasses(fail) == (T::ClassAll & ~2u) &&
	          T::ClassesText(T::PassingClasses(fail)) == "rcp, sqrt, exp, log, compares",
	      "a rsq mismatch removes the rsq class");
	Check(T::PassingClasses(cmp) == 31u && T::ClassesText(31u) == "rcp, rsq, sqrt, exp, log",
	      "one compare mismatch removes the compare class");
	Check(T::PassingClasses(T::Verdict {}) == 0u && T::ClassesText(0u) == "none", "nothing checked, no class");
	Check(cmp.mismatches == 1u &&
	          cmp.first == "OpFOrdLessThan(0x00000001, 0x00000002) = true, expected false",
	      "a compare mismatch names opcode and operands");
	// Operand order 1: operand op x.
	auto swapped = buffer;
	swapped[T::PlaneWord(T::ComparePlane(T::CompareOp::UnordEqual, 1, false), 0)] ^= 1u << 4u;
	const auto swap = T::Classify(swapped);
	Check(swap.first == "OpFUnordEqual(0x00000002, 0x00000001) = false, expected true",
	      "operand order 1 is operand op x");
	// The first mismatch is the first in op order; all are counted.
	auto two = forced;
	two[T::PlaneWord(T::UnaryActualPlane(T::UnaryOp::Rcp), 3)] = 0x7f7fffffu;
	const auto both = T::Classify(two);
	Check(both.mismatches == 2u && both.first.rfind("rcp(0x807fffff)", 0) == 0, "first mismatch order");
	// A wrong reference is a reference mismatch, not a result mismatch.
	auto reference = buffer;
	reference[T::PlaneWord(T::UnaryExpectedPlane(T::UnaryOp::Sqrt), 1)] = 0u;
	Check(!T::ClassifyReference(reference).Passed(), "a reference against the model");
	Check(T::Classify(std::vector<uint32_t>(10, 0u)).mismatches == 1u, "a short read-back fails");
}

void TestF16Block() {
	// Model.
	Check(T::F16ToF32Bits(0x0001u) == 0x33800000u, "f16 smallest denormal = 2^-24");
	Check(T::F16ToF32Bits(0x83ffu) == 0xb87fc000u, "f16 largest negative denormal");
	Check(T::F16ToF32Bits(0x0400u) == 0x38800000u && T::F16ToF32Bits(0x3c00u) == 0x3f800000u, "f16 normals");
	Check(T::F16ToF32Bits(0x8000u) == 0x80000000u && T::F16ToF32Bits(0x7c00u) == 0x7f800000u, "f16 -0 and inf");
	for (uint32_t h = 0; h < 0x7c00u; h++) {
		Check(T::F32ToF16BitsRne(T::F16ToF32Bits(h)) == h, "f16 -> f32 -> f16 round trip");
		Check(T::F32ToF16BitsRne(T::F16ToF32Bits(h | 0x8000u)) == (h | 0x8000u), "negative round trip");
	}
	Check(T::F32ToF16BitsRne(0x33000000u) == 0u, "2^-25 ties to even (0)");
	Check(T::F32ToF16BitsRne(0x33c00000u) == 2u, "1.5 * 2^-24 ties to even (2)");
	Check(T::F32ToF16BitsRne(0x34200000u) == 2u, "2.5 * 2^-24 ties to even (2)");
	Check(T::F32ToF16BitsRne(0x80000001u) == 0x8000u && T::F32ToF16BitsRne(0x007fffffu) == 0u,
	      "f32 denormals give signed zeros");
	Check(T::F32ToF16BitsRne(0x387fe000u) == 0x0400u, "1023.75 * 2^-24 rounds to the smallest normal");
	Check(T::F32ToF16BitsRne(0x3f801000u) == 0x3c00u && T::F32ToF16BitsRne(0x3f803000u) == 0x3c02u,
	      "normal ties to even");
	Check(T::F32ToF16BitsRne(0x47800000u) == 0x7c00u && T::F32ToF16BitsRne(0x7fc00000u) == 0x7e00u,
	      "overflow and NaN");

	// Inputs: every f16 denormal of both signs appears in both halves; pack inputs cover exact
	// values, ties and f32 denormals.
	const auto buffer = T::F16InitialBuffer();
	Check(buffer.size() == T::F16BufferWords, "f16 buffer size");
	std::set<uint32_t> low, high;
	for (uint32_t k = 0; k < T::F16Words; k++) {
		low.insert(buffer[T::F16UnpackBase + k] & 0xffffu);
		high.insert(buffer[T::F16UnpackBase + k] >> 16u);
	}
	Check(low.size() == 2046u && high.size() == 2046u, "every f16 denormal in each half");
	uint32_t ties = 0, f32_denormals = 0;
	for (uint32_t i = 0; i < T::F16PackValues; i++) {
		const auto v = buffer[T::F16PackBase + i];
		ties += (v & 0x7fffffffu) >= 0x33800000u && T::F32ToF16BitsRne(v) != T::F32ToF16BitsRne(v - 1u) ? 1u : 0u;
		f32_denormals += T::IsDenormalF32(v) ? 1u : 0u;
	}
	Check(ties > 1000u && f32_denormals == 2u, "pack inputs: ties and f32 denormals");

	// Classification.
	auto perfect = buffer;
	for (uint32_t k = 0; k < T::F16Words; k++) {
		const auto word = perfect[T::F16UnpackBase + k];
		perfect[T::F16UnpackOutBase + 2u * k]      = T::F16ToF32Bits(word & 0xffffu);
		perfect[T::F16UnpackOutBase + 2u * k + 1u] = T::F16ToF32Bits(word >> 16u);
		perfect[T::F16PackOutBase + k] = T::F32ToF16BitsRne(perfect[T::F16PackBase + 2u * k]) |
		                                 (T::F32ToF16BitsRne(perfect[T::F16PackBase + 2u * k + 1u]) << 16u);
	}
	const auto pass = T::ClassifyF16(perfect);
	Check(pass.Passed() && pass.checked == 4u * T::F16Words, "a preserving host passes the f16 block");
	auto flushed = perfect;
	flushed[T::F16UnpackOutBase + 0] = 0u; // UnpackHalf2x16(0x0001) flushed to +0
	const auto fail = T::ClassifyF16(flushed);
	Check(fail.mismatches == 1u && fail.by_op[0] == 1u &&
	          fail.first == "UnpackHalf2x16(0x00000001) = 0x00000000, expected 0x33800000",
	      "a flushed f16 denormal is a mismatch");
}

void TestDecision() {
	using F = T::Force;
	using O = T::Outcome;
	T::DecisionInput in;
	in.preserve16_declared = true;
	in.stage_a_classes     = T::ClassAll;
	Check(!T::StageBWanted(in) && T::Decide(in) == O::AllDeclared, "stage A passes: modes as declared");
	in.stage_a_classes = 30u; // rsq, sqrt, exp, log
	Check(T::StageBWanted(in), "a failing class asks for stage B");
	Check(T::Decide(in) == O::PerClass, "stage B not run: per class");
	in.stage_b_ran     = true;
	in.stage_b_classes = T::ClassAll;
	in.f16_passed      = true;
	Check(T::Decide(in) == O::NoPreserve16, "stage B and the f16 block pass: no DenormPreserve 16");
	Check(T::OutcomeClasses(O::NoPreserve16, 30u) == (T::ClassAll | T::ClassNoPreserve16) &&
	          T::OutcomeClasses(O::PerClass, 30u) == 30u && T::OutcomeClasses(O::AllDeclared, 30u) == T::ClassAll,
	      "outcome classes");
	in.f16_passed = false;
	Check(T::Decide(in) == O::PerClass, "an f16 mismatch keeps stage A");
	in.f16_passed      = true;
	in.stage_b_classes = 31u;
	Check(T::Decide(in) == O::PerClass, "a stage B mismatch keeps stage A");
	in.stage_b_classes   = T::ClassAll;
	in.native_f16_active = true;
	Check(!T::StageBWanted(in) && T::Decide(in) == O::PerClass, "native f16 refuses stage B");
	in.native_f16_active   = false;
	in.preserve16_declared = false;
	Check(!T::StageBWanted(in) && T::Decide(in) == O::PerClass, "nothing to drop without DenormPreserve 16");
	in.preserve16_declared = true;
	in.force               = F::StageA;
	Check(!T::StageBWanted(in) && T::Decide(in) == O::PerClass, "forced stage A");
	in.force           = F::StageB;
	in.stage_b_classes = 0u;
	in.f16_passed      = false;
	Check(T::StageBWanted(in) && T::Decide(in) == O::NoPreserve16, "forced stage B");
	in.force           = F::Fail;
	in.stage_b_classes = 29u; // the forced rsq mismatch in stage B too
	in.f16_passed      = true;
	Check(T::Decide(in) == O::PerClass, "forced mismatch: per class");
}

} // namespace

int main() {
	TestInputsAndOperands();
	TestModel();
	TestClassification();
	TestF16Block();
	TestDecision();
	std::printf("F32FlushSelfTestTests: ok (%u inputs, %u operands, %u buffer words)\n", T::InputCount,
	            T::OperandCount, T::BufferWords);
	return 0;
}
