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

} // namespace

int main() {
	TestInputsAndOperands();
	TestModel();
	TestClassification();
	std::printf("F32FlushSelfTestTests: ok (%u inputs, %u operands, %u buffer words)\n", T::InputCount,
	            T::OperandCount, T::BufferWords);
	return 0;
}
