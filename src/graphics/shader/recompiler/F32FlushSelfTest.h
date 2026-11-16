#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_F32FLUSHSELFTEST_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_F32FLUSHSELFTEST_H_

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Start-up GPU self-test of KYTY_F32_FLUSH_HOST=nvidia (CodegenOptions::f32_flush_host). In a
// program without f64 the emitter then leaves the flush of f32 denormal inputs of V_RCP/RSQ/SQRT/
// EXP/LOG_F32 and of two-register f32 compares to the host, which an RTX 3090 does in its default
// f32 mode without declaring it (shaderDenormFlushToZeroFloat32 false). The device layer runs one
// compute dispatch on the real device before any guest program is translated and keeps the switch
// only when every result matches (presentation/window/f32FlushSelfTest.cpp).
//
// The module (EmitModule, backend/spirv/spirvEmitterFlushSelfTest.cpp) is built by the emitter:
// the float-control execution modes of a program without f64 (DeclareFloatControls) and the
// instruction forms the emitter makes on the host-flush path (EmitFPRecip32 = OpFDiv 1.0/x,
// EmitFPRecipSqrt32 = InverseSqrt, EmitFPSqrt = Sqrt, EmitFPExp2 = Exp2, EmitFPLog2 = Log2, and the
// twelve OpFOrd*/OpFUnord* compares of EmitFloatCompare32 on two dynamic operands). Each
// invocation reads one denormal x from the buffer and computes every op twice: on x (the form a
// guest program gets) and on a copy of x flushed in the shader by integer ops (the zero of the
// same sign), the expected result. Compares run x against each operand of Operands() in both
// operand orders, the operand flushed the same way on the expected side.
//
// Buffer (u32 words, one storage buffer): [InputBase, +InputCount) inputs,
// [OperandBase, +OperandCount) compare operands, then Planes planes of InputCount words: per unary
// op the actual and the expected result bits; per compare op and operand order the actual and
// the expected results as a mask (bit j = operand j).
//
// Results compare bit for bit, except that any NaN equals any NaN (sign and payload are not
// compared: the guest's NaN payloads are not what the switch is about). For these inputs no
// expected result is a NaN (the ops of a signed zero give +-inf, +-0, 1.0 or -inf), so a NaN
// result of an op on a negative denormal (an unflushed rsq/sqrt/log) is a mismatch.
namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest {

inline constexpr uint32_t InputCount    = 4096; // 2048 magnitudes, each with both signs
inline constexpr uint32_t OperandCount  = 24;   // at most 32: one mask bit each
inline constexpr uint32_t UnaryOps      = 5;
inline constexpr uint32_t CompareOps    = 12;
inline constexpr uint32_t WorkgroupSize = 64;
inline constexpr uint32_t Planes        = 2u * UnaryOps + CompareOps * 2u * 2u;
inline constexpr uint32_t InputBase     = 0;
inline constexpr uint32_t OperandBase   = InputBase + InputCount;
inline constexpr uint32_t OutputBase    = OperandBase + OperandCount;
inline constexpr uint32_t BufferWords   = OutputBase + Planes * InputCount;
static_assert(InputCount % WorkgroupSize == 0 && OperandCount <= 32);

enum class UnaryOp : uint32_t { Rcp, Rsq, Sqrt, Exp, Log };
// The order of the emitter's compare forms (spirvEmitterInstructions.h).
enum class CompareOp : uint32_t {
	OrdEqual,
	UnordEqual,
	OrdNotEqual,
	UnordNotEqual,
	OrdLessThan,
	UnordLessThan,
	OrdGreaterThan,
	UnordGreaterThan,
	OrdLessThanEqual,
	UnordLessThanEqual,
	OrdGreaterThanEqual,
	UnordGreaterThanEqual,
};

[[nodiscard]] const char* UnaryName(UnaryOp op);
[[nodiscard]] const char* CompareName(CompareOp op); // the SPIR-V opcode name

[[nodiscard]] constexpr uint32_t UnaryActualPlane(UnaryOp op) {
	return static_cast<uint32_t>(op);
}
[[nodiscard]] constexpr uint32_t UnaryExpectedPlane(UnaryOp op) {
	return UnaryOps + static_cast<uint32_t>(op);
}
// order 0: x op operand; order 1: operand op x.
[[nodiscard]] constexpr uint32_t ComparePlane(CompareOp op, uint32_t order, bool expected) {
	return 2u * UnaryOps + (static_cast<uint32_t>(op) * 2u + order) * 2u + (expected ? 1u : 0u);
}
[[nodiscard]] constexpr uint32_t PlaneWord(uint32_t plane, uint32_t index) {
	return OutputBase + plane * InputCount + index;
}

// The zero of the same sign for a denormal (or zero), the bits unchanged otherwise.
[[nodiscard]] constexpr uint32_t FlushF32(uint32_t bits) {
	return (bits & 0x7fffffffu) < 0x00800000u ? bits & 0x80000000u : bits;
}
[[nodiscard]] constexpr bool IsDenormalF32(uint32_t bits) {
	return (bits & 0x7fffffffu) != 0u && (bits & 0x7fffffffu) < 0x00800000u;
}
[[nodiscard]] constexpr bool IsNanF32(uint32_t bits) {
	return (bits & 0x7fffffffu) > 0x7f800000u;
}
// The NaN rule above.
[[nodiscard]] constexpr bool ResultsEqual(uint32_t actual, uint32_t expected) {
	return actual == expected || (IsNanF32(actual) && IsNanF32(expected));
}

// InputCount denormals: index 2k and 2k+1 are one magnitude with sign + and -; the first pairs
// are the smallest (0x00000001) and the largest (0x007fffff) denormal, then powers of two and
// their neighbours, then a deterministic log-uniform spread.
[[nodiscard]] std::vector<uint32_t> Inputs();
// Zeros of both signs, denormals, the smallest normals, +-1, +-FLT_MAX, +-inf and NaNs.
[[nodiscard]] std::array<uint32_t, OperandCount> Operands();
// The buffer before the dispatch (inputs, operands, zeroed planes).
[[nodiscard]] std::vector<uint32_t> InitialBuffer();

// The expected-value model (IEEE results of the op on the flushed operands; only for zero and
// denormal inputs). The self-test compares against the shader's own expected results; the model
// documents them and the tests check the shader's against it.
[[nodiscard]] uint32_t ExpectedUnary(UnaryOp op, uint32_t input_bits);
[[nodiscard]] bool     ExpectedCompare(CompareOp op, uint32_t lhs_bits, uint32_t rhs_bits);

struct Verdict {
	uint64_t    checked    = 0; // results compared (unary results + single compare results)
	uint64_t    mismatches = 0;
	std::string first; // the first mismatch ("rsq(0x80000001) = 0x7fc00000, expected 0xff800000")
	// Mismatches by op: the UnaryOps unary ops, then the CompareOps compares (both orders).
	std::array<uint64_t, UnaryOps + CompareOps> by_op {};
	// "rcp 2046, OpFOrdLessThan 12": the ops with mismatches.
	[[nodiscard]] std::string ByOpText() const;
	[[nodiscard]] bool Passed() const { return checked != 0 && mismatches == 0; }
};
// The op classes whose results all matched, as CodegenOptions::f32_flush_host_classes bits
// (F32FlushClassRcp/Rsq/Sqrt/Exp/Log = 1 << UnaryOp, F32FlushClassCompare = 32 for all twelve
// compares); 0 for a verdict that checked nothing.
inline constexpr uint8_t ClassCompare = 32u;
inline constexpr uint8_t ClassAll     = 63u;
[[nodiscard]] uint8_t     PassingClasses(const Verdict& verdict);
// "rsq, sqrt, exp, log" ("none" for 0, "all" for ClassAll).
[[nodiscard]] std::string ClassesText(uint8_t classes);
// Actual results against the shader's expected results.
[[nodiscard]] Verdict Classify(std::span<const uint32_t> buffer);
// The shader's expected results against the model (a check of the reference itself).
[[nodiscard]] Verdict ClassifyReference(std::span<const uint32_t> buffer);
// KYTY_F32_FLUSH_SELFTEST=fail: replaces one actual result after the read-back with what an
// unflushing host gives (rsq(-smallest denormal) = NaN), so the failure path runs on any device.
void ForceMismatch(std::span<uint32_t> buffer);

// The self-test module (SPIR-V words; local size WorkgroupSize, one storage buffer at set 0
// binding 0 of BufferWords words, dispatch InputCount / WorkgroupSize groups). Empty, with
// `error` set, when the emitter would not make the host-flush forms (f32_flush_host off, or a
// shader-side flush emitted): such a module would test something else.
[[nodiscard]] std::vector<uint32_t> EmitModule(std::string* error);

} // namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_F32FLUSHSELFTEST_H_
