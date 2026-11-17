#ifndef EMULATOR_SRC_COMMON_CPUCHECK_H_
#define EMULATOR_SRC_COMMON_CPUCHECK_H_

#include <cstddef>
#include <cstdint>

// Start-up CPU check of builds compiled for a newer x86-64 level (CMake KYTY_MARCH=x86-64-v3, or
// KYTY_X86_64_V3): on an older CPU the first such instruction would end the program with an
// illegal-instruction crash, so kyty_emulator and kyty_launcher say what is missing and exit first.
// cpuCheck.cpp is compiled for the x86-64 baseline (CMake target kyty_cpu_check, whatever
// KYTY_MARCH is) and runs from the C run-time's C initializer table (.CRT$XIB), before the C++
// static initializers of every other translation unit. Only CPUID, XGETBV and Win32 calls.
// KYTY_TEST_CPU_CHECK=fail takes the failure path on any CPU (to see the message).
namespace Kyty::CpuCheck {

// The CPUID / XGETBV words the check reads. Leaves above the CPU's maximum leaf read as 0;
// xcr0 is 0 when the OS has not enabled XSAVE (CPUID.1:ECX.OSXSAVE clear).
struct CpuIdValues {
	uint32_t max_leaf          = 0; // CPUID.0:EAX
	uint32_t leaf1_ecx         = 0; // CPUID.1:ECX
	uint32_t leaf7_ebx         = 0; // CPUID.(7,0):EBX
	uint32_t max_extended_leaf = 0; // CPUID.80000000h:EAX
	uint32_t extended1_ecx     = 0; // CPUID.80000001h:ECX
	uint64_t xcr0              = 0; // XGETBV(0)
};

// One bit per feature a level needs. OsAvxState / OsAvx512State: the CPU has the instructions
// but the OS does not save their registers (XCR0), so they cannot be used either.
enum Feature : uint32_t {
	Sse3          = 1u << 0,
	Ssse3         = 1u << 1,
	Sse41         = 1u << 2,
	Sse42         = 1u << 3,
	Popcnt        = 1u << 4,
	Cx16          = 1u << 5,
	LahfSahf      = 1u << 6,
	Avx           = 1u << 7,
	Avx2          = 1u << 8,
	Bmi1          = 1u << 9,
	Bmi2          = 1u << 10,
	Fma           = 1u << 11,
	F16c          = 1u << 12,
	Lzcnt         = 1u << 13,
	Movbe         = 1u << 14,
	OsAvxState    = 1u << 15,
	Avx512f       = 1u << 16,
	Avx512bw      = 1u << 17,
	Avx512cd      = 1u << 18,
	Avx512dq      = 1u << 19,
	Avx512vl      = 1u << 20,
	OsAvx512State = 1u << 21,
};

// The features of x86-64 level `level` (2, 3 or 4; anything lower needs nothing) that `cpu` lacks;
// 0 when the CPU (and its OS) can run code compiled for that level.
[[nodiscard]] uint32_t Missing(const CpuIdValues& cpu, int level);

// The missing features' names ("AVX2, BMI2, ...") into `out` (NUL-terminated, cut to fit);
// the characters written without the NUL.
size_t FeatureNames(uint32_t missing, wchar_t* out, size_t capacity);

// The message the programs show: what the build needs, then what this CPU lacks.
size_t Message(int level, uint32_t missing, bool test, wchar_t* out, size_t capacity);

// This CPU's values (CPUID / XGETBV).
[[nodiscard]] CpuIdValues ReadCpu();

} // namespace Kyty::CpuCheck

#endif // EMULATOR_SRC_COMMON_CPUCHECK_H_
