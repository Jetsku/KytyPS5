// The start-up CPU check (common/cpuCheck.h): which x86-64 level features a set of CPUID / XCR0
// values lacks, and the message text.
#include "common/cpuCheck.h"

#include <cstdio>
#include <cstdlib>
#include <cwchar>

namespace {

using namespace Kyty::CpuCheck;

void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "CpuCheckTests: failed: %s\n", text);
		std::abort();
	}
}

// CPUID.1:ECX of a v3 CPU: SSE3, SSSE3, FMA, CX16, SSE4.1, SSE4.2, MOVBE, POPCNT, XSAVE, OSXSAVE,
// AVX, F16C.
constexpr uint32_t V3Leaf1 = (1u << 0) | (1u << 9) | (1u << 12) | (1u << 13) | (1u << 19) | (1u << 20) |
                             (1u << 22) | (1u << 23) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29);
constexpr uint32_t V3Leaf7 = (1u << 3) | (1u << 5) | (1u << 8); // BMI1, AVX2, BMI2
constexpr uint32_t V3Ext1  = (1u << 0) | (1u << 5);              // LAHF/SAHF, LZCNT
constexpr uint32_t Avx512  = (1u << 16) | (1u << 17) | (1u << 28) | (1u << 30) | (1u << 31);

CpuIdValues V3Cpu() {
	CpuIdValues cpu;
	cpu.max_leaf          = 0x10;
	cpu.leaf1_ecx         = V3Leaf1;
	cpu.leaf7_ebx         = V3Leaf7;
	cpu.max_extended_leaf = 0x80000008u;
	cpu.extended1_ecx     = V3Ext1;
	cpu.xcr0              = 0x7; // x87, SSE, AVX state
	return cpu;
}

bool Contains(const wchar_t* text, const wchar_t* part) {
	return std::wcsstr(text, part) != nullptr;
}

} // namespace

int main() {
	// A v3-capable CPU (Haswell, Zen) runs v1..v3 code, not v4.
	const auto v3 = V3Cpu();
	Check(Missing(v3, 3) == 0, "a v3 CPU passes the v3 check");
	Check(Missing(v3, 2) == 0 && Missing(v3, 1) == 0 && Missing(v3, 0) == 0, "and the lower levels");
	Check(Missing(v3, 4) == (Avx512f | Avx512bw | Avx512cd | Avx512dq | Avx512vl | OsAvx512State),
	      "a v3 CPU lacks AVX-512 for v4");

	// No AVX2 (e.g. Ivy Bridge: AVX, F16C, no AVX2/BMI/FMA/MOVBE/LZCNT).
	auto ivy      = v3;
	ivy.leaf1_ecx &= ~((1u << 12) | (1u << 22));
	ivy.leaf7_ebx = 0;
	ivy.extended1_ecx &= ~(1u << 5);
	Check(Missing(ivy, 3) == (Avx2 | Bmi1 | Bmi2 | Fma | Lzcnt | Movbe), "Ivy Bridge lacks AVX2, BMI, FMA, LZCNT, MOVBE");
	Check(Missing(ivy, 2) == 0, "Ivy Bridge is v2");

	// AVX2 in the CPU, but the OS does not save the YMM registers (XCR0 bit 2 clear, or no OSXSAVE).
	auto no_ymm = v3;
	no_ymm.xcr0 = 0x3;
	Check(Missing(no_ymm, 3) == OsAvxState, "AVX2 without OS YMM state: only the state is reported");
	auto no_osxsave      = v3;
	no_osxsave.leaf1_ecx &= ~(1u << 27);
	no_osxsave.xcr0      = 0x7; // ignored without OSXSAVE
	Check(Missing(no_osxsave, 3) == OsAvxState, "no OSXSAVE: XCR0 is not trusted");
	Check(Missing(no_ymm, 2) == 0, "v2 needs no AVX state");

	// One instruction set missing at a time.
	auto no_bmi2      = v3;
	no_bmi2.leaf7_ebx &= ~(1u << 8);
	Check(Missing(no_bmi2, 3) == Bmi2, "no BMI2");
	auto no_fma      = v3;
	no_fma.leaf1_ecx &= ~(1u << 12);
	Check(Missing(no_fma, 3) == Fma, "no FMA");
	auto no_lzcnt          = v3;
	no_lzcnt.extended1_ecx = 1;
	Check(Missing(no_lzcnt, 3) == Lzcnt, "no LZCNT");

	// A CPU too old to have leaf 7 or the extended leaves (leaves above the maximum are not read).
	auto old              = v3;
	old.max_leaf          = 6;
	old.max_extended_leaf = 0x80000000u;
	Check(Missing(old, 3) == (Avx2 | Bmi1 | Bmi2 | Lzcnt | LahfSahf),
	      "leaf 7 and 80000001h above the maximum count as absent");
	Check((Missing(old, 2) & LahfSahf) != 0, "LAHF/SAHF is in 80000001h");
	CpuIdValues nothing;
	Check((Missing(nothing, 2) & (Sse3 | Popcnt | Cx16)) == (Sse3 | Popcnt | Cx16), "an empty CPUID lacks everything");

	// v4: AVX-512 in the CPU and in XCR0.
	auto v4      = v3;
	v4.leaf7_ebx |= Avx512;
	v4.xcr0      = 0xE7;
	Check(Missing(v4, 4) == 0, "a v4 CPU passes v4");
	auto v4_no_zmm = v4;
	v4_no_zmm.xcr0 = 0x7;
	Check(Missing(v4_no_zmm, 4) == OsAvx512State, "AVX-512 without OS ZMM state");

	// Names and message.
	wchar_t names[256];
	FeatureNames(Avx2 | Bmi2 | Fma, names, 256);
	Check(std::wcscmp(names, L"AVX2, BMI2, FMA") == 0, "names in message order");
	FeatureNames(0, names, 256);
	Check(names[0] == L'\0', "no names for nothing missing");
	wchar_t small[8];
	const auto cut = FeatureNames(Avx2 | Bmi2 | Fma, small, 8);
	Check(cut == 7 && small[7] == L'\0', "a short buffer is cut and terminated");

	wchar_t message[1024];
	Message(3, Avx2 | Fma, false, message, 1024);
	Check(Contains(message, L"This build needs a CPU with AVX2, BMI2 and FMA (Intel Core 4th generation / AMD Ryzen or newer"),
	      "the message says what the build needs");
	Check(Contains(message, L"lacks: AVX2, FMA"), "the message names what is missing");
	Check(!Contains(message, L"KYTY_TEST_CPU_CHECK"), "no test note on a real failure");
	Message(3, 0, true, message, 1024);
	Check(Contains(message, L"KYTY_TEST_CPU_CHECK=fail"), "the test failure says so");

	// This CPU: the values are read (every machine that runs these tests has leaf 1).
	const auto here = ReadCpu();
	Check(here.max_leaf >= 1, "CPUID leaf 1 is there");
	std::printf("CpuCheckTests: ok (this CPU lacks for v3: 0x%x)\n", Missing(here, 3));
	return 0;
}
