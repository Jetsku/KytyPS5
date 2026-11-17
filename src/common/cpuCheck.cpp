// Compiled for the x86-64 baseline in every build (CMake target kyty_cpu_check): it must run on
// the CPUs it rejects. See cpuCheck.h.
#include "common/cpuCheck.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#pragma comment(lib, "user32")
#endif

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#define KYTY_CPU_CHECK_X86 1
#endif

#ifndef KYTY_CPU_CHECK_LEVEL
#define KYTY_CPU_CHECK_LEVEL 0
#endif

namespace Kyty::CpuCheck {

namespace {

constexpr uint64_t XcrSse       = 1u << 1;
constexpr uint64_t XcrYmm       = 1u << 2;
constexpr uint64_t XcrAvx512    = (1u << 5) | (1u << 6) | (1u << 7); // opmask, ZMM 0-15 upper, ZMM 16-31
constexpr uint32_t Ecx1OsXsave  = 1u << 27;
constexpr uint32_t Ecx1Avx      = 1u << 28;
constexpr uint32_t Ebx7Avx512f  = 1u << 16;

struct Bit {
	Feature  feature;
	uint32_t mask;
};

// CPUID.1:ECX
constexpr Bit Leaf1[] = {
    {Sse3, 1u << 0},   {Ssse3, 1u << 9},   {Fma, 1u << 12},    {Cx16, 1u << 13}, {Sse41, 1u << 19},
    {Sse42, 1u << 20}, {Movbe, 1u << 22}, {Popcnt, 1u << 23}, {Avx, Ecx1Avx},   {F16c, 1u << 29},
};
// CPUID.(7,0):EBX
constexpr Bit Leaf7[] = {
    {Bmi1, 1u << 3},      {Avx2, 1u << 5},       {Bmi2, 1u << 8},       {Avx512f, Ebx7Avx512f},
    {Avx512dq, 1u << 17}, {Avx512cd, 1u << 28}, {Avx512bw, 1u << 30}, {Avx512vl, 1u << 31},
};
// CPUID.80000001h:ECX
constexpr Bit Extended1[] = {{LahfSahf, 1u << 0}, {Lzcnt, 1u << 5}};

constexpr uint32_t Level2 = Sse3 | Ssse3 | Sse41 | Sse42 | Popcnt | Cx16 | LahfSahf;
constexpr uint32_t Level3 = Level2 | Avx | Avx2 | Bmi1 | Bmi2 | Fma | F16c | Lzcnt | Movbe | OsAvxState;
constexpr uint32_t Level4 = Level3 | Avx512f | Avx512bw | Avx512cd | Avx512dq | Avx512vl | OsAvx512State;

// Order of the names in messages.
constexpr struct {
	Feature        feature;
	const wchar_t* name;
} Names[] = {
    {Avx2, L"AVX2"},
    {Bmi2, L"BMI2"},
    {Fma, L"FMA"},
    {Avx, L"AVX"},
    {Bmi1, L"BMI1"},
    {F16c, L"F16C"},
    {Lzcnt, L"LZCNT"},
    {Movbe, L"MOVBE"},
    {OsAvxState, L"AVX register state enabled by the operating system"},
    {Sse3, L"SSE3"},
    {Ssse3, L"SSSE3"},
    {Sse41, L"SSE4.1"},
    {Sse42, L"SSE4.2"},
    {Popcnt, L"POPCNT"},
    {Cx16, L"CMPXCHG16B"},
    {LahfSahf, L"LAHF/SAHF"},
    {Avx512f, L"AVX512F"},
    {Avx512bw, L"AVX512BW"},
    {Avx512cd, L"AVX512CD"},
    {Avx512dq, L"AVX512DQ"},
    {Avx512vl, L"AVX512VL"},
    {OsAvx512State, L"AVX-512 register state enabled by the operating system"},
};

uint32_t Present(uint32_t word, const Bit* bits, size_t count) {
	uint32_t present = 0;
	for (size_t i = 0; i < count; i++) {
		if ((word & bits[i].mask) != 0) {
			present |= bits[i].feature;
		}
	}
	return present;
}

// Appends text; keeps room for the NUL.
struct Writer {
	wchar_t* out;
	size_t   capacity;
	size_t   size = 0;

	void Add(const wchar_t* text) {
		for (; *text != L'\0'; text++) {
			if (size + 1 >= capacity) {
				break;
			}
			out[size++] = *text;
		}
		if (capacity != 0) {
			out[size] = L'\0';
		}
	}
};

} // namespace

uint32_t Missing(const CpuIdValues& cpu, int level) {
	const uint32_t wanted = level >= 4 ? Level4 : level == 3 ? Level3 : level == 2 ? Level2 : 0;
	if (wanted == 0) {
		return 0;
	}
	uint32_t present = 0;
	if (cpu.max_leaf >= 1) {
		present |= Present(cpu.leaf1_ecx, Leaf1, sizeof(Leaf1) / sizeof(Leaf1[0]));
	}
	if (cpu.max_leaf >= 7) {
		present |= Present(cpu.leaf7_ebx, Leaf7, sizeof(Leaf7) / sizeof(Leaf7[0]));
	}
	if (cpu.max_extended_leaf >= 0x80000001u) {
		present |= Present(cpu.extended1_ecx, Extended1, sizeof(Extended1) / sizeof(Extended1[0]));
	}
	// The YMM (and ZMM) registers are usable only when the OS saves them on a context switch.
	const bool os_xsave = cpu.max_leaf >= 1 && (cpu.leaf1_ecx & Ecx1OsXsave) != 0;
	if (os_xsave && (cpu.xcr0 & (XcrSse | XcrYmm)) == (XcrSse | XcrYmm)) {
		present |= OsAvxState;
		if ((cpu.xcr0 & XcrAvx512) == XcrAvx512) {
			present |= OsAvx512State;
		}
	}
	uint32_t missing = wanted & ~present;
	// A CPU with the instructions whose OS does not enable their state: report the state alone.
	if ((missing & OsAvxState) != 0 && (present & Avx) != 0) {
		missing &= ~(Avx | Avx2 | Fma | F16c);
	}
	if ((missing & OsAvx512State) != 0 && (present & Avx512f) != 0) {
		missing &= ~(Avx512f | Avx512bw | Avx512cd | Avx512dq | Avx512vl);
	}
	return missing;
}

size_t FeatureNames(uint32_t missing, wchar_t* out, size_t capacity) {
	Writer writer {out, capacity};
	writer.Add(L"");
	bool first = true;
	for (const auto& entry: Names) {
		if ((missing & entry.feature) == 0) {
			continue;
		}
		if (!first) {
			writer.Add(L", ");
		}
		writer.Add(entry.name);
		first = false;
	}
	return writer.size;
}

size_t Message(int level, uint32_t missing, bool test, wchar_t* out, size_t capacity) {
	Writer writer {out, capacity};
	writer.Add(L"");
	if (level >= 4) {
		writer.Add(L"This build needs a CPU with AVX-512 (x86-64-v4).");
	} else if (level == 3) {
		writer.Add(L"This build needs a CPU with AVX2, BMI2 and FMA (Intel Core 4th generation / AMD Ryzen or "
		           L"newer; x86-64-v3).");
	} else {
		writer.Add(L"This build needs a CPU with SSE4.2 and POPCNT (x86-64-v2).");
	}
	writer.Add(L"\n\nThis computer's CPU lacks: ");
	wchar_t names[512];
	FeatureNames(missing, names, sizeof(names) / sizeof(names[0]));
	writer.Add(names[0] != L'\0' ? names : L"(nothing; see below)");
	if (test) {
		writer.Add(L"\n\n(Shown because KYTY_TEST_CPU_CHECK=fail is set: a test of this message.)");
	}
	writer.Add(L"\n\nThe program cannot run on this CPU and will close now.");
	return writer.size;
}

CpuIdValues ReadCpu() {
	CpuIdValues values;
#if defined(KYTY_CPU_CHECK_X86)
	const auto cpuid = [](uint32_t leaf, uint32_t sub, uint32_t regs[4]) {
#if defined(_MSC_VER)
		int r[4] = {};
		__cpuidex(r, static_cast<int>(leaf), static_cast<int>(sub));
		for (int i = 0; i < 4; i++) {
			regs[i] = static_cast<uint32_t>(r[i]);
		}
#else
		__cpuid_count(leaf, sub, regs[0], regs[1], regs[2], regs[3]);
#endif
	};
	uint32_t r[4] = {};
	cpuid(0, 0, r);
	values.max_leaf = r[0];
	if (values.max_leaf >= 1) {
		cpuid(1, 0, r);
		values.leaf1_ecx = r[2];
	}
	if (values.max_leaf >= 7) {
		cpuid(7, 0, r);
		values.leaf7_ebx = r[1];
	}
	cpuid(0x80000000u, 0, r);
	values.max_extended_leaf = r[0];
	if (values.max_extended_leaf >= 0x80000001u) {
		cpuid(0x80000001u, 0, r);
		values.extended1_ecx = r[2];
	}
	if ((values.leaf1_ecx & Ecx1OsXsave) != 0) {
#if defined(_MSC_VER) && !defined(__clang__)
		values.xcr0 = _xgetbv(0);
#else
		// XGETBV itself: clang's _xgetbv needs the xsave target feature on the caller.
		uint32_t low = 0, high = 0;
		__asm__ volatile("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
		values.xcr0 = (static_cast<uint64_t>(high) << 32) | low;
#endif
	}
#endif
	return values;
}

} // namespace Kyty::CpuCheck

#if defined(_WIN32) && KYTY_CPU_CHECK_LEVEL >= 2

namespace {

bool TestFailure() {
	wchar_t value[16] = {};
	const DWORD n     = GetEnvironmentVariableW(L"KYTY_TEST_CPU_CHECK", value, 16);
	return n == 4 && value[0] == L'f' && value[1] == L'a' && value[2] == L'i' && value[3] == L'l';
}

// Run from the C initializer table: before the C++ static initializers of the program and before
// main. Returns 0 (go on) or ends the process.
int __cdecl StartupCheck() {
	using namespace Kyty::CpuCheck;
	const bool     test    = TestFailure();
	const uint32_t missing = Missing(ReadCpu(), KYTY_CPU_CHECK_LEVEL);
	if (missing == 0 && !test) {
		return 0;
	}
	wchar_t text[1024];
	const size_t length = Message(KYTY_CPU_CHECK_LEVEL, missing, test, text, sizeof(text) / sizeof(text[0]));

	// The text goes to standard error when there is one. A message box as well when nobody would
	// read that: no standard error (the launcher), or a console of the program's own, which closes
	// with it (a double-clicked kyty_emulator.exe). Redirected output (a pipe or a file: a script, or
	// the launcher's precompile window) gets the text alone.
	bool         shown   = false;
	bool         console = false;
	const HANDLE error   = GetStdHandle(STD_ERROR_HANDLE);
	DWORD        list[2];
	const DWORD  sharing = GetConsoleProcessList(list, 2); // processes on this console, 0 without one
	if (error != nullptr && error != INVALID_HANDLE_VALUE) {
		DWORD written = 0;
		DWORD mode    = 0;
		console       = GetConsoleMode(error, &mode) != 0;
		if (console) {
			shown = WriteConsoleW(error, L"\n", 1, &written, nullptr) &&
			        WriteConsoleW(error, text, static_cast<DWORD>(length), &written, nullptr) &&
			        WriteConsoleW(error, L"\n", 1, &written, nullptr);
		} else {
			// A file or a pipe: the text is ASCII.
			char narrow[1024];
			DWORD n = 0;
			for (; n < length && n + 1 < sizeof(narrow); n++) {
				narrow[n] = text[n] < 128 ? static_cast<char>(text[n]) : '?';
			}
			narrow[n++] = '\n';
			shown       = WriteFile(error, narrow, n, &written, nullptr) != 0;
		}
	}
	if (!shown || (console && sharing <= 1)) {
		MessageBoxW(nullptr, text, L"Kyty: unsupported CPU", MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
	}
	ExitProcess(3);
}

} // namespace

// .CRT$XIA..XIZ: the C initializers, run by the CRT before the C++ initializers (.CRT$XC*). The
// CRT's own pre-C initialization is in .CRT$XIAA, before this.
#pragma section(".CRT$XIB", long, read)
extern "C" __declspec(allocate(".CRT$XIB")) int(__cdecl* kyty_cpu_check_startup)(void) = StartupCheck;
#pragma comment(linker, "/include:kyty_cpu_check_startup")

#endif
