#ifndef EMULATOR_SRC_COMMON_FPENV_H_
#define EMULATOR_SRC_COMMON_FPENV_H_

#include <cstdint>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// The console's floating-point environment for guest code (based on upstream f399d4818): MXCSR
// 0x9fc0 (all exceptions masked, round to nearest, flush-to-zero and denormals-are-zero) and the
// x87 control word 0x037f (64-bit precision, all exceptions masked).
//
// KYTY_GUEST_FP_ENV=1 (startup switch, default off): guest threads (before their entry), the main
// thread (before the first module initializer) and new fibers (sceFiber with FIBER_FLAG_SET_FPU_REGS)
// start with it, and a fiber switch carries each fiber's environment. Emulator code called from
// guest code on those threads (HLE functions, the fault handler) then runs with it too: the
// emulator's own float code that relies on IEEE denormals sets its MXCSR itself
// (x64InstructionEmulator) or keeps the guest's (guestInstructionPatcher). Off: host default
// (Windows: MXCSR 0x1f80, control word 0x027f), as before.

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

[[nodiscard]] inline bool GuestFpEnvEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_GUEST_FP_ENV");
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

#if defined(__x86_64__) || defined(_M_X64)
[[nodiscard]] inline FpEnv ReadFpEnv() {
	FpEnv env {};
	env.mxcsr = _mm_getcsr();
	asm volatile("fnstcw %0" : "=m"(env.control) : : "memory");
	return env;
}

inline void WriteFpEnv(FpEnv env) {
	_mm_setcsr(env.mxcsr);
	asm volatile("fldcw %0" : : "m"(env.control) : "memory");
}
#else
[[nodiscard]] inline FpEnv ReadFpEnv() {
	return {};
}
inline void WriteFpEnv(FpEnv /*env*/) {}
#endif

// Called where a host thread starts running guest code.
inline void EnterGuestFpEnv() {
	if (GuestFpEnvEnabled()) {
		WriteFpEnv(GuestFpEnv);
	}
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_FPENV_H_
