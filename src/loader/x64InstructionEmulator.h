#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// The host CPU's CPUID vendor string ("AuthenticAMD", "GenuineIntel", ...), and whether it names an
// AMD design. --amd-cpu defaults to on for every other vendor (main.cpp).
[[nodiscard]] std::string HostCpuVendor();
[[nodiscard]] bool        IsAmdCpuVendor(std::string_view vendor);

// The "AMD CPU patch" (--amd-cpu, KYTY_AMD_CPU): VRSQRTPS executions emulated so far by the trap
// fallback (sites without a native trampoline) and, with KYTY_AMD_CPU_TIMING=1, the time spent
// emulating them inside the handler (the trap's dispatch comes on top: ~2 us per trap on Windows,
// measured).
struct ReciprocalSqrtStats {
	uint64_t traps      = 0;
	uint64_t emulate_ns = 0;
};
// Reads KYTY_AMD_CPU_TIMING; called when the AMD CPU patch is applied to a module.
void                              ConfigureReciprocalSqrtStats();
[[nodiscard]] ReciprocalSqrtStats GetReciprocalSqrtStats();

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
