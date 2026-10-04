#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// The "AMD CPU patch" (--amd-cpu, KYTY_AMD_CPU): patched VRSQRTPS executions emulated so far and,
// with KYTY_AMD_CPU_TIMING=1, the time spent emulating them inside the handler (the trap's
// dispatch comes on top: ~2 us per trap on Windows, measured).
struct ReciprocalSqrtStats {
	uint64_t traps      = 0;
	uint64_t emulate_ns = 0;
};
[[nodiscard]] ReciprocalSqrtStats GetReciprocalSqrtStats();

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
