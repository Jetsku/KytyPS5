#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include "common/guestLoad.h"

#include <cstdint>
#include <string>
#include <string_view>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);

// Plain guest loads (common/guestLoad.h, registered with GuestLoad::Register): decodes the
// faulting instruction of `native_context` if it is a supported load whose memory source holds
// `fault_vaddr`, and completes it with given bytes (destination register written as the CPU
// would, VEX forms clearing the upper YMM bits, the instruction pointer advanced).
[[nodiscard]] bool DecodeGuestLoad(void* native_context, uint64_t fault_vaddr, GuestLoad::Load& load);
[[nodiscard]] bool ApplyGuestLoad(void* native_context, const GuestLoad::Load& load, const void* bytes);

// Store in place (KYTY_STORE_IN_PLACE / KYTY_STORE_CLEAN): writes [vaddr, vaddr + size) for the
// store at `pc` that faulted at fault_vaddr; false to decline.
using StoreWriter = bool (*)(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                             uint64_t size);
// Completes a faulting store whose memory operand covers fault_vaddr by handing its bytes to
// `write` instead of the faulting mapping, then steps past the instruction (adapted from
// usa-ratt/KytyPS5-GT7 21246d78a). Plain stores only: MOV and MOVNTI from a general-purpose
// register (or MOV from an immediate), and SSE/AVX moves of up to 16 bytes from an XMM register.
// False, with the context untouched, for anything else (read-modify-write, string and YMM stores
// among them) or when `write` declines.
[[nodiscard]] bool TryEmulateStore(void* native_context, uint64_t fault_vaddr, StoreWriter write);

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
