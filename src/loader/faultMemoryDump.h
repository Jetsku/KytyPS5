#ifndef EMULATOR_INCLUDE_EMULATOR_LOADER_FAULT_MEMORY_DUMP_H_
#define EMULATOR_INCLUDE_EMULATOR_LOADER_FAULT_MEMORY_DUMP_H_

#include <cstddef>
#include <cstdint>

// Which memory to print with a guest crash report: the bytes around every general-purpose
// register that points at readable memory. A crash on a corrupted heap object shows the object
// (and the one it points to) next to the registers, so a report tells whether the object was
// freed, overwritten past its end, or never initialised.
namespace Loader::FaultMemoryDump {

constexpr uint64_t BEFORE         = 0x40;
constexpr uint64_t AFTER          = 0xe0;
constexpr uint64_t MIN_ADDRESS    = 0x10000;
constexpr int      REGISTER_COUNT = 16;
constexpr int      RSP_INDEX      = 7; // rax rbx rcx rdx rsi rdi rbp rsp r8..r15; rsp is dumped already

struct Window {
	int      reg   = -1;
	uint64_t value = 0;
	uint64_t start = 0;
	uint64_t size  = 0;
};

// Fills `out` with up to `capacity` windows, in register order. `readable(start, size)` must
// not fault. A register whose window is not readable falls back to the bytes from the pointed-to
// address on; one whose first 0x80 bytes an earlier window already shows is skipped.
template <typename Readable>
size_t PickWindows(const uint64_t* regs, Readable&& readable, Window* out, size_t capacity) {
	size_t count = 0;
	for (int reg = 0; reg < REGISTER_COUNT && count < capacity; reg++) {
		const uint64_t value = regs[reg];
		if (reg == RSP_INDEX || value < MIN_ADDRESS || value > UINT64_MAX - AFTER) {
			continue;
		}
		bool covered = false;
		for (size_t i = 0; i < count; i++) {
			if (out[i].start <= value && value + 0x80 <= out[i].start + out[i].size) {
				covered = true;
				break;
			}
		}
		if (covered) {
			continue;
		}
		const uint64_t aligned = value & ~uint64_t {0xf};
		Window         window {reg, value, aligned - BEFORE, BEFORE + AFTER};
		if (!readable(window.start, window.size)) {
			window.start = aligned;
			window.size  = AFTER;
			if (!readable(window.start, window.size)) {
				continue;
			}
		}
		out[count++] = window;
	}
	return count;
}

} // namespace Loader::FaultMemoryDump

#endif // EMULATOR_INCLUDE_EMULATOR_LOADER_FAULT_MEMORY_DUMP_H_
