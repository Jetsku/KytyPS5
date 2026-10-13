#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_OWNEDCOUNTERS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_OWNEDCOUNTERS_H_

#include "common/liveSwitch.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

// KYTY_CP_OWNED_COUNTERS (default off, live): the draw path's statistics counters that only the
// GPU thread writes (the draw-run, draw-prep commit and binding-plan totals behind the 10-second
// console lines) are incremented with a relaxed load and store instead of a locked add, the way
// the profiler's per-thread counters are (common/profiler.h, Detail::Add). A continuation draw
// takes about 17 of them; each locked add costs some 20-25 cycles and orders the loads around it.
// Readers (the console lines, tests) see the same totals. Nothing feeds back into emulation.
// DrawRun::PrintSummary, called once per committed draw, then reads its clock every 256th call.
// verify: also checks the single-writer assumption: an increment on another thread is counted
// (reported once, at the first) and done with the locked add.
namespace Libs::Graphics::OwnedCounters {

inline int64_t ParseMode(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}

inline Live::Switch                 g_mode("KYTY_CP_OWNED_COUNTERS", ParseMode);
inline std::atomic<uint64_t>        g_foreign_writes {0};

inline void Add(std::atomic<uint64_t>& counter, uint64_t amount) noexcept {
	const auto mode = g_mode.Get();
	if (mode == 0) {
		counter.fetch_add(amount, std::memory_order_relaxed);
		return;
	}
	if (mode == 2 && !GuestGpu::IsGpuThread()) [[unlikely]] {
		if (g_foreign_writes.fetch_add(1, std::memory_order_relaxed) == 0) {
			std::printf("OwnedCountersVerify: a GPU-thread counter was incremented on another "
			            "thread\n");
			std::fflush(stdout);
		}
		counter.fetch_add(amount, std::memory_order_relaxed);
		return;
	}
	counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::OwnedCounters

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_OWNEDCOUNTERS_H_
