#ifndef EMULATOR_SRC_COMMON_HELPERPRIORITY_H_
#define EMULATOR_SRC_COMMON_HELPERPRIORITY_H_

#include "common/liveSwitch.h"
#include "common/threads.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

// KYTY_HELPER_PRIORITY (live, measurement; default unset: nothing changes): the host priority step
// of the emulator's helper threads that spin while they wait: the CP sequencer, the CP recorder,
// the draw-prep workers and the stage-prep helper. 1 above normal, 0 normal, -1 below normal (the
// game's normal priority band), -2 lowest. Unset: each keeps its own start-up priority (`own`: 1
// for the sequencer and the recorder, 0 for the others). The commit thread is not changed. Each
// helper applies a change to itself at its next wait or slot (one relaxed load otherwise).
// It exists to measure what the game's threads lose to helpers that outrank them on a small CPU.
namespace Common::HelperPriority {

inline constexpr int64_t Unset = 100;

inline std::atomic<uint32_t> g_generation {1};

inline Live::Switch g_step(
    "KYTY_HELPER_PRIORITY",
    [](const char* value) -> int64_t {
	    if (value == nullptr) {
		    return Unset;
	    }
	    return std::clamp<long>(std::strtol(value, nullptr, 10), -2, 2);
    },
    [](int64_t /*previous*/, int64_t /*value*/) { g_generation.fetch_add(1, std::memory_order_relaxed); });

inline void Check(int own) noexcept {
	thread_local uint32_t seen = 0;
	const auto            generation = g_generation.load(std::memory_order_relaxed);
	if (generation != seen) [[unlikely]] {
		const bool first = seen == 0;
		seen             = generation;
		const auto step  = g_step.Get();
		if (step == Unset && first) {
			return; // the thread's own start-up priority stands
		}
		Common::SetCurrentThreadPriorityStep(step == Unset ? own : static_cast<int>(step));
	}
}

} // namespace Common::HelperPriority

#endif // EMULATOR_SRC_COMMON_HELPERPRIORITY_H_
