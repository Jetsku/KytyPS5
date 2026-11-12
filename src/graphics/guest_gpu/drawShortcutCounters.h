#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DRAW_SHORTCUT_COUNTERS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DRAW_SHORTCUT_COUNTERS_H_

#include <atomic>
#include <cstdint>

// Always-on counters of two upstream draw shortcuts (relaxed adds on the command-processor thread):
// draws the empty-screen-scissor skip removes (KYTY_EMPTY_SCISSOR_SKIP, upstream b727a1faa) and
// DRAW_INDIRECT / DRAW_INDEX_INDIRECT packets that fall into multi-draw runs (KYTY_INDIRECT_RUNS,
// upstream b30f2d6c9). KYTY_DRAW_SHORTCUT_LOG=1 prints them every 600 guest flips (graphicsRun.cpp).
namespace Libs::Graphics::DrawShortcutCounters {

inline std::atomic<uint64_t> g_empty_scissor_skips {0};
// Indirect draw packets executed (each packet of a run counts once).
inline std::atomic<uint64_t> g_indirect_packets {0};
// Runs of two or more packets and the packets in them.
inline std::atomic<uint64_t> g_indirect_runs {0};
inline std::atomic<uint64_t> g_indirect_run_packets {0};

inline void Add(std::atomic<uint64_t>& counter, uint64_t amount) {
	counter.fetch_add(amount, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::DrawShortcutCounters

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_DRAW_SHORTCUT_COUNTERS_H_ */
