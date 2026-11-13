#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PREPSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PREPSTATS_H_

#include <algorithm>
#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

// Always-on draw-prep cost counters (cheap: a TSC read per claim, per park and per head; per-thread
// sums flushed into per-worker cache lines every 32 slots and at every park). Printed with the
// "DrawPrep 10s" line as "PrepStats 10s": per worker the slots prepared and the TSC time spent
// preparing (busy), spinning for work (spin) and parked; on the command processor the head
// states (ready, waited for, prepared there) with the wait time, and how long a finished
// preparation waited for its commit (lead).
namespace Libs::Graphics::DrawPrep::PrepStats {

[[nodiscard]] inline uint64_t Tsc() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return 0;
#endif
}

constexpr uint32_t MaxWorkers = 33; // index 0: the command processor (steals); 1..32 DrawPrep#k

struct alignas(64) WorkerCounters {
	std::atomic<uint64_t> slots {0};
	std::atomic<uint64_t> busy {0};
	std::atomic<uint64_t> spin {0};
	std::atomic<uint64_t> park {0};
	std::atomic<uint64_t> parks {0};
	// KYTY_DRAW_PREP_LEAN digest part: code digests taken from the cache, hashed (and stored), and
	// looked up where the cache could not be used.
	std::atomic<uint64_t> digest_hits {0};
	std::atomic<uint64_t> digest_misses {0};
	std::atomic<uint64_t> digest_uncacheable {0};
};

inline WorkerCounters g_workers[MaxWorkers];

struct Local {
	uint32_t index = 0;
	uint64_t slots = 0;
	uint64_t busy  = 0;
	uint64_t spin  = 0;
	uint64_t park  = 0;
	uint64_t parks = 0;
	uint64_t digest_hits        = 0;
	uint64_t digest_misses      = 0;
	uint64_t digest_uncacheable = 0;
	uint64_t unflushed = 0;
};

inline thread_local Local t_local;

inline void Flush(Local& local) noexcept {
	auto& g = g_workers[std::min(local.index, MaxWorkers - 1u)];
	g.slots.fetch_add(local.slots, std::memory_order_relaxed);
	g.busy.fetch_add(local.busy, std::memory_order_relaxed);
	g.spin.fetch_add(local.spin, std::memory_order_relaxed);
	g.park.fetch_add(local.park, std::memory_order_relaxed);
	g.parks.fetch_add(local.parks, std::memory_order_relaxed);
	g.digest_hits.fetch_add(local.digest_hits, std::memory_order_relaxed);
	g.digest_misses.fetch_add(local.digest_misses, std::memory_order_relaxed);
	g.digest_uncacheable.fetch_add(local.digest_uncacheable, std::memory_order_relaxed);
	const auto index = local.index;
	local            = Local {};
	local.index      = index;
}

// Command processor side (one thread; the reporter runs there too).
struct Commit {
	uint64_t ready     = 0; // head done when the commit wanted it
	uint64_t waited    = 0; // a worker held it: the commit spun
	uint64_t wait_tsc  = 0;
	uint64_t self      = 0; // nobody had claimed it: prepared on the command processor
	uint64_t self_tsc  = 0;
	uint64_t lead_tsc  = 0; // ready heads: TSC from Complete to the commit taking it
	uint64_t lead_hist[6] {}; // <2, <10, <50, <200, <1000 us, more
	uint64_t occupancy = 0; // window occupancy (published - head) at each head
	uint64_t heads     = 0;
};

inline Commit g_commit;

// GPU (command processor) thread idle time, by what it waited for (one thread, plain counters):
// - starved: inside a slice, the resolver had no op from the sequencer yet (its spin; the slice then
//   returns Blocked, so a longer starvation continues as idle time below);
// - empty: outside slices, no submission queued at all (the game has not submitted);
// - blocked: outside slices, submissions queued but every queue suspended (waits on GPU labels,
//   memory, a blocked op);
// - fence: outside slices, the next submission is held by the frame fence;
// - nowork: the thread sleeps in m_work_available.Wait with no submission and no service command
//   (graphicsRun.cpp; wholecp measurement branch: not counted in the buckets above).
struct CpIdle {
	uint64_t starved = 0, starved_n = 0;
	uint64_t empty = 0, blocked = 0, fence = 0;
	uint64_t nowork = 0;
};

inline CpIdle g_cp_idle;

// Spin time of the sequencer's waits (WaitSlow, until ready or parked) and of the recorder's idle
// wait for published commands (Ring::WaitPublished's spin).
struct Spins {
	std::atomic<uint64_t> sequencer {0};
	std::atomic<uint64_t> recorder {0};
};

inline Spins g_spin;

} // namespace Libs::Graphics::DrawPrep::PrepStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_PREPSTATS_H_
