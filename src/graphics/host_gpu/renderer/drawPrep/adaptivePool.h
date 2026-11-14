#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_ADAPTIVEPOOL_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_ADAPTIVEPOOL_H_

#include "graphics/host_gpu/renderer/drawPrep/prepStats.h"
#include "graphics/host_gpu/renderer/drawPrep/workerGate.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>

// KYTY_DRAW_PREP_ADAPT=1 (startup, default off): a demand-sized preparation pool without idle spin.
// The default pool (WorkerGate) keeps two hot workers spinning 200 us after every slot and wakes
// cold ones that spin 50 us per wake: at the Sky Garden six workers spin about as much CPU as they
// work (prep agent, 2026-11-11), CPU that a 6- or 8-core machine needs for the game's own threads.
//
// - Every worker spins only briefly for new work after its last slot (KYTY_DRAW_PREP_ADAPT_SPIN_US,
//   default 10), then parks.
// - The publisher wakes one parked worker after a publish when no worker is spinning (an awake
//   worker is either preparing, and returns to the window when done, or spinning, and claims the
//   new slot). So workers wake one at a time while all awake ones are busy: as many run as the
//   stream needs.
// - Only workers below the active cap claim and are woken; the others park. The publisher
//   recomputes the cap about every 250 ms from the workers' busy time (PrepStats):
//     cap = ceil(1.3 x busy cores) + 1, at most the configured workers and at most the logical
//   processors in the process affinity minus one (the workers and the commit thread all fit on
//   the allowed processors at once). KYTY_DRAW_PREP_ADAPT_MAX overrides that bound,
//   KYTY_DRAW_PREP_ADAPT_MIN raises the floor, default 1. The bound counts the processors the
//   process may run on (its affinity mask), not the machine's idle processors.
//   Intended bound: the pool never runs fewer workers than the commit thread's demand (the busy
//   term). Until 2026-11-14 the bound was (logical - 4) / 2, which cut the pool below that demand
//   on small masks (6 logical processors: 1 worker, 8: 2); the commit thread then prepared the
//   heads itself (Playroom at 6 threads: commit 16.9 -> 26.0 ms per flip, heads ready 12.5%).
//   At 16 logical processors and more the old bound was never below the six workers, so nothing
//   changes there.
// - The lowest-numbered parked worker is woken first, so the same few stay warm.
//
// No wakeup is lost. A parking worker sets its parked bit and leaves the spinning count (both
// seq_cst), then looks for claimable work once more before it sleeps on its own word. The publisher
// stores the published count (seq_cst), then reads the spinning count and the parked bits. If the
// worker's last look missed the slot, its bit was set and its spin left before the publisher's
// reads: the publisher then sees no spinner on its account and wakes a parked worker (that one or
// another) unless some other worker still counts as spinning, and that one's own last look comes
// after the publish and claims the slot. The one gap: a spinner whose cap dropped parks without
// claiming; the commit thread then prepares that head itself (TryClaimHead), as with any
// unclaimed head.
namespace Libs::Graphics::DrawPrep {

// The processor bound of the active-worker cap (pure: unit tests): logical processors minus the
// commit thread's, at least 1.
[[nodiscard]] constexpr uint32_t AdaptiveProcessorBound(uint32_t affinity_lps) noexcept {
	return affinity_lps > 2u ? affinity_lps - 1u : 1u;
}

// The active-worker cap (pure: unit tests). `busy_cores`: the workers' busy time over the last
// interval divided by its length.
[[nodiscard]] inline uint32_t AdaptiveCap(double busy_cores, uint32_t workers, uint32_t affinity_lps,
                                          uint32_t min_cap, uint32_t max_override) {
	const uint32_t bound = max_override != 0 ? max_override : AdaptiveProcessorBound(affinity_lps);
	const auto     wanted = static_cast<uint32_t>(std::ceil(1.3 * std::max(busy_cores, 0.0))) + 1u;
	auto           cap    = std::min({wanted, std::max(bound, 1u), workers});
	cap                   = std::max({cap, std::min(min_cap, workers), 1u});
	return cap;
}

class AdaptivePool {
public:
	static constexpr uint32_t MaxWorkers = 32;

	AdaptivePool(uint32_t workers, uint32_t initial_cap) noexcept
	    : m_workers(std::min(workers, MaxWorkers)) {
		m_cap.store(std::clamp(initial_cap, 1u, m_workers), std::memory_order_relaxed);
	}

	[[nodiscard]] uint32_t Workers() const noexcept { return m_workers; }
	[[nodiscard]] uint32_t Cap() const noexcept { return m_cap.load(std::memory_order_relaxed); }
	void SetCap(uint32_t cap) noexcept {
		m_cap.store(std::clamp(cap, 1u, m_workers), std::memory_order_seq_cst);
	}

	// Publisher, after each publish (whose store of the published count is seq_cst). True when it
	// woke a worker.
	bool OnPublish() noexcept {
		if (m_spinning.load(std::memory_order_seq_cst) != 0) {
			return false;
		}
		return WakeOne();
	}

	// Wakes the lowest-numbered parked worker below the cap. True when it woke one.
	bool WakeOne() noexcept {
		for (;;) {
			const auto cap    = m_cap.load(std::memory_order_seq_cst);
			const auto mask   = cap >= 64u ? ~uint64_t {0} : (uint64_t {1} << cap) - 1u;
			const auto parked = m_parked.load(std::memory_order_seq_cst) & mask;
			if (parked == 0) {
				return false;
			}
			const auto index = static_cast<uint32_t>(std::countr_zero(parked));
			const auto bit   = uint64_t {1} << index;
			if ((m_parked.fetch_and(~bit, std::memory_order_seq_cst) & bit) == 0) {
				continue; // it left by itself
			}
			m_words[index].signal.fetch_add(1, std::memory_order_seq_cst);
			m_words[index].signal.notify_one();
			m_wakes.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
	}

	// Shutdown, after `stop` is set.
	void WakeAll() noexcept {
		for (uint32_t index = 0; index < m_workers; index++) {
			m_words[index].signal.fetch_add(1, std::memory_order_seq_cst);
			m_words[index].signal.notify_all();
		}
	}

	// Worker `index` enters its idle spin / leaves it.
	void BeginSpin() noexcept { m_spinning.fetch_add(1, std::memory_order_seq_cst); }
	void EndSpin() noexcept { m_spinning.fetch_sub(1, std::memory_order_seq_cst); }

	// Worker `index` with nothing to do: leaves its spin (when `spinning`), and sleeps until woken
	// unless `claimable()` (re-checked after announcing) or `stop`.
	template <typename Claimable>
	void Park(uint32_t index, bool spinning, Claimable&& claimable,
	          const std::atomic<bool>& stop) noexcept {
		auto&      word     = m_words[index].signal;
		const auto bit      = uint64_t {1} << index;
		const auto observed = word.load(std::memory_order_seq_cst);
		m_parked.fetch_or(bit, std::memory_order_seq_cst);
		if (spinning) {
			EndSpin();
		}
		if (!(index < Cap() && claimable()) && !stop.load(std::memory_order_seq_cst)) {
			word.wait(observed, std::memory_order_seq_cst);
		}
		// Woken (the waker cleared the bit) or leaving by itself.
		m_parked.fetch_and(~bit, std::memory_order_seq_cst);
	}

	[[nodiscard]] uint64_t Wakes() const noexcept { return m_wakes.load(std::memory_order_relaxed); }
	[[nodiscard]] uint32_t Spinning() const noexcept {
		return m_spinning.load(std::memory_order_relaxed);
	}
	[[nodiscard]] uint64_t ParkedMask() const noexcept {
		return m_parked.load(std::memory_order_relaxed);
	}

	// Publisher: the cap's interval accounting (Engine::Workers::Wake). Returns true when it is time
	// to recompute (every `interval_ns`), checking the clock only every 64th call.
	[[nodiscard]] bool CapDue(uint64_t interval_ns) noexcept {
		if ((++m_cap_calls & 63u) != 0u) {
			return false;
		}
		const auto now = WorkerNowNs();
		if (m_cap_last_ns == 0) {
			m_cap_last_ns = now;
			return false;
		}
		return now - m_cap_last_ns >= interval_ns;
	}
	// Publisher: busy cores since the last call (worker busy TSC from PrepStats over the interval's
	// TSC), and the interval restarts.
	[[nodiscard]] double TakeBusyCores() noexcept {
		uint64_t busy = 0;
		for (uint32_t i = 1; i < PrepStats::MaxWorkers; i++) {
			busy += PrepStats::g_workers[i].busy.load(std::memory_order_relaxed);
		}
		const auto tsc   = PrepStats::Tsc();
		double     cores = 0.0;
		if (m_cap_last_tsc != 0 && tsc > m_cap_last_tsc) {
			cores = static_cast<double>(busy - m_cap_last_busy) / static_cast<double>(tsc - m_cap_last_tsc);
		}
		m_cap_last_busy = busy;
		m_cap_last_tsc  = tsc;
		m_cap_last_ns   = WorkerNowNs();
		return cores;
	}

private:
	struct alignas(64) Word {
		std::atomic<uint32_t> signal {0};
	};

	const uint32_t                        m_workers;
	alignas(64) std::atomic<uint32_t>     m_spinning {0};
	alignas(64) std::atomic<uint64_t>     m_parked {0};
	alignas(64) std::atomic<uint32_t>     m_cap {1};
	std::atomic<uint64_t>                 m_wakes {0};
	std::array<Word, MaxWorkers>          m_words {};
	// Publisher only.
	uint32_t m_cap_calls     = 0;
	uint64_t m_cap_last_ns   = 0;
	uint64_t m_cap_last_tsc  = 0;
	uint64_t m_cap_last_busy = 0;
};

// One worker of the adaptive pool (Engine::Workers::Run; the unit tests run the same loop).
// `prepare(slot, seq)` must complete the slot; `max_batch()` as in RunPreparationWorkerBatched.
template <typename WindowT, typename Prepare, typename MaxBatch>
void RunAdaptiveWorker(AdaptivePool& pool, WindowT& window, uint32_t index, uint64_t spin_ns,
                       const std::atomic<bool>& stop, Prepare&& prepare, MaxBatch&& max_batch) {
	auto& stats    = PrepStats::t_local;
	stats.index    = index + 1u;
	auto  idle_tsc = PrepStats::Tsc();
	bool  spinning = false;
	auto  idle_start = WorkerNowNs();
	uint32_t spins   = 0;
	while (!stop.load(std::memory_order_acquire)) {
		const bool active = index < pool.Cap();
		if (active) {
			uint64_t   seq   = 0;
			uint32_t   count = 1;
			const auto batch = static_cast<uint32_t>(max_batch());
			decltype(window.TryClaim(seq)) slot = nullptr;
			bool claimed = false;
			if (batch <= 1u) {
				slot    = window.TryClaim(seq);
				claimed = slot != nullptr;
			} else {
				claimed = window.TryReserve(seq, count, batch, ClaimBatchMinBacklog);
			}
			if (claimed) {
				if (spinning) {
					pool.EndSpin();
					spinning = false;
				}
				const auto claimed_tsc = PrepStats::Tsc();
				stats.spin += claimed_tsc - idle_tsc;
				// More published work than this claim, and nobody spinning to take it: one more
				// worker (the publisher only wakes while nobody spins, and a spinner it counted may
				// have taken an earlier slot).
				if (pool.Spinning() == 0 && window.HasClaimable()) {
					pool.WakeOne();
				}
				for (uint32_t i = 0; i < count; i++) {
					auto* claimed_slot = batch <= 1u ? slot : window.ClaimReserved(seq + i);
					if (claimed_slot == nullptr) {
						continue;
					}
					prepare(*claimed_slot, seq + i);
					stats.slots++;
				}
				idle_tsc = PrepStats::Tsc();
				stats.busy += idle_tsc - claimed_tsc;
				if (stats.slots >= 32u) {
					PrepStats::Flush(stats);
				}
				idle_start = WorkerNowNs();
				spins      = 0;
				continue;
			}
			if (!spinning && spin_ns != 0) {
				pool.BeginSpin();
				spinning   = true;
				idle_start = WorkerNowNs();
				spins      = 0;
			}
			if (spinning) {
				WorkerRelax();
				if ((++spins & 63u) != 0u || WorkerNowNs() - idle_start < spin_ns) {
					continue;
				}
			}
		}
		const auto park_tsc = PrepStats::Tsc();
		stats.spin += park_tsc - idle_tsc;
		pool.Park(index, spinning, [&window] { return window.HasClaimable(); }, stop);
		spinning = false;
		idle_tsc = PrepStats::Tsc();
		stats.park += idle_tsc - park_tsc;
		stats.parks++;
		PrepStats::Flush(stats);
		idle_start = WorkerNowNs();
		spins      = 0;
	}
	if (spinning) {
		pool.EndSpin();
	}
	PrepStats::Flush(stats);
}

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_ADAPTIVEPOOL_H_
