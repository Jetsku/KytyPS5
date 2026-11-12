#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WORKERGATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WORKERGATE_H_

#include "graphics/host_gpu/renderer/drawPrep/prepStats.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

// Draw-prep worker parking (KYTY_DRAW_PREP_HOT). At the Sky Garden rate the window needs about
// 1.2 busy workers, yet all six spun 200 us after every slot: about 3.7 cores of spin per flip.
//
// - Hot workers (the first `hot`) keep that behaviour: spin for a while after their last slot,
//   then park; the producer wakes parked hot workers on its next publish. While they keep up, a
//   publish costs the producer one relaxed-ordering load (no syscall).
// - Cold workers park as soon as nothing is claimable, and wake only when the unclaimed backlog
//   reaches `wake_backlog`. The wake is requested by whichever thread sees that backlog: a worker
//   right after its claim, or the producer on every `wake_backlog`-th publish. At most one cold
//   wake is in flight, so a burst costs one WakeByAddress call, not one per draw.
//
// No wakeup is lost: a sleeper announces itself (sleepers++, seq_cst) and then re-checks its
// predicate, and a waker changes the signal word after its own seq_cst update of what the
// predicate reads. Either the sleeper sees the new state, or the waker sees the sleeper.
//
// Cold wakes, default protocol: a wake sets a "pending" flag that only a sleeper leaving the park
// clears. That flag can stick: a cold worker that parks while the backlog is already high
// announces itself, skips the wait and clears the flag on its way out; a waker that saw it
// announced but sets the flag only after that clear leaves it set with no sleeper to clear it.
// Every later MaybeWakeCold then returns at once and parked cold workers never wake again (seen in
// game: no cold wake in a whole session, every waited head held by one of the two hot workers
// while other published slots waited unclaimed).
// KYTY_DRAW_PREP_COLD_TOKEN=1 (cold_token): a wake is a token instead (0 none, 1 one in flight). A
// cold worker that parks first takes an outstanding token, and only sleeps while there is none, so
// a token given while its sleeper was leaving is taken by the next cold worker to park instead of
// blocking every later wake. Still at most one cold wake in flight.
//
// Work stealing (KYTY_DRAW_PREP_STEAL, AwaitHead below): a producer that must commit a head a
// worker still holds prepares later unclaimed slots itself instead of idling.
namespace Libs::Graphics::DrawPrep {

class WorkerGate {
public:
	WorkerGate(uint32_t workers, uint32_t hot, uint32_t wake_backlog,
	           bool cold_token = false) noexcept
	    : m_hot(hot == 0 || hot > workers ? workers : hot), m_has_cold(m_hot < workers),
	      m_wake_backlog(wake_backlog == 0 ? 1u : wake_backlog), m_cold_token_mode(cold_token) {}

	[[nodiscard]] bool ColdTokenMode() const noexcept { return m_cold_token_mode; }

	[[nodiscard]] bool     Hot(uint32_t index) const noexcept { return index < m_hot; }
	[[nodiscard]] uint32_t HotCount() const noexcept { return m_hot; }
	[[nodiscard]] bool     HasCold() const noexcept { return m_has_cold; }
	[[nodiscard]] uint32_t WakeBacklog() const noexcept { return m_wake_backlog; }

	// Producer, after each publish (the publish itself is a seq_cst store).
	// `backlog` is evaluated only on every WakeBacklog()-th publish, so the producer rarely reads
	// the workers' contended claim counter. Returns true when it requested a cold wake.
	template <typename Backlog>
	bool OnPublish(Backlog&& backlog) noexcept {
		if (m_hot_sleepers.load(std::memory_order_seq_cst) != 0) {
			m_hot_signal.fetch_add(1, std::memory_order_seq_cst);
			m_hot_signal.notify_all();
		}
		if (!m_has_cold || ++m_publishes < m_wake_backlog) {
			return false;
		}
		m_publishes = 0;
		return MaybeWakeCold(backlog());
	}

	// Any thread that has just measured the unclaimed backlog. Returns true when it woke one.
	bool MaybeWakeCold(uint64_t backlog) noexcept {
		if (m_cold_token_mode) {
			if (backlog < m_wake_backlog || m_cold_sleepers.load(std::memory_order_seq_cst) == 0 ||
			    m_cold_token.load(std::memory_order_relaxed) != 0) {
				return false;
			}
			uint32_t none = 0;
			if (!m_cold_token.compare_exchange_strong(none, 1u, std::memory_order_seq_cst)) {
				return false; // another thread's wake is in flight
			}
			m_cold_token.notify_one();
			m_cold_wakes.fetch_add(1, std::memory_order_relaxed);
			return true;
		}
		if (backlog < m_wake_backlog || m_cold_sleepers.load(std::memory_order_seq_cst) == 0 ||
		    m_cold_wake_pending.load(std::memory_order_relaxed) ||
		    m_cold_wake_pending.exchange(true, std::memory_order_acq_rel)) {
			return false;
		}
		m_cold_signal.fetch_add(1, std::memory_order_seq_cst);
		m_cold_signal.notify_one();
		m_cold_wakes.fetch_add(1, std::memory_order_relaxed);
		return true;
	}

	// Hot worker with nothing claimable after its spin: sleeps until the next publish.
	template <typename Claimable>
	void ParkHot(Claimable&& claimable, const std::atomic<bool>& stop) noexcept {
		const auto observed = m_hot_signal.load(std::memory_order_seq_cst);
		m_hot_sleepers.fetch_add(1, std::memory_order_seq_cst);
		if (!claimable() && !stop.load(std::memory_order_seq_cst)) {
			m_hot_signal.wait(observed, std::memory_order_seq_cst);
		}
		m_hot_sleepers.fetch_sub(1, std::memory_order_seq_cst);
	}

	// Cold worker with nothing claimable: sleeps until the backlog reaches WakeBacklog().
	template <typename BacklogHigh>
	void ParkCold(BacklogHigh&& backlog_high, const std::atomic<bool>& stop) noexcept {
		if (m_cold_token_mode) {
			m_cold_sleepers.fetch_add(1, std::memory_order_seq_cst);
			for (;;) {
				// An outstanding wake (also one given while its intended sleeper was leaving) is
				// this worker's: it goes back to claiming instead of sleeping.
				uint32_t token = 1;
				if (m_cold_token.compare_exchange_strong(token, 0u, std::memory_order_seq_cst)) {
					break;
				}
				// Shutdown (WakeAll) leaves its own value, which no sleeper consumes.
				if (token == ColdTokenShutdown || backlog_high() ||
				    stop.load(std::memory_order_seq_cst)) {
					break;
				}
				m_cold_token.wait(0u, std::memory_order_seq_cst);
			}
			m_cold_sleepers.fetch_sub(1, std::memory_order_seq_cst);
			return;
		}
		const auto observed = m_cold_signal.load(std::memory_order_seq_cst);
		m_cold_sleepers.fetch_add(1, std::memory_order_seq_cst);
		if (!backlog_high() && !stop.load(std::memory_order_seq_cst)) {
			m_cold_signal.wait(observed, std::memory_order_seq_cst);
		}
		m_cold_sleepers.fetch_sub(1, std::memory_order_seq_cst);
		// Whoever leaves the park (woken or not) re-arms the next wake. A pending flag is only set
		// while a sleeper is announced, and that sleeper always gets here.
		m_cold_wake_pending.store(false, std::memory_order_release);
	}

	// Shutdown: after `stop` is set.
	void WakeAll() noexcept {
		m_hot_signal.fetch_add(1, std::memory_order_seq_cst);
		m_hot_signal.notify_all();
		m_cold_signal.fetch_add(1, std::memory_order_seq_cst);
		m_cold_signal.notify_all();
		// Not a wake token: one sleeper would consume it and the others sleep on.
		m_cold_token.store(ColdTokenShutdown, std::memory_order_seq_cst);
		m_cold_token.notify_all();
	}

	// KYTY_DRAW_PREP_ADAPT live switch: after WakeAll moved the workers out, makes the gate usable
	// again (the cold token leaves its shutdown state). No worker may be parked in the gate.
	void Rearm() noexcept {
		uint32_t shutdown = ColdTokenShutdown;
		(void)m_cold_token.compare_exchange_strong(shutdown, 0u, std::memory_order_seq_cst);
	}

	// Tests: a cold wake token is outstanding (KYTY_DRAW_PREP_COLD_TOKEN), or the default
	// protocol's pending flag is set.
	[[nodiscard]] bool ColdWakeOutstanding() const noexcept {
		return m_cold_token_mode ? m_cold_token.load(std::memory_order_seq_cst) != 0
		                         : m_cold_wake_pending.load(std::memory_order_seq_cst);
	}

	[[nodiscard]] uint64_t ColdWakes() const noexcept {
		return m_cold_wakes.load(std::memory_order_relaxed);
	}
	[[nodiscard]] uint32_t ColdSleepers() const noexcept {
		return m_cold_sleepers.load(std::memory_order_relaxed);
	}
	[[nodiscard]] uint32_t HotSleepers() const noexcept {
		return m_hot_sleepers.load(std::memory_order_relaxed);
	}

private:
	const uint32_t m_hot;
	const bool     m_has_cold;
	const uint32_t m_wake_backlog;
	const bool     m_cold_token_mode;
	uint32_t       m_publishes = 0; // producer only
	// KYTY_DRAW_PREP_COLD_TOKEN: the cold wake token (0 none, 1 one in flight, ColdTokenShutdown
	// after WakeAll).
	static constexpr uint32_t         ColdTokenShutdown = 2;
	alignas(64) std::atomic<uint32_t> m_cold_token {0};

	alignas(64) std::atomic<uint64_t> m_hot_signal {0};
	std::atomic<uint32_t>             m_hot_sleepers {0};
	alignas(64) std::atomic<uint64_t> m_cold_signal {0};
	std::atomic<uint32_t>             m_cold_sleepers {0};
	std::atomic<bool>                 m_cold_wake_pending {false};
	std::atomic<uint64_t>             m_cold_wakes {0};
};

inline uint64_t WorkerNowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

inline void WorkerRelax() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

// One preparation worker (Engine::Workers::Run; the unit tests run the same loop). `prepare` gets
// each claimed slot and its sequence number and must complete it; `on_cold_wake` is called after
// this worker woke a cold one. Workers look at the clock only every 256 idle spins.
// KYTY_DRAW_PREP_LEAN claim part: a worker reserves `max_batch()` slots at once only while at least
// this many are unclaimed, so a batch never holds back a slot another worker could start now.
inline constexpr uint64_t ClaimBatchMinBacklog = 4;

// The worker loop's idle hooks (KYTY_SPIN_BUDGET, common/spinGate.h): `SpinLimit(wanted)` at the
// first empty look of each idle period gives the spin allowed before parking; `Spun(tsc, parked)`
// reports each idle period's spin when it ends (a claim, or a park). The default changes nothing.
struct NoWorkerHooks {
	[[nodiscard]] uint64_t SpinLimit(uint64_t wanted) const noexcept { return wanted; }
	void                   Spun(uint64_t /*tsc*/, bool /*parked*/) const noexcept {}
};

template <typename WindowT, typename Prepare, typename OnColdWake, typename MaxBatch,
          typename Hooks = NoWorkerHooks>
void RunPreparationWorkerBatched(WorkerGate& gate, WindowT& window, uint32_t index,
                                 uint64_t hot_spin_ns, uint64_t cold_spin_ns,
                                 const std::atomic<bool>& stop, Prepare&& prepare,
                                 OnColdWake&& on_cold_wake, MaxBatch&& max_batch,
                                 Hooks&& hooks = Hooks {}) {
	const bool hot        = gate.Hot(index);
	const auto wanted     = hot ? hot_spin_ns : cold_spin_ns;
	auto       spin_limit = wanted;
	bool       idle_begun = false; // this idle period asked the hooks for its spin
	auto       idle_start = WorkerNowNs();
	uint32_t   spins      = 0;
	// PrepStats (prepStats.h): TSC split of this worker's time into preparing, spinning, parked.
	auto& stats    = PrepStats::t_local;
	stats.index    = index + 1u;
	auto  idle_tsc = PrepStats::Tsc();
	while (!stop.load(std::memory_order_acquire)) {
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
			const auto claimed_tsc = PrepStats::Tsc();
			stats.spin += claimed_tsc - idle_tsc;
			if (idle_begun) {
				hooks.Spun(claimed_tsc - idle_tsc, false);
				idle_begun = false;
			}
			if (gate.HasCold() && gate.MaybeWakeCold(window.Unclaimed())) {
				on_cold_wake();
			}
			for (uint32_t i = 0; i < count; i++) {
				auto* claimed_slot = batch <= 1u ? slot : window.ClaimReserved(seq + i);
				if (claimed_slot == nullptr) {
					continue; // the producer took it as the head
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
		if (!idle_begun) {
			idle_begun = true;
			spin_limit = hooks.SpinLimit(wanted);
		}
		WorkerRelax();
		if ((++spins & 255u) != 0u || WorkerNowNs() - idle_start < spin_limit) {
			continue;
		}
		const auto park_tsc = PrepStats::Tsc();
		stats.spin += park_tsc - idle_tsc;
		hooks.Spun(park_tsc - idle_tsc, true);
		idle_begun = false;
		if (hot) {
			gate.ParkHot([&window] { return window.HasClaimable(); }, stop);
		} else {
			gate.ParkCold([&window, &gate] { return window.Unclaimed() >= gate.WakeBacklog(); },
			              stop);
		}
		idle_tsc = PrepStats::Tsc();
		stats.park += idle_tsc - park_tsc;
		stats.parks++;
		PrepStats::Flush(stats);
		idle_start = WorkerNowNs();
	}
	PrepStats::Flush(stats);
}

template <typename WindowT, typename Prepare, typename OnColdWake>
void RunPreparationWorker(WorkerGate& gate, WindowT& window, uint32_t index, uint64_t hot_spin_ns,
                          uint64_t cold_spin_ns, const std::atomic<bool>& stop, Prepare&& prepare,
                          OnColdWake&& on_cold_wake) {
	RunPreparationWorkerBatched(gate, window, index, hot_spin_ns, cold_spin_ns, stop,
	                            std::forward<Prepare>(prepare), std::forward<OnColdWake>(on_cold_wake),
	                            [] { return 1u; });
}

// When the producer steals (KYTY_DRAW_PREP_STEAL, KYTY_DRAW_PREP_STEAL_AFTER_US). A stolen
// preparation that outlasts the head delays the head's commit, and every commit after it, so the
// producer steals only while the head is likely to take long and the workers are behind.
struct StealPolicy {
	uint32_t min_unclaimed = 0; // 0: never; else steal while at least this many slots are unclaimed
	uint64_t after_ns      = 0; // spin this long on the held head before the first steal
};

// What the producer's wait for a held head does with queued service commands at one look
// (Engine::CommitHead). Guard: the deadlock guard (every 1024th look, after 2 ms). Serve:
// KYTY_SERVE_IN_WAITS bit 1 (based on chenxiao07/KytyPS5 6801ddc13): every 64th look while
// commands are queued. `pending` is asked only at a look that could serve.
enum class HeadWaitService { None, Serve, Guard };
template <typename Pending>
[[nodiscard]] HeadWaitService HeadWaitServiceAt(uint32_t spins, uint64_t waited_ns, bool serve_in_waits,
                                                Pending&& pending) {
	if (serve_in_waits && (spins & 63u) == 63u && pending()) {
		return HeadWaitService::Serve;
	}
	if ((spins & 1023u) == 1023u && waited_ns > 2'000'000u) {
		return HeadWaitService::Guard;
	}
	return HeadWaitService::None;
}

struct AwaitStats {
	uint32_t stolen  = 0; // slots the producer prepared
	uint64_t spin_ns = 0; // time it spun, steals excluded
};

// Producer (Engine::CommitHead; the unit tests run the same code), when the head it must commit
// next is claimed by another thread and not done yet. Returns once the head is done.
// - It spins, calling `relax(spins, spin_start_ns)` between looks at the head.
// - After policy.after_ns, while at least policy.min_unclaimed published slots are unclaimed, it
//   claims the oldest of them with the workers' own TryClaim and hands it to `prepare`. That
//   callback must prepare and complete the slot exactly as a worker does. Like a worker's claim, a
//   claim that sees a backlog may wake a cold worker.
// - The producer publishes nothing meanwhile, so the unclaimed backlog only shrinks. Once it is
//   below the policy's minimum, the producer only spins.
// - Commit order is unchanged: the caller commits the head alone. A stolen slot is committed later,
//   when it is the head, like any worker-prepared one.
// - A steal never takes the head: the head is claimed already. So it delays the head's commit by
//   at most one preparation.
template <typename WindowT, typename Prepare, typename Relax, typename OnColdWake>
AwaitStats AwaitHead(WorkerGate& gate, WindowT& window, const StealPolicy& policy,
                     Prepare&& prepare, Relax&& relax, OnColdWake&& on_cold_wake) {
	AwaitStats stats;
	bool       may_steal   = policy.min_unclaimed != 0;
	uint32_t   until_check = 0; // the clock and the claim counter are read every 16th look
	const auto start       = WorkerNowNs();
	auto       spin_start  = start;
	for (uint32_t spins = 0; !window.HeadDone(); spins++) {
		if (may_steal && until_check-- == 0) {
			until_check    = 15;
			const auto now = WorkerNowNs();
			if (now - start >= policy.after_ns) {
				uint64_t seq  = 0;
				auto*    slot = window.Unclaimed() >= policy.min_unclaimed ? window.TryClaim(seq)
				                                                           : nullptr;
				if (slot == nullptr) {
					may_steal = false;
				} else {
					stats.spin_ns += now - spin_start;
					if (gate.HasCold() && gate.MaybeWakeCold(window.Unclaimed())) {
						on_cold_wake();
					}
					prepare(*slot, seq);
					stats.stolen++;
					until_check = 0; // look for the next one at once
					spin_start  = WorkerNowNs();
					continue;
				}
			}
		}
		relax(spins, spin_start);
	}
	stats.spin_ns += WorkerNowNs() - spin_start;
	return stats;
}

} // namespace Libs::Graphics::DrawPrep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_WORKERGATE_H_
