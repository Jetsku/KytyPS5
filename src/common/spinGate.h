#ifndef KYTY_COMMON_SPIN_GATE_H_
#define KYTY_COMMON_SPIN_GATE_H_

// KYTY_SPIN_GATE=1 (default off): helper threads that spin for new work before they block (the
// command recorder's idle wait for work, the CP sequencer's wait for its resolver) spin only while
// the PC keeps CPUs idle. A spin spares the producer a wake-up system call per hand-off, but on a
// PC with few cores it takes CPU time the game's own threads need (chenxiao07/KytyPS5 de192903c:
// held to 4 cores, 43-50 fps with 1% lows of 3-5 fps spinning, 57-63 fps without). The gate
// measures the system's idle CPUs (GetSystemTimes, spinning included) at most every 250 ms: it
// closes below an average of three idle CPUs and opens again from five. Closed, those waits block
// at once. Windows only (always open elsewhere). Adapted from chenxiao07/KytyPS5 de192903c.
//
// KYTY_SPIN_BUDGET=1 (live, default off; KYTY_CORE_BUDGET_PROFILE turns it on when unset): the
// gate decides from the core budget (common/coreBudget.h: the processors this process may use),
// not from the machine's idle time, which never closes under an affinity mask (a process held to
// 6 of 32 cores still sees ~20 idle CPUs) and misjudges small CPUs. Every spinning site asks it:
// the sequencer's wait (WaitSlow), the recorder's wait for work (Ring::WaitPublished), the commit
// thread's wait for a worker-held head and for the recorder (drain, recorded tick), the resolver's
// wait for sequencer ops, and the draw-prep workers' idle spin (gate pool; KYTY_DRAW_PREP_ADAPT
// workers already spin only 10 us).
// - A monitor thread (below normal priority, cold / general processors: never the CP's reserved
//   processor, never preempting it) measures every KYTY_SPIN_BUDGET_INTERVAL_MS (default 250) the
//   registered spinner threads, the guest and cold threads the placement keeps handles of
//   (QueryThreadCycleTime on kept handles, minus the spin the sites noted) and the process total
//   (QueryProcessCycleTime); what the measured threads do not explain counts as that many busy
//   threads (rounded). No system-wide thread snapshot: the scan costs microseconds (its cost is on
//   the log line). A thread busy at least KYTY_SPIN_BUDGET_BUSY_PCT (default 25) % of the interval
//   counts as busy (the game's own threads and ours alike); a registered spinner that is not busy
//   but spun at least 5 % counts as spinning.
// - The gate (GateFilter) decides on the mean demand of the last KYTY_SPIN_BUDGET_WINDOW (4)
//   intervals and holds each state at least KYTY_SPIN_BUDGET_HOLD_CLOSED_MS (2000) /
//   KYTY_SPIN_BUDGET_HOLD_OPEN_MS (1000). It closes when busy + spinning threads +
//   KYTY_SPIN_BUDGET_MARGIN (default 0) exceed the budget's logical processors; it opens again when
//   busy threads + every registered spinner that is not busy (as if they all spun) + margin + 1 fit
//   (backing off never reopens it).
// - Closed, every site spins at most KYTY_SPIN_BUDGET_SHORT_US (default 2) and then really waits:
//   the sequencer, the recorder and the commit thread's recorder waits park as they do after their
//   long spin; the commit thread parks on the held head's state word (the worker's Complete wakes
//   it; drawPrep/window.h); the resolver gives its processor to ready threads (SwitchToThread)
//   between looks until its own 2 ms exit.
// KYTY_SPIN_BUDGET_SITES (live, default 0x3F): which sites the closed gate caps (bit per Site:
// 1 sequencer, 2 recorder, 4 commit-head, 8 recorder-drain, 16 resolver, 32 worker); the others
// spin as with the gate open. KYTY_SPIN_BUDGET_SHORT_US and KYTY_SPIN_BUDGET_PIPE_SHORT_US
// (sequencer and recorder; default: the same) are live too.
// Always-on counters per site (waits, spin time, parks; cheap relaxed adds per wait, not per spin
// iteration) and one "SpinBudget 10s" line every KYTY_SPIN_STATS_PERIOD_S (default 10, 0 = never)
// with the gate state, the budget and the demand, also with every switch off.

#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace Common::SpinGate {

[[nodiscard]] bool Enabled();
// Whether a helper thread may spin for new work now (always true with the switch off).
[[nodiscard]] bool SpinAllowed();
// Times the gate closed since start-up.
[[nodiscard]] uint64_t Shuts();

// The spinning sites (KYTY_SPIN_BUDGET).
enum class Site : uint8_t {
	Sequencer,     // CpSeq::Sequencer::WaitSlow
	Recorder,      // Ring::WaitPublished (the recorder thread's wait for work)
	CommitHead,    // the commit thread's wait for a worker-held head (DrawPrep AwaitHead)
	RecorderDrain, // the commit thread's waits on the recorder (drain, recorded tick)
	Resolver,      // the resolver's wait for sequencer ops (starved)
	Worker,        // draw-prep gate-pool workers' idle spin
	Count
};
[[nodiscard]] const char* SiteName(Site site);

// The spin a site may do now before it waits: `wanted_ns` while both gates are open (or off), 0
// while KYTY_SPIN_GATE is closed, at most KYTY_SPIN_BUDGET_SHORT_US while KYTY_SPIN_BUDGET is
// closed. Registers the calling thread as a spinner of `site` (once per thread) and counts a wait.
[[nodiscard]] uint64_t SpinNs(Site site, uint64_t wanted_ns);
// KYTY_SPIN_BUDGET is on and closed now: sites should really wait after their short spin.
[[nodiscard]] bool BudgetClosed();
// KYTY_SPIN_BUDGET's switch value (profile applied).
[[nodiscard]] bool BudgetEnabled();
// KYTY_SPIN_BUDGET was set at startup (to any value) or the profile switch was: the waiting
// protocols that need a startup decision (the draw-prep head wake) are prepared.
[[nodiscard]] bool BudgetAvailable();
// The calling thread spun `tsc` TSC ticks in one wait of `site`.
void NoteSpin(Site site, uint64_t tsc);
// The calling thread's wait at `site` parked (blocked, or gave its processor away).
void NotePark(Site site, uint64_t count = 1);
// Gives the processor to a ready thread (SwitchToThread; std::this_thread::yield elsewhere). True
// when another thread ran.
bool GiveWay();

[[nodiscard]] inline uint64_t Tsc() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return 0;
#endif
}

// ---- Pure decision (tested) ----
struct GateInput {
	uint32_t budget_logical = 0;   // the budget's logical processors
	double   busy           = 0.0; // threads busy (spin excluded) >= the busy share
	double   spinning       = 0.0; // registered spinners not busy that spun >= 5 %
	double   idle_spinners  = 0.0; // registered spinners not busy (spinning or not)
	uint32_t margin         = 0;
};
// The gate's next state from its current one (one decision, no smoothing or hold).
[[nodiscard]] bool NextBudgetGateOpen(bool open, const GateInput& input);

// The gate with hysteresis (the monitor uses it): each interval's demand goes into a window of the
// last `window` intervals and the decision (NextBudgetGateOpen: close above the budget with the
// spinning threads, reopen only when every idle spinner + 1 fits) reads their average; a state is
// kept at least `hold_closed_ms` / `hold_open_ms` (KYTY_SPIN_BUDGET_WINDOW 4,
// KYTY_SPIN_BUDGET_HOLD_CLOSED_MS 2000, KYTY_SPIN_BUDGET_HOLD_OPEN_MS 1000). One interval of
// demand above the budget no longer flips it (at 6 cores: 161 closes in 6 minutes before).
struct GateConfig {
	uint32_t window         = 4;
	uint64_t hold_closed_ms = 2000;
	uint64_t hold_open_ms   = 1000;
};
class GateFilter {
public:
	explicit GateFilter(const GateConfig& config) noexcept: m_config(config) {
		if (m_config.window == 0 || m_config.window > MaxWindow) {
			m_config.window = 4;
		}
	}
	// One interval ending at `now_ms`; returns whether the gate is open after it.
	bool Step(uint64_t now_ms, const GateInput& input) noexcept {
		m_ring[m_next % m_config.window] = input;
		m_next++;
		const auto count = static_cast<uint32_t>(m_next < m_config.window ? m_next : m_config.window);
		GateInput  mean   = input;
		mean.busy = mean.spinning = mean.idle_spinners = 0.0;
		for (uint32_t i = 0; i < count; i++) {
			mean.busy += m_ring[i].busy / count;
			mean.spinning += m_ring[i].spinning / count;
			mean.idle_spinners += m_ring[i].idle_spinners / count;
		}
		m_mean = mean;
		const auto held = now_ms - m_since_ms;
		if (m_open ? held < m_config.hold_open_ms : held < m_config.hold_closed_ms) {
			return m_open;
		}
		const bool next = NextBudgetGateOpen(m_open, mean);
		if (next != m_open) {
			m_open     = next;
			m_since_ms = now_ms;
			m_closes += next ? 0u : 1u;
		}
		return m_open;
	}
	[[nodiscard]] bool      Open() const noexcept { return m_open; }
	[[nodiscard]] uint64_t  Closes() const noexcept { return m_closes; }
	[[nodiscard]] GateInput Mean() const noexcept { return m_mean; }

private:
	static constexpr uint32_t MaxWindow = 32;
	GateConfig m_config;
	GateInput  m_ring[MaxWindow] {};
	GateInput  m_mean {};
	uint64_t   m_next     = 0;
	uint64_t   m_since_ms = 0;
	uint64_t   m_closes   = 0;
	bool       m_open     = true;
};

struct ThreadSample {
	uint64_t cycles     = 0; // CPU cycles over the interval (TSC units)
	uint64_t spin       = 0; // noted spin over the interval (TSC units)
	bool     registered = false;
};
struct Demand {
	uint32_t busy          = 0;
	uint32_t spinning      = 0;
	uint32_t idle_spinners = 0;
	double   busy_cores    = 0.0; // all threads' cycles (spin excluded) / interval
	double   spin_cores    = 0.0; // registered threads' spin / interval
};
// Classifies the threads of one interval of `interval_tsc` ticks (busy_share: 0.25 for 25 %).
[[nodiscard]] Demand ClassifyThreads(const ThreadSample* samples, uint32_t count,
                                     uint64_t interval_tsc, double busy_share);
// The threads the scan does not measure one by one (driver threads, other emulator threads): the
// process's cycles minus the measured threads' over the interval, counted as that many busy
// threads (rounded) and added to the busy cores.
void AddUnmeasured(Demand& demand, uint64_t process_cycles, uint64_t measured_cycles,
                   uint64_t interval_tsc);

} // namespace Common::SpinGate

#endif /* KYTY_COMMON_SPIN_GATE_H_ */
