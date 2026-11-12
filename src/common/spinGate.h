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
// - A monitor thread (below normal priority, cold processors) measures every
//   KYTY_SPIN_BUDGET_INTERVAL_MS (default 250) each thread of the process: its CPU cycles
//   (QueryThreadCycleTime) minus the spin time the sites above noted for it. A thread busy at least
//   KYTY_SPIN_BUDGET_BUSY_PCT (default 25) % of the interval counts as a busy thread (the game's
//   own threads and ours alike); a registered spinner thread that is not busy but spun at least 5 %
//   counts as a spinning thread.
// - The gate closes when busy + spinning threads + KYTY_SPIN_BUDGET_MARGIN (default 0) exceed the
//   budget's logical processors; it opens again when busy threads + every registered spinner that
//   is not busy (as if they all spun) + margin + 1 fit (hysteresis: backing off never reopens it).
// - Closed, every site spins at most KYTY_SPIN_BUDGET_SHORT_US (default 2) and then really waits:
//   the sequencer, the recorder and the commit thread's recorder waits park as they do after their
//   long spin; the commit thread parks on the held head's state word (the worker's Complete wakes
//   it; drawPrep/window.h); the resolver gives its processor to ready threads (SwitchToThread)
//   between looks until its own 2 ms exit.
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
	uint32_t budget_logical = 0; // the budget's logical processors
	uint32_t busy           = 0; // threads busy (spin excluded) >= the busy share
	uint32_t spinning       = 0; // registered spinners not busy that spun >= 5 %
	uint32_t idle_spinners  = 0; // registered spinners not busy (spinning or not)
	uint32_t margin         = 0;
};
// The gate's next state from its current one.
[[nodiscard]] bool NextBudgetGateOpen(bool open, const GateInput& input);

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

} // namespace Common::SpinGate

#endif /* KYTY_COMMON_SPIN_GATE_H_ */
