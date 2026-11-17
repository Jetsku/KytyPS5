#ifndef EMULATOR_SRC_COMMON_LIVESWITCH_H_
#define EMULATOR_SRC_COMMON_LIVESWITCH_H_

#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>

// Live switches: KYTY_* flags that can change while the game runs, for same-process A/B
// measurement: an external harness writes the file named by KYTY_LIVE_FILE (see below).
//
// A flag that may change at runtime is a Live::Switch with static storage duration, defined at
// namespace scope so that it registers before main:
//
//   static Live::Switch g_idle_flush_draws("KYTY_IDLE_FLUSH_DRAWS", [](const char* value) -> int64_t {
//       return value != nullptr ? std::min(std::strtoul(value, nullptr, 10), 65536ul) : 8;
//   });
//   ...
//   if (g_idle_flush_draws.Get() == 0) { ... }
//
// The parser maps the variable's text (nullptr: unset or empty) to an int64_t. It runs once for the
// startup environment and again for every live change, so a flag reads the same with or without
// live control. Get() is one relaxed atomic load; read it where the value is used, not into a
// function-local static, or the switch never changes.
//
// KYTY_LIVE_FILE=<path> (default: no live control, nothing runs): a low-priority thread checks the
// file's time and size every 250 ms and, when they change, reads it: KEY=VALUE lines, # comments,
// KEY= means unset (the flag's default, as if the variable were absent), KYTY_LIVE_SEQ=<n> is a
// sequence number echoed in the log. The file is a desired state: keys it does not mention keep
// their value. The changes are applied on the command-processor thread at the next guest flip, after
// the flip's commands were flushed (CommandProcessor::ExecFlip / PrepareCpuFlip -> OnCpFlip), and
// every application is logged with the flip, the hang trace's time base (summary.csv t_ms, and
// t_us) and the command processor's busy time so far (cp_busy_us: time inside GuestGpu::Process,
// the hang trace's gfx_busy + compute_busy), so the frames, seconds and CP time between two
// applications are exact:
//   Live: seq 3 applied at CP flip 5123 (t_ms 183456, t_us 183456789, cp_busy_us 170123456): KYTY_IDLE_FLUSH_DRAWS=0 (8 -> 0)
//   Live: seq 4 applied at CP flip 5409 (t_ms 193461, t_us 193461012, cp_busy_us 179451234): no change
//   Live: seq 5: KYTY_FOO is not a live switch in this build (ignored)
//
// The flip is the frame boundary of the command processor (the resolver thread with KYTY_CP_SEQ=1).
// A switch read on other threads (the sequencer front, draw-prep workers, the recorder, guest
// threads) sees the new value at its next read, which may be a little before or after its own
// frame boundary: fine for parameters (thresholds, batch sizes, spins) and per-operation choices
// that are valid at any time; anything that builds state under one value and relies on it under
// the other needs an on-change hook (run on the command-processor thread at the flip, right after
// the store) or must stay a startup flag.
//
// KYTY_LIVE_TEST_CP_SPIN_US=<us> (default 0, live): busy-waits that long on the command-processor
// thread at every guest flip, after the flip's commands were flushed: a known CP cost per flip to
// validate an A/B harness.

namespace Live {

// nullptr: the variable is unset (or empty).
using Parser = int64_t (*)(const char* value);
// Runs on the command-processor thread at the flip that applies a change, after the new value is
// stored; previous != value.
using OnChange = void (*)(int64_t previous, int64_t value);

class Switch {
public:
	// Registers the switch. Its value is parse(getenv(name)), or parse(the live value) when a live
	// change for this name was applied before it registered.
	Switch(const char* name, Parser parse, OnChange on_change = nullptr) noexcept;

	Switch(const Switch&)            = delete;
	Switch& operator=(const Switch&) = delete;

	[[nodiscard]] int64_t Get() const noexcept { return m_value.load(std::memory_order_relaxed); }
	[[nodiscard]] bool    On() const noexcept { return Get() != 0; }
	[[nodiscard]] const char* Name() const noexcept { return m_name; }

private:
	friend struct Registry;
	friend bool ReloadFromEnvironment(const char* name);

	const char*          m_name;
	Parser               m_parse;
	OnChange             m_on_change;
	std::atomic<int64_t> m_value {0};
	Switch*              m_next = nullptr;
};

// Start-up only (main thread, before the threads that read switches start): re-parses the named
// switch(es) from the process environment, without the on-change hook. For start-up code that sets
// variables after the switches registered (KYTY_SMALL_CPU_RULE). False: no such switch.
bool ReloadFromEnvironment(const char* name);

// Common parsers.
// 0 only for "0"; unset and anything else: 1 (flags that default on).
int64_t ParseDefaultOn(const char* value);
// 1 for anything but "0"; unset: 0 (flags that default off).
int64_t ParseDefaultOff(const char* value);

// Starts the live-control thread when KYTY_LIVE_FILE is set (once; later calls do nothing). The
// command processor's thread calls it when it starts.
void Start();

// The command processor's safe point, on its thread at every guest flip: applies the staged
// changes, then the synthetic KYTY_LIVE_TEST_CP_SPIN_US wait.
void OnCpFlip();

// Around every GuestGpu::Process slice on the command processor's thread: its busy time for the log
// lines. A relaxed load and nothing else without KYTY_LIVE_FILE.
void CpSliceBegin() noexcept;
void CpSliceEnd() noexcept;

// Whether KYTY_LIVE_FILE live control runs.
[[nodiscard]] bool Enabled() noexcept;

// Cumulative GPU execution totals since process start (GpuTiming, KYTY_GPU_TIMING=1 or the hang
// trace). Once a provider is set, every application line also carries them after the CP part, so a
// same-process A/B reads GPU ms per flip per slot as well:
//   Live: seq 3 applied at CP flip 5123 (t_ms .., t_us .., cp_busy_us ..) gpu [busy_us 1, idle_us 2,
//   starved_us 3, cmdbufs 4]: ...
// The totals lag the CP by the GPU's completion latency (one or two flips).
struct GpuTotals {
	uint64_t busy_ns    = 0;
	uint64_t idle_ns    = 0;
	uint64_t starved_ns = 0;
	uint64_t cmdbufs    = 0;
};
void SetGpuTotalsProvider(GpuTotals (*provider)()) noexcept;

// wholecp measurement: cumulative CP idle buckets in TSC ticks (PrepStats::CpIdle) plus the TSC at the
// line, so a reader converts with the TSC rate between two lines:
//   ... cpidle [tsc T, nowork N, empty E, blocked B, fence F, starved S]: ...
struct CpIdleTotals {
	uint64_t tsc     = 0;
	uint64_t nowork  = 0;
	uint64_t empty   = 0;
	uint64_t blocked = 0;
	uint64_t fence   = 0;
	uint64_t starved = 0;
};
void SetCpIdleProvider(CpIdleTotals (*provider)()) noexcept;

// wholecp measurement: extra log lines written right after every application line (argument: the
// line's live sequence number), e.g. the KYTY_HLE_WAIT_STATS per-thread totals.
void SetExtraLinesProvider(std::string (*provider)(uint64_t seq)) noexcept;
// gpuhaz measurement: text appended to every application line after the counts (empty: nothing),
// e.g. " haz [disp N, ..]" cumulative KYTY_GPU_HAZARD counters.
void SetLineSuffixProvider(std::string (*provider)()) noexcept;

// jgcp measurement: event counters printed on every application line with KYTY_LIVE_FILE (once any
// counter is non-zero), so a
// per-event dose (KYTY_LIVE_TEST_*) converts into a cost per flip:
//   ... counts [drawidx D, suspend S, submits B]: ...
// drawidx: indexed draw ops the command processor executed (ExecDrawIndex, the per-draw spin's
// site); suspend: sceAgcSuspendPoint calls; submits: AGC DCB/command-buffer submit calls.
enum class TestCounter : uint32_t { DrawIndexOps, SuspendPoints, AgcSubmits, Count };
// One writer per counter at a time is not required (relaxed add).
void CountTest(TestCounter counter) noexcept;
// Single-writer form (the command processor's thread only): no locked instruction.
void CountTestSingleWriter(TestCounter counter) noexcept;
// Busy-waits `ns` nanoseconds on the calling thread (doses).
void TestSpinNs(int64_t ns) noexcept;

namespace Testing {
// What the live-control thread does with the file's text: parse it and stage the changes.
void StageText(std::string_view text);
[[nodiscard]] bool HasStaged() noexcept;
[[nodiscard]] uint64_t CpFlips() noexcept;
// Receives every log line instead of the console (nullptr: the console again).
void SetLogSink(void (*sink)(const std::string& line));
} // namespace Testing

} // namespace Live

#endif // EMULATOR_SRC_COMMON_LIVESWITCH_H_
