#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

#include "common/logging/log.h"

#include "common/hangTrace.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Libs::Graphics::CpGaps {

static uint64_t QueryCycles() noexcept {
#if defined(_WIN32)
	ULONG64 cycles = 0;
	QueryThreadCycleTime(GetCurrentThread(), &cycles);
	return cycles;
#else
	return 0;
#endif
}

// The calling thread's user + kernel CPU time in ns.
static uint64_t ThreadCpuNs() noexcept {
#if defined(_WIN32)
	FILETIME c {}, e {}, k {}, u {};
	if (!GetThreadTimes(GetCurrentThread(), &c, &e, &k, &u)) return 0;
	const auto t = [](const FILETIME& f) { return (static_cast<uint64_t>(f.dwHighDateTime) << 32u) | f.dwLowDateTime; };
	return (t(k) + t(u)) * 100u;
#else
	return 0;
#endif
}

const char* CatName(Cat cat) noexcept {
	static constexpr const char* names[] = {
	    "commit",      "headself",    "headwait.barrier", "headwait.start", "headwait.shallow",
	    "headwait.deep", "drawop",     "disp.program", "prog.prepare",
	    "prog.key",    "prog.materialize", "prog.permutation", "disp.pipeline",
	    "disp.bind",   "disp.bda",    "disp.rebind", "disp.emit", "disp.other",   "eop",
	    "eventwrite",  "writedata",   "dma",       "wait",       "flip",         "otherop",
	    "resolver",    "starved",     "service",   "slice",      "flush",        "gc",
	    "gpuwait.drain", "gpuwait.occlusion", "gpuwait.other", "recorder", "submitwait",
	    "texupload",   "protect",     "fault",     "idle",       "blocked"};
	static_assert(sizeof(names) / sizeof(names[0]) == CatCount, "one name per category");
	const auto index = static_cast<size_t>(cat);
	return index < CatCount ? names[index] : "?";
}

namespace {

uint64_t SteadyNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct Report {
	uint64_t start_ns  = 0;
	uint64_t start_tsc = 0;
	uint64_t frames    = 0;
};

std::array<std::atomic<uint64_t>, StopCount> g_stops {};
std::mutex                                  g_stop_regions_mutex;
std::unordered_map<uint64_t, uint64_t>      g_stop_regions; // 2 MiB region -> register reads

Report& GetReport() {
	static Report report; // GPU thread only
	return report;
}

void Print(ThreadState& state, Report& report, uint64_t now_ns, uint64_t now_tsc) {
	auto& ledger = state.ledger;
	ledger.Flush(now_tsc);
	const auto   wall_ticks = now_tsc - report.start_tsc;
	const double seconds    = static_cast<double>(now_ns - report.start_ns) * 1e-9;
	state.ticks_per_ns      = static_cast<double>(wall_ticks) /
	                     static_cast<double>(std::max<uint64_t>(1, now_ns - report.start_ns));
	const double frames  = static_cast<double>(std::max<uint64_t>(1, report.frames));
	const double ms_tick = 1e-6 / state.ticks_per_ns / frames; // ms per frame per tick
	uint64_t     covered = 0;
	for (size_t i = 0; i < CatCount; i++) {
		covered += ledger.Ticks(static_cast<Cat>(i));
	}
	// A commit's time includes what is nested in it (GPU waits, uploads, faults inside commits).
	auto commit = ledger.Ticks(Cat::Commit);
	for (size_t i = 0; i < CatCount; i++) {
		if (static_cast<Cat>(i) != Cat::Commit) {
			commit += ledger.InCommit(static_cast<Cat>(i));
		}
	}
	const auto loop = wall_ticks > covered ? wall_ticks - covered : 0;
	std::string line;
	char        text[256];
	std::snprintf(text, sizeof(text),
	              "CpGaps %.0fs: frames %" PRIu64 " wall %.2f ms/frame | commit %.2f (%" PRIu64
	              "/f) | gap %.2f =",
	              seconds, report.frames, static_cast<double>(wall_ticks) * ms_tick,
	              static_cast<double>(commit) * ms_tick,
	              static_cast<uint64_t>(static_cast<double>(ledger.Calls(Cat::Commit)) / frames),
	              static_cast<double>(wall_ticks - std::min(wall_ticks, commit)) * ms_tick);
	line += text;
	// The gap's categories by time (outside commits), then the nested ones' share inside commits.
	std::array<std::pair<uint64_t, Cat>, CatCount + 1> order {};
	size_t                                              count = 0;
	for (size_t i = 0; i < CatCount; i++) {
		const auto cat = static_cast<Cat>(i);
		if (cat == Cat::Commit) {
			continue;
		}
		const auto outside = ledger.Ticks(cat) - std::min(ledger.Ticks(cat), ledger.InCommit(cat));
		order[count++]     = {outside, cat};
	}
	std::sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(count),
	          [](const auto& a, const auto& b) { return a.first > b.first; });
	for (size_t i = 0; i < count; i++) {
		const auto [ticks, cat] = order[i];
		if (ledger.Calls(cat) == 0 && ticks == 0) {
			continue;
		}
		std::snprintf(text, sizeof(text), " %s %.3f (%.1f/f)", CatName(cat),
		              static_cast<double>(ticks) * ms_tick,
		              static_cast<double>(ledger.Calls(cat)) / frames);
		line += text;
	}
	std::snprintf(text, sizeof(text), " loop %.3f | in commit:", static_cast<double>(loop) * ms_tick);
	line += text;
	for (size_t i = 0; i < CatCount; i++) {
		const auto cat = static_cast<Cat>(i);
		if (cat != Cat::Commit && ledger.InCommit(cat) != 0) {
			std::snprintf(text, sizeof(text), " %s %.3f", CatName(cat),
			              static_cast<double>(ledger.InCommit(cat)) * ms_tick);
			line += text;
		}
	}
	static constexpr const char* event_names[] = {"write",      "current",     "unbounded", "other",
	                                              "cp-thread", "side-flushes", "side-flush-copies"};
	static_assert(sizeof(event_names) / sizeof(event_names[0]) == EventCount, "one name per event");
	line += " | drains/f:";
	for (size_t i = 0; i < EventCount; i++) {
		std::snprintf(text, sizeof(text), " %s %.2f", event_names[i],
		              static_cast<double>(state.events[i]) / frames);
		line += text;
	}
	state.events = {};
	static constexpr const char* stop_names[] = {"wait-self", "wait-other", "flip-wait", "predication",
	                                             "cond-exec", "branch",     "reg-read",  "cmd-bytes",
	                                             "pending-write", "reg-read-fine-clean",
	                                             "by-buffer-dirty", "by-tracker-gpu", "by-image-gpu",
	                                             "by-publication"};
	static_assert(sizeof(stop_names) / sizeof(stop_names[0]) == StopCount, "one name per stop");
	line += " | stops/f:";
	for (size_t i = 0; i < StopCount; i++) {
		std::snprintf(text, sizeof(text), " %s %.1f", stop_names[i],
		              static_cast<double>(g_stops[i].exchange(0, std::memory_order_relaxed)) / frames);
		line += text;
	}
	{
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		{
			std::scoped_lock lock(g_stop_regions_mutex);
			regions.assign(g_stop_regions.begin(), g_stop_regions.end());
			g_stop_regions.clear();
		}
		std::sort(regions.begin(), regions.end(),
		          [](const auto& a, const auto& b) { return a.second > b.second; });
		if (!regions.empty()) line += " | reg-read regions/f:";
		for (size_t i = 0; i < std::min<size_t>(regions.size(), 4); i++) {
			std::snprintf(text, sizeof(text), " 0x%" PRIx64 " %.1f", regions[i].first << 21u,
			              static_cast<double>(regions[i].second) / frames);
			line += text;
		}
	}
	if (ledger.Overflows() != 0) {
		std::snprintf(text, sizeof(text), " | overflows %" PRIu64, ledger.Overflows());
		line += text;
	}
	if (CpuEnabled()) {
		static uint64_t last_cpu_ns = 0;
		const auto      cpu_ns      = ThreadCpuNs();
		std::snprintf(text, sizeof(text), " | cp-cpu %.2f ms/f, cpu/wall incl:",
		              last_cpu_ns != 0 ? static_cast<double>(cpu_ns - last_cpu_ns) * 1e-6 / frames : 0.0);
		line += text;
		last_cpu_ns = cpu_ns;
		for (size_t i = 0; i < CatCount; i++) {
			if (state.cpu_wall[i] == 0) continue;
			std::snprintf(text, sizeof(text), " %s %.2f/%.2f", CatName(static_cast<Cat>(i)),
			              static_cast<double>(state.cpu_cycles[i]) * ms_tick, static_cast<double>(state.cpu_wall[i]) * ms_tick);
			line += text;
		}
		state.cpu_cycles = {};
		state.cpu_wall   = {};
	}
	// The console and the emulator log (the launcher's console is not kept).
	Log::WriteToConsoleAndLog(line + "\n");
	ledger.ResetTotals();
	report.start_ns  = now_ns;
	report.start_tsc = now_tsc;
	report.frames    = 0;
}

} // namespace

void AttachThread() {
	if (!Enabled() || t_state != nullptr) {
		return;
	}
	static ThreadState state; // the GPU thread's, for the process lifetime
	// A first TSC rate (refined at every report).
	const auto ns0  = SteadyNs();
	const auto tsc0 = Tsc();
	while (SteadyNs() - ns0 < 2'000'000u) {
	}
	const auto ns1     = SteadyNs();
	const auto tsc1    = Tsc();
	state.ticks_per_ns = static_cast<double>(tsc1 - tsc0) /
	                     static_cast<double>(std::max<uint64_t>(1, ns1 - ns0));
	auto& report       = GetReport();
	report.start_ns    = ns1;
	report.start_tsc   = tsc1;
	t_state            = &state;
	g_thread_cycles    = &QueryCycles;
	std::printf("CpGaps: command-processor time accounting on (KYTY_CP_GAP_STATS), TSC %.3f GHz\n",
	            state.ticks_per_ns);
	std::fflush(stdout);
}

void NoteStopSlow(Stop stop, uint64_t address) noexcept {
	g_stops[static_cast<size_t>(stop)].fetch_add(1, std::memory_order_relaxed);
	if (stop == Stop::RegisterRead) {
		std::scoped_lock lock(g_stop_regions_mutex);
		g_stop_regions[address >> 21u]++;
	}
}

// KYTY_CP_GAP_SLOW_MS=N (diagnostic, default 0 = off; needs KYTY_CP_GAP_STATS=1): a command-processor
// frame (guest frame boundary to the next) of N ms or more prints one line, "CpGaps slow frame", with
// its start on the frame-times clock (KYTY_FRAME_TIMES_LOG's t_us, HangTrace::NowNs) and the
// categories that took its time (exclusive ms, largest first; the nested ones also inside commits).
static void NoteSlowFrame(ThreadState& state, uint64_t now_ns) {
	static const uint64_t slow_ns = [] {
		const auto* value = std::getenv("KYTY_CP_GAP_SLOW_MS");
		return value != nullptr ? std::strtoull(value, nullptr, 10) * 1'000'000ull : 0ull;
	}();
	if (slow_ns == 0) {
		return;
	}
	static uint64_t                       last_ns = 0;
	static std::array<uint64_t, CatCount> last_ticks {};
	static std::array<uint64_t, CatCount> last_in_commit {};
	auto&                                 ledger = state.ledger;
	if (now_ns == 0) {
		// Print reset the totals: the next frame's deltas start from zero (its earlier part is lost).
		last_ticks     = {};
		last_in_commit = {};
		return;
	}
	const auto                            frame_ns = last_ns != 0 ? now_ns - last_ns : 0;
	if (frame_ns >= slow_ns) {
		ledger.Flush(Tsc());
		std::array<std::pair<uint64_t, Cat>, CatCount> order {};
		for (size_t i = 0; i < CatCount; i++) {
			const auto cat = static_cast<Cat>(i);
			order[i]       = {ledger.Ticks(cat) - std::min(ledger.Ticks(cat), last_ticks[i]), cat};
		}
		std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
		const double ms_tick = 1e-6 / state.ticks_per_ns;
		std::string  line;
		char         text[160];
		std::snprintf(text, sizeof(text), "CpGaps slow frame t=%.3f s: %.1f ms |",
		              static_cast<double>(HangTrace::NowNs() - frame_ns) * 1e-9, static_cast<double>(frame_ns) * 1e-6);
		line += text;
		for (size_t i = 0; i < 8 && order[i].first != 0; i++) {
			std::snprintf(text, sizeof(text), " %s %.1f", CatName(order[i].second),
			              static_cast<double>(order[i].first) * ms_tick);
			line += text;
		}
		line += " | in commit:";
		for (size_t i = 0; i < CatCount; i++) {
			const auto delta = ledger.InCommit(static_cast<Cat>(i)) - std::min(ledger.InCommit(static_cast<Cat>(i)), last_in_commit[i]);
			if (static_cast<double>(delta) * ms_tick >= 1.0) {
				std::snprintf(text, sizeof(text), " %s %.1f", CatName(static_cast<Cat>(i)),
				              static_cast<double>(delta) * ms_tick);
				line += text;
			}
		}
		std::printf("%s\n", line.c_str());
	}
	last_ns = now_ns;
	for (size_t i = 0; i < CatCount; i++) {
		last_ticks[i]     = ledger.Ticks(static_cast<Cat>(i));
		last_in_commit[i] = ledger.InCommit(static_cast<Cat>(i));
	}
}

void NoteFrame() {
	auto* state = t_state;
	if (state == nullptr) {
		return;
	}
	auto& report = GetReport();
	report.frames++;
	const auto now_ns = SteadyNs();
	NoteSlowFrame(*state, now_ns);
	// KYTY_CP_GAP_STATS_INTERVAL_MS (default 10000): the line's period (shorter for live A/B slots).
	static const uint64_t interval_ns = [] {
		const auto* value = std::getenv("KYTY_CP_GAP_STATS_INTERVAL_MS");
		const auto  ms    = value != nullptr ? std::strtoull(value, nullptr, 10) : 10000ull;
		return std::clamp<uint64_t>(ms, 250, 600000) * 1'000'000ull;
	}();
	if (now_ns - report.start_ns >= interval_ns) {
		Print(*state, report, now_ns, Tsc());
		NoteSlowFrame(*state, 0); // the totals were reset: restart the slow-frame baseline
	}
}

} // namespace Libs::Graphics::CpGaps
