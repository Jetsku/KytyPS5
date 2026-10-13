#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

namespace Libs::Graphics::CpGaps {

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
	if (ledger.Overflows() != 0) {
		std::snprintf(text, sizeof(text), " | overflows %" PRIu64, ledger.Overflows());
		line += text;
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
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
	std::printf("CpGaps: command-processor time accounting on (KYTY_CP_GAP_STATS), TSC %.3f GHz\n",
	            state.ticks_per_ns);
	std::fflush(stdout);
}

void NoteFrame() {
	auto* state = t_state;
	if (state == nullptr) {
		return;
	}
	auto& report = GetReport();
	report.frames++;
	const auto now_ns = SteadyNs();
	// KYTY_CP_GAP_STATS_INTERVAL_MS (default 10000): the line's period (shorter for live A/B slots).
	static const uint64_t interval_ns = [] {
		const auto* value = std::getenv("KYTY_CP_GAP_STATS_INTERVAL_MS");
		const auto  ms    = value != nullptr ? std::strtoull(value, nullptr, 10) : 10000ull;
		return std::clamp<uint64_t>(ms, 250, 600000) * 1'000'000ull;
	}();
	if (now_ns - report.start_ns >= interval_ns) {
		Print(*state, report, now_ns, Tsc());
	}
}

} // namespace Libs::Graphics::CpGaps
