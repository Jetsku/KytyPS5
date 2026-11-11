#include "common/kytyEvents.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
void Check(bool ok, const char* message, const std::string& text = {}) {
	if (!ok) {
		std::fprintf(stderr, "KytyEventsTests: failed: %s\n%s\n", message, text.c_str());
		std::abort();
	}
}
std::string Join(const std::vector<std::string>& lines) {
	std::string out;
	for (const auto& line: lines) out += line + "\n";
	return out;
}
bool Has(const std::vector<std::string>& lines, const char* part) {
	return Join(lines).find(part) != std::string::npos;
}

void GuestLines() {
	const auto lines = KytyEvents::SplitForTest(1, "GAME: Level has started: x\r\nsecond\npart", 42, 1.5);
	Check(lines.size() == 3, "two lines and a partial", Join(lines));
	Check(lines[0] == "GUEST: frame=42 t=1.500 | GAME: Level has started: x", "line format", lines[0]);
	Check(lines[1] == "GUEST: frame=42 t=1.500 | second", "second line", lines[1]);
	Check(lines[2] == "partial:part", "partial kept", lines[2]);
	const auto err = KytyEvents::SplitForTest(2, "oops\n", 0, 0.0);
	Check(err.size() == 1 && err[0] == "GUEST: frame=0 t=0.000 | [stderr] oops", "stderr tag", Join(err));
}

void LoadingAndIdle() {
	KytyEvents::Params params;
	KytyEvents::Detector d(params);
	uint64_t             t = 0, frames = 0, bytes = 0, started = 0;
	std::vector<std::string> all;
	auto step = [&](uint64_t mbps, uint64_t fps, int64_t in_flight) {
		t += 250;
		frames += fps / 4;
		bytes += mbps * 1048576 / 4;
		auto lines = d.Step(t, frames, bytes, started, in_flight);
		all.insert(all.end(), lines.begin(), lines.end());
		return lines;
	};
	for (int i = 0; i < 8; i++) step(100, 0, 0); // boot read, no flips
	Check(Has(all, "KYTY_EVENT: loading-start"), "loading-start on heavy reads", Join(all));
	Check(!Has(all, "first-flip"), "no flip yet", Join(all));
	started = 100;
	for (int i = 0; i < 8; i++) step(100, 60, 5); // menu flipping while still reading
	Check(Has(all, "KYTY_EVENT: first-flip frame=15"), "first-flip", Join(all));
	Check(!Has(all, "loading-end"), "still reading", Join(all));
	for (int i = 0; i < 8; i++) step(2, 60, 0); // quiet, but within the hold
	Check(!Has(all, "loading-end"), "hold not yet over", Join(all));
	for (int i = 0; i < 20; i++) step(2, 60, 0);
	Check(Has(all, "KYTY_EVENT: loading-end"), "loading-end after the hold", Join(all));
	Check(Has(all, "KYTY_EVENT: shader-compile-idle"), "compile idle after 3 s", Join(all));
	const auto idle_count = Join(all).find("shader-compile-idle");
	for (int i = 0; i < 20; i++) step(2, 60, 0);
	Check(Join(all).find("shader-compile-idle", idle_count + 1) == std::string::npos, "idle reported once per burst",
	      Join(all));
	// A small trickle of builds is not reported.
	started += 10;
	for (int i = 0; i < 20; i++) step(2, 60, 0);
	Check(Join(all).find("shader-compile-idle", idle_count + 1) == std::string::npos, "trickle not reported", Join(all));
	// No flips after the reads (a compile stall) keep the second load open.
	for (int i = 0; i < 8; i++) step(200, 0, 0);
	for (int i = 0; i < 40; i++) step(2, 0, 0);
	const auto text = Join(all);
	Check(text.find("loading-end") == text.rfind("loading-end"), "no flips keep the second load open", text);
	// 4 fps gameplay (Ghost of Yotei after Continue) ends it.
	for (int i = 0; i < 30; i++) step(1, 4, 0);
	const auto after = Join(all);
	Check(after.find("loading-end") != after.rfind("loading-end"), "4 fps closes the second load", after);
}

// The lines of the prepare run (KYTY_PIPELINE_LIST_PREPARE): the same head as the monitor's events.
void CallerEvents() {
	const auto line = KytyEvents::FormatEvent("prepare-progress", "phase=shaders shaders=3/10 pipelines=0/20 compute=0/0");
	Check(line.starts_with("KYTY_EVENT: prepare-progress frame=0 t="), "event head", line);
	Check(line.ends_with(" phase=shaders shaders=3/10 pipelines=0/20 compute=0/0"), "event keys", line);
	Check(line.find('\n') == std::string::npos, "one line, no newline", line);
	const auto bare = KytyEvents::FormatEvent("prepare-done", "");
	Check(bare.starts_with("KYTY_EVENT: prepare-done frame=0 t=") && bare.back() != ' ', "no trailing space", bare);
}
} // namespace

int main() {
	GuestLines();
	LoadingAndIdle();
	CallerEvents();
	std::printf("KytyEventsTests: ok\n");
	return 0;
}
