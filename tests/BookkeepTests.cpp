// KYTY_BK_* (drawPrep/bookkeep.h): the switch parser, the ready run of the per-run hand-over against
// the preparation window (drawPrep/window.h) driven as the commit thread and its workers drive it,
// the single-writer counters and the 10 s line.
#include "graphics/host_gpu/renderer/drawPrep/bookkeep.h"
#include "graphics/host_gpu/renderer/drawPrep/window.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::DrawPrep;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

void TestParse() {
	using Bookkeep::Mode;
	Check(Bookkeep::ParseMode(nullptr) == Mode::Off, "unset: off");
	Check(Bookkeep::ParseMode("") == Mode::Off, "empty: off");
	Check(Bookkeep::ParseMode("0") == Mode::Off, "0: off");
	Check(Bookkeep::ParseMode("1") == Mode::On, "1: on");
	Check(Bookkeep::ParseMode("on") == Mode::On, "on: on");
	Check(Bookkeep::ParseMode("2") == Mode::Verify, "2: verify");
	Check(Bookkeep::ParseMode("verify") == Mode::Verify, "verify: verify");
	Check(Bookkeep::ParseMode("3") == Mode::Off, "unknown: off");
}

void TestReadyRunAlone() {
	Bookkeep::ReadyRun run;
	Check(!run.Covers(0), "empty run covers nothing");
	std::vector<bool> done(16, false);
	done[5] = done[6] = done[7] = true;
	done[9]                     = true;
	const auto at = [&](uint64_t seq) { return seq < done.size() && done[seq]; };
	// Head 4 is ready (whatever its state word says: the caller proved it); 5..7 follow, 8 is not.
	Check(run.Extend(4, 16, 32, at) == 3, "three finished positions after the head");
	Check(run.Covers(4) && run.Covers(5) && run.Covers(7), "head and the finished positions");
	Check(!run.Covers(8) && !run.Covers(9), "the first unfinished position ends the run");
	Check(!run.Covers(3), "nothing before the head");
	// The tail bounds the run.
	Check(run.Extend(4, 6, 32, at) == 1, "the published tail bounds the run");
	Check(run.Covers(5) && !run.Covers(6), "tail excluded");
	// The cap bounds the run.
	std::vector<bool> all(100, true);
	const auto        any = [&](uint64_t seq) { return all[seq]; };
	Check(run.Extend(0, 100, 32, any) == 32, "cap of 32 positions beyond the head");
	Check(run.Covers(32) && !run.Covers(33), "cap edge");
	run.Reset();
	Check(!run.Covers(0) && !run.Covers(5), "reset");
	// A head with nothing after it.
	Check(run.Extend(9, 16, 32, at) == 0, "no finished position after the head");
	Check(run.Covers(9) && !run.Covers(10) && !run.Covers(8), "only the head");
}

// The commit thread's protocol against the window, with one worker preparing published slots: every
// head inside the ready run must be Done for its own position (what Engine::ReadyHead's verify
// checks), and heads are committed in order.
void TestReadyRunWindow() {
	struct Payload {
		uint64_t seq      = 0;
		uint64_t prepared = 0;
	};
	constexpr uint64_t Draws = 200'000;
	Window<Payload>    window(64);
	std::atomic<bool>  stop {false};
	std::thread        worker([&] {
        while (!stop.load(std::memory_order_acquire)) {
            uint64_t seq     = 0;
            auto*    payload = window.TryClaim(seq);
            if (payload == nullptr) {
                std::this_thread::yield();
                continue;
            }
            payload->prepared = payload->seq * 3u + 1u;
            window.Complete(seq);
        }
	});
	Bookkeep::ReadyRun run;
	uint64_t           published  = 0;
	uint64_t           committed  = 0;
	uint64_t           run_heads  = 0;
	uint64_t           mismatches = 0;
	uint64_t           wrong      = 0;
	while (committed < Draws) {
		// Producer: publish while there is room.
		while (published < Draws && !window.Full() && window.Tail() - window.Head() < 48) {
			auto& slot = window.Reserve();
			slot.seq   = published++;
			window.Publish();
		}
		if (window.Empty()) {
			continue;
		}
		const auto head = window.Head();
		auto&      slot = window.HeadPayload();
		if (run.Covers(head)) {
			run_heads++;
			if (!window.HeadDone()) {
				mismatches++;
			}
		} else {
			if (window.TryClaimHead()) {
				slot.prepared = slot.seq * 3u + 1u; // the commit thread prepares it itself
			} else {
				while (!window.HeadDone()) {
					std::this_thread::yield();
				}
			}
			(void)run.Extend(head, window.Tail(), 32, [&window](uint64_t seq) { return window.DoneAt(seq); });
		}
		if (slot.seq != committed || slot.prepared != slot.seq * 3u + 1u) {
			wrong++;
		}
		committed++;
		window.Retire();
	}
	stop.store(true, std::memory_order_release);
	worker.join();
	Check(mismatches == 0, "every head inside a ready run is Done for its position");
	Check(wrong == 0, "heads committed in order with their preparation visible");
	Check(run_heads != 0, "some heads came from ready runs");
	std::printf("ready run window: %llu draws, %llu from runs\n", static_cast<unsigned long long>(Draws),
	            static_cast<unsigned long long>(run_heads));
}

void TestCountersAndLine() {
	Bookkeep::Counters c;
	c.Add(c.heads, 10);
	c.Add(c.run_heads, 7);
	c.Add(c.runs, 2);
	c.Add(c.run_positions, 7);
	c.Add(c.run_kernel, 5);
	c.Fall(Bookkeep::Fallback::RecordNotKernel);
	c.Fall(Bookkeep::Fallback::RecordNotKernel);
	Bookkeep::ReportMismatch(c.adopt_mismatches, "adopt", "test", 0);
	Check(c.heads.load() == 10 && c.run_heads.load() == 7, "single-writer adds");
	Check(c.fallbacks[static_cast<size_t>(Bookkeep::Fallback::RecordNotKernel)].load() == 2, "fallback by reason");
	Check(c.adopt_mismatches.load() == 1, "mismatch counted");
	const auto line = Bookkeep::FormatLine(c, Bookkeep::Mode::On, Bookkeep::Mode::Verify, Bookkeep::Mode::Off,
	                                       Bookkeep::Mode::On);
	Check(line.find("handover on heads 10 from runs 7 (70.0%, kernel 5)") != std::string::npos, "handover part");
	Check(line.find("runs 2 (3.50 positions/run)") != std::string::npos, "positions per run");
	Check(line.find("adopt verify") != std::string::npos && line.find("record off") != std::string::npos,
	      "modes named");
	Check(line.find("record_not_kernel=2") != std::string::npos, "fallback names");
	Check(line.find("mismatches total=1") != std::string::npos, "mismatch total");
}

} // namespace

int main() {
	TestParse();
	TestReadyRunAlone();
	TestReadyRunWindow();
	TestCountersAndLine();
	if (g_failures != 0) {
		std::fprintf(stderr, "bookkeep_tests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("bookkeep_tests: all passed\n");
	return 0;
}
