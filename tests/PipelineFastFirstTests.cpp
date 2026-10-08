#include "graphics/host_gpu/renderer/pipeline/pipelineFastFirst.h"

#include <atomic>
#include <barrier>
#include <cstdio>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

static int g_failed = 0;
#define CHECK(expr)                                                                  \
	do {                                                                             \
		if (!(expr)) {                                                               \
			std::printf("PipelineFastFirstTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                              \
		}                                                                            \
	} while (0)

int main() {
	// KYTY_PIPELINE_FAST_FIRST values: a number (0 off), or "nvidia" (on for vendor 0x10de only).
	{
		CHECK(!FastFirstForVendor(nullptr, 0x10deu));
		CHECK(!FastFirstForVendor("", 0x10deu));
		CHECK(!FastFirstForVendor("0", 0x10deu));
		CHECK(FastFirstForVendor("1", 0x1002u));
		CHECK(FastFirstForVendor("nvidia", 0x10deu));
		CHECK(!FastFirstForVendor("nvidia", 0x1002u));
		CHECK(!FastFirstForVendor("nvidia", 0x8086u));
		CHECK(!FastFirstForVendor("amd", 0x1002u));
	}

	// Retire list: nothing expires early, expired entries leave in order, age 0 keeps all.
	{
		FastFirstRetireList<int> list;
		list.Add(1, 100);
		list.Add(2, 200);
		list.Add(3, 300);
		CHECK(list.TakeExpired(150, 0).empty());
		CHECK(list.TakeExpired(250, 100).size() == 1);
		CHECK(list.Size() == 2);
		const auto expired = list.TakeExpired(1000, 100);
		CHECK((expired == std::vector<int> {2, 3}));
		CHECK(list.Size() == 0);
		list.Add(4, 10);
		list.Add(5, 20);
		CHECK(list.TakeExpired(5, 1).empty()); // a clock before the entry never expires it
		const auto all = list.TakeAll();
		CHECK((all == std::vector<int> {4, 5}));
		CHECK(list.Size() == 0);
	}

	// Log gate: one claim per interval, even with several threads racing.
	{
		FastFirstLogGate gate(0, 100);
		CHECK(!gate.TryClaim(99));
		CHECK(gate.TryClaim(100));
		CHECK(!gate.TryClaim(150));
		CHECK(gate.TryClaim(200));
		FastFirstLogGate           racing(0, 1000);
		std::atomic<int>           winners {0};
		std::vector<std::thread>   threads;
		std::barrier               go(8);
		for (int i = 0; i < 8; ++i) {
			threads.emplace_back([&] {
				go.arrive_and_wait();
				if (racing.TryClaim(5000)) ++winners;
			});
		}
		for (auto& t: threads) t.join();
		CHECK(winners == 1);
	}

	// Report text carries the counts.
	{
		FastFirstCounters counters;
		counters.graphics_fast = 7;
		counters.compute_fast  = 3;
		counters.swaps         = 9;
		counters.saved_ns      = 2'500'000'000ull;
		const auto text = FormatFastFirst(SnapshotFastFirst(counters, 4));
		CHECK(text.find("10 fast builds (7 graphics, 3 compute") != std::string::npos);
		CHECK(text.find("9 optimized swaps") != std::string::npos);
		CHECK(text.find("saved 2500.0 ms") != std::string::npos);
		CHECK(text.find("4 pending") != std::string::npos);
	}

	// Scheduler: the cap counts reserved plus running work, a released slot is reusable, and the
	// peak is tracked.
	{
		FastFirstScheduler scheduler(2, 3);
		CHECK(scheduler.TryReserve());
		CHECK(scheduler.TryReserve());
		CHECK(scheduler.TryReserve());
		CHECK(!scheduler.TryReserve());
		CHECK(scheduler.Pending() == 3);
		scheduler.Release();
		CHECK(scheduler.TryReserve());
		scheduler.Release();
		scheduler.Release();
		scheduler.Release();
		CHECK(scheduler.Pending() == 0);
		CHECK(scheduler.Peak() == 3);
	}

	// Tasks run in parallel on the workers and release their slots.
	{
		FastFirstScheduler scheduler(2, 8);
		std::barrier       ready(3);
		std::promise<void> release;
		auto               gate = release.get_future().share();
		std::atomic<int>   done {0};
		for (int i = 0; i < 2; ++i) {
			CHECK(scheduler.TryReserve());
			CHECK(scheduler.Submit([&] {
				ready.arrive_and_wait();
				gate.wait();
				++done;
			}));
		}
		ready.arrive_and_wait();
		CHECK(scheduler.Pending() == 2);
		release.set_value();
		scheduler.Stop(10'000);
		CHECK(done == 2);
		CHECK(scheduler.Pending() == 0);
		CHECK(!scheduler.TryReserve());
		CHECK(!scheduler.Submit([] {}));
	}

	// Stop with a drain budget runs queued work; with none it skips it (and reports the skip).
	{
		std::atomic<int> ran {0}, skipped {0};
		FastFirstScheduler scheduler(1, 64);
		std::promise<void> entered, unblock;
		auto               unblocked = unblock.get_future().share();
		CHECK(scheduler.TryReserve());
		CHECK(scheduler.Submit([&] {
			entered.set_value();
			unblocked.wait();
			++ran;
		}));
		entered.get_future().wait();
		for (int i = 0; i < 10; ++i) {
			CHECK(scheduler.TryReserve());
			CHECK(scheduler.Submit([&] { ++ran; }, [&] { ++skipped; }));
		}
		std::thread stopper([&] { scheduler.Stop(0); });
		while (!scheduler.Stopped()) std::this_thread::yield();
		unblock.set_value();
		stopper.join();
		CHECK(ran == 1);     // the running task finished, the queued ones were skipped
		CHECK(skipped == 10);
		CHECK(scheduler.Pending() == 0);
	}
	{
		std::atomic<int> ran {0}, skipped {0};
		FastFirstScheduler scheduler(1, 64);
		std::promise<void> entered, unblock;
		auto               unblocked = unblock.get_future().share();
		CHECK(scheduler.TryReserve());
		CHECK(scheduler.Submit([&] {
			entered.set_value();
			unblocked.wait();
			++ran;
		}));
		entered.get_future().wait();
		for (int i = 0; i < 10; ++i) {
			CHECK(scheduler.TryReserve());
			CHECK(scheduler.Submit([&] { ++ran; }, [&] { ++skipped; }));
		}
		std::thread stopper([&] { scheduler.Stop(60'000); });
		while (!scheduler.Stopped()) std::this_thread::yield();
		unblock.set_value();
		stopper.join();
		CHECK(ran == 11);
		CHECK(skipped == 0);
	}

	if (g_failed != 0) {
		std::printf("PipelineFastFirstTests: %d checks failed\n", g_failed);
		return 1;
	}
	std::puts("Pipeline fast-first: retire list, log gate, report, cap, parallelism and drain passed");
	return 0;
}
