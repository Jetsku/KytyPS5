// Tests for graphics/host_gpu/deviceLostReport.h (the once-only GPU loss report).
#include "graphics/host_gpu/deviceLostReport.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::DeviceLostReport;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "DeviceLostReportTests: FAILED: %s\n", message);
		std::exit(1);
	}
}

struct Probe {
	std::atomic<int>  calls {0};
	std::atomic<bool> finished {false};
	int               sleep_ms     = 0;
	bool              reenter      = false;
	bool              reenter_ran  = false;
};

void ReporterFn(void* context) {
	auto* probe = static_cast<Probe*>(context);
	++probe->calls;
	if (probe->reenter) {
		// Code reached from a report can check results too; it must not wait for itself.
		probe->reenter_ran = DeviceLostReport::RunOnce();
	}
	if (probe->sleep_ms > 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(probe->sleep_ms));
	}
	probe->finished = true;
}

void TestNoReporterIsANoOp() {
	Check(!DeviceLostReport::RunOnce(), "RunOnce without a reporter does nothing");
	Probe probe;
	DeviceLostReport::Register(&ReporterFn, &probe);
	Check(DeviceLostReport::RunOnce(), "a reporter registered after an empty RunOnce still runs");
	Check(probe.calls == 1, "the late reporter ran once");
	DeviceLostReport::Unregister(&probe);
}

void TestRunsOnceAndRearmsOnRegister() {
	Probe probe;
	DeviceLostReport::Register(&ReporterFn, &probe);
	Check(DeviceLostReport::RunOnce(), "the first call runs the reporter");
	Check(!DeviceLostReport::RunOnce(), "the second call does not");
	Check(!DeviceLostReport::RunOnce(), "nor the third");
	Check(probe.calls == 1, "exactly one report");
	// A new registration (a new device) arms it again.
	DeviceLostReport::Register(&ReporterFn, &probe);
	Check(DeviceLostReport::RunOnce(), "registering again re-arms the report");
	Check(probe.calls == 2, "two reports in total");
	DeviceLostReport::Unregister(&probe);
}

void TestUnregisterOnlyMatchingContext() {
	Probe mine;
	Probe other;
	DeviceLostReport::Register(&ReporterFn, &mine);
	DeviceLostReport::Unregister(&other); // not the registered context: no effect
	Check(DeviceLostReport::RunOnce(), "an unrelated Unregister keeps the reporter");
	Check(mine.calls == 1, "the registered reporter ran");
	DeviceLostReport::Unregister(&mine);
	DeviceLostReport::Register(&ReporterFn, &mine);
	DeviceLostReport::Unregister(&mine);
	Check(!DeviceLostReport::RunOnce(), "after Unregister of its own context nothing runs");
	Check(mine.calls == 1, "no extra report after Unregister");
}

void TestConcurrentCallersWaitForTheReport() {
	Probe probe;
	probe.sleep_ms = 80;
	DeviceLostReport::Register(&ReporterFn, &probe);
	constexpr int            THREADS = 8;
	std::atomic<int>         ran {0};
	std::atomic<int>         returned_before_finish {0};
	std::vector<std::thread> threads;
	for (int i = 0; i < THREADS; i++) {
		threads.emplace_back([&] {
			if (DeviceLostReport::RunOnce()) ran++;
			if (!probe.finished) returned_before_finish++;
		});
	}
	for (auto& thread: threads) thread.join();
	Check(probe.calls == 1, "eight concurrent callers produce a single report");
	Check(ran == 1, "exactly one caller ran it");
	Check(returned_before_finish == 0, "every caller returned only after the report finished");
	DeviceLostReport::Unregister(&probe);
}

void TestReentrantCallDoesNotDeadlock() {
	Probe probe;
	probe.reenter = true;
	DeviceLostReport::Register(&ReporterFn, &probe);
	Check(DeviceLostReport::RunOnce(), "report with a re-entrant call completes");
	Check(!probe.reenter_ran, "the inner RunOnce returned false immediately");
	DeviceLostReport::Unregister(&probe);
}

void TestWaitIsBounded() {
	Probe probe;
	probe.sleep_ms = 3500; // longer than WaitLimit
	DeviceLostReport::Register(&ReporterFn, &probe);
	std::thread reporter([] { DeviceLostReport::RunOnce(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	const auto begin = std::chrono::steady_clock::now();
	const bool ran   = DeviceLostReport::RunOnce();
	const auto waited =
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
	Check(!ran, "a waiting caller did not run the report");
	Check(waited.count() < 3300, "a stuck report cannot hold other threads longer than WaitLimit");
	Check(waited.count() >= 2500, "the waiting caller did wait for the report first");
	reporter.join();
	DeviceLostReport::Unregister(&probe);
}

} // namespace

int main() {
	TestNoReporterIsANoOp();
	TestRunsOnceAndRearmsOnRegister();
	TestUnregisterOnlyMatchingContext();
	TestConcurrentCallersWaitForTheReport();
	TestReentrantCallDoesNotDeadlock();
	TestWaitIsBounded();
	std::puts("DeviceLostReportTests: passed");
	return 0;
}
