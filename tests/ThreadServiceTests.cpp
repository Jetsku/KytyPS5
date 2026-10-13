// Service-thread priority (KYTY_SERVICE_PRIORITY), sched_yield (Common::YieldToReadyThread), the
// sub-microsecond sleep (Common::YieldAndPauseMicro) and the pending-signal fast path
// (TakeLowestPendingSignal, KYTY_GUEST_SCHED). Prints the per-call costs it measures.
#include "common/condWaitUntil.h"
#include "common/threads.h"
#include "kernel/pendingSignals.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using Libs::LibKernel::TakeLowestPendingSignal;

std::atomic<uint64_t> g_empty_mask {0};

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ThreadServiceTests: failed: %s\n", text);
		std::abort();
	}
}

double NsPerCall(std::chrono::steady_clock::time_point start, uint64_t calls) {
	return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
	           .count() /
	       static_cast<double>(calls);
}

// 1. The service level defaults to highest; each level maps to its Windows priority, and the CP's
//    own level (KYTY_CP_PRIORITY) is unchanged.
void TestServicePriority() {
	const auto* env      = std::getenv("KYTY_SERVICE_PRIORITY");
	const int   expected = env != nullptr && env[0] >= '0' && env[0] <= '2' ? env[0] - '0' : 2;
	Check(Common::ServiceThreadPriorityLevel() == expected, "service priority level");
#if defined(_WIN32) && defined(__clang__)
	int service = 0;
	int cp      = 0;
	std::thread([&] {
		Common::RaiseServiceThreadPriority();
		service = GetThreadPriority(GetCurrentThread());
	}).join();
	std::thread([&] {
		Common::RaiseCurrentThreadPriority();
		cp = GetThreadPriority(GetCurrentThread());
	}).join();
	const int want = expected == 2   ? THREAD_PRIORITY_HIGHEST
	                 : expected == 1 ? THREAD_PRIORITY_ABOVE_NORMAL
	                                 : THREAD_PRIORITY_NORMAL;
	Check(service == want, "service thread priority");
	const auto* cp_env   = std::getenv("KYTY_CP_PRIORITY");
	const int   cp_level = cp_env != nullptr && cp_env[0] >= '0' && cp_env[0] <= '2' ? cp_env[0] - '0' : 1;
	const int   cp_want  = cp_level == 2   ? THREAD_PRIORITY_HIGHEST
	                       : cp_level == 1 ? THREAD_PRIORITY_ABOVE_NORMAL
	                                       : THREAD_PRIORITY_NORMAL;
	Check(cp == cp_want, "CP priority unchanged");
	std::printf("  service priority: level %d -> %d, CP -> %d\n", expected, service, cp);
#endif
}

// 2. sched_yield returns at once with nothing else to run, and lets a ready thread that shares
//    the CPU run.
void TestYield() {
	constexpr uint64_t CALLS = 200000;
	const auto         start = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < CALLS; i++) {
		(void)Common::YieldToReadyThread();
	}
	std::printf("  sched_yield alone: %.0f ns per call\n", NsPerCall(start, CALLS));

#if defined(_WIN32)
	// Both threads on one CPU: the yielding thread must let the worker finish its work.
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	Check(GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) != 0,
	      "affinity mask");
	DWORD_PTR cpu = 1;
	for (int bit = 63; bit >= 0; bit--) {
		if ((process_mask & (DWORD_PTR {1} << bit)) != 0) {
			cpu = DWORD_PTR {1} << bit;
			break;
		}
	}
	// The worker's loop is longer than a quantum, so the yielder always gets the CPU back at
	// least once while the worker is still ready, and its yield must then hand the CPU over.
	std::atomic<bool>     worker_done {false};
	std::atomic<uint64_t> switches {0};
	std::thread           yielder([&] {
        SetThreadAffinityMask(GetCurrentThread(), cpu);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!worker_done.load() && std::chrono::steady_clock::now() < deadline) {
            if (Common::YieldToReadyThread()) {
                switches.fetch_add(1);
            }
        }
	});
	const auto  start_work = std::chrono::steady_clock::now();
	std::thread worker([&] {
		SetThreadAffinityMask(GetCurrentThread(), cpu);
		Sleep(2);
		volatile uint64_t sink = 0;
		for (uint64_t i = 0; i < 200'000'000; i++) {
			sink = sink + i;
		}
		worker_done = true;
	});
	worker.join();
	yielder.join();
	Check(worker_done.load(), "the worker finished");
	Check(switches.load() > 0, "sched_yield hands the CPU to a ready thread on its CPU");
	std::printf("  sched_yield sharing a CPU: worker done in %.0f ms, %" PRIu64 " switches\n",
	            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
	                                                      start_work)
	                .count(),
	            switches.load());
#endif
}

// 3. A 1 us sleep lasts at least 1 us and returns soon after.
void TestYieldAndPauseMicro() {
	constexpr int CALLS  = 20000;
	double        worst  = 0;
	double        total  = 0;
	for (int i = 0; i < CALLS; i++) {
		const auto start = std::chrono::steady_clock::now();
		Common::YieldAndPauseMicro(1);
		const auto us =
		    std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
		        .count();
		Check(us >= 1.0, "1 us sleep never returns early");
		worst = std::max(worst, us);
		total += us;
	}
	std::printf("  1 us sleep: mean %.2f us, worst %.1f us\n", total / CALLS, worst);
}

// 4. Pending signals: lowest first, bits at or above the limit stay, an empty mask is one load.
void TestPendingSignals() {
	std::atomic<uint64_t> mask {0};
	Check(TakeLowestPendingSignal(mask, 130) == -1, "empty mask");
	mask = (uint64_t {1} << 5) | (uint64_t {1} << 3) | (uint64_t {1} << 40);
	Check(TakeLowestPendingSignal(mask, 130) == 3, "lowest first");
	Check(TakeLowestPendingSignal(mask, 130) == 5, "then the next");
	Check(TakeLowestPendingSignal(mask, 130) == 40, "then a high bit");
	Check(TakeLowestPendingSignal(mask, 130) == -1 && mask.load() == 0, "drained");
	mask = (uint64_t {1} << 3) | (uint64_t {1} << 5);
	Check(TakeLowestPendingSignal(mask, 4) == 3, "below the limit");
	Check(TakeLowestPendingSignal(mask, 4) == -1, "limit respected");
	Check(mask.load() == (uint64_t {1} << 5), "bits above the limit stay pending");
	Check(TakeLowestPendingSignal(mask, 0) == -1, "zero limit");
	mask = uint64_t {1} << 63;
	Check(TakeLowestPendingSignal(mask, 64) == 63, "bit 63");

	// Concurrent posts: every posted signal is taken at most once, none is lost at the end.
	mask = 0;
	std::atomic<uint64_t> posted {0};
	std::atomic<bool>     stop {false};
	uint64_t              taken = 0;
	std::thread           poster([&] {
        for (uint64_t i = 0; i < 2'000'000; i++) {
            const uint64_t bit = uint64_t {1} << (i % 64);
            if ((mask.fetch_or(bit) & bit) == 0) {
                posted.fetch_add(1);
            }
        }
        stop = true;
	});
	while (!stop.load() || mask.load() != 0) {
		if (TakeLowestPendingSignal(mask, 64) >= 0) {
			taken++;
		}
	}
	poster.join();
	Check(taken == posted.load(), "each newly posted signal is taken exactly once");

	// Cost with nothing pending: the fast path against the legacy 64-bit scan. A global mask, like
	// a thread's, so the compiler cannot keep it in a register.
	constexpr uint64_t CALLS = 5'000'000;
	auto&              empty = g_empty_mask;
	auto               start = std::chrono::steady_clock::now();
	int                   found = 0;
	for (uint64_t i = 0; i < CALLS; i++) {
		// Keeps the compiler from merging the loads of consecutive calls.
		std::atomic_signal_fence(std::memory_order_seq_cst);
		found += TakeLowestPendingSignal(empty, 130) >= 0 ? 1 : 0;
	}
	const double fast = NsPerCall(start, CALLS);
	start             = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < CALLS / 10; i++) {
		std::atomic_signal_fence(std::memory_order_seq_cst);
		for (int signum = 0; signum < 64; signum++) {
			const uint64_t bit = uint64_t {1} << signum;
			found += (empty.fetch_and(~bit) & bit) != 0 ? 1 : 0;
		}
	}
	const double legacy = NsPerCall(start, CALLS / 10);
	Check(found == 0, "nothing pending");
	std::printf("  pending-signal poll, nothing pending: %.1f ns (legacy scan %.1f ns)\n", fast,
	            legacy);
}

// Common::CondWaitUntil (KYTY_PRECISE_COND_WAITS, Senaxx d069e1e99): a 5 ms timed wait that nobody
// signals ends at the deadline, not up to a system tick late; a signal in the precise stretch is
// still seen; dispatch() runs between polls. The old form is measured for comparison only.
void TestCondWaitUntil() {
	std::mutex              mutex;
	std::condition_variable cv;
	const auto lateness_us = [&](bool precise) {
		std::vector<double> late;
		for (int i = 0; i < 25; i++) {
			std::unique_lock lock(mutex);
			const auto       deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(5000);
			uint32_t         dispatched = 0;
			Common::CondWaitUntil(lock, cv, [] { return false; }, deadline, 10000, precise,
			                      [&] { dispatched++; });
			late.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
			                                                         deadline)
			                   .count());
			Check(late.back() >= 0.0, "a timed wait never ends before its deadline");
		}
		std::sort(late.begin(), late.end());
		return late[late.size() / 2];
	};
	// The timing bounds are retried: a busy machine can delay one attempt, a real regression fails
	// all of them. The semantic checks (never early, flag seen) are strict on every attempt.
	constexpr int ATTEMPTS   = 3;
	const double  coarse     = lateness_us(false);
	double        precise    = 0.0;
	bool          precise_ok = false;
	for (int attempt = 0; attempt < ATTEMPTS && !precise_ok; attempt++) {
		precise    = lateness_us(true);
		precise_ok = precise < 400.0;
		std::printf("  5 ms timed wait, median lateness: precise %.0f us (attempt %d), condition variable "
		            "only %.0f us\n",
		            precise, attempt + 1, coarse);
	}
	Check(precise_ok, "a precise timed wait ends within 0.4 ms of its deadline");

	// A wake-up 4 ms into a 6 ms wait (inside the precise stretch) ends the wait at once.
	const auto wake_wait_us = [&] {
		bool        flag = false;
		std::thread waker([&] {
			std::this_thread::sleep_for(std::chrono::microseconds(4000));
			{
				std::scoped_lock lock(mutex);
				flag = true;
			}
			cv.notify_one();
		});
		const auto start = std::chrono::steady_clock::now();
		{
			std::unique_lock lock(mutex);
			Common::CondWaitUntil(lock, cv, [&] { return flag; }, start + std::chrono::microseconds(6000),
			                      10000, true, [] {});
			Check(flag, "the signalled flag is seen");
		}
		waker.join();
		return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
	};
	double waited    = 0.0;
	bool   waited_ok = false;
	for (int attempt = 0; attempt < ATTEMPTS && !waited_ok; attempt++) {
		waited    = wake_wait_us();
		waited_ok = waited < 5900.0;
		std::printf("  signal 4 ms into a 6 ms precise wait: ended after %.0f us (attempt %d)\n", waited,
		            attempt + 1);
	}
	Check(waited_ok, "a signal before the deadline ends a precise wait before the deadline");
}

} // namespace

int main() {
	TestServicePriority();
	TestYield();
	TestYieldAndPauseMicro();
	TestPendingSignals();
	TestCondWaitUntil();
	std::printf("ThreadServiceTests: all cases passed\n");
	return 0;
}
