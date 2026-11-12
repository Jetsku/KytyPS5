// Kernel semaphores (kernel/semaphore.cpp): queue order and counts handed out by a signal, cancel
// and delete, timeouts, and targeted wake-ups (upstream a0268f1de: a signal wakes only the waiters
// it gives counts to).

#include "common/gthreadStats.h"
#include "kernel/pthread.h"
#include "kernel/semaphore.h"
#include "libs/errno.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// The few kernel functions semaphore.cpp calls, without the rest of the kernel.
namespace Libs::LibKernel {
void KernelDispatchPendingSignalForCurrentThread() {}
int  PthreadGetCurrentPriorityForKernel() {
    return 700;
}
int KYTY_SYSV_ABI KernelClockGettime(KernelClockid /*clock_id*/, KernelTimespec* tp) {
	const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	tp->tv_sec  = ns / 1000000000;
	tp->tv_nsec = ns % 1000000000;
	return 0;
}
} // namespace Libs::LibKernel

namespace Libs::Posix {
KYTY_SYSV_ABI int* GetErrorAddr() {
	static thread_local int error = 0;
	return &error;
}
} // namespace Libs::Posix

namespace {

namespace Sema = Libs::LibKernel::Semaphore;
using Libs::LibKernel::KERNEL_ERROR_EACCES;
using Libs::LibKernel::KERNEL_ERROR_ECANCELED;
using Libs::LibKernel::KERNEL_ERROR_ETIMEDOUT;

constexpr int OK = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SemaphoreWakeTests: failed: %s\n", text);
		std::fflush(stderr);
		std::abort();
	}
}

void WaitForWaiters(Sema::KernelSema sem, int count) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (Sema::Testing::Waiting(sem) != count) {
		Check(std::chrono::steady_clock::now() < deadline, "waiters registered in time");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	// Let the last waiter reach its condition variable.
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

struct Waiters {
	std::mutex       mutex;
	std::vector<int> order;   // waiter ids in the order their waits returned OK
	std::vector<int> results; // per waiter id
	std::vector<std::thread> threads;

	explicit Waiters(int n): results(n, 1) {}

	// Starts waiter `id` (need `need`, timeout `micros` or none) and waits until it is queued.
	void Start(Sema::KernelSema sem, int id, int need, int expect_waiting, uint32_t micros = 0) {
		threads.emplace_back([this, sem, id, need, micros] {
			Libs::LibKernel::KernelUseconds timeout = micros;
			const int r = Sema::KernelWaitSema(sem, need, micros != 0 ? &timeout : nullptr);
			std::lock_guard lock(mutex);
			results[id] = r;
			if (r == OK) {
				order.push_back(id);
			}
		});
		WaitForWaiters(sem, expect_waiting);
	}

	size_t Done() {
		std::lock_guard lock(mutex);
		return order.size();
	}

	void Join() {
		for (auto& t: threads) {
			t.join();
		}
		threads.clear();
	}
};

void WaitDone(Waiters& w, size_t n) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
	while (w.Done() < n) {
		Check(std::chrono::steady_clock::now() < deadline, "woken waiters returned in time");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

// FIFO semaphore, three waiters of one count: each signal of one count wakes the oldest waiter.
void TestFifoOrder() {
	Sema::KernelSema sem = nullptr;
	Check(Sema::KernelCreateSema(&sem, "fifo", 1, 0, 8, nullptr) == OK, "create fifo semaphore");
	Waiters w(3);
	for (int i = 0; i < 3; i++) {
		w.Start(sem, i, 1, i + 1);
	}
	for (int i = 0; i < 3; i++) {
		Check(Sema::KernelSignalSema(sem, 1) == OK, "signal one count");
		WaitDone(w, static_cast<size_t>(i) + 1);
		Check(w.Done() == static_cast<size_t>(i) + 1, "one signal of one count wakes one waiter");
		Check(Sema::Testing::Waiting(sem) == 2 - i, "the others keep waiting");
	}
	w.Join();
	Check(w.order == std::vector<int>({0, 1, 2}), "waiters woken in FIFO order");
	Check(Sema::KernelPollSema(sem, 1) != OK, "no count left over");
	Check(Sema::KernelDeleteSema(sem) == OK, "delete fifo semaphore");
}

// Counts are handed out in queue order to every waiter they satisfy: a waiter needing more than
// is left does not block a later one needing less.
void TestCounts() {
	Sema::KernelSema sem = nullptr;
	Check(Sema::KernelCreateSema(&sem, "counts", 1, 0, 8, nullptr) == OK, "create counts semaphore");
	Waiters w(3);
	w.Start(sem, 0, 3, 1);
	w.Start(sem, 1, 1, 2);
	w.Start(sem, 2, 2, 3);
	Check(Sema::KernelSignalSema(sem, 2) == OK, "signal two counts");
	WaitDone(w, 1);
	{
		std::lock_guard lock(w.mutex);
		Check(w.order == std::vector<int>({1}), "two counts: the waiter needing three is skipped, the "
		                                        "one needing one runs, one count is left");
	}
	Check(Sema::Testing::Waiting(sem) == 2, "two waiters left");
	Check(Sema::KernelSignalSema(sem, 1) == OK, "signal one more count");
	WaitDone(w, 2);
	{
		std::lock_guard lock(w.mutex);
		Check(w.order == std::vector<int>({1, 2}), "the left-over count and the new one go to the "
		                                           "waiter needing two");
	}
	Check(Sema::KernelSignalSema(sem, 3) == OK, "signal three counts");
	WaitDone(w, 3);
	w.Join();
	Check(w.order == std::vector<int>({1, 2, 0}), "the waiter needing three runs last");
	Check(Sema::KernelPollSema(sem, 1) != OK, "no count left over");
	Check(Sema::KernelSignalSema(sem, 9) != OK, "signal above the maximum is rejected");
	Check(Sema::KernelDeleteSema(sem) == OK, "delete counts semaphore");
}

// A signal that satisfies one of three waiters wakes only that one: no waiter wakes without a
// count (counted by KYTY_GTHREAD_STATS).
void TestTargetedWakeups() {
	GThreadStats::Testing::SetEnabled(true);
	Sema::KernelSema sem = nullptr;
	Check(Sema::KernelCreateSema(&sem, "wakeups", 1, 0, 8, nullptr) == OK, "create semaphore");
	Waiters w(3);
	for (int i = 0; i < 3; i++) {
		w.Start(sem, i, 1, i + 1);
	}
	const auto before = GThreadStats::Sum();
	Check(Sema::KernelSignalSema(sem, 1) == OK, "signal one count");
	WaitDone(w, 1);
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	const auto after = GThreadStats::Sum();
	Check(after.sema_signal - before.sema_signal == 1, "one signal counted");
	Check(after.sema_signal_multi - before.sema_signal_multi == 1, "the signal found three waiters");
	Check(after.sema_signal_partial - before.sema_signal_partial == 1,
	      "the signal left two waiters not runnable");
	const auto spurious = after.sema_spurious - before.sema_spurious;
	Check(spurious == 0, "no waiter woke without a count");
	Check(Sema::KernelSignalSema(sem, 2) == OK, "signal the rest");
	WaitDone(w, 3);
	w.Join();
	Check(w.order.size() == 3 && w.order[0] == 0, "all three ran, the oldest first");
	Check(Sema::KernelDeleteSema(sem) == OK, "delete semaphore");
	GThreadStats::Testing::SetEnabled(false);
}

// Cancel wakes every waiter with ECANCELED and sets the count; delete wakes every waiter with
// EACCES; a timed wait times out.
void TestCancelDeleteTimeout() {
	Sema::KernelSema sem = nullptr;
	Check(Sema::KernelCreateSema(&sem, "cancel", 1, 0, 8, nullptr) == OK, "create semaphore");
	{
		Waiters w(2);
		w.Start(sem, 0, 1, 1);
		w.Start(sem, 1, 2, 2);
		int waiting = -1;
		Check(Sema::KernelCancelSema(sem, 3, &waiting) == OK, "cancel");
		Check(waiting == 2, "cancel reports two waiters");
		w.Join();
		Check(w.results[0] == KERNEL_ERROR_ECANCELED && w.results[1] == KERNEL_ERROR_ECANCELED,
		      "cancel wakes both waiters with ECANCELED");
		Check(Sema::KernelPollSema(sem, 3) == OK, "cancel set the count to three");
	}
	{
		Waiters w(1);
		w.Start(sem, 0, 1, 1, 200000);
		w.Join();
		Check(w.results[0] == KERNEL_ERROR_ETIMEDOUT, "timed wait times out");
		Check(Sema::Testing::Waiting(sem) == 0, "timed-out waiter left the queue");
	}
	{
		Waiters w(2);
		w.Start(sem, 0, 1, 1);
		w.Start(sem, 1, 1, 2);
		Check(Sema::KernelDeleteSema(sem) == OK, "delete with waiters");
		w.Join();
		Check(w.results[0] == KERNEL_ERROR_EACCES && w.results[1] == KERNEL_ERROR_EACCES,
		      "delete wakes both waiters with EACCES");
	}
}

} // namespace

int main() {
	TestFifoOrder();
	TestCounts();
	TestTargetedWakeups();
	TestCancelDeleteTimeout();
	std::printf("SemaphoreWakeTests: all passed\n");
	return 0;
}
