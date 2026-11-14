#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <condition_variable>
#include <mutex>
#endif

namespace Common {

void InitializeThreads();

// Raises the calling thread's scheduling priority for latency-critical emulator threads (the
// command processor). KYTY_CP_PRIORITY: 0 = leave unchanged, 1 = above normal (default),
// 2 = highest. Also opts the thread out of Windows power throttling (EcoQoS). No-op elsewhere.
void RaiseCurrentThreadPriority();

// The same for the short-running service threads the command processor waits on: the video-out
// present thread, the Vulkan queue-submission worker and the command schedulers' priority
// (completion) threads. They block between bursts and never spin. At the guest threads' priority
// a readied one waited for a guest thread's quantum to end, about 30 ms, whenever all CPUs were
// busy (the periodic 73-87 ms frames). KYTY_SERVICE_PRIORITY: 0 = leave unchanged, 1 = above
// normal, 2 = highest (default). No-op elsewhere.
void RaiseServiceThreadPriority();
[[nodiscard]] int ServiceThreadPriorityLevel();

// KYTY_THREAD_NAMES=1 (measurement, default off): the calling thread gets `name` as its OS thread
// description (SetThreadDescription), so that a sampler outside the process can tell the threads
// apart. Nothing with the switch off, and nothing off Windows.
void NameCurrentThreadForOs(const char* name);
// The calling thread's host priority: 2 highest, 1 above normal, 0 normal, -1 below normal, -2 lowest
// (KYTY_HELPER_PRIORITY, common/helperPriority.h). No-op off Windows.
void SetCurrentThreadPriorityStep(int step);
// sched_yield: gives the processor to another thread that is ready to run on this CPU and
// returns at once when there is none (FreeBSD sched_relinquish). True when another thread ran.
bool YieldToReadyThread();

// A sleep shorter than any host timer can wait (a few microseconds): yields once like a blocking
// sleep would, then pauses until `micros` have passed since the call. Never returns early.
void YieldAndPauseMicro(uint32_t micros);

// KYTY_SHORT_SLEEP_BLOCK=1 (default off; upstream KytyPS5 6f24b031f): short sleeps block on the
// high-resolution timer instead of spinning (Thread::SleepMicro/SleepNano up to 50 us) and guest
// sleeps of at most 1 us no longer yield-and-pause (kernel/pthread.cpp). Frees the CPUs that
// sleeping guest threads spin on, but a short sleep then lasts as long as the timer's resolution.
[[nodiscard]] bool ShortSleepsBlock();

using wait_poll_func_t = void (*)();

// Threads themselves are std::thread (upstream 2f6101c16 removed the Common::Thread class).
namespace Thread {

void SleepMicro(uint32_t micros);
void SleepNano(uint64_t nanos);
bool IsMainThread();
// Unique per thread; never reused by another thread.
int GetThreadIdUnique();

} // namespace Thread

// Recursive. Lock publishes a HangWatchdog wait record when it has to block.
class Mutex {
public:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Mutex();
	~Mutex();
#else
	Mutex() = default;
#endif

	void Lock();
	void Unlock();
	bool TryLock();

	KYTY_CLASS_NO_COPY(Mutex);

private:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A CRITICAL_SECTION.
	alignas(8) unsigned char m_cs[40];
#else
	std::recursive_mutex m_mutex;
#endif

	friend class CondVar;
};

class CondVar {
public:
	CondVar() = default;

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	void Signal();
	void SignalAll();
#else
	void Signal() { m_cv.notify_one(); }
	void SignalAll() { m_cv.notify_all(); }
#endif

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A CONDITION_VARIABLE, whose initial state is zero.
	void* m_cv = nullptr;
#else
	std::condition_variable_any m_cv;
#endif
};

class LockGuard {
public:
	using mutex_type = Mutex;

	// NOLINTNEXTLINE(google-runtime-references)
	explicit LockGuard(mutex_type& m): m_mutex(m) { m_mutex.Lock(); }

	~LockGuard() { m_mutex.Unlock(); }

	KYTY_CLASS_NO_COPY(LockGuard);

private:
	mutex_type& m_mutex;
};

} // namespace Common

#endif /* KYTY_COMMON_THREADS_H_ */
