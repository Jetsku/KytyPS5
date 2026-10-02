#ifndef KYTY_COMMON_THREADS_H_
#define KYTY_COMMON_THREADS_H_

#include "common/common.h"

#include <memory>
#include <string>

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

using thread_func_t    = void (*)(void*);
using wait_poll_func_t = void (*)();

struct ThreadPrivate;
struct MutexPrivate;
struct CondVarPrivate;

class Thread {
public:
	Thread(thread_func_t func, void* arg);
	~Thread();

	void Join();
	void Detach();

	// Once a thread has finished, the id may be reused by another thread.
	[[nodiscard]] std::string GetId() const;

	// The id is unique and can't be reused by another thread.
	[[nodiscard]] int GetUniqueId() const;

	static void SleepMicro(uint32_t micros);
	static void SleepNano(uint64_t nanos);
	static bool IsMainThread();

	// Get current thread id
	// Once a thread has finished, the id may be reused by another thread.
	static std::string GetThreadId();

	// Get current thread id
	// The id is unique and can't be reused by another thread.
	static int GetThreadIdUnique();

	KYTY_CLASS_NO_COPY(Thread);

private:
	std::unique_ptr<ThreadPrivate> m_thread;
};

class Mutex {
public:
	Mutex();
	~Mutex();

	void Lock();
	void Unlock();
	bool TryLock();

	friend class CondVar;

	KYTY_CLASS_NO_COPY(Mutex);

private:
	std::unique_ptr<MutexPrivate> m_mutex;
};

class CondVar {
public:
	CondVar();
	~CondVar();

	void Wait(Mutex* mutex);
	bool WaitFor(Mutex* mutex, uint32_t micros);
	void Signal();
	void SignalAll();

	static void SetWaitPollCallback(wait_poll_func_t callback);

	KYTY_CLASS_NO_COPY(CondVar);

private:
	std::unique_ptr<CondVarPrivate> m_cond_var;
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
