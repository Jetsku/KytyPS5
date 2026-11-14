#include "common/threads.h"

#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono> // IWYU pragma: keep
#include <cstdlib>
#include <ctime>
#include <memory>
#include <thread>

// macOS has no clock_nanosleep.
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS && !defined(__APPLE__)
#define KYTY_POSIX_HIGH_RES_SLEEP
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Common {

bool ShortSleepsBlock() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SHORT_SLEEP_BLOCK");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

} // namespace Common

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
// IWYU pragma: no_include <winbase.h>
constexpr uint64_t KYTY_SLEEP_SPIN_LIMIT_100NS = 500; // 50 us

// Spins of a contested Common::Mutex before it blocks (KYTY_CS_SPIN_COUNT, default 4000; 0 blocks
// at once and leaves the core to other threads). Read once, at the first mutex.
static DWORD CsSpinCount() {
	static const DWORD count = [] {
		const char* value = std::getenv("KYTY_CS_SPIN_COUNT");
		return value != nullptr && value[0] != '\0' ? static_cast<DWORD>(std::strtoul(value, nullptr, 10))
		                                            : DWORD {4000};
	}();
	return count;
}

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

// High-resolution waitable timers need Windows 10 1803, the floor MapViewOfFile3 already sets.
static void SleepHighResolution100ns(uint64_t units_100ns) {
	if (units_100ns == 0) {
		return;
	}

	// Keep spinning only where a kernel transition is
	// likely to cost more than the requested delay; ordinary millisecond sleeps use the
	// per-thread high-resolution waitable timer below (always with KYTY_SHORT_SLEEP_BLOCK=1).
	if (units_100ns <= KYTY_SLEEP_SPIN_LIMIT_100NS && !Common::ShortSleepsBlock()) {
		LARGE_INTEGER frequency {};
		LARGE_INTEGER start {};
		if (QueryPerformanceFrequency(&frequency) != 0 && QueryPerformanceCounter(&start) != 0 &&
		    frequency.QuadPart > 0) {
			const auto wait_ticks =
			    static_cast<LONGLONG>((static_cast<long double>(units_100ns) *
			                           static_cast<long double>(frequency.QuadPart)) /
			                          10000000.0L);
			const auto    deadline = start.QuadPart + std::max<LONGLONG>(wait_ticks, 1);
			LARGE_INTEGER now {};
			do {
				if (QueryPerformanceCounter(&now) == 0) {
					break;
				}
				YieldProcessor();
			} while (now.QuadPart < deadline);
			return;
		}
	}

	// Closed when the thread exits (upstream 2f6101c16; the handles used to leak with the thread).
	thread_local const std::unique_ptr<void, decltype(&CloseHandle)> timer {
	    [] {
		    HANDLE handle = CreateWaitableTimerExW(
		        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
		    return handle != nullptr ? handle : CreateWaitableTimerW(nullptr, TRUE, nullptr);
	    }(),
	    &CloseHandle};

	if (timer != nullptr) {
		LARGE_INTEGER due_time {};
		due_time.QuadPart = -static_cast<LONGLONG>(units_100ns);
		if (SetWaitableTimerEx(timer.get(), &due_time, 0, nullptr, nullptr, nullptr, 0) != 0) {
			WaitForSingleObject(timer.get(), INFINITE);
			return;
		}
	}

	std::this_thread::sleep_for(std::chrono::nanoseconds(units_100ns * 100));
}

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <synchapi.h>
// IWYU pragma: no_include <minwinbase.h>
// IWYU pragma: no_include <errhandlingapi.h>
// IWYU pragma: no_include <winerror.h>

static CRITICAL_SECTION* ToCs(unsigned char* storage) {
	return reinterpret_cast<CRITICAL_SECTION*>(storage);
}

static CONDITION_VARIABLE* ToCv(void** storage) {
	return reinterpret_cast<CONDITION_VARIABLE*>(storage);
}

#endif

#ifdef KYTY_POSIX_HIGH_RES_SLEEP
// Spin for very short waits; use an absolute deadline for longer waits (it preserves the requested
// sleep across signal interruptions).
static void SleepHighResolutionNanos(uint64_t nanos) {
	if (nanos == 0) {
		return;
	}

	constexpr uint64_t NANOS_PER_SEC = 1000000000;
	constexpr uint64_t SPIN_LIMIT_NS = 50000; // below this a context switch dominates

	timespec deadline {};
	if (clock_gettime(CLOCK_MONOTONIC, &deadline) != 0) {
		std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
		return;
	}

	auto target_nsec = static_cast<uint64_t>(deadline.tv_nsec) + nanos;
	deadline.tv_sec += static_cast<time_t>(target_nsec / NANOS_PER_SEC);
	deadline.tv_nsec = static_cast<long>(target_nsec % NANOS_PER_SEC);

	if (nanos <= SPIN_LIMIT_NS && !Common::ShortSleepsBlock()) {
		timespec now {};
		do {
			if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
				return;
			}
		} while (now.tv_sec < deadline.tv_sec ||
		         (now.tv_sec == deadline.tv_sec && now.tv_nsec < deadline.tv_nsec));
		return;
	}

	while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr) == EINTR) {
	}
}
#endif

namespace Common {

static wait_poll_func_t g_cond_wait_poll_callback = nullptr;

static std::thread::id g_main_thread;

void InitializeThreads() {
	g_main_thread = std::this_thread::get_id();
	// KYTY_CPU_RESERVE: the process default CPU sets, before most threads start.
	InitCpuPlacement();
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static void RaiseCurrentThreadPriorityLevel(int level) {
	if (level == 0) {
		return;
	}
	auto* thread = GetCurrentThread();
	(void)SetThreadPriority(thread, level == 2 ? THREAD_PRIORITY_HIGHEST
	                                           : THREAD_PRIORITY_ABOVE_NORMAL);
	// Keep the thread off efficiency scheduling (EcoQoS) even when the window is in the
	// background.
	THREAD_POWER_THROTTLING_STATE throttling {};
	throttling.Version     = THREAD_POWER_THROTTLING_CURRENT_VERSION;
	throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
	throttling.StateMask   = 0;
	(void)SetThreadInformation(thread, ThreadPowerThrottling, &throttling, sizeof(throttling));
}
#endif

static int PriorityLevelFromEnv(const char* name, int fallback) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] >= '0' && value[0] <= '2' ? value[0] - '0' : fallback;
}

void RaiseCurrentThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const int level = PriorityLevelFromEnv("KYTY_CP_PRIORITY", 1);
	RaiseCurrentThreadPriorityLevel(level);
#endif
}

int ServiceThreadPriorityLevel() {
	static const int level = PriorityLevelFromEnv("KYTY_SERVICE_PRIORITY", 2);
	return level;
}

void RaiseServiceThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	RaiseCurrentThreadPriorityLevel(ServiceThreadPriorityLevel());
#endif
}

void NameCurrentThreadForOs(const char* name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static const bool on = [] {
		const char* value = std::getenv("KYTY_THREAD_NAMES");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	if (!on || name == nullptr || name[0] == '\0') {
		return;
	}
	using SetDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
	static const auto set = reinterpret_cast<SetDescription>(reinterpret_cast<void*>(
	    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
	if (set == nullptr) {
		return;
	}
	wchar_t wide[96] = {};
	if (MultiByteToWideChar(CP_UTF8, 0, name, -1, wide, 95) > 0) {
		(void)set(GetCurrentThread(), wide);
	}
#else
	(void)name;
#endif
}

void SetCurrentThreadPriorityStep(int step) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const int priority = step >= 2    ? THREAD_PRIORITY_HIGHEST
	                     : step == 1  ? THREAD_PRIORITY_ABOVE_NORMAL
	                     : step == 0  ? THREAD_PRIORITY_NORMAL
	                     : step == -1 ? THREAD_PRIORITY_BELOW_NORMAL
	                                  : THREAD_PRIORITY_LOWEST;
	(void)SetThreadPriority(GetCurrentThread(), priority);
#else
	(void)step;
#endif
}

bool YieldToReadyThread() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return SwitchToThread() != 0;
#else
	std::this_thread::yield();
	return true;
#endif
}

void YieldAndPauseMicro(uint32_t micros) {
	const auto start = std::chrono::steady_clock::now();
	const auto end   = start + std::chrono::microseconds(micros);
	(void)YieldToReadyThread();
	while (std::chrono::steady_clock::now() < end) {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}
}

void Thread::SleepMicro(uint32_t micros) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns(static_cast<uint64_t>(micros) * 10);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(static_cast<uint64_t>(micros) * 1000);
#else
	std::this_thread::sleep_for(std::chrono::microseconds(micros));
#endif
}

void Thread::SleepNano(uint64_t nanos) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SleepHighResolution100ns((nanos + 99) / 100);
#elif defined(KYTY_POSIX_HIGH_RES_SLEEP)
	SleepHighResolutionNanos(nanos);
#else
	std::this_thread::sleep_for(std::chrono::nanoseconds(nanos));
#endif
}

bool Thread::IsMainThread() {
	return g_main_thread == std::this_thread::get_id();
}

int Thread::GetThreadIdUnique() {
	static std::atomic<int> counter = 0;
	static thread_local int tid     = ++counter;
	return tid;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
Mutex::Mutex() {
	static_assert(sizeof(CRITICAL_SECTION) == sizeof(m_cs) &&
	              alignof(CRITICAL_SECTION) <= alignof(Mutex));
	InitializeCriticalSectionAndSpinCount(ToCs(m_cs), CsSpinCount());
}

Mutex::~Mutex() {
	DeleteCriticalSection(ToCs(m_cs));
}
#endif

void Mutex::Lock() {
	// The uncontended path publishes no wait record. The slow path retains the resource even
	// when EnterCriticalSection/the standard mutex never returns.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (!HangWatchdog::Enabled()) {
		EnterCriticalSection(ToCs(m_cs));
		return;
	}
	if (TryEnterCriticalSection(ToCs(m_cs))) return;
	HangWatchdog::Scope wait("host-mutex", reinterpret_cast<uint64_t>(this));
	EnterCriticalSection(ToCs(m_cs));
#else
	if (!HangWatchdog::Enabled()) {
		m_mutex.lock();
		return;
	}
	if (m_mutex.try_lock()) return;
	HangWatchdog::Scope wait("host-mutex", reinterpret_cast<uint64_t>(this));
	m_mutex.lock();
#endif
}

void Mutex::Unlock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	LeaveCriticalSection(ToCs(m_cs));
#else
	m_mutex.unlock();
#endif
}

bool Mutex::TryLock() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return TryEnterCriticalSection(ToCs(m_cs)) != 0;
#else
	return m_mutex.try_lock();
#endif
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
void CondVar::Signal() {
	WakeConditionVariable(ToCv(&m_cv));
}

void CondVar::SignalAll() {
	WakeAllConditionVariable(ToCv(&m_cv));
}
#endif

void CondVar::Wait(Mutex* mutex) {
	HangWatchdog::Scope wait("host-condvar", reinterpret_cast<uint64_t>(this),
	                         reinterpret_cast<uint64_t>(mutex));
	auto* callback = g_cond_wait_poll_callback;
	if (callback == nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), INFINITE);
#else
		m_cv.wait(mutex->m_mutex);
#endif
		return;
	}
	// With a poll callback: wake every 10 ms to run it, outside the lock (a spurious wakeup for
	// the caller, which re-checks its condition).
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const bool woken = SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), 10) != 0 ||
	                   GetLastError() != ERROR_TIMEOUT;
#else
	const bool woken = m_cv.wait_for(mutex->m_mutex, std::chrono::microseconds(10000)) ==
	                   std::cv_status::no_timeout;
#endif
	if (!woken) {
		mutex->Unlock();
		callback();
		mutex->Lock();
	}
}

void CondVar::SetWaitPollCallback(wait_poll_func_t callback) {
	g_cond_wait_poll_callback = callback;
}

bool CondVar::WaitFor(Mutex* mutex, uint32_t micros) {
	HangWatchdog::Scope wait("host-condvar-timed", reinterpret_cast<uint64_t>(this), micros, 0, 0,
	                         reinterpret_cast<uint64_t>(mutex));
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// KYTY_US_WAITS=1 (default off; GT7 fork 29ebd192d): the kernel's timeouts are microseconds
	// (SceKernelUseconds; HR timers nanoseconds), but SleepConditionVariableCS counts whole
	// milliseconds on a 1 ms tick: a wait ended up to ~1.3 ms late, and never in under 1 ms. With
	// the switch the condition variable sleeps only until ~1 ms before the deadline; the rest is
	// slept outside it with the high-resolution timer in slices of at most 0.5 ms, each returned
	// as a spurious wakeup (every caller re-checks its condition and its own deadline).
	static const bool micro_deadlines = [] {
		const char* value = std::getenv("KYTY_US_WAITS");
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}();
	if (!micro_deadlines) {
		return SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs),
		                                micros < 1000 ? 1 : micros / 1000) != 0 ||
		       GetLastError() != ERROR_TIMEOUT;
	}
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(micros);
	if (micros >= 2000) {
		if (SleepConditionVariableCS(ToCv(&m_cv), ToCs(mutex->m_cs), (micros - 1000) / 1000) != 0 ||
		    GetLastError() != ERROR_TIMEOUT) {
			return true;
		}
	}
	const auto remaining = std::chrono::duration_cast<std::chrono::microseconds>(
	    deadline - std::chrono::steady_clock::now());
	if (remaining.count() <= 0) {
		return false;
	}
	LeaveCriticalSection(ToCs(mutex->m_cs));
	SleepHighResolution100ns(static_cast<uint64_t>(std::min<int64_t>(remaining.count(), 500)) * 10u);
	EnterCriticalSection(ToCs(mutex->m_cs));
	return std::chrono::steady_clock::now() < deadline;
#else
	return m_cv.wait_for(mutex->m_mutex, std::chrono::microseconds(micros)) ==
	       std::cv_status::no_timeout;
#endif
}

} // namespace Common
