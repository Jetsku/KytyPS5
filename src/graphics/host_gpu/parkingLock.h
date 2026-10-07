#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PARKINGLOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PARKINGLOCK_H_

#include "common/hangWatchdog.h"
#include "common/profiler.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

// KYTY_TRACKER_LOCK_PARK=0: the resource-tracking locks (memory tracker regions, texture cache,
// page manager regions) spin until they are free, as before.
[[nodiscard]] inline bool TrackerLockParkEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TRACKER_LOCK_PARK");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_TRACKER_LOCK_SPIN_US (default 20): how long a waiter spins before it parks.
[[nodiscard]] inline uint64_t TrackerLockSpinNs() {
	static const uint64_t ns = [] {
		const auto* value = std::getenv("KYTY_TRACKER_LOCK_SPIN_US");
		const auto  us    = value != nullptr ? std::strtoull(value, nullptr, 10) : 20ull;
		return std::min<uint64_t>(us, 1000000ull) * 1000ull;
	}();
	return ns;
}

inline void TrackerLockRelax() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#elif defined(__aarch64__)
	__asm__ volatile("yield");
#else
	std::this_thread::yield();
#endif
}

// A lock for critical sections that are usually short but can be long: the GPU thread holds
// tracking locks across upload copies and image refreshes, and any holder can be preempted.
// Waiters spin reading only (the holder keeps its cache line) with pause for
// KYTY_TRACKER_LOCK_SPIN_US, then park on the lock word (WaitOnAddress / futex). Unlock wakes one
// parked waiter, and only when one may be parked.
// States: 0 free, 1 held, 2 held and a waiter may be parked (Drepper's futex mutex).
class ParkingSpinLock final {
public:
	[[nodiscard]] bool try_lock() noexcept {
		uint32_t expected = 0;
		return m_state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
		                                       std::memory_order_relaxed);
	}

	void lock() noexcept {
		if (!try_lock()) [[unlikely]] {
			(void)LockContended();
		}
	}

	void unlock() noexcept {
		if (m_state.exchange(0, std::memory_order_release) == 2) [[unlikely]] {
			m_state.notify_one();
		}
	}

	// The contended acquisition (the caller's try_lock failed). With parking off it spins until
	// the lock is free: reading only with pause when `shared_spin`, else retrying the acquisition
	// back to back (the two historical spin loops). Returns true when it parked.
	bool LockContended(bool shared_spin = true, bool allow_parking = true) noexcept {
		HangWatchdog::Scope wait("resource-parking-lock", reinterpret_cast<uint64_t>(this), 0,
		                         HangWatchdog::Enabled() ? m_state.load(std::memory_order_relaxed) : 0);
		if (!allow_parking || !TrackerLockParkEnabled()) {
			for (;;) {
				if (shared_spin) {
					while (m_state.load(std::memory_order_relaxed) != 0) {
						TrackerLockRelax();
					}
				} else {
					std::atomic_signal_fence(std::memory_order_seq_cst);
				}
				if (try_lock()) {
					return false;
				}
			}
		}
		const auto budget = TrackerLockSpinNs();
		const auto start  = std::chrono::steady_clock::now();
		for (uint32_t spins = 1;; spins++) {
			if (m_state.load(std::memory_order_relaxed) == 0 && try_lock()) {
				return false;
			}
			TrackerLockRelax();
			if ((spins & 63u) == 0u &&
			    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                              std::chrono::steady_clock::now() - start)
			                              .count()) >= budget) {
				break;
			}
		}
		bool parked = false;
		while (m_state.exchange(2, std::memory_order_acquire) != 0) {
			parked = true;
			m_state.wait(2, std::memory_order_relaxed);
		}
		if (parked) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerLockParks);
		}
		return parked;
	}

private:
	std::atomic<uint32_t> m_state {0};
};

static_assert(std::atomic<uint32_t>::is_always_lock_free);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PARKINGLOCK_H_
