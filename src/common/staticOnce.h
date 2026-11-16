#ifndef KYTY_COMMON_STATICONCE_H_
#define KYTY_COMMON_STATICONCE_H_

#include "common/liveSwitch.h"

#include <atomic>
#include <cstdlib>
#include <mutex>
#include <new>
#include <type_traits>
#include <utility>

namespace Common {

// A function-local value computed once, without the thread-safe static's guard on every call.
//
// `static const T x = init();` inside a function is a thread-safe static: under the MSVC ABI
// (clang-cl) every call compares the guard with a thread-local epoch (a TLS index load, the thread's
// TLS block and the epoch), before the value is read. On the command processor of Astro's Playroom
// (GPU Jungle, ~10,000 draws per flip) 0.51 ms per flip of its sampled time sat on those guard lines
// of about a hundred gates (environment switches read on every draw).
//
//   static constinit Common::StaticOnce<bool> on_once;
//   const bool on = on_once.Get([] { return ...; });
//
// The object is constant-initialized (no guard of its own); Get() reads a state byte (acquire, a
// plain load on x86) and the value. The first call runs `init` exactly once under a mutex, as the
// thread-safe static did (side effects such as a log line happen once). T needs no default
// constructor; it is never destroyed (like most such gates' values, it lives until the process ends).
//
// KYTY_STATIC_ONCE_FAST (live, default 1): 0 restores the old per-call cost (a thread-safe static guard
// check is made on every Get, so a same-boot A/B measures what the guards cost); 1 skips it.
namespace StaticOnceDetail {
extern Live::Switch g_fast; // liveSwitch.cpp
} // namespace StaticOnceDetail

[[nodiscard]] inline bool StaticOnceFast() noexcept {
	return StaticOnceDetail::g_fast.Get() != 0;
}

// The old cost, for the A/B: one thread-safe static's guard check (its initializer is not a constant,
// and the value is kept alive, so the guard is not folded away).
inline void StaticOnceLegacyGuard() noexcept {
	static const bool guard = std::getenv("KYTY_STATIC_ONCE_FAST") == nullptr;
	__asm__ __volatile__("" : : "r"(guard));
}

template <typename T>
class StaticOnce {
public:
	constexpr StaticOnce() noexcept {} // NOLINT(modernize-use-equals-default): constinit

	StaticOnce(const StaticOnce&)            = delete;
	StaticOnce& operator=(const StaticOnce&) = delete;

	template <typename Init>
	[[nodiscard]] const T& Get(Init&& init) {
		if (!StaticOnceFast()) {
			StaticOnceLegacyGuard();
		}
		if (m_state.load(std::memory_order_acquire) != 0) [[likely]] {
			return *std::launder(reinterpret_cast<const T*>(m_storage));
		}
		return Slow(std::forward<Init>(init));
	}

private:
	template <typename Init>
	const T& Slow(Init&& init) {
		static std::mutex mutex; // first calls only
		std::lock_guard lock(mutex);
		if (m_state.load(std::memory_order_relaxed) == 0) {
			::new (static_cast<void*>(m_storage)) T(init());
			m_state.store(1, std::memory_order_release);
		}
		return *std::launder(reinterpret_cast<const T*>(m_storage));
	}

	std::atomic<uint8_t> m_state {0};
	alignas(T) unsigned char m_storage[sizeof(T)] {};
};

} // namespace Common

#endif // KYTY_COMMON_STATICONCE_H_
