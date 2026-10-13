#ifndef KYTY_COMMON_ONCEVALUE_H_
#define KYTY_COMMON_ONCEVALUE_H_

#include <atomic>
#include <cstdint>

namespace Common {

// A value computed once (an environment switch read) and kept for later calls, for gates on hot
// paths. A function-local `static const T x = init();` is a thread-safe static: under the MSVC ABI
// (clang-cl) every call compares its guard with a thread-local epoch, and the gate stays a call.
// The Sky Garden's command processor spent about 3-5% of its time in such gates
// (BufferLookupStats::Enabled alone about 1%). Here the slot is constant-initialized (no guard)
// and read with one relaxed load. `init` must be pure: two threads making the first call at once
// may both run it, and both store the same value.
inline constexpr int64_t OnceUnset = INT64_MIN;

template <typename Init>
[[nodiscard]] inline int64_t Once(std::atomic<int64_t>& slot, Init&& init) {
	auto value = slot.load(std::memory_order_relaxed);
	if (value == OnceUnset) [[unlikely]] {
		value = static_cast<int64_t>(init());
		slot.store(value, std::memory_order_relaxed);
	}
	return value;
}

} // namespace Common

#endif // KYTY_COMMON_ONCEVALUE_H_
