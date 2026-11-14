#ifndef EMULATOR_SRC_COMMON_GUESTPRIORITY_H_
#define EMULATOR_SRC_COMMON_GUESTPRIORITY_H_

#include "common/liveSwitch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

// KYTY_GUEST_PRIORITY_FLAT (live, default 0 = the band mapping of kernel/pthread.cpp unchanged): the
// host priority of the guest's priority bands.
//   1: the guest's high band runs at the normal band's host priority (one step down), so a high-band
//      job worker that spins no longer outranks the normal-band threads that produce its work.
//   2: the guest's normal band runs at the high band's host priority (one step up; the game's threads
//      then equal the emulator's normal-priority helpers).
// A guest thread applies a change to itself at its next HLE wait entry point (HleWait::Scope) and at
// its next priority change; new threads start with it.
// Why: on an affinity mask with fewer general processors than the game's high-band job workers,
// those workers held every processor while the normal-band threads they wait for were READY for
// seconds (Sky Garden 4C/8T, smallcpu 2026-11-14).
namespace Common::GuestPriority {

inline std::atomic<uint32_t> g_generation {1};
inline std::atomic<void (*)()> g_apply {nullptr}; // re-applies the calling guest thread's priority

inline Live::Switch g_mode(
    "KYTY_GUEST_PRIORITY_FLAT",
    [](const char* value) -> int64_t {
	    return value == nullptr ? 0 : std::clamp<long>(std::strtol(value, nullptr, 10), 0, 2);
    },
    [](int64_t /*previous*/, int64_t /*value*/) { g_generation.fetch_add(1, std::memory_order_relaxed); });

[[nodiscard]] inline int Mode() noexcept {
	return static_cast<int>(g_mode.Get());
}

// Pure (tested): the host priority step of a guest band value (high 0, normal -1, low -2) in `mode`.
[[nodiscard]] constexpr int BandHost(int band_host, int mode) noexcept {
	if (mode == 1 && band_host == 0) {
		return -1;
	}
	if (mode == 2 && band_host == -1) {
		return 0;
	}
	return band_host;
}

// One relaxed load unless the switch changed since this thread last looked.
inline void Check() noexcept {
	thread_local uint32_t seen = 1;
	const auto            generation = g_generation.load(std::memory_order_relaxed);
	if (generation != seen) [[unlikely]] {
		seen = generation;
		if (auto* apply = g_apply.load(std::memory_order_acquire); apply != nullptr) {
			apply();
		}
	}
}

} // namespace Common::GuestPriority

#endif // EMULATOR_SRC_COMMON_GUESTPRIORITY_H_
