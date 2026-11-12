#ifndef EMULATOR_SRC_COMMON_HANDOFFLEAN_H_
#define EMULATOR_SRC_COMMON_HANDOFFLEAN_H_

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// KYTY_HANDOFF_LEAN (live, default 0; Windows only): the command processor's per-draw hand-offs to
// threads that spin and sometimes park (the sequencer's executed-op word, the preparation window's
// head, the recorder ring's published position) are plain release stores followed by a relaxed
// load of the peer's "parked" word, instead of seq_cst stores and loads (a locked instruction each
// on x86, which drains the store buffer every draw).
//
// What keeps a wake from being lost is an asymmetric fence (Dekker with one heavy side): the
// parking side, after announcing itself (seq_cst) and before its final re-check of the published
// word, calls FlushProcessWriteBuffers, which makes every processor running a thread of this
// process drain its store buffer at some instant P during the call. For the publishing thread's
// store S and its later load L of the parked word: if S came before P it is globally visible before
// the re-check (which sees it and does not sleep); if S came after P, then L (after S in program
// order, and the compiler barrier keeps it there) runs after P, when the announcement is already
// visible (it was a locked store made before the call), so L sees "parked" and wakes the peer.
//
// The parking side pays the flush only while a publishing thread is lean: each publishing thread
// counts itself in g_active (a locked RMW) when it turns lean, before its first lean store, and
// out when it stops, after its last one (the RMW drains them). A park that reads the count as 0
// read it before the increment, so its announcement (a locked store before that read) is visible
// to every lean load of the parked word after the increment; a lean store before a decrement is
// visible once the decrement is, and later publications use the seq_cst handshake. Switching off
// therefore also removes the flush cost (a same-boot A/B measures both sides).
namespace Common::HandoffLean {

#if defined(_WIN32)
extern "C" __declspec(dllimport) void __stdcall FlushProcessWriteBuffers();
inline constexpr bool kAvailable = true;
#else
inline constexpr bool kAvailable = false;
#endif

inline std::atomic<uint32_t> g_active {0};
inline std::atomic<uint64_t> g_flushes {0};

// The mode, written by the parser of the live switch KYTY_HANDOFF_LEAN registered in cpSequencer.cpp
// (emulator only: this header is also used by tests that do not link the live-switch registry). The
// parser runs at startup and for every live change; a publishing thread reads the mode per call.
inline std::atomic<int64_t> g_mode {0};

inline int64_t Parse(const char* value) {
	const int64_t mode = kAvailable && value != nullptr ? std::strtoll(value, nullptr, 10) : 0;
	g_mode.store(mode, std::memory_order_relaxed);
	return mode;
}

// The publishing side: whether this publication may be lean.
[[nodiscard]] inline bool On() noexcept {
	static thread_local bool lean = false;
	const bool on = g_mode.load(std::memory_order_relaxed) != 0;
	if (on != lean) [[unlikely]] {
		if (on) {
			g_active.fetch_add(1, std::memory_order_seq_cst);
		} else {
			g_active.fetch_sub(1, std::memory_order_seq_cst);
		}
		lean = on;
	}
	return on;
}

// The publishing side, between the lean store and the load of the parked word.
inline void CompilerBarrier() noexcept {
	std::atomic_signal_fence(std::memory_order_seq_cst);
}

// The parking side, after announcing itself and before re-checking the published word.
inline void ParkFence() noexcept {
#if defined(_WIN32)
	if (g_active.load(std::memory_order_seq_cst) != 0) {
		FlushProcessWriteBuffers();
		if (const auto n = g_flushes.fetch_add(1, std::memory_order_relaxed) + 1; (n & 0xfffu) == 0) {
			std::printf("HandoffLean: %llu park fences\n", static_cast<unsigned long long>(n));
			std::fflush(stdout);
		}
	}
#endif
}

} // namespace Common::HandoffLean

#endif // EMULATOR_SRC_COMMON_HANDOFFLEAN_H_
