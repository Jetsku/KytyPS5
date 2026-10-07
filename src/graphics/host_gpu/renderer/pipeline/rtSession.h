#pragma once

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::RtSession {

// Session state of the software ray-tracing kernels (KYTY_RT_SOFTWARE, graphics/shader/recompiler).
//
// A GPU or driver that fails to build the pipeline of an RT kernel (an error result, an exception,
// a fault inside the driver) must not take the emulator down: software RT is marked failed for the
// rest of the session and every later dispatch of an RT kernel is skipped, as with
// KYTY_RT_SOFTWARE=0 (the tiled lighting and GI passes then leave their outputs untouched: black
// lighting). The next launch tries again; KYTY_RT_SOFTWARE=0 skips them from the start.

[[nodiscard]] inline std::atomic<bool>& FailedFlag() {
	static std::atomic<bool> failed {false};
	return failed;
}

// True once a RT pipeline build failed.
[[nodiscard]] inline bool Failed() {
	return FailedFlag().load(std::memory_order_acquire);
}

// Marks software RT failed; true for the caller that did it (that one logs the reason once).
inline bool MarkFailed() {
	return !FailedFlag().exchange(true, std::memory_order_acq_rel);
}

// Tests.
inline void ResetForTest() {
	FailedFlag().store(false, std::memory_order_release);
}

enum class DispatchAction : uint8_t {
	Run,           // the pipeline is ready
	WaitBounded,   // first dispatch that finds it still building: wait up to the budget
	Skip,          // not ready (or software RT failed): this dispatch is dropped
};

// What the command processor does with a dispatch of an RT kernel. The CP never waits for a
// pipeline build for long: a cold compile of a 300k-word kernel took 17 s on an RTX 50 GPU, and a
// stalled CP freezes the whole game (and trips the hang watchdog), while a skipped dispatch only
// leaves the lighting it feeds one frame behind. The first dispatch of a kernel waits up to
// `wait_budget_ms` (a cached pipeline builds in well under that, so a warm launch loses no frame
// of lighting); after that, and for every dispatch of a kernel whose build is still running, the
// dispatch is skipped. A budget of UINT32_MAX waits for the build (KYTY_RT_PIPELINE_ASYNC=0).
[[nodiscard]] inline DispatchAction DecideDispatch(bool session_failed, bool ready, bool first_request,
                                                   uint32_t wait_budget_ms) {
	if (session_failed) return DispatchAction::Skip;
	if (ready) return DispatchAction::Run;
	if (first_request && wait_budget_ms != 0) return DispatchAction::WaitBounded;
	return DispatchAction::Skip;
}

} // namespace Libs::Graphics::RtSession
