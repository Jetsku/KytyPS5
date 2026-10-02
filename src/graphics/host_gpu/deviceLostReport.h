#ifndef KYTY_GRAPHICS_HOST_GPU_DEVICE_LOST_REPORT_H_
#define KYTY_GRAPHICS_HOST_GPU_DEVICE_LOST_REPORT_H_

// Reports why the GPU was lost, once, from whichever thread sees VK_ERROR_DEVICE_LOST first.
//
// A lost device is reported by the first Vulkan call that happens to check its result: a pipeline
// creation, a queue submit, a semaphore query, the swapchain. Only the pipeline paths used to print
// the VK_EXT_device_fault report, so a loss caught anywhere else (for example "submit upload DMA
// copies failed: ErrorDeviceLost") left no GPU information in the log.
//
// The window registers one reporter after the device exists. Every place that sees
// eErrorDeviceLost calls RunOnce() before it exits the emulator.
//
// - The reporter runs at most once per registration. Threads that call RunOnce() while it runs
//   wait for it, at most WaitLimit, so the process does not exit under a report in progress and a
//   stuck driver call cannot hold the exit hostage.
// - A call made from inside the reporter (it can reach code that checks results too) returns at
//   once instead of waiting for itself.
// - No Vulkan types here: the header is usable from any layer and from a CPU-only test.

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace Libs::Graphics {

struct GraphicContext;

// Prints the VK_EXT_device_fault report for `graphics` (shaders.cpp), or a note when the extension
// is not enabled.
void ReportDeviceFault(const GraphicContext& graphics);

class DeviceLostReport {
public:
	using Reporter = void (*)(void* context);

	static constexpr std::chrono::seconds WaitLimit {3};

	// Replaces the reporter and re-arms it.
	static void Register(Reporter reporter, void* context) {
		auto&            state = State();
		std::scoped_lock lock(state.mutex);
		state.reporter = reporter;
		state.context  = context;
		if (state.phase != Phase::Running) {
			state.phase = Phase::Idle;
		}
	}

	// Drops the reporter if it was registered with `context` (the context is about to go away).
	static void Unregister(void* context) {
		auto&            state = State();
		std::scoped_lock lock(state.mutex);
		if (state.context == context) {
			state.reporter = nullptr;
			state.context  = nullptr;
		}
	}

	// Runs the reporter if it has not run yet. Returns true only for the call that ran it.
	static bool RunOnce() {
		if (Inside()) {
			return false;
		}
		auto&            state = State();
		std::unique_lock lock(state.mutex);
		if (state.phase == Phase::Running) {
			state.finished.wait_for(lock, WaitLimit, [&] { return state.phase != Phase::Running; });
			return false;
		}
		if (state.phase == Phase::Done || state.reporter == nullptr) {
			return false;
		}
		state.phase         = Phase::Running;
		const auto reporter = state.reporter;
		auto*      context  = state.context;
		lock.unlock();

		Inside() = true;
		reporter(context);
		Inside() = false;

		lock.lock();
		state.phase = Phase::Done;
		state.finished.notify_all();
		return true;
	}

private:
	enum class Phase { Idle, Running, Done };

	struct StateData {
		std::mutex              mutex;
		std::condition_variable finished;
		Reporter                reporter = nullptr;
		void*                   context  = nullptr;
		Phase                   phase    = Phase::Idle;
	};

	static StateData& State() {
		static StateData state;
		return state;
	}

	static bool& Inside() {
		thread_local bool inside = false;
		return inside;
	}
};

} // namespace Libs::Graphics

#endif
