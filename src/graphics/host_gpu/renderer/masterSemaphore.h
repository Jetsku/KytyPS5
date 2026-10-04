#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_

#include "common/common.h"
#include "graphics/host_gpu/queueSubmission.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <atomic>
#include <memory>

namespace Libs::Graphics {

struct GraphicContext;

class MasterSemaphore {
public:
	// track_dispatch: host submissions of this timeline happen on another thread than the one
	// allocating its ticks even without the submission broker (the CP recorder, commandRecorder.h);
	// a tick then counts as known complete only after its vkQueueSubmit returned.
	explicit MasterSemaphore(GraphicContext& graphics, bool track_dispatch = false);
	~MasterSemaphore();
	KYTY_CLASS_NO_COPY(MasterSemaphore);

	[[nodiscard]] uint64_t CurrentTick() const noexcept {
		return m_current_tick.load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t KnownGpuTick() const noexcept {
		// In queued mode the GPU can finish before the host driver returns. Every
		// resource-retirement caller must also wait for that host call to finish.
		const auto gpu_tick = m_gpu_tick.load(std::memory_order_acquire);
		return m_submission_progress
		           ? std::min(gpu_tick, m_submission_progress->dispatched_tick.load(
		                                    std::memory_order_acquire))
		           : gpu_tick;
	}
	[[nodiscard]] bool     IsFree(uint64_t tick) const noexcept { return KnownGpuTick() >= tick; }
	[[nodiscard]] uint64_t NextTick() noexcept {
		return m_current_tick.fetch_add(1, std::memory_order_release);
	}
	[[nodiscard]] vk::Semaphore Handle() const noexcept { return m_semaphore; }
	[[nodiscard]] const std::shared_ptr<SubmissionProgress>& GetSubmissionProgress() const noexcept {
		return m_submission_progress;
	}

	void Refresh();
	void Wait(uint64_t tick);

private:
	GraphicContext&       m_graphics;
	vk::Semaphore         m_semaphore = nullptr;
	std::atomic<uint64_t> m_gpu_tick {0};
	std::atomic<uint64_t> m_current_tick {1};
	std::shared_ptr<SubmissionProgress> m_submission_progress;
	uint32_t                            m_watchdog_timeline = UINT32_MAX;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_MASTERSEMAPHORE_H_
