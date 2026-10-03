#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_

#include <algorithm>
#include <cstdint>

// KYTY_VRAM_GC_BUDGET=1 (default 0: off; live switch): the caches collect against the device memory
// budget as it is now, frame-aged, instead of thresholds fixed at startup and submission ages.
//
// The stock collectors compute their marks once, from the budget at startup, as fractions of
// min(budget, 8 GiB): on a 10 GiB budget (a 12 GB card) the image collector's trigger is 1 GiB and
// its critical mark 8.4 GiB, the buffer collector's critical mark 8.4 GiB. The image collector frees
// images unused for 16-160 completed submissions (Astro Bot: ~180 per frame) and the buffer collector
// buffers unused for 80-160 (it cannot see uses through BDA pointers) with a GPU wait for the dirty
// ones, so on cards below ~16 GB both run all the time, re-create and re-upload what every frame
// uses, and the startup budget misses the budget other programs leave later.
//
// With the switch on, once per frame the planning budget B (GraphicContext::GetTotalMemoryBudget: the
// driver's VK_EXT_memory_budget budget minus a reserve) is read again (the collectors run on the
// command-processor thread after every completed submission, so not per collection) and:
//   - images: above ImageTrigger(B) the collector frees, once per frame and oldest first, clean
//     images not used for AgeFrames() frames (a quarter of that, at least 4, above Critical(B)),
//     until usage is back at the trigger; never a GPU-modified, bound or target image, never a
//     download (TextureCache::RetireUnusedImages); the native image pool keeps nothing above it;
//   - buffers: the stock buffer collector runs only above B (it is the last resort: its LRU cannot
//     see BDA reads), with its downloads only above Critical(B) + B / 16.
// KYTY_VRAM_PRESSURE_FRAMES (if set) is the age; otherwise 30 frames.
namespace Libs::Graphics::VramBudget {

[[nodiscard]] bool GcEnabled() noexcept;

constexpr uint64_t DefaultAgeFrames = 30;

[[nodiscard]] constexpr uint64_t ImageTrigger(uint64_t budget) noexcept {
	return budget / 4 * 3;
}

[[nodiscard]] constexpr uint64_t Critical(uint64_t budget) noexcept {
	return budget - budget / 16;
}

[[nodiscard]] constexpr uint64_t BufferTrigger(uint64_t budget) noexcept {
	return budget;
}

[[nodiscard]] constexpr uint64_t BufferCritical(uint64_t budget) noexcept {
	return budget + budget / 16;
}

struct ImagePlan {
	bool     retire     = false;
	uint64_t age_frames = 0; // free clean images unused for more than this many frames
	uint64_t bytes      = 0; // until this many bytes are freed
};

// What the image collector does this frame at `usage` of planning budget `budget`.
[[nodiscard]] constexpr ImagePlan PlanImages(uint64_t budget, uint64_t usage, uint64_t age_frames) noexcept {
	const auto trigger = ImageTrigger(budget);
	if (budget == 0 || usage <= trigger) {
		return {};
	}
	const auto age = age_frames != 0 ? age_frames : DefaultAgeFrames;
	return {true, usage >= Critical(budget) ? std::max<uint64_t>(age / 4, 4) : age, usage - trigger};
}

} // namespace Libs::Graphics::VramBudget

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_VRAMBUDGET_H_ */
