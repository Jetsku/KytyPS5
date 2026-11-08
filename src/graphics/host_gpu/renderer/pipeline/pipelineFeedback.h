#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics::PipelineFeedback {

// KYTY_PIPELINE_FEEDBACK=1 (diagnostic, default off): every graphics pipeline the pipeline cache
// creates through CreateGraphics asks the driver (VkPipelineCreationFeedback, Vulkan 1.3) whether
// it came from the application pipeline cache (a driver-cache hit) or was compiled. Counts and
// times per path go to Report(), which the "Pipeline skip" line and the exit report print. The
// feedback struct is chained onto a copy of the create info at the driver call only, after the
// content key was taken, so journal and known keys are unchanged.
enum class Path : uint32_t {
	Sync,    // created on the command processor (draw waits)
	Builder, // skip / prefetch builder threads
	Probe,   // driver-cache-only lookups (FAIL_ON_PIPELINE_COMPILE_REQUIRED)
	Replay,  // pipeline journal replay (start-up prewarm)
	Count
};

[[nodiscard]] bool Enabled();

// vkCreateGraphicsPipelines for one pipeline, counted when Enabled(); otherwise a plain call.
vk::Result CreateGraphics(vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
                          vk::Pipeline* pipeline, Path path);

// A pipeline taken from the adopted (prewarmed) set instead of the driver: no driver call at all.
void NoteAdopted(Path path);

// "Pipeline feedback: sync 12 (hit 10, compiled 2, ...) | builder ..." or empty when disabled or
// nothing was counted.
[[nodiscard]] std::string Report();

} // namespace Libs::Graphics::PipelineFeedback

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_
