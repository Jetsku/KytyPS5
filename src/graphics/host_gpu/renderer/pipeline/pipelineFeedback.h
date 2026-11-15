#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
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

namespace Libs::Graphics::PipelineStats {

// KYTY_PIPELINE_STATS=<hex hash>[,<hex hash>...] or all (diagnostic, default off): pipelines whose
// stages include one of these guest shaders (a 32-bit hash matches either half) are created with
// VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR and the driver's statistics of each executable
// (VK_KHR_pipeline_executable_properties: registers, instructions, local memory, ...) are printed
// once per pipeline as "Pipeline stats: ...". Only the driver's copy of the create info changes.
[[nodiscard]] bool Requested();
void               SetEnabled(bool on);
[[nodiscard]] bool Enabled();
// Every guest shader module, with its stage name and guest hash (FinishPermutation).
void RegisterModule(vk::ShaderModule module, const char* stage, uint64_t hash);
// Whether any stage of `info` (or `module`) is a selected guest shader.
[[nodiscard]] bool Selected(const vk::GraphicsPipelineCreateInfo& info);
[[nodiscard]] bool Selected(vk::ShaderModule module);
// The stages' "<stage> 0x<hash>" names.
[[nodiscard]] std::string Describe(const vk::PipelineShaderStageCreateInfo* stages, uint32_t count);
void Print(vk::Device device, vk::Pipeline pipeline, const std::string& what, vk::PipelineCreateFlags flags);

} // namespace Libs::Graphics::PipelineStats

namespace Libs::Graphics::EmptyFragment {

// KYTY_EMPTY_FS (default 0 = off; idea from chenxiao07/KytyPS5 5a7fa9a1b): a graphics pipeline the
// guest gives no pixel shader gets an empty fragment shader (no inputs, no outputs, early fragment
// tests). NVIDIA drains the pipe between consecutive draws whose pipelines differ in having a
// fragment stage (a depth-only draw beside a shaded one in the same pass); with the empty stage
// every pipeline has one. It changes no output: color outputs of a fragment shader that writes none
// are as undefined as without a fragment shader, and the depth/stencil work stays ahead of its
// invocations as without one.
//   1: on when the device is NVIDIA; 2: on for every vendor (tests).
// Applied to a copy of the create info at the driver call only (Apply), after every content key
// (pipeline journal, known keys, pipeline list snapshots, library keys) was taken from the
// original: no stored key or list entry depends on the switch or the device, and toggling it only
// makes the affected pipelines miss the driver's pipeline cache once.

// Once, on the device-owning thread, before any graphics pipeline is created.
void Configure(vk::Device device, uint32_t vendor_id);
// The empty module, or null when the switch is off for this device.
[[nodiscard]] vk::ShaderModule Module() noexcept;
// Pure decision (tests): whether a pipeline with these properties gets the empty stage.
[[nodiscard]] bool Eligible(bool library, bool linked, uint32_t stage_count, bool has_fragment,
                            bool rasterizer_discard) noexcept;

struct Storage {
	vk::GraphicsPipelineCreateInfo                 info {};
	std::array<vk::PipelineShaderStageCreateInfo, 8> stages {};
};
// `info`, or a copy of it in `storage` with the empty fragment stage appended when Module() is set
// and the pipeline is eligible (monolithic, has stages, no fragment stage, rasterization on).
[[nodiscard]] const vk::GraphicsPipelineCreateInfo& Apply(const vk::GraphicsPipelineCreateInfo& info,
                                                          Storage& storage) noexcept;
// Graphics pipeline libraries: the stage to use for a fragment shader library without one (null
// when off).
[[nodiscard]] const vk::PipelineShaderStageCreateInfo* LibraryStage() noexcept;
// Pipelines created with the empty stage so far.
[[nodiscard]] uint64_t AppliedCount() noexcept;
// Tests: use `module` (an opaque handle, nothing is created) as the empty module; null turns it off.
void SetModuleForTest(vk::ShaderModule module) noexcept;

} // namespace Libs::Graphics::EmptyFragment

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEFEEDBACK_H_
