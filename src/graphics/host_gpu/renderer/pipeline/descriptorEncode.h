#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORENCODE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORENCODE_H_

// The descriptor part of a commit's encode (drawEncode.h): the descriptor writes and push data built
// from the stages' prepared bindings, then the push constants, push descriptors and bindless set
// they record through the command buffer's shadow. The build reads the bindings and a few handles
// captured by the commit thread (DescriptorEnv); it touches no cache, tracker or device.

#include "graphics/host_gpu/renderer/drawEncode.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>

namespace Libs::Graphics::DrawEncode {

// What the build reads besides the bindings (RenderExecutor::MakeDescriptorEnv): the emulator's
// own buffers bound by some programs, and the uniform-range limits.
struct DescriptorEnv {
	vk::DescriptorBufferInfo bda_page_table {};
	vk::DescriptorBufferInfo fault_buffer {};
	vk::DescriptorBufferInfo mip_counters {};
	vk::DescriptorBufferInfo mip_canary {};
	// min(UniformBufferBytes, maxUniformBufferRange).
	vk::DeviceSize uniform_range = 0;
	// ShaderRecompiler::Spirv::SrtUniformEnabled().
	bool srt_uniform = false;
};

struct BuiltDescriptors {
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	vk::ShaderStageFlags           push_stages;
	bool                           mesh_stage = false;
	// A stage's shader data or flattened SRT was uploaded for this command (fresh_upload).
	bool fresh = false;
};

// The writes into scratch (cleared first) and the push data. partial: KYTY_DRAW_RUN_PUSH, only the
// per-draw descriptors (no image or sampler writes).
void BuildDescriptors(std::span<PreparedBindings* const> stages, vk::PipelineBindPoint point,
                      bool partial, const DescriptorEnv& env, DescriptorScratch& scratch,
                      BuiltDescriptors& built);

// Counters of a recorded push (FrameEvent.DescriptorPushMiss*), KYTY_PUSH_SHADOW_FRESH_SKIP and
// KYTY_DESCRIPTOR_OFFSET_AUDIT (descriptors.cpp).
void               CountPushMiss(int32_t result, std::span<const vk::WriteDescriptorSet> writes);
[[nodiscard]] bool FreshSkipEnabled();
[[nodiscard]] bool OffsetAuditEnabled();

// The push constants of the build (a mesh stage then records its draw dwords over them, so the
// shadow is forgotten).
template <typename Sink>
void EmitPushConstants(Sink& sink, Shadow& shadow, const ShadowFlags& flags, vk::PipelineLayout layout,
                       const BuiltDescriptors& built) {
	if (built.has_push_data) {
		shadow.PushConstants(sink, flags, layout, built.push_stages, sizeof(built.push_data),
		                     built.push_data.dwords.data());
	}
	if (built.mesh_stage) {
		shadow.InvalidatePushConstants();
	}
}

template <typename Sink>
int32_t EmitPushDescriptors(Sink& sink, Shadow& shadow, const ShadowFlags& flags,
                            vk::PipelineBindPoint point, vk::PipelineLayout layout,
                            const DescriptorScratch& scratch, bool known_miss) {
	return shadow.PushDescriptors(sink, flags, point, layout, 0, static_cast<uint32_t>(scratch.writes.size()),
	                              scratch.writes.data(), known_miss);
}

// Set 1: the bindless table (Senaxx). Binding it leaves set 0 as it is (the same layout).
template <typename Sink>
void EmitBindlessSet(Sink& sink, vk::PipelineBindPoint point, vk::PipelineLayout layout,
                     vk::DescriptorSet set) {
	sink.bindDescriptorSets(point, layout, 1, 1, &set, 0, nullptr);
}

} // namespace Libs::Graphics::DrawEncode

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORENCODE_H_
