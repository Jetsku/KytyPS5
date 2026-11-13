#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWENCODE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWENCODE_H_

// The encode of a draw or dispatch (binds, descriptor pushes, push constants and the draw command
// written into the command stream), separated from the commit's cache and barrier work so that it
// can run on any thread into any sink with the CommandSink/CommandStream::Encoder method names.
//
//  - Shadow: what the command buffer's state commands left in effect (bound pipelines, the
//    push-descriptor contents and epochs, the push constants). CommandBuffer owns one per command
//    buffer and forwards its BindPipeline/PushDescriptors/PushConstants/BindDescriptorSet to it;
//    the comparisons and updates are exactly those CommandBuffer made before.
//  - DescriptorScratch: the descriptor write arrays a commit builds (the executor's scratch before).
//
// Nothing here touches a cache, the barrier batcher, a scheduler or a Vulkan device: the sink is
// the only output. Unit tested without a device (tests/DrawEncodeTests.cpp).

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Libs::Graphics::DrawEncode {

// The descriptor writes a commit builds; pBufferInfo/pImageInfo point into buffers/images.
struct DescriptorScratch {
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<vk::DescriptorImageInfo>  images;
	std::vector<vk::WriteDescriptorSet>   writes;
	std::vector<uint32_t>                 occurrences; // per image binding of the stage being built
};

// The process-wide switches the shadow's comparisons depend on (read by the owner, which passes
// them in, so the shadow itself reads no configuration).
struct ShadowFlags {
	bool renderer_batch       = true; // Common::RendererBatchEnabled()
	bool push_constant_shadow = true; // PushConstantShadowEnabled()
	bool set_reuse            = true; // DescriptorSetReuseEnabled()
};

class Shadow {
public:
	// PushDescriptors result: the update was skipped (identical to the one still in effect), there
	// was nothing comparable (no earlier push in this command buffer for the bind point, another
	// layout, a bound set, or the dedup is off), the binding list differs, or the caller knew it
	// differs; otherwise the index of the first write whose descriptors differ.
	static constexpr int32_t PushAvoided   = -1;
	static constexpr int32_t PushMissState = -2;
	static constexpr int32_t PushMissShape = -3;
	static constexpr int32_t PushMissFresh = -4;

	struct DescriptorState {
		vk::PipelineLayout layout = nullptr;
		// Non-null: set 0 is this bound descriptor set (the push contents below are unused).
		vk::DescriptorSet                     bound_set = nullptr;
		std::vector<vk::WriteDescriptorSet>   writes;
		std::vector<vk::DescriptorBufferInfo> buffers;
		std::vector<vk::DescriptorImageInfo>  images;
	};
	struct PushConstantState {
		bool                     valid  = false;
		vk::PipelineLayout       layout = nullptr;
		vk::ShaderStageFlags     stages;
		uint32_t                 size = 0;
		std::array<uint32_t, 64> dwords {};
	};

	[[nodiscard]] static size_t Index(vk::PipelineBindPoint point) {
		EXIT_IF(point != vk::PipelineBindPoint::eGraphics && point != vk::PipelineBindPoint::eCompute);
		return point == vk::PipelineBindPoint::eCompute ? 1u : 0u;
	}

	// Whether binding `pipeline` records a command (not the one bound, or no dedup); the shadow then
	// holds it as bound. BindPipeline records it; the encode offload decides on the commit thread and
	// records on the encode thread (encodeOffload.h).
	[[nodiscard]] bool TakePipelineBind(const ShadowFlags& flags, vk::PipelineBindPoint point,
	                                    vk::Pipeline pipeline) {
		auto& current = bound_pipelines[Index(point)];
		if (flags.renderer_batch && current == pipeline) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineBindsAvoided);
			return false;
		}
		current = pipeline;
		return true;
	}

	// CommandBuffer::Begin: nothing is bound, every epoch advances, push constants are undefined.
	void Begin() {
		bound_pipelines = {};
		for (auto& state: descriptor_states) {
			state.layout    = nullptr;
			state.bound_set = nullptr;
		}
		for (auto& epoch: descriptor_epochs) {
			epoch++;
		}
		push_constants.valid = false;
	}

	[[nodiscard]] vk::Pipeline BoundPipeline(vk::PipelineBindPoint point) const {
		return bound_pipelines[point == vk::PipelineBindPoint::eCompute ? 1u : 0u];
	}
	[[nodiscard]] uint64_t DescriptorEpoch(vk::PipelineBindPoint point) const {
		return descriptor_epochs[point == vk::PipelineBindPoint::eCompute ? 1u : 0u];
	}
	void InvalidatePushConstants() { push_constants.valid = false; }

	void InvalidateDescriptors(vk::PipelineBindPoint point) {
		auto& state     = descriptor_states[Index(point)];
		state.layout    = nullptr;
		state.bound_set = nullptr;
		descriptor_epochs[Index(point)]++;
	}

	template <typename Sink>
	void BindPipeline(Sink& sink, const ShadowFlags& flags, vk::PipelineBindPoint point,
	                  vk::Pipeline pipeline) {
		if (TakePipelineBind(flags, point, pipeline)) {
			sink.bindPipeline(point, pipeline);
		}
	}

	template <typename Sink>
	void BindDescriptorSet(Sink& sink, const ShadowFlags& flags, vk::PipelineBindPoint point,
	                       vk::PipelineLayout layout, vk::DescriptorSet set) {
		auto& state = descriptor_states[Index(point)];
		EXIT_IF(set == nullptr || layout == nullptr);
		if (flags.set_reuse && state.bound_set == set && state.layout == layout) {
			// Still bound as set 0 with this layout: nothing since disturbed it (every other bind and
			// every push descriptor update of this bind point passes through the shadow), and
			// binding a pipeline never disturbs descriptor sets.
			Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorSetBindsAvoided);
			return;
		}
		sink.bindDescriptorSets(point, layout, 0, 1, &set, 0, nullptr);
		descriptor_epochs[Index(point)]++;
		state.layout    = flags.set_reuse ? layout : nullptr;
		state.bound_set = flags.set_reuse ? set : nullptr;
		state.writes.clear();
		state.buffers.clear();
		state.images.clear();
	}

	// Records pushConstants(layout, stages, 0, size, data) unless the last update recorded in this
	// command buffer was exactly this one (KYTY_PUSH_CONSTANT_SHADOW).
	template <typename Sink>
	void PushConstants(Sink& sink, const ShadowFlags& flags, vk::PipelineLayout layout,
	                   vk::ShaderStageFlags stages, uint32_t size, const void* data) {
		auto& shadow = push_constants;
		// Push-constant values persist in the command buffer across pipeline binds; values set with
		// this very layout are valid for every pipeline created with it.
		if (flags.push_constant_shadow && shadow.valid && shadow.layout == layout &&
		    shadow.stages == stages && shadow.size == size &&
		    std::memcmp(shadow.dwords.data(), data, size) == 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::PushConstantUpdatesAvoided);
			return;
		}
		sink.pushConstants(layout, stages, 0, size, data);
		Profiler::CountFrameEvent(Profiler::FrameEvent::PushConstantUpdates);
		shadow.valid = flags.push_constant_shadow && size <= sizeof(shadow.dwords);
		if (shadow.valid) {
			shadow.layout = layout;
			shadow.stages = stages;
			shadow.size   = size;
			std::memcpy(shadow.dwords.data(), data, size);
		}
	}

	// `known_miss`: the caller knows the update differs from every earlier one of this command
	// buffer (a descriptor refers to an upload made for this command, KYTY_PUSH_SHADOW_FRESH_SKIP):
	// recorded without the comparison and without keeping a copy for the next one.
	template <typename Sink>
	int32_t PushDescriptors(Sink& sink, const ShadowFlags& flags, vk::PipelineBindPoint point,
	                        vk::PipelineLayout layout, uint32_t set, uint32_t count,
	                        const vk::WriteDescriptorSet* writes, bool known_miss) {
		auto& state = descriptor_states[Index(point)];
		if (known_miss) {
			// The comparison could only miss. No copy is kept for the next update either: one equal
			// to this (its upload deduplicated to the same range) is then recorded again, which is
			// only redundant; the Sky Garden start avoids no push at all (DescriptorPushesAvoided).
			sink.pushDescriptorSetKHR(point, layout, set, count, writes);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushes);
			state.layout    = nullptr;
			state.bound_set = nullptr;
			return PushMissFresh;
		}
		bool   supported    = flags.renderer_batch && set == 0;
		size_t buffer_count = 0, image_count = 0;
		for (uint32_t i = 0; supported && i < count; ++i) {
			const auto& write       = writes[i];
			const bool  buffer_type = IsBufferType(write.descriptorType);
			const bool  image_type  = write.descriptorType == vk::DescriptorType::eSampler ||
			                         write.descriptorType == vk::DescriptorType::eSampledImage ||
			                         write.descriptorType == vk::DescriptorType::eStorageImage ||
			                         write.descriptorType == vk::DescriptorType::eCombinedImageSampler;
			supported = write.pNext == nullptr && write.pTexelBufferView == nullptr &&
			            ((buffer_type && write.pBufferInfo != nullptr) ||
			             (image_type && write.pImageInfo != nullptr));
			buffer_count += buffer_type ? write.descriptorCount : 0u;
			image_count += image_type ? write.descriptorCount : 0u;
		}
		int32_t result = PushMissState;
		if (supported && state.layout == layout && state.bound_set == nullptr) {
			result = state.writes.size() == count && state.buffers.size() == buffer_count &&
			                 state.images.size() == image_count
			             ? PushAvoided
			             : PushMissShape;
			size_t buffer_index = 0, image_index = 0;
			for (uint32_t i = 0; result == PushAvoided && i < count; ++i) {
				const auto& write = writes[i];
				const auto& old   = state.writes[i];
				if (write.dstBinding != old.dstBinding || write.dstArrayElement != old.dstArrayElement ||
				    write.descriptorCount != old.descriptorCount ||
				    write.descriptorType != old.descriptorType) {
					result = PushMissShape;
					break;
				}
				const bool buffer_type = IsBufferType(write.descriptorType);
				for (uint32_t j = 0; j < write.descriptorCount; ++j) {
					const bool equal = buffer_type ? write.pBufferInfo[j] == state.buffers[buffer_index++]
					                               : write.pImageInfo[j] == state.images[image_index++];
					if (!equal) {
						result = static_cast<int32_t>(i);
						break;
					}
				}
			}
		}
		if (result == PushAvoided) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushesAvoided);
			return result;
		}
		sink.pushDescriptorSetKHR(point, layout, set, count, writes);
		descriptor_epochs[Index(point)]++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DescriptorPushes);
		state.layout    = nullptr;
		state.bound_set = nullptr;
		if (!supported) {
			return result;
		}
		state.writes.assign(writes, writes + count);
		state.buffers.clear();
		state.images.clear();
		state.buffers.reserve(buffer_count);
		state.images.reserve(image_count);
		for (auto& write: state.writes) {
			if (IsBufferType(write.descriptorType)) {
				state.buffers.insert(state.buffers.end(), write.pBufferInfo,
				                     write.pBufferInfo + write.descriptorCount);
			} else {
				state.images.insert(state.images.end(), write.pImageInfo,
				                    write.pImageInfo + write.descriptorCount);
			}
			write.pBufferInfo = nullptr;
			write.pImageInfo  = nullptr;
		}
		state.layout = layout;
		return result;
	}

	std::array<vk::Pipeline, 2>    bound_pipelines {};
	std::array<DescriptorState, 2> descriptor_states;
	std::array<uint64_t, 2>        descriptor_epochs {};
	PushConstantState              push_constants;

private:
	[[nodiscard]] static bool IsBufferType(vk::DescriptorType type) {
		return type == vk::DescriptorType::eStorageBuffer || type == vk::DescriptorType::eUniformBuffer;
	}
};

// ------------------------------------------------------------------------------------------------
// The draw's other state commands and the draw command (the commit thread's ExecutePreparedDraw and
// the encode offload record them through these).

// The prepared vertex buffers (guest descriptor bounds as sizes, no strides).
template <typename Sink>
void EmitVertexBuffers(Sink& sink, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets,
                       const vk::DeviceSize* sizes) {
	for (uint32_t i = 0; i < count; i++) {
		EXIT_IF(buffers[i] == nullptr);
	}
	if (count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		sink.bindVertexBuffers2(0, count, buffers, offsets, sizes, nullptr);
	}
}

template <typename Sink>
void EmitIndexBuffer(Sink& sink, vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
	if (buffer == nullptr) {
		return;
	}
	sink.bindIndexBuffer(buffer, offset, type);
}

// A direct draw: one command, or for a legacy quad list one per quad.
struct DirectDraw {
	bool     indexed        = false;
	bool     quad_list      = false;
	uint32_t count          = 0; // indices or vertices
	uint32_t instance_count = 0;
	int32_t  vertex_offset  = 0; // indexed
	uint32_t first_vertex   = 0; // not indexed
	uint32_t first_instance = 0;
};

template <typename Sink>
void EmitDirectDraw(Sink& sink, const DirectDraw& draw) {
	if (!draw.quad_list) {
		if (draw.indexed) {
			sink.drawIndexed(draw.count, draw.instance_count, 0, draw.vertex_offset, draw.first_instance);
		} else {
			sink.draw(draw.count, draw.instance_count, draw.first_vertex, draw.first_instance);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED((draw.count & 0x3u) != 0);
	for (uint32_t i = 0; i < draw.count; i += 4) {
		if (draw.indexed) {
			sink.drawIndexed(4, draw.instance_count, i, draw.vertex_offset, draw.first_instance);
		} else {
			sink.draw(4, draw.instance_count, i + draw.first_vertex, draw.first_instance);
		}
	}
}

} // namespace Libs::Graphics::DrawEncode

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWENCODE_H_
