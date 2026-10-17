#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELAYOUTCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELAYOUTCACHE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// Interned renderer pipeline layouts (KYTY_LAYOUT_INTERN, default on; =0 gives every pipeline its
// own descriptor-set layout and pipeline layout, as before).
//
// Every renderer pipeline has one descriptor set (set 0) and one push-constant range at offset 0.
// Two pipelines whose set-0 bindings (binding number, type, count, stages) and push-constant range
// are equal get the same VkDescriptorSetLayout and VkPipelineLayout handles. Vulkan treats layouts
// created from identical parameters as identically defined, so nothing a shader sees changes; the
// shared handles let the push-descriptor and push-constant shadows (CommandBuffer) and the
// descriptor heap's per-layout set batches match across pipeline switches.
[[nodiscard]] bool PipelineLayoutInterningEnabled();

struct PipelineLayoutHandles {
	vk::DescriptorSetLayout set_layout            = nullptr;
	vk::PipelineLayout      pipeline_layout       = nullptr;
	bool                    uses_push_descriptors = false;
};

// Canonical identity of a renderer pipeline layout: set-0 bindings sorted by binding number, the
// push-descriptor flag and the push-constant range {stages, 0, size}.
struct PipelineLayoutSignature {
	std::vector<std::array<uint32_t, 4>> bindings; // binding, descriptor type, count, stage flags
	uint32_t                             push_stages      = 0;
	uint32_t                             push_size        = 0;
	bool                                 push_descriptors = false;
	// Bindless (Senaxx): the set-1 layout of a pipeline that samples bindless images (0: none).
	uint64_t                             bindless_set     = 0;

	bool operator==(const PipelineLayoutSignature&) const = default;
};

// Pure: the signature (and the push-descriptor decision, as before: the total descriptor count
// fits maxPushDescriptors) for these bindings. Immutable samplers are not used by the renderer.
[[nodiscard]] PipelineLayoutSignature
MakePipelineLayoutSignature(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                            vk::ShaderStageFlags push_stages, uint32_t push_size,
                            uint32_t max_push_descriptors);
[[nodiscard]] uint64_t HashPipelineLayoutSignature(const PipelineLayoutSignature& signature);

// The layouts for these bindings: interned (owned by the cache, destroyed by
// DestroyInternedPipelineLayouts) or, with interning disabled, newly created for the caller.
// bindless_set: the set-1 layout of a pipeline that samples bindless images (not owned), or null
// for every other pipeline, whose layout is then exactly as before.
[[nodiscard]] PipelineLayoutHandles
AcquirePipelineLayout(GraphicContext& graphics,
                      std::span<const vk::DescriptorSetLayoutBinding> bindings,
                      vk::ShaderStageFlags push_stages, uint32_t push_size,
                      vk::DescriptorSetLayout bindless_set = nullptr);
// Destroys a pipeline's layouts unless they are interned.
void ReleasePipelineLayout(GraphicContext& graphics, vk::PipelineLayout pipeline_layout,
                           vk::DescriptorSetLayout set_layout);
// Destroys every interned layout of graphics.device. Call after every pipeline using them is gone.
void DestroyInternedPipelineLayouts(GraphicContext& graphics);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELAYOUTCACHE_H_
