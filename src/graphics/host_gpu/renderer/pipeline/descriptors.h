#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;
struct BindlessHeap;

namespace DrawPrep {
struct StagePlan;
} // namespace DrawPrep

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
	// TextureBindingMemo entry (slot, unique tag) whose description `desc` equals; tag 0: none.
	uint64_t                   memo_tag  = 0;
	uint32_t                   memo_slot = 0;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// KYTY_UNNORMALIZED_FIX: the stage runs a hull shader's normalized-LOD copy
	// (ShaderProgram::normalized_samplers); its samplers are bound with FORCE_UNNORMALIZED cleared.
	bool normalize_unnormalized = false;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<TextureBinding>           images;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;
	// Workgroup count of a direct compute dispatch, for write-range proofs that bound addresses
	// by workgroup ids. Unknown for draws and indirect dispatches.
	std::array<uint32_t, 3>               dispatch_groups {};
	bool                                  has_dispatch_groups = false;
	// GET_LOD_STATS: some image of the stage has a mip-statistics counter (its per-draw field lacks
	// the no-counter flag), set with the fields in RebindBuffers. mip_stats_canary: bind the
	// verify canary instead of the counters (KYTY_LOD_STATS_PLAIN_VARIANT=verify, draw path).
	bool                                  mip_stats_active = false;
	bool                                  mip_stats_canary = false;
	// KYTY_DRAW_SEQUENCE_FAST (textures): the program and T# words `images` were last resolved from
	// by RenderExecutor::PrepareBindings (null program: none).
	const ShaderRecompiler::IR::CompiledShaderInfo*      texture_program = nullptr;
	std::vector<ShaderRecompiler::IR::DescriptorValue> texture_words;
	// The stage's earlier texture sets (program, T# words, resolved bindings), most recent first,
	// for stages alternating between a few sets (desert mesh stamps: about three per frame). Swapped
	// with the current set, never copied; their bindings are revalidated like the current ones.
	struct TextureSet {
		const ShaderRecompiler::IR::CompiledShaderInfo*      program = nullptr;
		std::vector<ShaderRecompiler::IR::DescriptorValue> words;
		std::vector<TextureBinding>                         images;
	};
	std::array<TextureSet, 3> texture_history {};
	// KYTY_XFRAME_REUSE=verify: the current set came from the cross-frame store and passed its
	// revalidation; RebindImages checks the views it acquires (reset by PrepareBindings).
	bool xframe_verify_views = false;
	// RebindBuffers: flattened_srt or shader_data_buffer was allocated for this binding (not a
	// reused upload of the same recording), so no earlier descriptor set refers to it.
	bool                                  fresh_upload = false;
	uint64_t                              write_preparation_tick = UINT64_MAX;
	// KYTY_DRAW_PREP_BINDINGS: the committed draw's plan for this stage (set by PrepareBindings,
	// null for dispatches and draws without a plan), and whether shader_data is the plan's
	// (user dwords and mip-statistics fields filled in, memory offsets zero).
	DrawPrep::StagePlan*                  plan             = nullptr;
	bool                                  plan_shader_data = false;
	// Bindless images (Senaxx; RenderExecutor::PrepareBindlessHeaps): flattened-SRT words to patch
	// (offset, region, entries) and the heaps whose images must be readable for this stage.
	std::vector<std::array<uint32_t, 3>>  bindless_patches;
	std::vector<BindlessHeap*>            bindless_heaps;
};

// NoteBufferOutOfBoundsMode's combination of a V# (OOB_SELECT, stride, swizzle, ADD_TID): the bit
// index it logs once per run.
[[nodiscard]] uint32_t BufferOutOfBoundsCombination(const ShaderBufferResource& descriptor);
// KYTY_BUFFER_RANGE_FIXES (descriptors.cpp).
[[nodiscard]] bool BufferRangeFixesEnabled();
// KYTY_EMIT_SPLIT (descriptors.cpp): 0 off, 1 resolved draws, 2 verify.
[[nodiscard]] int64_t EmitSplitMode();
// The final sampler dwords of a program's sampler binding (depth-compare bits cleared unless the
// shader compares, point filtering forced where the shader needs it): the SamplerCache key.
[[nodiscard]] ShaderSamplerResource
NativeSamplerDescriptor(const ShaderRecompiler::IR::CompiledShaderInfo& program, uint32_t index,
                        const ShaderRecompiler::IR::DescriptorValue& value);
// The user-data dwords of a stage's shader data (PrepareBindings), appended to `shader_data`.
void AppendUserShaderData(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                          const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                          std::vector<uint32_t>&                          shader_data);
// The GET_LOD_STATS field of every instrumented image (RebindBuffers), written into the
// shader data; true when some image has a mip-statistics counter.
bool WriteMipStatsFields(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                         const ShaderRecompiler::IR::ResourceSnapshot&   snapshot,
                         std::vector<uint32_t>&                          shader_data);

// Description of the fallback image for a null shader image descriptor.
[[nodiscard]] TextureCache::ImageDesc
NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                TextureCache::BindingType binding);

// KYTY_DMA_PREFAULT_KB=<n> (default 0, off; usa-ratt/KytyPS5-GT7 d3260d78e): before a shader
// that reads guest memory through its page table (DMA) runs, the first n KiB its user-data
// pointers point at are made resident (BufferCache::MakeResident). GT7's light-probe bake reads
// its per-probe constants that way, a new page per probe; pages it read as zeros wrote whole cubes
// of NaN (black or tinted lighting, different from load to load).
void PrefaultDmaPointers(BufferCache& buffers, const ShaderStageRuntime& stage);

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] uint32_t
NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

// Whether a bindless heap entry with this T# can be held by the set-1 image array `binding`
// (ir/BindlessBindings.h); anything else samples the placeholder.
[[nodiscard]] bool BindlessKeyCompatible(const ShaderTextureResource& descriptor, uint32_t binding);
[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
// KYTY_DEPTH_ENCODING_FIELDS: 1 = the T# differs from a supported depth encoding only in fields
// the texture unit does not use for a non-array 1D/2D resource, 2 = the same with CORNER_SAMPLES
// set, 0 = anything else (descriptors.cpp).
[[nodiscard]] uint32_t DepthEncodingFieldsVerdict(const ShaderTextureResource& descriptor, bool r128);
[[nodiscard]] bool IsSupportedSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                                   const ShaderTextureResource& descriptor,
                                                   vk::Format image_format, vk::Format view_format);
[[nodiscard]] uint64_t SampledDepthBindingSignature(const ShaderRecompiler::IR::ImageResource& resource,
                                                    const ShaderTextureResource& descriptor,
                                                    vk::Format image_format, vk::Format view_format);
// Exits on an unsupported storage view unless `fatal` is false: then it prints the report (once)
// and returns false.
bool ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size,
                            bool fatal = true);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
