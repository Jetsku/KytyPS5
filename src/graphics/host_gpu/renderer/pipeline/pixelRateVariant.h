#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXELRATEVARIANT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXELRATEVARIANT_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <vector>

// KYTY_EQAA_MIXED on NV mixed samples (from usa-ratt/KytyPS5-GT7 827bc377d, test c88a47a2f).
// A pixel shader that reads its sample index (the SampleId built-in) runs once per sample, and NV
// mixed samples (fewer colour samples than raster samples: how PS5 EQAA is drawn on NVIDIA)
// forbids per-sample shading (VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-
// 01415). GT7's G-buffer draws at 1920x2160 with 2 depth samples per colour sample lost the
// device there. Such a pixel shader gets a per-pixel copy whose sample index is a Private 0;
// NV mixed-sample pipelines use it, and a draw whose shader has none is skipped instead.
namespace Libs::Graphics {

// The SPIR-V of a pixel shader that reads the SampleId built-in, with that input gone and its
// reads giving 0; empty when the shader has no SampleId input or uses it in a way this does not
// rewrite (another per-sample input such as SamplePosition or a Sample-decorated input, the
// variable named by anything but a load, a decoration, a name or the entry point's interface).
[[nodiscard]] std::vector<uint32_t> SpirvWithoutSampleId(std::span<const uint32_t> spirv);
// Whether the module declares a SampleId built-in (or any other per-sample input).
[[nodiscard]] bool SpirvShadesPerSample(std::span<const uint32_t> spirv);
// KYTY_EQAA_PER_SAMPLE=1 (for testing): NV mixed-sample draws keep pixel shaders that read their
// sample index as they are, per-sample shading included, which NVIDIA does not allow there.
[[nodiscard]] bool EqaaPerSampleShading();

namespace PixelRate {

// The pipeline cache's per-pixel copies, by the id of the pixel program (ShaderProgram::id) they
// were made for. Registered when the program is finished on a device with NV mixed samples,
// forgotten (and the copy's module destroyed) with the program.
struct Lookup {
	bool             shades_per_sample = false; // the program reads SampleId (or another per-sample input)
	vk::ShaderModule module            = nullptr; // its per-pixel copy, null if none
};
void   Register(uint64_t program_id, bool shades_per_sample, vk::ShaderModule module);
void   Forget(vk::Device device, uint64_t program_id);
[[nodiscard]] Lookup Find(uint64_t program_id);

} // namespace PixelRate

// KYTY_UNNORMALIZED_FIX (default off; adapted from usa-ratt/KytyPS5-GT7 45ac971cc). The guest GPU
// samples a FORCE_UNNORMALIZED sampler at the LOD the shader names, with coordinates in that mip's
// texels. Vulkan allows unnormalized coordinates only at LOD 0 (samplerCache forces min/maxLod 0),
// so GT7's hull shaders, which cull car-body patches against a depth pyramid read this way at a
// level picked from the patch's size, read the wrong level. Hull shaders that sample get a copy
// whose explicit-LOD samples divide their coordinates by the size of the mip the LOD names; a draw
// whose HS samplers are all unnormalized runs that copy (with an id of its own) and binds its
// samplers as normalized ones (the S#'s LOD range and clamps as they are).
[[nodiscard]] bool UnnormalizedSamplingFix();
// The SPIR-V of a stage whose explicit-LOD samples (OpImageSampleExplicitLod with a Lod and
// nothing but offsets) take their coordinates divided by the size of the mip the LOD names, the
// nearest within the image (an array layer as it is); empty when it has none, or uses a sampler
// any other way (all or nothing: the copy's samplers are all made normalized).
[[nodiscard]] std::vector<uint32_t> SpirvNormalizedExplicitLod(std::span<const uint32_t> spirv);

namespace NormalizedLod {

struct Lookup {
	vk::ShaderModule module     = nullptr; // the SpirvNormalizedExplicitLod copy, null if none
	uint64_t         variant_id = 0;       // its program id (its pipelines differ)
};
void   Register(uint64_t program_id, vk::ShaderModule module, uint64_t variant_id);
void   Forget(vk::Device device, uint64_t program_id);
[[nodiscard]] Lookup Find(uint64_t program_id);

} // namespace NormalizedLod

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXELRATEVARIANT_H_
