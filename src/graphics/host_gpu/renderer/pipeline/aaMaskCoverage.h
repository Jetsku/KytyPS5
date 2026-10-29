#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_AAMASKCOVERAGE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_AAMASKCOVERAGE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <span>
#include <vector>

// KYTY_AA_MASK_COVERAGE=1 (default off, startup; adapted from the GT7 fork's fc4111066, "ghost
// cars", issue 30). PA_SC_AA_MASK holds the samples each pixel of a 2x2 quad may cover (16 bits a
// pixel: X0Y0 and X1Y0 in the first register, X0Y1 and X1Y1 in the second). Vulkan's sample mask is
// the same for every pixel, so a draw whose masks differ by pixel position (GT7's ghost cars cover
// half their pixels in a checkerboard) drew solid. Such a draw now runs a copy of its pixel shader
// that writes gl_SampleMask from its pixel's position (without early fragment tests, so depth and
// stencil follow the mask as on the guest), or a coverage-only fragment shader when it has none.
// The masks, reduced to the draw's raster samples, are constants of the copy and part of the
// pipeline key (PipelineStaticParameters::aa_mask).
//
// Adaptation: the fork rereads a pixel shader's SPIR-V from its shader disk cache entry. Here the
// pipeline cache keeps every pixel permutation's SPIR-V zstd-compressed in memory while the switch
// is on (Register), so the copies need no disk cache. A draw whose pixel shader runs the EQAA
// per-pixel copy (KYTY_EQAA_MIXED on NV mixed samples, pixelRateVariant.h) gets the AA copy of that
// per-pixel copy (sample index 0, then the mask), so both rewrites apply. The copies are made on a
// thread of their own; until one is ready, and for shaders that export coverage themselves, the
// draw runs as before.
namespace Libs::Graphics {

// The copy of a fragment shader that writes gl_SampleMask from the AA masks (already reduced to
// the draw's samples); empty when the shader writes gl_SampleMask itself or has a form the
// rewrite does not handle.
[[nodiscard]] std::vector<uint32_t> SpirvWithAaMask(std::span<const uint32_t> spirv, uint32_t mask01,
                                                    uint32_t mask23);
// A fragment shader with no outputs but gl_SampleMask from the AA masks.
[[nodiscard]] std::vector<uint32_t> CoverageOnlyFragmentSpirv(uint32_t mask01, uint32_t mask23);

namespace AaMaskCoverage {

[[nodiscard]] bool Enabled();

// A pixel program's SPIR-V (by its ShaderProgram id), kept compressed for its copies.
void Register(uint64_t program_id, uint64_t shader_hash, std::span<const uint32_t> spirv);
// The program is gone (its copies' modules are destroyed).
void Forget(vk::Device device, uint64_t program_id);
// The copy of program `program_id` (0: the coverage-only shader), from its EQAA per-pixel copy
// with `pixel_rate`, for these masks; null while it is being made (the request queues it) or when
// it cannot be made. Any thread.
[[nodiscard]] vk::ShaderModule Request(vk::Device device, uint64_t program_id, bool pixel_rate,
                                       uint32_t mask01, uint32_t mask23);
// As Request, without queueing anything (the pipeline build, after the key took the copy).
[[nodiscard]] vk::ShaderModule Lookup(uint64_t program_id, bool pixel_rate, uint32_t mask01,
                                      uint32_t mask23);
// Stops the worker and destroys the copies' modules (the pipeline cache's destruction).
void Shutdown(vk::Device device);

} // namespace AaMaskCoverage

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_AAMASKCOVERAGE_H_
