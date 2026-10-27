#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Canonical module-affecting resource state. Runtime addresses and descriptor payloads remain in
// ResourceSnapshot and therefore do not create shader permutations.
struct ResourceSpecialization {
	struct Buffer {
		uint32_t               packed_stride                   = 0;
		Prospero::BufferFormat descriptor_format               = Prospero::BufferFormat::kInvalid;
		uint32_t               descriptor_swizzle              = DstSel(4, 5, 6, 7);
		bool                   zero_stride_oob                 = false;
		bool                   operator==(const Buffer&) const = default;
	};

	struct Image {
		Prospero::TextureNumericClass numeric_class = Prospero::TextureNumericClass::Unsupported;
		Decoder::ImageDimension       dimension     = Decoder::ImageDimension::Unknown;
		uint32_t                      mip_count     = 1;
		Prospero::BufferFormat        conversion_format          = Prospero::BufferFormat::kInvalid;
		uint32_t                      shader_swizzle             = ShaderImageIdentitySwizzle;
		uint32_t                      indirect_root              = ImageResource::NoIndirectImage;
		uint32_t                      indirect_mapping_offset    = 0;
		uint32_t                      indirect_search_iterations = 0;
		bool                          cube                       = false;
		bool                          fmask                      = false;
		bool                          bindless                   = false;
		// KYTY_EQAA_MIXED, EQAA FMASK only: colour fragments stored per pixel (1, 2, 4 or 8) when
		// the format has fewer fragments than coverage samples, so samples share fragments; 0
		// otherwise (the identity map).
		uint8_t                       fmask_fragments            = 0;
		// KYTY_FLOAT_STORE_CLAMP (ImageResource::clamp_half_float).
		bool                          clamp_half_float           = false;
		bool                          operator==(const Image&) const = default;
	};

	std::vector<Buffer> buffers;
	std::vector<Image>  images;

	bool operator==(const ResourceSpecialization&) const = default;
};

// Extracts the descriptor/SRT value graph before resource specialization. The returned plan owns
// its values and is independent of the translated shader CFG.
ResourcePlan ExtractResourcePlan(const Program& program);

// Refreshes cached resources and specialization in place. A failed refresh must not be used.
// Threads may refresh one sealed plan concurrently when each supplies its own scratch.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          EvaluationScratch& scratch, ResourceSnapshot& snapshot,
                          ResourceSpecialization& specialization);
// Uses this thread's scratch.
bool MaterializeResources(const ResourcePlan& program, const SrtRuntime& runtime,
                          ResourceSnapshot& snapshot, ResourceSpecialization& specialization);

// KYTY_SPEC_UNMAPPED_NULL: whether a V# base is inside a guest mapping (the binding's test; the V#
// is nulled when not). Null: off. Set once by the renderer before draws.
void SetUnmappedBufferProbe(SrtMappedRange probe);
// Tests: KYTY_MS_DEPTH_SAMPLE0's specialization part (multisample depth T#s read through a
// single-sample dimension specialize as single-sample images) without the environment.
void SetMsDepthSample0ForTesting(bool enabled);
// Tests: KYTY_EQAA_MIXED's FMASK part (EQAA FMASK formats map sample i to fragment i mod
// fragments) without the environment.
void SetEqaaFmaskMapForTesting(bool enabled);
// V#s specialized as null by that rule so far.
[[nodiscard]] uint64_t UnmappedBufferNulls();

// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
