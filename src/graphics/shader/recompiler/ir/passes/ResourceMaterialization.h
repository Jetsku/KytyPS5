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
		// KYTY_RUNTIME_IMAGES (ImageResource::runtime).
		bool                          runtime                    = false;
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

// KYTY_BINDLESS_SAMPLERS: the bindless samplers of a program (SamplerResource::bindless), and the
// flattened-SRT offset of the two words (region base, record count) of the index-th of them, which
// MaterializeResources appends right after the program's SRT slots.
[[nodiscard]] inline uint32_t BindlessSamplerCount(const ShaderInfo& info) {
	uint32_t count = 0;
	for (const auto& sampler: info.samplers) {
		count += sampler.bindless ? 1u : 0u;
	}
	return count;
}
[[nodiscard]] inline uint32_t BindlessSamplerMappingOffset(size_t srt_slots, uint32_t index) {
	return static_cast<uint32_t>(srt_slots) + 2u * index;
}

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

// KYTY_WORKGROUP_IMAGES: whether the plan enumerates images over the dispatch's workgroup counts
// (DescriptorSource::IndirectDescriptor::workgroup_axis). Such a plan's materialization depends on
// SrtRuntime::workgroup_counts, which no read certificate covers.
[[nodiscard]] bool PlanReadsWorkgroupCounts(const ResourcePlan& program);
// Whether a materialization on this thread failed since the last call because the runtime had no
// workgroup counts (an indirect dispatch whose arguments were not read); clears the flag.
[[nodiscard]] bool TakeWorkgroupCountsMissing();

// KYTY_SPEC_UNMAPPED_NULL: whether a V# base is inside a guest mapping (the binding's test; the V#
// is nulled when not). Null: off. Set once by the renderer before draws.
void SetUnmappedBufferProbe(SrtMappedRange probe);
// KYTY_NULL_BUFFER_SHARE: records the live buffer slots of a journaled permutation as the slots' last
// live specializations, so a null V# seen first in a later session takes them (Erlou93 M115 seeding).
void SeedNullBufferShare(uint64_t shader_hash, uint32_t stage, uint32_t user_data_base,
                         uint32_t user_data_count, const ResourceSpecialization& specialization);
// Tests: KYTY_MS_DEPTH_SAMPLE0's specialization part (multisample depth T#s read through a
// single-sample dimension specialize as single-sample images) without the environment.
void SetMsDepthSample0ForTesting(bool enabled);
// Tests: KYTY_EQAA_MIXED's FMASK part (EQAA FMASK formats map sample i to fragment i mod
// fragments) without the environment.
void SetEqaaFmaskMapForTesting(bool enabled);
// V#s specialized as null by that rule so far.
[[nodiscard]] uint64_t UnmappedBufferNulls();

// Why ApplyResourceSpecialization cannot apply this specialization to the program (it would stop
// the emulator), or null: the permutation is skipped instead.
[[nodiscard]] const char* UnsupportedResourceSpecialization(const Program&                program,
                                                            const ResourceSpecialization& specialization);
// Applies an already-derived specialization to native IR before layout and emission.
void ApplyResourceSpecialization(Program& program, const ResourceSpecialization& specialization);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCEMATERIALIZATION_H_ */
