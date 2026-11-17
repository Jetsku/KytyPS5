#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEAR_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEAR_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
class Image;
struct GraphicContext;

// Native DCC clear materialization of GPU-written metadata (KYTY_DCC_GPU=1). TextureCache owns
// eligibility and guest-memory tracking; this helper only records native work.
//
// Per metadata slice it reproduces TextureCache's CPU fallback exactly, on the GPU and without a
// readback: a validation dispatch reads the slice, and when its first byte is a clear code the CPU
// decoder accepts for this binding and every byte equals it, an indirect dispatch writes the
// clear value to every texel of the image layer and consumes the key (all bytes 0xFF). Otherwise
// that dispatch has zero groups. The clear value's texel bits come from vkCmdClearColorImage on a
// small palette image in the view format (the conversion the CPU path's clear uses) and are
// stored through an unsigned integer view of the same texel size, so any color format whose texel
// is 1, 2, 4, 8 or 16 bytes is supported.
class DccClearHelper final {
public:
	// Codes in palette order. 0x10: the PS5 register clear (KYTY_DCC_CLEAR_KEY_10, textureCache.cpp).
	static constexpr std::array<uint8_t, 6> ClearCodes {0x00, 0x20, 0x40, 0x80, 0xC0, 0x10};
	// The CPU decoder's clear value per code in ClearCodes order; nullopt: the code is rejected.
	using ClearValues = std::array<std::optional<vk::ClearColorValue>, ClearCodes.size()>;

	enum class Support : uint8_t {
		Ok,
		Disabled,    // device lacks a required feature/limit
		Format,      // the view format has no unsigned integer storage alias
		Unsupported, // native image usage/flags/shape or size limits
	};

	DccClearHelper(GraphicContext& graphics, CommandScheduler& scheduler);
	~DccClearHelper();
	KYTY_CLASS_NO_COPY(DccClearHelper);

	[[nodiscard]] bool   Available() const noexcept { return m_supported; }
	[[nodiscard]] Support SupportsFormat(vk::Format view_format) const;
	[[nodiscard]] Support SupportsImage(const Image& image, vk::Format view_format,
	                                    uint64_t metadata_size) const;
	// Inspects one metadata slice and conditionally clears one layer of `image` (single mip) as
	// the CPU fallback would. The image and canonical metadata buffer must remain alive through
	// this scheduler tick. metadata_offset must be 4-byte aligned.
	// keep_keys: the slice's bytes are left as they are after a clear (video-out bindings).
	// decision_copy: the decision (4 bytes: the clear's group count, 0 = no clear) is copied to it at
	// decision_offset behind the clear (verification).
	void RecordSlice(Image& image, vk::Format view_format, uint32_t layer, vk::Buffer metadata,
	                 uint64_t metadata_offset, uint64_t metadata_size, const ClearValues& values,
	                 bool keep_keys = false, vk::Buffer decision_copy = nullptr,
	                 uint64_t decision_offset = 0);
	// Push::decodable_mask bit: keep the keys (gpu_dcc_clear.comp).
	static constexpr uint32_t KeepKeysBit = 0x80000000u;

private:
	struct Push {
		uint32_t metadata_base_words;
		uint32_t metadata_words;
		uint32_t width;
		uint32_t height;
		uint32_t clear_groups;
		uint32_t decodable_mask;
		uint32_t texel_bytes;
	};
	[[nodiscard]] static vk::Format AliasFormat(vk::Format view_format, uint32_t& texel_bytes);
	void RecordPalette(vk::CommandBuffer command, vk::Format view_format, uint32_t texel_bytes,
	                   const ClearValues& values);

	static constexpr uint32_t WorkgroupSize   = 128;
	static constexpr uint64_t MaxMetadataSize = 1024 * 1024;
	static constexpr uint64_t ScratchSize     = 32;
	static constexpr uint64_t PaletteSize     = ClearCodes.size() * 16;

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	bool                    m_supported         = false;
	vk::DescriptorSetLayout m_descriptor_layout = nullptr;
	vk::PipelineLayout      m_pipeline_layout   = nullptr;
	vk::Pipeline            m_validate_pipeline = nullptr;
	vk::Pipeline            m_clear_pipeline    = nullptr;
	// Reuse is ordered by GPU barriers; no CPU recycling or completion callback is needed.
	std::unique_ptr<Buffer> m_scratch;
	std::unique_ptr<Buffer> m_palette;
	// KYTY_DCC_PALETTE_REUSE: what the palette buffer holds since the last RecordPalette.
	bool                                              m_palette_valid  = false;
	vk::Format                                        m_palette_format = vk::Format::eUndefined;
	uint32_t                                          m_palette_mask   = 0;
	std::array<std::array<uint32_t, 4>, ClearCodes.size()> m_palette_bits {};
	// One 1x1 image with a layer per clear code, per view format.
	struct PaletteImages;
	std::unique_ptr<PaletteImages> m_palette_images;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DCCCLEAR_H_
