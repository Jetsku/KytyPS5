#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_

#include "graphics/guest_gpu/gpu_defs.h"

namespace Libs::Graphics::Prospero {

enum class ChannelOrderSupport : uint8_t {
	kNone,
	kStandardOnly,
	kAll,
};

struct RenderTargetFormatEncoding {
	BufferFormat        buffer_format = BufferFormat::kInvalid;
	uint8_t             components    = 0;
	ChannelOrderSupport order_support = ChannelOrderSupport::kNone;

	[[nodiscard]] constexpr bool IsValid() const {
		return buffer_format != BufferFormat::kInvalid && components >= 1u && components <= 4u &&
		       order_support != ChannelOrderSupport::kNone;
	}

	[[nodiscard]] constexpr bool SupportsOrder(ChannelOrder order) const {
		switch (order_support) {
			case ChannelOrderSupport::kStandardOnly: return order == ChannelOrder::kStandard;
			case ChannelOrderSupport::kAll:
				switch (order) {
					case ChannelOrder::kStandard:
					case ChannelOrder::kAlt:
					case ChannelOrder::kReversed:
					case ChannelOrder::kAltReversed: return true;
				}
				return false;
			default: return false;
		}
	}
};

enum class TextureNumericClass { Unsupported, Float, Uint, Sint };

RenderTargetFormatEncoding ResolveRenderTargetFormat(ChannelLayout layout, ChannelType type);
uint32_t                   NumBytesPerElement(BufferFormat format);
uint32_t                   BlockCompressedBytesPerBlock(BufferFormat format);
uint32_t                   RenderTargetBytesPerElement(BufferFormat format);
bool                       IsFmaskTextureFormat(BufferFormat format);
/// Whether the value names a format at all. The encoding leaves wide gaps, everything between 136
/// and 156 for instance, so a value well inside the enum's range can still be no format, and that
/// is what separates a descriptor from eight dwords that merely sit where one was expected.
bool                       IsDefinedBufferFormat(BufferFormat format);
bool                       IsSrgbFormat(BufferFormat format);
TextureNumericClass        SampledTextureNumericClass(BufferFormat format);
BufferFormat               RemapTextureFormat(BufferFormat format);
// KYTY_FMT30_UFLOAT=1 (Erlou93/KytyPS5-Yotei FIX_RT_11_11_10_FLOAT, default on there): format 30
// (COLOR_11_11_10 UNORM) render targets and textures are packed unsigned floats (B10G11R11_UFLOAT).
bool                       Fmt30AsFloat();
// KYTY_SRGB8_VIEW=1: a sampled 8 sRGB texture (k8Srgb, an R8_UNORM image) is read through an
// R8_SRGB view, so the sampler decodes each texel before filtering as the PS5 does (Erlou93/
// KytyPS5-Yotei); it replaces KYTY_SRGB8_DECODE's decode after filtering for 8 sRGB. Active when
// the device filters R8_SRGB (SetSrgb8ViewSupported, set once by the device layer). 8_8 sRGB keeps
// the shader decode: R8G8_SRGB is not a sampled format on the RTX 3090.
bool                       Srgb8ViewActive();
void                       SetSrgb8ViewSupported(bool supported);

} // namespace Libs::Graphics::Prospero

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GUEST_GPU_GPU_FORMAT_H_ */
