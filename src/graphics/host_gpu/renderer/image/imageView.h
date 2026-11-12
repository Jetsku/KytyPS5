#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEVIEW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEVIEW_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

namespace Libs::Graphics {

namespace ImageViewOps {

[[nodiscard]] vk::ImageAspectFlags DepthAspectMask(vk::Format format);
[[nodiscard]] bool                 FormatsCompatible(vk::Format base, vk::Format view) noexcept;

[[nodiscard]] inline bool IsFormatDepthCompatible(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD32Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32Uint:
		case vk::Format::eD16Unorm:
		case vk::Format::eR16Unorm: return true;
		default: return false;
	}
}
} // namespace ImageViewOps

[[nodiscard]] inline bool IsSupportedSampledDepthFormat(vk::Format image_format,
                                                        vk::Format view_format) noexcept {
	return DepthAspectTransferFormat(image_format) != vk::Format::eUndefined &&
	       ImageViewOps::IsFormatDepthCompatible(view_format);
}

[[noreturn]] inline void UnsupportedColorView(const char* usage, vk::Format image_format,
                                              vk::Format view_format, uint32_t swizzle) noexcept {
	EXIT("unsupported %s color image view: image_format=%d view_format=%d swizzle=0x%03x\n", usage,
	     static_cast<int>(image_format), static_cast<int>(view_format), swizzle);
}

[[nodiscard]] inline vk::Format SrgbUnormViewFormat(vk::Format image_format) noexcept {
	switch (image_format) {
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Srgb: return vk::Format::eR8G8B8A8Unorm;
		case vk::Format::eBc1RgbaSrgbBlock: return vk::Format::eBc1RgbaUnormBlock;
		case vk::Format::eBc2SrgbBlock: return vk::Format::eBc2UnormBlock;
		case vk::Format::eBc3SrgbBlock: return vk::Format::eBc3UnormBlock;
		case vk::Format::eBc7SrgbBlock: return vk::Format::eBc7UnormBlock;
		default: return image_format;
	}
}

[[nodiscard]] inline bool IsSupportedSampledColorView(vk::Format image_format,
                                                      vk::Format view_format,
                                                      uint32_t   swizzle) noexcept {
	return IsValidImageSwizzle(swizzle) &&
	       ImageViewOps::FormatsCompatible(image_format, view_format);
}

[[nodiscard]] inline uint32_t
SelectSampledColorView(vk::Format image_format, vk::Format view_format, uint32_t swizzle) noexcept {
	if (IsSupportedSampledColorView(image_format, view_format, swizzle)) {
		return swizzle;
	}
	UnsupportedColorView("sampled", image_format, view_format, swizzle);
}

[[nodiscard]] inline bool IsSupportedSampledDepthView(vk::Format image_format,
                                                      vk::Format view_format,
                                                      uint32_t   swizzle) noexcept {
	return IsSupportedSampledDepthFormat(image_format, view_format) &&
	       IsValidImageSwizzle(swizzle, 1);
}

// A shader samples a 32-bit colour format (e.g. R16G16_SFLOAT) of memory the cache holds as a plain
// D32 depth image. A depth image can only be viewed as a depth-compatible format, so the colour
// interpretation needs its own alias image, filled with the depth image's raw 32-bit texels.
[[nodiscard]] inline bool NeedsColorAliasForSampledDepth(vk::Format cached_format,
                                                         uint32_t   cached_bytes_per_block,
                                                         vk::Format requested_format,
                                                         uint32_t   requested_bytes_per_block) noexcept {
	return cached_format == vk::Format::eD32Sfloat &&
	       DepthAspectTransferFormat(requested_format) == vk::Format::eUndefined &&
	       requested_format != vk::Format::eUndefined &&
	       !ImageViewOps::IsFormatDepthCompatible(requested_format) &&
	       cached_bytes_per_block == sizeof(uint32_t) &&
	       requested_bytes_per_block == sizeof(uint32_t);
}

[[nodiscard]] inline bool
IsSupportedSampledDepthResource(const ShaderRecompiler::IR::ImageResource& resource) noexcept {
	if (resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return false;
	}
	if (resource.numeric_class == Prospero::TextureNumericClass::Unsupported) {
		return false;
	}
	if (resource.numeric_class != Prospero::TextureNumericClass::Float && resource.depth_compare) {
		return false;
	}
	return ((resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::None &&
	         resource.mip_count == 1u) ||
	        (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic &&
	         resource.mip_count != 0u)) &&
	       resource.read && !resource.written && !resource.atomic;
}

inline void ValidateStorageColorView(vk::Format image_format, vk::Format view_format,
                                     uint32_t swizzle) noexcept {
	if (!ImageViewOps::FormatsCompatible(image_format, view_format) ||
	    !IsValidImageSwizzle(swizzle)) {
		UnsupportedColorView("storage", image_format, view_format, swizzle);
	}
}

[[nodiscard]] inline bool
IsSupportedStorageImageResource(const ShaderRecompiler::IR::ImageResource& resource) noexcept {
	return resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage &&
	       (resource.numeric_class == Prospero::TextureNumericClass::Float ||
	        resource.numeric_class == Prospero::TextureNumericClass::Uint) &&
	       (resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D ||
	        resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray ||
	        resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D ||
	        resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D ||
	        resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray) &&
	       ((resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::None &&
	         resource.mip_count == 1u) ||
	        (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic &&
	         resource.mip_count != 0u)) &&
	       resource.written &&
	       (!resource.atomic ||
	        (resource.numeric_class == Prospero::TextureNumericClass::Uint && resource.read)) &&
	       !resource.depth_compare;
}

inline void
ValidateStorageImageResource(const ShaderRecompiler::IR::ImageResource& resource) noexcept {
	if (!IsSupportedStorageImageResource(resource)) {
		EXIT("unsupported storage color image resource: class=%u numeric=%u dimension=%u mip=%u "
		     "read=%d written=%d atomic=%d depth_compare=%d\n",
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, resource.atomic, resource.depth_compare);
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEVIEW_H_
