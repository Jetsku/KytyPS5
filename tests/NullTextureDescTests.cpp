#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(1);
	}
}

int main() {
	ShaderRecompiler::IR::ImageResource resource {};
	resource.numeric_class = Prospero::TextureNumericClass::Float;
	resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
	resource.depth_compare = true;
	const auto depth = NullTextureDesc(resource, TextureCache::BindingType::Texture);
	Check(depth.info.pixel_format == vk::Format::eD32Sfloat &&
	          depth.view_info.format == vk::Format::eD32Sfloat,
	      "null comparison texture must use D32_SFLOAT for both image and view");
	Check(depth.view_info.aspect == vk::ImageAspectFlagBits::eDepth,
	      "null comparison texture must use the depth aspect");
	Check(depth.view_info.usage == vk::ImageUsageFlagBits::eSampled &&
	          depth.info.bytes_per_block == 4 && depth.info.samples == 1,
	      "null comparison texture changed sampled usage or element layout");

	resource.depth_compare = false;
	const auto color = NullTextureDesc(resource, TextureCache::BindingType::Texture);
	Check(color.info.pixel_format == vk::Format::eR32Sfloat &&
	          color.view_info.format == vk::Format::eR32Sfloat &&
	          color.view_info.aspect == vk::ImageAspectFlagBits::eColor,
	      "ordinary float null texture must retain R32_SFLOAT/color");
	resource.numeric_class = Prospero::TextureNumericClass::Uint;
	const auto integer = NullTextureDesc(resource, TextureCache::BindingType::Storage);
	Check(integer.info.pixel_format == vk::Format::eR32Uint &&
	          integer.view_info.aspect == vk::ImageAspectFlagBits::eColor &&
	          integer.view_info.usage == vk::ImageUsageFlagBits::eStorage,
	      "non-comparison unsigned storage stub changed");
	resource.atomic64 = true;
	const auto atomic = NullTextureDesc(resource, TextureCache::BindingType::Storage);
	Check(atomic.view_info.format == vk::Format::eR64Uint &&
	          atomic.info.bytes_per_block == 8,
	      "non-comparison atomic64 stub changed");
	resource.atomic64 = false;
	resource.numeric_class = Prospero::TextureNumericClass::Sint;
	const auto signed_integer = NullTextureDesc(resource, TextureCache::BindingType::Storage);
	Check(signed_integer.info.pixel_format == vk::Format::eR32Sint &&
	          signed_integer.view_info.aspect == vk::ImageAspectFlagBits::eColor,
	      "non-comparison signed storage stub changed");
	std::puts("null_texture_desc: 7 descriptor checks passed (no Vulkan device created)");
}
