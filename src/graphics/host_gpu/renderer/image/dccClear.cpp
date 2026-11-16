#include "graphics/host_gpu/renderer/image/dccClear.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/profiler.h"
#include "gpu_dcc_shaders/gpu_dcc_clear_spv.h"
#include "gpu_dcc_shaders/gpu_dcc_validate_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <span>

namespace Libs::Graphics {

struct DccClearHelper::PaletteImages {
	std::map<vk::Format, std::unique_ptr<VulkanImage>> images;
};

DccClearHelper::DccClearHelper(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler),
      m_palette_images(std::make_unique<PaletteImages>()) {
	static_assert(sizeof(Push) == 28);
	const auto& limits = graphics.physical_device_properties.limits;
	m_supported = graphics.max_push_descriptors >= 4 &&
	              limits.maxComputeWorkGroupInvocations >= WorkgroupSize &&
	              limits.maxComputeWorkGroupSize[0] >= WorkgroupSize &&
	              limits.maxComputeWorkGroupCount[0] != 0 &&
	              limits.maxComputeSharedMemorySize >= sizeof(uint32_t) &&
	              limits.maxPushConstantsSize >= sizeof(Push) &&
	              graphics.StorageMinAlignment() <= std::numeric_limits<uint32_t>::max();
	if (!m_supported) {
		return;
	}

	const std::array<vk::DescriptorSetLayoutBinding, 4> bindings {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {3, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor_info.pBindings = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                              &m_descriptor_layout),
	                     "create DCC clear descriptor layout");
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &m_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &push_range;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout_info, nullptr,
	                                                         &m_pipeline_layout),
	                     "create DCC clear pipeline layout");
	const auto create_pipeline = [&](std::span<const uint32_t> spirv, vk::Pipeline& pipeline) {
		const auto module = CompileSPV(spirv, graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName = "main";
		vk::ComputePipelineCreateInfo create {};
		create.stage = stage;
		create.layout = m_pipeline_layout;
		const auto result =
		    graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
		graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create DCC clear compute pipeline");
	};
	create_pipeline(GPU_DCC_VALIDATE_SPV, m_validate_pipeline);
	create_pipeline(GPU_DCC_CLEAR_SPV, m_clear_pipeline);
	m_scratch = std::make_unique<Buffer>(
	    graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
	        vk::BufferUsageFlagBits::eTransferSrc,
	    ScratchSize);
	m_palette = std::make_unique<Buffer>(
	    graphics, scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
	    PaletteSize);
}

DccClearHelper::~DccClearHelper() {
	// The owning TextureCache is destroyed after RenderContext drains its scheduler.
	for (auto& [format, image]: m_palette_images->images) {
		if (image != nullptr && image->image != nullptr) {
			m_graphics.DeleteImage(*image);
		}
	}
	if (m_validate_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_validate_pipeline, nullptr);
	}
	if (m_clear_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_clear_pipeline, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

vk::Format DccClearHelper::AliasFormat(vk::Format view_format, uint32_t& texel_bytes) {
	// Same compatibility class (texel size) as the view format: a mutable-format image can be
	// viewed through it, and its texels hold exactly the view format's encoded bits.
	constexpr std::array<std::pair<vk::Format, uint32_t>, 5> aliases {{
	    {vk::Format::eR8Uint, 1},
	    {vk::Format::eR16Uint, 2},
	    {vk::Format::eR32Uint, 4},
	    {vk::Format::eR32G32Uint, 8},
	    {vk::Format::eR32G32B32A32Uint, 16},
	}};
	for (const auto& [alias, bytes]: aliases) {
		if (ImageViewOps::FormatsCompatible(alias, view_format) &&
		    ImageViewOps::FormatsCompatible(view_format, alias)) {
			texel_bytes = bytes;
			return alias;
		}
	}
	texel_bytes = 0;
	return vk::Format::eUndefined;
}

DccClearHelper::Support DccClearHelper::SupportsFormat(vk::Format view_format) const {
	if (!m_supported) {
		return Support::Disabled;
	}
	uint32_t   texel_bytes = 0;
	const auto alias       = AliasFormat(view_format, texel_bytes);
	// Depth/stencil and block-compressed formats have their own compatibility classes.
	if (alias == vk::Format::eUndefined) {
		return Support::Format;
	}
	const auto alias_features =
	    m_graphics.GetFormatProperties(alias).optimalTilingFeatures;
	const auto view_features = m_graphics.GetFormatProperties(view_format).optimalTilingFeatures;
	if (!(alias_features & vk::FormatFeatureFlagBits::eStorageImage) ||
	    !(view_features & vk::FormatFeatureFlagBits::eTransferDst) ||
	    !(view_features & vk::FormatFeatureFlagBits::eTransferSrc)) {
		return Support::Format;
	}
	return Support::Ok;
}

DccClearHelper::Support DccClearHelper::SupportsImage(const Image& image, vk::Format view_format,
                                                      uint64_t metadata_size) const {
	const auto format_support = SupportsFormat(view_format);
	if (format_support != Support::Ok) {
		return format_support;
	}
	uint32_t    texel_bytes = 0;
	const auto  alias       = AliasFormat(view_format, texel_bytes);
	const auto& native      = image.backing;
	if (native.image == nullptr || native.image_type != vk::ImageType::e2D ||
	    native.samples != 1 || native.mip_levels != 1 || native.extent.depth != 1 ||
	    native.extent.width == 0 || native.extent.height == 0 ||
	    !(native.usage & vk::ImageUsageFlagBits::eStorage) ||
	    !ImageViewOps::FormatsCompatible(native.format, alias) ||
	    (native.format != alias && !(native.flags & vk::ImageCreateFlagBits::eMutableFormat)) ||
	    metadata_size == 0 || metadata_size > MaxMetadataSize || metadata_size % 4 != 0) {
		return Support::Unsupported;
	}
	const auto& limits = m_graphics.physical_device_properties.limits;
	const uint64_t pixels = static_cast<uint64_t>(native.extent.width) * native.extent.height;
	// Keep the shader's rounded invocation count and index arithmetic within uint32_t.
	const uint64_t max_elements = std::numeric_limits<uint32_t>::max() - (WorkgroupSize - 1u);
	const auto alignment = std::max<uint64_t>(m_graphics.StorageMinAlignment(), 4);
	if (pixels > max_elements || metadata_size + alignment - 1 > limits.maxStorageBufferRange) {
		return Support::Unsupported;
	}
	return Support::Ok;
}

// KYTY_DCC_PALETTE_REUSE (live, default off): a DCC clear whose view format and decoded clear values
// equal those of the previous one reuses the palette buffer as it is (only RecordPalette writes it,
// in queue order) instead of rebuilding it with three barriers, up to six image clears and a copy.
static Live::Switch g_dcc_palette_reuse("KYTY_DCC_PALETTE_REUSE", Live::ParseDefaultOff);

void DccClearHelper::RecordPalette(vk::CommandBuffer command, vk::Format view_format,
                                   uint32_t texel_bytes, const ClearValues& values) {
	uint32_t                                               mask = 0;
	std::array<std::array<uint32_t, 4>, ClearCodes.size()> bits {};
	for (uint32_t index = 0; index < ClearCodes.size(); ++index) {
		if (values[index]) {
			mask |= 1u << index;
			std::memcpy(bits[index].data(), &*values[index], sizeof(bits[index]));
		}
	}
	if (g_dcc_palette_reuse.On() && m_palette_valid && m_palette_format == view_format &&
	    m_palette_mask == mask && m_palette_bits == bits) {
		return;
	}
	m_palette_valid  = true;
	m_palette_format = view_format;
	m_palette_mask   = mask;
	m_palette_bits   = bits;
	auto& slot = m_palette_images->images[view_format];
	if (slot == nullptr) {
		slot = std::make_unique<VulkanImage>();
		vk::ImageCreateInfo create {};
		create.imageType     = vk::ImageType::e2D;
		create.extent        = vk::Extent3D {1, 1, 1};
		create.mipLevels     = 1;
		create.arrayLayers   = static_cast<uint32_t>(ClearCodes.size());
		create.format        = view_format;
		create.tiling        = vk::ImageTiling::eOptimal;
		create.initialLayout = vk::ImageLayout::eUndefined;
		create.usage         = vk::ImageUsageFlagBits::eTransferSrc |
		               vk::ImageUsageFlagBits::eTransferDst;
		create.sharingMode = vk::SharingMode::eExclusive;
		create.samples     = vk::SampleCountFlagBits::e1;
		if (!m_graphics.CreateImage(create, *slot)) {
			EXIT("DCC clear: failed to create the palette image (format %d)\n",
			     static_cast<int>(view_format));
		}
	}
	const auto palette_image = slot->image;
	const vk::ImageSubresourceRange all_layers {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
	                                            static_cast<uint32_t>(ClearCodes.size())};

	// Earlier copies out of the image and shader reads of the palette buffer precede the
	// rewrites below (queue order; one palette serves every record).
	vk::ImageMemoryBarrier2 image_barrier {};
	image_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	image_barrier.srcAccessMask       = vk::AccessFlagBits2::eNone;
	image_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	image_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	image_barrier.oldLayout           = vk::ImageLayout::eUndefined;
	image_barrier.newLayout           = vk::ImageLayout::eTransferDstOptimal;
	image_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	image_barrier.image               = palette_image;
	image_barrier.subresourceRange    = all_layers;
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eNone;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = m_palette->Handle();
	buffer_barrier.offset              = 0;
	buffer_barrier.size                = PaletteSize;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount  = 1;
	dependency.pImageMemoryBarriers     = &image_barrier;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	command.pipelineBarrier2(dependency);

	// The same conversion the CPU fallback's ClearImage applies to the decoded clear value.
	std::array<vk::BufferImageCopy, ClearCodes.size()> copies {};
	uint32_t copy_count = 0;
	for (uint32_t index = 0; index < ClearCodes.size(); ++index) {
		if (!values[index]) {
			continue;
		}
		const vk::ImageSubresourceRange layer {vk::ImageAspectFlagBits::eColor, 0, 1, index, 1};
		command.clearColorImage(palette_image, vk::ImageLayout::eTransferDstOptimal,
		                        &*values[index], 1, &layer);
		auto& copy                 = copies[copy_count++];
		copy.bufferOffset          = uint64_t {index} * 16u;
		copy.imageSubresource      = {vk::ImageAspectFlagBits::eColor, 0, index, 1};
		copy.imageExtent           = vk::Extent3D {1, 1, 1};
	}
	(void)texel_bytes;
	image_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	image_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	image_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	image_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	image_barrier.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	image_barrier.newLayout     = vk::ImageLayout::eTransferSrcOptimal;
	dependency                          = {};
	dependency.imageMemoryBarrierCount  = 1;
	dependency.pImageMemoryBarriers     = &image_barrier;
	command.pipelineBarrier2(dependency);
	if (copy_count != 0) {
		command.copyImageToBuffer(palette_image, vk::ImageLayout::eTransferSrcOptimal,
		                          m_palette->Handle(), copy_count, copies.data());
	}
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	buffer_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	dependency                          = {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	command.pipelineBarrier2(dependency);
}

void DccClearHelper::RecordSlice(Image& image, vk::Format view_format, uint32_t layer,
                                 vk::Buffer metadata, uint64_t metadata_offset,
                                 uint64_t metadata_size, const ClearValues& values, bool keep_keys,
                                 vk::Buffer decision_copy, uint64_t decision_offset) {
	KYTY_GPU_OP_SITE("dcc.clear");
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(SupportsImage(image, view_format, metadata_size) != Support::Ok ||
	        metadata == nullptr || metadata_offset % 4 != 0 || layer >= image.backing.layers ||
	        metadata_offset > std::numeric_limits<uint64_t>::max() - metadata_size);
	uint32_t   texel_bytes = 0;
	const auto alias       = AliasFormat(view_format, texel_bytes);
	uint32_t   decodable   = 0;
	for (uint32_t index = 0; index < ClearCodes.size(); ++index) {
		decodable |= values[index] ? 1u << index : 0u;
	}
	const auto alignment = std::max<uint64_t>(m_graphics.StorageMinAlignment(), 4);
	const auto descriptor_offset = Common::AlignDown(metadata_offset, alignment);
	const auto prefix = metadata_offset - descriptor_offset;
	const auto pixels = static_cast<uint64_t>(image.backing.extent.width) * image.backing.extent.height;
	const auto elements = std::max(pixels, metadata_size / sizeof(uint32_t));
	const auto groups = std::min<uint64_t>(
	    (elements + WorkgroupSize - 1) / WorkgroupSize,
	    m_graphics.physical_device_properties.limits.maxComputeWorkGroupCount[0]);
	const Push push {static_cast<uint32_t>(prefix / sizeof(uint32_t)),
	                 static_cast<uint32_t>(metadata_size / sizeof(uint32_t)),
	                 image.backing.extent.width,
	                 image.backing.extent.height,
	                 static_cast<uint32_t>(groups),
	                 decodable | (keep_keys ? KeepKeysBit : 0u),
	                 texel_bytes};
	ImageViewInfo view_info {};
	view_info.format      = alias;
	view_info.type        = vk::ImageViewType::e2D;
	view_info.base_layer  = layer;
	view_info.layer_count = 1;
	view_info.usage       = vk::ImageUsageFlagBits::eStorage;
	const auto view = image.FindView(view_info);
	image.NoteContentWrite();

	m_scheduler.EndRendering();
	const auto command = m_scheduler.Current().Handle();
	RecordPalette(command, view_format, texel_bytes, values);
	image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	              ImageSubresourceRange {0, 1, layer, 1}, command);
	std::array<vk::BufferMemoryBarrier2, 2> before {};
	before[0].srcStageMask = vk::PipelineStageFlagBits2::eAllCommands |
	                         vk::PipelineStageFlagBits2::eHost;
	before[0].srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
	                          vk::AccessFlagBits2::eHostWrite;
	before[0].dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
	before[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before[0].buffer = metadata;
	before[0].offset = metadata_offset;
	before[0].size = metadata_size;
	before[1] = before[0];
	// One scratch allocation is reused in queue order, including across command buffers.
	before[1].srcStageMask = vk::PipelineStageFlagBits2::eDrawIndirect |
	                         vk::PipelineStageFlagBits2::eComputeShader;
	before[1].srcAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                          vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	before[1].dstAccessMask = vk::AccessFlagBits2::eShaderWrite;
	before[1].buffer = m_scratch->Handle();
	before[1].offset = 0;
	before[1].size = ScratchSize;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(before.size());
	dependency.pBufferMemoryBarriers = before.data();
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo metadata_info {metadata, descriptor_offset, prefix + metadata_size};
	const vk::DescriptorBufferInfo scratch_info {m_scratch->Handle(), 0, ScratchSize};
	const vk::DescriptorBufferInfo palette_info {m_palette->Handle(), 0, PaletteSize};
	const vk::DescriptorImageInfo image_info {nullptr, view, vk::ImageLayout::eGeneral};
	std::array<vk::WriteDescriptorSet, 4> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType =
		    index == 3 ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eStorageBuffer;
	}
	writes[0].pBufferInfo = &metadata_info;
	writes[1].pBufferInfo = &scratch_info;
	writes[2].pBufferInfo = &palette_info;
	writes[3].pImageInfo = &image_info;
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_validate_pipeline);
	command.dispatch(1, 1, 1);

	// Publish the decision and order every metadata read before conditional metadata writes.
	vk::MemoryBarrier2 decision {};
	decision.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
	decision.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	decision.dstStageMask = vk::PipelineStageFlagBits2::eDrawIndirect |
	                        vk::PipelineStageFlagBits2::eComputeShader;
	decision.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead |
	                         vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	dependency = {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers = &decision;
	command.pipelineBarrier2(dependency);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_clear_pipeline);
	command.dispatchIndirect(m_scratch->Handle(), 0);

	// Image state remains GENERAL/shader-write so subsequent transitions also retain this writer.
	vk::MemoryBarrier2 complete {};
	complete.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
	complete.srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	complete.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	complete.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.pMemoryBarriers = &complete;
	command.pipelineBarrier2(dependency);
	if (decision_copy != nullptr) {
		// The decision's group count (dispatch_args.x) for the verification, read by the host.
		const vk::BufferCopy copy {0, decision_offset, sizeof(uint32_t)};
		command.copyBuffer(m_scratch->Handle(), decision_copy, 1, &copy);
		vk::MemoryBarrier2 host {};
		host.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		host.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		host.dstStageMask  = vk::PipelineStageFlagBits2::eHost | vk::PipelineStageFlagBits2::eAllCommands;
		host.dstAccessMask = vk::AccessFlagBits2::eHostRead | vk::AccessFlagBits2::eMemoryRead |
		                     vk::AccessFlagBits2::eMemoryWrite;
		dependency.pMemoryBarriers = &host;
		command.pipelineBarrier2(dependency);
	}
}

} // namespace Libs::Graphics
