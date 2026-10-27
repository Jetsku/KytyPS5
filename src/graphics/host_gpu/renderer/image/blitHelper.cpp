#include "graphics/host_gpu/renderer/image/blitHelper.h"

#include "common/assert.h"
#include "gpu_blit_shaders/gpu_blit_color32_to_depth_spv.h"
#include "gpu_blit_shaders/gpu_blit_color_to_ms_depth_spv.h"
#include "gpu_blit_shaders/gpu_blit_depth_to_color32_spv.h"
#include "gpu_blit_shaders/gpu_blit_fs_triangle_spv.h"
#include "gpu_blit_shaders/gpu_blit_ms_depth_sample0_spv.h"
#include "gpu_blit_shaders/gpu_blit_ms_stencil_sample0_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

namespace {

[[nodiscard]] bool FormatHasStencil(vk::Format format) {
	return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
	       format == vk::Format::eD32SfloatS8Uint;
}

} // namespace

BlitHelper::BlitHelper(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	vk::DescriptorSetLayoutBinding texture_binding {};
	texture_binding.binding         = 0;
	texture_binding.descriptorType  = vk::DescriptorType::eSampledImage;
	texture_binding.descriptorCount = 1;
	texture_binding.stageFlags      = vk::ShaderStageFlagBits::eFragment;

	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = 1;
	descriptor_info.pBindings    = &texture_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create BlitHelper descriptor layout");

	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts    = &m_descriptor_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create BlitHelper pipeline layout");

	m_vertex_shader   = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, m_graphics.device);
	m_fragment_shader = CompileSPV(GPU_BLIT_COLOR_TO_MS_DEPTH_SPV, m_graphics.device);
	m_sample0_shader  = CompileSPV(GPU_BLIT_MS_DEPTH_SAMPLE0_SPV, m_graphics.device);
	m_stencil0_shader = CompileSPV(GPU_BLIT_MS_STENCIL_SAMPLE0_SPV, m_graphics.device);

	// Depth <-> 32-bit color reinterpretation passes.
	const auto& limits  = m_graphics.physical_device_properties.limits;
	const auto  r32uint = m_graphics.GetFormatProperties(vk::Format::eR32Uint);
	m_reinterpret_supported =
	    m_graphics.max_push_descriptors >= 2 &&
	    limits.maxPushConstantsSize >= sizeof(DepthToColorPush) &&
	    static_cast<bool>(r32uint.optimalTilingFeatures & vk::FormatFeatureFlagBits::eStorageImage);
	if (!m_reinterpret_supported) {
		return;
	}

	const std::array<vk::DescriptorSetLayoutBinding, 2> d2c_bindings {{
	    {0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	}};
	descriptor_info.bindingCount = static_cast<uint32_t>(d2c_bindings.size());
	descriptor_info.pBindings    = d2c_bindings.data();
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_d2c_descriptor_layout),
	                     "create depth-to-color descriptor layout");
	const vk::PushConstantRange d2c_push {vk::ShaderStageFlagBits::eCompute, 0,
	                                      sizeof(DepthToColorPush)};
	layout_info.pSetLayouts            = &m_d2c_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &d2c_push;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_d2c_pipeline_layout),
	    "create depth-to-color pipeline layout");
	{
		const auto module = CompileSPV(GPU_BLIT_DEPTH_TO_COLOR32_SPV, m_graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = module;
		create.stage.pName  = "main";
		create.layout       = m_d2c_pipeline_layout;
		const auto result   = m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr,
		                                                               &m_d2c_pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create depth-to-color pipeline");
	}

	const vk::DescriptorSetLayoutBinding c2d_binding {
	    0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eFragment, nullptr};
	descriptor_info.bindingCount = 1;
	descriptor_info.pBindings    = &c2d_binding;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_c2d_descriptor_layout),
	                     "create color-to-depth descriptor layout");
	const vk::PushConstantRange c2d_push {vk::ShaderStageFlagBits::eFragment, 0,
	                                      sizeof(ColorToDepthPush)};
	layout_info.pSetLayouts         = &m_c2d_descriptor_layout;
	layout_info.pPushConstantRanges = &c2d_push;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_c2d_pipeline_layout),
	    "create color-to-depth pipeline layout");
	m_c2d_fragment_shader = CompileSPV(GPU_BLIT_COLOR32_TO_DEPTH_SPV, m_graphics.device);
}

BlitHelper::~BlitHelper() {
	for (const auto& pipeline: m_pipelines) {
		m_graphics.device.destroyPipeline(pipeline.handle, nullptr);
	}
	if (m_d2c_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_d2c_pipeline, nullptr);
	}
	for (auto* layout: {&m_d2c_pipeline_layout, &m_c2d_pipeline_layout}) {
		if (*layout != nullptr) {
			m_graphics.device.destroyPipelineLayout(*layout, nullptr);
		}
	}
	for (auto* layout: {&m_d2c_descriptor_layout, &m_c2d_descriptor_layout}) {
		if (*layout != nullptr) {
			m_graphics.device.destroyDescriptorSetLayout(*layout, nullptr);
		}
	}
	if (m_c2d_fragment_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_c2d_fragment_shader, nullptr);
	}
	if (m_fragment_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_fragment_shader, nullptr);
	}
	for (auto* module: {&m_sample0_shader, &m_stencil0_shader}) {
		if (*module != nullptr) {
			m_graphics.device.destroyShaderModule(*module, nullptr);
		}
	}
	if (m_vertex_shader != nullptr) {
		m_graphics.device.destroyShaderModule(m_vertex_shader, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

vk::Pipeline BlitHelper::GetPipeline(PipelineKey key) {
	const auto cached = std::ranges::find(m_pipelines, key, &Pipeline::key);
	if (cached != m_pipelines.end()) {
		return cached->handle;
	}

	const auto samples = vulkan_sample_count(key.samples);
	EXIT_IF(samples == vk::SampleCountFlagBits {} || key.format == vk::Format::eUndefined);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_vertex_shader;
	stages[0].pName  = "main";
	const bool color32 = key.kind == PipelineKind::Color32ToDepth;
	// MsStencilSample0 writes a colour attachment (R8_UINT), the others a depth attachment.
	const bool color_output = key.kind == PipelineKind::MsStencilSample0;
	EXIT_IF(color32 && m_c2d_fragment_shader == nullptr);
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = color32                                      ? m_c2d_fragment_shader
	                   : key.kind == PipelineKind::MsDepthSample0   ? m_sample0_shader
	                   : key.kind == PipelineKind::MsStencilSample0 ? m_stencil0_shader
	                                                                : m_fragment_shader;
	stages[1].pName  = "main";

	vk::PipelineVertexInputStateCreateInfo vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = samples;
	vk::PipelineDepthStencilStateCreateInfo depth {};
	depth.depthTestEnable  = color_output ? VK_FALSE : VK_TRUE;
	depth.depthWriteEnable = color_output ? VK_FALSE : VK_TRUE;
	depth.depthCompareOp   = vk::CompareOp::eAlways;
	vk::PipelineColorBlendAttachmentState color_attachment {};
	color_attachment.colorWriteMask = vk::ColorComponentFlagBits::eR;
	vk::PipelineColorBlendStateCreateInfo color_blend {};
	if (color_output) {
		color_blend.attachmentCount = 1;
		color_blend.pAttachments    = &color_attachment;
	}
	const std::array dynamic_states {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic.pDynamicStates    = dynamic_states.data();

	vk::PipelineRenderingCreateInfo rendering {};
	if (color_output) {
		rendering.colorAttachmentCount    = 1;
		rendering.pColorAttachmentFormats = &key.format;
	} else {
		rendering.depthAttachmentFormat = key.format;
	}
	if (color32 && key.format == vk::Format::eD32SfloatS8Uint) {
		// CopyColor32ToDepth binds the combined view as the stencil attachment too.
		rendering.stencilAttachmentFormat = key.format;
	}

	vk::GraphicsPipelineCreateInfo create {};
	create.pNext               = &rendering;
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth;
	create.pColorBlendState    = &color_blend;
	create.pDynamicState       = &dynamic;
	create.layout              = color32 ? m_c2d_pipeline_layout : m_pipeline_layout;

	vk::Pipeline pipeline = nullptr;
	RequireVulkanSuccess(
	    m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline),
	    "create color-to-MS-depth pipeline");
	m_pipelines.push_back({key, pipeline});
	return pipeline;
}

void BlitHelper::ReinterpretColorAsMsDepth(Image& source, Image& destination) {
	KYTY_GPU_OP_SITE("blit.color_to_ms_depth");
	const auto& source_info      = source.info;
	const auto& destination_info = destination.info;
	EXIT_IF(DepthAspectTransferFormat(source_info.pixel_format) != vk::Format::eUndefined ||
	        DepthAspectTransferFormat(destination_info.pixel_format) == vk::Format::eUndefined ||
	        source_info.samples != 1 || destination_info.samples <= 1 ||
	        destination_info.samples > 4 || source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        source_info.extent.width != destination_info.extent.width ||
	        source_info.extent.height != destination_info.extent.height ||
	        source_info.extent.depth != 1 || destination_info.extent.depth != 1 ||
	        source.backing.image == nullptr || destination.backing.image == nullptr);
	destination.NoteContentWrite();
	m_scheduler.EndRendering();

	ImageViewInfo source_view_info {};
	source_view_info.format = source_info.pixel_format;
	source_view_info.type   = vk::ImageViewType::e2D;
	source_view_info.aspect = vk::ImageAspectFlagBits::eColor;
	source_view_info.usage  = vk::ImageUsageFlagBits::eSampled;
	const auto source_view  = source.FindView(source_view_info);

	ImageViewInfo destination_view_info {};
	destination_view_info.format = destination_info.pixel_format;
	destination_view_info.type   = vk::ImageViewType::e2D;
	destination_view_info.aspect = vk::ImageAspectFlagBits::eDepth;
	destination_view_info.usage  = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	const auto destination_view  = destination.FindView(destination_view_info);

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command);
	destination.Transit(ColorToMsDepthLayout, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, {},
	                    command);

	vk::RenderingAttachmentInfo depth_attachment {};
	depth_attachment.imageView               = destination_view;
	depth_attachment.imageLayout             = ColorToMsDepthLayout;
	depth_attachment.loadOp                  = vk::AttachmentLoadOp::eClear;
	depth_attachment.storeOp                 = vk::AttachmentStoreOp::eStore;
	depth_attachment.clearValue.depthStencil = {0.0f, 0};

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent = {destination_info.extent.width, destination_info.extent.height};
	rendering.layerCount        = 1;
	rendering.pDepthAttachment  = &depth_attachment;
	command.beginRendering(&rendering);

	vk::DescriptorImageInfo descriptor_image {};
	descriptor_image.imageView   = source_view;
	descriptor_image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	vk::WriteDescriptorSet descriptor_write {};
	descriptor_write.dstBinding      = 0;
	descriptor_write.descriptorCount = 1;
	descriptor_write.descriptorType  = vk::DescriptorType::eSampledImage;
	descriptor_write.pImageInfo      = &descriptor_image;
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1,
	                             &descriptor_write);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eGraphics,
	                     GetPipeline({destination_info.samples, destination_info.pixel_format}));

	const vk::Viewport viewport {0.0f,
	                             0.0f,
	                             static_cast<float>(destination_info.extent.width),
	                             static_cast<float>(destination_info.extent.height),
	                             0.0f,
	                             1.0f};
	const vk::Rect2D   scissor {{0, 0},
	                            {destination_info.extent.width, destination_info.extent.height}};
	command.setViewport(0, 1, &viewport);
	command.setScissor(0, 1, &scissor);
	command.draw(3, 1, 0, 0);
	command.endRendering();
	// The helper pipeline has static state that invalidates the guest draw's dynamic state.
	m_scheduler.GetDynamicState().Invalidate();
}

void BlitHelper::CopyMsDepthSample0(Image& source, Image& destination) {
	KYTY_GPU_OP_SITE("blit.ms_depth_sample0");
	CopyMsSample0(source, destination, false);
}

void BlitHelper::CopyMsStencilSample0(Image& source, Image& destination) {
	KYTY_GPU_OP_SITE("blit.ms_stencil_sample0");
	CopyMsSample0(source, destination, true);
}

void BlitHelper::CopyMsSample0(Image& source, Image& destination, bool stencil) {
	const auto& source_info      = source.info;
	const auto& destination_info = destination.info;
	EXIT_IF(DepthAspectTransferFormat(source.backing.format) == vk::Format::eUndefined ||
	        (stencil ? (!FormatHasStencil(source.backing.format) ||
	                    destination.backing.format != vk::Format::eR8Uint)
	                 : DepthAspectTransferFormat(destination.backing.format) ==
	                       vk::Format::eUndefined) ||
	        source.backing.samples <= 1 || destination.backing.samples != 1 ||
	        source.backing.image_type != vk::ImageType::e2D ||
	        destination.backing.image_type != vk::ImageType::e2D ||
	        source_info.extent.width != destination_info.extent.width ||
	        source_info.extent.height != destination_info.extent.height ||
	        source.backing.layers == 0 || source.backing.layers != destination.backing.layers ||
	        source.backing.image == nullptr || destination.backing.image == nullptr);
	destination.NoteContentWrite();
	m_scheduler.EndRendering();

	auto& command_buffer = m_scheduler.Current();
	auto  command        = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eDepthStencilReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead,
	               {}, command);
	const auto destination_layout = stencil ? vk::ImageLayout::eColorAttachmentOptimal
	                                        : vk::ImageLayout::eDepthStencilAttachmentOptimal;
	destination.Transit(destination_layout,
	                    stencil ? vk::AccessFlagBits2::eColorAttachmentWrite
	                            : vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
	                    {}, command);
	const auto pipeline =
	    GetPipeline({1, destination.backing.format,
	                 stencil ? PipelineKind::MsStencilSample0 : PipelineKind::MsDepthSample0});
	const uint32_t width  = destination_info.extent.width;
	const uint32_t height = destination_info.extent.height;

	// Layer by layer (an array's each), through 2D views.
	for (uint32_t layer = 0; layer < destination.backing.layers; layer++) {
		// FindView turns these view formats into the depth/stencil image's own format with the
		// depth or the stencil aspect.
		ImageViewInfo source_view_info {};
		source_view_info.format     = stencil ? vk::Format::eS8Uint : vk::Format::eD32Sfloat;
		source_view_info.type       = vk::ImageViewType::e2D;
		source_view_info.aspect     = stencil ? vk::ImageAspectFlagBits::eStencil
		                                      : vk::ImageAspectFlagBits::eDepth;
		source_view_info.base_layer = layer;
		source_view_info.usage      = vk::ImageUsageFlagBits::eSampled;
		const auto source_view      = source.FindView(source_view_info);

		ImageViewInfo destination_view_info {};
		destination_view_info.format     = destination.backing.format;
		destination_view_info.type       = vk::ImageViewType::e2D;
		destination_view_info.aspect     = stencil ? vk::ImageAspectFlagBits::eColor
		                                           : vk::ImageAspectFlagBits::eDepth;
		destination_view_info.base_layer = layer;
		destination_view_info.usage      = stencil ? vk::ImageUsageFlagBits::eColorAttachment
		                                           : vk::ImageUsageFlagBits::eDepthStencilAttachment;
		const auto destination_view      = destination.FindView(destination_view_info);

		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = destination_view;
		attachment.imageLayout = destination_layout;
		// The fullscreen triangle writes every texel of the render area.
		attachment.loadOp  = vk::AttachmentLoadOp::eDontCare;
		attachment.storeOp = vk::AttachmentStoreOp::eStore;

		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = vk::Extent2D {width, height};
		rendering.layerCount        = 1;
		if (stencil) {
			rendering.colorAttachmentCount = 1;
			rendering.pColorAttachments    = &attachment;
		} else {
			rendering.pDepthAttachment = &attachment;
		}
		command.beginRendering(&rendering);

		vk::DescriptorImageInfo descriptor_image {};
		descriptor_image.imageView   = source_view;
		descriptor_image.imageLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
		vk::WriteDescriptorSet descriptor_write {};
		descriptor_write.dstBinding      = 0;
		descriptor_write.descriptorCount = 1;
		descriptor_write.descriptorType  = vk::DescriptorType::eSampledImage;
		descriptor_write.pImageInfo      = &descriptor_image;
		command_buffer.PushDescriptors(vk::PipelineBindPoint::eGraphics, m_pipeline_layout, 0, 1,
		                               &descriptor_write);
		command_buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(width),
		                             static_cast<float>(height), 0.0f, 1.0f};
		const vk::Rect2D   scissor {{0, 0}, {width, height}};
		command.setViewport(0, 1, &viewport);
		command.setScissor(0, 1, &scissor);
		command.draw(3, 1, 0, 0);
		command.endRendering();
	}
	// The helper pipeline has static state that invalidates the guest draw's dynamic state.
	m_scheduler.GetDynamicState().Invalidate();
}

namespace {

[[nodiscard]] bool SameBaseExtent2D(const Image& a, const Image& b) {
	return a.backing.extent.width == b.backing.extent.width &&
	       a.backing.extent.height == b.backing.extent.height;
}

[[nodiscard]] bool SingleSample2D(const Image& image) {
	return image.backing.image != nullptr && image.backing.image_type == vk::ImageType::e2D &&
	       image.backing.samples == 1 && image.backing.extent.depth == 1 &&
	       image.backing.mip_levels != 0 && image.backing.layers != 0;
}

} // namespace

bool BlitHelper::SupportsColor32Side(const Image& color) const {
	const auto format = color.backing.format;
	return SingleSample2D(color) && DepthAspectTransferFormat(format) == vk::Format::eUndefined &&
	       !color.info.IsBlock() && vk::blockSize(format) == 4 &&
	       ImageViewOps::FormatsCompatible(format, vk::Format::eR32Uint) &&
	       static_cast<bool>(color.backing.usage & vk::ImageUsageFlagBits::eStorage) &&
	       (format == vk::Format::eR32Uint ||
	        static_cast<bool>(color.backing.flags & vk::ImageCreateFlagBits::eMutableFormat));
}

bool BlitHelper::SupportsDepthToColor32(const Image& source, const Image& destination) const {
	return m_reinterpret_supported && SingleSample2D(source) &&
	       DepthAspectTransferFormat(source.backing.format) == vk::Format::eD32Sfloat &&
	       static_cast<bool>(source.backing.usage & vk::ImageUsageFlagBits::eSampled) &&
	       SupportsColor32Side(destination) && SameBaseExtent2D(source, destination);
}

bool BlitHelper::SupportsColor32ToDepth(const Image& source, const Image& destination) const {
	// D32_SFLOAT, or D32_SFLOAT_S8_UINT through one view of both aspects bound as the depth and
	// the stencil attachment, whose stencil is loaded and stored unchanged (no stencil test).
	return m_reinterpret_supported && SingleSample2D(destination) &&
	       (destination.backing.format == vk::Format::eD32Sfloat ||
	        destination.backing.format == vk::Format::eD32SfloatS8Uint) &&
	       static_cast<bool>(destination.backing.usage &
	                         vk::ImageUsageFlagBits::eDepthStencilAttachment) &&
	       SupportsColor32Side(source) && SameBaseExtent2D(source, destination);
}

void BlitHelper::CopyDepthToColor32(Image& source, Image& destination) {
	KYTY_GPU_OP_SITE("image.reinterpret_depth_to_color");
	EXIT_IF(!SupportsDepthToColor32(source, destination));
	const uint32_t levels = std::min(source.backing.mip_levels, destination.backing.mip_levels);
	const uint32_t layers = std::min(source.backing.layers, destination.backing.layers);
	destination.NoteContentWrite();
	m_scheduler.EndRendering();

	ImageViewInfo source_view_info {};
	source_view_info.format      = source.backing.format;
	source_view_info.type        = vk::ImageViewType::e2DArray;
	source_view_info.aspect      = vk::ImageAspectFlagBits::eDepth;
	source_view_info.level_count = levels;
	source_view_info.layer_count = layers;
	source_view_info.usage       = vk::ImageUsageFlagBits::eSampled;
	const auto source_view       = source.FindView(source_view_info);

	auto&      command_buffer = m_scheduler.Current();
	const auto command        = command_buffer.HelperSink();
	source.Transit(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits2::eShaderRead, {},
	               command.Identity());
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite, {},
	                    command.Identity());
	command_buffer.BindPipeline(vk::PipelineBindPoint::eCompute, m_d2c_pipeline);
	for (uint32_t level = 0; level < levels; level++) {
		ImageViewInfo destination_view_info {};
		destination_view_info.format      = vk::Format::eR32Uint;
		destination_view_info.type        = vk::ImageViewType::e2DArray;
		destination_view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		destination_view_info.base_level  = level;
		destination_view_info.level_count = 1;
		destination_view_info.layer_count = layers;
		destination_view_info.usage       = vk::ImageUsageFlagBits::eStorage;
		const auto destination_view       = destination.FindView(destination_view_info);

		const vk::DescriptorImageInfo source_image {nullptr, source_view,
		                                            vk::ImageLayout::eShaderReadOnlyOptimal};
		const vk::DescriptorImageInfo destination_image {nullptr, destination_view,
		                                                 vk::ImageLayout::eGeneral};
		std::array<vk::WriteDescriptorSet, 2> writes {};
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[0].pImageInfo      = &source_image;
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType  = vk::DescriptorType::eStorageImage;
		writes[1].pImageInfo      = &destination_image;
		command_buffer.PushDescriptors(vk::PipelineBindPoint::eCompute, m_d2c_pipeline_layout, 0,
		                               static_cast<uint32_t>(writes.size()), writes.data());
		const DepthToColorPush push {std::max(source.backing.extent.width >> level, 1u),
		                             std::max(source.backing.extent.height >> level, 1u), level};
		command.pushConstants(m_d2c_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
		                      sizeof(push), &push);
		command.dispatch((push.width + 7u) / 8u, (push.height + 7u) / 8u, layers);
	}
}

void BlitHelper::CopyColor32ToDepth(Image& source, Image& destination) {
	KYTY_GPU_OP_SITE("image.reinterpret_color_to_depth");
	EXIT_IF(!SupportsColor32ToDepth(source, destination));
	const uint32_t levels = std::min(source.backing.mip_levels, destination.backing.mip_levels);
	const uint32_t layers = std::min(source.backing.layers, destination.backing.layers);
	destination.NoteContentWrite();
	m_scheduler.EndRendering();

	// A combined format keeps its stencil: the same view is the stencil attachment, loaded and
	// stored unchanged (stencil test off), so the attachment is also read.
	const bool with_stencil = destination.backing.format == vk::Format::eD32SfloatS8Uint;
	auto&      command_buffer = m_scheduler.Current();
	auto       command        = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {}, command);
	destination.Transit(ColorToMsDepthLayout,
	                    with_stencil ? vk::AccessFlagBits2::eDepthStencilAttachmentRead |
	                                       vk::AccessFlagBits2::eDepthStencilAttachmentWrite
	                                 : vk::AccessFlags2 {
	                                       vk::AccessFlagBits2::eDepthStencilAttachmentWrite},
	                    {}, command);
	const auto pipeline =
	    GetPipeline({1, destination.backing.format, PipelineKind::Color32ToDepth});
	for (uint32_t level = 0; level < levels; level++) {
		const uint32_t width  = std::max(destination.backing.extent.width >> level, 1u);
		const uint32_t height = std::max(destination.backing.extent.height >> level, 1u);

		ImageViewInfo source_view_info {};
		source_view_info.format      = vk::Format::eR32Uint;
		source_view_info.type        = vk::ImageViewType::e2DArray;
		source_view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		source_view_info.base_level  = level;
		source_view_info.level_count = 1;
		source_view_info.layer_count = layers;
		source_view_info.usage       = vk::ImageUsageFlagBits::eStorage;
		const auto source_view       = source.FindView(source_view_info);
		const vk::DescriptorImageInfo source_image {nullptr, source_view,
		                                            vk::ImageLayout::eGeneral};

		for (uint32_t layer = 0; layer < layers; layer++) {
			ImageViewInfo destination_view_info {};
			destination_view_info.format      = destination.backing.format;
			destination_view_info.type        = vk::ImageViewType::e2D;
			destination_view_info.aspect      = vk::ImageAspectFlagBits::eDepth;
			if (with_stencil) {
				destination_view_info.aspect |= vk::ImageAspectFlagBits::eStencil;
			}
			destination_view_info.base_level  = level;
			destination_view_info.level_count = 1;
			destination_view_info.base_layer  = layer;
			destination_view_info.layer_count = 1;
			destination_view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
			const auto destination_view       = destination.FindView(destination_view_info);

			vk::RenderingAttachmentInfo depth_attachment {};
			depth_attachment.imageView   = destination_view;
			depth_attachment.imageLayout = ColorToMsDepthLayout;
			// The fullscreen triangle writes every texel of the render area.
			depth_attachment.loadOp  = vk::AttachmentLoadOp::eDontCare;
			depth_attachment.storeOp = vk::AttachmentStoreOp::eStore;
			vk::RenderingAttachmentInfo stencil_attachment {};
			stencil_attachment.imageView   = destination_view;
			stencil_attachment.imageLayout = ColorToMsDepthLayout;
			stencil_attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
			stencil_attachment.storeOp     = vk::AttachmentStoreOp::eStore;

			vk::RenderingInfo rendering {};
			rendering.renderArea.extent  = vk::Extent2D {width, height};
			rendering.layerCount         = 1;
			rendering.pDepthAttachment   = &depth_attachment;
			rendering.pStencilAttachment = with_stencil ? &stencil_attachment : nullptr;
			command.beginRendering(&rendering);

			vk::WriteDescriptorSet write {};
			write.dstBinding      = 0;
			write.descriptorCount = 1;
			write.descriptorType  = vk::DescriptorType::eStorageImage;
			write.pImageInfo      = &source_image;
			command_buffer.PushDescriptors(vk::PipelineBindPoint::eGraphics,
			                               m_c2d_pipeline_layout, 0, 1, &write);
			command_buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			const ColorToDepthPush push {layer};
			command.pushConstants(m_c2d_pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0,
			                      sizeof(push), &push);
			const vk::Viewport viewport {0.0f, 0.0f, static_cast<float>(width),
			                             static_cast<float>(height), 0.0f, 1.0f};
			const vk::Rect2D   scissor {{0, 0}, {width, height}};
			command.setViewport(0, 1, &viewport);
			command.setScissor(0, 1, &scissor);
			command.draw(3, 1, 0, 0);
			command.endRendering();
		}
	}
}

} // namespace Libs::Graphics
