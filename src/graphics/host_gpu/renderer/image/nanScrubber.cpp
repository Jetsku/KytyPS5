#include "graphics/host_gpu/renderer/image/nanScrubber.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/nan_scrub_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"

#include <algorithm>
#include <span>

namespace Libs::Graphics {

namespace {
constexpr uint32_t LocalSize = 256;
} // namespace

NanScrubber::NanScrubber(GraphicContext& graphics): m_graphics(graphics) {
	// Off by default and not needed for rendering: a failure leaves the scrubber unusable (Valid()
	// is false) instead of stopping the emulator.
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	auto result = graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_set_layout);
	if (result != vk::Result::eSuccess) {
		LOGF("NanScrubber: create descriptor set layout failed: %s\n", vk::to_string(result).c_str());
		m_set_layout = nullptr;
		return;
	}
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_set_layout;
	result = graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout);
	if (result != vk::Result::eSuccess) {
		LOGF("NanScrubber: create pipeline layout failed: %s\n", vk::to_string(result).c_str());
		m_pipeline_layout = nullptr;
		return;
	}
	const std::span<const uint32_t> code {NAN_SCRUB_SPV};
	vk::ShaderModuleCreateInfo      module_info {};
	module_info.codeSize    = code.size_bytes();
	module_info.pCode       = code.data();
	vk::ShaderModule module = nullptr;
	result                  = graphics.device.createShaderModule(&module_info, nullptr, &module);
	if (result != vk::Result::eSuccess) {
		LOGF("NanScrubber: create shader module failed: %s\n", vk::to_string(result).c_str());
		return;
	}
	for (uint32_t packing = 0; packing < m_pipelines.size(); packing++) {
		const vk::SpecializationMapEntry entry {0, 0, sizeof(uint32_t)};
		vk::SpecializationInfo           specialization {};
		specialization.mapEntryCount = 1;
		specialization.pMapEntries   = &entry;
		specialization.dataSize      = sizeof(uint32_t);
		specialization.pData         = &packing;
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage               = vk::ShaderStageFlagBits::eCompute;
		stage.module              = module;
		stage.pName               = "main";
		stage.pSpecializationInfo = &specialization;
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_pipeline_layout;
		result = graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr,
		                                                &m_pipelines[packing]);
		if (result != vk::Result::eSuccess) {
			LOGF("NanScrubber: create pipeline failed: %s\n", vk::to_string(result).c_str());
			m_pipelines[packing] = nullptr;
		}
	}
	graphics.device.destroyShaderModule(module, nullptr);
}

NanScrubber::~NanScrubber() {
	for (const auto pipeline: m_pipelines) {
		if (pipeline) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
	}
	if (m_pipeline_layout) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_set_layout) {
		m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
	}
}

void NanScrubber::Scrub(CommandBuffer& command, vk::Buffer buffer, uint64_t offset, uint64_t size,
                        Packing packing) const {
	const uint64_t words = size / sizeof(uint32_t);
	if (!Valid() || buffer == nullptr || words == 0 || words > UINT32_MAX) {
		return;
	}
	command.EndRendering();
	const auto               sink = command.HelperSink();
	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = offset;
	barrier.size                = words * sizeof(uint32_t);
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	sink.pipelineBarrier2(dependency);

	const auto                   pipeline = m_pipelines[static_cast<uint32_t>(packing)];
	const vk::DescriptorBufferInfo info {buffer, offset, words * sizeof(uint32_t)};
	vk::WriteDescriptorSet       write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	command.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	(void)command.PushDescriptors(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, 1, &write);
	const auto max_groups = m_graphics.GetPhysicalDeviceProperties().limits.maxComputeWorkGroupCount[0];
	const uint64_t groups = std::clamp<uint64_t>((words + LocalSize - 1u) / LocalSize, 1u,
	                                             std::min<uint64_t>(max_groups, 65535u));
	sink.dispatch(static_cast<uint32_t>(groups), 1, 1);

	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	sink.pipelineBarrier2(dependency);
}

} // namespace Libs::Graphics
