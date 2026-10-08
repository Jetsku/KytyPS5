#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "gpu_tiler_shaders/bda_write_pages_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <bit>
#include <cinttypes>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults    = 1024;
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);
// KYTY_BDA_WRITES: bda_write_pages_process.comp's MAX_PAGE_FAULTS (1 GiB of 16 KiB pages). The list
// is the shader's: a u32 count in the first entry, page addresses from the second; the copy of the
// dropped-write counter follows it.
constexpr size_t MaxBdaWritePages = 65536;
constexpr size_t BdaPagesAreaSize = MaxBdaWritePages * sizeof(uint64_t);
constexpr size_t BdaDownloadSize  = BdaPagesAreaSize + 16;

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     BdaWritesEnabled() ? BufferCache::BDA_WRITES_FAULT_BUFFER_SIZE
                                        : BufferCache::CACHING_NUMPAGES / 8),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
                        MaxPendingFaults * PageFaultAreaSize) {
	SetVulkanObjectNameF(m_graphics.device, m_fault_buffer.Handle(), "Fault Buffer");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(
	    m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_fault_process_desc_layout),
	    "create fault-buffer descriptor layout");

	const auto module = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_fault_process_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(
	    nullptr, 1, &pipeline_info, nullptr, &m_fault_process_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create fault-buffer pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	if (m_bda_pages_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_bda_pages_pipeline, nullptr);
	}
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

void FaultManager::ProcessFaultBuffer() {
	KYTY_GPU_OP_SITE("fault.process");
	if (const auto wait_tick = m_fault_areas[m_current_area]; wait_tick != 0) {
		m_scheduler.Wait(wait_tick);
		m_scheduler.PopPendingOperations();
	}

	const auto offset = m_current_area * PageFaultAreaSize;
	auto*      mapped = m_download_buffer.Mapped().data() + offset;
	std::memset(mapped, 0, PageFaultAreaSize);
	m_download_buffer.Flush(offset, PageFaultAreaSize);

	vk::BufferMemoryBarrier2 pre_barrier {};
	pre_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	pre_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	pre_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	pre_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
	pre_barrier.buffer        = m_fault_buffer.Handle();
	pre_barrier.offset        = 0;
	pre_barrier.size           = m_fault_buffer.Size();
	auto post_barrier         = pre_barrier;
	post_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	post_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	post_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	post_barrier.dstAccessMask = vk::AccessFlagBits2::eShaderWrite;

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), 0, m_fault_buffer.Size()},
	    {m_download_buffer.Handle(), offset, PageFaultAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32;
	const auto num_workgroups = (num_threads + 63) / 64;
	command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	dependency.pBufferMemoryBarriers = &post_barrier;
	command.pipelineBarrier2(dependency);

	const auto area = m_current_area;
	m_scheduler.DeferOperation([this, mapped, offset, area] {
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		const auto  count  = static_cast<uint32_t>(faults[0]);
		for (uint32_t index = 1; index <= count; ++index) {
			fault_ranges.Add(faults[index], BufferCache::CACHING_PAGESIZE);
			LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", faults[index]);
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
		});
		m_fault_areas[area] = 0;
	});

	m_fault_areas[m_current_area++] = m_scheduler.CurrentTick();
	m_current_area %= MaxPendingFaults;
}

void FaultManager::PrepareBdaWrites() {
	if (m_bda_region_cleared) {
		return;
	}
	// Device-local memory starts undefined: clear the written-page bitmap and the counter once,
	// before the first dispatch that sets them (each collection clears what it read).
	constexpr auto offset = BufferCache::BDA_WRITE_BITMAP_WORD * sizeof(uint32_t);
	m_fault_buffer.Fill(offset, BufferCache::BDA_WRITES_FAULT_BUFFER_SIZE - offset, 0);
	m_bda_region_cleared = true;
}

FaultManager::BdaWrites FaultManager::CollectBdaWrites() {
	KYTY_GPU_OP_SITE("fault.bda-writes");
	EXIT_IF(!m_bda_region_cleared);
	if (m_bda_pages_pipeline == nullptr) {
		const auto module = CompileSPV(BDA_WRITE_PAGES_PROCESS_SPV, m_graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = m_fault_process_pipeline_layout;
		const auto result    = m_graphics.device.createComputePipelines(
            nullptr, 1, &pipeline_info, nullptr, &m_bda_pages_pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create BDA written-page pipeline");
		SetVulkanObjectNameF(m_graphics.device, m_bda_pages_pipeline, "BDA Written Page Parser");
		m_bda_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                          AllFlags, BdaDownloadSize);
	}
	auto* mapped = m_bda_download->Mapped().data();
	// Only the count (entry 0) is read before entries are: the shader writes entries 1..count and
	// the copy below overwrites the counter slot.
	std::memset(mapped, 0, sizeof(uint64_t));
	m_bda_download->Flush(0, sizeof(uint64_t));

	constexpr auto bitmap_offset  = BufferCache::BDA_WRITE_BITMAP_WORD * sizeof(uint32_t);
	constexpr auto bitmap_size    = BufferCache::FAULT_BITMAP_WORDS * sizeof(uint32_t);
	constexpr auto dropped_offset = BufferCache::BDA_DROPPED_WRITES_WORD * sizeof(uint32_t);

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	// The dispatch's shader atomics on the bitmap and the counter before the parse and the copy.
	vk::BufferMemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	before.dstStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                       vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	before.buffer = m_fault_buffer.Handle();
	before.offset = bitmap_offset;
	before.size   = BufferCache::BDA_WRITES_FAULT_BUFFER_SIZE - bitmap_offset;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo infos[] {
	    {m_fault_buffer.Handle(), bitmap_offset, bitmap_size},
	    {m_bda_download->Handle(), 0, BdaPagesAreaSize},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_bda_pages_pipeline);
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute,
	                                      m_fault_process_pipeline_layout, 0,
	                                      static_cast<uint32_t>(writes.size()), writes.data());
	command.dispatch(static_cast<uint32_t>((BufferCache::FAULT_BITMAP_WORDS + 63) / 64), 1, 1);
	const vk::BufferCopy counter_copy {dropped_offset, BdaPagesAreaSize, sizeof(uint32_t)};
	command.copyBuffer(m_fault_buffer.Handle(), m_bda_download->Handle(), 1, &counter_copy);
	vk::BufferMemoryBarrier2 cleared {};
	cleared.srcStageMask             = vk::PipelineStageFlagBits2::eTransfer;
	cleared.srcAccessMask            = vk::AccessFlagBits2::eTransferRead;
	cleared.dstStageMask             = vk::PipelineStageFlagBits2::eTransfer;
	cleared.dstAccessMask            = vk::AccessFlagBits2::eTransferWrite;
	cleared.buffer                   = m_fault_buffer.Handle();
	cleared.offset                   = dropped_offset;
	cleared.size                     = sizeof(uint32_t);
	dependency.pBufferMemoryBarriers = &cleared;
	command.pipelineBarrier2(dependency);
	command.fillBuffer(m_fault_buffer.Handle(), dropped_offset, sizeof(uint32_t), 0);
	// The next BDA writer sees the cleared bitmap and counter; the host sees the list.
	std::array<vk::BufferMemoryBarrier2, 2> after {};
	after[0].srcStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	after[0].srcAccessMask =
	    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	after[0].dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	after[0].buffer        = m_fault_buffer.Handle();
	after[0].offset        = bitmap_offset;
	after[0].size          = BufferCache::BDA_WRITES_FAULT_BUFFER_SIZE - bitmap_offset;
	after[1].srcStageMask =
	    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer;
	after[1].srcAccessMask =
	    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite;
	after[1].dstStageMask               = vk::PipelineStageFlagBits2::eHost;
	after[1].dstAccessMask              = vk::AccessFlagBits2::eHostRead;
	after[1].buffer                     = m_bda_download->Handle();
	after[1].offset                     = 0;
	after[1].size                       = BdaDownloadSize;
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
	dependency.pBufferMemoryBarriers    = after.data();
	command.pipelineBarrier2(dependency);

	m_scheduler.Wait(m_scheduler.CurrentTick());
	// Read back only what was written: the count, then its entries and the counter copy.
	m_bda_download->Invalidate(0, sizeof(uint64_t));
	BdaWrites result;
	uint32_t  count = 0;
	std::memcpy(&count, mapped, sizeof(count));
	result.overflow   = count > MaxBdaWritePages - 1;
	const auto stored = std::min<size_t>(count, MaxBdaWritePages - 1);
	if (stored != 0) {
		m_bda_download->Invalidate(sizeof(uint64_t), stored * sizeof(uint64_t));
	}
	m_bda_download->Invalidate(BdaPagesAreaSize, sizeof(uint32_t));
	result.pages.resize(stored);
	std::memcpy(result.pages.data(), mapped + sizeof(uint64_t), stored * sizeof(uint64_t));
	std::memcpy(&result.dropped, mapped + BdaPagesAreaSize, sizeof(uint32_t));
	return result;
}

} // namespace Libs::Graphics
