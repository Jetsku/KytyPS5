#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "gpu_tiler_shaders/fault_buffer_process_spv.h"
#include "gpu_tiler_shaders/bda_write_pages_process_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "common/liveSwitch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <bit>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr size_t MaxPageFaults    = 1024;
constexpr size_t PageFaultAreaSize = MaxPageFaults * sizeof(uint64_t);
// KYTY_BDA_WRITES: bda_write_pages_process.comp's MAX_PAGE_FAULTS (1 GiB of 16 KiB pages). The list
// is the shader's: a u32 count in the first entry, page addresses from the second; the copy of the
// dropped-write counter follows it.
constexpr size_t MaxBdaWritePages = 65536;
constexpr size_t BdaPagesAreaSize = MaxBdaWritePages * sizeof(uint64_t);
// KYTY_BDA_STORE_LOG: the copy of the store log (its count, then its entries) follows the counter.
constexpr size_t BdaStoreLogOffset = BdaPagesAreaSize + 16;
constexpr size_t BdaDownloadSize   = BdaStoreLogOffset + BufferCache::BDA_STORE_LOG_PREFIX_BYTES;

bool StoreLogEnabled() {
	return ShaderRecompiler::GetCodegenOptions().bda_store_log;
}

// KYTY_FAULT_SCAN_WORDS=16 (default 4 = upstream fd04c3fa9; 16 from chenxiao07/KytyPS5 b7aa2b877):
// bitmap words each invocation of fault_buffer_process.comp reads (its WORDS_PER_INVOCATION
// specialization constant). 4: one uvec4 per invocation (upstream's scan). 16: a quarter of those
// invocations, each reading four uvec4s, skipping zero blocks and storing only into nonzero words; the
// reported pages are the same. Any other value: 4.
uint32_t FaultScanWords() {
	static const uint32_t words = [] {
		const auto* value = std::getenv("KYTY_FAULT_SCAN_WORDS");
		const auto  n     = value != nullptr ? std::strtoul(value, nullptr, 10) : 4ul;
		return n == 16 ? 16u : 4u;
	}();
	return words;
}

// KYTY_FAULT_SCAN_MAPPED=1 (live, default 0): fault_buffer_process.comp scans only the bitmap words of
// pages inside guest GPU mappings (RenderContext's mapped ranges), one vkCmdDispatchBase per run of
// 64-invocation groups (8192 pages, 128 MiB each) that holds mapped memory, instead of the whole
// 1 TiB bitmap. Faults are taken only inside mappings (FindBuffer of the page); bits of unmapped
// pages stay set and are scanned once the range is mapped. Only with KYTY_FAULT_SCAN_WORDS=4.
Live::Switch g_fault_scan_mapped("KYTY_FAULT_SCAN_MAPPED", Live::ParseDefaultOff);

} // namespace

FaultManager::FaultManager(GraphicContext& graphics, CommandScheduler& scheduler,
                           BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache),
      m_fault_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                     ShaderRecompiler::GetCodegenOptions().bda_tags_gpu_count
                         ? BufferCache::BDA_TAG_COUNTER_WORD * sizeof(uint32_t) + 16 // KYTY_BDA_TAGS_GPU_COUNT
                     : BdaWritesEnabled() ? BufferCache::BDA_WRITES_FAULT_BUFFER_SIZE
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

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &m_fault_process_desc_layout;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_fault_process_pipeline_layout),
	    "create fault-buffer pipeline layout");

	// KYTY_FAULT_SCAN_WORDS: WORDS_PER_INVOCATION (constant 0).
	const uint32_t                   scan_words = FaultScanWords();
	const vk::SpecializationMapEntry scan_entry {0, 0, sizeof(uint32_t)};
	const vk::SpecializationInfo     scan_spec {1, &scan_entry, sizeof(uint32_t), &scan_words};
	m_fault_process_pipeline =
	    CreateComputePipeline(m_graphics.device, FAULT_BUFFER_PROCESS_SPV,
	                          m_fault_process_pipeline_layout, "create fault-buffer pipeline",
	                          &scan_spec);
	SetVulkanObjectNameF(m_graphics.device, m_fault_process_pipeline, "Fault Buffer Parser");
}

FaultManager::~FaultManager() {
	if (m_bda_pages_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_bda_pages_pipeline, nullptr);
	}
	if (m_fault_process_pipeline_base != nullptr) {
		m_graphics.device.destroyPipeline(m_fault_process_pipeline_base, nullptr);
	}
	m_graphics.device.destroyPipeline(m_fault_process_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_fault_process_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_fault_process_desc_layout, nullptr);
}

bool FaultManager::ScanMappedActive() noexcept {
	return g_fault_scan_mapped.Get() != 0;
}

void FaultManager::ProcessFaultBuffer(const std::vector<std::pair<uint64_t, uint64_t>>& mappings) {
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
	const auto command = m_scheduler.Current().HelperSink();
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &pre_barrier;
	command.pipelineBarrier2(dependency);
	const auto num_threads    = BufferCache::CACHING_NUMPAGES / 32 / FaultScanWords(); // Four words per invocation by default (fd04c3fa9).
	const auto num_workgroups = (num_threads + 63) / 64;
	// KYTY_FAULT_SCAN_MAPPED: runs of groups over the mapped pages (one group: 64 x 128 pages).
	std::vector<std::pair<uint32_t, uint32_t>> runs;
	if (g_fault_scan_mapped.Get() != 0 && FaultScanWords() == 4 && !mappings.empty()) {
		constexpr uint64_t GroupPages = 64u * 128u;
		for (const auto& [begin, end]: mappings) {
			if (end <= begin) continue;
			const uint64_t first = (begin >> BufferCache::CACHING_PAGEBITS) / GroupPages;
			const uint64_t last  = ((end - 1) >> BufferCache::CACHING_PAGEBITS) / GroupPages + 1;
			const auto     g0    = static_cast<uint32_t>(std::min<uint64_t>(first, num_workgroups));
			const auto     g1    = static_cast<uint32_t>(std::min<uint64_t>(last, num_workgroups));
			if (g1 <= g0) continue;
			if (!runs.empty() && g0 <= runs.back().second) {
				runs.back().second = std::max(runs.back().second, g1);
			} else {
				runs.emplace_back(g0, g1);
			}
		}
		std::sort(runs.begin(), runs.end());
		if (m_fault_process_pipeline_base == nullptr && !runs.empty()) {
			const uint32_t                   scan_words = FaultScanWords();
			const vk::SpecializationMapEntry scan_entry {0, 0, sizeof(uint32_t)};
			const vk::SpecializationInfo     scan_spec {1, &scan_entry, sizeof(uint32_t), &scan_words};
			const auto                       shader = CompileSPV(FAULT_BUFFER_PROCESS_SPV, m_graphics.device);
			vk::ComputePipelineCreateInfo    create {};
			create.flags                     = vk::PipelineCreateFlagBits::eDispatchBase;
			create.stage.stage               = vk::ShaderStageFlagBits::eCompute;
			create.stage.module              = shader;
			create.stage.pName               = "main";
			create.stage.pSpecializationInfo = &scan_spec;
			create.layout                    = m_fault_process_pipeline_layout;
			const auto result = m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr,
			                                                             &m_fault_process_pipeline_base);
			m_graphics.device.destroyShaderModule(shader, nullptr);
			RequireVulkanSuccess(result, "create fault-buffer pipeline (dispatch base)");
		}
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			uint64_t groups = 0;
			for (const auto& run: runs) groups += run.second - run.first;
			std::printf("KYTY_FAULT_SCAN_MAPPED: %zu mapped ranges -> %zu runs, %" PRIu64 " of %" PRIu64
			            " groups scanned\n",
			            mappings.size(), runs.size(), groups, static_cast<uint64_t>(num_workgroups));
		}
	}
	if (!runs.empty()) {
		m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline_base);
	} else {
		m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_fault_process_pipeline);
	}
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute,
	                             m_fault_process_pipeline_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	if (!runs.empty()) {
		for (const auto& [g0, g1]: runs) {
			command.dispatchBase(g0, 0, 0, g1 - g0, 1, 1);
		}
	} else {
		command.dispatch(static_cast<uint32_t>(num_workgroups), 1, 1);
	}
	// The host reads the parser's page list after the tick (the deferred operation below): only a
	// device-to-host dependency makes the shader's writes visible to host reads (Vulkan spec, host
	// access); waiting for the tick alone does not. Based on Senaxx/KytyPS5 1e7597c01.
	vk::BufferMemoryBarrier2 host_barrier {};
	host_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	host_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	host_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	host_barrier.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	host_barrier.buffer        = m_download_buffer.Handle();
	host_barrier.offset        = offset;
	host_barrier.size          = PageFaultAreaSize;
	const std::array post_barriers {post_barrier, host_barrier};
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(post_barriers.size());
	dependency.pBufferMemoryBarriers    = post_barriers.data();
	command.pipelineBarrier2(dependency);
	dependency.bufferMemoryBarrierCount = 1;
	// KYTY_BDA_TAGS_GPU_COUNT (diagnostic): a copy of the get_bda_pointer counters per area.
	const bool gpu_count = ShaderRecompiler::GetCodegenOptions().bda_tags_gpu_count;
	if (gpu_count) {
		if (m_tag_count_download == nullptr) {
			m_tag_count_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
			                                                AllFlags, MaxPendingFaults * 16);
		}
		vk::BufferMemoryBarrier2 count_barrier {};
		count_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		count_barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
		count_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		count_barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
		count_barrier.buffer        = m_fault_buffer.Handle();
		count_barrier.offset        = BufferCache::BDA_TAG_COUNTER_WORD * sizeof(uint32_t);
		count_barrier.size          = 16;
		dependency.pBufferMemoryBarriers = &count_barrier;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {BufferCache::BDA_TAG_COUNTER_WORD * sizeof(uint32_t), m_current_area * 16u, 16};
		command.copyBuffer(m_fault_buffer.Handle(), m_tag_count_download->Handle(), 1, &copy);
	}

	const auto area = m_current_area;
	BufferCache::s_tag_hops.scans.fetch_add(1, std::memory_order_relaxed);
	m_scheduler.DeferOperation([this, mapped, offset, area] {
		m_download_buffer.Invalidate(offset, PageFaultAreaSize);
		BufferCache::s_tag_hops.scans_read.fetch_add(1, std::memory_order_relaxed);
		if (m_tag_count_download != nullptr) {
			m_tag_count_download->Invalidate(area * 16u, 16);
			uint32_t now[3] {};
			std::memcpy(now, m_tag_count_download->Mapped().data() + area * 16u, sizeof(now));
			auto& hops = BufferCache::s_tag_hops;
			std::atomic<uint64_t>* totals[3] {&hops.gpu_calls, &hops.gpu_zero, &hops.gpu_untagged};
			for (int k = 0; k < 3; ++k) {
				totals[k]->fetch_add(static_cast<uint32_t>(now[k] - m_tag_count_last[k]), std::memory_order_relaxed);
				m_tag_count_last[k] = now[k];
			}
		}
		RangeSet    fault_ranges;
		const auto* faults = std::bit_cast<const uint64_t*>(mapped);
		// The parser counts every faulting page but stores at most MaxPageFaults - 1 records
		// (entries 1..MaxPageFaults - 1): a larger count would read past this area into another
		// batch's records or past the download buffer (chenxiao07/KytyPS5 d645c916b). Pages left
		// out stay set in the bitmap for the next scan (fd04c3fa9).
		const auto count = static_cast<uint32_t>(
		    std::min<uint64_t>(static_cast<uint32_t>(faults[0]), MaxPageFaults - 1));
		BufferCache::s_tag_hops.records.fetch_add(count, std::memory_order_relaxed);
		for (uint32_t index = 1; index <= count; ++index) {
			fault_ranges.Add(faults[index], BufferCache::CACHING_PAGESIZE);
			if (!m_buffer_cache.BdaAccessTagsBuilt()) { // tags: most reports are untagged backed pages
				LOGF("Accessed non-GPU cached memory at 0x%016" PRIx64 "\n", faults[index]);
			}
		}
		fault_ranges.ForEach([this](uint64_t start, uint64_t end) {
			EXIT_IF(end - start > std::numeric_limits<uint32_t>::max());
			(void)m_buffer_cache.FindBuffer(start, end - start);
			m_buffer_cache.NoteBdaFault(start, end - start);
			m_buffer_cache.NoteBdaReachFault(start, end - start); // KYTY_BDA_REACH=verify
			m_buffer_cache.NoteBdaTagFault(start, end - start);   // KYTY_BDA_ACCESS_TAGS
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
	const auto slot = RecordBdaWrites();
	m_scheduler.Wait(m_scheduler.CurrentTick());
	auto writes = ParseBdaWrites(slot);
	if (writes.log_tail != 0) {
		FetchBdaStoreLogTail(writes);
	}
	return writes;
}

// The store log's entries past the prefix the settle downloaded, appended to writes.stores. Nothing
// was recorded since the dispatch but the settle's own parse and copies, which clear the log's
// count and leave its entries alone, so the device log still holds this dispatch's stores.
void FaultManager::FetchBdaStoreLogTail(BdaWrites& writes) {
	KYTY_GPU_OP_SITE("fault.bda-log-tail");
	constexpr uint64_t tail_offset =
	    (BufferCache::BDA_STORE_LOG_WORD + 4 + 4 * BufferCache::BDA_STORE_LOG_PREFIX) * sizeof(uint32_t);
	constexpr uint64_t tail_capacity =
	    (BufferCache::BDA_STORE_LOG_ENTRIES - BufferCache::BDA_STORE_LOG_PREFIX) * sizeof(BdaStore);
	const uint64_t bytes = uint64_t {writes.log_tail} * sizeof(BdaStore);
	EXIT_IF(bytes > tail_capacity);
	if (m_bda_tail_download == nullptr) {
		m_bda_tail_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                               AllFlags, tail_capacity);
	}
	m_scheduler.EndRendering();
	const auto command = m_scheduler.Current().HelperSink();
	vk::BufferMemoryBarrier2 before {};
	before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	before.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
	before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	before.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
	before.buffer        = m_fault_buffer.Handle();
	before.offset        = tail_offset;
	before.size          = bytes;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &before;
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy copy {tail_offset, 0, bytes};
	command.copyBuffer(m_fault_buffer.Handle(), m_bda_tail_download->Handle(), 1, &copy);
	// The next BDA writer may log there again; the host sees the copy.
	std::array<vk::BufferMemoryBarrier2, 2> after {};
	after[0].srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after[0].srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	after[0].dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	after[0].dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	after[0].buffer        = m_fault_buffer.Handle();
	after[0].offset        = tail_offset;
	after[0].size          = bytes;
	after[1].srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	after[1].srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	after[1].dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	after[1].dstAccessMask = vk::AccessFlagBits2::eHostRead;
	after[1].buffer        = m_bda_tail_download->Handle();
	after[1].offset        = 0;
	after[1].size          = bytes;
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(after.size());
	dependency.pBufferMemoryBarriers    = after.data();
	command.pipelineBarrier2(dependency);
	m_scheduler.Wait(m_scheduler.CurrentTick());
	m_bda_tail_download->Invalidate(0, bytes);
	const auto first = writes.stores.size();
	writes.stores.resize(first + writes.log_tail);
	std::memcpy(writes.stores.data() + first, m_bda_tail_download->Mapped().data(), bytes);
	writes.log_tail = 0;
}

uint32_t FaultManager::RecordBdaWrites() {
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
	}
	const auto slot = m_bda_next_slot++ % BdaSlots;
	auto&      m_bda_download = m_bda_downloads[slot];
	if (m_bda_download == nullptr) {
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
	const auto command = m_scheduler.Current().HelperSink();
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
	const bool     store_log  = StoreLogEnabled();
	constexpr auto log_offset = BufferCache::BDA_STORE_LOG_WORD * sizeof(uint32_t);
	if (store_log) {
		// The first BDA_STORE_LOG_PREFIX entries; CollectBdaWrites fetches the rest when there are more.
		const vk::BufferCopy log_copy {log_offset, BdaStoreLogOffset,
		                               BufferCache::BDA_STORE_LOG_PREFIX_BYTES};
		command.copyBuffer(m_fault_buffer.Handle(), m_bda_download->Handle(), 1, &log_copy);
	}
	// The counter and the store log's count (the words between them are unused) are cleared.
	constexpr auto counters_size = log_offset + sizeof(uint32_t) - dropped_offset;
	vk::BufferMemoryBarrier2 cleared {};
	cleared.srcStageMask             = vk::PipelineStageFlagBits2::eTransfer;
	cleared.srcAccessMask            = vk::AccessFlagBits2::eTransferRead;
	cleared.dstStageMask             = vk::PipelineStageFlagBits2::eTransfer;
	cleared.dstAccessMask            = vk::AccessFlagBits2::eTransferWrite;
	cleared.buffer                   = m_fault_buffer.Handle();
	cleared.offset                   = dropped_offset;
	cleared.size                     = counters_size;
	dependency.pBufferMemoryBarriers = &cleared;
	command.pipelineBarrier2(dependency);
	command.fillBuffer(m_fault_buffer.Handle(), dropped_offset, counters_size, 0);
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
	return slot;
}

FaultManager::BdaWrites FaultManager::ParseBdaWrites(uint32_t slot) {
	auto&       m_bda_download = m_bda_downloads[slot % BdaSlots];
	const bool  store_log      = StoreLogEnabled();
	auto*       mapped         = m_bda_download->Mapped().data();
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
	if (store_log) {
		m_bda_download->Invalidate(BdaStoreLogOffset, sizeof(uint32_t));
		uint32_t logged = 0;
		std::memcpy(&logged, mapped + BdaStoreLogOffset, sizeof(logged));
		// The device log kept `kept` entries (its capacity); the download holds the first `entries`.
		const auto kept    = std::min<size_t>(logged, BufferCache::BDA_STORE_LOG_ENTRIES);
		const auto entries = std::min<size_t>(kept, BufferCache::BDA_STORE_LOG_PREFIX);
		result.log_overflow = logged - static_cast<uint32_t>(kept);
		result.log_tail     = static_cast<uint32_t>(kept - entries);
		if (entries != 0) {
			constexpr size_t first = BdaStoreLogOffset + 4 * sizeof(uint32_t);
			m_bda_download->Invalidate(first, entries * sizeof(BdaStore));
			result.stores.resize(entries);
			std::memcpy(result.stores.data(), mapped + first, entries * sizeof(BdaStore));
		}
	}
	return result;
}

} // namespace Libs::Graphics
