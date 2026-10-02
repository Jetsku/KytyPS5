#include "graphics/host_gpu/renderer/image/tiler.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "gpu_tiler_shaders/gpu_tiler_demote_d16_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_depth_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_depth_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_prt_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_render_target_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_standard256_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_standard4_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_load_standard64_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_depth_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_prt_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_render_target_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_standard256_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_standard4_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_image_store_standard64_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_promote_d16_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_prt_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_prt_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_render_target_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard256_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard4_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard4_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard64_3d_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_standard64_spv.h"
#include "gpu_tiler_shaders/gpu_tiler_swap_bgra16_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>

namespace Libs::Graphics {

TileManager::TileManager(GraphicContext& graphics, CommandScheduler& scheduler,
                         StreamBuffer& stream_buffer)
    : m_graphics(graphics), m_scheduler(scheduler), m_stream_buffer(stream_buffer) {
	static_assert(FamilyCount == 9);
	static_assert(sizeof(Push) == 64);
	std::array<vk::DescriptorSetLayoutBinding, 3> bindings {};
	for (uint32_t index = 0; index < 2; index++) {
		bindings[index] = {index, vk::DescriptorType::eStorageBuffer, 1,
		                   vk::ShaderStageFlagBits::eCompute, nullptr};
	}
	bindings[2] = {2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute,
	               nullptr};

	vk::DescriptorSetLayoutCreateInfo descriptor_info {};
	descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor_info.pBindings    = bindings.data();
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
	                                                                 &m_descriptor_layout),
	                     "create TileManager descriptor layout");

	const vk::PushConstantRange  push_range {vk::ShaderStageFlagBits::eCompute, 0, sizeof(Push)};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &m_descriptor_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(
	    m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_pipeline_layout),
	    "create TileManager pipeline layout");

	// KYTY_TILER_SCRATCH_POOL=0 restores one native scratch buffer per detile/tile operation;
	// KYTY_TILER_SCRATCH_POOL_MB bounds the idle pooled bytes (default 256).
	const auto* pool = std::getenv("KYTY_TILER_SCRATCH_POOL");
	if (pool == nullptr || std::strcmp(pool, "0") != 0) {
		m_scratch_pool_limit = 256ull * 1024 * 1024;
		if (const auto* limit = std::getenv("KYTY_TILER_SCRATCH_POOL_MB"); limit != nullptr) {
			m_scratch_pool_limit = std::strtoull(limit, nullptr, 10) * 1024ull * 1024ull;
		}
		// KYTY_TILER_SCRATCH_POOL_IDLE_MS=N (default 0: off): pooled scratch buffers no detile has
		// reused for N ms are destroyed (TrimScratchPool, from the image garbage collector).
		if (const auto* idle = std::getenv("KYTY_TILER_SCRATCH_POOL_IDLE_MS"); idle != nullptr) {
			m_scratch_idle = std::chrono::milliseconds(
			    static_cast<int64_t>(std::min(std::strtoull(idle, nullptr, 10), 3600000ull)));
		}
	}
	// Detile output is consumed only by buffer->image copies (and element-wise conversions)
	// that read exactly the width x height elements of each tile at its pitch, all of which the
	// dispatches write; only row/level padding was cleared. KYTY_TILER_CLEAR_SCRATCH=1 clears.
	if (const auto* clear = std::getenv("KYTY_TILER_CLEAR_SCRATCH");
	    clear != nullptr && std::strcmp(clear, "1") == 0) {
		m_clear_detile_scratch = true;
	}
}

TileManager::~TileManager() {
	for (const auto& scratch: m_scratch_pool) {
		VramStats::Note(VramStats::Kind::TilerScratch, true, -static_cast<int64_t>(scratch.capacity));
		vmaDestroyBuffer(m_graphics.allocator, scratch.buffer, scratch.allocation);
	}
	m_scratch_pool.clear();
	for (auto pipeline: m_pipelines) {
		if (pipeline != nullptr) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
	}
	for (auto pipeline: m_image_pipelines) {
		if (pipeline != nullptr) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
	}
	if (m_image_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_image_pipeline_layout, nullptr);
	}
	if (m_image_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_image_descriptor_layout, nullptr);
	}
	if (m_d16_to_d24 != nullptr) {
		m_graphics.device.destroyPipeline(m_d16_to_d24, nullptr);
	}
	if (m_d16_to_d32 != nullptr) {
		m_graphics.device.destroyPipeline(m_d16_to_d32, nullptr);
	}
	if (m_d24_to_d16 != nullptr) {
		m_graphics.device.destroyPipeline(m_d24_to_d16, nullptr);
	}
	if (m_d32_to_d16 != nullptr) {
		m_graphics.device.destroyPipeline(m_d32_to_d16, nullptr);
	}
	if (m_swap_bgra16 != nullptr) {
		m_graphics.device.destroyPipeline(m_swap_bgra16, nullptr);
	}
	if (m_pipeline_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	}
	if (m_descriptor_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_descriptor_layout, nullptr);
	}
}

TileManager::Scratch TileManager::AllocateScratch(uint64_t size) {
	EXIT_IF(size == 0);
	// Every upload detiles into a fresh scratch buffer. Reuse completed ones by size class
	// instead of creating and destroying a native buffer per upload.
	constexpr uint64_t MinClass   = 64ull * 1024;
	constexpr uint64_t LargeClass = 64ull * 1024 * 1024;
	const uint64_t     capacity   = m_scratch_pool_limit == 0 ? size
	                                : size <= LargeClass
	                                    ? std::max(MinClass, std::bit_ceil(size))
	                                    : Common::AlignUp(size, 16ull * 1024 * 1024);
	if (m_scratch_pool_limit != 0) {
		std::scoped_lock lock(m_scratch_mutex);
		for (size_t index = m_scratch_pool.size(); index > 0; --index) {
			if (m_scratch_pool[index - 1].capacity == capacity) {
				auto scratch = m_scratch_pool[index - 1];
				m_scratch_pool.erase(m_scratch_pool.begin() + static_cast<std::ptrdiff_t>(index - 1));
				m_scratch_pool_bytes -= capacity;
				Profiler::CountFrameEvent(Profiler::FrameEvent::TilerScratchPoolHits);
				scratch.size = size;
				return scratch;
			}
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TilerScratchPoolMisses);
	}
	// Native scratch creations only (pool misses, or every use with the pool off).
	MemoryStats::Count(MemoryStats::Counter::ScratchAllocs);
	MemoryStats::Count(MemoryStats::Counter::ScratchAllocBytes, capacity);
	const MemoryStats::ScopedTimer timer(MemoryStats::Counter::ScratchAllocNs);
	vk::BufferCreateInfo create {};
	create.size  = capacity;
	create.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
	               vk::BufferUsageFlagBits::eTransferDst;

	VmaAllocationCreateInfo allocate {};
	allocate.usage       = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	VkBuffer      buffer = VK_NULL_HANDLE;
	VmaAllocation memory = nullptr;
	const auto    raw    = static_cast<VkBufferCreateInfo>(create);
	RequireVulkanSuccess(static_cast<vk::Result>(vmaCreateBuffer(
	                         m_graphics.allocator, &raw, &allocate, &buffer, &memory, nullptr)),
	                     "allocate TileManager scratch buffer");
	VramStats::Note(VramStats::Kind::TilerScratch, true, static_cast<int64_t>(capacity));
	return {buffer, memory, size, capacity};
}

std::pair<uint64_t, uint64_t> TileManager::ScratchPoolBytes() {
	std::scoped_lock lock(m_scratch_mutex);
	return {m_scratch_pool_bytes, m_scratch_pool_limit};
}

void TileManager::TrimScratchPool() {
	if (m_scratch_idle.count() == 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now < m_scratch_trim_next) {
		return;
	}
	m_scratch_trim_next = now + std::chrono::milliseconds(100);
	std::vector<Scratch> expired;
	{
		std::scoped_lock lock(m_scratch_mutex);
		// Released in order: the oldest entries are at the front.
		size_t count = 0;
		while (count < m_scratch_pool.size() && now - m_scratch_pool[count].released >= m_scratch_idle) {
			m_scratch_pool_bytes -= m_scratch_pool[count].capacity;
			expired.push_back(m_scratch_pool[count]);
			++count;
		}
		m_scratch_pool.erase(m_scratch_pool.begin(),
		                     m_scratch_pool.begin() + static_cast<std::ptrdiff_t>(count));
	}
	for (const auto& old: expired) {
		VramStats::Note(VramStats::Kind::TilerScratch, true, -static_cast<int64_t>(old.capacity));
		vmaDestroyBuffer(m_graphics.allocator, old.buffer, old.allocation);
	}
}

void TileManager::DeferDestroy(Scratch scratch) {
	if (m_scratch_pool_limit != 0) {
		// Runs once the scheduler tick that uses the buffer has completed.
		m_scheduler.DeferOperation([this, scratch] { ReleaseScratch(scratch); });
		return;
	}
	auto allocator = m_graphics.allocator;
	m_scheduler.DeferOperation([allocator, scratch] {
		VramStats::Note(VramStats::Kind::TilerScratch, true, -static_cast<int64_t>(scratch.capacity));
		vmaDestroyBuffer(allocator, scratch.buffer, scratch.allocation);
	});
}

void TileManager::ReleaseScratch(Scratch scratch) {
	std::vector<Scratch> evicted;
	{
		std::scoped_lock lock(m_scratch_mutex);
		if (scratch.capacity <= m_scratch_pool_limit) {
			while (!m_scratch_pool.empty() &&
			       scratch.capacity > m_scratch_pool_limit - m_scratch_pool_bytes) {
				evicted.push_back(m_scratch_pool.front());
				m_scratch_pool_bytes -= m_scratch_pool.front().capacity;
				m_scratch_pool.erase(m_scratch_pool.begin());
			}
			scratch.released = std::chrono::steady_clock::now();
			m_scratch_pool.push_back(scratch);
			m_scratch_pool_bytes += scratch.capacity;
		} else {
			evicted.push_back(scratch);
		}
	}
	for (const auto& old: evicted) {
		VramStats::Note(VramStats::Kind::TilerScratch, true, -static_cast<int64_t>(old.capacity));
		vmaDestroyBuffer(m_graphics.allocator, old.buffer, old.allocation);
	}
}

void TileManager::Prepare(bool tile, uint64_t tiled_capacity, uint64_t linear_capacity,
                          std::span<const GpuTileInfo> infos, uint64_t source_base,
                          uint64_t target_base, std::vector<Dispatch>& dispatches,
                          std::span<const vk::BufferImageCopy> image_regions,
                          uint32_t image_texel, bool image_layer_views) {
	EXIT_IF(infos.empty() || tiled_capacity == 0 || linear_capacity == 0 ||
	        (!image_regions.empty() && image_regions.size() != infos.size()));
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	EXIT_NOT_IMPLEMENTED(tiled_capacity > UINT32_MAX || linear_capacity > UINT32_MAX);

	const auto checked_multiply = [](uint64_t left, uint64_t right, uint64_t& result) {
		return (left == 0 || right <= UINT64_MAX / left) && (result = left * right, true);
	};
	const auto checked_add = [](uint64_t left, uint64_t right, uint64_t& result) {
		return right <= UINT64_MAX - left && (result = left + right, true);
	};
	const auto valid_range = [](uint64_t offset, uint64_t size, uint64_t capacity) {
		return size != 0 && offset <= capacity && size <= capacity - offset;
	};

	dispatches.clear();
	dispatches.reserve(infos.size());
	for (const auto& info: infos) {
		TileBlockLayout block {};
		const uint32_t  tiled_width  = info.tiled_width != 0 ? info.tiled_width : info.pitch;
		const uint32_t  tiled_height = info.tiled_height != 0 ? info.tiled_height : info.height;
		const uint64_t  groups_x     = (static_cast<uint64_t>(info.width) + 7u) / 8u;
		const uint64_t  groups_y     = (static_cast<uint64_t>(info.height) + 7u) / 8u;
		EXIT_NOT_IMPLEMENTED(
		    !TileGetBlockLayout(info.family, info.bytes_per_element, block) || info.width == 0 ||
		    info.height == 0 || info.depth == 0 || info.pitch < info.width ||
		    groups_x > limits.maxComputeWorkGroupCount[0] ||
		    groups_y > limits.maxComputeWorkGroupCount[1] ||
		    info.depth > limits.maxComputeWorkGroupCount[2] ||
		    (!info.tail && (tiled_width < info.width || tiled_height < info.height)) ||
		    !valid_range(info.linear_offset, info.linear_size, linear_capacity) ||
		    !valid_range(info.tiled_offset, info.tiled_size, tiled_capacity) ||
		    (block.block_depth == 1 && info.depth != 1));

		uint64_t pitch_bytes = 0;
		EXIT_NOT_IMPLEMENTED(!checked_multiply(info.pitch, info.bytes_per_element, pitch_bytes) ||
		                     pitch_bytes > UINT32_MAX);
		uint64_t slice_bytes   = info.linear_slice_stride;
		uint64_t minimum_slice = 0;
		EXIT_NOT_IMPLEMENTED(!checked_multiply(pitch_bytes, info.height, minimum_slice));
		if (slice_bytes == 0) {
			slice_bytes = minimum_slice;
		}
		uint64_t linear_used = 0;
		uint64_t bytes       = 0;
		EXIT_NOT_IMPLEMENTED((info.depth > 1 && slice_bytes < minimum_slice) ||
		                     !checked_multiply(info.depth - 1u, slice_bytes, bytes) ||
		                     !checked_add(linear_used, bytes, linear_used) ||
		                     !checked_multiply(info.height - 1u, pitch_bytes, bytes) ||
		                     !checked_add(linear_used, bytes, linear_used) ||
		                     !checked_multiply(info.width, info.bytes_per_element, bytes) ||
		                     !checked_add(linear_used, bytes, linear_used) ||
		                     linear_used > info.linear_size || slice_bytes > UINT32_MAX);

		const uint64_t columns =
		    (static_cast<uint64_t>(tiled_width) + block.block_width - 1u) / block.block_width;
		const uint64_t rows =
		    (static_cast<uint64_t>(tiled_height) + block.block_height - 1u) / block.block_height;
		uint64_t blocks_per_slice = 0;
		EXIT_NOT_IMPLEMENTED(!checked_multiply(columns, rows, blocks_per_slice) ||
		                     columns > UINT32_MAX || blocks_per_slice > UINT32_MAX);
		if (info.tail) {
			EXIT_NOT_IMPLEMENTED(
			    info.family == TileBlockFamily::Standard256B || info.depth > block.block_depth ||
			    info.tail_x >= block.block_width || info.width > block.block_width - info.tail_x ||
			    info.tail_y >= block.block_height ||
			    info.height > block.block_height - info.tail_y ||
			    info.tiled_size < block.block_size);
		} else {
			const uint64_t slices =
			    (static_cast<uint64_t>(info.depth) + block.block_depth - 1u) / block.block_depth;
			uint64_t tiled_used = 0;
			EXIT_NOT_IMPLEMENTED(!checked_multiply(blocks_per_slice, slices, tiled_used) ||
			                     !checked_multiply(tiled_used, block.block_size, tiled_used) ||
			                     tiled_used > info.tiled_size);
		}

		const uint32_t alignment = std::min(info.bytes_per_element, 4u);
		EXIT_NOT_IMPLEMENTED(((info.linear_offset | info.tiled_offset | pitch_bytes | slice_bytes) &
		                      (alignment - 1u)) != 0);
		const uint64_t src = source_base + (tile ? info.linear_offset : info.tiled_offset);
		const uint64_t dst = target_base + (tile ? info.tiled_offset : info.linear_offset);
		EXIT_NOT_IMPLEMENTED(src > UINT32_MAX || dst > UINT32_MAX);

		const uint32_t family_index  = static_cast<uint32_t>(info.family);
		const uint32_t element_index = std::countr_zero(info.bytes_per_element);
		EXIT_NOT_IMPLEMENTED(family_index >= FamilyCount || element_index >= BytesPerElementCount);
		Dispatch dispatch {};
		dispatch.pipeline_slot =
		    ((tile ? FamilyCount : 0u) + family_index) * BytesPerElementCount + element_index;
		dispatch.push.src_base         = static_cast<uint32_t>(src);
		dispatch.push.dst_base         = static_cast<uint32_t>(dst);
		dispatch.push.width            = info.width;
		dispatch.push.height           = info.height;
		dispatch.push.depth            = info.depth;
		dispatch.push.surface_z        = info.surface_z;
		dispatch.push.pitch_bytes      = static_cast<uint32_t>(pitch_bytes);
		dispatch.push.slice_bytes      = static_cast<uint32_t>(slice_bytes);
		dispatch.push.blocks_per_row   = static_cast<uint32_t>(columns);
		dispatch.push.blocks_per_slice = static_cast<uint32_t>(blocks_per_slice);
		dispatch.push.tail_x           = info.tail_x;
		dispatch.push.tail_y           = info.tail_y;
		dispatch.push.tail             = info.tail;
		if (!image_regions.empty()) {
			// Image variants: element (x, y) of this tile is view texel imageOffset / texel +
			// (x, y) of the region's layer (ImageTransferEligible admits only 2D regions that
			// start on a block).
			EXIT_IF(image_texel == 0);
			const auto& region        = image_regions[dispatches.size()];
			dispatch.push.image_x     = static_cast<uint32_t>(region.imageOffset.x) / image_texel;
			dispatch.push.image_y     = static_cast<uint32_t>(region.imageOffset.y) / image_texel;
			dispatch.push.image_layer =
			    image_layer_views ? 0u : region.imageSubresource.baseArrayLayer;
		}
		dispatches.push_back(dispatch);
	}

	const uint64_t uniform_alignment =
	    std::max<uint64_t>(limits.minUniformBufferOffsetAlignment, 1);
	const uint64_t stride = Common::AlignUp<uint64_t>(sizeof(Push), uniform_alignment);
	EXIT_NOT_IMPLEMENTED(dispatches.size() > UINT64_MAX / stride);
	const uint64_t bytes  = dispatches.size() * stride;
	auto [mapped, offset] = m_stream_buffer.Map(bytes, uniform_alignment);
	EXIT_IF(mapped == nullptr);
	for (size_t index = 0; index < dispatches.size(); index++) {
		std::memcpy(mapped + index * stride, &dispatches[index].push, sizeof(Push));
		dispatches[index].params_offset = offset + index * stride;
	}
	m_stream_buffer.Commit();
}

vk::Pipeline TileManager::GetPipeline(uint32_t slot) {
	EXIT_IF(slot >= m_pipelines.size());
	if (m_pipelines[slot] != nullptr) {
		return m_pipelines[slot];
	}
	struct Shader {
		const uint32_t* code;
		size_t          words;
	};
	static constexpr std::array<Shader, FamilyCount> shaders {{
	    {GPU_TILER_STANDARD256_SPV, std::size(GPU_TILER_STANDARD256_SPV)},
	    {GPU_TILER_STANDARD4_SPV, std::size(GPU_TILER_STANDARD4_SPV)},
	    {GPU_TILER_STANDARD4_3D_SPV, std::size(GPU_TILER_STANDARD4_3D_SPV)},
	    {GPU_TILER_STANDARD64_SPV, std::size(GPU_TILER_STANDARD64_SPV)},
	    {GPU_TILER_STANDARD64_3D_SPV, std::size(GPU_TILER_STANDARD64_3D_SPV)},
	    {GPU_TILER_PRT_SPV, std::size(GPU_TILER_PRT_SPV)},
	    {GPU_TILER_PRT_3D_SPV, std::size(GPU_TILER_PRT_3D_SPV)},
	    {GPU_TILER_RENDER_TARGET_SPV, std::size(GPU_TILER_RENDER_TARGET_SPV)},
	    {GPU_TILER_DEPTH_SPV, std::size(GPU_TILER_DEPTH_SPV)},
	}};
	const uint32_t                                   element_index = slot % BytesPerElementCount;
	const uint32_t                   direction_index = slot / (FamilyCount * BytesPerElementCount);
	const uint32_t                   family_index    = (slot / BytesPerElementCount) % FamilyCount;
	const uint32_t                   values[] {1u << element_index, direction_index};
	const vk::SpecializationMapEntry entries[] {{0, 0, 4}, {1, 4, 4}};
	const vk::SpecializationInfo     specialization {2, entries, sizeof(values), values};
	const auto module =
	    CompileSPV({shaders[family_index].code, shaders[family_index].words}, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo create {};
	create.stage  = stage;
	create.layout = m_pipeline_layout;
	const auto result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &m_pipelines[slot]);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create TileManager pipeline");
	return m_pipelines[slot];
}

void TileManager::Record(vk::Buffer source, uint64_t source_offset,
                         uint64_t source_capacity, vk::Buffer target, uint64_t target_offset,
                         uint64_t target_capacity, std::span<Dispatch> dispatches,
                         bool clear_target) {
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t source_descriptor_offset = Common::AlignDown(source_offset, descriptor_alignment);
	const uint64_t target_descriptor_offset = Common::AlignDown(target_offset, descriptor_alignment);
	const uint64_t source_base              = source_offset - source_descriptor_offset;
	const uint64_t target_base              = target_offset - target_descriptor_offset;
	const uint64_t source_range             = Common::AlignUp(source_base + source_capacity, 4);
	const uint64_t target_range             = Common::AlignUp(target_base + target_capacity, 4);
	EXIT_NOT_IMPLEMENTED(source_range > limits.maxStorageBufferRange ||
	                     target_range > limits.maxStorageBufferRange || target_offset % 4 != 0 ||
	                     target_capacity % 4 != 0);

	m_scheduler.EndRendering();
	auto                    command = m_scheduler.Current().Handle();
	vk::BufferMemoryBarrier barriers[3] {};
	barriers[0].srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite;
	barriers[0].dstAccessMask = vk::AccessFlagBits::eShaderRead;
	barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[0].buffer              = source;
	barriers[0].offset              = source_offset;
	barriers[0].size                = source_capacity;
	barriers[1]                     = barriers[0];
	barriers[1].dstAccessMask =
	    clear_target ? vk::AccessFlagBits::eTransferWrite
	                 : vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barriers[1].buffer        = target;
	barriers[1].offset        = target_offset;
	barriers[1].size          = target_capacity;
	barriers[2]               = barriers[0];
	barriers[2].srcAccessMask = vk::AccessFlagBits::eHostWrite;
	barriers[2].dstAccessMask = vk::AccessFlagBits::eUniformRead;
	barriers[2].buffer        = m_stream_buffer.Handle();
	barriers[2].offset        = dispatches.front().params_offset;
	barriers[2].size =
	    dispatches.back().params_offset - dispatches.front().params_offset + sizeof(Push);
	command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	    vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer, {}, 0,
	    nullptr, 3, barriers, 0, nullptr);
	if (clear_target) {
		command.fillBuffer(target, target_offset, target_capacity, 0);
		barriers[1].srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		barriers[1].dstAccessMask =
		    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 1,
		                        &barriers[1], 0, nullptr);
	}

	const vk::DescriptorBufferInfo source_info {source, source_descriptor_offset, source_range};
	const vk::DescriptorBufferInfo target_info {target, target_descriptor_offset, target_range};
	for (auto& dispatch: dispatches) {
		const vk::DescriptorBufferInfo        params_info {m_stream_buffer.Handle(),
		                                                   dispatch.params_offset, sizeof(Push)};
		const vk::DescriptorBufferInfo        infos[] {source_info, target_info, params_info};
		std::array<vk::WriteDescriptorSet, 3> writes {};
		for (uint32_t index = 0; index < writes.size(); index++) {
			writes[index].dstBinding      = index;
			writes[index].descriptorCount = 1;
			writes[index].descriptorType  = index == 2 ? vk::DescriptorType::eUniformBuffer
			                                           : vk::DescriptorType::eStorageBuffer;
			writes[index].pBufferInfo     = &infos[index];
		}
		m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
		                             static_cast<uint32_t>(writes.size()), writes.data());
		m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, GetPipeline(dispatch.pipeline_slot));
		command.dispatch((dispatch.push.width + 7u) / 8u, (dispatch.push.height + 7u) / 8u,
		                 dispatch.push.depth);
	}

	barriers[1].srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barriers[1].dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eMemoryRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1,
	                        &barriers[1], 0, nullptr);
}

TileManager::Result TileManager::Detile(vk::Buffer tiled, uint64_t tiled_offset,
                                        uint64_t tiled_capacity, uint64_t linear_capacity,
                                        std::span<const GpuTileInfo> infos) {
	KYTY_GPU_OP_SITE("tiler.detile");
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t        source_base = tiled_offset & (descriptor_alignment - 1);
	std::vector<Dispatch> dispatches;
	Prepare(false, tiled_capacity, linear_capacity, infos, source_base, 0, dispatches);
	auto scratch = AllocateScratch(Common::AlignUp(linear_capacity, 4));
	DeferDestroy(scratch);
	Record(tiled, tiled_offset, tiled_capacity, scratch.buffer, 0, scratch.size, dispatches,
	       m_clear_detile_scratch);
	return {scratch.buffer, 0, linear_capacity};
}

void TileManager::Tile(vk::Buffer linear, uint64_t linear_offset, uint64_t linear_capacity,
                       vk::Buffer tiled, uint64_t tiled_offset, uint64_t tiled_capacity,
                       std::span<const GpuTileInfo> infos) {
	KYTY_GPU_OP_SITE("tiler.tile");
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t        source_base = linear_offset & (descriptor_alignment - 1);
	const uint64_t        target_base = tiled_offset & (descriptor_alignment - 1);
	std::vector<Dispatch> dispatches;
	Prepare(true, tiled_capacity, linear_capacity, infos, source_base, target_base, dispatches);
	Record(linear, linear_offset, linear_capacity, tiled, tiled_offset, tiled_capacity,
	       dispatches, false);
}

void TileManager::TileImage(Image& image, std::span<const vk::BufferImageCopy> regions,
                            vk::Buffer tiled, uint64_t tiled_offset, uint64_t tiled_capacity,
                            uint64_t linear_capacity, std::span<const GpuTileInfo> infos,
                            ColorTransform transform) {
	KYTY_GPU_OP_SITE("tiler.tile_image");
	EXIT_IF(regions.empty());
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t        target_base = tiled_offset & (descriptor_alignment - 1);
	std::vector<Dispatch> dispatches;
	// Reserve all stream parameters before creating a scheduler-lived scratch dependency:
	// StreamBuffer::Map is allowed to submit the current tick when it wraps.
	Prepare(true, tiled_capacity, linear_capacity, infos, 0, target_base, dispatches);
	auto linear = AllocateScratch(Common::AlignUp(linear_capacity, 4));
	DeferDestroy(linear);
	image.Download(regions, linear.buffer, 0, linear.size);
	Result source {linear.buffer, 0, linear.size};
	if (transform == ColorTransform::SwapBgra16) {
		source = SwapBgra16(source);
	}
	Record(source.buffer, source.offset, linear_capacity, tiled, tiled_offset, tiled_capacity,
	       dispatches, false);
}

bool TileManager::ImageDirectEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TILER_IMAGE_DIRECT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool TileManager::ImageDirectVerifyEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TILER_IMAGE_DIRECT_VERIFY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void TileManager::VerifyOnCompletion(const char* operation, uint64_t guest_address,
                                     vk::Buffer expected, uint64_t expected_offset,
                                     vk::Buffer actual, uint64_t actual_offset, uint64_t size,
                                     std::vector<std::pair<uint64_t, uint64_t>> ranges) {
	EXIT_IF(size == 0 || expected == nullptr || actual == nullptr);
	const uint64_t half = Common::AlignUp(size, 16);
	vk::BufferCreateInfo create {};
	create.size  = half * 2;
	create.usage = vk::BufferUsageFlagBits::eTransferDst;
	VmaAllocationCreateInfo allocate {};
	allocate.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	allocate.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
	VkBuffer          buffer = VK_NULL_HANDLE;
	VmaAllocation     memory = nullptr;
	VmaAllocationInfo info {};
	const auto        raw = static_cast<VkBufferCreateInfo>(create);
	RequireVulkanSuccess(static_cast<vk::Result>(vmaCreateBuffer(m_graphics.allocator, &raw,
	                                                             &allocate, &buffer, &memory, &info)),
	                     "allocate TileManager verify readback");
	m_scheduler.EndRendering();
	auto                                    command = m_scheduler.Current().Handle();
	std::array<vk::BufferMemoryBarrier2, 2> before {};
	for (auto& barrier: before) {
		barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
		barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
		barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.size                = size;
	}
	before[0].buffer = expected;
	before[0].offset = expected_offset;
	before[1].buffer = actual;
	before[1].offset = actual_offset;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(before.size());
	dependency.pBufferMemoryBarriers    = before.data();
	command.pipelineBarrier2(dependency);
	const vk::BufferCopy expected_copy {expected_offset, 0, size};
	const vk::BufferCopy actual_copy {actual_offset, half, size};
	command.copyBuffer(expected, buffer, 1, &expected_copy);
	command.copyBuffer(actual, buffer, 1, &actual_copy);
	vk::BufferMemoryBarrier2 after {};
	after.srcStageMask        = vk::PipelineStageFlagBits2::eCopy;
	after.srcAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	after.dstStageMask        = vk::PipelineStageFlagBits2::eHost;
	after.dstAccessMask       = vk::AccessFlagBits2::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer              = buffer;
	after.offset              = 0;
	after.size                = VK_WHOLE_SIZE;
	vk::DependencyInfo after_dependency {};
	after_dependency.bufferMemoryBarrierCount = 1;
	after_dependency.pBufferMemoryBarriers    = &after;
	command.pipelineBarrier2(after_dependency);
	// Also restores the order of the source buffers' later writers after these reads.
	vk::MemoryBarrier2 war {};
	war.srcStageMask = vk::PipelineStageFlagBits2::eCopy;
	war.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	vk::DependencyInfo war_dependency {};
	war_dependency.memoryBarrierCount = 1;
	war_dependency.pMemoryBarriers    = &war;
	command.pipelineBarrier2(war_dependency);

	auto allocator = m_graphics.allocator;
	m_scheduler.DeferOperation([allocator, buffer, memory, mapped = info.pMappedData, half, size,
	                            ranges = std::move(ranges), label = std::string(operation),
	                            guest_address] {
		(void)vmaInvalidateAllocation(allocator, memory, 0, VK_WHOLE_SIZE);
		const auto* bytes      = static_cast<const uint8_t*>(mapped);
		uint64_t    mismatches = 0;
		uint64_t    first      = UINT64_MAX;
		for (const auto& [offset, length]: ranges) {
			if (length == 0) {
				continue;
			}
			if (offset > size || length > size - offset) {
				++mismatches;
				first = std::min(first, offset);
				continue;
			}
			if (std::memcmp(bytes + offset, bytes + half + offset, length) != 0) {
				++mismatches;
				for (uint64_t index = 0; index < length; ++index) {
					if (bytes[offset + index] != bytes[half + offset + index]) {
						first = std::min(first, offset + index);
						break;
					}
				}
			}
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageVerifyChecks);
		if (mismatches != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageVerifyMismatches);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("Tiler image-direct verify: %s mismatch guest=0x%016" PRIx64
				     " ranges=%" PRIu64 " first=0x%" PRIx64 " size=0x%" PRIx64 "\n",
				     label.c_str(), guest_address, mismatches, first, size);
			}
		}
		vmaDestroyBuffer(allocator, buffer, memory);
	});
}

vk::Format TileManager::ImageViewFormat(uint32_t bytes_per_element, bool load) {
	static constexpr std::array<vk::Format, BytesPerElementCount> formats {
	    vk::Format::eR8Uint, vk::Format::eR16Uint, vk::Format::eR32Uint, vk::Format::eR32G32Uint,
	    vk::Format::eR32G32B32A32Uint};
	if (bytes_per_element == 0 || bytes_per_element > 16 ||
	    !std::has_single_bit(bytes_per_element)) {
		return vk::Format::eUndefined;
	}
	const auto element_index = static_cast<uint32_t>(std::countr_zero(bytes_per_element));
	const auto format        = formats[element_index];
	auto&      support       = m_image_view_support[(load ? BytesPerElementCount : 0u) + element_index];
	if (support == 0) {
		// The shaders access the image through format-less storage views.
		vk::FormatProperties3 properties3 {};
		vk::FormatProperties2 properties2 {};
		properties2.pNext = &properties3;
		m_graphics.physical_device.getFormatProperties2(format, &properties2);
		const auto features = properties3.optimalTilingFeatures;
		const auto access   = load ? vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat
		                           : vk::FormatFeatureFlagBits2::eStorageWriteWithoutFormat;
		const bool usable   = static_cast<bool>(features & vk::FormatFeatureFlagBits2::eStorageImage) &&
		                    static_cast<bool>(features & access) &&
		                    (!load || m_graphics.storage_image_read_without_format_enabled);
		support = usable ? 1u : 2u;
	}
	return support == 1u ? format : vk::Format::eUndefined;
}

bool TileManager::ImageLayerViews(const Image& image) {
	return image.info.IsBlock();
}

bool TileManager::ImageTransferEligible(const Image& image, bool load,
                                        std::span<const GpuTileInfo>         infos,
                                        std::span<const vk::BufferImageCopy> regions) {
	if (!ImageDirectEnabled() || infos.empty() || infos.size() != regions.size()) {
		return false;
	}
	const auto& backing = image.backing;
	// Block-compressed images (KYTY_TILER_IMAGE_DIRECT_BC): uploads only, through uncompressed
	// block-texel views whose texels are the image's 4x4 blocks (Image::ImageUsageFlags gives
	// them storage usage only where the device supports it).
	const bool block = image.info.IsBlock();
	if (backing.image == nullptr || backing.samples != 1 || image.info.samples != 1 ||
	    backing.image_type != vk::ImageType::e2D || image.info.IsVolume() ||
	    !(backing.usage & vk::ImageUsageFlagBits::eStorage) ||
	    DepthAspectTransferFormat(backing.format) != vk::Format::eUndefined ||
	    (block && (load || !ImageOps::BlockStorageUploadsEnabled() ||
	               !(backing.flags & vk::ImageCreateFlagBits::eBlockTexelViewCompatible)))) {
		return false;
	}
	const uint32_t element     = infos.front().bytes_per_element;
	const auto     view_format = ImageViewFormat(element, load);
	// Same size class: the view texel is exactly the backing texel, or for block formats exactly
	// one compressed block (a view of another size would reinterpret the texel layout).
	if (view_format == vk::Format::eUndefined ||
	    !ImageViewOps::FormatsCompatible(backing.format, view_format) ||
	    (block && vk::blockSize(backing.format) != element)) {
		return false;
	}
	const uint32_t texel = block ? 4u : 1u;
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	for (size_t index = 0; index < infos.size(); ++index) {
		const auto& info   = infos[index];
		const auto& region = regions[index];
		switch (info.family) {
			case TileBlockFamily::Standard256B:
			case TileBlockFamily::Standard4KB:
			case TileBlockFamily::Standard64KB:
			case TileBlockFamily::Prt64KB:
			case TileBlockFamily::RenderTarget64KB:
			case TileBlockFamily::Depth64KB: break;
			default: return false;
		}
		const auto& subresource = region.imageSubresource;
		// Element (x, y) of the tile is view texel (offset / texel + x, offset / texel + y): the
		// region must start on a block and cover exactly the tile's elements.
		if (info.bytes_per_element != element || info.depth != 1 || info.width == 0 ||
		    info.height == 0 || subresource.aspectMask != vk::ImageAspectFlagBits::eColor ||
		    subresource.layerCount != 1 || subresource.mipLevel >= backing.mip_levels ||
		    subresource.baseArrayLayer >= backing.layers || region.imageOffset.x < 0 ||
		    region.imageOffset.y < 0 || region.imageOffset.z != 0 ||
		    region.imageOffset.x % texel != 0 || region.imageOffset.y % texel != 0 ||
		    region.imageExtent.depth != 1 ||
		    (region.imageExtent.width + texel - 1u) / texel != info.width ||
		    (region.imageExtent.height + texel - 1u) / texel != info.height ||
		    (info.width + 7u) / 8u > limits.maxComputeWorkGroupCount[0] ||
		    (info.height + 7u) / 8u > limits.maxComputeWorkGroupCount[1]) {
			return false;
		}
		// The view's extent at this level, in view texels (blocks for block formats).
		const uint32_t mip_width =
		    (std::max(backing.extent.width >> subresource.mipLevel, 1u) + texel - 1u) / texel;
		const uint32_t mip_height =
		    (std::max(backing.extent.height >> subresource.mipLevel, 1u) + texel - 1u) / texel;
		if (static_cast<uint64_t>(region.imageOffset.x) / texel + info.width > mip_width ||
		    static_cast<uint64_t>(region.imageOffset.y) / texel + info.height > mip_height) {
			return false;
		}
	}
	return true;
}

vk::Pipeline TileManager::GetImagePipeline(bool load, TileBlockFamily family,
                                           uint32_t bytes_per_element) {
	const auto family_index  = static_cast<uint32_t>(family);
	const auto element_index = static_cast<uint32_t>(std::countr_zero(bytes_per_element));
	EXIT_IF(family_index >= FamilyCount || element_index >= BytesPerElementCount);
	const auto slot =
	    ((load ? FamilyCount : 0u) + family_index) * BytesPerElementCount + element_index;
	if (m_image_pipelines[slot] != nullptr) {
		return m_image_pipelines[slot];
	}
	if (m_image_descriptor_layout == nullptr) {
		const std::array<vk::DescriptorSetLayoutBinding, 3> bindings {{
		    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		    {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		    {2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
		descriptor_info.pBindings    = bindings.data();
		RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(
		                         &descriptor_info, nullptr, &m_image_descriptor_layout),
		                     "create TileManager image descriptor layout");
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount = 1;
		layout_info.pSetLayouts    = &m_image_descriptor_layout;
		RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&layout_info, nullptr,
		                                                            &m_image_pipeline_layout),
		                     "create TileManager image pipeline layout");
	}
	struct Shader {
		const uint32_t* code;
		size_t          words;
	};
	const auto pick = [&](const uint32_t* store_code, size_t store_words, const uint32_t* load_code,
	                      size_t load_words) {
		return load ? Shader {load_code, load_words} : Shader {store_code, store_words};
	};
	Shader shader {nullptr, 0};
	switch (family) {
		case TileBlockFamily::Standard256B:
			shader = pick(GPU_TILER_IMAGE_STORE_STANDARD256_SPV,
			              std::size(GPU_TILER_IMAGE_STORE_STANDARD256_SPV),
			              GPU_TILER_IMAGE_LOAD_STANDARD256_SPV,
			              std::size(GPU_TILER_IMAGE_LOAD_STANDARD256_SPV));
			break;
		case TileBlockFamily::Standard4KB:
			shader = pick(GPU_TILER_IMAGE_STORE_STANDARD4_SPV,
			              std::size(GPU_TILER_IMAGE_STORE_STANDARD4_SPV),
			              GPU_TILER_IMAGE_LOAD_STANDARD4_SPV,
			              std::size(GPU_TILER_IMAGE_LOAD_STANDARD4_SPV));
			break;
		case TileBlockFamily::Standard64KB:
			shader = pick(GPU_TILER_IMAGE_STORE_STANDARD64_SPV,
			              std::size(GPU_TILER_IMAGE_STORE_STANDARD64_SPV),
			              GPU_TILER_IMAGE_LOAD_STANDARD64_SPV,
			              std::size(GPU_TILER_IMAGE_LOAD_STANDARD64_SPV));
			break;
		case TileBlockFamily::Prt64KB:
			shader = pick(GPU_TILER_IMAGE_STORE_PRT_SPV, std::size(GPU_TILER_IMAGE_STORE_PRT_SPV),
			              GPU_TILER_IMAGE_LOAD_PRT_SPV, std::size(GPU_TILER_IMAGE_LOAD_PRT_SPV));
			break;
		case TileBlockFamily::RenderTarget64KB:
			shader = pick(GPU_TILER_IMAGE_STORE_RENDER_TARGET_SPV,
			              std::size(GPU_TILER_IMAGE_STORE_RENDER_TARGET_SPV),
			              GPU_TILER_IMAGE_LOAD_RENDER_TARGET_SPV,
			              std::size(GPU_TILER_IMAGE_LOAD_RENDER_TARGET_SPV));
			break;
		case TileBlockFamily::Depth64KB:
			shader = pick(GPU_TILER_IMAGE_STORE_DEPTH_SPV, std::size(GPU_TILER_IMAGE_STORE_DEPTH_SPV),
			              GPU_TILER_IMAGE_LOAD_DEPTH_SPV, std::size(GPU_TILER_IMAGE_LOAD_DEPTH_SPV));
			break;
		default: EXIT("TileManager: no image variant for tile family %u\n", family_index);
	}
	const uint32_t                   values[] {bytes_per_element, load ? 1u : 0u};
	const vk::SpecializationMapEntry entries[] {{0, 0, 4}, {1, 4, 4}};
	const vk::SpecializationInfo     specialization {2, entries, sizeof(values), values};
	const auto module = CompileSPV({shader.code, shader.words}, m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage               = vk::ShaderStageFlagBits::eCompute;
	stage.module              = module;
	stage.pName               = "main";
	stage.pSpecializationInfo = &specialization;
	vk::ComputePipelineCreateInfo create {};
	create.stage  = stage;
	create.layout = m_image_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr,
	                                                             &m_image_pipelines[slot]);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create TileManager image pipeline");
	return m_image_pipelines[slot];
}

void TileManager::RecordImage(Image& image, bool load, vk::Buffer tiled, uint64_t tiled_offset,
                              uint64_t tiled_capacity, std::span<const GpuTileInfo> infos,
                              std::span<const vk::BufferImageCopy> regions,
                              std::span<Dispatch>                  dispatches) {
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t tiled_descriptor_offset = Common::AlignDown(tiled_offset, descriptor_alignment);
	const uint64_t tiled_range =
	    Common::AlignUp(tiled_offset - tiled_descriptor_offset + tiled_capacity, 4);
	EXIT_NOT_IMPLEMENTED(tiled_range > limits.maxStorageBufferRange);
	const auto view_format = ImageViewFormat(infos.front().bytes_per_element, load);
	EXIT_IF(view_format == vk::Format::eUndefined);
	// Pipelines (and with the first one the image pipeline layout the descriptor pushes below
	// name) and one storage view per mip level (all layers), all before any command is recorded.
	std::vector<vk::Pipeline> pipelines;
	pipelines.reserve(infos.size());
	for (const auto& info: infos) {
		pipelines.push_back(GetImagePipeline(load, info.family, info.bytes_per_element));
	}
	// Uncompressed views of a block-compressed image must hold one level and one layer
	// (block-texel view rules): one view per (level, layer) there, per level otherwise.
	const bool layer_views = ImageLayerViews(image);
	struct LevelView {
		uint32_t      level = 0;
		uint32_t      layer = 0;
		vk::ImageView view  = nullptr;
	};
	std::vector<LevelView> views;
	const auto find_view = [&](const vk::BufferImageCopy& region) {
		const auto level = region.imageSubresource.mipLevel;
		const auto layer = layer_views ? region.imageSubresource.baseArrayLayer : 0u;
		return std::ranges::find_if(views, [&](const LevelView& entry) {
			return entry.level == level && entry.layer == layer;
		});
	};
	for (const auto& region: regions) {
		if (find_view(region) != views.end()) {
			continue;
		}
		ImageViewInfo view_info {};
		view_info.format      = view_format;
		view_info.type        = vk::ImageViewType::e2DArray;
		view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		view_info.base_level  = region.imageSubresource.mipLevel;
		view_info.level_count = 1;
		view_info.base_layer  = layer_views ? region.imageSubresource.baseArrayLayer : 0u;
		view_info.layer_count = layer_views ? 1u : image.backing.layers;
		view_info.usage       = vk::ImageUsageFlagBits::eStorage;
		views.push_back({view_info.base_level, view_info.base_layer, image.FindView(view_info)});
	}

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	// Before: the tiled bytes (possibly host-written staging) and the parameters are visible
	// to the dispatches; the image is in GENERAL for storage access, ordered after every
	// earlier access (Image::GetBarriers).
	const auto image_barriers =
	    image.GetBarriers(vk::ImageLayout::eGeneral,
	                      load ? vk::AccessFlagBits2::eShaderRead : vk::AccessFlagBits2::eShaderWrite,
	                      vk::PipelineStageFlagBits2::eComputeShader, std::nullopt);
	std::array<vk::BufferMemoryBarrier2, 2> before {};
	for (auto& barrier: before) {
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	}
	before[0].srcStageMask =
	    vk::PipelineStageFlagBits2::eAllCommands | vk::PipelineStageFlagBits2::eHost;
	before[0].srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eHostWrite;
	before[0].dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before[0].dstAccessMask =
	    load ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
	         : vk::AccessFlagBits2::eShaderRead;
	before[0].buffer        = tiled;
	before[0].offset        = tiled_offset;
	before[0].size          = tiled_capacity;
	before[1].srcStageMask  = vk::PipelineStageFlagBits2::eHost;
	before[1].srcAccessMask = vk::AccessFlagBits2::eHostWrite;
	before[1].dstStageMask  = vk::PipelineStageFlagBits2::eComputeShader;
	before[1].dstAccessMask = vk::AccessFlagBits2::eUniformRead;
	before[1].buffer        = m_stream_buffer.Handle();
	before[1].offset        = dispatches.front().params_offset;
	before[1].size = dispatches.back().params_offset - dispatches.front().params_offset + sizeof(Push);
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = static_cast<uint32_t>(before.size());
	dependency.pBufferMemoryBarriers    = before.data();
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	command.pipelineBarrier2(dependency);

	const vk::DescriptorBufferInfo tiled_info {tiled, tiled_descriptor_offset, tiled_range};
	for (size_t index = 0; index < dispatches.size(); ++index) {
		const auto& dispatch = dispatches[index];
		const auto  view     = find_view(regions[index])->view;
		const vk::DescriptorImageInfo  image_info {nullptr, view, vk::ImageLayout::eGeneral};
		const vk::DescriptorBufferInfo params_info {m_stream_buffer.Handle(),
		                                            dispatch.params_offset, sizeof(Push)};
		std::array<vk::WriteDescriptorSet, 3> writes {};
		for (uint32_t binding = 0; binding < writes.size(); binding++) {
			writes[binding].dstBinding      = binding;
			writes[binding].descriptorCount = 1;
		}
		writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
		writes[0].pBufferInfo    = &tiled_info;
		writes[1].descriptorType = vk::DescriptorType::eStorageImage;
		writes[1].pImageInfo     = &image_info;
		writes[2].descriptorType = vk::DescriptorType::eUniformBuffer;
		writes[2].pBufferInfo    = &params_info;
		EXIT_IF(m_image_pipeline_layout == nullptr);
		m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute,
		                                      m_image_pipeline_layout, 0,
		                                      static_cast<uint32_t>(writes.size()), writes.data());
		m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, pipelines[index]);
		command.dispatch((dispatch.push.width + 7u) / 8u, (dispatch.push.height + 7u) / 8u,
		                 dispatch.push.depth);
	}

	if (load) {
		// After: as TileImage's tile pass, the written tiled bytes for every later access.
		vk::BufferMemoryBarrier2 after = before[0];
		after.srcStageMask             = vk::PipelineStageFlagBits2::eComputeShader;
		after.srcAccessMask            = vk::AccessFlagBits2::eShaderWrite;
		after.dstStageMask             = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eMemoryRead;
		vk::DependencyInfo after_dependency {};
		after_dependency.bufferMemoryBarrierCount = 1;
		after_dependency.pBufferMemoryBarriers    = &after;
		command.pipelineBarrier2(after_dependency);
		return;
	}
	// After: the image leaves in the state Image::Upload leaves it in.
	image.NoteContentWrite();
	image.Transit(vk::ImageLayout::eGeneral,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {},
	              command);
}

bool TileManager::DetileToImage(Image& image, vk::Buffer tiled, uint64_t tiled_offset,
                                uint64_t tiled_capacity, uint64_t linear_capacity,
                                std::span<const GpuTileInfo>         infos,
                                std::span<const vk::BufferImageCopy> regions) {
	if (!ImageTransferEligible(image, false, infos, regions)) {
		return false;
	}
	KYTY_GPU_OP_SITE("tiler.detile_image");
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	std::vector<Dispatch> dispatches;
	Prepare(false, tiled_capacity, linear_capacity, infos, tiled_offset & (descriptor_alignment - 1),
	        0, dispatches, regions, image.info.IsBlock() ? 4u : 1u, ImageLayerViews(image));
	RecordImage(image, false, tiled, tiled_offset, tiled_capacity, infos, regions, dispatches);
	uint64_t bytes = 0;
	for (const auto& info: infos) {
		bytes += static_cast<uint64_t>(info.width) * info.height * info.bytes_per_element;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageUploads);
	Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageUploadBytes, bytes);
	if (image.info.IsBlock()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageBlockUploads);
		Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageBlockUploadBytes, bytes);
	}
	if (ImageDirectVerifyEnabled()) {
		// Expected: the linear bytes the buffer path would copy into the image. Actual: the image
		// read back with the same copies. Only the texel rows of each region are defined.
		const auto expected = Detile(tiled, tiled_offset, tiled_capacity, linear_capacity, infos);
		auto       actual   = AllocateScratch(Common::AlignUp(linear_capacity, 4));
		image.Download(regions, actual.buffer, 0, actual.size);
		std::vector<std::pair<uint64_t, uint64_t>> ranges;
		const uint32_t texel = image.info.IsBlock() ? 4u : 1u;
		for (size_t index = 0; index < regions.size(); ++index) {
			const auto&    region  = regions[index];
			const uint64_t element = infos[index].bytes_per_element;
			const uint64_t row_texels =
			    region.bufferRowLength != 0 ? region.bufferRowLength : region.imageExtent.width;
			const uint64_t pitch  = (row_texels + texel - 1u) / texel * element;
			const uint64_t width  = (region.imageExtent.width + texel - 1u) / texel;
			const uint64_t height = (region.imageExtent.height + texel - 1u) / texel;
			for (uint64_t y = 0; y < height; ++y) {
				ranges.emplace_back(region.bufferOffset + y * pitch, width * element);
			}
		}
		VerifyOnCompletion("upload", image.info.data.address, expected.buffer, expected.offset,
		                   actual.buffer, 0, linear_capacity, std::move(ranges));
		// Released with the tick of its last use (the readback copy).
		DeferDestroy(actual);
	}
	return true;
}

bool TileManager::TileFromImage(Image& image, std::span<const vk::BufferImageCopy> regions,
                                vk::Buffer tiled, uint64_t tiled_offset, uint64_t tiled_capacity,
                                uint64_t linear_capacity, std::span<const GpuTileInfo> infos) {
	// The tile pass rewrites whole dwords of the tiled buffer (sub-dword elements merge with
	// atomics), exactly as TileImage's buffer tile pass does.
	if (tiled_offset % 4 != 0 || tiled_capacity % 4 != 0 ||
	    !ImageTransferEligible(image, true, infos, regions)) {
		return false;
	}
	KYTY_GPU_OP_SITE("tiler.tile_from_image");
	const auto&    limits = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_alignment =
	    std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const bool verify = ImageDirectVerifyEnabled();
	Scratch    reference {};
	if (verify) {
		// Expected: the buffer path (TileImage) applied to a copy of the destination bytes taken
		// before the direct pass, so bytes no element covers compare equal too.
		reference = AllocateScratch(tiled_capacity);
		m_scheduler.EndRendering();
		auto                     command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
		before.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = tiled;
		before.offset              = tiled_offset;
		before.size                = tiled_capacity;
		vk::DependencyInfo dependency {};
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy snapshot {tiled_offset, 0, tiled_capacity};
		command.copyBuffer(tiled, reference.buffer, 1, &snapshot);
		vk::MemoryBarrier2 copied {};
		copied.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
		copied.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		copied.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		copied.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		vk::DependencyInfo copied_dependency {};
		copied_dependency.memoryBarrierCount = 1;
		copied_dependency.pMemoryBarriers    = &copied;
		command.pipelineBarrier2(copied_dependency);
		TileImage(image, regions, reference.buffer, 0, tiled_capacity, linear_capacity, infos);
	}
	std::vector<Dispatch> dispatches;
	Prepare(true, tiled_capacity, linear_capacity, infos, 0,
	        tiled_offset & (descriptor_alignment - 1), dispatches, regions,
	        image.info.IsBlock() ? 4u : 1u, ImageLayerViews(image));
	RecordImage(image, true, tiled, tiled_offset, tiled_capacity, infos, regions, dispatches);
	uint64_t bytes = 0;
	for (const auto& info: infos) {
		bytes += static_cast<uint64_t>(info.width) * info.height * info.bytes_per_element;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageDownloads);
	Profiler::CountFrameEvent(Profiler::FrameEvent::TilerImageDownloadBytes, bytes);
	if (verify) {
		VerifyOnCompletion("download", image.info.data.address, reference.buffer, 0, tiled,
		                   tiled_offset, tiled_capacity, {{0, tiled_capacity}});
		// Released with the tick of its last use: the stream parameters reserved in between may
		// have submitted the tick that took the snapshot.
		DeferDestroy(reference);
	}
	return true;
}

TileManager::Result TileManager::GetScratchBuffer(uint64_t size) {
	auto scratch = AllocateScratch(Common::AlignUp(size, 4));
	DeferDestroy(scratch);
	return {scratch.buffer, 0, scratch.size};
}

TileManager::StorageBinding TileManager::BindStorage(Result buffer, uint64_t size) const {
	const auto& limits            = m_graphics.GetPhysicalDeviceProperties().limits;
	const auto  alignment         = std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const auto  descriptor_offset = Common::AlignDown(buffer.offset, alignment);
	const auto  base              = buffer.offset - descriptor_offset;
	EXIT_IF(buffer.buffer == nullptr || size == 0 || buffer.size < size || base > UINT32_MAX ||
	        size > UINT64_MAX - base || base + size > UINT64_MAX - 3);
	const auto range = Common::AlignUp(base + size, 4);
	EXIT_IF(range > limits.maxStorageBufferRange || range > UINT32_MAX);
	return {{buffer.buffer, descriptor_offset, range}, static_cast<uint32_t>(base)};
}

uint32_t TileManager::ConversionRows(uint64_t offset, uint64_t row_stride, uint64_t active,
                                     uint32_t remaining, uint64_t alignment, uint64_t max_range,
                                     uint32_t max_groups) noexcept {
	if (row_stride == 0 || active == 0 || remaining == 0 || alignment == 0 || max_groups == 0) {
		return 0;
	}
	const auto prefix = offset % alignment;
	if (prefix >= max_range || active > max_range - prefix) {
		return 0;
	}
	const auto descriptor_rows = 1 + (max_range - prefix - active) / row_stride;
	return static_cast<uint32_t>(std::min<uint64_t>({remaining, descriptor_rows, max_groups}));
}

void TileManager::ConvertD16(Result source, Result target, D16Direction direction, bool d32,
                             const D16Layout& layout) {
	KYTY_GPU_OP_SITE("tiler.convert_d16");
	vk::Pipeline* pipeline_pointer = nullptr;
	if (direction == D16Direction::Promote) {
		pipeline_pointer = d32 ? &m_d16_to_d32 : &m_d16_to_d24;
	} else {
		pipeline_pointer = d32 ? &m_d32_to_d16 : &m_d24_to_d16;
	}
	auto& pipeline = *pipeline_pointer;
	if (pipeline == nullptr) {
		const uint32_t                   value = d32 ? 1u : 0u;
		const vk::SpecializationMapEntry entry {0, 0, sizeof(value)};
		const vk::SpecializationInfo     specialization {1, &entry, sizeof(value), &value};
		const uint32_t*                  code  = nullptr;
		size_t                           words = 0;
		if (direction == D16Direction::Promote) {
			code  = GPU_TILER_PROMOTE_D16_SPV;
			words = std::size(GPU_TILER_PROMOTE_D16_SPV);
		} else {
			code  = GPU_TILER_DEMOTE_D16_SPV;
			words = std::size(GPU_TILER_DEMOTE_D16_SPV);
		}
		const auto module = CompileSPV({code, words}, m_graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage               = vk::ShaderStageFlagBits::eCompute;
		stage.module              = module;
		stage.pName               = "main";
		stage.pSpecializationInfo = &specialization;
		vk::ComputePipelineCreateInfo create {};
		create.stage  = stage;
		create.layout = m_pipeline_layout;
		const auto result =
		    m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create D16 conversion pipeline");
	}

	const uint64_t source_element =
	    direction == D16Direction::Promote ? sizeof(uint16_t) : sizeof(uint32_t);
	const uint64_t target_element =
	    direction == D16Direction::Promote ? sizeof(uint32_t) : sizeof(uint16_t);
	const uint64_t source_active = static_cast<uint64_t>(layout.width) * source_element;
	const uint64_t target_active = static_cast<uint64_t>(layout.width) * target_element;
	const auto     required      = [](uint32_t height, uint32_t layers, uint64_t row_stride,
	                                  uint64_t slice_stride, uint64_t active) {
		EXIT_IF(height == 0 || layers == 0 || row_stride < active ||
		        (height - 1) > (UINT64_MAX - active) / row_stride);
		const auto slice = static_cast<uint64_t>(height - 1) * row_stride + active;
		EXIT_IF(slice_stride < slice || (layers - 1) > (UINT64_MAX - slice) / slice_stride);
		return static_cast<uint64_t>(layers - 1) * slice_stride + slice;
	};
	EXIT_IF(layout.width == 0 || layout.source_row_stride > UINT32_MAX ||
	        layout.target_row_stride > UINT32_MAX);
	const auto source_required = required(layout.height, layout.layers, layout.source_row_stride,
	                                      layout.source_slice_stride, source_active);
	const auto target_required = required(layout.height, layout.layers, layout.target_row_stride,
	                                      layout.target_slice_stride, target_active);
	EXIT_IF(source_required > UINT64_MAX - 3 || target_required > UINT64_MAX - 3);
	const auto source_barrier_size = Common::AlignUp(source_required, 4);
	const auto target_barrier_size = Common::AlignUp(target_required, 4);
	EXIT_IF(source.size < source_barrier_size || target.size < target_barrier_size);

	m_scheduler.EndRendering();
	auto                    command = m_scheduler.Current().Handle();
	vk::BufferMemoryBarrier barriers[2] {};
	barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[0].buffer              = source.buffer;
	barriers[0].offset              = source.offset;
	barriers[0].size                = source_barrier_size;
	barriers[0].srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite |
	                            vk::AccessFlagBits::eTransferWrite |
	                            vk::AccessFlagBits::eShaderWrite;
	barriers[0].dstAccessMask = vk::AccessFlagBits::eShaderRead;
	barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barriers[1].buffer              = target.buffer;
	barriers[1].offset              = target.offset;
	barriers[1].size                = target_barrier_size;
	barriers[1].srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	                            vk::AccessFlagBits::eHostWrite |
	                            vk::AccessFlagBits::eTransferWrite |
	                            vk::AccessFlagBits::eShaderWrite;
	barriers[1].dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	    vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 2, barriers, 0, nullptr);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	const auto& limits              = m_graphics.GetPhysicalDeviceProperties().limits;
	const auto descriptor_alignment = std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const auto rows_for = [&](Result buffer, uint64_t relative, uint64_t stride, uint64_t active,
	                          uint32_t remaining) {
		EXIT_IF(relative > buffer.size || buffer.offset > UINT64_MAX - relative);
		const auto offset = buffer.offset + relative;
		return ConversionRows(offset, stride, active, remaining, descriptor_alignment,
		                      limits.maxStorageBufferRange, limits.maxComputeWorkGroupCount[1]);
	};
	const auto groups_x = (static_cast<uint64_t>(layout.width) + 63u) / 64u;
	EXIT_IF(groups_x == 0 || groups_x > limits.maxComputeWorkGroupCount[0]);
	for (uint32_t layer = 0; layer < layout.layers; layer++) {
		for (uint32_t row = 0; row < layout.height;) {
			const auto source_relative =
			    layout.source_slice_stride * layer + layout.source_row_stride * row;
			const auto target_relative =
			    layout.target_slice_stride * layer + layout.target_row_stride * row;
			const auto remaining = layout.height - row;
			const auto rows = std::min(rows_for(source, source_relative, layout.source_row_stride,
			                                    source_active, remaining),
			                           rows_for(target, target_relative, layout.target_row_stride,
			                                    target_active, remaining));
			EXIT_IF(rows == 0);
			const auto source_span =
			    static_cast<uint64_t>(rows - 1) * layout.source_row_stride + source_active;
			const auto target_span =
			    static_cast<uint64_t>(rows - 1) * layout.target_row_stride + target_active;
			const auto source_binding = BindStorage(
			    {source.buffer, source.offset + source_relative, source.size - source_relative},
			    source_span);
			const auto target_binding = BindStorage(
			    {target.buffer, target.offset + target_relative, target.size - target_relative},
			    target_span);
			const vk::DescriptorBufferInfo infos[] {
			    source_binding.info,
			    target_binding.info,
			};
			std::array<vk::WriteDescriptorSet, 2> writes {};
			for (uint32_t index = 0; index < writes.size(); index++) {
				writes[index].dstBinding      = index;
				writes[index].descriptorCount = 1;
				writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
				writes[index].pBufferInfo     = &infos[index];
			}
			m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
			                             static_cast<uint32_t>(writes.size()), writes.data());
			Push push {};
			push.src_base    = source_binding.base;
			push.dst_base    = target_binding.base;
			push.width       = layout.width;
			push.height      = rows;
			push.pitch_bytes = static_cast<uint32_t>(layout.source_row_stride);
			push.slice_bytes = static_cast<uint32_t>(layout.target_row_stride);
			command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
			                      sizeof(push), &push);
			command.dispatch(static_cast<uint32_t>(groups_x), rows, 1);
			row += rows;
		}
	}
	barriers[1].srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barriers[1].dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eMemoryRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1,
	                        &barriers[1], 0, nullptr);
}

void TileManager::SwapBgra16(Result input, Result output, uint32_t pixels) {
	KYTY_GPU_OP_SITE("tiler.swap_bgra16");
	if (m_swap_bgra16 == nullptr) {
		const auto module = CompileSPV(GPU_TILER_SWAP_BGRA16_SPV, m_graphics.device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo create {};
		create.stage  = stage;
		create.layout = m_pipeline_layout;
		const auto result =
		    m_graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &m_swap_bgra16);
		m_graphics.device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create BGRA16 swap pipeline");
	}
	const uint64_t bytes = static_cast<uint64_t>(pixels) * 8u;
	EXIT_IF(pixels == 0);
	const auto input_binding  = BindStorage(input, bytes);
	const auto output_binding = BindStorage(output, bytes);

	const vk::DescriptorBufferInfo infos[] {
	    input_binding.info,
	    output_binding.info,
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); index++) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &infos[index];
	}
	vk::BufferMemoryBarrier barriers[2] {};
	for (uint32_t index = 0; index < 2; index++) {
		barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barriers[index].buffer              = infos[index].buffer;
		barriers[index].offset              = infos[index].offset;
		barriers[index].size                = infos[index].range;
	}
	barriers[0].srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite |
	                            vk::AccessFlagBits::eShaderWrite;
	barriers[0].dstAccessMask = vk::AccessFlagBits::eShaderRead;
	barriers[1].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
	barriers[1].dstAccessMask = vk::AccessFlagBits::eShaderWrite;
	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	    vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 2, barriers, 0, nullptr);
	m_scheduler.Current().BindPipeline(vk::PipelineBindPoint::eCompute, m_swap_bgra16);
	m_scheduler.Current().PushDescriptors(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	Push push {};
	push.src_base = input_binding.base;
	push.dst_base = output_binding.base;
	push.width    = pixels;
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
	                      &push);
	command.dispatch((pixels + 63u) / 64u, 1, 1);
	barriers[1].srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barriers[1].dstAccessMask = vk::AccessFlagBits::eTransferRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &barriers[1],
	                        0, nullptr);
}

TileManager::Result TileManager::SwapBgra16(Result input) {
	EXIT_NOT_IMPLEMENTED(input.size == 0 || input.size % 8u != 0 || input.size / 8u > UINT32_MAX);
	auto output = AllocateScratch(input.size);
	DeferDestroy(output);
	Result result {output.buffer, 0, output.size};
	SwapBgra16(input, result, static_cast<uint32_t>(input.size / 8u));
	return result;
}

void TileManager::SwapBgra16(Result input, Result output) {
	EXIT_NOT_IMPLEMENTED(input.size == 0 || input.size % 8u != 0 || input.size / 8u > UINT32_MAX ||
	                     output.size < input.size);
	SwapBgra16(input, output, static_cast<uint32_t>(input.size / 8u));
}

} // namespace Libs::Graphics
