#include "graphics/host_gpu/renderer/nativeRayTracing.h"

#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "gpu_tiler_shaders/rt_blas_walk_spv.h"
#include "gpu_tiler_shaders/rt_tlas_walk_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <span>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef near
#undef far
#undef MemoryBarrier
#endif

namespace Libs::Graphics {

namespace {

// KYTY_RT_HW_NATIVE=0 (live, default 1): the patched shaders run their guest loops (the context is
// invalid), for in-run A/B comparisons with the same pipelines.
Live::Switch g_native("KYTY_RT_HW_NATIVE", Live::ParseDefaultOn);
Live::Switch g_once_per_frame("KYTY_RT_HW_ONCE_PER_FRAME", Live::ParseDefaultOn);

constexpr uint32_t QueueCapacity    = 1u << 16u;
constexpr uint32_t InstanceCapacity = 1u << 16u;
constexpr uint32_t TableEntries     = 1u << 17u;
constexpr uint32_t SlotCapacity     = 1u << 15u;
constexpr uint32_t MissingCapacity  = 256;
constexpr uint32_t MaxJobs          = 512;
constexpr uint32_t ChangedCapacity  = 1024;
constexpr uint32_t PoolCapacity     = 1u << 24u;
constexpr uint64_t PoolQueueMax     = uint64_t {1} << 20u;
constexpr uint64_t BuildScratchSize = uint64_t {64} << 20u;
constexpr uint32_t MaxRetries       = 8;
constexpr uint64_t LateRetryPrepares = 64;
constexpr uint32_t DynamicPrepares  = 120;
constexpr uint64_t EvictFrames      = 1800;
constexpr uint64_t EvictScanPrepares = 8;
constexpr uint32_t MaxEvictions     = 256;
constexpr uint64_t MaxSlotTriangles = uint64_t {1} << 21u;
constexpr uint64_t MaxSlotBytes     = uint64_t {256} << 20u;
constexpr uint64_t LruMinAge        = 60;
constexpr uint64_t BudgetMin        = uint64_t {512} << 20u;
constexpr uint64_t BudgetMax        = uint64_t {4096} << 20u;
constexpr size_t   MaxRetryKeys     = 4096;
constexpr uint64_t TableTombstone   = 1;
// Gathered builds of a slot before it walks the guest BVH again.
constexpr uint32_t MaxGathers = 30;
// MODE_VERIFY reports a changed slot with this bit when its leaf count changed (not a refit).
constexpr uint32_t ChangedTopology = 0x80000000u;

constexpr uint32_t StateSize      = 96;
constexpr uint32_t MissingStride  = 48;
constexpr uint32_t JobStride      = 80;
constexpr uint32_t SlotStride     = 64;
constexpr uint32_t TableStride    = 16;
constexpr uint32_t ContextStride  = 64;
constexpr uint32_t ContextBytes   = NativeRayTracing::CountersOffset + NativeRayTracing::CounterCount * 4;
constexpr uint32_t InstanceStride = 64;

constexpr uint32_t ModeCount  = 0;
constexpr uint32_t ModeBuild  = 1;
constexpr uint32_t ModeVerify = 2;
constexpr uint32_t ModeGather = 3;

// RtState (rt_common.inc).
struct RtState {
	uint64_t header_va;
	uint64_t instance_base;
	uint32_t node_limit;
	uint32_t instance_count;
	uint32_t box_count;
	uint32_t overflow;
	uint32_t missing_count;
	uint32_t dispatch[3];
	uint32_t unknown_blas;
	uint32_t inactive;
	uint32_t max_depth;
	uint32_t verify_count;
	uint32_t verify[3];
	uint32_t changed_count;
	uint32_t pool_cursor;
	uint32_t pool_overflow;
	uint32_t pad[2];
};
static_assert(sizeof(RtState) == StateSize);
constexpr uint32_t StateDispatch = offsetof(RtState, dispatch);
constexpr uint32_t StateVerify   = offsetof(RtState, verify);

// The readback area: the state, the missing BLASes, the build jobs with their results, the slots
// whose BLAS changed, the slots the TLAS referenced, the verify counters.
constexpr uint64_t ReadbackMissing  = StateSize;
constexpr uint64_t ReadbackJobs     = ReadbackMissing + uint64_t {MissingCapacity} * MissingStride;
constexpr uint64_t ReadbackChanged  = ReadbackJobs + uint64_t {MaxJobs} * JobStride;
constexpr uint64_t ReadbackUsed     = ReadbackChanged + uint64_t {ChangedCapacity} * 4;
constexpr uint64_t ReadbackCounters = ReadbackUsed + uint64_t {SlotCapacity} * 4;
constexpr uint64_t ReadbackSize     = ReadbackCounters + NativeRayTracing::CounterCount * 4;

struct TlasPush {
	uint64_t user_ptr_va;
	uint64_t state;
	uint64_t queue;
	uint64_t instance_nodes;
	uint64_t instances;
	uint64_t table;
	uint64_t slots;
	uint64_t missing;
	uint64_t context;
	uint64_t tlas_address;
	uint64_t verify_list;
	uint64_t header_hint;
	uint32_t vsharp_offset;
	uint32_t header_offset;
	uint32_t root;
	uint32_t queue_capacity;
	uint32_t instance_capacity;
	uint32_t table_mask;
	uint32_t missing_capacity;
	uint32_t prepare_id;
	uint32_t verify;
	uint32_t flip_parity;
};

struct BlasPush {
	uint64_t jobs;
	uint64_t pool;
	uint64_t slots;
	uint64_t state;
	uint64_t changed;
	uint32_t mode;
	uint32_t pool_capacity;
	uint32_t job_count;
	uint32_t changed_capacity;
	uint32_t prepare_id;
	uint32_t pad;
};

struct Job {
	uint64_t blas_va;
	uint32_t root;
	uint32_t slot;
	uint64_t vertices0;
	uint64_t vertices1;
	uint64_t entries;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t result;
	uint32_t refit;
	uint64_t pad0;
	uint32_t pad1;
	uint32_t pad2;
	uint32_t leaves;
	uint32_t pad3;
};
static_assert(sizeof(Job) == JobStride);

struct SlotInfo {
	uint64_t as_address;
	uint64_t entry_table;
	uint32_t capacity0;
	uint32_t capacity1;
	uint32_t count0;
	uint32_t count1;
	uint32_t signature;
	uint32_t state;
	uint64_t blas_va;
	uint32_t root;
	uint32_t stamp;
	uint32_t leaf_count;
	uint32_t rebase;
};
static_assert(sizeof(SlotInfo) == SlotStride);

struct MissingEntry {
	uint64_t blas_va;
	uint32_t root;
	uint32_t table_index;
	uint32_t leaf_count;
	uint32_t box_count;
	uint32_t opaque_count;
	uint32_t alpha_count;
	uint32_t node_limit;
	uint32_t flags;
	uint32_t pad[2];
};
static_assert(sizeof(MissingEntry) == MissingStride);

constexpr vk::BufferUsageFlags DataFlags =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress |
    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eIndirectBuffer;
constexpr vk::BufferUsageFlags AsFlags = DataFlags | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                                         vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

constexpr vk::PipelineStageFlags2 AsBuildStage = vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR;

void MemoryBarrier(vk::CommandBuffer command, vk::PipelineStageFlags2 src_stages, vk::AccessFlags2 src_access,
                   vk::PipelineStageFlags2 dst_stages, vk::AccessFlags2 dst_access) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = src_stages;
	barrier.srcAccessMask = src_access;
	barrier.dstStageMask  = dst_stages;
	barrier.dstAccessMask = dst_access;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// vkCmdUpdateBuffer of a small array (at most 64 KiB, dword multiples).
void UpdateBuffer(vk::CommandBuffer command, vk::Buffer buffer, uint64_t offset, const void* data, uint32_t size) {
	EXIT_IF(size == 0 || size > 65536 || (size & 3u) != 0 || (offset & 3u) != 0);
	command.updateBuffer(buffer, offset, size, data);
}

void UpdateLarge(vk::CommandBuffer command, vk::Buffer buffer, uint64_t offset, const void* data, uint64_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	for (uint64_t done = 0; done < size;) {
		const auto chunk = static_cast<uint32_t>(std::min<uint64_t>(size - done, 65536));
		UpdateBuffer(command, buffer, offset + done, bytes + done, chunk);
		done += chunk;
	}
}

// BLASes the guest changed lately are rebuilt before every use: a fast build. Once they stop
// changing they are built once more for tracing.
constexpr vk::BuildAccelerationStructureFlagsKHR BlasFlags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
constexpr vk::BuildAccelerationStructureFlagsKHR FastBlasFlags =
    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;

// Non-indexed triangles (3 floats per vertex): geometry 0 opaque, geometry 1 alpha tested. The
// patched shaders trace with gl_RayFlagsNoOpaqueEXT, so both are candidates; the split keeps the
// entry table layout of the original.
vk::AccelerationStructureGeometryKHR TriangleGeometry(uint64_t address, uint32_t capacity, bool opaque) {
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
	geometry.flags        = opaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation;
	auto& triangles                    = geometry.geometry.triangles;
	triangles.sType                    = vk::StructureType::eAccelerationStructureGeometryTrianglesDataKHR;
	triangles.vertexFormat             = vk::Format::eR32G32B32Sfloat;
	triangles.vertexData.deviceAddress = address;
	triangles.vertexStride             = 12;
	triangles.maxVertex                = capacity * 3 - 1;
	triangles.indexType                = vk::IndexType::eNoneKHR;
	return geometry;
}

} // namespace

bool NativeRayTracing::FindTlasSource(const ShaderRecompiler::IR::ShaderInfo& info, TlasSource& source) noexcept {
	if (info.ray_query_sites == 0) {
		return false;
	}
	source = {info.ray_query_user_sgpr, info.ray_query_vsharp_offset, info.ray_query_header_offset,
	          info.ray_query_root};
	return true;
}

NativeRayTracing::Allocation NativeRayTracing::Arena::Allocate(uint64_t size, uint64_t alignment) {
	EXIT_IF(alignment == 0 || alignment > 256 || (256 % alignment) != 0);
	size = (size + 255u) & ~uint64_t {255};
	const auto carve = [&](Chunk& chunk, uint64_t offset, uint64_t length) {
		chunk.free_ranges.erase(offset);
		if (length > size) {
			chunk.free_ranges.emplace(offset + size, length - size);
		}
		chunk.free_bytes -= size;
		m_used += size;
		return Allocation {chunk.buffer.get(), offset, chunk.buffer->BufferDeviceAddress() + offset, size};
	};
	if (size <= ChunkSize / 2) {
		Chunk*   best        = nullptr;
		uint64_t best_offset = 0;
		uint64_t best_length = 0;
		for (auto& chunk: m_chunks) {
			if (chunk->dedicated || chunk->free_bytes < size || (best != nullptr && chunk->free_bytes >= best->free_bytes)) {
				continue;
			}
			uint64_t offset = 0;
			uint64_t length = UINT64_MAX;
			for (const auto& [range_offset, range_length]: chunk->free_ranges) {
				if (range_length >= size && range_length < length) {
					offset = range_offset;
					length = range_length;
				}
			}
			if (length != UINT64_MAX) {
				best        = chunk.get();
				best_offset = offset;
				best_length = length;
			}
		}
		if (best != nullptr) {
			return carve(*best, best_offset, best_length);
		}
	}
	auto chunk       = std::make_unique<Chunk>();
	chunk->dedicated = size > ChunkSize / 2;
	chunk->buffer    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AsFlags,
	                                            chunk->dedicated ? size : ChunkSize);
	chunk->free_ranges.emplace(0, chunk->buffer->Size());
	chunk->free_bytes = chunk->buffer->Size();
	m_reserved += chunk->buffer->Size();
	m_chunks.push_back(std::move(chunk));
	return carve(*m_chunks.back(), 0, m_chunks.back()->buffer->Size());
}

void NativeRayTracing::Arena::Free(const Allocation& allocation) {
	if (allocation.buffer == nullptr) {
		return;
	}
	const auto it = std::find_if(m_chunks.begin(), m_chunks.end(),
	                             [&](const auto& chunk) { return chunk->buffer.get() == allocation.buffer; });
	EXIT_IF(it == m_chunks.end());
	auto& chunk  = **it;
	auto  offset = allocation.offset;
	auto  length = allocation.size;
	m_used -= length;
	chunk.free_bytes += length;
	auto next = chunk.free_ranges.lower_bound(offset);
	if (next != chunk.free_ranges.end() && offset + length == next->first) {
		length += next->second;
		next = chunk.free_ranges.erase(next);
	}
	if (next != chunk.free_ranges.begin()) {
		const auto previous = std::prev(next);
		if (previous->first + previous->second == offset) {
			offset = previous->first;
			length += previous->second;
			chunk.free_ranges.erase(previous);
		}
	}
	chunk.free_ranges.emplace(offset, length);
	if (chunk.free_bytes == chunk.buffer->Size()) {
		const auto empty = std::count_if(m_chunks.begin(), m_chunks.end(),
		                                 [](const auto& c) { return !c->dedicated && c->free_bytes == c->buffer->Size(); });
		if (chunk.dedicated || empty > 1) {
			m_reserved -= chunk.buffer->Size();
			m_chunks.erase(it);
		}
	}
}

NativeRayTracing::NativeRayTracing(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_buffer_cache(buffer_cache), m_arena(graphics, scheduler) {
	m_enabled = graphics.native_ray_tracing_enabled && ShaderRecompiler::GetCodegenOptions().rt_hw;
	if (!m_enabled) {
		return;
	}
	CreatePipelines();
	CreateBuffers();
	CreateTlas(InstanceCapacity);
	if (const char* limit = std::getenv("KYTY_RT_ARENA_MB"); limit != nullptr && limit[0] != 0) {
		m_budget = std::strtoull(limit, nullptr, 10) << 20u;
	} else {
		const auto& memory = m_graphics.physical_device_memory_properties;
		uint64_t    heap   = 0;
		for (uint32_t i = 0; i < memory.memoryHeapCount; i++) {
			if (memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
				heap = std::max<uint64_t>(heap, memory.memoryHeaps[i].size);
			}
		}
		m_budget = std::clamp(heap / 4, BudgetMin, BudgetMax);
	}
	if (const char* every = std::getenv("KYTY_RT_HW_CHECK_EVERY"); every != nullptr && every[0] != 0) {
		m_verify_every = std::max<uint32_t>(1u, static_cast<uint32_t>(std::strtoul(every, nullptr, 10)));
	}
	if (const char* every = std::getenv("KYTY_RT_HW_STATS"); every != nullptr && every[0] != 0) {
		m_report_every = std::strtoull(every, nullptr, 10);
	}
	LOGF("NativeRayTracing: enabled, BLAS memory budget %llu MiB, verify %d\n",
	     static_cast<unsigned long long>(m_budget >> 20u), ShaderRecompiler::GetCodegenOptions().rt_hw_verify ? 1 : 0);
}

NativeRayTracing::~NativeRayTracing() {
	if (!m_enabled) {
		return;
	}
	auto device = m_graphics.device;
	for (auto& slot: m_slot_list) {
		if (slot && slot->handle) {
			device.destroyAccelerationStructureKHR(slot->handle);
		}
	}
	for (auto& retired: m_retired) {
		if (retired.handle) {
			device.destroyAccelerationStructureKHR(retired.handle);
		}
	}
	if (m_tlas) {
		device.destroyAccelerationStructureKHR(m_tlas);
	}
	device.destroyPipeline(m_tlas_pipeline);
	device.destroyPipeline(m_blas_pipeline);
	device.destroyPipelineLayout(m_tlas_layout);
	device.destroyPipelineLayout(m_blas_layout);
	device.destroyDescriptorSetLayout(m_descriptor_layout);
}

void NativeRayTracing::CreatePipelines() {
	auto                                 device = m_graphics.device;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1,
	                                              vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	RequireVulkanSuccess(device.createDescriptorSetLayout(&layout_info, nullptr, &m_descriptor_layout),
	                     "create native ray tracing descriptor layout");
	const auto make = [&](std::span<const uint32_t> code, uint32_t push_size, vk::PipelineLayout& layout,
	                      vk::Pipeline& pipeline, const char* name) {
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, push_size};
		vk::PipelineLayoutCreateInfo info {};
		info.setLayoutCount         = 1;
		info.pSetLayouts            = &m_descriptor_layout;
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&info, nullptr, &layout),
		                     "create native ray tracing pipeline layout");
		const auto module = CompileSPV(code, device);
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage  = vk::ShaderStageFlagBits::eCompute;
		stage.module = module;
		stage.pName  = "main";
		vk::ComputePipelineCreateInfo pipeline_info {};
		pipeline_info.stage  = stage;
		pipeline_info.layout = layout;
		const auto result    = device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
		device.destroyShaderModule(module, nullptr);
		RequireVulkanSuccess(result, "create native ray tracing pipeline");
		SetVulkanObjectNameF(device, pipeline, "{}", name);
	};
	make(RT_TLAS_WALK_SPV, sizeof(TlasPush), m_tlas_layout, m_tlas_pipeline, "RT TLAS walk");
	make(RT_BLAS_WALK_SPV, sizeof(BlasPush), m_blas_layout, m_blas_pipeline, "RT BLAS walk");
}

void NativeRayTracing::CreateBuffers() {
	const auto make = [&](uint64_t size, vk::BufferUsageFlags flags, const char* name) {
		auto buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, flags, size);
		SetVulkanObjectNameF(m_graphics.device, buffer->Handle(), "{}", name);
		return buffer;
	};
	m_state          = make(StateSize, DataFlags, "RT state");
	m_queue          = make(uint64_t {QueueCapacity} * 4, DataFlags, "RT queue");
	m_instance_nodes = make(uint64_t {InstanceCapacity} * 4, DataFlags, "RT instance nodes");
	m_instances      = make(uint64_t {InstanceCapacity} * InstanceStride, AsFlags, "RT instances");
	m_table          = make(uint64_t {TableEntries} * TableStride, DataFlags, "RT BLAS table");
	m_slots          = make(uint64_t {SlotCapacity} * SlotStride, DataFlags, "RT slots");
	m_missing        = make(uint64_t {MissingCapacity} * MissingStride, DataFlags, "RT missing");
	m_context        = make(ContextBytes, DataFlags, "RT context");
	m_pool           = make(uint64_t {PoolCapacity} * 4, DataFlags, "RT walk pool");
	m_verify_list    = make(uint64_t {SlotCapacity} * 4, DataFlags, "RT verify list");
	m_changed        = make(uint64_t {ChangedCapacity} * 4, DataFlags, "RT changed");
	m_jobs           = make(uint64_t {MaxJobs} * JobStride, DataFlags, "RT jobs");
	m_build_scratch  = make(BuildScratchSize, AsFlags, "RT build scratch");
	for (auto& readback: m_readbacks) {
		readback.buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, ReadbackSize);
	}
	m_context_address = m_context->BufferDeviceAddress();
}

void NativeRayTracing::CreateTlas(uint32_t capacity) {
	auto                                 device = m_graphics.device;
	vk::AccelerationStructureGeometryKHR geometry {};
	geometry.geometryType                       = vk::GeometryTypeKHR::eInstances;
	geometry.geometry.instances.sType           = vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
	geometry.geometry.instances.arrayOfPointers = VK_FALSE;
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eTopLevel;
	info.flags         = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = 1;
	info.pGeometries   = &geometry;
	vk::AccelerationStructureBuildSizesInfoKHR sizes {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info, &capacity, &sizes);
	m_tlas_storage = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AsFlags,
	                                          sizes.accelerationStructureSize);
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = m_tlas_storage->Handle();
	create.size   = sizes.accelerationStructureSize;
	create.type   = vk::AccelerationStructureTypeKHR::eTopLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &m_tlas), "create native TLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = m_tlas;
	m_tlas_address      = device.getAccelerationStructureAddressKHR(address_info);
	m_tlas_scratch_size = sizes.buildScratchSize;
	m_tlas_capacity     = capacity;
	EXIT_IF(m_tlas_scratch_size > BuildScratchSize);
}

bool NativeRayTracing::ComputeSlotSizes(uint32_t opaque, uint32_t alpha, SlotSizes& sizes, bool exact) {
	auto           device    = m_graphics.device;
	const uint64_t capacity0 = exact ? opaque : uint64_t {opaque} + opaque / 8 + 16;
	const uint64_t capacity1 = exact ? alpha : uint64_t {alpha} + alpha / 8 + 16;
	if (capacity0 + capacity1 > MaxSlotTriangles) {
		return false;
	}
	sizes.capacity0 = static_cast<uint32_t>(capacity0);
	sizes.capacity1 = static_cast<uint32_t>(capacity1);
	const std::array geometries {TriangleGeometry(0, sizes.capacity0, true), TriangleGeometry(0, sizes.capacity1, false)};
	const std::array counts {sizes.capacity0, sizes.capacity1};
	vk::AccelerationStructureBuildGeometryInfoKHR info {};
	info.type          = vk::AccelerationStructureTypeKHR::eBottomLevel;
	info.flags         = BlasFlags;
	info.mode          = vk::BuildAccelerationStructureModeKHR::eBuild;
	info.geometryCount = static_cast<uint32_t>(geometries.size());
	info.pGeometries   = geometries.data();
	vk::AccelerationStructureBuildSizesInfoKHR build {};
	device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &info, counts.data(), &build);
	{
		auto fast_info  = info;
		fast_info.flags = FastBlasFlags;
		vk::AccelerationStructureBuildSizesInfoKHR fast {};
		device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &fast_info,
		                                             counts.data(), &fast);
		build.accelerationStructureSize = std::max(build.accelerationStructureSize, fast.accelerationStructureSize);
		build.buildScratchSize          = std::max(build.buildScratchSize, fast.buildScratchSize);
	}
	sizes.storage   = (build.accelerationStructureSize + 255u) & ~uint64_t {255};
	sizes.vertices0 = (uint64_t {sizes.capacity0} * 36 + 255u) & ~uint64_t {255};
	sizes.vertices1 = (uint64_t {sizes.capacity1} * 36 + 255u) & ~uint64_t {255};
	sizes.entries   = ((uint64_t {sizes.capacity0} + sizes.capacity1) * 16 + 255u) & ~uint64_t {255};
	sizes.scratch   = build.buildScratchSize;
	if (sizes.scratch > BuildScratchSize || sizes.Total() > MaxSlotBytes) {
		return false;
	}
	return m_budget == 0 || m_arena.Used() + sizes.Total() <= m_budget;
}

uint64_t NativeRayTracing::SlotBytes(const Slot& slot) noexcept {
	return slot.storage.size + slot.vertices0.size + slot.vertices1.size + slot.entries.size;
}

void NativeRayTracing::AllocateSlot(Slot& slot, const SlotSizes& sizes) {
	auto device        = m_graphics.device;
	slot.capacity0     = sizes.capacity0;
	slot.capacity1     = sizes.capacity1;
	slot.storage       = m_arena.Allocate(sizes.storage, 256);
	slot.vertices0     = m_arena.Allocate(sizes.vertices0, 256);
	slot.vertices1     = m_arena.Allocate(sizes.vertices1, 256);
	slot.entries       = m_arena.Allocate(sizes.entries, 256);
	slot.build_scratch = sizes.scratch;
	slot.built         = false;
	vk::AccelerationStructureCreateInfoKHR create {};
	create.buffer = slot.storage.buffer->Handle();
	create.offset = slot.storage.offset;
	create.size   = sizes.storage;
	create.type   = vk::AccelerationStructureTypeKHR::eBottomLevel;
	RequireVulkanSuccess(device.createAccelerationStructureKHR(&create, nullptr, &slot.handle), "create native BLAS");
	vk::AccelerationStructureDeviceAddressInfoKHR address_info {};
	address_info.accelerationStructure = slot.handle;
	slot.as_address                    = device.getAccelerationStructureAddressKHR(address_info);
}

bool NativeRayTracing::CreateSlot(const PendingSlot& pending) {
	SlotSizes sizes;
	if (!ComputeSlotSizes(pending.opaque, pending.alpha, sizes)) {
		return false;
	}
	uint32_t index = 0;
	if (!m_free_slots.empty()) {
		index = m_free_slots.back();
		m_free_slots.pop_back();
	} else if (m_slot_list.size() < SlotCapacity) {
		index = static_cast<uint32_t>(m_slot_list.size());
		m_slot_list.emplace_back();
	} else {
		return false;
	}
	auto slot            = std::make_unique<Slot>();
	slot->blas_va        = pending.blas_va;
	slot->root           = pending.root;
	slot->table_index    = pending.table_index;
	slot->leaf_count     = pending.leaf_count;
	slot->last_use_frame = m_frame;
	AllocateSlot(*slot, sizes);
	slot->needs_build = true;
	m_slots_by_blas[slot->blas_va].push_back(index);
	m_build_queue.push_back(index);
	m_slot_list[index] = std::move(slot);
	m_retries.erase(pending.blas_va ^ (uint64_t {pending.root} << 48u));
	return true;
}

void NativeRayTracing::RetireResources(Slot& slot, uint32_t slot_index) {
	Retired retired {};
	retired.tick        = m_scheduler.CurrentTick();
	retired.prepare_id  = m_prepare_id;
	retired.handle      = slot.handle;
	retired.allocations = {slot.storage, slot.vertices0, slot.vertices1, slot.entries};
	retired.slot        = slot_index;
	slot.handle         = nullptr;
	slot.as_address     = 0;
	slot.storage = slot.vertices0 = slot.vertices1 = slot.entries = {};
	for (const auto& allocation: retired.allocations) {
		m_retired_bytes += allocation.size;
	}
	m_retired.push_back(retired);
}

void NativeRayTracing::DestroyRetired() {
	auto device = m_graphics.device;
	while (!m_retired.empty()) {
		auto&      retired        = m_retired.front();
		const bool readbacks_done = retired.slot == UINT32_MAX || m_prepare_id > retired.prepare_id + m_readbacks.size();
		if (!readbacks_done || !m_scheduler.IsFree(retired.tick)) {
			break;
		}
		if (retired.handle) {
			device.destroyAccelerationStructureKHR(retired.handle);
		}
		for (const auto& allocation: retired.allocations) {
			m_retired_bytes -= allocation.size;
			m_arena.Free(allocation);
		}
		if (retired.slot != UINT32_MAX) {
			m_slot_list[retired.slot].reset();
			m_free_slots.push_back(retired.slot);
		}
		m_retired.pop_front();
	}
}

void NativeRayTracing::EvictSlot(uint32_t index) {
	auto& slot       = *m_slot_list[index];
	slot.evicted     = true;
	slot.needs_build = false;
	if (const auto it = m_slots_by_blas.find(slot.blas_va); it != m_slots_by_blas.end()) {
		std::erase(it->second, index);
		if (it->second.empty()) {
			m_slots_by_blas.erase(it);
		}
	}
	RetireResources(slot, index);
}

void NativeRayTracing::EvictUnused(std::vector<uint32_t>& evicted) {
	if (m_frame > EvictFrames) {
		const auto limit = m_frame - EvictFrames;
		for (uint32_t index = 0; index < m_slot_list.size() && evicted.size() < MaxEvictions; index++) {
			auto* slot = m_slot_list[index].get();
			if (slot == nullptr || slot->evicted || slot->last_use_frame >= limit) {
				continue;
			}
			EvictSlot(index);
			evicted.push_back(index);
		}
	}
	EvictOverBudget(evicted);
}

void NativeRayTracing::EvictOverBudget(std::vector<uint32_t>& evicted) {
	const auto live = m_arena.Used() - m_retired_bytes;
	if (m_budget == 0 || live <= m_budget / 8 * 7 || m_frame <= LruMinAge) {
		return;
	}
	const auto                                 limit = m_frame - LruMinAge;
	std::vector<std::pair<uint64_t, uint32_t>> candidates;
	for (uint32_t index = 0; index < m_slot_list.size(); index++) {
		const auto* slot = m_slot_list[index].get();
		if (slot != nullptr && !slot->evicted && slot->last_use_frame < limit) {
			candidates.push_back({slot->last_use_frame, index});
		}
	}
	std::sort(candidates.begin(), candidates.end());
	auto current = live;
	for (const auto& [frame, index]: candidates) {
		if (current <= m_budget / 4 * 3 || evicted.size() >= MaxEvictions) {
			break;
		}
		current -= std::min(current, SlotBytes(*m_slot_list[index]));
		EvictSlot(index);
		evicted.push_back(index);
	}
}

void NativeRayTracing::ReportAgain(uint64_t blas_va, uint32_t root, uint32_t table_index) {
	if (m_retries.size() >= MaxRetryKeys) {
		for (const auto& [key, state]: m_retries) {
			if (state.count > MaxRetries) {
				m_retry.push_back(state.table_index);
			}
		}
		m_retries.clear();
	}
	auto& state       = m_retries[blas_va ^ (uint64_t {root} << 48u)];
	state.table_index = table_index;
	if (++state.count <= MaxRetries) {
		m_retry.push_back(table_index);
	} else {
		m_late_retries.push_back({m_prepare_id + LateRetryPrepares, table_index});
	}
}

void NativeRayTracing::RecordBuilds(vk::CommandBuffer command) {
	auto&                 readback = m_readbacks[m_readback_index];
	std::vector<Job>      jobs;
	std::vector<uint32_t> slots;
	std::vector<bool>     gathered;
	uint64_t              scratch  = 0;
	uint64_t              pool_use = 0;
	constexpr uint64_t    Alignment = 256;
	const auto            gathers  = [&](const Slot* slot) {
		return slot != nullptr && slot->built && slot->refit && slot->gathers < MaxGathers;
	};
	{
		Common::LockGuard lock(m_mutex);
		// The gathers (refit BLASes the frame traces: cheap, no walk) first, then the oldest builds;
		// the queue is served from its end.
		std::stable_sort(m_build_queue.begin(), m_build_queue.end(), [&](uint32_t a, uint32_t b) {
			const auto* slot_a = m_slot_list[a].get();
			const auto* slot_b = m_slot_list[b].get();
			if (gathers(slot_a) != gathers(slot_b)) {
				return !gathers(slot_a);
			}
			const auto age = [&](const Slot* slot) {
				return slot == nullptr || slot->built_frame == UINT64_MAX ? uint64_t {0} : slot->built_frame;
			};
			return age(slot_a) > age(slot_b);
		});
		std::vector<uint32_t> deferred;
		while (!m_build_queue.empty() && jobs.size() < MaxJobs) {
			const auto index = m_build_queue.back();
			if (m_slot_list[index] == nullptr || m_slot_list[index]->evicted || !m_slot_list[index]->needs_build) {
				m_build_queue.pop_back();
				continue;
			}
			auto&      slot = *m_slot_list[index];
			const auto size = (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
			if (scratch + size > BuildScratchSize) {
				break;
			}
			const bool     gather = gathers(&slot);
			const uint64_t queue  = gather ? 0
			                        : slot.leaf_count != 0
			                            ? std::min<uint64_t>(uint64_t {slot.leaf_count} + slot.leaf_count / 2 + 64, PoolQueueMax)
			                            : PoolQueueMax;
			m_build_queue.pop_back();
			if (pool_use != 0 && pool_use + queue > PoolCapacity / 2) {
				deferred.push_back(index);
				continue;
			}
			pool_use += queue;
			slot.needs_build = false;
			slot.gathers     = gather ? slot.gathers + 1 : 0;
			slot.refit       = false;
			scratch += size;
			const uint32_t refit = slot.dynamic_until != 0 ? 1u : 0u;
			Job            job {};
			job.blas_va   = slot.blas_va;
			job.root      = slot.root;
			job.slot      = index;
			job.vertices0 = slot.vertices0.address;
			job.vertices1 = slot.vertices1.address;
			job.entries   = slot.entries.address;
			job.capacity0 = slot.capacity0;
			job.capacity1 = slot.capacity1;
			job.refit     = refit;
			jobs.push_back(job);
			slots.push_back(index);
			gathered.push_back(gather);
		}
		m_build_queue.insert(m_build_queue.end(), deferred.rbegin(), deferred.rend());
	}
	if (jobs.empty()) {
		return;
	}
	// Walks first, then gathers.
	uint32_t walk_count = 0;
	{
		std::vector<Job>      walk_jobs, gather_jobs;
		std::vector<uint32_t> walk_slots, gather_slots;
		for (size_t i = 0; i < jobs.size(); i++) {
			(gathered[i] ? gather_jobs : walk_jobs).push_back(jobs[i]);
			(gathered[i] ? gather_slots : walk_slots).push_back(slots[i]);
		}
		walk_count = static_cast<uint32_t>(walk_jobs.size());
		m_stats.gathers += gather_jobs.size();
		m_stats.builds += walk_jobs.size();
		jobs       = std::move(walk_jobs);
		slots      = std::move(walk_slots);
		jobs.insert(jobs.end(), gather_jobs.begin(), gather_jobs.end());
		slots.insert(slots.end(), gather_slots.begin(), gather_slots.end());
	}
	// The slot infos (state 0 until built) and table entries (slot + 1) of the walked slots.
	for (uint32_t i = 0; i < walk_count; i++) {
		const auto  index = slots[i];
		const auto& slot  = *m_slot_list[index];
		SlotInfo    info {};
		info.as_address  = slot.as_address;
		info.entry_table = slot.entries.address;
		info.capacity0   = slot.capacity0;
		info.capacity1   = slot.capacity1;
		info.blas_va     = slot.blas_va;
		info.root        = slot.root;
		info.leaf_count  = slot.leaf_count;
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &info, sizeof(info));
		if (slot.table_index != UINT32_MAX) {
			const uint32_t encoded = index + 1;
			UpdateBuffer(command, m_table->Handle(), uint64_t {slot.table_index} * TableStride + 8, &encoded,
			             sizeof(encoded));
		}
	}
	UpdateLarge(command, m_jobs->Handle(), 0, jobs.data(), jobs.size() * JobStride);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	const BlasPush push {m_jobs->BufferDeviceAddress(),
	                     m_pool->BufferDeviceAddress(),
	                     m_slots->BufferDeviceAddress(),
	                     m_state->BufferDeviceAddress(),
	                     m_changed->BufferDeviceAddress(),
	                     ModeBuild,
	                     PoolCapacity,
	                     walk_count,
	                     ChangedCapacity,
	                     m_prepare_id,
	                     0};
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	if (walk_count != jobs.size()) {
		auto gather      = push;
		gather.jobs      = m_jobs->BufferDeviceAddress() + uint64_t {walk_count} * JobStride;
		gather.mode      = ModeGather;
		gather.job_count = static_cast<uint32_t>(jobs.size()) - walk_count;
		command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(gather), &gather);
		command.dispatch(gather.job_count, 1, 1);
		command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	}
	if (walk_count != 0) {
		command.dispatch(walk_count, 1, 1);
	}
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              AsBuildStage | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead);
	// One build per slot, all in one command (distinct scratch ranges).
	const auto                                                     count = static_cast<uint32_t>(slots.size());
	std::vector<vk::AccelerationStructureGeometryKHR>              geometries(count * 2);
	std::vector<vk::AccelerationStructureBuildGeometryInfoKHR>     infos(count);
	std::vector<vk::AccelerationStructureBuildRangeInfoKHR>        ranges(count * 2);
	std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers(count);
	uint64_t                                                       scratch_offset = 0;
	const auto                                                     scratch_base   = m_build_scratch->BufferDeviceAddress();
	for (uint32_t i = 0; i < count; i++) {
		auto&      slot = *m_slot_list[slots[i]];
		const bool fast = slot.dynamic_until != 0;
		slot.fast_built = fast;
		geometries[i * 2]     = TriangleGeometry(slot.vertices0.address, slot.capacity0, true);
		geometries[i * 2 + 1] = TriangleGeometry(slot.vertices1.address, slot.capacity1, false);
		auto& info                     = infos[i];
		info.type                      = vk::AccelerationStructureTypeKHR::eBottomLevel;
		info.flags                     = fast ? FastBlasFlags : BlasFlags;
		info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
		info.dstAccelerationStructure  = slot.handle;
		info.geometryCount             = 2;
		info.pGeometries               = &geometries[i * 2];
		info.scratchData.deviceAddress = scratch_base + scratch_offset;
		ranges[i * 2].primitiveCount     = slot.capacity0;
		ranges[i * 2 + 1].primitiveCount = slot.capacity1;
		range_pointers[i]                = &ranges[i * 2];
		scratch_offset += (slot.build_scratch + Alignment - 1) & ~(Alignment - 1);
	}
	command.buildAccelerationStructuresKHR(count, infos.data(), range_pointers.data());
	MemoryBarrier(command, AsBuildStage, vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
	              AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eTransferWrite);
	constexpr uint32_t built = 1;
	for (const auto index: slots) {
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride + offsetof(SlotInfo, state), &built,
		             sizeof(built));
		m_slot_list[index]->built       = true;
		m_slot_list[index]->built_frame = m_frame;
	}
	readback.build_count = static_cast<uint32_t>(slots.size());
	readback.build_slots = std::move(slots);
}

void NativeRayTracing::Prepare(vk::CommandBuffer command, uint64_t user_pointer, const TlasSource& source,
                               uint64_t frame, uint64_t header_hint) {
	if (!m_enabled) {
		return;
	}
	auto* page_table = m_buffer_cache.GetBdaPageTableBuffer();
	if (!m_cleared) {
		m_cleared = true;
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
		              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
		              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
		for (const auto* buffer: {m_table.get(), m_slots.get(), m_context.get()}) {
			command.fillBuffer(buffer->Handle(), 0, buffer->Size(), 0);
		}
		// Patched shaders find the context through the BDA page table.
		m_buffer_cache.EnsureBdaPageTableEntry(BufferCache::NATIVE_RT_CONTEXT_PAGE);
		UpdateBuffer(command, page_table->Handle(), BufferCache::NATIVE_RT_CONTEXT_PAGE * sizeof(uint64_t),
		             &m_context_address, sizeof(m_context_address));
		MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
		              vk::PipelineStageFlagBits2::eAllCommands,
		              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	}
	if (!g_native.On()) {
		// The shaders run their guest loops.
		if (m_context_valid) {
			m_context_valid = false;
			MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead,
			              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
			command.fillBuffer(m_context->Handle(), 0, ContextStride, 0);
			MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
			              vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryRead);
		}
		return;
	}
	// KYTY_RT_HW_ONCE_PER_FRAME (live, default 1): the TLAS of a header the host already prepared in
	// this frame (Miles' two RT kernels trace the same one) is used again.
	if (g_once_per_frame.On() && m_context_valid && header_hint != 0 && header_hint == m_prepared_header &&
	    frame == m_prepared_frame) {
		m_stats.reused++;
		return;
	}
	m_prepared_header = header_hint;
	m_prepared_frame  = frame;
	m_context_valid = true;
	auto& readback  = m_readbacks[m_readback_index];
	if (readback.tick != 0) {
		// The ring is full: the oldest readback must complete first (rare).
		m_scheduler.Wait(readback.tick);
		m_scheduler.PopPendingOperations();
	}
	readback.build_count = 0;
	readback.build_slots.clear();
	const auto source_key = source.user_sgpr << 16u | (source.vsharp_offset & 0xffffu);
	readback.source_key   = source_key;
	readback.frame        = frame;
	uint32_t                                   tlas_count = 0;
	std::vector<uint32_t>                      retry;
	std::vector<std::pair<uint32_t, uint32_t>> evicted;
	std::vector<std::pair<uint32_t, SlotInfo>> reallocated;
	std::vector<uint32_t>                      moved;
	{
		Common::LockGuard lock(m_mutex);
		m_frame = frame;
		DestroyRetired();
		for (const auto& pending: m_new_slots) {
			if (!CreateSlot(pending)) {
				ReportAgain(pending.blas_va, pending.root, pending.table_index);
			}
		}
		m_new_slots.clear();
		for (const auto& [index, opaque, alpha]: m_grow) {
			if (m_slot_list[index] == nullptr || m_slot_list[index]->evicted) {
				continue;
			}
			auto&     slot = *m_slot_list[index];
			SlotSizes sizes;
			if (!ComputeSlotSizes(opaque, alpha, sizes)) {
				continue;
			}
			RetireResources(slot, UINT32_MAX);
			AllocateSlot(slot, sizes);
			slot.leaf_count  = 0;
			slot.refit       = false;
			slot.needs_build = true;
			m_build_queue.push_back(index);
			moved.push_back(index);
		}
		m_grow.clear();
		if (m_prepare_id - m_last_scan_prepare >= EvictScanPrepares) {
			m_last_scan_prepare = m_prepare_id;
			std::vector<uint32_t> indices;
			EvictUnused(indices);
			for (const auto index: indices) {
				evicted.push_back({index, m_slot_list[index]->table_index});
			}
		}
		for (const auto index: moved) {
			const auto& slot = *m_slot_list[index];
			if (slot.evicted) {
				continue;
			}
			SlotInfo info {};
			info.as_address  = slot.as_address;
			info.entry_table = slot.entries.address;
			info.capacity0   = slot.capacity0;
			info.capacity1   = slot.capacity1;
			info.blas_va     = slot.blas_va;
			info.root        = slot.root;
			reallocated.push_back({index, info});
		}
		std::erase_if(m_late_retries, [&](const std::pair<uint64_t, uint32_t>& late) {
			if (late.first > m_prepare_id) {
				return false;
			}
			m_retry.push_back(late.second);
			return true;
		});
		retry.swap(m_retry);
		// Slots the guest changed lately are rebuilt before each use (c47094b5: before the RT
		// passes of the frame trace them, not after the verification of a later preparation).
		std::erase_if(m_dynamic, [&](uint32_t index) {
			auto* released = m_slot_list[index].get();
			if (released == nullptr || released->evicted) {
				return true;
			}
			auto& slot = *released;
			if (slot.dynamic_until <= m_prepare_id) {
				slot.dynamic_until = 0;
				if (slot.fast_built && !slot.needs_build) {
					slot.needs_build = true;
					slot.refit       = false;
					m_build_queue.push_back(index);
				}
				return true;
			}
			if (!slot.needs_build) {
				slot.needs_build = true;
				slot.refit       = slot.refit_change;
				m_build_queue.push_back(index);
			}
			return false;
		});
		const auto last = m_last_instance_counts[source_key];
		tlas_count      = std::min(m_tlas_capacity, std::max(1024u, (last + last / 4 + 255u) & ~255u));
	}
	m_prepare_id++;

	// Earlier guest work wrote the BVHs; earlier ray queries read the TLAS this rebuilds.
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer | AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
	const auto forget = [&](uint32_t table_index) {
		const std::array<uint64_t, 2> tombstone {TableTombstone, 0};
		UpdateBuffer(command, m_table->Handle(), uint64_t {table_index} * TableStride, tombstone.data(), sizeof(tombstone));
	};
	for (const auto index: retry) {
		forget(index);
	}
	for (const auto& [index, table_index]: evicted) {
		constexpr SlotInfo cleared {};
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &cleared, sizeof(cleared));
		if (table_index != UINT32_MAX) {
			forget(table_index);
		}
	}
	for (const auto& [index, info]: reallocated) {
		UpdateBuffer(command, m_slots->Handle(), uint64_t {index} * SlotStride, &info, sizeof(info));
	}
	command.fillBuffer(m_state->Handle(), 0, StateSize, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eTransferWrite);

	const vk::DescriptorBufferInfo page_info {page_table->Handle(), 0, page_table->Size()};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &page_info;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, 1, &write);

	RecordBuilds(command);

	command.fillBuffer(m_instances->Handle(), 0, uint64_t {tlas_count} * InstanceStride, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_tlas_layout, 0, 1, &write);
	TlasPush tlas_push {};
	tlas_push.user_ptr_va       = user_pointer;
	tlas_push.state             = m_state->BufferDeviceAddress();
	tlas_push.queue             = m_queue->BufferDeviceAddress();
	tlas_push.instance_nodes    = m_instance_nodes->BufferDeviceAddress();
	tlas_push.instances         = m_instances->BufferDeviceAddress();
	tlas_push.table             = m_table->BufferDeviceAddress();
	tlas_push.slots             = m_slots->BufferDeviceAddress();
	tlas_push.missing           = m_missing->BufferDeviceAddress();
	tlas_push.context           = m_context_address;
	tlas_push.tlas_address      = m_tlas_address;
	tlas_push.verify_list       = m_verify_list->BufferDeviceAddress();
	tlas_push.header_hint       = header_hint;
	m_stats.hinted += header_hint != 0 ? 1u : 0u;
	tlas_push.vsharp_offset     = source.vsharp_offset;
	tlas_push.header_offset     = source.header_offset;
	tlas_push.root              = source.root;
	tlas_push.queue_capacity    = QueueCapacity;
	tlas_push.instance_capacity = tlas_count;
	tlas_push.table_mask        = TableEntries - 1;
	tlas_push.missing_capacity  = MissingCapacity;
	tlas_push.prepare_id        = m_prepare_id;
	tlas_push.verify            = m_verify_every;
	// Every triangle is a candidate the guest leaf code tests for its facing: no host culling.
	tlas_push.flip_parity = 0u;
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_tlas_pipeline);
	command.pushConstants(m_tlas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(tlas_push), &tlas_push);
	command.dispatch(1, 1, 1);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eDrawIndirect | AsBuildStage,
	              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
	                  vk::AccessFlagBits2::eIndirectCommandRead);

	// Counts the leaves of the BLASes the walk found missing, and checks the built ones it uses.
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_blas_pipeline);
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_blas_layout, 0, 1, &write);
	BlasPush blas_push {m_missing->BufferDeviceAddress(),
	                    m_pool->BufferDeviceAddress(),
	                    m_slots->BufferDeviceAddress(),
	                    m_state->BufferDeviceAddress(),
	                    m_changed->BufferDeviceAddress(),
	                    ModeCount,
	                    PoolCapacity,
	                    0,
	                    ChangedCapacity,
	                    m_prepare_id,
	                    0};
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push), &blas_push);
	command.dispatchIndirect(m_state->Handle(), StateDispatch);
	blas_push.jobs = m_verify_list->BufferDeviceAddress();
	blas_push.mode = ModeVerify;
	command.pushConstants(m_blas_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blas_push), &blas_push);
	command.dispatchIndirect(m_state->Handle(), StateVerify);

	// The TLAS over the walk's instances (inactive ones have a null BLAS).
	{
		vk::AccelerationStructureGeometryKHR geometry {};
		geometry.geometryType = vk::GeometryTypeKHR::eInstances;
		geometry.geometry.instances.sType              = vk::StructureType::eAccelerationStructureGeometryInstancesDataKHR;
		geometry.geometry.instances.arrayOfPointers    = VK_FALSE;
		geometry.geometry.instances.data.deviceAddress = m_instances->BufferDeviceAddress();
		vk::AccelerationStructureBuildGeometryInfoKHR info {};
		info.type                      = vk::AccelerationStructureTypeKHR::eTopLevel;
		info.flags                     = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
		info.mode                      = vk::BuildAccelerationStructureModeKHR::eBuild;
		info.dstAccelerationStructure  = m_tlas;
		info.geometryCount             = 1;
		info.pGeometries               = &geometry;
		info.scratchData.deviceAddress = m_build_scratch->BufferDeviceAddress();
		vk::AccelerationStructureBuildRangeInfoKHR range {};
		range.primitiveCount      = tlas_count;
		const auto* range_pointer = &range;
		command.buildAccelerationStructuresKHR(1, &info, &range_pointer);
	}
	MemoryBarrier(command, AsBuildStage | vk::PipelineStageFlagBits2::eComputeShader,
	              vk::AccessFlagBits2::eAccelerationStructureWriteKHR | vk::AccessFlagBits2::eShaderWrite,
	              vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
	              vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead |
	                  vk::AccessFlagBits2::eTransferRead);

	// Reads back the state, the counted missing BLASes, the build results, the changed slots, the
	// used slots and the verify counters (then cleared).
	const std::array<vk::BufferCopy, 6> regions {{
	    {0, 0, StateSize},
	    {0, ReadbackMissing, uint64_t {MissingCapacity} * MissingStride},
	    {0, ReadbackJobs, uint64_t {MaxJobs} * JobStride},
	    {0, ReadbackChanged, uint64_t {ChangedCapacity} * 4},
	    {0, ReadbackUsed, uint64_t {SlotCapacity} * 4},
	    {CountersOffset, ReadbackCounters, uint64_t {CounterCount} * 4},
	}};
	command.copyBuffer(m_state->Handle(), readback.buffer->Handle(), 1, &regions[0]);
	command.copyBuffer(m_missing->Handle(), readback.buffer->Handle(), 1, &regions[1]);
	command.copyBuffer(m_jobs->Handle(), readback.buffer->Handle(), 1, &regions[2]);
	command.copyBuffer(m_changed->Handle(), readback.buffer->Handle(), 1, &regions[3]);
	command.copyBuffer(m_verify_list->Handle(), readback.buffer->Handle(), 1, &regions[4]);
	command.copyBuffer(m_context->Handle(), readback.buffer->Handle(), 1, &regions[5]);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead,
	              vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
	command.fillBuffer(m_context->Handle(), CountersOffset, uint64_t {CounterCount} * 4, 0);
	MemoryBarrier(command, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
	              vk::PipelineStageFlagBits2::eAllCommands,
	              vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite);
	readback.tick = m_scheduler.CurrentTick();
	m_scheduler.DeferOperation([this, &readback] { ProcessReadback(readback); });
	m_readback_index = (m_readback_index + 1) % m_readbacks.size();
}

void NativeRayTracing::ProcessReadback(Readback& readback) {
	readback.buffer->Invalidate(0, ReadbackSize);
	const auto* bytes = readback.buffer->Mapped().data();
	RtState     state {};
	std::memcpy(&state, bytes, sizeof(state));
	const auto        missing_count = std::min(state.missing_count, MissingCapacity);
	Common::LockGuard lock(m_mutex);
	for (uint32_t i = 0; i < missing_count; i++) {
		MissingEntry entry {};
		std::memcpy(&entry, bytes + ReadbackMissing + uint64_t {i} * MissingStride, sizeof(entry));
		if (entry.flags != 0 || entry.leaf_count == 0) {
			ReportAgain(entry.blas_va, entry.root, entry.table_index);
			continue;
		}
		m_new_slots.push_back(
		    {entry.blas_va, entry.root, entry.table_index, entry.leaf_count, entry.opaque_count, entry.alpha_count});
	}
	for (uint32_t i = 0; i < readback.build_count && i < readback.build_slots.size(); i++) {
		Job job {};
		std::memcpy(&job, bytes + ReadbackJobs + uint64_t {i} * JobStride, sizeof(job));
		const auto index = readback.build_slots[i];
		if (m_slot_list[index] == nullptr || m_slot_list[index]->evicted) {
			continue;
		}
		auto& slot = *m_slot_list[index];
		if ((job.result & 1u) == 0 && ((job.result >> 1u) & 5u) != 0) {
			// The walk found no room (pool or queue): built again, after a queue overflow with a
			// queue of the size the BLAS header allows.
			if (((job.result >> 1u) & 1u) != 0) {
				slot.leaf_count = 0;
			}
			if (!slot.needs_build) {
				slot.needs_build = true;
				slot.refit       = false;
				m_build_queue.push_back(index);
			}
		}
		if (job.leaves != 0 && (job.result & 0xfu) == 0) {
			slot.leaf_count = job.leaves;
		}
		if ((job.result & 1u) != 0) {
			m_grow.push_back({index, job.capacity0, job.capacity1});
		}
	}
	const auto changed = std::min(state.changed_count, ChangedCapacity);
	for (uint32_t i = 0; i < changed; i++) {
		uint32_t word = 0;
		std::memcpy(&word, bytes + ReadbackChanged + uint64_t {i} * 4, sizeof(word));
		const auto index = word & ~ChangedTopology;
		if (index >= m_slot_list.size() || m_slot_list[index] == nullptr || m_slot_list[index]->evicted) {
			continue;
		}
		auto& slot = *m_slot_list[index];
		if (slot.dynamic_until == 0) {
			m_dynamic.push_back(index);
		}
		slot.dynamic_until = m_prepare_id + DynamicPrepares;
		// e2fadd38 (adapted): the guest moved the boxes of the same leaves (a refit), so the host
		// gathers the new vertices through the entry table instead of walking the BVH again.
		const bool refit  = (word & ChangedTopology) == 0;
		slot.refit_change = refit;
		if (!slot.needs_build) {
			slot.needs_build = true;
			slot.refit       = refit;
			m_build_queue.push_back(index);
		} else if (!refit) {
			slot.refit = false;
		}
	}
	const auto used = std::min(state.verify_count, SlotCapacity);
	for (uint32_t i = 0; i < used; i++) {
		uint32_t index = 0;
		std::memcpy(&index, bytes + ReadbackUsed + uint64_t {i} * 4, sizeof(index));
		if (index < m_slot_list.size() && m_slot_list[index] != nullptr && !m_slot_list[index]->evicted) {
			auto& slot          = *m_slot_list[index];
			slot.last_use_frame = std::max(slot.last_use_frame, readback.frame);
		}
	}
	m_last_instance_counts[readback.source_key] = state.instance_count;
	// Statistics.
	m_stats.prepares++;
	m_stats.instances += state.instance_count;
	m_stats.inactive += state.inactive;
	m_stats.unknown += state.unknown_blas;
	m_stats.changed += changed;
	m_stats.missing += missing_count;
	m_stats.overflow += state.overflow != 0 || state.pool_overflow != 0 ? 1u : 0u;
	m_stats.empty += state.instance_count == 0 ? 1u : 0u;
	if (state.header_va != m_stats.last_header) {
		m_stats.last_header = state.header_va;
		m_stats.headers++;
	}
	for (uint32_t c = 0; c < CounterCount; c++) {
		uint32_t value = 0;
		std::memcpy(&value, bytes + ReadbackCounters + uint64_t {c} * 4, sizeof(value));
		m_stats.counters[c] += value;
	}
	readback.tick = 0;
	if (m_report_every != 0 && m_stats.prepares >= m_report_every) {
		Report();
	}
}

void NativeRayTracing::Report() {
	const auto& s = m_stats;
	uint32_t    live_slots = 0;
	for (const auto& slot: m_slot_list) {
		live_slots += slot != nullptr && !slot->evicted ? 1u : 0u;
	}
	LOGF("NativeRayTracing: %" PRIu64 " prepares, instances/prepare %.0f (inactive %.0f, unknown BLAS %.0f), "
	     "walk builds %" PRIu64 ", gathers %" PRIu64 ", changed %" PRIu64 ", missing %" PRIu64 ", overflows %" PRIu64
	     ", header changes %u, no instances %" PRIu64 ", header from the host %" PRIu64 ", reused in the frame %" PRIu64 ", slots %u, arena %.0f/%.0f MiB, last header 0x%" PRIx64 "\n",
	     s.prepares, static_cast<double>(s.instances) / static_cast<double>(s.prepares),
	     static_cast<double>(s.inactive) / static_cast<double>(s.prepares),
	     static_cast<double>(s.unknown) / static_cast<double>(s.prepares), s.builds, s.gathers, s.changed, s.missing,
	     s.overflow, s.headers, s.empty, s.hinted, s.reused, live_slots, static_cast<double>(m_arena.Used()) / 1048576.0,
	     static_cast<double>(m_arena.Reserved()) / 1048576.0, s.last_header);
	if (s.counters[0] != 0) {
		const auto rate = [&](uint32_t c) {
			return 100.0 * static_cast<double>(s.counters[c]) / static_cast<double>(s.counters[0]);
		};
		LOGF("NativeRayTracing verify: rays %" PRIu64 ", native hits %.2f%%, guest hits %.2f%%, hit/miss mismatch %.4f%%, "
		     "other instance %.4f%%, other triangle %.4f%%, distance %.4f%%, barycentrics %.4f%%\n",
		     s.counters[0], rate(1), rate(2), rate(3), rate(4), rate(5), rate(6), rate(7));
	}
	if (s.counters[8] != 0) {
		LOGF("NativeRayTracing lanes: %" PRIu64 " entered patched traversals, %" PRIu64 " natively (%.2f%%)\n", s.counters[8],
		     s.counters[9], 100.0 * static_cast<double>(s.counters[9]) / static_cast<double>(s.counters[8]));
	}
	m_stats = Stats {.last_header = s.last_header};
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
namespace {
// The allocated (not free) regions of the low 1 TiB of the process (the guest address space).
std::vector<std::pair<uint64_t, uint64_t>> UsedLowRegions() {
	std::vector<std::pair<uint64_t, uint64_t>> regions;
	uint64_t address = 0x10000;
	while (address < (uint64_t {1} << 40u)) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) == 0) {
			break;
		}
		const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto size = static_cast<uint64_t>(info.RegionSize);
		if (info.State != MEM_FREE) {
			regions.emplace_back(base, size);
		}
		address = base + size;
	}
	return regions;
}
} // namespace
#endif

bool PrepareNativeRayTracingRuntime(int gpu_index) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto before = UsedLowRegions();
	// The driver reserves large regions (about 136 GiB on an RTX 3090) at the lowest free addresses
	// above 4 GiB, which is where the guest's fixed mappings live (the AGC driver memory at
	// 0xfe0000000, user memory from 0x1000000000). While the instance and the device are created,
	// every free part of [4 GiB, 256 GiB) is held here, so the regions land above it, in guest
	// space that games leave unused.
	std::vector<void*> holds;
	for (uint64_t address = uint64_t {4} << 30u; address < (uint64_t {256} << 30u);) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<void*>(address), &info, sizeof(info)) == 0) {
			break;
		}
		const auto region = reinterpret_cast<uint64_t>(info.BaseAddress);
		// Reservations start at the 64 KiB allocation granularity.
		const auto base = (region + 0xffffu) & ~uint64_t {0xffff};
		const auto end  = std::min(region + static_cast<uint64_t>(info.RegionSize), uint64_t {256} << 30u);
		if (info.State == MEM_FREE && end > base) {
			void* hold = VirtualAlloc(reinterpret_cast<void*>(base), end - base, MEM_RESERVE, PAGE_NOACCESS);
			LOGF("NativeRayTracing: hold 0x%010llx-0x%010llx: %s (%lu)\n", static_cast<unsigned long long>(base),
			     static_cast<unsigned long long>(end), hold != nullptr ? "ok" : "failed", hold != nullptr ? 0ul : GetLastError());
			if (hold != nullptr) {
				holds.push_back(hold);
			}
		}
		address = region + static_cast<uint64_t>(info.RegionSize);
	}
	const auto release_holds = [&] {
		for (void* hold: holds) {
			VirtualFree(hold, 0, MEM_RELEASE);
		}
		holds.clear();
	};	// Keeps the loader (and so the driver) loaded after the temporary device is gone.
	static vk::detail::DynamicLoader loader;
	const auto get_proc = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	if (get_proc == nullptr) {
		release_holds();
		return false;
	}
	vk::detail::DispatchLoaderDynamic dispatch;
	dispatch.init(get_proc);
	vk::ApplicationInfo app {};
	app.pApplicationName = "Kyty RT initialization";
	app.apiVersion       = VK_API_VERSION_1_3;
	vk::InstanceCreateInfo create {};
	create.pApplicationInfo = &app;
	vk::Instance instance {};
	if (vk::createInstance(&create, nullptr, &instance, dispatch) != vk::Result::eSuccess) {
		release_holds();
		return false;
	}
	dispatch.init(instance);
	uint32_t count = 0;
	std::vector<vk::PhysicalDevice> physicals;
	if (instance.enumeratePhysicalDevices(&count, nullptr, dispatch) == vk::Result::eSuccess) {
		physicals.resize(count);
		if (instance.enumeratePhysicalDevices(&count, physicals.data(), dispatch) != vk::Result::eSuccess) {
			physicals.clear();
		}
	}
	bool warmed = false;
	for (size_t index = 0; index < physicals.size(); index++) {
		if (gpu_index >= 0 && index != static_cast<size_t>(gpu_index)) {
			continue;
		}
		const auto physical = physicals[index];
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR as_features {};
		vk::PhysicalDeviceRayQueryFeaturesKHR              rq_features {};
		as_features.pNext = &rq_features;
		vk::PhysicalDeviceFeatures2 features {};
		features.pNext = &as_features;
		physical.getFeatures2(&features, dispatch);
		if (!as_features.accelerationStructure || !rq_features.rayQuery) {
			continue;
		}
		uint32_t families_count = 0;
		physical.getQueueFamilyProperties(&families_count, nullptr, dispatch);
		std::vector<vk::QueueFamilyProperties> families(families_count);
		physical.getQueueFamilyProperties(&families_count, families.data(), dispatch);
		uint32_t family = 0;
		while (family < families_count && !(families[family].queueFlags & vk::QueueFlagBits::eCompute)) {
			family++;
		}
		if (family == families_count) {
			continue;
		}
		const float               priority = 1.0f;
		vk::DeviceQueueCreateInfo queue {};
		queue.queueFamilyIndex = family;
		queue.queueCount       = 1;
		queue.pQueuePriorities = &priority;
		vk::PhysicalDeviceVulkan12Features                 f12 {};
		vk::PhysicalDeviceRayQueryFeaturesKHR              rq {};
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR as {};
		f12.bufferDeviceAddress  = VK_TRUE;
		rq.rayQuery              = VK_TRUE;
		rq.pNext                 = &f12;
		as.accelerationStructure = VK_TRUE;
		as.pNext                 = &rq;
		const std::array<const char*, 3> extensions {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
		                                             VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
		                                             VK_KHR_RAY_QUERY_EXTENSION_NAME};
		vk::DeviceCreateInfo device_info {};
		device_info.pNext                   = &as;
		device_info.queueCreateInfoCount    = 1;
		device_info.pQueueCreateInfos       = &queue;
		device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		device_info.ppEnabledExtensionNames = extensions.data();
		vk::Device device {};
		if (physical.createDevice(&device_info, nullptr, &device, dispatch) == vk::Result::eSuccess) {
			dispatch.init(device);
			device.destroy(nullptr, dispatch);
			warmed = true;
		}
	}
	instance.destroy(nullptr, dispatch);
	release_holds();
	// The regions the driver runtime took (they stay out of the guest's reach).
	for (const auto& [base, size]: UsedLowRegions()) {
		// The large ones (the small ones are the driver's DLLs and heaps).
		if (size >= (uint64_t {1} << 30u) && std::find(before.begin(), before.end(), std::pair {base, size}) == before.end()) {
			LOGF("NativeRayTracing: driver runtime region 0x%010llx-0x%010llx\n", static_cast<unsigned long long>(base),
			     static_cast<unsigned long long>(base + size));
		}
	}
	LOGF("NativeRayTracing: driver runtime %s before the guest address space\n",
	     warmed ? "initialized" : "not initialized");
	return warmed;
#else
	(void)gpu_index;
	return false;
#endif
}

void PrepareNativeRayTracingDispatch(NativeRayTracing& rt, vk::CommandBuffer command,
                                     const ShaderRecompiler::IR::ShaderInfo& info,
                                     std::span<const uint32_t> user_data, uint64_t frame,
                                     const std::function<bool(uint64_t, uint64_t)>& readable) {
	NativeRayTracing::TlasSource source;
	if (!rt.Enabled() || !NativeRayTracing::FindTlasSource(info, source) || source.user_sgpr + 1u >= user_data.size()) {
		return;
	}
	const uint64_t user_pointer = uint64_t {user_data[source.user_sgpr]} | (uint64_t {user_data[source.user_sgpr + 1u]} << 32u);
	// The header address as the guest wrote it, when its memory holds no newer GPU writes (the table
	// the shader reads it from is plain guest memory, which the walk on the GPU sees only where a
	// buffer caches it); else the walk reads it.
	const uint64_t user = user_pointer & 0xffffffffffffull;
	uint64_t       header = 0;
	if (user != 0 && readable(user + source.vsharp_offset, 16)) {
		uint32_t vsharp[2] {};
		if (!LibKernel::Memory::TryReadGpuCleanBacking(user + source.vsharp_offset, vsharp, sizeof(vsharp))) {
			vsharp[0] = vsharp[1] = 0;
		}
		const uint64_t base = uint64_t {vsharp[0]} | (uint64_t {vsharp[1] & 0xffffu} << 32u);
		if (base != 0 && readable(base + source.header_offset, 8)) {
			if (!LibKernel::Memory::TryReadGpuCleanBacking(base + source.header_offset, &header, sizeof(header))) {
				header = 0;
			}
		}
	}
	rt.Prepare(command, user, source, frame, header);
}

} // namespace Libs::Graphics
