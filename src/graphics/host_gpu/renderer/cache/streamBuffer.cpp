#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/ramStats.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/vramStats.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <string>
#include <numeric>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

namespace {

constexpr size_t WATCHES_INITIAL_RESERVE = 0x4000;
constexpr size_t WATCHES_RESERVE_CHUNK   = 0x1000;

[[nodiscard]] VmaAllocationCreateFlags AllocationFlags(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::Upload:
		case MemoryUsage::Stream:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT |
			       VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		case MemoryUsage::Download:
			return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		case MemoryUsage::DeviceLocal: return {};
	}
	return {};
}

[[nodiscard]] VmaMemoryUsage AllocationUsage(MemoryUsage usage) {
	switch (usage) {
		case MemoryUsage::DeviceLocal:
		case MemoryUsage::Stream: return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		case MemoryUsage::Upload:
		case MemoryUsage::Download: return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
	}
	return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

[[nodiscard]] bool AlignUp(uint64_t value, uint64_t alignment, uint64_t& result) {
	if (alignment == 0) {
		result = value;
		return true;
	}
	const auto aligned = Common::AlignUp(value, alignment);
	if (aligned < value) {
		return false;
	}
	result = aligned;
	return true;
}

} // namespace

Buffer::Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
               uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size,
               bool transfer_shared, bool host_cached, bool sparse_residency)
    : m_graphics(&graphics), m_scheduler(&scheduler), m_usage(usage), m_cpu_address(cpu_address),
      m_size(size) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(graphics.allocator == nullptr || size == 0);

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = size;
	buffer_info.usage       = flags;
	const std::array<uint32_t, 2> families {graphics.queue_family, graphics.transfer_queue_family};
	if (transfer_shared && graphics.transfer_queue != nullptr) {
		buffer_info.sharingMode           = vk::SharingMode::eConcurrent;
		buffer_info.queueFamilyIndexCount = static_cast<uint32_t>(families.size());
		buffer_info.pQueueFamilyIndices   = families.data();
	}
	if (sparse_residency) {
		// No memory here: the owner binds pages to the ranges it uses (unbound ranges read zero).
		EXIT_IF(!graphics.sparse_residency_buffer_enabled || usage != MemoryUsage::DeviceLocal ||
		        host_cached || static_cast<bool>(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress));
		buffer_info.flags =
		    vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
		vk::Buffer sparse_buffer = nullptr;
		RequireVulkanSuccess(graphics.device.createBuffer(&buffer_info, nullptr, &sparse_buffer),
		                     "create sparse residency buffer");
		m_buffer = sparse_buffer;
		m_sparse = true;
		return;
	}

	// Buffers with device addresses (every guest buffer) take dedicated memory, a driver allocation
	// each. KYTY_BDA_SHARED_BLOCKS=1 (default off) lets the ones up to 64 MiB share VMA's memory
	// blocks instead: nothing needs the allocation's offset to be 0. The BDA page table stores
	// BufferDeviceAddress() + page offset per page (BufferCache::ChangeRegister), and
	// vkGetBufferDeviceAddress already includes the placement; mapped pointers come from VMA's
	// pMappedData and flushes/invalidates go through vmaFlush/InvalidateAllocation, both
	// allocation-relative. The allocator was created with VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
	// so its blocks carry the device-address flag.
	static const bool shared_blocks = [] {
		const auto* value = std::getenv("KYTY_BDA_SHARED_BLOCKS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	static constexpr uint64_t SharedBlockMax = 64ull * 1024 * 1024;
	const bool with_bda = bool(flags & vk::BufferUsageFlagBits::eShaderDeviceAddress);
	const VmaAllocationCreateFlags bda_flag =
	    with_bda && !(shared_blocks && size <= SharedBlockMax) ? VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
	                                                          : 0;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags =
	    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag | AllocationFlags(usage);
	allocation_info.usage = AllocationUsage(usage);
	allocation_info.preferredFlags = usage == MemoryUsage::DeviceLocal
	                                     ? VkMemoryPropertyFlags {}
	                                     : VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	if (host_cached) {
		allocation_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | bda_flag |
		                        VMA_ALLOCATION_CREATE_MAPPED_BIT |
		                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		allocation_info.usage         = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		allocation_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	}

	VmaAllocationInfo allocation_result {};
	VkBuffer          native_buffer = VK_NULL_HANDLE;
	const auto        create        = [&] {
		return static_cast<vk::Result>(vmaCreateBuffer(
		    graphics.allocator, static_cast<const VkBufferCreateInfo*>(buffer_info), &allocation_info,
		    &native_buffer, &m_allocation, &allocation_result));
	};
	auto result = create();
	if (result != vk::Result::eSuccess) {
		// Out of budget (VRAM full: a big texture pack, a smaller card). Not fatal: release what is
		// only retained for reuse and retry, then retry past the budget. Windows (WDDM) then pages
		// the allocation to system memory, slower but running; only a refusal of that stops here.
		graphics.LogMemoryBudget();
		const auto released = graphics.ReleaseRetainedMemory();
		result              = create();
		bool over_budget    = false;
		if (result != vk::Result::eSuccess) {
			allocation_info.flags &= ~static_cast<VmaAllocationCreateFlags>(
			    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
			result      = create();
			over_budget = result == vk::Result::eSuccess;
		}
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Kyty VRAM: a {}-byte buffer did not fit the memory budget; released {} retained bytes "
			    "and retried: {}\n",
			    size, released,
			    result != vk::Result::eSuccess
			        ? vk::to_string(result)
			        : std::string(over_budget ? "created past the budget (may page to system memory)"
			                                  : "created")));
		}
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	m_buffer = native_buffer;
	if (with_bda) {
		vk::BufferDeviceAddressInfo address_info {};
		address_info.buffer = m_buffer;
		m_device_address    = graphics.device.getBufferAddress(address_info);
		EXIT_IF(m_device_address == 0);
	}

	VkMemoryPropertyFlags properties = 0;
	vmaGetAllocationMemoryProperties(graphics.allocator, m_allocation, &properties);
	m_coherent = (properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
	if (allocation_result.pMappedData != nullptr) {
		m_mapped = {static_cast<uint8_t*>(allocation_result.pMappedData),
		            static_cast<size_t>(size)};
		Common::RamStats::Range((properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
		                           ? "Vulkan mapped device buffer"
		                           : "Vulkan mapped system buffer",
		                       m_mapped.data(), size);
	}
	if (VramStats::Enabled()) {
		const auto kind = cpu_address != 0              ? VramStats::Kind::GuestBuffer
		                  : usage != MemoryUsage::DeviceLocal ? VramStats::Kind::RingBuffer
		                                                      : VramStats::Kind::OtherBuffer;
		m_vram_bytes        = static_cast<uint64_t>(allocation_result.size);
		m_vram_kind         = static_cast<uint8_t>(kind);
		m_vram_device_local = (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
		VramStats::Note(kind, m_vram_device_local, static_cast<int64_t>(m_vram_bytes));
		VramStats::RegisterBuffer(this, {m_vram_bytes, cpu_address, static_cast<uint8_t>(usage),
		                                 m_vram_device_local, host_cached});
	}
}

Buffer::~Buffer() {
	if (m_vram_bytes != 0) {
		VramStats::Note(static_cast<VramStats::Kind>(m_vram_kind), m_vram_device_local,
		                -static_cast<int64_t>(m_vram_bytes));
		VramStats::UnregisterBuffer(this);
	}
	if (m_buffer != nullptr) {
		if (m_sparse) {
			// The owner frees the bound pages (after this, or before: the buffer is no longer used).
			m_graphics->device.destroyBuffer(m_buffer, nullptr);
		} else {
			vmaDestroyBuffer(m_graphics->allocator, m_buffer, m_allocation);
		}
	}
}

vk::DeviceAddress Buffer::BufferDeviceAddress() const noexcept {
	EXIT_IF(m_device_address == 0);
	return m_device_address;
}

bool Buffer::IsInBounds(uint64_t address, uint64_t size) const noexcept {
	return address >= m_cpu_address && size <= Size() && address - m_cpu_address <= Size() - size;
}

void Buffer::MarkContentWritten() {
	// Reusing a revision would make a retained GPU-content result appear current again.
	EXIT_IF(m_content_revision == UINT64_MAX);
	++m_content_revision;
}

void Buffer::Flush(uint64_t offset, uint64_t size) {
	EXIT_IF(m_mapped.empty() || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaFlushAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

void Buffer::Invalidate(uint64_t offset, uint64_t size) {
	EXIT_IF(m_usage != MemoryUsage::Download || offset > Size() || size > Size() - offset);
	if (!IsCoherent() && size != 0) {
		const auto result =
		    vmaInvalidateAllocation(m_graphics->allocator, m_allocation, offset, size);
		EXIT_NOT_IMPLEMENTED(static_cast<vk::Result>(result) != vk::Result::eSuccess);
	}
}

vk::BufferMemoryBarrier Buffer::Barrier(uint64_t offset, uint64_t size, vk::AccessFlags source,
                                        vk::AccessFlags destination) const {
	if (Handle() == nullptr || size == 0 || offset > Size() || size > Size() - offset) {
		EXIT("Buffer: invalid DMA barrier, handle=%p offset=0x%016" PRIx64 " size=0x%016" PRIx64
		     " capacity=0x%016" PRIx64 "\n",
		     static_cast<const void*>(Handle()), offset, size, Size());
	}
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = source;
	barrier.dstAccessMask       = destination;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = Handle();
	barrier.offset              = offset;
	barrier.size                = size;
	return barrier;
}

void Buffer::CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
                      uint64_t destination_offset, uint64_t size, vk::AccessFlags source_before,
                      vk::AccessFlags destination_before, vk::AccessFlags source_after,
                      vk::AccessFlags destination_after) {
	KYTY_GPU_OP_SITE("buffer.copy");
	if (size == 0 || source_offset > source.Size() || size > source.Size() - source_offset ||
	    destination_offset > Size() || size > Size() - destination_offset) {
		EXIT("Buffer: invalid copy range\n");
	}
	if (source.Handle() == Handle() && source_offset < destination_offset + size &&
	    destination_offset < source_offset + size) {
		EXIT("Buffer: overlapping self-copy\n");
	}
	command.EndRendering();
	const vk::BufferMemoryBarrier before[] = {
	    source.Barrier(source_offset, size, source_before, vk::AccessFlagBits::eTransferRead),
	    Barrier(destination_offset, size, destination_before, vk::AccessFlagBits::eTransferWrite),
	};
	const auto host_access  = vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eHostWrite;
	auto       before_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_before | destination_before) & host_access)) {
		before_stage |= vk::PipelineStageFlagBits::eHost;
	}
	const auto native = command.Handle();
	native.pipelineBarrier(before_stage, vk::PipelineStageFlagBits::eTransfer,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, before, 0, nullptr);
	const vk::BufferCopy copy {source_offset, destination_offset, size};
	native.copyBuffer(source.Handle(), Handle(), 1, &copy);
	MarkContentWritten();
	const vk::BufferMemoryBarrier after[] = {
	    source.Barrier(source_offset, size, vk::AccessFlagBits::eTransferRead, source_after),
	    Barrier(destination_offset, size, vk::AccessFlagBits::eTransferWrite, destination_after),
	};
	auto after_stage = vk::PipelineStageFlags {vk::PipelineStageFlagBits::eAllCommands};
	if (static_cast<bool>((source_after | destination_after) & host_access)) {
		after_stage |= vk::PipelineStageFlagBits::eHost;
	}
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, after_stage,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 2, after, 0, nullptr);
}

void Buffer::Fill(uint64_t offset, uint64_t size, uint32_t value) {
	KYTY_GPU_OP_SITE("buffer.fill");
	if (((offset | size) & 3u) != 0) {
		EXIT("Buffer: fill range must be dword aligned\n");
	}
	auto& command = Scheduler().Current();
	command.EndRendering();
	const auto before =
	    Barrier(offset, size, vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	            vk::AccessFlagBits::eTransferWrite);
	const auto native = command.Handle();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       0, nullptr, 1, &before, 0, nullptr);
	native.fillBuffer(Handle(), offset, size, value);
	MarkContentWritten();
	const auto after = Barrier(offset, size, vk::AccessFlagBits::eTransferWrite,
	                           vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
}

StreamBuffer::StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
                           uint64_t size, bool transfer_shared, vk::BufferUsageFlags extra_flags,
                           bool host_cached)
    : Buffer(graphics, scheduler, usage, 0, AllFlags | extra_flags, size, transfer_shared,
             host_cached),
      m_current_watches(WATCHES_INITIAL_RESERVE), m_previous_watches(WATCHES_INITIAL_RESERVE) {}

bool StreamBuffer::NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
                                        uint64_t& alignment) {
	if (coherent) {
		return true;
	}
	if (!AlignUp(size, atom, size)) {
		return false;
	}
	const auto divisor = std::gcd(alignment, atom);
	if (alignment != 0 && alignment / divisor > UINT64_MAX / atom) {
		return false;
	}
	alignment = alignment == 0 ? atom : alignment / divisor * atom;
	return true;
}

std::pair<uint8_t*, uint64_t> StreamBuffer::Map(uint64_t size, uint64_t alignment,
                                                bool allow_wait) {
	if (Mapped().empty()) {
		return {nullptr, 0};
	}
	uint64_t   mapped_size = size;
	const auto atom        = Graphics().physical_device_properties.limits.nonCoherentAtomSize;
	if (!NormalizeReservation(IsCoherent(), atom, mapped_size, alignment)) {
		return {nullptr, 0};
	}
	if (mapped_size > Size()) {
		return {nullptr, 0};
	}

	uint64_t aligned_offset = 0;
	if (!AlignUp(m_offset, alignment, aligned_offset)) {
		return {nullptr, 0};
	}

	const bool wrap = aligned_offset > Size() - mapped_size;
	if (wrap) {
		aligned_offset = 0;
	}

	auto wait_cursor = wrap ? size_t {0} : m_wait_cursor;
	auto wait_bound  = wrap ? uint64_t {0} : m_wait_bound;
	auto invalidation_mark =
	    wrap ? std::optional<size_t> {m_current_watch_cursor} : m_invalidation_mark;
	auto& pending_watches = wrap ? m_current_watches : m_previous_watches;
	if (!WaitPendingOperations(pending_watches, invalidation_mark, aligned_offset + mapped_size,
	                           allow_wait, wait_cursor, wait_bound)) {
		return {nullptr, 0};
	}

	if (wrap) {
		m_invalidation_mark    = invalidation_mark;
		m_current_watch_cursor = 0;
		std::swap(m_previous_watches, m_current_watches);
	}
	m_wait_cursor = wait_cursor;
	m_wait_bound  = wait_bound;
	m_offset      = aligned_offset;
	m_mapped_size = mapped_size;
	return {Mapped().data() + m_offset, m_offset};
}

void StreamBuffer::Commit() {
	if (Usage() != MemoryUsage::Download && m_mapped_size != 0) {
		Flush(m_offset, m_mapped_size);
	}

	m_offset += m_mapped_size;
	const auto tick = Scheduler().CurrentTick();
	if (m_current_watch_cursor != 0 && m_current_watches[m_current_watch_cursor - 1].tick == tick) {
		m_current_watches[m_current_watch_cursor - 1].upper_bound = m_offset;
		return;
	}
	if (m_current_watch_cursor + 1 >= m_current_watches.size()) {
		m_current_watches.resize(m_current_watches.size() + WATCHES_RESERVE_CHUNK);
	}
	auto& watch       = m_current_watches[m_current_watch_cursor++];
	watch.upper_bound = m_offset;
	watch.tick        = tick;
}

uint64_t StreamBuffer::Copy(const void* source, uint64_t size, uint64_t alignment) {
	EXIT_IF(source == nullptr);
	const auto [data, offset] = Map(size, alignment);
	EXIT_IF(data == nullptr);
	std::memcpy(data, source, static_cast<size_t>(size));
	Commit();
	return offset;
}

bool StreamBuffer::WaitPendingOperations(const std::vector<Watch>& watches,
                                         std::optional<size_t>     invalidation_mark,
                                         uint64_t requested_upper_bound, bool allow_wait,
                                         size_t& wait_cursor, uint64_t& wait_bound) {
	if (!invalidation_mark.has_value()) {
		return true;
	}
	while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
		const auto& watch = watches[wait_cursor];
		if (!Scheduler().IsFree(watch.tick) && !allow_wait) {
			return false;
		}
		{
			Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitStreamWrap);
			Scheduler().Wait(watch.tick);
		}
		if (Usage() == MemoryUsage::Download) {
			Scheduler().WaitPriorityOperations(watch.tick);
		}
		wait_bound = watch.upper_bound;
		++wait_cursor;
	}
	return true;
}

} // namespace Libs::Graphics
