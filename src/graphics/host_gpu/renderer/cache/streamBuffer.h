#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

VK_DEFINE_HANDLE(VmaAllocation)

namespace Libs::Graphics {

class CommandBuffer;
class CommandScheduler;
struct StreamBufferTestAccess;
struct GraphicContext;

enum class MemoryUsage : uint8_t {
	DeviceLocal,
	Upload,
	Download,
	Stream,
};

inline constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

inline constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

// KYTY_HOST_BUFFER_MB in bytes (0: off): device-local buffers at least this large live in system
// memory.
[[nodiscard]] uint64_t HostBufferBytes();

// Buffer over imported host memory (KYTY_INPLACE, InplaceHeap).
struct ImportHostTag {};

class Buffer {
public:
	// A buffer whose memory is the host range [host_pointer, +size) imported with
	// VK_EXT_external_memory_host (GraphicContext::external_memory_host_enabled), usable as
	// uniform, storage, vertex, index and indirect buffer and by device address. CpuAddress() is 0:
	// the owner computes offsets. Handle() is null when the import failed.
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, void* host_pointer, uint64_t size,
	       ImportHostTag);
	// transfer_shared: concurrent sharing with the upload DMA transfer queue family
	// (GraphicContext::transfer_queue_family), when the device has that queue.
	// host_cached: mapped cached host memory (CPU writes stay in the CPU caches, the GPU reads
	// them over the bus) instead of what `usage` picks (KYTY_STREAM_RING_HOST, BufferCache).
	// sparse_residency: a sparse residency buffer without memory (DeviceLocal only, needs
	// GraphicContext::sparse_residency_buffer_enabled); its owner binds memory to the ranges it
	// uses, the others read as zero (KYTY_BDA_PAGETABLE_SPARSE, BufferCache).
	Buffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	       uint64_t cpu_address, vk::BufferUsageFlags flags, uint64_t size,
	       bool transfer_shared = false, bool host_cached = false, bool sparse_residency = false);
	~Buffer();
	KYTY_CLASS_NO_COPY(Buffer);

	[[nodiscard]] vk::Buffer         Handle() const noexcept { return m_buffer; }
	[[nodiscard]] uint64_t           Size() const noexcept { return m_size; }
	[[nodiscard]] std::span<uint8_t> Mapped() const noexcept { return m_mapped; }
	[[nodiscard]] bool               IsCoherent() const noexcept { return m_coherent; }
	[[nodiscard]] bool               IsSparse() const noexcept { return m_sparse; }
	[[nodiscard]] MemoryUsage        Usage() const noexcept { return m_usage; }
	[[nodiscard]] uint64_t           CpuAddress() const noexcept { return m_cpu_address; }
	// GPU-thread command-recording revision, including writes that have not executed yet.
	[[nodiscard]] uint64_t ContentRevision() const noexcept { return m_content_revision; }
	void MarkContentWritten();
	[[nodiscard]] vk::DeviceAddress BufferDeviceAddress() const noexcept;
	[[nodiscard]] uint64_t           Offset(uint64_t address) const noexcept {
		return address - m_cpu_address;
	}
	[[nodiscard]] bool IsInBounds(uint64_t address, uint64_t size) const noexcept;
	void               IncreaseStreamScore(int score) noexcept { stream_score += score; }
	[[nodiscard]] int  StreamScore() const noexcept { return stream_score; }
	void               Flush(uint64_t offset, uint64_t size);
	void               Invalidate(uint64_t offset, uint64_t size);
	void CopyFrom(CommandBuffer& command, const Buffer& source, uint64_t source_offset,
	              uint64_t destination_offset, uint64_t size,
	              vk::AccessFlags source_before      = vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_before = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags source_after       = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlags destination_after  = vk::AccessFlagBits::eMemoryRead |
	                                                   vk::AccessFlagBits::eMemoryWrite,
	              bool            batch              = false);
	// Several non-overlapping regions from `source` in one copy command, with one barrier pair over
	// the spans the regions cover (KYTY_BDA_LIVE_BATCH page-table entry writes).
	void CopyRegionsFrom(CommandBuffer& command, const Buffer& source, std::span<const vk::BufferCopy> regions,
	                     vk::AccessFlags source_before = vk::AccessFlagBits::eMemoryWrite);
	// batch: the caller records nothing through a native handle obtained before this call until the
	// next barrier flush point, so KYTY_DMA_BARRIER_BATCH may leave the after barriers queued.
	void Fill(uint64_t offset, uint64_t size, uint32_t value, bool batch = false);
	// KYTY_DMA_BARRIER_BATCH (streamBuffer.cpp; live, default off).
	[[nodiscard]] static bool DmaBarrierBatchEnabled();

	// BufferCache state lives directly on the resource.
	bool   is_deleted   = false;
	int    stream_score = 0;
	size_t lru_id       = 0;
	// KYTY_BUFFER_LRU_SKIP=1 (BufferCache::TouchBuffer): the tick the LRU item last received.
	mutable uint64_t lru_tick = 0;
	// KYTY_GC_ASYNC_DOWNLOAD=1: collected with its download in flight (BufferCache::
	// RunGarbageCollector), and whether a draw or dispatch used it again before the download
	// landed (then it stays).
	bool         gc_pending = false;
	mutable bool gc_reused  = false;

protected:
	[[nodiscard]] GraphicContext&   Graphics() const noexcept { return *m_graphics; }
	[[nodiscard]] CommandScheduler& Scheduler() const noexcept { return *m_scheduler; }

private:
	[[nodiscard]] vk::BufferMemoryBarrier Barrier(uint64_t offset, uint64_t size,
	                                              vk::AccessFlags source,
	                                              vk::AccessFlags destination) const;

	GraphicContext*               m_graphics    = nullptr;
	CommandScheduler*             m_scheduler   = nullptr;
	MemoryUsage                   m_usage       = MemoryUsage::DeviceLocal;
	uint64_t                      m_cpu_address = 0;
	uint64_t                      m_content_revision = 1;
	vk::DeviceAddress             m_device_address = 0;
	vk::Buffer                    m_buffer     = nullptr;
	VmaAllocation                 m_allocation = nullptr;
	vk::DeviceMemory              m_imported_memory = nullptr;
	uint64_t                      m_size;
	bool                          m_coherent = false;
	bool                          m_sparse   = false;
	std::span<uint8_t>            m_mapped;
	// KYTY_VRAM_STATS (vramStats.h): what this buffer was counted as (0 bytes: not counted).
	uint64_t                      m_vram_bytes        = 0;
	uint8_t                       m_vram_kind         = 0;
	bool                          m_vram_device_local = false;
};

class StreamBuffer final: public Buffer {
public:
	// extra_flags: usage beyond AllFlags (e.g. eShaderDeviceAddress for a ring whose ranges
	// shaders reach by device address).
	StreamBuffer(GraphicContext& graphics, CommandScheduler& scheduler, MemoryUsage usage,
	             uint64_t size, bool transfer_shared = false,
	             vk::BufferUsageFlags extra_flags = {}, bool host_cached = false);

	[[nodiscard]] std::pair<uint8_t*, uint64_t> Map(uint64_t size, uint64_t alignment = 0,
	                                                bool allow_wait = true);
	void                                        Commit();
	[[nodiscard]] uint64_t Copy(const void* source, uint64_t size, uint64_t alignment = 0);
	// A Map() whose Commit() has not run yet: any Map() / Copy() now reuses its bytes.
	[[nodiscard]] bool ReservationPending() const noexcept { return m_reservation_pending; }
	// The caller drops its Map() without committing it (nothing was staged that a copy reads).
	void AbandonReservation() noexcept { m_reservation_pending = false; }

private:
	friend struct StreamBufferTestAccess;

	struct Watch {
		uint64_t tick        = 0;
		uint64_t upper_bound = 0;
	};

	[[nodiscard]] static bool NormalizeReservation(bool coherent, uint64_t atom, uint64_t& size,
	                                               uint64_t& alignment);
	[[nodiscard]] bool        WaitPendingOperations(const std::vector<Watch>& watches,
	                                                std::optional<size_t>     invalidation_mark,
	                                                uint64_t requested_upper_bound, bool allow_wait,
	                                                size_t& wait_cursor, uint64_t& wait_bound);

	uint64_t              m_offset      = 0;
	uint64_t              m_mapped_size = 0;
	bool                  m_reservation_pending = false;
	std::vector<Watch>    m_current_watches;
	size_t                m_current_watch_cursor = 0;
	std::optional<size_t> m_invalidation_mark;
	std::vector<Watch>    m_previous_watches;
	size_t                m_wait_cursor = 0;
	uint64_t              m_wait_bound  = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFER_H_
