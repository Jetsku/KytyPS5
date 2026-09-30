#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/uploadDma.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool IncrementalBdaSyncEnabled() {
	const auto* value = std::getenv("KYTY_BDA_INCREMENTAL_SYNC");
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// KYTY_BDA_HOT_SYNC=0 restores the U42 behaviour: any hot page disables incremental BDA
// synchronization (every pass scans every mapped buffer) and hot pages are snapshotted before
// their shadow compare. Only meaningful with KYTY_BDA_INCREMENTAL_SYNC=1.
bool BdaHotSyncEnabled() {
	const auto* value = std::getenv("KYTY_BDA_HOT_SYNC");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_BDA_HOT_SYNC_VERIFY=1 follows every hot pass with the full scan it replaced and counts
// the CPU-dirty non-hot pages that scan finds while the pass's epochs still hold
// (BdaSyncHotVerifyMismatches, expected 0); "exit" stops on the first one. Diagnostic: it costs
// the full scans again.
int BdaHotSyncVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BDA_HOT_SYNC_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

bool BdaHotSyncVerifyEnabled() {
	return BdaHotSyncVerifyMode() != 0;
}

// KYTY_BDA_DIRTY_LOG=0 restores the full scan of every mapped buffer whenever the fault epoch
// moved (bufferCache.h). Only meaningful with KYTY_BDA_INCREMENTAL_SYNC=1 and KYTY_BDA_HOT_SYNC.
bool BdaDirtyLogEnabled() {
	const auto* value = std::getenv("KYTY_BDA_DIRTY_LOG");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_BDA_DIRTY_LOG_VERIFY=1 follows every dirty-log pass with the full scan it replaced and
// counts the CPU-dirty non-hot pages that scan finds while the pass's epochs still hold
// (BdaSyncLogVerifyMismatches, expected 0); "exit" stops on the first one. Diagnostic: it costs
// the full scans again.
int BdaDirtyLogVerifyMode() {
	const auto* value = std::getenv("KYTY_BDA_DIRTY_LOG_VERIFY");
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "exit") == 0 ? 2 : 1;
}

// KYTY_UPLOAD_BATCH_SCOPED_FLUSH=0 restores recording every pending barrier at the end of each
// outermost UploadBatch scope, even when the scope queued no upload.
bool UploadBatchScopedFlushEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_UPLOAD_BATCH_SCOPED_FLUSH");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// Guest read faults copy GPU-owned bytes on a side command buffer unless this is "0".
bool SideReadbackEnabled() {
	const auto* value = std::getenv("KYTY_READBACK_SIDE_COPY");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// GPU-thread reads (CP reads of GPU-written data, GPU-thread faults) also use side copies and wait
// for them in place, unless KYTY_READBACK_SIDE_GPU_THREAD is "0" (then they always drain).
bool SideReadbackGpuThreadEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_READBACK_SIDE_GPU_THREAD");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// Aligned side-copy window in bytes: a power of two between 4 KiB and 1 MiB (default 64 KiB).
uint64_t SideReadbackWindow() {
	constexpr uint64_t Default = 64 * 1024;
	const auto*        value   = std::getenv("KYTY_READBACK_SIDE_COPY_WINDOW_KB");
	if (value == nullptr) {
		return Default;
	}
	char*      end = nullptr;
	const auto kib = std::strtoull(value, &end, 10);
	if (end == value || *end != '\0' || kib < 4 || kib > 1024 || (kib & (kib - 1)) != 0) {
		return Default;
	}
	return kib * 1024;
}

uint64_t ParseEnvU64(const char* name, uint64_t fallback) {
	const auto* value = std::getenv(name);
	if (value == nullptr) {
		return fallback;
	}
	char*      end    = nullptr;
	const auto parsed = std::strtoull(value, &end, 10);
	return end == value || *end != '\0' ? fallback : parsed;
}

// Guest write-fault policy of the buffer tracker (RegionManager::MarkWriteFault):
//   KYTY_FAULT_AHEAD_KB   aligned window made CPU-dirty around a write-faulting page: a power of
//                         two, 8..256 KiB (default 32); 0 or 4 disables fault-ahead.
//   KYTY_HOT_PAGES        0 disables hot pages. A page write-faulting in KYTY_HOT_PAGE_FRAMES
//                         consecutive guest frames (default 3) stays CPU-dirty and writable; its
//                         uploads copy it only when its contents differ from the last copy
//                         uploaded (exact shadow compare). It returns to normal tracking after
//                         KYTY_HOT_PAGE_QUIET_FRAMES frames (default 8) without a change or
//                         without an upload, on any GPU-side write that can reach it, or when its
//                         buffer is untracked. At most KYTY_HOT_PAGE_MAX pages (default 1024,
//                         4 KiB shadow each) are hot.
//   KYTY_UPLOAD_COPY_OUTSIDE_LOCK  0 copies written uploads with their region locks held (the
//                         previous behaviour) instead of MemoryTracker::ForEachWrittenUploadRange.
MemoryTracker::FaultPolicy BufferFaultPolicy() {
	MemoryTracker::FaultPolicy policy;
	const auto ahead_kib = ParseEnvU64("KYTY_FAULT_AHEAD_KB", 32);
	if (ahead_kib >= 8 && ahead_kib <= 256 && (ahead_kib & (ahead_kib - 1)) == 0) {
		policy.ahead_pages = static_cast<uint32_t>(ahead_kib * 1024 / TRACKER_PAGE_SIZE);
	}
	if (ParseEnvU64("KYTY_HOT_PAGES", 1) != 0) {
		policy.hot_frames = static_cast<uint32_t>(
		    std::clamp<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_FRAMES", 3), 1, 255));
		policy.hot_max =
		    static_cast<uint32_t>(std::min<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_MAX", 1024), 65536));
		if (policy.hot_max == 0) {
			policy.hot_frames = 0;
		}
	}
	policy.copy_outside_lock = ParseEnvU64("KYTY_UPLOAD_COPY_OUTSIDE_LOCK", 1) != 0;
	return policy;
}

uint32_t HotPageQuietFrames() {
	return static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_QUIET_FRAMES", 8), 1, 1000));
}

// BufferCache::m_hot_check_limit (0 disables).
uint32_t HotPageCheckLimit() {
	return static_cast<uint32_t>(
	    std::min<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_CHECK_LIMIT", 64), 1u << 20u));
}

// BufferCache::RangeMemo (bufferCache.h).
bool RangeMemoEnabled() {
	return ParseEnvU64("KYTY_BUFFER_RANGE_MEMO", 1) != 0;
}

// The small-read stream decision of ObtainBuffer from one MemoryTracker::QueryDirty.
bool DirtyQueryCombinedEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_BUFFER_DIRTY_QUERY_COMBINED", 1) != 0;
	return enabled;
}

int RangeMemoVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BUFFER_RANGE_MEMO_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// KYTY_TRACKER_RELAXED_QUERIES (default on): on the GPU thread, the small-read stream decision
// and the "nothing to upload" test of read synchronizations use the tracker's lock-free dirty
// mirrors (MemoryTracker::QueryDirtyRelaxed) instead of the region locks.
bool RelaxedQueriesEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_TRACKER_RELAXED_QUERIES", 1) != 0;
	return enabled;
}

// KYTY_BINDING_EPOCH_MEMO (default on; =0 off): read bindings reused within a sync epoch
// (BufferCache::BindingMemo).
bool BindingEpochMemoEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BINDING_EPOCH_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_BINDING_EPOCH_MEMO_VERIFY=1|exit: every hit also runs the normal path
// (BufferCache::VerifyBindingHit).
int BindingEpochMemoVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BINDING_EPOCH_MEMO_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// 0 when the variable is unset, empty or "0"; 2 for "exit"; 1 otherwise. Read per cache, so tests
// can build caches with either setting.
int EnvVerifyMode(const char* name) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "exit") == 0 ? 2 : 1;
}

// KYTY_BINDING_MEMO_CROSS_EPOCH (default on; =0 off): a cache-buffer read-binding memo from an
// earlier sync epoch is reused once the range is proven to have no CPU-dirty page
// (BufferCache::ObtainReadBinding).
bool BindingMemoCrossEpochEnabled() {
	return ParseEnvU64("KYTY_BINDING_MEMO_CROSS_EPOCH", 1) != 0;
}

// KYTY_WRITTEN_SYNC_SKIP (default on; =0 off): a written synchronization of a range that is
// entirely GPU-owned does nothing and is skipped (BufferCache::SynchronizeBuffer).
// KYTY_WRITTEN_SYNC_SKIP_VERIFY=1|exit runs the normal path anyway and counts skips it would
// have made while that path found pages to upload (WrittenSyncSkipVerifyMismatches).
bool WrittenSyncSkipEnabled() {
	return ParseEnvU64("KYTY_WRITTEN_SYNC_SKIP", 1) != 0;
}

// KYTY_FALSE_SHARING_WRITES (default off; =1 on): write faults on GPU-owned pages at bytes the GPU
// never wrote release the page without draining the GPU (BufferCache::TryFalseSharingWrite).
bool FalseSharingWritesEnabled() {
	return ParseEnvU64("KYTY_FALSE_SHARING_WRITES", 0) != 0;
}

// KYTY_BDA_SYNC_EPOCH_VERIFY=1|exit: every skipped BDA pass runs anyway and counts the pages the
// skip would have missed that no guest write explains (BufferCache::VerifyBdaEpochSkip).
int BdaEpochVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_BDA_SYNC_EPOCH_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// KYTY_TRACKER_RELAXED_VERIFY=1|exit: every relaxed answer is followed by the locked query. Only
// the transitions other threads can make in between (a page turning CPU-dirty, a GPU-dirty page
// published) may separate them, and only while the range's mutation serials moved.
int RelaxedVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_TRACKER_RELAXED_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// Readback window of guest write faults on GPU-owned pages, in bytes: a power of two between
// 4 KiB (only the faulting page) and 512 KiB (default, the read-drain window). Every page of the
// window loses GPU ownership, so a smaller window downloads fewer bytes per fault but makes a
// CPU writer walking a larger GPU-written range drain the GPU once per window instead.
uint64_t WriteFaultWindow() {
	static const uint64_t window = []() -> uint64_t {
		constexpr uint64_t Default = 512 * 1024;
		const auto*        value   = std::getenv("KYTY_WRITE_FAULT_WINDOW_KB");
		if (value == nullptr) {
			return Default;
		}
		char*      end = nullptr;
		const auto kib = std::strtoull(value, &end, 10);
		if (end == value || *end != '\0' || kib < 4 || kib > 512 || (kib & (kib - 1)) != 0) {
			return Default;
		}
		return kib * 1024;
	}();
	return window;
}

// Eager readback publication (BufferCache::IssueEagerReadbacks) unless KYTY_READBACK_EAGER is
// "0". Read per cache (not cached in a static) so tests can build caches with either setting.
bool EagerReadbackEnabled() {
	const auto* value = std::getenv("KYTY_READBACK_EAGER");
	return value == nullptr || !(value[0] == '0' && value[1] == '\0');
}

// KYTY_READBACK_EAGER_PAGES (hot pages kept, default 64, at most 256),
// KYTY_READBACK_EAGER_IDLE_FRAMES (frames without a readback before a page stops being hot,
// default 600) and KYTY_READBACK_EAGER_FRAME_BUDGET (copies per page and frame, default 4).
EagerReadbackPages::Limits EagerReadbackLimits() {
	EagerReadbackPages::Limits limits;
	limits.capacity =
	    static_cast<uint32_t>(std::min<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_PAGES", 64), 256));
	limits.idle_frames = static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_IDLE_FRAMES", 600), 1, 1000000));
	limits.frame_budget = static_cast<uint32_t>(
	    std::clamp<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_FRAME_BUDGET", 4), 1, 1024));
	return limits;
}

// Early submissions per frame after a recorded writer of a page the GPU thread reads back
// (KYTY_READBACK_EAGER_FLUSHES, default 8; 0 disables them).
uint32_t EagerFlushBudget() {
	return static_cast<uint32_t>(
	    std::min<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_FLUSHES", 8), 100000));
}

// Reads larger than this are bulk readbacks (e.g. DCC metadata): their pages do not become hot.
constexpr uint64_t EagerReadMaxBytes = 64 * 1024;

} // namespace

// One side-copy readback: the exact GPU-dirty bytes of a tracker-page-aligned window, copied
// by a command buffer outside the scheduler's recording. Completion (any thread, exactly once)
// publishes them to the backing and unprotects the window's pages no newer writer re-owned.
// An eager readback (KYTY_READBACK_EAGER) is the same record for one tracker page whose copy
// ends a scheduler recording instead: `value` is that recording's master tick, and the
// completion runner completes it (if no reader did first) once that tick is reached.
struct BufferCache::SideReadback {
	std::mutex              mutex;
	std::atomic<bool>       done {false};
	uint64_t                begin       = 0;
	uint64_t                end         = 0;
	uint64_t                value       = 0;
	uint32_t                slot        = 0;
	bool                    eager       = false;
	uint64_t                publication = 0;
	// HangTrace clock at issue (eager rows report issue-to-publication time).
	uint64_t                issue_ns    = 0;
	// Staged at (address - begin) within the slot.
	std::vector<GuestRange> ranges;
};

struct BufferCache::SideReadbackState {
	static constexpr uint32_t SlotCount = 16;
	// One tracker page each; a slot is busy from its eager copy's issue to its publication.
	static constexpr uint32_t EagerSlotCount = 64;
	struct Slot {
		vk::CommandBuffer command = nullptr;
		// Set by the GPU thread at issue, cleared after the slot's publication has read it.
		std::atomic<bool> busy {false};
	};

	SideReadbackState(GraphicContext& context, CommandScheduler& scheduler, uint64_t window_size,
	                  bool eager)
	    : graphics(context), window(window_size) {
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.queueFamilyIndex = graphics.queue_family;
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                  vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
		                     "create side-readback command pool");
		std::array<vk::CommandBuffer, SlotCount> buffers {};
		vk::CommandBufferAllocateInfo            allocate {};
		allocate.commandPool        = pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = SlotCount;
		RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&allocate, buffers.data()),
		                     "allocate side-readback command buffers");
		for (uint32_t index = 0; index < SlotCount; ++index) {
			slots[index].command = buffers[index];
		}
		vk::SemaphoreTypeCreateInfo type_info {};
		type_info.semaphoreType = vk::SemaphoreType::eTimeline;
		type_info.initialValue  = 0;
		vk::SemaphoreCreateInfo create_info {};
		create_info.pNext = &type_info;
		RequireVulkanSuccess(graphics.device.createSemaphore(&create_info, nullptr, &semaphore),
		                     "create side-readback timeline semaphore");
		staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
		                                   vk::BufferUsageFlagBits::eTransferDst,
		                                   window * SlotCount);
		EXIT_IF(staging->Mapped().empty());
		SetVulkanObjectNameF(graphics.device, staging->Handle(), "Kyty.SideReadbackStaging");
		if (eager) {
			eager_staging = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
			                                         vk::BufferUsageFlagBits::eTransferDst,
			                                         TRACKER_PAGE_SIZE * EagerSlotCount);
			EXIT_IF(eager_staging->Mapped().empty());
			SetVulkanObjectNameF(graphics.device, eager_staging->Handle(),
			                     "Kyty.EagerReadbackStaging");
		}
	}

	~SideReadbackState() {
		eager_staging.reset();
		staging.reset();
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
		if (pool != nullptr) {
			graphics.device.destroyCommandPool(pool, nullptr);
		}
	}

	KYTY_CLASS_NO_COPY(SideReadbackState);

	void Wait(uint64_t target) const {
		uint64_t current = 0;
		RequireVulkanSuccess(graphics.device.getSemaphoreCounterValue(semaphore, &current),
		                     "query side-readback semaphore");
		if (current >= target) {
			return;
		}
		vk::SemaphoreWaitInfo wait_info {};
		wait_info.semaphoreCount = 1;
		wait_info.pSemaphores    = &semaphore;
		wait_info.pValues        = &target;
		RequireVulkanSuccess(graphics.device.waitSemaphores(&wait_info, UINT64_MAX),
		                     "wait for side readback");
	}

	GraphicContext&                            graphics;
	const uint64_t                             window;
	vk::CommandPool                            pool      = nullptr;
	vk::Semaphore                              semaphore = nullptr;
	// GPU thread only: the last signal value handed out.
	uint64_t                                   next_value = 0;
	std::array<Slot, SlotCount>                slots;
	std::unique_ptr<Buffer>                    staging;
	// Eager copies (null unless KYTY_READBACK_EAGER): EagerSlotCount tracker pages.
	std::unique_ptr<Buffer>                    eager_staging;
	std::array<std::atomic<bool>, EagerSlotCount> eager_busy {};
	mutable std::mutex                         pending_mutex;
	std::vector<std::shared_ptr<SideReadback>> pending;
	std::atomic<size_t>                        pending_count {0};
};

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		buffer.CopyFrom(m_scheduler.Current(), m_staging_buffer, offset, buffer.Offset(address),
		                chunk, vk::AccessFlagBits::eHostWrite);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	InvalidateBdaSynchronization();
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	{
		// A pending side copy reads this buffer outside the scheduler's timeline; the deferred
		// erase below is not ordered after it.
		const auto& buffer = m_slot_buffers[id];
		CompleteSideReadbacks(buffer.CpuAddress(), buffer.Size());
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                       const std::shared_ptr<EarlyReleasedDownload>& early) {
	// An older side publication of these pages must reach the backing (and settle its pages)
	// before this newer download is queued; otherwise it could overwrite newer bytes.
	CompleteSideReadbacks(vaddr, size);
	KYTY_GPU_OP_SITE("buffercache.download");
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    // Ownership moves to the backing publication registered below (which bumps again).
		    CleanVerdict::Invalidate(address, bytes, Coherence::Source::BufferDirtySubtract);
		    m_gpu_modified_ranges.Subtract(address, bytes);
	    });
	if (copies.empty()) {
		return false;
	}

	const auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	if (mapped == nullptr) {
		EXIT("BufferCache: download exceeds 64 MiB staging buffer capacity\n");
	}
	m_download_buffer.Commit();
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), m_download_buffer.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_download_buffer.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	std::vector<GuestRange> publication_ranges;
	publication_ranges.reserve(copies.size());
	for (const auto& copy: copies) {
		publication_ranges.push_back({buffer_address + copy.srcOffset, copy.size});
	}
	if (early != nullptr) {
		// The pages are released before this publication lands: remember its bytes (uploads skip
		// them meanwhile) and, verifying, their guest contents now.
		early->ranges = publication_ranges;
		if (early->verify) {
			for (const auto& range: early->ranges) {
				const auto at = early->snapshot.size();
				early->snapshot.resize(at + range.size);
				(void)Libs::LibKernel::Memory::TryReadBacking(range.address, early->snapshot.data() + at,
				                                              range.size);
			}
		}
	}
	const auto publication = BeginBackingPublication(publication_ranges, m_scheduler.CurrentTick());
	m_scheduler.DeferPriorityOperation([this, mapped, offset, total_size, buffer_address, publication,
	                                    early, copies = std::move(copies)] {
		m_download_buffer.Invalidate(offset, total_size);
		if (early != nullptr && early->verify) {
			VerifyEarlyRelease(*early);
		}
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
		}
		EndBackingPublication(publication);
		if (early != nullptr) {
			early->published.store(true, std::memory_order_release);
		}
	});
	return true;
}

void BufferCache::VerifyEarlyRelease(const EarlyReleasedDownload& early) {
	// Priority worker, right before the publication writes these bytes: they must still hold what
	// they held when their pages were released, or a CPU write since is about to be overwritten.
	m_false_sharing_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingVerifyChecks);
	std::vector<uint8_t> current;
	uint64_t             at        = 0;
	uint64_t             conflicts = 0;
	for (const auto& range: early.ranges) {
		current.resize(range.size);
		(void)Libs::LibKernel::Memory::TryReadBacking(range.address, current.data(), range.size);
		if (std::memcmp(current.data(), early.snapshot.data() + at, range.size) != 0) {
			conflicts++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr,
				             "FalseSharingVerify: a CPU write to GPU-owned bytes of a released page "
				             "is overwritten by their publication: addr=0x%016" PRIx64
				             " size=0x%" PRIx64 "\n",
				             range.address, range.size);
			}
		}
		at += range.size;
	}
	if (conflicts == 0) {
		return;
	}
	m_false_sharing_totals.verify_conflicts.fetch_add(conflicts, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingVerifyConflicts, conflicts);
	if (m_false_sharing_verify == 2) {
		EXIT("FalseSharingVerify: a CPU write to GPU-owned bytes of a released page\n");
	}
}

void BufferCache::PruneEarlyReleased() {
	std::erase_if(m_early_released, [](const auto& entry) {
		return entry->published.load(std::memory_order_acquire);
	});
}

bool BufferCache::OverlapsUnpublished(uint64_t address, uint64_t size) const {
	const auto end = address + size;
	for (const auto& entry: m_early_released) {
		if (entry->published.load(std::memory_order_acquire)) {
			continue;
		}
		for (const auto& range: entry->ranges) {
			if (range.address < end && address < range.End()) {
				return true;
			}
		}
	}
	return false;
}

template <typename Emit>
void BufferCache::ForEachPublishedPart(uint64_t address, uint64_t size, Emit&& emit) const {
	// Only called while early releases are pending (a handful of ranges at most).
	const auto              end = address + size;
	std::vector<GuestRange> excluded;
	for (const auto& entry: m_early_released) {
		if (entry->published.load(std::memory_order_acquire)) {
			continue;
		}
		for (const auto& range: entry->ranges) {
			if (range.address < end && address < range.End()) {
				excluded.push_back(range);
			}
		}
	}
	if (excluded.empty()) {
		emit(address, size);
		return;
	}
	std::sort(excluded.begin(), excluded.end(),
	          [](const GuestRange& a, const GuestRange& b) { return a.address < b.address; });
	uint64_t cursor = address;
	for (const auto& range: excluded) {
		if (range.address > cursor) {
			emit(cursor, std::min(range.address, end) - cursor);
		}
		cursor = std::max(cursor, range.End());
		if (cursor >= end) {
			return;
		}
	}
	if (cursor < end) {
		emit(cursor, end - cursor);
	}
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_bda_incremental_sync(IncrementalBdaSyncEnabled()),
      m_bda_hot_sync(m_bda_incremental_sync && BdaHotSyncEnabled()),
      m_memory_tracker(page_manager, m_bda_incremental_sync, BufferFaultPolicy()),
      m_hot_quiet_frames(HotPageQuietFrames()),
      m_hot_check_limit(HotPageCheckLimit()),
      m_range_memo(RangeMemoEnabled() ? std::make_unique<RangeMemo[]>(RangeMemoSlots) : nullptr),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB,
                       graphics.transfer_queue != nullptr),
      m_upload_dma(UploadDma::Create(graphics, scheduler)),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_range_memo_verify = m_range_memo != nullptr ? RangeMemoVerifyMode() : 0;
	m_relaxed_queries = RelaxedQueriesEnabled();
	m_bda_dirty_log   = m_bda_hot_sync && BdaDirtyLogEnabled();
	if (m_bda_dirty_log) {
		m_memory_tracker.EnableDirtiedLog();
		m_bda_log_verify = BdaDirtyLogVerifyMode();
	}
	m_bda_epoch_skip  = SyncEpoch::Enabled() && ParseEnvU64("KYTY_BDA_SYNC_EPOCH", 1) != 0;
	// The verify mode tells guest writes from missed pages by the fault epoch, which the tracker
	// keeps only with incremental BDA synchronization.
	m_bda_epoch_verify = m_bda_epoch_skip && m_bda_incremental_sync ? BdaEpochVerifyMode() : 0;
	if (SyncEpoch::Enabled() && BindingEpochMemoEnabled()) {
		m_binding_memo        = std::make_unique<BindingMemo[]>(BindingMemoSlots);
		m_binding_memo_verify = BindingEpochMemoVerifyMode();
		m_binding_memo_cross  = BindingMemoCrossEpochEnabled();
	}
	m_written_sync_skip = WrittenSyncSkipEnabled();
	if (m_written_sync_skip) {
		m_written_sync_skip_verify = EnvVerifyMode("KYTY_WRITTEN_SYNC_SKIP_VERIFY");
	}
	m_false_sharing = FalseSharingWritesEnabled();
	if (m_false_sharing) {
		m_false_sharing_verify = EnvVerifyMode("KYTY_FALSE_SHARING_WRITES_VERIFY");
	}
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	if (m_upload_dma != nullptr) {
		m_scheduler.SetSubmitDependency(m_upload_dma.get(), 1);
	}
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (SideReadbackEnabled()) {
		// Eager copies reuse the side-readback registry, so they need it.
		const bool eager = EagerReadbackEnabled();
		m_side = std::make_unique<SideReadbackState>(m_graphics, m_scheduler, SideReadbackWindow(),
		                                             eager);
		if (eager) {
			m_eager_enabled      = true;
			m_eager              = EagerReadbackPages(EagerReadbackLimits());
			m_eager_flush_budget = EagerFlushBudget();
		}
	}
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (m_upload_dma != nullptr) {
		// The scheduler has drained by now; no later submission may wait on the upload DMA.
		m_scheduler.SetSubmitDependency(nullptr, 1);
	}
	CompleteAllSideReadbacks();
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size, bool write_fault) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	ForgetKnownFills(vaddr, size);
	const auto flush = [this, vaddr, size] { ReadMemory(vaddr, size, true); };
	if (!write_fault) {
		m_memory_tracker.InvalidateRegion(vaddr, size, flush);
		return;
	}
	// Pages fault-ahead opens take writes without faulting: forget their fills first, exactly as
	// for the faulting range (never GPU-dirty pages, so fills of GPU-owned ranges survive).
	m_memory_tracker.InvalidateRegionOnWriteFault(
	    vaddr, size, flush,
	    [this](uint64_t address, uint64_t bytes) noexcept { ForgetKnownFills(address, bytes); });
}

BufferCache::UploadBatch::UploadBatch(BufferCache& cache): m_cache(cache) {
	m_cache.m_upload_batch_depth++;
}

BufferCache::UploadBatch::~UploadBatch() {
	if (--m_cache.m_upload_batch_depth == 0 && m_cache.m_scheduler.Active()) {
		auto& command = m_cache.m_scheduler.Current();
		if (command.IsInvalid()) {
			return;
		}
		// The scope exists to record its queued uploads behind one barrier pair. Barriers that
		// were pending for other reasons (a previous draw's shader-write barrier, deferrable image
		// transitions) wait for the next flush point as everywhere else: recording them here ended
		// the rendering instance before every draw that followed a storage-writing draw and so
		// defeated KYTY_DRAW_WRITE_SINK (u42 Sky Garden: ~745 instance splits per flip).
		if (UploadBatchScopedFlushEnabled() && !command.HasPendingUploads()) {
			if (command.HasPendingBarriers()) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::UploadBatchFlushesDeferred);
			}
			return;
		}
		command.FlushBarriers();
	}
}

void BufferCache::AdvanceFrame() noexcept {
	m_memory_tracker.AdvanceFrame();
}

void BufferCache::EraseHotShadows(uint64_t vaddr, uint64_t size) {
	if (m_hot_shadows.empty()) {
		return;
	}
	auto it = m_hot_shadows.lower_bound(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
	while (it != m_hot_shadows.end() && it->first < vaddr + size) {
		it = m_hot_shadows.erase(it);
	}
}

void BufferCache::SettleHotPages(uint64_t vaddr, uint64_t size) {
	if (m_memory_tracker.HotPageCount() == 0) {
		return;
	}
	for (const auto page: m_memory_tracker.SettleHotPages(vaddr, size)) {
		// The page is write-protected now: its contents can no longer change unobserved. If they
		// differ from the last upload (or nothing was uploaded while hot), the CPU wrote since.
		const auto shadow = m_hot_shadows.find(page);
		if (shadow == m_hot_shadows.end() ||
		    std::memcmp(shadow->second.data.get(), reinterpret_cast<const void*>(page),
		                TRACKER_PAGE_SIZE) != 0) {
			m_memory_tracker.MarkRegionAsCpuModified(page, TRACKER_PAGE_SIZE);
		}
		if (shadow != m_hot_shadows.end()) {
			m_hot_shadows.erase(shadow);
		}
	}
}

void BufferCache::MaintainHotPages() {
	const auto frame = m_memory_tracker.Frame();
	if (frame - m_hot_sweep_frame < 8) {
		return;
	}
	m_hot_sweep_frame = frame;
	m_memory_tracker.SweepHotPages(m_hot_quiet_frames);
	std::erase_if(m_hot_shadows, [this, frame](const auto& entry) {
		return frame - entry.second.last_use > m_hot_quiet_frames;
	});
}

void BufferCache::CollectHotPages(Buffer& buffer, std::span<const GuestRange> hot_ranges,
                                  std::vector<vk::BufferCopy>& copies, uint64_t& total_size,
                                  std::vector<uint64_t>& demote, std::vector<uint64_t>& settle) {
	uint64_t hot_bytes = 0;
	for (const auto& range: hot_ranges) {
		hot_bytes += range.size;
	}
	if (m_hot_scratch.size() < hot_bytes) {
		m_hot_scratch.resize(hot_bytes);
	}
	const auto frame     = m_memory_tracker.Frame();
	const auto max_pages = m_memory_tracker.GetFaultPolicy().hot_max;
	uint64_t   staged    = 0;
	uint64_t   visited   = 0;
	uint64_t   skipped   = 0;
	uint64_t   settled   = 0;
	for (const auto& range: hot_ranges) {
		// The shadows of a run's pages, in address order: one lookup per run instead of one per
		// page. `next` stays the first shadow above the current page (an emplace below inserts
		// before it and keeps it valid).
		auto next = m_hot_shadows.lower_bound(range.address);
		for (auto page = range.address; page < range.End(); page += TRACKER_PAGE_SIZE) {
			visited++;
			auto shadow = m_hot_shadows.end();
			if (next != m_hot_shadows.end() && next->first == page) {
				shadow = next++;
			}
			const auto unchanged = [&](const void* contents) {
				return shadow != m_hot_shadows.end() &&
				       std::memcmp(shadow->second.data.get(), contents, TRACKER_PAGE_SIZE) == 0;
			};
			auto* snapshot = m_hot_scratch.data() + staged;
			// KYTY_BDA_HOT_SYNC: nearly every visit finds the page unchanged, so compare the live
			// (hot, hence never GPU-dirty and readable) page first and snapshot only one that
			// differs. A write racing either compare is seen by the next upload, whose shadow still
			// holds the old contents, exactly as a write landing right after a snapshot.
			bool same = m_bda_hot_sync && unchanged(reinterpret_cast<const void*>(page));
			if (!same) {
				// Snapshot the (writable) page once: the compare, the shadow and the upload all use
				// this copy, so the shadow always equals what the buffer receives.
				std::memcpy(snapshot, reinterpret_cast<const void*>(page), TRACKER_PAGE_SIZE);
				same = unchanged(snapshot);
			}
			if (same) {
				skipped++;
				auto& state    = shadow->second;
				state.last_use = frame;
				if (frame - state.last_change > m_hot_quiet_frames) {
					demote.push_back(page);
				} else if (m_hot_check_limit != 0 && ++state.unchanged_checks >= m_hot_check_limit) {
					// Checked far more often than written: back to fault tracking, clean (the
					// buffer holds exactly the shadow, which SettleHotPages compares again once the
					// page is write-protected).
					settle.push_back(page);
					settled++;
				}
				continue;
			}
			if (!m_early_released.empty() && OverlapsUnpublished(page, TRACKER_PAGE_SIZE)) {
				// KYTY_FALSE_SHARING_WRITES: GPU-owned bytes of this page still wait for their
				// publication, and the buffer holds them: upload only the other bytes of the
				// snapshot, and return the page to normal tracking (no shadow of mixed contents).
				ForEachPublishedPart(page, TRACKER_PAGE_SIZE, [&](uint64_t part, uint64_t bytes) {
					copies.emplace_back(total_size + (part - page), buffer.Offset(part), bytes);
				});
				m_false_sharing_totals.upload_splits++;
				Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingUploadSplits);
				demote.push_back(page);
				total_size += TRACKER_PAGE_SIZE;
				staged += TRACKER_PAGE_SIZE;
				continue;
			}
			if (shadow == m_hot_shadows.end()) {
				if (m_hot_shadows.size() < max_pages) {
					shadow = m_hot_shadows.emplace(page, HotShadow {}).first;
					shadow->second.data = std::make_unique<uint8_t[]>(TRACKER_PAGE_SIZE);
				} else {
					// No shadow to compare against: back to faulting on writes.
					demote.push_back(page);
				}
			}
			if (shadow != m_hot_shadows.end()) {
				std::memcpy(shadow->second.data.get(), snapshot, TRACKER_PAGE_SIZE);
				shadow->second.last_change      = frame;
				shadow->second.last_use         = frame;
				shadow->second.unchanged_checks = 0;
			}
			copies.emplace_back(total_size, buffer.Offset(page), TRACKER_PAGE_SIZE);
			total_size += TRACKER_PAGE_SIZE;
			staged += TRACKER_PAGE_SIZE;
		}
	}
	MemoryStats::Count(MemoryStats::Counter::HotUploadPages, visited);
	MemoryStats::Count(MemoryStats::Counter::HotUploadSkipped, skipped);
	if (settled != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::HotPageCheckSettles, settled);
		m_range_memo_totals.settles += settled;
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::ReadMemory);
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid readback range\n");
	}
	const auto      trace_start = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
	ReadMemoryTrace trace;
	const auto      record = [&](std::optional<HangTrace::ReadbackKind> kind) {
		if (!HangTrace::Enabled()) {
			return;
		}
		const auto previous = HangTrace::GetReadbackKind();
		if (kind) {
			HangTrace::SetReadbackKind(*kind);
		}
		HangTrace::RecordReadback(vaddr, size, trace.begin, trace.size, trace.downloaded,
		                          HangTrace::NowNs() - trace_start);
		HangTrace::SetReadbackKind(previous);
	};
	auto&      gpu        = m_scheduler.Context().GetGpu();
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);

	// Writes need the CPU-dirty transition of the drain path. Guest-thread reads use side copies;
	// GPU-thread reads do too (KYTY_READBACK_SIDE_GPU_THREAD, default on) and wait for the copy
	// themselves: it waits only for the producing recording, not for the current one, and the
	// current recording is neither split nor submitted.
	const bool gpu_thread = GuestGpu::IsGpuThread();
	const bool side_path =
	    m_side != nullptr && !is_write && (!gpu_thread || SideReadbackGpuThreadEnabled());
	if (OverlapsPendingSideReadback(page_begin, page_end)) {
		// Another fault already copies these pages (or an eager copy publishes them): wait for
		// (or finish) its publication instead of copying again. Writes and GPU-thread reads must
		// also be ordered after it.
		uint32_t eager = 0;
		{
			Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
			eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
		}
		if (eager != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerWaits);
		}
		// A GPU-thread read continues below: bytes re-dirtied since that copy was issued still
		// need their own readback before the caller reads.
		if (side_path && !gpu_thread) {
			// If a newer writer re-dirtied the page meanwhile it stays protected, and the
			// retried access faults into a fresh readback.
			if (eager == 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideDuplicateWaits);
			}
			record(eager != 0 ? HangTrace::ReadbackKind::FaultReadEager
			                  : HangTrace::ReadbackKind::FaultReadDuplicate);
			return;
		}
		if (side_path && eager != 0 &&
		    !m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
			// The eager copy published every GPU-owned byte of these pages: nothing is left.
			record(std::nullopt);
			return;
		}
	}
	if (!side_path) {
		gpu.SendCommandSync([&, this, vaddr, size, is_write, gpu_thread] {
			if (!is_write) {
				NoteEagerRead(vaddr, size, gpu_thread);
			}
			ReadMemoryDrain(vaddr, size, is_write, trace);
		});
		record(std::nullopt);
		return;
	}

	if (gpu_thread) {
		NoteEagerRead(vaddr, size, true);
		std::shared_ptr<SideReadback> issued;
		const auto                    result = TryIssueSideReadback(vaddr, size, issued);
		if (result == SideIssueResult::Issued) {
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::GpuWaitSideCopy);
				CompleteSideReadback(*issued);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackGpuThreadSideCopies);
			trace.begin      = issued->begin;
			trace.size       = issued->end - issued->begin;
			trace.downloaded = true;
			record(std::nullopt);
			return;
		}
		switch (result) {
			case SideIssueResult::CurrentWriter:
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackCurrentWriter);
				break;
			case SideIssueResult::Unbounded:
				Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackUnbounded);
				break;
			default: Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackOther); break;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackGpuThreadDrains);
		ReadMemoryDrain(vaddr, size, false, trace);
		record(std::nullopt);
		return;
	}

	std::shared_ptr<SideReadback> issued;
	auto                          result = SideIssueResult::Other;
	gpu.SendCommandSync([&, this, vaddr, size] {
		NoteEagerRead(vaddr, size, false);
		result = TryIssueSideReadback(vaddr, size, issued);
		if (result != SideIssueResult::Issued && result != SideIssueResult::Pending) {
			ReadMemoryDrain(vaddr, size, false, trace);
		}
	});
	switch (result) {
		case SideIssueResult::Issued: {
			// The GPU thread returned right after submitting; this guest thread waits.
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				CompleteSideReadback(*issued);
			}
			trace.begin      = issued->begin;
			trace.size       = issued->end - issued->begin;
			trace.downloaded = true;
			record(HangTrace::ReadbackKind::FaultReadSide);
			return;
		}
		case SideIssueResult::Pending: {
			// Another thread's side copy (or an eager copy) of the faulting page was issued after
			// the check above.
			uint32_t eager = 0;
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
			}
			Profiler::CountFrameEvent(eager != 0 ? Profiler::FrameEvent::ReadbackEagerWaits
			                                     : Profiler::FrameEvent::ReadbackSideDuplicateWaits);
			record(eager != 0 ? HangTrace::ReadbackKind::FaultReadEager
			                  : HangTrace::ReadbackKind::FaultReadDuplicate);
			return;
		}
		case SideIssueResult::CurrentWriter:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackCurrentWriter);
			break;
		case SideIssueResult::Unbounded:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackUnbounded);
			break;
		case SideIssueResult::Other:
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideFallbackOther);
			break;
	}
	record(std::nullopt);
}

void BufferCache::ReadMemoryDrain(uint64_t vaddr, uint64_t size, bool is_write,
                                  ReadMemoryTrace& trace) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	if (is_write && m_false_sharing && TryFalseSharingWrite(vaddr, size, trace)) {
		return;
	}
	auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

	// Widen nearby CPU reads so they share one GPU drain.
	const uint64_t     WindowSize   = is_write ? WriteFaultWindow() : 512 * 1024;
	const auto         buffer_begin = buffer.CpuAddress();
	const auto         buffer_end   = buffer_begin + buffer.Size();
	const auto window_begin = std::max(Common::AlignDown(vaddr, WindowSize), buffer_begin);
	const auto window_end = std::min(std::max(window_begin + WindowSize, vaddr + size), buffer_end);

	trace.begin = window_begin;
	trace.size  = window_end - window_begin;
	if (DownloadBufferMemory(buffer, window_begin, window_end - window_begin)) {
		trace.downloaded = true;
		const auto                    tick = m_scheduler.CurrentTick();
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitDrain);
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
	}
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

bool BufferCache::TryFalseSharingWrite(uint64_t vaddr, uint64_t size, ReadMemoryTrace& trace) {
	// KYTY_FALSE_SHARING_WRITES (bufferCache.h). The written bytes must be none the GPU owns...
	if (m_gpu_modified_ranges.Intersects(vaddr, size) || OverlapsUnpublished(vaddr, size)) {
		return false;
	}
	// ...and no address-writing shader may still be writing bytes nobody tracks.
	if (m_unbounded_write_tick != 0 && !m_scheduler.IsFree(m_unbounded_write_tick)) {
		return false;
	}
	// Never create a buffer here: GPU-dirty bytes always live in a registered buffer, and one
	// buffer owns every caching page it overlaps, hence the whole tracker page.
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return false;
	}
	auto&      buffer     = m_slot_buffers[*owner];
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!buffer.IsInBounds(page_begin, page_end - page_begin)) {
		return false;
	}
	// An image over the pages would refresh from guest memory once invalidated by this fault.
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		if (!m_texture_cache.FindImagesInRegion(page_begin, page_end - page_begin, true).empty()) {
			return false;
		}
	}
	// Pages the GPU owns without exact bytes (none to preserve) take the ordinary path, which
	// then releases them without waiting.
	if (!m_gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
		return false;
	}
	auto early    = std::make_shared<EarlyReleasedDownload>();
	early->verify = m_false_sharing_verify != 0;
	if (!DownloadBufferMemory(buffer, page_begin, page_end - page_begin, early)) {
		return false;
	}
	// Release the pages now: the download above is recorded behind every GPU writer of their
	// bytes, and its publication writes only those bytes. Uploads skip them until it lands.
	m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, page_end - page_begin);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	uint64_t bytes = 0;
	for (const auto& range: early->ranges) {
		bytes += range.size;
	}
	m_early_released.push_back(std::move(early));
	// Submit the recording that copies them soon (within the eager-flush budget): the sooner it
	// completes, the shorter the time guest reads of those bytes see their old contents.
	if (m_eager_flush_budget != 0) {
		m_eager_flush = true;
	}
	m_false_sharing_totals.writes++;
	m_false_sharing_totals.bytes += bytes;
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingWrites);
	Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingBytes, bytes);
	trace.begin      = page_begin;
	trace.size       = page_end - page_begin;
	trace.downloaded = true;
	return true;
}

BufferCache::SideIssueResult BufferCache::TryIssueSideReadback(
    uint64_t vaddr, uint64_t size, std::shared_ptr<SideReadback>& issued) {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_side == nullptr);
	auto&      side    = *m_side;
	const auto current = m_scheduler.CurrentTick();
	// An address-writing shader in the unsubmitted recording may write any buffer byte.
	if (m_unbounded_write_tick == current) {
		return SideIssueResult::Unbounded;
	}
	// Why a read is refused as Other (ReadbackSideOther* counters).
	const auto other = [](Profiler::FrameEvent cause) {
		Profiler::CountFrameEvent(cause);
		return SideIssueResult::Other;
	};
	// Never create a buffer here: GPU-dirty bytes always live in a registered buffer.
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return other(Profiler::FrameEvent::ReadbackSideOtherOwner);
	}
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return other(Profiler::FrameEvent::ReadbackSideOtherOwner);
	}
	const auto buffer_begin = buffer.CpuAddress();
	const auto buffer_end   = buffer_begin + buffer.Size();
	const auto page_begin   = std::max(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE), buffer_begin);
	const auto page_end = std::min(Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE), buffer_end);
	const auto aligned  = Common::AlignDown(vaddr, side.window);
	const auto window_begin = std::max(aligned, buffer_begin);
	const auto window_end   = std::min(aligned + side.window, buffer_end);
	if (((page_begin | page_end | window_begin | window_end) % TRACKER_PAGE_SIZE) != 0) {
		return other(Profiler::FrameEvent::ReadbackSideOtherAlignment);
	}

	// Prefer the aligned window (neighbouring polled values share one copy); fall back to the
	// faulting pages when a window byte is not eligible (e.g. written by the current recording).
	struct Candidate {
		uint64_t begin;
		uint64_t end;
	};
	const std::array<Candidate, 2> candidates {{{window_begin, window_end}, {page_begin, page_end}}};
	std::vector<GuestRange>        dirty;
	std::optional<Candidate>       chosen;
	uint64_t                       producer = 0;
	auto                           reason   = SideIssueResult::Other;
	// The cause of an Other result: no candidate fits the window unless one says otherwise.
	auto other_cause = Profiler::FrameEvent::ReadbackSideOtherWindow;
	for (size_t index = 0; index < candidates.size(); ++index) {
		const auto& candidate = candidates[index];
		if (index != 0 && candidate.begin == candidates[0].begin &&
		    candidate.end == candidates[0].end) {
			break;
		}
		if (candidate.begin > page_begin || candidate.end < page_end ||
		    candidate.end - candidate.begin > side.window) {
			continue;
		}
		if (OverlapsPendingSideReadback(candidate.begin, candidate.end)) {
			reason = SideIssueResult::Pending;
			continue;
		}
		// A queued drain/texture publication decides these bytes' final backing contents;
		// publishing next to it could reorder the backing writes.
		if (HasPendingBackingPublication(candidate.begin, candidate.end - candidate.begin)) {
			reason      = SideIssueResult::Other;
			other_cause = Profiler::FrameEvent::ReadbackSideOtherPublication;
			continue;
		}
		dirty.clear();
		uint64_t newest = m_write_tick_floor;
		m_gpu_modified_ranges.ForEachInRange(
		    candidate.begin, candidate.end - candidate.begin, [&](uint64_t start, uint64_t end) {
			    dirty.push_back({start, end - start});
			    newest = std::max(newest, m_write_ticks.MaxTick(start, end - start));
		    });
		if (dirty.empty()) {
			reason      = SideIssueResult::Other;
			other_cause = Profiler::FrameEvent::ReadbackSideOtherNoDirty;
			continue;
		}
		if (newest >= current) {
			reason = SideIssueResult::CurrentWriter;
			continue;
		}
		chosen   = candidate;
		producer = newest;
		break;
	}
	if (!chosen && reason == SideIssueResult::Other) {
		Profiler::CountFrameEvent(other_cause);
	}
	if (!chosen) {
		return reason;
	}

	uint32_t slot_index = SideReadbackState::SlotCount;
	for (uint32_t index = 0; index < SideReadbackState::SlotCount; ++index) {
		if (!side.slots[index].busy.load(std::memory_order_acquire)) {
			slot_index = index;
			break;
		}
	}
	if (slot_index == SideReadbackState::SlotCount) {
		return other(Profiler::FrameEvent::ReadbackSideOtherSlot);
	}
	auto&      slot         = side.slots[slot_index];
	const auto staging_base = uint64_t {slot_index} * side.window;

	auto readback    = std::make_shared<SideReadback>();
	readback->begin  = chosen->begin;
	readback->end    = chosen->end;
	readback->slot   = slot_index;
	readback->ranges = dirty;
	std::vector<vk::BufferCopy> copies;
	copies.reserve(dirty.size());
	uint64_t bytes = 0;
	for (const auto& range: dirty) {
		copies.emplace_back(buffer.Offset(range.address),
		                    staging_base + (range.address - chosen->begin), range.size);
		bytes += range.size;
	}

	// Ownership of the exact dirty bytes moves to the publication registered below, exactly as
	// in DownloadBufferMemory. The tracker pages stay GPU-owned (protected) until completion.
	for (const auto& range: dirty) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtySubtract);
	}
	for (const auto& range: dirty) {
		m_gpu_modified_ranges.Subtract(range.address, range.size);
	}
	m_memory_tracker.MarkReadbackPending(chosen->begin, chosen->end - chosen->begin);
	readback->publication = BeginBackingPublication(dirty, producer);

	const auto                 command = slot.command;
	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(command.begin(&begin_info), "begin side-readback command buffer");
	// The timeline wait below orders the producer (and, on the side queue, the newest unbounded
	// writer). On a shared queue this barrier's first scope additionally covers every earlier
	// submission, so the copy observes all submitted work. On the side queue later queue-0 work
	// can overwrite copied bytes while the copy runs; such a writer re-adds them as GPU-dirty
	// and keeps their pages protected (UnmarkReadbackPending retains them), so the possibly
	// torn published bytes are never read before a newer readback replaces them.
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = buffer.Offset(chosen->begin);
	before.size                = chosen->end - chosen->begin;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                        nullptr);
	command.copyBuffer(buffer.Handle(), side.staging->Handle(), static_cast<uint32_t>(copies.size()),
	                   copies.data());
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	after.buffer                  = side.staging->Handle();
	after.offset                  = staging_base;
	after.size                    = side.window;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	RequireVulkanSuccess(command.end(), "end side-readback command buffer");

	const auto value = ++side.next_value;
	readback->value  = value;
	slot.busy.store(true, std::memory_order_relaxed);

	// On the separate side queue nothing but the timeline wait orders the copy after queue-0
	// work, so it must also cover the newest unbounded (address) writer, whose bytes carry no
	// writer tick. It is older than `current` here (checked above), hence already submitted.
	const bool side_queue = m_graphics.side_queue != nullptr;
	if (side_queue) {
		producer = std::max(producer, m_unbounded_write_tick);
	}
	const auto                      master     = m_scheduler.GetMasterSemaphore().Handle();
	const uint64_t                  wait_value = producer;
	const vk::PipelineStageFlags    wait_stage = vk::PipelineStageFlagBits::eTransfer;
	const uint32_t                  wait_count = producer != 0 ? 1u : 0u;
	vk::TimelineSemaphoreSubmitInfo timeline {};
	timeline.waitSemaphoreValueCount   = wait_count;
	timeline.pWaitSemaphoreValues      = &wait_value;
	timeline.signalSemaphoreValueCount = 1;
	timeline.pSignalSemaphoreValues    = &value;
	vk::SubmitInfo submit {};
	submit.pNext                = &timeline;
	submit.waitSemaphoreCount   = wait_count;
	submit.pWaitSemaphores      = &master;
	submit.pWaitDstStageMask    = &wait_stage;
	submit.commandBufferCount   = 1;
	submit.pCommandBuffers      = &command;
	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores    = &side.semaphore;
	vk::Result submit_result;
	// KYTY_CP_RECORDER: older ticks may still be in the recorder's ring. Hand them to the queue or
	// the broker first (without queue_mutex, which the recorder may need): the producer before a
	// side-queue wait on it, every older tick before a copy on the shared queue.
	m_scheduler.WaitRecorded(side_queue ? producer : current - 1);
	if (side_queue) {
		// Every tick older than the current recording was handed to queue 0 or to the submission
		// broker. Drain the broker only if the producer has not reached the driver yet, so this
		// wait is never submitted ahead of its signal.
		const auto& progress = m_scheduler.GetMasterSemaphore().GetSubmissionProgress();
		if (progress != nullptr &&
		    progress->dispatched_tick.load(std::memory_order_acquire) < producer) {
			Common::LockGuard lock(m_graphics.queue_mutex);
			m_graphics.submission_queue.DrainPendingLocked();
		}
		Common::LockGuard lock(m_graphics.side_queue_mutex);
		submit_result = m_graphics.side_queue.submit(1, &submit, nullptr);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideQueueCopies);
	} else {
		// Every tick older than the current recording was handed to the queue or to the
		// submission broker (drained here first), so the producer is submitted before this
		// copy waits on it. The current recording follows this copy in submission order.
		Common::LockGuard lock(m_graphics.queue_mutex);
		m_graphics.submission_queue.DrainPendingLocked();
		submit_result = m_graphics.queue.submit(1, &submit, nullptr);
	}
	RequireVulkanSuccess(submit_result, "submit side readback");

	{
		std::lock_guard lock(side.pending_mutex);
		side.pending.push_back(readback);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideCopyBytes, bytes);
	issued = std::move(readback);
	return SideIssueResult::Issued;
}

bool BufferCache::CompleteSideReadback(SideReadback& readback) {
	std::scoped_lock lock(readback.mutex);
	if (readback.done.load(std::memory_order_acquire)) {
		return false;
	}
	auto&    side         = *m_side;
	Buffer*  staging      = nullptr;
	uint64_t staging_base = 0;
	if (readback.eager) {
		// The copy ends the recording of master tick `value`, behind all of its commands. A GPU
		// thread (CP) reader waiting here waits for that recording, never the current one.
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitSideCopy);
		m_scheduler.GetMasterSemaphore().Wait(readback.value);
		staging      = side.eager_staging.get();
		staging_base = uint64_t {readback.slot} * TRACKER_PAGE_SIZE;
	} else {
		side.Wait(readback.value);
		staging      = side.staging.get();
		staging_base = uint64_t {readback.slot} * side.window;
	}
	staging->Invalidate(staging_base, readback.end - readback.begin);
	const auto* staged = staging->Mapped().data() + staging_base;
	for (const auto& range: readback.ranges) {
		Libs::LibKernel::Memory::WriteBacking(range.address,
		                                      staged + (range.address - readback.begin), range.size);
	}
	EndBackingPublication(readback.publication);
	// Only pages whose pending mark survived lose GPU ownership: a newer recorded writer (or any
	// other GPU transition) since the issue keeps its page protected for a new readback.
	const auto unmark =
	    m_memory_tracker.UnmarkReadbackPending(readback.begin, readback.end - readback.begin);
	if (readback.eager) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerPagesUnmarked,
		                          unmark.unmarked_pages);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerPagesRetained,
		                          unmark.retained_pages);
		side.eager_busy[readback.slot].store(false, std::memory_order_release);
	} else {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSidePagesUnmarked,
		                          unmark.unmarked_pages);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSidePagesRetained,
		                          unmark.retained_pages);
		side.slots[readback.slot].busy.store(false, std::memory_order_release);
	}
	readback.done.store(true, std::memory_order_release);
	{
		std::lock_guard pending_lock(side.pending_mutex);
		const auto      found =
		    std::find_if(side.pending.begin(), side.pending.end(),
		                 [&readback](const auto& entry) { return entry.get() == &readback; });
		EXIT_IF(found == side.pending.end());
		side.pending.erase(found);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	if (readback.eager && HangTrace::Enabled()) {
		uint64_t bytes = 0;
		for (const auto& range: readback.ranges) {
			bytes += range.size;
		}
		const auto previous = HangTrace::GetReadbackKind();
		HangTrace::SetReadbackKind(HangTrace::ReadbackKind::EagerPublish);
		HangTrace::RecordReadback(readback.begin, bytes, readback.begin,
		                          readback.end - readback.begin, unmark.unmarked_pages != 0,
		                          HangTrace::NowNs() - readback.issue_ns);
		HangTrace::SetReadbackKind(previous);
	}
	return true;
}

bool BufferCache::OverlapsPendingSideReadback(uint64_t begin, uint64_t end) const {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	std::lock_guard lock(m_side->pending_mutex);
	return std::any_of(m_side->pending.begin(), m_side->pending.end(),
	                   [begin, end](const auto& entry) {
		                   return entry->begin < end && begin < entry->end;
	                   });
}

uint32_t BufferCache::CompleteSideReadbacks(uint64_t vaddr, uint64_t size) {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0 ||
	    !GuestRange {vaddr, size}.Valid()) {
		return 0;
	}
	std::vector<std::shared_ptr<SideReadback>> overlapping;
	{
		std::lock_guard lock(m_side->pending_mutex);
		for (const auto& entry: m_side->pending) {
			if (entry->begin < vaddr + size && vaddr < entry->end) {
				overlapping.push_back(entry);
			}
		}
	}
	// Pending entries never overlap each other (issue skips overlapping windows and pages), and
	// are in issue order, so completing them in this order keeps publications in submission
	// order.
	uint32_t eager = 0;
	for (const auto& entry: overlapping) {
		CompleteSideReadback(*entry);
		eager += entry->eager ? 1u : 0u;
	}
	return eager;
}

void BufferCache::CompleteAllSideReadbacks() {
	if (m_side == nullptr || m_side->pending_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	std::vector<std::shared_ptr<SideReadback>> all;
	{
		std::lock_guard lock(m_side->pending_mutex);
		all = m_side->pending;
	}
	for (const auto& entry: all) {
		CompleteSideReadback(*entry);
	}
}

void BufferCache::NoteBufferContentWrite(uint64_t vaddr, uint64_t size) {
	if (m_side == nullptr) {
		return;
	}
	m_write_ticks.Assign(vaddr, size, m_scheduler.CurrentTick());
	if (m_write_ticks.Size() >= m_write_tick_prune_size) {
		// Completed writers need no entry: their ranges read back as the prune floor.
		const auto completed = m_scheduler.GetMasterSemaphore().KnownGpuTick();
		m_write_ticks.Prune(completed);
		m_write_tick_floor      = std::max(m_write_tick_floor, completed);
		m_write_tick_prune_size = std::max<size_t>(1024, m_write_ticks.Size() * 2);
	}
	if (m_eager_enabled && !m_eager.Empty() && m_eager.NoteWrite(vaddr, size) &&
	    m_eager_flush_budget != 0) {
		// A page the command processor reads back: submit this recording once the writer is
		// recorded, so the read after it finds a submitted (ideally finished) producer.
		m_eager_flush = true;
	}
}

void BufferCache::NoteEagerRead(uint64_t vaddr, uint64_t size, bool gpu_thread_reader) {
	if (!m_eager_enabled || size == 0 || size > EagerReadMaxBytes ||
	    !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	const auto frame = m_memory_tracker.Frame();
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = first; page < last; page += TRACKER_PAGE_SIZE) {
		if (!m_eager.IsHot(page)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerHotPages);
		}
		m_eager.NoteRead(page, frame, gpu_thread_reader);
	}
}

void BufferCache::IssueEagerReadbacks() {
	// Called by the command processor right before it submits the current recording, between
	// packets, never from inside another cache operation: a Submit can also happen inside one
	// (e.g. a staging-ring wrap during a written upload, with tracker locks held), where issuing
	// would re-enter the tracker. Submissions without this call simply carry no eager copies.
	if (!m_eager_enabled || !GuestGpu::IsGpuThread() || CommandScheduler::InDeferredOperation() ||
	    !m_scheduler.Active() || m_scheduler.Current().IsInvalid()) {
		return;
	}
	// This submission is the early flush a recorded writer may have asked for.
	m_eager_flush = false;
	if (m_eager.Empty()) {
		return;
	}
	const auto frame = m_memory_tracker.Frame();
	m_eager.Sweep(frame);
	if (m_eager.Candidates() == 0) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	m_eager.IssueCandidates(frame, [this, tick](uint64_t page) {
		const auto result = TryIssueEagerReadback(page, tick);
		if (result == EagerReadbackPages::IssueResult::Retry) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerRetries);
		}
		return result;
	});
}

EagerReadbackPages::IssueResult BufferCache::TryIssueEagerReadback(uint64_t page, uint64_t tick) {
	using Result = EagerReadbackPages::IssueResult;
	auto&      side = *m_side;
	const auto end  = page + TRACKER_PAGE_SIZE;
	const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner) ||
	    !m_slot_buffers[*owner].IsInBounds(page, TRACKER_PAGE_SIZE)) {
		return Result::Drop;
	}
	auto& buffer = m_slot_buffers[*owner];
	// One publication per byte at a time, published in issue order (as for side readbacks).
	if (OverlapsPendingSideReadback(page, end) ||
	    HasPendingBackingPublication(page, TRACKER_PAGE_SIZE)) {
		return Result::Retry;
	}
	std::vector<GuestRange> dirty;
	uint64_t                newest = m_write_tick_floor;
	uint64_t                bytes  = 0;
	m_gpu_modified_ranges.ForEachInRange(page, TRACKER_PAGE_SIZE, [&](uint64_t start, uint64_t stop) {
		dirty.push_back({start, stop - start});
		newest = std::max(newest, m_write_ticks.MaxTick(start, stop - start));
		bytes += stop - start;
	});
	if (dirty.empty()) {
		return Result::Drop;
	}
	// Only bytes whose writers are already submitted: a writer registered by this recording may
	// record its shader command after this submission (a submit in the middle of a draw's
	// preparation), so its bytes wait for the next submission. That is the side-readback rule;
	// here the copy additionally follows the whole of this recording in queue order.
	if (newest >= tick) {
		return Result::Retry;
	}
	// Dirty bytes keep their tracker page GPU-owned (MemoryTracker validates the pairing).
	if (!m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
		return Result::Drop;
	}
	uint32_t slot = SideReadbackState::EagerSlotCount;
	for (uint32_t index = 0; index < SideReadbackState::EagerSlotCount; ++index) {
		if (!side.eager_busy[index].load(std::memory_order_acquire)) {
			slot = index;
			break;
		}
	}
	if (slot == SideReadbackState::EagerSlotCount) {
		return Result::Retry;
	}
	const auto staging_base = uint64_t {slot} * TRACKER_PAGE_SIZE;

	auto readback      = std::make_shared<SideReadback>();
	readback->begin    = page;
	readback->end      = end;
	readback->value    = tick;
	readback->slot     = slot;
	readback->eager    = true;
	readback->issue_ns = HangTrace::Enabled() ? HangTrace::NowNs() : 0;
	readback->ranges   = dirty;
	std::vector<vk::BufferCopy> copies;
	copies.reserve(dirty.size());
	for (const auto& range: dirty) {
		copies.emplace_back(buffer.Offset(range.address), staging_base + (range.address - page),
		                    range.size);
	}

	// Recorded last in this recording: the barrier's first scope covers every earlier command on
	// the queue, including all writers of these bytes (checked above to be submitted already).
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = buffer.Offset(page);
	before.size                = TRACKER_PAGE_SIZE;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), side.eager_staging->Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	after.buffer                  = side.eager_staging->Handle();
	after.offset                  = staging_base;
	after.size                    = TRACKER_PAGE_SIZE;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);

	// Ownership of the exact dirty bytes moves to the publication registered below, exactly as in
	// TryIssueSideReadback; the page stays GPU-owned (protected) until completion.
	for (const auto& range: dirty) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtySubtract);
	}
	for (const auto& range: dirty) {
		m_gpu_modified_ranges.Subtract(range.address, range.size);
	}
	m_memory_tracker.MarkReadbackPending(page, TRACKER_PAGE_SIZE);
	readback->publication = BeginBackingPublication(dirty, tick);
	side.eager_busy[slot].store(true, std::memory_order_relaxed);
	{
		std::lock_guard lock(side.pending_mutex);
		side.pending.push_back(readback);
		side.pending_count.store(side.pending.size(), std::memory_order_release);
	}
	// Queued for this recording's tick: the completion runner publishes it once the recording
	// has finished, unless a reader (guest fault, CP read) completed it first.
	m_scheduler.DeferPriorityOperation(
	    [this, readback = std::move(readback)] { CompleteSideReadback(*readback); });
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerCopies);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerCopyBytes, bytes);
	return Result::Issued;
}

bool BufferCache::TakeEagerFlushRequest(bool in_rendering) {
	if (!m_eager_flush || in_rendering) {
		return false;
	}
	m_eager_flush    = false;
	const auto frame = m_memory_tracker.Frame();
	if (frame != m_eager_flush_frame) {
		m_eager_flush_frame = frame;
		m_eager_flushes     = 0;
	}
	if (m_eager_flushes >= m_eager_flush_budget) {
		return false;
	}
	m_eager_flushes++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerFlushes);
	return true;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, PageTable::kAddressSpaceSize - end);
			}
			if (expands_right) {
				const auto minimum = CACHING_PAGESIZE * 2;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	const bool joined = overlap.first != overlap.last;
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	if (joined) {
		// The joined contents (including GPU-dirty bytes) are copied by this recording.
		NoteBufferContentWrite(overlap.begin, overlap.end - overlap.begin);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer, BdaSyncStats* stats,
                                    const char* upload_reason) {
	KYTY_GPU_OP_SITE("buffercache.upload");
	// KYTY_TRACKER_RELAXED_QUERIES: a read-only synchronization of a range without a CPU-dirty
	// page (hot pages are CPU-dirty too) collects and uploads nothing. Texel reads also download
	// GPU-written images; the BDA hot-pass verification scans in full.
	if (!is_written && !is_texel_buffer && (stats == nullptr || stats->verify_fault_epoch == 0) &&
	    RelaxedNothingToUpload(vaddr, size)) {
		return false;
	}
	// KYTY_WRITTEN_SYNC_SKIP (bufferCache.h): a range the GPU already owns entirely. (Only texel
	// READS do more than the tracker work below.)
	bool written_skip_verify = false;
	if (is_written && stats == nullptr && m_written_sync_skip && GuestGpu::IsGpuThread() &&
	    m_memory_tracker.IsRangeGpuOwned(vaddr, size)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkips);
		m_written_sync_totals.skips++;
		if (m_written_sync_skip_verify == 0) {
			return false;
		}
		written_skip_verify = true;
	}
	// KYTY_BUFFER_RANGE_MEMO: a read-only synchronization of a range that is still Clean does
	// nothing (bufferCache.h). Texel reads also download GPU-written images (not tracker state);
	// the BDA hot-pass verification scans in full.
	uint64_t memo_signature = 0;
	uint64_t memo_dirtied   = 0; // verify mode: dirtying serials before the lookup
	bool     memo_verify    = false;
	const bool memo_applies = !is_written && !is_texel_buffer && m_range_memo != nullptr &&
	                          (stats == nullptr || stats->verify_fault_epoch == 0);
	if (memo_applies) {
		if (m_range_memo_verify != 0) {
			memo_dirtied = m_memory_tracker.RangeDirtiedSignature(vaddr, size);
		}
		memo_signature = m_memory_tracker.RangeSignature(vaddr, size);
		const auto& memo = RangeMemoSlot(vaddr, size);
		if (memo_signature != 0 && memo.signature == memo_signature && memo.vaddr == vaddr &&
		    memo.size == size && memo.fact == RangeFact::Clean) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoCleanHits);
			m_range_memo_totals.clean_hits++;
			if (m_range_memo_verify == 0) {
				return false;
			}
			memo_verify = true;
			RunRangeMemoVerifyHook(vaddr, size);
		}
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	// KYTY_FALSE_SHARING_WRITES: bytes an early release left to their publication are the GPU's
	// in the buffer; CPU-dirty page runs over them upload only the other bytes until it lands.
	if (!m_early_released.empty()) {
		PruneEarlyReleased();
	}
	const bool exclude_unpublished = !m_early_released.empty();
	const auto append_copy = [&](std::vector<vk::BufferCopy>& list, uint64_t& list_size,
	                             uint64_t address, uint64_t bytes) noexcept {
		if (!exclude_unpublished) {
			list.emplace_back(list_size, buffer.Offset(address), bytes);
			list_size += bytes;
			return;
		}
		uint64_t emitted = 0;
		ForEachPublishedPart(address, bytes, [&](uint64_t part, uint64_t part_bytes) {
			list.emplace_back(list_size, buffer.Offset(part), part_bytes);
			list_size += part_bytes;
			emitted += part_bytes;
		});
		if (emitted != bytes) {
			m_false_sharing_totals.upload_splits++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::FalseSharingUploadSplits);
		}
	};
	uint8_t* reserved = nullptr;
	uint64_t reserved_offset = 0;
	uint64_t reserved_size = 0;
	if (Common::RendererBatchEnabled() && is_written && size <= MiB &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		// Reserve before entering writable tracker locks. The dirty set is collected
		// again under those locks, so a concurrent CPU write cannot be missed.
		const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
		reserved_size = Common::AlignUp(vaddr + size, CACHING_PAGESIZE) - begin;
		copies.reserve(static_cast<size_t>(reserved_size / CACHING_PAGESIZE));
		std::tie(reserved, reserved_offset) = m_staging_buffer.Map(reserved_size, 4);
	}
	// Hot pages (MemoryTracker) stay CPU-dirty: they are reported separately and copied only when
	// they differ from the shadow of the last copy this buffer received (CollectHotPages).
	std::vector<GuestRange> hot_ranges;
	std::vector<uint64_t>   demote_hot;
	std::vector<uint64_t>   settle_hot;
	size_t                guest_copies = 0;
	uint64_t              host_base    = 0;
	// Written uploads with KYTY_UPLOAD_COPY_OUTSIDE_LOCK: pages re-dirtied by a racing guest write
	// while the main copy ran unlocked, copied again under the tracker locks.
	std::vector<vk::BufferCopy> late_copies;
	uint64_t                    late_size = 0;
	vk::Buffer                  late_source;
	bool                        reserved_committed = false;
	const auto collect = [&](uint64_t address, uint64_t bytes, bool hot) noexcept {
		if (hot) {
			hot_ranges.push_back({address, bytes});
			if (stats != nullptr && stats->hot_ranges != nullptr) {
				// A full BDA pass remembers where its hot pages are (KYTY_BDA_HOT_SYNC).
				stats->hot_ranges->push_back({stats->buffer_id, address, bytes});
			}
			return;
		}
		if (stats != nullptr && stats->verify_fault_epoch != 0 &&
		    m_memory_tracker.FaultMutationEpoch() == stats->verify_fault_epoch &&
		    m_bda_structure_epoch.load(std::memory_order_acquire) == stats->verify_structure_epoch) {
			// Called under the region lock: a transition that dirtied this page published its
			// epoch change before the lock was released, so it would be visible here.
			stats->verify_mismatch_pages += bytes / TRACKER_PAGE_SIZE;
		}
		append_copy(copies, total_size, address, bytes);
	};
	// KYTY_UPLOAD_DMA_HOST_COPY (uploadDma.h): a read upload large enough for the copy engine
	// leaves its guest bytes to the DMA worker. The pages are already clean and write-protected
	// when upload() runs (ForEachUploadRange), as for the copy made here.
	std::vector<UploadHostCopy> host_copies;
	const auto upload = [&]() noexcept {
		// A normal upload replaces whatever a hot page shadow described.
		for (const auto& copy: copies) {
			EraseHotShadows(buffer.CpuAddress() + copy.dstOffset, copy.size);
		}
		guest_copies = copies.size();
		host_base    = total_size;
		if (!hot_ranges.empty()) {
			CollectHotPages(buffer, hot_ranges, copies, total_size, demote_hot, settle_hot);
		}
		if (reserved != nullptr && total_size <= reserved_size && guest_copies == copies.size()) {
			for (auto& copy: copies) {
				std::memcpy(reserved + copy.srcOffset,
				            reinterpret_cast<const void*>(buffer.CpuAddress() + copy.dstOffset),
				            copy.size);
				copy.srcOffset += reserved_offset;
			}
			if (!copies.empty()) source = m_staging_buffer.Handle();
		} else {
			reserved             = nullptr;
			const bool defer_host = !is_written && UploadBatchEnabled() && m_upload_dma != nullptr &&
			                        UploadDmaHostCopyEnabled() && !UploadDmaVerify() &&
			                        m_staging_buffer.IsCoherent() &&
			                        total_size >= m_upload_dma->MinBytes();
			source = UploadCopies(buffer, copies, total_size, guest_copies, m_hot_scratch.data(),
			                      host_base, defer_host ? &host_copies : nullptr);
		}
	};
	if (is_written && m_memory_tracker.GetFaultPolicy().copy_outside_lock) {
		m_memory_tracker.ForEachWrittenUploadRange(
		    vaddr, size,
		    [&](uint64_t address, uint64_t bytes) noexcept { collect(address, bytes, false); },
		    upload,
		    [&](uint64_t address, uint64_t bytes) noexcept {
			    append_copy(late_copies, late_size, address, bytes);
		    },
		    [&]() noexcept {
			    if (late_copies.empty()) {
				    return;
			    }
			    if (reserved != nullptr && source) {
				    // A staging Map() reuses the pending reservation until it is committed.
				    m_staging_buffer.Commit();
				    reserved_committed = true;
			    }
			    late_source = UploadCopies(buffer, late_copies, late_size);
		    });
	} else {
		m_memory_tracker.ForEachUploadRange(vaddr, size, is_written, collect, upload);
	}
	if (written_skip_verify) {
		// Nothing a CPU access could have changed since the check: it waits for this thread.
		Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkipVerifyChecks);
		m_written_sync_totals.verify_checks++;
		if (!copies.empty() || !hot_ranges.empty() || !late_copies.empty()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::WrittenSyncSkipVerifyMismatches);
			m_written_sync_totals.verify_mismatches++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr,
				             "WrittenSyncSkipVerify: a GPU-owned range had pages to upload: "
				             "addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
				             vaddr, size);
			}
			if (m_written_sync_skip_verify == 2) {
				EXIT("WrittenSyncSkipVerify: a GPU-owned range had pages to upload: addr=0x%016" PRIx64
				     " size=0x%" PRIx64 "\n",
				     vaddr, size);
			}
		}
	}
	if (memo_applies) {
		// Nothing collected: no page of the range was CPU-dirty (hot pages are CPU-dirty too).
		const bool collected = !copies.empty() || !hot_ranges.empty();
		if (memo_verify) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyChecks);
			m_range_memo_totals.verify_checks++;
			if (collected) {
				ClassifyRangeMemoDifference("a skipped synchronization found pages to upload",
				                            vaddr, size, memo_dirtied);
			}
		} else if (!collected && memo_signature != 0 &&
		           m_memory_tracker.RangeSignature(vaddr, size) == memo_signature) {
			RecordRangeFact(vaddr, size, memo_signature, RangeFact::Clean);
		}
	}
	for (const auto page: demote_hot) {
		m_memory_tracker.DemoteHotPages(page, TRACKER_PAGE_SIZE);
		EraseHotShadows(page, TRACKER_PAGE_SIZE);
	}
	for (const auto page: settle_hot) {
		SettleHotPages(page, TRACKER_PAGE_SIZE);
	}
	if (reserved != nullptr && source && !reserved_committed) {
		// Source copying and GPU ownership publication stayed consistent (under the tracker locks,
		// or unlocked with the late pass). Flush and ring bookkeeping need no tracker lock and
		// finish before native copy recording.
		m_staging_buffer.Commit();
		Profiler::CountFrameEvent(Profiler::FrameEvent::UploadReservationsOutsideLocks);
	}
	if (source || late_source) {
		if (stats != nullptr) {
			stats->upload_bytes += total_size + late_size;
			stats->upload_copies += copies.size() + late_copies.size();
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::BufferUploadBytes, total_size + late_size);
		if (HangTrace::Enabled()) {
			const char* reason = upload_reason;
			if (reason == nullptr) {
				reason = stats != nullptr    ? "bda-sync"
				         : is_written        ? "written-binding"
				         : is_texel_buffer   ? "texel-read"
				                             : "read-binding";
			}
			// Address is the first uploaded page run; span_bytes the requested range.
			HangTrace::RecordTransfer(HangTrace::TransferKind::BufferUpload, reason, "",
			                          copies.empty() ? vaddr
			                                         : buffer.CpuAddress() + copies.front().dstOffset,
			                          0, static_cast<uint32_t>(copies.size()), 0, total_size, size);
		}
		auto& command = m_scheduler.Current();
		if (UploadBatchEnabled() && !late_source) {
			// One barrier pair per flush for all queued uploads instead of one per upload. Inside
			// an UploadBatch scope the copy waits for the scope end (or an earlier flush point).
			source = StageUploadDma(source, copies, &host_copies);
			command.RequestUploadCopy(source, buffer.Handle(), copies);
			buffer.MarkContentWritten();
			if (m_upload_batch_depth == 0) {
				command.FlushBarriers();
			}
			if (is_texel_buffer && !is_written) {
				return SynchronizeBufferFromImage(buffer, vaddr, size);
			}
			return false;
		}
		if (command.ActiveRenderingSerial() != 0) {
			MemoryStats::Count(MemoryStats::Counter::UploadRenderSplits);
		}
		const bool both = source && late_source;
		MemoryStats::Count(MemoryStats::Counter::UploadCopies, both ? 2 : 1);
		MemoryStats::Count(MemoryStats::Counter::UploadBarriers, both ? 3 : 2);
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		if (source) {
			native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
			                  copies.data());
		}
		if (late_source) {
			if (source) {
				// The late copy rewrites pages of the first one: order the two writes.
				auto between          = before;
				between.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
				native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				                       vk::PipelineStageFlagBits::eTransfer,
				                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &between,
				                       0, nullptr);
			}
			native.copyBuffer(late_source, buffer.Handle(),
			                  static_cast<uint32_t>(late_copies.size()), late_copies.data());
		}
		buffer.MarkContentWritten();
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::StageUploadDma(vk::Buffer source, std::span<vk::BufferCopy> copies,
                                       std::vector<UploadHostCopy>* host_copies) {
	// Staging bytes UploadCopies left to the DMA worker, written here when the copy is not
	// staged after all (the graphics copies then read the staging ring directly).
	const auto write_host_copies = [host_copies] {
		if (host_copies != nullptr) {
			for (const auto& copy: *host_copies) {
				std::memcpy(copy.destination, copy.source, static_cast<size_t>(copy.size));
			}
			host_copies->clear();
		}
	};
	if (m_upload_dma == nullptr || source != m_staging_buffer.Handle() || copies.empty()) {
		write_host_copies();
		return source;
	}
	// The staged bytes of one upload are one packed range of the ring (SynchronizeBuffer and
	// UploadCopies place them back to back); the transfer copies that whole range.
	uint64_t begin = UINT64_MAX;
	uint64_t end   = 0;
	for (const auto& copy: copies) {
		begin = std::min(begin, copy.srcOffset);
		end   = std::max(end, copy.srcOffset + copy.size);
	}
	const auto size        = end - begin;
	const auto ring_offset = m_upload_dma->Stage(source, begin, size, host_copies);
	if (!ring_offset.has_value()) {
		write_host_copies();
		return source;
	}
	for (auto& copy: copies) {
		copy.srcOffset = copy.srcOffset - begin + *ring_offset;
	}
	if (UploadDmaVerify()) {
		// The ring bytes as this recording's graphics copies read them (the whole command buffer
		// runs after the transfer, and the ring range is not reused before this tick completes).
		auto snapshot = std::make_shared<std::vector<uint8_t>>(
		    m_staging_buffer.Mapped().data() + begin, m_staging_buffer.Mapped().data() + end);
		auto readback = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                         vk::BufferUsageFlagBits::eTransferDst, size);
		auto               native = m_scheduler.Current().Handle();
		const vk::BufferCopy copy {*ring_offset, 0, size};
		native.copyBuffer(m_upload_dma->RingHandle(), readback->Handle(), 1, &copy);
		vk::MemoryBarrier to_host {};
		to_host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eHost, {}, 1, &to_host, 0, nullptr, 0,
		                       nullptr);
		m_scheduler.DeferOperation([snapshot, readback, size] {
			readback->Invalidate(0, size);
			Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaVerifyChecks);
			if (std::memcmp(readback->Mapped().data(), snapshot->data(), size) != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::UploadDmaVerifyMismatches);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::printf("Upload DMA verify: %" PRIu64 " staged bytes differ in the ring\n",
					            size);
				}
			}
		});
	}
	return m_upload_dma->RingHandle();
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size, size_t guest_copies,
                                     const uint8_t* host_data, uint64_t host_base,
                                     std::vector<UploadHostCopy>* deferred) {
	if (copies.empty()) {
		return nullptr;
	}
	// The first guest_copies read guest memory at their destination; the rest read host_data
	// at (srcOffset - host_base).
	const auto source_of = [&](size_t index, const vk::BufferCopy& copy) -> const void* {
		if (index < guest_copies) {
			return reinterpret_cast<const void*>(buffer.CpuAddress() + copy.dstOffset);
		}
		return host_data + (copy.srcOffset - host_base);
	};

	auto [mapped, base_offset] = m_staging_buffer.Map(total_size, 4);
	if (mapped != nullptr) {
		for (size_t index = 0; index < copies.size(); index++) {
			auto& copy = copies[index];
			const void* alias = nullptr;
			if (deferred != nullptr && index < guest_copies) {
				// KYTY_UPLOAD_DMA_HOST_COPY: the DMA worker copies from the backing alias.
				alias = LibKernel::Memory::GuestBackingAlias(buffer.CpuAddress() + copy.dstOffset,
				                                             copy.size);
			}
			if (alias != nullptr) {
				deferred->push_back({mapped + copy.srcOffset, static_cast<const uint8_t*>(alias),
				                     copy.size});
			} else {
				std::memcpy(mapped + copy.srcOffset, source_of(index, copy), copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	for (size_t index = 0; index < copies.size(); index++) {
		const auto& copy = copies[index];
		std::memcpy(temporary->Mapped().data() + copy.srcOffset, source_of(index, copy),
		            copy.size);
	}
	temporary->Flush(0, total_size);
	const auto handle = temporary->Handle();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return handle;
}

void BufferCache::RecordRangeFact(uint64_t vaddr, uint64_t size, uint64_t signature,
                                  RangeFact fact) {
	auto& memo     = RangeMemoSlot(vaddr, size);
	memo.vaddr     = vaddr;
	memo.size      = size;
	memo.signature = signature;
	memo.fact      = fact;
	Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoRecords);
	m_range_memo_totals.records++;
}

void BufferCache::ClassifyRangeMemoDifference(const char* what, uint64_t vaddr, uint64_t size,
                                              uint64_t dirtied_before) {
	// Every transition that turns a page CPU-dirty advances its region's dirtying serial under the
	// region lock before the bit changes, and the hit's signature was read after `dirtied_before`.
	// Unchanged serials therefore mean the page was already dirty when the hit took place.
	if (dirtied_before == 0 ||
	    m_memory_tracker.RangeDirtiedSignature(vaddr, size) != dirtied_before) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyRaces);
		m_range_memo_totals.verify_races++;
		return;
	}
	ReportRangeMemoMismatch(what, vaddr, size);
}

void BufferCache::ReportRangeMemoMismatch(const char* what, uint64_t vaddr, uint64_t size) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyMismatches);
	m_range_memo_totals.verify_mismatches++;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::fprintf(stderr,
		             "BufferRangeMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", what,
		             vaddr, size);
	}
	if (RangeMemoVerifyMode() == 2) {
		EXIT("BufferRangeMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", what, vaddr,
		     size);
	}
}

bool BufferCache::RelaxedDirtySnapshot(uint64_t vaddr, uint64_t size,
                                       MemoryTracker::DirtyState& state) {
	if (!m_relaxed_queries || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const bool verify    = RelaxedVerifyMode() != 0;
	const auto signature = verify ? m_memory_tracker.RangeSignature(vaddr, size) : 0;
	if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state)) {
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedQueries);
	m_relaxed_totals.queries++;
	if (verify) {
		// Every region exists, so the locked query creates none.
		const auto locked = m_memory_tracker.QueryDirty(vaddr, size);
		VerifyRelaxedSnapshot(vaddr, size, state, locked, signature);
	}
	return true;
}

bool BufferCache::RelaxedNothingToUpload(uint64_t vaddr, uint64_t size) {
	if (!m_relaxed_queries || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const bool                verify    = RelaxedVerifyMode() != 0;
	const auto                signature = verify ? m_memory_tracker.RangeSignature(vaddr, size) : 0;
	MemoryTracker::DirtyState state;
	if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state) || state.cpu) {
		return false;
	}
	if (verify) {
		const auto locked = m_memory_tracker.QueryDirty(vaddr, size);
		if (!VerifyRelaxedSnapshot(vaddr, size, state, locked, signature) || locked.cpu) {
			// A page turned CPU-dirty in between: synchronize it now, as the locked path would.
			return false;
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedSyncSkips);
	m_relaxed_totals.sync_skips++;
	return true;
}

bool BufferCache::VerifyRelaxedSnapshot(uint64_t vaddr, uint64_t size,
                                        const MemoryTracker::DirtyState& relaxed,
                                        const MemoryTracker::DirtyState& locked,
                                        uint64_t                         signature) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyChecks);
	if (relaxed.cpu == locked.cpu && relaxed.gpu == locked.gpu) {
		return true;
	}
	// Other threads only make pages CPU-dirty and publish GPU-dirty ones, and every such change
	// advances the range's mutation serials first.
	const bool forbidden = (relaxed.cpu && !locked.cpu) || (!relaxed.gpu && locked.gpu);
	const bool quiet =
	    signature != 0 && m_memory_tracker.RangeSignature(vaddr, size) == signature;
	if (!forbidden && !quiet) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyRaces);
		return true;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyMismatches);
	m_relaxed_totals.mismatches++;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::fprintf(stderr,
		             "TrackerRelaxedVerify: addr=0x%016" PRIx64 " size=0x%" PRIx64
		             " relaxed cpu=%d gpu=%d, locked cpu=%d gpu=%d, serials %s\n",
		             vaddr, size, relaxed.cpu, relaxed.gpu, locked.cpu, locked.gpu,
		             quiet ? "unchanged" : "moved");
	}
	if (RelaxedVerifyMode() == 2) {
		EXIT("TrackerRelaxedVerify: a lock-free dirty snapshot differs from the locked query\n");
	}
	return false;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	if (!is_written && !is_texel_buffer && m_binding_memo != nullptr && GuestGpu::IsGpuThread()) {
		return ObtainReadBinding(vaddr, size, id);
	}
	return ObtainBufferNow(vaddr, size, is_written, is_texel_buffer, id, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainReadBinding(uint64_t vaddr, uint64_t size,
                                                            BufferId id) {
	// KYTY_BINDING_EPOCH_MEMO (bufferCache.h).
	const auto epoch  = SyncEpoch::Current();
	const auto before = m_memory_tracker.RangeSignature(vaddr, size);
	auto&      memo   = BindingMemoSlot(vaddr, size);
	if (before == 0 || memo.vaddr != vaddr || memo.size != size ||
	    memo.kind == BindingMemoKind::Empty) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSlot);
	} else if (memo.signature != before) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSignature);
	} else if (const bool stream = memo.kind == BindingMemoKind::Stream;
	           memo.guard != (stream ? m_scheduler.CurrentTick()
	                                 : m_bda_structure_epoch.load(std::memory_order_acquire))) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissGuard);
	} else {
		bool cross = false;
		if (memo.epoch != epoch) {
			// KYTY_BINDING_MEMO_CROSS_EPOCH (bufferCache.h): only a cache-buffer memo, and only
			// once no page of the range is CPU-dirty while the signature still holds. The relaxed
			// mirror answers "CPU-dirty" exactly on this thread (only it clears those bits); the
			// locked query also waits for a transition that advanced the serial but has not
			// changed its bits yet.
			MemoryTracker::DirtyState relaxed;
			cross = !stream && m_binding_memo_cross &&
			        m_memory_tracker.QueryDirtyRelaxed(vaddr, size, relaxed) && !relaxed.cpu &&
			        !m_memory_tracker.IsRegionCpuModified(vaddr, size) &&
			        m_memory_tracker.RangeSignature(vaddr, size) == before;
			if (!cross) {
				if (!stream && m_binding_memo_cross) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCrossRejects);
					m_binding_memo_totals.cross_rejects++;
				} else {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissEpoch);
				}
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCrossHits);
				m_binding_memo_totals.cross_hits++;
				memo.epoch = epoch;
			}
		}
		if (memo.epoch == epoch) {
			std::pair<Buffer*, uint64_t> hit {nullptr, memo.offset};
			if (stream) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoStreamHits);
				m_binding_memo_totals.stream_hits++;
				hit.first = &m_stream_buffer;
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCachedHits);
				m_binding_memo_totals.cached_hits++;
				hit.first = &m_slot_buffers[memo.id];
				TouchBuffer(*hit.first);
			}
			if (m_binding_memo_verify != 0) {
				const auto copy = memo;
				return VerifyBindingHit(copy, hit, id, cross);
			}
			return hit;
		}
	}
	BufferId   obtained {};
	const auto result = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
	RecordBinding(vaddr, size, epoch, before, result, obtained);
	return result;
}

void BufferCache::RecordBinding(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
                                const std::pair<Buffer*, uint64_t>& result, BufferId id) {
	if (before == 0) {
		return; // a tracker region did not exist yet (its creation made the range CPU-dirty)
	}
	BindingMemo entry;
	entry.vaddr  = vaddr;
	entry.size   = size;
	entry.epoch  = epoch;
	entry.offset = result.second;
	if (result.first == &m_stream_buffer) {
		// The decision and the copy saw one tracker state: no transition since `before`.
		if (m_memory_tracker.RangeSignature(vaddr, size) != before) {
			return;
		}
		entry.signature = before;
		entry.guard     = m_scheduler.CurrentTick();
		entry.kind      = BindingMemoKind::Stream;
	} else {
		const auto structure = m_bda_structure_epoch.load(std::memory_order_acquire);
		const auto after     = m_memory_tracker.RangeSignature(vaddr, size);
		if (structure == UINT64_MAX || after == 0 || IsBufferInvalid(id) ||
		    &m_slot_buffers[id] != result.first) {
			return;
		}
		if (size <= CACHING_PAGESIZE) {
			// The next small read decides from the bits `after` pins: record only when that is no
			// stream copy (not after a stream decision whose ring allocation failed, nor with hot
			// pages left CPU-dirty and nothing GPU-dirty).
			MemoryTracker::DirtyState state;
			if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state) || (state.cpu && !state.gpu) ||
			    m_memory_tracker.RangeSignature(vaddr, size) != after) {
				return;
			}
		}
		entry.signature = after;
		entry.guard     = structure;
		entry.id        = id;
		entry.kind      = BindingMemoKind::Cached;
	}
	BindingMemoSlot(vaddr, size) = entry;
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoRecords);
	m_binding_memo_totals.records++;
}

std::pair<Buffer*, uint64_t> BufferCache::VerifyBindingHit(const BindingMemo&           memo,
                                                           std::pair<Buffer*, uint64_t> hit,
                                                           BufferId id, bool cross) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyChecks);
	m_binding_memo_totals.verify_checks++;
	const auto vaddr = memo.vaddr;
	const auto size  = memo.size;
	// The tracker bits the hit relied on and the decision the normal path takes from them. A
	// transition since the hit's signature read (a guest write fault) raced this check.
	const auto state          = m_memory_tracker.QueryDirty(vaddr, size);
	const bool quiet          = m_memory_tracker.RangeSignature(vaddr, size) == memo.signature;
	const bool decides_stream = size <= CACHING_PAGESIZE && state.cpu && !state.gpu;
	BufferId   obtained {};
	const auto fresh = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
	const char* problem = nullptr;
	bool        race    = false;
	if (memo.kind == BindingMemoKind::Stream) {
		if (!decides_stream) {
			problem = "a stream-copy hit whose range is no longer copied";
		} else if (fresh.first == &m_stream_buffer) {
			race = std::memcmp(m_stream_buffer.Mapped().data() + hit.second,
			                   m_stream_buffer.Mapped().data() + fresh.second,
			                   static_cast<size_t>(size)) != 0;
		}
		// Otherwise the ring could not take the copy without waiting: the normal path bound the
		// cache buffer instead, as it may.
	} else if (decides_stream) {
		problem = "a cache-buffer hit whose range is now a stream copy";
	} else if (fresh != hit) {
		problem = "a cache-buffer hit bound a different buffer or offset";
	} else if (cross && state.cpu) {
		// The cross-epoch check found no CPU-dirty page under the region locks with this
		// signature: one now needs a transition since, which `quiet` rules out below.
		problem = "a cross-epoch cache-buffer hit whose range has CPU-dirty pages";
	} else {
		// Pages the synchronization uploaded: hot pages written since, or pages a racing fault
		// dirtied before the hit's signature was taken.
		race = state.cpu;
	}
	if (problem != nullptr && !quiet) {
		problem = nullptr;
		race    = true;
	}
	if (race) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyRaces);
		m_binding_memo_totals.verify_races++;
	}
	if (problem != nullptr) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyMismatches);
		m_binding_memo_totals.verify_mismatches++;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			std::fprintf(stderr,
			             "BindingEpochMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
			             problem, vaddr, size);
		}
		if (m_binding_memo_verify == 2) {
			EXIT("BindingEpochMemoVerify: %s: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n", problem,
			     vaddr, size);
		}
	}
	return fresh;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferNow(uint64_t vaddr, uint64_t size,
                                                          bool is_written, bool is_texel_buffer,
                                                          BufferId id, BufferId* obtained) {
	if (!is_written && size <= CACHING_PAGESIZE) {
		// A small read of a CPU-dirty range that is not GPU-dirty is copied into the stream buffer.
		// KYTY_BUFFER_RANGE_MEMO: the decision depends only on the range's tracker bits, so a
		// Stream or Clean fact recorded under the current signature decides it without the two
		// locked queries (the bytes are copied again every time).
		// KYTY_BUFFER_DIRTY_QUERY_COMBINED (default on): both tracker queries with each region
		// lock taken once (MemoryTracker::QueryDirty), the same decision.
		const auto decide = [&] {
			if (DirtyQueryCombinedEnabled()) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BufferDirtyQueriesCombined);
				const auto state = m_memory_tracker.QueryDirty(vaddr, size);
				return !state.gpu && state.cpu;
			}
			return !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
			       m_memory_tracker.IsRegionCpuModified(vaddr, size);
		};
		bool                      stream = false;
		MemoryTracker::DirtyState relaxed;
		if (RelaxedDirtySnapshot(vaddr, size, relaxed)) {
			// KYTY_TRACKER_RELAXED_QUERIES: the same decision from the lock-free mirrors.
			stream = !relaxed.gpu && relaxed.cpu;
		} else if (m_range_memo != nullptr) {
			// Verify mode: dirtying serials before the lookup (ClassifyRangeMemoDifference).
			const auto  dirtied = m_range_memo_verify != 0
			                          ? m_memory_tracker.RangeDirtiedSignature(vaddr, size)
			                          : uint64_t {0};
			const auto  signature = m_memory_tracker.RangeSignature(vaddr, size);
			const auto& memo      = RangeMemoSlot(vaddr, size);
			if (signature != 0 && memo.signature == signature && memo.vaddr == vaddr &&
			    memo.size == size) {
				// Clean: no page is CPU-dirty, so not a stream copy.
				stream = memo.fact == RangeFact::Stream;
				if (stream) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoStreamHits);
					m_range_memo_totals.stream_hits++;
				}
				if (m_range_memo_verify != 0) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::BufferRangeMemoVerifyChecks);
					m_range_memo_totals.verify_checks++;
					RunRangeMemoVerifyHook(vaddr, size);
					const bool decided = decide();
					if (decided != stream) {
						if (!stream) {
							// Clean, CPU-dirty now: a page may have turned dirty since the lookup.
							ClassifyRangeMemoDifference(
							    "a small read binding changed its stream decision", vaddr, size,
							    dirtied);
						} else {
							// Only this thread clears CPU-dirty pages and sets GPU-dirty ones.
							ReportRangeMemoMismatch(
							    "a small read binding changed its stream decision", vaddr, size);
						}
						stream = decided;
					}
				}
			} else {
				stream = decide();
				if (stream && signature != 0 &&
				    m_memory_tracker.RangeSignature(vaddr, size) == signature) {
					RecordRangeFact(vaddr, size, signature, RangeFact::Stream);
				}
			}
		} else {
			stream = decide();
		}
		if (stream) {
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
			if (mapped != nullptr && Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size)) {
				m_stream_buffer.Commit();
				return {&m_stream_buffer, offset};
			}
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	if (obtained != nullptr) {
		*obtained = id;
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		PreserveImagesForGpuWrite(id, vaddr, size);
		// Writable descriptors reserve a new version before recording their shader commands.
		buffer.MarkContentWritten();
		// An Add that changes nothing cannot stale a cached clean page, because pages with
		// dirty bytes are never cached clean. The tracker GPU bits that SynchronizeBuffer just
		// set are only ever set together with this Add; clean-read verdicts never read them.
		// A range already contained is left as it is (the Add would change nothing).
		if (!m_gpu_modified_ranges.Contains(vaddr, size)) {
			CleanVerdict::Invalidate(vaddr, size, Coherence::Source::BufferDirtyAdd);
			m_gpu_modified_ranges.Add(vaddr, size);
		}
		NoteBufferContentWrite(vaddr, size);
		ForgetKnownFills(vaddr, size);
		HangTrace::NoteGpuWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainWrittenBuffer(uint64_t vaddr, uint64_t size,
                                                              std::span<const GuestRange> written,
                                                              BufferId                    id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}
	for (const auto& range: written) {
		if (!range.Valid() || range.address < vaddr || range.End() > vaddr + size) {
			EXIT("BufferCache: written range 0x%016" PRIx64 "+0x%" PRIx64
			     " is outside its binding 0x%016" PRIx64 "+0x%" PRIx64 "\n",
			     range.address, range.size, vaddr, size);
		}
	}
	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	// Bytes the shader cannot write only need the upload a read binding gets. Each written range
	// then uploads anything dirtied meanwhile and becomes GPU-owned under the same tracker locks,
	// exactly as a whole writable binding does.
	(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, "written-binding");
	// Writable descriptors reserve a new version before recording their shader commands.
	buffer.MarkContentWritten();
	for (const auto& range: written) {
		(void)SynchronizeBuffer(buffer, range.address, range.size, true, false, nullptr,
		                        "written-binding");
		PreserveImagesForGpuWrite(id, range.address, range.size);
		if (!m_gpu_modified_ranges.Contains(range.address, range.size)) {
			CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtyAdd);
			m_gpu_modified_ranges.Add(range.address, range.size);
		}
		NoteBufferContentWrite(range.address, range.size);
		ForgetKnownFills(range.address, range.size);
		HangTrace::NoteGpuWrite(range.address, range.size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, "image-source");
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	auto [staging, stage_offset] = m_staging_buffer.Map(size, 16);
	if (staging == nullptr || (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	                           !Libs::LibKernel::Memory::TryReadPrtBacking(vaddr, staging, size))) {
		EXIT("BufferCache: failed to read mapped guest image backing\n");
	}
	m_staging_buffer.Commit();
	return {&m_staging_buffer, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	// Obtaining the destination moves overlapping GPU-modified images into it first (their
	// bytes outside the fill survive); only then do those images lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (!preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	}
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Fill);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	if (preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	}
	dst->Fill(dst_offset, size, value);
	RecordKnownFill(vaddr, size, value);
}

bool BufferCache::TryWriteDataGpu(uint64_t vaddr, const uint32_t* data, uint64_t size) {
	if (vaddr == 0 || data == nullptr || size == 0 || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    size > 65536 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	// Only bytes owned by recorded GPU work need ordering on the GPU timeline; everything else
	// keeps the CPU write (which faults into the usual invalidation when tracked).
	if (!HasGpuDirtyBytes(vaddr, size) && !HasPendingBackingPublication(vaddr, size)) {
		return false;
	}
	KYTY_GPU_OP_SITE("buffercache.write_data");
	// As for fills and copies: obtaining the destination first moves overlapping GPU-modified
	// images into it (KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE); only then do they lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (!preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	}
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Copy);
	auto [buffer, offset] = ObtainBuffer(vaddr, size, true, true);
	if (preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	}
	auto& command         = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer->Handle();
	before.offset              = offset;
	before.size                = size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.updateBuffer(buffer->Handle(), offset, size, data);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1, &after, 0,
	                       nullptr);
	return true;
}

void BufferCache::RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value) {
	if (size == 0) {
		return;
	}
	std::scoped_lock lock(m_known_fill_mutex);
	ForgetKnownFillsLocked(vaddr, size);
	if (m_known_fills.size() >= 64) {
		m_known_fills.erase(m_known_fills.begin());
	}
	m_known_fills.push_back({vaddr, size, value});
	m_has_known_fills.store(true, std::memory_order_release);
	m_known_fill_generation.fetch_add(1, std::memory_order_acq_rel);
}

std::optional<uint32_t> BufferCache::KnownFill(uint64_t vaddr, uint64_t size) const {
	std::scoped_lock lock(m_known_fill_mutex);
	return KnownFillLocked(vaddr, size);
}

std::optional<uint32_t> BufferCache::KnownFill(uint64_t vaddr, uint64_t size,
                                               uint64_t& generation) const {
	std::scoped_lock lock(m_known_fill_mutex);
	generation = m_known_fill_generation.load(std::memory_order_relaxed);
	return KnownFillLocked(vaddr, size);
}

std::optional<uint32_t> BufferCache::KnownFillLocked(uint64_t vaddr, uint64_t size) const {
	// The range may be covered by several adjacent fills (e.g. per-slice consumption); all of
	// them must carry the same value.
	const uint64_t          end    = vaddr + size;
	uint64_t                cursor = vaddr;
	std::optional<uint32_t> value;
	while (cursor < end) {
		const auto covering =
		    std::find_if(m_known_fills.begin(), m_known_fills.end(), [cursor](const KnownFillRange& fill) {
			    return cursor >= fill.address && cursor < fill.address + fill.size;
		    });
		if (covering == m_known_fills.end() || (value && *value != covering->value)) {
			return std::nullopt;
		}
		value  = covering->value;
		cursor = covering->address + covering->size;
	}
	return value;
}

void BufferCache::ForgetKnownFills(uint64_t vaddr, uint64_t size) {
	// Every guest write fault and writable binding lands here; almost always nothing is known.
	// A fill recorded concurrently with this check is ordered after this write either way.
	if (!m_has_known_fills.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock(m_known_fill_mutex);
	ForgetKnownFillsLocked(vaddr, size);
}

void BufferCache::ForgetKnownFillsLocked(uint64_t vaddr, uint64_t size) {
	// Keep the parts of each fill outside the written range.
	const uint64_t end      = vaddr + size;
	const bool     overlaps = std::any_of(
        m_known_fills.begin(), m_known_fills.end(), [vaddr, end](const KnownFillRange& fill) {
            return vaddr < fill.address + fill.size && fill.address < end;
        });
	if (!overlaps) {
		return;
	}
	std::vector<KnownFillRange> kept;
	kept.reserve(m_known_fills.size() + 1);
	for (const auto& fill: m_known_fills) {
		const uint64_t fill_end = fill.address + fill.size;
		if (end <= fill.address || fill_end <= vaddr) {
			kept.push_back(fill);
			continue;
		}
		if (fill.address < vaddr) {
			kept.push_back({fill.address, vaddr - fill.address, fill.value});
		}
		if (end < fill_end) {
			kept.push_back({end, fill_end - end, fill.value});
		}
	}
	m_known_fills.swap(kept);
	m_has_known_fills.store(!m_known_fills.empty(), std::memory_order_release);
	m_known_fill_generation.fetch_add(1, std::memory_order_acq_rel);
	if (!GuestGpu::IsGpuThread()) {
		m_known_fill_foreign.fetch_add(1, std::memory_order_acq_rel);
	}
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// KYTY_ALIAS_BYTES: bytes a GPU-modified image owns without starting at the source (or at the
	// destination) are not in guest memory either; the GPU path moves them into the buffer.
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size) &&
	    (!TextureCache::AliasBytesEnabled() ||
	     (!m_texture_cache.IsRegionGpuModified(src_vaddr, size) &&
	      !m_texture_cache.IsRegionGpuModified(dst_vaddr, size)))) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	// The source synchronizes from a GPU-modified image at its address and the destination moves
	// overlapping GPU-modified images into itself before those images lose GPU ownership.
	const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
	if (dst_memory && !preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Copy);
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	if (dst_memory && preserve_images) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

Buffer& BufferCache::PrepareImageBytes(const RangeSet& ranges, Buffer* target) {
	uint64_t first = UINT64_MAX;
	uint64_t last  = 0;
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		first = std::min(first, begin);
		last  = std::max(last, end);
	});
	EXIT_IF(first >= last);
	Buffer* buffer = target;
	if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(first, last - first)) {
		buffer = &m_slot_buffers[FindBuffer(first, last - first)];
	}
	TouchBuffer(*buffer);
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		(void)SynchronizeBuffer(*buffer, begin, end - begin, true, false, nullptr, "image-bytes");
	});
	return *buffer;
}

void BufferCache::CommitImageBytes(Buffer& buffer, const RangeSet& ranges) {
	buffer.MarkContentWritten();
	ranges.ForEach([&](uint64_t begin, uint64_t end) {
		const auto size = end - begin;
		if (!m_gpu_modified_ranges.Contains(begin, size)) {
			CleanVerdict::Invalidate(begin, size, Coherence::Source::BufferDirtyAdd);
		}
		m_gpu_modified_ranges.Add(begin, size);
		NoteBufferContentWrite(begin, size);
		ForgetKnownFills(begin, size);
		HangTrace::NoteGpuWrite(begin, size);
	});
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::IsRegionGpuModifiedRelaxed(uint64_t vaddr, uint64_t size) const {
	return m_memory_tracker.IsRegionGpuModifiedRelaxed(vaddr, size);
}

BufferCache::PageStates BufferCache::CountPageStates(uint64_t vaddr, uint64_t size) const {
	PageStates states;
	if (!GuestRange {vaddr, size}.Valid()) {
		return states;
	}
	const auto end = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < end;
	     page += TRACKER_PAGE_SIZE) {
		MemoryTracker::DirtyState state;
		if (!m_memory_tracker.QueryDirtyRelaxed(page, TRACKER_PAGE_SIZE, state)) {
			states.untracked++;
		} else if (state.gpu) {
			states.gpu_dirty++;
		} else if (!state.cpu) {
			states.clean++;
		}
	}
	return states;
}

bool BufferCache::GpuDirtyMirrorMatches(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.GpuMirrorMatches(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		return true;
	}
	// KYTY_FALSE_SHARING_WRITES: released before their publication landed.
	return !m_early_released.empty() && OverlapsUnpublished(vaddr, size);
}

std::optional<BufferContentRevision> BufferCache::GetContentRevision(uint64_t vaddr,
	                                                                uint64_t size) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	if (!GuestRange {vaddr, size}.Valid() ||
	    m_memory_tracker.IsRegionCpuModified(vaddr, size) ||
	    HasPendingBackingPublication(vaddr, size)) {
		return std::nullopt;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || IsBufferInvalid(*owner)) {
		return std::nullopt;
	}
	const auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return std::nullopt;
	}
	return BufferContentRevision {*owner, buffer.ContentRevision(), m_content_revision_epoch};
}

void BufferCache::InvalidateContentRevisions() {
	EXIT_IF(!GuestGpu::IsGpuThread() || m_content_revision_epoch == UINT64_MAX);
	// Unbounded GPU writes follow; retire clean-read verdicts along with the revisions.
	CleanVerdict::Invalidate(0, UINT64_MAX, Coherence::Source::ContentRevisions);
	++m_content_revision_epoch;
	// They may change any buffer byte, including bytes of hot pages.
	SettleHotPages(0, 0);
	// Their bytes carry no writer tick, so no readback may skip this recording.
	m_unbounded_write_tick = m_scheduler.CurrentTick();
	// Nor do they clear the GPU ownership of images they may overwrite: order them against image
	// writes instead (TextureCache::SupersedesGpuDirtyBytes).
	m_unbounded_write_serial = Image::NextContentSerial();
}

uint64_t BufferCache::BeginBackingPublication(std::span<const GuestRange> ranges, uint64_t tick) {
	EXIT_IF(!GuestGpu::IsGpuThread() || ranges.empty());
	for (const auto& range: ranges) {
		EXIT_IF(!range.Valid());
	}
	std::lock_guard lock(m_backing_publication_mutex);
	const auto token = ++m_next_backing_publication_token;
	EXIT_IF(token == 0);
	// Pending ranges are not clean for backing reads; retire verdicts before publishing them.
	for (const auto& range: ranges) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::PublicationBegin);
	}
	m_backing_publications.push_back({token, tick, {ranges.begin(), ranges.end()}});
	m_backing_publication_count.store(m_backing_publications.size(), std::memory_order_release);
	return token;
}

void BufferCache::EndBackingPublication(uint64_t token) {
	std::lock_guard lock(m_backing_publication_mutex);
	const auto found = std::find_if(m_backing_publications.begin(), m_backing_publications.end(),
	                                [token](const auto& entry) { return entry.token == token; });
	EXIT_IF(found == m_backing_publications.end());
	// Backing authority changes here (runs on the priority worker). Bump before the entry is
	// erased so a verdict evaluated after the erase is tagged with the new generation.
	for (const auto& range: found->ranges) {
		CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::PublicationEnd);
	}
	m_backing_publications.erase(found);
	// Publish completion only after the callback has written every registered backing range.
	m_backing_publication_count.store(m_backing_publications.size(), std::memory_order_release);
}

bool BufferCache::HasPendingBackingPublication(uint64_t vaddr, uint64_t size) const {
	return PendingBackingPublicationTick(vaddr, size).has_value();
}

std::optional<uint64_t> BufferCache::PendingBackingPublicationTick(uint64_t vaddr,
	                                                               uint64_t size) const {
	const GuestRange query {vaddr, size};
	EXIT_IF(!query.Valid());
	if (m_backing_publication_count.load(std::memory_order_acquire) == 0) {
		return std::nullopt;
	}
	std::lock_guard lock(m_backing_publication_mutex);
	std::optional<uint64_t> latest;
	for (const auto& entry: m_backing_publications) {
		for (const auto& range: entry.ranges) {
			if (range.address < query.End() && query.address < range.End()) {
				latest = latest ? std::max(*latest, entry.tick) : entry.tick;
				break;
			}
		}
	}
	return latest;
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	MaintainHotPages();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// Pending side readbacks keep tracker pages GPU-owned without exact dirty bytes; settle
	// them so the ownership checks and downloads below see a consistent state.
	CompleteAllSideReadbacks();

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_IF(!DownloadBufferMemory(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::InvalidateBdaSynchronization() noexcept {
	// Also without incremental synchronization: KYTY_BDA_SYNC_EPOCH skips a pass only while the
	// registered buffers and GPU mappings are the ones its last pass scanned.
	auto epoch = m_bda_structure_epoch.load(std::memory_order_relaxed);
	while (epoch != UINT64_MAX &&
	       !m_bda_structure_epoch.compare_exchange_weak(epoch, epoch + 1,
	                                                     std::memory_order_release,
	                                                     std::memory_order_relaxed)) {}
}

void BufferCache::SynchronizeBdaBuffers(const RangeSet& mapped_ranges) {
	// KYTY_BDA_SYNC_EPOCH (syncEpoch.h): once per synchronization epoch. After a completed pass,
	// memory the pass left clean can only need an upload again within the same epoch through a
	// guest CPU write, which races the draws that follow (the first pass of the next epoch uploads
	// it), or a change of the registered buffers or GPU mappings, which moves the BDA structure
	// epoch. Everything the command processor orders before later draws (packets writing memory,
	// waits, cache invalidations, service commands) advances the epoch first.
	const auto sync_epoch = SyncEpoch::Current();
	const auto structure  = m_bda_structure_epoch.load(std::memory_order_acquire);
	if (m_bda_epoch_skip && sync_epoch == m_bda_synced_epoch &&
	    structure == m_bda_synced_structure) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochSkips);
		m_bda_epoch_totals.skips++;
		if (m_bda_epoch_verify != 0) {
			VerifyBdaEpochSkip(mapped_ranges);
		}
		return;
	}
	const auto fault_epoch = m_memory_tracker.FaultMutationEpoch();
	SynchronizeBdaBuffersNow(mapped_ranges);
	m_bda_epoch_totals.passes++;
	m_bda_synced_epoch     = sync_epoch;
	m_bda_synced_structure = structure;
	m_bda_synced_fault     = fault_epoch;
}

void BufferCache::VerifyBdaEpochSkip(const RangeSet& mapped_ranges) {
	// The full scan the skip replaced. It uploads whatever it finds, so this mode stays correct.
	// A normal page it finds CPU-dirty while the fault epoch is still the one taken before the
	// last pass is a page that pass should have uploaded (no page turned CPU-dirty since): a
	// mismatch. Hot pages, written without faults, are compared with their shadows and uploaded
	// when changed: guest writes racing the draws, as the skip assumes.
	BdaSyncStats verify;
	verify.verify_fault_epoch     = m_bda_synced_fault;
	verify.verify_structure_epoch = m_bda_synced_structure;
	{
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, &verify](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, &verify);
		});
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochVerifyChecks);
	m_bda_epoch_totals.verify_checks++;
	if (verify.verify_mismatch_pages == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochVerifyMismatches,
	                          verify.verify_mismatch_pages);
	m_bda_epoch_totals.verify_mismatch_pages += verify.verify_mismatch_pages;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "BdaSyncEpochVerify: a skipped pass would have missed %" PRIu64
		             " CPU-dirty page(s) that turned dirty before the last pass\n",
		             verify.verify_mismatch_pages);
	}
	if (m_bda_epoch_verify == 2) {
		EXIT("BdaSyncEpochVerify: a skipped BDA pass missed CPU-dirty pages\n");
	}
}

void BufferCache::SynchronizeBdaBuffersNow(const RangeSet& mapped_ranges) {
	const bool collect = Profiler::AggregateEnabled();
	// Read these before scanning: a fault to an already scanned page must force the NEXT
	// pass, even if its dirty transition completed before this pass finished uploading.
	// CpuMutationEpoch() is saturated while any hot page exists (hot pages are written without
	// faults), which used to make every pass a full scan. KYTY_BDA_HOT_SYNC uses the fault epoch
	// instead and, while it and the structure epoch hold, re-examines exactly the pages the fault
	// epoch does not cover: the hot page runs the last full pass found in the scanned buffers.
	// Every other page of those buffers was uploaded (clean and write-protected) by that pass and
	// can only become CPU-dirty again through a transition that changes one of the epochs,
	// including the write fault that promotes a page to hot and a hot page's demotion.
	// KYTY_BDA_DIRTY_LOG: the fault epoch comes with the ranges logged up to it, taken together.
	uint64_t fault_epoch  = 0;
	bool     log_complete = false;
	if (m_bda_dirty_log) {
		log_complete = m_memory_tracker.TakeDirtiedRanges(m_bda_dirtied, fault_epoch);
		if (!log_complete) {
			m_bda_log_totals.overflows++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogOverflows);
		}
	}
	const auto cpu_epoch = !m_bda_incremental_sync ? 0
	                       : m_bda_dirty_log       ? fault_epoch
	                       : m_bda_hot_sync        ? m_memory_tracker.FaultMutationEpoch()
	                                               : m_memory_tracker.CpuMutationEpoch();
	const auto structure_epoch =
	    m_bda_incremental_sync ? m_bda_structure_epoch.load(std::memory_order_acquire) : 0;
	const bool structure_holds = m_bda_incremental_sync && cpu_epoch != UINT64_MAX &&
	                             structure_epoch != UINT64_MAX &&
	                             structure_epoch == m_bda_scanned_structure_epoch;
	if (structure_holds && cpu_epoch == m_bda_scanned_cpu_epoch) {
		if (m_bda_hot_ranges.empty()) {
			if (collect) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncSkips);
			}
			return;
		}
		BdaSyncStats stats;
		if (SynchronizeBdaHotRanges(stats)) {
			if (collect) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotPasses);
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotRanges,
				                          m_bda_hot_ranges.size());
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies,
				                          stats.upload_copies);
			}
			if (BdaHotSyncVerifyEnabled()) {
				VerifyBdaHotPass(mapped_ranges, cpu_epoch, structure_epoch);
			}
			return;
		}
		// A recorded buffer is gone although the structure epoch held: scan everything.
	} else if (structure_holds && m_bda_dirty_log && log_complete && m_bda_log_baseline &&
	           SynchronizeBdaDirtied(mapped_ranges)) {
		m_bda_scanned_cpu_epoch = cpu_epoch;
		if (m_bda_log_verify != 0) {
			m_bda_log_totals.verify_checks++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogVerifyChecks);
			const auto missed = VerifyBdaFullScan(mapped_ranges, cpu_epoch, structure_epoch);
			if (missed != 0) {
				m_bda_log_totals.verify_mismatch_pages += missed;
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogVerifyMismatches, missed);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::fprintf(stderr,
					             "BdaDirtyLogVerify: a dirty-log pass missed %" PRIu64
					             " CPU-dirty page(s) (fault epoch %" PRIu64 ")\n",
					             missed, cpu_epoch);
				}
				if (m_bda_log_verify == 2) {
					EXIT("BdaDirtyLogVerify: a dirty-log pass missed CPU-dirty pages\n");
				}
			}
		}
		return;
	}

	BdaSyncStats stats;
	if (m_bda_hot_sync) {
		m_bda_hot_ranges.clear();
		stats.hot_ranges = &m_bda_hot_ranges;
	}
	const bool keep_stats = collect || m_bda_hot_sync;
	{
		// Only uploads are recorded while scanning: all of them share one barrier pair.
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, keep_stats, &stats](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, keep_stats ? &stats : nullptr);
		});
	}
	if (m_bda_incremental_sync) {
		m_bda_scanned_cpu_epoch       = cpu_epoch;
		m_bda_scanned_structure_epoch = structure_epoch;
	}
	// The log taken above holds nothing this scan did not cover; later transitions log again.
	m_bda_log_baseline = m_bda_dirty_log;
	if (collect) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncPasses);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncScannedBuffers, stats.scanned_buffers);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies, stats.upload_copies);
	}
}

bool BufferCache::SynchronizeBdaDirtied(const RangeSet& mapped_ranges) {
	// A hot run grows at most a few entries per pass; past this, rebuild the list in a full scan.
	constexpr size_t MaxHotRanges = 4096;
	if (m_bda_hot_ranges.size() > MaxHotRanges) {
		return false;
	}
	for (const auto& range: m_bda_hot_ranges) {
		const auto* buffer = m_slot_buffers.try_get(range.id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(range.address, range.size)) {
			return false;
		}
	}
	// The logged ranges report the hot runs they turn up (new hot pages, which a write fault
	// logged when it promoted them); the recorded runs are re-examined as in a hot pass.
	std::vector<BdaHotRange> found;
	BdaSyncStats             stats;
	stats.hot_ranges = &found;
	uint64_t logged  = 0;
	{
		const UploadBatch upload_batch(*this);
		m_bda_dirtied.ForEach([&](uint64_t begin, uint64_t end) {
			logged++;
			mapped_ranges.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
				SynchronizeBuffersInRange(start, finish - start, &stats);
			});
		});
		BdaSyncStats hot_stats;
		for (const auto& range: m_bda_hot_ranges) {
			(void)SynchronizeBuffer(m_slot_buffers[range.id], range.address, range.size, false,
			                        false, &hot_stats);
		}
		stats.upload_bytes += hot_stats.upload_bytes;
		stats.upload_copies += hot_stats.upload_copies;
	}
	for (const auto& range: found) {
		const bool known =
		    std::any_of(m_bda_hot_ranges.begin(), m_bda_hot_ranges.end(), [&](const auto& other) {
			    return other.id == range.id && other.address == range.address &&
			           other.size == range.size;
		    });
		if (!known) {
			m_bda_hot_ranges.push_back(range);
		}
	}
	m_bda_log_totals.passes++;
	m_bda_log_totals.ranges += logged;
	if (Profiler::AggregateEnabled()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogPasses);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncLogRanges, logged);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncScannedBuffers, stats.scanned_buffers);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies, stats.upload_copies);
	}
	return true;
}

uint64_t BufferCache::VerifyBdaFullScan(const RangeSet& mapped_ranges, uint64_t fault_epoch,
                                        uint64_t structure_epoch) {
	// It uploads whatever it finds, so a verify mode stays correct when it reports a mismatch.
	BdaSyncStats verify;
	verify.verify_fault_epoch     = fault_epoch;
	verify.verify_structure_epoch = structure_epoch;
	{
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, &verify](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, &verify);
		});
	}
	return verify.verify_mismatch_pages;
}

void BufferCache::VerifyBdaHotPass(const RangeSet& mapped_ranges, uint64_t fault_epoch,
                                   uint64_t structure_epoch) {
	// The full scan the hot pass replaced.
	const auto missed = VerifyBdaFullScan(mapped_ranges, fault_epoch, structure_epoch);
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotVerifyChecks);
	if (missed == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncHotVerifyMismatches, missed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "BdaHotSyncVerify: a hot pass would have missed %" PRIu64
		             " CPU-dirty page(s) (fault epoch %" PRIu64 ", structure epoch %" PRIu64 ")\n",
		             missed, fault_epoch, structure_epoch);
	}
	if (BdaHotSyncVerifyMode() == 2) {
		EXIT("BdaHotSyncVerify: a hot pass missed CPU-dirty pages\n");
	}
}

bool BufferCache::SynchronizeBdaHotRanges(BdaSyncStats& stats) {
	for (const auto& range: m_bda_hot_ranges) {
		const auto* buffer = m_slot_buffers.try_get(range.id);
		if (buffer == nullptr || buffer->is_deleted || !buffer->IsInBounds(range.address, range.size)) {
			return false;
		}
	}
	// A hot run that is no longer hot is synchronized like any range: a demoted page (still
	// CPU-dirty) is uploaded and protected, a settled or re-owned one is left alone.
	const UploadBatch upload_batch(*this);
	for (const auto& range: m_bda_hot_ranges) {
		(void)SynchronizeBuffer(m_slot_buffers[range.id], range.address, range.size, false, false,
		                        &stats);
	}
	return true;
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BdaSyncStats* stats) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			if (stats != nullptr) {
				++stats->scanned_buffers;
				stats->buffer_id = it->second;
			}
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false, stats);
		}
	}
}

} // namespace Libs::Graphics
