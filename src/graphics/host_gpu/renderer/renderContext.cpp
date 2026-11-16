#include "graphics/host_gpu/renderer/renderContext.h"
#include "common/staticOnce.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/nativeRayTracing.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cruzStats.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/faultCost.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/vramStats.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <fmt/format.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <shared_mutex>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

// KYTY_RENDER_LOCK_ASYMMETRIC (RenderMutex): whether the owner's fast path is used. Requires the
// process-wide store flush.
bool RenderLockAsymmetric() {
#if defined(_WIN32)
	const auto* value = std::getenv("KYTY_RENDER_LOCK_ASYMMETRIC");
	return value != nullptr && std::strcmp(value, "1") == 0;
#else
	return false;
#endif
}

void FlushOtherProcessorsStores() {
#if defined(_WIN32)
	::FlushProcessWriteBuffers();
#endif
}

} // namespace

void RenderMutex::BecomeOwner() {
	EXIT_IF(t_owner);
	m_mutex.Lock(); // no requester inside: a later one finds the owner
	m_owner_exists = true;
	m_asymmetric   = RenderLockAsymmetric();
	m_owner_depth  = 0;
	m_owner_locked = false;
	t_owner        = true;
	m_mutex.Unlock();
}

void RenderMutex::ResignOwner() {
	EXIT_IF(!t_owner || m_owner_depth != 0);
	m_mutex.Lock();
	t_owner        = false;
	m_owner_exists = false;
	m_mutex.Unlock();
}

void RenderMutex::OwnerLockSlow() {
	// A requester holds the mutex or is about to (or the fast path is off): out, then the mutex.
	m_busy.store(0, std::memory_order_release);
	m_mutex.Lock();
	m_owner_locked = true;
}

void RenderMutex::OwnerUnlockSlow() {
	m_owner_locked = false;
	m_mutex.Unlock();
}

void RenderMutex::RequesterLock() {
	if (t_requester_depth++ != 0) {
		return;
	}
	m_mutex.Lock();
	if (!m_owner_exists || !m_asymmetric) {
		return;
	}
	m_requested.store(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_seq_cst);
	// The owner's `busy` store is visible now if it made it, and its next acquisition sees the
	// request.
	FlushOtherProcessorsStores();
	while (m_busy.load(std::memory_order_acquire) != 0) {
		std::this_thread::yield();
	}
}

void RenderMutex::RequesterUnlock() {
	if (--t_requester_depth != 0) {
		return;
	}
	m_requested.store(0, std::memory_order_release);
	m_mutex.Unlock();
}

NativeRayTracing& RenderContext::GetNativeRayTracing() {
	if (m_native_ray_tracing == nullptr) {
		m_native_ray_tracing = std::make_unique<NativeRayTracing>(m_graphics, m_command_scheduler, m_buffer_cache);
	}
	return *m_native_ray_tracing;
}

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this),
      m_command_scheduler(*this, graphics, CommandScheduler::Role::Guest),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_occlusion_counter(*this), m_lod_stats(*this),
      m_bindless_table(graphics, m_command_scheduler, m_page_manager) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// Bindless table (Senaxx): images the texture cache drops leave the table's slots first.
	m_texture_cache.on_bindless_unregister = [this](ImageId id, bool replaced) {
		m_bindless_table.OnImageUnregistered(id, replaced);
	};
	m_texture_cache.on_image_views_dropped = [this](std::span<const CachedImageView> views) {
		return m_bindless_table.ReleaseViews(views);
	};
	// What guest write tracking costs on this PC (faultCost.h), measured before the GPU caches
	// take their first fault.
	FaultCost::SetThreadDescriber([](char* name, uint64_t size) -> int {
		if (GuestGpu::IsGpuThread()) {
			return 0;
		}
		auto self = LibKernel::PthreadSelfOrNull();
		if (self == nullptr) {
			return 2;
		}
		char full[64] {};
		if (name != nullptr && size != 0 && LibKernel::PthreadGetname(self, full) == 0) {
			std::snprintf(name, static_cast<size_t>(size), "%s", full);
		}
		return 1;
	});
	FaultCost::RunStartupBenchmark();
	// KYTY_WAIT_ATTRIB: the command processor's op recording a GPU write (CpGaps' innermost op
	// scope, needs KYTY_CP_GAP_STATS), and the guest mapping of a region.
	FaultCost::SetOpDescriber([]() -> const char* {
		auto* state = CpGaps::t_state;
		if (state == nullptr) {
			return "?";
		}
		const auto depth = std::min(state->ledger.Depth(), CpGaps::Ledger::MaxDepth);
		for (uint32_t level = depth; level-- > 0;) {
			switch (const auto cat = state->ledger.At(level)) {
				case CpGaps::Cat::GpuWaitDrain:
				case CpGaps::Cat::GpuWaitOcclusion:
				case CpGaps::Cat::GpuWaitOther:
				case CpGaps::Cat::RecorderWait:
				case CpGaps::Cat::SubmitWait:
				case CpGaps::Cat::TextureUpload:
				case CpGaps::Cat::Protect:
				case CpGaps::Cat::Fault:
				case CpGaps::Cat::ProgPrepare:
				case CpGaps::Cat::ProgKey:
				case CpGaps::Cat::ProgMaterialize:
				case CpGaps::Cat::ProgPermutation: continue;
				default: return CpGaps::CatName(cat);
			}
		}
		return "none";
	});
	FaultCost::SetRangeDescriber([](uint64_t vaddr, uint64_t* start, uint64_t* size, char* name, uint64_t name_size) {
		return LibKernel::Memory::DescribeVirtualRange(vaddr, start, size, name, name_size);
	});
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
	std::unique_lock lock(m_gpu_notify_mutex);
	m_gpu_notify = m_gpu.get();
}

void RenderContext::NotifyGpuProgress() {
	std::shared_lock lock(m_gpu_notify_mutex);
	if (m_gpu_notify != nullptr) {
		m_gpu_notify->NotifyProgress();
	}
}

void RenderContext::ShutdownGpu() {
	{
		std::unique_lock lock(m_gpu_notify_mutex);
		m_gpu_notify = nullptr;
	}
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

static bool FaultWatchedOutsideEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_FAULT_WATCHED_OUTSIDE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		// KYTY_FAULT_WATCHED_OUTSIDE=1 (default off; TheCruZ/KytyPS5-GTA b7961f66): cache tracking may
		// still protect a page whose range left the GPU-mapped set; that fault is ours to resolve,
		// not a guest access violation.
		if (!FaultWatchedOutsideEnabled() || !GuestRange {fault_vaddr, fault_size}.Valid()) {
			return false;
		}
		const auto watched = m_page_manager.CountWatchedPages(fault_vaddr, fault_size);
		if (watched.write == 0 && watched.access == 0) {
			return false;
		}
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("RenderContext: resolving a fault on a watched page outside the GPU-mapped ranges "
			            "(0x%016" PRIx64 ")\n",
			            fault_vaddr);
		}
	}
	// The access retries once the image write-backs over its page land (upstream 9678c59b7;
	// only with --readback-storage-images, TextureCache::WriteBackTick is 0 otherwise).
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	const MemoryStats::ScopedTimer fault_timer(MemoryStats::Counter::FaultNs);
	const CpGaps::Scope            fault_gap(CpGaps::Cat::Fault);
	// KYTY_WAIT_ATTRIB: the command processor's faults by call site, page state and region.
	FaultCost::AttribScope attrib(access == PageFaultAccess::Write ? FaultCost::AttribKind::FaultWrite
	                                                               : FaultCost::AttribKind::FaultRead,
	                              FaultCost::InRetriedFault());
	if (attrib.Active()) {
		const auto page = Common::AlignDown(fault_vaddr, TRACKER_PAGE_SIZE);
		attrib.SetAddress(fault_vaddr);
		attrib.AddFlags((m_buffer_cache.IsRegionGpuModifiedRelaxed(page, TRACKER_PAGE_SIZE) ? FaultCost::AttribBufferGpu : 0u) |
		                (m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE) ? FaultCost::AttribImageGpu : 0u));
	}
	// Live cost numbers (faultCost.h); the slow-PC simulation's wait comes first, outside every lock.
	const auto start = FaultCost::NowNs();
	FaultCost::SimulateFault();
	const bool               map   = FaultCost::MapEnabled();
	const bool               split = map || attrib.Active();
	FaultCost::FaultRecord   record;
	MemoryStats::Count(access == PageFaultAccess::Write ? MemoryStats::Counter::WriteFaults
	                                                    : MemoryStats::Counter::ReadFaults);
	if (access == PageFaultAccess::Write) {
		// Both caches' releases of the page reach the host once, when the scope ends and no
		// tracking lock is held (KYTY_DEFER_UNPROTECT).
		const bool nested = PageManager::InDeferUnprotectScope();
		if (split) {
			FaultCost::MapInFault(true);
		}
		uint64_t mark = map ? FaultCost::NowNs() : 0;
		const auto lap = [&](FaultCost::Part part) {
			if (map) {
				const auto now = FaultCost::NowNs();
				record.part_ns[static_cast<int>(part)] += now - mark;
				mark = now;
			}
		};
		{
			const PageManager::DeferUnprotectScope defer_unprotect;
			m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size, true);
			lap(FaultCost::Part::Buffer);
			m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
			lap(FaultCost::Part::Texture);
			// Bindless heap pages (engine method): returns without a lock unless a heap is near.
			m_bindless_table.OnCpuWrite(fault_vaddr, fault_size);
			// The faulting page gets the protection its watchers ask even when neither cache
			// released anything here: another thread may have released it without having applied
			// the release yet. A fault inside an enclosing scope (an emulator write between that
			// scope's releases) applies at once: the enclosing scope ends only after the retry.
			m_page_manager.Reconcile(Common::AlignDown(fault_vaddr, TRACKER_PAGE_SIZE),
			                         TRACKER_PAGE_SIZE, nested);
		}
		lap(FaultCost::Part::Reconcile);
		if (split) {
			FaultCost::MapInFault(false);
		}
		if (nested) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DeferredUnprotectNestedFaults);
		}
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	const auto handler_ns  = FaultCost::NowNs() - start;
	if (access != PageFaultAccess::Write && GuestGpu::IsGpuThread()) {
		m_buffer_cache.NoteCpFault(fault_vaddr, handler_ns); // KYTY_CP_FAULT_WRITERS
	}
	const bool found_dirty = access == PageFaultAccess::Write && MemoryTracker::TakeFaultFoundDirty();
	if (found_dirty) {
		attrib.AddFlags(FaultCost::AttribFoundDirty);
	}
	FaultCost::NoteFault(handler_ns);
	if (map) {
		record.address     = fault_vaddr;
		record.handler_ns  = handler_ns;
		record.write       = access == PageFaultAccess::Write;
		record.found_dirty = found_dirty;
		FaultCost::MapFault(record);
	}
	return true;
}

namespace {

// KYTY_STORE_IN_PLACE=1: stores to pages protected because the GPU wrote them; KYTY_STORE_CLEAN=1:
// stores to CPU-clean pages watched for CPU writes (both default off).
bool StoreInPlaceGpuWritten() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_STORE_IN_PLACE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

bool StoreInPlaceClean() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_STORE_CLEAN");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// Stores done in place: each is a round trip to the GPU thread that leaves the page protected. A
// page the CPU writes over many times (a structure filled field by field) costs more that way than
// the normal path, after which it is writable: after PageBudget of them within WindowMs, the page's
// next store takes the normal path. A store site that spends the budget of BulkPages pages within
// a second is copying in bulk: it takes the normal path for BulkMs (the fork's budgets).
constexpr uint32_t StorePageBudget = 8;
constexpr uint64_t StoreWindowMs   = 100;
constexpr uint32_t StoreBulkPages  = 2;
constexpr uint64_t StoreBulkMs     = 2000;

bool TakeStoreBudget(uint64_t pc, uint64_t page) {
	struct PageEntry {
		uint64_t page     = UINT64_MAX;
		uint64_t since_ms = 0;
		uint32_t count    = 0;
	};
	struct SiteEntry {
		uint64_t pc            = 0;
		uint64_t since_ms      = 0;
		uint32_t spent         = 0; // pages whose budget it spent since since_ms
		uint64_t bulk_until_ms = 0;
	};
	static std::array<PageEntry, 1024> pages {};
	static std::array<SiteEntry, 256>  sites {};
	static std::mutex                  mutex;
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	std::scoped_lock lock(mutex);
	auto&            site = sites[(pc ^ (pc >> 8u)) % sites.size()];
	if (site.pc != pc) {
		site = {pc, now, 0, 0};
	}
	if (now < site.bulk_until_ms) {
		return false;
	}
	auto& entry = pages[(page / TRACKER_PAGE_SIZE) % pages.size()];
	if (entry.page != page || now - entry.since_ms >= StoreWindowMs) {
		entry = {page, now, 0};
	}
	if (entry.count >= StorePageBudget) {
		// This store takes the normal path, which leaves the page writable.
		if (now - site.since_ms >= 1000) {
			site.since_ms = now;
			site.spent    = 0;
		}
		if (++site.spent >= StoreBulkPages) {
			site.bulk_until_ms = now + StoreBulkMs;
			site.spent         = 0;
		}
		return false;
	}
	entry.count++;
	return true;
}

std::atomic<uint64_t> g_stores_in_place {0};
std::atomic<uint64_t> g_stores_declined {0};

} // namespace

bool RenderContext::IsStoreInPlaceCandidate(uint64_t fault_vaddr) noexcept {
	const bool gpu_written = StoreInPlaceGpuWritten();
	const bool clean       = StoreInPlaceClean();
	if ((!gpu_written && !clean) || GuestGpu::IsGpuThread() || !IsMapped(fault_vaddr, 1)) {
		return false;
	}
	if (clean) {
		return true;
	}
	const auto page = Common::AlignDown(fault_vaddr, TRACKER_PAGE_SIZE);
	return m_buffer_cache.IsRegionGpuModifiedRelaxed(page, TRACKER_PAGE_SIZE);
}

bool RenderContext::TryStoreInPlace(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr,
                                    const void* data, uint64_t size) noexcept {
	if (m_gpu == nullptr || GuestGpu::IsGpuThread() || size == 0 || !IsMapped(vaddr, size)) {
		return false;
	}
	const auto page = Common::AlignDown(fault_vaddr, TRACKER_PAGE_SIZE);
	if (!TakeStoreBudget(pc, page)) {
		g_stores_declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const bool gpu_written = StoreInPlaceGpuWritten();
	const bool clean       = StoreInPlaceClean();
	bool       written     = false;
	m_gpu->SendCommandSync([this, vaddr, data, size, gpu_written, clean, &written] {
		written = m_buffer_cache.StoreInPlace(vaddr, data, size, gpu_written, clean);
	});
	if (!written) {
		g_stores_declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	m_bindless_table.OnCpuWrite(vaddr, size);
	if (const auto n = g_stores_in_place.fetch_add(1, std::memory_order_relaxed) + 1;
	    (n & (n - 1)) == 0 && n >= 64) {
		LOGF("Store in place: %" PRIu64 " game-thread stores done on both copies, %" PRIu64
		     " declined (KYTY_STORE_IN_PLACE / KYTY_STORE_CLEAN)\n",
		     n, g_stores_declined.load(std::memory_order_relaxed));
	}
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (m_gpu != nullptr && tick == m_command_scheduler.CurrentTick()) {
		m_gpu->SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	// One host update per page for both caches, after their locks (KYTY_DEFER_UNPROTECT).
	const PageManager::DeferUnprotectScope defer_unprotect;
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_bindless_table.OnCpuWrite(vaddr, size);
	return true;
}

bool RenderContext::PrepareHostOverwrite(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	const auto watched = m_page_manager.CountWatchedPages(vaddr, size);
	if (watched.write == 0 && watched.access == 0) {
		return false;
	}
	return InvalidateMemory(vaddr, size);
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

std::string RenderContext::DescribeGpuBackingForRead(uint64_t vaddr, uint64_t size) {
	const auto images = m_texture_cache.DescribeGpuModifiedImages(vaddr, size);
	return fmt::format("mapped {} gpu_dirty {} pending_pub {} gpu_images [{}] writer {}", IsMapped(vaddr, size) ? 1 : 0,
	                   m_buffer_cache.HasGpuDirtyBytes(vaddr, size) ? 1 : 0,
	                   m_buffer_cache.HasPendingBackingPublication(vaddr, size) ? 1 : 0, images,
	                   m_buffer_cache.DescribeNewestWriter(vaddr));
}

bool RenderContext::SynchronizeGpuBackingForRead(uint64_t vaddr, uint64_t size) {
	if (!GuestGpu::IsGpuThread() || CommandScheduler::InDeferredOperation() ||
	    !m_command_scheduler.Active() || m_command_scheduler.Current().IsInvalid() ||
	    !IsMapped(vaddr, size)) {
		return false;
	}
	// KYTY_SRT_STALE_IMAGE_READS: GPU-modified images over the range either are no source of its
	// bytes any more (a CPU write superseded them: the backing read past them is exact, see
	// LibKernel::Memory::TryReadGpuBackingOverStaleImages) or have their owned bytes moved into the
	// buffer cache, whose GPU-dirty bytes the download below brings to the backing.
	const bool stale_image_reads = LibKernel::Memory::SrtStaleImageReadsEnabled();
	if (m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		if (!stale_image_reads) {
			return false;
		}
		if (m_texture_cache.IsRegionGpuOwnedForRead(vaddr, size)) {
			(void)m_texture_cache.MaterializeForRead(vaddr, size);
			if (m_texture_cache.IsRegionGpuOwnedForRead(vaddr, size)) {
				return false; // no download layout for an owner: its bytes stay with the image
			}
		}
	}

	// Exact dirty ranges are removed when a download is queued, before its backing publication.
	// Retired images can likewise have an outstanding publication without a live image owner.
	// Never hold the registry, mapping, texture-cache, or tracker locks over these waits.
	// Side readbacks are published by their faulting guest thread; finish any overlapping one
	// here (already submitted, so this never waits for the current recording).
	m_buffer_cache.CompleteSideReadbacks(vaddr, size);
	if (const auto tick = m_buffer_cache.PendingBackingPublicationTick(vaddr, size)) {
		m_command_scheduler.Wait(*tick);
		m_command_scheduler.WaitPriorityOperations(*tick);
	}
	if (m_buffer_cache.HasGpuDirtyBytes(vaddr, size)) {
		// Retain the CPU fault path's canonical-buffer discovery, clipped readback window,
		// actual backing copy, and dirty-page ownership transition.
		HangTrace::SetReadbackKind(HangTrace::ReadbackKind::GpuSync);
		m_buffer_cache.ReadMemory(vaddr, size);
		HangTrace::SetReadbackKind(HangTrace::ReadbackKind::Invalidate);
	}
	// A publication may already have finished since the failed strict read. Such a range is
	// ready too; callers still retry the actual read and bound their preparation retries.
	return IsMapped(vaddr, size) && !m_buffer_cache.HasGpuDirtyBytes(vaddr, size) &&
	       !(stale_image_reads ? m_texture_cache.IsRegionGpuOwnedForRead(vaddr, size)
	                           : m_texture_cache.IsRegionGpuModified(vaddr, size)) &&
	       !m_buffer_cache.HasPendingBackingPublication(vaddr, size);
}

// KYTY_MAP_INVALIDATE=1 (default off in this line; on in the Insomniac title presets): a new
// mapping is treated as a write of its whole range.
// Streaming games map the resident part of a texture and leave the rest of its range unmapped
// (Miles Morales' AMM pool); an image created then read zeros there, and its page watch could not
// protect pages that had no mapping. When the game maps the rest and loads the finer mips into it
// (AMPR file reads), no write faulted and the image kept the zeros: shared detail textures whose
// two finest mips stayed black made car paint black near the camera.
static bool MapInvalidateEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_MAP_INVALIDATE");
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		// GPU mapping changes retire clean-read verdicts (the backing translation has its own
		// generation in the guest address space).
		CleanVerdict::Invalidate(vaddr, size, Coherence::Source::MapMemory);
		m_mapped_ranges.Add(vaddr, size);
		m_buffer_cache.InvalidateBdaSynchronization();
		SyncEpoch::Advance();
	}
	if (!MapInvalidateEnabled() || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// Images and buffers over the new mapping hold the zeros read before it: invalidate them as a
	// file read into the range does. Untracked images watch the pages again at their refresh, now
	// that the pages can be protected.
	const bool images = m_texture_cache.IsRegionRegistered(vaddr, size);
	if (images) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("MapMemory: 0x%016" PRIx64 "+0x%" PRIx64
			     " maps into registered images: they are refreshed (KYTY_MAP_INVALIDATE)\n",
			     vaddr, size);
		}
	}
	(void)InvalidateMemory(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// The kernel unmaps every free range it is about to (re)map, most of which were never GPU
	// mapped. Buffers, images and GPU-dirty pages only exist inside GPU-mapped ranges, so such an
	// unmap has nothing to invalidate; skip the GPU round trip and full drain it would force.
	if (GuestRange {vaddr, size}.Valid()) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	const auto unmap = [this, vaddr, size] {
		// KYTY_UNMAP_DRAIN_SKIP=1 (default off; adapted from TheCruZ/KytyPS5-GTA 1f331bb7): the full
		// GPU drain is skipped when nothing in the range is owned by GPU work or written by a pending
		// completion: no GPU-dirty buffer bytes, no GPU-modified image, no pending backing
		// publication (downloads, side readbacks) and no deferred label write. The GPU never reads
		// guest memory itself (uploads copy it into cache buffers and images, freed by tick), and
		// the guest unmaps memory only after the labels of the work that used it. Streaming games
		// drained the GPU many times per frame here (GTA V: 5-8 ms per frame).
		static const bool drain_skip = [] {
			const auto* value = std::getenv("KYTY_UNMAP_DRAIN_SKIP");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		const bool skip_drain = drain_skip && m_command_scheduler.Active() && GuestRange {vaddr, size}.Valid() &&
		                        !m_buffer_cache.HasGpuDirtyBytes(vaddr, size) &&
		                        !m_texture_cache.IsRegionGpuModified(vaddr, size) &&
		                        !m_buffer_cache.HasPendingBackingPublication(vaddr, size) &&
		                        (m_gpu == nullptr || m_gpu->DeferredLabelTick(vaddr, size) == 0);
		Common::CruzStats::Add(Common::CruzStats::Counter::UnmapCalls);
		if (skip_drain) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::UnmapDrainSkips);
			Common::CruzStats::Add(Common::CruzStats::Counter::UnmapDrainSkips);
		} else if (m_command_scheduler.Active()) {
			Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitUnmap);
			const auto                    drain_start = Common::CruzStats::Enabled() ? Common::CruzStats::NowNs() : 0;
			const auto                    tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
			if (drain_start != 0) {
				Common::CruzStats::Add(Common::CruzStats::Counter::UnmapDrainNs,
				                       static_cast<uint64_t>(Common::CruzStats::NowNs() - drain_start));
				Common::CruzStats::Add(Common::CruzStats::Counter::UnmapDrains);
			}
		}
		m_buffer_cache.DropLivePages(vaddr, size); // KYTY_BDA_LIVE_PAGES (I-L2)
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_bindless_table.UnwatchRange(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		CleanVerdict::Invalidate(vaddr, size, Coherence::Source::UnmapMemory);
		m_mapped_ranges.Subtract(vaddr, size);
		m_buffer_cache.InvalidateBdaSynchronization();
		SyncEpoch::Advance();
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	// KYTY_BDA_LIVE_PAGES (bufferCacheLive.cpp): promoted redirects back, the forced set uploaded.
	m_buffer_cache.LivePagesPrePass();
	// KYTY_BDA_ACCESS_TAGS (bufferCacheTags.cpp): 1 = only the pages DMA shaders accessed and new
	// buffers; verify = the full pass plus the record of what 1 would have skipped.
	const auto tags = m_buffer_cache.BdaAccessTagsMode();
	BufferCache::s_tag_hops.prepares.fetch_add(1, std::memory_order_relaxed);
	if (tags == 1) {
		(void)m_buffer_cache.SynchronizeBdaTagged(m_mapped_ranges);
	} else {
		if (tags == 2) {
			m_buffer_cache.PrepareBdaTagVerify();
		}
		m_buffer_cache.SynchronizeBdaBuffers(m_mapped_ranges);
		m_buffer_cache.NoteBdaTagFullPass(); // registrations so far are uploaded
	}
	m_buffer_cache.ReportBdaTags();
	m_fault_process_pending = true;
}

void RenderContext::NoteHostBackingWrite(uint64_t vaddr, uint64_t size,
                                         HostWriter writer) noexcept {
	const auto states = m_buffer_cache.CountPageStates(vaddr, size);
	Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWrites);
	if (states.gpu_dirty == 0 && states.clean == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWriteGpuDirtyPages, states.gpu_dirty);
	Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWriteCleanPages, states.clean);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "HostBackingWrite: %s addr=0x%016" PRIx64 " size=0x%" PRIx64
		             " lands on %" PRIu64 " GPU-dirty and %" PRIu64
		             " clean tracked page(s) the tracker is not told about\n",
		             writer == HostWriter::LodStats ? "LOD-statistics report" : "occlusion result",
		             vaddr, size, states.gpu_dirty, states.clean);
	}
}

// KYTY_HOST_WRITE_TRACKING=0 leaves emulator writes of guest bytes untold to the tracker again
// (only NoteHostBackingWrite counts them). Same spellings as the other renderer switches.
static bool HostWriteTrackingEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_HOST_WRITE_TRACKING");
		if (value == nullptr || value[0] == '\0') {
			return true;
		}
		return !(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
		         std::strcmp(value, "off") == 0);
	}();
	return enabled;
}

void RenderContext::PrepareHostBackingWrite(uint64_t vaddr, uint64_t size,
                                            HostWriter writer) noexcept {
	// Only the GPU thread: it alone marks pages GPU-dirty, so the ownership checked below cannot
	// change before the invalidation (which would then download from a completion runner).
	// KYTY_BDA_LIVE_PAGES (I-C1): recorded work that reads these pages live completes first.
	m_buffer_cache.LivePagesBeforeCpWrite(vaddr, size);
	if (!HostWriteTrackingEnabled() || !GuestGpu::IsGpuThread() || !IsMapped(vaddr, size)) {
		NoteHostBackingWrite(vaddr, size, writer);
		return;
	}
	const auto states = m_buffer_cache.CountPageStates(vaddr, size);
	if (states.gpu_dirty == 0 && states.clean != 0 &&
	    !m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		// Clean tracked pages and no GPU-owned bytes: the transition a guest write fault makes
		// (CPU-dirty and writable, fills forgotten, images over them invalidated), before the
		// bytes land. The next GPU use of the range then uploads them, as after a guest write.
		Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWrites);
		Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWriteCleanPages, states.clean);
		Profiler::CountFrameEvent(Profiler::FrameEvent::HostBackingWritesTracked);
		(void)InvalidateMemory(vaddr, size);
		return;
	}
	// GPU-owned bytes (a guest write would first download them, which a write made in a
	// completion cannot), or nothing tracked as clean: counted and reported as before.
	NoteHostBackingWrite(vaddr, size, writer);
}

void RenderContext::NoteGuestProtection(uint64_t vaddr, uint64_t size, bool allows_read,
                                        bool allows_write) noexcept {
	Profiler::CountFrameEvent(Profiler::FrameEvent::GuestProtectCalls);
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		if (!m_mapped_ranges.Intersects(vaddr, size)) {
			return;
		}
	}
	const auto watched = m_page_manager.CountWatchedPages(vaddr, size);
	// Pages whose tracking protection the new guest mode replaces: a write-watched page that
	// becomes writable, an access-watched page that becomes readable or writable.
	const auto overridden = (allows_write ? watched.write : 0u) +
	                        (allows_read || allows_write ? watched.access : 0u);
	const bool restricts = !allows_read || !allows_write;
	Profiler::CountFrameEvent(Profiler::FrameEvent::GuestProtectWatchedPages,
	                          watched.write + watched.access);
	Profiler::CountFrameEvent(Profiler::FrameEvent::GuestProtectOverriddenPages, overridden);
	if (restricts) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::GuestProtectRestrictsGpuMemory);
	}
	if (overridden == 0 && !restricts) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "GuestProtect: addr=0x%016" PRIx64 " size=0x%" PRIx64
		             " read=%d write=%d on GPU memory: %" PRIu64 " write-watched and %" PRIu64
		             " access-watched page(s), %" PRIu64 " whose tracking protection it replaces\n",
		             vaddr, size, allows_read ? 1 : 0, allows_write ? 1 : 0, watched.write,
		             watched.access, overridden);
	}
}

void RenderContext::RunGarbageCollector() {
	const CpGaps::Scope gap(CpGaps::Cat::Gc);
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		// KYTY_FAULT_SCAN_MAPPED (faultManager.cpp): the scan is limited to the guest GPU mappings.
		std::vector<std::pair<uint64_t, uint64_t>> mapped;
		if (FaultManager::ScanMappedActive()) {
			std::shared_lock lock(m_mapped_ranges_mutex);
			mapped.reserve(m_mapped_ranges.Size());
			m_mapped_ranges.ForEach([&](uint64_t begin, uint64_t end) { mapped.emplace_back(begin, end); });
		}
		m_buffer_cache.ProcessFaultBuffer(mapped);
	}
	m_texture_cache.ProcessDownloadImages();
	m_buffer_cache.ProcessPendingBdaSettles(false);
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	m_graphics.TrimRetiredImages();
	if (VramStats::ReportDue()) {
		ReportVram();
	}
}

void RenderContext::ReportVram() {
	using VramStats::ToMiB;
	const auto now   = std::time(nullptr);
	std::tm    local {};
#if defined(_WIN32)
	localtime_s(&local, &now);
#else
	localtime_r(&now, &local);
#endif
	char clock[16];
	std::strftime(clock, sizeof(clock), "%H:%M:%S", &local);
	VramStats::Line("==== %s +%.1f s", clock, VramStats::Seconds());
	m_graphics.ReportVramStats();
	{
		std::shared_lock lock(m_mapped_ranges_mutex);
		uint64_t         mapped = 0;
		uint64_t         lowest = UINT64_MAX;
		uint64_t         highest = 0;
		m_mapped_ranges.ForEach([&](uint64_t begin, uint64_t end) {
			mapped += end - begin;
			lowest  = std::min(lowest, begin);
			highest = std::max(highest, end);
		});
		VramStats::Line("guest GPU mappings: %.1f MiB in %zu ranges, 0x%" PRIx64 "..0x%" PRIx64, ToMiB(mapped),
		                m_mapped_ranges.Size(), lowest == UINT64_MAX ? 0 : lowest, highest);
	}
	m_texture_cache.ReportVram();
	m_buffer_cache.ReportVram();
	m_pipeline_cache.ReportRamStats();
	VramStats::Flush();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
