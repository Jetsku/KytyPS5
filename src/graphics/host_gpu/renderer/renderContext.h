#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/samplerCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/lodStats.h"
#include "graphics/host_gpu/renderer/occlusion.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <array>
#include <atomic>
#include <span>
#include <string>
#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

// The renderer's lock (recursive, like Common::Mutex). The GPU thread (the command processor's
// resolver) takes it around every draw and dispatch, thousands of times a frame; other threads
// (presentation unless KYTY_PRESENT_UNLOCKED, a capture) rarely. Each Common::Mutex acquisition and
// release is an interlocked instruction that waits for the GPU thread's store buffer to drain.
// KYTY_RENDER_LOCK_ASYMMETRIC=1 (startup, default 0; Windows; chenxiao07/KytyPS5 e953c484b): an
// asymmetric Dekker lock. The GPU thread (the owner, BecomeOwner) marks itself inside with a plain
// store and proceeds unless a request is up. Another thread takes the mutex, raises `requested`,
// makes the owner's stores visible and the owner's later loads see the request
// (FlushProcessWriteBuffers: an interrupt on every processor of the process) and waits until the
// owner is outside. An owner that finds a request steps out and takes the mutex behind the
// requester. Off (or without the flush): the owner always takes the mutex, as before.
class RenderMutex {
public:
	RenderMutex()  = default;
	~RenderMutex() = default;
	KYTY_CLASS_NO_COPY(RenderMutex);

	// The calling thread (the GPU thread) takes the owner's side from now on, until it resigns
	// (outside any acquisition).
	void BecomeOwner();
	void ResignOwner();

	void Lock() {
		if (!t_owner) {
			RequesterLock();
			return;
		}
		if (m_owner_depth++ != 0) {
			return;
		}
		m_busy.store(1, std::memory_order_relaxed);
		// The store before the load in program order; the requester's flush orders them on the CPU.
		std::atomic_signal_fence(std::memory_order_seq_cst);
		if (!m_asymmetric || m_requested.load(std::memory_order_acquire) != 0) {
			OwnerLockSlow();
		}
	}
	void Unlock() {
		if (!t_owner) {
			RequesterUnlock();
			return;
		}
		if (--m_owner_depth != 0) {
			return;
		}
		if (m_owner_locked) {
			OwnerUnlockSlow();
			return;
		}
		m_busy.store(0, std::memory_order_release);
	}

private:
	void OwnerLockSlow();
	void OwnerUnlockSlow();
	void RequesterLock();
	void RequesterUnlock();

	Common::Mutex m_mutex;
	// The owner's: it is inside (written at each outermost acquisition), and its depth.
	alignas(64) std::atomic<uint32_t> m_busy {0};
	uint32_t m_owner_depth  = 0;
	bool     m_owner_locked = false; // its outermost acquisition took the mutex
	// The requesters' (under the mutex): one holds the mutex and waits for the owner to leave.
	alignas(64) std::atomic<uint32_t> m_requested {0};
	bool m_owner_exists = false;
	bool m_asymmetric   = false;
	inline static thread_local bool     t_owner           = false;
	inline static thread_local uint32_t t_requester_depth = 0;
};

class RenderLockGuard {
public:
	// NOLINTNEXTLINE(google-runtime-references)
	explicit RenderLockGuard(RenderMutex& mutex): m_mutex(mutex) { m_mutex.Lock(); }
	~RenderLockGuard() { m_mutex.Unlock(); }
	KYTY_CLASS_NO_COPY(RenderLockGuard);

private:
	RenderMutex& m_mutex;
};

class NativeRayTracing;
struct ShaderStageRuntime;

class RenderContext {
public:
	explicit RenderContext(GraphicContext& graphics);
	~RenderContext();
	KYTY_CLASS_NO_COPY(RenderContext);

	[[nodiscard]] GraphicContext&           GetGraphics() const noexcept { return m_graphics; }
	void                                    InitializeGpu(VideoOut::VideoOutDriver* video_out);
	void                                    ShutdownGpu();
	[[nodiscard]] GuestGpu&                 GetGpu() const;
	// Wakes guest queues suspended on external progress (e.g. a completed flip). Any thread;
	// a no-op before InitializeGpu and after ShutdownGpu has begun.
	void                                    NotifyGpuProgress();
	[[nodiscard]] VideoOut::VideoOutDriver& GetVideoOut() const;

	RenderMutex&        GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }
	OcclusionCounter&   GetOcclusionCounter() { return m_occlusion_counter; }
	LodStatsCounter&    GetLodStats() { return m_lod_stats; }
	BindlessTable&      GetBindlessTable() { return m_bindless_table; }
	// KYTY_RT_HW (nativeRayTracing.h): created at its first use, on the GPU thread.
	NativeRayTracing&   GetNativeRayTracing();

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	// Store in place (KYTY_STORE_IN_PLACE / KYTY_STORE_CLEAN, adapted from usa-ratt/KytyPS5-GT7
	// 21246d78a and 41848bb9a). A game thread's write fault: whether its store may be done in place
	// (a cheap hint; the GPU thread decides).
	[[nodiscard]] bool IsStoreInPlaceCandidate(uint64_t fault_vaddr) noexcept;
	// The store at `pc` that faulted at fault_vaddr: [vaddr, vaddr + size) written to both copies
	// by the GPU thread (BufferCache::StoreInPlace), the page left protected. False to decline.
	[[nodiscard]] bool TryStoreInPlace(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr,
	                                   const void* data, uint64_t size) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// A host write of the whole of [vaddr, vaddr + size) is about to happen (an APR file read):
	// when resource tracking watches any of its pages, one invalidation of the range (as the
	// kernel's file reads make) instead of a write fault per page. False: nothing watched.
	bool PrepareHostOverwrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// GPU preparation only, outside texture-cache/tracker locks. Does not download dirty images.
	[[nodiscard]] bool SynchronizeGpuBackingForRead(uint64_t vaddr, uint64_t size);
	// Diagnostics: why a read of [vaddr, vaddr + size) is not clean (mapping, GPU-dirty bytes, a
	// pending publication, GPU-modified images over it, the newest recorded buffer writer).
	[[nodiscard]] std::string DescribeGpuBackingForRead(uint64_t vaddr, uint64_t size);
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	// The host protection of the range was replaced behind the page tracker (PageManager::
	// ResyncHostProtection): the watched pages get the tracker's protection back.
	void ResyncHostProtection(uint64_t vaddr, uint64_t size) {
		m_page_manager.ResyncHostProtection(vaddr, size);
	}
	// KYTY_MPROTECT_TRACKED_ONCE (PageManager::ProtectGuestWritable).
	void ProtectGuestWritable(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode guest_mode) {
		m_page_manager.ProtectGuestWritable(vaddr, size, guest_mode);
	}
	void               PrepareBda();
	// KYTY_BDA_REACH=1|verify (default off, live; renderContextReach.cpp; based on TheCruZ/KytyPS5-GTA
	// SynchronizeDmaFootprint): the stages of one draw or dispatch. When every stage that reads
	// through device addresses has a reach proven for its inputs (ShaderInfo write_ranges.dma), only
	// the buffers inside the reach are synchronized; otherwise, and with the switch off, PrepareBda().
	struct BdaReachStage {
		const ShaderStageRuntime*                 runtime = nullptr;
		// Bindless patches of the stage's flattened SRT (PreparedBindings::bindless_patches).
		std::span<const std::array<uint32_t, 3>> bindless_patches;
		std::array<uint32_t, 3>                  groups {};
		bool                                     has_groups = false;
		// A mesh stage's index bytes [mesh_index_begin, mesh_index_end), when known (may be empty).
		bool                                     mesh_index_known = false;
		uint64_t                                 mesh_index_begin = 0;
		uint64_t                                 mesh_index_end   = 0;
	};
	void               PrepareBda(std::span<const BdaReachStage> stages);
	void               RunGarbageCollector();
	// KYTY_VRAM_STATS (vramStats.h): one GPU memory report (GPU thread).
	void               ReportVram();

	// Detectors for guest-memory changes resource tracking does not see. They only count
	// (FrameEvent.HostBackingWrite*, GuestProtect*) and log the first occurrences to stderr.
	//  - An emulator write of guest backing bytes outside a publication (LOD-statistics reports,
	//    occlusion results), called right before it lands, from any thread: the GPU-dirty pages
	//    it overwrites (a later readback of them brings the GPU's older bytes back) and the
	//    tracked clean pages (a GPU copy of them keeps the old bytes until the page is dirtied).
	//  - A guest protection change of GPU-mapped memory (KernelMprotect), before it applies: the
	//    watched pages whose watch it overrides (their writes, or all accesses, stop faulting), and
	//    changes that restrict access (a later tracking transition sets the tracking protection,
	//    not the guest's).
	enum class HostWriter : uint8_t { LodStats, Occlusion };
	void NoteHostBackingWrite(uint64_t vaddr, uint64_t size, HostWriter writer) noexcept;
	// Called right before an emulator write of guest backing bytes (LOD-statistics reports,
	// occlusion results) on the GPU thread, a completion included. KYTY_HOST_WRITE_TRACKING
	// (default on): when the range has clean tracked pages and no GPU-owned bytes, it gets the
	// transition a guest write fault gives it (InvalidateMemory: CPU-dirty and writable, images
	// invalidated), so the next GPU use uploads the new bytes; otherwise, and with =0, it is only
	// reported (NoteHostBackingWrite). FrameEvent HostBackingWritesTracked.
	void PrepareHostBackingWrite(uint64_t vaddr, uint64_t size, HostWriter writer) noexcept;
	void NoteGuestProtection(uint64_t vaddr, uint64_t size, bool allows_read,
	                         bool allows_write) noexcept;

	void AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id);
	void TriggerInterrupt(int event_id, uint32_t context_id);

private:
	struct InterruptEqRegistration {
		LibKernel::EventQueue::KernelEqueue eq       = LibKernel::EventQueue::KERNEL_EQUEUE_INVALID;
		int                                 event_id = 0;
	};

	// Waits for the image write-backs over the range's pages; returns whether any overlapped.
	bool WaitWriteBacks(uint64_t vaddr, uint64_t size);

	GraphicContext&           m_graphics;
	RenderMutex               m_mutex;
	RenderExecutor            m_render_executor;
	CommandScheduler          m_command_scheduler;
	DescriptorHeap            m_descriptor_heap;
	PipelineCache             m_pipeline_cache;
	SamplerCache              m_sampler_cache;
	PageManager               m_page_manager;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	OcclusionCounter          m_occlusion_counter;
	LodStatsCounter           m_lod_stats;
	// Created empty; its Vulkan objects exist only once a bindless program runs (EnsureCreated).
	BindlessTable             m_bindless_table;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	// Guards m_gpu_notify against GPU teardown (NotifyGpuProgress holds it shared).
	std::shared_mutex         m_gpu_notify_mutex;
	GuestGpu*                 m_gpu_notify = nullptr;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	std::unique_ptr<NativeRayTracing> m_native_ray_tracing;
	bool                      m_fault_process_pending = false;
	// KYTY_BDA_REACH state (renderContextReach.cpp; shared_ptr: complete only there).
	struct BdaReach;
	std::shared_ptr<BdaReach> m_bda_reach;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
