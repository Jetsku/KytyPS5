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
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "kernel/eventQueue.h"

#include <memory>
#include <shared_mutex>
#include <vector>

namespace Libs::VideoOut {
class VideoOutDriver;
}

namespace Libs::Graphics {

class GuestGpu;

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

	Common::Mutex&      GetMutex() { return m_mutex; }
	CommandScheduler&   GetCommandScheduler() { return m_command_scheduler; }
	PipelineCache&      GetPipelineCache() { return m_pipeline_cache; }
	DescriptorHeap&     GetDescriptorHeap() { return m_descriptor_heap; }
	SamplerCache&       GetSamplerCache() { return m_sampler_cache; }
	BufferCache&        GetBufferCache() { return m_buffer_cache; }
	TextureCache&       GetTextureCache() { return m_texture_cache; }
	RenderExecutor&     GetRenderExecutor() { return m_render_executor; }
	OcclusionCounter&   GetOcclusionCounter() { return m_occlusion_counter; }
	LodStatsCounter&    GetLodStats() { return m_lod_stats; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	// GPU preparation only, outside texture-cache/tracker locks. Does not download dirty images.
	[[nodiscard]] bool SynchronizeGpuBackingForRead(uint64_t vaddr, uint64_t size);
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               PrepareBda();
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

	GraphicContext&           m_graphics;
	Common::Mutex             m_mutex;
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
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	std::unique_ptr<GuestGpu> m_gpu;
	// Guards m_gpu_notify against GPU teardown (NotifyGpuProgress holds it shared).
	std::shared_mutex         m_gpu_notify_mutex;
	GuestGpu*                 m_gpu_notify = nullptr;
	VideoOut::VideoOutDriver* m_video_out = nullptr;
	bool                      m_fault_process_pending = false;
	bool                      m_bda_logged = false;

	Common::Mutex                        m_interrupt_mutex;
	std::vector<InterruptEqRegistration> m_interrupt_eqs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERCONTEXT_H_
