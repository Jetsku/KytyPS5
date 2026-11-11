#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEHEAP_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "graphics/host_gpu/renderer/cache/inplaceCandidates.h"

// KYTY_INPLACE (default off; 1 on, verify): the GPU reads guest buffer data where the game wrote
// it. Windows of the direct-memory backing alias (LibKernel::Memory::GuestBackingAlias, the
// always-RW view of the pagefile section that backs guest direct memory) are imported into Vulkan
// with VK_EXT_external_memory_host, and a read binding whose pages all took guest write faults in
// KYTY_INPLACE_FRAMES of the last 8 frames (default 2) is bound to the imported window instead of a
// cache buffer that is uploaded and write-protected each frame. Those pages stay hot (sticky
// CPU-dirty and writable: no faults, MemoryTracker hot pages without the idle sweep).
//   KYTY_INPLACE_WINDOW_MB (default 64): import granularity, in backing offsets.
//   KYTY_INPLACE_MAX_MB (default 512): cap of imported (pinned) memory; windows are only imported
//     over candidate pages, so a 16 GB PC can run it. Every import is logged with the total.
// Correctness rules:
//   - needs KYTY_LABEL_MODE=split or completion (labels reach the guest only after the host GPU
//     finished the work that read the memory); otherwise it refuses with a log line;
//   - not for GPU-dirty ranges, ranges with a pending backing publication or side readback,
//     ranges an image wrote (TextureCache::IsRegionGpuModified), texel or written bindings;
//   - a GPU write binding over a candidate page demotes it (Demote);
//   - a command-processor or emulator write of guest memory over a candidate page (every
//     StreamCopyDefer::NoteGuestWrite / NoteGpuOnlyWrite site and DMA_DATA) demotes the page and,
//     when an in-place read of its window is recorded but not complete, first completes that
//     work (BeforeCpWrite): the CP's CPU-side write must not reach earlier draws.
// verify: every in-place binding hashes its bytes when bound and again when its tick completed;
// a difference means the GPU may have read other bytes than the copy path would have uploaded
// (an upper bound: a change after the GPU read counts too). Counts and the first ranges logged.
// One console line every 10 s at a guest flip, "Inplace 10s: ...".
namespace Libs::Graphics {

class Buffer;
class CommandScheduler;
struct GraphicContext;

class InplaceHeap final {
public:
	enum class Mode : uint8_t { Off, On, Verify };
	// From KYTY_INPLACE, KYTY_LABEL_MODE and the device's VK_EXT_external_memory_host.
	[[nodiscard]] static Mode ConfiguredMode() noexcept;
	// Whether the device extension should be requested (KYTY_INPLACE set to on/verify).
	[[nodiscard]] static bool Requested() noexcept;

	InplaceHeap(GraphicContext& graphics, CommandScheduler& scheduler, Mode mode);
	~InplaceHeap();
	InplaceHeap(const InplaceHeap&)            = delete;
	InplaceHeap& operator=(const InplaceHeap&) = delete;

	[[nodiscard]] bool Verify() const noexcept { return m_mode == Mode::Verify; }

	// Any thread (write faults): consecutive-frame fault streaks make candidate pages.
	void NoteFault(uint64_t vaddr) noexcept;
	// Once per guest flip.
	void AdvanceFrame() noexcept;
	// Lock-free: every page of the range is a candidate.
	[[nodiscard]] bool AllCandidates(uint64_t vaddr, uint64_t size) const noexcept {
		return m_candidates.All(vaddr, size);
	}
	// GPU thread, after BufferCache checked the range's GPU state: the imported window buffer and
	// offset for [vaddr, +size), importing the window when the cap allows. False: copy path.
	[[nodiscard]] bool Bind(uint64_t vaddr, uint64_t size, Buffer*& buffer, uint64_t& offset);
	// Drops candidate pages of the range (GPU writers).
	void Demote(uint64_t vaddr, uint64_t size) noexcept;
	// Any thread, just before a CP / emulator CPU write of guest memory.
	void BeforeCpWrite(uint64_t vaddr, uint64_t size);

	[[nodiscard]] uint64_t PinnedBytes() const noexcept {
		return m_pinned_bytes.load(std::memory_order_relaxed);
	}

private:
	struct Window {
		std::unique_ptr<Buffer> buffer;
		uint64_t                backing_offset = 0;
		uint64_t                size           = 0;
		std::atomic<uint64_t>   last_read_tick {0};
		bool                    failed = false;
	};
	struct VerifyEntry {
		uint64_t tick;
		uint64_t vaddr;
		uint64_t size;
		uint64_t hash;
	};
	struct VerifyBatch {
		std::mutex               mutex;
		std::vector<VerifyEntry> entries;
	};
	void CheckVerifyBatch(VerifyBatch& batch);

	[[nodiscard]] Window* LookupWindow(uint64_t window_index);
	[[nodiscard]] Window* ImportWindow(uint64_t window_index);
	void Report(uint32_t frame);

	GraphicContext&                    m_graphics;
	CommandScheduler&                  m_scheduler;
	Mode                               m_mode;
	uint32_t                           m_promote_frames;
	uint64_t                           m_window_bytes;
	uint64_t                           m_max_bytes;
	uint64_t                           m_backing_base = 0;
	uint64_t                           m_backing_size = 0;
	InplaceCandidates                  m_candidates;
	std::mutex                         m_window_mutex;
	std::unordered_map<uint64_t, std::unique_ptr<Window>> m_windows;
	std::atomic<uint64_t>              m_pinned_bytes {0};
	std::shared_ptr<VerifyBatch>       m_verify_batch; // GPU thread
	uint64_t                           m_verify_tick = 0;
	// Counters since the last report (relaxed atomics: several threads).
	std::atomic<uint64_t> m_binds {0}, m_bind_bytes {0}, m_rejects_window {0}, m_rejects_cap {0},
	    m_promotions {0}, m_demotions_gpu {0}, m_cp_writes {0}, m_cp_waits {0}, m_cp_unordered {0},
	    m_verify_checks {0}, m_verify_mismatches {0};
	uint32_t              m_report_frames = 0;
	int64_t               m_report_start  = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_INPLACEHEAP_H_ */
