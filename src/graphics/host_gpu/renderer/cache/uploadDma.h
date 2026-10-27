#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADDMA_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADDMA_H_

#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;

// Upload DMA (KYTY_UPLOAD_DMA, default on; needs a transfer-only queue family and is off under
// RenderDoc). CPU-dirty buffer uploads are staged in the host upload ring and copied into the
// device-local cache buffers by the graphics queue, which reads the staged bytes over PCIe: about
// 80 us/MB on the RTX 3090 (Gen3 x16, no resizable BAR), 20 MB and 1.3-1.6 ms of graphics-queue
// time per Sky Garden frame in the U49 capture (one per-frame 16 MB buffer alone 1.2 ms).
//
// Stage() moves the PCIe part to the copy engine: it reserves room in a device-local ring and
// queues a transfer-queue copy of the staged bytes into it. A worker submits the queued copies in
// batches, each signalling a timeline semaphore; the graphics copy then reads the ring (VRAM to
// VRAM) and the guest scheduler's submission that contains it waits for the semaphore value at
// the transfer stage (SubmitDependency). The data and the command order of the graphics copies
// are unchanged; only where the host bytes cross the bus changes.
//
// Ring space is reused only after the guest tick that recorded the reading copy has completed
// (that tick's submission waited for the transfer, so the transfer is complete too); a transfer
// batch that writes reused space also waits on the master semaphore for those ticks, so the
// ordering is explicit on the device. A full ring (or an upload below KYTY_UPLOAD_DMA_MIN_KB)
// keeps the graphics-queue copy from the host ring.
//
// KYTY_UPLOAD_DMA_MB (default 64): ring size. KYTY_UPLOAD_DMA_MIN_KB (default 64): smallest upload
// moved to the copy engine (small copies are latency bound; the semaphore wait would cost more).
[[nodiscard]] bool UploadDmaRequested();
// KYTY_UPLOAD_DMA_VERIFY=1: every staged upload also copies its ring bytes, in the recording that
// reads them, into a host-visible buffer; at completion they are compared with a snapshot of the
// staged host bytes taken when the copy was queued (UploadDmaVerifyChecks / UploadDmaVerifyMismatches
// and a log line). The copy ends the active rendering instance, so verify runs are for correctness.
[[nodiscard]] bool UploadDmaVerify();
// KYTY_UPLOAD_DMA_HOST_COPY (default on; =0: the command processor copies as before). A staged
// read upload's guest bytes are copied into the host staging ring by the DMA worker, just before
// it submits the transfer that reads them, instead of by the command processor. The worker reads
// them through the direct-memory backing alias (LibKernel::Memory::GuestBackingAlias), which
// never faults whatever protection the tracker or the guest gives the guest view meanwhile.
// Order and data are those of the command-processor copy: the pages were made clean and
// write-protected before the copy is queued (MemoryTracker::ForEachUploadRange), so a guest write
// after that point faults and makes them CPU-dirty again for the next upload, whether the copy
// reads the bytes before or after it (a write racing the draws, as with the command-processor
// copy made right after the protection). The graphics copies that read the staged bytes run in a
// submission that waits for the transfer (PendingValue), which the worker submits after the
// host copies. Ranges without an alias (not direct memory, or spanning mappings) are copied by
// the command processor as before.
[[nodiscard]] bool UploadDmaHostCopyEnabled();
// KYTY_UPLOAD_DMA_WRITE_DRAIN=1 (live, default off). The host copy above runs later than the
// command processor's own writes of guest memory that come after the upload in the command
// stream (DUMP_CONST_RAM, WRITE_DATA, DMA_DATA, COPY_DATA, CP-side labels: everything that calls
// StreamCopyDefer::NoteGuestWrite). Those are ordered, not races: the draw that the upload is for
// must not see them. When the worker is late (a loaded CP thread, KYTY_DEFER_PROTECTS=all sending
// every small upload to it), it copied them into the earlier upload: the earlier draws read
// "future" constants, arguments or descriptors (reproduced as a device loss on
// big_hand_pirate_island with the worker delayed 2 ms, with or without KYTY_DEFER_PROTECTS). With
// this switch such a write first makes the queued host copies that read the same bytes itself
// (UploadDma::DrainHostCopies: it takes them from the worker, or waits while the worker is
// copying them). =labels also makes EVERY queued host copy first for the writes the guest may
// observe early (labels written at CP time, WRITE_DATA, flip labels: NoteGuestWrite), because the
// guest may then rewrite memory an earlier upload still reads; that closed the residual losses
// (verify-all plus a 2 ms delayed worker, 0 of 5 instead of 4 of 6) but costs about 4% on Sky
// Garden (as slow as KYTY_DEFER_PROTECTS=0). =all (diagnostic) does that for every write. KYTY_UPLOAD_DMA_WRITE_DRAIN_LOG=1 logs the
// first drains and a running count.
// =frame (chenxiao07/KytyPS5 5c6f80faf, adapted): as =1, and a flip (the frame boundary) also makes
// every queued host copy first. Once a frame's submissions are consumed the guest may rewrite what
// that frame's work read (the frame after next, double buffered), and a copy still queued then
// would take those bytes. A frame's drain costs little: the worker has usually caught up by then.
// =defer (the "labels after the copies" variant): as =frame, and a label the command processor
// would write at once (an end-of-pipe or release label it does not defer otherwise) while host
// copies are queued is written at its tick's completion instead (CommandProcessor::TryDeferLabel,
// with its interrupt after it). That tick's submission waits for the transfer the worker submits
// after its host copies, so the guest sees the label only after the copies read their bytes,
// without the command processor making them (=labels) or waiting.
[[nodiscard]] bool UploadDmaWriteDrainEnabled();
// KYTY_UPLOAD_DMA_HELP=1 (default off): a lock-free waiter does a queued batch itself.
[[nodiscard]] bool UploadDmaHelpEnabled();
// KYTY_UPLOAD_DMA_LATE_AUDIT=1 (diagnostic, default off): hashes each host copy's guest bytes when
// it is staged and again just before the worker copies them; a difference is a write that landed
// in between (from a writer the drain does not order: guest CPU threads, other emulator threads).
// Counted (UploadDma late audit line), first ones logged with their guest address.
[[nodiscard]] bool UploadDmaLateAuditEnabled();

// A copy of host bytes into a staging buffer's mapping that the DMA worker performs before it
// submits the transfer reading them (KYTY_UPLOAD_DMA_HOST_COPY). `guest`: the guest address the
// source bytes belong to (KYTY_UPLOAD_DMA_WRITE_DRAIN).
struct UploadHostCopy {
	uint8_t*       destination = nullptr;
	const uint8_t* source      = nullptr;
	uint64_t       size        = 0;
	uint64_t       guest       = 0;
	uint64_t       stage_hash  = 0; // KYTY_UPLOAD_DMA_LATE_AUDIT: the source bytes at Stage()
};

class UploadDma final: public SubmitDependency {
public:
	UploadDma(GraphicContext& graphics, CommandScheduler& scheduler, uint64_t ring_size,
	          uint64_t min_bytes);
	~UploadDma() override;
	KYTY_CLASS_NO_COPY(UploadDma);

	// Null when disabled or the device has no transfer queue.
	[[nodiscard]] static std::unique_ptr<UploadDma> Create(GraphicContext&   graphics,
	                                                       CommandScheduler& scheduler);

	// Recording thread only. Queues the copy of [source_offset, source_offset + size) of `source`
	// (host bytes written before this call, in a buffer shared with the transfer family) into the
	// ring and returns the ring offset of the copy, or nullopt (too small, or no ring space that
	// the current recording's tick may use). `host_copies` (taken only when the copy is queued)
	// write parts of that range first, on the worker; the rest must be written already.
	// protects (KYTY_DEFER_PROTECTS, taken only when the copy is queued): deferred write
	// protections of the pages the host copies read, applied by the worker before it copies them.
	// llow_small: queue it even below the minimum size.
	[[nodiscard]] std::optional<uint64_t> Stage(vk::Buffer source, uint64_t source_offset,
	                                            uint64_t                     size,
	                                            std::vector<UploadHostCopy>* host_copies = nullptr,
	                                            PageManager::PendingProtects* protects = nullptr,
	                                            bool allow_small = false);
	// KYTY_UPLOAD_DMA_WRITE_DRAIN, recording thread: before a command-processor write of guest
	// memory [address, address + size), waits until the worker has made every queued host copy
	// that reads those bytes. Returns true when it waited.
	// `all`: the write may be observed by the guest (a label), which may then rewrite memory any
	// queued host copy still reads: makes every queued host copy (KYTY_UPLOAD_DMA_WRITE_DRAIN=labels;
	// =1 only the overlapping ones; =all every one always). Copies are taken from the worker.
	// `frame`: a frame boundary (a flip); =frame then makes every queued host copy.
	bool DrainHostCopies(uint64_t address, uint64_t size, bool all = false, bool frame = false);
	// KYTY_UPLOAD_DMA_WRITE_DRAIN=defer, recording thread: host copies are queued that a label
	// written now could overtake (the label is then written at its tick's completion instead).
	[[nodiscard]] bool LabelsWaitForHostCopies();
	struct DrainStats {
		uint64_t checks = 0; // writes that overlapped the queued host-copy bounds
		uint64_t waits  = 0; // writes that made (or waited for) queued host copies first
		uint64_t wait_ns = 0;
		uint64_t stolen  = 0; // host copies the command processor made itself
	};
	[[nodiscard]] DrainStats GetDrainStats() const noexcept { return m_drain; }
	// Host-copy totals (tests): bytes queued by Stage and bytes the worker has copied.
	[[nodiscard]] uint64_t HostCopyBytesQueued() const noexcept { return m_host_bytes_queued; }
	[[nodiscard]] uint64_t HostCopyBytesDone() const noexcept {
		return m_host_bytes_done.load(std::memory_order_acquire);
	}
	// Tests: while held, the worker leaves queued jobs (host copies and transfers) alone. The
	// thread submitting a batch that depends on them would wait until the hold ends.
	void HoldWorkerForTest(bool hold);
	[[nodiscard]] vk::Buffer RingHandle() const noexcept;
	[[nodiscard]] uint64_t   RingSize() const noexcept { return m_ring_size; }
	// Copies queued so far (recording thread).
	[[nodiscard]] uint64_t   Staged() const noexcept { return m_enqueued; }
	[[nodiscard]] uint64_t   MinBytes() const noexcept { return m_min_bytes; }

	// SubmitDependency (the guest scheduler's submissions). A batch waits for the transfer on the
	// device, and reaches its queue only after the worker submitted that transfer (Submittable).
	[[nodiscard]] uint64_t               PendingValue() override;
	[[nodiscard]] vk::Semaphore          Semaphore() const override { return m_semaphore; }
	[[nodiscard]] vk::PipelineStageFlags WaitStages() const override {
		return vk::PipelineStageFlagBits::eTransfer;
	}
	[[nodiscard]] bool Submittable(uint64_t value) override {
		return m_submitted.load(std::memory_order_acquire) >= value;
	}
	void WaitSubmittable(uint64_t value) override;
	void WaitHost(uint64_t value) override;

private:
	// The host copies of one job. The worker makes them before it submits the job's transfer,
	// unless the command processor took them first (KYTY_UPLOAD_DMA_WRITE_DRAIN: a CP write the
	// guest may observe, or one over the same bytes). state: 0 queued, 1 being copied, 2 copied.
	struct HostCopyTicket {
		uint64_t                     value = 0;
		std::vector<UploadHostCopy>  copies;
		PageManager::PendingProtects protects; // the job's deferred protections (CP copies)
		uint64_t                     lo    = UINT64_MAX;
		uint64_t                     hi    = 0;
		std::atomic<uint32_t>        state {0};
	};
	// Makes the ticket's copies unless the other thread took them (then waits until they are
	// made). `cp`: the command processor (applies the deferred protections first). True when
	// this call copied.
	bool ClaimAndCopy(HostCopyTicket& ticket, bool cp);
	struct Job {
		vk::Buffer            source        = nullptr;
		uint64_t              source_offset = 0;
		uint64_t              ring_offset   = 0;
		uint64_t              size          = 0;
		uint64_t              value         = 0;
		uint64_t              reuse_tick    = 0; // newest completed tick whose ring bytes were released
		std::shared_ptr<HostCopyTicket> ticket; // host copies, made before the transfer
		PageManager::PendingProtects protects;   // applied by the worker before the host copies
	};
	struct Span {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	struct Batch {
		vk::CommandBuffer command = nullptr;
		uint64_t          value   = 0; // free once the semaphore reaches it
	};

	[[nodiscard]] std::optional<uint64_t> Allocate(uint64_t size);
	void                                  Worker(std::stop_token stop);
	// One batch taken from m_jobs: its deferred protections, host copies and transfer submission
	// (the worker, or a waiter that helps: KYTY_UPLOAD_DMA_HELP). One at a time (m_processing).
	void                                  ProcessJobs(std::vector<Job>& jobs);
	void                                  SubmitBatch(std::vector<Job>& jobs);

	GraphicContext&         m_graphics;
	CommandScheduler&       m_scheduler;
	uint64_t                m_ring_size = 0;
	uint64_t                m_min_bytes = 0;
	std::unique_ptr<Buffer> m_ring;
	vk::Semaphore           m_semaphore = nullptr;
	vk::CommandPool         m_pool      = nullptr;
	// Recording thread.
	std::deque<Span>        m_spans; // allocated ring ranges in allocation order
	uint64_t                m_head        = 0;
	// Newest reading tick of released ranges. Released space may be written by any later job, so
	// every job carries it (the master semaphore is monotonic and that tick has completed).
	uint64_t                m_reuse_tick  = 0;
	uint64_t                m_enqueued    = 0;
	uint64_t                m_host_bytes_queued = 0;
	uint64_t                m_known_value = 0; // semaphore value observed by PendingValue
	// Newest value staged while recording m_stage_tick. Only the submission of that tick reads
	// its ring bytes (the graphics copies are recorded before the tick's End at the latest), so
	// only it waits; later submissions order after those copies through their own barriers.
	uint64_t                m_stage_tick  = 0;
	uint64_t                m_stage_value = 0;
	// KYTY_UPLOAD_DMA_WRITE_DRAIN: tickets of queued host copies in staging order; copied ones
	// are pruned from the front.
	std::deque<std::shared_ptr<HostCopyTicket>> m_tickets;
	DrainStats              m_drain;
	// Worker.
	std::vector<Batch>      m_batches;
	size_t                  m_next_batch = 0;
	std::atomic<uint64_t>   m_host_bytes_done {0};
	// Newest value whose transfer vkQueueSubmit2 has returned for (written by the worker).
	std::atomic<uint64_t>   m_submitted {0};
	// Shared.
	std::mutex              m_mutex;
	std::condition_variable m_available;
	std::vector<Job>        m_jobs;
	bool                    m_stopping = false;
	bool                    m_hold     = false; // HoldWorkerForTest
	// KYTY_UPLOAD_DMA_HELP: a waiter is doing a batch (the worker leaves m_jobs alone meanwhile).
	bool                    m_processing = false;
	std::jthread            m_worker; // last: joined before the members it uses go away
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADDMA_H_
