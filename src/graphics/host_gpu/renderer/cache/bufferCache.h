#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/eagerReadbackPages.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/writeTickMap.h"

#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;
class UploadDma;
struct UploadHostCopy;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

struct BufferContentRevision {
	BufferId id;
	uint64_t write_revision;
	uint64_t global_epoch;
	bool operator==(const BufferContentRevision&) const noexcept = default;
};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	// write_fault: a guest write fault on the range (fault-ahead / hot-page policy applies).
	void                   InvalidateMemory(uint64_t vaddr, uint64_t size, bool write_fault = false);
	// Once per completed guest flip (any thread): the frame clock of hot-page detection.
	void                   AdvanceFrame() noexcept;
	// While one is alive (GPU thread), CPU-dirty uploads are only queued on the current command
	// buffer (KYTY_UPLOAD_BATCH, CommandBuffer::RequestUploadCopy); the outermost scope's end
	// records them all behind one barrier. Every command recorded meanwhile through
	// CommandBuffer::Handle() records the queue first, so only commands recorded through a handle
	// obtained BEFORE the scope began must not follow uploads made in it. A scope that queued no
	// upload leaves other pending barriers to the next flush point (KYTY_UPLOAD_BATCH_SCOPED_FLUSH,
	// default on; =0 records them at the scope end as before).
	class UploadBatch {
	public:
		explicit UploadBatch(BufferCache& cache);
		~UploadBatch();
		UploadBatch(const UploadBatch&)            = delete;
		UploadBatch& operator=(const UploadBatch&) = delete;

	private:
		BufferCache& m_cache;
	};
	// Reads use a side copy when every dirty byte they need was written by an already submitted
	// recording (KYTY_READBACK_SIDE_COPY=0 disables it). GPU-thread reads wait for their copy in
	// place (KYTY_READBACK_SIDE_GPU_THREAD=0 makes them drain instead).
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Publishes (waiting if necessary) every pending side readback overlapping the range. Any
	// thread; never waits for the current recording. Required before other ownership changes.
	// Returns how many of them were eager copies.
	uint32_t CompleteSideReadbacks(uint64_t vaddr, uint64_t size);
	void     CompleteAllSideReadbacks();
	// Eager readback publication (KYTY_READBACK_EAGER, default on; needs side readbacks). A page
	// whose GPU-written bytes a CPU reader needed read back becomes read-hot. When a submission
	// finds a hot page's dirty bytes all written by earlier (already submitted) recordings, it
	// appends a copy of them to its own command buffer and registers their publication; the
	// completion runner publishes them and unprotects the page when that submission completes,
	// unless a newer writer took the page meanwhile. A later read then finds the page clean
	// instead of faulting; a read before completion waits for that copy as for a side readback.
	// Values are unchanged: the bytes, their order against newer writers and the protection rules
	// are those of a side readback of the same bytes, only issued before the read.
	// The command processor calls this right before it submits the current recording, between
	// packets (GPU thread); never from inside another cache operation.
	void IssueEagerReadbacks();
	// True once after a recorded writer of a page the GPU thread reads back, outside a rendering
	// instance: the command processor then submits the recording right away, so the producer runs
	// (and its eager copy can complete) before that read, instead of the read draining the GPU.
	// Budgeted per frame (KYTY_READBACK_EAGER_FLUSHES, default 8; 0 disables).
	[[nodiscard]] bool TakeEagerFlushRequest(bool in_rendering);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// A writable binding whose shader can only write `written` (sub-ranges of [vaddr, vaddr +
	// size), e.g. from a write-range proof): the whole range is synchronized as for any binding,
	// but only `written` becomes GPU-owned (dirty, protected, write-ticked).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainWrittenBuffer(uint64_t vaddr, uint64_t size,
	                                                               std::span<const GuestRange> written,
	                                                               BufferId id = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// CP WRITE_DATA to bytes owned by recorded GPU work: records the write (vkCmdUpdateBuffer)
	// at its position in the GPU timeline and returns true; false leaves it to the CPU write.
	[[nodiscard]] bool TryWriteDataGpu(uint64_t vaddr, const uint32_t* data, uint64_t size);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// KYTY_ALIAS_BYTES (TextureCache::MaterializeOwnedBytes): the cache buffer that will own the
	// image bytes `ranges` (`target` when it covers them). Each range is synchronized as for a
	// writable binding: its CPU-dirty pages uploaded, its tracker pages GPU-owned. Then, once the
	// bytes are recorded into it, CommitImageBytes makes them GPU-dirty buffer bytes.
	[[nodiscard]] Buffer& PrepareImageBytes(const RangeSet& ranges, Buffer* target = nullptr);
	void                  CommitImageBytes(Buffer& buffer, const RangeSet& ranges);

	// Recorded GPU fills whose result has not been overwritten since: a range written by a
	// uniform-fill dispatch or FillBuffer, forgotten on any later GPU or CPU write to it. Lets
	// consumers learn a fill value (e.g. a DCC clear code) without reading GPU memory back.
	void RecordKnownFill(uint64_t vaddr, uint64_t size, uint32_t value);
	[[nodiscard]] std::optional<uint32_t> KnownFill(uint64_t vaddr, uint64_t size) const;
	// The same answer, with the generation of the recorded fills it was read from (under the same
	// lock). The generation moves whenever a fill is recorded or any part of one is forgotten, so
	// while it is unchanged KnownFill answers every range as it did (KYTY_DRAW_SEQUENCE_FAST).
	[[nodiscard]] std::optional<uint32_t> KnownFill(uint64_t vaddr, uint64_t size,
	                                                uint64_t& generation) const;
	[[nodiscard]] uint64_t KnownFillGeneration() const noexcept {
		return m_known_fill_generation.load(std::memory_order_acquire);
	}
	// Changes of the recorded fills made off the GPU thread (guest write faults). Verify modes
	// tell a change racing the GPU thread from a wrong reuse by it.
	[[nodiscard]] uint64_t ForeignKnownFillChanges() const noexcept {
		return m_known_fill_foreign.load(std::memory_order_acquire);
	}
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	// GPU-written bytes whose guest copy is not current: the exact GPU-dirty ranges, and the
	// bytes an early release left to their publication (KYTY_FALSE_SHARING_WRITES).
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// A native-buffer revision only: callers must separately rule out newer image ownership.
	// No buffer is created or synchronized. CPU-dirty and pending-publication ranges have no token.
	[[nodiscard]] std::optional<BufferContentRevision> GetContentRevision(uint64_t vaddr,
	                                                                    uint64_t size);
	// Invalidate retained results before commands whose writes cannot be bounded to one buffer.
	// Also disables side readbacks until the current recording is submitted.
	void InvalidateContentRevisions();
	// Image content serial (Image::NextContentSerial) taken when the latest unbounded writer was
	// bound: an image whose ContentSerial is larger was written after it.
	[[nodiscard]] uint64_t UnboundedWriteSerial() const noexcept { return m_unbounded_write_serial; }
	// Publications can outlive cache ownership. Registration is on the GPU thread; completion
	// is on the priority worker, and these queries never wait for GPU work or backing writes.
	[[nodiscard]] uint64_t BeginBackingPublication(std::span<const GuestRange> ranges,
	                                               uint64_t tick);
	void EndBackingPublication(uint64_t token);
	[[nodiscard]] bool HasPendingBackingPublication(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] std::optional<uint64_t> PendingBackingPublicationTick(uint64_t vaddr,
	                                                                  uint64_t size) const;
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Lock-free hint (MemoryTracker::IsRegionGpuModifiedRelaxed), any thread; and its verify check.
	[[nodiscard]] bool IsRegionGpuModifiedRelaxed(uint64_t vaddr, uint64_t size) const;
	// Tracker pages of [vaddr, vaddr + size) by state, from the lock-free mirrors (any thread; a
	// snapshot): GPU-dirty, clean (tracked, neither CPU- nor GPU-dirty: a GPU copy is current)
	// and untracked (no tracker region yet). The rest are CPU-dirty.
	struct PageStates {
		uint64_t gpu_dirty = 0;
		uint64_t clean     = 0;
		uint64_t untracked = 0;
	};
	[[nodiscard]] PageStates CountPageStates(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool GpuDirtyMirrorMatches(uint64_t vaddr, uint64_t size);
	void               ProcessFaultBuffer();
	// Caller holds the mapped-range lock. Cache/tracker iteration remains on the GPU thread.
	void               SynchronizeBdaBuffers(const RangeSet& mapped_ranges);
	// Map/unmap callers may run outside the GPU thread, under the mapped-range lock.
	void               InvalidateBdaSynchronization() noexcept;
	void               RunGarbageCollector();
	// KYTY_VRAM_STATS report lines (vramStats.h): guest buffers by age, the largest buffers of
	// every kind, the guest address span they cover, the GC thresholds (GPU thread).
	void               ReportVram();

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 40, 16>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// A run of hot pages a full BDA pass found inside one scanned buffer (KYTY_BDA_HOT_SYNC).
	struct BdaHotRange {
		BufferId id;
		uint64_t address = 0;
		uint64_t size    = 0;
	};
	struct BdaSyncStats {
		uint64_t scanned_buffers = 0;
		uint64_t upload_bytes    = 0;
		uint64_t upload_copies   = 0;
		// Full passes with KYTY_BDA_HOT_SYNC: the buffer being synchronized and where to record
		// the hot page runs it reports.
		BufferId                  buffer_id;
		std::vector<BdaHotRange>* hot_ranges = nullptr;
		// KYTY_BDA_HOT_SYNC_VERIFY: the epochs a hot pass relied on (0: not verifying), and the
		// normal CPU-dirty pages found while both still held, which the hot pass would have missed.
		uint64_t verify_fault_epoch     = 0;
		uint64_t verify_structure_epoch = 0;
		uint64_t verify_mismatch_pages  = 0;
	};
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size, BdaSyncStats* stats);
	// KYTY_BDA_HOT_SYNC pass: re-synchronizes only the hot page runs the last full pass recorded.
	// False (nothing done) when a recorded buffer is gone; the caller then scans fully.
	[[nodiscard]] bool SynchronizeBdaHotRanges(BdaSyncStats& stats);
	// KYTY_BDA_HOT_SYNC_VERIFY: the full scan after a hot pass that relied on these epochs.
	void VerifyBdaHotPass(const RangeSet& mapped_ranges, uint64_t fault_epoch,
	                      uint64_t structure_epoch);
	// The full scan a pass that relied on these epochs replaced (it uploads whatever it finds):
	// the normal CPU-dirty pages it finds while both epochs still hold, which that pass missed.
	[[nodiscard]] uint64_t VerifyBdaFullScan(const RangeSet& mapped_ranges, uint64_t fault_epoch,
	                                         uint64_t structure_epoch);
	// KYTY_BDA_DIRTY_LOG pass (SynchronizeBdaBuffersNow): synchronizes the ranges the tracker
	// logged since the last pass (m_bda_dirtied) and the recorded hot runs, instead of every
	// mapped buffer. False (nothing done) when a recorded hot run's buffer is gone.
	[[nodiscard]] bool SynchronizeBdaDirtied(const RangeSet& mapped_ranges);
	// The BDA synchronization pass itself (SynchronizeBdaBuffers decides whether it runs).
	void SynchronizeBdaBuffersNow(const RangeSet& mapped_ranges);
	// KYTY_BDA_SYNC_EPOCH_VERIFY: the full scan a skipped pass replaced.
	void VerifyBdaEpochSkip(const RangeSet& mapped_ranges);
	// KYTY_WRITTEN_SYNC_SKIP (default on; =0 off): a written synchronization (not a BDA pass; GPU
	// thread) of a range every page of which is GPU-dirty and none readback-pending
	// (MemoryTracker::IsRangeGpuOwned, under the region locks) returns at once: the written upload
	// would collect nothing and change no tracker bit, serial or protection. A writable binding
	// written again before any CPU access. FrameEvent WrittenSyncSkips.
	// KYTY_WRITTEN_SYNC_SKIP_VERIFY=1|exit: such a range goes through the normal path anyway, and
	// one that collected anything is a mismatch (WrittenSyncSkipVerifyMismatches; exit stops).
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer,
	                                     BdaSyncStats* stats = nullptr,
	                                     const char* upload_reason = nullptr);
	// KYTY_BUFFER_RANGE_MEMO (default on; =0 off). Facts about the tracker bits of a guest range,
	// valid while the range's MemoryTracker::RangeSignature is unchanged (no CPU-dirty, GPU-dirty,
	// hot or readback-pending bit of the range changed since):
	//  - Clean: no page of the range is CPU-dirty (normal or hot). A read-only synchronization
	//    (SynchronizeBuffer, not written, not a texel read) of such a range collects nothing and
	//    does nothing else, whatever buffer it is for, so it is skipped. Recorded when one found
	//    nothing to upload and the signature was the same before and after it.
	//  - Stream: a small read binding's range is CPU-dirty and not GPU-dirty, so ObtainBuffer copies
	//    it into the stream buffer (ObtainBuffer re-reads the bytes every time; only the decision
	//    is reused). Recorded when the signature was the same before and after the two queries.
	// A fact recorded while a transition was in progress fails the before/after comparison; a
	// transition after the lookup is a CPU write racing the draw, which the normal path would
	// equally miss. Direct-mapped, GPU thread only.
	// KYTY_BUFFER_RANGE_MEMO_VERIFY=1|exit re-evaluates every hit the normal way (uploading whatever
	// it finds) and counts disagreements (BufferRangeMemoVerifyMismatches; exit stops on the first).
	// The hit acted at its lock-free signature read, the re-evaluation happens later under the
	// region locks: a guest write fault in between (the re-evaluation often waits for the region
	// lock that very fault holds) dirties a page the hit legitimately did not see. Such a
	// difference counts as a race (BufferRangeMemoVerifyRaces), told apart by the range's
	// dirtying serials (MemoryTracker::RangeDirtiedSignature) read before the lookup; only a
	// difference no dirtying after that moment explains is a mismatch.
	enum class RangeFact : uint8_t { Clean, Stream };
	struct RangeMemo {
		uint64_t  vaddr     = 0;
		uint64_t  size      = 0;
		uint64_t  signature = 0; // 0: empty
		RangeFact fact      = RangeFact::Clean;
	};
	static constexpr size_t RangeMemoSlots = 8192;
	[[nodiscard]] RangeMemo& RangeMemoSlot(uint64_t vaddr, uint64_t size) noexcept {
		const auto hash = (vaddr >> 4u) * 0x9e3779b97f4a7c15ull ^ size * 0xc2b2ae3d27d4eb4full;
		return m_range_memo[static_cast<size_t>(hash >> 51u) & (RangeMemoSlots - 1)];
	}
	void RecordRangeFact(uint64_t vaddr, uint64_t size, uint64_t signature, RangeFact fact);
	void ReportRangeMemoMismatch(const char* what, uint64_t vaddr, uint64_t size);
	// KYTY_UPLOAD_DMA: moves the host -> VRAM part of the queued upload copies from the staging
	// ring to the copy engine (UploadDma) and returns the source the graphics copies then read
	// (the DMA ring, with the copies' source offsets rebased); otherwise `source` unchanged.
	// `host_copies` (KYTY_UPLOAD_DMA_HOST_COPY: staging bytes UploadCopies left unwritten) go to
	// the DMA worker with the copy, or are performed here when the copy is not staged.
	[[nodiscard]] vk::Buffer StageUploadDma(vk::Buffer source, std::span<vk::BufferCopy> copies,
	                                        std::vector<UploadHostCopy>* host_copies = nullptr);
	// A verify-mode difference that only pages turned CPU-dirty since the lookup can cause: a race
	// when the range's dirtying serials moved since `dirtied_before` (0: unknown, a race), a
	// mismatch otherwise.
	void ClassifyRangeMemoDifference(const char* what, uint64_t vaddr, uint64_t size,
	                                 uint64_t dirtied_before);
	// Tests only: runs right after a range-memo hit in verify mode, before its re-evaluation, so a
	// test can land a transition exactly in that window.
	using RangeMemoVerifyHook = void (*)(void* context, uint64_t vaddr, uint64_t size);
	static inline RangeMemoVerifyHook s_range_memo_verify_hook    = nullptr;
	static inline void*               s_range_memo_verify_context = nullptr;
	static void RunRangeMemoVerifyHook(uint64_t vaddr, uint64_t size) {
		if (s_range_memo_verify_hook != nullptr) {
			s_range_memo_verify_hook(s_range_memo_verify_context, vaddr, size);
		}
	}
	// KYTY_TRACKER_RELAXED_QUERIES (GPU thread): the range's dirty snapshot from the tracker's
	// lock-free mirrors (false: not available, use the locked queries); and whether a read-only
	// synchronization of the range would collect nothing (no CPU-dirty page).
	[[nodiscard]] bool RelaxedDirtySnapshot(uint64_t vaddr, uint64_t size,
	                                        MemoryTracker::DirtyState& state);
	[[nodiscard]] bool RelaxedNothingToUpload(uint64_t vaddr, uint64_t size);
	// KYTY_CP_CPU_ONLY_QUERY (default off, live): omit the GPU mirror when only CPU dirtiness
	// decides an upload. No cached state; the same mirrors and missing-region fallback apply.
	static bool QueryUploadSnapshot(const MemoryTracker& tracker, uint64_t vaddr, uint64_t size,
	                                MemoryTracker::DirtyState& state, bool& cpu_only);
	bool VerifyRelaxedSnapshot(uint64_t vaddr, uint64_t size,
	                           const MemoryTracker::DirtyState& relaxed,
	                           const MemoryTracker::DirtyState& locked, uint64_t signature,
	                           bool cpu_only = false);
	// KYTY_BINDING_EPOCH_MEMO (default on; =0 off; off with KYTY_SYNC_EPOCH=0). The result of a read
	// binding (ObtainBuffer: not written, not a texel read; GPU thread) of [vaddr, vaddr + size) is
	// reused by later read bindings of exactly that range while
	//  - the sync epoch is the one it was obtained in (syncEpoch.h): a guest CPU write since then
	//    races the draws, which may read the bytes from before it or after it;
	//  - the range's MemoryTracker::RangeSignature is unchanged: no tracker transition in the
	//    range's regions (uploads, GPU-dirty marks, readbacks, hot-page changes, write faults), so
	//    every tracker bit the binding's decision and synchronization read is the same;
	//  - a cache-buffer result: the buffer structure is unchanged (m_bda_structure_epoch moves on
	//    every Register/Unregister), so the range is in the same buffer at the same offset. It is
	//    recorded with the signature taken after its synchronization, and for a small read only
	//    when the tracker bits then do not make the next one a stream copy. Pages that
	//    synchronization would upload now are hot pages written since, or pages a guest write
	//    fault dirtied before the signature was taken: guest writes racing the draws;
	//  - a stream copy: the signature is the one taken before the decision and after the copy, and
	//    the stream buffer's tick is the one it was copied in (the ring never overwrites an
	//    allocation during its tick). Bytes written since race the draws.
	// A hit returns the same buffer and offset (touching a cache buffer's LRU entry): no dirty
	// query, page-table lookup, synchronization or stream copy. Every ordered change within an
	// epoch is a GPU-thread tracker transition or buffer registration; the emulator's own writes
	// of guest bytes (labels, WRITE_DATA, DMA, LOD and occlusion results) happen in fence packets,
	// which start a new epoch first, or complete asynchronously like GPU writes.
	// KYTY_BINDING_EPOCH_MEMO_VERIFY=1|exit: every hit also runs the normal path and returns its
	// result. A different decision (a stream copy against a cache buffer) or a different buffer or
	// offset while no tracker transition raced the check is a mismatch (exit stops on the first);
	// a stream copy whose bytes changed, or a cache-buffer range with CPU-dirty pages the normal
	// path uploads, counts as a race (BindingEpochMemoVerifyRaces).
	// KYTY_BINDING_MEMO_CROSS_EPOCH (default on; =0 off): a cache-buffer memo from an EARLIER epoch
	// whose signature and buffer structure still hold is reused when, under the region locks, no
	// page of the range is CPU-dirty (normal or hot) and the signature is still the same after that
	// check (the memo then moves to the current epoch). With no CPU-dirty page, every guest write to
	// the range faults first, which is a tracker transition; the normal path would then decide no
	// stream copy, find the same buffer and offset, and upload nothing (its only other inputs, the
	// tracker bits, are unchanged since the signature). The epoch only covers writes that do not
	// fault: to CPU-dirty pages, which this excludes. Emulator writes of backing bytes that bypass
	// the tracker are invisible to the normal path as well (RenderContext::PrepareHostBackingWrite).
	// A stream memo stays within its epoch (its bytes may have changed without a fault). Counters:
	// BindingEpochMemoCrossHits / CrossRejects, and why lookups missed (BindingEpochMemoMiss*).
	// In verify mode a cross-epoch hit finding CPU-dirty pages with the signature unchanged is a
	// mismatch, not a race.
	enum class BindingMemoKind : uint8_t { Empty, Stream, Cached };
	struct BindingMemo {
		uint64_t        vaddr     = 0;
		uint64_t        size      = 0;
		uint64_t        epoch     = 0;
		uint64_t        signature = 0;
		uint64_t        guard     = 0; // stream: tick; cached: buffer structure epoch
		uint64_t        offset    = 0;
		BufferId        id;
		BindingMemoKind kind = BindingMemoKind::Empty;
	};
	static constexpr size_t BindingMemoSlots = 2048;
	// KYTY_CP_COMMIT=bindslots: 32,768 slots, indexed by the hash's top 15 bits (11 above).
	static constexpr size_t BindingMemoSlotsLarge = 32768;
	static uint64_t BindingMemoHash(uint64_t vaddr, uint64_t size) noexcept {
		return (vaddr >> 4u) * 0x9e3779b97f4a7c15ull ^ size * 0xc2b2ae3d27d4eb4full;
	}
	[[nodiscard]] BindingMemo& BindingMemoSlot(uint64_t vaddr, uint64_t size) noexcept {
		const auto hash = BindingMemoHash(vaddr, size);
		return m_binding_memo[static_cast<size_t>(hash >> m_binding_memo_shift)];
	}
	// KYTY_CP_BINDING_HOT_MEMO (default off, live): a 4 KiB front tier for recent binding memos.
	// Copies carry exactly the main memo's certificates. Its hash prefix is shorter than either
	// main table's: a main-slot collision is also a hot-slot collision, and records mirror both.
	// A live-switch generation clears the tier on the next enabled lookup, including off periods
	// without any binding lookup. The main table and every invalidation guard stay unchanged.
	static constexpr size_t BindingHotMemoSlots = 64;
	static_assert(BindingHotMemoSlots <= BindingMemoSlots);
	[[nodiscard]] BindingMemo& BindingHotMemoSlot(uint64_t vaddr, uint64_t size) noexcept {
		return m_binding_hot_memo[static_cast<size_t>(BindingMemoHash(vaddr, size) >> 58u)];
	}
	std::array<BindingMemo, BindingHotMemoSlots> m_binding_hot_memo {};
	uint64_t m_binding_hot_generation = 0;
	bool     m_binding_hot_enabled = false;
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainReadBinding(uint64_t vaddr, uint64_t size,
	                                                             BufferId id);
	// The binding without the memo; *obtained receives the cache buffer's id (unchanged for a
	// stream copy).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferNow(uint64_t vaddr, uint64_t size,
	                                                           bool is_written, bool is_texel_buffer,
	                                                           BufferId id, BufferId* obtained);
	void RecordBinding(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
	                   const std::pair<Buffer*, uint64_t>& result, BufferId id);
	// `cross`: the hit came from another epoch (KYTY_BINDING_MEMO_CROSS_EPOCH).
	[[nodiscard]] std::pair<Buffer*, uint64_t> VerifyBindingHit(const BindingMemo& memo,
	                                                            std::pair<Buffer*, uint64_t> hit,
	                                                            BufferId id, bool cross);
	// `deferred` (KYTY_UPLOAD_DMA_HOST_COPY): guest copies with a backing alias are listed there
	// instead of copied, for StageUploadDma; the staging ring space is reserved either way.
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size, size_t guest_copies = SIZE_MAX,
	                                      const uint8_t* host_data = nullptr,
	                                      uint64_t       host_base = 0,
	                                      std::vector<UploadHostCopy>* deferred = nullptr);
	// Hot pages (GPU thread). Snapshots each hot page of hot_ranges, appends a copy (reading
	// m_hot_scratch from the first appended srcOffset on) for those that differ from their
	// shadow, and lists pages to return to normal tracking: `demote` (still CPU-dirty) and
	// `settle` (KYTY_HOT_PAGE_CHECK_LIMIT; SettleHotPages after the upload collection).
	void CollectHotPages(Buffer& buffer, std::span<const GuestRange> hot_ranges,
	                     std::vector<vk::BufferCopy>& copies, uint64_t& total_size,
	                     std::vector<uint64_t>& demote, std::vector<uint64_t>& settle);
	void EraseHotShadows(uint64_t vaddr, uint64_t size);
	// Before (or right after recording) a GPU-side write of the range that tracked GPU ownership
	// does not cover (image downloads into the buffer, unbounded address writers; size 0 = all):
	// returns its hot pages to normal tracking, clean unless their contents changed since their
	// last upload. From then on the ordinary fault tracking decides their next upload.
	void SettleHotPages(uint64_t vaddr, uint64_t size);
	void MaintainHotPages();
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// The texel-read download of one image that owns all of its bytes and starts at the read
	// (SynchronizeBufferFromImage's image found by TextureCache::FindImageFromRange).
	[[nodiscard]] bool SynchronizeBufferFromOwner(Buffer& buffer, Common::SlotId image_id);
	// Records a download of a GPU-modified image into `buffer` at the image's own guest address
	// (every mip level that fits). Caller holds the texture-cache lock and has checked that the
	// image may be downloaded. Returns the bytes covered from the image start, 0 when nothing
	// was recorded. skip_unchanged (texel reads, KYTY_TEXEL_SYNC_SKIP): record nothing when
	// neither the image nor the buffer changed since the last download (sets *skipped).
	[[nodiscard]] uint64_t RecordImageDownload(Buffer& buffer, Common::SlotId image_id,
	                                           bool skip_unchanged = false,
	                                           bool* skipped       = nullptr);
	// A GPU write about to own [vaddr, vaddr + size) of buffer `id` takes GPU ownership away from
	// every overlapping GPU-modified image (TextureCache::InvalidateMemoryFromGPU), after which
	// the image is rebuilt from the buffer. Moves each such image's contents into the buffer
	// first and makes those bytes GPU-owned, so neither the rebuild nor a CPU readback sees the
	// stale guest bytes (KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE=0 disables). GPU thread, before the
	// writer is recorded and before InvalidateMemoryFromGPU.
	void PreserveImagesForGpuWrite(BufferId id, uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool ImageWritebackOnGpuWriteEnabled();
	// KYTY_FALSE_SHARING_WRITES (TryFalseSharingWrite): a download whose tracker pages were handed
	// to a CPU writer before its publication landed. GPU thread only, except `published` (set by
	// the publication, after it wrote the backing) and the verify fields (read by it).
	struct EarlyReleasedDownload {
		std::atomic<bool>       published {false};
		std::vector<GuestRange> ranges;   // the GPU-owned bytes being published
		std::vector<uint8_t>    snapshot; // verify: their guest bytes when the pages were released
		bool                    verify = false;
	};
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	// `early`: the pages are released before the publication lands (KYTY_FALSE_SHARING_WRITES).
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                        const std::shared_ptr<EarlyReleasedDownload>& early = {});
	struct ReadMemoryTrace {
		uint64_t begin      = 0;
		uint64_t size       = 0;
		bool     downloaded = false;
	};
	// The drain path: GPU thread only.
	void ReadMemoryDrain(uint64_t vaddr, uint64_t size, bool is_write, ReadMemoryTrace& trace);
	// KYTY_FALSE_SHARING_WRITES (default off; =1 on). A guest write fault on a GPU-owned tracker
	// page whose written bytes the GPU never wrote (m_gpu_modified_ranges is byte-exact while
	// protection is per page: e.g. a per-frame CPU block right after a GPU-written binding that
	// ends inside the page). Instead of draining the GPU (submit, wait, download, then release the
	// page), it queues the download of the page's GPU-owned bytes and releases the page at once:
	// CPU-dirty and writable, those bytes left to their publication. Until it lands, uploads of
	// the page skip them (the buffer keeps the GPU's bytes), HasGpuDirtyBytes reports them (images
	// refresh from the buffer, CP reads wait for the publication) and side readbacks there keep
	// draining. GPU thread, called for a write fault. Kept for:
	//  - write faults whose bytes the GPU owns, an unbounded writer not known complete, pages
	//    with an image over them, or pages outside one registered buffer: the drain as before.
	// NOT exact: until the publication lands (the GPU finishes the recording that copies them), a
	// guest CPU read of the page's GPU-owned bytes sees their old guest contents, and a guest CPU
	// write to them is overwritten by the publication. The drain made both see the GPU's bytes.
	// KYTY_FALSE_SHARING_WRITES_VERIFY=1|exit compares those bytes at publication with their
	// contents at the release: a difference is a CPU write the publication overwrites
	// (FalseSharingVerifyConflicts; exit stops). Reads cannot be observed.
	[[nodiscard]] bool TryFalseSharingWrite(uint64_t vaddr, uint64_t size, ReadMemoryTrace& trace);
	// Early-released bytes whose publication has not landed (GPU thread).
	void PruneEarlyReleased();
	// KYTY_FALSE_SHARING_WRITES_VERIFY, on the priority worker right before a publication.
	void VerifyEarlyRelease(const EarlyReleasedDownload& early);
	[[nodiscard]] bool OverlapsUnpublished(uint64_t address, uint64_t size) const;
	// emit(address, bytes) for each part of [address, address + size) outside every unpublished
	// early-released range, in address order (GPU thread).
	template <typename Emit>
	void ForEachPublishedPart(uint64_t address, uint64_t size, Emit&& emit) const;

	// Side readbacks. Issue runs on the GPU thread; completion on any thread.
	struct SideReadback;
	struct SideReadbackState;
	enum class SideIssueResult : uint8_t {
		Issued,
		Pending,
		CurrentWriter,
		Unbounded,
		Other,
	};
	[[nodiscard]] SideIssueResult TryIssueSideReadback(uint64_t vaddr, uint64_t size,
	                                                  std::shared_ptr<SideReadback>& issued);
	// Returns true when this call published the readback (false: it was already done).
	bool CompleteSideReadback(SideReadback& readback);
	[[nodiscard]] bool OverlapsPendingSideReadback(uint64_t begin, uint64_t end) const;
	// Every GPU-side write of cached buffer contents for a guest range (GPU thread).
	void NoteBufferContentWrite(uint64_t vaddr, uint64_t size);
	// A reader needed a readback of [vaddr, vaddr + size): its pages become read-hot (GPU thread).
	void NoteEagerRead(uint64_t vaddr, uint64_t size, bool gpu_thread_reader);
	[[nodiscard]] EagerReadbackPages::IssueResult TryIssueEagerReadback(uint64_t page,
	                                                                    uint64_t tick);

	struct BackingPublication {
		uint64_t                token;
		uint64_t                tick;
		std::vector<GuestRange> ranges;
	};
	mutable std::mutex               m_backing_publication_mutex;
	std::vector<BackingPublication> m_backing_publications;
	std::atomic<size_t>              m_backing_publication_count {0};
	uint64_t                        m_next_backing_publication_token = 0;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	// KYTY_BDA_PAGETABLE_SPARSE (GraphicContext::sparse_residency_buffer_enabled): the BDA page
	// table is a sparse residency buffer. Its blocks get zeroed memory, bound before any entry in
	// them is written (EnsureBdaTableResident, before ChangeRegister's write); an unbound block
	// reads as zero, as the dense table's unwritten entries do. Declared before the table so that
	// the memory is freed after the buffer is destroyed. Null for the dense table.
	struct SparsePageTable;
	std::unique_ptr<SparsePageTable>                  m_bda_sparse;
	void EnsureBdaTableResident(uint64_t first_page, uint64_t page_count);
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	struct KnownFillRange {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint32_t value   = 0;
	};
	void                                              ForgetKnownFills(uint64_t vaddr, uint64_t size);
	void                                              ForgetKnownFillsLocked(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::optional<uint32_t>             KnownFillLocked(uint64_t vaddr, uint64_t size) const;
	mutable std::mutex                                m_known_fill_mutex;
	std::vector<KnownFillRange>                       m_known_fills;
	// m_known_fills is nonempty; written under m_known_fill_mutex, read without it.
	std::atomic<bool>                                 m_has_known_fills {false};
	// Advanced under m_known_fill_mutex by every change of m_known_fills (KnownFillGeneration), and
	// by those made off the GPU thread (ForeignKnownFillChanges).
	std::atomic<uint64_t>                             m_known_fill_generation {1};
	std::atomic<uint64_t>                             m_known_fill_foreign {0};
	const bool                                        m_bda_incremental_sync;
	// KYTY_BDA_HOT_SYNC (default on; needs KYTY_BDA_INCREMENTAL_SYNC=1): with hot pages present,
	// a BDA pass whose fault and structure epochs are unchanged re-synchronizes only the hot page
	// runs the last full pass recorded (m_bda_hot_ranges) instead of every mapped buffer, and hot
	// pages are compared with their shadow in place before any snapshot is taken.
	const bool                                        m_bda_hot_sync;
	MemoryTracker                                     m_memory_tracker;
	// Hot pages: exact copy of the last contents uploaded for each hot page (GPU thread only).
	// While a page is hot its buffer bytes equal this copy: every other write of them either
	// returns the page to normal tracking first (written uploads, SettleHotPages for image
	// downloads and address writers, untracking) or keeps the bytes (joins copy them). A normal
	// upload of the page erases the copy.
	struct HotShadow {
		std::unique_ptr<uint8_t[]> data;
		uint32_t                   last_change = 0;
		uint32_t                   last_use    = 0;
		// Uploads in a row that found the page unchanged (KYTY_HOT_PAGE_CHECK_LIMIT).
		uint32_t                   unchanged_checks = 0;
	};
	std::map<uint64_t, HotShadow>                     m_hot_shadows;
	std::vector<uint8_t>                              m_hot_scratch;
	uint32_t                                          m_hot_quiet_frames = 8;
	// KYTY_HOT_PAGE_CHECK_LIMIT (default 64; 0 disables): a hot page that this many uploads in a
	// row found unchanged returns to normal tracking as a clean, write-protected page
	// (SettleHotPages, which compares it with its shadow once more after protecting it). Its next
	// write then faults as for any tracked page. Hot pages suit pages written between most
	// uploads; a page every BDA draw re-examines but the CPU rewrites about once a frame costs
	// hundreds of 4 KiB compares per frame instead of one fault.
	uint32_t                                          m_hot_check_limit  = 64;
	// KYTY_BUFFER_RANGE_MEMO (nullptr when disabled).
	std::unique_ptr<RangeMemo[]>                      m_range_memo;
	// Always counted (GPU thread; the BufferRangeMemo* frame events need a connected profiler).
	struct RangeMemoTotals {
		uint64_t clean_hits        = 0;
		uint64_t stream_hits       = 0;
		uint64_t records           = 0;
		uint64_t settles           = 0;
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
		uint64_t verify_races      = 0;
	};
	RangeMemoTotals                                   m_range_memo_totals;
	int                                               m_range_memo_verify = 0; // RangeMemoVerifyMode
	// KYTY_TRACKER_RELAXED_QUERIES outcomes, always counted (tests read them).
	struct RelaxedTotals {
		uint64_t queries    = 0; // lock-free dirty snapshots taken
		uint64_t sync_skips = 0; // read synchronizations found empty without a lock
		uint64_t mismatches = 0; // verify mode
	};
	RelaxedTotals                                     m_relaxed_totals;
	bool                                              m_relaxed_queries = false;
	// KYTY_BINDING_EPOCH_MEMO (nullptr when disabled; GPU thread) and its outcomes (tests read them).
	std::unique_ptr<BindingMemo[]>                    m_binding_memo;
	uint32_t                                          m_binding_memo_shift = 53; // 64 - log2(slots)
	int                                               m_binding_memo_verify = 0;
	bool                                              m_binding_memo_cross  = false;
	struct BindingMemoTotals {
		uint64_t hot_hits          = 0; // successful hits through the optional front tier
		uint64_t stream_hits       = 0;
		uint64_t cached_hits       = 0; // cross-epoch hits included
		uint64_t cross_hits        = 0;
		uint64_t cross_rejects     = 0;
		uint64_t records           = 0;
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
		uint64_t verify_races      = 0;
	};
	BindingMemoTotals                                 m_binding_memo_totals;
	// KYTY_WRITTEN_SYNC_SKIP (SynchronizeBuffer) and its outcomes (tests read them).
	bool m_written_sync_skip        = false;
	int  m_written_sync_skip_verify = 0;
	struct WrittenSyncTotals {
		uint64_t skips             = 0;
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
	};
	WrittenSyncTotals m_written_sync_totals;
	// KYTY_FALSE_SHARING_WRITES (TryFalseSharingWrite): switch, verify mode, the releases whose
	// publication has not been seen landed yet (GPU thread), and the outcomes (tests read them;
	// the verify counts are updated by the publications).
	bool m_false_sharing        = false;
	int  m_false_sharing_verify = 0;
	std::vector<std::shared_ptr<EarlyReleasedDownload>> m_early_released;
	struct FalseSharingTotals {
		uint64_t              writes        = 0;
		uint64_t              bytes         = 0;
		uint64_t              upload_splits = 0;
		std::atomic<uint64_t> verify_checks {0};
		std::atomic<uint64_t> verify_conflicts {0};
	};
	FalseSharingTotals m_false_sharing_totals;
	uint32_t                                          m_upload_batch_depth = 0;
	uint32_t                                          m_hot_sweep_frame  = 0;
	std::atomic_uint64_t                               m_bda_structure_epoch {1};
	// GPU-thread-only snapshots taken BEFORE the last full scan, never after it.
	uint64_t                                          m_bda_scanned_cpu_epoch = 0;
	uint64_t                                          m_bda_scanned_structure_epoch = 0;
	// Hot page runs inside the buffers the last full BDA pass scanned (KYTY_BDA_HOT_SYNC; GPU
	// thread). Valid while the epochs of that pass hold: a page can only become hot through a
	// write fault, which changes the fault epoch, and buffers only change with the structure one.
	// Dirty-log passes add the hot runs they find in the logged ranges.
	std::vector<BdaHotRange>                          m_bda_hot_ranges;
	// KYTY_BDA_DIRTY_LOG (default on with KYTY_BDA_HOT_SYNC; =0 off). A BDA pass whose structure
	// epoch is unchanged but whose fault epoch moved synchronizes only the ranges the tracker
	// logged since the last pass (MemoryTracker::TakeDirtiedRanges, taken with the epoch they
	// account for) and the recorded hot runs, instead of every mapped buffer. After the last pass
	// every page of a mapped buffer was clean or hot; one can only have turned CPU-dirty since
	// through a transition that advanced the fault epoch, and each such transition logged its
	// range before advancing it, under its region lock and before changing the page's bits. A
	// full scan runs when the log overflowed, the buffers changed (structure epoch), or no full
	// scan with the log has run yet (m_bda_log_baseline).
	bool                                              m_bda_dirty_log    = false;
	bool                                              m_bda_log_baseline = false;
	int                                               m_bda_log_verify   = 0;
	RangeSet                                          m_bda_dirtied;
	struct BdaLogTotals {
		uint64_t passes                = 0;
		uint64_t ranges                = 0;
		uint64_t overflows             = 0;
		uint64_t verify_checks         = 0;
		uint64_t verify_mismatch_pages = 0;
	};
	BdaLogTotals                                      m_bda_log_totals;
	// KYTY_BDA_SYNC_EPOCH (SynchronizeBdaBuffers): the sync, BDA structure and fault epochs taken
	// before the last completed pass (GPU thread; 0: none yet), and the outcomes (tests read them).
	bool     m_bda_epoch_skip        = false;
	int      m_bda_epoch_verify      = 0;
	uint64_t m_bda_synced_epoch      = 0;
	uint64_t m_bda_synced_structure  = 0;
	uint64_t m_bda_synced_fault      = 0;
	// KYTY_BDA_SYNC_PER_SUBMISSION (default off): the guest submission (SyncEpoch::
	// CurrentSubmission) taken before the last completed pass; later passes of that submission
	// are skipped while the structure epoch holds.
	bool     m_bda_submission_skip   = false;
	uint64_t m_bda_synced_submission = 0;
	struct BdaEpochTotals {
		uint64_t passes                = 0;
		uint64_t skips                 = 0;
		uint64_t submission_skips      = 0;
		uint64_t verify_checks         = 0;
		uint64_t verify_mismatch_pages = 0;
	};
	BdaEpochTotals m_bda_epoch_totals;
	StreamBuffer                                      m_staging_buffer;
	// After the staging ring: destroyed first, waiting for its copies that read the ring.
	std::unique_ptr<UploadDma>                        m_upload_dma;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	// KYTY_VRAM_IDLE_BUFFER_FRAMES=N (default 0: off; see TextureCache's KYTY_VRAM_IDLE_FRAMES):
	// once every 32 frames, cache buffers unused for more than N presented frames without GPU-dirty
	// bytes are untracked and deleted, as the collector below deletes them, whatever the memory
	// usage. A later use recreates the buffer and uploads the guest bytes. Caution: a buffer only
	// shaders' BDA pointers read is never touched (no binding), so it looks unused while it is not.
	// KYTY_VRAM_PRESSURE_BUFFER_FRAMES=K: the same with K frames replaces the submission-age
	// collection above the trigger. m_frame_ticks: the GC tick each recent frame began at.
	void                                              RetireUnusedBuffers(uint64_t frame, uint64_t min_age);
	uint64_t                                          m_idle_frames      = 0;
	uint64_t                                          m_idle_next_frame  = 0;
	uint64_t                                          m_idle_freed       = 0;
	uint64_t                                          m_idle_freed_bytes = 0;
	uint64_t                                          m_pressure_frames  = 0;
	uint64_t                                          m_pressure_frame   = 0;
	// KYTY_VRAM_GC_BUDGET: the marks from the budget, read once per frame.
	uint64_t                                          m_budget_frame     = UINT64_MAX;
	uint64_t                                          m_budget_trigger   = 0;
	uint64_t                                          m_budget_critical  = 0;
	std::deque<std::pair<uint64_t, uint64_t>>         m_frame_ticks;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	uint64_t m_content_revision_epoch = 1;
	// Writer ticks of buffer contents (GPU thread). Missing ranges are no newer than the floor.
	WriteTickMap m_write_ticks;
	uint64_t     m_write_tick_floor      = 0;
	size_t       m_write_tick_prune_size = 1024;
	// Recording tick that holds an unbounded (address) GPU writer; 0 when none.
	uint64_t     m_unbounded_write_tick  = 0;
	uint64_t     m_unbounded_write_serial = 0;
	std::unique_ptr<SideReadbackState> m_side;
	// Eager readback publication (GPU thread): read-hot pages, and the early-submission request a
	// recorded writer of a GPU-thread-read page makes (with its per-frame budget).
	bool               m_eager_enabled       = false;
	EagerReadbackPages m_eager;
	bool               m_eager_flush         = false;
	uint32_t           m_eager_flush_budget  = 0;
	uint32_t           m_eager_flush_frame   = 0;
	uint32_t           m_eager_flushes       = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
