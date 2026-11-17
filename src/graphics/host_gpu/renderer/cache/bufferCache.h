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
#include "graphics/host_gpu/renderer/cache/streamCopyMemo.h"
#include "graphics/host_gpu/writeTickMap.h"

#include <array>
#include <atomic>
#include <string>
#include <chrono>
#include <deque>
#include <unordered_map>
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
class CommandRecorder;
class InplaceHeap;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

// KYTY_STREAM_COPY_DEFER=1 (default off): the stream copy of a small CPU-dirty read binding
// (ObtainBufferNow) is reserved in the stream ring on the command processor, but the bytes are
// copied by the recorder thread (KYTY_CP_RECORDER, CommandStream::Op::HostCopy) in stream order,
// so before the Submit of the commands that read them: about 2,300 copies, 7.5 MB and 0.45 ms of
// the command processor per Sky Garden frame. The guest bytes are then read later than the
// command processor's draw, but still before the GPU work runs, as the PS5's GPU reads them.
// Every command-processor write of guest memory between the two (NoteCpWrite, backing writes on
// the GPU thread) that overlaps a pending copy drains the recorder first, so such a write is
// never seen by an earlier draw. Off with the recorder off, in a direct window, with the memo
// verify modes, or a non-coherent stream ring.
namespace StreamCopyDefer {
[[nodiscard]] bool Enabled();
// GPU thread: a command-processor write of guest memory [address, address + size).
void NoteGuestWrite(uint64_t address, uint64_t size);
// The same for a CP write that cannot tell the guest that earlier GPU work is done before it is:
// the constant engine's DUMP_CONST_RAM, label and result writes made after their tick completed
// (that tick's submission waited for every host copy staged in or before it), timestamps, LOD
// stats, synthetic occlusion results. KYTY_UPLOAD_DMA_WRITE_DRAIN only orders it after the queued
// host copies that read the same bytes. NoteGuestWrite (labels written at CP time, WRITE_DATA,
// flip labels) makes every queued host copy first.
void NoteGpuOnlyWrite(uint64_t address, uint64_t size);
// GPU thread, at a flip (the frame boundary): KYTY_UPLOAD_DMA_WRITE_DRAIN=frame makes every
// queued host copy first (uploadDma.h).
void NoteFrameBoundary();
// GPU thread: KYTY_UPLOAD_DMA_WRITE_DRAIN=defer and host copies are queued, so a label written now
// is to be written at its tick's completion instead (CommandProcessor::TryDeferLabel).
[[nodiscard]] bool LabelWaitsForHostCopies();
// The stream-copy part both share (internal).
void NoteGuestWriteStream(uint64_t address, uint64_t size);
} // namespace StreamCopyDefer

// KYTY_BDA_WRITES=1 (or =verify): shaders store and do atomics through V#s they compute, through
// BDA, and the renderer settles each such dispatch synchronously (BufferCache::SettleBdaWrites).
[[nodiscard]] bool BdaWritesEnabled();
// KYTY_BDA_WRITES=verify: dropped writes and written pages under a GPU-modified image are fatal.
[[nodiscard]] bool BdaWritesVerify();

// Reads an image's guest range when part of it is not mapped (streamed textures map only their
// resident part): mapped bytes as they are, unmapped bytes as zero. False when nothing can be read.
[[nodiscard]] bool ReadPartlyUnmappedImageBacking(uint64_t vaddr, void* data, uint64_t size);

struct BufferContentRevision {
	BufferId id;
	uint64_t write_revision;
	uint64_t global_epoch;
	bool operator==(const BufferContentRevision&) const noexcept = default;
};

// KYTY_BDA_ACCESS_TAGS diagnostics, one counter per hop of a report (printed with BdaTags):
// programs emitted with the tag check, fault-buffer scans recorded and read back on the host,
// fault records read, and PrepareBda calls (operations that read through device addresses).
struct BdaTagHops {
	std::atomic<uint64_t> programs {0};
	std::atomic<uint64_t> scans {0};
	std::atomic<uint64_t> scans_read {0};
	std::atomic<uint64_t> records {0};
	std::atomic<uint64_t> prepares {0};
	// KYTY_BDA_TAGS_GPU_COUNT: the GPU counters, accumulated from their readbacks.
	std::atomic<uint64_t> gpu_calls {0};
	std::atomic<uint64_t> gpu_zero {0};
	std::atomic<uint64_t> gpu_untagged {0};
};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = uint64_t {1} << (40 - CACHING_PAGEBITS);
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// KYTY_RT_HW: the entry after the last page holds the device address of the native ray tracing
	// context (NativeRayTracing); guest addresses never reach it (they index pages below
	// CACHING_NUMPAGES). The page table buffer has room for it (BDA_PAGETABLE_ALLOCATION: one sparse
	// block more; EnsureBdaPageTableEntry makes it resident in a sparse table).
	static constexpr uint64_t NATIVE_RT_CONTEXT_PAGE = CACHING_NUMPAGES;
	static constexpr uint64_t BDA_PAGETABLE_ALLOCATION = BDA_PAGETABLE_SIZE + 65536;
	// Fault buffer layout: the fault bitmap (1 bit per page). With KYTY_BDA_WRITES it grows by the
	// written-page bitmap of the same shape, the dropped-write counter (writes to pages without a
	// cache buffer) and the store log (KYTY_BDA_STORE_LOG: a u32 count, then entries of 4 words
	// from BDA_STORE_LOG_WORD + 4: guest address low and high, the dword value and its byte mask
	// as bits), in that order.
	static constexpr uint64_t FAULT_BITMAP_WORDS      = CACHING_NUMPAGES / 32;
	static constexpr uint64_t BDA_WRITE_BITMAP_WORD   = FAULT_BITMAP_WORDS;
	static constexpr uint64_t BDA_DROPPED_WRITES_WORD = 2 * FAULT_BITMAP_WORDS;
	static constexpr uint64_t BDA_STORE_LOG_WORD      = BDA_DROPPED_WRITES_WORD + 4;
	// The log's capacity, a constant of the generated shaders: one dispatch's stores past it are
	// dropped. A settle downloads the first BDA_STORE_LOG_PREFIX entries with the written-page list;
	// a synchronous settle fetches the rest with a second copy when the log holds more
	// (FaultManager::FetchBdaStoreLogTail), so a settle without such a dispatch costs what it did
	// with a 4096-entry log. That capacity lost 16639 of the 20735 stores of a first dispatch into an
	// output array without a cache buffer (one dword per thread group; another title), and the
	// array's consumer hung the GPU.
	static constexpr uint64_t BDA_STORE_LOG_ENTRIES   = 262144;
	static constexpr uint64_t BDA_STORE_LOG_PREFIX    = 4096;
	static constexpr uint64_t BDA_STORE_LOG_BYTES     = (4 + 4 * BDA_STORE_LOG_ENTRIES) * sizeof(uint32_t);
	static constexpr uint64_t BDA_STORE_LOG_PREFIX_BYTES =
	    (4 + 4 * BDA_STORE_LOG_PREFIX) * sizeof(uint32_t);
	static constexpr uint64_t BDA_WRITES_FAULT_BUFFER_SIZE =
	    BDA_STORE_LOG_WORD * sizeof(uint32_t) + BDA_STORE_LOG_BYTES;
	// KYTY_BDA_TAGS_GPU_COUNT: get_bda_pointer calls, zero-entry calls, untagged-entry calls (u32,
	// wrapping) after the largest layout; the fault buffer grows by 16 bytes for them.
	static constexpr uint64_t BDA_TAG_COUNTER_WORD = BDA_WRITES_FAULT_BUFFER_SIZE / sizeof(uint32_t);

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	// write_fault: a guest write fault on the range (fault-ahead / hot-page policy applies).
	void                   InvalidateMemory(uint64_t vaddr, uint64_t size, bool write_fault = false);
	// Once per completed guest flip (any thread): the frame clock of hot-page detection.
	void                   AdvanceFrame() noexcept;
	// KYTY_INPLACE (inplaceHeap.h): a command-processor CPU write of guest memory (DMA_DATA) is
	// about to happen. Any thread.
	void                   InplaceBeforeCpWrite(uint64_t vaddr, uint64_t size);
	// KYTY_BDA_LIVE_PAGES = 0 | 1 | verify (bufferCacheLive.cpp; default 0, live; the state exists
	// only when the variable is set at start). A CPU-dirty page of a BDA-mapped buffer that the full
	// pass would upload is instead served to BDA reads from the guest bytes themselves: its page-table
	// entry points into an imported window of the guest backing alias, and the page stays CPU-dirty
	// and writable (no upload, no re-protection, no write fault). Invariants: rtskip DESIGN.md 4.
	[[nodiscard]] int64_t BdaLivePagesMode() const;
	[[nodiscard]] bool    BdaLivePagesBuilt() const { return m_bda_live != nullptr; }
	// Before a full BDA pass (RenderContext::PrepareBda): restores the pages a tag report or the
	// CP blacklist promoted, restores everything when the switch is 0, and uploads the forced set
	// (entries given back without an upload). GPU thread.
	void                   LivePagesPrePass();
	// A command-processor or emulator write of guest memory is about to happen (any thread; waits on
	// the GPU thread when it hits a page earlier recorded work can read live).
	void                   LivePagesBeforeCpWrite(uint64_t vaddr, uint64_t size);
	// The same for the StreamCopyDefer write hooks, on the cache that built the state (if any).
	static void            LiveCpWriteHook(uint64_t vaddr, uint64_t size);
	// Device-loss report lines of the state (SetDeviceLossNote; tries the lock, never waits).
	static void            LiveDeviceLossNote();
	// Guest unmap of [vaddr, vaddr + size): every redirect in it goes back (GPU thread).
	void                   DropLivePages(uint64_t vaddr, uint64_t size);
	// Whether the caching page of `vaddr` is served live (tests).
	[[nodiscard]] bool     IsBdaLivePage(uint64_t vaddr) const;
	// Whether it is served live and settled (a read upload made its cache copy current; tests).
	[[nodiscard]] bool     IsBdaLivePageSettled(uint64_t vaddr) const;
	// Tests only: marks the range GPU-dirty in the tracker without a written binding (the injection
	// the V5 check must catch).
	void                   TestingMarkGpuDirty(uint64_t vaddr, uint64_t size, bool mark = true);
	// Verify V5 over the settled pages now (the pre-pass runs it in verify mode); returns the count.
	uint64_t               CheckSettledLivePages();
	struct BdaLiveTotals {
		uint64_t redirects        = 0; // pages a full pass redirected instead of uploading
		uint64_t held             = 0; // pages redirected now
		uint64_t restore_upload   = 0; // restores: an upload covered the page
		uint64_t restore_written  = 0; // restores: a written binding over the page
		uint64_t restore_register = 0; // redirects dropped by a buffer (un)registration
		uint64_t restore_unmap    = 0; // restores: guest unmap
		uint64_t restore_read     = 0; // restores: a shader read the page live (tag report)
		uint64_t restore_cp       = 0; // restores: CP blacklist
		uint64_t restore_off      = 0; // restores: switch 0
		uint64_t restore_writes   = 0; // restores: before a BDA-writing program
		uint64_t forced_syncs     = 0; // forced-set ranges synchronized by the pre-pass
		uint64_t live_reads       = 0; // tag reports of pages read through a redirect
		uint64_t cp_hits          = 0; // CP / emulator writes over a redirected page
		uint64_t cp_waits         = 0; // of them, GPU waits on the GPU thread
		uint64_t cp_unordered     = 0; // of them, on another thread (counted only)
		uint64_t rejects_window   = 0; // candidate pages without an imported window
		uint64_t rejects_bound    = 0; // candidate pages a cache binding uploaded after a redirect
		uint64_t blocked_passes   = 0; // passes with BDA-write settles pending (no redirects)
		uint64_t faults           = 0; // guest write faults on tracked memory
		uint64_t v1               = 0; // verify: CPU-dirty pages left on the cache entry (must be 0)
		uint64_t v1_checks        = 0;
		uint64_t v1_skipped       = 0; // checks skipped: the fault epoch moved during the pass
		uint64_t v2               = 0; // verify: redirected pages that are GPU-dirty / hot / image-written
		uint64_t v2_gpu           = 0; //   of them GPU-dirty (bits or bytes)
		uint64_t v2_image         = 0; //   of them under a GPU-modified image range
		uint64_t v2_hot           = 0; //   of them hot (a hot page is CPU-dirty: not a stale read by itself)
		uint64_t v3               = 0; // verify: redirects alive at a BDA-writing program
		uint64_t v4               = 0; // verify: forced pages still CPU-dirty after their upload (I3, must be 0)
		uint64_t v4_checks        = 0;
		uint64_t v5               = 0; // verify: settled pages GPU-dirty / image-written at the pre-pass (I-B1)
		uint64_t v5_checks        = 0;
		uint64_t v6               = 0; // entry writes while a staging reservation was open (I-S1, must be 0)
		// KYTY_BDA_LIVE_BATCH (DESIGN.md 4.4)
		uint64_t settled          = 0; // read uploads over a redirected page (restore left to the pre-pass)
		uint64_t entry_copies     = 0; // page-table copy commands written for redirects / restores
		uint64_t entry_copies_rendering = 0; // of them issued inside an active rendering instance
		uint64_t entry_pages      = 0; // entries written by them
		// Qualification (LiveSyncBufferRange)
		uint64_t qual_calls       = 0; // calls during full passes
		uint64_t qual_fast        = 0; // of them taking the no-dirty, no-redirect fast path
		uint64_t qual_pages       = 0; // caching pages walked
		uint64_t qual_dirty       = 0; // of them CPU-dirty candidates that reached the locked checks
		uint64_t qual_segments    = 0; // SynchronizeBuffer calls the walk split the range into
		uint64_t qual_ns          = 0; // time in LiveSyncBufferRange (includes those synchronizations)
		uint64_t prepass_calls    = 0; // LivePagesPrePass calls (one per PrepareBda)
		uint64_t prepass_work     = 0; // of them with something pending
		uint64_t windows          = 0;
		uint64_t pinned_bytes     = 0;
		uint64_t import_failures  = 0;
	};
	[[nodiscard]] BdaLiveTotals GetBdaLiveTotals() const;
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
	// `retried_write`: a write fault whose instruction runs again after the handler returns
	// (KYTY_READBACK_SIDE_WRITES may then leave the page clean for that second fault).
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false,
	                                  bool retried_write = false);
	// GPU thread (adapted from usa-ratt/KytyPS5-GT7 21246d78a and 41848bb9a, BufferCache::
	// WriteClean): a game thread's faulting store of [vaddr, vaddr + size), done in place on both
	// copies: guest memory through the backing store, the cached buffer in the command stream
	// after the GPU's earlier writes. The page stays protected. `gpu_written`
	// (KYTY_STORE_IN_PLACE): pages protected because the GPU wrote them, which then need no
	// download first; `clean` (KYTY_STORE_CLEAN): CPU-clean pages watched for CPU writes, which
	// then stay clean and protected instead of becoming CPU-dirty for the next sync to upload and
	// protect again. False to decline (nothing to save, a readback or publication of these bytes
	// is still to land, an image holds GPU-written bytes there, no cached buffer holds them): the
	// fault then takes the normal path.
	[[nodiscard]] bool     StoreInPlace(uint64_t vaddr, const void* data, uint64_t size,
	                                    bool gpu_written, bool clean);
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
	// KYTY_DMA_PREFAULT_KB (GPU thread): a shader that reads guest memory through its page table
	// (DMA) reads a page no cached buffer covers as zeros and only records the fault. The
	// committed pages of [vaddr, vaddr + size) without a cached buffer get one, filled from guest
	// memory; those with one are kept fresh in the LRU. Returns how many pages had none.
	uint32_t MakeResident(uint64_t vaddr, uint64_t size);
	// KYTY_BDA_HEAPS: a BDA access faulted at [vaddr, vaddr + size): the GPU-mapped allocation holding it
	// becomes resident now and before every later BDA-writing dispatch (its other pages are the
	// destinations of that and the next dispatches, which would drop their stores and atomics).
	void NoteBdaFault(uint64_t vaddr, uint64_t size);
	// KYTY_STREAM_COPY_REUSE (streamCopyMemo.h; command-processor thread): a packet that may write or
	// wait on guest memory, or a new submission slice, was processed: no small stream copy made
	// before it is reused after it.
	void NewStreamEpoch() noexcept { m_stream_copy_memo.NewEpoch(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	// KYTY_KEEP_BINDINGS (descriptors.cpp, RebindBuffers; GPU thread): a read binding (not written,
	// not a texel read) whose ObtainBuffer was a cache-buffer result the binding memo now holds keeps
	// that result for the next identical binding while nothing it depends on moved: no region serial
	// anywhere (RegionManager::AnySerial, so every RangeSignature is unchanged), the same sync epoch
	// and buffer structure epoch (the memo's guard). A normal lookup then hits the same memo entry
	// (same key, signature, guard and epoch) and returns the same buffer and offset; the keep only
	// repeats the hit's touch. KeepBegin is read before the lookup, KeepCapture after it.
	struct KeepStamp {
		uint64_t token     = 0; // 0: no capture
		uint64_t epoch     = 0;
		uint64_t structure = 0;
	};
	struct KeptRead {
		KeepStamp stamp {};
		uint64_t  vaddr   = 0;
		uint64_t  size    = 0;
		BufferId  id {};
		Buffer*   buffer  = nullptr;
		uint64_t  offset  = 0;
		uint64_t  touched = UINT64_MAX;
	};
	[[nodiscard]] KeepStamp KeepBegin() const noexcept;
	// True when `kept` now holds a result a later KeepValid may return.
	bool KeepCapture(KeptRead& kept, const KeepStamp& stamp, uint64_t vaddr, uint64_t size, BufferId id,
	                 const std::pair<Buffer*, uint64_t>& result);
	// Whether ObtainBuffer(vaddr, size, false, false, id) would return kept's buffer and offset now
	// (then touched as a hit is).
	[[nodiscard]] bool KeepValid(KeptRead& kept, uint64_t vaddr, uint64_t size, BufferId id);
	// KYTY_RAW_READ_IMAGE_SYNC=1 (default off; Ghost of Yotei preset): a shader's raw (unformatted)
	// read-only binding whose range starts at a GPU-modified image takes the image's contents
	// first, as a texel read does (SynchronizeBufferFromImage); the caller passes the buffer its
	// ObtainBuffer returned. Without it such a read sees the buffer's stale bytes: Yotei's CS
	// 4337635433d63234 copies the sky LUTs that CS 95c013c75f8e77ad writes as storage images
	// through raw buffers and read zeros (cloud light direction wrong, cloud march skipped, dark
	// shadow combine and water). Port of Erlou93/KytyPS5-Yotei SyncBufferFromWrittenImage (there
	// inside ObtainBuffer, minus its image sources). Returns whether bytes were synchronized.
	// GPU thread, outside every tracker and texture-cache lock.
	[[nodiscard]] bool SyncRawReadFromWrittenImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	[[nodiscard]] static bool RawReadImageSyncEnabled();
	// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS=1 (default off; PPSA26344 preset; completes the port of
	// Erlou93/KytyPS5-Yotei SyncBufferFromWrittenImage, which runs inside every read-only ObtainBuffer):
	// the read-only obtains that are not shader bindings - index buffers, vertex buffer ranges, draw and
	// dispatch indirect arguments and counts - take a GPU-written image's contents as raw shader reads
	// do (SyncRawReadFromWrittenImage). Needs KYTY_RAW_READ_IMAGE_SYNC=1.
	// KYTY_RAW_READ_IMAGE_SYNC_ONCE=1 (default off; PPSA26344 preset; Erlou's buffer_sync_serial gate):
	// an image's contents are copied into a buffer once per image content serial (and buffer, start and
	// size) instead of on every read while the image stays GPU-modified.
	// Called before the obtain itself: the synchronization can join cache buffers, as ObtainBuffer can.
	void PrepareReadRange(uint64_t vaddr, uint64_t size);
	// KYTY_CP_BINDING_BATCH_PREFETCH (default on, live): a CPU cache hint for the binding memo
	// slot a later ObtainBuffer(vaddr, size, false, false) on the GPU thread looks up. A stage's
	// bindings issue all hints first (RenderExecutor::RebindBuffers), so the slots' cache misses
	// overlap instead of each lookup waiting for its own (the 32,768-slot table is 2 MiB; the
	// slot load was 4.1% of the command processor at the Sky Garden start view). No state
	// changes: every key, signature, epoch and guard check of the lookup still runs.
	// KYTY_CP_BINDING_BATCH_PREFETCH=find: the hints are issued earlier, by RenderExecutor::
	// FindBuffers for each read binding it discovers (before the stages' image work), instead of a
	// separate pass in RebindBuffers. `at_find` says which caller asks.
	void PrefetchReadBinding(uint64_t vaddr, uint64_t size, bool at_find = false) const noexcept;
	// A writable binding whose shader can only write `written` (sub-ranges of [vaddr, vaddr +
	// size), e.g. from a write-range proof): the whole range is synchronized as for any binding,
	// but only `written` becomes GPU-owned (dirty, protected, write-ticked).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainWrittenBuffer(uint64_t vaddr, uint64_t size,
	                                                               std::span<const GuestRange> written,
	                                                               BufferId id = {});
	[[nodiscard]] static bool ShaderWriteRetickEnabled();
	// Binding preparation may submit before its draw/dispatch is recorded. Retain that
	// command's final producer tick without expanding its GPU-dirty or protected ranges.
	void RetagShaderWrite(uint64_t vaddr, uint64_t size, uint64_t preparation_tick);
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
	// Both the first registration and the first consumer clear the table, so unowned entries read as zero.
	[[nodiscard]] Buffer* GetBdaPageTableBuffer();
	// The page table entry index (one past the guest pages: NATIVE_RT_CONTEXT_PAGE) has memory.
	void EnsureBdaPageTableEntry(uint64_t index);
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// The guest bytes of a range, staged for an image upload whatever buffer covers them (the
	// caller knows the GPU wrote none of them; ObtainBufferForImage copies from such a buffer).
	[[nodiscard]] std::pair<Buffer*, uint64_t> StageGuestBytesForImage(uint64_t vaddr, uint64_t size);
	// KYTY_PENDING_FILLS=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7 bdab60022): a
	// GPU fill (FillBuffer's GPU path, or a compute shader's uniform fill, both recorded as known
	// fills) that found no image still holds for its range, the bytes of it that were not mapped
	// then among them: a render or depth target created later inside it starts as the fill
	// (TextureCache::ClearFromKnownFill) instead of what its memory holds then.
	[[nodiscard]] static bool PendingFillsEnabled();
	// GPU thread, for a target made over [vaddr, vaddr + size): the dword of the known fill that
	// holds all of it, when the mapped bytes of it are all still GPU-written. `take` forgets the
	// range from the fill, which keeps the rest of it for other images.
	[[nodiscard]] bool FillOfNewImage(uint64_t vaddr, uint64_t size, uint32_t& value, bool take);
	// KYTY_PENDING_FILLS: a write over [vaddr, vaddr + size) the other paths do not see (an image's
	// GPU write, a CPU-path fill or copy), newer than the fills there.
	void ForgetFillsUnder(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	// GPU thread, KYTY_SPARSE_FILLS: a GPU fill of the mapped `spans` of a range with holes in its
	// mappings (a compute shader's fill of sparse memory), as the hardware's writes land there;
	// the holes, where the hardware drops the writes, are left alone.
	void FillMappedSpans(std::span<const std::pair<uint64_t, uint64_t>> spans, uint32_t value);
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
	// Every byte of the range is exact GPU-dirty data (upstream 9678c59b7, storage readback).
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) const {
		return m_gpu_modified_ranges.Contains(vaddr, size);
	}
	// Diagnostics (KYTY_CS_VERIFY): the registered cache buffer holding [vaddr, vaddr + size) and the
	// offset there, without any synchronization (nullptr: none). GPU thread.
	[[nodiscard]] std::pair<Buffer*, uint64_t> DebugPeekBuffer(uint64_t vaddr, uint64_t size);
	// Diagnostics: whether [vaddr, vaddr + size) has CPU-modified (not yet uploaded) bytes.
	[[nodiscard]] bool DebugCpuDirty(uint64_t vaddr, uint64_t size);
	// KYTY_SRT_GPU_FILL (GPU thread): the guest dword at `vaddr` is GPU-written bytes that one
	// registered buffer holds, with nothing newer elsewhere (no image over it, no pending
	// publication or early release), so a GPU copy from that buffer reads its current value.
	// The buffer's handle and the dword's offset in it go to `handle` / `offset` when not null.
	// No buffer is created or synchronized.
	[[nodiscard]] bool GpuFillSource(uint64_t vaddr, vk::Buffer* handle = nullptr,
	                                 uint64_t* offset = nullptr);
	// KYTY_READBACK_FLUSH_SIDE (bufferCache.cpp): guest reads served by a side copy after the
	// command processor submitted the recording that wrote their bytes (process-wide; tests).
	[[nodiscard]] static uint64_t ReadbackFlushSideCopies() noexcept;
	// KYTY_READBACK_SIDE_WRITES: guest write faults on GPU-owned pages served by a side copy.
	[[nodiscard]] static uint64_t ReadbackSideWriteCopies() noexcept;
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
	// Buffers unregistered so far (a join retires the buffers it absorbs, an unmap or the
	// collector deletes them): a draw or dispatch whose resolved bindings were made before the
	// count moved may hold a retired buffer (KYTY_REBIND_AFTER_JOIN, RenderExecutor).
	[[nodiscard]] uint64_t UnregisterEpoch() const noexcept { return m_unregister_epoch; }
	// Publications can outlive cache ownership. Registration is on the GPU thread; completion
	// is on the priority worker, and these queries never wait for GPU work or backing writes.
	[[nodiscard]] uint64_t BeginBackingPublication(std::span<const GuestRange> ranges,
	                                               uint64_t tick);
	void EndBackingPublication(uint64_t token);
	[[nodiscard]] bool HasPendingBackingPublication(uint64_t vaddr, uint64_t size) const;
	// GPU thread (KYTY_DIRECT_READ_WAIT): waits until no queued backing publication covers the range,
	// so a direct read of guest memory sees the bytes a download in flight is about to publish.
	void WaitBackingPublications(uint64_t vaddr, uint64_t size);
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
	void               ProcessFaultBuffer(const std::vector<std::pair<uint64_t, uint64_t>>& mapped = {});
	// Caller holds the mapped-range lock. Cache/tracker iteration remains on the GPU thread.
	void               SynchronizeBdaBuffers(const RangeSet& mapped_ranges);
	// Map/unmap callers may run outside the GPU thread, under the mapped-range lock.
	void               InvalidateBdaSynchronization() noexcept;
	// KYTY_BDA_REGISTER_LOG: a new registered buffer's range, logged for the next BDA pass
	// (ChangeRegister<true>); every other structure change goes through
	// InvalidateBdaSynchronization.
	void               NoteBdaRegistered(uint64_t vaddr, uint64_t size);
	// KYTY_BDA_REACH (bufferCacheReach.cpp; based on TheCruZ/KytyPS5-GTA SynchronizeDmaFootprint):
	// uploads the CPU writes of the buffers in `spans` (sorted, disjoint, caching-page aligned,
	// inside the GPU mappings) for an operation whose device-address reads cannot reach other
	// memory, instead of a full BDA pass. verify: the checks described in bufferCacheReach.cpp.
	// Returns the upload bytes. GPU thread.
	uint64_t           SynchronizeBdaReach(std::span<const std::pair<uint64_t, uint64_t>> spans, bool verify);
	// After a full BDA pass (or its skip) and whenever verification is off: every page-table entry
	// the verification left zero gets its buffer's address back. GPU thread.
	// keep_verifying false: the switch left verify (the verification stops).
	void               RestoreBdaReachTraps(bool keep_verifying);
	// The 4-byte aligned dwords at `address` as the GPU will read them: from the backing store, when
	// the range is not GPU-modified (no newer bytes on the GPU). False otherwise.
	[[nodiscard]] bool ReadGuestForBdaReach(uint64_t address, std::span<uint32_t> words);
	// Verification totals since start (any thread reads them for the report).
	struct BdaReachVerifyTotals {
		uint64_t checks          = 0; // limited synchronizations checked
		uint64_t inside_dirty    = 0; // normal CPU-dirty tracker pages left inside the reach (must be 0)
		uint64_t trapped         = 0; // caching pages whose entry was zeroed (dirty, outside the reach)
		uint64_t restored        = 0; // entries given back
		uint64_t trap_faults     = 0; // GPU page faults on pages the verification zeroed (must be 0)
		uint64_t log_overflows   = 0; // reach-log losses (pages dirtied then are not trapped)
	};
	[[nodiscard]] BdaReachVerifyTotals GetBdaReachVerifyTotals() const;
	// FaultManager: a page fault the GPU recorded at [start, start + size).
	void               NoteBdaReachFault(uint64_t start, uint64_t size);
	// KYTY_BDA_ACCESS_TAGS (bufferCacheTags.cpp; based on upstream KytyPS5 0ec2cd5d5): bit 0 of a
	// page-table entry tags a page a DMA shader accessed (shaders built with the codegen option
	// report untagged pages through the fault bitmap). Live policy, only with that codegen option:
	// 0 = the full BDA pass; 1 = the full pass uploads only tagged pages and buffers registered
	// since the last pass; verify = the full pass, plus the miss count described in
	// bufferCacheTags.cpp.
	static constexpr vk::DeviceAddress BDA_ACCESSED = 1;
	[[nodiscard]] int64_t BdaAccessTagsMode() const;
	[[nodiscard]] bool    BdaAccessTagsBuilt() const { return m_bda_tags != nullptr; }
	// The 10 s console line (BdaTags) while the codegen option is on. GPU thread.
	void               ReportBdaTags();
	// The tagged pass (mode 1) over the GPU mappings. Returns the upload bytes. GPU thread.
	uint64_t           SynchronizeBdaTagged(const RangeSet& mapped_ranges);
	// Mode verify, before the full pass: records the CPU-dirty pages the tagged pass would skip.
	void               PrepareBdaTagVerify();
	// After a full pass (modes 0 and verify): the buffers registered so far are uploaded, so they
	// leave the pending set (verify then models a tagged pass that ran at the same points).
	void               NoteBdaTagFullPass();
	// FaultManager: pages a DMA shader reported (after FindBuffer made them backed where possible).
	void               NoteBdaTagFault(uint64_t start, uint64_t size);
	struct BdaTagTotals {
		uint64_t tagged_pages   = 0; // pages tagged so far
		uint64_t tagged_passes  = 0; // mode 1 passes
		uint64_t tagged_bytes   = 0; // their uploads
		uint64_t verify_passes  = 0; // mode verify passes
		uint64_t would_skip     = 0; // pages a verify pass found the tagged pass would leave stale
		uint64_t misses         = 0; // such pages a DMA shader then reported (must be 0)
		uint64_t log_overflows  = 0; // reach-log losses during verify
	};
	[[nodiscard]] BdaTagTotals GetBdaTagTotals() const;
	inline static BdaTagHops s_tag_hops; // KYTY_BDA_ACCESS_TAGS diagnostics (above the class)
	void               RunGarbageCollector();
	// KYTY_VRAM_STATS report lines (vramStats.h): guest buffers by age, the largest buffers of
	// every kind, the guest address span they cover, the GC thresholds (GPU thread).
	void               ReportVram();
	// KYTY_BDA_WRITES (Profiling/analysis/BDA-WRITES-DESIGN.md v2). Before recording a dispatch whose
	// program writes through BDA (ShaderInfo::bda_writes): clears the written-page bitmap on first
	// use, forgets every known fill, and moves GPU-modified images that overlap cache buffers into
	// their buffers when image writebacks are on (counted either way). GPU thread.
	void PrepareBdaWrites();
	// Right after recording that dispatch: waits for it (a synchronous settle, before the command
	// processor continues) and takes every page it wrote into GPU ownership exactly as a writable
	// binding over the page would, so no later reader sees the page as clean. GPU thread.
	void SettleBdaWrites(uint64_t shader_hash);
	// KYTY_BDA_CPU_MERGE=1 (default 0): before a BDA-writing dispatch is prepared (before any
	// upload), the guest bytes of the pages this shader's earlier settles found CPU-dirty
	// (the guest wrote them while the dispatch ran). GPU thread.
	void SnapshotBdaMergePages(uint64_t shader_hash);
	// KYTY_CP_FAULT_WRITERS=1 (diagnostics): a read fault the command processor took on GPU-written
	// bytes, with its handler time: aggregated by the newest recorded GPU writer of the address
	// (shader, ticks between that write and the fault) and the op being prepared, every 10 s.
	void NoteCpFault(uint64_t vaddr, uint64_t handler_ns);
	// Diagnostics (KYTY_INDIRECT_ARGS_DIAG / KYTY_READINESS_DIAG, needs KYTY_CP_FAULT_WRITERS=1): the
	// newest recorded GPU buffer write over `vaddr` ("<kind> 0x<hash> <ticks> ticks ago"), or "none".
	[[nodiscard]] std::string DescribeNewestWriter(uint64_t vaddr);
	// KYTY_BDA_ASYNC_SETTLE (bufferCache.cpp): finishes the settles whose dispatch completed (with
	// wait_oldest, waits for the oldest first). GPU thread.
	void ProcessPendingBdaSettles(bool wait_oldest);

private:
	void SettleBdaWritesAsync(uint64_t shader_hash);
	struct PendingBdaSettle {
		uint64_t tick = 0;
		uint64_t hash = 0;
		uint32_t slot = 0;
		RangeSet predicted;
	};
	static constexpr uint32_t BdaWarmSettles    = 16;
	static constexpr uint32_t BdaRelearnSettles = 4;
	struct BdaPrediction {
		RangeSet newer;
		RangeSet older;
		uint32_t sync_left = BdaWarmSettles;
	};
	struct BdaAsyncTotals {
		uint64_t settles = 0, sync_learn = 0, predicted_pages = 0, late_pages = 0, stores = 0, dropped = 0,
		         overflows = 0;
	};
	std::deque<PendingBdaSettle>                 m_bda_pending;
	std::unordered_map<uint64_t, BdaPrediction>  m_bda_predict;
	// KYTY_BDA_CPU_MERGE: per shader, the pages its settles found CPU-dirty; the snapshots of them
	// taken before the current dispatch (page -> 4 KiB); a host-visible page for the GPU bytes.
	std::unordered_map<uint64_t, std::vector<uint64_t>> m_bda_merge_pages;
	std::unordered_map<uint64_t, std::vector<uint8_t>>  m_bda_merge_snapshots;
	std::unique_ptr<Buffer>                             m_bda_merge_download;
	// KYTY_CP_FAULT_WRITERS
	struct CpFaultWriter {
		uint64_t    vaddr = 0, size = 0, tick = 0, hash = 0;
		const char* kind  = "";
	};
	struct CpFaultStats {
		uint64_t    faults = 0, ns = 0, sample = 0, tick_delta = 0, reader = 0;
		const char* kind   = "";
		const char* reader_kind = "";
	};
	std::vector<CpFaultWriter>                    m_cp_fault_writers;
	size_t                                        m_cp_fault_writer_next = 0;
	std::unordered_map<uint64_t, CpFaultStats>    m_cp_fault_stats;
	std::chrono::steady_clock::time_point         m_cp_fault_print {};
	BdaAsyncTotals                               m_bda_async_totals;
	std::chrono::steady_clock::time_point        m_bda_async_print {};

	friend struct BufferCacheTestAccess;
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainImageStagingBuffer(uint64_t size);

	// Device memory is not zero-initialized, and a zero entry is what shaders read as unmapped.
	void EnsureBdaPageTableInitialized();

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
		// KYTY_BDA_REACH: a footprint-limited synchronization (upload accounting only).
		bool     reach                  = false;
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
	// `registered` (KYTY_BDA_REGISTER_LOG, may be null): the ranges of buffers registered since the
	// last pass, synchronized as well.
	[[nodiscard]] bool SynchronizeBdaDirtied(const RangeSet& mapped_ranges,
	                                         const RangeSet* registered = nullptr);
	// The BDA synchronization pass itself (SynchronizeBdaBuffers decides whether it runs).
	void SynchronizeBdaBuffersNow(const RangeSet& mapped_ranges);
	// KYTY_BUFFER_LOOKUP_STATS: the kind of the last SynchronizeBdaBuffersNow (BufferLookupStats::Part).
	uint8_t m_bda_pass_kind = 0;
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
	// the DMA worker with the copy, or are performed here when the copy is not staged. The
	// calling thread's deferred protections (KYTY_DEFER_PROTECTS) go with them, or are applied
	// here first. `allow_small`: stage it even below the DMA minimum (KYTY_DEFER_PROTECTS=all).
	[[nodiscard]] vk::Buffer StageUploadDma(vk::Buffer source, std::span<vk::BufferCopy> copies,
	                                        std::vector<UploadHostCopy>* host_copies = nullptr,
	                                        bool allow_small = false);
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
	// KYTY_BINDING_MEMO_TOUCH (default off, live): a cache-buffer hit skips TouchBuffer when this
	// memo already touched its buffer in the current GC tick (`touched`): the buffer's LRU item then
	// holds that tick, so the touch would change nothing (LeastRecentlyUsedCache::Touch returns at
	// once). A hit takes the buffer from `buffer` either way: the slot vector's object (slots live in
	// a deque, and the structure guard rules out the buffer's deletion).
	enum class BindingMemoKind : uint8_t { Empty, Stream, Cached };
	// One 64-byte line per slot (the table is cache-line aligned). Ranges of 4 GiB or more are
	// never memoized (no storage binding is that large).
	struct alignas(64) BindingMemo {
		uint64_t        vaddr     = 0;
		uint32_t        size      = 0;
		BindingMemoKind kind      = BindingMemoKind::Empty;
		// KYTY_BINDING_CROSS_SETTLED: a cross-epoch check (relaxed and locked CPU-dirty queries)
		// passed under `signature`.
		bool            settled   = false;
		uint64_t        epoch     = 0;
		uint64_t        signature = 0;
		uint64_t        guard     = 0; // stream: tick; cached: buffer structure epoch
		uint64_t        offset    = 0;
		Buffer*         buffer    = nullptr; // the cache buffer (or the stream buffer)
		uint64_t        touched   = UINT64_MAX; // cached: the GC tick of the memo's last touch
	};
	static_assert(sizeof(BindingMemo) == 64);
	static constexpr uint64_t BindingMemoMaxSize = uint64_t {1} << 32u;
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
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainReadBindingMemo(uint64_t vaddr, uint64_t size,
	                                                                 BufferId id);
	// KYTY_CP_COMMIT_STATS=steps: the last ObtainReadBindingMemo call was a memo hit.
	bool m_last_read_hit = false;
	// The binding without the memo; *obtained receives the cache buffer's id (unchanged for a
	// stream copy).
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferNow(uint64_t vaddr, uint64_t size,
	                                                           bool is_written, bool is_texel_buffer,
	                                                           BufferId id, BufferId* obtained);
	// KYTY_BINDING_LEAN_MISS (default off, live; 1, verify, exit). A read binding that missed its
	// memo, whose range has no CPU-dirty page (the tracker's lock-free mirror, exact on this thread:
	// only it clears those bits) while its signature is still the one read before the memo lookup,
	// and whose buffer from FindBuffers (`id`) is still registered and covers it: the normal path
	// would decide no stream copy (that needs a CPU-dirty page), keep `id` (FindBuffer would return
	// it), find nothing to upload, and record the memo under that signature. This takes the same
	// result and records the same memo from that one query, instead of ObtainBufferNow's and
	// RecordBinding's several queries and signatures. Anything else takes the normal path.
	// verify: the normal path runs as well and decides the binding; a different buffer or offset
	// while the signature is unchanged is a mismatch (BindingLeanMissVerifyMismatches; exit stops).
	[[nodiscard]] bool TryLeanReadMiss(uint64_t vaddr, uint64_t size, uint64_t epoch,
	                                   uint64_t before, BufferId id,
	                                   std::pair<Buffer*, uint64_t>& result);
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
	// `nothing` (optional): set when the call provably synchronized nothing because no image
	// starts at or overlaps the range (KYTY_TEXEL_SYNC_MEMO).
	[[nodiscard]] bool SynchronizeBufferFromImageCounted(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                                     bool* nothing);
	[[nodiscard]] bool SynchronizeBufferFromImageImpl(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                                  bool* nothing);
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
	// KYTY_BDA_CPU_MERGE: a settled page the guest also wrote during the dispatch keeps both
	// writers' bytes, as on the hardware: the dwords the dispatch changed (GPU bytes unlike the
	// snapshot) are written into guest memory before the page's upload. False: no snapshot (the
	// guest's page wins, as before).
	bool MergeBdaCpuDirtyPage(Buffer& buffer, uint64_t page, uint64_t shader_hash);
	// SettleBdaWrites for the written pages [vaddr, vaddr + size): the parts inside cache buffers.
	void SettleBdaWrittenRange(uint64_t vaddr, uint64_t size, uint64_t shader_hash,
	                           uint64_t& settled_pages);
	// SettleBdaWrites for the store log (KYTY_BDA_STORE_LOG): the dispatch's stores to pages without
	// a cache buffer, in log order. Their bytes live in guest memory only: each dword is merged into
	// the backing after the transition a guest write makes (CPU-dirty, fills forgotten, images over
	// it invalidated). A dword something on the GPU owns after all (GPU-dirty bytes, a pending
	// publication, a GPU-modified image) cannot be ordered against that owner here and is dropped as
	// before. Returns the dropped stores.
	uint32_t ApplyBdaStoreLog(std::span<const FaultManager::BdaStore> stores, uint64_t shader_hash);
	// Queues backing publication; callers wait before clearing dirty pages or reusing their data.
	// `early`: the pages are released before the publication lands (KYTY_FALSE_SHARING_WRITES).
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                        const std::shared_ptr<EarlyReleasedDownload>& early = {});
	struct ReadMemoryTrace {
		uint64_t begin      = 0;
		uint64_t size       = 0;
		bool     downloaded = false;
	};
	// KYTY_ASYNC_DRAIN (adapted from usa-ratt/KytyPS5-GT7's KYTY_ASYNC_READBACK): the download a
	// guest thread's drain records is submitted and the GPU thread goes on; the guest thread waits
	// for its publication, then FinishAsyncDrain releases the window on the GPU thread.
	struct AsyncDrain {
		uint64_t tick         = 0; // 0: nothing to wait for (done, or drained synchronously)
		uint64_t window_begin = 0;
		uint64_t window_end   = 0;
	};
	[[nodiscard]] static bool AsyncDrainEnabled();
	// The drain path: GPU thread only. With `async` (guest-thread faults only) a recorded download
	// is submitted without waiting and its tick returned in *async.
	void ReadMemoryDrain(uint64_t vaddr, uint64_t size, bool is_write, ReadMemoryTrace& trace,
	                     AsyncDrain* async = nullptr);
	// GPU thread, after the guest thread waited for async.tick's publication: the window's pages
	// lose their GPU-written protection when nothing was written to them since (else the drain is
	// redone synchronously), and a write's bytes become CPU-dirty, as the synchronous drain does.
	void FinishAsyncDrain(uint64_t vaddr, uint64_t size, bool is_write, const AsyncDrain& async,
	                      ReadMemoryTrace& trace);
	// KYTY_CLEAN_PAGE=1 (adapted from usa-ratt/KytyPS5-GT7 ba83f1a64, its round 135), GPU thread:
	// the faulting pages [page_begin, page_end) are no longer GPU-written for the tracker and hold
	// no GPU-written bytes, no bytes of a readback or publication still to land, and no GPU-written
	// image: another thread's read made them current between the fault and now. The access then
	// needs nothing (a write only its CPU-dirty transition), instead of a download of the window's
	// other GPU-written ranges that it waits for. False when the switch is off.
	// With KYTY_FAULT_PAGE_LIFT=1 also true when TryLiftFaultPages lifted the pages.
	[[nodiscard]] bool FaultPagesCurrent(uint64_t page_begin, uint64_t page_end);
	// KYTY_FAULT_PAGE_LIFT=1 (live, default off; adapted from usa-ratt/KytyPS5-GT7 6f3465806), GPU
	// thread: the faulting pages are still protected as GPU-written but hold no GPU-written bytes,
	// no bytes of a readback or publication still to land, no GPU-written image, and no address-
	// writing shader is still running. Their protection is lifted (UnmarkRegionAsGpuModified)
	// without a download, and true is returned. Without it the fault copies the window's other
	// GPU-written ranges and waits for their writer.
	[[nodiscard]] bool TryLiftFaultPages(uint64_t page_begin, uint64_t page_end);
	// KYTY_CLEAN_READ=1 (default off; adapted from usa-ratt/KytyPS5-GT7 19b6c5cfb), GPU thread: a
	// guest load of [vaddr, vaddr + size) faulted on a page protected because the GPU wrote other
	// bytes of it. When none of these bytes is GPU-written, on its way back from the GPU (pending
	// publication, early release), in a GPU-written image, or within reach of a running address-
	// writing shader, copies them from guest memory into `data` and returns true: the fault
	// handler completes the load with them and the page stays protected.
	[[nodiscard]] bool TryReadCleanBytes(uint64_t vaddr, uint64_t size, void* data);
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

public:
	// Colour metadata watches (KYTY_DCC_GPU_WIDE, TextureCache::TryMaterializeGpuMetadataClear; after
	// usa-ratt/KytyPS5-GT7's WatchMetadata): WatchMetadata starts watching [vaddr, vaddr + size) as
	// written by nothing since; any GPU write noted for the buffer cache (NoteBufferContentWrite),
	// an upload of CPU-dirty bytes, a CPU-side invalidation or an unbounded GPU write over it marks it
	// written. MetadataUnwritten: the exact range is watched, unwritten and not CPU-modified now.
	// Byte-exact, unlike a buffer's content revision (any write anywhere in the buffer).
	void               WatchMetadata(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool MetadataUnwritten(uint64_t vaddr, uint64_t size);
	// KYTY_CP_DECISION_MEMORY (graphicsRun.cpp): the same byte-exact watch, in its own set, for the
	// guest bytes a command-processor write put on the GPU timeline (TryWriteDataGpu) with a value the
	// command processor knows. CpValueUnwritten: the exact range is watched, unwritten since and not
	// CPU-modified now, so the known value is still the bytes' value.
	void               WatchCpValue(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool CpValueUnwritten(uint64_t vaddr, uint64_t size);
	// KYTY_CP_WRITE_THROUGH=1 (live, default off; based on Erlou93/KytyPS5-Yotei's
	// BufferCache::TryCpWriteThrough), GPU thread: a command-processor write (record-time label,
	// WRITE_DATA) of [vaddr, vaddr + size) into a page the GPU modified, none of whose written bytes
	// the GPU owns (not GPU-dirty, no pending publication or side readback, no GPU-modified image):
	// written to the backing (no fault, no drain) and to the cached GPU copy at this point of the
	// command stream, so both copies hold it and the page stays GPU-owned. False: the caller writes
	// it as before.
	[[nodiscard]] static bool CpWriteThroughEnabled();
	[[nodiscard]] bool        TryCpWriteThrough(uint64_t vaddr, const void* data, uint64_t size);

private:
	void NoteMetadataWrite(uint64_t vaddr, uint64_t size);
	struct MetadataWatch {
		uint64_t end     = 0;
		bool     written = false;
	};
	std::mutex                        m_metadata_watch_mutex;
	std::map<uint64_t, MetadataWatch> m_metadata_watches; // by first byte
	uint64_t                          m_metadata_watch_span = 0;
	std::atomic<size_t>               m_metadata_watch_count {0};
	std::map<uint64_t, MetadataWatch> m_cp_value_watches; // m_metadata_watch_mutex
	uint64_t                          m_cp_value_watch_span = 0;
	std::atomic<size_t>               m_cp_value_watch_count {0};
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
	// KYTY_PUBLICATION_SNAPSHOT: a seqlock copy of every pending range, rewritten under the mutex
	// at each registration and completion, so HasPendingBackingPublication readers (the draw-prep
	// workers' read gate, the command processor's clean predicates) never take the mutex.
	// m_publication_snapshot_count is PublicationSnapshotOverflow when the ranges do not fit.
	static constexpr uint32_t PublicationSnapshotRanges   = 64;
	static constexpr uint32_t PublicationSnapshotOverflow = UINT32_MAX;
	void RewritePublicationSnapshot(); // requires m_backing_publication_mutex
	[[nodiscard]] bool SnapshotPendingBackingPublication(uint64_t vaddr, uint64_t size,
	                                                     bool& answer) const;
	std::atomic<uint64_t>                                        m_publication_seq {0};
	std::atomic<uint32_t>                                        m_publication_snapshot_count {0};
	std::array<std::atomic<uint64_t>, PublicationSnapshotRanges> m_publication_begin {};
	std::array<std::atomic<uint64_t>, PublicationSnapshotRanges> m_publication_end {};

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
	bool                                              m_bda_pagetable_initialized = false;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	// KYTY_BDA_HEAPS: allocations made resident by NoteBdaFault, re-ensured before BDA writers.
	std::mutex                                        m_bda_heaps_mutex;
	std::vector<std::pair<uint64_t, uint64_t>>        m_bda_heaps;
	bool                                              m_bda_heaps_loaded = false;
	PageTable                                         m_page_table;
	// KYTY_BDA_REACH=verify state (bufferCacheReach.cpp), created on first use.
	struct BdaReachVerify;
	// shared_ptr: the type is complete only in bufferCacheReach.cpp.
	std::shared_ptr<BdaReachVerify>                   m_bda_reach;
	// KYTY_BDA_ACCESS_TAGS state (bufferCacheTags.cpp), created when the codegen option is on.
	struct BdaTags;
	std::shared_ptr<BdaTags>                          m_bda_tags;
	void CreateBdaTags();
	// ChangeRegister: tags the accessed pages among `entries` (page-table entries from first_page)
	// and queues a registered range for the next tagged pass.
	void ApplyBdaTags(uint64_t first_page, std::span<vk::DeviceAddress> entries, bool insert);
	// The entry of `page` with its accessed tag (0 stays 0).
	[[nodiscard]] vk::DeviceAddress TaggedBdaEntry(uint64_t page, vk::DeviceAddress entry) const;
	// Whether a DMA shader reported the caching page (false without tags).
	[[nodiscard]] bool BdaTagAccessed(uint64_t page) const;
	// ChangeRegister: page-table entries of [first_page, last_page) were rewritten.
	void NoteBdaReachRegister(uint64_t first_page, uint64_t last_page, bool insert);
	// KYTY_BDA_LIVE_PAGES state (bufferCacheLive.cpp), created when the variable is set at start.
	struct BdaLive;
	std::shared_ptr<BdaLive>                          m_bda_live;
	// Pages redirected now (lock-free fast path of the hooks).
	std::atomic<uint64_t>                             m_bda_live_count {0};
	// Guest write faults (any thread), for the report line.
	std::atomic<uint64_t>                             m_bda_live_faults {0};
	// Set by SynchronizeBdaBuffers around the full pass: SynchronizeBuffersInRange redirects then.
	bool                                              m_bda_live_pass = false;
	void CreateBdaLive();
	// SynchronizeBuffersInRange during a full pass with the switch on: qualifying caching pages of
	// [start, start + size) of `buffer` are redirected, the rest synchronized as before.
	void LiveSyncBufferRange(BufferId id, uint64_t start, uint64_t size, BdaSyncStats* stats);
	// The redirect entry of `page` (0 when not redirected); with its tag when tags exist and the page
	// was reported.
	[[nodiscard]] vk::DeviceAddress LiveEntry(uint64_t page) const;
	enum class LiveRestore : uint8_t { Upload, Written, Register, Unmap, Read, Cp, Off, Writes };
	// Gives the redirected pages of [vaddr, vaddr + size) their cache entries back (0 without a buffer)
	// and queues them in the forced set. Register: the caller rewrote the entries already. GPU thread.
	// With `writes` the entries are appended there for one later flush (WriteLiveEntries).
	void RestoreLivePages(uint64_t vaddr, uint64_t size, LiveRestore why,
	                      std::vector<std::pair<uint64_t, vk::DeviceAddress>>* writes = nullptr);
	// KYTY_BDA_LIVE_BATCH (I-B1): a read upload over redirected pages of [vaddr, vaddr + size) marks
	// them settled (restored at the next pre-pass) instead of restoring them; 0 = restore at once.
	void SettleLivePages(uint64_t vaddr, uint64_t size);
	// Writes page-table entries (page, entry): one copy when batching, else one copy per run.
	void WriteLiveEntries(std::vector<std::pair<uint64_t, vk::DeviceAddress>>& writes, bool batch);
	// The redirect entries a full pass queued (I-B2), written before the operation.
	void FlushLiveRedirects();
	// NoteBdaTagFault: a shader reported `page`; a redirected one is restored at the next pre-pass (I5).
	void LiveNoteTagRead(uint64_t page);
	// After a full pass ran: V1/V2 (verify), and the tick later CP writes wait for.
	void LivePagesAfterPass(const RangeSet& mapped_ranges, uint64_t fault_epoch_before);
	// KYTY_BDA_WRITES: before a BDA-writing program, every redirect is restored and uploaded (I-W1).
	void LivePagesBeforeBdaWrites();
	// Faults counter (InvalidateMemory on a write fault) and the 10 s report (AdvanceFrame).
	void ReportBdaLive();
	RangeSet                                          m_gpu_modified_ranges;
	uint64_t                                          m_unregister_epoch = 0;
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
	// KYTY_INDIRECT_FORMAT_COUNT (CodegenOptions.h): per-format call counters of the full decoder.
	std::unique_ptr<Buffer>                           m_format_counts;
	uint32_t                                          m_format_count_frames = 0;
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
	// KYTY_BINDING_CROSS_SETTLED (ObtainReadBindingMemo) outcomes.
	struct CrossEpochTotals {
		uint64_t fast = 0, verify_checks = 0, verify_mismatches = 0;
	};
	CrossEpochTotals                                  m_cross_epoch_totals;
	struct BindingMemoTotals {
		uint64_t hot_hits          = 0; // successful hits through the optional front tier
		uint64_t stream_hits       = 0;
		uint64_t cached_hits       = 0; // cross-epoch hits included
		uint64_t cross_hits        = 0;
		uint64_t cross_rejects     = 0;
		uint64_t touch_skips       = 0; // KYTY_BINDING_MEMO_TOUCH
		uint64_t lean_misses       = 0; // KYTY_BINDING_LEAN_MISS
		uint64_t lean_verify_checks     = 0;
		uint64_t lean_verify_mismatches = 0;
		uint64_t lean_verify_races      = 0;
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
	// KYTY_BDA_REGISTER_LOG=1 (default off; needs KYTY_BDA_DIRTY_LOG; adapted from the GT7 fork's
	// b970ba6fe, whose BDA walk visits only the regions where pages turned CPU-dirty or a buffer
	// was registered since the last walk). A new registered buffer moves the structure epoch, and
	// the pass after it used to scan every mapped buffer. With the log, a pass whose structure
	// changed only by new buffers (the layout epoch, moved by unregistrations and GPU mapping
	// changes, held) synchronizes the logged CPU-dirty ranges plus the new buffers' ranges.
	// m_bda_registered and the structure epoch move together under m_bda_registered_mutex, so a
	// pass takes the log with the epoch it accounts for.
	bool                                              m_bda_register_log = false;
	std::mutex                                        m_bda_registered_mutex;
	RangeSet                                          m_bda_registered;
	bool                                              m_bda_registered_overflow = false;
	std::atomic_uint64_t                              m_bda_layout_epoch {1};
	uint64_t                                          m_bda_scanned_layout_epoch = 0;
	uint64_t                                          m_bda_register_passes      = 0;
	struct BdaLogTotals {
		uint64_t passes                = 0;
		uint64_t ranges                = 0;
		uint64_t overflows             = 0;
		uint64_t verify_checks         = 0;
		uint64_t verify_mismatch_pages = 0;
	};
	BdaLogTotals                                      m_bda_log_totals;
	// Upload bytes of the last dirty-log pass (FaultCost::NoteBdaPass, the live cost log).
	uint64_t                                          m_bda_last_pass_bytes = 0;
	// KYTY_BDA_SYNC_EPOCH (SynchronizeBdaBuffers): the sync, BDA structure and fault epochs taken
	// before the last completed pass (GPU thread; 0: none yet), and the outcomes (tests read them).
	bool     m_bda_epoch_skip        = false;
	int      m_bda_epoch_verify      = 0;
	uint64_t m_bda_synced_epoch      = 0;
	uint64_t m_bda_synced_structure  = 0;
	uint64_t m_bda_synced_fault      = 0;
	// KYTY_BDA_SYNC_PER_SUBMISSION (default off, live switch in bufferCache.cpp): the guest
	// submission (SyncEpoch::CurrentSubmission) taken before the last completed pass; later passes
	// of that submission are skipped while the structure epoch holds. Every pass records it, so the
	// switch can turn on at any flip.
	uint64_t m_bda_synced_submission = 0;
	struct BdaEpochTotals {
		uint64_t passes                = 0;
		uint64_t skips                 = 0;
		uint64_t submission_skips      = 0;
		uint64_t verify_checks         = 0;
		uint64_t verify_mismatch_pages = 0;
		// KYTY_BDA_SYNC_PER_SUBMISSION=verify: passes the gate would have skipped, and their uploads.
		uint64_t submission_verify_passes  = 0;
		uint64_t submission_deferred_bytes = 0;
	};
	BdaEpochTotals m_bda_epoch_totals;
	// Bytes every BDA pass uploaded (only counted where a pass keeps its statistics: always for the
	// dirty-log and hot passes, for full scans with KYTY_BDA_HOT_SYNC, aggregates, or while
	// m_bda_count_uploads is set).
	uint64_t m_bda_pass_upload_bytes = 0;
	bool     m_bda_count_uploads     = false;
	// The 10 s console line of the submission gate (ReportBdaSubmissionGate).
	void                                  ReportBdaSubmissionGate();
	uint32_t                              m_bda_gate_report_calls = 0;
	std::chrono::steady_clock::time_point m_bda_gate_report_time {};
	BdaEpochTotals                        m_bda_gate_reported {};
	StreamBuffer                                      m_staging_buffer;
	// After the staging ring: destroyed first, waiting for its copies that read the ring.
	std::unique_ptr<UploadDma>                        m_upload_dma;
	// KYTY_INPLACE (inplaceHeap.h), null when off.
	std::unique_ptr<InplaceHeap>                      m_inplace;
	// GPU thread: [vaddr, +size) bound to an imported window when every page is a candidate and
	// guest memory holds its current bytes.
	[[nodiscard]] bool TryInplaceBinding(uint64_t vaddr, uint64_t size, std::pair<Buffer*, uint64_t>& result);
	StreamBuffer                                      m_stream_buffer;
	// KYTY_STREAM_COPY_REUSE (streamCopyMemo.h): the small stream copies of the current tick and epoch.
	StreamCopyMemo                                    m_stream_copy_memo;
	// KYTY_STREAM_COPY_DEFER: copies encoded for the recorder that it may not have executed yet
	// (ring position after each), in encoding order from m_deferred_head; [lo, hi) bounds them.
	struct DeferredCopy {
		uint64_t begin    = 0;
		uint64_t end      = 0;
		uint64_t position = 0;
	};
	[[nodiscard]] bool TryDeferStreamCopy(uint8_t* mapped, uint64_t vaddr, uint64_t size);
	void               PruneDeferredCopies(const CommandRecorder& recorder);
	friend void        StreamCopyDefer::NoteGuestWriteStream(uint64_t address, uint64_t size);
	std::vector<DeferredCopy> m_deferred;
	size_t                    m_deferred_head = 0;
	uint64_t                  m_deferred_lo   = UINT64_MAX;
	uint64_t                  m_deferred_hi   = 0;
	CommandRecorder*          m_deferred_recorder = nullptr;
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
