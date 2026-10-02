#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_HANGTRACE_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_HANGTRACE_H_

// Loading/hang flight recorder.
//
// Always-cheap, opt-in (KYTY_HANG_TRACE=1) recorder that runs from process start without a
// profiler connection. A background thread writes plain CSV files once per second, so an
// intermittent failure is captured whether or not anyone armed a trace beforehand. Two runs
// (one that loads, one that does not) can then be diffed directly.
//
// Output directory: KYTY_HANG_TRACE_DIR, or ./_HangTrace/<timestamp>-pid<pid>.
//   summary.csv        per-second totals (flips, APR reads/repeats, LOD packets, textures with
//                      mip-stat counters, per-queue GPU wait/busy, AgcSuspendPoint/Done waits)
//   apr.csv            every APR file read: path, range, destination, repeat count, guest callers
//   imports.csv        per-second call counts of every guest->HLE import (KYTY_HANG_TRACE_IMPORTS)
//   imports-index.csv  import id -> NID / host function / importing module
//   lodstats.csv       sampled GET_LOD_STATS packets with the buffer contents *before* the
//                      emulator overwrites them (shows what the guest left there)
//   texstats.csv       per-second use of T# mip-statistics counters (id, base, min LOD)
//   modules.csv        guest module load addresses, to map caller addresses to module offsets
//   queues.csv         per-second, per active guest GPU queue: submission wait and busy time
//   readbacks.csv      every GPU->CPU readback: cause, range, duration, guest thread and pc
//   images.csv         texture-cache image deletions with reason, and per-second native image
//                      create/destroy totals grouped by format, extent and usage
//   transfers.csv      per-second image uploads, buffer uploads, reinterpretation copies and
//                      alias synchronizations grouped by reason, address and shape (RecordTransfer)
//
// summary.csv ends with GPU timeline columns (KYTY_GPU_TIMING, default on with this trace), summed
// over the guest flips published in that second: gpu_busy_us (union of command-buffer spans),
// gpu_cmdbufs, gpu_latency_avg_us (recording start -> GPU start), gpu_idle_us, gpu_max_gap_us,
// gpu_starved_us (idle before the next buffer reached vkQueueSubmit), gpu_dispatch_latency_avg_us
// (vkQueueSubmit return -> GPU start) and gpu_dropped. Latency/starved columns are 0 without
// calibrated timestamps. After them come GPU recording counters (KYTY_GPU_OP_COUNTERS, default on
// with this trace; see graphics/host_gpu/renderer/gpuOpProfiler.h): gpu_render_passes (guest
// dynamic-rendering begins), gpu_barriers (guest pipeline barrier calls), gpu_layout_transitions
// (image barriers changing layout) and gpu_guest_cmdbufs (guest command buffers begun). Then the
// barrier batcher (KYTY_BARRIER_BATCH, graphics/host_gpu/renderer/render.h): gpu_barrier_requests,
// gpu_barriers_merged (joined an already pending batch), gpu_barriers_elided (covered by the
// previous barrier with nothing recorded since), gpu_barriers_sunk (a pending batch kept across a
// draw in the same rendering instance) and gpu_barrier_rp_splits (barrier flushes that ended an
// active rendering instance). After the transfer columns: gpu_rendering_ends (guest rendering
// instances ended, every cause; per-site Tracy plots GpuOps.EndRendering.<site>) and
// gpu_draw_write_sinks (post-draw shader-write barriers kept pending across a draw continuing the
// same instance, KYTY_DRAW_WRITE_SINK). Then the guest memory tracking / upload counters
// (MemoryCounter below, mem_* columns). Then shader/pipeline compilation (see RecordCompile):
// compile_programs, compile_translate_us, compile_emit_us, compile_validate_us, compile_module_us,
// compile_gfx_pipelines, compile_gfx_pipeline_us, compile_cs_pipelines, compile_cs_pipeline_us,
// compile_stall_us, compile_stall_max_us, compile_gfx_new, compile_gfx_perm, compile_gfx_variant.
// Then driver pipeline cache saves (RecordPipelineCacheSave): pcache_saves, pcache_save_kb,
// pcache_save_serialize_us, pcache_save_write_us, pcache_save_overlaps, pcache_save_overlap_us.
// Then compile_translation_reuses (programs specialized from a kept translation instead of
// translating again) and compile_clone_us (copying translations, RecordCompile clone_ns).
// Then validate_async_count and validate_async_us: spirv-val runs on the background validator
// (KYTY_SHADER_VALIDATION_ASYNC), off the compiling thread (RecordShaderValidation).
// Then graphics pipeline libraries (KYTY_PIPELINE_LIBRARY, RecordPipelineLibraryEvent):
// gpl_cache_hits, gpl_links, gpl_link_us, gpl_libraries, gpl_library_us, gpl_optimized,
// gpl_optimize_us. Memory counters added to MemoryCounter after mem_written_upload_late_pages
// follow at the end of the row (kMemoryColumnsBeforeCompile in hangTrace.cpp).
// Last, apr_grow_reads / apr_shrink_reads: reads from file offset 0 larger / smaller than the same
// file's previous offset-0 read (Astro Bot's texture streamer promoting a texture from its 128 KiB
// head to the full file / demoting it back); apr_shrink_max_per_flip: the most shrink reads
// between two flips (its eviction pass stops at 32 textures); apr_stream_mib: the sum of the
// latest offset-0 read of every file a "TextureStreamer" thread read, an estimate of the
// streamed-texture footprint (its pool holds 4.5 GiB = 4608 MiB and evicts above 85%). Level
// unloads issue no reads, so the estimate keeps a previous level's files: compare it within one
// level visit.
// New columns are only ever appended.
//
//   compiles.csv       one row per new shader program permutation or pipeline: phase times, the
//                      requesting thread, and for graphics pipelines which key fields differ from
//                      the closest existing pipeline of the same programs (RecordCompile)
//   timestamps.csv     one row per guest flip:
//                      - guest clock writes rewritten with GPU times, skipped, kept at record time
//                        and deferred (KYTY_EOP_TIMESTAMPS), with the mean and largest GPU - record
//                        shift and the command processor's publishing time;
//                      - the latest end - begin, in us, of each guest GPU timer ring
//                        (KYTY_HANG_TRACE_TIMESTAMP_RINGS, default Astro Bot's; 0 = none);
//                      - Astro Bot's dynamic-resolution controller state
//                        (KYTY_HANG_TRACE_DRS_PROBE: the eboot.bin offset of its pointer, 0 = off):
//                        resolution index, level, updates with room to upgrade, target fps,
//                        scalable and total GPU ms (RecordEopTimestamps).
//   placement.csv      each second, per thread role (cp, recorder, guest, host): placement samples,
//                      those on the CP's current physical core, the CP's core, and the samples per
//                      logical processor 0-31 (lp_other: 32 and up). See common/cpuPlacement.h.
//
// summary.csv's last column, pending_ops_max, is the deepest normal-operation queue a draw or
// dispatch left queued in that second (KYTY_PENDING_OPS_NOWAIT).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace HangTrace {

namespace Detail {
// KYTY_HANG_TRACE, read on first use by any thread: -1 until then, else 0 or 1 (constant-
// initialized, so correct before this module's dynamic initialization).
extern std::atomic<int8_t> g_enabled;
[[nodiscard]] bool ReadEnabled() noexcept;
// The KYTY_HANG_TRACE_CP_WATCH window: set once by Initialize when the CP trace is on (before
// renderer threads start), empty otherwise.
extern uint64_t g_cp_watch_begin;
extern uint64_t g_cp_watch_end;
// RecordTexture's count of resolved textures, one block per recording thread (written only by
// that thread; summary.csv tex_resolved sums them). Never freed.
struct alignas(64) TextureTally {
	std::atomic<uint64_t> resolved {0};
	TextureTally*         next = nullptr;
};
extern constinit thread_local TextureTally* t_texture_tally;
[[nodiscard]] TextureTally& AcquireTextureTally() noexcept;
// A resolved descriptor with the mip-statistics bit (tex_mipstats, texstats.csv).
void RecordMipStatsTexture(const uint32_t* fields) noexcept;
} // namespace Detail

// Inlined: checked on hot renderer paths.
[[nodiscard]] inline bool Enabled() {
	const auto value = Detail::g_enabled.load(std::memory_order_relaxed);
	return value >= 0 ? value != 0 : Detail::ReadEnabled();
}
[[nodiscard]] bool ImportsEnabled();

void Initialize();
void Shutdown();

// Output directory of this trace, or empty when the trace is disabled or failed to start.
[[nodiscard]] std::string OutputDirectory();

[[nodiscard]] uint64_t NowNs();

// Guest code map, used to validate stack-scanned return addresses.
void RegisterGuestCode(uint64_t base, uint64_t size, std::string_view name);
[[nodiscard]] bool IsGuestAddress(uint64_t address);

// Import call counters. Returns the address of a 64-bit counter the import thunk increments.
// One counter per distinct target, shared by all call sites.
[[nodiscard]] uint64_t* AllocateImportCounter(uint64_t target, std::string_view nid,
                                              std::string_view dbg_name, std::string_view program);
[[nodiscard]] uint64_t FindImportThunk(uint64_t target);
void                   SetImportThunk(uint64_t target, uint64_t thunk);

// APR: the guest return address of the current APR submit call on this thread.
void SetGuestCaller(uint64_t return_address);
void RecordAprRead(uint32_t file_id, std::string_view host_path, uint64_t file_offset,
                   uint64_t size, uint64_t destination, uint64_t bytes_read, int result,
                   std::string_view thread_name);

// GET_LOD_STATS: call before the emulator writes the destination.
void RecordLodStats(const void* destination, uint32_t size, uint32_t control);

// LOD report consumer discovery (KYTY_HANG_TRACE_LOD_WATCH, default on with the trace): after
// the emulator writes a report, its plain read/write pages are briefly made inaccessible; the
// first guest access is logged (pc, registers, surrounding code bytes) and the original
// protection restored. Disarm before the emulator itself reads or writes the report.
void DisarmLodReportWatch();
void ArmLodReportWatch(const void* destination, uint32_t size);
// Called from the host access-violation handler. Returns true when the fault was a watch hit.
[[nodiscard]] bool HandleLodWatchFault(uint64_t fault_vaddr, bool write, uint64_t pc,
                                       const uint64_t* gpr16, std::string_view thread_name);

// Texture descriptor (8 dwords) resolved for a draw/dispatch. Inlined: every resolved texture,
// ~50k per flip at the Sky Garden start view.
inline void RecordTexture(const uint32_t* fields) {
	if (!Enabled()) {
		return;
	}
	auto* tally = Detail::t_texture_tally;
	if (tally == nullptr) [[unlikely]] {
		tally = &Detail::AcquireTextureTally();
	}
	tally->resolved.store(tally->resolved.load(std::memory_order_relaxed) + 1,
	                      std::memory_order_relaxed);
	if (((fields[5] >> 25u) & 1u) != 0) [[unlikely]] {
		Detail::RecordMipStatsTexture(fields);
	}
}

// GPU->CPU readbacks (BufferCache::ReadMemory), attributed to the path that requested them.
// FaultReadSide / FaultReadDuplicate: a guest read fault served by a side copy (no drain of the
// current recording), or by waiting on another thread's pending side copy. FaultReadEager: a
// guest read fault that waited for (or finished) a pending eager copy (KYTY_READBACK_EAGER).
// EagerPublish: an eager copy published at completion; duration_us is issue to publication.
enum class ReadbackKind : uint8_t {
	Invalidate,
	FaultRead,
	FaultWrite,
	GpuSync,
	FaultReadSide,
	FaultReadDuplicate,
	FaultReadEager,
	EagerPublish
};
// Guest access fault context for readbacks on this thread (instruction address, guest thread).
void SetFaultContext(uint64_t pc, std::string_view thread_name);
void ClearFaultContext();
void SetReadbackKind(ReadbackKind kind);
[[nodiscard]] ReadbackKind GetReadbackKind();
void RecordReadback(uint64_t vaddr, uint64_t size, uint64_t window_begin, uint64_t window_size,
                    bool downloaded, uint64_t duration_ns);

// occlusion.csv (KYTY_GPU_OCCLUSION=1): "dump" rows for each ZPASS_DONE dump (scopes counted since
// the previous dump, plus the latest scope's target size, colour count, depth format and depth
// address), "publish" rows with the cumulative DB 0 sample count written to the guest for that
// dump, and "predicate" rows for each SET_PREDICATION op 1 (sample delta, condition, resulting
// skip). A begin/end pair whose published counts are equal although the pair brackets counted
// scopes means the samples were rejected; one bracketing no scopes never reached a counted scope.
struct OcclusionEvent {
	const char* event        = "";
	uint64_t    address      = 0;
	uint64_t    value        = 0;
	uint32_t    scopes       = 0;
	uint32_t    width        = 0;
	uint32_t    height       = 0;
	uint32_t    colors       = 0;
	bool        has_depth    = false;
	uint32_t    depth_format = 0;
	uint64_t    depth_address = 0;
	uint32_t    condition    = 0;
	bool        skip         = false;
	std::string detail; // "draw" rows: state of a draw recorded inside a counted scope
};
void RecordOcclusion(const OcclusionEvent& event);

// lodreports.csv (KYTY_LOD_STATS_MODE=gpu): rows for each GET_LOD_STATS report.
//   Record rows (when the packet is recorded): whether statistics were written now (has_latest;
//   only KYTY_LOD_REPORT_PUBLISH=rewrite/record write at record time), their sampled counters,
//   summed counts and mean finest mip, and GPU copies issued but not completed (report age).
//   Completion rows (pending_copies = UINT64_MAX): has_latest = 1 when the guest slot was written,
//   sampled_counters = counters with a finest mip, total_samples = summed counts (bits 0..23).
//   drawn_counters / counted_counters (completion rows): counters with a finest mip ("Drawn" in
//   the guest's debug view) and counters with a non-zero count ("MipClamp": with
//   KYTY_LOD_STATS_COUNT=clamp, textures sampled finer than their T# MIN_LOD, which the streamer
//   promotes to full resolution).
enum class LodReportKind : uint8_t { Record, Completion };
struct LodReportEvent {
	uint64_t      destination      = 0;
	uint32_t      control          = 0;
	bool          has_latest       = false;
	uint32_t      sampled_counters = 0;
	uint64_t      total_samples    = 0;
	double        mean_finest_mip  = 0.0;
	uint64_t      pending_copies   = 0;
	LodReportKind kind             = LodReportKind::Record;
	uint32_t      drawn_counters   = 0;
	uint32_t      counted_counters = 0;
};
void RecordLodReport(const LodReportEvent& event);

// cp.csv (KYTY_HANG_TRACE_CP=1): command-processor ordering events in record order (row id):
// submissions admitted, slices run, WAIT_REG_MEM results, label writes and GPU writes into the
// KYTY_HANG_TRACE_CP_WATCH window ("0xADDR:0xSIZE"). The GPU thread sets the queue and admission
// sequence it is processing (SetCpContext); events without an explicit queue carry that context.
[[nodiscard]] bool CpTraceEnabled();
void               SetCpContext(uint32_t queue, uint64_t sequence);
// Whether [address, address + size) overlaps the CP watch window (never without the CP trace:
// the window is only set with it). Inlined: checked on every texture-cache image use.
[[nodiscard]] inline bool CpWatch(uint64_t address, uint64_t size) {
	return Detail::g_cp_watch_end > Detail::g_cp_watch_begin && address < Detail::g_cp_watch_end &&
	       Detail::g_cp_watch_begin < address + size;
}
struct CpEvent {
	const char* event   = "";
	uint64_t    address = 0;
	uint64_t    value   = 0;
	uint64_t    ref     = 0;
	uint64_t    mask    = 0;
	int64_t     aux     = 0; // event-specific: compare function, result, count
	uint64_t    size    = 0;
	int64_t     queue   = -1; // -1: the current CP context
	uint64_t    seq     = 0;  // with queue >= 0
};
void RecordCp(const CpEvent& event);

// Which kind of recorded GPU write last marked a guest page GPU-owned (reported per readback).
enum class GpuWriteKind : uint8_t { ShaderStorage, OcclusionDump, Fill, Copy };
class ScopedGpuWriteKind {
public:
	explicit ScopedGpuWriteKind(GpuWriteKind kind);
	~ScopedGpuWriteKind();
	ScopedGpuWriteKind(const ScopedGpuWriteKind&)            = delete;
	ScopedGpuWriteKind& operator=(const ScopedGpuWriteKind&) = delete;

private:
	GpuWriteKind m_previous;
};
void NoteGpuWrite(uint64_t vaddr, uint64_t size);
// Kind name of the last recorded GPU buffer write to the page holding vaddr ("unknown" if none).
[[nodiscard]] const char* LastGpuWriteKind(uint64_t vaddr);
// Tests: what NoteGpuWrite recorded for the page holding vaddr (readbacks.csv reports it).
struct GpuPageWriter {
	bool         found = false;
	uint64_t     t_ms  = 0;
	uint64_t     size  = 0;
	GpuWriteKind kind  = GpuWriteKind::ShaderStorage;
};
[[nodiscard]] GpuPageWriter PageWriterForTest(uint64_t vaddr);

// Texture-cache image deletion reasons (set around FreeImage calls) and native image churn.
enum class ImageFreeReason : uint8_t {
	Other,
	DepthAssociation,
	DepthRecreate,
	OverlapLayout,
	OverlapMipMerge,
	OverlapStale,
	Expand,
	SmallerResources,
	Unmap,
	GarbageCollect,
	PressureCollect,
	ResidentIdle,
	Count
};
void SetImageFreeReason(ImageFreeReason reason);
void RecordImageFree(uint64_t address, uint32_t width, uint32_t height, uint32_t levels,
                     uint32_t layers, uint32_t format, uint64_t bytes, uint64_t age_ticks);
void RecordNativeImage(bool create, bool pool_hit, uint32_t format, uint32_t width, uint32_t height,
                       uint32_t levels, uint32_t usage, uint64_t bytes);

// Texture-cache / buffer-cache transfer attribution (transfers.csv), aggregated per second by
// kind, reason, detail, guest address and image shape. reason/detail must be string literals.
//   ImageUpload  guest memory -> native image refresh (reason: first-use, first-use-gpu-buffer,
//                cpu-write, cpu-edge-hash, gpu-buffer-write; detail: upload binding, or for
//                GPU buffer writes the kind of the last recorded write, see NoteGpuWrite)
//   BufferUpload CPU-dirty pages -> device buffer (reason: caller; detail: empty; address: first
//                uploaded byte; width: number of dirty page runs copied; format/height: 0)
//   ImageCopy    texture-cache reinterpretation copy (reason: path; detail: caller)
//   AliasSync    alias synchronization decision (reason: copy or skip)
// bytes: bytes moved. span_bytes: dirty span inside the image that forced the refresh (image
// uploads) or the requested range (buffer uploads).
enum class TransferKind : uint8_t { ImageUpload, BufferUpload, ImageCopy, AliasSync, Count };
void RecordTransfer(TransferKind kind, const char* reason, const char* detail, uint64_t address,
                    uint32_t format, uint32_t width, uint32_t height, uint64_t bytes,
                    uint64_t span_bytes);

// unclean.csv: draw-preparation reads refused as not provably clean (the DrawPrepFallbackUnclean
// cause), aggregated per second by reason, read purpose, calling host code and 4 KiB guest page,
// with the page's last recorded GPU writer (NoteGpuWrite). reason and purpose are string literals
// (purpose may be null); caller is the host return address of the read.
void RecordUncleanRead(uint64_t address, uint64_t size, const char* reason, const char* purpose,
                       uint64_t caller);

// Guest GPU scheduler.
void RecordQueueWait(uint32_t queue, uint64_t wait_ns);
void RecordQueueBusy(uint32_t queue, uint64_t busy_ns, bool complete);
void RecordDoneWait(uint64_t wait_ns);
void RecordFlip();

// GPU execution timing for one guest flip (see graphics/host_gpu/renderer/gpuTiming.h).
struct GpuFrame {
	uint64_t busy_ns             = 0;
	uint64_t idle_ns             = 0;
	uint64_t max_gap_ns          = 0;
	uint64_t starved_ns          = 0;
	uint64_t command_buffers     = 0;
	uint64_t latency_samples     = 0;
	uint64_t record_latency_ns   = 0; // summed over latency_samples
	uint64_t dispatch_latency_ns = 0; // summed over latency_samples
	uint64_t dropped             = 0;
};
void RecordGpuFrame(const GpuFrame& frame);

// Guest GPU recording counters for one guest flip (graphics/host_gpu/renderer/gpuOpProfiler.h).
struct GpuOpCounts {
	uint64_t render_passes      = 0;
	uint64_t barriers           = 0;
	uint64_t layout_transitions = 0;
	uint64_t command_buffers    = 0;
	// Barrier batcher (KYTY_BARRIER_BATCH); zero when it is disabled.
	uint64_t barrier_requests      = 0;
	uint64_t barriers_merged       = 0;
	uint64_t barriers_elided       = 0;
	uint64_t barriers_sunk         = 0;
	uint64_t barrier_render_splits = 0;
	// Appended: rendering instances ended (all causes) and post-draw write barriers sunk.
	uint64_t rendering_ends   = 0;
	uint64_t draw_write_sinks = 0;
};
void RecordGpuOpCounts(const GpuOpCounts& counts);

// Guest GPU timestamps (graphics/host_gpu/renderer/eopTimestamps.h), one timestamps.csv row per
// guest flip, written in every KYTY_EOP_TIMESTAMPS mode (the rewrite counts are zero in record).
struct EopTimestampFrame {
	static constexpr uint32_t MaxRings = 8;
	uint64_t rewritten   = 0; // slots rewritten with GPU times
	uint64_t skipped     = 0; // not rewritten: the guest had written the slot again
	uint64_t unavailable = 0; // no query slot, result or calibration: record time kept
	uint64_t deferred    = 0; // GPU times delivered by deferred label writes
	int64_t  shift_sum   = 0; // sum of GPU - record over rewritten and deferred, 10 ns ticks
	uint64_t shift_max   = 0; // largest |GPU - record|, 10 ns ticks
	uint64_t publish_ns  = 0; // command-processor time spent publishing
	uint64_t publishes   = 0;
	// Latest valid end - begin of each guest timer ring (KYTY_HANG_TRACE_TIMESTAMP_RINGS), in
	// microseconds; -1 when no slot pair is valid.
	uint32_t                      rings = 0;
	std::array<int64_t, MaxRings> ring_delta_us {};
	// Astro Bot's dynamic-resolution controller (KYTY_HANG_TRACE_DRS_PROBE), when readable.
	bool    drs_valid       = false;
	int32_t drs_index       = 0; // 0 1920x1080, 1-2 2432x1368, 3 3328x1872, 4 3840x2160
	int32_t drs_level       = 0;
	int32_t drs_room_frames = 0; // consecutive updates with room for the next level (31 upgrade)
	int32_t drs_fps         = 0; // budget 1000 / fps ms
	double  drs_scalable_ms = 0.0;
	double  drs_total_ms    = 0.0;
};
void RecordEopTimestamps(const EopTimestampFrame& frame);
// KYTY_PENDING_OPS_NOWAIT: the normal-operation queue depth when a draw/dispatch pop left it queued
// (summary.csv pending_ops_max: the largest per second).
void NotePendingOperations(uint64_t depth);
// The load address of a registered guest module (RegisterGuestCode), by file name.
[[nodiscard]] bool ModuleBase(std::string_view name, uint64_t& base);
// Copies `size` bytes of guest memory whose host pages are all readable now (never faults on
// guard, no-access or unmapped pages); false and nothing copied otherwise.
[[nodiscard]] bool TryReadReadable(uint64_t address, void* data, size_t size);

// Shader and pipeline compilation (graphics/host_gpu/renderer/pipeline/pipelineCache.cpp). One
// event per new program permutation (translate = ShaderRecompiler::TranslateProgram, emit =
// CompileProgram: specialization and SPIR-V emission, validate = spirv-val, module =
// vkCreateShaderModule) or new pipeline (pipeline = pipeline and layout creation, including
// vkCreateGraphicsPipelines/vkCreateComputePipelines). total_ns is the wall time of the whole
// compile on the thread that needed it. summary.csv adds each column per second.
enum class CompileKind : uint8_t { Program, GraphicsPipeline, ComputePipeline, Count };
// Graphics pipelines only: whether another pipeline already existed for the same program ids
// (Variant, detail names the differing key fields), only for the same guest shaders with other
// program permutations (Permutation), or for neither (New).
enum class PipelineOrigin : uint8_t { None, New, Permutation, Variant };
struct CompileEvent {
	CompileKind      kind         = CompileKind::Program;
	PipelineOrigin   origin       = PipelineOrigin::None;
	const char*      stage        = "";
	uint64_t         guest_hash   = 0;
	uint64_t         id           = 0; // program id (for pipelines: the first vertex program)
	uint64_t         id2          = 0; // pipelines: pixel program id
	uint64_t         translate_ns = 0;
	uint64_t         emit_ns      = 0;
	uint64_t         validate_ns  = 0;
	uint64_t         module_ns    = 0;
	uint64_t         pipeline_ns  = 0;
	uint64_t         total_ns     = 0;
	uint64_t         spirv_words  = 0;
	std::string_view detail;
	// Programs: copy of a kept translation (reused, translate_ns 0) or of a new one being kept.
	uint64_t         clone_ns = 0;
	bool             reused   = false;
	// Programs, persistent program cache (KYTY_PROGRAM_CACHE): key, lookups and decoding
	// (compiles.csv load_us), and whether the permutation was reloaded instead of emitted
	// (translate_ns and emit_ns 0; detail "+disk").
	uint64_t         load_ns   = 0;
	bool             from_disk = false;
};
void RecordCompile(const CompileEvent& event);
// A draw or dispatch spent stall_ns in the compile paths (new programs and pipelines, including
// lock waits). compile_stall_max_us is the longest single stall of the second.
void RecordCompileStall(uint64_t stall_ns);
// A driver pipeline cache save (periodic or at exit): payload bytes, vkGetPipelineCacheData time
// and file write time. Overlaps are pipelines created while a save was serializing the same
// cache (the driver may serialize them internally); create_ns is that creation's duration.
void RecordPipelineCacheSave(uint64_t bytes, uint64_t serialize_ns, uint64_t write_ns);
void RecordPipelineCacheSaveOverlap(uint64_t create_ns);
// One background spirv-val run (not on a compiling draw's thread).
void RecordShaderValidation(uint64_t validate_ns);
// Graphics pipeline libraries: a monolithic pipeline found in the driver cache (CacheHit), a
// fast link (Linked, link time), `count` new libraries (Library, their creation time), and a
// background optimized compile (Optimized, on a worker thread).
enum class PipelineLibraryEvent : uint8_t { CacheHit, Linked, Library, Optimized, Count };
void RecordPipelineLibraryEvent(PipelineLibraryEvent event, uint64_t ns, uint64_t count = 1);

// Guest memory tracking and buffer upload counters (graphics/host_gpu/memoryStats.h), summed per
// second into summary.csv columns appended after gpu_draw_write_sinks, in enum order (column
// names in hangTrace.cpp; *_us columns are nanosecond counters divided by 1000). Append only;
// counters added after WrittenUploadLatePages are emitted at the end of the row, after the
// compile columns, so that no existing column index moves.
//   mem_write_faults / mem_read_faults   guest write/read faults handled by RenderContext
//   mem_fault_us                         time inside RenderContext::HandleFault (both kinds)
//   mem_protect_calls / _pages           host protection calls that remove write access
//   mem_unprotect_calls / _pages         host protection calls that restore read-write access
//   mem_protect_us                       time inside those protection calls (both directions)
//   mem_tracker_lock_contended           region tracking spinlock acquisitions that had to spin
//   mem_scratch_allocs / _bytes / _us    tiler scratch buffers (vmaCreateBuffer) and their cost
//   mem_buffer_from_image                SynchronizeBufferFromImage downloads recorded
//   mem_upload_copies                    vkCmdCopyBuffer calls recording CPU-dirty uploads
//   mem_upload_barriers                  pipeline barriers recorded around those uploads
//   mem_upload_render_splits             uploads that had to end an active rendering instance
//   mem_image_writebacks / _bytes        GPU-modified images moved into a buffer before a GPU
//                                        write took their ownership (PreserveImagesForGpuWrite)
//   mem_image_writeback_partial          of those, downloads that covered only leading mips
//   mem_image_writeback_skips            overlapping GPU-modified images that could not be moved
//                                        (unsupported/unsafe/outside the buffer): contents lost
//   mem_fault_ahead_pages                pages a write fault made CPU-dirty ahead of use
//   mem_hot_promotions / _demotions      pages entering / leaving hot (sticky-dirty) tracking
//   mem_hot_upload_pages                 hot pages visited by uploads (compared with a shadow)
//   mem_hot_upload_skipped               of those, unchanged pages whose copy was skipped
//   mem_written_upload_late_pages        written-upload pages a racing guest write re-dirtied
//                                        while copied outside the tracker lock (copied again)
enum class MemoryCounter : uint8_t {
	WriteFaults,
	ReadFaults,
	FaultNs,
	ProtectCalls,
	ProtectPages,
	UnprotectCalls,
	UnprotectPages,
	ProtectNs,
	TrackerLockContended,
	ScratchAllocs,
	ScratchAllocBytes,
	ScratchAllocNs,
	BufferFromImageSyncs,
	UploadCopies,
	UploadBarriers,
	UploadRenderSplits,
	ImageWritebacks,
	ImageWritebackBytes,
	ImageWritebackPartial,
	ImageWritebackSkips,
	FaultAheadPages,
	HotPromotions,
	HotDemotions,
	HotUploadPages,
	HotUploadSkipped,
	WrittenUploadLatePages,
	Count
};
void CountMemory(MemoryCounter counter, uint64_t amount = 1);

// A memory predication packet (op 3, wait) submitted all recorded GPU work and blocked the command
// processor until it finished (CommandProcessor::ExecPredication); `ns` is the time spent blocked.
// Summed per second into summary.csv pred_flush_waits and pred_flush_wait_us, appended last.
void RecordPredicationFlushWait(uint64_t ns);

} // namespace HangTrace

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_HANGTRACE_H_ */
