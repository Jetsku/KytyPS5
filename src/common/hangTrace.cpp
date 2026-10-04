#include "common/hangTrace.h"

#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <process.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace HangTrace {

namespace Detail {
std::atomic<int8_t> g_enabled {-1};
uint64_t            g_cp_watch_begin = 0;
uint64_t            g_cp_watch_end   = 0;
} // namespace Detail

namespace {

constexpr uint32_t kMaxQueues   = 64;
constexpr uint32_t kMaxModules  = 256;
constexpr uint32_t kMaxCallers  = 10;
constexpr uint64_t kScanBytes   = 48 * 1024;
constexpr uint64_t kAprRowLimit = 4'000'000;

bool EnvFlag(const char* name, bool default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	return !(value[0] == '0' || value[0] == 'n' || value[0] == 'N' || value[0] == 'f' ||
	         value[0] == 'F');
}

const auto g_start = std::chrono::steady_clock::now();

struct GuestModule {
	uint64_t    base = 0;
	uint64_t    size = 0;
	std::string name;
};
std::array<GuestModule, kMaxModules> g_modules {};
std::atomic<uint32_t>                g_module_count {0};
std::mutex                           g_module_mutex;
std::vector<std::string>             g_pending_module_rows;

struct ImportEntry {
	alignas(64) uint64_t count = 0;
	uint64_t    published      = 0;
	uint64_t    target         = 0;
	uint64_t    thunk          = 0;
	uint32_t    id             = 0;
	std::string nid;
	std::string dbg_name;
	std::string program;
};
std::mutex                                     g_import_mutex;
std::deque<std::unique_ptr<ImportEntry>>       g_imports;
std::unordered_map<uint64_t, ImportEntry*>     g_imports_by_target;
std::atomic<uint32_t>                          g_import_count {0};
std::vector<std::string>                       g_pending_import_index_rows;

thread_local uint64_t g_guest_caller = 0;

struct AprKey {
	uint32_t file_id = 0;
	uint64_t offset  = 0;
	uint64_t size    = 0;
	bool     operator==(const AprKey&) const = default;
};
struct AprKeyHash {
	size_t operator()(const AprKey& k) const {
		uint64_t h = k.file_id * 0x9E3779B97F4A7C15ull;
		h ^= k.offset + 0x632BE59BD9B4E019ull + (h << 6u) + (h >> 2u);
		h ^= k.size + 0x94D049BB133111EBull + (h << 6u) + (h >> 2u);
		return static_cast<size_t>(h);
	}
};
struct AprInfo {
	uint64_t count     = 0;
	uint64_t first_ms  = 0;
	uint64_t last_ms   = 0;
	uint64_t last_dest = 0;
};
std::mutex                                        g_apr_mutex;
std::unordered_map<AprKey, AprInfo, AprKeyHash>   g_apr_keys;
std::vector<std::string>                          g_pending_apr_rows;
uint64_t                                          g_apr_rows_total = 0;
// Size of each file's latest read from offset 0 (guarded by g_apr_mutex). A streamer that
// re-reads a texture file's prefix to drop mips and the whole file to add them (Astro Bot: 128 KiB
// head <-> full file) shows its residency changes as size changes here.
std::unordered_map<uint32_t, uint64_t>            g_apr_offset0_size;
// The same for reads by texture-streamer threads (name contains "TextureStreamer", Astro Bot's
// GfxTextureStreamerThread), and their sum: an estimate of the streamed-texture footprint. Other
// threads' offset-0 reads are level data (about 7 GiB over a U48 run).
std::unordered_map<uint32_t, uint64_t>            g_apr_stream_size;
uint64_t                                          g_apr_stream_bytes = 0;

std::mutex               g_lod_mutex;
std::vector<std::string> g_pending_lod_rows;
std::atomic<uint64_t>    g_lod_sequence {0};

struct QueueStats {
	std::atomic<uint64_t> starts {0};
	std::atomic<uint64_t> wait_ns {0};
	std::atomic<uint64_t> wait_max_ns {0};
	std::atomic<uint64_t> busy_ns {0};
	std::atomic<uint64_t> slices {0};
	std::atomic<uint64_t> incomplete {0};
};
std::array<QueueStats, kMaxQueues> g_queues {};

constexpr uint64_t kReadbackRowLimit = 2'000'000;
constexpr uint64_t kImageRowLimit    = 2'000'000;

struct FaultContext {
	uint64_t pc = 0;
	char     thread[32] {};
};
thread_local FaultContext g_fault_context {};
thread_local ReadbackKind g_readback_kind = ReadbackKind::Invalidate;
thread_local ImageFreeReason g_image_free_reason = ImageFreeReason::Other;

thread_local GpuWriteKind g_gpu_write_kind = GpuWriteKind::ShaderStorage;
struct PageWriter {
	uint64_t t_ms = 0;
	uint64_t size = 0;
	GpuWriteKind kind = GpuWriteKind::ShaderStorage;
};
constexpr const char* kGpuWriteKindNames[] = {"shader-storage", "occlusion-dump", "fill", "copy"};
std::mutex                                  g_page_writer_mutex;
std::unordered_map<uint64_t, PageWriter>    g_page_writers;
// Every change of g_page_writers (NoteGpuWrite updates and clears). A thread's recent ranges below
// are what the map holds for their pages only while this is still the value after that thread's
// last update.
std::atomic<uint64_t> g_page_writer_version {0};
constexpr size_t      kClearPageWriters = 1'000'000;

// The last ranges this thread noted (NoteGpuWrite), invalidated by its own overlapping or clearing
// updates.
struct RecentGpuWrite {
	uint64_t     first = 0;
	uint64_t     last  = 0;
	uint64_t     t_ms  = 0;
	uint64_t     size  = 0;
	GpuWriteKind kind  = GpuWriteKind::ShaderStorage;
	bool         valid = false;
};
constinit thread_local std::array<RecentGpuWrite, 8> t_recent_writes {};
constinit thread_local uint64_t                      t_recent_version    = UINT64_MAX;
constinit thread_local uint32_t                      t_recent_next       = 0;
constinit thread_local bool                          t_recent_below_clear = false;

std::mutex               g_readback_mutex;
std::vector<std::string> g_pending_readback_rows;
uint64_t                 g_readback_rows_total = 0;

constexpr const char* kReadbackKindNames[] = {"invalidate",      "fault-read",
                                              "fault-write",     "gpu-sync",
                                              "fault-read-side", "fault-read-dup",
                                              "fault-read-eager", "eager-publish"};
static_assert(std::size(kReadbackKindNames) ==
                  static_cast<size_t>(ReadbackKind::EagerPublish) + 1,
              "one name per ReadbackKind");
constexpr const char* kImageFreeReasonNames[] = {
    "other",           "depth-association", "depth-recreate", "overlap-layout",
    "overlap-mip-merge", "overlap-stale",   "expand",         "smaller-resources",
    "unmap",           "gc",                "pressure-gc",    "resident-idle"};
static_assert(std::size(kImageFreeReasonNames) == static_cast<size_t>(ImageFreeReason::Count));

struct NativeImageKey {
	bool     create  = false;
	bool     hit     = false;
	uint32_t format  = 0;
	uint32_t width   = 0;
	uint32_t height  = 0;
	uint32_t levels  = 0;
	uint32_t usage   = 0;
	bool     operator==(const NativeImageKey&) const = default;
};
struct NativeImageKeyHash {
	size_t operator()(const NativeImageKey& k) const {
		uint64_t h = (static_cast<uint64_t>(k.format) << 32u) ^ k.usage ^
		             (static_cast<uint64_t>(k.create) << 63u) ^ (static_cast<uint64_t>(k.hit) << 62u);
		h ^= (static_cast<uint64_t>(k.width) << 20u) ^ (static_cast<uint64_t>(k.height) << 4u) ^ k.levels;
		return static_cast<size_t>(h * 0x9E3779B97F4A7C15ull);
	}
};
struct NativeImageTotals {
	uint64_t count = 0;
	uint64_t bytes = 0;
};
std::mutex               g_image_mutex;
std::vector<std::string> g_pending_image_rows;
uint64_t                 g_image_rows_total = 0;

constexpr const char* kTransferKindNames[] = {"image-upload", "buffer-upload", "image-copy",
                                              "alias-sync"};
static_assert(std::size(kTransferKindNames) == static_cast<size_t>(TransferKind::Count));
// Distinct keys per published second; further keys fold into address 0 of the same reason.
constexpr size_t kTransferKeyLimit = 8192;
struct TransferKey {
	TransferKind kind    = TransferKind::ImageUpload;
	const char*  reason  = nullptr;
	const char*  detail  = nullptr;
	uint64_t     address = 0;
	uint32_t     format  = 0;
	uint32_t     width   = 0;
	uint32_t     height  = 0;
	bool         operator==(const TransferKey&) const = default;
};
struct TransferKeyHash {
	size_t operator()(const TransferKey& k) const {
		uint64_t h = k.address * 0x9E3779B97F4A7C15ull;
		h ^= reinterpret_cast<uintptr_t>(k.reason) + (h << 6u) + (h >> 2u);
		h ^= reinterpret_cast<uintptr_t>(k.detail) + (h << 6u) + (h >> 2u);
		h ^= (static_cast<uint64_t>(k.format) << 40u) ^ (static_cast<uint64_t>(k.width) << 20u) ^
		     k.height ^ (static_cast<uint64_t>(k.kind) << 60u);
		return static_cast<size_t>(h * 0x94D049BB133111EBull);
	}
};
struct TransferTotals {
	uint64_t count      = 0;
	uint64_t bytes      = 0;
	uint64_t span_bytes = 0;
};
std::mutex                                                          g_transfer_mutex;
std::unordered_map<TransferKey, TransferTotals, TransferKeyHash>    g_transfers;

// unclean.csv (RecordUncleanRead). Distinct keys per published second; further keys fold into
// page 0 of the same reason, purpose and caller.
constexpr size_t kUncleanKeyLimit = 4096;
struct UncleanKey {
	const char* reason  = nullptr;
	const char* purpose = nullptr;
	uint64_t    caller  = 0;
	uint64_t    page    = 0;
	bool        operator==(const UncleanKey&) const = default;
};
struct UncleanKeyHash {
	size_t operator()(const UncleanKey& k) const {
		uint64_t h = k.page * 0x9E3779B97F4A7C15ull;
		h ^= reinterpret_cast<uintptr_t>(k.reason) + (h << 6u) + (h >> 2u);
		h ^= reinterpret_cast<uintptr_t>(k.purpose) + (h << 6u) + (h >> 2u);
		h ^= k.caller + (h << 6u) + (h >> 2u);
		return static_cast<size_t>(h * 0x94D049BB133111EBull);
	}
};
struct UncleanTotals {
	uint64_t count         = 0;
	uint64_t bytes         = 0;
	uint64_t first_address = 0;
	uint64_t first_size    = 0;
};
std::mutex                                                     g_unclean_mutex;
std::unordered_map<UncleanKey, UncleanTotals, UncleanKeyHash> g_unclean_reads;
std::unordered_map<NativeImageKey, NativeImageTotals, NativeImageKeyHash> g_native_images;

// LOD report watch state.
constexpr uint32_t kLodWatchMaxArms  = 96;   // total arming attempts per process
constexpr uint32_t kLodWatchInterval = 32;   // arm on every Nth report
constexpr uint32_t kLodWatchCodeHalf = 1024; // code bytes saved on each side of a new pc
struct WatchedPage {
	uint64_t page    = 0;
	uint32_t protect = 0;
};
std::mutex               g_watch_mutex;
std::vector<WatchedPage> g_watched_pages;
uint64_t                 g_watch_report_begin = 0;
uint64_t                 g_watch_report_end   = 0;
uint64_t                 g_watch_report_seq   = 0;
uint32_t                 g_watch_arms         = 0;
uint64_t                 g_watch_calls        = 0;
std::vector<uint64_t>    g_watch_seen_pcs;
std::vector<std::string> g_pending_watch_rows;

std::array<std::atomic<uint32_t>, 256> g_tex_count {};
std::array<std::atomic<uint64_t>, 256> g_tex_base {};
std::array<std::atomic<uint32_t>, 256> g_tex_info {};
// RecordTexture's per-thread counts (Detail::TextureTally), newest first.
std::atomic<Detail::TextureTally*> g_texture_tallies {nullptr};

// Textures resolved since the previous call, over every recording thread (the publisher only).
uint64_t TakeTexturesResolved() {
	static uint64_t published = 0;
	uint64_t        sum       = 0;
	for (auto* tally = g_texture_tallies.load(std::memory_order_acquire); tally != nullptr;
	     tally       = tally->next) {
		sum += tally->resolved.load(std::memory_order_relaxed);
	}
	const auto delta = sum - published;
	published        = sum;
	return delta;
}

struct Totals {
	std::atomic<uint64_t> flips {0};
	std::atomic<uint64_t> apr_reads {0};
	std::atomic<uint64_t> apr_bytes {0};
	std::atomic<uint64_t> apr_repeats {0};
	std::atomic<uint64_t> apr_errors {0};
	std::atomic<uint64_t> apr_grow_reads {0};
	std::atomic<uint64_t> apr_shrink_reads {0};
	std::atomic<uint64_t> apr_shrinks_since_flip {0}; // not reset by the summary
	std::atomic<uint64_t> apr_shrink_max_per_flip {0};
	std::atomic<uint64_t> pending_ops_max {0};
	std::atomic<uint64_t> lod_packets {0};
	std::atomic<uint64_t> lod_prior_nonzero {0};
	std::atomic<uint64_t> tex_mipstats {0};
	std::atomic<uint64_t> done_waits {0};
	std::atomic<uint64_t> done_ns {0};
	std::atomic<uint64_t> done_max_ns {0};
	std::atomic<uint64_t> readbacks {0};
	std::atomic<uint64_t> readback_ns {0};
	std::atomic<uint64_t> readback_downloads {0};
	std::atomic<uint64_t> image_frees {0};
	std::atomic<uint64_t> native_creates {0};
	std::atomic<uint64_t> native_create_bytes {0};
	std::atomic<uint64_t> gpu_busy_ns {0};
	std::atomic<uint64_t> gpu_idle_ns {0};
	std::atomic<uint64_t> gpu_max_gap_ns {0};
	std::atomic<uint64_t> gpu_starved_ns {0};
	std::atomic<uint64_t> gpu_cmdbufs {0};
	std::atomic<uint64_t> gpu_latency_samples {0};
	std::atomic<uint64_t> gpu_record_latency_ns {0};
	std::atomic<uint64_t> gpu_dispatch_latency_ns {0};
	std::atomic<uint64_t> gpu_dropped {0};
	std::atomic<uint64_t> gpu_render_passes {0};
	std::atomic<uint64_t> gpu_barriers {0};
	std::atomic<uint64_t> gpu_layout_transitions {0};
	std::atomic<uint64_t> gpu_guest_cmdbufs {0};
	std::atomic<uint64_t> gpu_barrier_requests {0};
	std::atomic<uint64_t> gpu_barriers_merged {0};
	std::atomic<uint64_t> gpu_barriers_elided {0};
	std::atomic<uint64_t> gpu_barriers_sunk {0};
	std::atomic<uint64_t> gpu_barrier_rp_splits {0};
	std::atomic<uint64_t> gpu_rendering_ends {0};
	std::atomic<uint64_t> gpu_draw_write_sinks {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(MemoryCounter::Count)> memory {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(TransferKind::Count)> transfer_count {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(TransferKind::Count)> transfer_bytes {};
	std::atomic<uint64_t> compile_programs {0};
	std::atomic<uint64_t> compile_translate_ns {0};
	std::atomic<uint64_t> compile_emit_ns {0};
	std::atomic<uint64_t> compile_validate_ns {0};
	std::atomic<uint64_t> compile_module_ns {0};
	std::atomic<uint64_t> compile_gfx_pipelines {0};
	std::atomic<uint64_t> compile_gfx_pipeline_ns {0};
	std::atomic<uint64_t> compile_cs_pipelines {0};
	std::atomic<uint64_t> compile_cs_pipeline_ns {0};
	std::atomic<uint64_t> compile_stall_ns {0};
	std::atomic<uint64_t> compile_stall_max_ns {0};
	std::atomic<uint64_t> compile_gfx_new {0};
	std::atomic<uint64_t> compile_gfx_perm {0};
	std::atomic<uint64_t> compile_gfx_variant {0};
	std::atomic<uint64_t> pcache_saves {0};
	std::atomic<uint64_t> pcache_save_bytes {0};
	std::atomic<uint64_t> pcache_save_serialize_ns {0};
	std::atomic<uint64_t> pcache_save_write_ns {0};
	std::atomic<uint64_t> pcache_save_overlaps {0};
	std::atomic<uint64_t> pcache_save_overlap_ns {0};
	std::atomic<uint64_t> compile_translation_reuses {0};
	std::atomic<uint64_t> compile_clone_ns {0};
	std::atomic<uint64_t> compile_disk_loads {0};
	std::atomic<uint64_t> compile_disk_load_ns {0};
	std::atomic<uint64_t> pred_flush_waits {0};
	std::atomic<uint64_t> pred_flush_wait_ns {0};
	std::atomic<uint64_t> validate_async {0};
	std::atomic<uint64_t> validate_async_ns {0};
	using PerLibraryEvent =
	    std::array<std::atomic<uint64_t>, static_cast<size_t>(PipelineLibraryEvent::Count)>;
	PerLibraryEvent gpl_count {};
	PerLibraryEvent gpl_ns {};
};
Totals g_totals;

constexpr uint64_t       kCompileRowLimit = 1'000'000;
constexpr const char*    kCompileKindNames[] = {"program", "graphics-pipeline", "compute-pipeline"};
static_assert(std::size(kCompileKindNames) == static_cast<size_t>(CompileKind::Count));
constexpr const char*    kPipelineOriginNames[] = {"", "new", "permutation", "variant"};
std::mutex               g_compile_mutex;
std::vector<std::string> g_pending_compile_rows;
uint64_t                 g_compile_rows_total = 0;
std::vector<std::string> g_pending_frame_rows; // guarded by g_compile_mutex
uint64_t                 g_frame_rows_total = 0;
uint64_t                 g_previous_flip_ns = 0;
bool FrameTimesEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_HANG_TRACE_FRAME_TIMES");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

struct MemoryCounterColumn {
	const char* name;
	uint64_t    divisor;
};
// summary.csv names of HangTrace::MemoryCounter, in enum order.
constexpr std::array<MemoryCounterColumn, static_cast<size_t>(MemoryCounter::Count)>
    kMemoryCounterColumns {{
        {"mem_write_faults", 1},
        {"mem_read_faults", 1},
        {"mem_fault_us", 1000},
        {"mem_protect_calls", 1},
        {"mem_protect_pages", 1},
        {"mem_unprotect_calls", 1},
        {"mem_unprotect_pages", 1},
        {"mem_protect_us", 1000},
        {"mem_tracker_lock_contended", 1},
        {"mem_scratch_allocs", 1},
        {"mem_scratch_bytes", 1},
        {"mem_scratch_us", 1000},
        {"mem_buffer_from_image", 1},
        {"mem_upload_copies", 1},
        {"mem_upload_barriers", 1},
        {"mem_upload_render_splits", 1},
        {"mem_image_writebacks", 1},
        {"mem_image_writeback_bytes", 1},
        {"mem_image_writeback_partial", 1},
        {"mem_image_writeback_skips", 1},
        {"mem_fault_ahead_pages", 1},
        {"mem_hot_promotions", 1},
        {"mem_hot_demotions", 1},
        {"mem_hot_upload_pages", 1},
        {"mem_hot_upload_skipped", 1},
        {"mem_written_upload_late_pages", 1},
    }};
static_assert(kMemoryCounterColumns.back().name != nullptr,
              "memory counter columns must match HangTrace::MemoryCounter");
// summary.csv emits the memory counters that existed when the compile columns were appended
// after them in that position; any later MemoryCounter goes to the end of the row, so every
// existing column keeps its index.
constexpr size_t kMemoryColumnsBeforeCompile =
    static_cast<size_t>(MemoryCounter::WrittenUploadLatePages) + 1;
static_assert(kMemoryColumnsBeforeCompile <= kMemoryCounterColumns.size());

std::mutex                  g_publish_mutex;
std::condition_variable_any g_publish_condition;
std::jthread                g_publisher;
std::filesystem::path       g_dir;

struct Files {
	std::FILE* summary       = nullptr;
	std::FILE* apr           = nullptr;
	std::FILE* imports       = nullptr;
	std::FILE* imports_index = nullptr;
	std::FILE* lod           = nullptr;
	std::FILE* tex           = nullptr;
	std::FILE* modules       = nullptr;
	std::FILE* queues        = nullptr;
	std::FILE* readbacks     = nullptr;
	std::FILE* images        = nullptr;
	std::FILE* lodwatch      = nullptr;
	std::FILE* occlusion     = nullptr;
	std::FILE* lodreports    = nullptr;
	std::FILE* transfers     = nullptr;
	std::FILE* compiles      = nullptr;
	std::FILE* frames        = nullptr;
	std::FILE* cp            = nullptr;
	std::FILE* unclean       = nullptr;
	std::FILE* timestamps    = nullptr;
	std::FILE* placement     = nullptr;
};
Files g_files;

std::mutex               g_timestamp_mutex;
std::vector<std::string> g_pending_timestamp_rows;

constexpr uint64_t       kCpRowLimit = 60'000'000; // ~6 GB at most; rows are flushed to disk
std::mutex               g_cp_mutex;
std::vector<std::string> g_pending_cp_rows;
uint64_t                 g_cp_rows_total = 0;
thread_local uint32_t    g_cp_queue      = UINT32_MAX;
thread_local uint64_t    g_cp_sequence   = 0;
using Detail::g_cp_watch_begin;
using Detail::g_cp_watch_end;

constexpr uint64_t       kOcclusionRowLimit = 2'000'000;
std::mutex               g_occlusion_mutex;
std::vector<std::string> g_pending_occlusion_rows;
uint64_t                 g_occlusion_rows_total = 0;
std::vector<std::string> g_pending_lodreport_rows; // guarded by g_occlusion_mutex
uint64_t                 g_lodreport_rows_total = 0;

void UpdateMax(std::atomic<uint64_t>& target, uint64_t value) {
	auto current = target.load(std::memory_order_relaxed);
	while (value > current &&
	       !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
	}
}

uint64_t NowMs() {
	return NowNs() / 1'000'000u;
}

uint64_t OsThreadId() {
#ifdef _WIN32
	return GetCurrentThreadId();
#else
	return static_cast<uint64_t>(gettid());
#endif
}

int ProcessId() {
#ifdef _WIN32
	return _getpid();
#else
	return static_cast<int>(getpid());
#endif
}

const GuestModule* FindModule(uint64_t address) {
	const auto count = g_module_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < count; i++) {
		const auto& m = g_modules[i];
		if (address >= m.base && address < m.base + m.size) {
			return &m;
		}
	}
	return nullptr;
}

#ifdef _WIN32
bool IsReadablePage(uint64_t address) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
		return false;
	}
	if (info.State != MEM_COMMIT) {
		return false;
	}
	const auto protect = info.Protect & 0xffu;
	if ((info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0) {
		return false;
	}
	return protect == PAGE_READONLY || protect == PAGE_READWRITE || protect == PAGE_WRITECOPY ||
	       protect == PAGE_EXECUTE_READ || protect == PAGE_EXECUTE_READWRITE ||
	       protect == PAGE_EXECUTE_WRITECOPY;
}

bool IsExecutablePage(uint64_t address) {
	MEMORY_BASIC_INFORMATION info {};
	if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
		return false;
	}
	if (info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0) {
		return false;
	}
	const auto protect = info.Protect & 0xffu;
	return protect == PAGE_EXECUTE_READ || protect == PAGE_EXECUTE_READWRITE ||
	       protect == PAGE_EXECUTE_WRITECOPY;
}
#endif

bool LooksLikeCallSite(uint64_t ra, const GuestModule& m) {
#ifndef _WIN32
	if (ra != 0) {
		return false; // stack scanning is implemented for Windows only
	}
#endif
	if (ra < m.base + 7) {
		return false;
	}
#ifdef _WIN32
	// Only inspect committed executable pages: a stale stack value may point into data that is
	// unmapped or GPU-protected, and touching it must never fault.
	if (!IsExecutablePage(ra - 7) || !IsExecutablePage(ra)) {
		return false;
	}
#endif
	const auto* p = reinterpret_cast<const uint8_t*>(ra);
	if (p[-5] == 0xE8) return true;                                       // call rel32
	if (p[-6] == 0xFF && p[-5] == 0x15) return true;                      // call [rip+disp32]
	if (p[-2] == 0xFF && (p[-1] & 0xF8u) == 0xD0) return true;            // call reg
	if (p[-2] == 0xFF && (p[-1] & 0xF8u) == 0x10 && (p[-1] & 7u) != 4u &&
	    (p[-1] & 7u) != 5u)
		return true;                                                      // call [reg]
	if (p[-3] == 0xFF && (p[-2] & 0xF8u) == 0x50 && (p[-2] & 7u) != 4u) return true; // [reg+d8]
	if (p[-4] == 0xFF && p[-3] == 0x54) return true;                      // [sib+d8]
	if (p[-6] == 0xFF && (p[-5] & 0xF8u) == 0x90 && (p[-5] & 7u) != 4u) return true; // [reg+d32]
	if (p[-7] == 0xFF && p[-6] == 0x94) return true;                      // [sib+d32]
	if (p[-3] == 0xFF && p[-2] == 0x14) return true;                      // [sib]
	return false;
}

std::string FormatAddress(uint64_t address) {
	if (const auto* m = FindModule(address); m != nullptr) {
		return fmt::format("{}+0x{:x}", m->name, address - m->base);
	}
	return fmt::format("0x{:x}", address);
}

// A host code address as "exe+0x<rva>" when it lies in the emulator's image (resolve with its
// PDB), else absolute.
std::string FormatHostAddress(uint64_t address) {
#ifdef _WIN32
	static const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	if (base != 0 && address >= base && address - base < 0x40000000u) {
		return fmt::format("exe+0x{:x}", address - base);
	}
#endif
	return fmt::format("0x{:x}", address);
}

// Heuristic guest call chain: scan the current stack for values that point just after a call
// instruction inside a registered guest module. Host frames are skipped automatically.
std::string CaptureGuestCallers() {
	std::string out;
#ifdef _WIN32
	uint8_t    marker = 0;
	const auto sp     = reinterpret_cast<uint64_t>(&marker) & ~uint64_t {7};
	uint64_t   checked_page = 0;
	uint32_t   found        = 0;
	for (uint64_t address = sp; address < sp + kScanBytes && found < kMaxCallers;
	     address += 8) {
		const auto page = address & ~uint64_t {0xfff};
		if (page != checked_page) {
			if (!IsReadablePage(page)) {
				break;
			}
			checked_page = page;
		}
		const auto value = *reinterpret_cast<const uint64_t*>(address);
		const auto* m    = FindModule(value);
		if (m == nullptr || !LooksLikeCallSite(value, *m)) {
			continue;
		}
		if (!out.empty()) {
			out += ';';
		}
		out += fmt::format("{}+0x{:x}", m->name, value - m->base);
		found++;
	}
#endif
	return out;
}

std::string CsvEscape(std::string_view text) {
	std::string out;
	out.reserve(text.size() + 2);
	out += '"';
	for (char c: text) {
		if (c == '"') {
			out += '"';
		}
		out += c;
	}
	out += '"';
	return out;
}

std::FILE* OpenFile(const char* name, const char* header) {
	const auto path = g_dir / name;
	auto*      file = std::fopen(path.string().c_str(), "wb");
	if (file != nullptr && header != nullptr) {
		std::fputs(header, file);
		std::fputc('\n', file);
	}
	return file;
}

void WriteRows(std::FILE* file, std::vector<std::string>& rows) {
	if (file == nullptr) {
		rows.clear();
		return;
	}
	for (const auto& row: rows) {
		std::fputs(row.c_str(), file);
		std::fputc('\n', file);
	}
	rows.clear();
}

void Publish() {
	const auto t_ms = NowMs();

	std::vector<std::string> rows;
	{
		std::scoped_lock lock(g_module_mutex);
		rows.swap(g_pending_module_rows);
	}
	WriteRows(g_files.modules, rows);

	{
		std::scoped_lock lock(g_import_mutex);
		rows.swap(g_pending_import_index_rows);
	}
	WriteRows(g_files.imports_index, rows);

	if (g_files.imports != nullptr) {
		std::scoped_lock lock(g_import_mutex);
		for (auto& entry: g_imports) {
			const auto value =
			    std::atomic_ref<uint64_t>(entry->count).load(std::memory_order_relaxed);
			if (value != entry->published) {
				std::fprintf(g_files.imports, "%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 "\n", t_ms,
				             entry->id, value - entry->published, value);
				entry->published = value;
			}
		}
	}

	{
		std::scoped_lock lock(g_apr_mutex);
		rows.swap(g_pending_apr_rows);
	}
	WriteRows(g_files.apr, rows);

	{
		std::scoped_lock lock(g_lod_mutex);
		rows.swap(g_pending_lod_rows);
	}
	WriteRows(g_files.lod, rows);

	{
		std::scoped_lock lock(g_watch_mutex);
		rows.swap(g_pending_watch_rows);
	}
	WriteRows(g_files.lodwatch, rows);

	{
		std::scoped_lock lock(g_readback_mutex);
		rows.swap(g_pending_readback_rows);
	}
	WriteRows(g_files.readbacks, rows);

	{
		std::scoped_lock lock(g_occlusion_mutex);
		rows.swap(g_pending_occlusion_rows);
	}
	WriteRows(g_files.occlusion, rows);

	{
		std::scoped_lock lock(g_occlusion_mutex);
		rows.swap(g_pending_lodreport_rows);
	}
	WriteRows(g_files.lodreports, rows);

	{
		std::scoped_lock lock(g_timestamp_mutex);
		rows.swap(g_pending_timestamp_rows);
	}
	WriteRows(g_files.timestamps, rows);

	if (g_files.placement != nullptr) {
		Common::PlacementHistogram histogram;
		Common::TakePlacementHistogram(histogram);
		static constexpr std::array<const char*, 4> kRoles {"cp", "recorder", "guest", "host"};
		static_assert(kRoles.size() == static_cast<size_t>(Common::ThreadRole::Count));
		for (size_t role = 0; role < histogram.samples.size(); role++) {
			uint64_t total = 0;
			uint64_t other = 0;
			for (uint32_t cpu = 0; cpu < Common::PlacementHistogram::Processors; cpu++) {
				total += histogram.samples[role][cpu];
				other += cpu >= 32 ? histogram.samples[role][cpu] : 0u;
			}
			if (total == 0) {
				continue;
			}
			std::string row = fmt::format(
			    "{},{},{},{},{}", t_ms, kRoles[role], total, histogram.on_cp_core[role],
			    histogram.cp_core == UINT32_MAX ? int64_t {-1} : int64_t {histogram.cp_core});
			for (uint32_t cpu = 0; cpu < 32; cpu++) {
				row += fmt::format(",{}", histogram.samples[role][cpu]);
			}
			row += fmt::format(",{}", other);
			std::fputs(row.c_str(), g_files.placement);
			std::fputc('\n', g_files.placement);
		}
	}

	if (g_files.cp != nullptr) {
		{
			std::scoped_lock lock(g_cp_mutex);
			rows.swap(g_pending_cp_rows);
		}
		WriteRows(g_files.cp, rows);
	}

	{
		std::scoped_lock lock(g_image_mutex);
		rows.swap(g_pending_image_rows);
		for (const auto& [key, totals]: g_native_images) {
			rows.push_back(fmt::format("{},{},,,{},{},{},{},,{},0x{:x},{},{}", t_ms,
			                           key.create ? (key.hit ? "native-create-pooled" : "native-create")
			                                      : "native-destroy",
			                           key.width, key.height, key.levels, 1, key.format, key.usage,
			                           totals.count, totals.bytes));
		}
		g_native_images.clear();
	}
	WriteRows(g_files.images, rows);

	{
		std::scoped_lock lock(g_transfer_mutex);
		for (const auto& [key, totals]: g_transfers) {
			rows.push_back(fmt::format(
			    "{},{},{},{},0x{:x},{},{},{},{},{},{}", t_ms,
			    kTransferKindNames[static_cast<uint32_t>(key.kind)], key.reason, key.detail,
			    key.address, key.format, key.width, key.height, totals.count, totals.bytes,
			    totals.span_bytes));
		}
		g_transfers.clear();
	}
	WriteRows(g_files.transfers, rows);

	{
		std::unordered_map<UncleanKey, UncleanTotals, UncleanKeyHash> unclean;
		{
			std::scoped_lock lock(g_unclean_mutex);
			unclean.swap(g_unclean_reads);
		}
		const auto now_ms = NowMs();
		std::scoped_lock lock(g_page_writer_mutex);
		for (const auto& [key, totals]: unclean) {
			std::string writer = ",,";
			if (auto it = g_page_writers.find(key.page >> 12u); it != g_page_writers.end()) {
				writer = fmt::format("{},{},{}",
				                     kGpuWriteKindNames[static_cast<uint32_t>(it->second.kind)],
				                     now_ms - std::min(now_ms, it->second.t_ms), it->second.size);
			}
			rows.push_back(fmt::format("{},{},{},{},0x{:x},{},{},0x{:x},{},{}", t_ms, key.reason,
			                           key.purpose != nullptr ? key.purpose : "",
			                           FormatHostAddress(key.caller), key.page, totals.count,
			                           totals.bytes, totals.first_address, totals.first_size,
			                           writer));
		}
	}
	WriteRows(g_files.unclean, rows);

	{
		std::scoped_lock lock(g_compile_mutex);
		rows.swap(g_pending_compile_rows);
	}
	WriteRows(g_files.compiles, rows);
	{
		std::scoped_lock lock(g_compile_mutex);
		rows.swap(g_pending_frame_rows);
	}
	WriteRows(g_files.frames, rows);

	if (g_files.tex != nullptr) {
		for (uint32_t id = 0; id < 256; id++) {
			const auto count = g_tex_count[id].exchange(0, std::memory_order_relaxed);
			if (count == 0) {
				continue;
			}
			const auto base = g_tex_base[id].load(std::memory_order_relaxed);
			const auto info = g_tex_info[id].load(std::memory_order_relaxed);
			std::fprintf(g_files.tex,
			             "%" PRIu64 ",%u,%u,0x%" PRIx64 ",%u,%u,%u,%u\n", t_ms, id, count, base,
			             info & 0xfffu, (info >> 12u) & 0xfffu, (info >> 24u) & 0xfu,
			             (info >> 28u) & 0xfu);
		}
	}

	if (g_files.summary != nullptr) {
		auto take = [](std::atomic<uint64_t>& v) { return v.exchange(0, std::memory_order_relaxed); };
		const auto flips       = take(g_totals.flips);
		const auto apr_reads   = take(g_totals.apr_reads);
		const auto apr_bytes   = take(g_totals.apr_bytes);
		const auto apr_repeats = take(g_totals.apr_repeats);
		const auto apr_errors  = take(g_totals.apr_errors);
		const auto lod         = take(g_totals.lod_packets);
		const auto lod_prior   = take(g_totals.lod_prior_nonzero);
		const auto tex_total   = TakeTexturesResolved();
		const auto tex_mip     = take(g_totals.tex_mipstats);
		const auto done_n      = take(g_totals.done_waits);
		const auto done_ns     = take(g_totals.done_ns);
		const auto done_max    = take(g_totals.done_max_ns);
		std::string line = fmt::format(
		    "{},{},{},{},{},{},{},{},{},{},{},{},{}", t_ms, flips, apr_reads, apr_bytes,
		    apr_repeats, apr_errors, lod, lod_prior, tex_total, tex_mip, done_n,
		    done_n != 0 ? done_ns / done_n / 1000u : 0, done_max / 1000u);
		uint64_t c_starts = 0, c_wait = 0, c_wmax = 0, c_busy = 0, c_slices = 0, c_inc = 0;
		for (uint32_t index = 0; index < kMaxQueues; index++) {
			auto&      q      = g_queues[index];
			const auto starts = take(q.starts);
			const auto wait   = take(q.wait_ns);
			const auto wmax   = take(q.wait_max_ns);
			const auto busy   = take(q.busy_ns);
			const auto slices = take(q.slices);
			const auto inc    = take(q.incomplete);
			if (starts == 0 && slices == 0) {
				if (index == 0) {
					line += ",0,0,0,0,0,0";
				}
				continue;
			}
			if (g_files.queues != nullptr) {
				std::fprintf(g_files.queues,
				             "%" PRIu64 ",%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
				             ",%" PRIu64 ",%" PRIu64 "\n",
				             t_ms, index, starts, starts != 0 ? wait / starts / 1000u : 0,
				             wmax / 1000u, busy / 1000u, slices, inc);
			}
			if (index == 0) {
				line += fmt::format(",{},{},{},{},{},{}", starts,
				                    starts != 0 ? wait / starts / 1000u : 0, wmax / 1000u,
				                    busy / 1000u, slices, inc);
			} else {
				c_starts += starts;
				c_wait += wait;
				c_wmax = std::max(c_wmax, wmax);
				c_busy += busy;
				c_slices += slices;
				c_inc += inc;
			}
		}
		line += fmt::format(",{},{},{},{},{},{}", c_starts,
		                    c_starts != 0 ? c_wait / c_starts / 1000u : 0, c_wmax / 1000u,
		                    c_busy / 1000u, c_slices, c_inc);
		line += fmt::format(",{},{},{},{},{},{}", take(g_totals.readbacks),
		                    take(g_totals.readback_ns) / 1000u, take(g_totals.readback_downloads),
		                    take(g_totals.image_frees), take(g_totals.native_creates),
		                    take(g_totals.native_create_bytes));
		// Appended GPU timeline columns; keep them last so older column indices stay valid.
		const auto gpu_samples = take(g_totals.gpu_latency_samples);
		const auto gpu_record  = take(g_totals.gpu_record_latency_ns);
		const auto gpu_dispatch = take(g_totals.gpu_dispatch_latency_ns);
		line += fmt::format(",{},{},{},{},{},{},{},{}", take(g_totals.gpu_busy_ns) / 1000u,
		                    take(g_totals.gpu_cmdbufs),
		                    gpu_samples != 0 ? gpu_record / gpu_samples / 1000u : 0,
		                    take(g_totals.gpu_idle_ns) / 1000u,
		                    take(g_totals.gpu_max_gap_ns) / 1000u,
		                    take(g_totals.gpu_starved_ns) / 1000u,
		                    gpu_samples != 0 ? gpu_dispatch / gpu_samples / 1000u : 0,
		                    take(g_totals.gpu_dropped));
		line += fmt::format(",{},{},{},{}", take(g_totals.gpu_render_passes),
		                    take(g_totals.gpu_barriers), take(g_totals.gpu_layout_transitions),
		                    take(g_totals.gpu_guest_cmdbufs));
		line += fmt::format(",{},{},{},{},{}", take(g_totals.gpu_barrier_requests),
		                    take(g_totals.gpu_barriers_merged), take(g_totals.gpu_barriers_elided),
		                    take(g_totals.gpu_barriers_sunk), take(g_totals.gpu_barrier_rp_splits));
		for (size_t kind = 0; kind < static_cast<size_t>(TransferKind::Count); kind++) {
			line += fmt::format(",{},{}", take(g_totals.transfer_count[kind]),
			                    take(g_totals.transfer_bytes[kind]));
		}
		line += fmt::format(",{},{}", take(g_totals.gpu_rendering_ends),
		                    take(g_totals.gpu_draw_write_sinks));
		const auto memory_columns = [&](size_t begin, size_t end) {
			for (size_t counter = begin; counter < end; counter++) {
				line += fmt::format(",{}", take(g_totals.memory[counter]) /
				                               kMemoryCounterColumns[counter].divisor);
			}
		};
		memory_columns(0, kMemoryColumnsBeforeCompile);
		// Compile columns (appended after the memory columns).
		line += fmt::format(",{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
		                    take(g_totals.compile_programs),
		                    take(g_totals.compile_translate_ns) / 1000u,
		                    take(g_totals.compile_emit_ns) / 1000u,
		                    take(g_totals.compile_validate_ns) / 1000u,
		                    take(g_totals.compile_module_ns) / 1000u,
		                    take(g_totals.compile_gfx_pipelines),
		                    take(g_totals.compile_gfx_pipeline_ns) / 1000u,
		                    take(g_totals.compile_cs_pipelines),
		                    take(g_totals.compile_cs_pipeline_ns) / 1000u,
		                    take(g_totals.compile_stall_ns) / 1000u,
		                    take(g_totals.compile_stall_max_ns) / 1000u,
		                    take(g_totals.compile_gfx_new), take(g_totals.compile_gfx_perm),
		                    take(g_totals.compile_gfx_variant));
		line += fmt::format(",{},{},{},{},{},{}", take(g_totals.pcache_saves),
		                    take(g_totals.pcache_save_bytes) / 1024u,
		                    take(g_totals.pcache_save_serialize_ns) / 1000u,
		                    take(g_totals.pcache_save_write_ns) / 1000u,
		                    take(g_totals.pcache_save_overlaps),
		                    take(g_totals.pcache_save_overlap_ns) / 1000u);
		line += fmt::format(",{},{}", take(g_totals.compile_translation_reuses),
		                    take(g_totals.compile_clone_ns) / 1000u);
		line += fmt::format(",{},{}", take(g_totals.validate_async),
		                    take(g_totals.validate_async_ns) / 1000u);
		const auto gpl = [&](PipelineLibraryEvent event, bool with_time) {
			const auto index = static_cast<size_t>(event);
			line += fmt::format(",{}", take(g_totals.gpl_count[index]));
			const auto ns = take(g_totals.gpl_ns[index]);
			if (with_time) line += fmt::format(",{}", ns / 1000u);
		};
		gpl(PipelineLibraryEvent::CacheHit, false);
		gpl(PipelineLibraryEvent::Linked, true);
		gpl(PipelineLibraryEvent::Library, true);
		gpl(PipelineLibraryEvent::Optimized, true);
		// Memory counters added after the compile columns (none yet).
		memory_columns(kMemoryColumnsBeforeCompile, kMemoryCounterColumns.size());
		uint64_t stream_bytes = 0;
		{
			std::scoped_lock apr_lock(g_apr_mutex);
			stream_bytes = g_apr_stream_bytes;
		}
		line += fmt::format(",{},{},{},{}", take(g_totals.apr_grow_reads),
		                    take(g_totals.apr_shrink_reads),
		                    take(g_totals.apr_shrink_max_per_flip), stream_bytes >> 20u);
		line += fmt::format(",{}", take(g_totals.pending_ops_max));
		// Persistent program cache (appended last so that no existing column moves).
		line += fmt::format(",{},{}", take(g_totals.compile_disk_loads),
		                    take(g_totals.compile_disk_load_ns) / 1000u);
		// Memory predication packets that drained the GPU (appended last).
		line += fmt::format(",{},{}", take(g_totals.pred_flush_waits),
		                    take(g_totals.pred_flush_wait_ns) / 1000u);
		std::fputs(line.c_str(), g_files.summary);
		std::fputc('\n', g_files.summary);
	}

	for (auto* file: {g_files.summary, g_files.apr, g_files.imports, g_files.imports_index,
	                  g_files.lod, g_files.tex, g_files.modules, g_files.queues, g_files.readbacks,
	                  g_files.images, g_files.lodwatch, g_files.transfers, g_files.compiles, g_files.frames,
	                  g_files.unclean, g_files.timestamps, g_files.placement}) {
		if (file != nullptr) {
			std::fflush(file);
		}
	}
}

} // namespace

bool Detail::ReadEnabled() noexcept {
	const bool enabled = EnvFlag("KYTY_HANG_TRACE", false);
	g_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
	return enabled;
}

bool ImportsEnabled() {
	static const bool enabled = Enabled() && EnvFlag("KYTY_HANG_TRACE_IMPORTS", true);
	return enabled;
}

bool CpTraceEnabled() {
	static const bool enabled = Enabled() && EnvFlag("KYTY_HANG_TRACE_CP", false);
	return enabled;
}

void SetCpContext(uint32_t queue, uint64_t sequence) {
	g_cp_queue    = queue;
	g_cp_sequence = sequence;
}

void RecordCp(const CpEvent& event) {
	if (!CpTraceEnabled()) {
		return;
	}
	const auto queue    = event.queue >= 0 ? static_cast<uint32_t>(event.queue) : g_cp_queue;
	const auto sequence = event.queue >= 0 ? event.seq : g_cp_sequence;
	const auto t_us     = NowNs() / 1000u;
	std::scoped_lock lock(g_cp_mutex);
	if (g_cp_rows_total >= kCpRowLimit) {
		return;
	}
	auto row = fmt::format("{},{},{},{},{},{},0x{:x},0x{:x},0x{:x},0x{:x},{},{}", t_us,
	                       g_cp_rows_total, OsThreadId(),
	                       queue == UINT32_MAX ? std::string("-") : fmt::format("0x{:x}", queue),
	                       sequence, event.event, event.address, event.value, event.ref,
	                       event.mask, event.aux, event.size);
	g_cp_rows_total++;
	g_pending_cp_rows.push_back(std::move(row));
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - g_start)
	                                 .count());
}

void Initialize() {
	if (!Enabled() || g_publisher.joinable()) {
		return;
	}
	if (const auto* dir = std::getenv("KYTY_HANG_TRACE_DIR"); dir != nullptr && dir[0] != 0) {
		g_dir = std::filesystem::path(dir);
	} else {
		const auto now  = std::chrono::system_clock::now();
		const auto secs = std::chrono::system_clock::to_time_t(now);
		std::tm    tm_v {};
#ifdef _WIN32
		localtime_s(&tm_v, &secs);
#else
		localtime_r(&secs, &tm_v);
#endif
		char stamp[32] {};
		std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", &tm_v);
		g_dir = std::filesystem::path("_HangTrace") / fmt::format("{}-pid{}", stamp, ProcessId());
	}
	std::error_code error;
	std::filesystem::create_directories(g_dir, error);
	if (error) {
		std::printf("Hang trace: cannot create output directory %s: %s\n", g_dir.string().c_str(),
		            error.message().c_str());
		return;
	}

	std::string summary_header =
	    "t_ms,flips,apr_reads,apr_bytes,apr_repeat_reads,apr_errors,lod_packets,"
	    "lod_prior_nonzero,tex_resolved,tex_mipstats,done_waits,done_avg_us,done_max_us";
	for (const char* q: {"gfx", "compute"}) {
		summary_header += fmt::format(",{0}_starts,{0}_wait_avg_us,{0}_wait_max_us,{0}_busy_us,"
		                              "{0}_slices,{0}_incomplete",
		                              q);
	}
	summary_header += ",readbacks,readback_us,readback_downloads,image_frees,native_creates,"
	                  "native_create_bytes";
	summary_header += ",gpu_busy_us,gpu_cmdbufs,gpu_latency_avg_us,gpu_idle_us,gpu_max_gap_us,"
	                  "gpu_starved_us,gpu_dispatch_latency_avg_us,gpu_dropped";
	summary_header += ",gpu_render_passes,gpu_barriers,gpu_layout_transitions,gpu_guest_cmdbufs";
	summary_header += ",gpu_barrier_requests,gpu_barriers_merged,gpu_barriers_elided,"
	                  "gpu_barriers_sunk,gpu_barrier_rp_splits";
	summary_header += ",xfer_image_uploads,xfer_image_upload_bytes,xfer_buffer_uploads,"
	                  "xfer_buffer_upload_bytes,xfer_image_copies,xfer_image_copy_bytes,"
	                  "xfer_alias_syncs,xfer_alias_sync_bytes";
	summary_header += ",gpu_rendering_ends,gpu_draw_write_sinks";
	const auto memory_header = [&](size_t begin, size_t end) {
		for (size_t counter = begin; counter < end; counter++) {
			summary_header += ',';
			summary_header += kMemoryCounterColumns[counter].name;
		}
	};
	memory_header(0, kMemoryColumnsBeforeCompile);
	summary_header += ",compile_programs,compile_translate_us,compile_emit_us,compile_validate_us,"
	                  "compile_module_us,compile_gfx_pipelines,compile_gfx_pipeline_us,"
	                  "compile_cs_pipelines,compile_cs_pipeline_us,compile_stall_us,"
	                  "compile_stall_max_us,compile_gfx_new,compile_gfx_perm,compile_gfx_variant";
	summary_header += ",pcache_saves,pcache_save_kb,pcache_save_serialize_us,pcache_save_write_us,"
	                  "pcache_save_overlaps,pcache_save_overlap_us";
	summary_header += ",compile_translation_reuses,compile_clone_us";
	summary_header += ",validate_async_count,validate_async_us";
	summary_header += ",gpl_cache_hits,gpl_links,gpl_link_us,gpl_libraries,gpl_library_us,"
	                  "gpl_optimized,gpl_optimize_us";
	memory_header(kMemoryColumnsBeforeCompile, kMemoryCounterColumns.size());
	// Reads from offset 0 that grew or shrank the file's previous offset-0 read, the most shrink
	// reads between two flips, and the texture-streamer footprint (MiB): streamed-texture
	// residency changes.
	summary_header += ",apr_grow_reads,apr_shrink_reads,apr_shrink_max_per_flip,apr_stream_mib";
	// KYTY_PENDING_OPS_NOWAIT: the deepest normal-operation queue a draw or dispatch left queued.
	summary_header += ",pending_ops_max";
	// Persistent program cache (KYTY_PROGRAM_CACHE): permutations reloaded instead of translated
	// and emitted, and the time their keys, lookups and decoding took.
	summary_header += ",compile_disk_loads,compile_disk_load_us";
	// Memory predication packets (op 3, wait) that submitted all GPU work and blocked until it
	// finished, and the time blocked.
	summary_header += ",pred_flush_waits,pred_flush_wait_us";
	g_files.summary = OpenFile("summary.csv", summary_header.c_str());
	g_files.compiles = OpenFile("compiles.csv",
	                            "t_ms,kind,stage,guest_hash,id,id2,origin,translate_us,emit_us,"
	                            "validate_us,module_us,pipeline_us,total_us,spirv_words,host_tid,"
	                            "detail,clone_us,load_us");
	if (FrameTimesEnabled()) g_files.frames = OpenFile("frames.csv", "t_us,interval_us");
	g_files.transfers = OpenFile("transfers.csv",
	                             "t_ms,kind,reason,detail,address,format,width,height,count,bytes,"
	                             "span_bytes");
	g_files.unclean   = OpenFile("unclean.csv",
	                             "t_ms,reason,purpose,caller,page,count,bytes,first_address,"
	                             "first_size,last_gpu_writer,last_gpu_write_age_ms,"
	                             "last_gpu_write_size");
	g_files.readbacks = OpenFile("readbacks.csv",
	                             "t_ms,kind,vaddr,size,window_begin,window_size,downloaded,"
	                             "duration_us,host_tid,thread,pc,stack_callers,last_gpu_writer,"
	                             "last_gpu_write_age_ms,last_gpu_write_size");
	g_files.occlusion = OpenFile("occlusion.csv",
	                             "t_ms,event,address,value,scopes,width,height,colors,has_depth,"
	                             "depth_format,depth_address,condition,skip,detail");
	g_files.lodreports = OpenFile("lodreports.csv",
	                              "t_ms,destination,control,has_latest,sampled_counters,"
	                              "total_samples,mean_finest_mip,pending_copies,drawn_counters,"
	                              "counted_counters");
	g_files.timestamps = OpenFile("timestamps.csv",
	                              "t_ms,rewritten,skipped,unavailable,deferred,shift_avg_us,"
	                              "shift_max_us,publish_us,publishes,ring0_us,ring1_us,ring2_us,"
	                              "ring3_us,ring4_us,ring5_us,ring6_us,ring7_us,drs_index,drs_level,"
	                              "drs_room_frames,drs_fps,drs_scalable_ms,drs_total_ms");
	{
		std::string header = "t_ms,role,samples,on_cp_core,cp_core";
		for (uint32_t cpu = 0; cpu < 32; cpu++) {
			header += ",lp" + std::to_string(cpu);
		}
		header += ",lp_other";
		g_files.placement = OpenFile("placement.csv", header.c_str());
	}
	if (CpTraceEnabled()) {
		g_files.cp = OpenFile("cp.csv", "t_us,row,host_tid,queue,seq,event,address,value,ref,mask,"
		                                "aux,size");
		if (const auto* watch = std::getenv("KYTY_HANG_TRACE_CP_WATCH"); watch != nullptr) {
			char*      end   = nullptr;
			const auto begin = std::strtoull(watch, &end, 0);
			if (end != nullptr && *end == ':') {
				const auto size  = std::strtoull(end + 1, nullptr, 0);
				g_cp_watch_begin = begin;
				g_cp_watch_end   = begin + size;
			}
		}
	}
	g_files.lodwatch  = OpenFile("lodwatch.csv",
	                             "t_ms,report_seq,fault_vaddr,report_offset,access,pc,thread,"
	                             "rax,rbx,rcx,rdx,rsi,rdi,rbp,rsp,r8,r9,r10,r11,r12,r13,r14,r15,"
	                             "code_file,stack_callers");
	g_files.images    = OpenFile("images.csv",
	                             "t_ms,event,address,age_ticks,width,height,levels,layers,reason,"
	                             "format,usage,count,bytes");
	g_files.apr     = OpenFile("apr.csv",
	                           "t_ms,host_tid,thread,file_id,offset,size,destination,bytes_read,"
	                               "result,repeat_count,first_seen_ms,previous_destination,path,"
	                               "caller,stack_callers");
	if (ImportsEnabled()) {
		g_files.imports       = OpenFile("imports.csv", "t_ms,import_id,delta,total");
		g_files.imports_index = OpenFile("imports-index.csv", "import_id,target,nid,host_function,program");
	}
	g_files.lod     = OpenFile("lodstats.csv",
	                           "t_ms,sequence,destination,size,control,report_reset,force_reset,"
	                               "prior_nonzero_dwords,prior_first_nonzero,prior_hash,prior_head8,"
	                               "prior_tail8");
	g_files.tex     = OpenFile("texstats.csv",
	                           "t_ms,counter_id,uses,last_base,min_lod,min_lod_warn,base_level,"
	                               "last_level");
	g_files.modules = OpenFile("modules.csv", "name,base,size");
	g_files.queues  = OpenFile("queues.csv",
	                           "t_ms,queue,starts,wait_avg_us,wait_max_us,busy_us,slices,incomplete");

	g_publisher = std::jthread([](std::stop_token stop) {
		std::unique_lock lock(g_publish_mutex);
		while (!stop.stop_requested()) {
			g_publish_condition.wait_for(lock, stop, std::chrono::seconds(1),
			                             [] { return false; });
			Publish();
		}
	});
	std::printf("Hang trace enabled (KYTY_HANG_TRACE=1): writing %s\n", g_dir.string().c_str());
}

std::string OutputDirectory() {
	if (!Enabled() || !g_publisher.joinable()) {
		return {};
	}
	return g_dir.string();
}

void Shutdown() {
	DisarmLodReportWatch();
	if (g_publisher.joinable()) {
		g_publisher.request_stop();
		g_publisher.join();
		Publish();
	}
	for (auto** file: {&g_files.summary, &g_files.apr, &g_files.imports, &g_files.imports_index,
	                   &g_files.lod, &g_files.tex, &g_files.modules, &g_files.queues,
	                   &g_files.readbacks, &g_files.images, &g_files.lodwatch,
	                   &g_files.transfers, &g_files.compiles, &g_files.frames, &g_files.unclean,
	                   &g_files.timestamps, &g_files.placement}) {
		if (*file != nullptr) {
			std::fclose(*file);
			*file = nullptr;
		}
	}
}

void RegisterGuestCode(uint64_t base, uint64_t size, std::string_view name) {
	HangWatchdog::RegisterGuestCode(base, size, name);
	if (!Enabled() || size == 0) {
		return;
	}
	std::scoped_lock lock(g_module_mutex);
	const auto count = g_module_count.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < count; i++) {
		if (g_modules[i].base == base) {
			return;
		}
	}
	if (count >= kMaxModules) {
		return;
	}
	g_modules[count].base = base;
	g_modules[count].size = size;
	g_modules[count].name = std::string(name);
	g_module_count.store(count + 1, std::memory_order_release);
	g_pending_module_rows.push_back(fmt::format("{},0x{:x},0x{:x}", CsvEscape(name), base, size));
}

bool IsGuestAddress(uint64_t address) {
	return FindModule(address) != nullptr;
}

uint64_t* AllocateImportCounter(uint64_t target, std::string_view nid, std::string_view dbg_name,
                                std::string_view program) {
	std::scoped_lock lock(g_import_mutex);
	if (auto it = g_imports_by_target.find(target); it != g_imports_by_target.end()) {
		return &it->second->count;
	}
	auto entry      = std::make_unique<ImportEntry>();
	entry->target   = target;
	entry->id       = g_import_count.fetch_add(1, std::memory_order_relaxed);
	entry->nid      = std::string(nid);
	entry->dbg_name = std::string(dbg_name);
	entry->program  = std::string(program);
	auto* raw       = entry.get();
	g_imports.push_back(std::move(entry));
	g_imports_by_target.emplace(target, raw);
	g_pending_import_index_rows.push_back(fmt::format("{},0x{:x},{},{},{}", raw->id, target,
	                                                  CsvEscape(raw->nid),
	                                                  CsvEscape(raw->dbg_name),
	                                                  CsvEscape(raw->program)));
	return &raw->count;
}

uint64_t FindImportThunk(uint64_t target) {
	std::scoped_lock lock(g_import_mutex);
	if (auto it = g_imports_by_target.find(target); it != g_imports_by_target.end()) {
		return it->second->thunk;
	}
	return 0;
}

void SetImportThunk(uint64_t target, uint64_t thunk) {
	std::scoped_lock lock(g_import_mutex);
	if (auto it = g_imports_by_target.find(target); it != g_imports_by_target.end()) {
		it->second->thunk = thunk;
	}
}

void SetGuestCaller(uint64_t return_address) {
	g_guest_caller = return_address;
}

void RecordAprRead(uint32_t file_id, std::string_view host_path, uint64_t file_offset,
                   uint64_t size, uint64_t destination, uint64_t bytes_read, int result,
                   std::string_view thread_name) {
	if (!Enabled()) {
		return;
	}
	const auto t_ms = NowMs();
	g_totals.apr_reads.fetch_add(1, std::memory_order_relaxed);
	g_totals.apr_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
	if (result != 0) {
		g_totals.apr_errors.fetch_add(1, std::memory_order_relaxed);
	}
	const auto caller  = g_guest_caller != 0 ? FormatAddress(g_guest_caller) : std::string();
	// A stack scan costs hundreds of VirtualQuery calls. Boot issues thousands of reads from a
	// few call sites, and scanning every one stalled the guest streamer (4 s -> 25 s to the
	// first level). Scan only the first reads of each call site, then every 1024th.
	thread_local std::unordered_map<uint64_t, uint32_t> scanned_callers;
	auto&      seen    = scanned_callers[g_guest_caller];
	const bool scan    = seen < 4 || (seen % 1024u) == 0;
	seen++;
	const auto callers = scan ? CaptureGuestCallers() : std::string();

	std::scoped_lock lock(g_apr_mutex);
	if (file_offset == 0 && result == 0 && bytes_read != 0) {
		auto& last = g_apr_offset0_size[file_id];
		if (last != 0 && bytes_read > last) {
			g_totals.apr_grow_reads.fetch_add(1, std::memory_order_relaxed);
		} else if (bytes_read < last) {
			g_totals.apr_shrink_reads.fetch_add(1, std::memory_order_relaxed);
			g_totals.apr_shrinks_since_flip.fetch_add(1, std::memory_order_relaxed);
		}
		last = bytes_read;
		if (thread_name.find("TextureStreamer") != std::string_view::npos) {
			auto& stream = g_apr_stream_size[file_id];
			g_apr_stream_bytes = g_apr_stream_bytes - stream + bytes_read;
			stream             = bytes_read;
		}
	}
	auto& info = g_apr_keys[AprKey {file_id, file_offset, size}];
	const auto previous_destination = info.last_dest;
	if (info.count == 0) {
		info.first_ms = t_ms;
	} else {
		g_totals.apr_repeats.fetch_add(1, std::memory_order_relaxed);
	}
	info.count++;
	info.last_ms   = t_ms;
	info.last_dest = destination;
	if (g_apr_rows_total >= kAprRowLimit) {
		return;
	}
	g_apr_rows_total++;
	g_pending_apr_rows.push_back(fmt::format(
	    "{},{},{},0x{:08x},{},{},0x{:x},{},{},{},{},0x{:x},{},{},{}", t_ms,
	    OsThreadId(), CsvEscape(thread_name), file_id, file_offset, size, destination, bytes_read, result,
	    info.count, info.first_ms, previous_destination, CsvEscape(host_path), caller,
	    CsvEscape(callers)));
}

void RecordLodStats(const void* destination, uint32_t size, uint32_t control) {
	if (!Enabled()) {
		return;
	}
	g_totals.lod_packets.fetch_add(1, std::memory_order_relaxed);
	if (destination == nullptr || size < 4) {
		return;
	}
	const auto* words      = static_cast<const uint32_t*>(destination);
	const auto  word_count = size / 4u;
	uint32_t    nonzero    = 0;
	int64_t     first_nz   = -1;
	uint64_t    hash       = 1469598103934665603ull;
	for (uint32_t i = 0; i < word_count; i++) {
		const auto w = words[i];
		if (w != 0) {
			nonzero++;
			if (first_nz < 0) {
				first_nz = i;
			}
		}
		hash = (hash ^ w) * 1099511628211ull;
	}
	if (nonzero != 0) {
		g_totals.lod_prior_nonzero.fetch_add(1, std::memory_order_relaxed);
	}
	const auto sequence = g_lod_sequence.fetch_add(1, std::memory_order_relaxed);
	if (sequence >= 512 && (sequence % 64u) != 0) {
		return;
	}
	std::string head;
	std::string tail;
	for (uint32_t i = 0; i < 8 && i < word_count; i++) {
		head += fmt::format("{}{:08x}", i == 0 ? "" : " ", words[i]);
	}
	const auto tail_start = word_count > 8 ? word_count - 8 : 0;
	for (uint32_t i = tail_start; i < word_count; i++) {
		tail += fmt::format("{}{:08x}", i == tail_start ? "" : " ", words[i]);
	}
	auto row = fmt::format("{},{},0x{:x},{},0x{:08x},{},{},{},{},0x{:016x},{},{}", NowMs(), sequence,
	                       reinterpret_cast<uint64_t>(destination), size, control,
	                       (control >> 19u) & 1u, (control >> 18u) & 1u, nonzero, first_nz, hash,
	                       head, tail);
	std::scoped_lock lock(g_lod_mutex);
	g_pending_lod_rows.push_back(std::move(row));
}

namespace {

bool LodWatchEnabled() {
	static const bool enabled = Enabled() && EnvFlag("KYTY_HANG_TRACE_LOD_WATCH", true);
	return enabled;
}

// Caller holds g_watch_mutex.
void RestoreWatchedPagesLocked() {
#ifdef _WIN32
	for (const auto& watched: g_watched_pages) {
		DWORD old = 0;
		VirtualProtect(reinterpret_cast<void*>(watched.page), 0x1000, watched.protect, &old);
	}
#endif
	g_watched_pages.clear();
}

} // namespace

void DisarmLodReportWatch() {
	if (!LodWatchEnabled()) {
		return;
	}
	std::scoped_lock lock(g_watch_mutex);
	RestoreWatchedPagesLocked();
}

void ArmLodReportWatch(const void* destination, uint32_t size) {
#ifdef _WIN32
	if (!LodWatchEnabled() || destination == nullptr || size == 0) {
		return;
	}
	std::scoped_lock lock(g_watch_mutex);
	const auto call = g_watch_calls++;
	// Skip the first reports (boot/menu) and then sample sparsely.
	if (call < 1024 || (call % kLodWatchInterval) != 0 || g_watch_arms >= kLodWatchMaxArms) {
		return;
	}
	RestoreWatchedPagesLocked();
	const auto begin = reinterpret_cast<uint64_t>(destination);
	const auto end   = begin + size;
	std::vector<WatchedPage> pages;
	for (uint64_t page = begin & ~uint64_t {0xfff}; page < end; page += 0x1000) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(page), &info, sizeof(info)) == 0 ||
		    info.State != MEM_COMMIT || info.Protect != PAGE_READWRITE) {
			// GPU-tracked or unusual pages keep their protection untouched.
			return;
		}
		pages.push_back({page, info.Protect});
	}
	for (const auto& watched: pages) {
		DWORD old = 0;
		if (VirtualProtect(reinterpret_cast<void*>(watched.page), 0x1000, PAGE_NOACCESS, &old) == 0) {
			RestoreWatchedPagesLocked();
			return;
		}
		g_watched_pages.push_back(watched);
	}
	g_watch_report_begin = begin;
	g_watch_report_end   = end;
	g_watch_report_seq   = call;
	g_watch_arms++;
#else
	(void)destination;
	(void)size;
#endif
}

bool HandleLodWatchFault(uint64_t fault_vaddr, bool write, uint64_t pc, const uint64_t* gpr16,
                         std::string_view thread_name) {
#ifdef _WIN32
	if (!LodWatchEnabled()) {
		return false;
	}
	std::string code_file;
	std::scoped_lock lock(g_watch_mutex);
	const auto page = fault_vaddr & ~uint64_t {0xfff};
	const bool hit  = std::any_of(g_watched_pages.begin(), g_watched_pages.end(),
	                              [page](const WatchedPage& w) { return w.page == page; });
	if (!hit) {
		return false;
	}
	// One access is enough per arming: restore every page so the guest continues untouched.
	RestoreWatchedPagesLocked();
	if (std::find(g_watch_seen_pcs.begin(), g_watch_seen_pcs.end(), pc) == g_watch_seen_pcs.end() &&
	    g_watch_seen_pcs.size() < 64) {
		g_watch_seen_pcs.push_back(pc);
		const auto code_begin = pc - kLodWatchCodeHalf;
		if (IsExecutablePage(code_begin) && IsExecutablePage(pc) &&
		    IsExecutablePage(pc + kLodWatchCodeHalf - 1)) {
			code_file = fmt::format("lodwatch-code-{:x}.bin", pc);
			if (auto* f = std::fopen((g_dir / code_file).string().c_str(), "wb"); f != nullptr) {
				std::fwrite(reinterpret_cast<const void*>(code_begin), 1, 2 * kLodWatchCodeHalf, f);
				std::fclose(f);
			}
		}
	}
	const bool in_report = fault_vaddr >= g_watch_report_begin && fault_vaddr < g_watch_report_end;
	std::string regs;
	for (int i = 0; i < 16; i++) {
		regs += fmt::format(",0x{:x}", gpr16[i]);
	}
	g_pending_watch_rows.push_back(fmt::format(
	    "{},{},0x{:x},{},{},{},{}{},{},{}", NowMs(), g_watch_report_seq, fault_vaddr,
	    in_report ? static_cast<int64_t>(fault_vaddr - g_watch_report_begin) : int64_t {-1},
	    write ? "write" : "read", FormatAddress(pc), CsvEscape(thread_name), regs, code_file,
	    CsvEscape(CaptureGuestCallers())));
	return true;
#else
	(void)fault_vaddr;
	(void)write;
	(void)pc;
	(void)gpr16;
	(void)thread_name;
	return false;
#endif
}

constinit thread_local Detail::TextureTally* Detail::t_texture_tally = nullptr;

Detail::TextureTally& Detail::AcquireTextureTally() noexcept {
	auto* tally = new TextureTally;
	auto* head  = g_texture_tallies.load(std::memory_order_relaxed);
	do {
		tally->next = head;
	} while (!g_texture_tallies.compare_exchange_weak(head, tally, std::memory_order_release,
	                                                  std::memory_order_relaxed));
	t_texture_tally = tally;
	return *tally;
}

void Detail::RecordMipStatsTexture(const uint32_t* fields) noexcept {
	g_totals.tex_mipstats.fetch_add(1, std::memory_order_relaxed);
	const auto id   = fields[6] & 0xffu;
	const auto base = ((fields[0] | (static_cast<uint64_t>(fields[1]) << 32u)) & 0xFFFFFFFFFFull)
	                  << 8u;
	const auto min_lod      = (fields[1] >> 8u) & 0xfffu;
	const auto min_lod_warn = (fields[5] >> 8u) & 0xfffu;
	const auto base_level   = (fields[3] >> 12u) & 0xfu;
	const auto last_level   = (fields[3] >> 16u) & 0xfu;
	g_tex_count[id].fetch_add(1, std::memory_order_relaxed);
	g_tex_base[id].store(base, std::memory_order_relaxed);
	g_tex_info[id].store(min_lod | (min_lod_warn << 12u) | (base_level << 24u) | (last_level << 28u),
	                     std::memory_order_relaxed);
}

void SetFaultContext(uint64_t pc, std::string_view thread_name) {
	g_fault_context.pc = pc;
	const auto n       = std::min(thread_name.size(), sizeof(g_fault_context.thread) - 1);
	std::memcpy(g_fault_context.thread, thread_name.data(), n);
	g_fault_context.thread[n] = '\0';
}

void ClearFaultContext() {
	g_fault_context.pc        = 0;
	g_fault_context.thread[0] = '\0';
	g_readback_kind           = ReadbackKind::Invalidate;
}

void SetReadbackKind(ReadbackKind kind) {
	g_readback_kind = kind;
}

ReadbackKind GetReadbackKind() {
	return g_readback_kind;
}

void RecordReadback(uint64_t vaddr, uint64_t size, uint64_t window_begin, uint64_t window_size,
                    bool downloaded, uint64_t duration_ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.readbacks.fetch_add(1, std::memory_order_relaxed);
	g_totals.readback_ns.fetch_add(duration_ns, std::memory_order_relaxed);
	if (downloaded) {
		g_totals.readback_downloads.fetch_add(1, std::memory_order_relaxed);
	}
	const auto kind   = g_readback_kind;
	const bool fault  = kind == ReadbackKind::FaultRead || kind == ReadbackKind::FaultWrite ||
	                   kind == ReadbackKind::FaultReadSide || kind == ReadbackKind::FaultReadDuplicate ||
	                   kind == ReadbackKind::FaultReadEager;
	const auto pc     = fault && g_fault_context.pc != 0 ? FormatAddress(g_fault_context.pc) : std::string();
	const auto thread = fault ? std::string_view(g_fault_context.thread) : std::string_view();
	// Stack scanning only helps when a guest thread is on this stack (CPU faults). It costs a dozen
	// VirtualQuery calls, so scan only the first few faults of each faulting instruction.
	thread_local std::unordered_map<uint64_t, uint32_t> scanned_pcs;
	const bool scan    = fault && scanned_pcs.size() < 4096 && scanned_pcs[g_fault_context.pc]++ < 4;
	const auto callers = scan ? CaptureGuestCallers() : std::string();
	const auto now_ms  = NowMs();
	std::string writer = ",,";
	{
		std::scoped_lock lock(g_page_writer_mutex);
		if (auto it = g_page_writers.find(vaddr >> 12u); it != g_page_writers.end()) {
			writer = fmt::format("{},{},{}", kGpuWriteKindNames[static_cast<uint32_t>(it->second.kind)],
			                     now_ms - std::min(now_ms, it->second.t_ms), it->second.size);
		}
	}
	auto row = fmt::format("{},{},0x{:x},{},0x{:x},{},{},{},{},{},{},{},{}", now_ms,
	                       kReadbackKindNames[static_cast<uint32_t>(kind)], vaddr, size, window_begin,
	                       window_size, downloaded ? 1 : 0, duration_ns / 1000u, OsThreadId(),
	                       CsvEscape(thread), pc, CsvEscape(callers), writer);
	std::scoped_lock lock(g_readback_mutex);
	if (g_readback_rows_total >= kReadbackRowLimit) {
		return;
	}
	g_readback_rows_total++;
	g_pending_readback_rows.push_back(std::move(row));
}

void RecordOcclusion(const OcclusionEvent& event) {
	if (!Enabled()) {
		return;
	}
	auto row = fmt::format("{},{},0x{:x},{},{},{},{},{},{},{},0x{:x},{},{},{}", NowMs(), event.event,
	                       event.address, event.value, event.scopes, event.width, event.height,
	                       event.colors, event.has_depth ? 1 : 0, event.depth_format,
	                       event.depth_address, event.condition, event.skip ? 1 : 0,
	                       CsvEscape(event.detail));
	std::scoped_lock lock(g_occlusion_mutex);
	if (g_occlusion_rows_total >= kOcclusionRowLimit) {
		return;
	}
	g_occlusion_rows_total++;
	g_pending_occlusion_rows.push_back(std::move(row));
}

void RecordLodReport(const LodReportEvent& event) {
	if (!Enabled()) {
		return;
	}
	// Completion rows keep the UINT64_MAX pending_copies marker of the U33 format.
	const auto pending =
	    event.kind == LodReportKind::Completion ? ~uint64_t {0} : event.pending_copies;
	auto row = fmt::format("{},0x{:x},0x{:x},{},{},{},{:.2f},{},{},{}", NowMs(), event.destination,
	                       event.control, event.has_latest ? 1 : 0, event.sampled_counters,
	                       event.total_samples, event.mean_finest_mip, pending,
	                       event.drawn_counters, event.counted_counters);
	std::scoped_lock lock(g_occlusion_mutex);
	if (g_lodreport_rows_total >= kOcclusionRowLimit) {
		return;
	}
	g_lodreport_rows_total++;
	g_pending_lodreport_rows.push_back(std::move(row));
}

ScopedGpuWriteKind::ScopedGpuWriteKind(GpuWriteKind kind) : m_previous(g_gpu_write_kind) {
	g_gpu_write_kind = kind;
}

ScopedGpuWriteKind::~ScopedGpuWriteKind() {
	g_gpu_write_kind = m_previous;
}

void NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	if (!Enabled() || size == 0) {
		return;
	}
	const PageWriter writer {NowMs(), size, g_gpu_write_kind};
	const auto       first = vaddr >> 12u;
	const auto       last  = (vaddr + size - 1) >> 12u;
	// The same range again (every draw notes its writable bindings) within the same millisecond:
	// if the map has not changed since this thread's last update, it already holds exactly these
	// values for these pages, so the update would change nothing. Not below the clearing size:
	// that update would clear the map.
	if (t_recent_below_clear &&
	    g_page_writer_version.load(std::memory_order_acquire) == t_recent_version) {
		for (const auto& recent: t_recent_writes) {
			if (recent.valid && recent.first == first && recent.last == last &&
			    recent.t_ms == writer.t_ms && recent.size == size && recent.kind == writer.kind) {
				return;
			}
		}
	}
	std::scoped_lock lock(g_page_writer_mutex);
	const bool       cleared = g_page_writers.size() > kClearPageWriters;
	if (cleared) {
		g_page_writers.clear();
	}
	// Large writable bindings: note only their first and last pages (sync pages are small).
	if (last - first > 64) {
		g_page_writers[first] = writer;
		g_page_writers[last]  = writer;
	} else {
		for (auto page = first; page <= last; page++) {
			g_page_writers[page] = writer;
		}
	}
	const auto version = g_page_writer_version.load(std::memory_order_relaxed) + 1;
	g_page_writer_version.store(version, std::memory_order_release);
	for (auto& recent: t_recent_writes) {
		// This update cleared the map or overwrote some of the range's pages, or another thread
		// changed the map since this thread's last update.
		if (cleared || t_recent_version + 1 != version ||
		    (recent.first <= last && first <= recent.last)) {
			recent.valid = false;
		}
	}
	t_recent_writes[t_recent_next++ % t_recent_writes.size()] = {first, last, writer.t_ms, size,
	                                                             writer.kind, true};
	t_recent_version     = version;
	t_recent_below_clear = g_page_writers.size() <= kClearPageWriters;
}

GpuPageWriter PageWriterForTest(uint64_t vaddr) {
	std::scoped_lock lock(g_page_writer_mutex);
	const auto       it = g_page_writers.find(vaddr >> 12u);
	if (it == g_page_writers.end()) {
		return {};
	}
	return {true, it->second.t_ms, it->second.size, it->second.kind};
}

const char* LastGpuWriteKind(uint64_t vaddr) {
	if (!Enabled()) {
		return "unknown";
	}
	std::scoped_lock lock(g_page_writer_mutex);
	const auto it = g_page_writers.find(vaddr >> 12u);
	return it != g_page_writers.end() ? kGpuWriteKindNames[static_cast<uint32_t>(it->second.kind)]
	                                  : "unknown";
}

void SetImageFreeReason(ImageFreeReason reason) {
	g_image_free_reason = reason;
}

void RecordImageFree(uint64_t address, uint32_t width, uint32_t height, uint32_t levels,
                     uint32_t layers, uint32_t format, uint64_t bytes, uint64_t age_ticks) {
	if (!Enabled()) {
		return;
	}
	g_totals.image_frees.fetch_add(1, std::memory_order_relaxed);
	const auto reason = g_image_free_reason;
	auto row = fmt::format("{},free,0x{:x},{},{},{},{},{},{},{},,1,{}", NowMs(), address, age_ticks,
	                       width, height, levels, layers,
	                       kImageFreeReasonNames[static_cast<uint32_t>(reason)], format, bytes);
	std::scoped_lock lock(g_image_mutex);
	if (g_image_rows_total >= kImageRowLimit) {
		return;
	}
	g_image_rows_total++;
	g_pending_image_rows.push_back(std::move(row));
}

void RecordNativeImage(bool create, bool pool_hit, uint32_t format, uint32_t width, uint32_t height,
                       uint32_t levels, uint32_t usage, uint64_t bytes) {
	if (!Enabled()) {
		return;
	}
	if (create) {
		g_totals.native_creates.fetch_add(1, std::memory_order_relaxed);
		g_totals.native_create_bytes.fetch_add(bytes, std::memory_order_relaxed);
	}
	std::scoped_lock lock(g_image_mutex);
	auto& totals = g_native_images[NativeImageKey {create, pool_hit, format, width, height, levels, usage}];
	totals.count++;
	totals.bytes += bytes;
}

void RecordTransfer(TransferKind kind, const char* reason, const char* detail, uint64_t address,
                    uint32_t format, uint32_t width, uint32_t height, uint64_t bytes,
                    uint64_t span_bytes) {
	if (!Enabled() || kind >= TransferKind::Count) {
		return;
	}
	g_totals.transfer_count[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
	g_totals.transfer_bytes[static_cast<size_t>(kind)].fetch_add(bytes, std::memory_order_relaxed);
	TransferKey key {kind,   reason != nullptr ? reason : "", detail != nullptr ? detail : "",
	                 address, format, width, height};
	std::scoped_lock lock(g_transfer_mutex);
	if (g_transfers.size() >= kTransferKeyLimit && !g_transfers.contains(key)) {
		key.address = 0;
		key.format  = 0;
		key.width   = 0;
		key.height  = 0;
	}
	auto& totals = g_transfers[key];
	totals.count++;
	totals.bytes += bytes;
	totals.span_bytes += span_bytes;
}

void RecordUncleanRead(uint64_t address, uint64_t size, const char* reason, const char* purpose,
                       uint64_t caller) {
	if (!Enabled()) {
		return;
	}
	UncleanKey key {reason != nullptr ? reason : "", purpose, caller, address & ~uint64_t {0xfff}};
	std::scoped_lock lock(g_unclean_mutex);
	if (g_unclean_reads.size() >= kUncleanKeyLimit && !g_unclean_reads.contains(key)) {
		key.page = 0;
	}
	auto& totals = g_unclean_reads[key];
	if (totals.count++ == 0) {
		totals.first_address = address;
		totals.first_size    = size;
	}
	totals.bytes += size;
}

void RecordQueueWait(uint32_t queue, uint64_t wait_ns) {
	if (!Enabled() || queue >= kMaxQueues) {
		return;
	}
	auto& q = g_queues[queue];
	q.starts.fetch_add(1, std::memory_order_relaxed);
	q.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
	UpdateMax(q.wait_max_ns, wait_ns);
}

void RecordQueueBusy(uint32_t queue, uint64_t busy_ns, bool complete) {
	if (!Enabled() || queue >= kMaxQueues) {
		return;
	}
	auto& q = g_queues[queue];
	q.busy_ns.fetch_add(busy_ns, std::memory_order_relaxed);
	q.slices.fetch_add(1, std::memory_order_relaxed);
	if (!complete) {
		q.incomplete.fetch_add(1, std::memory_order_relaxed);
	}
}

void RecordDoneWait(uint64_t wait_ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.done_waits.fetch_add(1, std::memory_order_relaxed);
	g_totals.done_ns.fetch_add(wait_ns, std::memory_order_relaxed);
	UpdateMax(g_totals.done_max_ns, wait_ns);
}

void RecordFlip() {
	HangWatchdog::NoteFlip();
	if (!Enabled()) {
		return;
	}
	g_totals.flips.fetch_add(1, std::memory_order_relaxed);
	UpdateMax(g_totals.apr_shrink_max_per_flip,
	          g_totals.apr_shrinks_since_flip.exchange(0, std::memory_order_relaxed));
	if (FrameTimesEnabled()) {
		std::scoped_lock lock(g_compile_mutex);
		const auto now = NowNs();
		if (g_previous_flip_ns != 0 && g_frame_rows_total < kCompileRowLimit) {
			g_pending_frame_rows.push_back(fmt::format("{},{}", now / 1000u,
			    (now - g_previous_flip_ns) / 1000u));
			++g_frame_rows_total;
		}
		g_previous_flip_ns = now;
	}
}

void RecordGpuFrame(const GpuFrame& frame) {
	if (!Enabled()) {
		return;
	}
	g_totals.gpu_busy_ns.fetch_add(frame.busy_ns, std::memory_order_relaxed);
	g_totals.gpu_idle_ns.fetch_add(frame.idle_ns, std::memory_order_relaxed);
	UpdateMax(g_totals.gpu_max_gap_ns, frame.max_gap_ns);
	g_totals.gpu_starved_ns.fetch_add(frame.starved_ns, std::memory_order_relaxed);
	g_totals.gpu_cmdbufs.fetch_add(frame.command_buffers, std::memory_order_relaxed);
	g_totals.gpu_latency_samples.fetch_add(frame.latency_samples, std::memory_order_relaxed);
	g_totals.gpu_record_latency_ns.fetch_add(frame.record_latency_ns, std::memory_order_relaxed);
	g_totals.gpu_dispatch_latency_ns.fetch_add(frame.dispatch_latency_ns,
	                                           std::memory_order_relaxed);
	g_totals.gpu_dropped.fetch_add(frame.dropped, std::memory_order_relaxed);
}

void RecordEopTimestamps(const EopTimestampFrame& frame) {
	if (!Enabled()) {
		return;
	}
	const auto samples = frame.rewritten + frame.deferred;
	// Reference-clock ticks are 10 ns.
	const double shift_avg_us =
	    samples != 0 ? static_cast<double>(frame.shift_sum) / static_cast<double>(samples) / 100.0
	                 : 0.0;
	std::string row = fmt::format("{},{},{},{},{},{:.1f},{:.1f},{:.1f},{}", NowMs(), frame.rewritten,
	                              frame.skipped, frame.unavailable, frame.deferred, shift_avg_us,
	                              static_cast<double>(frame.shift_max) / 100.0,
	                              static_cast<double>(frame.publish_ns) / 1000.0, frame.publishes);
	for (uint32_t i = 0; i < EopTimestampFrame::MaxRings; i++) {
		row += i < frame.rings && frame.ring_delta_us[i] >= 0
		           ? fmt::format(",{}", frame.ring_delta_us[i])
		           : std::string(",");
	}
	if (frame.drs_valid) {
		row += fmt::format(",{},{},{},{},{:.3f},{:.3f}", frame.drs_index, frame.drs_level,
		                   frame.drs_room_frames, frame.drs_fps, frame.drs_scalable_ms,
		                   frame.drs_total_ms);
	} else {
		row += ",,,,,,";
	}
	std::scoped_lock lock(g_timestamp_mutex);
	g_pending_timestamp_rows.push_back(std::move(row));
}

void NotePendingOperations(uint64_t depth) {
	if (!Enabled()) {
		return;
	}
	UpdateMax(g_totals.pending_ops_max, depth);
}

bool ModuleBase(std::string_view name, uint64_t& base) {
	const auto count = g_module_count.load(std::memory_order_acquire);
	for (uint32_t i = 0; i < count; i++) {
		if (g_modules[i].name == name) {
			base = g_modules[i].base;
			return true;
		}
	}
	return false;
}

bool TryReadReadable(uint64_t address, void* data, size_t size) {
#ifdef _WIN32
	if (size == 0 || UINT64_MAX - address < size) {
		return false;
	}
	// A kernel copy: a page that is (or becomes) no-access, e.g. protected by resource tracking
	// meanwhile, fails the call instead of faulting on this thread.
	SIZE_T copied = 0;
	return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), data,
	                         size, &copied) != 0 &&
	       copied == size;
#else
	(void)address;
	(void)data;
	(void)size;
	return false;
#endif
}

void CountMemory(MemoryCounter counter, uint64_t amount) {
	if (!Enabled() || counter >= MemoryCounter::Count) {
		return;
	}
	g_totals.memory[static_cast<size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
}

void RecordPredicationFlushWait(uint64_t ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.pred_flush_waits.fetch_add(1, std::memory_order_relaxed);
	g_totals.pred_flush_wait_ns.fetch_add(ns, std::memory_order_relaxed);
}

void RecordGpuOpCounts(const GpuOpCounts& counts) {
	if (!Enabled()) {
		return;
	}
	g_totals.gpu_render_passes.fetch_add(counts.render_passes, std::memory_order_relaxed);
	g_totals.gpu_barriers.fetch_add(counts.barriers, std::memory_order_relaxed);
	g_totals.gpu_layout_transitions.fetch_add(counts.layout_transitions,
	                                          std::memory_order_relaxed);
	g_totals.gpu_guest_cmdbufs.fetch_add(counts.command_buffers, std::memory_order_relaxed);
	g_totals.gpu_barrier_requests.fetch_add(counts.barrier_requests, std::memory_order_relaxed);
	g_totals.gpu_barriers_merged.fetch_add(counts.barriers_merged, std::memory_order_relaxed);
	g_totals.gpu_barriers_elided.fetch_add(counts.barriers_elided, std::memory_order_relaxed);
	g_totals.gpu_barriers_sunk.fetch_add(counts.barriers_sunk, std::memory_order_relaxed);
	g_totals.gpu_barrier_rp_splits.fetch_add(counts.barrier_render_splits,
	                                         std::memory_order_relaxed);
	g_totals.gpu_rendering_ends.fetch_add(counts.rendering_ends, std::memory_order_relaxed);
	g_totals.gpu_draw_write_sinks.fetch_add(counts.draw_write_sinks, std::memory_order_relaxed);
}

void RecordCompile(const CompileEvent& event) {
	if (!Enabled() || event.kind >= CompileKind::Count) {
		return;
	}
	const auto add = [](std::atomic<uint64_t>& total, uint64_t value) {
		total.fetch_add(value, std::memory_order_relaxed);
	};
	switch (event.kind) {
		case CompileKind::Program:
			add(g_totals.compile_programs, 1);
			add(g_totals.compile_translate_ns, event.translate_ns);
			add(g_totals.compile_clone_ns, event.clone_ns);
			if (event.reused) add(g_totals.compile_translation_reuses, 1);
			if (event.from_disk) add(g_totals.compile_disk_loads, 1);
			add(g_totals.compile_disk_load_ns, event.load_ns);
			add(g_totals.compile_emit_ns, event.emit_ns);
			add(g_totals.compile_validate_ns, event.validate_ns);
			add(g_totals.compile_module_ns, event.module_ns);
			break;
		case CompileKind::GraphicsPipeline:
			add(g_totals.compile_gfx_pipelines, 1);
			add(g_totals.compile_gfx_pipeline_ns, event.pipeline_ns);
			switch (event.origin) {
				case PipelineOrigin::New: add(g_totals.compile_gfx_new, 1); break;
				case PipelineOrigin::Permutation: add(g_totals.compile_gfx_perm, 1); break;
				case PipelineOrigin::Variant: add(g_totals.compile_gfx_variant, 1); break;
				default: break;
			}
			break;
		case CompileKind::ComputePipeline:
			add(g_totals.compile_cs_pipelines, 1);
			add(g_totals.compile_cs_pipeline_ns, event.pipeline_ns);
			break;
		default: break;
	}
	const auto origin = static_cast<size_t>(event.origin) < std::size(kPipelineOriginNames)
	                        ? kPipelineOriginNames[static_cast<size_t>(event.origin)]
	                        : "";
	auto row = fmt::format("{},{},{},0x{:016x},{},{},{},{},{},{},{},{},{},{},{},{},{},{}", NowMs(),
	                       kCompileKindNames[static_cast<size_t>(event.kind)],
	                       event.stage != nullptr ? event.stage : "", event.guest_hash, event.id,
	                       event.id2, origin, event.translate_ns / 1000u, event.emit_ns / 1000u,
	                       event.validate_ns / 1000u, event.module_ns / 1000u,
	                       event.pipeline_ns / 1000u, event.total_ns / 1000u, event.spirv_words,
	                       OsThreadId(), CsvEscape(event.detail), event.clone_ns / 1000u,
	                       event.load_ns / 1000u);
	std::scoped_lock lock(g_compile_mutex);
	if (g_compile_rows_total >= kCompileRowLimit) {
		return;
	}
	g_compile_rows_total++;
	g_pending_compile_rows.push_back(std::move(row));
}

void RecordCompileStall(uint64_t stall_ns) {
	if (!Enabled() || stall_ns == 0) {
		return;
	}
	g_totals.compile_stall_ns.fetch_add(stall_ns, std::memory_order_relaxed);
	UpdateMax(g_totals.compile_stall_max_ns, stall_ns);
}

void RecordPipelineCacheSave(uint64_t bytes, uint64_t serialize_ns, uint64_t write_ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.pcache_saves.fetch_add(1, std::memory_order_relaxed);
	g_totals.pcache_save_bytes.fetch_add(bytes, std::memory_order_relaxed);
	g_totals.pcache_save_serialize_ns.fetch_add(serialize_ns, std::memory_order_relaxed);
	g_totals.pcache_save_write_ns.fetch_add(write_ns, std::memory_order_relaxed);
}

void RecordPipelineLibraryEvent(PipelineLibraryEvent event, uint64_t ns, uint64_t count) {
	if (!Enabled() || event >= PipelineLibraryEvent::Count) {
		return;
	}
	const auto index = static_cast<size_t>(event);
	g_totals.gpl_count[index].fetch_add(count, std::memory_order_relaxed);
	g_totals.gpl_ns[index].fetch_add(ns, std::memory_order_relaxed);
}

void RecordShaderValidation(uint64_t validate_ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.validate_async.fetch_add(1, std::memory_order_relaxed);
	g_totals.validate_async_ns.fetch_add(validate_ns, std::memory_order_relaxed);
}

void RecordPipelineCacheSaveOverlap(uint64_t create_ns) {
	if (!Enabled()) {
		return;
	}
	g_totals.pcache_save_overlaps.fetch_add(1, std::memory_order_relaxed);
	g_totals.pcache_save_overlap_ns.fetch_add(create_ns, std::memory_order_relaxed);
}

} // namespace HangTrace
