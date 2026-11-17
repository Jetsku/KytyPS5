#include "loader/systemContent.h"
#include "common/staticOnce.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/watchRange.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/cruzStats.h"
#include "common/guestLoad.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/liveSwitch.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/faultCost.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/host_gpu/gpuBreadcrumbs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/readFaultDiag.h"
#include "graphics/host_gpu/renderer/cache/bufferLookupStats.h"
#include "graphics/host_gpu/renderer/cache/inplaceHeap.h"
#include "graphics/host_gpu/renderer/cache/inplaceStats.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/cache/uploadDma.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/vramBudget.h"
#include "graphics/host_gpu/vramStats.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/watchdogSubmit.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

namespace Libs::Graphics {

namespace {

Live::Switch g_cpu_copy_page_skip("KYTY_CPU_COPY_PAGE_SKIP", Live::ParseDefaultOff);
Live::Switch g_shader_write_retick("KYTY_SHADER_WRITE_RETICK", Live::ParseDefaultOff);

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

bool SmallUploadRingEnabled() {
	// Startup only: keep large requests correct through temporary upload buffers.
	const auto* flag = std::getenv("KYTY_RAM_SMALL_UPLOAD_RING");
	return flag != nullptr && std::strcmp(flag, "1") == 0;
}

uint64_t UploadRingSize() {
	return SmallUploadRingEnabled() ? 128 * MiB : 512 * MiB;
}

Live::Switch g_cpu_only_query("KYTY_CP_CPU_ONLY_QUERY", Live::ParseDefaultOff);
Live::Switch g_binding_memo_prefetch("KYTY_CP_BINDING_MEMO_PREFETCH", Live::ParseDefaultOff);
// KYTY_CP_BINDING_BATCH_PREFETCH (BufferCache::PrefetchReadBinding): 0 off, find (2) from
// FindBuffers, anything else (default) from RebindBuffers.
int64_t ParseBatchPrefetch(const char* value) {
	if (value != nullptr && std::strcmp(value, "find") == 0) {
		return 2;
	}
	return Live::ParseDefaultOn(value);
}
Live::Switch g_binding_batch_prefetch("KYTY_CP_BINDING_BATCH_PREFETCH", ParseBatchPrefetch);
std::atomic<uint64_t> g_binding_hot_generation {1};
void BindingHotChanged(int64_t, int64_t) {
	// The live registry serializes callbacks. Saturation makes lookups clear the tier each time.
	const auto generation = g_binding_hot_generation.load(std::memory_order_relaxed);
	if (generation != UINT64_MAX) {
		g_binding_hot_generation.store(generation + 1, std::memory_order_relaxed);
	}
}
Live::Switch g_binding_hot_memo("KYTY_CP_BINDING_HOT_MEMO", Live::ParseDefaultOff,
                               BindingHotChanged);
// 0 when unset, empty or "0"; 2 for "verify", 3 for "exit"; 1 otherwise.
int64_t ParseOnVerifyExit(const char* value) {
	if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
}
// KYTY_BINDING_MEMO_TOUCH (BufferCache::BindingMemo).
Live::Switch g_binding_memo_touch("KYTY_BINDING_MEMO_TOUCH", Live::ParseDefaultOff);
// KYTY_BINDING_LEAN_MISS (BufferCache::TryLeanReadMiss): 0 (default) off, 1 on, verify (2), exit (3).
Live::Switch g_binding_lean_miss("KYTY_BINDING_LEAN_MISS", ParseOnVerifyExit);
// KYTY_BINDING_CROSS_SETTLED (ObtainReadBindingMemo): 0 (default) off, 1 on, verify (2).
Live::Switch g_binding_cross_settled("KYTY_BINDING_CROSS_SETTLED", ParseOnVerifyExit);

// KYTY_FAULT_AHEAD_ADAPT (live): the fault-ahead window of write faults
// (MemoryTracker::SetFaultAheadOverride, applied at every guest flip).
//   auto   (default) 256 KiB, 512 KiB or 1 MiB by FaultCost::SlowLevel() 0 / 1 / 2: how slow
//          protection changes are on this PC (measured, only ever rising; Linux with mprotect
//          tracking: 1 MiB from the start)
//   <KiB>  a power of two, 8..4096
//   0      KYTY_FAULT_AHEAD_KB alone (32 KiB), as before
// At the Sky Garden the guest's job threads fill ~16 MB of triple-buffered data per frame. With
// 32 KiB windows that took ~1,700 faults per frame, 62% of them duplicates (another thread wrote
// the same page while the first fault was in flight). 256 KiB windows: ~300 faults and ~100
// loosening calls per frame instead of ~1,700 and ~620, for ~14% more upload bytes (pages a window
// opened that the guest did not write are uploaded once more than needed: bytes, not faults).
// 1 MiB here: ~110 faults, but ~72 us each in the handler and twice the upload bytes of 32 KiB, so
// only PCs with slow protection changes get it (the slow-PC simulation: 20.7 -> 26.4 fps).
// The handler's own time grows with the window, so it must not choose the window: an earlier rule
// based on it climbed to 1 MiB on this PC too.
Live::Switch g_fault_ahead_adapt("KYTY_FAULT_AHEAD_ADAPT", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
		return -2;
	}
	if (std::strcmp(value, "0") == 0) {
		return 0;
	}
	char*      end = nullptr;
	const auto kib = std::strtoll(value, &end, 10);
	if (end == value || kib < 8 || kib > 4096 || (kib & (kib - 1)) != 0) {
		return 0;
	}
	return kib;
});

// KYTY_HOT_PAGE_MAX_LIVE=<pages> / KYTY_HOT_PAGE_FRAMES_LIVE=<frames> (live, default 0 = the
// startup KYTY_HOT_PAGE_MAX / KYTY_HOT_PAGE_FRAMES): see MemoryTracker::SetHotOverride. Applied
// once per guest flip. GT7 races re-fault ~800 pages a frame in the frame they were uploaded.
Live::Switch g_hot_max_live("KYTY_HOT_PAGE_MAX_LIVE", [](const char* value) -> int64_t {
	return value == nullptr ? 0 : static_cast<int64_t>(std::min<uint64_t>(std::strtoull(value, nullptr, 10), 65536));
});
Live::Switch g_hot_frames_live("KYTY_HOT_PAGE_FRAMES_LIVE", [](const char* value) -> int64_t {
	return value == nullptr ? 0 : static_cast<int64_t>(std::min<uint64_t>(std::strtoull(value, nullptr, 10), 255));
});

// KYTY_FAULT_AHEAD_SEQUENTIAL=1 (default off; live): the fault-ahead window (KYTY_FAULT_AHEAD_KB /
// KYTY_FAULT_AHEAD_ADAPT size) opens only when the page below the faulting page is CPU-dirty already
// (a sequential fill), forward from the faulting page; any other write fault makes only its own
// pages CPU-dirty (FaultPolicy::sequential_ahead). Fewer pages uploaded that the guest did not
// write, possibly more faults. Adapted from TheCruZ/KytyPS5-GTA InvalidateFaultedPage.
Live::Switch g_fault_ahead_sequential("KYTY_FAULT_AHEAD_SEQUENTIAL", Live::ParseDefaultOff);

uint32_t FaultAheadOverridePages() {
	const auto value = g_fault_ahead_adapt.Get();
	if (value == 0) {
		return 0;
	}
	uint64_t kib = static_cast<uint64_t>(value);
	if (value == -2) {
		kib = uint64_t {256} << std::clamp(FaultCost::SlowLevel(), 0, 2);
	}
	return static_cast<uint32_t>(kib * 1024 / TRACKER_PAGE_SIZE);
}

// KYTY_BDA_SYNC_PER_SUBMISSION=1|verify (default off; live): see BufferCache::SynchronizeBdaBuffers.
// 1: at most one BDA pass per guest submission while the BDA structure holds. verify: every pass the
// gate would skip runs anyway and its uploads are counted, i.e. the bytes the CPU wrote within the
// submission that 1 leaves for the next submission's pass. Both values only decide whether a pass
// runs now, and every pass records the epochs both read, so it can switch at any flip.
int64_t ParseBdaSyncPerSubmission(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_bda_sync_per_submission("KYTY_BDA_SYNC_PER_SUBMISSION", ParseBdaSyncPerSubmission);

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
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_UPLOAD_BATCH_SCOPED_FLUSH");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
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

// Aligned side-copy window in bytes: a power of two between 4 KiB and 4 MiB (default 4 MiB, was
// 64 KiB). It is also the staging stride of each side-readback slot (16 slots: 64 MiB of download
// memory), so the live window below never exceeds it. Miles Morales' game threads poll
// GPU-written bytes across neighbouring pages: one launch, 5 ABBA quartets at the "Previously on"
// prompt, 4 MiB vs 64 KiB: +17.8% flips/s (SE 4.2%), CP time per flip unchanged (+2.5 ms, SE 4.6),
// guest read faults ~690 -> ~510 per frame.
// Default 64 KiB in this line (as int21); the Insomniac title presets set 4096.
uint64_t SideReadbackWindow() {
	constexpr uint64_t Default = 64 * 1024;
	const auto*        value   = std::getenv("KYTY_READBACK_SIDE_COPY_WINDOW_KB");
	if (value == nullptr) {
		return Default;
	}
	char*      end = nullptr;
	const auto kib = std::strtoull(value, &end, 10);
	if (end == value || *end != '\0' || kib < 4 || kib > 4096 || (kib & (kib - 1)) != 0) {
		return Default;
	}
	return kib * 1024;
}

// KYTY_READBACK_SIDE_WINDOW_LIVE_KB (live; unset or 0 = KYTY_READBACK_SIDE_COPY_WINDOW_KB): the
// aligned window a side readback copies around a fault, a power of two of at least 4 KiB, capped
// at the startup window (the staging stride). A wider window serves the neighbouring GPU-written
// bytes of a polled page with the same copy, so fewer guest reads fault and wait (port of senaxx
// 2559c2737, which widened its drain window from 512 KiB to 4 MiB for the same engine's polling).
Live::Switch g_side_window_live("KYTY_READBACK_SIDE_WINDOW_LIVE_KB", [](const char* value) -> int64_t {
	if (value == nullptr) {
		return 0;
	}
	const auto kib = std::strtoull(value, nullptr, 10);
	if (kib < 4 || kib > 4096 || (kib & (kib - 1)) != 0) {
		return 0;
	}
	return static_cast<int64_t>(kib * 1024);
});

// KYTY_READBACK_MERGE_GAP_KB=<KiB> (live, default 0 = one region per range; 64 was the fork's): GPU-written ranges
// of one readback less than this far apart share one vkCmdCopyBuffer region (chenxiao07 9239c5773).
// The bytes between them are copied too but never written back: write-backs and publications keep
// the exact ranges. Default off: Astro Bot measured 14-18 readback commands of 20-32 ranges per flip
// (Sky Garden, snow, clock tower), which a 64 KiB gap cut to 14-20 regions while doubling the bytes
// copied at snow/clock tower (85 -> 176 KiB per flip); ours was one command per readback already.
Live::Switch g_readback_merge_gap("KYTY_READBACK_MERGE_GAP_KB", [](const char* value) -> int64_t {
	if (value == nullptr) {
		return 0;
	}
	return static_cast<int64_t>(std::min<uint64_t>(std::strtoull(value, nullptr, 10), 64 * 1024)) *
	       1024;
});

// KYTY_READBACK_FLUSH_SIDE (live, default 0; =1 on): a guest-thread read of
// GPU-written bytes that the unsubmitted recording wrote (or that an address-writing shader in it
// may have written) is served by a side copy instead of a drain. The command processor submits the
// recording (the drain submits it as well), which makes those bytes ones an already submitted
// recording wrote, issues the side copy that waits for it on the GPU, and returns; the guest thread
// then waits for the copy's publication, as for any side readback, while the command processor
// continues. The bytes, their publication and the protection rules are the side path's; only the
// command processor no longer waits for the GPU to finish everything queued before it serves the
// next op.
Live::Switch g_readback_flush_side("KYTY_READBACK_FLUSH_SIDE", Live::ParseDefaultOff);

// KYTY_READBACK_SPAN (live, default 0; =1 on; chenxiao07/KytyPS5 0a0f59ac6, adapted): a side
// readback's aligned window also takes the registered host buffers next to the faulting one, up to
// the aligned window (KYTY_READBACK_SIDE_COPY_WINDOW_KB), instead of stopping at the faulting
// buffer's ends. Where the guest reads back a region made of many small (16 KiB caching page)
// buffers, each buffer otherwise costs its own fault, side copy and wait. The copies are split at
// buffer ends, a barrier and a copy command per source buffer. chenxiao measured Demon's Souls'
// Boletaria walk -0.76 ms a frame (guest readbacks 51 -> 19 a frame, read faults 120 -> 89).
Live::Switch g_readback_span("KYTY_READBACK_SPAN", Live::ParseDefaultOff);
std::atomic<uint64_t> g_readback_span_copies {0};
std::atomic<uint64_t> g_readback_span_drains {0}; // ReadMemoryDrain windows that took neighbours
std::atomic<uint64_t> g_readback_flush_side_copies {0};

// KYTY_READBACK_SIDE_WRITES (live, default on; =0 drains as before): a write fault on a GPU-owned
// page, taken by the exception handler on a thread other than the command processor (so that the
// faulting instruction runs again), takes the side path as well, instead of a drain. The command
// processor issues the side copy of the page's GPU-written bytes (submitting the current recording
// first when it wrote them, as KYTY_READBACK_FLUSH_SIDE does for reads) and returns; the faulting
// thread waits for the copy's publication, which leaves the page clean and write-protected, and
// returns from the fault without the CPU-dirty transition. The write then faults once more, now on
// a clean page, and that fault makes the page CPU-dirty and writable on the faulting thread, as any
// write fault does (a GPU writer recorded meanwhile makes it GPU-owned again: the second fault
// starts over). The drain made the page CPU-dirty on the command processor right after waiting for
// the GPU; the bytes the write sees and the order against newer writers are the same. Other
// CPU-dirty transitions (emulator writes reported through InvalidateMemory, direct HandleFault
// calls) still drain: nothing faults again after them.
Live::Switch g_readback_side_writes("KYTY_READBACK_SIDE_WRITES", Live::ParseDefaultOn);

// KYTY_FAULT_PAGE_LIFT (live, default off; =1 on): see BufferCache::TryLiftFaultPages.
Live::Switch g_fault_page_lift("KYTY_FAULT_PAGE_LIFT", Live::ParseDefaultOff);

// KYTY_ASYNC_DRAIN counts (BufferCache::AsyncDrainEnabled).
namespace {
struct AsyncDrainStats {
	std::atomic<uint64_t> issued {0};
	std::atomic<uint64_t> released {0};
	std::atomic<uint64_t> redrained {0};
	std::atomic<uint64_t> wait_ns {0};
};
AsyncDrainStats g_async_drain;

void ReportAsyncDrain() {
	static std::atomic<uint64_t> last_ns {0};
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	auto previous = last_ns.load(std::memory_order_relaxed);
	if (previous == 0) {
		last_ns.compare_exchange_strong(previous, now, std::memory_order_relaxed);
		return;
	}
	if (now - previous < 30'000'000'000ull ||
	    !last_ns.compare_exchange_strong(previous, now, std::memory_order_relaxed)) {
		return;
	}
	const auto issued    = g_async_drain.issued.exchange(0, std::memory_order_relaxed);
	const auto released  = g_async_drain.released.exchange(0, std::memory_order_relaxed);
	const auto redrained = g_async_drain.redrained.exchange(0, std::memory_order_relaxed);
	const auto wait_ns   = g_async_drain.wait_ns.exchange(0, std::memory_order_relaxed);
	std::printf("Async drain 30s: %" PRIu64 " guest drains waited on their own thread (%.0f us "
	            "each), %" PRIu64 " windows released, %" PRIu64
	            " drained again on the command processor (KYTY_ASYNC_DRAIN)\n",
	            issued, issued != 0 ? static_cast<double>(wait_ns) / 1e3 / static_cast<double>(issued) : 0.0,
	            released, redrained);
}
} // namespace

// KYTY_READBACK_EAGER_WRITES (live, default off; =1 on): a guest write fault on GPU-owned bytes
// makes its pages eager-hot as a read fault does (KYTY_READBACK_EAGER). The next recordings that
// write them then end with a copy of their bytes, published when the recording completes, so a
// later write fault finds the page clean (an ordinary write fault, no command-processor round
// trip and no wait for the GPU) or waits only for that copy. GT7 race: per-frame arenas the GPU
// writes and the job threads write a frame later (FaultMap "flushes"). With it, raise
// KYTY_READBACK_EAGER_PAGES (default 64; up to 2048).
Live::Switch g_readback_eager_writes("KYTY_READBACK_EAGER_WRITES", Live::ParseDefaultOff);
std::atomic<uint64_t> g_readback_side_writes_copies {0};

// KYTY_READ_FAULT_PUBLISHED (live, default off; =1 on, =verify): a guest thread's read fault on a
// page whose bytes the GPU owns does not wait for the GPU. When the faulting instruction is a
// plain load (common/guestLoad.h) of bytes within one page, no image owns the page, and a side or
// eager readback published the page at most KYTY_READ_FAULT_PUBLISHED_FRAMES guest frames ago
// (default 2), the load completes in the handler with the page's backing bytes: the GPU-written
// bytes as last published, the other bytes as they are. The page stays protected (no access) for
// everyone else, so the command processor and the emulator still synchronize through their own
// faults and later writes still fault; the read marks the page read-hot, so its eager copy
// publishes the new bytes once their writer completes. Other reads take the waiting path. This is
// what a CPU sees that reads GPU-written memory without waiting for the writer: legitimate only for
// reads the game does not order after the writer (with KYTY_LABEL_MODE=record, Kyty cannot tell,
// since end-of-pipe labels are written when recorded). Astro Bot's render threads read a GPU-written
// value this way every frame (eboot 0x736fa9e).
// =verify takes the waiting path for every read, and compares the bytes the load would have been
// served with those it finds after the wait (equal / different counts, the first differences).
// One "Kyty read-fault published" line every 10 s while on.
int64_t ParseReadFaultPublished(const char* value) {
	if (value == nullptr || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_read_fault_published("KYTY_READ_FAULT_PUBLISHED", ParseReadFaultPublished);

uint32_t ReadFaultPublishedFrames() {
	static const uint32_t frames = [] {
		const auto* value = std::getenv("KYTY_READ_FAULT_PUBLISHED_FRAMES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 2u;
	}();
	return frames;
}

struct ReadFaultPublishedState {
	std::mutex                             mutex;
	std::unordered_map<uint64_t, uint32_t> published_frame; // tracker page -> guest frame
	std::atomic<uint64_t>                  served {0};
	std::atomic<uint64_t>                  undecodable {0};
	std::atomic<uint64_t>                  image {0};
	std::atomic<uint64_t>                  stale {0};
	std::atomic<uint64_t>                  other {0};
	std::atomic<uint64_t>                  burst {0};
	std::atomic<uint64_t>                  verify_equal {0};
	std::atomic<uint64_t>                  verify_different {0};
	std::atomic<uint64_t>                  report_ns {0};
	std::atomic<uint32_t>                  report_frame {0};
	std::atomic<uint32_t>                  logged {0};
};
ReadFaultPublishedState& ReadFaultPublished() {
	static ReadFaultPublishedState state;
	return state;
}

// While the switch is on: the guest frame in which a readback last published each page of `ranges`.
void NoteReadFaultPublication(std::span<const GuestRange> ranges, uint32_t frame) {
	auto&           state = ReadFaultPublished();
	std::lock_guard lock(state.mutex);
	for (const auto& range: ranges) {
		const auto last = range.address + range.size;
		for (auto page = Common::AlignDown(range.address, TRACKER_PAGE_SIZE); page < last;
		     page += TRACKER_PAGE_SIZE) {
			state.published_frame[page] = frame;
		}
	}
	if (state.published_frame.size() > 8192) {
		std::erase_if(state.published_frame,
		              [frame](const auto& entry) { return frame - entry.second > 600; });
	}
}

[[nodiscard]] bool ReadFaultPublishedRecently(uint64_t page, uint32_t frame) {
	auto&           state = ReadFaultPublished();
	std::lock_guard lock(state.mutex);
	const auto      found = state.published_frame.find(page);
	return found != state.published_frame.end() &&
	       frame - found->second <= ReadFaultPublishedFrames();
}

void ReportReadFaultPublished(uint32_t frame) {
	auto&      state = ReadFaultPublished();
	const auto now   = ReadFaultDiag::NowNs();
	auto       last  = state.report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		state.report_ns.compare_exchange_strong(last, now);
		state.report_frame.store(frame, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10'000'000'000ull || !state.report_ns.compare_exchange_strong(last, now)) {
		return;
	}
	const auto frames = std::max<uint32_t>(1, frame - state.report_frame.exchange(frame));
	const auto take   = [](std::atomic<uint64_t>& counter) { return counter.exchange(0); };
	const auto served = take(state.served);
	std::printf("Kyty read-fault published 10s: %u frames, served %" PRIu64 " (%.2f/frame); waited: "
	            "not a plain load %" PRIu64 ", image %" PRIu64 ", not published recently %" PRIu64
	            ", other %" PRIu64 ", copy burst %" PRIu64 "; verify: equal %" PRIu64 ", different %" PRIu64 "\n",
	            frames, served, static_cast<double>(served) / frames, take(state.undecodable),
	            take(state.image), take(state.stale), take(state.other), take(state.burst), take(state.verify_equal),
	            take(state.verify_different));
}

// The number of regions copies (source ascending) give when a copy starting at most `gap` bytes
// after the previous region's source end joins it.
uint64_t CountMergedRegions(std::span<const vk::BufferCopy> copies, uint64_t gap) {
	uint64_t count = 0;
	uint64_t begin = 0;
	uint64_t end   = 0;
	for (const auto& copy: copies) {
		if (count == 0 || copy.srcOffset < begin || copy.srcOffset > end + gap) {
			count++;
			begin = copy.srcOffset;
			end   = copy.srcOffset + copy.size;
		} else {
			end = std::max(end, copy.srcOffset + copy.size);
		}
	}
	return count;
}

// KYTY_READBACK_REGION_LOG=1: every 10 s, the readback copy commands, their ranges and regions
// (and the regions a 64 KiB gap would give) per CP flip.
void RecordReadbackRegions(std::span<const vk::BufferCopy> ranges,
                           std::span<const vk::BufferCopy> regions) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_READBACK_REGION_LOG");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	struct Totals {
		std::atomic<uint64_t> commands {0};
		std::atomic<uint64_t> ranges {0};
		std::atomic<uint64_t> regions {0};
		std::atomic<uint64_t> regions_64k {0};
		std::atomic<uint64_t> range_bytes {0};
		std::atomic<uint64_t> region_bytes {0};
		std::atomic<int64_t>  last_ms {0};
		std::atomic<uint64_t> last_flip {0};
	};
	static Totals totals;
	uint64_t      range_bytes  = 0;
	uint64_t      region_bytes = 0;
	for (const auto& range: ranges) {
		range_bytes += range.size;
	}
	for (const auto& region: regions) {
		region_bytes += region.size;
	}
	totals.commands.fetch_add(1, std::memory_order_relaxed);
	totals.ranges.fetch_add(ranges.size(), std::memory_order_relaxed);
	totals.regions.fetch_add(regions.size(), std::memory_order_relaxed);
	totals.regions_64k.fetch_add(CountMergedRegions(ranges, 64 * 1024), std::memory_order_relaxed);
	totals.range_bytes.fetch_add(range_bytes, std::memory_order_relaxed);
	totals.region_bytes.fetch_add(region_bytes, std::memory_order_relaxed);
	const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                           std::chrono::steady_clock::now().time_since_epoch())
	                           .count();
	auto last = totals.last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		if (totals.last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
			totals.last_flip.store(Live::Testing::CpFlips(), std::memory_order_relaxed);
		}
		return;
	}
	if (now_ms - last < 10000 ||
	    !totals.last_ms.compare_exchange_strong(last, now_ms, std::memory_order_relaxed)) {
		return;
	}
	const auto flip  = Live::Testing::CpFlips();
	const auto flips = std::max<uint64_t>(1, flip - totals.last_flip.exchange(flip));
	const auto per   = [flips](std::atomic<uint64_t>& value) {
        return static_cast<double>(value.exchange(0, std::memory_order_relaxed)) /
               static_cast<double>(flips);
	};
	const auto commands    = per(totals.commands);
	const auto range_count = per(totals.ranges);
	const auto region      = per(totals.regions);
	const auto region_64k  = per(totals.regions_64k);
	const auto bytes       = per(totals.range_bytes);
	const auto copied      = per(totals.region_bytes);
	std::printf("ReadbackRegions 10s: %" PRIu64 " flips, per flip: %.1f copy commands, %.1f ranges, %.1f "
	     "regions (%.1f with a 64 KiB gap), %.1f KiB written back, %.1f KiB copied\n",
	     flips, commands, range_count, region, region_64k, bytes / 1024.0, copied / 1024.0);
}

// The regions of copies whose source and destination advance together (a side-readback window):
// a copy starting at most `gap` bytes after the previous region's source end joins it.
std::vector<vk::BufferCopy> MergeLinearCopies(const std::vector<vk::BufferCopy>& copies,
                                              uint64_t gap) {
	if (gap == 0 || copies.size() < 2) {
		return copies;
	}
	std::vector<vk::BufferCopy> regions;
	regions.reserve(copies.size());
	for (const auto& copy: copies) {
		if (!regions.empty()) {
			auto&      last     = regions.back();
			const auto last_end = last.srcOffset + last.size;
			if (copy.srcOffset >= last.srcOffset && copy.srcOffset <= last_end + gap &&
			    copy.dstOffset >= last.dstOffset &&
			    copy.dstOffset - last.dstOffset == copy.srcOffset - last.srcOffset) {
				last.size = std::max(last_end, copy.srcOffset + copy.size) - last.srcOffset;
				continue;
			}
		}
		regions.push_back(copy);
	}
	return regions;
}

// Packs download parts (source ascending) into regions: a part starting at most `gap` bytes after
// the previous region's source end joins it. Each part's dstOffset becomes its place inside its
// region's packed bytes (regions 64-byte aligned, as before). Returns the packed size.
uint64_t PackDownloadRegions(std::vector<vk::BufferCopy>& parts, uint64_t gap,
                             std::vector<vk::BufferCopy>& regions) {
	regions.clear();
	regions.reserve(parts.size());
	uint64_t packed = 0;
	for (auto& part: parts) {
		if (gap != 0 && !regions.empty()) {
			auto&      last     = regions.back();
			const auto last_end = last.srcOffset + last.size;
			if (part.srcOffset >= last.srcOffset && part.srcOffset <= last_end + gap) {
				packed -= Common::AlignUp(last.size, uint64_t {64});
				last.size = std::max(last_end, part.srcOffset + part.size) - last.srcOffset;
				packed += Common::AlignUp(last.size, uint64_t {64});
				part.dstOffset = last.dstOffset + (part.srcOffset - last.srcOffset);
				continue;
			}
		}
		part.dstOffset = packed;
		regions.emplace_back(part.srcOffset, packed, part.size);
		packed += Common::AlignUp(part.size, uint64_t {64});
	}
	return packed;
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
		// KYTY_INPLACE: pages bound in place must stay writable (hot) without faulting, so the hot
		// set defaults to the in-place cap (KYTY_INPLACE_MAX_MB in 4 KiB pages) unless set.
		const uint64_t hot_default =
		    InplaceHeap::ConfiguredMode() != InplaceHeap::Mode::Off
		        ? (ParseEnvU64("KYTY_INPLACE_MAX_MB", 512) << 20u) / TRACKER_PAGE_SIZE
		        : 1024;
		policy.hot_max =
		    static_cast<uint32_t>(std::min<uint64_t>(ParseEnvU64("KYTY_HOT_PAGE_MAX", hot_default), 65536));
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

// KYTY_STREAM_DIRECT_READ=1 (default off; upstream 37501b5a7): a small read binding copied into
// the stream buffer reads the guest bytes through the guest mapping instead of the backing store
// (TryReadBacking). Its pages are CPU-dirty and not GPU-dirty, so no protection stops the read.
BufferCache* g_defer_cache = nullptr; // the BufferCache whose stream copies are deferred
InplaceHeap* g_inplace_heap = nullptr; // KYTY_INPLACE: CP writes check its candidate pages
UploadDma*   g_write_drain_dma = nullptr; // KYTY_UPLOAD_DMA_WRITE_DRAIN: the cache's upload DMA

bool StreamCopyProbe() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_COPY_PROBE", 0) != 0;
	return enabled;
}

void StreamCopyProbeAdd(uint64_t size, uint64_t cycles, uint64_t vaddr) {
	static uint64_t count = 0, bytes = 0, total = 0, large = 0, start = 0, dup = 0, dup_bytes = 0;
	static uint64_t hist[8] = {};
	static std::unordered_map<uint64_t, uint64_t> last;
	{
		const auto key = vaddr ^ (size << 44u);
		const auto t   = __rdtsc();
		auto [it, inserted] = last.try_emplace(key, t);
		if (!inserted) {
			if (t - it->second < 33'600'000ull) { // 8 ms
				dup++;
				dup_bytes += size;
			}
			it->second = t;
		}
	}
	count++;
	bytes += size;
	total += cycles;
	hist[std::min<uint64_t>(std::bit_width(size) / 2u, 7u)]++;
	large += size > 4096 ? 1u : 0u;
	const auto now = __rdtsc();
	if (start == 0) {
		start = now;
	}
	if (now - start >= 42'000'000'000ull) {
		std::printf("StreamCopy 10s: dup8ms %" PRIu64 " dupbytes %" PRIu64 " keys %zu ", dup, dup_bytes, last.size());
		dup = dup_bytes = 0;
		last.clear();
		std::printf("copies %" PRIu64 " bytes %" PRIu64 " cycles %" PRIu64
		            " large %" PRIu64 " sizes(log4) %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
		            " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
		            count, bytes, total, large, hist[0], hist[1], hist[2], hist[3], hist[4], hist[5],
		            hist[6], hist[7]);
		std::fflush(stdout);
		count = bytes = total = large = 0;
		std::fill(std::begin(hist), std::end(hist), 0u);
		start = now;
	}
}

bool StreamDirectReadEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_DIRECT_READ", 0) != 0;
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

// KYTY_WRITE_DATA_GPU_PAGE (default 0 in this line, 1 in the Insomniac title presets; live): BufferCache::TryWriteDataGpu.
Live::Switch g_write_data_gpu_page("KYTY_WRITE_DATA_GPU_PAGE", [](const char* value) -> int64_t {
	return value == nullptr || value[0] == '\0' ? 0 : static_cast<int64_t>(std::strtoll(value, nullptr, 10));
});

// KYTY_TEXEL_RELAXED_SYNC (default 0 in this line, 2 in the Insomniac title presets; 1; 0 off; live): formatted (texel) read bindings use the
// same lock-free "nothing to upload" test as raw reads (1), and the read-binding memo
// (ObtainReadBinding, 2), followed by the image step (SynchronizeBufferFromImage) that only texel
// reads do. Before, every texel read walked the tracker under its region locks: Miles Morales binds
// ~6 texel buffers of ~4 MiB per draw, and each took 15.5 us (1: 3.7 us, 2: 0.6 us), two thirds of
// the command processor's time. The tracker part of a read synchronization only uploads CPU-dirty
// page runs and clears their CPU-dirty bits (GPU ownership, image staleness and epochs are the
// written path's), so with no CPU-dirty page it does nothing. KYTY_TRACKER_RELAXED_VERIFY and
// KYTY_BINDING_EPOCH_MEMO_VERIFY check both shortcuts (Miles Morales: 0 mismatches); Astro Bot's
// Sky Garden CP time is unchanged (live A/B 0 vs 2: 34.16 vs 34.20 ms/flip).
Live::Switch g_texel_relaxed_sync("KYTY_TEXEL_RELAXED_SYNC", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	return static_cast<int64_t>(std::strtoll(value, nullptr, 10));
});
bool TexelRelaxedSyncEnabled() {
	return g_texel_relaxed_sync.Get() != 0;
}
bool TexelBindingMemoEnabled() {
	return g_texel_relaxed_sync.Get() >= 2;
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
	static constexpr auto mode_init = [] {
		const auto* value = std::getenv("KYTY_TRACKER_RELAXED_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const int& mode = mode_once.Get(mode_init);
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

// KYTY_READBACK_EAGER_PAGES (hot pages kept, default 64, at most 2048),
// KYTY_READBACK_EAGER_IDLE_FRAMES (frames without a readback before a page stops being hot,
// default 600) and KYTY_READBACK_EAGER_FRAME_BUDGET (copies per page and frame, default 4).
EagerReadbackPages::Limits EagerReadbackLimits() {
	EagerReadbackPages::Limits limits;
	limits.capacity =
	    static_cast<uint32_t>(std::min<uint64_t>(ParseEnvU64("KYTY_READBACK_EAGER_PAGES", 64), 2048));
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
	// Side copies: the master tick the copy waits for (KYTY_READ_FAULT_DIAG).
	uint64_t                producer    = 0;
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
		HangWatchdog::Scope wait("side-readback-gpu",
		                         reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(semaphore)),
		                         target);
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

// KYTY_BDA_PAGETABLE_SPARSE: memory for the sparse page table, in chunks of zeroed device memory
// handed out one sparse block at a time. A chunk is zeroed (through a plain buffer over its memory)
// and that fill has completed before any of its blocks is bound, and every bind is waited for here:
// a block reads zero before its bind (residencyNonResidentStrict) and after it, so no GPU work, in
// flight or recorded later, can see anything but zero or an entry written after the bind.
struct BufferCache::SparsePageTable {
	static constexpr uint32_t ChunkBlocks = 32;

	struct Chunk {
		VmaAllocation  allocation = nullptr;
		vk::DeviceMemory memory   = nullptr;
		uint64_t       offset     = 0;
		uint32_t       used       = 0;
	};

	SparsePageTable(GraphicContext& context, vk::Buffer buffer, uint64_t size)
	    : graphics(context), table(buffer) {
		vk::MemoryRequirements requirements {};
		graphics.device.getBufferMemoryRequirements(table, &requirements);
		block_size       = requirements.alignment;
		memory_type_bits = requirements.memoryTypeBits;
		EXIT_IF(block_size == 0 || size % block_size != 0 || memory_type_bits == 0);
		bound.assign(static_cast<size_t>(size / block_size), 0);
		vk::CommandPoolCreateInfo pool_info {};
		pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                             vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool_info.queueFamilyIndex = graphics.queue_family;
		RequireVulkanSuccess(graphics.device.createCommandPool(&pool_info, nullptr, &pool),
		                     "create BDA page table command pool");
		vk::FenceCreateInfo fence_info {};
		RequireVulkanSuccess(graphics.device.createFence(&fence_info, nullptr, &fence),
		                     "create BDA page table fence");
		LOGF("BDA page table: sparse residency, %zu blocks of %" PRIu64 " KiB (KYTY_BDA_PAGETABLE_SPARSE)\n",
		     bound.size(), block_size / 1024);
	}

	~SparsePageTable() {
		for (const auto& chunk: chunks) {
			VramStats::Note(VramStats::Kind::OtherBuffer, true,
			                -static_cast<int64_t>(block_size * ChunkBlocks));
			VramStats::UnregisterBuffer(chunk.allocation);
			vmaFreeMemory(graphics.allocator, chunk.allocation);
		}
		if (fence != nullptr) {
			graphics.device.destroyFence(fence, nullptr);
		}
		if (pool != nullptr) {
			graphics.device.destroyCommandPool(pool, nullptr);
		}
	}

	// One submission to the side queue (queue 0 without one), waited for on the host.
	template <typename Submit>
	void SubmitAndWait(Submit&& submit, const char* operation) {
		HangWatchdog::Scope wait("sparse-page-table",
		                         reinterpret_cast<uint64_t>(static_cast<VkFence>(fence)));
		vk::Result result {};
		if (graphics.side_queue != nullptr) {
			Common::LockGuard lock(graphics.side_queue_mutex);
			result = submit(graphics.side_queue);
		} else {
			Common::LockGuard lock(graphics.queue_mutex);
			result = submit(graphics.queue);
		}
		RequireVulkanSuccess(result, operation);
		RequireVulkanSuccess(graphics.device.waitForFences(1, &fence, VK_TRUE, UINT64_MAX), operation);
		RequireVulkanSuccess(graphics.device.resetFences(1, &fence), operation);
	}

	// Allocates a chunk of device memory the table can bind and zeroes it.
	void AddChunk() {
		const uint64_t     bytes = block_size * ChunkBlocks;
		vk::BufferCreateInfo zero_info {};
		zero_info.size  = bytes;
		zero_info.usage = vk::BufferUsageFlagBits::eTransferDst;
		vk::Buffer zero_buffer = nullptr;
		RequireVulkanSuccess(graphics.device.createBuffer(&zero_info, nullptr, &zero_buffer),
		                     "create BDA page table zero buffer");
		vk::MemoryRequirements zero_requirements {};
		graphics.device.getBufferMemoryRequirements(zero_buffer, &zero_requirements);
		VkMemoryRequirements requirements {};
		requirements.size           = bytes;
		requirements.alignment      = std::max<uint64_t>(block_size, zero_requirements.alignment);
		requirements.memoryTypeBits = memory_type_bits & zero_requirements.memoryTypeBits;
		EXIT_IF(requirements.memoryTypeBits == 0);
		// Own device memory, not dedicated to a resource: both buffers may bind it.
		VmaAllocationCreateInfo create {};
		create.flags         = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
		create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		Chunk             chunk;
		VmaAllocationInfo info {};
		RequireVulkanSuccess(static_cast<vk::Result>(vmaAllocateMemory(graphics.allocator, &requirements,
		                                                               &create, &chunk.allocation, &info)),
		                     "allocate BDA page table memory");
		chunk.memory = info.deviceMemory;
		chunk.offset = info.offset;
		EXIT_IF(chunk.offset % block_size != 0);
		RequireVulkanSuccess(static_cast<vk::Result>(vmaBindBufferMemory(graphics.allocator,
		                                                                 chunk.allocation, zero_buffer)),
		                     "bind BDA page table zero buffer");
		vk::CommandBufferAllocateInfo command_info {};
		command_info.commandPool        = pool;
		command_info.level              = vk::CommandBufferLevel::ePrimary;
		command_info.commandBufferCount = 1;
		vk::CommandBuffer command = nullptr;
		RequireVulkanSuccess(graphics.device.allocateCommandBuffers(&command_info, &command),
		                     "allocate BDA page table command buffer");
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		RequireVulkanSuccess(command.begin(&begin), "begin BDA page table zero fill");
		command.fillBuffer(zero_buffer, 0, VK_WHOLE_SIZE, 0);
		RequireVulkanSuccess(command.end(), "end BDA page table zero fill");
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		SubmitAndWait([&](vk::Queue queue) { return queue.submit(1, &submit, fence); },
		              "zero BDA page table memory");
		graphics.device.freeCommandBuffers(pool, 1, &command);
		graphics.device.destroyBuffer(zero_buffer, nullptr);
		VramStats::Note(VramStats::Kind::OtherBuffer, true, static_cast<int64_t>(bytes));
		VramStats::RegisterBuffer(chunk.allocation, {bytes, 0, 0, true, false});
		chunks.push_back(chunk);
	}

	// Binds zeroed memory behind every block of table bytes [offset, offset + size) without any.
	void EnsureResident(uint64_t offset, uint64_t size) {
		const auto first = offset / block_size;
		const auto last  = (offset + size - 1) / block_size;
		std::vector<vk::SparseMemoryBind> binds;
		for (auto block = first; block <= last; ++block) {
			if (bound[block] != 0) {
				continue;
			}
			if (chunks.empty() || chunks.back().used == ChunkBlocks) {
				AddChunk();
			}
			auto&                chunk = chunks.back();
			vk::SparseMemoryBind bind {};
			bind.resourceOffset = block * block_size;
			bind.size           = block_size;
			bind.memory         = chunk.memory;
			bind.memoryOffset   = chunk.offset + uint64_t {chunk.used} * block_size;
			binds.push_back(bind);
			chunk.used++;
			bound[block] = 1;
		}
		if (binds.empty()) {
			return;
		}
		vk::SparseBufferMemoryBindInfo buffer_bind {};
		buffer_bind.buffer    = table;
		buffer_bind.bindCount = static_cast<uint32_t>(binds.size());
		buffer_bind.pBinds    = binds.data();
		vk::BindSparseInfo bind_info {};
		bind_info.bufferBindCount = 1;
		bind_info.pBufferBinds    = &buffer_bind;
		SubmitAndWait([&](vk::Queue queue) { return queue.bindSparse(1, &bind_info, fence); },
		              "bind BDA page table memory");
		bound_blocks += binds.size();
	}

	GraphicContext&      graphics;
	vk::Buffer           table;
	uint64_t             block_size       = 0;
	uint32_t             memory_type_bits = 0;
	std::vector<uint8_t> bound;
	std::vector<Chunk>   chunks;
	uint64_t             bound_blocks = 0;
	vk::CommandPool      pool  = nullptr;
	vk::Fence            fence = nullptr;
};

void BufferCache::EnsureBdaTableResident(uint64_t first_page, uint64_t page_count) {
	if (m_bda_sparse == nullptr || page_count == 0) {
		return;
	}
	m_bda_sparse->EnsureResident(first_page * sizeof(vk::DeviceAddress),
	                             page_count * sizeof(vk::DeviceAddress));
}

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
	m_unregister_epoch++;
	ChangeRegister<false>(id);
}

void BufferCache::EnsureBdaPageTableInitialized() {
	if (m_bda_pagetable_initialized) {
		return;
	}
	// A sparse table reads as zero wherever nothing is bound (KYTY_BDA_PAGETABLE_SPARSE).
	if (m_bda_pagetable_buffer.IsSparse()) {
		m_bda_pagetable_initialized = true;
		return;
	}
	// Recorded before the first table write or read, and ordered by Fill barriers.
	m_bda_pagetable_buffer.Fill(0, m_bda_pagetable_buffer.Size(), 0);
	m_bda_pagetable_initialized = true;
}

void BufferCache::EnsureBdaPageTableEntry(uint64_t index) {
	EnsureBdaPageTableInitialized();
	if (m_bda_sparse != nullptr) {
		m_bda_sparse->EnsureResident(index * sizeof(vk::DeviceAddress), sizeof(vk::DeviceAddress));
	}
}

Buffer* BufferCache::GetBdaPageTableBuffer() {
	EnsureBdaPageTableInitialized();
	return &m_bda_pagetable_buffer;
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	if constexpr (insert) {
		if (m_bda_register_log) {
			NoteBdaRegistered(buffer.CpuAddress(), buffer.Size());
		} else {
			InvalidateBdaSynchronization();
		}
	} else {
		InvalidateBdaSynchronization();
	}
	EnsureBdaPageTableInitialized();
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
		buffer.lru_id   = m_lru_cache.Insert(id, m_gc_tick);
		buffer.lru_tick = m_gc_tick;
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		ApplyBdaTags(pages.first, addresses, true); // KYTY_BDA_ACCESS_TAGS
		EnsureBdaTableResident(pages.first, size_pages);
		WriteDataBuffer(m_bda_pagetable_buffer, pages.first * sizeof(vk::DeviceAddress),
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		// With VK_EXT_memory_budget the collector sets m_total_used_memory to the device-local
		// usage, which can be below the buffers' own bytes (allocations outside the device-local
		// heaps when VRAM runs short), so it saturates. Without it the counter is the buffers'
		// own bytes and must not underflow.
		EXIT_IF(buffer.Size() > m_total_used_memory && !m_graphics.CanReportMemoryUsage());
		m_total_used_memory -= std::min(m_total_used_memory, buffer.Size());
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(pages.first * sizeof(vk::DeviceAddress),
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
	NoteBdaReachRegister(pages.first, pages.last_exclusive, insert); // KYTY_BDA_REACH=verify
	if (m_bda_live_count.load(std::memory_order_relaxed) != 0) {
		// KYTY_BDA_LIVE_PAGES: the entries were just rewritten; the redirects of these pages end.
		RestoreLivePages(pages.first << CACHING_PAGEBITS, size_pages << CACHING_PAGEBITS, LiveRestore::Register);
	}
}

// KYTY_BUFFER_LRU_SKIP=1 (default off): draws touch the same buffers many times per GC tick. The
// buffer mirrors its LRU item's tick (lru_tick, set by Insert and every Touch, the only writers of
// the item's tick), so a touch in a tick the item already holds, which LeastRecentlyUsedCache::Touch
// would return from at once, skips reading the scattered item. The LRU order is unchanged.
static bool BufferLruSkipEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_BUFFER_LRU_SKIP");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}));
	return enabled;
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (buffer.is_deleted) {
		return;
	}
	if (buffer.gc_pending) {
		buffer.gc_reused = true; // KYTY_GC_ASYNC_DOWNLOAD: it stays
	}
	if (BufferLruSkipEnabled()) {
		if (buffer.lru_tick >= m_gc_tick) {
			return;
		}
		buffer.lru_tick = m_gc_tick;
	}
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
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
	// Nearby ranges share one copy region (KYTY_READBACK_MERGE_GAP_KB); `copies` stay the exact
	// parts written back.
	std::vector<vk::BufferCopy> regions;
	total_size = PackDownloadRegions(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()), regions);
	RecordReadbackRegions(copies, regions);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	// A download larger than the staging ring gets a buffer of its own, released after the
	// publication below (upstream 18a5f04d0).
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}
	for (auto& region: regions) {
		region.dstOffset += offset;
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
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(regions.size()), regions.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
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
	                                    early, copies = std::move(copies),
	                                    owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
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

// KYTY_STREAM_RING_HOST=1 (default off): the 64 MiB stream ring (per-draw copies of small
// CPU-written buffers, flattened SRTs, shader data, index data) lives in cached host memory instead
// of the device-local host-visible heap (write-combined over the PCIe BAR), so the CPU's writes and
// the accesses after them do not stall on write-combining; the GPU reads the ring over the bus.
// Idea from BryanKAdams/KytyPS5 a25a2b9 (measured there on ReBAR; this machine has ReBAR off, so
// the ring sits in the 256 MiB BAR window today).
static bool StreamRingHostEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_STREAM_RING_HOST");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_ALLOCATION,
                             false, false,
                             graphics.sparse_residency_buffer_enabled),
      m_bda_incremental_sync(IncrementalBdaSyncEnabled()),
      m_bda_hot_sync(m_bda_incremental_sync && BdaHotSyncEnabled()),
      m_memory_tracker(page_manager, m_bda_incremental_sync, BufferFaultPolicy()),
      m_hot_quiet_frames(HotPageQuietFrames()),
      m_hot_check_limit(HotPageCheckLimit()),
      m_range_memo(RangeMemoEnabled() ? std::make_unique<RangeMemo[]>(RangeMemoSlots) : nullptr),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, UploadRingSize(),
                       graphics.transfer_queue != nullptr),
      m_upload_dma(UploadDma::Create(graphics, scheduler)),
      // Device address: the indirect mesh task shader reads its parameters from this ring
      // (upstream abc509d33).
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB, false,
                      vk::BufferUsageFlagBits::eShaderDeviceAddress, StreamRingHostEnabled()),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	m_range_memo_verify = m_range_memo != nullptr ? RangeMemoVerifyMode() : 0;
	// KYTY_TRACKER_LOCKFREE_DIRTY (memoryTracker.h): the GPU thread owns the GPU-dirty bits.
	MemoryTracker::SetLockFreeDirtyOwner(&GuestGpu::IsGpuThread);
	if (m_upload_dma != nullptr) {
		g_write_drain_dma = m_upload_dma.get(); // KYTY_UPLOAD_DMA_WRITE_DRAIN
	}
	if (const auto mode = InplaceHeap::ConfiguredMode(); mode != InplaceHeap::Mode::Off) {
		if (graphics.external_memory_host_enabled && Libs::LibKernel::Memory::GuestBackingBase() != 0) {
			m_inplace       = std::make_unique<InplaceHeap>(graphics, scheduler, mode);
			g_inplace_heap  = m_inplace.get();
		} else {
			std::printf("Kyty in-place reads: off (no VK_EXT_external_memory_host or no backing alias)\n");
		}
	}
	m_relaxed_queries = RelaxedQueriesEnabled();
	m_bda_dirty_log   = m_bda_hot_sync && BdaDirtyLogEnabled();
	if (m_bda_dirty_log) {
		m_memory_tracker.EnableDirtiedLog();
		m_bda_log_verify = BdaDirtyLogVerifyMode();
		// KYTY_BDA_REGISTER_LOG (bufferCache.h; default off).
		m_bda_register_log = m_bda_incremental_sync && ParseEnvU64("KYTY_BDA_REGISTER_LOG", 0) != 0;
	}
	m_bda_epoch_skip  = SyncEpoch::Enabled() && ParseEnvU64("KYTY_BDA_SYNC_EPOCH", 1) != 0;
	// The verify mode tells guest writes from missed pages by the fault epoch, which the tracker
	// keeps only with incremental BDA synchronization.
	m_bda_epoch_verify = m_bda_epoch_skip && m_bda_incremental_sync ? BdaEpochVerifyMode() : 0;
	if (SyncEpoch::Enabled() && BindingEpochMemoEnabled()) {
		const bool large      = CpCommit::Enabled(CpCommit::Part::BindSlots);
		m_binding_memo_shift  = large ? 49u : 53u;
		m_binding_memo        = std::make_unique<BindingMemo[]>(large ? BindingMemoSlotsLarge
		                                                              : BindingMemoSlots);
		m_binding_memo_verify = BindingEpochMemoVerifyMode();
		m_binding_memo_cross  = BindingMemoCrossEpochEnabled();
	}
	MemoryTracker::SetFaultAheadOverride(FaultAheadOverridePages());
	MemoryTracker::SetSequentialAhead(g_fault_ahead_sequential.Get() != 0);
	if (ParseEnvU64("KYTY_INDIRECT_FORMAT_COUNT", 0) == 1) {
		m_format_counts = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0,
		    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, 256 * sizeof(uint32_t));
		std::memset(m_format_counts->Mapped().data(), 0, 256 * sizeof(uint32_t));
		m_format_counts->Flush(0, 256 * sizeof(uint32_t));
		ShaderRecompiler::Spirv::SetIndirectFormatCountAddress(m_format_counts->BufferDeviceAddress());
		LOGF("KYTY_INDIRECT_FORMAT_COUNT: counting full-decoder calls per format\n");
	}
	m_written_sync_skip = WrittenSyncSkipEnabled();
	if (m_written_sync_skip) {
		m_written_sync_skip_verify = EnvVerifyMode("KYTY_WRITTEN_SYNC_SKIP_VERIFY");
	}
	m_false_sharing = FalseSharingWritesEnabled();
	if (m_false_sharing) {
		m_false_sharing_verify = EnvVerifyMode("KYTY_FALSE_SHARING_WRITES_VERIFY");
	}
	if (m_bda_pagetable_buffer.IsSparse()) {
		m_bda_sparse = std::make_unique<SparsePageTable>(m_graphics, m_bda_pagetable_buffer.Handle(),
		                                                 BDA_PAGETABLE_ALLOCATION);
	}
	CreateBdaTags(); // KYTY_BDA_ACCESS_TAGS (bufferCacheTags.cpp)
	CreateBdaLive(); // KYTY_BDA_LIVE_PAGES (bufferCacheLive.cpp)
	// Buffers have their own switches: shaders reach guest memory through BDA pointers without a
	// binding, which never touches a buffer's LRU entry, so "unused for N frames" can be wrong for a
	// buffer only BDA code reads (its draws then fault until the buffer is recreated). The image
	// switches (KYTY_VRAM_IDLE_FRAMES, KYTY_VRAM_PRESSURE_FRAMES) do not retire buffers.
	m_idle_frames     = ParseEnvU64("KYTY_VRAM_IDLE_BUFFER_FRAMES", 0);
	m_pressure_frames = ParseEnvU64("KYTY_VRAM_PRESSURE_BUFFER_FRAMES", 0);
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	if (m_upload_dma != nullptr) {
		m_scheduler.SetSubmitDependency(m_upload_dma.get(), 1);
	}
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
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
	if (g_defer_cache == this) {
		g_defer_cache = nullptr;
	}
	if (m_inplace != nullptr && g_inplace_heap == m_inplace.get()) {
		g_inplace_heap = nullptr;
	}
	if (m_upload_dma != nullptr && g_write_drain_dma == m_upload_dma.get()) {
		g_write_drain_dma = nullptr;
	}
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
	NoteMetadataWrite(vaddr, size);
	const auto flush = [this, vaddr, size] { ReadMemory(vaddr, size, true); };
	if (!write_fault) {
		m_memory_tracker.InvalidateRegion(vaddr, size, flush);
		return;
	}
	InplaceStats::NoteFault(vaddr);
	if (m_bda_live != nullptr) {
		m_bda_live_faults.fetch_add(1, std::memory_order_relaxed);
	}
	if (m_inplace != nullptr) {
		m_inplace->NoteFault(vaddr);
	}
	// KYTY_READBACK_SIDE_WRITES: only a fault taken by the exception handler is retried.
	const auto fault_flush = [this, vaddr, size] {
		ReadMemory(vaddr, size, true, FaultCost::InRetriedFault());
	};
	// Pages fault-ahead opens take writes without faulting: forget their fills first, exactly as
	// for the faulting range (never GPU-dirty pages, so fills of GPU-owned ranges survive).
	m_memory_tracker.InvalidateRegionOnWriteFault(
	    vaddr, size, fault_flush,
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
		if (command.KeepUploadsForDispatch()) {
			return; // KYTY_DISPATCH_OVERLAP_UPLOADS: recorded at the dispatch's own flush
		}
		command.FlushBarriers();
	}
}

void BufferCache::AdvanceFrame() noexcept {
	m_memory_tracker.AdvanceFrame();
	InplaceStats::AdvanceFrame();
	if (m_inplace != nullptr) {
		m_inplace->AdvanceFrame();
	}
	FaultCost::AdvanceFrame();
	if (m_bda_live != nullptr) {
		ReportBdaLive(); // KYTY_BDA_LIVE_PAGES: one line every 10 s
	}
	// KYTY_FAULT_AHEAD_ADAPT, applied once per guest flip (the cost model moves slowly).
	MemoryTracker::SetFaultAheadOverride(FaultAheadOverridePages());
	MemoryTracker::SetSequentialAhead(g_fault_ahead_sequential.Get() != 0);
	MemoryTracker::SetHotOverride(static_cast<uint32_t>(g_hot_frames_live.Get()),
	                              static_cast<uint32_t>(g_hot_max_live.Get()));
	if (m_format_counts != nullptr && ++m_format_count_frames % 600 == 0) {
		m_format_counts->Invalidate(0, 256 * sizeof(uint32_t));
		const auto* counts = reinterpret_cast<const uint32_t*>(m_format_counts->Mapped().data());
		std::string text;
		for (uint32_t format = 0; format < 128; format++) {
			if (counts[format] != 0) text += fmt::format(" {}:{}", format, counts[format]);
		}
		LOGF("Indirect formats via the full decoder (format:calls, cumulative; with "
		     "KYTY_INDIRECT_FORMAT_COMPARE: compact/full mismatches):%s\n",
		     text.empty() ? " none" : text.c_str());
		// KYTY_INDIRECT_FORMAT_COMPARE: the first mismatch (V# words 0-3, index, voffset, soffset,
		// compact x/y/z/w, full x/y/z/w) at words 129..143, claimed through word 128.
		if (counts[128] != 0) {
			std::string record;
			for (uint32_t word = 129; word < 144; word++) record += fmt::format(" {:08x}", counts[word]);
			LOGF("KYTY_INDIRECT_FORMAT_COMPARE first mismatch (V# w0-3, index, voffset, soffset | compact "
			     "xyzw | full xyzw):%s\n",
			     record.c_str());
		}
		std::fflush(stdout);
	}
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
	// KYTY_INPLACE: pages bound in place get no uploads, so the idle sweep would demote them
	// (write-protect them, and they would fault again) although the GPU reads them every frame.
	if (m_inplace == nullptr) {
		m_memory_tracker.SweepHotPages(m_hot_quiet_frames);
	}
	std::erase_if(m_hot_shadows, [this, frame](const auto& entry) {
		return frame - entry.second.last_use > m_hot_quiet_frames;
	});
}

// KYTY_HOT_PAGE_SPANS=1 (default off): a changed hot page uploads only the span from its first to
// its last 64-byte line that differs from the page's shadow (what the buffer holds), not the whole
// 4 KiB page. Idea from TheCruZ/KytyPS5-GTA (uploads only what the guest wrote).
static bool HotPageSpansEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_HOT_PAGE_SPANS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// [first, last) byte offsets of the 64-byte lines of a 4 KiB page that differ; the whole page when
// none differ (the caller found a difference, so that only happens with a racing write).
static std::pair<uint64_t, uint64_t> DifferingLines(const uint8_t* shadow, const uint8_t* page) {
	constexpr uint64_t Line  = 64;
	uint64_t           first = 0;
	while (first < TRACKER_PAGE_SIZE && std::memcmp(shadow + first, page + first, Line) == 0) {
		first += Line;
	}
	if (first == TRACKER_PAGE_SIZE) {
		return {0, TRACKER_PAGE_SIZE};
	}
	uint64_t last = TRACKER_PAGE_SIZE;
	while (last > first + Line && std::memcmp(shadow + last - Line, page + last - Line, Line) == 0) {
		last -= Line;
	}
	return {first, last};
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
			bool fresh = false;
			if (shadow == m_hot_shadows.end()) {
				if (m_hot_shadows.size() < max_pages) {
					shadow = m_hot_shadows.emplace(page, HotShadow {}).first;
					fresh  = true;
					shadow->second.data = std::make_unique<uint8_t[]>(TRACKER_PAGE_SIZE);
				} else {
					// No shadow to compare against: back to faulting on writes.
					demote.push_back(page);
				}
			}
			// KYTY_HOT_PAGE_SPANS: with a shadow of the buffer's contents, only the span of 64-byte
			// lines that differ from it is uploaded (moved to the start of the snapshot slot).
			uint64_t span_begin = 0;
			uint64_t span_bytes = TRACKER_PAGE_SIZE;
			if (shadow != m_hot_shadows.end() && !fresh && HotPageSpansEnabled()) {
				const auto [first, last] = DifferingLines(shadow->second.data.get(), snapshot);
				span_begin               = first;
				span_bytes               = last - first;
			}
			if (shadow != m_hot_shadows.end()) {
				std::memcpy(shadow->second.data.get(), snapshot, TRACKER_PAGE_SIZE);
				shadow->second.last_change      = frame;
				shadow->second.last_use         = frame;
				shadow->second.unchanged_checks = 0;
			}
			if (span_begin != 0) {
				std::memmove(snapshot, snapshot + span_begin, span_bytes);
			}
			copies.emplace_back(total_size, buffer.Offset(page) + span_begin, span_bytes);
			total_size += span_bytes;
			staged += span_bytes;
		}
	}
	MemoryStats::Count(MemoryStats::Counter::HotUploadPages, visited);
	MemoryStats::Count(MemoryStats::Counter::HotUploadSkipped, skipped);
	if (settled != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::HotPageCheckSettles, settled);
		m_range_memo_totals.settles += settled;
	}
}

namespace {

// KYTY_READ_FAULT_DIAG (readFaultDiag.h): one guest read fault of the diagnosed page, printed
// when ReadMemory returns.
class ReadFaultDiagScope {
public:
	ReadFaultDiagScope(uint64_t vaddr, MasterSemaphore& master) {
		m_fault.start_ns    = ReadFaultDiag::NowNs();
		m_fault.fault_vaddr = vaddr;
		auto& state         = ReadFaultDiag::Get();
		m_write_ns          = state.write_ns.load(std::memory_order_acquire);
		m_write_tick        = state.write_tick.load(std::memory_order_relaxed);
		m_write_caller      = state.write_caller.load(std::memory_order_relaxed);
		m_current           = master.CurrentTick();
		master.Refresh();
		m_gpu = master.KnownGpuTick();
		if (const auto& progress = master.GetSubmissionProgress(); progress != nullptr) {
			m_dispatched = progress->dispatched_tick.load(std::memory_order_acquire);
		}
		ReadFaultDiag::t_fault = &m_fault;
	}
	~ReadFaultDiagScope() {
		ReadFaultDiag::t_fault = nullptr;
		auto& state = ReadFaultDiag::Get();
		if (state.lines.fetch_add(1, std::memory_order_relaxed) >= MaxLines()) {
			return;
		}
		const auto end     = ReadFaultDiag::NowNs();
		const auto labels  = ReadFaultDiag::LabelsBetween(m_write_ns, m_fault.start_ns);
		const auto us      = [](uint64_t ns) { return static_cast<double>(ns) / 1e3; };
		const auto age     = [&](uint64_t ns) { return ns != 0 ? us(m_fault.start_ns - ns) : -1.0; };
		const auto publish = m_fault.phases_end_ns != 0 ? end - m_fault.phases_end_ns : 0;
		char       thread[64] {};
		FaultCost::DescribeThread(thread, sizeof(thread));
		const auto rip  = FaultCost::FaultInstruction();
		const auto base = FaultCost::ExeBase();
		{
			// The faulting guest instruction's bytes, once per instruction.
			static std::mutex                   seen_mutex;
			static std::unordered_set<uint64_t> seen;
			std::unique_lock                    seen_lock(seen_mutex);
			if (rip != 0 && seen.size() < 64 && seen.insert(rip).second) {
				seen_lock.unlock();
				uint8_t code[16] {};
				if (HangTrace::TryReadReadable(rip, code, sizeof(code))) {
					std::string hex;
					for (const auto byte: code) {
						hex += fmt::format("{:02x} ", byte);
					}
					std::printf("ReadFaultDiag code 0x%" PRIx64 ": %s\n", rip, hex.c_str());
				}
			}
		}
		std::printf(
		    "ReadFaultDiag t=%.3f thr='%s' rip=0x%" PRIx64 " path=%s off=0x%03" PRIx64
		    " dirty=%d waittick=%" PRIu64 " cur=%" PRIu64 " gpu=%" PRIu64 " disp=%" PRIu64
		    " | total %.1f sendsync %.1f dispatch %.1f gpu %.1f copy %.1f publish %.1f us"
		    " | write age %.1f us tick %" PRIu64 " caller kyty+0x%" PRIx64
		    " | labels after write %" PRIu64 ", first %s 0x%" PRIx64 "=0x%" PRIx64 " age %.1f us,"
		    " last %s 0x%" PRIx64 "=0x%" PRIx64 " age %.1f us, newest any age %.1f us\n",
		    us(m_fault.start_ns) / 1e6, thread, rip, m_path, m_fault.fault_vaddr - ReadFaultDiag::Page(),
		    m_fault.byte_known ? (m_fault.byte_dirty ? 1 : 0) : -1, m_fault.wait_tick, m_current, m_gpu,
		    m_dispatched, us(end - m_fault.start_ns), us(m_fault.sendsync_ns), us(m_fault.dispatch_ns),
		    us(m_fault.gpu_ns), us(m_fault.copy_ns), us(publish), age(m_write_ns), m_write_tick,
		    m_write_caller - std::min(m_write_caller, base), labels.count, labels.first.kind,
		    labels.first.address, labels.first.value, age(labels.first.ns), labels.last.kind,
		    labels.last.address, labels.last.value, age(labels.last.ns), age(labels.last_any_ns));
	}
	ReadFaultDiagScope(const ReadFaultDiagScope&)            = delete;
	ReadFaultDiagScope& operator=(const ReadFaultDiagScope&) = delete;

	void                  SetPath(const char* path) noexcept { m_path = path; }
	ReadFaultDiag::Fault& Fault() noexcept { return m_fault; }

private:
	static uint32_t MaxLines() {
		static const uint32_t lines = [] {
			const auto* value = std::getenv("KYTY_READ_FAULT_DIAG_LINES");
			return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : 4000u;
		}();
		return lines;
	}

	ReadFaultDiag::Fault m_fault;
	const char*          m_path         = "other";
	uint64_t             m_write_ns     = 0;
	uint64_t             m_write_tick   = 0;
	uint64_t             m_write_caller = 0;
	uint64_t             m_current      = 0;
	uint64_t             m_gpu          = 0;
	uint64_t             m_dispatched   = 0;
};

} // namespace

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write, bool retried_write) {
	std::optional<ReadFaultDiagScope> diag;
	if (ReadFaultDiag::Covers(vaddr, size) && !is_write && !GuestGpu::IsGpuThread()) {
		diag.emplace(vaddr, m_scheduler.GetMasterSemaphore());
	}
	// KYTY_READ_FAULT_DIAG=auto: the first slow guest read fault (after the load) chooses the page.
	struct AutoChooser {
		uint64_t vaddr = 0;
		uint64_t start = 0;
		~AutoChooser() {
			if (start != 0 && ReadFaultDiag::NowNs() - start > 2'000'000) {
				ReadFaultDiag::AutoChoose(vaddr);
			}
		}
	} auto_chooser;
	if (!diag && !is_write && ReadFaultDiag::AutoPending() && !GuestGpu::IsGpuThread() &&
	    FaultCost::Frame() > 3000) {
		auto_chooser.vaddr = vaddr;
		auto_chooser.start = ReadFaultDiag::NowNs();
	}
	// KYTY_READ_FAULT_PUBLISHED: complete a guest load with the page's published bytes, or (verify)
	// remember what it would have read and compare after the wait below.
	struct PublishedVerify {
		GuestLoad::Load load;
		uint8_t         before[16] {};
		uint32_t        frame = 0;
		~PublishedVerify() {
			uint8_t after[16] {};
			if (!LibKernel::Memory::TryReadBacking(load.address, after, load.size)) {
				return;
			}
			auto&      state = ReadFaultPublished();
			const bool equal = std::memcmp(before, after, load.size) == 0;
			(equal ? state.verify_equal : state.verify_different).fetch_add(1, std::memory_order_relaxed);
			if (!equal && state.logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::string text;
				for (uint32_t i = 0; i < load.size; i++) {
					text += fmt::format("{:02x}{:02x} ", before[i], after[i]);
				}
				std::printf("Kyty read-fault published verify: 0x%" PRIx64 " (%u bytes) served/waited "
				            "bytes differ: %s\n",
				            load.address, load.size, text.c_str());
			}
		}
	};
	std::optional<PublishedVerify> published_verify;
	if (const auto published_mode = g_read_fault_published.Get();
	    published_mode != 0 && !is_write && GuestLoad::t_native_context != nullptr &&
	    !GuestGpu::IsGpuThread()) {
		auto&      state = ReadFaultPublished();
		const auto frame = m_memory_tracker.Frame();
		ReportReadFaultPublished(frame);
		const auto      page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		GuestLoad::Load load;
		uint8_t         bytes[16] {};
		// A copy loop over the page would fault on every load: after a burst, wait (which
		// publishes the page and makes it readable) instead.
		thread_local uint64_t burst_page  = 0;
		thread_local uint64_t burst_ns    = 0;
		thread_local uint32_t burst_count = 0;
		const auto            now_ns      = ReadFaultDiag::NowNs();
		burst_count = burst_page == page && now_ns - burst_ns < 1'000'000 ? burst_count + 1 : 0;
		burst_page  = page;
		burst_ns    = now_ns;
		if (burst_count >= 8) {
			state.burst.fetch_add(1, std::memory_order_relaxed);
		} else if (!GuestLoad::Decode(vaddr, load) ||
		    Common::AlignDown(load.address + load.size - 1, TRACKER_PAGE_SIZE) != page) {
			state.undecodable.fetch_add(1, std::memory_order_relaxed);
		} else if (!IsRegionGpuModifiedRelaxed(page, TRACKER_PAGE_SIZE) ||
		           m_texture_cache.IsRegionGpuModified(page, TRACKER_PAGE_SIZE)) {
			// Not (only) buffer bytes the GPU owns: an image's contents are not in the backing.
			state.image.fetch_add(1, std::memory_order_relaxed);
		} else if (!ReadFaultPublishedRecently(page, frame)) {
			state.stale.fetch_add(1, std::memory_order_relaxed);
		} else if (!LibKernel::Memory::TryReadBacking(load.address, bytes, load.size)) {
			state.other.fetch_add(1, std::memory_order_relaxed);
		} else if (published_mode == 2) {
			published_verify.emplace();
			published_verify->load  = load;
			published_verify->frame = frame;
			std::memcpy(published_verify->before, bytes, load.size);
		} else if (GuestLoad::Apply(load, bytes)) {
			state.served.fetch_add(1, std::memory_order_relaxed);
			// Keep the page read-hot so its eager copy keeps publishing the GPU's new bytes.
			(void)m_scheduler.Context().GetGpu().TrySendCommand(
			    [this, page] { NoteEagerRead(page, TRACKER_PAGE_SIZE, false); });
			if (diag) {
				diag->SetPath("published");
			}
			return;
		} else {
			state.other.fetch_add(1, std::memory_order_relaxed);
		}
	}
	const auto diag_path = [&diag](const char* path) {
		if (diag) {
			diag->SetPath(path);
		}
	};
	HangWatchdog::Scope       read("buffer-readback", vaddr, size, 0, 0, is_write);
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
	// KYTY_FAULT_MAP: how this write fault's flush is served (FaultCost::NoteFlushKind).
	struct FlushNote {
		bool                 active  = false;
		FaultCost::FlushKind kind    = FaultCost::FlushKind::Other;
		uint64_t             wait_ns = 0;
		uint64_t             start   = 0;
		void Begin() {
			if (active) start = FaultCost::NowNs();
		}
		void End() {
			if (active && start != 0) wait_ns += FaultCost::NowNs() - start;
			start = 0;
		}
		~FlushNote() {
			if (active) FaultCost::NoteFlushKind(kind, wait_ns);
		}
	} flush_note {.active = is_write && FaultCost::SplitActive()};

	// Writes need the CPU-dirty transition of the drain path (or, KYTY_READBACK_SIDE_WRITES, of
	// a second fault). Guest-thread reads use side copies; GPU-thread reads do too
	// (KYTY_READBACK_SIDE_GPU_THREAD, default on) and wait for the copy themselves: it waits only
	// for the producing recording, not for the current one, and the current recording is neither
	// split nor submitted.
	const bool gpu_thread = GuestGpu::IsGpuThread();
	const bool side_write = is_write && retried_write && !gpu_thread && m_side != nullptr &&
	                        g_readback_side_writes.On();
	const bool side_path  = m_side != nullptr && (!is_write || side_write) &&
	                       (!gpu_thread || SideReadbackGpuThreadEnabled());
	if (OverlapsPendingSideReadback(page_begin, page_end)) {
		// Another fault already copies these pages (or an eager copy publishes them): wait for
		// (or finish) its publication instead of copying again. Writes and GPU-thread reads must
		// also be ordered after it.
		uint32_t eager = 0;
		{
			Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
			flush_note.Begin();
			eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
			flush_note.End();
		}
		flush_note.kind = FaultCost::FlushKind::Pending;
		if (eager != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackEagerWaits);
		}
		// A GPU-thread read continues below: bytes re-dirtied since that copy was issued still
		// need their own readback before the caller reads.
		if (side_path && !gpu_thread) {
			diag_path(eager != 0 ? "pending-eager" : "pending-side");
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
	// KYTY_ASYNC_DRAIN: a guest thread's drain is waited for here, not on the command processor.
	AsyncDrain  async_drain;
	AsyncDrain* async = !gpu_thread && AsyncDrainEnabled() ? &async_drain : nullptr;
	const auto  finish_async = [&, this, vaddr, size, is_write] {
		if (async_drain.tick == 0) {
			return;
		}
		{
			Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitDrain);
			const auto                    start = FaultCost::NowNs();
			flush_note.Begin();
			m_scheduler.WaitPriorityOperations(async_drain.tick);
			flush_note.End();
			g_async_drain.wait_ns.fetch_add(FaultCost::NowNs() - start, std::memory_order_relaxed);
		}
		gpu.SendCommandSync([&, this, vaddr, size, is_write] {
			FinishAsyncDrain(vaddr, size, is_write, async_drain, trace);
		});
		ReportAsyncDrain();
	};
	if (!side_path) {
		flush_note.kind = FaultCost::FlushKind::Drain;
		diag_path("drain-nonside");
		gpu.SendCommandSync([&, this, vaddr, size, is_write, gpu_thread] {
			if (!is_write || g_readback_eager_writes.On()) {
				NoteEagerRead(vaddr, size, gpu_thread);
			}
			CpGaps::Note(is_write ? CpGaps::Event::DrainWrite : CpGaps::Event::DrainOther);
			const FaultCost::AttribReadbackScope attrib(
			    vaddr, gpu_thread ? (is_write ? FaultCost::AttribPath::CpWriteDrain : FaultCost::AttribPath::CpReadDrain)
			                      : (is_write ? FaultCost::AttribPath::GuestWriteDrain
			                                  : FaultCost::AttribPath::GuestReadDrain));
			ReadMemoryDrain(vaddr, size, is_write, trace, async);
		});
		finish_async();
		record(std::nullopt);
		return;
	}

	if (gpu_thread) {
		NoteEagerRead(vaddr, size, true);
		if (FaultPagesCurrent(page_begin, page_end)) { // KYTY_CLEAN_PAGE
			record(std::nullopt);
			return;
		}
		std::shared_ptr<SideReadback> issued;
		const auto                    result = TryIssueSideReadback(vaddr, size, issued);
		if (result == SideIssueResult::Issued) {
			{
				Profiler::ScopedFrameWait            side_wait(Profiler::FrameWait::GpuWaitSideCopy);
				const FaultCost::AttribReadbackScope attrib_path(vaddr, FaultCost::AttribPath::CpReadSide);
				FaultCost::AttribScope               attrib(FaultCost::AttribKind::SideWait, false);
				attrib.SetAddress(vaddr);
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
		CpGaps::Note(CpGaps::Event::DrainGpuThread);
		const FaultCost::AttribReadbackScope attrib(vaddr, FaultCost::AttribPath::CpReadDrain);
		ReadMemoryDrain(vaddr, size, false, trace);
		record(std::nullopt);
		return;
	}

	std::shared_ptr<SideReadback> issued;
	auto                          result    = SideIssueResult::Other;
	const auto                    sync_mark = diag ? ReadFaultDiag::NowNs() : 0;
	bool                          current   = false; // KYTY_CLEAN_PAGE
	// KYTY_CLEAN_READ (default 0; 1 on, read once; adapted from usa-ratt/KytyPS5-GT7 19b6c5cfb): a
	// guest code's plain load (common/guestLoad.h) of bytes the GPU has not written, on a page
	// protected because the GPU wrote other bytes of it, is completed here with the bytes from
	// guest memory (TryReadCleanBytes decides on the GPU thread). The page stays protected; no
	// side copy, download or GPU wait. After 16 such loads of one page within 1 ms on a thread
	// (a copy loop), the normal path makes the page current instead.
	static const bool clean_read = ParseEnvU64("KYTY_CLEAN_READ", 0) != 0;
	GuestLoad::Load   clean_load;
	uint8_t           clean_bytes[16] {};
	bool              clean_candidate = false;
	bool              clean_served    = false;
	if (clean_read && !is_write && GuestLoad::t_native_context != nullptr) {
		thread_local uint64_t burst_page  = 0;
		thread_local uint64_t burst_ns    = 0;
		thread_local uint32_t burst_count = 0;
		const auto            now_ns      = ReadFaultDiag::NowNs();
		burst_count = burst_page == page_begin && now_ns - burst_ns < 1'000'000 ? burst_count + 1 : 0;
		burst_page  = page_begin;
		burst_ns    = now_ns;
		clean_candidate = burst_count < 16 && GuestLoad::Decode(vaddr, clean_load) &&
		                  clean_load.size != 0 && clean_load.size <= sizeof(clean_bytes);
	}
	const auto issue = [&, this, vaddr, size, is_write] {
		if (clean_candidate && TryReadCleanBytes(clean_load.address, clean_load.size, clean_bytes)) {
			clean_served = true;
			return;
		}
		clean_served = false;
		if (!is_write || g_readback_eager_writes.On()) {
			NoteEagerRead(vaddr, size, false);
		}
		// KYTY_CLEAN_PAGE: another thread's read made the faulting pages current meanwhile; a
		// side copy would only take the window's other GPU-written ranges, and this thread would
		// wait for it.
		if (FaultPagesCurrent(page_begin, page_end)) {
			if (is_write) {
				m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			}
			current = true;
			return;
		}
		result = TryIssueSideReadback(vaddr, size, issued);
		if ((result == SideIssueResult::CurrentWriter || result == SideIssueResult::Unbounded) &&
		    (g_readback_flush_side.On() || is_write)) {
			// KYTY_READBACK_FLUSH_SIDE (reads) / KYTY_READBACK_SIDE_WRITES: submit the writing
			// recording, then copy aside.
			CpGaps::Note(CpGaps::Event::SideFlushes);
			(void)m_scheduler.FlushObservable();
			result = TryIssueSideReadback(vaddr, size, issued);
			if (result == SideIssueResult::Issued) {
				CpGaps::Note(CpGaps::Event::SideFlushCopies);
				g_readback_flush_side_copies.fetch_add(1, std::memory_order_relaxed);
			}
		}
		if (result == SideIssueResult::Issued && is_write) {
			g_readback_side_writes_copies.fetch_add(1, std::memory_order_relaxed);
		}
		if (result != SideIssueResult::Issued && result != SideIssueResult::Pending) {
			CpGaps::Note(is_write                                   ? CpGaps::Event::DrainWrite
			             : result == SideIssueResult::CurrentWriter ? CpGaps::Event::DrainCurrentWriter
			             : result == SideIssueResult::Unbounded     ? CpGaps::Event::DrainUnbounded
			                                                        : CpGaps::Event::DrainOther);
			const FaultCost::AttribReadbackScope attrib(
			    vaddr, is_write ? FaultCost::AttribPath::GuestWriteDrain : FaultCost::AttribPath::GuestReadDrain);
			ReadMemoryDrain(vaddr, size, is_write, trace, async);
		}
	};
	gpu.SendCommandSync([&issue] { issue(); });
	if (clean_served) {
		static std::atomic<uint64_t> served {0};
		if (GuestLoad::Apply(clean_load, clean_bytes)) {
			if (const auto n = served.fetch_add(1, std::memory_order_relaxed) + 1;
			    (n & (n - 1)) == 0 && n >= 64) {
				LOGF("ReadMemory: %" PRIu64 " guest loads of bytes the GPU has not written served "
				     "from guest memory without a GPU wait (KYTY_CLEAN_READ)\n",
				     n);
			}
			diag_path("clean-read");
			record(std::nullopt);
			return;
		}
		// Not applied (the context stays untouched): take the normal path.
		clean_candidate = false;
		gpu.SendCommandSync([&issue] { issue(); });
	}
	if (current) {
		flush_note.kind = FaultCost::FlushKind::Current;
		diag_path("current");
		record(std::nullopt);
		return;
	}
	if (diag) {
		diag->Fault().sendsync_ns = ReadFaultDiag::NowNs() - sync_mark;
		switch (result) {
			case SideIssueResult::Issued: diag_path("side-issued"); break;
			case SideIssueResult::Pending: diag_path("pending-after-issue"); break;
			case SideIssueResult::CurrentWriter: diag_path("drain-current-writer"); break;
			case SideIssueResult::Unbounded: diag_path("drain-unbounded"); break;
			case SideIssueResult::Other: diag_path("drain-other"); break;
		}
	}
	switch (result) {
		case SideIssueResult::Issued: {
			// The GPU thread returned right after submitting; this guest thread waits.
			flush_note.kind = FaultCost::FlushKind::Side;
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				flush_note.Begin();
				CompleteSideReadback(*issued);
				flush_note.End();
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
			flush_note.kind = FaultCost::FlushKind::Pending;
			{
				Profiler::ScopedFrameWait side_wait(Profiler::FrameWait::ReadbackSideWait);
				flush_note.Begin();
				eager = CompleteSideReadbacks(page_begin, page_end - page_begin);
				flush_note.End();
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
	flush_note.kind = FaultCost::FlushKind::Drain; // ReadMemoryDrain ran on the command processor
	finish_async();
	record(std::nullopt);
}

uint64_t BufferCache::ReadbackFlushSideCopies() noexcept {
	return g_readback_flush_side_copies.load(std::memory_order_relaxed);
}

uint64_t BufferCache::ReadbackSideWriteCopies() noexcept {
	return g_readback_side_writes_copies.load(std::memory_order_relaxed);
}

bool BufferCache::StoreInPlace(uint64_t vaddr, const void* data, uint64_t size, bool gpu_written,
                               bool clean) {
	if (!GuestGpu::IsGpuThread() || size == 0 || data == nullptr || !GuestRange {vaddr, size}.Valid() ||
	    CommandScheduler::InDeferredOperation() || !m_scheduler.Active() ||
	    m_scheduler.Current().IsInvalid()) {
		return false; // the buffer's copy needs a recording to go into
	}
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
		// A page with no GPU-written bytes left only has its protection lifted on the normal
		// path, without a download: the stores after this one are then free, where done in place
		// each would fault again.
		if (!gpu_written || !HasGpuDirtyBytes(page_begin, page_end - page_begin)) {
			return false;
		}
	} else if (!clean || m_memory_tracker.IsRegionCpuModified(page_begin, page_end - page_begin)) {
		// CPU-dirty pages are writable: the fault was for another watcher (an image).
		return false;
	}
	// A readback or publication still to land would put older bytes over the store.
	if (HasPendingBackingPublication(vaddr, size) || OverlapsPendingSideReadback(page_begin, page_end) ||
	    (!m_early_released.empty() && OverlapsUnpublished(vaddr, size)) ||
	    m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return false;
	}
	auto& buffer = m_slot_buffers[*owner];
	if (!buffer.IsInBounds(vaddr, size)) {
		return false;
	}
	// A queued host copy or deferred stream copy that still reads these bytes is made first, as
	// for a command-processor write.
	StreamCopyDefer::NoteGuestWrite(vaddr, size);
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	WriteDataBuffer(buffer, vaddr, data, size);
	buffer.MarkContentWritten();
	ForgetKnownFills(vaddr, size);
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		// Overwritten: guest memory holds the value the GPU copy will have.
		CleanVerdict::Invalidate(vaddr, size, Coherence::Source::BufferDirtySubtract);
		m_gpu_modified_ranges.Subtract(vaddr, size);
	}
	// What a CPU store over these bytes tells the rest: draw-preparation certificates, images
	// over them (made CPU-modified, as a write fault does).
	Coherence::NoteContentWrite(vaddr, size, Coherence::Source::CpWrite);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool BufferCache::TryLiftFaultPages(uint64_t page_begin, uint64_t page_end) {
	const auto bytes = page_end - page_begin;
	// Still GPU-written for the tracker, but no byte left to download or to wait for.
	if (!m_memory_tracker.IsRegionGpuModified(page_begin, bytes) ||
	    HasGpuDirtyBytes(page_begin, bytes) || HasPendingBackingPublication(page_begin, bytes) ||
	    OverlapsPendingSideReadback(page_begin, page_end) ||
	    m_texture_cache.IsRegionGpuModified(page_begin, bytes)) {
		return false;
	}
	// An address-writing shader still running may write bytes of these pages nobody tracks yet.
	if (m_unbounded_write_tick != 0 && !m_scheduler.IsFree(m_unbounded_write_tick)) {
		return false;
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(page_begin, bytes);
	static std::atomic<uint64_t> lifted {0};
	if (const auto n = lifted.fetch_add(1, std::memory_order_relaxed) + 1;
	    (n & (n - 1)) == 0 && n >= 64) {
		LOGF("ReadMemory: lifted the protection of %" PRIu64 " faulting page ranges without GPU-"
		     "written bytes, no download (KYTY_FAULT_PAGE_LIFT)\n",
		     n);
	}
	return true;
}

bool BufferCache::TryReadCleanBytes(uint64_t vaddr, uint64_t size, void* data) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	if (size == 0 || data == nullptr || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	// Only pages the buffer tracker protects as GPU-written; other faults take their own path.
	const auto page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (!m_memory_tracker.IsRegionGpuModified(page_begin, page_end - page_begin)) {
		return false;
	}
	// None of the bytes may be GPU-written, early released or waiting for a publication, nor in
	// a GPU-written image whose contents are not in guest memory.
	if (HasGpuDirtyBytes(vaddr, size) || HasPendingBackingPublication(vaddr, size) ||
	    m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	// An address-writing shader still running may write any buffer byte nobody tracks yet.
	if (m_unbounded_write_tick != 0 && !m_scheduler.IsFree(m_unbounded_write_tick)) {
		return false;
	}
	return LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

bool BufferCache::FaultPagesCurrent(uint64_t page_begin, uint64_t page_end) {
	// KYTY_FAULT_PAGE_LIFT: live switch g_fault_page_lift, see TryLiftFaultPages.
	static const bool enabled = ParseEnvU64("KYTY_CLEAN_PAGE", 0) != 0;
	if (page_end <= page_begin) {
		return false;
	}
	if (g_fault_page_lift.On() && TryLiftFaultPages(page_begin, page_end)) {
		return true;
	}
	if (!enabled) {
		return false;
	}
	const auto bytes = page_end - page_begin;
	if (m_memory_tracker.IsRegionGpuModified(page_begin, bytes) ||
	    HasGpuDirtyBytes(page_begin, bytes) || HasPendingBackingPublication(page_begin, bytes) ||
	    OverlapsPendingSideReadback(page_begin, page_end) ||
	    m_texture_cache.IsRegionGpuModified(page_begin, bytes)) {
		return false;
	}
	static std::atomic<uint64_t> current {0};
	if (const auto n = current.fetch_add(1, std::memory_order_relaxed) + 1;
	    (n & (n - 1)) == 0 && n >= 64) {
		LOGF("ReadMemory: %" PRIu64 " faults on pages another read had made current went on at "
		     "once (KYTY_CLEAN_PAGE)\n",
		     n);
	}
	return true;
}

// KYTY_ASYNC_DRAIN (default 0; 1 on, read once; adapted from usa-ratt/KytyPS5-GT7's
// KYTY_ASYNC_READBACK, default on there): a guest thread's fault that takes the drain path (a write
// fault the side copy refused, a read whose side copy was refused) no longer makes the command
// processor wait for the GPU. The command processor records the window's download as before,
// submits the recording and goes on; the faulting thread waits for the download's publication
// itself (WaitPriorityOperations), then the command processor releases the window
// (FinishAsyncDrain) if no GPU write reached it in between, or drains again synchronously.
bool BufferCache::AsyncDrainEnabled() {
	static const bool enabled = ParseEnvU64("KYTY_ASYNC_DRAIN", 0) != 0;
	return enabled;
}



void BufferCache::FinishAsyncDrain(uint64_t vaddr, uint64_t size, bool is_write,
                                   const AsyncDrain& async, ReadMemoryTrace& trace) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	const auto begin = async.window_begin;
	const auto bytes = async.window_end - async.window_begin;
	// What the synchronous drain releases after its wait, provided nothing was recorded since that
	// put GPU-written bytes (or a newer readback) in the window: then the window holds exactly what
	// it held after the publication.
	const bool clean = !HasGpuDirtyBytes(begin, bytes) && !HasPendingBackingPublication(begin, bytes) &&
	                   !OverlapsPendingSideReadback(Common::AlignDown(begin, TRACKER_PAGE_SIZE),
	                                                Common::AlignUp(begin + bytes, TRACKER_PAGE_SIZE)) &&
	                   (m_early_released.empty() || !OverlapsUnpublished(begin, bytes)) &&
	                   !m_texture_cache.IsRegionGpuModified(begin, bytes) &&
	                   (m_unbounded_write_tick == 0 || m_scheduler.IsFree(m_unbounded_write_tick));
	if (!clean) {
		g_async_drain.redrained.fetch_add(1, std::memory_order_relaxed);
		ReadMemoryDrain(vaddr, size, is_write, trace);
		return;
	}
	g_async_drain.released.fetch_add(1, std::memory_order_relaxed);
	m_memory_tracker.UnmarkRegionAsGpuModified(begin, bytes);
	if (is_write) {
		m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	}
}

void BufferCache::ReadMemoryDrain(uint64_t vaddr, uint64_t size, bool is_write,
                                  ReadMemoryTrace& trace, AsyncDrain* async) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	HangWatchdog::Scope drain("buffer-readback-drain", vaddr, size, 0, 0, is_write);
	if (is_write && !IsRegionRegistered(vaddr, size)) {
		return;
	}
	// KYTY_CLEAN_PAGE: nothing of these pages is left to download.
	if (FaultPagesCurrent(Common::AlignDown(vaddr, TRACKER_PAGE_SIZE),
	                      Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE))) {
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
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

	// KYTY_READBACK_SPAN (reads): the registered buffers next to this one, within the aligned
	// window, share the drain (one wait instead of a fault and a drain each).
	std::vector<std::pair<Buffer*, GuestRange>> neighbours;
	if (!is_write && g_readback_span.On()) {
		const auto aligned_begin = Common::AlignDown(vaddr, WindowSize);
		const auto aligned_end   = aligned_begin + WindowSize;
		const auto owner_at      = [&](uint64_t at) -> Buffer* {
            const auto* next = m_page_table.Find(at >> PageTable::kPageBits);
            if (next == nullptr || !*next || IsBufferInvalid(*next)) {
                return nullptr;
            }
            auto& found = m_slot_buffers[*next];
            return &found != &buffer && found.IsInBounds(at, 1) ? &found : nullptr;
		};
		for (auto at = window_begin; at > aligned_begin;) {
			auto* next = owner_at(at - 1);
			if (next == nullptr) {
				break;
			}
			const auto begin = std::max(aligned_begin, next->CpuAddress());
			neighbours.push_back({next, {begin, at - begin}});
			at = begin;
		}
		for (auto at = window_end; at < aligned_end;) {
			auto* next = owner_at(at);
			if (next == nullptr) {
				break;
			}
			const auto end = std::min(aligned_end, next->CpuAddress() + next->Size());
			neighbours.push_back({next, {at, end - at}});
			at = end;
		}
	}
	trace.begin = window_begin;
	trace.size  = window_end - window_begin;
	std::vector<GuestRange> drained;
	const bool main_downloaded = DownloadBufferMemory(buffer, window_begin, window_end - window_begin);
	bool       downloaded      = main_downloaded;
	for (const auto& [next, range]: neighbours) {
		if (DownloadBufferMemory(*next, range.address, range.size)) {
			downloaded = true;
			drained.push_back(range);
		}
	}
	if (!neighbours.empty() && downloaded) {
		const auto count = g_readback_span_drains.fetch_add(1, std::memory_order_relaxed) + 1;
		if ((count & (count - 1)) == 0 && count <= 4096) {
			std::printf("Readback span: %" PRIu64 " drains took neighbouring buffers (KYTY_READBACK_SPAN)\n", count);
			std::fflush(stdout);
		}
	}
	if (downloaded) {
		trace.downloaded = true;
		if (async != nullptr) {
			// KYTY_ASYNC_DRAIN: submit the download and go on; the guest thread waits for it.
			async->tick         = m_scheduler.FlushObservable();
			async->window_begin = window_begin;
			async->window_end   = window_end;
			g_async_drain.issued.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		const auto                    tick = m_scheduler.CurrentTick();
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitDrain);
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		if (main_downloaded) {
			m_memory_tracker.UnmarkRegionAsGpuModified(window_begin, window_end - window_begin);
		}
		for (const auto& range: drained) {
			m_memory_tracker.UnmarkRegionAsGpuModified(range.address, range.size);
		}
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
	const auto live_window = static_cast<uint64_t>(g_side_window_live.Get());
	const auto window = live_window != 0 ? std::min(live_window, side.window) : side.window;
	const auto aligned  = Common::AlignDown(vaddr, window);
	auto       window_begin = std::max(aligned, buffer_begin);
	auto       window_end   = std::min(aligned + window, buffer_end);
	if (((page_begin | page_end | window_begin | window_end) % TRACKER_PAGE_SIZE) != 0) {
		return other(Profiler::FrameEvent::ReadbackSideOtherAlignment);
	}
	// KYTY_READBACK_SPAN: the registered buffers next to this one, within the aligned window.
	const auto owner_at = [&](uint64_t at) -> Buffer* {
		const auto* next = m_page_table.Find(at >> PageTable::kPageBits);
		if (next == nullptr || !*next || IsBufferInvalid(*next)) {
			return nullptr;
		}
		auto& found = m_slot_buffers[*next];
		return found.IsInBounds(at, 1) ? &found : nullptr;
	};
	bool spans = false;
	if (g_readback_span.On()) {
		auto wide_begin = window_begin;
		auto wide_end   = window_end;
		while (wide_begin > aligned) {
			const auto* next = owner_at(wide_begin - 1);
			if (next == nullptr) {
				break;
			}
			wide_begin = std::max(aligned, next->CpuAddress());
		}
		while (wide_end < aligned + side.window) {
			const auto* next = owner_at(wide_end);
			if (next == nullptr) {
				break;
			}
			wide_end = std::min(aligned + side.window, next->CpuAddress() + next->Size());
		}
		if ((wide_begin != window_begin || wide_end != window_end) &&
		    ((wide_begin | wide_end) % TRACKER_PAGE_SIZE) == 0) {
			window_begin = wide_begin;
			window_end   = wide_end;
			spans        = true;
		}
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
	// KYTY_READBACK_SPAN: the copies per source buffer (the window's buffers are all registered).
	struct SpanGroup {
		Buffer*                     source = nullptr;
		std::vector<vk::BufferCopy> copies;
		uint64_t                    lo = UINT64_MAX;
		uint64_t                    hi = 0;
	};
	std::vector<SpanGroup> groups;
	for (const auto& range: dirty) {
		bytes += range.size;
		if (!spans) {
			copies.emplace_back(buffer.Offset(range.address),
			                    staging_base + (range.address - chosen->begin), range.size);
			continue;
		}
		for (uint64_t at = range.address, range_end = range.address + range.size; at < range_end;) {
			auto* source = buffer.IsInBounds(at, 1) ? &buffer : owner_at(at);
			EXIT_IF(source == nullptr);
			const auto next  = std::min(range_end, source->CpuAddress() + source->Size());
			auto       group = std::find_if(groups.begin(), groups.end(),
			                                [&](const SpanGroup& g) { return g.source == source; });
			if (group == groups.end()) {
				groups.push_back({source, {}, UINT64_MAX, 0});
				group = std::prev(groups.end());
			}
			group->copies.emplace_back(source->Offset(at), staging_base + (at - chosen->begin),
			                           next - at);
			group->lo = std::min(group->lo, at);
			group->hi = std::max(group->hi, next);
			at        = next;
		}
	}
	if (spans && groups.size() == 1 && groups.front().source == &buffer) {
		// Only the faulting buffer's bytes are dirty: the plain recording below.
		copies.swap(groups.front().copies);
		groups.clear();
		spans = false;
	}
	if (spans) {
		for (const auto& group: groups) {
			copies.insert(copies.end(), group.copies.begin(), group.copies.end());
		}
		const auto count = g_readback_span_copies.fetch_add(1, std::memory_order_relaxed) + 1;
		if ((count & (count - 1)) == 0 && count <= 4096) {
			std::printf("Readback span: %" PRIu64 " side readbacks copied from more than the faulting "
			            "buffer (KYTY_READBACK_SPAN)\n",
			            count);
			std::fflush(stdout);
		}
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
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Side,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Side,
		                                           {.op   = GpuBreadcrumbs::OpSideReadback,
		                                            .arg0 = static_cast<uint32_t>(bytes),
		                                            .arg4 = chosen->begin,
		                                            .tick = producer}));
	}
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
	if (!spans) {
		before.buffer = buffer.Handle();
		before.offset = buffer.Offset(chosen->begin);
		before.size   = chosen->end - chosen->begin;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                        nullptr);
		// Nearby dirty ranges share one region (KYTY_READBACK_MERGE_GAP_KB): the staging window
		// mirrors the guest window, and completion writes back only readback->ranges.
		const auto regions =
		    MergeLinearCopies(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()));
		RecordReadbackRegions(copies, regions);
		command.copyBuffer(buffer.Handle(), side.staging->Handle(),
		                   static_cast<uint32_t>(regions.size()), regions.data());
	} else {
		// KYTY_READBACK_SPAN: a barrier and a copy command per source buffer.
		for (const auto& group: groups) {
			vk::BufferMemoryBarrier source_barrier = before;
			source_barrier.buffer                  = group.source->Handle();
			source_barrier.offset                  = group.source->Offset(group.lo);
			source_barrier.size                    = group.hi - group.lo;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
			                        &source_barrier, 0, nullptr);
			const auto regions = MergeLinearCopies(
			    group.copies, static_cast<uint64_t>(g_readback_merge_gap.Get()));
			RecordReadbackRegions(group.copies, regions);
			command.copyBuffer(group.source->Handle(), side.staging->Handle(),
			                   static_cast<uint32_t>(regions.size()), regions.data());
		}
	}
	vk::BufferMemoryBarrier after = before;
	after.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask           = vk::AccessFlagBits::eHostRead;
	after.buffer                  = side.staging->Handle();
	after.offset                  = staging_base;
	after.size                    = side.window;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                        {}, 0, nullptr, 1, &after, 0, nullptr);
	if (GpuBreadcrumbs::Active()) {
		GpuBreadcrumbs::Write(command, GpuBreadcrumbs::Queue::Side,
		                      GpuBreadcrumbs::Note(GpuBreadcrumbs::Queue::Side,
		                                           {.op = GpuBreadcrumbs::OpSideReadbackEnd, .tick = producer}));
	}
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
	readback->producer = producer;
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
		// wait is never submitted ahead of its signal; only up to the producer, after waiting
		// without queue_mutex for the work its batches read (SubmitDependency).
		const auto& progress = m_scheduler.GetMasterSemaphore().GetSubmissionProgress();
		if (progress != nullptr &&
		    progress->dispatched_tick.load(std::memory_order_acquire) < producer) {
			m_graphics.submission_queue.WaitPendingDependencies(progress.get(), producer);
			Common::LockGuard lock(m_graphics.queue_mutex);
			m_graphics.submission_queue.DrainThroughLocked(progress.get(), producer);
		}
		Common::LockGuard lock(m_graphics.side_queue_mutex);
		HangWatchdog::Scope native(
		    "vkQueueSubmit-side-readback",
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.side_queue)), value,
		    producer);
		NoteWatchdogSubmit(m_graphics.side_queue, submit);
		submit_result = m_graphics.side_queue.submit(1, &submit, nullptr);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ReadbackSideQueueCopies);
	} else {
		// Every tick older than the current recording was handed to the queue or to the
		// submission broker (drained here first), so the producer is submitted before this
		// copy waits on it. The current recording follows this copy in submission order.
		Common::LockGuard lock(m_graphics.queue_mutex);
		m_graphics.submission_queue.DrainPendingLocked();
		HangWatchdog::Scope native(
		    "vkQueueSubmit-shared-readback",
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics.queue)), value, producer);
		NoteWatchdogSubmit(m_graphics.queue, submit);
		submit_result = m_graphics.queue.submit(1, &submit, nullptr);
	}
	if (submit_result == vk::Result::eErrorDeviceLost) {
		// The fatal exit below has no breadcrumbs of its own: report the device fault and NV
		// checkpoints first (KYTY_DEVICE_FAULT_DIAGNOSTICS=1).
		DumpDeviceLossDiagnostics(m_graphics, value);
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

// KYTY_READ_FAULT_DIAG: what a publication of the diagnosed page wrote (its first 16).
static void DiagPrintPublication(const std::vector<GuestRange>& ranges, uint64_t begin, bool eager,
                                 const uint8_t* staged) {
	auto& state = ReadFaultDiag::Get();
	if (state.publishes.fetch_add(1, std::memory_order_relaxed) >= 16) {
		return;
	}
	const auto  page = ReadFaultDiag::Page();
	std::string text;
	uint64_t    bytes = 0;
	for (const auto& range: ranges) {
		const auto first = std::max(range.address, page);
		const auto last  = std::min(range.address + range.size, page + ReadFaultDiag::RangeSize());
		if (first >= last) {
			continue;
		}
		bytes += last - first;
		if (text.size() < 400) {
			text += fmt::format(" +0x{:x}:{}", first - page, last - first);
		}
	}
	std::string hex;
	for (const auto& range: ranges) {
		const auto first = std::max(range.address, page);
		const auto last  = std::min(range.address + range.size, page + ReadFaultDiag::RangeSize());
		for (auto a = first; a < last && hex.size() < 3 * 64; a++) {
			hex += fmt::format("{:02x} ", staged[a - begin]);
		}
		if (!hex.empty()) {
			break;
		}
	}
	std::printf("ReadFaultDiag publish %s: %" PRIu64 " page bytes in%s | first bytes %s\n",
	            eager ? "eager" : "side", bytes, text.c_str(), hex.c_str());
}

bool BufferCache::CompleteSideReadback(SideReadback& readback) {
	HangWatchdog::Scope wait("readback-publication", readback.begin, readback.value, readback.end,
	                         0, readback.eager);
	HangWatchdog::DebugDelay("readback", readback.begin);
	if (auto* diag = ReadFaultDiag::t_fault;
	    diag != nullptr && !diag->waited && !readback.done.load(std::memory_order_acquire)) {
		// KYTY_READ_FAULT_DIAG: split this reader's wait by polling each condition in turn.
		diag->waited    = true;
		auto&      master = m_scheduler.GetMasterSemaphore();
		const auto tick   = readback.eager ? readback.value : readback.producer;
		diag->wait_tick   = tick;
		diag->byte_known  = true;
		for (const auto& range: readback.ranges) {
			diag->byte_dirty |= diag->fault_vaddr >= range.address &&
			                    diag->fault_vaddr < range.address + range.size;
		}
		auto mark = ReadFaultDiag::NowNs();
		const auto lap = [&](uint64_t& part) {
			const auto now = ReadFaultDiag::NowNs();
			part += now - mark;
			mark = now;
		};
		if (const auto& progress = master.GetSubmissionProgress(); progress != nullptr) {
			while (progress->dispatched_tick.load(std::memory_order_acquire) < tick &&
			       !readback.done.load(std::memory_order_acquire)) {
				std::this_thread::yield();
			}
		}
		lap(diag->dispatch_ns);
		while (!readback.done.load(std::memory_order_acquire)) {
			master.Refresh();
			if (master.KnownGpuTick() >= tick) {
				break;
			}
			std::this_thread::yield();
		}
		lap(diag->gpu_ns);
		if (!readback.eager && !readback.done.load(std::memory_order_acquire)) {
			m_side->Wait(readback.value);
		}
		lap(diag->copy_ns);
		diag->phases_end_ns = mark;
	}
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
	if (ReadFaultDiag::Covers(readback.begin, readback.end - readback.begin)) {
		DiagPrintPublication(readback.ranges, readback.begin, readback.eager, staged);
	}
	if (g_read_fault_published.On()) {
		NoteReadFaultPublication(readback.ranges, m_memory_tracker.Frame());
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

void BufferCache::NoteCpFault(uint64_t vaddr, uint64_t handler_ns) {
	if (!WatchRange::CpFaultWritersEnabled()) {
		return;
	}
	// The newest recorded writer of the address.
	const CpFaultWriter* writer = nullptr;
	const size_t         count  = m_cp_fault_writers.size();
	for (size_t i = 0; i < count && writer == nullptr; i++) {
		const auto& w = m_cp_fault_writers[(m_cp_fault_writer_next + count - 1 - i) % count];
		if (vaddr >= w.vaddr && vaddr < w.vaddr + w.size) {
			writer = &w;
		}
	}
	const uint64_t key = (writer != nullptr ? writer->hash : 0) ^ ((vaddr >> 16u) * 0x9e3779b97f4a7c15ull);
	auto& stats = m_cp_fault_stats[key];
	if (stats.faults == 0) {
		stats.sample      = vaddr;
		stats.kind        = writer != nullptr ? writer->kind : "none";
		stats.tick_delta  = writer != nullptr ? m_scheduler.CurrentTick() - writer->tick : UINT64_MAX;
		stats.reader      = WatchRange::CurrentOpHash();
		stats.reader_kind = WatchRange::CurrentOpKind();
	}
	stats.faults++;
	stats.ns += handler_ns;
	(void)stats;
	const auto now = std::chrono::steady_clock::now();
	if (m_cp_fault_print.time_since_epoch().count() == 0) {
		m_cp_fault_print = now;
		return;
	}
	if (now - m_cp_fault_print < std::chrono::seconds(10)) {
		return;
	}
	m_cp_fault_print = now;
	std::vector<std::pair<uint64_t, CpFaultStats>> rows(m_cp_fault_stats.begin(), m_cp_fault_stats.end());
	std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
	std::printf("CpFaultWriters 10s: %zu groups\n", rows.size());
	for (size_t i = 0; i < rows.size() && i < 16; i++) {
		const auto& s = rows[i].second;
		std::printf("  CpFaultWriters: %llu faults %.1f ms sample 0x%016llx written by %s 0x%016llx "
		            "(%lld ticks before the first fault), last op %s 0x%016llx\n",
		            static_cast<unsigned long long>(s.faults), static_cast<double>(s.ns) * 1e-6,
		            static_cast<unsigned long long>(s.sample), s.kind,
		            static_cast<unsigned long long>(rows[i].first ^ ((s.sample >> 16u) * 0x9e3779b97f4a7c15ull)),
		            static_cast<long long>(s.tick_delta), s.reader_kind,
		            static_cast<unsigned long long>(s.reader));
	}
	std::fflush(stdout);
	m_cp_fault_stats.clear();
}

std::string BufferCache::DescribeNewestWriter(uint64_t vaddr) {
	const size_t count = m_cp_fault_writers.size();
	for (size_t i = 0; i < count; i++) {
		const auto& w = m_cp_fault_writers[(m_cp_fault_writer_next + count - 1 - i) % count];
		if (w.size != 0 && vaddr >= w.vaddr && vaddr < w.vaddr + w.size) {
			return fmt::format("{} 0x{:016x} ({} ticks ago, 0x{:x}+0x{:x})", w.kind != nullptr ? w.kind : "?", w.hash,
			                   m_scheduler.CurrentTick() - w.tick, w.vaddr, w.size);
		}
	}
	return "none";
}

void BufferCache::WatchMetadata(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	// A bound on the watches kept: past it they start again (every range then inspects once more).
	constexpr size_t MaxWatches = 4096;
	std::scoped_lock lock(m_metadata_watch_mutex);
	if (m_metadata_watches.size() >= MaxWatches) {
		m_metadata_watches.clear();
		m_metadata_watch_span = 0;
	}
	// Other watches over the range go: the range is current as inspected now.
	for (auto it = m_metadata_watches.lower_bound(vaddr >= m_metadata_watch_span ? vaddr - m_metadata_watch_span : 0);
	     it != m_metadata_watches.end() && it->first < vaddr + size;) {
		if (it->second.end > vaddr && it->first != vaddr) {
			it = m_metadata_watches.erase(it);
		} else {
			++it;
		}
	}
	m_metadata_watches[vaddr] = {vaddr + size, false};
	m_metadata_watch_span     = std::max(m_metadata_watch_span, size);
	m_metadata_watch_count.store(m_metadata_watches.size(), std::memory_order_relaxed);
}

bool BufferCache::MetadataUnwritten(uint64_t vaddr, uint64_t size) {
	{
		std::scoped_lock lock(m_metadata_watch_mutex);
		const auto       found = m_metadata_watches.find(vaddr);
		if (found == m_metadata_watches.end() || found->second.end != vaddr + size || found->second.written) {
			return false;
		}
	}
	return !m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::NoteMetadataWrite(uint64_t vaddr, uint64_t size) {
	if ((m_metadata_watch_count.load(std::memory_order_relaxed) == 0 &&
	     m_cp_value_watch_count.load(std::memory_order_relaxed) == 0) ||
	    size == 0) {
		return;
	}
	const auto       end = vaddr + size < vaddr ? UINT64_MAX : vaddr + size;
	std::scoped_lock lock(m_metadata_watch_mutex);
	for (auto it = m_metadata_watches.lower_bound(vaddr >= m_metadata_watch_span ? vaddr - m_metadata_watch_span : 0);
	     it != m_metadata_watches.end() && it->first < end; ++it) {
		if (it->second.end > vaddr) {
			it->second.written = true;
		}
	}
	for (auto it = m_cp_value_watches.lower_bound(vaddr >= m_cp_value_watch_span ? vaddr - m_cp_value_watch_span : 0);
	     it != m_cp_value_watches.end() && it->first < end; ++it) {
		if (it->second.end > vaddr) {
			it->second.written = true;
		}
	}
}

void BufferCache::WatchCpValue(uint64_t vaddr, uint64_t size) {
	if (size == 0 || !GuestRange {vaddr, size}.Valid()) {
		return;
	}
	constexpr size_t MaxWatches = 1024;
	std::scoped_lock lock(m_metadata_watch_mutex);
	if (m_cp_value_watches.size() >= MaxWatches) {
		m_cp_value_watches.clear();
		m_cp_value_watch_span = 0;
	}
	// Older watches over the range go: this write is the bytes' newest value now.
	for (auto it = m_cp_value_watches.lower_bound(vaddr >= m_cp_value_watch_span ? vaddr - m_cp_value_watch_span : 0);
	     it != m_cp_value_watches.end() && it->first < vaddr + size;) {
		if (it->second.end > vaddr && it->first != vaddr) {
			it = m_cp_value_watches.erase(it);
		} else {
			++it;
		}
	}
	m_cp_value_watches[vaddr] = {vaddr + size, false};
	m_cp_value_watch_span     = std::max(m_cp_value_watch_span, size);
	m_cp_value_watch_count.store(m_cp_value_watches.size(), std::memory_order_relaxed);
}

bool BufferCache::CpValueUnwritten(uint64_t vaddr, uint64_t size) {
	{
		std::scoped_lock lock(m_metadata_watch_mutex);
		const auto       found = m_cp_value_watches.find(vaddr);
		if (found == m_cp_value_watches.end() || found->second.end != vaddr + size || found->second.written) {
			return false;
		}
	}
	return !m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::NoteBufferContentWrite(uint64_t vaddr, uint64_t size) {
	if (WatchRange::CpFaultWritersEnabled()) [[unlikely]] {
		if (m_cp_fault_writers.size() < 8192) {
			m_cp_fault_writers.resize(8192);
		}
		m_cp_fault_writers[m_cp_fault_writer_next] = {vaddr, size, m_scheduler.CurrentTick(),
		                                              WatchRange::CurrentOpHash(),
		                                              WatchRange::CurrentOpKind()};
		m_cp_fault_writer_next = (m_cp_fault_writer_next + 1) % m_cp_fault_writers.size();
	}
	NoteMetadataWrite(vaddr, size);
	if (m_side == nullptr) {
		return;
	}
	m_write_ticks.Assign(vaddr, size, m_scheduler.CurrentTick());
	ReadFaultDiag::NoteGpuWrite(vaddr, size, m_scheduler.CurrentTick(),
	                            reinterpret_cast<uint64_t>(__builtin_return_address(0)));
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

bool BufferCache::ShaderWriteRetickEnabled() {
	return g_shader_write_retick.On();
}

void BufferCache::RetagShaderWrite(uint64_t vaddr, uint64_t size, uint64_t preparation_tick) {
	if (vaddr != 0 && size != 0 && preparation_tick != m_scheduler.CurrentTick()) {
		NoteBufferContentWrite(vaddr, size);
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
	// Encoded with the recorder (no drain): the completion runner reads the staging only after
	// this tick completed, which the recorder's submission of it precedes.
	const auto              native = command.HelperSink();
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
	const auto regions =
	    MergeLinearCopies(copies, static_cast<uint64_t>(g_readback_merge_gap.Get()));
	RecordReadbackRegions(copies, regions);
	native.copyBuffer(buffer.Handle(), side.eager_staging->Handle(),
	                  static_cast<uint32_t>(regions.size()), regions.data());
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
	// Diagnostic only: allow a guest reader to acquire this publication before the CP flushes
	// its producer tick, opening the issue-to-submit race window without changing ordering.
	HangWatchdog::DebugDelay("eager-issue", page);
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

// usa-ratt/KytyPS5-GT7 d3260d78e.
uint32_t BufferCache::MakeResident(uint64_t vaddr, uint64_t size) {
	if (size == 0 || vaddr > UINT64_MAX - size) {
		return 0;
	}
	uint32_t   made  = 0;
	const auto first = vaddr & ~(CACHING_PAGESIZE - 1u);
	const auto last  = (vaddr + size - 1u) & ~(CACHING_PAGESIZE - 1u);
	for (uint64_t page = first; page <= last; page += CACHING_PAGESIZE) {
		if (!GuestRange {page, CACHING_PAGESIZE}.Valid()) {
			break;
		}
		const auto* owner = m_page_table.Find(page >> PageTable::kPageBits);
		if (owner != nullptr && *owner) {
			// Read through the page table alone, it would age out of the cache.
			TouchBuffer(m_slot_buffers[*owner]);
			continue;
		}
		if (Libs::LibKernel::Memory::ClampRangeSizeQuiet(page, CACHING_PAGESIZE) !=
		    CACHING_PAGESIZE) {
			continue;
		}
		(void)ObtainBuffer(page, CACHING_PAGESIZE, false, false);
		made++;
	}
	return made;
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
	const BufferLookupStats::PartScope stats_scope(BufferLookupStats::Part::CreateBuffer);
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
	// An upload of CPU-dirty bytes over watched metadata changes it (MetadataUnwritten).
	if ((m_metadata_watch_count.load(std::memory_order_relaxed) != 0 ||
	     m_cp_value_watch_count.load(std::memory_order_relaxed) != 0) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		NoteMetadataWrite(vaddr, size);
	}
	// KYTY_TRACKER_RELAXED_QUERIES: a read-only synchronization of a range without a CPU-dirty
	// page (hot pages are CPU-dirty too) collects and uploads nothing. Texel reads also download
	// GPU-written images; the BDA hot-pass verification scans in full.
	if (!is_written && (!is_texel_buffer || TexelRelaxedSyncEnabled()) &&
	    (stats == nullptr || stats->verify_fault_epoch == 0) && RelaxedNothingToUpload(vaddr, size)) {
		// A texel read still takes the bytes of GPU-written images that start at its address.
		return is_texel_buffer ? SynchronizeBufferFromImage(buffer, vaddr, size) : false;
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
	const bool reservation_made = reserved != nullptr;
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
	// KYTY_DEFER_PROTECTS (pageManager.h): the collection of a read upload that may go to the DMA
	// worker defers the write protections of the pages it cleans. The worker applies them before
	// it copies the pages; every copy of their guest bytes made on this thread applies them first
	// (below when the upload stays on this thread, UploadCopies, StageUploadDma), and the scope's
	// end applies what is left.
	const int  defer_mode     = PageManager::DeferProtectsMode();
	const bool defer_protects = defer_mode != 0 && !is_written && UploadBatchEnabled() &&
	                            m_upload_dma != nullptr && UploadDmaHostCopyEnabled() &&
	                            !UploadDmaVerify() && m_staging_buffer.IsCoherent();
	const PageManager::DeferProtectsScope defer_scope(defer_protects);
	bool                                  force_dma = false;
	const auto upload = [&]() noexcept {
		if (defer_protects) {
			PageManager::DeferProtectsScope::Stop(); // the collection is over
		}
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
			// KYTY_DEFER_PROTECTS=all: an upload with deferred protections goes to the worker
			// whatever its size.
			force_dma = defer_protects && (defer_mode & 2) != 0 && PageManager::HasPendingProtects();
			const bool defer_host = !is_written && UploadBatchEnabled() && m_upload_dma != nullptr &&
			                        UploadDmaHostCopyEnabled() && !UploadDmaVerify() &&
			                        m_staging_buffer.IsCoherent() &&
			                        (total_size >= m_upload_dma->MinBytes() || force_dma);
			if (!defer_host) {
				PageManager::ApplyPendingProtects(); // this thread copies the pages next
			}
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
	if (reservation_made && m_staging_buffer.ReservationPending()) {
		m_staging_buffer.AbandonReservation(); // nothing staged in it is read by a copy
	}
	// KYTY_BDA_LIVE_PAGES (bufferCacheLive.cpp): an upload (the cache copy is current again) or a
	// written binding (GPU-dirty bytes must not be served from guest memory, I-W3) ends the redirect
	// of every caching page it touches. KYTY_BDA_LIVE_BATCH (I-B1): a read upload only settles the
	// pages; their entries go back in one batch at the next pre-pass.
	// I-S1 (rtskip DESIGN.md 7): only after the staging reservation above is committed. A restore writes
	// its page-table entries through the same staging ring (StreamBuffer::Copy maps at the uncommitted
	// offset), so before the commit it overwrote the guest bytes staged for this upload: the cache
	// buffer received page-table entries instead (Astro Bot pirate level: a header whose loop bound
	// became a device address hung a one-workgroup dispatch until the device was lost).
	if (m_bda_live_count.load(std::memory_order_relaxed) != 0 &&
	    (is_written || !copies.empty() || !late_copies.empty())) {
		if (is_written) {
			RestoreLivePages(vaddr, size, LiveRestore::Written);
		} else {
			SettleLivePages(vaddr, size);
		}
	}
	if (source || late_source) {
		if (InplaceStats::Enabled()) {
			InplaceStats::NoteUploadBytes(total_size + late_size, copies.size() + late_copies.size());
			const auto path = stats != nullptr ? (stats->reach ? InplaceStats::UploadPath::BdaReach : InplaceStats::UploadPath::BdaSync)
			                  : is_written     ? InplaceStats::UploadPath::Written
			                  : is_texel_buffer ? InplaceStats::UploadPath::Texel
			                                    : InplaceStats::UploadPath::Read;
			for (size_t i = 0; i < copies.size(); i++) {
				const auto& copy = copies[i];
				InplaceStats::NoteUpload(buffer.CpuAddress() + copy.dstOffset, copy.size);
				// Copies past guest_copies were added by CollectHotPages (changed hot pages).
				InplaceStats::NoteUploadDetail(path, i >= guest_copies, buffer.CpuAddress() + copy.dstOffset,
				                               copy.size);
			}
			for (const auto& copy: late_copies) {
				InplaceStats::NoteUpload(buffer.CpuAddress() + copy.dstOffset, copy.size);
				InplaceStats::NoteUploadDetail(path, false, buffer.CpuAddress() + copy.dstOffset, copy.size);
			}
		}
		if (stats != nullptr) {
			stats->upload_bytes += total_size + late_size;
			stats->upload_copies += copies.size() + late_copies.size();
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::BufferUploadBytes, total_size + late_size);
		// The overlay's "Buffer transfers": every path, the staged reservation included.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size + late_size);
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
			source = StageUploadDma(source, copies, &host_copies, force_dma);
			command.RequestUploadCopy(source, buffer.Handle(), copies);
			buffer.MarkContentWritten();
			if (m_upload_batch_depth == 0 && !command.KeepUploadsForDispatch()) {
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
                                       std::vector<UploadHostCopy>* host_copies, bool allow_small) {
	// Staging bytes UploadCopies left to the DMA worker, written here when the copy is not
	// staged after all (the graphics copies then read the staging ring directly).
	const auto write_host_copies = [host_copies] {
		PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: before the bytes are read
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
	// KYTY_DEFER_PROTECTS: the deferred protections go to the worker with the host copies.
	auto       protects    = PageManager::TakePendingProtects();
	const auto ring_offset = m_upload_dma->Stage(source, begin, size, host_copies, &protects,
	                                             allow_small && !protects.empty());
	if (!ring_offset.has_value()) {
		PageManager::ApplyProtects(protects, false);
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

// KYTY_UPLOAD_UNMAPPED_ZERO=1 (default off; the GT7 preset; Senaxx 3e8c01a09 via
// usa-ratt/KytyPS5-GT7): an upload of a buffer range the guest no longer has mapped (AMM maps
// streaming textures in blocks; a bindless texture resolve uploaded one byte past the 32 KiB
// block the game had mapped: an access violation in UploadCopies) reads zeros instead. Off by
// default: one range query per guest copy.
static bool UploadUnmappedZeroEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UPLOAD_UNMAPPED_ZERO");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// True (and `destination` zeroed, logged) when [address, address + size) is not all mapped.
static bool ZeroUnmappedUpload(uint8_t* destination, uint64_t address, uint64_t size) {
	if (!UploadUnmappedZeroEnabled() || LibKernel::Memory::IsCommittedSpan(address, size)) {
		return false;
	}
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
		LOGF("BufferCache: upload of unmapped guest memory 0x%016" PRIx64 " size=0x%" PRIx64
		     " reads as zero\n",
		     address, size);
	}
	std::memset(destination, 0, static_cast<size_t>(size));
	return true;
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
			if (alias == nullptr && index < guest_copies) {
				PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: copied here
			}
			if (alias != nullptr) {
				deferred->push_back({mapped + copy.srcOffset, static_cast<const uint8_t*>(alias),
				                     copy.size, buffer.CpuAddress() + copy.dstOffset});
			} else if (index >= guest_copies ||
			           !ZeroUnmappedUpload(mapped + copy.srcOffset,
			                               buffer.CpuAddress() + copy.dstOffset, copy.size)) {
				std::memcpy(mapped + copy.srcOffset, source_of(index, copy), copy.size);
			}
			copy.srcOffset += base_offset;
		}
		m_staging_buffer.Commit();
		return m_staging_buffer.Handle();
	}

	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         vk::BufferUsageFlagBits::eTransferSrc, total_size);
	PageManager::ApplyPendingProtects(); // KYTY_DEFER_PROTECTS: copied here
	for (size_t index = 0; index < copies.size(); index++) {
		const auto& copy = copies[index];
		if (index < guest_copies &&
		    ZeroUnmappedUpload(temporary->Mapped().data() + copy.srcOffset,
		                       buffer.CpuAddress() + copy.dstOffset, copy.size)) {
			continue;
		}
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

bool BufferCache::QueryUploadSnapshot(const MemoryTracker& tracker, uint64_t vaddr, uint64_t size,
                                      MemoryTracker::DirtyState& state, bool& cpu_only) {
	cpu_only = g_cpu_only_query.On();
	if (!cpu_only) {
		return tracker.QueryDirtyRelaxed(vaddr, size, state);
	}
	state = {};
	return tracker.QueryCpuDirtyRelaxed(vaddr, size, state.cpu);
}

bool BufferCache::RelaxedNothingToUpload(uint64_t vaddr, uint64_t size) {
	if (!m_relaxed_queries || !GuestGpu::IsGpuThread()) {
		return false;
	}
	const bool                verify    = RelaxedVerifyMode() != 0;
	const auto                signature = verify ? m_memory_tracker.RangeSignature(vaddr, size) : 0;
	MemoryTracker::DirtyState state;
	bool                     cpu_only = false;
	if (!QueryUploadSnapshot(m_memory_tracker, vaddr, size, state, cpu_only) || state.cpu) {
		return false;
	}
	if (verify) {
		const auto locked = m_memory_tracker.QueryDirty(vaddr, size);
		if (!VerifyRelaxedSnapshot(vaddr, size, state, locked, signature, cpu_only) || locked.cpu) {
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
                                        uint64_t                         signature, bool cpu_only) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::TrackerRelaxedVerifyChecks);
	if (relaxed.cpu == locked.cpu && (cpu_only || relaxed.gpu == locked.gpu)) {
		return true;
	}
	// Other threads only make pages CPU-dirty and publish GPU-dirty ones, and every such change
	// advances the range's mutation serials first.
	const bool forbidden = (relaxed.cpu && !locked.cpu) ||
	                       (!cpu_only && !relaxed.gpu && locked.gpu);
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
	if (m_inplace != nullptr) {
		if (is_written) {
			m_inplace->Demote(vaddr, size);
		} else if (!is_texel_buffer) {
			std::pair<Buffer*, uint64_t> result;
			if (TryInplaceBinding(vaddr, size, result)) {
				return result;
			}
		}
	}
	if (!is_written && m_binding_memo != nullptr && size < BindingMemoMaxSize &&
	    GuestGpu::IsGpuThread()) {
		if (!is_texel_buffer) {
			return ObtainReadBinding(vaddr, size, id);
		}
		if (TexelBindingMemoEnabled()) {
			// KYTY_TEXEL_RELAXED_SYNC=2: the memo decides the tracker part; a stream copy is a
			// guest-byte copy as before (ObtainBufferNow streams small texel reads too).
			auto result = ObtainReadBinding(vaddr, size, id);
			if (result.first != &m_stream_buffer) {
				(void)SynchronizeBufferFromImage(*result.first, vaddr, size);
			}
			return result;
		}
	}
	return ObtainBufferNow(vaddr, size, is_written, is_texel_buffer, id, nullptr);
}

bool BufferCache::TryInplaceBinding(uint64_t vaddr, uint64_t size, std::pair<Buffer*, uint64_t>& result) {
	if (!m_inplace->AllCandidates(vaddr, size) || !GuestGpu::IsGpuThread()) {
		return false;
	}
	// Guest memory must hold the current bytes: no GPU-written bytes the cache still owns, no
	// readback or backing publication about to land, no image that wrote the range.
	MemoryTracker::DirtyState state;
	if (!m_memory_tracker.QueryDirtyRelaxed(vaddr, size, state)) {
		state = m_memory_tracker.QueryDirty(vaddr, size);
	}
	const uint64_t page_begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const uint64_t page_end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	if (state.gpu || HasPendingBackingPublication(vaddr, size) || OverlapsPendingSideReadback(page_begin, page_end) ||
	    (!m_early_released.empty() && OverlapsUnpublished(vaddr, size)) ||
	    m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	Buffer*  buffer = nullptr;
	uint64_t offset = 0;
	if (!m_inplace->Bind(vaddr, size, buffer, offset)) {
		return false;
	}
	result = {buffer, offset};
	return true;
}

void BufferCache::InplaceBeforeCpWrite(uint64_t vaddr, uint64_t size) {
	if (m_inplace != nullptr) {
		m_inplace->BeforeCpWrite(vaddr, size);
	}
	LivePagesBeforeCpWrite(vaddr, size); // KYTY_BDA_LIVE_PAGES (I-C1): DMA_DATA
}

void BufferCache::PrefetchReadBinding(uint64_t vaddr, uint64_t size, bool at_find) const noexcept {
	if (m_binding_memo == nullptr || g_binding_batch_prefetch.Get() != (at_find ? 2 : 1)) {
		return;
	}
	const auto* slot =
	    &m_binding_memo[static_cast<size_t>(BindingMemoHash(vaddr, size) >> m_binding_memo_shift)];
#if defined(__clang__) || defined(__GNUC__)
	__builtin_prefetch(slot, 0, 3);
#elif defined(_M_X64) || defined(_M_IX86)
	_mm_prefetch(reinterpret_cast<const char*>(slot), _MM_HINT_T0);
#endif
}

// KYTY_BINDING_MEMO_STATS=1 (diagnostic, default off): ObtainReadBinding outcomes, one console line
// every 10 s (GPU thread only).
namespace {
struct BindingMemoStats {
	uint64_t calls = 0, no_region = 0, empty = 0, other_key = 0, signature = 0, guard = 0,
	         epoch = 0, stream_hits = 0, cached_hits = 0, slow_stream = 0;
	std::chrono::steady_clock::time_point last {};
};
bool BindingMemoStatsEnabled() {
	static std::atomic<int64_t> on_once {Common::OnceUnset};
	const auto on = static_cast<bool>(Common::Once(on_once, [] {
		const char* value = std::getenv("KYTY_BINDING_MEMO_STATS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}));
	return on;
}
BindingMemoStats& GetBindingMemoStats() {
	static BindingMemoStats stats;
	return stats;
}
void PrintBindingMemoStats() {
	auto& st = GetBindingMemoStats();
	if ((st.calls & 0xffffu) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (st.last.time_since_epoch().count() == 0) {
		st.last = now;
		return;
	}
	if (now - st.last < std::chrono::seconds(10)) {
		return;
	}
	const double c = static_cast<double>(std::max<uint64_t>(1, st.calls)) / 100.0;
	std::printf("BindingMemoStats 10s: calls %llu hits stream %.1f%% cached %.1f%% | misses: no-region "
	            "%.1f%% empty %.1f%% other-key %.1f%% signature %.1f%% guard %.1f%% epoch %.1f%% | slow "
	            "path stream result %.1f%%\n",
	            static_cast<unsigned long long>(st.calls), static_cast<double>(st.stream_hits) / c,
	            static_cast<double>(st.cached_hits) / c, static_cast<double>(st.no_region) / c,
	            static_cast<double>(st.empty) / c, static_cast<double>(st.other_key) / c,
	            static_cast<double>(st.signature) / c, static_cast<double>(st.guard) / c,
	            static_cast<double>(st.epoch) / c, static_cast<double>(st.slow_stream) / c);
	std::fflush(stdout);
	st      = {};
	st.last = now;
}
} // namespace

// KYTY_CP_COMMIT_STATS=steps: the memo lookup's time by outcome (diagnostic only).
std::pair<Buffer*, uint64_t> BufferCache::ObtainReadBinding(uint64_t vaddr, uint64_t size,
                                                            BufferId id) {
	if (!CommitStats::StepsEnabled()) [[likely]] {
		return ObtainReadBindingMemo(vaddr, size, id);
	}
	m_last_read_hit   = false;
	const auto start  = CommitStats::Now();
	const auto result = ObtainReadBindingMemo(vaddr, size, id);
	const bool hit    = m_last_read_hit;
	CommitStats::AddSub(hit ? CommitStats::Sub::ReadHit : CommitStats::Sub::ReadMiss,
	                    CommitStats::Now() - start);
	CommitStats::AddItems(hit ? CommitStats::Item::ReadHits : CommitStats::Item::ReadMisses, 1);
	return result;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainReadBindingMemo(uint64_t vaddr, uint64_t size,
                                                                BufferId id) {
	// KYTY_BINDING_EPOCH_MEMO (bufferCache.h).
	// KYTY_CP_BINDING_MEMO_PREFETCH (default off, live): issue the read hint before the tracker
	// query, so its region/serial loads can overlap the random memo-line fetch. This is a CPU cache
	// hint only: every key, signature, epoch, tick and structure check below still runs.
	m_binding_hot_enabled = g_binding_hot_memo.On();
	if (m_binding_hot_enabled) {
		const auto generation = g_binding_hot_generation.load(std::memory_order_relaxed);
		if (generation == UINT64_MAX || m_binding_hot_generation != generation) {
			for (auto& entry: m_binding_hot_memo) {
				entry.kind = BindingMemoKind::Empty;
			}
			m_binding_hot_generation = generation;
		}
	}
	auto& full = BindingMemoSlot(vaddr, size);
	auto& hot  = BindingHotMemoSlot(vaddr, size);
	const bool hot_key = m_binding_hot_enabled && hot.kind != BindingMemoKind::Empty &&
	                     hot.vaddr == vaddr && hot.size == size;
	auto& memo = hot_key ? hot : full;
	if (!hot_key && g_binding_memo_prefetch.On()) {
#if defined(__clang__) || defined(__GNUC__)
		__builtin_prefetch(&memo, 0, 3);
#elif defined(_M_X64) || defined(_M_IX86)
		_mm_prefetch(reinterpret_cast<const char*>(&memo), _MM_HINT_T0);
#endif
	}
	const auto sig_start = CommitStats::StepsEnabled() ? CommitStats::Now() : 0;
	const auto epoch  = SyncEpoch::Current();
	const auto before = m_memory_tracker.RangeSignature(vaddr, size);
	if (sig_start != 0) {
		CommitStats::AddSub(CommitStats::Sub::ReadSig, CommitStats::Now() - sig_start);
	}
	const bool key    = memo.kind != BindingMemoKind::Empty && memo.vaddr == vaddr && memo.size == size;
	BindingMemoStats* stats = nullptr;
	if (BindingMemoStatsEnabled()) [[unlikely]] {
		stats = &GetBindingMemoStats();
		stats->calls++;
	}
	using LookupOutcome = BufferLookupStats::Outcome;
	using LookupMiss    = BufferLookupStats::Miss;
	if (BufferLookupStats::Enabled() && BufferLookupStats::KeysEnabled()) [[unlikely]] {
		BufferLookupStats::NoteKey(vaddr, size, m_memory_tracker.Frame());
	}
	if (before == 0 || !key) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSlot);
		if (stats != nullptr) {
			(before == 0 ? stats->no_region
			             : memo.kind == BindingMemoKind::Empty ? stats->empty : stats->other_key)++;
		}
		BufferLookupStats::AddMiss(before == 0                            ? LookupMiss::NoRegion
		                           : memo.kind == BindingMemoKind::Empty ? LookupMiss::Empty
		                                                                  : LookupMiss::OtherKey);
	} else if (memo.signature != before) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissSignature);
		if (stats != nullptr) {
			stats->signature++;
		}
		BufferLookupStats::AddMiss(LookupMiss::Signature);
	} else if (const bool stream = memo.kind == BindingMemoKind::Stream;
	           memo.guard != (stream ? m_scheduler.CurrentTick()
	                                 : m_bda_structure_epoch.load(std::memory_order_acquire))) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoMissGuard);
		if (stats != nullptr) {
			stats->guard++;
		}
		BufferLookupStats::AddMiss(LookupMiss::Guard);
	} else {
		bool cross = false;
		if (memo.epoch != epoch) {
			// KYTY_BINDING_MEMO_CROSS_EPOCH (bufferCache.h): only a cache-buffer memo, and only
			// once no page of the range is CPU-dirty while the signature still holds. The relaxed
			// mirror answers "CPU-dirty" exactly on this thread (only it clears those bits); the
			// locked query also waits for a transition that advanced the serial but has not
			// changed its bits yet.
			MemoryTracker::DirtyState relaxed;
			bool                     cpu_only = false;
			// KYTY_BINDING_CROSS_SETTLED (live, default 0): a memo whose cross-epoch check already
			// passed under the signature it still has skips the two CPU-dirty queries. Every change
			// of a region's dirty bits advances its serial first, under the region lock, and that
			// check ended with the locked query and an unchanged signature: no transition was in
			// progress then, and an unchanged signature means none started since. The final
			// signature comparison still runs. 2 (verify) runs the queries too and counts the
			// checks where they would reject.
			const auto settled_mode = g_binding_cross_settled.Get();
			const bool settled      = settled_mode != 0 && !stream && m_binding_memo_cross && memo.settled;
			if (settled && settled_mode == 1) {
				m_cross_epoch_totals.fast++;
				cross = m_memory_tracker.RangeSignature(vaddr, size) == before;
			} else {
				cross = !stream && m_binding_memo_cross &&
				        QueryUploadSnapshot(m_memory_tracker, vaddr, size, relaxed, cpu_only) &&
				        !relaxed.cpu && !m_memory_tracker.IsRegionCpuModified(vaddr, size) &&
				        m_memory_tracker.RangeSignature(vaddr, size) == before;
				if (settled) {
					m_cross_epoch_totals.verify_checks++;
					if (!cross && m_memory_tracker.RangeSignature(vaddr, size) == before) {
						const auto count = ++m_cross_epoch_totals.verify_mismatches;
						if (count <= 16) {
							std::fprintf(stderr,
							             "BindingCrossSettled: a settled memo is CPU-dirty with its "
							             "signature unchanged: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
							             vaddr, size);
						}
					}
					if ((m_cross_epoch_totals.verify_checks & 0xfffffu) == 0) {
						std::printf("BindingCrossSettled verify: %" PRIu64 " checks, %" PRIu64
						            " mismatches\n",
						            m_cross_epoch_totals.verify_checks,
						            m_cross_epoch_totals.verify_mismatches);
						std::fflush(stdout);
					}
				}
			}
			if (cross) {
				memo.settled = true;
			}
			if (!cross) {
				if (stats != nullptr) {
					stats->epoch++;
				}
				BufferLookupStats::AddMiss(LookupMiss::Epoch);
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
			m_last_read_hit = true;
			if (m_binding_hot_enabled) {
				if (hot_key) {
					m_binding_memo_totals.hot_hits++;
				} else {
					hot = memo;
				}
			}
			std::pair<Buffer*, uint64_t> hit {nullptr, memo.offset};
			if (stats != nullptr) {
				(stream ? stats->stream_hits : stats->cached_hits)++;
				PrintBindingMemoStats();
			}
			if (stream) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoStreamHits);
				m_binding_memo_totals.stream_hits++;
				hit.first = &m_stream_buffer;
				BufferLookupStats::SetOutcome(LookupOutcome::MemoStream);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoCachedHits);
				m_binding_memo_totals.cached_hits++;
				const BufferLookupStats::PartScope touch_scope(BufferLookupStats::Part::ReadTouch);
				hit.first = memo.buffer;
				if (!g_binding_memo_touch.On()) {
					TouchBuffer(*hit.first);
				} else {
					// KYTY_BINDING_MEMO_TOUCH (bufferCache.h).
					if (memo.touched != m_gc_tick) {
						TouchBuffer(*hit.first);
						memo.touched = m_gc_tick;
					} else {
						m_binding_memo_totals.touch_skips++;
					}
				}
				BufferLookupStats::SetOutcome(cross ? LookupOutcome::MemoCross : LookupOutcome::MemoHit);
			}
			if (m_binding_memo_verify != 0) {
				const auto copy = memo;
				return VerifyBindingHit(copy, hit, id, cross);
			}
			return hit;
		}
	}
	// KYTY_LAZY_FIND_BUFFER (RenderExecutor::PrepareGraphicsBindings): the miss finds the buffer.
	if (IsBufferInvalid(id) && g_binding_lean_miss.On()) {
		id = FindBuffer(vaddr, size);
	}
	if (std::pair<Buffer*, uint64_t> lean {};
	    g_binding_lean_miss.On() && TryLeanReadMiss(vaddr, size, epoch, before, id, lean)) {
		return lean;
	}
	BufferId                     obtained {};
	BufferLookupStats::PartScope now_scope(BufferLookupStats::Part::ReadNow);
	const auto                   result = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
	now_scope.Stop();
	if (stats != nullptr) {
		stats->slow_stream += result.first == &m_stream_buffer ? 1u : 0u;
		PrintBindingMemoStats();
	}
	BufferLookupStats::SetOutcome(result.first == &m_stream_buffer ? LookupOutcome::MissStream
	                                                               : LookupOutcome::MissCached);
	const BufferLookupStats::PartScope record_scope(BufferLookupStats::Part::ReadRecord);
	RecordBinding(vaddr, size, epoch, before, result, obtained);
	return result;
}

BufferCache::KeepStamp BufferCache::KeepBegin() const noexcept {
	if (!RegionManager::AnySerialEnabled() || m_binding_memo == nullptr) {
		return {};
	}
	// The token first: a serial advance after it is seen by KeepValid.
	KeepStamp stamp;
	stamp.token     = RegionManager::AnySerial();
	stamp.epoch     = SyncEpoch::Current();
	stamp.structure = m_bda_structure_epoch.load(std::memory_order_acquire);
	return stamp;
}

bool BufferCache::KeepCapture(KeptRead& kept, const KeepStamp& stamp, uint64_t vaddr, uint64_t size,
                              BufferId id, const std::pair<Buffer*, uint64_t>& result) {
	kept.stamp.token = 0;
	if (stamp.token == 0 || result.first == nullptr || result.first == &m_stream_buffer ||
	    size >= BindingMemoMaxSize || !GuestGpu::IsGpuThread()) {
		return false;
	}
	// The entry the lookup used (ObtainReadBindingMemo's choice of the hot or the full slot): a
	// cache-buffer result taken or recorded under the stamp's epoch and buffer structure.
	const auto& full    = BindingMemoSlot(vaddr, size);
	const auto& hot     = BindingHotMemoSlot(vaddr, size);
	const bool  hot_key = m_binding_hot_enabled && hot.kind != BindingMemoKind::Empty && hot.vaddr == vaddr &&
	                     hot.size == size;
	const auto& memo    = hot_key ? hot : full;
	if (memo.kind != BindingMemoKind::Cached || memo.vaddr != vaddr || memo.size != size ||
	    memo.epoch != stamp.epoch || memo.guard != stamp.structure || memo.buffer != result.first ||
	    memo.offset != result.second) {
		return false;
	}
	kept.stamp   = stamp;
	kept.vaddr   = vaddr;
	kept.size    = size;
	kept.id      = id;
	kept.buffer  = result.first;
	kept.offset  = result.second;
	kept.touched = m_gc_tick; // the lookup touched it (or recorded it after obtaining it)
	return true;
}

bool BufferCache::KeepValid(KeptRead& kept, uint64_t vaddr, uint64_t size, BufferId id) {
	if (kept.stamp.token == 0 || kept.vaddr != vaddr || kept.size != size || kept.id != id ||
	    RegionManager::AnySerial() != kept.stamp.token || SyncEpoch::Current() != kept.stamp.epoch ||
	    m_bda_structure_epoch.load(std::memory_order_acquire) != kept.stamp.structure ||
	    m_scheduler.Current().IsInvalid()) {
		return false;
	}
	// The normal lookup must still find the entry: another range's record may have replaced it, and
	// a lookup that misses decides again (over hot pages written since, too). The slot it would
	// read (ObtainReadBindingMemo's hot or full choice, including the hot table's reset on a new
	// generation) holds the key, the kept result, the epoch and the guard; its signature is the one
	// the kept lookup checked or recorded, as no region serial moved.
	const auto  generation = g_binding_hot_generation.load(std::memory_order_relaxed);
	const auto& full       = BindingMemoSlot(vaddr, size);
	const auto& hot        = BindingHotMemoSlot(vaddr, size);
	const bool  hot_key    = g_binding_hot_memo.On() && generation != UINT64_MAX &&
	                     m_binding_hot_generation == generation && hot.kind != BindingMemoKind::Empty &&
	                     hot.vaddr == vaddr && hot.size == size;
	const auto& memo       = hot_key ? hot : full;
	if (memo.kind != BindingMemoKind::Cached || memo.vaddr != vaddr || memo.size != size ||
	    memo.epoch != kept.stamp.epoch || memo.guard != kept.stamp.structure || memo.buffer != kept.buffer ||
	    memo.offset != kept.offset) {
		return false;
	}
	// A hit's touch: within one GC tick a repeated touch changes nothing (KYTY_BINDING_MEMO_TOUCH).
	if (kept.touched != m_gc_tick) {
		TouchBuffer(*kept.buffer);
		kept.touched = m_gc_tick;
	}
	return true;
}

bool BufferCache::TryLeanReadMiss(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
                                  BufferId id, std::pair<Buffer*, uint64_t>& result) {
	if (before == 0 || !m_relaxed_queries || IsBufferInvalid(id) ||
	    !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		return false;
	}
	const BufferLookupStats::PartScope lean_scope(BufferLookupStats::Part::ReadNow);
	bool cpu_dirty = true;
	if (!m_memory_tracker.QueryCpuDirtyRelaxed(vaddr, size, cpu_dirty) || cpu_dirty ||
	    m_memory_tracker.RangeSignature(vaddr, size) != before) {
		return false;
	}
	const auto structure = m_bda_structure_epoch.load(std::memory_order_acquire);
	if (structure == UINT64_MAX) {
		return false;
	}
	auto& buffer = m_slot_buffers[id];
	result       = {&buffer, buffer.Offset(vaddr)};
	m_binding_memo_totals.lean_misses++;
	BufferLookupStats::SetOutcome(BufferLookupStats::Outcome::MissLean);
	if (g_binding_lean_miss.Get() >= 2) {
		// The normal path decides (and synchronizes) the binding.
		m_binding_memo_totals.lean_verify_checks++;
		BufferId   obtained {};
		const auto normal = ObtainBufferNow(vaddr, size, false, false, id, &obtained);
		if (normal != result) {
			if (m_memory_tracker.RangeSignature(vaddr, size) != before) {
				m_binding_memo_totals.lean_verify_races++;
			} else {
				m_binding_memo_totals.lean_verify_mismatches++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::fprintf(stderr,
					             "BindingLeanMissVerify: the normal path bound another buffer or "
					             "offset: addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
					             vaddr, size);
				}
				if (g_binding_lean_miss.Get() == 3) {
					EXIT("BindingLeanMissVerify: the normal path bound another buffer or offset: "
					     "addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
					     vaddr, size);
				}
			}
		}
		result = normal;
		RecordBinding(vaddr, size, epoch, before, normal, obtained);
		return true;
	}
	TouchBuffer(buffer);
	// RecordBinding's cache-buffer entry, with the signature the decision was taken under.
	BindingMemo entry;
	entry.vaddr     = vaddr;
	entry.size      = static_cast<uint32_t>(size);
	entry.epoch     = epoch;
	entry.offset    = result.second;
	entry.signature = before;
	entry.guard     = structure;
	entry.buffer    = &buffer;
	entry.touched   = m_gc_tick;
	entry.kind      = BindingMemoKind::Cached;
	BindingMemoSlot(vaddr, size) = entry;
	if (m_binding_hot_enabled) {
		BindingHotMemoSlot(vaddr, size) = entry;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoRecords);
	m_binding_memo_totals.records++;
	return true;
}

void BufferCache::RecordBinding(uint64_t vaddr, uint64_t size, uint64_t epoch, uint64_t before,
                                const std::pair<Buffer*, uint64_t>& result, BufferId id) {
	if (before == 0) {
		return; // a tracker region did not exist yet (its creation made the range CPU-dirty)
	}
	BindingMemo entry;
	entry.vaddr  = vaddr;
	entry.size   = static_cast<uint32_t>(size);
	entry.epoch  = epoch;
	entry.offset = result.second;
	entry.buffer = result.first;
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
		// The normal path touched the buffer in this tick (ObtainBufferNow).
		entry.touched   = m_gc_tick;
		entry.kind      = BindingMemoKind::Cached;
	}
	BindingMemoSlot(vaddr, size) = entry;
	if (m_binding_hot_enabled) {
		BindingHotMemoSlot(vaddr, size) = entry;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoRecords);
	m_binding_memo_totals.records++;
}

std::pair<Buffer*, uint64_t> BufferCache::VerifyBindingHit(const BindingMemo&           memo,
                                                           std::pair<Buffer*, uint64_t> hit,
                                                           BufferId id, bool cross) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::BindingEpochMemoVerifyChecks);
	m_binding_memo_totals.verify_checks++;
	const auto vaddr = memo.vaddr;
	const uint64_t size = memo.size;
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

bool StreamCopyDefer::Enabled() {
	static const bool enabled = ParseEnvU64("KYTY_STREAM_COPY_DEFER", 0) != 0;
	return enabled;
}

void BufferCache::PruneDeferredCopies(const CommandRecorder& recorder) {
	const auto consumed = recorder.RingConsumed();
	while (m_deferred_head < m_deferred.size() && m_deferred[m_deferred_head].position <= consumed) {
		m_deferred_head++;
	}
	if (m_deferred_head == m_deferred.size()) {
		m_deferred.clear();
		m_deferred_head = 0;
		m_deferred_lo   = UINT64_MAX;
		m_deferred_hi   = 0;
	}
}

bool BufferCache::TryDeferStreamCopy(uint8_t* mapped, uint64_t vaddr, uint64_t size) {
	if (!StreamCopyDefer::Enabled() || m_range_memo_verify != 0 || m_binding_memo_verify != 0 ||
	    !m_stream_buffer.IsCoherent() || !CpCommit::Enabled(CpCommit::Part::StreamRead) ||
	    !GuestGpu::IsGpuThread()) {
		return false;
	}
	const auto& command  = m_scheduler.Current();
	auto*       recorder = command.Recorder();
	if (recorder == nullptr || recorder->GetMode() != CommandRecorder::Mode::Thread) {
		return false;
	}
	if (m_deferred_recorder != recorder) {
		// A different recorder (never in practice): the pending copies of the old one drain first.
		if (m_deferred_recorder != nullptr && !m_deferred.empty()) {
			m_deferred_recorder->Drain(nullptr, false);
		}
		m_deferred.clear();
		m_deferred_head     = 0;
		m_deferred_lo       = UINT64_MAX;
		m_deferred_hi       = 0;
		m_deferred_recorder = recorder;
	}
	if (!CommandSink(command).hostCopy(mapped, vaddr, size)) {
		return false; // a direct window: the recorder is drained, copy now
	}
	g_defer_cache = this;
	PruneDeferredCopies(*recorder);
	m_deferred.push_back({vaddr, vaddr + size, recorder->RingWritePosition()});
	m_deferred_lo = std::min(m_deferred_lo, vaddr);
	m_deferred_hi = std::max(m_deferred_hi, vaddr + size);
	return true;
}

void StreamCopyDefer::NoteGpuOnlyWrite(uint64_t address, uint64_t size) {
	if (auto* inplace = g_inplace_heap; inplace != nullptr && size != 0) {
		inplace->BeforeCpWrite(address, size);
	}
	BufferCache::LiveCpWriteHook(address, size); // KYTY_BDA_LIVE_PAGES (I-C1)
	if (auto* dma = g_write_drain_dma; dma != nullptr && size != 0 && UploadDmaWriteDrainEnabled() &&
	                                   GuestGpu::IsGpuThread()) {
		(void)dma->DrainHostCopies(address, size, false);
	}
	NoteGuestWriteStream(address, size);
}

bool StreamCopyDefer::LabelWaitsForHostCopies() {
	auto* dma = g_write_drain_dma;
	return dma != nullptr && UploadDmaWriteDrainEnabled() && GuestGpu::IsGpuThread() &&
	       dma->LabelsWaitForHostCopies();
}

void StreamCopyDefer::NoteFrameBoundary() {
	if (auto* dma = g_write_drain_dma; dma != nullptr && UploadDmaWriteDrainEnabled() &&
	                                   GuestGpu::IsGpuThread()) {
		(void)dma->DrainHostCopies(0, 1, true, true);
	}
}

void StreamCopyDefer::NoteGuestWrite(uint64_t address, uint64_t size) {
	// KYTY_UPLOAD_DMA_WRITE_DRAIN (uploadDma.h): the guest may observe this write (a label) and
	// then rewrite memory an earlier upload still has to read, so every queued host copy is made
	// first.
	if (auto* inplace = g_inplace_heap; inplace != nullptr && size != 0) {
		inplace->BeforeCpWrite(address, size);
	}
	BufferCache::LiveCpWriteHook(address, size); // KYTY_BDA_LIVE_PAGES (I-C1)
	if (auto* dma = g_write_drain_dma; dma != nullptr && size != 0 && UploadDmaWriteDrainEnabled() &&
	                                   GuestGpu::IsGpuThread()) {
		(void)dma->DrainHostCopies(address, size, true);
	}
	NoteGuestWriteStream(address, size);
}

void StreamCopyDefer::NoteGuestWriteStream(uint64_t address, uint64_t size) {
	auto* cache = g_defer_cache;
	if (cache == nullptr || cache->m_deferred.empty() || size == 0 || !GuestGpu::IsGpuThread()) {
		return;
	}
	const auto end = address + size;
	if (end <= cache->m_deferred_lo || address >= cache->m_deferred_hi) {
		return;
	}
	auto* recorder = cache->m_deferred_recorder;
	cache->PruneDeferredCopies(*recorder);
	bool overlap = false;
	for (size_t i = cache->m_deferred_head; i < cache->m_deferred.size() && !overlap; i++) {
		const auto& copy = cache->m_deferred[i];
		overlap          = address < copy.end && copy.begin < end;
	}
	if (!overlap) {
		return;
	}
	static std::atomic<uint64_t> drains {0};
	if (drains.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("StreamCopyDefer: a CP write [0x%016" PRIx64 ", +0x%" PRIx64
		            ") overlaps a pending stream copy: recorder drained\n",
		            address, size);
		std::fflush(stdout);
	}
	recorder->Drain(nullptr, false);
	cache->m_deferred.clear();
	cache->m_deferred_head = 0;
	cache->m_deferred_lo   = UINT64_MAX;
	cache->m_deferred_hi   = 0;
}

// KYTY_STREAM_COPY_REUSE=1 (live, default off; based on Senaxx/KytyPS5 8bdb5b43f, adapted to the
// op stream: CommandProcessor::Process and ExecuteOp end the epoch): streamCopyMemo.h.
static Live::Switch g_stream_copy_reuse("KYTY_STREAM_COPY_REUSE", Live::ParseDefaultOff);

// Copies made and reused while the switch is on (command-processor thread), printed at every
// power of two of the copies made.
static void NoteStreamCopyReuse(bool reused, uint64_t size) {
	static uint64_t made = 0, reuses = 0, reused_bytes = 0;
	if (reused) {
		reuses++;
		reused_bytes += size;
		return;
	}
	if (++made >= 4096 && (made & (made - 1)) == 0) {
		std::printf("StreamCopyReuse: %" PRIu64 " copies made, %" PRIu64 " reused (%.1f%% of bindings, %.1f MB "
		            "not copied) (KYTY_STREAM_COPY_REUSE)\n",
		            made, reuses, 100.0 * static_cast<double>(reuses) / static_cast<double>(made + reuses),
		            static_cast<double>(reused_bytes) / 1.0e6);
		std::fflush(stdout);
	}
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
		// KYTY_STREAM_COPY_REUSE: the same range copied earlier in this tick and epoch.
		const bool     reuse      = stream && g_stream_copy_reuse.On();
		const uint64_t reuse_tick = reuse ? m_scheduler.CurrentTick() : 0;
		if (reuse) {
			if (uint64_t offset = 0; m_stream_copy_memo.Find(vaddr, size, reuse_tick, offset)) {
				NoteStreamCopyReuse(true, size);
				return {&m_stream_buffer, offset};
			}
		}
		if (stream) {
			const auto alignment = std::max<uint64_t>(
			    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
			auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
			bool read = false;
			if (mapped != nullptr && StreamDirectReadEnabled()) {
				std::memcpy(mapped, reinterpret_cast<const void*>(vaddr), size);
				read = true;
			} else if (mapped != nullptr && TryDeferStreamCopy(mapped, vaddr, size)) {
				read = true; // KYTY_STREAM_COPY_DEFER: the recorder copies the bytes
			} else if (mapped != nullptr) {
				// KYTY_CP_COMMIT=streamread: the reservation is committed only after a successful
				// read, so the direct read may leave partial bytes in it when it fails.
				const uint64_t probe_t0 = StreamCopyProbe() ? __rdtsc() : 0;
				read = CpCommit::Enabled(CpCommit::Part::StreamRead)
				           ? Libs::LibKernel::Memory::TryReadBackingDirect(vaddr, mapped, size)
				           : Libs::LibKernel::Memory::TryReadBacking(vaddr, mapped, size);
				if (probe_t0 != 0) {
					StreamCopyProbeAdd(size, __rdtsc() - probe_t0, vaddr);
				}
			}
			if (read) {
				InplaceStats::NoteStream(vaddr, size);
				m_stream_buffer.Commit();
				if (reuse) {
					m_stream_copy_memo.Note(vaddr, size, reuse_tick, offset);
					NoteStreamCopyReuse(false, size);
				}
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
		WatchRange::NoteWriter("bind", vaddr, size);
		if (!m_gpu_modified_ranges.Contains(vaddr, size)) {
			CleanVerdict::Invalidate(vaddr, size, Coherence::Source::BufferDirtyAdd);
			WatchRange::Note("GPU-modified range #1", vaddr, size);
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
		WatchRange::NoteWriter("written-range", range.address, range.size);
		if (!m_gpu_modified_ranges.Contains(range.address, range.size)) {
			CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtyAdd);
			WatchRange::Note("GPU-modified range #2", range.address, range.size);
			m_gpu_modified_ranges.Add(range.address, range.size);
		}
		NoteBufferContentWrite(range.address, range.size);
		ForgetKnownFills(range.address, range.size);
		HangTrace::NoteGpuWrite(range.address, range.size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainImageStagingBuffer(uint64_t size) {
	auto [mapped, offset] = m_staging_buffer.Map(size, 16);
	if (mapped != nullptr) return {&m_staging_buffer, offset};
	// The ring (fixed small ring or the default one) must not impose a maximum guest image size
	// (upstream 9bcf80bf6: a 576 MiB image exceeded the default 512 MiB ring). Keep this one-off
	// source alive until the submission using it has completed, as UploadCopies does for buffers.
	auto temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0,
	                                         AllFlags, size);
	auto* source = temporary.get();
	m_scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {source, 0};
}

// An image whose memory is only partly mapped: streaming games (Miles Morales' AMM pool) map the
// resident part of a texture and leave the rest of its range unmapped. The unmapped bytes read as
// zeros, as a GPU read of an unmapped page does on the console, instead of ending the emulator.
bool ReadPartlyUnmappedImageBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (!Libs::LibKernel::Memory::CopySparseMemory(vaddr, data, size)) {
		return false;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("Image backing 0x%016" PRIx64 "+0x%" PRIx64
		     " is partly unmapped: its unmapped bytes read as zero\n",
		     vaddr, size);
	}
	return true;
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	// KYTY_IMAGE_UPLOAD_GUEST_BYTES=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7
	// 95b1f059f): an image upload over a cached buffer takes the guest bytes, staged, when the GPU
	// wrote none of them (and none is on its way back), instead of copying from the buffer. The
	// guest bytes are what the CPU wrote; the buffer is only as fresh as its own tracking, which
	// missed writes next to other cached data (GT7's light grids went stale that way).
	static const bool guest_bytes = [] {
		const char* value = std::getenv("KYTY_IMAGE_UPLOAD_GUEST_BYTES");
		return value != nullptr && value[0] == '1';
	}();
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	bool        stage_guest = false;
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			stage_guest = guest_bytes && !IsRegionGpuModified(vaddr, size) &&
			              !HasGpuDirtyBytes(vaddr, size) && !HasPendingBackingPublication(vaddr, size);
			if (!stage_guest) {
				TouchBuffer(buffer);
				(void)SynchronizeBuffer(buffer, vaddr, size, false, false, nullptr, "image-source");
				return {&buffer, buffer.Offset(vaddr)};
			}
			static std::atomic<uint64_t> staged {0};
			if (const auto n = staged.fetch_add(1, std::memory_order_relaxed) + 1;
			    n <= 4 || (n & (n - 1u)) == 0) {
				LOGF("BufferCache: image upload 0x%012" PRIx64 "+0x%" PRIx64
				     " takes its guest bytes, not the cached buffer over it (%" PRIu64
				     " so far; KYTY_IMAGE_UPLOAD_GUEST_BYTES)\n",
				     vaddr, size, n);
			}
		}
	}
	if (!stage_guest && IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}
	return StageGuestBytesForImage(vaddr, size);
}

std::pair<Buffer*, uint64_t> BufferCache::StageGuestBytesForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}

	auto [source, stage_offset] = ObtainImageStagingBuffer(size);
	if (source == nullptr) EXIT("BufferCache: failed to read mapped guest image backing\n");
	auto* staging = source->Mapped().data() + stage_offset;
	if (!Libs::LibKernel::Memory::TryReadBacking(vaddr, staging, size) &&
	    !Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, staging, size) &&
	    !ReadPartlyUnmappedImageBacking(vaddr, staging, size)) {
		// KYTY_IMAGE_UNMAPPED_ZERO=1 (default 0; PPSA26344 preset): an image range that runs past the
		// guest's reserved memory reads page by page: what the guest has (mapped or reserved) is
		// copied, the rest reads as zero, instead of ending the emulator. A streamed texture's chain
		// stores its largest levels last; when they are not loaded their pages are not even reserved,
		// while the levels the T# views are (the case of Erlou93/KytyPS5-Yotei's streamed-view
		// sub-chain: Ghost of Yotei's terrain snow mask).
		static const bool zero = [] {
			const char* value = std::getenv("KYTY_IMAGE_UNMAPPED_ZERO");
			return value != nullptr && value[0] == '1';
		}();
		if (!zero) {
			EXIT("BufferCache: failed to read mapped guest image backing 0x%016" PRIx64 "+0x%" PRIx64 "\n",
			     vaddr, size);
		}
		constexpr uint64_t Page   = 0x4000;
		uint64_t           copied = 0;
		for (uint64_t offset = 0; offset < size;) {
			const auto bytes = std::min(size - offset, Page - ((vaddr + offset) & (Page - 1u)));
			if (Libs::LibKernel::Memory::CopySparseMemory(vaddr + offset, staging + offset, bytes)) {
				copied += bytes;
			} else {
				std::memset(staging + offset, 0, bytes);
			}
			offset += bytes;
		}
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("KYTY_IMAGE_UNMAPPED_ZERO: image backing 0x%016" PRIx64 "+0x%" PRIx64
			     " runs past guest memory: 0x%" PRIx64 " bytes copied, the rest reads as zero\n",
			     vaddr, size, copied);
		}
	}
	if (source == &m_staging_buffer) m_staging_buffer.Commit();
	else source->Flush(0, size);
	return {source, stage_offset};
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value, true);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	const bool exact_meta = m_texture_cache.ClearMetaRange(vaddr, size);
	// KYTY_META_CLEAR_COVER: a fill that starts below a metadata surface and covers it clears it.
	const uint32_t covered_meta = MetaClearCoverEnabled() ? m_texture_cache.ClearMetaCovered(vaddr, size) : 0u;
	if (const auto traced = MetaTraceAddress();
	    traced != 0 && traced >= vaddr && traced - vaddr < size && MetaTraceLine()) {
		LOGF("MetaTrace: fill base 0x%" PRIx64 " size 0x%" PRIx64 " value 0x%08x exact %d covered %u\n", vaddr,
		     size, value, exact_meta ? 1 : 0, covered_meta);
	}
	if (!IsRegionGpuModified(vaddr, size)) {
		ForgetFillsUnder(vaddr, size); // KYTY_PENDING_FILLS
		// KYTY_FILL_INVALIDATE_FIRST=1 (default off; TheCruZ/KytyPS5-GTA d1046f83): invalidate the
		// cached buffers and images of the whole range at once (what each page's write fault does)
		// when any page of it is watched, then write without faulting. GTA V clears DCC metadata
		// with uniform fills every frame: one fault per tracked page before.
		static constexpr auto invalidate_first_init = [] {
			const auto* value = std::getenv("KYTY_FILL_INVALIDATE_FIRST");
			return value != nullptr && std::strcmp(value, "1") == 0;
		};
		static constinit Common::StaticOnce<std::remove_cvref_t<decltype(invalidate_first_init())>> invalidate_first_once;
		const bool& invalidate_first = invalidate_first_once.Get(invalidate_first_init);
		if (invalidate_first) {
			(void)Libs::LibKernel::Memory::PrepareHostOverwrite(vaddr, size);
		}
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
	dst->Fill(dst_offset, size, value, true);
	RecordKnownFill(vaddr, size, value);
}

void BufferCache::FillMappedSpans(std::span<const std::pair<uint64_t, uint64_t>> spans,
                                  uint32_t                                       value) {
	for (const auto& [vaddr, size]: spans) {
		if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || !GuestRange {vaddr, size}.Valid()) {
			EXIT("BufferCache: invalid mapped fill span 0x%016" PRIx64 "+0x%" PRIx64 "\n", vaddr, size);
		}
		// As FillBuffer's GPU path: a fill on the GPU timeline, after the work recorded before it.
		const bool preserve_images = ImageWritebackOnGpuWriteEnabled();
		if (!preserve_images) {
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
		HangTrace::ScopedGpuWriteKind trace_kind(HangTrace::GpuWriteKind::Fill);
		auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
		if (preserve_images) {
			m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
		}
		dst->Fill(dst_offset, size, value, true);
		RecordKnownFill(vaddr, size, value);
	}
}

// KYTY_CP_WRITE_THROUGH (live, default 0): each command-processor write decides on its own.
static Live::Switch g_cp_write_through("KYTY_CP_WRITE_THROUGH", Live::ParseDefaultOff);

bool BufferCache::CpWriteThroughEnabled() {
	return g_cp_write_through.Get() != 0;
}

bool BufferCache::TryCpWriteThrough(uint64_t vaddr, const void* data, uint64_t size) {
	if (vaddr == 0 || data == nullptr || size == 0 || (vaddr & 3u) != 0 || (size & 3u) != 0 || size > 65536 ||
	    !GuestRange {vaddr, size}.Valid() || !GuestGpu::IsGpuThread()) {
		return false;
	}
	// Only a write that would fault on a GPU-modified page; any other write stays a plain store.
	if (!m_memory_tracker.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	if (HasGpuDirtyBytes(vaddr, size) || HasPendingBackingPublication(vaddr, size) ||
	    OverlapsPendingSideReadback(vaddr, vaddr + size) || m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	// The GPU copy: the buffer that owns the page must hold the whole range (or no buffer at all).
	Buffer*    buffer = nullptr;
	const auto first  = m_page_table.Find(vaddr >> PageTable::kPageBits);
	const auto last   = m_page_table.Find((vaddr + size - 1) >> PageTable::kPageBits);
	const bool owned_first = first != nullptr && *first;
	const bool owned_last  = last != nullptr && *last;
	if (owned_first || owned_last) {
		if (!owned_first || !owned_last || *first != *last) {
			return false;
		}
		buffer = &m_slot_buffers[*first];
		if (!buffer->IsInBounds(vaddr, size)) {
			return false;
		}
	}
	if (!Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
		return false;
	}
	if (buffer != nullptr) {
		KYTY_GPU_OP_SITE("buffercache.cp_write_through");
		const auto offset  = buffer->Offset(vaddr);
		auto&      command = m_scheduler.Current();
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
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
		                       nullptr, 1, &before, 0, nullptr);
		native.updateBuffer(buffer->Handle(), offset, size, data);
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 0,
		                       nullptr, 1, &after, 0, nullptr);
	}
	// A CPU-side image of these bytes (none GPU-modified, checked above) is stale now, as after the fault.
	m_texture_cache.InvalidateMemory(vaddr, size);
	NoteMetadataWrite(vaddr, size);
	return true;
}

bool BufferCache::TryWriteDataGpu(uint64_t vaddr, const uint32_t* data, uint64_t size) {
	if (vaddr == 0 || data == nullptr || size == 0 || (vaddr & 3u) != 0 || (size & 3u) != 0 ||
	    size > 65536 || !GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	// Only bytes owned by recorded GPU work need ordering on the GPU timeline; everything else
	// keeps the CPU write (which faults into the usual invalidation when tracked).
	// KYTY_WRITE_DATA_GPU_PAGE (default 1; 0 off; live): the 4 KiB pages around the range count
	// too. A CPU write into a page the GPU owns faults and drains the GPU even when the GPU wrote
	// other bytes of that page (Miles Morales writes 208-byte constants next to 1448 bytes a
	// compute shader wrote: ~3 drains and ~12 ms of the command processor per frame).
	const bool page_granular = g_write_data_gpu_page.Get() != 0;
	const auto page_begin    = vaddr & ~uint64_t {TRACKER_PAGE_SIZE - 1};
	const auto page_end      = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~uint64_t {TRACKER_PAGE_SIZE - 1};
	const auto check_begin   = page_granular ? page_begin : vaddr;
	const auto check_size    = page_granular ? page_end - page_begin : size;
	if (!HasGpuDirtyBytes(check_begin, check_size) &&
	    !HasPendingBackingPublication(check_begin, check_size)) {
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
	// KYTY_WRITE_DATA_GPU_RELEASE=1 (default off; adapted from TheCruZ/KytyPS5-GTA f4d1d0c7): from
	// this point of the stream the bytes are the ones just written. They also go to the backing
	// store and leave the GPU-written ranges, so reads around the GPU's writes take them from
	// there instead of reading the buffer back (GTA V Fidelity: WRITE_DATA records next to a
	// dispatch's writes, ~5 ms of GPU waits per frame). A later GPU write marks them again. Not
	// while a download, side readback or image over them is pending (as StoreInPlace): publishing
	// it would put older bytes over the new ones.
	static const bool release = [] {
		const auto* value = std::getenv("KYTY_WRITE_DATA_GPU_RELEASE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (release) {
		const auto page_first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
		const auto page_last  = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
		if (!HasPendingBackingPublication(vaddr, size) && !OverlapsPendingSideReadback(page_first, page_last) &&
		    (m_early_released.empty() || !OverlapsUnpublished(vaddr, size)) &&
		    !m_texture_cache.IsRegionGpuModified(vaddr, size) && m_gpu_modified_ranges.Intersects(vaddr, size)) {
			StreamCopyDefer::NoteGuestWrite(vaddr, size);
			if (Libs::LibKernel::Memory::TryWriteBacking(vaddr, data, size)) {
				CleanVerdict::Invalidate(vaddr, size, Coherence::Source::BufferDirtySubtract);
				m_gpu_modified_ranges.Subtract(vaddr, size);
				Coherence::NoteContentWrite(vaddr, size, Coherence::Source::CpWrite);
				Profiler::CountFrameEvent(Profiler::FrameEvent::WriteDataGpuReleased);
				Common::CruzStats::Add(Common::CruzStats::Counter::WriteDataReleased);
			}
		}
	}
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

bool BufferCache::PendingFillsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PENDING_FILLS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool BufferCache::FillOfNewImage(uint64_t vaddr, uint64_t size, uint32_t& value, bool take) {
	if (!PendingFillsEnabled() || !GuestGpu::IsGpuThread() || size == 0 ||
	    !GuestRange {vaddr, size}.Valid() || !m_has_known_fills.load(std::memory_order_acquire)) {
		return false;
	}
	std::scoped_lock lock(m_known_fill_mutex);
	const auto       known = KnownFillLocked(vaddr, size);
	if (!known) {
		return false;
	}
	// The part of the image the fill wrote on the GPU (what is mapped now): still GPU-written, or
	// something wrote over it unseen (its buffer was collected, the pages are no longer tracked).
	const auto mapped = Libs::LibKernel::Memory::ClampRangeSizeQuiet(vaddr, size);
	if (mapped != 0 && !m_gpu_modified_ranges.Contains(vaddr, mapped)) {
		ForgetKnownFillsLocked(vaddr, size);
		return false;
	}
	value = *known;
	if (take) {
		// The rest of the fill stays for other images.
		ForgetKnownFillsLocked(vaddr, size);
	}
	return true;
}

void BufferCache::ForgetFillsUnder(uint64_t vaddr, uint64_t size) {
	if (PendingFillsEnabled() && size != 0) {
		ForgetKnownFills(vaddr, size);
	}
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
	if (const auto traced = MetaTraceAddress();
	    traced != 0 && dst_memory && traced >= dst_vaddr && traced - dst_vaddr < size && MetaTraceLine()) {
		LOGF("MetaTrace: copy dst 0x%" PRIx64 " src 0x%" PRIx64 " size 0x%" PRIx64 " (no metadata clear)\n",
		     dst_vaddr, src_vaddr, size);
	}
	// KYTY_ALIAS_BYTES: bytes a GPU-modified image owns without starting at the source (or at the
	// destination) are not in guest memory either; the GPU path moves them into the buffer.
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size) &&
	    (!TextureCache::AliasBytesEnabled() ||
	     (!m_texture_cache.IsRegionGpuModified(src_vaddr, size) &&
	      !m_texture_cache.IsRegionGpuModified(dst_vaddr, size)))) {
		ForgetFillsUnder(dst_vaddr, size); // KYTY_PENDING_FILLS
		if (!g_cpu_copy_page_skip.On()) {
			std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr), size);
		} else if ((dst_vaddr >= src_vaddr ? dst_vaddr - src_vaddr : src_vaddr - dst_vaddr) < size) {
			// Page-by-page copying would corrupt later source bytes of an overlapping copy.
			std::memmove(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr), size);
		} else {
			for (uint64_t at = 0; at < size;) {
				const auto bytes = std::min(TRACKER_PAGE_SIZE - (dst_vaddr + at) % TRACKER_PAGE_SIZE, size - at);
				auto* to = reinterpret_cast<void*>(dst_vaddr + at);
				const auto* from = reinterpret_cast<const void*>(src_vaddr + at);
				// Only a successful clean-backing comparison can suppress a write. Dirty GPU
				// ownership is checked above; an unmapped/failed proof takes the original copy.
				if (LibKernel::Memory::CompareGpuCleanBacking(dst_vaddr + at, from, bytes) !=
				    LibKernel::Memory::BackingCompare::Equal) {
					std::memcpy(to, from, bytes);
				}
				at += bytes;
			}
		}
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
	if (src_memory && src->is_deleted) {
		// Obtaining the destination created a buffer that joined (and deleted) the source's. The
		// deleted buffer is freed once the current tick completes, and that synchronization may
		// already have submitted the tick the join was recorded in: copy from the joined buffer.
		std::tie(src, src_offset) = ObtainBuffer(src_vaddr, size, false, true, BufferId {});
		EXIT_IF(src->is_deleted);
	}
	// KYTY_DMA_BARRIER_BATCH: nothing is recorded through an earlier native handle after this copy.
	dst->CopyFrom(command, *src, src_offset, dst_offset, size, vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	              vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite, true);
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
		WatchRange::Note("GPU-modified range #3", begin, size);
		WatchRange::NoteWriter("image-bytes", begin, size);
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

std::pair<Buffer*, uint64_t> BufferCache::DebugPeekBuffer(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return {nullptr, 0};
	}
	auto& buffer = m_slot_buffers[*owner];
	if (buffer.is_deleted || !buffer.IsInBounds(vaddr, size)) {
		return {nullptr, 0};
	}
	return {&buffer, buffer.Offset(vaddr)};
}

bool BufferCache::DebugCpuDirty(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

bool BufferCache::GpuFillSource(uint64_t vaddr, vk::Buffer* handle, uint64_t* offset) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	constexpr uint64_t size = sizeof(uint32_t);
	if ((vaddr & (size - 1u)) != 0 || !GuestRange {vaddr, size}.Valid() ||
	    !m_gpu_modified_ranges.Contains(vaddr, size)) {
		return false;
	}
	if ((!m_early_released.empty() && OverlapsUnpublished(vaddr, size)) ||
	    HasPendingBackingPublication(vaddr, size)) {
		return false;
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner == nullptr || !*owner || IsBufferInvalid(*owner)) {
		return false;
	}
	auto& buffer = m_slot_buffers[*owner];
	if (buffer.is_deleted || !buffer.IsInBounds(vaddr, size) ||
	    m_texture_cache.IsRegionGpuModified(vaddr, size)) {
		return false;
	}
	if (handle != nullptr) {
		*handle = buffer.Handle();
	}
	if (offset != nullptr) {
		*offset = buffer.Offset(vaddr);
	}
	return true;
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
	// Unbounded GPU writes may change any watched metadata byte.
	NoteMetadataWrite(0, UINT64_MAX);
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
	RewritePublicationSnapshot();
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
	RewritePublicationSnapshot();
	// Publish completion only after the callback has written every registered backing range.
	m_backing_publication_count.store(m_backing_publications.size(), std::memory_order_release);
}

// KYTY_PUBLICATION_SNAPSHOT (live, default 1): HasPendingBackingPublication reads the seqlock copy
// of the pending ranges instead of scanning the list under m_backing_publication_mutex. Every
// recorded read of the draw-prep workers asks it (the read gate's pending-publication hint), and so
// do the command processor's clean predicates; with publications almost always pending, 6 workers
// and the command processor serialized on the mutex (Sky Garden sampling: ~30% of the workers'
// preparation time in the lock). The answer is the locked scan's at one instant inside the call,
// as before. 0 off (the locked scan); verify: both, counts answers that differ twice in a row (a
// registration or completion between two scans can make one pair differ legitimately); exit:
// stops at the first such difference.
static Live::Switch g_publication_snapshot("KYTY_PUBLICATION_SNAPSHOT", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 1;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	if (std::strcmp(value, "exit") == 0) {
		return 3;
	}
	return std::strcmp(value, "0") == 0 ? 0 : 1;
});
static std::atomic<uint64_t> g_publication_snapshot_checks {0};
static std::atomic<uint64_t> g_publication_snapshot_mismatches {0};
static std::atomic<uint64_t> g_publication_snapshot_locked {0}; // verify mode: scans that took the mutex

void BufferCache::RewritePublicationSnapshot() {
	uint64_t count = 0;
	for (const auto& entry: m_backing_publications) {
		count += entry.ranges.size();
	}
	// Seqlock writer (serialized by the mutex): odd while the copy changes.
	const auto seq = m_publication_seq.load(std::memory_order_relaxed);
	m_publication_seq.store(seq + 1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_release);
	if (count > PublicationSnapshotRanges) {
		m_publication_snapshot_count.store(PublicationSnapshotOverflow, std::memory_order_relaxed);
	} else {
		uint32_t index = 0;
		for (const auto& entry: m_backing_publications) {
			for (const auto& range: entry.ranges) {
				m_publication_begin[index].store(range.address, std::memory_order_relaxed);
				m_publication_end[index].store(range.End(), std::memory_order_relaxed);
				index++;
			}
		}
		m_publication_snapshot_count.store(index, std::memory_order_relaxed);
	}
	m_publication_seq.store(seq + 2, std::memory_order_release);
}

// The seqlock read: false when the copy overflowed or kept changing (the caller scans locked).
bool BufferCache::SnapshotPendingBackingPublication(uint64_t vaddr, uint64_t size,
                                                    bool& answer) const {
	const auto end = vaddr + size;
	for (uint32_t attempt = 0; attempt < 64; attempt++) {
		const auto seq = m_publication_seq.load(std::memory_order_acquire);
		if ((seq & 1u) != 0) {
			continue;
		}
		const auto count = m_publication_snapshot_count.load(std::memory_order_relaxed);
		if (count == PublicationSnapshotOverflow) {
			return false;
		}
		bool found = false;
		for (uint32_t i = 0; i < count && !found; i++) {
			found = m_publication_begin[i].load(std::memory_order_relaxed) < end &&
			        vaddr < m_publication_end[i].load(std::memory_order_relaxed);
		}
		std::atomic_thread_fence(std::memory_order_acquire);
		if (m_publication_seq.load(std::memory_order_relaxed) == seq) {
			answer = found;
			return true;
		}
	}
	return false;
}

bool BufferCache::HasPendingBackingPublication(uint64_t vaddr, uint64_t size) const {
	const auto mode = g_publication_snapshot.Get();
	if (mode == 0) {
		return PendingBackingPublicationTick(vaddr, size).has_value();
	}
	const GuestRange query {vaddr, size};
	EXIT_IF(!query.Valid());
	if (m_backing_publication_count.load(std::memory_order_acquire) == 0) {
		return false;
	}
	const auto locked = [&] { return PendingBackingPublicationTick(vaddr, size).has_value(); };
	bool answer = false;
	if (!SnapshotPendingBackingPublication(vaddr, size, answer)) {
		answer = locked();
		if (mode >= 2) {
			g_publication_snapshot_locked.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (mode >= 2) [[unlikely]] {
		const auto checks = g_publication_snapshot_checks.fetch_add(1, std::memory_order_relaxed) + 1;
		if (answer != locked()) {
			bool again = false;
			if (!SnapshotPendingBackingPublication(vaddr, size, again)) {
				again = locked();
			}
			if (again != locked()) {
				const auto count =
				    g_publication_snapshot_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
				if (count <= 16) {
					std::printf("PublicationSnapshot: snapshot and locked scan differ for [0x%" PRIx64
					            ", +0x%" PRIx64 ")\n",
					            vaddr, size);
				}
				if (mode == 3) {
					EXIT("PublicationSnapshot: snapshot and locked scan differ\n");
				}
			}
		}
		if ((checks & 0xffffffu) == 0) {
			std::printf("PublicationSnapshot verify: %" PRIu64 " checks, %" PRIu64
			            " mismatches, %" PRIu64 " locked scans\n",
			            checks, g_publication_snapshot_mismatches.load(std::memory_order_relaxed),
			            g_publication_snapshot_locked.load(std::memory_order_relaxed));
		}
	}
	return answer;
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

// KYTY_DIRECT_READ_WAIT (adapted from KytyPS5-Frank 1c9ad7586): a direct read of guest memory
// that bypasses page protection (TryReadBufferBacking: the SRT walker's buffer reads) found bytes a
// queued download still has to publish: their older value is in the backing until the
// publication's tick completes and its priority operation has run. Waits for that publication
// (a few rounds, as a later download can queue another one), else drains everything recorded.
void BufferCache::WaitBackingPublications(uint64_t vaddr, uint64_t size) {
	EXIT_IF(!GuestGpu::IsGpuThread());
	for (int round = 0; round < 8; round++) {
		const auto tick = PendingBackingPublicationTick(vaddr, size);
		if (!tick) {
			return;
		}
		if (!m_scheduler.IsFree(*tick)) {
			m_scheduler.Wait(*tick);
		}
		m_scheduler.WaitPriorityOperations(*tick);
	}
	if (HasPendingBackingPublication(vaddr, size)) {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

namespace {
// Buffers that KYTY_HOST_BUFFER_MB placed in system memory cost no video memory, and the one such
// a title has is its whole heap (2.7 GB): collecting it downloads it when the GPU wrote to it (a
// ~100 ms drain) and the next draw uploads all of it again. They stay unless
// KYTY_GC_HOST_BUFFERS=1 (adapted from Senaxx/wolverine 75dea37fd).
bool KeptOutOfCollection(uint64_t size) {
	static const bool gc_host = std::getenv("KYTY_GC_HOST_BUFFERS") != nullptr;
	const auto        bytes   = HostBufferBytes();
	return !gc_host && bytes != 0 && size >= bytes;
}
} // namespace

void BufferCache::RetireUnusedBuffers(uint64_t frame, uint64_t min_age) {
	if (frame <= min_age) {
		return;
	}
	// The GC tick the newest recorded frame up to frame - min_age began at: an LRU item below it
	// was last used before that frame (LRU ticks are GC ticks).
	const auto limit_frame = frame - min_age;
	const auto newer       = std::upper_bound(m_frame_ticks.begin(), m_frame_ticks.end(), limit_frame,
	                                          [](uint64_t value, const auto& entry) { return value < entry.first; });
	if (newer == m_frame_ticks.begin()) {
		return;
	}
	const auto            limit_tick = std::prev(newer)->second;
	std::vector<BufferId> candidates;
	size_t                scanned = 0;
	m_lru_cache.ForEachItemBelow(limit_tick, [&](BufferId id) {
		const auto& buffer = m_slot_buffers[id];
		if (!buffer.is_deleted && !KeptOutOfCollection(buffer.Size()) &&
		    !m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			candidates.push_back(id);
		}
		return ++scanned >= 4096 || candidates.size() >= 64;
	});
	if (candidates.empty()) {
		return;
	}
	// As the collector below: settle pending side readbacks before the ownership checks.
	CompleteAllSideReadbacks();
	for (const auto id: candidates) {
		if (IsBufferInvalid(id)) {
			continue;
		}
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "idle retirement");
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			continue;
		}
		m_idle_freed_bytes += buffer.Size();
		m_idle_freed++;
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		DeleteBuffer(id);
	}
}

void BufferCache::RunGarbageCollector() {
	MaintainHotPages();
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	const uint64_t frame = m_memory_tracker.Frame();
	if (m_idle_frames != 0 || m_pressure_frames != 0) {
		if (m_frame_ticks.empty() || m_frame_ticks.back().first != frame) {
			m_frame_ticks.emplace_back(frame, tick);
		}
		const auto keep = std::max(m_idle_frames, m_pressure_frames) + 64;
		while (m_frame_ticks.size() > keep) {
			m_frame_ticks.pop_front();
		}
	}
	if (m_idle_frames != 0 && frame >= m_idle_next_frame) {
		m_idle_next_frame = frame + 32;
		RetireUnusedBuffers(frame, m_idle_frames);
	}
	// KYTY_VRAM_GC_BUDGET (vramBudget.h): the marks follow the budget as it is now; the collector
	// below (blind to BDA reads) only runs above the planning budget itself.
	auto trigger  = m_trigger_gc_memory;
	auto critical = m_critical_gc_memory;
	if (VramBudget::GcEnabled() && m_graphics.CanReportMemoryUsage()) {
		// The budget once per frame (this runs on the command-processor thread after every
		// completed submission).
		if (frame != m_budget_frame) {
			m_budget_frame    = frame;
			const auto budget = m_graphics.GetTotalMemoryBudget();
			m_budget_trigger  = VramBudget::BufferTrigger(budget);
			m_budget_critical = VramBudget::BufferCritical(budget);
		}
		trigger  = m_budget_trigger;
		critical = m_budget_critical;
	}
	if (m_total_used_memory < trigger) {
		return;
	}
	if (m_pressure_frames != 0) {
		// KYTY_VRAM_PRESSURE_BUFFER_FRAMES: once per frame, buffers unused for that many
		// frames (a quarter, at least two, above the critical mark) without GPU-dirty bytes, instead
		// of the submission-age collection below (no downloads).
		if (frame != m_pressure_frame) {
			m_pressure_frame = frame;
			RetireUnusedBuffers(frame, m_total_used_memory >= critical
			                               ? std::max<uint64_t>(m_pressure_frames / 4, 2)
			                               : m_pressure_frames);
		}
		return;
	}
	// Pending side readbacks keep tracker pages GPU-owned without exact dirty bytes; settle
	// them so the ownership checks and downloads below see a consistent state.
	CompleteAllSideReadbacks();

	const bool     aggressive = m_total_used_memory >= critical;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (buffer.gc_pending) {
			return false; // KYTY_GC_ASYNC_DOWNLOAD: its download is in flight, dropped below
		}
		if (KeptOutOfCollection(buffer.Size())) {
			return false;
		}
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
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	// KYTY_GC_ASYNC_DOWNLOAD=1 (default off; Sakura 45bc441ec + 3aefac099 via usa-ratt/
	// KytyPS5-GT7): the command processor does not wait here. A deferred operation (it runs once
	// the GPU has finished this tick and the tick's priority operations, which publish the
	// downloads, are done) drops the buffers through DeleteBuffer (address table cleared,
	// destroyed after the then-current tick: later work may still reach it by device address).
	// A buffer a draw or dispatch bound again meanwhile, or one the GPU wrote again (its range in
	// m_gpu_modified_ranges, also by device address), stays. The wait took 37 ms of a 290 ms race
	// frame on a 6 GB card in the fork.
	static const bool async_download = [] {
		const char* value = std::getenv("KYTY_GC_ASYNC_DOWNLOAD");
		return value != nullptr && value[0] == '1';
	}();
	if (async_download && m_scheduler.Active()) {
		struct Pending {
			BufferId id;
			uint64_t address;
		};
		std::vector<Pending> pending;
		pending.reserve(dirty_buffers.size());
		for (const auto id: dirty_buffers) {
			auto& buffer      = m_slot_buffers[id];
			buffer.gc_pending = true;
			buffer.gc_reused  = false;
			pending.push_back({id, buffer.CpuAddress()});
		}
		m_scheduler.DeferOperation([this, pending = std::move(pending)] {
			static uint64_t dropped = 0;
			static uint64_t kept    = 0;
			for (const auto& [id, address]: pending) {
				auto* buffer = m_slot_buffers.try_get(id);
				if (buffer == nullptr || buffer->is_deleted || !buffer->gc_pending ||
				    buffer->CpuAddress() != address) {
					continue;
				}
				buffer->gc_pending = false;
				if (buffer->gc_reused ||
				    m_gpu_modified_ranges.Intersects(buffer->CpuAddress(), buffer->Size())) {
					buffer->gc_reused = false;
					kept++;
					continue;
				}
				m_memory_tracker.UnmarkRegionAsGpuModified(buffer->CpuAddress(), buffer->Size());
				if (m_memory_tracker.IsRegionGpuModified(buffer->CpuAddress(), buffer->Size())) {
					kept++;
					continue;
				}
				m_memory_tracker.UntrackMemory(buffer->CpuAddress(), buffer->Size());
				DeleteBuffer(id);
				Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
				if (++dropped % 1024 == 1) {
					LOGF("BufferCache: %" PRIu64 " GPU-written buffers dropped after their "
					     "downloads landed, %" PRIu64 " kept (used again meanwhile; "
					     "KYTY_GC_ASYNC_DOWNLOAD)\n",
					     dropped, kept);
				}
			}
		});
		return;
	}
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

void BufferCache::ReportVram() {
	using VramStats::ToMiB;
	struct Bucket {
		uint64_t bytes = 0;
		uint64_t count = 0;
		void     Add(uint64_t value) {
            bytes += value;
            ++count;
		}
	};
	struct Entry {
		uint64_t bytes   = 0;
		uint64_t address = 0;
		uint64_t age     = 0;
		bool     dirty   = false;
	};
	// GC ticks (completed submissions) per presented frame since the last report, to express the
	// LRU ages in frames.
	static uint64_t last_tick  = 0;
	static uint64_t last_frame = 0;
	const auto      frame      = static_cast<uint64_t>(m_memory_tracker.Frame());
	const auto      tick       = m_gc_tick;
	const double    per_frame  = frame > last_frame ? static_cast<double>(tick - last_tick) /
	                                                   static_cast<double>(frame - last_frame)
	                                                : 0.0;
	last_tick  = tick;
	last_frame = frame;
	static constexpr std::array<uint64_t, 4>    age_limits {1, 10, 60, 600};
	static constexpr std::array<const char*, 5> age_names {"<=1", "2-10", "11-60", "61-600", ">600"};
	std::array<Bucket, age_names.size()> by_age {};
	Bucket             all;
	Bucket             dirty;
	uint64_t           lowest  = UINT64_MAX;
	uint64_t           highest = 0;
	std::vector<Entry> entries;
	entries.reserve(m_buffers.size());
	for (const auto& [address, id]: m_buffers) {
		const auto& buffer    = m_slot_buffers[id];
		const auto  age_ticks = tick - std::min<uint64_t>(tick, m_lru_cache.TickOf(buffer.lru_id));
		const auto  age       = per_frame > 0.0
		                            ? static_cast<uint64_t>(static_cast<double>(age_ticks) / per_frame)
		                            : age_ticks;
		const bool  gpu_dirty = m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size());
		size_t      bucket    = 0;
		while (bucket < age_limits.size() && age > age_limits[bucket]) {
			++bucket;
		}
		by_age[bucket].Add(buffer.Size());
		all.Add(buffer.Size());
		if (gpu_dirty) {
			dirty.Add(buffer.Size());
		}
		lowest  = std::min(lowest, address);
		highest = std::max(highest, address + buffer.Size());
		entries.push_back({buffer.Size(), address, age, gpu_dirty});
	}
	std::string ages;
	for (size_t index = 0; index < age_names.size(); ++index) {
		char part[64];
		std::snprintf(part, sizeof(part), " %s:%.1f(%llu)", age_names[index], ToMiB(by_age[index].bytes),
		              static_cast<unsigned long long>(by_age[index].count));
		ages += part;
	}
	VramStats::Line("guest buffers: %.1f MiB (%llu), with GPU-dirty bytes %.1f MiB (%llu), guest span "
	                "0x%" PRIx64 "..0x%" PRIx64 " | frames since use (%.1f GC ticks/frame):%s",
	                ToMiB(all.bytes), static_cast<unsigned long long>(all.count), ToMiB(dirty.bytes),
	                static_cast<unsigned long long>(dirty.count), lowest == UINT64_MAX ? 0 : lowest,
	                highest, per_frame, ages.c_str());
	const size_t top = std::min<size_t>(entries.size(), 10);
	std::partial_sort(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(top),
	                  entries.end(), [](const Entry& a, const Entry& b) { return a.bytes > b.bytes; });
	for (size_t index = 0; index < top; ++index) {
		VramStats::Line("  guest top %.1f MiB guest=0x%" PRIx64 " unused=%llu frames%s",
		                ToMiB(entries[index].bytes), entries[index].address,
		                static_cast<unsigned long long>(entries[index].age),
		                entries[index].dirty ? " gpu-dirty" : "");
	}
	auto buffers = VramStats::BufferSnapshot();
	std::erase_if(buffers, [](const VramStats::BufferEntry& entry) { return entry.cpu_address != 0; });
	std::sort(buffers.begin(), buffers.end(),
	          [](const auto& a, const auto& b) { return a.bytes > b.bytes; });
	static constexpr std::array<const char*, 4> usage_names {"device-local", "upload", "download",
	                                                         "stream"};
	std::string line;
	for (size_t index = 0; index < std::min<size_t>(buffers.size(), 14); ++index) {
		char part[96];
		std::snprintf(part, sizeof(part), " %.1f:%s/%s%s", ToMiB(buffers[index].bytes),
		              buffers[index].usage < usage_names.size() ? usage_names[buffers[index].usage] : "?",
		              buffers[index].device_local ? "vram" : "sysmem",
		              buffers[index].host_cached ? "/cached" : "");
		line += part;
	}
	VramStats::Line("non-guest buffers (%zu), largest MiB:%s", buffers.size(), line.c_str());
	if (m_bda_sparse != nullptr) {
		VramStats::Line("BDA page table: sparse, %llu of %zu blocks bound (%.1f of %.1f MiB) in %zu chunks",
		                static_cast<unsigned long long>(m_bda_sparse->bound_blocks),
		                m_bda_sparse->bound.size(),
		                ToMiB(m_bda_sparse->bound_blocks * m_bda_sparse->block_size),
		                ToMiB(BDA_PAGETABLE_SIZE), m_bda_sparse->chunks.size());
	} else {
		VramStats::Line("BDA page table: dense, %.1f MiB", ToMiB(BDA_PAGETABLE_SIZE));
	}
	VramStats::Line("buffer GC: usage %.1f MiB, trigger %.1f critical %.1f MiB, GC tick %llu | idle "
	                "retirement (KYTY_VRAM_IDLE_BUFFER_FRAMES=%llu) deleted %.1f MiB (%llu) in total",
	                ToMiB(m_total_used_memory), ToMiB(m_trigger_gc_memory), ToMiB(m_critical_gc_memory),
	                static_cast<unsigned long long>(m_gc_tick),
	                static_cast<unsigned long long>(m_idle_frames), ToMiB(m_idle_freed_bytes),
	                static_cast<unsigned long long>(m_idle_freed));
}

void BufferCache::ProcessFaultBuffer(const std::vector<std::pair<uint64_t, uint64_t>>& mapped) {
	m_fault_manager.ProcessFaultBuffer(mapped);
}

void BufferCache::NoteBdaRegistered(uint64_t vaddr, uint64_t size) {
	// The range and the structure epoch move together (SynchronizeBdaBuffersNow takes both).
	constexpr size_t MaxRegisteredRanges = 4096;
	std::lock_guard lock(m_bda_registered_mutex);
	if (!m_bda_registered_overflow) {
		m_bda_registered.Add(vaddr, size);
		m_bda_registered_overflow = m_bda_registered.Size() > MaxRegisteredRanges;
	}
	auto epoch = m_bda_structure_epoch.load(std::memory_order_relaxed);
	while (epoch != UINT64_MAX &&
	       !m_bda_structure_epoch.compare_exchange_weak(epoch, epoch + 1,
	                                                     std::memory_order_release,
	                                                     std::memory_order_relaxed)) {}
}

void BufferCache::InvalidateBdaSynchronization() noexcept {
	// KYTY_BDA_REGISTER_LOG: a change other than a new buffer; the next pass scans everything.
	std::unique_lock<std::mutex> lock;
	if (m_bda_register_log) {
		lock = std::unique_lock(m_bda_registered_mutex);
		m_bda_layout_epoch.fetch_add(1, std::memory_order_release);
	}
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
	const auto submission = SyncEpoch::CurrentSubmission();
	const auto structure  = m_bda_structure_epoch.load(std::memory_order_acquire);
	if (m_bda_epoch_skip && sync_epoch == m_bda_synced_epoch &&
	    structure == m_bda_synced_structure) {
		BufferLookupStats::AddPart(BufferLookupStats::Part::BdaSkip, 0);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncEpochSkips);
		m_bda_epoch_totals.skips++;
		if (m_bda_epoch_verify != 0) {
			VerifyBdaEpochSkip(mapped_ranges);
		}
		return;
	}
	// KYTY_BDA_SYNC_PER_SUBMISSION=1 (default off, live; upstream KytyPS5 309ba4f5, Senaxx
	// a329b69a8): at most one pass per guest submission, instead of one per epoch. What the game
	// wrote before submitting reaches every BDA read of the submission, as on the console. A CPU
	// write made while the submission runs reaches BDA reads only in the next submission, even
	// behind a fence that orders it (a WAIT_REG_MEM on a CPU-written label): the epoch pass would
	// have uploaded it. New buffers and GPU mapping changes still run the pass (structure epoch): a
	// new buffer's pages start CPU-dirty and have never been uploaded. KYTY_BDA_SYNC_PER_SUBMISSION=
	// verify runs those passes and counts what they upload (the bytes 1 would defer).
	const auto per_submission  = g_bda_sync_per_submission.Get();
	const bool same_submission = per_submission != 0 && submission == m_bda_synced_submission &&
	                             structure == m_bda_synced_structure;
	if (same_submission && per_submission == 1) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncSubmissionSkips);
		m_bda_epoch_totals.submission_skips++;
		ReportBdaSubmissionGate();
		return;
	}
	const auto fault_epoch     = m_memory_tracker.FaultMutationEpoch();
	const auto uploaded_before = m_bda_pass_upload_bytes;
	m_bda_count_uploads        = same_submission;
	const auto pass_start      = BufferLookupStats::Now();
	// KYTY_BDA_LIVE_PAGES: no new redirect while BDA-write settles are pending (I-W2).
	const auto live_mode = BdaLivePagesMode(); // 1 / 2 (verify) redirect; 3 (check) only verifies
	m_bda_live_pass      = (live_mode == 1 || live_mode == 2) && m_bda_pending.empty();
	SynchronizeBdaBuffersNow(mapped_ranges);
	m_bda_live_pass = false;
	if (m_bda_live != nullptr) {
		LivePagesAfterPass(mapped_ranges, fault_epoch);
	}
	if (pass_start != 0) {
		BufferLookupStats::AddPart(static_cast<BufferLookupStats::Part>(m_bda_pass_kind),
		                           BufferLookupStats::Now() - pass_start);
	}
	m_bda_count_uploads = false;
	m_bda_epoch_totals.passes++;
	if (same_submission) {
		m_bda_epoch_totals.submission_verify_passes++;
		m_bda_epoch_totals.submission_deferred_bytes += m_bda_pass_upload_bytes - uploaded_before;
	}
	if (per_submission != 0) {
		ReportBdaSubmissionGate();
	}
	m_bda_synced_epoch      = sync_epoch;
	m_bda_synced_submission = submission;
	m_bda_synced_structure  = structure;
	m_bda_synced_fault      = fault_epoch;
}

void BufferCache::ReportBdaSubmissionGate() {
	// Every 10 s while the gate is on: passes, skips and (verify) the bytes a skip would defer.
	if ((++m_bda_gate_report_calls & 1023u) != 0) {
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_bda_gate_report_time == std::chrono::steady_clock::time_point {}) {
		m_bda_gate_report_time = now;
		m_bda_gate_reported    = m_bda_epoch_totals;
		return;
	}
	const auto elapsed = std::chrono::duration<double>(now - m_bda_gate_report_time).count();
	if (elapsed < 10.0) {
		return;
	}
	const auto& totals = m_bda_epoch_totals;
	const auto& last   = m_bda_gate_reported;
	std::printf("BdaSyncPerSubmission %.0fs (%s): %" PRIu64 " passes, %" PRIu64
	            " skipped by the submission gate, %" PRIu64 " epoch skips; verify %" PRIu64
	            " gated passes run, %" PRIu64 " bytes they uploaded\n",
	            elapsed, g_bda_sync_per_submission.Get() == 2 ? "verify" : "on",
	            totals.passes - last.passes, totals.submission_skips - last.submission_skips,
	            totals.skips - last.skips,
	            totals.submission_verify_passes - last.submission_verify_passes,
	            totals.submission_deferred_bytes - last.submission_deferred_bytes);
	std::fflush(stdout);
	m_bda_gate_report_time = now;
	m_bda_gate_reported    = totals;
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
	// KYTY_BDA_REGISTER_LOG: the structure epoch, the layout epoch and the buffers registered up
	// to that structure epoch, taken together.
	RangeSet registered;
	bool     registered_complete = false;
	uint64_t layout_epoch        = 0;
	uint64_t structure_epoch     = 0;
	if (m_bda_register_log) {
		std::lock_guard lock(m_bda_registered_mutex);
		structure_epoch     = m_bda_structure_epoch.load(std::memory_order_acquire);
		layout_epoch        = m_bda_layout_epoch.load(std::memory_order_acquire);
		registered_complete = !m_bda_registered_overflow;
		std::swap(registered, m_bda_registered);
		m_bda_registered_overflow = false;
	} else if (m_bda_incremental_sync) {
		structure_epoch = m_bda_structure_epoch.load(std::memory_order_acquire);
	}
	const bool structure_holds = m_bda_incremental_sync && cpu_epoch != UINT64_MAX &&
	                             structure_epoch != UINT64_MAX &&
	                             structure_epoch == m_bda_scanned_structure_epoch;
	// Only new buffers since the last pass (their ranges are in `registered`).
	const bool structure_grew = m_bda_register_log && !structure_holds && registered_complete &&
	                            cpu_epoch != UINT64_MAX && structure_epoch != UINT64_MAX &&
	                            layout_epoch == m_bda_scanned_layout_epoch;
	using LookupPart = BufferLookupStats::Part;
	m_bda_pass_kind  = static_cast<uint8_t>(LookupPart::BdaFull);
	if (structure_holds && cpu_epoch == m_bda_scanned_cpu_epoch) {
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaNone);
		if (m_bda_hot_ranges.empty()) {
			if (collect) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncSkips);
			}
			FaultCost::NoteBdaPass(0, 0);
			return;
		}
		BdaSyncStats stats;
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaHot);
		if (SynchronizeBdaHotRanges(stats)) {
			FaultCost::NoteBdaPass(1, stats.upload_bytes);
			m_bda_pass_upload_bytes += stats.upload_bytes;
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
		m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaFull);
	} else if ((structure_holds || structure_grew) && m_bda_dirty_log && log_complete &&
	           m_bda_log_baseline &&
	           (m_bda_pass_kind = static_cast<uint8_t>(LookupPart::BdaLog),
	            SynchronizeBdaDirtied(mapped_ranges, structure_grew ? &registered : nullptr))) {
		FaultCost::NoteBdaPass(2, m_bda_last_pass_bytes);
		m_bda_scanned_cpu_epoch = cpu_epoch;
		if (structure_grew) {
			m_bda_scanned_structure_epoch = structure_epoch;
			m_bda_register_passes++;
		}
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
	m_bda_pass_kind = static_cast<uint8_t>(
	    m_bda_incremental_sync && structure_epoch != m_bda_scanned_structure_epoch
	        ? LookupPart::BdaFullNew
	        : LookupPart::BdaFull);
	if (m_bda_hot_sync) {
		m_bda_hot_ranges.clear();
		stats.hot_ranges = &m_bda_hot_ranges;
	}
	const bool keep_stats = collect || m_bda_hot_sync || m_bda_count_uploads;
	{
		// Only uploads are recorded while scanning: all of them share one barrier pair.
		const UploadBatch upload_batch(*this);
		mapped_ranges.ForEach([this, keep_stats, &stats](uint64_t start, uint64_t end) {
			SynchronizeBuffersInRange(start, end - start, keep_stats ? &stats : nullptr);
		});
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
	if (m_bda_incremental_sync) {
		m_bda_scanned_cpu_epoch       = cpu_epoch;
		m_bda_scanned_structure_epoch = structure_epoch;
		m_bda_scanned_layout_epoch    = layout_epoch;
	}
	// The log taken above holds nothing this scan did not cover; later transitions log again.
	m_bda_log_baseline = m_bda_dirty_log;
	FaultCost::NoteBdaPass(3, stats.upload_bytes);
	if (collect) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncPasses);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncScannedBuffers, stats.scanned_buffers);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadBytes, stats.upload_bytes);
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSyncUploadCopies, stats.upload_copies);
	}
}

bool BufferCache::SynchronizeBdaDirtied(const RangeSet& mapped_ranges, const RangeSet* registered) {
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
		if (registered != nullptr) {
			// KYTY_BDA_REGISTER_LOG: new buffers, whose CPU-dirty pages no fault logged.
			registered->ForEach([&](uint64_t begin, uint64_t end) {
				logged++;
				mapped_ranges.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
					SynchronizeBuffersInRange(start, finish - start, &stats);
				});
			});
		}
		BdaSyncStats hot_stats;
		for (const auto& range: m_bda_hot_ranges) {
			(void)SynchronizeBuffer(m_slot_buffers[range.id], range.address, range.size, false,
			                        false, &hot_stats);
		}
		stats.upload_bytes += hot_stats.upload_bytes;
		stats.upload_copies += hot_stats.upload_copies;
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
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
	m_bda_last_pass_bytes = stats.upload_bytes;
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
			if (m_bda_live_pass) {
				// KYTY_BDA_LIVE_PAGES (bufferCacheLive.cpp): the full pass redirects qualifying pages.
				LiveSyncBufferRange(it->second, start, finish - start, stats);
				continue;
			}
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false, stats);
		}
	}
}

namespace {

enum class BdaWritesMode { Off, On, Verify };

BdaWritesMode GetBdaWritesMode() {
	static const BdaWritesMode mode = [] {
		const auto* value = std::getenv("KYTY_BDA_WRITES");
		// On by default, like CodegenOptions::bda_writes (KYTY_BDA_WRITES=0 turns both off).
		if (value == nullptr || value[0] == '\0') {
			return BdaWritesMode::On;
		}
		if (std::strcmp(value, "0") == 0) {
			return BdaWritesMode::Off;
		}
		return std::strcmp(value, "verify") == 0 ? BdaWritesMode::Verify : BdaWritesMode::On;
	}();
	return mode;
}

// One console line per shader for each kind of BDA-write anomaly.
bool FirstBdaWriteNote(uint64_t shader_hash, uint32_t kind) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	std::scoped_lock                    lock(mutex);
	return noted.insert(shader_hash ^ (static_cast<uint64_t>(kind) << 62u)).second;
}

} // namespace

bool BdaWritesEnabled() {
	return GetBdaWritesMode() != BdaWritesMode::Off;
}

bool BdaWritesVerify() {
	return GetBdaWritesMode() == BdaWritesMode::Verify;
}

namespace {

// KYTY_BDA_HEAPS (default off): see BufferCache::NoteBdaFault. KYTY_BDA_HEAP_MB (default 64) is the
// window around a fault (inside its allocation) that it makes resident.
bool BdaHeapsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BDA_HEAPS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// The windows learned so far persist in _PipelineCache/<title>.bdaheaps ("start size" in hex per line):
// the first dispatch that stores to a cold page drops those stores, so the next start makes them resident
// before it.
std::filesystem::path BdaHeapsPath() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".bdaheaps");
}

uint64_t BdaHeapLimit() {
	static const uint64_t limit = [] {
		const auto* value = std::getenv("KYTY_BDA_HEAP_MB");
		const auto  mb    = value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : 64u;
		return mb << 20u;
	}();
	return limit;
}

} // namespace

void BufferCache::NoteBdaFault(uint64_t vaddr, uint64_t size) {
	if (!BdaHeapsEnabled() || size == 0) {
		return;
	}
	uint64_t alloc_start = 0;
	uint64_t alloc_bytes = 0;
	if (!Libs::LibKernel::Memory::GetGpuBufferMapping(vaddr, &alloc_start, &alloc_bytes)) {
		return;
	}
	// The window is a multiple of its size, clipped to the allocation: neighbouring faults share it.
	const auto window = std::max<uint64_t>(BdaHeapLimit(), CACHING_PAGESIZE);
	const auto start  = std::max(alloc_start, vaddr & ~(window - 1u));
	const auto end    = std::min(alloc_start + alloc_bytes, (vaddr & ~(window - 1u)) + window);
	if (end <= start || end - start <= size) {
		return;
	}
	const auto bytes = end - start;
	{
		std::scoped_lock lock {m_bda_heaps_mutex};
		for (const auto& [heap_start, heap_size]: m_bda_heaps) {
			if (heap_start == start && heap_size == bytes) {
				return;
			}
		}
		m_bda_heaps.emplace_back(start, bytes);
		if (const auto path = BdaHeapsPath(); !path.empty()) {
			std::error_code error;
			std::filesystem::create_directories(path.parent_path(), error);
			if (auto* file = std::fopen(path.string().c_str(), "w"); file != nullptr) {
				for (const auto& [heap_start, heap_size]: m_bda_heaps) {
					std::fprintf(file, "%" PRIx64 " %" PRIx64 "\n", heap_start, heap_size);
				}
				std::fclose(file);
			}
		}
	}
	(void)FindBuffer(start, bytes);
	LOGF("KYTY_BDA_HEAPS: BDA fault at 0x%016" PRIx64 ": 0x%016" PRIx64 "+0x%" PRIx64
	            " is resident now\n",
	            vaddr, start, bytes);
	
}

void BufferCache::PrepareBdaWrites() {
	EXIT_IF(!BdaWritesEnabled());
	LivePagesBeforeBdaWrites(); // KYTY_BDA_LIVE_PAGES (I-W1)
	if (BdaHeapsEnabled()) {
		std::vector<std::pair<uint64_t, uint64_t>> heaps;
		{
			std::scoped_lock lock {m_bda_heaps_mutex};
			if (!m_bda_heaps_loaded) {
				m_bda_heaps_loaded = true;
				if (const auto path = BdaHeapsPath(); !path.empty()) {
					if (auto* file = std::fopen(path.string().c_str(), "r"); file != nullptr) {
						unsigned long long heap_start = 0;
						unsigned long long heap_size  = 0;
						while (std::fscanf(file, "%llx %llx", &heap_start, &heap_size) == 2) {
							m_bda_heaps.emplace_back(heap_start, heap_size);
						}
						std::fclose(file);
						LOGF("KYTY_BDA_HEAPS: %zu window(s) loaded from %s\n", m_bda_heaps.size(),
						     path.string().c_str());
					}
				}
			}
			heaps = m_bda_heaps;
		}
		for (const auto& [start, bytes]: heaps) {
			uint64_t mapped_start = 0;
			uint64_t mapped_size  = 0;
			if (Libs::LibKernel::Memory::GetGpuBufferMapping(start, &mapped_start, &mapped_size) &&
			    mapped_start <= start && start + bytes <= mapped_start + mapped_size) {
				(void)FindBuffer(start, bytes);
			}
		}
	}
	m_fault_manager.PrepareBdaWrites();
	// The dispatch may overwrite any known fill.
	ForgetKnownFills(0, uint64_t {1} << 40u);
	// A GPU-modified image over a cache buffer holds bytes its buffer lacks. A BDA write into that
	// buffer and the settle's image invalidation would drop them: move them into the buffer first
	// when image writebacks are on, exactly as a writable binding over the range does (only pages
	// with a cache buffer can receive a BDA write).
	std::vector<std::pair<BufferId, GuestRange>> overlaps;
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		m_texture_cache.m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (!image.IsGpuModified()) {
				return;
			}
			const auto begin = image.info.data.address;
			const auto end   = image.info.data.End();
			auto       it    = m_buffers.upper_bound(begin);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < end; ++it) {
				const auto& buffer = m_slot_buffers[it->second];
				const auto  start  = std::max(buffer.CpuAddress(), begin);
				const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
				if (start < finish) {
					overlaps.push_back({it->second, GuestRange {start, finish - start}});
				}
			}
		});
	}
	if (overlaps.empty()) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaAliasedImages, overlaps.size());
	if (!ImageWritebackOnGpuWriteEnabled()) {
		// As for a writable binding with writebacks off: the image contents are not moved.
		return;
	}
	for (const auto& [id, range]: overlaps) {
		if (!IsBufferInvalid(id) && m_slot_buffers[id].IsInBounds(range.address, range.size)) {
			PreserveImagesForGpuWrite(id, range.address, range.size);
		}
	}
}

// KYTY_BDA_STORE_LOG_STATS=1: "BdaSettle 10s: ..." console lines while BDA-writing dispatches
// settle (none printed: none settled). GPU thread.
static void NoteBdaSettleStats(size_t logged, uint32_t dropped) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BDA_STORE_LOG_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	if (!enabled) {
		return;
	}
	static uint64_t settles = 0;
	static uint64_t stores  = 0;
	static uint64_t drops   = 0;
	static uint64_t total   = 0;
	static auto     start   = std::chrono::steady_clock::now();
	settles++;
	total++;
	stores += logged;
	drops += dropped;
	const auto now = std::chrono::steady_clock::now();
	const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
	if (ms < 10000) {
		return;
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "BdaSettle 10s: {} settles, {} logged stores, {} dropped writes in {:.1f} s; total {} settles\n",
	    settles, stores, drops, static_cast<double>(ms) / 1000.0, total));
	settles = stores = drops = 0;
	start = now;
}

// KYTY_BDA_ASYNC_SETTLE (live, default 0): SettleBdaWrites without the wait. The pages a shader's
// BDA writes reached at its last two settles are taken into GPU ownership right after the dispatch
// is recorded (as a writable binding over them would: readers order after this dispatch), the
// written-page list is recorded into a download slot, and the settle is finished once the
// dispatch completed (ProcessPendingBdaSettles, from later settles and the garbage collector):
// the logged stores are applied, and pages outside the prediction are settled then ("late",
// counted: a reader between the dispatch and that settle could have seen them clean).
// =2: the prediction is every page the shader's settles ever listed, not the last two settles';
// =3: also the whole cache buffers those pages lie in.
Live::Switch g_bda_async_settle("KYTY_BDA_ASYNC_SETTLE", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtol(value, nullptr, 10) : 0;
});

// KYTY_BDA_SETTLE_IMAGE_STARTS (live, default 0; adapted from Erlou93/KytyPS5-Yotei's
// InvalidateMemoryFromGPU, which acts only on images starting at the written address): a settled
// BDA write takes GPU ownership only from the images whose first byte lies in its written pages.
// Off: from every image overlapping the pages (16 KiB granules), which drops the GPU-only bytes of
// render targets that merely share a page with the written records.
static Live::Switch g_bda_settle_image_starts("KYTY_BDA_SETTLE_IMAGE_STARTS", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtol(value, nullptr, 10) : 0;
});

void BufferCache::ProcessPendingBdaSettles(bool wait_oldest) {
	if (wait_oldest && !m_bda_pending.empty()) {
		Profiler::ScopedFrameWait wait(Profiler::FrameWait::BdaSettle);
		m_scheduler.Wait(m_bda_pending.front().tick);
	}
	while (!m_bda_pending.empty() && m_scheduler.IsFree(m_bda_pending.front().tick)) {
		auto pending = std::move(m_bda_pending.front());
		m_bda_pending.pop_front();
		auto writes = m_fault_manager.ParseBdaWrites(pending.slot);
		// Later dispatches may have logged over the entries past the downloaded prefix: lost here.
		writes.dropped += writes.log_tail;
		if (!writes.stores.empty()) {
			writes.dropped += ApplyBdaStoreLog(writes.stores, pending.hash);
			m_bda_async_totals.stores += writes.stores.size();
		}
		m_bda_async_totals.dropped += writes.dropped;
		RangeSet actual;
		if (writes.overflow) {
			for (const auto& [vaddr, id]: m_buffers) {
				actual.Add(vaddr, m_slot_buffers[id].Size());
			}
			m_bda_async_totals.overflows++;
		} else {
			for (const auto page: writes.pages) {
				actual.Add(page, CACHING_PAGESIZE);
			}
		}
		uint64_t late_pages = 0;
		actual.ForEach([&](uint64_t start, uint64_t end) {
			for (uint64_t page = start; page < end; page += CACHING_PAGESIZE) {
				if (!pending.predicted.Contains(page, CACHING_PAGESIZE)) {
					SettleBdaWrittenRange(page, CACHING_PAGESIZE, pending.hash, late_pages);
				}
			}
		});
		m_bda_async_totals.late_pages += late_pages;
		auto& history = m_bda_predict[pending.hash];
		if (late_pages != 0) {
			history.sync_left = std::max(history.sync_left, BdaRelearnSettles);
		}
		if (g_bda_async_settle.Get() >= 2) {
			// =2: the prediction keeps every page the shader ever wrote (over-marking is safe).
			actual.ForEach([&](uint64_t start, uint64_t end) { history.newer.Add(start, end - start); });
		} else {
			history.older = std::move(history.newer);
			history.newer = std::move(actual);
		}
	}
}

void BufferCache::SettleBdaWritesAsync(uint64_t shader_hash) {
	ProcessPendingBdaSettles(false);
	if (m_bda_pending.size() >= FaultManager::BdaSlots) {
		ProcessPendingBdaSettles(true);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettles);
	PendingBdaSettle pending;
	pending.hash = shader_hash;
	if (const auto it = m_bda_predict.find(shader_hash); it != m_bda_predict.end()) {
		it->second.newer.ForEach([&](uint64_t start, uint64_t end) { pending.predicted.Add(start, end - start); });
		it->second.older.ForEach([&](uint64_t start, uint64_t end) { pending.predicted.Add(start, end - start); });
	}
	if (g_bda_async_settle.Get() >= 3) {
		// =3: the whole cache buffers the predicted pages lie in (ring buffers move their writes).
		RangeSet whole;
		pending.predicted.ForEach([&](uint64_t start, uint64_t end) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < end; ++it) {
				const auto& buffer = m_slot_buffers[it->second];
				if (buffer.CpuAddress() + buffer.Size() > start) {
					whole.Add(buffer.CpuAddress(), buffer.Size());
				}
			}
		});
		whole.ForEach([&](uint64_t start, uint64_t end) { pending.predicted.Add(start, end - start); });
	}
	uint64_t predicted_pages = 0;
	pending.predicted.ForEach([&](uint64_t start, uint64_t end) {
		SettleBdaWrittenRange(start, end - start, shader_hash, predicted_pages);
	});
	m_bda_async_totals.predicted_pages += predicted_pages;
	m_bda_async_totals.settles++;
	pending.slot = m_fault_manager.RecordBdaWrites();
	pending.tick = m_scheduler.CurrentTick();
	m_bda_pending.push_back(std::move(pending));
	const auto now = std::chrono::steady_clock::now();
	if (m_bda_async_print.time_since_epoch().count() == 0) {
		m_bda_async_print = now;
	} else if (now - m_bda_async_print >= std::chrono::seconds(10)) {
		m_bda_async_print = now;
		auto& t = m_bda_async_totals;
		std::printf("BdaAsyncSettle 10s: %llu async settles, %llu synchronous to learn, %llu predicted pages, "
		            "%llu late pages, %llu logged stores, %llu dropped, %llu overflows\n",
		            static_cast<unsigned long long>(t.settles), static_cast<unsigned long long>(t.sync_learn),
		            static_cast<unsigned long long>(t.predicted_pages),
		            static_cast<unsigned long long>(t.late_pages), static_cast<unsigned long long>(t.stores),
		            static_cast<unsigned long long>(t.dropped), static_cast<unsigned long long>(t.overflows));
		std::fflush(stdout);
		t = {};
	}
}

void BufferCache::SettleBdaWrites(uint64_t shader_hash) {
	EXIT_IF(!BdaWritesEnabled());
	bool learn = false;
	if (g_bda_async_settle.On()) {
		// A shader settles synchronously until its prediction is warm (BdaWarmSettles) and again
		// for a few settles after a late page, so every page it writes then is known before the
		// command processor continues.
		auto& prediction = m_bda_predict[shader_hash];
		if (prediction.sync_left == 0) {
			SettleBdaWritesAsync(shader_hash);
			return;
		}
		prediction.sync_left--;
		learn = true;
	}
	ProcessPendingBdaSettles(!m_bda_pending.empty());
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::BdaSettle);
	// Records the compaction after the dispatch, submits and waits: the pages it wrote.
	auto writes = m_fault_manager.CollectBdaWrites();
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettles);
	if (!writes.stores.empty()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaLoggedStores, writes.stores.size());
		writes.dropped += ApplyBdaStoreLog(writes.stores, shader_hash);
		if (FirstBdaWriteNote(shader_hash, 5)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_STORE_LOG: CS shader 0x{:016x} stored {} dword(s) to pages without a cache "
			    "buffer; the settle wrote them into guest memory.\n",
			    shader_hash, writes.stores.size()));
		}
	}
	NoteBdaSettleStats(writes.stores.size(), writes.dropped);
	if (writes.dropped != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::BdaDroppedWrites, writes.dropped);
		if (FirstBdaWriteNote(shader_hash, 0)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_WRITES: CS shader 0x{:016x} dropped {} write(s) to pages without a cache "
			    "buffer (their faults create one for later dispatches).\n",
			    shader_hash, writes.dropped));
		}
		if (BdaWritesVerify()) {
			EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " dropped %u BDA write(s)\n",
			     shader_hash, writes.dropped);
		}
	}
	RangeSet written;
	if (writes.overflow) {
		// More pages than the list holds (the rest of the bits are gone): every page that could
		// have received a write, i.e. every cache buffer.
		for (const auto& [vaddr, id]: m_buffers) {
			written.Add(vaddr, m_slot_buffers[id].Size());
		}
		if (FirstBdaWriteNote(shader_hash, 1)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote more pages than one settle lists; every "
			    "cache buffer is taken into GPU ownership.\n",
			    shader_hash));
		}
	} else {
		for (const auto page: writes.pages) {
			written.Add(page, CACHING_PAGESIZE);
		}
	}
	{
		// KYTY_BDA_ALIAS_DIAG=1 (diagnostic): the written pages of a shader's first 4 settles.
		static const bool written_diag = [] {
			const auto* value = std::getenv("KYTY_BDA_ALIAS_DIAG");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		static std::unordered_map<uint64_t, uint32_t> settles;
		if (written_diag && settles[shader_hash]++ < 4u) {
			std::string ranges;
			uint32_t    listed = 0;
			written.ForEach([&](uint64_t start, uint64_t end) {
				if (listed++ < 48u) ranges += fmt::format(" 0x{:x}+0x{:x}", start, end - start);
			});
			Log::WriteToConsoleAndLog(fmt::format("BdaWrittenDiag: CS 0x{:016x} settle {}: {} range(s), {} logged, {} dropped:{}\n",
			                                      shader_hash, settles[shader_hash], listed, writes.stores.size(),
			                                      writes.dropped, ranges));
		}
	}
	uint64_t settled_pages = 0;
	written.ForEach([&](uint64_t start, uint64_t end) {
		SettleBdaWrittenRange(start, end - start, shader_hash, settled_pages);
	});
	Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettlePages, settled_pages);
	if (learn) {
		auto& history = m_bda_predict[shader_hash];
		written.ForEach([&](uint64_t start, uint64_t end) { history.newer.Add(start, end - start); });
		m_bda_async_totals.sync_learn++;
	}
}

// KYTY_BDA_CPU_MERGE=1 (default 0): see BufferCache::MergeBdaCpuDirtyPage.
// Ghost of Yotei's Proxy Set Cull (CS 0x039ed97b41485281) writes its results through BDA into
// pages of CPU memory the game writes in the same frame; the settle's upload of the guest page
// replaced the shader's bytes there, so those cull results were lost every frame.
// On in the Yotei preset: standing at the ruined-village spawn, grass shadows flickered (one-frame
// dark leaf-shaped shadows on the ground; gone with the shadow-pass draws skipped); 240 s with the
// switch on vs off, same build: see the preset commit for the counts.
static bool BdaCpuMergeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BDA_CPU_MERGE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void BufferCache::SnapshotBdaMergePages(uint64_t shader_hash) {
	if (!BdaCpuMergeEnabled()) return;
	m_bda_merge_snapshots.clear();
	const auto it = m_bda_merge_pages.find(shader_hash);
	if (it == m_bda_merge_pages.end()) return;
	for (const auto page: it->second) {
		// Guest bytes equal the page's GPU bytes unless the GPU owns newer ones.
		if (m_memory_tracker.IsRegionGpuModified(page, CACHING_PAGESIZE) ||
		    m_gpu_modified_ranges.Intersects(page, CACHING_PAGESIZE) ||
		    HasPendingBackingPublication(page, CACHING_PAGESIZE)) {
			continue;
		}
		std::vector<uint8_t> bytes(CACHING_PAGESIZE);
		if (Libs::LibKernel::Memory::TryReadBacking(page, bytes.data(), bytes.size())) {
			m_bda_merge_snapshots.emplace(page, std::move(bytes));
		}
	}
}

bool BufferCache::MergeBdaCpuDirtyPage(Buffer& buffer, uint64_t page, uint64_t shader_hash) {
	static std::atomic<uint32_t> lines {0};
	const auto snapshot = m_bda_merge_snapshots.find(page);
	if (snapshot == m_bda_merge_snapshots.end() || !buffer.IsInBounds(page, CACHING_PAGESIZE)) {
		if (lines.fetch_add(1, std::memory_order_relaxed) < 64u) {
			Log::WriteToConsoleAndLog(fmt::format("KYTY_BDA_CPU_MERGE: CS 0x{:016x} page 0x{:x}: no snapshot, the guest's "
			                                      "page wins\n",
			                                      shader_hash, page));
		}
		return false;
	}
	if (m_bda_merge_download == nullptr) {
		m_bda_merge_download = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                                vk::BufferUsageFlagBits::eTransferDst, CACHING_PAGESIZE);
	}
	// The GPU bytes of the page: the dispatch completed (the synchronous settle waited for it).
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
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                       nullptr, 1, &before, 0, nullptr);
	vk::BufferCopy copy {buffer.Offset(page), 0, CACHING_PAGESIZE};
	native.copyBuffer(buffer.Handle(), m_bda_merge_download->Handle(), 1, &copy);
	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = m_bda_merge_download->Handle();
	after.size          = CACHING_PAGESIZE;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
	                       &after, 0, nullptr);
	m_scheduler.Wait(m_scheduler.CurrentTick());
	m_bda_merge_download->Invalidate(0, CACHING_PAGESIZE);
	const auto* gpu      = reinterpret_cast<const uint32_t*>(m_bda_merge_download->Mapped().data());
	const auto* original = reinterpret_cast<const uint32_t*>(snapshot->second.data());
	uint32_t    merged   = 0;
	uint32_t    both     = 0;
	for (uint32_t i = 0; i < CACHING_PAGESIZE / sizeof(uint32_t); i++) {
		if (gpu[i] == original[i]) continue;
		const uint64_t address = page + uint64_t {i} * sizeof(uint32_t);
		uint32_t       guest   = 0;
		if (!Libs::LibKernel::Memory::TryReadBacking(address, &guest, sizeof(guest)) || guest == gpu[i]) continue;
		both += guest != original[i] ? 1u : 0u;
		const uint32_t value = gpu[i];
		if (Libs::LibKernel::Memory::TryWriteBacking(address, &value, sizeof(value))) merged++;
	}
	m_bda_merge_snapshots.erase(snapshot);
	if (lines.fetch_add(1, std::memory_order_relaxed) < 64u) {
		Log::WriteToConsoleAndLog(fmt::format("KYTY_BDA_CPU_MERGE: CS 0x{:016x} page 0x{:x}: {} GPU dword(s) merged into "
		                                      "the guest page ({} the guest wrote too)\n",
		                                      shader_hash, page, merged, both));
	}
	return true;
}

void BufferCache::SettleBdaWrittenRange(uint64_t vaddr, uint64_t size, uint64_t shader_hash,
                                        uint64_t& settled_pages) {
	// Collect first: settling a part may create or merge buffers (not here, but keep the loop
	// independent of the map).
	std::vector<std::pair<BufferId, GuestRange>> parts;
	const auto                                   end = vaddr + size;
	auto                                         it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			parts.push_back({it->second, GuestRange {start, finish - start}});
		}
	}
	for (const auto& [id, range]: parts) {
		const auto start = range.address;
		const auto bytes = range.size;
		settled_pages += (bytes + CACHING_PAGESIZE - 1) / CACHING_PAGESIZE;
		// An image that owned bytes here missed their move into the buffer before the write
		// (writebacks off, or a GPU-modified image this dispatch made): its bytes are lost now.
		if (m_texture_cache.IsRegionGpuModified(start, bytes)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaAliasHits);
			// KYTY_BDA_ALIAS_DIAG=1 (diagnostic): every such hit with the images, at most 300 lines.
			static const bool alias_diag = [] {
				const auto* value = std::getenv("KYTY_BDA_ALIAS_DIAG");
				return value != nullptr && std::strcmp(value, "1") == 0;
			}();
			static uint32_t alias_lines = 0;
			if (alias_diag && alias_lines < 300u) {
				alias_lines++;
				Log::WriteToConsoleAndLog(fmt::format("BdaAliasDiag: CS 0x{:016x} wrote 0x{:x}+0x{:x} under [{}]\n",
				                                      shader_hash, start, bytes,
				                                      m_texture_cache.DescribeGpuModifiedImages(start, bytes)));
			}
			if (FirstBdaWriteNote(shader_hash, 2)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote 0x{:x}+0x{:x}, under a GPU-modified "
				    "image.\n",
				    shader_hash, start, bytes));
			}
			if (BdaWritesVerify()) {
				EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " wrote 0x%016" PRIx64
				     "+0x%" PRIx64 " under a GPU-modified image\n",
				     shader_hash, start, bytes);
			}
		}
		// PrepareBda uploaded every CPU-dirty page before the dispatch, and CommitBindings then
		// returned every hot page (CPU-dirty by construction) to clean tracking, or to CPU-dirty
		// when its shadow no longer matched (has_address_writes: InvalidateContentRevisions). A
		// written page that is CPU-dirty now was written by the guest while the dispatch was being
		// recorded or ran: the upload below then puts the guest's page over the dispatch's bytes
		// (hardware would keep both writers' bytes).
		if (m_memory_tracker.IsRegionCpuModified(start, bytes)) {
			uint64_t dirty_pages = 0;
			for (uint64_t page = Common::AlignDown(start, CACHING_PAGESIZE); page < start + bytes;
			     page += CACHING_PAGESIZE) {
				const auto first = std::max(page, start);
				const auto last  = std::min(page + CACHING_PAGESIZE, start + bytes);
				dirty_pages += m_memory_tracker.IsRegionCpuModified(first, last - first) ? 1u : 0u;
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::BdaSettleCpuDirtyPages, dirty_pages);
			// KYTY_BDA_ALIAS_DIAG=1 (diagnostic): every such overlap, at most 400 lines.
			{
				static const bool cpu_dirty_diag = [] {
					const auto* value = std::getenv("KYTY_BDA_ALIAS_DIAG");
					return value != nullptr && std::strcmp(value, "1") == 0;
				}();
				static std::atomic<uint32_t> cpu_dirty_lines {0};
				if (cpu_dirty_diag && cpu_dirty_lines.fetch_add(1, std::memory_order_relaxed) < 400u) {
					Log::WriteToConsoleAndLog(fmt::format("BdaCpuDirtyDiag: CS 0x{:016x} wrote 0x{:x}+0x{:x}, {} page(s) CPU-dirty\n",
					                                      shader_hash, start, bytes, dirty_pages));
				}
			}
			if (FirstBdaWriteNote(shader_hash, 3)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "KYTY_BDA_WRITES: CS shader 0x{:016x} wrote 0x{:x}+0x{:x}, which the guest also "
				    "wrote during the dispatch ({} page(s)).\n",
				    shader_hash, start, bytes, dirty_pages));
			}
			if (BdaWritesVerify()) {
				EXIT("KYTY_BDA_WRITES=verify: shader 0x%016" PRIx64 " wrote 0x%016" PRIx64
				     "+0x%" PRIx64 ", which the guest wrote during the dispatch\n",
				     shader_hash, start, bytes);
			}
		}
		if (BdaCpuMergeEnabled() && m_memory_tracker.IsRegionCpuModified(start, bytes)) {
			auto& merge_buffer = m_slot_buffers[id];
			for (uint64_t page = Common::AlignDown(start, CACHING_PAGESIZE); page < start + bytes;
			     page += CACHING_PAGESIZE) {
				const auto first = std::max(page, start);
				const auto last  = std::min(page + CACHING_PAGESIZE, start + bytes);
				if (!m_memory_tracker.IsRegionCpuModified(first, last - first)) continue;
				auto& known = m_bda_merge_pages[shader_hash];
				if (std::find(known.begin(), known.end(), page) == known.end() && known.size() < 256u) {
					known.push_back(page);
				}
				(void)MergeBdaCpuDirtyPage(merge_buffer, page, shader_hash);
			}
		}
		// What a writable binding over the range records (ObtainBuffer), after the fact: the
		// dispatch has completed, so the pages become GPU-owned before anything else is recorded
		// or read.
		auto& buffer = m_slot_buffers[id];
		(void)SynchronizeBuffer(buffer, start, bytes, true, false, nullptr, "bda-write");
		buffer.MarkContentWritten();
		if (!m_gpu_modified_ranges.Contains(start, bytes)) {
			CleanVerdict::Invalidate(start, bytes, Coherence::Source::BufferDirtyAdd);
		}
		WatchRange::Note("GPU-modified range #4", start, bytes);
		WatchRange::NoteWriter("bda-settle", start, bytes);
		m_gpu_modified_ranges.Add(start, bytes);
		NoteBufferContentWrite(start, bytes);
		ForgetKnownFills(start, bytes);
		HangTrace::NoteGpuWrite(start, bytes);
		// Overlapping images are rebuilt from the buffer (the Water agent's rule for BDA writes).
		// KYTY_BDA_SETTLE_IMAGE_STARTS=1: only the images that start in the written pages.
		if (g_bda_settle_image_starts.Get() != 0) {
			m_texture_cache.InvalidateMemoryFromGPUImageStarts(start, bytes);
		} else {
			m_texture_cache.InvalidateMemoryFromGPU(start, bytes);
		}
	}
}

uint32_t BufferCache::ApplyBdaStoreLog(std::span<const FaultManager::BdaStore> stores,
                                       uint64_t shader_hash) {
	// The guest-write transition once per dword run, before any byte lands.
	RangeSet dwords;
	for (const auto& store: stores) {
		// A shader storing through a garbage pointer logs an address outside the guest space.
		if (!GuestRange {store.Address() & ~uint64_t {3}, sizeof(uint32_t)}.Valid()) {
			continue;
		}
		dwords.Add(store.Address() & ~uint64_t {3}, sizeof(uint32_t));
	}
	RangeSet refused;
	dwords.ForEach([&](uint64_t start, uint64_t end) {
		const auto bytes = end - start;
		if (HasGpuDirtyBytes(start, bytes) || HasPendingBackingPublication(start, bytes) ||
		    m_texture_cache.IsRegionGpuModified(start, bytes)) {
			refused.Add(start, bytes);
			return;
		}
		InvalidateMemory(start, bytes);
		m_texture_cache.InvalidateMemory(start, bytes);
	});
	uint32_t dropped = 0;
	for (const auto& store: stores) {
		const auto address = store.Address() & ~uint64_t {3};
		uint32_t   word    = 0;
		if (!GuestRange {address, sizeof(word)}.Valid()) {
			dropped++;
			continue;
		}
		if (refused.Intersects(address, sizeof(word)) ||
		    !Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word))) {
			dropped++;
			continue;
		}
		word = (word & ~store.mask) | (store.value & store.mask);
		if (!Libs::LibKernel::Memory::TryWriteBacking(address, &word, sizeof(word))) {
			dropped++;
			continue;
		}
		HangTrace::NoteGpuWrite(address, sizeof(word));
	}
	// A memory write the command processor orders before later work (syncEpoch.h).
	SyncEpoch::Advance();
	if (!refused.Empty() && FirstBdaWriteNote(shader_hash, 4)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_BDA_STORE_LOG: CS shader 0x{:016x} stored to bytes without a cache buffer that the "
		    "GPU still owns; those stores are dropped.\n",
		    shader_hash));
	}
	return dropped;
}

} // namespace Libs::Graphics
