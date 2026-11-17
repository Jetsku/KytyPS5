#include "kernel/memory.h"
#include "common/staticOnce.h"
#include "graphics/host_gpu/watchRange.h"
#include "common/prepPhases.h"

#include "common/assert.h"
#include "common/cruzStats.h"
#include "common/gthreadStats.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/platform/uffdWriteWatch.h"
#include "common/profiler.h"
#include "common/skipInvalid.h"
#include "common/ramStats.h"
#include "common/rendererBatch.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/gpuReadDelegate.h"
#include "graphics/host_gpu/gpuTouchedPages.h"
#include "graphics/host_gpu/renderer/drawPrep/readSet.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/shader.h"
#include "libs/errno.h"
#include "libs/libs.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "loader/demonsSoulsTitle.h"

#include <magic_enum.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h> // IWYU pragma: keep
#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
#ifndef MEM_COALESCE_PLACEHOLDERS
#define MEM_COALESCE_PLACEHOLDERS 0x00000001
#endif
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::StreamCopyDefer {
void NoteGuestWrite(uint64_t address, uint64_t size); // bufferCache.h
} // namespace Libs::Graphics::StreamCopyDefer

namespace Libs::LibKernel::Memory {

namespace VirtualMemory = Common::VirtualMemory;

LIB_NAME("libkernel", "libkernel");

constexpr int PROT_CPU_READ  = 0x01;
constexpr int PROT_CPU_WRITE = 0x02;
constexpr int PROT_CPU_EXEC  = 0x04;
constexpr int PROT_GPU_READ  = 0x10;
constexpr int PROT_GPU_WRITE = 0x20;

enum class GpuAccessMode { NoAccess, Read, Write, ReadWrite };

constexpr uint64_t PAGE_TABLE_POOL_SIZE   = 4ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t PAGE_TABLE_GRANULARITY = 2ull * 1024ull * 1024ull;
constexpr int      PAGE_TABLE_POOL_ENTRIES =
    static_cast<int>(PAGE_TABLE_POOL_SIZE / PAGE_TABLE_GRANULARITY);
constexpr uint64_t DEFAULT_FLEXIBLE_MEMORY_SIZE = 1ull * 1024ull * 1024ull * 1024ull;
constexpr uint64_t DEFAULT_GUEST_MEMORY_SIZE    = 13824ull * 1024ull * 1024ull;
// PS5 Pro mode (--trinity on a title that supports it; upstream 651ef73a0): more guest memory.
constexpr uint64_t TRINITY_EXTRA_MEMORY_SIZE = 1216ull * 1024ull * 1024ull;

// KYTY_UPSTREAM_MEMORY_DEFAULTS=1 (default off; upstream 09753bf8f, 0a140911a): upstream's memory
// sizes. A game gets 12.5 GiB; a fixed 64 MiB cache and the mapping table memory its param.json
// asks for past the 64 MiB included for each of the CPU, GPU and AMM tables come off the top;
// flexible memory defaults to 448 MiB and loaded modules' segments are charged to it; the page
// table stats report the param.json table sizes. Off: 13.5 GiB, 1 GiB flexible memory by default,
// module segments not charged, a fixed 4 GiB page table pool per side (as before).
static bool UpstreamMemoryDefaults() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_UPSTREAM_MEMORY_DEFAULTS");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}
constexpr uint64_t UPSTREAM_GUEST_MEMORY_BUDGET   = 12800ull * 1024ull * 1024ull;
constexpr uint64_t UPSTREAM_DEFAULT_FLEXIBLE_SIZE = 448ull * 1024ull * 1024ull;
constexpr uint64_t UPSTREAM_RESERVED_CACHE_SIZE   = 64ull * 1024ull * 1024ull;
constexpr uint64_t UPSTREAM_INCLUDED_TABLE_MEMORY = 64ull * 1024ull * 1024ull;
constexpr uint64_t UPSTREAM_MAPPING_TABLE_SIZE    = 4ull * 1024ull;

static uint64_t                      g_guest_memory_size           = DEFAULT_GUEST_MEMORY_SIZE;
static uint64_t                      g_flexible_memory_size        = DEFAULT_FLEXIBLE_MEMORY_SIZE;
static bool                          g_flexible_memory_size_frozen = false;
// Page table entries per side (KernelGetPageTableStats).
static uint64_t g_cpu_page_table_entries = PAGE_TABLE_POOL_ENTRIES;
static uint64_t g_gpu_page_table_entries = PAGE_TABLE_POOL_ENTRIES;
static Graphics::RenderContext*       g_gpu_resources               = nullptr;
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static void (*g_test_gpu_range)(bool, uint64_t, size_t) = nullptr;
#endif

static Graphics::RenderContext& GetGpuResources() {
	EXIT_IF(g_gpu_resources == nullptr);
	return *g_gpu_resources;
}

static bool IsGpuAddressRange(uint64_t vaddr, uint64_t size) {
	constexpr uint64_t GPU_ADDRESS_LIMIT = 1ull << 40u;
	return vaddr != 0 && size != 0 && vaddr < GPU_ADDRESS_LIMIT && size < GPU_ADDRESS_LIMIT - vaddr;
}

// Tracker-gap detector (RenderContext::NoteGuestProtection): the host protection `mode` is about
// to be applied to [vaddr, vaddr + size) directly, not through resource tracking's PageManager.
static void NoteGuestProtection(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	const auto bits = static_cast<uint32_t>(mode);
	GetGpuResources().NoteGuestProtection(vaddr, size, (bits & 1u) != 0, (bits & 2u) != 0);
}

// KYTY_REMAP_TRACKER_PROTECTION=0 turns it off: after a range's host protection was replaced
// behind the page tracker (the neighbours of a partly unmapped Windows view mapped again with the
// view's access, views an unmap rolled back, a guest mprotect that allows writes), its watched
// pages get the tracker's protection back. Without it CPU writes to still-tracked buffers there
// stopped faulting and the GPU kept using stale bytes (chenxiao07 3f3825158 / e6dfd8e6b).
static bool RemapTrackerProtectionEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_REMAP_TRACKER_PROTECTION");
		return value == nullptr || !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// KYTY_TRACKER_SPARE=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 7d3a20afc, 9ae50c01c):
// the page tracker's protections (ProtectTransient) change only mappings with backing, the memory
// the GPU can use, leaving the program's code, stacks and runtime memory (private mappings) alone,
// and leave the live stacks of threads that run on memory the game gave them writable. A GPU
// buffer over the first page of GT7's code made it non-executable (a freeze after a license test);
// one over a worker's stack made it read-only, and the write fault on the stack a thread runs on
// ends the process silently (Scapes).
static bool TrackerSpareEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TRACKER_SPARE");
		return value != nullptr && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// KYTY_PROTECT_LOCK_SHARED=1 (default off; adapted from usa-ratt/KytyPS5-GT7 ba83f1a64, mainline
// PR #1815): the page tracker's protections (ProtectTransient) take the guest address space's
// lock shared instead of exclusively. They only read the mappings (and the live stacks), and the
// page manager's region lock already orders changes of the same pages; every change of the
// mappings, and the game's own mprotect, still takes it exclusively. Faulting game threads and
// the GPU thread then no longer queue behind each other's protection calls.
static bool ProtectLockShared() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PROTECT_LOCK_SHARED");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static void ResyncGpuRangeProtection(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size) ||
	    !RemapTrackerProtectionEnabled()) {
		return;
	}
	// How often the host protection of GPU memory was replaced (a writable guest mprotect, a remap)
	// before this resync re-protected the tracker's pages: a CPU write in between is not seen by the
	// tracker (Senaxx/KytyPS5 c453d3fd5 closes that window for mprotect with one protect of
	// min(guest, tracker) mode). Counted on the console to size the window.
	static std::atomic<uint64_t> resyncs {0};
	if (const auto n = resyncs.fetch_add(1, std::memory_order_relaxed) + 1; n <= 4 || (n & (n - 1)) == 0) {
		std::printf("TrackerResync: host protection of GPU memory 0x%016" PRIx64 " +0x%" PRIx64
		            " replaced and resynced (%" PRIu64 " so far)\n",
		            vaddr, size, n);
		std::fflush(stdout);
	}
	GetGpuResources().ResyncHostProtection(vaddr, size);
}

// KYTY_MPROTECT_TRACKED_ONCE=1 (live, default off; based on Senaxx/KytyPS5 c453d3fd5): a guest
// mprotect that allows writes over GPU memory gives the tracker's watched pages their protection in
// the same host call (PageManager::ProtectGuestWritable) instead of opening them with the guest's
// mode and re-protecting them afterwards (ResyncGpuRangeProtection), a window in which a CPU write
// to a watched page was not seen.
static Live::Switch g_mprotect_tracked_once("KYTY_MPROTECT_TRACKED_ONCE", Live::ParseDefaultOff);

static void MapGpuRange(uint64_t vaddr, uint64_t size) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_gpu_range != nullptr) g_test_gpu_range(true, vaddr, size);
#endif
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	GetGpuResources().MapMemory(vaddr, size);
}

static void UnmapGpuRange(uint64_t vaddr, uint64_t size) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	if (g_test_gpu_range != nullptr) g_test_gpu_range(false, vaddr, size);
#endif
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return;
	}
	GetGpuResources().UnmapMemory(vaddr, size);
}

static bool DecodeMemoryProtection(int prot, VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	EXIT_IF(mode == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	bool cpu_read  = (prot & PROT_CPU_READ) != 0;
	bool cpu_write = (prot & PROT_CPU_WRITE) != 0;
	bool cpu_exec  = (prot & PROT_CPU_EXEC) != 0;
	bool gpu_read  = (prot & PROT_GPU_READ) != 0;
	bool gpu_write = (prot & PROT_GPU_WRITE) != 0;

	if ((prot &
	     (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC | PROT_GPU_READ | PROT_GPU_WRITE)) == 0 &&
	    prot != 0) {
		return false;
	}

	if (gpu_read && gpu_write) {
		*gpu_mode = GpuAccessMode::ReadWrite;
	} else if (gpu_read) {
		*gpu_mode = GpuAccessMode::Read;
	} else if (gpu_write) {
		*gpu_mode = GpuAccessMode::Write;
	} else {
		*gpu_mode = GpuAccessMode::NoAccess;
	}

	bool host_read  = cpu_read || gpu_read;
	bool host_write = cpu_write || gpu_write;

	if (host_write) {
		host_read = true;
	}

	*mode = VirtualMemory::Mode::NoAccess;
	if (cpu_exec) {
		*mode = (host_write ? VirtualMemory::Mode::ExecuteReadWrite
		                    : (host_read ? VirtualMemory::Mode::ExecuteRead
		                                 : VirtualMemory::Mode::Execute));
	} else if (host_write) {
		*mode = VirtualMemory::Mode::ReadWrite;
	} else if (host_read) {
		*mode = VirtualMemory::Mode::Read;
	}

	return true;
}

static void CopyVirtualRangeName(char* dst, const char* name) {
	EXIT_IF(dst == nullptr);

	std::memset(dst, 0, KERNEL_MAXIMUM_NAME_LENGTH);

	if (name != nullptr) {
		std::strncpy(dst, name, KERNEL_MAXIMUM_NAME_LENGTH - 1);
	}
}

static bool VirtualRangesOverlap(uint64_t left_start, uint64_t left_size, uint64_t right_start,
                                 uint64_t right_size) {
	if (left_size == 0 || right_size == 0) {
		return false;
	}

	auto left_end = (UINT64_MAX - left_start < left_size ? UINT64_MAX : left_start + left_size);
	auto right_end =
	    (UINT64_MAX - right_start < right_size ? UINT64_MAX : right_start + right_size);

	return left_start < right_end && right_start < left_end;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
static callback_func_t g_test_before_backing_map = nullptr;
static callback_func_t g_test_backing_read       = nullptr;
#endif

#include "memoryAddressSpace.inc"

enum class VirtualRangeType {
	Reserved,
	PoolReserved,
	Direct,
	Flexible,
	Pooled,
	Stack,
	Code,
	Runtime,
};

static bool IsReservedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Reserved || type == VirtualRangeType::PoolReserved;
}

static bool IsPooledRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Pooled || type == VirtualRangeType::PoolReserved;
}

static bool IsCommittedRangeType(VirtualRangeType type) {
	return !IsReservedRangeType(type);
}

static bool IsPrivateCommittedRangeType(VirtualRangeType type) {
	return type == VirtualRangeType::Stack || type == VirtualRangeType::Code ||
	       type == VirtualRangeType::Runtime;
}

// KYTY_CLAMP_RANGE_MEMO (default on; =0 off): VirtualRanges::ClampRangeSize answers from a
// per-thread cache of runs of adjacent committed ranges, valid while the ranges are unchanged
// (VirtualRanges::m_generation). Buffer bindings clamp every V# range (tens of thousands per
// frame); the locked lookup was about 1.2 ms per flip on the command processor (U49).
static bool ClampRangeMemoEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_CLAMP_RANGE_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

// KYTY_CLAMP_COVERAGE_GEN=1 (default off; adapted from usa-ratt/KytyPS5-GT7 0b97a337b): the clamp
// answers (KYTY_CLAMP_RANGE_MEMO's per-thread runs, and the binding plans' certificate through
// VirtualRangesGeneration) stay valid across protection and name changes, which cannot change which
// bytes are committed: only the changes of coverage (add, remove, release, replace) advance the
// generation they check. A streaming game changes protections hundreds of times a second, and
// every change made every thread's next clamps miss and re-walk the ranges under the lock.
static bool ClampCoverageGenerationEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_CLAMP_COVERAGE_GEN");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

// KYTY_CLAMP_RANGE_MEMO_VERIFY=1|exit: every cached answer is compared with the locked lookup.
static int ClampRangeMemoVerifyMode() {
	static constexpr auto mode_init = [] {
		const auto* value = std::getenv("KYTY_CLAMP_RANGE_MEMO_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const int& mode = mode_once.Get(mode_init);
	return mode;
}

namespace {

// A run [begin, end) of adjacent committed ranges of one VirtualRanges, seen at `generation`.
// `end_is_run_end`: the run was followed to its end (otherwise a bounded walk stopped earlier and
// only requests ending inside it are answered).
struct ClampRun {
	const void* owner          = nullptr;
	uint64_t    generation     = 0;
	uint64_t    begin          = 0;
	uint64_t    end            = 0;
	bool        end_is_run_end = false;
};

struct ClampRuns {
	std::array<ClampRun, 4> entries {};
	uint32_t                next = 0;
};

thread_local ClampRuns t_clamp_runs;

struct ClampRangeMemoTotals {
	uint64_t hits              = 0;
	uint64_t misses            = 0;
	uint64_t verify_checks     = 0;
	uint64_t verify_mismatches = 0;
	uint64_t verify_races      = 0;
};

thread_local ClampRangeMemoTotals t_clamp_totals;

// KYTY_CLAMP_RANGE_BITMAP=1 (default off; adapted from TheCruZ/KytyPS5-GTA 1f331bb7): ClampRangeSize
// answers from a lock-free bitmap of committed 4 KiB pages below 1 TiB, kept under the ranges'
// lock by every coverage change. Streaming changes the mappings many times per frame, which
// retires KYTY_CLAMP_RANGE_MEMO's per-thread runs, and the misses take the ranges' lock (GTA V:
// 15-35 ms per frame on the GPU threads). Tests force it with TestSetClampRangeBitmap.
std::atomic<int> g_clamp_range_bitmap {-1};

bool ClampRangeBitmapEnabled() {
	auto value = g_clamp_range_bitmap.load(std::memory_order_relaxed);
	if (value < 0) {
		const auto* env = std::getenv("KYTY_CLAMP_RANGE_BITMAP");
		value           = env != nullptr && std::strcmp(env, "1") == 0 ? 1 : 0;
		g_clamp_range_bitmap.store(value, std::memory_order_relaxed);
	}
	return value != 0;
}

// One bit per committed 4 KiB page of [0, 1 TiB). Pages change under the owner's lock; leaves are
// allocated on first use and live as long as the map. A query racing a change orders as if it
// preceded or followed it (bits are set before they are cleared, never the reverse).
class CommittedPageMap {
public:
	static constexpr uint64_t Unknown = UINT64_MAX;

	CommittedPageMap() = default;
	~CommittedPageMap() {
		for (auto& leaf: m_leaves) {
			delete[] leaf.load(std::memory_order_relaxed);
		}
	}
	KYTY_CLASS_NO_COPY(CommittedPageMap);

	// Owner's lock held. A range that is not page aligned makes the map imprecise for good (every
	// query then takes the locked path); outside the window nothing is recorded.
	void Set(uint64_t start, uint64_t size, bool committed) {
		if (size == 0) {
			return;
		}
		if (((start | size) & PageMask) != 0) {
			m_imprecise.store(true, std::memory_order_relaxed);
			return;
		}
		if (size > WindowSize || start > WindowSize - size) {
			return;
		}
		const auto first = start >> PageBits;
		const auto end   = (start + size) >> PageBits;
		for (auto page = first; page < end;) {
			const auto leaf_end = std::min(end, (page / LeafPages + 1) * LeafPages);
			auto*      words    = Leaf(page / LeafPages, committed);
			if (words == nullptr) {
				page = leaf_end;
				continue;
			}
			while (page < leaf_end) {
				const auto bit   = page % 64u;
				const auto count = std::min<uint64_t>(64u - bit, leaf_end - page);
				const auto mask  = (count == 64u ? ~uint64_t {0} : ((uint64_t {1} << count) - 1u)) << bit;
				auto&      word  = words[(page % LeafPages) / 64u];
				if (committed) {
					word.fetch_or(mask, std::memory_order_release);
				} else {
					word.fetch_and(~mask, std::memory_order_release);
				}
				page += count;
			}
		}
	}

	// How many bytes from `address` (at most `size`) lie in contiguous committed pages, or Unknown
	// when the map cannot tell.
	[[nodiscard]] uint64_t CommittedRun(uint64_t address, uint64_t size) const {
		if (m_imprecise.load(std::memory_order_relaxed) || size == 0 || address >= WindowSize ||
		    size > WindowSize - address) {
			return Unknown;
		}
		const auto first = address >> PageBits;
		const auto end   = (address + size + PageMask) >> PageBits;
		for (auto page = first; page < end;) {
			const auto* words = m_leaves[page / LeafPages].load(std::memory_order_acquire);
			const auto  bit   = page % 64u;
			const auto  count = std::min<uint64_t>(64u - bit, end - page);
			const auto  word  = words == nullptr ? 0 : words[(page % LeafPages) / 64u].load(std::memory_order_acquire);
			// The first page of the run that is not committed.
			const auto missing = ~(word >> bit) & (count == 64u ? ~uint64_t {0} : ((uint64_t {1} << count) - 1u));
			if (missing != 0) {
				const auto run_end = (page + static_cast<uint64_t>(std::countr_zero(missing))) << PageBits;
				return run_end <= address ? 0 : std::min(size, run_end - address);
			}
			page += count;
		}
		return size;
	}

private:
	static constexpr uint64_t PageBits   = 12;
	static constexpr uint64_t PageMask   = (uint64_t {1} << PageBits) - 1u;
	static constexpr uint64_t WindowSize = uint64_t {1} << 40u;
	static constexpr uint64_t LeafPages  = uint64_t {1} << 19u; // 2 GiB, 64 KiB of bits
	static constexpr uint64_t LeafCount  = (WindowSize >> PageBits) / LeafPages;

	std::atomic<uint64_t>* Leaf(uint64_t index, bool create) {
		auto* leaf = m_leaves[index].load(std::memory_order_relaxed);
		if (leaf == nullptr && create) {
			leaf = new std::atomic<uint64_t>[LeafPages / 64u] {};
			m_leaves[index].store(leaf, std::memory_order_release);
		}
		return leaf;
	}

	std::array<std::atomic<std::atomic<uint64_t>*>, LeafCount> m_leaves {};
	std::atomic_bool                                            m_imprecise {false};
};

} // namespace

class VirtualRanges {
public:
	struct Range {
		uint64_t         start          = 0;
		uint64_t         size           = 0;
		uint64_t         offset         = 0;
		int              protection     = 0;
		int              memory_type    = 0;
		VirtualRangeType type           = VirtualRangeType::Reserved;
		bool             disallow_merge = false;
		char             name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	bool Add(uint64_t start, uint64_t size, uint64_t offset, int protection, int memory_type,
	         VirtualRangeType type, const char* name, bool disallow_merge = false) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked(true);

		if (start == 0 || size == 0) {
			return false;
		}
		auto position = LowerBound(start);
		if ((position != m_ranges.end() &&
		     VirtualRangesOverlap(start, size, position->start, position->size)) ||
		    (position != m_ranges.begin() &&
		     VirtualRangesOverlap(start, size, std::prev(position)->start,
		                          std::prev(position)->size))) {
			return false;
		}

		Range r {};
		r.start          = start;
		r.size           = size;
		r.offset         = offset;
		r.protection     = protection;
		r.memory_type    = memory_type;
		r.type           = type;
		r.disallow_merge = disallow_merge;
		CopyVirtualRangeName(r.name, name);
		const auto index = static_cast<size_t>(position - m_ranges.begin());
		m_ranges.insert(position, r);
		MergeAroundUnlocked(index);
		RefreshCommittedUnlocked(start, size);
		return true;
	}

	bool Remove(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked(true);

		auto position = LowerBound(start);
		if (position != m_ranges.end() && position->start == start && position->size == size) {
			m_ranges.erase(position);
			RefreshCommittedUnlocked(start, size);
			return true;
		}
		// RemoveUnlocked keeps the ranges sorted and merged (it used to be followed by a full
		// MergeUnlocked sort, which only re-did that).
		const bool removed = RemoveUnlocked(start, size);
		RefreshCommittedUnlocked(start, size);
		return removed;
	}

	bool HasOverlap(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);

		return FindOverlap(start, size) != nullptr;
	}

	bool QueryOverlap(uint64_t start, uint64_t size, Range* out) {
		EXIT_IF(out == nullptr);
		Common::LockGuard lock(m_mutex);

		const auto* overlap = FindOverlap(start, size);
		if (overlap == nullptr) {
			return false;
		}
		*out = *overlap;
		return true;
	}

	bool ReleaseReserved(uint64_t start, uint64_t size) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked(true);

		// Starts are unique (the ranges never overlap and are never empty), so the range that
		// starts at `start` is the only candidate (adapted from usa-ratt/KytyPS5-GT7 bab2ca8f7:
		// no walk from the first range).
		auto position = LowerBound(start);
		if (position != m_ranges.end() && position->start == start && position->size == size &&
		    IsReservedRangeType(position->type)) {
			m_ranges.erase(position);
		}
		return true;
	}

	bool ReplaceSpan(uint64_t start, uint64_t size, VirtualRangeType expected_type,
	                 uint64_t offset, int protection, int memory_type, VirtualRangeType type,
	                 const char* name, bool disallow_merge = false) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked(true);

		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return false;
		}

		auto current = start;
		const auto end = start + size;
		while (current < end) {
			// The one range that can contain `current` (the ranges are sorted and never overlap):
			// a binary search instead of a walk from the first range for every piece (adapted
			// from usa-ratt/KytyPS5-GT7 bab2ca8f7).
			const Range* candidate = nullptr;
			auto         next      = std::upper_bound(
                m_ranges.begin(), m_ranges.end(), current,
                [](uint64_t value, const Range& range) { return value < range.start; });
			if (next != m_ranges.begin()) {
				const auto& r = *std::prev(next);
				if (r.type == expected_type && current >= r.start && current < End(r.start, r.size)) {
					candidate = &r;
				}
			}
			if (candidate == nullptr) {
				return false;
			}
			current = std::min(end, End(candidate->start, candidate->size));
		}

		Range replacement {};
		replacement.start          = start;
		replacement.size           = size;
		replacement.offset         = offset;
		replacement.protection     = protection;
		replacement.memory_type    = memory_type;
		replacement.type           = type;
		replacement.disallow_merge = disallow_merge;
		CopyVirtualRangeName(replacement.name, name);
		RemoveUnlocked(start, size);
		auto position = LowerBound(start);
		const auto index = static_cast<size_t>(position - m_ranges.begin());
		m_ranges.insert(position, replacement);
		MergeAroundUnlocked(index);
		RefreshCommittedUnlocked(start, size);
		return true;
	}

	void Rename(uint64_t start, uint64_t size, const char* name) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked();

		auto position = LowerBound(start);
		if (position != m_ranges.end() && position->start == start && position->size == size) {
			CopyVirtualRangeName(position->name, name);
			MergeAroundUnlocked(static_cast<size_t>(position - m_ranges.begin()));
			return;
		}
		EditUnlocked(start, size, [name](Range* r) { CopyVirtualRangeName(r->name, name); });
	}

	void Protect(uint64_t start, uint64_t size, int protection) {
		Common::LockGuard lock(m_mutex);
		BumpGenerationUnlocked();

		EditUnlocked(start, size, [protection](Range* r) { r->protection = protection; });
	}

	bool Query(uint64_t addr, int flags, Range* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);

		auto next = std::upper_bound(
		    m_ranges.begin(), m_ranges.end(), addr,
		    [](uint64_t value, const Range& range) { return value < range.start; });
		if (next != m_ranges.begin()) {
			auto current = std::prev(next);
			if (addr < End(current->start, current->size)) {
				*out = *current;
				return true;
			}
		}
		if (flags != 1 || next == m_ranges.end()) {
			return false;
		}

		*out = *next;
		return true;
	}

	// Ranges overlapping the span, clipped to it; holes are skipped. Returns the bytes covered
	// (upstream 2069acdf4).
	uint64_t QueryFragments(uint64_t start, uint64_t size, std::vector<Range>* out) {
		EXIT_IF(out == nullptr);

		Common::LockGuard lock(m_mutex);
		out->clear();
		if (size == 0) {
			return 0;
		}

		const auto end     = End(start, size);
		uint64_t   covered = 0;
		// From the range that can contain `start` (every earlier one ends at or before it): a
		// binary search instead of a walk from the first range (adapted from usa-ratt/KytyPS5-GT7
		// bab2ca8f7).
		auto first = std::upper_bound(
		    m_ranges.begin(), m_ranges.end(), start,
		    [](uint64_t value, const Range& range) { return value < range.start; });
		if (first != m_ranges.begin()) {
			--first;
		}
		for (auto it = first; it != m_ranges.end() && it->start < end; ++it) {
			const auto& range     = *it;
			const auto  range_end = End(range.start, range.size);
			if (range_end <= start) {
				continue;
			}

			Range part = range;
			part.start = std::max(start, range.start);
			part.size  = std::min(end, range_end) - part.start;
			if (part.type == VirtualRangeType::Direct) {
				part.offset += part.start - range.start;
			}
			covered += part.size;
			out->push_back(part);
		}
		return covered;
	}

	// Fragments of a span with no holes.
	bool QuerySpan(uint64_t start, uint64_t size, std::vector<Range>* out) {
		if (start == 0 || size == 0 || size > UINT64_MAX - start ||
		    QueryFragments(start, size, out) != size) {
			out->clear();
			return false;
		}
		return true;
	}

	// The committed prefix of [virtual_addr, virtual_addr + size): how far the run of adjacent
	// committed ranges containing virtual_addr covers it (0 when virtual_addr is not committed).
	// KYTY_CLAMP_RANGE_MEMO: a run this thread found while the ranges were as they are now answers
	// without the lock; every change of the ranges advances m_generation, under m_mutex, before it
	// starts, so an unchanged generation means the run still exists as found.
	uint64_t ClampRangeSize(uint64_t virtual_addr, uint64_t size) {
		if (virtual_addr == 0 || size == 0 || size > UINT64_MAX - virtual_addr) {
			return 0;
		}
		Common::CruzStats::Add(Common::CruzStats::Counter::ClampCalls);
		if (ClampRangeBitmapEnabled()) {
			const auto generation = ClampGenerationUnlocked(std::memory_order_acquire);
			if (const auto run = m_committed.CommittedRun(virtual_addr, size); run != CommittedPageMap::Unknown) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::ClampRangeBitmapHits);
				Common::CruzStats::Add(Common::CruzStats::Counter::ClampBitmapHits);
				t_clamp_totals.hits++;
				if (ClampRangeMemoVerifyMode() != 0) {
					VerifyClampHit(virtual_addr, size, run, generation);
				}
				return run;
			}
		}
		if (ClampRangeMemoEnabled()) {
			const auto generation = ClampGenerationUnlocked(std::memory_order_acquire);
			for (const auto& run: t_clamp_runs.entries) {
				if (run.owner != this || run.generation != generation || virtual_addr < run.begin ||
				    virtual_addr >= run.end) {
					continue;
				}
				uint64_t clamped = 0;
				if (size <= run.end - virtual_addr) {
					clamped = size;
				} else if (run.end_is_run_end) {
					clamped = run.end - virtual_addr;
				} else {
					break; // beyond the part of the run that was walked
				}
				t_clamp_totals.hits++;
				Common::CruzStats::Add(Common::CruzStats::Counter::ClampMemoHits);
				if (ClampRangeMemoVerifyMode() != 0) {
					VerifyClampHit(virtual_addr, size, clamped, generation);
				}
				return clamped;
			}
			t_clamp_totals.misses++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::ClampRangeMemoMisses);
		}
		Common::CruzStats::Add(Common::CruzStats::Counter::ClampLocked);
		Common::LockGuard lock(m_mutex);
		return ClampRangeSizeUnlocked(virtual_addr, size, ClampRangeMemoEnabled());
	}

	// The committed parts of [start, start + size), adjacent ranges merged, in order. False (and
	// none) when there are more than max_spans.
	bool CommittedSpans(uint64_t start, uint64_t size, size_t max_spans,
	                    std::vector<std::pair<uint64_t, uint64_t>>* spans) {
		spans->clear();
		if (start == 0 || size == 0 || size > UINT64_MAX - start) {
			return true;
		}
		const auto        end = start + size;
		Common::LockGuard lock(m_mutex);
		auto              next = std::upper_bound(
            m_ranges.begin(), m_ranges.end(), start,
            [](uint64_t value, const Range& candidate) { return value < candidate.start; });
		if (next != m_ranges.begin()) {
			--next;
		}
		for (; next != m_ranges.end() && next->start < end; ++next) {
			if (!IsCommittedRangeType(next->type) || next->size == 0) {
				continue;
			}
			const auto from = std::max(start, next->start);
			const auto to   = std::min(end, End(next->start, next->size));
			if (from >= to) {
				continue;
			}
			if (!spans->empty() && spans->back().first + spans->back().second == from) {
				spans->back().second += to - from;
				continue;
			}
			if (spans->size() >= max_spans) {
				spans->clear();
				return false;
			}
			spans->emplace_back(from, to - from);
		}
		return true;
	}

	// The generation every change advances before it starts (acquire): an unchanged value later
	// means every ClampRangeSize answer made after this read still holds.
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}
	// What ClampRangeSize answers depend on (KYTY_CLAMP_COVERAGE_GEN: only coverage changes).
	[[nodiscard]] uint64_t ClampGeneration() const noexcept {
		return ClampGenerationUnlocked(std::memory_order_acquire);
	}

	uint64_t CountPageTableEntries(bool gpu) {
		Common::LockGuard lock(m_mutex);

		uint64_t used = 0;
		for (const auto& r: m_ranges) {
			if (!IsCommittedRangeType(r.type) || r.size == 0) {
				continue;
			}

			const bool has_cpu_access =
			    (r.protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0;
			const bool has_gpu_access = (r.protection & (PROT_GPU_READ | PROT_GPU_WRITE)) != 0;
			if (gpu ? !has_gpu_access : !has_cpu_access) {
				continue;
			}

			const auto end         = End(r.start, r.size);
			const auto first_entry = r.start / PAGE_TABLE_GRANULARITY;
			const auto last_entry  = (end - 1) / PAGE_TABLE_GRANULARITY;
			used += last_entry - first_entry + 1;
		}

		return used;
	}

private:
	static uint64_t End(uint64_t start, uint64_t size) {
		return (UINT64_MAX - start < size ? UINT64_MAX : start + size);
	}

	// Callers hold m_mutex, before they change m_ranges (KYTY_CLAMP_RANGE_MEMO). `coverage`: the
	// change can change which bytes are committed (KYTY_CLAMP_COVERAGE_GEN).
	void BumpGenerationUnlocked(bool coverage = false) noexcept {
		m_generation.fetch_add(1, std::memory_order_acq_rel);
		if (coverage) {
			m_coverage_generation.fetch_add(1, std::memory_order_acq_rel);
		}
	}

	// The generation the clamp answers are valid under: every change's, or with
	// KYTY_CLAMP_COVERAGE_GEN only the coverage changes'.
	[[nodiscard]] uint64_t ClampGenerationUnlocked(std::memory_order order) const noexcept {
		return ClampCoverageGenerationEnabled() ? m_coverage_generation.load(order)
		                                        : m_generation.load(order);
	}

	// KYTY_CLAMP_RANGE_BITMAP: the committed-page bits of [start, start + size) as m_ranges has them
	// now; callers hold m_mutex. Committed pages are set before uncommitted ones are cleared.
	void RefreshCommittedUnlocked(uint64_t start, uint64_t size) {
		if (!ClampRangeBitmapEnabled() || start == 0 || size == 0 || size > UINT64_MAX - start) {
			return;
		}
		const auto end   = start + size;
		auto       first = std::upper_bound(m_ranges.begin(), m_ranges.end(), start,
		                                    [](uint64_t value, const Range& range) { return value < range.start; });
		if (first != m_ranges.begin()) {
			--first;
		}
		uint64_t                                   covered = start;
		std::vector<std::pair<uint64_t, uint64_t>> gaps;
		for (auto it = first; it != m_ranges.end() && it->start < end; ++it) {
			const auto from = std::max(start, it->start);
			const auto to   = std::min(end, End(it->start, it->size));
			if (from >= to) {
				continue;
			}
			if (IsCommittedRangeType(it->type)) {
				m_committed.Set(from, to - from, true);
				if (from > covered) {
					gaps.emplace_back(covered, from - covered);
				}
				covered = to;
			}
		}
		if (covered < end) {
			gaps.emplace_back(covered, end - covered);
		}
		for (const auto& [gap, bytes]: gaps) {
			m_committed.Set(gap, bytes, false);
		}
	}

	// ClampRangeSize's lookup; callers hold m_mutex. `remember`: record the run for this thread.
	uint64_t ClampRangeSizeUnlocked(uint64_t virtual_addr, uint64_t size, bool remember) {
		auto vma = std::upper_bound(
		    m_ranges.begin(), m_ranges.end(), virtual_addr,
		    [](uint64_t value, const Range& range) { return value < range.start; });
		if (vma == m_ranges.begin()) {
			return 0;
		}
		--vma;

		const auto vma_end = End(vma->start, vma->size);
		if (virtual_addr < vma->start || virtual_addr >= vma_end ||
		    !IsCommittedRangeType(vma->type)) {
			return 0;
		}
		// The run of adjacent committed ranges around the address, walking at most RunWalk ranges
		// each way: a bounded walk only narrows what later lookups can be answered from.
		constexpr uint32_t RunWalk   = 64;
		uint64_t           run_begin = vma->start;
		{
			auto     previous = vma;
			uint32_t walked   = 0;
			while (previous != m_ranges.begin() && walked++ < RunWalk) {
				--previous;
				if (!IsCommittedRangeType(previous->type) ||
				    End(previous->start, previous->size) != run_begin) {
					break;
				}
				run_begin = previous->start;
			}
		}
		uint64_t run_end        = vma_end;
		bool     end_is_run_end = true;
		auto     next           = std::next(vma);
		for (uint32_t walked = 0; next != m_ranges.end() && next->start == run_end &&
		                          IsCommittedRangeType(next->type);
		     ++next) {
			if (walked++ == RunWalk) {
				end_is_run_end = false;
				break;
			}
			run_end = End(next->start, next->size);
		}
		if (remember) {
			auto& runs = t_clamp_runs;
			runs.entries[runs.next++ % runs.entries.size()] = {
			    this, ClampGenerationUnlocked(std::memory_order_relaxed), run_begin, run_end,
			    end_is_run_end};
		}
		if (size <= run_end - virtual_addr) {
			return size;
		}
		if (end_is_run_end) {
			return run_end - virtual_addr;
		}
		// The walk stopped inside the run (`next` is the first range it did not take): follow the
		// rest as far as the request reaches, as the walk would have.
		uint64_t clamped_size = run_end - virtual_addr;
		uint64_t expected     = run_end;
		for (; next != m_ranges.end() && next->start == expected &&
		       IsCommittedRangeType(next->type) && clamped_size < size;
		     ++next) {
			const auto chunk = std::min(size - clamped_size, next->size);
			clamped_size += chunk;
			expected += chunk;
		}
		return clamped_size;
	}

	// KYTY_CLAMP_RANGE_MEMO_VERIFY: the locked lookup after a cached answer. A different answer
	// while the generation is still the one the hit saw is a mismatch; otherwise a change of the
	// ranges raced the lookup.
	void VerifyClampHit(uint64_t virtual_addr, uint64_t size, uint64_t cached, uint64_t generation) {
		Common::LockGuard lock(m_mutex);
		t_clamp_totals.verify_checks++;
		const auto locked = ClampRangeSizeUnlocked(virtual_addr, size, false);
		if (locked == cached) {
			return;
		}
		if (ClampGenerationUnlocked(std::memory_order_relaxed) != generation) {
			t_clamp_totals.verify_races++;
			return;
		}
		t_clamp_totals.verify_mismatches++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::ClampRangeMemoVerifyMismatches);
		std::fprintf(stderr,
		             "ClampRangeMemoVerify: addr=0x%016" PRIx64 " size=0x%" PRIx64
		             " cached 0x%" PRIx64 ", locked 0x%" PRIx64 "\n",
		             virtual_addr, size, cached, locked);
		if (ClampRangeMemoVerifyMode() == 2) {
			EXIT("ClampRangeMemoVerify: a cached committed-range answer differs\n");
		}
	}

	static bool SameMergeKey(const Range& left, const Range& right) {
		if (left.disallow_merge || right.disallow_merge || left.type == VirtualRangeType::Direct ||
		    right.type == VirtualRangeType::Direct) {
			return false;
		}

		return left.type == right.type && left.protection == right.protection &&
		       left.memory_type == right.memory_type &&
		       std::strncmp(left.name, right.name, KERNEL_MAXIMUM_NAME_LENGTH) == 0;
	}

	static void AddPiece(std::vector<Range>* ranges, const Range& source, uint64_t start,
	                     uint64_t end) {
		EXIT_IF(ranges == nullptr);

		if (end <= start) {
			return;
		}

		Range piece = source;
		piece.start = start;
		piece.size  = end - start;
		if (piece.type == VirtualRangeType::Direct) {
			piece.offset += start - source.start;
		}
		ranges->push_back(piece);
	}

	std::vector<Range>::iterator LowerBound(uint64_t start) {
		return std::lower_bound(
		    m_ranges.begin(), m_ranges.end(), start,
		    [](const Range& range, uint64_t value) { return range.start < value; });
	}

	void MergeAroundUnlocked(size_t index) {
		if (index >= m_ranges.size()) {
			return;
		}
		if (index != 0) {
			auto& previous = m_ranges[index - 1];
			auto& current  = m_ranges[index];
			if (End(previous.start, previous.size) == current.start &&
			    SameMergeKey(previous, current)) {
				previous.size += current.size;
				m_ranges.erase(m_ranges.begin() + static_cast<std::ptrdiff_t>(index));
				index--;
			}
		}
		while (index + 1 < m_ranges.size()) {
			auto& current = m_ranges[index];
			auto& next    = m_ranges[index + 1];
			if (End(current.start, current.size) != next.start || !SameMergeKey(current, next)) {
				break;
			}
			current.size += next.size;
			m_ranges.erase(m_ranges.begin() + static_cast<std::ptrdiff_t>(index + 1));
		}
	}

	// The ranges that can overlap [start, start + size): [*first, *last) in m_ranges (sorted by
	// start, never overlapping). Edits and removals rewrite only this span: rebuilding and
	// re-sorting the whole vector per call cost seconds in Ratchet & Clank's streaming, which maps
	// and unmaps thousands of direct ranges (they never merge) through KernelBatchMap.
	void OverlapSpanUnlocked(uint64_t start, uint64_t size, size_t* first, size_t* last) {
		auto begin = std::upper_bound(
		    m_ranges.begin(), m_ranges.end(), start,
		    [](uint64_t value, const Range& range) { return value < range.start; });
		if (begin != m_ranges.begin() && End(std::prev(begin)->start, std::prev(begin)->size) > start) {
			--begin;
		}
		const auto span_end = End(start, size);
		auto       end      = begin;
		while (end != m_ranges.end() && end->start < span_end) {
			++end;
		}
		*first = static_cast<size_t>(begin - m_ranges.begin());
		*last  = static_cast<size_t>(end - m_ranges.begin());
	}

	// Replaces m_ranges[first, last) with `pieces`, then merges the new span with itself and its
	// neighbours (what MergeUnlocked would do there; the rest is already merged).
	void RewriteSpanUnlocked(size_t first, size_t last, const std::vector<Range>& pieces,
	                         bool merge) {
		// The pieces overwrite the span in place and only the difference is inserted or erased:
		// one move of the vector's tail at most (none when the count is unchanged), where an erase
		// and an insert moved it twice (tens of thousands of ranges while GT7 streams; adapted
		// from usa-ratt/KytyPS5-GT7 bab2ca8f7).
		const size_t old_count = last - first;
		const size_t common    = std::min(old_count, pieces.size());
		std::copy(pieces.begin(), pieces.begin() + static_cast<std::ptrdiff_t>(common),
		          m_ranges.begin() + static_cast<std::ptrdiff_t>(first));
		if (pieces.size() > old_count) {
			m_ranges.insert(m_ranges.begin() + static_cast<std::ptrdiff_t>(first + common),
			                pieces.begin() + static_cast<std::ptrdiff_t>(common), pieces.end());
		} else if (old_count > pieces.size()) {
			m_ranges.erase(m_ranges.begin() + static_cast<std::ptrdiff_t>(first + common),
			               m_ranges.begin() + static_cast<std::ptrdiff_t>(last));
		}
		if (!merge || m_ranges.size() < 2) {
			return;
		}
		size_t index = (first != 0 ? first - 1 : 0);
		size_t hi    = first + pieces.size(); // pairs (index, index + 1) up to (hi - 1, hi)
		while (index < hi && index + 1 < m_ranges.size()) {
			auto& current = m_ranges[index];
			auto& next    = m_ranges[index + 1];
			if (End(current.start, current.size) == next.start && SameMergeKey(current, next)) {
				current.size += next.size;
				m_ranges.erase(m_ranges.begin() + static_cast<std::ptrdiff_t>(index + 1));
				hi--;
			} else {
				index++;
			}
		}
	}

	template <typename EditFunc>
	void EditUnlocked(uint64_t start, uint64_t size, EditFunc edit) {
		if (size == 0) {
			return;
		}

		size_t first = 0;
		size_t last  = 0;
		OverlapSpanUnlocked(start, size, &first, &last);

		std::vector<Range> out;
		auto               edit_end = End(start, size);

		for (size_t index = first; index < last; index++) {
			const auto& r     = m_ranges[index];
			auto        r_end = End(r.start, r.size);
			if (!VirtualRangesOverlap(start, size, r.start, r.size)) {
				out.push_back(r);
				continue;
			}

			auto mid_start = std::max(start, r.start);
			auto mid_end   = std::min(edit_end, r_end);

			AddPiece(&out, r, r.start, mid_start);

			Range mid = r;
			mid.start = mid_start;
			mid.size  = mid_end - mid_start;
			if (mid.type == VirtualRangeType::Direct) {
				mid.offset += mid_start - r.start;
			}
			edit(&mid);
			out.push_back(mid);

			AddPiece(&out, r, mid_end, r_end);
		}

		RewriteSpanUnlocked(first, last, out, true);
	}

	bool RemoveUnlocked(uint64_t start, uint64_t size) {
		if (size == 0) {
			return false;
		}

		size_t first = 0;
		size_t last  = 0;
		OverlapSpanUnlocked(start, size, &first, &last);

		std::vector<Range> out;
		bool               removed = false;
		auto               rem_end = End(start, size);

		for (size_t index = first; index < last; index++) {
			const auto& r     = m_ranges[index];
			auto        r_end = End(r.start, r.size);
			if (!VirtualRangesOverlap(start, size, r.start, r.size)) {
				out.push_back(r);
				continue;
			}

			removed = true;
			AddPiece(&out, r, r.start, std::max(start, r.start));
			AddPiece(&out, r, std::min(rem_end, r_end), r_end);
		}

		if (removed) {
			// A removal only opens gaps: nothing new becomes adjacent, so no merge.
			RewriteSpanUnlocked(first, last, out, false);
		}
		return removed;
	}

	Range* FindOverlap(uint64_t start, uint64_t size) {
		auto position = LowerBound(start);
		if (position != m_ranges.end() &&
		    VirtualRangesOverlap(start, size, position->start, position->size)) {
			return &*position;
		}
		if (position != m_ranges.begin()) {
			auto previous = std::prev(position);
			if (VirtualRangesOverlap(start, size, previous->start, previous->size)) {
				return &*previous;
			}
		}
		return nullptr;
	}

	std::vector<Range> m_ranges;
	Common::Mutex      m_mutex;
	// Advanced before every change of m_ranges (KYTY_CLAMP_RANGE_MEMO); starts at 1.
	std::atomic<uint64_t> m_generation {1};
	// Advanced before every change that can change committed coverage (KYTY_CLAMP_COVERAGE_GEN).
	std::atomic<uint64_t> m_coverage_generation {1};
	CommittedPageMap      m_committed; // KYTY_CLAMP_RANGE_BITMAP
};

// KYTY_DMEM_OVERFLOW_MB=<MiB> (default 0, off; the GT7 preset sets 16384; usa-ratt/KytyPS5-GT7
// 094ad2a1e, e1134351f, 8b7d34444): room past the guest's direct and flexible memory for direct
// allocations that do not fit in the direct memory. The game asks only within the size it is
// told, and some loads of the same GT7 race took it past that: it held one or two 2.3 GB blocks
// (and a 1.6 GB one) that other loads did without, its 5.3 GB race pool did not fit, and it
// crashed on the null pointer a failed allocation returned or its loading screen never ended.
// Such an allocation (a search up to the end of the direct memory that found no room) is given
// room here instead, logged; the size the game is told does not change. The guest backing grows
// by the room: reserved only where used with KYTY_GUEST_BACKING_LAZY_COMMIT=1, committed with the
// rest otherwise (the default backing commits it all at boot).
static uint64_t DirectMemoryOverflowSize() {
	static const uint64_t size = [] {
		uint64_t mb = 0;
		// Decimal MiB only (anything else keeps it off).
		if (const char* value = std::getenv("KYTY_DMEM_OVERFLOW_MB");
		    value != nullptr && value[0] != '\0' &&
		    std::strspn(value, "0123456789") == std::strlen(value) && std::strlen(value) <= 6) {
			mb = std::strtoull(value, nullptr, 10);
		}
		// Whole 2 MiB pages (rounded up), at most 64 GiB.
		return ((std::min<uint64_t>(mb, 65536u) + 1u) & ~uint64_t {1}) << 20u;
	}();
	return size;
}

class PhysicalMemory {
public:
	struct AllocatedBlock {
		uint64_t            start_addr;
		uint64_t            size;
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		int                 memory_type;
		bool                pool_expansion;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	PhysicalMemory() {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(0, Size());
		// KYTY_DMEM_OVERFLOW_MB: the room, past the flexible memory. Searches of the direct memory
		// stop at Size(), so only Alloc's overflow search reaches it.
		if (DirectMemoryOverflowSize() != 0) {
			m_free.emplace(TotalSize(), DirectMemoryOverflowSize());
		}
	}
	virtual ~PhysicalMemory() = default;

	KYTY_CLASS_NO_COPY(PhysicalMemory);

	static uint64_t TotalSize() { return g_guest_memory_size; }
	static uint64_t Size() {
		EXIT_IF(g_flexible_memory_size >= TotalSize());
		return TotalSize() - g_flexible_memory_size;
	}

	bool Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
	           uint64_t* phys_addr_out, int memory_type, bool pool_expansion = false);
	bool Available(uint64_t search_start, uint64_t search_end, size_t alignment,
	               uint64_t* phys_addr_out, uint64_t* size_out);
	bool Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
	             GpuAccessMode* gpu_mode);
	bool Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot, VirtualMemory::Mode mode,
	         GpuAccessMode gpu_mode, const char* name, int memory_type);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	bool Find(uint64_t phys_addr, bool next, PhysicalMemory::AllocatedBlock* out);
	bool CanMapDirect(uint64_t phys_addr, size_t len);
	bool ReleasePoolExpansion(uint64_t phys_addr, size_t len);
	bool GetAllocatedSpan(uint64_t phys_addr, size_t len, std::vector<AllocatedBlock>* blocks);
	std::vector<AllocatedBlock> FindMappings(uint64_t phys_addr, size_t len);
	void ProtectMapping(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	                    GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);

	[[nodiscard]] Common::Mutex&                            GetMutex() { return m_mutex; }
	[[nodiscard]] const std::map<uint64_t, AllocatedBlock>& GetPhysicalBlocks() const {
		return m_physical;
	}
	[[nodiscard]] const std::vector<AllocatedBlock>& GetMappings() const { return m_mappings; }

private:
	void ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
	                      uint64_t size);
	void AddFreeRange(uint64_t start, uint64_t size);

	std::map<uint64_t, AllocatedBlock> m_physical;
	std::map<uint64_t, uint64_t>       m_free;
	std::vector<AllocatedBlock>        m_mappings;
	Common::Mutex                      m_mutex;
	// Direct allocations given room past the direct memory (DirectMemoryOverflowSize).
	uint64_t                           m_overflow_allocations = 0;
};

class FlexibleMemory {
public:
	struct AllocatedBlock {
		uint64_t            map_vaddr;
		uint64_t            map_size;
		uint64_t            backing_offset;
		uint64_t            host_vaddr;
		uint64_t            host_size;
		int                 prot;
		VirtualMemory::Mode mode;
		GpuAccessMode       gpu_mode;
		char                name[KERNEL_MAXIMUM_NAME_LENGTH];
	};

	FlexibleMemory() {
		EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
		m_free.emplace(PhysicalMemory::Size(), Size());
	}
	virtual ~FlexibleMemory() = default;

	KYTY_CLASS_NO_COPY(FlexibleMemory);

	static uint64_t Size() { return g_flexible_memory_size; }
	uint64_t        Available();
	void            Charge(uint64_t size);
	void            Release(uint64_t size);

	bool Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode, GpuAccessMode gpu_mode,
	         const char* name);
	bool Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
	           uint64_t* host_vaddr_to_release = nullptr);
	bool Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
	          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode);
	void Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
	             GpuAccessMode gpu_mode);
	void SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name);

	[[nodiscard]] Common::Mutex&                     GetMutex() { return m_mutex; }
	[[nodiscard]] const std::vector<AllocatedBlock>& GetBlocks() const { return m_allocated; }

private:
	void ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
	                      uint64_t size);
	void AddFreeRange(uint64_t start, uint64_t size);

	std::vector<AllocatedBlock>  m_allocated;
	std::map<uint64_t, uint64_t> m_free;
	uint64_t                     m_allocated_total = 0;
	Common::Mutex                m_mutex;
};

class PooledMemory {
public:
	struct Mapping {
		uint64_t      vaddr;
		uint64_t      size;
		uint64_t      phys_addr;
		GpuAccessMode gpu_mode;
	};

	void                   Expand(uint64_t phys_addr, uint64_t size);
	bool                   ReleaseExpansion(uint64_t phys_addr, uint64_t size);
	bool                   Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
	                                std::vector<Mapping>* mappings);
	bool                   Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings);
	bool                   Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode);
	[[nodiscard]] uint64_t Available();
	[[nodiscard]] std::vector<Mapping> GetMappings();

private:
	struct PhysicalRange {
		uint64_t start;
		uint64_t size;
	};

	void AddFreeUnlocked(uint64_t start, uint64_t size);
	bool QueryUnlocked(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) const;

	std::vector<PhysicalRange> m_free;
	std::vector<PhysicalRange> m_expansions;
	std::vector<Mapping>       m_mappings;
	Common::Mutex              m_mutex;
};

static std::unique_ptr<PhysicalMemory>    g_physical_memory;
static std::unique_ptr<FlexibleMemory>    g_flexible_memory;
static std::unique_ptr<PooledMemory>      g_pooled_memory;
static std::unique_ptr<VirtualRanges>     g_virtual_ranges;
static std::unique_ptr<GuestAddressSpace> g_guest_address_space;
static callback_func_t                    g_alloc_callback        = nullptr;
static callback_func_t                    g_free_callback         = nullptr;
static std::atomic<uint64_t>              g_memory_pool_committed = 0;
static void                               MemoryPoolSubtractCommitted(uint64_t len);

// Memory locking (upstream fc1369e65's operation lock + shared state lock, adapted). Lock order,
// outermost first:
//   1. g_memory_operation_mutex (recursive): every operation that changes guest memory (map,
//      unmap, protect, reserve, pool, name, PRT aperture) holds it throughout, as before.
//   2. g_memory_state_mutex (shared): a thread's outermost hold of (1) also holds it exclusively
//      (MemoryOperationLock). The guest's query calls (KernelVirtualQuery,
//      KernelQueryMemoryProtection, KernelIsStack, KernelDirectMemoryQuery, the available-size and
//      page-table and pool-stat queries) take it shared instead of (1) (MemoryStateReadLock), so
//      they no longer serialize with each other and see an operation's changes only all or none.
//      A query made inside an operation on the same thread (an allocation callback, a batch) reads
//      under the thread's own exclusive hold.
//   3. The objects' own mutexes: VirtualRanges::m_mutex, the PhysicalMemory, FlexibleMemory and
//      PooledMemory mutexes, g_prt_aperture_mutex (never two of them nested, except
//      FlexibleMemory's around the address-space calls of its Map/Unmap).
//   4. GuestAddressSpace::m_mutex, then GuestBackingStore::m_mutex.
// The lock-free and locked readers on the GPU, worker and fault threads take neither (1) nor (2):
// ClampRangeSize's per-thread runs and committed-page bitmap (KYTY_CLAMP_RANGE_MEMO/_BITMAP,
// valid while VirtualRanges::m_generation is unchanged), the backing store's per-thread mapping
// records (TryReadBackingDirect, KYTY_MAP_REGION_EPOCH: the map generation and region epochs) and
// the tracker's ProtectTransient (KYTY_PROTECT_LOCK_SHARED). Their generations still move at the
// innermost lock, (3) or (4), before each change, with (1) and (2) already held, so a cached answer
// is valid exactly as before, and their locked fallbacks wait only for (3) or (4), never for a whole
// operation. This replaces upstream's lock-free page table (1b59ee62b) and page-walk clamp
// (3bd22044d), which duplicate those paths.
// Unlike upstream, (2) is held for the whole operation, graphics calls and callbacks included:
// no graphics path takes (1) or (2), and our operations are not split into a validation part and
// one state change (upstream's restructure), so releasing (2) between their steps would let a query
// see half an operation.
static std::recursive_mutex  g_memory_operation_mutex;
static std::shared_mutex     g_memory_state_mutex;
static thread_local uint32_t t_memory_operation_depth = 0;

// KYTY_GTHREAD_STATS (common/gthreadStats.h): acquisitions, contended acquisitions and wait time
// of the two locks below, guest threads apart from the others ("memlock": MemoryOperationLock, the
// operation mutex and the state lock together; "memread": MemoryStateReadLock). Off: one relaxed
// load, then the plain locks.
template <typename TryLock, typename Lock>
static void LockCounted(TryLock&& try_lock, Lock&& lock, uint64_t& wait_ns) {
	if (try_lock()) {
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	lock();
	wait_ns += 1u + static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                          std::chrono::steady_clock::now() - start)
	                                          .count());
}

static void NoteLockStats(std::atomic<uint64_t>* acq, std::atomic<uint64_t>* contended,
                          std::atomic<uint64_t>* wait, uint64_t wait_ns) {
	const int kind = GThreadStats::IsGuestThread() ? 0 : 1;
	GThreadStats::Add(acq[kind]);
	if (wait_ns != 0) {
		GThreadStats::Add(contended[kind]);
		GThreadStats::Add(wait[kind], wait_ns);
	}
}

class MemoryOperationLock {
public:
	MemoryOperationLock() {
		if (GThreadStats::Enabled()) {
			LockCountedStats();
			return;
		}
		g_memory_operation_mutex.lock();
		if (t_memory_operation_depth++ == 0) {
			g_memory_state_mutex.lock();
		}
	}
	~MemoryOperationLock() {
		if (--t_memory_operation_depth == 0) {
			g_memory_state_mutex.unlock();
		}
		g_memory_operation_mutex.unlock();
	}
	KYTY_CLASS_NO_COPY(MemoryOperationLock);

private:
	static void LockCountedStats() {
		uint64_t wait_ns = 0;
		LockCounted([] { return g_memory_operation_mutex.try_lock(); },
		            [] { g_memory_operation_mutex.lock(); }, wait_ns);
		if (t_memory_operation_depth++ == 0) {
			LockCounted([] { return g_memory_state_mutex.try_lock(); },
			            [] { g_memory_state_mutex.lock(); }, wait_ns);
		}
		auto& counters = GThreadStats::Local();
		NoteLockStats(counters.mem_acq, counters.mem_contended, counters.mem_wait_ns, wait_ns);
	}
};

class MemoryStateReadLock {
public:
	MemoryStateReadLock(): m_locked(t_memory_operation_depth == 0) {
		if (!m_locked) {
			return;
		}
		if (GThreadStats::Enabled()) {
			uint64_t wait_ns = 0;
			LockCounted([] { return g_memory_state_mutex.try_lock_shared(); },
			            [] { g_memory_state_mutex.lock_shared(); }, wait_ns);
			auto& counters = GThreadStats::Local();
			NoteLockStats(counters.mem_read_acq, counters.mem_read_contended, counters.mem_read_wait_ns,
			              wait_ns);
			return;
		}
		g_memory_state_mutex.lock_shared();
	}
	~MemoryStateReadLock() {
		if (m_locked) {
			g_memory_state_mutex.unlock_shared();
		}
	}
	KYTY_CLASS_NO_COPY(MemoryStateReadLock);

private:
	bool m_locked;
};

// The base address the PS5 kernel hands out for hint-less user mappings. Guest code can
// assume mappings it did not place explicitly are at or above this (Sony's libc rejects a
// heap below it), so hint-less searches must not fall back to the low system-managed range.
static constexpr uint64_t GUEST_DEFAULT_MAP_BASE = 0x200000000ull;

static uint64_t FindGuestFreeRange(uint64_t search_addr, uint64_t size, uint64_t alignment) {
	auto find_in = [&](uint64_t begin, uint64_t end) {
		auto current = begin;
		while (current < end && size <= end - current) {
			const auto candidate =
			    g_guest_address_space->FindFreeAligned(current, end, size, alignment);
			if (candidate == 0) {
				return uint64_t {0};
			}
			VirtualRanges::Range overlap {};
			if (!g_virtual_ranges->QueryOverlap(candidate, size, &overlap)) {
				return candidate;
			}
			current = overlap.start + overlap.size;
		}
		return uint64_t {0};
	};

	if (search_addr != 0) {
		// A hint inside the AMM window is not honoured: that window belongs to AMM map commands.
		if (search_addr < HOST_USER_ALLOC_END) {
			return find_in(search_addr, HOST_USER_ALLOC_END);
		}
		search_addr = 0;
	}
	auto addr = find_in(GUEST_DEFAULT_MAP_BASE, HOST_SYSTEM_MANAGED_MAX + 1u);
	if (addr == 0) {
		addr = find_in(HOST_USER_MIN, HOST_USER_ALLOC_END);
	}
	return addr;
}

bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
	// KYTY_STREAM_COPY_DEFER: a pending recorder copy of these bytes runs first.
	Graphics::StreamCopyDefer::NoteGuestWrite(vaddr, size);
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryWriteBacking(vaddr, data, size);
}

bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryReadBacking(vaddr, data, size);
}

bool TryReadBackingDirect(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->TryReadBackingDirect(vaddr, data, size);
}

const void* GuestBackingAlias(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space != nullptr ? g_guest_address_space->BackingAlias(vaddr, size)
	                                        : nullptr;
}

uint64_t GuestBackingBase() {
	return g_guest_address_space != nullptr ? g_guest_address_space->GetBackingBase() : 0;
}

uint64_t GuestBackingSize() {
	return g_guest_address_space != nullptr ? g_guest_address_space->GetBackingSize() : 0;
}

static std::atomic<bool (*)(uint64_t)> g_pending_amm_map_covers {nullptr};

void SetPendingAmmMapHook(bool (*covers)(uint64_t vaddr)) {
	g_pending_amm_map_covers.store(covers, std::memory_order_release);
}

bool PendingAmmMapCovers(uint64_t vaddr) {
	auto* covers = g_pending_amm_map_covers.load(std::memory_order_acquire);
	return covers != nullptr && covers(vaddr);
}

bool IsGpuFillableWord(uint64_t vaddr) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, sizeof(uint32_t)) ||
	    !Graphics::GuestGpu::IsGpuThread()) {
		return false;
	}
	return GetGpuResources().GetBufferCache().GpuFillSource(vaddr);
}

bool TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		if (!Graphics::GuestGpu::IsGpuThread()) {
			return false;
		}
		auto& buffers = GetGpuResources().GetBufferCache();
		// KYTY_DIRECT_READ_WAIT=1 (default 0; on in the PPSA03671 title preset): bytes a queued
		// download has yet to publish are not current in the backing either (BufferCache::
		// WaitBackingPublications; adapted from KytyPS5-Frank 1c9ad7586).
		static const bool wait_publications = [] {
			const char* value = std::getenv("KYTY_DIRECT_READ_WAIT");
			return value != nullptr && value[0] == '1';
		}();
		if (buffers.HasGpuDirtyBytes(vaddr, size)) {
			buffers.ReadMemory(vaddr, size);
		} else if (wait_publications && buffers.HasPendingBackingPublication(vaddr, size)) {
			buffers.WaitBackingPublications(vaddr, size);
		}
	}
	return TryReadBacking(vaddr, data, size);
}

// The exact GPU-ownership predicates of a clean backing read. GPU thread only.
static bool IsGpuRangeCleanForBackingRead(uint64_t vaddr, uint64_t size) {
	auto& resources = GetGpuResources();
	return !resources.GetBufferCache().HasGpuDirtyBytes(vaddr, size) &&
	       !resources.GetBufferCache().HasPendingBackingPublication(vaddr, size) &&
	       !resources.GetTextureCache().IsRegionGpuModified(vaddr, size);
}

// KYTY_CLEAN_UNTOUCHED (drawPrep.cpp): no page of the range was ever touched by a GPU-side
// transition (the sticky pages of gpuTouchedPages.h, maintained with KYTY_CP_SEQ=1). Such a range
// is clean for a backing read: every source that can make the exact predicates below refuse it
// marks its pages before its change becomes observable.
static bool NeverGpuTouched(uint64_t vaddr, uint64_t size) {
	return Graphics::GpuTouched::Enabled() && size != 0 && vaddr <= UINT64_MAX - size &&
	       !Graphics::GpuTouched::AnyTouched(Graphics::DrawPrep::UntouchedFine(), vaddr, vaddr + size);
}

static std::atomic<uint64_t> g_untouched_verdict_checks {0};
static std::atomic<uint64_t> g_untouched_verdict_mismatches {0};

// verify / exit: an untouched range the exact predicates call unclean.
static void VerifyUntouchedVerdict(uint64_t vaddr, uint64_t size, bool exact) {
	const auto checks = g_untouched_verdict_checks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!exact) {
		const auto count = g_untouched_verdict_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count <= 16) {
			std::printf("CleanUntouched: untouched range [0x%" PRIx64 ", +0x%" PRIx64
			            ") is not clean\n",
			            vaddr, size);
		}
		if (Graphics::DrawPrep::UntouchedMode() == 3) {
			EXIT("CleanUntouched: an untouched range is not clean\n");
		}
	}
	if ((checks & 0xfffffu) == 0) {
		std::printf("CleanUntouched verdict verify: %" PRIu64 " untouched queries, %" PRIu64
		            " mismatches\n",
		            checks, g_untouched_verdict_mismatches.load(std::memory_order_relaxed));
	}
}

// The clean verdict of a GPU address range (verdict cache when enabled). GPU thread or an
// active GpuReadDelegate scope.
static bool QueryGpuCleanVerdict(uint64_t vaddr, uint64_t size) {
	namespace CleanVerdict = Graphics::CleanVerdict;
	if (const auto untouched = Graphics::DrawPrep::UntouchedMode();
	    untouched != 0 && NeverGpuTouched(vaddr, size)) {
		if (untouched == 1) {
			return true;
		}
		const bool exact = IsGpuRangeCleanForBackingRead(vaddr, size);
		VerifyUntouchedVerdict(vaddr, size, exact);
		return exact;
	}
	if (!CleanVerdict::Enabled()) {
		return IsGpuRangeCleanForBackingRead(vaddr, size);
	}
	// Only the ownership verdict is cached; bytes are always read fresh by the callers.
	static thread_local CleanVerdict::Table verdicts;
	const auto result = CleanVerdict::Query(verdicts, vaddr, size, IsGpuRangeCleanForBackingRead);
	Profiler::CountFrameEvent(result.hit ? Profiler::FrameEvent::CleanVerdictHits
	                                     : Profiler::FrameEvent::CleanVerdictMisses);
	if (result.stores != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CleanVerdictStores, result.stores);
	}
	return result.clean;
}

// The GPU-ownership gate of an exact clean backing read.
static bool GpuCleanGate(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size)) {
		// A draw-preparation helper may probe only inside a GPU-thread fork window
		// (gpuReadDelegate.h), during which the GPU-thread-owned dirty state is stable.
		if (!Graphics::GuestGpu::IsGpuThread() && !Graphics::GpuReadDelegate::Active()) {
			return false;
		}
		if (!QueryGpuCleanVerdict(vaddr, size)) {
			return false;
		}
	}
	return true;
}

// KYTY_DRAW_PREP_PRIVATE_READS=1 (default off): a clean read (draw-prep worker reads, their commit-
// time certificate checks, the command processor's clean reads) of bytes in private mappings, which
// have no backing (the program image: GT7 keeps its shaders and their AGC metadata in it), copies
// them in place instead of failing. Only ranges the GPU has not mapped (GpuResources::IsMapped): the GPU
// never writes them, so they are never protected against reads, and the serial path reads them in
// place anyway.
// Without it every GT7 stage preparation fails ("backing", ~1/3 of draws at the title, ~45% in
// races) and runs serially on the command processor.
static bool PrivateReadsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_PRIVATE_READS");
		return value != nullptr && (value[0] == '1' || value[0] == '2');
	}();
	return enabled;
}

// Whether every page of the range is readable by the host now (not protected against reads, as the
// page tracker protects GPU-written pages): the in-place copy then cannot fault.
static bool HostReadable(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto end = vaddr + size;
	for (auto at = vaddr; at < end;) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) != sizeof(info) ||
		    info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0 ||
		    (info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
		                     PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY)) == 0) {
			return false;
		}
		at = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
	}
	return true;
#else
	(void)vaddr;
	(void)size;
	return false;
#endif
}

static bool TryReadPrivate(uint64_t vaddr, void* data, uint64_t size) {
	if (!PrivateReadsEnabled() || g_guest_address_space == nullptr) {
		return false;
	}
	// GPU-mapped private memory (GT7's program image holds its shader code): none of it may be
	// GPU-written or waiting for a publication (the lock-free mirror on workers; the command
	// processor's callers passed the exact gate before).
	if (g_gpu_resources != nullptr && g_gpu_resources->IsMapped(vaddr, size)) {
		auto& buffers = GetGpuResources().GetBufferCache();
		if (buffers.IsRegionGpuModifiedRelaxed(vaddr, size) || buffers.HasPendingBackingPublication(vaddr, size)) {
			return false;
		}
	}
	// KYTY_DRAW_PREP_PRIVATE_READS=2 also checks each page's host protection first (VirtualQuery: slow under
	// the address-space lock the fault handlers' protection changes take).
	static const bool check = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_PRIVATE_READS");
		return value != nullptr && value[0] == '2';
	}();
	return (!check || HostReadable(vaddr, size)) && g_guest_address_space->TryReadPrivateInPlace(vaddr, data, size);
}

static bool TryReadGpuCleanBackingExact(uint64_t vaddr, void* data, uint64_t size) {
	return GpuCleanGate(vaddr, size) && (TryReadBacking(vaddr, data, size) || TryReadPrivate(vaddr, data, size));
}

bool BackingInPlaceEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_BACKING_INPLACE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// inspect(bytes) reads the backing bytes of the range in place (GuestBackingStore::
// TryInspectBacking; it may run twice).
template <typename Inspect>
static bool TryInspectBacking(uint64_t vaddr, uint64_t size, Inspect& inspect,
                              InPlaceStats* stats) {
	bool       locked = false;
	const bool done   = g_guest_address_space != nullptr &&
	                  g_guest_address_space->TryInspectBacking(vaddr, size, inspect, &locked);
	if (stats != nullptr) {
		stats->inspected++;
		stats->locked += locked ? 1u : 0u;
	}
	if (!done && PrivateReadsEnabled() && size <= 64u * 1024u * 1024u) {
		// KYTY_DRAW_PREP_PRIVATE_READS: bytes of private mappings, inspected from a copy.
		static thread_local std::vector<uint8_t> copy;
		copy.resize(static_cast<size_t>(size));
		if (TryReadPrivate(vaddr, copy.data(), size)) {
			inspect(static_cast<const uint8_t*>(copy.data()));
			return true;
		}
	}
	return done;
}

// KYTY_DRAW_PREP_LOCKFREE_HINT (default on): the GPU-dirty hint of preparing workers reads the
// tracker's lock-free mirror of the GPU-dirty bits instead of taking the region locks the GPU
// thread takes for every binding. The hint only decides whether a worker computes on the bytes at
// all: it was a racy snapshot already (the GPU thread may mark a range right after it), and the
// commit decides with the exact predicate either way. KYTY_DRAW_PREP_LOCKFREE_HINT_VERIFY=1|exit
// also takes the locks and checks the mirror against the bits (equal under the locks).
static bool LockFreeHintEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_LOCKFREE_HINT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static int LockFreeHintVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_LOCKFREE_HINT_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

// KYTY_DRAW_PREP_GPU_DIRTY_HINT=1 restores the page-level refusal (default off): a non-GPU preparing
// thread refused a read because its tracker PAGE is GPU-dirty. The hint is only an efficiency
// filter (see above): the commit re-validates every recorded range with the exact byte-range
// predicate, so a preparation that read bytes the GPU had really written still falls back.
// Demon's Souls reads 16-byte records from a few pages that compute shaders write elsewhere and the
// hint refused ~98% of those reads (2/3 of all draws fell back, 104 -> 84 ms per frame without it);
// Astro Bot's Sky Garden start view went from 27.6 to 30.2 fps with a steadier 2-vblank pacing.
static bool GpuDirtyHintEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_GPU_DIRTY_HINT");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static bool GpuDirtyHint(Graphics::BufferCache& buffers, uint64_t vaddr, uint64_t size) {
	if (!GpuDirtyHintEnabled()) {
		return false;
	}
	if (!LockFreeHintEnabled()) {
		return buffers.IsRegionGpuModified(vaddr, size);
	}
	const bool dirty = buffers.IsRegionGpuModifiedRelaxed(vaddr, size);
	if (LockFreeHintVerifyMode() != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepHintVerifyChecks);
		if (!buffers.GpuDirtyMirrorMatches(vaddr, size)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepHintVerifyMismatches);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				LOGF("DrawPrepHintVerify: GPU-dirty mirror differs from the tracker bits: "
				     "addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
				     vaddr, size);
			}
			if (LockFreeHintVerifyMode() == 2) {
				EXIT("DrawPrepHintVerify: GPU-dirty mirror differs from the tracker bits\n");
			}
		}
	}
	return dirty;
}

// The host return address of the calling function (diagnostics).
#if defined(_MSC_VER) && !defined(__clang__)
#define KYTY_MEMORY_CALLER() reinterpret_cast<uint64_t>(_ReturnAddress())
#else
#define KYTY_MEMORY_CALLER() reinterpret_cast<uint64_t>(__builtin_return_address(0))
#endif

// DrawPrepFallbackUnclean diagnostics: why a preparation's read was refused as not provably
// clean. Always counted (FrameEvent.DrawPrepUnclean*); with the hang trace also aggregated in
// unclean.csv by reason, read purpose, calling host code and page, with the page's last GPU writer.
static void NoteDrawPrepUnclean(Profiler::FrameEvent event, const char* reason, uint64_t vaddr,
                                uint64_t size, uint64_t caller) {
	Profiler::CountFrameEvent(event);
	if (HangTrace::Enabled()) {
		HangTrace::RecordUncleanRead(vaddr, size, reason, Graphics::DrawPrep::ReadPurpose(), caller);
	}
}

// Which exact predicate refused a clean read (GPU thread or a GpuReadDelegate scope; a thread
// with neither is refused before any predicate).
static const char* ExactUncleanReason(uint64_t vaddr, uint64_t size) {
	if (!Graphics::GuestGpu::IsGpuThread() && !Graphics::GpuReadDelegate::Active()) {
		return "thread";
	}
	auto& resources = GetGpuResources();
	if (resources.GetBufferCache().HasGpuDirtyBytes(vaddr, size)) {
		return "buffer-gpu-dirty";
	}
	if (resources.GetBufferCache().HasPendingBackingPublication(vaddr, size)) {
		return "publication";
	}
	if (resources.GetTextureCache().IsRegionGpuModified(vaddr, size)) {
		return "image-gpu-modified";
	}
	return "verdict";
}

// Draw-prep (readSet.h): every read of a speculative draw preparation is recorded for its
// certificate. The GPU thread applies the exact clean predicate. Other preparing threads cannot
// read the GPU thread's unlocked dirty state; they gate reads with a conservative, thread-safe
// hint (tracker GPU-dirty pages, which cover every buffer GPU-dirty byte, and pending backing
// publications) so that they rarely compute on stale bytes, and the commit re-validates every
// range with the exact predicate. The hint does not consult the texture cache: a range owned by
// a GPU-modified image is caught at commit.
// The gate of a draw-prep read: the failure it takes, None when the backing may be read.
// `caller`: the host code that asked for the read (diagnostics).
static Graphics::DrawPrep::ReadFailure DrawPrepGate(const Graphics::DrawPrep::Recorder& recorder,
                                                    uint64_t vaddr, uint64_t size,
                                                    uint64_t caller) {
	using Graphics::DrawPrep::ReadFailure;
	using Event = Profiler::FrameEvent;
	if (recorder.exact) {
		if (GpuCleanGate(vaddr, size)) {
			return ReadFailure::None;
		}
		NoteDrawPrepUnclean(Event::DrawPrepUncleanExact,
		                    HangTrace::Enabled() ? ExactUncleanReason(vaddr, size) : "exact", vaddr,
		                    size, caller);
		return ReadFailure::Unclean;
	}
	if (g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) &&
	    !(Graphics::DrawPrep::UntouchedMode() != 0 && NeverGpuTouched(vaddr, size))) {
		auto& buffers = GetGpuResources().GetBufferCache();
		if (GpuDirtyHint(buffers, vaddr, size)) {
			NoteDrawPrepUnclean(Event::DrawPrepUncleanHintGpuDirty, "hint-gpu-dirty", vaddr, size,
			                    caller);
			return ReadFailure::Unclean;
		}
		if (buffers.HasPendingBackingPublication(vaddr, size)) {
			NoteDrawPrepUnclean(Event::DrawPrepUncleanHintPublication, "hint-publication", vaddr,
			                    size, caller);
			return ReadFailure::Unclean;
		}
	}
	return ReadFailure::None;
}

// KYTY_DRAW_PREP_BACKING_LOG=1 (default 0, diagnostics): the worker reads that failed because the
// backing store holds no translation for the bytes, aggregated by owning guest range, read purpose
// and calling host code, printed every 10 s by the thread that crosses the period.
namespace {
struct BackingFailEntry {
	uint64_t    count      = 0;
	uint64_t    min_size   = UINT64_MAX;
	uint64_t    max_size   = 0;
	uint64_t    sample     = 0;
	uint64_t    range_start = 0;
	uint64_t    range_size = 0;
	uint64_t    dmem       = 0;
	int         prot       = 0;
	const char* type       = "none";
	const char* purpose    = "";
	uint64_t    caller     = 0;
	bool        backing_head = false; // the first byte has a backing translation
	char        name[40] {};
};
std::mutex                                     g_backing_fail_mutex;
std::unordered_map<uint64_t, BackingFailEntry> g_backing_fails;
uint64_t                                       g_backing_fail_period_ns = 0;
uint64_t                                       g_backing_fail_total     = 0;
} // namespace

static bool BackingFailLogEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_BACKING_LOG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static void NoteBackingFailure(uint64_t vaddr, uint64_t size, uint64_t caller) {
	VirtualRanges::Range range {};
	const bool known = g_virtual_ranges != nullptr && g_virtual_ranges->Query(vaddr, 0, &range);
	const char* purpose = Graphics::DrawPrep::ReadPurpose();
	const uint64_t key = (known ? range.start : (vaddr >> 20u)) ^ (caller * 0x9e3779b97f4a7c15ull) ^
	                     reinterpret_cast<uint64_t>(purpose);
	const bool backing_head =
	    g_guest_address_space != nullptr && g_guest_address_space->BackingContains(vaddr, 1);
	const auto now = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count());
	std::vector<BackingFailEntry> report;
	uint64_t                      total = 0;
	{
		std::lock_guard lock(g_backing_fail_mutex);
		auto& entry = g_backing_fails[key];
		if (entry.count == 0) {
			entry.sample       = vaddr;
			entry.caller       = caller;
			entry.purpose      = purpose != nullptr ? purpose : "";
			entry.backing_head = backing_head;
			if (known) {
				entry.range_start = range.start;
				entry.range_size  = range.size;
				entry.dmem        = range.offset;
				entry.prot        = range.protection;
				entry.type        = magic_enum::enum_name(range.type).data();
				std::snprintf(entry.name, sizeof(entry.name), "%s", range.name);
			}
		}
		entry.count++;
		entry.min_size = std::min(entry.min_size, size);
		entry.max_size = std::max(entry.max_size, size);
		g_backing_fail_total++;
		if (g_backing_fail_period_ns == 0) {
			g_backing_fail_period_ns = now;
		} else if (now - g_backing_fail_period_ns >= 10'000'000'000ull) {
			for (auto& [k, value]: g_backing_fails) {
				report.push_back(value);
			}
			total = g_backing_fail_total;
			g_backing_fails.clear();
			g_backing_fail_total     = 0;
			g_backing_fail_period_ns = now;
		}
	}
	if (report.empty()) {
		return;
	}
	std::sort(report.begin(), report.end(),
	          [](const auto& a, const auto& b) { return a.count > b.count; });
	static const uint64_t module_base = [] {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
#else
		return uint64_t {0};
#endif
	}();
	std::printf("DrawPrepBacking 10s: %" PRIu64 " worker reads without a backing translation, %zu groups\n",
	            total, report.size());
	for (size_t i = 0; i < report.size() && i < 24; i++) {
		const auto& e = report[i];
		std::printf("  DrawPrepBacking: %" PRIu64 " reads size %" PRIu64 "-%" PRIu64
		            " sample 0x%016" PRIx64 " range 0x%016" PRIx64 "+0x%" PRIx64
		            " type=%s prot=0x%x dmem=0x%" PRIx64 " name='%s' head-backed=%s purpose=%s "
		            "caller=exe+0x%" PRIx64 "\n",
		            e.count, e.min_size, e.max_size, e.sample, e.range_start, e.range_size, e.type,
		            static_cast<unsigned>(e.prot), e.dmem, e.name, e.backing_head ? "yes" : "no",
		            e.purpose, e.caller - module_base);
	}
	std::fflush(stdout);
}

// KYTY_DRAW_PREP_BACKING_DIAG=1 (diagnostic, default off): draw-prep worker reads that found no
// backing (DrawPrep fallback "backing"), by 1 MiB region, with the size, the calling host code
// (module offset) and whether the first byte alone has a backing; one console line every 10 s.
static void NoteDrawPrepBackingDiag(uint64_t vaddr, uint64_t size, uint64_t caller) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_PREP_BACKING_DIAG");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled) {
		return;
	}
	struct Site {
		uint64_t count = 0, size = 0, caller = 0, vaddr = 0;
		bool     first_byte = false;
		bool     gpu = false, priv = false;
	};
	static std::mutex                               mutex;
	static std::unordered_map<uint64_t, Site>       sites;
	static uint64_t                                 last_ns = 0;
	uint8_t    byte  = 0;
	const bool first = TryReadBacking(vaddr, &byte, 1);
	const auto now   = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	std::scoped_lock lock(mutex);
	auto&            site = sites[(vaddr >> 20u) ^ (static_cast<uint64_t>(first) << 63u)];
	site.count++;
	site.size       = size;
	site.vaddr      = vaddr;
	site.first_byte = first;
	site.gpu        = g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size);
	uint8_t probe   = 0;
	site.priv       = g_guest_address_space != nullptr && g_guest_address_space->TryReadPrivateInPlace(vaddr, &probe, 1);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	site.caller = caller - reinterpret_cast<uint64_t>(GetModuleHandleA(nullptr));
#else
	site.caller = caller;
#endif
	if (last_ns == 0) {
		last_ns = now;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	last_ns = now;
	std::vector<std::pair<uint64_t, Site>> top(sites.begin(), sites.end());
	std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second.count > b.second.count; });
	std::string text;
	for (size_t i = 0; i < top.size() && i < 8; i++) {
		const auto& s = top[i].second;
		char line[160];
		std::snprintf(line, sizeof(line), " | %" PRIu64 " x 0x%" PRIx64 "+%" PRIu64 " caller +0x%" PRIx64 "%s%s%s",
		              s.count, s.vaddr, s.size, s.caller, s.first_byte ? " (first byte backed)" : "",
		              s.gpu ? " gpu-range" : "", s.priv ? " private" : " not-private");
		text += line;
	}
	std::printf("DrawPrep backing misses 10s:%s\n", text.c_str());
	sites.clear();
}

// A backing read that failed after the gate (exact reads fail as Unclean, as before).
static Graphics::DrawPrep::ReadFailure DrawPrepBackingFailure(
    const Graphics::DrawPrep::Recorder& recorder, uint64_t vaddr, uint64_t size, uint64_t caller) {
	using Graphics::DrawPrep::ReadFailure;
	if (!recorder.exact) {
		if (BackingFailLogEnabled()) [[unlikely]] {
			NoteBackingFailure(vaddr, size, caller);
		}
		NoteDrawPrepBackingDiag(vaddr, size, caller);
		return ReadFailure::Backing;
	}
	NoteDrawPrepUnclean(Profiler::FrameEvent::DrawPrepUncleanBacking, "backing", vaddr, size,
	                    caller);
	return ReadFailure::Unclean;
}

// The gated read of TryReadForDrawPrep, without recording it. False (and a failed read set) when
// it cannot be served.
static bool ReadForDrawPrep(Graphics::DrawPrep::Recorder& recorder, uint64_t vaddr, void* data,
                            uint64_t size, uint64_t caller) {
	using Graphics::DrawPrep::ReadFailure;
	auto& reads = *recorder.reads;
	if (reads.Failed()) {
		return false;
	}
	if (const auto failure = DrawPrepGate(recorder, vaddr, size, caller);
	    failure != ReadFailure::None) {
		reads.Fail(failure);
		return false;
	}
	if (!TryReadBacking(vaddr, data, size) && !TryReadPrivate(vaddr, data, size)) {
		reads.Fail(DrawPrepBackingFailure(recorder, vaddr, size, caller));
		return false;
	}
	return true;
}

// KYTY_DRAW_PREP_DIRECT_READS (live, default 1; workers only, the exact predicate path is
// unchanged): a draw-prep worker's read whose whole 4 KiB page passed the workers' read hint
// earlier in the same preparation copies straight from the page's host backing bytes, found once
// per page through the thread's backing mapping records, and checks the backing map generation
// afterwards (a map change in between redoes the read the usual way). The hint is only a hint:
// the commit's certificate (logged transitions, then the exact clean predicate or the bytes)
// decides, as for every worker read, so checking it once per page instead of once per read
// changes no outcome. The read set also extends a read that continues the previous one
// (ReadSet::Record). verify: each direct copy is compared with a TryReadBacking of the same bytes;
// differences that persist on a second read are counted (exit: stop at the first).
static std::atomic<uint64_t> g_direct_read_checks {0};
static std::atomic<uint64_t> g_direct_read_mismatches {0};

// The workers' read hint (DrawPrepGate without the exact path and the diagnostics) for a whole
// page.
static bool DrawPrepHintClean(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return true;
	}
	if (Graphics::DrawPrep::UntouchedMode() != 0 && NeverGpuTouched(vaddr, size)) {
		return true;
	}
	auto& buffers = GetGpuResources().GetBufferCache();
	return !GpuDirtyHint(buffers, vaddr, size) && !buffers.HasPendingBackingPublication(vaddr, size);
}

static void VerifyDirectRead(uint64_t vaddr, const void* data, uint64_t size) {
	const auto checks = g_direct_read_checks.fetch_add(1, std::memory_order_relaxed) + 1;
	std::array<uint8_t, 64> reference {};
	if (size > reference.size()) {
		return;
	}
	const auto differs = [&] {
		return TryReadBacking(vaddr, reference.data(), size) &&
		       std::memcmp(reference.data(), data, static_cast<size_t>(size)) != 0;
	};
	if (differs() && differs()) {
		const auto count = g_direct_read_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count <= 16) {
			std::printf("DrawPrepDirectReads: a direct read of [0x%" PRIx64 ", +%" PRIu64
			            ") differs from the backing read\n",
			            vaddr, size);
		}
		if (Graphics::DrawPrep::DirectReadsMode() == 3) {
			EXIT("DrawPrepDirectReads: a direct read differs from the backing read\n");
		}
	}
	if ((checks & 0xffffffu) == 0) {
		std::printf("DrawPrepDirectReads verify: %" PRIu64 " checks, %" PRIu64 " mismatches\n",
		            checks, g_direct_read_mismatches.load(std::memory_order_relaxed));
	}
}

// KYTY_DRAW_PREP_NULL_READS=1 (default off; Ghost of Yotei preset): a preparation's read that lies
// wholly in the first 64 KiB, which no guest or host mapping ever contains
// (ShaderRecompiler::IR::NeverMappedAddress), reads zeros, exactly as the serial path's readers do
// for such reads (ReadsUnmappedGuestMemory, SrtWalker::InPlaceReadable / ReadRawWord: a null
// table or descriptor pointer on a path the shader does not take). The bytes can never change, so
// the read needs no certificate entry. Without the switch such a read fails the preparation
// ("backing"), and the whole draw is prepared again serially on the command processor: ~1,600
// draws per frame at Ghost of Yotei's torii, through null-based V# reads at 0xa0 and 0. Live.
static Live::Switch g_draw_prep_null_reads("KYTY_DRAW_PREP_NULL_READS", Live::ParseDefaultOff);

static bool DrawPrepNullReadsEnabled() {
	return g_draw_prep_null_reads.On();
}

static bool TryReadForDrawPrep(Graphics::DrawPrep::Recorder& recorder, uint64_t vaddr, void* data,
                               uint64_t size, uint64_t caller) {
	constexpr uint64_t PageMask      = 4095u;
	constexpr uint64_t NullRegionEnd = 0x10000;
	if (vaddr < NullRegionEnd && size <= NullRegionEnd - vaddr && DrawPrepNullReadsEnabled()) {
		if (recorder.reads->Failed()) {
			return false;
		}
		std::memset(data, 0, static_cast<size_t>(size));
		return true;
	}
	const bool direct = !recorder.exact && g_guest_address_space != nullptr && Graphics::DrawPrep::DirectReadsMode() != 0;
	if (direct && (vaddr >> 12u) == recorder.direct_page && size <= 64u &&
	    (vaddr & PageMask) + size <= 4096u && !recorder.reads->Failed()) {
		std::memcpy(data, recorder.direct_host + (vaddr & PageMask), static_cast<size_t>(size));
		std::atomic_thread_fence(std::memory_order_acquire);
		if (GuestAddressSpace::BackingMappingTag(vaddr) == recorder.direct_generation) {
			if (Graphics::DrawPrep::DirectReadsMode() >= 2) [[unlikely]] {
				VerifyDirectRead(vaddr, data, size);
			}
			return recorder.reads->Record(vaddr, data, size);
		}
		recorder.direct_page = UINT64_MAX; // the backing map changed: find the page again
	}
	if (!ReadForDrawPrep(recorder, vaddr, data, size, caller)) {
		return false;
	}
	if (direct && (vaddr & PageMask) + size <= 4096u && (vaddr >> 12u) != recorder.direct_page) {
		const auto page       = vaddr & ~PageMask;
		uint64_t   generation = 0;
		const auto* host      = g_guest_address_space->BackingAliasCached(page, 4096u, generation);
		if (host != nullptr && DrawPrepHintClean(page, 4096u)) {
			recorder.direct_page       = vaddr >> 12u;
			recorder.direct_host       = host;
			recorder.direct_generation = generation;
		}
	}
	return recorder.reads->Record(vaddr, data, size);
}

bool TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size) {
	if (auto* recorder = Graphics::DrawPrep::ActiveRecorder(); recorder != nullptr) [[unlikely]] {
		// KYTY_DRAW_PREP_PHASES: the recorded reads' share of the preparation.
		const auto start = Common::PrepPhases::SubStart();
		const bool ok    = TryReadForDrawPrep(*recorder, vaddr, data, size, KYTY_MEMORY_CALLER());
		Common::PrepPhases::SubAdd(Common::PrepPhases::Sub::Reads, start);
		Common::PrepPhases::NoteRead(size);
		return ok;
	}
	return TryReadGpuCleanBackingExact(vaddr, data, size);
}

bool TryReadGpuCleanBackingDigest(uint64_t vaddr, void* data, uint64_t size, uint64_t& digest) {
	auto* recorder = Graphics::DrawPrep::ActiveRecorder();
	if (recorder == nullptr) {
		if (!TryReadGpuCleanBackingExact(vaddr, data, size)) {
			return false;
		}
		digest = XXH3_64bits(data, size);
		return true;
	}
	if (!ReadForDrawPrep(*recorder, vaddr, data, size, KYTY_MEMORY_CALLER())) {
		return false;
	}
	digest = XXH3_64bits(data, size);
	return recorder->reads->RecordDigest(vaddr, size, digest);
}

BackingCompare CompareGpuCleanBacking(uint64_t vaddr, const void* expected, uint64_t size,
                                      InPlaceStats* stats) {
	// A preparation must record what it reads (TryReadGpuCleanBacking).
	EXIT_IF(Graphics::DrawPrep::ActiveRecorder() != nullptr);
	if (!GpuCleanGate(vaddr, size)) {
		return BackingCompare::Unavailable;
	}
	bool equal   = false;
	auto compare = [&](const uint8_t* bytes) {
		equal = std::memcmp(bytes, expected, static_cast<size_t>(size)) == 0;
	};
	if (!TryInspectBacking(vaddr, size, compare, stats)) {
		return BackingCompare::Unavailable;
	}
	return equal ? BackingCompare::Equal : BackingCompare::Different;
}

bool HashGpuCleanBacking(uint64_t vaddr, uint64_t size, uint64_t& digest, InPlaceStats* stats) {
	auto hash = [&](const uint8_t* bytes) { digest = XXH3_64bits(bytes, static_cast<size_t>(size)); };
	auto* recorder = Graphics::DrawPrep::ActiveRecorder();
	if (recorder == nullptr) {
		return GpuCleanGate(vaddr, size) && TryInspectBacking(vaddr, size, hash, stats);
	}
	using Graphics::DrawPrep::ReadFailure;
	auto& reads = *recorder->reads;
	if (reads.Failed()) {
		return false;
	}
	const auto caller = KYTY_MEMORY_CALLER();
	if (const auto failure = DrawPrepGate(*recorder, vaddr, size, caller);
	    failure != ReadFailure::None) {
		reads.Fail(failure);
		return false;
	}
	if (!TryInspectBacking(vaddr, size, hash, stats)) {
		reads.Fail(DrawPrepBackingFailure(*recorder, vaddr, size, caller));
		return false;
	}
	return reads.RecordDigest(vaddr, size, digest);
}

bool CodeDigestCacheable(uint64_t vaddr, uint64_t size, uint64_t& tag_begin, uint64_t& tag_end) {
	constexpr uint32_t RegionShift = 26; // GuestAddressSpace's 64 MiB mapping-tag regions
	if (g_guest_address_space == nullptr || !Graphics::GpuTouched::Enabled() || size == 0 ||
	    vaddr > UINT64_MAX - size) {
		return false;
	}
	const auto last = vaddr + size - 1u;
	if ((last >> RegionShift) - (vaddr >> RegionShift) > 1u) {
		return false;
	}
	// The 4 KiB marks: coarser than the 1 KiB ones, so never untouched where those are not.
	if (Graphics::GpuTouched::AnyTouched(false, vaddr, vaddr + size)) {
		return false;
	}
	tag_begin = GuestAddressSpace::BackingMappingTag(vaddr);
	tag_end   = GuestAddressSpace::BackingMappingTag(last);
	return true;
}

bool HashBackingUnrecorded(uint64_t vaddr, uint64_t size, uint64_t& digest) {
	auto hash = [&](const uint8_t* bytes) { digest = XXH3_64bits(bytes, static_cast<size_t>(size)); };
	return TryInspectBacking(vaddr, size, hash, nullptr);
}

bool IsGpuMapped(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && g_gpu_resources->IsMapped(vaddr, size);
}

bool IsCommittedSpan(uint64_t vaddr, uint64_t size) {
	if (g_virtual_ranges == nullptr || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}
	const auto end     = vaddr + size;
	auto       current = vaddr;
	while (current < end) {
		VirtualRanges::Range range {};
		if (!g_virtual_ranges->Query(current, 0, &range) || !IsCommittedRangeType(range.type) ||
		    range.start + range.size <= current) {
			return false;
		}
		current = range.start + range.size;
	}
	return true;
}

std::string DescribeAddressForCrash(uint64_t vaddr) {
	std::string out;
	char        line[256];
	if (g_virtual_ranges != nullptr) {
		VirtualRanges::Range range {};
		if (g_virtual_ranges->Query(vaddr, 0, &range)) {
			std::snprintf(line, sizeof(line),
			              "guest range: 0x%016" PRIx64 "+0x%" PRIx64 " type=%s prot=0x%x dmem=0x%" PRIx64 "\n",
			              range.start, range.size, magic_enum::enum_name(range.type).data(),
			              static_cast<unsigned>(range.protection), range.offset);
		} else {
			std::snprintf(line, sizeof(line), "guest range: none (not reserved or mapped)\n");
		}
		out += line;
	}
	std::snprintf(line, sizeof(line), "gpu mapped: %s, backing: %s, queued AMM map covers it: %s\n",
	              IsGpuMapped(vaddr, 1) ? "yes" : "no",
	              g_guest_address_space != nullptr && g_guest_address_space->BackingContains(vaddr, 1) ? "yes" : "no",
	              PendingAmmMapCovers(vaddr) ? "yes" : "no");
	out += line;
	return out;
}

bool DescribeVirtualRange(uint64_t vaddr, uint64_t* start, uint64_t* size, char* name, uint64_t name_size) {
	VirtualRanges::Range range {};
	if (g_virtual_ranges == nullptr || !g_virtual_ranges->Query(vaddr, 0, &range)) {
		return false;
	}
	*start = range.start;
	*size  = range.size;
	if (name != nullptr && name_size != 0) {
		std::snprintf(name, static_cast<size_t>(name_size), "%.*s", static_cast<int>(sizeof(range.name)), range.name);
	}
	return true;
}

static bool IsInPrtAperture(uint64_t address, uint64_t size);

// The committed, non-private, GPU-mapped allocation containing vaddr (not sparse/PRT).
bool GetGpuBufferMapping(uint64_t vaddr, uint64_t* start, uint64_t* size) {
	VirtualRanges::Range range {};
	if (g_virtual_ranges == nullptr) return false;
	const auto generation = g_virtual_ranges->Generation();
	if (!g_virtual_ranges->Query(vaddr, 0, &range) || !IsCommittedRangeType(range.type) ||
	    IsPrivateCommittedRangeType(range.type) || range.start > vaddr ||
	    range.size <= vaddr - range.start || IsInPrtAperture(vaddr, 1)) {
		return false;
	}
	if (!IsGpuMapped(range.start, range.size) || generation != g_virtual_ranges->Generation()) {
		return false;
	}
	*start = range.start;
	*size  = range.size;
	return true;
}

// DrawPrep::Validate's clean check after a clean coherence-log check over [since, now]
// (KYTY_DRAW_PREP_CLEAN_SINCE). GPU thread only. The verdict of a whole 4 KiB page that was found
// clean when the coherence generation was g (loaded before the exact predicates ran) still holds
// now if g >= since: every transition that can make the page's bytes unclean logs the range it
// affects before the change is observable, and those transitions run on this thread (GPU-dirty
// marks, pending publications, GPU-modified images), so one that touched the requested range after
// g lies in [since, now] and failed the caller's log check. The table keeps verdicts across commits
// (the per-generation verdict cache drops them at every transition anywhere). Any other page is
// checked exactly, as IsGpuCleanForRead does.
bool IsGpuCleanForReadSince(uint64_t vaddr, uint64_t size, uint64_t since) {
	namespace CleanVerdict = Graphics::CleanVerdict;
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return true;
	}
	if (!Graphics::GuestGpu::IsGpuThread() || size == 0) {
		return IsGpuCleanForRead(vaddr, size);
	}
	const auto first = vaddr >> CleanVerdict::PAGE_BITS;
	const auto last  = (vaddr + size - 1) >> CleanVerdict::PAGE_BITS;
	if (first == 0 || last - first >= CleanVerdict::MAX_CACHED_PAGES) {
		return QueryGpuCleanVerdict(vaddr, size);
	}
	if (Graphics::DrawPrep::UntouchedMode() == 1 && NeverGpuTouched(vaddr, size)) {
		return true;
	}
	struct Entry {
		uint64_t page       = 0;
		uint64_t generation = 0;
	};
	constexpr size_t                         Entries = 4096;
	static thread_local std::array<Entry, Entries> verdicts {};
	for (auto page = first; page <= last; page++) {
		auto& entry = verdicts[static_cast<size_t>((page ^ (page >> 12u)) % Entries)];
		if (entry.page == page && entry.generation >= since && entry.generation != 0) {
			continue;
		}
		const auto generation = CleanVerdict::Generation();
		if (IsGpuRangeCleanForBackingRead(page << CleanVerdict::PAGE_BITS, CleanVerdict::PAGE_SIZE)) {
			entry = {page, generation};
			continue;
		}
		// Not the whole page: the requested bytes decide, exactly.
		return QueryGpuCleanVerdict(vaddr, size);
	}
	return true;
}

bool IsGpuCleanForRead(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return true;
	}
	if (!Graphics::GuestGpu::IsGpuThread() && !Graphics::GpuReadDelegate::Active()) {
		return false;
	}
	return QueryGpuCleanVerdict(vaddr, size);
}

bool SrtStaleImageReadsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SRT_STALE_IMAGE_READS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool TryReadGpuBackingOverStaleImages(uint64_t vaddr, void* data, uint64_t size) {
	if (!SrtStaleImageReadsEnabled() || g_gpu_resources == nullptr || size == 0 ||
	    !IsGpuAddressRange(vaddr, size) || !Graphics::GuestGpu::IsGpuThread() ||
	    Graphics::DrawPrep::ActiveRecorder() != nullptr) {
		return false;
	}
	auto& resources = GetGpuResources();
	auto& buffers   = resources.GetBufferCache();
	auto& textures  = resources.GetTextureCache();
	if (buffers.HasGpuDirtyBytes(vaddr, size) || buffers.HasPendingBackingPublication(vaddr, size) ||
	    !textures.IsRegionGpuModified(vaddr, size) || textures.IsRegionGpuOwnedForRead(vaddr, size)) {
		return false;
	}
	if (!TryReadBacking(vaddr, data, size)) {
		return false;
	}
	static std::atomic<uint64_t> reads {0};
	const auto n = reads.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((n & (n - 1)) == 0) {
		std::printf("SrtStaleImageReads: %" PRIu64 " resource-table reads under superseded GPU-modified "
		            "images read the guest bytes (last 0x%" PRIx64 "+0x%" PRIx64 ")\n",
		            n, vaddr, size);
	}
	return true;
}

bool SynchronizeGpuBackingForRead(uint64_t vaddr, uint64_t size) {
	return g_gpu_resources != nullptr && IsGpuAddressRange(vaddr, size) &&
	       GetGpuResources().SynchronizeGpuBackingForRead(vaddr, size);
}

std::string DescribeGpuBackingForRead(uint64_t vaddr, uint64_t size) {
	if (g_gpu_resources == nullptr || !IsGpuAddressRange(vaddr, size)) {
		return "no GPU range";
	}
	return GetGpuResources().DescribeGpuBackingForRead(vaddr, size);
}

uint64_t ClampRangeSizeQuiet(uint64_t vaddr, uint64_t size) {
	return g_virtual_ranges != nullptr ? g_virtual_ranges->ClampRangeSize(vaddr, size) : 0;
}

bool CommittedSpans(uint64_t vaddr, uint64_t size, size_t max_spans,
                    std::vector<std::pair<uint64_t, uint64_t>>* spans) {
	EXIT_IF(spans == nullptr);
	if (g_virtual_ranges == nullptr) {
		spans->assign(1, {vaddr, size});
		return true;
	}
	return g_virtual_ranges->CommittedSpans(vaddr, size, max_spans, spans);
}

uint64_t CommittedExtent(uint64_t vaddr, uint64_t size) {
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	if (!CommittedSpans(vaddr, size, SIZE_MAX, &spans) || spans.empty()) {
		return 0;
	}
	return spans.back().first + spans.back().second - vaddr;
}

uint64_t VirtualRangesGeneration() noexcept {
	return g_virtual_ranges != nullptr ? g_virtual_ranges->ClampGeneration() : 0;
}

uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size) {
	EXIT_IF(g_virtual_ranges == nullptr);

	const auto clamped_size = g_virtual_ranges->ClampRangeSize(vaddr, size);
	if (clamped_size == 0) {
		EXIT("Memory: attempted to access invalid address 0x%016" PRIx64 " with size 0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	if (clamped_size != size) {
		LOGF("Memory: clamped buffer range addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     " to 0x%016" PRIx64 "\n",
		     vaddr, size, clamped_size);
	}
	return clamped_size;
}

void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept {
	if (TryWriteBacking(vaddr, data, size)) {
		return;
	}
	if (!Common::SkipInvalidEnabled()) {
		EXIT("Memory: required direct-backing write failed, addr=0x%016" PRIx64
		     " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	// KYTY_SKIP_INVALID (chenxiao07/KytyPS5 c680f0545): GPU results (readbacks, downloads) for guest
	// memory unmapped meanwhile have nowhere to go. The pages still backed are written, the rest
	// (unmapped, or private memory without a backing view) is dropped.
	constexpr uint64_t GuestPage = 0x4000;
	const auto*        source    = static_cast<const uint8_t*>(data);
	uint64_t           written   = 0;
	uint64_t           first_dropped = UINT64_MAX;
	for (uint64_t at = vaddr, end = vaddr + size; at < end;) {
		const auto next = std::min((at & ~(GuestPage - 1)) + GuestPage, end);
		if (TryWriteBacking(at, source + (at - vaddr), next - at)) {
			written += next - at;
		} else if (first_dropped == UINT64_MAX) {
			first_dropped = at;
		}
		at = next;
	}
	static std::atomic_bool logged {false};
	if (Common::LogOnce(logged)) {
		LOGF("Memory: GPU data for guest memory without backing dropped, addr=0x%016" PRIx64
		     " size=0x%" PRIx64 " (0x%" PRIx64 " bytes written, first dropped page 0x%016" PRIx64
		     "; logged once)\n",
		     vaddr, size, written, first_dropped);
	}
}

void InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (size == 0) {
		return;
	}
	(void)GetGpuResources().InvalidateMemory(vaddr, size);
}

bool PrepareHostOverwrite(uint64_t vaddr, uint64_t size) {
	if (size == 0 || g_gpu_resources == nullptr) {
		return false;
	}
	return g_gpu_resources->PrepareHostOverwrite(vaddr, size);
}

void InstallGpuResources(Graphics::RenderContext* resources) noexcept {
	EXIT_IF(resources != nullptr && g_gpu_resources != nullptr);
	g_gpu_resources = resources;
}

bool IsStoreInPlaceCandidate(uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->IsStoreInPlaceCandidate(fault_vaddr);
}

bool TryStoreInPlace(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                     uint64_t size) noexcept {
	return g_gpu_resources != nullptr &&
	       g_gpu_resources->TryStoreInPlace(pc, fault_vaddr, vaddr, data, size);
}

bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	return g_gpu_resources != nullptr && g_gpu_resources->HandleFault(access, fault_vaddr);
}

uint64_t WaitForTransientUnmap(uint64_t fault_vaddr) noexcept {
	return TransientUnmapRetryEnabled() ? TransientUnmapWindows::Wait(fault_vaddr) : 0;
}

bool RetryTransientUnmapFault(uint64_t fault_vaddr, uint64_t window) noexcept {
	// A page that keeps faulting after it was mapped again is a real access violation.
	constexpr uint32_t    MaxRetries  = 16;
	thread_local uint64_t last_page   = 0;
	thread_local uint64_t last_window = 0;
	thread_local uint32_t retries     = 0;
	const auto            page        = fault_vaddr & ~uint64_t {0x3fff};
	retries     = (page == last_page && window == last_window ? retries + 1 : 1);
	last_page   = page;
	last_window = window;
	if (retries > MaxRetries) {
		return false;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("Memory: retried a guest access to 0x%016" PRIx64
		            " made while its mapping was being replaced\n",
		            fault_vaddr);
		std::fflush(stdout);
	}
	return true;
}

void SetTransientUnmapRetryForTest(bool enabled) noexcept {
	g_transient_unmap_retry.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

struct PrtAperture {
	uint64_t address = 0;
	uint64_t size    = 0;
};

constexpr int      PRT_APERTURE_MAX_INDEX = 2;
constexpr uint64_t PRT_PAGE_SIZE          = 0x4000;
constexpr uint64_t PRT_APERTURE_START     = 0x0f00000000ull;
constexpr uint64_t PRT_APERTURE_END       = 0xfc00000000ull;

static std::array<PrtAperture, PRT_APERTURE_MAX_INDEX + 1> g_prt_apertures {};
static Common::Mutex                                       g_prt_aperture_mutex;

static bool IsInPrtAperture(uint64_t address, uint64_t size = 1) {
	if (size == 0 || UINT64_MAX - address < size) {
		return false;
	}
	Common::LockGuard lock(g_prt_aperture_mutex);

	for (const auto& aperture: g_prt_apertures) {
		if (address >= aperture.address) {
			const auto offset = address - aperture.address;
			if (offset < aperture.size && size <= aperture.size - offset) {
				return true;
			}
		}
	}

	return false;
}

bool TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	if (g_guest_address_space == nullptr || g_virtual_ranges == nullptr ||
	    !g_virtual_ranges->QuerySpan(vaddr, size, &ranges)) {
		return false;
	}
	if (std::any_of(ranges.begin(), ranges.end(), [](const auto& range) {
		    return !IsReservedRangeType(range.type) &&
		           !g_guest_address_space->BackingContains(range.start, range.size);
	    })) {
		return false;
	}
	return g_guest_address_space->TryReadSparseBacking(vaddr, data, size);
}

// Upstream 40692dfb6: TryReadSparseBacking plus the private mappings in the backing gaps, read
// in place (reservations and unmapped gaps read as zeros).
bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
	if (g_guest_address_space == nullptr) {
		return false;
	}
	return g_guest_address_space->CopySparseMemory<true>(vaddr, data, size);
}

static bool SelfTestSub64SharedPlaceholderAlias() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr uint64_t PageSize    = 0x4000;
	const auto         granularity = g_guest_address_space->GetGranularity();
	if (granularity < PageSize * 2u) {
		LOGF_COLOR(
		    Log::Color::Yellow,
		    "\t direct-memory sub-64K placeholder self-test skipped: granularity too small\n");
		return true;
	}

	const auto base = FindGuestFreeRange(0, granularity, granularity);
	if (base == 0) {
		LOGF_COLOR(Log::Color::Red,
		           "\t direct-memory sub-64K placeholder self-test: reserve unavailable\n");
		return false;
	}

	const auto alias          = base + PageSize;
	bool       ok             = false;
	auto       failure_reason = GuestBackingStore::FailureReason::None;
	if (g_guest_address_space->MapBacking(alias, PageSize, PageSize, VirtualMemory::Mode::ReadWrite,
	                                      &failure_reason)) {
		auto* ptr = reinterpret_cast<uint64_t*>(alias);
		*ptr      = 0x4b59545953553634ull; // "KYTYSU64"
		ok        = (*ptr == 0x4b59545953553634ull);
		std::memset(ptr, 0, PageSize);

		ok = g_guest_address_space->UnmapBacking(alias, PageSize) && ok;
	}

	LOGF_COLOR(
	    ok ? Log::Color::Green : Log::Color::Red,
	    "\t direct-memory sub-64K placeholder self-test: %s%s%s\n", ok ? "ok" : "failed",
	    ok ? "" : ", reason = ", ok ? "" : GuestBackingStore::GetFailureReasonName(failure_reason));
	return ok;
#else
	return true;
#endif
}

static int ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size);

// The memory sizes from the game's param.json (Loader::SystemContentGetMemoryParams); it replaces
// the emulator's SetFlexibleMemorySize call, which upstream removed with its param.json reader.
static void InitializeMemorySizes() {
	const auto params        = Loader::SystemContentGetMemoryParams();
	const auto trinity_extra = Loader::SystemContentIsTrinityMode() ? TRINITY_EXTRA_MEMORY_SIZE : 0;
	if (UpstreamMemoryDefaults()) {
		g_flexible_memory_size = params.flexible_size != 0 ? params.flexible_size : UPSTREAM_DEFAULT_FLEXIBLE_SIZE;
		const auto cpu_tables  = std::max(params.cpu_table_memory, UPSTREAM_INCLUDED_TABLE_MEMORY);
		const auto gpu_tables  = std::max(params.gpu_table_memory, UPSTREAM_INCLUDED_TABLE_MEMORY);
		const auto amm_tables  = std::max(params.amm_table_memory, UPSTREAM_INCLUDED_TABLE_MEMORY);
		const auto overhead    = cpu_tables + gpu_tables + amm_tables - 3 * UPSTREAM_INCLUDED_TABLE_MEMORY;
		const auto budget      = UPSTREAM_GUEST_MEMORY_BUDGET + trinity_extra;
		EXIT_IF(g_flexible_memory_size + overhead >= budget - UPSTREAM_RESERVED_CACHE_SIZE);
		g_guest_memory_size      = budget - UPSTREAM_RESERVED_CACHE_SIZE - overhead;
		g_cpu_page_table_entries = cpu_tables / UPSTREAM_MAPPING_TABLE_SIZE;
		g_gpu_page_table_entries = gpu_tables / UPSTREAM_MAPPING_TABLE_SIZE;
	} else {
		g_guest_memory_size = DEFAULT_GUEST_MEMORY_SIZE + trinity_extra;
		if (params.flexible_size != 0) {
			constexpr uint64_t GuestPageSize = 0x4000;
			EXIT_IF((params.flexible_size & (GuestPageSize - 1u)) != 0 ||
			        params.flexible_size >= g_guest_memory_size);
			g_flexible_memory_size = params.flexible_size;
		}
	}
	LOGF("\t flexible memory size = %" PRIu64 " MiB, direct memory size = %" PRIu64 " MiB%s\n",
	     g_flexible_memory_size / (1024ull * 1024ull),
	     (g_guest_memory_size - g_flexible_memory_size) / (1024ull * 1024ull),
	     UpstreamMemoryDefaults() ? " (KYTY_UPSTREAM_MEMORY_DEFAULTS)" : "");
}

void Initialize() {
	if (!g_flexible_memory_size_frozen) {
		InitializeMemorySizes();
	}
	g_flexible_memory_size_frozen = true;
	g_guest_address_space = std::make_unique<GuestAddressSpace>(PhysicalMemory::TotalSize() +
	                                                            DirectMemoryOverflowSize());
	if (DirectMemoryOverflowSize() != 0) {
		LOGF("Direct memory: %" PRIu64 " MiB of room past the direct memory for allocations that "
		     "do not fit in it (KYTY_DMEM_OVERFLOW_MB)\n",
		     DirectMemoryOverflowSize() >> 20u);
	}
	g_physical_memory     = std::make_unique<PhysicalMemory>();
	g_flexible_memory     = std::make_unique<FlexibleMemory>();
	const auto backing = g_guest_address_space->GetBackingBase();
	Common::RamStats::Range("guest direct physical capacity", reinterpret_cast<void*>(backing),
	                       PhysicalMemory::Size());
	Common::RamStats::Range("guest flexible physical capacity",
	                       reinterpret_cast<void*>(backing + PhysicalMemory::Size()),
	                       FlexibleMemory::Size());
	g_pooled_memory       = std::make_unique<PooledMemory>();
	g_virtual_ranges      = std::make_unique<VirtualRanges>();
	EXIT_IF(!g_guest_address_space->SelfTest());
	EXIT_IF(!SelfTestSub64SharedPlaceholderAlias());
}

void Shutdown() {
	g_pooled_memory.reset();
	g_flexible_memory.reset();
	g_physical_memory.reset();
	g_virtual_ranges.reset();
	g_guest_address_space.reset();
}

struct AlignedPos {
	uint64_t value = 0;
	bool     valid = false;
};

static constexpr AlignedPos GetAlignedPos(uint64_t pos, size_t alignment) {
	if (alignment == 0) {
		return {pos, true};
	}

	const auto remainder = pos % alignment;
	const auto increment = (remainder != 0 ? alignment - remainder : 0);
	if (increment > UINT64_MAX - pos) {
		return {};
	}

	return {pos + increment, true};
}

static_assert(!GetAlignedPos(UINT64_MAX - 1, 4).valid);

void RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func) {
	EXIT_IF(g_alloc_callback != nullptr || g_free_callback != nullptr);
	EXIT_IF(alloc_func == nullptr || free_func == nullptr);

	MemoryOperationLock memory_operation_lock;

	g_alloc_callback = alloc_func;
	g_free_callback  = free_func;

	g_physical_memory->GetMutex().Lock();
	for (const auto& b: g_physical_memory->GetMappings()) {
		if (b.map_vaddr != 0 && b.map_size != 0) {
			g_alloc_callback(b.map_vaddr, b.map_size);
		}
	}
	g_physical_memory->GetMutex().Unlock();

	g_flexible_memory->GetMutex().Lock();
	for (const auto& b: g_flexible_memory->GetBlocks()) {
		g_alloc_callback(b.map_vaddr, b.map_size);
	}
	g_flexible_memory->GetMutex().Unlock();

	for (const auto& mapping: g_pooled_memory->GetMappings()) {
		g_alloc_callback(mapping.vaddr, mapping.size);
	}
}

void SetFlexibleMemorySize(uint64_t size) {
	constexpr uint64_t GuestPageSize = 0x4000;
	EXIT_IF(g_flexible_memory_size_frozen || g_guest_address_space != nullptr);
	EXIT_IF(size == 0 || (size & (GuestPageSize - 1u)) != 0 || size >= PhysicalMemory::TotalSize());
	g_flexible_memory_size = size;
	LOGF("\t flexible memory size = 0x%016" PRIx64 " (%" PRIu64 " MiB)\n", size,
	     size / (1024ull * 1024ull));
}

bool PhysicalMemory::Alloc(uint64_t search_start, uint64_t search_end, size_t len, size_t alignment,
                           uint64_t* phys_addr_out, int memory_type, bool pool_expansion) {
	if (phys_addr_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	const auto requested_end = search_end;
	search_end               = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	// The first free range holding `len` in [first, last), or m_free.end() (`position` its place).
	const auto find = [&](uint64_t first, uint64_t last, uint64_t& position) {
		auto range = m_free.upper_bound(first);
		if (range != m_free.begin()) {
			range--;
		}
		for (; range != m_free.end() && range->first < last; ++range) {
			const auto range_end   = std::min<uint64_t>(range->first + range->second, last);
			const auto lower_bound = std::max(range->first, first);
			const auto aligned     = GetAlignedPos(lower_bound, alignment);
			const auto free_pos    = aligned.value;
			if (!aligned.valid || free_pos < lower_bound || free_pos > range_end ||
			    len > range_end - free_pos) {
				continue;
			}
			position = free_pos;
			return range;
		}
		return m_free.end();
	};
	uint64_t free_pos = 0;
	auto     range    = find(search_start, search_end, free_pos);
	if (range == m_free.end() && !pool_expansion && DirectMemoryOverflowSize() != 0 &&
	    requested_end >= Size()) {
		// KYTY_DMEM_OVERFLOW_MB: a direct search up to the end of the direct memory that found
		// no room there takes the room past it.
		range = find(TotalSize(), TotalSize() + DirectMemoryOverflowSize(), free_pos);
		if (range != m_free.end()) {
			const auto n = ++m_overflow_allocations;
			if (n <= 16 || std::has_single_bit(n)) {
				uint64_t in_overflow = 0;
				for (const auto& [start, block]: m_physical) {
					if (start >= TotalSize()) {
						in_overflow += block.size;
					}
				}
				LOGF("Direct memory: 0x%" PRIx64 " bytes did not fit in the game's %" PRIu64
				     " MiB: given room past it at 0x%" PRIx64 " (%" PRIu64 " MiB of the %" PRIu64
				     " MiB there in use before it; %" PRIu64
				     " such allocations so far; KYTY_DMEM_OVERFLOW_MB)\n",
				     static_cast<uint64_t>(len), Size() >> 20u, free_pos, in_overflow >> 20u,
				     DirectMemoryOverflowSize() >> 20u, n);
			}
		}
	}
	if (range != m_free.end()) {
		AllocatedBlock b {};
		b.size           = len;
		b.start_addr     = free_pos;
		b.gpu_mode       = GpuAccessMode::NoAccess;
		b.map_size       = 0;
		b.map_vaddr      = 0;
		b.prot           = 0;
		b.mode           = VirtualMemory::Mode::NoAccess;
		b.memory_type    = memory_type;
		b.pool_expansion = pool_expansion;

		ConsumeFreeRange(range, free_pos, len);
		EXIT_IF(!m_physical.emplace(b.start_addr, b).second);

		*phys_addr_out = free_pos;
		return true;
	}

	return false;
}

bool PhysicalMemory::Available(uint64_t search_start, uint64_t search_end, size_t alignment,
                               uint64_t* phys_addr_out, uint64_t* size_out) {
	if (phys_addr_out == nullptr || size_out == nullptr) {
		return false;
	}

	Common::LockGuard lock(m_mutex);

	search_end = std::min<uint64_t>(search_end, Size());
	if (search_start >= search_end) {
		return false;
	}

	uint64_t best_addr = 0;
	uint64_t best_size = 0;

	for (const auto& [range_start, range_size]: m_free) {
		if (range_start >= search_end) {
			break;
		}
		const auto range_end   = std::min<uint64_t>(range_start + range_size, search_end);
		const auto lower_bound = std::max(range_start, search_start);
		const auto aligned     = GetAlignedPos(lower_bound, alignment);
		const auto free_pos    = aligned.value;
		if (aligned.valid && free_pos >= lower_bound && free_pos < range_end &&
		    range_end - free_pos > best_size) {
			best_addr = free_pos;
			best_size = range_end - free_pos;
		}
	}

	if (best_size == 0) {
		return false;
	}

	*phys_addr_out = best_addr;
	*size_out      = best_size;
	return true;
}

void PhysicalMemory::ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
                                      uint64_t size) {
	const auto range_start = range->first;
	const auto range_end   = range->first + range->second;
	m_free.erase(range);
	if (range_start < start) {
		m_free.emplace(range_start, start - range_start);
	}
	if (start + size < range_end) {
		m_free.emplace(start + size, range_end - start - size);
	}
}

void PhysicalMemory::AddFreeRange(uint64_t start, uint64_t size) {
	auto end  = start + size;
	auto next = m_free.lower_bound(start);
	if (next != m_free.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = m_free.erase(previous);
		}
	}
	while (next != m_free.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = m_free.erase(next);
	}
	m_free.emplace(start, end - start);
}

bool PhysicalMemory::Release(uint64_t start, size_t len, uint64_t* vaddr, uint64_t* size,
                             GpuAccessMode* gpu_mode) {
	EXIT_IF(vaddr == nullptr);
	EXIT_IF(size == nullptr);
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	auto next = m_physical.upper_bound(start);
	if (next == m_physical.begin()) {
		return false;
	}
	auto  it = std::prev(next);
	auto& b  = it->second;
	if (b.pool_expansion || start < b.start_addr || start >= b.start_addr + b.size ||
	    len > b.start_addr + b.size - start) {
		return false;
	}

	if (start == b.start_addr && len == b.size) {
		*vaddr    = b.map_vaddr;
		*size     = b.map_size;
		*gpu_mode = b.gpu_mode;

		m_physical.erase(it);
		AddFreeRange(start, len);
		return true;
	}
	if (start > b.start_addr && start + len < b.start_addr + b.size) {
		auto old_start = b.start_addr;
		auto old_end   = b.start_addr + b.size;

		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - old_start) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock right = b;
		right.start_addr     = start + len;
		right.size           = old_end - right.start_addr;
		if (right.map_vaddr != 0) {
			right.map_vaddr += right.start_addr - old_start;
			right.map_size = right.size;
		}

		b.size = start - old_start;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}

		m_physical.emplace(right.start_addr, right);
		AddFreeRange(start, len);
		return true;
	}
	if (start == b.start_addr && len < b.size) {
		*vaddr    = b.map_vaddr;
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		AllocatedBlock remaining = b;
		m_physical.erase(it);
		remaining.start_addr += len;
		remaining.size -= len;
		if (remaining.map_vaddr != 0) {
			remaining.map_vaddr += len;
			remaining.map_size -= len;
		}
		m_physical.emplace(remaining.start_addr, remaining);
		AddFreeRange(start, len);
		return true;
	}
	if (start > b.start_addr && start + len == b.start_addr + b.size) {
		*vaddr    = (b.map_vaddr != 0 ? b.map_vaddr + (start - b.start_addr) : 0);
		*size     = (b.map_vaddr != 0 ? len : 0);
		*gpu_mode = b.gpu_mode;

		b.size = start - b.start_addr;
		if (b.map_vaddr != 0) {
			b.map_size = b.size;
		}
		AddFreeRange(start, len);
		return true;
	}

	return false;
}

bool PhysicalMemory::Map(uint64_t vaddr, uint64_t phys_addr, size_t len, int prot,
                         VirtualMemory::Mode mode, GpuAccessMode gpu_mode, const char* name,
                         int memory_type) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	auto current = phys_addr;
	auto next    = m_physical.upper_bound(current);
	if (next == m_physical.begin()) {
		return false;
	}
	auto first = std::prev(next);
	while (current < phys_addr + len) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		current = std::min<uint64_t>(phys_addr + len, block_end);
		if (current < phys_addr + len) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}

	AllocatedBlock mapping = first->second;
	mapping.start_addr     = phys_addr;
	mapping.size           = len;
	mapping.map_vaddr      = vaddr;
	mapping.map_size       = len;
	mapping.host_vaddr     = vaddr;
	mapping.host_size      = len;
	mapping.prot           = prot;
	mapping.mode           = mode;
	mapping.gpu_mode       = gpu_mode;
	mapping.memory_type    = memory_type;
	CopyVirtualRangeName(mapping.name, name);
	m_mappings.push_back(mapping);

	return true;
}

bool PhysicalMemory::CanMapDirect(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);

	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	const auto end     = phys_addr + len;
	auto       current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			return false;
		}
		current = std::min(end, block_end);
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::ReleasePoolExpansion(uint64_t phys_addr, size_t len) {
	Common::LockGuard lock(m_mutex);
	const auto        it = m_physical.find(phys_addr);
	if (it == m_physical.end() || !it->second.pool_expansion || it->second.size != len) {
		return false;
	}
	m_physical.erase(it);
	AddFreeRange(phys_addr, len);
	return true;
}

bool PhysicalMemory::GetAllocatedSpan(uint64_t phys_addr, size_t len,
                                      std::vector<AllocatedBlock>* blocks) {
	EXIT_IF(blocks == nullptr);
	blocks->clear();
	if (len == 0 || UINT64_MAX - phys_addr < len) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	const auto        end     = phys_addr + len;
	auto              current = phys_addr;
	while (current < end) {
		auto block = m_physical.upper_bound(current);
		if (block == m_physical.begin()) {
			blocks->clear();
			return false;
		}
		--block;
		const auto block_end = block->second.start_addr + block->second.size;
		if (block->second.pool_expansion || current < block->second.start_addr ||
		    current >= block_end) {
			blocks->clear();
			return false;
		}

		auto part       = block->second;
		part.start_addr = current;
		part.size       = std::min(end, block_end) - current;
		blocks->push_back(part);
		current += part.size;
		if (current < end) {
			const auto following = std::next(block);
			if (following == m_physical.end() || following->second.start_addr != current) {
				blocks->clear();
				return false;
			}
		}
	}
	return true;
}

bool PhysicalMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}

	auto set_host_release_if_last = [this, host_vaddr_to_release](uint64_t host_vaddr,
	                                                              uint64_t host_size) {
		if (host_vaddr_to_release == nullptr || host_vaddr == 0 || host_size == 0) {
			return;
		}
		const bool still_mapped = std::any_of(
		    m_mappings.begin(), m_mappings.end(), [host_vaddr, host_size](const auto& block) {
			    return block.host_vaddr == host_vaddr && block.host_size == host_size;
		    });
		if (!still_mapped) {
			*host_vaddr_to_release = host_vaddr;
		}
	};

	size_t index = 0;
	for (auto& b: m_mappings) {
		if (b.map_vaddr == vaddr && b.map_size == size) {
			*gpu_mode             = b.gpu_mode;
			const auto host_vaddr = b.host_vaddr;
			const auto host_size  = b.host_size;

			m_mappings.erase(m_mappings.begin() + static_cast<std::ptrdiff_t>(index));
			set_host_release_if_last(host_vaddr, host_size);

			return true;
		}
		if (vaddr > b.map_vaddr && vaddr + size < b.map_vaddr + b.map_size) {
			*gpu_mode = b.gpu_mode;

			AllocatedBlock right = b;
			right.start_addr += (vaddr + size) - b.map_vaddr;
			right.size      = b.map_vaddr + b.map_size - (vaddr + size);
			right.map_size  = right.size;
			right.map_vaddr = vaddr + size;

			b.size     = vaddr - b.map_vaddr;
			b.map_size = b.size;
			m_mappings.push_back(right);
			return true;
		}
		if (vaddr == b.map_vaddr && size < b.map_size) {
			*gpu_mode = b.gpu_mode;

			b.start_addr += size;
			b.size -= size;
			b.map_vaddr += size;
			b.map_size -= size;
			return true;
		}
		if (vaddr > b.map_vaddr && vaddr + size == b.map_vaddr + b.map_size) {
			*gpu_mode = b.gpu_mode;

			b.size     = vaddr - b.map_vaddr;
			b.map_size = b.size;
			return true;
		}
		index++;
	}

	return false;
}

void PhysicalMemory::ProtectMapping(uint64_t vaddr, uint64_t size, int prot,
                                    VirtualMemory::Mode mode, GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	updated.reserve(m_mappings.size() + 2);
	for (const auto& block: m_mappings) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			updated.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.size     = overlap_start - block.map_vaddr;
			left.map_size = left.size;
			updated.push_back(left);
		}
		auto middle       = block;
		middle.start_addr = block.start_addr + overlap_start - block.map_vaddr;
		middle.size       = overlap_end - overlap_start;
		middle.map_vaddr  = overlap_start;
		middle.map_size   = middle.size;
		middle.prot       = prot;
		middle.mode       = mode;
		middle.gpu_mode   = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right       = block;
			right.start_addr = block.start_addr + overlap_end - block.map_vaddr;
			right.size       = block_end - overlap_end;
			right.map_vaddr  = overlap_end;
			right.map_size   = right.size;
			updated.push_back(right);
		}
	}
	m_mappings = std::move(updated);
}

bool PhysicalMemory::Find(uint64_t phys_addr, bool next, AllocatedBlock* out) {
	EXIT_IF(out == nullptr);

	Common::LockGuard lock(m_mutex);

	auto following = m_physical.upper_bound(phys_addr);
	if (following != m_physical.begin()) {
		const auto& block = std::prev(following)->second;
		if (phys_addr < block.start_addr + block.size) {
			*out = block;
			return true;
		}
	}

	if (next && following != m_physical.end()) {
		*out = following->second;
		return true;
	}

	return false;
}

std::vector<PhysicalMemory::AllocatedBlock> PhysicalMemory::FindMappings(uint64_t phys_addr,
                                                                         size_t   len) {
	Common::LockGuard lock(m_mutex);

	std::vector<AllocatedBlock> mappings;
	for (const auto& m: m_mappings) {
		if (!VirtualRangesOverlap(phys_addr, len, m.start_addr, m.size)) {
			continue;
		}

		const uint64_t overlap_start = std::max<uint64_t>(phys_addr, m.start_addr);
		const uint64_t overlap_end   = std::min<uint64_t>(phys_addr + len, m.start_addr + m.size);
		AllocatedBlock part          = m;
		part.start_addr              = overlap_start;
		part.size                    = overlap_end - overlap_start;
		part.map_vaddr += overlap_start - m.start_addr;
		part.map_size = part.size;
		mappings.push_back(part);
	}

	std::sort(mappings.begin(), mappings.end(),
	          [](const auto& a, const auto& b) { return a.map_vaddr < b.map_vaddr; });
	return mappings;
}

bool PhysicalMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	return std::any_of(m_mappings.begin(), m_mappings.end(),
	                   [vaddr, base_addr, len, prot, mode, gpu_mode](auto& b) {
		                   if (vaddr >= b.map_vaddr && vaddr < b.map_vaddr + b.map_size) {
			                   if (base_addr != nullptr) {
				                   *base_addr = b.map_vaddr;
			                   }
			                   if (len != nullptr) {
				                   *len = b.map_size;
			                   }
			                   if (prot != nullptr) {
				                   *prot = b.prot;
			                   }
			                   if (mode != nullptr) {
				                   *mode = b.mode;
			                   }
			                   if (gpu_mode != nullptr) {
				                   *gpu_mode = b.gpu_mode;
			                   }

			                   return true;
		                   }
		                   return false;
	                   });
}

void PhysicalMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_mappings) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			CopyVirtualRangeName(b.name, name);
		}
	}
}

bool FlexibleMemory::Map(uint64_t vaddr, size_t len, int prot, VirtualMemory::Mode mode,
                         GpuAccessMode gpu_mode, const char* name) {
	Common::LockGuard lock(m_mutex);

	const auto available = (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
	if (len == 0 || len > available) {
		LOGF_COLOR(Log::Color::Red,
		           "\t flexible memory exhausted: configured = 0x%016" PRIx64
		           ", allocated = 0x%016" PRIx64 ", available = 0x%016" PRIx64
		           ", requested = 0x%016" PRIx64 "\n",
		           Size(), m_allocated_total, available, static_cast<uint64_t>(len));
		return false;
	}

	std::vector<AllocatedBlock> blocks;
	auto                        current   = vaddr;
	auto                        remaining = static_cast<uint64_t>(len);
	for (const auto& [backing_offset, free_size]: m_free) {
		if (remaining == 0) {
			break;
		}
		const auto     chunk = std::min(remaining, free_size);
		AllocatedBlock block {};
		block.map_vaddr      = current;
		block.map_size       = chunk;
		block.backing_offset = backing_offset;
		block.host_vaddr     = vaddr;
		block.host_size      = len;
		block.prot           = prot;
		block.mode           = mode;
		block.gpu_mode       = gpu_mode;
		CopyVirtualRangeName(block.name, name);
		blocks.push_back(block);
		current += chunk;
		remaining -= chunk;
	}
	if (remaining != 0) {
		return false;
	}

	// The caller validated the target; failing here is an emulator fault (upstream 563d80156: no
	// re-mapping rollback).
	for (const auto& block: blocks) {
		EXIT_IF(!g_guest_address_space->ZeroBacking(block.backing_offset, block.map_size));
		EXIT_IF(!g_guest_address_space->MapBacking(block.map_vaddr, block.map_size,
		                                           block.backing_offset, block.mode));
	}

	for (const auto& block: blocks) {
		auto next = m_free.upper_bound(block.backing_offset);
		EXIT_IF(next == m_free.begin());
		auto range = std::prev(next);
		EXIT_IF(block.backing_offset < range->first ||
		        block.map_size > range->first + range->second - block.backing_offset);
		ConsumeFreeRange(range, block.backing_offset, block.map_size);
		m_allocated.push_back(block);
	}
	std::sort(m_allocated.begin(), m_allocated.end(),
	          [](const auto& left, const auto& right) { return left.map_vaddr < right.map_vaddr; });
	m_allocated_total += len;

	return true;
}

bool FlexibleMemory::Unmap(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode,
                           uint64_t* host_vaddr_to_release) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard lock(m_mutex);

	if (host_vaddr_to_release != nullptr) {
		*host_vaddr_to_release = 0;
	}
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	bool       found   = false;
	for (const auto& block: m_allocated) {
		if (block.map_vaddr + block.map_size <= current) {
			continue;
		}
		if (block.map_vaddr > current || block.map_vaddr >= end) {
			break;
		}
		if (!found) {
			*gpu_mode = block.gpu_mode;
			found     = true;
		}
		current = std::min(end, block.map_vaddr + block.map_size);
		if (current == end) {
			break;
		}
	}
	if (!found || current != end || !g_guest_address_space->UnmapBacking(vaddr, size)) {
		return false;
	}

	std::vector<AllocatedBlock> remaining;
	remaining.reserve(m_allocated.size() + 1);
	uint64_t removed = 0;
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			remaining.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		const auto overlap_size  = overlap_end - overlap_start;
		AddFreeRange(block.backing_offset + overlap_start - block.map_vaddr, overlap_size);
		removed += overlap_size;

		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			remaining.push_back(left);
		}
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			remaining.push_back(right);
		}
	}
	EXIT_IF(removed != size || removed > m_allocated_total);
	m_allocated = std::move(remaining);
	m_allocated_total -= removed;
	return true;
}

void FlexibleMemory::ConsumeFreeRange(std::map<uint64_t, uint64_t>::iterator range, uint64_t start,
                                      uint64_t size) {
	const auto range_start = range->first;
	const auto range_end   = range->first + range->second;
	m_free.erase(range);
	if (range_start < start) {
		m_free.emplace(range_start, start - range_start);
	}
	if (start + size < range_end) {
		m_free.emplace(start + size, range_end - start - size);
	}
}

void FlexibleMemory::AddFreeRange(uint64_t start, uint64_t size) {
	auto end  = start + size;
	auto next = m_free.lower_bound(start);
	if (next != m_free.begin()) {
		auto previous = std::prev(next);
		if (previous->first + previous->second >= start) {
			start = previous->first;
			end   = std::max(end, previous->first + previous->second);
			next  = m_free.erase(previous);
		}
	}
	while (next != m_free.end() && next->first <= end) {
		end  = std::max(end, next->first + next->second);
		next = m_free.erase(next);
	}
	m_free.emplace(start, end - start);
}

void FlexibleMemory::Protect(uint64_t vaddr, uint64_t size, int prot, VirtualMemory::Mode mode,
                             GpuAccessMode gpu_mode) {
	Common::LockGuard lock(m_mutex);
	if (size == 0 || UINT64_MAX - vaddr < size) {
		return;
	}

	const auto                  end = vaddr + size;
	std::vector<AllocatedBlock> updated;
	updated.reserve(m_allocated.size() + 2);
	for (const auto& block: m_allocated) {
		const auto block_end = block.map_vaddr + block.map_size;
		if (!VirtualRangesOverlap(vaddr, size, block.map_vaddr, block.map_size)) {
			updated.push_back(block);
			continue;
		}
		const auto overlap_start = std::max(vaddr, block.map_vaddr);
		const auto overlap_end   = std::min(end, block_end);
		if (block.map_vaddr < overlap_start) {
			auto left     = block;
			left.map_size = overlap_start - block.map_vaddr;
			updated.push_back(left);
		}
		auto middle           = block;
		middle.map_vaddr      = overlap_start;
		middle.map_size       = overlap_end - overlap_start;
		middle.backing_offset = block.backing_offset + overlap_start - block.map_vaddr;
		middle.prot           = prot;
		middle.mode           = mode;
		middle.gpu_mode       = gpu_mode;
		updated.push_back(middle);
		if (overlap_end < block_end) {
			auto right           = block;
			right.map_vaddr      = overlap_end;
			right.map_size       = block_end - overlap_end;
			right.backing_offset = block.backing_offset + overlap_end - block.map_vaddr;
			updated.push_back(right);
		}
	}
	m_allocated = std::move(updated);
}

bool FlexibleMemory::Find(uint64_t vaddr, uint64_t* base_addr, size_t* len, int* prot,
                          VirtualMemory::Mode* mode, GpuAccessMode* gpu_mode) {
	Common::LockGuard lock(m_mutex);

	return std::any_of(m_allocated.begin(), m_allocated.end(),
	                   [vaddr, base_addr, len, prot, mode, gpu_mode](auto& b) {
		                   if (vaddr >= b.map_vaddr && vaddr < b.map_vaddr + b.map_size) {
			                   if (base_addr != nullptr) {
				                   *base_addr = b.map_vaddr;
			                   }
			                   if (len != nullptr) {
				                   *len = b.map_size;
			                   }
			                   if (prot != nullptr) {
				                   *prot = b.prot;
			                   }
			                   if (mode != nullptr) {
				                   *mode = b.mode;
			                   }
			                   if (gpu_mode != nullptr) {
				                   *gpu_mode = b.gpu_mode;
			                   }

			                   return true;
		                   }
		                   return false;
	                   });
}

void FlexibleMemory::SetVirtualRangeName(uint64_t vaddr, uint64_t len, const char* name) {
	Common::LockGuard lock(m_mutex);

	for (auto& b: m_allocated) {
		if (VirtualRangesOverlap(vaddr, len, b.map_vaddr, b.map_size)) {
			CopyVirtualRangeName(b.name, name);
		}
	}
}

void FlexibleMemory::Charge(uint64_t size) {
	Common::LockGuard lock(m_mutex);

	m_allocated_total += size;
}

void FlexibleMemory::Release(uint64_t size) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(size > m_allocated_total);
	m_allocated_total -= size;
}

uint64_t FlexibleMemory::Available() {
	Common::LockGuard lock(m_mutex);

	return (Size() >= m_allocated_total ? Size() - m_allocated_total : 0);
}

void PooledMemory::AddFreeUnlocked(uint64_t start, uint64_t size) {
	if (size == 0) {
		return;
	}

	m_free.push_back({start, size});
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });

	std::vector<PhysicalRange> merged;
	for (const auto& range: m_free) {
		if (!merged.empty() && range.start <= merged.back().start + merged.back().size) {
			const auto end =
			    std::max(merged.back().start + merged.back().size, range.start + range.size);
			merged.back().size = end - merged.back().start;
		} else {
			merged.push_back(range);
		}
	}
	m_free = std::move(merged);
}

void PooledMemory::Expand(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	m_expansions.push_back({phys_addr, size});
	AddFreeUnlocked(phys_addr, size);
}

bool PooledMemory::ReleaseExpansion(uint64_t phys_addr, uint64_t size) {
	Common::LockGuard lock(m_mutex);
	const auto        expansion = std::find_if(m_expansions.begin(), m_expansions.end(),
	                                           [phys_addr, size](const auto& range) {
		                                    return range.start == phys_addr && range.size == size;
	                                           });
	if (expansion == m_expansions.end()) {
		return false;
	}
	if (std::any_of(m_mappings.begin(), m_mappings.end(), [phys_addr, size](const auto& mapping) {
		    return VirtualRangesOverlap(phys_addr, size, mapping.phys_addr, mapping.size);
	    })) {
		return false;
	}

	const auto end = phys_addr + size;
	const auto free_range =
	    std::find_if(m_free.begin(), m_free.end(), [phys_addr, end](const auto& r) {
		    return phys_addr >= r.start && end <= r.start + r.size;
	    });
	if (free_range == m_free.end()) {
		return false;
	}

	const auto old        = *free_range;
	const auto old_end    = old.start + old.size;
	const auto left_size  = phys_addr - old.start;
	const auto right_size = old_end - end;
	m_free.erase(free_range);
	if (left_size != 0) {
		m_free.push_back({old.start, left_size});
	}
	if (right_size != 0) {
		m_free.push_back({end, right_size});
	}
	std::sort(m_free.begin(), m_free.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });
	m_expansions.erase(expansion);
	return true;
}

bool PooledMemory::Allocate(uint64_t vaddr, uint64_t size, GpuAccessMode gpu_mode,
                            std::vector<Mapping>* mappings) {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	Common::LockGuard lock(m_mutex);
	auto              free      = m_free;
	auto              current   = vaddr;
	auto              remaining = size;
	for (auto& range: free) {
		if (remaining == 0 || range.size == 0) {
			continue;
		}
		const auto part_size = std::min(range.size, remaining);
		mappings->push_back({current, part_size, range.start, gpu_mode});
		range.start += part_size;
		range.size -= part_size;
		current += part_size;
		remaining -= part_size;
	}
	if (remaining != 0) {
		mappings->clear();
		return false;
	}

	free.erase(
	    std::remove_if(free.begin(), free.end(), [](const auto& range) { return range.size == 0; }),
	    free.end());
	m_free = std::move(free);
	m_mappings.insert(m_mappings.end(), mappings->begin(), mappings->end());
	return true;
}

bool PooledMemory::QueryUnlocked(uint64_t vaddr, uint64_t size,
                                 std::vector<Mapping>* mappings) const {
	EXIT_IF(mappings == nullptr);
	mappings->clear();
	if (vaddr == 0 || size == 0 || UINT64_MAX - vaddr < size) {
		return false;
	}

	const auto end     = vaddr + size;
	auto       current = vaddr;
	while (current < end) {
		const auto it =
		    std::find_if(m_mappings.begin(), m_mappings.end(), [current](const auto& m) {
			    return current >= m.vaddr && current < m.vaddr + m.size;
		    });
		if (it == m_mappings.end()) {
			mappings->clear();
			return false;
		}
		const auto part_size = std::min(end, it->vaddr + it->size) - current;
		mappings->push_back(
		    {current, part_size, it->phys_addr + (current - it->vaddr), it->gpu_mode});
		current += part_size;
	}
	return true;
}

bool PooledMemory::Query(uint64_t vaddr, uint64_t size, std::vector<Mapping>* mappings) {
	Common::LockGuard lock(m_mutex);
	return QueryUnlocked(vaddr, size, mappings);
}

bool PooledMemory::Release(uint64_t vaddr, uint64_t size, GpuAccessMode* gpu_mode) {
	EXIT_IF(gpu_mode == nullptr);

	Common::LockGuard    lock(m_mutex);
	std::vector<Mapping> released;
	if (!QueryUnlocked(vaddr, size, &released)) {
		return false;
	}

	*gpu_mode                = released.front().gpu_mode;
	const auto           end = vaddr + size;
	std::vector<Mapping> kept;
	for (const auto& mapping: m_mappings) {
		const auto mapping_end = mapping.vaddr + mapping.size;
		const auto cut_start   = std::max(vaddr, mapping.vaddr);
		const auto cut_end     = std::min(end, mapping_end);
		if (cut_start >= cut_end) {
			kept.push_back(mapping);
			continue;
		}
		if (mapping.vaddr < cut_start) {
			auto left = mapping;
			left.size = cut_start - mapping.vaddr;
			kept.push_back(left);
		}
		if (cut_end < mapping_end) {
			auto right      = mapping;
			right.vaddr     = cut_end;
			right.size      = mapping_end - cut_end;
			right.phys_addr = mapping.phys_addr + (cut_end - mapping.vaddr);
			kept.push_back(right);
		}
	}
	m_mappings = std::move(kept);
	for (const auto& mapping: released) {
		AddFreeUnlocked(mapping.phys_addr, mapping.size);
	}
	return true;
}

uint64_t PooledMemory::Available() {
	Common::LockGuard lock(m_mutex);
	uint64_t          available = 0;
	for (const auto& range: m_free) {
		available += range.size;
	}
	return available;
}

std::vector<PooledMemory::Mapping> PooledMemory::GetMappings() {
	Common::LockGuard lock(m_mutex);
	return m_mappings;
}

// The caller validated the mappings (PooledMemory::Query); failing here is an emulator fault
// (upstream e2777436f: exit instead of rolling back).
static void UnmapPooledBacking(const std::vector<PooledMemory::Mapping>& mappings) {
	for (const auto& mapping: mappings) {
		if (!g_guest_address_space->UnmapBacking(mapping.vaddr, mapping.size)) {
			EXIT("failed to unmap pooled memory: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     mapping.vaddr, mapping.size);
		}
	}
}

static bool IsReservedSpan(uint64_t start, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	return g_virtual_ranges->QuerySpan(start, size, &ranges) &&
	       std::all_of(ranges.begin(), ranges.end(),
	                   [](const auto& range) { return range.type == VirtualRangeType::Reserved; });
}

// The ABI entry points and batch operations hold g_memory_operation_mutex.
static int MapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                             const char* name) {
	EXIT_NOT_IMPLEMENTED(addr_in_out == nullptr);

	constexpr size_t   PAGE_SIZE                = 0x4000;
	constexpr size_t   MAXIMUM_NAME_SIZE        = 32;
	constexpr uint64_t DEFAULT_PS5_BASE         = 0x200000000;
	constexpr uint32_t GUEST_MAP_FIXED          = 0x10;
	constexpr uint32_t GUEST_MAP_NO_OVERWRITE   = 0x80;
	constexpr uint32_t GUEST_MAP_DMEM_COMPAT    = 0x400;
	constexpr uint32_t GUEST_MAP_UNKNOWN_8000   = 0x8000;
	constexpr uint32_t GUEST_MAP_NO_COALESCE    = 0x400000;
	constexpr uint32_t GUEST_MAP_ALIGNMENT_MASK = 0xff000000;
	constexpr uint32_t SUPPORTED_MAP_BITS       = GUEST_MAP_FIXED | GUEST_MAP_NO_OVERWRITE |
	                                              GUEST_MAP_DMEM_COMPAT | GUEST_MAP_UNKNOWN_8000 |
	                                              GUEST_MAP_NO_COALESCE | GUEST_MAP_ALIGNMENT_MASK;

	if (len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (len > g_flexible_memory->Available()) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= MAXIMUM_NAME_SIZE) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	const auto map_flags       = static_cast<uint32_t>(flags);
	const auto alignment_shift = (map_flags & GUEST_MAP_ALIGNMENT_MASK) >> 24u;
	if ((map_flags & ~SUPPORTED_MAP_BITS) != 0 ||
	    (alignment_shift != 0 && (alignment_shift < 14 || alignment_shift > 31))) {
		LOGF_COLOR(Log::Color::Red, "\t unsupported flags = 0x%08" PRIx32 "\n", map_flags);
		return KERNEL_ERROR_EINVAL;
	}
	const uint64_t map_alignment =
	    alignment_shift != 0 ? uint64_t {1} << alignment_shift : PAGE_SIZE;

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		EXIT("unknown prot: %d\n", prot);
	}

	// KYTY_TRANSIENT_UNMAP_RETRY: guest threads may use a mapping that MAP_FIXED replaces: until
	// the new one is in place, their faults wait and are retried.
	std::optional<TransientUnmapWindows::Scope> replacing;

	auto     in_addr         = reinterpret_cast<uint64_t>(*addr_in_out);
	uint64_t out_addr        = 0;
	bool     reserved_target = false;

	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0 ||
		    (in_addr & (map_alignment - 1u)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}
		// Upstream 466551e64: the replacement is validated up front and its error returned.
		if (!IsReservedSpan(in_addr, len)) {
			if (TransientUnmapRetryEnabled()) {
				replacing.emplace(in_addr, in_addr + len);
			}
			const int result = ReplaceFixedRangeWithReserved(in_addr, len);
			if (result != OK) {
				return result;
			}
		}
		UnmapGpuRange(in_addr, len);
		reserved_target = true;
		out_addr        = in_addr;
	} else {
		const auto search_addr = (in_addr != 0 ? in_addr : DEFAULT_PS5_BASE);
		out_addr               = FindGuestFreeRange(search_addr, len, map_alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
		}
	}

	*addr_in_out = reinterpret_cast<void*>(out_addr);

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (!g_flexible_memory->Map(out_addr, len, prot, mode, gpu_mode, name)) {
		LOGF_COLOR(Log::Color::Red, "\t [Fail]\n");
		return KERNEL_ERROR_ENOMEM;
	}

	const bool published = reserved_target
	                           ? g_virtual_ranges->ReplaceSpan(
	                                 out_addr, len, VirtualRangeType::Reserved, 0, prot, 0,
	                                 VirtualRangeType::Flexible, name,
	                                 (map_flags & GUEST_MAP_NO_COALESCE) != 0)
	                           : g_virtual_ranges->Add(out_addr, len, 0, prot, 0,
	                                                   VirtualRangeType::Flexible, name,
	                                                   (map_flags & GUEST_MAP_NO_COALESCE) != 0);
	// The target was validated above (upstream dcdef60bd: no rollback).
	EXIT_IF(!published);

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t size     = %" PRIu64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t name     = %s\n"
	     "\t gpu_mode = %s\n",
	     in_addr, out_addr, len, magic_enum::enum_name(mode), static_cast<uint32_t>(flags), name,
	     magic_enum::enum_name(gpu_mode));

	MapGpuRange(out_addr, len);
	Libs::Graphics::WatchRange::Note("map direct memory", out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}

	return OK;
}

int32_t KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot,
                                                   int flags, const char* name) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return MapFlexibleMemory(addr_in_out, len, prot, flags, name);
}

int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return MapFlexibleMemory(addr_in_out, len, prot, flags, "");
}

int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	const auto address = reinterpret_cast<uint64_t>(addr);

	LOGF("\t index = %d\n"
	     "\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     index, address, static_cast<uint64_t>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (len != 0 && (address == 0 || (address & (PRT_PAGE_SIZE - 1u)) != 0 ||
	                 (len & (PRT_PAGE_SIZE - 1u)) != 0 || address < PRT_APERTURE_START ||
	                 len > PRT_APERTURE_END - address)) {
		return KERNEL_ERROR_EINVAL;
	}

	PrtAperture old {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		old = g_prt_apertures[static_cast<size_t>(index)];
	}
	if (old.size != 0) {
		UnmapGpuRange(old.address, old.size);
	}
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		g_prt_apertures[static_cast<size_t>(index)] =
		    len == 0 ? PrtAperture {} : PrtAperture {address, static_cast<uint64_t>(len)};
	}
	if (len != 0) {
		MapGpuRange(address, len);
	}

	LOGF_COLOR(Log::Color::Green, "\t[Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len) {
	PRINT_NAME();

	LOGF("\t index = %d\n"
	     "\t addr  = %p\n"
	     "\t len   = %p\n",
	     index, static_cast<void*>(addr), static_cast<void*>(len));

	if (index < 0 || index > PRT_APERTURE_MAX_INDEX) {
		return KERNEL_ERROR_EINVAL;
	}

	if (addr == nullptr || len == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	PrtAperture aperture {};
	{
		Common::LockGuard lock(g_prt_aperture_mutex);
		aperture = g_prt_apertures[static_cast<size_t>(index)];
	}

	*addr = reinterpret_cast<void*>(aperture.address);
	*len  = static_cast<size_t>(aperture.size);

	LOGF_COLOR(Log::Color::Green,
	           "\t *addr = 0x%016" PRIx64 "\n"
	           "\t *len  = 0x%016" PRIx64 "\n"
	           "\t[Ok]\n",
	           aperture.address, aperture.size);

	return OK;
}

// Names change in whole 16 KiB pages, like protections (upstream 100fdacf8).
static int RenameVirtualRange(uint64_t vaddr, uint64_t len, const char* name) {
	constexpr uint64_t page_size   = 0x4000;
	const auto         start       = vaddr & ~(page_size - 1);
	const auto         page_offset = vaddr - start;
	if (len > UINT64_MAX - (page_size - 1) - page_offset) {
		return KERNEL_ERROR_EINVAL;
	}
	if (len != 0) {
		const auto size = (page_offset + len + page_size - 1) & ~(page_size - 1);
		g_physical_memory->SetVirtualRangeName(start, size, name);
		g_flexible_memory->SetVirtualRangeName(start, size, name);
		g_virtual_ranges->Rename(start, size, name);
	}
	return OK;
}

int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t name = %s\n",
	     vaddr, len, name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	return RenameVirtualRange(vaddr, len, name);
}

int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n",
	     vaddr, len);

	return RenameVirtualRange(vaddr, len, "");
}

static bool FreeGuestMemoryOwner(uint64_t vaddr, uint64_t size) {
	if (!g_guest_address_space->ReleaseCommitted(vaddr, size) ||
	    !g_virtual_ranges->Remove(vaddr, size)) return false;
	Libs::Graphics::ShaderUnmapCode(vaddr, size);
	return true;
}

// One fragment of a validated span (VirtualRanges::QuerySpan/QueryFragments: start, size and the
// direct offset are the fragment's own), already unmapped from the GPU. Failing here is an emulator
// fault (upstream e2777436f: exit instead of rolling back). `keep_pool_reserved`: a pooled
// fragment becomes pool-reserved again (a decommit).
static void UnmapMemoryOwner(const VirtualRanges::Range& range, bool keep_pool_reserved = false) {
	const auto    vaddr        = range.start;
	const auto    len          = range.size;
	GpuAccessMode gpu_mode     = GpuAccessMode::NoAccess;
	uint64_t      ignored_host = 0;
	bool          unmapped     = false;
	switch (range.type) {
		case VirtualRangeType::Reserved:
		case VirtualRangeType::PoolReserved: unmapped = g_guest_address_space->ReleaseFree(vaddr, len); break;
		case VirtualRangeType::Code:
		case VirtualRangeType::Runtime: unmapped = FreeGuestMemoryOwner(vaddr, len); break;
		case VirtualRangeType::Pooled: {
			std::vector<PooledMemory::Mapping> mappings;
			if (g_pooled_memory->Query(vaddr, len, &mappings)) {
				UnmapPooledBacking(mappings);
				unmapped = g_pooled_memory->Release(vaddr, len, &gpu_mode);
			}
			break;
		}
		case VirtualRangeType::Direct:
			unmapped = g_guest_address_space->UnmapBacking(vaddr, len) &&
			           g_physical_memory->Unmap(vaddr, len, &gpu_mode, &ignored_host);
			break;
		case VirtualRangeType::Stack: unmapped = g_guest_address_space->ReleaseCommitted(vaddr, len); break;
		case VirtualRangeType::Flexible:
			unmapped = g_flexible_memory->Unmap(vaddr, len, &gpu_mode, &ignored_host);
			break;
	}
	if (!unmapped) {
		EXIT("failed to unmap guest memory: addr=0x%016" PRIx64 " size=0x%016" PRIx64 " type=%s\n",
		     vaddr, static_cast<uint64_t>(len), magic_enum::enum_name(range.type).data());
	}
	if (range.type == VirtualRangeType::Code || range.type == VirtualRangeType::Runtime) {
		return; // FreeGuestMemoryOwner removed the range and unmapped its shader code
	}

	g_virtual_ranges->Remove(vaddr, len);
	if (IsReservedRangeType(range.type)) {
		return;
	}
	Libs::Graphics::ShaderUnmapCode(vaddr, len);
	if (keep_pool_reserved) {
		EXIT_IF(!g_virtual_ranges->Add(vaddr, len, 0, 0, 0, VirtualRangeType::PoolReserved,
		                               range.name));
	}

	if (g_free_callback != nullptr) {
		g_free_callback(vaddr, len);
	}
	if (range.type == VirtualRangeType::Pooled) {
		MemoryPoolSubtractCommitted(len);
	}
}

// Reserved fragments and holes hold nothing the GPU can cache, so each run of adjacent committed
// fragments is unmapped from the GPU with one call (upstream d384084e3).
static void UnmapGpuRanges(const std::vector<VirtualRanges::Range>& ranges) {
	for (size_t i = 0; i < ranges.size(); i++) {
		if (IsReservedRangeType(ranges[i].type)) continue;
		const auto start = ranges[i].start;
		auto       end   = start + ranges[i].size;
		while (i + 1 < ranges.size() && ranges[i + 1].start == end &&
		       !IsReservedRangeType(ranges[i + 1].type)) {
			end += ranges[++i].size;
		}
		UnmapGpuRange(start, end - start);
	}
}

// The ABI entry point and batch operations hold the operation lock.
static int UnmapMemory(uint64_t vaddr, size_t len) {
	Libs::Graphics::WatchRange::Note("unmap", vaddr, len);
	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     vaddr, len);

	// Upstream b02375764: an unaligned address is rejected, the length rounds up to whole pages.
	constexpr uint64_t page_size = 0x4000;
	if (len == 0 || (vaddr & (page_size - 1)) != 0 ||
	    len > UINT64_MAX - vaddr - (page_size - 1)) {
		return KERNEL_ERROR_EINVAL;
	}
	len = (len + page_size - 1) & ~(page_size - 1);
	std::vector<VirtualRanges::Range> ranges;
	if (!g_virtual_ranges->QuerySpan(vaddr, len, &ranges)) {
		return KERNEL_ERROR_EACCES;
	}
	UnmapGpuRanges(ranges);
	for (const auto& range: ranges) {
		UnmapMemoryOwner(range);
	}
	return OK;
}

int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return UnmapMemory(vaddr, len);
}

// Upstream checks its extended-memory AMM window here; this line's AMM window is libAmpr's
// (AMM_VA_START, validated by the AMPR command builders), so only shape is checked.
static bool IsAmmMemoryRange(uint64_t vaddr, size_t size) {
	constexpr uint64_t page_size = 0x4000;
	return size != 0 && ((vaddr | size) & (page_size - 1)) == 0 && vaddr != 0 &&
	       UINT64_MAX - vaddr >= size;
}

int UnmapAmmMemory(uint64_t vaddr, size_t size) {
	MemoryOperationLock memory_operation_lock;
	if (!IsAmmMemoryRange(vaddr, size)) return KERNEL_ERROR_EINVAL;

	// AMM unmap accepts holes, including a completely unmapped range.
	std::vector<VirtualRanges::Range> ranges;
	g_virtual_ranges->QueryFragments(vaddr, size, &ranges);
	UnmapGpuRanges(ranges);
	for (const auto& range: ranges) {
		UnmapMemoryOwner(range);
	}
	return OK;
}

// KYTY_DIRECT_MEMORY_MB=<n> (default unset: all of it; chenxiao07/KytyPS5 1a7b17903, 01523253a):
// the game is told it has at most n MiB of direct memory (sceKernelGetDirectMemorySize;
// sceKernelAvailableDirectMemorySize and sceKernelAllocateMainDirectMemory search below it).
// =ds: 12 GiB (a base PS5's) for Demon's Souls only, all of it for other games. Demon's Souls'
// engine takes all it is told at start-up and its texture streamer fills it until ~400 MiB are
// left: told 12 GiB instead of 12.5 it ended 0.46 GB lower in the PC's memory over a long tour.
static uint64_t ReportedDirectMemorySize() {
	static const uint64_t cap = [] {
		const char* value = std::getenv("KYTY_DIRECT_MEMORY_MB");
		if (value == nullptr || *value == '\0') {
			return ~uint64_t {0};
		}
		uint64_t mib = 0;
		if (std::strcmp(value, "ds") == 0) {
			if (!Loader::DemonsSouls::IsSupportedGame()) {
				return ~uint64_t {0};
			}
			mib = 12288;
		} else {
			mib = std::strtoull(value, nullptr, 10);
		}
		if (mib == 0) {
			return ~uint64_t {0};
		}
		std::printf("Direct memory: at most %llu MiB told to the game (KYTY_DIRECT_MEMORY_MB=%s)\n",
		            static_cast<unsigned long long>(mib), value);
		std::fflush(stdout);
		return mib << 20u;
	}();
	return std::min<uint64_t>(PhysicalMemory::Size(), cap);
}

size_t KYTY_SYSV_ABI KernelGetDirectMemorySize() {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::DirectMemorySizeQueries);

	return ReportedDirectMemorySize();
}

int KYTY_SYSV_ABI KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                  size_t alignment, int64_t* phys_addr_out,
                                                  size_t* size_out) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::DirectMemoryAvailableQueries);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n",
	     search_start, search_end, static_cast<uint64_t>(alignment));

	if (phys_addr_out == nullptr || size_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*phys_addr_out = 0;
	*size_out      = 0;

	if (search_start < 0 || search_end < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	if (search_end <= search_start) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_ENOMEM;
	}

	// KYTY_DIRECT_MEMORY_MB: nothing past the size the game is told.
	search_end = std::min<int64_t>(search_end, static_cast<int64_t>(ReportedDirectMemorySize()));
	uint64_t phys_addr = 0;
	uint64_t size      = 0;
	bool     available = false;
	if (search_end > search_start) {
		// The state lock is not held across guest memory writes.
		MemoryStateReadLock memory_state_lock;
		available = g_physical_memory->Available(static_cast<uint64_t>(search_start),
		                                         static_cast<uint64_t>(search_end), alignment,
		                                         &phys_addr, &size);
	}
	if (!available) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_ENOMEM;
	}

	*phys_addr_out = static_cast<int64_t>(phys_addr);
	*size_out      = static_cast<size_t>(size);
	if (Profiler::LoadingEnabled()) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::DirectMemoryAvailableSuccesses);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::DirectMemoryAvailableBytesSum, size);
	}

	LOGF_COLOR(Log::Color::Green,
	           "\t phys_addr = 0x%016" PRIx64 "\n"
	           "\t size      = 0x%016" PRIx64 "\n"
	           "\t[Ok]\n",
	           phys_addr, size);

	return OK;
}

int KYTY_SYSV_ABI KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                          int* gpu_available) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableStatsQueries);

	if (cpu_total == nullptr || cpu_available == nullptr || gpu_total == nullptr ||
	    gpu_available == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	// The state lock is not held across guest memory writes.
	uint64_t cpu_used = 0;
	uint64_t gpu_used = 0;
	if (g_virtual_ranges != nullptr) {
		MemoryStateReadLock memory_state_lock;
		cpu_used = g_virtual_ranges->CountPageTableEntries(false);
		gpu_used = g_virtual_ranges->CountPageTableEntries(true);
	}

	const uint64_t cpu_entries = g_cpu_page_table_entries;
	const uint64_t gpu_entries = g_gpu_page_table_entries;
	*cpu_total                 = static_cast<int>(cpu_entries);
	*gpu_total                 = static_cast<int>(gpu_entries);
	*cpu_available = static_cast<int>(cpu_entries - std::min<uint64_t>(cpu_used, cpu_entries));
	*gpu_available = static_cast<int>(gpu_entries - std::min<uint64_t>(gpu_used, gpu_entries));
	if (Profiler::LoadingEnabled()) {
		const auto cpu_free = cpu_entries - std::min<uint64_t>(cpu_used, cpu_entries);
		const auto gpu_free = gpu_entries - std::min<uint64_t>(gpu_used, gpu_entries);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableStatsSuccesses);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableCpuAvailableSum, cpu_free);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableGpuAvailableSum, gpu_free);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableCpuZeroAvailable, cpu_free == 0);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::PageTableGpuZeroAvailable, gpu_free == 0);
	}

	LOGF_COLOR(Log::Color::Green,
	           "\t cpu_total     = %d\n"
	           "\t cpu_available = %d\n"
	           "\t gpu_total     = %d\n"
	           "\t gpu_available = %d\n"
	           "\t[Ok]\n",
	           *cpu_total, *cpu_available, *gpu_total, *gpu_available);

	return OK;
}

int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size) {
	PRINT_NAME();

	LOGF("\t offset    = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     offset, flags, info_size);

	struct QueryInfo {
		int64_t start;
		int64_t end;
		int     memory_type;
	};

	if (offset < 0 || (flags != 0 && flags != 1) || info_size != sizeof(QueryInfo) ||
	    info == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	auto* query_info = static_cast<QueryInfo*>(info);

	// The state lock is not held across guest memory writes (a fault there may need an operation).
	enum class Answer { Found, Terminal, Missing };
	auto                           answer = Answer::Found;
	PhysicalMemory::AllocatedBlock block {};
	{
		MemoryStateReadLock memory_state_lock;
		Common::LockGuard   lock(g_physical_memory->GetMutex());
		const auto&         blocks  = g_physical_memory->GetPhysicalBlocks();
		auto              current = blocks.upper_bound(static_cast<uint64_t>(offset));
		if (current != blocks.begin()) {
			auto previous = std::prev(current);
			if (static_cast<uint64_t>(offset) <
			    previous->second.start_addr + previous->second.size) {
				current = previous;
			}
		}
		// KYTY_DMEM_OVERFLOW_MB: a walk of the direct memory (flags 1) from inside it ends at its
		// end; the blocks given room past it answer only queries of their own addresses.
		const bool past_end = flags == 1 && current != blocks.end() &&
		                      static_cast<uint64_t>(offset) < PhysicalMemory::Size() &&
		                      current->second.start_addr >= PhysicalMemory::Size();
		if (current == blocks.end() || past_end ||
		    (flags == 0 && (static_cast<uint64_t>(offset) < current->second.start_addr ||
		                    static_cast<uint64_t>(offset) >=
		                        current->second.start_addr + current->second.size))) {
			answer = flags == 1 && static_cast<uint64_t>(offset) < PhysicalMemory::Size()
			             ? Answer::Terminal
			             : Answer::Missing;
		} else {
			block        = current->second;
			uint64_t end = block.start_addr + block.size;
			for (auto following = std::next(current);
			     following != blocks.end() && following->second.start_addr == end &&
			     following->second.memory_type == block.memory_type;
			     ++following) {
				end += following->second.size;
			}
			block.size = end - block.start_addr;
		}
	}
	if (answer == Answer::Terminal) {
		query_info->start       = static_cast<int64_t>(PhysicalMemory::Size());
		query_info->end         = static_cast<int64_t>(PhysicalMemory::Size());
		query_info->memory_type = 0;
		LOGF_COLOR(Log::Color::Green, "\t terminal    = true\n\t[Ok]\n");
		return OK;
	}
	if (answer == Answer::Missing) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_EACCES;
	}

	query_info->start       = static_cast<int64_t>(block.start_addr);
	query_info->end         = static_cast<int64_t>(block.start_addr + block.size);
	query_info->memory_type = block.memory_type;

	LOGF_COLOR(Log::Color::Green,
	           "\t start       = %016" PRIx64 "\n"
	           "\t end         = %016" PRIx64 "\n"
	           "\t memory_type = %d\n"
	           "\t[Ok]\n",
	           query_info->start, query_info->end, query_info->memory_type);

	return OK;
}

int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t memory_type  = %d\n",
	     search_start, search_end, len, alignment, memory_type);

	constexpr uint64_t PAGE_SIZE = 0x4000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (PAGE_SIZE - 1u)) != 0 || (alignment != 0 && (alignment & (PAGE_SIZE - 1u)) != 0) ||
	    phys_addr_out == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t addr = 0;
	if (!g_physical_memory->Alloc(search_start, search_end, len, alignment, &addr, memory_type)) {
		LOGF_COLOR(Log::Color::Red, "\t[Fail]\n");
		return KERNEL_ERROR_EAGAIN;
	}

	*phys_addr_out = static_cast<int64_t>(addr);

	LOGF_COLOR(Log::Color::Green, "\tphys_addr    = %016" PRIx64 "\n\t[Ok]\n", addr);

	return OK;
}

int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t memory_type  = %d\n",
	     len, alignment, memory_type);

	return KernelAllocateDirectMemory(0, static_cast<int64_t>(ReportedDirectMemorySize()), len,
	                                  alignment, memory_type, phys_addr_out);
}

static int ReleaseDirectMemoryInternal(int64_t start, size_t len) {
	MemoryOperationLock memory_operation_lock;

	if (g_pooled_memory->ReleaseExpansion(static_cast<uint64_t>(start), len)) {
		if (!g_physical_memory->ReleasePoolExpansion(static_cast<uint64_t>(start), len)) {
			EXIT("failed to release physical pool expansion\n");
		}
		return OK;
	}

	std::vector<PhysicalMemory::AllocatedBlock> allocated_span;
	if (!g_physical_memory->GetAllocatedSpan(static_cast<uint64_t>(start), len, &allocated_span)) {
		return KERNEL_ERROR_EACCES;
	}

	const auto mapped_aliases = g_physical_memory->FindMappings(start, len);
	for (const auto& alias: mapped_aliases) {
		VirtualRanges::Range range {};
		if (!g_guest_address_space->BackingContains(alias.map_vaddr, alias.map_size) ||
		    !g_virtual_ranges->Query(alias.map_vaddr, 0, &range) ||
		    range.type != VirtualRangeType::Direct ||
		    alias.map_size > range.start + range.size - alias.map_vaddr) {
			EXIT("direct-memory alias escaped guest address-space ownership\n");
		}
	}

	// Validation precedes the first unmap; failing after it is an emulator fault (upstream
	// ef0614642: no rollback of the aliases).
	std::vector<VirtualRanges::Range> aliases;
	aliases.reserve(mapped_aliases.size());
	for (const auto& alias: mapped_aliases) {
		VirtualRanges::Range range {};
		range.start  = alias.map_vaddr;
		range.size   = alias.map_size;
		range.offset = alias.start_addr;
		range.type   = VirtualRangeType::Direct;
		aliases.push_back(range);
	}
	// Adjacent aliases are unmapped from the GPU with one call (UnmapGpuRanges, upstream d384084e3).
	std::sort(aliases.begin(), aliases.end(),
	          [](const auto& left, const auto& right) { return left.start < right.start; });
	UnmapGpuRanges(aliases);
	for (const auto& alias: aliases) {
		UnmapMemoryOwner(alias);
	}

	for (const auto& block: allocated_span) {
		uint64_t      unused_vaddr = 0;
		uint64_t      unused_size  = 0;
		GpuAccessMode unused_gpu   = GpuAccessMode::NoAccess;
		EXIT_IF(!g_physical_memory->Release(block.start_addr, block.size, &unused_vaddr,
		                                    &unused_size, &unused_gpu));
	}

	// KYTY_DROP_RELEASED=1 (default off; GT7 fork 7717116d8): the released range's host pages go
	// back to the system; its views are unmapped above, and a later allocation gets undefined bytes (as
	// on the console: zeros or the old ones). GT7 peaked
	// at 9.8 GB of direct memory while using 3.5 GB, and the rest stayed resident.
	static const bool drop_released = [] {
		const char* value = std::getenv("KYTY_DROP_RELEASED");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	if (drop_released) {
		static std::atomic<uint64_t> dropped_total {0};
		uint64_t                     dropped = 0;
		for (const auto& block: allocated_span) {
			dropped += g_guest_address_space->DropBacking(block.start_addr, block.size);
		}
		const auto total = dropped_total.fetch_add(dropped) + dropped;
		if (dropped >= (uint64_t {64} << 20u)) {
			LOGF("Direct memory: %" PRIu64 " MiB of released memory handed back to the system (%" PRIu64
			     " MiB so far, KYTY_DROP_RELEASED)\n",
			     dropped >> 20u, total >> 20u);
		}
	}

	return OK;
}

static int ValidateDirectReleaseRange(int64_t start, size_t len) {
	constexpr uint64_t PAGE_SIZE = 0x4000;
	return start < 0 || (static_cast<uint64_t>(start) & (PAGE_SIZE - 1u)) != 0 ||
	               (len & (PAGE_SIZE - 1u)) != 0
	           ? KERNEL_ERROR_EINVAL
	           : OK;
}

// KYTY_RELEASE_GAPS=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 bdab60022): the
// unchecked KernelReleaseDirectMemory over a range that is not one allocated span (a gap, or a part
// already released) releases each allocated run in it, as the console does; off, it releases
// nothing and the memory stays allocated (GT7's pools then ran out of direct memory).
static bool ReleaseGapsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_RELEASE_GAPS");
		return value != nullptr && !(value[0] == '0' && value[1] == '\0');
	}();
	return enabled;
}

// The allocated parts of [start, start + len) (pool expansions left out), as runs of adjacent
// blocks.
static std::vector<std::pair<uint64_t, uint64_t>> AllocatedRuns(uint64_t start, uint64_t len) {
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	if (len == 0 || len > UINT64_MAX - start) {
		return runs;
	}
	const auto        end = start + len;
	Common::LockGuard lock(g_physical_memory->GetMutex());
	const auto&       blocks = g_physical_memory->GetPhysicalBlocks();
	auto              block  = blocks.upper_bound(start);
	if (block != blocks.begin()) {
		--block;
	}
	for (; block != blocks.end() && block->second.start_addr < end; ++block) {
		const auto& b     = block->second;
		const auto  b_end = b.start_addr + b.size;
		if (b_end <= start || b.pool_expansion) {
			continue;
		}
		const auto part_start = std::max(start, b.start_addr);
		const auto part_end   = std::min(end, b_end);
		if (!runs.empty() && runs.back().first + runs.back().second == part_start) {
			runs.back().second += part_end - part_start;
		} else {
			runs.emplace_back(part_start, part_end - part_start);
		}
	}
	return runs;
}

int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     start, len);

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK) {
		return validation;
	}
	if (len != 0) {
		MemoryOperationLock memory_operation_lock;
		std::vector<PhysicalMemory::AllocatedBlock> span;
		if (ReleaseDirectMemoryInternal(start, len) != OK && ReleaseGapsEnabled() &&
		    !g_physical_memory->GetAllocatedSpan(static_cast<uint64_t>(start), len, &span)) {
			// Not one allocated span (a failed unmap of an alias is left as it was).
			uint64_t freed = 0;
			uint32_t parts = 0;
			for (const auto& [part_start, part_size]: AllocatedRuns(static_cast<uint64_t>(start), len)) {
				if (ReleaseDirectMemoryInternal(static_cast<int64_t>(part_start), part_size) == OK) {
					freed += part_size;
					parts++;
				}
			}
			static std::atomic<uint64_t> partial_releases {0};
			const auto                   count = partial_releases.fetch_add(1) + 1u;
			if (count <= 32u || std::has_single_bit(count)) {
				LOGF("ReleaseDirectMemory: 0x%016" PRIx64 "+0x%" PRIx64 " is not one allocated span: "
				     "its %u allocated parts were released (0x%" PRIx64 " bytes; %" PRIu64
				     " such releases so far)\n",
				     static_cast<uint64_t>(start), static_cast<uint64_t>(len), parts, freed, count);
			}
		}
	}
	return OK;
}

int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len) {
	PRINT_NAME();

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n",
	     start, len);

	const int validation = ValidateDirectReleaseRange(start, len);
	if (validation != OK || len == 0) {
		return validation;
	}

	const int result = ReleaseDirectMemoryInternal(start, len);
	return result == KERNEL_ERROR_EACCES ? KERNEL_ERROR_ENOENT : result;
}

// The ABI entry points and batch operations hold g_memory_operation_mutex.
static int MapDirectMemory(void** addr, size_t len, int prot, int flags,
                           int64_t direct_memory_start, size_t alignment, const char* name,
                           int type = -1) {
	if (addr == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}
	constexpr uint64_t PAGE_SIZE              = 0x4000;
	constexpr int      GUEST_MAP_FIXED        = 0x10;
	constexpr int      GUEST_MAP_NO_OVERWRITE = 0x80;

	if (len == 0 || (len & (PAGE_SIZE - 1u)) != 0 || direct_memory_start < 0 ||
	    (static_cast<uint64_t>(direct_memory_start) & (PAGE_SIZE - 1u)) != 0 ||
	    (alignment != 0 && (alignment & (alignment - 1u)) != 0 &&
	     (alignment & (PAGE_SIZE - 1u)) != 0)) {
		return KERNEL_ERROR_EINVAL;
	}
	if ((prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EACCES;
	}

	bool fixed        = ((flags & GUEST_MAP_FIXED) != 0);
	bool no_overwrite = ((flags & GUEST_MAP_NO_OVERWRITE) != 0);

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (!g_physical_memory->CanMapDirect(static_cast<uint64_t>(direct_memory_start), len)) {
		return KERNEL_ERROR_ENOMEM;
	}

	// KYTY_TRANSIENT_UNMAP_RETRY: guest threads may use a mapping that MAP_FIXED replaces: until
	// the new one is in place, their faults wait and are retried.
	std::optional<TransientUnmapWindows::Scope> replacing;

	auto     in_addr        = reinterpret_cast<uint64_t>(*addr);
	uint64_t out_addr       = 0;
	bool     replace_target = false;
	auto     replaced_type  = VirtualRangeType::Reserved;
	auto     shared_failure = GuestBackingStore::FailureReason::None;
	// Direct mappings must remain views of the single backing object. Anonymous fallbacks break
	// aliasing and lose direct-memory contents when a range is unmapped and mapped again.
	auto map_shared_fixed = [&](uint64_t target_addr) -> bool {
		return g_guest_address_space->MapBacking(target_addr, len, direct_memory_start, mode,
		                                         &shared_failure);
	};
	// Upstream 466551e64/dcdef60bd: the target is validated up front; failing to replace it after
	// that is an emulator fault instead of a restore of the old views.
	std::vector<PhysicalMemory::AllocatedBlock> replaced_mappings;
	if (fixed) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1u)) != 0 ||
		    (alignment != 0 && in_addr % alignment != 0)) {
			return KERNEL_ERROR_EINVAL;
		}
		if (no_overwrite && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		std::vector<VirtualRanges::Range> reserved_ranges;
		if (g_virtual_ranges->QuerySpan(in_addr, len, &reserved_ranges) &&
		    std::all_of(reserved_ranges.begin(), reserved_ranges.end(), [](const auto& range) {
			    return range.type == VirtualRangeType::Direct;
		    })) {
			uint64_t covered = 0;
			{
				Common::LockGuard lock(g_physical_memory->GetMutex());
				for (const auto& mapping: g_physical_memory->GetMappings()) {
					const auto start = std::max(in_addr, mapping.map_vaddr);
					const auto end = std::min(in_addr + len, mapping.map_vaddr + mapping.map_size);
					if (start >= end) {
						continue;
					}
					auto old = mapping;
					old.start_addr += start - old.map_vaddr;
					old.map_vaddr = start;
					old.size = old.map_size = end - start;
					covered += old.map_size;
					replaced_mappings.push_back(old);
				}
			}
			if (covered != len) {
				return KERNEL_ERROR_EBUSY;
			}
			UnmapGpuRange(in_addr, len);
			for (const auto& old: replaced_mappings) {
				GpuAccessMode old_gpu_mode = GpuAccessMode::NoAccess;
				EXIT_IF(!g_physical_memory->Unmap(old.map_vaddr, old.map_size, &old_gpu_mode));
			}
			// Keep the old view accessible until mmap atomically replaces it. In particular,
			// do not publish a temporary reserved/PROT_NONE range between two direct views.
			EXIT_IF(!map_shared_fixed(in_addr));
			replaced_type  = VirtualRangeType::Direct;
			replace_target = true;
			out_addr       = in_addr;
		} else
#endif
		{
			if (!IsReservedSpan(in_addr, len)) {
				if (TransientUnmapRetryEnabled()) {
					replacing.emplace(in_addr, in_addr + len);
				}
				const int result = ReplaceFixedRangeWithReserved(in_addr, len);
				if (result != OK) {
					return result;
				}
			}
			UnmapGpuRange(in_addr, len);
			replace_target = true;
			if (map_shared_fixed(in_addr)) {
				out_addr = in_addr;
			}
		}
	} else {
		constexpr size_t DEFAULT_ALIGNMENT = 0x4000;
		alignment                          = (alignment != 0 ? alignment : DEFAULT_ALIGNMENT);
		if (in_addr != 0 && IsReservedSpan(in_addr, len)) {
			UnmapGpuRange(in_addr, len);
			replace_target = true;
			if (map_shared_fixed(in_addr)) {
				out_addr = in_addr;
			}
		}
		if (!replace_target) {
			out_addr = FindGuestFreeRange(in_addr, len, alignment);
			if (out_addr != 0) {
				UnmapGpuRange(out_addr, len);
				if (!map_shared_fixed(out_addr)) {
					out_addr = 0;
				}
			}
		}
	}

	*addr = reinterpret_cast<void*>(out_addr);

	LOGF("\t in_addr  = 0x%016" PRIx64 "\n"
	     "\t out_addr = 0x%016" PRIx64 "\n"
	     "\t dmem     = 0x%016" PRIx64 "\n"
	     "\t size     = 0x%016" PRIx64 "\n"
	     "\t mode     = %s\n"
	     "\t flags    = 0x%08" PRIx32 "\n"
	     "\t align    = 0x%016" PRIx64 "\n"
	     "\t gpu_mode = %s\n"
	     "\t reason   = %s\n",
	     in_addr, out_addr, static_cast<uint64_t>(direct_memory_start), len,
	     magic_enum::enum_name(mode), static_cast<uint32_t>(flags), alignment,
	     magic_enum::enum_name(gpu_mode), GuestBackingStore::GetFailureReasonName(shared_failure));

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	PhysicalMemory::AllocatedBlock mapped_block {};
	g_physical_memory->Find(direct_memory_start, false, &mapped_block);
	const int memory_type = type == -1 ? mapped_block.memory_type : type;
	// CanMapDirect and the target were validated under the operation lock.
	EXIT_IF(!g_physical_memory->Map(out_addr, direct_memory_start, len, prot, mode, gpu_mode, name,
	                                memory_type));
	const bool published =
	    replace_target
	        ? g_virtual_ranges->ReplaceSpan(out_addr, len, replaced_type, direct_memory_start, prot,
	                                        memory_type, VirtualRangeType::Direct, name)
	        : g_virtual_ranges->Add(out_addr, len, direct_memory_start, prot, memory_type,
	                                VirtualRangeType::Direct, name);
	EXIT_IF(!published);

	MapGpuRange(out_addr, len);
	Libs::Graphics::WatchRange::Note("map direct memory", out_addr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(out_addr, len);
	}
	// The replaced views stay accessible until here, so their callbacks follow the new one.
	if (g_free_callback != nullptr && replaced_type == VirtualRangeType::Direct) {
		for (const auto& old: replaced_mappings) {
			g_free_callback(old.map_vaddr, old.map_size);
		}
	}

	LOGF_COLOR(Log::Color::Green, "\t [Ok]\n");

	return OK;
}

int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return MapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment, "");
}

int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return MapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment, "", type);
}

int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t name = %s\n", name != nullptr ? name : "(null)");

	if (name == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (std::strlen(name) >= KERNEL_MAXIMUM_NAME_LENGTH) {
		return KERNEL_ERROR_ENAMETOOLONG;
	}

	return MapDirectMemory(addr, len, prot, flags, direct_memory_start, alignment, name);
}

int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled() {
	PRINT_NAME();

	return 0;
}

int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot) {
	PRINT_NAME();

	EXIT_NOT_IMPLEMENTED(addr == nullptr);

	// The state lock is not held across guest memory writes (a fault there may need an operation).
	VirtualRanges::Range range {};
	{
		MemoryStateReadLock memory_state_lock;
		if (!g_virtual_ranges->Query(reinterpret_cast<uint64_t>(addr), 0, &range)) {
			return KERNEL_ERROR_EACCES;
		}
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(range.start);
	}
	if (end != nullptr) {
		*end = reinterpret_cast<void*>(range.start + range.size);
	}
	if (prot != nullptr) {
		*prot = range.protection;
	}

	return OK;
}

// Validation precedes the first unmap; failing after it is an emulator fault (upstream 466551e64:
// validated up front instead of rolled back).
static int ReplaceFixedRangeWithReserved(uint64_t start, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	g_virtual_ranges->QueryFragments(start, size, &ranges);
	if (!g_guest_address_space->Owns(start, size) ||
	    std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return range.type == VirtualRangeType::Pooled; })) {
		return KERNEL_ERROR_EINVAL;
	}
	UnmapGpuRanges(ranges);
	for (const auto& range: ranges) {
		UnmapMemoryOwner(range);
	}
	EXIT_IF(!g_virtual_ranges->Add(start, size, 0, 0, 0, VirtualRangeType::Reserved, "anon"));
	return OK;
}

int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	const auto in_addr = (addr != nullptr ? reinterpret_cast<uint64_t>(*addr) : 0);

	LOGF("\t in_addr   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n",
	     in_addr, len, flags, alignment);

	constexpr size_t PAGE_SIZE              = 0x4000;
	constexpr int    GUEST_MAP_FIXED        = 0x10;
	constexpr int    GUEST_MAP_NO_OVERWRITE = 0x80;

	if (addr == nullptr || len == 0 || (len & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 && (alignment & (alignment - 1)) != 0 &&
	    (alignment & (PAGE_SIZE - 1)) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t out_addr            = 0;
	bool     range_already_added = false;
	if ((flags & GUEST_MAP_FIXED) != 0) {
		if (in_addr == 0 || (in_addr & (PAGE_SIZE - 1)) != 0) {
			return KERNEL_ERROR_EINVAL;
		}
		if ((flags & GUEST_MAP_NO_OVERWRITE) != 0 && g_virtual_ranges->HasOverlap(in_addr, len)) {
			return KERNEL_ERROR_ENOMEM;
		}
		const int result = ReplaceFixedRangeWithReserved(in_addr, len);
		if (result != OK) {
			return result;
		}
		out_addr            = in_addr;
		range_already_added = true;
	} else {
		alignment = (alignment != 0 ? alignment : PAGE_SIZE);
		out_addr  = FindGuestFreeRange(in_addr, len, alignment);
		if (out_addr != 0) {
			UnmapGpuRange(out_addr, len);
		}
	}

	if (out_addr == 0) {
		return KERNEL_ERROR_ENOMEM;
	}

	if (!range_already_added &&
	    !g_virtual_ranges->Add(out_addr, len, 0, 0, 0, VirtualRangeType::Reserved, "anon")) {
		return KERNEL_ERROR_EBUSY;
	}

	*addr = reinterpret_cast<void*>(out_addr);

	LOGF("\t out_addr  = 0x%016" PRIx64 "\n", out_addr);

	return OK;
}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
void TestSetClampRangeBitmap(bool enabled) {
	g_clamp_range_bitmap.store(enabled ? 1 : 0, std::memory_order_relaxed);
}

TestClampTotals TestClampRangeMemoTotals() {
	const auto& totals = t_clamp_totals;
	return {totals.hits, totals.misses, totals.verify_checks, totals.verify_mismatches,
	        totals.verify_races};
}

void TestSetGpuRangeCallback(void (*callback)(bool, uint64_t, size_t)) {
	g_test_gpu_range = callback;
}

void TestBeforeNextBackingMap(callback_func_t callback) {
	g_test_before_backing_map = callback;
}

void TestSetBackingReadCallback(callback_func_t callback) {
	g_test_backing_read = callback;
}

bool TestTryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size) {
	return g_guest_address_space->TryReadSparseBacking(vaddr, data, size);
}

bool TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->TestContainsFree(vaddr, size);
}

bool TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size) {
	return g_guest_address_space->Owns(vaddr, size);
}

bool TestGuestBackingOutsideAddressSpace() {
	return !g_guest_address_space->OverlapsOwned(g_guest_address_space->GetBackingBase(),
	                                             g_guest_address_space->GetBackingSize());
}

uint64_t TestGuestBackingSize() {
	return g_guest_address_space->GetBackingSize();
}

uint64_t TestGuestBackingBase() {
	return g_guest_address_space->GetBackingBase();
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool TestWindowsBackingViewModes() {
	constexpr uint64_t size = 0x10000;
	GuestBackingStore backing(size);
	void* placeholder = VirtualAlloc2(GetCurrentProcess(), nullptr, size,
	                                  MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS,
	                                  nullptr, 0);
	if (placeholder == nullptr) {
		return false;
	}
	const auto address = reinterpret_cast<uint64_t>(placeholder);
	bool valid = true;
	for (auto [mode, protection]: std::array {
	         std::pair {VirtualMemory::Mode::NoAccess, DWORD {PAGE_NOACCESS}},
	         std::pair {VirtualMemory::Mode::Read, DWORD {PAGE_READONLY}},
	         std::pair {VirtualMemory::Mode::ReadWrite, DWORD {PAGE_READWRITE}},
	         std::pair {VirtualMemory::Mode::ExecuteReadWrite, DWORD {PAGE_EXECUTE_READWRITE}}}) {
		if (!backing.MapFixed(address, size, 0, mode)) {
			valid = false;
			break;
		}
		MEMORY_BASIC_INFORMATION info {};
		// A no-access view is mapped read-write and then protected (upstream 1b59ee62b's check).
		const DWORD map_protection =
		    protection == PAGE_NOACCESS ? DWORD {PAGE_READWRITE} : protection;
		valid = VirtualQuery(placeholder, &info, sizeof(info)) != 0 && info.Type == MEM_MAPPED &&
		        info.Protect == protection && info.AllocationProtect == map_protection && valid;
		EXIT_IF(!backing.Unmap(address, size));
	}
	return VirtualFree(placeholder, 0, MEM_RELEASE) != 0 && valid;
}
#endif

bool TestGuestFreeRangeBounds() {
	return GuestFreeRangeContains(0x10000, 0x20000, 0x18000, 0x4000) &&
	       !GuestFreeRangeContains(0x10000, 0x20000, 0x40000, 0x4000) &&
	       !GuestFreeRangeContains(UINT64_MAX - 0x1000, 0x2000, UINT64_MAX - 0x800, 0x400);
}
#endif

int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size) {
	PRINT_NAME();

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr      = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t info_size = 0x%016" PRIx64 "\n",
	     vaddr, flags, info_size);

	if (info == nullptr || info_size != sizeof(VirtualQueryInfo) || (flags != 0 && flags != 1)) {
		return KERNEL_ERROR_EINVAL;
	}

	// The state lock is not held across guest memory writes (a fault there may need an operation).
	VirtualRanges::Range candidate {};
	bool                 prt = false;
	{
		MemoryStateReadLock memory_state_lock;
		if (!g_virtual_ranges->Query(vaddr, flags, &candidate)) {
			return KERNEL_ERROR_EACCES;
		}
		prt = IsInPrtAperture(vaddr);
	}

	std::memset(info, 0, sizeof(VirtualQueryInfo));
	info->start        = candidate.start;
	info->end          = candidate.start + candidate.size;
	info->offset       = candidate.offset;
	info->protection   = candidate.protection;
	info->memory_type  = candidate.memory_type;
	info->is_flexible  = (candidate.type == VirtualRangeType::Flexible ? 1 : 0);
	info->is_direct    = (candidate.type == VirtualRangeType::Direct ? 1 : 0);
	info->is_stack     = (candidate.type == VirtualRangeType::Stack ? 1 : 0);
	info->is_pooled    = (IsPooledRangeType(candidate.type) ? 1 : 0);
	info->is_committed = (IsCommittedRangeType(candidate.type) ? 1 : 0);
	info->is_gpu_prt   = prt;
	CopyVirtualRangeName(info->name, candidate.name);

	static std::atomic<uint32_t> log_count {0};
	if (log_count.fetch_add(1) < 64) {
		LOGF("\t start       = 0x%016" PRIx64 "\n"
		     "\t end         = 0x%016" PRIx64 "\n"
		     "\t offset      = 0x%016" PRIx64 "\n"
		     "\t protection  = 0x%08" PRIx32 "\n"
		     "\t memory_type = %d\n"
		     "\t flexible    = %d\n"
		     "\t direct      = %d\n"
		     "\t name        = %s\n",
		     static_cast<uint64_t>(info->start), static_cast<uint64_t>(info->end), info->offset,
		     info->protection, info->memory_type, static_cast<int>(info->is_flexible),
		     static_cast<int>(info->is_direct), info->name);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end) {
	PRINT_NAME();

	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n", vaddr);

	VirtualRanges::Range candidate {};
	{
		MemoryStateReadLock memory_state_lock;
		if (!g_virtual_ranges->Query(vaddr, 0, &candidate)) {
			return KERNEL_ERROR_EACCES;
		}
	}

	uint64_t stack_start = 0;
	uint64_t stack_end   = 0;
	if (candidate.type == VirtualRangeType::Stack) {
		stack_start = candidate.start;
		stack_end   = candidate.start + candidate.size;
	}

	if (start != nullptr) {
		*start = reinterpret_cast<void*>(stack_start);
	}

	if (end != nullptr) {
		*end = reinterpret_cast<void*>(stack_end);
	}

	LOGF("\t start = 0x%016" PRIx64 "\n"
	     "\t end   = 0x%016" PRIx64 "\n",
	     stack_start, stack_end);

	return OK;
}

int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::FlexibleAvailableQueries);

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	uint64_t available = 0;
	{
		MemoryStateReadLock memory_state_lock;
		available = g_flexible_memory->Available();
	}
	*size = available;
	if (Profiler::LoadingEnabled()) {
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::FlexibleAvailableSuccesses);
		Profiler::CountLoadingEvent(Profiler::LoadingEvent::FlexibleAvailableBytesSum, available);
	}

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size) {
	PRINT_NAME();

	if (size == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	*size = FlexibleMemory::Size();

	LOGF("\t *size = 0x%016" PRIx64 "\n", *size);

	return OK;
}

static int ProgramProtection(VirtualMemory::Mode mode) {
	const auto protection = static_cast<int>(mode);
	if ((protection & ~(PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC)) != 0) {
		EXIT("unsupported program-memory protection: 0x%08x\n", protection);
	}
	return protection;
}

static std::vector<VirtualRanges::Range> RequireGuestRuntimeMemory(uint64_t vaddr, uint64_t size) {
	std::vector<VirtualRanges::Range> ranges;
	if (g_virtual_ranges == nullptr || !g_virtual_ranges->QuerySpan(vaddr, size, &ranges) ||
	    std::any_of(ranges.begin(), ranges.end(), [](const auto& range) {
		    return range.type != VirtualRangeType::Code && range.type != VirtualRangeType::Runtime;
	    })) {
		EXIT("guest runtime range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}
	return ranges;
}

static uint64_t AllocateGuestRuntimeMemory(uint64_t search_addr, uint64_t size,
                                           VirtualMemory::Mode mode, const char* name,
                                           VirtualRangeType type, bool fixed) {
	MemoryOperationLock memory_operation_lock;

	constexpr uint64_t GuestPageSize = 0x4000;
	if (size == 0 || size > UINT64_MAX - (GuestPageSize - 1u) || name == nullptr ||
	    (fixed && (search_addr == 0 || (search_addr & (GuestPageSize - 1u)) != 0))) {
		return 0;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto vaddr =
	    fixed ? search_addr : FindGuestFreeRange(search_addr, mapped_size, GuestPageSize);
	if (vaddr == 0 || g_virtual_ranges->HasOverlap(vaddr, mapped_size)) {
		return 0;
	}
	UnmapGpuRange(vaddr, mapped_size);
	if (!g_guest_address_space->Commit(vaddr, mapped_size, mode)) {
		return 0;
	}
	if (!g_virtual_ranges->Add(vaddr, mapped_size, 0, ProgramProtection(mode), 0, type, name)) {
		EXIT_IF(!g_guest_address_space->ReleaseCommitted(vaddr, mapped_size));
		return 0;
	}
	MapGpuRange(vaddr, mapped_size);
	return vaddr;
}

// Upstream 0a140911a, only with KYTY_UPSTREAM_MEMORY_DEFAULTS=1: loaded modules' segments use
// flexible memory without taking its backing. Returns the size charged (0 when off).
uint64_t ChargeFlexibleMemory(uint64_t size) {
	if (!UpstreamMemoryDefaults() || size == 0 || g_flexible_memory == nullptr) {
		return 0;
	}

	g_flexible_memory->Charge(size);
	return size;
}

void ReleaseFlexibleMemory(uint64_t size) {
	if (size != 0 && g_flexible_memory != nullptr) {
		g_flexible_memory->Release(size);
	}

}

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Code, false);
}

void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	MemoryOperationLock memory_operation_lock;

	const auto ranges = RequireGuestRuntimeMemory(vaddr, size);
	if (std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return range.type != VirtualRangeType::Code; })) {
		EXIT("program-memory range is not fully mapped: addr=0x%016" PRIx64 " size=0x%016" PRIx64
		     "\n",
		     vaddr, size);
	}

	const auto host_mode = VirtualMemory::IsExecute(mode) ? VirtualMemory::Mode::ExecuteReadWrite
	                                                      : VirtualMemory::Mode::ReadWrite;
	NoteGuestProtection(vaddr, size, host_mode);
	EXIT_IF(!g_guest_address_space->Protect(vaddr, size, host_mode));
	g_virtual_ranges->Protect(vaddr, size, ProgramProtection(mode));
}

uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                               const char* name, bool fixed) {
	return AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Runtime,
	                                  fixed);
}

uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size, VirtualMemory::Mode mode,
                                  const char* name) {
	const auto vaddr =
	    AllocateGuestRuntimeMemory(search_addr, size, mode, name, VirtualRangeType::Stack, false);
	if (vaddr != 0 && g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, size);
	}
	return vaddr;
}

bool ProtectGuestMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
                        VirtualMemory::Mode* old_mode) {
	MemoryOperationLock memory_operation_lock;
	constexpr uint64_t                    GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (vaddr & (GuestPageSize - 1u))) {
		return false;
	}
	const auto aligned_addr = vaddr & ~(GuestPageSize - 1u);
	const auto aligned_size =
	    (size + (vaddr - aligned_addr) + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	const auto ranges = RequireGuestRuntimeMemory(aligned_addr, aligned_size);
	if (old_mode != nullptr) {
		*old_mode = static_cast<VirtualMemory::Mode>(
		    ranges.front().protection & (PROT_CPU_READ | PROT_CPU_WRITE | PROT_CPU_EXEC));
	}
	NoteGuestProtection(aligned_addr, aligned_size, mode);
	if (!g_guest_address_space->Protect(aligned_addr, aligned_size, mode)) {
		return false;
	}
	g_virtual_ranges->Protect(aligned_addr, aligned_size, ProgramProtection(mode));
	return true;
}

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
	return g_guest_address_space != nullptr &&
	       g_guest_address_space->ProtectTransient(vaddr, size, mode);
}

void AddLiveThreadStack(uint64_t vaddr, uint64_t size) {
	if (TrackerSpareEnabled() && g_guest_address_space != nullptr) {
		g_guest_address_space->AddLiveStack(vaddr, size);
	}
}

void RemoveLiveThreadStack(uint64_t vaddr, uint64_t size) {
	// The tracker's protection of what it still watches there comes back (outside the
	// address-space lock: the tracker protects through it).
	if (TrackerSpareEnabled() && g_guest_address_space != nullptr &&
	    g_guest_address_space->RemoveLiveStack(vaddr, size)) {
		ResyncGpuRangeProtection(vaddr, size);
	}
}

bool FreeGuestMemory(uint64_t vaddr, uint64_t size) {
	MemoryOperationLock memory_operation_lock;

	constexpr uint64_t GuestPageSize = 0x4000;
	if (vaddr == 0 || size == 0 || size > UINT64_MAX - (GuestPageSize - 1u)) {
		return false;
	}
	const auto mapped_size = (size + GuestPageSize - 1u) & ~(GuestPageSize - 1u);
	(void)RequireGuestRuntimeMemory(vaddr, mapped_size);
	UnmapGpuRange(vaddr, mapped_size);
	return FreeGuestMemoryOwner(vaddr, mapped_size);
}

// The ABI entry points and batch operations hold g_memory_operation_mutex.
static int ProtectMemory(const void* addr, size_t len, int prot) {
	auto vaddr = reinterpret_cast<uint64_t>(addr);

	LOGF("\t addr = 0x%016" PRIx64 "\n"
	     "\t len  = 0x%016" PRIx64 "\n"
	     "\t prot = 0x%08" PRIx32 "\n",
	     vaddr, static_cast<uint64_t>(len), static_cast<uint32_t>(prot));

	constexpr uint64_t PAGE_SIZE    = 0x4000;
	auto               aligned_addr = vaddr & ~(PAGE_SIZE - 1);
	const auto         page_offset  = vaddr - aligned_addr;
	if (len > UINT64_MAX - (PAGE_SIZE - 1) - page_offset) {
		return KERNEL_ERROR_EINVAL;
	}
	auto aligned_len = (len + page_offset + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);

	if (aligned_len == 0) {
		return OK;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;

	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode) ||
	    !g_guest_address_space->Owns(aligned_addr, aligned_len)) {
		return KERNEL_ERROR_EINVAL;
	}
	// Upstream 2069acdf4: only the committed ranges of the span are protected; reserved ranges and
	// holes have nothing to protect (this exited before).
	std::vector<VirtualRanges::Range> ranges;
	g_virtual_ranges->QueryFragments(aligned_addr, aligned_len, &ranges);
	for (const auto& range: ranges) {
		if (!IsCommittedRangeType(range.type)) {
			continue;
		}
		NoteGuestProtection(range.start, range.size, mode);
		// KYTY_MPROTECT_TRACKED_ONCE: a writable protection of GPU memory in one step with the
		// tracker's (PageManager::ProtectGuestWritable), not protect-then-resync below.
		const bool tracked_once = (prot & PROT_CPU_WRITE) != 0 && g_mprotect_tracked_once.On() &&
		                          g_gpu_resources != nullptr && IsGpuAddressRange(range.start, range.size) &&
		                          RemapTrackerProtectionEnabled() && range.start % 0x1000u == 0 &&
		                          range.size % 0x1000u == 0;
		if (tracked_once) {
			GetGpuResources().ProtectGuestWritable(range.start, range.size, mode);
		} else if (!g_guest_address_space->Protect(range.start, range.size, mode)) {
			EXIT("host memory-protection update failed: addr=0x%016" PRIx64 " size=0x%016" PRIx64
			     " prot=0x%08x\n",
			     range.start, range.size, prot);
		}
		if (range.type == VirtualRangeType::Direct) {
			g_physical_memory->ProtectMapping(range.start, range.size, prot, mode, gpu_mode);
		} else if (range.type == VirtualRangeType::Flexible) {
			g_flexible_memory->Protect(range.start, range.size, prot, mode, gpu_mode);
		}
		g_virtual_ranges->Protect(range.start, range.size, prot);
		// A writable guest protection replaced the tracker's on watched pages: their CPU writes
		// would stop faulting.
		if ((prot & PROT_CPU_WRITE) != 0 && !tracked_once) {
			ResyncGpuRangeProtection(range.start, range.size);
		}
	}

	return OK;
}

int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	return ProtectMemory(addr, len, prot);
}

int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot) {
	PRINT_NAME();
	MemoryOperationLock memory_operation_lock;
	LOGF("\t type = %d\n", type);
	return ProtectMemory(addr, len, prot);
}

int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t entries         = %p\n"
	     "\t num_entries     = %d\n"
	     "\t num_entries_out = %p\n"
	     "\t flags           = 0x%08" PRIx32 "\n",
	     static_cast<void*>(entries), num_entries, static_cast<void*>(num_entries_out),
	     static_cast<uint32_t>(flags));

	enum MemoryOpTypes {
		MAP_OP_MAP_DIRECT   = 0,
		MAP_OP_UNMAP        = 1,
		MAP_OP_PROTECT      = 2,
		MAP_OP_MAP_FLEXIBLE = 3,
		MAP_OP_TYPE_PROTECT = 4,
	};

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	int processed = 0;

	for (int i = 0; i < num_entries; i++, processed++) {
		auto* entry = &entries[i];

		LOGF("\t [%d] start = %p, offset = 0x%016" PRIx64 ", length = 0x%016" PRIx64
		     ", prot = 0x%02" PRIx32 ", type = 0x%02" PRIx32 ", op = %d\n",
		     i, entry->start, entry->offset, entry->length,
		     static_cast<uint32_t>(entry->protection), static_cast<uint32_t>(entry->type),
		     entry->operation);

		if (entry->length == 0 || entry->operation < MAP_OP_MAP_DIRECT ||
		    entry->operation > MAP_OP_TYPE_PROTECT) {
			break;
		}

		int result = OK;
		switch (entry->operation) {
			case MAP_OP_MAP_DIRECT:
				result = MapDirectMemory(&entry->start, entry->length, entry->protection, flags,
				                         static_cast<int64_t>(entry->offset), 0, "anon");
				break;
			case MAP_OP_UNMAP:
				result = UnmapMemory(reinterpret_cast<uint64_t>(entry->start), entry->length);
				break;
			case MAP_OP_PROTECT:
			case MAP_OP_TYPE_PROTECT:
				result = ProtectMemory(entry->start, entry->length, entry->protection);
				break;
			case MAP_OP_MAP_FLEXIBLE:
				result = MapFlexibleMemory(&entry->start, entry->length, entry->protection, flags,
				                           "anon");
				break;
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			if (num_entries_out != nullptr) {
				*num_entries_out = processed;
			}
			return result;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	return (processed == num_entries ? OK : KERNEL_ERROR_EINVAL);
}

int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out) {
	constexpr int GUEST_MAP_FIXED = 0x10;
	return KernelBatchMap2(entries, num_entries, num_entries_out, GUEST_MAP_FIXED);
}

static bool IsAligned(uint64_t value, uint64_t alignment) {
	return alignment == 0 || (value & (alignment - 1u)) == 0;
}

static void MemoryPoolSubtractCommitted(uint64_t len) {
	auto current = g_memory_pool_committed.load(std::memory_order_relaxed);
	while (current != 0) {
		const auto next = (current > len ? current - len : 0);
		if (g_memory_pool_committed.compare_exchange_weak(current, next,
		                                                  std::memory_order_relaxed)) {
			return;
		}
	}
}

int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	constexpr uint64_t POOL_PAGE_SIZE = 0x10000;
	if (search_start < 0 || search_end <= search_start || len == 0 ||
	    (len & (POOL_PAGE_SIZE - 1u)) != 0 || phys_addr_out == nullptr ||
	    (alignment != 0 &&
	     (!std::has_single_bit(alignment) || (alignment & (POOL_PAGE_SIZE - 1u)) != 0))) {
		return KERNEL_ERROR_EINVAL;
	}
	if (static_cast<uint64_t>(search_end - search_start) < len) {
		return KERNEL_ERROR_ENOMEM;
	}

	const auto effective_alignment = (alignment != 0 ? alignment : POOL_PAGE_SIZE);
	uint64_t   phys_addr           = 0;
	if (!g_physical_memory->Alloc(static_cast<uint64_t>(search_start),
	                              static_cast<uint64_t>(search_end), len, effective_alignment,
	                              &phys_addr, 0, true)) {
		return KERNEL_ERROR_ENOMEM;
	}

	g_pooled_memory->Expand(phys_addr, len);
	*phys_addr_out = static_cast<int64_t>(phys_addr);

	LOGF("\t search_start = 0x%016" PRIx64 "\n"
	     "\t search_end   = 0x%016" PRIx64 "\n"
	     "\t len          = 0x%016" PRIx64 "\n"
	     "\t alignment    = 0x%016" PRIx64 "\n"
	     "\t phys_addr    = 0x%016" PRIx64 "\n",
	     search_start, search_end, static_cast<uint64_t>(len), static_cast<uint64_t>(alignment),
	     phys_addr);
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t addr_in   = 0x%016" PRIx64 "\n"
	     "\t len       = 0x%016" PRIx64 "\n"
	     "\t alignment = 0x%016" PRIx64 "\n"
	     "\t flags     = 0x%08" PRIx32 "\n"
	     "\t addr_out  = %p\n",
	     reinterpret_cast<uint64_t>(addr_in), static_cast<uint64_t>(len),
	     static_cast<uint64_t>(alignment), static_cast<uint32_t>(flags),
	     static_cast<void*>(addr_out));

	constexpr uint64_t POOL_RESERVE_ALIGNMENT = 0x200000;

	if (addr_out == nullptr || len == 0 || !IsAligned(len, POOL_RESERVE_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}
	if (alignment != 0 &&
	    (!std::has_single_bit(alignment) || !IsAligned(alignment, POOL_RESERVE_ALIGNMENT))) {
		return KERNEL_ERROR_EINVAL;
	}

	void*      out_addr          = addr_in;
	const auto reserve_alignment = (alignment != 0 ? alignment : POOL_RESERVE_ALIGNMENT);
	const int  ret = KernelReserveVirtualRange(&out_addr, len, flags, reserve_alignment);
	if (ret == OK) {
		// The reservation may merge with a neighboring one, so only its own span changes type
		// (upstream df9b8ae9f).
		EXIT_IF(!g_virtual_ranges->ReplaceSpan(reinterpret_cast<uint64_t>(out_addr), len,
		                                       VirtualRangeType::Reserved, 0, 0, 0,
		                                       VirtualRangeType::PoolReserved, "anon"));
		*addr_out = out_addr;
		LOGF("\t out_addr  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(out_addr));
	}

	return ret;
}

int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t type  = 0x%08" PRIx32 "\n"
	     "\t prot  = 0x%08" PRIx32 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len), static_cast<uint32_t>(type),
	     static_cast<uint32_t>(prot), static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;
	constexpr int      PROT_CPU_EXEC         = 0x04;

	// Upstream a8deda923: the address must be 64 KiB aligned too.
	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT) ||
	    !IsAligned(vaddr, POOL_COMMIT_ALIGNMENT) || (prot & PROT_CPU_EXEC) != 0) {
		return KERNEL_ERROR_EINVAL;
	}

	VirtualMemory::Mode mode     = VirtualMemory::Mode::NoAccess;
	GpuAccessMode       gpu_mode = GpuAccessMode::NoAccess;
	if (!DecodeMemoryProtection(prot, &mode, &gpu_mode)) {
		return KERNEL_ERROR_EINVAL;
	}

	// Upstream c78ecaed8: the span is validated up front (pool memory only, none of it committed)
	// instead of rolled back.
	std::vector<VirtualRanges::Range> ranges;
	if (!g_virtual_ranges->QuerySpan(vaddr, len, &ranges) ||
	    std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return !IsPooledRangeType(range.type); })) {
		return KERNEL_ERROR_EINVAL;
	}
	if (std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return range.type == VirtualRangeType::Pooled; })) {
		return KERNEL_ERROR_EBUSY;
	}

	std::vector<PooledMemory::Mapping> mappings;
	if (!g_pooled_memory->Allocate(vaddr, len, gpu_mode, &mappings)) {
		return KERNEL_ERROR_ENOMEM;
	}

	// Validation precedes the first backing map; failing after it is an emulator fault.
	for (const auto& mapping: mappings) {
		auto failure_reason = GuestBackingStore::FailureReason::None;
		if (!g_guest_address_space->MapBacking(mapping.vaddr, mapping.size, mapping.phys_addr, mode,
		                                       &failure_reason)) {
			EXIT("failed to map pooled memory: %s\n",
			     GuestBackingStore::GetFailureReasonName(failure_reason));
		}
	}
	for (const auto& range: ranges) {
		EXIT_IF(!g_virtual_ranges->ReplaceSpan(range.start, range.size,
		                                       VirtualRangeType::PoolReserved, 0, prot, type,
		                                       VirtualRangeType::Pooled, range.name));
	}

	MapGpuRange(vaddr, len);

	if (g_alloc_callback != nullptr) {
		g_alloc_callback(vaddr, len);
	}

	g_memory_pool_committed.fetch_add(len, std::memory_order_relaxed);

	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	LOGF("\t addr  = 0x%016" PRIx64 "\n"
	     "\t len   = 0x%016" PRIx64 "\n"
	     "\t flags = 0x%08" PRIx32 "\n",
	     reinterpret_cast<uint64_t>(addr), static_cast<uint64_t>(len),
	     static_cast<uint32_t>(flags));

	constexpr uint64_t POOL_COMMIT_ALIGNMENT = 0x10000;

	if (addr == nullptr || len == 0 || !IsAligned(len, POOL_COMMIT_ALIGNMENT)) {
		return KERNEL_ERROR_EINVAL;
	}

	// Upstream a8deda923: the address must be 64 KiB aligned too.
	const auto vaddr = reinterpret_cast<uint64_t>(addr);
	if (!IsAligned(vaddr, POOL_COMMIT_ALIGNMENT) || UINT64_MAX - vaddr < len) {
		return KERNEL_ERROR_EINVAL;
	}
	std::vector<VirtualRanges::Range> ranges;
	if (!g_virtual_ranges->QuerySpan(vaddr, len, &ranges) ||
	    std::any_of(ranges.begin(), ranges.end(),
	                [](const auto& range) { return !IsPooledRangeType(range.type); })) {
		return KERNEL_ERROR_EACCES;
	}

	// Upstream d384084e3/e2777436f: one GPU unmap per run of committed pieces; failing after the
	// validation is an emulator fault.
	UnmapGpuRanges(ranges);
	for (const auto& range: ranges) {
		if (range.type == VirtualRangeType::Pooled) {
			UnmapMemoryOwner(range, true);
		}
	}
	return OK;
}

int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags) {
	PRINT_NAME();

	MemoryOperationLock memory_operation_lock;

	if (entries == nullptr || num_entries < 0) {
		return KERNEL_ERROR_EINVAL;
	}

	enum MemoryPoolOp {
		POOL_OP_COMMIT       = 1,
		POOL_OP_DECOMMIT     = 2,
		POOL_OP_PROTECT      = 3,
		POOL_OP_TYPE_PROTECT = 4,
		POOL_OP_MOVE         = 5,
	};

	int processed = 0;
	int result    = OK;

	for (int i = 0; i < num_entries; i++, processed++) {
		const auto& entry = entries[i];
		switch (entry.op) {
			case POOL_OP_COMMIT:
				result = KernelMemoryPoolCommit(entry.commit.addr, entry.commit.len,
				                                entry.commit.type, entry.commit.prot, entry.flags);
				break;
			case POOL_OP_DECOMMIT:
				result =
				    KernelMemoryPoolDecommit(entry.decommit.addr, entry.decommit.len, entry.flags);
				break;
			case POOL_OP_PROTECT:
				result = KernelMprotect(entry.protect.addr, entry.protect.len, entry.protect.prot);
				break;
			case POOL_OP_TYPE_PROTECT:
				result = KernelMtypeprotect(entry.type_protect.addr, entry.type_protect.len,
				                            entry.type_protect.type, entry.type_protect.prot);
				break;
			case POOL_OP_MOVE:
			default: result = KERNEL_ERROR_EINVAL; break;
		}

		if (result != OK) {
			break;
		}
	}

	if (num_entries_out != nullptr) {
		*num_entries_out = processed;
	}

	(void)flags;
	return result;
}

int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size) {
	PRINT_NAME();
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::MemoryPoolStatsQueries);

	if (output == nullptr && output_size != 0) {
		return KERNEL_ERROR_EFAULT;
	}

	KernelMemoryPoolBlockStats stats {};
	constexpr uint64_t         BLOCK_SIZE = 0x10000;
	// Writers are serialized by the operation lock; readers take no lock (MemoryPoolSubtractCommitted).
	const uint64_t committed = g_memory_pool_committed.load(std::memory_order_relaxed);
	uint64_t       available = 0;
	if (g_pooled_memory != nullptr) {
		// The state lock is not held across guest memory writes.
		MemoryStateReadLock memory_state_lock;
		available = g_pooled_memory->Available();
	}
	Profiler::CountLoadingEvent(Profiler::LoadingEvent::MemoryPoolAvailableBytesSum, available);

	stats.available_flushed_blocks = static_cast<int32_t>(available / BLOCK_SIZE);
	stats.available_cached_blocks  = 0;
	stats.allocated_flushed_blocks = static_cast<int32_t>(committed / BLOCK_SIZE);
	stats.allocated_cached_blocks  = 0;

	const auto copy_size = std::min(output_size, sizeof(stats));
	if (copy_size != 0) {
		std::memcpy(output, &stats, copy_size);
	}

	return OK;
}

DebugStats GetDebugStats() {
	DebugStats stats;
	stats.direct_total       = PhysicalMemory::Size();
	stats.flexible_total     = FlexibleMemory::Size();
	stats.pool_committed     = g_memory_pool_committed.load(std::memory_order_relaxed);
	stats.cpu_page_entries_total = g_cpu_page_table_entries;
	stats.gpu_page_entries_total = g_gpu_page_table_entries;
	if (g_physical_memory != nullptr) {
		Common::LockGuard lock(g_physical_memory->GetMutex());
		for (const auto& [start, block]: g_physical_memory->GetPhysicalBlocks()) {
			// This line's PhysicalMemory does not record pooled/automatic kinds (performance panel).
			stats.direct_allocated += block.size;
		}
		for (const auto& mapping: g_physical_memory->GetMappings()) {
			stats.direct_mapped += mapping.map_size;
		}
	}
	if (g_flexible_memory != nullptr) {
		stats.flexible_used = stats.flexible_total - g_flexible_memory->Available();
	}
	if (g_virtual_ranges != nullptr) {
		stats.cpu_page_entries = g_virtual_ranges->CountPageTableEntries(false);
		stats.gpu_page_entries = g_virtual_ranges->CountPageTableEntries(true);
	}
	return stats;
}

} // namespace Libs::LibKernel::Memory
