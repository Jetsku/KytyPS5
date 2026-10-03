#include "graphics/host_gpu/pageManager.h"

#include "common/alignment.h"
#include "common/profiler.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/parkingLock.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <unistd.h>
#endif

namespace Libs::Graphics {
namespace {

constexpr uint64_t PAGE_SIZE    = TRACKER_PAGE_SIZE;
constexpr uint64_t REGION_SIZE  = TRACKER_REGION_SIZE;
constexpr uint64_t ADDRESS_SIZE = TRACKER_ADDRESS_SIZE;
constexpr uint64_t REGION_COUNT = ADDRESS_SIZE / REGION_SIZE;

constexpr uint64_t REGION_PAGES = REGION_SIZE / PAGE_SIZE;

[[noreturn]] void FailFast(const char* reason = nullptr) noexcept {
	std::fputs("PageManager fail-fast: ", stderr);
	std::fputs(reason != nullptr ? reason : "invalid page state", stderr);
	std::fputc('\n', stderr);
	std::fflush(stderr);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TerminateProcess(GetCurrentProcess(), static_cast<UINT>(EXCEPTION_NONCONTINUABLE_EXCEPTION));
#endif
	std::_Exit(322);
}

[[noreturn]] void Fatal(const char* format, ...) {
	std::fputs("PageManager fatal: ", stderr);
	va_list args;
	va_start(args, format);
	std::vfprintf(stderr, format, args);
	va_end(args);
	std::fputc('\n', stderr);
	std::fflush(stderr);
	std::_Exit(322);
}

// The region lock. Deferred write-unprotects hold it across their host call, so waiters spin,
// then park (KYTY_TRACKER_LOCK_PARK); with parking off they retry back to back as before.
class SpinGuard final {
public:
	explicit SpinGuard(ParkingSpinLock& lock): m_lock(lock) {
		if (!m_lock.try_lock()) [[unlikely]] {
			(void)m_lock.LockContended(false);
		}
	}
	~SpinGuard() { m_lock.unlock(); }
	KYTY_CLASS_NO_COPY(SpinGuard);

private:
	ParkingSpinLock& m_lock;
};

void ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		Fatal("invalid range vaddr=0x%016" PRIx64 ", size=0x%016" PRIx64, vaddr, size);
	}
}

std::atomic<int> g_defer_mode_for_tests {-1};

struct DeferCounters {
	std::atomic<uint64_t> spans {0};
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> settled {0};
	std::atomic<uint64_t> overflows {0};
	std::atomic<uint64_t> verify_checks {0};
	std::atomic<uint64_t> verify_mismatches {0};
};
DeferCounters         g_defer;
std::atomic<uint32_t> g_verify_logged {0};

bool ReuseAppliedProtection() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PAGE_PROTECT_REUSE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

PageManager::DeferMode ReadDeferMode() {
	const auto* value = std::getenv("KYTY_DEFER_UNPROTECT");
	auto        mode  = PageManager::DeferMode::On;
	if (value != nullptr && std::strcmp(value, "0") == 0) {
		mode = PageManager::DeferMode::Off;
	} else if (value != nullptr && std::strcmp(value, "verify") == 0) {
		mode = PageManager::DeferMode::Verify;
	}
	std::printf("Kyty page protection: write-watcher releases %s (KYTY_DEFER_UNPROTECT)\n",
	            mode == PageManager::DeferMode::Off ? "under the caller's lock"
	            : mode == PageManager::DeferMode::Verify
	                ? "after the caller's lock, verified"
	                : "after the caller's lock");
	std::fflush(stdout);
	return mode;
}

} // namespace

PageManager::DeferMode PageManager::GetDeferMode() {
	const int forced = g_defer_mode_for_tests.load(std::memory_order_relaxed);
	if (forced >= 0) [[unlikely]] {
		return static_cast<DeferMode>(forced);
	}
	static const DeferMode mode = ReadDeferMode();
	return mode;
}

void PageManager::SetDeferModeForTests(DeferMode mode) {
	g_defer_mode_for_tests.store(static_cast<int>(mode), std::memory_order_relaxed);
}

PageManager::DeferStats PageManager::GetDeferStats() {
	DeferStats stats;
	stats.spans             = g_defer.spans.load(std::memory_order_relaxed);
	stats.calls             = g_defer.calls.load(std::memory_order_relaxed);
	stats.settled           = g_defer.settled.load(std::memory_order_relaxed);
	stats.overflows         = g_defer.overflows.load(std::memory_order_relaxed);
	stats.verify_checks     = g_defer.verify_checks.load(std::memory_order_relaxed);
	stats.verify_mismatches = g_defer.verify_mismatches.load(std::memory_order_relaxed);
	return stats;
}

struct PageManager::Impl {
	// Host protection levels, in increasing strictness.
	enum class Level : uint8_t { ReadWrite = 0, Read = 1, NoAccess = 2 };

	static Level ToLevel(Common::VirtualMemory::Mode mode) noexcept {
		switch (mode) {
			case Common::VirtualMemory::Mode::NoAccess: return Level::NoAccess;
			case Common::VirtualMemory::Mode::Read: return Level::Read;
			default: return Level::ReadWrite;
		}
	}
	static Common::VirtualMemory::Mode ToMode(Level level) noexcept {
		switch (level) {
			case Level::NoAccess: return Common::VirtualMemory::Mode::NoAccess;
			case Level::Read: return Common::VirtualMemory::Mode::Read;
			case Level::ReadWrite: break;
		}
		return Common::VirtualMemory::Mode::ReadWrite;
	}
	static const char* LevelName(int level) noexcept {
		switch (level) {
			case 0: return "read-write";
			case 1: return "read-only";
			case 2: return "no-access";
			default: return "unknown";
		}
	}

	struct PageState {
		uint8_t write_watchers  : 7 = 0;
		uint8_t access_watchers : 1 = 0;

		[[nodiscard]] Common::VirtualMemory::Mode Perms() const noexcept {
			if (access_watchers != 0) {
				return Common::VirtualMemory::Mode::NoAccess;
			}
			if (write_watchers != 0) {
				return Common::VirtualMemory::Mode::Read;
			}
			return Common::VirtualMemory::Mode::ReadWrite;
		}

		template <int delta, bool is_read>
		uint32_t AddDelta(uint64_t address) {
			static_assert(delta >= -1 && delta <= 1);
			if constexpr (is_read) {
				if constexpr (delta == 1) {
					if (access_watchers != 0) {
						Fatal("read-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++access_watchers;
				} else if constexpr (delta == -1) {
					if (access_watchers == 0) {
						Fatal("read-watcher underflow at 0x%016" PRIx64, address);
					}
					return --access_watchers;
				} else {
					return access_watchers;
				}
			} else {
				if constexpr (delta == 1) {
					if (write_watchers == 0x7f) {
						Fatal("write-watcher overflow at 0x%016" PRIx64, address);
					}
					return ++write_watchers;
				} else if constexpr (delta == -1) {
					if (write_watchers == 0) {
						Fatal("write-watcher underflow at 0x%016" PRIx64, address);
					}
					return --write_watchers;
				} else {
					return write_watchers;
				}
			}
		}
	};
	static_assert(sizeof(PageState) == 1);

	// Locks, always taken in this order: host_lock, then lock.
	//  - lock guards the watcher counts and `applied`. With deferral on it is never held across a
	//    host call, so releasing a watcher inside a scope (a write fault, under the memory
	//    tracker's region lock) never waits for another thread's VirtualProtect.
	//  - host_lock serializes the host calls of the region, and the decisions they rest on: every
	//    change that is not a deferred release (watches, read-watcher changes, synchronous
	//    releases) and every deferred update holds it from reading the counts until the host call
	//    is recorded. The only change that can happen meanwhile is a deferred release, which only
	//    loosens the counts. So at every host_lock release no page's host protection is looser
	//    than its counts, and the last update to run writes the latest state.
	struct Region {
		ParkingSpinLock                     host_lock;
		ParkingSpinLock                     lock;
		std::array<PageState, REGION_PAGES> pages;
		// The protection last requested from the host for each page, under both locks. ReadWrite
		// until a watcher first protects the page: what this class always assumed of unwatched pages.
		std::array<Level, REGION_PAGES> applied {};
	};

	// Host calls collected under `lock` and made after it is released (under host_lock).
	struct Run {
		uint16_t first = 0;
		uint16_t last  = 0;
		Level    level = Level::ReadWrite;
	};
	struct RunList {
		static constexpr size_t      Capacity = 64;
		std::array<Run, Capacity>    runs {};
		size_t                       count = 0;
		bool Add(size_t first, size_t last, Level level) noexcept {
			if (count == Capacity) {
				return false;
			}
			runs[count++] = {static_cast<uint16_t>(first), static_cast<uint16_t>(last), level};
			return true;
		}
	};

	// The calling thread's pending write-unprotects (DeferUnprotectScope).
	struct DeferredSpan {
		Impl*    impl   = nullptr;
		Region*  region = nullptr;
		uint64_t base   = 0;
		uint16_t first  = 0;
		uint16_t last   = 0;
	};
	static constexpr size_t DeferredSpanCapacity = 16;
	// A new span joins the previous one of the same region across a gap of up to this many pages:
	// applying scans the pages in between, and changes only those whose host state is stale.
	static constexpr size_t DeferredMergeGap = 64;
	struct DeferredBatch {
		uint32_t                                       depth = 0;
		uint32_t                                       count = 0;
		std::array<DeferredSpan, DeferredSpanCapacity> spans {};
	};
	static thread_local DeferredBatch t_deferred;

	Impl() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		if (info.dwPageSize != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32,
			      static_cast<uint32_t>(info.dwPageSize));
		}
#elif defined(__APPLE__)
		// Under Rosetta the host page size is 4 KB, matching TRACKER_PAGE_SIZE.
		if (static_cast<uint64_t>(getpagesize()) != PAGE_SIZE) {
			Fatal("unsupported host page size 0x%08" PRIx32, static_cast<uint32_t>(getpagesize()));
		}
#else
		const auto host_page_size = ::sysconf(_SC_PAGESIZE);
		if (host_page_size < 0 || static_cast<uint64_t>(host_page_size) != PAGE_SIZE) {
			Fatal("unsupported host page size %ld", static_cast<long>(host_page_size));
		}
#endif
		regions = std::make_unique<std::atomic<Region*>[]>(REGION_COUNT);
	}

	~Impl() {
		for (const auto& region: region_storage) {
			SpinGuard lock(region->lock);
			for (auto& page: region->pages) {
				if (page.write_watchers != 0 || page.access_watchers != 0) {
					FailFast("PageManager destroyed with live page state");
				}
			}
		}
	}

	Region* FindRegion(uint64_t vaddr) const noexcept {
		return vaddr < ADDRESS_SIZE ? regions[vaddr / REGION_SIZE].load(std::memory_order_acquire)
		                            : nullptr;
	}

	Region* GetOrCreateRegion(uint64_t vaddr) {
		const auto index = vaddr / REGION_SIZE;
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		std::lock_guard lock(region_mutex);
		if (auto* region = regions[index].load(std::memory_order_acquire); region != nullptr) {
			return region;
		}
		auto  region = std::make_unique<Region>();
		auto* ptr    = region.get();
		region_storage.push_back(std::move(region));
		regions[index].store(ptr, std::memory_order_release);
		return ptr;
	}

	void Protect(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode) noexcept {
		const bool unprotect = mode == Common::VirtualMemory::Mode::ReadWrite;
		MemoryStats::Count(unprotect ? MemoryStats::Counter::UnprotectCalls
		                             : MemoryStats::Counter::ProtectCalls);
		MemoryStats::Count(unprotect ? MemoryStats::Counter::UnprotectPages
		                             : MemoryStats::Counter::ProtectPages,
		                   size / PAGE_SIZE);
		const MemoryStats::ScopedTimer timer(MemoryStats::Counter::ProtectNs);
		if (!Libs::LibKernel::Memory::ProtectGuestHostMemory(vaddr, size, mode)) {
			Fatal("address-space protection failed at 0x%016" PRIx64 ", mode=0x%08" PRIx32, vaddr,
			      static_cast<uint32_t>(mode));
		}
	}

	// Caller holds both region locks: protects pages [first, last) and records what it requested.
	void ProtectRun(Region& region, uint64_t base_addr, size_t first, size_t last,
	                Common::VirtualMemory::Mode mode) noexcept {
		Protect(base_addr + first * PAGE_SIZE, (last - first) * PAGE_SIZE, mode);
		std::fill(region.applied.data() + first, region.applied.data() + last, ToLevel(mode));
		if (GetDeferMode() == DeferMode::Verify) {
			VerifyHostLocked(region, base_addr, first, last);
		}
	}

	// Caller holds host_lock and not `lock`: makes the collected host calls, then records them
	// under `lock` (and verifies [first, last) in verify mode).
	void ProtectRuns(Region& region, uint64_t base_addr, const RunList& runs, size_t first,
	                 size_t last) noexcept {
		for (size_t index = 0; index < runs.count; index++) {
			const auto& run = runs.runs[index];
			Protect(base_addr + size_t {run.first} * PAGE_SIZE,
			        static_cast<size_t>(run.last - run.first) * PAGE_SIZE, ToMode(run.level));
		}
		const bool verify = GetDeferMode() == DeferMode::Verify;
		if (runs.count == 0 && !verify) {
			return;
		}
		SpinGuard counts(region.lock);
		for (size_t index = 0; index < runs.count; index++) {
			const auto& run = runs.runs[index];
			std::fill(region.applied.data() + run.first, region.applied.data() + run.last,
			          run.level);
			if (verify) {
				VerifyHostLocked(region, base_addr, run.first, run.last);
			}
		}
		if (verify) {
			VerifyStrictLocked(region, base_addr, first, last);
		}
	}

	// Brings the host protection of pages [first, last) to what their counts ask, calling the
	// host only for the runs whose recorded protection differs (a run may bridge pages already at
	// its level). Takes both locks itself. Returns the host calls made.
	uint32_t ApplySpan(Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		SpinGuard host(region.host_lock);
		RunList   runs;
		uint32_t  calls = 0;
		{
			SpinGuard  counts(region.lock);
			bool       open      = false;
			size_t     run_begin = 0;
			size_t     run_end   = 0;
			Level      run_level = Level::ReadWrite;
			const auto flush     = [&] {
				if (open) {
					if (!runs.Add(run_begin, run_end, run_level)) {
						// More runs than the list holds: this one is made under `lock`.
						ProtectRun(region, base_addr, run_begin, run_end, ToMode(run_level));
					}
					calls++;
					open = false;
				}
			};
			for (size_t page = first; page < last; page++) {
				const auto desired = ToLevel(region.pages[page].Perms());
				if (desired != region.applied[page]) {
					if (open && run_level == desired) {
						run_end = page + 1;
					} else {
						flush();
						open      = true;
						run_begin = page;
						run_end   = page + 1;
						run_level = desired;
					}
				} else if (open && desired != run_level) {
					flush();
				}
			}
			flush();
		}
		ProtectRuns(region, base_addr, runs, first, last);
		return calls;
	}

	// Caller holds region.lock (not host_lock), inside a DeferUnprotectScope: releases one write
	// watcher of the selected pages and records the pages this makes looser for the scope's end.
	// Returns false, with the span in [lo, hi), when the thread's batch is full: the caller
	// applies it once it has released `lock`.
	template <bool masked>
	bool ReleaseDeferredLocked(Region& region, uint64_t base_addr, size_t first, size_t last,
	                           const RegionBits* mask, size_t& lo, size_t& hi) noexcept {
		lo = REGION_PAGES;
		hi = 0;
		for (size_t page_index = first; page_index < last; page_index++) {
			if constexpr (masked) {
				if (!mask->Get(page_index)) {
					continue;
				}
			}
			auto&      page      = region.pages[page_index];
			const auto old_perms = page.Perms();
			(void)page.AddDelta<-1, false>(base_addr + page_index * PAGE_SIZE);
			if (page.Perms() != old_perms) {
				lo = std::min(lo, page_index);
				hi = page_index + 1;
			}
		}
		if (lo >= hi) {
			return true;
		}
		g_defer.spans.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DeferredUnprotectSpans);
		return RecordDeferred(region, base_addr, lo, hi);
	}

	// Adds pages [first, last) of `region` to the calling thread's pending spans. False when the
	// batch is full.
	bool RecordDeferred(Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		auto& batch = t_deferred;
		if (batch.count != 0) {
			auto& previous = batch.spans[batch.count - 1];
			if (previous.region == &region && first <= size_t {previous.last} + DeferredMergeGap &&
			    last + DeferredMergeGap >= size_t {previous.first}) {
				previous.first = static_cast<uint16_t>(std::min<size_t>(previous.first, first));
				previous.last  = static_cast<uint16_t>(std::max<size_t>(previous.last, last));
				return true;
			}
		}
		if (batch.count == batch.spans.size()) {
			return false;
		}
		batch.spans[batch.count++] = {this, &region, base_addr, static_cast<uint16_t>(first),
		                              static_cast<uint16_t>(last)};
		return true;
	}

	static void CountApplied(uint32_t calls) noexcept {
		if (calls != 0) {
			g_defer.calls.fetch_add(calls, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DeferredUnprotectCalls, calls);
		} else {
			g_defer.settled.fetch_add(1, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DeferredUnprotectSettled);
		}
	}

	// End of the outermost scope: applies the calling thread's pending spans from the counts as
	// they are now.
	static void ApplyDeferredBatch() noexcept {
		auto& batch = t_deferred;
		if (batch.count == 0) {
			return;
		}
		const auto spans = batch.spans;
		const auto count = batch.count;
		batch.count      = 0;
		for (uint32_t index = 0; index < count; index++) {
			const auto& span = spans[index];
			CountApplied(span.impl->ApplySpan(*span.region, span.base, span.first, span.last));
		}
	}

	// KYTY_DEFER_UNPROTECT=verify. Callers hold region.lock.
	static void ReportMismatch(const char* kind, uint64_t address, int counts, int applied,
	                           int host) noexcept {
		g_defer.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ProtectVerifyMismatches);
		if (g_verify_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::fprintf(stderr,
			             "PageManager verify: %s at 0x%016" PRIx64
			             ": watchers ask %s, recorded %s, host %s\n",
			             kind, address, LevelName(counts), LevelName(applied), LevelName(host));
			std::fflush(stderr);
		}
	}
	static void CountCheck() noexcept {
		g_defer.verify_checks.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ProtectVerifyChecks);
	}

	// Calls visit(page_index, host_level) for every committed page of [first, last); the level is
	// -1 when the host protection is none of the three this class sets. Windows only.
	template <typename Visit>
	static void ForEachHostLevel(uint64_t base_addr, size_t first, size_t last, Visit&& visit) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const auto end = base_addr + last * PAGE_SIZE;
		for (auto address = base_addr + first * PAGE_SIZE; address < end;) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
				return;
			}
			const auto run_end =
			    std::min(end, reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize);
			if (info.State == MEM_COMMIT) {
				int level = -1;
				switch (info.Protect & 0xffu) {
					case PAGE_NOACCESS: level = 2; break;
					case PAGE_READONLY:
					case PAGE_EXECUTE_READ: level = 1; break;
					case PAGE_READWRITE:
					case PAGE_WRITECOPY:
					case PAGE_EXECUTE_READWRITE:
					case PAGE_EXECUTE_WRITECOPY: level = 0; break;
					default: break;
				}
				for (auto page = address; page < run_end; page += PAGE_SIZE) {
					visit(static_cast<size_t>((page - base_addr) / PAGE_SIZE), level);
				}
			}
			address = std::max(run_end, address + PAGE_SIZE);
		}
#else
		(void)base_addr;
		(void)first;
		(void)last;
		(void)visit;
#endif
	}

	// Verify checks: the caller holds both region locks (host_lock, so no host call of the region
	// is in flight). After a host call: the host holds what was requested.
	void VerifyHostLocked(Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		CountCheck();
		ForEachHostLevel(base_addr, first, last, [&](size_t page, int host) {
			const auto applied = static_cast<int>(region.applied[page]);
			if (host != applied) {
				ReportMismatch("host differs from the protection just set",
				               base_addr + page * PAGE_SIZE,
				               static_cast<int>(ToLevel(region.pages[page].Perms())), applied, host);
			}
		});
	}

	// After any change: no page is looser than its counts ask, neither as recorded nor on the
	// host (the exactness invariant; a pending release only leaves pages stricter).
	void VerifyStrictLocked(Region& region, uint64_t base_addr, size_t first, size_t last) noexcept {
		CountCheck();
		for (size_t page = first; page < last; page++) {
			const auto counts  = static_cast<int>(ToLevel(region.pages[page].Perms()));
			const auto applied = static_cast<int>(region.applied[page]);
			if (applied < counts) {
				ReportMismatch("recorded protection looser than its watchers",
				               base_addr + page * PAGE_SIZE, counts, applied, -1);
			}
		}
		ForEachHostLevel(base_addr, first, last, [&](size_t page, int host) {
			const auto counts = static_cast<int>(ToLevel(region.pages[page].Perms()));
			if (host < counts) {
				ReportMismatch("host protection looser than its watchers",
				               base_addr + page * PAGE_SIZE, counts,
				               static_cast<int>(region.applied[page]), host);
			}
		});
	}

	template <bool track, bool is_read, bool masked>
	void UpdateRegionWatchers(Region& region, uint64_t base_addr, size_t first, size_t last,
	                          const RegionBits* mask = nullptr) {
		const auto mode = GetDeferMode();
		if constexpr (!track && !is_read) {
			if (t_deferred.depth != 0 && mode != DeferMode::Off) {
				size_t lo = 0;
				size_t hi = 0;
				bool   recorded;
				{
					SpinGuard counts(region.lock);
					recorded = ReleaseDeferredLocked<masked>(region, base_addr, first, last, mask,
					                                         lo, hi);
				}
				if (!recorded) {
					// The thread's batch is full: apply now, as without deferral (the caller's own
					// lock is still held).
					g_defer.overflows.fetch_add(1, std::memory_order_relaxed);
					Profiler::CountFrameEvent(Profiler::FrameEvent::DeferredUnprotectOverflows);
					CountApplied(ApplySpan(region, base_addr, lo, hi));
				}
				return;
			}
		}
		SpinGuard host(region.host_lock);
		RunList   runs;
		{
			SpinGuard counts(region.lock);
			UpdateCountsLocked<track, is_read, masked>(region, base_addr, first, last, mask,
			                                           mode == DeferMode::Off ? nullptr : &runs);
		}
		// With deferral on, the host calls leave `lock`: a deferred release on another thread (the
		// only change possible meanwhile) never waits for them.
		ProtectRuns(region, base_addr, runs, first, last);
	}

	// Caller holds both locks: updates the counts of the selected pages and protects the pages
	// whose protection this changes. With `runs` the host calls are collected for after `lock`
	// (a full list falls back to calling here); without, they are made here, as always before.
	template <bool track, bool is_read, bool masked>
	void UpdateCountsLocked(Region& region, uint64_t base_addr, size_t first, size_t last,
	                        const RegionBits* mask, RunList* runs) {
		const bool reuse_applied = track && !is_read && ReuseAppliedProtection();
		auto      perms                 = region.pages[first].Perms();
		uint64_t  range_begin           = 0;
		uint64_t  range_bytes           = 0;
		uint64_t  potential_range_bytes = 0;

		const auto release_pending = [&] {
			if (range_bytes != 0) {
				const auto run_end = range_begin + range_bytes / PAGE_SIZE;
				if (runs == nullptr || !runs->Add(range_begin, run_end, ToLevel(perms))) {
					ProtectRun(region, base_addr, range_begin, run_end, perms);
				}
				range_bytes           = 0;
				potential_range_bytes = 0;
			}
		};

		for (size_t page_index = first; page_index < last; page_index++) {
			auto&      page    = region.pages[page_index];
			const auto address = base_addr + page_index * PAGE_SIZE;
			const bool update  = !masked || mask->Get(page_index);

			const auto old_perms = page.Perms();
			const auto new_count = update ? page.AddDelta<track ? 1 : -1, is_read>(address)
			                              : page.AddDelta<0, is_read>(address);
			const auto new_perms = page.Perms();

			if (new_perms != perms) [[unlikely]] {
				release_pending();
				perms = new_perms;
			} else if (range_bytes != 0) {
				potential_range_bytes += PAGE_SIZE;
			}

			if (!update) {
				continue;
			}

			const bool watcher_edge = (track && new_count == 1) || (!track && new_count == 0);
			if (watcher_edge && old_perms != new_perms) {
				// A deferred write release may still have the host read-only when an upload
				// watches again. Both locks are held, so `applied` has no host call in flight;
				// the pending release will reconcile the new counts at its scope's end.
				// New pages and stricter access-watch transitions still make their host call.
				if (reuse_applied && region.applied[page_index] == ToLevel(new_perms)) {
					continue;
				}
				if (range_bytes == 0) {
					range_begin           = page_index;
					potential_range_bytes = PAGE_SIZE;
				}
				range_bytes = potential_range_bytes;
			}
		}

		release_pending();
	}

	template <bool track, bool is_read>
	void UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
		ValidateRange(vaddr, size);
		const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
		const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
		for (auto chunk_begin = begin; chunk_begin < end;) {
			const auto chunk_end = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
			const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
			auto*      region = track ? GetOrCreateRegion(chunk_begin) : FindRegion(chunk_begin);
			if (region == nullptr) {
				Fatal("untracking unknown page 0x%016" PRIx64, chunk_begin);
			}
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			UpdateRegionWatchers<track, is_read, false>(*region, region_base, first, last);
			chunk_begin = chunk_end;
		}
	}

	std::unique_ptr<std::atomic<Region*>[]> regions;
	std::vector<std::unique_ptr<Region>>    region_storage;
	std::mutex                              region_mutex;
};

thread_local PageManager::Impl::DeferredBatch PageManager::Impl::t_deferred;

static_assert(std::atomic<void*>::is_always_lock_free);

PageManager::DeferUnprotectScope::DeferUnprotectScope() noexcept {
	Impl::t_deferred.depth++;
}

PageManager::DeferUnprotectScope::~DeferUnprotectScope() {
	if (--Impl::t_deferred.depth == 0) {
		Impl::ApplyDeferredBatch();
	}
}

bool PageManager::InDeferUnprotectScope() noexcept {
	return Impl::t_deferred.depth != 0;
}

void PageManager::Reconcile(uint64_t vaddr, uint64_t size, bool now) {
	if (GetDeferMode() == DeferMode::Off || !GuestRange {vaddr, size}.Valid()) {
		// Nothing is ever deferred: every host call was made under the region lock.
		return;
	}
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, PAGE_SIZE);
	for (auto chunk_begin = begin; chunk_begin < end;) {
		const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
		const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
		const auto first       = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
		const auto last        = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
		auto*      region      = m_impl->FindRegion(chunk_begin);
		chunk_begin            = chunk_end;
		if (region == nullptr) {
			continue; // never watched: nothing to update
		}
		if (!now && Impl::t_deferred.depth != 0 &&
		    m_impl->RecordDeferred(*region, region_base, first, last)) {
			continue;
		}
		Impl::CountApplied(m_impl->ApplySpan(*region, region_base, first, last));
	}
}

PageManager::PageManager(): m_impl(std::make_unique<Impl>()) {}

PageManager::~PageManager() = default;

uint64_t PageManager::GetPageSize() const {
	return PAGE_SIZE;
}

template <bool track>
void PageManager::UpdatePageWatchers(uint64_t vaddr, uint64_t size) {
	m_impl->UpdatePageWatchers<track, false>(vaddr, size);
}

template void PageManager::UpdatePageWatchers<true>(uint64_t, uint64_t);
template void PageManager::UpdatePageWatchers<false>(uint64_t, uint64_t);

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(uint64_t base_addr, RegionBits& mask) {
	if (base_addr % REGION_SIZE != 0 || base_addr >= ADDRESS_SIZE ||
	    REGION_SIZE > ADDRESS_SIZE - base_addr) {
		Fatal("invalid tracking region base 0x%016" PRIx64, base_addr);
	}

	const auto start_range = mask.FirstRange();
	const auto end_range   = mask.LastRange();
	if (start_range.first == REGION_PAGES) {
		FailFast("empty region watcher mask");
	}
	const auto first = start_range.first;
	const auto last  = end_range.second;
	if (start_range.second == end_range.second) {
		m_impl->UpdatePageWatchers<track, is_read>(base_addr + first * PAGE_SIZE,
		                                           (last - first) * PAGE_SIZE);
		return;
	}

	auto* region = track ? m_impl->GetOrCreateRegion(base_addr) : m_impl->FindRegion(base_addr);
	if (region == nullptr) {
		Fatal("untracking unknown region 0x%016" PRIx64, base_addr);
	}
	m_impl->UpdateRegionWatchers<track, is_read, true>(*region, base_addr, first, last, &mask);
}

PageManager::WatchedPages PageManager::CountWatchedPages(uint64_t vaddr, uint64_t size) {
	WatchedPages result;
	if (!GuestRange {vaddr, size}.Valid() || vaddr >= ADDRESS_SIZE) {
		return result;
	}
	const auto begin = Common::AlignDown(vaddr, PAGE_SIZE);
	const auto end   = std::min(Common::AlignUp(vaddr + size, PAGE_SIZE), ADDRESS_SIZE);
	for (auto chunk_begin = begin; chunk_begin < end;) {
		const auto chunk_end   = std::min(end, Common::AlignUp(chunk_begin + 1, REGION_SIZE));
		const auto region_base = Common::AlignDown(chunk_begin, REGION_SIZE);
		if (auto* region = m_impl->FindRegion(chunk_begin); region != nullptr) {
			SpinGuard  lock(region->lock);
			const auto first = static_cast<size_t>((chunk_begin - region_base) / PAGE_SIZE);
			const auto last  = static_cast<size_t>((chunk_end - region_base) / PAGE_SIZE);
			for (auto index = first; index < last; index++) {
				const auto& page = region->pages[index];
				result.access += page.access_watchers != 0 ? 1u : 0u;
				result.write += page.access_watchers == 0 && page.write_watchers != 0 ? 1u : 0u;
			}
		}
		chunk_begin = chunk_end;
	}
	return result;
}

template void PageManager::UpdatePageWatchersForRegion<true, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<true, false>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, true>(uint64_t, RegionBits&);
template void PageManager::UpdatePageWatchersForRegion<false, false>(uint64_t, RegionBits&);

} // namespace Libs::Graphics
