#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/rendererBatch.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/parkingLock.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <utility>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

// Waiters spin, then park (ParkingSpinLock, KYTY_TRACKER_LOCK_PARK); with parking off they spin
// as before: reading only with pause under KYTY_RENDERER_BATCH, else retrying back to back.
class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		if (m_owner.load(std::memory_order_relaxed) == thread) {
			EXIT("recursive region tracking lock\n");
		}
		if (!m_lock.try_lock()) [[unlikely]] {
			MemoryStats::Count(MemoryStats::Counter::TrackerLockContended);
			if (m_owner.load(std::memory_order_relaxed) == thread) {
				EXIT("recursive region tracking lock while contended\n");
			}
			(void)m_lock.LockContended(Common::RendererBatchEnabled());
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		if (m_owner.load(std::memory_order_relaxed) != CurrentThread()) {
			EXIT("region tracking lock released by non-owner\n");
		}
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.unlock();
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		return static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	ParkingSpinLock      m_lock;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
		PublishCpuMirror();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	// Mutation serial (MemoryTracker::RangeSignature). Every change of this region's CPU-dirty,
	// GPU-dirty, hot or readback-pending bits advances it, under `lock` and before the bits change,
	// so it never goes back and an unchanged value means the bits did not change in between. It
	// starts at 1, so an existing region always contributes to a signature.
	[[nodiscard]] uint64_t Serial() const noexcept {
		return m_serial.load(std::memory_order_acquire);
	}
	// Dirtying serial (MemoryTracker::RangeDirtiedSignature): advanced, under `lock` and before the
	// bits change, by every transition that can turn a page of this region CPU-dirty (write
	// faults, including hot promotion and fault-ahead, and explicit CPU-dirty marks). Uploads and
	// other clears never advance it, so a verify mode can tell a page dirtied after a moment from
	// one that was already dirty then.
	[[nodiscard]] uint64_t Dirtied() const noexcept {
		return m_dirtied.load(std::memory_order_acquire);
	}

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().AnyInRange(start, end);
	}

	// IsModified<Gpu> without `lock`, on the lock-free mirror of the GPU-dirty bits (any thread).
	// Every change of those bits republishes the mirror under `lock` (PublishGpuMirror), so a
	// thread holding `lock` sees the mirror equal to the bits; without it, each 64-page word is
	// read atomically as it was at some moment, which makes this a hint (DrawPrep worker reads).
	[[nodiscard]] bool IsGpuModifiedRelaxed(uint64_t offset, uint64_t size) const noexcept {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return RegionBits::AnyInRange(start, end, [this](size_t word) {
			return m_gpu_mirror[word].load(std::memory_order_relaxed);
		});
	}
	// Verify mode, caller holds `lock`: the mirror words of the range equal the GPU-dirty bits.
	[[nodiscard]] bool GpuMirrorMatches(uint64_t offset, uint64_t size) const noexcept {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		for (auto word = start / 64u; word <= (end - 1) / 64u; word++) {
			if (m_gpu_mirror[word].load(std::memory_order_relaxed) != m_gpu_dirty.Word(word)) {
				return false;
			}
		}
		return true;
	}
	// IsModified<Cpu> without `lock`, on the lock-free mirror of the CPU-dirty bits, republished
	// under `lock` after every change of those bits (PublishCpuMirror).
	[[nodiscard]] bool IsCpuModifiedRelaxed(uint64_t offset, uint64_t size) const noexcept {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return RegionBits::AnyInRange(start, end, [this](size_t word) {
			return m_cpu_mirror[word].load(std::memory_order_relaxed);
		});
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.AnyInRange(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (m_cpu_dirty.AnyInRange(start, end)) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		auto& bits = GetBits<source>();
		{
			const RegionBits current(bits, start, end);
			const bool       changes = enable ? current.Count() != end - start : current.Any();
			if (changes || (source == DirtySource::Gpu &&
			                RegionBits(m_readback_pending, start, end).Any())) {
				Bump();
			}
			if (source == DirtySource::Cpu && enable && changes) {
				BumpDirtied();
			}
		}
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			PublishCpuMirror(start, end);
		}
		if constexpr (source == DirtySource::Gpu) {
			// Any GPU ownership transition supersedes an outstanding side readback: a newer
			// writer (enable) or an explicit download/unmark (disable) now owns these pages.
			m_readback_pending.UnsetRange(start, end);
			PublishGpuMirror(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			if (mask.Any() ||
			    (source == DirtySource::Gpu && RegionBits(m_readback_pending, start, end).Any())) {
				Bump();
			}
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				PublishCpuMirror(start, end);
				UpdateProtection<true, false>();
			} else {
				m_readback_pending.UnsetRange(start, end);
				PublishGpuMirror(start, end);
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// A guest write fault on [vaddr, vaddr + size), none of which is GPU-dirty: the pages become
	// CPU-dirty exactly as ChangeState<Cpu, true> makes them. Additionally
	//  - fault-ahead: every page of the aligned `policy.ahead_pages` window around them that is
	//    neither GPU-dirty nor already CPU-dirty becomes CPU-dirty too (one protection update).
	//    A CPU-dirty page is only ever uploaded (it may have changed), so extra pages cost upload
	//    bytes, never correctness; other page watchers (images) keep their own protection;
	//  - hot pages: a page write-faulting in `policy.hot_frames` consecutive frames becomes hot
	//    (sticky CPU-dirty, see CollectUpload) while `hot_count` stays below `policy.hot_max`.
	// on_ahead(address, bytes) receives each run of fault-ahead pages before they become
	// writable (still under the region lock). Returns the pages marked by fault-ahead and the
	// pages promoted to hot.
	struct FaultPolicy {
		uint32_t ahead_pages = 1; // power of two dividing TRACKER_REGION_PAGES; 1 = off
		uint32_t hot_frames  = 0; // 0 = hot pages off
		uint32_t hot_max     = 0;
		// Written uploads copy with no region lock held (MemoryTracker::ForEachWrittenUploadRange).
		bool copy_outside_lock = false;
	};
	struct FaultResult {
		uint64_t ahead_pages = 0;
		uint32_t promoted    = 0;
		uint32_t refused     = 0; // pages that qualified for hot while `policy.hot_max` were hot
	};
	template <typename AheadFunc>
	FaultResult MarkWriteFault(uint64_t vaddr, uint64_t size, const FaultPolicy& policy,
	                           uint32_t frame, std::atomic_uint32_t& hot_count,
	                           AheadFunc&& on_ahead) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if (m_gpu_dirty.AnyInRange(start, end)) {
			EXIT("CPU dirty state conflicts with GPU dirty state\n");
		}
		FaultResult result;
		// Dirty bits (and possibly the hot set) change below.
		Bump();
		BumpDirtied();
		// Only pages this tracker protected fault through it; already CPU-dirty pages fault for
		// another watcher (an image) and are not part of a fault/reprotect cycle.
		const RegionBits already_dirty(m_cpu_dirty, start, end);
		m_cpu_dirty.SetRange(start, end);
		// The pages whose CPU-dirty bits may change: the faulting ones and the fault-ahead window.
		size_t changed_begin = start;
		size_t changed_end   = end;
		if (policy.ahead_pages > 1) {
			const size_t window_begin = start / policy.ahead_pages * policy.ahead_pages;
			const size_t window_end   = std::min<size_t>(
                (end + policy.ahead_pages - 1) / policy.ahead_pages * policy.ahead_pages,
                TRACKER_REGION_PAGES);
			changed_begin = std::min(changed_begin, window_begin);
			changed_end   = std::max(changed_end, window_end);
			RegionBits all;
			all.Fill();
			const RegionBits window = RegionBits(all, window_begin, window_end) & ~m_gpu_dirty &
			                          ~m_cpu_dirty;
			result.ahead_pages = window.Count();
			m_cpu_dirty |= window;
			for (const auto [first, last]: window) {
				on_ahead(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
			}
		}
		// Published before the pages become writable: a guest write can only land after it.
		PublishCpuMirror(changed_begin, changed_end);
		UpdateProtection<false, false>();
		if (policy.hot_frames == 0) {
			return result;
		}
		if (m_history == nullptr) {
			m_history = std::make_unique<FaultHistory>();
		}
		const auto current  = static_cast<uint16_t>(frame);
		const auto previous = static_cast<uint16_t>(frame - 1);
		for (size_t page = start; page < end; page++) {
			auto& streak = m_history->streak[page];
			auto& last   = m_history->frame[page];
			if (already_dirty.Get(page) || (streak != 0 && last == current)) {
				continue;
			}
			streak = (streak != 0 && last == previous && streak != UINT8_MAX)
			             ? static_cast<uint8_t>(streak + 1)
			             : uint8_t {1};
			last = current;
			if (streak < policy.hot_frames) {
				continue;
			}
			if (hot_count.fetch_add(1, std::memory_order_relaxed) >= policy.hot_max) {
				hot_count.fetch_sub(1, std::memory_order_relaxed);
				result.refused++;
				continue;
			}
			streak = 0;
			m_hot.Set(page);
			m_history->used[page] = current;
			result.promoted++;
		}
		return result;
	}

	// Upload collection (MemoryTracker::ForEachUploadRange): clears the CPU-dirty state of the
	// range and write-protects it, except for hot pages when `keep_hot`: those stay CPU-dirty and
	// writable (no fault on the next write) and are reported with hot = true so the caller can
	// skip unchanged contents. Without `keep_hot` (a GPU writer takes the range) hot pages
	// return to normal first. func(address, bytes, hot) receives page runs.
	template <typename Func>
	uint32_t CollectUpload(uint64_t vaddr, uint64_t size, bool keep_hot, uint32_t frame,
	                       std::atomic_uint32_t& hot_count, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		RegionBits hot(m_hot, start, end);
		uint32_t   demoted = 0;
		// Hot pages stay as they are (keep_hot) or return to normal tracking; CPU-dirty pages that
		// are not hot are cleared. Recording which hot pages an upload visited changes no bits.
		if ((hot.Any() && !keep_hot) || (RegionBits(m_cpu_dirty, start, end) & ~hot).Any()) {
			Bump();
		}
		if (hot.Any()) {
			if (keep_hot) {
				for (const auto [first, last]: hot) {
					for (auto page = first; page < last; page++) {
						m_history->used[page] = static_cast<uint16_t>(frame);
					}
				}
			} else {
				demoted = static_cast<uint32_t>(hot.Count());
				m_hot ^= hot;
				hot_count.fetch_sub(demoted, std::memory_order_relaxed);
				hot.Clear();
			}
		}
		// Hot pages are always CPU-dirty, so `normal` holds exactly the pages to clear.
		const RegionBits normal = RegionBits(m_cpu_dirty, start, end) & ~hot;
		if (normal.Any()) {
			m_cpu_dirty ^= normal;
			PublishCpuMirror(start, end);
			UpdateProtection<true, false>();
		}
		for (const auto [first, last]: normal) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE, false);
		}
		for (const auto [first, last]: hot) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE, true);
		}
		return demoted;
	}

	// Returns hot pages of the range to normal tracking. They stay CPU-dirty (and writable)
	// until their next upload clears and protects them. Returns the pages demoted.
	uint32_t DemoteHot(uint64_t vaddr, uint64_t size, std::atomic_uint32_t& hot_count) {
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits hot(m_hot, start, end);
		const auto       demoted = static_cast<uint32_t>(hot.Count());
		if (demoted != 0) {
			Bump();
			m_hot ^= hot;
			hot_count.fetch_sub(demoted, std::memory_order_relaxed);
		}
		return demoted;
	}

	// Returns the hot pages of the range to normal tracking as CLEAN pages: clears their CPU-dirty
	// bits and write-protects them, reporting each page to func(address). The caller compares each
	// with its last uploaded contents afterwards and marks changed ones CPU-dirty again; a write
	// landing after the protection faults and marks the page itself.
	template <typename Func>
	uint32_t SettleHot(uint64_t vaddr, uint64_t size, std::atomic_uint32_t& hot_count,
	                   Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits hot(m_hot, start, end);
		const auto       settled = static_cast<uint32_t>(hot.Count());
		if (settled == 0) {
			return 0;
		}
		Bump();
		m_hot ^= hot;
		hot_count.fetch_sub(settled, std::memory_order_relaxed);
		m_cpu_dirty ^= hot;
		PublishCpuMirror(start, end);
		UpdateProtection<true, false>();
		for (const auto [first, last]: hot) {
			for (auto page = first; page < last; page++) {
				func(m_cpu_addr + page * TRACKER_PAGE_SIZE);
			}
		}
		return settled;
	}

	// Demotes hot pages no upload visited for more than `idle_frames` frames.
	uint32_t SweepHot(uint32_t frame, uint32_t idle_frames, std::atomic_uint32_t& hot_count) {
		if (m_hot.None()) {
			return 0;
		}
		RegionBits idle;
		for (const auto [first, last]: m_hot) {
			for (auto page = first; page < last; page++) {
				const auto age =
				    static_cast<uint16_t>(static_cast<uint16_t>(frame) - m_history->used[page]);
				if (age > idle_frames) {
					idle.Set(page);
				}
			}
		}
		const auto demoted = static_cast<uint32_t>(idle.Count());
		if (demoted != 0) {
			Bump();
			m_hot ^= idle;
			hot_count.fetch_sub(demoted, std::memory_order_relaxed);
		}
		return demoted;
	}

	[[nodiscard]] bool IsHot(uint64_t vaddr, uint64_t size) const {
		const auto [start, end] = GetPageRange(vaddr, size);
		return m_hot.AnyInRange(start, end);
	}

	// Caller holds `lock`: every page of [m_cpu_addr + offset, + size) is GPU-dirty (so none is
	// CPU-dirty or hot) and none is marked readback-pending. A written upload of such pages
	// (CollectUpload, then ChangeState<Gpu, true>) collects nothing and changes no bit, serial or
	// protection.
	[[nodiscard]] bool IsGpuOwned(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		const bool all_gpu_dirty = !RegionBits::AnyInRange(
		    start, end, [this](size_t word) { return ~m_gpu_dirty.Word(word); });
		return all_gpu_dirty && !m_readback_pending.AnyInRange(start, end);
	}

	// Side readbacks (BufferCache::ReadMemory). Pending marks the GPU-dirty pages of a range whose
	// exact dirty bytes were handed to one side-copy publication. Any later GPU transition of a
	// page clears its mark, so completion unprotects only pages that no newer writer re-owned.
	void MarkReadbackPending(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits dirty(m_gpu_dirty, start, end);
		if (dirty.Any()) {
			Bump();
		}
		for (const auto [first, last]: dirty) {
			m_readback_pending.SetRange(first, last);
		}
	}

	// Returns {pages unprotected, GPU-dirty pages retained in the range}.
	std::pair<uint64_t, uint64_t> ClearReadbackPending(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		const RegionBits pending(m_readback_pending, start, end);
		uint64_t         cleared = 0;
		if (pending.Any()) {
			Bump();
		}
		for (const auto [first, last]: pending) {
			m_gpu_dirty.UnsetRange(first, last);
			m_readback_pending.UnsetRange(first, last);
			cleared += last - first;
		}
		if (cleared != 0) {
			PublishGpuMirror(start, end);
		}
		uint64_t         retained = 0;
		const RegionBits dirty(m_gpu_dirty, start, end);
		for (const auto [first, last]: dirty) {
			retained += last - first;
		}
		if (cleared != 0) {
			UpdateProtection<false, true>();
		}
		return {cleared, retained};
	}

	TrackingSpinLock lock;

private:
	// Callers hold `lock`: the serial advances before the bits it describes change.
	void Bump() noexcept { m_serial.fetch_add(1, std::memory_order_release); }
	// Callers hold `lock`, before pages become CPU-dirty (Dirtied).
	void BumpDirtied() noexcept { m_dirtied.fetch_add(1, std::memory_order_release); }

	// Callers hold `lock`, after changing the GPU-dirty bits of pages [first, last), and no other
	// (IsGpuModifiedRelaxed). Only the mirror words of those pages are stored: the other words
	// already equal their bits, and threads reading them keep their cache lines.
	void PublishGpuMirror(size_t first = 0, size_t last = TRACKER_REGION_PAGES) noexcept {
		for (size_t word = first / 64u; word < (last + 63u) / 64u; word++) {
			m_gpu_mirror[word].store(m_gpu_dirty.Word(word), std::memory_order_relaxed);
		}
	}
	// Callers hold `lock` (or construct), after changing the CPU-dirty bits of pages [first,
	// last), and no other, and before any page this makes CPU-dirty becomes writable
	// (IsCpuModifiedRelaxed). Only the mirror words of those pages are stored, as above: a write
	// fault no longer rewrites both mirror cache lines the GPU thread reads.
	void PublishCpuMirror(size_t first = 0, size_t last = TRACKER_REGION_PAGES) noexcept {
		for (size_t word = first / 64u; word < (last + 63u) / 64u; word++) {
			m_cpu_mirror[word].store(m_cpu_dirty.Word(word), std::memory_order_relaxed);
		}
	}

	template <bool track, bool is_read>
	void UpdateProtection() {
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	// Write-fault history for hot-page detection, allocated on the first tracked fault.
	struct FaultHistory {
		std::array<uint16_t, TRACKER_REGION_PAGES> frame {};  // frame of the last write fault
		std::array<uint8_t, TRACKER_REGION_PAGES>  streak {}; // consecutive faulting frames
		std::array<uint16_t, TRACKER_REGION_PAGES> used {};   // hot: frame of the last upload
	};

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_readback_pending;
	RegionBits   m_writable;
	RegionBits   m_readable;
	// Hot pages: always a subset of m_cpu_dirty (never GPU-dirty).
	RegionBits                    m_hot;
	std::unique_ptr<FaultHistory> m_history;
	std::atomic<uint64_t>         m_serial {1};
	std::atomic<uint64_t>         m_dirtied {1};
	// Lock-free copies of m_gpu_dirty and m_cpu_dirty (Publish*Mirror), on their own cache lines.
	alignas(64) std::array<std::atomic<uint64_t>, RegionBits::Words> m_gpu_mirror {};
	alignas(64) std::array<std::atomic<uint64_t>, RegionBits::Words> m_cpu_mirror {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
