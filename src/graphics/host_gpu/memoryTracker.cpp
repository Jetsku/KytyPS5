#include "graphics/host_gpu/memoryTracker.h"
#include "common/staticOnce.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "graphics/host_gpu/watchRange.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager, bool track_cpu_mutations,
                             FaultPolicy fault_policy)
    : m_page_manager(page_manager), m_track_cpu_mutations(track_cpu_mutations),
      m_fault_policy(fault_policy) {
	if (m_fault_policy.ahead_pages == 0 ||
	    (m_fault_policy.ahead_pages & (m_fault_policy.ahead_pages - 1)) != 0 ||
	    m_fault_policy.ahead_pages > TRACKER_REGION_PAGES) {
		EXIT("invalid memory tracker fault-ahead window\n");
	}
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
}

bool MemoryTracker::IsRegionHot(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsHot(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::DemoteHotPages(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	if (m_hot_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	uint32_t demoted = 0;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		if (manager->IsHot(manager->GetCpuAddr() + offset, bytes)) {
			// The demoted pages stay CPU-dirty and writable, now outside the hot set: publish
			// before the change like every other transition FaultMutationEpoch() covers.
			NotifyCpuMutation(manager->GetCpuAddr() + offset, bytes);
		}
		demoted += manager->DemoteHot(manager->GetCpuAddr() + offset, bytes, m_hot_count);
	});
	MemoryStats::Count(MemoryStats::Counter::HotDemotions, demoted);
}

std::vector<uint64_t> MemoryTracker::SettleHotPages(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	std::vector<uint64_t> pages;
	if (m_hot_count.load(std::memory_order_relaxed) == 0) {
		return pages;
	}
	const auto collect = [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		(void)manager->SettleHot(manager->GetCpuAddr() + offset, bytes, m_hot_count,
		                         [&pages](uint64_t page) { pages.push_back(page); });
	};
	if (size != 0) {
		Iterate<false>(vaddr, size, collect);
	} else {
		std::vector<RegionManager*> managers;
		{
			std::lock_guard lock(m_region_mutex);
			managers.reserve(m_region_storage.size());
			for (const auto& manager: m_region_storage) {
				managers.push_back(manager.get());
			}
		}
		for (auto* manager: managers) {
			collect(manager, 0, TRACKER_REGION_SIZE);
		}
	}
	MemoryStats::Count(MemoryStats::Counter::HotDemotions, pages.size());
	return pages;
}

void MemoryTracker::SweepHotPages(uint32_t idle_frames) {
	CheckNotInUploadCallback();
	if (m_hot_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	std::vector<RegionManager*> managers;
	{
		std::lock_guard lock(m_region_mutex);
		managers.reserve(m_region_storage.size());
		for (const auto& manager: m_region_storage) {
			managers.push_back(manager.get());
		}
	}
	const auto frame   = Frame();
	uint32_t   demoted = 0;
	for (auto* manager: managers) {
		std::scoped_lock lock(manager->lock);
		// Swept pages stay CPU-dirty and writable outside the hot set (see DemoteHotPages). A
		// region without hot pages cannot change here; otherwise publish conservatively.
		if (manager->IsHot(manager->GetCpuAddr(), TRACKER_REGION_SIZE)) {
			NotifyCpuMutation(manager->GetCpuAddr(), TRACKER_REGION_SIZE);
		}
		demoted += manager->SweepHot(frame, idle_frames, m_hot_count);
	}
	MemoryStats::Count(MemoryStats::Counter::HotDemotions, demoted);
}

MemoryTracker::~MemoryTracker() = default;

void MemoryTracker::AdvanceCpuMutationEpoch() noexcept {
	// Publish before dirtying/unprotecting under the region lock. A scanner observing this
	// token still acquires that lock; saturation permanently disables token reuse, avoiding ABA.
	auto epoch = m_cpu_mutation_epoch.load(std::memory_order_relaxed);
	while (epoch != UINT64_MAX &&
	       !m_cpu_mutation_epoch.compare_exchange_weak(epoch, epoch + 1,
	                                                   std::memory_order_release,
	                                                   std::memory_order_relaxed)) {}
}

void MemoryTracker::NotifyCpuMutation(uint64_t vaddr, uint64_t size) noexcept {
	if (m_reach_log_on.load(std::memory_order_acquire) && size != 0) {
		std::scoped_lock lock(m_dirtied_mutex);
		if (!m_reach_log_overflow) {
			if (m_reach_log.Size() >= DirtiedLogMaxRanges) {
				m_reach_log_overflow = true;
				m_reach_log.Clear();
			} else {
				m_reach_log.Add(vaddr, size);
			}
		}
	}
	if (!m_track_cpu_mutations) {
		return;
	}
	if (!m_dirtied_log.load(std::memory_order_acquire)) {
		AdvanceCpuMutationEpoch();
		return;
	}
	// The range and the epoch advance under one lock: a take (TakeDirtiedRanges) that sees the
	// advanced epoch also sees the range. Callers hold the region lock and change the pages'
	// bits after this, so whoever handles a taken range under that lock sees the new bits.
	std::scoped_lock lock(m_dirtied_mutex);
	if (!m_dirtied_overflow && size != 0) {
		if (m_dirtied.Size() >= DirtiedLogMaxRanges) {
			m_dirtied_overflow = true;
			m_dirtied.Clear();
		} else {
			m_dirtied.Add(vaddr, size);
		}
	}
	AdvanceCpuMutationEpoch();
}

bool MemoryTracker::TakeDirtiedRanges(RangeSet& ranges, uint64_t& epoch) {
	std::scoped_lock lock(m_dirtied_mutex);
	ranges.Clear();
	std::swap(ranges, m_dirtied);
	epoch                 = m_cpu_mutation_epoch.load(std::memory_order_acquire);
	const bool complete   = !m_dirtied_overflow;
	m_dirtied_overflow    = false;
	return complete;
}

bool MemoryTracker::TakeReachLog(RangeSet& ranges) {
	std::scoped_lock lock(m_dirtied_mutex);
	ranges.Clear();
	std::swap(ranges, m_reach_log);
	const bool complete  = !m_reach_log_overflow;
	m_reach_log_overflow = false;
	return complete;
}

std::pair<uint64_t, uint64_t> MemoryTracker::FaultWindow(uint64_t offset, uint64_t bytes,
                                                         uint64_t ahead, bool sequential) noexcept {
	// Page indices as RegionManager::MarkWriteFault computes its window.
	const uint64_t start = offset / TRACKER_PAGE_SIZE;
	const uint64_t end   = (offset + bytes + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
	if (ahead <= 1) {
		return {start * TRACKER_PAGE_SIZE, end * TRACKER_PAGE_SIZE};
	}
	const uint64_t window_begin = start / ahead * ahead;
	uint64_t window_end =
	    std::min<uint64_t>((end + ahead - 1) / ahead * ahead, TRACKER_REGION_PAGES);
	if (sequential) {
		// The forward window [start, start + ahead) of a sequential fill (a superset either way).
		window_end = std::max(window_end, std::min<uint64_t>(start + ahead, TRACKER_REGION_PAGES));
	}
	return {window_begin * TRACKER_PAGE_SIZE, std::max(end, window_end) * TRACKER_PAGE_SIZE};
}

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	// New regions start entirely CPU dirty. Notify before making the region visible.
	NotifyCpuMutation(index * TRACKER_REGION_SIZE, TRACKER_REGION_SIZE);
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

namespace {

// KYTY_TRACKER_LOCKFREE_DIRTY (MemoryTracker::SetLockFreeDirtyOwner).
std::atomic<MemoryTracker::OwnerThreadFn> g_lockfree_owner {nullptr};

bool LockFreeDirtyHere() noexcept {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_TRACKER_LOCKFREE_DIRTY");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	if (!enabled) {
		return false;
	}
	const auto is_owner = g_lockfree_owner.load(std::memory_order_acquire);
	return is_owner != nullptr && is_owner();
}

} // namespace

void MemoryTracker::SetLockFreeDirtyOwner(OwnerThreadFn is_owner) noexcept {
	g_lockfree_owner.store(is_owner, std::memory_order_release);
}

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	if (LockFreeDirtyHere()) {
		return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			return manager->IsModifiedAcquire<DirtySource::Cpu>(offset, bytes);
		});
	}
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	if (LockFreeDirtyHere()) {
		return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
			return manager->IsModifiedAcquire<DirtySource::Gpu>(offset, bytes);
		});
	}
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	});
}

MemoryTracker::DirtyState MemoryTracker::QueryDirty(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	DirtyState state;
	uint64_t   visited = 0;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		visited++;
		std::scoped_lock lock(manager->lock);
		state.gpu = state.gpu || manager->IsModified<DirtySource::Gpu>(offset, bytes);
		state.cpu = state.cpu || manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
	const auto regions = (vaddr + size - 1) / TRACKER_REGION_SIZE - vaddr / TRACKER_REGION_SIZE + 1;
	if (!state.gpu && visited != regions) {
		// IsRegionCpuModified would create the missing regions, entirely CPU dirty.
		Iterate<true>(vaddr, size, [](RegionManager*, uint64_t, uint64_t) {});
		state.cpu = true;
	}
	return state;
}

bool MemoryTracker::QueryDirtyRelaxed(uint64_t vaddr, uint64_t size, DirtyState& state) const {
	ValidateRange(vaddr, size);
	state             = {};
	uint64_t remaining = size;
	uint64_t index     = vaddr / TRACKER_REGION_SIZE;
	uint64_t offset    = vaddr % TRACKER_REGION_SIZE;
	while (remaining != 0) {
		const auto  bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return false;
		}
		state.gpu = state.gpu || manager->IsGpuModifiedRelaxed(offset, bytes);
		state.cpu = state.cpu || manager->IsCpuModifiedRelaxed(offset, bytes);
		remaining -= bytes;
		offset = 0;
		index++;
	}
	return true;
}

bool MemoryTracker::QueryCpuDirtyRelaxed(uint64_t vaddr, uint64_t size, bool& dirty) const {
	ValidateRange(vaddr, size);
	dirty              = false;
	uint64_t remaining = size;
	uint64_t index     = vaddr / TRACKER_REGION_SIZE;
	uint64_t offset    = vaddr % TRACKER_REGION_SIZE;
	while (remaining != 0) {
		const auto  bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return false;
		}
		dirty = dirty || manager->IsCpuModifiedRelaxed(offset, bytes);
		remaining -= bytes;
		offset = 0;
		index++;
	}
	return true;
}

bool MemoryTracker::IsRegionGpuModifiedRelaxed(uint64_t vaddr, uint64_t size) const {
	ValidateRange(vaddr, size);
	uint64_t remaining = size;
	uint64_t index     = vaddr / TRACKER_REGION_SIZE;
	uint64_t offset    = vaddr % TRACKER_REGION_SIZE;
	while (remaining != 0) {
		const auto  bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
		const auto* manager = m_regions[index].load(std::memory_order_acquire);
		if (manager != nullptr && manager->IsGpuModifiedRelaxed(offset, bytes)) {
			return true;
		}
		remaining -= bytes;
		offset = 0;
		index++;
	}
	return false;
}

bool MemoryTracker::IsRangeGpuOwned(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	ValidateRange(vaddr, size);
	uint64_t remaining = size;
	uint64_t index     = vaddr / TRACKER_REGION_SIZE;
	uint64_t offset    = vaddr % TRACKER_REGION_SIZE;
	while (remaining != 0) {
		const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
		auto*      manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) {
			return false; // a missing region is entirely CPU dirty once created
		}
		{
			std::scoped_lock lock(manager->lock);
			if (!manager->IsGpuOwned(offset, bytes)) {
				return false;
			}
		}
		remaining -= bytes;
		offset = 0;
		index++;
	}
	return true;
}

bool MemoryTracker::GpuMirrorMatches(uint64_t vaddr, uint64_t size) {
	bool matches = true;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		matches = matches && manager->GpuMirrorMatches(offset, bytes);
	});
	return matches;
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	WatchRange::Note("tracker CPU-modified", vaddr, size);
	WatchRange::NoteWriter("cpu-fault", vaddr, size);
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [this](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		NotifyCpuMutation(manager->GetCpuAddr() + offset, bytes);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	WatchRange::Note("tracker GPU-modified", vaddr, size);
	WatchRange::NoteWriter("tracker-gpu", vaddr, size);
	CheckNotInUploadCallback();
	CleanVerdict::Invalidate(vaddr, size, Coherence::Source::TrackerGpuMark);
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	WatchRange::Note("tracker GPU-unmark", vaddr, size);
	CheckNotInUploadCallback();
	CleanVerdict::Invalidate(vaddr, size, Coherence::Source::TrackerGpuUnmark);
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::MarkReadbackPending(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	// No clean-verdict bump: no dirty or protection state changes here.
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->MarkReadbackPending(manager->GetCpuAddr() + offset, bytes);
	});
}

MemoryTracker::ReadbackUnmarkResult MemoryTracker::UnmarkReadbackPending(uint64_t vaddr,
                                                                         uint64_t size) {
	CheckNotInUploadCallback();
	CleanVerdict::Invalidate(vaddr, size, Coherence::Source::TrackerReadbackUnmark);
	ReadbackUnmarkResult result;
	Iterate<false>(vaddr, size, [&result](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		const auto [unmarked, retained] =
		    manager->ClearReadbackPending(manager->GetCpuAddr() + offset, bytes);
		result.unmarked_pages += unmarked;
		result.retained_pages += retained;
	});
	return result;
}

void MemoryTracker::UntrackMemory(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	std::vector<RegionManager*> managers;
	managers.reserve((vaddr % TRACKER_REGION_SIZE + size + TRACKER_REGION_SIZE - 1) /
	                 TRACKER_REGION_SIZE);
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t, uint64_t) {
		managers.push_back(manager);
	});

	std::vector<std::unique_lock<TrackingSpinLock>> locks;
	locks.reserve(managers.size());
	for (auto* manager: managers) {
		locks.emplace_back(manager->lock);
	}
	if (Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		    return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	    })) {
		EXIT("cannot untrack GPU-dirty memory\n");
	}
	uint32_t demoted = 0;
	Iterate<false>(vaddr, size, [&](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		NotifyCpuMutation(manager->GetCpuAddr() + offset, bytes);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
		demoted += manager->DemoteHot(manager->GetCpuAddr() + offset, bytes, m_hot_count);
	});
	MemoryStats::Count(MemoryStats::Counter::HotDemotions, demoted);
}

} // namespace Libs::Graphics
