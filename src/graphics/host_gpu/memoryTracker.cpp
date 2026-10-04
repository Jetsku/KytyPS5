#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveSwitch.h"

#include <cstdlib>

namespace Libs::Graphics {

namespace {

// KYTY_HOT_PAGE_MAX_LIVE (default unset: KYTY_HOT_PAGE_MAX's startup value): a live hot page limit
// for A/B runs, read at every write fault and hot page collection.
Live::Switch g_hot_page_max_live("KYTY_HOT_PAGE_MAX_LIVE", [](const char* value) -> int64_t {
	return value == nullptr ? 0
	                        : static_cast<int64_t>(std::min<unsigned long long>(
	                              std::strtoull(value, nullptr, 10), 65536));
});

} // namespace

uint32_t MemoryTracker::HotMax() const noexcept {
	const auto live = g_hot_page_max_live.Get();
	return live > 0 ? static_cast<uint32_t>(live) : m_fault_policy.hot_max;
}

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
	std::vector<uint64_t> pages;
	SettleHotPages(vaddr, size, pages);
	return pages;
}

void MemoryTracker::SettleHotPages(uint64_t vaddr, uint64_t size, std::vector<uint64_t>& pages) {
	CheckNotInUploadCallback();
	if (m_hot_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	const auto listed = pages.size();
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
	MemoryStats::Count(MemoryStats::Counter::HotDemotions, pages.size() - listed);
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
		// Swept pages stay CPU-dirty and writable outside the hot set (see DemoteHotPages): each
		// demoted run is published before it changes. Pages that stay hot change nothing (logging
		// the whole region made the next dirty-log pass rescan every buffer in it).
		demoted += manager->SweepHot(frame, idle_frames, m_hot_count,
		                             [this](uint64_t address, uint64_t bytes) noexcept {
			                             NotifyCpuMutation(address, bytes);
		                             });
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

std::pair<uint64_t, uint64_t> MemoryTracker::FaultWindow(uint64_t offset, uint64_t bytes,
                                                         uint64_t ahead) noexcept {
	// Page indices as RegionManager::MarkWriteFault computes its window.
	const uint64_t start = offset / TRACKER_PAGE_SIZE;
	const uint64_t end   = (offset + bytes + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE;
	if (ahead <= 1) {
		return {start * TRACKER_PAGE_SIZE, end * TRACKER_PAGE_SIZE};
	}
	const uint64_t window_begin = start / ahead * ahead;
	const uint64_t window_end =
	    std::min<uint64_t>((end + ahead - 1) / ahead * ahead, TRACKER_REGION_PAGES);
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

bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
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
	CheckNotInUploadCallback();
	Iterate<true>(vaddr, size, [this](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		NotifyCpuMutation(manager->GetCpuAddr() + offset, bytes);
		manager->ChangeState<DirtySource::Cpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::MarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	CheckNotInUploadCallback();
	CleanVerdict::Invalidate(vaddr, size, Coherence::Source::TrackerGpuMark);
	Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, true>(manager->GetCpuAddr() + offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
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
