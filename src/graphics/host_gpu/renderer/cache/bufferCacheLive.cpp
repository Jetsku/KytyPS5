// KYTY_BDA_LIVE_PAGES = 0 | 1 | verify (default 0, live; the state exists only when the variable
// is set at start, because the device needs VK_EXT_external_memory_host from creation).
//
// The full BDA pass (an operation whose device-address reads have no proven reach, e.g. the
// software ray-tracing kernels) uploads every CPU-dirty page of every BDA-mapped buffer and
// write-protects it again; the guest's next write to it faults. With this switch a qualifying
// CPU-dirty caching page is not uploaded: its BDA page-table entry points into an imported window
// of the guest backing alias (bdaLiveWindows.h), so a BDA read of it reads the guest bytes as they
// are when the GPU reads them (as on the console). The page stays CPU-dirty and writable: no upload,
// no re-protection, no fault. The tracker state is untouched (it is the state after a write fault).
//
// Invariants (rtskip DESIGN.md section 4):
//   I1   redirected => CPU-dirty, not GPU-dirty, not hot, no GPU-modified image, no pending backing
//        publication, no early-released download, not CP-written since (verify V2 after each pass);
//   I2   after a full pass during which the fault epoch did not move, every CPU-dirty non-hot page of
//        a registered buffer in the GPU mappings is redirected or was uploaded (verify V1);
//   I3   an entry that goes back to the cache without an upload queues its page in the forced set,
//        which the next full pass (LivePagesPrePass) or a BDA-writing program uploads first;
//   I-W1 before a BDA-writing program every redirect is restored and uploaded (verify V3 counts
//        redirects still alive there); I-W2 no new redirect while BDA-write settles are pending;
//        I-W3 a written binding over a redirected page restores it (SynchronizeBuffer);
//   I-C1 a CP / emulator write over a redirected page on the GPU thread waits for the last full
//        pass's tick (earlier recorded reads must not see it) and the page is never redirected
//        again; on another thread it is counted (the game's own writes, ordered by its labels);
//   I-L1 imported windows live until teardown; I-L2 guest unmap restores the pages first;
//   I-L3 refused with KYTY_DROP_RELEASED (backing pages handed back under an import);
//   I-A1 / I-A2 import alignment, and 16 KiB + 64 bytes of a page inside one window.
// Do not combine with KYTY_BDA_REACH=verify, the BdaHotSync / BdaDirtyLog / BdaSyncEpoch verify
// modes or KYTY_BDA_ACCESS_TAGS=1 / verify (they rewrite entries or rescan pages themselves).
// With the KYTY_BDA_ACCESS_TAGS codegen option, a page read through its redirect is reported (its
// entry has no tag), keeps its redirect (now tagged) and is uploaded at the next full pass:
// performance only, the read itself saw current bytes.
//   KYTY_BDA_LIVE_WINDOW_MB (default 64): import window, a power of two.
//   KYTY_BDA_LIVE_MAX_MB (default 1024): cap of imported (pinned) memory.

#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/liveSwitch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bdaLiveWindows.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace {

int64_t ParseLivePages(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	// check (3): no redirects, only the V1/V2 checks (what the pass leaves without this switch).
	if (std::strcmp(value, "check") == 0) {
		return 3;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}

// Redirect modes (1, verify); check and 0 redirect nothing.
bool Redirects(int64_t mode) {
	return mode == 1 || mode == 2;
}
Live::Switch g_bda_live_pages("KYTY_BDA_LIVE_PAGES", ParseLivePages);

BufferCache* g_live_cache = nullptr; // the cache whose state the StreamCopyDefer hooks reach

uint64_t LiveEnvU64(const char* name, uint64_t fallback) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
}

constexpr uint64_t PAGE = BufferCache::CACHING_PAGESIZE;

} // namespace

struct BufferCache::BdaLive {
	mutable std::mutex                     mutex;
	std::map<uint64_t, vk::DeviceAddress>  pages;      // caching page -> redirect entry (untagged)
	std::unordered_set<uint64_t>           cp_written; // pages a CP write hit while redirected
	RangeSet                               forced;     // entries given back without an upload
	std::unique_ptr<BdaLiveWindows>        windows;
	std::vector<std::unique_ptr<Buffer>>   imports;
	uint64_t                               read_tick = 0; // last full pass that left redirects
	// Verify diagnostics: the last event of each page (1 = redirected, 2 + LiveRestore = restored).
	std::unordered_map<uint64_t, uint8_t>  history;
	// Performance only: pages a cache binding uploaded after their redirect are not redirected again
	// until this many full passes later (KYTY_BDA_LIVE_BOUND_PASSES, default 64; 0 = off).
	std::unordered_map<uint64_t, uint64_t> bound_until;
	uint64_t                               bound_passes = 64;
	uint64_t                               passes       = 0;
	void Note(uint64_t page, uint8_t event) {
		if (history.size() < (1u << 20u) || history.contains(page)) {
			history[page] = event;
		}
	}
	BdaLiveTotals                          totals;
	BdaLiveTotals                          reported;
	std::chrono::steady_clock::time_point  report_time {};
	uint32_t                               report_frames = 0;
	BufferCache*                           owner         = nullptr;
	~BdaLive() {
		if (g_live_cache == owner) {
			g_live_cache = nullptr;
		}
	}
};

void BufferCache::CreateBdaLive() {
	const auto* value = std::getenv("KYTY_BDA_LIVE_PAGES");
	if (value == nullptr || value[0] == '\0') {
		return;
	}
	if (LiveEnvU64("KYTY_DROP_RELEASED", 0) != 0) {
		std::printf("Kyty BDA live pages: refused, KYTY_DROP_RELEASED hands backing pages back under the imports\n");
		return;
	}
	const auto base = LibKernel::Memory::GuestBackingBase();
	const auto size = LibKernel::Memory::GuestBackingSize();
	if (!m_graphics.external_memory_host_enabled || base == 0 || size == 0) {
		std::printf("Kyty BDA live pages: refused (no VK_EXT_external_memory_host or no backing alias)\n");
		return;
	}
	auto live          = std::make_shared<BdaLive>();
	live->owner        = this;
	live->bound_passes = LiveEnvU64("KYTY_BDA_LIVE_BOUND_PASSES", 64);
	auto* state = live.get();
	const uint64_t window = std::clamp<uint64_t>(LiveEnvU64("KYTY_BDA_LIVE_WINDOW_MB", 64), 1, 1024) << 20u;
	const uint64_t cap    = LiveEnvU64("KYTY_BDA_LIVE_MAX_MB", 1024) << 20u;
	live->windows         = std::make_unique<BdaLiveWindows>(
        base, size, window, cap, [this, state](uint64_t host, uint64_t bytes) -> uint64_t {
            const auto start  = std::chrono::steady_clock::now();
            auto       buffer = std::make_unique<Buffer>(m_graphics, m_scheduler, reinterpret_cast<void*>(host), bytes,
                                                   ImportHostTag {});
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (!buffer->Handle()) {
                std::printf("Kyty BDA live pages: importing the backing window 0x%" PRIx64 " + %" PRIu64
                            " MiB failed\n",
                            host - LibKernel::Memory::GuestBackingBase(), bytes >> 20u);
                std::fflush(stdout);
                return 0;
            }
            const auto device = buffer->BufferDeviceAddress();
            state->imports.push_back(std::move(buffer));
            std::printf("Kyty BDA live pages: imported backing window 0x%" PRIx64 " + %" PRIu64
                        " MiB in %.1f ms (%zu windows)\n",
                        host - LibKernel::Memory::GuestBackingBase(), bytes >> 20u, ms, state->imports.size());
            std::fflush(stdout);
            return device;
        });
	if (!live->windows->Valid()) {
		std::printf("Kyty BDA live pages: refused, the backing (0x%" PRIx64 ", %" PRIu64
		            " bytes) or the window size breaks the 4 KiB import rule\n",
		            base, size);
		return;
	}
	m_bda_live   = std::move(live);
	g_live_cache = this;
	std::printf("Kyty BDA live pages: built (KYTY_BDA_LIVE_PAGES=%s), windows %" PRIu64 " MiB, cap %" PRIu64
	            " MiB, backing %" PRIu64 " MiB\n",
	            value, window >> 20u, cap >> 20u, size >> 20u);
	std::fflush(stdout);
}

int64_t BufferCache::BdaLivePagesMode() const {
	return m_bda_live != nullptr ? g_bda_live_pages.Get() : 0;
}

bool BufferCache::IsBdaLivePage(uint64_t vaddr) const {
	if (m_bda_live == nullptr) {
		return false;
	}
	std::scoped_lock lock(m_bda_live->mutex);
	return m_bda_live->pages.contains(vaddr >> CACHING_PAGEBITS);
}

vk::DeviceAddress BufferCache::LiveEntry(uint64_t page) const {
	if (m_bda_live == nullptr) {
		return 0;
	}
	vk::DeviceAddress entry = 0;
	{
		std::scoped_lock lock(m_bda_live->mutex);
		const auto       it = m_bda_live->pages.find(page);
		if (it == m_bda_live->pages.end()) {
			return 0;
		}
		entry = it->second;
	}
	return BdaTagAccessed(page) ? (entry | BDA_ACCESSED) : entry;
}

void BufferCache::LiveSyncBufferRange(BufferId id, uint64_t start, uint64_t size, BdaSyncStats* stats) {
	auto&      buffer = m_slot_buffers[id];
	auto&      live   = *m_bda_live;
	const auto end    = start + size;
	// Fast path: no CPU-dirty page in the range and no redirect in it: the plain synchronization
	// (which then finds nothing either).
	{
		bool cpu = true;
		if (m_memory_tracker.QueryCpuDirtyRelaxed(start, size, cpu) && !cpu) {
			bool any_live = false;
			{
				std::scoped_lock lock(live.mutex);
				const auto       it = live.pages.lower_bound(start >> CACHING_PAGEBITS);
				any_live = it != live.pages.end() && it->first <= ((end - 1) >> CACHING_PAGEBITS);
			}
			if (!any_live) {
				(void)SynchronizeBuffer(buffer, start, size, false, false, stats);
				return;
			}
		}
	}
	// Conservative: no redirect while early-released downloads are pending (their bytes are the GPU's).
	const bool may_redirect = m_early_released.empty();
	std::vector<std::pair<uint64_t, vk::DeviceAddress>> fresh;
	uint64_t run = start;
	for (auto page = start >> CACHING_PAGEBITS; (page << CACHING_PAGEBITS) < end; ++page) {
		const auto address = page << CACHING_PAGEBITS;
		bool       is_live = false;
		{
			std::scoped_lock lock(live.mutex);
			is_live = live.pages.contains(page);
		}
		vk::DeviceAddress entry = 0;
		if (!is_live && may_redirect && address >= buffer.CpuAddress() &&
		    address + PAGE <= buffer.CpuAddress() + buffer.Size()) {
			bool blocked = false;
			{
				std::scoped_lock lock(live.mutex);
				blocked = live.cp_written.contains(page);
				// Performance only: a page a cache binding uploaded after its redirect is read by draws
				// anyway; redirecting it again would only add two entry writes per frame.
				if (!blocked) {
					const auto h = live.bound_until.find(page);
					if (h != live.bound_until.end()) {
						if (live.passes < h->second) {
							blocked = true;
							live.totals.rejects_bound++;
						} else {
							live.bound_until.erase(h);
						}
					}
				}
			}
			const auto state = blocked ? MemoryTracker::DirtyState {} : m_memory_tracker.QueryDirty(address, PAGE);
			if (!blocked && state.cpu && !state.gpu && !BdaTagAccessed(page) &&
			    !m_memory_tracker.IsRegionHot(address, PAGE) && !HasGpuDirtyBytes(address, PAGE) &&
			    !HasPendingBackingPublication(address, PAGE) && !m_texture_cache.IsRegionGpuModified(address, PAGE)) {
				const auto* host = static_cast<const uint8_t*>(LibKernel::Memory::GuestBackingAlias(address, PAGE));
				// I-A2: the page and the 64 bytes after it lie in one imported window.
				entry = host != nullptr ? live.windows->Translate(reinterpret_cast<uint64_t>(host), PAGE + 64) : 0;
				if (entry == 0) {
					std::scoped_lock lock(live.mutex);
					live.totals.rejects_window++;
				}
			}
		}
		if (!is_live && entry == 0) {
			continue; // synchronized with its run
		}
		const auto seg_begin = std::max(address, start);
		const auto seg_end   = std::min(address + PAGE, end);
		if (seg_begin > run) {
			(void)SynchronizeBuffer(buffer, run, seg_begin - run, false, false, stats);
		}
		run = seg_end;
		if (!is_live) {
			fresh.emplace_back(page, entry);
		}
	}
	if (run < end) {
		(void)SynchronizeBuffer(buffer, run, end - run, false, false, stats);
	}
	if (fresh.empty()) {
		return;
	}
	{
		std::scoped_lock lock(live.mutex);
		for (const auto& [page, entry]: fresh) {
			live.pages.emplace(page, entry);
			live.Note(page, 1);
		}
		live.totals.redirects += fresh.size();
		m_bda_live_count.store(live.pages.size(), std::memory_order_relaxed);
	}
	// The entries, in stream order before the operation (consecutive pages in one write).
	std::vector<vk::DeviceAddress> run_entries;
	for (size_t first = 0; first < fresh.size();) {
		size_t last = first + 1;
		while (last < fresh.size() && fresh[last].first == fresh[last - 1].first + 1) {
			++last;
		}
		run_entries.clear();
		for (size_t k = first; k < last; ++k) {
			run_entries.push_back(fresh[k].second);
		}
		EnsureBdaTableResident(fresh[first].first, last - first);
		WriteDataBuffer(m_bda_pagetable_buffer, fresh[first].first * sizeof(vk::DeviceAddress), run_entries.data(),
		                run_entries.size() * sizeof(vk::DeviceAddress));
		first = last;
	}
}

void BufferCache::RestoreLivePages(uint64_t vaddr, uint64_t size, LiveRestore why) {
	if (m_bda_live == nullptr || size == 0) {
		return;
	}
	auto&                 live = *m_bda_live;
	std::vector<uint64_t> pages;
	{
		std::scoped_lock lock(live.mutex);
		const auto       first = vaddr >> CACHING_PAGEBITS;
		const auto       last  = (vaddr + size - 1) >> CACHING_PAGEBITS;
		for (auto it = live.pages.lower_bound(first); it != live.pages.end() && it->first <= last;) {
			pages.push_back(it->first);
			live.Note(it->first, static_cast<uint8_t>(2 + static_cast<uint8_t>(why)));
			if ((why == LiveRestore::Upload || why == LiveRestore::Written) && live.bound_passes != 0 &&
			    live.bound_until.size() < (1u << 20u)) {
				live.bound_until[it->first] = live.passes + live.bound_passes;
			}
			it = live.pages.erase(it);
		}
		m_bda_live_count.store(live.pages.size(), std::memory_order_relaxed);
		if (pages.empty()) {
			return;
		}
		auto& t = live.totals;
		switch (why) {
			case LiveRestore::Upload: t.restore_upload += pages.size(); break;
			case LiveRestore::Written: t.restore_written += pages.size(); break;
			case LiveRestore::Register: t.restore_register += pages.size(); break;
			case LiveRestore::Unmap: t.restore_unmap += pages.size(); break;
			case LiveRestore::Read: t.restore_read += pages.size(); break;
			case LiveRestore::Cp: t.restore_cp += pages.size(); break;
			case LiveRestore::Off: t.restore_off += pages.size(); break;
			case LiveRestore::Writes: t.restore_writes += pages.size(); break;
		}
		if (why != LiveRestore::Register && why != LiveRestore::Unmap) {
			// I3: CPU-dirty bytes may remain under the cache entry; uploaded first by the next full pass.
			for (const auto page: pages) {
				live.forced.Add(page << CACHING_PAGEBITS, PAGE);
			}
		}
	}
	if (why == LiveRestore::Register) {
		return; // ChangeRegister wrote these entries itself
	}
	std::vector<vk::DeviceAddress> entries;
	for (size_t first = 0; first < pages.size();) {
		size_t last = first + 1;
		while (last < pages.size() && pages[last] == pages[last - 1] + 1) {
			++last;
		}
		entries.clear();
		for (size_t k = first; k < last; ++k) {
			vk::DeviceAddress entry = 0;
			const auto*       owner = m_page_table.Find(pages[k]);
			if (owner != nullptr && *owner) {
				const auto& buffer = m_slot_buffers[*owner];
				entry              = buffer.BufferDeviceAddress() +
				        ((pages[k] - (buffer.CpuAddress() >> CACHING_PAGEBITS)) << CACHING_PAGEBITS);
			}
			entries.push_back(TaggedBdaEntry(pages[k], entry));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages[first] * sizeof(vk::DeviceAddress), entries.data(),
		                entries.size() * sizeof(vk::DeviceAddress));
		first = last;
	}
}

void BufferCache::LivePagesPrePass() {
	if (m_bda_live == nullptr) {
		return;
	}
	auto&                 live = *m_bda_live;
	const auto            mode = BdaLivePagesMode();
	std::vector<uint64_t> read_pages;
	std::vector<uint64_t> cp_pages;
	bool                  all = false;
	{
		std::scoped_lock lock(live.mutex);
		if (live.pages.empty() && live.forced.Empty()) {
			return;
		}
		all = !Redirects(mode) && !live.pages.empty();
		if (!all) {
			for (const auto& [page, entry]: live.pages) {
				if (live.cp_written.contains(page)) {
					cp_pages.push_back(page);
				}
			}
		}
	}
	if (all) {
		RestoreLivePages(0, uint64_t {1} << 40u, LiveRestore::Off);
	} else {
		if (m_bda_tags != nullptr) {
			std::vector<uint64_t> held;
			{
				std::scoped_lock lock(live.mutex);
				for (const auto& [page, entry]: live.pages) {
					held.push_back(page);
				}
			}
			for (const auto page: held) {
				if (BdaTagAccessed(page)) {
					read_pages.push_back(page);
				}
			}
			if (!read_pages.empty()) {
				std::scoped_lock lock(live.mutex);
				live.totals.live_reads += read_pages.size();
			}
		}
		for (const auto page: read_pages) {
			RestoreLivePages(page << CACHING_PAGEBITS, PAGE, LiveRestore::Read);
		}
		for (const auto page: cp_pages) {
			RestoreLivePages(page << CACHING_PAGEBITS, PAGE, LiveRestore::Cp);
		}
	}
	RangeSet forced;
	{
		std::scoped_lock lock(live.mutex);
		std::swap(forced, live.forced);
		if (!forced.Empty()) {
			live.totals.forced_syncs += forced.Size();
		}
	}
	if (forced.Empty()) {
		return;
	}
	// Only mapped pages are here: unmap drops its pages from the set (DropLivePages).
	const UploadBatch upload_batch(*this);
	forced.ForEach([this](uint64_t begin, uint64_t end) { SynchronizeBuffersInRange(begin, end - begin, nullptr); });
}

void BufferCache::LivePagesBeforeBdaWrites() {
	if (m_bda_live == nullptr) {
		return;
	}
	if (m_bda_live_count.load(std::memory_order_relaxed) != 0) {
		RestoreLivePages(0, uint64_t {1} << 40u, LiveRestore::Writes);
	}
	RangeSet forced;
	{
		std::scoped_lock lock(m_bda_live->mutex);
		std::swap(forced, m_bda_live->forced);
	}
	if (!forced.Empty()) {
		const UploadBatch upload_batch(*this);
		forced.ForEach([this](uint64_t begin, uint64_t end) { SynchronizeBuffersInRange(begin, end - begin, nullptr); });
	}
	std::scoped_lock lock(m_bda_live->mutex);
	m_bda_live->totals.v3 += m_bda_live->pages.size(); // must stay 0 (I-W1)
}

void BufferCache::DropLivePages(uint64_t vaddr, uint64_t size) {
	if (m_bda_live == nullptr || size == 0) {
		return;
	}
	// Off the GPU thread only at teardown (the GPU stops): the entries are not rewritten then.
	RestoreLivePages(vaddr, size, GuestGpu::IsGpuThread() ? LiveRestore::Unmap : LiveRestore::Register);
	std::scoped_lock lock(m_bda_live->mutex);
	m_bda_live->forced.Subtract(vaddr, size);
}

void BufferCache::LiveCpWriteHook(uint64_t vaddr, uint64_t size) {
	if (auto* cache = g_live_cache; cache != nullptr) {
		cache->LivePagesBeforeCpWrite(vaddr, size);
	}
}

void BufferCache::LivePagesBeforeCpWrite(uint64_t vaddr, uint64_t size) {
	if (m_bda_live == nullptr || size == 0 || m_bda_live_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	auto&    live = *m_bda_live;
	bool     hit  = false;
	uint64_t tick = 0;
	{
		std::scoped_lock lock(live.mutex);
		const auto       first = vaddr >> CACHING_PAGEBITS;
		const auto       last  = (vaddr + size - 1) >> CACHING_PAGEBITS;
		for (auto it = live.pages.lower_bound(first); it != live.pages.end() && it->first <= last; ++it) {
			hit = true;
			if (live.cp_written.size() < (1u << 20u)) {
				live.cp_written.insert(it->first); // never redirected again; restored at the next pre-pass
			}
		}
		if (!hit) {
			return;
		}
		live.totals.cp_hits++;
		tick = live.read_tick;
		if (!GuestGpu::IsGpuThread()) {
			live.totals.cp_unordered++;
			return;
		}
	}
	// I-C1: work recorded up to the last full pass may read the page live; it completes first.
	if (tick != 0 && tick <= m_scheduler.CurrentTick() && !m_scheduler.IsFree(tick)) {
		{
			std::scoped_lock lock(live.mutex);
			live.totals.cp_waits++;
		}
		m_scheduler.Wait(tick);
	}
}

void BufferCache::LivePagesAfterPass(const RangeSet& mapped_ranges, uint64_t fault_epoch_before) {
	auto&      live = *m_bda_live;
	const auto mode = BdaLivePagesMode();
	{
		std::scoped_lock lock(live.mutex);
		live.passes++;
		if (!live.pages.empty()) {
			live.read_tick = m_scheduler.CurrentTick();
		}
		if (mode != 0 && !m_bda_pending.empty()) {
			live.totals.blocked_passes++;
		}
	}
	if (mode != 2 && mode != 3) {
		return;
	}
	// V2 (I1): every redirect is still CPU-dirty-only and not hot / image-written.
	std::vector<uint64_t> held;
	{
		std::scoped_lock lock(live.mutex);
		for (const auto& [page, entry]: live.pages) {
			held.push_back(page);
		}
	}
	uint64_t v2 = 0;
	for (const auto page: held) {
		const auto address = page << CACHING_PAGEBITS;
		const auto state   = m_memory_tracker.QueryDirty(address, PAGE);
		if (state.gpu || m_memory_tracker.IsRegionHot(address, PAGE) || m_texture_cache.IsRegionGpuModified(address, PAGE)) {
			v2++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr, "BdaLiveVerify V2: redirected page 0x%" PRIx64 " is GPU-dirty, hot or image-written\n",
				             address);
			}
		}
	}
	// V1 (I2): only when no fault moved the epoch during the pass (racing guest writes are allowed).
	uint64_t v1      = 0;
	bool     checked = m_memory_tracker.FaultMutationEpoch() == fault_epoch_before;
	if (checked) {
		mapped_ranges.ForEach([&](uint64_t begin, uint64_t end) {
			auto it = m_buffers.upper_bound(begin);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < end; ++it) {
				const auto& buffer = m_slot_buffers[it->second];
				const auto  lo     = std::max(buffer.CpuAddress(), begin);
				const auto  hi     = std::min(buffer.CpuAddress() + buffer.Size(), end);
				if (lo >= hi) {
					continue;
				}
				bool any = true;
				if (m_memory_tracker.QueryCpuDirtyRelaxed(lo, hi - lo, any) && !any) {
					continue;
				}
				for (auto page = lo >> CACHING_PAGEBITS; (page << CACHING_PAGEBITS) < hi; ++page) {
					const auto a  = std::max(page << CACHING_PAGEBITS, lo);
					const auto b  = std::min((page + 1) << CACHING_PAGEBITS, hi);
					bool       cpu = false;
					if (!m_memory_tracker.QueryCpuDirtyRelaxed(a, b - a, cpu) || !cpu) {
						continue;
					}
					if (m_memory_tracker.IsRegionHot(page << CACHING_PAGEBITS, PAGE)) {
						continue;
					}
					std::scoped_lock lock(live.mutex);
					if (live.pages.contains(page) || live.forced.Intersects(page << CACHING_PAGEBITS, PAGE)) {
						continue;
					}
					v1++;
					static std::atomic<uint32_t> logged {0};
					if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
						// Diagnostics: the page's last live-pages event (none / redirected / restored by
						// cause), the pass kind, and the buffer around it.
						static const char* const kEvents[] = {"none",     "redirected", "restored upload",
						                                      "restored written", "restored register",
						                                      "restored unmap", "restored read", "restored cp",
						                                      "restored off", "restored writers"};
						const auto h     = live.history.find(page);
						const auto event = h != live.history.end() ? h->second : 0;
						std::fprintf(stderr,
						             "BdaLiveVerify V1: CPU-dirty page 0x%" PRIx64
						             " left on its cache entry after a full pass (mode %" PRId64 ", pass kind %u, last event %s, "
						             "buffer 0x%" PRIx64 " + 0x%" PRIx64 ")\n",
						             page << CACHING_PAGEBITS, mode, static_cast<uint32_t>(m_bda_pass_kind),
						             event < std::size(kEvents) ? kEvents[event] : "?", buffer.CpuAddress(), buffer.Size());
					}
				}
			}
		});
	}
	std::scoped_lock lock(live.mutex);
	live.totals.v2 += v2;
	live.totals.v1 += v1;
	if (checked) {
		live.totals.v1_checks++;
	} else {
		live.totals.v1_skipped++;
	}
}

BufferCache::BdaLiveTotals BufferCache::GetBdaLiveTotals() const {
	if (m_bda_live == nullptr) {
		return {};
	}
	std::scoped_lock lock(m_bda_live->mutex);
	auto             t = m_bda_live->totals;
	t.held             = m_bda_live->pages.size();
	t.faults           = m_bda_live_faults.load(std::memory_order_relaxed);
	t.windows          = m_bda_live->windows->Windows();
	t.pinned_bytes     = m_bda_live->windows->PinnedBytes();
	t.import_failures  = m_bda_live->windows->Failures();
	return t;
}

void BufferCache::ReportBdaLive() {
	auto& live = *m_bda_live;
	live.report_frames++;
	const auto now = std::chrono::steady_clock::now();
	if (live.report_time == std::chrono::steady_clock::time_point {}) {
		live.report_time   = now;
		live.reported      = GetBdaLiveTotals();
		live.report_frames = 0;
		return;
	}
	const auto elapsed = std::chrono::duration<double>(now - live.report_time).count();
	if (elapsed < 10.0) {
		return;
	}
	const auto   t = GetBdaLiveTotals();
	const auto&  r = live.reported;
	const double f = std::max<uint32_t>(live.report_frames, 1);
	const auto   mode = BdaLivePagesMode();
	const auto   per  = [f](uint64_t a, uint64_t b) { return static_cast<double>(a - b) / f; };
	std::printf("BdaLivePages %.0fs (%s): %u frames | per frame: redirects %.1f (%.2f MB not uploaded at redirect), "
	            "restores upload %.1f written %.1f read %.1f cp %.1f unmap %.1f register %.1f off %.1f writers %.1f, "
	            "forced syncs %.1f, faults %.1f | held %" PRIu64 " pages (%.1f MB) | cp hits %.2f/f waits %.2f/f "
	            "unordered %.2f/f, window rejects %.1f/f, bound rejects %.1f/f, blocked passes %.2f/f | windows %" PRIu64 " (%" PRIu64
	            " MiB pinned, %" PRIu64 " failed) | verify V1 %" PRIu64 " (checks %" PRIu64 ", skipped %" PRIu64
	            "), V2 %" PRIu64 ", V3 %" PRIu64 " (totals)\n",
	            elapsed, mode == 3 ? "check" : (mode == 2 ? "verify" : (mode == 1 ? "on" : "off")), live.report_frames,
	            per(t.redirects, r.redirects), per(t.redirects, r.redirects) * PAGE / 1e6,
	            per(t.restore_upload, r.restore_upload), per(t.restore_written, r.restore_written),
	            per(t.restore_read, r.restore_read), per(t.restore_cp, r.restore_cp),
	            per(t.restore_unmap, r.restore_unmap), per(t.restore_register, r.restore_register),
	            per(t.restore_off, r.restore_off), per(t.restore_writes, r.restore_writes),
	            per(t.forced_syncs, r.forced_syncs), per(t.faults, r.faults), t.held,
	            static_cast<double>(t.held * PAGE) / 1e6, per(t.cp_hits, r.cp_hits), per(t.cp_waits, r.cp_waits),
	            per(t.cp_unordered, r.cp_unordered), per(t.rejects_window, r.rejects_window),
	            per(t.rejects_bound, r.rejects_bound),
	            per(t.blocked_passes, r.blocked_passes), t.windows, t.pinned_bytes >> 20u, t.import_failures, t.v1,
	            t.v1_checks, t.v1_skipped, t.v2, t.v3);
	std::fflush(stdout);
	live.report_time   = now;
	live.reported      = t;
	live.report_frames = 0;
}

} // namespace Libs::Graphics
