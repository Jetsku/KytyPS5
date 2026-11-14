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
//   I-B1 KYTY_BDA_LIVE_BATCH=1 (default, live): a read upload over a redirected page only settles
//        it (the guest bytes the entry reads are at least as new as the uploaded copy); settled
//        pages go back in one batch at the next pre-pass (verify V5: none of them GPU-written);
//   I-B2 entry writes of one flush (a pass's redirects, the pre-pass restores) are one copy; a
//        restore first flushes the queued redirects. KYTY_BDA_LIVE_BATCH=0: one copy per run, at once.
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
#include "graphics/host_gpu/watchRange.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
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
Live::Switch g_bda_live_batch("KYTY_BDA_LIVE_BATCH", Live::ParseDefaultOn);
Live::Switch g_bda_live_bound("KYTY_BDA_LIVE_BOUND_PASSES", [](const char* value) -> int64_t {
	return value != nullptr && value[0] != '\0' ? static_cast<int64_t>(std::min<uint64_t>(std::strtoull(value, nullptr, 0), 1u << 20u)) : 0;
});

BufferCache* g_live_cache = nullptr; // the cache whose state the StreamCopyDefer hooks reach

uint64_t LiveEnvU64(const char* name, uint64_t fallback) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
}

constexpr uint64_t PAGE = BufferCache::CACHING_PAGESIZE;
constexpr uint64_t SUBPAGES = PAGE / TRACKER_PAGE_SIZE;

using LiveWrites = std::vector<std::pair<uint64_t, vk::DeviceAddress>>;

} // namespace

struct BufferCache::BdaLive {
	struct Page {
		vk::DeviceAddress entry      = 0; // redirect entry (untagged)
		uint64_t          pass       = 0; // full-pass number of the redirect
		uint8_t           cpu_mask   = 0; // verify: CPU-dirty 4 KiB sub-pages at the redirect
	};
	mutable std::mutex                     mutex;
	std::map<uint64_t, Page>               pages;      // caching page -> redirect
	std::set<uint64_t>                     settled;    // redirected pages a read upload made current (I-B1)
	LiveWrites                             queued;     // redirect entries of the running pass (I-B2)
	std::unordered_set<uint64_t>           cp_written; // pages a CP write hit while redirected
	std::set<uint64_t>                     cp_pending;   // redirected pages a CP write hit (restored at the pre-pass)
	std::set<uint64_t>                     read_pending; // redirected pages a shader read (tag report)
	RangeSet                               forced;     // entries given back without an upload
	std::unique_ptr<BdaLiveWindows>        windows;
	std::vector<std::unique_ptr<Buffer>>   imports;
	uint64_t                               read_tick = 0; // last full pass that left redirects
	// Verify diagnostics: the last event of each page (1 = redirected, 2 + LiveRestore = restored).
	std::unordered_map<uint64_t, uint8_t>  history;
	// Performance only: pages a cache binding uploaded after their redirect are not redirected again
	// until this many full passes later (KYTY_BDA_LIVE_BOUND_PASSES, live, default 0 = off: with the
	// batched restores the churn costs one batched entry write per pass, while a blocked page is uploaded,
	// re-protected and faults again; Astro's Playroom: 35 rejects/f, faults 15 -> 19/f, 0.7 -> 1.5 MB/f).
	std::unordered_map<uint64_t, uint64_t> bound_until;
	uint64_t                               passes       = 0;
	// KYTY_BDA_LIVE_LOG=N (diagnostic, default 0): the first N full passes that redirect print one line
	// each (tick, new redirects as guest ranges, held, windows); imports name the page that made them.
	uint64_t                               log_left     = 0;
	std::vector<uint64_t>                  pass_new;      // pages the running pass redirected (log only)
	uint64_t                               trigger        = 0; // page address whose window is imported
	uint64_t                               trigger_buffer = 0;
	uint64_t                               trigger_size   = 0;
	uint64_t                               first_tick     = 0; // tick of the first pass that redirected
	uint64_t                               last_tick      = 0; // tick of the latest pass that redirected
	uint64_t                               last_pass_new  = 0;
	std::vector<uint64_t>                  last_new;      // pages the latest redirecting pass added (sorted)
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
                        " MiB in %.1f ms (%zu windows; device 0x%" PRIx64 ", for page 0x%" PRIx64
                        " of buffer 0x%" PRIx64 " + 0x%" PRIx64 ", tick %" PRIu64 ")\n",
                        host - LibKernel::Memory::GuestBackingBase(), bytes >> 20u, ms, state->imports.size(),
                        static_cast<uint64_t>(device), state->trigger, state->trigger_buffer, state->trigger_size,
                        m_scheduler.CurrentTick());
            std::fflush(stdout);
            return device;
        });
	if (!live->windows->Valid()) {
		std::printf("Kyty BDA live pages: refused, the backing (0x%" PRIx64 ", %" PRIu64
		            " bytes) or the window size breaks the 4 KiB import rule\n",
		            base, size);
		return;
	}
	live->log_left = LiveEnvU64("KYTY_BDA_LIVE_LOG", 0);
	m_bda_live   = std::move(live);
	g_live_cache = this;
	SetDeviceLossNote(&BufferCache::LiveDeviceLossNote);
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

bool BufferCache::IsBdaLivePageSettled(uint64_t vaddr) const {
	if (m_bda_live == nullptr) {
		return false;
	}
	std::scoped_lock lock(m_bda_live->mutex);
	return m_bda_live->settled.contains(vaddr >> CACHING_PAGEBITS);
}

void BufferCache::TestingMarkGpuDirty(uint64_t vaddr, uint64_t size, bool mark) {
	if (mark) {
		m_memory_tracker.MarkRegionAsGpuModified(vaddr, size);
	} else {
		m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	}
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
		entry = it->second.entry;
	}
	return BdaTagAccessed(page) ? (entry | BDA_ACCESSED) : entry;
}

void BufferCache::WriteLiveEntries(LiveWrites& writes, bool batch) {
	if (writes.empty()) {
		return;
	}
	// I-S1 (DESIGN.md 7): the entries go through the staging ring; an open reservation there (an
	// upload staged but not committed) would be overwritten by them. Every caller restores only after
	// the reservation is committed; V6 counts a regression (always on, one stderr line).
	if (m_staging_buffer.ReservationPending()) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			std::fprintf(stderr, "BdaLiveVerify V6: page-table entries written while a staging reservation is open\n");
		}
		std::scoped_lock lock(m_bda_live->mutex);
		m_bda_live->totals.v6++;
	}
	const auto plan      = PlanBdaLiveEntryWrites(writes);
	const bool rendering = m_scheduler.Current().ActiveRenderingSerial() != 0;
	for (const auto& run: plan.runs) {
		EnsureBdaTableResident(run.first_page, run.count);
	}
	constexpr uint64_t E     = sizeof(vk::DeviceAddress);
	const uint64_t     bytes = plan.data.size() * E;
	uint64_t           copies = 0;
	if (batch && plan.runs.size() > 1 && bytes <= m_staging_buffer.Size()) {
		const auto                  offset = m_staging_buffer.Copy(plan.data.data(), bytes, E);
		std::vector<vk::BufferCopy> regions;
		regions.reserve(plan.runs.size());
		for (const auto& run: plan.runs) {
			regions.emplace_back(offset + run.data_index * E, m_bda_pagetable_buffer.Offset(run.first_page * E),
			                     run.count * E);
		}
		m_bda_pagetable_buffer.CopyRegionsFrom(m_scheduler.Current(), m_staging_buffer, regions,
		                                       vk::AccessFlagBits::eHostWrite);
		copies = 1;
	} else {
		for (const auto& run: plan.runs) {
			WriteDataBuffer(m_bda_pagetable_buffer, run.first_page * E, plan.data.data() + run.data_index,
			                run.count * E);
			copies++;
		}
	}
	writes.clear();
	std::scoped_lock lock(m_bda_live->mutex);
	auto&            t = m_bda_live->totals;
	t.entry_copies += copies;
	t.entry_pages += plan.data.size();
	if (rendering) {
		t.entry_copies_rendering += copies;
	}
}

void BufferCache::FlushLiveRedirects() {
	if (m_bda_live == nullptr) {
		return;
	}
	LiveWrites queued;
	{
		std::scoped_lock lock(m_bda_live->mutex);
		if (m_bda_live->queued.empty()) {
			return;
		}
		std::swap(queued, m_bda_live->queued);
	}
	WriteLiveEntries(queued, true);
}

void BufferCache::LiveNoteTagRead(uint64_t page) {
	if (m_bda_live == nullptr || m_bda_live_count.load(std::memory_order_relaxed) == 0) {
		return;
	}
	std::scoped_lock lock(m_bda_live->mutex);
	if (m_bda_live->pages.contains(page)) {
		m_bda_live->read_pending.insert(page); // I5: uploaded at the next pre-pass
	}
}

void BufferCache::SettleLivePages(uint64_t vaddr, uint64_t size) {
	if (m_bda_live == nullptr || size == 0) {
		return;
	}
	if (g_bda_live_batch.Get() == 0) {
		RestoreLivePages(vaddr, size, LiveRestore::Upload);
		return;
	}
	auto&            live = *m_bda_live;
	std::scoped_lock lock(live.mutex);
	const auto       first = vaddr >> CACHING_PAGEBITS;
	const auto       last  = (vaddr + size - 1) >> CACHING_PAGEBITS;
	for (auto it = live.pages.lower_bound(first); it != live.pages.end() && it->first <= last; ++it) {
		if (live.settled.insert(it->first).second) {
			live.totals.settled++;
			if (WatchRange::Enabled()) {
				WatchRange::Note("live pages: settle", it->first << CACHING_PAGEBITS, PAGE);
			}
		}
	}
}

void BufferCache::LiveSyncBufferRange(BufferId id, uint64_t start, uint64_t size, BdaSyncStats* stats) {
	const auto t0         = std::chrono::steady_clock::now();
	auto&      buffer     = m_slot_buffers[id];
	auto&      live       = *m_bda_live;
	const auto end        = start + size;
	const auto first_page = start >> CACHING_PAGEBITS;
	const auto last_page  = (end - 1) >> CACHING_PAGEBITS;
	uint64_t   walked = 0, dirty = 0, segments = 0, bound = 0;
	const auto account = [&](bool fast) {
		const auto ns = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
		std::scoped_lock lock(live.mutex);
		auto&            t = live.totals;
		t.qual_calls++;
		t.qual_fast += fast ? 1 : 0;
		t.qual_pages += walked;
		t.qual_dirty += dirty;
		t.qual_segments += segments;
		t.qual_ns += ns;
		t.rejects_bound += bound;
	};
	// The redirected pages of the range, under one lock (only this thread changes the map).
	std::vector<uint64_t> held;
	{
		std::scoped_lock lock(live.mutex);
		for (auto it = live.pages.lower_bound(first_page); it != live.pages.end() && it->first <= last_page; ++it) {
			held.push_back(it->first);
		}
	}
	// Fast path: no CPU-dirty page in the range and no redirect in it: the plain synchronization
	// (which then finds nothing either).
	{
		bool cpu = true;
		if (held.empty() && m_memory_tracker.QueryCpuDirtyRelaxed(start, size, cpu) && !cpu) {
			(void)SynchronizeBuffer(buffer, start, size, false, false, stats);
			segments = 1;
			account(true);
			return;
		}
	}
	// Conservative: no redirect while early-released downloads are pending (their bytes are the GPU's).
	const bool may_redirect = m_early_released.empty();
	const bool verify       = BdaLivePagesMode() == 2;
	struct Fresh {
		uint64_t          page;
		vk::DeviceAddress entry;
		uint8_t           cpu_mask;
	};
	std::vector<Fresh> fresh;
	uint64_t           run = start;
	size_t             h   = 0;
	for (auto page = first_page; page <= last_page; ++page) {
		walked++;
		const auto address   = page << CACHING_PAGEBITS;
		const bool in_buffer = address >= buffer.CpuAddress() && address + PAGE <= buffer.CpuAddress() + buffer.Size();
		while (h < held.size() && held[h] < page) {
			++h;
		}
		const bool        is_live = h < held.size() && held[h] == page;
		vk::DeviceAddress entry   = 0;
		uint8_t           mask    = 0;
		if (!is_live && may_redirect && in_buffer) {
			// A page the lock-free mirror shows clean synchronizes with its run (nothing to upload).
			bool page_cpu = true;
			if (!(m_memory_tracker.QueryCpuDirtyRelaxed(address, PAGE, page_cpu) && !page_cpu)) {
				dirty++;
				bool blocked = false;
				{
					std::scoped_lock lock(live.mutex);
					blocked = live.cp_written.contains(page);
					// Performance only: a page a cache binding uploaded after its redirect is read by
					// draws anyway; redirecting it again would only add entry writes every frame.
					if (!blocked) {
						const auto b = live.bound_until.find(page);
						if (b != live.bound_until.end()) {
							if (live.passes < b->second) {
								blocked = true;
								bound++;
							} else {
								live.bound_until.erase(b);
							}
						}
					}
				}
				if (!blocked) {
					// Cheapest checks first: the lock-free dirty mirrors (exact on the GPU thread), then
					// the window (I-A2: the page and the 64 bytes after it lie in one imported window),
					// then the locked queries.
					MemoryTracker::DirtyState state;
					if (!m_memory_tracker.QueryDirtyRelaxed(address, PAGE, state)) {
						state = m_memory_tracker.QueryDirty(address, PAGE);
					}
					if (state.cpu && !state.gpu) {
						live.trigger        = address; // names the page in an import line
						live.trigger_buffer = buffer.CpuAddress();
						live.trigger_size   = buffer.Size();
						const auto* host =
						    static_cast<const uint8_t*>(LibKernel::Memory::GuestBackingAlias(address, PAGE));
						const auto dev =
						    host != nullptr ? live.windows->Translate(reinterpret_cast<uint64_t>(host), PAGE + 64) : 0;
						if (dev == 0) {
							std::scoped_lock lock(live.mutex);
							live.totals.rejects_window++;
						} else if (!BdaTagAccessed(page) && !m_memory_tracker.IsRegionHot(address, PAGE) &&
						           !HasGpuDirtyBytes(address, PAGE) && !HasPendingBackingPublication(address, PAGE) &&
						           !m_texture_cache.IsRegionGpuModified(address, PAGE)) {
							entry = dev;
							if (verify) {
								for (uint64_t k = 0; k < SUBPAGES; ++k) {
									bool sub = false;
									if (m_memory_tracker.QueryCpuDirtyRelaxed(address + k * TRACKER_PAGE_SIZE,
									                                          TRACKER_PAGE_SIZE, sub) &&
									    sub) {
										mask |= static_cast<uint8_t>(1u << k);
									}
								}
							}
						}
					}
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
			segments++;
		}
		run = seg_end;
		if (!is_live) {
			fresh.push_back({page, entry, mask});
		}
	}
	if (run < end) {
		(void)SynchronizeBuffer(buffer, run, end - run, false, false, stats);
		segments++;
	}
	if (!fresh.empty()) {
		// The entries, in stream order before the operation: queued for one write at the end of the
		// pass (I-B2), or written now per run of consecutive pages (KYTY_BDA_LIVE_BATCH=0).
		const bool batch = g_bda_live_batch.Get() != 0;
		LiveWrites writes;
		{
			std::scoped_lock lock(live.mutex);
			for (const auto& f: fresh) {
				live.pages.emplace(f.page, BdaLive::Page {f.entry, live.passes, f.cpu_mask});
				live.Note(f.page, 1);
				live.pass_new.push_back(f.page);
				if (batch) {
					live.queued.emplace_back(f.page, f.entry);
				} else {
					writes.emplace_back(f.page, f.entry);
				}
			}
			live.totals.redirects += fresh.size();
			m_bda_live_count.store(live.pages.size(), std::memory_order_relaxed);
		}
		if (WatchRange::Enabled()) {
			for (const auto& f: fresh) {
				WatchRange::Note("live pages: redirect", f.page << CACHING_PAGEBITS, PAGE);
			}
		}
		WriteLiveEntries(writes, false);
	}
	account(false);
}

void BufferCache::RestoreLivePages(uint64_t vaddr, uint64_t size, LiveRestore why, LiveWrites* writes) {
	if (m_bda_live == nullptr || size == 0) {
		return;
	}
	auto&                 live  = *m_bda_live;
	const auto            first = vaddr >> CACHING_PAGEBITS;
	const auto            last  = (vaddr + size - 1) >> CACHING_PAGEBITS;
	if (why == LiveRestore::Register) {
		// ChangeRegister rewrote these entries already: a queued redirect of them must not land after.
		std::scoped_lock lock(live.mutex);
		std::erase_if(live.queued, [&](const auto& w) { return w.first >= first && w.first <= last; });
	} else {
		FlushLiveRedirects(); // I-B2: a queued redirect never lands after the restore of its page
	}
	std::vector<uint64_t> pages;
	{
		std::scoped_lock lock(live.mutex);
		for (auto it = live.pages.lower_bound(first); it != live.pages.end() && it->first <= last;) {
			pages.push_back(it->first);
			live.Note(it->first, static_cast<uint8_t>(2 + static_cast<uint8_t>(why)));
			if ((why == LiveRestore::Upload || why == LiveRestore::Written) && g_bda_live_bound.Get() != 0 &&
			    live.bound_until.size() < (1u << 20u)) {
				live.bound_until[it->first] = live.passes + static_cast<uint64_t>(g_bda_live_bound.Get());
			}
			live.settled.erase(it->first);
			live.cp_pending.erase(it->first);
			live.read_pending.erase(it->first);
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
	if (WatchRange::Enabled()) {
		static const char* const kWhy[] = {"upload", "written", "register", "unmap", "read", "cp", "off", "writers"};
		const auto               index  = static_cast<size_t>(why);
		const std::string        what   = std::string("live pages: restore ") + (index < std::size(kWhy) ? kWhy[index] : "?");
		for (const auto page: pages) {
			WatchRange::Note(what.c_str(), page << CACHING_PAGEBITS, PAGE);
		}
	}
	if (why == LiveRestore::Register) {
		return; // ChangeRegister wrote these entries itself
	}
	LiveWrites local;
	auto&      out = writes != nullptr ? *writes : local;
	for (const auto page: pages) {
		vk::DeviceAddress entry = 0;
		const auto*       owner = m_page_table.Find(page);
		if (owner != nullptr && *owner) {
			const auto& buffer = m_slot_buffers[*owner];
			entry = buffer.BufferDeviceAddress() + ((page - (buffer.CpuAddress() >> CACHING_PAGEBITS)) << CACHING_PAGEBITS);
		}
		out.emplace_back(page, TaggedBdaEntry(page, entry));
	}
	if (writes == nullptr) {
		WriteLiveEntries(local, g_bda_live_batch.Get() != 0);
	}
}

uint64_t BufferCache::CheckSettledLivePages() {
	if (m_bda_live == nullptr) {
		return 0;
	}
	auto&                 live = *m_bda_live;
	std::vector<uint64_t> settled;
	{
		std::scoped_lock lock(live.mutex);
		settled.assign(live.settled.begin(), live.settled.end());
	}
	// V5 (I-B1): a settled page kept its redirect after a read upload; nothing may have written GPU
	// bytes over it since (a written binding restores at once).
	uint64_t v5 = 0;
	for (const auto page: settled) {
		const auto address = page << CACHING_PAGEBITS;
		const bool gpu     = m_memory_tracker.QueryDirty(address, PAGE).gpu || HasGpuDirtyBytes(address, PAGE);
		const bool image   = m_texture_cache.IsRegionGpuModified(address, PAGE);
		if (gpu || image) {
			v5++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr, "BdaLiveVerify V5: settled page 0x%" PRIx64 " is%s%s at the pre-pass\n", address,
				             gpu ? " GPU-dirty" : "", image ? " image-written" : "");
			}
		}
	}
	std::scoped_lock lock(live.mutex);
	live.totals.v5 += v5;
	live.totals.v5_checks++;
	return v5;
}

void BufferCache::LivePagesPrePass() {
	if (m_bda_live == nullptr) {
		return;
	}
	// Runs before every PrepareBda (about 200 per frame on Sky Garden with reach 0), so it only
	// touches pages that have something pending, never all redirected pages.
	FlushLiveRedirects();
	auto&                 live  = *m_bda_live;
	const auto            mode  = BdaLivePagesMode();
	const bool            batch = g_bda_live_batch.Get() != 0;
	std::vector<uint64_t> read_pages;
	std::vector<uint64_t> cp_pages;
	std::vector<uint64_t> settled;
	bool                  all = false;
	{
		std::scoped_lock lock(live.mutex);
		live.totals.prepass_calls++;
		all = !Redirects(mode) && !live.pages.empty();
		if (!all && live.settled.empty() && live.cp_pending.empty() && live.read_pending.empty() &&
		    live.forced.Empty()) {
			return;
		}
		live.totals.prepass_work++;
		if (!all) {
			cp_pages.assign(live.cp_pending.begin(), live.cp_pending.end());
			read_pages.assign(live.read_pending.begin(), live.read_pending.end());
			settled.assign(live.settled.begin(), live.settled.end());
			live.totals.live_reads += read_pages.size();
		}
	}
	// The restores of this pre-pass, written as one flush.
	LiveWrites writes;
	if (all) {
		RestoreLivePages(0, uint64_t {1} << 40u, LiveRestore::Off, &writes);
	} else {
		if (mode == 2 && !settled.empty()) {
			(void)CheckSettledLivePages();
		}
		for (const auto page: settled) {
			RestoreLivePages(page << CACHING_PAGEBITS, PAGE, LiveRestore::Upload, &writes);
		}
		for (const auto page: read_pages) {
			RestoreLivePages(page << CACHING_PAGEBITS, PAGE, LiveRestore::Read, &writes);
		}
		for (const auto page: cp_pages) {
			RestoreLivePages(page << CACHING_PAGEBITS, PAGE, LiveRestore::Cp, &writes);
		}
	}
	WriteLiveEntries(writes, batch);
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
	const auto epoch = m_memory_tracker.FaultMutationEpoch();
	{
		const UploadBatch upload_batch(*this);
		forced.ForEach([this](uint64_t begin, uint64_t end) { SynchronizeBuffersInRange(begin, end - begin, nullptr); });
	}
	if (BdaLivePagesMode() != 2 || m_memory_tracker.FaultMutationEpoch() != epoch) {
		return;
	}
	// V4 (I3): a forced page that is still CPU-dirty (not hot) after its upload, with no fault in
	// between, kept stale bytes under its cache entry.
	uint64_t v4 = 0;
	forced.ForEach([&](uint64_t begin, uint64_t end) {
		for (auto page = begin >> CACHING_PAGEBITS; (page << CACHING_PAGEBITS) < end; ++page) {
			const auto address = page << CACHING_PAGEBITS;
			bool       cpu     = false;
			const auto* owner   = m_page_table.Find(page);
			if (owner != nullptr && *owner && m_memory_tracker.QueryCpuDirtyRelaxed(address, PAGE, cpu) &&
			    cpu && !m_memory_tracker.IsRegionHot(address, PAGE) && !IsBdaLivePage(address)) {
				v4++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
					std::fprintf(stderr, "BdaLiveVerify V4: forced page 0x%" PRIx64 " still CPU-dirty after its upload\n",
					             address);
				}
			}
		}
	});
	std::scoped_lock lock(live.mutex);
	live.totals.v4 += v4;
	live.totals.v4_checks++;
}

void BufferCache::LivePagesBeforeBdaWrites() {
	if (m_bda_live == nullptr) {
		return;
	}
	FlushLiveRedirects();
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
			live.cp_pending.insert(it->first);
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
	FlushLiveRedirects(); // I-B2: the pass's redirect entries, before the operation is recorded
	auto&      live = *m_bda_live;
	const auto mode = BdaLivePagesMode();
	{
		std::scoped_lock lock(live.mutex);
		live.passes++;
		if (!live.pages.empty()) {
			live.read_tick = m_scheduler.CurrentTick();
		}
		if (!live.pass_new.empty()) {
			const auto tick = m_scheduler.CurrentTick();
			live.first_tick = live.first_tick != 0 ? live.first_tick : tick;
			live.last_tick     = tick;
			live.last_pass_new = live.pass_new.size();
			std::sort(live.pass_new.begin(), live.pass_new.end());
			live.last_new = live.pass_new;
			if (live.log_left != 0) {
				live.log_left--;
				std::string ranges;
				uint32_t    listed = 0;
				for (size_t k = 0; k < live.pass_new.size();) {
					size_t j = k + 1;
					while (j < live.pass_new.size() && live.pass_new[j] == live.pass_new[j - 1] + 1) {
						++j;
					}
					if (listed++ < 24u) {
						char text[64];
						std::snprintf(text, sizeof(text), " 0x%" PRIx64 "+0x%" PRIx64, live.pass_new[k] << CACHING_PAGEBITS,
						              static_cast<uint64_t>(j - k) << CACHING_PAGEBITS);
						ranges += text;
					}
					k = j;
				}
				std::printf("BdaLiveLog: pass %" PRIu64 " tick %" PRIu64 " kind %u: +%zu redirects in %u range(s)%s%s | held %zu, "
				            "windows %" PRIu64 " (%" PRIu64 " MiB)\n",
				            live.passes, tick, static_cast<uint32_t>(m_bda_pass_kind), live.pass_new.size(), listed,
				            ranges.c_str(), listed > 24u ? " ..." : "", live.pages.size(), live.windows->Windows(),
				            live.windows->PinnedBytes() >> 20u);
				std::fflush(stdout);
			}
			live.pass_new.clear();
		}
		if (mode != 0 && !m_bda_pending.empty()) {
			live.totals.blocked_passes++;
		}
	}
	if (mode != 2 && mode != 3) {
		return;
	}
	// V2 (I1): every redirect is still not GPU-dirty, not hot and not image-written. Split by cause;
	// hot alone is not a stale read (a hot page is CPU-dirty: the redirect reads the guest bytes).
	std::vector<std::pair<uint64_t, BdaLive::Page>> held;
	{
		std::scoped_lock lock(live.mutex);
		for (const auto& [page, info]: live.pages) {
			held.emplace_back(page, info);
		}
	}
	uint64_t v2 = 0, v2_gpu = 0, v2_hot = 0, v2_image = 0;
	for (const auto& [page, info]: held) {
		const auto address = page << CACHING_PAGEBITS;
		const auto state   = m_memory_tracker.QueryDirty(address, PAGE);
		const bool gpu     = state.gpu || HasGpuDirtyBytes(address, PAGE);
		const bool hot     = m_memory_tracker.IsRegionHot(address, PAGE);
		const bool image   = m_texture_cache.IsRegionGpuModified(address, PAGE);
		if (!gpu && !hot && !image) {
			continue;
		}
		v2++;
		v2_gpu += gpu ? 1 : 0;
		v2_hot += hot ? 1 : 0;
		v2_image += image ? 1 : 0;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
			// Diagnostics: the 4 KiB sub-page bits (bit k = sub-page k) now and at the redirect, the
			// passes since the redirect, settled, and the page's last live-pages event.
			uint32_t cpu_now = 0, hot_now = 0, gpu_now = 0;
			for (uint64_t k = 0; k < SUBPAGES; ++k) {
				const auto sub = address + k * TRACKER_PAGE_SIZE;
				const auto s   = m_memory_tracker.QueryDirty(sub, TRACKER_PAGE_SIZE);
				cpu_now |= s.cpu ? (1u << k) : 0u;
				gpu_now |= s.gpu ? (1u << k) : 0u;
				hot_now |= m_memory_tracker.IsRegionHot(sub, TRACKER_PAGE_SIZE) ? (1u << k) : 0u;
			}
			bool     settled = false;
			uint64_t passes  = 0;
			{
				std::scoped_lock lock(live.mutex);
				settled = live.settled.contains(page);
				passes  = live.passes - info.pass;
			}
			std::fprintf(stderr,
			             "BdaLiveVerify V2: redirected page 0x%" PRIx64 " is%s%s%s | sub-pages now cpu 0x%x gpu 0x%x hot 0x%x, "
			             "cpu at redirect 0x%x, %" PRIu64 " passes since the redirect, settled %d, tags %d\n",
			             address, gpu ? " GPU-dirty" : "", hot ? " hot" : "", image ? " image-written" : "", cpu_now,
			             gpu_now, hot_now, info.cpu_mask, passes, settled ? 1 : 0, BdaTagAccessed(page) ? 1 : 0);
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
	live.totals.v2_gpu += v2_gpu;
	live.totals.v2_hot += v2_hot;
	live.totals.v2_image += v2_image;
	live.totals.v1 += v1;
	if (checked) {
		live.totals.v1_checks++;
	} else {
		live.totals.v1_skipped++;
	}
}

void BufferCache::LiveDeviceLossNote() {
	auto* cache = g_live_cache;
	if (cache == nullptr || cache->m_bda_live == nullptr) {
		return;
	}
	auto&             live = *cache->m_bda_live;
	std::unique_lock lock(live.mutex, std::try_to_lock);
	if (!lock.owns_lock()) {
		std::printf("  BDA live pages: state locked by another thread at the loss (mode %" PRId64 ")\n",
		            cache->BdaLivePagesMode());
		return;
	}
	std::printf("  BDA live pages at the loss: mode %" PRId64 ", held %zu pages, settled %zu, passes %" PRIu64
	            ", first redirecting pass at tick %" PRIu64 ", latest at tick %" PRIu64 " (+%" PRIu64
	            "), last read tick %" PRIu64 ", windows %" PRIu64 " (%" PRIu64 " MiB), redirects %" PRIu64 " total\n",
	            cache->BdaLivePagesMode(), live.pages.size(), live.settled.size(), live.passes, live.first_tick,
	            live.last_tick, live.last_pass_new, live.read_tick, live.windows->Windows(),
	            live.windows->PinnedBytes() >> 20u, live.totals.redirects);
	{
		std::string ranges;
		uint32_t    runs = 0;
		for (size_t k = 0; k < live.last_new.size();) {
			size_t j = k + 1;
			while (j < live.last_new.size() && live.last_new[j] == live.last_new[j - 1] + 1) {
				++j;
			}
			if (runs++ < 64u) {
				char text[64];
				std::snprintf(text, sizeof(text), " 0x%" PRIx64 "+0x%" PRIx64, live.last_new[k] << CACHING_PAGEBITS,
				              static_cast<uint64_t>(j - k) << CACHING_PAGEBITS);
				ranges += text;
			}
			k = j;
		}
		std::printf("  latest redirecting pass (tick %" PRIu64 ") added %zu pages in %u run(s):%s%s\n", live.last_tick,
		            live.last_new.size(), runs, ranges.c_str(), runs > 64u ? " ..." : "");
	}
	uint32_t listed = 0;
	for (auto it = live.pages.begin(); it != live.pages.end() && listed < 96u;) {
		auto next = std::next(it);
		auto last = it;
		while (next != live.pages.end() && next->first == last->first + 1) {
			last = next++;
		}
		std::printf("    held 0x%" PRIx64 "+0x%" PRIx64 " -> device 0x%" PRIx64 "\n", it->first << CACHING_PAGEBITS,
		            (last->first - it->first + 1) << CACHING_PAGEBITS, static_cast<uint64_t>(it->second.entry));
		listed++;
		it = next;
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
	            " MiB pinned, %" PRIu64 " failed) | verify V1 (global, includes what the pass leaves without this "
	            "switch) %" PRIu64 " (checks %" PRIu64 ", skipped %" PRIu64 "), V2 %" PRIu64 ", V3 %" PRIu64
	            ", V4 %" PRIu64 " (checks %" PRIu64 ") (totals)\n",
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
	            t.v1_checks, t.v1_skipped, t.v2, t.v3, t.v4, t.v4_checks);
	// KYTY_BDA_LIVE_BATCH and the qualification walk (DESIGN.md 4.4). Entry copies are page-table
	// copy commands of this switch; "in rendering" ones ended an active rendering instance.
	std::printf("BdaLivePages batch (%s): per frame: entry copies %.2f (in rendering %.2f, %.1f entries), settled %.1f | "
	            "qualification calls %.1f (fast %.1f), pages walked %.1f, dirty candidates %.1f, sync segments %.1f, "
	            "%.3f ms in the walk incl. its synchronizations | pre-pass calls %.1f (with work %.1f) | V2 split gpu %" PRIu64 " image %" PRIu64 " hot %" PRIu64
	            ", V5 %" PRIu64 " (checks %" PRIu64 "), V6 %" PRIu64 " (totals)\n",
	            g_bda_live_batch.Get() != 0 ? "on" : "off", per(t.entry_copies, r.entry_copies),
	            per(t.entry_copies_rendering, r.entry_copies_rendering), per(t.entry_pages, r.entry_pages),
	            per(t.settled, r.settled), per(t.qual_calls, r.qual_calls), per(t.qual_fast, r.qual_fast),
	            per(t.qual_pages, r.qual_pages), per(t.qual_dirty, r.qual_dirty), per(t.qual_segments, r.qual_segments),
	            per(t.qual_ns, r.qual_ns) / 1e6, per(t.prepass_calls, r.prepass_calls), per(t.prepass_work, r.prepass_work), t.v2_gpu, t.v2_image, t.v2_hot, t.v5, t.v5_checks, t.v6);
	std::fflush(stdout);
	live.report_time   = now;
	live.reported      = t;
	live.report_frames = 0;
}

} // namespace Libs::Graphics
