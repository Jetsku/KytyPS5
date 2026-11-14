// KYTY_BDA_ACCESS_TAGS: the full BDA pass limited to the pages DMA shaders accessed. Based on
// upstream KytyPS5 0ec2cd5d5 ("re-queue written BDA pages only after a DMA shader accessed them"),
// adapted to our pass design (epochs, dirty log, hot runs, KYTY_BDA_REACH).
//
// Shaders built with the codegen option (CodegenOptions::bda_access_tags, the same switch set at
// start) read a backed page whose page-table entry lacks the accessed tag (bit 0) as it is on the
// GPU and set its bit in the fault bitmap; the fault scan tags the page (NoteBdaTagFault). Entries
// keep their tag when a buffer is registered again or a KYTY_BDA_REACH=verify trap is undone.
//
// Live policy (only with the codegen option; without it the switch reads 0):
//   0       the full pass (RenderContext::PrepareBda), as without tags; tags are still maintained.
//   1       the tagged pass instead: uploads the CPU writes of tagged pages and of every buffer
//           registered since the last pass, tagged or full (a new buffer's bytes were never
//           uploaded; a full pass uploads them and clears that set, NoteBdaTagFullPass). A
//           CPU-dirty page that no DMA shader has read stays CPU-dirty and stale on the GPU; its
//           first DMA read sees the stale bytes, reports the page, and the next pass uploads it.
//   verify  the full pass (correct bytes), and before it the CPU-dirty, backed, untagged pages
//           outside the pending registrations are recorded (would_skip): the pages the tagged pass
//           would have left stale at this point. A later DMA report of such a page is a miss: in
//           mode 1 that read would have seen stale bytes.
// What verify cannot see: a page dirtied with identical bytes counts as a miss (overcount); a
// page the GPU overwrote after the CPU write counts too; a would-skip page read by a non-DMA
// binding is not a tag concern (bindings synchronize themselves); pages dirtied between the
// record and the GPU's execution of the operation (the race every pass has); candidates come
// from the tracker's reach log plus the hot runs of the last full pass, so a reach-log overflow
// rescans every buffer (counted). Do not combine with KYTY_BDA_REACH=verify (both use the log
// and rewrite page-table entries).

#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/liveSwitch.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

namespace {

int64_t ParseBdaAccessTags(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}
Live::Switch g_bda_access_tags("KYTY_BDA_ACCESS_TAGS", ParseBdaAccessTags);

} // namespace

struct BufferCache::BdaTags {
	mutable std::mutex           mutex;
	RangeSet                     accessed;   // caching pages DMA shaders reported
	RangeSet                     pending;    // registered since the last tagged pass
	std::unordered_set<uint64_t> would_skip; // page indices (verify)
	RangeSet                     log;
	bool                         log_on = false;
	BdaTagTotals                 totals;
	BdaTagTotals                 reported;
	std::chrono::steady_clock::time_point report_time {};
	uint32_t                     report_calls = 0;
};

void BufferCache::CreateBdaTags() {
	if (ShaderRecompiler::GetCodegenOptions().bda_access_tags && m_bda_tags == nullptr) {
		m_bda_tags = std::make_shared<BdaTags>();
	}
}

int64_t BufferCache::BdaAccessTagsMode() const {
	return m_bda_tags != nullptr ? g_bda_access_tags.Get() : 0;
}

void BufferCache::ApplyBdaTags(uint64_t first_page, std::span<vk::DeviceAddress> entries, bool insert) {
	if (m_bda_tags == nullptr || entries.empty()) {
		return;
	}
	auto&            tags = *m_bda_tags;
	std::scoped_lock lock(tags.mutex);
	const auto       begin = first_page << CACHING_PAGEBITS;
	const auto       size  = static_cast<uint64_t>(entries.size()) << CACHING_PAGEBITS;
	tags.accessed.ForEachInRange(begin, size, [&](uint64_t first, uint64_t last) {
		for (auto page = first >> CACHING_PAGEBITS; page < ((last + CACHING_PAGESIZE - 1) >> CACHING_PAGEBITS); ++page) {
			auto& entry = entries[page - first_page];
			if (entry != 0) {
				entry |= BDA_ACCESSED;
			}
		}
	});
	if (insert) {
		// The next tagged pass uploads the new buffer's CPU-dirty pages: they were never uploaded.
		tags.pending.Add(begin, size);
		if (!tags.would_skip.empty()) {
			for (auto page = first_page; page < first_page + entries.size(); ++page) {
				tags.would_skip.erase(page);
			}
		}
	}
}

vk::DeviceAddress BufferCache::TaggedBdaEntry(uint64_t page, vk::DeviceAddress entry) const {
	// KYTY_BDA_LIVE_PAGES: a redirected page keeps its redirect (with its own tag).
	if (entry != 0 && m_bda_live_count.load(std::memory_order_relaxed) != 0) {
		if (const auto live = LiveEntry(page); live != 0) {
			return live;
		}
	}
	if (m_bda_tags == nullptr || entry == 0) {
		return entry;
	}
	std::scoped_lock lock(m_bda_tags->mutex);
	return m_bda_tags->accessed.Contains(page << CACHING_PAGEBITS, CACHING_PAGESIZE) ? (entry | BDA_ACCESSED) : entry;
}

bool BufferCache::BdaTagAccessed(uint64_t page) const {
	if (m_bda_tags == nullptr) {
		return false;
	}
	std::scoped_lock lock(m_bda_tags->mutex);
	return m_bda_tags->accessed.Contains(page << CACHING_PAGEBITS, CACHING_PAGESIZE);
}

void BufferCache::NoteBdaTagFault(uint64_t start, uint64_t size) {
	if (m_bda_tags == nullptr || size == 0) {
		return;
	}
	auto&                          tags = *m_bda_tags;
	std::vector<uint64_t>          pages;
	{
		std::scoped_lock lock(tags.mutex);
		for (auto page = start >> CACHING_PAGEBITS; page < ((start + size + CACHING_PAGESIZE - 1) >> CACHING_PAGEBITS) &&
		                                            page < CACHING_NUMPAGES;
		     ++page) {
			const auto address = page << CACHING_PAGEBITS;
			if (tags.accessed.Contains(address, CACHING_PAGESIZE)) {
				continue; // reported before its tag reached the GPU
			}
			tags.accessed.Add(address, CACHING_PAGESIZE);
			tags.totals.tagged_pages++;
			if (tags.would_skip.erase(page) != 0) {
				tags.totals.misses++;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
					std::fprintf(stderr,
					             "BdaTagVerify: a DMA shader read page 0x%" PRIx64
					             " that the tagged pass would have left stale\n",
					             address);
				}
			}
			pages.push_back(page);
		}
	}
	for (const auto page: pages) {
		LiveNoteTagRead(page); // KYTY_BDA_LIVE_PAGES (I5)
	}
	// The tagged entries of the backed pages (sorted: the loop runs upwards).
	std::vector<vk::DeviceAddress> entries;
	for (size_t first = 0; first < pages.size();) {
		size_t last = first + 1;
		while (last < pages.size() && pages[last] == pages[last - 1] + 1) ++last;
		entries.assign(last - first, 0);
		bool any = false;
		for (size_t k = first; k < last; ++k) {
			const auto* owner = m_page_table.Find(pages[k]);
			if (owner != nullptr && *owner) {
				const auto& buffer     = m_slot_buffers[*owner];
				entries[k - first] = (buffer.BufferDeviceAddress() +
				                      ((pages[k] - (buffer.CpuAddress() >> CACHING_PAGEBITS)) << CACHING_PAGEBITS)) |
				                     BDA_ACCESSED;
				// KYTY_BDA_LIVE_PAGES: a page read through its redirect keeps the redirect, now tagged
				// (LiveEntry adds the tag: the page is in the accessed set); the next full pass
				// uploads it (LivePagesPrePass sees it accessed).
				if (m_bda_live_count.load(std::memory_order_relaxed) != 0) {
					if (const auto live = LiveEntry(pages[k]); live != 0) {
						entries[k - first] = live;
					}
				}
				any = true;
			}
		}
		// Unbacked pages keep their zero entries (a registration tags them later).
		size_t run = 0;
		for (size_t k = 0; k <= entries.size(); ++k) {
			if (k < entries.size() && entries[k] != 0) {
				++run;
				continue;
			}
			if (run != 0) {
				WriteDataBuffer(m_bda_pagetable_buffer, (pages[first] + k - run) * sizeof(vk::DeviceAddress),
				                entries.data() + (k - run), run * sizeof(vk::DeviceAddress));
				run = 0;
			}
		}
		(void)any;
		first = last;
	}
}

uint64_t BufferCache::SynchronizeBdaTagged(const RangeSet& mapped_ranges) {
	auto&                                      tags = *m_bda_tags;
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	{
		std::scoped_lock lock(tags.mutex);
		const auto       collect = [&](uint64_t begin, uint64_t end) {
            mapped_ranges.ForEachInRange(begin, end - begin, [&](uint64_t first, uint64_t last) {
                spans.emplace_back(first, last);
            });
		};
		tags.accessed.ForEach(collect);
		tags.pending.ForEach(collect);
		tags.pending.Clear();
	}
	std::sort(spans.begin(), spans.end());
	size_t out = 0;
	for (const auto& span: spans) {
		if (out != 0 && span.first <= spans[out - 1].second) {
			spans[out - 1].second = std::max(spans[out - 1].second, span.second);
		} else {
			spans[out++] = span;
		}
	}
	spans.resize(out);
	BdaSyncStats stats;
	{
		const UploadBatch upload_batch(*this);
		for (const auto& [begin, end]: spans) {
			SynchronizeBuffersInRange(begin, end - begin, &stats);
		}
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
	std::scoped_lock lock(tags.mutex);
	tags.totals.tagged_passes++;
	tags.totals.tagged_bytes += stats.upload_bytes;
	return stats.upload_bytes;
}

void BufferCache::PrepareBdaTagVerify() {
	auto&            tags = *m_bda_tags;
	std::scoped_lock lock(tags.mutex);
	tags.totals.verify_passes++;
	RangeSet candidates;
	const auto all_buffers = [&] {
		for (const auto& [address, id]: m_buffers) {
			candidates.Add(address, m_slot_buffers[id].Size());
		}
	};
	if (!tags.log_on) {
		tags.log_on = true;
		m_memory_tracker.EnableReachLog(true);
		all_buffers(); // pages dirtied before the log started
	} else if (!m_memory_tracker.TakeReachLog(tags.log)) {
		tags.totals.log_overflows++;
		all_buffers();
	} else {
		tags.log.ForEach([&](uint64_t begin, uint64_t end) { candidates.Add(begin, end - begin); });
	}
	// Hot runs stay CPU-dirty after a full pass and are written without faults.
	for (const auto& range: m_bda_hot_ranges) {
		candidates.Add(range.address, range.size);
	}
	candidates.ForEach([&](uint64_t begin, uint64_t end) {
		for (auto page = begin >> CACHING_PAGEBITS; page < ((end + CACHING_PAGESIZE - 1) >> CACHING_PAGEBITS) && page < CACHING_NUMPAGES;
		     ++page) {
			const auto address = page << CACHING_PAGEBITS;
			if (tags.would_skip.contains(page) || tags.accessed.Contains(address, CACHING_PAGESIZE) ||
			    tags.pending.Contains(address, CACHING_PAGESIZE)) {
				continue;
			}
			const auto* owner = m_page_table.Find(page);
			if (owner == nullptr || !*owner) {
				continue;
			}
			if (m_memory_tracker.IsRegionCpuModified(address, CACHING_PAGESIZE)) {
				tags.would_skip.insert(page);
				tags.totals.would_skip++;
				static const bool trace = std::getenv("KYTY_BDA_TAGS_TRACE") != nullptr;
				if (trace) {
					std::fprintf(stderr, "BdaTagVerify: would skip page 0x%" PRIx64 "\n", address);
				}
			}
		}
	});
}

void BufferCache::NoteBdaTagFullPass() {
	if (m_bda_tags == nullptr) {
		return;
	}
	std::scoped_lock lock(m_bda_tags->mutex);
	m_bda_tags->pending.Clear();
}

BufferCache::BdaTagTotals BufferCache::GetBdaTagTotals() const {
	if (m_bda_tags == nullptr) {
		return {};
	}
	std::scoped_lock lock(m_bda_tags->mutex);
	return m_bda_tags->totals;
}

void BufferCache::ReportBdaTags() {
	// Every 10 s while the codegen option is on: one console line (counts since the last line).
	if (m_bda_tags == nullptr || (++m_bda_tags->report_calls & 255u) != 0) {
		return;
	}
	auto&      tags = *m_bda_tags;
	const auto now  = std::chrono::steady_clock::now();
	BdaTagTotals t;
	uint64_t     accessed_bytes = 0;
	size_t       would_skip     = 0;
	{
		std::scoped_lock lock(tags.mutex);
		if (tags.report_time == std::chrono::steady_clock::time_point {}) {
			tags.report_time = now;
			tags.reported    = tags.totals;
			return;
		}
		if (std::chrono::duration<double>(now - tags.report_time).count() < 10.0) {
			return;
		}
		t = tags.totals;
		tags.accessed.ForEach([&](uint64_t begin, uint64_t end) { accessed_bytes += end - begin; });
		would_skip = tags.would_skip.size();
	}
	const auto& r       = tags.reported;
	const auto  elapsed = std::chrono::duration<double>(now - tags.report_time).count();
	const auto  mode    = g_bda_access_tags.Get();
	auto& hops = s_tag_hops;
	std::printf("BdaTags hops (totals): programs with the check %" PRIu64 ", fault scans %" PRIu64 " recorded / %" PRIu64
	            " read, fault records %" PRIu64 ", full-pass prepares %" PRIu64
	            " | GPU get_bda_pointer calls %" PRIu64 ", zero entries %" PRIu64 ", untagged %" PRIu64 "\n",
	            hops.programs.load(), hops.scans.load(), hops.scans_read.load(), hops.records.load(),
	            hops.prepares.load(), hops.gpu_calls.load(), hops.gpu_zero.load(), hops.gpu_untagged.load());
	std::printf("BdaTags %.0fs (%s): tagged pages +%" PRIu64 " (accessed %.1f MB), tagged passes %" PRIu64
	            " uploading %.2f MB, verify passes %" PRIu64 " | verify: would-skip pages +%" PRIu64
	            " (open %zu), misses %" PRIu64 " (total %" PRIu64 "), log overflows %" PRIu64 "\n",
	            elapsed, mode == 2 ? "verify" : (mode == 1 ? "on" : "off"), t.tagged_pages - r.tagged_pages,
	            accessed_bytes / 1e6, t.tagged_passes - r.tagged_passes, (t.tagged_bytes - r.tagged_bytes) / 1e6,
	            t.verify_passes - r.verify_passes, t.would_skip - r.would_skip, would_skip, t.misses - r.misses,
	            t.misses, t.log_overflows - r.log_overflows);
	std::fflush(stdout);
	std::scoped_lock lock(tags.mutex);
	tags.report_time = now;
	tags.reported    = t;
}

} // namespace Libs::Graphics
