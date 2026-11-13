// KYTY_BDA_REACH: footprint-limited synchronization of the buffers that shaders read through
// computed addresses (BDA). Based on TheCruZ/KytyPS5-GTA (RenderContext::SynchronizeDmaFootprint,
// BufferCache::SynchronizeBuffersInSpans), adapted: the reach comes from the interval tape of the
// write-range analysis (WriteRangeEvaluator::DmaSpans), and a reach that is not proven for the
// draw's inputs falls back to the full pass.
//
// Production (KYTY_BDA_REACH=1): the CPU writes of the buffers inside the reach are uploaded; a
// CPU-dirty page outside it stays CPU-dirty, unprotected and present in the page table. It is
// stale on the GPU, and no operation reads it: every limited operation's reads lie inside its own
// reach (the analysis), and every other operation takes the full pass first, which uploads it.
//
// Verification (KYTY_BDA_REACH=verify), on top of production:
//  (a) After the limited upload, the same spans are scanned again with the fault epoch taken
//      before the upload: a normal (not hot) tracker page still CPU-dirty inside the reach although
//      no write fault happened since is a page the limited upload missed (inside_dirty, must be 0).
//  (b) Traps: every caching page that is CPU-dirty, has a registered buffer and lies outside the
//      reach gets a zero page-table entry, recorded in the command stream before the operation. A
//      shader read of it then records a GPU page fault (the existing fault scan) and reads zero
//      instead of stale bytes; such faults are counted (trap_faults, must be 0). A limited
//      operation whose reach covers a trapped page uploads it and gives its entry back first; a
//      full pass gives every entry back. The candidates are the ranges the tracker's reach log
//      reports dirtied (write faults, new regions, demotions), the ranges of buffers registered
//      since, the previous limited operation's reach (hot pages stay CPU-dirty after an upload),
//      and after a full pass the hot runs it found.
// What verification cannot see: pages a guest thread dirties between the trap scan and the GPU's
// execution of the operation (the same race the full pass has), and stale bytes of a hot page the
// guest rewrites after the previous operation's reach covered it, until the next trap scan.

#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <unordered_set>

namespace Libs::Graphics {

struct BufferCache::BdaReachVerify {
	bool                   active = false;
	// Caching pages whose page-table entry the verification zeroed.
	std::set<uint64_t>     trapped;
	// Every page trapped since its buffer was registered (faults on them are counted).
	std::mutex             fault_mutex;
	std::unordered_set<uint64_t> ever_trapped;
	// Candidates for the next trap scan.
	RangeSet               candidates;
	RangeSet               log;
	std::vector<std::pair<uint64_t, uint64_t>> previous;
	BdaReachVerifyTotals   totals;
};

uint64_t BufferCache::SynchronizeBdaReach(std::span<const std::pair<uint64_t, uint64_t>> spans, bool verify) {
	BdaSyncStats stats;
	stats.reach = true;
	// Before the upload: a page that turns CPU-dirty after it advances this epoch.
	const auto fault_epoch     = m_memory_tracker.FaultMutationEpoch();
	const auto structure_epoch = m_bda_structure_epoch.load(std::memory_order_acquire);
	{
		const UploadBatch upload_batch(*this);
		for (const auto& [begin, end]: spans) {
			SynchronizeBuffersInRange(begin, end - begin, &stats);
		}
	}
	m_bda_pass_upload_bytes += stats.upload_bytes;
	if (!verify) {
		RestoreBdaReachTraps(false);
		return stats.upload_bytes;
	}
	if (m_bda_reach == nullptr) {
		m_bda_reach = std::make_shared<BdaReachVerify>();
	}
	auto& state = *m_bda_reach;
	if (!state.active) {
		state.active = true;
		m_memory_tracker.EnableReachLog(true);
		// Pages dirtied before the log started: the first scan covers every mapped buffer.
		for (const auto& [address, id]: m_buffers) {
			state.candidates.Add(address, m_slot_buffers[id].Size());
		}
	}
	state.totals.checks++;
	// (a) The spans again, with the epochs taken before the upload (it uploads what it finds).
	if (fault_epoch != 0 && fault_epoch != UINT64_MAX) {
		BdaSyncStats check;
		check.reach                  = true;
		check.verify_fault_epoch     = fault_epoch;
		check.verify_structure_epoch = structure_epoch;
		{
			const UploadBatch upload_batch(*this);
			for (const auto& [begin, end]: spans) {
				SynchronizeBuffersInRange(begin, end - begin, &check);
			}
		}
		state.totals.inside_dirty += check.verify_mismatch_pages;
		if (check.verify_mismatch_pages != 0) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr, "BdaReachVerify: %" PRIu64 " CPU-dirty page(s) left inside the reach\n",
				             check.verify_mismatch_pages);
			}
		}
	}
	// (b) Entries of the reach come back (their pages were just uploaded).
	const auto in_spans = [&](uint64_t page) {
		const auto address = page << CACHING_PAGEBITS;
		const auto found   = std::upper_bound(spans.begin(), spans.end(), address,
		                                      [](uint64_t value, const auto& span) { return value < span.second; });
		return found != spans.end() && found->first <= address;
	};
	std::vector<uint64_t> restore;
	for (const auto& [begin, end]: spans) {
		for (auto it = state.trapped.lower_bound(begin >> CACHING_PAGEBITS);
		     it != state.trapped.end() && *it < (end >> CACHING_PAGEBITS);) {
			restore.push_back(*it);
			it = state.trapped.erase(it);
		}
	}
	const auto write_entries = [&](std::span<const uint64_t> pages, bool zero) {
		std::vector<vk::DeviceAddress> addresses;
		for (size_t first = 0; first < pages.size();) {
			size_t last = first + 1;
			while (last < pages.size() && pages[last] == pages[last - 1] + 1) ++last;
			const auto run = last - first;
			if (zero) {
				m_bda_pagetable_buffer.Fill(pages[first] * sizeof(vk::DeviceAddress),
				                            run * sizeof(vk::DeviceAddress), 0);
			} else {
				addresses.assign(run, 0);
				for (size_t k = 0; k < run; ++k) {
					const auto* entry = m_page_table.Find(pages[first + k]);
					if (entry != nullptr && *entry) {
						const auto& buffer = m_slot_buffers[*entry];
						addresses[k]       = TaggedBdaEntry(pages[first + k], buffer.BufferDeviceAddress() +
						               ((pages[first + k] - (buffer.CpuAddress() >> CACHING_PAGEBITS)) << CACHING_PAGEBITS));
					}
				}
				WriteDataBuffer(m_bda_pagetable_buffer, pages[first] * sizeof(vk::DeviceAddress), addresses.data(),
				                run * sizeof(vk::DeviceAddress));
			}
			first = last;
		}
	};
	std::sort(restore.begin(), restore.end());
	write_entries(restore, false);
	state.totals.restored += restore.size();
	// Candidates: the reach log, buffers registered since (NoteBdaReachRegister), the previous reach.
	if (!m_memory_tracker.TakeReachLog(state.log)) {
		state.totals.log_overflows++;
		for (const auto& [address, id]: m_buffers) {
			state.candidates.Add(address, m_slot_buffers[id].Size());
		}
	}
	state.log.ForEach([&](uint64_t begin, uint64_t end) { state.candidates.Add(begin, end - begin); });
	for (const auto& [begin, end]: state.previous) {
		state.candidates.Add(begin, end - begin);
	}
	state.previous.assign(spans.begin(), spans.end());
	std::vector<uint64_t> trap;
	state.candidates.ForEach([&](uint64_t begin, uint64_t end) {
		const uint64_t first = begin >> CACHING_PAGEBITS;
		const uint64_t last  = (end + CACHING_PAGESIZE - 1) >> CACHING_PAGEBITS;
		for (uint64_t page = first; page < last && page < CACHING_NUMPAGES; ++page) {
			if (in_spans(page) || state.trapped.contains(page)) {
				continue;
			}
			const auto* entry = m_page_table.Find(page);
			if (entry == nullptr || !*entry) {
				continue;
			}
			if (m_memory_tracker.IsRegionCpuModified(page << CACHING_PAGEBITS, CACHING_PAGESIZE)) {
				trap.push_back(page);
			}
		}
	});
	state.candidates.Clear();
	write_entries(trap, true);
	for (const auto page: trap) {
		state.trapped.insert(page);
	}
	{
		std::scoped_lock lock(state.fault_mutex);
		state.ever_trapped.insert(trap.begin(), trap.end());
	}
	{
		std::scoped_lock lock(state.fault_mutex);
		state.totals.trapped += trap.size();
	}
	return stats.upload_bytes;
}

void BufferCache::RestoreBdaReachTraps(bool keep_verifying) {
	if (m_bda_reach == nullptr || !m_bda_reach->active) {
		return;
	}
	auto& state = *m_bda_reach;
	if (!keep_verifying) {
		// The switch left verify: every entry back, the log off.
		m_memory_tracker.EnableReachLog(false);
		state.active = false;
		state.candidates.Clear();
		state.previous.clear();
	} else {
		// After a full pass the hot runs stay CPU-dirty: the next scan looks at them again.
		for (const auto& range: m_bda_hot_ranges) {
			state.candidates.Add(range.address, range.size);
		}
	}
	if (state.trapped.empty()) {
		return;
	}
	std::vector<uint64_t> pages(state.trapped.begin(), state.trapped.end());
	state.trapped.clear();
	std::vector<vk::DeviceAddress> addresses;
	for (size_t first = 0; first < pages.size();) {
		size_t last = first + 1;
		while (last < pages.size() && pages[last] == pages[last - 1] + 1) ++last;
		addresses.assign(last - first, 0);
		for (size_t k = first; k < last; ++k) {
			const auto* entry = m_page_table.Find(pages[k]);
			if (entry != nullptr && *entry) {
				const auto& buffer   = m_slot_buffers[*entry];
				addresses[k - first] = TaggedBdaEntry(pages[k], buffer.BufferDeviceAddress() +
				                       ((pages[k] - (buffer.CpuAddress() >> CACHING_PAGEBITS)) << CACHING_PAGEBITS));
			}
			// A trapped page is CPU-dirty: a later scan may trap it again.
			if (state.active) {
				state.candidates.Add(pages[k] << CACHING_PAGEBITS, CACHING_PAGESIZE);
			}
		}
		WriteDataBuffer(m_bda_pagetable_buffer, pages[first] * sizeof(vk::DeviceAddress), addresses.data(),
		                (last - first) * sizeof(vk::DeviceAddress));
		first = last;
	}
	state.totals.restored += pages.size();
}

void BufferCache::NoteBdaReachRegister(uint64_t first_page, uint64_t last_page, bool insert) {
	if (m_bda_reach == nullptr || !m_bda_reach->active) {
		return;
	}
	auto& state = *m_bda_reach;
	// ChangeRegister rewrote these entries: they are present (or zero for a removed buffer) and
	// not the verification's any more; a new buffer's pages start CPU-dirty.
	for (auto it = state.trapped.lower_bound(first_page); it != state.trapped.end() && *it < last_page;) {
		it = state.trapped.erase(it);
	}
	if (insert) {
		state.candidates.Add(first_page << CACHING_PAGEBITS, (last_page - first_page) << CACHING_PAGEBITS);
	} else {
		std::scoped_lock lock(state.fault_mutex);
		for (auto page = first_page; page < last_page; ++page) {
			state.ever_trapped.erase(page);
		}
	}
}

void BufferCache::NoteBdaReachFault(uint64_t start, uint64_t size) {
	if (m_bda_reach == nullptr || size == 0) {
		return;
	}
	auto&            state = *m_bda_reach;
	std::scoped_lock lock(state.fault_mutex);
	uint64_t         hits = 0;
	for (uint64_t page = start >> CACHING_PAGEBITS; page < (start + size + CACHING_PAGESIZE - 1) >> CACHING_PAGEBITS;
	     ++page) {
		if (state.ever_trapped.contains(page)) {
			hits++;
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
				std::fprintf(stderr, "BdaReachVerify: a shader read trapped page 0x%" PRIx64 " outside its reach\n",
				             page << CACHING_PAGEBITS);
			}
		}
	}
	state.totals.trap_faults += hits;
}

BufferCache::BdaReachVerifyTotals BufferCache::GetBdaReachVerifyTotals() const {
	if (m_bda_reach == nullptr) {
		return {};
	}
	std::scoped_lock lock(m_bda_reach->fault_mutex);
	return m_bda_reach->totals;
}

bool BufferCache::ReadGuestForBdaReach(uint64_t address, std::span<uint32_t> words) {
	const auto bytes = words.size_bytes();
	if ((address & 3u) != 0u || bytes == 0 || IsRegionGpuModified(address, bytes)) {
		return false;
	}
	const auto* host = LibKernel::Memory::GuestBackingAlias(address, bytes);
	if (host == nullptr) {
		return false;
	}
	std::memcpy(words.data(), host, bytes);
	return true;
}

} // namespace Libs::Graphics
