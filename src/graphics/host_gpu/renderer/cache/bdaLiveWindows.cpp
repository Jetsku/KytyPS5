#include "graphics/host_gpu/renderer/cache/bdaLiveWindows.h"

#include <algorithm>
#include <utility>

namespace Libs::Graphics {

BdaLiveWindows::BdaLiveWindows(uint64_t backing_base, uint64_t backing_size, uint64_t window_bytes,
                               uint64_t cap_bytes, ImportFn import)
    : m_base(backing_base), m_size(backing_size), m_window_bytes(window_bytes), m_cap(cap_bytes),
      m_import(std::move(import)) {
	const bool power_of_two = window_bytes != 0 && (window_bytes & (window_bytes - 1)) == 0;
	m_valid = m_import != nullptr && backing_base != 0 && backing_size != 0 && power_of_two &&
	          window_bytes >= 65536 && backing_base % IMPORT_ALIGNMENT == 0 &&
	          backing_size % IMPORT_ALIGNMENT == 0;
}

uint64_t BdaLiveWindows::Translate(uint64_t host, uint64_t bytes) {
	if (!m_valid || bytes == 0 || host < m_base || host - m_base >= m_size || bytes > m_size - (host - m_base)) {
		m_range_rejects++;
		return 0;
	}
	const uint64_t offset = host - m_base;
	const uint64_t index  = offset / m_window_bytes;
	const uint64_t start  = index * m_window_bytes;
	const uint64_t size   = std::min(m_window_bytes, m_size - start);
	if (offset + bytes > start + size) {
		m_range_rejects++; // crosses the window's end: a read could run past the imported buffer
		return 0;
	}
	auto it = m_windows.find(index);
	if (it == m_windows.end()) {
		if (m_pinned + size > m_cap) {
			m_cap_rejects++;
			return 0;
		}
		Window window;
		window.size   = size;
		window.device = m_import(m_base + start, size);
		if (window.device == 0) {
			m_failures++;
		} else {
			m_imported++;
			m_pinned += size;
		}
		it = m_windows.emplace(index, window).first;
	}
	if (it->second.device == 0) {
		return 0;
	}
	return it->second.device + (offset - start);
}

BdaLiveEntryPlan PlanBdaLiveEntryWrites(std::span<const std::pair<uint64_t, uint64_t>> writes) {
	BdaLiveEntryPlan plan;
	if (writes.empty()) {
		return plan;
	}
	// (page, issue index): sorted by page, the last issue of each page kept.
	std::vector<std::pair<uint64_t, size_t>> order;
	order.reserve(writes.size());
	for (size_t k = 0; k < writes.size(); ++k) {
		order.emplace_back(writes[k].first, k);
	}
	std::sort(order.begin(), order.end());
	plan.data.reserve(order.size());
	for (size_t k = 0; k < order.size(); ++k) {
		if (k + 1 < order.size() && order[k + 1].first == order[k].first) {
			continue; // a later write of the same page follows
		}
		const auto page = order[k].first;
		if (!plan.runs.empty() && plan.runs.back().first_page + plan.runs.back().count == page) {
			plan.runs.back().count++;
		} else {
			plan.runs.push_back({page, 1, plan.data.size()});
		}
		plan.data.push_back(writes[order[k].second].second);
	}
	return plan;
}

} // namespace Libs::Graphics
