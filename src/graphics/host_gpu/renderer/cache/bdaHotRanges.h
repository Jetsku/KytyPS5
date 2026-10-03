#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace Libs::Graphics {

// Range has id (including generation), address and size. Invalid input remains unchanged.
template <typename Range>
bool MergeBdaHotRanges(std::vector<Range>& ranges) {
	for (const auto& range: ranges) {
		if (range.size == 0 || range.size > std::numeric_limits<uint64_t>::max() - range.address) {
			return false;
		}
	}
	std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) {
		return a.id == b.id ? a.address < b.address : a.id < b.id;
	});
	size_t kept = 0;
	for (size_t i = 0; i < ranges.size(); ++i) {
		const auto next = ranges[i];
		if (kept != 0) {
			auto& previous = ranges[kept - 1];
			const auto previous_end = previous.address + previous.size;
			if (previous.id == next.id && next.address <= previous_end) {
				previous.size = std::max(previous_end, next.address + next.size) - previous.address;
				continue;
			}
		}
		ranges[kept++] = next;
	}
	ranges.resize(kept);
	return true;
}

} // namespace Libs::Graphics
