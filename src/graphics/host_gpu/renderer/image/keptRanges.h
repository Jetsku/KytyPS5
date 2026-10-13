#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_KEPT_RANGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_KEPT_RANGES_H_

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// KYTY_IMAGE_GPU_KEEP (TextureCache::KeepGpuBytes): the guest ranges of a GPU-written image that a
// CPU write or a partial unmap replaced, while the image's other bytes stay what the GPU rendered.
// Pure range arithmetic (no Vulkan), shared by the texture cache and its tests.
namespace Libs::Graphics::KeptRanges {

// [first, second): sorted, disjoint, never empty, never adjacent (Add merges touching ranges).
using Range = std::pair<uint64_t, uint64_t>;
using List  = std::vector<Range>;

// Adds [begin, end); returns how many of its bytes were not covered before.
inline uint64_t Add(List& ranges, uint64_t begin, uint64_t end) {
	if (begin >= end) {
		return 0;
	}
	uint64_t covered = 0;
	List     merged;
	merged.reserve(ranges.size() + 1);
	auto   new_begin = begin;
	auto   new_end   = end;
	size_t index     = 0;
	for (; index < ranges.size() && ranges[index].second < begin; index++) {
		merged.push_back(ranges[index]);
	}
	for (; index < ranges.size() && ranges[index].first <= end; index++) {
		const auto& range = ranges[index];
		// The overlap (0 when the range only touches [begin, end)).
		covered += std::min(range.second, end) - std::max(range.first, begin);
		new_begin = std::min(new_begin, range.first);
		new_end   = std::max(new_end, range.second);
	}
	merged.emplace_back(new_begin, new_end);
	for (; index < ranges.size(); index++) {
		merged.push_back(ranges[index]);
	}
	ranges = std::move(merged);
	return (end - begin) - covered;
}

// Removes [begin, end).
inline void Subtract(List& ranges, uint64_t begin, uint64_t end) {
	if (begin >= end) {
		return;
	}
	List kept;
	kept.reserve(ranges.size() + 1);
	for (const auto& range: ranges) {
		if (range.second <= begin || range.first >= end) {
			kept.push_back(range);
			continue;
		}
		if (range.first < begin) {
			kept.emplace_back(range.first, begin);
		}
		if (end < range.second) {
			kept.emplace_back(end, range.second);
		}
	}
	ranges = std::move(kept);
}

[[nodiscard]] inline bool Intersects(const List& ranges, uint64_t begin, uint64_t end) {
	if (begin >= end) {
		return false;
	}
	const auto found = std::lower_bound(ranges.begin(), ranges.end(), begin,
	                                    [](const Range& range, uint64_t value) { return range.second <= value; });
	return found != ranges.end() && found->first < end;
}

[[nodiscard]] inline uint64_t Bytes(const List& ranges) {
	uint64_t total = 0;
	for (const auto& range: ranges) {
		total += range.second - range.first;
	}
	return total;
}

// The pages a CPU write to [address, address + size) gives up in an image at [image_begin,
// image_end): the write widened to `granule` (a power of two, a multiple of `page`), clipped to the
// pages the image lies on. {0, 0} when the write misses those pages.
[[nodiscard]] inline Range WriteSpan(uint64_t address, uint64_t size, uint64_t image_begin,
                                     uint64_t image_end, uint64_t granule, uint64_t page) {
	if (size == 0 || image_begin >= image_end || address >= UINT64_MAX - size) {
		return {0, 0};
	}
	const auto pages_begin = image_begin & ~(page - 1);
	const auto pages_end   = (image_end + page - 1) & ~(page - 1);
	const auto begin       = std::max(address & ~(granule - 1), pages_begin);
	const auto end         = std::min((address + size + granule - 1) & ~(granule - 1), pages_end);
	if (begin >= end) {
		return {0, 0};
	}
	return {begin, end};
}

// The layers of an array of `layers` slices of `slice` bytes from `base` whose bytes intersect
// `ranges`.
[[nodiscard]] inline std::vector<uint32_t> DirtyLayers(const List& ranges, uint64_t base, uint64_t slice,
                                                       uint32_t layers) {
	std::vector<uint32_t> picked;
	for (uint32_t layer = 0; layer < layers; layer++) {
		if (Intersects(ranges, base + slice * layer, base + slice * (layer + 1))) {
			picked.push_back(layer);
		}
	}
	return picked;
}

} // namespace Libs::Graphics::KeptRanges

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_KEPT_RANGES_H_
