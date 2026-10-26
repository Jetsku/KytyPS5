#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GPU_WRITE_RANGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GPU_WRITE_RANGES_H_

#include "graphics/host_gpu/renderer/image/keptRanges.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// KYTY_IMAGE_PARTIAL_GPU_WRITE (TextureCache::InvalidateMemoryFromGPU / UploadGpuWrittenRanges):
// the guest ranges of an image that GPU buffer writes replaced since its last refresh, and which
// parts of the image a refresh must rebuild from the buffer for them. Pure range arithmetic (no
// Vulkan), shared by the texture cache and gpu_write_ranges_tests.
namespace Libs::Graphics::GpuWriteRanges {

using KeptRanges::List;

// More separate ranges than this make the refresh whole: their bookkeeping and the per-range
// materialisation would cost more than they save.
constexpr size_t kMaxRanges = 64;

// Records a buffer write of [address, address + size) clipped to the image's bytes
// [image_begin, image_end). False when the ranges overflow kMaxRanges (the caller then refreshes
// the image whole); a write that misses the image changes nothing.
[[nodiscard]] inline bool Record(List& ranges, uint64_t address, uint64_t size, uint64_t image_begin,
                                 uint64_t image_end) {
	if (size == 0 || address >= image_end || image_begin >= image_end) {
		return true;
	}
	const auto end   = address + size < address ? image_end : std::min(address + size, image_end);
	const auto begin = std::max(address, image_begin);
	if (begin >= end) {
		return true;
	}
	(void)KeptRanges::Add(ranges, begin, end);
	return ranges.size() <= kMaxRanges;
}

// [begin, end) minus `ranges` (sorted, disjoint).
[[nodiscard]] inline List Complement(const List& ranges, uint64_t begin, uint64_t end) {
	List out;
	auto at = begin;
	for (const auto& [first, second]: ranges) {
		if (second <= at) {
			continue;
		}
		if (first >= end) {
			break;
		}
		if (first > at) {
			out.emplace_back(at, first);
		}
		at = std::max(at, second);
		if (at >= end) {
			break;
		}
	}
	if (at < end) {
		out.emplace_back(at, end);
	}
	return out;
}

// A 2D tiled subresource outside the mip tail: `rows` block rows of `row_bytes` bytes each from
// guest address `tiled_begin`. Block families address a texel as (block row * blocks per row +
// block column) * block size plus an offset that depends only on its position inside the block,
// so a run of block rows is an independent sub-surface. Returns the runs [first, last) of block
// rows whose bytes intersect `ranges`, merged when adjacent.
[[nodiscard]] inline std::vector<std::pair<uint32_t, uint32_t>>
DirtyRowRuns(const List& ranges, uint64_t tiled_begin, uint64_t row_bytes, uint32_t rows) {
	std::vector<std::pair<uint32_t, uint32_t>> runs;
	if (row_bytes == 0 || rows == 0) {
		return runs;
	}
	const auto tiled_end = tiled_begin + row_bytes * rows;
	for (const auto& [first, second]: ranges) {
		if (second <= tiled_begin || first >= tiled_end) {
			continue;
		}
		const auto begin = std::max(first, tiled_begin) - tiled_begin;
		const auto end   = std::min(second, tiled_end) - tiled_begin;
		const auto r0    = static_cast<uint32_t>(begin / row_bytes);
		const auto r1    = static_cast<uint32_t>((end + row_bytes - 1) / row_bytes);
		if (!runs.empty() && r0 <= runs.back().second) {
			runs.back().second = std::max(runs.back().second, r1);
		} else {
			runs.emplace_back(r0, r1);
		}
	}
	return runs;
}

} // namespace Libs::Graphics::GpuWriteRanges

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_GPU_WRITE_RANGES_H_
