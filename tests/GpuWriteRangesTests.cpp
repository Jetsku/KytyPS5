// KYTY_IMAGE_PARTIAL_GPU_WRITE range bookkeeping (image/gpuWriteRanges.h): recording GPU buffer
// writes per image, the complement a partial rebuild must prove unchanged, and the block-row runs
// of a tiled subresource that a partial rebuild refreshes.
#include "graphics/host_gpu/renderer/image/gpuWriteRanges.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace Libs::Graphics;
using GpuWriteRanges::List;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

constexpr uint64_t kImage = 0x530000000ull;
constexpr uint64_t kSize  = 4096ull * 4096ull * 16ull; // 16 layers of 16 MiB

void TestRecord() {
	List r;
	Check(GpuWriteRanges::Record(r, kImage - 100, 50, kImage, kImage + kSize) && r.empty(),
	      "write before the image is ignored");
	Check(GpuWriteRanges::Record(r, kImage + kSize, 50, kImage, kImage + kSize) && r.empty(),
	      "write after the image is ignored");
	Check(GpuWriteRanges::Record(r, kImage - 16, 32, kImage, kImage + kSize) && r.size() == 1 &&
	          r[0].first == kImage && r[0].second == kImage + 16,
	      "write over the start is clipped");
	Check(GpuWriteRanges::Record(r, kImage + kSize - 8, 64, kImage, kImage + kSize) &&
	          r.size() == 2 && r[1].second == kImage + kSize,
	      "write over the end is clipped");
	Check(GpuWriteRanges::Record(r, kImage + 16, kSize - 24, kImage, kImage + kSize) &&
	          r.size() == 1 && r[0].first == kImage && r[0].second == kImage + kSize,
	      "touching writes merge into one range");
	Check(GpuWriteRanges::Record(r, UINT64_MAX - 4, 100, kImage, kImage + kSize) && r.size() == 1,
	      "wrapping write ignored");

	List many;
	bool ok = true;
	for (uint64_t i = 0; i < GpuWriteRanges::kMaxRanges; i++) {
		ok = GpuWriteRanges::Record(many, kImage + i * 0x20000, 0x10000, kImage, kImage + kSize) && ok;
	}
	Check(ok && many.size() == GpuWriteRanges::kMaxRanges, "kMaxRanges separate ranges fit");
	List merged = many;
	Check(!GpuWriteRanges::Record(many, kImage + GpuWriteRanges::kMaxRanges * 0x20000, 0x10000, kImage,
	                              kImage + kSize),
	      "one more separate range overflows");
	Check(GpuWriteRanges::Record(merged, kImage + 0x10000, 0x10000, kImage, kImage + kSize) &&
	          merged.size() == GpuWriteRanges::kMaxRanges - 1,
	      "a write joining two ranges reduces the count");
}

void TestComplement() {
	const List r {{kImage + 0x1000, kImage + 0x2000}, {kImage + 0x5000, kImage + 0x6000}};
	const auto c = GpuWriteRanges::Complement(r, kImage, kImage + 0x8000);
	Check(c.size() == 3 && c[0] == KeptRanges::Range {kImage, kImage + 0x1000} &&
	          c[1] == KeptRanges::Range {kImage + 0x2000, kImage + 0x5000} &&
	          c[2] == KeptRanges::Range {kImage + 0x6000, kImage + 0x8000},
	      "complement of two inner ranges");
	const auto all = GpuWriteRanges::Complement({}, kImage, kImage + 0x8000);
	Check(all.size() == 1 && all[0].first == kImage && all[0].second == kImage + 0x8000,
	      "complement of nothing is everything");
	const auto none = GpuWriteRanges::Complement({{kImage - 5, kImage + 0x9000}}, kImage, kImage + 0x8000);
	Check(none.empty(), "complement of a covering range is empty");
	const auto edge = GpuWriteRanges::Complement({{kImage, kImage + 0x1000}, {kImage + 0x7000, kImage + 0x8000}},
	                                             kImage, kImage + 0x8000);
	Check(edge.size() == 1 && edge[0].first == kImage + 0x1000 && edge[0].second == kImage + 0x7000,
	      "complement with ranges at both edges");

	// Random: complement and ranges partition the image exactly.
	std::mt19937_64 rng(3);
	bool            exact = true;
	for (int round = 0; round < 500; round++) {
		List ranges;
		for (int i = 0; i < 10; i++) {
			const auto a = kImage + rng() % 0x10000;
			(void)GpuWriteRanges::Record(ranges, a, 1 + rng() % 0x800, kImage, kImage + 0x10000);
		}
		const auto comp  = GpuWriteRanges::Complement(ranges, kImage, kImage + 0x10000);
		uint64_t   total = KeptRanges::Bytes(ranges) + KeptRanges::Bytes(comp);
		for (const auto& [b, e]: comp) {
			exact = exact && !KeptRanges::Intersects(ranges, b, e);
		}
		exact = exact && total == 0x10000;
	}
	Check(exact, "ranges and complement partition the image");
}

void TestRowRuns() {
	// 1920x1080 RGBA16F render target: 128x64-texel 64 KiB blocks, 15 per row, 17 rows.
	constexpr uint64_t row_bytes = 15ull * 65536ull;
	constexpr uint32_t rows      = 17;
	const uint64_t     base      = kImage;
	// One 64 KiB block in row 5 (Astro's typical compute write).
	auto runs = GpuWriteRanges::DirtyRowRuns({{base + 5 * row_bytes + 3 * 65536, base + 5 * row_bytes + 4 * 65536}},
	                                         base, row_bytes, rows);
	Check(runs.size() == 1 && runs[0].first == 5 && runs[0].second == 6, "one block -> one block row");
	// A write across a row boundary covers both rows.
	runs = GpuWriteRanges::DirtyRowRuns({{base + 2 * row_bytes - 16, base + 2 * row_bytes + 16}}, base,
	                                    row_bytes, rows);
	Check(runs.size() == 1 && runs[0].first == 1 && runs[0].second == 3, "boundary write -> two rows");
	// Separate writes in adjacent rows merge, distant ones do not.
	runs = GpuWriteRanges::DirtyRowRuns({{base + 3 * row_bytes, base + 3 * row_bytes + 1},
	                                     {base + 4 * row_bytes + 7, base + 4 * row_bytes + 9},
	                                     {base + 10 * row_bytes, base + 10 * row_bytes + 4}},
	                                    base, row_bytes, rows);
	Check(runs.size() == 2 && runs[0] == std::pair<uint32_t, uint32_t> {3, 5} &&
	          runs[1] == std::pair<uint32_t, uint32_t> {10, 11},
	      "adjacent rows merge, distant runs stay apart");
	// Writes outside the rows (padding after the last row, other subresources) select nothing.
	runs = GpuWriteRanges::DirtyRowRuns({{base - 100, base}, {base + rows * row_bytes, base + rows * row_bytes + 64}},
	                                    base, row_bytes, rows);
	Check(runs.empty(), "writes outside the rows select nothing");
	// A whole-surface write selects every row.
	runs = GpuWriteRanges::DirtyRowRuns({{base, base + rows * row_bytes}}, base, row_bytes, rows);
	Check(runs.size() == 1 && runs[0].first == 0 && runs[0].second == rows, "whole write -> all rows");

	// Random: every written byte inside the rows lies in a selected run; every run has a write.
	std::mt19937_64 rng(11);
	bool            covered = true;
	bool            needed  = true;
	for (int round = 0; round < 300; round++) {
		List ranges;
		for (int i = 0; i < 6; i++) {
			const auto a = base + rng() % (rows * row_bytes + row_bytes);
			(void)GpuWriteRanges::Record(ranges, a, 1 + rng() % (2 * row_bytes), base,
			                             base + rows * row_bytes + row_bytes);
		}
		const auto r = GpuWriteRanges::DirtyRowRuns(ranges, base, row_bytes, rows);
		List       selected;
		for (const auto& [r0, r1]: r) {
			(void)KeptRanges::Add(selected, base + r0 * row_bytes, base + r1 * row_bytes);
			bool any = false;
			for (uint32_t row = r0; row < r1; row++) {
				any = KeptRanges::Intersects(ranges, base + row * row_bytes, base + (row + 1) * row_bytes);
				needed = needed && any;
			}
		}
		for (const auto& [b, e]: ranges) {
			const auto cb = std::max(b, base);
			const auto ce = std::min(e, base + rows * row_bytes);
			if (cb < ce) {
				const auto unselected = GpuWriteRanges::Complement(selected, cb, ce);
				covered               = covered && unselected.empty();
			}
		}
	}
	Check(covered, "every written byte of the rows is in a selected run");
	Check(needed, "every selected row holds a written byte");
}

void TestDirtyLayers() {
	// Beyond Good & Evil's case: whole layers of a 16-layer array written through buffers.
	constexpr uint64_t slice = kSize / 16;
	List               r;
	(void)GpuWriteRanges::Record(r, kImage + 3 * slice, slice, kImage, kImage + kSize);
	(void)GpuWriteRanges::Record(r, kImage + 9 * slice, slice, kImage, kImage + kSize);
	const auto layers = KeptRanges::DirtyLayers(r, kImage, slice, 16);
	Check(layers.size() == 2 && layers[0] == 3 && layers[1] == 9, "two whole layers -> two layers");
	List partial;
	(void)GpuWriteRanges::Record(partial, kImage + 4 * slice - 1, 2, kImage, kImage + kSize);
	const auto two = KeptRanges::DirtyLayers(partial, kImage, slice, 16);
	Check(two.size() == 2 && two[0] == 3 && two[1] == 4, "a write across a layer boundary -> both");
}

} // namespace

int main() {
	TestRecord();
	TestComplement();
	TestRowRuns();
	TestDirtyLayers();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("gpu_write_ranges_tests: all passed\n");
	return 0;
}
