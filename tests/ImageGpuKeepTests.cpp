#include "graphics/host_gpu/renderer/image/keptRanges.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool condition, const char* description) {
	if (!condition) {
		std::fprintf(stderr, "ImageGpuKeepTests: failed: %s\n", description);
		std::abort();
	}
}

constexpr uint64_t Page    = 0x4000;
constexpr uint64_t Granule = 0x10000;

void TestAdd() {
	KeptRanges::List ranges;
	Check(KeptRanges::Add(ranges, 0x1000, 0x2000) == 0x1000, "first range is new");
	Check(KeptRanges::Add(ranges, 0x1000, 0x2000) == 0, "same range adds nothing");
	Check(KeptRanges::Add(ranges, 0x4000, 0x5000) == 0x1000, "disjoint range");
	Check(ranges.size() == 2, "two disjoint ranges stay apart");
	Check(KeptRanges::Add(ranges, 0x2000, 0x4000) == 0x2000, "gap filled");
	Check(ranges.size() == 1 && ranges[0] == KeptRanges::Range(0x1000, 0x5000), "touching ranges merge");
	Check(KeptRanges::Add(ranges, 0x0, 0x8000) == 0x8000 - 0x4000, "superset counts only new bytes");
	Check(ranges.size() == 1 && ranges[0] == KeptRanges::Range(0x0, 0x8000), "superset replaces");
	Check(KeptRanges::Add(ranges, 0x9000, 0x9000) == 0 && ranges.size() == 1, "empty range ignored");
	Check(KeptRanges::Bytes(ranges) == 0x8000, "byte count");
	KeptRanges::List unordered;
	(void)KeptRanges::Add(unordered, 0x8000, 0x9000);
	(void)KeptRanges::Add(unordered, 0x1000, 0x2000);
	(void)KeptRanges::Add(unordered, 0x4000, 0x5000);
	Check(unordered.size() == 3 && unordered[0].first == 0x1000 && unordered[1].first == 0x4000 &&
	          unordered[2].first == 0x8000,
	      "ranges stay sorted");
	(void)KeptRanges::Add(unordered, 0x1800, 0x8800);
	Check(unordered.size() == 1 && unordered[0] == KeptRanges::Range(0x1000, 0x9000),
	      "one range spanning several merges them all");
}

void TestSubtract() {
	KeptRanges::List ranges;
	(void)KeptRanges::Add(ranges, 0x0, 0x10000);
	KeptRanges::Subtract(ranges, 0x4000, 0x8000);
	Check(ranges.size() == 2 && ranges[0] == KeptRanges::Range(0x0, 0x4000) &&
	          ranges[1] == KeptRanges::Range(0x8000, 0x10000),
	      "a hole splits a range");
	KeptRanges::Subtract(ranges, 0x0, 0x4000);
	Check(ranges.size() == 1 && ranges[0].first == 0x8000, "a whole range goes");
	KeptRanges::Subtract(ranges, 0xc000, 0x20000);
	Check(ranges.size() == 1 && ranges[0] == KeptRanges::Range(0x8000, 0xc000), "a tail goes");
	KeptRanges::Subtract(ranges, 0x0, 0x1000);
	Check(ranges.size() == 1 && KeptRanges::Bytes(ranges) == 0x4000, "a miss changes nothing");
}

void TestIntersects() {
	KeptRanges::List ranges;
	(void)KeptRanges::Add(ranges, 0x1000, 0x2000);
	(void)KeptRanges::Add(ranges, 0x8000, 0x9000);
	Check(!KeptRanges::Intersects(ranges, 0x0, 0x1000), "range ending at a start misses");
	Check(KeptRanges::Intersects(ranges, 0x0, 0x1001), "one byte overlap");
	Check(!KeptRanges::Intersects(ranges, 0x2000, 0x8000), "the gap between misses");
	Check(KeptRanges::Intersects(ranges, 0x7fff, 0x8001), "second range");
	Check(KeptRanges::Intersects(ranges, 0x0, 0x10000), "covering range");
	Check(!KeptRanges::Intersects(ranges, 0x9000, 0xa000), "after the last");
	Check(!KeptRanges::Intersects(ranges, 0x1800, 0x1800), "empty query");
	Check(!KeptRanges::Intersects({}, 0x0, 0x10000), "no ranges");
}

void TestWriteSpan() {
	// A 4-byte write in the middle of a page-aligned 1 MiB image: the 64 KiB granule around it.
	const uint64_t image = 0x10000000;
	auto           span  = KeptRanges::WriteSpan(image + 0x23456, 4, image, image + 0x100000, Granule, Page);
	Check(span == KeptRanges::Range(image + 0x20000, image + 0x30000), "granule around a small write");
	// A write across a granule boundary takes both granules.
	span = KeptRanges::WriteSpan(image + 0x2fffe, 4, image, image + 0x100000, Granule, Page);
	Check(span == KeptRanges::Range(image + 0x20000, image + 0x40000), "write across granules");
	// An image that is not page aligned: the span stops at the pages it lies on.
	const uint64_t odd = 0x20001100;
	span = KeptRanges::WriteSpan(odd + 0x10, 4, odd, odd + 0x6000, Granule, Page);
	Check(span == KeptRanges::Range(0x20000000, 0x20008000), "clipped to the image's pages");
	// A neighbour's write on a page the image shares.
	span = KeptRanges::WriteSpan(0x20000080, 4, odd, odd + 0x6000, Granule, Page);
	Check(span == KeptRanges::Range(0x20000000, 0x20008000), "a shared page belongs to the span");
	// A write past the image's pages.
	span = KeptRanges::WriteSpan(0x30000000, 4, odd, odd + 0x6000, Granule, Page);
	Check(span.first == 0 && span.second == 0, "a write elsewhere");
	Check(KeptRanges::WriteSpan(image, 0, image, image + 0x1000, Granule, Page).second == 0, "an empty write");
}

void TestDirtyLayers() {
	// An 8-layer array of 4 MiB layers (the shadow-pool shape, smaller).
	const uint64_t   base  = 0x40000000;
	const uint64_t   slice = 0x400000;
	KeptRanges::List dirty;
	Check(KeptRanges::DirtyLayers(dirty, base, slice, 8).empty(), "no dirty layer");
	(void)KeptRanges::Add(dirty, base + slice * 3 + 0x1000, base + slice * 3 + 0x2000);
	auto layers = KeptRanges::DirtyLayers(dirty, base, slice, 8);
	Check(layers.size() == 1 && layers[0] == 3, "one written layer");
	(void)KeptRanges::Add(dirty, base + slice * 5 - 0x10, base + slice * 5 + 0x10);
	layers = KeptRanges::DirtyLayers(dirty, base, slice, 8);
	Check(layers.size() == 3 && layers[0] == 3 && layers[1] == 4 && layers[2] == 5,
	      "a write across a layer boundary dirties both layers");
	// An unmap of layers 6 and 7 (exact bytes).
	(void)KeptRanges::Add(dirty, base + slice * 6, base + slice * 8);
	layers = KeptRanges::DirtyLayers(dirty, base, slice, 8);
	Check(layers.size() == 5 && layers.back() == 7, "unmapped layers");
	Check(layers.front() == 3, "layers 0-2 stay the GPU's");
}

} // namespace

int main() {
	TestAdd();
	TestSubtract();
	TestIntersects();
	TestWriteSpan();
	TestDirtyLayers();
	std::printf("ImageGpuKeepTests: all tests passed\n");
	return 0;
}
