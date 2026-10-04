#include "common/slotVector.h"
#include "graphics/host_gpu/renderer/cache/bdaHotRanges.h"

#include <array>
#include <cstdio>
#include <limits>
#include <random>

namespace {
struct Range {
	Common::SlotId id;
	uint64_t address;
	uint64_t size;
	bool operator==(const Range&) const = default;
};
int failures = 0;
void Expect(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "BdaHotRanges: %s\n", message);
		++failures;
	}
}

void TestOverlapAndHoles() {
	const Common::SlotId id {4, 7};
	std::vector<Range> ranges {{id, 100, 100}, {id, 110, 10}, {id, 180, 40},
	                           {id, 220, 10}, {id, 250, 5}, {id, 100, 100}};
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges), "valid ranges accepted");
	Expect(ranges == std::vector<Range>{{id, 100, 130}, {id, 250, 5}},
	       "containment, overlap and adjacency merge without bridging a hole");
	const auto once = ranges;
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges) && ranges == once, "merge is idempotent");
}

void TestIdentityAndGeneration() {
	std::vector<Range> ranges {{{9, 2}, 100, 30}, {{9, 1}, 100, 30},
	                           {{3, 1}, 100, 30}, {{9, 1}, 115, 30}};
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges), "generation ranges accepted");
	Expect(ranges == std::vector<Range>{{{3, 1}, 100, 30}, {{9, 1}, 100, 45}, {{9, 2}, 100, 30}},
	       "buffer identities and reused slot generations stay separate");
}

void TestBounds() {
	constexpr auto maximum = std::numeric_limits<uint64_t>::max();
	std::vector<Range> ranges {{{1, 1}, maximum - 20, 10}, {{1, 1}, maximum - 10, 10}};
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges) && ranges.size() == 1 && ranges[0].size == 20,
	       "adjacency near uint64 limit does not wrap");
	for (const auto invalid: {Range {{1, 1}, maximum - 3, 4}, Range {{1, 1}, 4, 0}}) {
		ranges = {{{2, 1}, 200, 20}, invalid, {{1, 1}, 100, 10}};
		const auto original = ranges;
		Expect(!Libs::Graphics::MergeBdaHotRanges(ranges) && ranges == original,
		       "invalid input rejected before mutation");
	}
	ranges.clear();
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges) && ranges.empty(), "empty input accepted");
}

// Compare byte coverage using an independent bitmap oracle, across unsorted random overlaps.
void TestCoverage() {
	std::mt19937 random(0xBDA7);
	for (unsigned run = 0; run < 200; ++run) {
		std::vector<Range> ranges;
		std::array<std::array<bool, 256>, 4> before {}, after {};
		for (unsigned i = 0; i < 64; ++i) {
			const auto index = static_cast<uint32_t>(random() % 4);
			const auto begin = static_cast<uint32_t>(random() % 240);
			const auto size = static_cast<uint32_t>(random() % 16 + 1);
			ranges.push_back({Common::SlotId {index, 1}, begin, size});
			for (auto p = begin; p < begin + size; ++p) before[index][p] = true;
		}
		Expect(Libs::Graphics::MergeBdaHotRanges(ranges), "random ranges accepted");
		bool canonical = true;
		for (size_t i = 0; i < ranges.size(); ++i) {
			const auto& range = ranges[i];
			for (auto p = range.address; p < range.address + range.size; ++p) after[range.id.index][p] = true;
			if (i != 0 && ranges[i - 1].id == range.id) {
				canonical &= ranges[i - 1].address + ranges[i - 1].size < range.address;
			}
		}
		Expect(before == after, "random union preserves every byte and every buffer");
		if (!canonical) {
			Expect(false, "random output has no overlap, duplicate or adjacent runs per buffer");
			break;
		}
	}
}

void TestRepeatedPartialRanges() {
	std::vector<Range> ranges {{{8, 3}, 0x1000, 0x1000}};
	for (unsigned i = 0; i < 5000; ++i) ranges.push_back({{8, 3}, 0x1000 + (i % 100) * 4, 4});
	Expect(Libs::Graphics::MergeBdaHotRanges(ranges) && ranges.size() == 1,
	       "partial reports do not grow a single hot run past the full-scan limit");
}
} // namespace

int main() {
	TestOverlapAndHoles();
	TestIdentityAndGeneration();
	TestBounds();
	TestCoverage();
	TestRepeatedPartialRanges();
	std::printf("BdaHotRanges: %d failures\n", failures);
	return failures == 0 ? 0 : 1;
}
