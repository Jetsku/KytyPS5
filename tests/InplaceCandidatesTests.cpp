// KYTY_INPLACE candidate pages (graphics/host_gpu/renderer/cache/inplaceCandidates.h).
#include "graphics/host_gpu/renderer/cache/inplaceCandidates.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using Libs::Graphics::InplaceCandidates;

namespace {

int g_failures = 0;

void Check(bool condition, const char* what, int line) {
	if (!condition) {
		std::printf("FAILED line %d: %s\n", line, what);
		g_failures++;
	}
}

#define CHECK(x) Check((x), #x, __LINE__)

constexpr uint64_t Page = 4096;

void TestPromotionNeedsConsecutiveFrames() {
	InplaceCandidates c(3);
	const uint64_t    a = 0x2010000000ull;
	CHECK(!c.NoteFault(a));
	c.AdvanceFrame();
	CHECK(!c.NoteFault(a + 100)); // same page, second frame
	CHECK(!c.All(a, 16));
	c.AdvanceFrame();
	CHECK(c.NoteFault(a + 8)); // third consecutive frame: promoted
	CHECK(c.All(a, Page));
	CHECK(c.Any(a, 1));
	CHECK(c.Count() == 1);
	// Faulting again in the same frame does not promote twice.
	CHECK(!c.NoteFault(a));
	CHECK(c.Count() == 1);
}

void TestRingPeriodPromotes() {
	// A page of a ring the game rewrites every third frame: three faults within 8 frames.
	InplaceCandidates c(3);
	const uint64_t    a = 0x123456000ull;
	CHECK(!c.NoteFault(a)); // frame 1
	c.AdvanceFrame();
	c.AdvanceFrame();
	c.AdvanceFrame();
	CHECK(!c.NoteFault(a)); // frame 4
	c.AdvanceFrame();
	c.AdvanceFrame();
	c.AdvanceFrame();
	CHECK(c.NoteFault(a)); // frame 7: 3 faulting frames of the last 8
}

void TestSparseFaultsDoNotPromote() {
	InplaceCandidates c(2);
	const uint64_t    a = 0x223456000ull;
	(void)c.NoteFault(a);
	for (int i = 0; i < 8; i++) {
		c.AdvanceFrame();
	}
	CHECK(!c.NoteFault(a)); // the earlier fault is 8 frames old: out of the window
	c.AdvanceFrame();
	CHECK(c.NoteFault(a));
}
void Promote(InplaceCandidates& c, uint64_t vaddr, int frames) {
	for (int f = 0; f < frames; f++) {
		(void)c.NoteFault(vaddr);
		c.AdvanceFrame();
	}
}

void TestRangesAndClear() {
	InplaceCandidates c(1);
	const uint64_t    a = 0x7000000000ull;
	Promote(c, a, 1);
	Promote(c, a + Page, 1);
	CHECK(c.All(a, 2 * Page));
	CHECK(!c.All(a, 2 * Page + 1)); // third page not a candidate
	CHECK(c.Any(a + 2 * Page - 1, 2));
	CHECK(!c.Any(a + 2 * Page, Page));
	CHECK(!c.All(a, 0));
	CHECK(c.Clear(a + Page, 1) == 1);
	CHECK(!c.All(a, 2 * Page));
	CHECK(c.All(a, Page));
	CHECK(c.Count() == 1);
	// A cleared page needs a new streak (promote_frames = 1: one frame).
	Promote(c, a + Page, 1);
	CHECK(c.All(a, 2 * Page));
	// Ranges past MaxPages are refused by All() even when every page is a candidate.
	CHECK(!c.All(a, (InplaceCandidates::MaxPages + 1) * Page));
}

void TestChunkBoundaryAndLimits() {
	InplaceCandidates c(1);
	const uint64_t    chunk = uint64_t {1} << InplaceCandidates::ChunkBits;
	const uint64_t    a     = 5 * chunk - Page; // last page of one chunk, first of the next
	Promote(c, a, 1);
	Promote(c, a + Page, 1);
	CHECK(c.All(a, 2 * Page));
	CHECK(c.Any(0, 6 * chunk)); // skips absent chunks
	CHECK(c.Clear(0, 6 * chunk) == 2);
	CHECK(c.Count() == 0);
	CHECK(!c.Any(a, 2 * Page));
	// Addresses beyond 40 bits are never candidates.
	CHECK(!c.NoteFault(uint64_t {1} << 41));
	CHECK(!c.Any(uint64_t {1} << 41, Page));
}

void TestConcurrentFaults() {
	InplaceCandidates        c(1);
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([&c, t] {
			for (uint64_t i = 0; i < 1000; i++) {
				(void)c.NoteFault(0x100000000ull + (static_cast<uint64_t>(t) * 1000 + i) * Page);
			}
		});
	}
	for (auto& th: threads) {
		th.join();
	}
	CHECK(c.Count() == 4000);
	CHECK(c.All(0x100000000ull, 1000 * Page));
}

} // namespace

int main() {
	TestPromotionNeedsConsecutiveFrames();
	TestRingPeriodPromotes();
	TestSparseFaultsDoNotPromote();
	TestRangesAndClear();
	TestChunkBoundaryAndLimits();
	TestConcurrentFaults();
	if (g_failures != 0) {
		std::printf("inplace_candidates_tests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("inplace_candidates_tests: all passed\n");
	return EXIT_SUCCESS;
}
