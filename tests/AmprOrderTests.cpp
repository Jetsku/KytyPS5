// AMM submission order (libs/amprOrder.h): an unmap leaves out the parts a later submission mapped.
#include "libs/amprOrder.h"

#include <cstdio>
#include <cstdlib>

namespace {
namespace Order = Libs::LibAmpr::AmmOrder;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "AmprOrderTests: %s\n", message);
		std::abort();
	}
}

void Reset() {
	std::scoped_lock lock(Order::g_mutex);
	Order::g_ranges.clear();
}

// An unmap from an older submission keeps what a later one mapped in its range.
void TestOlderUnmapKeepsNewerMap() {
	Reset();
	const auto older = Order::NextSeq();
	const auto newer = Order::NextSeq();
	Order::Mapped(0x10000, 0x8000, older);
	Order::Mapped(0x12000, 0x2000, newer); // remapped inside by a later submission
	uint64_t   kept   = 0;
	const auto pieces = Order::UnmapPieces(0x10000, 0x8000, older, &kept);
	Check(kept == 0x2000, "the newer map's bytes are kept");
	Check(pieces.size() == 2 && pieces[0].first == 0x10000 && pieces[0].second == 0x2000 &&
	          pieces[1].first == 0x14000 && pieces[1].second == 0x4000,
	      "the unmap removes the older parts around it");
	// The kept part still belongs to the newer submission: an unmap from it removes it.
	const auto again = Order::UnmapPieces(0x12000, 0x2000, newer, &kept);
	Check(kept == 0 && again.size() == 1 && again[0].first == 0x12000 && again[0].second == 0x2000,
	      "the newer submission's own unmap removes it");
}

// An unmap from the newest submission (or of untracked memory) removes all of its range.
void TestNewestUnmapRemovesAll() {
	Reset();
	const auto first = Order::NextSeq();
	Order::Mapped(0x20000, 0x4000, first);
	const auto later  = Order::NextSeq();
	uint64_t   kept   = 0;
	const auto pieces = Order::UnmapPieces(0x1f000, 0x6000, later, &kept);
	Check(kept == 0 && pieces.size() == 1 && pieces[0].first == 0x1f000 && pieces[0].second == 0x6000,
	      "a later unmap removes everything, untracked bytes included");
	const auto none = Order::UnmapPieces(0x20000, 0x4000, first, &kept);
	Check(kept == 0 && none.size() == 1, "nothing stays tracked after the unmap");
}

// A remap by a later submission replaces the older record (splitting it).
void TestRemapSplits() {
	Reset();
	const auto a = Order::NextSeq();
	Order::Mapped(0x30000, 0x3000, a);
	const auto b = Order::NextSeq();
	Order::Mapped(0x31000, 0x1000, b);
	uint64_t   kept   = 0;
	const auto pieces = Order::UnmapPieces(0x30000, 0x3000, a, &kept);
	Check(kept == 0x1000 && pieces.size() == 2, "the middle page mapped later is kept");
}

} // namespace

int main() {
	TestOlderUnmapKeepsNewerMap();
	TestNewestUnmapRemovesAll();
	TestRemapSplits();
	std::printf("AmprOrderTests: all passed\n");
	return 0;
}
