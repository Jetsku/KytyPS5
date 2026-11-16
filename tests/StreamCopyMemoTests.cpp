// KYTY_STREAM_COPY_REUSE (graphics/host_gpu/renderer/cache/streamCopyMemo.h): a small stream copy is
// reused for the same range only within the tick and epoch it was made in.

#include "graphics/host_gpu/renderer/cache/streamCopyMemo.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

using Libs::Graphics::StreamCopyMemo;

namespace {

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("StreamCopyMemoTests: FAILED: %s\n", what);
		g_failures++;
	}
}

void TestReuseWithinTickAndEpoch() {
	auto     memo   = std::make_unique<StreamCopyMemo>();
	uint64_t offset = 0;
	Check(!memo->Find(0x10000, 256, 5, offset), "an empty memo finds nothing");
	memo->Note(0x10000, 256, 5, 0x400);
	Check(memo->Find(0x10000, 256, 5, offset) && offset == 0x400, "the same range in the same tick and epoch");
	Check(!memo->Find(0x10000, 128, 5, offset), "another size");
	Check(!memo->Find(0x10010, 256, 5, offset), "another address");
	Check(!memo->Find(0x10000, 256, 6, offset), "a later tick (the copy's command buffer was submitted)");
	memo->NewEpoch();
	Check(!memo->Find(0x10000, 256, 5, offset), "a packet that may write memory ended the reuse");
	memo->Note(0x10000, 256, 5, 0x800);
	Check(memo->Find(0x10000, 256, 5, offset) && offset == 0x800, "a new copy in the new epoch");
}

void TestUnusedEntriesNeverMatch() {
	auto     memo   = std::make_unique<StreamCopyMemo>();
	uint64_t offset = 0;
	// An entry never written holds zeros: address 0, size 0, tick 0 must not match it.
	Check(!memo->Find(0, 0, 0, offset), "an unused entry");
}

void TestCollisionReplaces() {
	auto memo = std::make_unique<StreamCopyMemo>();
	// Two ranges in one slot: the later copy replaces the earlier; the earlier is copied again.
	const uint64_t a = 0x20000;
	uint64_t       b = 0;
	for (uint64_t candidate = a + 0x10; candidate < a + 0x1000000; candidate += 0x10) {
		if (StreamCopyMemo::SlotOf(candidate, 64) == StreamCopyMemo::SlotOf(a, 64)) {
			b = candidate;
			break;
		}
	}
	Check(b != 0, "a colliding address exists");
	memo->Note(a, 64, 1, 0x100);
	memo->Note(b, 64, 1, 0x200);
	uint64_t offset = 0;
	Check(!memo->Find(a, 64, 1, offset), "the replaced range is not found");
	Check(memo->Find(b, 64, 1, offset) && offset == 0x200, "the replacing range is found");
}

void TestSlotsInRange() {
	for (uint64_t address = 0; address < (1ull << 24u); address += 0x1230) {
		for (uint64_t size: {4ull, 16ull, 256ull, 1023ull, 4096ull, 16384ull}) {
			if (StreamCopyMemo::SlotOf(address, size) >= StreamCopyMemo::Slots) {
				Check(false, "a slot index past the table");
				return;
			}
		}
	}
}

} // namespace

int main() {
	TestReuseWithinTickAndEpoch();
	TestUnusedEntriesNeverMatch();
	TestCollisionReplaces();
	TestSlotsInRange();
	if (g_failures != 0) {
		return EXIT_FAILURE;
	}
	std::printf("StreamCopyMemoTests: ok\n");
	return EXIT_SUCCESS;
}
