// KYTY_SEGMENT_HEAP: the process heap's kind as common/processHeap.h reads it. Built twice: plain
// (expects the NT heap) and with src/segmentHeap.manifest embedded (expects the Segment Heap), which
// also checks that the manifest reaches the executable through this toolchain.
#include "common/processHeap.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main() {
#if defined(KYTY_EXPECT_SEGMENT_HEAP)
	const auto expected = Common::ProcessHeapKind::Segment;
#else
	const auto expected = Common::ProcessHeapKind::Nt;
#endif
	const auto kind = Common::GetProcessHeapKind();
	std::printf("process heap: %s (expected %s)\n", Common::ProcessHeapKindName(kind),
	            Common::ProcessHeapKindName(expected));
	if (kind != expected) {
#if defined(_WIN32)
		const auto* words = static_cast<const unsigned*>(static_cast<const void*>(GetProcessHeap()));
		for (int i = 0; i < 64; i++) {
			if (words[i] == 0xEEFFEEFFu || words[i] == 0xFFEEFFEEu || words[i] == 0xDDEEDDEEu) {
				std::printf("signature 0x%08x at +0x%x\n", words[i], i * 4);
			}
		}
#endif
		std::printf("FAIL\n");
		return 1;
	}
	// The heap serves malloc/new from several threads either way.
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; t++) {
		threads.emplace_back([] {
			for (int i = 0; i < 100000; i++) {
				auto* block = new char[static_cast<size_t>(16 + (i % 512))];
				block[0]    = static_cast<char>(i);
				delete[] block;
			}
		});
	}
	for (auto& thread: threads) {
		thread.join();
	}
	std::printf("PASS\n");
	return 0;
}
