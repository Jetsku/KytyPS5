// Tests for kernel/pointerScan.h (post-mortem search for a faulting GPU address in memory).
#include "kernel/pointerScan.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Libs::LibKernel::Memory::PointerScan;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "PointerScanTests: FAILED: %s\n", message);
		std::exit(1);
	}
}

// The fault the GPU reported in a Crash Bandicoot 4 run: page 0xbeefdeadb000, precision 4096.
constexpr uint64_t kLow    = 0x0000beefdeadb000ull;
constexpr uint64_t kHigh   = 0x0000beefdeadbfffull;
constexpr uint64_t kPoison = 0xdeadbeefdeadbeefull;

void TestMatchSemantics() {
	const PointerScan scan(kLow, kHigh);
	Check(scan.Matches(kPoison), "0xDEADBEEFDEADBEEF has the low 48 bits 0xBEEFDEADBEEF: inside the page");
	Check(scan.Matches(0x0000beefdeadbeefull), "the same pointer with zero top bits");
	Check(scan.Matches(kLow), "the first byte of the range");
	Check(scan.Matches(kHigh), "the last byte of the range");
	Check(!scan.Matches(kLow - 1), "one below the range (the subtraction must not wrap into it)");
	Check(!scan.Matches(kHigh + 1), "one above the range");
	Check(!scan.Matches(0), "zero is not an address in the range");
	Check(!scan.Matches(UINT64_MAX), "all ones: low 48 bits 0xFFFFFFFFFFFF");
	Check(!scan.Matches(0x0000beefdeadc000ull), "the next page");
}

void TestRunsAndOffsets() {
	PointerScan scan(kLow, kHigh);
	std::vector<uint64_t> block(512, 0x1111111111111111ull);
	for (size_t i = 10; i < 110; i++) {
		block[i] = kPoison;
	}
	block[300] = kPoison;
	scan.AddBlock(block.data(), block.size(), 0x10000);
	scan.NoteScanned(block.size() * 8);

	Check(scan.Hits() == 101, "100 filled values plus one isolated");
	Check(scan.Runs().size() == 2, "the fill is one run, the isolated value another");
	Check(scan.Runs()[0].offset == 0x10000 + 10 * 8, "the run starts at its byte offset");
	Check(scan.Runs()[0].qwords == 100, "the fill is 100 values long");
	Check(scan.Runs()[0].value == kPoison, "the run remembers the value");
	Check(scan.Runs()[1].offset == 0x10000 + 300 * 8, "the isolated value's offset");
	Check(scan.ScannedBytes() == 4096, "scanned bytes are what the caller noted");
}

void TestRunContinuesAcrossBlocks() {
	PointerScan           scan(kLow, kHigh);
	std::vector<uint64_t> first(512, 0);
	std::vector<uint64_t> second(512, 0);
	for (size_t i = 500; i < 512; i++) {
		first[i] = kPoison;
	}
	for (size_t i = 0; i < 20; i++) {
		second[i] = kPoison;
	}
	scan.AddBlock(first.data(), first.size(), 0);
	scan.AddBlock(second.data(), second.size(), 4096);
	Check(scan.Runs().size() == 1, "contiguous blocks with the same value are one run");
	Check(scan.Runs()[0].qwords == 32, "12 at the end of the first block plus 20 at the start of the second");
}

void TestDifferentValuesAreSeparateRuns() {
	PointerScan           scan(kLow, kHigh);
	std::vector<uint64_t> block(8, 0);
	block[0] = kLow + 8;
	block[1] = kLow + 8;
	block[2] = kLow + 16; // adjacent but another value
	block[3] = 0;         // a gap
	block[4] = kLow + 16;
	scan.AddBlock(block.data(), block.size(), 0);
	Check(scan.Runs().size() == 3, "equal neighbours merge; a different value or a gap starts a new run");
	Check(scan.Runs()[0].qwords == 2 && scan.Runs()[1].qwords == 1 && scan.Runs()[2].qwords == 1,
	      "run lengths");
}

void TestNoMatches() {
	PointerScan           scan(kLow, kHigh);
	std::vector<uint64_t> block(512, 0x0000700000000000ull);
	scan.AddBlock(block.data(), block.size(), 0);
	Check(scan.Hits() == 0 && scan.Runs().empty(), "nothing matches, nothing is recorded");
}

void TestRunCap() {
	PointerScan           scan(kLow, kHigh);
	std::vector<uint64_t> block(2 * PointerScan::kMaxRuns + 8);
	for (size_t i = 0; i < block.size(); i++) {
		block[i] = kLow + (i % 2 == 0 ? 8 : 16); // alternating values: every value is its own run
	}
	scan.AddBlock(block.data(), block.size(), 0);
	Check(scan.Hits() == block.size(), "every match is counted");
	Check(scan.Runs().size() == PointerScan::kMaxRuns, "recorded runs are capped");
	Check(scan.DroppedRuns() == block.size() - PointerScan::kMaxRuns, "the rest are counted as dropped");
}

void TestTopRunsAndFormat() {
	PointerScan           scan(kLow, kHigh);
	std::vector<uint64_t> block(64, 0);
	block[1]  = kLow + 1;                                    // a run of 1
	for (size_t i = 10; i < 30; i++) block[i] = kPoison;     // a run of 20
	for (size_t i = 40; i < 45; i++) block[i] = kLow + 0x10; // a run of 5
	scan.AddBlock(block.data(), block.size(), 0x1000);

	const auto top = scan.TopRuns(2);
	Check(top.size() == 2, "limit applies");
	Check(top[0].qwords == 20 && top[1].qwords == 5, "the longest two, in offset order");

	const std::string text = scan.Format([](uint64_t offset) { return offset == 0x1000 + 80 ? std::string(" guest 0xabc") : std::string(); });
	Check(text.find("hits: 26 values in 3 runs") != std::string::npos, "the summary line");
	Check(text.find("guest 0xabc") != std::string::npos, "the describe callback is used");
	Check(text.find("20 x 0xdeadbeefdeadbeef") != std::string::npos, "the run line shows count and value");
}

} // namespace

int main() {
	TestMatchSemantics();
	TestRunsAndOffsets();
	TestRunContinuesAcrossBlocks();
	TestDifferentValuesAreSeparateRuns();
	TestNoMatches();
	TestRunCap();
	TestTopRunsAndFormat();
	std::puts("PointerScanTests: all passed");
	return 0;
}
