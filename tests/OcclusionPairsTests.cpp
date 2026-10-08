#include "graphics/host_gpu/renderer/occlusionPairs.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Libs::Graphics::OcclusionDumpPairs;
using Kind = OcclusionDumpPairs::Kind;

static void Require(bool value, const char* what) {
	if (!value) {
		std::fprintf(stderr, "occlusion pairs invariant failed: %s\n", what);
		std::abort();
	}
}

int main() {
	{
		// A pair at a 16-byte aligned address (the only layout recognised before 2026-10-06).
		OcclusionDumpPairs pairs;
		Require(pairs.Observe(0x4036786c0) == Kind::Begin, "aligned begin");
		Require(!pairs.Empty(), "aligned pair open");
		Require(pairs.Observe(0x4036786c8) == Kind::End, "aligned end");
		Require(pairs.Empty(), "aligned pair closed");
	}
	{
		// Astro Bot volcano_mountain, camera tilted down: depth-only proxy pairs at 8 mod 16. The end
		// dump (address % 16 == 0) must be recognised, or the visibility proxy's result is not
		// ordered before the label the game reads it after (the lava then disappears).
		OcclusionDumpPairs pairs;
		for (uint64_t frame = 0; frame < 4; ++frame) {
			Require(pairs.Observe(0x502df8e98) == Kind::Begin, "misaligned begin");
			Require(pairs.Observe(0x502df8ea0) == Kind::End, "misaligned end");
			Require(pairs.Observe(0x502df8f98) == Kind::Begin, "second misaligned begin");
			Require(pairs.Observe(0x502df8fa0) == Kind::End, "second misaligned end");
			Require(pairs.Empty(), "misaligned pairs closed");
		}
	}
	{
		// Nested pairs (an outer scope around per-object scopes), mixed alignments.
		OcclusionDumpPairs pairs;
		Require(pairs.Observe(0x1000) == Kind::Begin, "outer begin");
		Require(pairs.Observe(0x1108) == Kind::Begin, "inner begin");
		Require(pairs.Observe(0x1110) == Kind::End, "inner end");
		Require(pairs.Observe(0x1200) == Kind::Begin, "inner begin 2");
		Require(pairs.Observe(0x1208) == Kind::End, "inner end 2");
		Require(pairs.OpenCount() == 1, "outer still open");
		Require(pairs.Observe(0x1008) == Kind::End, "outer end");
		Require(pairs.Empty(), "nested closed");
	}
	{
		// A repeated begin keeps its pair open; the end then closes it once.
		OcclusionDumpPairs pairs;
		Require(pairs.Observe(0x2008) == Kind::Begin, "begin");
		Require(pairs.Observe(0x2008) == Kind::RepeatedBegin, "repeated begin");
		Require(pairs.OpenCount() == 1, "one open");
		Require(pairs.Observe(0x2010) == Kind::End, "end after repeat");
		Require(pairs.Observe(0x2010) == Kind::Begin, "a later dump at the end address begins");
	}
	{
		// A begin whose end never comes is dropped after MaxAgeDumps dumps, so a later pair begun 8
		// bytes after it is not mistaken for its end.
		OcclusionDumpPairs pairs;
		Require(pairs.Observe(0x3000) == Kind::Begin, "stale begin");
		for (uint64_t i = 0; i < OcclusionDumpPairs::MaxAgeDumps / 2; ++i) {
			Require(pairs.Observe(0x9000) == Kind::Begin, "filler begin");
			Require(pairs.Observe(0x9008) == Kind::End, "filler end");
		}
		Require(pairs.Dropped() == 1, "stale pair dropped");
		Require(pairs.Observe(0x3008) == Kind::Begin, "begin after the stale pair expired");
		Require(pairs.Observe(0x3010) == Kind::End, "its end");
		Require(pairs.Empty(), "all closed");
	}
	{
		// Too many open pairs: the oldest is dropped and counted.
		OcclusionDumpPairs pairs;
		for (uint64_t i = 0; i <= OcclusionDumpPairs::MaxOpen; ++i) {
			Require(pairs.Observe(0x10000 + i * 0x100) == Kind::Begin, "many begins");
		}
		Require(pairs.OpenCount() == OcclusionDumpPairs::MaxOpen, "bounded");
		Require(pairs.Dropped() == 1, "overflow counted");
		Require(pairs.Observe(0x10008) == Kind::Begin, "dropped pair's end is no end");
	}
	std::puts("occlusion pairs: ok");
	return 0;
}
