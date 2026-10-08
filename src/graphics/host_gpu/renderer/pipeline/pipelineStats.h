#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics {

// KYTY_PIPELINE_STATS=1 (diagnostic, default off): one console line every 10 s, "Pipelines 10s: ...",
// with the pipelines created in the interval by how they were made, the time draws and dispatches
// waited for them, the program permutations compiled, and the background pipeline work. Graphics:
// hit = taken from the driver cache by a probe (no compile), fast = built unoptimized first
// (KYTY_PIPELINE_FAST_FIRST), full = created synchronously as usual (a compile, or an unprobed driver
// cache hit), other = prefetched or library-linked. Compute: the same three, plus the software
// ray-tracing kernels published from background builds (rt) and compute pipelines a draw found
// prebuilt by the start-up prewarm (prewarmed). This header holds the Vulkan-free formatting.
struct PipelineStatsValues {
	uint64_t gfx_hit = 0, gfx_fast = 0, gfx_full = 0, gfx_other = 0, gfx_cp_ns = 0;
	uint64_t cs_hit = 0, cs_fast = 0, cs_full = 0, cs_cp_ns = 0, cs_prewarmed = 0, rt = 0;
	uint64_t known_misses = 0;              // probes of keys known to this PC that missed
	uint64_t programs = 0, translations = 0, replayed = 0;
	uint64_t stall_ns = 0;                  // every compile stall of a draw or dispatch (programs too)
	uint64_t optimized = 0, optimize_ns = 0; // fast-first background swaps
	uint64_t prewarm_built = 0, prewarm_ns = 0;
	uint64_t journal_built = 0, journal_skipped = 0, journal_ns = 0, journal_unpredicted = 0;
};

// Counters only grow; the line shows `now - before`. worst_ns is the interval's longest single stall.
[[nodiscard]] inline std::string FormatPipelineStats(const PipelineStatsValues& now,
                                                     const PipelineStatsValues& before,
                                                     uint64_t worst_ns, double seconds) {
	const auto d  = [](uint64_t a, uint64_t b) { return static_cast<unsigned long long>(a >= b ? a - b : 0); };
	const auto ms = [](uint64_t a, uint64_t b) { return static_cast<double>(a >= b ? a - b : 0) / 1e6; };
	const auto gfx = d(now.gfx_hit + now.gfx_fast + now.gfx_full + now.gfx_other,
	                   before.gfx_hit + before.gfx_fast + before.gfx_full + before.gfx_other);
	const auto cs  = d(now.cs_hit + now.cs_fast + now.cs_full, before.cs_hit + before.cs_fast + before.cs_full);
	char text[768];
	std::snprintf(
	    text, sizeof(text),
	    "Pipelines %.0fs: gfx %llu (hit %llu, fast %llu, full %llu, other %llu) cs %llu (hit %llu, fast "
	    "%llu, full %llu; prewarmed %llu) rt %llu; known-key misses %llu | CP wait %.1f ms (gfx %.1f, cs "
	    "%.1f; all compile stalls %.1f, worst %.1f) | shaders %llu (translated %llu, replayed %llu) | bg: "
	    "optimized %llu (%.1f ms), prewarm %llu (%.1f ms), journal %llu built %llu skipped (%.1f ms), "
	    "unpredicted %llu",
	    seconds, gfx, d(now.gfx_hit, before.gfx_hit), d(now.gfx_fast, before.gfx_fast),
	    d(now.gfx_full, before.gfx_full), d(now.gfx_other, before.gfx_other), cs,
	    d(now.cs_hit, before.cs_hit), d(now.cs_fast, before.cs_fast), d(now.cs_full, before.cs_full),
	    d(now.cs_prewarmed, before.cs_prewarmed), d(now.rt, before.rt),
	    d(now.known_misses, before.known_misses),
	    ms(now.gfx_cp_ns + now.cs_cp_ns, before.gfx_cp_ns + before.cs_cp_ns),
	    ms(now.gfx_cp_ns, before.gfx_cp_ns), ms(now.cs_cp_ns, before.cs_cp_ns),
	    ms(now.stall_ns, before.stall_ns), static_cast<double>(worst_ns) / 1e6,
	    d(now.programs, before.programs), d(now.translations, before.translations),
	    d(now.replayed, before.replayed), d(now.optimized, before.optimized),
	    ms(now.optimize_ns, before.optimize_ns), d(now.prewarm_built, before.prewarm_built),
	    ms(now.prewarm_ns, before.prewarm_ns), d(now.journal_built, before.journal_built),
	    d(now.journal_skipped, before.journal_skipped), ms(now.journal_ns, before.journal_ns),
	    d(now.journal_unpredicted, before.journal_unpredicted));
	return text;
}

// Whether an interval is worth a line: anything was created, compiled, built or waited for.
[[nodiscard]] inline bool PipelineStatsChanged(const PipelineStatsValues& now, const PipelineStatsValues& before) {
	return now.gfx_hit != before.gfx_hit || now.gfx_fast != before.gfx_fast || now.gfx_full != before.gfx_full ||
	       now.gfx_other != before.gfx_other || now.cs_hit != before.cs_hit || now.cs_fast != before.cs_fast ||
	       now.cs_full != before.cs_full || now.cs_prewarmed != before.cs_prewarmed || now.rt != before.rt ||
	       now.programs != before.programs || now.stall_ns != before.stall_ns ||
	       now.optimized != before.optimized || now.prewarm_built != before.prewarm_built ||
	       now.journal_built != before.journal_built || now.journal_skipped != before.journal_skipped;
}

} // namespace Libs::Graphics
