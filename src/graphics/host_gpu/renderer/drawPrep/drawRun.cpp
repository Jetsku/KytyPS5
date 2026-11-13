#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"

#include "common/assert.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <xxhash.h>

namespace Libs::Graphics::DrawRun {

namespace Detail {

// KYTY_DRAW_RUN: unset, empty, "0", "off": 0; "verify": 2; "exit": 3; anything else: 1.
static int64_t ParseMode(const char* value) {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0 ||
	    std::strcmp(value, "off") == 0) {
		return 0;
	}
	if (std::strcmp(value, "exit") == 0) {
		return 3;
	}
	return std::strcmp(value, "verify") == 0 ? 2 : 1;
}

// Activity is not maintained while off. Invalidate certificates across every mode switch so a
// record preceding an off period cannot certify operations performed during that period.
Live::Switch g_mode("KYTY_DRAW_RUN", ParseMode,
                    [](int64_t, int64_t) { NoteForeignActivity(); });
Live::Switch g_acquire("KYTY_DRAW_RUN_ACQUIRE", Live::ParseDefaultOff);
Live::Switch g_push("KYTY_DRAW_RUN_PUSH", Live::ParseDefaultOff);
// A record made under one value must not be continued under the other.
Live::Switch g_writes("KYTY_DRAW_RUN_WRITES", Live::ParseDefaultOff,
                      [](int64_t, int64_t) { NoteForeignActivity("KYTY_DRAW_RUN_WRITES change"); });

// A record made with recall on is only recalled while it stays on.
Live::Switch g_recall("KYTY_DRAW_RUN_RECALL", Live::ParseDefaultOff,
                      [](int64_t, int64_t) { NoteForeignActivity("KYTY_DRAW_RUN_RECALL change"); });

// KYTY_DRAW_RUN_IMAGE_EPOCH: unset, "0": 0; "verify": 2; "exit": 3; anything else: 1.
Live::Switch g_image_epoch("KYTY_DRAW_RUN_IMAGE_EPOCH", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
});

// KYTY_DRAW_RUN_CONT_ELIGIBLE: as KYTY_DRAW_RUN_IMAGE_EPOCH (0, 1, verify 2, exit 3).
Live::Switch g_cont_eligible("KYTY_DRAW_RUN_CONT_ELIGIBLE", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
});

// KYTY_DRAW_RUN_TARGETS: as KYTY_DRAW_RUN_IMAGE_EPOCH (0, 1, verify 2, exit 3).
Live::Switch g_targets("KYTY_DRAW_RUN_TARGETS", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::strcmp(value, "exit") == 0 ? 3 : 1;
});

// A record made with the swap on is only continued by a swap while it stays on.
Live::Switch g_pipeswap("KYTY_DRAW_RUN_PIPESWAP", Live::ParseDefaultOff,
                        [](int64_t, int64_t) { NoteForeignActivity("KYTY_DRAW_RUN_PIPESWAP change"); });
Live::Switch g_pipecont("KYTY_PIPE_CONT", Live::ParseDefaultOff,
                        [](int64_t, int64_t) { NoteForeignActivity("KYTY_PIPE_CONT change"); });

} // namespace Detail

namespace {

std::atomic<uint64_t> g_activity {1};
Totals                g_totals;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Seed-chained one-shot hashes (the streaming state type is opaque here).
uint64_t HashBytes(uint64_t hash, const void* data, size_t size) {
	return XXH3_64bits_withSeed(data, size, hash);
}

uint64_t HashValues(uint64_t hash, const std::vector<ShaderRecompiler::IR::DescriptorValue>& values) {
	const uint64_t count = values.size();
	hash                 = HashBytes(hash, &count, sizeof(count));
	for (const auto& value: values) {
		const auto dwords =
		    std::min<uint32_t>(value.dword_count, static_cast<uint32_t>(value.dwords.size()));
		hash = HashBytes(hash ^ value.dword_count, value.dwords.data(), dwords * sizeof(uint32_t));
	}
	return hash;
}

} // namespace

namespace {

// What advanced the activity epoch last, and how often each source ended a run (Miss::Activity):
// the "activity by" part of the 10-second line. The table is the GPU thread's (CountMiss).
std::atomic<const char*> g_last_activity {"start"};
struct ActivitySource {
	const char* what  = nullptr;
	uint64_t    count = 0;
	uint64_t    last  = 0;
};
std::array<ActivitySource, 24> g_activity_misses {};

} // namespace

void NoteForeignActivity(const char* what) noexcept {
	g_last_activity.store(what, std::memory_order_relaxed);
	g_activity.fetch_add(1, std::memory_order_relaxed);
}

bool QuietOpsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_RUN_QUIET_OPS");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_DRAW_RUN_QUIET_SYNTHETIC (live, default on): a mode change ends the current run.
Live::Switch g_quiet_synthetic("KYTY_DRAW_RUN_QUIET_SYNTHETIC", Live::ParseDefaultOn,
                               [](int64_t, int64_t) { NoteForeignActivity(); });

bool QuietSyntheticEnabled() {
	return g_quiet_synthetic.On();
}

bool IndirectRunsEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DRAW_RUN_INDIRECT");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint64_t ActivityEpoch() noexcept {
	return g_activity.load(std::memory_order_relaxed);
}

uint64_t StructureKey(const HW::Context& context, const HW::UserConfig& user_config,
                      const DrawPrep::PreparedDraw& prepared, uint64_t* targets_key,
                      uint64_t* swap_key, uint64_t* pipe_key) {
	if (targets_key != nullptr) {
		*targets_key = 0;
	}
	if (swap_key != nullptr) {
		*swap_key = 0;
	}
	if (pipe_key != nullptr) {
		*pipe_key = 0;
	}
	if (!prepared.ok || prepared.failure != DrawPrep::Failure::None ||
	    !prepared.vertex_info.stage.program) {
		return 0;
	}
	// Every context register (targets, depth/stencil, blend, raster, viewports, shader registers).
	uint64_t hash = HashBytes(0x9e3779b97f4a7c15ull, &context, sizeof(context));
	// The user-config registers but the per-draw index offset and object id.
	auto config = user_config;
	config.SetIndexOffset(0);
	config.SetObjectId(0);
	hash = HashBytes(hash, &config, sizeof(config));
	if (targets_key != nullptr) {
		*targets_key = hash != 0 ? hash : 1;
	}
	// The programs and the T#/S# words of both stages (V# words, user data and the flattened SRT are
	// per-draw data).
	const std::array<const void*, 2> programs {prepared.vertex_info.stage.program,
	                                          prepared.pixel_active ? prepared.pixel_info.stage.program
	                                                                : nullptr};
	const uint32_t flags = (prepared.pixel_active ? 1u : 0u) |
	                       (prepared.programs.VertexStageCount() == 1 ? 2u : 0u);
	// The words: hashed once, then chained into the key with and without the vertex program.
	uint64_t words = HashValues(0x51ed27c3ull, prepared.vertex_prep.resources.images);
	words          = HashValues(words, prepared.vertex_prep.resources.samplers);
	if (prepared.pixel_active) {
		words = HashValues(words, prepared.pixel_prep.resources.images);
		words = HashValues(words, prepared.pixel_prep.resources.samplers);
	}
	const auto chain = [&](const std::array<const void*, 2>& chained) {
		uint64_t result = HashBytes(hash, chained.data(), sizeof(chained));
		result          = HashBytes(result, &flags, sizeof(flags));
		result          = HashBytes(result, &words, sizeof(words));
		return result != 0 ? result : 1;
	};
	if (swap_key != nullptr) {
		*swap_key = chain({nullptr, programs[1]});
	}
	if (pipe_key != nullptr) {
		*pipe_key = chain({nullptr, nullptr});
	}
	return chain(programs);
}

Totals& GetTotals() {
	return g_totals;
}

namespace {

Live::Switch g_key_diag("KYTY_DRAW_RUN_KEY_DIAG", Live::ParseDefaultOff);

const char* const kKeyPartNames[KpCount] = {"targets", "clear",    "depth",       "blend",    "blendcolor",
                                            "depthstate", "stencil", "clearvalues", "viewport", "raster",
                                            "shregs",  "ucfg",     "vsprog",      "psprog",   "vsT",
                                            "vsS",     "psT",      "psS"};
const char* const kIneligNames[static_cast<size_t>(KeyDiag::Inelig::Count)] = {
    "none",        "notreached", "notengine",  "notvalidated", "nokey",         "mesh",
    "indirect",    "vstages",    "writes",     "feedback",     "norendering",   "renderstate",
    "debugdump",   "depthclear", "stencilclear", "colorclear", "gds",           "program",
    "texattachment", "storage",  "depthpromotion", "alias"};

// GPU thread only (the commit path and PrintSummary).
struct KeyDiagCombo {
	uint32_t mask        = 0;
	uint64_t count       = 0;
	uint64_t resolved    = 0; // reached the structure comparison
	uint64_t same_pipe   = 0;
	uint64_t same_struct = 0; // same pipeline and structure
};
struct KeyDiagState {
	constexpr static size_t           Inelig = static_cast<size_t>(KeyDiag::Inelig::Count);
	std::array<uint64_t, Inelig>      inelig {};
	std::array<uint64_t, Inelig>      prev_invalid {};
	std::array<uint64_t, Inelig>      prev_invalid_same {};
	uint64_t                          key_diff       = 0;
	uint64_t                          recallable     = 0;
	uint64_t                          recallable_any = 0;
	std::array<uint64_t, KpCount + 1> parts {}; // the last one: no part differs (padding)
	std::array<KeyDiagCombo, 64>      combos {};
	uint32_t                          combo_used     = 0;
	uint64_t                          combo_overflow = 0;
	uint64_t                          resolved = 0, same_pipe = 0, same_struct = 0;
};
KeyDiagState g_kd;

} // namespace

bool KeyDiagEnabled() {
	return g_key_diag.On();
}

void StructureKeyParts(const HW::Context& context, const HW::UserConfig& user_config,
                       const DrawPrep::PreparedDraw& prepared, KeyParts& parts) {
	parts.fill(0);
	if (!prepared.ok || prepared.failure != DrawPrep::Failure::None ||
	    !prepared.vertex_info.stage.program) {
		return;
	}
	context.VisitKeyDiagParts([&parts](uint32_t part, const void* data, size_t size) {
		parts[part] = HashBytes(parts[part] + 0x9e3779b97f4a7c15ull * (part + 1), data, size);
	});
	auto config = user_config;
	config.SetIndexOffset(0);
	config.SetObjectId(0);
	parts[KpUserConfig]  = HashBytes(11, &config, sizeof(config));
	const uint32_t flags = (prepared.pixel_active ? 1u : 0u) |
	                       (prepared.programs.VertexStageCount() == 1 ? 2u : 0u);
	const void* vs = prepared.vertex_info.stage.program;
	const void* ps = prepared.pixel_active ? prepared.pixel_info.stage.program : nullptr;
	parts[KpVsProgram] = HashBytes(HashBytes(12, &vs, sizeof(vs)), &flags, sizeof(flags));
	parts[KpPsProgram] = HashBytes(13, &ps, sizeof(ps));
	parts[KpVsT]       = HashValues(14, prepared.vertex_prep.resources.images);
	parts[KpVsS]       = HashValues(15, prepared.vertex_prep.resources.samplers);
	if (prepared.pixel_active) {
		parts[KpPsT] = HashValues(16, prepared.pixel_prep.resources.images);
		parts[KpPsS] = HashValues(17, prepared.pixel_prep.resources.samplers);
	}
}

namespace KeyDiag {

void CountIneligible(Inelig reason) noexcept {
	g_kd.inelig[static_cast<size_t>(reason)]++;
}

void CountPrevInvalid(Inelig prev, bool same_key) noexcept {
	g_kd.prev_invalid[static_cast<size_t>(prev)]++;
	g_kd.prev_invalid_same[static_cast<size_t>(prev)] += same_key ? 1u : 0u;
}

int CountKeyMismatch(uint32_t mask, bool recallable, bool recallable_any_activity) noexcept {
	auto& kd = g_kd;
	kd.key_diff++;
	kd.recallable += recallable ? 1u : 0u;
	kd.recallable_any += recallable_any_activity ? 1u : 0u;
	if (mask == 0) {
		kd.parts[KpCount]++;
	}
	for (uint32_t i = 0; i < KpCount; i++) {
		kd.parts[i] += (mask >> i) & 1u;
	}
	for (uint32_t i = 0; i < kd.combo_used; i++) {
		if (kd.combos[i].mask == mask) {
			kd.combos[i].count++;
			return static_cast<int>(i);
		}
	}
	if (kd.combo_used < kd.combos.size()) {
		auto& combo = kd.combos[kd.combo_used];
		combo.mask  = mask;
		combo.count = 1;
		return static_cast<int>(kd.combo_used++);
	}
	kd.combo_overflow++;
	return -1;
}

void CountResolved(int index, bool same_pipeline, bool same_structure) noexcept {
	auto& kd = g_kd;
	kd.resolved++;
	kd.same_pipe += same_pipeline ? 1u : 0u;
	kd.same_struct += same_pipeline && same_structure ? 1u : 0u;
	if (index >= 0 && static_cast<uint32_t>(index) < kd.combo_used) {
		auto& combo = kd.combos[static_cast<size_t>(index)];
		combo.resolved++;
		combo.same_pipe += same_pipeline ? 1u : 0u;
		combo.same_struct += same_pipeline && same_structure ? 1u : 0u;
	}
}

} // namespace KeyDiag

namespace {

void PrintKeyDiag(double seconds) {
	auto&       kd = g_kd;
	std::string line;
	char        buf[256];
	uint64_t    prev_total = 0;
	uint64_t    prev_same  = 0;
	for (size_t i = 0; i < kd.prev_invalid.size(); i++) {
		prev_total += kd.prev_invalid[i];
		prev_same += kd.prev_invalid_same[i];
	}
	std::snprintf(buf, sizeof(buf), "DrawRunKeyDiag %.0fs: prev-invalid %" PRIu64 " (same key %" PRIu64 ") by",
	              seconds, prev_total, prev_same);
	line += buf;
	for (size_t i = 0; i < kd.prev_invalid.size(); i++) {
		if (kd.prev_invalid[i] != 0) {
			std::snprintf(buf, sizeof(buf), " %s=%" PRIu64 "/%" PRIu64, kIneligNames[i], kd.prev_invalid[i],
			              kd.prev_invalid_same[i]);
			line += buf;
		}
	}
	std::snprintf(buf, sizeof(buf),
	              " | key-diff %" PRIu64 " (recallable %" PRIu64 ", any-activity %" PRIu64 "; resolved %" PRIu64
	              ", same pipeline %" PRIu64 ", same pipeline+structure %" PRIu64 ") parts",
	              kd.key_diff, kd.recallable, kd.recallable_any, kd.resolved, kd.same_pipe, kd.same_struct);
	line += buf;
	for (uint32_t i = 0; i <= KpCount; i++) {
		if (kd.parts[i] != 0) {
			std::snprintf(buf, sizeof(buf), " %s=%" PRIu64, i < KpCount ? kKeyPartNames[i] : "none",
			              kd.parts[i]);
			line += buf;
		}
	}
	// The most frequent exact combinations.
	auto sorted = kd.combos;
	std::sort(sorted.begin(), sorted.begin() + kd.combo_used,
	          [](const auto& a, const auto& b) { return a.count > b.count; });
	line += " combos";
	for (uint32_t i = 0; i < 20 && i < kd.combo_used; i++) {
		std::string name;
		for (uint32_t p = 0; p < KpCount; p++) {
			if ((sorted[i].mask >> p) & 1u) {
				name += name.empty() ? "" : "+";
				name += kKeyPartNames[p];
			}
		}
		if (name.empty()) {
			name = "none";
		}
		std::snprintf(buf, sizeof(buf), " [%s=%" PRIu64 " res %" PRIu64 " pipe %" PRIu64 " same %" PRIu64 "]",
		              name.c_str(), sorted[i].count, sorted[i].resolved, sorted[i].same_pipe,
		              sorted[i].same_struct);
		line += buf;
	}
	if (kd.combo_overflow != 0) {
		std::snprintf(buf, sizeof(buf), " overflow=%" PRIu64, kd.combo_overflow);
		line += buf;
	}
	uint64_t inelig_total = 0;
	for (size_t i = 1; i < kd.inelig.size(); i++) {
		inelig_total += kd.inelig[i];
	}
	std::snprintf(buf, sizeof(buf), " | ineligible %" PRIu64 " by", inelig_total);
	line += buf;
	for (size_t i = 1; i < kd.inelig.size(); i++) {
		if (kd.inelig[i] != 0) {
			std::snprintf(buf, sizeof(buf), " %s=%" PRIu64, kIneligNames[i], kd.inelig[i]);
			line += buf;
		}
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
	kd = KeyDiagState {};
}

} // namespace

void CountMiss(Miss miss) noexcept {
	OwnedCounters::Add(g_totals.misses[static_cast<uint32_t>(miss)], 1);
	if (miss == Miss::Activity) {
		const char* what = g_last_activity.load(std::memory_order_relaxed);
		for (auto& source: g_activity_misses) {
			if (source.what == what || source.what == nullptr) {
				source.what = what;
				source.count++;
				break;
			}
		}
	}
}

void CountVerifyCheck() noexcept {
	g_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
}

uint32_t MismatchLogLimit() {
	static const uint32_t limit = [] {
		const auto* value = std::getenv("KYTY_DRAW_RUN_LOG_LIMIT");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 64u;
	}();
	return limit;
}

void ReportMismatch(const char* what, uint64_t detail) {
	g_totals.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < MismatchLogLimit()) {
		const char* part = std::strstr(what, "acquisition") ? "acquisition" :
		                   std::strstr(what, "transition") ? "transitions" :
		                   std::strstr(what, "dynamic") ? "dynamic state" :
		                   std::strstr(what, "view") ? "views" :
		                   std::strstr(what, "sampler") ? "samplers" :
		                   std::strstr(what, "texture") ? "textures" : "targets";
		std::printf("DrawRunVerify: part=%s: a continuation would have reused a different %s (detail %" PRIu64
		            " = 0x%" PRIx64 ")\n",
		            part, what, detail, detail);
		std::fflush(stdout);
	}
	if (VerifyExit()) {
		EXIT("DrawRunVerify: a continuation would have reused a different %s\n", what);
	}
}

void PrintSummary() {
	if (!Enabled()) {
		return;
	}
	static uint64_t                                          last_ns = 0;
	static std::array<uint64_t, 27>                          last {};
	static std::array<uint64_t, static_cast<size_t>(Miss::Count)> last_misses {};
	// KYTY_CP_OWNED_COUNTERS (ownedCounters.h): the clock is read every 256th call (the line is
	// due every 10 s; the commit path calls this once per draw).
	static uint32_t calls = 0;
	if (OwnedCounters::g_mode.Get() != 0 && last_ns != 0 && (++calls & 255u) != 0) {
		return;
	}
	const auto                                               now = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	const std::array<uint64_t, 27> values {
	    g_totals.draws.load(std::memory_order_relaxed),
	    g_totals.eligible.load(std::memory_order_relaxed),
	    g_totals.key_matches.load(std::memory_order_relaxed),
	    g_totals.continued.load(std::memory_order_relaxed),
	    g_totals.late_fallbacks.load(std::memory_order_relaxed),
	    g_totals.pipeline_lookups.load(std::memory_order_relaxed),
	    g_totals.verify_checks.load(std::memory_order_relaxed),
	    g_totals.verify_mismatches.load(std::memory_order_relaxed),
	    g_totals.alias_excluded.load(std::memory_order_relaxed),
	    g_totals.acquire_reused.load(std::memory_order_relaxed),
	    g_totals.dynamic_emitted.load(std::memory_order_relaxed),
	    g_totals.partial_pushes.load(std::memory_order_relaxed),
	    g_totals.depth_promotions_excluded.load(std::memory_order_relaxed),
	    g_totals.depth_promotions_deferred.load(std::memory_order_relaxed),
	    g_totals.write_records.load(std::memory_order_relaxed),
	    g_totals.write_continued.load(std::memory_order_relaxed),
	    g_totals.recall_matches.load(std::memory_order_relaxed),
	    g_totals.recalled.load(std::memory_order_relaxed),
	    g_totals.recall_sets_missed.load(std::memory_order_relaxed),
	    g_totals.epoch_skips.load(std::memory_order_relaxed),
	    g_totals.epoch_checks.load(std::memory_order_relaxed),
	    g_totals.epoch_mismatches.load(std::memory_order_relaxed),
	    g_totals.targets_kept.load(std::memory_order_relaxed),
	    g_totals.targets_late.load(std::memory_order_relaxed),
	    g_totals.swap_matches.load(std::memory_order_relaxed),
	    g_totals.swapped.load(std::memory_order_relaxed),
	    g_totals.swap_incompatible.load(std::memory_order_relaxed)};
	std::array<uint64_t, 27> delta {};
	for (size_t i = 0; i < values.size(); i++) {
		delta[i] = values[i] - last[i];
	}
	static const char* const miss_names[] = {"activity", "command",  "instance", "programs",
	                                         "validation", "images", "pipeline", "writes"};
	std::string misses;
	for (size_t i = 0; i < last_misses.size(); i++) {
		const auto value = g_totals.misses[i].load(std::memory_order_relaxed);
		if (value != last_misses[i]) {
			misses += " " + std::string(miss_names[i]) + "=" + std::to_string(value - last_misses[i]);
			last_misses[i] = value;
		}
	}
	std::string by;
	for (auto& source: g_activity_misses) {
		if (source.what != nullptr && source.count != source.last) {
			by += " " + std::string(source.what) + "=" + std::to_string(source.count - source.last);
			source.last = source.count;
		}
	}
	if (!by.empty()) {
		misses += " (activity by" + by + ")";
	}
	std::printf("DrawRun %.0fs (%s): %" PRIu64 " draws, %" PRIu64 " eligible, %" PRIu64
	            " key matches, %" PRIu64 " continued (%.1f%%), %" PRIu64
	            " late fallbacks; misses:%s; verify %" PRIu64 " checks, %" PRIu64
	            " mismatches; %" PRIu64 " alias-excluded, %" PRIu64 " acquisitions reused, %" PRIu64
	            " dynamic re-emitted, %" PRIu64 " partial pushes, %" PRIu64 " depth-promotions excluded, %" PRIu64
	            " deferred; writes: %" PRIu64 " records, %" PRIu64 " continued; recall: %" PRIu64
	            " matches, %" PRIu64 " recalled, %" PRIu64 " sets missed; image epoch: %" PRIu64
	            " skips, verify %" PRIu64 " checks, %" PRIu64 " mismatches; targets: %" PRIu64
	            " kept, %" PRIu64 " late; pipeswap: %" PRIu64 " matches, %" PRIu64 " swapped, %" PRIu64
	            " incompatible\n",
	            static_cast<double>(now - last_ns) * 1e-9,
	            GetMode() == Mode::Verify ? "verify" : "on", delta[0], delta[1], delta[2], delta[3],
	            delta[0] != 0 ? 100.0 * static_cast<double>(delta[3]) / static_cast<double>(delta[0])
	                          : 0.0,
	            delta[4], misses.empty() ? " none" : misses.c_str(), delta[6], delta[7], delta[8],
	            delta[9], delta[10], delta[11], delta[12], delta[13], delta[14], delta[15], delta[16],
	            delta[17], delta[18], delta[19], delta[20], delta[21], delta[22], delta[23], delta[24],
	            delta[25], delta[26]);
	if (PipeContMode() != 0) {
		static std::array<uint64_t, 5 + static_cast<size_t>(PipeReject::Count)> pc_last {};
		std::array<uint64_t, 5 + static_cast<size_t>(PipeReject::Count)> pc {};
		pc[0] = g_totals.pc_attempts.load(std::memory_order_relaxed);
		pc[1] = g_totals.pc_accepted.load(std::memory_order_relaxed);
		pc[2] = g_totals.pc_ps_changed.load(std::memory_order_relaxed);
		pc[3] = g_totals.pc_partial.load(std::memory_order_relaxed);
		pc[4] = g_totals.pc_key_misses.load(std::memory_order_relaxed);
		for (size_t i = 0; i < static_cast<size_t>(PipeReject::Count); i++) {
			pc[5 + i] = g_totals.pc_rejects[i].load(std::memory_order_relaxed);
		}
		std::array<uint64_t, pc.size()> d {};
		for (size_t i = 0; i < pc.size(); i++) {
			d[i] = pc[i] - pc_last[i];
		}
		pc_last = pc;
		std::printf("DrawRun pipecont (mode %" PRId64 "): %" PRIu64 " attempts, %" PRIu64
		            " accepted (%" PRIu64 " new pixel program), %" PRIu64
		            " partial pushes; rejected: cert=%" PRIu64 " vs-bindings=%" PRIu64
		            " vs-outputs=%" PRIu64 " ps-bindings=%" PRIu64 " ps-outputs=%" PRIu64
		            " writes=%" PRIu64 "; pipe-key differs (regs/words/flags)=%" PRIu64 "\n",
		            PipeContMode(), d[0], d[1], d[2], d[3], d[6], d[7], d[8], d[9], d[10], d[11], d[4]);
	}
	std::fflush(stdout);
	if (KeyDiagEnabled()) {
		PrintKeyDiag(static_cast<double>(now - last_ns) * 1e-9);
	}
	last    = values;
	last_ns = now;
}

} // namespace Libs::Graphics::DrawRun
