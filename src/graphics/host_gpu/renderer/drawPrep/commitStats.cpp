#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include <xxhash.h>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace Libs::Graphics::CommitStats {

namespace Detail {
bool ReadEnabled() {
	const auto* value = std::getenv("KYTY_CP_COMMIT_STATS");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
} // namespace Detail

uint64_t Hash(const void* data, uint64_t size, uint64_t seed) {
	return XXH3_64bits_withSeed(data, static_cast<size_t>(size), seed);
}

namespace {

uint64_t Tsc() {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
#endif
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

constexpr size_t PhaseCount = static_cast<size_t>(Phase::Count);

enum DrawClass : uint32_t { ClassStart, ClassContinue, ClassOther, ClassCount };

struct Aggregate {
	uint64_t                           draws  = 0;
	uint64_t                           gap    = 0; // cycles since the previous commit ended
	std::array<uint64_t, PhaseCount>   cycles {};
};

constexpr uint32_t RunBuckets = 8; // 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65+

uint32_t RunBucket(uint32_t length) {
	uint32_t bucket = 0;
	uint32_t limit  = 1;
	while (bucket + 1 < RunBuckets && length > limit) {
		bucket++;
		limit *= 2;
	}
	return bucket;
}

struct State {
	// The draw being committed.
	bool                             active   = false;
	bool                             recorded = false;
	uint64_t                         begin    = 0;
	uint64_t                         last     = 0;
	uint64_t                         gap      = 0;
	std::array<uint64_t, PhaseCount> cycles {};
	DrawShape                        shape;
	// The previous recorded draw.
	bool      previous_valid = false;
	DrawShape previous;
	uint64_t  previous_end = 0;
	uint32_t  run_length   = 0;
	// Interval totals.
	std::array<Aggregate, ClassCount> classes {};
	std::array<uint64_t, RunBuckets>  runs {};      // runs ended, by length bucket
	std::array<uint64_t, RunBuckets>  run_draws {}; // their draws
	// Continuations whose per-draw data repeats the previous draw's.
	uint64_t same_buffers = 0;
	uint64_t same_push    = 0;
	uint64_t same_tables  = 0;
	uint64_t same_index   = 0;
	uint64_t same_vertex  = 0;
	uint64_t same_all     = 0;
	// Why a draw started a new run (first difference, in this order).
	uint64_t why_first     = 0;
	uint64_t why_instance  = 0;
	uint64_t why_pipeline  = 0;
	uint64_t why_structure = 0;
	uint64_t why_dynamic   = 0;
	uint64_t why_writes    = 0;
	// Cross-frame.
	std::vector<uint64_t>        frame_bind;
	std::vector<uint64_t>        frame_struct;
	std::unordered_set<uint64_t> bind_1, bind_2, struct_1, struct_2;
	uint64_t                     frames        = 0;
	uint64_t                     xf_draws      = 0;
	uint64_t                     xf_bind_1     = 0;
	uint64_t                     xf_bind_12    = 0;
	uint64_t                     xf_struct_1   = 0;
	uint64_t                     xf_struct_12  = 0;
	uint64_t                     xf_bind_in    = 0; // within the frame (an earlier draw of it)
	std::unordered_set<uint64_t> bind_frame_set;
	// Interval.
	uint64_t interval_ns  = 0;
	uint64_t interval_tsc = 0;
};

State& GetState() {
	// GPU thread only.
	static State state;
	return state;
}

void EndRun(State& s) {
	if (s.run_length == 0) {
		return;
	}
	const auto bucket = RunBucket(s.run_length);
	s.runs[bucket]++;
	s.run_draws[bucket] += s.run_length;
	s.run_length = 0;
}

void Print(State& s, uint64_t now_ns, uint64_t now_tsc) {
	if (s.interval_ns == 0) {
		s.interval_ns  = now_ns;
		s.interval_tsc = now_tsc;
		return;
	}
	if (now_ns - s.interval_ns < 10'000'000'000ull) {
		return;
	}
	const double ns_per_tick = static_cast<double>(now_ns - s.interval_ns) /
	                           static_cast<double>(std::max<uint64_t>(1, now_tsc - s.interval_tsc));
	const double seconds     = static_cast<double>(now_ns - s.interval_ns) * 1e-9;
	static const char* const phase_names[PhaseCount] = {
	    "setup", "programs", "targets", "prepbind", "findbuf", "images", "buffers",
	    "vtxidx", "pipeline", "acquire", "commitbind", "emit", "post"};
	static const char* const class_names[ClassCount] = {"start", "cont", "other"};
	uint64_t draws = 0;
	for (const auto& c: s.classes) {
		draws += c.draws;
	}
	std::string line;
	char        text[256];
	std::snprintf(text, sizeof(text),
	              "CommitStats %.0fs: frames %" PRIu64 " draws %" PRIu64 " (%.0f/s)", seconds,
	              s.frames, draws, static_cast<double>(draws) / seconds);
	line += text;
	for (uint32_t k = 0; k < ClassCount; k++) {
		const auto& c = s.classes[k];
		if (c.draws == 0) {
			continue;
		}
		uint64_t total = 0;
		for (const auto v: c.cycles) {
			total += v;
		}
		const double scale = ns_per_tick * 1e-3 / static_cast<double>(c.draws);
		std::snprintf(text, sizeof(text), " | %s %" PRIu64 " (%.1f%%) us/draw %.2f gap %.2f [",
		              class_names[k], c.draws,
		              100.0 * static_cast<double>(c.draws) / static_cast<double>(draws),
		              static_cast<double>(total) * scale, static_cast<double>(c.gap) * scale);
		line += text;
		for (size_t p = 0; p < PhaseCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", phase_names[p],
			              static_cast<double>(c.cycles[p]) * scale);
			line += text;
		}
		line += "]";
	}
	const auto cont = std::max<uint64_t>(1, s.classes[ClassContinue].draws);
	const auto pct  = [](uint64_t part, uint64_t whole) {
        return 100.0 * static_cast<double>(part) / static_cast<double>(std::max<uint64_t>(1, whole));
	};
	std::snprintf(text, sizeof(text),
	              " | cont same: buffers %.0f%% push %.0f%% tables %.0f%% index %.0f%% vertex "
	              "%.0f%% all %.0f%%",
	              pct(s.same_buffers, cont), pct(s.same_push, cont), pct(s.same_tables, cont),
	              pct(s.same_index, cont), pct(s.same_vertex, cont), pct(s.same_all, cont));
	line += text;
	const auto starts = s.why_first + s.why_instance + s.why_pipeline + s.why_structure +
	                    s.why_dynamic + s.why_writes;
	std::snprintf(text, sizeof(text),
	              " | start why: first %.0f%% instance %.0f%% pipeline %.0f%% structure %.0f%% "
	              "dynamic %.0f%% writes %.0f%%",
	              pct(s.why_first, starts), pct(s.why_instance, starts), pct(s.why_pipeline, starts),
	              pct(s.why_structure, starts), pct(s.why_dynamic, starts), pct(s.why_writes, starts));
	line += text;
	line += " | runs (len:runs/draws)";
	static const char* const bucket_names[RunBuckets] = {"1",    "2",     "3-4",   "5-8",
	                                                     "9-16", "17-32", "33-64", "65+"};
	for (uint32_t b = 0; b < RunBuckets; b++) {
		std::snprintf(text, sizeof(text), " %s:%" PRIu64 "/%" PRIu64, bucket_names[b], s.runs[b],
		              s.run_draws[b]);
		line += text;
	}
	std::snprintf(text, sizeof(text),
	              " | xframe draws %" PRIu64 " bind f-1 %.1f%% f-1|2 %.1f%% same-frame %.1f%%; "
	              "struct f-1 %.1f%% f-1|2 %.1f%%",
	              s.xf_draws, pct(s.xf_bind_1, s.xf_draws), pct(s.xf_bind_12, s.xf_draws),
	              pct(s.xf_bind_in, s.xf_draws), pct(s.xf_struct_1, s.xf_draws),
	              pct(s.xf_struct_12, s.xf_draws));
	line += text;
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
	s.classes      = {};
	s.runs         = {};
	s.run_draws    = {};
	s.same_buffers = s.same_push = s.same_tables = s.same_index = s.same_vertex = s.same_all = 0;
	s.why_first = s.why_instance = s.why_pipeline = s.why_structure = s.why_dynamic =
	    s.why_writes                                                 = 0;
	s.frames = s.xf_draws = s.xf_bind_1 = s.xf_bind_12 = s.xf_struct_1 = s.xf_struct_12 =
	    s.xf_bind_in                                                    = 0;
	s.interval_ns  = now_ns;
	s.interval_tsc = now_tsc;
}

} // namespace

void BeginDraw() {
	if (!Enabled()) {
		return;
	}
	auto&      s   = GetState();
	const auto now = Tsc();
	s.active       = true;
	s.recorded     = false;
	s.begin        = now;
	s.last         = now;
	s.gap          = s.previous_end != 0 ? now - s.previous_end : 0;
	s.cycles       = {};
}

void Mark(Phase phase) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (!s.active) {
		return;
	}
	const auto now = Tsc();
	s.cycles[static_cast<size_t>(phase)] += now - s.last;
	s.last = now;
}

void Skip() {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (s.active) {
		s.last = Tsc();
	}
}

void NoteRecorded(const DrawShape& shape) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (!s.active) {
		return;
	}
	s.recorded = true;
	s.shape    = shape;
}

void EndDraw() {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (!s.active) {
		return;
	}
	const auto now = Tsc();
	s.cycles[static_cast<size_t>(Phase::Post)] += now - s.last;
	s.active         = false;
	s.previous_end   = now;
	DrawClass klass  = ClassOther;
	if (s.recorded) {
		const auto& a = s.previous;
		const auto& b = s.shape;
		if (!s.previous_valid) {
			s.why_first++;
			klass = ClassStart;
		} else if (a.rendering_serial == 0 || a.rendering_serial != b.rendering_serial ||
		           a.command != b.command) {
			s.why_instance++;
			klass = ClassStart;
		} else if (a.pipeline != b.pipeline) {
			s.why_pipeline++;
			klass = ClassStart;
		} else if (a.structure != b.structure) {
			s.why_structure++;
			klass = ClassStart;
		} else if (a.dynamic != b.dynamic) {
			s.why_dynamic++;
			klass = ClassStart;
		} else if (a.shader_writes) {
			s.why_writes++;
			klass = ClassStart;
		} else {
			klass = ClassContinue;
			const bool buffers = a.buffers == b.buffers;
			const bool push    = a.push == b.push;
			const bool tables  = a.tables == b.tables;
			const bool index   = a.index == b.index;
			const bool vertex  = a.vertex == b.vertex;
			s.same_buffers += buffers ? 1u : 0u;
			s.same_push += push ? 1u : 0u;
			s.same_tables += tables ? 1u : 0u;
			s.same_index += index ? 1u : 0u;
			s.same_vertex += vertex ? 1u : 0u;
			s.same_all += buffers && push && tables && index && vertex ? 1u : 0u;
		}
		if (klass == ClassStart) {
			EndRun(s);
		}
		s.run_length++;
		s.previous       = b;
		s.previous_valid = true;
		// Cross-frame.
		s.xf_draws++;
		s.xf_bind_1 += s.bind_1.contains(b.bind_input) ? 1u : 0u;
		s.xf_bind_12 += s.bind_1.contains(b.bind_input) || s.bind_2.contains(b.bind_input) ? 1u : 0u;
		s.xf_bind_in += s.bind_frame_set.contains(b.bind_input) ? 1u : 0u;
		s.xf_struct_1 += s.struct_1.contains(b.struct_input) ? 1u : 0u;
		s.xf_struct_12 +=
		    s.struct_1.contains(b.struct_input) || s.struct_2.contains(b.struct_input) ? 1u : 0u;
		s.frame_bind.push_back(b.bind_input);
		s.frame_struct.push_back(b.struct_input);
		s.bind_frame_set.insert(b.bind_input);
	}
	auto& c = s.classes[klass];
	c.draws++;
	c.gap += s.gap;
	for (size_t p = 0; p < PhaseCount; p++) {
		c.cycles[p] += s.cycles[p];
	}
	Print(s, NowNs(), now);
}

void OnFrameBoundary() {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	s.frames++;
	s.bind_2   = std::move(s.bind_1);
	s.struct_2 = std::move(s.struct_1);
	s.bind_1.clear();
	s.struct_1.clear();
	s.bind_1.insert(s.frame_bind.begin(), s.frame_bind.end());
	s.struct_1.insert(s.frame_struct.begin(), s.frame_struct.end());
	s.frame_bind.clear();
	s.frame_struct.clear();
	s.bind_frame_set.clear();
	// A frame boundary is never inside a run of one rendering instance's draws.
	EndRun(s);
	s.previous_valid = false;
}

} // namespace Libs::Graphics::CommitStats
