#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"

#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

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
bool ReadIndirectOnly() {
	const auto* value = std::getenv("KYTY_CP_COMMIT_STATS");
	return value != nullptr && std::strcmp(value, "indirect") == 0;
}
bool ReadSteps() {
	const auto* value = std::getenv("KYTY_CP_COMMIT_STATS");
	return value != nullptr && std::strcmp(value, "steps") == 0;
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
constexpr size_t SubCount   = static_cast<size_t>(Sub::Count);
constexpr size_t ItemCount  = static_cast<size_t>(Item::Count);
constexpr size_t StepCount  = static_cast<size_t>(Step::Count);

enum DrawClass : uint32_t { ClassStart, ClassContinue, ClassOther, ClassCount };

// StartPhases: the actual draw-run decision, and for a start the first shape difference.
enum Reason : uint32_t {
	ReasonCont,
	ReasonFirst,
	ReasonInstance,
	ReasonPipeline,
	ReasonStructure,
	ReasonDynamic,
	ReasonWrites,
	ReasonRunMiss,
	ReasonCount
};
const char* const kReasonNames[ReasonCount] = {"cont",      "first",   "instance", "pipeline",
                                               "structure", "dynamic", "writes",   "runmiss"};

struct ReasonAggregate {
	uint64_t                               draws     = 0;
	uint64_t                               gap       = 0;
	uint64_t                               gap_draws = 0; // draws whose gap split is known
	uint64_t                               gap_known = 0; // their gap
	std::array<uint64_t, PhaseCount>       cycles {};
	std::array<uint64_t, SubCount>         sub {};
	std::array<uint64_t, ItemCount>        items {};
	std::array<uint64_t, CpGaps::CatCount> gap_cats {};
	std::array<uint64_t, StepCount>        steps {};     // KYTY_CP_COMMIT_STATS=steps: commit
	std::array<uint64_t, StepCount>        gap_steps {}; // and the gap before the draw
};

struct Aggregate {
	uint64_t                           draws  = 0;
	uint64_t                           gap    = 0; // cycles since the previous commit ended
	uint64_t                           head_self = 0; // of which: head prepared on this thread
	uint64_t                           head_wait = 0; // of which: waiting for a worker's head
	uint64_t                           self_draws = 0;
	uint64_t                           wait_draws = 0;
	std::array<uint64_t, PhaseCount>   cycles {};
	std::array<uint64_t, SubCount>     sub {};
	std::array<uint64_t, ItemCount>    items {};
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
	uint64_t                         head_self = 0;
	uint64_t                         head_wait = 0;
	uint64_t                         pending_self = 0;
	uint64_t                         pending_wait = 0;
	std::array<uint64_t, PhaseCount> cycles {};
	std::array<uint64_t, SubCount>   sub {};
	std::array<uint64_t, ItemCount>  items {};
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
	// The draws KYTY_DRAW_RUN actually continued (DrawShape::run_continued), and their shares:
	// of the "writes" starts above, and every draw's phase time by the actual run decision.
	uint64_t why_writes_run   = 0;
	uint64_t actual_cont      = 0;
	uint64_t actual_cont_cyc  = 0;
	uint64_t actual_start     = 0;
	uint64_t actual_start_cyc = 0;
	uint64_t writes_prev_cont = 0; // continued draws whose previous draw wrote (run writes)
	// Starts whose pipeline and structure (DrawShape) an earlier draw of the same rendering
	// instance, or of the same frame, recorded; and whose structure inputs the previous frame or the
	// one before had.
	std::unordered_set<uint64_t> instance_shapes, frame_shapes;
	uint64_t                     instance_serial    = 0;
	uint64_t                     start_inst_repeat  = 0;
	uint64_t                     start_frame_repeat = 0;
	uint64_t                     start_xf_struct12  = 0;
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
	// StartPhases.
	std::array<ReasonAggregate, ReasonCount> reasons {};
	// CpGaps ledger totals at the previous draw's end and their change up to this draw's start.
	std::array<uint64_t, CpGaps::CatCount> gap_snapshot {};
	bool                                   gap_snapshot_valid = false;
	std::array<uint64_t, CpGaps::CatCount> gap_delta {};
	bool                                   gap_delta_valid = false;
	// KYTY_CP_COMMIT_STATS=steps: the boundary cursor, this draw's steps and the gap's steps
	// closed since the previous draw ended (moved into gap_steps at BeginDraw).
	uint64_t                        step_last = 0;
	std::array<uint64_t, StepCount> steps {};
	std::array<uint64_t, StepCount> gap_steps {};
	std::array<uint64_t, StepCount> pending_gap_steps {};
	// Interval.
	uint64_t interval_ns  = 0;
	uint64_t interval_tsc = 0;
};

// The GPU thread's CpGaps ledger totals (false when that diagnostic is off).
bool ReadLedger(std::array<uint64_t, CpGaps::CatCount>& ticks) {
	if (!CpGaps::Enabled()) {
		return false;
	}
	const auto* state = CpGaps::t_state;
	if (state == nullptr) {
		return false;
	}
	for (size_t i = 0; i < CpGaps::CatCount; i++) {
		ticks[i] = state->ledger.Ticks(static_cast<CpGaps::Cat>(i));
	}
	return true;
}

const char* const kPhaseNames[PhaseCount] = {"setup",  "programs", "targets", "prepbind", "findbuf",
                                             "images", "buffers",  "vtxidx",  "pipeline", "acquire",
                                             "commitbind", "emit", "post"};
const char* const kSubNames[SubCount]     = {
    "textures",    "samplers",  "views",     "colors",   "depth",    "acqcolor",
    "acqdepth",    "push",      "dynamic",   "begin",    "acqcand",  "runrecord",
    "prefetch",    "latefallback", "vtxcommit", "transitions", "descbuild", "pushconst",
    "descpush",    "descset",   "bindpipe",  "barriers", "drawcmd",
    "readhit",     "readmiss",  "readsig",  "vallog",   "valclean"};
const char* const kItemNames[ItemCount] = {
    "textures",  "samplers",   "set-current", "set-history",   "set-miss",
    "set-repeated", "color-hit", "color-miss", "depth-hit",    "depth-miss",
    "target-repeat", "target-lookup", "descwrites", "transitions", "pipeline-new",
    "read-hits",  "read-misses", "log-entries", "cert-ranges"};

const char* const kStepNames[StepCount] = {
    "g.committail", "g.retire",      "g.opexit",     "g.loop",        "g.ophead",      "g.drawop",
    "g.head",       "g.commitin",    "bindregs",     "popready",      "debuginfo",     "lock",
    "beginrun",     "targetops",     "vscheck",      "drawchecks",    "statereset",    "psactive",
    "takeprepared", "validate",      "apply",        "runcandidate",  "markbindings",  "restart",
    "exechead",     "prepstages",    "findbuffers",  "bda",           "images",        "rbhead",
    "rbbindings",   "rbmipstats",    "rbtables",     "rbbatchend",    "vertexacquire", "indexprepare",
    "latecheck",    "pipelinesel",   "acquire",      "barriersafe",   "drawscope",     "vertexcommit",
    "cbhead",       "cbtransitions", "cbbuild",      "cbpushconst",   "cbdescriptors", "cbtail",
    "indexcommit",  "dynamicstate",  "begin",        "bindpipeline",  "drawcmd",       "occlusion",
    "runrecord",    "resetbindings", "committail",   "calib"};

void PrintSteps(State& s, double seconds, double ns_per_tick) {
	const double frames = static_cast<double>(std::max<uint64_t>(1, s.frames));
	std::string  line;
	char         text[256];
	std::snprintf(text, sizeof(text), "Steps %.0fs: frames %" PRIu64, seconds, s.frames);
	line += text;
	for (uint32_t r = 0; r < ReasonCount; r++) {
		const auto& a = s.reasons[r];
		if (a.draws == 0) {
			continue;
		}
		const double scale  = ns_per_tick * 1e-3 / static_cast<double>(a.draws);
		uint64_t     commit = 0;
		uint64_t     gap    = 0;
		for (size_t p = 0; p < StepCount; p++) {
			commit += a.steps[p];
			gap += a.gap_steps[p];
		}
		std::snprintf(text, sizeof(text), " | %s %.1f/f commit %.3f gap %.3f [", kReasonNames[r],
		              static_cast<double>(a.draws) / frames, static_cast<double>(commit) * scale,
		              static_cast<double>(gap) * scale);
		line += text;
		bool first = true;
		for (size_t p = 0; p < StepCount; p++) {
			const double us = static_cast<double>(a.steps[p]) * scale;
			if (us < 0.0005) {
				continue;
			}
			std::snprintf(text, sizeof(text), "%s%s %.3f", first ? "" : " ", kStepNames[p], us);
			line += text;
			first = false;
		}
		line += "] gapsteps[";
		first = true;
		for (size_t p = 0; p < StepCount; p++) {
			const double us = static_cast<double>(a.gap_steps[p]) * scale;
			if (us < 0.0005) {
				continue;
			}
			std::snprintf(text, sizeof(text), "%s%s %.3f", first ? "" : " ", kStepNames[p], us);
			line += text;
			first = false;
		}
		line += "]";
	}
	std::printf("%s\n", line.c_str());
}

void PrintStartPhases(State& s, double seconds, double ns_per_tick) {
	const double frames = static_cast<double>(std::max<uint64_t>(1, s.frames));
	std::string  line;
	char         text[256];
	std::snprintf(text, sizeof(text), "StartPhases %.0fs: frames %" PRIu64, seconds, s.frames);
	line += text;
	for (uint32_t r = 0; r < ReasonCount; r++) {
		const auto& a = s.reasons[r];
		if (a.draws == 0) {
			continue;
		}
		uint64_t total = 0;
		for (const auto v: a.cycles) {
			total += v;
		}
		const double scale = ns_per_tick * 1e-3 / static_cast<double>(a.draws);
		std::snprintf(text, sizeof(text), " | %s %.1f/f us/draw %.2f gap %.2f [", kReasonNames[r],
		              static_cast<double>(a.draws) / frames, static_cast<double>(total) * scale,
		              static_cast<double>(a.gap) * scale);
		line += text;
		for (size_t p = 0; p < PhaseCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", kPhaseNames[p],
			              static_cast<double>(a.cycles[p]) * scale);
			line += text;
		}
		line += "] sub[";
		for (size_t p = 0; p < SubCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", kSubNames[p],
			              static_cast<double>(a.sub[p]) * scale);
			line += text;
		}
		line += "] items[";
		for (size_t p = 0; p < ItemCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", kItemNames[p],
			              static_cast<double>(a.items[p]) / static_cast<double>(a.draws));
			line += text;
		}
		line += "]";
		if (a.gap_draws != 0) {
			// The gap's categories in us per draw of those with a known split. The previous
			// commit's own time ("commit", accounted when its scope closes after the end mark) is
			// not part of the gap and left out.
			const double gscale = ns_per_tick * 1e-3 / static_cast<double>(a.gap_draws);
			std::array<std::pair<uint64_t, size_t>, CpGaps::CatCount> order {};
			uint64_t                                                    scoped = 0;
			for (size_t i = 0; i < CpGaps::CatCount; i++) {
				const bool commit = static_cast<CpGaps::Cat>(i) == CpGaps::Cat::Commit;
				order[i]          = {commit ? 0 : a.gap_cats[i], i};
				scoped += commit ? 0 : a.gap_cats[i];
			}
			std::sort(order.begin(), order.end(),
			          [](const auto& x, const auto& y) { return x.first > y.first; });
			std::snprintf(text, sizeof(text), " gapsplit(%.0f%%)[",
			              100.0 * static_cast<double>(a.gap_draws) / static_cast<double>(a.draws));
			line += text;
			bool first = true;
			for (const auto& [ticks, index]: order) {
				const double us = static_cast<double>(ticks) * gscale;
				if (us < 0.01) {
					break;
				}
				std::snprintf(text, sizeof(text), "%s%s %.2f", first ? "" : " ",
				              CpGaps::CatName(static_cast<CpGaps::Cat>(index)), us);
				line += text;
				first = false;
			}
			std::snprintf(text, sizeof(text), "%sunscoped %.2f]", first ? "" : " ",
			              static_cast<double>(a.gap_known - std::min(a.gap_known, scoped)) * gscale);
			line += text;
		}
	}
	std::printf("%s\n", line.c_str());
	if (StepsEnabled()) {
		PrintSteps(s, seconds, ns_per_tick);
	}
	s.reasons = {};
}

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
	const auto* const phase_names = kPhaseNames;
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
		std::snprintf(text, sizeof(text),
		              " | %s %" PRIu64 " (%.1f%%) us/draw %.2f gap %.2f (headself %.2f in %.0f%% headwait "
		              "%.2f in %.0f%%) [",
		              class_names[k], c.draws,
		              100.0 * static_cast<double>(c.draws) / static_cast<double>(draws),
		              static_cast<double>(total) * scale, static_cast<double>(c.gap) * scale,
		              static_cast<double>(c.head_self) * scale,
		              100.0 * static_cast<double>(c.self_draws) / static_cast<double>(c.draws),
		              static_cast<double>(c.head_wait) * scale,
		              100.0 * static_cast<double>(c.wait_draws) / static_cast<double>(c.draws));
		line += text;
		for (size_t p = 0; p < PhaseCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", phase_names[p],
			              static_cast<double>(c.cycles[p]) * scale);
			line += text;
		}
		line += "] sub[";
		const auto* const sub_names = kSubNames;
		for (size_t p = 0; p < SubCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", sub_names[p],
			              static_cast<double>(c.sub[p]) * scale);
			line += text;
		}
		const auto* const item_names = kItemNames;
		line += "] items[";
		for (size_t p = 0; p < ItemCount; p++) {
			std::snprintf(text, sizeof(text), "%s%s %.2f", p == 0 ? "" : " ", item_names[p],
			              static_cast<double>(c.items[p]) / static_cast<double>(c.draws));
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
	{
		const double ticks_us = ns_per_tick * 1e-3;
		std::snprintf(text, sizeof(text),
		              " | writes starts run-continued %.1f%% | actual run: cont %" PRIu64
		              " (%.2f us/draw, after a writing draw %" PRIu64 ") start %" PRIu64
		              " (%.2f us/draw)",
		              pct(s.why_writes_run, s.why_writes), s.actual_cont,
		              static_cast<double>(s.actual_cont_cyc) * ticks_us /
		                  static_cast<double>(std::max<uint64_t>(1, s.actual_cont)),
		              s.writes_prev_cont, s.actual_start,
		              static_cast<double>(s.actual_start_cyc) * ticks_us /
		                  static_cast<double>(std::max<uint64_t>(1, s.actual_start)));
		line += text;
	}
	std::snprintf(text, sizeof(text),
	              " | start repeats: instance %.1f%% frame %.1f%% xframe struct f-1|2 %.1f%%",
	              pct(s.start_inst_repeat, starts), pct(s.start_frame_repeat, starts),
	              pct(s.start_xf_struct12, starts));
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
	PrintStartPhases(s, seconds, ns_per_tick);
	std::fflush(stdout);
	s.classes      = {};
	s.runs         = {};
	s.run_draws    = {};
	s.same_buffers = s.same_push = s.same_tables = s.same_index = s.same_vertex = s.same_all = 0;
	s.why_first = s.why_instance = s.why_pipeline = s.why_structure = s.why_dynamic =
	    s.why_writes                                                 = 0;
	s.why_writes_run = s.actual_cont = s.actual_cont_cyc = s.actual_start = s.actual_start_cyc =
	    s.writes_prev_cont                                               = 0;
	s.start_inst_repeat = s.start_frame_repeat = s.start_xf_struct12 = 0;
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
	s.head_self    = s.pending_self;
	s.head_wait    = s.pending_wait;
	s.pending_self = 0;
	s.pending_wait = 0;
	s.cycles       = {};
	s.sub          = {};
	s.items        = {};
	if (StepsEnabled()) {
		if (s.step_last != 0) {
			s.pending_gap_steps[static_cast<size_t>(Step::GapCommitIn)] += now - s.step_last;
		}
		s.gap_steps         = s.pending_gap_steps;
		s.pending_gap_steps = {};
		s.steps             = {};
		s.step_last         = now;
	}
	s.gap_delta_valid = false;
	if (s.gap_snapshot_valid && s.previous_end != 0) {
		std::array<uint64_t, CpGaps::CatCount> ticks {};
		if (ReadLedger(ticks)) {
			bool valid = true;
			for (size_t i = 0; i < CpGaps::CatCount; i++) {
				// A report in between reset the totals: no split for this gap.
				valid = valid && ticks[i] >= s.gap_snapshot[i];
				s.gap_delta[i] = ticks[i] - s.gap_snapshot[i];
			}
			s.gap_delta_valid = valid;
		}
	}
	s.gap_snapshot_valid = false;
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
		s.last      = Tsc();
		s.step_last = s.last;
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
	if (StepsEnabled()) {
		s.steps[static_cast<size_t>(Step::CommitTail)] += now - s.step_last;
	}
	s.active         = false;
	s.previous_end   = now;
	DrawClass klass  = ClassOther;
	Reason    reason = ReasonCount;
	if (s.recorded) {
		const auto& a = s.previous;
		const auto& b = s.shape;
		if (!s.previous_valid) {
			s.why_first++;
			klass  = ClassStart;
			reason = ReasonFirst;
		} else if (a.rendering_serial == 0 || a.rendering_serial != b.rendering_serial ||
		           a.command != b.command) {
			s.why_instance++;
			klass  = ClassStart;
			reason = ReasonInstance;
		} else if (a.pipeline != b.pipeline) {
			s.why_pipeline++;
			klass  = ClassStart;
			reason = ReasonPipeline;
		} else if (a.structure != b.structure) {
			s.why_structure++;
			klass  = ClassStart;
			reason = ReasonStructure;
		} else if (a.dynamic != b.dynamic) {
			s.why_dynamic++;
			klass  = ClassStart;
			reason = ReasonDynamic;
		} else if (a.shader_writes) {
			s.why_writes++;
			s.why_writes_run += b.run_continued ? 1u : 0u;
			klass  = ClassStart;
			reason = ReasonWrites;
		} else {
			klass  = ClassContinue;
			reason = ReasonRunMiss;
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
		const uint64_t shape_key = b.structure ^ (b.pipeline * 0x9e3779b97f4a7c15ull);
		if (b.rendering_serial != s.instance_serial) {
			s.instance_shapes.clear();
			s.instance_serial = b.rendering_serial;
		}
		if (klass == ClassStart) {
			s.start_inst_repeat += s.instance_shapes.contains(shape_key) ? 1u : 0u;
			s.start_frame_repeat += s.frame_shapes.contains(shape_key) ? 1u : 0u;
			s.start_xf_struct12 +=
			    s.struct_1.contains(b.struct_input) || s.struct_2.contains(b.struct_input) ? 1u : 0u;
		}
		s.instance_shapes.insert(shape_key);
		s.frame_shapes.insert(shape_key);
		s.run_length++;
		{
			uint64_t cycles = 0;
			for (size_t p = 0; p < PhaseCount; p++) {
				cycles += s.cycles[p];
			}
			if (b.run_continued) {
				s.actual_cont++;
				s.actual_cont_cyc += cycles;
				s.writes_prev_cont += s.previous_valid && a.shader_writes ? 1u : 0u;
			} else {
				s.actual_start++;
				s.actual_start_cyc += cycles;
			}
		}
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
	if (reason != ReasonCount) {
		if (s.shape.run_continued) {
			reason = ReasonCont;
		}
		auto& r = s.reasons[reason];
		r.draws++;
		r.gap += s.gap;
		if (s.gap_delta_valid) {
			r.gap_draws++;
			r.gap_known += s.gap;
			for (size_t i = 0; i < CpGaps::CatCount; i++) {
				r.gap_cats[i] += s.gap_delta[i];
			}
		}
		for (size_t p = 0; p < PhaseCount; p++) {
			r.cycles[p] += s.cycles[p];
		}
		for (size_t p = 0; p < SubCount; p++) {
			r.sub[p] += s.sub[p];
		}
		for (size_t p = 0; p < ItemCount; p++) {
			r.items[p] += s.items[p];
		}
		for (size_t p = 0; p < StepCount; p++) {
			r.steps[p] += s.steps[p];
			r.gap_steps[p] += s.gap_steps[p];
		}
	}
	s.gap_snapshot_valid = ReadLedger(s.gap_snapshot);
	auto& c = s.classes[klass];
	c.draws++;
	c.gap += s.gap;
	c.head_self += s.head_self;
	c.head_wait += s.head_wait;
	c.self_draws += s.head_self != 0 ? 1u : 0u;
	c.wait_draws += s.head_wait != 0 ? 1u : 0u;
	for (size_t p = 0; p < PhaseCount; p++) {
		c.cycles[p] += s.cycles[p];
	}
	for (size_t p = 0; p < SubCount; p++) {
		c.sub[p] += s.sub[p];
	}
	for (size_t p = 0; p < ItemCount; p++) {
		c.items[p] += s.items[p];
	}
	Print(s, NowNs(), now);
	// The next gap starts after this diagnostic's own work (the classification and sets above).
	s.previous_end = Tsc();
	s.step_last    = s.previous_end;
}

void Detail::StepAt(Step step) {
	auto&      s   = GetState();
	const auto now = Tsc();
	if (s.step_last != 0) {
		const auto index = static_cast<size_t>(step);
		if (s.active && step >= Step::BindRegs) {
			s.steps[index] += now - s.step_last;
		} else {
			s.pending_gap_steps[index] += now - s.step_last;
		}
	}
	s.step_last = now;
}

void AddSub(Sub sub, uint64_t cycles) {
	auto& s = GetState();
	if (s.active) {
		s.sub[static_cast<size_t>(sub)] += cycles;
	}
}

void AddItems(Item item, uint64_t count) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (s.active) {
		s.items[static_cast<size_t>(item)] += count;
	}
}

uint64_t Now() {
	return Enabled() ? Tsc() : 0;
}

void NoteHead(uint64_t self_cycles, uint64_t wait_cycles) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	s.pending_self += self_cycles;
	s.pending_wait += wait_cycles;
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
	s.frame_shapes.clear();
	// A frame boundary is never inside a run of one rendering instance's draws.
	EndRun(s);
	s.previous_valid = false;
}

} // namespace Libs::Graphics::CommitStats
