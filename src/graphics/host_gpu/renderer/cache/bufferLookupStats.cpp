#include "graphics/host_gpu/renderer/cache/bufferLookupStats.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace Libs::Graphics::BufferLookupStats {

namespace Detail {
bool ReadEnabled() {
	const auto* value = std::getenv("KYTY_BUFFER_LOOKUP_STATS");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}
} // namespace Detail

bool KeysEnabled() {
	static const bool keys = [] {
		const auto* value = std::getenv("KYTY_BUFFER_LOOKUP_STATS");
		return value != nullptr && std::strcmp(value, "keys") == 0;
	}();
	return keys;
}

namespace {

constexpr size_t OutcomeCount = static_cast<size_t>(Outcome::Count);
constexpr size_t PartCount    = static_cast<size_t>(Part::Count);
constexpr size_t MissCount    = static_cast<size_t>(Miss::Count);

uint64_t ReadTsc() {
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

struct State {
	Outcome                            outcome = Outcome::None;
	std::array<uint64_t, OutcomeCount> binding_count {};
	std::array<uint64_t, OutcomeCount> binding_cycles {};
	std::array<uint64_t, PartCount>    part_count {};
	std::array<uint64_t, PartCount>    part_cycles {};
	std::array<uint64_t, MissCount>    misses {};
	uint64_t                           stages         = 0;
	uint64_t                           epoch_advances = 0;
	uint64_t                           last_epoch     = 0;
	uint64_t                           interval_ns    = 0;
	uint64_t                           interval_tsc   = 0;
	// Keys (NoteKey).
	std::unordered_set<uint64_t> keys_frame;
	std::unordered_set<uint64_t> keys_previous;
	uint32_t                     key_frame     = 0;
	uint64_t                     key_frames    = 0;
	uint64_t                     keys_distinct = 0;
	uint64_t                     keys_repeated = 0; // distinct keys the previous frame had
	uint64_t                     key_lookups   = 0;
};

State& GetState() {
	// GPU thread only.
	static State state;
	return state;
}

void Print(State& s) {
	const auto now_ns  = NowNs();
	const auto now_tsc = ReadTsc();
	if (s.interval_ns == 0) {
		s.interval_ns  = now_ns;
		s.interval_tsc = now_tsc;
		return;
	}
	if (now_ns - s.interval_ns < 10'000'000'000ull) {
		return;
	}
	const double ns_per_tick = static_cast<double>(now_ns - s.interval_ns) /
	                           static_cast<double>(now_tsc - s.interval_tsc + 1);
	const double seconds     = static_cast<double>(now_ns - s.interval_ns) * 1e-9;
	static const char* const outcome_names[OutcomeCount] = {
	    "none", "hit", "cross", "stream-hit", "miss", "miss-stream", "miss-lean",
	    "texel", "written", "narrow", "null"};
	static const char* const part_names[PartCount] = {
	    "findbuf",  "bda",    "prefetch",  "wranges",   "invalidate", "mipstats",
	    "tables",   "descr",  "read-now",  "read-rec",  "read-touch", "find-call",
	    "bda-skip", "bda-none", "bda-hot", "bda-log", "bda-full", "bda-full-new", "create"};
	static const char* const miss_names[MissCount] = {"no-region", "empty", "other-key",
	                                                  "signature", "guard", "epoch"};
	std::string line;
	char        text[256];
	uint64_t    bindings = 0;
	double      binding_ms = 0;
	for (size_t k = 0; k < OutcomeCount; k++) {
		bindings += s.binding_count[k];
		binding_ms += static_cast<double>(s.binding_cycles[k]) * ns_per_tick * 1e-6;
	}
	std::snprintf(text, sizeof(text),
	              "BufferLookup %.0fs: stages %.0f/s bindings %.0f/s (%.1f ms/s) |", seconds,
	              static_cast<double>(s.stages) / seconds, static_cast<double>(bindings) / seconds,
	              binding_ms / seconds);
	line += text;
	for (size_t k = 0; k < OutcomeCount; k++) {
		if (s.binding_count[k] == 0) {
			continue;
		}
		const double ns = static_cast<double>(s.binding_cycles[k]) * ns_per_tick;
		std::snprintf(text, sizeof(text), " %s %.0f/s %.0fns (%.2f ms/s)", outcome_names[k],
		              static_cast<double>(s.binding_count[k]) / seconds,
		              ns / static_cast<double>(s.binding_count[k]), ns * 1e-6 / seconds);
		line += text;
	}
	line += " | parts";
	for (size_t p = 0; p < PartCount; p++) {
		if (s.part_count[p] == 0) {
			continue;
		}
		const double ns = static_cast<double>(s.part_cycles[p]) * ns_per_tick;
		std::snprintf(text, sizeof(text), " %s %.0f/s %.0fns (%.2f ms/s)", part_names[p],
		              static_cast<double>(s.part_count[p]) / seconds,
		              ns / static_cast<double>(s.part_count[p]), ns * 1e-6 / seconds);
		line += text;
	}
	line += " | misses";
	for (size_t m = 0; m < MissCount; m++) {
		std::snprintf(text, sizeof(text), " %s %.0f/s", miss_names[m],
		              static_cast<double>(s.misses[m]) / seconds);
		line += text;
	}
	std::snprintf(text, sizeof(text), " | epoch %.0f/s (per stage %.2f)",
	              static_cast<double>(s.epoch_advances) / seconds,
	              static_cast<double>(s.epoch_advances) / static_cast<double>(s.stages + 1));
	line += text;
	if (s.key_frames != 0) {
		std::snprintf(text, sizeof(text),
		              " | keys/frame %.0f (lookups %.0f), in previous frame %.1f%%",
		              static_cast<double>(s.keys_distinct) / static_cast<double>(s.key_frames),
		              static_cast<double>(s.key_lookups) / static_cast<double>(s.key_frames),
		              100.0 * static_cast<double>(s.keys_repeated) /
		                  static_cast<double>(s.keys_distinct + 1));
		line += text;
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
	s.binding_count  = {};
	s.binding_cycles = {};
	s.part_count     = {};
	s.part_cycles    = {};
	s.misses         = {};
	s.stages = s.epoch_advances = 0;
	s.key_frames = s.keys_distinct = s.keys_repeated = s.key_lookups = 0;
	s.interval_ns  = now_ns;
	s.interval_tsc = now_tsc;
}

} // namespace

namespace Detail {

uint64_t Tsc() {
	return ReadTsc();
}

void SetOutcome(Outcome outcome) {
	if (Enabled()) {
		GetState().outcome = outcome;
	}
}

Outcome TakeOutcome() {
	if (!Enabled()) {
		return Outcome::None;
	}
	auto&      s       = GetState();
	const auto outcome = s.outcome;
	s.outcome          = Outcome::None;
	return outcome;
}

void AddBinding(Outcome outcome, uint64_t cycles) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	s.binding_count[static_cast<size_t>(outcome)]++;
	s.binding_cycles[static_cast<size_t>(outcome)] += cycles;
}

void AddPart(Part part, uint64_t cycles) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	s.part_count[static_cast<size_t>(part)]++;
	s.part_cycles[static_cast<size_t>(part)] += cycles;
}

void AddMiss(Miss miss) {
	if (Enabled()) {
		GetState().misses[static_cast<size_t>(miss)]++;
	}
}

void NoteKey(uint64_t vaddr, uint64_t size, uint32_t frame) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	if (frame != s.key_frame) {
		if (s.key_frame != 0) {
			s.key_frames++;
			s.keys_distinct += s.keys_frame.size();
			for (const auto key: s.keys_frame) {
				s.keys_repeated += s.keys_previous.contains(key) ? 1u : 0u;
			}
		}
		s.keys_previous.swap(s.keys_frame);
		s.keys_frame.clear();
		s.key_frame = frame;
	}
	s.key_lookups++;
	s.keys_frame.insert(vaddr * 0x9e3779b97f4a7c15ull ^ size);
}

void NoteStage(uint64_t epoch) {
	if (!Enabled()) {
		return;
	}
	auto& s = GetState();
	s.stages++;
	if (s.last_epoch != 0) {
		s.epoch_advances += epoch - s.last_epoch;
	}
	s.last_epoch = epoch;
	Print(s);
}

} // namespace Detail

} // namespace Libs::Graphics::BufferLookupStats
