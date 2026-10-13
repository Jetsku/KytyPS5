#ifndef KYTY_COMMON_PREPPHASES_H_
#define KYTY_COMMON_PREPPHASES_H_

#include "common/onceValue.h"

#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

// KYTY_DRAW_PREP_PHASES=1 (measurement only, default off): the split of a draw-prep slot's
// preparation (DrawPrep::Engine::Workers::PrepareClaimed, on a DrawPrep worker or the command
// processor stealing a slot) into consecutive phases, plus overlapping sub-buckets inside them
// (the recorded guest reads, the SRT refresh and descriptor evaluation of the materialization).
// One "PrepPhases 10s" console line per 10 s: us per slot for each phase and sub-bucket.
// Timing is per thread (TSC), flushed into shared counters every 128 slots.
namespace Common::PrepPhases {

enum class Phase : uint8_t {
	Reset,         // Prepare's output reset, eligibility, coherence generations, export mapping
	VsProgram,     // PrepareProgram of the vertex/mesh stage (registers, shader header, key inputs)
	PsProgram,     // PrepareProgram of the pixel stage, dual-source blending, clip space
	PsLookup,      // pixel stage: program key and source lookup
	PsMaterialize, // pixel stage: MaterializeResources (SRT walk, V#/T#/S# evaluation)
	PsBind,        // pixel stage: permutation lookup and Bind
	VsLookup,      // parameter linking, vertex key and source lookup
	VsMaterialize, // vertex stage: MaterializeResources
	VsBind,        // vertex stage: permutation lookup and Bind
	Finish,        // the read set's Finish (sort and coalesce)
	Certificate,   // the certificate ranges (KYTY_DRAW_PREP_CERT_RANGES)
	RunKey,        // DrawRun structure key, repeat-trace hashes
	Plan,          // binding plan (and pipeline prefetch)
	Complete,      // the hand-off to the commit (Window::Complete)
	Count
};

enum class Sub : uint8_t {
	Reads,      // recorded guest reads: the clean gate, the copy and the record (inside the phases)
	SrtRefresh, // MaterializeResources: active sources, flat SRT buffer, uniform fill
	Buffers,    // MaterializeResources: V# evaluation (and fused specialization)
	Images,     // MaterializeResources: T# evaluation, indirect images
	Samplers,   // MaterializeResources: S# evaluation, the specialization
	Count
};

namespace Detail {
[[nodiscard]] bool ReadEnabled();

struct ThreadState {
	bool     active = false;
	uint64_t start  = 0;
	uint64_t last   = 0;
	uint64_t slots  = 0;
	uint64_t total  = 0;
	uint64_t reads  = 0;
	uint64_t bytes  = 0;
	uint64_t phase[static_cast<size_t>(Phase::Count)] {};
	uint64_t sub[static_cast<size_t>(Sub::Count)] {};
};
inline thread_local ThreadState t_state;

void Flush(ThreadState& state);
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
	static std::atomic<int64_t> enabled_once {OnceUnset};
	return static_cast<bool>(Once(enabled_once, [] { return Detail::ReadEnabled(); }));
}

[[nodiscard]] inline uint64_t Tsc() {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return 0;
#endif
}

// Starts timing one slot on this thread.
inline void Begin() {
	if (!Enabled()) [[likely]] {
		return;
	}
	auto& s  = Detail::t_state;
	s.active = true;
	s.start  = Tsc();
	s.last   = s.start;
}

// The time since the previous mark (or Begin) belongs to `phase`.
inline void Mark(Phase phase) {
	auto& s = Detail::t_state;
	if (!s.active) [[likely]] {
		return;
	}
	const auto now = Tsc();
	s.phase[static_cast<size_t>(phase)] += now - s.last;
	s.last = now;
}

// Ends the slot: the time since the last mark belongs to `phase`.
inline void End(Phase phase) {
	auto& s = Detail::t_state;
	if (!s.active) [[likely]] {
		return;
	}
	Mark(phase);
	s.total += s.last - s.start;
	s.active = false;
	if ((++s.slots & 127u) == 0u) {
		Detail::Flush(s);
	}
}

// Overlapping sub-bucket timing: SubStart() is 0 outside a timed slot.
[[nodiscard]] inline uint64_t SubStart() {
	return Detail::t_state.active ? Tsc() : 0;
}
inline void SubAdd(Sub sub, uint64_t start) {
	if (start != 0) {
		Detail::t_state.sub[static_cast<size_t>(sub)] += Tsc() - start;
	}
}
inline void NoteRead(uint64_t size) {
	auto& s = Detail::t_state;
	if (s.active) {
		s.reads++;
		s.bytes += size;
	}
}

// A guest frame boundary (the command processor's frame reset).
void NoteFrame();

} // namespace Common::PrepPhases

#endif // KYTY_COMMON_PREPPHASES_H_
