#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BOOKKEEP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BOOKKEEP_H_

// Serial reduction of the commit thread's per-draw bookkeeping (stage 1 of the whole-thread plan).
// Each part has its own live switch (0 off, 1 on, 2 verify) and counters; the 10 s line is printed
// with the DrawPrep summary while any part is on.
//
// KYTY_BK_HANDOVER (drawPrep.cpp, Engine::ReadyHead): the head's ready check once per run of
//   prepared slots instead of once per draw. When the head is ready, the following positions a
//   worker has finished are noted with one acquire load each; their heads then skip the claim
//   attempt, the done check, the wait path and the lead statistics. Retire stays per draw (its
//   store is the sequencer's wake handshake while KYTY_HANDOFF_LEAN is off). Verify: such a head's
//   state word is checked to be Done for its position (what the per-draw protocol would have seen),
//   and every mismatch is counted. Also with the switch: the four 10 s summary printers run their
//   period checks every 256th commit instead of after every draw.
// KYTY_BK_ADOPT (renderDraw.cpp, ApplyPreparedDraw): a kernel continuation's stage preparations are
//   adopted by pointer (the draw state's stage runtimes point at the slot's preparations) instead of
//   being swapped member by member into the draw state. Verify: adopted, plus a value copy taken at
//   adoption that is compared with the slot's preparations after the draw.
// KYTY_BK_RECORD (renderDraw.cpp, DrawRunRecordDraw): a kernel continuation's record update writes
//   only the descriptor epoch; the other fields it rewrote are equal by the continuation's own
//   checks. Verify: the fields are compared with what the full update writes, which is then used.
// KYTY_BK_LEAN (renderDraw.cpp, ExecutePreparedDraw / ContinuationKernel): a continuation's rendering
//   state is the run record's, used by reference instead of a default-constructed copy assigned from
//   it; the kernel's taken counter is a plain add of its only writer (no locked RMW per draw); the
//   mesh-segment scratch is one function-local vector instead of a thread_local (no TLS lookup per
//   draw; one thread draws). Verify: the copy is made as well and compared with the record after the
//   draw (the reference sees any change the copy would not).
//
// Everything in this header is free of the renderer runtime (tested in tests/BookkeepTests.cpp).

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace Libs::Graphics::DrawPrep::Bookkeep {

enum class Mode : uint8_t { Off = 0, On = 1, Verify = 2 };

[[nodiscard]] inline Mode ParseMode(const char* value) noexcept {
	if (value == nullptr || value[0] == '\0') {
		return Mode::Off;
	}
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
		return Mode::On;
	}
	if (std::strcmp(value, "2") == 0 || std::strcmp(value, "verify") == 0) {
		return Mode::Verify;
	}
	return Mode::Off;
}

// KYTY_BK_HANDOVER: the positions [head, end) the commit thread has seen finished by a worker. A
// position's slot cannot change state after Done until the commit thread retires it, so a head
// inside the run is ready without another look at its state word.
class ReadyRun {
public:
	[[nodiscard]] bool Covers(uint64_t head) const noexcept { return head < m_end && head >= m_begin; }

	// The head is ready (prepared by a worker or by this thread): note it and the positions after it
	// that `done_at(position)` reports finished, up to `tail` (published) and at most `cap` beyond the
	// head. Returns how many positions beyond the head were noted.
	template <typename DoneAt>
	uint32_t Extend(uint64_t head, uint64_t tail, uint32_t cap, DoneAt&& done_at) {
		m_begin = head;
		m_end   = head + 1u;
		while (m_end < tail && m_end - head <= cap && done_at(m_end)) {
			m_end++;
		}
		return static_cast<uint32_t>(m_end - head - 1u);
	}

	void Reset() noexcept { m_begin = m_end = 0; }
	[[nodiscard]] uint64_t End() const noexcept { return m_end; }

private:
	uint64_t m_begin = 0;
	uint64_t m_end   = 0;
};

// Why a part did not apply to a draw it could have covered.
enum class Fallback : uint8_t {
	AdoptVerifyPaths, // KYTY_BK_ADOPT: KYTY_DRAW_PREP_VERIFY or KYTY_RENDER_STATE_VERIFY reads the swap
	RecordNotKernel,  // KYTY_BK_RECORD: a continuation outside the kernel (pipeline swap, texture delta)
	Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> kFallbackNames {
    "adopt_verify_paths", "record_not_kernel"};

// Process-wide totals; the commit thread writes them (plain relaxed stores of its own counts are
// enough: one writer). Relaxed loads by the printer.
struct Counters {
	// KYTY_BK_HANDOVER
	std::atomic<uint64_t> heads {0};          // heads committed while the switch was on
	std::atomic<uint64_t> run_heads {0};      // heads taken from a ready run (no per-draw check)
	std::atomic<uint64_t> runs {0};           // ready runs noted with at least one position beyond the head
	std::atomic<uint64_t> run_positions {0};  // positions noted beyond the heads
	std::atomic<uint64_t> run_kernel {0};     // run heads whose draw the continuation kernel took
	std::atomic<uint64_t> handover_checks {0};
	std::atomic<uint64_t> handover_mismatches {0};
	// KYTY_BK_ADOPT
	std::atomic<uint64_t> adopted {0};
	std::atomic<uint64_t> adopt_checks {0};
	std::atomic<uint64_t> adopt_mismatches {0};
	// KYTY_BK_RECORD
	std::atomic<uint64_t> record_lean {0};
	std::atomic<uint64_t> record_checks {0};
	std::atomic<uint64_t> record_mismatches {0};
	// KYTY_BK_LEAN
	std::atomic<uint64_t> lean_rendering {0}; // continuations using the record's rendering state
	std::atomic<uint64_t> lean_checks {0};
	std::atomic<uint64_t> lean_mismatches {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Fallback::Count)> fallbacks {};

	void Add(std::atomic<uint64_t>& counter, uint64_t n = 1) noexcept {
		counter.store(counter.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
	}
	void Fall(Fallback why) noexcept { Add(fallbacks[static_cast<size_t>(why)]); }
};

// A field-wise mismatch report: prints the first `limit` occurrences of a verify mismatch.
inline void ReportMismatch(std::atomic<uint64_t>& mismatches, const char* part, const char* what,
                           uint64_t limit = 16) {
	const auto n = mismatches.load(std::memory_order_relaxed) + 1u;
	mismatches.store(n, std::memory_order_relaxed);
	if (n <= limit) {
		std::printf("Bookkeep verify: %s mismatch (%s), %" PRIu64 " so far\n", part, what, n);
		std::fflush(stdout);
	}
}

[[nodiscard]] inline const char* ModeName(Mode mode) noexcept {
	return mode == Mode::On ? "on" : mode == Mode::Verify ? "verify" : "off";
}

// The 10 s line (totals since the start).
[[nodiscard]] inline std::string FormatLine(const Counters& c, Mode handover, Mode adopt, Mode record, Mode lean) {
	const auto get   = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
	const auto heads = get(c.heads);
	const auto runs  = get(c.runs);
	char       text[1024];
	std::snprintf(text, sizeof(text),
	              "Bookkeep: handover %s heads %" PRIu64 " from runs %" PRIu64 " (%.1f%%, kernel %" PRIu64
	              ") runs %" PRIu64 " (%.2f positions/run) verify %" PRIu64 "/%" PRIu64
	              " | adopt %s %" PRIu64 " verify %" PRIu64 "/%" PRIu64 " | record %s %" PRIu64 " verify %" PRIu64
	              "/%" PRIu64 " | lean %s rendering %" PRIu64 " verify %" PRIu64 "/%" PRIu64
	              " | fallbacks",
	              ModeName(handover), heads, get(c.run_heads),
	              heads != 0 ? 100.0 * static_cast<double>(get(c.run_heads)) / static_cast<double>(heads) : 0.0,
	              get(c.run_kernel), runs,
	              runs != 0 ? static_cast<double>(get(c.run_positions)) / static_cast<double>(runs) : 0.0,
	              get(c.handover_mismatches), get(c.handover_checks), ModeName(adopt), get(c.adopted),
	              get(c.adopt_mismatches), get(c.adopt_checks), ModeName(record), get(c.record_lean),
	              get(c.record_mismatches), get(c.record_checks), ModeName(lean), get(c.lean_rendering),
	              get(c.lean_mismatches), get(c.lean_checks));
	std::string line = text;
	for (size_t i = 0; i < kFallbackNames.size(); i++) {
		line += " ";
		line += kFallbackNames[i];
		line += "=" + std::to_string(get(c.fallbacks[i]));
	}
	line += " | mismatches total=" + std::to_string(get(c.handover_mismatches) + get(c.adopt_mismatches) +
	                                                get(c.record_mismatches) + get(c.lean_mismatches));
	return line;
}

} // namespace Libs::Graphics::DrawPrep::Bookkeep

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BOOKKEEP_H_
