#include "common/prepPhases.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

namespace Common::PrepPhases {

namespace {

constexpr size_t PhaseCount = static_cast<size_t>(Phase::Count);
constexpr size_t SubCount   = static_cast<size_t>(Sub::Count);

struct Shared {
	std::atomic<uint64_t>                          slots {0};
	std::atomic<uint64_t>                          total {0};
	std::atomic<uint64_t>                          reads {0};
	std::atomic<uint64_t>                          bytes {0};
	std::array<std::atomic<uint64_t>, PhaseCount>  phase {};
	std::array<std::atomic<uint64_t>, SubCount>    sub {};
	std::atomic<uint64_t>                          frames {0};
	std::atomic<uint64_t>                          interval_ns {0};
	std::atomic<uint64_t>                          interval_tsc {0};
	std::atomic<bool>                              printing {false};
};

Shared& GetShared() {
	static Shared shared;
	return shared;
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void MaybePrint(Shared& g) {
	const auto now_ns  = NowNs();
	const auto now_tsc = Tsc();
	auto       begin   = g.interval_ns.load(std::memory_order_relaxed);
	if (begin == 0) {
		uint64_t expected = 0;
		if (g.interval_ns.compare_exchange_strong(expected, now_ns)) {
			g.interval_tsc.store(now_tsc, std::memory_order_relaxed);
		}
		return;
	}
	if (now_ns - begin < 10'000'000'000ull || g.printing.exchange(true)) {
		return;
	}
	begin                = g.interval_ns.load(std::memory_order_relaxed);
	const auto begin_tsc = g.interval_tsc.load(std::memory_order_relaxed);
	g.interval_ns.store(now_ns, std::memory_order_relaxed);
	g.interval_tsc.store(now_tsc, std::memory_order_relaxed);
	const double us_per_tick = static_cast<double>(now_ns - begin) * 1e-3 /
	                           static_cast<double>(std::max<uint64_t>(1, now_tsc - begin_tsc));
	const auto   slots  = g.slots.exchange(0);
	const auto   frames = g.frames.exchange(0);
	const double per    = us_per_tick / static_cast<double>(std::max<uint64_t>(1, slots));
	static const char* const phase_names[PhaseCount] = {
	    "reset", "vsprog", "psprog", "pslookup", "psmat", "psbind", "vslookup",
	    "vsmat", "vsbind", "finish", "cert", "runkey", "plan", "complete"};
	static const char* const sub_names[SubCount] = {"reads", "srt", "buffers", "images",
	                                                "samplers"};
	const double fcount = static_cast<double>(std::max<uint64_t>(1, frames));
	const double scount = static_cast<double>(std::max<uint64_t>(1, slots));
	const double total  = static_cast<double>(g.total.exchange(0)) * per;
	char         line[512];
	std::snprintf(line, sizeof(line),
	              "PrepPhases %.0fs: frames %" PRIu64 " slots %" PRIu64
	              " (%.0f/f) total %.2f us/slot (%.1f ms/f) reads %.1f/slot %.0f B/slot |",
	              static_cast<double>(now_ns - begin) * 1e-9, frames, slots,
	              static_cast<double>(slots) / fcount, total,
	              total * static_cast<double>(slots) * 1e-3 / fcount,
	              static_cast<double>(g.reads.exchange(0)) / scount,
	              static_cast<double>(g.bytes.exchange(0)) / scount);
	std::string text(line);
	for (size_t i = 0; i < PhaseCount; i++) {
		std::snprintf(line, sizeof(line), " %s %.2f", phase_names[i],
		              static_cast<double>(g.phase[i].exchange(0)) * per);
		text += line;
	}
	text += " | sub";
	for (size_t i = 0; i < SubCount; i++) {
		std::snprintf(line, sizeof(line), " %s %.2f", sub_names[i],
		              static_cast<double>(g.sub[i].exchange(0)) * per);
		text += line;
	}
	std::printf("%s\n", text.c_str());
	std::fflush(stdout);
	g.printing.store(false);
}

} // namespace

namespace Detail {
bool ReadEnabled() {
	const auto* value = std::getenv("KYTY_DRAW_PREP_PHASES");
	return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

void Flush(ThreadState& state) {
	auto& g = GetShared();
	g.slots.fetch_add(128, std::memory_order_relaxed);
	g.total.fetch_add(state.total, std::memory_order_relaxed);
	g.reads.fetch_add(state.reads, std::memory_order_relaxed);
	g.bytes.fetch_add(state.bytes, std::memory_order_relaxed);
	state.total = state.reads = state.bytes = 0;
	for (size_t i = 0; i < PhaseCount; i++) {
		g.phase[i].fetch_add(state.phase[i], std::memory_order_relaxed);
		state.phase[i] = 0;
	}
	for (size_t i = 0; i < SubCount; i++) {
		g.sub[i].fetch_add(state.sub[i], std::memory_order_relaxed);
		state.sub[i] = 0;
	}
	MaybePrint(g);
}
} // namespace Detail

void NoteFrame() {
	if (Enabled()) {
		GetShared().frames.fetch_add(1, std::memory_order_relaxed);
	}
}

} // namespace Common::PrepPhases
