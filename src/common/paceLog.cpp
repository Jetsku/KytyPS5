#include "common/paceLog.h"
#include "common/staticOnce.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace PaceLog {

namespace {

struct Entry {
	uint64_t id          = 0;
	uint64_t reserve_ns  = 0;
	uint64_t ready_ns    = 0;
	uint64_t flipsub_ns  = 0;
	uint64_t framesub_ns = 0;
	uint64_t cpstart_ns  = 0;
	uint64_t done_ns     = 0;
};

FILE*                  g_file = nullptr;
FILE*                  g_waits = nullptr;
std::mutex             g_mutex;
std::array<Entry, 64>  g_entries {};
uint64_t               g_origin_ns = 0;
// CP-thread state of the frame being processed.
uint64_t g_cur_sub_ns       = 0;
uint64_t g_cur_framesub_ns  = 0;
uint64_t g_cur_cpstart_ns   = 0;
uint64_t g_last_done_ns     = 0;
uint64_t g_done_wait_ns     = 0;
uint64_t g_gpu_busy = 0, g_gpu_idle = 0, g_gpu_starved = 0, g_gpu_cmdbufs = 0, g_gpu_last_end = 0;

uint64_t Rel(uint64_t ns) {
	return ns > g_origin_ns ? (ns - g_origin_ns) / 1000u : 0;
}

} // namespace

bool Enabled() {
	static const bool enabled = [] {
		const auto* path = std::getenv("KYTY_PACE_LOG");
		if (path == nullptr || path[0] == 0) {
			return false;
		}
		g_file = std::fopen(path, "w");
		if (g_file == nullptr) {
			return false;
		}
		std::fprintf(g_file, "present_us,vblank,reserve_us,ready_us,flipsub_us,framesub_us,cpstart_us,"
		                     "done_us,donewait_us,present_cost_us,gpu_busy_us,gpu_idle_us,"
		                     "gpu_starved_us,gpu_cmdbufs,gpu_last_end_us\n");
		g_waits = std::fopen((std::string(path) + ".waits.csv").c_str(), "w");
		if (g_waits != nullptr) {
			std::fprintf(g_waits, "begin_us,end_us,tid,filter,ident,events,name\n");
		}
		g_origin_ns = NowNs();
		return true;
	}();
	return enabled;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void NoteDone(uint64_t call_ns, uint64_t wait_ns) {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	g_last_done_ns = call_ns;
	g_done_wait_ns += wait_ns;
}

void NoteCpSubmission(uint64_t enqueue_ns, bool frame_start) {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	g_cur_sub_ns = enqueue_ns;
	if (frame_start) {
		g_cur_framesub_ns = enqueue_ns;
		g_cur_cpstart_ns  = NowNs();
	}
}

void NoteReserve(uint64_t request_id) {
	if (!Enabled()) {
		return;
	}
	const auto      now = NowNs();
	std::lock_guard lock(g_mutex);
	auto&           e = g_entries[request_id % g_entries.size()];
	e                 = {};
	e.id              = request_id;
	e.reserve_ns      = now;
	e.flipsub_ns      = g_cur_sub_ns;
	e.framesub_ns     = g_cur_framesub_ns;
	e.cpstart_ns      = g_cur_cpstart_ns;
	e.done_ns         = g_last_done_ns;
}

void NoteReady(uint64_t request_id) {
	if (!Enabled()) {
		return;
	}
	const auto      now = NowNs();
	std::lock_guard lock(g_mutex);
	auto&           e = g_entries[request_id % g_entries.size()];
	if (e.id == request_id) {
		e.ready_ns = now;
	}
}

void NoteGpuFrame(uint64_t busy_ns, uint64_t idle_ns, uint64_t starved_ns, uint64_t cmdbufs,
                  uint64_t last_end_ns) {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	g_gpu_busy     = busy_ns;
	g_gpu_idle     = idle_ns;
	g_gpu_starved  = starved_ns;
	g_gpu_cmdbufs  = cmdbufs;
	g_gpu_last_end = last_end_ns;
}

void NoteWait(uint64_t begin_ns, uint64_t end_ns, int tid, int filter, uint64_t ident, int events,
              const char* name) {
	if (!Enabled() || g_waits == nullptr || end_ns < begin_ns + 100'000u) {
		return;
	}
	std::lock_guard lock(g_mutex);
	std::fprintf(g_waits, "%" PRIu64 ",%" PRIu64 ",%d,%d,0x%" PRIx64 ",%d,%s\n", Rel(begin_ns),
	             Rel(end_ns), tid, filter, ident, events, name != nullptr ? name : "");
	static uint32_t lines = 0;
	if ((++lines & 255u) == 0) {
		std::fflush(g_waits);
	}
}

bool StallTraceEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_STALL_TRACE");
		return value != nullptr && value[0] == '1' && Enabled() && g_waits != nullptr;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

void NoteStall(uint64_t begin_ns, uint64_t end_ns, StallCode code, uint64_t hash, int kind, const char* name) {
	static const uint64_t min_ns = [] {
		const auto* value = std::getenv("KYTY_STALL_TRACE_MIN_MS");
		return (value != nullptr && value[0] != 0 ? std::strtoull(value, nullptr, 10) : 20ull) * 1'000'000ull;
	}();
	if (!StallTraceEnabled() || end_ns < begin_ns || end_ns - begin_ns < min_ns) {
		return;
	}
#ifdef _WIN32
	const int tid = static_cast<int>(GetCurrentThreadId());
#else
	const int tid = 0;
#endif
	std::lock_guard lock(g_mutex);
	std::fprintf(g_waits, "%" PRIu64 ",%" PRIu64 ",%d,%d,0x%" PRIx64 ",%d,%s\n", Rel(begin_ns), Rel(end_ns), tid,
	             static_cast<int>(code), hash, kind, name != nullptr ? name : "");
	std::fflush(g_waits);
}

void NotePresent(uint64_t request_id, uint64_t vblank, uint64_t tick_ns, uint64_t present_ns) {
	if (!Enabled()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	const auto&     e     = g_entries[request_id % g_entries.size()];
	const bool      match = e.id == request_id;
	std::fprintf(g_file,
	             "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
	             ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64
	             ",%" PRIu64 "\n",
	             Rel(tick_ns), vblank, match ? Rel(e.reserve_ns) : 0, match ? Rel(e.ready_ns) : 0,
	             match ? Rel(e.flipsub_ns) : 0, match ? Rel(e.framesub_ns) : 0,
	             match ? Rel(e.cpstart_ns) : 0, match ? Rel(e.done_ns) : 0, g_done_wait_ns / 1000u,
	             present_ns / 1000u, g_gpu_busy / 1000u, g_gpu_idle / 1000u, g_gpu_starved / 1000u,
	             g_gpu_cmdbufs, Rel(g_gpu_last_end));
	g_done_wait_ns = 0;
	static uint32_t lines = 0;
	if ((++lines & 63u) == 0) {
		std::fflush(g_file);
	}
}

} // namespace PaceLog
