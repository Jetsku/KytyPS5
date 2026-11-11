#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Console lines that test scripts can wait for instead of sleeping fixed times.
//
// Guest console (KYTY_GUEST_CONSOLE, on by default, =0 turns it off): text the game writes to
// fd 1/2 (POSIX write, sceKernelWrite, writev) is split into lines and printed as
//   "GUEST: frame=<n> t=<s> | <text>"
// (frame = guest flips so far, t = seconds since the emulator started), flushed per line.
//
// Emulator events (KYTY_EVENTS, on by default, =0 turns them off), printed as
//   "KYTY_EVENT: <name> frame=<n> t=<s> [key=value ...]"
//   first-flip            the first guest flip;
//   loading-start         guest file reads (read/pread/readv and APR read commands) reach
//                         KYTY_EVENTS_LOAD_ON_MBPS (default 48) over the last second;
//   loading-end           while loading: reads stay under KYTY_EVENTS_LOAD_OFF_MBPS (default 16)
//                         and flips at or above KYTY_EVENTS_MIN_FPS (default 2) for
//                         KYTY_EVENTS_HOLD_MS (default 5000) without a break (Ghost of Yotei
//                         plays at 4-5 fps after its first load, so the fps floor is low);
//   shader-compile-idle   after the first flip: at least KYTY_EVENTS_IDLE_MIN_BUILDS (default 50)
//                         shader/pipeline builds started since the last idle period and none has
//                         been in flight for KYTY_EVENTS_IDLE_MS (default 3000); smaller trickles
//                         are not reported.
// The "Prepare shaders" run (KYTY_PIPELINE_LIST_PREPARE=1, pipelineCache.cpp) prints two more in the
// same form, whatever KYTY_EVENTS says (the launcher's Precompile window reads them):
//   prepare-progress      about once a second: phase=<shaders|pipelines|compute|saving>
//                         shaders=<done>/<total> pipelines=<done>/<total> compute=<done>/<total>
//                         (replayed shader permutations, graphics pipelines, prewarmed compute
//                         pipelines; a total grows while earlier work adds to it), and once more
//                         with phase=saving before the caches are written;
//   prepare-done          once, before the process ends: seconds=<s> shaders=<n> pipelines=<n>
//                         known=<n> compute=<n> (what was built; known: pipelines the driver cache
//                         held already).
// KYTY_EVENTS_TRACE=1 adds one "KYTY_EVENT_TRACE:" line per second with the raw rates.
namespace KytyEvents {

[[nodiscard]] bool GuestConsoleEnabled();
[[nodiscard]] bool Enabled();

// Text the guest wrote to fd 1 or 2.
void GuestWrite(int fd, const void* data, size_t size);

void NoteFlip();
void NoteFileRead(uint64_t bytes);
void NoteBuildStart();
void NoteBuildEnd();

[[nodiscard]] uint64_t Frames();
[[nodiscard]] double   Seconds();
// "KYTY_EVENT: <name> frame=<n> t=<s> <extra>" for an event its caller prints (no newline).
[[nodiscard]] std::string FormatEvent(const char* name, const std::string& extra);
// Steady-clock nanoseconds (std::chrono::steady_clock epoch) at which the n-th (1-based, n <= 4)
// loading-end event was printed; 0 until then (or with KYTY_EVENTS=0).
[[nodiscard]] int64_t LoadingEndNs(uint32_t n);

struct Params {
	double   load_on_mbps  = 48.0;
	double   load_off_mbps = 16.0;
	double   min_fps       = 2.0;
	uint64_t hold_ms       = 5000;
	uint64_t idle_ms       = 3000;
	uint64_t idle_min      = 50;
};

// The event state machine, fed one sample of the running totals per tick (the monitor thread
// ticks every 250 ms). Returns the event lines without the newline.
class Detector {
public:
	explicit Detector(const Params& params): m_params(params) {}
	std::vector<std::string> Step(uint64_t now_ms, uint64_t frames, uint64_t read_bytes,
	                              uint64_t builds_started, int64_t builds_in_flight);
	[[nodiscard]] bool   Loading() const { return m_loading; }
	[[nodiscard]] double ReadMbps() const { return m_read_mbps; }
	[[nodiscard]] double Fps() const { return m_fps; }

private:
	struct Sample {
		uint64_t ms = 0, frames = 0, bytes = 0;
	};
	Params              m_params;
	std::vector<Sample> m_samples;
	double              m_read_mbps        = 0.0;
	double              m_fps              = 0.0;
	bool                m_first_flip       = false;
	bool                m_loading          = false;
	uint64_t            m_load_start_ms    = 0;
	uint64_t            m_load_bytes0      = 0;
	uint64_t            m_quiet_since_ms   = UINT64_MAX;
	uint64_t            m_idle_since_ms    = UINT64_MAX;
	uint64_t            m_builds_reported  = 0;
};

// "GUEST: ..." lines for `text` written to `fd`, as GuestWrite prints them (for the tests).
std::vector<std::string> SplitForTest(int fd, const std::string& text, uint64_t frame, double seconds);

} // namespace KytyEvents
