#include "common/kytyEvents.h"

#include "common/guestProbe.h"
#include "common/logging/log.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace KytyEvents {
namespace {

const auto g_start = std::chrono::steady_clock::now();

uint64_t ClockMs() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_start).count());
}

bool EnvOn(const char* name) {
	const char* value = std::getenv(name);
	return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

double EnvDouble(const char* name, double fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') return fallback;
	return std::strtod(value, nullptr);
}

std::atomic<uint64_t> g_frames {0};
std::atomic<uint64_t> g_read_bytes {0};
std::atomic<uint64_t> g_builds_started {0};
std::atomic<int64_t>  g_builds_in_flight {0};
// Steady-clock nanoseconds (since its epoch) of the first loading-end events (LoadingEndNs).
std::array<std::atomic<int64_t>, 4> g_load_end_ns {};
std::atomic<uint32_t>               g_load_ends {0};

constexpr size_t MaxPartial = 4096; // a line without a newline is printed once this long

struct GuestBuffers {
	std::mutex                 mutex;
	std::array<std::string, 2> partial;
};
GuestBuffers& Buffers() {
	static GuestBuffers buffers;
	return buffers;
}

std::string GuestLine(int fd, std::string_view text, uint64_t frame, double seconds) {
	while (!text.empty() && (text.back() == '\r' || text.back() == '\n')) text.remove_suffix(1);
	return fmt::format("GUEST: frame={} t={:.3f} |{} {}", frame, seconds, fd == 2 ? " [stderr]" : "", text);
}

// Appends `data` to `partial` and moves the finished lines to `out`.
void Split(int fd, std::string& partial, std::string_view data, uint64_t frame, double seconds,
           std::vector<std::string>& out) {
	for (;;) {
		const auto nl = data.find('\n');
		if (nl == std::string_view::npos) break;
		partial.append(data.substr(0, nl));
		out.push_back(GuestLine(fd, partial, frame, seconds));
		partial.clear();
		data.remove_prefix(nl + 1);
	}
	partial.append(data);
	if (partial.size() >= MaxPartial) {
		out.push_back(GuestLine(fd, partial, frame, seconds));
		partial.clear();
	}
}

std::string EventLine(const char* name, uint64_t frame, uint64_t now_ms, const std::string& extra) {
	return fmt::format("KYTY_EVENT: {} frame={} t={:.3f}{}{}", name, frame, static_cast<double>(now_ms) / 1000.0,
	                   extra.empty() ? "" : " ", extra);
}

void Print(const std::string& line) {
	if (line.find("KYTY_EVENT: loading-end") != std::string::npos) {
		const auto index = g_load_ends.fetch_add(1, std::memory_order_relaxed);
		if (index < g_load_end_ns.size()) {
			g_load_end_ns[index].store(std::chrono::steady_clock::now().time_since_epoch().count(),
			                           std::memory_order_release);
		}
	}
	Log::WriteToConsoleAndLog(line + "\n");
	// KYTY_GUEST_PROBE_ARM_LOAD_END=<n> (diagnostic): arm the deferred ('^') guest probes at the n-th
	// loading-end event (Ghost of Yotei: 1 = title menu, 2 = gameplay), so per-frame probes do not slow the load.
	static const unsigned long arm_at = [] {
		const char* v = std::getenv("KYTY_GUEST_PROBE_ARM_LOAD_END");
		return v != nullptr ? std::strtoul(v, nullptr, 10) : 0ul;
	}();
	static unsigned long load_ends = 0;
	if (arm_at != 0 && line.find("KYTY_EVENT: loading-end") != std::string::npos && ++load_ends == arm_at) {
		Common::GuestProbe::ArmDeferred("loading-end");
	}
}

void MonitorThread() {
#ifdef _WIN32
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
	Params params;
	params.load_on_mbps  = EnvDouble("KYTY_EVENTS_LOAD_ON_MBPS", params.load_on_mbps);
	params.load_off_mbps = EnvDouble("KYTY_EVENTS_LOAD_OFF_MBPS", params.load_off_mbps);
	params.min_fps       = EnvDouble("KYTY_EVENTS_MIN_FPS", params.min_fps);
	params.hold_ms       = static_cast<uint64_t>(EnvDouble("KYTY_EVENTS_HOLD_MS", static_cast<double>(params.hold_ms)));
	params.idle_ms       = static_cast<uint64_t>(EnvDouble("KYTY_EVENTS_IDLE_MS", static_cast<double>(params.idle_ms)));
	params.idle_min      = static_cast<uint64_t>(EnvDouble("KYTY_EVENTS_IDLE_MIN_BUILDS", static_cast<double>(params.idle_min)));
	const bool trace     = std::getenv("KYTY_EVENTS_TRACE") != nullptr && EnvOn("KYTY_EVENTS_TRACE");
	Detector   detector(params);
	uint64_t   last_trace_ms = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
		try {
			const auto now    = ClockMs();
			const auto frames = g_frames.load(std::memory_order_relaxed);
			const auto lines  = detector.Step(now, frames, g_read_bytes.load(std::memory_order_relaxed),
			                                  g_builds_started.load(std::memory_order_relaxed),
			                                  g_builds_in_flight.load(std::memory_order_relaxed));
			for (const auto& line: lines) Print(line);
			if (trace && now - last_trace_ms >= 1000) {
				last_trace_ms = now;
				Print(fmt::format("KYTY_EVENT_TRACE: frame={} t={:.3f} fps={:.1f} read_mbps={:.1f} builds_in_flight={} "
				                  "builds_started={} loading={}",
				                  frames, static_cast<double>(now) / 1000.0, detector.Fps(), detector.ReadMbps(),
				                  g_builds_in_flight.load(std::memory_order_relaxed),
				                  g_builds_started.load(std::memory_order_relaxed), detector.Loading() ? 1 : 0));
			}
		} catch (...) {
			// Logging must never change emulation.
		}
	}
}

void EnsureMonitor() {
	static std::once_flag once;
	std::call_once(once, [] {
		if (!Enabled()) return;
		try {
			std::thread(MonitorThread).detach();
		} catch (...) {
		}
	});
}

} // namespace

bool GuestConsoleEnabled() {
	static const bool enabled = EnvOn("KYTY_GUEST_CONSOLE");
	return enabled;
}

bool Enabled() {
	static const bool enabled = EnvOn("KYTY_EVENTS");
	return enabled;
}

uint64_t Frames() {
	return g_frames.load(std::memory_order_relaxed);
}

double Seconds() {
	return static_cast<double>(ClockMs()) / 1000.0;
}

std::string FormatEvent(const char* name, const std::string& extra) {
	return EventLine(name, Frames(), ClockMs(), extra);
}

int64_t LoadingEndNs(uint32_t n) {
	if (n == 0 || n > g_load_end_ns.size()) return 0;
	return g_load_end_ns[n - 1].load(std::memory_order_acquire);
}

void GuestWrite(int fd, const void* data, size_t size) {
	if (!GuestConsoleEnabled() || data == nullptr || size == 0 || (fd != 1 && fd != 2)) return;
	try {
		std::vector<std::string> lines;
		{
			auto&            buffers = Buffers();
			std::scoped_lock lock(buffers.mutex);
			Split(fd, buffers.partial[fd - 1], std::string_view(static_cast<const char*>(data), size), Frames(),
			      Seconds(), lines);
		}
		for (const auto& line: lines) Print(line);
	} catch (...) {
	}
}

void NoteFlip() {
	if (g_frames.fetch_add(1, std::memory_order_relaxed) == 0) EnsureMonitor();
}

void NoteFileRead(uint64_t bytes) {
	g_read_bytes.fetch_add(bytes, std::memory_order_relaxed);
	if (bytes != 0) EnsureMonitor();
}

void NoteBuildStart() {
	g_builds_started.fetch_add(1, std::memory_order_relaxed);
	g_builds_in_flight.fetch_add(1, std::memory_order_relaxed);
}

void NoteBuildEnd() {
	g_builds_in_flight.fetch_sub(1, std::memory_order_relaxed);
}

std::vector<std::string> Detector::Step(uint64_t now_ms, uint64_t frames, uint64_t read_bytes,
                                        uint64_t builds_started, int64_t builds_in_flight) {
	std::vector<std::string> out;

	m_samples.push_back({now_ms, frames, read_bytes});
	// Keep a window of at least one second.
	while (m_samples.size() > 2 && now_ms - m_samples[1].ms >= 1000) m_samples.erase(m_samples.begin());
	const auto& first = m_samples.front();
	const auto  span  = now_ms - first.ms;
	if (span > 0) {
		m_read_mbps = static_cast<double>(read_bytes - first.bytes) * 1000.0 / static_cast<double>(span) / 1048576.0;
		m_fps       = static_cast<double>(frames - first.frames) * 1000.0 / static_cast<double>(span);
	}

	if (!m_first_flip && frames > 0) {
		m_first_flip = true;
		out.push_back(EventLine("first-flip", frames, now_ms, ""));
	}

	if (span >= 750) {
		if (!m_loading) {
			if (m_read_mbps >= m_params.load_on_mbps) {
				m_loading        = true;
				m_load_start_ms  = first.ms;
				m_load_bytes0    = first.bytes;
				m_quiet_since_ms = UINT64_MAX;
				out.push_back(EventLine("loading-start", frames, now_ms, fmt::format("read_mbps={:.1f}", m_read_mbps)));
			}
		} else {
			const bool quiet = m_read_mbps < m_params.load_off_mbps && m_fps >= m_params.min_fps;
			if (!quiet) {
				m_quiet_since_ms = UINT64_MAX;
			} else if (m_quiet_since_ms == UINT64_MAX) {
				m_quiet_since_ms = now_ms;
			} else if (now_ms - m_quiet_since_ms >= m_params.hold_ms) {
				m_loading = false;
				out.push_back(EventLine(
				    "loading-end", frames, now_ms,
				    fmt::format("load_s={:.1f} read_mb={:.0f} quiet_since={:.3f} fps={:.1f}",
				                static_cast<double>(m_quiet_since_ms - m_load_start_ms) / 1000.0,
				                static_cast<double>(read_bytes - m_load_bytes0) / 1048576.0,
				                static_cast<double>(m_quiet_since_ms) / 1000.0, m_fps)));
			}
		}
	}

	if (m_first_flip) {
		if (builds_in_flight > 0 || builds_started == m_builds_reported) {
			m_idle_since_ms = UINT64_MAX;
		} else if (m_idle_since_ms == UINT64_MAX) {
			m_idle_since_ms = now_ms;
		} else if (now_ms - m_idle_since_ms >= m_params.idle_ms) {
			if (builds_started - m_builds_reported >= m_params.idle_min) {
				out.push_back(EventLine("shader-compile-idle", frames, now_ms,
				                        fmt::format("builds={} idle_since={:.3f}", builds_started - m_builds_reported,
				                                    static_cast<double>(m_idle_since_ms) / 1000.0)));
			}
			m_builds_reported = builds_started;
			m_idle_since_ms   = UINT64_MAX;
		}
	}
	return out;
}

std::vector<std::string> SplitForTest(int fd, const std::string& text, uint64_t frame, double seconds) {
	std::string              partial;
	std::vector<std::string> out;
	Split(fd, partial, text, frame, seconds, out);
	if (!partial.empty()) out.push_back("partial:" + partial);
	return out;
}

} // namespace KytyEvents
