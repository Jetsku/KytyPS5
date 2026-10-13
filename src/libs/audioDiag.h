#ifndef EMULATOR_INCLUDE_EMULATOR_AUDIO_DIAG_H_
#define EMULATOR_INCLUDE_EMULATOR_AUDIO_DIAG_H_

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Console diagnostics for the audio stack: port opens and closes with their parameters, error
// results handed to the guest, the DualSense audio route and AvPlayer's guest buffers. They go
// to stdout because the console text is what users send with a crash report; the log file is
// often missing. KYTY_AUDIO_DIAG=0 turns them off. A line budget keeps a title that opens ports
// in a loop from flooding the console, and each (API, error) pair prints once.
// Once audio.cpp installs the log writer, the console reporter's lines also reach the printf file
// (_kyty.txt), which is what most players send.
namespace Libs::Audio::Diag {

// Writes one finished line to the console and the log file (Log::WriteToConsoleAndLog).
using LogWriter = void (*)(const std::string& text);
inline std::atomic<LogWriter>& ConsoleLogWriter() {
	static std::atomic<LogWriter> writer {nullptr};
	return writer;
}

class Reporter {
public:
	static constexpr int LINE_BUDGET   = 256;
	static constexpr int ERROR_ENTRIES = 64;

	explicit Reporter(std::FILE* sink, bool enabled = true, bool use_log_writer = false)
	    : m_sink(sink), m_enabled(enabled), m_use_log_writer(use_log_writer) {}

	[[nodiscard]] bool Enabled() const { return m_enabled && m_sink != nullptr; }

#if defined(__GNUC__) || defined(__clang__)
	__attribute__((format(printf, 2, 3)))
#endif
	void Print(const char* format, ...) {
		va_list args;
		va_start(args, format);
		VPrint(format, args);
		va_end(args);
	}

	void VPrint(const char* format, va_list args) {
		if (!Enabled()) {
			return;
		}
		const int line = m_lines.fetch_add(1, std::memory_order_relaxed);
		if (line > LINE_BUDGET) {
			return;
		}
		char text[512];
		if (line == LINE_BUDGET) {
			std::snprintf(text, sizeof(text), "%d lines printed; further audio diagnostics are off",
			              LINE_BUDGET);
		} else {
			std::vsnprintf(text, sizeof(text), format, args);
		}
		const LogWriter writer = m_use_log_writer ? ConsoleLogWriter().load() : nullptr;
		if (writer != nullptr) {
			writer(std::string("AudioDiag: ") + text + "\n");
			return;
		}
		std::lock_guard lock(m_mutex);
		std::fprintf(m_sink, "AudioDiag: %s\n", text);
		std::fflush(m_sink);
	}

	// Reports an error result the emulator returns to the guest, once per (api, code), and
	// returns the code so a call site can write `return Diag::Error(...)`.
	int Error(const char* api, int code, int64_t detail) {
		if (Enabled() && FirstError(api, code)) {
			Print("%s returned 0x%08x to the game (detail %lld)", api, static_cast<unsigned>(code),
			      static_cast<long long>(detail));
		}
		return code;
	}

	[[nodiscard]] int LinesPrinted() const { return m_lines.load(std::memory_order_relaxed); }

private:
	bool FirstError(const char* api, int code) {
		std::lock_guard lock(m_mutex);
		for (int i = 0; i < m_error_count; i++) {
			if (m_errors[i].code == code && std::strcmp(m_errors[i].api, api) == 0) {
				return false;
			}
		}
		if (m_error_count < ERROR_ENTRIES) {
			m_errors[m_error_count++] = {api, code};
		}
		return true;
	}

	struct ErrorKey {
		const char* api  = "";
		int         code = 0;
	};

	std::FILE*                           m_sink;
	bool                                 m_enabled;
	bool                                 m_use_log_writer;
	std::atomic<int>                     m_lines {0};
	std::mutex                           m_mutex;
	std::array<ErrorKey, ERROR_ENTRIES> m_errors {};
	int                                  m_error_count = 0;
};

inline bool EnabledByEnvironment() {
	const char* value = std::getenv("KYTY_AUDIO_DIAG");
	return value == nullptr || value[0] == '\0' ||
	       !(std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0 ||
	         std::strcmp(value, "false") == 0);
}

inline Reporter& Console() {
	static Reporter reporter(stdout, EnabledByEnvironment(), true);
	return reporter;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void Print(const char* format, ...) {
	va_list args;
	va_start(args, format);
	Console().VPrint(format, args);
	va_end(args);
}

// KYTY_AUDIO_TRACE=1: output timing trace (SDL queue levels, clears, re-primes, device-side
// shortfalls, AudioOut2 push results) with a summary per port every TRACE_INTERVAL_US. Off by
// default; the lines go to the console and the log file and have no line budget.
inline bool TraceEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_AUDIO_TRACE");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

inline constexpr uint64_t TRACE_INTERVAL_US = 10000000;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline void Trace(const char* format, ...) {
	char    text[1024];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (const LogWriter writer = ConsoleLogWriter().load(); writer != nullptr) {
		writer(std::string("AudioTrace: ") + text + "\n");
	} else {
		std::printf("AudioTrace: %s\n", text);
		std::fflush(stdout);
	}
}

// An error result handed to the guest: one AudioDiag line per (api, code), and with
// KYTY_AUDIO_TRACE one AudioTrace line per occurrence (the first ERROR_TRACE_BUDGET).
inline constexpr int ERROR_TRACE_BUDGET = 512;
inline int Error(const char* api, int code, int64_t detail = 0) {
	if (TraceEnabled()) {
		static std::atomic<int> lines {0};
		if (lines.fetch_add(1, std::memory_order_relaxed) < ERROR_TRACE_BUDGET) {
			Trace("%s returned 0x%08x to the game (detail %lld)", api, static_cast<unsigned>(code),
			      static_cast<long long>(detail));
		}
	}
	return Console().Error(api, code, detail);
}

} // namespace Libs::Audio::Diag

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_DIAG_H_
