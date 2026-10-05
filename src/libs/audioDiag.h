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

// Console diagnostics for the audio stack: port opens and closes with their parameters, error
// results handed to the guest, the DualSense audio route and AvPlayer's guest buffers. They go
// to stdout because the console text is what users send with a crash report; the log file is
// often missing. KYTY_AUDIO_DIAG=0 turns them off. A line budget keeps a title that opens ports
// in a loop from flooding the console, and each (API, error) pair prints once.
namespace Libs::Audio::Diag {

class Reporter {
public:
	static constexpr int LINE_BUDGET   = 256;
	static constexpr int ERROR_ENTRIES = 64;

	explicit Reporter(std::FILE* sink, bool enabled = true): m_sink(sink), m_enabled(enabled) {}

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
		std::lock_guard lock(m_mutex);
		if (line == LINE_BUDGET) {
			std::fprintf(m_sink, "AudioDiag: %d lines printed; further audio diagnostics are off\n",
			             LINE_BUDGET);
		} else {
			char text[512];
			std::vsnprintf(text, sizeof(text), format, args);
			std::fprintf(m_sink, "AudioDiag: %s\n", text);
		}
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
	static Reporter reporter(stdout, EnabledByEnvironment());
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

inline int Error(const char* api, int code, int64_t detail = 0) {
	return Console().Error(api, code, detail);
}

} // namespace Libs::Audio::Diag

#endif // EMULATOR_INCLUDE_EMULATOR_AUDIO_DIAG_H_
