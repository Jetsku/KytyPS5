#include "common/assert.h"

#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Common {

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	HangWatchdog::NoteFatal(text, file, line);
	Log::WriteFatal(BuildFatalReport(title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	return 1;
}

int DbgExitIfHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr),
	                 file, line);
}

int DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	return DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

int DbgExitHandler(const char* file, int line, std::string_view text) {
	HangWatchdog::NoteFatal(text, file, line);
	Log::WriteFatal(BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

int DbgExitHandler(const char* file, int line, fmt::text_style style, std::string_view text) {
	HangWatchdog::NoteFatal(text, file, line);
	Log::WriteFatal(style, BuildFatalReport("--- Error ---", text, file, line));
	return 1;
}

void DbgExit(int status) {
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
#if defined(_WIN32)
	// From chenxiao07/KytyPS5 150517a8a. std::_Exit is ExitProcess: it kills the other threads
	// wherever they are, then runs every DLL's detach routine on this thread, where a driver's can
	// wait forever on what a killed thread held (a crashed emulator stayed a one-thread process that
	// could not be killed, holding its memory and the exe). KYTY_FATAL_EXIT_PROCESS=1 keeps _Exit.
	static const bool exit_process = [] {
		const char* value = std::getenv("KYTY_FATAL_EXIT_PROCESS");
		return value != nullptr && value[0] == '1';
	}();
	if (!exit_process) {
		TerminateProcess(GetCurrentProcess(), static_cast<UINT>(status));
	}
#endif
	std::_Exit(status);
}

} // namespace Common
