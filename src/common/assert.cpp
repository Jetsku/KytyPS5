#include "common/assert.h"

#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <thread>

namespace Common {

static std::string BuildFatalReport(const char* title, std::string_view text, const char* file,
                                    int line) {
	return fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL, title, text, file,
	                   line);
}

// A fatal error must end the process even when an emergency shutdown step blocks (a lock held by
// the thread that failed, the profiler's shutdown, a driver call on a lost device): once the
// first fatal path starts, a detached watchdog terminates the process after a few seconds.
// The normal path still ends earlier through DbgExit's _Exit.
static void StartFatalWatchdog(int status) {
	static std::once_flag started;
	std::call_once(started, [status] {
		try {
			std::thread([status] {
				std::this_thread::sleep_for(std::chrono::seconds(8));
				std::fputs("Fatal error: emergency shutdown did not finish in 8 s, terminating\n",
				           stderr);
				std::fflush(nullptr);
				std::_Exit(status);
			}).detach();
		} catch (...) {
			// No thread: the shutdown proceeds as before.
		}
	});
}

static int DbgReport(const char* title, std::string_view text, const char* file, int line) {
	HangWatchdog::NoteFatal(text, file, line);
	StartFatalWatchdog(1);
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
	StartFatalWatchdog(status);
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(status);
}

} // namespace Common
