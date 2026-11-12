#include "common/assert.h"

#include "common/hangWatchdog.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>

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

// The calling thread's return addresses as module+offset (resolved offline against the PDB, e.g.
// llvm-symbolizer --obj=kyty_emulator.exe with the offset added to the image base).
static std::string FatalStack() {
#if defined(_WIN32)
	void*      frames[48] = {};
	const auto count      = CaptureStackBackTrace(2, 48, frames, nullptr);
	std::string text      = "--- Stack ---";
	for (USHORT i = 0; i < count; i++) {
		HMODULE module = nullptr;
		char    name[MAX_PATH] = {};
		if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		                       static_cast<LPCSTR>(frames[i]), &module) != 0 &&
		    GetModuleFileNameA(module, name, MAX_PATH) != 0) {
			const char* base = std::strrchr(name, '\\');
			text += fmt::format("\n  {}+0x{:x}", base != nullptr ? base + 1 : name,
			                    reinterpret_cast<uintptr_t>(frames[i]) - reinterpret_cast<uintptr_t>(module));
		} else {
			text += fmt::format("\n  0x{:x}", reinterpret_cast<uintptr_t>(frames[i]));
		}
	}
	return text + "\n";
#else
	return {};
#endif
}

// Upstream 0631e343f: the handlers no longer return. The report goes to the console and the log
// (the former Log::WriteFatal), then the process ends with status 321 (the former EXIT_HALT).
[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line, bool with_stack) {
	HangWatchdog::NoteFatal(text, file, line);
	Log::WriteToConsoleAndLog(BuildFatalReport(title, text, file, line) +
	                          (with_stack ? FatalStack() : std::string {}));
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
		TerminateProcess(GetCurrentProcess(), 321);
	}
#endif
	std::_Exit(321);
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file,
	          line, true);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line, true);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	DbgReport("--- Error ---", text, file, line, false);
}

} // namespace Common
