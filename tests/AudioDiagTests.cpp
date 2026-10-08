#include "libs/audioDiag.h"
#include "loader/faultMemoryDump.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AudioDiagTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

std::string ReadAll(std::FILE* file) {
	std::fflush(file);
	std::rewind(file);
	std::string text;
	char        chunk[512];
	size_t      n = 0;
	while ((n = std::fread(chunk, 1, sizeof(chunk), file)) > 0) {
		text.append(chunk, n);
	}
	return text;
}

size_t Count(const std::string& text, const char* needle) {
	size_t count = 0;
	for (size_t at = text.find(needle); at != std::string::npos;
	     at = text.find(needle, at + std::strlen(needle))) {
		count++;
	}
	return count;
}

void TestPrefixAndErrorOnce() {
	std::FILE* sink = std::tmpfile();
	Check(sink != nullptr, "tmpfile");
	Libs::Audio::Diag::Reporter reporter(sink);
	reporter.Print("port %d open", 3);
	Check(reporter.Error("sceAudioOutOpen", -2144993277, 7) == -2144993277,
	      "Error returns the code it reports");
	(void)reporter.Error("sceAudioOutOpen", -2144993277, 8);
	(void)reporter.Error("sceAudioOutClose", -2144993277, 9);
	(void)reporter.Error("sceAudioOutOpen", -2144993276, 10);
	const auto text = ReadAll(sink);
	std::fclose(sink);
	Check(Count(text, "AudioDiag: port 3 open\n") == 1, "plain line with prefix");
	Check(Count(text, "sceAudioOutOpen returned 0x80260003") == 1,
	      "an (api, code) pair prints once");
	Check(Count(text, "(detail 7)") == 1, "the first occurrence's detail is shown");
	Check(Count(text, "sceAudioOutClose returned 0x80260003") == 1,
	      "the same code from another api prints");
	Check(Count(text, "sceAudioOutOpen returned 0x80260004") == 1,
	      "another code from the same api prints");
}

void TestBudget() {
	std::FILE* sink = std::tmpfile();
	Check(sink != nullptr, "tmpfile");
	Libs::Audio::Diag::Reporter reporter(sink);
	for (int i = 0; i < Libs::Audio::Diag::Reporter::LINE_BUDGET + 50; i++) {
		reporter.Print("line %d", i);
	}
	const auto text = ReadAll(sink);
	std::fclose(sink);
	Check(Count(text, "AudioDiag: line ") ==
	          static_cast<size_t>(Libs::Audio::Diag::Reporter::LINE_BUDGET),
	      "lines stop at the budget");
	Check(Count(text, "further audio diagnostics are off") == 1, "the cut-off is announced once");
}

void TestDisabled() {
	std::FILE* sink = std::tmpfile();
	Check(sink != nullptr, "tmpfile");
	Libs::Audio::Diag::Reporter reporter(sink, false);
	reporter.Print("hidden");
	Check(reporter.Error("api", -1, 0) == -1, "a disabled reporter still returns the code");
	const auto text = ReadAll(sink);
	std::fclose(sink);
	Check(text.empty(), "a disabled reporter prints nothing");
}

void TestFaultWindows() {
	namespace Dump = Loader::FaultMemoryDump;
	// Two readable "heap" ranges; everything else unreadable.
	const uint64_t heap_a = 0x32f766000, heap_a_end = 0x32f768000;
	const uint64_t heap_b = 0x32f770000, heap_b_end = 0x32f771000;
	auto readable = [&](uint64_t start, uint64_t size) {
		const uint64_t end = start + size;
		return (start >= heap_a && end <= heap_a_end) || (start >= heap_b && end <= heap_b_end);
	};
	uint64_t regs[Dump::REGISTER_COUNT] = {};
	regs[0]  = 0x32f770d50; // rax: the object whose index field was garbage
	regs[2]  = 0xfffffffa21000000; // rcx: not an address
	regs[3]  = 0x200; // rdx: small integer
	regs[7]  = 0x32f7669f0; // rsp: excluded, the stack is printed separately
	regs[13] = 0x32f7669b0; // r13: the bus
	regs[15] = 0x32f7669a0; // r15: its control block 0x10 before, inside r13's window
	regs[14] = 0x32f770010; // r14: near the start of heap_b, so only the bytes after it are readable
	Dump::Window windows[Dump::REGISTER_COUNT];
	const auto   count = Dump::PickWindows(regs, readable, windows, Dump::REGISTER_COUNT);
	Check(count == 3, "rax, r13 and r14 get windows; rcx, rdx, rsp and the covered r15 do not");
	Check(windows[0].reg == 0 && windows[0].start == 0x32f770d50 - Dump::BEFORE &&
	          windows[0].size == Dump::BEFORE + Dump::AFTER,
	      "rax window spans 0x40 before to 0xe0 after");
	Check(windows[0].start + windows[0].size >= 0x32f770d50 + 0xc8 + 8,
	      "the window holds the whole field at +0xc8");
	Check(windows[1].reg == 13 && windows[1].start == 0x32f7669b0 - Dump::BEFORE,
	      "r13 window");
	Check(windows[2].reg == 14 && windows[2].start == 0x32f770010 &&
	          windows[2].size == Dump::AFTER,
	      "an unreadable lead-in falls back to the bytes from the address on");

	Dump::Window one[1];
	Check(Dump::PickWindows(regs, readable, one, 1) == 1 && one[0].reg == 0,
	      "capacity bounds the windows");
}

} // namespace

int main() {
	TestPrefixAndErrorOnce();
	TestBudget();
	TestDisabled();
	TestFaultWindows();
	std::printf("AudioDiagTests: all passed\n");
	return 0;
}
