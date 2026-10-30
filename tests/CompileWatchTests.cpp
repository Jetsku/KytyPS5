#include "common/compileWatch.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
void Check(bool ok, const char* message, const std::string& text = {}) {
	if (!ok) {
		std::fprintf(stderr, "CompileWatchTests: failed: %s\n%s\n", message, text.c_str());
		std::abort();
	}
}
bool Has(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

void QuietWhenShort() {
	CompileWatch::ResetForTest();
	const auto            t0 = CompileWatch::NowMsForTest();
	CompileWatch::Build   build(CompileWatch::Kind::GraphicsPipeline, 0x11, 0x22);
	CompileWatch::Wait    wait("draw", CompileWatch::Kind::GraphicsPipeline, 0x11, 0x22);
	Check(CompileWatch::CheckForTest(t0 + 1000).empty(), "a 1 s wait prints nothing");
}

void BlockedOnceThenPeriodic() {
	CompileWatch::ResetForTest();
	const auto          t0 = CompileWatch::NowMsForTest();
	CompileWatch::QueueCounter queued("background compute");
	queued.Add();
	CompileWatch::Build rt(CompileWatch::Kind::RtPipeline, 0x6d9edffa4c081747ull);
	{
		CompileWatch::Wait wait("dispatch", CompileWatch::Kind::RtPipeline, 0x6d9edffa4c081747ull);
		const auto         first = CompileWatch::CheckForTest(t0 + 2500);
		Check(Has(first, "Compile wait: dispatch blocked for 2.5 s on RT pipeline 0x6d9edffa4c081747 (building for 2.5 s)"),
		      "blocked line names the build", first);
		Check(Has(first, "1 build(s) in flight (oldest: RT pipeline 0x6d9edffa4c081747 2.5 s); 1 background compute queued"),
		      "periodic line lists builds and queues", first);
		const auto second = CompileWatch::CheckForTest(t0 + 3500);
		Check(second.empty(), "both lines are rate-limited", second);
		const auto third = CompileWatch::CheckForTest(t0 + 13000);
		Check(!Has(third, "dispatch blocked for 13.0 s on") || Has(third, "in flight"), "blocked line only once", third);
		Check(Has(third, "build(s) in flight"), "periodic line again after the period", third);
	}
	queued.Sub();
}

void NotStartedYet() {
	CompileWatch::ResetForTest();
	const auto         t0 = CompileWatch::NowMsForTest();
	CompileWatch::Wait wait("draw", CompileWatch::Kind::GraphicsPipeline, 0xaa, 0xbb);
	const auto         text = CompileWatch::CheckForTest(t0 + 2100);
	Check(Has(text, "draw blocked for 2.1 s on graphics pipeline VS 0x00000000000000aa PS 0x00000000000000bb (not started yet)"),
	      "a wait without a build", text);
}

void FrameStall() {
	CompileWatch::ResetForTest();
	Check(CompileWatch::CheckForTest(CompileWatch::NowMsForTest() + 60000).empty(), "no stall before the first flip");
	CompileWatch::NoteFlip();
	const auto t0 = CompileWatch::NowMsForTest();
	CompileWatch::NoteQueueWait(0, "WAIT_REG_MEM", 0x1234);
	CompileWatch::Build build(CompileWatch::Kind::RtPipeline, 0x42);
	auto text = CompileWatch::CheckForTest(t0 + 9000);
	Check(!Has(text, "Frame stall"), "9 s without a flip is not a stall", text);
	text = CompileWatch::CheckForTest(t0 + 11000);
	Check(Has(text, "Frame stall: no flip for 11.0 s (after flip 1). Command processor waits: queue 0 WAIT_REG_MEM 0x1234"),
	      "stall line names the CP wait", text);
	Check(Has(text, "No draw or dispatch waits for a compile. 1 build(s) in flight: RT pipeline 0x0000000000000042"),
	      "stall line lists builds", text);
	text = CompileWatch::CheckForTest(t0 + 20000);
	Check(!Has(text, "Frame stall"), "the stall line repeats only after 30 s", text);
	text = CompileWatch::CheckForTest(t0 + 41500);
	Check(Has(text, "Frame stall"), "the stall line repeats", text);
	CompileWatch::ClearQueueWait(0);
	CompileWatch::NoteFlip();
	text = CompileWatch::CheckForTest(CompileWatch::NowMsForTest() + 1000);
	Check(!Has(text, "Frame stall"), "a flip ends the stall", text);
}

void Precompile() {
	CompileWatch::ResetForTest();
	const auto t0 = CompileWatch::NowMsForTest();
	CompileWatch::NotePrecompile(120, 2580);
	CompileWatch::Build build(CompileWatch::Kind::Shader, 0x5);
	const auto          text = CompileWatch::CheckForTest(t0 + 6000);
	Check(Has(text, "shader 0x0000000000000005 6.0 s); precompile 120/2580"), "precompile progress shown", text);
}
} // namespace

int main() {
	QuietWhenShort();
	BlockedOnceThenPeriodic();
	NotStartedYet();
	FrameStall();
	Precompile();
	std::printf("CompileWatchTests: passed\n");
	return 0;
}
