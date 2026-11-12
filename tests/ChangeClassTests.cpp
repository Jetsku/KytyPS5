// KYTY_CP_COMMIT_STATS change classes (drawPrep/changeClass.h): the difference mask of a recorded draw
// against the previous one, and the class (reason) the statistics file it under.
#include "graphics/host_gpu/renderer/drawPrep/changeClass.h"

#include <cstdio>

namespace {

using namespace Libs::Graphics::CommitStats;
using namespace Libs::Graphics::CommitStats::ChangeClass;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

DrawShape Base() {
	DrawShape s;
	s.rendering_serial = 7;
	s.command          = 0x1000;
	s.pipeline         = 0x2000;
	s.programs         = 11;
	s.textures         = 12;
	s.samplers         = 13;
	s.attachments      = 14;
	s.dynamic          = 15;
	s.buffers          = 16;
	return s;
}

void TestMask() {
	const auto a = Base();
	auto       b = Base();
	Check(MaskOf(nullptr, b) == ChangeFirst, "no previous draw: first");
	Check(MaskOf(&a, b) == 0, "identical shapes: no change");
	b.buffers = 99;
	b.push    = 98;
	Check(MaskOf(&a, b) == 0, "per-draw data is not a change");
	b             = Base();
	b.textures    = 1;
	b.samplers    = 2;
	b.dynamic     = 3;
	Check(MaskOf(&a, b) == (ChangeTextures | ChangeSamplers | ChangeDynamic), "texture + sampler + dynamic bits");
	b             = Base();
	b.attachments = 1;
	Check(MaskOf(&a, b) == ChangePass, "attachments: pass");
	b         = Base();
	b.command = 1;
	Check(MaskOf(&a, b) == ChangePass, "command buffer: pass");
	b                  = Base();
	b.rendering_serial = 8;
	Check(MaskOf(&a, b) == ChangePass, "rendering instance: pass");
	auto outside             = Base();
	outside.rendering_serial = 0;
	Check(MaskOf(&outside, Base()) == ChangePass, "previous outside rendering: pass");
	auto writer          = Base();
	writer.shader_writes = true;
	Check(MaskOf(&writer, Base()) == ChangeWrites, "previous wrote");
	b          = Base();
	b.programs = 1;
	b.pipeline = 1;
	Check(MaskOf(&a, b) == (ChangePrograms | ChangePipeline), "programs + pipeline");
}

void TestReason() {
	auto b = Base();
	Check(ReasonOf(ChangeFirst, b) == ReasonFirst, "first");
	Check(ReasonOf(ChangeFirst | ChangePass | ChangePrograms, b) == ReasonFirst, "first wins");
	Check(ReasonOf(ChangePass | ChangePrograms | ChangeTextures, b) == ReasonPass, "pass before programs");
	Check(ReasonOf(ChangePrograms | ChangePipeline | ChangeTextures, b) == ReasonPrograms, "programs before pipeline");
	Check(ReasonOf(ChangePipeline | ChangeTextures, b) == ReasonPipeState, "pipeline with the same programs");
	Check(ReasonOf(ChangeTextures | ChangeSamplers | ChangeDynamic, b) == ReasonTexSamp, "textures and samplers");
	Check(ReasonOf(ChangeTextures | ChangeDynamic, b) == ReasonTextures, "textures before dynamic");
	Check(ReasonOf(ChangeSamplers, b) == ReasonSamplers, "samplers alone");
	Check(ReasonOf(ChangeDynamic | ChangeWrites, b) == ReasonDynamic, "dynamic before writes");
	Check(ReasonOf(ChangeWrites, b) == ReasonWrites, "writes alone");
	Check(ReasonOf(0, b) == ReasonRunMiss, "nothing differs: the run missed");
	b.run_continued = true;
	b.kernel        = KernelOutcome::Taken;
	Check(ReasonOf(ChangeTextures, b) == ReasonContTaken, "a continued draw is filed by its kernel outcome");
	b.kernel = KernelOutcome::Left;
	Check(ReasonOf(0, b) == ReasonContLeft, "left");
	b.kernel = KernelOutcome::Declined;
	Check(ReasonOf(0, b) == ReasonContDeclined, "declined");
	b.kernel = KernelOutcome::NoEntry;
	Check(ReasonOf(0, b) == ReasonContNoEntry, "no entry");
	b.kernel = KernelOutcome::Off;
	Check(ReasonOf(0, b) == ReasonContNoEntry, "kernel off counts as no entry");
	Check(kReasonNames[ReasonOther][0] == 'o' && kChangeLetters[7] == 'F', "names line up");
}

} // namespace

int main() {
	TestMask();
	TestReason();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("change_class: all passed\n");
	return 0;
}
