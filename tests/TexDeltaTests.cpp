// KYTY_DRAW_RUN_TEXDELTA (drawPrep/texDelta.h): the key split (full key vs texture-free key), the
// candidate decision, the texture verdict and the record's mark replacement.
#include "graphics/host_gpu/renderer/drawPrep/texDelta.h"

#include <cstdio>
#include <vector>

namespace {

using namespace Libs::Graphics::DrawRun::TexDelta;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

void TestKeySplit() {
	int                              vs = 0;
	int                              ps = 0;
	int                              other_ps = 0;
	const std::array<const void*, 2> programs {&vs, &ps};
	const uint64_t                   registers = 0x1234;
	const uint64_t                   words_a   = 0xaaaa;
	const uint64_t                   words_b   = 0xbbbb;
	const auto                       key_a     = ChainKey(registers, programs, 3, words_a);
	const auto                       key_b     = ChainKey(registers, programs, 3, words_b);
	Check(key_a != key_b, "different T#/S# words: different keys");
	Check(TextureFreeKey(registers, programs, 3) == TextureFreeKey(registers, programs, 3),
	      "same registers and programs: same texture-free key");
	Check(TextureFreeKey(registers, programs, 3) != TextureFreeKey(registers + 1, programs, 3),
	      "registers are part of the texture-free key");
	Check(TextureFreeKey(registers, programs, 3) != TextureFreeKey(registers, {&vs, &other_ps}, 3),
	      "programs are part of the texture-free key");
	Check(TextureFreeKey(registers, programs, 3) != TextureFreeKey(registers, programs, 1),
	      "stage flags are part of the texture-free key");
	Check(TextureFreeKey(registers, programs, 3) != key_a && TextureFreeKey(registers, programs, 3) != key_b,
	      "a texture-free key never stands for a full key");
	Check(ChainKey(registers, {nullptr, &ps}, 3, words_a) != key_a, "the swap key differs from the key");
	Check(ChainKey(0, {nullptr, nullptr}, 0, 0) != 0 && TextureFreeKey(0, {nullptr, nullptr}, 0) != 0,
	      "keys are never 0");
}

Inputs DeltaInputs() {
	Inputs in;
	in.mode            = 1;
	in.record_valid    = true;
	in.key             = 10;
	in.record_key      = 11;
	in.free_key        = 20;
	in.record_free_key = 20;
	return in;
}

void TestDecide() {
	auto in = DeltaInputs();
	Check(Decide(in) == Decision::Delta, "keys differ, texture-free keys equal: delta");
	in.mode = 2;
	Check(Decide(in) == Decision::Delta, "verify mode decides alike");
	in      = DeltaInputs();
	in.mode = 0;
	Check(Decide(in) == Decision::Off, "switch off");
	in              = DeltaInputs();
	in.record_valid = false;
	Check(Decide(in) == Decision::Off, "no record");
	in     = DeltaInputs();
	in.key = 0;
	Check(Decide(in) == Decision::Off, "no key");
	in            = DeltaInputs();
	in.record_key = in.key;
	Check(Decide(in) == Decision::KeyMatched, "same key: a plain continuation");
	in                 = DeltaInputs();
	in.record_free_key = 21;
	Check(Decide(in) == Decision::Other, "texture-free keys differ");
	in          = DeltaInputs();
	in.free_key = 0;
	in.record_free_key = 0;
	Check(Decide(in) == Decision::Other, "no texture-free key (switch was off when prepared)");
	in               = DeltaInputs();
	in.record_writes = true;
	Check(Decide(in) == Decision::Writes, "the record writes buffers");
	in             = DeltaInputs();
	in.draw_writes = true;
	Check(Decide(in) == Decision::Writes, "the draw writes buffers");
}

void TestTextures() {
	const std::vector<AttachmentFacts> attachments {{5, 0x10000, 0x1000}, {6, 0x20000, 0x2000}, {0, 0, 0}};
	std::vector<TextureFacts>          textures {{1, false, 0x30000, 0x100}, {2, false, 0, 0}};
	Check(CheckTextures(textures, attachments) == TextureVerdict::Ok, "separate textures");
	textures.push_back({6, false, 0x20000, 0x2000});
	Check(CheckTextures(textures, attachments) == TextureVerdict::Attachment, "a texture is an attachment");
	textures.back() = {7, false, 0x20800, 0x100};
	Check(CheckTextures(textures, attachments) == TextureVerdict::OverAttachment, "a texture over an attachment");
	textures.back() = {7, false, 0x21fff, 0x10};
	Check(CheckTextures(textures, attachments) == TextureVerdict::OverAttachment, "partial overlap at the end");
	textures.back() = {7, false, 0x22000, 0x10};
	Check(CheckTextures(textures, attachments) == TextureVerdict::Ok, "adjacent is not over");
	textures.back() = {7, true, 0x40000, 0x10};
	Check(CheckTextures(textures, attachments) == TextureVerdict::Storage, "a storage binding");
	textures.back() = {0, false, 0, 0};
	Check(CheckTextures(textures, attachments) == TextureVerdict::Ok, "a null texture and a null attachment");
	Check(CheckTextures({}, attachments) == TextureVerdict::Ok, "no textures");
}

struct Mark {
	int  id      = 0;
	bool texture = false;
};

void TestReplace() {
	std::vector<Mark>       marks {{1, true}, {2, true}, {3, true}, {10, false}, {11, false}};
	const std::vector<Mark> fresh {{7, true}};
	auto                    begin = ReplaceTextureMarks<Mark>(marks, 3, fresh);
	Check(begin == 1 && marks.size() == 3, "one texture, two attachments");
	Check(marks[0].id == 7 && marks[1].id == 10 && marks[2].id == 11, "textures first, attachments kept in order");
	const std::vector<Mark> more {{20, true}, {21, true}, {22, true}, {23, true}};
	begin = ReplaceTextureMarks<Mark>(marks, begin, more);
	Check(begin == 4 && marks.size() == 6 && marks[4].id == 10 && marks[5].id == 11, "more textures");
	begin = ReplaceTextureMarks<Mark>(marks, begin, {});
	Check(begin == 0 && marks.size() == 2 && marks[0].id == 10, "no textures");
	std::vector<Mark> only_textures {{1, true}};
	begin = ReplaceTextureMarks<Mark>(only_textures, 1, fresh);
	Check(begin == 1 && only_textures.size() == 1 && only_textures[0].id == 7, "no attachments");
}

} // namespace

int main() {
	TestKeySplit();
	TestDecide();
	TestTextures();
	TestReplace();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("tex_delta: all passed\n");
	return 0;
}
