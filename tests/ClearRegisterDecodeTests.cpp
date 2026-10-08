// Register clear decoding (imageInfo.h, KYTY_CLEAR_REGISTER_WIDE): the colour a DCC clear-to-register
// key (0x20) or a CMASK fast clear stands for, from CB_COLORn_CLEAR_WORD0/1.
//
// Regression: Astro Bot's galaxy map clears its RGBA16F normal G-buffer with DCC key 0x20. The
// 32-bit decoder rejected the format, so the clear was dropped and moving objects left smeared
// copies in the target (red ribbons and bands over the nebula after the lighting pass).
#include "graphics/host_gpu/renderer/image/imageInfo.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool condition, const char* description) {
	if (!condition) {
		std::fprintf(stderr, "ClearRegisterDecodeTests: failed: %s\n", description);
		std::abort();
	}
}

bool Near(float a, float b) {
	return std::fabs(a - b) <= 1e-6f * std::max(1.0f, std::fabs(b));
}

void CheckFloat4(const vk::ClearColorValue& clear, float r, float g, float b, float a,
                 const char* description) {
	Check(Near(clear.float32[0], r) && Near(clear.float32[1], g) && Near(clear.float32[2], b) &&
	          Near(clear.float32[3], a),
	      description);
}

void TestRgba16FloatUsesBothWords() {
	vk::ClearColorValue clear {};
	// Halves: R = 1.0 (0x3c00), G = -2.0 (0xc000) in word0; B = 0.5 (0x3800), A = 0 in word1.
	Check(DecodePackedColorClear64(vk::Format::eR16G16B16A16Sfloat, 0xc0003c00u, 0x00003800u, clear),
	      "RGBA16F register clear decodes");
	CheckFloat4(clear, 1.0f, -2.0f, 0.5f, 0.0f, "RGBA16F channels come from word0 and word1");
	clear.float32 = std::array {7.0f, 7.0f, 7.0f, 7.0f};
	Check(DecodePackedColorClear64(vk::Format::eR16G16B16A16Sfloat, 0, 0, clear),
	      "RGBA16F zero clear decodes");
	CheckFloat4(clear, 0.0f, 0.0f, 0.0f, 0.0f, "RGBA16F zero clear is all zero");
	// The 32-bit decoder alone (the old DCC path) cannot describe a 64-bit texel.
	Check(!DecodePackedColorClear(vk::Format::eR16G16B16A16Sfloat, 0, clear),
	      "32-bit decoder rejects RGBA16F");
}

void TestNarrowFormats() {
	vk::ClearColorValue clear {};
	Check(DecodePackedColorClear64(vk::Format::eR16G16Sfloat, 0x3c00bc00u, 0, clear), "RG16F decodes");
	CheckFloat4(clear, -1.0f, 1.0f, 0.0f, 0.0f, "RG16F channel 0 in the low half");
	Check(DecodePackedColorClear64(vk::Format::eR16Sfloat, 0x00003555u, 0, clear), "R16F decodes");
	Check(std::fabs(clear.float32[0] - 0.33325195f) < 1e-6f, "R16F value");
	Check(DecodePackedColorClear64(vk::Format::eR16G16Unorm, 0xffff0000u, 0, clear), "RG16 unorm decodes");
	CheckFloat4(clear, 0.0f, 1.0f, 0.0f, 0.0f, "RG16 unorm channels");
	Check(DecodePackedColorClear64(vk::Format::eR8G8Unorm, 0x0000ff00u, 0, clear), "RG8 unorm decodes");
	CheckFloat4(clear, 0.0f, 1.0f, 0.0f, 0.0f, "RG8 unorm channels");
	// B10G11R11: R and G 11-bit 1.0 = 0x3c0, B 10-bit 1.0 = 0x1e0; R in the low bits.
	const uint32_t ones = 0x3c0u | (0x3c0u << 11u) | (0x1e0u << 22u);
	Check(DecodePackedColorClear64(vk::Format::eB10G11R11UfloatPack32, ones, 0, clear),
	      "R11G11B10F decodes");
	CheckFloat4(clear, 1.0f, 1.0f, 1.0f, 1.0f, "R11G11B10F ones");
	// R = 0.5 (exponent 14), G = 2.0 (exponent 16), B = 0.
	const uint32_t mixed = (14u << 6u) | ((16u << 6u) << 11u);
	Check(DecodePackedColorClear64(vk::Format::eB10G11R11UfloatPack32, mixed, 0, clear),
	      "R11G11B10F mixed decodes");
	CheckFloat4(clear, 0.5f, 2.0f, 0.0f, 1.0f, "R11G11B10F mixed");
}

void TestUnchangedFormats() {
	vk::ClearColorValue clear {};
	Check(DecodePackedColorClear64(vk::Format::eR8G8B8A8Unorm, 0xff0000ffu, 0x12345678u, clear),
	      "RGBA8 still decodes from word0");
	CheckFloat4(clear, 1.0f, 0.0f, 0.0f, 1.0f, "RGBA8 ignores word1");
	Check(!DecodePackedColorClear64(vk::Format::eBc1RgbaUnormBlock, 0, 0, clear),
	      "block-compressed formats stay undecoded");
}

} // namespace

int main() {
	TestRgba16FloatUsesBothWords();
	TestNarrowFormats();
	TestUnchangedFormats();
	std::printf("ClearRegisterDecodeTests: all passed\n");
	return 0;
}
