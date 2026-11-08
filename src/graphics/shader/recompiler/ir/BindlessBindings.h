#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Descriptor set 1 of a pipeline that samples bindless images (the host side is BindlessTable).
inline constexpr uint32_t BindlessDescriptorSet = 1;
inline constexpr uint32_t BindlessImages2D      = 0;
inline constexpr uint32_t BindlessImages2DArray = 1;
inline constexpr uint32_t BindlessImagesCube    = 2;
inline constexpr uint32_t BindlessImages3D      = 3;
inline constexpr uint32_t BindlessTranslation   = 4;
inline constexpr uint32_t BindlessFeedback      = 5;
// KYTY_BINDLESS_SAMPLERS (Senaxx's mirrored sampler heaps): samplers mirrored from guest sampler
// heaps (S# records); slot 0 is the stand-in sampler (BindlessDefaultSampler). The binding exists
// only with the switch.
inline constexpr uint32_t BindlessSamplers      = 6;
// The S# of the stand-in sampler: wrap on every axis, trilinear, full LOD range.
inline constexpr uint32_t BindlessDefaultSampler[4] = {
    0x00000000u,                             // wrap on every axis
    0xfffu << 12u,                           // max_lod 255.9
    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
    0x00000000u};
// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): images whose T# the shader
// loads at run time (ImageResource::runtime). The table maps a T# to its translation entry:
// RuntimeImageEntries entries of RuntimeImageEntryWords words, the eight T# dwords, the
// translation index + 1 (0: empty) and the binding; the host fills it. A T# the table lacks is
// requested in the request buffer: word 0 counts requests, a bitset of one bit per table entry
// (its home entry) lets one invocation request each, then RuntimeImageRequests records of the
// eight dwords and the binding + 1. Both bindings exist only with the switch.
inline constexpr uint32_t BindlessRuntimeTable     = 7;
inline constexpr uint32_t BindlessRuntimeRequests  = 8;
inline constexpr uint32_t RuntimeImageEntries      = 32768;
inline constexpr uint32_t RuntimeImageEntryWords   = 12;
inline constexpr uint32_t RuntimeImageProbes       = 4;
inline constexpr uint32_t RuntimeImageRequests     = 1024;
inline constexpr uint32_t RuntimeImageRequestWords = 9;
inline constexpr uint32_t RuntimeImageBitsOffset   = 16; // words
inline constexpr uint32_t RuntimeImageRingOffset   = RuntimeImageBitsOffset + RuntimeImageEntries / 32u;
inline constexpr uint32_t RuntimeImageRequestBufferWords =
    RuntimeImageRingOffset + RuntimeImageRequests * RuntimeImageRequestWords;
// The home entry of a T# in the table, as the shader computes it (FNV-1a over the dwords and the
// binding, then a final mix).
[[nodiscard]] constexpr uint32_t RuntimeImageHash(const uint32_t* dwords, uint32_t binding) {
	uint32_t hash = 0x811c9dc5u;
	for (uint32_t i = 0; i < 8u; i++) {
		hash = (hash ^ dwords[i]) * 0x01000193u;
	}
	hash = (hash ^ binding) * 0x01000193u;
	hash ^= hash >> 15u;
	return hash & (RuntimeImageEntries - 1u);
}
// A translation entry for a key whose texture is not resident yet; the shader samples a
// placeholder slot.
inline constexpr uint32_t BindlessPending = 0xffffffffu;
// The placeholder slot pending keys sample (slots 0-2 of each image array are placeholders).
inline constexpr uint32_t BindlessPendingSlot = 2;
// Feedback buffer: word 0 (an out-of-range key), then one summary word per chunk of
// 2^BindlessFeedbackChunkShift entries that a shader set when it raised a flag in that chunk, then the
// heaps' regions. The host copies and scans only the chunks marked (BindlessTable).
inline constexpr uint32_t BindlessFeedbackSummary    = 1;
inline constexpr uint32_t BindlessFeedbackChunkShift = 10;
inline constexpr uint32_t BindlessFeedbackMaxEntries = 32u << 20u; // the largest translation buffer
inline constexpr uint32_t BindlessFeedbackChunks     = BindlessFeedbackMaxEntries >> BindlessFeedbackChunkShift;
inline constexpr uint32_t BindlessFeedbackFirstRegion = BindlessFeedbackSummary + BindlessFeedbackChunks;
// KYTY_BINDLESS_3D_AS_ARRAY (CodegenOptions::bindless_3d_as_array): in a 3D binding's region, the
// key's T# is a 2D or 2D-array texture, held in the 2D-array binding at the slot in the low bits.
// The hardware addresses by the T# type, not the instruction's dimension, so a 3D sample of it
// reads layer round(z). Pending (all bits set) is told apart by the bit above.
inline constexpr uint32_t BindlessSlotArrayFlag = 0x40000000u;
inline constexpr uint32_t BindlessSlotFlagMask  = 0xc0000000u;

// Whether resource tracking may turn an indirect image the enumeration cannot cover into a
// bindless one (instead of skipping the program). The renderer sets it once, before the first
// shader is compiled, when the device supports descriptor indexing with update-after-bind
// sampled images and KYTY_BINDLESS is not 0. Off: programs translate as before.
void SetBindlessImagesEnabled(bool enabled);
[[nodiscard]] bool BindlessImagesEnabled();

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
