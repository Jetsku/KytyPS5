#ifndef EMULATOR_SRC_LIBS_AUDIOPROPAGATION_H_
#define EMULATOR_SRC_LIBS_AUDIOPROPAGATION_H_

#include <cstddef>
#include <cstdint>

// libSceAudioPropagation (AudioPropagation_v1) state behind the HLE functions in libAudio.cpp.
//
// The library traces sound paths through the game's rooms, portals and materials and renders each
// source into an ambisonic bed. Kyty does not simulate propagation; the functions keep the objects
// the game creates (system, rooms, portals, materials, sources) as handles and answer the queries
// the way an empty scene would: no rays to trace, no audio paths. A source's render writes its
// input as the direct sound into the omnidirectional (W) channel of the bed, unattenuated and
// without direction, instead of leaving the bed silent.
//
// Signatures and structures were read from Astro Bot's calls (PPSA21567 1.018):
//  - every structure starts with a StructDescriptor {id, size};
//  - attribute lists are arrays of Attribute (32 bytes);
//  - a source's input is attribute 4: a pointer to an InputBuffer (mono float samples);
//  - SourceRender takes RenderParams (descriptor 0x010107d6, 0x30 bytes): the source, the output
//    bed and its size in bytes, and the output format (2: 16 interleaved float channels, ACN
//    order, third-order ambisonics; the game sizes the bed at 64 bytes per input sample).
//
// KYTY_AUDIO_PROPAGATION_RENDER: "direct" (default) writes the direct sound into channel 0,
// "silent" leaves every output bed zeroed (the old stub behaviour, with handles).
namespace Libs::AudioPropagation {

using Handle = uint64_t;

struct StructDescriptor {
	uint32_t id;
	uint32_t reserved;
	uint64_t size;
};
static_assert(sizeof(StructDescriptor) == 16);

struct Attribute {
	uint32_t    id;
	uint32_t    reserved0;
	const void* value;
	uint64_t    size;
	uint32_t    flags;
	uint32_t    reserved1;
};
static_assert(sizeof(Attribute) == 32);

struct InputBuffer {
	const float* samples;
	uint32_t     size_bytes;
	uint32_t     reserved0;
	uint64_t     reserved1;
};
static_assert(sizeof(InputBuffer) == 24);

struct RenderParams {
	StructDescriptor desc;
	Handle           source;
	float*           output;
	uint64_t         output_size_bytes;
	uint32_t         format;
	uint32_t         reserved;
};
static_assert(sizeof(RenderParams) == 48);

constexpr uint32_t SourceAttributeInput = 4;

// Kyty's codes for calls the game never makes (null outputs); the library's own are not known.
constexpr int32_t ErrorInvalidParam = static_cast<int32_t>(0x80EA0001u);
constexpr int32_t ErrorNoPath       = static_cast<int32_t>(0x80EA0002u);

enum class Kind : uint8_t { System, Room, Portal, Material, Source };

enum class RenderMode : uint8_t { Direct, Silent };
[[nodiscard]] RenderMode GetRenderMode();
// Tests.
void SetRenderMode(RenderMode mode);

// A new nonzero handle, registered as a live object of that kind.
[[nodiscard]] Handle Create(Kind kind);
// Forgets the handle (false if it was not live).
bool Destroy(Handle handle);
[[nodiscard]] bool IsLive(Handle handle, Kind kind);
[[nodiscard]] size_t LiveCount();
// Tests: forget every object.
void Reset();

// Source attributes; only the input buffer is kept. Unknown sources are ignored.
void SetSourceAttributes(Handle source, const Attribute* attributes, uint32_t count);

// Renders one source into its output bed (see above). Returns 0, or ErrorInvalidParam.
[[nodiscard]] int32_t Render(const RenderParams& params);

} // namespace Libs::AudioPropagation

#endif // EMULATOR_SRC_LIBS_AUDIOPROPAGATION_H_
