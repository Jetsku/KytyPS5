#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_TEXDELTA_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_TEXDELTA_H_

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <xxhash.h>

// KYTY_DRAW_RUN_TEXDELTA (live, default 0; 1 on, 2 verify; with KYTY_DRAW_RUN=1): a draw whose
// structure key differs from the record's ONLY in the T# and S# words of its stages continues the
// run as a texture-delta continuation. At the Sky Garden about 175 draws per frame, at the pirate
// island about 310 and in Astro's Playroom about 170 differ from the record in nothing else
// (KYTY_CP_COMMIT_STATS class "textures", candidate "key").
//  - the texture-free key (DrawRun::StructureKey: every context register, the user-config registers
//    but the index offset and object id, the programs and stage flags, without the T#/S# words)
//    equals the record's; the rest of the certificate is a continuation's (activity, command
//    buffer, tick, rendering instance, validation, the same programs, slice offset);
//  - neither the record nor the draw writes buffers (no KYTY_DRAW_RUN_WRITES overlap test against
//    textures that are resolved only later);
//  - the draw resolves its textures and samplers, their views and transitions as on the normal path,
//    and pushes all its descriptors; it keeps the record's targets, attachment acquisition and
//    rendering state, and its dynamic state (a function of the same registers and targets);
//  - after the buffer work the attachments must be as the record left them (the textures are this
//    draw's own), and no resolved texture may be an attachment, a storage binding or lie over an
//    attachment's memory; otherwise the draw falls back late, as any continuation;
//  - the record keeps its attachments and takes the draw's texture marks and key.
// verify (2): a would-be texture-delta draw takes the normal path, and what it would have kept is
// compared with what the normal path computed (targets, rendering state, attachment feedback,
// attachment state after the acquisition, dynamic state); "TexDelta verify" lines.
namespace Libs::Graphics::DrawRun::TexDelta {

// The structure key's chaining (DrawRun::StructureKey): the register hash, the stage programs, the
// stage flags and the hash of the T#/S# words. Never 0.
[[nodiscard]] inline uint64_t ChainKey(uint64_t registers, const std::array<const void*, 2>& programs,
                                       uint32_t flags, uint64_t words) {
	uint64_t result = XXH3_64bits_withSeed(programs.data(), sizeof(programs), registers);
	result          = XXH3_64bits_withSeed(&flags, sizeof(flags), result);
	result          = XXH3_64bits_withSeed(&words, sizeof(words), result);
	return result != 0 ? result : 1;
}

// The key without the T#/S# words (a seed of its own, so it never stands for a full key). Never 0.
[[nodiscard]] inline uint64_t TextureFreeKey(uint64_t registers, const std::array<const void*, 2>& programs,
                                             uint32_t flags) {
	constexpr uint64_t kTag = 0x7e47de17a5ull;
	uint64_t           result = XXH3_64bits_withSeed(programs.data(), sizeof(programs), registers ^ kTag);
	result                    = XXH3_64bits_withSeed(&flags, sizeof(flags), result);
	return result != 0 ? result : 1;
}

enum class Decision : uint8_t {
	Off,        // the switch is off, or no record
	KeyMatched, // a plain continuation candidate (not a delta)
	Other,      // the texture-free keys differ too
	Writes,     // the record or the draw writes buffers
	Delta,      // a texture-delta candidate
};

struct Inputs {
	int64_t  mode            = 0; // KYTY_DRAW_RUN_TEXDELTA
	bool     record_valid    = false;
	uint64_t key             = 0;
	uint64_t record_key      = 0;
	uint64_t free_key        = 0;
	uint64_t record_free_key = 0;
	bool     record_writes   = false;
	bool     draw_writes     = false;
};

// DrawRunCandidate's front: whether a draw whose key may differ from the record's is a delta.
[[nodiscard]] inline Decision Decide(const Inputs& in) {
	if (in.mode == 0 || !in.record_valid || in.key == 0) {
		return Decision::Off;
	}
	if (in.key == in.record_key) {
		return Decision::KeyMatched;
	}
	if (in.free_key == 0 || in.free_key != in.record_free_key) {
		return Decision::Other;
	}
	if (in.record_writes || in.draw_writes) {
		return Decision::Writes;
	}
	return Decision::Delta;
}

// Why a delta's resolved textures cannot keep the record's acquisition (0: they can).
enum class TextureVerdict : uint8_t { Ok, Attachment, Storage, OverAttachment };

struct TextureFacts {
	uint64_t id      = 0; // image identity
	bool     storage = false;
	uint64_t address = 0;
	uint64_t size    = 0;
};
struct AttachmentFacts {
	uint64_t id      = 0;
	uint64_t address = 0;
	uint64_t size    = 0;
};

[[nodiscard]] inline TextureVerdict CheckTextures(std::span<const TextureFacts>    textures,
                                                  std::span<const AttachmentFacts> attachments) {
	for (const auto& texture: textures) {
		if (texture.storage) {
			return TextureVerdict::Storage;
		}
		for (const auto& attachment: attachments) {
			if (attachment.id != 0 && texture.id == attachment.id) {
				return TextureVerdict::Attachment;
			}
		}
		for (const auto& attachment: attachments) {
			if (texture.size != 0 && attachment.size != 0 && texture.address < attachment.address + attachment.size &&
			    attachment.address < texture.address + texture.size) {
				return TextureVerdict::OverAttachment;
			}
		}
	}
	return TextureVerdict::Ok;
}

// The record's marks after a delta: the draw's texture marks first, then the record's attachment
// marks (marks[attachments_begin..]) unchanged. Returns the new attachments_begin.
template <typename Mark>
[[nodiscard]] uint32_t ReplaceTextureMarks(std::vector<Mark>& marks, uint32_t attachments_begin,
                                           std::span<const Mark> textures) {
	std::vector<Mark> result;
	result.reserve(textures.size() + (marks.size() - attachments_begin));
	result.insert(result.end(), textures.begin(), textures.end());
	result.insert(result.end(), marks.begin() + attachments_begin, marks.end());
	marks.swap(result);
	return static_cast<uint32_t>(textures.size());
}

} // namespace Libs::Graphics::DrawRun::TexDelta

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_TEXDELTA_H_
