#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CHANGECLASS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CHANGECLASS_H_

#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"

#include <cstdint>

// KYTY_CP_COMMIT_STATS: the change classes of a committed draw against the previous recorded draw of
// the same frame (StartPhases/Steps reasons, ChangeMasks). Header-only for the unit tests.
namespace Libs::Graphics::CommitStats::ChangeClass {

// The actual draw-run decision (a continuation, by the continuation kernel's outcome), and for a
// start the first difference, in this order: first of a frame; pass (rendering instance, command
// buffer or attachments); programs; pipeline with the same programs (render state baked into the
// pipeline, vertex input); textures and/or samplers alone; dynamic state alone; the previous draw
// wrote (shader writes); nothing differs (the run missed). "other": a commit that recorded no draw.
enum Reason : uint32_t {
	ReasonContTaken,
	ReasonContLeft,
	ReasonContDeclined,
	ReasonContNoEntry,
	ReasonFirst,
	ReasonPass,
	ReasonPrograms,
	ReasonPipeState,
	ReasonTextures,
	ReasonSamplers,
	ReasonTexSamp,
	ReasonDynamic,
	ReasonWrites,
	ReasonRunMiss,
	ReasonOther,
	ReasonCount
};
inline constexpr const char* kReasonNames[ReasonCount] = {
    "cont.taken", "cont.left", "cont.declined", "cont.noentry", "first",  "pass",    "programs", "pipestate",
    "textures",   "samplers",  "texsamp",       "dynamic",      "writes", "runmiss", "other"};

// Every difference at once (ChangeMasks).
enum ChangeBit : uint32_t {
	ChangePass      = 1u << 0u,
	ChangePrograms  = 1u << 1u,
	ChangePipeline  = 1u << 2u,
	ChangeTextures  = 1u << 3u,
	ChangeSamplers  = 1u << 4u,
	ChangeDynamic   = 1u << 5u,
	ChangeWrites    = 1u << 6u,
	ChangeFirst     = 1u << 7u,
	ChangeMaskCount = 1u << 8u
};
// pass, programs, pipeline, textures, samplers, dynamic, writes, first
inline constexpr char kChangeLetters[] = "IPLXMDWF";

// previous: the previous recorded draw of the frame (nullptr: none).
[[nodiscard]] inline uint32_t MaskOf(const DrawShape* previous, const DrawShape& b) {
	if (previous == nullptr) {
		return ChangeFirst;
	}
	const auto& a    = *previous;
	uint32_t    mask = 0;
	if (a.rendering_serial == 0 || a.rendering_serial != b.rendering_serial || a.command != b.command ||
	    a.attachments != b.attachments) {
		mask |= ChangePass;
	}
	mask |= a.programs != b.programs ? ChangePrograms : 0u;
	mask |= a.pipeline != b.pipeline ? ChangePipeline : 0u;
	mask |= a.textures != b.textures ? ChangeTextures : 0u;
	mask |= a.samplers != b.samplers ? ChangeSamplers : 0u;
	mask |= a.dynamic != b.dynamic ? ChangeDynamic : 0u;
	mask |= a.shader_writes ? ChangeWrites : 0u;
	return mask;
}

[[nodiscard]] inline Reason ReasonOf(uint32_t mask, const DrawShape& b) {
	if (b.run_continued) {
		switch (b.kernel) {
			case KernelOutcome::Taken: return ReasonContTaken;
			case KernelOutcome::Left: return ReasonContLeft;
			case KernelOutcome::Declined: return ReasonContDeclined;
			default: return ReasonContNoEntry;
		}
	}
	if ((mask & ChangeFirst) != 0) return ReasonFirst;
	if ((mask & ChangePass) != 0) return ReasonPass;
	if ((mask & ChangePrograms) != 0) return ReasonPrograms;
	if ((mask & ChangePipeline) != 0) return ReasonPipeState;
	if ((mask & (ChangeTextures | ChangeSamplers)) == (ChangeTextures | ChangeSamplers)) return ReasonTexSamp;
	if ((mask & ChangeTextures) != 0) return ReasonTextures;
	if ((mask & ChangeSamplers) != 0) return ReasonSamplers;
	if ((mask & ChangeDynamic) != 0) return ReasonDynamic;
	if ((mask & ChangeWrites) != 0) return ReasonWrites;
	return ReasonRunMiss;
}

} // namespace Libs::Graphics::CommitStats::ChangeClass

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_CHANGECLASS_H_
