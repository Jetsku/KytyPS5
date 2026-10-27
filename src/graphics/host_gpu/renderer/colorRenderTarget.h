#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// KYTY_TARGET_DESC_MEMO=0 disables the color/depth target description memos (render.h).
[[nodiscard]] bool TargetDescMemoEnabled();
// KYTY_CMASK_FAST_CLEAR=0 ignores CMASK fast clears of colour targets (colorRenderTarget.cpp).
[[nodiscard]] bool CmaskFastClearEnabled();
// KYTY_CMASK_MSAA=1 (default off; upstream fe3857a17, PPSA13794): CMASK fast clears of
// multisampled targets are materialized as well (all samples take the clear colour).
[[nodiscard]] bool CmaskMsaaEnabled();
// KYTY_EQAA_MIXED=1 (default off; usa-ratt/KytyPS5-GT7 18979e65b): EQAA colour targets of any
// coverage/fragment ratio get a host image at the fragment count, and their draws use mixed
// attachment samples (or the single-sample fallback) against the depth target's count.
[[nodiscard]] bool EqaaMixedTargets();

// KYTY_DRAW_SEQUENCE_FAST: a draw reuses what the draws before it resolved when it consumes the
// same state and the reuse is proven exact (colorRenderTarget.cpp). Unset or 1: every part; 0:
// none; otherwise a comma-separated list of parts (targets, textures).
enum class DrawSequencePart : uint32_t {
	// Render/depth target lookups of a slot whose registers did not change
	// (TextureCache::RepeatLookup).
	Targets  = 1u << 0u,
	// A stage's texture bindings when its program and T#/S# words did not change.
	Textures = 1u << 1u,
};
[[nodiscard]] bool DrawSequenceEnabled(DrawSequencePart part);
// KYTY_DRAW_SEQUENCE_VERIFY=1|exit: every reuse is checked against the full path, which then
// provides the result; a difference is counted (DrawSequenceVerifyMismatches; exit stops on the
// first). 0: off.
[[nodiscard]] int  DrawSequenceVerifyMode();
void               ReportDrawSequenceMismatch(const char* what);

// KYTY_RENDER_STATE_FAST: per-draw copies and resets of the draw state that are provably
// redundant are skipped. Unset or 1: every part; 0: none (the full copies and resets as before);
// otherwise a comma-separated list of parts (vertex, swap, reset).
enum class RenderStatePart : uint32_t {
	// ApplyPreparedDraw (renderDraw.cpp) copies a prepared vertex input (about 10 KB) as its used
	// array prefixes and the fields after the arrays. Entries past the counts hold their default
	// values in source and destination alike, so the destination gets the bytes of a full copy.
	VertexCopy = 1u << 0u,
	// ApplyPreparedDraw swaps the stage preparations member by member (vector swaps) instead of
	// through three whole-structure moves.
	PrepSwap = 1u << 1u,
	// The draw state's colour/depth target entries and pixel interface are not value-initialised
	// before each draw: ResolveRenderColorTarget/ResolveRenderDepthTarget assign every field of the
	// entry they resolve (and value-initialise it when there is no target), and every program
	// preparation path assigns the pixel interface.
	Reset = 1u << 2u,
};
[[nodiscard]] bool RenderStateFastEnabled(RenderStatePart part);
// KYTY_RENDER_STATE_VERIFY=1|exit: every skipped copy or reset is checked against the full one,
// which then provides the result: the vertex copy byte for byte, the swap against copies of both
// preparations, a resolved target entry field by field against a resolution into a
// value-initialised entry. A difference is counted (RenderStateVerifyMismatches; exit stops on
// the first). 0: off.
[[nodiscard]] int  RenderStateVerifyMode();
void               ReportRenderStateMismatch(const char* what);

// Every field of a target description, compared one by one (padding excluded).
[[nodiscard]] bool SameImageDesc(const TextureCache::ImageDesc& a, const TextureCache::ImageDesc& b);

struct RenderColorInfo {
	// Discovery keeps guest image information but can remap the view into a larger cache image.
	TextureCache::ImageDesc         desc;
	ImageId                         image_id;
	uint32_t                        target_slot      = 0;
	uint32_t                        guest_mip_level   = 0;
	uint32_t                        guest_array_layer = 0;
	Prospero::ColorComponentMapping export_mapping;

	[[nodiscard]] vk::Extent2D Extent() const {
		return {std::max(desc.info.extent.width >> guest_mip_level, 1u),
		        std::max(desc.info.extent.height >> guest_mip_level, 1u)};
	}
};

// Every field compared one by one (padding excluded).
[[nodiscard]] bool SameRenderColorInfo(const RenderColorInfo& a, const RenderColorInfo& b);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COLORRENDERTARGET_H_
