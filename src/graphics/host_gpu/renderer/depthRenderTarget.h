#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

namespace HW {
class Context;
} // namespace HW

// Without a stencil plane, Hi-Stencil fields are inactive. An active plane is compatible when
// Hi-Stencil is disabled or HTile backing is present.
inline constexpr bool depth_htile_stencil_acceleration_compatible(bool has_stencil, bool has_htile,
                                                                  bool htile_stencil_disabled) {
	return !has_stencil || htile_stencil_disabled || has_htile;
}

// Whether a depth-bounds test can reject a sample. A depth attachment stores values in [0, 1]:
// UNORM formats by definition, float formats unless VK_EXT_depth_range_unrestricted is enabled.
// Bounds that contain [0, 1] therefore reject nothing, and the test is left out. Astro Bot enables
// it with bounds [0, 1] on its 3D draws; GPUs without the depthBounds feature (Apple M1-M4) need
// that to run them.
inline bool depth_bounds_test_effective(bool enable, float min_bounds, float max_bounds,
                                        vk::Format format, bool depth_range_unrestricted) {
	const bool float_depth =
	    format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint;
	const bool unit_range = !float_depth || !depth_range_unrestricted;
	return enable && !(unit_range && min_bounds <= 0.0f && max_bounds >= 1.0f);
}

struct RenderDepthInfo {
	// Discovery keeps guest image information but can remap the view into a larger cache image.
	TextureCache::ImageDesc     desc;
	bool                        depth_clear_enable       = false;
	vk::ImageAspectFlags        load_clear_aspects;
	float                       depth_clear_value        = 0.0f;
	bool                        depth_test_enable        = false;
	// Effective draw writes; discovery applies test, target-write and clear controls.
	bool                        depth_write_enable       = false;
	vk::CompareOp               depth_compare_op         = vk::CompareOp::eNever;
	bool                        depth_bounds_test_enable = false;
	float                       depth_min_bounds         = 0.0f;
	float                       depth_max_bounds         = 0.0f;
	bool                        stencil_clear_enable     = false;
	uint8_t                     stencil_clear_value      = 0;
	bool                        stencil_test_enable      = false;
	vk::StencilOpState          stencil_front;
	vk::StencilOpState          stencil_back;
	ImageId                     image_id;

	[[nodiscard]] vk::ImageAspectFlags AttachmentWriteAspects() const;
};

// Every field compared one by one (floats by their bits, padding excluded).
[[nodiscard]] bool SameRenderDepthInfo(const RenderDepthInfo& a, const RenderDepthInfo& b);

// Draw-prep binding plans (KYTY_DRAW_PREP_BINDINGS): what ResolveRenderDepthTarget gives a draw
// with these registers, as far as the graphics pipeline key uses it, assuming the target's image
// is found: whether there is a depth target, its view format and sample count, and the
// depth-bounds state (RenderDepthInfo's defaults without a target). False where that resolution
// would stop the emulator for a reason seen here. Any thread.
struct DepthTargetPrediction {
	bool       with_depth         = false;
	vk::Format format             = vk::Format::eUndefined;
	uint32_t   samples            = 0;
	bool       bounds_test_enable = false;
	float      min_bounds         = 0.0f;
	float      max_bounds         = 0.0f;
};
[[nodiscard]] bool PredictRenderDepthTarget(const GraphicContext& graphics, const HW::Context& hw,
                                            DepthTargetPrediction& prediction);

inline vk::ImageAspectFlags DepthFeedbackAspects(vk::ImageAspectFlags draw_writes,
                                                 const ImageViewInfo& target,
                                                 const ImageViewInfo& sampled) {
	if (!ImageRangeOverlaps(target.base_level, target.level_count, sampled.base_level,
	                        sampled.level_count) ||
	    !ImageRangeOverlaps(target.base_layer, target.layer_count, sampled.base_layer,
	                        sampled.layer_count)) {
		return {};
	}
	return draw_writes & sampled.aspect;
}

inline vk::ImageAspectFlags DepthReadableAspects(vk::ImageLayout layout) {
	switch (layout) {
		case vk::ImageLayout::eDepthReadOnlyOptimal:
		case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal:
			return vk::ImageAspectFlagBits::eDepth;
		case vk::ImageLayout::eStencilReadOnlyOptimal:
		case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal:
			return vk::ImageAspectFlagBits::eStencil;
		case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
		case vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT:
		case vk::ImageLayout::eGeneral:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default:
			return {};
	}
}

// Aspects a draw may write through a depth/stencil attachment in `layout` (no attachment layout
// restricts depth/stencil tests).
inline vk::ImageAspectFlags DepthWritableAspects(vk::ImageLayout layout) {
	switch (layout) {
		case vk::ImageLayout::eDepthStencilAttachmentOptimal:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal:
		case vk::ImageLayout::eDepthAttachmentOptimal: return vk::ImageAspectFlagBits::eDepth;
		case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal:
		case vk::ImageLayout::eStencilAttachmentOptimal: return vk::ImageAspectFlagBits::eStencil;
		default: return {};
	}
}

// KYTY_DEPTH_LAYOUT_STABLE: the attachment layout of a depth target that the draw does not
// sample. depth_attachment_layout() picks the narrowest layout for each draw's writes, so draws
// that alternate stencil writes (a stencil mark followed by draws that only test it) toggled the
// image between DEPTH_STENCIL_ATTACHMENT and DEPTH_ATTACHMENT_STENCIL_READ_ONLY: a layout
// transition, and so a new rendering instance, before every such draw. The layout of an
// attachment nothing samples is not observable by the draw: every standard depth/stencil
// attachment layout allows the tests, and a writable one also the writes. So the image keeps its
// current layout when that is a standard attachment layout for its aspects that allows this
// draw's writes (`current_whole`: one tracked state covers the image), and otherwise takes the
// fully writable layout, which every later draw can keep.
inline vk::ImageLayout depth_stable_attachment_layout(vk::ImageLayout current, bool current_whole,
                                                      vk::ImageAspectFlags available,
                                                      vk::ImageAspectFlags writes) {
	const bool has_depth   = static_cast<bool>(available & vk::ImageAspectFlagBits::eDepth);
	const bool has_stencil = static_cast<bool>(available & vk::ImageAspectFlagBits::eStencil);
	bool       standard    = false;
	switch (current) {
		case vk::ImageLayout::eDepthStencilAttachmentOptimal:
		case vk::ImageLayout::eDepthStencilReadOnlyOptimal: standard = true; break;
		case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal:
		case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal:
			standard = has_depth && has_stencil;
			break;
		case vk::ImageLayout::eDepthAttachmentOptimal:
		case vk::ImageLayout::eDepthReadOnlyOptimal: standard = has_depth && !has_stencil; break;
		case vk::ImageLayout::eStencilAttachmentOptimal:
		case vk::ImageLayout::eStencilReadOnlyOptimal: standard = has_stencil && !has_depth; break;
		default: break;
	}
	if (current_whole && standard && !(writes & ~DepthWritableAspects(current))) {
		return current;
	}
	if (!has_stencil) {
		return vk::ImageLayout::eDepthAttachmentOptimal;
	}
	if (!has_depth) {
		return vk::ImageLayout::eStencilAttachmentOptimal;
	}
	return vk::ImageLayout::eDepthStencilAttachmentOptimal;
}

inline vk::ImageLayout depth_attachment_layout(const RenderDepthInfo& depth) {
	const auto available     = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
	const auto writes        = depth.AttachmentWriteAspects();
	const bool has_depth     = static_cast<bool>(available & vk::ImageAspectFlagBits::eDepth);
	const bool has_stencil   = static_cast<bool>(available & vk::ImageAspectFlagBits::eStencil);
	// The attachment layout must permit load clears as well as draw writes.
	const bool depth_write   = static_cast<bool>(writes & vk::ImageAspectFlagBits::eDepth);
	const bool stencil_write = static_cast<bool>(writes & vk::ImageAspectFlagBits::eStencil);
	if (!has_stencil) {
		return depth_write ? vk::ImageLayout::eDepthAttachmentOptimal
		                   : vk::ImageLayout::eDepthReadOnlyOptimal;
	}
	if (!has_depth) {
		return stencil_write ? vk::ImageLayout::eStencilAttachmentOptimal
		                     : vk::ImageLayout::eStencilReadOnlyOptimal;
	}
	if (depth_write && stencil_write) {
		return vk::ImageLayout::eDepthStencilAttachmentOptimal;
	}
	if (depth_write) {
		return vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal;
	}
	if (stencil_write) {
		return vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal;
	}
	return vk::ImageLayout::eDepthStencilReadOnlyOptimal;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEPTHRENDERTARGET_H_
