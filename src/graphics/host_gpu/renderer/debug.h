#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <string>

namespace Libs::Graphics {

class CommandBuffer;

namespace HW {
class Context;
class UserConfig;
struct RenderTarget;
struct ScanModeControl;
struct ScreenViewport;
} // namespace HW

struct ScissorRect {
	int left   = 0;
	int top    = 0;
	int right  = 0;
	int bottom = 0;
};

uint32_t                 render_target_mask_slot(uint32_t mask, uint32_t slot);
// KYTY_SKIP_INACTIVE_PS=1 (default off; upstream 6956b454f): a colour target counts as written
// only when the pixel shader's export format for it (SPI_SHADER_COL_FORMAT) is not ZERO, so a
// pixel shader that writes no such target and has no depth or coverage side effects does not run.
[[nodiscard]] bool       SkipInactivePixelShadersEnabled();
// Slots written with KYTY_SKIP_INACTIVE_PS: CB_TARGET_MASK & CB_SHADER_MASK nonzero and a nonzero
// export format. 0xff without the flag (no slot excluded). Any thread.
[[nodiscard]] uint32_t   DrawColorOutputFilter(const HW::Context& ctx);
uint32_t                 render_target_first_bound_slot(const CommandBuffer& buffer);
bool                     graphics_debug_dump_enabled();
void                     uc_print(const char* func, const HW::UserConfig& uc);
void                     uc_check(const HW::UserConfig& uc);
std::string              rt_print(const char* func, const HW::RenderTarget& rt);
bool                     RenderIsColorTileModeLinear(Prospero::TileMode tile_mode);
void                     hw_print(const CommandBuffer& buffer);
void                     hw_check(const CommandBuffer& buffer);
void                     LogDrawPhase(const char* draw_name, const char* phase);
ScissorRect calc_final_scissor(const HW::ScreenViewport& vp, const HW::ScanModeControl& smc,
                               vk::Extent2D extent, uint32_t viewport_index);
// calc_final_scissor before its clamp to the framebuffer, without its log: false for a clip-rect
// rule it does not support (the scissor is then left as the other rectangles make it, which
// calc_final_scissor reports). Any thread.
[[nodiscard]] bool calc_scissor_unclamped(const HW::ScreenViewport& vp,
                                          const HW::ScanModeControl& smc, uint32_t viewport_index,
                                          ScissorRect& scissor);
// calc_final_scissor's clamp of an unclamped scissor to the framebuffer.
[[nodiscard]] ScissorRect clamp_scissor(const ScissorRect& scissor, vk::Extent2D extent);
// Draw-prep binding plans (KYTY_DRAW_PREP_BINDINGS hwcheck): whether uc_check and hw_check would
// neither stop the emulator nor log for these registers (their log-once flags and counters only
// move one way, so true stays true). Any thread.
[[nodiscard]] bool hw_checks_quiet(const HW::Context& hw, const HW::UserConfig& uc);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DEBUG_H_
