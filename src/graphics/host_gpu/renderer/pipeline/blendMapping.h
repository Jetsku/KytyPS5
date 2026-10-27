#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_

#include <cstdint>

namespace Libs::Graphics {

namespace HW {
struct BlendControl;
}
namespace Prospero {
struct ColorComponentMapping;
}

enum class BlendMappingSupport {
	Direct,
	SourceAlpha, // Requires logical alpha in the second blend source.
	SourceAlphaOne, // Source alpha for RGB, one for alpha, with a shared destination equation.
	SourceAlphaZero, // Source alpha for RGB, zero for alpha, with a shared destination equation.
	OneAlphaZero,    // One for RGB, zero for alpha, a source-alpha destination shared by both.
	Unsupported,
};

// KYTY_UI_TEXT_BLEND=1 (default off; usa-ratt/KytyPS5-GT7 429692dd3, from Franseven): with
// KYTY_BLEND_ALPHA_REMAP, colour One and alpha Zero over a source-alpha destination factor (UI
// text into a target whose export mapping moves alpha) is blended through the second source
// (OneAlphaZero), and exports to render targets that are not bound no longer block taking MRT1.
[[nodiscard]] bool UiTextBlendEnabled();

bool                BlendFactorIsDualSource(uint8_t factor);
BlendMappingSupport ClassifyBlendMapping(const HW::BlendControl&                blend,
                                         const Prospero::ColorComponentMapping& mapping,
                                         bool one_alpha_zero = UiTextBlendEnabled());

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BLENDMAPPING_H_
