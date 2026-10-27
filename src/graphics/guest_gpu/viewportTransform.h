#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_VIEWPORTTRANSFORM_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_VIEWPORTTRANSFORM_H_

#include <cstdint>

// KYTY_VTE_CNTL=1 (default off; the GT7 preset; masterzhang233/KytyPS5-GT7 df7fc6108, adapted).
// PA_CL_VTE_CNTL: bits 0..5 enable the viewport's X/Y/Z scale and offset each; VTX_XY_FMT and
// VTX_Z_FMT (bits 8, 9) mean the position's X/Y and Z are already divided by W, VTX_W0_FMT
// (bit 10) that W is the incoming W (otherwise its reciprocal). The default 0x43f is a plain clip
// position with every scale and offset on, which is all the renderer modelled before.
namespace Libs::Graphics::ViewportTransform {

inline constexpr uint32_t DefaultControl     = 0x43f;
inline constexpr uint32_t PositionFormatMask = 0x700;
inline constexpr uint32_t PredividedXY       = 0x100;
inline constexpr uint32_t PredividedZ        = 0x200;
inline constexpr uint32_t IncomingW          = 0x400;

template <unsigned Axis>
constexpr float Scale(uint32_t control, float value) {
	static_assert(Axis < 3);
	return (control & (1u << (Axis * 2u))) != 0 ? value : 1.0f;
}

template <unsigned Axis>
constexpr float Offset(uint32_t control, float value) {
	static_assert(Axis < 3);
	return (control & (2u << (Axis * 2u))) != 0 ? value : 0.0f;
}

// ShaderClipSpaceTransform::vte_format: the three position-format bits, with VTX_W0_FMT
// inverted so that the default (incoming W, nothing divided) is 0.
constexpr uint8_t PackPositionFormat(uint32_t control) {
	return static_cast<uint8_t>((((control & PositionFormatMask) ^ IncomingW) >> 8u) & 7u);
}

constexpr uint32_t UnpackPositionFormat(uint8_t packed) {
	return ((static_cast<uint32_t>(packed) & 7u) << 8u) ^ IncomingW;
}

} // namespace Libs::Graphics::ViewportTransform

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_VIEWPORTTRANSFORM_H_
