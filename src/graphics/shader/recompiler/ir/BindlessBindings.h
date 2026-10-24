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
// A translation entry for a key whose texture is not resident yet; the shader samples a
// placeholder slot.
inline constexpr uint32_t BindlessPending = 0xffffffffu;
// The placeholder slot pending keys sample (slots 0-2 of each image array are placeholders).
inline constexpr uint32_t BindlessPendingSlot = 2;

// Whether resource tracking may turn an indirect image the enumeration cannot cover into a
// bindless one (instead of skipping the program). The renderer sets it once, before the first
// shader is compiled, when the device supports descriptor indexing with update-after-bind
// sampled images and KYTY_BINDLESS is not 0. Off: programs translate as before.
void SetBindlessImagesEnabled(bool enabled);
[[nodiscard]] bool BindlessImagesEnabled();

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_IR_BINDLESSBINDINGS_H_
