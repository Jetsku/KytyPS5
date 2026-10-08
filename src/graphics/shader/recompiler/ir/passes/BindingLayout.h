#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

// Pixel shaders that sample images record, per GET_LOD_STATS counter, the finest mip level
// sampled and a sample count (KYTY_LOD_STATS_MODE=gpu, the default).
[[nodiscard]] bool UsesMipStats(const Program& program);

// Programs whose BVH node tests are counted (BvhIntersectRay with KYTY_RT_NODE_BUDGET or
// KYTY_RT_NODE_STATS). The count reports through GDS, so such a program binds GDS.
[[nodiscard]] bool UsesBvhNodeCount(const Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
