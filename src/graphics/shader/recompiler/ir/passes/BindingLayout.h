#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// lds_storage: a compute program's LDS lives in the SharedMemory device buffer
// (ShaderComputeInputInfo::lds_storage).
void AllocateBindings(Program& program, uint32_t push_data_start_dword = 0,
                      bool lds_storage = false);

const DescriptorBinding* FindBinding(const BindingLayout& layout, DescriptorBindingKind kind);

// Which shared memories the program's DS instructions access.
struct SharedMemoryUse {
	bool lds = false;
	bool gds = false;
};
[[nodiscard]] SharedMemoryUse CollectSharedMemoryUse(const Program& program);

// A compute program with LDS accesses whose LDS lives in the device buffer: it binds SharedMemory.
[[nodiscard]] bool UsesLdsStorage(const Program& program, bool lds_storage);

// Pixel shaders that sample images record, per GET_LOD_STATS counter, the finest mip level
// sampled and a sample count (KYTY_LOD_STATS_MODE=gpu, the default).
[[nodiscard]] bool UsesMipStats(const Program& program);

// Programs whose BVH node tests are counted (BvhIntersectRay with KYTY_RT_NODE_BUDGET or
// KYTY_RT_NODE_STATS). The count reports through GDS, so such a program binds GDS.
[[nodiscard]] bool UsesBvhNodeCount(const Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_BINDINGLAYOUT_H_ */
