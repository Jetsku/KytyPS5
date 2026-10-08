#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects immutable resource topology from typed SSA handles, interns their resolved dwords in
// descriptor_sources, then writes dense indices to handle flags and MemoryInfo.
// indirect_scalar_buffers (KYTY_SRT_VARIANT_READS): an S_BUFFER_LOAD whose V# is not a valid
// runtime value reads through BDA (ResourceKind::IndirectBuffer) instead of failing, and any other
// descriptor that is not a valid runtime value makes the result the PC of its first use instead of
// failing; the caller then drops the program. Returns UINT32_MAX when every descriptor resolved.
// bda_writes (KYTY_BDA_WRITES): a raw store or atomic through such a V# in a compute program
// writes through BDA as well (ShaderInfo::bda_writes).
uint32_t TrackResources(Program& program, bool indirect_scalar_buffers = false,
                        bool bda_writes = false);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCETRACKING_H_ */
