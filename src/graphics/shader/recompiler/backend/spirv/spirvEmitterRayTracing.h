#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVEMITTERRAYTRACING_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BACKEND_SPIRV_SPIRVEMITTERRAYTRACING_H_

#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

// Operands of one IMAGE_BVH*_INTERSECT_RAY node test as SPIR-V ids: the four raw T# dwords and
// the node pointer as u32, the ray as f32 (A16 operands already widened).
struct BvhOperands {
	std::array<uint32_t, 4> tsharp {};
	uint32_t                node_lo = 0;
	uint32_t                node_hi = 0; // 0 for IMAGE_BVH_INTERSECT_RAY
	uint32_t                extent  = 0;
	std::array<uint32_t, 3> origin {};
	std::array<uint32_t, 3> direction {};
	std::array<uint32_t, 3> inverse {};
};

// The instruction's node test for one lane (KYTY_RT_SOFTWARE semantics: every node type, the T#
// size check, the BDA node fetch), returning its four result dwords as a U32x4 id. No EXEC test:
// the caller branches around it for inactive lanes. The program must have info.uses_dma set.
// With a synthetic T# (triangle_return_mode 1, sorting on), a triangle pointer returns {t_num,
// t_denom, I_num, J_num} and a box pointer the sorted hit children, exactly as the instruction.
uint32_t EmitBvhNodeTest(ValueEmitContext& ctx, const BvhOperands& operands);

// KYTY_RT_FUNCTION (default 1): defines the module's single software BVH node test function
// (state.bvh_node_test_function): the node test of EmitBvhNodeTest with its operands as parameters,
// called by every IMAGE_BVH*_INTERSECT_RAY site instead of being inlined at each. A shader with n
// sites then carries one copy of the 300k-word traversal helpers instead of n. Does nothing for a
// program without software BVH instructions, or with KYTY_RT_FUNCTION=0 (sites inline the test).
// Emitted before the guest function (like get_bda_pointer); needs info.uses_dma.
void DefineBvhNodeTestFunction(EmitterState& state);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

#endif
