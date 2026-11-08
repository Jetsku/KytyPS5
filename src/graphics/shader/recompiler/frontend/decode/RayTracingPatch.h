#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_

#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <cstdint>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

// Native ray tracing of guest BVH traversal loops (KYTY_RT_HW, adapted from TheCruZ's
// native-hardware-raytracing branch of KytyPS5-GTA).
//
// Games that use sce::Psr traverse its BVHs in software with a traversal loop compiled into their
// shaders. The patch keeps the guest code that sets up an instance (the object space ray, the
// BLAS descriptors) and the code of a leaf (the triangle test result, the alpha test, the
// closest-hit update), and replaces the stack walks with host ray queries over acceleration
// structures that mirror the guest BVHs (NativeRayTracing): every triangle the hardware finds is a
// candidate, the wave runs the guest instance setup once per instance with the lanes whose
// candidate is in it and then the guest leaf code of each lane's own candidate; a lane whose leaf
// code accepted the hit (its result registers changed) confirms the candidate, which shortens its
// query. The guest code alone decides what is a hit, so the results are the guest's for every
// candidate the hardware reports.
//
// The original loop stays in the shader: RT_INIT falls back to it when the host has no
// acceleration structure for the TLAS the shader traces.
//
// Site kind PsrClosest (Marvel's Spider-Man: Miles Morales, kernels 0x6d9e..., 0x208e...): a
// scalar TLAS walk (the node in an SGPR, a stack in VGPR lanes) whose instance setup is scalar,
// with a per-lane BLAS walk (an 8-entry LDS stack per lane plus a global overflow) whose leaves
// are Psr leaf lists ({AMD triangle node, info word}); the leaf code tests one triangle, runs the
// alpha test (a bindless sample) and keeps the nearest hit.
//
// Pseudo instructions (Instruction::offset = site):
//   RT_INIT    image_nsa_addr[0..7] = origin xyz, direction xyz, t max and cull mask VGPRs
//              (NoRegister: every instance), src0 = SGPR pair of the guest TLAS header address.
//              SCC = native traversal.
//   RT_SELECT  dst = SGPR of the TLAS node; advances the queries of the lanes without a
//              candidate, picks the smallest instance any lane's candidate is in, writes its node
//              and sets EXEC to the lanes whose candidate is in it. SCC = some lane has one.
//   RT_LEAF    dst = VGPR of the BLAS node, dst2 = VGPR of the leaf entry's info word: the lane's
//              candidate; EXEC = the lanes of the instance; image_nsa_addr[0..2] = the result
//              registers (distance, node, instance record low) the leaf code writes on a hit.
//   RT_RESOLVE image_nsa_addr[0..2] as RT_LEAF: lanes whose result changed confirm their
//              candidate; dmask 1: the setup culled the instance (no confirm).
//   RT_EXIT    EXEC = the lanes that started the traversal; ends the native traversal.
//   RT_VERIFY_SAVE  (KYTY_RT_HW_VERIFY) keeps the native results of image_nsa_addr[0..7] and
//              restores their values at RT_INIT, then the guest loop runs again.
//   RT_VERIFY_CHECK compares the guest loop's results with the kept ones (counted per ray).
enum class RayTracingKind : uint32_t {
	PsrClosest,
};

// A register a site does not have.
constexpr uint32_t NoRegister = 0xffffffffu;

struct RayTracingSite {
	RayTracingKind kind = RayTracingKind::PsrClosest;
	// The instruction that falls through to the loop header; the patch enters there.
	uint32_t entry_pc = 0;
	// The loop header (the fallback target, also the target of the stack pop).
	uint32_t loop_pc = 0;
	uint32_t origin[3] {};
	uint32_t direction[3] {};
	uint32_t t_max     = 0;
	uint32_t cull_mask = NoRegister;
	// SGPR pair of the TLAS header address at entry_pc.
	uint32_t header_sgpr = 0;
	// The TLAS node SGPR (the instance setup reads the instance from it) and its root value.
	uint32_t node_sgpr = 0;
	uint32_t root      = 0;
	// The TLAS stack pointer SGPR and its value at the entry (verify mode restores it).
	uint32_t stack_sgpr  = NoRegister;
	uint32_t stack_value = 0;
	// The instance setup [setup_pc, blas_pc) and where it leaves when it culls the instance.
	uint32_t setup_pc     = 0;
	uint32_t cull_exit_pc = 0;
	// The BLAS box/triangle test and its node VGPR, the info word VGPR of the leaf entry.
	uint32_t blas_pc   = 0;
	uint32_t leaf_node = 0;
	uint32_t leaf_word = 0;
	// The leaf code [leaf_pc, leaf_end_pc) of a triangle.
	uint32_t leaf_pc     = 0;
	uint32_t leaf_end_pc = 0;
	// The VGPRs the leaf code writes for an accepted hit: distance (= t max), barycentrics, info
	// word, triangle node, instance record (pair), and the alpha test counter (NoRegister: none).
	uint32_t result_t        = 0;
	uint32_t result_u        = NoRegister;
	uint32_t result_v        = NoRegister;
	uint32_t result_word     = NoRegister;
	uint32_t result_node     = 0;
	uint32_t result_instance = 0;
	uint32_t alpha_counter   = NoRegister;
	// Where the loop exits to.
	uint32_t exit_pc = 0;
};

struct RayTracingPatch {
	std::vector<RayTracingSite> sites;
	// The guest TLAS (see Program::RayTracingSource).
	uint32_t user_sgpr     = 0;
	uint32_t vsharp_offset = 0;
	uint32_t header_offset = 0;
	uint32_t root          = 0;
	// Traversal loops the finder saw but could not match (for the log).
	uint32_t        unmatched = 0;
	std::string     report;
};

// Finds the traversal loops of a decoded program by their structure (see RayTracingPatch.cpp);
// false when the program has none that matches in every detail (it keeps its own traversal).
bool DeriveRayTracingPatch(const Program& program, RayTracingPatch& patch);

// Applies `patch` to a decoded program (and records its TLAS source in
// Program::ray_tracing_source). `verify` adds the per-ray comparison with the guest loop. Returns
// false, leaving the program unchanged, when the program does not match what the patch expects.
bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program, bool verify = false);

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_RAYTRACINGPATCH_H_
