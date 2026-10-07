#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Denormal execution modes the host device can honor (VK_KHR_shader_float_controls, core in
// Vulkan 1.2). Guest shaders run with FLOAT_MODE 0xC0: f32 denormals flushed, f16/f64 kept.
// The device layer computes which modes are legal under its denormBehaviorIndependence.
struct HostFloatControls {
	bool denorm_flush_f32    = false;
	bool denorm_preserve_f16 = false;
	bool denorm_preserve_f64 = false;
};

void              SetHostFloatControls(const HostFloatControls& controls);
HostFloatControls GetHostFloatControls();

// Storage-buffer robustness the device guarantees. With robustBufferAccess2 enabled and
// robustStorageBufferAccessSizeAlignment == 1, a 32-bit load from a storage-buffer descriptor
// returns 0 exactly when some byte of it lies outside the descriptor range, which is the
// shader's own "dword index < OpArrayLength" check. The device layer sets this once.
struct HostBufferRobustness {
	bool storage_dword_loads_return_zero = false;
};

void                 SetHostBufferRobustness(const HostBufferRobustness& robustness);
HostBufferRobustness GetHostBufferRobustness();

// Derivatives in compute shaders (IMAGE_GET_LOD: OpImageQueryLod under DerivativeGroupQuads). The
// device layer enables VK_KHR_compute_shader_derivatives, else VK_NV_compute_shader_derivatives
// (the same capability and execution mode, declared with the NV SPIR-V extension). None: the device
// has neither; such a shader still declares the KHR extension, which the driver may reject, and the
// emitter names it once.
enum class HostComputeDerivatives : uint8_t { Khr, Nv, None };

// Optional image features of the device (set once by the device layer).
struct HostImageFeatures {
	// shaderResourceMinLod: the MinLod image operand, used for IMAGE_SAMPLE*_CL.
	bool                   min_lod             = false;
	HostComputeDerivatives compute_derivatives = HostComputeDerivatives::Khr;
};

void              SetHostImageFeatures(const HostImageFeatures& features);
HostImageFeatures GetHostImageFeatures();

// The clock S_MEMREALTIME reads (VK_KHR_shader_clock, set once by the device layer). S_MEMREALTIME
// is a free-running 64-bit counter at 100 MHz. The emitter reads OpReadClockKHR at this scope and
// shifts the value toward 100 MHz: right by `shift` bits when it is positive, left when negative.
// Device scope (shaderDeviceClock) is the device-wide clock; Subgroup scope (shaderSubgroupClock)
// is the fallback; None (no shader clock) keeps the placeholder UINT64_MAX.
enum class HostClockScope : uint8_t { None, Subgroup, Device };

struct HostShaderClock {
	HostClockScope scope = HostClockScope::None;
	int32_t        shift = 0;
};

void            SetHostShaderClock(const HostShaderClock& clock);
HostShaderClock GetHostShaderClock();

// Defined values for geometry outputs a guest shader can leave unwritten (KYTY_VS_OUTPUT_GUARD and
// KYTY_MESH_OUTPUT_GUARD, auto/on/off, set once by the device layer; auto is on for NVIDIA Turing
// only). A guest export runs under EXEC, so a vertex whose lane skips its position export leaves
// gl_Position (and the zero-position clip guard plane) undefined, and an NGG mesh lane that skips
// its exports leaves its vertex and primitive undefined. Where the hardware happens to read such values as 0 nothing changes:
//   vertex_outputs: vertex/evaluation shaders store position (0,0,0,0) and a clip-guard distance
//     of -1 before the guest code, so an unexported vertex culls its primitives.
//   mesh_outputs: mesh shaders start each lane's position at (0,0,0,0) and primitive as culled,
//     clamp the allocation to the declared OutputVertices/OutputPrimitives, and cull (with index 0)
//     a primitive that names a vertex past the allocated count (undefined in VK_EXT_mesh_shader).
struct HostGeometryGuard {
	bool vertex_outputs = false;
	bool mesh_outputs   = false;
};

void              SetHostGeometryGuard(const HostGeometryGuard& guard);
HostGeometryGuard GetHostGeometryGuard();

// Vulkan reports no rate for the shader clock. The device clock counts at the rate of timestamp
// queries on NVIDIA (1 GHz) and AMD (the 100 MHz reference clock), and Intel's subgroup clock is
// its timestamp counter, so the shift is taken from timestampPeriod: the one that brings the rate
// into [66.7, 133.3) MHz (3 at 1 GHz, 0 at 100 MHz, -2 at 19.2 MHz), between -8 and 8.
[[nodiscard]] int32_t RealtimeClockShift(double timestamp_period_ns);

// mip_stats_records=false emits the plain variant of a GET_LOD_STATS-instrumented pixel shader:
// the same bindings and code, without the per-sample feedback (and so without its storage-buffer
// atomics, which force depth/stencil tests after the shader for a shader that can discard).
std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info, bool mip_stats_records = true);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
