#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <span>
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

// Emitter statistics since process start (offline comparisons, --journal-compile): shader-side
// f32 denormal flushes emitted by stage (0 pixel, 1 compute, 2 geometry), and non-compute
// programs with function-scope LDS by KYTY_LDS_SLOTS verdict (0 slots, 1 array kept).
uint64_t EmittedDenormFlushes(uint32_t stage_index);
uint64_t LdsSlotVerdicts(uint32_t verdict);

// Storage-buffer robustness the device guarantees. With robustBufferAccess2 enabled and
// robustStorageBufferAccessSizeAlignment == 1, a 32-bit load from a storage-buffer descriptor
// returns 0 exactly when some byte of it lies outside the descriptor range, which is the
// shader's own "dword index < OpArrayLength" check. The device layer sets this once.
struct HostBufferRobustness {
	bool storage_dword_loads_return_zero = false;
	// robustBufferAccess2 with robustUniformBufferAccessSizeAlignment <= 16 and
	// uniformBufferStandardLayout: a dword uniform load past the range rounded up to that alignment
	// returns 0,
	// so constant-only buffers can bind as uniform buffers (KYTY_UNIFORM_BUFFERS).
	bool uniform_dword_loads_return_zero = false;
};

void                 SetHostBufferRobustness(const HostBufferRobustness& robustness);
HostBufferRobustness GetHostBufferRobustness();

// KYTY_SRT_UNIFORM (CodegenOptions::srt_uniform) on a device with the uniform dword-load path: the
// flattened SRT binding is a uniform buffer (emitter and descriptor layout agree through this).
bool SrtUniformEnabled();

// KYTY_BUFFER_PAIR_ATOMIC: the device has 64-bit atomics on storage buffers and physical storage
// buffers (shaderBufferInt64Atomics) and starts every storage-buffer binding 8-byte aligned
// (minStorageBufferOffsetAlignment a multiple of 8), so the u64 view's element i of a binding is the
// guest qword at binding byte 8i. Off: DWORDX2 pairs through bound V#s keep the dword path (pairs
// through BDA check the host address itself). The device layer sets this once.
void SetHostBufferPairAtomics(bool supported);
bool GetHostBufferPairAtomics();
// KYTY_MESH_LAYER=drop|amd (default keep; GT7 fork 67d59fe29, issue 21 there): AMD's Windows driver
// crashed making every pipeline whose mesh shader writes the per-primitive Layer output (RX 6950 XT,
// drivers 26.5.1 and 26.10.1). With drop (on any GPU) or amd (on an AMD GPU) mesh shaders declare no
// Layer output: a draw into a target of one layer draws the same, a layered target gets everything
// in its first layer. The device layer sets this once, before any shader is compiled.
void SetHostMeshLayerDrop(bool drop);
bool GetHostMeshLayerDrop();
// Coherent (GLC/DLC) buffer loads (GT7 fork 171338e72 + f9586d18c). Under the GLSL450 memory model a
// Volatile load from a Coherent buffer is visible only together with barriers: NVIDIA (an RTX 3090)
// reloads anyway, AMD's drivers keep serving the cached value (a workgroup polling for its
// predecessor's published value never sees it), and an RTX 5090 did too (GT7 ray tracing mode's car
// interiors blotchy, the wheels glowing white). How such a load is made:
//   Plain   a volatile load (the default; nothing changes)
//   Barrier a volatile load and a device-scope acquire-release barrier after it (costly on the GPU)
//   Acquire one OpAtomicLoad at device scope with acquire semantics
//   Relaxed one OpAtomicLoad at device scope, unordered
// KYTY_COHERENT_LOAD_MODE=plain|barrier|acquire|relaxed picks one; without it
// KYTY_COHERENT_LOAD_ACQUIRE=1 gives Barrier, =auto Barrier on every vendor but NVIDIA (the fork's
// default), else Plain. Scalar coherent loads take the mode too unless it is Plain.
enum class CoherentLoadMode : uint32_t { Plain = 0, Barrier = 1, Acquire = 2, Relaxed = 3 };
// The device layer: whether the device's vendor wants the barrier (KYTY_COHERENT_LOAD_ACQUIRE=auto).
void                      SetCoherentLoadVendorDefault(bool barrier);
[[nodiscard]] CoherentLoadMode GetCoherentLoadMode();
[[nodiscard]] const char* CoherentLoadModeName(CoherentLoadMode mode);
// KYTY_INDIRECT_FORMAT_COUNT (CodegenOptions.h): device address of the per-format counters of the
// DontInline full formatted-load decoder (0: off). The buffer cache sets it once.
void     SetIndirectFormatCountAddress(uint64_t address);
uint64_t IndirectFormatCountAddress();

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

// KYTY_POPS_INTERLOCK: the device enabled VK_EXT_fragment_shader_interlock with
// fragmentShaderPixelInterlock (set once by the device layer; false until then).
void SetHostFragmentInterlock(bool supported);
bool GetHostFragmentInterlock();
// KYTY_IMPLICIT_LOD_STAGES: the device enabled computeDerivativeGroupLinear (with the compute
// derivatives extension of HostImageFeatures::compute_derivatives).
void SetHostComputeDerivativesLinear(bool supported);
bool GetHostComputeDerivativesLinear();

// The clock S_MEMREALTIME reads (VK_KHR_shader_clock, set once by the device layer). S_MEMREALTIME
// is a free-running 64-bit counter at 100 MHz. The emitter reads OpReadClockKHR at this scope and
// shifts the value toward 100 MHz: right by `shift` bits when it is positive, left when negative.
// Device scope (shaderDeviceClock) is the device-wide clock; Subgroup scope (shaderSubgroupClock)
// is the fallback; None (no shader clock) keeps the placeholder UINT64_MAX.
enum class HostClockScope : uint8_t { None, Subgroup, Device };

struct HostShaderClock {
	HostClockScope scope = HostClockScope::None;
	int32_t        shift = 0;
	// > 1: divide the clock by this instead of shifting (shift is then 0). See RealtimeClockDivisor.
	uint32_t divisor = 1;
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

// Graphics stages whose host subgroups can be wider than a wave32 program's wave (set once by the
// device layer). The AMD proprietary driver runs vertex, mesh and pixel shaders on 64-wide subgroups
// and lets only compute pipelines require 32 (requiredSubgroupSizeStages = COMPUTE), so one host
// subgroup holds two guest waves; the emitter then keeps each wave to its own 32 lanes
// (KYTY_WAVE32_CLUSTERS, CodegenOptions).
struct HostWaveClusters {
	bool vertex = false; // VS, LS and DS programs (never given a required subgroup size)
	bool pixel  = false;
	bool mesh   = false;
};

void             SetHostWaveClusters(const HostWaveClusters& clusters);
HostWaveClusters GetHostWaveClusters();
// Whether this program keeps each wave32 guest wave to its own 32 lanes of the host subgroup.
[[nodiscard]] bool WaveClustersActive(const IR::Program& program);

// Vulkan reports no rate for the shader clock. The device clock counts at the rate of timestamp
// queries on NVIDIA (1 GHz) and AMD (the 100 MHz reference clock), and Intel's subgroup clock is
// its timestamp counter, so the shift is taken from timestampPeriod: the one that brings the rate
// into [66.7, 133.3) MHz (3 at 1 GHz, 0 at 100 MHz, -2 at 19.2 MHz), between -8 and 8.
[[nodiscard]] int32_t RealtimeClockShift(double timestamp_period_ns);
// KYTY_MEMREALTIME_EXACT=1 (default off; TheCruZ/KytyPS5-GTA dcd6c059): the whole divisor nearest to
// rate / 100 MHz (10 at 1 GHz, 8 at 800 MHz, 1 at 100 MHz and slower), 1..1024. The shift above
// makes a 1 GHz clock 125 MHz (25% fast): guest timeouts and frame pacers that compare counter
// differences with intervals in 100 MHz ticks expire early.
[[nodiscard]] constexpr uint32_t RealtimeClockDivisor(uint32_t period_ps) {
	if (period_ps == 0u) {
		return 1u;
	}
	const uint32_t divisor = (10000u + period_ps / 2u) / period_ps;
	return divisor < 1u ? 1u : (divisor > 1024u ? 1024u : divisor);
}
// The clock setting for a device clock with this timestamp period: the shift (RealtimeClockShift),
// or with exact = true and a clock faster than 100 MHz the divisor (a power of two stays a shift).
[[nodiscard]] HostShaderClock RealtimeClockScale(double timestamp_period_ns, bool exact);

// mip_stats_records=false emits the plain variant of a GET_LOD_STATS-instrumented pixel shader:
// the same bindings and code, without the per-sample feedback (and so without its storage-buffer
// atomics, which force depth/stencil tests after the shader for a shader that can discard).
std::vector<uint32_t> EmitProgram(const IR::Program& program,
                                  ShaderStageInputInfo input_info, bool mip_stats_records = true);

// KYTY_HUGE_HELPER_OUTLINE: in a module of at least 100,000 words, each of `helpers` (function
// ids, 0 = none, at most 4) with at least 16 call sites gets the DontInline function control.
void OutlineHugeModuleHelpers(std::vector<uint32_t>& words, std::span<const uint32_t> helpers);

// Why a mesh program cannot run in passes (ShaderMeshInputInfo::passes), or nullptr.
[[nodiscard]] const char* MeshPassesUnsupported(const IR::Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
