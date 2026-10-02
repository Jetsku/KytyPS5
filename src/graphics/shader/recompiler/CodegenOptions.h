#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

// How V_MAD_F32/V_MAC_F32/V_MADMK_F32/V_MADAK_F32 (unfused on PS5: the product is rounded before
// the add) are emitted, and which float arithmetic may not be contracted by the host compiler.
enum class MadMode : uint8_t {
	// Every MAD is an FMul plus an FAdd, and every guest FMul/FAdd/FSub is NoContraction: bit
	// exact everywhere, at the cost of one extra instruction per MAD.
	Exact,
	// Exact on the data flow that feeds position exports (plus Invariant on the position built-in),
	// fused FMA elsewhere: positions computed by different shaders (depth pre-pass and main pass)
	// match bit for bit, while pixel-shader math keeps the cheaper FMA.
	Position,
	// Every MAD is a fused FMA and the host may contract freely (the behaviour before MadMode).
	Fused,
};

// Which pixel shaders start with EXEC holding only the non-helper invocations (KYTY_PS_LIVE_EXEC).
enum class PsLiveExec : uint8_t {
	// Every invocation starts with its EXEC bit set, helper invocations included.
	Off,
	// Pixel shaders that contain DS_APPEND or DS_CONSUME.
	AppendConsume,
	// Every pixel shader.
	All,
};

// Switches for code-generation changes that must stay revertible at runtime. Every field is read
// from its environment variable once (first use); tests may replace the whole set. Programs are
// cached in memory only and the driver pipeline cache is keyed by the SPIR-V code, so changing a
// switch between runs needs no cache invalidation.
struct CodegenOptions {
	// KYTY_MOVREL_RANGE=0: keep V_MOVRELS/V_MOVRELD select chains over every VGPR above the base
	// instead of folding the compares that the M0 value set proves false.
	bool movrel_range = true;
	// KYTY_MOVREL_KNOWN_ZEROS=1 (default off): with movrel_range, a V_MOVRELS/V_MOVRELD compare of M0
	// against a register offset that sets a bit M0 never has also folds when M0's value set is
	// unknown. Known zero bits pass through shifts by constants, masks, sums, products, bit-field
	// extracts, selects, phis and lane reads: M0 = loop counter << 2 (Astro Bot's foliage vertex
	// shaders) leaves every fourth register of each chain. Exact.
	bool movrel_known_zeros = false;
	// KYTY_MOVREL_SWITCH=1 (default off): a V_MOVRELS select chain of 16 or more links is emitted as
	// an OpSwitch on M0 / 16 whose cases hold the select chains of their 16 index values. M0 is
	// uniform, so the host runs one short chain instead of a select per register. Exact.
	bool movrel_switch = false;
	// KYTY_UNIFORM_LANE_READS=1 (default off): the shuffle result of V_READFIRSTLANE/V_READLANE
	// (every lane already holds the same value) passes through OpGroupNonUniformBroadcastFirst,
	// which returns it unchanged but lets the host compiler treat it and everything derived from
	// it (a waterfall loop's key, addresses, loop exits) as uniform. Exact.
	bool uniform_lane_reads = false;
	// KYTY_SHORT_F32_HELPERS=1 (default off): NaN tests are one OpIsNan instead of exponent and
	// mantissa tests (float-to-int conversions, the legacy min/max/med3, float atomics). Exact.
	// (Bryan's companion change, dropping the sin/cos |x| >= 2^23 select, is not exact on the RTX
	// 3090 and is not applied.)
	bool short_f32_helpers = false;
	// KYTY_FAST_FMINMAX=0: emulate f32 min/max/min3/max3/med3 with two bit classifications per
	// min/max instead of one compare-and-select plus a single two-zeros test.
	bool fast_float_min_max = true;
	// KYTY_FAST_PKRTZ=0: convert V_CVT_PKRTZ_F16_F32 halves with the original select chain instead
	// of the shorter integer formulation.
	bool fast_pkrtz = true;
	// KYTY_SINGLE_F2I_SATURATION=0: repeat the float-to-int NaN/range saturation in the emitter even
	// though the translator's saturated conversion already guarantees an in-range operand.
	bool single_f2i_saturation = true;
	// KYTY_LOD_STATS_GATE=0: record GET_LOD_STATS feedback for every sample, including images whose
	// T# has no mip-statistics counter, and issue the finest-level AtomicUMin unconditionally.
	bool lod_stats_gate = true;
	// KYTY_ROBUST_BUFFER_LOADS=0: bounds-check every plain dword storage-buffer load in the shader
	// even when the device's robustBufferAccess2 already returns zero for out-of-range dwords.
	bool robust_buffer_loads = true;
	// KYTY_MAD_MODE=exact|position|fused, see MadMode.
	MadMode mad_mode = MadMode::Position;
	// KYTY_INTERP_MODES=0: interpolate every pixel input at the pixel center with the shader-wide
	// perspective (NoPerspective on all inputs when LINEAR_CENTER is enabled) instead of per input
	// from the I/J pair its V_INTERP_P2 reads use (centroid, sample, linear).
	bool interp_modes = true;
	// KYTY_SAMPLE_OFFSETS=0: ignore the texel offsets of IMAGE_SAMPLE*_O (the behaviour before
	// U50) instead of applying them.
	bool sample_offsets = true;
	// KYTY_SAMPLE_LOD_CLAMP=0: ignore the LOD clamp of IMAGE_SAMPLE*_CL (the behaviour before U50)
	// instead of applying it as the sample's minimum LOD.
	bool sample_lod_clamp = true;
	// KYTY_HOST_FTZ_INPUTS=1: when the module declares DenormFlushToZero 32, leave the denormal
	// inputs of rcp/rsq/sqrt/exp/log to the host instead of flushing them in the shader. Vulkan
	// only says such operands "may" be flushed, so this is opt-in for hosts that pass
	// CodegenTranscendentalDenormInputs with "host FTZ declared" (the RTX 3090 on driver 610.74
	// has no f32 flush-to-zero, so it never applies there).
	bool host_ftz_inputs = false;
	// KYTY_EXEC_SELECTS=0: keep every EXEC-masked VGPR merge Select(exec, new, old) instead of
	// replacing the ones whose old value no lane can observe (IR::EliminateExecSelects).
	bool exec_selects = true;
	// KYTY_PS_APPEND_LIVE_ELECTION=0: in a pixel shader, let DS_APPEND/DS_CONSUME elect the first
	// EXEC lane for the counter atomic even when it is a helper invocation. By default the lane is
	// elected among the non-helper invocations: a pixel shader's EXEC can include the helper
	// invocations of partially covered quads (always without KYTY_PS_LIVE_EXEC, and in whole quad
	// mode with it), and Vulkan discards a helper invocation's atomic and leaves its result
	// undefined, so electing a helper broadcast an undefined base to the whole subgroup
	// (duplicate append indices; Astro Bot's GI ray-bundle linked lists then form cycles and
	// hang the GPU). The added count stays popcount(EXEC), so the per-lane indices a shader
	// derives with V_MBCNT stay unique.
	bool ps_append_live_election = true;
	// KYTY_PS_LIVE_EXEC=0|1|all (default 1): pixel shaders that contain DS_APPEND/DS_CONSUME (1)
	// or all pixel shaders (all) start with EXEC holding only the non-helper invocations; 0 starts
	// every pixel shader with a full EXEC. On the PS5 the initial EXEC is the pixel valid mask and
	// S_WQM_B64 adds the helper lanes of partially covered quads; with a full EXEC the exact-mode
	// EXEC a shader restores before its stores and atomics still holds the helpers, so a
	// DS_APPEND also counts and indexes the helper lanes and leaves slots that nothing writes
	// (unwritten particle slots in Astro Bot's mesh-particle emitter 0x4dd1f85484fc31f2, unlinked
	// nodes in its GI ray-bundle linked lists).
	PsLiveExec ps_live_exec = PsLiveExec::AppendConsume;
	// KYTY_LOOP_GUARD=<n> with KYTY_LOOP_GUARD_SHADERS=<hash>[,<hash>...] (hexadecimal guest shader
	// hashes): a diagnostic for a GPU hang suspected in a shader loop. Every structured loop of a
	// listed shader counts iterations against one per-invocation budget; an invocation that has
	// run more than <n> iterations takes each loop's exit edge, and at return it adds one to the
	// last GDS dword, which the command processor reports at flips ("Loop guard"). It changes the
	// guarded shaders' results when it fires. Off unless both variables are set.
	uint32_t              loop_guard_budget = 0;
	std::vector<uint64_t> loop_guard_shaders;
	// KYTY_SRT_VARIANT_READS=1: a scalar read whose address is only known inside the shader (not a
	// valid runtime value: e.g. a BVH traversal's loop-carried instance pointer, or data the GPU
	// produces) is planned as a runtime read instead of a flat SRT slot. A flat slot is evaluated
	// once before the dispatch, so such a slot always fails to evaluate and the whole dispatch is
	// dropped. An S_BUFFER_LOAD through a V# read that way then reads through BDA, and a program with
	// another such descriptor (no BDA path) is dropped, as before. Reads whose address can be
	// evaluated before the dispatch keep their flat slots.
	bool srt_variant_reads = false;
	// KYTY_NATIVE_INDIRECT_MESH=1|on|verify|exit: mesh draw dword 3 equal to
	// IR::PushData::MeshIndirectSentinel makes mesh shaders read their six draw dwords from the
	// parameter block at the device address in dwords 0-1 (a GPU-converted indirect mesh draw,
	// renderer/meshIndirect.h). Unset, 0 or "empty": the dwords are only ever pushed.
	bool mesh_indirect_params = false;
	// KYTY_REALTIME_CLOCK=0: S_MEMREALTIME returns the placeholder UINT64_MAX (the behaviour before
	// the real clock) instead of the host GPU clock. With the placeholder a guest spin-wait timed by
	// S_MEMREALTIME never times out, and only a GPU reset ends it. The emitter reads the clock the
	// device layer enabled (Spirv::HostShaderClock); a device without one keeps the placeholder.
	bool realtime_clock = true;
	// KYTY_DPP_SKIP_INACTIVE=0: a DPP source lane that EXEC disables reads zero (the behaviour
	// before), so the receiving lane gets op(0, x). On the guest such a source is invalid like a
	// vacated one, and without bound_ctrl the receiving lane keeps its value (PS5 ISA, DPP options);
	// so does a lane the host subgroup lacks. A work-list loop whose DPP row scan then yields a
	// wrong minimum never ends (a GPU hang). Not for DPP8 or fetch-inactive. Exact.
	bool dpp_skip_inactive = true;
	// KYTY_LANE_REDUCTIONS=0: a V_READLANE of the last lane of a DPP row scan with every lane enabled
	// (IR::MatchLaneReduction) stays an emulated scan instead of a native subgroup reduction over the
	// row or row pair. A partly filled host subgroup lacks lanes the scan would read; the reduction
	// counts them as the operation's identity, as the guest shader's own masking makes them.
	bool lane_reductions = true;
	// KYTY_DISPATCHER_CAP=<n> (default 4096; 0: no cap): a program whose CFG cannot be structured runs
	// as a dispatcher loop over its blocks (IR::Program::dispatcher_fallback), and an invocation
	// leaves that loop after n block transitions. A guest loop that never ends (for example a trip
	// count read from memory an unemulated pass left unwritten) then gives that invocation wrong
	// results instead of hanging the GPU until a device reset. Structured programs are unaffected.
	uint32_t dispatcher_cap = 4096;
};

// True when KYTY_LOOP_GUARD applies to the guest shader with this hash.
[[nodiscard]] bool LoopGuardApplies(uint64_t shader_hash);

[[nodiscard]] const CodegenOptions& GetCodegenOptions();
// Test hook: replaces the options for subsequent compilations. Not thread-safe; call it only
// while no shader is being compiled.
void SetCodegenOptions(const CodegenOptions& options);

} // namespace Libs::Graphics::ShaderRecompiler

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_
