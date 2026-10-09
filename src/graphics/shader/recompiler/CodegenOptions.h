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

// Where a wave32 graphics program keeps each guest wave to its own 32 lanes of the host subgroup
// (KYTY_WAVE32_CLUSTERS).
enum class Wave32Clusters : uint8_t {
	// Never: a host subgroup wider than 32 runs two guest waves as one (the behaviour before).
	Off,
	// In the stages the device layer reports as wider than 32 (Spirv::HostWaveClusters).
	Auto,
	// In every wave32 program but hull shaders, compute included (a test of the emitted code on
	// 32-wide hosts, where it must give the same results).
	Force,
};

// How S_WAITCNT lgkmcnt(0) after LDS writes orders them for the other lanes of the wave
// (KYTY_LDS_WAITCNT_BARRIER).
enum class LdsWaitcntBarrier : uint8_t {
	// No barrier (the behaviour before the ISA-accuracy change; pink particle clouds on the title
	// screen, blue artifacts in Astro's Playroom).
	Off,
	// OpMemoryBarrier at subgroup scope: every lane of a guest wave lives in one host subgroup
	// (a split wave64 runs two lanes per invocation), so this is all lgkmcnt(0) promises.
	Subgroup,
	// OpMemoryBarrier at workgroup scope (int16.1). The AMD driver turns a workgroup-scope fence in
	// a mesh shader into a workgroup barrier; reached by only some waves (inside S_CBRANCH_EXECZ
	// regions other waves skip) it hangs the GPU.
	Workgroup,
};

// Where a compute shader's LDS lives (KYTY_LDS_DEVICE_BUFFER, PlanComputeLds).
enum class LdsDeviceBuffer : uint8_t {
	// Workgroup memory only; an allocation above the device limit is clamped to it (int16.1 and
	// AMD tests 1-2: Astro Bot's 48 KiB compute shaders then write past their 32 KiB on AMD, the
	// stretched water geometry of issue 16).
	Off,
	// A device buffer, one region per workgroup of the dispatch, when the allocation exceeds the
	// device limit; workgroup memory otherwise.
	Auto,
	// The device buffer for every compute shader with LDS (a test of the buffer path on any host).
	Force,
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
	// KYTY_PS_HELPER_ATOMICS_SKIP=0: a pixel shader's helper invocations run the compare-exchange
	// loops of emulated read-modify-writes (sub-dword stores, float min/max, INC/DEC; AtomicUpdate in
	// spirvEmitterInternal.h) like the other lanes. By default they skip them: Vulkan does not perform
	// a helper invocation's atomics and leaves their results undefined, so the loop's exit test need
	// never hold in a helper lane and the wave can spin until a device reset (AMD keeps helper lanes
	// in the wave). Skipping changes nothing observable: a helper's atomic has no effect and no
	// defined result.
	bool ps_helper_atomics_skip = true;
	// KYTY_VOLATILE_LOADS=1 (diagnostic, default 0): every buffer, scalar-buffer and BDA (FLAT/GLOBAL,
	// indirect V#) access is Volatile, not only MUBUF accesses with GLC/DLC. A guest loop that polls
	// memory written by another workgroup or queue then rereads it instead of a cached or hoisted value.
	bool volatile_loads = false;
	// KYTY_LOOP_GUARD=<n> with KYTY_LOOP_GUARD_SHADERS=<hash>[,<hash>...] (hexadecimal guest shader
	// hashes): a diagnostic for a GPU hang suspected in a shader loop. Every structured loop of a
	// listed shader counts iterations against one per-invocation budget; an invocation that has
	// run more than <n> iterations takes each loop's exit edge, and at return it adds one to the
	// last GDS dword, which the command processor reports at flips ("Loop guard"). It changes the
	// guarded shaders' results when it fires. Off unless both variables are set.
	uint32_t              loop_guard_budget = 0;
	std::vector<uint64_t> loop_guard_shaders;
	// KYTY_LOOP_GUARD_SHADERS=all: every shader is guarded (with KYTY_LOOP_GUARD). An exhausted
	// invocation also records its shader hash (GDS dwords end - LoopGuardHashGdsFromEnd and the one
	// after), which the "Loop guard" report names: one run finds a shader that never leaves a loop.
	bool loop_guard_all = false;
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
	// KYTY_IR_LINEAR_USES=1 (default off; Senaxx 5145dc1f9): IR use-list bookkeeping without the
	// quadratic searches (Inst::ReplaceUsesWith takes a use list over at once, AddUse searches for
	// duplicates only in debug builds, RemoveUse searches from the end, RemoveIdentities drops the
	// removed identities' entries in one pass, ~Program detaches without maintaining use lists) and
	// RewriteToSsa seals a block as soon as every predecessor is filled. Translation time only: the
	// same program, but early sealing can create a loop header's phis in another order, which
	// renumbers the module's ids (same size and meaning; the driver compiles it as a new module).
	bool ir_linear_uses = false;
	// KYTY_FOLD_LANE_MASKS=1 (default off; Senaxx 5189ea360): reads of the current lane's bit of a
	// wave mask that is known per lane (a ballot of a predicate, an all-zero or all-one constant,
	// and bitwise logic or WQM of those) become that predicate (IR::FoldLaneMasks). Smaller SPIR-V
	// and shorter driver compiles; not in hull shaders, and in pixel shaders ANDed with the lane's
	// own bit of a ballot of true (helper invocations may sit out of ballots).
	bool fold_lane_masks = false;
	// KYTY_RT_STUB=1: translate IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY (MIMG
	// 0xe6/0xe7) as "no intersection" for every lane instead of skipping compute dispatches that
	// contain them (and failing other stages). A box node returns four invalid child pointers
	// (0xffffffff); a triangle node returns t_num=+inf, t_denom=1.0 and zero in dwords 2-3 (a
	// cleared hit_status in triangle return mode 0). Diagnostic only: it never reports a hit.
	bool rt_stub = false;
	// KYTY_RT_SOFTWARE (default 1): translate the BVH instructions exactly in software (IR
	// BvhIntersectRay, lowered in the SPIR-V backend; spec: RT-SOFTWARE-DESIGN.md). Takes precedence
	// over KYTY_RT_STUB. Astro Bot's own tiled deferred lighting and its GI probe tracing contain
	// these instructions; without a BVH mode their whole dispatches are skipped, which leaves the
	// scene unlit (black robots on the title screen). KYTY_RT_SOFTWARE=0 with KYTY_RT_STUB=1 keeps
	// the passes but lets every ray miss (no ray-traced shadows); KYTY_RT_SOFTWARE=0 alone restores
	// the old skip.
	bool rt_software = true;
	// KYTY_RT_TYPE6=0 (with KYTY_RT_SOFTWARE): node type 6 misses (four invalid children) like
	// RDNA2's user node, instead of being decoded as the PS5 shared-exponent box.
	bool rt_type6 = true;
	// KYTY_BDA_WRITES=1 (or =verify): raw stores and atomics through a V# the shader computes
	// (Psr's BVH builders) write guest memory through BDA instead of failing resource tracking. The
	// renderer settles each such dispatch synchronously: it waits for it and marks the pages it wrote
	// GPU-owned before the CP continues (BDA-WRITES-DESIGN.md). Default 1: Astro Bot's BVH builders
	// need it once its game patches are off; KYTY_BDA_WRITES=0 restores the old refusal.
	bool bda_writes = true;
	// KYTY_RT_NODE_BUDGET=<n> (with KYTY_RT_SOFTWARE; 0 = no limit): the most BVH node tests one
	// guest lane runs. A garbage or cyclic BVH can keep the guest's traversal looping forever and
	// lose the device. Past the budget every node test misses without reading memory, the
	// invocation takes each loop's exit edge (as KYTY_LOOP_GUARD does), and at return it adds one to
	// GDS dword end - RtNodeBudgetGdsFromEnd, which the command processor reports at flips ("RT
	// node budget"). Every invocation of a wave executes the instruction for each node the wave's
	// packet traversal visits, so the count is the wave's traversal length. The default is a safety
	// net far above expected traversals (KYTY_RT_NODE_STATS measures them).
	uint32_t rt_node_budget = 8192;
	// KYTY_RT_NODE_STATS=1 (with KYTY_RT_SOFTWARE): at return, each invocation that ran node tests
	// adds one to the GDS dword of its per-lane count's power of two (bin k holds counts in
	// [2^k, 2^(k+1)), at end - RtNodeStatsGdsFromEnd - k), reported at flips. A diagnostic for the
	// budget.
	bool rt_node_stats = false;
	// KYTY_WAVE32_CLUSTERS=0|auto|force (default auto), see Wave32Clusters. A wave32 vertex, mesh or
	// pixel program on a host subgroup wider than its wave (AMD: 64 in those stages, which cannot
	// require 32) shares the subgroup with another guest wave. Without clusters its EXEC, VCC and
	// ballot words read the first 32 lanes of the subgroup, its lane reads and V_READFIRSTLANE take
	// lanes of the other wave, and V_MBCNT counts them: a waterfall or work-list loop of the second
	// wave never removes its own lanes and spins until a device reset, and DS_APPEND hands its lanes
	// duplicate indices. With clusters each guest wave uses host lanes 32k..32k+31 only: lane ids
	// are the low 5 bits, ballots take the wave's own word, shuffles stay in the wave, lane reads
	// are not broadcast across the subgroup, and lane reductions use the emulated scan.
	Wave32Clusters wave32_clusters = Wave32Clusters::Auto;
	// KYTY_LDS_WAITCNT_BARRIER=subgroup|workgroup|0 (default subgroup), see LdsWaitcntBarrier. An
	// S_WAITCNT lgkmcnt(0) with an LDS write pending since the last barrier, in compute and mesh
	// shaders (the other stages keep LDS per invocation), gets a memory barrier so lanes see each
	// other's writes. It orders only the wave's own LDS operations, so it is never an execution
	// barrier and never wider than the subgroup by default: Astro Bot's title-screen particle mesh
	// shader 0xc739f9614016bed4 (VS 0x20c94d46ce55a14b) has three in S_CBRANCH_EXECZ regions, and
	// with workgroup scope it lost the device on AMD at the next S_BARRIER.
	LdsWaitcntBarrier lds_waitcnt_barrier = LdsWaitcntBarrier::Subgroup;
	// KYTY_LDS_DEVICE_BUFFER=auto|0|force (default auto), see LdsDeviceBuffer. Astro Bot asks for
	// 48 KiB of LDS in some compute shaders; NVIDIA allows 48 KiB, AMD's Windows driver 32 KiB.
	LdsDeviceBuffer lds_device_buffer = LdsDeviceBuffer::Auto;
	// KYTY_LDS_LIMIT_OVERRIDE=<bytes> (test mode, default 0 = the device's own): treat the device's
	// compute LDS limit as at most this many bytes, so a 48 KiB host takes the AMD path.
	uint32_t lds_limit_override = 0;
	// KYTY_PS_PER_VERTEX=0 (AMD test 3 diagnostic, default on): V_INTERP_MOV P10/P20 reads (a pixel
	// shader reading raw vertex attributes) use the interpolated attribute instead of PerVertexKHR
	// inputs and fragment barycentrics. Wrong values in those shaders; for a driver that crashes
	// compiling per-vertex inputs (two RX 6900 XT crashes in the same VS/PS pair, AMD test 2).
	bool ps_per_vertex = true;
	// KYTY_CLIP_GUARD=0 (AMD test 3 diagnostic, default on): vertex shaders do not reserve a clip
	// plane for the zero-position cull (the "emitted zero-position clip guard").
	bool clip_guard = true;
};

// How a compute shader's LDS allocation is placed on this host (KYTY_LDS_DEVICE_BUFFER,
// KYTY_LDS_LIMIT_OVERRIDE). dwords: the size the program declares and bounds its accesses by;
// storage: it lives in the SharedMemory device buffer; clamped: dwords is below the request.
struct ComputeLdsPlan {
	uint32_t dwords      = 0;
	uint32_t limit_bytes = 0;
	bool     storage     = false;
	bool     clamped     = false;
};
[[nodiscard]] ComputeLdsPlan PlanComputeLds(uint32_t requested_dwords, uint32_t device_limit_bytes);
// Dwords between consecutive workgroups' regions of the LDS device buffer: the LDS size rounded up
// to 16 bytes, so 64-bit LDS atomics stay aligned (guest sizes are multiples of 512 bytes).
[[nodiscard]] constexpr uint32_t LdsStorageRegionDwords(uint32_t lds_dwords) {
	return (lds_dwords + 3u) & ~3u;
}

// GDS dwords, counted from the end of GDS, that KYTY_RT_NODE_BUDGET and KYTY_RT_NODE_STATS report
// through (the last one is KYTY_LOOP_GUARD's).
inline constexpr uint32_t RtNodeBudgetGdsFromEnd = 2;
inline constexpr uint32_t RtNodeStatsGdsFromEnd  = 3;
inline constexpr uint32_t RtNodeStatsBins        = 24;
// KYTY_LOOP_GUARD: the low and high dwords of the last exhausted shader's hash, past the RT bins.
inline constexpr uint32_t LoopGuardHashGdsFromEnd = RtNodeStatsGdsFromEnd + RtNodeStatsBins + 1;

// True when KYTY_LOOP_GUARD applies to the guest shader with this hash.
[[nodiscard]] bool LoopGuardApplies(uint64_t shader_hash);

[[nodiscard]] const CodegenOptions& GetCodegenOptions();
// Test hook: replaces the options for subsequent compilations. Not thread-safe; call it only
// while no shader is being compiled.
void SetCodegenOptions(const CodegenOptions& options);

} // namespace Libs::Graphics::ShaderRecompiler

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_
