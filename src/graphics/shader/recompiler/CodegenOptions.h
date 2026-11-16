#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_

#include <algorithm>
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
	// KYTY_MOVREL_KNOWN_ZEROS (default on since claude/u59-gpu; =0 off): with movrel_range, a V_MOVRELS/V_MOVRELD compare of M0
	// against a register offset that sets a bit M0 never has also folds when M0's value set is
	// unknown. Known zero bits pass through shifts by constants, masks, sums, products, bit-field
	// extracts, selects, phis and lane reads: M0 = loop counter << 2 (Astro Bot's foliage vertex
	// shaders) leaves every fourth register of each chain. Exact.
	bool movrel_known_zeros = true;
	// KYTY_MOVREL_SWITCH (default on since claude/u59-gpu; =0 off): a V_MOVRELS select chain of 16 or more links is emitted as
	// an OpSwitch on M0 / 16 whose cases hold the select chains of their 16 index values. M0 is
	// uniform, so the host runs one short chain instead of a select per register. Exact.
	bool movrel_switch = true;
	// KYTY_UNIFORM_LANE_READS (default on since claude/u59-gpu; =0 off): the shuffle result of V_READFIRSTLANE/V_READLANE
	// (every lane already holds the same value) passes through OpGroupNonUniformBroadcastFirst,
	// which returns it unchanged but lets the host compiler treat it and everything derived from
	// it (a waterfall loop's key, addresses, loop exits) as uniform. Exact.
	bool uniform_lane_reads = true;
	// KYTY_SHORT_F32_HELPERS=1 (default off): NaN tests are one OpIsNan instead of exponent and
	// mantissa tests (float-to-int conversions, the legacy min/max/med3, float atomics). Exact.
	// (Bryan's companion change, dropping the sin/cos |x| >= 2^23 select, is not exact on the RTX
	// 3090 and is not applied.)
	bool short_f32_helpers = false;
	// KYTY_VS_LANE_WRAP32=0: V_MBCNT of a wave32 vertex shader no longer wraps the lane index at 32
	// when KYTY_WAVE32_CLUSTERS=0 (Integer.cpp). With clusters (the default) the lane id is already
	// the low 5 bits on hosts with wider subgroups, and a 32-wide host has no higher lanes, so the
	// wrap is emitted only with clusters off.
	bool vs_lane_wrap32 = true;
	// KYTY_FAST_FMINMAX=0: emulate f32 min/max/min3/max3/med3 with two bit classifications per
	// min/max instead of one compare-and-select plus a single two-zeros test.
	bool fast_float_min_max = true;
	// KYTY_NATIVE_FMINMAX=1 (EmK530/KytyPS5 d6825f587): with KYTY_FAST_FMINMAX, f32 min/max become
	// GLSL NMin/NMax. They return the other operand for one NaN like the guest, but leave which
	// zero min/max(-0, +0) returns unspecified and may treat denormals differently from the
	// guest's compare (see EmitFastMinMaxF32), so this stays opt-in.
	bool native_float_min_max = false;
	// KYTY_RELAXED_FLOAT=1 or --relaxed-shader-float (default off; upstream 8dbee8635): native float
	// ops whose NaN, signed-zero and denormal results may differ from the guest: f32/f64 min/max as
	// NMin/NMax (as KYTY_NATIVE_FMINMAX, also on the slow path), V_MED3_F32 from native min/max,
	// sin/cos without the signed-zero-preserving range cycle, and no shader-side f32 denormal input
	// flushes (EmitFlushF32DenormToSignedZero). The command-line option is applied at device
	// creation (vulkanWindow.cpp).
	bool relaxed_float = false;
	// KYTY_DCACHE_INV=1 (default off; upstream 22ef16a04): BUFFER_GL0_INV/GL1_INV emit a device
	// acquire barrier and make the program's storage buffers coherent (slower memory access).
	bool dcache_invalidate = false;
	// KYTY_READFIRSTLANE_EXEC0=1 (default off; upstream 919fac13d): V_READFIRSTLANE with EXEC
	// zero reads lane 0 (FindILsb + select) instead of BallotFindLSB's undefined lane.
	bool readfirstlane_exec_zero = false;
	// KYTY_UNIFORM_BUFFERS=1 (EmK530/KytyPS5 407de5105): bind constant-only buffers as uniform
	// buffers instead of storage buffers (IR::UniformBufferResources). Off by default: neutral in
	// Miles Morales on an RTX 3090 (GPU busy within noise), and on NVIDIA's 16-byte uniform
	// robustness granularity a read just past a V#'s end may return data instead of 0.
	bool uniform_buffers = false;
	// KYTY_WAVE64_WARPS=1 (EmK530/KytyPS5 420fc4ead): a wave64 compute program whose workgroup is
	// the one wave runs on a 32-wide host as two host subgroups, one guest lane per invocation,
	// instead of two lanes per invocation (Spirv::SplitWaveEligible). Cross-lane operations
	// exchange the halves through workgroup memory; a branch whose region holds no exchange or
	// barrier is decided by each half on its own lanes. KYTY_LANE_LOCAL programs stay as they are.
	bool wave64_warps = false;
	// KYTY_FAST_PKRTZ=0: convert V_CVT_PKRTZ_F16_F32 halves with the original select chain instead
	// of the shorter integer formulation.
	bool fast_pkrtz = true;
	// KYTY_SINGLE_F2I_SATURATION=0: repeat the float-to-int NaN/range saturation in the emitter even
	// though the translator's saturated conversion already guarantees an in-range operand.
	bool single_f2i_saturation = true;
	// KYTY_LOD_STATS_GATE=0: record GET_LOD_STATS feedback for every sample, including images whose
	// T# has no mip-statistics counter, and issue the finest-level AtomicUMin unconditionally.
	bool lod_stats_gate = true;
	// KYTY_LAYER_CLAMP=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 8c71f0a6a): an exported
	// render target array index past the bound layers (undefined in Vulkan: an NVIDIA device fault
	// writing at address 0 in GT7 race draws) selects layer 0, as the hardware and D3D do. The
	// bound layer count is a specialization constant (IR::LayerLimitSpecId) of the pipeline.
	bool layer_clamp = false;
	// KYTY_ROBUST_BUFFER_LOADS=0: bounds-check every plain dword storage-buffer load in the shader
	// even when the device's robustBufferAccess2 already returns zero for out-of-range dwords.
	bool robust_buffer_loads = true;
	// KYTY_ROBUST_BUFFER_STORES=0 (upstream ee034be4b relies on robustBufferAccess2 for stores): keep the
	// explicit bounds check on plain dword stores and 8/16-bit-view subword stores to storage buffers
	// even when the device drops writes past the descriptor range (HostBufferRobustness, the same
	// device condition as robust_buffer_loads). Atomics and 32-bit-view subword stores always keep it.
	bool robust_buffer_stores = true;
	// KYTY_BRANCHLESS_LOADS=1 (default off; YOTEI 434138316 / 01714a120 / 76a51a908, bruno 6ddeb3388):
	// a raw dword or subword load of a storage buffer that the device bounds-checks (robust_buffer_loads
	// on a robustBufferAccess2 device, DeviceChecksDwordLoad) runs in every lane and selects zero where
	// EXEC is off, instead of branching around it: its address is unchanged, an inactive lane's
	// out-of-range address reads the device's zero. Same results; fewer branches and phis per access.
	// Formatted, indirect and BDA loads keep their branches; so does everything on a device without
	// robustBufferAccess2.
	bool branchless_loads = false;
	// KYTY_BRANCHLESS_LDS=1 (default off): with branchless_loads, a workgroup-memory LDS or scratch dword
	// load also selects its element index to 0 (an element every such array has) where EXEC or the
	// bounds rule it out, then selects zero for the value, instead of two nested branches. Measured
	// slower on the RTX 3090 for Astro Bot's heaviest LDS kernel (cs 0x6856320a9c6f89fd: 1.42 -> 2.33 ms).
	// Device-buffer LDS and GDS keep their branches.
	bool branchless_lds = false;
	// KYTY_FLAT_APERTURE_BVH=1 (default off; Wolverine preset): a program with an
	// IMAGE_BVH*_INTERSECT_RAY routes its FLAT accesses by aperture even when it never reads an
	// aperture register. The PS5 ray-traversal stack addresses LDS/scratch through FLAT with
	// hard-coded aperture high words (V_MED3_U32 hi, 0x80000000, 0x70000000), so its pushes were
	// global stores, which cannot compile (Wolverine cs 0x2ea3957af2daf690).
	bool flat_aperture_bvh = false;
	// KYTY_LANE_LOCAL=1 (default off; chenxiao07/KytyPS5 9907780e1, 9bb47e15d): a wave64 compute program
	// proven lane-local (Frontend::IsWaveLaneLocal: its lanes' results depend on no lane outside their
	// 32-lane half) runs one guest lane per host invocation on a host whose compute subgroups are fixed at
	// 32 lanes (NVIDIA), each subgroup a half wave with its own copy of the scalar state, instead of two
	// lanes per invocation. Hosts with 64-wide subgroups already run one lane per invocation. A program
	// the proof accepts only with KYTY_LANE_LOCAL_UNIFORM_LOADS=1 keeps two lanes. KYTY_LANE_LOCAL_LOG=1
	// prints each verdict.
	bool lane_local = false;
	// KYTY_READONLY_BUFFERS=0: leave the storage buffers of a program that never stores to them
	// (no buffer store or atomic, no coherent load) without NonWritable. With it, the AMD driver
	// loads a uniform address once per wave into scalar registers instead of once per lane
	// (serbru20066666/brunoKytyPs5 def478029: a lighting shader of ASTRO's PLAYROOM 3.45 -> 2.76 ms
	// on an RX 6800 XT); other drivers may use their read-only caches.
	bool readonly_buffers = true;
	// KYTY_COHERENT_GLC_ONLY=1 (default off; Erlou93/KytyPS5-Yotei): in a program that writes no
	// buffer memory (no buffer store or atomic, no 8/16-bit coherent access, no u64 buffer view)
	// only the cache-bypassing (GLC/DLC) buffer accesses go through a Coherent alias of the storage
	// buffers ("buffers_coherent"); every other access uses the plain aliases, which may then be
	// NonWritable (KYTY_READONLY_BUFFERS). Without it one GLC access makes every buffer access of
	// the program Coherent. A program with BUFFER_GL0/GL1_INV (KYTY_DCACHE_INV) keeps whole-program
	// coherence unless the value is 2.
	uint8_t coherent_glc_only = 0;
	// KYTY_NATIVE_F16=1 (default off; Erlou93/KytyPS5-Yotei): V_PK_ADD/MUL/MIN/MAX_F16 without
	// clamp/omod become one f16vec2 instruction (Float16, RoundingModeRTE 16,
	// SignedZeroInfNanPreserve 16) instead of two f32 operations on converted lanes. Needs a device
	// with shaderFloat16 and those 16-bit float controls (SetHostNativeHalf). Checked on an RTX 3090
	// over all 2^32 lane pairs against the f32 path: add and mul identical; min and max identical
	// except NaN payloads when both lanes are NaN (signaling NaNs keep the existing quieting).
	bool native_f16 = false;
	// KYTY_F32_FLUSH_HOST=1 (default off; Erlou93/KytyPS5-Yotei): no shader-side flush of f32
	// denormal inputs of compares and of rcp/rsq/sqrt/exp/log in programs without f64: the host's
	// default f32 mode already flushes them. Checked on an RTX 3090 over all 2^24-2 denormals: those
	// five ops and the six compares against 16 values give the flushed operand's result. The value
	// "nvidia" turns it on for an NVIDIA device only, and only after a start-up GPU self-test of
	// that behaviour passes (F32FlushSelfTest.h; KYTY_F32_FLUSH_SELFTEST=off skips the test, =fail
	// forces its failure path). The self-test (CodegenF32FlushSelfTest) on an RTX 3090 with the
	// modes the emulator declares there (DenormPreserve 16) found rcp and every compare NOT flushing
	// (rcp(0x00400000) = 0x7f000000), as did a translated guest program; with no denormal mode
	// declared (KYTY_SHADER_FLOAT_CONTROLS=0) all of them flushed. So "nvidia" stays off there.
	bool f32_flush_host = false;
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
	// KYTY_INDIRECT_COHERENT=1 (default 0): a GLC/DLC load or dword store through a V# the shader computed
	// (ResourceKind::IndirectBuffer, read through BDA) is Coherent and Volatile, as the same load
	// through a bound V# already is. Without it the load is a plain one the driver may hoist out of
	// a polling loop or serve from a non-coherent cache: a decoupled look-back scan (Ratchet &
	// Clank's 0xba61d861c2b03285) spins forever on a partition flag another workgroup published.
	bool indirect_coherent = false;
	// KYTY_BUFFER_PAIR_ATOMIC=1 (default 0; on in the PPSA01417 title preset): a raw
	// BUFFER_LOAD_DWORDX2 / BUFFER_STORE_DWORDX2 with GLC or DLC, through a bound V# or one the
	// shader computed (BDA), whose two dwords are contiguous, in range and 8-byte aligned at run
	// time, is one 64-bit atomic load or store (Device scope, acquire / release) instead of two
	// dword accesses, as an aligned DWORDX2 is on the guest GPU. Miles Morales' TLAS builder
	// 0x49d3e4dbb4dd1ac4 publishes {status, prefix} pairs of a decoupled look-back scan so; with two
	// dword accesses a polling workgroup can read the new status with the old prefix. Any other
	// pair keeps the dword path; so does every pair on a host without 8-byte aligned storage-buffer
	// bindings (SetHostBufferPairAtomics).
	bool buffer_pair_atomic = false;
	// KYTY_INDIRECT_FORMAT_INLINE=1 (default 0; on in the PPSA01417 and PPSA03671 title presets): the
	// runtime formatted-load helper (indirect_formatted_load) is an ordinary function the driver may
	// inline instead of a DontInline one. Miles Morales' RT kernel 0x208e855cd52d25f9 and PPSA03671's
	// partitioned material kernels lost the device with InstructionPointerInvalid / Shader Instruction
	// Error at the end of the SASS of modules holding the DontInline helper.
	bool indirect_format_inline = false;
	// KYTY_INDIRECT_FORMAT_FAST=1 or =<format>[,...] (default off): the loads call a small inlinable
	// decoder of the common formats; only the other formats call the DontInline full decoder
	// (spirvEmitterMemory.cpp IndirectFormatFastSet). Keeps KYTY_INDIRECT_FORMAT_INLINE's crash fix
	// without inlining the whole format switch into every RT kernel (driver memory).
	bool indirect_format_fast = false;
	// KYTY_INDIRECT_FORMAT_COMPACT=1 (default 0): the formatted loads through a computed V# call one
	// small inlinable decoder for every format instead of the per-format switch (FAST / INLINE /
	// DontInline). The format selects a packed descriptor (component widths, numeric kind, count)
	// from a constant table; the element is read as up to four dwords with two page-table lookups
	// and every component is extracted and converted generically. Same results as the switch; no
	// DontInline function (device loss in huge RT kernels) and a small inlined copy per call site
	// (driver compile memory). Takes precedence over KYTY_INDIRECT_FORMAT_FAST / _INLINE.
	bool indirect_format_compact = false;
	// KYTY_INDIRECT_RAW_OUTLINE=1 (default 0): each raw BUFFER_LOAD through a V# the shader computed
	// (ResourceKind::IndirectBuffer, unformatted) calls one DontInline function per load shape that
	// decodes the V#, checks the range and reads through BDA, instead of inlining ~200 SPIR-V words per
	// dword at every load (adapted from Erlou93/KytyPS5-Yotei F1, indirect_raw_dword / _x<N>). Same
	// results. Ghost of Yotei's 0x4ca0cf7e5a0c4757 has ~500 such loads; its driver compile took 65 s.
	bool indirect_raw_outline = false;
	// KYTY_HUGE_HELPER_OUTLINE=0 turns off (default on; only acts with KYTY_INDIRECT_RAW_OUTLINE): in a
	// module of at least 100,000 words that already holds the DontInline raw-load functions, the
	// compact formatted-load decoder and get_bda_pointer become DontInline when they have at least 16
	// call sites. The NVIDIA compiler spent 10-35 s inlining them at every call site (Ghost of Yotei
	// CS 0xc1989e057150728f: 132 decoder calls, 27-34 s -> 4 s offline on an RTX 3090; 0x4ca0cf7e5a0c4757
	// 19 s -> 3.5 s; 0xe8e383355f93ab53, 169 get_bda_pointer calls, 11 s -> 4 s). Same results; the
	// helpers run as calls. Limited to modules that already call DontInline helpers because DontInline
	// decoders lost the device in other titles' huge kernels (see indirect_format_inline).
	bool huge_helper_outline = true;
	// KYTY_NULL_BUFFER_SHARE=1 (default 0): a null buffer V# (all zeros) specializes like the last live
	// V# of its slot (ResourceMaterialization.cpp SpecializeBuffer), so a shader whose slots are null in
	// some dispatches and live in others has one permutation instead of one per null/live combination
	// (Ghost of Yotei: 0x79b0066f0cd1f3cd had 15, each a 50-100 s driver compile; Erlou93/KytyPS5-Yotei
	// M115). The renderer binds such a slot as the null buffer at the null marker offset
	// (NullBufferMarkerOffset), so every access is out of range: loads read 0, stores drop, and a
	// formatted read's DST_SEL constants are replaced by 0 as a null V# gives.
	bool null_buffer_share = false;
	// KYTY_BOUNDED_KEY_IMAGES=1 (default 0; PPSA26344 preset): a T# read from a scalar-address table
	// at key << 5 whose key is an unsigned bit-field extract of constant width (at most 6 bits) or a
	// low-bit mask (at most 63) enumerates every key below that bound, as a loop-bounded key does
	// (adapted from Erlou93/KytyPS5-Yotei's MatchDescriptorTable bounds). Ghost of Yotei's compute
	// shaders take a 5-bit slot field from a record. Off: such programs are skipped, as before.
	bool bounded_key_images = false;
	// KYTY_R128_TABLE_IMAGES=1 (default 0; PPSA26344 preset): a T# read from a table for image
	// instructions that all use a 128-bit T# (R128=1) plans only its four words: the upper four are
	// constants, not table reads. Bindless heaps of such images zero the words past the 16-byte T#.
	// Off: such programs are skipped, as before (Ghost of Yotei's compute shaders).
	bool r128_table_images = false;
	// KYTY_INDIRECT_FORMAT_STORES=1 (default 0; NOT in the PPSA26344 preset: with it Ghost of
	// Yotei's village lighting turned magenta on half the screen, so the port is still wrong): a formatted 32-bit
	// BUFFER_STORE_FORMAT_X..XYZW through a V# a compute shader computed is written through BDA (with
	// KYTY_BDA_WRITES) in the V#'s run-time format: the whole element (missing channels zero),
	// packed formats as one 16/32-bit piece, others per 8/16/32-bit channel (adapted from
	// Erlou93/KytyPS5-Yotei's StoreIndirectFormatted, M48). Off: such programs are skipped, as before.
	bool indirect_format_stores = false;
	// KYTY_DROP_GRAPHICS_INDIRECT_STORES=1 (default 0; PPSA26344 preset; an approximation): a raw
	// store (not an atomic) through a V# a graphics-stage shader computed is removed instead of
	// skipping every draw of the program (BDA writes are compute-only). Each dropped store is
	// logged once per program. Ghost of Yotei's split merged hull program HS 0x93ec51ae70b429ce
	// stores a byte that way; without the switch its tessellated draws are skipped.
	bool drop_graphics_indirect_stores = false;
	// KYTY_GRAPHICS_BDA_WRITES=1 (default 0; PPSA26344 preset): a raw store or atomic through a V#
	// a graphics-stage shader computed writes through BDA as in compute (KYTY_BDA_WRITES); the
	// renderer prepares the write log before such a draw and settles it right after the draw
	// (RenderExecutor::ExecutePreparedDraw). For Ghost of Yotei's split merged hull program
	// HS 0x93ec51ae70b429ce, whose byte stores feed later GPU work (dropping them lost the device).
	bool graphics_bda_writes = false;
	// KYTY_INDIRECT_FORMAT_STORES_LOG=1 (default 0; with KYTY_INDIRECT_FORMAT_STORES): a formatted
	// store piece for a page without a cache buffer is appended to the BDA store log (written into
	// guest memory by the settle) instead of being dropped, as raw stores are (KYTY_BDA_STORE_LOG).
	bool indirect_format_stores_log = false;
	// KYTY_INDIRECT_FORMAT_D16=1 (default 0; PPSA26344 preset): BUFFER_LOAD_FORMAT_D16_XYZ/XYZW (two dwords) through a
	// V# the shader computed read through the indirect format decoder, then pack component pairs into
	// dwords as the D16 load does (16-bit float for float/norm formats, clamped 16-bit integers for
	// UINT/SINT, chosen from the V#'s format at run time). Off: such programs are skipped, as before.
	// Ghost of Yotei's merged hull programs and several compute shaders use them.
	bool indirect_format_d16 = false;
	// KYTY_INDIRECT_FORMAT_INLINE_HASHES=<hex>[,<hex>...] (default unset): with
	// KYTY_INDIRECT_FORMAT_FAST, these programs inline the full decoder (as
	// KYTY_INDIRECT_FORMAT_INLINE does for all), the others keep the fast + DontInline pair.
	// KYTY_INDIRECT_FORMAT_COUNT=1 (diagnostic): every call of the DontInline full decoder adds one
	// to a per-format counter in a host-visible buffer (Spirv::SetIndirectFormatCountAddress); the
	// buffer cache prints the counts every 600 flips ("Indirect formats via the full decoder").
	// Its address is part of the codegen fingerprint: use with KYTY_PROGRAM_CACHE_SAVE=0.
	// KYTY_CFG_CLONE_EXITS=1 (default 0): before a CFG falls back to the dispatcher loop, a small
	// terminal block (no successors, at most 8 instructions) entered from several blocks gets one
	// copy per entering block, and structurization is retried. Pixel shaders branch to a shared
	// "every lane discarded: null export, s_endpgm" tail from inside nested selections, which made
	// the region of the outer selection externally entered (Wolverine's 0x09161b28ae4039bb). The
	// copies run the same instructions; nothing after them can tell them apart.
	bool cfg_clone_exits = false;
	// KYTY_CFG_CLONE_REGIONS=1 (default 0): before a CFG falls back to the dispatcher loop, a
	// selection whose region is also entered from outside the selection (a jump into the middle of
	// an if) gets a private copy of the entered part for the outside predecessors, and
	// structurization is retried (ShaderCFG.cpp CloneEnteredRegion: loop headers and back-edge
	// targets are never copied; at most as many instructions as the program has). The copies run
	// the same instructions. Ghost of Yotei's CS 0x2b6417d1a9a30084 ran as a 158-block dispatcher
	// loop for ~3 s of GPU time per frame.
	bool cfg_clone_regions = false;
	// KYTY_STORAGE_LENGTH_ZERO=1 (default 0): a storage-buffer OpArrayLength of 0 counts as unbounded
	// (0xffffffff), leaving the bounds check to the device. NVIDIA returns 0 for a binding of 2 GiB or
	// more (Insomniac binds its whole vertex pool, 3 GiB, through one V#); with this switch the
	// descriptor range is no longer capped (KYTY_STORAGE_RANGE_CAP auto), so data past 2 GiB reads.
	bool storage_length_zero = false;
	// KYTY_HEAP_KEY_MASK=1 (default 0; on in the PPSA03671 title preset): a T# read from a heap at
	// (key << 5) & mask, with the mask's low five bits clear and a key only the shader knows (a
	// V_READFIRSTLANE of a lane value), is planned as the heap record of key & (mask >> 5): a bindless
	// image when the device has them (ResourceTracking, MatchTableOffset). With such a key, a heap T#
	// whose word 0 the shader also uses as a value (0x885fa47aec73f696 spills it with V_WRITELANE
	// before the sample) gets the same bindless plan. Without it such a program is skipped
	// ("computes a descriptor at runtime that has no BDA path"). Exact; only programs that were
	// skipped change.
	bool heap_key_mask = false;
	// KYTY_WAVE64_ACTIVE_ZERO=1 (default 0; on in the PPSA26344 title preset; adapted from
	// peepostare's YOTEI-BRANCH 9984ebea7): when one host invocation runs both halves of a wave64
	// (other_half), an EXECZ/VCCZ/SCCZ branch holds when both halves' ballots equal the host
	// subgroup's active mask instead of all ones. A host subgroup with invocations that are not
	// running (a partial wave, invocations that already returned) has no ballot bit for them, so the
	// all-ones compare never held and a loop exiting on such a branch spun forever (a GPU hang).
	// Identical results when every host invocation is active; one extra ballot per such branch.
	bool wave64_active_zero = false;
	// KYTY_LANE2_SHARE=1 (default 0; on in the PPSA26344 title preset; adapted from Erlou93/KytyPS5-Yotei
	// spirvEmitterLaneShare.cpp): a wave64 compute program that runs two lanes per host invocation
	// (lane_count 2) emits an instruction whose value is provably the same in both halves (scalar
	// state, loop counters, addresses from user data) once; the high half reuses the low half's id.
	// Exact: the second copy would be the same SPIR-V on the same ids. KYTY_LANE2_SHARE_LOG=1 prints
	// the counts per program.
	bool lane2_share = false;
	// KYTY_KNOWN_BITS_MASKS=1 (default 0; on in the PPSA26344 title preset; adapted from
	// Erlou93/KytyPS5-Yotei ConstantPropagation.cpp PossibleBits): an AND with a constant mask that keeps
	// every bit its other operand can have (KnownZeros), and a BitFieldUExtract from bit 0 of a value
	// with no bit at or above the width, become that operand. Exact (needs KYTY_MOVREL_RANGE, the
	// known-zero analysis).
	bool known_bits_masks = false;
	// KYTY_GDS_M0_BASE=1 (default 0; on in the PPSA26344 title preset; adapted from Erlou93/KytyPS5-Yotei
	// Translator::DsAddress): DS reads, writes and atomics on GDS add M0[31:16] (the GDS base) to their
	// address, as DS_APPEND/DS_CONSUME already do. Without it the base is ignored.
	bool gds_m0_base = false;
	// KYTY_LDS_ATOMIC_ORDER=1 (default 0; on in the PPSA26344 title preset; adapted from
	// Erlou93/KytyPS5-Yotei M110): DS_ADD_RTN_U32/DS_SUB_RTN_U32 on workgroup LDS return their old values in
	// lane order for lanes that address the same dword, as the hardware does (EmitLaneOrderedLdsAdd).
	bool lds_atomic_order = false;
	// KYTY_SPLIT_LDS_WRITE_SYNC=1 (default 0; on in the PPSA26344 title preset; adapted from
	// Erlou93/KytyPS5-Yotei EmitPairLdsWriteSync): in a split wave (KYTY_WAVE64_WARPS) an LDS write is
	// followed by a workgroup barrier and preceded by one only when this half read the LDS since the last
	// barrier, instead of a barrier before every LDS access of a program that writes the LDS. The same
	// orderings (a read sees the other half's earlier writes; a write does not overtake its earlier reads)
	// with far fewer barriers in programs that read the LDS often.
	bool split_lds_write_sync = false;
	// KYTY_STORAGE_TABLE_LOOSE=1 (default 0; on in the PPSA26344 title preset; adapted from
	// Erlou93/KytyPS5-Yotei M120): a storage image table the GPU indexes accepts candidates that differ
	// from its exemplar only in a plain dimension or in the swizzle of a format without conversion; each
	// write through the table takes the candidate's own coordinates. Before, such a table failed the
	// stage ("indirect image table ... has incompatible candidates").
	bool storage_table_loose = false;
	// KYTY_SAMPLED_TABLE_READS=1 (default 0; not in the PPSA26344 preset yet; adapted from
	// Erlou93/KytyPS5-Yotei EmitImageAccess): an IMAGE_LOAD through a sampled image table the GPU
	// indexes reads the candidate its key selects (one switch case per candidate, each with its own
	// coordinates), and the table accepts candidates that differ from its exemplar only in a plain
	// dimension when every access to it is a load or a coordinates-only sample. Before, such a table
	// failed the stage ("indirect image table ... has incompatible candidates"; Ghost of Yotei's
	// "Update Oct Maps" CS e8e383355f93ab53). Its partner CS 0x0b0ec2bac5920bd3 needs
	// KYTY_INDIRECT_FORMAT_STORES (or KYTY_INDIRECT_FORMAT_STORES_SHADERS): in one boot with both
	// switches the village lighting had no magenta, while the stores alone (0b0e without e8e3) gave it.
	bool sampled_table_reads = false;
	// KYTY_SCALAR_LOAD_GROUPS=1 (default 0; on in the PPSA26344 title preset; adapted from
	// Erlou93/KytyPS5-Yotei FindScalarLoadGroups): the dwords of one S_LOAD_DWORDXn through a scalar
	// address share one BDA page lookup when they lie in one page (EmitScalarLoadGroup). Exact.
	bool scalar_load_groups = false;
	// KYTY_SRT_UNIFORM=1 (default 0; adapted from Erlou93/KytyPS5-Yotei's uniform constant buffers):
	// the flattened SRT binding is a uniform buffer instead of a storage buffer (a dword array in a
	// uniform block, uniformBufferStandardLayout; the host binds the same uploaded words with a range
	// of at most 64 KiB). Reads of it (S_LOAD of SRT slots, runtime mapping tables) then come from
	// NVIDIA's constant cache, constant-index reads often as instruction operands, instead of one L1
	// load each. Same bytes, same results. Needs the uniform dword-load path (KYTY_UNIFORM_BUFFERS'
	// device requirements); without it the binding stays a storage buffer.
	bool srt_uniform = false;
	// KYTY_NULL_IMAGE_KEEP_DIM=1 (default 0; on in the PPSA26344 title preset; TheCruZ 4f1b2ebc): a
	// null image keeps the shader's dimension instead of the canonical 2D view (fewer program variants).
	bool null_image_keep_dim = false;
	// KYTY_STORAGE_SWIZZLE_CANON=1 (default 0; on in the PPSA26344 title preset; TheCruZ 72e95dcd): a
	// storage image swizzle that stores every channel of its format from its own component is
	// specialized as the identity (fewer program variants; same stored bits).
	bool storage_swizzle_canon = false;
	// KYTY_RAW_SMALL_SINT_STORAGE=1 (default 0; adapted from Erlou93/KytyPS5-Yotei): a write-only,
	// non-atomic storage image of 16- or 8-bit SINT format stores the same low bits through an
	// unsigned view (R16_UINT / R8_UINT), as 32-bit SINT already does, instead of failing the
	// program. Ghost of Yotei's cloud shadow precompute (CS 753c552f, R16 SINT) was dropped without it.
	bool raw_small_sint_storage = false;
	// KYTY_POINT_INTEGER_IMAGES=1 (default 0; Erlou93/KytyPS5-Yotei): every UINT image samples with point
	// filtering, as SINT and converted UINT already do (no linear filter for integer formats in
	// Vulkan, VUID-04553; Yotei's CS 839cc50b samples an 8_UINT id map with a bilinear S#).
	bool point_integer_images = false;
	// KYTY_COMPARE_SAMPLER_SPLIT=1 (default 0; Erlou93/KytyPS5-Yotei FloatCompare, from peepostare
	// f0bed387): one S# serving a plain sample and a depth-compare sample gets a host sampler for each
	// (before, the plain sample also got the compare sampler: undefined on a colour view).
	bool compare_sampler_split = false;
	// KYTY_WATERFALL_ACTIVE_LEADER=1 (default 0; Erlou93/KytyPS5-Yotei EmitReadLane M124): the
	// waterfall V_READLANE (lane = S_FF1(pending) & 63) of a graphics shader on one host lane per
	// invocation reads the lowest lane that is pending AND active; with only inactive lanes pending it
	// returns a key no active invocation holds.
	bool waterfall_active_leader = false;
	// KYTY_WAVE64_PRESENT_EXEC=1 (default 0; Erlou93/KytyPS5-Yotei PresentExec): a wave64 graphics
	// shader runs in 32-wide host subgroups; scalar EXEC writes (S_LSHR_B64 exec, -1, 64-n) are
	// masked to the lanes the host subgroup has, so waterfall loops cannot spin on absent lanes.
	bool wave64_present_exec = false;
	// KYTY_EMBEDDED_FETCH_NO_SGPR_OFFSET=1 (default 0; Erlou93/KytyPS5-Yotei DetectEmbeddedVertexFetch): a
	// vertex buffer descriptor loaded from the attribute table at a runtime SGPR offset is not an
	// embedded fetch; the load runs as an ordinary buffer load with the descriptor the shader computes
	// (terrain grid shaders patch format/offset at runtime).
	bool embedded_fetch_no_sgpr_offset = false;
	// KYTY_MOV_NO_FLOAT_MODS=1 (default 0; Erlou93/KytyPS5-Yotei DIAG_MOV_NO_FLOAT_MODS): v_mov_b32 is a
	// bit move (B32 operand in AMD's ISA tables), so the neg/abs input modifiers of its VOP3/SDWA/DPP
	// forms leave the source unchanged. Ghost of Yotei's cloud shadow CS a2dfc83fbe2e55b8 (pc 0x2e4
	// v_mov_b32_sdwa v12, -s106): with the modifier every cloud shadow texel got the same sign.
	bool mov_no_float_mods = false;
	// KYTY_SRGB8_DECODE=1 (default 0; Erlou93/KytyPS5-Yotei IsSrgbFormat / SrgbToLinear): a sampled
	// 8 or 8_8 sRGB texture (UNorm backing, vulkanCommon.cpp) is decoded from sRGB in the shader on
	// every fetch, gather and sample (after filtering; Erlou samples 8 sRGB through an R8_SRGB view).
	bool srgb8_decode = false;
	// ISA-check fixes (RDNA2 ISA REPORT-ISA-2/3, claude/u59-isa-fix-b; default 0, on in the PPSA26344
	// title preset):
	// KYTY_POPS_INTERLOCK=1: a pixel shader with a POPS section (S_SETREG_B32 hwreg(POPS_PACKER),
	// the pops_exiting_wave_id wait, S_SENDMSG MSG_ORDERED_PS_DONE; IR::ShaderInfo::pops) runs its
	// whole body inside a VK_EXT_fragment_shader_interlock critical section, pixel ordered
	// (PixelInterlockOrderedEXT): overlapping fragments execute in primitive order, as the guest's
	// ordered section makes its image read-modify-write. The device must expose
	// fragmentShaderPixelInterlock (Spirv::HostFragmentInterlock); without it such programs keep
	// the unordered code and are named once.
	bool pops_interlock = false;
	// KYTY_FORMAT_MISSING_CANON=1: a formatted buffer load whose V# dst_sel names a channel the
	// format lacks reads 0 (X/Y/Z) or 1 (W), the canonical RGBA expansion (ISA 8.2.5, Table 31),
	// instead of wrapping onto the format's channels (Format::ResolveFormattedSource). Bound V#s,
	// computed V#s and embedded vertex fetches. Bound V#s that hit it are logged (first 32).
	bool format_missing_canonical = false;
	// KYTY_INDIRECT_FORMAT_ALL=1: a formatted load through a computed V# decodes every buffer format
	// (USCALED/SSCALED, the non-float 10_11_11/11_11_10 and 2_10_10_10 UINT/SINT) instead of reading
	// 0 outside the vertex-format list (IndirectFormatDecoded).
	bool indirect_format_all = false;
	// KYTY_FORMAT_STORE_DST_SEL=1: BUFFER_STORE_FORMAT_* through a bound V# honour the V# dst_sel
	// (Table 31): memory channel c takes the data component whose dst_sel names it (0 if none), the
	// inverse map image stores already apply (StoreTexel).
	bool format_store_dst_sel = false;
	// KYTY_BARRIER_ACQUIRE_STORES=1 (default 0): S_BARRIER in a compute-like program that stores to
	// buffers or images (bound or computed V#s, storage images) also orders buffer and image memory
	// (UniformMemory | ImageMemory semantics, workgroup scope), not only LDS: waves read the other
	// waves' stores with plain loads right after the barrier (mainline ceb154d9e; Erlou93's Yotei
	// cloth modules carry it). Not the cause of Ghost of Yotei's stiff menu banner (A/B 2026-11-08).
	bool barrier_acquire_stores = false;
	// KYTY_IMAGE_GLC_COHERENT=1: IMAGE_LOAD* / IMAGE_STORE* with GLC or DLC (ISA Tables 29/41, the
	// buffer meaning) access the storage image through a Coherent alias of its binding, so only those
	// accesses bypass the non-coherent caches (the whole-binding Coherent decoration of buffers costs
	// a lot on NVIDIA). Sampled images (IMAGE_SAMPLE* with GLC) cannot be Coherent: unchanged.
	bool image_glc_coherent = false;
	// KYTY_IMPLICIT_LOD_STAGES=1: an implicit-LOD sample outside pixel shaders follows the ISA, which
	// has no stage restriction: in compute shaders (wave32, device with computeDerivativeGroupLinear)
	// the LOD comes from the derivatives across each group of four consecutive lanes (the RDNA2
	// quad, DerivativeGroupLinear); in other stages, and where derivatives are unavailable, it is LOD
	// 0 as before, but IMAGE_SAMPLE_B keeps its bias (LOD = bias).
	bool implicit_lod_stages = false;
	// KYTY_BUFFER_OOB_MODES=1: a bound V# with OOB_SELECT 0 and a stride (Table 35: index >= NUM_RECORDS
	// or offset >= stride) is out of range for an offset (VGPR + instruction offset, per dword) at or past
	// the stride, instead of reading the next record (IR::BufferOobOffsetCheck, a specialization bit).
	// OOB_SELECT 1 with stride 0 (index only) is not modelled: the binding range (NUM_RECORDS bytes)
	// still bounds the offset.
	bool buffer_oob_modes = false;
	// KYTY_DEPTH_BITS_READ=1 (default 0; re-applies upstream aafe014d0, which a merge dropped): a raw
	// load of a kDepth k16UInt T# reads the D16 surface's codes: the host binds it as UNorm and the
	// shader converts back (ReadsDepthBits). Before, such loads failed the depth view check (null).
	bool depth_bits_read = false;
	// KYTY_SRT_IMAGE_REACH=1 (default off; upstream 9801fd862): an image that only unreached blocks
	// use reads a zero descriptor even when its descriptor source is live through another resource,
	// and an indirect image's table key is then not evaluated (an unreached image's key can be
	// garbage, which refused the draw). Off: every image of a live source is materialized.
	bool srt_image_reach = false;
	// KYTY_LOOP_GUARD=<n> with KYTY_LOOP_GUARD_SHADERS=<hash>[,<hash>...] (hexadecimal guest shader
	// hashes): a diagnostic for a GPU hang suspected in a shader loop. Every structured loop of a
	// listed shader counts iterations against one per-invocation budget; an invocation that has
	// run more than <n> iterations takes each loop's exit edge, and at return it adds one to the
	// last GDS dword, which the command processor reports at flips ("Loop guard"). It changes the
	// guarded shaders' results when it fires. Off unless both variables are set.
	uint32_t              loop_guard_budget = 0;
	std::vector<uint64_t> loop_guard_shaders;
	// KYTY_BDA_FORCE_SHADERS=<hash>[,<hash>...] (default none): in these compute programs every raw
	// buffer access that the BDA path supports goes through BDA, also through V#s the command
	// processor could evaluate: their V# words become shader-only SRT slots (KYTY_SRT_GPU_FILL
	// leaves them to the GPU), so the command processor never reads them. For V#s that a compute
	// pass writes shortly before the listed program runs (Miles Morales' 0x275ffa68d9798b8e), where
	// reading them on the command processor waits for the GPU every frame. Writes are
	// KYTY_BDA_WRITES writes (settled per dispatch).
	std::vector<uint64_t> bda_force_shaders;
	// KYTY_INDIRECT_FORMAT_STORES_SHADERS=<hash>[,<hash>...] (default none): KYTY_INDIRECT_FORMAT_STORES
	// for these programs only (Ghost of Yotei's "Update Oct Maps" CS 0x0b0ec2bac5920bd3 writes its GI
	// volume records with BUFFER_STORE_FORMAT_XYZW through a computed V#), so one program's stores can
	// be enabled while the global switch stays off. Use IndirectFormatStoresFor(hash).
	std::vector<uint64_t> indirect_format_stores_shaders;
	[[nodiscard]] bool IndirectFormatStoresFor(uint64_t hash) const {
		return indirect_format_stores ||
		       std::find(indirect_format_stores_shaders.begin(), indirect_format_stores_shaders.end(),
		                 hash) != indirect_format_stores_shaders.end();
	}
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
	// Default on since int18 (before, only u59-preset.json set it): without it Astro Bot's tiled
	// deferred-lighting and GI kernels are dropped, and with the game's lighting patches off lit
	// geometry renders black. KYTY_SRT_VARIANT_READS=0 restores the old planning.
	bool srt_variant_reads = true;
	// KYTY_SHADER_CALLS=1 (default off in this line; on in the Insomniac title presets): compute
	// programs that call functions through S_SWAPPC_B64 have their callees expanded into them
	// (ShaderCalls.h) instead of being skipped. Programs without
	// calls are not affected either way, so it is not part of the codegen fingerprint: an expanded
	// program is a different input (its code and key carry the callees).
	bool shader_calls = false;
	// KYTY_CALL_TRACE_WAVE32=1 (default 0; on in the PPSA03671 title preset): the S_SWAPPC_B64 call
	// trace (AnalyzeShaderCalls) counts a V_CMP mask or carry-out of a wave32 program as one SGPR
	// (not a pair), and a register range that nothing before the call writes and no later write can
	// loop back to holds its entry value. PPSA03671's tiled material kernels (0x384f47083f0d5e08 and
	// three more) were skipped because a V_CMP to s11 looked like a write of their SRT pointer s12.
	// Only programs whose trace failed change; not in the fingerprint (as KYTY_SHADER_CALLS).
	bool call_trace_wave32 = false;
	// KYTY_CALL_LOOP_CAP=<n> (default 0: off; adapted from usa-ratt/KytyPS5-GT7 08197df0b,
	// 7e695ec62 and 655faf07e): a placed callee body with a back edge and no barrier
	// (Decoder::ExpandShaderCalls) is bracketed by two S_NOPs (Decoder::ShaderCallLoopBodyBegin /
	// End), and the loops whose headers lie in such a body draw on one per-invocation budget of <n>
	// rounds: once it is spent, the invocation takes each such loop's exit edge (the loops around
	// the call are left alone, so every remaining target is still called). GT7's lit material
	// functions walk a per-pixel light list whose length they read from an image; a length from data
	// the emulator does not reproduce ran up to 65,535 rounds a pixel and lost the device.
	uint32_t call_loop_cap = 0;
	// KYTY_CALL_LOOPS=0 (default 1): callee bodies with a back edge are left out of the expansions
	// (their calls take the skip path) instead of being placed.
	bool call_loops = true;
	// KYTY_CALL_WATERFALL_CLEAR=1 (default off): a call inside a waterfall that keeps its "lanes
	// left" flag in a VGPR (cleared before the call, tested after the exec restore, the loop
	// repeated: Decoder::ExpandShaderCalls) clears the flag again where the call returns, so a callee
	// that leaves it set does not repeat its lanes forever (GT7's material pass lost the device).
	bool call_waterfall_clear = false;
	// KYTY_NATIVE_INDIRECT_MESH=1|on|verify|exit: mesh draw dword 3 equal to
	// IR::PushData::MeshIndirectSentinel makes mesh shaders read their six draw dwords from the
	// parameter block at the device address in dwords 0-1 (a GPU-converted indirect mesh draw,
	// renderer/meshIndirect.h). Unset, 0 or "empty": the dwords are only ever pushed.
	bool mesh_indirect_params = false;
	// KYTY_BDA_ACCESS_TAGS=1|verify (any value but 0; default off; based on upstream KytyPS5 0ec2cd5d5):
	// get_bda_pointer also records a page fault for a backed page whose page-table entry lacks the
	// accessed tag (bit 0, BufferCache::BDA_ACCESSED) and strips the tag from the pointer. The host
	// tags the pages DMA shaders reported (bufferCacheTags.cpp). The live value of the same switch
	// picks the synchronization policy.
	bool bda_access_tags = false;
	// KYTY_BDA_TAGS_GPU_COUNT=1 (diagnostic, with bda_access_tags): get_bda_pointer also counts its
	// calls, the calls on a zero entry and the calls on an untagged entry in three words after the
	// fault buffer layout (BufferCache::BDA_TAG_COUNTER_WORD); the BdaTags line prints them.
	bool bda_tags_gpu_count = false;
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
	// KYTY_SPLIT_DPP_LANE=1 (default 0; RDNA2 ISA check 2026-11-06): in a split wave64 (KYTY_WAVE64_WARPS)
	// the EXEC test of a DPP source lane (EmitSourceLaneActive) reads the host subgroup's 32-lane ballot
	// with the source lane taken mod 32. Off, lanes 32-63 test bits 32-63 of a 32-lane ballot (zero), so
	// a BC=0 row_shr scan never writes the upper half.
	bool split_dpp_lane = false;
	// KYTY_MOV_INT_MODS=1 (default 0; ISA p.137/p.131): the neg/abs modifiers of V_MOV_B32 and
	// V_CNDMASK_B32 flip/clear the sign bit as integer XOR/AND (D.u = S0.u with the sign changed), bit
	// exact under any host float mode, instead of OpFNegate/OpFAbs. Replaces KYTY_MOV_NO_FLOAT_MODS.
	bool mov_int_mods = false;
	// KYTY_MAD_FLUSH=1 (default 0): with unfused MADs (KYTY_MAD_MODE=exact/position slice), flush f32
	// denormal operands, the rounded product and the sum to signed zero in the shader (legacy MAD has
	// no denormal support). Equivalent when the module declares DenormFlushToZero 32 (FLOAT_MODE 0xC0).
	bool mad_flush = false;
	// KYTY_F16_EXACT_ROUND=1 (default 0): f32 -> f16 results (V_CVT_F16_F32, f16 VOP1/VOP2/VOP3, MAD_MIX,
	// V_FMA_F16, f32-path packed f16) use an integer round-to-nearest-even conversion that a host
	// compiler cannot fold away against a later UnpackHalf2x16; MAD_MIXLO/HI round once (f64 Fma;
	// not in a CS with FLOAT_MODE != 0xC0, which has no Float64 path).
	bool f16_exact_round = false;
	// KYTY_MOVREL_FULL_M0=1 (default 0; ISA 3.6.1/3.7): V_MOVRELS/V_MOVRELD/V_MOVRELSD index with the
	// whole 32-bit M0 instead of M0[7:0]: an out-of-range source reads v0, an out-of-range destination
	// writes nothing.
	bool movrel_full_m0 = false;
	// KYTY_OMOD_ZERO=1 (default 0; ISA p.44): an OMOD multiply/divide turns a -0 result into +0.
	bool omod_zero = false;
	// KYTY_CVT_F32_F64_INF=1 (default 0): V_CVT_F32_F64 overflow (|x| above the f32 range after
	// rounding) gives +-INF (round to nearest, FLOAT_MODE 0xC0) instead of +-FLT_MAX.
	bool cvt_f32_f64_inf = false;
	// KYTY_BFM_DEFINED=1 (default 0): S_BFM_B32 is ((1 << S0[4:0]) - 1) << S1[4:0] computed with
	// shifts, defined (bits past 31 dropped) when offset + count > 32, instead of OpBitFieldInsert.
	bool bfm_defined = false;
	// KYTY_WAVE64_HIGH_LANES=1 (default 0): in a graphics wave64 (VS/PS) run on 32-wide host subgroups,
	// a constant V_READLANE of lane >= 32 resolves through the V_WRITELANE chain of that lane
	// (ReadLaneElimination unreachable_high_lanes), 0 on paths that never wrote it, instead of an
	// undefined subgroup shuffle id. Runtime (SGPR) lane selectors >= 32 stay undefined.
	bool wave64_high_lanes = false;
	// KYTY_LANE_REDUCTIONS=0: a V_READLANE of the last lane of a DPP row scan with every lane enabled
	// (IR::MatchLaneReduction) stays an emulated scan instead of a native subgroup reduction over the
	// row or row pair. A partly filled host subgroup lacks lanes the scan would read; the reduction
	// counts them as the operation's identity, as the guest shader's own masking makes them.
	bool lane_reductions = true;
	// KYTY_F32_FLUSH_HOST: the op classes (F32FlushClass* bits) whose f32 denormal inputs are left to
	// the host while f32_flush_host is on: all of them with =1, the classes that pass the start-up
	// self-test with "nvidia" (SettleF32FlushHost). The other classes keep the shader-side flush.
	uint8_t f32_flush_host_classes = 0x3f; // F32FlushClassAll (placed in padding: sizeof stays 256)
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
	// KYTY_INT_MRT_OUTPUTS=1 (default off; upstream 26ca60923): pixel exports to UINT and SINT render
	// targets are typed as integer vectors of the target's signedness (Vulkan requires the output
	// type to match the attachment's numeric type); compressed 16-bit exports to SINT targets are
	// sign-extended instead of unpacked as FP16. Off: only 32_ABGR exports are unsigned vectors.
	bool int_mrt_outputs = false;
	// KYTY_UFLOAT_STORE_RTZ=1 (default off; upstream 59d8256b3): formatted buffer stores to 11/10-bit
	// unsigned floats truncate (round toward zero through f16) and store negatives and NaN as zero.
	// Off: round to nearest even, KYTY_FLOAT_STORE_CLAMP overflow, NaN kept.
	bool ufloat_store_rtz = false;
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
	// KYTY_RT_SOFTWARE=auto (the default; also what an unset or empty variable means): software RT
	// on, whatever the GPU. "1" forces it on, "0" turns it off. The switch for users whose GPU or
	// driver fails the ray tracing kernels (a game that crashes right after the first lighting
	// dispatches, device loss on AMD): KYTY_RT_SOFTWARE=0 skips those dispatches again. A pipeline
	// the driver fails to build for such a kernel turns software RT off for the session on its own
	// (rtSession.h). rt_software_auto only records which spelling selected the mode.
	bool rt_software_auto = true;
	// KYTY_RT_FUNCTION=0|1|2 (with KYTY_RT_SOFTWARE; default 1): how a BVH instruction lowers. 1: the
	// node test (box and triangle tests, traversal helpers, ~100k SPIR-V words) is ONE function per
	// module that every BVH instruction calls (OpFunctionCall); a kernel with three instructions
	// was 322k words with the test inlined at each. 2: the same with the DontInline function
	// control, so the driver does not inline it again. 0: inline at every site (the pre-int16
	// code, for comparisons).
	uint32_t rt_function = 1;
	// KYTY_RT_TYPE6=0 (with KYTY_RT_SOFTWARE): node type 6 misses (four invalid children) like
	// RDNA2's user node, instead of being decoded as the PS5 shared-exponent box.
	bool rt_type6 = true;
	// KYTY_BDA_WRITES=1 (or =verify): raw stores and atomics through a V# the shader computes
	// (Psr's BVH builders) write guest memory through BDA instead of failing resource tracking. The
	// renderer settles each such dispatch synchronously: it waits for it and marks the pages it wrote
	// GPU-owned before the CP continues (BDA-WRITES-DESIGN.md). Default 1: Astro Bot's BVH builders
	// need it once its game patches are off; KYTY_BDA_WRITES=0 restores the old refusal.
	bool bda_writes = true;
	// KYTY_BDA_STORE_LOG (default 1, with bda_writes): a raw store through such a V# to a page
	// without a cache buffer, which BDA cannot reach, is appended to the fault buffer's store log
	// (BufferCache::BDA_STORE_LOG_WORD) instead of being dropped; the synchronous settle writes the
	// logged dwords into guest memory, where those bytes live (BufferCache::SettleBdaWrites). Such a
	// page gets a cache buffer only after its fault is processed, so before this every dispatch
	// until then lost its stores there (a counted consumer whose output V# takes NUM_RECORDS from a
	// GPU-written count wrote nothing). Atomics there still drop (no value to return).
	// KYTY_BDA_STORE_LOG=0 restores the drop.
	bool bda_store_log = true;
	// KYTY_BINDLESS_3D_AS_ARRAY=1 (default 0; on in the PPSA03671 title preset; adapted from
	// KytyPS5-Frank fed216ce8): a bindless 3D image instruction whose heap key holds a 2D or
	// 2D-array T# reads it as a 2D array (the GPU addresses by the T# type). The host resolves such
	// a key into the 2D-array binding with BindlessSlotArrayFlag; the shader declares both views,
	// samples both and selects by the flag. Off: such keys sample the placeholder, as before.
	// (Declared here, where it filled padding; the three switches below took sizeof(CodegenOptions)
	// to 120.)
	bool bindless_3d_as_array = false;
	// KYTY_RUNTIME_IMAGES=1 (default 0; on in the GT7 title preset; adapted from usa-ratt/KytyPS5-GT7
	// 2fa6c6514): with bindless images, a sampled image whose T# the shader loads at run time (GT7's
	// material functions read theirs through the call record's data pointer) and that is neither a
	// value the host can evaluate nor a heap entry at a key, becomes a runtime image
	// (ImageResource::runtime): the shader hashes the eight T# dwords and looks them up in the
	// runtime image table (set 1, bindings 7 and 8); a T# the table lacks samples the pending
	// placeholder and is requested, and the host resolves it like a heap key. Off: such a program
	// is skipped ("descriptor computed at runtime without a BDA path"), as before.
	bool runtime_images = false;
	// KYTY_RUNTIME_INT_IMAGES=1 (default 0; needs KYTY_RUNTIME_IMAGES; adapted from usa-ratt/
	// KytyPS5-GT7 2fa6c6514): a load (IMAGE_LOAD*, not a sample) of a runtime image whose T# has an
	// integer format reads the texture through an alias of its bindless array with that component
	// type (uint or sint), and the host resolves such T#s into the runtime heap with integer views.
	// Off: such a load reads the pending placeholder (zeros), as before.
	bool runtime_int_images = false;
	// KYTY_BINDLESS_STRIDE=1 (default 0; on in the GT7 title preset; adapted from
	// usa-ratt/KytyPS5-GT7 a6cd035f0): a bindless heap T# selected as key * stride (an immediate
	// multiply, or a shift other than 5) with a 4-byte-aligned stride of at least 32 bytes (GT7's
	// 188-byte material records) is looked up like a plain heap; the host reads key k's T# at
	// table_offset + k * stride. Off: only 32-byte T# arrays, as before.
	bool bindless_strided = false;
	// KYTY_BINDLESS_SAMPLERS=1 (default 0; on in the GT7 title preset; a minimal port of Senaxx's
	// mirrored sampler heaps, 59cb8a83e and 269b12f77): the S# beside a bindless or runtime image
	// that the shader reads from a sampler heap at a key it computes (four scalar reads of one V# at
	// key * stride + offset, stride 16 or more) samples with the game's own sampler instead of the
	// linear/wrap stand-in: the host mirrors the heap's S# records into a sampler array (set 1,
	// binding 6) and the shader indexes it with the key. Keys outside the heap, and heaps the host
	// cannot read, use the stand-in (slot 0). Off: the stand-in for every such sampler, as before.
	bool bindless_samplers = false;
	// KYTY_CALL_TRACE_REACH=1 (default 0; on in the GT7 title preset; adapted from
	// usa-ratt/KytyPS5-GT7 a6cd035f0): when the S_SWAPPC_B64 table trace finds no unique definition
	// of a register range across control flow, it asks the instruction-level control flow graph:
	// the writers from which the use can be reached; none (the range holds its entry value) or one
	// that every path from the entry passes. Only programs whose trace failed change; not in the
	// fingerprint (as KYTY_CALL_TRACE_WAVE32).
	bool call_trace_reach = false;
	// KYTY_VS_FULL_WAVE=1 (default 0): a vertex shader's merged wave info (s3) counts one full wave
	// of vertices (wave_size), as the tessellation evaluation stage's does, instead of the subgroup's
	// active invocations. Every host invocation is a vertex the guest wave would run, so the guest's
	// EXEC (built from that count) is all ones in every lane that exists; the per-lane EXEC selects
	// then fold away and the old values they kept live free their registers (NVIDIA: 216-235
	// registers in Astro Bot's skinned vertex shaders). Guest reads of EXEC as a value (S_BCNT1) count
	// the whole wave.
	bool vs_full_wave = false;
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
	// KYTY_RT_HW=1 (default off; needs VK_KHR_ray_query, see NativeRayTracing): compute programs
	// whose sce::Psr traversal loops RayTracingPatch.h recognizes trace them with host ray queries
	// over acceleration structures that mirror the guest BVHs (the guest code still decides each
	// hit). The guest loop stays as the fallback. Cleared at startup when the device cannot.
	bool rt_hw = false;
	// KYTY_RT_HW_VERIFY=1 (with KYTY_RT_HW): each patched traversal also runs the guest loop and
	// counts per ray where the results differ (NativeRayTracing reports the counters).
	bool rt_hw_verify = false;
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
	// KYTY_LDS_SLOTS=1 (default off; strxno/KytyPS5-GT7 220abc01c, adapted): a non-compute stage
	// keeps its LDS in a function-scope array of its own invocation, which the driver puts in local
	// memory because it is indexed dynamically. When every LDS access of the program is a dword
	// LoadSharedU32/WriteSharedU32 at (LaneId << 2) + a constant offset (the guest compiler's
	// per-lane VGPR spill area), lane l always reaches element l + offset / 4 of its own array: each
	// distinct offset becomes one Function variable (a register) instead (StaticLdsSlot). Inactive
	// lanes read zero and write nothing, as the array's accesses do. KYTY_LDS_SLOTS_LOG=1 prints the
	// verdict per program.
	bool lds_slots = false;
	// KYTY_FLOAT_STORE_CLAMP=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 cfa8a479e): the
	// guest GPU converts a float to a 16-bit float the D3D way, rounding toward zero, so a finite
	// value beyond the half range is stored as the largest half (65504) of its sign; the host's
	// conversion gives infinity, which one bright pixel spreads over the frame through bloom and
	// exposure. Stores into 16-bit float storage images, 32-bit exports to 16-bit float render
	// targets (ShaderPixelInputInfo::target_half_float_mask) and formatted 16-bit float buffer
	// stores clamp first; formatted 11/10-bit float stores give the largest finite value instead of
	// infinity for finite overflow.
	bool float_store_clamp = false;
	// KYTY_VTE_CNTL=1 (default off; the GT7 preset; masterzhang233/KytyPS5-GT7 df7fc6108): honour
	// PA_CL_VTE_CNTL (guest_gpu/viewportTransform.h). Position exports of a pre-divided or 1/W
	// format are made homogeneous again (ShaderClipSpaceTransform::vte_format), the viewport drops
	// the scales and offsets VTE disables, and the CLIP_DISABLE emulation's affine offset is
	// homogeneous (offset * W, minus W), which only matched before for W == 1.
	bool vte_cntl = false;
	// KYTY_SNAPSHOT_WRITTEN_GUARD=1 (default off; the GT7 preset; Senaxx/KytyPS5-GT7 2e002fb55):
	// the SRT plan's CPU snapshot of scalar loads through CPU-evaluable pointers (PlanBuilder, the
	// fork's 8d77c9242, which our line already has) leaves out a load whose pointer base is also the
	// base of a buffer or address the shader writes or updates atomically: its waves must see those
	// writes, which a snapshot taken before the dispatch does not.
	bool snapshot_written_guard = false;
	// KYTY_TESS_INDEXED=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 70341d775): the LS/HS
	// stride analysis (Tessellation.cpp) also accepts HS LDS reads whose control point is a loaded
	// index (a stencil table's) when their stride matches the LS one, folds scalar constants and
	// masks, and the lowering lets the HS read its own control-point outputs and patch constants
	// back (behind an inserted barrier), resolves addresses under conjunctions of its EXEC and
	// folds the shifted patch ordinal; HS and TES share the patch-constant range the analysis
	// found. Factor or patch-constant addresses that still do not fold skip the stage's draws.
	bool tess_indexed = false;
	// KYTY_TESS_SINGLE_CP=1 (default off; upstream 3b3e2475d): the LS/HS stride analysis of a hull
	// shader with one output control point takes the packed HS ID's low byte (the patch ordinal) as
	// the varying index, as each HS lane is then one patch, and treats stores whose address falls with
	// the patch as patch constants. Off: the patch ordinal reads as patch 0 (the behaviour before).
	bool tess_single_cp = false;
	// KYTY_WORKGROUP_IMAGES=1 (default off; the GT7 preset; adapted from upstream KytyPS5 0ee880090
	// and c23de6218, nmzik): a compute image whose T# is read from a table at
	// workgroup_id[axis] * stride + offset (GT7's CS 0xe042c5de1f18d724: two storage T#s per Z group
	// in 148-byte records) is enumerated per dispatch over that axis' workgroup count
	// (SrtRuntime::workgroup_counts): the host binds one image per group and the shader selects its
	// own group's image by the workgroup id. Off: such a program is skipped ("computes a descriptor
	// at runtime that has no BDA path"), as before.
	bool workgroup_images = false;
	// KYTY_TABLE_KEY_BOUNDS=1 (default off; adapted from upstream KytyPS5 dfcf1c0b0, 20bb268a4,
	// a3ac6d0d3 and b2053c286): an address-table image key that neither a bit scan nor a bounded
	// loop covers is enumerated up to a constant maximum proven from its value (truncated clamped
	// float arithmetic, signed/unsigned minimums, masks, shifts, waterfall ReadFirstLane of a
	// bounded lane value) or from guards on the access (key < C, key <= C); failing that, a
	// zero-based per-lane loop key takes the runtime count every uniform zero-based loop of the
	// program shares (a heuristic, logged). Off: such programs are skipped or go bindless, as before.
	bool table_key_bounds = false;
	// GPU-selected image keys from upstream KytyPS5's selector proofs (all default off; bits
	// SelectorKeys*). Off: such programs are enumerated over the whole material buffer, go bindless
	// or are skipped, as before.
	// - KYTY_SELECTOR_BITSCAN=1 (f297c0bfe): a scalar material record whose index a bit scan of a
	//   nonzero value (or another proven constant maximum) selects reads only records 0..maximum.
	// - KYTY_SELECTOR_EXCLUDED=1 (f297c0bfe): a material key that a uniform guard rules out on every
	//   path to the image is left out of the enumeration.
	// - KYTY_SELECTOR_MASK_RANKS=1 (99f4eba22, with BITSCAN): the bit-scanned value keeps only bits of
	//   one scalar mask load, so only the records at the ranks some reachable mask word sets are read.
	// - KYTY_SELECTOR_SHARED_LIST=1 (0e9c77b0a, ffb6ad6ca): a material record selected by
	//   V_READFIRSTLANE from a one-wave LDS append list is bounded by the list's producer bound.
	// - KYTY_SELECTOR_BITSET=1 (49ebbfd62 part, 145578ac0, 914a3bb4b): an address-table key
	//   base - V_READFIRSTLANE(rank) is a reversed range; a rank built from four- or eight-word light
	//   bitmaps reads only the keys at the set ranks of the bitmap buffer.
	uint8_t selector_keys = 0;
	// KYTY_NATIVE_WAVE64_BRANCH=1 (default off; Senaxx 49723097f): a wave64 compute or mesh program
	// on a 64-wide host subgroup (AMD) takes a ballot for its VCC/EXEC/SCC branches, so the wave
	// branches as one, as the emulated two-half path does on 32-wide hosts. Off: each lane branches
	// on its own bit there (no effect on NVIDIA, whose subgroups are 32 wide).
	bool native_wave64_branch = false;
	// KYTY_SAMPLER_READS_DEFER=1 (default off; the GT7 preset after testing): with bindless images, an
	// S_BUFFER_LOAD S# read that feeds only sampler handles is collected after the walk (and only if an
	// ordinary sampler still uses it), so a bindless or stand-in sampler plan no longer strands it as a
	// buffer resource whose V# words dead code elimination removed (materialization failed). Off: the
	// walk collects it in place, as before (other titles' binding layouts stay as they were).
	bool sampler_reads_deferred = false;
	// KYTY_PS_FLUSH / KYTY_CS_FLUSH / KYTY_VS_FLUSH=off (default exact; strxno/KytyPS5-GT7 220abc01c,
	// 39a4bc75b): on a device without DenormFlushToZero for f32 (the RTX 3090 and 4090 drivers),
	// skip the shader-side denormal flushes in pixel, compute or geometry (vertex, local, mesh,
	// tessellation) programs. Our MADs never flush; the remaining flushes are the inputs of
	// rcp/rsq/sqrt/exp/log, compute float compares with FLOAT_MODE denormal inputs off, and f32<->f64
	// conversions. Off, a denormal there stays denormal instead of reading as a signed zero
	// (rcp(denormal) is then a large finite value, not an infinity). Bits: DenormFlushSkip*.
	uint8_t denorm_flush_skip = 0;
	// KYTY_PS_EXEC_WORDS=1 (adapted from upstream 5429888d4): a pixel shader's EXEC and VCC words
	// leave out the helper lanes the per-lane EXEC keeps for derivatives: the entry words are the
	// live lanes, VALU masks are limited to the EXEC words, SAVEEXEC combines the words bitwise.
	bool ps_exec_words = false;
	// KYTY_FRACT_CLAMP=1 (default off; upstream e864cb20b): V_FRACT_F16/F32/F64 results are clamped
	// below one (the largest value under 1.0 in the destination precision), as the hardware returns
	// for a tiny negative source where x - floor(x) rounds to 1.0. One compare and select per fract.
	bool fract_clamp = false;
};
inline constexpr uint8_t SelectorKeysBitscan    = 1u;
inline constexpr uint8_t SelectorKeysExcluded   = 2u;
inline constexpr uint8_t SelectorKeysMaskRanks  = 4u;
inline constexpr uint8_t SelectorKeysSharedList = 8u;
inline constexpr uint8_t SelectorKeysBitset     = 16u;
inline constexpr uint8_t DenormFlushSkipPixel    = 1u;
inline constexpr uint8_t DenormFlushSkipCompute  = 2u;
inline constexpr uint8_t DenormFlushSkipGeometry = 4u;
// CodegenOptions::f32_flush_host_classes.
inline constexpr uint8_t F32FlushClassRcp     = 1u;  // V_RCP_F32 (OpFDiv 1.0/x)
inline constexpr uint8_t F32FlushClassRsq     = 2u;  // V_RSQ_F32 (InverseSqrt)
inline constexpr uint8_t F32FlushClassSqrt    = 4u;  // V_SQRT_F32 (Sqrt)
inline constexpr uint8_t F32FlushClassExp     = 8u;  // V_EXP_F32 (Exp2)
inline constexpr uint8_t F32FlushClassLog     = 16u; // V_LOG_F32 (Log2)
inline constexpr uint8_t F32FlushClassCompare = 32u; // f32 compares of two register operands
inline constexpr uint8_t F32FlushClassAll     = 63u;
// Not an op class: the self-test chose its stage B (modules leave DenormPreserve 16 out; the host
// float controls say so too). Part of the fingerprint through f32_flush_host_classes.
inline constexpr uint8_t F32FlushClassNoPreserve16 = 64u;

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
// KYTY_NULL_BUFFER_SHARE: the shader-data byte offset of a buffer slot bound as the null buffer (an
// 8-bit field; real offsets stay below the storage/uniform offset alignment).
inline constexpr uint32_t NullBufferMarkerOffset = 255;
inline constexpr uint32_t RtNodeBudgetGdsFromEnd = 2;
inline constexpr uint32_t RtNodeStatsGdsFromEnd  = 3;
inline constexpr uint32_t RtNodeStatsBins        = 24;
// KYTY_LOOP_GUARD: the low and high dwords of the last exhausted shader's hash, past the RT bins.
inline constexpr uint32_t LoopGuardHashGdsFromEnd = RtNodeStatsGdsFromEnd + RtNodeStatsBins + 1;

// True when KYTY_LOOP_GUARD applies to the guest shader with this hash.
[[nodiscard]] bool LoopGuardApplies(uint64_t shader_hash);

[[nodiscard]] const CodegenOptions& GetCodegenOptions();
// KYTY_FORMAT_MISSING_CANON: logs a formatted access whose V# dst_sel names a channel the format
// lacks (the first 32 programs/formats, then a running count). format: Prospero::BufferFormat.
void NoteFormattedSelectorMissing(uint64_t shader_hash, uint32_t format, uint32_t swizzle,
                                  const char* where);
// KYTY_LANE_LOCAL needs every compute subgroup of a wave64 program to be exactly 32 lanes: a device
// whose subgroup size is fixed at 32, or one whose compute pipelines require 32 (the device layer sets
// it once; false until then, which keeps lane-local programs off).
void               SetHostComputeSubgroups32(bool fixed_32);
[[nodiscard]] bool GetHostComputeSubgroups32();
// KYTY_UNIFORM_BUFFERS needs a device that reads 0 for a uniform dword past the bound range
// (Spirv::HostBufferRobustness::uniform_dword_loads_return_zero, mirrored here by
// Spirv::SetHostBufferRobustness for IR::UniformBufferResources).
void               SetHostUniformBufferRobust(bool robust);
[[nodiscard]] bool GetHostUniformBufferRobust();
// KYTY_NATIVE_F16 needs shaderFloat16 (enabled by the device layer) plus RoundingModeRTE and
// SignedZeroInfNanPreserve for 16-bit floats, independent of the 32/64-bit modes.
void               SetHostNativeHalf(bool supported);
[[nodiscard]] bool GetHostNativeHalf();
// Whether KYTY_NATIVE_F16 applies (option and device).
[[nodiscard]] bool NativeHalfActive();
// Test hook: replaces the options for subsequent compilations. Not thread-safe; call it only
// while no shader is being compiled.
// Diagnostic per-program MadMode::Exact (KYTY_MAD_EXACT_SHADERS, parsed by the pipeline cache,
// which also salts the program disk-cache key of every program it selects, so the codegen
// fingerprint and the other programs' cache entries stay as they are): a selected program is
// emitted as with KYTY_MAD_MODE=exact while the others keep mad_mode. Selection: `stage_mask`
// (bit = static_cast<uint32_t>(ShaderType)) and, when `hashes` is non-empty or `hash_mask` is
// non-zero, a hash in `hashes` or (hash & hash_mask) == hash_value. Inactive by default.
struct MadExactFilter {
	bool                  active     = false;
	uint32_t              stage_mask = 0;
	std::vector<uint64_t> hashes;
	uint64_t              hash_mask  = 0;
	uint64_t              hash_value = 0;
};
void SetMadExactFilter(MadExactFilter filter);
// The program uses MadMode::Exact because of the filter (false while it is inactive).
[[nodiscard]] bool MadExactProgram(uint32_t stage, uint64_t shader_hash);
void SetCodegenOptions(const CodegenOptions& options);
// KYTY_F32_FLUSH_HOST, KYTY_VS_FULL_WAVE and KYTY_LDS_SLOTS take the value "nvidia": on only when
// the selected device is an NVIDIA one (vendor 0x10de), off otherwise. Called by the device layer
// once the device is chosen, before any program is translated or fingerprinted.
void ApplyHostVendor(uint32_t vendor_id);
// KYTY_F32_FLUSH_HOST=nvidia on an NVIDIA device is on only until the device layer's start-up GPU
// self-test (F32FlushSelfTest.h) has checked the host behaviour it relies on: true from
// ApplyHostVendor until SettleF32FlushHost, which keeps the host flush for the op classes that
// passed (F32FlushClass* bits; 0 turns the switch off), once, before any program is translated or
// fingerprinted. KYTY_F32_FLUSH_HOST=1 stays on
// without a test.
[[nodiscard]] bool F32FlushHostAwaitsSelfTest();
void               SettleF32FlushHost(uint8_t classes);

// The BVH lowering mode in effect on this thread: rt_function, unless this thread overrides it
// (KYTY_RT_DUMP_DIR also emits each RT module inline, mode 0, for size and build-time comparisons).
[[nodiscard]] uint32_t RtFunctionMode();
// -1 clears the override.
void SetThreadRtFunctionOverride(int mode);

} // namespace Libs::Graphics::ShaderRecompiler

#endif // EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_CODEGENOPTIONS_H_
