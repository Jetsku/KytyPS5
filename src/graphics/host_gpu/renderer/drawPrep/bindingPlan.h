#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BINDINGPLAN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BINDINGPLAN_H_

#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <vector>

// Draw-prep P4b (KYTY_DRAW_PREP_BINDINGS, Profiling/analysis/P4B-WORKER-LOOKUPS.md): the thread
// that prepared a draw also computes its binding plan, the parts of the binding and pipeline
// resolution that are pure functions of the draw's register snapshot and its preparation, or
// lookups in tables that change only with a generation. The command processor uses a plan item in
// place of computing it only when the preparation was committed (DrawPrep::Validate accepted it:
// the serial path then derives the same values from the same inputs) and the item's certificate
// holds:
//  - V# and vertex ranges (decode, ClampRangeSize): the guest virtual-range generation read before
//    the preparing thread's first clamp is unchanged (LibKernel::Memory::VirtualRangesGeneration);
//  - pipeline: the draw's resolved targets, topology, restart flag and plain-pixel choice are the
//    ones its key was built for, and PipelineCache::PipelineGeneration is the one the object was
//    found under (objects are never freed; a replaced one bumps the generation);
//  - sampler handles: permanent (the sampler cache never evicts);
//  - shader-data user dwords and mip-statistics fields, texture memo hashes, program flags, the
//    target export mapping: pure.
// A preparing thread never calls into the buffer or texture cache, the tracker, the scheduler or
// Vulkan objects, never writes a shared memo and never exits or logs: where the serial code would,
// it leaves the item to the command processor, which runs today's code in order. Every side effect
// (FindBuffer, ObtainBuffer and its memo touches, texture resolution, uploads, pipeline creation)
// stays on the command processor.
//
// KYTY_DRAW_PREP_BINDINGS=0 (default) | 1 (every part) | a comma list of parts: pipeline, buffers,
// userdata, samplers, textures, statics (P4b-1); hwcheck, dynamic (P4b-1b); texturememo (P4b-2). Plans come from the preparing worker threads (and the
// command processor's KYTY_DRAW_PREP_STEAL preparations); a head the command processor prepares
// itself has none. KYTY_DRAW_PREP=inline computes them on the command processor (tests).
// KYTY_DRAW_PREP_BINDINGS_VERIFY=1|exit: every plan item the command processor would use is also
// computed the serial way and compared; the serial value is used (DrawPrepBindingVerify*).
namespace Libs::Graphics {

struct DrawIndexArgs;
struct GraphicContext;
class SamplerCache;
class TextureBindingMemo;

namespace DrawPrep {

struct RegisterSnapshot;
struct PreparedDraw;
struct BindingPlanContext;

enum class BindingPart : uint32_t {
	Pipeline = 1u << 0u,
	Buffers  = 1u << 1u, // V# ranges (FindBuffers) and vertex-buffer ranges
	UserData = 1u << 2u, // shader-data user dwords, mip-statistics fields
	Samplers = 1u << 3u,
	Textures = 1u << 4u, // texture memo hashes
	Statics  = 1u << 5u, // program flags (shader writes, barrier sinking, scissor union), export map
	HwCheck  = 1u << 6u, // P4b-1b: uc_check/hw_check verdict
	Dynamic  = 1u << 7u, // P4b-1b: dynamic viewports and scissors
	TextureMemo = 1u << 8u, // P4b-2: texture memo hints, runs of memo hits under one lock
};
inline constexpr uint32_t AllBindingParts = (1u << 9u) - 1u;

// 0: off.
[[nodiscard]] uint32_t BindingParts();
void PrefetchBindingPipeline(const BindingPlanContext& context, const RegisterSnapshot& registers,
                             const DrawIndexArgs* index_args, const PreparedDraw& prepared);
[[nodiscard]] inline bool BindingPartEnabled(uint32_t parts, BindingPart part) {
	return (parts & static_cast<uint32_t>(part)) != 0;
}
// 0 off, 1 count/log, 2 exit on the first mismatch.
[[nodiscard]] int BindingsVerifyMode();
// Verify mode: one plan item compared with its serial value (DrawPrepBindingVerifyChecks).
void CountBindingVerifyCheck();
// A verify-mode difference: counted, logged (first 32), and fatal in exit mode.
void ReportBindingMismatch(const char* item, uint32_t detail = 0);

// Process-wide totals, always counted on the command processor (the DrawPrepBinding* frame events
// need a connected profiler); tests read them.
struct BindingTotals {
	std::atomic<uint64_t> plans {0};          // valid plans that reached their commit
	std::atomic<uint64_t> used {0};           // commits whose preparation was validated
	std::atomic<uint64_t> pipelines_used {0}; // pipelines taken from plans
	std::atomic<uint64_t> ranges_used {0};    // stages whose V# ranges came from plans
	std::atomic<uint64_t> shader_data_used {0};
	std::atomic<uint64_t> samplers_used {0};  // sampler handles taken from plans
	std::atomic<uint64_t> vertex_ranges_used {0};
	std::atomic<uint64_t> statics_used {0};
	std::atomic<uint64_t> hw_checks_skipped {0}; // P4b-1b
	std::atomic<uint64_t> viewports_used {0};    // P4b-1b
	std::atomic<uint64_t> texture_run_hits {0};  // P4b-2: bindings resolved in runs
	std::atomic<uint64_t> view_run_hits {0};     // P4b-2: views acquired in runs
	// Pipelines left to the command processor by the preparing threads (not created yet, or a key
	// the serial path refuses; the map lock busy), and plan pipelines whose certificate failed.
	std::atomic<uint64_t> pipeline_abstains {0};
	std::atomic<uint64_t> pipeline_busy {0};
	std::atomic<uint64_t> pipeline_fallbacks {0};
	std::atomic<uint64_t> verify_checks {0};
	std::atomic<uint64_t> verify_mismatches {0};
};
[[nodiscard]] BindingTotals& GetBindingTotals();

// One shader stage's plan (the vertex stage of a draw without tessellation, or the pixel stage).
struct StagePlan {
	struct Range {
		uint64_t address = 0; // 0 with size 0: a null binding
		uint64_t size    = 0; // the V# size, which ClampRangeSize leaves unchanged
		uint32_t out_of_bounds_mode = 0; // BufferOutOfBoundsCombination (NoteBufferOutOfBoundsMode)
	};
	// FindBuffers: every V# of the program, in order.
	bool               ranges_valid = false;
	std::vector<Range> ranges;
	// PrepareBindings/RebindBuffers: the stage's shader data with the user dwords and the
	// mip-statistics fields filled in and the memory-offset dwords zero.
	bool                  shader_data_valid = false;
	std::vector<uint32_t> shader_data;
	// Whether some image of the stage has a mip-statistics counter (RebindBuffers' fields).
	bool                  mip_stats_active = false;
	// PrepareBindings: one per program sampler; null where the command processor resolves it.
	bool                     samplers_valid = false;
	std::vector<vk::Sampler> samplers;
	// PrepareBindings: TextureBindingMemo::Hash of every image binding's key.
	bool                  texture_hashes_valid = false;
	std::vector<uint64_t> texture_hashes;
	// P4b-2: the memo entry tag found for every image binding's key (0: none).
	bool                  texture_tags_valid = false;
	std::vector<uint64_t> texture_tags;

	void Reset() {
		ranges_valid         = false;
		shader_data_valid    = false;
		mip_stats_active     = false;
		samplers_valid       = false;
		texture_hashes_valid = false;
		texture_tags_valid   = false;
	}
};

// AcquireVertexBuffers' guest ranges, without its buffer-cache acquisition.
struct VertexRangePlan {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;
	struct Merged {
		uint64_t base_address  = 0;
		uint64_t requested_end = 0;
		uint64_t acquired_end  = 0; // base + ClampRangeSize (a plan's equals requested_end)
	};
	bool                             valid        = false;
	uint32_t                         buffer_count = 0; // vs_input_info.buffers_num
	uint32_t                         merged_count = 0;
	std::array<uint64_t, MaxBuffers> sizes {};        // per vertex buffer; 0: a null binding
	std::array<Merged, MaxBuffers>   merged {};       // sorted by base, overlapping ones merged
	std::array<uint8_t, MaxBuffers>  merged_index {}; // per non-empty buffer: its merged range
};

// P4b-1b dynamic: SetGraphicsDynamicParams' viewports (an empty slot's width already made
// positive, marked in empty_mask) and its scissors before the framebuffer clamp.
struct DynamicViewportPlan {
	static constexpr uint32_t MaxViewports = 16;
	bool                                    valid      = false;
	uint32_t                                count      = 0;
	uint32_t                                empty_mask = 0;
	std::array<vk::Viewport, MaxViewports>  viewports {};
	std::array<ScissorRect, MaxViewports>   scissors {};
};

struct BindingPlan {
	// Computed from its slot's snapshot and preparation; reset after every commit.
	bool      valid = false;
	uint32_t  parts = 0; // BindingParts() when computed
	// LibKernel::Memory::VirtualRangesGeneration before the first clamp.
	uint64_t  vm_generation = 0;
	bool      pixel_active  = false;
	StagePlan vertex;
	StagePlan pixel;
	VertexRangePlan vertex_ranges;
	// KYTY_LOD_STATS_PLAIN_VARIANT=on: the draw uses the pixel program without mip statistics.
	bool plain_pixel = false;
	// Pipeline: the object GetGraphicsPipeline returns for the draw and the generation it was
	// found under (null: the command processor looks it up), and the inputs its key took that are
	// only known at commit: the resolved targets as predicted from the registers, the topology
	// and the primitive-restart flag.
	const PipelineCache::Pipeline*  pipeline            = nullptr;
	uint64_t                        pipeline_generation = 0;
	PipelineCache::PipelineTargets  targets;
	vk::PrimitiveTopology           topology          = vk::PrimitiveTopology::ePointList;
	bool                            primitive_restart = false;
	// Statics: ExecutePreparedDraw's shader write stages, DrawIsBarrierSafe's program conditions
	// (for plain_pixel above) and the scissor union of KYTY_ALIAS_BYTES (written_valid).
	bool                   statics_valid         = false;
	vk::PipelineStageFlags shader_write_stages   = {};
	bool                   programs_barrier_safe = false;
	bool                   written_valid         = false;
	vk::Rect2D             written {};
	// P4b-1b hwcheck: uc_check and hw_check would neither stop the emulator nor log (hw_checks_quiet).
	bool                hw_checks_quiet = false;
	DynamicViewportPlan viewports;

	void Reset() {
		valid         = false;
		parts         = 0;
		vm_generation = 0;
		pixel_active  = false;
		vertex.Reset();
		pixel.Reset();
		vertex_ranges.valid = false;
		plain_pixel         = false;
		pipeline            = nullptr;
		pipeline_generation = 0;
		statics_valid       = false;
		written_valid       = false;
		hw_checks_quiet     = false;
		viewports.valid     = false;
	}
};

// What a preparing thread may use besides the slot: lookups only.
struct BindingPlanContext {
	PipelineCache*            pipelines    = nullptr;
	SamplerCache*             samplers     = nullptr;
	const GraphicContext*     graphics     = nullptr;
	const TextureBindingMemo* texture_memo = nullptr; // FindHint only
};

// The plan of a draw whose preparation succeeded (prepared.ok), on the preparing thread, from the
// same register snapshot. index_args: the arguments of an indexed draw (null for DrawIndexAuto).
// Leaves the plan invalid when the switch is off or the preparation failed.
void ComputeBindingPlan(const BindingPlanContext& context, const RegisterSnapshot& registers,
                        const DrawIndexArgs* index_args, const PreparedDraw& prepared,
                        BindingPlan& plan);
// The engine, after committing a draw that had a valid plan: whether the draw used it
// (DrawPrepBindingPlansUsed) or dropped it (DrawPrepBindingPlansDropped).
void CountCommittedPlan(bool used);

// The command processor, where ExecutePreparedDraw looks its pipeline up: the plan's pipeline
// when its certificate holds (the draw's targets, topology, restart flag and plain-pixel choice
// are the ones the key was built for, and the pipeline generation is unchanged); null otherwise
// (counted as DrawPrepBindingFallbackPipeline when the plan had one).
[[nodiscard]] const PipelineCache::Pipeline*
PlannedPipeline(const PipelineCache& cache, const BindingPlan& plan,
                std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
                vk::PrimitiveTopology topology, bool primitive_restart_enable, bool plain_pixel);

// Shared with the serial draw path (renderDraw.cpp), so the two cannot differ:
// GetDrawTopology (false: nothing is drawn).
[[nodiscard]] bool DrawTopology(const HW::UserConfig& user_config, vk::PrimitiveTopology& topology);
// ResolvePrimitiveRestart without its index scan: Scan when only the scan can decide (a custom
// reset index), Unsupported where it stops the emulator.
enum class RestartDecision : uint8_t { Disabled, Enabled, Scan, Unsupported };
[[nodiscard]] RestartDecision DecidePrimitiveRestart(const HW::UserConfig& user_config,
                                                     const HW::Context& registers,
                                                     uint32_t element_size, bool allow_custom);
// The restart index of a Scan decision.
[[nodiscard]] uint32_t PrimitiveRestartIndex(const HW::Context& registers, uint32_t element_size);
// KYTY_MESH_RESTART (renderDraw.cpp): mesh draws accept a custom restart index.
[[nodiscard]] bool MeshPrimitiveRestartEnabled();
// AcquireVertexBuffers' guest ranges (sizes, sorted and merged ranges, their requested ends; the
// acquired ends and merged indices are left to the caller). False where it stops the emulator
// (an invalid range), with `invalid` the offending buffer.
[[nodiscard]] bool CollectVertexRanges(const ShaderVertexInputInfo& info, VertexRangePlan& ranges,
                                       uint32_t& invalid);
// The per-buffer merged range of AcquireVertexBuffers (the first one holding the buffer's address
// below its acquired end); false when none does.
[[nodiscard]] bool AssignVertexRanges(const ShaderVertexInputInfo& info, VertexRangePlan& ranges,
                                      uint32_t& unassigned);
// SetGraphicsDynamicParams' viewports and unclamped scissors (P4b-1b); false where a scissor's
// clip-rect rule is unsupported (the serial path reports it).
[[nodiscard]] bool PlanDynamicViewports(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                        bool indexed_viewports, DynamicViewportPlan& plan);
// DrawIsBarrierSafe's conditions on a stage's program (the GDS buffer and the images are checked
// at commit).
[[nodiscard]] bool ProgramBarrierSafe(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                      bool plain_pixel);

} // namespace DrawPrep
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_BINDINGPLAN_H_
