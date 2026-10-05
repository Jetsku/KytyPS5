#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFastFirst.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class ProgramDiskCache;
class ShaderJournal;
class ShaderPrecompiler;
struct RenderColorInfo;
struct RenderDepthInfo;
class CommandBuffer;

namespace HW {
class Context;
class Shader;
class UserConfig;
struct ComputeShaderInfo;
} // namespace HW

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       depth_bounds_test_enable = false;
	float                      depth_min_bounds         = 0.0f;
	float                      depth_max_bounds         = 0.0f;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       cull_front                                         = false;
	bool                       cull_back                                          = false;
	bool                       face                                               = false;
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
	// Target 0 blends with logical alpha from the second source (KYTY_BLEND_ALPHA_REMAP).
	bool                       blend_alpha_source_remap                           = false;

	bool operator==(const PipelineStaticParameters& other) const noexcept;
};

#pragma pack(pop)

static_assert(std::is_trivially_copyable_v<PipelineStaticParameters>);
static_assert(std::is_standard_layout_v<PipelineStaticParameters>);
static_assert(alignof(PipelineStaticParameters) == 1);
static_assert(sizeof(PipelineStaticParameters) == 126);

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;

	bool operator==(const PipelineRenderingState&) const = default;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride                           = 0;
		bool     instance                         = false;
		bool     operator==(const Binding&) const = default;
	};
	struct Attribute {
		uint32_t offset                             = 0;
		uint8_t  binding                            = 0;
		bool     operator==(const Attribute&) const = default;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;

	bool operator==(const PipelineVertexInputState&) const = default;
};

struct ShaderProgram {
	uint64_t         id     = 0;
	vk::ShaderModule module = nullptr;

	explicit operator bool() const { return id != 0 && module != nullptr; }
};

class PipelineCache {
public:
	explicit PipelineCache(GraphicContext& graphics);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Final save of the driver pipeline cache (exit). Stops the periodic saver first, then
	// destroys the driver cache; later pipelines are created without one. Also writes the
	// persistent program cache's pending records (it stays usable).
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		bool                    uses_push_descriptors = false;
	};

	struct GraphicsPrograms {
		std::array<ShaderProgram, 3> vertex;
		ShaderProgram pixel;

		[[nodiscard]] uint32_t VertexStageCount() const { return vertex[1] ? 3u : 1u; }
	};

	// One compiled module of a shader source; owned by the program cache and never moved or
	// freed before the cache is destroyed.
	struct Permutation;

	// Output of one stage's program preparation, owned by the draw (or dispatch) rather than by
	// the shared cache entry. The stage runtime's `resources` points into it, so it must outlive
	// the draw's binding and commit. Reusing one object keeps its vectors' capacity.
	struct StagePrep {
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const Permutation*                           permutation = nullptr;
	};

	struct GraphicsStagePreps {
		std::array<StagePrep, 3> vertex;
		StagePrep                pixel;
	};

	// GetGraphicsPrograms prepares LS/HS/TES (all three vertex_info entries) exactly when this
	// holds; otherwise it only writes vertex_info[0].
	[[nodiscard]] static bool TessellationActive(const HW::UserConfig& user_config);

	// KYTY_LOD_STATS_PLAIN_VARIANT: the prepared pixel program compiled without GET_LOD_STATS
	// feedback (same bindings and pipeline layout), or an empty program when it has none.
	[[nodiscard]] static ShaderProgram PlainPixelProgram(const StagePrep& prep);

	GraphicsPrograms
	GetGraphicsPrograms(const HW::VertexShaderInfo& vertex_regs,
	                    const HW::PixelShaderInfo& pixel_regs, const HW::ShaderRegisters& sh,
	                    const HW::Context& context, const HW::UserConfig& user_config,
	                    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	                    bool pixel_active, std::array<ShaderVertexInputInfo, 3>& vertex_info,
	                    ShaderPixelInputInfo& pixel_info, GraphicsStagePreps& stage_preps);
	// Draw-prep: the program preparation of one draw as GetGraphicsPrograms would do it, but
	// speculative: every guest read goes through the active DrawPrep recorder (readSet.h), nothing
	// is synchronized or read back. Normally uses only published programs; the optional
	// KYTY_PIPELINE_PREFETCH_PROGRAMS also publishes pure compilations from clean snapshots.
	// Safe on any thread. Writes vertex_info (stage 0),
	// pixel_info, the two stage preps and programs; outputs are meaningful only for Ok.
	enum class SpeculativeResult : uint8_t {
		Ok,
		Ineligible,   // tessellation or O15 resource reuse: always serial
		NotPublished, // a source or permutation is missing (the serial path compiles it)
		ReadFailed,   // a read was refused; the recorder holds the reason
	};
	[[nodiscard]] SpeculativeResult PrepareGraphicsProgramsSpeculative(
	    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
	    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
	    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
	    bool pixel_active, ShaderVertexInputInfo& vertex_info, ShaderPixelInputInfo& pixel_info,
	    StagePrep& vertex_prep, StagePrep& pixel_prep, GraphicsPrograms& programs,
	    uint64_t* compile_ns = nullptr);
	ShaderProgram GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                const HW::ShaderRegisters&   sh,
	                                ShaderComputeInputInfo& input_info, StagePrep& stage_prep);

	Pipeline& GetGraphicsPipeline(std::span<const RenderColorInfo>       colors,
	                              const RenderDepthInfo&                 depth,
	                              std::span<const ShaderVertexInputInfo> vertex_info,
	                              CommandBuffer& command, const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs);

	// Draw-prep binding plans (KYTY_DRAW_PREP_BINDINGS, drawPrep/bindingPlan.h). Everything the
	// graphics pipeline key takes from a draw's resolved colour and depth targets: the key is a
	// function of this, the draw's registers, its vertex and pixel interfaces, its programs, its
	// topology and its primitive-restart flag.
	struct PipelineTargets {
		struct Color {
			uint32_t                        slot    = 0;
			vk::Format                      format  = vk::Format::eUndefined;
			uint32_t                        samples = 0;
			Prospero::ColorComponentMapping export_mapping {};
		};
		std::array<Color, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
		uint32_t                                        color_count = 0;
		// A depth attachment (a view format and an image), its view format and sample count.
		bool       with_depth    = false;
		vk::Format depth_format  = vk::Format::eUndefined;
		uint32_t   depth_samples = 0;
		// RenderDepthInfo's depth-bounds state (whether or not there is a depth attachment).
		bool  depth_bounds_test_enable = false;
		float depth_min_bounds         = 0.0f;
		float depth_max_bounds         = 0.0f;
	};
	// Whether resolved targets are the ones `targets` describes (floats compared by their bits).
	[[nodiscard]] static bool SamePipelineTargets(const PipelineTargets&           targets,
	                                              std::span<const RenderColorInfo> colors,
	                                              const RenderDepthInfo&           depth);
	// Early exact-key compilation. Speculation only warms a pipeline; the ordered draw still
	// obtains its actual key and waits for completion. No draw is deferred or discarded.
	[[nodiscard]] bool PipelinePrefetchEnabled() const noexcept;
	struct PrefetchTotals {
		uint64_t submitted = 0, used = 0, compile_ns = 0, wait_ns = 0, max_wait_ns = 0;
		uint64_t programs = 0;
	};
	[[nodiscard]] PrefetchTotals GetPrefetchTotals() const;
	// KYTY_PIPELINE_FAST_FIRST counters (all zero when it is off); tests and diagnostics.
	[[nodiscard]] FastFirstSnapshot GetFastFirstTotals() const;
	void NoteProgramPrefetchWait(uint64_t ns);
	void PrefetchGraphicsPipeline(const PipelineTargets& targets, const HW::Context& ctx,
	    const HW::UserConfig& user_config, const ShaderVertexInputInfo& vertex_info,
	    const ShaderPixelInputInfo* pixel_info, vk::PrimitiveTopology topology,
	    bool primitive_restart_enable, const GraphicsPrograms& programs);
	void PrefetchGraphicsPipeline(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
	    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
	    const ShaderPixelInputInfo* pixel_info, vk::PrimitiveTopology topology,
	    bool primitive_restart_enable, const GraphicsPrograms& programs);
	enum class PlanLookup : uint8_t {
		Found,
		Absent,      // no pipeline for the key yet (GetGraphicsPipeline creates it)
		Busy,        // the map lock is held (a creation can hold it for tens of milliseconds)
		Unsupported, // GetGraphicsPipeline would stop the emulator for these inputs
	};
	// On a draw-prep thread: the pipeline GetGraphicsPipeline returns for a draw with these inputs
	// when it exists already, and the generation it was found under (PipelineGeneration). Never
	// creates, waits for the map lock (it is only tried), logs or exits; a small per-thread memo
	// answers repeated keys without the lock.
	[[nodiscard]] PlanLookup FindGraphicsPipelineForPlan(
	    const PipelineTargets& targets, const HW::Context& ctx, const HW::UserConfig& user_config,
	    const ShaderVertexInputInfo& vs_input_info, const ShaderPixelInputInfo* ps_input_info,
	    vk::PrimitiveTopology topology, bool primitive_restart_enable,
	    const GraphicsPrograms& programs, const Pipeline*& pipeline, uint64_t& generation);
	// Advanced whenever a cached pipeline object is replaced. Unchanged since a lookup: the
	// lookup's object is still the one the map holds for its key.
	[[nodiscard]] uint64_t PipelineGeneration() const noexcept {
		return m_pipeline_generation.load(std::memory_order_acquire);
	}
	// The command processor taking a plan's pipeline in place of GetGraphicsPipeline: what that
	// lookup does besides finding the object (the EXEC_ON_NOOP note, the compile-stall report).
	void NotePlannedPipeline(const RenderDepthInfo& depth, const ShaderPixelInputInfo* ps_input_info);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info,
	                             const ShaderProgram&          compute_program);
	// The pipeline a compute dispatch runs, or null when the dispatch must be skipped. Programs
	// without software BVH instructions get GetComputePipeline's (built on this thread). A
	// software ray-tracing kernel (program.info.uses_bvh, KYTY_RT_SOFTWARE) builds on a background
	// thread: its first dispatch waits up to KYTY_RT_PIPELINE_WAIT_MS (default 250) for it, then
	// every dispatch is skipped until it is ready, so a cold 300k-word compile (17 s on an RTX 50
	// GPU) never freezes the command processor; the lighting it feeds lags. Null also, for every
	// RT kernel, once any RT pipeline failed to build (rtSession.h: software RT is then off for the
	// session, logged once). KYTY_RT_PIPELINE_ASYNC=0 builds in the background but waits for it.
	[[nodiscard]] Pipeline* TryGetComputePipeline(const ShaderComputeInputInfo& input_info,
	                                              const ShaderProgram&          compute_program);
	// Starts the background build of an RT kernel's pipeline without waiting (the shader precompile
	// replay: the game's first dispatch then finds it built). Does nothing for other programs.
	void PrewarmComputePipeline(const ShaderComputeInputInfo& input_info,
	                            const ShaderProgram&          compute_program);
	// RT kernels' pipelines: published, failed, and the time the CP waited for them (tests, log).
	struct RtPipelineTotals {
		uint64_t requested = 0, published = 0, failed = 0, skipped_dispatches = 0, waited = 0;
		uint64_t wait_ns = 0, build_ns = 0, max_build_ns = 0;
	};
	[[nodiscard]] RtPipelineTotals GetRtPipelineTotals() const;

	// Process-wide totals of the program caches (tests and diagnostics): permutations created
	// (emitted or reloaded), TranslateProgram runs, and the persistent program cache's reloads and
	// verify-mode comparisons (KYTY_PROGRAM_CACHE, programDiskCache.h).
	struct ProgramTotals {
		uint64_t programs          = 0;
		uint64_t translations      = 0;
		uint64_t source_hits       = 0;
		uint64_t permutation_hits  = 0;
		uint64_t verify_checks     = 0;
		uint64_t verify_mismatches = 0;
		// Permutations published by the shader precompile replay (KYTY_SHADER_PRECOMPILE).
		uint64_t replayed          = 0;
	};
	[[nodiscard]] static ProgramTotals Totals();
	// The persistent program cache; null when KYTY_PROGRAM_CACHE is off.
	[[nodiscard]] ProgramDiskCache* GetProgramDiskCache() const { return m_program_disk.get(); }
	// Returns once the checks KYTY_PROGRAM_CACHE_VERIFY=background has queued are done (tests).
	void WaitProgramChecks();
	// Returns once the shader precompile replay (KYTY_SHADER_PRECOMPILE, shaderPrecompile.h) has
	// replayed every journal entry (tests); at once when it is off. The journal is also written.
	void WaitShaderPrecompile();
	// Diagnostic only; skips busy cache locks instead of waiting for compiles/preparation.
	void ReportRamStats();

private:
	struct ProgramCache;
	struct PipelineDiagnostics;
	struct DriverCacheSaver;
	struct LibraryState;
	struct FastFirstState;

	struct GraphicsPipelineKey {
		PipelineRenderingState   rendering;
		std::array<uint64_t, 3>  vertex_shader_ids {};
		uint64_t                 ps_shader_id = 0;
		PipelineVertexInputState vertex_input;
		PipelineStaticParameters static_params;

		bool operator==(const GraphicsPipelineKey& other) const {
			return rendering == other.rendering && vertex_shader_ids == other.vertex_shader_ids &&
			       ps_shader_id == other.ps_shader_id && vertex_input == other.vertex_input &&
			       static_params == other.static_params;
		}
	};

	struct PipelineKeyHash {
		static void Mix(std::size_t& hash, std::size_t value) {
			hash ^= value + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (hash << 6u) +
			        (hash >> 2u);
		}

		static void MixStaticParams(std::size_t& hash, const PipelineStaticParameters& params);

		static void MixRendering(std::size_t& hash, const PipelineRenderingState& rendering) {
			Mix(hash, rendering.color_count);
			for (uint32_t i = 0; i < rendering.color_count; i++) {
				Mix(hash, static_cast<uint32_t>(rendering.color_formats[i]));
			}
			Mix(hash, static_cast<uint32_t>(rendering.depth_format));
			Mix(hash, static_cast<uint32_t>(rendering.stencil_format));
		}
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const {
			std::size_t hash = 0;
			PipelineKeyHash::MixRendering(hash, key.rendering);
			for (const auto id: key.vertex_shader_ids) {
				PipelineKeyHash::Mix(hash, id);
			}
			PipelineKeyHash::Mix(hash, key.ps_shader_id);
			PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
			for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
				PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
			}
			PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
			for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
				PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
			}
			PipelineKeyHash::MixStaticParams(hash, key.static_params);
			return hash;
		}
	};

	GraphicContext&               m_graphics;
	std::unique_ptr<ProgramCache> m_program_cache;
	// Persistent translated-program cache (KYTY_PROGRAM_CACHE, programDiskCache.h); null when
	// off. m_program_cache refers to it.
	std::unique_ptr<ProgramDiskCache> m_program_disk;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::unordered_map<GraphicsPipelineKey, std::unique_ptr<Pipeline>, GraphicsPipelineKeyHash>
	                                                        m_graphics_pipelines;
	std::unordered_map<uint64_t, std::unique_ptr<Pipeline>> m_compute_pipelines;
	// Why new graphics pipelines were needed (compiles.csv, compile totals); guarded by m_mutex.
	std::unique_ptr<PipelineDiagnostics> m_diagnostics;
	// Guards the pipeline maps and the driver cache. ProgramCache has its own locks.
	Common::Mutex m_mutex;
	// Periodic crash-safe saves of m_driver_cache on a background thread (KYTY_PIPELINE_CACHE_SAVE).
	std::unique_ptr<DriverCacheSaver> m_saver;
	// Graphics pipeline libraries and the background compiles that replace linked pipelines
	// (KYTY_PIPELINE_LIBRARY, pipelineLibrary.h); null when off or unsupported.
	std::unique_ptr<LibraryState> m_library;
	// Unoptimized-first pipeline creation and the background optimized compiles
	// (KYTY_PIPELINE_FAST_FIRST, pipelineFastFirst.h); null when off.
	std::unique_ptr<FastFirstState> m_fast_first;
	struct PrefetchState;
	std::unique_ptr<PrefetchState> m_prefetch;
	// Background pipeline builds of software RT kernels (TryGetComputePipeline).
	struct RtState;
	std::unique_ptr<RtState> m_rt;
	std::once_flag           m_rt_once;
	[[nodiscard]] RtState&   Rt();
	// Shader precompile (KYTY_SHADER_PRECOMPILE=1, shaderPrecompile.h): the journal of compiled
	// permutations' inputs and the background replay of the entries an earlier run left in it.
	std::unique_ptr<ShaderJournal>     m_shader_journal;
	std::unique_ptr<ShaderPrecompiler> m_shader_precompiler;
	// Bumped whenever a cached pipeline object is replaced (a linked pipeline by its optimized
	// build), so that per-thread lookup memos do not keep returning the replaced object. Starts in
	// a range of its own per cache instance, so memos never match another instance.
	std::atomic<uint64_t> m_pipeline_generation {0};

	// GetGraphicsPipeline's key. fatal: stop the emulator where the key's inputs are unsupported,
	// as the serial lookup does, and note EXEC_ON_NOOP; otherwise return false for them, silently.
	bool BuildGraphicsPipelineKey(const PipelineTargets& targets, const HW::Context& ctx,
	                              const HW::UserConfig&        user_config,
	                              const ShaderVertexInputInfo& vs_input_info,
	                              const ShaderPixelInputInfo* ps_input_info,
	                              vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                              const GraphicsPrograms& programs, bool fatal,
	                              GraphicsPipelineKey& key) const;

	void InitializeDriverCache();
	void InitializeProgramDiskCache();
	void InitializeShaderPrecompile();
	void StopShaderPrecompile();
	// Serializes m_driver_cache and atomically replaces the cache file. Returns the payload size
	// written, 0 on failure, or UINT64_MAX for a periodic save skipped over the size cap.
	uint64_t WriteDriverCache(bool periodic);
	void     NotePipelineCreated(uint64_t create_ns);
	// Background compile finished: swaps `optimized` in for the linked pipeline cached under `key`.
	void ReplaceLinkedPipeline(const GraphicsPipelineKey* key, vk::Pipeline linked,
	                           vk::Pipeline optimized);
	// Fast-first: swaps the optimized pipeline in for the unoptimized one cached under the key (a
	// new object; the old one is retired), and destroys retired handles that are old enough.
	void ReplaceFastPipeline(const GraphicsPipelineKey* graphics_key, uint64_t compute_id,
	                         vk::Pipeline fast, vk::Pipeline optimized, uint64_t fast_ns,
	                         uint64_t optimize_ns);
};

// Creates a graphics pipeline from a complete monolithic create info instead of
// vkCreateGraphicsPipelines (the pipeline-library path). `layout_signature` identifies the
// definition of info.layout (identically defined layouts have equal signatures).
using GraphicsPipelineCreateHook =
    std::function<vk::Result(const vk::GraphicsPipelineCreateInfo& info,
                             std::span<const uint32_t> layout_signature, vk::Pipeline* pipeline)>;

// KYTY_PIPELINE_DYNAMIC_STATE (default on): cull mode, front face, the depth-bounds test enable and
// the depth bounds are dynamic state of every renderer graphics pipeline, recorded per draw by
// SetGraphicsDynamicParams, instead of pipeline-key fields (=0: baked into each pipeline).
[[nodiscard]] bool PipelineDynamicRasterStateEnabled();

void LogPipelineTrace(const char* phase, uint64_t vertex_program_id, uint64_t pixel_program_id);
// Stops the emulator at a mesh (NGG) draw on a device without mesh shaders, naming the missing
// extension and the device: such draws have no fallback.
[[noreturn]] void ExitWithoutMeshShaders(const GraphicContext& graphics);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache,
                            const GraphicsPipelineCreateHook*      create_hook = nullptr);
// Creates a compute pipeline from the complete create info instead of vkCreateComputePipelines
// (KYTY_PIPELINE_FAST_FIRST). The create info and its chained structures live only for the call.
using ComputePipelineCreateHook =
    std::function<vk::Result(const vk::ComputePipelineCreateInfo& info, vk::Pipeline* pipeline)>;
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache,
                            const ComputePipelineCreateHook* create_hook = nullptr,
                            // Not null: a failed creation is returned here (and cleaned up)
                            // instead of stopping the emulator.
                            vk::Result* result_out = nullptr);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
