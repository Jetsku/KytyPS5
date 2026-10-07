#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// Graphics pipeline libraries (VK_EXT_graphics_pipeline_library), KYTY_PIPELINE_LIBRARY.
//
// A new graphics pipeline is normally one synchronous vkCreateGraphicsPipelines on the command
// processor that compiles every stage. With libraries, the four parts of a pipeline (vertex input
// interface, pre-rasterization shaders, fragment shader, fragment output interface) are created
// separately, kept, and reused by every later pipeline that shares that part; a new pipeline is a
// fast link (no link-time optimization) of the four. The optimized monolithic pipeline, created
// from exactly the create info the renderer would have used without libraries, is compiled on a
// background thread and replaces the linked one when it is ready (PipelineCache).
//
// When the device supports pipeline creation cache control, the monolithic pipeline is first
// requested with VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT: a pipeline the driver
// cache already holds is used directly, so warm boots take the same pipelines as without
// libraries.
//
// Only pipelines built from the structures CreatePipelineInternal uses, with a vertex shader and
// optionally a fragment shader, are linked; mesh, tessellation and rect-list pipelines, and any
// create info with other chained structures, are created monolithically as before.
//
// Libraries are shared only between pipelines whose layouts are identically defined, and a
// layout holds the bindings of every stage: a vertex shader paired with fragment shaders of
// different bindings gets one pre-rasterization library per layout. The layout identity is the
// interned layout signature (pipelineLayoutCache.h, KYTY_LAYOUT_INTERN) or, without interning,
// the ordered binding list (pipeline/shaders.cpp).
//
// Off by default. A fast-linked pipeline is compiled without whole-pipeline optimization and is
// later replaced by the optimized build. Positions stay bit-identical between the two only
// because the recompiler decorates the position Invariant and computes it without contraction
// (KYTY_MAD_MODE exact or position, the default; not with =fused): otherwise a multi-pass effect
// that depth-tests EQUAL against an earlier pass could lose fragments while a linked pipeline is
// in use. Other outputs (pixel-shader math keeps FMA contraction) may still differ in the last
// bits for the frames a linked pipeline renders, and which frames those are depends on
// background compile timing, so runs are not pixel-reproducible with libraries on. Enabling the
// extension also makes some drivers (NVIDIA) compile shader modules at creation, moving compile
// time from pipeline creation to program compiles.
//
// Switches: KYTY_PIPELINE_LIBRARY=1 enables it (the device extensions are enabled only then);
// KYTY_PIPELINE_LIBRARY_OPTIMIZE=0 keeps the fast-linked pipelines instead of replacing them;
// KYTY_PIPELINE_LIBRARY_THREADS (default 2) background compile threads;
// KYTY_PIPELINE_LIBRARY_PROBE=0 skips the driver-cache probe (links even cached pipelines).
[[nodiscard]] bool PipelineLibraryRequested();

// Deep copy of a monolithic graphics pipeline create info, restricted to the structures the
// renderer uses (pipeline/shaders.cpp). The copy owns everything it points to and does not move.
class GraphicsPipelineSnapshot {
public:
	// Null when `info` uses anything the copy does not reproduce exactly. allow_mesh also copies
	// mesh-shader pipelines (no vertex input; stages may require their subgroup size), which the
	// library path cannot link.
	[[nodiscard]] static std::unique_ptr<GraphicsPipelineSnapshot>
	Capture(const vk::GraphicsPipelineCreateInfo& info, bool allow_mesh = false);

	GraphicsPipelineSnapshot(const GraphicsPipelineSnapshot&)            = delete;
	GraphicsPipelineSnapshot& operator=(const GraphicsPipelineSnapshot&) = delete;

	// The monolithic create info, pointing into this object.
	[[nodiscard]] const vk::GraphicsPipelineCreateInfo& Info() const { return m_create; }

	// Every field of the create info as words, with each shader module replaced by `module_hash`
	// of it and without the layout (KYTY_PIPELINE_KNOWN and KYTY_PIPELINE_JOURNAL key pipelines by
	// this plus the layout signature). False when a module has no hash (0).
	using ModuleHashFn = std::function<uint64_t(vk::ShaderModule)>;
	[[nodiscard]] bool Serialize(std::vector<uint32_t>& out, const ModuleHashFn& module_hash) const;
	// The snapshot Serialize wrote, with the modules `module_for` gives for the hashes and `layout`.
	// Null when the words are malformed or a module is missing (null).
	using ModuleLookupFn = std::function<vk::ShaderModule(uint64_t)>;
	[[nodiscard]] static std::unique_ptr<GraphicsPipelineSnapshot>
	Deserialize(std::span<const uint32_t> words, const ModuleLookupFn& module_for, vk::PipelineLayout layout);
	// The module hashes Serialize wrote (vertex first), without looking modules up.
	[[nodiscard]] static std::vector<uint64_t> SerializedModules(std::span<const uint32_t> words);

private:
	friend class GraphicsPipelineLibrary;
	GraphicsPipelineSnapshot() = default;
	void Wire();

	vk::GraphicsPipelineCreateInfo                             m_create {};
	vk::PipelineRenderingCreateInfo                            m_rendering {};
	std::vector<vk::Format>                                    m_color_formats;
	std::vector<vk::PipelineShaderStageCreateInfo>             m_stages;
	std::vector<std::string>                                   m_stage_names;
	std::vector<vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo> m_stage_subgroups; // 0: none
	bool                                                       m_mesh = false;
	vk::PipelineVertexInputStateCreateInfo                     m_vertex_input {};
	std::vector<vk::VertexInputBindingDescription>             m_bindings;
	std::vector<vk::VertexInputAttributeDescription>           m_attributes;
	vk::PipelineInputAssemblyStateCreateInfo                   m_input_assembly {};
	vk::PipelineViewportStateCreateInfo                        m_viewport {};
	bool                                                       m_has_depth_clip_control = false;
	vk::PipelineViewportDepthClipControlCreateInfoEXT          m_depth_clip_control {};
	vk::PipelineRasterizationStateCreateInfo                   m_rasterization {};
	bool                                                       m_has_depth_clip = false;
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT       m_depth_clip {};
	bool                                                       m_has_provoking_vertex = false;
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT m_provoking_vertex {};
	vk::PipelineMultisampleStateCreateInfo                     m_multisample {};
	bool                                                       m_has_depth_stencil = false;
	vk::PipelineDepthStencilStateCreateInfo                    m_depth_stencil {};
	vk::PipelineColorBlendStateCreateInfo                      m_color_blend {};
	std::vector<vk::PipelineColorBlendAttachmentState>         m_blend_attachments;
	bool                                                       m_has_color_write = false;
	vk::PipelineColorWriteCreateInfoEXT                        m_color_write {};
	std::vector<vk::Bool32>                                    m_color_write_enables;
	vk::PipelineDynamicStateCreateInfo                         m_dynamic {};
	std::vector<vk::DynamicState>                              m_dynamic_states;
	vk::ShaderModule                                           m_vertex_module   = nullptr;
	vk::ShaderModule                                           m_fragment_module = nullptr;
};

// The library cache and link step. Not thread-safe: the caller serializes Create (PipelineCache
// holds its m_mutex). Libraries live until this object is destroyed, which must happen after every
// pipeline linked from them.
class GraphicsPipelineLibrary {
public:
	explicit GraphicsPipelineLibrary(GraphicContext& graphics);
	~GraphicsPipelineLibrary();
	GraphicsPipelineLibrary(const GraphicsPipelineLibrary&)            = delete;
	GraphicsPipelineLibrary& operator=(const GraphicsPipelineLibrary&) = delete;

	// Supported by the device and requested (KYTY_PIPELINE_LIBRARY).
	[[nodiscard]] static bool Enabled(const GraphicContext& graphics);

	enum class Path : uint8_t {
		Monolithic, // ineligible or failed: created exactly as without libraries
		CacheHit,   // monolithic, already in the driver cache (no compile)
		Linked,     // fast-linked from libraries; `optimize` holds the monolithic create info
	};
	struct Result {
		vk::Result   result   = vk::Result::eSuccess;
		vk::Pipeline pipeline = nullptr;
		Path         path     = Path::Monolithic;
		// Why a Monolithic pipeline was not linked: "ineligible" (tessellation, rect-list or mesh
		// stages, or a structure the snapshot does not copy), "probe-failed", "library-failed" or
		// "link-failed".
		const char*                               fallback = nullptr;
		std::unique_ptr<GraphicsPipelineSnapshot> optimize;
		uint32_t                                  libraries_created = 0;
		uint64_t                                  libraries_ns      = 0;
		uint64_t                                  link_ns           = 0;
	};

	// Creates the pipeline `info` describes (a complete monolithic create info). `layout_signature`
	// identifies the definition of info.layout: libraries are shared only between pipelines whose
	// layouts are identically defined, as linking requires.
	[[nodiscard]] Result Create(const vk::GraphicsPipelineCreateInfo& info,
	                            std::span<const uint32_t>             layout_signature,
	                            vk::PipelineCache                     driver_cache);

	[[nodiscard]] size_t LibraryCount() const;

private:
	using Key = std::vector<uint32_t>;
	struct KeyHash {
		size_t operator()(const Key& key) const;
	};
	using LibraryMap = std::unordered_map<Key, vk::Pipeline, KeyHash>;

	vk::Pipeline GetLibrary(LibraryMap& libraries, Key key,
	                        const vk::GraphicsPipelineCreateInfo& info,
	                        vk::PipelineCache driver_cache, Result& result);

	GraphicContext& m_graphics;
	LibraryMap      m_vertex_input;
	LibraryMap      m_pre_rasterization;
	LibraryMap      m_fragment_shader;
	LibraryMap      m_fragment_output;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
