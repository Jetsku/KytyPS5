#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include "common/assert.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

uint64_t LibraryClockNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Library keys are the exact field values of the state a library consumes, serialized into
// words: equal keys describe identical library create infos (with an identically defined layout).
class KeyWriter {
public:
	explicit KeyWriter(std::vector<uint32_t>& key): m_key(key) {}
	void U32(uint32_t value) { m_key.push_back(value); }
	void U64(uint64_t value) {
		m_key.push_back(static_cast<uint32_t>(value));
		m_key.push_back(static_cast<uint32_t>(value >> 32u));
	}
	void F32(float value) { m_key.push_back(std::bit_cast<uint32_t>(value)); }
	template <typename T>
	void Enum(T value) {
		m_key.push_back(static_cast<uint32_t>(value));
	}
	template <typename T>
	void Flags(T value) {
		m_key.push_back(static_cast<uint32_t>(static_cast<typename T::MaskType>(value)));
	}
	template <typename T>
	void Handle(T handle) {
		U64(reinterpret_cast<uint64_t>(static_cast<typename T::CType>(handle)));
	}
	void Words(std::span<const uint32_t> words) {
		U32(static_cast<uint32_t>(words.size()));
		m_key.insert(m_key.end(), words.begin(), words.end());
	}

private:
	std::vector<uint32_t>& m_key;
};

void WriteDynamic(KeyWriter& key, const std::vector<vk::DynamicState>& states) {
	key.U32(static_cast<uint32_t>(states.size()));
	for (const auto state: states) {
		key.Enum(state);
	}
}

void WriteMultisample(KeyWriter& key, const vk::PipelineMultisampleStateCreateInfo& ms) {
	key.Flags(ms.flags);
	key.Enum(ms.rasterizationSamples);
	key.U32(ms.sampleShadingEnable);
	key.F32(ms.minSampleShading);
	key.U32(ms.alphaToCoverageEnable);
	key.U32(ms.alphaToOneEnable);
}

void WriteStencil(KeyWriter& key, const vk::StencilOpState& state) {
	key.Enum(state.failOp);
	key.Enum(state.passOp);
	key.Enum(state.depthFailOp);
	key.Enum(state.compareOp);
	key.U32(state.compareMask);
	key.U32(state.writeMask);
	key.U32(state.reference);
}

} // namespace

bool PipelineLibraryRequested() {
	static const bool requested = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_LIBRARY");
		return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
	}();
	return requested;
}

namespace {

// KYTY_PIPELINE_LIBRARY_PROBE=0 skips the driver-cache probe, so that every new eligible pipeline
// is linked from libraries even when the driver's own cache already holds it (measures the cold
// path on a machine whose caches are warm).
bool ProbeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_LIBRARY_PROBE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

} // namespace

size_t GraphicsPipelineLibrary::KeyHash::operator()(const Key& key) const {
	uint64_t hash = 0xcbf29ce484222325ull;
	for (const auto word: key) {
		hash = (hash ^ word) * 0x100000001b3ull;
	}
	return static_cast<size_t>(hash);
}

GraphicsPipelineLibrary::GraphicsPipelineLibrary(GraphicContext& graphics): m_graphics(graphics) {}

GraphicsPipelineLibrary::~GraphicsPipelineLibrary() {
	for (auto* libraries:
	     {&m_vertex_input, &m_pre_rasterization, &m_fragment_shader, &m_fragment_output}) {
		for (const auto& [key, library]: *libraries) {
			(void)key;
			m_graphics.device.destroyPipeline(library, nullptr);
		}
	}
}

bool GraphicsPipelineLibrary::Enabled(const GraphicContext& graphics) {
	return PipelineLibraryRequested() && graphics.pipeline_library_enabled;
}

size_t GraphicsPipelineLibrary::LibraryCount() const {
	return m_vertex_input.size() + m_pre_rasterization.size() + m_fragment_shader.size() +
	       m_fragment_output.size();
}

vk::Pipeline GraphicsPipelineLibrary::GetLibrary(LibraryMap& libraries, Key key,
                                                 const vk::GraphicsPipelineCreateInfo& info,
                                                 vk::PipelineCache driver_cache, Result& result) {
	if (const auto found = libraries.find(key); found != libraries.end()) {
		return found->second;
	}
	const auto   begin   = LibraryClockNs();
	vk::Pipeline library = nullptr;
	if (m_graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr, &library) !=
	        vk::Result::eSuccess ||
	    library == nullptr) {
		return nullptr;
	}
	result.libraries_ns += LibraryClockNs() - begin;
	result.libraries_created++;
	libraries.emplace(std::move(key), library);
	return library;
}

GraphicsPipelineLibrary::Result
GraphicsPipelineLibrary::Create(const vk::GraphicsPipelineCreateInfo& info,
                                std::span<const uint32_t>             layout_signature,
                                vk::PipelineCache                     driver_cache) {
	Result     result;
	const auto monolithic = [&](const char* reason) {
		result.path     = Path::Monolithic;
		result.fallback = reason;
		result.result   = m_graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr,
		                                                            &result.pipeline);
		result.optimize.reset();
		return std::move(result);
	};
	auto snapshot = GraphicsPipelineSnapshot::Capture(info);
	if (snapshot == nullptr) {
		return monolithic("ineligible");
	}
	const auto& s = *snapshot;

	// A pipeline already in the driver cache needs no compile: take the monolithic one directly.
	if (m_graphics.pipeline_creation_cache_control_enabled && ProbeEnabled()) {
		auto probe            = s.Info();
		probe.flags           = vk::PipelineCreateFlagBits::eFailOnPipelineCompileRequired;
		vk::Pipeline pipeline = nullptr;
		const auto   probed =
		    m_graphics.device.createGraphicsPipelines(driver_cache, 1, &probe, nullptr, &pipeline);
		if (probed == vk::Result::eSuccess && pipeline != nullptr) {
			result.path     = Path::CacheHit;
			result.pipeline = pipeline;
			return result;
		}
		if (pipeline != nullptr) {
			m_graphics.device.destroyPipeline(pipeline, nullptr);
		}
		if (probed != vk::Result::ePipelineCompileRequired) {
			return monolithic("probe-failed");
		}
	}

	// Vertex input interface.
	Key vi_key;
	{
		KeyWriter key(vi_key);
		key.U32(0x5649u);
		key.U32(static_cast<uint32_t>(s.m_bindings.size()));
		for (const auto& binding: s.m_bindings) {
			key.U32(binding.binding);
			key.U32(binding.stride);
			key.Enum(binding.inputRate);
		}
		key.U32(static_cast<uint32_t>(s.m_attributes.size()));
		for (const auto& attribute: s.m_attributes) {
			key.U32(attribute.location);
			key.U32(attribute.binding);
			key.Enum(attribute.format);
			key.U32(attribute.offset);
		}
		key.Enum(s.m_input_assembly.topology);
		key.U32(s.m_input_assembly.primitiveRestartEnable);
		WriteDynamic(key, s.m_dynamic_states);
	}
	// Pre-rasterization shaders.
	Key pr_key;
	{
		KeyWriter key(pr_key);
		key.U32(0x5052u);
		key.Handle(s.m_vertex_module);
		key.Words(layout_signature);
		key.U32(s.m_rendering.viewMask);
		key.U32(s.m_viewport.viewportCount);
		key.U32(s.m_viewport.scissorCount);
		key.U32(s.m_has_depth_clip_control ? 1u : 0u);
		key.U32(s.m_depth_clip_control.negativeOneToOne);
		const auto& rs = s.m_rasterization;
		key.U32(rs.depthClampEnable);
		key.U32(rs.rasterizerDiscardEnable);
		key.Enum(rs.polygonMode);
		key.Flags(rs.cullMode);
		key.Enum(rs.frontFace);
		key.U32(rs.depthBiasEnable);
		key.F32(rs.depthBiasConstantFactor);
		key.F32(rs.depthBiasClamp);
		key.F32(rs.depthBiasSlopeFactor);
		key.F32(rs.lineWidth);
		key.U32(s.m_has_depth_clip ? 1u : 0u);
		key.U32(s.m_depth_clip.depthClipEnable);
		key.U32(s.m_has_provoking_vertex ? 1u : 0u);
		key.Enum(s.m_provoking_vertex.provokingVertexMode);
		WriteDynamic(key, s.m_dynamic_states);
	}
	// Fragment shader.
	Key fs_key;
	{
		KeyWriter key(fs_key);
		key.U32(0x4653u);
		key.Handle(s.m_fragment_module);
		key.Words(layout_signature);
		key.U32(s.m_rendering.viewMask);
		key.U32(s.m_has_depth_stencil ? 1u : 0u);
		const auto& ds = s.m_depth_stencil;
		key.Flags(ds.flags);
		key.U32(ds.depthTestEnable);
		key.U32(ds.depthWriteEnable);
		key.Enum(ds.depthCompareOp);
		key.U32(ds.depthBoundsTestEnable);
		key.U32(ds.stencilTestEnable);
		WriteStencil(key, ds.front);
		WriteStencil(key, ds.back);
		key.F32(ds.minDepthBounds);
		key.F32(ds.maxDepthBounds);
		WriteMultisample(key, s.m_multisample);
		WriteDynamic(key, s.m_dynamic_states);
	}
	// Fragment output interface.
	Key fo_key;
	{
		KeyWriter key(fo_key);
		key.U32(0x464fu);
		key.U32(s.m_rendering.viewMask);
		key.U32(static_cast<uint32_t>(s.m_color_formats.size()));
		for (const auto format: s.m_color_formats) {
			key.Enum(format);
		}
		key.Enum(s.m_rendering.depthAttachmentFormat);
		key.Enum(s.m_rendering.stencilAttachmentFormat);
		const auto& cb = s.m_color_blend;
		key.Flags(cb.flags);
		key.U32(cb.logicOpEnable);
		key.Enum(cb.logicOp);
		key.U32(static_cast<uint32_t>(s.m_blend_attachments.size()));
		for (const auto& attachment: s.m_blend_attachments) {
			key.U32(attachment.blendEnable);
			key.Enum(attachment.srcColorBlendFactor);
			key.Enum(attachment.dstColorBlendFactor);
			key.Enum(attachment.colorBlendOp);
			key.Enum(attachment.srcAlphaBlendFactor);
			key.Enum(attachment.dstAlphaBlendFactor);
			key.Enum(attachment.alphaBlendOp);
			key.Flags(attachment.colorWriteMask);
		}
		for (const auto constant: cb.blendConstants) {
			key.F32(constant);
		}
		key.U32(s.m_has_color_write ? 1u : 0u);
		key.U32(static_cast<uint32_t>(s.m_color_write_enables.size()));
		for (const auto enable: s.m_color_write_enables) {
			key.U32(enable);
		}
		WriteMultisample(key, s.m_multisample);
		WriteDynamic(key, s.m_dynamic_states);
	}

	// Library create infos, all from the snapshot: each carries only its subset's state. Dynamic
	// states outside a library's subset are ignored by the implementation, so every library gets
	// the pipeline's full list. The pre-rasterization and fragment shader libraries use the
	// pipeline's layout; the key's layout signature guarantees that any pipeline later linked with
	// them has an identically defined one. Their rendering info carries only the view mask (as
	// shader libraries are commonly built); the formats belong to the fragment output interface.
	const auto library_create = [&](vk::GraphicsPipelineLibraryFlagsEXT       flags,
	                                vk::GraphicsPipelineLibraryCreateInfoEXT& library_info,
	                                const void*                               library_next) {
		library_info       = vk::GraphicsPipelineLibraryCreateInfoEXT {};
		library_info.flags = flags;
		library_info.pNext = const_cast<void*>(library_next);
		vk::GraphicsPipelineCreateInfo create {};
		create.flags             = vk::PipelineCreateFlagBits::eLibraryKHR;
		create.pNext             = &library_info;
		create.pDynamicState     = &s.m_dynamic;
		create.basePipelineIndex = -1;
		return create;
	};
	vk::PipelineRenderingCreateInfo shader_rendering {};
	shader_rendering.viewMask = s.m_rendering.viewMask;
	vk::GraphicsPipelineLibraryCreateInfoEXT library_info {};

	auto vi_info = library_create(vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface,
	                              library_info, nullptr);
	vi_info.pVertexInputState   = &s.m_vertex_input;
	vi_info.pInputAssemblyState = &s.m_input_assembly;
	const auto vi = GetLibrary(m_vertex_input, std::move(vi_key), vi_info, driver_cache, result);

	const auto* vertex_stage = &s.m_stages[0];
	for (const auto& stage: s.m_stages) {
		if (stage.stage == vk::ShaderStageFlagBits::eVertex) vertex_stage = &stage;
	}
	auto pr_info = library_create(vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders,
	                              library_info, &shader_rendering);
	pr_info.stageCount          = 1;
	pr_info.pStages             = vertex_stage;
	pr_info.pViewportState      = &s.m_viewport;
	pr_info.pRasterizationState = &s.m_rasterization;
	pr_info.layout              = s.m_create.layout;
	const auto pr = vi == nullptr ? nullptr
	                              : GetLibrary(m_pre_rasterization, std::move(pr_key), pr_info,
	                                           driver_cache, result);

	const vk::PipelineShaderStageCreateInfo* fragment_stage = nullptr;
	for (const auto& stage: s.m_stages) {
		if (stage.stage == vk::ShaderStageFlagBits::eFragment) fragment_stage = &stage;
	}
	// Fragment shader state takes a depth/stencil state even without a depth attachment; the
	// default one tests nothing, and the pipeline's own (all tests are dynamic state) otherwise.
	const vk::PipelineDepthStencilStateCreateInfo no_depth_stencil {};
	auto fs_info       = library_create(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader,
	                                    library_info, &shader_rendering);
	fs_info.stageCount = fragment_stage != nullptr ? 1u : 0u;
	fs_info.pStages    = fragment_stage;
	fs_info.pDepthStencilState = s.m_has_depth_stencil ? &s.m_depth_stencil : &no_depth_stencil;
	fs_info.pMultisampleState  = &s.m_multisample;
	fs_info.layout             = s.m_create.layout;
	const auto fs = pr == nullptr ? nullptr
	                              : GetLibrary(m_fragment_shader, std::move(fs_key), fs_info,
	                                           driver_cache, result);

	auto fo_info = library_create(vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface,
	                              library_info, &s.m_rendering);
	fo_info.pColorBlendState  = &s.m_color_blend;
	fo_info.pMultisampleState = &s.m_multisample;
	const auto fo = fs == nullptr ? nullptr
	                              : GetLibrary(m_fragment_output, std::move(fo_key), fo_info,
	                                           driver_cache, result);
	if (fo == nullptr) {
		// A library failed; nothing is linked from partial state.
		return monolithic("library-failed");
	}

	// Fast link: no VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT.
	const std::array<vk::Pipeline, 4> libraries {vi, pr, fs, fo};
	vk::PipelineLibraryCreateInfoKHR  link_libraries {};
	link_libraries.libraryCount = static_cast<uint32_t>(libraries.size());
	link_libraries.pLibraries   = libraries.data();
	vk::GraphicsPipelineCreateInfo link {};
	link.pNext              = &link_libraries;
	link.layout             = s.m_create.layout;
	link.basePipelineIndex  = -1;
	const auto   link_begin = LibraryClockNs();
	vk::Pipeline pipeline   = nullptr;
	if (m_graphics.device.createGraphicsPipelines(driver_cache, 1, &link, nullptr, &pipeline) !=
	        vk::Result::eSuccess ||
	    pipeline == nullptr) {
		return monolithic("link-failed");
	}
	result.link_ns  = LibraryClockNs() - link_begin;
	result.path     = Path::Linked;
	result.pipeline = pipeline;
	result.optimize = std::move(snapshot);
	return result;
}

} // namespace Libs::Graphics
