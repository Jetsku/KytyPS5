#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include <algorithm>
#include <bit>
#include <cstring>

// GraphicsPipelineSnapshot (pipelineLibrary.h): the deep copy of a monolithic graphics pipeline
// create info and its word serialization. Vulkan-free apart from the structure definitions, so the
// pipeline_snapshot_tests can exercise it without a device.
namespace Libs::Graphics {

namespace {

// Walks a const pNext chain.
const vk::BaseInStructure* Next(const void* next) {
	return static_cast<const vk::BaseInStructure*>(next);
}

} // namespace

std::unique_ptr<GraphicsPipelineSnapshot>
GraphicsPipelineSnapshot::Capture(const vk::GraphicsPipelineCreateInfo& info, bool allow_mesh) {
	std::unique_ptr<GraphicsPipelineSnapshot> copy(new GraphicsPipelineSnapshot());
	auto&                                     s = *copy;
	// A mesh pipeline has no vertex input or input assembly state.
	s.m_mesh = allow_mesh && info.stageCount != 0 && info.pStages[0].stage == vk::ShaderStageFlagBits::eMeshEXT;
	if (info.flags != vk::PipelineCreateFlags {} || info.renderPass != nullptr ||
	    info.basePipelineHandle != nullptr || info.layout == nullptr ||
	    info.pTessellationState != nullptr || (info.pVertexInputState == nullptr) != s.m_mesh ||
	    (info.pInputAssemblyState == nullptr) != s.m_mesh || info.pViewportState == nullptr ||
	    info.pRasterizationState == nullptr || info.pMultisampleState == nullptr ||
	    info.pColorBlendState == nullptr || info.pDynamicState == nullptr) {
		return nullptr;
	}
	// The create info itself: exactly one VkPipelineRenderingCreateInfo.
	const vk::PipelineRenderingCreateInfo* rendering = nullptr;
	for (const auto* next = Next(info.pNext); next != nullptr; next = next->pNext) {
		if (next->sType != vk::StructureType::ePipelineRenderingCreateInfo ||
		    rendering != nullptr) {
			return nullptr;
		}
		rendering = reinterpret_cast<const vk::PipelineRenderingCreateInfo*>(next);
	}
	if (rendering == nullptr) return nullptr;
	s.m_rendering.viewMask                = rendering->viewMask;
	s.m_rendering.depthAttachmentFormat   = rendering->depthAttachmentFormat;
	s.m_rendering.stencilAttachmentFormat = rendering->stencilAttachmentFormat;
	s.m_color_formats.assign(rendering->pColorAttachmentFormats,
	                         rendering->pColorAttachmentFormats + rendering->colorAttachmentCount);

	// Stages: one vertex (or, with allow_mesh, mesh) shader and at most one fragment shader, plain
	// modules; a mesh pipeline's stages may require their subgroup size.
	const auto first_stage = s.m_mesh ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex;
	for (uint32_t i = 0; i < info.stageCount; ++i) {
		const auto& stage    = info.pStages[i];
		uint32_t    subgroup = 0;
		if (stage.pNext != nullptr) {
			const auto* next = Next(stage.pNext);
			if (!s.m_mesh || next->sType != vk::StructureType::ePipelineShaderStageRequiredSubgroupSizeCreateInfo ||
			    next->pNext != nullptr) {
				return nullptr;
			}
			subgroup = reinterpret_cast<const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo*>(next)
			               ->requiredSubgroupSize;
		}
		if (stage.flags != vk::PipelineShaderStageCreateFlags {} ||
		    stage.pSpecializationInfo != nullptr || stage.module == nullptr ||
		    stage.pName == nullptr) {
			return nullptr;
		}
		s.m_stage_subgroups.push_back({.requiredSubgroupSize = subgroup});
		if (stage.stage == first_stage && s.m_vertex_module == nullptr) {
			s.m_vertex_module = stage.module;
		} else if (stage.stage == vk::ShaderStageFlagBits::eFragment &&
		           s.m_fragment_module == nullptr) {
			s.m_fragment_module = stage.module;
		} else {
			return nullptr;
		}
		s.m_stages.push_back(stage);
		s.m_stage_names.emplace_back(stage.pName);
	}
	if (s.m_vertex_module == nullptr) return nullptr;

	if (!s.m_mesh) {
		const auto& vi = *info.pVertexInputState;
		if (vi.pNext != nullptr || vi.flags != vk::PipelineVertexInputStateCreateFlags {})
			return nullptr;
		s.m_bindings.assign(vi.pVertexBindingDescriptions,
		                    vi.pVertexBindingDescriptions + vi.vertexBindingDescriptionCount);
		s.m_attributes.assign(vi.pVertexAttributeDescriptions,
		                      vi.pVertexAttributeDescriptions + vi.vertexAttributeDescriptionCount);

		const auto& ia = *info.pInputAssemblyState;
		if (ia.pNext != nullptr || ia.flags != vk::PipelineInputAssemblyStateCreateFlags {})
			return nullptr;
		s.m_input_assembly.topology               = ia.topology;
		s.m_input_assembly.primitiveRestartEnable = ia.primitiveRestartEnable;
	}

	const auto& vp = *info.pViewportState;
	// Viewports and scissors are dynamic with count: no arrays.
	if (vp.flags != vk::PipelineViewportStateCreateFlags {} || vp.pViewports != nullptr ||
	    vp.pScissors != nullptr) {
		return nullptr;
	}
	s.m_viewport.viewportCount = vp.viewportCount;
	s.m_viewport.scissorCount  = vp.scissorCount;
	for (const auto* next = Next(vp.pNext); next != nullptr; next = next->pNext) {
		if (next->sType != vk::StructureType::ePipelineViewportDepthClipControlCreateInfoEXT ||
		    s.m_has_depth_clip_control) {
			return nullptr;
		}
		s.m_has_depth_clip_control = true;
		s.m_depth_clip_control.negativeOneToOne =
		    reinterpret_cast<const vk::PipelineViewportDepthClipControlCreateInfoEXT*>(next)
		        ->negativeOneToOne;
	}

	const auto& rs = *info.pRasterizationState;
	if (rs.flags != vk::PipelineRasterizationStateCreateFlags {}) return nullptr;
	s.m_rasterization.depthClampEnable        = rs.depthClampEnable;
	s.m_rasterization.rasterizerDiscardEnable = rs.rasterizerDiscardEnable;
	s.m_rasterization.polygonMode             = rs.polygonMode;
	s.m_rasterization.cullMode                = rs.cullMode;
	s.m_rasterization.frontFace               = rs.frontFace;
	s.m_rasterization.depthBiasEnable         = rs.depthBiasEnable;
	s.m_rasterization.depthBiasConstantFactor = rs.depthBiasConstantFactor;
	s.m_rasterization.depthBiasClamp          = rs.depthBiasClamp;
	s.m_rasterization.depthBiasSlopeFactor    = rs.depthBiasSlopeFactor;
	s.m_rasterization.lineWidth               = rs.lineWidth;
	// The renderer chains the provoking-vertex state before the depth-clip state; keep that order.
	for (const auto* next = Next(rs.pNext); next != nullptr; next = next->pNext) {
		if (next->sType == vk::StructureType::ePipelineRasterizationDepthClipStateCreateInfoEXT &&
		    !s.m_has_depth_clip) {
			const auto& clip =
			    *reinterpret_cast<const vk::PipelineRasterizationDepthClipStateCreateInfoEXT*>(
			        next);
			if (clip.flags != vk::PipelineRasterizationDepthClipStateCreateFlagsEXT {})
				return nullptr;
			s.m_has_depth_clip             = true;
			s.m_depth_clip.depthClipEnable = clip.depthClipEnable;
		} else if (next->sType ==
		               vk::StructureType::ePipelineRasterizationProvokingVertexStateCreateInfoEXT &&
		           !s.m_has_provoking_vertex && !s.m_has_depth_clip) {
			s.m_has_provoking_vertex = true;
			s.m_provoking_vertex.provokingVertexMode =
			    reinterpret_cast<const vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT*>(
			        next)
			        ->provokingVertexMode;
		} else {
			return nullptr;
		}
	}

	const auto& ms = *info.pMultisampleState;
	if (ms.pNext != nullptr || ms.pSampleMask != nullptr) return nullptr;
	s.m_multisample.flags                 = ms.flags;
	s.m_multisample.rasterizationSamples  = ms.rasterizationSamples;
	s.m_multisample.sampleShadingEnable   = ms.sampleShadingEnable;
	s.m_multisample.minSampleShading      = ms.minSampleShading;
	s.m_multisample.alphaToCoverageEnable = ms.alphaToCoverageEnable;
	s.m_multisample.alphaToOneEnable      = ms.alphaToOneEnable;

	if (info.pDepthStencilState != nullptr) {
		const auto& ds = *info.pDepthStencilState;
		if (ds.pNext != nullptr) return nullptr;
		s.m_has_depth_stencil   = true;
		s.m_depth_stencil       = ds;
		s.m_depth_stencil.pNext = nullptr;
	}

	const auto& cb                 = *info.pColorBlendState;
	s.m_color_blend.flags          = cb.flags;
	s.m_color_blend.logicOpEnable  = cb.logicOpEnable;
	s.m_color_blend.logicOp        = cb.logicOp;
	s.m_color_blend.blendConstants = cb.blendConstants;
	s.m_blend_attachments.assign(cb.pAttachments, cb.pAttachments + cb.attachmentCount);
	for (const auto* next = Next(cb.pNext); next != nullptr; next = next->pNext) {
		if (next->sType != vk::StructureType::ePipelineColorWriteCreateInfoEXT ||
		    s.m_has_color_write) {
			return nullptr;
		}
		const auto& write   = *reinterpret_cast<const vk::PipelineColorWriteCreateInfoEXT*>(next);
		s.m_has_color_write = true;
		s.m_color_write_enables.assign(write.pColorWriteEnables,
		                               write.pColorWriteEnables + write.attachmentCount);
	}

	const auto& dyn = *info.pDynamicState;
	if (dyn.pNext != nullptr || dyn.flags != vk::PipelineDynamicStateCreateFlags {}) return nullptr;
	s.m_dynamic_states.assign(dyn.pDynamicStates, dyn.pDynamicStates + dyn.dynamicStateCount);

	s.m_create.layout            = info.layout;
	s.m_create.basePipelineIndex = info.basePipelineIndex;
	s.Wire();
	return copy;
}

void GraphicsPipelineSnapshot::Wire() {
	m_rendering.pNext                   = nullptr;
	m_rendering.colorAttachmentCount    = static_cast<uint32_t>(m_color_formats.size());
	m_rendering.pColorAttachmentFormats = m_color_formats.data();
	for (size_t i = 0; i < m_stages.size(); ++i) {
		m_stages[i].pName = m_stage_names[i].c_str();
		m_stages[i].pNext = m_stage_subgroups[i].requiredSubgroupSize != 0 ? &m_stage_subgroups[i] : nullptr;
	}
	m_vertex_input.vertexBindingDescriptionCount   = static_cast<uint32_t>(m_bindings.size());
	m_vertex_input.pVertexBindingDescriptions      = m_bindings.data();
	m_vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(m_attributes.size());
	m_vertex_input.pVertexAttributeDescriptions    = m_attributes.data();
	m_viewport.pNext              = m_has_depth_clip_control ? &m_depth_clip_control : nullptr;
	m_depth_clip.pNext            = nullptr;
	m_provoking_vertex.pNext      = m_has_depth_clip ? &m_depth_clip : nullptr;
	m_rasterization.pNext         = m_has_provoking_vertex ? static_cast<void*>(&m_provoking_vertex)
	                                : m_has_depth_clip     ? static_cast<void*>(&m_depth_clip)
	                                                       : nullptr;
	m_color_write.attachmentCount = static_cast<uint32_t>(m_color_write_enables.size());
	m_color_write.pColorWriteEnables = m_color_write_enables.data();
	m_color_blend.pNext              = m_has_color_write ? &m_color_write : nullptr;
	m_color_blend.attachmentCount    = static_cast<uint32_t>(m_blend_attachments.size());
	m_color_blend.pAttachments       = m_blend_attachments.data();
	m_dynamic.dynamicStateCount      = static_cast<uint32_t>(m_dynamic_states.size());
	m_dynamic.pDynamicStates         = m_dynamic_states.data();

	const auto layout            = m_create.layout;
	const auto base_index        = m_create.basePipelineIndex;
	m_create                     = vk::GraphicsPipelineCreateInfo {};
	m_create.pNext               = &m_rendering;
	m_create.stageCount          = static_cast<uint32_t>(m_stages.size());
	m_create.pStages             = m_stages.data();
	m_create.pVertexInputState   = m_mesh ? nullptr : &m_vertex_input;
	m_create.pInputAssemblyState = m_mesh ? nullptr : &m_input_assembly;
	m_create.pViewportState      = &m_viewport;
	m_create.pRasterizationState = &m_rasterization;
	m_create.pMultisampleState   = &m_multisample;
	m_create.pDepthStencilState  = m_has_depth_stencil ? &m_depth_stencil : nullptr;
	m_create.pColorBlendState    = &m_color_blend;
	m_create.pDynamicState       = &m_dynamic;
	m_create.layout              = layout;
	m_create.basePipelineIndex   = base_index;
}

namespace {

constexpr uint32_t SnapshotMagic = 0x324e5350u; // "PSN2"
constexpr uint32_t SnapshotEnd   = 0x00444e45u; // "END"

class WordWriter {
public:
	explicit WordWriter(std::vector<uint32_t>& out): m_out(out) {}
	void U32(uint32_t value) { m_out.push_back(value); }
	void U64(uint64_t value) {
		U32(static_cast<uint32_t>(value));
		U32(static_cast<uint32_t>(value >> 32u));
	}
	void F32(float value) { U32(std::bit_cast<uint32_t>(value)); }
	template <typename T>
	void Enum(T value) {
		U32(static_cast<uint32_t>(value));
	}
	template <typename T>
	void Flags(T value) {
		U32(static_cast<uint32_t>(static_cast<typename T::MaskType>(value)));
	}
	void Text(const std::string& text) {
		U32(static_cast<uint32_t>(text.size()));
		for (size_t i = 0; i < text.size(); i += 4) {
			uint32_t word = 0;
			std::memcpy(&word, text.data() + i, std::min<size_t>(4, text.size() - i));
			U32(word);
		}
	}
	void Stencil(const vk::StencilOpState& s) {
		Enum(s.failOp);
		Enum(s.passOp);
		Enum(s.depthFailOp);
		Enum(s.compareOp);
		U32(s.compareMask);
		U32(s.writeMask);
		U32(s.reference);
	}

private:
	std::vector<uint32_t>& m_out;
};

class WordReader {
public:
	explicit WordReader(std::span<const uint32_t> words): m_words(words) {}
	uint32_t U32() {
		if (m_position >= m_words.size()) {
			m_failed = true;
			return 0;
		}
		return m_words[m_position++];
	}
	uint64_t U64() {
		const uint64_t low = U32();
		return low | (static_cast<uint64_t>(U32()) << 32u);
	}
	float F32() { return std::bit_cast<float>(U32()); }
	template <typename T>
	T Enum() {
		return static_cast<T>(U32());
	}
	template <typename T>
	T Flags() {
		return T(static_cast<typename T::MaskType>(U32()));
	}
	// A count of at most `limit` items (a malformed record must not allocate gigabytes).
	uint32_t Count(uint32_t limit) {
		const auto count = U32();
		if (count > limit) m_failed = true;
		return m_failed ? 0 : count;
	}
	std::string Text() {
		const auto  size = Count(256);
		std::string text(size, '\0');
		for (size_t i = 0; i < size; i += 4) {
			const auto word = U32();
			std::memcpy(text.data() + i, &word, std::min<size_t>(4, size - i));
		}
		return text;
	}
	vk::StencilOpState Stencil() {
		vk::StencilOpState s {};
		s.failOp      = Enum<vk::StencilOp>();
		s.passOp      = Enum<vk::StencilOp>();
		s.depthFailOp = Enum<vk::StencilOp>();
		s.compareOp   = Enum<vk::CompareOp>();
		s.compareMask = U32();
		s.writeMask   = U32();
		s.reference   = U32();
		return s;
	}
	[[nodiscard]] bool Failed() const { return m_failed; }
	[[nodiscard]] bool AtEnd() const { return m_position == m_words.size(); }

private:
	std::span<const uint32_t> m_words;
	size_t                    m_position = 0;
	bool                      m_failed   = false;
};

} // namespace

bool GraphicsPipelineSnapshot::Serialize(std::vector<uint32_t>& out, const ModuleHashFn& module_hash) const {
	WordWriter w(out);
	w.U32(SnapshotMagic);
	w.U32(m_mesh ? 1u : 0u);
	w.U32(m_rendering.viewMask);
	w.Enum(m_rendering.depthAttachmentFormat);
	w.Enum(m_rendering.stencilAttachmentFormat);
	w.U32(static_cast<uint32_t>(m_color_formats.size()));
	for (const auto format: m_color_formats) w.Enum(format);
	w.U32(static_cast<uint32_t>(m_stages.size()));
	for (size_t i = 0; i < m_stages.size(); ++i) {
		const auto hash = module_hash(m_stages[i].module);
		if (hash == 0) return false;
		w.Flags(vk::ShaderStageFlags(m_stages[i].stage));
		w.U64(hash);
		w.Text(m_stage_names[i]);
		w.U32(m_stage_subgroups[i].requiredSubgroupSize);
	}
	w.U32(static_cast<uint32_t>(m_bindings.size()));
	for (const auto& b: m_bindings) {
		w.U32(b.binding);
		w.U32(b.stride);
		w.Enum(b.inputRate);
	}
	w.U32(static_cast<uint32_t>(m_attributes.size()));
	for (const auto& a: m_attributes) {
		w.U32(a.location);
		w.U32(a.binding);
		w.Enum(a.format);
		w.U32(a.offset);
	}
	w.Enum(m_input_assembly.topology);
	w.U32(m_input_assembly.primitiveRestartEnable);
	w.U32(m_viewport.viewportCount);
	w.U32(m_viewport.scissorCount);
	w.U32(m_has_depth_clip_control ? 1u : 0u);
	w.U32(m_depth_clip_control.negativeOneToOne);
	const auto& rs = m_rasterization;
	w.U32(rs.depthClampEnable);
	w.U32(rs.rasterizerDiscardEnable);
	w.Enum(rs.polygonMode);
	w.Flags(rs.cullMode);
	w.Enum(rs.frontFace);
	w.U32(rs.depthBiasEnable);
	w.F32(rs.depthBiasConstantFactor);
	w.F32(rs.depthBiasClamp);
	w.F32(rs.depthBiasSlopeFactor);
	w.F32(rs.lineWidth);
	w.U32(m_has_provoking_vertex ? 1u : 0u);
	w.Enum(m_provoking_vertex.provokingVertexMode);
	w.U32(m_has_depth_clip ? 1u : 0u);
	w.U32(m_depth_clip.depthClipEnable);
	const auto& ms = m_multisample;
	w.Flags(ms.flags);
	w.Enum(ms.rasterizationSamples);
	w.U32(ms.sampleShadingEnable);
	w.F32(ms.minSampleShading);
	w.U32(ms.alphaToCoverageEnable);
	w.U32(ms.alphaToOneEnable);
	w.U32(m_has_depth_stencil ? 1u : 0u);
	const auto& ds = m_depth_stencil;
	w.Flags(ds.flags);
	w.U32(ds.depthTestEnable);
	w.U32(ds.depthWriteEnable);
	w.Enum(ds.depthCompareOp);
	w.U32(ds.depthBoundsTestEnable);
	w.U32(ds.stencilTestEnable);
	w.Stencil(ds.front);
	w.Stencil(ds.back);
	w.F32(ds.minDepthBounds);
	w.F32(ds.maxDepthBounds);
	const auto& cb = m_color_blend;
	w.Flags(cb.flags);
	w.U32(cb.logicOpEnable);
	w.Enum(cb.logicOp);
	for (const auto constant: cb.blendConstants) w.F32(constant);
	w.U32(static_cast<uint32_t>(m_blend_attachments.size()));
	for (const auto& a: m_blend_attachments) {
		w.U32(a.blendEnable);
		w.Enum(a.srcColorBlendFactor);
		w.Enum(a.dstColorBlendFactor);
		w.Enum(a.colorBlendOp);
		w.Enum(a.srcAlphaBlendFactor);
		w.Enum(a.dstAlphaBlendFactor);
		w.Enum(a.alphaBlendOp);
		w.Flags(a.colorWriteMask);
	}
	w.U32(m_has_color_write ? 1u : 0u);
	w.U32(static_cast<uint32_t>(m_color_write_enables.size()));
	for (const auto enable: m_color_write_enables) w.U32(enable);
	w.U32(static_cast<uint32_t>(m_dynamic_states.size()));
	for (const auto state: m_dynamic_states) w.Enum(state);
	w.U32(static_cast<uint32_t>(m_create.basePipelineIndex));
	w.U32(SnapshotEnd);
	return true;
}

std::unique_ptr<GraphicsPipelineSnapshot>
GraphicsPipelineSnapshot::Deserialize(std::span<const uint32_t> words, const ModuleLookupFn& module_for,
                                      vk::PipelineLayout layout) {
	std::unique_ptr<GraphicsPipelineSnapshot> copy(new GraphicsPipelineSnapshot());
	auto&                                     s = *copy;
	WordReader                                r(words);
	if (r.U32() != SnapshotMagic) return nullptr;
	s.m_mesh                              = r.U32() != 0;
	s.m_rendering.viewMask                = r.U32();
	s.m_rendering.depthAttachmentFormat   = r.Enum<vk::Format>();
	s.m_rendering.stencilAttachmentFormat = r.Enum<vk::Format>();
	for (auto n = r.Count(8); n != 0; --n) s.m_color_formats.push_back(r.Enum<vk::Format>());
	for (auto n = r.Count(2); n != 0; --n) {
		vk::PipelineShaderStageCreateInfo stage {};
		stage.stage       = static_cast<vk::ShaderStageFlagBits>(r.U32());
		const auto hash   = r.U64();
		auto       name   = r.Text();
		s.m_stage_subgroups.push_back({.requiredSubgroupSize = r.U32()});
		if (r.Failed()) return nullptr;
		stage.module = module_for(hash);
		if (stage.module == nullptr) return nullptr;
		const auto first = s.m_mesh ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex;
		if (stage.stage == first && s.m_vertex_module == nullptr) {
			s.m_vertex_module = stage.module;
		} else if (stage.stage == vk::ShaderStageFlagBits::eFragment && s.m_fragment_module == nullptr) {
			s.m_fragment_module = stage.module;
		} else {
			return nullptr;
		}
		s.m_stages.push_back(stage);
		s.m_stage_names.push_back(std::move(name));
	}
	if (s.m_vertex_module == nullptr) return nullptr;
	for (auto n = r.Count(64); n != 0; --n) {
		vk::VertexInputBindingDescription b {};
		b.binding   = r.U32();
		b.stride    = r.U32();
		b.inputRate = r.Enum<vk::VertexInputRate>();
		s.m_bindings.push_back(b);
	}
	for (auto n = r.Count(64); n != 0; --n) {
		vk::VertexInputAttributeDescription a {};
		a.location = r.U32();
		a.binding  = r.U32();
		a.format   = r.Enum<vk::Format>();
		a.offset   = r.U32();
		s.m_attributes.push_back(a);
	}
	s.m_input_assembly.topology               = r.Enum<vk::PrimitiveTopology>();
	s.m_input_assembly.primitiveRestartEnable = r.U32();
	s.m_viewport.viewportCount                = r.U32();
	s.m_viewport.scissorCount                 = r.U32();
	s.m_has_depth_clip_control                = r.U32() != 0;
	s.m_depth_clip_control.negativeOneToOne   = r.U32();
	auto& rs                   = s.m_rasterization;
	rs.depthClampEnable        = r.U32();
	rs.rasterizerDiscardEnable = r.U32();
	rs.polygonMode             = r.Enum<vk::PolygonMode>();
	rs.cullMode                = r.Flags<vk::CullModeFlags>();
	rs.frontFace               = r.Enum<vk::FrontFace>();
	rs.depthBiasEnable         = r.U32();
	rs.depthBiasConstantFactor = r.F32();
	rs.depthBiasClamp          = r.F32();
	rs.depthBiasSlopeFactor    = r.F32();
	rs.lineWidth               = r.F32();
	s.m_has_provoking_vertex                 = r.U32() != 0;
	s.m_provoking_vertex.provokingVertexMode = r.Enum<vk::ProvokingVertexModeEXT>();
	s.m_has_depth_clip                       = r.U32() != 0;
	s.m_depth_clip.depthClipEnable           = r.U32();
	auto& ms                 = s.m_multisample;
	ms.flags                 = r.Flags<vk::PipelineMultisampleStateCreateFlags>();
	ms.rasterizationSamples  = r.Enum<vk::SampleCountFlagBits>();
	ms.sampleShadingEnable   = r.U32();
	ms.minSampleShading      = r.F32();
	ms.alphaToCoverageEnable = r.U32();
	ms.alphaToOneEnable      = r.U32();
	s.m_has_depth_stencil    = r.U32() != 0;
	auto& ds                 = s.m_depth_stencil;
	ds.flags                 = r.Flags<vk::PipelineDepthStencilStateCreateFlags>();
	ds.depthTestEnable       = r.U32();
	ds.depthWriteEnable      = r.U32();
	ds.depthCompareOp        = r.Enum<vk::CompareOp>();
	ds.depthBoundsTestEnable = r.U32();
	ds.stencilTestEnable     = r.U32();
	ds.front                 = r.Stencil();
	ds.back                  = r.Stencil();
	ds.minDepthBounds        = r.F32();
	ds.maxDepthBounds        = r.F32();
	auto& cb                 = s.m_color_blend;
	cb.flags                 = r.Flags<vk::PipelineColorBlendStateCreateFlags>();
	cb.logicOpEnable         = r.U32();
	cb.logicOp               = r.Enum<vk::LogicOp>();
	for (auto& constant: cb.blendConstants) constant = r.F32();
	for (auto n = r.Count(8); n != 0; --n) {
		vk::PipelineColorBlendAttachmentState a {};
		a.blendEnable         = r.U32();
		a.srcColorBlendFactor = r.Enum<vk::BlendFactor>();
		a.dstColorBlendFactor = r.Enum<vk::BlendFactor>();
		a.colorBlendOp        = r.Enum<vk::BlendOp>();
		a.srcAlphaBlendFactor = r.Enum<vk::BlendFactor>();
		a.dstAlphaBlendFactor = r.Enum<vk::BlendFactor>();
		a.alphaBlendOp        = r.Enum<vk::BlendOp>();
		a.colorWriteMask      = r.Flags<vk::ColorComponentFlags>();
		s.m_blend_attachments.push_back(a);
	}
	s.m_has_color_write = r.U32() != 0;
	for (auto n = r.Count(8); n != 0; --n) s.m_color_write_enables.push_back(r.U32());
	for (auto n = r.Count(64); n != 0; --n) s.m_dynamic_states.push_back(r.Enum<vk::DynamicState>());
	s.m_create.basePipelineIndex = static_cast<int32_t>(r.U32());
	if (r.U32() != SnapshotEnd || r.Failed() || !r.AtEnd()) return nullptr;
	s.m_create.layout = layout;
	s.Wire();
	return copy;
}

std::vector<uint64_t> GraphicsPipelineSnapshot::SerializedModules(std::span<const uint32_t> words) {
	std::vector<uint64_t> hashes;
	WordReader            r(words);
	if (r.U32() != SnapshotMagic) return {};
	(void)r.U32();
	(void)r.U32();
	(void)r.U32();
	(void)r.U32();
	for (auto n = r.Count(8); n != 0; --n) (void)r.U32();
	for (auto n = r.Count(2); n != 0; --n) {
		(void)r.U32();
		hashes.push_back(r.U64());
		(void)r.Text();
		(void)r.U32();
	}
	return r.Failed() ? std::vector<uint64_t> {} : hashes;
}

} // namespace Libs::Graphics
