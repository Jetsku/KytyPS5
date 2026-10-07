// GraphicsPipelineSnapshot serialization (graphics/host_gpu/renderer/pipeline/pipelineLibrary.h,
// pipelineSnapshot.cpp), the record format of KYTY_PIPELINE_JOURNAL and the key of KYTY_PIPELINE_KNOWN:
// a captured create info serializes with module hashes in place of modules, deserializes into an
// identical create info with the modules looked up again, and malformed or unresolvable records are
// refused. No device is needed: modules and layouts are opaque handles here.
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace Libs::Graphics;

static int g_failed = 0;
#define CHECK(expr)                                                                  \
	do {                                                                             \
		if (!(expr)) {                                                               \
			std::printf("PipelineSnapshotTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                              \
		}                                                                            \
	} while (0)

namespace {

template <typename Handle>
Handle FakeHandle(uint64_t value) {
	return Handle(reinterpret_cast<typename Handle::CType>(value));
}

uint64_t HandleValue(vk::ShaderModule module) {
	return reinterpret_cast<uint64_t>(static_cast<VkShaderModule>(module));
}

// A create info shaped like CreatePipelineInternal's (pipeline/shaders.cpp), owning its arrays.
struct Example {
	vk::Format                                                 colors[2] = {vk::Format::eR8G8B8A8Unorm,
	                                                                        vk::Format::eR16G16B16A16Sfloat};
	vk::PipelineRenderingCreateInfo                            rendering {};
	vk::PipelineShaderStageCreateInfo                          stages[2] {};
	vk::VertexInputBindingDescription                          binding {};
	vk::VertexInputAttributeDescription                        attributes[2] {};
	vk::PipelineVertexInputStateCreateInfo                     vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo                   input_assembly {};
	vk::PipelineViewportDepthClipControlCreateInfoEXT          clip_control {};
	vk::PipelineViewportStateCreateInfo                        viewport {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT       depth_clip {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking {};
	vk::PipelineRasterizationStateCreateInfo                   rasterization {};
	vk::PipelineMultisampleStateCreateInfo                     multisample {};
	vk::PipelineDepthStencilStateCreateInfo                    depth_stencil {};
	vk::PipelineColorBlendAttachmentState                      blend[2] {};
	vk::Bool32                                                 write_enables[2] = {VK_TRUE, VK_FALSE};
	vk::PipelineColorWriteCreateInfoEXT                        color_write {};
	vk::PipelineColorBlendStateCreateInfo                      color_blend {};
	vk::DynamicState dynamic_states[3] = {vk::DynamicState::eViewportWithCount, vk::DynamicState::eScissorWithCount,
	                                      vk::DynamicState::eCullMode};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	vk::GraphicsPipelineCreateInfo     info {};

	Example() {
		rendering.colorAttachmentCount    = 2;
		rendering.pColorAttachmentFormats = colors;
		rendering.depthAttachmentFormat   = vk::Format::eD32Sfloat;
		stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
		stages[0].module = FakeHandle<vk::ShaderModule>(0x1000);
		stages[0].pName  = "main";
		stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
		stages[1].module = FakeHandle<vk::ShaderModule>(0x2000);
		stages[1].pName  = "main";
		binding          = {0, 48, vk::VertexInputRate::eVertex};
		attributes[0]    = {0, 0, vk::Format::eR32G32B32Sfloat, 0};
		attributes[1]    = {3, 0, vk::Format::eR8G8B8A8Unorm, 36};
		vertex_input.vertexBindingDescriptionCount   = 1;
		vertex_input.pVertexBindingDescriptions      = &binding;
		vertex_input.vertexAttributeDescriptionCount = 2;
		vertex_input.pVertexAttributeDescriptions    = attributes;
		input_assembly.topology                      = vk::PrimitiveTopology::eTriangleStrip;
		input_assembly.primitiveRestartEnable        = VK_TRUE;
		clip_control.negativeOneToOne                = VK_TRUE;
		viewport.pNext                               = &clip_control;
		viewport.viewportCount                       = 1;
		viewport.scissorCount                        = 1;
		depth_clip.depthClipEnable                   = VK_TRUE;
		provoking.provokingVertexMode                = vk::ProvokingVertexModeEXT::eLastVertex;
		provoking.pNext                              = &depth_clip;
		rasterization.pNext                          = &provoking;
		rasterization.polygonMode                    = vk::PolygonMode::eFill;
		rasterization.cullMode                       = vk::CullModeFlagBits::eBack;
		rasterization.frontFace                      = vk::FrontFace::eClockwise;
		rasterization.depthBiasEnable                = VK_TRUE;
		rasterization.depthBiasConstantFactor        = 1.5f;
		rasterization.depthBiasSlopeFactor           = -0.25f;
		rasterization.lineWidth                      = 1.0f;
		multisample.rasterizationSamples             = vk::SampleCountFlagBits::e4;
		multisample.alphaToCoverageEnable            = VK_TRUE;
		depth_stencil.depthTestEnable                = VK_TRUE;
		depth_stencil.depthCompareOp                 = vk::CompareOp::eGreaterOrEqual;
		depth_stencil.stencilTestEnable              = VK_TRUE;
		depth_stencil.front.passOp                   = vk::StencilOp::eReplace;
		depth_stencil.back.compareMask               = 0x7f;
		depth_stencil.maxDepthBounds                 = 1.0f;
		blend[0].blendEnable                         = VK_TRUE;
		blend[0].srcColorBlendFactor                 = vk::BlendFactor::eSrcAlpha;
		blend[0].dstColorBlendFactor                 = vk::BlendFactor::eOneMinusSrcAlpha;
		blend[0].colorWriteMask                      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eA;
		blend[1].colorWriteMask                      = vk::ColorComponentFlagBits::eG;
		color_write.attachmentCount                  = 2;
		color_write.pColorWriteEnables               = write_enables;
		color_blend.pNext                            = &color_write;
		color_blend.logicOp                          = vk::LogicOp::eCopy;
		color_blend.attachmentCount                  = 2;
		color_blend.pAttachments                     = blend;
		color_blend.blendConstants[2]                = 0.5f;
		dynamic.dynamicStateCount                    = 3;
		dynamic.pDynamicStates                       = dynamic_states;
		info.pNext               = &rendering;
		info.stageCount          = 2;
		info.pStages             = stages;
		info.pVertexInputState   = &vertex_input;
		info.pInputAssemblyState = &input_assembly;
		info.pViewportState      = &viewport;
		info.pRasterizationState = &rasterization;
		info.pMultisampleState   = &multisample;
		info.pDepthStencilState  = &depth_stencil;
		info.pColorBlendState    = &color_blend;
		info.pDynamicState       = &dynamic;
		info.layout              = FakeHandle<vk::PipelineLayout>(0x77);
		info.basePipelineIndex   = -1;
	}
};

uint64_t HashOf(vk::ShaderModule module) {
	const auto value = HandleValue(module);
	return value == 0 ? 0 : value * 0x9E3779B97F4A7C15ull;
}

vk::ShaderModule ModuleFor(uint64_t hash) {
	for (const uint64_t value: {0x1000ull, 0x2000ull}) {
		if (value * 0x9E3779B97F4A7C15ull == hash) return FakeHandle<vk::ShaderModule>(value + 1); // another handle, same code
	}
	return nullptr;
}

} // namespace

int main() {
	const Example example;
	const auto    snapshot = GraphicsPipelineSnapshot::Capture(example.info);
	CHECK(snapshot != nullptr);
	if (snapshot == nullptr) return 1;

	std::vector<uint32_t> words;
	CHECK(snapshot->Serialize(words, HashOf));
	CHECK(words.size() > 60);
	const auto modules = GraphicsPipelineSnapshot::SerializedModules(words);
	CHECK(modules.size() == 2 && modules[0] == HashOf(example.stages[0].module) &&
	      modules[1] == HashOf(example.stages[1].module));

	// Round trip: the rebuilt create info serializes to the same words (the modules it found map
	// back to the same hashes) and carries the new layout.
	const auto layout  = FakeHandle<vk::PipelineLayout>(0x99);
	const auto rebuilt = GraphicsPipelineSnapshot::Deserialize(words, ModuleFor, layout);
	CHECK(rebuilt != nullptr);
	if (rebuilt != nullptr) {
		std::vector<uint32_t> again;
		CHECK(rebuilt->Serialize(again, [](vk::ShaderModule module) { return HashOf(FakeHandle<vk::ShaderModule>(HandleValue(module) - 1)); }));
		CHECK(again == words);
		const auto& info = rebuilt->Info();
		CHECK(info.layout == layout);
		CHECK(info.stageCount == 2 && HandleValue(info.pStages[1].module) == 0x2001 &&
		      std::string(info.pStages[0].pName) == "main");
		const auto* rendering = static_cast<const vk::PipelineRenderingCreateInfo*>(info.pNext);
		CHECK(rendering != nullptr && rendering->colorAttachmentCount == 2 &&
		      rendering->pColorAttachmentFormats[1] == vk::Format::eR16G16B16A16Sfloat &&
		      rendering->depthAttachmentFormat == vk::Format::eD32Sfloat);
		CHECK(info.pVertexInputState->vertexAttributeDescriptionCount == 2 &&
		      info.pVertexInputState->pVertexAttributeDescriptions[1].offset == 36);
		CHECK(info.pInputAssemblyState->primitiveRestartEnable == VK_TRUE);
		CHECK(info.pViewportState->pNext != nullptr);
		const auto* provoking =
		    static_cast<const vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT*>(info.pRasterizationState->pNext);
		CHECK(provoking != nullptr && provoking->provokingVertexMode == vk::ProvokingVertexModeEXT::eLastVertex &&
		      provoking->pNext != nullptr);
		CHECK(info.pRasterizationState->depthBiasSlopeFactor == -0.25f);
		CHECK(info.pDepthStencilState != nullptr && info.pDepthStencilState->back.compareMask == 0x7f);
		CHECK(info.pColorBlendState->attachmentCount == 2 && info.pColorBlendState->blendConstants[2] == 0.5f);
		const auto* write = static_cast<const vk::PipelineColorWriteCreateInfoEXT*>(info.pColorBlendState->pNext);
		CHECK(write != nullptr && write->attachmentCount == 2 && write->pColorWriteEnables[1] == VK_FALSE);
		CHECK(info.pDynamicState->dynamicStateCount == 3 && info.basePipelineIndex == -1);
		// The rebuilt create info is itself capturable (it is what the renderer would have passed).
		CHECK(GraphicsPipelineSnapshot::Capture(info) != nullptr);
	}

	// Equal create infos serialize equally; any state difference changes the words.
	{
		Example other;
		std::vector<uint32_t> same;
		CHECK(GraphicsPipelineSnapshot::Capture(other.info)->Serialize(same, HashOf) && same == words);
		other.blend[1].colorWriteMask = vk::ColorComponentFlagBits::eB;
		std::vector<uint32_t> changed;
		CHECK(GraphicsPipelineSnapshot::Capture(other.info)->Serialize(changed, HashOf) && changed != words);
	}

	// Refusals: an unhashed module, a missing module, truncated or trailing words, a bad magic.
	{
		std::vector<uint32_t> out;
		CHECK(!snapshot->Serialize(out, [](vk::ShaderModule) { return uint64_t {0}; }));
		CHECK(GraphicsPipelineSnapshot::Deserialize(words, [](uint64_t) { return vk::ShaderModule {}; }, layout) == nullptr);
		for (const size_t cut: {size_t {1}, size_t {10}, words.size() / 2, words.size() - 1}) {
			const std::vector<uint32_t> truncated(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(cut));
			CHECK(GraphicsPipelineSnapshot::Deserialize(truncated, ModuleFor, layout) == nullptr);
		}
		auto trailing = words;
		trailing.push_back(0);
		CHECK(GraphicsPipelineSnapshot::Deserialize(trailing, ModuleFor, layout) == nullptr);
		auto bad = words;
		bad[0] ^= 1u;
		CHECK(GraphicsPipelineSnapshot::Deserialize(bad, ModuleFor, layout) == nullptr);
		CHECK(GraphicsPipelineSnapshot::SerializedModules(bad).empty());
		auto huge = words;
		huge[4] = 1000000; // color count
		CHECK(GraphicsPipelineSnapshot::Deserialize(huge, ModuleFor, layout) == nullptr);
	}

	// Ineligible create infos are not captured (tessellation).
	{
		Example tess;
		vk::PipelineTessellationStateCreateInfo tessellation {};
		tess.info.pTessellationState = &tessellation;
		CHECK(GraphicsPipelineSnapshot::Capture(tess.info) == nullptr);
	}

	if (g_failed != 0) {
		std::printf("PipelineSnapshotTests: %d checks failed\n", g_failed);
		return 1;
	}
	std::puts("Pipeline snapshot: capture, serialization round trip and refusals passed");
	return 0;
}
