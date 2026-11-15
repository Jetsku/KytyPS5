#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/aaMaskCoverage.h"
#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineFeedback.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// IDK: maybe we can remove it?
constexpr uint8_t kTemporaryVertexAttribFormat113 =
    static_cast<uint8_t>(Prospero::VertexAttribFormat::k16_16SInt);
constexpr uint32_t kTemporaryPs5BufferFormat121 = 121u;

static bool NarrowInputFormat(vk::Format& format, uint32_t& size, uint32_t used_components) {
	if (used_components == 0 || used_components >= size) {
		return false;
	}

	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: format = vk::Format::eR16Sfloat; break;
				case 2: format = vk::Format::eR16G16Sfloat; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: format = vk::Format::eR8Unorm; break;
				case 2: format = vk::Format::eR8G8Unorm; break;
				default: return false;
			}
			size = used_components;
			return true;
		case vk::Format::eR8G8B8A8Snorm:
			if (used_components != 2) {
				return false;
			}
			format = vk::Format::eR8G8Snorm;
			size   = 2;
			return true;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: format = vk::Format::eR8Uint; break;
				case 2: format = vk::Format::eR8G8Uint; break;
				default: return false;
			}
			size = used_components;
			return true;
		default: break;
	}

	return false;
}

static void GetInputFormat(const ShaderBufferResource& res, vk::Format& format, uint32_t& size,
                           uint32_t used_components) {
	const auto fmt        = res.Format();
	const auto raw_format = res.RawFormat();
	if (raw_format == kTemporaryVertexAttribFormat113) {
		static std::atomic_bool logged_113 = false;
		if (!logged_113.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: temporary: accepting invalid PS5 buffer format 113 as "
			     "vk::Format::eR32G32B32A32Sfloat\n");
		}
		format = vk::Format::eR32G32B32A32Sfloat;
		size   = 4;
		if (NarrowInputFormat(format, size, used_components)) {
			LOGF("InputFormat: narrowing fmt=%u to %s for used_components=%u\n", raw_format,
			     vk::to_string(format).c_str(), used_components);
		}
		return;
	}
	if (raw_format == kTemporaryPs5BufferFormat121) {
		static std::atomic_bool logged_121 = false;
		if (!logged_121.exchange(true, std::memory_order_relaxed)) {
			LOGF("InputFormat: accepting PS5 buffer format 121 as vk::Format::eR16G16Sfloat\n");
		}
		format = vk::Format::eR16G16Sfloat;
		size   = 2;
		return;
	}

	format = VulkanFormat(fmt);
	size   = ShaderRecompiler::Format::GetFormatInfo(fmt).component_count;
	if (format == vk::Format::eUndefined || size == 0) {
		EXIT("unknown vertex format: fmt = %u descriptor=%08x,%08x,%08x,%08x\n",
		     raw_format, res.fields[0], res.fields[1], res.fields[2], res.fields[3]);
	}

	if (NarrowInputFormat(format, size, used_components)) {
		static std::atomic<uint64_t> log_count = 0;
		auto                         log_id    = log_count.fetch_add(1);
		if (log_id < 32) {
			LOGF("VertexInput: narrowed vertex format to %" PRIu32
			     " component(s) for shader fetch\n",
			     used_components);
		}
	}
}

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (const auto& binding: program.bindings.descriptors) {
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, binding.kind),
		     NativeDescriptorType(binding.kind), NativeDescriptorCount(binding), stage, nullptr});
	}
}

static bool UsesBindless(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return std::ranges::any_of(program.info.images, &ShaderRecompiler::IR::ImageResource::bindless) ||
	       std::ranges::any_of(program.info.samplers,
	                           &ShaderRecompiler::IR::SamplerResource::bindless);
}

// Set 0 plus one push-constant range {push_stages, 0, NativePushConstantSize}; interned by
// binding signature (pipelineLayoutCache.h, KYTY_LAYOUT_INTERN). A pipeline that samples bindless
// images (pipeline.uses_bindless) also gets set 1, the shared bindless table (Senaxx).
static void AssignPipelineLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                 std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                 vk::ShaderStageFlags                            push_stages) {
	EXIT_IF(pipeline.pipeline_layout != nullptr || pipeline.descriptor_set_layout != nullptr);
	vk::DescriptorSetLayout bindless_set = nullptr;
	if (pipeline.uses_bindless) {
		bindless_set = BindlessSetLayout(graphics);
		EXIT_IF(bindless_set == nullptr);
	}
	const auto layouts = AcquirePipelineLayout(graphics, bindings, push_stages,
	                                           ShaderRecompiler::IR::NativePushConstantSize,
	                                           bindless_set);
	pipeline.descriptor_set_layout = layouts.set_layout;
	pipeline.pipeline_layout       = layouts.pipeline_layout;
	pipeline.uses_push_descriptors = layouts.uses_push_descriptors;
	EXIT_NOT_IMPLEMENTED(pipeline.pipeline_layout == nullptr);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const PipelineRenderingState&          rendering,
                            const PipelineVertexInputState&        vertex_input,
                            std::span<const ShaderVertexInputInfo> vertex_info,
                            const ShaderPixelInputInfo*            ps_input_info,
                            const PipelineCache::GraphicsPrograms& programs,
                            const PipelineStaticParameters&        static_params,
                            vk::PipelineCache                      driver_cache,
                            const GraphicsPipelineCreateHook*      create_hook,
                            vk::Result*                            result_out) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	const bool  tessellation   = vertex_info.size() == 3;
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(!vertex_program || (ps_active && !pixel_program));
	const bool with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                        rendering.stencil_format != vk::Format::eUndefined;
	EXIT_IF(!vs_input_info.stage);
	const bool mesh = vs_input_info.stage.program->stage == ShaderType::Mesh;
	if (mesh && !graphics.mesh_shader_enabled) {
		ExitWithoutMeshShaders(graphics);
	}
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		tess_eval_shader_module    = CompileSPV(shaders.evaluation, graphics.device);
	}

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	uint32_t                          shader_stage_count = 0;
	// KYTY_LAYER_CLAMP: the bound layer count for the exported layer index's clamp.
	const uint32_t                   layer_limit_value = rendering.layer_limit;
	const vk::SpecializationMapEntry layer_limit_entry {ShaderRecompiler::IR::LayerLimitSpecId, 0,
	                                                    sizeof(uint32_t)};
	const vk::SpecializationInfo layer_limit_spec {1, &layer_limit_entry, sizeof(uint32_t),
	                                               &layer_limit_value};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
		if (layer_limit_value != 0) {
			shader_stages[shader_stage_count - 1].pSpecializationInfo = &layer_limit_spec;
		}
	}
	if (rect_list) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationControl,
		                                       .module = tess_control_shader_module,
		                                       .pName  = "main"};
		shader_stages[shader_stage_count++] = {.stage =
		                                           vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                       .module = tess_eval_shader_module,
		                                       .pName  = "main"};
	}
	if (ps_active) {
		// KYTY_EQAA_MIXED through NV mixed samples (fewer colour than raster samples) must not
		// shade per sample (VUID-VkPipelineMultisampleStateCreateInfo-rasterizationSamples-01415):
		// a pixel shader that reads its sample index runs its per-pixel copy (pixelRateVariant.h;
		// draws of one without a copy are skipped, RenderExecutor::ExecutePreparedDraw).
		auto module     = pixel_program.module;
		bool pixel_rate = false;
		if (rendering.color_samples != 0 &&
		    graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv &&
		    !EqaaPerSampleShading()) {
			if (const auto variant = PixelRate::Find(pixel_program.id); variant.module != nullptr) {
				module     = variant.module;
				pixel_rate = true;
			}
		}
		// KYTY_AA_MASK_COVERAGE: the copy that applies the key's AA masks (of the per-pixel copy
		// above when the draw runs that one).
		if (static_params.aa_mask[0] != 0xffffffffu || static_params.aa_mask[1] != 0xffffffffu) {
			if (const auto masked = AaMaskCoverage::Lookup(pixel_program.id, pixel_rate, static_params.aa_mask[0],
			                                               static_params.aa_mask[1]);
			    masked != nullptr) {
				module = masked;
			}
		}
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = module,
		                                       .pName  = "main"};
	} else if (static_params.aa_mask[0] != 0xffffffffu || static_params.aa_mask[1] != 0xffffffffu) {
		// KYTY_AA_MASK_COVERAGE: a draw without a pixel shader whose AA masks leave samples out (the
		// stencil marks of a ghost car): a fragment shader that only writes its coverage.
		if (const auto masked =
		        AaMaskCoverage::Lookup(0, false, static_params.aa_mask[0], static_params.aa_mask[1]);
		    masked != nullptr && shader_stage_count < std::size(shader_stages)) {
			shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
			                                       .module = masked,
			                                       .pName  = "main"};
		}
	}
	// One guest wave per host subgroup where the device lets the stage require it (mesh and
	// pixel shaders on AMD); GraphicsSubgroupSize and the mesh's host_subgroup_size
	// (FinishMeshStage) agree. NVIDIA has one subgroup size and chains nothing.
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo stage_subgroup_sizes[4] {};
	for (uint32_t index = 0; index < shader_stage_count; index++) {
		auto&      stage   = shader_stages[index];
		const bool is_mesh = stage.stage == vk::ShaderStageFlagBits::eMeshEXT;
		if (!is_mesh && (stage.stage != vk::ShaderStageFlagBits::eFragment || ps_input_info == nullptr ||
		                 !ps_input_info->stage)) {
			continue;
		}
		const auto wave_size = is_mesh ? vs_input_info.mesh.wave_size
		                               : ps_input_info->stage.program->wave_size;
		const auto required  = graphics.GraphicsSubgroupSize(stage.stage, wave_size);
		if (required != 0) {
			stage_subgroup_sizes[index].requiredSubgroupSize = required;
			stage.pNext                                      = &stage_subgroup_sizes[index];
		} else if (wave_size < graphics.subgroup_size) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				LOGF("Vulkan subgroup: wave%u %s shader on a %u-wide host subgroup the device "
				     "cannot narrow; KYTY_WAVE32_CLUSTERS keeps each wave to its own lanes\n",
				     wave_size, is_mesh ? "mesh" : "pixel", graphics.subgroup_size);
			}
		}
	}

	vk::VertexInputAttributeDescription input_attr[ShaderVertexInputInfo::RES_MAX] {};
	vk::VertexInputBindingDescription   input_desc[ShaderVertexInputInfo::RES_MAX] {};

	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		input_desc[binding].binding   = binding;
		input_desc[binding].stride    = vertex_input.bindings[binding].stride;
		input_desc[binding].inputRate = vertex_input.bindings[binding].instance
		                                    ? vk::VertexInputRate::eInstance
		                                    : vk::VertexInputRate::eVertex;
	}
	uint32_t input_attribute_count = 0;
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		uint32_t   attr_size     = 4;
		const auto registers_num = vs_input_info.resources_dst[index].registers_num;
		const auto compiled_components =
		    vs_input_info.stage.program->info.vertex_fetch_components[index];
		// Embedded fetches with only constant selectors emit no GetAttribute. They need
		// no native vertex input, including when their unused descriptor format is zero.
		// Preserve guest locations while compacting the native descriptions: later inputs
		// must not shift into a constant input's location.
		if (vs_input_info.fetch_embedded && compiled_components == 0) {
			continue;
		}
		auto& native_attribute     = input_attr[input_attribute_count++];
		native_attribute.binding  = vertex_input.attributes[index].binding;
		native_attribute.location = index;
		native_attribute.offset   = vertex_input.attributes[index].offset;
		const auto used_components =
		    compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
		GetInputFormat(vs_input_info.resources[index], native_attribute.format, attr_size,
		               static_cast<uint32_t>(used_components));

		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("VertexInputState[%u]: attr=%u binding=%u offset=%u stride=%u fmt=%d "
				     "src_fmt=%u dst=v%u regs=%u"
				     " fetched_components=%u attr_size=%u swizzle=%u,%u,%u,%u\n",
				     log_id, index, native_attribute.binding, native_attribute.offset,
				     input_desc[native_attribute.binding].stride,
				     static_cast<int>(native_attribute.format),
				     static_cast<uint32_t>(vs_input_info.resources[index].Format()),
				     static_cast<uint32_t>(vs_input_info.resources_dst[index].register_start),
				     static_cast<uint32_t>(registers_num), static_cast<uint32_t>(used_components),
				     attr_size, static_cast<uint32_t>(vs_input_info.resources[index].DstSelX()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelY()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelZ()),
				     static_cast<uint32_t>(vs_input_info.resources[index].DstSelW()));
			}
		}

		if (vs_input_info.resources[index].OutOfBounds() != 0) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				LOGF("VertexInput: temporary: accepting PS5 out-of-bounds behavior %" PRIu8 "\n",
				     vs_input_info.resources[index].OutOfBounds());
			}
		}

		EXIT_IF(registers_num < 1 || registers_num > 4);
	}

	vk::PipelineVertexInputStateCreateInfo vertex_input_info {};
	vertex_input_info.vertexBindingDescriptionCount   = vertex_input.binding_count;
	vertex_input_info.pVertexBindingDescriptions      = input_desc;
	vertex_input_info.vertexAttributeDescriptionCount = input_attribute_count;
	vertex_input_info.pVertexAttributeDescriptions    = input_attr;

	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = static_params.topology;
	input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineViewportDepthClipControlCreateInfoEXT depth_clip_control {};
	depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);

	vk::PipelineViewportStateCreateInfo viewport_state {};
	viewport_state.pNext = &depth_clip_control;

	vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
	if (static_params.cull_back) {
		cull_mode |= vk::CullModeFlagBits::eBack;
	}
	if (static_params.cull_front) {
		cull_mode |= vk::CullModeFlagBits::eFront;
	}

	vk::FrontFace front_face =
	    (static_params.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise);

	vk::PipelineRasterizationDepthClipStateCreateInfoEXT clip_ext {};
	clip_ext.depthClipEnable = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineRasterizationStateCreateInfo rasterizer {};
	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied; without
	// VK_EXT_depth_clip_enable the clamp also turns clipping off (DeviceCompat::DepthClampEnable).
	rasterizer.depthClampEnable = DeviceCompat::DepthClampEnable(graphics.depth_clip_enable_enabled,
	                                                             static_params.depth_clip_enable)
	                                  ? VK_TRUE
	                                  : VK_FALSE;
	if (graphics.depth_clip_enable_enabled) {
		rasterizer.pNext = &clip_ext;
	}
#endif
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT provoking_vertex {};
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		provoking_vertex.pNext = rasterizer.pNext;
		rasterizer.pNext = &provoking_vertex;
	}
	rasterizer.cullMode  = cull_mode;
	rasterizer.frontFace = front_face;
	rasterizer.polygonMode = static_params.polygon_mode;
	rasterizer.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampling {};
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i]);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i]);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i])
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i])
		         : color_blend_attachment[i].dstColorBlendFactor);
		color_blend_attachment[i].alphaBlendOp =
		    (static_params.separate_alpha_blend[i] ? GetBlendOp(static_params.alpha_comb_fcn[i])
		                                           : color_blend_attachment[i].colorBlendOp);
	}

	vk::Bool32 color_write_enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		color_write_enable[i] = VK_TRUE;
	}

	vk::PipelineColorWriteCreateInfoEXT color_write {};
	color_write.attachmentCount    = rendering.color_count;
	color_write.pColorWriteEnables = color_write_enable;

	vk::PipelineColorBlendStateCreateInfo color_blending {};
	// Without VK_EXT_color_write_enable (MoltenVK, older drivers) drop the dynamic color-write
	// struct and rely on each attachment's static colorWriteMask (all channels enabled by default).
	if (graphics.color_write_enable_enabled) {
		color_blending.pNext = &color_write;
	}
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	pipeline.uses_bindless               = false;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		graphics_stages |= native_stage;
		pipeline.uses_bindless |= UsesBindless(*stage.stage.program);
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
		pipeline.uses_bindless |= UsesBindless(*ps_input_info->stage.program);
	}
	AssignPipelineLayout(graphics, pipeline, descriptor_bindings, graphics_stages);
	vk::Result result = vk::Result::eSuccess;

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_info {};
	depth_stencil_info.depthBoundsTestEnable =
#if defined(__APPLE__)
	    VK_FALSE; // MoltenVK lacks the depthBounds feature; depth-bounds testing is disabled
#else
	    (static_params.depth_bounds_test_enable ? VK_TRUE : VK_FALSE);
#endif
	depth_stencil_info.minDepthBounds    = static_params.depth_min_bounds;
	depth_stencil_info.maxDepthBounds    = static_params.depth_max_bounds;

	std::vector<vk::DynamicState> dynamic_states {
	    vk::DynamicState::eViewportWithCount,
	    vk::DynamicState::eScissorWithCount,
	    vk::DynamicState::eLineWidth,
	    vk::DynamicState::eDepthTestEnable,
	    vk::DynamicState::eDepthWriteEnable,
	    vk::DynamicState::eDepthCompareOp,
	    vk::DynamicState::eDepthBiasEnable,
	    vk::DynamicState::eDepthBias,
	    vk::DynamicState::eStencilTestEnable,
	    vk::DynamicState::eStencilOp,
	    vk::DynamicState::eStencilCompareMask,
	    vk::DynamicState::eStencilReference,
	    vk::DynamicState::eStencilWriteMask,
	    vk::DynamicState::eBlendConstants,
	};
	if (graphics.color_write_enable_enabled && rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
	if (PipelineDynamicRasterStateEnabled()) {
		// Core Vulkan 1.3 (no feature bit). The key holds zeroes for these fields; the draw records
		// the values from the same registers (SetGraphicsDynamicParams).
		dynamic_states.push_back(vk::DynamicState::eCullMode);
		dynamic_states.push_back(vk::DynamicState::eFrontFace);
#if !defined(__APPLE__)
		dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
		dynamic_states.push_back(vk::DynamicState::eDepthBounds);
#endif
	}
	if (graphics.attachment_feedback_loop_enabled) {
		dynamic_states.push_back(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT);
	}

	vk::PipelineDynamicStateCreateInfo dynamic_state {};
	dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
	dynamic_state.pDynamicStates    = dynamic_states.data();

	vk::GraphicsPipelineCreateInfo  pipeline_info {};
	vk::PipelineRenderingCreateInfo rendering_info {};
	rendering_info.colorAttachmentCount    = rendering.color_count;
	rendering_info.pColorAttachmentFormats = rendering.color_formats.data();
	rendering_info.depthAttachmentFormat   = rendering.depth_format;
	rendering_info.stencilAttachmentFormat = rendering.stencil_format;
	// KYTY_EQAA_MIXED (usa-ratt/KytyPS5-GT7 18979e65b, df310f7a9): mixed attachment samples,
	// rasterization at the larger count and each attachment at its own
	// (VK_AMD_mixed_attachment_samples / VK_NV_framebuffer_mixed_samples). Through NV's the colour
	// sample takes its own raster sample's coverage (TRUNCATE) where the device lists it: with
	// MERGE a pixel whose raster samples two triangles of one blended mesh share blends twice.
	std::array<vk::SampleCountFlagBits, RENDER_COLOR_ATTACHMENTS_MAX> color_attachment_samples {};
	vk::AttachmentSampleCountInfoAMD               attachment_sample_count {};
	vk::PipelineCoverageReductionStateCreateInfoNV coverage_reduction {};
	if (rendering.color_samples != 0) {
		EXIT_IF(graphics.mixed_attachment_samples ==
		        GraphicContext::MixedAttachmentSamples::None);
		for (uint32_t i = 0; i < rendering.color_count; i++) {
			color_attachment_samples[i] = vulkan_sample_count(rendering.color_samples);
		}
		attachment_sample_count.colorAttachmentCount    = rendering.color_count;
		attachment_sample_count.pColorAttachmentSamples = color_attachment_samples.data();
		attachment_sample_count.depthStencilAttachmentSamples =
		    vulkan_sample_count(rendering.depth_samples);
		attachment_sample_count.pNext = rendering_info.pNext;
		rendering_info.pNext          = &attachment_sample_count;
		const auto raster_samples     = static_cast<uint32_t>(multisampling.rasterizationSamples);
		if (graphics.mixed_attachment_samples == GraphicContext::MixedAttachmentSamples::Nv &&
		    graphics.CoverageTruncate(raster_samples, rendering.depth_samples,
		                              rendering.color_samples)) {
			coverage_reduction.coverageReductionMode = vk::CoverageReductionModeNV::eTruncate;
			coverage_reduction.pNext                 = multisampling.pNext;
			multisampling.pNext                      = &coverage_reduction;
			static std::atomic_bool logged           = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				LOGF("Pipeline: NV mixed samples (raster %u, depth %u, colour %u): coverage "
				     "reduction TRUNCATE\n",
				     raster_samples, rendering.depth_samples, rendering.color_samples);
			}
		}
	}
	pipeline_info.pNext                    = &rendering_info;
	pipeline_info.stageCount               = shader_stage_count;
	pipeline_info.pStages                  = shader_stages;
	pipeline_info.pVertexInputState        = mesh ? nullptr : &vertex_input_info;
	pipeline_info.pInputAssemblyState      = mesh ? nullptr : &input_assembly;
	vk::PipelineTessellationStateCreateInfo tessellation_state {};
	tessellation_state.patchControlPoints =
	    tessellation ? vs_input_info.tess.input_control_points : 3u;
	pipeline_info.pTessellationState = (rect_list || tessellation) ? &tessellation_state : nullptr;
	pipeline_info.pViewportState          = &viewport_state;
	pipeline_info.pRasterizationState     = &rasterizer;
	pipeline_info.pMultisampleState       = &multisampling;
	pipeline_info.pDepthStencilState      = (with_depth ? &depth_stencil_info : nullptr);
	pipeline_info.pColorBlendState        = &color_blending;
	pipeline_info.pDynamicState           = &dynamic_state;
	pipeline_info.layout                  = pipeline.pipeline_layout;
	pipeline_info.basePipelineIndex       = -1;

	const auto driver_begin = std::chrono::steady_clock::now();
	if (create_hook != nullptr) {
		// The pipeline-library path shares libraries only between identically defined layouts.
		// Interned (pipelineLayoutCache.h): pipelines with equal canonical signatures get the same
		// layout handles, so that signature identifies the definition. Not interned: each pipeline
		// creates its layout from these bindings in this order, so the key keeps the order.
		std::vector<uint32_t> layout_signature;
		layout_signature.reserve(descriptor_bindings.size() * 4u + 4u);
		if (PipelineLayoutInterningEnabled()) {
			const auto canonical = MakePipelineLayoutSignature(
			    descriptor_bindings, graphics_stages, ShaderRecompiler::IR::NativePushConstantSize,
			    graphics.max_push_descriptors);
			layout_signature.push_back(1u);
			layout_signature.push_back(canonical.push_descriptors ? 1u : 0u);
			layout_signature.push_back(canonical.push_stages);
			layout_signature.push_back(canonical.push_size);
			for (const auto& binding: canonical.bindings) {
				layout_signature.insert(layout_signature.end(), binding.begin(), binding.end());
			}
		} else {
			layout_signature.push_back(0u);
			layout_signature.push_back(pipeline.uses_push_descriptors ? 1u : 0u);
			layout_signature.push_back(
			    static_cast<vk::ShaderStageFlags::MaskType>(graphics_stages));
			layout_signature.push_back(ShaderRecompiler::IR::NativePushConstantSize);
			for (const auto& binding: descriptor_bindings) {
				layout_signature.push_back(binding.binding);
				layout_signature.push_back(static_cast<uint32_t>(binding.descriptorType));
				layout_signature.push_back(binding.descriptorCount);
				layout_signature.push_back(
				    static_cast<vk::ShaderStageFlags::MaskType>(binding.stageFlags));
			}
		}
		if (pipeline.uses_bindless) {
			// Set 1 (the bindless table): one word more, so no other layout shares the key and the
			// journal replay, which reads four words per binding, leaves such pipelines out.
			layout_signature.push_back(0xb1d1e55u);
		}
		result = (*create_hook)(pipeline_info, layout_signature, &pipeline.pipeline);
	} else {
		result = PipelineFeedback::CreateGraphics(graphics.device, driver_cache, pipeline_info, &pipeline.pipeline,
		                                          PipelineFeedback::Path::Sync);
	}
	Profiler::AddFrameWait(Profiler::FrameWait::GraphicsPipelineDriver, 1,
	                       static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                 std::chrono::steady_clock::now() - driver_begin)
	                                                 .count()));
	const auto destroy_tess_modules = [&] {
		if (tess_control_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
		}
		if (tess_eval_shader_module != nullptr) {
			graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
		}
	};
	if (result_out != nullptr) {
		// The caller can do without this pipeline (KYTY_PIPELINE_SKIP_PROBE's driver-cache lookup):
		// the result is returned, and what was created for a failed one given back.
		*result_out = result;
		if (result != vk::Result::eSuccess || pipeline.pipeline == nullptr) {
			if (result == vk::Result::eSuccess) {
				*result_out = vk::Result::eErrorUnknown;
			}
			if (pipeline.pipeline != nullptr) {
				graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
				pipeline.pipeline = nullptr;
			}
			ReleasePipelineLayout(graphics, pipeline.pipeline_layout, pipeline.descriptor_set_layout);
			pipeline.pipeline_layout       = nullptr;
			pipeline.descriptor_set_layout = nullptr;
			destroy_tess_modules();
			return;
		}
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	destroy_tess_modules();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache,
                            const ComputePipelineCreateHook* create_hook,
                            vk::Result* result_out) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	// One guest wave per host subgroup: the wave size on AMD, 32 where the driver would otherwise
	// pick the width (Intel), nothing on NVIDIA (GraphicContext::ComputeSubgroupSize).
	const auto wave_size = input_info.stage.program->wave_size;
	const auto required  = graphics.ComputeSubgroupSize(wave_size, input_info.host_subgroup_size);
	if (required != 0) {
		comp_subgroup_size.requiredSubgroupSize = required;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	} else if (graphics.min_subgroup_size < graphics.max_subgroup_size) {
		static std::atomic_bool logged {false};
		if (!logged.exchange(true)) {
			LOGF("Vulkan subgroup: wave%u compute shader on a device that cannot require its "
			     "subgroup size (%u to %u); its lane operations may split or mix waves\n",
			     wave_size, graphics.min_subgroup_size, graphics.max_subgroup_size);
		}
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_bindless = UsesBindless(*input_info.stage.program);
	AssignPipelineLayout(graphics, pipeline, descriptor_bindings,
	                     vk::ShaderStageFlagBits::eCompute);
	vk::Result result = vk::Result::eSuccess;

	vk::ComputePipelineCreateInfo info {};
	// KYTY_ORDERED_APPEND: GDS programs may run per workgroup through vkCmdDispatchBase.
	if (OrderedAppendSplitEnabled() && input_info.stage.program != nullptr &&
	    (ShaderRecompiler::IR::FindBinding(input_info.stage.program->bindings, ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr)) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;
	// KYTY_PIPELINE_STATS (pipelineFeedback.h): the driver's statistics of selected shaders.
	const bool stats = PipelineStats::Enabled() && PipelineStats::Selected(compute_module);
	if (stats) {
		info.flags |= vk::PipelineCreateFlagBits::eCaptureStatisticsKHR;
	}

	if (create_hook != nullptr) {
		result = (*create_hook)(info, &pipeline.pipeline);
	} else {
		result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
		                                                &pipeline.pipeline);
	}
	if (stats && result == vk::Result::eSuccess && pipeline.pipeline != nullptr) {
		PipelineStats::Print(graphics.device, pipeline.pipeline, PipelineStats::Describe(&info.stage, 1), info.flags);
	}
	if (result_out != nullptr) {
		// The caller can live without this pipeline (an optional ray-tracing kernel): report the
		// failure instead of stopping the emulator, and give back what was created for it.
		*result_out = result;
		if (result != vk::Result::eSuccess || pipeline.pipeline == nullptr) {
			if (result == vk::Result::eSuccess) {
				*result_out = vk::Result::eErrorUnknown;
			}
			if (pipeline.pipeline != nullptr) {
				graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
				pipeline.pipeline = nullptr;
			}
			ReleasePipelineLayout(graphics, pipeline.pipeline_layout, pipeline.descriptor_set_layout);
			pipeline.pipeline_layout       = nullptr;
			pipeline.descriptor_set_layout = nullptr;
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
