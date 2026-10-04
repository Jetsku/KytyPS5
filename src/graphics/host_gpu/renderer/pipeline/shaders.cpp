#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "kernel/memory.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/host_gpu/deviceLostReport.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Declared in deviceLostReport.h: also called from the DeviceLostReport hook (vulkanWindow.cpp).
void ReportDeviceFault(const GraphicContext& graphics) {
	if (!graphics.device_fault_enabled ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceFaultInfoEXT == nullptr) {
		Log::WriteFatal("GPU device lost; no fault details because VK_EXT_device_fault is not "
		                "enabled (set KYTY_GPU_FAULT_REPORT=1; launcher: the \"GPU fault "
		                "report\" option; --graphics-debug-dump also enables it)\n");
		return;
	}
	vk::DeviceFaultCountsEXT counts {};
	auto result = graphics.device.getFaultInfoEXT(&counts, nullptr);
	if (result != vk::Result::eSuccess) {
		Log::WriteFatal(fmt::format("GPU fault query failed: {}\n", vk::to_string(result)));
		return;
	}
	// Bound diagnostic allocation even when the driver reports many fault records.
	counts.addressInfoCount = std::min(counts.addressInfoCount, 256u);
	counts.vendorInfoCount  = std::min(counts.vendorInfoCount, 256u);
	counts.vendorBinarySize = 0;
	std::vector<vk::DeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
	std::vector<vk::DeviceFaultVendorInfoEXT> vendors(counts.vendorInfoCount);
	vk::DeviceFaultInfoEXT info {};
	info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
	info.pVendorInfos  = vendors.empty() ? nullptr : vendors.data();
	result = graphics.device.getFaultInfoEXT(&counts, &info);
	if (result != vk::Result::eSuccess && result != vk::Result::eIncomplete) {
		Log::WriteFatal(fmt::format("GPU fault details failed: {}\n", vk::to_string(result)));
		return;
	}
	std::string report = fmt::format("--- GPU fault report ---\n{}\n", info.description.data());
	for (size_t i = 0; i < std::min<size_t>(counts.addressInfoCount, addresses.size()); ++i) {
		const auto& address = addresses[i];
		report += fmt::format("address: type={} reported=0x{:016x} precision={}\n",
		                      vk::to_string(address.addressType), address.reportedAddress,
		                      address.addressPrecision);
	}
	for (size_t i = 0; i < std::min<size_t>(counts.vendorInfoCount, vendors.size()); ++i) {
		const auto& vendor = vendors[i];
		report += fmt::format("vendor: {} code=0x{:x} data=0x{:x}\n", vendor.description.data(),
		                      vendor.vendorFaultCode, vendor.vendorFaultData);
	}
	Log::WriteFatal(report);

	// An invalid access is usually a pointer the shader loaded from guest memory: find where that value
	// sits. Printed after the report above, with a short budget, because other threads that hit the same
	// loss wait for this function only for DeviceLostReport::WaitLimit before the process exits.
	constexpr uint32_t kScanBudgetMs = 900;
	size_t             scans         = 0;
	for (size_t i = 0; i < std::min<size_t>(counts.addressInfoCount, addresses.size()) && scans < 2; ++i) {
		const auto& address = addresses[i];
		if (address.addressType != vk::DeviceFaultAddressTypeEXT::eReadInvalid &&
		    address.addressType != vk::DeviceFaultAddressTypeEXT::eWriteInvalid) {
			continue;
		}
		const uint64_t precision = std::max<uint64_t>(address.addressPrecision, 1);
		const uint64_t low       = address.reportedAddress - address.reportedAddress % precision;
		const uint64_t high      = low + precision - 1;
		// Addresses near zero would match every zero in memory.
		if (low < 0x10000 || (high >> 48) != 0) {
			continue;
		}
		scans++;
		Log::WriteFatal(fmt::format("--- Guest memory scan for the faulting address 0x{:x} ---\n{}",
		                            address.reportedAddress,
		                            Libs::LibKernel::Memory::ScanGuestMemoryForAddressRange(low, high, kScanBudgetMs)));
	}
}

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

static vk::BlendFactor GetBlendFactor(uint32_t factor, bool remap_source_alpha) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha:
			return remap_source_alpha ? vk::BlendFactor::eSrc1Color : vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return remap_source_alpha ? vk::BlendFactor::eOneMinusSrc1Color
			                          : vk::BlendFactor::eOneMinusSrcAlpha;
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

// Set 0 plus one push-constant range {push_stages, 0, NativePushConstantSize}; interned by
// binding signature (pipelineLayoutCache.h, KYTY_LAYOUT_INTERN).
static void AssignPipelineLayout(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                                 std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                 vk::ShaderStageFlags                            push_stages) {
	EXIT_IF(pipeline.pipeline_layout != nullptr || pipeline.descriptor_set_layout != nullptr);
	const auto layouts = AcquirePipelineLayout(graphics, bindings, push_stages,
	                                           ShaderRecompiler::IR::NativePushConstantSize);
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
                            const GraphicsPipelineCreateHook*      create_hook) {
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
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list =
	    !mesh && !tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;

	vk::ShaderModule tess_control_shader_module = nullptr;
	vk::ShaderModule tess_eval_shader_module    = nullptr;

	if (rect_list) {
		const auto shaders =
		    BuildRectListShaders(vs_input_info, ps_active ? ps_input_info : nullptr);
		tess_control_shader_module = CompileSPV(shaders.control, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TCS done module=%p\n",
			     static_cast<void*>(tess_control_shader_module));
		}

		tess_eval_shader_module = CompileSPV(shaders.evaluation, graphics.device);
		if (graphics_debug_dump_enabled()) {
			LOGF("PipelineTrace: vkCreateShaderModule RectList TES done module=%p\n",
			     static_cast<void*>(tess_eval_shader_module));
		}
	}

	EXIT_NOT_IMPLEMENTED(
	    rect_list && (tess_control_shader_module == nullptr || tess_eval_shader_module == nullptr));

	vk::PipelineShaderStageCreateInfo shader_stages[4] {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo fragment_subgroup_size {};
	uint32_t                          shader_stage_count = 0;
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		shader_stages[shader_stage_count++] = {.stage =
		                                           NativeShaderStage(vertex_info[i].logical_stage),
		                                       .module = programs.vertex[i].module,
		                                       .pName  = "main"};
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
		shader_stages[shader_stage_count++] = {.stage  = vk::ShaderStageFlagBits::eFragment,
		                                       .module = pixel_program.module,
		                                       .pName  = "main"};
		// EXEC/VCC masks describe one guest wave; keep native fragment lanes in that wave.
		const auto wave_size = ps_input_info->wave_size;
		if (graphics.compute_subgroup_size_control_enabled &&
		    (graphics.required_subgroup_size_stages & vk::ShaderStageFlagBits::eFragment) &&
		    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
			fragment_subgroup_size.requiredSubgroupSize = wave_size;
			shader_stages[shader_stage_count - 1].pNext = &fragment_subgroup_size;
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
	// The DB clamps depth to the viewport range after polygon offset is applied.
	rasterizer.depthClampEnable = VK_TRUE;
	rasterizer.pNext = &clip_ext;
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

	vk::PipelineColorBlendAttachmentState color_blend_attachment[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t i = 0; i < rendering.color_count; i++) {
		EXIT_NOT_IMPLEMENTED((static_params.color_mask[i] & ~0x0fu) != 0);
		color_blend_attachment[i].colorWriteMask =
		    vk::ColorComponentFlags {static_params.color_mask[i]};
		color_blend_attachment[i].blendEnable = static_params.blend_enable[i] ? VK_TRUE : VK_FALSE;
		// Only target 0 can use the second blend source carrying logical alpha.
		const bool remap_source_alpha         = i == 0 && static_params.blend_alpha_source_remap;
		color_blend_attachment[i].srcColorBlendFactor =
		    GetBlendFactor(static_params.color_srcblend[i], remap_source_alpha);
		color_blend_attachment[i].dstColorBlendFactor =
		    GetBlendFactor(static_params.color_destblend[i], remap_source_alpha);
		color_blend_attachment[i].colorBlendOp = GetBlendOp(static_params.color_comb_fcn[i]);
		color_blend_attachment[i].srcAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_srcblend[i], remap_source_alpha)
		         : color_blend_attachment[i].srcColorBlendFactor);
		color_blend_attachment[i].dstAlphaBlendFactor =
		    (static_params.separate_alpha_blend[i]
		         ? GetBlendFactor(static_params.alpha_destblend[i], remap_source_alpha)
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
	// MoltenVK lacks VK_EXT_color_write_enable; drop the dynamic color-write struct on macOS
	// and rely on each attachment's static colorWriteMask (all channels enabled by default).
#if !defined(__APPLE__)
	color_blending.pNext = &color_write;
#endif
	color_blending.logicOp         = vk::LogicOp::eCopy;
	color_blending.attachmentCount = rendering.color_count;
	color_blending.pAttachments    = color_blend_attachment;

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
	for (const auto& stage: vertex_info) {
		const auto native_stage = NativeShaderStage(stage.logical_stage);
		AddLayoutBindings(descriptor_bindings, *stage.stage.program, native_stage);
		graphics_stages |= native_stage;
	}
	if (ps_active) {
		EXIT_IF(!ps_input_info->stage);
		AddLayoutBindings(descriptor_bindings, *ps_input_info->stage.program,
		                  vk::ShaderStageFlagBits::eFragment);
	}
	AssignPipelineLayout(graphics, pipeline, descriptor_bindings, graphics_stages);
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: pipeline layout VS=%" PRIu64 " PS=%" PRIu64 " layout=%p push=%d\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<void*>(pipeline.pipeline_layout), pipeline.uses_push_descriptors ? 1 : 0);
	}
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
#if !defined(__APPLE__)
	if (rendering.color_count != 0) {
		dynamic_states.push_back(vk::DynamicState::eColorWriteEnableEXT);
	}
#endif
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

	EXIT_IF(pipeline.pipeline != nullptr);

	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines begin VS=%" PRIu64 " PS=%" PRIu64
		     " topology=%" PRIu32 " color_mask=0x%08" PRIx32
		     " depth=%s blend=%s dyn_states=%" PRIu32 "\n",
		     vertex_program.id, ps_active ? pixel_program.id : 0,
		     static_cast<uint32_t>(static_params.topology), static_params.color_mask[0],
		     (with_depth ? "true" : "false"), (static_params.blend_enable[0] ? "true" : "false"),
		     dynamic_state.dynamicStateCount);
	}
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
		result = (*create_hook)(pipeline_info, layout_signature, &pipeline.pipeline);
	} else {
		result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info, nullptr,
		                                                 &pipeline.pipeline);
	}
	Profiler::AddFrameWait(Profiler::FrameWait::GraphicsPipelineDriver, 1,
	                       static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                 std::chrono::steady_clock::now() - driver_begin)
	                                                 .count()));
	if (graphics_debug_dump_enabled()) {
		LOGF("PipelineTrace: vkCreateGraphicsPipelines done result=%s pipeline=%p\n",
		     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	}
	if (result == vk::Result::eOperationDeferredKHR) {
		// The hook took a copy of the create info for a background compile (KYTY_ASYNC_PIPELINES,
		// PipelineCache::TryGetGraphicsPipeline); the copy refuses tessellation, so no module
		// created here is still needed.
		EXIT_IF(create_hook == nullptr || pipeline.pipeline != nullptr ||
		        tess_control_shader_module != nullptr || tess_eval_shader_module != nullptr);
		return;
	}
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			ReportDeviceFault(graphics);
		}
		EXIT("vkCreateGraphicsPipelines failed: %s\n", vk::to_string(result).c_str());
	}

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);

	if (tess_control_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_control_shader_module, nullptr);
	}
	if (tess_eval_shader_module != nullptr) {
		graphics.device.destroyShaderModule(tess_eval_shader_module, nullptr);
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderComputeInputInfo& input_info,
                            vk::ShaderModule compute_module, vk::PipelineCache driver_cache) {
	EXIT_IF(compute_module == nullptr);

	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = compute_module;
	comp_shader_stage_info.pName  = "main";
	EXIT_IF(!input_info.stage);
	const auto wave_size = input_info.stage.program->wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, *input_info.stage.program,
	                  vk::ShaderStageFlagBits::eCompute);
	AssignPipelineLayout(graphics, pipeline, descriptor_bindings,
	                     vk::ShaderStageFlagBits::eCompute);
	LOGF("PipelineTrace: pipeline layout CS layout=%p push=%d\n",
	     static_cast<void*>(pipeline.pipeline_layout), pipeline.uses_push_descriptors ? 1 : 0);
	vk::Result result = vk::Result::eSuccess;

	vk::ComputePipelineCreateInfo info {};
	info.stage             = comp_shader_stage_info;
	info.layout            = pipeline.pipeline_layout;
	info.basePipelineIndex = -1;

	EXIT_IF(pipeline.pipeline != nullptr);

	LOGF("PipelineTrace: vkCreateComputePipelines begin layout=%p\n",
	     static_cast<void*>(pipeline.pipeline_layout));
	result = graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr,
	                                                &pipeline.pipeline);
	LOGF("PipelineTrace: vkCreateComputePipelines done result=%s pipeline=%p\n",
	     vk::to_string(result).c_str(), static_cast<void*>(pipeline.pipeline));
	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			ReportDeviceFault(graphics);
		}
		EXIT("vkCreateComputePipelines failed: %s, shader=0x%016" PRIx64 " wave_size=%u\n",
		     vk::to_string(result).c_str(), input_info.stage.program->shader_hash, wave_size);
	}

	EXIT_NOT_IMPLEMENTED(pipeline.pipeline == nullptr);
}

} // namespace Libs::Graphics
