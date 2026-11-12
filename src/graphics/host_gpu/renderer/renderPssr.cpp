#include "common/assert.h"
#include "common/logging/log.h"
#include "gpu_blit_shaders/gpu_pssr_linear_spv.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/dlss.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <optional>
#include <vector>

// Experimental Pro mode (--trinity): the host pass of libPsml's PSSR request (R_PSSR), ported from
// upstream 651ef73a0 (renderPssr.cpp, gpu_pssr_linear.comp). Upstream's PSSR is a bilinear upscale
// of the request's active input rectangle into the output image, not Sony's network; so is this.
// With a DLSS model folder (--dlss, NVIDIA only) a request with depth and motion vectors is
// upscaled by DLSS instead (upstream 782a85c62, dlss.cpp); the linear pass stays the fallback.
// Differences from upstream: it records through our CommandBuffer (batched transitions, the
// recorder-aware bind/push/dispatch calls; DLSS records into the native handle) and holds its
// pipeline and DLSS contexts in PssrState; a request the pass cannot run (formats, tiling, sizes,
// overlap) is logged and skipped instead of stopping the emulator. Only reachable in Pro mode: the
// packet is never produced otherwise.

namespace Libs::Graphics {
namespace {

struct PssrParameters {
	uint32_t input_size[2];
	uint32_t output_size[2];
	uint32_t output_channels[4];
};

// DLSS reads and writes with stages and accesses of its own.
void DlssMemoryBarrier(vk::CommandBuffer command) {
	const vk::MemoryBarrier2 barrier {.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
	                                  .dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
	                                  .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
	                                                   vk::AccessFlagBits2::eMemoryWrite};
	command.pipelineBarrier2(
	    vk::DependencyInfo {.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

bool Is2dColor(const ShaderTextureResource& descriptor) {
	return !descriptor.IsNull() &&
	       (descriptor.Type() == Prospero::ImageType::kColor2D ||
	        descriptor.Type() == Prospero::ImageType::kColor2DArray) &&
	       descriptor.TileMode() != Prospero::TileMode::kDepth;
}

// A request the pass skips: logged for its first occurrences.
void SkipRequest(const char* reason, const Pssr::Request& request) {
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16u) {
		LOGF("PSSR: request %ux%u skipped: %s (input T# %08x %08x %08x %08x, output T# %08x %08x "
		     "%08x %08x)\n",
		     request.width, request.height, reason, request.input[0], request.input[1],
		     request.input[2], request.input[3], request.output[0], request.output[1],
		     request.output[2], request.output[3]);
	}
}

} // namespace

struct RenderExecutor::PssrState {
	explicit PssrState(GraphicContext& graphics): device(graphics.device) {
		const std::array bindings {
		    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eSampledImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute},
		    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1,
		                                    vk::ShaderStageFlagBits::eCompute}};
		vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		descriptor_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		descriptor_info.bindingCount = static_cast<uint32_t>(bindings.size());
		descriptor_info.pBindings    = bindings.data();
		RequireVulkanSuccess(device.createDescriptorSetLayout(&descriptor_info, nullptr,
		                                                      &descriptor_layout),
		                     "create PSSR descriptor layout");
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0,
		                                  sizeof(PssrParameters)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &descriptor_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(device.createPipelineLayout(&layout_info, nullptr, &pipeline_layout),
		                     "create PSSR pipeline layout");
		pipeline = CreateComputePipeline(device, GPU_PSSR_LINEAR_SPV, pipeline_layout,
		                                 "create PSSR pipeline");
	}
	~PssrState() {
		for (const auto& context: dlss_contexts) {
			if (context.feature != nullptr) {
				Dlss::ReleaseFeature(context.feature);
			}
		}
		device.destroyPipeline(pipeline, nullptr);
		device.destroyPipelineLayout(pipeline_layout, nullptr);
		device.destroyDescriptorSetLayout(descriptor_layout, nullptr);
	}
	PssrState(const PssrState&)            = delete;
	PssrState& operator=(const PssrState&) = delete;

	vk::Device              device;
	vk::DescriptorSetLayout descriptor_layout = nullptr;
	vk::PipelineLayout      pipeline_layout   = nullptr;
	vk::Pipeline            pipeline          = nullptr;

	// One DLSS feature per PSSR context (the owner of the history); a library has at most four.
	struct DlssContext {
		uint64_t       context;
		Dlss::Creation creation;
		Dlss::Feature* feature;
	};
	std::vector<DlssContext> dlss_contexts;

	// Upstream RenderExecutor::Upscale (782a85c62): records DLSS for the request into command;
	// false when DLSS cannot take it (the linear pass runs instead).
	bool Upscale(CommandScheduler& scheduler, vk::CommandBuffer command,
	             const Pssr::Request& request, Dlss::Inputs& inputs) {
		const auto release = [&](Dlss::Feature* feature) {
			if (feature != nullptr) {
				scheduler.DeferOperation([feature] { Dlss::ReleaseFeature(feature); });
			}
		};
		auto context = std::ranges::find(dlss_contexts, request.context, &DlssContext::context);
		bool reset   = (request.flags & Pssr::kReset) != 0;
		if (context == dlss_contexts.end() || !Dlss::Takes(context->creation, request, inputs)) {
			// A render size no quality mode takes scales linearly.
			const auto creation = Dlss::Choose(*Dlss::Capabilities(), request, inputs);
			if (!creation) {
				return false;
			}
			if (context != dlss_contexts.end()) {
				release(context->feature);
			} else {
				if (dlss_contexts.size() == 4) {
					release(dlss_contexts.front().feature);
					dlss_contexts.erase(dlss_contexts.begin());
				}
				context = dlss_contexts.insert(dlss_contexts.end(), {request.context, {}, nullptr});
			}
			context->creation = *creation;
			context->feature  = Dlss::CreateFeature(command, *creation);
			reset             = true;
		}
		if (context->feature == nullptr) {
			return false;
		}
		DlssMemoryBarrier(command);
		if (!Dlss::Evaluate(command, context->feature, request, inputs, reset)) {
			release(context->feature);
			context->feature = nullptr;
			return false;
		}
		DlssMemoryBarrier(command);
		return true;
	}
};

void RenderExecutor::ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopReadyOperations();
	RenderLockGuard lock(m_context.GetMutex());
	// Like a dispatch, the pass ends the run of the draws before it (drawPrep/drawRun.h).
	BeginDrawRun();
	ShaderTextureResource input_descriptor {};
	ShaderTextureResource output_descriptor {};
	std::copy_n(request.input, 8, input_descriptor.fields);
	std::copy_n(request.output, 8, output_descriptor.fields);
	if (!Is2dColor(input_descriptor) || !Is2dColor(output_descriptor) ||
	    !Pssr::IsColorFormat(input_descriptor.Format()) ||
	    output_descriptor.Format() != Prospero::BufferFormat::k11_11_10Float ||
	    output_descriptor.TileMode() != Prospero::TileMode::kStandard256B ||
	    output_descriptor.BaseLevel() != 0 || output_descriptor.LastLevel() != 0 ||
	    output_descriptor.MaxMip() != 0 || output_descriptor.BaseArray5() != 0 ||
	    output_descriptor.Depth() != 0 || request.width == 0 || request.height == 0) {
		SkipRequest("unsupported input or output descriptor", request);
		return;
	}
	const auto input_width =
	    std::max(1u, (input_descriptor.Width5() + 1u) >> input_descriptor.BaseLevel());
	const auto input_height =
	    std::max(1u, (input_descriptor.Height5() + 1u) >> input_descriptor.BaseLevel());
	if (request.width > input_width || request.height > input_height) {
		SkipRequest("active rectangle larger than the input", request);
		return;
	}
	PssrParameters parameters {
	    {request.width, request.height},
	    {output_descriptor.Width5() + 1u, output_descriptor.Height5() + 1u},
	    {4, 4, 4, 4}};
	// The output channel c stores the colour component its descriptor swizzle reads it from.
	for (uint32_t component = 0; component < 3; ++component) {
		for (uint32_t source = 0; source < 4; ++source) {
			if (GetDstSel(output_descriptor.DstSelXYZW(), source) == 4u + component) {
				parameters.output_channels[component] = source;
				break;
			}
		}
	}
	const auto resolve = [&](const uint32_t* words, bool output) {
		ShaderRecompiler::IR::ImageResource resource {};
		resource.resource_class = output ? ShaderRecompiler::IR::ImageResourceClass::Storage
		                                 : ShaderRecompiler::IR::ImageResourceClass::Sampled;
		resource.numeric_class  = Prospero::TextureNumericClass::Float;
		resource.dimension      = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
		resource.read           = !output;
		resource.written        = output;
		ShaderRecompiler::IR::DescriptorValue value {};
		std::copy_n(words, 8, value.dwords.begin());
		value.dword_count = 8;
		auto binding      = ResolveTexture(resource, value);
		binding.desc.view_info.level_count = 1;
		binding.desc.view_info.min_lod     = 0;
		return binding;
	};
	auto& cache  = m_context.GetTextureCache();
	auto  input  = resolve(request.input, false);
	BindImage(input.image_id, false);
	auto output = resolve(request.output, true);
	BindImage(output.image_id, true);
	if (input.desc.info.data.address < output.desc.info.data.End() &&
	    output.desc.info.data.address < input.desc.info.data.End()) {
		ResetBindings();
		SkipRequest("input and output memory overlap", request);
		return;
	}
	// Depth, motion vectors and exposure for DLSS, which writes the output unswizzled.
	const std::array<const uint32_t*, 3>         dlss_words {request.depth, request.motion,
	                                                         request.exposure};
	std::array<std::optional<TextureBinding>, 3> dlss_bindings;
	if (Dlss::Capabilities() != nullptr && Dlss::Accepts(request) &&
	    parameters.output_channels[0] == 0 && parameters.output_channels[1] == 1 &&
	    parameters.output_channels[2] == 2) {
		for (size_t i = 0; i < dlss_words.size(); ++i) {
			ShaderTextureResource descriptor {};
			std::copy_n(dlss_words[i], 8, descriptor.fields);
			if (!descriptor.IsNull()) {
				dlss_bindings[i] = resolve(dlss_words[i], false);
				BindImage(dlss_bindings[i]->image_id, false);
			}
		}
	}
	// Resolving the output may have dropped or rebuilt the input's image (and the reverse for a
	// rediscovered binding): such a binding is resolved again before its view is acquired.
	const auto acquire = [&](TextureBinding& binding, const uint32_t* words, bool storage) {
		auto& previous = cache.GetImage(binding.image_id);
		if (!previous.registered || previous.binding.needs_rebind) {
			previous.binding = {};
			binding          = resolve(words, storage);
			BindImage(binding.image_id, storage);
		}
		binding.image_view = cache.FindTexture(binding.image_id, binding.desc);
	};
	for (size_t i = 0; i < dlss_words.size(); ++i) {
		if (dlss_bindings[i]) {
			acquire(*dlss_bindings[i], dlss_words[i], false);
		}
	}
	acquire(input, request.input, false);
	acquire(output, request.output, true);
	auto&       source           = cache.GetImage(input.image_id);
	auto&       destination      = cache.GetImage(output.image_id);
	const auto& source_view      = input.desc.view_info;
	const auto& destination_view = output.desc.view_info;
	if (source.info.IsDepth() || destination.info.IsDepth() || source.info.samples != 1 ||
	    destination.info.samples != 1 || input.image_view == nullptr ||
	    output.image_view == nullptr ||
	    (input.image_id == output.image_id && source_view.base_level == destination_view.base_level &&
	     source_view.base_layer == destination_view.base_layer)) {
		ResetBindings();
		SkipRequest("depth, multisampled or self-overlapping images", request);
		return;
	}
	if (m_pssr == nullptr) {
		m_pssr = std::make_shared<PssrState>(m_context.GetGraphics());
	}
	auto& pass = *m_pssr;
	buffer.EndRendering();
	// Transitions are batched (recorded before the dispatch below, Sink()).
	const auto identity = buffer.Identity();
	source.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
	               ImageSubresourceRange {source_view.base_level, 1, source_view.base_layer, 1},
	               identity, true);
	destination.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite,
	                    ImageSubresourceRange {destination_view.base_level, 1,
	                                           destination_view.base_layer, 1},
	                    identity, true);
	if (dlss_bindings[0] && dlss_bindings[1]) {
		const auto range = [](const ImageViewInfo& view) {
			return ImageSubresourceRange {view.base_level, 1, view.base_layer, 1};
		};
		for (const auto& binding: dlss_bindings) {
			if (binding) {
				cache.GetImage(binding->image_id)
				    .Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead,
				             range(binding->desc.view_info), identity, true);
			}
		}
		const auto resource = [&](const TextureBinding& binding, bool write) {
			const auto& image  = cache.GetImage(binding.image_id);
			const auto& view   = binding.desc.view_info;
			const auto  width  = std::max(binding.desc.info.extent.width >> view.base_level, 1u);
			const auto  height = std::max(binding.desc.info.extent.height >> view.base_level, 1u);
			// DLSS takes a depth image in its backing format and depth aspect.
			const bool depth  = image.info.IsDepth();
			const auto aspect = depth ? vk::ImageAspectFlagBits::eDepth : view.aspect;
			return Dlss::Resource {
			    .view       = static_cast<VkImageView>(binding.image_view),
			    .image      = static_cast<VkImage>(image.backing.image),
			    .range      = {static_cast<VkImageAspectFlags>(aspect), view.base_level, 1,
			                   view.base_layer, 1},
			    .format     = static_cast<VkFormat>(depth ? image.backing.format : view.format),
			    .width      = width,
			    .height     = height,
			    .type       = 0,
			    .read_write = static_cast<uint8_t>(write ? 1 : 0)};
		};
		auto exposure = dlss_bindings[2] ? resource(*dlss_bindings[2], false) : Dlss::Resource {};
		Dlss::Inputs inputs {resource(input, false), resource(*dlss_bindings[0], false),
		                     resource(*dlss_bindings[1], false),
		                     dlss_bindings[2] ? &exposure : nullptr, resource(output, true)};
		// The native handle records the batched transitions first (Handle()).
		if (pass.Upscale(m_context.GetCommandScheduler(), buffer.Handle(), request, inputs)) {
			ResetBindings();
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 4u) {
				LOGF("PSSR: DLSS upscale %ux%u -> %ux%u\n", request.width, request.height,
				     parameters.output_size[0], parameters.output_size[1]);
			}
			return;
		}
	}
	buffer.BeginEmission();
	const std::array images {
	    vk::DescriptorImageInfo {nullptr, input.image_view, vk::ImageLayout::eGeneral},
	    vk::DescriptorImageInfo {nullptr, output.image_view, vk::ImageLayout::eGeneral}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType =
		    i == 0 ? vk::DescriptorType::eSampledImage : vk::DescriptorType::eStorageImage;
		writes[i].pImageInfo = &images[i];
	}
	ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pass.pipeline);
	(void)buffer.PushDescriptors(vk::PipelineBindPoint::eCompute, pass.pipeline_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	buffer.PushConstants(pass.pipeline_layout, vk::ShaderStageFlagBits::eCompute,
	                     sizeof(parameters), &parameters);
	buffer.Sink().dispatch((parameters.output_size[0] + 7) / 8, (parameters.output_size[1] + 7) / 8,
	                       1);
	ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 4u) {
		LOGF("PSSR: linear upscale %ux%u -> %ux%u (input 0x%010" PRIx64 ", output 0x%010" PRIx64
		     ")\n",
		     request.width, request.height, parameters.output_size[0], parameters.output_size[1],
		     input_descriptor.Base40(), output_descriptor.Base40());
	}
}

} // namespace Libs::Graphics
