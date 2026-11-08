#include "graphics/host_gpu/renderer/pipeline/pipelineLayoutCache.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

struct SignatureHash {
	size_t operator()(const PipelineLayoutSignature& signature) const noexcept {
		return static_cast<size_t>(HashPipelineLayoutSignature(signature));
	}
};

struct DeviceLayouts {
	std::unordered_map<PipelineLayoutSignature, PipelineLayoutHandles, SignatureHash> layouts;
};

// Keyed by device: pipelines of one device share its layouts. Guarded by g_mutex, since
// pipelines may be created on more than one thread.
std::mutex                                                    g_mutex;
std::unordered_map<VkDevice, std::unique_ptr<DeviceLayouts>> g_devices;

PipelineLayoutHandles CreateLayouts(GraphicContext&                                 graphics,
                                    std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                    bool push_descriptors, vk::ShaderStageFlags push_stages,
                                    uint32_t push_size, vk::DescriptorSetLayout bindless_set) {
	PipelineLayoutHandles handles;
	handles.uses_push_descriptors = push_descriptors;

	vk::DescriptorSetLayoutCreateInfo set_create {};
	set_create.flags        = push_descriptors
	                              ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                              : vk::DescriptorSetLayoutCreateFlags {};
	set_create.bindingCount = static_cast<uint32_t>(bindings.size());
	set_create.pBindings    = bindings.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(&set_create, nullptr, &handles.set_layout) !=
	        vk::Result::eSuccess);
	if (!push_descriptors) {
		DescriptorHeap::RegisterLayout(handles.set_layout, bindings); // KYTY_DESCRIPTOR_FIT
	}

	const vk::PushConstantRange  push_constants {push_stages, 0, push_size};
	vk::PipelineLayoutCreateInfo layout_create {};
	const std::array<vk::DescriptorSetLayout, 2> set_layouts {handles.set_layout, bindless_set};
	layout_create.setLayoutCount         = bindless_set != nullptr ? 2u : 1u;
	layout_create.pSetLayouts            = set_layouts.data();
	layout_create.pushConstantRangeCount = 1;
	layout_create.pPushConstantRanges    = &push_constants;
	const auto result =
	    graphics.device.createPipelineLayout(&layout_create, nullptr, &handles.pipeline_layout);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || handles.pipeline_layout == nullptr);
	return handles;
}

} // namespace

bool PipelineLayoutInterningEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_LAYOUT_INTERN");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

PipelineLayoutSignature
MakePipelineLayoutSignature(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                            vk::ShaderStageFlags push_stages, uint32_t push_size,
                            uint32_t max_push_descriptors) {
	PipelineLayoutSignature signature;
	signature.bindings.reserve(bindings.size());
	uint64_t descriptor_count = 0;
	for (const auto& binding: bindings) {
		EXIT_IF(binding.pImmutableSamplers != nullptr);
		signature.bindings.push_back(
		    {binding.binding, static_cast<uint32_t>(binding.descriptorType),
		     binding.descriptorCount,
		     static_cast<uint32_t>(static_cast<VkShaderStageFlags>(binding.stageFlags))});
		descriptor_count += binding.descriptorCount;
	}
	// A set layout's bindings are identified by binding number; their order in pBindings has no
	// meaning, so sorting only lets equal sets share a key.
	std::ranges::sort(signature.bindings);
	signature.push_stages      = static_cast<uint32_t>(static_cast<VkShaderStageFlags>(push_stages));
	signature.push_size        = push_size;
	signature.push_descriptors = descriptor_count <= max_push_descriptors;
	return signature;
}

uint64_t HashPipelineLayoutSignature(const PipelineLayoutSignature& signature) {
	const uint64_t seed = (static_cast<uint64_t>(signature.push_stages) << 32u) ^
	                      (static_cast<uint64_t>(signature.push_size) << 1u) ^
	                      (signature.push_descriptors ? 1u : 0u) ^ (signature.bindless_set * 0x9e3779b97f4a7c15ull);
	return XXH3_64bits_withSeed(signature.bindings.data(),
	                            signature.bindings.size() * sizeof(signature.bindings[0]), seed);
}

PipelineLayoutHandles AcquirePipelineLayout(GraphicContext& graphics,
                                            std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                            vk::ShaderStageFlags push_stages, uint32_t push_size,
                                            vk::DescriptorSetLayout bindless_set) {
	auto signature = MakePipelineLayoutSignature(bindings, push_stages, push_size,
	                                             graphics.max_push_descriptors);
	signature.bindless_set =
	    reinterpret_cast<uint64_t>(static_cast<VkDescriptorSetLayout>(bindless_set));
	if (!PipelineLayoutInterningEnabled()) {
		return CreateLayouts(graphics, bindings, signature.push_descriptors, push_stages, push_size,
		                     bindless_set);
	}
	std::scoped_lock lock {g_mutex};
	auto& device = g_devices[static_cast<VkDevice>(graphics.device)];
	if (!device) {
		device = std::make_unique<DeviceLayouts>();
	}
	if (const auto found = device->layouts.find(signature); found != device->layouts.end()) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineLayoutsShared);
		return found->second;
	}
	std::vector<vk::DescriptorSetLayoutBinding> sorted(bindings.begin(), bindings.end());
	std::ranges::sort(sorted, {}, &vk::DescriptorSetLayoutBinding::binding);
	const auto handles = CreateLayouts(graphics, sorted, signature.push_descriptors, push_stages,
	                                   push_size, bindless_set);
	device->layouts.emplace(std::move(signature), handles);
	Profiler::CountFrameEvent(Profiler::FrameEvent::PipelineLayoutsCreated);
	return handles;
}

void ReleasePipelineLayout(GraphicContext& graphics, vk::PipelineLayout pipeline_layout,
                           vk::DescriptorSetLayout set_layout) {
	if (PipelineLayoutInterningEnabled()) {
		return;
	}
	if (pipeline_layout != nullptr) {
		graphics.device.destroyPipelineLayout(pipeline_layout, nullptr);
	}
	if (set_layout != nullptr) {
		graphics.device.destroyDescriptorSetLayout(set_layout, nullptr);
	}
}

void DestroyInternedPipelineLayouts(GraphicContext& graphics) {
	std::unique_ptr<DeviceLayouts> device;
	{
		std::scoped_lock lock {g_mutex};
		const auto found = g_devices.find(static_cast<VkDevice>(graphics.device));
		if (found == g_devices.end()) {
			return;
		}
		device = std::move(found->second);
		g_devices.erase(found);
	}
	if (!device) {
		return;
	}
	for (const auto& [signature, handles]: device->layouts) {
		(void)signature;
		graphics.device.destroyPipelineLayout(handles.pipeline_layout, nullptr);
		graphics.device.destroyDescriptorSetLayout(handles.set_layout, nullptr);
	}
}

} // namespace Libs::Graphics
