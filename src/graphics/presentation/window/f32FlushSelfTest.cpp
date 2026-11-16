#include "graphics/presentation/window/f32FlushSelfTest.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace Libs::Graphics {

namespace FlushSelfTest = ShaderRecompiler::FlushSelfTest;

namespace {

// Host-visible memory for the job's one buffer (coherent and cached preferred).
uint32_t FindMemoryType(const vk::PhysicalDeviceMemoryProperties& memory, uint32_t allowed, bool* coherent) {
	using Flag = vk::MemoryPropertyFlagBits;
	const vk::MemoryPropertyFlags wanted[] = {Flag::eHostVisible | Flag::eHostCoherent | Flag::eHostCached,
	                                          Flag::eHostVisible | Flag::eHostCoherent, Flag::eHostVisible};
	for (const auto flags: wanted) {
		for (uint32_t type = 0; type < memory.memoryTypeCount; type++) {
			if ((allowed & (1u << type)) != 0u && (memory.memoryTypes[type].propertyFlags & flags) == flags) {
				*coherent = static_cast<bool>(memory.memoryTypes[type].propertyFlags & Flag::eHostCoherent);
				return type;
			}
		}
	}
	return UINT32_MAX;
}

// The job's Vulkan objects, destroyed in reverse order unless the GPU may still use them.
struct Job {
	vk::Device              device;
	vk::Buffer              buffer;
	vk::DeviceMemory        memory;
	vk::DescriptorSetLayout set_layout;
	vk::PipelineLayout      pipeline_layout;
	vk::DescriptorPool      pool;
	vk::ShaderModule        module;
	vk::Pipeline            pipeline;
	vk::CommandPool         command_pool;
	vk::Fence               fence;
	bool                    in_flight = false;

	~Job() {
		if (in_flight || device == nullptr) {
			return; // a job that did not finish keeps its objects (a few MiB, once)
		}
		if (fence) device.destroyFence(fence, nullptr);
		if (command_pool) device.destroyCommandPool(command_pool, nullptr);
		if (pipeline) device.destroyPipeline(pipeline, nullptr);
		if (module) device.destroyShaderModule(module, nullptr);
		if (pool) device.destroyDescriptorPool(pool, nullptr);
		if (pipeline_layout) device.destroyPipelineLayout(pipeline_layout, nullptr);
		if (set_layout) device.destroyDescriptorSetLayout(set_layout, nullptr);
		if (buffer) device.destroyBuffer(buffer, nullptr);
		if (memory) device.freeMemory(memory, nullptr);
	}
};

bool Check(vk::Result result, const char* what, std::string* error) {
	if (result == vk::Result::eSuccess) {
		return true;
	}
	*error = std::string(what) + " failed (" + vk::to_string(result) + ")";
	return false;
}

bool Dispatch(vk::Device device, const vk::PhysicalDeviceMemoryProperties& memory_properties, vk::Queue queue,
              uint32_t queue_family, bool force_mismatch, F32FlushSelfTestRun* run) {
	auto*      error = &run->error;
	auto       mark  = std::chrono::steady_clock::now();
	const auto phase = [&mark](double* out) {
		const auto now = std::chrono::steady_clock::now();
		*out           = std::chrono::duration<double, std::milli>(now - mark).count();
		mark           = now;
	};
	const auto code = FlushSelfTest::EmitModule(error);
	phase(&run->emit_ms);
	if (code.empty()) {
		return false;
	}
	Job job;
	job.device       = device;
	const auto bytes = static_cast<vk::DeviceSize>(FlushSelfTest::BufferWords) * 4u;

	vk::BufferCreateInfo buffer_info {};
	buffer_info.size        = bytes;
	buffer_info.usage       = vk::BufferUsageFlagBits::eStorageBuffer;
	buffer_info.sharingMode = vk::SharingMode::eExclusive;
	if (!Check(device.createBuffer(&buffer_info, nullptr, &job.buffer), "vkCreateBuffer", error)) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	device.getBufferMemoryRequirements(job.buffer, &requirements);
	bool       coherent = false;
	const auto type     = FindMemoryType(memory_properties, requirements.memoryTypeBits, &coherent);
	if (type == UINT32_MAX) {
		*error = "no host-visible memory type for the buffer";
		return false;
	}
	vk::MemoryAllocateInfo allocate {};
	allocate.allocationSize  = requirements.size;
	allocate.memoryTypeIndex = type;
	if (!Check(device.allocateMemory(&allocate, nullptr, &job.memory), "vkAllocateMemory", error) ||
	    !Check(device.bindBufferMemory(job.buffer, job.memory, 0), "vkBindBufferMemory", error)) {
		return false;
	}
	void* mapped = nullptr;
	if (!Check(device.mapMemory(job.memory, 0, VK_WHOLE_SIZE, {}, &mapped), "vkMapMemory", error)) {
		return false;
	}
	auto* words = static_cast<uint32_t*>(mapped);
	{
		const auto initial = FlushSelfTest::InitialBuffer();
		std::memcpy(words, initial.data(), initial.size() * sizeof(uint32_t));
	}
	vk::MappedMemoryRange range {};
	range.memory = job.memory;
	range.offset = 0;
	range.size   = VK_WHOLE_SIZE;
	if (!coherent && !Check(device.flushMappedMemoryRanges(1, &range), "vkFlushMappedMemoryRanges", error)) {
		return false;
	}

	vk::DescriptorSetLayoutBinding binding {};
	binding.binding         = 0;
	binding.descriptorType  = vk::DescriptorType::eStorageBuffer;
	binding.descriptorCount = 1;
	binding.stageFlags      = vk::ShaderStageFlagBits::eCompute;
	vk::DescriptorSetLayoutCreateInfo set_layout_info {};
	set_layout_info.bindingCount = 1;
	set_layout_info.pBindings    = &binding;
	if (!Check(device.createDescriptorSetLayout(&set_layout_info, nullptr, &job.set_layout),
	           "vkCreateDescriptorSetLayout", error)) {
		return false;
	}
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount = 1;
	pipeline_layout_info.pSetLayouts    = &job.set_layout;
	if (!Check(device.createPipelineLayout(&pipeline_layout_info, nullptr, &job.pipeline_layout),
	           "vkCreatePipelineLayout", error)) {
		return false;
	}
	vk::DescriptorPoolSize pool_size {};
	pool_size.type            = vk::DescriptorType::eStorageBuffer;
	pool_size.descriptorCount = 1;
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	if (!Check(device.createDescriptorPool(&pool_info, nullptr, &job.pool), "vkCreateDescriptorPool", error)) {
		return false;
	}
	vk::DescriptorSetAllocateInfo set_info {};
	set_info.descriptorPool     = job.pool;
	set_info.descriptorSetCount = 1;
	set_info.pSetLayouts        = &job.set_layout;
	vk::DescriptorSet set;
	if (!Check(device.allocateDescriptorSets(&set_info, &set), "vkAllocateDescriptorSets", error)) {
		return false;
	}
	vk::DescriptorBufferInfo descriptor {};
	descriptor.buffer = job.buffer;
	descriptor.offset = 0;
	descriptor.range  = bytes;
	vk::WriteDescriptorSet write {};
	write.dstSet          = set;
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &descriptor;
	device.updateDescriptorSets(1, &write, 0, nullptr);

	vk::ShaderModuleCreateInfo module_info {};
	module_info.codeSize = code.size() * sizeof(uint32_t);
	module_info.pCode    = code.data();
	if (!Check(device.createShaderModule(&module_info, nullptr, &job.module), "vkCreateShaderModule", error)) {
		return false;
	}
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = job.module;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = job.pipeline_layout;
	if (!Check(device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &job.pipeline),
	           "vkCreateComputePipelines", error)) {
		return false;
	}

	vk::CommandPoolCreateInfo command_pool_info {};
	command_pool_info.flags            = vk::CommandPoolCreateFlagBits::eTransient;
	command_pool_info.queueFamilyIndex = queue_family;
	if (!Check(device.createCommandPool(&command_pool_info, nullptr, &job.command_pool), "vkCreateCommandPool",
	           error)) {
		return false;
	}
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = job.command_pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer commands;
	if (!Check(device.allocateCommandBuffers(&command_info, &commands), "vkAllocateCommandBuffers", error)) {
		return false;
	}
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	if (!Check(commands.begin(&begin), "vkBeginCommandBuffer", error)) {
		return false;
	}
	commands.bindPipeline(vk::PipelineBindPoint::eCompute, job.pipeline);
	commands.bindDescriptorSets(vk::PipelineBindPoint::eCompute, job.pipeline_layout, 0, 1, &set, 0, nullptr);
	commands.dispatch(FlushSelfTest::InputCount / FlushSelfTest::WorkgroupSize, 1, 1);
	vk::MemoryBarrier to_host {};
	to_host.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	to_host.dstAccessMask = vk::AccessFlagBits::eHostRead;
	commands.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eHost, {}, 1,
	                         &to_host, 0, nullptr, 0, nullptr);
	if (!Check(commands.end(), "vkEndCommandBuffer", error)) {
		return false;
	}
	phase(&run->pipeline_ms);
	vk::FenceCreateInfo fence_info {};
	if (!Check(device.createFence(&fence_info, nullptr, &job.fence), "vkCreateFence", error)) {
		return false;
	}
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &commands;
	if (!Check(queue.submit(1, &submit, job.fence), "vkQueueSubmit", error)) {
		return false;
	}
	job.in_flight = true;
	if (!Check(device.waitForFences(1, &job.fence, VK_TRUE, 5'000'000'000ull), "vkWaitForFences (5 s)", error)) {
		return false;
	}
	job.in_flight = false;
	phase(&run->gpu_ms);
	if (!coherent && !Check(device.invalidateMappedMemoryRanges(1, &range), "vkInvalidateMappedMemoryRanges", error)) {
		return false;
	}
	std::vector<uint32_t> result(words, words + FlushSelfTest::BufferWords);
	device.unmapMemory(job.memory);
	if (force_mismatch) {
		FlushSelfTest::ForceMismatch(result);
	}
	run->ran       = true;
	run->verdict   = FlushSelfTest::Classify(result);
	run->reference = FlushSelfTest::ClassifyReference(result);
	phase(&run->check_ms);
	return true;
}

} // namespace

F32FlushSelfTestRun RunF32FlushSelfTest(vk::Device device, const vk::PhysicalDeviceMemoryProperties& memory,
                                        vk::Queue queue, uint32_t queue_family, bool force_mismatch) {
	F32FlushSelfTestRun run;
	const auto          start = std::chrono::steady_clock::now();
	(void)Dispatch(device, memory, queue, queue_family, force_mismatch, &run);
	run.milliseconds =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	return run;
}

void SettleF32FlushHostSelfTest(GraphicContext& graphics) {
	if (!ShaderRecompiler::F32FlushHostAwaitsSelfTest()) {
		return;
	}
	const auto             vendor = graphics.physical_device_properties.vendorID;
	const char*            value  = std::getenv("KYTY_F32_FLUSH_SELFTEST");
	const std::string_view setting(value != nullptr ? value : "");
	if (setting == "off") {
		ShaderRecompiler::SettleF32FlushHost(FlushSelfTest::ClassAll);
		std::printf("Kyty GPU capabilities: KYTY_F32_FLUSH_HOST=nvidia is on for all op classes (device vendor "
		            "0x%04x; self-test skipped: KYTY_F32_FLUSH_SELFTEST=off)\n",
		            vendor);
		std::fflush(stdout);
		return;
	}
	const bool force   = setting == "fail";
	const auto run     = RunF32FlushSelfTest(graphics.device, graphics.physical_device_memory_properties,
	                                         graphics.queue, graphics.queue_family, force);
	// Per op class: the classes whose every result matched keep the host flush.
	const auto classes = run.ran ? FlushSelfTest::PassingClasses(run.verdict) : uint8_t {0};
	ShaderRecompiler::SettleF32FlushHost(classes);
	char phases[128];
	std::snprintf(phases, sizeof(phases), "%.1f ms: emit %.1f, pipeline %.1f, GPU %.1f, check %.1f",
	              run.milliseconds, run.emit_ms, run.pipeline_ms, run.gpu_ms, run.check_ms);
	const auto reference =
	    run.reference.mismatches == 0
	        ? std::string("reference = IEEE model")
	        : "reference differs from the IEEE model in " + std::to_string(run.reference.mismatches) +
	              " results, first " + run.reference.first;
	if (!run.ran) {
		std::printf("Kyty GPU capabilities: KYTY_F32_FLUSH_HOST=nvidia is off: the self-test could not run "
		            "(device vendor 0x%04x; %s; %s)\n",
		            vendor, run.error.c_str(), phases);
	} else if (run.verdict.mismatches == 0) {
		std::printf("Kyty GPU capabilities: KYTY_F32_FLUSH_HOST=nvidia is on for all op classes (device vendor "
		            "0x%04x; self-test passed: %" PRIu64 " results of rcp/rsq/sqrt/exp/log and 12 compares on %u "
		            "denormals equal the flushed inputs' results, %s; %s)\n",
		            vendor, run.verdict.checked, FlushSelfTest::InputCount, reference.c_str(), phases);
	} else {
		const auto off   = static_cast<uint8_t>(FlushSelfTest::ClassAll & ~classes);
		const auto state = classes != 0u ? "on for " + FlushSelfTest::ClassesText(classes) + ", off for " +
		                                       FlushSelfTest::ClassesText(off)
		                                 : std::string("off: the self-test failed");
		std::printf("Kyty GPU capabilities: KYTY_F32_FLUSH_HOST=nvidia is %s (device vendor "
		            "0x%04x; self-test: %" PRIu64 " of %" PRIu64 " results differ (%s), first %s; %s; %s)%s\n",
		            state.c_str(), vendor, run.verdict.mismatches, run.verdict.checked,
		            run.verdict.ByOpText().c_str(), run.verdict.first.c_str(), reference.c_str(), phases,
		            force ? " (forced by KYTY_F32_FLUSH_SELFTEST=fail)" : "");
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics
