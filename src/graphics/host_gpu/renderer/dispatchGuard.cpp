#include "graphics/host_gpu/renderer/dispatchGuard.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_mesh_shaders/gpu_dispatch_guard_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

struct PushData {
	uint32_t args_word = 0;
	uint32_t out_word  = 0;
	uint32_t max_dim   = 0;
	uint32_t max_total = 0;
	uint32_t local[3]  = {0, 0, 0};
};

constexpr uint32_t SlotBytes = 32;
constexpr uint64_t RingSize  = uint64_t {1024} * SlotBytes;

uint32_t EnvLimit(const char* name, uint32_t fallback) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0))
	                                             : fallback;
}

} // namespace

bool DispatchGuard::Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_INDIRECT_GUARD");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

DispatchGuard::DispatchGuard(RenderContext& context): m_context(context) {}

DispatchGuard::~DispatchGuard() {
	auto device = m_context.GetGraphics().device;
	if (m_pipeline) device.destroyPipeline(m_pipeline);
	if (m_layout) device.destroyPipelineLayout(m_layout);
	if (m_descriptors) device.destroyDescriptorSetLayout(m_descriptors);
}

void DispatchGuard::Initialize() {
	if (m_pipeline) {
		return;
	}
	auto& graphics = m_context.GetGraphics();
	const std::array<vk::DescriptorSetLayoutBinding, 2> bindings {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptor {};
	descriptor.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor, nullptr, &m_descriptors),
	                     "create dispatch-guard descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushData)};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &m_layout),
	                     "create dispatch-guard layout");
	const auto module = CompileSPV(GPU_DISPATCH_GUARD_SPV, graphics.device);
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.layout       = m_layout;
	pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName  = "main";
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr,
	                                                           &m_pipeline);
	graphics.device.destroyShaderModule(module);
	RequireVulkanSuccess(result, "create dispatch-guard pipeline");
	m_ring = std::make_unique<StreamBuffer>(graphics, m_context.GetCommandScheduler(),
	                                        MemoryUsage::Download, RingSize, false,
	                                        vk::BufferUsageFlagBits::eShaderDeviceAddress);
	SetVulkanObjectNameF(graphics.device, m_ring->Handle(), "Kyty.DispatchGuardRing");
}

DispatchGuard::Slot DispatchGuard::Reserve() {
	Initialize();
	const auto alignment =
	    std::max<uint64_t>(SlotBytes, m_context.GetGraphics().StorageMinAlignment());
	const auto [mapped, offset] = m_ring->Map(SlotBytes, alignment);
	EXIT_IF(mapped == nullptr);
	m_ring->Commit();
	return Slot {m_ring->Handle(), offset};
}

vk::DeviceAddress DispatchGuard::SlotThreads(const Slot& slot) const {
	EXIT_IF(m_ring == nullptr || slot.buffer != m_ring->Handle());
	return m_ring->BufferDeviceAddress() + slot.offset + 4u * sizeof(uint32_t);
}

void DispatchGuard::Record(CommandBuffer& buffer, const Slot& slot, vk::Buffer args,
                           uint64_t args_offset, uint64_t shader_hash,
                           const std::array<uint32_t, 3>& local_size, bool guard) {
	Initialize();
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	// Every earlier write (the record's producer) before the guard reads it.
	buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllCommands,
	                            vk::AccessFlagBits2::eMemoryWrite,
	                            vk::PipelineStageFlagBits2::eComputeShader,
	                            vk::AccessFlagBits2::eShaderStorageRead, BarrierOrigin::IndirectArgs);
	const auto aligned = Common::AlignDown(args_offset, graphics.StorageMinAlignment());
	EXIT_IF(((args_offset - aligned) & 3u) != 0);
	const std::array<vk::DescriptorBufferInfo, 2> infos {{
	    {args, aligned, args_offset - aligned + 12u},
	    {slot.buffer, slot.offset, SlotBytes},
	}};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	(void)buffer.PushDescriptors(vk::PipelineBindPoint::eCompute, m_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	static const uint32_t max_dim   = EnvLimit("KYTY_INDIRECT_GUARD_MAX_DIM", 65535u);
	static const uint32_t max_total = EnvLimit("KYTY_INDIRECT_GUARD_MAX_GROUPS", 1u << 22u);
	PushData push;
	push.args_word = static_cast<uint32_t>((args_offset - aligned) / 4u);
	push.out_word  = 0;
	push.max_dim   = guard ? max_dim : 0u;
	push.max_total = max_total;
	for (uint32_t i = 0; i < 3u; i++) {
		push.local[i] = local_size[0] != 0 ? std::max(local_size[i], 1u) : 0u;
	}
	// Through the command buffer's push-constant shadow: the guest pipeline's push data that
	// follows must not be taken as already pushed.
	buffer.PushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, sizeof(push), &push);
	buffer.Sink().dispatch(1, 1, 1);
	// The guard's write before the indirect read of the dispatch that follows (and a host read, and
	// the dispatch's own read of the thread counts with KYTY_INDIRECT_THREADS_GPU).
	buffer.RequestMemoryBarrier(
	    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	    vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eHost |
	        vk::PipelineStageFlagBits2::eComputeShader,
	    vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eHostRead |
	        vk::AccessFlagBits2::eShaderStorageRead,
	    BarrierOrigin::IndirectArgs);
	if (guard) {
		scheduler.DeferPriorityOperation([this, slot, shader_hash] { Check(slot, shader_hash); });
	}
}

void DispatchGuard::Check(const Slot& slot, uint64_t shader_hash) {
	m_ring->Invalidate(slot.offset, SlotBytes);
	uint32_t words[8] {};
	std::memcpy(words, m_ring->Mapped().data() + slot.offset, sizeof(words));
	static std::atomic<uint32_t> seen {0};
	static std::atomic<uint32_t> replaced {0};
	const bool                   trace = std::getenv("KYTY_INDIRECT_GUARD_TRACE") != nullptr;
	if (words[3] != 0) {
		const auto count = replaced.fetch_add(1, std::memory_order_relaxed);
		if (count < 32u) {
			std::printf("DispatchGuard: indirect dispatch of CS 0x%016" PRIx64
			            " had counts 0x%08x x 0x%08x x 0x%08x; dispatched as 0 (replaced #%u)\n",
			            shader_hash, words[4], words[5], words[6], count + 1u);
			std::fflush(stdout);
		}
	} else if (trace && seen.fetch_add(1, std::memory_order_relaxed) < 64u) {
		std::printf("DispatchGuard: CS 0x%016" PRIx64 " counts %u x %u x %u\n", shader_hash,
		            words[4], words[5], words[6]);
		std::fflush(stdout);
	}
}

} // namespace Libs::Graphics
