#include "graphics/host_gpu/renderer/occlusion.h"
#include "common/hangTrace.h"
#include "common/alignment.h"
#include "common/profiler.h"
#include "common/liveSwitch.h"
#include "gpu_dcc_shaders/gpu_dcc_occlusion_batch_spv.h"
#include "gpu_dcc_shaders/gpu_dcc_occlusion_spv.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {
namespace {
// Reset a bounded unused query prefix at once. Each query still begins and ends at exactly the
// old points, and every copy/reduction/publication is unchanged. Safe to switch per preparation.
Live::Switch g_reset_batch("KYTY_OCCLUSION_RESET_BATCH", Live::ParseDefaultOff);
}
bool OcclusionCounter::Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GPU_OCCLUSION");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

bool OcclusionCounter::BatchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OCCLUSION_BATCH");
		const bool  on    = value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "verify") == 0);
		if (Enabled()) {
			std::printf("Occlusion counter: batched reductions %s (KYTY_OCCLUSION_BATCH)\n",
			            on ? (BatchVerifyEnabled() ? "on, verified against per-dump reductions" : "on") : "off");
		}
		return on;
	}();
	return enabled;
}

bool OcclusionCounter::BatchVerifyEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OCCLUSION_BATCH");
		return value != nullptr && std::strcmp(value, "verify") == 0;
	}();
	return enabled;
}

uint32_t OcclusionCounter::SlotCount() {
	static const uint32_t count = [] {
		uint32_t slots = DefaultPublishSlots;
		if (const auto* value = std::getenv("KYTY_OCCLUSION_SLOTS"); value != nullptr && value[0] != '\0') {
			const auto parsed = std::strtoul(value, nullptr, 10);
			if (parsed >= 16 && parsed <= 262144) {
				slots = static_cast<uint32_t>(parsed);
			}
		}
		if (Enabled() && slots != DefaultPublishSlots) {
			std::printf("Occlusion counter: %u publish slots (KYTY_OCCLUSION_SLOTS)\n", slots);
		}
		return slots;
	}();
	return count;
}

void OcclusionCounter::ReferenceBatch(uint64_t counter, const std::vector<uint64_t>& results,
                                      const std::vector<uint32_t>& prefixes,
                                      std::vector<uint64_t>& values, uint64_t& counter_after) {
	// What the per-dump path computes: before each dump the queries pending since the previous one
	// are added to the counter (masked to 63 bits), and the dump publishes the counter.
	constexpr uint64_t mask = (1ull << 63u) - 1u;
	values.clear();
	uint64_t sum   = counter;
	size_t   query = 0;
	for (const auto prefix: prefixes) {
		const auto end = std::min<size_t>(prefix, results.size());
		for (; query < end; ++query) {
			sum += results[query];
		}
		sum &= mask;
		values.push_back(sum);
	}
	for (; query < results.size(); ++query) {
		sum += results[query];
	}
	counter_after = sum & mask;
}

OcclusionCounter::OcclusionCounter(RenderContext& context): m_context(context) {}
OcclusionCounter::~OcclusionCounter() {
	// RenderContext drains its scheduler before member destruction (the scheduler is declared
	// before this counter, so it is still alive here).
	if (m_batch_pipeline) {
		m_context.GetCommandScheduler().SetPreSubmitHook(nullptr, nullptr);
	}
	auto device = m_context.GetGraphics().device;
	if (m_pool) device.destroyQueryPool(m_pool);
	if (m_pipeline) device.destroyPipeline(m_pipeline);
	if (m_layout) device.destroyPipelineLayout(m_layout);
	if (m_descriptors) device.destroyDescriptorSetLayout(m_descriptors);
	if (m_batch_pipeline) device.destroyPipeline(m_batch_pipeline);
	if (m_batch_layout) device.destroyPipelineLayout(m_batch_layout);
	if (m_batch_descriptors) device.destroyDescriptorSetLayout(m_batch_descriptors);
}

void OcclusionCounter::Initialize() {
	if (m_pool) return;
	auto& graphics = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	EXIT_IF(!graphics.precise_occlusion_enabled || graphics.max_push_descriptors < 3);
	vk::QueryPoolCreateInfo query {};
	query.queryType = vk::QueryType::eOcclusion;
	query.queryCount = QueryCapacity;
	RequireVulkanSuccess(graphics.device.createQueryPool(&query, nullptr, &m_pool), "create occlusion pool");
	const std::array<vk::DescriptorSetLayoutBinding, 3> bindings {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptor {};
	descriptor.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor.pBindings = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor, nullptr, &m_descriptors),
	                     "create occlusion descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 12};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount = 1;
	layout.pSetLayouts = &m_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &m_layout), "create occlusion layout");
	const auto module = CompileSPV(GPU_DCC_OCCLUSION_SPV, graphics.device);
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.layout = m_layout;
	pipeline.stage.stage = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName = "main";
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &m_pipeline);
	graphics.device.destroyShaderModule(module);
	RequireVulkanSuccess(result, "create occlusion reduction pipeline");
	const auto usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
	m_slot_count = SlotCount();
	m_slot_ticks.assign(m_slot_count, 0);
	m_counter = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage, 256);
	m_result = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage,
	                                  QueryCapacity * sizeof(uint64_t));
	m_publish = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, usage,
	                                   uint64_t {m_slot_count} * PublishSlotSize);
	scheduler.Current().Handle().fillBuffer(m_counter->Handle(), 0, 256, 0);
	if (BatchEnabled()) {
		InitializeBatch();
	}
}

void OcclusionCounter::InitializeBatch() {
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	EXIT_IF(graphics.max_push_descriptors < 4);
	const std::array<vk::DescriptorSetLayoutBinding, 4> bindings {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptor {};
	descriptor.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor, nullptr, &m_batch_descriptors),
	                     "create occlusion batch descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, 16};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_batch_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &m_batch_layout),
	                     "create occlusion batch layout");
	const auto module = CompileSPV(GPU_DCC_OCCLUSION_BATCH_SPV, graphics.device);
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.layout       = m_batch_layout;
	pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName  = "main";
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr, &m_batch_pipeline);
	graphics.device.destroyShaderModule(module);
	RequireVulkanSuccess(result, "create occlusion batch pipeline");
	// Written by the CPU at each dump, read by the batch dispatch of that dump's command buffer;
	// an entry is rewritten only after its slot's previous tick completed (Dump).
	m_prefix = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                    vk::BufferUsageFlagBits::eStorageBuffer,
	                                    uint64_t {m_slot_count} * TableEntrySize);
	if (BatchVerifyEnabled()) {
		const auto usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;
		m_verify_counter = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage, 256);
		m_verify_result  = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage,
		                                            QueryCapacity * sizeof(uint64_t));
		m_verify_publish = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, usage,
		                                            uint64_t {m_slot_count} * PublishSlotSize);
		scheduler.Current().Handle().fillBuffer(m_verify_counter->Handle(), 0, 256, 0);
	}
	scheduler.SetPreSubmitHook(&OcclusionCounter::PreSubmit, this);
}

void OcclusionCounter::VerifyDispatch(uint32_t mode, vk::Buffer output, uint64_t offset, uint64_t range,
                                      uint32_t count) {
	// The per-dump reduction (Dispatch) on the verify buffers.
	auto&      command   = m_context.GetCommandScheduler().Current();
	const auto sink      = command.Sink();
	const auto alignment = m_context.GetGraphics().StorageMinAlignment();
	const auto aligned   = Common::AlignDown(offset, alignment);
	const vk::DescriptorBufferInfo infos[] {{m_verify_result->Handle(), 0, m_verify_result->Size()},
	                                        {m_verify_counter->Handle(), 0, 8},
	                                        {output, aligned, range + offset - aligned}};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {},
	                     1, &barrier, 0, nullptr, 0, nullptr);
	command.BindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.PushDescriptors(vk::PipelineBindPoint::eCompute, m_layout, 0, static_cast<uint32_t>(writes.size()),
	                        writes.data());
	const uint32_t push[] {mode, static_cast<uint32_t>(offset - aligned), count};
	command.PushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, sizeof(push), push);
	sink.dispatch(1, 1, 1);
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands, {},
	                     1, &barrier, 0, nullptr, 0, nullptr);
}

void OcclusionCounter::VerifyReduce() {
	// Adds the queries ended since the last verify reduction to the verify counter (outside
	// rendering), exactly as FlushPending adds the pending ones without the batch.
	KYTY_GPU_OP_SITE("occlusion.verify");
	const uint32_t fresh = m_pending - m_verified;
	if (fresh == 0) return;
	m_context.GetCommandScheduler().Current().Sink().copyQueryPoolResults(
	    m_pool, m_verified, fresh, m_verify_result->Handle(), 0, 8,
	    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	VerifyDispatch(0, m_verify_counter->Handle(), 0, m_verify_counter->Size(), fresh);
	m_verified = m_pending;
}

void OcclusionCounter::PreSubmit(void* context) {
	static_cast<OcclusionCounter*>(context)->FlushBatch();
}

void OcclusionCounter::FlushBatch() {
	KYTY_GPU_OP_SITE("occlusion.batch");
	if (!m_batch_pipeline) return;
	auto& scheduler = m_context.GetCommandScheduler();
	if (m_active) {
		// A counted instance is open (the scheduler is submitting mid-instance): end it so its
		// query joins this batch. Ending rendering may itself flush (pool full, Accumulate).
		scheduler.EndRendering();
	}
	if (m_batch_count == 0 && m_pending == 0) return;
	EXIT_IF(m_active || m_prepared);
	// The batch's dumps took this tick as their completion tick: the reduction must be recorded
	// into this very command buffer (nothing below may submit it).
	const auto tick    = scheduler.CurrentTick();
	if (m_verify_counter) {
		VerifyReduce(); // the queries after the batch's last dump reach the verify counter too
	}
	auto&      command = scheduler.Current();
	const auto sink    = command.Sink();
	if (m_pending != 0) {
		sink.copyQueryPoolResults(m_pool, 0, m_pending, m_result->Handle(), 0, 8,
		                          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	}
	const vk::DescriptorBufferInfo infos[] {{m_result->Handle(), 0, m_result->Size()},
	                                        {m_counter->Handle(), 0, 8},
	                                        {m_publish->Handle(), 0, m_publish->Size()},
	                                        {m_prefix->Handle(), 0, m_prefix->Size()}};
	std::array<vk::WriteDescriptorSet, 4> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader, {},
	                     1, &barrier, 0, nullptr, 0, nullptr);
	command.BindPipeline(vk::PipelineBindPoint::eCompute, m_batch_pipeline);
	command.PushDescriptors(vk::PipelineBindPoint::eCompute, m_batch_layout, 0,
	                        static_cast<uint32_t>(writes.size()), writes.data());
	const uint32_t push[] {m_batch_first_slot, m_batch_count, m_pending, m_slot_count};
	command.PushConstants(m_batch_layout, vk::ShaderStageFlagBits::eCompute, sizeof(push), push);
	sink.dispatch(1, 1, 1);
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands, {},
	                     1, &barrier, 0, nullptr, 0, nullptr);
	EXIT_IF(scheduler.CurrentTick() != tick);
	m_pending     = 0;
	m_reset_window.Reduced();
	m_verified    = 0;
	m_batch_count = 0;
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionReductions);
}

bool OcclusionCounter::GateEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OCCLUSION_GATE");
		const bool  on    = value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
		if (Enabled()) {
			std::printf("Occlusion counter: dump-pair gate %s (KYTY_OCCLUSION_GATE)\n",
			            on ? "on" : "off");
		}
		return on;
	}();
	return enabled;
}

bool OcclusionCounter::WouldCount(uint32_t control) const noexcept {
	return Enabled() && ControlCounts(control) && GateOpen();
}

void OcclusionCounter::BreakGate(const char* reason, uint64_t address) {
	if (m_gate_broken) return;
	m_gate_broken = true;
	m_open_pairs.clear();
	std::printf("Occlusion counter: dump-pair gate disabled (%s, address=0x%016" PRIx64
	            "); counting every instance from now on\n",
	            reason, address);
	std::fflush(stdout);
}

void OcclusionCounter::UpdateOpenPairs(uint64_t address) {
	if (!GateEnabled() || m_gate_broken) return;
	const auto begin = address & ~uint64_t {0xf};
	const auto found = std::find(m_open_pairs.begin(), m_open_pairs.end(), begin);
	if ((address & 0xfu) == 0) {
		// Begin dump: every instance until its end dump is counted. A repeated begin at an open
		// pair keeps it open (its later end still differs against the newest begin).
		if (found == m_open_pairs.end()) {
			if (m_open_pairs.size() >= MaxOpenPairs) {
				BreakGate("too many open dump pairs", address);
				return;
			}
			m_open_pairs.push_back(begin);
		}
	} else if ((address & 0xfu) == 8u) {
		if (found == m_open_pairs.end()) {
			// An end without an observed begin: its begin value may predate gated instances.
			BreakGate("end dump without an open begin", address);
			return;
		}
		m_open_pairs.erase(found);
	} else {
		BreakGate("dump address outside the begin/end pair layout", address);
	}
}

void OcclusionCounter::Prepare(uint32_t control) {
	EXIT_IF(m_active || m_pending >= QueryCapacity);
	m_prepared = false;
	if (!Enabled() || !ControlCounts(control)) return;
	// GFX10 ZPASS enable 1 counts all samples. Other counter selectors and slice
	// filtering require additional emulation; never report them as invisibility.
	if ((control & 0x00ffff00u) != 0x100u || (control >> 24u) != 0x11u) {
		EXIT("unsupported occlusion counter mode: DB_COUNT_CONTROL=0x%08x\n", control);
	}
	if (!GateOpen()) {
		// No dump pair is open: nothing counted here can reach a value the guest reads.
		Profiler::CountFrameEvent(Profiler::FrameEvent::OcclusionScopesGated);
		return;
	}
	Initialize();
	const auto reset = m_reset_window.Prepare(m_pending, QueryCapacity, g_reset_batch.On() ? 64u : 1u);
	if (reset.count != 0) {
		m_context.GetCommandScheduler().Current().Sink().resetQueryPool(m_pool, reset.first, reset.count);
	}
	m_prepared = true;
}

void OcclusionCounter::Begin() {
	if (!m_prepared) return;
	m_context.GetCommandScheduler().Current().Sink().beginQuery(m_pool, m_pending, vk::QueryControlFlagBits::ePrecise);
	m_active = true;
	m_prepared = false;
	++m_scopes_since_dump;
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionScopes);
}

void OcclusionCounter::End() {
	if (!m_active) return;
	m_context.GetCommandScheduler().Current().Sink().endQuery(m_pool, m_pending);
	m_active = false;
	++m_pending;
}

void OcclusionCounter::Accumulate() {
	// Keep ended queries in distinct slots across native rendering boundaries and
	// submissions. Only a guest snapshot or bounded pool exhaustion needs a sum.
	if (m_pending == QueryCapacity) {
		if (m_batch_pipeline) {
			FlushBatch();
		} else {
			FlushPending();
		}
	}
}

void OcclusionCounter::FlushPending() {
	KYTY_GPU_OP_SITE("occlusion.flush");
	EXIT_IF(m_active || m_prepared);
	if (!m_pending) return;
	auto native = m_context.GetCommandScheduler().Current().Handle();
	native.copyQueryPoolResults(m_pool, 0, m_pending, m_result->Handle(), 0, 8,
	                            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	Dispatch(0, m_counter->Handle(), 0, m_counter->Size());
	m_pending = 0;
	m_reset_window.Reduced();
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionReductions);
}

void OcclusionCounter::Dispatch(uint32_t mode, vk::Buffer output, uint64_t offset, uint64_t range) {
	KYTY_GPU_OP_SITE("occlusion.reduce");
	auto& command = m_context.GetCommandScheduler().Current();
	auto native = command.Handle();
	const auto alignment = m_context.GetGraphics().StorageMinAlignment();
	const auto aligned = Common::AlignDown(offset, alignment);
	EXIT_IF(offset - aligned > UINT32_MAX || range > UINT32_MAX);
	const vk::DescriptorBufferInfo infos[] {
	    {m_result->Handle(), 0, m_result->Size()}, {m_counter->Handle(), 0, 8}, {output, aligned, range + offset - aligned}};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t i = 0; i < writes.size(); ++i) {
		writes[i].dstBinding = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo = &infos[i];
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
	                       {}, 1, &barrier, 0, nullptr, 0, nullptr);
	command.BindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.PushDescriptors(vk::PipelineBindPoint::eCompute, m_layout, 0,
	                        static_cast<uint32_t>(writes.size()), writes.data());
	const uint32_t push[] {mode, static_cast<uint32_t>(offset - aligned), m_pending};
	native.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
	native.dispatch(1, 1, 1);
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
	                       {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

// Astro Bot reads a visibility proxy's result right after the label that follows its end dump.
// Kyty writes labels at record time, so either the result must be published before the CP
// continues (sync; verified 2026-09-27: the Sky Garden water renders only then), or that label
// must be written only after the publication (defer-label). KYTY_OCCLUSION_SYNC_PROXY=0 restores
// plain asynchronous publication.
OcclusionCounter::ProxyMode OcclusionCounter::GetProxyMode() {
	static const ProxyMode mode = [] {
		const auto* legacy = std::getenv("KYTY_OCCLUSION_SYNC_PROXY");
		if (legacy != nullptr && legacy[0] == '0') {
			return ProxyMode::Off;
		}
		const auto* value = std::getenv("KYTY_OCCLUSION_PROXY_MODE");
		const auto  mode  = value != nullptr && std::strcmp(value, "sync") == 0 ? ProxyMode::Sync
		                                                                        : ProxyMode::DeferLabel;
		if (Enabled()) {
			std::printf("Occlusion counter: visibility-proxy mode %s (KYTY_OCCLUSION_PROXY_MODE)\n",
			            mode == ProxyMode::Sync ? "sync" : "defer-label");
		}
		return mode;
	}();
	return mode;
}

bool OcclusionCounter::SyncProxyDumps() {
	return GetProxyMode() != ProxyMode::Off;
}

bool OcclusionCounter::PriorityPublication() {
	static const bool enabled = [] {
		const auto* label_mode = std::getenv("KYTY_LABEL_MODE");
		return GetProxyMode() == ProxyMode::DeferLabel ||
		       (label_mode != nullptr && std::strcmp(label_mode, "completion") == 0);
	}();
	return enabled;
}

bool OcclusionCounter::SplitEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OCCLUSION_SPLIT");
		const bool  on    = value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
		return on && BatchEnabled();
	}();
	return enabled;
}

bool OcclusionCounter::Dump(uint64_t address) {
	auto& scheduler = m_context.GetCommandScheduler();
	if (!m_pool) {
		scheduler.EndRendering(); // Initialize records outside rendering
	}
	Initialize();
	const bool batch = static_cast<bool>(m_batch_pipeline);
	// KYTY_OCCLUSION_SPLIT: close only the instance's query; the instance stays active.
	const bool              active = scheduler.Active();
	const uint64_t          instance = active ? scheduler.Current().ActiveRenderingSerial() : 0;
	const vk::CommandBuffer command  = active ? scheduler.Current().Identity() : vk::CommandBuffer {};
	const auto              tick     = scheduler.CurrentTick();
	const auto              slot_index = static_cast<uint32_t>(m_issued % m_slot_count);
	// Not with the batch verify (its per-dump reductions record outside rendering), nor when the
	// slot must be waited for (the wait may submit this command buffer).
	bool split = batch && SplitEnabled() && instance != 0 && !m_prepared && !m_verify_counter &&
	             (m_issued < m_slot_count || scheduler.IsFree(m_slot_ticks[slot_index]));
	if (split) {
		End();
		if (m_pending >= QueryCapacity) {
			split = false; // the pool must be reduced outside rendering (Accumulate)
		}
	}
	if (!split) {
		scheduler.EndRendering();
	}
	if (!batch) {
		FlushPending();
	}
	// Reduce into a private slot. A slot is reused only after its previous publication's tick
	// has completed (the wait bounds the dumps in flight; with a batch a wait on the current tick
	// submits it, and the pre-submit hook reduces the batch first).
	const auto slot = static_cast<uint32_t>(m_issued % m_slot_count);
	if (m_issued >= m_slot_count) {
		// The slot's previous publication must have read it, not only its GPU work completed.
		Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitOcclusion);
		if (!scheduler.IsFree(m_slot_ticks[slot])) {
			scheduler.Wait(m_slot_ticks[slot]);
		}
		if (PriorityPublication()) {
			scheduler.WaitPriorityOperations(m_slot_ticks[slot]);
		}
	}
	const uint64_t slot_offset = uint64_t {slot} * PublishSlotSize;
	const auto     tag         = static_cast<uint32_t>(m_issued);
	if (batch) {
		// KYTY_OCCLUSION_BATCH: the queries ended so far in this batch precede the dump; FlushBatch
		// reduces them and writes the slot (and this tag) before this command buffer is submitted.
		auto* entries = reinterpret_cast<uint32_t*>(m_prefix->Mapped().data()) + uint64_t {slot} * 2u;
		entries[0]    = m_pending;
		entries[1]    = tag;
		if (!m_prefix->IsCoherent()) {
			m_prefix->Flush(uint64_t {slot} * TableEntrySize, TableEntrySize);
		}
		if (m_batch_count == 0) {
			m_batch_first_slot = slot;
		}
		++m_batch_count;
		if (m_verify_counter) {
			// The per-dump path's value for this dump, into the verify slot.
			VerifyReduce();
			VerifyDispatch(1, m_verify_publish->Handle(), slot_offset, 248, 0);
		}
	} else {
		scheduler.EndRendering();
		Dispatch(1, m_publish->Handle(), slot_offset, 248);
	}
	m_slot_ticks[slot] = scheduler.CurrentTick();
	++m_issued;
	// The shader writes the first qword of each of the 16 interleaved begin/end pairs and leaves
	// the other member untouched; publish exactly those qwords. The batch shader also writes the
	// dump's tag to the slot's last qword (never published): a slot that was not reduced before its
	// command buffer was submitted, or one reduced for another dump, fails the check below.
	auto publish = [this, address, slot_offset, batch, tag] {
		m_publish->Invalidate(slot_offset, batch ? PublishSlotSize : 248);
		const auto* source = m_publish->Mapped().data() + slot_offset;
		if (batch) {
			uint32_t written[2] {};
			std::memcpy(written, source + 248, sizeof(written));
			if (written[0] != tag || written[1] != ~tag) {
				const auto count = m_batch_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
				if (count <= 8 || (count & (count - 1)) == 0) {
					std::printf("Occlusion counter: batch slot check failed for dump %u (slot tag %u/%u, "
					            "address=0x%016" PRIx64 "), %" PRIu64 " so far\n",
					            tag, written[0], ~written[1], address, count);
					std::fflush(stdout);
				}
			}
			if (m_verify_publish) {
				// KYTY_OCCLUSION_BATCH=verify: the 16 published qwords against the per-dump reduction's.
				m_verify_publish->Invalidate(slot_offset, 248);
				const auto* expected = m_verify_publish->Mapped().data() + slot_offset;
				bool same = true;
				for (uint32_t db = 0; db < 16u && same; db++) {
					same = std::memcmp(source + db * 16u, expected + db * 16u, sizeof(uint64_t)) == 0;
				}
				const auto checks = m_verify_checks.fetch_add(1, std::memory_order_relaxed) + 1;
				if (!same) {
					uint64_t got = 0, want = 0;
					std::memcpy(&got, source, sizeof(got));
					std::memcpy(&want, expected, sizeof(want));
					const auto count = m_verify_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
					if (count <= 8 || (count & (count - 1)) == 0) {
						std::printf("Occlusion counter: batch verify mismatch for dump %u at 0x%016" PRIx64
						            ": batch 0x%016" PRIx64 ", per-dump 0x%016" PRIx64 " (%" PRIu64 " so far)\n",
						            tag, address, got, want, count);
						std::fflush(stdout);
					}
				}
				if (checks % 100000u == 0) {
					std::printf("Occlusion counter: batch verify %" PRIu64 " publications checked, %" PRIu64
					            " mismatches, %" PRIu64 " slot tag failures\n",
					            checks, m_verify_mismatches.load(std::memory_order_relaxed),
					            m_batch_mismatches.load(std::memory_order_relaxed));
					std::fflush(stdout);
				}
			}
		}
		m_context.PrepareHostBackingWrite(address, 248, RenderContext::HostWriter::Occlusion);
		for (uint32_t db = 0; db < 16u; db++) {
			(void)LibKernel::Memory::TryWriteBacking(address + db * 16u, source + db * 16u,
			                                         sizeof(uint64_t));
		}
		// Backing bytes changed outside a publication: logged after the write.
		Coherence::NoteContentWrite(address, 248, Coherence::Source::OcclusionWrite);
		if (HangTrace::Enabled()) {
			uint64_t db0 = 0;
			std::memcpy(&db0, source, sizeof(db0));
			HangTrace::OcclusionEvent event;
			event.event   = "publish";
			event.address = address;
			event.value   = db0 & ~(1ull << 63u);
			HangTrace::RecordOcclusion(event);
		}
		m_published.fetch_add(1, std::memory_order_release);
	};
	if (PriorityPublication()) {
		// TryWriteBacking and the host-visible slot are safe on the completion runner.
		scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		scheduler.DeferOperation(std::move(publish));
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionDumps);
	// Rendering has ended above and the value is published from everything counted so far: a
	// begin opens its pair for the instances that follow, an end closes it after its snapshot.
	UpdateOpenPairs(address);
	if (HangTrace::Enabled()) {
		HangTrace::OcclusionEvent event;
		event.event         = "dump";
		event.address       = address;
		event.value         = m_issued;
		event.scopes        = m_scopes_since_dump;
		event.width         = m_last_scope.width;
		event.height        = m_last_scope.height;
		event.colors        = m_last_scope.colors;
		event.has_depth     = m_last_scope.has_depth;
		event.depth_format  = m_last_scope.depth_format;
		event.depth_address = m_last_scope.depth_address;
		HangTrace::RecordOcclusion(event);
	}
	// An end dump sits 8 bytes after its begin dump (interleaved begin/end pairs). A pair whose
	// latest counted scope rendered only depth is a visibility proxy (e.g. a bounding box).
	const bool sync = SyncProxyDumps() && (address & 0xfu) == 8u && m_scopes_since_dump != 0 &&
	                  m_last_scope.colors == 0 && m_last_scope.has_depth;
	m_scopes_since_dump = 0;
	m_last_scope        = {};
	// A slot wait above may have submitted the command buffer (ending the instance).
	split = split && scheduler.Active() && scheduler.Current().Identity() == command &&
	        scheduler.CurrentTick() == tick && scheduler.Current().ActiveRenderingSerial() == instance;
	if (split && WouldCount(scheduler.Current().OcclusionControl())) {
		// The instance is counted from here on: its next query begins inside it, or, without a
		// reset index, the next draw begins a new instance (Prepare resets outside rendering).
		if (m_pending < QueryCapacity && m_reset_window.IsReset(m_pending)) {
			scheduler.Current().Sink().beginQuery(m_pool, m_pending,
			                                      vk::QueryControlFlagBits::ePrecise);
			m_active = true;
			++m_scopes_since_dump;
			m_last_scope = m_instance_scope;
			Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionScopes);
		} else {
			scheduler.EndRendering();
			split = false;
		}
	}
	m_last_dump_kept = split;
	return sync;
}
}
