#include "graphics/host_gpu/renderer/occlusion.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/alignment.h"
#include "common/profiler.h"
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

// KYTY_OCCLUSION_DIRECT=1 (default off, live): the reductions record natively behind a recorder
// drain (CommandBuffer::Handle), as before they went through the command sink. For A/B runs.
Live::Switch g_occlusion_direct("KYTY_OCCLUSION_DIRECT", Live::ParseDefaultOff);

// The sink the reductions record through. With KYTY_OCCLUSION_DIRECT, Handle() first opens a
// direct window, so the sink's calls record natively there.
CommandSink ReductionSink(CommandBuffer& command) {
	if (g_occlusion_direct.On()) {
		(void)command.Handle();
	}
	return command.Sink();
}

} // namespace
bool OcclusionCounter::Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GPU_OCCLUSION");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

OcclusionCounter::OcclusionCounter(RenderContext& context): m_context(context) {}
OcclusionCounter::~OcclusionCounter() {
	// RenderContext drains its scheduler before member destruction.
	auto device = m_context.GetGraphics().device;
	if (m_pool) device.destroyQueryPool(m_pool);
	if (m_pipeline) device.destroyPipeline(m_pipeline);
	if (m_layout) device.destroyPipelineLayout(m_layout);
	if (m_descriptors) device.destroyDescriptorSetLayout(m_descriptors);
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
	m_counter = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage, 256);
	m_result = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, usage,
	                                  QueryCapacity * sizeof(uint64_t));
	m_publish = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, usage,
	                                   PublishSlots * PublishSlotSize);
	scheduler.Current().Handle().fillBuffer(m_counter->Handle(), 0, 256, 0);
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
	m_context.GetCommandScheduler().Current().Sink().resetQueryPool(m_pool, m_pending, 1);
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
	if (m_pending == QueryCapacity) FlushPending();
}

void OcclusionCounter::FlushPending() {
	KYTY_GPU_OP_SITE("occlusion.flush");
	EXIT_IF(m_active || m_prepared);
	if (!m_pending) return;
	// Through the sink: with KYTY_CP_RECORDER these commands are encoded in order, not recorded
	// natively after a recorder drain (most of the CP's drain waits in Crash Bandicoot 4).
	const auto sink = ReductionSink(m_context.GetCommandScheduler().Current());
	sink.copyQueryPoolResults(m_pool, 0, m_pending, m_result->Handle(), 0, 8,
	                          vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	Dispatch(0, m_counter->Handle(), 0, m_counter->Size());
	m_pending = 0;
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeOcclusionReductions);
}

void OcclusionCounter::Dispatch(uint32_t mode, vk::Buffer output, uint64_t offset, uint64_t range) {
	KYTY_GPU_OP_SITE("occlusion.reduce");
	auto& command = m_context.GetCommandScheduler().Current();
	const auto sink = ReductionSink(command);
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
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eComputeShader,
	                     {}, 1, &barrier, 0, nullptr, 0, nullptr);
	command.BindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	command.PushDescriptors(vk::PipelineBindPoint::eCompute, m_layout, 0,
	                        static_cast<uint32_t>(writes.size()), writes.data());
	const uint32_t push[] {mode, static_cast<uint32_t>(offset - aligned), m_pending};
	sink.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), push);
	sink.dispatch(1, 1, 1);
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
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

bool OcclusionCounter::Dump(uint64_t address) {
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	Initialize();
	FlushPending();
	// Reduce into a private slot. A slot is reused only after its previous publication's tick
	// has completed (1024 dumps in flight never happens in practice; the wait bounds it).
	const auto slot = static_cast<uint32_t>(m_issued % PublishSlots);
	if (m_issued >= PublishSlots) {
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
	scheduler.EndRendering();
	Dispatch(1, m_publish->Handle(), slot_offset, 248);
	m_slot_ticks[slot] = scheduler.CurrentTick();
	++m_issued;
	// The shader writes the first qword of each of the 16 interleaved begin/end pairs and leaves
	// the other member untouched; publish exactly those qwords.
	auto publish = [this, address, slot_offset] {
		m_publish->Invalidate(slot_offset, 248);
		const auto* source = m_publish->Mapped().data() + slot_offset;
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
	return sync;
}
}
