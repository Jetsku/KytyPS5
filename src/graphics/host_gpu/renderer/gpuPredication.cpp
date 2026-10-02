#include "graphics/host_gpu/renderer/gpuPredication.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "gpu_mesh_shaders/gpu_predicate_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::GpuPredication {

Mode GetMode() {
	static const Mode mode = [] {
		const auto* value = std::getenv("KYTY_PREDICATION_MODE");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "drain") == 0) {
			return Mode::Drain;
		}
		if (std::strcmp(value, "precise") == 0) {
			return Mode::Precise;
		}
		if (std::strcmp(value, "gpu") == 0) {
			return Mode::Gpu;
		}
		EXIT("unknown KYTY_PREDICATION_MODE: %s (drain, precise, gpu)\n", value);
		return Mode::Drain;
	}();
	return mode;
}

bool ExtensionRequested() {
	return GetMode() == Mode::Gpu;
}

Totals& GetTotals() {
	static Totals totals;
	return totals;
}

namespace {

// Push constants of gpu_predicate.comp, in declaration order.
struct PushData {
	uint32_t source_word = 0;
	uint32_t slot        = 0;
	uint32_t condition   = 0;
};
static_assert(sizeof(PushData) == 12);

} // namespace

Predicates::Predicates(RenderContext& context): m_context(context) {}

Predicates::~Predicates() {
	// RenderContext shuts its scheduler down before members die.
	auto device = m_context.GetGraphics().device;
	if (m_pipeline) device.destroyPipeline(m_pipeline);
	if (m_layout) device.destroyPipelineLayout(m_layout);
	if (m_descriptors) device.destroyDescriptorSetLayout(m_descriptors);
}

bool Predicates::Supported() const {
	return m_context.GetGraphics().conditional_rendering_enabled;
}

void Predicates::Initialize() {
	if (m_pipeline) {
		return;
	}
	auto& graphics = m_context.GetGraphics();
	const std::array<vk::DescriptorSetLayoutBinding, 3> bindings {{
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	}};
	vk::DescriptorSetLayoutCreateInfo descriptor {};
	descriptor.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	descriptor.bindingCount = static_cast<uint32_t>(bindings.size());
	descriptor.pBindings    = bindings.data();
	RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&descriptor, nullptr, &m_descriptors),
	                     "create predicate descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushData)};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &m_layout),
	                     "create predicate layout");
	const auto module = CompileSPV(GPU_PREDICATE_SPV, graphics.device);
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.layout       = m_layout;
	pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName  = "main";
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr,
	                                                           &m_pipeline);
	graphics.device.destroyShaderModule(module);
	RequireVulkanSuccess(result, "create predicate pipeline");
	auto&          scheduler = m_context.GetCommandScheduler();
	constexpr auto size      = uint64_t {Slots} * sizeof(uint32_t);
	m_predicates             = std::make_unique<Buffer>(
        graphics, scheduler, MemoryUsage::DeviceLocal, 0,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eConditionalRenderingEXT,
        size);
	m_readback = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0,
	                                      vk::BufferUsageFlagBits::eStorageBuffer, size);
	SetVulkanObjectNameF(graphics.device, m_predicates->Handle(), "Kyty.GuestPredicates");
	SetVulkanObjectNameF(graphics.device, m_readback->Handle(), "Kyty.GuestPredicatesReadback");
}

Predicates::Entry& Predicates::Lookup(uint32_t id) {
	auto& entry = m_entries[id % Slots];
	EXIT_IF(id == 0 || entry.id != id);
	return entry;
}

uint32_t Predicates::Record(CommandBuffer& buffer, const Buffer& source, uint64_t source_offset,
                            uint32_t condition) {
	KYTY_GPU_OP_SITE("predication.snapshot");
	Initialize();
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	if (++m_next_id == 0) {
		m_next_id = 1;
	}
	const auto id    = m_next_id;
	const auto slot  = id % Slots;
	auto&      entry = m_entries[slot];
	// The slot's previous predicate may still be read by a submitted recording. Within the
	// current one, the barrier below orders the new write after those reads.
	if (entry.id != 0 && entry.last_tick < scheduler.CurrentTick() &&
	    !scheduler.IsFree(entry.last_tick)) {
		GetTotals().slot_waits.fetch_add(1, std::memory_order_relaxed);
		scheduler.Wait(entry.last_tick);
	}
	// A dispatch cannot be recorded inside dynamic rendering.
	scheduler.EndRendering();
	// Every earlier write of the predicate (a shader, a transfer, an upload) and every earlier
	// read of the slot before the snapshot.
	buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllCommands,
	                            vk::AccessFlagBits2::eMemoryWrite,
	                            vk::PipelineStageFlagBits2::eComputeShader,
	                            vk::AccessFlagBits2::eShaderStorageRead |
	                                vk::AccessFlagBits2::eShaderStorageWrite,
	                            BarrierOrigin::IndirectArgs);
	const auto aligned = Common::AlignDown(source_offset, graphics.StorageMinAlignment());
	EXIT_IF(((source_offset - aligned) & 3u) != 0);
	constexpr auto size = uint64_t {Slots} * sizeof(uint32_t);
	const std::array<vk::DescriptorBufferInfo, 3> infos {{
	    {source.Handle(), aligned, source_offset - aligned + sizeof(uint64_t)},
	    {m_predicates->Handle(), 0, size},
	    {m_readback->Handle(), 0, size},
	}};
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	buffer.BindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
	(void)buffer.PushDescriptors(vk::PipelineBindPoint::eCompute, m_layout, 0,
	                             static_cast<uint32_t>(writes.size()), writes.data());
	PushData push;
	push.source_word = static_cast<uint32_t>((source_offset - aligned) / 4u);
	push.slot        = slot;
	push.condition   = condition;
	// Sink() records the pending batch (the barrier above) first without draining the recorder.
	auto sink = buffer.Sink();
	sink.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	sink.dispatch(1, 1, 1);
	// The snapshot before the predicated draws' conditional-rendering reads and Resolve's host
	// read. Recorded by the next BeginRendering or at the end of the recording.
	buffer.RequestMemoryBarrier(
	    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	    vk::PipelineStageFlagBits2::eConditionalRenderingEXT | vk::PipelineStageFlagBits2::eHost,
	    vk::AccessFlagBits2::eConditionalRenderingReadEXT | vk::AccessFlagBits2::eHostRead,
	    BarrierOrigin::IndirectArgs);
	entry            = {};
	entry.id         = id;
	entry.write_tick = scheduler.CurrentTick();
	entry.last_tick  = entry.write_tick;
	GetTotals().recorded.fetch_add(1, std::memory_order_relaxed);
	return id;
}

vk::ConditionalRenderingBeginInfoEXT Predicates::Use(uint32_t id) {
	auto& entry     = Lookup(id);
	entry.last_tick = m_context.GetCommandScheduler().CurrentTick();
	GetTotals().gated_draws.fetch_add(1, std::memory_order_relaxed);
	vk::ConditionalRenderingBeginInfoEXT info {};
	info.buffer = m_predicates->Handle();
	info.offset = uint64_t {id % Slots} * sizeof(uint32_t);
	return info;
}

bool Predicates::Resolve(uint32_t id) {
	auto& entry = Lookup(id);
	if (entry.resolved < 0) {
		GetTotals().resolves.fetch_add(1, std::memory_order_relaxed);
		// Submits the current recording when it wrote the slot.
		m_context.GetCommandScheduler().Wait(entry.write_tick);
		const auto offset = uint64_t {id % Slots} * sizeof(uint32_t);
		m_readback->Invalidate(offset, sizeof(uint32_t));
		uint32_t draw = 0;
		std::memcpy(&draw, m_readback->Mapped().data() + offset, sizeof(draw));
		entry.resolved = draw != 0 ? 1 : 0;
	}
	return entry.resolved != 0;
}

} // namespace Libs::Graphics::GpuPredication
