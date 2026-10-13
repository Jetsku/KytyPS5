#include "graphics/host_gpu/renderer/meshIndirect.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/profiler.h"
#include "gpu_mesh_shaders/gpu_mesh_indirect_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::MeshIndirect {

Mode GetMode() {
	static const Mode mode = [] {
		const auto* value = std::getenv("KYTY_NATIVE_INDIRECT_MESH");
		// Default on: the CPU path drains the GPU for every argument record it reads.
		if (value == nullptr || value[0] == '\0') {
			return Mode::On;
		}
		if (std::strcmp(value, "empty") == 0) {
			return Mode::Empty;
		}
		if (std::strcmp(value, "0") == 0) {
			return Mode::Off;
		}
		if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
			return Mode::On;
		}
		if (std::strcmp(value, "verify") == 0) {
			return Mode::Verify;
		}
		if (std::strcmp(value, "exit") == 0) {
			return Mode::VerifyExit;
		}
		std::fprintf(stderr,
		             "KYTY_NATIVE_INDIRECT_MESH=%s is not 0, empty, 1, on, verify or exit; using "
		             "empty\n",
		             value);
		return Mode::Empty;
	}();
	return mode;
}

bool ConversionEnabled() {
	const auto mode = GetMode();
	return mode == Mode::On || mode == Mode::Verify || mode == Mode::VerifyExit;
}

Totals& GetTotals() {
	static Totals totals;
	return totals;
}

Conversion Convert(const Inputs& inputs, std::span<const uint32_t, 5> args) {
	Conversion result;
	uint32_t   count     = args[0];
	const auto instances = args[1];
	uint32_t   vertex_offset  = 0;
	uint32_t   first_instance = 0;
	uint64_t   address        = 0;
	if (inputs.indexed) {
		// CommandProcessor::DrawIndirect: the count is clamped to INDEX_BUFFER_SIZE (the start
		// index is not); the address is INDEX_BASE + start * index size. DrawIndex takes the
		// vertex offset from the record's base vertex and the first instance from the record.
		if (inputs.index_buffer_size != 0) {
			count = std::min(count, inputs.index_buffer_size);
			if (uint64_t {args[2]} + count > inputs.index_buffer_size) {
				result.status |= StatusIndexRange;
			}
		}
		address        = inputs.index_base + uint64_t {args[2]} * inputs.index_size;
		vertex_offset  = args[3];
		first_instance = args[4];
	} else {
		// DrawAuto: the first vertex and first instance of the record.
		vertex_offset  = args[2];
		first_instance = args[3];
	}
	// DrawIndex/DrawAuto return without indices or instances; ExecutePreparedDraw without a
	// primitive (ShaderMeshInputInfo::InputPrimitiveCount), and splits the instances into
	// dispatches of MeshInstancesPerDispatch (it stops the emulator when that is 0).
	uint32_t groups       = 0;
	uint32_t per_dispatch = 0;
	if (count != 0 && instances != 0) {
		const auto size = inputs.primitive_size;
		const auto primitives =
		    count < size ? 0u : (count - size) / inputs.primitive_step + 1u;
		if (primitives != 0) {
			groups       = (primitives - 1u) / inputs.primitives_per_group + 1u;
			per_dispatch = groups > inputs.max_groups_x
			                   ? 0u
			                   : std::min(inputs.max_groups_y, inputs.max_groups_total / groups);
			if (per_dispatch == 0) {
				result.status |= StatusLimits;
				groups = 0;
			} else if ((instances - 1u) / per_dispatch >= Records) {
				result.status |= StatusOverflow;
			}
		}
	}
	const auto index_size = inputs.indexed ? inputs.index_size : 0u;
	for (uint32_t k = 0; k < Records; k++) {
		const auto first = k * per_dispatch;
		const bool live  = groups != 0 && first < instances;
		result.commands[k] = {live ? groups : 0u,
		                      live ? std::min(per_dispatch, instances - first) : 0u, live ? 1u : 0u};
		result.params[k]   = {count,
		                      vertex_offset,
		                      first_instance + first,
		                      index_size,
		                      static_cast<uint32_t>(address),
		                      static_cast<uint32_t>(address >> 32u),
		                      0u,
		                      0u};
	}
	return result;
}

bool AlwaysEmpty(const Inputs& inputs) {
	// InputPrimitiveCount(min(count, INDEX_BUFFER_SIZE)) is 0 for every count.
	return inputs.indexed && inputs.index_buffer_size != 0 &&
	       inputs.index_buffer_size < inputs.primitive_size;
}

namespace {

// Push constants of gpu_mesh_indirect.comp, in declaration order.
struct PushData {
	uint32_t args_word            = 0;
	uint32_t out_word             = 0;
	uint32_t indexed              = 0;
	uint32_t index_base_lo        = 0;
	uint32_t index_base_hi        = 0;
	uint32_t index_size           = 0;
	uint32_t index_buffer_size    = 0;
	uint32_t primitive_size       = 0;
	uint32_t primitive_step       = 0;
	uint32_t primitives_per_group = 0;
	uint32_t max_groups_x         = 0;
	uint32_t max_groups_y         = 0;
	uint32_t max_groups_total     = 0;
	uint32_t generation           = 0;
};
static_assert(sizeof(PushData) == 56);

// 1,024 slots: a slot is reused only after the recording that wrote it and its completion check
// finished (a Download ring waits for both on wrap).
constexpr uint64_t RingSize = uint64_t {1024} * SlotBytes;

} // namespace

Converter::Converter(RenderContext& context): m_context(context) {}

Converter::~Converter() {
	// RenderContext shuts its scheduler down (draining the completion checks) before members die.
	auto device = m_context.GetGraphics().device;
	if (m_pipeline) device.destroyPipeline(m_pipeline);
	if (m_layout) device.destroyPipelineLayout(m_layout);
	if (m_descriptors) device.destroyDescriptorSetLayout(m_descriptors);
}

void Converter::Initialize() {
	if (m_pipeline) {
		return;
	}
	// Push descriptors (VK_KHR_push_descriptor) are required by the renderer itself.
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
	                     "create mesh-indirect descriptors");
	const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(PushData)};
	vk::PipelineLayoutCreateInfo layout {};
	layout.setLayoutCount         = 1;
	layout.pSetLayouts            = &m_descriptors;
	layout.pushConstantRangeCount = 1;
	layout.pPushConstantRanges    = &push;
	RequireVulkanSuccess(graphics.device.createPipelineLayout(&layout, nullptr, &m_layout),
	                     "create mesh-indirect layout");
	const auto module = CompileSPV(GPU_MESH_INDIRECT_SPV, graphics.device);
	vk::ComputePipelineCreateInfo pipeline {};
	pipeline.layout       = m_layout;
	pipeline.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline.stage.module = module;
	pipeline.stage.pName  = "main";
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipeline, nullptr,
	                                                           &m_pipeline);
	graphics.device.destroyShaderModule(module);
	RequireVulkanSuccess(result, "create mesh-indirect pipeline");
	m_ring = std::make_unique<StreamBuffer>(graphics, m_context.GetCommandScheduler(),
	                                        MemoryUsage::Download, RingSize, false,
	                                        vk::BufferUsageFlagBits::eShaderDeviceAddress);
	SetVulkanObjectNameF(graphics.device, m_ring->Handle(), "Kyty.MeshIndirectRing");
}

Converter::Slot Converter::Reserve() {
	Initialize();
	const auto alignment =
	    std::max<uint64_t>(SlotBytes, m_context.GetGraphics().StorageMinAlignment());
	const auto [mapped, offset] = m_ring->Map(SlotBytes, alignment);
	EXIT_IF(mapped == nullptr);
	m_ring->Commit();
	Slot slot;
	slot.buffer     = m_ring->Handle();
	slot.offset     = offset;
	slot.address    = m_ring->BufferDeviceAddress() + offset;
	slot.generation = ++m_generation;
	return slot;
}

void Converter::Record(CommandBuffer& buffer, const Slot& slot, const Inputs& inputs,
                       vk::Buffer args, uint64_t args_offset) {
	KYTY_GPU_OP_SITE("draw.mesh_indirect");
	Initialize();
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	// A dispatch cannot be recorded inside dynamic rendering.
	scheduler.EndRendering();
	// Every earlier write (the argument record's producer: a shader, a transfer, an upload)
	// before the conversion reads the record. Recorded by Handle() below.
	buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllCommands,
	                            vk::AccessFlagBits2::eMemoryWrite,
	                            vk::PipelineStageFlagBits2::eComputeShader,
	                            vk::AccessFlagBits2::eShaderStorageRead, BarrierOrigin::IndirectArgs);
	const auto record_size = inputs.indexed ? 20u : 16u;
	const auto aligned     = Common::AlignDown(args_offset, graphics.StorageMinAlignment());
	EXIT_IF(((args_offset - aligned) & 3u) != 0);
	const std::array<vk::DescriptorBufferInfo, 2> infos {{
	    {args, aligned, args_offset - aligned + record_size},
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
	PushData push;
	push.args_word            = static_cast<uint32_t>((args_offset - aligned) / 4u);
	push.out_word             = 0;
	push.indexed              = inputs.indexed ? 1u : 0u;
	push.index_base_lo        = static_cast<uint32_t>(inputs.index_base);
	push.index_base_hi        = static_cast<uint32_t>(inputs.index_base >> 32u);
	push.index_size           = inputs.index_size;
	push.index_buffer_size    = inputs.index_buffer_size;
	push.primitive_size       = inputs.primitive_size;
	push.primitive_step       = inputs.primitive_step;
	push.primitives_per_group = inputs.primitives_per_group;
	push.max_groups_x         = inputs.max_groups_x;
	push.max_groups_y         = inputs.max_groups_y;
	push.max_groups_total     = inputs.max_groups_total;
	push.generation           = slot.generation;
	// Sink() records the pending batch (the barrier above) first, as Handle() did, and resets the
	// push-constant shadow, without draining the CP recorder.
	auto sink = buffer.Sink();
	sink.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
	sink.dispatch(1, 1, 1);
	// The conversion's writes before the draw's indirect command reads and its mesh shaders'
	// parameter loads, and before the completion check's host read. Recorded by the draw's
	// BeginRendering (the instance was ended above, so the barrier cannot be sunk past it).
	buffer.RequestMemoryBarrier(
	    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
	    vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eMeshShaderEXT |
	        vk::PipelineStageFlagBits2::eHost,
	    vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderStorageRead |
	        vk::AccessFlagBits2::eHostRead,
	    BarrierOrigin::IndirectArgs);
	scheduler.DeferPriorityOperation([this, slot, inputs] { Check(slot, inputs); });
	GetTotals().draws.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectDraws);
}

std::array<uint32_t, SlotDwords> Converter::ReadSlot(const Slot& slot) const {
	std::array<uint32_t, SlotDwords> words {};
	m_ring->Invalidate(slot.offset, SlotBytes);
	std::memcpy(words.data(), m_ring->Mapped().data() + slot.offset, SlotBytes);
	return words;
}

void Converter::Check(const Slot& slot, const Inputs& inputs) {
	// Completion runner: the recording that wrote the slot has finished.
	auto&      totals = GetTotals();
	const auto words  = ReadSlot(slot);
	if (words[GenerationWord] != slot.generation) {
		// Not reachable with the Download ring (reuse waits for this check); counted, not trusted.
		totals.stale.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto status = words[StatusWord];
	if (status != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectStatus);
		if ((status & StatusOverflow) != 0) totals.status_overflow.fetch_add(1);
		if ((status & StatusLimits) != 0) totals.status_limits.fetch_add(1);
		if ((status & StatusIndexRange) != 0) totals.status_index_range.fetch_add(1);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::fprintf(stderr,
			             "MeshIndirect: status 0x%x (1: instances beyond %u dispatches dropped, 2: "
			             "draw beyond the host mesh limits skipped, 4: indices read past "
			             "INDEX_BUFFER_SIZE) args %08x %08x %08x %08x %08x\n",
			             status, Records, words[ArgsCopyWord], words[ArgsCopyWord + 1],
			             words[ArgsCopyWord + 2], words[ArgsCopyWord + 3], words[ArgsCopyWord + 4]);
		}
	}
	const auto mode = GetMode();
	if (mode != Mode::Verify && mode != Mode::VerifyExit) {
		return;
	}
	totals.checks.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectVerifyChecks);
	std::array<uint32_t, 5> args {};
	std::copy_n(words.begin() + ArgsCopyWord, args.size(), args.begin());
	const auto expected = Convert(inputs, args);
	Conversion gpu;
	for (uint32_t k = 0; k < Records; k++) {
		std::copy_n(words.begin() + k * CommandDwords, CommandDwords, gpu.commands[k].begin());
		std::copy_n(words.begin() + ParamsWord + k * ParamDwords, ParamDwords, gpu.params[k].begin());
	}
	gpu.status = status;
	if (gpu == expected) {
		return;
	}
	totals.mismatches.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "MeshIndirectVerify: the GPU conversion differs from the CPU path: args %08x "
		             "%08x %08x %08x %08x indexed=%u; command[0] gpu %u,%u,%u cpu %u,%u,%u; "
		             "params[0] gpu %08x %08x %08x %08x %08x %08x cpu %08x %08x %08x %08x %08x "
		             "%08x; status gpu 0x%x cpu 0x%x\n",
		             args[0], args[1], args[2], args[3], args[4], inputs.indexed ? 1u : 0u,
		             gpu.commands[0][0], gpu.commands[0][1], gpu.commands[0][2],
		             expected.commands[0][0], expected.commands[0][1], expected.commands[0][2],
		             gpu.params[0][0], gpu.params[0][1], gpu.params[0][2], gpu.params[0][3],
		             gpu.params[0][4], gpu.params[0][5], expected.params[0][0],
		             expected.params[0][1], expected.params[0][2], expected.params[0][3],
		             expected.params[0][4], expected.params[0][5], gpu.status, expected.status);
	}
	if (mode == Mode::VerifyExit) {
		EXIT("MeshIndirectVerify: the GPU conversion differs from the CPU path\n");
	}
}

} // namespace Libs::Graphics::MeshIndirect
