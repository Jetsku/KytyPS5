#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CallSplitPlan.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include "gpu_mesh_shaders/gpu_call_bin_spv.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fmt/format.h>
#include <unordered_map>

// KYTY_SHADER_CALLS_SPLIT (shader/recompiler/CallSplit.h): a dispatch of a kernel with call loops
// becomes
//   R_0 (the dispatch's own workgroups) -> per site k: bin_k -> C_k,t (indirect, one per callee
//   slot) -> R_ak (indirect)
// over one call-state buffer per kernel: the wave header words, the wave blocks, and per site the
// indirect commands and wave lists of the bin pass. The record maps of the sites' tables go
// through an upload ring.
namespace Libs::Graphics {

namespace CallSplit = ShaderRecompiler::CallSplit;

namespace {

struct BinPush {
	uint64_t state;
	uint64_t header;
	uint64_t map;
	uint64_t args;
	uint64_t lists;
	uint32_t stride;
	uint32_t waves;
	uint32_t site;
	uint32_t exec_lo;
	uint32_t exec_hi;
	uint32_t record_dword;
	uint32_t records;
	uint32_t targets;
};
static_assert(sizeof(BinPush) == 72);

constexpr uint64_t Align(uint64_t value) {
	return (value + 255u) & ~uint64_t {255};
}

struct Layout {
	uint64_t              header = 0;
	uint64_t              state  = 0;
	std::vector<uint64_t> args;
	std::vector<uint64_t> lists;
	uint64_t              control = 0; // first args byte (cleared each dispatch)
	uint64_t              size    = 0;
};

Layout MakeLayout(const CallSplit::Run& run, uint32_t waves) {
	Layout layout;
	layout.header = 0;
	layout.state  = Align(uint64_t {waves} * 4u);
	uint64_t at   = Align(layout.state + uint64_t {waves} * run.wave_dwords * 4u);
	layout.control = at;
	for (const auto& site: run.sites) {
		layout.args.push_back(at);
		at += Align((site.callees.size() + 1u) * 12u);
	}
	const auto control_end = at;
	for (const auto& site: run.sites) {
		layout.lists.push_back(at);
		at += Align((site.callees.size() + 1u) * uint64_t {waves} * 4u);
	}
	layout.size = at;
	(void)control_end;
	return layout;
}

struct Totals {
	std::atomic<uint64_t> dispatches {0};
	std::atomic<uint64_t> callee_dispatches {0};
	std::atomic<uint64_t> skipped_callees {0};
	std::atomic<uint64_t> skipped {0};
};
Totals g_totals;

} // namespace

struct RenderExecutor::CallSplitState {
	explicit CallSplitState(RenderContext& context): context(context) {}
	~CallSplitState() {
		auto device = context.GetGraphics().device;
		if (pipeline) device.destroyPipeline(pipeline);
		if (layout) device.destroyPipelineLayout(layout);
	}

	void Initialize() {
		if (pipeline) return;
		auto&                       graphics = context.GetGraphics();
		const vk::PushConstantRange push {vk::ShaderStageFlagBits::eCompute, 0, sizeof(BinPush)};
		vk::PipelineLayoutCreateInfo info {};
		info.pushConstantRangeCount = 1;
		info.pPushConstantRanges    = &push;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&info, nullptr, &layout),
		                     "create call-split bin layout");
		const auto                  module = CompileSPV(GPU_CALL_BIN_SPV, graphics.device);
		vk::ComputePipelineCreateInfo create {};
		create.layout       = layout;
		create.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		create.stage.module = module;
		create.stage.pName  = "main";
		const auto result   = graphics.device.createComputePipelines(nullptr, 1, &create, nullptr, &pipeline);
		graphics.device.destroyShaderModule(module);
		RequireVulkanSuccess(result, "create call-split bin pipeline");
		maps = std::make_unique<StreamBuffer>(graphics, context.GetCommandScheduler(), MemoryUsage::Upload,
		                                      uint64_t {16} << 20u, false,
		                                      vk::BufferUsageFlagBits::eShaderDeviceAddress);
		SetVulkanObjectNameF(graphics.device, maps->Handle(), "Kyty.CallSplitMaps");
	}

	Buffer& StateBuffer(uint64_t kernel, uint64_t size) {
		auto& slot = buffers[kernel];
		if (slot == nullptr || slot->Size() < size) {
			if (slot != nullptr) {
				// Released once the recordings that used it completed.
				context.GetCommandScheduler().DeferOperation(
				    [old = std::shared_ptr<Buffer>(std::move(slot))]() mutable { old.reset(); });
			}
			const auto grown = std::max(size, slot != nullptr ? slot->Size() + slot->Size() / 2u : size);
			slot = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
			                                MemoryUsage::DeviceLocal, 0,
			                                AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, grown);
			SetVulkanObjectNameF(context.GetGraphics().device, slot->Handle(), "Kyty.CallSplitState");
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SHADER_CALLS_SPLIT: call-state buffer of kernel 0x{:016x}: {} MiB\n", kernel,
			    grown >> 20u));
		}
		return *slot;
	}

	RenderContext&                                         context;
	vk::PipelineLayout                                     layout   = nullptr;
	vk::Pipeline                                           pipeline = nullptr;
	std::unique_ptr<StreamBuffer>                          maps;
	std::unordered_map<uint64_t, std::unique_ptr<Buffer>> buffers;
};

void RenderExecutor::DispatchCallSplit(CommandBuffer& buffer, const ShaderComputeInputInfo& pristine,
                                       const std::shared_ptr<const void>& run_pointer,
                                       const HW::ComputeShaderInfo& cs_regs,
                                       const HW::ShaderRegisters& sh_regs, uint32_t groups_x,
                                       uint32_t groups_y, uint32_t groups_z) {
	const auto& run   = *static_cast<const CallSplit::Run*>(run_pointer.get());
	const auto  waves = groups_x * groups_y * groups_z;
	if (m_call_split == nullptr) {
		m_call_split = std::make_shared<CallSplitState>(m_context);
	}
	auto& split = *m_call_split;
	split.Initialize();
	const auto layout   = MakeLayout(run, waves);
	auto&      state    = split.StateBuffer(run.kernel, layout.size);
	const auto address  = state.BufferDeviceAddress();
	auto&      pipelines = m_context.GetPipelineCache();

	// One program of the chain: its preparation through the pipeline cache as a dispatch of the
	// kernel, then the dispatch (direct: the kernel's workgroups; indirect: a bin pass's command).
	const auto dispatch = [&](const CallSplit::Code& code, uint64_t list, const vk::Buffer* indirect,
	                          uint64_t indirect_offset) {
		auto info = pristine;
		PipelineCache::SelectCallSplitCode(&code);
		const auto program_handle = pipelines.GetComputeProgram(cs_regs, sh_regs, info, m_compute_prep);
		PipelineCache::SelectCallSplitCode(nullptr);
		if (!program_handle) {
			return false;
		}
		auto* pipeline = pipelines.TryGetComputePipeline(info, program_handle);
		if (pipeline == nullptr) {
			ResetBindings();
			return false;
		}
		const auto& program  = *info.stage.program;
		auto&       bindings = m_compute_bindings;
		PrepareBindings(info.stage, bindings);
		if (const auto dword = program.bindings.call_state_dword;
		    dword != ShaderRecompiler::IR::PushData::NoStart) {
			const uint64_t header = address + layout.header;
			const uint64_t base   = address + layout.state;
			const uint32_t params[ShaderRecompiler::IR::BindingLayout::CallStateDwords] = {
			    static_cast<uint32_t>(base),   static_cast<uint32_t>(base >> 32u),
			    static_cast<uint32_t>(list),   static_cast<uint32_t>(list >> 32u),
			    static_cast<uint32_t>(header), static_cast<uint32_t>(header >> 32u),
			    run.wave_dwords};
			std::copy(std::begin(params), std::end(params), bindings.shader_data.begin() + dword);
		}
		bindings.dispatch_groups     = {groups_x, groups_y, groups_z};
		bindings.has_dispatch_groups = indirect == nullptr;
		FindBuffers(bindings);
		if (program.info.uses_dma) {
			m_context.PrepareBda();
		}
		if (program.info.bda_writes) {
			m_context.GetBufferCache().PrepareBdaWrites();
		}
		RebindImages(bindings);
		const std::array<uint32_t, 3> groups {groups_x, groups_y, groups_z};
		BindComputeLds(info, bindings, indirect == nullptr ? groups.data() : nullptr);
		RebindBuffers(bindings);
		PreparedBindings* stage = &bindings;
		buffer.BeginEmission();
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, *pipeline, std::span {&stage, 1u});
		ShaderWriteHazardBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		buffer.BindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
		if (indirect == nullptr) {
			buffer.Sink().dispatch(groups_x, groups_y, groups_z);
		} else {
			buffer.Sink().dispatchIndirect(*indirect, indirect_offset);
		}
		ShaderAccessBarrier(buffer, vk::PipelineStageFlagBits::eComputeShader);
		buffer.NoteGuestDispatch(true);
		if (program.info.bda_writes) {
			m_context.GetBufferCache().SettleBdaWrites(program.shader_hash);
		}
		ResetBindings();
		return true;
	};
	const auto to_compute = [&](vk::PipelineStageFlags2 source, vk::AccessFlags2 access) {
		buffer.RequestMemoryBarrier(source, access,
		                            vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eShaderStorageRead |
		                                vk::AccessFlagBits2::eShaderStorageWrite |
		                                vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
	};

	// The headers and the bin commands start at zero; the wave blocks need no clearing (a wave
	// writes its block before any program reads it).
	{
		// Every earlier use of the buffer (the previous dispatch's chain) before the clears.
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eShaderStorageRead |
		                                vk::AccessFlagBits2::eShaderStorageWrite |
		                                vk::AccessFlagBits2::eIndirectCommandRead,
		                            vk::PipelineStageFlagBits2::eClear |
		                                vk::PipelineStageFlagBits2::eCopy,
		                            vk::AccessFlagBits2::eTransferWrite, BarrierOrigin::IndirectArgs);
		auto sink = buffer.Sink();
		sink.fillBuffer(state.Handle(), layout.header, Align(uint64_t {waves} * 4u), 0);
		sink.fillBuffer(state.Handle(), layout.control,
		                (layout.lists.empty() ? layout.size : layout.lists.front()) - layout.control, 0);
	}
	// The record maps.
	std::vector<uint64_t> maps;
	for (const auto& site: run.sites) {
		const auto bytes            = std::max<uint64_t>(site.record_slots.size() * 4u, 4u);
		const auto [mapped, offset] = split.maps->Map(bytes, 256);
		EXIT_IF(mapped == nullptr);
		if (!site.record_slots.empty()) {
			std::memcpy(mapped, site.record_slots.data(), site.record_slots.size() * 4u);
		}
		split.maps->Commit();
		maps.push_back(split.maps->BufferDeviceAddress() + offset);
	}
	to_compute(vk::PipelineStageFlagBits2::eClear | vk::PipelineStageFlagBits2::eCopy |
	               vk::PipelineStageFlagBits2::eHost,
	           vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eHostWrite);

	const auto ordinal = g_totals.dispatches.fetch_add(1, std::memory_order_relaxed);
	// KYTY_SHADER_CALLS_SPLIT_LOG=N: the steps of the first N split dispatches.
	static const uint64_t log_limit = [] {
		const auto* value = std::getenv("KYTY_SHADER_CALLS_SPLIT_LOG");
		return value != nullptr ? std::strtoull(value, nullptr, 0) : 0ull;
	}();
	std::string steps;
	const auto note = [&](const char* what, bool ran) {
		if (ordinal < log_limit) steps += fmt::format(" {}{}", what, ran ? "" : "(skipped)");
	};
	const bool started = dispatch(*run.start, 0, nullptr, 0);
	note("R_0", started);
	if (!started) {
		g_totals.skipped.fetch_add(1, std::memory_order_relaxed);
	}
	for (uint32_t k = 1; started && k <= run.sites.size(); k++) {
		const auto& site    = run.sites[k - 1u];
		const auto  targets = static_cast<uint32_t>(site.callees.size());
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		BinPush push {};
		push.state        = address + layout.state;
		push.header       = address + layout.header;
		push.map          = maps[k - 1u];
		push.args         = address + layout.args[k - 1u];
		push.lists        = address + layout.lists[k - 1u];
		push.stride       = run.wave_dwords;
		push.waves        = waves;
		push.site         = k;
		push.exec_lo      = site.exec_lo_slot;
		push.exec_hi      = site.exec_hi_slot;
		push.record_dword = site.record_dword;
		push.records      = static_cast<uint32_t>(site.record_slots.size());
		push.targets      = targets;
		buffer.BeginEmission();
		buffer.BindPipeline(vk::PipelineBindPoint::eCompute, split.pipeline);
		{
			auto sink = buffer.Sink();
			sink.pushConstants(split.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
			sink.dispatch(waves, 1, 1);
		}
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		const auto handle = state.Handle();
		for (uint32_t t = 0; t < targets; t++) {
			if (site.callees[t] == nullptr) {
				g_totals.skipped_callees.fetch_add(1, std::memory_order_relaxed);
				continue;
			}
			const bool ran = dispatch(*site.callees[t], address + layout.lists[k - 1u] + uint64_t {t} * waves * 4u,
			                          &handle, layout.args[k - 1u] + uint64_t {t} * 12u);
			note("C", ran);
			if (ran) {
				g_totals.callee_dispatches.fetch_add(1, std::memory_order_relaxed);
			} else {
				g_totals.skipped_callees.fetch_add(1, std::memory_order_relaxed);
			}
		}
		to_compute(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite);
		const bool resumed =
		    site.resume != nullptr &&
		    dispatch(*site.resume, address + layout.lists[k - 1u] + uint64_t {targets} * waves * 4u, &handle,
		             layout.args[k - 1u] + uint64_t {targets} * 12u);
		note(k == 1 ? "bin1 R_a1" : "bin R_a", resumed);
		if (!resumed) {
			g_totals.skipped.fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (ordinal < log_limit) {
		Log::WriteToConsoleAndLog(fmt::format("KYTY_SHADER_CALLS_SPLIT: dispatch {} of kernel 0x{:016x} ({} waves):{}\n",
		                                      ordinal, run.kernel, waves, steps));
	}
}

} // namespace Libs::Graphics
