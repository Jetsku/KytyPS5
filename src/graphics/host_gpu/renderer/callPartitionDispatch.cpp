#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CallSplitPlan.h"
#include "graphics/shader/recompiler/ShaderCalls.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <unordered_map>
#include <vector>

// KYTY_CALL_PARTITIONS (shader/recompiler/ShaderCalls.h): a dispatch of a partitioned program runs
// once per partition, each with its own program (one target inlined) and the program's report set
// (a small device buffer of reported target keys), which is copied to the host every few
// dispatches; new keys become partitions of the later dispatches.
namespace Libs::Graphics {

namespace {

constexpr uint64_t ReportBytes = uint64_t {ShaderRecompiler::IR::CallPartitionReportSlots} * 4u;

// KYTY_CALL_PARTITION_READBACK=N (default 4): the report set is read back every N dispatches of a
// program (every dispatch while it has no known target).
uint32_t ReadbackInterval() {
	static const uint32_t value = [] {
		const char* text = std::getenv("KYTY_CALL_PARTITION_READBACK");
		return std::max(text != nullptr ? static_cast<uint32_t>(std::strtoul(text, nullptr, 0)) : 4u, 1u);
	}();
	return value;
}

struct Totals {
	std::atomic<uint64_t> dispatches {0};
	std::atomic<uint64_t> parts {0};
	std::atomic<uint64_t> probes {0};
	std::atomic<int64_t>  logged_at {0};
};
Totals g_totals;

} // namespace

struct RenderExecutor::CallPartitionState {
	explicit CallPartitionState(RenderContext& context): context(context) {}

	struct Program {
		std::unique_ptr<Buffer> report;
		uint32_t                dispatches = 0;
	};

	const ShaderRecompiler::CallSplit::Code* Code(const ShaderRecompiler::CallPartitionPart& part) {
		auto& slot = codes[part.digest];
		if (slot == nullptr) {
			slot         = std::make_unique<ShaderRecompiler::CallSplit::Code>();
			slot->words  = *part.code;
			slot->digest = part.digest;
		}
		return slot.get();
	}

	std::shared_ptr<Buffer> TakeReadback() {
		{
			std::scoped_lock lock(readback_mutex);
			if (!free_readbacks.empty()) {
				auto found = std::move(free_readbacks.back());
				free_readbacks.pop_back();
				return found;
			}
		}
		return std::make_shared<Buffer>(context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download,
		                                0, AllFlags, ReportBytes);
	}
	void ReturnReadback(std::shared_ptr<Buffer> readback) {
		std::scoped_lock lock(readback_mutex);
		free_readbacks.push_back(std::move(readback));
	}

	RenderContext&                                                                     context;
	std::unordered_map<uint64_t, Program>                                              programs;
	std::unordered_map<uint64_t, std::unique_ptr<ShaderRecompiler::CallSplit::Code>> codes;
	std::mutex                                                                         readback_mutex;
	std::vector<std::shared_ptr<Buffer>>                                               free_readbacks;
};

void RenderExecutor::DispatchPartitions(CommandBuffer& buffer, const std::shared_ptr<const void>& run_pointer,
                                        const std::function<void(const PartitionStep&)>& dispatch) {
	const auto& run = *static_cast<const ShaderRecompiler::CallPartitionRun*>(run_pointer.get());
	if (m_call_partition == nullptr) {
		m_call_partition = std::make_shared<CallPartitionState>(m_context);
	}
	auto& state   = *m_call_partition;
	auto& program = state.programs[run.program];
	buffer.EndRendering();
	if (program.report == nullptr) {
		program.report = std::make_unique<Buffer>(m_context.GetGraphics(), m_context.GetCommandScheduler(),
		                                          MemoryUsage::DeviceLocal, 0,
		                                          AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		                                          ReportBytes);
		SetVulkanObjectNameF(m_context.GetGraphics().device, program.report->Handle(), "Kyty.CallPartitionReport");
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader,
		                            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		                            vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
		                            BarrierOrigin::IndirectArgs);
		buffer.BeginEmission();
		buffer.Sink().fillBuffer(program.report->Handle(), 0, ReportBytes, 0);
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eClear, vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eComputeShader,
		                            vk::AccessFlagBits2::eShaderStorageRead | vk::AccessFlagBits2::eShaderStorageWrite,
		                            BarrierOrigin::IndirectArgs);
		Log::WriteToConsoleAndLog(fmt::format("KYTY_CALL_PARTITIONS: CS shader 0x{:016x}: program 0x{:016x} is "
		                                      "partitioned ({} partition(s) at its first dispatch).\n",
		                                      run.shader_hash, run.program, run.parts.size()));
	}
	const auto report = program.report->BufferDeviceAddress();
	bool       probe  = false;
	for (const auto& part: run.parts) {
		PartitionStep step;
		step.code   = state.Code(part);
		step.report = report;
		step.target = part.target;
		probe |= part.target == 0;
		dispatch(step);
	}
	g_totals.dispatches.fetch_add(1, std::memory_order_relaxed);
	g_totals.parts.fetch_add(run.parts.size(), std::memory_order_relaxed);
	g_totals.probes.fetch_add(probe ? 1u : 0u, std::memory_order_relaxed);
	// The report set to the host.
	if (probe || program.dispatches++ % ReadbackInterval() == 0) {
		auto readback = state.TakeReadback();
		buffer.EndRendering();
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderStorageWrite,
		                            vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
		                            BarrierOrigin::IndirectArgs);
		buffer.BeginEmission();
		const vk::BufferCopy region {0, 0, ReportBytes};
		buffer.Sink().copyBuffer(program.report->Handle(), readback->Handle(), 1, &region);
		auto*          owner = &state;
		const uint64_t key   = run.program;
		m_context.GetCommandScheduler().DeferOperation([owner, readback, key]() mutable {
			if (!readback->IsCoherent()) readback->Invalidate(0, readback->Size());
			const auto bytes = readback->Mapped();
			std::vector<uint32_t> keys(ShaderRecompiler::IR::CallPartitionReportSlots);
			std::memcpy(keys.data(), bytes.data(), ReportBytes);
			PipelineCache::ReportPartitionKeys(key, keys);
			owner->ReturnReadback(std::move(readback));
		});
	}
	const auto now  = std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
	                     .count();
	auto       last = g_totals.logged_at.load(std::memory_order_relaxed);
	if (now - last >= 10 && g_totals.logged_at.compare_exchange_strong(last, now)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_CALL_PARTITIONS: totals: {} partitioned dispatches, {} partition dispatches, {} with the probe\n",
		    g_totals.dispatches.load(), g_totals.parts.load(), g_totals.probes.load()));
	}
}

} // namespace Libs::Graphics
