#include <fmt/format.h>
#include <mutex>
#include <map>
#include <chrono>
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

#include <cinttypes>
#include <optional>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics, bool track_dispatch)
    : m_graphics(graphics) {
	if (graphics.submission_queue.Enabled() || track_dispatch) {
		m_submission_progress = std::make_shared<SubmissionProgress>();
	}
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	m_watchdog_timeline = HangWatchdog::RegisterTimeline(
	    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)));
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	HangWatchdog::Scope query("master-counter-query",
	                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                          HangWatchdog::Enabled() ? CurrentTick() : 0);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-counter-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          CurrentTick(),
		                          static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) DumpDeviceLossDiagnostics(m_graphics, CurrentTick());
		EXIT("MasterSemaphore: counter query failed: %s (submission tick %" PRIu64 ", gpu tick %" PRIu64 ")\n",
		     vk::to_string(result).c_str(), CurrentTick(), m_gpu_tick.load(std::memory_order_acquire));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
	if (HangWatchdog::Enabled()) {
		HangWatchdog::UpdateTimeline(
		    m_watchdog_timeline, CurrentTick(), counter,
		    m_submission_progress ? m_submission_progress->dispatched_tick.load() : CurrentTick() - 1);
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	// KYTY_CP_GAP_STATS (GPU thread only): readback drains, occlusion publication, the rest.
	const auto      reason = Profiler::CurrentGpuWaitReason();
	const CpGaps::Scope gap(reason == Profiler::FrameWait::GpuWaitDrain ? CpGaps::Cat::GpuWaitDrain
	                        : reason == Profiler::FrameWait::GpuWaitOcclusion
	                            ? CpGaps::Cat::GpuWaitOcclusion
	                            : CpGaps::Cat::GpuWaitOther);
	// Attribute CP-thread blocking to its caller (Profiler::ScopedGpuWaitReason). Other threads
	// (the completion runner, guest threads) wait here by design and are not counted.
	std::optional<Profiler::ScopedFrameWait> frame_wait;
	if (GuestGpu::IsGpuThread() && Profiler::AggregateEnabled()) {
		frame_wait.emplace(Profiler::CurrentGpuWaitReason());
	}
	if (m_submission_progress) {
		auto submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
		HangWatchdog::Scope dispatch(
		    "master-dispatch", reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		    tick, submitted, 0, HangWatchdog::Enabled() ? CurrentTick() : 0);
		while (submitted < tick) {
			m_submission_progress->dispatched_tick.wait(submitted, std::memory_order_acquire);
			submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
			dispatch.Observed(submitted);
		}
	}
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	HangWatchdog::Scope wait("master-gpu",
	                         reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                         tick, HangWatchdog::Enabled() ? KnownGpuTick() : 0, 0,
	                         HangWatchdog::Enabled() ? CurrentTick() : 0);
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto wait_start = std::chrono::steady_clock::now();
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	{
		// KYTY_GPU_WAIT_STATS=1: every 5 s, the waits on the GPU timeline by reason and thread kind.
		static const bool wait_stats = [] {
			const char* v = std::getenv("KYTY_GPU_WAIT_STATS");
			return v != nullptr && v[0] == '1';
		}();
		if (wait_stats) {
			static std::mutex                                      stats_mutex;
			static std::map<std::pair<int, uint32_t>, std::pair<uint64_t, uint64_t>> stats;
			static auto                                            last = std::chrono::steady_clock::now();
			const auto now = std::chrono::steady_clock::now();
			const auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now - wait_start).count();
			const int  gpu_thread = GuestGpu::IsGpuThread() ? 1 : 0;
			std::scoped_lock lock(stats_mutex);
			auto& entry = stats[{gpu_thread, static_cast<uint32_t>(Profiler::CurrentGpuWaitReason())}];
			entry.first++;
			entry.second += static_cast<uint64_t>(ns);
			if (now - last > std::chrono::seconds(5)) {
				const double secs = std::chrono::duration<double>(now - last).count();
				std::string text = fmt::format("GpuWaitStats {:.1f}s:", secs);
				for (const auto& [key, value]: stats) {
					text += fmt::format(" [{} reason {}: {} waits {:.1f} ms]", key.first ? "GPU" : "other",
					                    key.second, value.first, value.second / 1e6);
				}
				std::printf("%s\n", text.c_str());
				stats.clear();
				last = now;
			}
		}
	}
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-wait-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          tick, static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics, tick);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

} // namespace Libs::Graphics
