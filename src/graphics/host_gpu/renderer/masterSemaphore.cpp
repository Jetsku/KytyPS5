#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/deviceLostReport.h"
#include "graphics/host_gpu/graphicContext.h"

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
		if (result == vk::Result::eErrorDeviceLost) {
			DeviceLostReport::RunOnce();
			DumpDeviceLossDiagnostics(m_graphics, CurrentTick());
		}
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
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-wait-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          tick, static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) {
			DeviceLostReport::RunOnce();
			DumpDeviceLossDiagnostics(m_graphics, tick);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	Refresh();
}

} // namespace Libs::Graphics
