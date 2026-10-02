#include "graphics/host_gpu/queueSubmission.h"

#include "common/cpuPlacement.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace Libs::Graphics {

namespace {

bool HasOnlyMasterSignal(const QueuedSubmission& record) {
	return record.progress != nullptr && record.master_semaphore != nullptr && record.tick != 0 &&
	       record.submit.num_wait_semaphores == 0 && record.submit.num_signal_semaphores == 1 &&
	       record.submit.signal_semaphores[0] == record.master_semaphore &&
	       record.submit.signal_ticks[0] == record.tick;
}

bool CanCoalesceAfter(const QueuedSubmission& previous, const QueuedSubmission& next) {
	// Preserve observable completions as group ends. The next record may itself
	// be protected: adding an unobserved prefix does not move its final signal.
	return !previous.preserve_completion && HasOnlyMasterSignal(previous) &&
	       HasOnlyMasterSignal(next) && previous.progress == next.progress &&
	       previous.master_semaphore == next.master_semaphore &&
	       previous.tick != std::numeric_limits<uint64_t>::max() &&
	       next.tick == previous.tick + 1;
}

} // namespace

QueueSubmissionBroker::~QueueSubmissionBroker() {
	Shutdown();
}

void QueueSubmissionBroker::Initialize(GraphicContext& graphics) {
	EXIT_IF(m_initialized || graphics.queue == nullptr);
	m_graphics    = &graphics;
	m_initialized = true;
	const auto* mode = std::getenv("KYTY_SUBMISSION_MODE");
	m_enabled = mode != nullptr && std::strcmp(mode, "queued") == 0;
	const auto* coalesce = std::getenv("KYTY_SUBMISSION_COALESCE");
	m_coalesce = m_enabled && coalesce != nullptr && std::strcmp(coalesce, "1") == 0;
	if (m_enabled) {
		std::printf("Kyty submission mode: queued (up to %zu commands per driver call; "
		            "completion coalescing %s)\n", MaxBatch, m_coalesce ? "enabled" : "disabled");
		std::fflush(stdout);
		m_worker = std::jthread([this] { Worker(); });
	}
}

void QueueSubmissionBroker::Enqueue(QueuedSubmission submission) {
	EXIT_IF(!m_enabled || submission.command == nullptr);
	std::unique_lock lock(m_mutex);
	// The worker needs only queue_mutex and this mutex. In particular it never
	// needs the renderer lock, which the producer can own while waiting for space.
	m_space_available.wait(lock, [this] { return m_stopping || m_pending.size() < MaxQueued; });
	EXIT_IF(m_stopping);
	m_pending.push_back(std::move(submission));
	++m_queued_count;
	m_peak_queued = std::max(m_peak_queued, m_pending.size());
	if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
		TracyPlot("SubmissionQueue.Pending", static_cast<double>(m_pending.size()));
		TracyPlot("SubmissionQueue.Commands", static_cast<double>(m_queued_count));
	}
	lock.unlock();
	m_available.notify_one();
}

void QueueSubmissionBroker::Worker() {
	KYTY_PROFILER_THREAD("Vulkan queue submission");
	// It only blocks between submissions; the CP's work reaches the GPU through it.
	Common::RaiseServiceThreadPriority();
	uint32_t placement_count = 0; // placement samples (common/cpuPlacement.h), every 16th drain
	for (;;) {
		{
			std::unique_lock lock(m_mutex);
			m_available.wait(lock, [this] { return m_stopping || !m_pending.empty(); });
			if (m_stopping && m_pending.empty()) {
				return;
			}
		}
		// Never pop before this lock. A direct queue operation can acquire it and
		// drain older records itself without waiting for this worker.
		KYTY_PROFILER_DETAIL_BLOCK("SubmissionQueue::WorkerDrain");
		Common::LockGuard queue_lock(m_graphics->queue_mutex);
		DrainPendingLocked();
		if ((++placement_count & 15u) == 0u) {
			Common::SamplePlacement(Common::ThreadRole::Host);
		}
	}
}

void QueueSubmissionBroker::DrainPendingLocked() {
	if (!m_enabled) {
		return;
	}
	size_t remaining;
	{
		std::lock_guard lock(m_mutex);
		remaining = m_pending.size();
	}
	// Drain the records preceding this direct queue operation. New arrivals can
	// follow it, and cannot keep presentation or capture boundaries waiting forever.
	while (remaining != 0) {
		std::array<QueuedSubmission, MaxBatch> records;
		const auto count = std::min(remaining, MaxBatch);
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_pending.size() < count);
			for (size_t i = 0; i < count; ++i) {
				records[i] = std::move(m_pending.front());
				m_pending.pop_front();
			}
			if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
				TracyPlot("SubmissionQueue.Pending", static_cast<double>(m_pending.size()));
			}
		}
		m_space_available.notify_all();
		SubmitBatch(records.data(), count);
		remaining -= count;
	}
}

void QueueSubmissionBroker::SubmitBatch(const QueuedSubmission* records, size_t count) {
	EXIT_IF(count == 0 || count > MaxBatch);
	std::array<vk::TimelineSemaphoreSubmitInfo, MaxBatch> timelines {};
	std::array<vk::SubmitInfo, MaxBatch> submits {};
	std::array<vk::CommandBuffer, MaxBatch> commands {};
	size_t protected_count = 0;
	for (size_t i = 0; i < count; ++i) {
		commands[i] = records[i].command;
		protected_count += records[i].preserve_completion ? 1 : 0;
	}
	size_t native_count = 0;
	for (size_t first = 0; first < count;) {
		size_t last = first;
		if (m_coalesce) {
			while (last + 1 < count && CanCoalesceAfter(records[last], records[last + 1])) {
				++last;
			}
		}
		// The non-coalescing path retains exactly one VkSubmitInfo and the
		// original waits/signals for every record. Coalesced groups use the last
		// master value and keep the command buffers in their original order.
		const auto& original = records[last].submit;
		auto& timeline = timelines[native_count];
		timeline.waitSemaphoreValueCount   = original.num_wait_semaphores;
		timeline.pWaitSemaphoreValues      = original.wait_ticks.data();
		timeline.signalSemaphoreValueCount = original.num_signal_semaphores;
		timeline.pSignalSemaphoreValues    = original.signal_ticks.data();
		auto& submit = submits[native_count];
		submit.pNext                = &timeline;
		submit.waitSemaphoreCount   = original.num_wait_semaphores;
		submit.pWaitSemaphores      = original.wait_semaphores.data();
		submit.pWaitDstStageMask    = original.wait_stages.data();
		submit.commandBufferCount   = static_cast<uint32_t>(last - first + 1);
		submit.pCommandBuffers      = &commands[first];
		submit.signalSemaphoreCount = original.num_signal_semaphores;
		submit.pSignalSemaphores    = original.signal_semaphores.data();
		++native_count;
		first = last + 1;
	}
	EXIT_IF(native_count == 0 || native_count > count);
	vk::Result result;
	{
		KYTY_PROFILER_DETAIL_BLOCK("SubmissionQueue::DriverSubmit");
		Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::DriverSubmit);
		result = m_graphics->queue.submit(static_cast<uint32_t>(native_count), submits.data(), nullptr);
	}
	++m_driver_calls;
	m_native_submits += native_count;
	m_coalesced_boundaries += count - native_count;
	m_protected_boundaries += protected_count;
	if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
		// BatchSize retains its B2 meaning: original command buffers per driver
		// call. NativeSubmitInfos records the separate VkSubmitInfo reduction.
		TracyPlot("SubmissionQueue.BatchSize", static_cast<double>(count));
		TracyPlot("SubmissionQueue.DriverCalls", static_cast<double>(m_driver_calls));
		TracyPlot("SubmissionQueue.NativeSubmitInfos", static_cast<double>(m_native_submits));
		TracyPlot("SubmissionQueue.NativeSubmitInfosPerCall", static_cast<double>(native_count));
		TracyPlot("SubmissionQueue.CoalescedBoundaries", static_cast<double>(m_coalesced_boundaries));
		TracyPlot("SubmissionQueue.ProtectedBoundaries", static_cast<double>(m_protected_boundaries));
		TracyPlot("SubmissionQueue.ProtectedBoundariesPerCall", static_cast<double>(protected_count));
	}
	if (result == vk::Result::eErrorDeviceLost) DumpDeviceLossDiagnostics(*m_graphics, records[0].tick, true);
	if (result != vk::Result::eSuccess) {
		// A batched driver failure does not identify a single offending entry.
		// Preserve each entry's diagnostics rather than reading a reused wrapper.
		for (size_t i = 0; i < count; ++i) {
			const auto& record = records[i];
			std::printf("vkQueueSubmit batch entry %zu/%zu failed: %s (%d), tick=%" PRIu64
			            " debug_op=%u debug_submit=%" PRIu64
			            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
			            i + 1, count, vk::to_string(result).c_str(), static_cast<int>(result),
			            record.tick, record.debug_op, record.debug_submit, record.debug_arg0,
			            record.debug_arg1, record.debug_arg2, record.debug_arg3, record.debug_arg4);
		}
		std::fflush(stdout);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	uint64_t dispatch_ns = 0;
	for (size_t i = 0; i < count; ++i) {
		if (records[i].dispatch_ns != nullptr) {
			if (dispatch_ns == 0) {
				dispatch_ns = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now().time_since_epoch())
				        .count());
			}
			*records[i].dispatch_ns = dispatch_ns;
		}
	}
	// Publish only after the entire native call has returned. This preserves
	// command-buffer, descriptor, and semaphore lifetime even if the GPU finished
	// a submission before the driver's host call returned.
	for (size_t i = 0; i < count; ++i) {
		const auto& record = records[i];
		EXIT_IF(!record.progress);
		record.progress->dispatched_tick.store(record.tick, std::memory_order_release);
		record.progress->dispatched_tick.notify_all();
	}
}

void QueueSubmissionBroker::Shutdown() {
	{
		std::lock_guard lock(m_mutex);
		if (!m_initialized || m_stopped) {
			return;
		}
		m_stopping = true;
	}
	m_space_available.notify_all();
	m_available.notify_all();
	if (m_worker.joinable()) {
		m_worker.join();
	}
	{
		std::lock_guard lock(m_mutex);
		EXIT_IF(!m_pending.empty());
		m_stopped = true;
	}
	if (m_enabled) {
		std::printf("Kyty queued submissions: %" PRIu64 " commands, %" PRIu64
		            " driver calls, %" PRIu64 " native submit infos, %" PRIu64
		            " coalesced boundaries, %" PRIu64 " protected boundaries, peak pending=%zu\n",
		            m_queued_count, m_driver_calls, m_native_submits, m_coalesced_boundaries,
		            m_protected_boundaries, m_peak_queued);
		std::fflush(stdout);
	}
}

} // namespace Libs::Graphics
