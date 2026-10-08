#include "graphics/host_gpu/queueSubmission.h"

#include "common/cpuPlacement.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/watchdogSubmit.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

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

uint64_t SteadyNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Host dependency waits since the last log line, which follows a wait at most once a minute.
std::atomic<uint64_t> g_host_waits {0};
std::atomic<uint64_t> g_host_wait_ns {0};
std::atomic<uint64_t> g_host_wait_max_ns {0};
std::atomic<uint64_t> g_host_wait_report_ns {0};

void NoteHostWait(uint64_t start_ns) {
	const auto now  = SteadyNs();
	const auto wait = now - start_ns;
	g_host_waits.fetch_add(1, std::memory_order_relaxed);
	g_host_wait_ns.fetch_add(wait, std::memory_order_relaxed);
	auto max = g_host_wait_max_ns.load(std::memory_order_relaxed);
	while (wait > max && !g_host_wait_max_ns.compare_exchange_weak(max, wait,
	                                                               std::memory_order_relaxed)) {
	}
	auto last = g_host_wait_report_ns.load(std::memory_order_relaxed);
	if (last == 0) {
		g_host_wait_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed);
		return;
	}
	constexpr uint64_t Interval = 60'000'000'000ull;
	if (now - last < Interval ||
	    !g_host_wait_report_ns.compare_exchange_strong(last, now, std::memory_order_relaxed)) {
		return;
	}
	const auto waits = g_host_waits.exchange(0, std::memory_order_relaxed);
	const auto total = g_host_wait_ns.exchange(0, std::memory_order_relaxed);
	const auto worst = g_host_wait_max_ns.exchange(0, std::memory_order_relaxed);
	std::printf("Kyty submit waits: last %.0f s, %" PRIu64
	            " batches waited on the host for texture staging copies or upload DMA submits, "
	            "%.2f ms in total, longest %.2f ms\n",
	            static_cast<double>(now - last) / 1e9, waits, static_cast<double>(total) / 1e6,
	            static_cast<double>(worst) / 1e6);
	std::fflush(stdout);
}

} // namespace

bool SubmitWaitBeforeSignal() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SUBMIT_WAIT_BEFORE_SIGNAL");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void SubmitInfo::WaitHostDependencies() const {
	uint64_t start_ns = 0;
	for (uint32_t i = 0; i < num_host_dependencies; ++i) {
		auto* const dependency = host_dependencies[i];
		const auto  value      = host_dependency_values[i];
		if (dependency->Submittable(value)) {
			continue;
		}
		if (start_ns == 0) {
			start_ns = SteadyNs();
		}
		Profiler::ScopedFrameWait wait(Profiler::FrameWait::SubmitDependencyWait);
		const CpGaps::Scope       gap(CpGaps::Cat::SubmitWait);
		dependency->WaitSubmittable(value);
	}
	if (start_ns != 0) {
		NoteHostWait(start_ns);
	}
}

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
	HangWatchdog::Scope wait("submission-broker-space", reinterpret_cast<uint64_t>(this), MaxQueued,
	                         m_pending.size(), 0, submission.tick);
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
		SubmitInfo blocked;
		bool       waiting = false;
		{
			// Never pop before this lock. A direct queue operation can acquire it and
			// drain older records itself without waiting for this worker.
			KYTY_PROFILER_DETAIL_BLOCK("SubmissionQueue::WorkerDrain");
			Common::LockGuard queue_lock(m_graphics->queue_mutex);
			waiting = DrainReadyLocked(&blocked);
			if ((++placement_count & 15u) == 0u) {
				Common::SamplePlacement(Common::ThreadRole::Host);
			}
		}
		if (waiting) {
			// Work the oldest record reads is still running or not submitted yet. Wait for it
			// without queue_mutex: presentation and direct queue users go on meanwhile, and wait
			// for it themselves only if they must submit that record first. It stays queued.
			blocked.WaitHostDependencies();
		}
	}
}

bool QueueSubmissionBroker::DrainReadyLocked(SubmitInfo* blocked) {
	size_t remaining;
	{
		std::lock_guard lock(m_mutex);
		remaining = m_pending.size();
	}
	while (remaining != 0) {
		std::array<QueuedSubmission, MaxBatch> records;
		size_t                                 count   = 0;
		bool                                   stopped = false;
		{
			std::lock_guard lock(m_mutex);
			EXIT_IF(m_pending.size() < remaining);
			const auto limit = std::min(remaining, MaxBatch);
			while (count < limit) {
				auto& next = m_pending.front();
				if (!next.submit.HostDependenciesReady()) {
					*blocked = next.submit;
					stopped  = true;
					break;
				}
				records[count++] = std::move(next);
				m_pending.pop_front();
			}
			if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
				TracyPlot("SubmissionQueue.Pending", static_cast<double>(m_pending.size()));
			}
		}
		if (count != 0) {
			m_space_available.notify_all();
			SubmitBatch(records.data(), count);
			remaining -= count;
		}
		if (stopped) {
			return true;
		}
	}
	return false;
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
		SubmitInOrder(records.data(), count);
		remaining -= count;
	}
}

void QueueSubmissionBroker::DrainThroughLocked(const SubmissionProgress* progress, uint64_t tick) {
	if (!m_enabled || progress == nullptr) {
		return;
	}
	while (progress->dispatched_tick.load(std::memory_order_acquire) < tick) {
		std::array<QueuedSubmission, MaxBatch> records;
		size_t                                 count = 0;
		{
			std::lock_guard lock(m_mutex);
			bool            last = false;
			while (!last && count < MaxBatch && !m_pending.empty()) {
				auto& next = m_pending.front();
				last       = next.progress.get() == progress && next.tick >= tick;
				records[count++] = std::move(next);
				m_pending.pop_front();
			}
			if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
				TracyPlot("SubmissionQueue.Pending", static_cast<double>(m_pending.size()));
			}
		}
		if (count == 0) {
			return; // not pending: the caller made sure the tick reached the broker
		}
		m_space_available.notify_all();
		SubmitInOrder(records.data(), count);
	}
}

void QueueSubmissionBroker::DrainReadyForPresentLocked() {
	if (!m_enabled) {
		return;
	}
	std::vector<QueuedSubmission> ready;
	{
		std::lock_guard lock(m_mutex);
		// Schedulers with a held record; their later records stay behind it, in order.
		std::array<const SubmissionProgress*, 4> held {};
		size_t                                   held_count = 0;
		bool                                     hold_all   = false;
		std::deque<QueuedSubmission>             kept;
		ready.reserve(m_pending.size());
		for (auto& record: m_pending) {
			const auto* progress = record.progress.get();
			bool        hold     = hold_all || std::find(held.begin(), held.begin() + held_count,
			                                             progress) != held.begin() + held_count;
			if (!hold && !record.submit.HostDependenciesReady()) {
				hold = true;
				if (held_count < held.size()) {
					held[held_count++] = progress;
				} else {
					hold_all = true;
				}
			}
			if (hold) {
				kept.push_back(std::move(record));
			} else {
				ready.push_back(std::move(record));
			}
		}
		m_pending = std::move(kept);
		if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
			TracyPlot("SubmissionQueue.Pending", static_cast<double>(m_pending.size()));
		}
	}
	if (ready.empty()) {
		return;
	}
	m_space_available.notify_all();
	for (size_t first = 0; first < ready.size(); first += MaxBatch) {
		SubmitBatch(ready.data() + first, std::min(MaxBatch, ready.size() - first));
	}
}

void QueueSubmissionBroker::WaitPendingDependencies(const SubmissionProgress* progress,
                                                    uint64_t                  tick) {
	if (!m_enabled) {
		return;
	}
	// Per dependency, the newest value a pending record names: values only grow along the queue.
	SubmitInfo needs;
	bool       found = progress == nullptr;
	{
		std::lock_guard lock(m_mutex);
		for (const auto& record: m_pending) {
			const auto& submit = record.submit;
			for (uint32_t i = 0; i < submit.num_host_dependencies; ++i) {
				auto* const dependency = submit.host_dependencies[i];
				const auto  value      = submit.host_dependency_values[i];
				uint32_t    slot       = 0;
				while (slot < needs.num_host_dependencies &&
				       needs.host_dependencies[slot] != dependency) {
					++slot;
				}
				if (slot < needs.num_host_dependencies) {
					needs.host_dependency_values[slot] =
					    std::max(needs.host_dependency_values[slot], value);
				} else if (slot < SubmitInfo::MaxHostDependencies) {
					needs.AddHostDependency(dependency, value);
				}
			}
			if (progress != nullptr && record.progress.get() == progress && record.tick >= tick) {
				found = record.tick == tick;
				break;
			}
		}
	}
	if (!found) {
		return; // the worker has taken that tick meanwhile
	}
	// The dependencies outlive the records (SubmitInfo), which the worker may submit meanwhile.
	needs.WaitHostDependencies();
}

void QueueSubmissionBroker::SubmitInOrder(const QueuedSubmission* records, size_t count) {
	size_t first = 0;
	for (size_t i = 0; i < count; ++i) {
		if (records[i].submit.HostDependenciesReady()) {
			continue;
		}
		// The records before it reach the GPU first; then this thread waits for its work.
		if (i > first) {
			SubmitBatch(records + first, i - first);
			first = i;
		}
		records[i].submit.WaitHostDependencies();
	}
	if (first < count) {
		SubmitBatch(records + first, count - first);
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
		HangWatchdog::Scope       native(
		    "vkQueueSubmit-broker",
		    reinterpret_cast<uint64_t>(static_cast<VkQueue>(m_graphics->queue)),
		    records[count - 1].tick, records[0].tick, native_count, count);
		HangWatchdog::DebugDelay("submit", records[count - 1].tick);
		if (HangWatchdog::Enabled()) {
			for (size_t i = 0; i < native_count; ++i)
				NoteWatchdogSubmit(m_graphics->queue, submits[i]);
		}
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
