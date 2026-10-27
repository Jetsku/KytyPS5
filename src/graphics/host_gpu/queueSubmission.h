#ifndef KYTY_GRAPHICS_HOST_GPU_QUEUESUBMISSION_H_
#define KYTY_GRAPHICS_HOST_GPU_QUEUESUBMISSION_H_

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace Libs::Graphics {

struct GraphicContext;

// Work that commands of a batch read and that finishes, or reaches its queue, after the batch was
// recorded: TextureCache staging copies made on a host worker (StagingCopier) and upload DMA
// transfers submitted by their own worker (UploadDma).
//
// Whichever thread calls vkQueueSubmit for such a batch (the submission broker, the CP recorder or
// the CP) first waits on the host until the work has finished (host work) or its signal has been
// submitted (another queue's work): SubmitInfo::WaitHostDependencies. So every semaphore a
// submitted batch waits for has its signal operation submitted already, and so does every
// present, which waits for all earlier batches of the queue
// (VUID-vkQueuePresentKHR-pWaitSemaphores-03268). A GPU wait for a later host signal froze RTX 50
// systems: vkQueuePresentKHR blocked in the kernel (hardware-scheduled present), and the staging
// copier's vkSignalSemaphore, which the GPU was waiting for, then blocked behind it.
class SubmitDependency {
public:
	virtual ~SubmitDependency() = default;
	// A value the next batch depends on, or 0 when all work handed out so far is done (finished on
	// the host, so its writes precede vkQueueSubmit, or known complete on the device).
	[[nodiscard]] virtual uint64_t      PendingValue()    = 0;
	// The semaphore the batch waits for the value on, or null when the host wait before
	// vkQueueSubmit is all it needs (host work).
	[[nodiscard]] virtual vk::Semaphore Semaphore() const = 0;
	// Stages of the batch that wait for it.
	[[nodiscard]] virtual vk::PipelineStageFlags WaitStages() const {
		return vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer;
	}
	// Whether a batch depending on `value` may be submitted now, and the host wait until it may.
	// Any thread; never call them with a lock the dependency's worker takes.
	[[nodiscard]] virtual bool Submittable(uint64_t value) = 0;
	virtual void               WaitSubmittable(uint64_t value) = 0;
	// Host-side wait for completion, used when the batch has no free wait slot.
	virtual void WaitHost(uint64_t value) = 0;
};

// KYTY_UPLOAD_DMA_HELP (adapted from usa-ratt/KytyPS5-GT7 3307e49f3): set on a thread that waits
// for SubmitDependency work while it holds no lock (the submission broker's worker). Such a waiter
// may do the dependency's queued batch itself, when its worker has not started it, instead of
// waiting for the worker to wake up.
class SubmitDependencyHelpScope final {
public:
	SubmitDependencyHelpScope() noexcept: m_previous(t_active) { t_active = true; }
	~SubmitDependencyHelpScope() { t_active = m_previous; }
	SubmitDependencyHelpScope(const SubmitDependencyHelpScope&)            = delete;
	SubmitDependencyHelpScope& operator=(const SubmitDependencyHelpScope&) = delete;
	[[nodiscard]] static bool Active() noexcept { return t_active; }

private:
	static inline thread_local bool t_active = false;
	bool                            m_previous;
};

// KYTY_SUBMIT_WAIT_BEFORE_SIGNAL=1 (startup, default off): batches wait for SubmitDependency work
// on the GPU only, as up to int7: the staging copier signals a timeline semaphore from the host and
// batches may reach the queue before the upload DMA transfers they wait for. Comparisons only; it
// can freeze Windows systems with hardware-accelerated GPU scheduling (RTX 50).
[[nodiscard]] bool SubmitWaitBeforeSignal();

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores       = 3;
	static constexpr uint32_t MaxHostDependencies = 2;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                        num_wait_semaphores   = 0;
	uint32_t                                        num_signal_semaphores = 0;
	// SubmitDependency work the batch reads, waited for before vkQueueSubmit. The dependencies
	// outlive every batch that names them (their owners drain the scheduler first).
	std::array<SubmitDependency*, MaxHostDependencies> host_dependencies {};
	std::array<uint64_t, MaxHostDependencies>          host_dependency_values {};
	uint32_t                                          num_host_dependencies = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]   = tick;
	}

	void AddHostDependency(SubmitDependency* dependency, uint64_t value) {
		EXIT_IF(dependency == nullptr || value == 0 ||
		        num_host_dependencies >= MaxHostDependencies);
		host_dependencies[num_host_dependencies]        = dependency;
		host_dependency_values[num_host_dependencies++] = value;
	}

	[[nodiscard]] bool HostDependenciesReady() const {
		for (uint32_t i = 0; i < num_host_dependencies; ++i) {
			if (!host_dependencies[i]->Submittable(host_dependency_values[i])) {
				return false;
			}
		}
		return true;
	}

	// Waits until the batch may be submitted (SubmitDependency). Outside queue_mutex if possible.
	void WaitHostDependencies() const;
};

// A completed GPU signal does not prove vkQueueSubmit has returned on its host
// thread. Reuse and teardown need both facts. Records retain this token so its
// notification stays alive even when the owning scheduler finishes shutting down.
struct SubmissionProgress {
	std::atomic<uint64_t> dispatched_tick {0};
};

// Own every argument used by vkQueueSubmit. No pointer into the scheduler's mutable
// CommandBuffer wrapper or the producer's stack may survive enqueueing.
struct QueuedSubmission {
	SubmitInfo        submit;
	std::shared_ptr<SubmissionProgress> progress;
	vk::Semaphore     master_semaphore = nullptr;
	vk::CommandBuffer command = nullptr;
	uint64_t          tick = 0;
	// This tick has a callback or an explicit CPU wait. It may terminate a
	// coalesced group, but its signal must not move past a later command buffer.
	bool              preserve_completion = false;
	// Optional KYTY_GPU_TIMING slot field: steady_clock nanoseconds when the native vkQueueSubmit
	// containing this record returned. Written before dispatched_tick is published (release), and
	// read only after KnownGpuTick() covers this tick, so the slot outlives the write.
	uint64_t*         dispatch_ns = nullptr;
	uint32_t          debug_op = 0;
	uint64_t          debug_submit = 0;
	uint32_t          debug_arg0 = 0;
	uint32_t          debug_arg1 = 0;
	uint32_t          debug_arg2 = 0;
	uint32_t          debug_arg3 = 0;
	uint64_t          debug_arg4 = 0;
};

// One broker per native queue, shared by the guest and presentation schedulers.
// Queued mode is opt-in through KYTY_SUBMISSION_MODE=queued.
class QueueSubmissionBroker {
public:
	QueueSubmissionBroker() = default;
	~QueueSubmissionBroker();
	KYTY_CLASS_NO_COPY(QueueSubmissionBroker);

	void Initialize(GraphicContext& graphics);
	[[nodiscard]] bool Enabled() const noexcept { return m_enabled; }
	// The producer must not hold queue_mutex: capacity backpressure waits for a
	// consumer that takes that mutex. A scheduler retains its existing single-
	// producer ownership, so its allocated ticks enter this FIFO in order.
	void Enqueue(QueuedSubmission submission);

	// The caller MUST own GraphicContext::queue_mutex. Both the worker and direct
	// queue users acquire that mutex before taking records, so older records can
	// never be detached but still waiting to enter the native queue. A record whose
	// host dependencies are not ready is waited for here, once the records before it
	// were submitted.
	void DrainPendingLocked();
	// The same, but only until `progress` has dispatched `tick`; later records stay queued for the
	// worker. For an operation on another queue that waits for that tick.
	void DrainThroughLocked(const SubmissionProgress* progress, uint64_t tick);
	// Without queue_mutex: waits for the host dependencies of the records pending now (or of those
	// up to `progress`'s record of `tick`), so a drain under queue_mutex that follows rarely waits
	// while holding it.
	void WaitPendingDependencies(const SubmissionProgress* progress = nullptr, uint64_t tick = 0);
	// The caller MUST own queue_mutex. For presentation: submits the pending records that can be
	// submitted now, in order, and never waits. A record whose host dependencies are not ready stays
	// queued with every later record of its scheduler; other schedulers' records (the presenter's
	// blit, which reads only a completed frame) go ahead of it. So the present waits for nothing
	// that has not been submitted, and the game's texture copies never delay it.
	void DrainReadyForPresentLocked();
	// Called by the window owner after both schedulers stop, before device destruction. Joins
	// without holding queue_mutex and submits all accepted records before returning.
	void Shutdown();

private:
	static constexpr size_t MaxQueued = 256;
	static constexpr size_t MaxBatch  = 64;
	void Worker();
	// Worker, queue_mutex held: submits the records pending at entry up to the first one whose host
	// dependencies are not ready. Returns true, with that record's SubmitInfo in `blocked`, when it
	// stopped there; the worker then waits for them without queue_mutex.
	bool DrainReadyLocked(SubmitInfo* blocked);
	// Submits `count` popped records in order, waiting for host dependencies in between.
	void SubmitInOrder(const QueuedSubmission* records, size_t count);
	void SubmitBatch(const QueuedSubmission* records, size_t count);

	GraphicContext*              m_graphics = nullptr;
	bool                         m_enabled = false;
	bool                         m_coalesce = false;
	bool                         m_initialized = false;
	bool                         m_stopping = false;
	bool                         m_stopped = false;
	std::mutex                   m_mutex;
	std::condition_variable      m_available;
	std::condition_variable      m_space_available;
	std::deque<QueuedSubmission> m_pending;
	std::jthread                 m_worker;
	uint64_t                     m_queued_count = 0;
	uint64_t                     m_driver_calls = 0;
	uint64_t                     m_native_submits = 0;
	uint64_t                     m_coalesced_boundaries = 0;
	uint64_t                     m_protected_boundaries = 0;
	size_t                       m_peak_queued = 0;
};

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_HOST_GPU_QUEUESUBMISSION_H_
