#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

#include <queue>

#include <thread>
#include <vector>

namespace Libs::Graphics {

class GpuTimestampRing;
class CommandRecorder;
class EopTimestampRing;

// Host work that commands of the current recording read and that finishes after recording
// (TextureCache staging copies made on a worker). Submit makes the batch wait for it on the
// GPU through a host-signalled timeline semaphore, so the recording thread never blocks.
class SubmitDependency {
public:
	virtual ~SubmitDependency() = default;
	// A value of Semaphore() the next batch must wait for, or 0 when all work handed out so far
	// has finished on the host (its writes then precede vkQueueSubmit).
	[[nodiscard]] virtual uint64_t      PendingValue()    = 0;
	[[nodiscard]] virtual vk::Semaphore Semaphore() const = 0;
	// Stages of the batch that wait for it.
	[[nodiscard]] virtual vk::PipelineStageFlags WaitStages() const {
		return vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer;
	}
	// Host-side wait, used when the batch has no free wait slot.
	virtual void WaitHost(uint64_t value) = 0;
};

class CommandScheduler {
public:
	// Diagnostic attribution only; both kinds retain the same completion boundary.
	enum class PriorityOperationKind { Generic, EopInterrupt };
	// Only the guest scheduler carries KYTY_GPU_TIMING timestamps. Presenter submissions wait on
	// image acquisition at the transfer stage, so their top-of-pipe spans would include that wait.
	enum class Role { Guest, Presenter };

	CommandScheduler(RenderContext& context, GraphicContext& graphics, Role role);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           FlushAndWait();
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {}, bool force_completion = false);
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	// Runs every normal operation whose tick has completed, in order. Before each one it waits for
	// the priority operations of that tick and earlier: a normal operation may free what they
	// still use.
	void                      PopPendingOperations();
	// The same, for the draw and dispatch paths' opportunistic housekeeping. With
	// KYTY_PENDING_OPS_NOWAIT (default on) it never waits for the priority runner. At the first
	// completed operation whose tick still has priority operations pending, it stops and leaves
	// that operation and those after it queued, in order, for a later pop. Every blocking caller
	// (Finish, the fault manager, explicit waits) still uses PopPendingOperations.
	void                      PopReadyOperations();
	void                      DrainPriorityOperations();
	// KYTY_PRIORITY_WAIT_SPIN_US (default 0): first spin that long for the priority runner.
	void                      WaitPriorityOperations(uint64_t tick);
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(
	    Common::UniqueFunction<void>&& operation,
	    PriorityOperationKind kind = PriorityOperationKind::Generic);
	[[nodiscard]] static bool InDeferredOperation() noexcept;
	// Called on the completion runner after every priority operation (e.g. to wake queues
	// suspended on what it published). Clear it, then DrainPriorityOperations, before the
	// context dies: the call happens while the operation is still marked active.
	using ProgressHook = void (*)(void* context);
	void SetProgressHook(ProgressHook hook, void* context);
	// KYTY_PRIORITY_WAKE_BATCH=0 restores a runner wake per queued operation and a waiter
	// broadcast after every operation.
	[[nodiscard]] static bool PriorityWakeupsBatched();

	// Draw-prep: the register set the current command buffer reads. A committed draw's buffer is
	// pointed at that draw's register snapshot (the live registers may already belong to later
	// packets) and restored afterwards; it survives command-buffer restarts in between because
	// the scheduler reuses one CommandBuffer wrapper.
	struct RegisterBinding {
		HW::Context*    registers   = nullptr;
		HW::UserConfig* user_config = nullptr;
		HW::Shader*     shaders     = nullptr;
	};
	[[nodiscard]] RegisterBinding BindRegisters(HW::Context& registers, HW::UserConfig& user_config,
	                                            HW::Shader& shaders) noexcept {
		const RegisterBinding previous {m_command.m_registers, m_command.m_user_config,
		                                m_command.m_shaders};
		m_command.Bind(registers, user_config, shaders);
		return previous;
	}
	void RestoreRegisters(const RegisterBinding& binding) noexcept {
		m_command.m_registers   = binding.registers;
		m_command.m_user_config = binding.user_config;
		m_command.m_shaders     = binding.shaders;
	}

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	// KYTY_EOP_TIMESTAMPS=gpu: the guest timestamp query ring; null in record mode.
	[[nodiscard]] EopTimestampRing* GuestTimestamps() const noexcept { return m_eop_timestamps.get(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }
	// Recording-thread only; the owner clears it before it is destroyed. Slot 0: TextureCache
	// staging copies; slot 1: BufferCache upload DMA.
	static constexpr uint32_t SubmitDependencySlots = 2;
	void SetSubmitDependency(SubmitDependency* dependency, uint32_t slot = 0) noexcept {
		m_submit_dependencies[slot] = dependency;
	}
	// KYTY_CP_RECORDER (commandRecorder.h): the guest scheduler's recorder, or null.
	[[nodiscard]] CommandRecorder* Recorder() const noexcept { return m_recorder.get(); }
	// Waits until the recorder has handed tick `tick` (and every older one) to the queue or the
	// submission broker; a no-op without a recorder. Code that assumes "every tick older than the
	// current recording reached the queue or the broker" calls it first. from_producer: called by
	// the recording thread (the recorder is then woken first). Never call it while holding
	// GraphicContext::queue_mutex: the recorder may need it to submit.
	void WaitRecorded(uint64_t tick, bool from_producer = true);

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();
		// KYTY_CP_RECORDER: the recorder records from this pool; growing it (the only pool call
		// the recording thread makes) drains the recorder first.
		void SetRecorder(CommandRecorder* recorder) noexcept { m_recorder = recorder; }

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		CommandRecorder*               m_recorder = nullptr;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);
	// Pops normal operations (see PopPendingOperations/PopReadyOperations).
	void PopOperations(bool wait_for_priority);
	// m_operation_mutex held: no priority operation of `tick` or earlier is queued or running.
	[[nodiscard]] bool PriorityDoneLocked(uint64_t tick) const noexcept;

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	// The completion runner sleeps on its own condition, so a push can wake exactly it.
	std::condition_variable      m_priority_available;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	// Threads in WaitPriorityOperations/DrainPriorityOperations (m_operation_mutex).
	uint32_t                     m_priority_waiters     = 0;
	// Bumped by the priority runner after each operation; WaitPriorityOperations' spin polls it
	// instead of taking m_operation_mutex.
	std::atomic<uint64_t>        m_priority_progress {0};
	ProgressHook                 m_progress_hook         = nullptr;
	void*                        m_progress_hook_context = nullptr;
	OperationState               m_operation_state      = OperationState::Open;
	// Guarded by m_operation_mutex, alongside callback registration and the
	// queued-submit tick transition. Captured into each owned submission record.
	bool                         m_preserve_current_completion = false;
	// Aggregate tracing reasons, guarded by m_operation_mutex. They never affect submission.
	bool                         m_diagnostic_eop_completion     = false;
	bool                         m_diagnostic_generic_completion = false;
	// KYTY_GPU_TIMING ring; null when disabled. Owned by the recording producer, like m_command.
	std::unique_ptr<GpuTimestampRing> m_gpu_timing;
	// KYTY_EOP_TIMESTAMPS=gpu query ring (guest scheduler only); null otherwise. Recording
	// producer, except the deferred-label reads (EopTimestampRing::TakeDeferred) on the priority
	// runner, which is joined before members declared above it are destroyed.
	std::unique_ptr<EopTimestampRing> m_eop_timestamps;
	// Guest scheduler with KYTY_GPU_OP_PROFILE / counters enabled (gpuOpProfiler.h).
	bool m_gpu_ops = false;
	// KYTY_PENDING_REFRESH_US: when the draw-entry pop last queried the GPU's progress (steady
	// clock nanoseconds; the recording producer only).
	uint64_t m_last_pending_refresh_ns = 0;
	std::array<SubmitDependency*, SubmitDependencySlots> m_submit_dependencies {};
	// KYTY_CP_RECORDER: the recorder thread of the guest scheduler (after m_gpu_timing, whose
	// ring it drives; destroyed before it).
	std::unique_ptr<CommandRecorder> m_recorder;
	// Declared last: the runner starts in the constructor and uses the members above.
	std::jthread m_priority_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
