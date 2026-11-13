#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

namespace CpSeq {
class Sequencer;
} // namespace CpSeq

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void              Submit(std::span<const uint32_t> draw_commands,
	                         std::span<const uint32_t> constant_commands);
	void              SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void              SubmitFlipPreparation(uint64_t request_id);
	void              Done();
	[[nodiscard]] int GetFrameNum() const;
	// The admission sequence of the latest submission on any queue (0: none yet). Any thread.
	[[nodiscard]] uint64_t SubmittedWork();
	// Waits until the command processor has completed every submission admitted up to `marker`
	// (SubmittedWork), or the GPU stops: true. False when `cancel` is set first. Not on the GPU
	// thread.
	[[nodiscard]] bool WaitForWork(uint64_t marker, const std::atomic_bool& cancel);

	[[nodiscard]] static bool IsGpuThread() noexcept;

	// Like SendCommand, but returns false (dropping nothing: the caller keeps the command's
	// work) once the GPU no longer accepts external commands (shutdown). Any thread.
	[[nodiscard]] bool TrySendCommand(Common::UniqueFunction<void>&& command);
	// Something a suspended (blocked) queue may wait for has changed: clear the blocked marks so
	// the scheduler retries them now, and wake it. Any thread.
	void NotifyProgress();
	// Some async compute queue has a submission that is not suspended. GPU thread.
	[[nodiscard]] bool HasRunnableComputeWork();

	// End-of-pipe labels whose guest write is deferred to their tick's completion (see
	// CommandProcessor::TryDeferLabel). Registration and completion run on the GPU thread.
	// cp_visible (KYTY_LABEL_MODE=split): the command processor's own reads (WAIT_REG_MEM) see
	// `value` while the guest write is pending (CpVisibleLabelValue).
	void AddDeferredLabel(uint64_t address, uint32_t size, uint64_t tick, uint64_t value = 0,
	                      bool cp_visible = false);
	void RemoveDeferredLabel(uint64_t address, uint64_t tick);
	[[nodiscard]] bool HasDeferredLabels() const noexcept {
		return m_deferred_label_count.load(std::memory_order_acquire) != 0;
	}
	// Newest tick of a pending deferred label overlapping the range, or 0.
	[[nodiscard]] uint64_t DeferredLabelTick(uint64_t address, uint64_t size);
	// The bytes [address, address + size) as the newest pending CP-visible deferred labels
	// covering them would write them, over `value` (which holds the guest memory bytes). False
	// when no such label overlaps the range.
	[[nodiscard]] bool CpVisibleLabelValue(uint64_t address, uint32_t size, void* value);
	// A WAIT_REG_MEM passed on the CP-visible pending labels covering the range.
	void NoteCpVisibleLabelsWaited(uint64_t address, uint64_t size);
	// A command-processor write to [address, address + size): pending labels there are no longer
	// what the command processor reads, and the guest write of one a CP wait passed on is skipped
	// (on hardware it landed before that wait passed, so before this write).
	void DropCpVisibleLabels(uint64_t address, uint64_t size);
	// The pending label (address, tick) was replaced by a later CP write (DropCpVisibleLabels).
	[[nodiscard]] bool DeferredLabelSuperseded(uint64_t address, uint64_t tick);
	// A pending CP-visible label overlaps [address, address + size) (KYTY_LABEL_MODE=split).
	[[nodiscard]] bool CpVisibleLabelOverlaps(uint64_t address, uint64_t size);

private:
	struct DeferredLabel {
		uint64_t address    = 0;
		uint64_t size       = 0;
		uint64_t tick       = 0;
		uint64_t value      = 0;
		bool     cp_visible = false;
		bool     cp_waited  = false;
		bool     superseded = false;
	};
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Graphics, Compute, FlipPreparation };

	struct Submission {
		SubmissionType            type     = SubmissionType::Graphics;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		std::span<const uint32_t> constant_commands;
		Pm4Execution              command_execution;
		Pm4Execution              constant_execution;
		bool                      reset_processor   = false;
		bool                      started           = false;
		bool                      command_complete  = false;
		bool                      constant_complete = false;
		bool                      blocked           = false;
		bool                      slice_progress    = false; // the last slice advanced
		uint64_t                  flip_request_id   = 0;
		uint64_t                  enqueue_ns        = 0;
		uint64_t                  pace_enqueue_ns   = 0; // KYTY_PACE_LOG (steady clock)
		uint64_t                  sequence          = 0; // admission order (m_queue_mutex)
		// Last sequence admitted before the latest Done at admission (KYTY_FRAME_FENCE): the
		// submission starts only after all of those have completed. 0 = no fence.
		uint64_t frame_fence = 0;
		// When the scheduler first passed this front over for the fence (safety timeout), or 0.
		uint64_t fence_hold_ns = 0;
		// KYTY_CP_SEQ=1: the resolver parses and executes this (constant-engine) submission
		// itself, in Direct mode, instead of executing the sequencer's ops.
		bool handoff = false;
	};

	void              Enqueue(Submission submission);
	void              WaitForIdle();
	void              ProcessCommands();
	[[nodiscard]] bool HasPendingCommands() const noexcept {
		return m_pending_commands.load(std::memory_order_acquire) != 0;
	}
	bool              Process(Submission& submission);
	// KYTY_CP_SEQ=1: a queue-0 submission, executed from the sequencer's ops.
	bool              ProcessSequenced(Submission& submission);
	// The graphics submission's CE/DE streams parsed and executed on this thread (Direct).
	bool              ProcessGraphicsDirect(Submission& submission, CommandProcessor& cp);
	// The submission may start (or continue) now as far as the frame fence is concerned.
	// Requires m_queue_mutex.
	[[nodiscard]] bool FrameFencePassed(const Submission& submission) const;
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_submission_mutex;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	std::deque<DeferredLabel>                      m_deferred_labels; // m_queue_mutex
	std::atomic_uint32_t                           m_deferred_label_count {0};
	std::atomic_bool                               m_cp_visible_labels {false}; // ever added
	// Some queue front is marked blocked (set under m_queue_mutex; NotifyProgress fast path).
	std::atomic_bool                               m_has_blocked {false};
	// Bounded Done (KYTY_AGC_DONE_MODE): admission sequence numbers of submissions not yet
	// completed, the last sequence admitted before the latest Done, and its waiters.
	std::set<uint64_t>                             m_in_flight;
	uint64_t                                       m_next_submission_sequence = 1;
	uint64_t                                       m_done_boundary            = 0;
	uint32_t                                       m_done_waiters             = 0;
	Common::CondVar                                m_done_progress;
	// WaitForWork: the markers waited for; m_work_progress is signalled once the smallest is reached.
	std::multiset<uint64_t>                        m_work_waits;
	Common::CondVar                                m_work_progress;
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_graphics_done     = true;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;
	// KYTY_CP_SEQ=1 (cpSequencer.h): the graphics queue's front thread; destroyed before m_gfx_cp.
	struct SequencerDeleter {
		void operator()(CpSeq::Sequencer* sequencer) const noexcept;
	};
	std::unique_ptr<CpSeq::Sequencer, SequencerDeleter> m_sequencer;

	uint64_t        m_submit_id = 0;
	std::atomic_int m_done_num  = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
