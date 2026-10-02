#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSEQUENCER_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSEQUENCER_H_

// P3b, KYTY_CP_SEQ=1 (cpOps.h, Profiling/analysis/P3-SEQUENCER.md): the graphics queue's front on
// its own thread, "CpSequencer".
//
// Roles:
// - The sequencer thread parses every graphics submission in admission order (GuestGpu::Enqueue
//   hands them over) with the command processor's front (FrontMode::Thread). It emits each
//   submission's ops between a StreamBegin and a StreamEnd op into the processor's op ring. It
//   takes register snapshots for the ops that read registers and publishes direct draws to the
//   draw-prep window.
// - The resolver is the GPU thread. GuestGpu::Process runs a graphics submission by executing
//   its ops (CommandProcessor::ResolveSubmission), with today's scheduling: slices, yields,
//   suspension and retry of waits, the frame fence, service commands between ops.
//
// Ordering points: the sequencer stops until the resolver has executed the op at
// - a lockstep op (waits, predication, COND_EXEC, branches, lockstep reads). The resolver hands
//   the result back through Answer;
// - the start of a submission admitted with a frame fence (NoteStarted);
// - a constant-engine submission, which the resolver parses and executes itself (Handoff op,
//   NoteHandoffDone);
// - command bytes under a pending CP write (NoteExecuted).
//
// Waiting: the sequencer spins KYTY_CP_SEQ_SPIN_US (default 200), then parks on the resolver's
// progress word. The resolver wakes a parked sequencer only when it can use the wake: once the
// executed ops reach the wait's threshold (a pending write's op, the op that frees a window slot
// or a snapshot), a few ops before a lockstep op it waits for (so that it spins again when the
// answer comes, instead of waking up after it), and at every explicit notification (answers, the
// frame fence, handoffs).
// The words both sides use for a wait are stored and loaded seq_cst: a sequencer that announces
// itself parked and re-checks, and a resolver that stores and then looks for a parked sequencer,
// cannot both miss each other. Nothing the sequencer waits for depends on the sequencer, and it
// holds no lock while it waits.
//
// Register snapshots (dispatches, indirect draws, and draws outside the draw-prep window):
// KYTY_CP_SEQ_SNAPSHOTS entries (default 64), taken by the sequencer, released by the resolver
// in order after the op executed.

#include "graphics/guest_gpu/command_processor/cpOps.h"
#include "graphics/guest_gpu/hardwareContext.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {
class CommandProcessor;
} // namespace Libs::Graphics

namespace Libs::Graphics::CpSeq {

// The front's register files at an op.
struct RegisterState {
	HW::Context    context;
	HW::UserConfig user_config;
	HW::Shader     shaders;
};

class SnapshotRing {
public:
	explicit SnapshotRing(uint32_t capacity): m_entries(capacity) {}

	[[nodiscard]] uint32_t Capacity() const noexcept {
		return static_cast<uint32_t>(m_entries.size());
	}
	// Producer.
	[[nodiscard]] bool HasFree() const noexcept {
		return m_acquired - m_released.load(std::memory_order_seq_cst) < m_entries.size();
	}
	// Requires HasFree(). The entry's index, for the op.
	[[nodiscard]] uint32_t Acquire() noexcept {
		return static_cast<uint32_t>(m_acquired++ % m_entries.size());
	}
	[[nodiscard]] RegisterState& Entry(uint32_t index) noexcept { return m_entries[index]; }
	// Consumer: the oldest acquired entry is no longer used.
	void Release() noexcept { m_released.fetch_add(1, std::memory_order_seq_cst); }

private:
	std::vector<RegisterState> m_entries;
	uint64_t                   m_acquired = 0; // producer only
	alignas(64) std::atomic<uint64_t> m_released {0};
};

// A queue-0 submission in admission order.
struct Intake {
	uint64_t                  sequence    = 0; // GuestGpu admission sequence
	uint64_t                  frame_fence = 0;
	std::span<const uint32_t> commands;
	std::span<const uint32_t> constant_commands;
	bool                      graphics        = true; // false: a CPU flip preparation
	bool                      reset_processor = false;
};

class Sequencer {
public:
	explicit Sequencer(CommandProcessor& processor);
	~Sequencer();
	Sequencer(const Sequencer&)            = delete;
	Sequencer& operator=(const Sequencer&) = delete;

	// Starts the thread (once the processor is attached).
	void Start();
	// Stops and joins the thread; the resolver must have finished with the op ring.
	void Stop();

	// GuestGpu::Enqueue: a queue-0 submission, in admission order.
	void Admit(const Intake& intake);

	// ---- Resolver (GPU thread) ----
	void NoteStarted(uint64_t submission);
	void Answer(uint64_t op_sequence, uint64_t value);
	// Every op before `next_op` has executed (and released its snapshot and window slot).
	void NoteExecuted(uint64_t next_op);
	void NoteHandoffDone(uint64_t submission);
	[[nodiscard]] SnapshotRing& Snapshots() noexcept { return m_snapshots; }

	// ---- Sequencer thread (the processor's front) ----
	// Waits until `ready()` holds: spins, then parks on the resolver's progress. While parked,
	// NoteExecuted wakes it once the executed ops reach `wake_at` (0: at every op); explicit
	// notifications always wake it. After a first wake that found `ready()` false, `rewake_at`
	// applies. False when the sequencer is stopping. The time counts as FrameWait
	// CpSeqSequencerWait.
	template <typename Ready>
	bool Wait(Ready&& ready, uint64_t wake_at = 0, uint64_t rewake_at = 0) {
		if (ready()) {
			return true;
		}
		return WaitSlow(ReadyRef(ready), wake_at, rewake_at);
	}
	// The result of lockstep op `op_sequence` (waits for it).
	[[nodiscard]] uint64_t AwaitAnswer(uint64_t op_sequence);
	[[nodiscard]] uint64_t Executed() const noexcept {
		return m_executed.load(std::memory_order_seq_cst);
	}
	// Lockstep op `op_sequence` has been answered (P3c: the speculative parse polls it).
	[[nodiscard]] bool Answered(uint64_t op_sequence) const noexcept {
		return m_answered.load(std::memory_order_acquire) == op_sequence + 1u;
	}
	[[nodiscard]] uint64_t Started() const noexcept {
		return m_started.load(std::memory_order_seq_cst);
	}
	[[nodiscard]] uint64_t HandoffsDone() const noexcept {
		return m_handoff_done.load(std::memory_order_seq_cst);
	}
	[[nodiscard]] bool Stopping() const noexcept {
		return m_stopping.load(std::memory_order_acquire);
	}

private:
	// A non-owning reference to a predicate (the parking path is not a template).
	class ReadyRef {
	public:
		template <typename Ready>
		explicit ReadyRef(Ready& ready)
		    : m_object(const_cast<void*>(static_cast<const void*>(&ready))),
		      m_call([](void* object) { return static_cast<bool>((*static_cast<Ready*>(object))()); }) {
		}
		bool operator()() const { return m_call(m_object); }

	private:
		void* m_object;
		bool (*m_call)(void*);
	};
	bool WaitSlow(const ReadyRef& ready, uint64_t wake_at, uint64_t rewake_at);
	void Bump() noexcept;
	void Run();

	CommandProcessor& m_processor;
	SnapshotRing      m_snapshots;
	uint64_t          m_spin_ns = 0;

	std::mutex              m_intake_mutex;
	std::condition_variable m_intake_ready;
	std::deque<Intake>      m_intake;
	bool                    m_stop = false; // m_intake_mutex
	std::atomic<bool>       m_stopping {false};
	std::thread             m_thread;

	// Resolver progress: bumped by every resolver notification; the parked sequencer waits on it.
	alignas(64) std::atomic<uint64_t> m_progress {0};
	std::atomic<uint32_t> m_parked {0};
	std::atomic<uint64_t> m_wake_at {0}; // the parked wait's executed-op threshold
	std::atomic<uint64_t> m_executed {0};
	std::atomic<uint64_t> m_answered {0}; // lockstep op sequence + 1 of the last answer
	uint64_t              m_answer_value = 0;
	std::atomic<uint64_t> m_started {0};
	std::atomic<uint64_t> m_handoff_done {0};
};

// Thread-mode switches (read once).
[[nodiscard]] uint32_t SnapshotCount();
[[nodiscard]] uint64_t SequencerSpinNs();
[[nodiscard]] uint64_t ResolverSpinNs();

} // namespace Libs::Graphics::CpSeq

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSEQUENCER_H_
