#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ENCODEOFFLOAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ENCODEOFFLOAD_H_

// KYTY_ENC_OFFLOAD (live; default 0; 1 on; verify): the emission of draws (drawEncode.h) runs on one
// in-order emit thread between the commit thread and the recorder (the C1 cut at the emission point:
// resolve on the commit thread, a value packet per draw part, emit later in order).
//
//  - The commit thread does a draw's cache and barrier work and builds its descriptor writes as
//    before, then posts its encode parts (job data it owns until they are spliced) and goes on.
//  - The encode thread runs the jobs in order into its own command-stream ring, with an encoder of
//    the command buffer's options, updating the command buffer's shadow contents (push constants,
//    push-descriptor state). While jobs are pending that shadow belongs to the encode thread; the
//    commit thread decides only what it owns (bound pipelines, epochs, the dynamic-state shadow).
//  - Settle: before the commit thread's next packet (the command buffer's encoder open hook) and
//    before it touches the shadow contents (CommandBuffer's settle hook) it waits for the pending
//    jobs and splices their packets into the command stream in order (Encoder::Splice: verify
//    sequence and digest kept). So the stream holds exactly what the inline encode would have
//    written, in the same order.
//  - Verify: jobs run on a copy of the shadow; the commit thread encodes inline as well, and each
//    job's packets (replayed as (op, argument hash)) and resulting shadow are compared with the
//    inline ones; the inline result is used.
//
// Engine is the generic hand-over (no renderer types): unit tested in tests/EncodeOffloadTests.cpp.

#include "graphics/host_gpu/renderer/commandStream.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace Libs::Graphics::EncodeOffload {

class Engine {
public:
	// Runs one job into the encode thread's encoder.
	using RunFn = void (*)(void* job, CommandStream::Encoder& out);

	static constexpr uint32_t MaxJobs = 64;

	Engine();
	~Engine();
	Engine(const Engine&)            = delete;
	Engine& operator=(const Engine&) = delete;

	// The encode thread's encoder takes the command stream encoder's options (verify blocks and
	// barrier sites; never capture-mode sites). Re-created only while nothing is pending.
	void Configure(const CommandStream::Encoder::Options& options);

	// Commit thread. Posts a job; false when MaxJobs are pending (the caller settles or encodes
	// inline). `job` stays the caller's and must not change until the job is settled or discarded.
	[[nodiscard]] bool Post(void* job, RunFn run);
	[[nodiscard]] uint32_t Pending() const noexcept { return m_posted - m_settled; }
	// Waits for every pending job and appends its packets to `target` in posting order. Returns the
	// bytes appended.
	uint64_t Settle(CommandStream::Encoder& target);
	// Verify: waits for the oldest pending job and logs its packets (as replayed calls) into `log`,
	// then drops them (nothing is spliced).
	template <typename Log>
	void TakeOldest(Log& log);

	// Counters (relaxed; read by the reporter).
	struct Stats {
		std::atomic<uint64_t> jobs {0};
		std::atomic<uint64_t> settles {0};
		std::atomic<uint64_t> spliced_bytes {0};
		std::atomic<uint64_t> spliced_jobs {0};
		std::atomic<uint64_t> waits {0};      // settles that found a job not yet run
		std::atomic<uint64_t> wait_ns {0};    // sampled (every 64th settle)
		std::atomic<uint64_t> splice_ns {0};  // sampled (every 64th settle)
		std::atomic<uint64_t> timed {0};
		std::atomic<uint64_t> worker_ns {0};  // sampled (every 64th job)
		std::atomic<uint64_t> worker_timed {0};
		std::atomic<uint64_t> parks {0};
	};
	[[nodiscard]] Stats& GetStats() noexcept { return m_stats; }

private:
	struct Slot {
		void*                 job = nullptr;
		RunFn                 run = nullptr;
		uint64_t              end = 0; // ring position after the job's packets
		std::atomic<uint32_t> done {0};
	};
	void     Loop();
	void     WaitDone(Slot& slot);
	uint64_t Now() const;

	CommandStream::Ring                     m_ring {4u << 20u};
	std::unique_ptr<CommandStream::Encoder> m_encoder;
	std::array<Slot, MaxJobs>               m_slots;
	// m_posted: commit thread writes, encode thread reads; m_run: encode thread only.
	alignas(64) std::atomic<uint64_t> m_posted_shared {0};
	std::atomic<uint32_t> m_parked {0};
	std::atomic<uint32_t> m_bell {0}; // the encode thread parks on it (Post and the destructor ring it)
	std::atomic<bool>     m_stop {false};
	alignas(64) uint64_t  m_posted  = 0; // commit thread
	uint64_t              m_settled = 0; // commit thread
	alignas(64) uint64_t  m_run     = 0; // encode thread
	Stats                 m_stats;
	std::thread           m_thread;
};

template <typename Log>
void Engine::TakeOldest(Log& log) {
	EXIT_IF(Pending() == 0);
	auto& slot = m_slots[m_settled % MaxJobs];
	WaitDone(slot);
	CommandStream::ReplayState state;
	while (m_ring.ReadPosition() < slot.end) {
		const auto* header = m_ring.Peek();
		EXIT_IF(header == nullptr || header->size == 0);
		if (header->op != CommandStream::Op::Wrap) {
			CommandStream::Replay(*header, log, state, nullptr, nullptr);
		}
		m_ring.Advance(header->size);
	}
	CommandStream::WaitStats stats;
	m_ring.Release(stats);
	slot.done.store(0, std::memory_order_relaxed);
	m_settled++;
}

} // namespace Libs::Graphics::EncodeOffload

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_ENCODEOFFLOAD_H_
