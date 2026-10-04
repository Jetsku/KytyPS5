#include "graphics/guest_gpu/command_processor/cpSequencer.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::CpSeq {

namespace {

uint64_t EnvUnsigned(const char* name, uint64_t fallback, uint64_t low, uint64_t high) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	const auto parsed = std::strtoull(value, nullptr, 10);
	return parsed < low ? low : (parsed > high ? high : parsed);
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void Relax() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

} // namespace

uint32_t SnapshotCount() {
	static const auto count =
	    static_cast<uint32_t>(EnvUnsigned("KYTY_CP_SEQ_SNAPSHOTS", 64, 4, 4096));
	return count;
}

uint64_t SequencerSpinNs() {
	static const auto ns = EnvUnsigned("KYTY_CP_SEQ_SPIN_US", 200, 0, 1000000) * 1000u;
	return ns;
}

uint64_t ResolverSpinNs() {
	static const auto ns = EnvUnsigned("KYTY_CP_SEQ_RESOLVER_SPIN_US", 50, 0, 1000000) * 1000u;
	return ns;
}

Sequencer::Sequencer(CommandProcessor& processor)
    : m_processor(processor), m_snapshots(SnapshotCount()), m_spin_ns(SequencerSpinNs()) {}

Sequencer::~Sequencer() {
	Stop();
}

void Sequencer::Start() {
	EXIT_IF(m_thread.joinable());
	m_thread = std::thread([this] { Run(); });
}

void Sequencer::Stop() {
	{
		std::lock_guard lock(m_intake_mutex);
		m_stop = true;
	}
	m_stopping.store(true, std::memory_order_seq_cst);
	m_intake_ready.notify_all();
	Bump();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

void Sequencer::Admit(const Intake& intake) {
	{
		std::lock_guard lock(m_intake_mutex);
		m_intake.push_back(intake);
	}
	m_intake_ready.notify_one();
}

void Sequencer::Bump() noexcept {
	m_progress.fetch_add(1, std::memory_order_seq_cst);
	if (m_parked.load(std::memory_order_seq_cst) != 0) {
		m_progress.notify_all();
	}
}

void Sequencer::NoteStarted(uint64_t submission) {
	m_started.store(submission, std::memory_order_seq_cst);
	Bump();
}

void Sequencer::Answer(uint64_t op_sequence, uint64_t value) {
	m_answer_value = value;
	m_answered.store(op_sequence + 1u, std::memory_order_seq_cst);
	Bump();
}

void Sequencer::NoteExecuted(uint64_t next_op) {
	// Only a parked sequencer needs the bump (a spinning one reads the words directly). The
	// seq_cst store keeps the load of m_parked after it: either the sequencer's re-check after
	// announcing itself sees the new value, or this thread sees the announcement.
	m_executed.store(next_op, std::memory_order_seq_cst);
	if (m_parked.load(std::memory_order_seq_cst) != 0 &&
	    next_op >= m_wake_at.load(std::memory_order_seq_cst)) {
		Bump();
	}
}

void Sequencer::NoteHandoffDone(uint64_t submission) {
	m_handoff_done.store(submission, std::memory_order_seq_cst);
	Bump();
}

bool Sequencer::WaitSlow(const ReadyRef& ready, uint64_t wake_at, uint64_t rewake_at) {
	Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::CpSeqSequencerWait);
	auto                      start = NowNs();
	for (uint32_t spins = 0;; spins++) {
		if (ready()) {
			return true;
		}
		if (Stopping()) {
			return false;
		}
		if ((spins & 63u) != 63u || NowNs() - start < m_spin_ns) {
			Relax();
			continue;
		}
		// Park: set the threshold, announce, re-check, sleep on the progress word. A resolver
		// notification after the announcement changes the word, so the wait returns.
		m_wake_at.store(wake_at, std::memory_order_seq_cst);
		const auto observed = m_progress.load(std::memory_order_seq_cst);
		m_parked.fetch_add(1, std::memory_order_seq_cst);
		if (!ready() && !Stopping()) {
			HangWatchdog::Scope wait("sequencer-progress", reinterpret_cast<uint64_t>(this),
			                         wake_at, observed, 0, rewake_at);
			m_progress.wait(observed, std::memory_order_seq_cst);
		}
		m_parked.fetch_sub(1, std::memory_order_seq_cst);
		// Spin again after a wake (it is early, e.g. before a lockstep op's answer); a further
		// park uses the later threshold.
		wake_at = rewake_at;
		start   = NowNs();
	}
}

uint64_t Sequencer::AwaitAnswer(uint64_t op_sequence) {
	HangWatchdog::Scope wait("sequencer-answer", reinterpret_cast<uint64_t>(this), op_sequence + 1,
	                         HangWatchdog::Enabled() ? m_answered.load() : 0);
	Profiler::CountFrameEvent(Profiler::FrameEvent::CpSeqBarriers);
	// Pre-wake: a parked sequencer is woken this many ops before the lockstep op, so it spins
	// again when the answer comes (the resolver has nothing else to execute until then).
	constexpr uint64_t PreWakeOps = 8;
	(void)Wait(
	    [this, op_sequence] {
		    return m_answered.load(std::memory_order_seq_cst) == op_sequence + 1u;
	    },
	    op_sequence > PreWakeOps ? op_sequence - PreWakeOps : 0u, UINT64_MAX);
	return m_answer_value;
}

void Sequencer::Run() {
	KYTY_PROFILER_THREAD("CpSequencer");
	// The resolver waits for it at every ordering point, like a CP waits for its fetcher.
	Common::RaiseCurrentThreadPriority();
#if defined(_WIN32)
	if (const auto* ideal = std::getenv("KYTY_CP_SEQ_IDEAL_CPU"); ideal != nullptr) {
		const auto       cpu = static_cast<uint32_t>(std::strtoul(ideal, nullptr, 10));
		PROCESSOR_NUMBER number {};
		number.Group  = static_cast<WORD>(cpu / 64u);
		number.Number = static_cast<BYTE>(cpu % 64u);
		(void)SetThreadIdealProcessorEx(GetCurrentThread(), &number, nullptr);
	}
#endif
	for (;;) {
		Intake intake;
		{
			std::unique_lock    lock(m_intake_mutex);
			HangWatchdog::Scope wait("sequencer-intake", reinterpret_cast<uint64_t>(this));
			m_intake_ready.wait(lock, [this] { return m_stop || !m_intake.empty(); });
			if (m_intake.empty()) {
				return;
			}
			intake = m_intake.front();
			m_intake.pop_front();
		}
		HangWatchdog::SetCpContext(0, intake.sequence);
		const bool running = m_processor.SequenceSubmission(intake);
		HangWatchdog::SetCpContext(UINT32_MAX, 0);
		if (!running) {
			return; // stopping
		}
	}
}

} // namespace Libs::Graphics::CpSeq
