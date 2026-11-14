#include "graphics/host_gpu/renderer/pipeline/stagePrepWorker.h"

#include "common/assert.h"
#include "common/helperPriority.h"
#include "common/liveSwitch.h"
#include "common/profiler.h"
#include "graphics/host_gpu/gpuReadDelegate.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

void CpuRelax() {
#if defined(_M_X64) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// KYTY_STAGE_PREP_PARALLEL (live, default on; =0 serial): the pixel stage of a draw is prepared on
// the helper while the vertex stage is prepared on the calling thread. Live so that the stage walks
// can be A/B'd in one launch (senaxx 816198d2b measured its own version as a loss): off, Get() returns
// null and callers take the serial path; the helper, once started, stays parked.
Live::Switch g_stage_prep_parallel("KYTY_STAGE_PREP_PARALLEL", Live::ParseDefaultOn);

bool ParallelEnabled() {
	return g_stage_prep_parallel.On();
}

uint64_t SpinNs() {
	const auto* value = std::getenv("KYTY_STAGE_PREP_SPIN_US");
	if (value == nullptr || *value == '\0') {
		return 250'000;
	}
	return std::strtoull(value, nullptr, 10) * 1000u;
}

} // namespace

StagePrepWorker* StagePrepWorker::Get() {
	if (!ParallelEnabled()) {
		return nullptr;
	}
	// Leaked on purpose: the helper is detached and may still be parked at process exit.
	static auto* const worker = new StagePrepWorker();
	return worker;
}

StagePrepWorker::StagePrepWorker(): m_spin_ns(SpinNs()) {
	std::thread([this] { Run(); }).detach();
}

bool StagePrepWorker::TryFork(Function function, void* context) {
	const auto posted = m_posted.load(std::memory_order_relaxed);
	if (m_completed.load(std::memory_order_acquire) != posted) {
		return false;
	}
	if (m_sleeping.load(std::memory_order_seq_cst)) {
		// Waking takes far longer than one stage; wake it for the next draw instead.
		m_signal.fetch_add(1, std::memory_order_seq_cst);
		m_signal.notify_one();
		return false;
	}
	m_function = function;
	m_context  = context;
	// Paired with the helper's seq_cst store of m_sleeping and reload of m_posted: either the
	// helper sees this job before parking, or this thread sees it parking and wakes it.
	m_posted.store(posted + 1u, std::memory_order_seq_cst);
	if (m_sleeping.load(std::memory_order_seq_cst)) {
		m_signal.fetch_add(1, std::memory_order_seq_cst);
		m_signal.notify_one();
	}
	return true;
}

void StagePrepWorker::Join() {
	const auto posted = m_posted.load(std::memory_order_relaxed);
	if (m_completed.load(std::memory_order_acquire) == posted) {
		return;
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::StagePrepJoin);
	while (m_completed.load(std::memory_order_acquire) != posted) {
		CpuRelax();
	}
}

void StagePrepWorker::Run() {
	Profiler::SetThreadName("DrawPrep#0");
	uint64_t seen = 0;
	for (;;) {
		Common::HelperPriority::Check(0); // KYTY_HELPER_PRIORITY (measurement)
		auto     idle_start = NowNs();
		uint32_t spins      = 0;
		uint64_t posted     = 0;
		for (;;) {
			posted = m_posted.load(std::memory_order_acquire);
			if (posted != seen) {
				break;
			}
			CpuRelax();
			if ((++spins & 1023u) != 0u || NowNs() - idle_start < m_spin_ns) {
				continue;
			}
			const auto signal = m_signal.load(std::memory_order_seq_cst);
			m_sleeping.store(true, std::memory_order_seq_cst);
			if (m_posted.load(std::memory_order_seq_cst) == seen) {
				m_signal.wait(signal, std::memory_order_seq_cst);
			}
			m_sleeping.store(false, std::memory_order_seq_cst);
			idle_start = NowNs();
		}
		EXIT_IF(posted != seen + 1u);
		seen = posted;
		{
			Profiler::ScopedFrameWait  job_time(Profiler::FrameWait::StagePrepHelper);
			GpuReadDelegate::Scope delegate;
			m_function(m_context);
		}
		m_completed.store(seen, std::memory_order_release);
	}
}

} // namespace Libs::Graphics
