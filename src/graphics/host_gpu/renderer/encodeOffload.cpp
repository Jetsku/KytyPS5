#include "graphics/host_gpu/renderer/encodeOffload.h"

#include "common/assert.h"
#include "common/profiler.h"

#include <chrono>

#if defined(_MSC_VER) || defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define KYTY_ENC_PAUSE() _mm_pause()
#else
#define KYTY_ENC_PAUSE() ((void)0)
#endif

namespace Libs::Graphics::EncodeOffload {

Engine::Engine() {
	m_thread = std::thread([this] { Loop(); });
}

Engine::~Engine() {
	m_stop.store(true, std::memory_order_seq_cst);
	m_bell.fetch_add(1, std::memory_order_seq_cst);
	m_bell.notify_all();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

uint64_t Engine::Now() const {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count());
}

void Engine::Configure(const CommandStream::Encoder::Options& options) {
	CommandStream::Encoder::Options own;
	own.verify        = options.verify;
	own.barrier_sites = options.barrier_sites;
	own.all_sites     = false;
	own.policy        = options.policy;
	if (m_encoder != nullptr) {
		const auto& current = m_encoder->GetOptions();
		if (current.verify == own.verify && current.barrier_sites == own.barrier_sites) {
			return;
		}
		EXIT_IF(Pending() != 0);
	}
	m_encoder = std::make_unique<CommandStream::Encoder>(m_ring, own);
}

bool Engine::Post(void* job, RunFn run) {
	EXIT_IF(m_encoder == nullptr || job == nullptr || run == nullptr);
	if (Pending() >= MaxJobs) {
		return false;
	}
	auto& slot = m_slots[m_posted % MaxJobs];
	slot.job   = job;
	slot.run   = run;
	slot.end   = 0;
	m_posted++;
	m_stats.jobs.fetch_add(1, std::memory_order_relaxed);
	m_posted_shared.store(m_posted, std::memory_order_seq_cst);
	if (m_parked.load(std::memory_order_seq_cst) != 0) {
		m_bell.fetch_add(1, std::memory_order_seq_cst);
		m_bell.notify_one();
	}
	return true;
}

void Engine::Loop() {
	Profiler::SetThreadName("EncodeOffload");
	for (;;) {
		auto posted = m_posted_shared.load(std::memory_order_acquire);
		if (posted == m_run) {
			// Spin about 50 us (a continuation is about 2 us), then park.
			const auto start = Now();
			while ((posted = m_posted_shared.load(std::memory_order_acquire)) == m_run &&
			       !m_stop.load(std::memory_order_relaxed) && Now() - start < 50000) {
				KYTY_ENC_PAUSE();
			}
			if (posted == m_run) {
				if (m_stop.load(std::memory_order_acquire)) {
					return;
				}
				const auto bell = m_bell.load(std::memory_order_seq_cst);
				m_parked.store(1, std::memory_order_seq_cst);
				posted = m_posted_shared.load(std::memory_order_seq_cst);
				if (posted == m_run && !m_stop.load(std::memory_order_seq_cst)) {
					m_stats.parks.fetch_add(1, std::memory_order_relaxed);
					m_bell.wait(bell, std::memory_order_acquire);
				}
				m_parked.store(0, std::memory_order_relaxed);
				continue;
			}
		}
		auto&      slot  = m_slots[m_run % MaxJobs];
		const bool timed = (m_run & 63u) == 0;
		const auto start = timed ? Now() : 0;
		slot.run(slot.job, *m_encoder);
		m_ring.Publish();
		slot.end = m_ring.WritePosition();
		if (timed) {
			m_stats.worker_ns.fetch_add(Now() - start, std::memory_order_relaxed);
			m_stats.worker_timed.fetch_add(1, std::memory_order_relaxed);
		}
		slot.done.store(1, std::memory_order_release);
		m_run++;
	}
}

void Engine::WaitDone(Slot& slot) {
	if (slot.done.load(std::memory_order_acquire) != 0) {
		return;
	}
	m_stats.waits.fetch_add(1, std::memory_order_relaxed);
	uint32_t spins = 0;
	while (slot.done.load(std::memory_order_acquire) == 0) {
		if (++spins < 4096) {
			KYTY_ENC_PAUSE();
		} else {
			std::this_thread::yield();
		}
	}
}

uint64_t Engine::Settle(CommandStream::Encoder& target) {
	if (Pending() == 0) {
		return 0;
	}
	const auto settles = m_stats.settles.fetch_add(1, std::memory_order_relaxed);
	const bool timed   = (settles & 63u) == 0;
	uint64_t   wait_ns = 0;
	uint64_t   bytes   = 0;
	const auto start   = timed ? Now() : 0;
	const auto jobs    = Pending();
	for (; m_settled < m_posted; m_settled++) {
		auto& slot = m_slots[m_settled % MaxJobs];
		if (timed && slot.done.load(std::memory_order_acquire) == 0) {
			const auto wait_start = Now();
			WaitDone(slot);
			wait_ns += Now() - wait_start;
		} else {
			WaitDone(slot);
		}
		bytes += target.Splice(m_ring, slot.end);
		slot.done.store(0, std::memory_order_relaxed);
	}
	if (timed) {
		const auto total = Now() - start;
		m_stats.wait_ns.fetch_add(wait_ns, std::memory_order_relaxed);
		m_stats.splice_ns.fetch_add(total - wait_ns, std::memory_order_relaxed);
		m_stats.timed.fetch_add(1, std::memory_order_relaxed);
	}
	m_stats.spliced_bytes.fetch_add(bytes, std::memory_order_relaxed);
	m_stats.spliced_jobs.fetch_add(jobs, std::memory_order_relaxed);
	return bytes;
}

} // namespace Libs::Graphics::EncodeOffload
