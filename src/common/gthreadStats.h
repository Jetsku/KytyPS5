#ifndef EMULATOR_SRC_COMMON_GTHREADSTATS_H_
#define EMULATOR_SRC_COMMON_GTHREADSTATS_H_

#include <atomic>
#include <cstdint>

// KYTY_GTHREAD_STATS=1 (default off; diagnostics): counters on the game's own threads, printed
// cumulatively once per second (console and log) on the hang trace's time base, so a measured
// window (two Live: marker lines) reads per-frame rates as differences:
//   GThreadStats t_us 183456789: sema sig 1200 multi 30 partial 25 block 900 wake 950 spurious 40 |
//   fp probes 52000 de 0 ue 3 threads_de 0 threads_ue 1 | memlock guest 7000/12/850us host 300/0/0us
// sema: kernel semaphore signals, signals that found two or more waiters, of those the signals that
// left at least one waiter not runnable ("partial"), waits that blocked, wake-ups of a blocked
// waiter (the condition variable returned before its timeout) and wake-ups that found the waiter
// still not runnable (it went back to sleep). fp: probes of MXCSR at sync entry points on guest
// threads; de / ue: probes that found the denormal-operand / underflow flag set since the thread's
// previous probe (the flags are cleared after reading). memlock: acquisitions / contended
// acquisitions / time spent waiting for the kernel memory operation lock (the recursive operation
// mutex plus, at a thread's outermost hold, the state lock exclusively), by guest and other threads.
// memread: the same for the shared state lock the guest's memory queries take.
// Off: every hook is one relaxed load of a bool.

namespace GThreadStats {

namespace Detail {
extern std::atomic<bool> g_enabled;
void FpProbeSlow();
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

// Starts the reporter thread when KYTY_GTHREAD_STATS is set (once).
void Start();

// The current thread runs guest code (guest threads and the main thread).
void MarkGuestThread();
[[nodiscard]] bool IsGuestThread();

// MXCSR denormal / underflow flags since the thread's previous probe.
inline void FpProbe() {
	if (Enabled()) {
		Detail::FpProbeSlow();
	}
}

struct Counters {
	std::atomic<uint64_t> sema_signal {0};
	std::atomic<uint64_t> sema_signal_multi {0};
	std::atomic<uint64_t> sema_signal_partial {0};
	std::atomic<uint64_t> sema_block {0};
	std::atomic<uint64_t> sema_wake {0};
	std::atomic<uint64_t> sema_spurious {0};
	std::atomic<uint64_t> fp_probes {0};
	std::atomic<uint64_t> fp_de {0};
	std::atomic<uint64_t> fp_ue {0};
	std::atomic<uint64_t> fp_thread_de {0};
	std::atomic<uint64_t> fp_thread_ue {0};
	std::atomic<uint64_t> mem_acq[2] {};
	std::atomic<uint64_t> mem_contended[2] {};
	std::atomic<uint64_t> mem_wait_ns[2] {};
	std::atomic<uint64_t> mem_read_acq[2] {};
	std::atomic<uint64_t> mem_read_contended[2] {};
	std::atomic<uint64_t> mem_read_wait_ns[2] {};
};

// This thread's counters (registered on first use; summed by the reporter).
[[nodiscard]] Counters& Local();

inline void Add(std::atomic<uint64_t>& counter, uint64_t value = 1) {
	counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
}

// Sums over all threads (tests and the reporter).
struct Totals {
	uint64_t sema_signal = 0, sema_signal_multi = 0, sema_signal_partial = 0, sema_block = 0,
	         sema_wake = 0, sema_spurious = 0, fp_probes = 0, fp_de = 0, fp_ue = 0,
	         fp_thread_de = 0, fp_thread_ue = 0, mem_acq[2] {}, mem_contended[2] {},
	         mem_wait_ns[2] {}, mem_read_acq[2] {}, mem_read_contended[2] {}, mem_read_wait_ns[2] {};
};
[[nodiscard]] Totals Sum();

namespace Testing {
void SetEnabled(bool enabled);
} // namespace Testing

} // namespace GThreadStats

#endif // EMULATOR_SRC_COMMON_GTHREADSTATS_H_
