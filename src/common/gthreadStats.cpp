#include "common/gthreadStats.h"

#include "common/hangTrace.h"
#include "common/logging/log.h"

#include <chrono>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace GThreadStats {

namespace Detail {
std::atomic<bool> g_enabled {[] {
	const char* value = std::getenv("KYTY_GTHREAD_STATS");
	return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
}()};
} // namespace Detail

namespace {

std::mutex& RegistryMutex() {
	static std::mutex mutex;
	return mutex;
}

std::vector<Counters*>& Registry() {
	static std::vector<Counters*> registry;
	return registry;
}

thread_local Counters* t_counters = nullptr;
thread_local bool      t_guest    = false;
thread_local bool      t_seen_de  = false;
thread_local bool      t_seen_ue  = false;

} // namespace

Counters& Local() {
	if (t_counters == nullptr) {
		// Never freed: a finished thread's counts stay in the totals.
		auto* counters = new Counters;
		std::lock_guard lock(RegistryMutex());
		Registry().push_back(counters);
		t_counters = counters;
	}
	return *t_counters;
}

void MarkGuestThread() {
	t_guest = true;
}

bool IsGuestThread() {
	return t_guest;
}

void Detail::FpProbeSlow() {
#if defined(__x86_64__) || defined(_M_X64)
	auto&          counters = Local();
	const uint32_t mxcsr    = _mm_getcsr();
	Add(counters.fp_probes);
	constexpr uint32_t DE = 0x02u;
	constexpr uint32_t UE = 0x10u;
	if ((mxcsr & (DE | UE)) != 0) {
		if ((mxcsr & DE) != 0) {
			Add(counters.fp_de);
			if (!t_seen_de) {
				t_seen_de = true;
				Add(counters.fp_thread_de);
			}
		}
		if ((mxcsr & UE) != 0) {
			Add(counters.fp_ue);
			if (!t_seen_ue) {
				t_seen_ue = true;
				Add(counters.fp_thread_ue);
			}
		}
		_mm_setcsr(mxcsr & ~(DE | UE));
	}
#endif
}

Totals Sum() {
	Totals t;
	std::lock_guard lock(RegistryMutex());
	for (const auto* c: Registry()) {
		t.sema_signal += c->sema_signal.load(std::memory_order_relaxed);
		t.sema_signal_multi += c->sema_signal_multi.load(std::memory_order_relaxed);
		t.sema_signal_partial += c->sema_signal_partial.load(std::memory_order_relaxed);
		t.sema_block += c->sema_block.load(std::memory_order_relaxed);
		t.sema_wake += c->sema_wake.load(std::memory_order_relaxed);
		t.sema_spurious += c->sema_spurious.load(std::memory_order_relaxed);
		t.fp_probes += c->fp_probes.load(std::memory_order_relaxed);
		t.fp_de += c->fp_de.load(std::memory_order_relaxed);
		t.fp_ue += c->fp_ue.load(std::memory_order_relaxed);
		t.fp_thread_de += c->fp_thread_de.load(std::memory_order_relaxed);
		t.fp_thread_ue += c->fp_thread_ue.load(std::memory_order_relaxed);
		for (int i = 0; i < 2; i++) {
			t.mem_acq[i] += c->mem_acq[i].load(std::memory_order_relaxed);
			t.mem_contended[i] += c->mem_contended[i].load(std::memory_order_relaxed);
			t.mem_wait_ns[i] += c->mem_wait_ns[i].load(std::memory_order_relaxed);
			t.mem_read_acq[i] += c->mem_read_acq[i].load(std::memory_order_relaxed);
			t.mem_read_contended[i] += c->mem_read_contended[i].load(std::memory_order_relaxed);
			t.mem_read_wait_ns[i] += c->mem_read_wait_ns[i].load(std::memory_order_relaxed);
		}
	}
	return t;
}

void Start() {
	if (!Enabled()) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(1));
				const auto t = Sum();
				Log::WriteToConsoleAndLog(fmt::format(
				    "GThreadStats t_us {}: sema sig {} multi {} partial {} block {} wake {} spurious {} | "
				    "fp probes {} de {} ue {} threads_de {} threads_ue {} | memlock guest {}/{}/{}us "
				    "host {}/{}/{}us | memread guest {}/{}/{}us host {}/{}/{}us\n",
				    HangTrace::NowNs() / 1000u, t.sema_signal, t.sema_signal_multi,
				    t.sema_signal_partial, t.sema_block, t.sema_wake, t.sema_spurious, t.fp_probes,
				    t.fp_de, t.fp_ue, t.fp_thread_de, t.fp_thread_ue, t.mem_acq[0], t.mem_contended[0],
				    t.mem_wait_ns[0] / 1000u, t.mem_acq[1], t.mem_contended[1],
				    t.mem_wait_ns[1] / 1000u, t.mem_read_acq[0], t.mem_read_contended[0],
				    t.mem_read_wait_ns[0] / 1000u, t.mem_read_acq[1], t.mem_read_contended[1],
				    t.mem_read_wait_ns[1] / 1000u));
			}
		}).detach();
	});
}

void Testing::SetEnabled(bool enabled) {
	Detail::g_enabled.store(enabled, std::memory_order_relaxed);
}

} // namespace GThreadStats
