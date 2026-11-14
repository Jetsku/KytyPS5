#include "common/hleWaitStats.h"

#include "common/liveSwitch.h"

#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace HleWait {

namespace {

const bool g_on = [] {
	const char* value = std::getenv("KYTY_HLE_WAIT_STATS");
	return value != nullptr && value[0] == '1';
}();

constexpr const char* kNames[Count] = {"yield",    "usleep",   "nanosleep", "sema",      "eventflag", "eq_flip",
                                       "eq_vblank", "eq_vo_other", "eq_graphics", "eq_user", "eq_other", "eq_none",
                                       "cond",     "cond_timed", "mutex",   "rw_read",   "rw_write",  "agc_suspend",
                                       "agc_submit", "flip_submit"};

struct PerThread {
	uint32_t              tid = 0;
	std::atomic<uint64_t> ticks[Count] {};
	std::atomic<uint64_t> calls[Count] {};
};

std::mutex& Mutex() {
	static std::mutex mutex;
	return mutex;
}
std::vector<PerThread*>& Threads() {
	static std::vector<PerThread*> threads;
	return threads;
}

std::string Lines(uint64_t seq) {
	std::string out;
	const auto  tsc = Tsc();
	std::lock_guard lock(Mutex());
	for (const auto* t: Threads()) {
		std::string row;
		for (uint32_t i = 0; i < Count; i++) {
			const auto c = t->calls[i].load(std::memory_order_relaxed);
			if (c != 0) {
				row += fmt::format(" {} {}/{}", kNames[i], t->ticks[i].load(std::memory_order_relaxed), c);
			}
		}
		if (!row.empty()) {
			out += fmt::format("HleWait seq {} tsc {} tid {}:{}\n", seq, tsc, t->tid, row);
		}
	}
	return out;
}

const bool g_registered = [] {
	if (g_on) {
		Live::SetExtraLinesProvider(&Lines);
	}
	return true;
}();

PerThread* Local() {
	thread_local PerThread* local = nullptr;
	if (local == nullptr) {
		local = new PerThread; // never freed: a finished thread keeps its totals
#if defined(_WIN32)
		local->tid = static_cast<uint32_t>(::GetCurrentThreadId());
#endif
		std::lock_guard lock(Mutex());
		Threads().push_back(local);
	}
	return local;
}

} // namespace

bool Enabled() noexcept {
	return g_on;
}

void Add(Id id, uint64_t ticks) noexcept {
	auto* t = Local();
	t->ticks[id].store(t->ticks[id].load(std::memory_order_relaxed) + ticks, std::memory_order_relaxed);
	t->calls[id].store(t->calls[id].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

} // namespace HleWait
