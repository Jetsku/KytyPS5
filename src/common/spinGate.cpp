#include "common/spinGate.h"

#include "common/coreBudget.h"
#include "common/cpuPlacement.h"
#include "common/liveSwitch.h"
#include "common/profiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h> // IWYU pragma: keep
#endif

namespace Common::SpinGate {

static std::atomic<bool>     g_open {true};
static std::atomic<int64_t>  g_next_ms {0};
static std::atomic<uint64_t> g_shuts {0};

bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_SPIN_GATE");
		const bool  set   = value != nullptr && value[0] == '1';
		if (set) {
			std::printf("SpinGate: helper threads spin for work only while 3+ CPUs idle (KYTY_SPIN_GATE)\n");
		}
		return set;
	}();
	return on;
}

#if defined(_WIN32)
// The refreshing thread only (the g_next_ms exchange orders consecutive refreshes).
static uint64_t g_last_idle  = 0;
static uint64_t g_last_total = 0;

static uint64_t FileTimeValue(const FILETIME& time) {
	return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
}

static void Refresh() {
	FILETIME idle {};
	FILETIME kernel {};
	FILETIME user {};
	if (GetSystemTimes(&idle, &kernel, &user) == 0) {
		return;
	}
	const uint64_t idle_value  = FileTimeValue(idle);
	const uint64_t total_value = FileTimeValue(kernel) + FileTimeValue(user); // kernel includes idle
	const uint64_t idle_delta  = idle_value - g_last_idle;
	const uint64_t total_delta = total_value - g_last_total;
	const bool     first       = g_last_total == 0;
	g_last_idle                = idle_value;
	g_last_total               = total_value;
	if (first || total_delta == 0) {
		return;
	}
	static const uint32_t cpus = [] {
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		return info.dwNumberOfProcessors == 0 ? 1u
		                                      : static_cast<uint32_t>(info.dwNumberOfProcessors);
	}();
	// Average idle CPUs over the interval: the idle share of all CPUs' time times their count.
	const double idle_cpus =
	    static_cast<double>(idle_delta) * cpus / static_cast<double>(total_delta);
	const bool open = g_open.load(std::memory_order_relaxed);
	if (open && idle_cpus < 3.0) {
		g_open.store(false, std::memory_order_relaxed);
		g_shuts.fetch_add(1, std::memory_order_relaxed);
	} else if (!open && idle_cpus >= 5.0) {
		g_open.store(true, std::memory_order_relaxed);
	}
}
#endif

bool SpinAllowed() {
	if (!Enabled()) {
		return true;
	}
#if defined(_WIN32)
	const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                        std::chrono::steady_clock::now().time_since_epoch())
	                        .count();
	int64_t next = g_next_ms.load(std::memory_order_acquire);
	if (now >= next &&
	    g_next_ms.compare_exchange_strong(next, now + 250, std::memory_order_acq_rel)) {
		Refresh();
	}
#endif
	return g_open.load(std::memory_order_relaxed);
}

uint64_t Shuts() {
	return g_shuts.load(std::memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------
// KYTY_SPIN_BUDGET

const char* SiteName(Site site) {
	switch (site) {
		case Site::Sequencer: return "sequencer";
		case Site::Recorder: return "recorder";
		case Site::CommitHead: return "commit-head";
		case Site::RecorderDrain: return "recorder-drain";
		case Site::Resolver: return "resolver";
		case Site::Worker: return "worker";
		default: return "?";
	}
}

bool NextBudgetGateOpen(bool open, const GateInput& input) {
	if (input.budget_logical == 0) {
		return true;
	}
	if (open) {
		return input.busy + input.spinning + input.margin <= input.budget_logical;
	}
	return input.busy + input.idle_spinners + input.margin + 1u <= input.budget_logical;
}

Demand ClassifyThreads(const ThreadSample* samples, uint32_t count, uint64_t interval_tsc,
                       double busy_share) {
	Demand demand;
	if (interval_tsc == 0) {
		return demand;
	}
	const auto interval = static_cast<double>(interval_tsc);
	for (uint32_t i = 0; i < count; i++) {
		const auto& sample = samples[i];
		const auto  spin   = sample.registered ? std::min(sample.spin, sample.cycles) : 0u;
		const auto  work   = sample.cycles - spin;
		demand.busy_cores += static_cast<double>(work) / interval;
		demand.spin_cores += static_cast<double>(spin) / interval;
		if (static_cast<double>(work) >= busy_share * interval) {
			demand.busy++;
		} else if (sample.registered) {
			demand.idle_spinners++;
			if (static_cast<double>(spin) >= 0.05 * interval) {
				demand.spinning++;
			}
		}
	}
	return demand;
}

void AddUnmeasured(Demand& demand, uint64_t process_cycles, uint64_t measured_cycles,
                   uint64_t interval_tsc) {
	if (interval_tsc == 0 || process_cycles <= measured_cycles) {
		return;
	}
	const auto cores =
	    static_cast<double>(process_cycles - measured_cycles) / static_cast<double>(interval_tsc);
	demand.busy_cores += cores;
	demand.busy += static_cast<uint32_t>(cores + 0.5);
}

namespace {

constexpr auto     SiteCount = static_cast<size_t>(Site::Count);
constexpr uint32_t MaxSlots  = 64;

uint64_t EnvU64(const char* name, uint64_t fallback, uint64_t low, uint64_t high) {
	const auto* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return std::clamp<uint64_t>(std::strtoull(value, nullptr, 10), low, high);
}

Live::Switch g_budget_switch("KYTY_SPIN_BUDGET", ParseTriState);

int64_t ParseUs(const char* value, int64_t fallback) {
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	return static_cast<int64_t>(std::min<unsigned long long>(std::strtoull(value, nullptr, 10), 1000000ull));
}

// KYTY_SPIN_BUDGET_SITES (live, default 0x3F = all six): the sites the closed gate caps, bit per
// Site (1 sequencer, 2 recorder, 4 commit-head, 8 recorder-drain, 16 resolver, 32 worker).
Live::Switch g_sites_switch("KYTY_SPIN_BUDGET_SITES", [](const char* value) -> int64_t {
	return value == nullptr || *value == '\0' ? 0x3F
	                                          : static_cast<int64_t>(std::strtoull(value, nullptr, 0) & 0x3Fu);
});
// KYTY_SPIN_BUDGET_SHORT_US (live, default 2): the closed gate's spin. KYTY_SPIN_BUDGET_PIPE_SHORT_US
// (live, default: the same): the sequencer's and recorder's (their parks are wake-ups on the
// commit thread's path).
Live::Switch g_short_switch("KYTY_SPIN_BUDGET_SHORT_US",
                            [](const char* value) -> int64_t { return ParseUs(value, 2); });
Live::Switch g_pipe_short_switch("KYTY_SPIN_BUDGET_PIPE_SHORT_US",
                                 [](const char* value) -> int64_t { return ParseUs(value, -1); });

uint64_t ShortNs(Site site) {
	if (site == Site::Sequencer || site == Site::Recorder) {
		if (const auto pipe = g_pipe_short_switch.Get(); pipe >= 0) {
			return static_cast<uint64_t>(pipe) * 1000u;
		}
	}
	return static_cast<uint64_t>(g_short_switch.Get()) * 1000u;
}

// One registered spinner thread. Every counter has one writer (the thread), so updates are a
// relaxed load and store, no locked instruction; the monitor reads them.
struct alignas(64) ThreadSlot {
	std::atomic<uint32_t> tid {0};
	std::atomic<uint32_t> sites {0}; // bit per site it waited at
	std::atomic<void*>    handle {nullptr}; // a kept thread handle (the monitor's cycle reads)
	std::array<std::atomic<uint64_t>, SiteCount> waits {};
	std::array<std::atomic<uint64_t>, SiteCount> spin {};
	std::array<std::atomic<uint64_t>, SiteCount> parks {};
};

std::array<ThreadSlot, MaxSlots> g_slots {};
std::atomic<uint32_t>            g_slot_count {0};
// Threads beyond MaxSlots share these (locked adds).
ThreadSlot g_overflow;

thread_local ThreadSlot* t_slot = nullptr;

uint32_t CurrentTid() {
#if defined(_WIN32)
	return static_cast<uint32_t>(GetCurrentThreadId());
#else
	return 0;
#endif
}

void StartMonitor();

ThreadSlot& Slot() {
	if (t_slot != nullptr) [[likely]] {
		return *t_slot;
	}
	const auto index = g_slot_count.fetch_add(1, std::memory_order_acq_rel);
	if (index < MaxSlots) {
		g_slots[index].tid.store(CurrentTid(), std::memory_order_release);
#if defined(_WIN32)
		HANDLE handle = nullptr;
		if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
		                    THREAD_QUERY_LIMITED_INFORMATION, FALSE, 0) != 0) {
			g_slots[index].handle.store(handle, std::memory_order_release);
		}
#endif
		t_slot = &g_slots[index];
	} else {
		t_slot = &g_overflow;
	}
	StartMonitor();
	return *t_slot;
}

void Add(std::atomic<uint64_t>& counter, uint64_t value, bool shared) {
	if (shared) {
		counter.fetch_add(value, std::memory_order_relaxed);
	} else {
		counter.store(counter.load(std::memory_order_relaxed) + value, std::memory_order_relaxed);
	}
}

std::atomic<bool> g_closed {false};

} // namespace

bool BudgetEnabled() {
	return EffectiveSwitch(g_budget_switch.Get());
}

bool BudgetAvailable() {
	static const bool available = [] {
		const auto set = [](const char* name) {
			const auto* value = std::getenv(name);
			return value != nullptr && *value != '\0';
		};
		return set("KYTY_SPIN_BUDGET") || set("KYTY_CORE_BUDGET_PROFILE");
	}();
	return available;
}

bool BudgetClosed() {
	return g_closed.load(std::memory_order_relaxed) && BudgetEnabled();
}

uint64_t SpinNs(Site site, uint64_t wanted_ns) {
	auto&      slot   = Slot();
	const auto index  = static_cast<size_t>(site);
	const bool shared = &slot == &g_overflow;
	Add(slot.waits[index], 1, shared);
	const auto bit = 1u << index;
	if ((slot.sites.load(std::memory_order_relaxed) & bit) == 0) {
		slot.sites.fetch_or(bit, std::memory_order_relaxed);
	}
	if (Enabled() && !SpinAllowed()) {
		return 0;
	}
	if (BudgetClosed() && ((g_sites_switch.Get() >> index) & 1) != 0) {
		return std::min(wanted_ns, ShortNs(site));
	}
	return wanted_ns;
}

void NoteSpin(Site site, uint64_t tsc) {
	auto& slot = Slot();
	Add(slot.spin[static_cast<size_t>(site)], tsc, &slot == &g_overflow);
}

void NotePark(Site site, uint64_t count) {
	auto& slot = Slot();
	Add(slot.parks[static_cast<size_t>(site)], count, &slot == &g_overflow);
}

bool GiveWay() {
#if defined(_WIN32)
	return SwitchToThread() != 0;
#else
	std::this_thread::yield();
	return false;
#endif
}

namespace {

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

struct SiteTotals {
	uint64_t waits = 0, spin = 0, parks = 0;
};

std::array<SiteTotals, SiteCount> SumSites() {
	std::array<SiteTotals, SiteCount> totals {};
	const auto count = std::min(g_slot_count.load(std::memory_order_acquire), MaxSlots);
	const auto add   = [&totals](const ThreadSlot& slot) {
		for (size_t site = 0; site < SiteCount; site++) {
			totals[site].waits += slot.waits[site].load(std::memory_order_relaxed);
			totals[site].spin += slot.spin[site].load(std::memory_order_relaxed);
			totals[site].parks += slot.parks[site].load(std::memory_order_relaxed);
		}
	};
	for (uint32_t i = 0; i < count; i++) {
		add(g_slots[i]);
	}
	add(g_overflow);
	return totals;
}

#if defined(_WIN32)
// The scan (monitor thread only): the registered spinner threads (their kept handles), the guest
// and cold threads the placement keeps handles of, and the process total. No system-wide thread
// snapshot: a few dozen QueryThreadCycleTime calls and one QueryProcessCycleTime.
struct ThreadWatch {
	struct Entry {
		uint64_t cycles = 0;
		uint64_t spin   = 0;
		bool     seen   = false;
	};
	std::unordered_map<uint32_t, Entry> threads;
	std::vector<ThreadSample>           samples;
	uint64_t                            process_cycles = 0;
	uint64_t                            measured       = 0; // this interval's measured cycles
	uint64_t                            process_delta  = 0;
	bool                                primed         = false;

	ThreadSample Take(uint32_t tid, uint64_t cycles, uint64_t spin, bool registered) {
		auto [it, added] = threads.try_emplace(tid);
		auto& entry      = it->second;
		ThreadSample sample;
		if (!added) {
			sample.cycles = cycles >= entry.cycles ? cycles - entry.cycles : 0;
			sample.spin   = spin >= entry.spin ? spin - entry.spin : 0;
		}
		sample.registered = registered;
		entry.cycles      = cycles;
		entry.spin        = spin;
		entry.seen        = true;
		return sample;
	}

	// One interval's samples (cycles and spin since the previous call; a thread's first sample is 0).
	void Sample() {
		samples.clear();
		measured = 0;
		for (auto& [tid, entry]: threads) {
			entry.seen = false;
		}
		const auto count = std::min(g_slot_count.load(std::memory_order_acquire), MaxSlots);
		for (uint32_t i = 0; i < count; i++) {
			auto* handle = static_cast<HANDLE>(g_slots[i].handle.load(std::memory_order_acquire));
			ULONG64 cycles = 0;
			if (handle == nullptr || QueryThreadCycleTime(handle, &cycles) == 0) {
				continue;
			}
			uint64_t spin = 0;
			for (size_t site = 0; site < SiteCount; site++) {
				spin += g_slots[i].spin[site].load(std::memory_order_relaxed);
			}
			samples.push_back(Take(g_slots[i].tid.load(std::memory_order_acquire), cycles, spin, true));
		}
		VisitPlacedThreads(
		    [](void* context, uint32_t tid, void* handle, bool /*guest*/) {
			    auto&   self   = *static_cast<ThreadWatch*>(context);
			    ULONG64 cycles = 0;
			    if (handle == nullptr || QueryThreadCycleTime(static_cast<HANDLE>(handle), &cycles) == 0) {
				    return;
			    }
			    if (const auto it = self.threads.find(tid); it != self.threads.end() && it->second.seen) {
				    return; // a registered spinner as well
			    }
			    self.samples.push_back(self.Take(tid, cycles, 0, false));
		    },
		    this);
		for (auto it = threads.begin(); it != threads.end();) {
			it = it->second.seen ? std::next(it) : threads.erase(it);
		}
		for (const auto& sample: samples) {
			measured += sample.cycles;
		}
		ULONG64 process = 0;
		(void)QueryProcessCycleTime(GetCurrentProcess(), &process);
		process_delta  = primed && process >= process_cycles ? process - process_cycles : 0;
		process_cycles = process;
		if (!primed) {
			measured = 0;
		}
		primed = true;
	}
};
#endif

std::atomic<bool> g_monitor_started {false};

void MonitorLoop() {
	Profiler::SetThreadName("Spin budget monitor");
	// Below normal priority on the cold / general processors: never the CP's reserved processor,
	// and never preempting it.
	PlaceCurrentThreadCold();
	LowerCurrentThreadPriority();
	const auto interval_ms = EnvU64("KYTY_SPIN_BUDGET_INTERVAL_MS", 250, 20, 10000);
	const auto period_s    = EnvU64("KYTY_SPIN_STATS_PERIOD_S", 10, 0, 1000000);
	const auto busy_share  = static_cast<double>(EnvU64("KYTY_SPIN_BUDGET_BUSY_PCT", 25, 1, 100)) / 100.0;
	const auto margin      = static_cast<uint32_t>(EnvU64("KYTY_SPIN_BUDGET_MARGIN", 0, 0, 64));
	GateConfig config;
	config.window         = static_cast<uint32_t>(EnvU64("KYTY_SPIN_BUDGET_WINDOW", 4, 1, 32));
	config.hold_closed_ms = EnvU64("KYTY_SPIN_BUDGET_HOLD_CLOSED_MS", 2000, 0, 600000);
	config.hold_open_ms   = EnvU64("KYTY_SPIN_BUDGET_HOLD_OPEN_MS", 1000, 0, 600000);
#if defined(_WIN32)
	ThreadWatch watch;
#endif
	std::optional<GateFilter> filter;
	uint64_t closes_seen = 0; // the filter's closes at the last line
	bool     watching    = false;
	uint64_t closed_ns   = 0;
	auto     start_ns    = NowNs();
	auto     last_ns     = start_ns;
	auto     last_tsc    = Tsc();
	auto     period_ns0  = last_ns;
	auto     period_tsc0 = last_tsc;
	auto     last_totals = SumSites();
	uint64_t scan_ns     = 0;
	uint64_t scan_max_ns = 0;
	uint32_t scans       = 0;
	uint32_t threads     = 0;
	Demand   demand;
	GateInput input;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
		const auto now_ns  = NowNs();
		const auto now_tsc = Tsc();
		const auto now_ms  = static_cast<uint64_t>(now_ns - start_ns) / 1'000'000u;
		const bool on      = BudgetEnabled();
		if (g_closed.load(std::memory_order_relaxed)) {
			closed_ns += static_cast<uint64_t>(now_ns - last_ns);
		}
		if (on) {
			const auto budget = CurrentCoreBudget();
			const auto t0     = NowNs();
#if defined(_WIN32)
			watch.Sample();
			threads = static_cast<uint32_t>(watch.samples.size());
			if (watching) {
				const auto interval = now_tsc - last_tsc;
				demand = ClassifyThreads(watch.samples.data(), threads, interval, busy_share);
				AddUnmeasured(demand, watch.process_delta, watch.measured, interval);
			}
#endif
			const auto cost = static_cast<uint64_t>(NowNs() - t0);
			scan_ns += cost;
			scan_max_ns = std::max(scan_max_ns, cost);
			scans++;
			if (!filter) {
				filter.emplace(config);
			}
			if (watching) {
				input.budget_logical = budget.logical;
				input.busy           = demand.busy;
				input.spinning       = demand.spinning;
				input.idle_spinners  = demand.idle_spinners;
				input.margin         = margin;
				g_closed.store(!filter->Step(now_ms, input), std::memory_order_relaxed);
			}
			watching = true;
		} else {
			watching             = false;
			input.budget_logical = CurrentCoreBudget().logical;
			filter.reset();
			closes_seen = 0;
			g_closed.store(false, std::memory_order_relaxed);
		}
		last_ns  = now_ns;
		last_tsc = now_tsc;
		if (period_s == 0 || static_cast<uint64_t>(now_ns - period_ns0) < period_s * 1'000'000'000ull) {
			continue;
		}
		const auto   totals  = SumSites();
		const double seconds = static_cast<double>(now_ns - period_ns0) / 1e9;
		const double tsc_ms  = static_cast<double>(now_tsc - period_tsc0) / (seconds * 1000.0);
		std::string  sites;
		for (size_t site = 0; site < SiteCount; site++) {
			const auto waits = totals[site].waits - last_totals[site].waits;
			const auto spin  = totals[site].spin - last_totals[site].spin;
			const auto parks = totals[site].parks - last_totals[site].parks;
			char       part[160];
			std::snprintf(part, sizeof(part), " | %s: waits %.0f/s spin %.1f ms/s parks %.0f/s",
			              SiteName(static_cast<Site>(site)), static_cast<double>(waits) / seconds,
			              tsc_ms > 0.0 ? static_cast<double>(spin) / tsc_ms / seconds : 0.0,
			              static_cast<double>(parks) / seconds);
			sites += part;
		}
		const auto closes = filter ? filter->Closes() - closes_seen : 0;
		closes_seen       = filter ? filter->Closes() : 0;
		const auto mean   = filter ? filter->Mean() : input;
		std::printf("SpinBudget %.0fs: KYTY_SPIN_BUDGET=%d profile=%d legacy_gate=%d sites=0x%llx short=%lld/%lldus, gate %s "
		            "(closed %.1f%% of the time, %llu closes), budget %u lps, demand busy %.1f spinning %.1f "
		            "idle-spinners %.1f (window mean; busy %.2f spin %.2f cores), scan %u threads %.0f us avg "
		            "%.0f us max (%.3f ms/s)%s\n",
		            seconds, on ? 1 : 0, CoreBudgetProfileOn() ? 1 : 0, Enabled() ? 1 : 0,
		            static_cast<unsigned long long>(g_sites_switch.Get()), static_cast<long long>(g_short_switch.Get()),
		            static_cast<long long>(g_pipe_short_switch.Get()),
		            g_closed.load(std::memory_order_relaxed) ? "closed" : "open",
		            100.0 * static_cast<double>(closed_ns) / (seconds * 1e9),
		            static_cast<unsigned long long>(closes), input.budget_logical, mean.busy,
		            mean.spinning, mean.idle_spinners, demand.busy_cores, demand.spin_cores, threads,
		            scans != 0 ? static_cast<double>(scan_ns) / scans / 1000.0 : 0.0,
		            static_cast<double>(scan_max_ns) / 1000.0,
		            static_cast<double>(scan_ns) / 1e6 / seconds, sites.c_str());
		std::fflush(stdout);
		last_totals = totals;
		period_ns0  = now_ns;
		period_tsc0 = now_tsc;
		closed_ns   = 0;
		scan_ns     = 0;
		scan_max_ns = 0;
		scans       = 0;
	}
}

void StartMonitor() {
	if (g_monitor_started.load(std::memory_order_relaxed) ||
	    g_monitor_started.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	std::thread(MonitorLoop).detach();
}

} // namespace

} // namespace Common::SpinGate
