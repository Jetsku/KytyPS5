#include "common/coreBudget.h"

#include "common/liveSwitch.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>

namespace Common {

namespace {

std::string Mask(uint64_t mask) {
	std::string text;
	for (uint32_t bit = 0; bit < 64; bit++) {
		if (((mask >> bit) & 1u) == 0) {
			continue;
		}
		uint32_t last = bit;
		while (last + 1 < 64 && ((mask >> (last + 1)) & 1u) != 0) {
			last++;
		}
		if (!text.empty()) {
			text += ',';
		}
		text += last == bit ? std::to_string(bit) : std::to_string(bit) + "-" + std::to_string(last);
		bit = last;
	}
	return text.empty() ? "none" : text;
}

} // namespace

CoreBudget ComputeCoreBudget(const std::vector<CpuSetInfo>& sets) {
	CoreBudget budget;
	struct Core {
		uint32_t total   = 0;
		uint32_t allowed = 0;
		uint8_t  efficiency = 0;
	};
	std::map<uint32_t, Core>     cores;
	std::map<uint8_t, uint32_t>  per_cache;
	std::set<uint8_t>            classes;
	for (const auto& set: sets) {
		if (set.group != 0 || set.logical >= 64) {
			continue;
		}
		auto& core      = cores[set.core];
		core.total++;
		core.efficiency = std::max(core.efficiency, set.efficiency_class);
		if (!set.allowed) {
			continue;
		}
		core.allowed++;
		budget.logical++;
		budget.mask |= uint64_t {1} << set.logical;
		per_cache[set.cache]++;
		classes.insert(set.efficiency_class);
	}
	if (budget.logical == 0) {
		budget.description = "no allowed processor in group 0";
		return budget;
	}
	budget.top_efficiency = *classes.rbegin();
	budget.hybrid         = classes.size() > 1;
	for (const auto& set: sets) {
		if (set.group == 0 && set.logical < 64 && set.allowed &&
		    set.efficiency_class == budget.top_efficiency) {
			budget.performance_logical++;
			budget.performance_mask |= uint64_t {1} << set.logical;
		}
	}
	for (const auto& [index, core]: cores) {
		if (core.allowed == 0) {
			continue;
		}
		budget.cores++;
		budget.whole_cores += core.allowed == core.total ? 1u : 0u;
		budget.smt_cores += core.allowed >= 2 ? 1u : 0u;
		budget.performance_cores += core.efficiency == budget.top_efficiency ? 1u : 0u;
	}
	budget.caches = static_cast<uint32_t>(per_cache.size());
	for (const auto& [cache, count]: per_cache) {
		budget.largest_cache_logical = std::max(budget.largest_cache_logical, count);
	}
	budget.description = std::to_string(budget.logical) + " logical processors on " +
	                     std::to_string(budget.cores) + " cores (" +
	                     std::to_string(budget.smt_cores) + " with SMT, " +
	                     std::to_string(budget.whole_cores) + " whole), " +
	                     std::to_string(budget.caches) + (budget.caches == 1 ? " L3" : " L3s") +
	                     ", logical " + Mask(budget.mask);
	if (budget.hybrid) {
		budget.description += "; hybrid: " + std::to_string(budget.performance_cores) +
		                      " fastest-class cores (logical " + Mask(budget.performance_mask) + ")";
	}
	return budget;
}

int64_t ParseTriState(const char* value) {
	if (value == nullptr || *value == '\0') {
		return -1;
	}
	return value[0] == '0' && value[1] == '\0' ? 0 : 1;
}

namespace {

std::array<std::atomic<void (*)(bool)>, 8> g_listeners {};

void NotifyProfile(int64_t /*previous*/, int64_t value) {
	for (auto& slot: g_listeners) {
		if (auto* listener = slot.load(std::memory_order_acquire); listener != nullptr) {
			listener(value != 0);
		}
	}
}

Live::Switch& ProfileSwitch() {
	static Live::Switch profile("KYTY_CORE_BUDGET_PROFILE", Live::ParseDefaultOff, NotifyProfile);
	return profile;
}

Live::Switch& BudgetSwitch() {
	static Live::Switch budget("KYTY_CORE_BUDGET", ParseTriState);
	return budget;
}

// Registered before main as the header asks (the function-local statics above make the order safe
// for callers during static initialisation).
[[maybe_unused]] const bool g_registered = [] {
	(void)ProfileSwitch();
	(void)BudgetSwitch();
	return true;
}();

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

std::mutex            g_mutex;
CoreBudget            g_budget;     // under g_mutex
std::string           g_logged;     // the description last logged
std::atomic<int64_t>  g_next_ms {0}; // next re-read
std::atomic<uint32_t> g_logical {0};

void ReadLocked(bool log) {
	uint64_t affinity = 0;
	auto     budget   = ComputeCoreBudget(ReadProcessCpuSets(&affinity));
	g_logical.store(budget.logical, std::memory_order_relaxed);
	if (log && budget.description != g_logged) {
		g_logged = budget.description;
		std::printf("Kyty core budget (KYTY_CORE_BUDGET): %s\n", budget.description.c_str());
		std::fflush(stdout);
	}
	g_budget = std::move(budget);
}

} // namespace

bool EffectiveSwitch(int64_t value) {
	return EffectiveSwitch(value, CoreBudgetProfileOn());
}

bool CoreBudgetProfileOn() {
	return ProfileSwitch().On();
}

bool CoreBudgetEnabled() {
	return EffectiveSwitch(BudgetSwitch().Get());
}

void OnCoreBudgetProfileChange(void (*listener)(bool on)) {
	for (auto& slot: g_listeners) {
		void (*none)(bool) = nullptr;
		if (slot.compare_exchange_strong(none, listener, std::memory_order_acq_rel)) {
			return;
		}
	}
}

CoreBudget CurrentCoreBudget(bool force) {
	const auto now  = NowMs();
	auto       next = g_next_ms.load(std::memory_order_acquire);
	std::lock_guard lock(g_mutex);
	if (force || now >= next || g_logical.load(std::memory_order_relaxed) == 0) {
		g_next_ms.store(now + 1000, std::memory_order_release);
		ReadLocked(CoreBudgetEnabled());
	}
	return g_budget;
}

uint32_t CoreBudgetLogical() {
	const auto now = NowMs();
	if (now >= g_next_ms.load(std::memory_order_acquire) ||
	    g_logical.load(std::memory_order_relaxed) == 0) {
		(void)CurrentCoreBudget();
	}
	return g_logical.load(std::memory_order_relaxed);
}

} // namespace Common
