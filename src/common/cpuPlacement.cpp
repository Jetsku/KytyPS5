#include "common/cpuPlacement.h"

#include "common/coreBudget.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/profiler.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_set>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// After windows.h.
#include <tlhelp32.h>
#endif

namespace Common {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

std::string MaskText(uint64_t mask) {
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
		text +=
		    last == bit ? std::to_string(bit) : std::to_string(bit) + "-" + std::to_string(last);
		bit = last;
	}
	return text.empty() ? "none" : text;
}

} // namespace

CpuLayout ComputeCpuLayout(const std::vector<CpuSetInfo>& sets, CpuReserveMode mode,
                           int forced_cp_logical, uint32_t min_general, uint64_t preferred_mask,
                           const ReserveRules& rules) {
	CpuLayout layout;
	if (mode == CpuReserveMode::Off) {
		layout.description = "off";
		return layout;
	}
	// KYTY_CPU_RESERVE_SCALE: the budget the rules read, and the fastest efficiency class.
	const bool scaled      = rules.scaled;
	const auto budget      = scaled ? ComputeCoreBudget(sets) : CoreBudget {};
	const auto budget_cores = rules.budget_cores != 0 ? rules.budget_cores : budget.cores;
	const auto budget_lps   = rules.budget_logical != 0 ? rules.budget_logical : budget.logical;
	// Hybrid: only the fastest class counts for the cache choice and as candidates.
	const bool only_top = scaled && budget.hybrid;
	struct Core {
		int                            index      = 0;
		uint8_t                        cache      = 0;
		uint8_t                        preference = 0;
		uint8_t                        efficiency = 0;
		bool                           whole      = true; // every logical processor allowed
		bool                           any        = false; // some logical processor allowed
		bool                           has_zero   = false;
		std::vector<const CpuSetInfo*> processors;
	};
	std::map<int, Core>         cores;
	std::map<uint8_t, uint32_t> allowed_per_cache;
	std::map<uint8_t, bool>     preferred_cache; // holds processors of `preferred_mask`
	for (const auto& set: sets) {
		if (set.group != 0 || set.logical >= 64) {
			continue;
		}
		auto& core      = cores[set.core];
		core.index      = set.core;
		core.cache      = set.cache;
		core.preference = std::max(core.preference, set.scheduling_class);
		core.efficiency = std::max(core.efficiency, set.efficiency_class);
		core.whole      = core.whole && set.allowed;
		core.any        = core.any || set.allowed;
		core.has_zero   = core.has_zero || set.logical == 0;
		core.processors.push_back(&set);
		if (set.allowed && (!only_top || set.efficiency_class == budget.top_efficiency)) {
			allowed_per_cache[set.cache]++;
		}
		if (((preferred_mask >> set.logical) & 1u) != 0) {
			preferred_cache[set.cache] = true;
		}
	}
	if (allowed_per_cache.empty()) {
		layout.description = "no allowed processor in group 0";
		return layout;
	}
	// The cache holding most allowed processors (ties: the preferred one, the V-cache CCD, then the
	// lower index).
	uint8_t    cache     = 0;
	uint32_t   best      = 0;
	const auto preferred = [&](uint8_t index) {
		const auto it = preferred_cache.find(index);
		return it != preferred_cache.end() && it->second;
	};
	for (const auto& [index, count]: allowed_per_cache) {
		if (count > best || (count == best && preferred(index) && !preferred(cache))) {
			best  = count;
			cache = index;
		}
	}
	auto needed = mode == CpuReserveMode::Stages       ? size_t {3}
	              : mode == CpuReserveMode::CpRecorder ? size_t {2}
	                                                   : size_t {1};
	// KYTY_CPU_RESERVE_SCALE: whole cores only while the budget has enough cores for the game's
	// threads (cp+recorder and stages need one more each, otherwise they fall back to cp); else one
	// logical processor for the CP, or nothing.
	bool one_logical = false;
	std::string scale_note;
	if (scaled) {
		const auto whole_ok = [&](size_t count) {
			return budget_cores >= rules.whole_min_cores + static_cast<uint32_t>(count - 1u);
		};
		if (needed > 1 && !whole_ok(needed)) {
			scale_note = "; " + std::to_string(budget_cores) + " cores in the budget: CP core only";
			needed     = 1;
		}
		if (!whole_ok(1)) {
			if (budget_lps < rules.logical_min) {
				layout.description = "only " + std::to_string(budget_lps) + " logical processors on " +
				                     std::to_string(budget_cores) +
				                     " cores in the budget: nothing reserved (KYTY_CPU_RESERVE_SCALE)";
				return layout;
			}
			one_logical = true;
		}
	}
	const auto top_class = [&](const Core& core) {
		return !only_top || core.efficiency == budget.top_efficiency;
	};
	std::vector<const Core*> candidates;
	for (const auto& [index, core]: cores) {
		if ((one_logical ? core.any : core.whole) && core.cache == cache && top_class(core)) {
			candidates.push_back(&core);
		}
	}
	const auto order = [scaled](std::vector<const Core*>& list) {
		std::sort(list.begin(), list.end(), [](const Core* a, const Core* b) {
			if (a->has_zero != b->has_zero) {
				return !a->has_zero; // the core of processor 0 (interrupts) last
			}
			if (a->preference != b->preference) {
				return a->preference > b->preference;
			}
			return a->index < b->index;
		});
		if (scaled && !list.empty()) {
			// KYTY_CPU_RESERVE_SCALE: the CP's core by the CPU's preferred-core ranking first (the
			// core of processor 0 only loses ties); the other reserved cores keep the order above.
			auto best = list.begin();
			for (auto it = list.begin(); it != list.end(); ++it) {
				if ((*it)->preference > (*best)->preference) {
					best = it;
				}
			}
			std::rotate(list.begin(), best, best + 1);
		}
	};
	order(candidates);
	if (scaled && !one_logical && candidates.size() < needed && needed == 1 &&
	    budget_lps >= rules.logical_min) {
		// No whole core (an odd affinity mask): one logical processor still fits.
		candidates.clear();
		for (const auto& [index, core]: cores) {
			if (core.any && core.cache == cache && top_class(core)) {
				candidates.push_back(&core);
			}
		}
		order(candidates);
		one_logical = true;
	}
	if (one_logical) {
		// The CP's logical processor: the forced one, else the best candidate core's first allowed
		// processor with the highest scheduling class.
		const CpuSetInfo* chosen = nullptr;
		if (forced_cp_logical >= 0) {
			for (const auto& set: sets) {
				if (set.group == 0 && set.logical == forced_cp_logical && set.allowed) {
					chosen = &set;
				}
			}
			if (chosen == nullptr) {
				layout.description = "KYTY_CPU_RESERVE_CORE=" + std::to_string(forced_cp_logical) +
				                     " is not an allowed processor";
				return layout;
			}
		} else if (!candidates.empty()) {
			for (const auto* processor: candidates[0]->processors) {
				if (processor->allowed &&
				    (chosen == nullptr || processor->scheduling_class > chosen->scheduling_class)) {
					chosen = processor;
				}
			}
		}
		if (chosen == nullptr) {
			layout.description = "no allowed processor in the largest cache";
			return layout;
		}
		for (const auto& set: sets) {
			if (set.group != 0 || set.logical >= 64 || !set.allowed) {
				continue;
			}
			const auto bit = uint64_t {1} << set.logical;
			if (&set == chosen) {
				layout.cp_ids.push_back(set.id);
				layout.cp_mask |= bit;
			} else {
				layout.general_ids.push_back(set.id);
				layout.general_mask |= bit;
			}
		}
		layout.reserved    = true;
		layout.cp_core     = chosen->core;
		layout.description = "CP on logical " + MaskText(layout.cp_mask) + " alone (core " +
		                     std::to_string(chosen->core) + ", preference " +
		                     std::to_string(chosen->scheduling_class) + "; " +
		                     std::to_string(budget_cores) + " cores / " + std::to_string(budget_lps) +
		                     " logical in the budget: one logical processor, KYTY_CPU_RESERVE_SCALE)" +
		                     ", other threads on " + MaskText(layout.general_mask) + scale_note;
		return layout;
	}
	if (forced_cp_logical >= 0) {
		const Core* forced = nullptr;
		for (const auto& [index, core]: cores) {
			for (const auto* processor: core.processors) {
				if (processor->logical == forced_cp_logical) {
					forced = &core;
				}
			}
		}
		if (forced == nullptr || !forced->whole) {
			layout.description = "KYTY_CPU_RESERVE_CORE=" + std::to_string(forced_cp_logical) +
			                     " is not a core whose processors are all allowed";
			return layout;
		}
		candidates.erase(std::remove(candidates.begin(), candidates.end(), forced),
		                 candidates.end());
		candidates.insert(candidates.begin(), forced);
	}
	if (candidates.size() < needed) {
		layout.description =
		    "fewer than " + std::to_string(needed) + " whole allowed cores in the largest cache";
		return layout;
	}
	const Core* cp       = candidates[0];
	const Core* recorder  = needed >= 2 ? candidates[1] : nullptr;
	const Core* sequencer = needed >= 3 ? candidates[2] : nullptr;
	uint32_t    general  = 0;
	for (const auto& set: sets) {
		if (set.group != 0 || set.logical >= 64 || !set.allowed) {
			continue;
		}
		const auto bit = uint64_t {1} << set.logical;
		if (set.core == cp->index) {
			layout.cp_ids.push_back(set.id);
			layout.cp_mask |= bit;
		} else if (recorder != nullptr && set.core == recorder->index) {
			layout.recorder_ids.push_back(set.id);
			layout.recorder_mask |= bit;
		} else if (sequencer != nullptr && set.core == sequencer->index) {
			layout.sequencer_ids.push_back(set.id);
			layout.sequencer_mask |= bit;
		} else {
			layout.general_ids.push_back(set.id);
			layout.general_mask |= bit;
			general++;
		}
	}
	if (general < min_general) {
		layout             = {};
		layout.description = "only " + std::to_string(general) +
		                     " logical processors would remain for the other threads";
		return layout;
	}
	layout.reserved      = true;
	layout.cp_core       = cp->index;
	layout.recorder_core = recorder != nullptr ? recorder->index : -1;
	layout.sequencer_core = sequencer != nullptr ? sequencer->index : -1;
	layout.description   = "CP on logical " + MaskText(layout.cp_mask) + " (core " +
	                       std::to_string(cp->index) + ", preference " +
	                       std::to_string(cp->preference) + ")";
	if (recorder != nullptr) {
		layout.description += ", recorder on logical " + MaskText(layout.recorder_mask) +
		                      " (core " + std::to_string(recorder->index) + ")";
	}
	if (sequencer != nullptr) {
		layout.description += ", sequencer on logical " + MaskText(layout.sequencer_mask) +
		                      " (core " + std::to_string(sequencer->index) + ")";
	}
	layout.description += ", other threads on " + MaskText(layout.general_mask) + scale_note;
	return layout;
}

namespace {

std::string MiB(uint64_t bytes) {
	const auto mib = bytes / (1024u * 1024u);
	return mib != 0 ? std::to_string(mib) + " MiB" : std::to_string(bytes / 1024u) + " KiB";
}

} // namespace

VCacheTopology DetectVCache(const std::vector<L3Cache>& caches) {
	VCacheTopology topology;
	// One entry per distinct processor mask (the largest size if one is reported twice).
	std::map<uint64_t, uint64_t> by_mask;
	for (const auto& cache: caches) {
		if (cache.group != 0 || cache.mask == 0) {
			continue;
		}
		auto& bytes = by_mask[cache.mask];
		bytes       = std::max(bytes, cache.bytes);
	}
	std::vector<std::pair<uint64_t, uint64_t>> groups(by_mask.begin(), by_mask.end());
	// Largest first; equal sizes by their lowest processor.
	std::sort(groups.begin(), groups.end(), [](const auto& a, const auto& b) {
		if (a.second != b.second) {
			return a.second > b.second;
		}
		return std::countr_zero(a.first) < std::countr_zero(b.first);
	});
	topology.groups = static_cast<uint32_t>(groups.size());
	std::string sizes;
	for (const auto& [mask, bytes]: groups) {
		sizes += (sizes.empty() ? "" : ", ") + MiB(bytes) + " on logical " + MaskText(mask);
	}
	if (groups.empty()) {
		topology.description = "no L3 cache reported";
		return topology;
	}
	if (groups.size() == 1) {
		topology.description = "one L3 (" + sizes + ")";
		return topology;
	}
	const auto largest = groups[0].second;
	const auto second  = groups[1].second;
	// Clearly larger: at least 1.5 times (96 against 32 MiB is 3 times).
	if (largest == 0 || second * 3u > largest * 2u) {
		topology.description = "L3 groups of similar size (" + sizes + ")";
		return topology;
	}
	topology.dual         = true;
	topology.vcache_mask  = groups[0].first;
	topology.vcache_bytes = largest;
	topology.other_bytes  = second;
	for (size_t i = 1; i < groups.size(); i++) {
		topology.other_mask |= groups[i].first;
	}
	topology.description = "dual-CCD X3D (L3 " + sizes + ")";
	return topology;
}

VCacheDecision DecideVCachePlacement(VCacheMode mode, const VCacheTopology& topology,
                                     bool external_rule, bool cpu_sets) {
	VCacheDecision decision;
	if (mode == VCacheMode::Off) {
		decision.reason = "off (KYTY_VCACHE_PLACEMENT=0)";
		return decision;
	}
	if (!topology.dual) {
		decision.reason = "not a dual-CCD X3D: " + topology.description + "; no change";
		return decision;
	}
	if (mode == VCacheMode::Auto && cpu_sets) {
		decision.reason = topology.description + ", KYTY_CPU_SETS is set: no change";
		return decision;
	}
	if (mode == VCacheMode::Auto && external_rule) {
		decision.reason = topology.description +
		                  ", an external CPU-set/affinity rule is active (Process Lasso or similar): "
		                  "left to it, no change";
		return decision;
	}
	decision.engage = true;
	decision.reason = topology.description;
	return decision;
}

CpuLayout ComputeVCacheLayout(const std::vector<CpuSetInfo>& sets, CpuReserveMode mode,
                              int forced_cp_logical, const VCacheTopology& topology,
                              uint32_t min_general, const ReserveRules& rules) {
	CpuLayout layout;
	// KYTY_CPU_RESERVE_SCALE reads the whole process's budget (guest threads use both CCDs).
	auto scaled_rules = rules;
	if (rules.scaled && rules.budget_cores == 0) {
		const auto budget           = ComputeCoreBudget(sets);
		scaled_rules.budget_cores   = budget.cores;
		scaled_rules.budget_logical = budget.logical;
	}
	if (!topology.dual) {
		layout.description = "not a dual-CCD X3D";
		return layout;
	}
	const auto in = [](uint64_t mask, const CpuSetInfo& set) {
		return set.group == 0 && set.logical < 64 && ((mask >> set.logical) & 1u) != 0;
	};
	std::vector<CpuSetInfo> hot       = sets;
	uint32_t                hot_count = 0;
	std::vector<uint32_t>   cold_ids;
	uint64_t                cold_mask = 0;
	for (auto& set: hot) {
		if (set.allowed && in(topology.other_mask, set)) {
			cold_ids.push_back(set.id);
			cold_mask |= uint64_t {1} << set.logical;
		}
		set.allowed = set.allowed && in(topology.vcache_mask, set);
		hot_count += set.allowed ? 1u : 0u;
	}
	if (cold_ids.empty()) {
		layout.description = "no allowed processor outside the V-cache CCD";
		return layout;
	}
	if (hot_count < min_general) {
		layout.description =
		    "only " + std::to_string(hot_count) + " allowed processors on the V-cache CCD";
		return layout;
	}
	if (mode != CpuReserveMode::Off) {
		layout = ComputeCpuLayout(hot, mode, forced_cp_logical, min_general, 0, scaled_rules);
		if (!layout.reserved) {
			layout.description =
			    "KYTY_CPU_RESERVE does not fit in the V-cache CCD (" + layout.description + ")";
			return layout;
		}
	} else {
		for (const auto& set: hot) {
			if (set.allowed) {
				layout.general_ids.push_back(set.id);
				layout.general_mask |= uint64_t {1} << set.logical;
			}
		}
		layout.description = "no reserved core";
	}
	layout.vcache      = true;
	layout.cold_ids    = std::move(cold_ids);
	layout.cold_mask   = cold_mask;
	layout.description = "hot threads on logical " + MaskText(layout.general_mask) + " (" +
	                     layout.description + "), cold threads on " + MaskText(cold_mask);
	return layout;
}

VCacheMode ParseVCacheMode(const char* text, bool* valid) {
	if (valid != nullptr) {
		*valid = true;
	}
	if (text == nullptr || *text == '\0' || std::strcmp(text, "auto") == 0) {
		return VCacheMode::Auto;
	}
	if (std::strcmp(text, "0") == 0 || std::strcmp(text, "off") == 0) {
		return VCacheMode::Off;
	}
	if (std::strcmp(text, "1") == 0 || std::strcmp(text, "on") == 0) {
		return VCacheMode::On;
	}
	if (valid != nullptr) {
		*valid = false;
	}
	return VCacheMode::Auto;
}

VCacheMode GetVCacheMode() {
	static const VCacheMode mode = [] {
		const auto* value = EnvValue("KYTY_VCACHE_PLACEMENT");
		bool        valid = true;
		const auto  mode  = ParseVCacheMode(value, &valid);
		if (!valid) {
			std::printf("KYTY_VCACHE_PLACEMENT must be auto, 0 or 1 (got '%s'): auto\n", value);
			std::fflush(stdout);
		}
		return mode;
	}();
	return mode;
}

uint64_t GeneralPoolAffinity(uint64_t affinity, const CpuLayout& layout) {
	if (!layout.reserved || layout.general_mask == 0 || (affinity & layout.general_mask) != 0) {
		return affinity;
	}
	return layout.general_mask;
}

CpuReserveMode ParseCpuReserveMode(const char* text, bool* valid) {
	if (valid != nullptr) {
		*valid = true;
	}
	if (text == nullptr || *text == '\0' || std::strcmp(text, "off") == 0 ||
	    std::strcmp(text, "0") == 0) {
		return CpuReserveMode::Off;
	}
	if (std::strcmp(text, "cp") == 0 || std::strcmp(text, "1") == 0) {
		return CpuReserveMode::Cp;
	}
	if (std::strcmp(text, "cp+recorder") == 0 || std::strcmp(text, "2") == 0) {
		return CpuReserveMode::CpRecorder;
	}
	if (std::strcmp(text, "stages") == 0 || std::strcmp(text, "3") == 0) {
		return CpuReserveMode::Stages;
	}
	if (valid != nullptr) {
		*valid = false;
	}
	return CpuReserveMode::Off;
}

CpuReserveMode GetCpuReserveMode() {
	static const CpuReserveMode mode = [] {
		const auto* value = EnvValue("KYTY_CPU_RESERVE");
		bool        valid = true;
		const auto  mode  = ParseCpuReserveMode(value, &valid);
		if (!valid) {
			std::printf("KYTY_CPU_RESERVE must be off, cp, cp+recorder or stages (got '%s'): off\n", value);
		}
		return mode;
	}();
	return mode;
}

namespace {
// KYTY_CPU_RESERVE_SCALE (live): a change (or a profile change while it is unset) makes the monitor
// apply the layout again within a second.
std::atomic<bool> g_rules_changed {false};
Live::Switch      g_reserve_scale("KYTY_CPU_RESERVE_SCALE", ParseTriState,
                                  [](int64_t, int64_t) { g_rules_changed.store(true); });
[[maybe_unused]] const bool g_scale_listener = [] {
	OnCoreBudgetProfileChange([](bool) { g_rules_changed.store(true); });
	return true;
}();

uint32_t EnvRule(const char* name, uint32_t fallback) {
	const auto* value = EnvValue(name);
	return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : fallback;
}
} // namespace

ReserveRules CurrentReserveRules() {
	static const uint32_t whole_min   = EnvRule("KYTY_CPU_RESERVE_WHOLE_MIN_CORES", 8);
	static const uint32_t logical_min = EnvRule("KYTY_CPU_RESERVE_LOGICAL_MIN", 8);
	ReserveRules rules;
	rules.scaled          = EffectiveSwitch(g_reserve_scale.Get());
	rules.whole_min_cores = whole_min;
	rules.logical_min     = logical_min;
	return rules;
}

bool RepinEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_CPU_RESERVE_REPIN");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint64_t ParseLogicalMask(const char* text, bool* valid) {
	const auto fail = [valid] {
		if (valid != nullptr) {
			*valid = false;
		}
		return uint64_t {0};
	};
	if (valid != nullptr) {
		*valid = true;
	}
	if (text == nullptr || *text == '\0') {
		return fail();
	}
	uint64_t    mask = 0;
	const char* p    = text;
	for (;;) {
		char*      end   = nullptr;
		const auto first = std::strtoul(p, &end, 10);
		if (end == p) {
			return fail();
		}
		auto last = first;
		p         = end;
		if (*p == '-') {
			p++;
			last = std::strtoul(p, &end, 10);
			if (end == p) {
				return fail();
			}
			p = end;
		}
		if (last < first || last > 63) {
			return fail();
		}
		for (auto bit = first; bit <= last; bit++) {
			mask |= uint64_t {1} << bit;
		}
		if (*p == '\0') {
			return mask;
		}
		if (*p != ',') {
			return fail();
		}
		p++;
	}
}

uint64_t CpuSetsMask() {
	static const uint64_t mask = [] {
		const auto* value = EnvValue("KYTY_CPU_SETS");
		if (value == nullptr) {
			return uint64_t {0};
		}
		bool       valid  = true;
		const auto parsed = ParseLogicalMask(value, &valid);
		if (!valid) {
			std::printf("KYTY_CPU_SETS must list logical processors 0-63 like 16-31 or 0-7,16-23 "
			            "(got '%s'): unset\n",
			            value);
			std::fflush(stdout);
		}
		return parsed;
	}();
	return mask;
}

bool ReassertEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_CPU_RESERVE_REASSERT");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

namespace {

#if defined(_WIN32)

// The process default CPU sets now (sorted CPU set ids; empty: none assigned).
std::vector<ULONG> ProcessDefaultCpuSets() {
	std::vector<ULONG> ids(256);
	ULONG              count = 0;
	if (GetProcessDefaultCpuSets(GetCurrentProcess(), ids.data(), static_cast<ULONG>(ids.size()),
	                             &count) == 0) {
		count = 0;
	}
	ids.resize(count);
	std::sort(ids.begin(), ids.end());
	return ids;
}

// Every CPU set of the system, `allowed` when in the process affinity mask, not allocated to
// another process, and in the constraint: KYTY_CPU_SETS when set, otherwise the external default
// CPU sets (if any: those found at startup, or another process's that replaced ours later).
std::vector<CpuSetInfo> ReadCpuSets(uint64_t& affinity, const std::vector<ULONG>& external) {
	const auto              restriction = CpuSetsMask();
	std::vector<CpuSetInfo> sets;
	ULONG                   length = 0;
	(void)GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(length);
	if (length == 0 ||
	    GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
	                               length, &length, GetCurrentProcess(), 0) == 0) {
		return sets;
	}
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	affinity = GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)
	               ? static_cast<uint64_t>(process_mask)
	               : ~uint64_t {0};
	for (ULONG offset = 0; offset < length;) {
		const auto* info =
		    reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
		if (info->Size == 0) {
			break;
		}
		if (info->Type == CpuSetInformation) {
			const auto& cpu = info->CpuSet;
			CpuSetInfo  set;
			set.id               = cpu.Id;
			set.group            = cpu.Group;
			set.logical          = cpu.LogicalProcessorIndex;
			set.core             = cpu.CoreIndex;
			set.cache            = cpu.LastLevelCacheIndex;
			set.scheduling_class = cpu.SchedulingClass;
			set.efficiency_class = cpu.EfficiencyClass;
			const bool ours     = cpu.Allocated == 0 || cpu.AllocatedToTargetProcess != 0;
			const bool in_constraint =
			    restriction != 0
			        ? cpu.LogicalProcessorIndex < 64 &&
			              ((restriction >> cpu.LogicalProcessorIndex) & 1u) != 0
			        : external.empty() ||
			              std::find(external.begin(), external.end(), cpu.Id) != external.end();
			set.allowed = cpu.Group == 0 && cpu.LogicalProcessorIndex < 64 &&
			              ((affinity >> cpu.LogicalProcessorIndex) & 1u) != 0 && ours &&
			              in_constraint;
			sets.push_back(set);
		}
		offset += info->Size;
	}
	return sets;
}

// The L3 cache groups (GetLogicalProcessorInformationEx cache relations).
std::vector<L3Cache> ReadL3Caches() {
	std::vector<L3Cache> caches;
	DWORD                length = 0;
	(void)GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
	if (length == 0) {
		return caches;
	}
	std::vector<uint8_t> buffer(length);
	if (GetLogicalProcessorInformationEx(
	        RelationCache, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()),
	        &length) == 0) {
		return caches;
	}
	for (DWORD offset = 0; offset < length;) {
		const auto* info =
		    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);
		if (info->Size == 0) {
			break;
		}
		const auto& cache = info->Cache;
		if (info->Relationship == RelationCache && cache.Level == 3 &&
		    (cache.Type == CacheUnified || cache.Type == CacheData)) {
			// GroupCount is 0 (a reserved field) before Windows 11: one GroupMask.
			const WORD count = cache.GroupCount == 0 ? WORD {1} : cache.GroupCount;
			for (WORD i = 0; i < count; i++) {
				const auto& group = cache.GroupMasks[i];
				caches.push_back({group.Group, static_cast<uint64_t>(group.Mask), cache.CacheSize});
			}
		}
		offset += info->Size;
	}
	return caches;
}

// The process affinity mask is narrower than the system's (a Process Lasso affinity rule, or a
// launcher's start /affinity).
bool AffinityRestricted(uint64_t* process = nullptr) {
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
		return false;
	}
	if (process != nullptr) {
		*process = static_cast<uint64_t>(process_mask);
	}
	return process_mask != system_mask;
}

#endif

struct State {
	std::mutex              mutex;
	bool                    initialized = false;
	uint64_t                affinity    = 0; // the process mask the layout was computed from
	CpuLayout               layout;
	std::vector<CpuSetInfo> sets;
#if defined(_WIN32)
	// The threads placed on reserved cores (their sets are re-applied when the layout changes).
	std::array<HANDLE, static_cast<size_t>(ThreadRole::Count)> threads {};
	HANDLE sequencer = nullptr; // PlaceCurrentThreadSequencer (KYTY_CPU_RESERVE=stages)
	bool applied_default = false; // our process default CPU sets are in effect
	// The constraint without KYTY_CPU_SETS: the default CPU sets found at startup, replaced by
	// another process's when it replaces ours (KYTY_CPU_RESERVE_REASSERT). Restored when nothing
	// is reserved.
	std::vector<ULONG> external;
	// The process default CPU sets this module applied last (sorted), while applied_default.
	std::vector<ULONG> applied;
	// PlaceCurrentThreadCold threads (KYTY_VCACHE_PLACEMENT), forgotten when they exit.
	struct ColdThread {
		DWORD  tid    = 0;
		HANDLE handle = nullptr;
	};
	std::vector<ColdThread> cold;
	// PlaceCurrentThreadGuest threads (guest pthreads and the guest main thread).
	std::vector<ColdThread> guest;
#endif
	// KYTY_VCACHE_PLACEMENT.
	VCacheTopology                        topology;
	bool                                  external_rule = false; // seen once: for good
	bool                                  settled       = false; // auto: the settle time passed
	bool                                  vcache_applied = false;
	std::chrono::steady_clock::time_point started;
	std::string                           vcache_logged; // the decision last logged
	// ScanThreadAffinities: threads already counted (FrameEvents count each once).
	std::mutex                   scan_mutex;
	std::unordered_set<uint32_t> hard_seen;
	std::unordered_set<uint32_t> reserved_seen;
	uint32_t                     logged = 0;
};

// Never destroyed: the monitor thread may still run while static destructors do.
State& GetState() {
	static auto* state = new State;
	return *state;
}

// Sampling state, lock-free.
std::atomic<uint64_t> g_cp_mask {0};          // the reserved CP processors (0: none)
std::atomic<uint64_t> g_recorder_mask {0};    // the reserved recorder processors (0: none)
std::atomic<uint32_t> g_cp_core {UINT32_MAX}; // the CP's last sampled core
std::array<std::array<std::atomic<uint32_t>, PlacementHistogram::Processors>,
           static_cast<size_t>(ThreadRole::Count)>
                                                                          g_histogram {};
std::array<std::atomic<uint32_t>, static_cast<size_t>(ThreadRole::Count)> g_on_cp_core {};
std::atomic<bool>                                                         g_topology_ready {false};
std::array<std::atomic<uint32_t>, 256>                                    g_core_of {};
std::atomic<uint32_t>                                                     g_reasserts {0};

void PublishTopology(const std::vector<CpuSetInfo>& sets) {
	for (auto& core: g_core_of) {
		core.store(UINT32_MAX, std::memory_order_relaxed);
	}
	for (const auto& set: sets) {
		const auto index = static_cast<uint32_t>(set.group) * 64u + set.logical;
		if (index < g_core_of.size()) {
			g_core_of[index].store(static_cast<uint32_t>(set.group) * 256u + set.core,
			                       std::memory_order_relaxed);
		}
	}
	g_topology_ready.store(true, std::memory_order_release);
}

int ForcedCpLogical() {
	const auto* value = EnvValue("KYTY_CPU_RESERVE_CORE");
	return value != nullptr ? static_cast<int>(std::strtol(value, nullptr, 10)) : -1;
}

#if defined(_WIN32)
void ApplyThreadSets(HANDLE thread, const std::vector<uint32_t>& ids) {
	if (thread == nullptr) {
		return;
	}
	std::vector<ULONG> list(ids.begin(), ids.end());
	(void)SetThreadSelectedCpuSets(thread, list.empty() ? nullptr : list.data(),
	                               static_cast<ULONG>(list.size()));
}
#endif

#if defined(_WIN32)
// The logical processors (group 0) of CPU set ids, by the last CPU set table read.
uint64_t LogicalMaskOfIds(const State& state, const std::vector<ULONG>& ids) {
	uint64_t mask = 0;
	for (const auto& set: state.sets) {
		if (set.group == 0 && set.logical < 64 &&
		    std::find(ids.begin(), ids.end(), set.id) != ids.end()) {
			mask |= uint64_t {1} << set.logical;
		}
	}
	return mask;
}
#endif

// KYTY_VCACHE_SETTLE_MS (default 3000): with KYTY_VCACHE_PLACEMENT=auto, how long after startup
// the monitor waits for an external rule (Process Lasso applies its CPU sets about 0.6 s after the
// process starts) before it applies the V-cache placement.
uint32_t VCacheSettleMs() {
	static const uint32_t ms = [] {
		const auto* value = EnvValue("KYTY_VCACHE_SETTLE_MS");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 3000u;
	}();
	return ms;
}

// The monitor watches for the V-cache placement: auto on a dual-CCD X3D with no external rule seen
// yet, or 1 on a dual-CCD X3D.
bool VCacheWatchedLocked(const State& state) {
	const auto mode = GetVCacheMode();
	return state.topology.dual && (mode == VCacheMode::On ||
	                               (mode == VCacheMode::Auto && !state.external_rule));
}

// State mutex held: the V-cache decision now (`layout` set when it applies).
std::string VCacheDecisionLocked(const State& state, CpuLayout* layout) {
	const auto mode     = GetVCacheMode();
	const auto decision = DecideVCachePlacement(mode, state.topology, state.external_rule,
	                                            CpuSetsMask() != 0);
	if (!decision.engage) {
		return decision.reason;
	}
	if (mode == VCacheMode::Auto && !state.settled) {
		return decision.reason + ": waiting " + std::to_string(VCacheSettleMs()) +
		       " ms for an external CPU-set/affinity rule first";
	}
	auto computed = ComputeVCacheLayout(state.sets, GetCpuReserveMode(), ForcedCpLogical(),
	                                    state.topology, kMinGeneralProcessors, CurrentReserveRules());
	if (!computed.vcache) {
		return decision.reason + ": not applied, " + computed.description;
	}
	const auto text = decision.reason + ": " + computed.description;
	if (layout != nullptr) {
		*layout = std::move(computed);
	}
	return text;
}

void LogVCacheLocked(State& state, const char* why, const std::string& text) {
	if (text == state.vcache_logged) {
		return;
	}
	state.vcache_logged = text;
	const auto mode     = GetVCacheMode();
	std::printf("Kyty V-cache placement (KYTY_VCACHE_PLACEMENT=%s, %s): %s\n",
	            mode == VCacheMode::Auto ? "auto" : mode == VCacheMode::On ? "1" : "0", why,
	            text.c_str());
	std::fflush(stdout);
}

// KYTY_VCACHE_GUEST=all|hot|cold (default all): with the V-cache placement, guest threads run on
// every allowed processor outside the reserved cores (all), only on the hot processors of the
// V-cache CCD (hot, the int19.3-int20 behaviour) or only on the other CCD (cold). With hot, Astro
// Bot's game threads shared 14 logical processors with the emulator's hot threads; in
// speedrun_inflator its Draw Geometry thread then fell a frame behind and overflowed its command
// buffer (Bug C, every boot). all: Sky Garden +3 %, sponge_dirty_bath +2 %, pirate island equal
// (7950X3D + RTX 3090, 3 boots each, non-PGO).
enum class GuestPlacement : uint8_t { All, Hot, Cold };

GuestPlacement GetGuestPlacement() {
	static const GuestPlacement mode = [] {
		const auto* value = EnvValue("KYTY_VCACHE_GUEST");
		if (value != nullptr && std::strcmp(value, "hot") == 0) {
			return GuestPlacement::Hot;
		}
		if (value != nullptr && std::strcmp(value, "cold") == 0) {
			return GuestPlacement::Cold;
		}
		return GuestPlacement::All;
	}();
	return mode;
}

bool GuestWide() {
	return GetGuestPlacement() != GuestPlacement::Hot;
}

#if defined(_WIN32)
// State mutex held: the CPU sets of guest threads, empty for the process default.
std::vector<uint32_t> GuestIdsLocked(const State& state) {
	if (!state.layout.vcache || !GuestWide()) {
		return {};
	}
	if (GetGuestPlacement() == GuestPlacement::Cold) {
		return state.layout.cold_ids;
	}
	std::vector<uint32_t> ids = state.layout.general_ids;
	ids.insert(ids.end(), state.layout.cold_ids.begin(), state.layout.cold_ids.end());
	return ids;
}
#endif

// State mutex held: computes the layout for the current process mask and constraint and applies
// it: the process default CPU sets (the general processors of a reservation, the hot processors
// of the V-cache placement, or every allowed processor with only KYTY_CPU_SETS), the reserved
// threads' and the cold threads' selected CPU sets. `keep_default`: another process's default CPU
// sets stay in effect (an external rule appeared; KYTY_CPU_RESERVE_REASSERT is off).
void ApplyLayoutLocked(State& state, const char* why, bool log = true, bool keep_default = false) {
	uint64_t affinity = 0;
#if defined(_WIN32)
	state.sets = ReadCpuSets(affinity, state.external);
#else
	state.sets.clear();
#endif
	state.affinity = affinity;
	PublishTopology(state.sets);
	const auto mode        = GetCpuReserveMode();
	const auto restriction = CpuSetsMask();
	CpuLayout  vcache_layout;
	const auto vcache_text = VCacheDecisionLocked(state, &vcache_layout);
	if (vcache_layout.vcache) {
		state.layout = std::move(vcache_layout);
		if (GetGuestPlacement() == GuestPlacement::All) {
			state.layout.description += "; guest threads on every processor outside the reserved "
			                            "cores (KYTY_VCACHE_GUEST=all)";
		} else if (GetGuestPlacement() == GuestPlacement::Cold) {
			state.layout.description += "; guest threads on the cold processors (KYTY_VCACHE_GUEST=cold)";
		}
	} else {
		// Ties between caches go to the V-cache CCD (measured, not by index).
		state.layout = ComputeCpuLayout(state.sets, mode, ForcedCpLogical(), kMinGeneralProcessors,
		                                state.topology.dual ? state.topology.vcache_mask : 0,
		                                CurrentReserveRules());
	}
	if (log) {
		LogVCacheLocked(state, why, vcache_text);
	}
	const bool active = mode != CpuReserveMode::Off || restriction != 0 || state.layout.vcache;
#if defined(_WIN32)
	if (!active && !state.applied_default && !state.vcache_applied) {
		return;
	}
	std::vector<ULONG> wanted;
	if (state.layout.reserved || state.layout.vcache) {
		wanted.assign(state.layout.general_ids.begin(), state.layout.general_ids.end());
	} else if (restriction != 0) {
		uint64_t mask = 0;
		for (const auto& set: state.sets) {
			if (set.allowed) {
				wanted.push_back(set.id);
				mask |= uint64_t {1} << set.logical;
			}
		}
		const auto text = "every thread on logical " + MaskText(mask) + " (KYTY_CPU_SETS)";
		state.layout.description =
		    mode == CpuReserveMode::Off ? text : state.layout.description + "; " + text;
	}
	if (keep_default) {
		state.applied_default = false;
		state.applied.clear();
	} else {
		if (!wanted.empty() && SetProcessDefaultCpuSets(GetCurrentProcess(), wanted.data(),
		                                                static_cast<ULONG>(wanted.size())) == 0) {
			state.layout.description =
			    "SetProcessDefaultCpuSets failed (" + std::to_string(GetLastError()) + ")";
			state.layout.reserved = false;
			state.layout.vcache   = false;
			wanted.clear();
		}
		if (wanted.empty()) {
			if (state.applied_default) {
				// Nothing applied any more: restore the external defaults.
				(void)SetProcessDefaultCpuSets(GetCurrentProcess(),
				                               state.external.empty() ? nullptr : state.external.data(),
				                               static_cast<ULONG>(state.external.size()));
				state.applied_default = false;
			}
			state.applied.clear();
		} else {
			state.applied_default = true;
			std::sort(wanted.begin(), wanted.end());
			state.applied = std::move(wanted);
		}
	}
	const bool reserved = state.layout.reserved;
	ApplyThreadSets(state.threads[static_cast<size_t>(ThreadRole::Cp)],
	                reserved ? state.layout.cp_ids : std::vector<uint32_t> {});
	ApplyThreadSets(state.threads[static_cast<size_t>(ThreadRole::Recorder)],
	                reserved ? state.layout.recorder_ids : std::vector<uint32_t> {});
	ApplyThreadSets(state.sequencer, reserved ? state.layout.sequencer_ids : std::vector<uint32_t> {});
	for (size_t role = 0; role < state.threads.size(); role++) {
		if (role != static_cast<size_t>(ThreadRole::Cp) &&
		    role != static_cast<size_t>(ThreadRole::Recorder)) {
			ApplyThreadSets(state.threads[role], {});
		}
	}
	for (const auto& thread: state.cold) {
		ApplyThreadSets(thread.handle,
		                state.layout.vcache ? state.layout.cold_ids : std::vector<uint32_t> {});
	}
	const auto guest_ids = GuestIdsLocked(state);
	for (const auto& thread: state.guest) {
		ApplyThreadSets(thread.handle, guest_ids);
	}
	state.vcache_applied = state.layout.vcache;
#else
	state.layout.reserved = false;
	state.layout.vcache   = false;
	if (!active) {
		return;
	}
#endif
	g_cp_mask.store(state.layout.reserved ? state.layout.cp_mask : 0, std::memory_order_relaxed);
	g_recorder_mask.store(state.layout.reserved ? state.layout.recorder_mask : 0,
	                      std::memory_order_relaxed);
	if (log && active) {
		std::printf("Kyty CPU placement (%s%s, %s): process mask %s: %s\n",
		            restriction != 0 ? "KYTY_CPU_RESERVE/KYTY_CPU_SETS" : "KYTY_CPU_RESERVE",
		            state.layout.vcache ? "/KYTY_VCACHE_PLACEMENT" : "", why,
		            MaskText(state.affinity).c_str(), state.layout.description.c_str());
		std::fflush(stdout);
	}
}

// Once a second: the layout against the process mask (and, with KYTY_CPU_RESERVE_REASSERT, the
// process default CPU sets); every 5 s: the hard-affinity scan.
void StartMonitor() {
#if defined(_WIN32)
	const bool maintain = GetCpuReserveMode() != CpuReserveMode::Off || CpuSetsMask() != 0 ||
	                      [] {
		                      auto&           state = GetState();
		                      std::lock_guard lock(state.mutex);
		                      return VCacheWatchedLocked(state);
	                      }();
	const bool repin    = maintain && RepinEnabled();
	const bool scan     = repin || PlacementSamplingEnabled();
	if (!maintain && !scan) {
		return;
	}
	std::thread([maintain, repin, scan] {
		Profiler::SetThreadName("CPU placement monitor");
		PlaceCurrentThreadCold();
		for (uint32_t second = 1;; second++) {
			std::this_thread::sleep_for(std::chrono::seconds(1));
			if (maintain) {
				MaintainCpuPlacement();
			}
			if (scan && second % 5 == 2) {
				(void)ScanThreadAffinities(repin);
			}
		}
	}).detach();
#endif
}

} // namespace

void InitCpuPlacement() {
	auto& state = GetState();
	{
		std::lock_guard lock(state.mutex);
		if (state.initialized) {
			return;
		}
		state.initialized = true;
#if defined(_WIN32)
		// The process default CPU sets someone else set before startup (a Process Lasso CPU-set
		// rule), or empty.
		state.external = ProcessDefaultCpuSets();
		state.topology = DetectVCache(ReadL3Caches());
		// KYTY_VCACHE_PLACEMENT=auto stays off when another process's rule is already in effect.
		state.external_rule = !state.external.empty() || AffinityRestricted();
#endif
		state.started = std::chrono::steady_clock::now();
		state.settled = VCacheSettleMs() == 0;
		ApplyLayoutLocked(state, "startup");
	}
	StartMonitor();
}

void MaintainCpuPlacement() {
#if defined(_WIN32)
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	if (!state.initialized) {
		return;
	}
	const bool vcache_watched = VCacheWatchedLocked(state);
	if (GetCpuReserveMode() == CpuReserveMode::Off && CpuSetsMask() == 0 && !vcache_watched &&
	    !state.vcache_applied) {
		return;
	}
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
		return;
	}
	if (vcache_watched && GetVCacheMode() == VCacheMode::Auto) {
		// An external rule: the affinity narrowed, or process default CPU sets that are not ours.
		auto       current  = ProcessDefaultCpuSets();
		const bool replaced = state.applied_default ? current != state.applied : !current.empty();
		if (process_mask != system_mask || replaced) {
			// Dropped for good; from here on everything is as without the V-cache placement.
			state.external_rule = true;
			if (static_cast<uint64_t>(process_mask) != state.affinity) {
				// As without it: the layout follows the new process mask.
				ApplyLayoutLocked(state, "process affinity changed, external rule");
				return;
			}
			if (state.layout.vcache && !ReassertEnabled()) {
				// Only the default CPU sets were replaced. Without the V-cache placement nothing would
				// have noticed: the other process's sets stay, the reserved threads keep theirs.
				if (CpuSetsMask() == 0) {
					state.external = std::move(current);
				}
				ApplyLayoutLocked(state, "external CPU sets appeared", true, true);
				return;
			}
			// Logged here; KYTY_CPU_RESERVE_REASSERT (below) applies the layout again if it is on.
			LogVCacheLocked(state, "external CPU sets appeared", VCacheDecisionLocked(state, nullptr));
		} else if (!state.settled && std::chrono::steady_clock::now() - state.started >=
		                                 std::chrono::milliseconds(VCacheSettleMs())) {
			state.settled = true;
			ApplyLayoutLocked(state, "no external rule after the settle time");
			return;
		}
	}
	if (static_cast<uint64_t>(process_mask) != state.affinity) {
		ApplyLayoutLocked(state, "process affinity changed");
		return;
	}
	if (g_rules_changed.exchange(false)) {
		ApplyLayoutLocked(state, CurrentReserveRules().scaled ? "KYTY_CPU_RESERVE_SCALE on"
		                                                      : "KYTY_CPU_RESERVE_SCALE off");
		return;
	}
	if (!ReassertEnabled() || !state.applied_default) {
		return;
	}
	// KYTY_CPU_RESERVE_REASSERT: another process replaced the default CPU sets applied here (Process
	// Lasso's CPU-set rule does so once, shortly after the process starts). Without KYTY_CPU_SETS
	// its sets become the constraint, then the layout is applied again.
	auto current = ProcessDefaultCpuSets();
	if (current == state.applied) {
		return;
	}
	const auto count    = g_reasserts.fetch_add(1, std::memory_order_relaxed) + 1u;
	const bool log      = count <= 8u || count % 60u == 0u;
	const auto replaced = LogicalMaskOfIds(state, current);
	const auto ours     = LogicalMaskOfIds(state, state.applied);
	if (CpuSetsMask() == 0) {
		state.external = std::move(current);
	}
	if (log) {
		std::printf("Kyty CPU placement: another process replaced the process default CPU sets "
		            "(logical %s instead of %s); applying the layout again (re-assert %u, "
		            "KYTY_CPU_RESERVE_REASSERT)\n",
		            MaskText(replaced).c_str(), MaskText(ours).c_str(), count);
		std::fflush(stdout);
	}
	ApplyLayoutLocked(state, "re-assert", log);
#endif
}

uint32_t PlacementReasserts() {
	return g_reasserts.load(std::memory_order_relaxed);
}

void PlaceCurrentThread(ThreadRole role) {
	InitCpuPlacement();
#if defined(_WIN32)
	if (role != ThreadRole::Cp && role != ThreadRole::Recorder) {
		return; // the process default already keeps them off the reserved cores
	}
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	auto&           slot = state.threads[static_cast<size_t>(role)];
	if (slot != nullptr) {
		CloseHandle(slot);
		slot = nullptr;
	}
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, 0,
	                    FALSE, DUPLICATE_SAME_ACCESS) != 0) {
		slot = handle;
	}
	if (state.layout.reserved) {
		ApplyThreadSets(GetCurrentThread(),
		                role == ThreadRole::Cp ? state.layout.cp_ids : state.layout.recorder_ids);
	}
#else
	(void)role;
#endif
}

void PlaceCurrentThreadSequencer() {
	InitCpuPlacement();
#if defined(_WIN32)
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	if (state.sequencer != nullptr) {
		CloseHandle(state.sequencer);
		state.sequencer = nullptr;
	}
	HANDLE handle = nullptr;
	if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, 0,
	                    FALSE, DUPLICATE_SAME_ACCESS) != 0) {
		state.sequencer = handle;
	}
	if (state.layout.reserved && !state.layout.sequencer_ids.empty()) {
		ApplyThreadSets(GetCurrentThread(), state.layout.sequencer_ids);
	}
#endif
}

#if defined(_WIN32)
namespace {

// The calling thread as a cold (or guest) thread while it lives.
struct ColdRegistration {
	explicit ColdRegistration(bool guest_thread = false): guest(guest_thread) {
		InitCpuPlacement();
		auto&           state = GetState();
		std::lock_guard lock(state.mutex);
		HANDLE          handle = nullptr;
		if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle, 0,
		                    FALSE, DUPLICATE_SAME_ACCESS) != 0) {
			(guest ? state.guest : state.cold).push_back({tid, handle});
		}
		if (guest) {
			if (const auto ids = GuestIdsLocked(state); !ids.empty()) {
				ApplyThreadSets(GetCurrentThread(), ids);
			}
		} else if (state.layout.vcache) {
			ApplyThreadSets(GetCurrentThread(), state.layout.cold_ids);
		}
	}
	~ColdRegistration() {
		auto&           state = GetState();
		std::lock_guard lock(state.mutex);
		auto&           list = guest ? state.guest : state.cold;
		for (auto it = list.begin(); it != list.end(); ++it) {
			if (it->tid == tid) {
				CloseHandle(it->handle);
				list.erase(it);
				break;
			}
		}
	}
	ColdRegistration(const ColdRegistration&)            = delete;
	ColdRegistration& operator=(const ColdRegistration&) = delete;

	bool  guest = false;
	DWORD tid   = GetCurrentThreadId();
};

} // namespace
#endif

void PlaceCurrentThreadCold() {
#if defined(_WIN32)
	thread_local ColdRegistration registration;
	(void)registration;
#endif
}

void LowerCurrentThreadPriority() {
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
}

void PlaceCurrentThreadGuest() {
#if defined(_WIN32)
	thread_local ColdRegistration registration(true);
	(void)registration;
#endif
}

CpuLayout CurrentCpuLayout() {
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	return state.layout;
}

namespace {

#if defined(_WIN32)
std::string ThreadDescription(HANDLE thread) {
	using Get                = HRESULT(WINAPI*)(HANDLE, PWSTR*);
	static const auto getter = reinterpret_cast<Get>(
	    GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription"));
	PWSTR wide = nullptr;
	if (getter == nullptr || FAILED(getter(thread, &wide)) || wide == nullptr) {
		return {};
	}
	char text[128] {};
	(void)WideCharToMultiByte(CP_UTF8, 0, wide, -1, text, sizeof(text) - 1, nullptr, nullptr);
	LocalFree(wide);
	return text;
}
#endif

} // namespace

AffinityScan ScanThreadAffinities(bool repin) {
	AffinityScan scan;
#if defined(_WIN32)
	InitCpuPlacement();
	const auto layout       = CurrentCpuLayout();
	DWORD_PTR  process_mask = 0;
	DWORD_PTR  system_mask  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
		return scan;
	}
	const auto process  = static_cast<uint64_t>(process_mask);
	HANDLE     snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snapshot == INVALID_HANDLE_VALUE) {
		return scan;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.scan_mutex);
	const auto      pid = GetCurrentProcessId();
	THREADENTRY32   entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Thread32First(snapshot, &entry); more != 0;
	     more      = Thread32Next(snapshot, &entry)) {
		if (entry.th32OwnerProcessID != pid) {
			continue;
		}
		scan.threads++;
		const DWORD access =
		    THREAD_QUERY_LIMITED_INFORMATION | (repin ? THREAD_SET_LIMITED_INFORMATION : 0u);
		HANDLE thread = OpenThread(access, FALSE, entry.th32ThreadID);
		if (thread == nullptr) {
			continue;
		}
		GROUP_AFFINITY affinity {};
		const auto     mask = GetThreadGroupAffinity(thread, &affinity) != 0 && affinity.Group == 0
		                          ? static_cast<uint64_t>(affinity.Mask)
		                          : process;
		// A strict subset of the process mask (a mask with other bits is mid-update by a process
		// affinity change).
		if (mask != process && (mask & ~process) == 0) {
			scan.hard++;
			const auto tid          = static_cast<uint32_t>(entry.th32ThreadID);
			const bool new_hard     = state.hard_seen.insert(tid).second;
			const bool reserved     = layout.reserved && (mask & layout.general_mask) == 0;
			const bool new_confined = reserved && state.reserved_seen.insert(tid).second;
			if (new_hard) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::CpuPlacementHardAffinity);
			}
			if (reserved) {
				scan.reserved_only++;
				if (new_confined) {
					Profiler::CountFrameEvent(
					    Profiler::FrameEvent::CpuPlacementHardAffinityReserved);
				}
			}
			uint64_t repinned_to = 0;
			if (reserved && repin) {
				const auto target = GeneralPoolAffinity(mask, layout) & process;
				if (target != 0 &&
				    SetThreadAffinityMask(thread, static_cast<DWORD_PTR>(target)) != 0) {
					scan.repinned++;
					repinned_to = target;
					Profiler::CountFrameEvent(Profiler::FrameEvent::CpuPlacementRepinned);
				}
			}
			if ((new_hard || new_confined || repinned_to != 0) && state.logged < 32) {
				state.logged++;
				const std::string moved =
				    repinned_to != 0 ? ", moved to " + MaskText(repinned_to) : std::string();
				std::printf(
				    "Kyty CPU placement: thread %u '%s' has hard affinity %s (process %s)%s%s\n",
				    tid, ThreadDescription(thread).c_str(), MaskText(mask).c_str(),
				    MaskText(process).c_str(), reserved ? ", confined to reserved cores" : "",
				    moved.c_str());
				std::fflush(stdout);
			}
		}
		CloseHandle(thread);
	}
	CloseHandle(snapshot);
#else
	(void)repin;
#endif
	return scan;
}

bool PlacementSamplingEnabled() {
	static const bool enabled = [] {
		const auto* value = EnvValue("KYTY_CPU_PLACEMENT_SAMPLES");
		if (value != nullptr) {
			return std::strcmp(value, "0") != 0;
		}
		return Profiler::AggregateEnabled() || HangTrace::Enabled();
	}();
	return enabled;
}

uint32_t CurrentLogicalProcessor() {
#if defined(_WIN32)
	PROCESSOR_NUMBER number {};
	GetCurrentProcessorNumberEx(&number);
	return static_cast<uint32_t>(number.Group) * 64u + number.Number;
#else
	return UINT32_MAX;
#endif
}

uint32_t CoreOfLogicalProcessor(uint32_t logical) {
	if (!g_topology_ready.load(std::memory_order_acquire)) {
		InitCpuPlacement();
	}
	return logical < g_core_of.size() ? g_core_of[logical].load(std::memory_order_relaxed)
	                                  : UINT32_MAX;
}

std::vector<CpuSetInfo> ReadProcessCpuSets(uint64_t* affinity) {
	uint64_t mask = 0;
	std::vector<CpuSetInfo> sets;
#if defined(_WIN32)
	auto& state = GetState();
	{
		std::lock_guard lock(state.mutex);
		// Before the placement started: the default CPU sets now are another process's.
		const auto external = state.initialized ? state.external : ProcessDefaultCpuSets();
		sets                = ReadCpuSets(mask, external);
	}
#endif
	if (affinity != nullptr) {
		*affinity = mask;
	}
	return sets;
}

void VisitPlacedThreads(void (*visit)(void* context, uint32_t tid, void* handle, bool guest),
                        void* context) {
#if defined(_WIN32)
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	for (const auto& thread: state.guest) {
		visit(context, static_cast<uint32_t>(thread.tid), thread.handle, true);
	}
	for (const auto& thread: state.cold) {
		visit(context, static_cast<uint32_t>(thread.tid), thread.handle, false);
	}
#else
	(void)visit;
	(void)context;
#endif
}

uint32_t ProcessAffinityProcessorCount() {
	if (CoreBudgetEnabled()) {
		if (const auto logical = CoreBudgetLogical(); logical != 0) {
			return logical;
		}
	}
#if defined(_WIN32)
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) != 0 && process_mask != 0) {
		return static_cast<uint32_t>(std::popcount(static_cast<uint64_t>(process_mask)));
	}
#endif
	return std::max(1u, std::thread::hardware_concurrency());
}

void SamplePlacement(ThreadRole role) {
	if (!PlacementSamplingEnabled()) {
		return;
	}
	const auto logical = CurrentLogicalProcessor();
	const auto core    = CoreOfLogicalProcessor(logical);
	const auto index   = static_cast<size_t>(role);
	if (logical < PlacementHistogram::Processors) {
		g_histogram[index][logical].fetch_add(1, std::memory_order_relaxed);
	}
	using E             = Profiler::FrameEvent;
	const auto off_mask = [logical](uint64_t reserved) {
		return reserved != 0 && (logical >= 64 || ((reserved >> logical) & 1u) == 0);
	};
	if (role == ThreadRole::Cp) {
		g_cp_core.store(core, std::memory_order_relaxed);
		Profiler::CountFrameEvent(E::CpuPlacementCpSamples);
		if (off_mask(g_cp_mask.load(std::memory_order_relaxed))) {
			Profiler::CountFrameEvent(E::CpuPlacementCpOffCore);
		}
		return;
	}
	const bool on_cp_core = core != UINT32_MAX && core == g_cp_core.load(std::memory_order_relaxed);
	if (on_cp_core) {
		g_on_cp_core[index].fetch_add(1, std::memory_order_relaxed);
	}
	if (role == ThreadRole::Recorder) {
		// Its samples and same-core samples are CpRecorderPlacementSamples/SameCoreSamples.
		if (off_mask(g_recorder_mask.load(std::memory_order_relaxed))) {
			Profiler::CountFrameEvent(E::CpuPlacementRecorderOffCore);
		}
		return;
	}
	const bool guest = role == ThreadRole::Guest;
	Profiler::CountFrameEvent(guest ? E::CpuPlacementGuestSamples : E::CpuPlacementHostSamples);
	if (on_cp_core) {
		Profiler::CountFrameEvent(guest ? E::CpuPlacementGuestOnCpCore
		                                : E::CpuPlacementHostOnCpCore);
	}
}

void TakePlacementHistogram(PlacementHistogram& out) {
	for (size_t role = 0; role < out.samples.size(); role++) {
		for (size_t cpu = 0; cpu < PlacementHistogram::Processors; cpu++) {
			out.samples[role][cpu] = g_histogram[role][cpu].exchange(0, std::memory_order_relaxed);
		}
		out.on_cp_core[role] = g_on_cp_core[role].exchange(0, std::memory_order_relaxed);
	}
	out.cp_core = g_cp_core.load(std::memory_order_relaxed);
}

} // namespace Common
