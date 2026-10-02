#include "common/cpuPlacement.h"

#include "common/hangTrace.h"
#include "common/profiler.h"

#include <algorithm>
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
                           int forced_cp_logical, uint32_t min_general) {
	CpuLayout layout;
	if (mode == CpuReserveMode::Off) {
		layout.description = "off";
		return layout;
	}
	struct Core {
		int                            index      = 0;
		uint8_t                        cache      = 0;
		uint8_t                        preference = 0;
		bool                           whole      = true; // every logical processor allowed
		bool                           has_zero   = false;
		std::vector<const CpuSetInfo*> processors;
	};
	std::map<int, Core>         cores;
	std::map<uint8_t, uint32_t> allowed_per_cache;
	for (const auto& set: sets) {
		if (set.group != 0 || set.logical >= 64) {
			continue;
		}
		auto& core      = cores[set.core];
		core.index      = set.core;
		core.cache      = set.cache;
		core.preference = std::max(core.preference, set.scheduling_class);
		core.whole      = core.whole && set.allowed;
		core.has_zero   = core.has_zero || set.logical == 0;
		core.processors.push_back(&set);
		if (set.allowed) {
			allowed_per_cache[set.cache]++;
		}
	}
	if (allowed_per_cache.empty()) {
		layout.description = "no allowed processor in group 0";
		return layout;
	}
	// The cache holding most allowed processors (ties: the lower index, the V-cache CCD here).
	uint8_t  cache = 0;
	uint32_t best  = 0;
	for (const auto& [index, count]: allowed_per_cache) {
		if (count > best) {
			best  = count;
			cache = index;
		}
	}
	std::vector<const Core*> candidates;
	for (const auto& [index, core]: cores) {
		if (core.whole && core.cache == cache) {
			candidates.push_back(&core);
		}
	}
	std::sort(candidates.begin(), candidates.end(), [](const Core* a, const Core* b) {
		if (a->has_zero != b->has_zero) {
			return !a->has_zero; // the core of processor 0 (interrupts) last
		}
		if (a->preference != b->preference) {
			return a->preference > b->preference;
		}
		return a->index < b->index;
	});
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
	const size_t needed = mode == CpuReserveMode::CpRecorder ? 2u : 1u;
	if (candidates.size() < needed) {
		layout.description =
		    "fewer than " + std::to_string(needed) + " whole allowed cores in the largest cache";
		return layout;
	}
	const Core* cp       = candidates[0];
	const Core* recorder = needed == 2 ? candidates[1] : nullptr;
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
	layout.description   = "CP on logical " + MaskText(layout.cp_mask) + " (core " +
	                       std::to_string(cp->index) + ", preference " +
	                       std::to_string(cp->preference) + ")";
	if (recorder != nullptr) {
		layout.description += ", recorder on logical " + MaskText(layout.recorder_mask) +
		                      " (core " + std::to_string(recorder->index) + ")";
	}
	layout.description += ", other threads on " + MaskText(layout.general_mask);
	return layout;
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
			std::printf("KYTY_CPU_RESERVE must be off, cp or cp+recorder (got '%s'): off\n", value);
		}
		return mode;
	}();
	return mode;
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
			const bool ours      = cpu.Allocated == 0 || cpu.AllocatedToTargetProcess != 0;
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
	bool applied_default = false; // our process default CPU sets are in effect
	// The constraint without KYTY_CPU_SETS: the default CPU sets found at startup, replaced by
	// another process's when it replaces ours (KYTY_CPU_RESERVE_REASSERT). Restored when nothing
	// is reserved.
	std::vector<ULONG> external;
	// The process default CPU sets this module applied last (sorted), while applied_default.
	std::vector<ULONG> applied;
#endif
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

// State mutex held: computes the layout for the current process mask and constraint and applies
// it: the process default CPU sets (the general processors of a reservation, or every allowed
// processor with only KYTY_CPU_SETS) and the reserved threads' selected CPU sets.
void ApplyLayoutLocked(State& state, const char* why, bool log = true) {
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
	state.layout           = ComputeCpuLayout(state.sets, mode, ForcedCpLogical());
	if (mode == CpuReserveMode::Off && restriction == 0) {
		return;
	}
#if defined(_WIN32)
	std::vector<ULONG> wanted;
	if (state.layout.reserved) {
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
	if (!wanted.empty() && SetProcessDefaultCpuSets(GetCurrentProcess(), wanted.data(),
	                                                static_cast<ULONG>(wanted.size())) == 0) {
		state.layout.description =
		    "SetProcessDefaultCpuSets failed (" + std::to_string(GetLastError()) + ")";
		state.layout.reserved = false;
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
		for (auto thread: state.threads) {
			ApplyThreadSets(thread, {});
		}
	} else {
		state.applied_default = true;
		std::sort(wanted.begin(), wanted.end());
		state.applied = std::move(wanted);
		if (state.layout.reserved) {
			ApplyThreadSets(state.threads[static_cast<size_t>(ThreadRole::Cp)],
			                state.layout.cp_ids);
			ApplyThreadSets(state.threads[static_cast<size_t>(ThreadRole::Recorder)],
			                state.layout.recorder_ids);
		} else {
			for (auto thread: state.threads) {
				ApplyThreadSets(thread, {});
			}
		}
	}
#else
	state.layout.reserved = false;
#endif
	g_cp_mask.store(state.layout.reserved ? state.layout.cp_mask : 0, std::memory_order_relaxed);
	g_recorder_mask.store(state.layout.reserved ? state.layout.recorder_mask : 0,
	                      std::memory_order_relaxed);
	if (log) {
		std::printf("Kyty CPU placement (%s, %s): process mask %s: %s\n",
		            restriction != 0 ? "KYTY_CPU_RESERVE/KYTY_CPU_SETS" : "KYTY_CPU_RESERVE", why,
		            MaskText(state.affinity).c_str(), state.layout.description.c_str());
		std::fflush(stdout);
	}
}

// Once a second: the layout against the process mask (and, with KYTY_CPU_RESERVE_REASSERT, the
// process default CPU sets); every 5 s: the hard-affinity scan.
void StartMonitor() {
#if defined(_WIN32)
	const bool maintain = GetCpuReserveMode() != CpuReserveMode::Off || CpuSetsMask() != 0;
	const bool repin    = maintain && RepinEnabled();
	const bool scan     = repin || PlacementSamplingEnabled();
	if (!maintain && !scan) {
		return;
	}
	std::thread([maintain, repin, scan] {
		Profiler::SetThreadName("CPU placement monitor");
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
#endif
		ApplyLayoutLocked(state, "startup");
	}
	StartMonitor();
}

void MaintainCpuPlacement() {
#if defined(_WIN32)
	if (GetCpuReserveMode() == CpuReserveMode::Off && CpuSetsMask() == 0) {
		return;
	}
	DWORD_PTR process_mask = 0;
	DWORD_PTR system_mask  = 0;
	if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0) {
		return;
	}
	auto&           state = GetState();
	std::lock_guard lock(state.mutex);
	if (!state.initialized) {
		return;
	}
	if (static_cast<uint64_t>(process_mask) != state.affinity) {
		ApplyLayoutLocked(state, "process affinity changed");
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
