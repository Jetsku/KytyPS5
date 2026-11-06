// KYTY_CPU_RESERVE (common/cpuPlacement.h).
// 1. ComputeCpuLayout on synthetic CPU set tables (this machine's among them), GeneralPoolAffinity
//    and ParseCpuReserveMode.
// 2. Live (Windows), in the mode --mode picks (cp, the default, or cp+recorder): the CP and recorder
//    threads stay on their cores; threads created before and after the layout stay off them, also
//    after a Process Lasso-like affinity change; placement samples and their counters agree; a hard
//    affinity confined to the CP's core is found and moved to the general processors, and one that
//    meets them is narrowed by the CPU sets; process default CPU sets replaced by another process
//    (Process Lasso's CPU-set rule) are applied again (KYTY_CPU_RESERVE_REASSERT).
//    KYTY_CPU_SETS: its list parser (ParseLogicalMask).
// 3. KYTY_VCACHE_PLACEMENT: DetectVCache, DecideVCachePlacement and ComputeVCacheLayout on fake
//    topologies (7950X3D both ways round, 7900X3D, 7800X3D/9800X3D, 7950X, Intel hybrid, 6-core
//    12-thread, external CPU sets). --vcache sets|affinity|pending (Windows, a dual-CCD X3D only,
//    otherwise skipped): the placement applied live, then dropped when another process's default
//    CPU sets or affinity mask appear; pending: auto waits for its settle time first.
// 4. --bench [seconds] [reps]: KYTY_CPU_RESERVE=off against cp, in child processes on logical
//    processors 0-15 at HIGH priority (Process Lasso's rule for the game). A CP-like thread runs
//    fixed work chunks after ~100 us blocking waits, next to bursty guest-like threads and a yield
//    poller; it reports the CP's chunk time and wake latency and the guests' throughput.

#include "common/cpuPlacement.h"
#include "common/profiler.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

using Common::CpuLayout;
using Common::CpuReserveMode;
using Common::CpuSetInfo;
using Common::L3Cache;
using Common::VCacheMode;
using Profiler::FrameEvent;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpuPlacementTests: failed: %s\n", text);
		std::fflush(stderr);
		std::abort();
	}
}

void SetEnv(const char* name, const char* value) {
#if defined(_WIN32)
	(void)_putenv_s(name, value);
#else
	(void)setenv(name, value, 1);
#endif
}

uint32_t Count(uint64_t mask) {
	return static_cast<uint32_t>(std::popcount(mask));
}

// This machine (Ryzen 9 7950X3D) as GetSystemCpuSetInformation reports it: SMT pairs, CCD0
// (logical 0-15, cache 0, V-cache) and CCD1 (16-31, cache 16), CPPC SchedulingClass.
std::vector<CpuSetInfo> MachineTable(uint64_t allowed) {
	static constexpr uint8_t Classes[32] = {6,  6,  7,  7,  2,  2,  4,  4,  3,  3,  5,
	                                        5,  0,  0,  1,  1,  11, 11, 14, 14, 12, 12,
	                                        14, 14, 10, 10, 13, 13, 8,  8,  9,  9};
	std::vector<CpuSetInfo> sets;
	for (uint32_t lp = 0; lp < 32; lp++) {
		CpuSetInfo set;
		set.id               = 256 + lp;
		set.logical          = static_cast<uint8_t>(lp);
		set.core             = static_cast<uint8_t>(lp & ~1u);
		set.cache            = static_cast<uint8_t>(lp < 16 ? 0 : 16);
		set.scheduling_class = Classes[lp];
		set.allowed          = ((allowed >> lp) & 1u) != 0;
		sets.push_back(set);
	}
	return sets;
}

void TestLayouts() {
	using Common::ComputeCpuLayout;
	const auto lasso = MachineTable(0xFFFF);
	{
		const auto layout = ComputeCpuLayout(lasso, CpuReserveMode::Off, -1);
		Check(!layout.reserved && layout.cp_ids.empty() && layout.general_ids.empty(),
		      "off reserves nothing");
	}
	{
		const auto layout = ComputeCpuLayout(lasso, CpuReserveMode::Cp, -1);
		Check(layout.reserved && layout.cp_core == 2 && layout.cp_mask == 0xC,
		      "Lasso mask: CP on logical 2-3 (preference 7)");
		Check(layout.recorder_mask == 0 && layout.general_mask == (0xFFFFu & ~0xCu),
		      "Lasso mask: the other 14 processors are general");
		Check(layout.cp_ids == std::vector<uint32_t> {258, 259} && layout.general_ids.size() == 14,
		      "Lasso mask: CPU set ids");
	}
	{
		const auto layout = ComputeCpuLayout(lasso, CpuReserveMode::CpRecorder, -1);
		Check(layout.reserved && layout.cp_mask == 0xC && layout.recorder_core == 10 &&
		          layout.recorder_mask == 0xC00 &&
		          layout.recorder_ids == std::vector<uint32_t> {266, 267},
		      "cp+recorder: recorder on logical 10-11 (preference 5; core 0 comes last)");
		Check(layout.general_mask == (0xFFFFu & ~0xC0Cu) && layout.general_ids.size() == 12,
		      "cp+recorder: 12 general processors");
	}
	{
		const auto layout = ComputeCpuLayout(lasso, CpuReserveMode::Stages, -1);
		Check(layout.reserved && layout.cp_mask == 0xC && layout.recorder_mask == 0xC00 &&
		          Count(layout.sequencer_mask) == 2 && layout.sequencer_ids.size() == 2 &&
		          (layout.sequencer_mask & (0xC0Cu | 0x3u)) == 0 && layout.sequencer_core >= 0,
		      "stages: the sequencer gets a third whole core (not core 0)");
		Check(layout.general_mask == (0xFFFFu & ~(0xC0Cu | layout.sequencer_mask)) &&
		          layout.general_ids.size() == 10,
		      "stages: 10 general processors");
		Check(ComputeCpuLayout(MachineTable(0xFFF), CpuReserveMode::Stages, -1).reserved,
		      "12 processors: stages leave 6 (the minimum)");
		Check(!ComputeCpuLayout(MachineTable(0x3FF), CpuReserveMode::Stages, -1).reserved,
		      "10 processors: no stages");
	}
	{
		// Before Process Lasso's mask: both caches hold 16, the lower index (V-cache) wins.
		const auto layout = ComputeCpuLayout(MachineTable(0xFFFFFFFF), CpuReserveMode::Cp, -1);
		Check(layout.reserved && layout.cp_mask == 0xC && Count(layout.general_mask) == 30,
		      "full mask: CP on logical 2-3, 30 general processors");
	}
	{
		const auto layout = ComputeCpuLayout(lasso, CpuReserveMode::Cp, 9);
		Check(layout.reserved && layout.cp_core == 8 && layout.cp_mask == 0x300,
		      "KYTY_CPU_RESERVE_CORE=9: core 8");
		const auto both = ComputeCpuLayout(lasso, CpuReserveMode::CpRecorder, 9);
		Check(both.reserved && both.cp_mask == 0x300 && both.recorder_mask == 0xC,
		      "forced CP core: the recorder gets the best other core");
	}
	{
		// Logical 3 not allowed: core 2 is partial, neither chosen nor forceable.
		const auto partial = MachineTable(0xFFFFull & ~0x8ull);
		const auto layout  = ComputeCpuLayout(partial, CpuReserveMode::Cp, -1);
		Check(layout.reserved && layout.cp_core == 10 && layout.cp_mask == 0xC00,
		      "partial core 2: CP on core 10");
		Check((layout.general_mask & 0x8) == 0 && (layout.general_mask & 0x4) != 0,
		      "partial core 2: logical 2 general, 3 unused");
		const auto forced = ComputeCpuLayout(partial, CpuReserveMode::Cp, 2);
		Check(!forced.reserved && forced.description.find("KYTY_CPU_RESERVE_CORE") != std::string::npos,
		      "forcing a partial core reserves nothing");
		const auto missing = ComputeCpuLayout(partial, CpuReserveMode::Cp, 40);
		Check(!missing.reserved, "forcing an unknown processor reserves nothing");
	}
	{
		// Too few processors: 8 leave 6 (the minimum) for cp, 4 for cp+recorder; 6 leave 4.
		Check(ComputeCpuLayout(MachineTable(0xFF), CpuReserveMode::Cp, -1).reserved,
		      "8 processors: cp");
		Check(!ComputeCpuLayout(MachineTable(0xFF), CpuReserveMode::CpRecorder, -1).reserved,
		      "8 processors: no cp+recorder");
		const auto six = ComputeCpuLayout(MachineTable(0x3F), CpuReserveMode::Cp, -1);
		Check(!six.reserved && six.cp_ids.empty() && six.general_ids.empty() &&
		          six.description.find("only 4") != std::string::npos,
		      "6 processors: nothing reserved");
		Check(ComputeCpuLayout(MachineTable(0x3F), CpuReserveMode::Cp, -1, 4).reserved,
		      "6 processors with a minimum of 4: cp");
	}
	{
		// Two caches: 6 allowed processors in cache 0, 10 in cache 16. The CP goes to cache 16, on
		// its best core (18 and 22 tie at 14: the lower index).
		const auto layout =
		    ComputeCpuLayout(MachineTable(0xFC00ull | 0x3FF0000ull), CpuReserveMode::Cp, -1);
		Check(layout.reserved && layout.cp_core == 18 && layout.cp_mask == 0xC0000,
		      "largest cache: CP on logical 18-19");
		Check(Count(layout.general_mask) == 14, "largest cache: every other allowed processor");
	}
	{
		// The core of logical processor 0 comes last, whatever its preference.
		auto sets                = MachineTable(0xFFFF);
		sets[0].scheduling_class = 15;
		sets[1].scheduling_class = 15;
		Check(ComputeCpuLayout(sets, CpuReserveMode::Cp, -1).cp_core == 2, "core 0 comes last");
	}
	{
		Check(!ComputeCpuLayout(MachineTable(0), CpuReserveMode::Cp, -1).reserved,
		      "no allowed processor");
		auto other_group = MachineTable(0xFFFF);
		for (auto& set: other_group) {
			set.group = 1;
		}
		Check(!ComputeCpuLayout(other_group, CpuReserveMode::Cp, -1).reserved, "group 0 only");
	}
	std::puts("CpuPlacementTests: layouts ok");
}

constexpr uint64_t MiB = 1024ull * 1024ull;

L3Cache L3(uint64_t mask, uint64_t mib) {
	return {0, mask, mib * MiB};
}

bool Contains(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

// A CPU set table of `logical` processors with SMT pairs, one cache index per `per_cache`.
std::vector<CpuSetInfo> GenericTable(uint32_t logical, uint32_t per_cache) {
	std::vector<CpuSetInfo> sets;
	for (uint32_t lp = 0; lp < logical; lp++) {
		CpuSetInfo set;
		set.id      = 256 + lp;
		set.logical = static_cast<uint8_t>(lp);
		set.core    = static_cast<uint8_t>(lp & ~1u);
		set.cache   = static_cast<uint8_t>(lp / per_cache * per_cache);
		set.allowed = true;
		sets.push_back(set);
	}
	return sets;
}

void TestVCache() {
	using Common::ComputeCpuLayout;
	using Common::ComputeVCacheLayout;
	using Common::DecideVCachePlacement;
	using Common::DetectVCache;
	const auto full = MachineTable(0xFFFFFFFF);
	{
		// 7950X3D as on this machine: CCD0 (0-15) 96 MiB, CCD1 (16-31) 32 MiB.
		const auto topology = DetectVCache({L3(0xFFFF, 96), L3(0xFFFF0000, 32)});
		Check(topology.dual && topology.groups == 2 && topology.vcache_mask == 0xFFFF &&
		          topology.other_mask == 0xFFFF0000 && topology.vcache_bytes == 96 * MiB &&
		          topology.other_bytes == 32 * MiB,
		      "7950X3D: CCD0 is the V-cache CCD");
		Check(Contains(topology.description, "96 MiB on logical 0-15") &&
		          Contains(topology.description, "32 MiB on logical 16-31"),
		      "7950X3D: description");
		const auto decision = DecideVCachePlacement(VCacheMode::Auto, topology, false, false);
		Check(decision.engage, "7950X3D, no external rule: applies");
		const auto cp = ComputeVCacheLayout(full, CpuReserveMode::Cp, -1, topology);
		Check(cp.vcache && cp.reserved && cp.cp_mask == 0xC && cp.general_mask == (0xFFFFu & ~0xCu) &&
		          cp.cold_mask == 0xFFFF0000ull && cp.cold_ids.size() == 16 &&
		          cp.general_ids.size() == 14,
		      "7950X3D cp: CP on 2-3, hot 0-1,4-15, cold 16-31");
		const auto both = ComputeVCacheLayout(full, CpuReserveMode::CpRecorder, -1, topology);
		Check(both.vcache && both.cp_mask == 0xC && both.recorder_mask == 0xC00 &&
		          (both.general_mask & 0xFFFF0000ull) == 0 && Count(both.general_mask) == 12,
		      "7950X3D cp+recorder: both reserved cores on the V-cache CCD");
		const auto off = ComputeVCacheLayout(full, CpuReserveMode::Off, -1, topology);
		Check(off.vcache && !off.reserved && off.cp_ids.empty() && off.general_mask == 0xFFFF &&
		          off.cold_mask == 0xFFFF0000ull,
		      "7950X3D without a reservation: hot 0-15, cold 16-31");
		// Process Lasso's affinity (0-15): nothing is left for the cold threads.
		const auto lasso = ComputeVCacheLayout(MachineTable(0xFFFF), CpuReserveMode::Cp, -1, topology);
		Check(!lasso.vcache && Contains(lasso.description, "outside the V-cache CCD"),
		      "only the V-cache CCD allowed: not applied");
		// Too few hot processors.
		const auto few = ComputeVCacheLayout(MachineTable(0xF000000F), CpuReserveMode::Off, -1, topology);
		Check(!few.vcache && Contains(few.description, "only 4"), "4 hot processors: not applied");
		// KYTY_CPU_RESERVE_CORE on the other CCD: the reservation does not fit.
		const auto forced = ComputeVCacheLayout(full, CpuReserveMode::Cp, 20, topology);
		Check(!forced.vcache && Contains(forced.description, "does not fit"),
		      "forced CP core outside the V-cache CCD: not applied");
		// Without the placement the legacy tie (16 against 16) goes to the V-cache CCD, here also the
		// lower index: unchanged.
		Check(ComputeCpuLayout(full, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors,
		                       topology.vcache_mask)
		              .cp_mask == 0xC,
		      "7950X3D legacy tie: CCD0");
	}
	{
		// The same chip the other way round: CCD1 (16-31) has the V-cache.
		const auto topology = DetectVCache({L3(0xFFFF, 32), L3(0xFFFF0000, 96)});
		Check(topology.dual && topology.vcache_mask == 0xFFFF0000ull && topology.other_mask == 0xFFFF,
		      "reversed 7950X3D: CCD1 is the V-cache CCD (by size, not index)");
		const auto cp = ComputeVCacheLayout(full, CpuReserveMode::Cp, -1, topology);
		// Cores 18 and 22 tie at preference 14: the lower index.
		Check(cp.vcache && cp.cp_mask == 0xC0000ull &&
		          cp.general_mask == (0xFFFF0000ull & ~0xC0000ull) && cp.cold_mask == 0xFFFF,
		      "reversed 7950X3D: CP on 18-19, hot 16-31, cold 0-15");
		const auto tie = ComputeCpuLayout(full, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors,
		                                  topology.vcache_mask);
		Check(tie.reserved && tie.cp_mask == 0xC0000ull,
		      "reversed 7950X3D legacy tie: the V-cache CCD, not the lower index");
		Check(ComputeCpuLayout(full, CpuReserveMode::Cp, -1).cp_mask == 0xC,
		      "no preference: the lower index as before");
		// Most allowed processors still decide before the tie-break.
		Check(ComputeCpuLayout(MachineTable(0xFFFF), CpuReserveMode::Cp, -1,
		                       Common::kMinGeneralProcessors, topology.vcache_mask)
		              .cp_mask == 0xC,
		      "reversed, mask 0-15: the CP stays in the allowed cache");
	}
	{
		// 7900X3D: 6 cores (12 logical) per CCD, 96 + 32 MiB.
		const auto topology = DetectVCache({L3(0xFFF, 96), L3(0xFFF000, 32)});
		const auto layout =
		    ComputeVCacheLayout(GenericTable(24, 12), CpuReserveMode::Cp, -1, topology);
		Check(topology.dual && layout.vcache && (layout.cp_mask & ~0xFFFull) == 0 &&
		          Count(layout.cp_mask) == 2 && Count(layout.general_mask) == 10 &&
		          layout.cold_mask == 0xFFF000ull,
		      "7900X3D: CP and 10 hot processors on CCD0, cold on CCD1");
	}
	{
		// 7800X3D / 9800X3D: one CCD, one 96 MiB L3.
		const auto topology = DetectVCache({L3(0xFFFF, 96)});
		Check(!topology.dual && topology.groups == 1 && Contains(topology.description, "one L3"),
		      "7800X3D: one L3");
		const auto decision = DecideVCachePlacement(VCacheMode::Auto, topology, false, false);
		Check(!decision.engage && Contains(decision.reason, "not a dual-CCD X3D"),
		      "7800X3D: no change");
		Check(!DecideVCachePlacement(VCacheMode::On, topology, false, false).engage,
		      "7800X3D, KYTY_VCACHE_PLACEMENT=1: still nothing to do");
		Check(!ComputeVCacheLayout(MachineTable(0xFFFF), CpuReserveMode::Cp, -1, topology).vcache,
		      "7800X3D: no layout");
	}
	{
		// 7950X: two CCDs, 32 MiB each.
		const auto topology = DetectVCache({L3(0xFFFF, 32), L3(0xFFFF0000, 32)});
		Check(!topology.dual && topology.groups == 2 && Contains(topology.description, "similar size"),
		      "7950X: not X3D");
		Check(!DecideVCachePlacement(VCacheMode::Auto, topology, false, false).engage,
		      "7950X: no change");
		// Close sizes do not count either (40 against 32 MiB).
		Check(!DetectVCache({L3(0xFFFF, 40), L3(0xFFFF0000, 32)}).dual, "40 against 32 MiB: similar");
		Check(DetectVCache({L3(0xFFFF, 48), L3(0xFFFF0000, 32)}).dual, "48 against 32 MiB: 1.5 times");
	}
	{
		// Intel hybrid (8 P-cores with SMT, 12 E-cores): one 33 MiB L3 for all 28.
		const auto topology = DetectVCache({L3(0xFFFFFFF, 33)});
		Check(!topology.dual && !DecideVCachePlacement(VCacheMode::Auto, topology, false, false).engage,
		      "Intel hybrid: no change");
	}
	{
		// 6 cores, 12 threads, one 32 MiB L3.
		const auto topology = DetectVCache({L3(0xFFF, 32)});
		Check(!topology.dual && !DecideVCachePlacement(VCacheMode::Auto, topology, false, false).engage,
		      "6-core 12-thread: no change");
	}
	{
		// Other shapes: duplicates merge, other groups and empty masks are ignored, three groups.
		Check(!DetectVCache({}).dual && Contains(DetectVCache({}).description, "no L3"), "no L3");
		Check(!DetectVCache({L3(0xFFFF, 96), L3(0xFFFF, 96)}).dual, "one L3 reported twice");
		Check(!DetectVCache({L3(0xFFFF, 96), {1, 0xFFFF, 32 * MiB}, L3(0, 32)}).dual,
		      "group 1 and empty masks ignored");
		const auto three = DetectVCache({L3(0xFF, 96), L3(0xFF00, 32), L3(0xFF0000, 32)});
		Check(three.dual && three.vcache_mask == 0xFF && three.other_mask == 0xFFFF00,
		      "three L3s: the largest against the rest");
		Check(!DetectVCache({L3(0xFF, 96), L3(0xFF00, 96), L3(0xFF0000, 32)}).dual,
		      "two equal largest L3s: not one V-cache CCD");
	}
	{
		// External rules and the switch.
		const auto topology = DetectVCache({L3(0xFFFF, 96), L3(0xFFFF0000, 32)});
		const auto external = DecideVCachePlacement(VCacheMode::Auto, topology, true, false);
		Check(!external.engage && Contains(external.reason, "external CPU-set/affinity rule"),
		      "external CPU sets present: no change");
		Check(!DecideVCachePlacement(VCacheMode::Auto, topology, false, true).engage,
		      "KYTY_CPU_SETS: no change");
		Check(DecideVCachePlacement(VCacheMode::On, topology, true, true).engage,
		      "KYTY_VCACHE_PLACEMENT=1: also with an external rule");
		Check(!DecideVCachePlacement(VCacheMode::Off, topology, false, false).engage,
		      "KYTY_VCACHE_PLACEMENT=0: off");
		bool valid = false;
		Check(Common::ParseVCacheMode(nullptr, &valid) == VCacheMode::Auto && valid, "vcache unset");
		Check(Common::ParseVCacheMode("auto", &valid) == VCacheMode::Auto && valid, "vcache auto");
		Check(Common::ParseVCacheMode("0", &valid) == VCacheMode::Off && valid, "vcache 0");
		Check(Common::ParseVCacheMode("off", &valid) == VCacheMode::Off && valid, "vcache off");
		Check(Common::ParseVCacheMode("1", &valid) == VCacheMode::On && valid, "vcache 1");
		Check(Common::ParseVCacheMode("on", &valid) == VCacheMode::On && valid, "vcache on");
		Check(Common::ParseVCacheMode("2", &valid) == VCacheMode::Auto && !valid, "vcache unknown");
	}
	std::puts("CpuPlacementTests: V-cache topologies ok");
}

void TestAffinityAndModes() {
	const auto layout = Common::ComputeCpuLayout(MachineTable(0xFFFF), CpuReserveMode::Cp, -1);
	Check(Common::GeneralPoolAffinity(0x4, layout) == layout.general_mask,
	      "a hard affinity confined to the CP's core gets every general processor");
	Check(Common::GeneralPoolAffinity(0x14, layout) == 0x14,
	      "a hard affinity that meets the general processors is kept");
	Check(Common::GeneralPoolAffinity(0x4, CpuLayout {}) == 0x4, "nothing reserved: kept");
	bool valid = false;
	Check(Common::ParseCpuReserveMode("cp", &valid) == CpuReserveMode::Cp && valid, "mode cp");
	Check(Common::ParseCpuReserveMode("1", &valid) == CpuReserveMode::Cp && valid, "mode 1");
	Check(Common::ParseCpuReserveMode("cp+recorder", &valid) == CpuReserveMode::CpRecorder && valid,
	      "mode cp+recorder");
	Check(Common::ParseCpuReserveMode("2", &valid) == CpuReserveMode::CpRecorder && valid, "mode 2");
	Check(Common::ParseCpuReserveMode("stages", &valid) == CpuReserveMode::Stages && valid,
	      "mode stages");
	Check(Common::ParseCpuReserveMode("3", &valid) == CpuReserveMode::Stages && valid, "mode 3");
	Check(Common::ParseCpuReserveMode("off", &valid) == CpuReserveMode::Off && valid, "mode off");
	Check(Common::ParseCpuReserveMode(nullptr, &valid) == CpuReserveMode::Off && valid, "unset");
	Check(Common::ParseCpuReserveMode("bogus", &valid) == CpuReserveMode::Off && !valid,
	      "unknown text");
	// KYTY_CPU_SETS.
	Check(Common::ParseLogicalMask("0-15", &valid) == 0xFFFFull && valid, "mask 0-15");
	Check(Common::ParseLogicalMask("16-31", &valid) == 0xFFFF0000ull && valid, "mask 16-31");
	Check(Common::ParseLogicalMask("0-3,8,10-11", &valid) == 0xD0Full && valid, "mask list");
	Check(Common::ParseLogicalMask("63", &valid) == (1ull << 63u) && valid, "mask 63");
	Check(Common::ParseLogicalMask("0-63", &valid) == ~0ull && valid, "mask 0-63");
	for (const char* bad: {"", "x", "3-1", "0-64", "64", "1,", ",1", "1-", "1 2", "0-15;"}) {
		Check(Common::ParseLogicalMask(bad, &valid) == 0 && !valid, bad);
	}
	Check(Common::ParseLogicalMask(nullptr, &valid) == 0 && !valid, "mask nullptr");
	std::puts("CpuPlacementTests: general-pool affinity and modes ok");
}

#if defined(_WIN32)

volatile uint64_t g_sink = 0;

// Busy for `ms` milliseconds; the logical processors it ran on (and placement samples as `role`).
uint64_t RunAndRecord(uint32_t ms, const Common::ThreadRole* role = nullptr) {
	uint64_t   seen = 0;
	uint64_t   acc  = 0;
	const auto end  = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
	while (std::chrono::steady_clock::now() < end) {
		for (uint32_t i = 0; i < 4000; i++) {
			acc = acc * 6364136223846793005ull + i;
		}
		const auto logical = Common::CurrentLogicalProcessor();
		if (logical < 64) {
			seen |= uint64_t {1} << logical;
		}
		if (role != nullptr) {
			Common::SamplePlacement(*role);
		}
	}
	g_sink = acc;
	return seen;
}

uint64_t ProcessMask() {
	DWORD_PTR process = 0;
	DWORD_PTR system  = 0;
	Check(GetProcessAffinityMask(GetCurrentProcess(), &process, &system) != 0,
	      "GetProcessAffinityMask");
	return process;
}

// One busy thread per logical processor, recording where each ran.
struct Load {
	explicit Load(uint64_t& seen): seen(seen) {
		const auto count = std::max(2u, std::thread::hardware_concurrency());
		per_thread.resize(count);
		for (uint32_t i = 0; i < count; i++) {
			threads.emplace_back([this, i] {
				const auto role = Common::ThreadRole::Guest;
				while (!stop.load(std::memory_order_relaxed)) {
					per_thread[i] |= RunAndRecord(20, &role);
				}
			});
		}
	}
	~Load() {
		stop.store(true);
		for (auto& thread: threads) {
			thread.join();
		}
		for (const auto mask: per_thread) {
			seen |= mask;
		}
	}
	Load(const Load&)            = delete;
	Load& operator=(const Load&) = delete;

	uint64_t&                seen;
	std::atomic<bool>        stop {false};
	std::vector<uint64_t>    per_thread;
	std::vector<std::thread> threads;
};

void TestLive(CpuReserveMode mode) {
	// A thread that exists before the layout is applied (like the process's early threads).
	std::atomic<int> phase {0};
	uint64_t         early_seen = 0;
	std::thread      early([&] {
        while (phase.load() == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        early_seen = RunAndRecord(500);
    });
	std::this_thread::sleep_for(std::chrono::milliseconds(20));

	const auto original = ProcessMask();
	Common::InitCpuPlacement();
	const auto layout = Common::CurrentCpuLayout();
	std::printf("CpuPlacementTests: live %s, process mask 0x%llx: %s\n",
	            mode == CpuReserveMode::Cp ? "cp" : "cp+recorder",
	            static_cast<unsigned long long>(original), layout.description.c_str());
	if (!layout.reserved) {
		phase = 1;
		early.join();
		std::puts("CpuPlacementTests: live skipped (nothing reserved on this machine)");
		return;
	}
	Check(mode != CpuReserveMode::CpRecorder || layout.recorder_mask != 0,
	      "cp+recorder reserved a recorder core");
	const uint64_t reserved = layout.cp_mask | layout.recorder_mask;

	// Phase 1: the CP, the recorder, the early thread and a full load, all busy at once.
	uint64_t cp_seen       = 0;
	uint64_t recorder_seen = 0;
	uint64_t load_seen     = 0;
	{
		Load        load(load_seen);
		std::thread cp([&] {
			Common::PlaceCurrentThread(Common::ThreadRole::Cp);
			const auto role = Common::ThreadRole::Cp;
			cp_seen         = RunAndRecord(500, &role);
		});
		std::thread recorder([&] {
			Common::PlaceCurrentThread(Common::ThreadRole::Recorder);
			const auto role = Common::ThreadRole::Recorder;
			recorder_seen   = RunAndRecord(500, &role);
		});
		phase = 1;
		cp.join();
		recorder.join();
		early.join();
	}
	std::printf("  ran on: CP 0x%llx, recorder 0x%llx, early thread 0x%llx, load 0x%llx\n",
	            static_cast<unsigned long long>(cp_seen),
	            static_cast<unsigned long long>(recorder_seen),
	            static_cast<unsigned long long>(early_seen),
	            static_cast<unsigned long long>(load_seen));
	Check(cp_seen != 0 && (cp_seen & ~layout.cp_mask) == 0, "the CP ran off its core");
	if (mode == CpuReserveMode::CpRecorder) {
		Check(recorder_seen != 0 && (recorder_seen & ~layout.recorder_mask) == 0,
		      "the recorder ran off its core");
	} else {
		Check(recorder_seen != 0 && (recorder_seen & reserved) == 0,
		      "the recorder (cp mode) ran on the CP's core");
	}
	Check((load_seen & reserved) == 0, "a thread created after the layout ran on a reserved core");
	Check((early_seen & reserved) == 0,
	      "a thread created before the layout ran on a reserved core");
	Check(load_seen != 0 && (load_seen & ~layout.general_mask) == 0,
	      "the load ran outside the general processors");

	// Placement samples: the histogram and the counters agree with where the threads ran.
	Common::PlacementHistogram histogram;
	Common::TakePlacementHistogram(histogram);
	for (uint32_t lp = 0; lp < Common::PlacementHistogram::Processors; lp++) {
		const auto bit = uint64_t {1} << lp;
		Check(histogram.samples[0][lp] == 0 || (layout.cp_mask & bit) != 0,
		      "CP samples outside its core");
		Check(histogram.samples[2][lp] == 0 || (reserved & bit) == 0,
		      "guest samples on a reserved core");
	}
	Check(histogram.on_cp_core[2] == 0, "guest samples on the CP's core");
	Check(Profiler::FrameEventTotal(FrameEvent::CpuPlacementCpSamples) > 0 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementCpOffCore) == 0 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementGuestSamples) > 0 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementGuestOnCpCore) == 0 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementRecorderOffCore) == 0,
	      "placement counters");

	// Phase 2: hard affinities. One confined to a CP processor (CPU sets cannot move it), one that
	// also allows a general processor (the CPU sets narrow it to that one).
	const uint64_t cp_bit      = uint64_t {1} << std::countr_zero(layout.cp_mask);
	const uint64_t general_bit = uint64_t {1} << std::countr_zero(layout.general_mask);
	std::atomic<int> hard_phase {0};
	uint64_t         confined_seen  = 0;
	uint64_t         confined_after = 0;
	uint64_t         mixed_seen     = 0;
	Common::AffinityScan scan;
	Common::AffinityScan repin;
	{
		uint64_t    ignored = 0;
		Load        load(ignored);
		std::thread confined([&] {
			(void)SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(cp_bit));
			confined_seen = RunAndRecord(200);
			hard_phase    = 1;
			while (hard_phase.load() != 2) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			confined_after = RunAndRecord(300);
		});
		std::thread mixed([&] {
			(void)SetThreadAffinityMask(GetCurrentThread(),
			                            static_cast<DWORD_PTR>(cp_bit | general_bit));
			mixed_seen = RunAndRecord(300);
			while (hard_phase.load() != 2) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		});
		while (hard_phase.load() != 1) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		scan       = Common::ScanThreadAffinities(false);
		repin      = Common::ScanThreadAffinities(true);
		hard_phase = 2;
		confined.join();
		mixed.join();
	}
	std::printf("  hard affinity: confined to 0x%llx ran on 0x%llx, after the move on 0x%llx; "
	            "0x%llx ran on 0x%llx; scan: %u threads, %u hard, %u confined; repin moved %u\n",
	            static_cast<unsigned long long>(cp_bit),
	            static_cast<unsigned long long>(confined_seen),
	            static_cast<unsigned long long>(confined_after),
	            static_cast<unsigned long long>(cp_bit | general_bit),
	            static_cast<unsigned long long>(mixed_seen), scan.threads, scan.hard,
	            scan.reserved_only, repin.repinned);
	Check(confined_seen == cp_bit, "a hard affinity on the CP's core ran elsewhere");
	Check(scan.hard >= 2 && scan.reserved_only >= 1 && repin.repinned >= 1,
	      "the scan missed the hard affinities");
	Check(confined_after != 0 && (confined_after & reserved) == 0,
	      "the moved thread still ran on a reserved core");
	Check(mixed_seen == general_bit, "the CPU sets did not narrow a hard affinity");
	Check(Profiler::FrameEventTotal(FrameEvent::CpuPlacementHardAffinity) >= 2 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementHardAffinityReserved) >= 1 &&
	          Profiler::FrameEventTotal(FrameEvent::CpuPlacementRepinned) >= 1,
	      "hard-affinity counters");

	// Phase 3: a Process Lasso-like affinity change to the first half of the processors. The
	// layout follows it (the monitor does this once a second).
	uint64_t half = 0;
	for (uint64_t rest = original; Count(half) < Count(original) / 2; rest &= rest - 1) {
		half |= rest & (~rest + 1); // the lowest half of the allowed processors
	}
	if (Count(half) >= 8) {
		Check(SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(half)) != 0,
		      "SetProcessAffinityMask");
		Common::MaintainCpuPlacement();
		const auto narrowed = Common::CurrentCpuLayout();
		std::printf("  after the mask 0x%llx: %s\n", static_cast<unsigned long long>(half),
		            narrowed.description.c_str());
		Check(narrowed.reserved && ((narrowed.cp_mask | narrowed.recorder_mask |
		                             narrowed.general_mask) &
		                            ~half) == 0,
		      "the layout did not follow the process mask");
		const uint64_t narrowed_reserved = narrowed.cp_mask | narrowed.recorder_mask;
		uint64_t       cp_after          = 0;
		uint64_t       load_after        = 0;
		{
			Load        load(load_after);
			std::thread cp([&] {
				Common::PlaceCurrentThread(Common::ThreadRole::Cp);
				cp_after = RunAndRecord(300);
			});
			cp.join();
		}
		Check(cp_after != 0 && (cp_after & ~narrowed.cp_mask) == 0,
		      "the CP ran off its core after the mask change");
		Check((load_after & narrowed_reserved) == 0 && (load_after & ~half) == 0,
		      "the load ran on a reserved core after the mask change");
		(void)SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(original));
		Common::MaintainCpuPlacement();
	}

	// Phase 4 (KYTY_CPU_RESERVE_REASSERT, set in main): another process replaces the process
	// default CPU sets with every allowed processor, as Process Lasso's CPU-set rule does shortly
	// after the game starts. The next monitor pass applies the layout again, once.
	const auto before = Common::CurrentCpuLayout();
	if (before.reserved) {
		std::vector<ULONG> every;
		for (const auto& ids: {before.cp_ids, before.recorder_ids, before.general_ids}) {
			every.insert(every.end(), ids.begin(), ids.end());
		}
		Check(SetProcessDefaultCpuSets(GetCurrentProcess(), every.data(),
		                               static_cast<ULONG>(every.size())) != 0,
		      "SetProcessDefaultCpuSets (the other process)");
		const auto reasserts = Common::PlacementReasserts();
		Common::MaintainCpuPlacement();
		const auto after = Common::CurrentCpuLayout();
		std::vector<ULONG> current(256);
		ULONG              count = 0;
		Check(GetProcessDefaultCpuSets(GetCurrentProcess(), current.data(),
		                               static_cast<ULONG>(current.size()), &count) != 0,
		      "GetProcessDefaultCpuSets");
		current.resize(count);
		std::sort(current.begin(), current.end());
		std::vector<ULONG> general(after.general_ids.begin(), after.general_ids.end());
		std::sort(general.begin(), general.end());
		std::printf("  after another process set every processor: %u re-assert(s), %s\n",
		            Common::PlacementReasserts() - reasserts, after.description.c_str());
		Check(Common::PlacementReasserts() == reasserts + 1, "the replaced sets were not re-asserted");
		Check(after.reserved && after.cp_mask == before.cp_mask && current == general,
		      "the default CPU sets are not the general processors again");
		uint64_t load_after = 0;
		{
			Load load(load_after);
			std::this_thread::sleep_for(std::chrono::milliseconds(300));
		}
		Check(load_after != 0 && (load_after & (after.cp_mask | after.recorder_mask)) == 0,
		      "a thread ran on a reserved core after the re-assert");
		Common::MaintainCpuPlacement();
		Check(Common::PlacementReasserts() == reasserts + 1, "an unchanged state was re-asserted");
	}
	std::puts("CpuPlacementTests: live placement ok");
}

// The selected CPU sets of a thread (empty: none).
std::vector<ULONG> ThreadSets(HANDLE thread) {
	std::vector<ULONG> ids(256);
	ULONG              count = 0;
	if (GetThreadSelectedCpuSets(thread, ids.data(), static_cast<ULONG>(ids.size()), &count) == 0) {
		count = 0;
	}
	ids.resize(count);
	return ids;
}

// --vcache sets|affinity|pending, with KYTY_CPU_RESERVE=cp and KYTY_VCACHE_PLACEMENT=auto.
int TestVCacheLive(const char* scenario) {
	const bool pending = std::strcmp(scenario, "pending") == 0;
	SetEnv("KYTY_CPU_RESERVE", "cp");
	SetEnv("KYTY_CPU_RESERVE_CORE", "");
	SetEnv("KYTY_CPU_RESERVE_REASSERT", "");
	SetEnv("KYTY_CPU_SETS", "");
	SetEnv("KYTY_VCACHE_PLACEMENT", "auto");
	SetEnv("KYTY_VCACHE_SETTLE_MS", pending ? "600000" : "0");
	const auto original = ProcessMask();
	Common::InitCpuPlacement();
	const auto layout = Common::CurrentCpuLayout();
	std::printf("CpuPlacementTests: V-cache %s, process mask 0x%llx: %s\n", scenario,
	            static_cast<unsigned long long>(original), layout.description.c_str());
	if (pending) {
		Check(!layout.vcache, "auto applied before its settle time");
		std::puts("CpuPlacementTests: V-cache pending ok");
		return 0;
	}
	if (!layout.vcache) {
		std::puts("CpuPlacementTests: V-cache live skipped (not a dual-CCD X3D without an external "
		          "rule here)");
		return 0;
	}
	Check(layout.reserved && layout.cold_mask != 0 && (layout.cp_mask & layout.cold_mask) == 0 &&
	          (layout.general_mask & layout.cold_mask) == 0,
	      "V-cache layout: CP and hot processors off the cold CCD");

	// Hot load, the CP and a cold thread at once.
	std::atomic<int> phase {0};
	HANDLE           cold_handle = nullptr;
	uint64_t         cold_seen   = 0;
	uint64_t         cold_after  = 0;
	std::thread      cold([&] {
        Common::PlaceCurrentThreadCold();
        Common::PlaceCurrentThreadCold(); // idempotent
        (void)DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(),
		                           &cold_handle, 0, FALSE, DUPLICATE_SAME_ACCESS);
        cold_seen = RunAndRecord(400);
        phase     = 1;
        while (phase.load() != 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        cold_after = RunAndRecord(300);
        phase      = 3;
    });
	uint64_t load_seen = 0;
	uint64_t cp_seen   = 0;
	{
		Load        load(load_seen);
		std::thread cp([&] {
			Common::PlaceCurrentThread(Common::ThreadRole::Cp);
			cp_seen = RunAndRecord(400);
		});
		cp.join();
		while (phase.load() != 1) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	std::printf("  ran on: CP 0x%llx, hot load 0x%llx, cold thread 0x%llx\n",
	            static_cast<unsigned long long>(cp_seen),
	            static_cast<unsigned long long>(load_seen),
	            static_cast<unsigned long long>(cold_seen));
	Check(cp_seen != 0 && (cp_seen & ~layout.cp_mask) == 0, "the CP ran off its core");
	Check(load_seen != 0 && (load_seen & ~layout.general_mask) == 0,
	      "a hot thread ran outside the V-cache CCD's general processors");
	Check(cold_seen != 0 && (cold_seen & ~layout.cold_mask) == 0,
	      "the cold thread ran outside the other CCD");

	// An external rule appears. sets: another process sets every processor as the process default
	// CPU sets (Process Lasso's CPU-set rule); affinity: the process mask narrows to the V-cache CCD.
	std::vector<ULONG> every;
	for (const auto& ids: {layout.cp_ids, layout.general_ids, layout.cold_ids}) {
		every.insert(every.end(), ids.begin(), ids.end());
	}
	std::sort(every.begin(), every.end());
	const bool sets = std::strcmp(scenario, "sets") == 0;
	if (sets) {
		Check(SetProcessDefaultCpuSets(GetCurrentProcess(), every.data(),
		                               static_cast<ULONG>(every.size())) != 0,
		      "SetProcessDefaultCpuSets (the other process)");
	} else {
		Check(SetProcessAffinityMask(GetCurrentProcess(),
		                             static_cast<DWORD_PTR>(layout.cp_mask | layout.general_mask)) != 0,
		      "SetProcessAffinityMask");
	}
	Common::MaintainCpuPlacement();
	const auto after = Common::CurrentCpuLayout();
	std::printf("  after the external rule (%s): %s\n", scenario, after.description.c_str());
	Check(!after.vcache && after.cold_mask == 0, "the V-cache placement was not dropped");
	Check(ThreadSets(cold_handle).empty(), "the cold thread kept its CPU sets");
	std::vector<ULONG> current(256);
	ULONG              count = 0;
	Check(GetProcessDefaultCpuSets(GetCurrentProcess(), current.data(),
	                               static_cast<ULONG>(current.size()), &count) != 0,
	      "GetProcessDefaultCpuSets");
	current.resize(count);
	std::sort(current.begin(), current.end());
	if (sets) {
		// As without the placement (no KYTY_CPU_RESERVE_REASSERT): the other process's sets stay and
		// the CP keeps its core.
		Check(current == every, "the other process's default CPU sets were replaced");
		Check(after.reserved && after.cp_mask == layout.cp_mask, "the CP lost its core");
	} else {
		// As without the placement: the KYTY_CPU_RESERVE layout within the new mask.
		std::vector<ULONG> general(after.general_ids.begin(), after.general_ids.end());
		std::sort(general.begin(), general.end());
		Check(after.reserved && ((after.cp_mask | after.general_mask) & ~(layout.cp_mask | layout.general_mask)) == 0 &&
		          current == general,
		      "the layout did not follow the new process mask");
	}
	phase = 2;
	while (phase.load() != 3) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	cold.join();
	CloseHandle(cold_handle);
	Common::MaintainCpuPlacement();
	Check(!Common::CurrentCpuLayout().vcache, "the V-cache placement came back after an external rule");
	if (!sets) {
		(void)SetProcessAffinityMask(GetCurrentProcess(), static_cast<DWORD_PTR>(original));
		Common::MaintainCpuPlacement();
		Check(!Common::CurrentCpuLayout().vcache,
		      "the V-cache placement came back after the mask was restored");
	}
	std::printf("  cold thread after the rule ran on 0x%llx\n",
	            static_cast<unsigned long long>(cold_after));
	std::puts("CpuPlacementTests: V-cache live ok");
	return 0;
}

// ---- --bench ----

int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// CP-like work (the SMT probe's proxy): hash-map lookups, a 10 KB copy, hashing, branchy dispatch.
struct Proxy {
	explicit Proxy(uint64_t seed, uint32_t entries) {
		std::mt19937_64 rng(seed);
		map.reserve(entries);
		for (uint32_t i = 0; i < entries; i++) {
			const auto key = rng();
			map[key]       = rng();
			keys.push_back(key);
		}
		std::shuffle(keys.begin(), keys.end(), rng);
		src.resize(10240);
		dst.resize(10240);
		for (auto& byte: src) {
			byte = static_cast<uint8_t>(rng());
		}
		ops.resize(1u << 16u);
		for (auto& op: ops) {
			op = static_cast<uint8_t>(rng() % 7);
		}
	}
	uint64_t Step(uint64_t i) {
		uint64_t acc = i;
		for (uint64_t j = 0; j < 64; j++) {
			const auto it = map.find(keys[(acc + j) % keys.size()]);
			acc += it != map.end() ? (it->second & 0xffff) : 1;
		}
		src[acc % src.size()] ^= static_cast<uint8_t>(acc);
		std::memcpy(dst.data(), src.data(), src.size());
		uint64_t hash = 1469598103934665603ull;
		for (size_t k = 0; k < 4096; k += 8) {
			uint64_t value = 0;
			std::memcpy(&value, dst.data() + k, 8);
			hash = (hash ^ value) * 1099511628211ull;
		}
		for (uint32_t j = 0; j < 256; j++) {
			switch (ops[(i * 256 + j) & 0xffff]) {
				case 0: acc += j; break;
				case 1: acc ^= hash >> (j & 31u); break;
				case 2: acc *= 3; break;
				case 3: acc -= hash; break;
				case 4: acc = (acc << 1u) | (acc >> 63u); break;
				case 5: acc += ops[j] * 7u; break;
				default: acc ^= 0x9e3779b97f4a7c15ull; break;
			}
		}
		return acc + hash;
	}
	std::unordered_map<uint64_t, uint64_t> map;
	std::vector<uint64_t>                  keys;
	std::vector<uint8_t>                   src;
	std::vector<uint8_t>                   dst;
	std::vector<uint8_t>                   ops;
};

HANDLE HighResolutionTimer() {
	constexpr DWORD HighResolution = 0x00000002; // CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
	HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, HighResolution, TIMER_ALL_ACCESS);
	if (timer == nullptr) {
		timer = CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
	}
	Check(timer != nullptr, "CreateWaitableTimerExW");
	return timer;
}

void SleepMicro(HANDLE timer, uint32_t us) {
	LARGE_INTEGER due {};
	due.QuadPart = -static_cast<LONGLONG>(us) * 10;
	(void)SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
	(void)WaitForSingleObject(timer, INFINITE);
}

double Percentile(std::vector<uint32_t> values, double p) {
	if (values.empty()) {
		return 0.0;
	}
	std::sort(values.begin(), values.end());
	return values[static_cast<size_t>(p * static_cast<double>(values.size() - 1))];
}

// One measurement in this process: the environment already holds KYTY_CPU_RESERVE.
int BenchChild(double seconds, uint32_t guests) {
	(void)SetProcessAffinityMask(GetCurrentProcess(), 0xFFFF); // Process Lasso's rule
	(void)SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
	Common::InitCpuPlacement();

	std::atomic<bool>     stop {false};
	std::atomic<bool>     measuring {false};
	std::atomic<uint64_t> guest_steps {0};
	std::atomic<uint64_t> polls {0};
	std::vector<std::thread> threads;
	// Guest jobs: bursts of 20-400 us of work, then 50-500 us asleep; half integer (the proxy),
	// half float streaming.
	for (uint32_t g = 0; g < guests; g++) {
		threads.emplace_back([&, g] {
			std::mt19937         rng(g + 1);
			Proxy                proxy(g + 1, 50000);
			std::vector<float>   a(1u << 18u, 1.0f);
			std::vector<float>   b(1u << 18u, 2.0f);
			const HANDLE         timer = HighResolutionTimer();
			uint64_t             steps = 0;
			uint64_t             acc   = 0;
			size_t               at    = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				const auto end = NowNs() + static_cast<int64_t>(20 + rng() % 381) * 1000;
				while (NowNs() < end) {
					if ((g & 1u) == 0) {
						acc += proxy.Step(steps);
					} else {
						for (size_t k = 0; k < 4096; k++, at = (at + 1) & (a.size() - 1)) {
							a[at] = a[at] * 0.999f + b[at] * 0.001f;
						}
					}
					if (measuring.load(std::memory_order_relaxed)) {
						steps++;
					}
				}
				SleepMicro(timer, 50 + rng() % 451);
			}
			g_sink = acc + static_cast<uint64_t>(a[7]);
			guest_steps.fetch_add(steps);
			CloseHandle(timer);
		});
	}
	// The guest fence poller: sched_yield in a loop.
	threads.emplace_back([&] {
		uint64_t count = 0;
		while (!stop.load(std::memory_order_relaxed)) {
			(void)SwitchToThread();
			if (measuring.load(std::memory_order_relaxed)) {
				count++;
			}
		}
		polls = count;
	});

	// The CP: a work chunk after each wake-up. The producer (GPU completions, guest submissions)
	// wakes it about 100 us after the previous chunk.
	const HANDLE          work = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	const HANDLE          done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
	std::atomic<int64_t>  signalled {0};
	std::vector<uint32_t> wake_ns;
	std::vector<uint32_t> work_ns;
	wake_ns.reserve(1u << 20u);
	work_ns.reserve(1u << 20u);
	uint64_t    chunks = 0;
	std::thread cp([&] {
		(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL); // CP level 1
		Common::PlaceCurrentThread(Common::ThreadRole::Cp);
		Proxy    proxy(99, 400000);
		uint64_t i   = 0;
		uint64_t acc = 0;
		for (;;) {
			(void)WaitForSingleObject(work, INFINITE);
			if (stop.load()) {
				break;
			}
			const auto woke = NowNs();
			for (uint32_t k = 0; k < 40; k++) {
				acc += proxy.Step(i++);
			}
			const auto finished = NowNs();
			if (measuring.load(std::memory_order_relaxed)) {
				wake_ns.push_back(static_cast<uint32_t>(std::min<int64_t>(woke - signalled.load(), UINT32_MAX)));
				work_ns.push_back(static_cast<uint32_t>(finished - woke));
				chunks++;
			}
			(void)SetEvent(done);
		}
		g_sink = acc;
	});
	std::thread producer([&] {
		(void)SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST); // service threads
		const HANDLE timer    = HighResolutionTimer();
		const auto   start    = NowNs();
		const auto   warm     = start + 500'000'000;
		const auto   end      = warm + static_cast<int64_t>(seconds * 1e9);
		bool         measured = false;
		for (auto now = start; now < end; now = NowNs()) {
			if (!measured && now >= warm) {
				measured = true;
				measuring.store(true);
			}
			SleepMicro(timer, 100);
			signalled.store(NowNs());
			(void)SetEvent(work);
			(void)WaitForSingleObject(done, INFINITE);
		}
		measuring.store(false);
		stop.store(true);
		(void)SetEvent(work);
		CloseHandle(timer);
	});
	producer.join();
	cp.join();
	for (auto& thread: threads) {
		thread.join();
	}
	double work_mean = 0.0;
	for (const auto value: work_ns) {
		work_mean += value;
	}
	work_mean = work_ns.empty() ? 0.0 : work_mean / static_cast<double>(work_ns.size());
	std::printf("RESULT chunks_per_s=%.1f work_p50_us=%.2f work_p99_us=%.2f work_mean_us=%.2f "
	            "wake_p50_us=%.2f wake_p99_us=%.2f wake_p999_us=%.2f guest_steps_per_s=%.0f "
	            "polls_per_s=%.0f\n",
	            static_cast<double>(chunks) / seconds, Percentile(work_ns, 0.5) / 1e3,
	            Percentile(work_ns, 0.99) / 1e3, work_mean / 1e3, Percentile(wake_ns, 0.5) / 1e3,
	            Percentile(wake_ns, 0.99) / 1e3, Percentile(wake_ns, 0.999) / 1e3,
	            static_cast<double>(guest_steps.load()) / seconds,
	            static_cast<double>(polls.load()) / seconds);
	std::fflush(stdout);
	return 0;
}

struct BenchResult {
	double chunks_per_s = 0, work_p50 = 0, work_p99 = 0, work_mean = 0, wake_p50 = 0, wake_p99 = 0,
	       wake_p999 = 0, guest_steps = 0, polls = 0;
};

bool RunChild(const std::wstring& exe, const char* mode, double seconds, uint32_t guests,
              BenchResult& result, std::string& placement) {
	(void)SetEnvironmentVariableA("KYTY_CPU_RESERVE", mode);
	SECURITY_ATTRIBUTES attributes {};
	attributes.nLength        = sizeof(attributes);
	attributes.bInheritHandle = TRUE;
	HANDLE read_pipe          = nullptr;
	HANDLE write_pipe         = nullptr;
	Check(CreatePipe(&read_pipe, &write_pipe, &attributes, 0) != 0, "CreatePipe");
	(void)SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);
	STARTUPINFOW startup {};
	startup.cb         = sizeof(startup);
	startup.dwFlags    = STARTF_USESTDHANDLES;
	startup.hStdOutput = write_pipe;
	startup.hStdError  = write_pipe;
	startup.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
	wchar_t arguments[512];
	(void)std::swprintf(arguments, 512, L"\"%ls\" --bench-child %.1f %u", exe.c_str(), seconds,
	                    guests);
	PROCESS_INFORMATION process {};
	const bool created = CreateProcessW(exe.c_str(), arguments, nullptr, nullptr, TRUE, 0, nullptr,
	                                    nullptr, &startup, &process) != 0;
	CloseHandle(write_pipe);
	if (!created) {
		CloseHandle(read_pipe);
		return false;
	}
	std::string output;
	char        buffer[4096];
	DWORD       read = 0;
	while (ReadFile(read_pipe, buffer, sizeof(buffer), &read, nullptr) != 0 && read != 0) {
		output.append(buffer, read);
	}
	(void)WaitForSingleObject(process.hProcess, INFINITE);
	CloseHandle(process.hProcess);
	CloseHandle(process.hThread);
	CloseHandle(read_pipe);
	const auto line = output.find("Kyty CPU placement");
	if (line != std::string::npos) {
		placement = output.substr(line, output.find('\n', line) - line);
	}
	const auto at = output.find("RESULT ");
	if (at == std::string::npos) {
		std::printf("bench child failed:\n%s\n", output.c_str());
		return false;
	}
	return std::sscanf(output.c_str() + at,
	                   "RESULT chunks_per_s=%lf work_p50_us=%lf work_p99_us=%lf work_mean_us=%lf "
	                   "wake_p50_us=%lf wake_p99_us=%lf wake_p999_us=%lf guest_steps_per_s=%lf "
	                   "polls_per_s=%lf",
	                   &result.chunks_per_s, &result.work_p50, &result.work_p99, &result.work_mean,
	                   &result.wake_p50, &result.wake_p99, &result.wake_p999, &result.guest_steps,
	                   &result.polls) == 9;
}

// Busy fraction of every processor over `ms` (GetSystemTimes).
double SystemBusy(uint32_t ms) {
	const auto ticks = [](FILETIME time) {
		return (static_cast<uint64_t>(time.dwHighDateTime) << 32u) | time.dwLowDateTime;
	};
	FILETIME idle0, kernel0, user0, idle1, kernel1, user1;
	(void)GetSystemTimes(&idle0, &kernel0, &user0);
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	(void)GetSystemTimes(&idle1, &kernel1, &user1);
	const auto idle  = ticks(idle1) - ticks(idle0);
	const auto total = (ticks(kernel1) - ticks(kernel0)) + (ticks(user1) - ticks(user0));
	return total == 0 ? 0.0 : 1.0 - static_cast<double>(idle) / static_cast<double>(total);
}

double Median(std::vector<double> values) {
	std::sort(values.begin(), values.end());
	return values.empty() ? 0.0 : values[values.size() / 2];
}

int Bench(double seconds, uint32_t reps) {
	wchar_t path[MAX_PATH];
	Check(GetModuleFileNameW(nullptr, path, MAX_PATH) != 0, "GetModuleFileNameW");
	const std::wstring exe = path;
	std::printf("bench: %.1f s per run after 0.5 s warm-up, %u repetitions (off, cp alternating), "
	            "logical processors 0-15, HIGH priority class; system busy before: %.1f%%\n",
	            seconds, reps, SystemBusy(1000) * 100.0);
	for (const uint32_t guests: {10u, 24u}) {
		std::vector<BenchResult> runs[2];
		std::string              placement;
		for (uint32_t rep = 0; rep < reps; rep++) {
			for (int mode = 0; mode < 2; mode++) {
				BenchResult result;
				std::string line;
				if (!RunChild(exe, mode == 0 ? "off" : "cp", seconds, guests, result, line)) {
					return 1;
				}
				if (mode == 1 && placement.empty()) {
					placement = line;
				}
				runs[mode].push_back(result);
				std::printf("  guests %2u rep %u %-3s: CP %6.0f chunks/s, chunk p50 %6.1f p99 %6.1f "
				            "mean %6.1f us, wake p50 %5.1f p99 %6.1f p99.9 %7.1f us, guest %9.0f "
				            "steps/s, polls %9.0f/s\n",
				            guests, rep, mode == 0 ? "off" : "cp", result.chunks_per_s,
				            result.work_p50, result.work_p99, result.work_mean, result.wake_p50,
				            result.wake_p99, result.wake_p999, result.guest_steps, result.polls);
				std::fflush(stdout);
			}
		}
		const auto median = [&](int mode, double BenchResult::*field) {
			std::vector<double> values;
			for (const auto& run: runs[mode]) {
				values.push_back(run.*field);
			}
			return Median(values);
		};
		const auto report = [&](const char* name, double BenchResult::*field) {
			const auto off = median(0, field);
			const auto cp  = median(1, field);
			std::printf("  guests %2u median %-18s off %10.2f  cp %10.2f  (%+.1f%%)\n", guests, name,
			            off, cp, off != 0.0 ? (cp / off - 1.0) * 100.0 : 0.0);
		};
		std::printf("  cp placement: %s\n", placement.c_str());
		report("CP chunks/s", &BenchResult::chunks_per_s);
		report("chunk p50 us", &BenchResult::work_p50);
		report("chunk p99 us", &BenchResult::work_p99);
		report("chunk mean us", &BenchResult::work_mean);
		report("wake p50 us", &BenchResult::wake_p50);
		report("wake p99 us", &BenchResult::wake_p99);
		report("wake p99.9 us", &BenchResult::wake_p999);
		report("guest steps/s", &BenchResult::guest_steps);
		report("polls/s", &BenchResult::polls);
	}
	std::printf("bench: system busy after: %.1f%%\n", SystemBusy(1000) * 100.0);
	return 0;
}

#endif

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
	if (argc >= 2 && std::strcmp(argv[1], "--bench-child") == 0) {
		const double   seconds = argc > 2 ? std::atof(argv[2]) : 3.0;
		const uint32_t guests  = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 10u;
		return BenchChild(seconds, guests);
	}
	if (argc >= 3 && std::strcmp(argv[1], "--vcache") == 0) {
		Profiler::Detail::g_event_sink.store(Profiler::Detail::CounterSink::Thread);
		TestVCache();
		return TestVCacheLive(argv[2]);
	}
	if (argc >= 2 && std::strcmp(argv[1], "--bench") == 0) {
		const double   seconds = argc > 2 ? std::atof(argv[2]) : 4.0;
		const uint32_t reps    = argc > 3 ? static_cast<uint32_t>(std::atoi(argv[3])) : 4u;
		return Bench(seconds, reps);
	}
#endif
	auto mode = CpuReserveMode::Cp;
	if (argc >= 3 && std::strcmp(argv[1], "--mode") == 0) {
		bool valid = false;
		mode       = Common::ParseCpuReserveMode(argv[2], &valid);
		Check(valid && mode != CpuReserveMode::Off, "--mode cp|cp+recorder");
	}
	// Before anything reads them (both are evaluated once).
	SetEnv("KYTY_CPU_RESERVE", mode == CpuReserveMode::Cp ? "cp" : "cp+recorder");
	SetEnv("KYTY_CPU_PLACEMENT_SAMPLES", "1");
	SetEnv("KYTY_CPU_RESERVE_CORE", "");
	SetEnv("KYTY_CPU_RESERVE_REASSERT", "1");
	SetEnv("KYTY_CPU_SETS", "");
	SetEnv("KYTY_VCACHE_PLACEMENT", "0"); // the KYTY_CPU_RESERVE behaviour alone (--vcache: with it)
	Profiler::Detail::g_event_sink.store(Profiler::Detail::CounterSink::Thread);

	TestLayouts();
	TestVCache();
	TestAffinityAndModes();
	Check(Common::GetCpuReserveMode() == mode, "KYTY_CPU_RESERVE");
#if defined(_WIN32)
	TestLive(mode);
#else
	std::puts("CpuPlacementTests: live placement skipped (Windows only)");
#endif
	std::puts("CpuPlacementTests: ok");
	return 0;
}
