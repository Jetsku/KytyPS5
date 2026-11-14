// The core budget (common/coreBudget.h), the budget spin gate (common/spinGate.h, KYTY_SPIN_BUDGET)
// and the small-CPU placement rules (common/cpuPlacement.h, KYTY_CPU_RESERVE_SCALE) on synthetic
// CPU set tables: this machine's 7950X3D (full, Process Lasso's mask and the masks the timing
// scripts use), 6 cores without SMT, 4 cores / 8 threads, an 8-core part without SMT and an Intel
// hybrid. The full 7950X3D must decide exactly as without the rules.

#include "common/coreBudget.h"
#include "common/cpuPlacement.h"
#include "common/spinGate.h"
#include "common/guestPriority.h"
#include "common/topologyRule.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using Common::CpuLayout;
using Common::CpuReserveMode;
using Common::CpuSetInfo;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CoreBudgetTests: failed: %s\n", text);
		std::fflush(stderr);
		std::abort();
	}
}

bool Contains(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

// Ryzen 9 7950X3D as GetSystemCpuSetInformation reports it (as CpuPlacementTests).
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

// `cores` cores without SMT, one L3, preferences given.
[[maybe_unused]] std::vector<CpuSetInfo> NoSmtTable(uint32_t cores, const std::vector<uint8_t>& classes) {
	std::vector<CpuSetInfo> sets;
	for (uint32_t lp = 0; lp < cores; lp++) {
		CpuSetInfo set;
		set.id               = 256 + lp;
		set.logical          = static_cast<uint8_t>(lp);
		set.core             = static_cast<uint8_t>(lp);
		set.scheduling_class = lp < classes.size() ? classes[lp] : 0;
		set.allowed          = true;
		sets.push_back(set);
	}
	return sets;
}

// Intel hybrid: 8 P-cores with SMT (logical 0-15, efficiency class 1) and 16 E-cores (16-31,
// class 0), one L3. E-cores get the higher scheduling classes here on purpose: the rules must
// still keep the CP on a P-core.
std::vector<CpuSetInfo> HybridTable() {
	std::vector<CpuSetInfo> sets;
	for (uint32_t lp = 0; lp < 32; lp++) {
		CpuSetInfo set;
		set.id               = 256 + lp;
		set.logical          = static_cast<uint8_t>(lp);
		set.core             = static_cast<uint8_t>(lp < 16 ? lp & ~1u : lp);
		set.efficiency_class = lp < 16 ? 1 : 0;
		set.scheduling_class = lp < 16 ? static_cast<uint8_t>(lp == 6 || lp == 7 ? 9 : 4) : 12;
		set.allowed          = true;
		sets.push_back(set);
	}
	return sets;
}

[[maybe_unused]] bool SameDecision(const CpuLayout& a, const CpuLayout& b) {
	return a.reserved == b.reserved && a.cp_core == b.cp_core && a.recorder_core == b.recorder_core &&
	       a.sequencer_core == b.sequencer_core && a.cp_ids == b.cp_ids &&
	       a.recorder_ids == b.recorder_ids && a.sequencer_ids == b.sequencer_ids &&
	       a.general_ids == b.general_ids && a.cp_mask == b.cp_mask &&
	       a.recorder_mask == b.recorder_mask && a.sequencer_mask == b.sequencer_mask &&
	       a.general_mask == b.general_mask && a.vcache == b.vcache && a.cold_ids == b.cold_ids &&
	       a.cold_mask == b.cold_mask && a.description == b.description;
}

[[maybe_unused]] Common::ReserveRules Scaled() {
	Common::ReserveRules rules;
	rules.scaled = true;
	return rules;
}

void TestBudget() {
	using Common::ComputeCoreBudget;
	{
		const auto b = ComputeCoreBudget(MachineTable(0xFFFFFFFF));
		Check(b.logical == 32 && b.cores == 16 && b.whole_cores == 16 && b.smt_cores == 16 &&
		          b.caches == 2 && b.largest_cache_logical == 16 && !b.hybrid &&
		          b.performance_logical == 32 && b.mask == 0xFFFFFFFFull,
		      "7950X3D full: 32 logical, 16 cores, 2 L3s");
	}
	{
		const auto b = ComputeCoreBudget(MachineTable(0x0FFF0000));
		Check(b.logical == 12 && b.cores == 6 && b.whole_cores == 6 && b.smt_cores == 6 &&
		          b.caches == 1,
		      "mask 0x0FFF0000: 6 cores / 12 logical");
		Check(Contains(b.description, "12 logical processors on 6 cores"), "description");
	}
	{
		const auto b = ComputeCoreBudget(MachineTable(0x05550000));
		Check(b.logical == 6 && b.cores == 6 && b.whole_cores == 0 && b.smt_cores == 0,
		      "mask 0x05550000: 6 cores without SMT (one logical of each)");
	}
	{
		const auto b = ComputeCoreBudget(MachineTable(0x00FF0000));
		Check(b.logical == 8 && b.cores == 4 && b.smt_cores == 4, "mask 0x00FF0000: 4 cores / 8 threads");
	}
	{
		const auto b = ComputeCoreBudget(MachineTable(0x7));
		Check(b.logical == 3 && b.cores == 2 && b.whole_cores == 1 && b.smt_cores == 1,
		      "odd mask 0x7: one whole core and half of another");
	}
	{
		const auto b = ComputeCoreBudget(HybridTable());
		Check(b.hybrid && b.top_efficiency == 1 && b.performance_logical == 16 &&
		          b.performance_cores == 8 && b.cores == 24 && b.logical == 32 &&
		          b.performance_mask == 0xFFFFull,
		      "hybrid: 8 P-cores (16 logical) of 24 cores");
		Check(Contains(b.description, "hybrid"), "hybrid description");
	}
	{
		const auto b = ComputeCoreBudget(MachineTable(0));
		Check(!b.Valid() && b.logical == 0, "nothing allowed");
	}
	Check(Common::ParseTriState(nullptr) == -1 && Common::ParseTriState("") == -1 &&
	          Common::ParseTriState("0") == 0 && Common::ParseTriState("1") == 1 &&
	          Common::ParseTriState("on") == 1,
	      "tri-state parse");
	Check(!Common::EffectiveSwitch(-1, false) && Common::EffectiveSwitch(-1, true) &&
	          !Common::EffectiveSwitch(0, true) && Common::EffectiveSwitch(1, false),
	      "explicit values win over the profile, unset follows it");
	std::puts("CoreBudgetTests: budget ok");
}

void TestGate() {
	using namespace Common::SpinGate;
	// One interval of 1,000,000 ticks.
	constexpr uint64_t I = 1'000'000;
	{
		// 6-core mask, Sky Garden-like: 8 game/emulator threads busy, 6 idle workers spinning
		// 20 %, sequencer and recorder spinning 60 % / 45 % beside little work.
		std::vector<ThreadSample> samples;
		for (int i = 0; i < 8; i++) {
			samples.push_back({I * 7 / 10, 0, i == 0}); // commit thread registered, busy
		}
		for (int i = 0; i < 6; i++) {
			samples.push_back({I / 4, I / 5, true}); // 5 % work, 20 % spin
		}
		samples.push_back({I * 7 / 10, I * 6 / 10, true}); // sequencer: 10 % work
		samples.push_back({I / 2, I * 45 / 100, true});    // recorder
		samples.push_back({I / 100, 0, false});            // an idle game thread
		const auto d = ClassifyThreads(samples.data(), static_cast<uint32_t>(samples.size()), I, 0.25);
		Check(d.busy == 8 && d.spinning == 8 && d.idle_spinners == 8, "6C: 8 busy, 8 spinning");
		GateInput in;
		in.budget_logical = 12;
		in.busy           = d.busy;
		in.spinning       = d.spinning;
		in.idle_spinners  = d.idle_spinners;
		Check(!NextBudgetGateOpen(true, in), "6C: 8 + 8 > 12 closes the gate");
		// Closed, the spinners back off: their spin falls under 5 %, the gate still stays shut.
		in.spinning = 0;
		Check(!NextBudgetGateOpen(false, in), "6C closed: 8 + 8 idle spinners + 1 > 12 stays shut");
		Check(NextBudgetGateOpen(true, in), "...and an open gate with nobody spinning stays open");
	}
	{
		GateInput in;
		in.budget_logical = 32;
		in.busy           = 12;
		in.spinning       = 9;
		in.idle_spinners  = 9;
		Check(NextBudgetGateOpen(true, in), "16C: 12 + 9 fits in 32");
		Check(NextBudgetGateOpen(false, in), "16C closed: reopens (12 + 9 + 1 <= 32)");
	}
	{
		// Hysteresis: no oscillation when only the spin share changes.
		GateInput in;
		in.budget_logical = 16;
		in.busy           = 10;
		in.spinning       = 7;
		in.idle_spinners  = 7;
		bool open = true;
		open      = NextBudgetGateOpen(open, in);
		Check(!open, "8C: 10 + 7 > 16 closes");
		in.spinning = 0;
		for (int i = 0; i < 10; i++) {
			open = NextBudgetGateOpen(open, in);
		}
		Check(!open, "8C: backing off never reopens it");
		in.busy = 8;
		open    = NextBudgetGateOpen(open, in);
		Check(open, "8C: the game's load falls: 8 + 7 + 1 <= 16 reopens");
		in.margin = 2;
		in.spinning = 7;
		Check(!NextBudgetGateOpen(true, in), "margin counts");
	}
	{
		GateInput in; // no budget read yet: never closes
		in.busy = 100;
		Check(NextBudgetGateOpen(true, in), "no budget: open");
		const ThreadSample s {500, 900, true};
		const auto d = ClassifyThreads(&s, 1, 1000, 0.25);
		Check(d.busy == 0 && d.spinning == 1 && d.spin_cores == 0.5,
		      "spin above the cycles is clamped to them");
		Check(ClassifyThreads(&s, 1, 0, 0.25).busy == 0, "empty interval");
	}
	{
		// Threads the scan does not measure: the process total minus the measured ones.
		Demand d;
		d.busy = 3;
		AddUnmeasured(d, 5 * I, 2 * I + I / 4, I); // 2.75 cores unexplained
		Check(d.busy == 6 && d.busy_cores > 2.74 && d.busy_cores < 2.76, "unmeasured: 2.75 cores = 3 busy threads");
		AddUnmeasured(d, I, 2 * I, I);
		Check(d.busy == 6, "unmeasured: measured above the total adds nothing");
	}
	{
		// Hysteresis. The 6-core trace that flapped: 250 ms intervals, demand above the budget in
		// one interval of three (busy 10 + spinning 4 against 12), below it otherwise (busy 6, 3
		// idle spinners). One decision per interval flips the gate twice every 750 ms.
		GateInput high;
		high.budget_logical = 12;
		high.busy           = 10;
		high.spinning       = 4;
		high.idle_spinners  = 4;
		GateInput low;
		low.budget_logical = 12;
		low.busy           = 6;
		low.spinning       = 1;
		low.idle_spinners  = 3;
		uint32_t raw_changes = 0;
		bool     raw         = true;
		for (int i = 0; i < 240; i++) {
			const bool next = NextBudgetGateOpen(raw, i % 3 == 0 ? high : low);
			raw_changes += next != raw ? 1u : 0u;
			raw = next;
		}
		Check(raw_changes >= 100, "the raw decision flaps on the trace");
		GateFilter filter(GateConfig {});
		bool       open      = true;
		uint64_t   last_flip = 0;
		uint32_t   changes   = 0;
		bool       holds_ok  = true;
		for (int i = 1; i <= 240; i++) { // 60 s
			const uint64_t now  = static_cast<uint64_t>(i) * 250u;
			const bool     next = filter.Step(now, i % 3 == 0 ? high : low);
			if (next != open) {
				holds_ok &= now - last_flip >= (open ? 1000u : 2000u) || last_flip == 0;
				last_flip = now;
				changes++;
				open = next;
			}
		}
		Check(holds_ok, "filter: every state is held its minimum time");
		Check(changes <= 1, "filter: the flapping trace changes the state at most once (mean demand)");
		// Sustained demand above the budget closes it within the window, and a sustained fall reopens
		// it only after the closed hold.
		GateFilter sustained(GateConfig {});
		uint64_t   closed_at = 0;
		uint64_t   opened_at = 0;
		for (int i = 1; i <= 80; i++) {
			const uint64_t now = 1000u + static_cast<uint64_t>(i) * 250u;
			const bool     o   = sustained.Step(now, i <= 20 ? high : low);
			if (!o && closed_at == 0) {
				closed_at = now;
			}
			if (o && closed_at != 0 && opened_at == 0) {
				opened_at = now;
			}
		}
		Check(closed_at != 0 && closed_at <= 1000u + 250u * 4u, "filter: sustained high closes within the window");
		Check(opened_at != 0 && opened_at - closed_at >= 2000u, "filter: reopens after the closed hold");
		Check(sustained.Closes() == 1, "filter: one close");
	}
	// The runtime gate with the switch unset and no profile: spins as wanted, counts.
	Check(!BudgetEnabled() && !BudgetClosed(), "KYTY_SPIN_BUDGET unset: off");
	Check(SpinNs(Site::Sequencer, 200'000) == 200'000, "off: the wanted spin");
	Check(SpinNs(Site::CommitHead, UINT64_MAX) == UINT64_MAX, "off: unbounded stays unbounded");
	NoteSpin(Site::Sequencer, 1234);
	NotePark(Site::Sequencer);
	Check(std::string(SiteName(Site::Resolver)) == "resolver", "site names");
	std::puts("CoreBudgetTests: spin gate ok");
}

void TestPlacement() {
	using Common::ComputeCpuLayout;
	using Common::ComputeVCacheLayout;
	using Common::DetectVCache;
	const auto rules = Scaled();
	// 1. The full 7950X3D decides exactly as before: every mode, every forced CP processor, with and
	// without the V-cache tie preference, and the V-cache layout. Process Lasso's mask and the
	// 8-core halves too in cp mode (the shipped preset); cp+recorder and stages need 9 and 10 cores
	// with the rules, so on 8 cores they fall back to cp (tested below).
	const auto topology = DetectVCache({{0, 0xFFFF, 96ull << 20u}, {0, 0xFFFF0000, 32ull << 20u}});
	for (const uint64_t mask: {0xFFFFFFFFull, 0xFFFFull, 0xFFFF0000ull}) {
		const auto sets = MachineTable(mask);
		const bool full = mask == 0xFFFFFFFFull;
		for (const auto mode: {CpuReserveMode::Cp, CpuReserveMode::CpRecorder, CpuReserveMode::Stages,
		                       CpuReserveMode::Off}) {
			if (!full && (mode == CpuReserveMode::CpRecorder || mode == CpuReserveMode::Stages)) {
				continue;
			}
			for (int forced = -1; forced < 34; forced++) {
				for (const uint64_t preferred: {uint64_t {0}, uint64_t {0xFFFF}}) {
					const auto legacy = ComputeCpuLayout(sets, mode, forced, Common::kMinGeneralProcessors, preferred);
					const auto scaled = ComputeCpuLayout(sets, mode, forced, Common::kMinGeneralProcessors, preferred, rules);
					Check(SameDecision(legacy, scaled), "7950X3D: unchanged decision (ComputeCpuLayout)");
				}
				const auto legacy = ComputeVCacheLayout(sets, mode, forced, topology);
				const auto scaled = ComputeVCacheLayout(sets, mode, forced, topology, Common::kMinGeneralProcessors, rules);
				Check(SameDecision(legacy, scaled), "7950X3D: unchanged decision (ComputeVCacheLayout)");
			}
		}
	}
	{
		const auto full = ComputeVCacheLayout(MachineTable(0xFFFFFFFF), CpuReserveMode::Cp, -1, topology,
		                                      Common::kMinGeneralProcessors, rules);
		Check(full.vcache && full.cp_mask == 0xC && full.cold_mask == 0xFFFF0000ull,
		      "7950X3D full, rules on: CP on 2-3, cold 16-31 as today");
	}
	// 2. 6 cores / 12 threads (mask 0x0FFF0000): the CP gets one logical processor, its sibling
	// stays general.
	{
		const auto sets   = MachineTable(0x0FFF0000);
		const auto legacy = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1);
		Check(legacy.reserved && legacy.cp_mask == 0xC0000ull, "6C legacy: whole core 18-19");
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(layout.reserved && layout.cp_mask == (1ull << 18u) && layout.cp_ids.size() == 1 &&
		          layout.cp_core == 18 && std::popcount(layout.general_mask) == 11 &&
		          (layout.general_mask & (1ull << 19u)) != 0,
		      "6C rules: CP alone on logical 18, sibling 19 general, 11 general");
		Check(Contains(layout.description, "one logical processor"), "6C: description");
		const auto both = ComputeCpuLayout(sets, CpuReserveMode::CpRecorder, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(both.reserved && both.recorder_ids.empty() && std::popcount(both.cp_mask) == 1,
		      "6C rules, cp+recorder: falls back to cp, one logical");
		const auto forced = ComputeCpuLayout(sets, CpuReserveMode::Cp, 21, Common::kMinGeneralProcessors, 0, rules);
		Check(forced.reserved && forced.cp_mask == (1ull << 21u), "6C rules: KYTY_CPU_RESERVE_CORE=21");
		Check(!ComputeCpuLayout(sets, CpuReserveMode::Cp, 3, Common::kMinGeneralProcessors, 0, rules).reserved,
		      "6C rules: a forced processor outside the mask reserves nothing");
	}
	// 3. 4 cores / 8 threads (0x00FF0000): one logical (legacy took a whole core, a quarter).
	{
		const auto sets   = MachineTable(0x00FF0000);
		Check(std::popcount(ComputeCpuLayout(sets, CpuReserveMode::Cp, -1).cp_mask) == 2,
		      "4C/8T legacy: a whole core");
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(layout.reserved && layout.cp_mask == (1ull << 18u) && std::popcount(layout.general_mask) == 7,
		      "4C/8T rules: one logical, 7 general");
	}
	// 4. 6 cores without SMT (0x05550000): nothing.
	{
		const auto sets   = MachineTable(0x05550000);
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(!layout.reserved && layout.cp_ids.empty() && Contains(layout.description, "nothing reserved"),
		      "6C/6T rules: nothing reserved");
		const auto native = ComputeCpuLayout(NoSmtTable(6, {3, 5, 4, 1, 2, 0}), CpuReserveMode::Cp, -1,
		                                     Common::kMinGeneralProcessors, 0, rules);
		Check(!native.reserved, "a native 6-core part without SMT: nothing reserved");
	}
	// 5. 8 cores without SMT: a whole core (one logical), CPPC best core even if it holds logical 0.
	{
		const auto sets   = NoSmtTable(8, {9, 5, 7, 3, 2, 1, 4, 6});
		const auto legacy = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, 6);
		Check(legacy.reserved && legacy.cp_core == 2, "8C/8T legacy: core 0 avoided, core 2");
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, 6, 0, rules);
		Check(layout.reserved && layout.cp_core == 0 && layout.cp_mask == 1 &&
		          std::popcount(layout.general_mask) == 7,
		      "8C/8T rules: the preferred core 0 (logical 0 only loses ties)");
		auto tie       = NoSmtTable(8, {9, 5, 9, 3, 2, 1, 4, 6});
		const auto tl  = ComputeCpuLayout(tie, CpuReserveMode::Cp, -1, 6, 0, rules);
		Check(tl.cp_core == 2, "8C/8T rules, tie with core 0: core 2");
	}
	// 6. Core 0 most preferred on the Lasso mask: the CP moves there only with the rules on.
	{
		auto sets                = MachineTable(0xFFFF);
		sets[0].scheduling_class = 15;
		sets[1].scheduling_class = 15;
		Check(ComputeCpuLayout(sets, CpuReserveMode::Cp, -1).cp_core == 2, "legacy: core 0 last");
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(layout.cp_core == 0 && layout.cp_mask == 0x3, "rules: the CPPC best core 0");
		const auto rec = ComputeCpuLayout(sets, CpuReserveMode::CpRecorder, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(rec.reserved && rec.cp_core == 0 && rec.recorder_core == -1 && rec.recorder_ids.empty() &&
		          Contains(rec.description, "CP core only"),
		      "rules, cp+recorder on 8 cores: falls back to cp only");
	}
	// 7. Hybrid: never an E-core, even with a higher scheduling class.
	{
		const auto sets   = HybridTable();
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(layout.reserved && layout.cp_core == 6 && layout.cp_mask == 0xC0,
		      "hybrid rules: the best P-core 6 (logical 6-7)");
		Check(std::popcount(layout.general_mask) == 30, "hybrid rules: E-cores stay general");
		auto p_only = sets;
		for (auto& set: p_only) {
			set.allowed = set.logical >= 16 || set.logical < 8; // 4 P-cores + 16 E-cores
		}
		const auto few = ComputeCpuLayout(p_only, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(few.reserved && (few.cp_mask & 0xFFFF0000ull) == 0, "hybrid rules: still a P-core");
		const auto legacy = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1);
		Check(legacy.reserved && (legacy.cp_mask & 0xFFFF0000ull) != 0,
		      "hybrid legacy picks an E-core by scheduling class (the gap the rules close)");
	}
	// 8. An odd mask with no whole core: one logical processor.
	{
		const auto sets   = MachineTable(0x5555);
		Check(!ComputeCpuLayout(sets, CpuReserveMode::Cp, -1).reserved, "legacy: no whole core, nothing");
		const auto layout = ComputeCpuLayout(sets, CpuReserveMode::Cp, -1, Common::kMinGeneralProcessors, 0, rules);
		Check(layout.reserved && layout.cp_mask == 0x4 && std::popcount(layout.general_mask) == 7,
		      "odd mask 0x5555 (8 cores, 8 logical): one logical, logical 2");
	}
	std::puts("CoreBudgetTests: placement rules ok");
}

// KYTY_SMALL_CPU_RULE (common/topologyRule.h) and KYTY_GUEST_PRIORITY_FLAT (common/guestPriority.h).
std::string RuleFor(uint64_t mask) {
	const auto budget = Common::ComputeCoreBudget(MachineTable(mask));
	return Common::TopologyRule::Describe(
	    Common::TopologyRule::Select(budget.cores, budget.smt_cores, budget.logical));
}

void TestSmallCpuRule() {
	using Common::GuestPriority::BandHost;
	Check(BandHost(0, 0) == 0 && BandHost(-1, 0) == -1 && BandHost(-2, 0) == -2, "flat 0: bands unchanged");
	Check(BandHost(0, 1) == -1 && BandHost(-1, 1) == -1 && BandHost(-2, 1) == -2, "flat 1: high band down");
	Check(BandHost(0, 2) == 0 && BandHost(-1, 2) == 0 && BandHost(-2, 2) == -2, "flat 2: normal band up");

	// The full machine and 8C/16T: nothing changes.
	Check(RuleFor(0xFFFFFFFFull) == "nothing", "rule: 16C/32T changes nothing");
	Check(RuleFor(0xFFFF) == "nothing", "rule: 8C/16T changes nothing");
	Check(RuleFor(0xFFFF0000ull) == "nothing", "rule: 8C/16T on the other CCD changes nothing");
	// 6C/12T (both CCDs' masks as the timing scripts use them).
	const char* six = "KYTY_GUEST_PRIORITY_FLAT=1, KYTY_SPIN_BUDGET=1";
	Check(RuleFor(0x0FFF) == six, "rule: 6C/12T");
	Check(RuleFor(0x0FFF0000ull) == six, "rule: 6C/12T other CCD");
	// 4C/8T and 6 cores without SMT: flat bands only.
	const char* few = "KYTY_GUEST_PRIORITY_FLAT=1";
	Check(RuleFor(0x0555) == few, "rule: 6 cores without SMT");
	Check(RuleFor(0x05550000ull) == few, "rule: 6 cores without SMT other CCD");
	Check(RuleFor(0x00FF) == few, "rule: 4C/8T");
	Check(RuleFor(0x00FF0000ull) == few, "rule: 4C/8T other CCD");
	// 5C/10T: 10 logical processors still get the spin budget.
	Check(RuleFor(0x03FF) == six, "rule: 5C/10T");
	Check(Common::TopologyRule::Select(0, 0, 0).empty(), "rule: no budget, nothing");
	std::puts("CoreBudgetTests: small-CPU rule ok");
}

} // namespace

int main() {
	TestBudget();
	TestGate();
	TestPlacement();
	TestSmallCpuRule();
	std::puts("CoreBudgetTests: all ok");
	return 0;
}
