// The core budget (common/coreBudget.h), the budget spin gate (common/spinGate.h, KYTY_SPIN_BUDGET)
// and the small-CPU placement rules (common/cpuPlacement.h, KYTY_CPU_RESERVE_SCALE) on synthetic
// CPU set tables: this machine's 7950X3D (full, Process Lasso's mask and the masks the timing
// scripts use), 6 cores without SMT, 4 cores / 8 threads, an 8-core part without SMT and an Intel
// hybrid. The full 7950X3D must decide exactly as without the rules.

#include "common/coreBudget.h"
#include "common/cpuPlacement.h"
#include "common/spinGate.h"

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
	// The runtime gate with the switch unset and no profile: spins as wanted, counts.
	Check(!BudgetEnabled() && !BudgetClosed(), "KYTY_SPIN_BUDGET unset: off");
	Check(SpinNs(Site::Sequencer, 200'000) == 200'000, "off: the wanted spin");
	Check(SpinNs(Site::CommitHead, UINT64_MAX) == UINT64_MAX, "off: unbounded stays unbounded");
	NoteSpin(Site::Sequencer, 1234);
	NotePark(Site::Sequencer);
	Check(std::string(SiteName(Site::Resolver)) == "resolver", "site names");
	std::puts("CoreBudgetTests: spin gate ok");
}

} // namespace

int main() {
	TestBudget();
	TestGate();
	std::puts("CoreBudgetTests: all ok");
	return 0;
}
