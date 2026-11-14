#ifndef EMULATOR_SRC_COMMON_TOPOLOGYRULE_H_
#define EMULATOR_SRC_COMMON_TOPOLOGYRULE_H_

#include <cstdint>
#include <string>
#include <vector>

// KYTY_SMALL_CPU_RULE=1 (start-up, default off): switches chosen from the processors this process may
// use (the core budget, common/coreBudget.h: physical cores, cores with SMT, logical processors),
// for CPUs smaller than the 16-core machine the defaults were tuned on. Each line of the table in
// topologyRule.cpp names the measurement it comes from (smallcpu, Astro Bot / Astro's Playroom,
// 2026-11-14). The rule's values override the environment and the presets for its keys only; a
// live file (KYTY_LIVE_FILE) can still change every live key afterwards. One "Kyty small-CPU rule"
// line is logged at start-up, also when it selects nothing.
namespace Common::TopologyRule {

struct Setting {
	std::string key;
	std::string value;
	std::string why;
};

// Pure (tested): the settings for a budget of `cores` physical cores, `smt_cores` of them with two
// or more allowed logical processors, and `logical` allowed logical processors. Empty: change nothing.
[[nodiscard]] std::vector<Setting> Select(uint32_t cores, uint32_t smt_cores, uint32_t logical);

// "KEY=VALUE, KEY=VALUE" (or "nothing").
[[nodiscard]] std::string Describe(const std::vector<Setting>& settings);

// Called by main before the CPU placement starts (InitializeThreads): with KYTY_SMALL_CPU_RULE=1 it
// reads the budget, sets the selected variables in the process environment (start-up readers),
// re-reads the live switches among them (Live::ReloadFromEnvironment), and logs the line.
void ApplyAtStartup();

} // namespace Common::TopologyRule

#endif // EMULATOR_SRC_COMMON_TOPOLOGYRULE_H_
