#include "common/topologyRule.h"

#include "common/coreBudget.h"
#include "common/cpuPlacement.h"
#include "common/liveSwitch.h"

#include <cstdio>
#include <cstdlib>

namespace Common::TopologyRule {

std::vector<Setting> Select(uint32_t cores, uint32_t smt_cores, uint32_t logical) {
	std::vector<Setting> out;
	if (logical == 0 || logical >= 16) {
		// 16 logical processors and more: the shipped defaults (measured on 8C/16T and 16C/32T).
		return out;
	}
	(void)smt_cores;
	// Below 16 logical processors the game's high-band job workers spin on as many processors as the
	// game gets; at 4C/8T with a reserved core they held all six general processors while the
	// normal-band threads they wait for were READY for seconds (stalls of 2.5 s).
	out.push_back({"KYTY_GUEST_PRIORITY_FLAT", "1", "game high band = normal band"});
	// 6C/12T: the helpers' spin (2.3 lp) is what the starved game threads lack; the spin gate frees it
	// without slowing the commit thread.
	out.push_back({"KYTY_SPIN_BUDGET", "1", "spin only on processors nobody else wants"});
	if (cores < 6) {
		// 4C/8T: a whole core for the commit thread leaves 6 processors for everything else.
		out.push_back({"KYTY_CPU_RESERVE", "off", "no core reserved below 6 cores"});
	}
	return out;
}

std::string Describe(const std::vector<Setting>& settings) {
	if (settings.empty()) {
		return "nothing";
	}
	std::string text;
	for (const auto& setting: settings) {
		if (!text.empty()) {
			text += ", ";
		}
		text += setting.key + "=" + setting.value;
	}
	return text;
}

void ApplyAtStartup() {
	const char* on = std::getenv("KYTY_SMALL_CPU_RULE");
	if (on == nullptr || on[0] == '\0' || (on[0] == '0' && on[1] == '\0')) {
		return;
	}
	uint64_t   affinity = 0;
	const auto budget   = ComputeCoreBudget(ReadProcessCpuSets(&affinity));
	const auto settings = Select(budget.cores, budget.smt_cores, budget.logical);
	std::string staged;
	for (const auto& setting: settings) {
#if defined(_WIN32)
		(void)_putenv_s(setting.key.c_str(), setting.value.c_str());
#else
		(void)setenv(setting.key.c_str(), setting.value.c_str(), 1);
#endif
		staged += setting.key + "=" + setting.value + "\n";
	}
	if (!staged.empty()) {
		// Live switches registered before main and read the environment then: stage the same values
		// (applied at the first guest flip, as a live file's would be).
		Live::Testing::StageText(staged);
	}
	std::printf("Kyty small-CPU rule (KYTY_SMALL_CPU_RULE): %s: %s\n", budget.description.c_str(),
	            Describe(settings).c_str());
	for (const auto& setting: settings) {
		std::printf("  %s=%s: %s\n", setting.key.c_str(), setting.value.c_str(), setting.why.c_str());
	}
	std::fflush(stdout);
}

} // namespace Common::TopologyRule
