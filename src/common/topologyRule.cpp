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
	(void)cores;
	(void)smt_cores;
	// Below 16 logical processors the game's high-band job workers (Astro Bot: Havok) spin on as many
	// processors as the game gets. 4C/8T (6 general processors) and 6 cores without SMT: they held
	// every processor while the normal-band threads they wait for stayed READY, 1.5-4 fps in stalls
	// of 2.5 s; flat bands: 26.6 and 33.3 fps, the stall came back when the switch went off.
	out.push_back({"KYTY_GUEST_PRIORITY_FLAT", "1", "game high band = normal band (spinning job workers)"});
	if (logical >= 10) {
		// 6C/12T: the helpers' spin (2.3 lp) is what the READY game threads lack; the spin gate frees
		// it without slowing the commit thread (Sky Garden +3.4 fps). Not below 10: at 6 logical
		// processors the core-budget profile doubled the commit thread's time.
		out.push_back({"KYTY_SPIN_BUDGET", "1", "helpers spin only on processors nobody else wants"});
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
	for (const auto& setting: settings) {
#if defined(_WIN32)
		(void)_putenv_s(setting.key.c_str(), setting.value.c_str());
#else
		(void)setenv(setting.key.c_str(), setting.value.c_str(), 1);
#endif
		// Live switches registered before main and read the environment then.
		(void)Live::ReloadFromEnvironment(setting.key.c_str());
	}
	std::printf("Kyty small-CPU rule (KYTY_SMALL_CPU_RULE): %s: %s\n", budget.description.c_str(),
	            Describe(settings).c_str());
	for (const auto& setting: settings) {
		std::printf("  %s=%s: %s\n", setting.key.c_str(), setting.value.c_str(), setting.why.c_str());
	}
	std::fflush(stdout);
}

} // namespace Common::TopologyRule
