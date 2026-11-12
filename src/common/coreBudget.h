#ifndef EMULATOR_SRC_COMMON_COREBUDGET_H_
#define EMULATOR_SRC_COMMON_COREBUDGET_H_

#include "common/cpuPlacement.h"

#include <cstdint>
#include <string>
#include <vector>

// The core budget: the processors this process may actually use, as one source of truth for the
// spin gate (common/spinGate.h, KYTY_SPIN_BUDGET), the small-CPU placement rules
// (common/cpuPlacement.h, KYTY_CPU_RESERVE_SCALE) and the draw-prep worker bound
// (KYTY_DRAW_PREP_ADAPT).
//
// It is computed from the CPU set table (GetSystemCpuSetInformation) with `allowed` as the
// placement reads it: the process affinity mask, processors not allocated to another process, and
// KYTY_CPU_SETS or another process's default CPU sets (Process Lasso). The emulator's own default
// CPU sets (the KYTY_CPU_RESERVE layout) do not shrink it: the reserved core is still the
// process's. Group 0 only, as the placement.
//
// KYTY_CORE_BUDGET=1 (live, default off): the worker bound and Common::ProcessAffinityProcessorCount
// use the budget's logical processors (affinity and CPU sets) instead of the affinity mask's count,
// and a "Kyty core budget" line logs the budget whenever it changes. The spin gate and the
// placement rules use the budget whenever their own switches are on.
//
// KYTY_CORE_BUDGET_PROFILE=1 (live, default off): the "budget profile". Every core-budget switch
// left unset takes the profile's value: KYTY_CORE_BUDGET, KYTY_SPIN_BUDGET, KYTY_CPU_RESERVE_SCALE
// and KYTY_DRAW_PREP_ADAPT. A switch set explicitly (also to 0) keeps its own value. Meant to become
// the default later.
//
// Hybrid CPUs (Intel P/E cores): the efficiency class of each processor (higher = faster class).
// The budget counts the processors of the top class separately; the CP is never placed on a lower
// class with KYTY_CPU_RESERVE_SCALE.
namespace Common {

struct CoreBudget {
	uint32_t logical     = 0; // allowed logical processors
	uint32_t cores       = 0; // physical cores with at least one allowed logical processor
	uint32_t whole_cores = 0; // physical cores whose every logical processor is allowed
	uint32_t smt_cores   = 0; // physical cores with two or more allowed logical processors
	uint32_t caches      = 0; // last-level cache groups holding an allowed processor
	uint32_t largest_cache_logical = 0; // allowed processors of the fullest such cache
	bool     hybrid      = false;       // allowed processors of more than one efficiency class
	uint8_t  top_efficiency      = 0;   // the fastest allowed efficiency class
	uint32_t performance_logical = 0;   // allowed processors of that class
	uint32_t performance_cores   = 0;   // physical cores of that class with an allowed processor
	uint64_t mask             = 0;      // the allowed logical processors (group 0)
	uint64_t performance_mask = 0;      // of the top efficiency class
	std::string description;

	[[nodiscard]] bool Valid() const noexcept { return logical != 0; }
};

// Pure (tested): the budget of a CPU set table (group 0, `allowed` processors).
[[nodiscard]] CoreBudget ComputeCoreBudget(const std::vector<CpuSetInfo>& sets);

// Tri-state switch text: unset (nullptr or empty) -1, "0" 0, anything else 1. The core-budget
// switches use it so that unset follows KYTY_CORE_BUDGET_PROFILE.
[[nodiscard]] int64_t ParseTriState(const char* value);
// A tri-state value with the profile applied: explicit 0/1 win, unset (-1) is the profile.
[[nodiscard]] bool EffectiveSwitch(int64_t value);
// Pure (tested): the same with the profile given.
[[nodiscard]] constexpr bool EffectiveSwitch(int64_t value, bool profile) {
	return value < 0 ? profile : value != 0;
}

[[nodiscard]] bool CoreBudgetProfileOn(); // KYTY_CORE_BUDGET_PROFILE (live)
[[nodiscard]] bool CoreBudgetEnabled();   // KYTY_CORE_BUDGET, profile applied (live)
// Called on the command processor's thread after a live change of KYTY_CORE_BUDGET_PROFILE (at most
// 8 listeners; registration before the change).
void OnCoreBudgetProfileChange(void (*listener)(bool on));

// The budget now. Re-read from the system at most once a second (whoever calls first after that
// does it; cheap otherwise). `force`: read now.
[[nodiscard]] CoreBudget CurrentCoreBudget(bool force = false);
// The budget's logical processors (0 until it was read once; reads it then). One relaxed load
// after the first second.
[[nodiscard]] uint32_t CoreBudgetLogical();

} // namespace Common

#endif // EMULATOR_SRC_COMMON_COREBUDGET_H_
