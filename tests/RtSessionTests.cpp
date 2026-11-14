#include "graphics/host_gpu/renderer/pipeline/rtSession.h"

#include <cstdint>
#include <cstdio>

// rtSession.h: what a dispatch of a background-built kernel does, and how long its first dispatch
// may wait for the pipeline.
int main() {
	using namespace Libs::Graphics::RtSession;
	constexpr uint32_t Kernel = 120000, Other = 250, Forever = UINT32_MAX;

	// A direct RT kernel waits (its skipped dispatches can leave lighting never recomputed); a
	// call-split program never waits; a huge compute program keeps the short budget.
	if (FirstWaitBudgetMs(KernelClass::Direct, Kernel, Other) != Kernel) return 1;
	if (FirstWaitBudgetMs(KernelClass::CallSplit, Kernel, Other) != 0) return 2;
	if (FirstWaitBudgetMs(KernelClass::Huge, Kernel, Other) != Other) return 3;
	// KYTY_RT_PIPELINE_ASYNC=0: both budgets are for ever; call-split programs still never wait.
	if (FirstWaitBudgetMs(KernelClass::Direct, Forever, Forever) != Forever) return 4;
	if (FirstWaitBudgetMs(KernelClass::CallSplit, Forever, Forever) != 0) return 5;

	// The decision: ready runs, the first request waits within its budget, later ones are skipped.
	if (DecideDispatch(false, true, false, 0) != DispatchAction::Run) return 10;
	if (DecideDispatch(false, false, true, Kernel) != DispatchAction::WaitBounded) return 11;
	if (DecideDispatch(false, false, false, Kernel) != DispatchAction::Skip) return 12;
	if (DecideDispatch(false, false, true, FirstWaitBudgetMs(KernelClass::CallSplit, Kernel, Other)) !=
	    DispatchAction::Skip) {
		return 13;
	}
	if (DecideDispatch(true, true, true, Kernel) != DispatchAction::Skip) return 14;

	// Session failure state.
	ResetForTest();
	if (Failed()) return 20;
	if (!MarkFailed() || MarkFailed() || !Failed()) return 21;
	ResetForTest();
	if (Failed()) return 22;

	std::puts("rt_session_tests: ok");
	return 0;
}
