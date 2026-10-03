#include "graphics/host_gpu/vramBudget.h"

#include "common/liveSwitch.h"

namespace Libs::Graphics::VramBudget {

namespace {

Live::Switch g_vram_gc_budget("KYTY_VRAM_GC_BUDGET", Live::ParseDefaultOff);

} // namespace

bool GcEnabled() noexcept {
	return g_vram_gc_budget.On();
}

} // namespace Libs::Graphics::VramBudget
