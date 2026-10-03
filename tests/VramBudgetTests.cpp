// KYTY_VRAM_GC_BUDGET marks and plans (graphics/host_gpu/vramBudget.h) for a 24 GB, a 12 GB and an
// 8 GB card's planning budgets.
#include "graphics/host_gpu/vramBudget.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics::VramBudget;

constexpr uint64_t MiB = 1024ull * 1024;
constexpr uint64_t GiB = 1024 * MiB;

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::printf("VramBudgetTests: failed: %s\n", what);
		g_failures++;
	}
}

void TestMarks() {
	const uint64_t budget = 9 * GiB + 512 * MiB; // a 12 GB card's planning budget
	Expect(ImageTrigger(budget) == budget / 4 * 3, "image trigger is three quarters of the budget");
	Expect(Critical(budget) == budget - budget / 16, "critical mark 1/16 below the budget");
	Expect(BufferTrigger(budget) == budget, "buffer collector only at the budget");
	Expect(BufferCritical(budget) > budget, "buffer downloads only above the budget");
	Expect(ImageTrigger(budget) < Critical(budget), "trigger below the critical mark");
}

void TestPlans() {
	// 24 GB card, Sky Garden (9.6 GB): nothing to do (the stock trigger would be 7.3 GiB).
	const uint64_t large = 22 * GiB + 512 * MiB;
	Expect(!PlanImages(large, 9600 * MiB, 0).retire, "24 GB: no retirement at 9.6 GB");
	// 12 GB card: 5.8 GB (Sky Garden with KYTY_FUNCTION_ARRAY_SHRINK) is below the trigger.
	const uint64_t medium = 9 * GiB + 512 * MiB;
	Expect(!PlanImages(medium, 5800 * MiB, 0).retire, "12 GB: no retirement at 5.8 GB");
	const auto above = PlanImages(medium, 8 * GiB, 0);
	Expect(above.retire && above.age_frames == DefaultAgeFrames, "12 GB: 8 GiB retires 30-frame-old images");
	Expect(above.bytes == 8 * GiB - ImageTrigger(medium), "12 GB: down to the trigger");
	const auto critical = PlanImages(medium, 9 * GiB + 256 * MiB, 0);
	Expect(critical.retire && critical.age_frames == DefaultAgeFrames / 4,
	       "12 GB: above the critical mark a quarter of the age");
	// 8 GB card: 5.8 GB is above the trigger.
	const uint64_t small = 6 * GiB + 300 * MiB;
	const auto     tight = PlanImages(small, 5800 * MiB, 0);
	Expect(tight.retire && tight.age_frames == DefaultAgeFrames, "8 GB: 5.8 GB retires 30-frame-old images");
	// KYTY_VRAM_PRESSURE_FRAMES as the age, and its floor of 4 frames.
	Expect(PlanImages(small, 5800 * MiB, 60).age_frames == 60, "age from KYTY_VRAM_PRESSURE_FRAMES");
	Expect(PlanImages(small, 6200 * MiB, 8).age_frames == 4, "a quarter, at least 4 frames");
	// No budget (the driver reports none): nothing.
	Expect(!PlanImages(0, 5 * GiB, 0).retire, "no budget: no retirement");
}

} // namespace

int main() {
	TestMarks();
	TestPlans();
	if (g_failures != 0) {
		std::printf("%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("vram_budget: all tests passed\n");
	return 0;
}
