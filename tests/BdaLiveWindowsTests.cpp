// KYTY_BDA_LIVE_PAGES import-window bookkeeping (graphics/host_gpu/renderer/cache/bdaLiveWindows.h).
#include "graphics/host_gpu/renderer/cache/bdaLiveWindows.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using Libs::Graphics::BdaLiveWindows;

namespace {

int g_failures = 0;

void Check(bool condition, const char* what, int line) {
	if (!condition) {
		std::printf("FAILED line %d: %s\n", line, what);
		g_failures++;
	}
}

#define CHECK(x) Check((x), #x, __LINE__)

constexpr uint64_t MiB    = 1ull << 20u;
constexpr uint64_t Base   = 0x7f0000000000ull;
constexpr uint64_t Window = 64 * MiB;

struct FakeImport {
	std::vector<std::pair<uint64_t, uint64_t>> calls; // (host, bytes)
	bool                                       fail = false;
	uint64_t operator()(uint64_t host, uint64_t bytes) {
		calls.emplace_back(host, bytes);
		return fail ? 0 : 0x100000000000ull + (host - Base); // device = fixed offset from host
	}
};

void TestTranslateImportsOnce() {
	FakeImport     fake;
	BdaLiveWindows w(Base, 200 * MiB, Window, 1024 * MiB, [&](uint64_t h, uint64_t b) { return fake(h, b); });
	CHECK(w.Valid());
	const uint64_t host = Base + 70 * MiB + 0x4000; // window 1
	CHECK(w.Translate(host, 0x4040) == 0x100000000000ull + 70 * MiB + 0x4000);
	CHECK(fake.calls.size() == 1 && fake.calls[0].first == Base + Window && fake.calls[0].second == Window);
	// Same window again: no second import; another page of it translates by offset.
	CHECK(w.Translate(host + 0x8000, 0x4040) == 0x100000000000ull + 70 * MiB + 0xc000);
	CHECK(fake.calls.size() == 1);
	CHECK(w.Windows() == 1 && w.PinnedBytes() == Window);
}

void TestLastWindowIsShorter() {
	FakeImport     fake;
	BdaLiveWindows w(Base, 200 * MiB, Window, 1024 * MiB, [&](uint64_t h, uint64_t b) { return fake(h, b); });
	CHECK(w.Translate(Base + 199 * MiB, 0x4040) != 0); // window 3 = [192, 200) MiB
	CHECK(fake.calls.size() == 1 && fake.calls[0].second == 8 * MiB);
	CHECK(fake.calls[0].second % BdaLiveWindows::IMPORT_ALIGNMENT == 0);
}

void TestRangeRules() {
	FakeImport     fake;
	BdaLiveWindows w(Base, 200 * MiB, Window, 1024 * MiB, [&](uint64_t h, uint64_t b) { return fake(h, b); });
	// The page's last bytes plus 64 past the window end: rejected (a load could leave the import).
	CHECK(w.Translate(Base + Window - 0x4000, 0x4040) == 0);
	CHECK(w.RangeRejects() == 1);
	// Outside the backing on either side, or running past its end.
	CHECK(w.Translate(Base - 0x4000, 0x4000) == 0);
	CHECK(w.Translate(Base + 200 * MiB, 0x4000) == 0);
	CHECK(w.Translate(Base + 200 * MiB - 0x2000, 0x4000) == 0);
	CHECK(w.Translate(Base, 0) == 0);
	CHECK(fake.calls.empty());
}

void TestCapAndFailure() {
	FakeImport     fake;
	BdaLiveWindows w(Base, 400 * MiB, Window, 100 * MiB, [&](uint64_t h, uint64_t b) { return fake(h, b); });
	CHECK(w.Translate(Base + 0x4000, 0x4040) != 0);           // window 0 (64 MiB pinned)
	CHECK(w.Translate(Base + Window + 0x4000, 0x4040) == 0);  // window 1 would pass the 100 MiB cap
	CHECK(w.CapRejects() == 1 && w.Windows() == 1);
	FakeImport     bad;
	bad.fail = true;
	BdaLiveWindows f(Base, 400 * MiB, Window, 1024 * MiB, [&](uint64_t h, uint64_t b) { return bad(h, b); });
	CHECK(f.Translate(Base + 0x4000, 0x4040) == 0);
	CHECK(f.Translate(Base + 0x8000, 0x4040) == 0);
	CHECK(bad.calls.size() == 1 && f.Failures() == 1 && f.Windows() == 0); // a failed window is not retried
}

void TestInvalidConfigurations() {
	const auto never = [](uint64_t, uint64_t) -> uint64_t { return 1; };
	CHECK(!BdaLiveWindows(Base + 0x800, 200 * MiB, Window, 1024 * MiB, never).Valid()); // base not 4 KiB aligned
	CHECK(!BdaLiveWindows(Base, 200 * MiB + 0x100, Window, 1024 * MiB, never).Valid()); // size not 4 KiB aligned
	CHECK(!BdaLiveWindows(Base, 200 * MiB, 48 * MiB, 1024 * MiB, never).Valid());       // not a power of two
	CHECK(!BdaLiveWindows(Base, 200 * MiB, 0x8000, 1024 * MiB, never).Valid());         // smaller than 64 KiB
	CHECK(!BdaLiveWindows(0, 200 * MiB, Window, 1024 * MiB, never).Valid());
	CHECK(!BdaLiveWindows(Base, 200 * MiB, Window, 1024 * MiB, nullptr).Valid());
	BdaLiveWindows invalid(Base + 0x800, 200 * MiB, Window, 1024 * MiB, never);
	CHECK(invalid.Translate(Base + 0x4000, 0x4000) == 0);
}

} // namespace

int main() {
	TestTranslateImportsOnce();
	TestLastWindowIsShorter();
	TestRangeRules();
	TestCapAndFailure();
	TestInvalidConfigurations();
	if (g_failures != 0) {
		std::printf("bda_live_windows: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("bda_live_windows: all passed\n");
	return 0;
}
