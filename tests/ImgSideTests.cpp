// KYTY_IMG_* (drawPrep/imgSide.h): the switch parser, the second-level way choice of KYTY_IMG_MEMO2
// and the switch counters.
#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"

#include <array>
#include <cstdint>
#include <cstdio>

// The diagnostic part lives in imgSide.cpp (with the live switches); this test links only the header.
namespace Libs::Graphics::ImgSide::Detail {
bool ReadEnabled() {
	return false;
}
void Add(C /*counter*/, uint64_t /*n*/) {}
void NoteFullKey(uint64_t /*hash*/) {}
void PrintStats() {}
} // namespace Libs::Graphics::ImgSide::Detail

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

void TestParse() {
	using ImgSide::Mode;
	Check(ImgSide::ParseMode(nullptr) == Mode::Off, "unset: off");
	Check(ImgSide::ParseMode("") == Mode::Off, "empty: off");
	Check(ImgSide::ParseMode("0") == Mode::Off, "0: off");
	Check(ImgSide::ParseMode("1") == Mode::On, "1: on");
	Check(ImgSide::ParseMode("on") == Mode::On, "on: on");
	Check(ImgSide::ParseMode("2") == Mode::Verify, "2: verify");
	Check(ImgSide::ParseMode("verify") == Mode::Verify, "verify: verify");
	Check(ImgSide::ParseMode("x") == Mode::Off, "unknown: off");
}

// A model of one second-level set: keys (0 = empty way).
struct Set {
	std::array<uint64_t, 4> keys {};
	uint8_t                 next = 0;
	uint32_t                Put(uint64_t key) {
        const auto way = ImgSide::PickWay(
            4, next, [&](uint32_t w) { return keys[w] == key; }, [&](uint32_t w) { return keys[w] == 0; });
        keys[way] = key;
        return way;
	}
};

void TestPickWay() {
	Set set;
	Check(set.Put(11) == 0, "first key: first empty way");
	Check(set.Put(12) == 1, "second key: next empty way");
	Check(set.Put(11) == 0, "same key: its own way again (no duplicate)");
	Check(set.Put(13) == 2 && set.Put(14) == 3, "fills the empty ways in order");
	// Full: round robin from way 0.
	Check(set.Put(15) == 0, "full: round robin 0");
	Check(set.Put(16) == 1, "full: round robin 1");
	Check(set.keys[0] == 15 && set.keys[1] == 16 && set.keys[2] == 13 && set.keys[3] == 14,
	      "replaced the round-robin ways only");
	Check(set.Put(14) == 3, "full: a present key keeps its way");
	set.keys[2] = 0;
	Check(set.Put(17) == 2, "an emptied way is used before round robin");
	// Round robin wraps.
	uint8_t next = 255;
	const auto way = ImgSide::PickWay(4, next, [](uint32_t) { return false; }, [](uint32_t) { return false; });
	Check(way == 3 && next == 0, "round robin wraps with the 8-bit counter");
}

void TestCounters() {
	const auto before = ImgSide::SwitchCounters()[static_cast<size_t>(ImgSide::S::Memo2Hit)].load();
	ImgSide::Bump(ImgSide::S::Memo2Hit);
	ImgSide::Bump(ImgSide::S::Memo2Hit, 3);
	Check(ImgSide::SwitchCounters()[static_cast<size_t>(ImgSide::S::Memo2Hit)].load() == before + 4,
	      "counter adds");
	// The diagnostic counters are off without KYTY_IMG_STATS (Count is a no-op).
	ImgSide::Count(ImgSide::C::Draws, 5);
	Check(!ImgSide::StatsEnabled(), "stats off by default");
}

} // namespace

int main() {
	TestParse();
	TestPickWay();
	TestCounters();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("img_side_tests: all passed\n");
	return 0;
}
