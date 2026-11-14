// KYTY_IMG_* (drawPrep/imgSide.h): the switch parser, the second-level way choice of KYTY_IMG_MEMO2,
// KYTY_IMG_INPLACE's slot numbering, KYTY_IMG_PARTNER's decision table and the switch counters.
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

void TestSlots() {
	// 4096 primary slots, 8192 sets x 4 ways behind them.
	constexpr uint32_t primary = 4096;
	Check(!ImgSide::IsSecondSlot(0, primary) && !ImgSide::IsSecondSlot(primary - 1, primary),
	      "primary slots are below the primary count");
	Check(ImgSide::SecondSlot(primary, 0, 4, 0) == primary, "first second-level slot follows them");
	Check(ImgSide::IsSecondSlot(ImgSide::SecondSlot(primary, 0, 4, 0), primary), "and is a second-level slot");
	Check(ImgSide::SecondSlot(primary, 8191, 4, 3) == primary + 32767, "last second-level slot");
	// Distinct (set, way) pairs get distinct slots.
	bool distinct = true;
	for (uint32_t set = 0; set < 3; set++) {
		for (uint32_t way = 0; way < 4; way++) {
			distinct = distinct && ImgSide::SecondSlot(primary, set, 4, way) - primary == set * 4 + way;
		}
	}
	Check(distinct, "slots are set * ways + way after the primary ones");
}

void TestPartnerDecision() {
	using ImgSide::Mode;
	using ImgSide::PartnerAction;
	// Off: the old rule only.
	Check(ImgSide::PartnerDecision(Mode::Off, true, true) == PartnerAction::Miss, "off: old rule refuses");
	Check(ImgSide::PartnerDecision(Mode::Off, false, false) == PartnerAction::Hit, "off: old rule answers");
	// On: whether SyncAliasFromOwner would do nothing, whatever the old rule says.
	Check(ImgSide::PartnerDecision(Mode::On, true, true) == PartnerAction::Hit, "on: no owner to sync from");
	Check(ImgSide::PartnerDecision(Mode::On, true, false) == PartnerAction::HitSync, "on: an owner qualifies: sync in the hit");
	Check(ImgSide::PartnerDecision(Mode::On, false, false) == PartnerAction::HitSync,
	      "on: a syncable stencil image the old rule let answer is synchronized");
	Check(ImgSide::PartnerDecision(Mode::On, false, true) == PartnerAction::Hit, "on: owner of the bytes");
	// Verify: the old result, with a claim where the new rule would answer.
	Check(ImgSide::PartnerDecision(Mode::Verify, true, true) == PartnerAction::Claim, "verify: claim");
	Check(ImgSide::PartnerDecision(Mode::Verify, true, false) == PartnerAction::Claim, "verify: claim (sync)");
	Check(ImgSide::PartnerDecision(Mode::Verify, false, false) == PartnerAction::Hit, "verify: old hit kept");
}

void TestReasonOrder() {
	// renderDraw.cpp counts AcqWhyCaller + n for DrawRunAcquireRefusal's n (2 .. 14).
	const auto at = [](uint8_t n) { return static_cast<ImgSide::C>(static_cast<uint8_t>(ImgSide::C::AcqWhyCaller) + n); };
	Check(at(1) == ImgSide::C::AcqWhyOff && at(2) == ImgSide::C::AcqWhyNoPrev && at(9) == ImgSide::C::AcqWhyInstance &&
	          at(13) == ImgSide::C::AcqWhyImages && at(14) == ImgSide::C::AcqWhyNone,
	      "acquisition refusal codes map onto the AcqWhy counters");
	Check(static_cast<uint8_t>(ImgSide::C::AcqWhyNone) + 1 == static_cast<uint8_t>(ImgSide::C::SeedWhyNotReached),
	      "the seed reasons follow the refusal codes");
	// renderDraw.cpp counts SeedWhyNotReached + RenderExecutor::m_seed_why (0 .. 13).
	const auto seed = [](uint8_t n) {
		return static_cast<ImgSide::C>(static_cast<uint8_t>(ImgSide::C::SeedWhyNotReached) + n);
	};
	Check(seed(1) == ImgSide::C::SeedWhyKernel && seed(2) == ImgSide::C::SeedWhyEligible &&
	          seed(3) == ImgSide::C::SeedWhyOff && seed(10) == ImgSide::C::SeedWhyTextures &&
	          seed(11) == ImgSide::C::SeedWhyDepthPromotion && seed(13) == ImgSide::C::SeedWhyRecorded,
	      "seed outcome codes map onto the SeedWhy counters");
	Check(static_cast<uint8_t>(ImgSide::C::SeedWhyRecorded) + 1 == static_cast<uint8_t>(ImgSide::C::Count),
	      "SeedWhyRecorded is the last counter");
}

void TestDepthAlias() {
	using ImgSide::DaRecord;
	// The replay of ResolveDepthOverlap's alias-branch effect.
	bool rebind = false;
	Check(!ImgSide::ApplyRebind(false, false, rebind) && !rebind, "neither bound nor a target: nothing");
	Check(ImgSide::ApplyRebind(true, false, rebind) && rebind, "bound: needs_rebind set");
	rebind = false;
	Check(ImgSide::ApplyRebind(false, true, rebind) && rebind, "a target: needs_rebind set");
	rebind = true;
	Check(ImgSide::ApplyRebind(true, true, rebind) && rebind, "already set: applied again (epoch moves)");
	// Record's decision.
	Check(ImgSide::DepthAliasRecord(true, false, 25, 64) == DaRecord::Keep, "pure, 25 pages: kept");
	Check(ImgSide::DepthAliasRecord(true, false, 64, 64) == DaRecord::Keep, "the limit itself: kept");
	Check(ImgSide::DepthAliasRecord(true, false, 65, 64) == DaRecord::Span, "over the limit: span");
	Check(ImgSide::DepthAliasRecord(false, false, 1, 64) == DaRecord::Impure, "a recreate: impure");
	Check(ImgSide::DepthAliasRecord(false, true, 100, 64) == DaRecord::Impure, "impure before the other reasons");
	Check(ImgSide::DepthAliasRecord(true, true, 1, 64) == DaRecord::Other, "rebased / DCC / no pages: other");
	// The span: outer region [10, 35), image page 10, inner [10, 35): [10, 35).
	auto span = ImgSide::DepthAliasSpan(10, 35, 10, 35, 10);
	Check(span.lo == 10 && span.hi == 35, "region = inner: the region");
	span = ImgSide::DepthAliasSpan(10, 12, UINT64_MAX, 0, 10);
	Check(span.lo == 10 && span.hi == 12, "no inner region (Same answers only)");
	span = ImgSide::DepthAliasSpan(10, 12, 8, 40, 10);
	Check(span.lo == 8 && span.hi == 40, "a larger inner region widens the span");
	span = ImgSide::DepthAliasSpan(10, 12, UINT64_MAX, 0, 50);
	Check(span.lo == 10 && span.hi == 51, "the image's first page outside the region is included");
}

void TestAcqFb() {
	// The content serial an acquisition leaves (and a KYTY_IMG_ACQFB reuse replays).
	Check(!ImgSide::AcqFbSerialKept(false, false, true), "colour target: always a fresh serial");
	Check(!ImgSide::AcqFbSerialKept(false, true, true), "colour target written: fresh");
	Check(ImgSide::AcqFbSerialKept(true, false, true), "unwritten depth with alias-sync skip: restored");
	Check(!ImgSide::AcqFbSerialKept(true, true, true), "written depth: fresh");
	Check(!ImgSide::AcqFbSerialKept(true, false, false), "without alias-sync skip nothing is restored");
	// The switch counters the verify boots read are distinct and in the line.
	Check(static_cast<uint8_t>(ImgSide::S::AcqFbMismatch) + 1 == static_cast<uint8_t>(ImgSide::S::Frames),
	      "acqfb counters end right before frames");
	Check(static_cast<uint8_t>(ImgSide::S::DaMismatch) + 1 == static_cast<uint8_t>(ImgSide::S::AcqFbSame),
	      "depthalias counters precede the acqfb ones");
}

} // namespace

int main() {
	TestParse();
	TestPickWay();
	TestCounters();
	TestSlots();
	TestPartnerDecision();
	TestReasonOrder();
	TestDepthAlias();
	TestAcqFb();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("img_side_tests: all passed\n");
	return 0;
}
