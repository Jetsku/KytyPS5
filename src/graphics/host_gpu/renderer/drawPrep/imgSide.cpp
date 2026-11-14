#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"

#include "common/liveSwitch.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace Libs::Graphics::ImgSide {

namespace {

constexpr size_t CounterCount = static_cast<size_t>(C::Count);

const char* const g_names[CounterCount] = {
    "draws",      "stg-full",    "stg-kept",    "tex-setrep", "tex-run",     "tex-hit",
    "tex-reval",  "tex-full",    "full-rebind", "miss-nomemo", "miss-empty", "miss-conflict",
    "miss-dcc",   "miss-image",  "miss-resident", "miss-page", "miss-partner", "rec-ok",
    "rec-reject", "rec-none",    "full-repkey", "full-repkey2", "view-rep",   "view-run",
    "view-hit",   "view-find",   "view-noent",  "view-mips",  "touch",       "touch-same",
    "bind",       "bind-target", "reset-img",   "acquire",    "acquire-same", "target-run",
    "find-target", "find-target-rep", "transit", "transit-noop",
    "rej-rebased", "rej-other-image", "rej-dcc", "rej-null", "rej-mode", "rej-image", "rej-pages",
    "rej-backing", "partner-atonce", "partner-noowner", "partner-owner",
    "ovl-pure", "ovl-pure-pages4", "ovl-pure-one", "bk-firstpage", "bk-ovl-impure", "bk-ovl-inserted",
    "bk-sb-none", "bk-sb-other", "bk-sb-multi", "bk-desc-changed",
    "acqwhy-caller", "acqwhy-off", "acqwhy-noprev", "acqwhy-engine", "acqwhy-record", "acqwhy-count",
    "acqwhy-activity", "acqwhy-command", "acqwhy-tick", "acqwhy-instance", "acqwhy-written",
    "acqwhy-targets", "acqwhy-feedback", "acqwhy-images", "acqwhy-none",
    "seedwhy-notreached", "seedwhy-kernel", "seedwhy-eligible", "seedwhy-off", "seedwhy-notengine",
    "seedwhy-feedback", "seedwhy-rendering", "seedwhy-writes", "seedwhy-clears", "seedwhy-program",
    "seedwhy-textures", "seedwhy-depthpromo", "seedwhy-overattach", "seedwhy-recorded"};
static_assert(sizeof(g_names) / sizeof(g_names[0]) == CounterCount);

std::array<std::atomic<uint64_t>, CounterCount> g_counters {};
std::atomic<uint64_t>                           g_frames {0};

// Shadow table of the key hashes resolved in full: hash and the frame of its last resolution.
constexpr uint64_t ShadowSlots = uint64_t {1} << 20u;
struct ShadowEntry {
	uint64_t hash  = 0;
	uint64_t frame = 0;
};
std::unique_ptr<ShadowEntry[]> g_shadow;

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

} // namespace

bool Detail::ReadEnabled() {
	const char* value = std::getenv("KYTY_IMG_STATS");
	return value != nullptr && std::strcmp(value, "1") == 0;
}

void Detail::Add(C counter, uint64_t n) {
	auto& c = g_counters[static_cast<size_t>(counter)];
	c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

void Detail::NoteFullKey(uint64_t hash) {
	// The commit thread only (full resolutions).
	if (!g_shadow) {
		g_shadow = std::make_unique<ShadowEntry[]>(ShadowSlots);
	}
	const auto frame = g_frames.load(std::memory_order_relaxed) + 1u; // 0 = never
	auto&      entry = g_shadow[hash & (ShadowSlots - 1u)];
	if (entry.frame != 0 && entry.hash == hash) {
		Add(C::FullRepeatKey, 1);
		if (entry.frame + 2u >= frame) {
			Add(C::FullRepeatKey2, 1);
		}
	}
	entry.hash  = hash;
	entry.frame = frame;
}

void Detail::PrintStats() {
	static uint64_t                          last_ns     = 0;
	static uint64_t                          last_frames = 0;
	static std::array<uint64_t, CounterCount> last {};
	const auto                               now = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	const auto frames = g_frames.load(std::memory_order_relaxed);
	const auto df     = frames - last_frames;
	std::string line  = "ImgSide 10s: frames " + std::to_string(df) + " per frame:";
	for (size_t i = 0; i < CounterCount; i++) {
		const auto v = g_counters[i].load(std::memory_order_relaxed);
		char       text[64];
		std::snprintf(text, sizeof(text), " %s %.1f", g_names[i],
		              df != 0 ? static_cast<double>(v - last[i]) / static_cast<double>(df) : 0.0);
		line += text;
		last[i] = v;
	}
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
	last_ns     = now;
	last_frames = frames;
}

namespace {
Live::Switch g_memo2("KYTY_IMG_MEMO2", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_inplace("KYTY_IMG_INPLACE", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_overlap("KYTY_IMG_OVERLAP", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_acqseed("KYTY_IMG_ACQSEED", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_partner("KYTY_IMG_PARTNER", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_depthalias("KYTY_IMG_DEPTHALIAS", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
Live::Switch g_acqfb("KYTY_IMG_ACQFB", [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });

const char* const g_switch_names[static_cast<size_t>(S::Count)] = {
    "memo2 evict",    "memo2 hit",     "memo2 miss",       "memo2 verify",   "memo2 mismatch",
    "memo copies",    "inplace hit",   "inplace verify",   "inplace mismatch", "partner hit", "partner sync", "inplace run hit",
    "partner verify", "partner mismatch", "overlap rec", "overlap hit", "overlap verify", "overlap mismatch", "acqseed rec", "acqseed hit", "acqseed verify", "acqseed mismatch",
    "ovlrej timed-taken", "ovlrej timed-many", "ovlrej depth", "ovlrej expand", "ovlrej merge", "ovlrej contains",
    "ovlrej inserted", "ovlrej pages", "ovlrej other", "seedref attachment", "seedref program", "seedref writes",
    "seedref other",
    "da recordable", "da rej impure", "da rej span", "da rej other", "da rec", "da hit", "da rebind",
    "da verify", "da mismatch", "acqfb same", "acqfb repeat", "acqfb repeat+serial", "acqfb repeat feedback",
    "acqfb keep", "acqfb state", "acqfb transit", "acqfb marks", "acqfb noinstance", "acqfb hit", "acqfb verify",
    "acqfb mismatch",
    "frames"};
static_assert(sizeof(g_switch_names) / sizeof(g_switch_names[0]) == static_cast<size_t>(S::Count));

bool AnySwitchOn() {
	return Memo2Mode() != Mode::Off || InplaceMode() != Mode::Off || PartnerMode() != Mode::Off ||
	       OverlapMode() != Mode::Off || AcqSeedMode() != Mode::Off || DepthAliasMode() != Mode::Off ||
	       AcqFbMode() != Mode::Off;
}

void PrintSwitches() {
	static uint64_t                                                last_ns = 0;
	static std::array<uint64_t, static_cast<size_t>(S::Count)> last {};
	const auto                                                     now     = NowNs();
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	last_ns = now;
	if (!AnySwitchOn()) {
		for (size_t i = 0; i < last.size(); i++) {
			last[i] = SwitchCounters()[i].load(std::memory_order_relaxed);
		}
		return;
	}
	std::string line = "ImgSwitch 10s: memo2 " + std::to_string(static_cast<int>(Memo2Mode())) +
	                   " inplace " + std::to_string(static_cast<int>(InplaceMode())) + " partner " +
	                   std::to_string(static_cast<int>(PartnerMode())) + " overlap " +
	                   std::to_string(static_cast<int>(OverlapMode())) + " acqseed " +
	                   std::to_string(static_cast<int>(AcqSeedMode())) + " depthalias " +
	                   std::to_string(static_cast<int>(DepthAliasMode())) + " acqfb " +
	                   std::to_string(static_cast<int>(AcqFbMode())) + " |";
	for (size_t i = 0; i < last.size(); i++) {
		const auto v = SwitchCounters()[i].load(std::memory_order_relaxed);
		line += " " + std::string(g_switch_names[i]) + " " + std::to_string(v - last[i]) + " (total " +
		        std::to_string(v) + ")";
		last[i] = v;
	}
	uint64_t mismatches = 0;
	for (const auto counter: {S::Memo2Mismatch, S::InplaceMismatch, S::PartnerMismatch, S::OverlapMismatch,
	                           S::AcqSeedMismatch, S::DaMismatch, S::AcqFbMismatch}) {
		mismatches += SwitchCounters()[static_cast<size_t>(counter)].load(std::memory_order_relaxed);
	}
	line += " | mismatches total " + std::to_string(mismatches);
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}
} // namespace

Mode Memo2Mode() {
	return static_cast<Mode>(g_memo2.Get());
}

Mode InplaceMode() {
	return static_cast<Mode>(g_inplace.Get());
}

Mode PartnerMode() {
	return static_cast<Mode>(g_partner.Get());
}

Mode OverlapMode() {
	return static_cast<Mode>(g_overlap.Get());
}

Mode AcqSeedMode() {
	return static_cast<Mode>(g_acqseed.Get());
}

Mode DepthAliasMode() {
	return static_cast<Mode>(g_depthalias.Get());
}

Mode AcqFbMode() {
	return static_cast<Mode>(g_acqfb.Get());
}

void ReportMismatch(S counter, const char* what) {
	Bump(counter);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr, "ImgSwitch verify: %s\n", what);
		std::printf("ImgSwitch verify: %s\n", what);
		std::fflush(stdout);
	}
}

void PrintSummary() {
	if (StatsEnabled()) {
		Detail::PrintStats();
	}
	PrintSwitches();
}

void OnFrame() {
	Bump(S::Frames);
	if (StatsEnabled()) {
		g_frames.fetch_add(1, std::memory_order_relaxed);
	}
}

} // namespace Libs::Graphics::ImgSide
