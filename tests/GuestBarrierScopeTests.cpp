#include "graphics/host_gpu/renderer/render.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

static void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "%s\n", message);
		std::exit(1);
	}
}

static uint32_t ModeOf(int64_t bits, GuestBarrierKind kind) {
	return static_cast<uint32_t>((bits >> (2 * static_cast<uint32_t>(kind))) & 3);
}

int main() {
	// KYTY_GUEST_BARRIER_SCOPE parser (render.h): 2 bits per kind, 0 full, 1 narrow, 2 drop.
	Check(ParseGuestBarrierScope(nullptr) == 0, "unset must be all full");
	Check(ParseGuestBarrierScope("") == 0, "empty must be all full");
	const auto narrow = ParseGuestBarrierScope("narrow");
	Check(ModeOf(narrow, GuestBarrierKind::CsPartialFlush) == 1 &&
	          ModeOf(narrow, GuestBarrierKind::GfxPartialFlush) == 1 &&
	          ModeOf(narrow, GuestBarrierKind::CbDbWriteback) == 1 &&
	          ModeOf(narrow, GuestBarrierKind::MetaWriteback) == 1,
	      "shorthand narrow must cover the four EVENT_WRITE kinds");
	Check(ModeOf(narrow, GuestBarrierKind::ReleaseMem) == 0 && ModeOf(narrow, GuestBarrierKind::Pm4Sync) == 0,
	      "shorthand narrow must leave rel and sync full");
	const auto mixed = ParseGuestBarrierScope("cs=drop,cbdb=narrow,rel=drop");
	Check(ModeOf(mixed, GuestBarrierKind::CsPartialFlush) == 2 && ModeOf(mixed, GuestBarrierKind::GfxPartialFlush) == 0 &&
	          ModeOf(mixed, GuestBarrierKind::CbDbWriteback) == 1 && ModeOf(mixed, GuestBarrierKind::ReleaseMem) == 2,
	      "per-kind list");
	const auto later = ParseGuestBarrierScope("drop,gfx=full");
	Check(ModeOf(later, GuestBarrierKind::CsPartialFlush) == 2 && ModeOf(later, GuestBarrierKind::GfxPartialFlush) == 0,
	      "later items override earlier ones");
	const auto all = ParseGuestBarrierScope("all=drop");
	for (uint32_t kind = 0; kind < static_cast<uint32_t>(GuestBarrierKind::Count); kind++) {
		Check(ModeOf(all, static_cast<GuestBarrierKind>(kind)) == 2, "all=drop must cover every kind");
	}
	Check(ParseGuestBarrierScope("cs=fast") < 0, "unknown mode must be rejected");
	Check(ParseGuestBarrierScope("xyz=drop") < 0, "unknown kind must be rejected");
	Check(ParseGuestBarrierScope("sometimes") < 0, "unknown shorthand must be rejected");
	std::printf("guest barrier scope tests passed\n");
	return 0;
}
