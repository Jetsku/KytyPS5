// Guest TSC domain tests (kernel/guestTsc.h): the 2^32 Hz frequency cap of KytyPS5/KytyPS5#1910,
// tick scaling, monotonicity, KYTY_GUEST_TSC_FREQUENCY, and the GPU reference-clock / EOP
// timestamp conversions fed with the guest values, on a simulated 4.7 GHz host.
#include "graphics/host_gpu/renderer/eopTimestampClock.h"
#include "graphics/host_gpu/renderer/referenceClock.h"
#include "kernel/guestTsc.h"

#include <cstdint>
#include <cstdio>
#include <random>

namespace {

using namespace Libs::LibKernel;
using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "FAILED: %s\n", text);
		g_failures++;
	}
}

uint64_t Distance(uint64_t a, uint64_t b) {
	return a > b ? a - b : b - a;
}

constexpr uint64_t kHost47 = 4700000000ull; // 9800X3D
constexpr uint64_t kHost42 = 4200000000ull; // 7950X3D
constexpr uint64_t kBase   = 0x0000123456789abcull;

void TestChooseFrequency() {
	using GuestTsc::ChooseFrequency;
	using GuestTsc::kMaxGuestTscFrequency;
	Check(ChooseFrequency(kHost47, nullptr) == kMaxGuestTscFrequency, "4.7 GHz is capped");
	Check(ChooseFrequency(kHost47, nullptr) <= UINT32_MAX, "the cap fits 32 bits");
	Check(static_cast<uint32_t>(ChooseFrequency(kHost47, nullptr)) == kMaxGuestTscFrequency,
	      "a 32-bit copy of the capped frequency is exact");
	Check(ChooseFrequency(kHost42, nullptr) == kHost42, "4.2 GHz is reported unchanged");
	Check(ChooseFrequency(kMaxGuestTscFrequency, nullptr) == kMaxGuestTscFrequency,
	      "the cap itself is unchanged");
	Check(ChooseFrequency(kHost47, "") == kMaxGuestTscFrequency, "empty switch = default");
	Check(ChooseFrequency(kHost47, "0") == kHost47, "switch 0 reports the host frequency");
	Check(ChooseFrequency(kHost42, "0") == kHost42, "switch 0 below the cap");
	Check(ChooseFrequency(kHost47, "1600000000") == 1600000000ull, "forced frequency");
	Check(ChooseFrequency(kHost42, "3000000000") == 3000000000ull, "forced below the host");
	Check(ChooseFrequency(kHost47, "abc") == kMaxGuestTscFrequency, "junk = default");
	Check(ChooseFrequency(kHost47, "12x") == kMaxGuestTscFrequency, "trailing junk = default");
	Check(ChooseFrequency(kHost47, "5") == kMaxGuestTscFrequency, "too low = default");
	Check(ChooseFrequency(0, nullptr) == 0, "unknown host frequency stays 0");
}

void TestIdentityBelowCap() {
	const GuestTsc::Scale scale(kHost42, GuestTsc::ChooseFrequency(kHost42, nullptr), kBase);
	Check(scale.Identity(), "4.2 GHz host: identity mapping");
	bool same = true;
	std::mt19937_64 rng(1);
	for (int i = 0; i < 10000; i++) {
		const uint64_t raw = rng();
		same               = same && scale.ToGuest(raw) == raw && scale.Ticks(raw) == raw;
	}
	same = same && scale.ToGuest(0) == 0 && scale.ToGuest(UINT64_MAX) == UINT64_MAX;
	Check(same, "4.2 GHz host: ticks are bit-identical");
	const GuestTsc::Scale off(kHost47, GuestTsc::ChooseFrequency(kHost47, "0"), kBase);
	Check(off.Identity() && off.ToGuest(kBase + 12345) == kBase + 12345,
	      "switch 0: raw host ticks");
}

void TestScale47() {
	const uint64_t        guest = GuestTsc::ChooseFrequency(kHost47, nullptr);
	const GuestTsc::Scale scale(kHost47, guest, kBase);
	Check(!scale.Identity(), "4.7 GHz host: scaled");
	Check(scale.ToGuest(kBase) == kBase, "process start maps to itself");
	Check(scale.ToGuest(kBase + kHost47) == kBase + guest, "one host second = one guest second");
	Check(scale.Ticks(kHost47 * 3600) == guest * 3600, "one hour, exact");
	Check(scale.Ticks(kHost47 / 1000) == guest / 1000, "one millisecond, exact");
	Check(scale.ToGuest(kBase - kHost47) == kBase - guest, "a second before the start");
	Check(scale.ToGuest(0) <= kBase, "far before the start does not wrap");

	// Monotonic over random increasing raw values, and the rate stays within one tick.
	std::mt19937_64 rng(7);
	uint64_t        raw   = kBase - 1000;
	uint64_t        prev  = scale.ToGuest(raw);
	bool            mono  = true;
	bool            exact = true;
	for (int i = 0; i < 200000; i++) {
		raw += rng() % (i % 100 == 0 ? 50000000ull : 5000ull);
		const auto g = scale.ToGuest(raw);
		mono         = mono && g >= prev;
		prev         = g;
		if (raw >= kBase) {
			const long double expect = static_cast<long double>(kBase) +
			                           static_cast<long double>(raw - kBase) * guest / kHost47;
			exact = exact && Distance(g, static_cast<uint64_t>(expect)) <= 1;
		}
	}
	Check(mono, "guest ticks never go backwards");
	Check(exact, "guest ticks within one tick of raw * guest / host");

	// Every consecutive raw tick: no step larger than one guest tick, none backwards.
	bool fine = true;
	prev      = scale.ToGuest(kBase + 999000000);
	for (uint64_t r = kBase + 999000001; r < kBase + 999100000; r++) {
		const auto g = scale.ToGuest(r);
		fine         = fine && g >= prev && g - prev <= 1;
		prev         = g;
	}
	Check(fine, "consecutive raw ticks step by 0 or 1 guest tick");

	// Process time counter = ReadTsc - start value.
	Check(scale.Ticks(123456789) == scale.ToGuest(kBase + 123456789) - kBase,
	      "process time counter matches ReadTsc");

	// Arbitrary forced frequency (no common factor) keeps the ratio.
	const GuestTsc::Scale odd(kHost47 + 1, 4294967291ull, kBase);
	const auto            ticks = odd.Ticks(kHost47 + 1);
	Check(Distance(ticks, 4294967291ull) <= 4, "co-prime frequencies: one second within 4 ticks");
}

void TestReferenceClock47() {
	// Sync::ReadReferenceClock with the guest values: real time at 100 MHz.
	const uint64_t        guest = GuestTsc::ChooseFrequency(kHost47, nullptr);
	const GuestTsc::Scale scale(kHost47, guest, kBase);
	uint64_t              r0 = 0;
	uint64_t              r1 = 0;
	Check(Sync::ScaleReferenceClock(scale.ToGuest(kBase + kHost47), guest, r0) &&
	          Sync::ScaleReferenceClock(scale.ToGuest(kBase + 2 * kHost47), guest, r1),
	      "reference clock scales");
	Check(Distance(r1 - r0, Sync::GRAPHICS_REFERENCE_CLOCK_FREQUENCY) <= 1,
	      "one host second = 100M reference ticks");
	// The broken 32-bit view the cap avoids: 4.7 GHz truncated is 0.405 GHz.
	Check(static_cast<uint32_t>(kHost47) < 500000000u, "the uncapped value wraps (sanity)");
}

void TestEopConversion47() {
	// EopTimestampRing::Recalibrate fills pair_tsc/tsc_frequency from KernelReadTsc /
	// KernelGetTscFrequency, i.e. guest values. A device timestamp 1.5 s after the calibration
	// must map to the reference clock of the guest TSC read at that moment.
	const uint64_t        guest = GuestTsc::ChooseFrequency(kHost47, nullptr);
	const GuestTsc::Scale scale(kHost47, guest, kBase);
	constexpr uint64_t    kQpcFreq   = 10000000;
	const uint64_t        raw_at_cal = kBase + 7 * kHost47;
	EopTimestamps::ClockMap map;
	map.device_raw    = 1000000000;
	map.device_qpc    = 50000000000ull;
	map.qpc_frequency = kQpcFreq;
	map.period_ns     = 1.0;
	map.valid_bits    = 64;
	map.pair_qpc      = map.device_qpc; // TSC read at the same instant
	map.pair_tsc      = scale.ToGuest(raw_at_cal);
	map.tsc_frequency = guest;
	Check(map.Valid(), "EOP map valid");
	uint64_t reference = 0;
	Check(EopTimestamps::ToReference(map, map.device_raw + 1500000000ull, reference),
	      "EOP conversion succeeds");
	uint64_t expect = 0;
	Check(Sync::ScaleReferenceClock(scale.ToGuest(raw_at_cal + kHost47 * 3 / 2), guest, expect),
	      "expected reference");
	Check(Distance(reference, expect) <= 2, "EOP timestamp = reference clock of the guest TSC");
	uint64_t at_cal = 0;
	Check(EopTimestamps::ToReference(map, map.device_raw, at_cal) &&
	          Distance(reference - at_cal, 150000000) <= 2,
	      "1.5 device seconds = 150M reference ticks");
}

} // namespace

int main() {
	TestChooseFrequency();
	TestIdentityBelowCap();
	TestScale47();
	TestReferenceClock47();
	TestEopConversion47();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("guest_tsc_tests: all passed\n");
	return 0;
}
