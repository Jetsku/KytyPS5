#ifndef EMULATOR_SRC_KERNEL_GUESTTSC_H_
#define EMULATOR_SRC_KERNEL_GUESTTSC_H_

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>

// The TSC frequency and ticks the guest sees (sceKernelGetTscFrequency, sceKernelReadTsc,
// sceKernelGetProcessTimeCounter[Frequency] and everything derived from them: video-out event
// and flip-status timestamps, the GPU reference clock, EOP timestamps).
//
// Some titles keep the TSC frequency in 32 bits (Beyond Good & Evil 20th Anniversary,
// KytyPS5/KytyPS5#1910). A host TSC above 2^32 Hz (AMD Zen 4/5 parts with a 4.3 GHz+ base clock)
// wraps there: 4.7 GHz is seen as 0.405 GHz and the game runs 11.6x too fast. The guest frequency
// is therefore capped at kMaxGuestTscFrequency and guest ticks advance at that rate. Below the cap
// the mapping is the identity, bit for bit.
//
// Guest ticks are anchored at the process start (base): guest = base + (raw - base) * guest / host.
// Titles that also execute RDTSC themselves (Astro Bot does) see raw host ticks; anchoring keeps
// those within (1 - guest/host) of the process time from the API values instead of that share of
// the host uptime.
//
// KYTY_GUEST_TSC_FREQUENCY=<Hz> forces a guest frequency; 0 turns the cap off (raw host ticks).
// Header-only for guest_tsc_tests.
namespace Libs::LibKernel::GuestTsc {

// Below 2^32 = 4294967296 and a multiple of the 100 kHz measurement rounding (the value of
// f55fb928e).
constexpr uint64_t kMaxGuestTscFrequency = 4294000000ull;
// Lowest forced frequency accepted from KYTY_GUEST_TSC_FREQUENCY.
constexpr uint64_t kMinGuestTscFrequency = 1000000ull;

// The guest frequency for a measured host frequency. `env` is the KYTY_GUEST_TSC_FREQUENCY value
// (nullptr or empty: default cap; "0": host frequency; otherwise the frequency in Hz, at least
// kMinGuestTscFrequency, else the default).
[[nodiscard]] inline uint64_t ChooseFrequency(uint64_t measured, const char* env) noexcept {
	if (measured == 0) {
		return 0;
	}
	if (env != nullptr && *env != '\0') {
		char*      end   = nullptr;
		const auto value = std::strtoull(env, &end, 10);
		if (end != env && *end == '\0') {
			if (value == 0) {
				return measured;
			}
			if (value >= kMinGuestTscFrequency) {
				return value;
			}
		}
	}
	return measured < kMaxGuestTscFrequency ? measured : kMaxGuestTscFrequency;
}

// raw host ticks -> guest ticks. Monotonic (non-decreasing) in raw; exact identity when the two
// frequencies are equal.
class Scale {
public:
	Scale() = default;

	Scale(uint64_t host_frequency, uint64_t guest_frequency, uint64_t base) noexcept
	    : m_base(base) {
		if (host_frequency == 0 || guest_frequency == 0 || host_frequency == guest_frequency) {
			return;
		}
		const auto g = std::gcd(host_frequency, guest_frequency);
		m_num        = guest_frequency / g;
		m_den        = host_frequency / g;
		// (delta % den) * num must fit in 64 bits. Measured frequencies are multiples of 100 kHz,
		// so the reduced terms are tiny; arbitrary forced values lose at most ~2^-31 of the ratio.
		while (m_num > std::numeric_limits<uint64_t>::max() / m_den) {
			m_num = (m_num >> 1u) | 1u;
			m_den = (m_den >> 1u) | 1u;
		}
		m_identity = false;
	}

	[[nodiscard]] bool     Identity() const noexcept { return m_identity; }
	[[nodiscard]] uint64_t Base() const noexcept { return m_base; }

	[[nodiscard]] uint64_t ToGuest(uint64_t raw) const noexcept {
		if (m_identity) {
			return raw;
		}
		if (raw >= m_base) {
			return m_base + Ticks(raw - m_base);
		}
		// Another core's TSC may trail the process-start sample by a few ticks.
		const auto back = Ticks(m_base - raw);
		return back <= m_base ? m_base - back : 0;
	}

	// Guest ticks of `elapsed` host ticks (a span, not a timestamp).
	[[nodiscard]] uint64_t Ticks(uint64_t elapsed) const noexcept {
		if (m_identity) {
			return elapsed;
		}
		return (elapsed / m_den) * m_num + ((elapsed % m_den) * m_num) / m_den;
	}

private:
	uint64_t m_base     = 0;
	uint64_t m_num      = 1;
	uint64_t m_den      = 1;
	bool     m_identity = true;
};

} // namespace Libs::LibKernel::GuestTsc

#endif // EMULATOR_SRC_KERNEL_GUESTTSC_H_
