#ifndef KYTY_COMMON_TIMER_H_
#define KYTY_COMMON_TIMER_H_

#include <chrono>
#include <cstdint>

namespace Common {

inline uint64_t SteadyNs() {
	return static_cast<uint64_t>(
	    std::chrono::nanoseconds(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Measures from its construction.
class Timer final {
public:
	[[nodiscard]] double GetTimeS() const {
		return std::chrono::duration<double>(std::chrono::steady_clock::now() - m_start).count();
	}

	// The host performance counter (Windows QueryPerformanceCounter; elsewhere steady_clock
	// nanoseconds). Kept for our callers that map it to Vulkan's calibrated QPC time domain
	// (eopTimestamps, videoOut pacing) after upstream 159149c50 dropped it. timer.cpp.
	[[nodiscard]] static uint64_t QueryPerformanceFrequency();
	[[nodiscard]] static uint64_t QueryPerformanceCounter();

private:
	std::chrono::steady_clock::time_point m_start = std::chrono::steady_clock::now();
};

} // namespace Common

#endif /* KYTY_COMMON_TIMER_H_ */
