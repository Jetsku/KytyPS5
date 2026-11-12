#include "common/timer.h"

#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Common {

uint64_t Timer::QueryPerformanceFrequency() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	LARGE_INTEGER f {};
	::QueryPerformanceFrequency(&f);
	return static_cast<uint64_t>(f.QuadPart);
#else
	return 1000000000;
#endif
}

uint64_t Timer::QueryPerformanceCounter() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	LARGE_INTEGER c {};
	::QueryPerformanceCounter(&c);
	return static_cast<uint64_t>(c.QuadPart);
#else
	return SteadyNs();
#endif
}

} // namespace Common
