#include "common/sehGuard.h"

#include "common/hostException.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Common {

#ifdef _WIN32
uint32_t CallCatchingStructuredException(void (*fn)(void*), void* context) {
	// Kyty's vectored handler (hostException.cpp) runs before any __except and ends the process
	// for a fault it cannot resolve; the probe depth makes it pass this thread's faults on.
	HostException::EnterProbe();
	uint32_t code = 0;
	__try {
		fn(context);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		code = static_cast<uint32_t>(GetExceptionCode());
	}
	HostException::LeaveProbe();
	return code;
}
#else
uint32_t CallCatchingStructuredException(void (*fn)(void*), void* context) {
	fn(context);
	return 0;
}
#endif

} // namespace Common
