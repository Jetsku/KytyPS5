#include "common/sehGuard.h"

#include "common/hostException.h"

#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#endif

namespace Common {

#ifdef _WIN32
static int SehGuardFilter(EXCEPTION_POINTERS* pointers, uint64_t* address) {
	if (address != nullptr && pointers != nullptr && pointers->ExceptionRecord != nullptr) {
		*address = reinterpret_cast<uint64_t>(pointers->ExceptionRecord->ExceptionAddress);
	}
	return EXCEPTION_EXECUTE_HANDLER;
}

uint32_t CallCatchingStructuredException(void (*fn)(void*), void* context, uint64_t* address) {
	// Kyty's vectored handler (hostException.cpp) runs before any __except and ends the process
	// for a fault it cannot resolve; the probe depth makes it pass this thread's faults on.
	HostException::EnterProbe();
	uint32_t code = 0;
	__try {
		fn(context);
	} __except (SehGuardFilter(GetExceptionInformation(), address)) {
		code = static_cast<uint32_t>(GetExceptionCode());
	}
	HostException::LeaveProbe();
	return code;
}

std::string DescribeCodeAddress(uint64_t address) {
	char    text[MAX_PATH + 64] = {};
	HMODULE module              = nullptr;
	char    path[MAX_PATH]      = {};
	if (address != 0 &&
	    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCSTR>(address), &module) != 0 &&
	    module != nullptr && GetModuleFileNameA(module, path, MAX_PATH) != 0) {
		const char* name = path;
		for (const char* p = path; *p != '\0'; p++) {
			if (*p == '\\' || *p == '/') name = p + 1;
		}
		std::snprintf(text, sizeof(text), "%s+0x%llx", name,
		              static_cast<unsigned long long>(address - reinterpret_cast<uint64_t>(module)));
		return text;
	}
	std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(address));
	return text;
}
#else
uint32_t CallCatchingStructuredException(void (*fn)(void*), void* context, uint64_t* address) {
	(void)address;
	fn(context);
	return 0;
}

std::string DescribeCodeAddress(uint64_t address) {
	char text[32] = {};
	std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(address));
	return text;
}
#endif

} // namespace Common
