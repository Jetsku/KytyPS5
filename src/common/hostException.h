#ifndef KYTY_COMMON_HOST_EXCEPTION_H_
#define KYTY_COMMON_HOST_EXCEPTION_H_

#include "common/common.h"

namespace Common::HostException {

enum class ExceptionType { Unknown, AccessViolation, IllegalInstruction };

enum class AccessViolationType { Unknown, Read, Write, Execute };

struct ExceptionInfo {
	ExceptionType       type                   = ExceptionType::Unknown;
	AccessViolationType access_violation_type  = AccessViolationType::Unknown;
	uint64_t            access_violation_vaddr = 0;
	uint64_t            exception_address      = 0;
	uint64_t            rax                    = 0;
	uint64_t            rbx                    = 0;
	uint64_t            rcx                    = 0;
	uint64_t            rdx                    = 0;
	uint64_t            rsi                    = 0;
	uint64_t            rdi                    = 0;
	uint64_t            rbp                    = 0;
	uint64_t            rsp                    = 0;
	uint64_t            r8                     = 0;
	uint64_t            r9                     = 0;
	uint64_t            r10                    = 0;
	uint64_t            r11                    = 0;
	uint64_t            r12                    = 0;
	uint64_t            r13                    = 0;
	uint64_t            r14                    = 0;
	uint64_t            r15                    = 0;
	uint32_t            native_code            = 0;
	// Platform-specific mutable context, valid only for the duration of the handler call.
	void* native_context = nullptr;
};

using Handler = bool (*)(const ExceptionInfo&);

bool InstallHandler(Handler handler);
// Windows: additionally registers `handler` at the FRONT of the vectored exception handler list
// for access violations only. It must return false (never terminate) for faults it does not
// resolve: those continue to any other vectored handler and then to InstallHandler's, which
// stays registered last. Returns false (nothing installed) on other platforms, where the
// single signal handler already runs first.
bool InstallFirstAccessHandler(Handler handler);

// Windows: between EnterProbe and LeaveProbe the calling thread's faults skip InstallHandler's
// handler (which ends the process for a fault it cannot resolve) and reach the thread's own
// __try/__except. Vectored handlers run before frame-based handlers, so without this a fault in
// a guarded diagnostic, such as a stack walk through a corrupt guest frame, ends the process.
// Calls nest. Plain calls rather than a scope object: functions using __try cannot hold objects
// with destructors. No-ops on other platforms.
void EnterProbe();
void LeaveProbe();

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
bool InitializeThreadSignalStack();
#endif

} // namespace Common::HostException

#endif /* KYTY_COMMON_HOST_EXCEPTION_H_ */
