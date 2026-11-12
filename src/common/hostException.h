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

// Windows: the stack range the TEB gives while guest code runs on a guest stack (pthread.cpp
// RunOnGuestStack, runtimeLinker.cpp RunEntry). Those used to zero StackBase and StackLimit so that
// __chkstk never probes guest memory. With Control Flow Guard on for the process (a Windows that
// turns CFG on for every program), ntdll's RtlGuardRestoreContext checks that the rsp an exception
// handler resumes with lies within the TEB's range and fails fast otherwise (0xC0000409, code 13
// FAST_FAIL_INVALID_SET_OF_CONTEXT): the first write-tracking fault resumed on a guest thread ends
// the process silently, no log line (chenxiao07/KytyPS5 issue #56).
// KYTY_GUEST_TEB_STACK_RANGE=1 (default off): the TEB gives GuestStacksLow..GuestStacksHigh, the
// region every guest pthread stack lies in (pthread.cpp PTHREAD_STACK_BOTTOM..TOP), when the
// thread's host stack (TEB DeallocationStack, gs:0x1478) lies above it, else zeros as before.
// __chkstk probes only below the range's low end: never, as with zeros. The kernel's stack growth
// needs the faulting address between DeallocationStack and the range's top: never, with the host
// stack above it. Game-allocated fiber stacks outside the region are not covered.
// Scratch registers: rcx, rdx, r10, r11. `flag` is an asm memory operand naming
// g_guest_teb_stack_range. Ported from chenxiao07/KytyPS5 e7109edce (the switch added).
inline constexpr uint64_t GuestStacksLow  = 0x0000040000ull;
inline constexpr uint64_t GuestStacksHigh = 0x07efff8000ull;
extern int32_t            g_guest_teb_stack_range;
#define KYTY_GUEST_TEB_STACK_RANGE_ASM(flag)                                                     \
	"movl " flag ", %%r11d\n\t"                                                                    \
	"xorq %%r10, %%r10\n\t"                                                                        \
	"movq %%gs:0x1478, %%rcx\n\t"                                                                  \
	"movabsq $0x7efff8000, %%rdx\n\t"                                                              \
	"cmpq %%rdx, %%rcx\n\t"                                                                        \
	"cmovbq %%r10, %%rdx\n\t"                                                                      \
	"testl %%r11d, %%r11d\n\t"                                                                     \
	"cmovzq %%r10, %%rdx\n\t"                                                                      \
	"movq %%rdx, %%gs:0x08\n\t"                                                                    \
	"movl $0x40000, %%ecx\n\t"                                                                     \
	"testq %%rdx, %%rdx\n\t"                                                                       \
	"cmovzq %%r10, %%rcx\n\t"                                                                      \
	"movq %%rcx, %%gs:0x10\n\t"

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
// The host signal that delivers guest signals. Fault handling blocks it, so a guest handler runs
// only once the faulting instruction is resolved.
int GuestSignalDispatchSignal();
#endif

} // namespace Common::HostException

#endif /* KYTY_COMMON_HOST_EXCEPTION_H_ */
