#ifndef EMULATOR_SRC_COMMON_HLEWAITSTATS_H_
#define EMULATOR_SRC_COMMON_HLEWAITSTATS_H_

#include "common/guestPriority.h"

#include <atomic>
#include <cstdint>

#if defined(_M_X64) || defined(__x86_64__)
#include <intrin.h>
#endif

// wholecp measurement branch. KYTY_HLE_WAIT_STATS=1 (default off): time (TSC ticks) and calls per
// guest thread inside the HLE entry points where a guest thread can wait (yield, sleeps, semaphores,
// event flags, event queues by returned event, condition variables, mutex / rwlock acquisition,
// sceAgcSuspendPoint, AGC submits, video-out flip submits). With KYTY_LIVE_FILE, every live line is
// followed by one "HleWait" line per thread with a non-zero total (cumulative):
//   HleWait seq 3 tsc T tid 1234: yield 1200/5000 usleep 0/0 ... (ticks/calls per entry point)
// The scope covers the whole HLE call (blocking and the call's own CPU time).
// Off: one load of a const bool per call.
namespace HleWait {

enum Id : uint32_t {
	Yield,
	Usleep,
	Nanosleep,
	Sema,
	EventFlag,
	EqFlip,
	EqVblank,
	EqVideoOutOther,
	EqGraphics,
	EqUser,
	EqOther,
	EqNone,
	CondWait,
	CondTimed,
	MutexLock,
	RwRead,
	RwWrite,
	AgcSuspend,
	AgcSubmit,
	FlipSubmit,
	Count
};

[[nodiscard]] bool Enabled() noexcept;
void Add(Id id, uint64_t ticks) noexcept;

[[nodiscard]] inline uint64_t Tsc() noexcept {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return 0;
#endif
}

// Times the enclosing scope as `id`; Set() changes the id before the scope ends (event queues).
class Scope {
public:
	explicit Scope(Id id) noexcept: m_id(id), m_start(Enabled() ? Tsc() : 0) {
		Common::GuestPriority::Check(); // KYTY_GUEST_PRIORITY_FLAT (live)
	}
	~Scope() {
		if (m_start != 0) {
			Add(m_id, Tsc() - m_start);
		}
	}
	void Set(Id id) noexcept { m_id = id; }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Id       m_id;
	uint64_t m_start;
};

} // namespace HleWait

#endif // EMULATOR_SRC_COMMON_HLEWAITSTATS_H_
