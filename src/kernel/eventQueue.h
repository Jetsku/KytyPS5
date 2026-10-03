#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_EVENTQUEUE_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_EVENTQUEUE_H_

#include "common/abi.h"
#include "common/common.h"
#include "kernel/pthread.h"

#include <deque>
#include <memory>

namespace Libs::LibKernel::EventQueue {

constexpr int16_t KERNEL_EVFILT_TIMER     = -7;
constexpr int16_t KERNEL_EVFILT_READ      = -1;
constexpr int16_t KERNEL_EVFILT_WRITE     = -2;
constexpr int16_t KERNEL_EVFILT_USER      = -11;
constexpr int16_t KERNEL_EVFILT_FILE      = -4;
constexpr int16_t KERNEL_EVFILT_GRAPHICS  = -14;
constexpr int16_t KERNEL_EVFILT_VIDEO_OUT = -13;
constexpr int16_t KERNEL_EVFILT_HRTIMER   = -15;

class KernelEqueuePrivate;
struct KernelEqueueEvent;

using KernelEqueue    = int64_t;
using KernelEqueueRef = std::shared_ptr<KernelEqueuePrivate>;

constexpr KernelEqueue KERNEL_EQUEUE_INVALID = 0;

using trigger_func_t = void (*)(KernelEqueueEvent* event, void* trigger_data);
using reset_func_t   = void (*)(KernelEqueueEvent* event);
using delete_func_t  = void (*)(KernelEqueue eq, KernelEqueueEvent* event);

struct KernelEvent {
	uintptr_t ident  = 0;
	int16_t   filter = 0;
	uint16_t  flags  = 0;
	uint32_t  fflags = 0;
	intptr_t  data   = 0;
	void*     udata  = nullptr;
};

struct KernelFilter {
	void*                 data = nullptr;
	std::shared_ptr<void> owner;
	trigger_func_t        trigger_func      = nullptr;
	reset_func_t          reset_func        = nullptr;
	delete_func_t         delete_event_func = nullptr;
};

struct KernelEqueueEvent {
	bool                    triggered   = false;
	uint64_t                deadline_ns = 0;
	uint64_t                interval_ns = 0;
	KernelEvent             event;
	KernelFilter            filter;
	// Legacy mode only (KYTY_EQUEUE_COALESCE=0): one entry per trigger that arrived while the
	// event was already pending. By default such triggers are merged into `event`, the way a
	// kqueue knote is updated in place (see KernelEqueueApplyTrigger).
	std::deque<KernelEvent> pending_events;
	// The event's place in its queue's active list, which sets the delivery order (kqueue_scan):
	// assigned when the event becomes pending, and again when a level-triggered event is re-queued
	// behind the others after its delivery.
	uint64_t active_seq = 0;
	// Triggers merged into the pending state since its last delivery, and whether one of them
	// replaced a different `data` value. Diagnostics only (FrameEvent.EqueueCoalesced*, and the
	// KYTY_EQUEUE_COALESCE=verify log).
	uint64_t coalesced             = 0;
	bool     coalesced_data_change = false;
	uint64_t verify_logged         = 0;
};

// How a trigger reaches an event that is already pending (KYTY_EQUEUE_COALESCE):
// - Coalesce (default, "1"): the pending state absorbs it. The filter's counters keep counting
//   and `data` holds the newest value, so one bounded state per registered event, as in kqueue.
// - Legacy ("0"): the old behaviour, one queued copy per trigger (unbounded).
// - Verify ("verify"): Coalesce, and also prints every event whose merged-trigger count reaches a
//   new power of two, with its queue, ident, filter and whether `data` changed.
enum class EqueueCoalesceMode { Legacy, Coalesce, Verify };
[[nodiscard]] EqueueCoalesceMode KernelEqueueCoalesceMode();
void                             KernelEqueueSetCoalesceModeForTests(EqueueCoalesceMode mode);

// For filters' trigger functions, under the queue lock: `next` is the event state after one more
// trigger, computed from the event's current state (the pending one, or the cleared one).
void KernelEqueueApplyTrigger(KernelEqueueEvent* event, const KernelEvent& next);

struct EqueueCoalesceStats {
	uint64_t coalesced_triggers = 0; // triggers merged into an already pending event
	uint64_t data_changes       = 0; // of those, triggers that replaced a different `data`
	uint64_t merged_deliveries  = 0; // deliveries that carried at least one merged trigger
	uint64_t max_merged         = 0; // most triggers merged into one delivery
};
[[nodiscard]] EqueueCoalesceStats KernelEqueueGetCoalesceStats();

[[nodiscard]] KernelEqueueRef KernelPinEqueue(KernelEqueue eq);

int KYTY_SYSV_ABI KernelAddEvent(KernelEqueue eq, const KernelEqueueEvent& event);
int KYTY_SYSV_ABI KernelTriggerEvent(KernelEqueue eq, uintptr_t ident, int16_t filter,
                                     void* trigger_data);
int KYTY_SYSV_ABI KernelDeleteEvent(KernelEqueue eq, uintptr_t ident, int16_t filter);

int KYTY_SYSV_ABI KernelCreateEqueue(KernelEqueue* eq, const char* name);
int KYTY_SYSV_ABI KernelDeleteEqueue(KernelEqueue eq);
int KYTY_SYSV_ABI KernelWaitEqueue(KernelEqueue eq, KernelEvent* ev, int num, int* out,
                                   const KernelUseconds* timo);
int KYTY_SYSV_ABI KernelAddUserEvent(KernelEqueue eq, int id);
int KYTY_SYSV_ABI KernelAddUserEventEdge(KernelEqueue eq, int id);
int KYTY_SYSV_ABI KernelTriggerUserEvent(KernelEqueue eq, int id, void* udata);
int KYTY_SYSV_ABI KernelTriggerUserEventForAll(int id, void* udata);
int KYTY_SYSV_ABI KernelDeleteUserEvent(KernelEqueue eq, int id);
int KYTY_SYSV_ABI KernelAddTimerEvent(KernelEqueue eq, int id, KernelUseconds usec,
                                      void* udata);
int KYTY_SYSV_ABI KernelAddHRTimerEvent(KernelEqueue eq, int id, const KernelTimespec* ts,
                                        void* udata);
int KYTY_SYSV_ABI KernelDeleteHRTimerEvent(KernelEqueue eq, int id);
int KYTY_SYSV_ABI KernelAddAmprEvent(KernelEqueue eq, int id, void* udata);
int KYTY_SYSV_ABI KernelAddAmprSystemEvent(KernelEqueue eq, int id, void* udata);
int KYTY_SYSV_ABI KernelDeleteAmprEvent(KernelEqueue eq, int id);
int KYTY_SYSV_ABI KernelDeleteAmprSystemEvent(KernelEqueue eq, int id);

intptr_t KYTY_SYSV_ABI  KernelGetEventData(const KernelEvent* ev);
intptr_t KYTY_SYSV_ABI  KernelGetEventFflags(const KernelEvent* ev);
int KYTY_SYSV_ABI       KernelGetEventFilter(const KernelEvent* ev);
uintptr_t KYTY_SYSV_ABI KernelGetEventId(const KernelEvent* ev);
void* KYTY_SYSV_ABI     KernelGetEventUserData(const KernelEvent* ev);
int KYTY_SYSV_ABI       KernelGetEventError(const KernelEvent* ev);

} // namespace Libs::LibKernel::EventQueue

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_EVENTQUEUE_H_ */
