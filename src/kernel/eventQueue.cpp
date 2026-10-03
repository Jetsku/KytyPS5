#include "kernel/eventQueue.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/timer.h"
#include "kernel/eventQueueFilters.h"
#include "libs/errno.h"
#include "libs/libs.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <list>
#include <unordered_map>
#include <vector>

namespace Libs::LibKernel::EventQueue {

LIB_NAME("libkernel", "libkernel");

constexpr uint16_t EV_ADD     = 0x01;
constexpr uint16_t EV_ONESHOT = 0x10;
constexpr uint16_t EV_CLEAR   = 0x20;
constexpr uint16_t EV_ERROR   = 0x4000;

static uint64_t MonotonicTimeNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

namespace {

std::atomic<int> g_coalesce_mode_for_tests {-1};

struct CoalesceStatsCounters {
	std::atomic<uint64_t> coalesced_triggers {0};
	std::atomic<uint64_t> data_changes {0};
	std::atomic<uint64_t> merged_deliveries {0};
	std::atomic<uint64_t> max_merged {0};
};
CoalesceStatsCounters g_coalesce_stats;

EqueueCoalesceMode ReadCoalesceMode() {
	const auto* value = std::getenv("KYTY_EQUEUE_COALESCE");
	auto        mode  = EqueueCoalesceMode::Coalesce;
	if (value != nullptr && std::strcmp(value, "0") == 0) {
		mode = EqueueCoalesceMode::Legacy;
	} else if (value != nullptr && std::strcmp(value, "verify") == 0) {
		mode = EqueueCoalesceMode::Verify;
	}
	std::printf("Kyty event queues: %s (KYTY_EQUEUE_COALESCE)\n",
	            mode == EqueueCoalesceMode::Legacy ? "one queued copy per trigger (legacy)"
	            : mode == EqueueCoalesceMode::Verify
	                ? "repeated triggers coalesce, merges reported (verify)"
	                : "repeated triggers coalesce");
	std::fflush(stdout);
	return mode;
}

} // namespace

EqueueCoalesceMode KernelEqueueCoalesceMode() {
	const int forced = g_coalesce_mode_for_tests.load(std::memory_order_relaxed);
	if (forced >= 0) [[unlikely]] {
		return static_cast<EqueueCoalesceMode>(forced);
	}
	static const EqueueCoalesceMode mode = ReadCoalesceMode();
	return mode;
}

void KernelEqueueSetCoalesceModeForTests(EqueueCoalesceMode mode) {
	g_coalesce_mode_for_tests.store(static_cast<int>(mode), std::memory_order_relaxed);
}

EqueueCoalesceStats KernelEqueueGetCoalesceStats() {
	EqueueCoalesceStats stats;
	stats.coalesced_triggers = g_coalesce_stats.coalesced_triggers.load(std::memory_order_relaxed);
	stats.data_changes       = g_coalesce_stats.data_changes.load(std::memory_order_relaxed);
	stats.merged_deliveries  = g_coalesce_stats.merged_deliveries.load(std::memory_order_relaxed);
	stats.max_merged         = g_coalesce_stats.max_merged.load(std::memory_order_relaxed);
	return stats;
}

void KernelEqueueApplyTrigger(KernelEqueueEvent* event, const KernelEvent& next) {
	EXIT_IF(event == nullptr);
	if (!event->triggered) {
		event->event     = next;
		event->triggered = true;
		return;
	}
	if (KernelEqueueCoalesceMode() == EqueueCoalesceMode::Legacy) {
		event->pending_events.push_back(next);
		return;
	}
	// Exact coalescing: `next` was derived from the pending state, so the filter's counters
	// include this trigger and `data` is the newest value. Nothing else is kept.
	const bool data_change = next.data != event->event.data;
	event->event           = next;
	event->coalesced++;
	event->coalesced_data_change |= data_change;
	g_coalesce_stats.coalesced_triggers.fetch_add(1, std::memory_order_relaxed);
	Profiler::CountFrameEvent(Profiler::FrameEvent::EqueueCoalescedTriggers);
	if (data_change) {
		g_coalesce_stats.data_changes.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::EqueueCoalescedDataChanges);
	}
	// Events the guest does not consume (the legacy queue grew without bound on these): reported
	// at 1024 merged triggers and at every power of two after it.
	const auto merged = event->coalesced;
	if (merged >= 1024u && (merged & (merged - 1u)) == 0u &&
	    KernelEqueueCoalesceMode() == EqueueCoalesceMode::Verify) {
		std::printf("Equeue coalesce verify: ident=0x%" PRIx64 " filter=%d udata=0x%016" PRIx64
		            " has %" PRIu64 " triggers pending as one\n",
		            static_cast<uint64_t>(event->event.ident), static_cast<int>(event->event.filter),
		            reinterpret_cast<uint64_t>(event->event.udata), merged + 1u);
		std::fflush(stdout);
	}
}

static std::unordered_map<KernelEqueue, KernelEqueueRef> g_equeues;
static Common::Mutex                                     g_equeues_mutex;
static uint64_t                                          g_next_equeue = 1;

class KernelEqueuePrivate {
public:
	explicit KernelEqueuePrivate(KernelEqueue handle): m_handle(handle) {}
	virtual ~KernelEqueuePrivate();

	KYTY_CLASS_NO_COPY(KernelEqueuePrivate);

	[[nodiscard]] const std::string& GetName() const { return m_name; }
	void                             SetName(const std::string& m_name) { this->m_name = m_name; }

	int AddEvent(const KernelEqueueEvent& event);
	int TriggerEvent(uintptr_t ident, int16_t filter, void* trigger_data);
	int DeleteEvent(uintptr_t ident, int16_t filter);

	int  GetTriggeredEvents(KernelEvent* ev, int num);
	int  WaitForEvents(KernelEvent* ev, int num, uint32_t micros);
	void Close();

private:
	int  GetTriggeredEventsLegacy(KernelEvent* ev, int num);
	void NoteDelivery(KernelEqueueEvent& event);
	void TriggerExpiredTimers(uint64_t now_ns);
	bool GetNextTimerWaitMicros(uint64_t now_ns, uint32_t* wait_micros) const;

	std::list<KernelEqueueEvent> m_events;
	Common::Mutex                m_mutex;
	Common::CondVar              m_cond_var;
	std::string                  m_name;
	KernelEqueue                 m_handle          = KERNEL_EQUEUE_INVALID;
	bool                         m_closed          = false;
	uint64_t                     m_next_active_seq = 1;
};

KernelEqueuePrivate::~KernelEqueuePrivate() {
	Close();
}

void KernelEqueuePrivate::Close() {
	Common::LockGuard lock(m_mutex);

	if (m_closed) {
		return;
	}
	m_closed = true;
	for (auto& event: m_events) {
		if (event.filter.delete_event_func != nullptr) {
			auto owner = event.filter.owner;
			event.filter.delete_event_func(m_handle, &event);
		}
	}
	m_events.clear();
	m_cond_var.SignalAll();
}

int KernelEqueuePrivate::GetTriggeredEvents(KernelEvent* ev, int num) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(num < 1);
	if (m_closed) {
		return KERNEL_ERROR_EBADF;
	}

	TriggerExpiredTimers(MonotonicTimeNs());

	if (KernelEqueueCoalesceMode() == EqueueCoalesceMode::Legacy) {
		return GetTriggeredEventsLegacy(ev, num);
	}

	// As kqueue_scan: every event pending when the scan starts is reported at most once, in the
	// order it became pending. A level-triggered event that is still pending after its delivery
	// (no EV_CLEAR and no reset) goes to the back of the list, past this scan's marker.
	const uint64_t marker = m_next_active_seq;
	int            ret    = 0;
	while (ret < num) {
		auto next = m_events.end();
		for (auto it = m_events.begin(); it != m_events.end(); ++it) {
			if (it->triggered && it->active_seq < marker &&
			    (next == m_events.end() || it->active_seq < next->active_seq)) {
				next = it;
			}
		}
		if (next == m_events.end()) {
			break;
		}
		auto& event = *next;
		ev[ret++]   = event.event;
		NoteDelivery(event);
		if ((event.event.flags & EV_ONESHOT) != 0) {
			m_events.erase(next);
			continue;
		}
		if (event.filter.reset_func != nullptr) {
			event.filter.reset_func(&event);
		} else if ((event.event.flags & EV_CLEAR) != 0) {
			event.triggered    = false;
			event.event.fflags = 0;
			event.event.data   = 0;
		}
		if (event.triggered) {
			event.active_seq = m_next_active_seq++;
		}
	}

	return ret;
}

void KernelEqueuePrivate::NoteDelivery(KernelEqueueEvent& event) {
	const auto merged      = event.coalesced;
	const bool data_change = event.coalesced_data_change;
	event.coalesced             = 0;
	event.coalesced_data_change = false;
	if (merged == 0) {
		return;
	}
	g_coalesce_stats.merged_deliveries.fetch_add(1, std::memory_order_relaxed);
	auto seen = g_coalesce_stats.max_merged.load(std::memory_order_relaxed);
	while (seen < merged && !g_coalesce_stats.max_merged.compare_exchange_weak(
	                            seen, merged, std::memory_order_relaxed)) {
	}
	if (KernelEqueueCoalesceMode() != EqueueCoalesceMode::Verify) {
		return;
	}
	// Report each event's first merge and every new power of two after it (bounded output).
	if (event.verify_logged != 0 && merged < event.verify_logged * 2u) {
		return;
	}
	event.verify_logged = merged;
	std::printf("Equeue coalesce verify: queue '%s' ident=0x%" PRIx64
	            " filter=%d delivered %" PRIu64 " triggers as one (data %s, fflags=0x%" PRIx32
	            ")\n",
	            m_name.c_str(), static_cast<uint64_t>(event.event.ident),
	            static_cast<int>(event.event.filter), merged + 1u,
	            data_change ? "changed" : "unchanged", event.event.fflags);
	std::fflush(stdout);
}

int KernelEqueuePrivate::GetTriggeredEventsLegacy(KernelEvent* ev, int num) {
	int ret = 0;

	for (auto it = m_events.begin(); it != m_events.end();) {
		auto& event = *it;
		bool  erase = false;
		while (event.triggered) {
			ev[ret++] = event.event;
			if ((event.event.flags & EV_ONESHOT) != 0) {
				erase = true;
				break;
			}

			if (event.filter.reset_func != nullptr) {
				event.filter.reset_func(&event);
			} else if ((event.event.flags & EV_CLEAR) != 0) {
				event.triggered    = false;
				event.event.fflags = 0;
				event.event.data   = 0;
			}

			if (!event.pending_events.empty()) {
				event.event = event.pending_events.front();
				event.pending_events.pop_front();
				event.triggered = true;
			}

			if (ret >= num) {
				break;
			}
		}
		it = (erase ? m_events.erase(it) : std::next(it));
		if (ret >= num) {
			break;
		}
	}

	return ret;
}

void KernelEqueuePrivate::TriggerExpiredTimers(uint64_t now_ns) {
	// Expired timers become pending in deadline order, the order their callouts would have fired.
	// A periodic timer (interval_ns != 0) counts every period that expired in its data and moves
	// its deadline past now; it stays pending (and keeps its place) until it is consumed.
	for (;;) {
		KernelEqueueEvent* earliest = nullptr;
		for (auto& event: m_events) {
			const bool periodic =
			    event.interval_ns != 0 && event.event.filter == KERNEL_EVFILT_TIMER;
			if ((!event.triggered || periodic) && event.deadline_ns != 0 &&
			    event.deadline_ns <= now_ns &&
			    (earliest == nullptr || event.deadline_ns < earliest->deadline_ns)) {
				earliest = &event;
			}
		}
		if (earliest == nullptr) {
			return;
		}
		if (earliest->event.filter == KERNEL_EVFILT_TIMER) {
			const auto count = earliest->interval_ns == 0
			                       ? uint64_t {earliest->triggered ? 0u : 1u}
			                       : 1 + (now_ns - earliest->deadline_ns) / earliest->interval_ns;
			earliest->event.data += static_cast<intptr_t>(count);
			earliest->deadline_ns += count * earliest->interval_ns;
		}
		if (!earliest->triggered) {
			earliest->triggered  = true;
			earliest->active_seq = m_next_active_seq++;
		}
	}
}

bool KernelEqueuePrivate::GetNextTimerWaitMicros(uint64_t now_ns, uint32_t* wait_micros) const {
	EXIT_IF(wait_micros == nullptr);

	uint64_t nearest_deadline = UINT64_MAX;
	for (const auto& event: m_events) {
		if (!event.triggered && event.deadline_ns != 0) {
			nearest_deadline = std::min(nearest_deadline, event.deadline_ns);
		}
	}
	if (nearest_deadline == UINT64_MAX) {
		return false;
	}

	const auto remaining_ns = nearest_deadline > now_ns ? nearest_deadline - now_ns : 0;
	const auto rounded_us   = std::max<uint64_t>(1, (remaining_ns + 999u) / 1000u);
	*wait_micros            = static_cast<uint32_t>(std::min<uint64_t>(rounded_us, UINT32_MAX));
	return true;
}

int KernelEqueuePrivate::WaitForEvents(KernelEvent* ev, int num, uint32_t micros) {
	Common::LockGuard lock(m_mutex);

	EXIT_IF(num < 1);
	if (m_closed) {
		return KERNEL_ERROR_EBADF;
	}

	uint32_t      elapsed = 0;
	Common::Timer t;
	t.Start();

	for (;;) {
		int ret = GetTriggeredEvents(ev, num);

		if (ret != 0 || (elapsed >= micros && micros != 0)) {
			return ret;
		}

		uint32_t   timer_wait = 0;
		const bool has_timer  = GetNextTimerWaitMicros(MonotonicTimeNs(), &timer_wait);
		if (micros == 0 && !has_timer) {
			m_cond_var.Wait(&m_mutex);
		} else {
			const auto external_wait = micros != 0 ? micros - elapsed : UINT32_MAX;
			m_cond_var.WaitFor(&m_mutex,
			                   has_timer ? std::min(external_wait, timer_wait) : external_wait);
		}

		elapsed = static_cast<uint32_t>(t.GetTimeS() * 1000000.0);
	}

	return 0;
}

int KernelEqueuePrivate::AddEvent(const KernelEqueueEvent& event) {
	Common::LockGuard lock(m_mutex);

	if (m_closed) {
		return KERNEL_ERROR_EBADF;
	}
	auto it = std::find_if(m_events.begin(), m_events.end(),
	                       [ident = event.event.ident, filter = event.event.filter](const auto& e) {
		                       return e.event.ident == ident && e.event.filter == filter;
	                       });
	if (it != m_events.end()) {
		TriggerExpiredTimers(MonotonicTimeNs());
		it->deadline_ns = event.deadline_ns;
		it->interval_ns = event.interval_ns;
		it->event.udata = event.event.udata;
		for (auto& pending: it->pending_events) {
			pending.udata = event.event.udata;
		}
	} else {
		auto& added = m_events.emplace_back(event);
		added.pending_events.clear();
		added.coalesced             = 0;
		added.coalesced_data_change = false;
		added.verify_logged         = 0;
		added.active_seq            = added.triggered ? m_next_active_seq++ : 0;
	}

	m_cond_var.Signal();
	return OK;
}

int KernelEqueuePrivate::TriggerEvent(uintptr_t ident, int16_t filter, void* trigger_data) {
	Common::LockGuard lock(m_mutex);

	if (m_closed) {
		return KERNEL_ERROR_EBADF;
	}
	auto it = std::find_if(m_events.begin(), m_events.end(), [ident, filter](const auto& e) {
		return e.event.ident == ident && e.event.filter == filter;
	});
	if (it != m_events.end()) {
		auto&      event          = *it;
		const bool was_triggered = event.triggered;

		if (event.filter.trigger_func != nullptr) {
			event.filter.trigger_func(&event, trigger_data);
		} else {
			event.triggered = true;
		}
		// An event that is already pending keeps its place in the active list (kqueue does not
		// re-queue an active knote); a newly pending one goes to the back.
		if (!was_triggered && event.triggered) {
			event.active_seq = m_next_active_seq++;
		}

		m_cond_var.Signal();

		return OK;
	}

	return KERNEL_ERROR_ENOENT;
}

static void UserEventTriggerFunc(KernelEqueueEvent* event, void* trigger_data) {
	EXIT_IF(event == nullptr);
	event->triggered   = true;
	event->event.data  = reinterpret_cast<intptr_t>(trigger_data);
	event->event.udata = trigger_data;
}

static void AmprEventTriggerFunc(KernelEqueueEvent* event, void* trigger_data) {
	EXIT_IF(event == nullptr);
	KernelEqueueApplyTrigger(
	    event, AmprNextState(event->event, static_cast<uint64_t>(
	                                           reinterpret_cast<uintptr_t>(trigger_data))));
}

static void UserEventResetFunc(KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	if ((event->event.flags & EV_CLEAR) != 0) {
		event->triggered    = false;
		event->event.fflags = 0;
		event->event.data   = 0;
	}
}

int KernelEqueuePrivate::DeleteEvent(uintptr_t ident, int16_t filter) {
	Common::LockGuard lock(m_mutex);

	if (m_closed) {
		return KERNEL_ERROR_EBADF;
	}
	auto it = std::find_if(m_events.begin(), m_events.end(), [ident, filter](const auto& e) {
		return e.event.ident == ident && e.event.filter == filter;
	});
	if (it != m_events.end()) {
		auto& event = *it;

		if (event.filter.delete_event_func != nullptr) {
			auto owner = event.filter.owner;
			event.filter.delete_event_func(m_handle, &event);
		}

		m_events.erase(it);

		return OK;
	}

	return KERNEL_ERROR_ENOENT;
}

KernelEqueueRef KernelPinEqueue(KernelEqueue eq) {
	if (eq == KERNEL_EQUEUE_INVALID) {
		return {};
	}

	Common::LockGuard lock(g_equeues_mutex);
	const auto        entry = g_equeues.find(eq);
	return entry != g_equeues.end() ? entry->second : KernelEqueueRef {};
}

int KYTY_SYSV_ABI KernelCreateEqueue(KernelEqueue* eq, const char* name) {
	PRINT_NAME();

	if (eq == nullptr || name == nullptr) {
		return KERNEL_ERROR_EINVAL;
	}

	{
		Common::LockGuard lock(g_equeues_mutex);
		if (g_next_equeue > static_cast<uint64_t>(std::numeric_limits<KernelEqueue>::max())) {
			EXIT("event queue handle space exhausted\n");
		}
		*eq        = static_cast<KernelEqueue>(g_next_equeue++);
		auto owner = std::make_shared<KernelEqueuePrivate>(*eq);
		owner->SetName(std::string(name));
		g_equeues.emplace(*eq, std::move(owner));
	}

	LOGF("\tEqueue create: %s\n", name);

	return OK;
}

int KYTY_SYSV_ABI KernelAddEvent(KernelEqueue eq, const KernelEqueueEvent& event) {
	auto owner = KernelPinEqueue(eq);
	if (!owner) {
		return KERNEL_ERROR_EBADF;
	}

	return owner->AddEvent(event);
}

int KYTY_SYSV_ABI KernelTriggerEvent(KernelEqueue eq, uintptr_t ident, int16_t filter,
                                     void* trigger_data) {
	auto owner = KernelPinEqueue(eq);
	if (!owner) {
		return KERNEL_ERROR_EBADF;
	}

	return owner->TriggerEvent(ident, filter, trigger_data);
}

int KYTY_SYSV_ABI KernelDeleteEvent(KernelEqueue eq, uintptr_t ident, int16_t filter) {
	auto owner = KernelPinEqueue(eq);
	if (!owner) {
		return KERNEL_ERROR_EBADF;
	}

	return owner->DeleteEvent(ident, filter);
}

int KYTY_SYSV_ABI KernelDeleteEqueue(KernelEqueue eq) {
	PRINT_NAME();

	KernelEqueueRef owner;

	{
		Common::LockGuard lock(g_equeues_mutex);
		const auto        entry = g_equeues.find(eq);
		if (entry == g_equeues.end()) {
			return KERNEL_ERROR_EBADF;
		}
		owner = std::move(entry->second);
		g_equeues.erase(entry);
	}

	LOGF("\tEqueue delete: %s\n", owner->GetName().c_str());
	owner->Close();

	return OK;
}

int KYTY_SYSV_ABI KernelWaitEqueue(KernelEqueue eq, KernelEvent* ev, int num, int* out,
                                   const KernelUseconds* timo) {
	PRINT_NAME();

	auto owner = KernelPinEqueue(eq);
	if (!owner) {
		return KERNEL_ERROR_EBADF;
	}

	if (ev == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}

	if (num < 1) {
		return KERNEL_ERROR_EINVAL;
	}

	EXIT_NOT_IMPLEMENTED(out == nullptr);

	LOGF("\tEqueue wait: %s, caller = 0x%016" PRIx64 ", eq = 0x%016" PRIx64 ", ev = 0x%016" PRIx64
	     ", num = %d, timo = %s, thread_id = %d\n",
	     owner->GetName().c_str(), reinterpret_cast<uint64_t>(__builtin_return_address(0)),
	     static_cast<uint64_t>(eq), reinterpret_cast<uint64_t>(ev), num,
	     (timo == nullptr ? "inf" : fmt::format("{}", *timo).c_str()),
	     Common::Thread::GetThreadIdUnique());

	if (timo == nullptr) {
		*out = owner->WaitForEvents(ev, num, 0);
	}

	if (timo != nullptr) {
		if (*timo == 0) {
			*out = owner->GetTriggeredEvents(ev, num);
		} else {
			*out = owner->WaitForEvents(ev, num, *timo);
		}
	}

	if (*out == KERNEL_ERROR_EBADF) {
		return KERNEL_ERROR_EBADF;
	}
	if (*out == 0) {
		LOGF("\tEqueue wait timedout: %s\n", owner->GetName().c_str());
		return KERNEL_ERROR_ETIMEDOUT;
	}

	LOGF("\tEqueue wait received %u events: ident = 0x%016" PRIx64
	     ", filter = %d, flags = 0x%04" PRIx16 ", fflags = 0x%08" PRIx32 ", data = 0x%016" PRIx64
	     ", udata = 0x%016" PRIx64 "\n",
	     *out, static_cast<uint64_t>(ev[0].ident), ev[0].filter, ev[0].flags, ev[0].fflags,
	     static_cast<uint64_t>(ev[0].data), reinterpret_cast<uint64_t>(ev[0].udata));

	return OK;
}

int KYTY_SYSV_ABI KernelAddUserEvent(KernelEqueue eq, int id) {
	PRINT_NAME();

	LOGF("\t user event add: eq = 0x%016" PRIx64 ", id = %d\n", static_cast<uint64_t>(eq), id);

	KernelEqueueEvent event {};
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = KERNEL_EVFILT_USER;
	event.event.flags         = EV_ADD;
	event.event.fflags        = 0;
	event.event.data          = 0;
	event.event.udata         = nullptr;
	event.filter.trigger_func = UserEventTriggerFunc;
	event.filter.reset_func   = UserEventResetFunc;

	return KernelAddEvent(eq, event);
}

int KYTY_SYSV_ABI KernelAddUserEventEdge(KernelEqueue eq, int id) {
	PRINT_NAME();

	LOGF("\t user event edge add: eq = 0x%016" PRIx64 ", id = %d\n", static_cast<uint64_t>(eq), id);

	KernelEqueueEvent event {};
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = KERNEL_EVFILT_USER;
	event.event.flags         = EV_ADD | EV_CLEAR;
	event.event.fflags        = 0;
	event.event.data          = 0;
	event.event.udata         = nullptr;
	event.filter.trigger_func = UserEventTriggerFunc;
	event.filter.reset_func   = UserEventResetFunc;

	return KernelAddEvent(eq, event);
}

int KYTY_SYSV_ABI KernelTriggerUserEvent(KernelEqueue eq, int id, void* udata) {
	PRINT_NAME();

	LOGF("\t user event trigger: eq = 0x%016" PRIx64 ", id = %d, udata = 0x%016" PRIx64 "\n",
	     static_cast<uint64_t>(eq), id, reinterpret_cast<uint64_t>(udata));

	return KernelTriggerEvent(eq, static_cast<uintptr_t>(id), KERNEL_EVFILT_USER, udata);
}

int KYTY_SYSV_ABI KernelTriggerUserEventForAll(int id, void* udata) {
	int                          triggered = 0;
	std::vector<KernelEqueueRef> queues;

	{
		Common::LockGuard lock(g_equeues_mutex);
		queues.reserve(g_equeues.size());
		for (const auto& [handle, queue]: g_equeues) {
			queues.push_back(queue);
		}
	}
	for (const auto& eq: queues) {
		if (eq->TriggerEvent(static_cast<uintptr_t>(id), KERNEL_EVFILT_USER, udata) == OK) {
			triggered++;
		}
	}

	return (triggered > 0 ? OK : KERNEL_ERROR_ENOENT);
}

int KYTY_SYSV_ABI KernelDeleteUserEvent(KernelEqueue eq, int id) {
	PRINT_NAME();

	LOGF("\t user event delete: eq = 0x%016" PRIx64 ", id = %d\n", static_cast<uint64_t>(eq), id);

	return KernelDeleteEvent(eq, static_cast<uintptr_t>(id), KERNEL_EVFILT_USER);
}

static int AddTimerEvent(KernelEqueue eq, int id, uint64_t delay_ns, bool periodic, void* udata) {
	const auto now_ns = MonotonicTimeNs();
	KernelEqueueEvent event {};
	event.deadline_ns  = delay_ns <= UINT64_MAX - now_ns ? now_ns + delay_ns : UINT64_MAX;
	event.interval_ns  = periodic ? delay_ns : 0;
	event.event.ident  = static_cast<uintptr_t>(id);
	event.event.filter = periodic ? KERNEL_EVFILT_TIMER : KERNEL_EVFILT_HRTIMER;
	event.event.flags  = EV_ADD | (periodic ? EV_CLEAR : EV_ONESHOT);
	event.event.udata  = udata;
	return KernelAddEvent(eq, event);
}

int KYTY_SYSV_ABI KernelAddTimerEvent(KernelEqueue eq, int id, KernelUseconds usec, void* udata) {
	return AddTimerEvent(eq, id, static_cast<uint64_t>(usec) * 1000, true, udata);
}

int KYTY_SYSV_ABI KernelAddHRTimerEvent(KernelEqueue eq, int id, const KernelTimespec* ts,
                                        void* udata) {
	if (ts == nullptr) {
		return KERNEL_ERROR_EFAULT;
	}
	if (ts->tv_sec < 0 || ts->tv_nsec < 0 || ts->tv_nsec >= 1000000000ll ||
	    static_cast<uint64_t>(ts->tv_sec) >
	        (UINT64_MAX - static_cast<uint64_t>(ts->tv_nsec)) / 1000000000ull) {
		return KERNEL_ERROR_EINVAL;
	}

	const auto delay_ns =
	    static_cast<uint64_t>(ts->tv_sec) * 1000000000ull + static_cast<uint64_t>(ts->tv_nsec);
	return AddTimerEvent(eq, id, delay_ns, false, udata);
}

int KYTY_SYSV_ABI KernelDeleteHRTimerEvent(KernelEqueue eq, int id) {
	return KernelDeleteEvent(eq, static_cast<uintptr_t>(id), KERNEL_EVFILT_HRTIMER);
}

int KYTY_SYSV_ABI KernelAddAmprEvent(KernelEqueue eq, int id, void* udata) {
	PRINT_NAME();

	LOGF("\t AMPR event add: eq = 0x%016" PRIx64 ", id = %d, udata = 0x%016" PRIx64 "\n",
	     static_cast<uint64_t>(eq), id, reinterpret_cast<uint64_t>(udata));

	if (eq != KERNEL_EQUEUE_INVALID) {
		KernelEqueueEvent event {};
		event.event.ident         = static_cast<uintptr_t>(id);
		event.event.filter        = KERNEL_EVFILT_USER;
		event.event.flags         = EV_ADD | EV_CLEAR;
		event.event.fflags        = 0;
		event.event.data          = 0;
		event.event.udata         = udata;
		event.filter.trigger_func = AmprEventTriggerFunc;
		event.filter.reset_func   = UserEventResetFunc;
		(void)KernelAddEvent(eq, event);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelAddAmprSystemEvent(KernelEqueue eq, int id, void* udata) {
	PRINT_NAME();

	LOGF("\t AMPR system event add: eq = 0x%016" PRIx64 ", id = %d, udata = 0x%016" PRIx64 "\n",
	     static_cast<uint64_t>(eq), id, reinterpret_cast<uint64_t>(udata));

	return KernelAddAmprEvent(eq, id, udata);
}

int KYTY_SYSV_ABI KernelDeleteAmprEvent(KernelEqueue eq, int id) {
	PRINT_NAME();

	LOGF("\t AMPR event delete: eq = 0x%016" PRIx64 ", id = %d\n", static_cast<uint64_t>(eq), id);

	if (eq != KERNEL_EQUEUE_INVALID) {
		(void)KernelDeleteEvent(eq, static_cast<uintptr_t>(id), KERNEL_EVFILT_USER);
	}

	return OK;
}

int KYTY_SYSV_ABI KernelDeleteAmprSystemEvent(KernelEqueue eq, int id) {
	PRINT_NAME();

	LOGF("\t AMPR system event delete: eq = 0x%016" PRIx64 ", id = %d\n", static_cast<uint64_t>(eq),
	     id);

	return KernelDeleteAmprEvent(eq, id);
}

intptr_t KYTY_SYSV_ABI KernelGetEventData(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr) {
		return ev->data;
	}

	return 0;
}

intptr_t KYTY_SYSV_ABI KernelGetEventFflags(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr) {
		return ev->fflags;
	}

	return 0;
}

int KYTY_SYSV_ABI KernelGetEventFilter(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr) {
		return ev->filter;
	}

	return 0;
}

uintptr_t KYTY_SYSV_ABI KernelGetEventId(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr) {
		return ev->ident;
	}

	return 0;
}

void* KYTY_SYSV_ABI KernelGetEventUserData(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr) {
		return ev->udata;
	}

	return nullptr;
}

int KYTY_SYSV_ABI KernelGetEventError(const KernelEvent* ev) {
	PRINT_NAME();

	if (ev != nullptr && (ev->flags & EV_ERROR) != 0) {
		return static_cast<int>(ev->data);
	}

	return 0;
}

} // namespace Libs::LibKernel::EventQueue
