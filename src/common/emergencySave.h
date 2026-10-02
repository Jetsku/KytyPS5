#ifndef KYTY_COMMON_EMERGENCY_SAVE_H_
#define KYTY_COMMON_EMERGENCY_SAVE_H_

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace Common {

// Created during healthy startup: the faulting thread only signals and waits, never allocates
// a thread or executes the save itself (it may already own a lock needed by the save).
// A timeout does not cancel a driver call. The owner must retain this worker and the callback's
// resources until process exit, or call Stop() before destroying any of them on normal shutdown.
class EmergencySave {
public:
	explicit EmergencySave(std::function<void()> save):
	    m_save(std::move(save)), m_thread([this] { Run(); }), m_worker_id(m_thread.get_id()) {}
	~EmergencySave() { Stop(); }
	EmergencySave(const EmergencySave&) = delete;
	EmergencySave& operator=(const EmergencySave&) = delete;

	// All callers share the first request's deadline as well as its once-only save. Repeated
	// fatal handlers must not restart the wait budget. False: timeout, cancellation, recursion.
	bool Request(std::chrono::milliseconds budget) {
		std::unique_lock lock(m_mutex);
		if (m_finished) return true;
		if (m_stop) return false;
		// IDs can be reused after a joined worker exits; terminal state takes precedence.
		if (std::this_thread::get_id() == m_worker_id) return false;
		if (!m_requested) {
			m_deadline = std::chrono::steady_clock::now() + budget;
			m_requested = true;
		}
		m_cv.notify_all();
		m_cv.wait_until(lock, m_deadline, [this] { return m_finished || m_stop; });
		return m_finished;
	}

	// Normal teardown only, on the owner thread, before resource destruction or clean Save().
	// This may join a running save; emergency exit uses Request's bounded wait instead.
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_cv.notify_all();
		if (m_thread.joinable()) m_thread.join();
	}

private:
	void Run() {
		std::unique_lock lock(m_mutex);
		m_cv.wait(lock, [this] { return m_requested || m_stop; });
		if (m_stop) return;
		lock.unlock();
		m_save();
		lock.lock();
		m_finished = true;
		lock.unlock();
		m_cv.notify_all();
	}

	std::function<void()> m_save;
	std::mutex m_mutex;
	std::condition_variable m_cv;
	bool m_requested = false;
	bool m_finished = false;
	bool m_stop = false;
	std::chrono::steady_clock::time_point m_deadline;
	std::thread m_thread;
	const std::thread::id m_worker_id;
};

} // namespace Common

#endif
