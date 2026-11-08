#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// KYTY_PIPELINE_SKIP_ORDER (pipelineCache.cpp; adapted from usa-ratt/KytyPS5-GT7 rounds 141 and
// 145, 3adca0037 / bf5ffc229): the order the workers take tasks in. Off (the default): first in,
// first out, a promoted task first.
struct PipelineCompileOrdering {
	// The task a draw asked for most recently first (each Submit and Promote stamps it); worker 0
	// takes the oldest instead, so that none starves.
	bool recent_first = false;
	// Tasks no draw asked for in the last stale_ms run on the first background_workers workers
	// only (the others leave the cores to the game), except for stall_ms after NoteStall.
	size_t   background_workers = 2;
	uint64_t stale_ms           = 2000;
	uint64_t stall_ms           = 500;
};

// Bounded driver work. Stop drains accepted tasks and joins before their Vulkan objects die.
// Tasks must publish failures to their owner (std::packaged_task does this automatically).
class PipelineCompileQueue {
public:
	using Ordering = PipelineCompileOrdering;
	struct Task {
		uint64_t              ticket   = 0;
		std::function<void()> run;
		uint64_t              asked_ms = 0;
	};

	// `thread_init` runs first on each worker (Common::PlaceCurrentThreadCold in the emulator).
	explicit PipelineCompileQueue(size_t threads, size_t limit, std::function<void()> thread_init = {},
	                              Ordering ordering = {})
	    : m_limit(limit), m_ordering(ordering) {
		for (size_t i = 0; i < threads; ++i) {
			m_threads.emplace_back([this, thread_init, i] {
				if (thread_init) thread_init();
				Run(i);
			});
		}
	}
	~PipelineCompileQueue() { Stop(); }
	PipelineCompileQueue(const PipelineCompileQueue&) = delete;
	PipelineCompileQueue& operator=(const PipelineCompileQueue&) = delete;

	// `urgent`: work a dispatch is waiting for; it goes first and is accepted past the limit.
	bool Submit(std::function<void()> task, uint64_t ticket = 0, bool urgent = false) {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopping || (!urgent && m_outstanding >= m_limit)) return false;
			if (urgent) {
				m_tasks.push_front({ticket, std::move(task), NowMs()});
			} else {
				m_tasks.push_back({ticket, std::move(task), NowMs()});
			}
			++m_outstanding;
		}
		m_wake.notify_one();
		return true;
	}
	// A required draw takes priority over speculative look-ahead. Running work is not cancelled.
	void Promote(uint64_t ticket) {
		{
			std::lock_guard lock(m_mutex);
			for (auto it = m_tasks.begin(); it != m_tasks.end(); ++it) {
				if (it->ticket != ticket) continue;
				auto task     = std::move(*it);
				task.asked_ms = NowMs();
				m_tasks.erase(it);
				m_tasks.push_front(std::move(task));
				break;
			}
		}
		if (m_ordering.recent_first) m_wake.notify_all();
	}
	// With the ordering: the caller is stalled on a task (a draw that may not be skipped waits for
	// it), so every worker may take any task for the next stall_ms; the game is stopped meanwhile.
	void NoteStall() {
		if (!m_ordering.recent_first) return;
		{
			std::lock_guard lock(m_mutex);
			m_stalled_until = NowMs() + m_ordering.stall_ms;
		}
		m_wake.notify_all();
	}
	// Drops the tasks that have not started (running ones finish). Their owners must not wait for
	// them any more: a std::packaged_task or promise left unfulfilled reports a broken promise.
	size_t DropPending() {
		std::lock_guard lock(m_mutex);
		const auto dropped = m_tasks.size();
		m_tasks.clear();
		m_outstanding -= dropped;
		return dropped;
	}
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stopping = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) if (thread.joinable()) thread.join();
		m_threads.clear();
	}

	// With the ordering: the index of the task worker `worker` takes now, if any (tests pass their
	// own clock).
	[[nodiscard]] static std::optional<size_t> PickOrdered(const std::deque<Task>& tasks, size_t worker,
	                                                       uint64_t now_ms, uint64_t stalled_until_ms,
	                                                       const Ordering& ordering) {
		if (tasks.empty()) return std::nullopt;
		size_t pick = 0;
		if (worker != 0) {
			// The most recently asked; among equal stamps the latest submitted.
			for (size_t i = 1; i < tasks.size(); ++i) {
				if (tasks[i].asked_ms >= tasks[pick].asked_ms) pick = i;
			}
		}
		const bool background = worker < ordering.background_workers || now_ms < stalled_until_ms;
		if (background || now_ms - tasks[pick].asked_ms < ordering.stale_ms) return pick;
		return std::nullopt;
	}

private:
	static uint64_t NowMs() {
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                 std::chrono::steady_clock::now().time_since_epoch())
		                                 .count());
	}

	void Run(size_t index) {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock lock(m_mutex);
				if (!m_ordering.recent_first) {
					m_wake.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
					if (m_tasks.empty()) return;
					task = std::move(m_tasks.front().run);
					m_tasks.pop_front();
				} else {
					std::optional<size_t> pick;
					for (;;) {
						if (m_stopping) {
							// Accepted tasks are drained in any order.
							if (m_tasks.empty()) return;
							pick = 0;
							break;
						}
						pick = PickOrdered(m_tasks, index, NowMs(), m_stalled_until, m_ordering);
						if (pick.has_value()) break;
						// A task turns stale with time alone: look again every 200 ms.
						if (m_tasks.empty()) {
							m_wake.wait(lock);
						} else {
							m_wake.wait_for(lock, std::chrono::milliseconds(200));
						}
					}
					task = std::move(m_tasks[*pick].run);
					m_tasks.erase(m_tasks.begin() + static_cast<std::ptrdiff_t>(*pick));
				}
			}
			task();
			{
				std::lock_guard lock(m_mutex);
				--m_outstanding;
			}
		}
	}

	std::mutex m_mutex;
	std::condition_variable m_wake;
	std::deque<Task> m_tasks;
	size_t m_outstanding = 0;
	size_t m_limit;
	Ordering m_ordering;
	uint64_t m_stalled_until = 0;
	bool m_stopping = false;
	std::vector<std::thread> m_threads;
};

} // namespace Libs::Graphics
