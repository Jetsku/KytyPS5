#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

// Bounded driver work. Stop drains accepted tasks and joins before their Vulkan objects die.
// Tasks must publish failures to their owner (std::packaged_task does this automatically).
class PipelineCompileQueue {
public:
	explicit PipelineCompileQueue(size_t threads, size_t limit): m_limit(limit) {
		for (size_t i = 0; i < threads; ++i) m_threads.emplace_back([this] { Run(); });
	}
	~PipelineCompileQueue() { Stop(); }
	PipelineCompileQueue(const PipelineCompileQueue&) = delete;
	PipelineCompileQueue& operator=(const PipelineCompileQueue&) = delete;

	bool Submit(std::function<void()> task, uint64_t ticket = 0) {
		{
			std::lock_guard lock(m_mutex);
			if (m_stopping || m_outstanding >= m_limit) return false;
			m_tasks.push_back({ticket, std::move(task)});
			++m_outstanding;
		}
		m_wake.notify_one();
		return true;
	}
	// A required draw takes priority over speculative look-ahead. Running work is not cancelled.
	void Promote(uint64_t ticket) {
		std::lock_guard lock(m_mutex);
		for (auto it = m_tasks.begin(); it != m_tasks.end(); ++it) {
			if (it->ticket != ticket) continue;
			auto task = std::move(*it);
			m_tasks.erase(it);
			m_tasks.push_front(std::move(task));
			return;
		}
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

private:
	void Run() {
		for (;;) {
			std::function<void()> task;
			{
				std::unique_lock lock(m_mutex);
				m_wake.wait(lock, [this] { return m_stopping || !m_tasks.empty(); });
				if (m_tasks.empty()) return;
				task = std::move(m_tasks.front().run);
				m_tasks.pop_front();
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
	struct Task { uint64_t ticket; std::function<void()> run; };
	std::deque<Task> m_tasks;
	size_t m_outstanding = 0;
	size_t m_limit;
	bool m_stopping = false;
	std::vector<std::thread> m_threads;
};

} // namespace Libs::Graphics
