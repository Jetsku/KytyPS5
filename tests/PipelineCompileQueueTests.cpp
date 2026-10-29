#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <atomic>
#include <barrier>
#include <cstdio>
#include <deque>
#include <future>
#include <stdexcept>

int main() {
	using Libs::Graphics::PipelineCompileQueue;
	// Both workers must enter before either can finish: verifies actual parallel compilation.
	std::barrier ready(3);
	std::promise<void> release;
	auto gate = release.get_future().share();
	std::atomic<int> completed {0};
	PipelineCompileQueue queue(2, 2);
	for (int i = 0; i < 2; ++i) {
		if (!queue.Submit([&] { ready.arrive_and_wait(); gate.wait(); ++completed; })) return 1;
	}
	ready.arrive_and_wait();
	if (queue.Submit([] {})) return 2; // Bound includes the running tasks.
	release.set_value();
	queue.Stop();
	if (completed != 2 || queue.Submit([] {})) return 3;
	queue.Stop(); // Idempotent shutdown.

	std::promise<void> entered, unblock;
	auto unblocked = unblock.get_future().share();
	std::vector<int> order;
	PipelineCompileQueue priority(1, 8);
	priority.Submit([&] { entered.set_value(); unblocked.wait(); });
	entered.get_future().wait();
	priority.Submit([&] { order.push_back(1); }, 1);
	priority.Submit([&] { order.push_back(2); }, 2);
	priority.Submit([&] { order.push_back(3); }, 3);
	priority.Promote(3);
	unblock.set_value();
	priority.Stop();
	if (order != std::vector<int> {3, 1, 2}) return 8;

	// Shutdown drains queued work, including an exception published through packaged_task.
	std::atomic<int> drained {0};
	auto failure = std::make_shared<std::packaged_task<void()>>([] { throw std::runtime_error("driver failure"); });
	auto result = failure->get_future();
	{
		PipelineCompileQueue draining(1, 64);
		if (!draining.Submit([failure] { (*failure)(); })) return 4;
		for (int i = 0; i < 32; ++i) if (!draining.Submit([&] { ++drained; })) return 5;
	}
	if (drained != 32) return 6;
	try { result.get(); return 7; } catch (const std::runtime_error&) {}

	// KYTY_PIPELINE_SKIP_ORDER: worker 0 takes the oldest task, the others the most recently asked
	// (the latest submitted among equal stamps); a task no draw asked for in stale_ms only goes to a
	// background worker, or to any while the caller is stalled.
	{
		using Task = PipelineCompileQueue::Task;
		PipelineCompileQueue::Ordering ordering;
		ordering.recent_first       = true;
		ordering.background_workers = 1;
		ordering.stale_ms           = 2000;
		std::deque<Task> tasks;
		tasks.push_back({1, {}, 1000});
		tasks.push_back({2, {}, 5000});
		tasks.push_back({3, {}, 5000});
		tasks.push_back({4, {}, 3000});
		const auto pick = [&](size_t worker, uint64_t now, uint64_t stalled = 0) {
			const auto index = PipelineCompileQueue::PickOrdered(tasks, worker, now, stalled, ordering);
			return index.has_value() ? tasks[*index].ticket : 0u;
		};
		if (pick(0, 5100) != 1) return 9;  // worker 0: the oldest, a background worker
		if (pick(1, 5100) != 3) return 10; // the most recently asked, the later of two
		if (pick(1, 8000) != 0) return 11; // stale: not for a foreground worker
		if (pick(1, 8000, 8400) != 3) return 12; // stalled: any worker helps
		if (pick(0, 8000) != 1) return 13; // a background worker takes stale work
		if (PipelineCompileQueue::PickOrdered({}, 1, 0, 0, ordering).has_value()) return 14;
	}
	// With the ordering, every accepted task still runs, and Stop drains the queue.
	{
		PipelineCompileQueue::Ordering ordering;
		ordering.recent_first = true;
		std::atomic<int> ran {0};
		{
			PipelineCompileQueue ordered(3, 64, {}, ordering);
			for (uint64_t i = 1; i <= 32; ++i) {
				if (!ordered.Submit([&] { ++ran; }, i)) return 15;
			}
			ordered.Promote(7);
			ordered.NoteStall();
		}
		if (ran != 32) return 16;
	}
	std::puts("Pipeline compiler: parallelism, bounded admission, drain, shutdown, error publication and "
	          "ordering passed");
	return 0;
}
