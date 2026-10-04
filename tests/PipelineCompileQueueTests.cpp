#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <atomic>
#include <barrier>
#include <cstdio>
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
	std::puts("Pipeline compiler: parallelism, bounded admission, drain, shutdown and error publication passed");
	return 0;
}
