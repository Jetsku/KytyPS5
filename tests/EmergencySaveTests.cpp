#include "common/emergencySave.h"
#include "common/subsystems.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

static std::atomic<bool> saved_before_shutdown {false};
static std::atomic<int> emergency_callbacks {0};

struct TestLifecycle {
	static constexpr const char* name = "EmergencySaveTest";
	static void initialize() {}
	static void emergency_shutdown() {
		if (!saved_before_shutdown) std::abort();
		++emergency_callbacks;
		Common::Subsystems::EmergencyShutdownActive(); // recursive error must not recurse forever
	}
};

static void Require(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", text);
		std::exit(1);
	}
}

int main() {
	// Simulate the crashed thread owning a resource needed by the saver. The caller must
	// return within its budget without destroying the worker or its borrowed resources.
	std::promise<void> release;
	auto released = release.get_future().share();
	std::atomic<int> calls {0};
	std::promise<void> entered;
	Common::EmergencySave saver([&] { ++calls; entered.set_value(); released.wait(); });
	const auto begin = std::chrono::steady_clock::now();
	Require(!saver.Request(30ms), "blocked save must time out");
	Require(std::chrono::steady_clock::now() - begin < 1s, "bounded emergency wait");
	const auto retry = std::chrono::steady_clock::now();
	Require(!saver.Request(1s), "repeat error shares expired deadline");
	Require(std::chrono::steady_clock::now() - retry < 500ms, "repeat does not restart budget");
	entered.get_future().wait();
	release.set_value();
	saver.Stop();
	Require(saver.Request(1s), "same save completes when resource becomes available");
	std::vector<std::thread> callers;
	for (int i = 0; i < 12; ++i) {
		callers.emplace_back([&] { Require(saver.Request(1s), "concurrent request completes"); });
	}
	for (auto& caller: callers) caller.join();
	Require(calls == 1, "repeated and concurrent errors must save only once");
	saver.Stop();

	std::atomic<int> cancelled_calls {0};
	Common::EmergencySave cancelled([&] { ++cancelled_calls; });
	cancelled.Stop();
	Require(!cancelled.Request(1s), "normal shutdown cancels dormant worker");
	Require(cancelled_calls == 0, "normal shutdown does not run emergency save");

	Common::EmergencySave* recursive_ptr = nullptr;
	std::atomic<bool> reentrant_returned {false};
	Common::EmergencySave recursive([&] {
		reentrant_returned = !recursive_ptr->Request(1s);
	});
	recursive_ptr = &recursive;
	Require(recursive.Request(1s), "outer save completes");
	Require(reentrant_returned, "worker must not wait for itself on recursive failure");

	// EXIT_IF invokes emergency shutdown twice (report, then exit). Both share one deadline.
	{
		Common::Subsystems subsystems;
		subsystems.Initialize<TestLifecycle>();
		std::promise<void> unblock;
		auto ready = unblock.get_future().share();
		std::promise<void> worker_entered;
		auto blocked = std::make_shared<Common::EmergencySave>([&] {
			worker_entered.set_value();
			ready.wait();
		});
		Common::Subsystems::SetEmergencySave(blocked);
		const auto start = std::chrono::steady_clock::now();
		Common::Subsystems::EmergencyShutdownActive();
		Common::Subsystems::EmergencyShutdownActive();
		Require(std::chrono::steady_clock::now() - start < 3500ms, "fatal path shares 2 s budget");
		Require(emergency_callbacks == 0, "timed-out save retains emergency dependencies");
		worker_entered.get_future().wait();
		unblock.set_value();
		Common::Subsystems::SetEmergencySave(nullptr);
	}

	// The emergency callback is invoked after persistence and only once, even for recursive
	// error handling. Normal teardown unregisters/joins before destroying the captured owner.
	{
		Common::Subsystems subsystems;
		subsystems.Initialize<TestLifecycle>();
		auto owner = std::make_shared<int>(42);
		std::weak_ptr<int> weak = owner;
		auto emergency = std::make_shared<Common::EmergencySave>([owner] {
			saved_before_shutdown = *owner == 42;
		});
		Common::Subsystems::SetEmergencySave(emergency);
		owner.reset();
		emergency.reset();
		Require(!weak.expired(), "registered worker retains resource owner");
		Common::Subsystems::EmergencyShutdownActive();
		Common::Subsystems::EmergencyShutdownActive();
		Require(emergency_callbacks == 1, "emergency hooks execute once after saving");
		Common::Subsystems::SetEmergencySave(nullptr);
		Require(weak.expired(), "normal unregistration releases owner after worker stops");
	}
	std::puts("Emergency save tests passed");
}
