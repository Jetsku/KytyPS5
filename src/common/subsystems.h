#ifndef KYTY_COMMON_SUBSYSTEMS_H_
#define KYTY_COMMON_SUBSYSTEMS_H_

#include "common/common.h"

#include <cstdio>
#include <atomic>
#include <memory>
#include <vector>

namespace Common {

class EmergencySave;

class Subsystems {
public:
	using Callback = void (*)();

	explicit Subsystems(bool print = false);
	~Subsystems();

	template <typename Lifecycle>
	void Initialize() {
		Lifecycle::initialize();
		m_active.push_back({ShutdownCallback<Lifecycle>(), EmergencyCallback<Lifecycle>()});
		if (m_print) {
			std::printf("Initialized: %s\n", Lifecycle::name);
		}
	}

	void Destroy();
	void EmergencyShutdown();

	static void EmergencyShutdownActive();
	// Set during healthy initialization; clear before normal cache/resource teardown.
	// Clearing joins the worker and is never called by the emergency path.
	static void SetEmergencySave(std::shared_ptr<EmergencySave> save);

	KYTY_CLASS_NO_COPY(Subsystems);

private:
	template <typename Lifecycle>
	static consteval Callback ShutdownCallback() {
		if constexpr (requires { Lifecycle::shutdown; }) {
			return Lifecycle::shutdown;
		}
		return nullptr;
	}

	template <typename Lifecycle>
	static consteval Callback EmergencyCallback() {
		if constexpr (requires { Lifecycle::emergency_shutdown; }) {
			return Lifecycle::emergency_shutdown;
		}
		return nullptr;
	}

	struct Entry {
		Callback shutdown;
		Callback emergency_shutdown;
	};

	std::vector<Entry> m_active;
	bool               m_print;
	std::atomic<bool>  m_emergency_started {false};
};

} // namespace Common

#endif /* KYTY_COMMON_SUBSYSTEMS_H_ */
