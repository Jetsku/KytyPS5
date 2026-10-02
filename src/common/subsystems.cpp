#include "common/subsystems.h"
#include "common/emergencySave.h"

namespace Common {

static Subsystems* g_active_subsystems = nullptr;
static std::atomic<std::shared_ptr<EmergencySave>> g_emergency_save;

Subsystems::Subsystems(bool print): m_print(print) {
	m_active.reserve(16);
	g_active_subsystems = this;
}

Subsystems::~Subsystems() {
	Destroy();
	if (g_active_subsystems == this) {
		g_active_subsystems = nullptr;
	}
}

void Subsystems::Destroy() {
	for (auto it = m_active.rbegin(); it != m_active.rend(); ++it) {
		if (it->shutdown != nullptr) {
			it->shutdown();
		}
	}
	m_active.clear();
}

void Subsystems::EmergencyShutdown() {
	// Persist before logging/profiling are shut down. Each failing thread waits for the same
	// save, so a concurrent EXIT cannot terminate the process immediately and cut it short.
	if (auto save = g_emergency_save.load(std::memory_order_acquire)) {
		if (!save->Request(std::chrono::seconds(2))) {
			// The worker may still be in driver/file I/O. Keep its dependencies alive until _Exit.
			return;
		}
	}
	if (m_emergency_started.exchange(true, std::memory_order_acq_rel)) return;
	for (auto it = m_active.rbegin(); it != m_active.rend(); ++it) {
		if (it->emergency_shutdown != nullptr) {
			it->emergency_shutdown();
		}
	}
	m_active.clear();
}

void Subsystems::SetEmergencySave(std::shared_ptr<EmergencySave> save) {
	auto previous = g_emergency_save.exchange(std::move(save), std::memory_order_acq_rel);
	if (previous != nullptr) previous->Stop();
}

void Subsystems::EmergencyShutdownActive() {
	if (g_active_subsystems != nullptr) {
		g_active_subsystems->EmergencyShutdown();
	}
}

} // namespace Common
