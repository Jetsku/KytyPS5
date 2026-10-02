#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SYNCEPOCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SYNCEPOCH_H_

#include "common/profiler.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// The guest synchronization epoch (KYTY_SYNC_EPOCH, default on).
//
// A guest CPU write only has to become visible to GPU work at a point that orders the two: the
// start of a submission, a wait on memory (WAIT_REG_MEM and its relatives, which is how the GPU
// waits for a CPU-written label), a cache invalidation (ACQUIRE_MEM, SURFACE_SYNC), and the
// command-processor packets that write guest memory themselves. Between two such points a CPU
// write races the GPU on the real hardware too: the GPU may read the memory before or after it.
//
// The epoch advances at every such point, conservatively:
//  - before every packet the draw-prep classifier calls a fence (packetClass.h), except register
//    loads from memory (SET_*_REG_INDIRECT), which only read guest memory;
//  - at the start of every command-processor slice (another queue may have run in between);
//  - after every service command the command processor runs (mapping changes, readbacks,
//    deferred label writes) and on every GPU mapping change.
// A fact "range R was made current for the GPU at epoch E" may therefore be reused while the
// epoch is still E, as long as nothing ORDERED changed R since: GPU-side ownership transitions and
// emulator writes of backing bytes, which the facts' users check separately.
namespace Libs::Graphics::SyncEpoch {

namespace Detail {
inline std::atomic<uint64_t> g_epoch {1};
inline std::atomic<uint64_t> g_submission {1};
} // namespace Detail

// KYTY_SYNC_EPOCH=0 disables every consumer of the epoch (they then check on every use).
[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_SYNC_EPOCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Starts at 1; never 0.
[[nodiscard]] inline uint64_t Current() noexcept {
	return Detail::g_epoch.load(std::memory_order_acquire);
}

// Any thread. The RMW orders the new epoch before whatever the caller does next.
inline void Advance() noexcept {
	Detail::g_epoch.fetch_add(1, std::memory_order_acq_rel);
	Profiler::CountFrameEvent(Profiler::FrameEvent::SyncEpochAdvances);
}

// Guest submissions started: the GPU thread advances this at the first slice of every queue
// submission (GuestGpu::Process, ProcessSequenced), coarser than the epoch, which also advances
// at every fence inside a submission. Only KYTY_BDA_SYNC_PER_SUBMISSION reads it
// (BufferCache::SynchronizeBdaBuffers). Starts at 1; never 0.
[[nodiscard]] inline uint64_t CurrentSubmission() noexcept {
	return Detail::g_submission.load(std::memory_order_acquire);
}

inline void AdvanceSubmission() noexcept {
	Detail::g_submission.fetch_add(1, std::memory_order_acq_rel);
}

} // namespace Libs::Graphics::SyncEpoch

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_SYNCEPOCH_H_
