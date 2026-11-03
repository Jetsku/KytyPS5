#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EOPTIMESTAMPS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EOPTIMESTAMPS_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/eopTimestampClock.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <vector>

// GPU-accurate guest timestamps (KYTY_EOP_TIMESTAMPS).
//
// The guest samples the GPU clock with end-of-pipe events: RELEASE_MEM or EVENT_WRITE_EOP with the
// clock data select. On hardware the value is the time the GPU finished everything before the
// packet. Kyty writes such a timestamp, like every label, when the command processor records the
// packet, so the deltas the guest computes measure recording time. Astro Bot's dynamic resolution
// controller (eboot 0x73e5060) compares those deltas, from its GPU timer rings, with a 16.67 ms
// budget. So when the command processor is fast it raises the resolution whatever the GPU costs.
//
// record (default): unchanged.
// gpu:
// - Every such timestamp is still written with its record-time value at once. Labels, fences and
//   their order are untouched.
// - A timestamp query is also recorded at the same point of the command buffer.
// - Once its tick has completed, the command processor rewrites the slot with the query's time,
//   converted to the guest clock. It does so at its next clock write, where the draw-prep window
//   is empty, unless the guest has written the slot since.
// - A timestamp deferred with its label (TryDeferLabel) gets the converted value from the deferred
//   write instead.
// - The command processor never waits: an unavailable result or slot keeps the record-time value.
// gpu-verify: gpu, and each published value is checked against its record-time value (the GPU
// cannot finish before the packet was recorded) and against the previously published value (the
// single host queue executes in recording order).
//
// busy (adapted from Erlou93/KytyPS5-Yotei's KYTY_FIX_VIRTUAL_GPU_CLOCK): gpu, and the published
// clock leaves out the host GPU's idle time between command buffers (each buffer gets a
// top-of-pipe stamp at its start and an all-commands stamp at its end; the time from one buffer's
// end to the next buffer's start is idle). The guest then measures its GPU work, not the time the
// GPU waited for the emulator to record the next buffer. Ghost of Yotei's dynamic resolution reads
// these timers (Erlou: 3840x2160 -> 1920x1080 + FSR3, 4.63 -> 5.40 fps on their PC). The clock
// falls behind the reference clock by the idle time so far.
//
// COPY_DATA from the clock reads it when the command processor processes the packet, on hardware as
// here, and stays at record time.
namespace Libs::Graphics {

class CommandSink;
struct GraphicContext;

namespace EopTimestamps {

enum class Mode : uint8_t { Record, Gpu, GpuVerify, Busy };
// KYTY_EOP_TIMESTAMPS=record|gpu|gpu-verify|busy. Evaluated once.
[[nodiscard]] Mode GetMode();
[[nodiscard]] inline bool GpuEnabled() {
	return GetMode() != Mode::Record;
}
// gpu-verify: mismatches found so far in this process (always counted; tests read it).
[[nodiscard]] uint64_t VerifyMismatches();

// Once per completed guest flip (VideoOut), with the hang trace only. Writes a timestamps.csv row
// with this flip's rewrite statistics, and the latest (end - begin) of the guest GPU timer rings
// (KYTY_HANG_TRACE_TIMESTAMP_RINGS). It also writes the state of Astro Bot's dynamic-resolution
// controller (KYTY_HANG_TRACE_DRS_PROBE). Both are recorded in every mode.
void OnGuestFlip();

} // namespace EopTimestamps

// The query ring of one guest command scheduler (KYTY_EOP_TIMESTAMPS=gpu). Except TakeDeferred,
// every member runs on the scheduler's recording thread (the GPU thread).
class EopTimestampRing {
public:
	static constexpr uint32_t DefaultCapacity = 16384; // about 18 frames at 880 timestamps each
	static constexpr uint32_t NoSlot          = EopTimestamps::QueryRing::NoSlot;

	explicit EopTimestampRing(GraphicContext& graphics, uint32_t capacity = DefaultCapacity);
	~EopTimestampRing();
	KYTY_CLASS_NO_COPY(EopTimestampRing);

	// False when the queue family has no timestamps or no calibration is available: every call is
	// then a no-op and the timestamps keep their record-time values.
	[[nodiscard]] bool Valid() const noexcept { return m_pool != nullptr; }

	// Directly after the command buffer begins, outside rendering: resets the slots already read.
	// Through a CommandSink: with KYTY_CP_RECORDER the resets and timestamp writes are recorder
	// packets, in stream order, and never drain the recorder.
	void BeginCommand(const CommandSink& sink, uint64_t tick = UINT64_MAX);
	// busy: before the command buffer of `tick` ends, an all-commands stamp marks its end.
	void EndCommand(const CommandSink& sink, uint64_t tick);
	// At an end-of-pipe clock write: records a timestamp query at this point of the command
	// buffer. Returns its slot, or NoSlot (no free slot: the record-time value stays).
	[[nodiscard]] uint32_t RecordQuery(const CommandSink& sink);
	// The slot's value goes to `address`, which now holds `record_value`, once `tick` completed.
	void Queue(uint32_t slot, uint64_t tick, uint64_t address, uint64_t record_value);
	// At a packet boundary with the draw-prep window empty: rewrites every queued timestamp whose
	// tick is known complete (never waits).
	void Publish(uint64_t known_gpu_tick);
	// Any thread, once the slot's tick has completed (a deferred label's write): the converted
	// value, or record_value when unavailable. Frees the slot either way.
	[[nodiscard]] uint64_t TakeDeferred(uint32_t slot, uint64_t record_value);

private:
	struct Entry {
		uint64_t tick         = 0;
		uint64_t address      = 0;
		uint64_t record_value = 0;
		uint32_t slot         = NoSlot;
	};

	void Recalibrate(uint64_t now_ns, bool force);
	void PublishEntry(const Entry& entry, bool available, uint64_t device_raw);
	// busy: queues a command-buffer start (BeginMarker) or end (EndMarker) stamp.
	void QueueMarker(const CommandSink& sink, uint64_t tick, uint64_t marker, bool top_of_pipe);
	static constexpr uint64_t BeginMarker = 1; // Entry::address of the markers (never a guest slot)
	static constexpr uint64_t EndMarker   = 2;

	GraphicContext&          m_graphics;
	vk::QueryPool            m_pool = nullptr;
	EopTimestamps::QueryRing m_ring;
	std::deque<Entry>        m_queue;   // queued timestamps in recording (and tick) order
	std::vector<uint64_t>    m_results; // scratch: {timestamp, availability} per slot
	vk::TimeDomainKHR        m_host_domain = vk::TimeDomainKHR::eDevice;
	uint32_t                 m_valid_bits  = 0;
	double                   m_period_ns   = 0.0;
	// m_map is written by the recording thread under m_map_mutex; TakeDeferred copies it under
	// the mutex. The recording thread reads it without the lock.
	std::mutex              m_map_mutex;
	EopTimestamps::ClockMap m_map;
	uint64_t                m_calibrated_ns = 0;
	bool                    m_verify         = false;
	uint64_t                m_last_published = 0; // gpu-verify: the previous published value
	// busy: idle device ticks so far (also read by TakeDeferred on another thread), and the latest
	// stamp seen in stream order.
	bool                  m_busy = false;
	std::atomic<uint64_t> m_idle_raw {0};
	uint64_t              m_last_raw = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_EOPTIMESTAMPS_H_
