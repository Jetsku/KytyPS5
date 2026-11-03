#include "graphics/host_gpu/renderer/eopTimestamps.h"

#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/timer.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics {

namespace {

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// A recalibration every 100 ms keeps the drift between the TSC and QPC rates (the TSC frequency is
// measured, rounded to 100 kHz) well below a microsecond at the converted values.
constexpr uint64_t kCalibrationPeriodNs = 100'000'000;
// gpu-verify: a published value may lead its record-time value or the previous value by this much
// (calibration deviation, 20 us in reference-clock ticks) before it counts as a mismatch.
constexpr uint64_t kVerifyToleranceTicks = 2000;

// Per-flip statistics for the hang trace (OnGuestFlip reads and clears them).
struct FlipCounters {
	std::atomic<uint64_t> rewritten {0};
	std::atomic<uint64_t> skipped {0};
	std::atomic<uint64_t> unavailable {0};
	std::atomic<uint64_t> deferred {0};
	std::atomic<int64_t>  shift_sum {0}; // reference-clock ticks (10 ns): published - record
	std::atomic<uint64_t> shift_max {0};
	std::atomic<uint64_t> publish_ns {0};
	std::atomic<uint64_t> publishes {0};
};
FlipCounters g_flip;

void CountShift(uint64_t published, uint64_t record_value) {
	const auto shift = static_cast<int64_t>(published - record_value);
	g_flip.shift_sum.fetch_add(shift, std::memory_order_relaxed);
	const auto magnitude = static_cast<uint64_t>(shift < 0 ? -shift : shift);
	auto       current   = g_flip.shift_max.load(std::memory_order_relaxed);
	while (magnitude > current &&
	       !g_flip.shift_max.compare_exchange_weak(current, magnitude, std::memory_order_relaxed)) {
	}
}

std::atomic<uint64_t> g_verify_mismatches {0};

void VerifyMismatch(const char* what, uint64_t address, uint64_t published, uint64_t reference) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsVerifyMismatches);
	g_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("EopTimestampsVerify: %s: addr=0x%016" PRIx64 " published=0x%016" PRIx64
		     " reference=0x%016" PRIx64 "\n",
		     what, address, published, reference);
	}
}

} // namespace

namespace EopTimestamps {

Mode GetMode() {
	static const Mode mode = [] {
		const auto* value = std::getenv("KYTY_EOP_TIMESTAMPS");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "record") == 0 ||
		    std::strcmp(value, "0") == 0) {
			return Mode::Record;
		}
		if (std::strcmp(value, "gpu") == 0 || std::strcmp(value, "1") == 0) {
			return Mode::Gpu;
		}
		if (std::strcmp(value, "gpu-verify") == 0) {
			return Mode::GpuVerify;
		}
		if (std::strcmp(value, "busy") == 0) {
			return Mode::Busy;
		}
		std::printf("KYTY_EOP_TIMESTAMPS must be record, gpu, gpu-verify or busy (got '%s'): record\n",
		            value);
		return Mode::Record;
	}();
	return mode;
}

uint64_t VerifyMismatches() {
	return g_verify_mismatches.load(std::memory_order_relaxed);
}

} // namespace EopTimestamps

EopTimestampRing::EopTimestampRing(GraphicContext& graphics, uint32_t capacity)
    : m_graphics(graphics), m_ring(capacity) {
	m_verify = EopTimestamps::GetMode() == EopTimestamps::Mode::GpuVerify;
	m_busy   = EopTimestamps::GetMode() == EopTimestamps::Mode::Busy;
	uint32_t family_count = 0;
	graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
	if (graphics.queue_family >= family_count) {
		return;
	}
	m_valid_bits = families[graphics.queue_family].timestampValidBits;
	m_period_ns  = graphics.physical_device_properties.limits.timestampPeriod;
#ifdef _WIN32
	m_host_domain = vk::TimeDomainKHR::eQueryPerformanceCounter;
#else
	m_host_domain = vk::TimeDomainKHR::eClockMonotonic;
#endif
	const char* reason = nullptr;
	if (m_valid_bits == 0 || m_valid_bits > 64 || !(m_period_ns > 0.0)) {
		reason = "the queue family has no timestamps";
	} else if (!GpuTiming::CalibrationEnabled() ||
	           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR == nullptr) {
		reason = "no calibrated timestamps (VK_KHR/EXT_calibrated_timestamps)";
	}
	if (reason == nullptr) {
		vk::QueryPoolCreateInfo create {};
		create.queryType  = vk::QueryType::eTimestamp;
		create.queryCount = m_ring.Capacity();
		if (graphics.device.createQueryPool(&create, nullptr, &m_pool) != vk::Result::eSuccess) {
			m_pool = nullptr;
			reason = "query pool creation failed";
		}
	}
	if (reason == nullptr) {
		m_results.resize(size_t {m_ring.Capacity()} * 2u);
		Recalibrate(NowNs(), true);
		if (!m_map.Valid()) {
			reason = "calibration failed";
		}
	}
	if (reason != nullptr) {
		if (m_pool != nullptr) {
			graphics.device.destroyQueryPool(m_pool, nullptr);
			m_pool = nullptr;
		}
		std::printf("Kyty end-of-pipe timestamps: GPU mode unavailable (%s); record time\n",
		            reason);
		std::fflush(stdout);
		return;
	}
	std::printf("Kyty end-of-pipe timestamps: GPU times%s (KYTY_EOP_TIMESTAMPS), %u queries, "
	            "%u valid bits, %.3f ns/tick\n",
	            m_verify ? ", verified" : "", m_ring.Capacity(), m_valid_bits, m_period_ns);
	std::fflush(stdout);
}

EopTimestampRing::~EopTimestampRing() {
	if (m_pool != nullptr) {
		// The owning scheduler's shutdown waited for every submitted tick.
		m_graphics.device.destroyQueryPool(m_pool, nullptr);
	}
}

void EopTimestampRing::Recalibrate(uint64_t now_ns, bool force) {
	if (!force && m_map.Valid() && now_ns - m_calibrated_ns < kCalibrationPeriodNs) {
		return;
	}
	std::array<vk::CalibratedTimestampInfoKHR, 2> infos {};
	infos[0].timeDomain = vk::TimeDomainKHR::eDevice;
	infos[1].timeDomain = m_host_domain;
	std::array<uint64_t, 2> values {};
	uint64_t                deviation = 0;
	if (m_graphics.device.getCalibratedTimestampsKHR(static_cast<uint32_t>(infos.size()),
	                                                 infos.data(), values.data(), &deviation) !=
	    vk::Result::eSuccess) {
		return;
	}
	// The TSC read between two QPC reads: its QPC is their midpoint.
	const auto qpc_before = Common::Timer::QueryPerformanceCounter();
	const auto tsc        = LibKernel::KernelReadTsc();
	const auto qpc_after  = Common::Timer::QueryPerformanceCounter();
	EopTimestamps::ClockMap map;
	map.device_raw    = values[0];
	map.device_qpc    = values[1];
	map.qpc_frequency = Common::Timer::QueryPerformanceFrequency();
	map.period_ns     = m_period_ns;
	map.valid_bits    = m_valid_bits;
	map.pair_qpc      = qpc_before + (qpc_after - qpc_before) / 2u;
	map.pair_tsc      = tsc;
	map.tsc_frequency = LibKernel::KernelGetTscFrequency();
	if (!map.Valid()) {
		return;
	}
	std::lock_guard lock(m_map_mutex);
	m_map           = map;
	m_calibrated_ns = now_ns;
}

void EopTimestampRing::BeginCommand(const CommandSink& sink, uint64_t tick) {
	if (!Valid()) {
		return;
	}
	(void)m_ring.ResetConsumed(
	    [&](uint32_t first, uint32_t count) { sink.resetQueryPool(m_pool, first, count); });
	if (m_busy && tick != UINT64_MAX) {
		QueueMarker(sink, tick, BeginMarker, true);
	}
}

void EopTimestampRing::EndCommand(const CommandSink& sink, uint64_t tick) {
	if (Valid() && m_busy) {
		QueueMarker(sink, tick, EndMarker, false);
	}
}

void EopTimestampRing::QueueMarker(const CommandSink& sink, uint64_t tick, uint64_t marker,
                                   bool top_of_pipe) {
	const auto slot = m_ring.Allocate();
	if (slot == NoSlot) {
		return; // a missing marker only makes that gap count as busy
	}
	sink.writeTimestamp2(top_of_pipe ? vk::PipelineStageFlagBits2::eTopOfPipe
	                                 : vk::PipelineStageFlagBits2::eAllCommands,
	                     m_pool, slot);
	m_queue.push_back({tick, marker, 0, slot});
}

uint32_t EopTimestampRing::RecordQuery(const CommandSink& sink) {
	if (!Valid()) {
		return NoSlot;
	}
	const auto slot = m_ring.Allocate();
	if (slot == NoSlot) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsUnavailable);
		g_flip.unavailable.fetch_add(1, std::memory_order_relaxed);
		return NoSlot;
	}
	// When everything recorded before it has completed: the end-of-pipe semantics. Allowed inside
	// a rendering instance; it touches no guest resource, so pending batched barriers may follow.
	sink.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_pool, slot);
	return slot;
}

void EopTimestampRing::Queue(uint32_t slot, uint64_t tick, uint64_t address,
                             uint64_t record_value) {
	if (slot == NoSlot) {
		return;
	}
	m_queue.push_back({tick, address, record_value, slot});
}

void EopTimestampRing::PublishEntry(const Entry& entry, bool available, uint64_t device_raw) {
	if (m_busy) {
		if (entry.address == BeginMarker || entry.address == EndMarker) {
			if (available) {
				// A buffer that starts after everything before it ended: the gap was idle.
				if (entry.address == BeginMarker && m_last_raw != 0 && device_raw > m_last_raw) {
					m_idle_raw.fetch_add(device_raw - m_last_raw, std::memory_order_relaxed);
				}
				m_last_raw = std::max(m_last_raw, device_raw);
			}
			return;
		}
		if (available) {
			m_last_raw = std::max(m_last_raw, device_raw);
			device_raw -= std::min(device_raw, m_idle_raw.load(std::memory_order_relaxed));
		}
	}
	uint64_t published = 0;
	if (!available || !EopTimestamps::ToReference(m_map, device_raw, published)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsUnavailable);
		g_flip.unavailable.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto* slot = reinterpret_cast<uint64_t*>(entry.address);
	// The guest may have reused the slot since (a ring of timer slots): keep what it wrote.
	uint64_t current = 0;
	std::memcpy(&current, slot, sizeof(current));
	if (current != entry.record_value) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsSkipped);
		g_flip.skipped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	if (m_verify) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsVerifyChecks);
		if (published + kVerifyToleranceTicks < entry.record_value) {
			VerifyMismatch("GPU time before recording", entry.address, published,
			               entry.record_value);
		}
		if (m_last_published != 0 && published + kVerifyToleranceTicks < m_last_published) {
			VerifyMismatch("GPU time before an earlier timestamp", entry.address, published,
			               m_last_published);
		}
		m_last_published = std::max(m_last_published, published);
	}
	std::memcpy(slot, &published, sizeof(published));
	// Emulator write outside the draw-prep log's fence positions (like deferred labels).
	Coherence::NoteContentWrite(entry.address, sizeof(published), Coherence::Source::CpWrite);
	Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsRewritten);
	g_flip.rewritten.fetch_add(1, std::memory_order_relaxed);
	CountShift(published, entry.record_value);
	if (HangTrace::CpTraceEnabled()) {
		HangTrace::CpEvent event;
		event.event   = "label-ts-gpu";
		event.address = entry.address;
		event.value   = published;
		event.ref     = entry.record_value;
		event.size    = sizeof(published);
		HangTrace::RecordCp(event);
	}
}

void EopTimestampRing::Publish(uint64_t known_gpu_tick) {
	if (m_queue.empty() || m_queue.front().tick > known_gpu_tick) {
		return;
	}
	Profiler::ScopedFrameWait wait(Profiler::FrameWait::EopTimestampPublish);
	const auto start = NowNs();
	Recalibrate(start, false);
	size_t ready = 0;
	while (ready < m_queue.size() && m_queue[ready].tick <= known_gpu_tick) {
		ready++;
	}
	// One read per run of consecutive slots. Never WAIT or PARTIAL: the ticks are complete, and
	// an unavailable result keeps the record-time value.
	size_t index = 0;
	while (index < ready) {
		const auto first = m_queue[index].slot;
		size_t     count = 1;
		while (index + count < ready && m_queue[index + count].slot == first + count) {
			count++;
		}
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_pool, first, static_cast<uint32_t>(count), sizeof(uint64_t) * 2u * count,
		    m_results.data(), sizeof(uint64_t) * 2u,
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		const bool read = result == vk::Result::eSuccess || result == vk::Result::eNotReady;
		for (size_t i = 0; i < count; i++) {
			const auto& entry = m_queue[index + i];
			PublishEntry(entry, read && m_results[i * 2u + 1u] != 0, m_results[i * 2u]);
			m_ring.Consume(entry.slot);
		}
		index += count;
	}
	m_queue.erase(m_queue.begin(), m_queue.begin() + static_cast<std::ptrdiff_t>(ready));
	g_flip.publishes.fetch_add(1, std::memory_order_relaxed);
	g_flip.publish_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
}

uint64_t EopTimestampRing::TakeDeferred(uint32_t slot, uint64_t record_value) {
	if (!Valid() || slot == NoSlot) {
		return record_value;
	}
	std::array<uint64_t, 2> result {};
	const auto              status = m_graphics.device.getQueryPoolResults(
        m_pool, slot, 1, sizeof(result), result.data(), sizeof(uint64_t) * 2u,
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
	EopTimestamps::ClockMap map;
	{
		std::lock_guard lock(m_map_mutex);
		map = m_map;
	}
	uint64_t published = 0;
	if (m_busy) {
		// busy: the idle time measured so far (markers not published yet are not in it).
		result[0] -= std::min(result[0], m_idle_raw.load(std::memory_order_relaxed));
	}
	const bool ok = (status == vk::Result::eSuccess || status == vk::Result::eNotReady) &&
	                result[1] != 0 && EopTimestamps::ToReference(map, result[0], published);
	m_ring.Consume(slot);
	if (!ok) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsUnavailable);
		g_flip.unavailable.fetch_add(1, std::memory_order_relaxed);
		return record_value;
	}
	if (m_verify) {
		// Deferred writes run on the completion runner in tick order, but their slot order is not
		// tracked here: only the record-time bound is checked.
		Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsVerifyChecks);
		if (published + kVerifyToleranceTicks < record_value) {
			VerifyMismatch("deferred GPU time before recording", 0, published, record_value);
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::EopTimestampsDeferred);
	g_flip.deferred.fetch_add(1, std::memory_order_relaxed);
	CountShift(published, record_value);
	return published;
}

namespace EopTimestamps {

namespace {

// Astro Bot 01.018's GPU timer rings seen in the U51 CP trace (8 slots, begin at B + 0x10 + 32 i,
// end at B + 0x110 + 32 i), the dynamic-resolution inputs among them.
constexpr const char* kDefaultRings = "0x3082f4f10,0x3082f5148,0x3135ebbe0,0x313600020,"
                                      "0x313600380,0x313600700,0x300929270,0x300929480";
// Offset in eboot.bin of the pointer to Astro Bot 01.018's dynamic-resolution controller.
constexpr uint64_t kDefaultDrsProbe = 0xee29f68;

std::vector<uint64_t> RingBases() {
	std::vector<uint64_t> bases;
	const auto*           value = std::getenv("KYTY_HANG_TRACE_TIMESTAMP_RINGS");
	std::string           list  = value != nullptr ? value : kDefaultRings;
	if (list == "0") {
		return bases;
	}
	const char* p = list.c_str();
	while (*p != '\0' && bases.size() < HangTrace::EopTimestampFrame::MaxRings) {
		char*      end  = nullptr;
		const auto base = std::strtoull(p, &end, 0);
		if (end == p) {
			break;
		}
		if (base != 0) {
			bases.push_back(base);
		}
		p = *end == ',' ? end + 1 : end;
	}
	return bases;
}

// Guest memory without faulting: the backing view (GPU-visible memory), else a page that is
// readable now (module data, CPU heaps).
bool ReadGuest(uint64_t address, void* data, size_t size) {
	return LibKernel::Memory::TryReadBacking(address, data, size) ||
	       HangTrace::TryReadReadable(address, data, size);
}

// The latest valid (end - begin) of a ring in microseconds, -1 when none is valid.
int64_t LatestRingDelta(uint64_t base) {
	// begin[i] at +0x10 + 32 i, end[i] at +0x110 + 32 i: 0x200 bytes from +0x10.
	std::array<uint64_t, 64> words {};
	if (!ReadGuest(base + 0x10, words.data(), sizeof(words))) {
		return -1;
	}
	uint64_t best_begin = 0;
	int64_t  delta      = -1;
	for (uint32_t i = 0; i < 8; i++) {
		const auto begin = words[i * 4u];
		const auto end   = words[32u + i * 4u];
		if (begin != 0 && end >= begin && begin > best_begin) {
			best_begin = begin;
			delta      = static_cast<int64_t>((end - begin) / 100u); // 100 MHz ticks -> us
		}
	}
	return delta;
}

// Astro Bot's controller object: +0x0c resolution index (0..4: 1920x1080, 2432x1368 (twice),
// 3328x1872, 3840x2160), +0x10 scalable GPU ms, +0x18 total GPU ms, +0x20 target fps, +0x24
// frames with room to upgrade, +0x28 level.
void ProbeDrs(uint64_t probe, HangTrace::EopTimestampFrame& frame) {
	uint64_t module = 0;
	if (probe == 0 || !HangTrace::ModuleBase("eboot.bin", module)) {
		return;
	}
	uint64_t object = 0;
	if (!ReadGuest(module + probe, &object, sizeof(object)) || object == 0) {
		return;
	}
	std::array<uint8_t, 0x30> bytes {};
	if (!ReadGuest(object, bytes.data(), bytes.size())) {
		return;
	}
	int32_t index = 0, fps = 0, room = 0, level = 0;
	double  scalable = 0.0, total = 0.0;
	std::memcpy(&index, bytes.data() + 0x0c, 4);
	std::memcpy(&scalable, bytes.data() + 0x10, 8);
	std::memcpy(&total, bytes.data() + 0x18, 8);
	std::memcpy(&fps, bytes.data() + 0x20, 4);
	std::memcpy(&room, bytes.data() + 0x24, 4);
	std::memcpy(&level, bytes.data() + 0x28, 4);
	const auto plausible = [](double ms) { return std::isfinite(ms) && ms >= 0.0 && ms < 10000.0; };
	if (index < 0 || index > 4 || fps <= 0 || fps > 240 || !plausible(scalable) ||
	    !plausible(total)) {
		return;
	}
	frame.drs_valid        = true;
	frame.drs_index        = index;
	frame.drs_level        = level;
	frame.drs_room_frames  = room;
	frame.drs_fps          = fps;
	frame.drs_scalable_ms  = scalable;
	frame.drs_total_ms     = total;
}

} // namespace

void OnGuestFlip() {
	if (!HangTrace::Enabled()) {
		return;
	}
	static const std::vector<uint64_t> rings = RingBases();
	static const uint64_t              probe = [] {
        const auto* value = std::getenv("KYTY_HANG_TRACE_DRS_PROBE");
        return value != nullptr ? std::strtoull(value, nullptr, 0) : kDefaultDrsProbe;
	}();
	HangTrace::EopTimestampFrame frame;
	frame.rewritten   = g_flip.rewritten.exchange(0, std::memory_order_relaxed);
	frame.skipped     = g_flip.skipped.exchange(0, std::memory_order_relaxed);
	frame.unavailable = g_flip.unavailable.exchange(0, std::memory_order_relaxed);
	frame.deferred    = g_flip.deferred.exchange(0, std::memory_order_relaxed);
	frame.shift_sum   = g_flip.shift_sum.exchange(0, std::memory_order_relaxed);
	frame.shift_max   = g_flip.shift_max.exchange(0, std::memory_order_relaxed);
	frame.publish_ns  = g_flip.publish_ns.exchange(0, std::memory_order_relaxed);
	frame.publishes   = g_flip.publishes.exchange(0, std::memory_order_relaxed);
	frame.rings       = static_cast<uint32_t>(rings.size());
	for (size_t i = 0; i < rings.size(); i++) {
		frame.ring_delta_us[i] = LatestRingDelta(rings[i]);
	}
	ProbeDrs(probe, frame);
	HangTrace::RecordEopTimestamps(frame);
}

} // namespace EopTimestamps

} // namespace Libs::Graphics
