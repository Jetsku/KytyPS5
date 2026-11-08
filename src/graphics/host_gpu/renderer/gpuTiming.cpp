#include "graphics/host_gpu/renderer/gpuTiming.h"

#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/liveSwitch.h"
#include "common/paceLog.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/eopTimestamps.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

namespace {

// Collect only once this many submitted slots are complete (or all of them, when fewer are in
// flight), so the driver query cost is paid per batch rather than per command buffer.
constexpr uint32_t kCollectBatch = 16;
// Upper bound of slots read by one non-forced collection; keeps the producer's work bounded.
constexpr uint32_t kCollectMax = 256;
// Completed samples waiting for the next guest flip. Without flips (loading), newer samples are
// dropped and counted rather than growing without bound.
constexpr size_t kMaxPending = size_t {1} << 16;
// A single command buffer longer than this is treated as an ambiguous wrap and rejected.
constexpr double kMaxSpanNs = 10'000'000'000.0;

using Sample = GpuTimingSample;

struct DeviceInfo {
	vk::Device        device      = nullptr;
	double            period_ns   = 0.0;
	uint32_t          valid_bits  = 0;
	uint64_t          mask        = 0;
	bool              calibrated  = false;
	vk::TimeDomainKHR host_domain = vk::TimeDomainKHR::eDevice;
};

const char* g_calibration_extension = nullptr;

// Guards g_device and g_pending. Producers hold it only to append one collected batch.
std::mutex          g_mutex;
// Cumulative totals of every aggregated flip (Live::GpuTotals).
std::atomic<uint64_t> g_total_busy_ns {0};
std::atomic<uint64_t> g_total_idle_ns {0};
std::atomic<uint64_t> g_total_starved_ns {0};
std::atomic<uint64_t> g_total_cmdbufs {0};
DeviceInfo          g_device;
std::vector<Sample> g_pending;
std::atomic<uint64_t> g_dropped {0};
std::atomic<uint64_t> g_unavailable {0};

struct Span {
	int64_t       start  = 0; // device ticks relative to the aggregation reference
	int64_t       end    = 0;
	const Sample* sample = nullptr;
};

// Aggregator state, touched only by OnGuestFlip (serialized by g_flip_mutex).
std::mutex          g_flip_mutex;
std::vector<Sample> g_work;
std::vector<Span>   g_spans;
bool                g_have_previous_end = false;
uint64_t            g_previous_end_raw  = 0;

// Difference a - b of two raw timestamps, modulo the valid bits and sign-extended, so that a
// counter with fewer than 64 valid bits unwraps correctly for spans below half its range.
int64_t SignedDelta(uint64_t a, uint64_t b, const DeviceInfo& info) {
	uint64_t delta = (a - b) & info.mask;
	if (info.valid_bits < 64 && ((delta >> (info.valid_bits - 1)) & 1u) != 0) {
		delta |= ~info.mask;
	}
	return static_cast<int64_t>(delta);
}

// Converts a calibrated host-domain value to the steady_clock nanoseconds used by NowNs().
uint64_t HostDomainToSteadyNs(uint64_t value) {
#ifdef _WIN32
	// MSVC's steady_clock is QueryPerformanceCounter scaled exactly like this.
	static const uint64_t frequency = [] {
		LARGE_INTEGER f {};
		QueryPerformanceFrequency(&f);
		return static_cast<uint64_t>(f.QuadPart);
	}();
	return (value / frequency) * 1'000'000'000u + (value % frequency) * 1'000'000'000u / frequency;
#else
	// libstdc++/libc++ steady_clock is CLOCK_MONOTONIC in nanoseconds.
	return value;
#endif
}

struct Calibration {
	bool     valid      = false;
	uint64_t device_raw = 0;
	uint64_t steady_ns  = 0;
};

Calibration Calibrate(const DeviceInfo& info) {
	Calibration result;
	if (!info.calibrated || info.device == nullptr) {
		return result;
	}
	std::array<vk::CalibratedTimestampInfoKHR, 2> infos {};
	infos[0].timeDomain = vk::TimeDomainKHR::eDevice;
	infos[1].timeDomain = info.host_domain;
	std::array<uint64_t, 2> values {};
	uint64_t                deviation = 0;
	if (info.device.getCalibratedTimestampsKHR(static_cast<uint32_t>(infos.size()), infos.data(),
	                                           values.data(), &deviation) != vk::Result::eSuccess) {
		return result;
	}
	result.valid      = true;
	result.device_raw = values[0];
	result.steady_ns  = HostDomainToSteadyNs(values[1]);
	return result;
}

uint64_t ToUnsignedNs(double value) {
	return value > 0.0 ? static_cast<uint64_t>(value) : 0;
}

} // namespace

namespace GpuTiming {

bool Enabled() {
	static const bool enabled = [] {
		const auto* setting = std::getenv("KYTY_GPU_TIMING");
		if (setting != nullptr && *setting != '\0') {
			return std::strcmp(setting, "0") != 0;
		}
		return HangTrace::Enabled() || Profiler::AggregateEnabled();
	}();
	return enabled;
}

uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

const char* SelectCalibrationExtension(const std::vector<vk::ExtensionProperties>& available) {
	// KYTY_EOP_TIMESTAMPS=gpu converts guest timestamps with the same calibration.
	if (!Enabled() && !EopTimestamps::GpuEnabled()) {
		return nullptr;
	}
	for (const char* name:
	     {VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME}) {
		const bool found = std::any_of(available.begin(), available.end(), [name](const auto& ext) {
			return std::strcmp(ext.extensionName, name) == 0;
		});
		if (found) {
			return name;
		}
	}
	return nullptr;
}

void NoteCalibrationExtensionEnabled(const char* name) {
	g_calibration_extension = name;
}

bool CalibrationEnabled() {
	return g_calibration_extension != nullptr;
}

void OnGuestFlip() {
	if (!Enabled()) {
		return;
	}
	std::lock_guard flip_lock(g_flip_mutex);
	static const bool provider = [] {
		// Live A/B lines (KYTY_LIVE_FILE) carry the totals (liveSwitch.h).
		Live::SetGpuTotalsProvider([]() -> Live::GpuTotals {
			return {g_total_busy_ns.load(std::memory_order_relaxed),
			        g_total_idle_ns.load(std::memory_order_relaxed),
			        g_total_starved_ns.load(std::memory_order_relaxed),
			        g_total_cmdbufs.load(std::memory_order_relaxed)};
		});
		return true;
	}();
	(void)provider;
	DeviceInfo info;
	{
		std::lock_guard lock(g_mutex);
		info = g_device;
		g_work.swap(g_pending);
	}
	const auto dropped     = g_dropped.exchange(0, std::memory_order_relaxed);
	const auto unavailable = g_unavailable.exchange(0, std::memory_order_relaxed);
	if (info.valid_bits == 0) {
		g_work.clear();
		return;
	}
	// One calibration per flip; drift across a single frame is negligible next to its deviation.
	const auto calibration = Calibrate(info);

	auto& spans = g_spans;
	spans.clear();
	const uint64_t reference =
	    g_have_previous_end ? g_previous_end_raw : (g_work.empty() ? 0 : g_work.front().gpu_start);
	uint64_t rejected = 0;
	for (const auto& sample: g_work) {
		const auto duration = static_cast<int64_t>((sample.gpu_end - sample.gpu_start) & info.mask);
		if (static_cast<double>(duration) * info.period_ns > kMaxSpanNs) {
			++rejected;
			continue;
		}
		const auto start = SignedDelta(sample.gpu_start, reference, info);
		spans.push_back({start, start + duration, &sample});
	}
	std::sort(spans.begin(), spans.end(),
	          [](const Span& a, const Span& b) { return a.start < b.start; });

	const auto to_steady = [&](int64_t relative) {
		const auto from_calibration =
		    static_cast<double>(SignedDelta(reference, calibration.device_raw, info) + relative) *
		    info.period_ns;
		return static_cast<double>(calibration.steady_ns) + from_calibration;
	};

	// KYTY_GPU_SPAN_LOG=<path> (diagnostic, default off): one line per guest command buffer with
	// its GPU start/end and its host record/submit/dispatch/observe times, in microseconds on the
	// hang trace's clock (cp.csv t_us), so GPU idle gaps can be matched with CP and guest events.
	static FILE*   span_log    = nullptr;
	static int64_t span_offset = 0;
	static const bool span_log_on = [] {
		const auto* path = std::getenv("KYTY_GPU_SPAN_LOG");
		if (path == nullptr || path[0] == '\0') {
			return false;
		}
		span_log = std::fopen(path, "w");
		if (span_log == nullptr) {
			return false;
		}
		span_offset = static_cast<int64_t>(NowNs()) - static_cast<int64_t>(HangTrace::NowNs());
		std::fprintf(span_log, "gpu_start_us,gpu_end_us,record_us,submit_us,dispatch_us,observed_us\n");
		return true;
	}();
	if (span_log_on && calibration.valid) {
		const auto rel = [](double steady_ns) {
			return static_cast<long long>((steady_ns - static_cast<double>(span_offset)) / 1000.0);
		};
		for (const auto& span: spans) {
			const auto& s = *span.sample;
			std::fprintf(span_log, "%lld,%lld,%lld,%lld,%lld,%lld\n", rel(to_steady(span.start)),
			             rel(to_steady(span.end)), rel(static_cast<double>(s.record_ns)),
			             rel(static_cast<double>(s.submit_ns)), rel(static_cast<double>(s.dispatch_ns)),
			             rel(static_cast<double>(s.observed_ns)));
		}
		std::fflush(span_log);
	}

	// Union of [start, end] spans. The cursor starts at the previous flip's union end, so an
	// overlap with already-counted work is not counted twice and the gap from the previous
	// flip's last completed span is attributed to this flip.
	bool     have_cursor = g_have_previous_end;
	int64_t  cursor      = 0;
	int64_t  busy        = 0;
	int64_t  idle        = 0;
	int64_t  max_gap     = 0;
	uint64_t gaps        = 0;
	double   starved_ns  = 0.0;
	double   record_latency_ns = 0.0, dispatch_latency_ns = 0.0, observe_lag_ns = 0.0;
	uint64_t latency_samples = 0;
	for (const auto& span: spans) {
		if (!have_cursor) {
			busy += span.end - span.start;
			cursor      = span.end;
			have_cursor = true;
		} else {
			if (span.start > cursor) {
				const auto gap = span.start - cursor;
				idle += gap;
				max_gap = std::max(max_gap, gap);
				++gaps;
				if (calibration.valid && span.sample->dispatch_ns != 0) {
					// Portion of the gap during which this span's buffer had not yet been handed
					// to the driver: the GPU (for guest work) was starved by the CPU side.
					const auto idle_begin = to_steady(cursor);
					const auto dispatch   = static_cast<double>(span.sample->dispatch_ns);
					if (dispatch > idle_begin) {
						starved_ns += std::min(static_cast<double>(gap) * info.period_ns,
						                       dispatch - idle_begin);
					}
				}
			}
			if (span.end > cursor) {
				busy += span.end - std::max(span.start, cursor);
				cursor = span.end;
			}
		}
		if (calibration.valid && span.sample->dispatch_ns != 0) {
			const auto start_steady = to_steady(span.start);
			record_latency_ns += std::max(0.0, start_steady - static_cast<double>(span.sample->record_ns));
			dispatch_latency_ns +=
			    std::max(0.0, start_steady - static_cast<double>(span.sample->dispatch_ns));
			observe_lag_ns +=
			    std::max(0.0, static_cast<double>(span.sample->observed_ns) - to_steady(span.end));
			++latency_samples;
		}
	}
	if (have_cursor) {
		g_previous_end_raw  = (reference + static_cast<uint64_t>(cursor)) & info.mask;
		g_have_previous_end = true;
	}

	HangTrace::GpuFrame frame;
	frame.busy_ns              = ToUnsignedNs(static_cast<double>(busy) * info.period_ns);
	frame.idle_ns              = ToUnsignedNs(static_cast<double>(idle) * info.period_ns);
	frame.max_gap_ns           = ToUnsignedNs(static_cast<double>(max_gap) * info.period_ns);
	frame.starved_ns           = ToUnsignedNs(starved_ns);
	frame.command_buffers      = spans.size();
	frame.latency_samples      = latency_samples;
	frame.record_latency_ns    = ToUnsignedNs(record_latency_ns);
	frame.dispatch_latency_ns  = ToUnsignedNs(dispatch_latency_ns);
	frame.dropped              = dropped + unavailable + rejected;
	HangTrace::RecordGpuFrame(frame);
	PaceLog::NoteGpuFrame(frame.busy_ns, frame.idle_ns, frame.starved_ns, frame.command_buffers,
	                      calibration.valid && have_cursor ? ToUnsignedNs(to_steady(cursor)) : 0);
	g_total_busy_ns.fetch_add(frame.busy_ns, std::memory_order_relaxed);
	g_total_idle_ns.fetch_add(frame.idle_ns, std::memory_order_relaxed);
	g_total_starved_ns.fetch_add(frame.starved_ns, std::memory_order_relaxed);
	g_total_cmdbufs.fetch_add(frame.command_buffers, std::memory_order_relaxed);

	Profiler::AddFrameWait(Profiler::FrameWait::GpuBusy, frame.command_buffers, frame.busy_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::GpuIdle, gaps, frame.idle_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::GpuStarved, gaps, frame.starved_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::GpuRecordToStart, latency_samples,
	                       frame.record_latency_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::GpuDispatchToStart, latency_samples,
	                       frame.dispatch_latency_ns);
	Profiler::AddFrameWait(Profiler::FrameWait::GpuEndToObserved, latency_samples,
	                       ToUnsignedNs(observe_lag_ns));
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuTimingDropped, frame.dropped);

	g_work.clear();
}

} // namespace GpuTiming

GpuTimestampRing::GpuTimestampRing(GraphicContext& graphics, uint32_t pair_capacity)
    : m_graphics(graphics) {
	EXIT_IF(pair_capacity == 0 || graphics.queue_family == static_cast<uint32_t>(-1));
	uint32_t family_count = 0;
	graphics.physical_device.getQueueFamilyProperties(&family_count, nullptr);
	std::vector<vk::QueueFamilyProperties> families(family_count);
	graphics.physical_device.getQueueFamilyProperties(&family_count, families.data());
	EXIT_IF(graphics.queue_family >= family_count);

	DeviceInfo info;
	info.device     = graphics.device;
	info.valid_bits = families[graphics.queue_family].timestampValidBits;
	info.period_ns  = graphics.physical_device_properties.limits.timestampPeriod;
	if (info.valid_bits == 0 || info.valid_bits > 64 || !std::isfinite(info.period_ns) ||
	    info.period_ns <= 0.0) {
		std::printf("Kyty GPU timing disabled: queue family %u has %u timestamp bits, period %g ns\n",
		            graphics.queue_family, info.valid_bits, info.period_ns);
		std::fflush(stdout);
		return;
	}
	// Avoid a shift by 64 for full-width counters.
	info.mask = info.valid_bits == 64 ? std::numeric_limits<uint64_t>::max()
	                                  : (uint64_t {1} << info.valid_bits) - 1u;

#ifdef _WIN32
	info.host_domain = vk::TimeDomainKHR::eQueryPerformanceCounter;
#else
	info.host_domain = vk::TimeDomainKHR::eClockMonotonic;
#endif
	if (g_calibration_extension != nullptr &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR != nullptr &&
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceCalibrateableTimeDomainsKHR != nullptr) {
		uint32_t domain_count = 0;
		if (graphics.physical_device.getCalibrateableTimeDomainsKHR(&domain_count, nullptr) ==
		    vk::Result::eSuccess) {
			std::vector<vk::TimeDomainKHR> domains(domain_count);
			if (graphics.physical_device.getCalibrateableTimeDomainsKHR(&domain_count,
			                                                            domains.data()) ==
			    vk::Result::eSuccess) {
				domains.resize(domain_count);
				const auto has = [&domains](vk::TimeDomainKHR domain) {
					return std::find(domains.begin(), domains.end(), domain) != domains.end();
				};
				info.calibrated = has(vk::TimeDomainKHR::eDevice) && has(info.host_domain);
			}
		}
	}

	vk::QueryPoolCreateInfo create {};
	create.queryType  = vk::QueryType::eTimestamp;
	create.queryCount = pair_capacity * 2u;
	if (graphics.device.createQueryPool(&create, nullptr, &m_pool) != vk::Result::eSuccess) {
		m_pool = nullptr;
		std::printf("Kyty GPU timing disabled: timestamp query pool creation failed\n");
		std::fflush(stdout);
		return;
	}
	m_slots.resize(pair_capacity);
	m_results.resize(size_t {kCollectMax} * 4u);
	m_samples.resize(kCollectMax);
	{
		std::lock_guard lock(g_mutex);
		g_device = info;
		g_pending.reserve(4096);
	}
	std::printf("Kyty GPU timing enabled (KYTY_GPU_TIMING): %u pairs, %u valid bits, %.3f ns/tick, "
	            "calibration %s%s\n",
	            pair_capacity, info.valid_bits, info.period_ns,
	            info.calibrated ? "via " : "unavailable (no CPU->GPU latency)",
	            info.calibrated ? g_calibration_extension : "");
	std::fflush(stdout);
}

GpuTimestampRing::~GpuTimestampRing() {
	if (m_pool == nullptr) {
		return;
	}
	{
		std::lock_guard lock(g_mutex);
		g_device.device     = nullptr;
		g_device.calibrated = false;
	}
	// The owning scheduler's shutdown waited for every submitted tick, so no pending command
	// buffer can still reference the pool.
	m_graphics.device.destroyQueryPool(m_pool, nullptr);
}

void GpuTimestampRing::BeginCommand(vk::CommandBuffer buffer, uint64_t record_ns) {
	if (!Valid()) {
		return;
	}
	EXIT_IF(m_recording != UINT32_MAX);
	const auto capacity = static_cast<uint32_t>(m_slots.size());
	if (m_count == capacity) {
		// Never block or force progress for a diagnostic slot.
		g_dropped.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto index = (m_head + m_count) % capacity;
	++m_count;
	m_recording = index;
	auto& slot  = m_slots[index];
	EXIT_IF(slot.state != SlotState::Free);
	slot           = {};
	slot.tick      = std::numeric_limits<uint64_t>::max(); // never complete before Submitted()
	slot.record_ns = record_ns != 0 ? record_ns : GpuTiming::NowNs();
	slot.state     = SlotState::Recording;
	// Reset must precede the new writes in the same buffer and be outside rendering. The previous
	// generation of this pair was collected, so its commands have completed.
	buffer.resetQueryPool(m_pool, index * 2u, 2);
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, m_pool, index * 2u);
}

void GpuTimestampRing::EndCommand(vk::CommandBuffer buffer) {
	if (m_recording == UINT32_MAX) {
		return;
	}
	auto& slot = m_slots[m_recording];
	EXIT_IF(slot.state != SlotState::Recording);
	buffer.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_pool, m_recording * 2u + 1u);
	slot.state = SlotState::Recorded;
}

uint64_t* GpuTimestampRing::Submitted(uint64_t tick, uint64_t submit_ns) {
	if (m_recording == UINT32_MAX) {
		return nullptr;
	}
	auto& slot = m_slots[m_recording];
	EXIT_IF(slot.state != SlotState::Recorded);
	slot.tick      = tick;
	slot.submit_ns = submit_ns;
	slot.state     = SlotState::Submitted;
	m_recording    = UINT32_MAX;
	return &slot.dispatch_ns;
}

void GpuTimestampRing::Collect(uint64_t known_gpu_tick, bool force) {
	if (!Valid()) {
		return;
	}
	const auto capacity  = static_cast<uint32_t>(m_slots.size());
	const auto submitted = m_count - (m_recording != UINT32_MAX ? 1u : 0u);
	if (submitted == 0) {
		return;
	}
	// Slots are allocated and submitted in tick order, so completion is a prefix of the ring.
	if (!force) {
		const auto probe = std::min(submitted, kCollectBatch);
		if (m_slots[(m_head + probe - 1) % capacity].tick > known_gpu_tick) {
			return;
		}
	}
	const auto limit = force ? submitted : std::min(submitted, kCollectMax);
	uint32_t   ready = 0;
	while (ready < limit && m_slots[(m_head + ready) % capacity].tick <= known_gpu_tick) {
		++ready;
	}
	if (ready == 0) {
		return;
	}
	const auto observed = GpuTiming::NowNs();
	const auto first_run = std::min(ready, capacity - m_head);
	CollectRun(m_head, first_run, observed);
	if (ready > first_run) {
		CollectRun(0, ready - first_run, observed);
	}
	m_head = (m_head + ready) % capacity;
	m_count -= ready;
}

void GpuTimestampRing::CollectRun(uint32_t first, uint32_t count, uint64_t observed_ns) {
	auto& results = m_results;
	auto& samples = m_samples;
	while (count != 0) {
		const auto chunk = std::min(count, kCollectMax);
		// Never WAIT or PARTIAL: the owning ticks are complete, and an unavailable entry is
		// discarded (counted) instead of stalling the producer.
		const auto result = m_graphics.device.getQueryPoolResults(
		    m_pool, first * 2u, chunk * 2u, sizeof(uint64_t) * 4u * chunk, results.data(),
		    sizeof(uint64_t) * 2u,
		    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
		const bool read = result == vk::Result::eSuccess || result == vk::Result::eNotReady;
		uint32_t   valid = 0;
		for (uint32_t i = 0; i < chunk; ++i) {
			auto& slot = m_slots[first + i];
			EXIT_IF(slot.state != SlotState::Submitted);
			if (read && results[i * 4 + 1] != 0 && results[i * 4 + 3] != 0) {
				samples[valid++] = {results[i * 4], results[i * 4 + 2], slot.record_ns,
				                    slot.submit_ns, slot.dispatch_ns, observed_ns};
			}
			slot.state = SlotState::Free;
		}
		if (valid != chunk) {
			g_unavailable.fetch_add(chunk - valid, std::memory_order_relaxed);
		}
		if (valid != 0) {
			std::lock_guard lock(g_mutex);
			const auto room = kMaxPending - std::min(kMaxPending, g_pending.size());
			const auto kept = std::min<size_t>(room, valid);
			g_pending.insert(g_pending.end(), samples.begin(), samples.begin() + kept);
			if (kept != valid) {
				g_dropped.fetch_add(valid - kept, std::memory_order_relaxed);
			}
		}
		first += chunk;
		count -= chunk;
	}
}

} // namespace Libs::Graphics
