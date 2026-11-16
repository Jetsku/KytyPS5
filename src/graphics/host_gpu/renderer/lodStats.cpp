#include "graphics/host_gpu/renderer/lodStats.h"
#include "common/staticOnce.h"

#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/profiler.h"
#include "graphics/host_gpu/coherenceLog.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

namespace {

constexpr uint64_t CounterBytes = uint64_t {LodStatsCounter::Entries} * 2u * sizeof(uint32_t);
using LodStatsReport::PackEntry;
using LodStatsReport::PackReport;
using LodStatsReport::Unsampled;

} // namespace

bool LodStatsCounter::Enabled() {
	static const bool enabled = [] {
		const auto* mode = std::getenv("KYTY_LOD_STATS_MODE");
		return mode == nullptr || mode[0] == 0 || std::strcmp(mode, "gpu") == 0;
	}();
	return enabled;
}

LodStatsCounter::Publish LodStatsCounter::PublishMode() {
	static const Publish mode = [] {
		const auto result = LodStatsReport::ParsePublish(
		    std::getenv("KYTY_LOD_REPORT_PUBLISH"), std::getenv("KYTY_LOD_REPORT_COMPLETION_WRITE"));
		std::printf("GET_LOD_STATS reports: %s\n", result == Publish::Completion ? "completion"
		                                           : result == Publish::Rewrite  ? "rewrite"
		                                                                         : "record");
		return result;
	}();
	return mode;
}

bool LodStatsCounter::CompletionWriteEnabled() {
	return PublishMode() != Publish::Record;
}

bool LodStatsCounter::CountClamped() {
	static const bool clamped = [] {
		const bool result = LodStatsReport::ParseCountClamped(std::getenv("KYTY_LOD_STATS_COUNT"));
		std::printf("GET_LOD_STATS count: %s\n", result ? "clamp" : "samples");
		return result;
	}();
	return clamped;
}

LodStatsCounter::Plain LodStatsCounter::PlainVariant() {
	static constexpr auto mode_init = [] {
		const auto* value = std::getenv("KYTY_LOD_STATS_PLAIN_VARIANT");
		Plain result = Plain::On;
		if (value != nullptr && value[0] != 0) {
			result = std::strcmp(value, "0") == 0        ? Plain::Off
			         : std::strcmp(value, "verify") == 0 ? Plain::Verify
			                                             : Plain::On;
		}
		if (Enabled()) {
			std::printf("GET_LOD_STATS plain variant: %s\n", result == Plain::Off  ? "off"
			                                                  : result == Plain::On ? "on"
			                                                                        : "verify");
		}
		return result;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(mode_init())>> mode_once;
	const Plain& mode = mode_once.Get(mode_init);
	return mode;
}

LodStatsCounter::LodStatsCounter(RenderContext& context): m_context(context) {}

LodStatsCounter::~LodStatsCounter() = default;

Buffer& LodStatsCounter::CounterBuffer() {
	if (m_counters == nullptr) {
		// Contents stay undefined until the first report resets them; that report is not
		// published.
		const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
		                   vk::BufferUsageFlagBits::eTransferSrc |
		                   vk::BufferUsageFlagBits::eTransferDst;
		m_counters = std::make_unique<Buffer>(m_context.GetGraphics(),
		                                      m_context.GetCommandScheduler(),
		                                      MemoryUsage::DeviceLocal, 0, usage, CounterBytes);
	}
	return *m_counters;
}

Buffer& LodStatsCounter::CanaryBuffer() {
	EXIT_IF(m_canary == nullptr);
	return *m_canary;
}

void LodStatsCounter::CheckCanary(uint64_t slot_offset) {
	m_canary_publish->Invalidate(slot_offset, CounterBytes);
	const auto* words =
	    reinterpret_cast<const uint32_t*>(m_canary_publish->Mapped().data() + slot_offset);
	uint32_t changed = 0;
	uint32_t first   = UINT32_MAX;
	for (uint32_t counter = 0; counter < Entries; counter++) {
		if (words[counter] != Unsampled || words[Entries + counter] != 0u) {
			changed++;
			first = std::min(first, counter);
		}
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsCanaryChecks);
	if (changed != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsCanaryMismatches, changed);
		const auto total = m_canary_mismatches.fetch_add(changed, std::memory_order_relaxed) + changed;
		if (m_canary_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("GET_LOD_STATS plain-variant verify: %u counters recorded by draws classified "
			            "as counter-free (first %u: finest=0x%08x count=%u), %llu so far\n",
			            changed, first, words[first], words[Entries + first],
			            static_cast<unsigned long long>(total));
		}
	}
}

void LodStatsCounter::Report(uint64_t destination, uint32_t size, uint32_t control) {
	KYTY_GPU_OP_SITE("lodstats.report");
	const auto mode = PublishMode();
	// GET_LOD_STATS control bit 19 reports and resets, bit 18 forces a reset.
	const bool reset_requested = ((control >> 18u) & 3u) != 0u;
	// Completion publication writes up to the report size into any buffer that holds the
	// header; the record-time modes keep their U26..U47 rule (only complete reports).
	const bool writes_report = destination != 0 &&
	                           (mode == Publish::Completion ? size >= sizeof(uint32_t)
	                                                        : size >= ReportSize);
	// Astro Bot probes with an empty GET_LOD_STATS (no buffer, no reset) before each report: no
	// memory is written and the counters keep counting. The record-time modes also folded it
	// into their newest statistics.
	if (mode == Publish::Completion && m_initialized && !writes_report && !reset_requested) {
		return;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	auto& counters = CounterBuffer();
	if (m_publish == nullptr) {
		const auto usage = vk::BufferUsageFlagBits::eTransferDst;
		m_publish = std::make_unique<Buffer>(m_context.GetGraphics(), scheduler,
		                                     MemoryUsage::Download, 0, usage,
		                                     PublishSlots * PublishSlotSize);
	}
	// Record-time modes: the guest may read this report as soon as the packet is recorded (EOP
	// labels are written at record time). Write the newest completed statistics now; until the
	// first GPU copy has completed, write an unready report ("no data yet").
	const bool record_write       = mode != Publish::Completion && writes_report;
	auto       record_time_report = std::make_shared<std::array<uint8_t, ReportSize>>();
	if (record_write) {
		auto&                     report = *record_time_report;
		HangTrace::LodReportEvent event;
		{
			std::scoped_lock lock(m_latest_mutex);
			if (m_has_latest) {
				report = m_latest;
			}
			event.has_latest       = m_has_latest;
			event.sampled_counters = m_latest_sampled;
			event.total_samples    = m_latest_samples;
			event.mean_finest_mip  = m_latest_mean_finest;
		}
		m_context.PrepareHostBackingWrite(destination, report.size(),
		                                  RenderContext::HostWriter::LodStats);
		(void)LibKernel::Memory::TryWriteBacking(destination, report.data(), report.size());
		Coherence::NoteContentWrite(destination, report.size(), Coherence::Source::LodStatsWrite);
		if (HangTrace::Enabled()) {
			event.destination    = destination;
			event.control        = control;
			event.pending_copies = m_issued - m_completed.load(std::memory_order_acquire);
			HangTrace::RecordLodReport(event);
		}
	} else if (writes_report && HangTrace::Enabled()) {
		// Completion publication: nothing is written now; the row marks the packet's position.
		HangTrace::RecordLodReport({.destination    = destination,
		                            .control        = control,
		                            .pending_copies = m_issued -
		                                              m_completed.load(std::memory_order_acquire)});
	}

	auto native = scheduler.Current().HelperSink();

	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &barrier, 0, nullptr, 0,
	                       nullptr);

	// Completion publication copies only for reports it writes; the record-time modes copy every
	// packet (each completed copy becomes their newest statistics).
	const bool publish     = m_initialized && (mode != Publish::Completion || writes_report);
	uint64_t   slot_offset = 0;
	if (publish) {
		const auto slot = static_cast<uint32_t>(m_issued % PublishSlots);
		if (m_issued >= PublishSlots && !scheduler.IsFree(m_slot_ticks[slot])) {
			Profiler::ScopedGpuWaitReason wait_reason(Profiler::FrameWait::GpuWaitLodStats);
			scheduler.Wait(m_slot_ticks[slot]);
		}
		slot_offset = uint64_t {slot} * PublishSlotSize;
		// The wait above may have finished the recording; record into the current one.
		native = scheduler.Current().HelperSink();
		const vk::BufferCopy copy {0, slot_offset, CounterBytes};
		native.copyBuffer(counters.Handle(), m_publish->Handle(), 1, &copy);
	}
	// KYTY_LOD_STATS_PLAIN_VARIANT=verify: everything that draws classified as counter-free
	// recorded since the previous report went into the canary, which must still be in its reset
	// state. Checked with the report's copy, then re-armed.
	const bool verify       = PlainVariant() == Plain::Verify;
	const bool check_canary = verify && m_canary_ready && publish;
	if (check_canary) {
		if (m_canary_publish == nullptr) {
			m_canary_publish = std::make_unique<Buffer>(
			    m_context.GetGraphics(), scheduler, MemoryUsage::Download, 0,
			    vk::BufferUsageFlagBits::eTransferDst, PublishSlots * PublishSlotSize);
		}
		const vk::BufferCopy copy {0, slot_offset, CounterBytes};
		native.copyBuffer(m_canary->Handle(), m_canary_publish->Handle(), 1, &copy);
	}

	const bool reset = !m_initialized || reset_requested;
	if ((reset && publish) || check_canary) {
		// The reset overwrites the counters the copy above reads (write-after-read hazard found
		// by synchronization validation): the fills must not start before the copy has read.
		vk::MemoryBarrier war {};
		war.srcAccessMask = vk::AccessFlagBits::eTransferRead;
		war.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &war, 0, nullptr, 0,
		                       nullptr);
	}
	if (reset) {
		constexpr uint64_t MinBytes = uint64_t {Entries} * sizeof(uint32_t);
		native.fillBuffer(counters.Handle(), 0, MinBytes, Unsampled);
		native.fillBuffer(counters.Handle(), MinBytes, MinBytes, 0u);
	}
	if (verify && (!m_canary_ready || check_canary)) {
		if (m_canary == nullptr) {
			const auto usage = vk::BufferUsageFlagBits::eStorageBuffer |
			                   vk::BufferUsageFlagBits::eTransferSrc |
			                   vk::BufferUsageFlagBits::eTransferDst;
			m_canary = std::make_unique<Buffer>(m_context.GetGraphics(), scheduler,
			                                    MemoryUsage::DeviceLocal, 0, usage, CounterBytes);
		}
		constexpr uint64_t MinBytes = uint64_t {Entries} * sizeof(uint32_t);
		native.fillBuffer(m_canary->Handle(), 0, MinBytes, Unsampled);
		native.fillBuffer(m_canary->Handle(), MinBytes, MinBytes, 0u);
		// Draws recorded after the barrier below may bind it.
		m_canary_ready = true;
	}
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eHostRead;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands | vk::PipelineStageFlagBits::eHost,
	                       {}, 1, &barrier, 0, nullptr, 0, nullptr);
	m_initialized = true;

	if (!publish) {
		return;
	}
	m_slot_ticks[m_issued % PublishSlots] = scheduler.CurrentTick();
	++m_issued;
	const auto write_size = writes_report ? std::min<uint32_t>(size, ReportSize) : 0u;
	scheduler.DeferOperation([this, slot_offset, destination, mode, writes_report, write_size,
	                          record_time_report, control, check_canary] {
		if (check_canary) {
			CheckCanary(slot_offset);
		}
		m_publish->Invalidate(slot_offset, CounterBytes);
		const auto* words = reinterpret_cast<const uint32_t*>(m_publish->Mapped().data() + slot_offset);
		if (mode == Publish::Completion) {
			// The packet's own interval, visible once the GPU has produced it: the guest's
			// consumer skips the current slot while its header is zero.
			std::array<uint8_t, ReportSize> exact {};
			const auto summary = PackReport(words, exact.data());
			if (writes_report) {
				m_context.PrepareHostBackingWrite(destination, write_size,
				                                  RenderContext::HostWriter::LodStats);
				(void)LibKernel::Memory::TryWriteBacking(destination, exact.data(), write_size);
				Coherence::NoteContentWrite(destination, write_size,
				                            Coherence::Source::LodStatsWrite);
			}
			if (HangTrace::Enabled()) {
				HangTrace::RecordLodReport({.destination      = destination,
				                            .control          = control,
				                            .has_latest       = writes_report,
				                            .sampled_counters = summary.drawn,
				                            .total_samples    = summary.count_total,
				                            .mean_finest_mip  = summary.mean_finest,
				                            .kind             = HangTrace::LodReportKind::Completion,
				                            .drawn_counters   = summary.drawn,
				                            .counted_counters = summary.counted});
			}
			m_completed.fetch_add(1, std::memory_order_release);
			return;
		}
		std::array<uint8_t, ReportSize> report {};
		const uint32_t valid = 1;
		std::memcpy(report.data(), &valid, sizeof(valid));
		if (mode == Publish::Rewrite && writes_report) {
			// This packet's own interval is now known. The record-time write had to use older
			// statistics; replace them with what hardware writes at this packet, unless the guest
			// has already consumed or rewritten the slot (then a late write would look new).
			std::array<uint8_t, ReportSize> exact {};
			const auto summary = PackReport(words, exact.data());
			std::array<uint8_t, ReportSize> current {};
			const bool unchanged =
			    LibKernel::Memory::TryReadBacking(destination, current.data(), current.size()) &&
			    current == *record_time_report;
			if (unchanged) {
				m_context.PrepareHostBackingWrite(destination, exact.size(),
				                                  RenderContext::HostWriter::LodStats);
				(void)LibKernel::Memory::TryWriteBacking(destination, exact.data(), exact.size());
				Coherence::NoteContentWrite(destination, exact.size(),
				                            Coherence::Source::LodStatsWrite);
			}
			if (HangTrace::Enabled()) {
				HangTrace::RecordLodReport({.destination      = destination,
				                            .control          = control,
				                            .has_latest       = unchanged, // 1 = slot rewritten
				                            .sampled_counters = summary.drawn,
				                            .total_samples    = summary.count_total,
				                            .mean_finest_mip  = summary.mean_finest,
				                            .kind             = HangTrace::LodReportKind::Completion,
				                            .drawn_counters   = summary.drawn,
				                            .counted_counters = summary.counted});
			}
		}
		std::scoped_lock lock(m_latest_mutex);
		uint32_t sampled      = 0;
		uint64_t samples      = 0;
		uint64_t finest_total = 0;
		for (uint32_t counter = 0; counter < Counters; counter++) {
			// Guests issue several reports per frame, each covering part of it; combine the two
			// newest intervals so a texture sampled in only one part is not reported unsampled.
			const auto finest = std::min(words[counter], m_previous_finest[counter]);
			const auto count  = words[Entries + counter] + m_previous_count[counter];
			m_previous_finest[counter] = words[counter];
			m_previous_count[counter]  = words[Entries + counter];
			const auto entry = PackEntry(finest, count, counter);
			if (finest != Unsampled) {
				sampled++;
				samples += count;
				finest_total += std::min<uint32_t>(finest, 14u);
			}
			std::memcpy(report.data() + 64 + counter * sizeof(uint64_t), &entry, sizeof(entry));
		}
		m_latest             = report;
		m_has_latest         = true;
		m_latest_sampled     = sampled;
		m_latest_samples     = samples;
		m_latest_mean_finest = sampled != 0 ? static_cast<double>(finest_total) / sampled : 0.0;
		m_completed.fetch_add(1, std::memory_order_release);
	});
}

} // namespace Libs::Graphics
