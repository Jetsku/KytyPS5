#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/hangWatchdog.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/faultCost.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
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

// KYTY_GPU_WAIT_CALLERS=1 (diagnostic, default off): every GPU-thread wait that blocks is timed and
// keyed by its host call stack; every 10 s the console gets the stacks with the most wait time as
// emulator module offsets ("GpuWaitCallers 10s: ..."), for llvm-symbolizer against the build's PDB.
class GpuWaitCallers {
public:
	static bool Enabled() {
		static const bool enabled = [] {
			const auto* value = std::getenv("KYTY_GPU_WAIT_CALLERS");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}

	static GpuWaitCallers& Instance() {
		static GpuWaitCallers instance;
		return instance;
	}

	struct Probe {
		static constexpr uint32_t Frames = 22;
		void*                     frames[Frames] {};
		uint16_t                  count  = 0;
		uint8_t                   reason = 0;
		std::chrono::steady_clock::time_point start;
	};

	static void Begin(Probe& probe, uint8_t reason) {
#if defined(_WIN32)
		probe.count = RtlCaptureStackBackTrace(1, Probe::Frames, probe.frames, nullptr);
#endif
		probe.reason = reason;
		probe.start  = std::chrono::steady_clock::now();
	}

	// GPU thread only.
	void End(const Probe& probe) {
		const auto now = std::chrono::steady_clock::now();
		const auto us =
		    std::chrono::duration_cast<std::chrono::microseconds>(now - probe.start).count();
		uint64_t key = probe.reason;
		for (uint16_t i = 0; i < probe.count; ++i) {
			key = (key ^ reinterpret_cast<uint64_t>(probe.frames[i])) * 0x100000001b3ull;
		}
		auto& entry = m_entries[key];
		if (entry.frames.empty()) {
			entry.reason = probe.reason;
			entry.frames.assign(probe.frames, probe.frames + probe.count);
		}
		entry.calls++;
		entry.us += static_cast<uint64_t>(us);
		if (m_last.time_since_epoch().count() == 0) {
			m_last = now;
		}
		if (now - m_last >= std::chrono::seconds(10)) {
			Report(std::chrono::duration<double>(now - m_last).count());
			m_last = now;
		}
	}

private:
	struct Entry {
		uint64_t           calls  = 0;
		uint64_t           us     = 0;
		uint8_t            reason = 0;
		std::vector<void*> frames;
	};

	void Report(double seconds) {
		std::vector<const Entry*> sorted;
		uint64_t                  total = 0;
		for (const auto& [key, entry]: m_entries) {
			if (entry.calls != 0) {
				sorted.push_back(&entry);
				total += entry.us;
			}
		}
		std::sort(sorted.begin(), sorted.end(),
		          [](const Entry* a, const Entry* b) { return a->us > b->us; });
#if defined(_WIN32)
		const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
#else
		const uint64_t base = 0;
#endif
		std::printf("GpuWaitCallers %.0fs: %.1f ms waited in %zu stacks\n", seconds,
		            static_cast<double>(total) / 1000.0, sorted.size());
		for (size_t i = 0; i < std::min<size_t>(sorted.size(), 12); ++i) {
			const auto& entry = *sorted[i];
			std::printf("GpuWaitCallers  #%zu reason %u: %" PRIu64 " waits, %.1f ms:", i,
			            static_cast<unsigned>(entry.reason), entry.calls,
			            static_cast<double>(entry.us) / 1000.0);
			for (auto* frame: entry.frames) {
				const auto address = reinterpret_cast<uint64_t>(frame);
				if (address >= base && address - base < 0x10000000ull) {
					std::printf(" +0x%" PRIx64, address - base);
				} else {
					std::printf(" 0x%" PRIx64, address);
				}
			}
			std::printf("\n");
		}
		std::fflush(stdout);
		for (auto& [key, entry]: m_entries) {
			entry.calls = 0;
			entry.us    = 0;
		}
	}

	std::unordered_map<uint64_t, Entry>   m_entries;
	std::chrono::steady_clock::time_point m_last {};
};

} // namespace

MasterSemaphore::MasterSemaphore(GraphicContext& graphics, bool track_dispatch)
    : m_graphics(graphics) {
	if (graphics.submission_queue.Enabled() || track_dispatch) {
		m_submission_progress = std::make_shared<SubmissionProgress>();
	}
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	m_watchdog_timeline = HangWatchdog::RegisterTimeline(
	    reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)));
}

MasterSemaphore::~MasterSemaphore() {
	m_graphics.device.destroySemaphore(m_semaphore, nullptr);
}

void MasterSemaphore::Refresh() {
	HangWatchdog::Scope query("master-counter-query",
	                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                          HangWatchdog::Enabled() ? CurrentTick() : 0);
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-counter-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          CurrentTick(),
		                          static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) DumpDeviceLossDiagnostics(m_graphics, CurrentTick());
		EXIT("MasterSemaphore: counter query failed: %s (submission tick %" PRIu64 ", gpu tick %" PRIu64 ")\n",
		     vk::to_string(result).c_str(), CurrentTick(), m_gpu_tick.load(std::memory_order_acquire));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
	if (HangWatchdog::Enabled()) {
		HangWatchdog::UpdateTimeline(
		    m_watchdog_timeline, CurrentTick(), counter,
		    m_submission_progress ? m_submission_progress->dispatched_tick.load() : CurrentTick() - 1);
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	// KYTY_CP_GAP_STATS (GPU thread only): readback drains, occlusion publication, the rest.
	const auto      reason = Profiler::CurrentGpuWaitReason();
	const CpGaps::Scope gap(reason == Profiler::FrameWait::GpuWaitDrain ? CpGaps::Cat::GpuWaitDrain
	                        : reason == Profiler::FrameWait::GpuWaitOcclusion
	                            ? CpGaps::Cat::GpuWaitOcclusion
	                            : CpGaps::Cat::GpuWaitOther);
	// KYTY_WAIT_ATTRIB: the command processor's GPU waits by call site, reason and readback.
	FaultCost::AttribScope attrib(FaultCost::AttribKind::GpuWait, false);
	attrib.SetReason(static_cast<uint8_t>(reason));
	// Attribute CP-thread blocking to its caller (Profiler::ScopedGpuWaitReason). Other threads
	// (the completion runner, guest threads) wait here by design and are not counted.
	std::optional<Profiler::ScopedFrameWait> frame_wait;
	if (GuestGpu::IsGpuThread() && Profiler::AggregateEnabled()) {
		frame_wait.emplace(Profiler::CurrentGpuWaitReason());
	}
	if (m_submission_progress) {
		auto submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
		HangWatchdog::Scope dispatch(
		    "master-dispatch", reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		    tick, submitted, 0, HangWatchdog::Enabled() ? CurrentTick() : 0);
		while (submitted < tick) {
			m_submission_progress->dispatched_tick.wait(submitted, std::memory_order_acquire);
			submitted = m_submission_progress->dispatched_tick.load(std::memory_order_acquire);
			dispatch.Observed(submitted);
		}
	}
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	std::optional<GpuWaitCallers::Probe> caller_probe;
	if (GpuWaitCallers::Enabled() && GuestGpu::IsGpuThread()) {
		caller_probe.emplace();
		GpuWaitCallers::Begin(*caller_probe, static_cast<uint8_t>(reason));
	}
	HangWatchdog::Scope wait("master-gpu",
	                         reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
	                         tick, HangWatchdog::Enabled() ? KnownGpuTick() : 0, 0,
	                         HangWatchdog::Enabled() ? CurrentTick() : 0);
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	if (result != vk::Result::eSuccess) {
		HangWatchdog::Scope error("master-wait-error",
		                          reinterpret_cast<uint64_t>(static_cast<VkSemaphore>(m_semaphore)),
		                          tick, static_cast<uint64_t>(static_cast<int64_t>(result)));
		if (result == vk::Result::eErrorDeviceLost) {
			DumpDeviceLossDiagnostics(m_graphics, tick);
		}
		EXIT("MasterSemaphore: wait for tick %" PRIu64 " failed: %s (gpu tick %" PRIu64 ")\n", tick,
		     vk::to_string(result).c_str(), m_gpu_tick.load(std::memory_order_acquire));
	}
	if (caller_probe) {
		GpuWaitCallers::Instance().End(*caller_probe);
	}
	Refresh();
}

} // namespace Libs::Graphics
