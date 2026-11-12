#include "graphics/host_gpu/renderer/cache/inplaceHeap.h"


#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

uint64_t EnvU64(const char* name, uint64_t fallback) {
	const auto* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' ? std::strtoull(value, nullptr, 0) : fallback;
}

int64_t NowMs() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

void Say(const std::string& text) {
	Log::WriteToConsoleAndLog(text);
}

} // namespace

bool InplaceHeap::Requested() noexcept {
	static const bool requested = [] {
		const auto* value = std::getenv("KYTY_INPLACE");
		return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "verify") == 0);
	}();
	return requested;
}

InplaceHeap::Mode InplaceHeap::ConfiguredMode() noexcept {
	if (!Requested()) {
		return Mode::Off;
	}
	static const Mode mode = [] {
		const auto* label = std::getenv("KYTY_LABEL_MODE");
		const bool  label_ok =
		    label != nullptr && (std::strcmp(label, "split") == 0 || std::strcmp(label, "completion") == 0);
		if (!label_ok) {
			Say(fmt::format("Kyty in-place reads: refused, KYTY_LABEL_MODE is '{}' (needs split or completion: "
			                "labels must reach the guest only after the GPU read the memory)\n",
			                label != nullptr ? label : "record (default)"));
			return Mode::Off;
		}
		const auto* value = std::getenv("KYTY_INPLACE");
		return std::strcmp(value, "verify") == 0 ? Mode::Verify : Mode::On;
	}();
	return mode;
}

InplaceHeap::InplaceHeap(GraphicContext& graphics, CommandScheduler& scheduler, Mode mode)
    : m_graphics(graphics), m_scheduler(scheduler), m_mode(mode),
      m_promote_frames(static_cast<uint32_t>(std::clamp<uint64_t>(EnvU64("KYTY_INPLACE_FRAMES", 2), 1, 8))),
      m_window_bytes(std::clamp<uint64_t>(EnvU64("KYTY_INPLACE_WINDOW_MB", 64), 1, 1024) << 20u),
      m_max_bytes(EnvU64("KYTY_INPLACE_MAX_MB", 512) << 20u), m_candidates(m_promote_frames) {
	// The window size is a power of two so a window index is a shift.
	while ((m_window_bytes & (m_window_bytes - 1)) != 0) {
		m_window_bytes &= m_window_bytes - 1;
	}
	m_backing_base = LibKernel::Memory::GuestBackingBase();
	m_backing_size = LibKernel::Memory::GuestBackingSize();
	m_report_start = NowMs();
	Say(fmt::format("Kyty in-place reads: {} (KYTY_INPLACE), windows {} MiB, cap {} MiB pinned, promote after {} "
	                "faulting frames of 8, backing alias 0x{:x} + {} MiB\n",
	                mode == Mode::Verify ? "verify" : "on", m_window_bytes >> 20u, m_max_bytes >> 20u,
	                m_promote_frames, m_backing_base, m_backing_size >> 20u));
}

InplaceHeap::~InplaceHeap() = default;

void InplaceHeap::NoteFault(uint64_t vaddr) noexcept {
	if (m_candidates.NoteFault(vaddr)) {
		m_promotions.fetch_add(1, std::memory_order_relaxed);
	}
}

void InplaceHeap::Demote(uint64_t vaddr, uint64_t size) noexcept {
	if (m_candidates.Clear(vaddr, size) != 0) {
		m_demotions_gpu.fetch_add(1, std::memory_order_relaxed);
	}
}
InplaceHeap::Window* InplaceHeap::LookupWindow(uint64_t window_index) {
	std::scoped_lock lock {m_window_mutex};
	const auto it = m_windows.find(window_index);
	return it != m_windows.end() ? it->second.get() : nullptr;
}

InplaceHeap::Window* InplaceHeap::ImportWindow(uint64_t window_index) {
	std::scoped_lock lock {m_window_mutex};
	auto& slot = m_windows[window_index];
	if (slot != nullptr) {
		return slot.get();
	}
	slot                 = std::make_unique<Window>();
	auto& window         = *slot;
	window.backing_offset = window_index * m_window_bytes;
	window.size           = std::min(m_window_bytes, m_backing_size - window.backing_offset);
	const uint64_t pinned = m_pinned_bytes.load(std::memory_order_relaxed);
	if (pinned + window.size > m_max_bytes) {
		window.failed = true;
		Say(fmt::format("Kyty in-place reads: window at backing offset 0x{:x} not imported: the cap of {} MiB "
		                "pinned is reached (KYTY_INPLACE_MAX_MB)\n",
		                window.backing_offset, m_max_bytes >> 20u));
		return &window;
	}
	const auto start = std::chrono::steady_clock::now();
	auto       buffer = std::make_unique<Buffer>(m_graphics, m_scheduler,
	                                       reinterpret_cast<void*>(m_backing_base + window.backing_offset),
	                                       window.size, ImportHostTag {});
	const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	if (!buffer->Handle()) {
		window.failed = true;
		Say(fmt::format("Kyty in-place reads: importing the window at backing offset 0x{:x} ({} MiB) failed\n",
		                window.backing_offset, window.size >> 20u));
		return &window;
	}
	window.buffer = std::move(buffer);
	const auto total = m_pinned_bytes.fetch_add(window.size, std::memory_order_relaxed) + window.size;
	Say(fmt::format("Kyty in-place reads: imported backing window 0x{:x} + {} MiB in {:.1f} ms; pinned now {} MiB "
	                "of {} MiB\n",
	                window.backing_offset, window.size >> 20u, ms, total >> 20u, m_max_bytes >> 20u));
	return &window;
}

bool InplaceHeap::Bind(uint64_t vaddr, uint64_t size, Buffer*& buffer, uint64_t& offset) {
	const auto* host = static_cast<const uint8_t*>(LibKernel::Memory::GuestBackingAlias(vaddr, size));
	if (host == nullptr) {
		m_rejects_window.fetch_add(1, std::memory_order_relaxed);
		return false; // not direct memory, or spans mappings
	}
	const uint64_t backing_offset = reinterpret_cast<uint64_t>(host) - m_backing_base;
	const uint64_t window_index   = backing_offset / m_window_bytes;
	if ((backing_offset + size - 1) / m_window_bytes != window_index) {
		m_rejects_window.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	Window* window = LookupWindow(window_index);
	if (window == nullptr) {
		window = ImportWindow(window_index);
	}
	if (window->failed || window->buffer == nullptr) {
		m_rejects_cap.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const uint64_t tick = m_scheduler.CurrentTick();
	window->last_read_tick.store(tick, std::memory_order_release);
	buffer = window->buffer.get();
	offset = backing_offset - window->backing_offset;
	m_binds.fetch_add(1, std::memory_order_relaxed);
	m_bind_bytes.fetch_add(size, std::memory_order_relaxed);
	if (m_mode == Mode::Verify) {
		// One completion operation per tick checks every binding of that tick, after the tick's
		// work and before the tick's labels (priority operations run in order): the guest cannot
		// have been told the work is done, so a difference is a write that raced the GPU's read.
		if (m_verify_batch == nullptr || m_verify_tick != tick) {
			m_verify_batch = std::make_shared<VerifyBatch>();
			m_verify_tick  = tick;
			m_scheduler.DeferPriorityOperation([this, batch = m_verify_batch] { CheckVerifyBatch(*batch); });
		}
		std::scoped_lock lock {m_verify_batch->mutex};
		m_verify_batch->entries.push_back({tick, vaddr, size, XXH3_64bits(host, size)});
	}
	return true;
}

void InplaceHeap::CheckVerifyBatch(VerifyBatch& batch) {
	std::scoped_lock lock {batch.mutex};
	for (const auto& entry: batch.entries) {
		m_verify_checks.fetch_add(1, std::memory_order_relaxed);
		const auto* host = LibKernel::Memory::GuestBackingAlias(entry.vaddr, entry.size);
		if (host == nullptr || XXH3_64bits(host, entry.size) != entry.hash) {
			const auto n = m_verify_mismatches.fetch_add(1, std::memory_order_relaxed);
			if (n < 32) {
				Say(fmt::format("Kyty in-place verify: [0x{:x}, +0x{:x}) changed between its binding (tick {}) and "
				                "that tick's completion\n",
				                entry.vaddr, entry.size, entry.tick));
			}
		}
	}
	batch.entries.clear();
}

void InplaceHeap::BeforeCpWrite(uint64_t vaddr, uint64_t size) {
	if (!m_candidates.Any(vaddr, size)) {
		return;
	}
	m_cp_writes.fetch_add(1, std::memory_order_relaxed);
	(void)m_candidates.Clear(vaddr, size);
	const auto* host = static_cast<const uint8_t*>(LibKernel::Memory::GuestBackingAlias(vaddr, size));
	if (host == nullptr) {
		return;
	}
	const uint64_t backing_offset = reinterpret_cast<uint64_t>(host) - m_backing_base;
	const Window*  window         = LookupWindow(backing_offset / m_window_bytes);
	if (window == nullptr || window->buffer == nullptr) {
		return;
	}
	const uint64_t tick = window->last_read_tick.load(std::memory_order_acquire);
	if (tick == 0 || m_scheduler.IsFree(tick)) {
		return;
	}
	if (!GuestGpu::IsGpuThread()) {
		// Not the recording thread: it cannot submit the recording, so the write is not ordered.
		m_cp_unordered.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	m_cp_waits.fetch_add(1, std::memory_order_relaxed);
	if (tick >= m_scheduler.CurrentTick()) {
		m_scheduler.FlushAndWait();
	} else {
		m_scheduler.Wait(tick);
	}
}

void InplaceHeap::AdvanceFrame() noexcept {
	m_candidates.AdvanceFrame();
	m_report_frames++;
	Report(0);
}

void InplaceHeap::Report(uint32_t /*frame*/) {
	const int64_t now = NowMs();
	if (now - m_report_start < 10000) {
		return;
	}
	const double f = std::max<uint32_t>(m_report_frames, 1);
	auto take = [](std::atomic<uint64_t>& v) { return static_cast<double>(v.exchange(0, std::memory_order_relaxed)); };
	const double binds = take(m_binds), bytes = take(m_bind_bytes), rw = take(m_rejects_window),
	             rc = take(m_rejects_cap), pr = take(m_promotions), dg = take(m_demotions_gpu), cw = take(m_cp_writes),
	             cwait = take(m_cp_waits), cun = take(m_cp_unordered), vc = take(m_verify_checks);
	std::printf("Inplace %.0fs: %.0f frames | binds %.0f/f %.2f MB/f | rejects window %.1f/f cap %.1f/f | candidates "
	            "%" PRIu64 " pages, promoted %.1f/f, demoted by GPU writers %.2f/f | cp writes on candidates %.2f/f, "
	            "waits %.2f/f, unordered %.2f/f | pinned %" PRIu64 " MiB | verify checks %.0f mismatches %" PRIu64
	            " (total)\n",
	            (now - m_report_start) / 1000.0, f, binds / f, bytes / f / 1e6, rw / f, rc / f,
	            m_candidates.Count(), pr / f, dg / f, cw / f, cwait / f, cun / f,
	            PinnedBytes() >> 20u, vc, m_verify_mismatches.load(std::memory_order_relaxed));
	std::fflush(stdout);
	m_report_frames = 0;
	m_report_start  = now;
}

} // namespace Libs::Graphics
