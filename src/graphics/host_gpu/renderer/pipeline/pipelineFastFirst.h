#pragma once

#include "graphics/host_gpu/renderer/pipeline/pipelineCompileQueue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// KYTY_PIPELINE_FAST_FIRST (default off). A graphics or compute pipeline that a draw needs and
// that is not built yet is first created with VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT (much
// faster to compile on NVIDIA) and used at once. The optimized pipeline, created from the same
// create info into the driver pipeline cache, is compiled on a background worker and replaces the
// fast one for later draws. This header holds the Vulkan-free parts (scheduling, accounting,
// retirement) so they can be tested without a device.
//
// Switches: KYTY_PIPELINE_FAST_FIRST=1 enables it (=nvidia: only on NVIDIA GPUs); KYTY_PIPELINE_FAST_FIRST_THREADS (default 2)
// background compile threads; KYTY_PIPELINE_FAST_FIRST_MAX_PENDING (default 1024) cap on queued
// plus running optimized compiles (beyond it a new pipeline is built optimized, synchronously, as
// without the switch); KYTY_PIPELINE_FAST_FIRST_RETIRE_S (default 60, 0 = until exit) how long a
// replaced pipeline handle stays alive for command buffers that may still use it;
// KYTY_PIPELINE_FAST_FIRST_PROBE (default 0: every eligible new pipeline is built fast and
// optimized later; 1 first probes the driver cache, which NVIDIA's own on-disk cache can answer for
// pipelines this process never compiled, and a cold Sky Garden run then waited 1.38 s on pipeline
// creation instead of 0.31 s);
// KYTY_PIPELINE_FAST_FIRST_DRAIN_S (default 5) how long exit waits for queued optimized compiles
// so the saved driver cache holds them.

[[nodiscard]] inline uint64_t FastFirstEnvU64(const char* name, uint64_t default_value) {
	const char* text = std::getenv(name);
	if (text == nullptr || *text == '\0') return default_value;
	char*      end   = nullptr;
	const auto value = std::strtoull(text, &end, 10);
	return end != text ? value : default_value;
}

// KYTY_PIPELINE_FAST_FIRST=nvidia: on only for NVIDIA devices (vendor 0x10de), whose unoptimized
// pipelines run close to optimized speed. Other drivers may run DISABLE_OPTIMIZATION code far
// slower (a candidate for the device loss on an AMD Radeon RX 6800S at Astro Bot's title scene,
// whose first frames all use fast builds), and a dispatch past the OS GPU timeout loses the device.
[[nodiscard]] inline bool FastFirstForVendor(const char* value, uint32_t vendor_id) {
	if (value == nullptr || *value == '\0') return false;
	if (std::strcmp(value, "nvidia") == 0) return vendor_id == 0x10deu;
	char*      end    = nullptr;
	const auto number = std::strtoull(value, &end, 10);
	return end != value && number != 0;
}

[[nodiscard]] inline bool PipelineFastFirstRequested(uint32_t vendor_id) {
	return FastFirstForVendor(std::getenv("KYTY_PIPELINE_FAST_FIRST"), vendor_id);
}
[[nodiscard]] inline uint64_t FastFirstNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

struct FastFirstCounters {
	std::atomic<uint64_t> graphics_fast {0};   // pipelines used unoptimized first
	std::atomic<uint64_t> compute_fast {0};
	std::atomic<uint64_t> fast_ns {0};         // time the draws waited for the fast builds
	std::atomic<uint64_t> cache_hits {0};      // already in the driver cache: built optimized, no compile
	std::atomic<uint64_t> ineligible {0};      // built optimized as usual (tessellation, mesh, ...)
	std::atomic<uint64_t> cap_fallbacks {0};   // built optimized as usual: background queue full
	std::atomic<uint64_t> fast_failed {0};     // the fast build failed; built optimized as usual
	std::atomic<uint64_t> swaps {0};           // optimized pipelines swapped in
	std::atomic<uint64_t> swaps_dropped {0};   // optimized pipeline discarded (entry changed)
	std::atomic<uint64_t> optimize_failed {0}; // the unoptimized pipeline stays
	std::atomic<uint64_t> optimize_skipped {0}; // dropped at exit after the drain budget
	std::atomic<uint64_t> optimize_ns {0};     // worker time spent on optimized compiles
	std::atomic<uint64_t> saved_ns {0};        // sum of (optimized compile - fast build) per swap
	std::atomic<uint64_t> handles_destroyed {0};
	std::atomic<uint64_t> max_pending {0};
	std::atomic<uint64_t> seen {0};            // every new pipeline that reached the fast-first path
	std::atomic<uint64_t> probe_ns {0};        // time spent in driver-cache probes
};

struct FastFirstSnapshot {
	uint64_t graphics_fast = 0, compute_fast = 0, fast_ns = 0, cache_hits = 0, ineligible = 0;
	uint64_t cap_fallbacks = 0, fast_failed = 0, swaps = 0, swaps_dropped = 0, optimize_failed = 0;
	uint64_t optimize_skipped = 0, optimize_ns = 0, saved_ns = 0, handles_destroyed = 0;
	uint64_t pending = 0, max_pending = 0, seen = 0, probe_ns = 0;
};

[[nodiscard]] inline FastFirstSnapshot SnapshotFastFirst(const FastFirstCounters& c, uint64_t pending) {
	const auto get = [](const std::atomic<uint64_t>& v) { return v.load(std::memory_order_relaxed); };
	return {get(c.graphics_fast), get(c.compute_fast), get(c.fast_ns), get(c.cache_hits),
	        get(c.ineligible), get(c.cap_fallbacks), get(c.fast_failed), get(c.swaps),
	        get(c.swaps_dropped), get(c.optimize_failed), get(c.optimize_skipped),
	        get(c.optimize_ns), get(c.saved_ns), get(c.handles_destroyed), pending,
	        get(c.max_pending), get(c.seen), get(c.probe_ns)};
}

[[nodiscard]] inline std::string FormatFastFirst(const FastFirstSnapshot& s) {
	char text[640];
	std::snprintf(text, sizeof(text),
	              "Pipeline fast-first: %llu fast builds (%llu graphics, %llu compute; draws waited "
	              "%.1f ms), %llu optimized swaps (worker %.1f ms, estimated stall saved %.1f ms), "
	              "%llu pending (peak %llu); %llu new pipelines seen; built optimized as usual: "
	              "%llu driver-cache hits (probe %.1f ms), "
	              "%llu ineligible, %llu queue full, %llu fast build failed; %llu optimized compiles "
	              "failed, %llu dropped, %llu replaced handles destroyed",
	              static_cast<unsigned long long>(s.graphics_fast + s.compute_fast),
	              static_cast<unsigned long long>(s.graphics_fast),
	              static_cast<unsigned long long>(s.compute_fast), static_cast<double>(s.fast_ns) / 1e6,
	              static_cast<unsigned long long>(s.swaps), static_cast<double>(s.optimize_ns) / 1e6,
	              static_cast<double>(s.saved_ns) / 1e6, static_cast<unsigned long long>(s.pending),
	              static_cast<unsigned long long>(s.max_pending),
	              static_cast<unsigned long long>(s.seen),
	              static_cast<unsigned long long>(s.cache_hits), static_cast<double>(s.probe_ns) / 1e6,
	              static_cast<unsigned long long>(s.ineligible),
	              static_cast<unsigned long long>(s.cap_fallbacks),
	              static_cast<unsigned long long>(s.fast_failed),
	              static_cast<unsigned long long>(s.optimize_failed),
	              static_cast<unsigned long long>(s.optimize_skipped),
	              static_cast<unsigned long long>(s.handles_destroyed));
	return text;
}

// Once-per-interval gate shared by every thread that may report (compare-exchange: one winner).
class FastFirstLogGate {
public:
	explicit FastFirstLogGate(uint64_t start_ns, uint64_t interval_ns = 60'000'000'000ull)
	    : m_next(start_ns + interval_ns), m_interval(interval_ns) {}
	[[nodiscard]] bool TryClaim(uint64_t now_ns) {
		auto next = m_next.load(std::memory_order_relaxed);
		while (now_ns >= next) {
			if (m_next.compare_exchange_weak(next, now_ns + m_interval, std::memory_order_relaxed))
				return true;
		}
		return false;
	}

private:
	std::atomic<uint64_t> m_next;
	uint64_t              m_interval;
};

// Replaced (unoptimized) objects. Command buffers recorded with the old object may still be in
// flight, and other threads may hold a reference, so an object is destroyed only after `age_ns`
// since its retirement (0 = never before TakeAll). Not thread-safe: the owner locks.
template <class T>
class FastFirstRetireList {
public:
	void Add(T value, uint64_t now_ns) { m_items.push_back({std::move(value), now_ns}); }
	[[nodiscard]] std::vector<T> TakeExpired(uint64_t now_ns, uint64_t age_ns) {
		std::vector<T> out;
		if (age_ns == 0) return out;
		auto keep = m_items.begin();
		for (auto it = m_items.begin(); it != m_items.end(); ++it) {
			if (now_ns >= it->second && now_ns - it->second >= age_ns) {
				out.push_back(std::move(it->first));
			} else {
				if (keep != it) *keep = std::move(*it);
				++keep;
			}
		}
		m_items.erase(keep, m_items.end());
		return out;
	}
	[[nodiscard]] std::vector<T> TakeAll() {
		std::vector<T> out;
		for (auto& item: m_items) out.push_back(std::move(item.first));
		m_items.clear();
		return out;
	}
	[[nodiscard]] size_t Size() const { return m_items.size(); }

private:
	std::vector<std::pair<T, uint64_t>> m_items;
};

// Bounded background optimization. A slot is reserved (TryReserve) before the fast build so the
// cap holds even when several threads create pipelines; it is released when the task finishes or
// is dropped. Stop() admits nothing more, lets queued tasks run until the drain deadline and
// skips the rest.
class FastFirstScheduler {
public:
	FastFirstScheduler(size_t threads, size_t max_pending)
	    : m_cap(std::max<size_t>(1, max_pending)), m_queue(threads, SIZE_MAX) {}
	FastFirstScheduler(const FastFirstScheduler&)            = delete;
	FastFirstScheduler& operator=(const FastFirstScheduler&) = delete;
	~FastFirstScheduler() { Stop(0); }

	[[nodiscard]] bool TryReserve() {
		if (m_stopping.load(std::memory_order_acquire)) return false;
		auto pending = m_pending.load(std::memory_order_relaxed);
		while (pending < m_cap) {
			if (m_pending.compare_exchange_weak(pending, pending + 1, std::memory_order_acq_rel)) {
				auto peak = m_peak.load(std::memory_order_relaxed);
				while (pending + 1 > peak &&
				       !m_peak.compare_exchange_weak(peak, pending + 1, std::memory_order_relaxed)) {}
				return true;
			}
		}
		return false;
	}
	void Release() { m_pending.fetch_sub(1, std::memory_order_acq_rel); }

	// Runs `task` on a worker for a slot taken with TryReserve. `skipped` runs instead when the
	// drain deadline has passed. Returns false (slot released) once stopped.
	bool Submit(std::function<void()> task, std::function<void()> skipped = {}) {
		const bool accepted = m_queue.Submit([this, task = std::move(task), skipped = std::move(skipped)] {
			if (m_dropping.load(std::memory_order_acquire) &&
			    FastFirstNowNs() >= m_deadline_ns.load(std::memory_order_acquire)) {
				if (skipped) skipped();
			} else {
				task();
			}
			Release();
		});
		if (!accepted) Release();
		return accepted;
	}
	[[nodiscard]] size_t   Pending() const { return m_pending.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Peak() const { return m_peak.load(std::memory_order_relaxed); }
	[[nodiscard]] bool     Stopped() const { return m_stopping.load(std::memory_order_acquire); }

	void Stop(uint64_t drain_ms) {
		if (!m_stopping.exchange(true, std::memory_order_acq_rel)) {
			m_deadline_ns.store(FastFirstNowNs() + drain_ms * 1'000'000ull, std::memory_order_release);
			m_dropping.store(true, std::memory_order_release);
		}
		m_queue.Stop();
	}

private:
	size_t                m_cap;
	std::atomic<size_t>   m_pending {0};
	std::atomic<uint64_t> m_peak {0};
	std::atomic<bool>     m_stopping {false};
	std::atomic<bool>     m_dropping {false};
	std::atomic<uint64_t> m_deadline_ns {UINT64_MAX};
	PipelineCompileQueue  m_queue; // Last: joined before the rest is destroyed.
};

} // namespace Libs::Graphics
