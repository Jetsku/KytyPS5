#ifndef KYTY_COMMON_CRUZ_STATS_H_
#define KYTY_COMMON_CRUZ_STATS_H_

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// KYTY_CRUZ_STATS=1 (default off): totals of the guest unmap drains and ClampRangeSize lookups
// (KYTY_UNMAP_DRAIN_SKIP, KYTY_CLAMP_RANGE_BITMAP, KYTY_WRITE_DATA_GPU_RELEASE A/B runs), printed
// to stdout every 10 s as one "CruzStats:" line of cumulative counts. Relaxed atomics, counted
// only while the switch is on.
namespace Common::CruzStats {

enum class Counter : uint32_t {
	UnmapCalls,      // RenderContext::UnmapMemory calls over GPU-mapped ranges
	UnmapDrains,     // of them, full GPU drains (Finish + priority operations)
	UnmapDrainNs,    // time in those drains
	UnmapDrainSkips, // KYTY_UNMAP_DRAIN_SKIP
	ClampCalls,      // VirtualRanges::ClampRangeSize lookups
	ClampBitmapHits, // KYTY_CLAMP_RANGE_BITMAP answers
	ClampMemoHits,   // KYTY_CLAMP_RANGE_MEMO answers
	ClampLocked,     // locked lookups
	WriteDataReleased,
	Count
};

namespace Detail {
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<int64_t>  g_next_report_ns {0};
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CRUZ_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

inline int64_t NowNs() noexcept {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

inline void MaybeReport() noexcept {
	const auto now  = NowNs();
	auto       next = Detail::g_next_report_ns.load(std::memory_order_relaxed);
	if (next == 0) {
		Detail::g_next_report_ns.compare_exchange_strong(next, now + 10'000'000'000LL, std::memory_order_relaxed);
		return;
	}
	if (now < next ||
	    !Detail::g_next_report_ns.compare_exchange_strong(next, now + 10'000'000'000LL, std::memory_order_relaxed)) {
		return;
	}
	const auto get = [](Counter c) { return Detail::g_counters[static_cast<int>(c)].load(std::memory_order_relaxed); };
	std::printf("CruzStats: unmap_calls=%" PRIu64 " unmap_drains=%" PRIu64 " unmap_drain_ms=%.1f unmap_drain_skips=%" PRIu64
	            " clamp_calls=%" PRIu64 " clamp_bitmap=%" PRIu64 " clamp_memo=%" PRIu64 " clamp_locked=%" PRIu64
	            " write_data_released=%" PRIu64 "\n",
	            get(Counter::UnmapCalls), get(Counter::UnmapDrains),
	            static_cast<double>(get(Counter::UnmapDrainNs)) / 1e6, get(Counter::UnmapDrainSkips),
	            get(Counter::ClampCalls), get(Counter::ClampBitmapHits), get(Counter::ClampMemoHits),
	            get(Counter::ClampLocked), get(Counter::WriteDataReleased));
	std::fflush(stdout);
}

inline void Add(Counter counter, uint64_t amount = 1) noexcept {
	if (!Enabled()) {
		return;
	}
	const auto value = Detail::g_counters[static_cast<int>(counter)].fetch_add(amount, std::memory_order_relaxed);
	// The clock is read once per 1024 lookups on the hot path, at every unmap.
	if (counter != Counter::ClampCalls || (value & 1023u) == 0) {
		MaybeReport();
	}
}

} // namespace Common::CruzStats

#endif // KYTY_COMMON_CRUZ_STATS_H_
