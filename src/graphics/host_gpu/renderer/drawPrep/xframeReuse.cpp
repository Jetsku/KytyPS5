#include "graphics/host_gpu/renderer/drawPrep/xframeReuse.h"

#include "common/assert.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics::XFrameReuse {

namespace Detail {
// A live switch: each stage chooses between equivalent resolutions; the store stays valid across
// changes (every reused set is revalidated).
Live::Switch g_mode("KYTY_XFRAME_REUSE",
                    [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });
} // namespace Detail

namespace {
Totals g_totals;
} // namespace

uint32_t SlotCount() {
	static const uint32_t slots = [] {
		const char* value = std::getenv("KYTY_XFRAME_REUSE_SLOTS");
		return value != nullptr && value[0] != '\0'
		           ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10))
		           : 4096u;
	}();
	return slots;
}

bool StatsEnabled() {
	static const bool enabled = [] {
		const auto* stats  = std::getenv("KYTY_XFRAME_REUSE_STATS");
		const auto* commit = std::getenv("KYTY_CP_COMMIT_STATS");
		return (stats != nullptr && std::strcmp(stats, "1") == 0) ||
		       (commit != nullptr && commit[0] != '\0' && std::strcmp(commit, "0") != 0);
	}();
	return enabled || VerifyMode();
}

Totals& GetTotals() {
	return g_totals;
}

void ReportMismatch(const char* what, uint64_t detail) {
	g_totals.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
		std::printf("XFrameReuseVerify: a reused texture set differs from the full resolution: %s "
		            "(detail %" PRIu64 ")\n",
		            what, detail);
		std::fflush(stdout);
	}
	if (GetMode() == Mode::Exit) {
		EXIT("XFrameReuseVerify: a reused texture set differs from the full resolution: %s\n", what);
	}
}

void PrintSummary() {
	if (!Enabled()) {
		return;
	}
	static uint64_t                last_ns = 0;
	static std::array<uint64_t, 7> last {};
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	const std::array<uint64_t, 7> values {
	    g_totals.new_sets.load(std::memory_order_relaxed),
	    g_totals.stored.load(std::memory_order_relaxed),
	    g_totals.found.load(std::memory_order_relaxed),
	    g_totals.repeated.load(std::memory_order_relaxed),
	    g_totals.bindings.load(std::memory_order_relaxed),
	    g_totals.verify_checks.load(std::memory_order_relaxed),
	    g_totals.verify_mismatches.load(std::memory_order_relaxed)};
	std::array<uint64_t, 7> d {};
	for (size_t i = 0; i < d.size(); i++) {
		d[i] = values[i] - last[i];
	}
	static std::array<uint64_t, 8> last_failures {};
	std::string                    failures;
	static const char* const       failure_names[8] = {"none", "dcc", "entry", "image",
	                                                   "residency", "page", "partner", "dcc-pages"};
	for (size_t i = 0; i < last_failures.size(); i++) {
		const auto value = g_totals.failures[i].load(std::memory_order_relaxed);
		if (value != last_failures[i]) {
			failures += " " + std::string(failure_names[i]) + "=" + std::to_string(value - last_failures[i]);
			last_failures[i] = value;
		}
	}
	std::printf("XFrameReuse %.0fs (%s): %" PRIu64 " new stage texture sets, %" PRIu64
	            " found in the store (%.1f%%), %" PRIu64 " revalidated (%.1f%%, %" PRIu64
	            " bindings); failed:%s; %" PRIu64 " stored, store %u/%u used, %" PRIu64
	            " evictions; verify %" PRIu64 " checks, %" PRIu64 " mismatches\n",
	            static_cast<double>(now - last_ns) * 1e-9, VerifyMode() ? "verify" : "on", d[0],
	            d[2], d[0] != 0 ? 100.0 * static_cast<double>(d[2]) / static_cast<double>(d[0]) : 0.0,
	            d[3], d[0] != 0 ? 100.0 * static_cast<double>(d[3]) / static_cast<double>(d[0]) : 0.0,
	            d[4], failures.empty() ? " none" : failures.c_str(), d[1], g_totals.used.load(std::memory_order_relaxed), SlotCount(),
	            g_totals.evictions.load(std::memory_order_relaxed), d[5], d[6]);
	std::fflush(stdout);
	last    = values;
	last_ns = now;
}

} // namespace Libs::Graphics::XFrameReuse
