#pragma once

#include "common/hangTrace.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics::BdaSyncDiagnostics {

// Opt-in process-wide counters, including mapping changes from guest threads.
// They observe decisions; they never advance epochs or change synchronization.
// Output is diagnostic: atomics/clock/CSV writes add overhead when enabled.
enum class Event : size_t {
	Registers, Unregisters, Maps, Unmaps, Calls, Passes, EpochSkips, SubmissionSkips,
	EpochChanged, SubmissionChanged, StructureChanged, FullScans, HotPasses,
	DirtyLogPasses, IncrementalSkips, FullScanNs, HotPassNs, DirtyLogNs,
	HotRangeRunsChecked, HotRangeRunsMerged, NewBufferPasses, NewBufferNs, Count
};

namespace Detail {
class Counters {
public:
	Counters() {
		const auto* path = std::getenv("KYTY_BDA_SYNC_DIAGNOSTICS_FILE");
		if (path == nullptr || path[0] == '\0') return;
		file = std::fopen(path, "w");
		if (file == nullptr) {
			std::fprintf(stderr, "BDA sync diagnostics: cannot open output file\n");
			return;
		}
		begin = HangTrace::NowNs();
		last_flush.store(begin, std::memory_order_relaxed);
		std::fprintf(file, "begin_ms,end_ms,registers,unregisters,maps,unmaps,calls,passes,"
		                   "epoch_skips,submission_skips,epoch_changed,submission_changed,"
		                   "structure_changed,full_scans,hot_passes,dirty_log_passes,"
		                   "incremental_skips,full_scan_ns,hot_pass_ns,dirty_log_ns,"
		                   "hot_range_runs_checked,hot_range_runs_merged,"
		                   "new_buffer_passes,new_buffer_ns\n");
		std::fflush(file);
	}
	~Counters() {
		if (file == nullptr) return;
		Flush(HangTrace::NowNs());
		std::fclose(file);
	}
	[[nodiscard]] bool Enabled() const { return file != nullptr; }
	void Add(Event event, uint64_t amount = 1) {
		values[static_cast<size_t>(event)].fetch_add(amount, std::memory_order_relaxed);
	}
	void MaybeFlush() {
		const auto now = HangTrace::NowNs();
		if (now - last_flush.load(std::memory_order_relaxed) < 1'000'000'000) return;
		std::lock_guard lock(mutex);
		if (now - begin < 1'000'000'000) return;
		Flush(now);
	}

private:
	void Flush(uint64_t end) {
		if (end <= begin) return;
		std::fprintf(file, "%.3f,%.3f", begin / 1.0e6, end / 1.0e6);
		// Exchanges are independent: concurrent events may straddle a reporting interval.
		for (auto& value: values) {
			std::fprintf(file, ",%llu", static_cast<unsigned long long>(
			    value.exchange(0, std::memory_order_relaxed)));
		}
		std::fprintf(file, "\n");
		std::fflush(file);
		begin = end;
		last_flush.store(end, std::memory_order_relaxed);
	}
	FILE* file = nullptr;
	uint64_t begin = 0;
	std::atomic<uint64_t> last_flush {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Event::Count)> values {};
	std::mutex mutex;
};

inline Counters& GetCounters() {
	static Counters counters;
	return counters;
}
} // namespace Detail

inline void Record(Event event, uint64_t amount = 1) {
	auto& counters = Detail::GetCounters();
	if (counters.Enabled()) counters.Add(event, amount);
}

// Successful path body only; excludes epoch checks and dirty-log snapshot acquisition.
// Timings are CPU wall durations and include uploads, batching and range maintenance.
class TimedPath {
public:
	explicit TimedPath(Event event): counters(Detail::GetCounters()), event(event) {
		if (counters.Enabled()) start = HangTrace::NowNs();
	}
	~TimedPath() {
		if (counters.Enabled()) counters.Add(event, HangTrace::NowNs() - start);
	}
	TimedPath(const TimedPath&) = delete;
	TimedPath& operator=(const TimedPath&) = delete;

private:
	Detail::Counters& counters;
	Event event;
	uint64_t start = 0;
};

class SyncScope {
public:
	SyncScope(bool epoch_changed, bool submission_changed, bool structure_changed)
	    : counters(Detail::GetCounters()) {
		if (!counters.Enabled()) return;
		counters.Add(Event::Calls);
		if (epoch_changed) counters.Add(Event::EpochChanged);
		if (submission_changed) counters.Add(Event::SubmissionChanged);
		if (structure_changed) counters.Add(Event::StructureChanged);
	}
	~SyncScope() {
		if (counters.Enabled()) counters.MaybeFlush();
	}
	SyncScope(const SyncScope&) = delete;
	SyncScope& operator=(const SyncScope&) = delete;

private:
	Detail::Counters& counters;
};

} // namespace Libs::Graphics::BdaSyncDiagnostics
