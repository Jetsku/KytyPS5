#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <unordered_map>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics::VramStats {

namespace {

constexpr size_t KindCount = static_cast<size_t>(Kind::Count);

struct Counters {
	std::array<std::array<std::atomic<int64_t>, 2>, KindCount> bytes {};
	std::array<std::array<std::atomic<int64_t>, 2>, KindCount> count {};
};

Counters& GetCounters() {
	static Counters counters;
	return counters;
}

struct Registry {
	std::mutex                                   mutex;
	std::unordered_map<const void*, BufferEntry> buffers;
};

Registry& GetRegistry() {
	static Registry registry;
	return registry;
}

uint32_t IntervalSeconds() {
	static const uint32_t seconds = [] {
		const auto* value = std::getenv("KYTY_VRAM_STATS_SECONDS");
		const auto  parsed = value != nullptr ? std::strtoul(value, nullptr, 10) : 10ul;
		return static_cast<uint32_t>(parsed == 0 ? 10ul : std::min(parsed, 3600ul));
	}();
	return seconds;
}

std::FILE* Output() {
	static std::FILE* file = [] {
		std::string path;
		if (const auto* value = std::getenv("KYTY_VRAM_STATS_FILE"); value != nullptr && *value != '\0') {
			path = value;
		} else {
#if defined(_WIN32)
			const auto pid = _getpid();
#else
			const auto pid = getpid();
#endif
			path = "kyty-vram-stats-" + std::to_string(pid) + ".log";
		}
		std::FILE* opened = std::fopen(path.c_str(), "a");
		if (opened == nullptr) {
			std::printf("KYTY_VRAM_STATS: cannot open %s, reporting to stdout\n", path.c_str());
			return stdout;
		}
		std::printf("KYTY_VRAM_STATS: GPU memory report every %u s in %s\n", IntervalSeconds(),
		            path.c_str());
		std::fflush(stdout);
		return opened;
	}();
	return file;
}

std::chrono::steady_clock::time_point& Start() {
	static auto start = std::chrono::steady_clock::now();
	return start;
}

} // namespace

bool Enabled() noexcept {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_VRAM_STATS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void Note(Kind kind, bool device_local, int64_t bytes) noexcept {
	if (!Enabled() || kind >= Kind::Count || bytes == 0) {
		return;
	}
	auto&      counters = GetCounters();
	const auto index    = static_cast<size_t>(kind);
	const auto memory   = device_local ? 1 : 0;
	counters.bytes[index][memory].fetch_add(bytes, std::memory_order_relaxed);
	counters.count[index][memory].fetch_add(bytes > 0 ? 1 : -1, std::memory_order_relaxed);
}

KindTotals Totals(Kind kind) noexcept {
	KindTotals totals;
	if (kind >= Kind::Count) {
		return totals;
	}
	auto&      counters = GetCounters();
	const auto index    = static_cast<size_t>(kind);
	for (int memory = 0; memory < 2; memory++) {
		totals.bytes[memory] = counters.bytes[index][memory].load(std::memory_order_relaxed);
		totals.count[memory] = counters.count[index][memory].load(std::memory_order_relaxed);
	}
	return totals;
}

const char* KindName(Kind kind) noexcept {
	switch (kind) {
		case Kind::Image: return "images";
		case Kind::GuestBuffer: return "guest-buffers";
		case Kind::RingBuffer: return "rings";
		case Kind::OtherBuffer: return "other-buffers";
		case Kind::TilerScratch: return "tiler-scratch";
		case Kind::Count: break;
	}
	return "?";
}

void RegisterBuffer(const void* key, const BufferEntry& entry) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           registry = GetRegistry();
	std::lock_guard lock(registry.mutex);
	registry.buffers[key] = entry;
}

void UnregisterBuffer(const void* key) noexcept {
	if (!Enabled()) {
		return;
	}
	auto&           registry = GetRegistry();
	std::lock_guard lock(registry.mutex);
	registry.buffers.erase(key);
}

std::vector<BufferEntry> BufferSnapshot() {
	std::vector<BufferEntry> entries;
	if (!Enabled()) {
		return entries;
	}
	auto&           registry = GetRegistry();
	std::lock_guard lock(registry.mutex);
	entries.reserve(registry.buffers.size());
	for (const auto& [key, entry]: registry.buffers) {
		(void)key;
		entries.push_back(entry);
	}
	return entries;
}

bool ReportDue() noexcept {
	if (!Enabled()) {
		return false;
	}
	static std::chrono::steady_clock::time_point next {};
	const auto now = std::chrono::steady_clock::now();
	if (next == std::chrono::steady_clock::time_point {}) {
		Start() = now;
		next    = now + std::chrono::seconds(IntervalSeconds());
		return false;
	}
	if (now < next) {
		return false;
	}
	next = now + std::chrono::seconds(IntervalSeconds());
	return true;
}

double Seconds() noexcept {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - Start()).count();
}

void Line(const char* format, ...) noexcept {
	auto* file = Output();
	std::fputs("VRAM ", file);
	va_list args;
	va_start(args, format);
	std::vfprintf(file, format, args);
	va_end(args);
	std::fputc('\n', file);
}

void Flush() noexcept {
	std::fflush(Output());
}

} // namespace Libs::Graphics::VramStats
