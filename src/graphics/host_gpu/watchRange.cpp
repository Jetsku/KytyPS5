#include "graphics/host_gpu/watchRange.h"

#include "common/logging/log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics::WatchRange {
namespace {

struct Config {
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // [begin, end)
	uint32_t                                   lines = 400;
};

const Config& Get() {
	static const Config config = [] {
		Config c;
		const char* text = std::getenv("KYTY_WATCH_RANGE");
		while (text != nullptr && *text != '\0') {
			char*          end   = nullptr;
			const uint64_t begin = std::strtoull(text, &end, 0);
			if (end == text || *end != ':') break;
			const char*    size_text = end + 1;
			const uint64_t size      = std::strtoull(size_text, &end, 0);
			if (end == size_text) break;
			if (size != 0) c.ranges.emplace_back(begin, begin + size);
			text = *end == ',' ? end + 1 : end;
			if (*end != ',') break;
		}
		if (const char* lines = std::getenv("KYTY_WATCH_RANGE_LINES"); lines != nullptr) {
			c.lines = static_cast<uint32_t>(std::strtoul(lines, nullptr, 0));
		}
		return c;
	}();
	return config;
}

std::atomic<const char*> g_kind {"none"};
std::atomic<uint64_t>    g_hash {0};
std::atomic<uint64_t>    g_ops {0};
std::atomic<uint32_t>    g_lines {0};

} // namespace

bool Enabled() noexcept {
	return !Get().ranges.empty();
}

void SetCurrentOp(const char* kind, uint64_t shader_hash) noexcept {
	if (!Enabled()) return;
	g_kind.store(kind, std::memory_order_relaxed);
	g_hash.store(shader_hash, std::memory_order_relaxed);
	g_ops.fetch_add(1, std::memory_order_relaxed);
}

void Note(const char* what, uint64_t address, uint64_t size) {
	const auto& config = Get();
	if (config.ranges.empty()) return;
	for (const auto& [begin, end]: config.ranges) {
		if (address + size <= begin || address >= end) continue;
		const auto line = g_lines.fetch_add(1, std::memory_order_relaxed);
		if (line >= config.lines) return;
		const auto id = std::hash<std::thread::id> {}(std::this_thread::get_id());
		Log::WriteToConsoleAndLog(fmt::format(
		    "KYTY_WATCH_RANGE: {} 0x{:x}+0x{:x} (watch 0x{:x}+0x{:x}) thread {:x}; last GPU op #{} {} "
		    "0x{:016x}{}\n",
		    what, address, size, begin, end - begin, id & 0xffffu, g_ops.load(std::memory_order_relaxed),
		    g_kind.load(std::memory_order_relaxed), g_hash.load(std::memory_order_relaxed),
		    line + 1u == config.lines ? " (last line)" : ""));
		return;
	}
}

} // namespace Libs::Graphics::WatchRange
