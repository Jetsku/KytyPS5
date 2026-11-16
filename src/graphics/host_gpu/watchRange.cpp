#include "graphics/host_gpu/watchRange.h"
#include "common/staticOnce.h"

#include "common/logging/log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <vector>

namespace Libs::Graphics::WatchRange {
namespace {

struct Config {
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // [begin, end)
	uint32_t                                   lines = 400;
};

const Config& Get() {
	static constexpr auto config_init = [] {
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
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(config_init())>> config_once;
	const Config& config = config_once.Get(config_init);
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

bool CpFaultWritersEnabled() noexcept {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_CP_FAULT_WRITERS");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

uint64_t CurrentOpHash() noexcept {
	return g_hash.load(std::memory_order_relaxed);
}

const char* CurrentOpKind() noexcept {
	return g_kind.load(std::memory_order_relaxed);
}

bool WritersEnabled() noexcept;

void SetCurrentOp(const char* kind, uint64_t shader_hash) noexcept {
	if (!Enabled() && !CpFaultWritersEnabled() && !WritersEnabled()) return;
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

void NoteDispatch(uint64_t shader_hash, bool async_compute, uint32_t groups_x, uint32_t groups_y,
                  uint32_t groups_z, bool indirect) {
	static const std::vector<uint64_t> shaders = [] {
		std::vector<uint64_t> list;
		const char*           text = std::getenv("KYTY_WATCH_SHADERS");
		while (text != nullptr && *text != '\0') {
			char*      end   = nullptr;
			const auto value = std::strtoull(text, &end, 16);
			if (end == text) break;
			list.push_back(value);
			text = *end == ',' ? end + 1 : end;
			if (*end != ',') break;
		}
		return list;
	}();
	if (shaders.empty()) return;
	bool listed = false;
	for (const auto hash: shaders) listed |= hash == shader_hash;
	if (!listed) return;
	static std::atomic<uint32_t> lines {0};
	const auto line = lines.fetch_add(1, std::memory_order_relaxed);
	if (line >= Get().lines) return;
	Log::WriteToConsoleAndLog(fmt::format(
	    "KYTY_WATCH_SHADERS: op #{} CS 0x{:016x} on the {} queue, {} groups {}x{}x{}\n",
	    g_ops.load(std::memory_order_relaxed), shader_hash, async_compute ? "async compute" : "graphics",
	    indirect ? "indirect" : "direct", groups_x, groups_y, groups_z));
}

namespace {

struct WriterConfig {
	std::vector<std::pair<uint64_t, uint64_t>> ranges; // [begin, end)
	uint64_t                                   end = 0; // non-zero when any range is set
	[[nodiscard]] bool Covers(uint64_t address, uint64_t size) const {
		for (const auto& [b, e]: ranges) {
			if (address < e && address + size > b) return true;
		}
		return false;
	}
};

const WriterConfig& Writers() {
	static const WriterConfig config = [] {
		WriterConfig c;
		const char*  text = std::getenv("KYTY_WATCH_WRITERS");
		while (text != nullptr && *text != '\0') {
			char*          end   = nullptr;
			const uint64_t begin = std::strtoull(text, &end, 0);
			if (end == text || *end != ':') break;
			const char*    size_text = end + 1;
			const uint64_t size      = std::strtoull(size_text, &end, 0);
			if (end == size_text) break;
			if (size != 0) {
				c.ranges.emplace_back(begin, begin + size);
				c.end = 1;
			}
			if (*end != ',') break;
			text = end + 1;
		}
		return c;
	}();
	return config;
}

struct WriterEntry {
	uint64_t    op     = 0;
	uint64_t    hash   = 0;
	const char* kind   = "";
	const char* what   = "";
	uint32_t    frame  = 0;
	uint32_t    groups[3] {};
	uint32_t    ud[16] {};
	uint64_t    first  = 0; // the first written byte in the page
	uint64_t    bytes  = 0;
	uint32_t    count  = 1; // consecutive registrations of the same op
};

struct OpDetail {
	uint32_t frame = 0;
	uint32_t groups[3] {};
	uint32_t ud[16] {};
};

std::mutex                                             g_writers_mutex;
std::unordered_map<uint64_t, std::vector<WriterEntry>> g_writers; // page -> newest last
std::atomic<uint32_t>                                  g_detail_frame {0};
OpDetail                                               g_detail; // GPU thread writes, read under the lock

} // namespace

bool WritersEnabled() noexcept {
	return Writers().end != 0;
}

void SetCurrentOpDetail(uint32_t frame, const uint32_t* user_data, uint32_t user_data_count, uint32_t groups_x,
                        uint32_t groups_y, uint32_t groups_z) noexcept {
	if (!WritersEnabled()) return;
	std::scoped_lock lock(g_writers_mutex);
	g_detail       = {};
	g_detail.frame = frame;
	g_detail.groups[0] = groups_x;
	g_detail.groups[1] = groups_y;
	g_detail.groups[2] = groups_z;
	for (uint32_t i = 0; i < 16 && i < user_data_count && user_data != nullptr; i++) g_detail.ud[i] = user_data[i];
	g_detail_frame.store(frame, std::memory_order_relaxed);
}

void NoteWriter(const char* what, uint64_t address, uint64_t size) {
	const auto& config = Writers();
	if (config.end == 0 || size == 0 || !config.Covers(address, size)) return;
	// Clip to the watched ranges (a binding can be large).
	uint64_t first = UINT64_MAX;
	uint64_t last  = 0;
	for (const auto& [b, e]: config.ranges) {
		if (address < e && address + size > b) {
			first = std::min(first, std::max(address, b));
			last  = std::max(last, std::min(address + size, e));
		}
	}
	const uint64_t op    = g_ops.load(std::memory_order_relaxed);
	const uint64_t hash  = g_hash.load(std::memory_order_relaxed);
	const char*    kind  = g_kind.load(std::memory_order_relaxed);
	std::scoped_lock lock(g_writers_mutex);
	for (uint64_t page = first & ~uint64_t {0xfff}; page < last; page += 0x1000) {
		auto&          list  = g_writers[page];
		const uint64_t start = std::max(page, first);
		const uint64_t stop  = std::min(page + 0x1000, last);
		WriterEntry e;
		e.op    = op;
		e.hash  = hash;
		e.kind  = kind;
		e.what  = what;
		e.frame = g_detail.frame;
		std::memcpy(e.groups, g_detail.groups, sizeof(e.groups));
		std::memcpy(e.ud, g_detail.ud, sizeof(e.ud));
		e.first = start;
		e.bytes = stop - start;
		list.push_back(e);
		if (list.size() > 48) list.erase(list.begin());
	}
}

std::string DescribeWriters(uint64_t address) {
	const auto& config = Writers();
	if (config.end == 0 || !config.Covers(address, 1)) return {};
	std::scoped_lock lock(g_writers_mutex);
	const auto       it = g_writers.find(address & ~uint64_t {0xfff});
	if (it == g_writers.end()) return " writers[none]";
	std::string text = " writers[";
	for (const auto& e: it->second) {
		if (address < e.first || address >= e.first + e.bytes) continue;
		text += fmt::format(" {{op#{} f{} {} 0x{:016x} {} 0x{:x}+0x{:x} x{} g{}x{}x{} ud", e.op, e.frame, e.kind, e.hash, e.what,
		                    e.first, e.bytes, e.count, e.groups[0], e.groups[1], e.groups[2]);
		for (const auto w: e.ud) text += fmt::format(" {:08x}", w);
		text += "}";
	}
	return text + " ]";
}

bool NewestWriterUd(uint64_t address, uint32_t* ud0, uint32_t* ud1) {
	const auto& config = Writers();
	if (config.end == 0 || !config.Covers(address, 1)) return false;
	std::scoped_lock lock(g_writers_mutex);
	const auto       it = g_writers.find(address & ~uint64_t {0xfff});
	if (it == g_writers.end()) return false;
	for (auto e = it->second.rbegin(); e != it->second.rend(); ++e) {
		if (address < e->first || address >= e->first + e->bytes) continue;
		*ud0 = e->ud[0];
		*ud1 = e->ud[1];
		return true;
	}
	return false;
}

} // namespace Libs::Graphics::WatchRange
