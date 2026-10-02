#include "common/liveSwitch.h"

#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Live {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

using Value = std::optional<std::string>; // nullopt: unset

// One version of the live file: its sequence number, the KEY=VALUE pairs in file order (a key
// listed twice keeps its last value) and the lines it could not read.
struct Staged {
	std::optional<uint64_t>                    seq;
	std::vector<std::pair<std::string, Value>> values;
	std::vector<std::string>                   errors;
};

std::mutex        g_staged_mutex;
Staged            g_staged;
std::atomic<bool> g_has_staged {false};

std::atomic<uint64_t> g_cp_flips {0};
std::atomic<bool>     g_enabled {false};
std::string           g_path;

// Command-processor busy time (inside GuestGpu::Process, the hang trace's gfx_busy + compute_busy),
// kept with KYTY_LIVE_FILE only. Used on the command processor's thread only.
uint64_t g_cp_busy_ns     = 0;
uint64_t g_cp_slice_start = 0; // 0: outside a slice

std::atomic<void (*)(const std::string&)> g_log_sink {nullptr};

void WriteLog(const std::string& line) {
	if (auto* sink = g_log_sink.load(std::memory_order_acquire); sink != nullptr) {
		sink(line);
		return;
	}
	Log::WriteToConsoleAndLog(line);
}

std::string_view Trim(std::string_view text) {
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
		text.remove_prefix(1);
	}
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
		text.remove_suffix(1);
	}
	return text;
}

bool ValidKey(std::string_view key) {
	return !key.empty() && std::all_of(key.begin(), key.end(), [](char c) {
		return (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
	});
}

// UTF-8 (with or without a byte-order mark) or UTF-16LE with a byte-order mark (what Windows
// PowerShell 5.1's ">" writes); the keys and values are ASCII.
std::string Decode(std::string_view bytes) {
	if (bytes.size() >= 2 && static_cast<unsigned char>(bytes[0]) == 0xFFu &&
	    static_cast<unsigned char>(bytes[1]) == 0xFEu) {
		std::string text;
		for (size_t i = 2; i + 1 < bytes.size(); i += 2) {
			text.push_back(bytes[i + 1] == 0 ? bytes[i] : '?');
		}
		return text;
	}
	if (bytes.size() >= 3 && static_cast<unsigned char>(bytes[0]) == 0xEFu &&
	    static_cast<unsigned char>(bytes[1]) == 0xBBu &&
	    static_cast<unsigned char>(bytes[2]) == 0xBFu) {
		bytes.remove_prefix(3);
	}
	return std::string(bytes);
}

Staged Parse(std::string_view bytes) {
	const auto text = Decode(bytes);
	Staged     staged;
	size_t     begin = 0;
	while (begin < text.size()) {
		auto end = text.find('\n', begin);
		if (end == std::string::npos) {
			end = text.size();
		}
		const auto line = Trim(std::string_view(text).substr(begin, end - begin));
		begin           = end + 1;
		if (line.empty() || line.front() == '#' || line.front() == ';') {
			continue;
		}
		const auto equals = line.find('=');
		if (equals == std::string_view::npos) {
			staged.errors.emplace_back(line);
			continue;
		}
		std::string key(Trim(line.substr(0, equals)));
		std::transform(key.begin(), key.end(), key.begin(),
		               [](char c) { return static_cast<char>(std::toupper(static_cast<unsigned char>(c))); });
		const auto value = Trim(line.substr(equals + 1));
		if (!ValidKey(key)) {
			staged.errors.emplace_back(line);
			continue;
		}
		if (key == "KYTY_LIVE_SEQ") {
			const std::string digits(value);
			char*             parse_end = nullptr;
			const auto        seq       = std::strtoull(digits.c_str(), &parse_end, 10);
			if (digits.empty() || parse_end == nullptr || *parse_end != '\0') {
				staged.errors.emplace_back(line);
			} else {
				staged.seq = seq;
			}
			continue;
		}
		Value parsed = value.empty() ? Value {} : Value {std::string(value)};
		auto  it     = std::find_if(staged.values.begin(), staged.values.end(),
		                            [&key](const auto& pair) { return pair.first == key; });
		if (it != staged.values.end()) {
			it->second = std::move(parsed);
		} else {
			staged.values.emplace_back(std::move(key), std::move(parsed));
		}
	}
	return staged;
}

void Stage(Staged staged) {
	std::lock_guard lock(g_staged_mutex);
	g_staged = std::move(staged);
	g_has_staged.store(true, std::memory_order_release);
}

std::string SeqText(const std::optional<uint64_t>& seq) {
	return seq.has_value() ? std::to_string(*seq) : std::string("-");
}

// The whole file, opened so that the writer can still replace or delete it meanwhile.
bool ReadWholeFile(const std::string& path, std::string& out) {
	out.clear();
#if defined(_WIN32)
	// getenv's text is in the ANSI code page (as std::filesystem::path reads a std::string).
	const int wide_size = MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, nullptr, 0);
	if (wide_size <= 0) {
		return false;
	}
	std::wstring wide(static_cast<size_t>(wide_size), L'\0');
	MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, wide.data(), wide_size);
	HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ,
	                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
	                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (file == INVALID_HANDLE_VALUE) {
		return false;
	}
	char  buffer[4096];
	DWORD got = 0;
	bool  ok  = true;
	for (;;) {
		if (ReadFile(file, buffer, sizeof(buffer), &got, nullptr) == 0) {
			ok = false;
			break;
		}
		if (got == 0) {
			break;
		}
		out.append(buffer, got);
		if (out.size() > (1u << 20u)) {
			ok = false; // not a live file
			break;
		}
	}
	CloseHandle(file);
	return ok;
#else
	std::FILE* file = std::fopen(path.c_str(), "rb");
	if (file == nullptr) {
		return false;
	}
	char buffer[4096];
	for (size_t got = 0; (got = std::fread(buffer, 1, sizeof(buffer), file)) > 0;) {
		out.append(buffer, got);
	}
	std::fclose(file);
	return out.size() <= (1u << 20u);
#endif
}

void PollLoop() {
	Profiler::SetThreadName("Kyty.Live");
#if defined(_WIN32)
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#endif
	bool                            have = false;
	std::filesystem::file_time_type last_time {};
	uintmax_t                       last_size = 0;
	for (;;) {
		std::error_code error;
		const auto      time = std::filesystem::last_write_time(g_path, error);
		if (!error) {
			const auto size = std::filesystem::file_size(g_path, error);
			if (!error && (!have || time != last_time || size != last_size)) {
				std::string bytes;
				if (ReadWholeFile(g_path, bytes)) {
					have      = true;
					last_time = time;
					last_size = size;
					Stage(Parse(bytes));
				}
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}
}

void SpinMicroseconds(int64_t micros) {
	const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(micros);
	while (std::chrono::steady_clock::now() < end) {
#if defined(_M_X64) || defined(__x86_64__)
		_mm_pause();
#else
		std::this_thread::yield();
#endif
	}
}

} // namespace

// Every switch, and the live values applied by name (for switches that register later). Never
// destroyed: switches and the live-control thread may use it during exit.
struct Registry {
	std::mutex                                 mutex;
	Switch*                                    head = nullptr;
	std::vector<std::pair<std::string, Value>> applied;

	static Registry& Get() {
		static auto* registry = new Registry;
		return *registry;
	}

	static void Apply(uint64_t flip);
	static std::string Names();
};

Switch::Switch(const char* name, Parser parse, OnChange on_change) noexcept
    : m_name(name), m_parse(parse), m_on_change(on_change) {
	auto&           registry = Registry::Get();
	std::lock_guard lock(registry.mutex);
	const auto      it = std::find_if(registry.applied.begin(), registry.applied.end(),
	                                  [name](const auto& pair) { return pair.first == name; });
	if (it != registry.applied.end()) {
		m_value.store(parse(it->second.has_value() ? it->second->c_str() : nullptr),
		              std::memory_order_relaxed);
	} else {
		m_value.store(parse(EnvValue(name)), std::memory_order_relaxed);
	}
	m_next        = registry.head;
	registry.head = this;
}

std::string Registry::Names() {
	auto&                    registry = Get();
	std::vector<std::string> names;
	{
		std::lock_guard lock(registry.mutex);
		for (const Switch* s = registry.head; s != nullptr; s = s->m_next) {
			names.emplace_back(s->m_name);
		}
	}
	std::sort(names.begin(), names.end());
	names.erase(std::unique(names.begin(), names.end()), names.end());
	std::string text;
	for (const auto& name: names) {
		text += text.empty() ? "" : ", ";
		text += name;
	}
	return text.empty() ? std::string("none") : text;
}

void Registry::Apply(uint64_t flip) {
	// The hang trace's time base (summary.csv t_ms), with or without the trace, and the command
	// processor's busy time so far (this runs inside a slice: its part up to now counts).
	const auto now_ns  = HangTrace::NowNs();
	const auto t_ms    = now_ns / 1'000'000u;
	const auto busy_ns = g_cp_busy_ns + (g_cp_slice_start != 0 ? now_ns - g_cp_slice_start : 0);
	Staged     staged;
	{
		std::lock_guard lock(g_staged_mutex);
		staged = std::move(g_staged);
		g_staged = {};
		g_has_staged.store(false, std::memory_order_relaxed);
	}
	std::string              changes;
	std::vector<std::string> unknown;
	{
		auto&           registry = Get();
		std::lock_guard lock(registry.mutex);
		for (const auto& [key, value]: staged.values) {
			bool found  = false;
			bool listed = false;
			for (Switch* s = registry.head; s != nullptr; s = s->m_next) {
				if (key != s->m_name) {
					continue;
				}
				found           = true;
				const auto next = s->m_parse(value.has_value() ? value->c_str() : nullptr);
				const auto previous = s->m_value.exchange(next, std::memory_order_relaxed);
				if (previous == next) {
					continue;
				}
				if (s->m_on_change != nullptr) {
					s->m_on_change(previous, next);
				}
				if (!listed) {
					listed = true;
					changes += changes.empty() ? "" : ", ";
					changes += fmt::format("{}={} ({} -> {})", key, value.value_or(""), previous, next);
				}
			}
			auto it = std::find_if(registry.applied.begin(), registry.applied.end(),
			                       [&key](const auto& pair) { return pair.first == key; });
			if (it != registry.applied.end()) {
				it->second = value;
			} else {
				registry.applied.emplace_back(key, value);
			}
			if (!found) {
				unknown.push_back(key);
			}
		}
	}
	const auto seq = SeqText(staged.seq);
	WriteLog(fmt::format("Live: seq {} applied at CP flip {} (t_ms {}, t_us {}, cp_busy_us {}): {}\n",
	                     seq, flip, t_ms, now_ns / 1000u, busy_ns / 1000u,
	                     changes.empty() ? std::string("no change") : changes));
	for (const auto& key: unknown) {
		WriteLog(fmt::format("Live: seq {}: {} is not a live switch in this build (ignored)\n", seq,
		                     key));
	}
	for (const auto& line: staged.errors) {
		WriteLog(fmt::format("Live: seq {}: cannot read the line '{}' (ignored)\n", seq, line));
	}
}

namespace {

// KYTY_LIVE_TEST_CP_SPIN_US: the synthetic CP cost per flip (liveSwitch.h).
Switch g_test_cp_spin_us("KYTY_LIVE_TEST_CP_SPIN_US", [](const char* value) -> int64_t {
	return value != nullptr ? std::clamp<int64_t>(std::strtoll(value, nullptr, 10), 0, 1'000'000) : 0;
});

} // namespace

int64_t ParseDefaultOn(const char* value) {
	return value == nullptr || std::strcmp(value, "0") != 0 ? 1 : 0;
}

int64_t ParseDefaultOff(const char* value) {
	return value != nullptr && std::strcmp(value, "0") != 0 ? 1 : 0;
}

bool Enabled() noexcept {
	return g_enabled.load(std::memory_order_relaxed);
}

void Start() {
	static std::once_flag once;
	std::call_once(once, [] {
		const auto* path = EnvValue("KYTY_LIVE_FILE");
		if (path == nullptr) {
			return;
		}
		g_path = path;
		g_enabled.store(true, std::memory_order_relaxed);
		WriteLog(fmt::format("Live: watching KYTY_LIVE_FILE={} every 250 ms; changes apply at the "
		                     "next guest flip; live switches: {}\n",
		                     g_path, Registry::Names()));
		std::thread(PollLoop).detach();
	});
}

void CpSliceBegin() noexcept {
	if (g_enabled.load(std::memory_order_relaxed)) {
		g_cp_slice_start = HangTrace::NowNs();
	}
}

void CpSliceEnd() noexcept {
	if (g_cp_slice_start != 0) {
		g_cp_busy_ns += HangTrace::NowNs() - g_cp_slice_start;
		g_cp_slice_start = 0;
	}
}

void OnCpFlip() {
	const auto flip = g_cp_flips.fetch_add(1, std::memory_order_relaxed) + 1;
	if (g_has_staged.load(std::memory_order_acquire)) {
		Registry::Apply(flip);
	}
	if (const auto spin_us = g_test_cp_spin_us.Get(); spin_us > 0) {
		SpinMicroseconds(spin_us);
	}
}

namespace Testing {

void StageText(std::string_view text) {
	Stage(Parse(text));
}

bool HasStaged() noexcept {
	return g_has_staged.load(std::memory_order_acquire);
}

uint64_t CpFlips() noexcept {
	return g_cp_flips.load(std::memory_order_relaxed);
}

void SetLogSink(void (*sink)(const std::string& line)) {
	g_log_sink.store(sink, std::memory_order_release);
}

} // namespace Testing

} // namespace Live
