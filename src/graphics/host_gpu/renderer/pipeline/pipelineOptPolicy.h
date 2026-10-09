#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// Which pipelines are never built optimized, and the guard around optimized builds (AMD test 4).
// AMD's Windows Vulkan driver faults (a null read at amdvlk64.dll+0x22240fc, deep in its
// optimizing pipeline compiler) while building a few of Astro Bot's pipelines optimized; the
// unoptimized builds of the same pairs (VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT) succeed, and
// AMD's offline compiler builds every module fine. This header holds the Vulkan-free parts.
//
// KYTY_PIPELINE_OPTIMIZE=0 (default 1): no pipeline is ever built optimized. With
// KYTY_PIPELINE_FAST_FIRST the unoptimized first build stays for good (no background optimized
// compile); without it every build has DISABLE_OPTIMIZATION. Unoptimized code may run slower.
//
// KYTY_PIPELINE_NO_OPT_SHADERS=<hash,hash,...>: pipelines with any of these guest shader hashes
// (VS, PS or CS; hex, with or without 0x) are built unoptimized; all others as usual. Unset, AMD
// devices (vendor 0x1002) get AmdNoOptShaders, the pixel shaders seen crashing the driver; others
// get no list. "none" clears the list, and the learned entries below are not applied either.
//
// KYTY_PIPELINE_OPT_FAULT_GUARD=0|1 (default 1 on AMD, 0 elsewhere; Windows only): an optimized
// graphics pipeline build that faults inside the driver is caught; the unoptimized pipeline is
// used instead, and the pipeline's shader hashes are appended to _PipelineCache/<title>.noopt.txt
// for this GPU and driver, so later runs build that pipeline unoptimized from the start. A driver
// that faulted may be left in a bad state (a lock it held, memory it was writing); see
// PipelineCache::NoteOptimizeFault for what the emulator does to limit that.

constexpr uint32_t PipelineOptAmdVendorId = 0x1002u;

// PS 0xa6d0a69b25e2707a (AMD test 3, issue #22: its optimized builds crashed, the unoptimized
// builds of the same pairs did not), 0xd40543e536984200 (game 1.007) and 0x13495e6ee1376edc (its
// 1.018 equivalent), 0x34e789a13e4cb246 (1.019).
inline constexpr std::array<uint64_t, 4> AmdNoOptShaders = {
    0xa6d0a69b25e2707aull, 0xd40543e536984200ull, 0x13495e6ee1376edcull, 0x34e789a13e4cb246ull};

[[nodiscard]] inline std::string_view TrimPipelineOptText(std::string_view text) {
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
		text.remove_prefix(1);
	}
	while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
		text.remove_suffix(1);
	}
	return text;
}

// A guest shader hash: hex digits, optionally prefixed by 0x. Zero is not a hash.
[[nodiscard]] inline std::optional<uint64_t> ParseShaderHash(std::string_view text) {
	text = TrimPipelineOptText(text);
	if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		text.remove_prefix(2);
	}
	if (text.empty() || text.size() > 16) return std::nullopt;
	uint64_t value = 0;
	for (const char c: text) {
		uint64_t digit = 0;
		if (c >= '0' && c <= '9') {
			digit = static_cast<uint64_t>(c - '0');
		} else if (c >= 'a' && c <= 'f') {
			digit = static_cast<uint64_t>(c - 'a' + 10);
		} else if (c >= 'A' && c <= 'F') {
			digit = static_cast<uint64_t>(c - 'A' + 10);
		} else {
			return std::nullopt;
		}
		value = (value << 4u) | digit;
	}
	if (value == 0) return std::nullopt;
	return value;
}

// Splits at ',', ';' and white space. Malformed tokens go to `bad` (when given).
inline std::vector<uint64_t> ParseShaderHashList(std::string_view text,
                                                 std::vector<std::string>* bad = nullptr) {
	std::vector<uint64_t> out;
	size_t                begin = 0;
	const auto            flush = [&](size_t end) {
        const auto token = TrimPipelineOptText(text.substr(begin, end - begin));
        if (!token.empty()) {
            if (const auto hash = ParseShaderHash(token)) {
                if (std::find(out.begin(), out.end(), *hash) == out.end()) out.push_back(*hash);
            } else if (bad != nullptr) {
                bad->emplace_back(token);
            }
        }
	};
	for (size_t i = 0; i < text.size(); i++) {
		const char c = text[i];
		if (c == ',' || c == ';' || std::isspace(static_cast<unsigned char>(c)) != 0) {
			flush(i);
			begin = i + 1;
		}
	}
	flush(text.size());
	return out;
}

struct PipelineOptSettings {
	enum class ListSource { None, BuiltIn, Env, Cleared };
	bool                     optimize    = true;
	bool                     fault_guard = false;
	// Whether the learned entries of the .noopt.txt file apply ("none" turns them off too).
	bool                     use_learned = true;
	ListSource               source      = ListSource::None;
	std::vector<uint64_t>    shaders;
	std::vector<std::string> bad_tokens;
};

[[nodiscard]] inline bool PipelineOptFlag(const char* text, bool default_value) {
	if (text == nullptr || *text == '\0') return default_value;
	char*      end   = nullptr;
	const auto value = std::strtoull(text, &end, 10);
	return end != text ? value != 0 : default_value;
}

[[nodiscard]] inline PipelineOptSettings ParsePipelineOptSettings(const char* optimize,
                                                                  const char* shaders,
                                                                  const char* fault_guard,
                                                                  uint32_t    vendor_id) {
	PipelineOptSettings settings;
	const bool          amd = vendor_id == PipelineOptAmdVendorId;
	settings.optimize       = PipelineOptFlag(optimize, true);
#ifdef _WIN32
	settings.fault_guard = PipelineOptFlag(fault_guard, amd);
#else
	(void)fault_guard;
	settings.fault_guard = false;
#endif
	const auto list = shaders != nullptr ? TrimPipelineOptText(shaders) : std::string_view {};
	if (shaders == nullptr || list.empty()) {
		if (amd) {
			settings.source = PipelineOptSettings::ListSource::BuiltIn;
			settings.shaders.assign(AmdNoOptShaders.begin(), AmdNoOptShaders.end());
		}
	} else if (list == "none" || list == "NONE" || list == "0") {
		settings.source      = PipelineOptSettings::ListSource::Cleared;
		settings.use_learned = false;
	} else {
		settings.source  = PipelineOptSettings::ListSource::Env;
		settings.shaders = ParseShaderHashList(list, &settings.bad_tokens);
	}
	return settings;
}

[[nodiscard]] inline PipelineOptSettings PipelineOptSettingsFromEnv(uint32_t vendor_id) {
	return ParsePipelineOptSettings(std::getenv("KYTY_PIPELINE_OPTIMIZE"),
	                                std::getenv("KYTY_PIPELINE_NO_OPT_SHADERS"),
	                                std::getenv("KYTY_PIPELINE_OPT_FAULT_GUARD"), vendor_id);
}

// The shader hashes of one pipeline: up to three vertex-side stages and the pixel shader (graphics),
// or the compute shader in [0]. Zero: no shader there.
using PipelineShaderHashes = std::array<uint64_t, 4>;

// Entries of shader hashes; a pipeline matches an entry when it contains every hash of it. Listed
// shaders are single-hash entries; a learned fault is the pipeline's whole set. Thread-safe.
class PipelineNoOptList {
public:
	// Returns false when the entry was there already (or is empty).
	bool Add(std::span<const uint64_t> hashes) {
		std::vector<uint64_t> entry;
		for (const auto h: hashes) {
			if (h != 0 && std::find(entry.begin(), entry.end(), h) == entry.end()) entry.push_back(h);
		}
		if (entry.empty()) return false;
		std::sort(entry.begin(), entry.end());
		std::lock_guard lock(m_mutex);
		if (std::find(m_entries.begin(), m_entries.end(), entry) != m_entries.end()) return false;
		m_entries.push_back(std::move(entry));
		return true;
	}
	// The first entry the pipeline matches, or nullopt.
	[[nodiscard]] std::optional<std::vector<uint64_t>> Match(
	    std::span<const uint64_t> pipeline) const {
		std::lock_guard lock(m_mutex);
		for (const auto& entry: m_entries) {
			const bool all = std::all_of(entry.begin(), entry.end(), [&](uint64_t h) {
				return std::find(pipeline.begin(), pipeline.end(), h) != pipeline.end();
			});
			if (all) return entry;
		}
		return std::nullopt;
	}
	[[nodiscard]] size_t Size() const {
		std::lock_guard lock(m_mutex);
		return m_entries.size();
	}

private:
	mutable std::mutex                 m_mutex;
	std::vector<std::vector<uint64_t>> m_entries;
};

// The GPU and driver a learned entry was found on: vendor, device and driver version, in hex.
[[nodiscard]] inline std::string PipelineOptDeviceKey(uint32_t vendor_id, uint32_t device_id,
                                                      uint32_t driver_version) {
	char text[40] = {};
	std::snprintf(text, sizeof(text), "%04x:%04x:%08x", vendor_id, device_id, driver_version);
	return text;
}

// One line of the .noopt.txt file: "<device key> <hash>+<hash>..." (hashes as 0x%016x), no newline.
[[nodiscard]] inline std::string FormatNoOptLine(std::string_view          device_key,
                                                 std::span<const uint64_t> hashes) {
	std::string line(device_key);
	char        sep = ' ';
	for (const auto h: hashes) {
		if (h == 0) continue;
		char text[24] = {};
		std::snprintf(text, sizeof(text), "%c0x%016" PRIx64, sep, h);
		line += text;
		sep = '+';
	}
	return line;
}

// The entries of a .noopt.txt file that were found on `device_key`. '#' starts a comment; a line
// of another GPU or driver, or a malformed one, is skipped.
[[nodiscard]] inline std::vector<std::vector<uint64_t>> ParseNoOptFile(std::string_view text,
                                                                       std::string_view device_key) {
	std::vector<std::vector<uint64_t>> entries;
	size_t                             begin = 0;
	while (begin < text.size()) {
		auto end = text.find('\n', begin);
		if (end == std::string_view::npos) end = text.size();
		auto line = text.substr(begin, end - begin);
		begin     = end + 1;
		if (const auto hash_mark = line.find('#'); hash_mark != std::string_view::npos) {
			line = line.substr(0, hash_mark);
		}
		line = TrimPipelineOptText(line);
		if (line.empty()) continue;
		const auto space = line.find_first_of(" \t");
		if (space == std::string_view::npos || line.substr(0, space) != device_key) continue;
		auto                  rest = TrimPipelineOptText(line.substr(space + 1));
		std::vector<uint64_t> entry;
		bool                  ok   = true;
		size_t                from = 0;
		while (from <= rest.size()) {
			auto plus = rest.find('+', from);
			if (plus == std::string_view::npos) plus = rest.size();
			const auto hash = ParseShaderHash(rest.substr(from, plus - from));
			if (!hash) {
				ok = false;
				break;
			}
			entry.push_back(*hash);
			from = plus + 1;
		}
		if (ok && !entry.empty()) entries.push_back(std::move(entry));
	}
	return entries;
}

} // namespace Libs::Graphics
