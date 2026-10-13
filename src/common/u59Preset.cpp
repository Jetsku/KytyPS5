#include "common/u59Preset.h"

#include "common/stringUtils.h"

#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <system_error>

namespace Common::U59Preset {

bool Parse(std::string_view text, std::vector<Entry>& entries, std::string& error) {
	entries.clear();
	const auto document = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
	if (document.is_discarded() || !document.is_object()) {
		error = "not a JSON object";
		return false;
	}
	for (const auto& [key, value]: document.items()) {
		if (!key.starts_with("KYTY_") && !key.starts_with("TRACY_")) {
			error = fmt::format("key \"{}\" does not start with KYTY_ or TRACY_", key);
			entries.clear();
			return false;
		}
		if (!value.is_string()) {
			error = fmt::format("the value of {} is not a string", key);
			entries.clear();
			return false;
		}
		entries.push_back({key, value.get<std::string>()});
	}
	return true;
}

LoadResult Load(const std::vector<std::filesystem::path>& dirs) {
	LoadResult result;
	for (const auto& dir: dirs) {
		result.searched.push_back(dir);
		const auto      path = dir / FileName;
		std::error_code ec;
		if (!std::filesystem::exists(path, ec)) {
			continue;
		}
		result.path = path;
		std::ifstream file(path, std::ios::binary);
		if (!file) {
			result.status = Status::Unreadable;
			result.error  = "the file cannot be opened";
			return result;
		}
		const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		if (file.bad()) {
			result.status = Status::Unreadable;
			result.error  = "reading the file failed";
			return result;
		}
		result.status = Parse(text, result.entries, result.error) ? Status::Loaded : Status::Invalid;
		return result;
	}
	result.status = Status::Missing;
	return result;
}

std::string Describe(const LoadResult& result) {
	static constexpr std::string_view consequence =
	    "Games start with the emulator's built-in defaults, which are slower and use more video memory "
	    "than the release settings. Put the u59-preset.json from the release archive next to the "
	    "launcher.";
	switch (result.status) {
		case Status::Loaded: return {};
		case Status::Missing: {
			std::string where;
			for (const auto& dir: result.searched) {
				where += where.empty() ? "" : "; ";
				where += PathToString(dir);
			}
			return fmt::format("{} was not found (looked in: {}). {}", FileName,
			                   where.empty() ? std::string("nowhere") : where, consequence);
		}
		case Status::Unreadable:
			return fmt::format("{} cannot be read ({}: {}). {}", FileName, PathToString(result.path),
			                   result.error, consequence);
		case Status::Invalid:
			return fmt::format("{} is not valid and was not applied ({}: {}). {}", FileName,
			                   PathToString(result.path), result.error, consequence);
	}
	return {};
}

const std::vector<std::string_view>& MarkerKeys() {
	static const std::vector<std::string_view> keys = {
	    "KYTY_CP_SEQ",
	    "KYTY_CP_RECORDER",
	    "KYTY_SUBMISSION_MODE",
	    "KYTY_PROGRAM_CACHE",
	    "KYTY_SRT_READ_RUNS",
	    "KYTY_TEXTURE_CLEAN_PROOFS",
	    "KYTY_FUNCTION_ARRAY_SHRINK",
	    "KYTY_SHADER_PRECOMPILE",
	    "KYTY_PIPELINE_JOURNAL",
	};
	return keys;
}

std::vector<std::string> StartupWarnings(const EnvLookup& env) {
	const auto is_set = [&](const char* name) {
		const char* value = env(name);
		return value != nullptr && value[0] != '\0';
	};
	const auto is_zero = [&](const char* name) {
		const char* value = env(name);
		return value != nullptr && std::strcmp(value, "0") == 0;
	};

	std::vector<std::string> lines;
	std::string              missing;
	size_t                   missing_count = 0;
	for (const auto key: MarkerKeys()) {
		if (!is_set(std::string(key).c_str())) {
			missing += missing.empty() ? "" : ", ";
			missing += key;
			missing_count++;
		}
	}
	if (missing_count == MarkerKeys().size()) {
		lines.push_back(fmt::format(
		    "Warning: {} not applied (none of its settings are in the environment): running with the "
		    "built-in defaults, which are slower and use more video memory. Start games from the "
		    "launcher with {} in the launcher's folder, or set the file's variables yourself.\n",
		    FileName, FileName));
	} else if (missing_count != 0) {
		lines.push_back(fmt::format(
		    "Warning: {} looks old or edited: {} not set (built-in defaults used). Use the file from "
		    "this release.\n",
		    FileName, missing));
	}
	if (is_zero("KYTY_SRT_VARIANT_READS")) {
		lines.emplace_back(
		    "Warning: KYTY_SRT_VARIANT_READS=0: the ray-traced lighting and GI kernels are skipped, so "
		    "lit geometry renders black unless the game's lighting patches are on.\n");
	}
	// KYTY_RT_STUB=1 keeps the passes running (every ray misses): lit, without ray-traced shadows.
	if (is_zero("KYTY_RT_SOFTWARE") && (!is_set("KYTY_RT_STUB") || is_zero("KYTY_RT_STUB"))) {
		lines.emplace_back(
		    "Warning: KYTY_RT_SOFTWARE=0: software ray tracing is off, so lit geometry renders black "
		    "unless the game's lighting patches are on.\n");
	}
	return lines;
}

} // namespace Common::U59Preset
