#include "common/u59Preset.h"

#include "common/stringUtils.h"

#include <algorithm>
#include <cstring>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <system_error>

namespace Common::U59Preset {

bool Parse(std::string_view text, std::vector<Entry>& entries, std::string& error) {
	entries.clear();
	// Hand-edited presets (issue #85: two keys added, then the whole file was not applied and the
	// game ran on the code defaults): a trailing comma and // comments are accepted, and any other
	// syntax error names its line and column.
	nlohmann::json document;
	try {
		document = nlohmann::json::parse(text.begin(), text.end(), nullptr, true,
		                                 /*ignore_comments=*/true, /*ignore_trailing_commas=*/true);
	} catch (const nlohmann::json::parse_error& e) {
		std::string_view what = e.what();
		if (const auto at = what.find("] "); what.starts_with("[json.exception") && at != what.npos) {
			what.remove_prefix(at + 2);
		}
		error = fmt::format("not valid JSON ({}); check for a missing comma or quote", what);
		return false;
	}
	if (!document.is_object()) {
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

static LoadResult LoadNamed(const std::vector<std::filesystem::path>& dirs, std::string_view name) {
	LoadResult result;
	for (const auto& dir: dirs) {
		result.searched.push_back(dir);
		const auto      path = dir / std::filesystem::path(std::string(name));
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

LoadResult Load(const std::vector<std::filesystem::path>& dirs) {
	return LoadNamed(dirs, FileName);
}

std::string TitleFileName(std::string_view title_id) {
	return fmt::format("u59-preset.{}.json", title_id);
}

LoadResult LoadTitle(const std::vector<std::filesystem::path>& dirs, std::string_view title_id) {
	const bool valid = title_id.size() == 9 &&
	                   std::all_of(title_id.begin(), title_id.end(), [](char ch) {
		                   return (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
	                   });
	if (!valid) {
		return {};
	}
	return LoadNamed(dirs, TitleFileName(title_id));
}

std::string TitleIdFromParamJson(std::string_view text) {
	const auto document = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
	if (document.is_discarded() || !document.is_object()) {
		return {};
	}
	if (const auto it = document.find("titleId"); it != document.end() && it->is_string()) {
		if (auto id = it->get<std::string>(); !id.empty()) {
			return id;
		}
	}
	if (const auto it = document.find("contentId"); it != document.end() && it->is_string()) {
		const auto& content_id = it->get_ref<const std::string&>();
		if (content_id.size() == 36 && content_id[6] == '-' && content_id[16] == '_') {
			return content_id.substr(7, 9);
		}
	}
	return {};
}

bool EmulatorTitleFallbackWanted(const char* title_preset_env) {
	return title_preset_env == nullptr || title_preset_env[0] == '\0';
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
	    // Missing from the int19.4 release preset (draw reuse off): an old file is named.
	    "KYTY_DRAW_RUN_RECALL",
	    "KYTY_DRAW_RUN_TARGETS",
	    "KYTY_DRAW_RUN_PIPESWAP",
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
