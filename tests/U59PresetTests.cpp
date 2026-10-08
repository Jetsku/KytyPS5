#include "common/stringUtils.h"
#include "common/u59Preset.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define KYTY_TEST_PID _getpid()
#else
#include <unistd.h>
#define KYTY_TEST_PID getpid()
#endif

namespace {

using namespace Common::U59Preset;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "U59PresetTests: failed: %s\n", message);
		std::abort();
	}
}

bool Contains(const std::string& text, const std::string& part) {
	return text.find(part) != std::string::npos;
}

struct FakeEnv {
	std::map<std::string, std::string> values;

	[[nodiscard]] EnvLookup Lookup() const {
		return [this](const char* name) -> const char* {
			const auto it = values.find(name);
			return it == values.end() ? nullptr : it->second.c_str();
		};
	}
};

FakeEnv EnvFrom(const std::vector<Entry>& entries) {
	FakeEnv env;
	for (const auto& entry: entries) {
		env.values[entry.key] = entry.value;
	}
	return env;
}

void WriteFile(const std::filesystem::path& path, const std::string& text) {
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	file << text;
	Check(static_cast<bool>(file), "the test can write its preset file");
}

void TestParse() {
	std::vector<Entry> entries;
	std::string        error;
	Check(Parse(R"({"KYTY_A": "1", "TRACY_B": ""})", entries, error) && entries.size() == 2,
	      "KYTY_ and TRACY_ string values parse");
	Check(Parse("{}", entries, error) && entries.empty(), "an empty object is a valid preset");
	Check(!Parse("{\"KYTY_A\": \"1\"", entries, error) && entries.empty() && !error.empty(),
	      "truncated JSON is invalid");
	Check(!Parse(R"(["KYTY_A"])", entries, error), "an array is not a preset");
	Check(!Parse(R"({"KYTY_A": "1", "PATH": "x"})", entries, error) && entries.empty() &&
	          Contains(error, "PATH"),
	      "a key outside KYTY_/TRACY_ rejects the whole file and is named");
	Check(!Parse(R"({"KYTY_A": 1})", entries, error) && Contains(error, "KYTY_A"),
	      "a non-string value rejects the file and is named");
}

void TestLoad() {
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty_u59_preset_tests_" + std::to_string(KYTY_TEST_PID));
	std::error_code ec;
	std::filesystem::remove_all(root, ec);
	const auto first  = root / "launcher";
	const auto second = root;
	std::filesystem::create_directories(first, ec);
	Check(!ec, "the test can create its directories");

	auto missing = Load({first, second});
	Check(missing.status == Status::Missing && missing.searched.size() == 2 && missing.path.empty(),
	      "no file: Missing, both directories searched");
	const auto missing_text = Describe(missing);
	Check(Contains(missing_text, "u59-preset.json was not found") &&
	          Contains(missing_text, Common::PathToString(first)) &&
	          Contains(missing_text, Common::PathToString(second)),
	      "the Missing message names every directory searched");

	WriteFile(second / FileName, R"({"KYTY_CP_SEQ": "1"})");
	auto parent = Load({first, second});
	Check(parent.status == Status::Loaded && parent.entries.size() == 1 &&
	          parent.entries[0].key == "KYTY_CP_SEQ" && parent.path == second / FileName,
	      "the file one folder up is found when the first folder has none");
	Check(Describe(parent).empty(), "a loaded preset needs no message");

	WriteFile(first / FileName, R"({"KYTY_CP_SEQ": 1})");
	auto invalid = Load({first, second});
	Check(invalid.status == Status::Invalid && invalid.entries.empty() &&
	          invalid.path == first / FileName,
	      "the first folder's file decides, even when it is invalid");
	const auto invalid_text = Describe(invalid);
	Check(Contains(invalid_text, "not valid") && Contains(invalid_text, "KYTY_CP_SEQ") &&
	          Contains(invalid_text, Common::PathToString(first / FileName)),
	      "the Invalid message names the file and the reason");

	std::filesystem::remove_all(root, ec);
}

// The preset this repository ships parses, covers every marker and sets what the defaults need.
void TestShippedPreset() {
	const auto shipped = Load({std::filesystem::path(KYTY_U59_PRESET_DIR)});
	Check(shipped.status == Status::Loaded, "tools/u59-preset.json loads");
	const auto env = EnvFrom(shipped.entries);
	for (const auto key: MarkerKeys()) {
		Check(env.values.contains(std::string(key)), "tools/u59-preset.json sets every marker key");
	}
	Check(StartupWarnings(env.Lookup()).empty(), "the shipped preset gives no startup warning");
	const auto it = env.values.find("KYTY_SRT_VARIANT_READS");
	Check(it == env.values.end() || it->second != "0",
	      "the shipped preset does not turn KYTY_SRT_VARIANT_READS off");
}

void TestStartupWarnings() {
	FakeEnv empty;
	const auto none = StartupWarnings(empty.Lookup());
	Check(none.size() == 1 && Contains(none[0], "u59-preset.json not applied") &&
	          none[0].ends_with("\n"),
	      "no preset variables: one 'not applied' line");

	FakeEnv full;
	for (const auto key: MarkerKeys()) {
		full.values[std::string(key)] = "1";
	}
	Check(StartupWarnings(full.Lookup()).empty(), "every marker set: no warning");

	FakeEnv partial = full;
	partial.values.erase("KYTY_PIPELINE_JOURNAL");
	partial.values["KYTY_SHADER_PRECOMPILE"] = "";
	const auto old = StartupWarnings(partial.Lookup());
	Check(old.size() == 1 && Contains(old[0], "looks old or edited") &&
	          Contains(old[0], "KYTY_PIPELINE_JOURNAL") && Contains(old[0], "KYTY_SHADER_PRECOMPILE") &&
	          !Contains(old[0], "KYTY_CP_SEQ"),
	      "some markers unset (or empty): the missing ones are named");

	FakeEnv variant_off = full;
	variant_off.values["KYTY_SRT_VARIANT_READS"] = "0";
	const auto variant = StartupWarnings(variant_off.Lookup());
	Check(variant.size() == 1 && Contains(variant[0], "KYTY_SRT_VARIANT_READS=0") &&
	          Contains(variant[0], "black"),
	      "KYTY_SRT_VARIANT_READS=0 warns about black lighting");
	variant_off.values["KYTY_SRT_VARIANT_READS"] = "1";
	Check(StartupWarnings(variant_off.Lookup()).empty(), "KYTY_SRT_VARIANT_READS=1 is fine");

	FakeEnv rt_off = full;
	rt_off.values["KYTY_RT_SOFTWARE"] = "0";
	const auto rt = StartupWarnings(rt_off.Lookup());
	Check(rt.size() == 1 && Contains(rt[0], "KYTY_RT_SOFTWARE=0"), "KYTY_RT_SOFTWARE=0 warns");
	rt_off.values["KYTY_RT_STUB"] = "1";
	Check(StartupWarnings(rt_off.Lookup()).empty(),
	      "KYTY_RT_SOFTWARE=0 with KYTY_RT_STUB=1 keeps the passes: no warning");
	rt_off.values["KYTY_RT_SOFTWARE"] = "auto";
	rt_off.values.erase("KYTY_RT_STUB");
	Check(StartupWarnings(rt_off.Lookup()).empty(), "KYTY_RT_SOFTWARE=auto is fine");
}

void TestCodegenDefaults() {
	const Libs::Graphics::ShaderRecompiler::CodegenOptions defaults;
	Check(defaults.srt_variant_reads,
	      "KYTY_SRT_VARIANT_READS defaults on (lighting without u59-preset.json)");
	Check(defaults.rt_software && defaults.rt_software_auto, "software RT defaults on (auto)");
}

} // namespace

int main() {
	TestParse();
	TestLoad();
	TestShippedPreset();
	TestStartupWarnings();
	TestCodegenDefaults();
	std::printf("U59PresetTests: all passed\n");
	return 0;
}
