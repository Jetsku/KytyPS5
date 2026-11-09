#include "common/stringUtils.h"
#include "common/textureSparseResidency.h"
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
	// Issue #85: keys added by hand with a comma after the last one.
	Check(Parse("{\n  \"KYTY_A\": \"1\",\n  \"KYTY_B\": \"2\",\n}\n", entries, error) &&
	          entries.size() == 2 && entries[1].key == "KYTY_B",
	      "a trailing comma after the last key is accepted");
	Check(Parse("{\n  // added by me\n  \"KYTY_A\": \"1\"\n}", entries, error) && entries.size() == 1,
	      "a // comment line is accepted");
	Check(!Parse("{\n  \"KYTY_A\": \"1\"\n  \"KYTY_B\": \"2\"\n}", entries, error) &&
	          entries.empty() && Contains(error, "line 3"),
	      "a missing comma is rejected and its line is named");
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

// u59-preset.<TITLE_ID>.json: found like the bundled preset, only for a well-formed title id.
void TestLoadTitle() {
	const auto root = std::filesystem::temp_directory_path() /
	                  ("kyty_u59_title_preset_tests_" + std::to_string(KYTY_TEST_PID));
	std::error_code ec;
	std::filesystem::remove_all(root, ec);
	const auto first  = root / "launcher";
	const auto second = root;
	std::filesystem::create_directories(first, ec);
	Check(!ec, "the test can create its directories");

	Check(TitleFileName("PPSA01417") == "u59-preset.PPSA01417.json", "the title file name");
	Check(LoadTitle({first, second}, "PPSA01417").status == Status::Missing,
	      "no title file: Missing");
	WriteFile(second / "u59-preset.PPSA01417.json", R"({"KYTY_DRAW_RUN": "0"})");
	const auto title = LoadTitle({first, second}, "PPSA01417");
	Check(title.status == Status::Loaded && title.entries.size() == 1 &&
	          title.entries[0].key == "KYTY_DRAW_RUN" && title.entries[0].value == "0" &&
	          title.path == second / "u59-preset.PPSA01417.json",
	      "the title file one folder up is found");
	Check(LoadTitle({first, second}, "PPSA21567").status == Status::Missing,
	      "another title does not take this title's file");
	// The title presets this repository ships parse.
	for (const char* id: {"PPSA01340", "PPSA01341", "PPSA01342", "PPSA01417", "PPSA01474", "PPSA03671", "PPSA08705"}) {
		const auto shipped = LoadTitle({std::filesystem::path(KYTY_U59_PRESET_DIR)}, id);
		Check(shipped.status == Status::Loaded && !shipped.entries.empty(),
		      "every shipped tools/u59-preset.<TITLE_ID>.json loads");
	}
	for (const char* bad: {"", "PPSA0141", "../PPSA0141", "ppsa01417", "PPSA01417x"}) {
		Check(LoadTitle({first, second}, bad).status == Status::Missing,
		      "a malformed title id loads nothing");
	}
	std::filesystem::remove_all(root, ec);
}

// The emulator's own load of the title preset when the launcher did not apply it.
void TestEmulatorTitleFallback() {
	Check(TitleIdFromParamJson(R"({"titleId": "PPSA01474", "contentId": "UP9000-PPSA01474_00-RATCHETPS5000000"})") ==
	          "PPSA01474",
	      "param.json titleId is the title id");
	Check(TitleIdFromParamJson(R"({"contentId": "UP9000-PPSA01417_00-0000000000000000"})") == "PPSA01417",
	      "without titleId, the contentId's nine characters are the title id");
	Check(TitleIdFromParamJson(R"({"contentId": "short"})").empty(), "a malformed contentId gives nothing");
	Check(TitleIdFromParamJson("not json").empty(), "text that is not JSON gives nothing");
	Check(TitleIdFromParamJson("[]").empty(), "a JSON array gives nothing");
	Check(EmulatorTitleFallbackWanted(nullptr) && EmulatorTitleFallbackWanted(""),
	      "no KYTY_TITLE_PRESET (an int21 launcher, a direct start): the emulator loads the file");
	Check(!EmulatorTitleFallbackWanted("C:/Kyty/u59-preset.PPSA01474.json"),
	      "a launcher that applied the file is not overridden");
	Check(!EmulatorTitleFallbackWanted("none"), "KYTY_TITLE_PRESET=none turns the emulator's load off");
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
	// Draw reuse (int19.4 shipped without these three) and sparse residency only below 16 GB.
	for (const char* key: {"KYTY_DRAW_RUN_RECALL", "KYTY_DRAW_RUN_TARGETS", "KYTY_DRAW_RUN_PIPESWAP"}) {
		const auto found = env.values.find(key);
		Check(found != env.values.end() && found->second == "1",
		      "the shipped preset turns draw-run recall, targets and pipeswap on");
	}
	Check(!env.values.contains("KYTY_DRAW_RUN_MESH"), "the shipped preset leaves KYTY_DRAW_RUN_MESH unset");
	const auto sparse = env.values.find("KYTY_TEXTURE_SPARSE_RESIDENCY");
	Check(sparse != env.values.end() && sparse->second == "auto",
	      "the shipped preset sets KYTY_TEXTURE_SPARSE_RESIDENCY=auto");
}

// KYTY_TEXTURE_SPARSE_RESIDENCY=auto: off from the 16 GB class (reported heap >= 15 GiB) up.
void TestTextureSparseResidency() {
	namespace Sparse = Common::TextureSparseResidency;
	constexpr uint64_t MiB = 1ull << 20u;
	Check(Sparse::Parse(nullptr) == Sparse::Mode::Off, "unset: off (code default)");
	Check(Sparse::Parse("0") == Sparse::Mode::Off && Sparse::Parse("") == Sparse::Mode::Off,
	      "0 and empty: off");
	Check(Sparse::Parse("1") == Sparse::Mode::On && Sparse::Parse("on") == Sparse::Mode::On, "1: on");
	Check(Sparse::Parse("auto") == Sparse::Mode::Auto, "auto parses");

	const uint64_t rtx3090 = 24348 * MiB; // 23.8 GiB
	const uint64_t card16  = 16076 * MiB; // a 16 GB card's heap (15.7 GiB)
	const uint64_t card12  = 12010 * MiB; // 11.7 GiB
	const uint64_t card8   = 8018 * MiB;
	Check(!Sparse::Requested(Sparse::Mode::Auto, rtx3090), "auto: off on a 24 GB card");
	Check(!Sparse::Requested(Sparse::Mode::Auto, card16), "auto: off on a 16 GB card");
	Check(!Sparse::Requested(Sparse::Mode::Auto, Sparse::AutoOffMinHeapBytes), "auto: off at the cut");
	Check(Sparse::Requested(Sparse::Mode::Auto, Sparse::AutoOffMinHeapBytes - 1), "auto: on below the cut");
	Check(Sparse::Requested(Sparse::Mode::Auto, card12) && Sparse::Requested(Sparse::Mode::Auto, card8),
	      "auto: on on 12 GB and 8 GB cards");
	Check(Sparse::Requested(Sparse::Mode::Auto, 0), "auto: on when no device-local heap is reported");
	Check(Sparse::Requested(Sparse::Mode::On, rtx3090) && !Sparse::Requested(Sparse::Mode::Off, card8),
	      "explicit 1 and 0 ignore the VRAM size");

	const auto off = Sparse::Describe("auto", Sparse::Mode::Auto, rtx3090, true);
	Check(off == "Texture sparse residency (KYTY_TEXTURE_SPARSE_RESIDENCY=auto): off, 23.8 GiB VRAM "
	             ">= 16 GiB class\n",
	      "auto off line");
	const auto on = Sparse::Describe("auto", Sparse::Mode::Auto, card12, true);
	Check(Contains(on, "=auto): on, 11.7 GiB VRAM < 16 GiB class"), "auto on line");
	const auto missing = Sparse::Describe("auto", Sparse::Mode::Auto, card8, false);
	Check(Contains(missing, "unavailable"), "auto on without device support: unavailable");
	const auto forced = Sparse::Describe("1", Sparse::Mode::On, rtx3090, true);
	Check(Contains(forced, "=1): on, 23.8 GiB VRAM"), "explicit 1 line");
	Check(Contains(Sparse::Describe(nullptr, Sparse::Mode::Off, card8, false), "=unset): off"),
	      "unset line");
	Check(std::string(Sparse::ResolvedName(Sparse::Mode::Auto, false)) == "auto:off" &&
	          std::string(Sparse::ResolvedName(Sparse::Mode::Auto, true)) == "auto:on" &&
	          std::string(Sparse::ResolvedName(Sparse::Mode::On, true)) == "on",
	      "resolved names for the stability line");
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
	TestLoadTitle();
	TestEmulatorTitleFallback();
	TestShippedPreset();
	TestTextureSparseResidency();
	TestStartupWarnings();
	TestCodegenDefaults();
	std::printf("U59PresetTests: all passed\n");
	return 0;
}
