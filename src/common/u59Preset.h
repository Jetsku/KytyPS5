#ifndef EMULATOR_SRC_COMMON_U59PRESET_H_
#define EMULATOR_SRC_COMMON_U59PRESET_H_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

// u59-preset.json: the KYTY_*/TRACY_* environment the release archive ships next to launcher.exe.
// The launcher loads it (Load) and puts its variables into the environment of every game it starts;
// a launcher that does not find it says where it looked (Describe). At startup the emulator checks
// whether the preset's settings reached it (StartupWarnings): a direct start of kyty_emulator, or a
// launcher without the file, runs with the code defaults, which are slower and use more VRAM.
namespace Common::U59Preset {

inline constexpr std::string_view FileName = "u59-preset.json";

struct Entry {
	std::string key;
	std::string value;
};

enum class Status : uint8_t {
	Loaded,     // entries hold the file's variables
	Missing,    // no directory in `searched` holds the file
	Unreadable, // the file exists but cannot be read
	Invalid,    // not a JSON object of "KYTY_*"/"TRACY_*" string values
};

struct LoadResult {
	Status                             status = Status::Missing;
	std::filesystem::path              path;     // the file found (every status but Missing)
	std::vector<std::filesystem::path> searched; // the directories looked in, in order
	std::vector<Entry>                 entries;
	std::string                        error;    // Unreadable/Invalid: why
};

// Parses preset text. False with `error` set for anything but a JSON object whose keys start with
// KYTY_ or TRACY_ and whose values are strings (the launcher then applies nothing).
[[nodiscard]] bool Parse(std::string_view text, std::vector<Entry>& entries, std::string& error);

// The first directory of `dirs` that holds FileName decides; Missing when none does.
[[nodiscard]] LoadResult Load(const std::vector<std::filesystem::path>& dirs);

// One paragraph for the launcher (window and log) about a preset that was not applied, naming the
// file or the directories searched. Empty for Loaded.
[[nodiscard]] std::string Describe(const LoadResult& result);

// Environment lookup: the variable's value, or nullptr when it is unset.
using EnvLookup = std::function<const char*(const char*)>;

// Settings of the current preset whose code default differs: when none of them is set, the preset
// did not reach the emulator; when only some are, it is an older or edited file.
[[nodiscard]] const std::vector<std::string_view>& MarkerKeys();

// The console lines the emulator prints at startup, each ending in '\n'; empty when nothing is wrong:
// - none of MarkerKeys() set: u59-preset.json was not applied;
// - some of them unset: the preset is partial (an old or edited file), with the missing names;
// - KYTY_SRT_VARIANT_READS=0, or KYTY_RT_SOFTWARE=0 without KYTY_RT_STUB: the ray-traced lighting
//   kernels are skipped, so lighting is black unless the game's lighting patches are on.
[[nodiscard]] std::vector<std::string> StartupWarnings(const EnvLookup& env);

} // namespace Common::U59Preset

#endif // EMULATOR_SRC_COMMON_U59PRESET_H_
