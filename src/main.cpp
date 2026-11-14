#include "common/archive.h"
#include "common/common.h"
#include "common/file.h"
#include "common/processHeap.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/topologyRule.h"
#include "common/u59Preset.h"
#include "common/wheelConfig.h"
#include "emulator.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/x64InstructionEmulator.h"

#include <algorithm>
#include <charconv>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string_view>
#include <vector>
#include <fmt/format.h>
#include <magic_enum.hpp>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#if defined(KYTY_PGO_GENERATE)
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

// compiler-rt profile runtime (instrumented builds, CMake KYTY_PGO_GENERATE).
extern "C" void __llvm_profile_set_filename(const char*);
extern "C" int  __llvm_profile_write_file(void);
#endif

using namespace Common;
using namespace Emulator;

#if defined(KYTY_PGO_GENERATE)
// The profile runtime writes its counters only at a normal exit, but the timing harness stops the
// emulator with TerminateProcess. With KYTY_PGO_FILE=<file.profraw>, a thread writes them every
// KYTY_PGO_DUMP_SECONDS (default 20) to <file>.tmp and renames that over <file>, so the file always
// holds the last complete dump.
static void StartPgoDumps() {
	const char* file = std::getenv("KYTY_PGO_FILE");
	if (file == nullptr || *file == '\0') {
		::printf("PGO: instrumented build without KYTY_PGO_FILE: no profile is written\n");
		return;
	}
	const char* seconds_text = std::getenv("KYTY_PGO_DUMP_SECONDS");
	const int   seconds      = (seconds_text != nullptr && std::atoi(seconds_text) > 0) ? std::atoi(seconds_text) : 20;
	// "%p" in the name becomes the process id, so repeated launches (a player's own training
	// sessions) each keep their own profile instead of overwriting the previous one.
	std::string name = file;
	if (const auto at = name.find("%p"); at != std::string::npos) {
#if defined(_WIN32)
		name.replace(at, 2, std::to_string(_getpid()));
#else
		name.replace(at, 2, std::to_string(getpid()));
#endif
	}
	static const std::string target = name;
	static const std::string temp   = target + ".tmp";
	::printf("PGO: writing the profile to %s every %d s\n", target.c_str(), seconds);
	std::thread([seconds]() {
		for (uint32_t dump = 1;; dump++) {
			std::this_thread::sleep_for(std::chrono::seconds(seconds));
			__llvm_profile_set_filename(temp.c_str());
			const int       written = __llvm_profile_write_file();
			std::error_code error;
			if (written == 0) {
				std::filesystem::rename(temp, target, error);
			}
			::printf("PGO: dump %u %s\n", dump, written != 0 ? "failed to write" : (error ? "failed to rename" : "written"));
			::fflush(stdout);
		}
	}).detach();
}
#endif

static void PrintUsage() {
	::printf("%s, %s, ver = %s, git = %s, date = %s\n",
	         KYTY_BUILD == KYTY_BUILD_DEBUG ? "Debug" : "Release", KYTY_TOOLCHAIN, KYTY_VERSION,
	         KYTY_GIT_VERSION, __DATE__);
	::printf("kyty_emulator --game <dir|elf|zar> [options]\n\n");
	::printf("Options:\n");
	::printf("  --game <dir|elf|zar>                 Game directory, ELF, or ZArchive to load.\n");
	::printf("  --game-patch <json>                  ETAHen cheat file.\n");
	::printf("  --addcont-dir <dir>                  Add-ons (DLC) folder (default: none).\n");
	::printf(
	    "  --addcont <label>[=<dir>]            Add-on (DLC) by entitlement label with its data in\n"
	    "                                       <dir>, or license only without <dir>. Repeatable.\n");
	::printf("  --screen-width <num>                 Window width. Default: 1280.\n");
	::printf("  --screen-height <num>                Window height. Default: 720.\n");
	::printf(
	    "  --user-name <name>                   Local user name (1-16 bytes). Default: Kyty.\n");
	::printf("  --user-id <num>                      Local user ID. Default: %d.\n",
	         Config::DEFAULT_USER_ID);
	::printf("  --mic <name>                        Capture from this microphone; omit for silence.\n");
	::printf("  --controller-color <#RRGGBB>        Override the controller lightbar color.\n");
	::printf("  --controller-volume <0-100>         DualSense speaker volume. Default: 50.\n");
	::printf("  --controller-vibration <0-100>      DualSense vibration intensity. Default: 100.\n");
	::printf("  --keyboard-player <0-4>             Keyboard player; 0 shares player 1 with the gamepad. Default: 0.\n");
	::printf("  --audio-master-volume <0-200>       Volume of everything on the main output. Default: 100.\n");
	::printf("  --audio-main-volume <0-200>         Game sound (main ports). Default: 100.\n");
	::printf("  --audio-music-volume <0-200>        Music on BGM ports. Default: 100.\n");
	::printf("  --audio-pad-speaker-volume <0-200>  Controller-speaker sounds played on the main output\n"
	         "                                      when no DualSense takes them. Default: %u.\n",
	         Config::DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME);
	::printf("  --audio-objects-volume <0-200>      3D audio objects (AudioOut2 object ports), mixed into\n"
	         "                                      the game sound. Default: 100.\n");
	::printf("  --audio-objects <on|off>            Play 3D audio objects (off: dropped). Default: on.\n");
	::printf(
	    "  --present-mode <value>               Fifo, Mailbox, or Immediate. Default: Mailbox.\n");
	::printf(
	    "  --gpu <index>                        Vulkan physical device index. Default: auto.\n");
	::printf("  --fullscreen                         Run in borderless desktop fullscreen.\n");
	::printf("  --hdr                                Enable HDR10 output when supported.\n");
	::printf(
	    "  --hide-cursor                        Hide the cursor after 2 s idle. Default: off.\n");
	::printf(
	    "  --stretch                            Stretch the image to the window. Default: off.\n");
	::printf("  --no-splash                          Disable the startup splash screen.\n");
	::printf("  --vr                                 Enable the virtual VR headset.\n");
	::printf("  --trinity                            Enable experimental PS5 Pro mode for supported titles\n"
	         "                                       (PSSR uses linear scaling unless --dlss is set). Default: off.\n");
	::printf("  --dlss <folder>                      With --trinity, upscale PSSR with the DLSS\n"
	         "                                       model in <folder> instead of linearly.\n");
	::printf("  --dlss-preset <A-O>                  DLSS preset. Default: the model's own.\n");
	::printf("  --wheel <true|false>                 Show the game a racing wheel fed by host\n"
	         "                                       devices (see --wheel-config).\n");
	::printf("  --wheel-config <json>                Wheel mapping file. Default: wheel.json in\n"
	         "                                       the working directory.\n");
	::printf("  --amd-cpu                            Intel CPU compatibility: emulate the PS5's AMD-only\n"
	         "                                       instructions (EXTRQ/INSERTQ, AMD VRSQRTPS results).\n"
	         "                                       Default: on unless the CPU is an AMD (KYTY_AMD_CPU=\n"
	         "                                       1|0|auto overrides).\n");
	::printf("  --no-amd-cpu                         Turn Intel CPU compatibility off on a non-AMD CPU.\n");
	::printf("  --guest-tsc                          Run the guest TSC at the PS5's 1.6 GHz.\n");
	::printf("  --vblank-frequency <num>             Virtual vblank frequency. Default: 60.\n");
	::printf("  --console-language <0-29>            Console language. Default: 1 (English US).\n");
	::printf("  --vulkan-validation <true|false>     Enable Vulkan validation.\n");
	::printf("  --network-connected <true|false>     Use the host local network (default: false).\n");
	::printf("  --emulate-psn                        Simulate PSN sign-in for compatibility.\n");
	::printf("  --gpu-assisted-validation <t|f>      Bounds-check shader accesses on the GPU.\n"
	         "                                       Implies --vulkan-validation; very slow.\n");
	::printf("  --shader-validation <true|false>     Enable shader validation.\n");
	::printf("  --tessellation                      Draw tessellation patches; skipped by default.\n");
	::printf("  --gpu-occlusion <on|off>             Accurate GPU occlusion queries (on) or always visible\n"
	         "                                       (off, faster). Overrides KYTY_GPU_OCCLUSION (the\n"
	         "                                       bundled preset: off).\n");
	::printf("  --relaxed-shader-float               Native GPU float ops in shaders; faster, can be inaccurate.\n");
	::printf("  --lod-stats                          Report bound textures in mip statistics.\n");
	::printf("  --shader-optimization-type <value>   None, Size, or Performance.\n");
	::printf("  --shader-log-direction <value>       Silent, Console, or File.\n");
	::printf("  --shader-log-folder <path>           Shader log output folder.\n");
	::printf("  --command-buffer-dump <true|false>   Enable command buffer dumps.\n");
	::printf("  --command-buffer-dump-folder <path>  Command buffer dump folder.\n");
	::printf("  --graphics-debug-dump <true|false>   Enable graphics debug dumps.\n");
	::printf("  --printf-direction <value>           Silent, Console, or File.\n");
	::printf("  --printf-output-file <path>          Guest printf output file.\n");
	::printf("  --profile                            Enable the Tracy profiler.\n");
	::printf("  --spirv-debug-printf <true|false>    Enable SPIR-V debug printf.\n");
	::printf(
	    "  --readback-linear-images <true|false> Read back writable linear images on submit.\n");
	::printf(
	    "  --readback-tiled-images <true|false>  Read back small writable tiled images on submit.\n");
	::printf(
	    "  --readback-storage-images <true|false> Read back storage images on GL2 writeback.\n");
	::printf(
	    "  --sync-raw-image-buffers <true|false> Synchronize raw reads of GPU images. Default: false.\n");
	::printf(
	    "  --flush-draws <num>                  Submit GPU work every <num> draws; 0 is off.\n");
	::printf(
	    "  --trophy-notifications <true|false>   Show trophy unlock toasts and play their sound.\n");
	::printf("  --perf-overlay                       Show the performance panel (F2 or Cmd+P).\n");
	::printf(
	    "  --trophy-duration <seconds>          Trophy toast duration, 5-10. Default: 6.\n");
	::printf("  --playgo-hack                       Use the supplied PlayGo stub fallback.\n");
	::printf(
	    "  --skip-notice-screen <true|false>    Skip startup logos and notices in supported games.\n"
	    "                                      Default: false.\n");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	::printf("  --redzone                            Protect the guest SysV red zone.\n");
#endif
	::printf("  --keymap <Control=Input>             DualSense mapping; may be repeated.\n");
	::printf("  --rd                                 Enable RenderDoc capture.\n");
}

static bool NextArg(int argc, char* argv[], int& index, std::string& out) {
	if (index + 1 >= argc) {
		return false;
	}

	index++;
	out = argv[index];
	return true;
}

static bool ParseBool(const std::string& value, bool& out) {
	if (Common::EqualNoCase(value, "true") || value == "1" || Common::EqualNoCase(value, "yes") ||
	    Common::EqualNoCase(value, "on")) {
		out = true;
		return true;
	}

	if (Common::EqualNoCase(value, "false") || value == "0" || Common::EqualNoCase(value, "no") ||
	    Common::EqualNoCase(value, "off")) {
		out = false;
		return true;
	}

	return false;
}

template <typename E>
static bool ParseEnum(const std::string& value, E& out) {
	auto enum_value = magic_enum::enum_cast<E>(value.c_str());
	if (!enum_value.has_value()) {
		return false;
	}

	out = enum_value.value();
	return true;
}

static bool ParseConsoleLanguage(const std::string& value, uint32_t& out) {
	uint32_t language = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), language);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    language > Config::MAX_CONSOLE_LANGUAGE) {
		return false;
	}
	out = language;
	return true;
}

static bool ParseUint32(const std::string& value, uint32_t& out) {
	uint32_t number   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseControllerColor(const std::string& value, Config::ControllerColor& out) {
	if (value.size() != 7 || value[0] != '#') {
		return false;
	}
	uint32_t rgb = 0;
	auto [end, error] = std::from_chars(value.data() + 1, value.data() + value.size(), rgb, 16);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = {static_cast<uint8_t>(rgb >> 16), static_cast<uint8_t>(rgb >> 8),
	       static_cast<uint8_t>(rgb)};
	return true;
}

static bool ParseInt32(const std::string& value, int32_t& out) {
	int32_t number    = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
	if (error != std::errc {} || end != value.data() + value.size()) {
		return false;
	}
	out = number;
	return true;
}

static bool ParseUserId(const std::string& value, int32_t& out) {
	int32_t user_id   = 0;
	auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), user_id);
	if (error != std::errc {} || end != value.data() + value.size() ||
	    !Config::IsConfiguredUserIdValid(user_id)) {
		return false;
	}
	out = user_id;
	return true;
}

// --amd-cpu forces the AMD instruction patch on, --no-amd-cpu off; without either it follows the
// CPU vendor: off on an AMD CPU, which runs the instructions natively. On any other CPU the game's
// EXTRQ would otherwise trap at run time, where on Windows the exception frame can overwrite the
// guest's red zone (amd_cpu_emulation_tests), and VRSQRTPS would give the host's approximation.
enum class AmdCpuChoice { Automatic, On, Off };

static void ResolveAmdCpu(AmdCpuChoice choice, const std::string& source, Config::ConfigOptions& config) {
	if (choice != AmdCpuChoice::Automatic) {
		config.amd_cpu_enabled  = choice == AmdCpuChoice::On;
		config.amd_cpu_decision = fmt::format("{} ({})", config.amd_cpu_enabled ? "on" : "off", source);
		return;
	}
	const auto vendor = Loader::X64InstructionEmulator::HostCpuVendor();
#if defined(__APPLE__)
	// The launcher does not offer the patch on macOS, and Darwin's signal frames keep the red zone.
	config.amd_cpu_enabled  = false;
	config.amd_cpu_decision = fmt::format("off (automatic on macOS, CPU vendor {})", vendor);
#else
	config.amd_cpu_enabled  = !Loader::X64InstructionEmulator::IsAmdCpuVendor(vendor);
	config.amd_cpu_decision = fmt::format("{} (automatic: CPU vendor {}{})", config.amd_cpu_enabled ? "on" : "off",
	                                      vendor, config.amd_cpu_enabled ? " is not AMD" : "");
#endif
}

static bool ParseArgs(int argc, char* argv[], RunOptions& options, bool& show_help) {
	show_help = false;
	AmdCpuChoice amd_cpu        = AmdCpuChoice::Automatic;
	std::string  amd_cpu_source = "automatic";

	for (int i = 1; i < argc; i++) {
		std::string arg = std::string(argv[i]);
		std::string value;

		if (arg == "--help" || arg == "-h") {
			show_help = true;
			continue;
		}

		if (arg == "--rd") {
			options.config.renderdoc_enabled = true;
			continue;
		}

		if (arg == "--fullscreen") {
			options.config.fullscreen_enabled = true;
			continue;
		}

		if (arg == "--hdr") {
			options.config.hdr_enabled = true;
			continue;
		}

		if (arg == "--hide-cursor") {
			options.config.hide_cursor_enabled = true;
			continue;
		}

		if (arg == "--no-splash") {
			options.config.splash_enabled = false;
			continue;
		}

		if (arg == "--vr") {
			options.config.vr_enabled = true;
			continue;
		}

		if (arg == "--perf-overlay") {
			options.config.perf_overlay_enabled = true;
			continue;
		}

		// Value-less like upstream's flags (the launcher passes them without a value).
		if (arg == "--stretch") {
			options.config.stretch_enabled = true;
			continue;
		}

		if (arg == "--trinity") {
			options.config.pro_enabled = true;
			continue;
		}

		if (arg == "--emulate-psn") {
			options.config.psn_emulation_enabled = true;
			continue;
		}

		// --shader-cache and --occlusion-queries (upstream launcher flags) are accepted and have no effect:
		// ours keeps its own program cache and --gpu-occlusion.
		if (arg == "--shader-cache") {
			options.config.shader_cache_enabled = true;
			continue;
		}

		if (arg == "--occlusion-queries") {
			options.config.occlusion_queries_enabled = true;
			continue;
		}


		if (arg == "--amd-cpu" || arg == "--no-amd-cpu") {
			amd_cpu        = arg == "--amd-cpu" ? AmdCpuChoice::On : AmdCpuChoice::Off;
			amd_cpu_source = arg;
			continue;
		}

		if (arg == "--playgo-hack") {
			options.config.playgo_hack_enabled = true;
			continue;
		}

		if (arg == "--tessellation") {
			options.config.tessellation_enabled = true;
			continue;
		}

		if (arg == "--profile") {
			options.config.profiler_enabled = true;
			continue;
		}

		// Upstream flags (2f885e25b, 8dbee8635, 84b1dfbb3), value-less as there.
		if (arg == "--guest-tsc") {
			options.config.guest_tsc_enabled = true;
			continue;
		}

		if (arg == "--relaxed-shader-float") {
			options.config.relaxed_shader_float = true;
			continue;
		}

		if (arg == "--lod-stats") {
			options.config.lod_stats_enabled = true;
			continue;
		}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (arg == "--redzone") {
			options.config.red_zone_protection_enabled = true;
			continue;
		}
#endif

		if (!arg.starts_with("--")) {
			::printf("game input must be provided with --game\n");
			return false;
		}

		if (!NextArg(argc, argv, i, value)) {
			::printf("missing value for %s\n", arg.c_str());
			return false;
		}

		if (arg == "--game") {
			if (!options.app0_dir.empty()) {
				::printf("--game can only be specified once\n");
				return false;
			}

			value           = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (Common::File::IsDirectoryExisting(path)) {
				options.app0_dir = path;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::IsSupportedArchive(path) && Common::File::IsFileExisting(path)) {
				const auto root = Common::MakeArchivePath(path);
				if (!Common::File::IsFileExisting(root / "eboot.bin")) {
					::printf("Archive does not contain eboot.bin: %s\n", value.c_str());
					return false;
				}
				options.app0_dir = root;
				options.elf      = "/app0/eboot.bin";
			} else if (Common::File::IsFileExisting(path)) {
				options.app0_dir = path.parent_path();

				if (options.app0_dir.empty()) {
					options.app0_dir = ".";
				}

				options.elf = std::filesystem::path("/app0") / path.filename();
			} else {
				::printf("--game must point to an existing directory, ELF, or archive: %s\n",
				         value.c_str());
				return false;
			}
		} else if (arg == "--guest-exec") {
			options.elf = Common::PathFromUtf8(Common::FixFilenameSlash(value));
			options.config.guest_arguments.assign(argv + i + 1, argv + argc);
			break;
		} else if (arg == "--game-patch") {
			if (!options.game_patch.empty()) {
				::printf("--game-patch can only be specified once\n");
				return false;
			}
			value = Common::FixFilenameSlash(value);
			const auto path = Common::PathFromUtf8(value);

			if (!Common::File::IsFileExisting(path)) {
				::printf("--game-patch must point to an existing file: %s\n", value.c_str());
				return false;
			}
			options.game_patch = path;
		} else if (arg == "--addcont-dir") {
			if (!options.addcont_dir.empty()) {
				::printf("--addcont-dir can only be specified once\n");
				return false;
			}
			options.addcont_dir = Common::PathFromUtf8(Common::FixFilenameSlash(value));
		} else if (arg == "--addcont") {
			auto entry = Common::ParseAddContent(value);
			if (!entry) {
				::printf("--addcont needs <16-character A-Z/0-9 label>[=<existing folder>]: %s\n",
				         value.c_str());
				return false;
			}
			options.addcont.push_back(std::move(*entry));
		} else if (arg == "--screen-width") {
			if (!ParseUint32(value, options.config.screen_width) ||
			    options.config.screen_width == 0) {
				::printf("invalid screen width: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--screen-height") {
			if (!ParseUint32(value, options.config.screen_height) ||
			    options.config.screen_height == 0) {
				::printf("invalid screen height: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--user-name") {
			if (value.empty() || value.size() > Config::MAX_USER_NAME_LENGTH) {
				::printf("invalid user name: must contain 1-%zu bytes\n",
				         Config::MAX_USER_NAME_LENGTH);
				return false;
			}
			options.config.user_name = value;
		} else if (arg == "--user-id") {
			if (!ParseUserId(value, options.config.user_id)) {
				::printf("invalid user ID: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--mic") {
			options.config.audio_input_device = value;
		} else if (arg == "--controller-color") {
			Config::ControllerColor color {};
			if (!ParseControllerColor(value, color)) {
				::printf("invalid controller color (expected #RRGGBB): %s\n", value.c_str());
				return false;
			}
			options.config.controller_color = color;
		} else if (arg == "--controller-volume") {
			if (!ParseUint32(value, options.config.controller_speaker_volume) ||
			    options.config.controller_speaker_volume > 100) {
				::printf("invalid controller volume: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--controller-vibration") {
			if (!ParseUint32(value, options.config.controller_vibration_intensity) ||
			    options.config.controller_vibration_intensity > 100) {
				::printf("invalid controller vibration intensity: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--keyboard-player") {
			if (!ParseUint32(value, options.config.keyboard_player) ||
			    options.config.keyboard_player > static_cast<uint32_t>(Config::MAX_PLAYERS)) {
				::printf("invalid keyboard player: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--audio-master-volume" || arg == "--audio-main-volume" ||
		           arg == "--audio-music-volume" || arg == "--audio-pad-speaker-volume" ||
		           arg == "--audio-objects-volume") {
			auto& target = arg == "--audio-master-volume" ? options.config.audio_master_volume
			               : arg == "--audio-main-volume" ? options.config.audio_main_volume
			               : arg == "--audio-music-volume"
			                   ? options.config.audio_music_volume
			               : arg == "--audio-objects-volume"
			                   ? options.config.audio_objects_volume
			                   : options.config.audio_pad_speaker_main_volume;
			if (!ParseUint32(value, target) || target > Config::MAX_AUDIO_VOLUME) {
				::printf("invalid %s (expected 0-%u): %s\n", arg.c_str(), Config::MAX_AUDIO_VOLUME,
				         value.c_str());
				return false;
			}
		} else if (arg == "--audio-objects") {
			if (!ParseBool(value, options.config.audio_objects_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--present-mode") {
			if (!ParseEnum(value, options.config.present_mode)) {
				::printf("invalid present mode: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--dlss") {
			options.config.dlss_folder = Common::PathFromUtf8(value);
		} else if (arg == "--dlss-preset") {
			if (value.size() != 1 || value[0] < 'A' || value[0] > 'O') {
				::printf("invalid DLSS preset: %s\n", value.c_str());
				return false;
			}
			options.config.dlss_preset = static_cast<uint32_t>(value[0] - 'A' + 1);
		} else if (arg == "--gpu") {
			if (!ParseInt32(value, options.config.gpu_index)) {
				::printf("invalid gpu index: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--vblank-frequency") {
			if (!ParseUint32(value, options.config.vblank_frequency)) {
				::printf("invalid vblank frequency: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--flush-draws") {
			if (!ParseUint32(value, options.config.flush_draws)) {
				::printf("invalid draw count: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--trophy-duration") {
			if (!ParseUint32(value, options.config.trophy_duration_seconds)) {
				::printf("invalid trophy duration: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--console-language") {
			if (!ParseConsoleLanguage(value, options.config.console_language)) {
				::printf("invalid console language: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--gpu-occlusion") {
			bool on = false;
			if (!ParseBool(value, on)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
			// The renderer reads KYTY_GPU_OCCLUSION once, at the first occlusion query.
#ifdef _WIN32
			_putenv_s("KYTY_GPU_OCCLUSION", on ? "1" : "0");
#else
			setenv("KYTY_GPU_OCCLUSION", on ? "1" : "0", 1);
#endif
		} else if (arg == "--network-connected") {
			if (!ParseBool(value, options.config.network_connected)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--vulkan-validation") {
			if (!ParseBool(value, options.config.vulkan_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--gpu-assisted-validation") {
			if (!ParseBool(value, options.config.gpu_assisted_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-validation") {
			if (!ParseBool(value, options.config.shader_validation_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--shader-optimization-type") {
			if (!ParseEnum(value, options.config.shader_optimization_type)) {
				::printf("invalid shader optimization type: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-direction") {
			if (!ParseEnum(value, options.config.shader_log_direction)) {
				::printf("invalid shader log direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--shader-log-folder") {
			options.config.shader_log_folder = Common::PathFromUtf8(value);
		} else if (arg == "--command-buffer-dump") {
			if (!ParseBool(value, options.config.command_buffer_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--command-buffer-dump-folder") {
			options.config.command_buffer_dump_folder = Common::PathFromUtf8(value);
		} else if (arg == "--graphics-debug-dump") {
			if (!ParseBool(value, options.config.graphics_debug_dump_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--printf-direction") {
			if (!ParseEnum(value, options.config.printf_direction)) {
				::printf("invalid printf direction: %s\n", value.c_str());
				return false;
			}
		} else if (arg == "--wheel") {
			bool enabled = false;
			if (!ParseBool(value, enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
			Config::Wheel::SetEnabled(enabled);
		} else if (arg == "--wheel-config") {
			Config::Wheel::SetConfigPath(Common::PathFromUtf8(value));
		} else if (arg == "--printf-output-file") {
			options.config.printf_output_file = Common::PathFromUtf8(value);
		} else if (arg == "--spirv-debug-printf") {
			if (!ParseBool(value, options.config.spirv_debug_printf_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-linear-images") {
			if (!ParseBool(value, options.config.readback_linear_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-tiled-images") {
			if (!ParseBool(value, options.config.readback_tiled_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--readback-storage-images") {
			if (!ParseBool(value, options.config.readback_storage_images)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--sync-raw-image-buffers") {
			if (!ParseBool(value, options.config.sync_raw_image_buffers)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--trophy-notifications") {
			if (!ParseBool(value, options.config.trophy_enabled)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--skip-notice-screen") {
			if (!ParseBool(value, options.config.skip_notice_screen)) {
				::printf("invalid boolean for %s: %s\n", arg.c_str(), value.c_str());
				return false;
			}
		} else if (arg == "--keymap") {
			const auto split = value.find('=');
			if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
				::printf("invalid keymap: %s\n", value.c_str());
				return false;
			}
			options.config.keymap.push_back(value);
		} else {
			::printf("unknown option: %s\n", arg.c_str());
			return false;
		}
	}

	if (options.config.gpu_assisted_validation_enabled) {
		options.config.vulkan_validation_enabled = true;
	}

	// KYTY_AMD_CPU=1|0|auto overrides --amd-cpu / --no-amd-cpu (the launcher's "Intel CPU
	// compatibility"), for presets and test harnesses.
	if (const char* env = std::getenv("KYTY_AMD_CPU"); env != nullptr && env[0] != '\0') {
		amd_cpu        = std::strcmp(env, "auto") == 0 ? AmdCpuChoice::Automatic
		                 : std::strcmp(env, "0") == 0  ? AmdCpuChoice::Off
		                                               : AmdCpuChoice::On;
		amd_cpu_source = amd_cpu == AmdCpuChoice::Automatic ? "automatic" : fmt::format("KYTY_AMD_CPU={}", env);
	}
	ResolveAmdCpu(amd_cpu, amd_cpu_source, options.config);

	// KYTY_READBACK_LINEAR_IMAGES=1|0 overrides --readback-linear-images (the launcher's "Readback"
	// box), so a title preset can turn it on for a game that needs it: Miles Morales reads a
	// GPU-written linear image on the CPU during the boot logos and freezes there without it.
	if (const char* env = std::getenv("KYTY_READBACK_LINEAR_IMAGES"); env != nullptr && env[0] != '\0') {
		bool on = options.config.readback_linear_images;
		if (!ParseBool(env, on)) {
			::printf("invalid boolean for KYTY_READBACK_LINEAR_IMAGES: %s\n", env);
			return false;
		}
		if (on != options.config.readback_linear_images) {
			::printf("KYTY_READBACK_LINEAR_IMAGES=%s: read-back of linear images %s (the command line had it %s)\n",
			         env, on ? "on" : "off", on ? "off" : "on");
		}
		options.config.readback_linear_images = on;
	}

	return show_help || (!options.app0_dir.empty() && !options.elf.empty());
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// NVIDIA Smooth Motion makes its frames with CUDA in the driver's present layer, which initializes
// CUDA only when the swapchain is created. By then the guest address space owns every free range
// below GUEST_ADDRESS_SPACE_END (kernel/memory.h), CUDA finds no room for its address range there,
// and cuInit fails with CUDA_ERROR_OUT_OF_MEMORY (2): Smooth Motion stays off (the layer's log is in
// C:/NvPresent/logs). So CUDA is initialized here, before any reservation, with the free ranges
// below a floor held for the duration of cuInit and released right after it: CUDA's range lands
// above the floor, and the guest address space later reserves everything around it.
//
// The floor is found, not configured. CUDA needs its range below 1 TiB (on an RTX 5070 Ti with
// driver 617.42 it reserves about 7-48 GiB there; with every free range below 1024 GiB held,
// cuInit fails with error 2), and the guest's fixed-address mappings must stay free, so the best
// floor is the end of the guest range itself: CUDA then sits right above it and opens no hole in
// guest VA (measured: 0xfd00000000-0xfec0000000 with the floor at 1008 GiB). The first attempt uses
// that floor. A failed cuInit can be retried, so if it fails (a driver that wants more room, or a
// guest range that grew towards 1 TiB) the floor steps down by 64 GiB, skipping the AMM window at the
// top of the guest range (kernel/memory.h kAmmWindowBase), until cuInit succeeds; the
// log line then says how much of the guest range CUDA took, since a fixed-address guest mapping
// there would fail.
// KYTY_CUDA_EARLY_INIT=1 turns it on (default off: it adds a DLL, CUDA threads and startup time to
// every NVIDIA run, while only RTX 40/50 drivers with Smooth Motion gain anything).
// KYTY_CUDA_VA_FLOOR_GB=<GiB> forces one floor (no search). Without nvcuda.dll (no NVIDIA driver)
// nothing happens.
static uint64_t ReservedBytesIn(uint64_t begin, uint64_t end) {
	uint64_t total = 0;
	uint64_t next  = begin;
	while (next < end) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(next), &info, sizeof(info)) == 0) {
			break;
		}
		const auto base       = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto region_end = std::min<uint64_t>(end, base + static_cast<uint64_t>(info.RegionSize));
		if (region_end <= next) {
			break;
		}
		if (info.State != MEM_FREE) {
			total += region_end - std::max<uint64_t>(base, next);
		}
		next = region_end;
	}
	return total;
}

static void InitCudaEarly() {
	if (const char* value = std::getenv("KYTY_CUDA_EARLY_INIT"); value == nullptr || std::strcmp(value, "1") != 0) {
		return;
	}
	HMODULE cuda = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
	if (cuda == nullptr) {
		return;
	}
	using CuInit    = int(__stdcall*)(unsigned int);
	const auto init = reinterpret_cast<CuInit>(GetProcAddress(cuda, "cuInit"));
	if (init == nullptr) {
		return;
	}
	constexpr uint64_t GiB        = uint64_t {1} << 30u;
	constexpr uint64_t Step       = 64 * GiB;
	constexpr uint64_t LowestTry  = 64 * GiB;
	const uint64_t     guest_end  = Libs::LibKernel::Memory::GUEST_ADDRESS_SPACE_END;
	uint64_t           first      = guest_end;
	bool               search     = true;
	if (const char* value = std::getenv("KYTY_CUDA_VA_FLOOR_GB"); value != nullptr && std::atoi(value) > 0) {
		first  = std::min<uint64_t>(static_cast<uint64_t>(std::atoi(value)), 65536) * GiB;
		search = false;
	}
	const uint64_t amm_begin = Libs::LibKernel::Memory::kAmmWindowBase;
	const uint64_t amm_end   = amm_begin + Libs::LibKernel::Memory::kAmmWindowSize;
	int            result    = -1;
	uint64_t       floor     = first;
	uint64_t       taken     = 0;
	uint64_t       amm_taken = 0;
	int            tries     = 0;
	for (;;) {
		// Hold the free ranges below the floor.
		std::vector<void*> held;
		uint64_t           next = 0x10000;
		while (next < floor) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(next), &info, sizeof(info)) == 0) {
				break;
			}
			const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
			const auto end  = std::min<uint64_t>(floor, base + static_cast<uint64_t>(info.RegionSize));
			if (end <= next) {
				break;
			}
			// Reservations start on the 64 KiB allocation granularity.
			const auto start = (std::max<uint64_t>(base, next) + 0xffffu) & ~uint64_t {0xffff};
			if (info.State == MEM_FREE && end > start) {
				void* ptr = VirtualAlloc(reinterpret_cast<void*>(start), end - start, MEM_RESERVE, PAGE_NOACCESS);
				if (ptr != nullptr) {
					held.push_back(ptr);
				}
			}
			next = end;
		}
		const auto before     = floor < guest_end ? ReservedBytesIn(floor, guest_end) : 0;
		const auto amm_before = ReservedBytesIn(amm_begin, amm_end);
		result                = init(0);
		tries++;
		if (result == 0 && floor < guest_end) {
			taken     = ReservedBytesIn(floor, guest_end) - before;
			amm_taken = ReservedBytesIn(amm_begin, amm_end) - amm_before;
		}
		for (void* ptr: held) {
			VirtualFree(ptr, 0, MEM_RELEASE);
		}
		if (result == 0 || !search || floor <= LowestTry) {
			break;
		}
		floor = floor > LowestTry + Step ? floor - Step : LowestTry;
		// CUDA's range lands right above the floor: a floor inside the AMM window (or less than one
		// step below it) would put it where AMM map commands place pages at fixed addresses
		// (kAmmWindowBase), so the search continues one step below the window.
		if (floor > amm_begin - Step && floor < amm_end) {
			floor = amm_begin - Step;
		}
	}
	if (result != 0) {
		::printf("CUDA: early initialization failed (cuInit %d, %d floor(s) tried down to %" PRIu64
		         " GiB), NVIDIA Smooth Motion will not start\n",
		         result, tries, floor / GiB);
	} else if (taken == 0) {
		::printf("CUDA: initialized above the guest address space (floor %" PRIu64 " GiB, %d tr%s)\n", floor / GiB,
		         tries, tries == 1 ? "y" : "ies");
	} else {
		::printf("CUDA: initialized with the floor at %" PRIu64 " GiB, inside the guest address space (ends at %" PRIu64
		         " GiB): CUDA holds %" PRIu64 " MiB of guest VA, which fixed-address guest mappings cannot use%s\n",
		         floor / GiB, guest_end / GiB, taken >> 20u,
		         amm_taken != 0 ? " (some of it in the AMM window: games that use AMM may fail, turn "
		                          "KYTY_CUDA_EARLY_INIT off for them)"
		                        : "");
	}
}
#endif

// An exception no one catches (a worker thread of the program cache, for one) ends the process
// with 0xC0000409 and no message: name the exception first, then abort as std::terminate does.
static void ReportUncaughtException() {
#if defined(__cpp_exceptions)
	try {
		if (const auto exception = std::current_exception(); exception != nullptr) {
			std::rethrow_exception(exception);
		}
		std::fprintf(stderr, "terminate called without an active exception\n");
	} catch (const std::exception& error) {
		std::fprintf(stderr, "terminate: uncaught exception: %s\n", error.what());
	} catch (...) {
		std::fprintf(stderr, "terminate: uncaught exception of an unknown type\n");
	}
#else
	std::fprintf(stderr, "terminate called\n");
#endif
	std::fflush(stderr);
	std::abort();
}

// The directory of kyty_emulator itself (empty when unknown).
static std::filesystem::path EmulatorDirectory() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::vector<wchar_t> path(32768);
	const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
	if (length == 0 || length >= path.size()) {
		return {};
	}
	return std::filesystem::path(std::wstring(path.data(), length)).parent_path();
#else
	std::error_code ec;
	auto            exe = std::filesystem::read_symlink("/proc/self/exe", ec);
	return ec ? std::filesystem::path {} : exe.parent_path();
#endif
}

// u59-preset.<TITLE_ID>.json when no launcher applied it. Per-title presets are applied by the
// launcher (KYTY_TITLE_PRESET names the file), but a launcher older than them (int21 and before)
// starting a newer emulator, or a direct start, ran the game without its title's switches: Ratchet &
// Clank then ran without KYTY_INDIRECT_COHERENT and the rest of its preset and lost the device at
// its first gameplay load. The emulator then loads the file itself, from its own folder or the one
// above (where the launcher looks), with the launcher's precedence: over the bundled preset and the
// environment, under the command line (it runs before the arguments are parsed, so
// --gpu-occlusion still wins). KYTY_TITLE_PRESET set to anything (e.g. "none") skips it.
static void ApplyTitlePresetFallback(int argc, char* argv[]) {
	if (!U59Preset::EmulatorTitleFallbackWanted(std::getenv("KYTY_TITLE_PRESET"))) {
		return;
	}
	std::string game;
	for (int i = 1; i + 1 < argc; i++) {
		if (std::string_view(argv[i]) == "--game") {
			game = argv[i + 1];
			break;
		}
	}
	if (game.empty()) {
		return;
	}
	const auto      path = PathFromUtf8(FixFilenameSlash(game));
	std::error_code ec;
	std::filesystem::path app0;
	if (std::filesystem::is_directory(path, ec)) {
		app0 = path;
	} else if (IsSupportedArchive(path) && File::IsFileExisting(path)) {
		app0 = MakeArchivePath(path);
	} else if (std::filesystem::is_regular_file(path, ec)) {
		app0 = path.parent_path();
	} else {
		return;
	}
	File param;
	param.Open(app0 / "sce_sys" / "param.json", File::Mode::Read);
	if (param.IsInvalid()) {
		return;
	}
	const auto buffer = param.ReadWholeBuffer();
	param.Close();
	const auto title_id = U59Preset::TitleIdFromParamJson(
	    std::string_view(reinterpret_cast<const char*>(buffer.data()), buffer.size()));
	const auto dir = EmulatorDirectory();
	if (title_id.empty() || dir.empty()) {
		return;
	}
	const auto result = U59Preset::LoadTitle({dir, dir.parent_path()}, title_id);
	if (result.status == U59Preset::Status::Missing) {
		return;
	}
	if (result.status != U59Preset::Status::Loaded) {
		std::printf("%s was not applied: %s\n", U59Preset::TitleFileName(title_id).c_str(),
		            result.error.c_str());
		return;
	}
	const auto set = [](const std::string& key, const std::string& value) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		_putenv_s(key.c_str(), value.c_str());
#else
		setenv(key.c_str(), value.c_str(), 1);
#endif
	};
	for (const auto& entry: result.entries) {
		set(entry.key, entry.value);
	}
	// Emulator::Run prints "Per-title preset applied: <file>" into the log.
	set("KYTY_TITLE_PRESET", PathToString(result.path));
	std::printf("The launcher did not apply %s: the emulator applied its %zu variable(s).\n",
	            U59Preset::TitleFileName(title_id).c_str(), result.entries.size());
}

static int Main(int argc, char* argv[]) {
	std::set_terminate(ReportUncaughtException);
#if defined(KYTY_SEGMENT_HEAP)
	// KYTY_SEGMENT_HEAP build: which heap the manifest gave (a log for the A/B's record).
	std::printf("Process heap: %s (KYTY_SEGMENT_HEAP build)\n",
	            Common::ProcessHeapKindName(Common::GetProcessHeapKind()));
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	InitCudaEarly();
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// KYTY_RADV_TRANSFER_QUEUE=1 (default off; chenxiao07/KytyPS5 1bfdf6ded): asks RADV for its
	// transfer-only queue family, which it shows only when asked, before the driver loads (Mesa 26
	// reads RADV_EXPERIMENTAL, older Mesa RADV_PERFTEST; other drivers neither), unless the variable
	// is set already. The upload DMA (KYTY_UPLOAD_DMA) then has a copy queue of its own on AMD;
	// chenxiao's readbacks gained 19-27% on an RX 9060 XT / 9070 XT with it.
	if (const char* ask = std::getenv("KYTY_RADV_TRANSFER_QUEUE"); ask != nullptr && ask[0] == '1') {
		setenv("RADV_EXPERIMENTAL", "transfer_queue", 0);
		setenv("RADV_PERFTEST", "transfer_queue", 0);
	}
#endif
	// KYTY_SMALL_CPU_RULE (common/topologyRule.h): before the CPU placement reads its switches.
	Common::TopologyRule::ApplyAtStartup();
	InitializeThreads();
#if defined(KYTY_PGO_GENERATE)
	// After the address-space reservations: the thread's stack must not land in the guest range.
	StartPgoDumps();
#endif

	RunOptions options;
	bool       show_help = false;

	if (argc < 2) {
		PrintUsage();
		return 0;
	}

	ApplyTitlePresetFallback(argc, argv);

	if (!ParseArgs(argc, argv, options, show_help)) {
		PrintUsage();
		return 1;
	}

	if (show_help) {
		PrintUsage();
		return 0;
	}

	Run(options);

	return 0;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

int wmain(int argc, wchar_t* argv[]) {
    std::vector<std::string> utf8_args;
    utf8_args.reserve(static_cast<size_t>(argc));

    for (int index = 0; index < argc; index++) {
        const std::wstring_view wide(argv[index]);
        const std::u16string utf16(wide.begin(), wide.end());

        utf8_args.push_back(Common::Utf16ToUtf8(utf16));
    }

    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_args.size());

    for (auto& argument: utf8_args) {
        utf8_argv.push_back(argument.data());
    }

    return Main(argc, utf8_argv.data());
}

#else

int main(int argc, char* argv[]) {
    return Main(argc, argv);
}

#endif
