#ifndef KYTY_COMMON_EMULATOR_CONFIG_H_
#define KYTY_COMMON_EMULATOR_CONFIG_H_

#include "common/common.h"

#include <cstddef>
#include <array>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace Config {

void Initialize();
void Shutdown();

enum class ShaderOptimizationType { None, Size, Performance };

struct Lifecycle {
	static constexpr const char* name       = "Config";
	static constexpr auto        initialize = Config::Initialize;
	static constexpr auto        shutdown   = Config::Shutdown;
};

enum class LogDirection { Silent, Console, File };

enum class PresentMode { Fifo, Mailbox, Immediate };

using Keymap = std::vector<std::string>;
using ControllerColor = std::array<uint8_t, 3>;

constexpr uint32_t DEFAULT_CONSOLE_LANGUAGE = 1;
constexpr uint32_t MAX_CONSOLE_LANGUAGE     = 29;
constexpr std::size_t MAX_USER_NAME_LENGTH = 16;
constexpr int32_t DEFAULT_USER_ID           = 1000;
constexpr uint32_t MAX_AUDIO_VOLUME         = 200;
// Astro Bot sends a lot of its sound to the DualSense speaker (PADSPK); played on the main output
// it measured up to ~19 dB above the main mix (2026-10-06). The console speaker is small, so it
// is turned down (30% = -10.5 dB) when no DualSense plays it.
constexpr uint32_t DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME = 30;

constexpr int     MAX_PLAYERS         = 4;
constexpr int32_t PLAYER_USER_ID_STEP = 0x100;
constexpr int32_t MAX_USER_ID =
    std::numeric_limits<int32_t>::max() - (MAX_PLAYERS - 1) * PLAYER_USER_ID_STEP;

constexpr bool IsConfiguredUserIdValid(int32_t user_id) {
	constexpr int32_t USER_ID_EVERYONE = 0xfe;
	constexpr int32_t USER_ID_SYSTEM   = 0xff;
	return user_id >= 0 && user_id <= MAX_USER_ID && user_id != USER_ID_EVERYONE &&
	       user_id != USER_ID_SYSTEM;
}

struct ConfigOptions {
	uint32_t               screen_width                = 1280;
	uint32_t               screen_height               = 720;
	std::string            user_name                   = "Kyty";
	int32_t                user_id                     = DEFAULT_USER_ID;
	std::string            audio_input_device;
	std::optional<ControllerColor> controller_color;
	uint32_t               controller_speaker_volume      = 50;
	uint32_t               controller_vibration_intensity = 100;
	// Host mix levels in percent (0-200, 100 = unity); see libs/audioMix.h.
	uint32_t               audio_master_volume            = 100;
	uint32_t               audio_main_volume              = 100;
	uint32_t               audio_music_volume             = 100;
	uint32_t               audio_pad_speaker_main_volume  = DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME;
	// AudioOut2 3D object ports, mixed into the main bed (main and master apply on top).
	uint32_t               audio_objects_volume           = 100;
	bool                   audio_objects_enabled          = true;
	// 0 shares player 1 with the first gamepad; 1..MAX_PLAYERS dedicates that player to the keyboard.
	uint32_t               keyboard_player                = 0;
	PresentMode            present_mode                = PresentMode::Mailbox;
	int32_t                gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   hdr_enabled                 = false;
	bool                   hide_cursor_enabled         = false;
	bool                   stretch_enabled             = false;
	bool                   splash_enabled              = true;
	bool                   vr_enabled                  = false;
	bool                   pro_enabled                 = false;
	// Folder with the user's DLSS model; DLSS upscales PSSR in Pro mode when set.
	std::filesystem::path  dlss_folder;
	uint32_t               dlss_preset                 = 0; // 0 is the model default
	bool                   amd_cpu_enabled             = false;
	std::string            amd_cpu_decision; // why amd_cpu_enabled is set as it is, for the log
	bool                   guest_tsc_enabled           = false;
	bool                   network_connected           = false;
	bool                   psn_emulation_enabled       = false;
	uint32_t               vblank_frequency            = 60;
	uint32_t               console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	bool                   shader_cache_enabled        = false;
	// Kept for our launcher and scripts (upstream 57963dd12 dropped the unread option).
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::None;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	std::filesystem::path  shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	std::filesystem::path  command_buffer_dump_folder  = "_Buffers";
	bool                   graphics_debug_dump_enabled = false;
	LogDirection           printf_direction            = LogDirection::Silent;
	std::filesystem::path  printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   spirv_debug_printf_enabled  = false;
	bool                   gpu_assisted_validation_enabled = false;
	bool                   renderdoc_enabled           = false;
	bool                   readback_linear_images      = false;
	bool                   readback_tiled_images       = false;
	bool                   readback_storage_images     = false;
	bool                   sync_raw_image_buffers      = false;
	uint32_t               flush_draws                 = 0; // 0 is off
	bool                   tessellation_enabled        = false;
	bool                   occlusion_queries_enabled   = false;
	bool                   relaxed_shader_float        = false;
	bool                   lod_stats_enabled           = false;
	bool                   trophy_enabled              = true;
	uint32_t               trophy_duration_seconds     = 6;
	bool                   perf_overlay_enabled            = false;
	bool                   playgo_hack_enabled         = false;
	bool                   skip_notice_screen          = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	bool red_zone_protection_enabled = false;
#endif
	Keymap keymap;
	// Ordinary boots need a non-null argv[0]; --guest-exec supplies the caller's arguments.
	std::vector<std::string> guest_arguments {""};
};

void Load(const ConfigOptions& cfg);

uint32_t GetScreenWidth();
uint32_t GetScreenHeight();
const std::string& GetUserName();
int32_t  GetUserId();
const std::string& GetAudioInputDevice();
const std::optional<ControllerColor>& GetControllerColor();
uint32_t GetControllerSpeakerVolume();
uint32_t GetControllerVibrationIntensity();
uint32_t GetAudioMasterVolume();
uint32_t GetAudioMainVolume();
uint32_t GetAudioMusicVolume();
uint32_t GetAudioPadSpeakerOnMainVolume();
uint32_t GetAudioObjectsVolume();
bool     AudioObjectsEnabled();
uint32_t GetKeyboardPlayer();
// Player 0 is the configured user; later players derive their IDs from it.
int32_t  GetPlayerUserId(int player);
// Returns -1 when the user ID does not belong to any player.
int      GetPlayerFromUserId(int32_t user_id);
// Players after the first are guests: they have no profile, trophies or NP account.
bool     IsGuestUserId(int32_t user_id);
PresentMode GetPresentMode();
int32_t GetGpuIndex();
bool     FullscreenEnabled();
bool     HdrEnabled();
bool     HideCursorEnabled();
bool     StretchEnabled();
bool     VrEnabled();
bool     Ps5ProEnabled();
const std::filesystem::path& GetDlssFolder();
uint32_t GetDlssPreset();
bool     AmdCpuEnabled();
const std::string& GetAmdCpuDecision();
bool     GuestTscEnabled();
bool     IsNetworkConnected();
bool     PsnEmulationEnabled();
uint32_t GetVblankFrequency();
uint32_t GetConsoleLanguage();
bool     VulkanValidationEnabled();

bool                   ShaderValidationEnabled();
bool                   ShaderCacheEnabled();
ShaderOptimizationType GetShaderOptimizationType();
LogDirection           GetShaderLogDirection();
std::filesystem::path  GetShaderLogFolder();

bool                  CommandBufferDumpEnabled();
std::filesystem::path GetCommandBufferDumpFolder();

// Read per draw on the command processor (debug.cpp LogDrawPhase, descriptors.cpp): copies of the
// loaded options, so the check inlines (the common library is not part of the emulator's LTO).
namespace Detail {
inline bool g_graphics_debug_dump_enabled = false;
inline bool g_sync_raw_image_buffers      = false;
} // namespace Detail

inline bool GraphicsDebugDumpEnabled() {
	return Detail::g_graphics_debug_dump_enabled;
}

LogDirection          GetPrintfDirection();
std::filesystem::path GetPrintfOutputFile();

bool ProfilerEnabled();

bool SpirvDebugPrintfEnabled();

bool GpuAssistedValidationEnabled();

bool RenderDocEnabled();
bool ReadbackLinearImagesEnabled();
bool ReadbackTiledImagesEnabled();
bool ReadbackStorageImagesEnabled();
inline bool SyncRawImageBuffersEnabled() {
	return Detail::g_sync_raw_image_buffers;
}
uint32_t GetFlushDraws();
bool TessellationEnabled();
bool OcclusionQueriesEnabled();
bool RelaxedShaderFloatEnabled();
bool LodStatsEnabled();
bool TrophyEnabled();
uint32_t GetTrophyDurationSeconds();
bool PlayGoHackEnabled();
bool SkipNoticeScreen();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled();
#endif

const Keymap& GetKeymap();
const std::vector<std::string>& GetGuestArguments();

} // namespace Config

#endif /* KYTY_COMMON_EMULATOR_CONFIG_H_ */
