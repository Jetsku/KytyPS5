#ifndef LAUNCHER_INCLUDE_CONFIGURATION_H_
#define LAUNCHER_INCLUDE_CONFIGURATION_H_

#include "common/emulatorConfig.h"

#include <QByteArray>
#include <QChar>
#include <QColor>
#include <QJsonObject>
#include <QMetaEnum>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QPair>
#include <QProcessEnvironment>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QVariant>

template <class T>
inline QStringList EnumToList() {
	QStringList ret;
	auto        me    = QMetaEnum::fromType<T>();
	int         count = me.keyCount();
	for (int i = 0; i < count; i++) {
		ret << QString(me.key(i));
	}
	return ret;
}

template <class T>
T TextToEnum(const QString& text) {
	auto me = QMetaEnum::fromType<T>();
	return static_cast<T>(me.keyToValue(text.toUtf8().constData()));
}

template <class T>
QString EnumToText(T value) {
	return QString(QMetaEnum::fromType<T>().valueToKey(static_cast<int>(value)));
}

struct ControllerSettings {
	QString color;
	int     speaker_volume      = 50;
	int     vibration_intensity = 100;
	// 0 shares player 1 with the first gamepad; 1..MAX_PLAYERS dedicates that player to the keyboard.
	int     keyboard_player     = 0;
	bool    wheel_enabled       = false;
	// wheel.json beside the settings file; set when the settings are read, not stored.
	QString wheel_config;

	void WriteSettings(QSettings* s) const {
		s->setValue("controller_color", color);
		s->setValue("controller_speaker_volume", speaker_volume);
		s->setValue("controller_vibration_intensity", vibration_intensity);
		s->setValue("controller_keyboard_player", keyboard_player);
		s->setValue("wheel_enabled", wheel_enabled);
	}

	void ReadSettings(QSettings* s) {
		const QColor saved_color(s->value("controller_color").toString());
		color = saved_color.isValid() ? saved_color.name(QColor::HexRgb) : QString {};
		const auto read_percent = [s](const char* key, int fallback) {
			bool      ok    = false;
			const int value = s->value(key, fallback).toInt(&ok);
			return ok ? qBound(0, value, 100) : fallback;
		};
		speaker_volume      = read_percent("controller_speaker_volume", 50);
		vibration_intensity = read_percent("controller_vibration_intensity", 100);
		keyboard_player = qBound(0, s->value("controller_keyboard_player", 0).toInt(),
		                         Config::MAX_PLAYERS);
		wheel_enabled       = s->value("wheel_enabled", false).toBool();
	}
};

// Host mix levels (percent, 0..Config::MAX_AUDIO_VOLUME), global like the controller settings.
struct AudioMixSettings {
	int master      = 100;
	int main        = 100;
	int music       = 100;
	int pad_speaker = static_cast<int>(Config::DEFAULT_AUDIO_PAD_SPEAKER_MAIN_VOLUME);
	int objects     = 100;
	bool objects_enabled = true;

	void WriteSettings(QSettings* s) const {
		s->setValue("audio_master_volume", master);
		s->setValue("audio_main_volume", main);
		s->setValue("audio_music_volume", music);
		s->setValue("audio_pad_speaker_main_volume", pad_speaker);
		s->setValue("audio_objects_volume", objects);
		s->setValue("audio_objects_enabled", objects_enabled);
	}

	void ReadSettings(QSettings* s) {
		const auto read_percent = [s](const char* key, int fallback) {
			bool      ok    = false;
			const int value = s->value(key, fallback).toInt(&ok);
			return ok ? qBound(0, value, static_cast<int>(Config::MAX_AUDIO_VOLUME)) : fallback;
		};
		const AudioMixSettings defaults;
		master      = read_percent("audio_master_volume", defaults.master);
		main        = read_percent("audio_main_volume", defaults.main);
		music       = read_percent("audio_music_volume", defaults.music);
		pad_speaker = read_percent("audio_pad_speaker_main_volume", defaults.pad_speaker);
		objects     = read_percent("audio_objects_volume", defaults.objects);
		objects_enabled = s->value("audio_objects_enabled", defaults.objects_enabled).toBool();
	}
};

class Configuration: public QObject {
	Q_OBJECT

public:
	static constexpr int DEFAULT_CONSOLE_LANGUAGE = 1;
	static constexpr int MAX_CONSOLE_LANGUAGE     = 29;

	static constexpr int MIN_SCREEN_SIZE = 320;
	static constexpr int MAX_SCREEN_SIZE = 16384;

	static QString NormalizeResolution(const QString& text);
	// The window size of a new configuration, of Reset, and of a saved value that cannot be read.
	static QString DefaultScreenResolution() { return QStringLiteral("1920x1080"); }

	enum class ShaderOptimizationType { None, Size, Performance };
	Q_ENUM(ShaderOptimizationType)
	static ShaderOptimizationType ShaderOptimizationTypeFromText(const QString& text);

	enum class PresentMode { Fifo, Mailbox, Immediate };
	Q_ENUM(PresentMode)

	enum class LogDirection { Silent, Console, File };
	Q_ENUM(LogDirection)

	enum class PssrUpscaler { Linear, DLSS };
	Q_ENUM(PssrUpscaler)

	enum class GameStatus { Unknown, InGame, Logo, DoesntBoot, MainMenu };
	Q_ENUM(GameStatus)

	Configuration() = default;

	QString     name;
	QString     title_id;    /* Serial / title id from sce_sys/param.json */
	QString     gameVersion; /* appVersion / contentVersion from sce_sys/param.json */
	QString     firmwareVer; /* requiredSystemSoftwareVersion from sce_sys/param.json */
	QString     basedir;     /* Game base directory */
	QString     game_path;   /* Launcher-unique game path */
	QString     addons_dir;  /* Global setting; not saved per game. */
	QStringList addcont;     /* User add-ons ("LABEL" or "LABEL=DIR"), saved by title ID. */
	bool        custom_settings = false;
	bool        pro_supported   = false; /* attribute3 bit 22 from sce_sys/param.json */
	GameStatus  game_status     = GameStatus::Unknown;
	QString     game_comment;

	// Controller preferences always come from the global configuration.
	ControllerSettings controller;
	// So do the audio mix levels.
	AudioMixSettings audio_mix;
	void WriteGlobalOnlySettings(QSettings* s) const { audio_mix.WriteSettings(s); }
	void ReadGlobalOnlySettings(QSettings* s) { audio_mix.ReadSettings(s); }
	void CopyGlobalOnlySettingsFrom(const Configuration& other) {
		controller = other.controller;
		audio_mix  = other.audio_mix;
	}

	QString                screen_resolution           = DefaultScreenResolution();
	QString                user_name                   = "Kyty";
	int                    user_id                     = Config::DEFAULT_USER_ID;
	QString                audio_input_device;
	PresentMode            present_mode                = PresentMode::Mailbox;
	bool                   smooth_frametime_fix        = false;
	int                    gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   hdr_enabled                 = false;
	bool                   hide_cursor_enabled         = false;
	bool                   stretch_enabled             = false;
	bool                   readback_linear_images      = false;
	bool                   readback_tiled_images       = false;
	bool                   readback_storage_images     = false;
	bool                   sync_raw_image_buffers      = false;
	int                    flush_draws                 = 0;
	bool                   tessellation_enabled        = false;
	bool                   occlusion_queries_enabled   = false;
	// Occlusion mode, saved per game (key gpu_occlusion_accurate): false = the bundled preset's
	// (performance: always visible), true = accurate GPU occlusion queries (--gpu-occlusion on).
	// A game config saved before it was per game inherits the global value when read.
	bool                   gpu_occlusion_accurate      = false;
	bool                   relaxed_shader_float        = false;
	// "Faster GPU synchronisation (experimental)", per game (key fast_gpu_sync): the game starts
	// with FastGpuSyncVariables() (ApplyFastGpuSync).
	bool                   fast_gpu_sync               = false;
	bool                   shader_cache_enabled        = true;
	bool                   trophy_enabled              = true;
	int                    trophy_duration_seconds     = 6;
	bool                   perf_overlay_enabled        = false;
	bool                   skip_notice_screen          = false;
	bool                   network_connected           = false;
	bool                   psn_emulation_enabled       = false;
	int                    vblank_frequency            = 60;
	int                    console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = false;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::Performance;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	QString                shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	QString                command_buffer_dump_folder  = "_Buffers";
	LogDirection           printf_direction            = LogDirection::Silent;
	QString                printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   renderdoc_enabled           = false;
	bool                   pro_enabled                 = false;
	PssrUpscaler           pssr_upscaler               = PssrUpscaler::Linear;
	QString                dlss_folder;
	QString                amd_cpu_mode                = QStringLiteral("auto"); // or "on", "off"
	bool                   guest_tsc_enabled           = false;
#if defined(_WIN32)
	bool red_zone_protection_enabled = false;
#endif
	QStringList host_input_mapping;

	QString elf = QStringLiteral("eboot.bin");

	void CopyEmulatorSettingsFrom(const Configuration& other) {
		screen_resolution           = other.screen_resolution;
		user_name                   = other.user_name;
		user_id                     = other.user_id;
		audio_input_device          = other.audio_input_device;
		present_mode                = other.present_mode;
		smooth_frametime_fix        = other.smooth_frametime_fix;
		gpu_index                   = other.gpu_index;
		fullscreen_enabled          = other.fullscreen_enabled;
		hdr_enabled                 = other.hdr_enabled;
		hide_cursor_enabled         = other.hide_cursor_enabled;
		stretch_enabled             = other.stretch_enabled;
		readback_linear_images      = other.readback_linear_images;
		readback_tiled_images       = other.readback_tiled_images;
		readback_storage_images     = other.readback_storage_images;
		sync_raw_image_buffers      = other.sync_raw_image_buffers;
		flush_draws                 = other.flush_draws;
		tessellation_enabled        = other.tessellation_enabled;
		occlusion_queries_enabled   = other.occlusion_queries_enabled;
		gpu_occlusion_accurate      = other.gpu_occlusion_accurate;
		relaxed_shader_float        = other.relaxed_shader_float;
		fast_gpu_sync               = other.fast_gpu_sync;
		shader_cache_enabled        = other.shader_cache_enabled;
		trophy_enabled              = other.trophy_enabled;
		trophy_duration_seconds     = other.trophy_duration_seconds;
		perf_overlay_enabled        = other.perf_overlay_enabled;
		skip_notice_screen          = other.skip_notice_screen;
		network_connected           = other.network_connected;
		psn_emulation_enabled       = other.psn_emulation_enabled;
		vblank_frequency            = other.vblank_frequency;
		console_language            = other.console_language;
		vulkan_validation_enabled   = other.vulkan_validation_enabled;
		shader_validation_enabled   = other.shader_validation_enabled;
		shader_optimization_type    = other.shader_optimization_type;
		shader_log_direction        = other.shader_log_direction;
		shader_log_folder           = other.shader_log_folder;
		command_buffer_dump_enabled = other.command_buffer_dump_enabled;
		command_buffer_dump_folder  = other.command_buffer_dump_folder;
		printf_direction            = other.printf_direction;
		printf_output_file          = other.printf_output_file;
		profiler_enabled            = other.profiler_enabled;
		renderdoc_enabled           = other.renderdoc_enabled;
		pro_enabled                 = other.pro_enabled;
		pssr_upscaler               = other.pssr_upscaler;
		dlss_folder                 = other.dlss_folder;
		amd_cpu_mode                = other.amd_cpu_mode;
		guest_tsc_enabled           = other.guest_tsc_enabled;
#if defined(_WIN32)
		red_zone_protection_enabled = other.red_zone_protection_enabled;
#endif
		host_input_mapping = other.host_input_mapping;
	}

	void CopyGameInfoFrom(const Configuration& other) {
		name            = other.name;
		title_id        = other.title_id;
		gameVersion     = other.gameVersion;
		firmwareVer     = other.firmwareVer;
		basedir         = other.basedir;
		game_path       = other.game_path;
		custom_settings = other.custom_settings;
		pro_supported   = other.pro_supported;
		game_status     = other.game_status;
		game_comment    = other.game_comment;
	}

	void WriteSettings(QSettings* s) const;
	void ReadSettings(QSettings* s);

	[[nodiscard]] QVariantMap GameSettings() const;

	// The emulator switches of fast_gpu_sync: precise GPU hazards, ranged indirect arguments, guest
	// release barriers dropped where nothing waits on them, batched DMA barriers.
	[[nodiscard]] static const QList<QPair<QString, QString>>& FastGpuSyncVariables();
	// With fast_gpu_sync, adds each of FastGpuSyncVariables() that environment does not set yet:
	// a value from u59-preset.json, a title preset or the user's environment wins. Without it,
	// nothing changes.
	void ApplyFastGpuSync(QProcessEnvironment& environment) const;
	// Call on a temporary configuration: validation can fail after loading values.
	bool SetGameSettings(const QJsonObject& settings, QString& error);

	// The launcher settings a title was tested with (the table in configuration.cpp), as in a game
	// config export: setting name -> text value. Empty for a title without an entry. A game without
	// a config of its own runs with them over the global settings; Reset and the Recommended
	// button of its dialog set them.
	[[nodiscard]] static QJsonObject RecommendedGameSettings(const QString& title_id);
	// Sets RecommendedGameSettings(title_id) over the current values and leaves every other
	// setting as it is; false (nothing changed) for a title without an entry.
	bool ApplyRecommendedGameSettings(const QString& title_id);

private:
	template <class Settings>
	void ReadGameSettingsValues(const Settings& s);
};

Q_DECLARE_METATYPE(Configuration*)

#endif /* LAUNCHER_INCLUDE_CONFIGURATION_H_ */
