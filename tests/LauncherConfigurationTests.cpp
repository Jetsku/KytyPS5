// Launcher configuration: the per-game occlusion mode, the faster GPU synchronisation option and
// the shader optimization fallback.
#include "configuration.h"

#include <QJsonObject>
#include <QProcessEnvironment>
#include <QSettings>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "LauncherConfigurationTests: %s\n", text);
		std::abort();
	}
}

// Mirrors ConfigurationListWidget::ReadSettings for one game entry: the entry starts from the
// global occlusion mode, so configs saved while the mode was global-only keep it.
void ReadGame(QSettings& s, int index, const Configuration& global, Configuration& info) {
	s.beginReadArray(QStringLiteral("GameConfigurations"));
	s.setArrayIndex(index);
	info.gpu_occlusion_accurate = global.gpu_occlusion_accurate;
	info.ReadSettings(&s);
	s.endArray();
}
} // namespace

int main() {
	QTemporaryDir dir;
	Check(dir.isValid(), "cannot create a temporary directory");
	const auto path = dir.filePath(QStringLiteral("Kyty.ini"));

	{
		// Layout written by launchers where the occlusion mode was global only.
		QSettings s(path, QSettings::IniFormat);
		s.beginGroup(QStringLiteral("GlobalConfiguration"));
		s.setValue(QStringLiteral("shader_optimization_type"), QStringLiteral("Performance"));
		s.setValue(QStringLiteral("gpu_occlusion_accurate"), true);
		s.endGroup();
		s.beginWriteArray(QStringLiteral("GameConfigurations"));
		s.setArrayIndex(0); // old game config: no occlusion key, empty optimization type
		s.setValue(QStringLiteral("game_path"), QStringLiteral("A"));
		s.setValue(QStringLiteral("shader_optimization_type"), QString());
		s.setArrayIndex(1); // new game config with its own value
		s.setValue(QStringLiteral("game_path"), QStringLiteral("B"));
		s.setValue(QStringLiteral("gpu_occlusion_accurate"), false);
		s.setValue(QStringLiteral("shader_optimization_type"), QStringLiteral("Size"));
		s.setArrayIndex(2); // unknown optimization type
		s.setValue(QStringLiteral("game_path"), QStringLiteral("C"));
		s.setValue(QStringLiteral("shader_optimization_type"), QStringLiteral("Fastest"));
		s.endArray();
	}

	QSettings     s(path, QSettings::IniFormat);
	Configuration global;
	s.beginGroup(QStringLiteral("GlobalConfiguration"));
	global.ReadSettings(&s);
	s.endGroup();
	Check(global.gpu_occlusion_accurate, "global occlusion value not read");
	Check(global.shader_optimization_type == Configuration::ShaderOptimizationType::Performance,
	      "global optimization type");

	Configuration a;
	ReadGame(s, 0, global, a);
	Check(a.gpu_occlusion_accurate, "old game config did not inherit the global occlusion value");
	Check(a.shader_optimization_type == Configuration::ShaderOptimizationType::Performance,
	      "empty optimization type did not fall back to Performance");
	Configuration b;
	ReadGame(s, 1, global, b);
	Check(!b.gpu_occlusion_accurate, "game occlusion value overridden by the global one");
	Check(b.shader_optimization_type == Configuration::ShaderOptimizationType::Size,
	      "Size not read");
	Configuration c;
	ReadGame(s, 2, global, c);
	Check(c.shader_optimization_type == Configuration::ShaderOptimizationType::Performance,
	      "unknown optimization type did not fall back to Performance");

	// Saved and copied with the other game settings.
	Check(b.GameSettings().value(QStringLiteral("gpu_occlusion_accurate")).toString() ==
	          QStringLiteral("false"),
	      "occlusion value missing from GameSettings");
	Configuration copy;
	copy.CopyEmulatorSettingsFrom(a);
	Check(copy.gpu_occlusion_accurate, "CopyEmulatorSettingsFrom dropped the occlusion value");

	{
		QSettings out(dir.filePath(QStringLiteral("out.ini")), QSettings::IniFormat);
		a.WriteSettings(&out);
		out.sync();
		Configuration back;
		back.ReadSettings(&out);
		Check(back.gpu_occlusion_accurate, "occlusion value not written per game");
	}

	// Game config import accepts the key and validates it.
	Configuration imported;
	QString       error;
	Check(imported.SetGameSettings(QJsonObject {{QStringLiteral("gpu_occlusion_accurate"),
	                                             QStringLiteral("true")}},
	                               error),
	      "import of gpu_occlusion_accurate failed");
	Check(imported.gpu_occlusion_accurate, "imported occlusion value not applied");
	Configuration invalid;
	Check(!invalid.SetGameSettings(QJsonObject {{QStringLiteral("shader_optimization_type"),
	                                             QStringLiteral("Fastest")}},
	                               error),
	      "invalid optimization type accepted on import");

	// "Faster GPU synchronisation": off by default and in configs saved before it existed.
	Check(!Configuration().fast_gpu_sync && !a.fast_gpu_sync && !b.fast_gpu_sync, "fast GPU sync is off by default");
	Check(b.GameSettings().value(QStringLiteral("fast_gpu_sync")).toString() == QStringLiteral("false"),
	      "fast GPU sync missing from GameSettings");
	{
		Configuration on;
		on.fast_gpu_sync = true;
		QSettings out(dir.filePath(QStringLiteral("fast.ini")), QSettings::IniFormat);
		on.WriteSettings(&out);
		out.sync();
		Check(out.value(QStringLiteral("fast_gpu_sync")).toString() == QStringLiteral("true"), "fast GPU sync not written");
		Configuration back;
		back.ReadSettings(&out);
		Check(back.fast_gpu_sync, "fast GPU sync not read back");
		Configuration copied;
		copied.CopyEmulatorSettingsFrom(on);
		Check(copied.fast_gpu_sync, "CopyEmulatorSettingsFrom dropped fast GPU sync");
	}
	Configuration fast_imported;
	Check(fast_imported.SetGameSettings(QJsonObject {{QStringLiteral("fast_gpu_sync"), QStringLiteral("true")}}, error) &&
	          fast_imported.fast_gpu_sync,
	      "import of fast_gpu_sync failed");
	Configuration fast_invalid;
	Check(!fast_invalid.SetGameSettings(QJsonObject {{QStringLiteral("fast_gpu_sync"), QStringLiteral("maybe")}}, error),
	      "invalid fast_gpu_sync accepted on import");
	{
		const auto& variables = Configuration::FastGpuSyncVariables();
		Check(variables.size() == 4, "four switches");
		const auto value = [&](const char* name) {
			for (const auto& [key, v]: variables) {
				if (key == QLatin1String(name)) {
					return v;
				}
			}
			return QString();
		};
		Check(value("KYTY_GPU_HAZARD") == QStringLiteral("3") && value("KYTY_INDIRECT_ARGS_RANGES") == QStringLiteral("1") &&
		          value("KYTY_GUEST_BARRIER_SCOPE") == QStringLiteral("rel=drop") &&
		          value("KYTY_DMA_BARRIER_BATCH") == QStringLiteral("1"),
		      "the switches and their values");

		Configuration     off;
		QProcessEnvironment environment;
		environment.insert(QStringLiteral("KYTY_OTHER"), QStringLiteral("x"));
		off.ApplyFastGpuSync(environment);
		Check(environment.keys().size() == 1, "off: the environment is unchanged");

		Configuration on;
		on.fast_gpu_sync = true;
		on.ApplyFastGpuSync(environment);
		for (const auto& [key, v]: variables) {
			Check(environment.value(key) == v, "on: every switch is set");
		}
		// A preset (or the user) that sets one keeps its value; the others are added.
		QProcessEnvironment preset;
		preset.insert(QStringLiteral("KYTY_GPU_HAZARD"), QStringLiteral("0"));
		preset.insert(QStringLiteral("KYTY_GUEST_BARRIER_SCOPE"), QString());
		on.ApplyFastGpuSync(preset);
		Check(preset.value(QStringLiteral("KYTY_GPU_HAZARD")) == QStringLiteral("0"), "a preset's KYTY_GPU_HAZARD wins");
		Check(preset.contains(QStringLiteral("KYTY_GUEST_BARRIER_SCOPE")) &&
		          preset.value(QStringLiteral("KYTY_GUEST_BARRIER_SCOPE")).isEmpty(),
		      "an empty value set by a preset wins too");
		Check(preset.value(QStringLiteral("KYTY_INDIRECT_ARGS_RANGES")) == QStringLiteral("1") &&
		          preset.value(QStringLiteral("KYTY_DMA_BARRIER_BATCH")) == QStringLiteral("1"),
		      "the switches the preset leaves alone are added");
	}

	Check(Configuration::ShaderOptimizationTypeFromText(QStringLiteral("None")) ==
	          Configuration::ShaderOptimizationType::None,
	      "None not parsed");
	Check(Configuration::ShaderOptimizationTypeFromText(QString()) ==
	          Configuration::ShaderOptimizationType::Performance,
	      "empty text not Performance");

	// The default window size, also for a saved value that cannot be read; a saved size is kept.
	{
		Check(Configuration().screen_resolution == QStringLiteral("1920x1080"), "the default window is 1920x1080");
		QSettings out(dir.filePath(QStringLiteral("res.ini")), QSettings::IniFormat);
		out.setValue(QStringLiteral("screen_resolution"), QStringLiteral("not a size"));
		Configuration broken;
		broken.ReadSettings(&out);
		Check(broken.screen_resolution == QStringLiteral("1920x1080"), "an unreadable size becomes the default");
		out.setValue(QStringLiteral("screen_resolution"), QStringLiteral("1280x720"));
		Configuration saved;
		saved.ReadSettings(&out);
		Check(saved.screen_resolution == QStringLiteral("1280x720"), "a saved size is kept");
	}

	// Recommended settings: the launcher settings a title was tested with.
	{
		const char* const listed[] = {"PPSA01317", "PPSA01340", "PPSA01341", "PPSA01342", "PPSA01417",
		                              "PPSA01474", "PPSA03671", "PPSA08705", "PPSA26344"};
		for (const char* title: listed) {
			const auto recommended = Configuration::RecommendedGameSettings(QString::fromLatin1(title));
			Check(!recommended.isEmpty(), "a listed title has recommended settings");
			// Every entry is a setting of a game config export with a valid value (a setting this
			// platform lacks is left out, as ApplyRecommendedGameSettings does).
			Configuration probe;
			QJsonObject   known;
			for (auto it = recommended.constBegin(); it != recommended.constEnd(); ++it) {
				Check(it.value().isString(), "recommended values are texts");
				if (probe.GameSettings().contains(it.key())) {
					known.insert(it.key(), it.value());
				}
			}
			Check(probe.SetGameSettings(known, error), "recommended settings are valid game settings");
			Check(probe.readback_linear_images && probe.tessellation_enabled,
			      "the tested base: linear readback and tessellation");
		}
		// Astro Bot and Astro's Playroom were measured with the default settings: no entry.
		const char* const unlisted[] = {"PPSA21567", "PPSA21564", "PPSA01325", "", "PPSA99999"};
		for (const char* title: unlisted) {
			Check(Configuration::RecommendedGameSettings(QString::fromLatin1(title)).isEmpty(),
			      "no entry for a title that runs with the default settings");
		}
		Check(!Configuration::RecommendedGameSettings(QStringLiteral(" ppsa01474 ")).isEmpty(),
		      "the title id is trimmed and upper-cased");

		// Over a user's settings: the recommended ones are set, every other one keeps its value.
		Configuration user;
		user.screen_resolution        = QStringLiteral("2560x1440");
		user.vblank_frequency         = 120;
		user.user_id                  = 7;
		user.present_mode             = Configuration::PresentMode::Fifo;
		user.printf_direction         = Configuration::LogDirection::File;
		user.fast_gpu_sync            = true;
		user.amd_cpu_mode             = QStringLiteral("off");
		user.shader_optimization_type = Configuration::ShaderOptimizationType::Size;
		user.trophy_duration_seconds  = 9;
		user.host_input_mapping       = QStringList {QStringLiteral("a=b")};
		user.elf                      = QStringLiteral("other.bin");
		const auto before = user.GameSettings();
		Check(user.ApplyRecommendedGameSettings(QStringLiteral("PPSA01474")), "the title has an entry");
		Check(user.readback_linear_images && user.readback_tiled_images && user.tessellation_enabled &&
		          user.gpu_occlusion_accurate,
		      "the title's settings are set");
		Check(!user.readback_storage_images, "a setting outside the entry stays off");
		auto after = user.GameSettings();
		for (const auto& key: Configuration::RecommendedGameSettings(QStringLiteral("PPSA01474")).keys()) {
			after.insert(key, before.value(key));
		}
		Check(after == before, "every other setting keeps its value");
		Check(user.screen_resolution == QStringLiteral("2560x1440") && user.vblank_frequency == 120 &&
		          user.user_id == 7 && user.present_mode == Configuration::PresentMode::Fifo &&
		          user.elf == QStringLiteral("other.bin"),
		      "the user's own values are kept");
		Check(user.host_input_mapping == QStringList {QStringLiteral("a=b")}, "the input mapping is kept");

		Configuration plain;
		const auto    plain_before = plain.GameSettings();
		Check(!plain.ApplyRecommendedGameSettings(QStringLiteral("PPSA21567")) &&
		          plain.GameSettings() == plain_before,
		      "a title without an entry changes nothing");
		Configuration other;
		Check(other.ApplyRecommendedGameSettings(QStringLiteral("PPSA01317")) &&
		          !other.readback_tiled_images && !other.gpu_occlusion_accurate,
		      "a title gets its own entry only");
#if defined(_WIN32)
		Configuration zone;
		Check(zone.ApplyRecommendedGameSettings(QStringLiteral("PPSA03671")) &&
		          zone.red_zone_protection_enabled,
		      "red zone protection is part of the entry on Windows");
#endif
	}

	std::puts("LauncherConfigurationTests: OK");
	return 0;
}
