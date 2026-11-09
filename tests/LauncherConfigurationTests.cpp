// Launcher configuration: the per-game occlusion mode and the shader optimization fallback.
#include "configuration.h"

#include <QJsonObject>
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

	Check(Configuration::ShaderOptimizationTypeFromText(QStringLiteral("None")) ==
	          Configuration::ShaderOptimizationType::None,
	      "None not parsed");
	Check(Configuration::ShaderOptimizationTypeFromText(QString()) ==
	          Configuration::ShaderOptimizationType::Performance,
	      "empty text not Performance");

	std::puts("LauncherConfigurationTests: OK");
	return 0;
}
