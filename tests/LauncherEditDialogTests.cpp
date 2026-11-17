// The game configuration dialog without showing it: Reset and the Recommended button of a title with
// tested launcher settings (Configuration::RecommendedGameSettings), and what Save writes.
#include "configuration.h"
#include "configurationEditDialog.h"

#include <QApplication>
#include <QCheckBox>
#include <QPushButton>
#include <QSpinBox>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "LauncherEditDialogTests: failed: %s\n", text);
		std::abort();
	}
}

template <class Widget>
Widget* Find(QDialog& dialog, const char* name) {
	auto* widget = dialog.findChild<Widget*>(QLatin1String(name));
	Check(widget != nullptr, name);
	return widget;
}

bool Checked(QDialog& dialog, const char* name) {
	return Find<QCheckBox>(dialog, name)->isChecked();
}

// The four settings of the Ratchet & Clank entry.
bool RecommendedSet(QDialog& dialog) {
	return Checked(dialog, "checkBox_readback") && Checked(dialog, "checkBox_readback_tiled") &&
	       Checked(dialog, "checkBox_tessellation") && Checked(dialog, "checkBox_gpu_occlusion");
}
} // namespace

int main(int argc, char** argv) {
#if !defined(_WIN32)
	// Nothing is shown. On Windows only the "windows" platform is deployed beside the launcher.
	qputenv("QT_QPA_PLATFORM", "offscreen");
#endif
	QApplication application(argc, argv);

	// A title with an entry, opened with a config that has none of its settings (one saved by an
	// earlier launcher).
	{
		Configuration info;
		info.title_id = QStringLiteral("PPSA01474");
		ConfigurationEditDialog dialog(info);
		Check(!Checked(dialog, "checkBox_readback") && !Checked(dialog, "checkBox_readback_tiled") &&
		          !Checked(dialog, "checkBox_tessellation") && !Checked(dialog, "checkBox_gpu_occlusion"),
		      "the dialog opens with the values it is given");

		auto* recommended = dialog.findChild<QPushButton*>(QStringLiteral("recommended_button"));
		Check(recommended != nullptr, "a title with an entry has the Recommended button");
		const auto tip = recommended->toolTip();
		Check(tip.contains(Find<QCheckBox>(dialog, "checkBox_readback")->text()) &&
		          tip.contains(Find<QCheckBox>(dialog, "checkBox_readback_tiled")->text()) &&
		          tip.contains(Find<QCheckBox>(dialog, "checkBox_tessellation")->text()) &&
		          tip.contains(Find<QCheckBox>(dialog, "checkBox_gpu_occlusion")->text()),
		      "the tooltip names the settings by their checkbox texts");
		Check(!tip.contains(QStringLiteral("readback_tiled_images")), "no raw setting names in the tooltip");

		// The user's own choices, then Recommended: its settings are set, the choices stay.
		auto* vblank  = Find<QSpinBox>(dialog, "spinBox_vblank_frequency");
		auto* storage = Find<QCheckBox>(dialog, "checkBox_readback_storage");
		auto* fast    = Find<QCheckBox>(dialog, "checkBox_fast_gpu_sync");
		vblank->setValue(120);
		storage->setChecked(true);
		fast->setChecked(true);
		recommended->click();
		Check(RecommendedSet(dialog), "Recommended sets the title's settings");
		Check(vblank->value() == 120 && storage->isChecked() && fast->isChecked(),
		      "Recommended keeps the other choices");

		// Reset: the built-in defaults with the title's settings on top.
		Find<QPushButton>(dialog, "clear_button")->click();
		Check(RecommendedSet(dialog), "Reset keeps the title's settings on");
		Check(vblank->value() == 60 && !storage->isChecked() && !fast->isChecked(),
		      "Reset returns the other settings to the defaults");

		// Save writes them to the game's configuration.
		Find<QPushButton>(dialog, "ok_button")->click();
		Check(dialog.result() == QDialog::Accepted, "Save accepts the dialog");
		Check(info.readback_linear_images && info.readback_tiled_images && info.tessellation_enabled &&
		          info.gpu_occlusion_accurate && !info.readback_storage_images && !info.fast_gpu_sync &&
		          info.vblank_frequency == 60,
		      "Save writes the dialog's values");
	}

	// A title without an entry: no button, Reset is the built-in defaults.
	{
		Configuration info;
		info.title_id               = QStringLiteral("PPSA21567");
		info.readback_linear_images = true;
		ConfigurationEditDialog dialog(info);
		Check(dialog.findChild<QPushButton*>(QStringLiteral("recommended_button")) == nullptr,
		      "a title without an entry has no Recommended button");
		Check(Checked(dialog, "checkBox_readback"), "the dialog opens with the values it is given");
		Find<QPushButton>(dialog, "clear_button")->click();
		Check(!Checked(dialog, "checkBox_readback") && !Checked(dialog, "checkBox_readback_tiled") &&
		          !Checked(dialog, "checkBox_tessellation") && !Checked(dialog, "checkBox_gpu_occlusion"),
		      "Reset of a title without an entry is the built-in defaults");
	}

	// The global settings dialog has no title: no button, Reset is the built-in defaults.
	{
		Configuration info;
		info.tessellation_enabled = true;
		ConfigurationEditDialog dialog(info);
		dialog.SetGlobalSettings({}, {});
		Check(dialog.findChild<QPushButton*>(QStringLiteral("recommended_button")) == nullptr,
		      "the global settings have no Recommended button");
		Find<QPushButton>(dialog, "clear_button")->click();
		Check(!Checked(dialog, "checkBox_tessellation") && !Checked(dialog, "checkBox_readback"),
		      "Reset of the global settings is the built-in defaults");
	}

	std::puts("LauncherEditDialogTests: OK");
	return 0;
}
