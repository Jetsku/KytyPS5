// The game list's Precompile column: its place among the columns of the form, and the row button's
// state and tooltip (ConfigurationItem).
#include "configuration.h"
#include "configurationItem.h"
#include "gameViews.h"
#include "precompile.h"
#include "ui_configuration_list_widget.h"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QPushButton>
#include <QTemporaryDir>
#include <QThreadPool>

#include <cstdio>
#include <cstdlib>
#include <memory>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "LauncherPrecompileItemTests: failed: %s\n", text);
		std::abort();
	}
}

std::unique_ptr<Configuration> Game(const QString& folder, const QString& title, const QString& version) {
	auto info         = std::make_unique<Configuration>();
	info->name        = QStringLiteral("Test game");
	info->basedir     = folder;
	info->game_path   = folder;
	info->title_id    = title;
	info->gameVersion = version;
	return info;
}
} // namespace

int main(int argc, char** argv) {
#if !defined(_WIN32)
	// Nothing is shown. On Windows only the "windows" platform is deployed beside the launcher.
	qputenv("QT_QPA_PLATFORM", "offscreen");
#endif
	QApplication application(argc, argv);

	QWidget                     host;
	Ui::ConfigurationListWidget ui;
	ui.setupUi(&host);
	auto* list = ui.cfgs_list;

	// The constants name the form's columns: the new one sits between the game and its serial.
	const auto* header = list->headerItem();
	Check(list->columnCount() == 9, "the list has nine columns");
	Check(header->text(GAME_NAME_COLUMN) == QStringLiteral("Game") &&
	          header->text(GAME_PRECOMPILE_COLUMN) == QStringLiteral("Precompile") &&
	          header->text(GAME_SERIAL_COLUMN) == QStringLiteral("Serial") &&
	          header->text(GAME_VERSION_COLUMN) == QStringLiteral("Game Version") &&
	          header->text(GAME_FIRMWARE_VERSION_COLUMN) == QStringLiteral("Firmware Version") &&
	          header->text(GAME_SIZE_COLUMN) == QStringLiteral("Size") &&
	          header->text(GAME_PATH_COLUMN) == QStringLiteral("Path") &&
	          header->text(GAME_STATUS_COLUMN) == QStringLiteral("Status") &&
	          header->text(GAME_COMMENT_COLUMN) == QStringLiteral("Comments"),
	      "the column constants follow the form");
	Check(GAME_PRECOMPILE_COLUMN == GAME_NAME_COLUMN + 1 && GAME_SERIAL_COLUMN == GAME_PRECOMPILE_COLUMN + 1,
	      "Precompile is the column after Game and before Serial");

	QTemporaryDir game_folder;
	QTemporaryDir emulator;
	Check(game_folder.isValid() && emulator.isValid(), "cannot create temporary directories");
	const auto title   = QStringLiteral("PPSA00001");
	const auto version = QStringLiteral("01.000.000");

	auto* item   = new ConfigurationItem(Game(game_folder.path(), title, version), list);
	auto* button = item->GetPrecompileButton();
	Check(button != nullptr && button->text() == QStringLiteral("Precompile"), "the row has a Precompile button");
	Check(list->itemWidget(item, GAME_PRECOMPILE_COLUMN) != nullptr &&
	          list->itemWidget(item, GAME_PRECOMPILE_COLUMN)->isAncestorOf(button),
	      "the button is the Precompile column's widget");
	Check(item->text(GAME_NAME_COLUMN) == QStringLiteral("Test game") && item->text(GAME_SERIAL_COLUMN) == title &&
	          item->text(GAME_VERSION_COLUMN) == version && item->text(GAME_PATH_COLUMN) == game_folder.path() &&
	          item->text(GAME_PRECOMPILE_COLUMN).isEmpty(),
	      "the row's texts are in their columns");
	Check(list->itemWidget(item, GAME_STATUS_COLUMN) != nullptr && list->itemWidget(item, GAME_COMMENT_COLUMN) != nullptr,
	      "the status and comment editors kept their columns");

	// Before the emulator is found there is nothing to run.
	Check(!button->isEnabled() && button->toolTip().isEmpty(), "no emulator folder: the button is off");

	const auto expected = Precompile::InstallPath(emulator.path(), title, version);
	item->SetPrecompileState(emulator.path(), false);
	Check(button->isEnabled(), "a game without a list can be clicked (to choose one)");
	Check(button->toolTip().contains(QStringLiteral("No shader list is installed")) &&
	          button->toolTip().contains(QDir::toNativeSeparators(expected)),
	      "the tooltip says that no list is installed and where it would go");

	Check(QDir().mkpath(QFileInfo(expected).absolutePath()), "cannot create PipelineLists");
	QFile list_file(expected);
	Check(list_file.open(QIODevice::WriteOnly) && list_file.write("KYPLST1", 8) == 8, "cannot write a list");
	list_file.close();
	item->SetPrecompileState(emulator.path(), false);
	Check(button->isEnabled() && button->toolTip().contains(QStringLiteral("Shader list: ")) &&
	          button->toolTip().contains(QDir::toNativeSeparators(expected)),
	      "the tooltip names the installed list");

	// Off while this game runs (or is being precompiled: the launcher sets the same flag).
	item->SetRunning(true);
	Check(!button->isEnabled() && item->IsRunning(), "a running game's button is off");
	item->SetRunning(false);
	Check(button->isEnabled(), "the button comes back when the game ends");
	// Off while another precompile runs.
	item->SetPrecompileState(emulator.path(), true);
	Check(!button->isEnabled(), "a precompile elsewhere turns the button off");
	item->SetPrecompileState(emulator.path(), false);
	Check(button->isEnabled(), "and its end turns it on again");

	// The Export list button below Precompile: hidden without kyty_pipeline_list, on while the game
	// does not run, its tooltip says whether anything was recorded.
	auto* export_button = item->GetExportListButton();
	Check(export_button != nullptr && export_button->text() == QStringLiteral("Export list") &&
	          list->itemWidget(item, GAME_PRECOMPILE_COLUMN)->isAncestorOf(export_button),
	      "the row has an Export list button in the Precompile column");
	Check(export_button->isHidden(), "no tool: the button is hidden");
	item->SetPrecompileState(emulator.path(), false, true);
	Check(!export_button->isHidden() && export_button->isEnabled(), "with the tool the button is shown and on");
	Check(export_button->toolTip().contains(QStringLiteral("Nothing has been recorded")), "nothing recorded yet");
	const auto cache = QDir(emulator.path()).filePath(QStringLiteral("_PipelineCache"));
	Check(QDir().mkpath(cache), "cannot create _PipelineCache");
	for (const auto* name: {"PPSA00001.shaders.journal", "PPSA00001.pipelines.journal"}) {
		QFile journal(QDir(cache).filePath(QLatin1String(name)));
		Check(journal.open(QIODevice::WriteOnly) && journal.write("journal", 7) == 7, "cannot write a journal");
	}
	item->SetPrecompileState(emulator.path(), false, true);
	Check(export_button->toolTip().contains(QStringLiteral("PPSA00001-01.000.000-shaderlist.kyplst")),
	      "with journals the tooltip names the file the list goes to");
	item->SetRunning(true);
	Check(!export_button->isEnabled(), "off while the game runs");
	item->SetRunning(false);
	item->SetPrecompileState(emulator.path(), true, true);
	Check(!export_button->isEnabled(), "off while a precompile runs");
	item->SetPrecompileState(emulator.path(), false, true);
	Check(export_button->isEnabled(), "on again");

	auto* untitled = new ConfigurationItem(Game(game_folder.path(), {}, version), list);
	untitled->SetPrecompileState(emulator.path(), false);
	Check(!untitled->GetPrecompileButton()->isEnabled() &&
	          untitled->GetPrecompileButton()->toolTip().contains(QStringLiteral("no title ID")),
	      "a game without a title ID cannot have a list");

	auto* missing = new ConfigurationItem(Game(game_folder.filePath(QStringLiteral("gone")), title, version), list);
	missing->SetPrecompileState(emulator.path(), false);
	Check(!missing->GetPrecompileButton()->isEnabled(), "a game whose folder is gone cannot be precompiled");

	// Sorting by the button column orders by name, as the Game column does.
	item->GetInfo().name     = QStringLiteral("B");
	untitled->GetInfo().name = QStringLiteral("A");
	item->Update();
	untitled->Update();
	list->sortItems(GAME_PRECOMPILE_COLUMN, Qt::AscendingOrder);
	Check(list->indexOfTopLevelItem(untitled) < list->indexOfTopLevelItem(item), "the Precompile column sorts by name");

	// The rows measure their folders on pool threads.
	QThreadPool::globalInstance()->waitForDone();
	QApplication::processEvents();
	std::printf("LauncherPrecompileItemTests: ok\n");
	return 0;
}
