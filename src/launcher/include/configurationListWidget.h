#ifndef CONFIGURATION_LIST_WIDGET_H
#define CONFIGURATION_LIST_WIDGET_H

#include "configuration.h"

#include <QMap>
#include <QString>
#include <QStringList>
#include <QWidget>

#include <memory>

class ConfigurationItem;
class CompatibilityDatabase;
class ControllerPreview;
class QEvent;
class QTreeWidgetItem;

namespace Ui {
class ConfigurationListWidget;
} // namespace Ui

class ConfigurationListWidget: public QWidget {
	Q_OBJECT

public:
	explicit ConfigurationListWidget(QWidget* parent = nullptr);
	~ConfigurationListWidget() override;

	void SetRunEnabled(bool flag) { m_run_enabled = flag; }
	// The emulator's folder: trophies, and the shader lists of the Precompile buttons.
	void SetRuntimeDirectory(const QString& directory);
	// A precompile (of any game) started or ended: only one runs at a time.
	void SetPrecompileRunning(bool running);
	// After a game started or ended, or a shader list was installed.
	void UpdatePrecompileButtons();
	// As a click on the Precompile button of the game in this folder or archive (the launcher's
	// --precompile argument); false when the list has no such game.
	bool PressPrecompile(const QString& game);
	// Selects the row of the game in this folder or archive (the launcher's --export-list argument);
	// null when the list has no such game.
	ConfigurationItem* SelectGame(const QString& game);
	// Whether a row of this title runs (its folder and its archive are two rows).
	[[nodiscard]] bool IsTitleRunning(const QString& title_id) const;
	// As a right click on the row of the game in this folder or archive (the launcher's
	// --context-menu argument, for tests); false when the list has no such game.
	bool OpenContextMenu(const QString& game);
	void SetControllerPreview(ControllerPreview* preview) { m_controller_preview = preview; }

	[[nodiscard]] const ConfigurationItem* GetSelectedItem() const { return m_selected_item; }
	ConfigurationItem*                     GetSelectedItem() { return m_selected_item; }

	[[nodiscard]] const QString& GetSettingsFile() const { return m_settings_file; }
	[[nodiscard]] const QString& GetGlobalControllerColor() const {
		return m_global_info.controller.color;
	}
	[[nodiscard]] std::unique_ptr<Configuration>
	CreateConfiguration(const ConfigurationItem& item) const;

	bool EnsureGameDirectory();
	void ScanGameDirectory();
	void ViewTrophies();
	void ViewTrophyOverview();

signals:

	void Run();
	// The selected game's Precompile button (or menu entry) was used.
	void PrecompileShaders();
	// The selected game's Export list button (or menu entry) was used.
	void ExportShaderList();
	void Select();
	void PreviewControllerColor(const QString& color);

protected:
	void changeEvent(QEvent* event) override;

public slots:
	void WriteSettings();
	void ReadSettings();

protected slots:

	void edit_configuration();
	void delete_configuartion();
	void edit_global_settings();
	void edit_input_mapping();
	void edit_wheel();
	void list_itemDoubleClicked(QTreeWidgetItem* witem, int column);
	void open_game_folder();
	void remove_save_data();
	void compress_game();
	void choose_shader_list();
	void filter_configurations(const QString& text);

private:
	void               SelectItem(QTreeWidgetItem* witem);
	void               ShowContextMenu(QTreeWidgetItem* witem);
	void               ApplyCompatibility();
	void               UpdateToolbarIcons();
	void               ImportGameSettings(QWidget* parent);
	void               ExportGameSettings(QWidget* parent) const;
	[[nodiscard]] bool HasValidGameDirectory() const;

	ConfigurationItem*            m_selected_item = nullptr;
	bool                          m_run_enabled   = true;
	bool                          m_precompile_running = false;
	Ui::ConfigurationListWidget*  m_ui            = nullptr;
	QString                       m_settings_file;
	QString                       m_runtime_directory;
	QStringList                   m_game_dirs;
	QString                       m_addons_dir;
	QMap<QString, QStringList>    m_addcont; // User add-ons by title ID.
	Configuration                 m_global_info;
	QMap<QString, Configuration*> m_custom_infos;
	CompatibilityDatabase*        m_compatibility = nullptr;
	ControllerPreview*            m_controller_preview = nullptr;
};

#endif // CONFIGURATION_LIST_WIDGET_H
