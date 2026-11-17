#ifndef CONFIGURATION_ITEM_H
#define CONFIGURATION_ITEM_H

#include <QIcon>
#include <QObject>
#include <QString>
#include <QTreeWidgetItem>

#include <memory>

class QComboBox;
class QLabel;
class QLineEdit;
class QPushButton;
class QTreeWidget;
class QWidget;

class Configuration;
class ConfigurationItem: public QObject, public QTreeWidgetItem {
public:
	explicit ConfigurationItem(std::unique_ptr<Configuration> info, QTreeWidget* parent);
	~ConfigurationItem() override;

	void Update(bool reload_icon = false);
	bool operator<(const QTreeWidgetItem& other) const override;

	Configuration&                     GetInfo() { return *m_info; }
	[[nodiscard]] const Configuration& GetInfo() const { return *m_info; }
	// void                                     SetInfo(Configuration* info);
	// Also set while the game's shaders are precompiled: that blocks Run like a running game.
	void               SetRunning(bool state);
	void               SetCompatibilityEditable(bool editable);
	[[nodiscard]] bool IsRunning() const { return m_running; }
	QComboBox*         GetStatusCombo() { return m_status_combo; }
	QLineEdit*         GetCommentEdit() { return m_comment_edit; }
	QPushButton*       GetPrecompileButton() { return m_precompile_button; }
	// Below it: exports the shader list of what this game's sessions recorded (shaderListExport.h).
	QPushButton*       GetExportListButton() { return m_export_button; }
	// The emulator's folder (its PipelineLists holds the shader lists, its _PipelineCache the
	// journals) and whether a precompile of any game, or this game from another row, runs: the
	// buttons' tooltips and whether they can be clicked. `can_export`: kyty_pipeline_list is there
	// (without it the Export list button is hidden).
	void               SetPrecompileState(const QString& emulator_dir, bool blocked, bool can_export = false);

private:
	void UpdateIcon();
	void UpdateStatusIndicator();
	void UpdatePrecompileButton();

	std::unique_ptr<Configuration> m_info;
	QIcon                          m_icon;
	bool                           m_running            = false;
	bool                           m_precompile_blocked = false;
	QString                        m_emulator_dir;
	QWidget*                       m_precompile_widget = nullptr;
	QPushButton*                   m_precompile_button = nullptr;
	QPushButton*                   m_export_button     = nullptr;
	bool                           m_can_export        = false;
	QComboBox*                     m_status_combo      = nullptr;
	QLabel*                        m_status_indicator  = nullptr;
	QWidget*                       m_status_widget     = nullptr;
	QLineEdit*                     m_comment_edit      = nullptr;
};

#endif // CONFIGURATION_ITEM_H
