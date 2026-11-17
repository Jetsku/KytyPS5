#include "configurationItem.h"

#include "configuration.h"
#include "gameContent.h"
#include "gameViews.h"
#include "precompile.h"
#include "shaderListExport.h"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <QFont>
#include <QFutureWatcher>
#include <QHBoxLayout>
#include <QIcon>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QPixmap>
#include <QPushButton>
#include <QSize>
#include <QStringList>
#include <QStyle>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QVersionNumber>
#include <QWheelEvent>
#include <QWidget>
#include <QtConcurrentRun>

namespace {

enum Column {
	NameColumn            = GAME_NAME_COLUMN,
	PrecompileColumn      = GAME_PRECOMPILE_COLUMN,
	SerialColumn          = GAME_SERIAL_COLUMN,
	GameVersionColumn     = GAME_VERSION_COLUMN,
	FirmwareVersionColumn = GAME_FIRMWARE_VERSION_COLUMN,
	SizeColumn            = GAME_SIZE_COLUMN,
	PathColumn            = GAME_PATH_COLUMN,
	StatusColumn          = GAME_STATUS_COLUMN,
	CommentsColumn        = GAME_COMMENT_COLUMN,
};

class NoWheelComboBox: public QComboBox {
public:
	explicit NoWheelComboBox(QWidget* parent = nullptr): QComboBox(parent) {}

protected:
	void wheelEvent(QWheelEvent* event) override { event->ignore(); }
};

QString GetStatusText(Configuration::GameStatus status) {
	switch (status) {
		case Configuration::GameStatus::Unknown: return QStringLiteral("Unknown");
		case Configuration::GameStatus::MainMenu: return QStringLiteral("Main menu");
		case Configuration::GameStatus::InGame: return QStringLiteral("In game");
		case Configuration::GameStatus::Logo: return QStringLiteral("Logo");
		case Configuration::GameStatus::DoesntBoot: return QStringLiteral("Doesn't boot");
	}

	return QStringLiteral("Unknown");
}

QString GetStatusColor(Configuration::GameStatus status) {
	switch (status) {
		case Configuration::GameStatus::InGame: return QStringLiteral("#2fb344");
		case Configuration::GameStatus::MainMenu: return QStringLiteral("#2f80ed");
		case Configuration::GameStatus::Logo: return QStringLiteral("#f2c94c");
		case Configuration::GameStatus::DoesntBoot: return QStringLiteral("#e55353");
		case Configuration::GameStatus::Unknown: return QStringLiteral("#8a8a8a");
	}

	return QStringLiteral("#8a8a8a");
}

void AddStatus(QComboBox* combo, Configuration::GameStatus status) {
	combo->addItem(GetStatusText(status), static_cast<int>(status));
}

void SetStatus(QComboBox* combo, Configuration::GameStatus status) {
	const int index = combo->findData(static_cast<int>(status));
	combo->setCurrentIndex(index >= 0 ? index : 0);
}

QString GetPathText(const Configuration& info) {
	return !info.game_path.isEmpty() ? info.game_path : info.basedir;
}

QString GetDisplayText(const Configuration& info) {
	QStringList lines({info.name});

	if (!info.title_id.isEmpty()) {
		lines.append(QStringLiteral("Serial ID: %1").arg(info.title_id));
	}
	if (!info.gameVersion.isEmpty()) {
		lines.append(QStringLiteral("Game version: %1").arg(info.gameVersion));
	}
	if (!info.firmwareVer.isEmpty()) {
		lines.append(QStringLiteral("Firmware version: %1").arg(info.firmwareVer));
	}

	const auto path = GetPathText(info);
	if (!path.isEmpty()) {
		lines.append(QStringLiteral("Path: %1").arg(path));
	}
	if (!info.game_comment.isEmpty()) {
		lines.append(QStringLiteral("Comment: %1").arg(info.game_comment));
	}

	return lines.join(QLatin1Char('\n'));
}

QString GetSortText(const Configuration& info) {
	return QStringLiteral("%1\n%2").arg(info.name.toCaseFolded(), info.game_path.toCaseFolded());
}

void MakeTransparent(QWidget* widget) {
	widget->setAutoFillBackground(false);
	widget->setAttribute(Qt::WA_NoSystemBackground);
	widget->setAttribute(Qt::WA_TranslucentBackground);
}

} // namespace

ConfigurationItem::ConfigurationItem(std::unique_ptr<Configuration> info, QTreeWidget* parent)
    : QTreeWidgetItem(parent), m_info(std::move(info)) {
	setSizeHint(NameColumn, QSize(0, 72));

	// Buttons of their natural height in the middle of the row, one above the other.
	m_precompile_widget = new QWidget(parent);
	MakeTransparent(m_precompile_widget);
	auto* precompile_layout = new QVBoxLayout(m_precompile_widget);
	precompile_layout->setContentsMargins(6, 0, 6, 0);
	precompile_layout->setSpacing(4);
	precompile_layout->addStretch(1);
	m_precompile_button = new QPushButton(QStringLiteral("Precompile"), m_precompile_widget);
	m_precompile_button->setObjectName(QStringLiteral("game_precompile_button"));
	m_precompile_button->setFocusPolicy(Qt::NoFocus);
	precompile_layout->addWidget(m_precompile_button);
	m_export_button = new QPushButton(QStringLiteral("Export list"), m_precompile_widget);
	m_export_button->setObjectName(QStringLiteral("game_export_list_button"));
	m_export_button->setFocusPolicy(Qt::NoFocus);
	m_export_button->setVisible(false);
	precompile_layout->addWidget(m_export_button);
	precompile_layout->addStretch(1);
	parent->setItemWidget(this, PrecompileColumn, m_precompile_widget);

	m_status_widget = new QWidget(parent);
	MakeTransparent(m_status_widget);
	auto* layout = new QHBoxLayout(m_status_widget);
	layout->setContentsMargins(4, 0, 4, 0);
	layout->setSpacing(6);

	m_status_indicator = new QLabel(m_status_widget);
	m_status_indicator->setObjectName(QStringLiteral("game_status_indicator"));
	m_status_indicator->setFixedSize(QSize(12, 12));
	layout->addWidget(m_status_indicator);

	m_status_combo = new NoWheelComboBox(m_status_widget);
	m_status_combo->setObjectName(QStringLiteral("game_status_editor"));
	MakeTransparent(m_status_combo);
	AddStatus(m_status_combo, Configuration::GameStatus::Unknown);
	AddStatus(m_status_combo, Configuration::GameStatus::MainMenu);
	AddStatus(m_status_combo, Configuration::GameStatus::InGame);
	AddStatus(m_status_combo, Configuration::GameStatus::Logo);
	AddStatus(m_status_combo, Configuration::GameStatus::DoesntBoot);
	m_status_combo->setSizeAdjustPolicy(QComboBox::AdjustToContents);
	m_status_combo->setFixedWidth(125);
	layout->addWidget(m_status_combo);
	layout->addStretch(1);

	parent->setItemWidget(this, StatusColumn, m_status_widget);

	m_comment_edit = new QLineEdit(parent);
	m_comment_edit->setObjectName(QStringLiteral("game_comment_editor"));
	MakeTransparent(m_comment_edit);
	m_comment_edit->setClearButtonEnabled(true);
	m_comment_edit->setFrame(false);
	parent->setItemWidget(this, CommentsColumn, m_comment_edit);

	Update(true);
	SetRunning(false);

	setData(SizeColumn, Qt::UserRole, qint64(-1));
	setText(SizeColumn, QStringLiteral("\u2014"));
	auto* watcher = new QFutureWatcher<qint64>(this);
	connect(watcher, &QFutureWatcher<qint64>::finished, this, [this, watcher]() {
		const auto bytes = watcher->result();
		setData(SizeColumn, Qt::UserRole, bytes);
		if (bytes >= 0) {
			setText(SizeColumn,
			        QLocale().formattedDataSize(bytes, 2, QLocale::DataSizeTraditionalFormat));
		}
		watcher->deleteLater();
	});
	watcher->setFuture(QtConcurrent::run([path = m_info->basedir]() -> qint64 {
		if (path.isEmpty()) {
			return -1;
		}
		const QFileInfo info(path);
		if (GameContent::IsArchive(path)) {
			return info.size();
		}
		if (!QDir(path).exists()) {
			return -1;
		}
		qint64       bytes = 0;
		QDirIterator files(path, QDir::Files | QDir::Hidden | QDir::System | QDir::NoSymLinks,
		                   QDirIterator::Subdirectories);
		while (files.hasNext()) {
			files.next();
			if (files.fileInfo().isFile()) {
				bytes += files.fileInfo().size();
			}
		}
		return bytes;
	}));
}

ConfigurationItem::~ConfigurationItem() = default;

void ConfigurationItem::Update(bool reload_icon) {
	const auto display_text = GetDisplayText(*m_info);
	const auto path         = GetPathText(*m_info);

	setText(NameColumn, m_info->name);
	setData(NameColumn, Qt::UserRole, m_info->pro_supported);
	setText(SerialColumn, m_info->title_id);
	setText(GameVersionColumn,
	        m_info->gameVersion.isEmpty() ? QStringLiteral("\u2014") : m_info->gameVersion);
	setText(FirmwareVersionColumn,
	        m_info->firmwareVer.isEmpty() ? QStringLiteral("\u2014") : m_info->firmwareVer);
	setText(PathColumn, path);
	setText(PrecompileColumn, {});
	setText(StatusColumn, {});
	setText(CommentsColumn, {});
	for (int column = NameColumn; column <= CommentsColumn; column++) {
		setToolTip(column, display_text);
	}
	SetStatus(m_status_combo, m_info->game_status);
	if (m_comment_edit->text() != m_info->game_comment) {
		m_comment_edit->setText(m_info->game_comment);
	}

	if (reload_icon) {
		const auto icon_data = GameContent::ReadFile(
		    m_info->basedir, QStringLiteral("sce_sys/icon0.png"), GameContent::MaxImageSize);
		QPixmap icon;
		m_icon = !icon_data.isEmpty() && icon.loadFromData(icon_data) ? QIcon(icon) : QIcon {};
	}
	UpdateIcon();
	UpdateStatusIndicator();
}

bool ConfigurationItem::operator<(const QTreeWidgetItem& other) const {
	const auto* other_item = dynamic_cast<const ConfigurationItem*>(&other);
	if (other_item == nullptr) {
		return QTreeWidgetItem::operator<(other);
	}

	const int column = treeWidget() != nullptr ? treeWidget()->sortColumn() : NameColumn;
	switch (column) {
		case NameColumn:
		case PrecompileColumn: // every row holds the same button
			return GetSortText(*m_info) < GetSortText(*other_item->m_info);
		case SizeColumn:
			return data(SizeColumn, Qt::UserRole).toLongLong() <
			       other.data(SizeColumn, Qt::UserRole).toLongLong();
		case StatusColumn:
			return GetStatusText(m_info->game_status) <
			       GetStatusText(other_item->m_info->game_status);
		case GameVersionColumn:
		case FirmwareVersionColumn: {
			const auto& version =
			    column == GameVersionColumn ? m_info->gameVersion : m_info->firmwareVer;
			const auto& other_version = column == GameVersionColumn
			                                ? other_item->m_info->gameVersion
			                                : other_item->m_info->firmwareVer;
			if (version.isEmpty() || other_version.isEmpty()) {
				return version.isEmpty() && !other_version.isEmpty();
			}
			return QVersionNumber::compare(QVersionNumber::fromString(version),
			                               QVersionNumber::fromString(other_version)) < 0;
		}
		case CommentsColumn:
			return m_info->game_comment.toCaseFolded() <
			       other_item->m_info->game_comment.toCaseFolded();
		default: return text(column).toCaseFolded() < other.text(column).toCaseFolded();
	}
}

void ConfigurationItem::SetRunning(bool state) {
	m_running = state;

	QFont f = font(NameColumn);
	f.setBold(state);
	for (int column = NameColumn; column <= CommentsColumn; column++) {
		setFont(column, f);
	}

	UpdateIcon();
	UpdatePrecompileButton();
}

void ConfigurationItem::SetPrecompileState(const QString& emulator_dir, bool blocked, bool can_export) {
	m_emulator_dir       = emulator_dir;
	m_precompile_blocked = blocked;
	m_can_export         = can_export;
	UpdatePrecompileButton();
}

// Off while this game or a precompile runs; the tooltip names the shader list the emulator will
// load for the game, or says that none is installed.
void ConfigurationItem::UpdatePrecompileButton() {
	const auto& title   = m_info->title_id;
	const auto& version = m_info->gameVersion;
	const bool  game    = !m_info->basedir.isEmpty() && QFileInfo::exists(m_info->basedir);
	const bool  valid   = !m_emulator_dir.isEmpty() && Precompile::IsValidGame(title, version);
	m_precompile_button->setEnabled(game && valid && !m_running && !m_precompile_blocked);

	QString tip;
	if (m_emulator_dir.isEmpty()) {
		// The emulator has not been looked for yet.
	} else if (!Precompile::IsValidGame(title, version)) {
		tip = QStringLiteral("This game has no title ID, so it cannot have a shader list.");
	} else if (const auto list = Precompile::FindList(m_emulator_dir, title, version);
	           !list.isEmpty()) {
		tip = QStringLiteral("Builds this game's shaders and pipelines before you play.\n"
		                     "Shader list: %1")
		          .arg(QDir::toNativeSeparators(list));
	} else {
		tip = QStringLiteral("No shader list is installed for this game (%1).\n"
		                     "Click to choose a .kyplst file.")
		          .arg(QDir::toNativeSeparators(
		              Precompile::InstallPath(m_emulator_dir, title, version)));
	}
	m_precompile_button->setToolTip(tip);

	// Export list: off while this game runs or is precompiled (the journals are being written).
	// Without recorded journals it stays on: a click says how to record them.
	m_export_button->setVisible(m_can_export);
	m_export_button->setEnabled(m_can_export && game && valid && !m_running && !m_precompile_blocked);
	QString export_tip;
	if (m_emulator_dir.isEmpty() || !m_can_export) {
		// Not looked for yet, or no kyty_pipeline_list.
	} else if (!Precompile::IsValidGame(title, version)) {
		export_tip = QStringLiteral("This game has no title ID, so it cannot have a shader list.");
	} else if (ShaderListExport::FindJournals(m_emulator_dir, title).IsEmpty()) {
		export_tip = QStringLiteral("Nothing has been recorded for this game yet: play it first.\n"
		                            "Kyty records the shaders a game uses while you play.");
	} else {
		export_tip = QStringLiteral("Makes a shader list of what your sessions of this game recorded,\n"
		                            "to send to other players: %1")
		                 .arg(QDir::toNativeSeparators(ShaderListExport::DefaultOutput(m_emulator_dir, title, version)));
	}
	m_export_button->setToolTip(export_tip);
}

void ConfigurationItem::SetCompatibilityEditable(bool editable) {
	m_status_combo->setEnabled(editable);
	m_comment_edit->setReadOnly(!editable);
	m_comment_edit->setClearButtonEnabled(editable);
}

void ConfigurationItem::UpdateIcon() {
	if (!m_icon.isNull()) {
		setIcon(NameColumn, m_icon);
		return;
	}

	if (m_running) {
		setIcon(NameColumn, QApplication::style()->standardIcon(QStyle::SP_MediaPlay));
	} else if (m_info->custom_settings) {
		setIcon(NameColumn, QApplication::style()->standardIcon(QStyle::SP_FileIcon));
	} else {
		setIcon(NameColumn, QApplication::style()->standardIcon(QStyle::SP_ComputerIcon));
	}
}

void ConfigurationItem::UpdateStatusIndicator() {
	m_status_indicator->setStyleSheet(
	    QStringLiteral("background-color: %1;").arg(GetStatusColor(m_info->game_status)));
	m_status_indicator->setToolTip(GetStatusText(m_info->game_status));
}
