#include "mainDialog.h"

#include "cheatFile.h"
#include "configuration.h"
#include "configurationItem.h"
#include "configurationListWidget.h"
#include "controllerPreview.h"
#include "gameContent.h"
#include "precompile.h"
#include "precompileDialog.h"
#include "shaderListExport.h"
#include "titleSound.h"
#include "updateChecker.h"

#include "common/u59Preset.h"

#include <QApplication>
#include <QByteArray>
#include <QCheckBox>
#include <QComboBox>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QLabel>
#include <QMessageBox>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QPushButton>
#include <QUrl>
#include <QRadioButton>
#include <QRegularExpression>
#include <QSettings>
#include <QStringList>
#include <QStyleHints>
#include <QVariant>
#include <QtCore>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "ui_main_dialog.h"

#if defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
#endif

// IWYU pragma: no_include <minwindef.h>
// IWYU pragma: no_include <processthreadsapi.h>
// IWYU pragma: no_include <winbase.h>

class QWidget;

#if defined(_WIN32)
constexpr char  EMULATOR_EXE[] = "kyty_emulator.exe";
constexpr char  CMD_EXE[]      = "cmd.exe";
constexpr DWORD CMD_X_CHARS    = 175;
constexpr DWORD CMD_Y_CHARS    = 1000;
#else
constexpr char EMULATOR_EXE[] = "kyty_emulator";
#endif
constexpr char SETTINGS_MAIN_DIALOG[]        = "MainDialog";
constexpr char SETTINGS_MAIN_LAST_GEOMETRY[] = "geometry";
constexpr char SETTINGS_CHECK_UPDATES[]      = "check_updates_on_startup";
constexpr char SETTINGS_CLOSE_TERMINAL[]     = "close_terminal";
constexpr char SETTINGS_THEME[]              = "theme";

class MainDialogPrivate: public QObject {
	Q_OBJECT

public:
	explicit MainDialogPrivate(QObject* parent = nullptr): QObject(parent) {}
	~MainDialogPrivate() override;

	void Setup(MainDialog* main_dialog);

	/*slots:*/

	void Update();
	void FindInterpreter();
	void Run();
	void PrecompileShaders();
	void ExportShaderList();
	void SetPresetWarning(const QString& text);

	[[nodiscard]] const QString& GetInterpreter() const { return m_interpreter; }

	static void WriteSettings(QSettings& s);
	static void ReadSettings(QSettings& s);

private:
	static QByteArray g_last_geometry;
	static bool       g_check_updates_on_startup;
	static bool       g_close_terminal;
	static int        g_theme;

	static void ApplyTheme(int theme) {
		auto scheme = Qt::ColorScheme::Unknown;
		switch (theme) {
			case 1: scheme = Qt::ColorScheme::Dark; break;
			case 2: scheme = Qt::ColorScheme::Light; break;
			default: break;
		}
		QApplication::styleHints()->setColorScheme(scheme);
	}

	Ui::MainDialog* m_ui             = {nullptr};
	MainDialog*     m_main_dialog    = nullptr;
	UpdateChecker*  m_update_checker = nullptr;
	ControllerPreview m_controller_preview;
	TitleSound      m_title_sound;
	QString         m_interpreter;
	QString         m_preset_warning;
	QLabel*         m_preset_label   = nullptr;
	bool            m_started        = false;
	// The window of the last precompile, until it is closed.
	QPointer<PrecompileDialog> m_precompile;
	bool                       m_precompiling = false;
	QString                    m_precompile_title; // its game: no row of it runs meanwhile
	bool                       m_exporting = false;
	QString                    m_export_out; // --export-list-out: the list's file instead of the default

	// Running games and a running precompile.
	int m_running_games = 0;
};

QByteArray MainDialogPrivate::g_last_geometry;
bool       MainDialogPrivate::g_check_updates_on_startup = true;
bool       MainDialogPrivate::g_close_terminal           = false;
int        MainDialogPrivate::g_theme                    = 0;

MainDialog::MainDialog(QWidget* parent): QDialog(parent), m_p(new MainDialogPrivate(this)) {
	m_p->Setup(this);
}

MainDialogPrivate::~MainDialogPrivate() {
	delete m_ui;
}

void MainDialogPrivate::Setup(MainDialog* main_dialog) {
	m_ui = new Ui::MainDialog;
	m_ui->setupUi(main_dialog);
	m_ui->widget->SetControllerPreview(&m_controller_preview);

	m_main_dialog = main_dialog;
	m_update_checker = new UpdateChecker(main_dialog);
	m_ui->check_updates_on_startup->setChecked(g_check_updates_on_startup);
	m_ui->close_terminal->setChecked(g_close_terminal);
	m_ui->theme_combo->setCurrentIndex(g_theme);
	ApplyTheme(g_theme);
	m_ui->check_updates_link->setVisible(UpdateChecker::IsSupported());
	m_ui->check_updates_on_startup->setVisible(UpdateChecker::IsSupported());
#if defined(__APPLE__)
	m_ui->close_terminal->setVisible(false); // The emulator runs without a terminal.
#endif

	main_dialog->setWindowFlags(Qt::Dialog | Qt::WindowMinMaxButtonsHint |
	                            Qt::WindowCloseButtonHint);

	connect(main_dialog, &MainDialog::Start, this, &MainDialogPrivate::FindInterpreter,
	        Qt::QueuedConnection);
	connect(m_ui->widget, &ConfigurationListWidget::Select, this, [this]() {
		Update();
		const auto* item = m_ui->widget->GetSelectedItem();
		m_title_sound.SetTitle(m_running_games == 0 && item != nullptr ? item->GetInfo().basedir
		                                                               : QString());
	});
	connect(m_ui->widget, &ConfigurationListWidget::PreviewControllerColor, this,
	        [this](const QString& color) {
		        if (m_running_games == 0) {
			        m_controller_preview.SetColor(color);
		        }
	        });
	connect(m_ui->widget, &ConfigurationListWidget::Run, this, &MainDialogPrivate::Run);
	connect(m_ui->widget, &ConfigurationListWidget::PrecompileShaders, this,
	        &MainDialogPrivate::PrecompileShaders);
	connect(m_ui->widget, &ConfigurationListWidget::ExportShaderList, this,
	        &MainDialogPrivate::ExportShaderList);
	connect(m_ui->check_updates_link, &QLabel::linkActivated, this,
	        [this](const QString&) { m_update_checker->Check(true); });
	connect(m_update_checker, &UpdateChecker::CheckingChanged, m_ui->check_updates_link,
	        &QLabel::setDisabled);
	connect(m_update_checker, &UpdateChecker::InstallingChanged, this, &MainDialogPrivate::Update);
	connect(m_ui->check_updates_on_startup, &QCheckBox::toggled, this, [this](bool checked) {
		g_check_updates_on_startup = checked;
		m_ui->widget->WriteSettings();
	});
	connect(m_ui->close_terminal, &QCheckBox::toggled, this, [this](bool checked) {
		g_close_terminal = checked;
		m_ui->widget->WriteSettings();
	});
	connect(m_ui->theme_combo, &QComboBox::currentIndexChanged, this, [this](int index) {
		g_theme = index;
		ApplyTheme(index);
		m_ui->widget->WriteSettings();
	});
	connect(main_dialog, &MainDialog::Resize, [this]() {
		g_last_geometry = m_main_dialog->saveGeometry();
		m_ui->widget->WriteSettings();
	});

	m_ui->label_settings_file->setText(tr("Settings file: ") + m_ui->widget->GetSettingsFile());

	m_main_dialog->restoreGeometry(g_last_geometry);

	Update();
}

void MainDialogPrivate::FindInterpreter() {
	UpdateChecker::ShowPreviousResult(m_main_dialog);
	QDir search_dir(QApplication::applicationDirPath());
	m_interpreter = search_dir.absoluteFilePath(EMULATOR_EXE);

	if (!QFile::exists(m_interpreter)) {
		search_dir.cdUp();
		m_interpreter = search_dir.absoluteFilePath(EMULATOR_EXE);
	}

	bool found = QFile::exists(m_interpreter);
	m_ui->widget->SetRuntimeDirectory(QFileInfo(m_interpreter).absolutePath());

	if (found) {
		m_ui->label_Interpreter->setText(tr("Emulator: ") + m_interpreter);

		QProcess test;
		test.setProgram(m_interpreter);
		test.start();
		test.waitForFinished();

		auto output = QString(test.readAllStandardOutput());
		auto lines  = output.split(QRegularExpression("[\r\n]"), Qt::SkipEmptyParts);

		if (lines.count() >= 2) {
			m_ui->label_Version->setText(
			    tr("Version: ") + (lines.at(0).startsWith("exe_name") ? lines.at(1) : lines.at(0)));
		} else {
			found = false;
		}
	}

	if (!found) {
		QMessageBox::critical(m_main_dialog, tr("Error"), tr("Can't find emulator"));
		QApplication::quit();
		return;
	}

	// Prompt for a game folder when none are configured, but keep the launcher
	// open if the user dismisses the dialog (quitting here can segfault during
	// nested modal shutdown / background compatibility load).
	m_ui->widget->EnsureGameDirectory();

	m_ui->label_settings_file->setText(tr("Settings file: ") + m_ui->widget->GetSettingsFile());

	m_started = true;
	if (!m_preset_warning.isEmpty()) {
		QMessageBox::warning(m_main_dialog, tr("Release settings not applied"), m_preset_warning);
	}

	Update();
	if (m_ui->check_updates_on_startup->isChecked()) {
		m_update_checker->Check(false);
	}

	// --precompile <game folder or archive> (scripts, tests): the launcher starts as if that game's
	// Precompile button had been clicked. Item widgets of the list are not reachable through UI
	// Automation, so nothing else can press the button without the mouse. With --precompile-list <file>
	// the list that game asks for is that file, as if it had been chosen in the file dialog.
	const auto arguments = QCoreApplication::arguments();
	if (const auto at = arguments.indexOf(QStringLiteral("--context-menu"));
	    at >= 0 && at + 1 < arguments.size()) {
		QMetaObject::invokeMethod(
		    this,
		    [this, game = arguments.at(at + 1)]() {
			    if (!m_ui->widget->OpenContextMenu(game)) {
				    qWarning().noquote() << "--context-menu: no game in the list at" << game;
			    }
		    },
		    Qt::QueuedConnection);
	}
	if (const auto at = arguments.indexOf(QStringLiteral("--precompile-list"));
	    at >= 0 && at + 1 < arguments.size()) {
		PrecompileDialog::SetTestList(arguments.at(at + 1));
	}
	if (const auto at = arguments.indexOf(QStringLiteral("--precompile"));
	    at >= 0 && at + 1 < arguments.size()) {
		QMetaObject::invokeMethod(
		    this,
		    [this, game = arguments.at(at + 1)]() {
			    if (!m_ui->widget->PressPrecompile(game)) {
				    qWarning().noquote() << "--precompile: no game in the list at" << game;
			    }
		    },
		    Qt::QueuedConnection);
	}
	// --export-list <game folder or archive> [--export-list-out <file>] (scripts, tests): as the
	// game's Export list button, also when the button is off (the export then says why it cannot
	// run); the result goes to the log as "export-list: ok: ..." or "export-list: failed: ...".
	// After --precompile when both are given.
	if (const auto at = arguments.indexOf(QStringLiteral("--export-list-out"));
	    at >= 0 && at + 1 < arguments.size()) {
		m_export_out = QFileInfo(arguments.at(at + 1)).absoluteFilePath();
	}
	if (const auto at = arguments.indexOf(QStringLiteral("--export-list"));
	    at >= 0 && at + 1 < arguments.size()) {
		QMetaObject::invokeMethod(
		    this,
		    [this, game = arguments.at(at + 1)]() {
			    if (m_ui->widget->SelectGame(game) == nullptr) {
				    qWarning().noquote() << "--export-list: no game in the list at" << game;
				    return;
			    }
			    ExportShaderList();
		    },
		    Qt::QueuedConnection);
	}
}

void MainDialogPrivate::SetPresetWarning(const QString& text) {
	m_preset_warning = text;
	if (m_preset_label == nullptr) {
		m_preset_label = new QLabel(m_main_dialog);
		m_preset_label->setWordWrap(true);
		m_preset_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
		m_preset_label->setStyleSheet(QStringLiteral("QLabel { color: #d03030; font-weight: bold; }"));
		m_ui->verticalLayout_4->insertWidget(0, m_preset_label);
	}
	m_preset_label->setText(text);
	m_preset_label->setVisible(!text.isEmpty());
	if (m_started && !text.isEmpty()) {
		QMessageBox::warning(m_main_dialog, tr("Release settings not applied"), text);
	}
}

void MainDialog::SetPresetWarning(const QString& text) {
	m_p->SetPresetWarning(text);
}

static QString BoolArg(bool value) {
	return value ? QStringLiteral("true") : QStringLiteral("false");
}

static void AddFlag(QStringList& args, bool enabled, const char* name) {
	if (enabled) {
		args << name;
	}
}

static QStringList CreateEmulatorArgs(const Configuration& info) {
	QStringList args;
	auto        r = Configuration::NormalizeResolution(info.screen_resolution).split('x');

	if (r.size() != 2) {
		return {};
	}

	args << "--screen-width" << r.at(0);
	args << "--screen-height" << r.at(1);
	args << "--user-name" << info.user_name;
	args << "--user-id" << QString::number(info.user_id);
	if (!info.audio_input_device.isEmpty()) {
		args << "--mic" << info.audio_input_device;
	}
	if (!info.controller.color.isEmpty()) {
		args << "--controller-color" << info.controller.color;
	}
	args << "--controller-volume" << QString::number(info.controller.speaker_volume);
	args << "--controller-vibration" << QString::number(info.controller.vibration_intensity);
	args << "--audio-master-volume" << QString::number(info.audio_mix.master);
	args << "--audio-main-volume" << QString::number(info.audio_mix.main);
	args << "--audio-music-volume" << QString::number(info.audio_mix.music);
	args << "--audio-pad-speaker-volume" << QString::number(info.audio_mix.pad_speaker);
	args << "--audio-objects-volume" << QString::number(info.audio_mix.objects);
	args << "--audio-objects" << (info.audio_mix.objects_enabled ? "on" : "off");
	if (info.gpu_occlusion_accurate) {
		// Overrides the bundled preset's KYTY_GPU_OCCLUSION (performance mode by default).
		args << "--gpu-occlusion" << "on";
	}
	args << "--keyboard-player" << QString::number(info.controller.keyboard_player);
	args << "--present-mode" << EnumToText(info.present_mode);
	if (info.gpu_index >= 0) {
		args << "--gpu" << QString::number(info.gpu_index);
	}
	AddFlag(args, info.fullscreen_enabled, "--fullscreen");
	AddFlag(args, info.hdr_enabled, "--hdr");
	AddFlag(args, info.hide_cursor_enabled, "--hide-cursor");
	AddFlag(args, info.stretch_enabled, "--stretch");
	args << "--readback-linear-images" << BoolArg(info.readback_linear_images);
	args << "--readback-tiled-images" << BoolArg(info.readback_tiled_images);
	args << "--readback-storage-images" << BoolArg(info.readback_storage_images);
	args << "--sync-raw-image-buffers" << BoolArg(info.sync_raw_image_buffers);
	args << "--flush-draws" << QString::number(info.flush_draws);
	args << "--trophy-notifications" << BoolArg(info.trophy_enabled);
	args << "--trophy-duration" << QString::number(info.trophy_duration_seconds);
	AddFlag(args, info.perf_overlay_enabled, "--perf-overlay");
	args << "--skip-notice-screen" << BoolArg(info.skip_notice_screen);
	args << "--network-connected" << BoolArg(info.network_connected);
	AddFlag(args, info.psn_emulation_enabled, "--emulate-psn");
	AddFlag(args, info.tessellation_enabled, "--tessellation");
	AddFlag(args, info.relaxed_shader_float, "--relaxed-shader-float");
	args << "--vblank-frequency" << QString::number(info.vblank_frequency);
	args << "--console-language" << QString::number(info.console_language);
	args << "--vulkan-validation" << BoolArg(info.vulkan_validation_enabled);
	args << "--shader-validation" << BoolArg(info.shader_validation_enabled);
	args << "--shader-optimization-type"
	     << EnumToText(Configuration::ShaderOptimizationTypeFromText(
	            EnumToText(info.shader_optimization_type)));
	args << "--shader-log-direction" << EnumToText(info.shader_log_direction);
	args << "--shader-log-folder" << info.shader_log_folder;
	if (qEnvironmentVariable("KYTY_CAPTURE_SHADER_DETAILS") == QStringLiteral("1")) {
		args << "--graphics-debug-dump" << "true";
	}
	args << "--command-buffer-dump" << BoolArg(info.command_buffer_dump_enabled);
	args << "--command-buffer-dump-folder" << info.command_buffer_dump_folder;
	args << "--printf-direction" << EnumToText(info.printf_direction);
	args << "--printf-output-file" << info.printf_output_file;
	if (info.profiler_enabled) {
		args << "--profile";
	}
	args << "--spirv-debug-printf" << "false";
	AddFlag(args, info.pro_enabled, "--trinity");
	if (info.pro_enabled && info.pssr_upscaler == Configuration::PssrUpscaler::DLSS) {
		args << "--dlss" << info.dlss_folder;
	}
	// Neither flag: the emulator decides from the CPU vendor.
	if (info.amd_cpu_mode == QStringLiteral("on")) {
		args << "--amd-cpu";
	} else if (info.amd_cpu_mode == QStringLiteral("off")) {
		args << "--no-amd-cpu";
	}
	AddFlag(args, info.guest_tsc_enabled, "--guest-tsc");
#if defined(_WIN32)
	AddFlag(args, info.red_zone_protection_enabled, "--redzone");
#endif
	for (const auto& binding: info.host_input_mapping) {
		args << "--keymap" << binding;
	}
	if (info.controller.wheel_enabled && !info.controller.wheel_config.isEmpty()) {
		args << "--wheel" << BoolArg(true);
		args << "--wheel-config" << info.controller.wheel_config;
	}
	AddFlag(args, info.renderdoc_enabled, "--rd");

	QString game = info.basedir;
	if (!info.elf.isEmpty() && !GameContent::IsArchive(info.basedir)) {
		game = QDir(info.basedir).filePath(info.elf);
	}
	args << "--game" << game;
	if (!info.addons_dir.isEmpty()) {
		args << "--addcont-dir" << info.addons_dir;
	}
	for (const auto& entry: info.addcont) {
		args << "--addcont" << entry;
	}

	const auto patch_plan = Cheats::PlanPath(info.title_id);
	if (QFileInfo::exists(patch_plan)) {
		args << "--game-patch" << patch_plan;
	}

	return args;
}

#ifdef __linux__
// Runs the command in bash, then waits for a key (with close_on_clean_exit, only after an error).
// The command goes in as positional parameters ("$@"), so it needs no quoting.
static QStringList BashCommand(const QString& interpreter, const QStringList& args,
                               bool close_on_clean_exit) {
	const QString run = close_on_clean_exit ? "\"$@\" && exit" : "\"$@\"";
	return QStringList {"bash", "-c", run + "; echo Press any key...; read -n1", "kyty",
	                    interpreter} +
	       args;
}

// Find a terminal and its command separator.
static bool FindTerminal(QString* program, QStringList* prefix) {
	struct TerminalSpec {
		const char* executable;
		const char* separator; // nullptr when the command follows immediately
	};

	static const TerminalSpec candidates[] = {
	    {"x-terminal-emulator", "-e"},
	    {"gnome-terminal", "--"},
	    {"konsole", "-e"},
	    {"xfce4-terminal", "-x"},
	    {"mate-terminal", "--"},
	    {"tilix", "-e"},
	    {"alacritty", "-e"},
	    {"kitty", nullptr},
	    {"foot", nullptr},
	    {"wezterm", "-e"},
	    {"urxvt", "-e"},
	    {"xterm", "-e"},
	};

	const auto try_candidate = [program, prefix](const QString& executable, const char* separator) {
		const auto resolved = QStandardPaths::findExecutable(executable);
		if (resolved.isEmpty()) {
			return false;
		}
		*program = resolved;
		prefix->clear();
		if (separator != nullptr) {
			*prefix << QString::fromLatin1(separator);
		}
		return true;
	};

	if (const auto from_env = qEnvironmentVariable("TERMINAL"); !from_env.isEmpty()) {
		// Reuse the known separator for an explicit terminal.
		const auto  env_name  = QFileInfo(from_env).fileName();
		const char* separator = "-e";
		for (const auto& candidate: candidates) {
			if (env_name == QLatin1String(candidate.executable)) {
				separator = candidate.separator;
				break;
			}
		}
		if (try_candidate(from_env, separator)) {
			return true;
		}
	}

	for (const auto& candidate: candidates) {
		if (try_candidate(QString::fromLatin1(candidate.executable), candidate.separator)) {
			return true;
		}
	}

	return false;
}
#endif

#if defined(_WIN32)
// cmd /K arguments that keep the console open (with close_on_clean_exit, only after an error).
// A quote is written as "" rather than \" so cmd's quote count stays even and operators inside
// arguments stay quoted.
static QString CmdCommand(const QString& interpreter, const QStringList& args,
                          bool close_on_clean_exit) {
	static const QRegularExpression quote_or_end(QStringLiteral(R"((\\*)("|\z))"));
	// The UCRT reads 2n backslashes before a quote as n.
	auto quoted = QStringList {interpreter} + args;
	quoted.replaceInStrings(quote_or_end, QStringLiteral(R"(\1\1\2\2)"));
	return QStringLiteral("/K \"\"") + quoted.join(QStringLiteral("\" \"")) +
	       (close_on_clean_exit ? "\" && exit\"" : "\"\"");
}
#endif

// u59-preset.<TITLE_ID>.json next to the launcher (or one folder up, like u59-preset.json): its
// variables go over the bundled preset in this game's process only. The emulator names the file
// (KYTY_TITLE_PRESET) in its startup log.
static void ApplyTitlePreset(QProcess* process, const QString& title_id) {
	// The launcher reuses its QProcess: every launch starts from the system environment, so a
	// previous title's preset never carries over (also when this title has no valid preset).
	process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());
	const QDir                         app_dir(QApplication::applicationDirPath());
	std::vector<std::filesystem::path> dirs {std::filesystem::path(app_dir.absolutePath().toStdU16String())};
	if (QDir parent = app_dir; parent.cdUp()) {
		dirs.emplace_back(parent.absolutePath().toStdU16String());
	}
	const auto result = Common::U59Preset::LoadTitle(dirs, title_id.toStdString());
	if (result.status == Common::U59Preset::Status::Missing) {
		return;
	}
	if (result.status != Common::U59Preset::Status::Loaded) {
		qWarning().noquote() << QString::fromStdString(Common::U59Preset::TitleFileName(title_id.toStdString()))
		                     << "was not applied:" << QString::fromStdString(result.error);
		return;
	}
	auto environment = QProcessEnvironment::systemEnvironment();
	for (const auto& entry: result.entries) {
		environment.insert(QString::fromStdString(entry.key), QString::fromStdString(entry.value));
	}
	environment.insert(QStringLiteral("KYTY_TITLE_PRESET"),
	                   QString::fromStdU16String(result.path.u16string()));
	process->setProcessEnvironment(environment);
}

// What a normal run and a prepare run share: the game's arguments, and on the process its
// environment (title preset, frametime fix) and working directory. Empty arguments: the configuration
// is not valid.
static QStringList SetupEmulatorProcess(QProcess* process, const QString& interpreter,
                                        const Configuration& info) {
	ApplyTitlePreset(process, info.title_id);

	auto args = CreateEmulatorArgs(info);
	if (args.isEmpty()) {
		return args;
	}

	process->setWorkingDirectory(QFileInfo(interpreter).absolutePath());
	// The per-game "faster GPU synchronisation" checkbox: its switches go where the bundled preset,
	// the title preset (above) or the user's environment has not set them.
	{
		auto environment = process->processEnvironment();
		info.ApplyFastGpuSync(environment);
		process->setProcessEnvironment(environment);
	}
	// The per-game "smooth frametime fix" checkbox (Immediate mode only) turns the absolute
	// vblank timeline on. Otherwise the variable is left alone, so a value from u59-preset.json
	// or the user's environment still applies (the emulator default is off). It goes over the
	// environment ApplyTitlePreset set, so a title preset stays applied.
	if (info.present_mode == Configuration::PresentMode::Immediate && info.smooth_frametime_fix) {
		auto environment = process->processEnvironment();
		environment.insert(QStringLiteral("KYTY_PRESENT_ABS_VBLANK"), QStringLiteral("1"));
		process->setProcessEnvironment(environment);
	}
	return args;
}

void MainDialog::RunInterpreter(QProcess* process, const Configuration& info, bool close_terminal) {
	const auto& interpreter = m_p->GetInterpreter();

	const auto args = SetupEmulatorProcess(process, interpreter, info);
	if (args.isEmpty()) {
		QMessageBox::critical(this, tr("Error"), tr("Invalid emulator configuration"));
		QApplication::quit();
		return;
	}

	// Without a terminal, the emulator runs directly.
	process->setProgram(interpreter);
	process->setArguments(args);
#ifdef __linux__
	QString     terminal;
	QStringList terminal_prefix;
	if (FindTerminal(&terminal, &terminal_prefix)) {
		process->setProgram(terminal);
		process->setArguments(terminal_prefix + BashCommand(interpreter, args, close_terminal));
	}
#elif defined(_WIN32)
	// Use nativeArguments so Qt does not re-quote the /K command string.
	process->setProgram(CMD_EXE);
	process->setArguments({});
	process->setNativeArguments(
	    CmdCommand(QDir::toNativeSeparators(interpreter), args, close_terminal));
#endif
#if defined(_WIN32)
	process->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
		args->flags |= static_cast<uint32_t>(CREATE_NEW_CONSOLE);
		args->startupInfo->dwFlags &= ~static_cast<DWORD>(STARTF_USESTDHANDLES);
		args->startupInfo->dwFlags |= static_cast<DWORD>(STARTF_USECOUNTCHARS);
		args->startupInfo->dwXCountChars = CMD_X_CHARS;
		args->startupInfo->dwYCountChars = CMD_Y_CHARS;
		// args->startupInfo->dwFlags |= static_cast<DWORD>(STARTF_USEFILLATTRIBUTE);
		// args->startupInfo->dwFillAttribute =
		//     static_cast<DWORD>(BACKGROUND_BLUE) | static_cast<DWORD>(FOREGROUND_RED) |
		//     static_cast<DWORD>(FOREGROUND_INTENSITY);
	});
#endif
	process->start();
#if !defined(_WIN32)
	// Report immediate launch failures.
	if (!process->waitForStarted(5000)) {
		QMessageBox::critical(
		    this, tr("Error"),
		    tr("Failed to start:\n%1\n\n%2").arg(process->program(), process->errorString()));
		return;
	}
#endif
	process->waitForFinished(100);
}

void MainDialog::WriteSettings(QSettings& s) {
	MainDialogPrivate::WriteSettings(s);
}

void MainDialog::ReadSettings(QSettings& s) {
	MainDialogPrivate::ReadSettings(s);
}

void MainDialog::resizeEvent(QResizeEvent* event) {
	emit Resize();
	QDialog::resizeEvent(event);
}

void MainDialogPrivate::WriteSettings(QSettings& s) {
	s.beginGroup(SETTINGS_MAIN_DIALOG);

	if (!g_last_geometry.isEmpty()) {
		s.setValue(SETTINGS_MAIN_LAST_GEOMETRY, g_last_geometry);
	}
	s.setValue(SETTINGS_CHECK_UPDATES, g_check_updates_on_startup);
	s.setValue(SETTINGS_CLOSE_TERMINAL, g_close_terminal);
	s.setValue(SETTINGS_THEME, g_theme);

	s.endGroup();
}

void MainDialogPrivate::ReadSettings(QSettings& s) {
	s.beginGroup(SETTINGS_MAIN_DIALOG);

	g_last_geometry = s.value(SETTINGS_MAIN_LAST_GEOMETRY, g_last_geometry).toByteArray();
	g_check_updates_on_startup = s.value(SETTINGS_CHECK_UPDATES, true).toBool();
	g_close_terminal           = s.value(SETTINGS_CLOSE_TERMINAL, false).toBool();
	g_theme                    = s.value(SETTINGS_THEME, 0).toInt();
	if (g_theme < 0 || g_theme > 2) {
		g_theme = 0;
	}

	s.endGroup();
}

void MainDialogPrivate::Run() {
	if (m_update_checker->IsInstalling()) {
		return;
	}
	QPointer<ConfigurationItem> item = m_ui->widget->GetSelectedItem();
	if (item == nullptr) {
		return;
	}

	item->SetRunning(true);
	m_running_games++;
	m_controller_preview.SetSuspended(true);
	m_title_sound.SetTitle({});

	auto* process = new QProcess(this);
	connect(process, &QProcess::stateChanged, this,
	        [this, process, item](QProcess::ProcessState state) {
		        if (state == QProcess::NotRunning) {
			        if (item != nullptr) {
				        item->SetRunning(false);
			        }
			        m_running_games--;
			        process->deleteLater();
			        Update();
		        }
	        });
	auto info = m_ui->widget->CreateConfiguration(*item);
	m_main_dialog->RunInterpreter(process, *info, g_close_terminal);

	Update();
}

// The game list's Precompile button: the emulator's prepare run (precompile.h) on the selected
// game, with the arguments and environment of a normal run, started without a console; a window
// shows its progress. A game without a shader list gets one chosen first.
void MainDialogPrivate::PrecompileShaders() {
	if (m_update_checker->IsInstalling() || m_precompiling) {
		return;
	}
	QPointer<ConfigurationItem> item = m_ui->widget->GetSelectedItem();
	if (item == nullptr || item->IsRunning()) {
		return;
	}
	const auto info         = m_ui->widget->CreateConfiguration(*item);
	const auto emulator_dir = QFileInfo(m_interpreter).absolutePath();
	// The run ends before the game shows a picture: its window need not cover the screen.
	info->fullscreen_enabled = false;

	auto       process = std::make_unique<QProcess>();
	const auto args    = SetupEmulatorProcess(process.get(), m_interpreter, *info);
	if (args.isEmpty()) {
		QMessageBox::critical(m_main_dialog, tr("Error"), tr("Invalid emulator configuration"));
		return;
	}

	// A preset that names a list file of its own: the emulator loads that one, and says so when it
	// cannot.
	auto       environment = process->processEnvironment();
	const bool custom =
	    Precompile::IsCustomList(environment.value(QLatin1String(Precompile::ENV_LIST)));
	auto list = custom ? QString()
	                   : Precompile::FindList(emulator_dir, info->title_id, info->gameVersion);
	bool installed_now = false;
	if (!custom && list.isEmpty()) {
		list = PrecompileDialog::ChooseList(m_main_dialog, emulator_dir, *info, true);
		// The dialogs ran an event loop: the row may be gone or busy by now.
		if (list.isEmpty() || item == nullptr || item->IsRunning() || m_precompiling) {
			return;
		}
		installed_now = true;
	}

	Precompile::PrepareEnvironment(&environment);
	process->setProcessEnvironment(environment);
	process->setProgram(m_interpreter);
	process->setArguments(args);

	if (m_precompile != nullptr) {
		m_precompile->close(); // the result of an earlier run
	}
	item->SetRunning(true);
	m_running_games++;
	m_precompiling     = true;
	m_precompile_title = info->title_id;
	m_ui->widget->SetPrecompileRunning(true);

	const QPointer<MainDialogPrivate> self(this);
	m_precompile = new PrecompileDialog(
	    info->name, list, installed_now, process.release(),
	    [self, item]() {
		    if (self == nullptr) {
			    return; // the launcher is closing
		    }
		    if (item != nullptr) {
			    item->SetRunning(false);
		    }
		    self->m_running_games--;
		    self->m_precompiling = false;
		    self->m_ui->widget->SetPrecompileRunning(false);
		    self->Update();
	    },
	    m_main_dialog);
	m_precompile->show();

	Update();
}

// Runs kyty_pipeline_list with a modal window that says what it does and can cancel it. The exit
// code, or -1 when it could not be started or was cancelled; its output in `output`.
static int RunListTool(QWidget* parent, const QString& tool, const QStringList& args, const QString& label,
                       QString* output, bool* cancelled) {
	QProcess process;
	process.setProgram(tool);
	process.setArguments(args);
	process.setProcessChannelMode(QProcess::MergedChannels);
	process.start(QIODevice::ReadOnly);
	if (!process.waitForStarted(10000)) {
		*output = process.errorString();
		return -1;
	}
	QProgressDialog progress(label, QObject::tr("Cancel"), 0, 0, parent);
	progress.setWindowTitle(QObject::tr("Export shader list"));
	progress.setWindowModality(Qt::WindowModal);
	progress.setMinimumDuration(0);
	progress.setAutoClose(false);
	progress.setAutoReset(false);
	QEventLoop loop;
	QObject::connect(&process, &QProcess::finished, &loop, &QEventLoop::quit);
	QObject::connect(&progress, &QProgressDialog::canceled, &loop, [&process, cancelled]() {
		*cancelled = true;
		process.kill();
	});
	// The tool's lines as they come: while it searches the game's files it says how far it is.
	QByteArray pending;
	const auto take = [&]() {
		pending += process.readAll();
		const auto end = pending.lastIndexOf('\n');
		if (end < 0) {
			return;
		}
		const auto text = QString::fromUtf8(pending.left(end + 1));
		pending.remove(0, end + 1);
		*output += text;
		const auto lines = text.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
		for (auto line = lines.crbegin(); line != lines.crend(); ++line) {
			ShaderListExport::Scan scan;
			if (ShaderListExport::ParseScan(*line, &scan)) {
				progress.setLabelText(label + QStringLiteral("\n\n") + ShaderListExport::ScanText(scan));
				progress.setRange(0, 1000);
				progress.setValue(ShaderListExport::ScanPermille(scan));
				break;
			}
		}
	};
	QObject::connect(&process, &QProcess::readyRead, &progress, take);
	if (process.state() != QProcess::NotRunning) {
		progress.show();
		loop.exec();
	}
	process.waitForFinished(5000);
	take();
	*output += QString::fromUtf8(pending + process.readAll());
	if (*cancelled || process.exitStatus() != QProcess::NormalExit) {
		return -1;
	}
	return process.exitCode();
}

// The game list's Export list button: a shader list of what this player's sessions of the selected
// game recorded (shaderListExport.h), checked for shader code before it is kept, then where it is.
void MainDialogPrivate::ExportShaderList() {
	if (m_update_checker->IsInstalling() || m_exporting) {
		return;
	}
	QPointer<ConfigurationItem> item = m_ui->widget->GetSelectedItem();
	if (item == nullptr) {
		return;
	}
	const auto  info         = m_ui->widget->CreateConfiguration(*item);
	const auto  emulator_dir = QFileInfo(m_interpreter).absolutePath();
	const auto  window_title = tr("Export shader list");
	const auto& game_name    = info->name;
	const auto  fail         = [this, &window_title](const QString& text, QMessageBox::Icon icon = QMessageBox::Warning) {
		qInfo().noquote() << "export-list: failed:" << text;
		QMessageBox box(icon, window_title, text, QMessageBox::Ok, m_main_dialog);
		box.setTextFormat(Qt::PlainText);
		box.exec();
	};

	const auto tool = ShaderListExport::FindTool(QApplication::applicationDirPath());
	if (tool.isEmpty()) {
		fail(tr("kyty_pipeline_list is missing: it belongs next to the launcher. Unpack the whole release "
		        "archive again."));
		return;
	}
	if (!Precompile::IsValidGame(info->title_id, info->gameVersion)) {
		fail(tr("%1 has no title ID, so it cannot have a shader list.").arg(game_name));
		return;
	}
	if (info->basedir.isEmpty() || !QFileInfo::exists(info->basedir)) {
		fail(tr("The game files of %1 are not found at %2.")
		         .arg(game_name, QDir::toNativeSeparators(info->basedir)));
		return;
	}
	// A running session (or a precompile) of this game still writes what it records.
	if (item->IsRunning() || (m_precompiling && m_precompile_title == info->title_id) ||
	    m_ui->widget->IsTitleRunning(info->title_id)) {
		fail(tr("%1 is running, or its shaders are being precompiled. Close it first: the game is still "
		        "writing what it records.")
		         .arg(game_name),
		     QMessageBox::Information);
		return;
	}
	const auto journals = ShaderListExport::FindJournals(emulator_dir, info->title_id);
	const auto nothing_recorded =
	    tr("Nothing has been recorded for %1 yet.\n\nKyty records the shaders and pipelines a game uses while "
	       "you play it with this launcher. Play the game for a while, close it, then export the list.")
	        .arg(game_name);
	if (journals.IsEmpty()) {
		fail(nothing_recorded, QMessageBox::Information);
		return;
	}
	const auto out = m_export_out.isEmpty()
	                     ? ShaderListExport::DefaultOutput(emulator_dir, info->title_id, info->gameVersion)
	                     : m_export_out;
	if (!QDir().mkpath(QFileInfo(out).absolutePath())) {
		fail(tr("Could not create the folder %1.").arg(QDir::toNativeSeparators(QFileInfo(out).absolutePath())));
		return;
	}
	const auto partial = ShaderListExport::PartialPath(out);
	QFile::remove(partial);
	// A folder game started from another executable than eboot.bin: that one holds the shaders.
	QString executable;
	if (!GameContent::IsArchive(info->basedir) && !info->elf.isEmpty() && info->elf != QStringLiteral("eboot.bin")) {
		executable = QDir(info->basedir).filePath(info->elf);
	}

	// The list the game's sessions ran with (as Precompile finds it): the emulator records only what
	// it did not get from that list, so the export takes it in.
	QProcess probe;
	(void)SetupEmulatorProcess(&probe, m_interpreter, *info);
	const auto base_list =
	    ShaderListExport::BaseList(emulator_dir, info->title_id, info->gameVersion,
	                               probe.processEnvironment().value(QLatin1String(Precompile::ENV_LIST)));

	m_exporting = true;
	struct Done {
		bool* flag;
		~Done() { *flag = false; }
	} done {&m_exporting};
	QString output;
	bool    cancelled = false;
	int     code      = RunListTool(m_main_dialog, tool,
	                                ShaderListExport::ExportArguments(partial, info->basedir, executable, journals, base_list),
	                                tr("Making the shader list of %1...").arg(game_name), &output, &cancelled);
	ShaderListExport::Written written;
	if (cancelled) {
		QFile::remove(partial);
		fail(tr("Cancelled."), QMessageBox::Information);
		return;
	}
	if (code != 0 || !ShaderListExport::ParseWritten(output, &written)) {
		QFile::remove(partial);
		fail(tr("The export failed:\n\n%1").arg(ShaderListExport::Tail(output, 6)));
		return;
	}
	if (written.shaders == 0) {
		QFile::remove(partial);
		fail(nothing_recorded, QMessageBox::Information);
		return;
	}
	QString verify_output;
	code = RunListTool(m_main_dialog, tool,
	                   ShaderListExport::VerifyArguments(partial, info->basedir, executable, journals),
	                   tr("Checking that the list holds no shader code..."), &verify_output, &cancelled);
	if (cancelled) {
		QFile::remove(partial);
		fail(tr("Cancelled."), QMessageBox::Information);
		return;
	}
	if (code != 0) {
		QFile::remove(partial);
		fail(code == 1 ? tr("The check found shader code in the list, so the list was not kept.")
		               : tr("The list could not be checked, so it was not kept:\n\n%1")
		                     .arg(ShaderListExport::Tail(verify_output, 6)));
		return;
	}
	QString error;
	if (!ShaderListExport::Keep(partial, out, &error)) {
		fail(error);
		return;
	}
	m_ui->widget->UpdatePrecompileButtons();

	const auto native = QDir::toNativeSeparators(out);
	qInfo().noquote() << "export-list: ok:" << native << written.shaders << "shaders" << written.permutations
	                  << "permutations" << written.pipelines << "pipelines";
	// With a list installed for the game: what the sessions added to it.
	QString                   added_text;
	ShaderListExport::Written added;
	if (ShaderListExport::ParseAdded(output, &added)) {
		added_text = added.shaders == 0 && added.permutations == 0 && added.pipelines == 0
		                 ? tr("Your sessions added nothing to the shader list this game already has here: the file "
		                      "holds that list.\n\n")
		                 : tr("It holds the shader list this game already has here and what your sessions added to "
		                      "it: %1 shaders, %2 permutations, %3 pipelines.\n\n")
		                       .arg(added.shaders)
		                       .arg(added.permutations)
		                       .arg(added.pipelines);
	}
	QMessageBox box(QMessageBox::Information, window_title,
	                tr("The shader list of %1 is saved:\n%2\n\n%3 shaders, %4 permutations, %5 pipelines.\n\n%6"
	                   "Send this one file to other players. It holds where each shader is in the game files and "
	                   "its hashes, no game code. Do not send the _PipelineCache folder, the .journal files or any "
	                   "game files.")
	                    .arg(game_name, native)
	                    .arg(written.shaders)
	                    .arg(written.permutations)
	                    .arg(written.pipelines)
	                    .arg(added_text),
	                QMessageBox::Close, m_main_dialog);
	box.setTextFormat(Qt::PlainText);
	box.setTextInteractionFlags(Qt::TextSelectableByMouse);
	auto* open = box.addButton(tr("Open folder"), QMessageBox::ActionRole);
	box.setDefaultButton(open);
	box.exec();
	if (box.clickedButton() == open) {
#if defined(_WIN32)
		// explorer.exe wants the quotes after the comma: Qt would quote the whole argument.
		QProcess explorer;
		explorer.setProgram(QStringLiteral("explorer.exe"));
		explorer.setNativeArguments(QStringLiteral("/select,\"%1\"").arg(native));
		explorer.startDetached();
#else
		QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(out).absolutePath()));
#endif
	}
}

void MainDialogPrivate::Update() {
	const auto* item = m_ui->widget->GetSelectedItem();

	bool run_enabled = item != nullptr && !item->IsRunning() && !m_update_checker->IsInstalling();

	if (run_enabled) {
		const auto& info = item->GetInfo();
		auto        dir  = info.basedir;
		run_enabled      = !dir.isEmpty() && (QDir(dir).exists() || GameContent::IsArchive(dir));
		// The game's other row (its folder and its archive) while its shaders are precompiled: the
		// game and the prepare run would write the same caches.
		if (m_precompiling && info.title_id == m_precompile_title) {
			run_enabled = false;
		}
	}

	m_ui->widget->SetRunEnabled(run_enabled);
	m_ui->widget->UpdatePrecompileButtons();
	m_update_checker->SetGameRunning(m_running_games > 0);
	if (m_running_games > 0) {
		m_controller_preview.SetSuspended(true);
		return;
	}
	m_controller_preview.SetSuspended(false);
	m_controller_preview.SetColor(m_ui->widget->GetGlobalControllerColor());
}

#include "mainDialog.moc"
