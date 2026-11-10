#include "precompileDialog.h"

#include "configuration.h"

#include <QAbstractButton>
#include <QCloseEvent>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QMetaObject>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRegularExpression>
#include <QTimer>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace {

constexpr int MAX_LINES        = 400;  // the output box keeps the last lines only
constexpr int SHOW_INTERVAL_MS = 150;  // a console-direction run prints thousands of lines
constexpr int MAX_LINE_BYTES   = 65536; // output without a newline is shown once this long
constexpr int BAR_RANGE        = 1000;
// The bar is full only when the emulator has saved the caches and ended.
constexpr int BAR_BEFORE_END = 990;

} // namespace

PrecompileDialog::PrecompileDialog(const QString& game, const QString& list, bool installed_now,
                                   QProcess* process, std::function<void()> finished,
                                   QWidget* parent)
    : QDialog(parent), m_list(list), m_installed_now(installed_now), m_process(process),
      m_finished(std::move(finished)) {
	setObjectName(QStringLiteral("precompile_dialog"));
	setWindowTitle(tr("Precompile shaders"));
	setAttribute(Qt::WA_DeleteOnClose);
	resize(760, 440);

	auto* layout = new QVBoxLayout(this);

	auto* title = new QLabel(this);
	title->setTextFormat(Qt::PlainText);
	title->setWordWrap(true);
	title->setTextInteractionFlags(Qt::TextSelectableByMouse);
	title->setText(tr("%1\nShader list: %2")
	                   .arg(game, list.isEmpty() ? tr("the file KYTY_PIPELINE_LIST names")
	                                             : QDir::toNativeSeparators(list)));
	layout->addWidget(title);

	m_status = new QLabel(tr("Starting the emulator..."), this);
	m_status->setObjectName(QStringLiteral("precompile_status"));
	layout->addWidget(m_status);

	m_bar = new QProgressBar(this);
	m_bar->setObjectName(QStringLiteral("precompile_progress"));
	m_bar->setRange(0, BAR_RANGE);
	m_bar->setValue(0);
	layout->addWidget(m_bar);

	m_output = new QPlainTextEdit(this);
	m_output->setObjectName(QStringLiteral("precompile_output"));
	m_output->setReadOnly(true);
	m_output->setMaximumBlockCount(MAX_LINES);
	m_output->setLineWrapMode(QPlainTextEdit::NoWrap);
	m_output->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));
	layout->addWidget(m_output, 1);

	m_result_label = new QLabel(this);
	m_result_label->setObjectName(QStringLiteral("precompile_result"));
	m_result_label->setTextFormat(Qt::PlainText);
	m_result_label->setWordWrap(true);
	m_result_label->setTextInteractionFlags(Qt::TextSelectableByMouse);
	auto result_font = m_result_label->font();
	result_font.setBold(true);
	m_result_label->setFont(result_font);
	m_result_label->hide();
	layout->addWidget(m_result_label);

	auto* buttons = new QHBoxLayout;
	buttons->addStretch(1);
	m_button = new QPushButton(tr("Cancel"), this);
	m_button->setObjectName(QStringLiteral("precompile_cancel"));
	buttons->addWidget(m_button);
	layout->addLayout(buttons);
	connect(m_button, &QPushButton::clicked, this, [this]() {
		if (m_running) {
			Cancel();
		} else {
			close();
		}
	});

	m_show_timer = new QTimer(this);
	m_show_timer->setInterval(SHOW_INTERVAL_MS);
	connect(m_show_timer, &QTimer::timeout, this, [this]() { ShowLines(); });
	m_show_timer->start();

	m_process->setParent(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() { ReadOutput(); });
	connect(m_process, &QProcess::finished, this,
	        [this](int exit_code, QProcess::ExitStatus status) { ProcessFinished(exit_code, status); });
	connect(m_process, &QProcess::errorOccurred, this,
	        [this](QProcess::ProcessError error) { ProcessFailed(error); });
	m_running = true;
	m_clock.start();
	// Started from the event loop: a start that fails at once reports to a caller that has finished
	// setting the run up.
	QMetaObject::invokeMethod(
	    this,
	    [this]() {
		    if (m_running && !m_cancelled) {
			    m_process->start(QIODevice::ReadOnly);
		    }
	    },
	    Qt::QueuedConnection);
}

PrecompileDialog::~PrecompileDialog() {
	// Closed with the launcher while the run goes on: it ends with its window.
	m_process->disconnect(this);
	if (m_process->state() != QProcess::NotRunning) {
		m_process->kill();
		m_process->waitForFinished(3000);
	}
	if (m_running) {
		m_running = false;
		if (m_finished) {
			m_finished();
		}
	}
}

QString PrecompileDialog::s_test_list;

QString PrecompileDialog::ChooseList(QWidget* parent, const QString& emulator_dir,
                                     const Configuration& game, bool explain_missing) {
	const auto title  = tr("Precompile shaders");
	const auto target = Precompile::InstallPath(emulator_dir, game.title_id, game.gameVersion);
	if (target.isEmpty()) {
		QMessageBox::information(
		    parent, title,
		    tr("%1 has no title ID, so a shader list cannot be installed for it.").arg(game.name));
		return {};
	}
	const auto native_target = QDir::toNativeSeparators(target);
	const auto game_text     = game.gameVersion.isEmpty()
	                               ? QStringLiteral("%1, %2").arg(game.name, game.title_id)
	                               : tr("%1, %2 version %3")
	                                     .arg(game.name, game.title_id, game.gameVersion);
	QString source = s_test_list;
	if (source.isEmpty()) {
		const auto text =
		    (explain_missing ? tr("No shader list is installed for %1.") : tr("Shader list for %1."))
		        .arg(game_text) +
		    QStringLiteral("\n\n") +
		    tr("A shader list (.kyplst) names the shaders and pipelines a game uses, so the emulator "
		       "can build them all before you play instead of one by one while you play.") +
		    QStringLiteral("\n\n") + tr("The file you choose is copied to %1.").arg(native_target);
		QMessageBox question(QMessageBox::Information, title, text, QMessageBox::Cancel, parent);
		question.setTextFormat(Qt::PlainText);
		auto* choose = question.addButton(tr("Choose a .kyplst file..."), QMessageBox::AcceptRole);
		question.setDefaultButton(choose);
		question.exec();
		if (question.clickedButton() != choose) {
			return {};
		}

		source = QFileDialog::getOpenFileName(parent, tr("Choose a shader list"), {},
		                                      tr("Shader lists (*.kyplst);;All files (*)"));
		if (source.isEmpty()) {
			return {};
		}
	}
	QString error;
	if (!Precompile::IsListFile(source, &error)) {
		QMessageBox::warning(parent, title, error);
		return {};
	}
	// Never over an installed list without asking.
	bool replace = false;
	if (QFileInfo::exists(target) &&
	    QFileInfo(target).canonicalFilePath() != QFileInfo(source).canonicalFilePath()) {
		if (QMessageBox::question(parent, title,
		                          tr("%1 already exists.\n\nReplace it?").arg(native_target),
		                          QMessageBox::Yes | QMessageBox::No,
		                          QMessageBox::No) != QMessageBox::Yes) {
			return {};
		}
		replace = true;
	}
	if (!Precompile::InstallList(source, target, replace, &error)) {
		QMessageBox::warning(parent, title, error);
		return {};
	}
	return target;
}

void PrecompileDialog::ReadOutput() {
	m_partial += m_process->readAllStandardOutput();
	qsizetype start = 0;
	for (;;) {
		const auto end = m_partial.indexOf('\n', start);
		if (end < 0) {
			break;
		}
		TakeLine(QString::fromUtf8(m_partial.constData() + start, end - start));
		start = end + 1;
	}
	m_partial.remove(0, start);
	if (m_partial.size() >= MAX_LINE_BYTES) {
		TakeLine(QString::fromUtf8(m_partial));
		m_partial.clear();
	}
}

void PrecompileDialog::TakeLine(const QString& raw) {
	// Console output may be coloured, and ends in "\r\n" on Windows.
	static const QRegularExpression colour(QStringLiteral("\x1b\\[[0-9;]*m"));
	auto                            line = raw;
	line.remove(colour);
	while (!line.isEmpty() && line.back().isSpace()) {
		line.chop(1);
	}
	if (line.isEmpty()) {
		return;
	}

	Precompile::Progress progress;
	if (Precompile::ParseProgress(line, &progress)) {
		// A total grows while shaders publish compute programs: the bar never steps back.
		m_permille =
		    std::max(m_permille, std::min(Precompile::ProgressPermille(progress), BAR_BEFORE_END));
		m_bar->setValue(m_permille);
		m_status->setText(Precompile::ProgressText(progress));
		return;
	}
	Precompile::Result result;
	if (Precompile::ParseDone(line, &result)) {
		const bool event = line.contains(QLatin1String("KYTY_EVENT:"));
		m_result         = result;
		m_has_result     = true;
		if (event) {
			return; // the line before it says the same in words
		}
	}
	if (const auto list_line = Precompile::ListLine(line); !list_line.isEmpty()) {
		m_list_line = list_line;
	}
	m_pending.append(line);
	if (m_pending.size() > MAX_LINES) {
		m_pending.remove(0, m_pending.size() - MAX_LINES);
	}
}

void PrecompileDialog::ShowLines() {
	if (m_pending.isEmpty()) {
		return;
	}
	m_output->appendPlainText(m_pending.join(QLatin1Char('\n')));
	m_pending.clear();
}

void PrecompileDialog::Finish(const QString& text, bool ok) {
	if (!m_running) {
		return;
	}
	m_running = false;
	m_show_timer->stop();
	ShowLines();
	m_bar->setValue(ok ? BAR_RANGE : m_permille);
	m_status->setText(ok ? tr("Finished.") : tr("Not finished."));
	m_result_label->setText(text);
	m_result_label->show();
	m_button->setText(tr("Close"));
	m_button->setEnabled(true);
	if (m_finished) {
		m_finished();
	}
}

void PrecompileDialog::ProcessFinished(int exit_code, QProcess::ExitStatus status) {
	ReadOutput();
	if (!m_partial.isEmpty()) {
		TakeLine(QString::fromUtf8(m_partial));
		m_partial.clear();
	}
	if (m_cancelled) {
		Finish(tr("Cancelled."), false);
	} else if (status == QProcess::CrashExit) {
		Finish(tr("The emulator stopped unexpectedly (exit code 0x%1).")
		           .arg(static_cast<quint32>(exit_code), 8, 16, QLatin1Char('0')),
		       false);
	} else if (exit_code == 0) {
		Finish(m_has_result ? Precompile::DoneText(m_result)
		                    : tr("Finished in %1 s.").arg(m_clock.elapsed() / 1000),
		       true);
	} else if (exit_code == Precompile::EXIT_NO_LIST) {
		auto text = m_list_line.isEmpty()
		                ? tr("The emulator found no usable shader list for this game.")
		                : m_list_line;
		// A file copied into place for this run would be refused at every start of the game too.
		if (m_installed_now && QFile::remove(m_list)) {
			text += QStringLiteral("\n") + tr("The chosen file was removed again.");
		}
		Finish(text, false);
	} else {
		Finish(tr("The emulator ended with exit code %1.").arg(exit_code), false);
	}
}

void PrecompileDialog::ProcessFailed(QProcess::ProcessError error) {
	// Every other error ends in ProcessFinished.
	if (error == QProcess::FailedToStart) {
		Finish(tr("The emulator could not be started: %1").arg(m_process->errorString()), false);
	}
}

void PrecompileDialog::Cancel() {
	if (!m_running || m_cancelled) {
		return;
	}
	m_cancelled = true;
	if (m_process->state() == QProcess::NotRunning) {
		Finish(tr("Cancelled."), false);
		return;
	}
	m_status->setText(tr("Stopping the emulator..."));
	m_button->setEnabled(false);
	m_process->kill();
}

// Escape and the window's close button stop a run first; the window stays to say so.
void PrecompileDialog::reject() {
	if (m_running) {
		Cancel();
		return;
	}
	QDialog::reject();
}

void PrecompileDialog::closeEvent(QCloseEvent* event) {
	if (m_running) {
		Cancel();
		event->ignore();
		return;
	}
	QDialog::closeEvent(event);
}
