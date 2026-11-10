#ifndef PRECOMPILE_DIALOG_H
#define PRECOMPILE_DIALOG_H

#include "precompile.h"

#include <QByteArray>
#include <QDialog>
#include <QElapsedTimer>
#include <QProcess>
#include <QString>
#include <QStringList>

#include <functional>

class Configuration;
class QCloseEvent;
class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QTimer;

// The window of one prepare run (precompile.h): a progress bar from the emulator's progress lines,
// the last lines it printed, Cancel, and the result at the end.
class PrecompileDialog final: public QDialog {
public:
	// The process has its program, arguments, environment and working directory; the window starts
	// it and owns it. `installed_now`: the list was copied into place for this run, so it is removed
	// again when the emulator refuses it. `finished` is called once, when the process has ended
	// (or could not be started).
	PrecompileDialog(const QString& game, const QString& list, bool installed_now, QProcess* process,
	                 std::function<void()> finished, QWidget* parent = nullptr);
	~PrecompileDialog() override;

	// Explains what a shader list is and lets the user choose one for the game; the file is checked
	// and copied to its place in the emulator's folder. Returns that place, or empty when nothing was
	// installed. With `explain_missing` the question starts from "no list is installed".
	static QString ChooseList(QWidget* parent, const QString& emulator_dir, const Configuration& game,
	                          bool explain_missing);
	// Test seam (the launcher's --precompile-list argument): this file is taken as the user's
	// choice, without the question and the file dialog; the checks and the copy are the same.
	static void SetTestList(const QString& path) { s_test_list = path; }

	void reject() override;

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	static QString s_test_list;
	void ReadOutput();
	void TakeLine(const QString& line);
	void ShowLines();
	void Finish(const QString& text, bool ok);
	void ProcessFinished(int exit_code, QProcess::ExitStatus status);
	void ProcessFailed(QProcess::ProcessError error);
	void Cancel();

	QString               m_list;
	bool                  m_installed_now = false;
	QProcess*             m_process       = nullptr;
	std::function<void()> m_finished;
	bool                  m_running    = false;
	bool                  m_cancelled  = false;
	bool                  m_has_result = false;
	Precompile::Result    m_result;
	QString               m_list_line; // the emulator's last line about the list
	QByteArray            m_partial;   // output after the last newline
	QStringList           m_pending;   // lines not shown yet
	int                   m_permille = 0;
	QElapsedTimer         m_clock;
	QLabel*               m_status       = nullptr;
	QProgressBar*         m_bar          = nullptr;
	QPlainTextEdit*       m_output       = nullptr;
	QLabel*               m_result_label = nullptr;
	QPushButton*          m_button       = nullptr;
	QTimer*               m_show_timer   = nullptr;
};

#endif // PRECOMPILE_DIALOG_H
