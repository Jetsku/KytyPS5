#ifndef PRECOMPILE_H
#define PRECOMPILE_H

#include <QByteArray>
#include <QProcessEnvironment>
#include <QString>
#include <QtGlobal>

// The game list's Precompile button: the emulator's prepare run (KYTY_PIPELINE_LIST_PREPARE=1) builds
// the shaders and pipelines that a shader list (a .kyplst pipeline list) names for the game and saves
// them in its caches, then exits. This is what the launcher knows about it without a window: where
// the emulator looks for a game's list, and the lines the run prints.
namespace Precompile {

inline constexpr char ENV_PREPARE[] = "KYTY_PIPELINE_LIST_PREPARE";
inline constexpr char ENV_LIST[]    = "KYTY_PIPELINE_LIST";
inline constexpr char ENV_REPLAY[]  = "KYTY_SHADER_PRECOMPILE_REPLAY";
// The environment of a prepare run, over the game's own (bundled preset, title preset): the
// prepare switch, and the shader replay on. A title preset may keep the replay off for play
// (KYTY_SHADER_PRECOMPILE_REPLAY=0: the game then only records); with it off the prepare run
// would load the list and build nothing.
void PrepareEnvironment(QProcessEnvironment* environment);
// The emulator's exit code when it found no usable list.
inline constexpr int EXIT_NO_LIST = 2;

// Whether a list can be named after the game at all (the names go into a file name).
[[nodiscard]] bool IsValidGame(const QString& title_id, const QString& version);

// The game's list in the emulator's folder, as the emulator looks for it:
// PipelineLists/<title>_<version>.kyplst, else PipelineLists/<title>.kyplst. Empty when neither exists.
[[nodiscard]] QString FindList(const QString& emulator_dir, const QString& title_id,
                               const QString& version);
// Where a list chosen for the game goes: the versioned name (the title's alone without a version).
// Empty for a game IsValidGame refuses.
[[nodiscard]] QString InstallPath(const QString& emulator_dir, const QString& title_id,
                                  const QString& version);
// KYTY_PIPELINE_LIST naming a file of its own (anything but unset, "0" and "1"): the emulator then
// loads that file instead of the game's list in PipelineLists.
[[nodiscard]] bool IsCustomList(const QString& setting);

// Whether the bytes start a list file (the title and version check is the emulator's).
[[nodiscard]] bool HasListMagic(const QByteArray& head);
[[nodiscard]] bool IsListFile(const QString& path, QString* error);
// Copies a list to its place and creates the folder. An existing file is replaced only with
// `replace`; without it the copy fails.
[[nodiscard]] bool InstallList(const QString& source, const QString& target, bool replace,
                               QString* error);

// "KYTY_EVENT: prepare-progress ... phase=<name> shaders=<done>/<total> pipelines=<done>/<total>
// compute=<done>/<total>", about once a second.
struct Progress {
	QString phase;
	quint64 shaders_done    = 0;
	quint64 shaders_total   = 0;
	quint64 pipelines_done  = 0;
	quint64 pipelines_total = 0;
	quint64 compute_done    = 0;
	quint64 compute_total   = 0;
};
[[nodiscard]] bool ParseProgress(const QString& line, Progress* progress);
// 0..1000 of the whole run: shaders fill 0..150, graphics pipelines 150..900, compute pipelines
// 900..990 (each by the count ProgressText shows), 1000 while saving.
[[nodiscard]] int     ProgressPermille(const Progress& progress);
[[nodiscard]] QString ProgressText(const Progress& progress);

// The run's last line: "KYTY_EVENT: prepare-done ... seconds=<s> shaders=<n> pipelines=<n> known=<n>
// compute=<n>", or the "Prepare shaders: done in ..." line of an emulator without the event.
struct Result {
	double  seconds   = 0.0;
	quint64 shaders   = 0;
	quint64 pipelines = 0;
	quint64 known     = 0;
	quint64 compute   = 0;
};
[[nodiscard]] bool    ParseDone(const QString& line, Result* result);
[[nodiscard]] QString DoneText(const Result& result);

// The emulator's lines about the list itself ("Pipeline list: ..."): the last one before exit code 2
// says why the list was not used. Returns the text from "Pipeline list:" on, empty for other lines.
[[nodiscard]] QString ListLine(const QString& line);

} // namespace Precompile

#endif // PRECOMPILE_H
