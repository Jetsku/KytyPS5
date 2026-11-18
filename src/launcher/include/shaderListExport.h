#ifndef SHADER_LIST_EXPORT_H
#define SHADER_LIST_EXPORT_H

#include <QString>
#include <QStringList>
#include <QtGlobal>

// The game list's "Export list" button: a shader list (.kyplst) of what the player's own sessions
// recorded (the emulator's _PipelineCache/<title>.shaders.journal and .pipelines.journal, written
// while a game runs with the bundled preset), to send to other players. kyty_pipeline_list (next to
// the launcher or one folder up, like the emulator) does the work, as in the kyty_shaderlist_export
// window: export, then verify (no 64-byte window of shader code is in the file). The list is written
// under a temporary name and moved to its place only when both steps pass. This is what the launcher
// knows about it without a window.
namespace ShaderListExport {

inline constexpr char PRODUCER[]   = "kyty launcher export";
inline constexpr char OUTPUT_DIR[] = "ShaderListExports";

// kyty_pipeline_list(.exe) in `launcher_dir` or its parent; empty when neither holds it.
[[nodiscard]] QString FindTool(const QString& launcher_dir);

struct Journals {
	QString shaders;
	QString pipelines;
	[[nodiscard]] bool IsEmpty() const { return shaders.isEmpty() && pipelines.isEmpty(); }
};
// The title's journals in the emulator's folder: each one that exists and is not empty. A session
// whose shaders all came from an installed list writes pipelines only. (Journals that hold nothing
// to list give an export of 0 shaders: the launcher says the same then.)
[[nodiscard]] Journals FindJournals(const QString& emulator_dir, const QString& title_id);

// The list the game's sessions ran with, as the emulator picks it: the file KYTY_PIPELINE_LIST names
// (`setting`, when it names a file of its own), else the game's list in PipelineLists. Empty when
// there is none or lists are off ("0"). The emulator records only what it did not get from that
// list, so the export takes the list in (kyty_pipeline_list export --base).
[[nodiscard]] QString BaseList(const QString& emulator_dir, const QString& title_id, const QString& version,
                               const QString& setting);

// <emulator_dir>/ShaderListExports/<title>-<version>-shaderlist.kyplst (the export window's name);
// empty for a game without a usable title ID (Precompile::IsValidGame).
[[nodiscard]] QString DefaultOutput(const QString& emulator_dir, const QString& title_id, const QString& version);
// Where the list is written until both steps pass: next to its place, so the move is a rename.
[[nodiscard]] QString PartialPath(const QString& out);

// The tool's arguments. `game`: the game folder or .zar; `executable`: empty for its eboot.bin;
// `base_list`: BaseList, may be empty. The export also searches the game's data files for the
// shaders that are not in the executable (--scan).
[[nodiscard]] QStringList ExportArguments(const QString& partial, const QString& game, const QString& executable,
                                          const Journals& journals, const QString& base_list);
[[nodiscard]] QStringList VerifyArguments(const QString& partial, const QString& game, const QString& executable,
                                          const Journals& journals);

// The counts of the export's "wrote <file>: <n> bytes, <n> shaders, <n> permutations, <n> pipelines"
// line; false when the output has none.
struct Written {
	quint64 shaders      = 0;
	quint64 permutations = 0;
	quint64 pipelines    = 0;
};
[[nodiscard]] bool ParseWritten(const QString& output, Written* written);
// The export's "new since the base list: <n> shaders, <n> permutations, <n> pipelines" line: what the
// sessions added to the base list. False when the output has none (no base list was used).
[[nodiscard]] bool ParseAdded(const QString& output, Written* added);

// "scan: <f> of <g> files, <x> of <y> GB read, <n> of <m> codes located", about every two seconds
// while the export searches the game's files for shaders it has not located yet.
struct Scan {
	quint64 files       = 0;
	quint64 files_total = 0;
	double  read_gb  = 0.0;
	double  total_gb = 0.0;
	quint64 located  = 0;
	quint64 wanted   = 0;
};
[[nodiscard]] bool    ParseScan(const QString& line, Scan* scan);
// 0..999: the mean of the part of the files and the part of the bytes (many small files come first;
// the search ends as soon as every shader is located).
[[nodiscard]] int     ScanPermille(const Scan& scan);
[[nodiscard]] QString ScanText(const Scan& scan);

// Moves the checked list to `out` (replacing an older one there). False with `error` set when it
// cannot; the partial file is removed either way.
[[nodiscard]] bool Keep(const QString& partial, const QString& out, QString* error);

// The last `count` lines of the tool's output with text, for a message.
[[nodiscard]] QString Tail(const QString& output, int count);

} // namespace ShaderListExport

#endif // SHADER_LIST_EXPORT_H
