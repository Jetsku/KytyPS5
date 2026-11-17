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
	[[nodiscard]] bool IsEmpty() const { return shaders.isEmpty(); }
};
// The title's journals in the emulator's folder; empty unless both files exist and are not empty.
// (Journals that hold no shader yet give an export of 0 shaders: the launcher says the same then.)
[[nodiscard]] Journals FindJournals(const QString& emulator_dir, const QString& title_id);

// <emulator_dir>/ShaderListExports/<title>-<version>-shaderlist.kyplst (the export window's name);
// empty for a game without a usable title ID (Precompile::IsValidGame).
[[nodiscard]] QString DefaultOutput(const QString& emulator_dir, const QString& title_id, const QString& version);
// Where the list is written until both steps pass: next to its place, so the move is a rename.
[[nodiscard]] QString PartialPath(const QString& out);

// The tool's arguments. `game`: the game folder or .zar; `executable`: empty for its eboot.bin.
[[nodiscard]] QStringList ExportArguments(const QString& partial, const QString& game, const QString& executable,
                                          const Journals& journals);
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

// Moves the checked list to `out` (replacing an older one there). False with `error` set when it
// cannot; the partial file is removed either way.
[[nodiscard]] bool Keep(const QString& partial, const QString& out, QString* error);

// The last `count` lines of the tool's output with text, for a message.
[[nodiscard]] QString Tail(const QString& output, int count);

} // namespace ShaderListExport

#endif // SHADER_LIST_EXPORT_H
