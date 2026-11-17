// The launcher's Export list button without its windows (shaderListExport.h): where it finds
// kyty_pipeline_list and the journals, the file names, the tool's arguments, the export's counts,
// and that only a checked list is kept.
#include "shaderListExport.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "LauncherShaderListExportTests: failed: %s\n", text);
		std::abort();
	}
}

void Write(const QString& path, const QByteArray& bytes) {
	Check(QDir().mkpath(QFileInfo(path).absolutePath()), "cannot create a folder");
	QFile file(path);
	Check(file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size(), "cannot write a file");
}

QByteArray Read(const QString& path) {
	QFile file(path);
	return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

} // namespace

int main(int argc, char** argv) {
	QCoreApplication application(argc, argv);
	using namespace ShaderListExport;

	QTemporaryDir root;
	Check(root.isValid(), "cannot create a temporary directory");
	const QDir base(root.path());
#if defined(_WIN32)
	const QString tool_name = QStringLiteral("kyty_pipeline_list.exe");
#else
	const QString tool_name = QStringLiteral("kyty_pipeline_list");
#endif

	// The tool: next to the launcher, else one folder up (a build tree), else none.
	const auto launcher_dir = base.filePath(QStringLiteral("kyty/launcher"));
	Check(QDir().mkpath(launcher_dir), "cannot create the launcher folder");
	Check(FindTool(launcher_dir).isEmpty(), "no tool anywhere: none found");
	Write(base.filePath(QStringLiteral("kyty/") + tool_name), "tool");
	Check(FindTool(launcher_dir) == QDir::cleanPath(base.filePath(QStringLiteral("kyty/") + tool_name)),
	      "the tool one folder up");
	Write(QDir(launcher_dir).filePath(tool_name), "tool");
	Check(FindTool(launcher_dir) == QDir::cleanPath(QDir(launcher_dir).filePath(tool_name)), "the tool next to the launcher first");

	// Journals: both files, not empty.
	const auto emulator = base.filePath(QStringLiteral("kyty"));
	const auto title    = QStringLiteral("PPSA00001");
	const auto shaders  = QDir(emulator).filePath(QStringLiteral("_PipelineCache/PPSA00001.shaders.journal"));
	const auto pipes    = QDir(emulator).filePath(QStringLiteral("_PipelineCache/PPSA00001.pipelines.journal"));
	Check(FindJournals(emulator, title).IsEmpty(), "no journals yet");
	Write(shaders, "journal");
	Check(FindJournals(emulator, title).IsEmpty(), "one journal is not enough");
	Write(pipes, QByteArray());
	Check(FindJournals(emulator, title).IsEmpty(), "an empty pipeline journal is nothing recorded");
	Write(pipes, "journal");
	const auto journals = FindJournals(emulator, title);
	Check(!journals.IsEmpty() && journals.shaders == QDir::cleanPath(shaders) && journals.pipelines == QDir::cleanPath(pipes),
	      "both journals are found");
	Check(FindJournals(emulator, QStringLiteral("PPSA00002")).IsEmpty(), "another title has none");
	Check(FindJournals(emulator, QStringLiteral("../PPSA00001")).IsEmpty(), "a title that is not a name is refused");

	// File names.
	const auto out = DefaultOutput(emulator, title, QStringLiteral("01.018.000"));
	Check(out == QDir::cleanPath(QDir(emulator).filePath(QStringLiteral("ShaderListExports/PPSA00001-01.018.000-shaderlist.kyplst"))),
	      "the default name is the export window's");
	Check(DefaultOutput(emulator, title, {}).endsWith(QStringLiteral("/PPSA00001-shaderlist.kyplst")), "a game without a version");
	Check(DefaultOutput(emulator, {}, QStringLiteral("1")).isEmpty(), "a game without a title ID has none");
	Check(PartialPath(out) == out + QStringLiteral(".partial"), "the partial file sits next to the list");

	// Arguments: those of the export window, with native paths; --exe only when given.
	const auto partial = PartialPath(out);
	const auto game    = base.filePath(QStringLiteral("games/PPSA00001"));
	const auto ea      = ExportArguments(partial, game, {}, journals);
	Check(ea == QStringList({QStringLiteral("export"), QStringLiteral("--out"), QDir::toNativeSeparators(partial),
	                         QStringLiteral("--game"), QDir::toNativeSeparators(game), QStringLiteral("--shaders"),
	                         QDir::toNativeSeparators(journals.shaders), QStringLiteral("--pipelines"),
	                         QDir::toNativeSeparators(journals.pipelines), QStringLiteral("--producer"),
	                         QStringLiteral("kyty launcher export")}),
	      "export arguments");
	const auto va = VerifyArguments(partial, game, game + QStringLiteral("/other.elf"), journals);
	Check(va == QStringList({QStringLiteral("verify"), QDir::toNativeSeparators(partial), QStringLiteral("--game"),
	                         QDir::toNativeSeparators(game), QStringLiteral("--exe"),
	                         QDir::toNativeSeparators(game + QStringLiteral("/other.elf")), QStringLiteral("--shaders"),
	                         QDir::toNativeSeparators(journals.shaders)}),
	      "verify arguments with another executable");

	// The export's counts (kyty_pipeline_list's Write line).
	Written written;
	Check(!ParseWritten(QStringLiteral("PPSA00001 01.018.000, executable 1 bytes\nlocated 0 of 0\n"), &written),
	      "no wrote line");
	Check(ParseWritten(QStringLiteral("run 0: 3 sources\r\nwrote C:\\x y\\l.kyplst.partial: 171538 bytes, 1173 shaders, "
	                                  "1322 permutations, 5253 pipelines, 6 levels\r\n"),
	                   &written) &&
	          written.shaders == 1173 && written.permutations == 1322 && written.pipelines == 5253,
	      "the counts of the wrote line");
	Check(ParseWritten(QStringLiteral("wrote a: 10 bytes, 0 shaders, 0 permutations, 0 pipelines, 0 levels\n"), &written) &&
	          written.shaders == 0,
	      "an empty list");

	// Keep: the checked list replaces an older one; the partial file never stays.
	Write(partial, "new list");
	QString error;
	Check(Keep(partial, out, &error) && Read(out) == "new list" && !QFileInfo::exists(partial), "the list is kept");
	Write(partial, "newer list");
	Check(Keep(partial, out, &error) && Read(out) == "newer list" && !QFileInfo::exists(partial), "an older list is replaced");
	// A target that cannot be replaced (a folder of that name): the partial file is removed.
	const auto blocked = base.filePath(QStringLiteral("blocked.kyplst"));
	Check(QDir().mkpath(QDir(blocked).filePath(QStringLiteral("inside"))), "cannot create a folder");
	Write(PartialPath(blocked), "list");
	Check(!Keep(PartialPath(blocked), blocked, &error) && !error.isEmpty() && !QFileInfo::exists(PartialPath(blocked)),
	      "a list that cannot be placed leaves no partial file");

	// Tail: the last lines with text.
	Check(Tail(QStringLiteral("a\n\n b \r\nc\nd\n"), 2) == QStringLiteral("c\nd"), "the last two lines");
	Check(Tail(QStringLiteral("only"), 6) == QStringLiteral("only"), "fewer lines than asked");

	std::printf("LauncherShaderListExportTests: ok\n");
	return 0;
}
