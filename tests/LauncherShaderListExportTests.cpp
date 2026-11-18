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
	// A session whose shaders all came from an installed list writes pipelines only.
	Write(pipes, "journal");
	const auto pipelines_only = FindJournals(emulator, title);
	Check(!pipelines_only.IsEmpty() && pipelines_only.shaders.isEmpty() && pipelines_only.pipelines == QDir::cleanPath(pipes),
	      "a pipeline journal alone is found");
	Check(QFile::remove(pipes), "cannot remove a file");
	Write(shaders, "journal");
	Write(pipes, QByteArray());
	const auto shaders_only = FindJournals(emulator, title);
	Check(!shaders_only.IsEmpty() && shaders_only.shaders == QDir::cleanPath(shaders) && shaders_only.pipelines.isEmpty(),
	      "an empty pipeline journal is not a journal");
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
	const auto ea      = ExportArguments(partial, game, {}, journals, {});
	Check(ea == QStringList({QStringLiteral("export"), QStringLiteral("--out"), QDir::toNativeSeparators(partial),
	                         QStringLiteral("--game"), QDir::toNativeSeparators(game), QStringLiteral("--shaders"),
	                         QDir::toNativeSeparators(journals.shaders), QStringLiteral("--pipelines"),
	                         QDir::toNativeSeparators(journals.pipelines), QStringLiteral("--producer"),
	                         QStringLiteral("kyty launcher export"), QStringLiteral("--scan")}),
	      "export arguments: the data files are searched");
	// With the list the sessions ran with, and a session that wrote pipelines only.
	const auto list = QDir(emulator).filePath(QStringLiteral("PipelineLists/PPSA00001.kyplst"));
	const auto eb   = ExportArguments(partial, game, {}, pipelines_only, list);
	Check(eb.mid(5) == QStringList({QStringLiteral("--shaders"), QStringLiteral("none"), QStringLiteral("--pipelines"),
	                                QDir::toNativeSeparators(pipelines_only.pipelines), QStringLiteral("--producer"),
	                                QStringLiteral("kyty launcher export"), QStringLiteral("--scan"), QStringLiteral("--base"),
	                                QDir::toNativeSeparators(list)}),
	      "export arguments with a base list and no shader journal");
	Check(VerifyArguments(partial, game, {}, pipelines_only) ==
	          QStringList({QStringLiteral("verify"), QDir::toNativeSeparators(partial), QStringLiteral("--game"),
	                       QDir::toNativeSeparators(game)}),
	      "verify arguments without a shader journal");

	// The base list: the one the emulator loads for the game.
	const auto version = QStringLiteral("01.018.000");
	Check(BaseList(emulator, title, version, {}).isEmpty(), "no list installed");
	Write(list, "list");
	Check(BaseList(emulator, title, version, {}) == QDir::cleanPath(list), "the game's list");
	Check(BaseList(emulator, title, version, QStringLiteral("1")) == QDir::cleanPath(list), "KYTY_PIPELINE_LIST=1 is that list too");
	const auto versioned = QDir(emulator).filePath(QStringLiteral("PipelineLists/PPSA00001_01.018.000.kyplst"));
	Write(versioned, "list");
	Check(BaseList(emulator, title, version, {}) == QDir::cleanPath(versioned), "the list of the game's version first");
	Check(BaseList(emulator, title, version, QStringLiteral("0")).isEmpty(), "lists off");
	Write(QDir(emulator).filePath(QStringLiteral("my/own.kyplst")), "list");
	Check(BaseList(emulator, title, version, QStringLiteral("my/own.kyplst")) ==
	          QDir::cleanPath(QDir(emulator).filePath(QStringLiteral("my/own.kyplst"))),
	      "a list KYTY_PIPELINE_LIST names, from the emulator's folder");
	Check(BaseList(emulator, title, version, QStringLiteral("missing.kyplst")).isEmpty(), "a named list that is not there");
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

	// What the sessions added to the base list, and the file search's progress lines.
	Written added;
	Check(!ParseAdded(QStringLiteral("wrote a: 10 bytes, 1 shaders, 1 permutations, 1 pipelines, 0 levels\n"), &added),
	      "no base list: no line");
	Check(ParseAdded(QStringLiteral("  pipelines 857: 43 new\r\nnew since the base list: 2 shaders, 3 permutations, 43 pipelines\r\n"
	                                "wrote a: 10 bytes, 1050 shaders, 1120 permutations, 1076 pipelines, 0 levels\r\n"),
	                 &added) &&
	          added.shaders == 2 && added.permutations == 3 && added.pipelines == 43,
	      "the counts of the new line");
	Scan scan;
	Check(!ParseScan(QStringLiteral("scan: located 3 in d/material"), &scan), "a line about one file is not progress");
	Check(!ParseScan(QStringLiteral("scan: located 2 of 2 (2 of 203 files, 0.3 GB read)"), &scan), "nor is the last line");
	Check(ParseScan(QStringLiteral("scan: 150 of 200 files, 12.3 of 94.9 GB read, 640 of 869 codes located\r"), &scan) &&
	          scan.files == 150 && scan.files_total == 200 && scan.located == 640 && scan.wanted == 869 && scan.read_gb > 12.29 &&
	          scan.read_gb < 12.31 && scan.total_gb > 94.89 && scan.total_gb < 94.91,
	      "a progress line");
	Check(!ParseScan(QStringLiteral("scan: 12.3 of 94.9 GB read, 640 of 869 codes located"), &scan), "a line without the files");
	// Three quarters of the files and 13% of the bytes: the mean of the two.
	Check(ScanPermille(scan) == 439, "the part of the search done");
	Check(ScanText(scan).contains(QStringLiteral("150 of 200 files, 12.3 of 94.9 GB")) && ScanText(scan).contains(QStringLiteral("869")),
	      "the progress text");
	Check(ScanPermille(Scan {}) == 0 && ScanPermille(Scan {7, 7, 5.0, 5.0, 1, 1}) == 999 &&
	          ScanPermille(Scan {100000, 200000, 0.0, 88.0, 0, 9}) == 250,
	      "the ends of the range, and many small files first");

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
