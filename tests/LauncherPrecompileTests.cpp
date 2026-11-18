// The launcher's Precompile button without its windows (precompile.h): where a game's shader list
// is looked for and installed, and the lines of the emulator's prepare run.
#include "precompile.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QProcessEnvironment>
#include <QTemporaryDir>

#include <cstdio>
#include <cstdlib>

namespace {
void Check(bool condition, const char* text) {
	if (!condition) {
		std::fprintf(stderr, "LauncherPrecompileTests: failed: %s\n", text);
		std::abort();
	}
}

void WriteFile(const QString& path, const QByteArray& bytes) {
	QFile file(path);
	Check(file.open(QIODevice::WriteOnly) && file.write(bytes) == bytes.size(), "cannot write a test file");
}

QByteArray ReadFile(const QString& path) {
	QFile file(path);
	return file.open(QIODevice::ReadOnly) ? file.readAll() : QByteArray();
}

const QByteArray g_list = QByteArray("KYPLST1\0", 8) + QByteArray("\x02\0\0\0 rest of a list", 20);

void ListPaths() {
	QTemporaryDir emulator;
	Check(emulator.isValid(), "cannot create a temporary directory");
	const auto dir       = emulator.path();
	const auto title     = QStringLiteral("PPSA00001");
	const auto version   = QStringLiteral("01.018.000");
	const auto versioned = QDir::cleanPath(dir + QStringLiteral("/PipelineLists/PPSA00001_01.018.000.kyplst"));
	const auto plain     = QDir::cleanPath(dir + QStringLiteral("/PipelineLists/PPSA00001.kyplst"));

	Check(Precompile::FindList(dir, title, version).isEmpty(), "no PipelineLists folder: no list");
	Check(Precompile::InstallPath(dir, title, version) == versioned, "a list is installed under the versioned name");
	Check(Precompile::InstallPath(dir, title, {}) == plain, "without a version the title names the list");

	Check(QDir(dir).mkpath(QStringLiteral("PipelineLists")), "cannot create PipelineLists");
	Check(Precompile::FindList(dir, title, version).isEmpty(), "an empty PipelineLists folder: no list");
	WriteFile(plain, g_list);
	Check(Precompile::FindList(dir, title, version) == plain, "the title's list is the fallback");
	Check(Precompile::FindList(dir, title, {}) == plain, "a game without a version uses the title's list");
	WriteFile(versioned, g_list);
	Check(Precompile::FindList(dir, title, version) == versioned, "the versioned list comes first");
	Check(Precompile::FindList(dir, title, QStringLiteral("01.019.000")) == plain,
	      "another version falls back to the title's list");
	Check(Precompile::FindList(dir, QStringLiteral("PPSA00002"), version).isEmpty(), "another title has no list");
	Check(QDir(dir).mkpath(QStringLiteral("PipelineLists/PPSA00003.kyplst")) &&
	          Precompile::FindList(dir, QStringLiteral("PPSA00003"), {}).isEmpty(),
	      "a folder is not a list");

	// Names that cannot go into a file name.
	Check(Precompile::IsValidGame(title, version) && Precompile::IsValidGame(title, {}), "ordinary names");
	Check(!Precompile::IsValidGame({}, version) && !Precompile::IsValidGame(QStringLiteral("../PPSA00001"), version) &&
	          !Precompile::IsValidGame(title, QStringLiteral("1/2")) &&
	          !Precompile::IsValidGame(QStringLiteral("PPSA 00001"), version),
	      "names with separators or spaces are refused");
	Check(Precompile::FindList(dir, QStringLiteral("../PPSA00001"), version).isEmpty() &&
	          Precompile::InstallPath(dir, {}, version).isEmpty(),
	      "a refused game has no list and no place for one");

	Check(!Precompile::IsCustomList({}) && !Precompile::IsCustomList(QStringLiteral("0")) &&
	          !Precompile::IsCustomList(QStringLiteral("1")) &&
	          Precompile::IsCustomList(QStringLiteral("lists/mine.kyplst")),
	      "KYTY_PIPELINE_LIST names a file only with another value than 0 and 1");
}

void Install() {
	QTemporaryDir folder;
	Check(folder.isValid(), "cannot create a temporary directory");
	const auto source = folder.filePath(QStringLiteral("download.kyplst"));
	const auto other  = folder.filePath(QStringLiteral("other.kyplst"));
	const auto text   = folder.filePath(QStringLiteral("notes.kyplst"));
	const auto target = Precompile::InstallPath(folder.filePath(QStringLiteral("emulator")), QStringLiteral("PPSA00001"),
	                                            QStringLiteral("01.000.000"));
	WriteFile(source, g_list);
	WriteFile(other, g_list + QByteArray("more"));
	WriteFile(text, QByteArray("KYPLST2 not a list"));

	QString error;
	Check(Precompile::HasListMagic(g_list) && !Precompile::HasListMagic(g_list.left(7)) &&
	          !Precompile::HasListMagic(QByteArray("KYPLST1 ", 8)) && !Precompile::HasListMagic({}),
	      "the magic is eight bytes with the terminator");
	Check(Precompile::IsListFile(source, &error) && error.isEmpty(), "a list file is accepted");
	Check(!Precompile::IsListFile(text, &error) && error.contains(QStringLiteral("not a shader list")),
	      "another file is refused with a reason");
	error.clear();
	Check(!Precompile::IsListFile(folder.filePath(QStringLiteral("missing.kyplst")), &error) && !error.isEmpty(),
	      "a missing file is refused with a reason");

	error.clear();
	Check(Precompile::InstallList(source, target, false, &error) && error.isEmpty(), "install into a new folder");
	Check(ReadFile(target) == g_list && ReadFile(source) == g_list, "the list is copied, the source stays");
	Check(!Precompile::InstallList(other, target, false, &error) && error.contains(QStringLiteral("already exists")) &&
	          ReadFile(target) == g_list,
	      "an installed list is not replaced without the flag");
	error.clear();
	Check(Precompile::InstallList(other, target, true, &error) && ReadFile(target) == g_list + QByteArray("more"),
	      "replace an installed list");
	Check(!QFileInfo::exists(target + QStringLiteral(".new")), "no temporary copy is left");
	Check(Precompile::InstallList(target, target, false, &error) && ReadFile(target) == g_list + QByteArray("more"),
	      "choosing the installed file itself keeps it");
	Check(!Precompile::InstallList(folder.filePath(QStringLiteral("missing.kyplst")), target, true, &error) &&
	          ReadFile(target) == g_list + QByteArray("more"),
	      "a failed copy keeps the installed list");
	// A read-only download must stay replaceable once installed.
	Check(QFile::setPermissions(source, QFile::ReadOwner | QFile::ReadUser), "cannot make the source read-only");
	Check(Precompile::InstallList(source, target, true, &error) && Precompile::InstallList(other, target, true, &error) &&
	          ReadFile(target) == g_list + QByteArray("more"),
	      "a list installed from a read-only file can be replaced");
	QFile::setPermissions(source, QFile::ReadOwner | QFile::WriteOwner | QFile::ReadUser | QFile::WriteUser);
}

void ProgressLines() {
	Precompile::Progress p;
	Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress frame=0 t=12.345 phase=shaders "
	                                               "shaders=120/1069 pipelines=0/2680 compute=3/7"),
	                                &p),
	      "a progress line");
	Check(p.phase == QStringLiteral("shaders") && p.shaders_done == 120 && p.shaders_total == 1069 &&
	          p.pipelines_done == 0 && p.pipelines_total == 2680 && p.compute_done == 3 && p.compute_total == 7,
	      "the progress counts");
	Check(Precompile::ProgressPermille(p) == 120 * 150 / 1069,
	      "the shader step fills the first part of the bar by the count the text shows");
	Check(Precompile::ProgressText(p) == QStringLiteral("Step 1 of 3, building shaders: 120 of 1069"),
	      "the shader phase text");

	// A log prefix before the event, "\r" after it, the fields in another order, no compute field.
	Check(Precompile::ParseProgress(QStringLiteral("[12:00:01] KYTY_EVENT: prepare-progress pipelines=2680/2680 "
	                                               "shaders=1069/1069 phase=saving\r"),
	                                &p),
	      "a progress line with a prefix");
	Check(p.phase == QStringLiteral("saving") && p.compute_total == 0 && Precompile::ProgressPermille(p) == 1000 &&
	          Precompile::ProgressText(p) == QStringLiteral("Saving the caches..."),
	      "everything done");
	Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress phase=pipelines shaders=10/10 "
	                                               "pipelines=5/20 compute=0/0"),
	                                &p) &&
	          Precompile::ProgressText(p) == QStringLiteral("Step 2 of 3, building pipelines: 5 of 20") &&
	          Precompile::ProgressPermille(p) == 150 + 5 * 750 / 20,
	      "the pipeline phase");
	Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress phase=compute shaders=10/10 "
	                                               "pipelines=20/20 compute=1/4"),
	                                &p) &&
	          Precompile::ProgressText(p) == QStringLiteral("Step 3 of 3, building compute pipelines: 1 of 4") &&
	          Precompile::ProgressPermille(p) == 900 + 1 * 90 / 4,
	      "the compute phase");
	// A count beyond its total (a total read a moment earlier) never overfills the bar.
	Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress shaders=12/10 pipelines=0/10"), &p) &&
	          Precompile::ProgressPermille(p) == 150 &&
	          Precompile::ProgressText(p) == QStringLiteral("Step 2 of 3, building pipelines: 0 of 10"),
	      "counts are clamped and the phase follows from them");
	Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress shaders=0/0 pipelines=0/0"), &p) &&
	          Precompile::ProgressPermille(p) == 0,
	      "nothing counted yet");

	// The bar never steps back across the steps of a run and ends full.
	int last = 0;
	for (const char* line : {"phase=shaders shaders=0/10 pipelines=0/20 compute=0/0",
	                         "phase=shaders shaders=10/10 pipelines=0/20 compute=0/0",
	                         "phase=pipelines shaders=10/10 pipelines=10/20 compute=2/4",
	                         "phase=pipelines shaders=10/10 pipelines=20/20 compute=4/4",
	                         "phase=compute shaders=10/10 pipelines=20/20 compute=2/4",
	                         "phase=compute shaders=10/10 pipelines=20/20 compute=4/4",
	                         "phase=saving shaders=10/10 pipelines=20/20 compute=4/4"}) {
		Check(Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress ") + QLatin1String(line), &p) &&
		          Precompile::ProgressPermille(p) >= last,
		      "the bar does not step back between the steps");
		last = Precompile::ProgressPermille(p);
	}
	Check(last == 1000, "the bar ends full");

	const Precompile::Progress before = p;
	Check(!Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: loading-start frame=0 t=1.000"), &p) &&
	          !Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progressive shaders=1/2 pipelines=1/2"), &p) &&
	          !Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress shaders=1 pipelines=1/2"), &p) &&
	          !Precompile::ParseProgress(QStringLiteral("KYTY_EVENT: prepare-progress shaders=a/b pipelines=1/2"), &p) &&
	          !Precompile::ParseProgress(QStringLiteral("Pipeline replay: 5/20 (25%)"), &p) &&
	          !Precompile::ParseProgress({}, &p),
	      "other lines are not progress");
	Check(p.shaders_total == before.shaders_total && p.pipelines_total == before.pipelines_total,
	      "a refused line leaves the last progress");
}

void DoneLines() {
	Precompile::Result r;
	Check(Precompile::ParseDone(QStringLiteral("KYTY_EVENT: prepare-done frame=0 t=80.100 seconds=75.3 shaders=1069 "
	                                           "pipelines=2680 known=0 compute=12"),
	                            &r),
	      "the done event");
	Check(r.seconds > 75.29 && r.seconds < 75.31 && r.shaders == 1069 && r.pipelines == 2680 && r.known == 0 &&
	          r.compute == 12,
	      "the done counts");
	Check(Precompile::DoneText(r) == QStringLiteral("Done in 75 s: 1069 shaders, 2680 pipelines, 12 compute pipelines"),
	      "the result text");

	Check(Precompile::ParseDone(QStringLiteral("KYTY_EVENT: prepare-done seconds=4.26 shaders=1069 pipelines=0 "
	                                           "known=2680 compute=0"),
	                            &r) &&
	          Precompile::DoneText(r) ==
	              QStringLiteral("Done in 4.3 s: 1069 shaders, 0 pipelines (2680 more pipelines were in the cache already)"),
	      "a run on a warm cache");

	// An emulator without the event: its sentence.
	Check(Precompile::ParseDone(
	          QStringLiteral("Prepare shaders: done in 61.0 s (shaders 20.5 s, pipelines and saving 40.5 s): 1069 "
	                         "permutations replayed, 2680 graphics pipelines built, 3 compute pipelines prewarmed; driver "
	                         "cache 123456789 bytes"),
	          &r) &&
	          r.shaders == 1069 && r.pipelines == 2680 && r.compute == 3 && r.known == 0 && r.seconds > 60.9 &&
	          r.seconds < 61.1,
	      "the done sentence");

	Check(!Precompile::ParseDone(QStringLiteral("KYTY_EVENT: prepare-done seconds=x shaders=1 pipelines=1"), &r) &&
	          !Precompile::ParseDone(QStringLiteral("KYTY_EVENT: prepare-done seconds=1.0 pipelines=1"), &r) &&
	          !Precompile::ParseDone(QStringLiteral("Prepare shaders: building the pipeline list's shaders and pipelines"),
	                                 &r) &&
	          !Precompile::ParseDone(QStringLiteral("KYTY_EVENT: prepare-progress shaders=1/1 pipelines=1/1"), &r),
	      "other lines are not the end");

	Check(Precompile::ListLine(QStringLiteral("[info] Pipeline list: PipelineLists\\PPSA00001_01.000.000.kyplst is for "
	                                          "PPSA00002 version 01.000.000, the game is PPSA00001 version "
	                                          "01.000.000: ignored\r")) ==
	          QStringLiteral("Pipeline list: PipelineLists\\PPSA00001_01.000.000.kyplst is for PPSA00002 version "
	                         "01.000.000, the game is PPSA00001 version 01.000.000: ignored"),
	      "the emulator's reason, without the log prefix");
	Check(Precompile::ListLine(QStringLiteral("Prepare shaders: no usable pipeline list (KYTY_PIPELINE_LIST); nothing "
	                                          "to do"))
	          .isEmpty() &&
	          Precompile::ListLine(QStringLiteral("Pipeline replay: 5/20")).isEmpty(),
	      "other lines say nothing about the list");
}
} // namespace

// The prepare run's environment: the prepare switch, and the replay on over a preset's 0.
void PrepareRunEnvironment() {
	QProcessEnvironment plain;
	plain.insert(QStringLiteral("KYTY_OTHER"), QStringLiteral("x"));
	Precompile::PrepareEnvironment(&plain);
	Check(plain.value(QStringLiteral("KYTY_PIPELINE_LIST_PREPARE")) == QStringLiteral("1") &&
	          plain.value(QStringLiteral("KYTY_SHADER_PRECOMPILE_REPLAY")) == QStringLiteral("1") &&
	          plain.value(QStringLiteral("KYTY_OTHER")) == QStringLiteral("x"),
	      "the prepare switch and the replay are set, other variables stay");
	QProcessEnvironment preset;
	preset.insert(QStringLiteral("KYTY_SHADER_PRECOMPILE_REPLAY"), QStringLiteral("0"));
	Precompile::PrepareEnvironment(&preset);
	Check(preset.value(QStringLiteral("KYTY_SHADER_PRECOMPILE_REPLAY")) == QStringLiteral("1"),
	      "a preset that keeps the replay off for play does not switch it off for the prepare run");
}

int main() {
	PrepareRunEnvironment();
	ListPaths();
	Install();
	ProgressLines();
	DoneLines();
	std::printf("LauncherPrecompileTests: ok\n");
	return 0;
}
