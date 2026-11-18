#include "shaderListExport.h"

#include "precompile.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>

namespace ShaderListExport {

namespace {

#if defined(_WIN32)
constexpr char TOOL_NAME[] = "kyty_pipeline_list.exe";
#else
constexpr char TOOL_NAME[] = "kyty_pipeline_list";
#endif

QString Native(const QString& path) {
	return QDir::toNativeSeparators(path);
}

void AddGame(QStringList* args, const QString& game, const QString& executable) {
	*args << QStringLiteral("--game") << Native(game);
	if (!executable.isEmpty()) {
		*args << QStringLiteral("--exe") << Native(executable);
	}
}

} // namespace

QString FindTool(const QString& launcher_dir) {
	QDir dir(launcher_dir);
	for (int up = 0; up < 2; up++) {
		const auto path = dir.absoluteFilePath(QLatin1String(TOOL_NAME));
		if (QFileInfo(path).isFile()) {
			return QDir::cleanPath(path);
		}
		if (!dir.cdUp()) {
			break;
		}
	}
	return {};
}

Journals FindJournals(const QString& emulator_dir, const QString& title_id) {
	if (!Precompile::IsValidGame(title_id, {})) {
		return {};
	}
	const QDir      cache(QDir(emulator_dir).filePath(QStringLiteral("_PipelineCache")));
	const QFileInfo shaders(cache.filePath(title_id + QStringLiteral(".shaders.journal")));
	const QFileInfo pipelines(cache.filePath(title_id + QStringLiteral(".pipelines.journal")));
	const auto found = [](const QFileInfo& file) {
		return file.isFile() && file.size() != 0 ? QDir::cleanPath(file.absoluteFilePath()) : QString();
	};
	return {found(shaders), found(pipelines)};
}

QString BaseList(const QString& emulator_dir, const QString& title_id, const QString& version, const QString& setting) {
	if (setting == QLatin1String("0")) {
		return {};
	}
	if (Precompile::IsCustomList(setting)) {
		// As the emulator opens it: from its own folder (its working directory).
		const QFileInfo file(QDir(emulator_dir), setting);
		return file.isFile() ? QDir::cleanPath(file.absoluteFilePath()) : QString();
	}
	return Precompile::FindList(emulator_dir, title_id, version);
}

QString DefaultOutput(const QString& emulator_dir, const QString& title_id, const QString& version) {
	if (!Precompile::IsValidGame(title_id, version)) {
		return {};
	}
	const auto name = version.isEmpty() ? title_id : title_id + QLatin1Char('-') + version;
	return QDir::cleanPath(QDir(QDir(emulator_dir).filePath(QLatin1String(OUTPUT_DIR)))
	                           .filePath(name + QStringLiteral("-shaderlist.kyplst")));
}

QString PartialPath(const QString& out) {
	return out + QStringLiteral(".partial");
}

QStringList ExportArguments(const QString& partial, const QString& game, const QString& executable,
                            const Journals& journals, const QString& base_list) {
	// The tool takes the journals in pairs; "none" stands for the one a session did not write.
	const auto journal = [](const QString& path) { return path.isEmpty() ? QStringLiteral("none") : Native(path); };
	QStringList args {QStringLiteral("export"), QStringLiteral("--out"), Native(partial)};
	AddGame(&args, game, executable);
	args << QStringLiteral("--shaders") << journal(journals.shaders) << QStringLiteral("--pipelines")
	     << journal(journals.pipelines) << QStringLiteral("--producer") << QLatin1String(PRODUCER)
	     << QStringLiteral("--scan");
	if (!base_list.isEmpty()) {
		args << QStringLiteral("--base") << Native(base_list);
	}
	return args;
}

QStringList VerifyArguments(const QString& partial, const QString& game, const QString& executable,
                            const Journals& journals) {
	QStringList args {QStringLiteral("verify"), Native(partial)};
	AddGame(&args, game, executable);
	if (!journals.shaders.isEmpty()) {
		args << QStringLiteral("--shaders") << Native(journals.shaders);
	}
	return args;
}

bool ParseWritten(const QString& output, Written* written) {
	static const QRegularExpression line(
	    QStringLiteral("^wrote .*: [0-9]+ bytes, ([0-9]+) shaders, ([0-9]+) permutations, ([0-9]+) pipelines"),
	    QRegularExpression::MultilineOption);
	QRegularExpressionMatch last;
	for (auto it = line.globalMatch(output); it.hasNext();) {
		last = it.next();
	}
	if (!last.hasMatch()) {
		return false;
	}
	written->shaders      = last.captured(1).toULongLong();
	written->permutations = last.captured(2).toULongLong();
	written->pipelines    = last.captured(3).toULongLong();
	return true;
}

bool ParseAdded(const QString& output, Written* added) {
	static const QRegularExpression line(
	    QStringLiteral("^new since the base list: ([0-9]+) shaders, ([0-9]+) permutations, ([0-9]+) pipelines"),
	    QRegularExpression::MultilineOption);
	QRegularExpressionMatch last;
	for (auto it = line.globalMatch(output); it.hasNext();) {
		last = it.next();
	}
	if (!last.hasMatch()) {
		return false;
	}
	added->shaders      = last.captured(1).toULongLong();
	added->permutations = last.captured(2).toULongLong();
	added->pipelines    = last.captured(3).toULongLong();
	return true;
}

bool ParseScan(const QString& line, Scan* scan) {
	static const QRegularExpression pattern(QStringLiteral(
	    "^scan: ([0-9]+) of ([0-9]+) files, ([0-9]+(?:\\.[0-9]+)?) of ([0-9]+(?:\\.[0-9]+)?) GB read, "
	    "([0-9]+) of ([0-9]+) codes located"));
	const auto match = pattern.match(line.trimmed());
	if (!match.hasMatch()) {
		return false;
	}
	scan->files       = match.captured(1).toULongLong();
	scan->files_total = match.captured(2).toULongLong();
	scan->read_gb     = match.captured(3).toDouble();
	scan->total_gb    = match.captured(4).toDouble();
	scan->located     = match.captured(5).toULongLong();
	scan->wanted      = match.captured(6).toULongLong();
	return true;
}

int ScanPermille(const Scan& scan) {
	const double bytes = scan.total_gb > 0.0 ? scan.read_gb / scan.total_gb : 0.0;
	const double files = scan.files_total != 0 ? static_cast<double>(scan.files) / static_cast<double>(scan.files_total) : 0.0;
	return qBound(0, static_cast<int>((bytes + files) * 500.0), 999);
}

QString ScanText(const Scan& scan) {
	return QStringLiteral("Searching the game's files for %1 shaders that are not in its executable:\n"
	                      "%5 of %6 files, %2 of %3 GB read, %4 found. It stops when all are found.")
	    .arg(scan.wanted)
	    .arg(scan.read_gb, 0, 'f', 1)
	    .arg(scan.total_gb, 0, 'f', 1)
	    .arg(scan.located)
	    .arg(scan.files)
	    .arg(scan.files_total);
}

bool Keep(const QString& partial, const QString& out, QString* error) {
	// The list was written next to its place: replacing the old one is a remove and a rename.
	if (QFileInfo::exists(out) && !QFile::remove(out)) {
		QFile::remove(partial);
		*error = QStringLiteral("%1 exists and cannot be replaced.").arg(Native(out));
		return false;
	}
	if (!QFile::rename(partial, out)) {
		QFile::remove(partial);
		*error = QStringLiteral("Could not save %1.").arg(Native(out));
		return false;
	}
	return true;
}

QString Tail(const QString& output, int count) {
	QStringList lines;
	for (const auto& line: output.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts)) {
		if (!line.trimmed().isEmpty()) {
			lines.append(line.trimmed());
		}
	}
	if (lines.size() > count) {
		lines = lines.mid(lines.size() - count);
	}
	return lines.join(QLatin1Char('\n'));
}

} // namespace ShaderListExport
