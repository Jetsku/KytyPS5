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
	if (!shaders.isFile() || !pipelines.isFile() || shaders.size() == 0 || pipelines.size() == 0) {
		return {};
	}
	return {QDir::cleanPath(shaders.absoluteFilePath()), QDir::cleanPath(pipelines.absoluteFilePath())};
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
                            const Journals& journals) {
	QStringList args {QStringLiteral("export"), QStringLiteral("--out"), Native(partial)};
	AddGame(&args, game, executable);
	args << QStringLiteral("--shaders") << Native(journals.shaders) << QStringLiteral("--pipelines")
	     << Native(journals.pipelines) << QStringLiteral("--producer") << QLatin1String(PRODUCER);
	return args;
}

QStringList VerifyArguments(const QString& partial, const QString& game, const QString& executable,
                            const Journals& journals) {
	QStringList args {QStringLiteral("verify"), Native(partial)};
	AddGame(&args, game, executable);
	args << QStringLiteral("--shaders") << Native(journals.shaders);
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
