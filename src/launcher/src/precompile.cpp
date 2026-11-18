#include "precompile.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QRegularExpression>
#include <QStringView>

#include <algorithm>
#include <cstring>

namespace Precompile {

namespace {

constexpr char LIST_DIR[]    = "PipelineLists";
constexpr char LIST_SUFFIX[] = ".kyplst";
// PipelineList::FileMagic (pipelineList.h).
constexpr char LIST_MAGIC[8] = {'K', 'Y', 'P', 'L', 'S', 'T', '1', '\0'};

// The "key=value" fields after an event's name; false when the line is not that event.
bool EventFields(const QString& line, const char* name, QHash<QString, QString>* fields) {
	const auto marker = QStringLiteral("KYTY_EVENT: ") + QLatin1String(name);
	const auto at     = line.indexOf(marker);
	if (at < 0) {
		return false;
	}
	const auto rest = QStringView(line).mid(at + marker.size());
	if (!rest.isEmpty() && !rest.front().isSpace()) {
		return false;
	}
	for (const auto field: rest.split(u' ', Qt::SkipEmptyParts)) {
		const auto equals = field.indexOf(u'=');
		if (equals > 0) {
			fields->insert(field.left(equals).toString(), field.mid(equals + 1).trimmed().toString());
		}
	}
	return true;
}

bool ParseCount(const QString& text, quint64* count) {
	bool ok = false;
	*count  = text.toULongLong(&ok);
	return ok;
}

// "<done>/<total>"
bool ParsePair(const QString& text, quint64* done, quint64* total) {
	const auto slash = text.indexOf(u'/');
	return slash > 0 && ParseCount(text.left(slash), done) && ParseCount(text.mid(slash + 1), total);
}

} // namespace

bool IsValidGame(const QString& title_id, const QString& version) {
	static const QRegularExpression title_pattern(QStringLiteral("\\A[A-Za-z0-9_-]+\\z"));
	static const QRegularExpression version_pattern(QStringLiteral("\\A[A-Za-z0-9._-]*\\z"));
	return title_pattern.match(title_id).hasMatch() && version_pattern.match(version).hasMatch();
}

QString FindList(const QString& emulator_dir, const QString& title_id, const QString& version) {
	if (!IsValidGame(title_id, version)) {
		return {};
	}
	const QDir  lists(QDir(emulator_dir).filePath(QLatin1String(LIST_DIR)));
	QStringList names;
	if (!version.isEmpty()) {
		names << title_id + QLatin1Char('_') + version + QLatin1String(LIST_SUFFIX);
	}
	names << title_id + QLatin1String(LIST_SUFFIX);
	for (const auto& name: names) {
		const auto path = lists.filePath(name);
		if (QFileInfo(path).isFile()) {
			return QDir::cleanPath(path);
		}
	}
	return {};
}

QString InstallPath(const QString& emulator_dir, const QString& title_id, const QString& version) {
	if (!IsValidGame(title_id, version)) {
		return {};
	}
	const auto name = version.isEmpty() ? title_id : title_id + QLatin1Char('_') + version;
	return QDir::cleanPath(QDir(QDir(emulator_dir).filePath(QLatin1String(LIST_DIR)))
	                           .filePath(name + QLatin1String(LIST_SUFFIX)));
}

void PrepareEnvironment(QProcessEnvironment* environment) {
	environment->insert(QLatin1String(ENV_PREPARE), QStringLiteral("1"));
	environment->insert(QLatin1String(ENV_REPLAY), QStringLiteral("1"));
}

bool IsCustomList(const QString& setting) {
	return !setting.isEmpty() && setting != QLatin1String("0") && setting != QLatin1String("1");
}

bool HasListMagic(const QByteArray& head) {
	return head.size() >= static_cast<qsizetype>(sizeof(LIST_MAGIC)) &&
	       std::memcmp(head.constData(), LIST_MAGIC, sizeof(LIST_MAGIC)) == 0;
}

bool IsListFile(const QString& path, QString* error) {
	QFile file(path);
	if (!file.open(QIODevice::ReadOnly)) {
		*error = QStringLiteral("%1 cannot be opened: %2")
		             .arg(QDir::toNativeSeparators(path), file.errorString());
		return false;
	}
	if (!HasListMagic(file.read(sizeof(LIST_MAGIC)))) {
		*error = QStringLiteral("%1 is not a shader list (.kyplst) file.")
		             .arg(QDir::toNativeSeparators(path));
		return false;
	}
	return true;
}

bool InstallList(const QString& source, const QString& target, bool replace, QString* error) {
	const QFileInfo target_info(target);
	const auto      native_target = QDir::toNativeSeparators(target);
	if (const auto existing = target_info.canonicalFilePath();
	    !existing.isEmpty() && existing == QFileInfo(source).canonicalFilePath()) {
		return true; // chosen where it belongs
	}
	if (target_info.exists() && !replace) {
		*error = QStringLiteral("%1 already exists.").arg(native_target);
		return false;
	}
	if (!QDir().mkpath(target_info.absolutePath())) {
		*error = QStringLiteral("Could not create the folder %1.")
		             .arg(QDir::toNativeSeparators(target_info.absolutePath()));
		return false;
	}
	// Copied next to its place first: a copy that fails leaves the old list as it was.
	const auto copy = target + QStringLiteral(".new");
	QFile::remove(copy);
	if (!QFile::copy(source, copy)) {
		*error = QStringLiteral("Could not copy %1 to %2.")
		             .arg(QDir::toNativeSeparators(source), native_target);
		return false;
	}
	// A read-only source must not leave a list that cannot be replaced later.
	QFile::setPermissions(copy, QFile::permissions(copy) | QFile::WriteOwner | QFile::WriteUser);
	if ((target_info.exists() && !QFile::remove(target)) || !QFile::rename(copy, target)) {
		QFile::remove(copy);
		*error = QStringLiteral("Could not replace %1.").arg(native_target);
		return false;
	}
	return true;
}

bool ParseProgress(const QString& line, Progress* progress) {
	QHash<QString, QString> fields;
	Progress                parsed;
	if (!EventFields(line, "prepare-progress", &fields) ||
	    !ParsePair(fields.value(QStringLiteral("shaders")), &parsed.shaders_done,
	               &parsed.shaders_total) ||
	    !ParsePair(fields.value(QStringLiteral("pipelines")), &parsed.pipelines_done,
	               &parsed.pipelines_total)) {
		return false;
	}
	// Not in every line of every emulator: an absent count is none.
	if (!ParsePair(fields.value(QStringLiteral("compute")), &parsed.compute_done,
	               &parsed.compute_total)) {
		parsed.compute_done = parsed.compute_total = 0;
	}
	parsed.phase = fields.value(QStringLiteral("phase"));
	*progress    = parsed;
	return true;
}

namespace {

enum class Step { Shaders, Pipelines, Compute, Saving };

// The step the run is in: the emulator's phase, else the first one whose counts are not complete.
Step CurrentStep(const Progress& progress) {
	if (progress.phase == QLatin1String("saving")) {
		return Step::Saving;
	}
	if (progress.phase == QLatin1String("compute")) {
		return Step::Compute;
	}
	if (progress.phase == QLatin1String("pipelines") ||
	    (progress.phase.isEmpty() && progress.shaders_total != 0 && progress.shaders_done >= progress.shaders_total)) {
		return Step::Pipelines;
	}
	return Step::Shaders;
}

} // namespace

int ProgressPermille(const Progress& progress) {
	// The bar covers the whole run and the text names the step: each step fills its own part of the
	// bar with the count the text shows. The parts follow where a run spends its time when the
	// driver has no cached pipelines (shaders about 10%, graphics pipelines most of the rest); with a
	// warm driver cache the later steps just go by quicker.
	constexpr int shaders_end   = 150;
	constexpr int pipelines_end = 900;
	constexpr int compute_end   = 990;
	const auto    part = [](quint64 done, quint64 total, int from, int to) {
		if (total == 0) {
			return from;
		}
		return from + static_cast<int>(std::min(done, total) * static_cast<quint64>(to - from) / total);
	};
	switch (CurrentStep(progress)) {
		case Step::Shaders: return part(progress.shaders_done, progress.shaders_total, 0, shaders_end);
		case Step::Pipelines:
			return part(progress.pipelines_done, progress.pipelines_total, shaders_end, pipelines_end);
		case Step::Compute: return part(progress.compute_done, progress.compute_total, pipelines_end, compute_end);
		case Step::Saving: break;
	}
	return 1000;
}

QString ProgressText(const Progress& progress) {
	const auto counts = [](int step, const char* what, quint64 done, quint64 total) {
		return QStringLiteral("Step %1 of 3, building %2: %3 of %4")
		    .arg(step)
		    .arg(QLatin1String(what))
		    .arg(done)
		    .arg(total);
	};
	switch (CurrentStep(progress)) {
		case Step::Saving: return QStringLiteral("Saving the caches...");
		case Step::Compute: return counts(3, "compute pipelines", progress.compute_done, progress.compute_total);
		case Step::Pipelines: return counts(2, "pipelines", progress.pipelines_done, progress.pipelines_total);
		case Step::Shaders: break;
	}
	return counts(1, "shaders", progress.shaders_done, progress.shaders_total);
}

bool ParseDone(const QString& line, Result* result) {
	QHash<QString, QString> fields;
	Result                  parsed;
	if (EventFields(line, "prepare-done", &fields)) {
		bool seconds_ok = false;
		parsed.seconds  = fields.value(QStringLiteral("seconds")).toDouble(&seconds_ok);
		if (!seconds_ok || !ParseCount(fields.value(QStringLiteral("shaders")), &parsed.shaders) ||
		    !ParseCount(fields.value(QStringLiteral("pipelines")), &parsed.pipelines)) {
			return false;
		}
		if (!ParseCount(fields.value(QStringLiteral("known")), &parsed.known)) {
			parsed.known = 0;
		}
		if (!ParseCount(fields.value(QStringLiteral("compute")), &parsed.compute)) {
			parsed.compute = 0;
		}
		*result = parsed;
		return true;
	}
	static const QRegularExpression done_line(
	    QStringLiteral("Prepare shaders: done in ([0-9.]+) s[^:]*: ([0-9]+) permutations replayed, "
	                   "([0-9]+) graphics pipelines built, ([0-9]+) compute pipelines prewarmed"));
	const auto match = done_line.match(line);
	if (!match.hasMatch()) {
		return false;
	}
	parsed.seconds   = match.captured(1).toDouble();
	parsed.shaders   = match.captured(2).toULongLong();
	parsed.pipelines = match.captured(3).toULongLong();
	parsed.compute   = match.captured(4).toULongLong();
	*result          = parsed;
	return true;
}

QString DoneText(const Result& result) {
	auto text = QStringLiteral("Done in %1 s: %2 shaders, %3 pipelines")
	                .arg(QString::number(result.seconds, 'f', result.seconds < 10.0 ? 1 : 0))
	                .arg(result.shaders)
	                .arg(result.pipelines);
	if (result.compute != 0) {
		text += QStringLiteral(", %1 compute pipelines").arg(result.compute);
	}
	if (result.known != 0) {
		text += QStringLiteral(" (%1 more pipelines were in the cache already)").arg(result.known);
	}
	return text;
}

QString ListLine(const QString& line) {
	const auto at = line.indexOf(QLatin1String("Pipeline list: "));
	return at < 0 ? QString() : line.mid(at).trimmed();
}

} // namespace Precompile
