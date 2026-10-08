// kyty_pipeline_list: makes and checks KYPLST1 pipeline lists (pipelineList.h, KYTY_PIPELINE_LIST).
//
//   export --out <list> --game <app0 dir> (--shaders <shader journal> --pipelines <pipeline journal>)...
//          [--exe <executable>] [--producer <text>]
//       Merges recorded journals (_PipelineCache/<title>.shaders.journal and .pipelines.journal of one
//       or more recording runs) into a list. Shaders are located in the game executable; the list
//       keeps their offsets and hashes, never their code.
//   merge --out <list> <list>...        Union of lists of the same title, version and input layout.
//   info <list>                         Header, counts, levels, section sizes.
//   check <list> --game <app0 dir> [--exe <executable>]
//       Resolves the list against a game copy, as the emulator does at start-up.
//   verify <list> --game <app0 dir> [--exe <executable>] [--shaders <shader journal>]...
//       The no-code proof: no 64-byte window of the listed shaders' code (read from the executable)
//       or of any journaled shader's code occurs in the file or in any decompressed section.
//       Exit code 1 when one does.
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <xxhash.h>

using namespace Libs::Graphics;

namespace {

struct Arguments {
	std::string                                      command;
	std::string                                      out, game, exe, producer = "kyty_pipeline_list";
	std::vector<std::string>                         shaders, pipelines, inputs;
};

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& bytes) {
	std::ifstream   in(path, std::ios::binary);
	std::error_code error;
	const auto      size = std::filesystem::file_size(path, error);
	if (!in || error) return false;
	bytes.resize(static_cast<size_t>(size));
	return bytes.empty() || static_cast<bool>(in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)));
}

bool WriteFile(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	return static_cast<bool>(out);
}

// Title and version as the emulator reads them (loader/systemContent.cpp: titleId, contentVersion).
bool ReadParam(const std::filesystem::path& game, std::string& title, std::string& version) {
	std::vector<uint8_t> bytes;
	if (!ReadFile(game / "sce_sys" / "param.json", bytes)) return false;
	const auto json = nlohmann::json::parse(bytes.begin(), bytes.end(), nullptr, false);
	if (json.is_discarded() || !json.is_object()) return false;
	if (json.contains("titleId") && json["titleId"].is_string()) title = json["titleId"].get<std::string>();
	if (json.contains("contentVersion") && json["contentVersion"].is_string()) version = json["contentVersion"].get<std::string>();
	if (json.contains("appVersion") && json["appVersion"].is_string()) version = json["appVersion"].get<std::string>();
	return !title.empty() && !version.empty();
}

std::filesystem::path Executable(const Arguments& args) {
	return args.exe.empty() ? std::filesystem::path(args.game) / "eboot.bin" : std::filesystem::path(args.exe);
}

// "kyty shader journal; device ...; input infos A B C" -> {A, B, C}.
bool InputSizes(const std::vector<uint8_t>& identity, std::array<uint32_t, 3>& sizes) {
	const std::string text(identity.begin(), identity.end());
	const auto        at = text.find("input infos ");
	if (at == std::string::npos) return false;
	const char* cursor = text.c_str() + at + 12;
	for (auto& size: sizes) {
		char* end = nullptr;
		size      = static_cast<uint32_t>(std::strtoul(cursor, &end, 10));
		if (end == cursor) return false;
		cursor = end;
	}
	return true;
}

uint64_t NowUnix() {
	return static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

void PrintExport(const PipelineList::ExportStats& s) {
	std::printf("  sources %llu: %llu new, %llu duplicate, %llu not found in the executable\n",
	            static_cast<unsigned long long>(s.sources), static_cast<unsigned long long>(s.sources_located),
	            static_cast<unsigned long long>(s.sources_duplicate), static_cast<unsigned long long>(s.sources_unlocated));
	std::printf("  permutations %llu: %llu new, %llu dropped with their source\n",
	            static_cast<unsigned long long>(s.entries), static_cast<unsigned long long>(s.entries_kept),
	            static_cast<unsigned long long>(s.entries_dropped));
	std::printf("  pipelines %llu: %llu new, %llu merged, %llu with a stage not in the shader journal, %llu malformed\n",
	            static_cast<unsigned long long>(s.pipelines), static_cast<unsigned long long>(s.pipelines_kept),
	            static_cast<unsigned long long>(s.pipelines_merged), static_cast<unsigned long long>(s.pipelines_unresolved),
	            static_cast<unsigned long long>(s.pipelines_bad));
}

int Write(const PipelineList::File& file, const std::string& out) {
	const auto bytes = PipelineList::Encode(file);
	if (!WriteFile(out, bytes)) {
		std::fprintf(stderr, "cannot write %s\n", out.c_str());
		return 1;
	}
	std::printf("wrote %s: %zu bytes, %zu shaders, %zu permutations, %zu pipelines, %zu levels\n", out.c_str(), bytes.size(),
	            file.shaders.size(), file.permutations.size(), file.pipelines.size(), file.levels.size());
	return 0;
}

int Export(const Arguments& args) {
	if (args.out.empty() || args.game.empty() || args.shaders.empty() || args.shaders.size() != args.pipelines.size()) {
		std::fprintf(stderr, "export needs --out, --game and pairs of --shaders/--pipelines\n");
		return 2;
	}
	PipelineList::Info info;
	if (!ReadParam(args.game, info.title_id, info.app_version)) {
		std::fprintf(stderr, "cannot read the title and version from %s/sce_sys/param.json\n", args.game.c_str());
		return 2;
	}
	info.producer     = args.producer;
	info.created_unix = NowUnix();
	std::vector<uint8_t> image;
	if (!ReadFile(Executable(args), image)) {
		std::fprintf(stderr, "cannot read %s\n", Executable(args).string().c_str());
		return 2;
	}
	std::printf("%s %s, executable %zu bytes\n", info.title_id.c_str(), info.app_version.c_str(), image.size());
	// The journals; every source's code located in one pass over the executable.
	struct Run {
		std::unique_ptr<ShaderJournal>   shaders;
		std::unique_ptr<PipelineJournal> pipelines;
	};
	std::vector<Run> runs;
	bool             have_sizes = false;
	for (size_t i = 0; i < args.shaders.size(); i++) {
		std::vector<uint8_t> shader_identity, pipeline_identity;
		if (!ShaderJournal::ReadIdentity(args.shaders[i], shader_identity) ||
		    !PipelineJournal::ReadIdentity(args.pipelines[i], pipeline_identity)) {
			std::fprintf(stderr, "%s or %s is not a journal\n", args.shaders[i].c_str(), args.pipelines[i].c_str());
			return 2;
		}
		std::array<uint32_t, 3> sizes {};
		if (!InputSizes(shader_identity, sizes)) {
			std::fprintf(stderr, "%s names no input info sizes\n", args.shaders[i].c_str());
			return 2;
		}
		if (have_sizes && sizes != info.input_info_sizes) {
			std::fprintf(stderr, "%s was recorded by an emulator with other input infos\n", args.shaders[i].c_str());
			return 2;
		}
		info.input_info_sizes = sizes;
		have_sizes            = true;
		Run run;
		ShaderJournal::Settings shader_settings;
		shader_settings.path              = args.shaders[i];
		shader_settings.identity          = shader_identity;
		shader_settings.background_writer = false;
		run.shaders                       = std::make_unique<ShaderJournal>(std::move(shader_settings));
		PipelineJournal::Settings pipeline_settings;
		pipeline_settings.path              = args.pipelines[i];
		pipeline_settings.identity          = pipeline_identity;
		pipeline_settings.background_writer = false;
		pipeline_settings.sealed_writes     = true; // never writes
		run.pipelines                       = std::make_unique<PipelineJournal>(std::move(pipeline_settings));
		std::printf("run %zu: %zu sources, %zu permutations, %zu pipelines\n", i, run.shaders->Sources().size(),
		            run.shaders->Entries().size(), run.pipelines->Loaded().size());
		runs.push_back(std::move(run));
	}
	std::vector<PipelineList::CodeQuery> queries;
	std::map<std::pair<uint64_t, uint64_t>, size_t> query_of;
	for (const auto& run: runs) {
		for (const auto& source: run.shaders->Sources()) {
			const std::span bytes(reinterpret_cast<const uint8_t*>(source.code.data()), source.code.size() * 4);
			const auto      query = PipelineList::QueryFor(bytes);
			if (query_of.try_emplace({query.hash.low, query.hash.high}, queries.size()).second) queries.push_back(query);
		}
	}
	const auto found = PipelineList::FindCode(image, queries);
	size_t     located = 0;
	for (const auto& offset: found) located += offset.has_value() ? 1 : 0;
	std::printf("located %zu of %zu distinct shader codes in the executable\n", located, queries.size());
	const PipelineList::Builder::LocateFn locate = [&](std::span<const uint32_t> code) -> std::optional<uint64_t> {
		const auto hash = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(code.data()), code.size_bytes()});
		const auto it   = query_of.find({hash.low, hash.high});
		return it == query_of.end() ? std::nullopt : found[it->second];
	};
	PipelineList::Builder     builder(info);
	PipelineList::ExportStats stats;
	for (const auto& run: runs) {
		builder.AddJournals(run.shaders->Sources(), run.shaders->Entries(), run.pipelines->Loaded(), locate, stats);
	}
	PrintExport(stats);
	return Write(builder.Finish(), args.out);
}

bool Load(const std::string& path, PipelineList::File& file) {
	std::vector<uint8_t> bytes;
	std::string          error;
	if (!ReadFile(path, bytes)) {
		std::fprintf(stderr, "cannot read %s\n", path.c_str());
		return false;
	}
	if (!PipelineList::Decode(bytes, file, &error)) {
		std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
		return false;
	}
	return true;
}

int Merge(const Arguments& args) {
	if (args.out.empty() || args.inputs.empty()) {
		std::fprintf(stderr, "merge needs --out and input lists\n");
		return 2;
	}
	std::unique_ptr<PipelineList::Builder> builder;
	PipelineList::Info                     info;
	PipelineList::ExportStats              stats;
	for (const auto& input: args.inputs) {
		PipelineList::File file;
		if (!Load(input, file)) return 2;
		if (builder == nullptr) {
			info              = file.info;
			info.producer     = args.producer;
			info.created_unix = NowUnix();
			builder           = std::make_unique<PipelineList::Builder>(info);
		} else if (file.info.title_id != info.title_id || file.info.app_version != info.app_version ||
		           file.info.input_info_sizes != info.input_info_sizes) {
			std::fprintf(stderr, "%s is for another title, version or input layout\n", input.c_str());
			return 2;
		}
		builder->AddList(file, stats);
	}
	PrintExport(stats);
	return Write(builder->Finish(), args.out);
}

int Info(const Arguments& args) {
	if (args.inputs.size() != 1) return 2;
	std::vector<uint8_t> bytes;
	if (!ReadFile(args.inputs[0], bytes)) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	std::printf("%s: %zu bytes\n  title %s, version %s, producer %s, input infos %u %u %u, created %llu\n",
	            args.inputs[0].c_str(), bytes.size(), file.info.title_id.c_str(), file.info.app_version.c_str(),
	            file.info.producer.c_str(), file.info.input_info_sizes[0], file.info.input_info_sizes[1],
	            file.info.input_info_sizes[2], static_cast<unsigned long long>(file.info.created_unix));
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	(void)PipelineList::DecodeSections(bytes, sections);
	for (const auto& [id, raw]: sections) std::printf("  section %u: %zu raw bytes\n", id, raw.size());
	size_t kinds[3] = {}, code_bytes = 0;
	for (const auto& shader: file.shaders) {
		kinds[static_cast<size_t>(shader.kind)]++;
		code_bytes += shader.code_size * 4u;
	}
	size_t mesh = 0, stages[3] = {};
	std::vector<size_t> per_level(file.levels.size());
	for (const auto& pipeline: file.pipelines) {
		mesh += pipeline.words.size() > 1 && pipeline.words[1] != 0 ? 1 : 0;
		stages[std::min<size_t>(2, pipeline.stages.size())]++;
		for (const auto level: pipeline.levels) per_level[level]++;
	}
	std::printf("  shaders %zu (vertex %zu, pixel %zu, compute %zu; %zu code bytes referenced)\n", file.shaders.size(),
	            kinds[0], kinds[1], kinds[2], code_bytes);
	std::printf("  permutations %zu, pipelines %zu (%zu mesh, %zu without a pixel stage)\n", file.permutations.size(),
	            file.pipelines.size(), mesh, stages[1]);
	for (size_t i = 0; i < file.levels.size(); i++) std::printf("  level %s: %zu pipelines\n", file.levels[i].c_str(), per_level[i]);
	return 0;
}

PipelineList::Resolved ResolveAgainst(const Arguments& args, const PipelineList::File& file, std::vector<uint8_t>& image,
                                      PipelineList::ResolveStats& stats) {
	if (!ReadFile(Executable(args), image)) {
		std::fprintf(stderr, "cannot read %s\n", Executable(args).string().c_str());
		image.clear();
	}
	const PipelineList::ReadCodeFn read = [&](PipelineList::CodeFile, uint64_t offset, std::span<uint8_t> out) {
		if (offset > image.size() || out.size() > image.size() - offset) return false;
		std::memcpy(out.data(), image.data() + offset, out.size());
		return true;
	};
	return PipelineList::Resolve(file, file.info.input_info_sizes, read, [&](PipelineList::CodeFile) { return image; }, stats);
}

int Check(const Arguments& args) {
	if (args.inputs.size() != 1 || args.game.empty()) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	std::string title, version;
	if (!ReadParam(args.game, title, version)) return 2;
	std::printf("list for %s %s; game %s %s%s\n", file.info.title_id.c_str(), file.info.app_version.c_str(), title.c_str(),
	            version.c_str(), title == file.info.title_id && version == file.info.app_version ? "" : ": MISMATCH (ignored at run time)");
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats stats;
	const auto                 resolved = ResolveAgainst(args, file, image, stats);
	(void)resolved;
	std::printf("shaders %llu: %llu match (%llu at another offset), %llu differ, %llu unreadable; permutations %llu of %llu; "
	            "pipelines %llu of %llu usable\n",
	            static_cast<unsigned long long>(stats.shaders), static_cast<unsigned long long>(stats.resolved),
	            static_cast<unsigned long long>(stats.relocated), static_cast<unsigned long long>(stats.mismatched),
	            static_cast<unsigned long long>(stats.unreadable), static_cast<unsigned long long>(stats.permutations_kept),
	            static_cast<unsigned long long>(stats.permutations), static_cast<unsigned long long>(stats.pipelines_kept),
	            static_cast<unsigned long long>(stats.pipelines));
	return stats.resolved == stats.shaders ? 0 : 1;
}

int Verify(const Arguments& args) {
	if (args.inputs.size() != 1 || args.game.empty()) return 2;
	std::vector<uint8_t> bytes;
	PipelineList::File   file;
	if (!ReadFile(args.inputs[0], bytes) || !Load(args.inputs[0], file)) return 2;
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats stats;
	const auto                 resolved = ResolveAgainst(args, file, image, stats);
	PipelineList::CodeWindows  windows;
	uint64_t                   code_bytes = 0;
	for (size_t i = 0; i < resolved.sources.size(); i++) {
		if (!resolved.usable[i]) continue;
		const auto& code = resolved.sources[i].code;
		windows.Add({reinterpret_cast<const uint8_t*>(code.data()), code.size() * 4});
		code_bytes += code.size() * 4;
	}
	for (const auto& path: args.shaders) {
		std::vector<uint8_t> identity;
		if (!ShaderJournal::ReadIdentity(path, identity)) {
			std::fprintf(stderr, "%s is not a shader journal\n", path.c_str());
			return 2;
		}
		ShaderJournal::Settings settings;
		settings.path              = path;
		settings.identity          = identity;
		settings.background_writer = false;
		ShaderJournal journal(std::move(settings));
		for (const auto& source: journal.Sources()) {
			windows.Add({reinterpret_cast<const uint8_t*>(source.code.data()), source.code.size() * 4});
			code_bytes += source.code.size() * 4;
		}
	}
	std::printf("known code: %llu bytes of %llu listed shaders (%llu resolved) and %zu journal(s): %zu windows (%llu "
	            "low-entropy windows skipped)\n",
	            static_cast<unsigned long long>(code_bytes), static_cast<unsigned long long>(stats.shaders),
	            static_cast<unsigned long long>(stats.resolved), args.shaders.size(), windows.Count(),
	            static_cast<unsigned long long>(windows.SkippedLowEntropy()));
	size_t     hits = 0;
	const auto scan = [&](const char* what, std::span<const uint8_t> data) {
		const auto found = windows.FindIn(data);
		std::printf("  %s: %zu bytes, %zu code windows%s\n", what, data.size(), found.size(), found.empty() ? "" : " FOUND");
		for (const auto offset: found) std::printf("    at %llu\n", static_cast<unsigned long long>(offset));
		hits += found.size();
	};
	scan("file", bytes);
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	if (!PipelineList::DecodeSections(bytes, sections)) return 2;
	for (const auto& [id, raw]: sections) {
		const auto name = "section " + std::to_string(id);
		scan(name.c_str(), raw);
	}
	std::printf("%s\n", hits == 0 ? "verify: no shader code in the list" : "verify: FAILED, the list contains shader code");
	return hits == 0 && stats.resolved != 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
	Arguments args;
	if (argc < 2) {
		std::fprintf(stderr, "usage: kyty_pipeline_list export|merge|info|check|verify ... (see the source header)\n");
		return 2;
	}
	args.command = argv[1];
	for (int i = 2; i < argc; i++) {
		const std::string arg   = argv[i];
		const auto        value = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
		if (arg == "--out") {
			args.out = value();
		} else if (arg == "--game") {
			args.game = value();
		} else if (arg == "--exe") {
			args.exe = value();
		} else if (arg == "--producer") {
			args.producer = value();
		} else if (arg == "--shaders") {
			args.shaders.push_back(value());
		} else if (arg == "--pipelines") {
			args.pipelines.push_back(value());
		} else {
			args.inputs.push_back(arg);
		}
	}
	if (args.command == "export") return Export(args);
	if (args.command == "merge") return Merge(args);
	if (args.command == "info") return Info(args);
	if (args.command == "check") return Check(args);
	if (args.command == "verify") return Verify(args);
	std::fprintf(stderr, "unknown command %s\n", args.command.c_str());
	return 2;
}
