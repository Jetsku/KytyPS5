// kyty_pipeline_list: makes and checks KYPLST1 pipeline lists (pipelineList.h, KYTY_PIPELINE_LIST).
//
//   export --out <list> --game <app0 dir or .zar> (--shaders <shader journal> --pipelines <pipeline journal>)...
//          [--exe <executable>] [--producer <text>] [--data <file relative to app0>]... [--scan]
//       Merges recorded journals (_PipelineCache/<title>.shaders.journal and .pipelines.journal of one
//       or more recording runs) into a list. Shaders are located in the game executable, else in the
//       --data files (Insomniac titles: d/material, d/materialgraph); the list keeps their file,
//       offsets and hashes, never their code.
//       --scan: the codes still not found are looked for in every other file of the game folder by
//       their footer (adapted from Erlou93/KytyPS5-Yotei's shader hints): PS5 shader binaries end in a
//       48-byte block that starts with "barefoot"; every such block in a file is a candidate end of a
//       code, which is kept when the bytes before it have the code's size and hashes. The files that
//       hold codes join the list's data files, so no --data names are needed (codes without the footer
//       still need --exe/--data).
//       --base <list>: the list the recording sessions ran with (the one the emulator loaded for the
//       game). The emulator journals only what it compiled itself, so the journals of a session with
//       a list lack the list's shaders, and a recorded pipeline with a stage from the list could not
//       be kept. The base list is resolved against the game and its usable part joins the export; a
//       list of the int21 release is upgraded as the emulator upgrades it and gives its shaders and
//       permutations only. With --scan its data files and their folders are read first.
//   merge --out <list> <list>...        Union of lists of the same title, version and input layout.
//   info <list>                         Header, counts, levels, section sizes.
//   check <list> --game <app0 dir or .zar> [--exe <executable>]
//       Resolves the list against a game copy, as the emulator does at start-up.
//   retarget --list <list> --exe <executable> --title <id> --app-ver <version> --out <list>
//       Re-stamps a list for another copy of the game (another version or SKU): each shader is looked up in
//       the other executable (at its recorded offset, then by hash), shaders not found are dropped with
//       their pipelines, the title and version are replaced. Prints the match counts.
//   upgrade <list> --out <list>
//       Rewrites the vertex stage input infos of a list recorded before the mesh pass and
//       tessellation clamp fields (stored size 16 bytes smaller) into this build's layout, with those
//       fields at their defaults; other input infos must already match. A list of the int21 release
//       (its layout is older still) is refused with a note: the emulator upgrades it when it loads it.
//   verify <list> --game <app0 dir or .zar> [--exe <executable>] [--shaders <shader journal>]...
//       The no-code proof: no 64-byte window of the listed shaders' code (read from the executable)
//       or of any journaled shader's code occurs in the file or in any decompressed section.
//       Exit code 1 when one does.
// --game is the game folder (the one with eboot.bin and sce_sys) or a .zar archive of it (the
// launcher's "Compress game"); every command reads the same files from either, and an export of a
// folder and of its archive writes the same list. SOURCE_DATE_EPOCH (seconds), when set, is the
// creation time written into lists instead of the current time.
#include "common/stringUtils.h"
#include "exportBase.h"
#include "gameFiles.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineListInt21.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <xxhash.h>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")
#endif

using namespace Libs::Graphics;

namespace {

struct Arguments {
	std::string                                      command;
	std::string                                      out, game, exe, list, title, app_ver, producer = "kyty_pipeline_list";
	std::vector<std::string>                         shaders, pipelines, inputs, data;
	bool                                             scan = false;
	std::string                                      base;
};

// --scan: the footer every PS5 shader binary ends with (48 bytes, starting with "barefoot").
constexpr size_t FooterBytes   = 48;
constexpr char   FooterText[8] = {'b', 'a', 'r', 'e', 'f', 'o', 'o', 't'};

// XXH3-64 of a code's footer (| 1, never 0); 0 when the code does not end with one.
uint64_t FooterHash(std::span<const uint8_t> code) {
	if (code.size() < FooterBytes) return 0;
	const auto* footer = code.data() + code.size() - FooterBytes;
	if (std::memcmp(footer, FooterText, sizeof(FooterText)) != 0) return 0;
	return XXH3_64bits(footer, FooterBytes) | 1u;
}

using PipelineListTool::GameFiles;

// Calls fn(file offset of the byte after the footer, bytes ending there (at most max_before + footer))
// for every footer in the file, reading it in 64 MiB chunks. False when the file cannot be read.
template <typename Fn>
bool ForEachFooter(const GameFiles::Stream& read, size_t max_before, Fn&& fn) {
	if (!read) return false;
	constexpr size_t     Chunk = size_t {64} << 20u;
	std::vector<uint8_t> buffer;
	std::vector<uint8_t> chunk(Chunk);
	uint64_t             buffer_start = 0; // file offset of buffer[0]
	size_t               searched     = 0; // buffer positions below this were searched already
	for (;;) {
		const auto got = read(chunk.data(), chunk.size());
		if (got == 0) break;
		buffer.insert(buffer.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(got));
		size_t pos = searched;
		while (pos + FooterBytes <= buffer.size()) {
			const auto* hit = static_cast<const uint8_t*>(std::memchr(buffer.data() + pos, 'b', buffer.size() - FooterBytes + 1 - pos));
			if (hit == nullptr) break;
			pos = static_cast<size_t>(hit - buffer.data());
			if (std::memcmp(hit, FooterText, sizeof(FooterText)) == 0) {
				const size_t end   = pos + FooterBytes;
				const size_t begin = end > max_before + FooterBytes ? end - max_before - FooterBytes : 0;
				fn(buffer_start + end, std::span<const uint8_t>(buffer.data() + begin, end - begin));
			}
			pos++;
		}
		searched = buffer.size() >= FooterBytes ? buffer.size() - FooterBytes + 1 : 0;
		// Keep what a code ending in the next chunk may start in, and the unsearched tail.
		const size_t keep = std::min(buffer.size(), max_before + FooterBytes);
		const size_t drop = buffer.size() - keep;
		buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(drop));
		buffer_start += drop;
		searched = searched > drop ? searched - drop : 0;
		if (got < chunk.size()) break;
	}
	return true;
}

// Arguments and list file names are UTF-8; paths are built from them as such (characters outside the code
// page survive; adapted from Erlou93/KytyPS5-Yotei d5b030b).
std::filesystem::path U8(const std::string& text) {
	return Common::PathFromUtf8(text);
}

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

// The game folder or archive of --game; null (with a message) when an archive cannot be opened.
std::unique_ptr<GameFiles> OpenGame(const Arguments& args) {
	auto game = GameFiles::Open(U8(args.game));
	if (game == nullptr) std::fprintf(stderr, "cannot open the archive %s\n", args.game.c_str());
	return game;
}

// --exe, else the game's eboot.bin.
std::string ExecutableName(const Arguments& args) {
	return args.exe.empty() ? args.game + "/eboot.bin" : args.exe;
}

bool ReadExecutable(const Arguments& args, GameFiles& game, std::vector<uint8_t>& image) {
	return args.exe.empty() ? game.Read("eboot.bin", image) : ReadFile(U8(args.exe), image);
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
	// Reproducible lists (the same bytes from a folder and its archive): SOURCE_DATE_EPOCH.
	if (const char* fixed = std::getenv("SOURCE_DATE_EPOCH"); fixed != nullptr && *fixed != '\0') {
		char*      end   = nullptr;
		const auto value = std::strtoull(fixed, &end, 10);
		if (end != fixed && *end == '\0') return static_cast<uint64_t>(value);
	}
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
	if (!WriteFile(U8(out), bytes)) {
		std::fprintf(stderr, "cannot write %s\n", out.c_str());
		return 1;
	}
	std::printf("wrote %s: %zu bytes, %zu shaders, %zu permutations, %zu pipelines, %zu levels\n", out.c_str(), bytes.size(),
	            file.shaders.size(), file.permutations.size(), file.pipelines.size(), file.levels.size());
	return 0;
}

bool                   Load(const std::string& path, PipelineList::File& file);
PipelineList::Resolved ResolveWith(GameFiles& game, const PipelineList::File& file, const std::vector<uint8_t>& image,
                                   const std::array<uint32_t, 3>& sizes, PipelineList::ResolveStats& stats);

// export --base: the list the recording sessions ran with (the launcher passes the one the emulator
// loads for the game). The emulator journals what it compiles itself: a permutation it took from an
// installed list is in no journal, while a recorded pipeline names its stages by their permutation
// identities, so without that list every pipeline with such a stage would be dropped (a session with a
// complete list would export nothing). The list is resolved against the game as the emulator resolves
// it and, when it is of the int21 release, upgraded as the emulator upgrades it; its usable shaders and
// permutations then join the export as if they had been journaled, which gives them the identities the
// emulator gave them. Its pipelines join too, unless they were recorded with the int21 layout or
// another pipeline state format (the emulator cannot replay most of those).
struct BaseList {
	bool               used = false;
	PipelineList::File kept; // the part usable with this game, at the offsets found in it
	// The kept shaders as journal sources (code filled) and the kept permutations as journal entries.
	std::vector<ShaderJournal::Source> sources;
	std::vector<ShaderJournal::Entry>  entries;
	// Where a code (or a merged stage's back half) sits, by its hash: the data file's name (empty: the
	// executable) and the offset.
	std::map<std::pair<uint64_t, uint64_t>, std::pair<std::string, uint64_t>> where;
};

void LoadBase(const Arguments& args, GameFiles& game, const PipelineList::Info& info, const std::vector<uint8_t>& image,
              BaseList& base) {
	const auto unused = [&](const char* why) { std::printf("base list %s: not used, %s\n", args.base.c_str(), why); };
	PipelineList::File file;
	if (!Load(args.base, file)) return unused("it cannot be read");
	if (file.info.title_id != info.title_id) return unused("it is for another title");
	uint64_t upgraded_infos = 0, upgraded_specializations = 0;
	bool     int21          = false;
	if (file.info.input_info_sizes != info.input_info_sizes) {
		const auto layout = Int21VertexLayout();
		if (info.input_info_sizes[0] != ShaderStoredInputInfoSize<ShaderVertexInputInfo>() ||
		    file.info.input_info_sizes[0] != layout.from_size || file.info.input_info_sizes[1] != info.input_info_sizes[1] ||
		    file.info.input_info_sizes[2] != info.input_info_sizes[2]) {
			return unused("its input layout is neither the journals' nor the int21 release's");
		}
		PipelineList::UpgradeStats upgrade;
		PipelineList::UpgradeVertexInputInfos(file, layout, info.input_info_sizes[0], upgrade);
		upgraded_infos = upgrade.upgraded;
		for (auto& permutation: file.permutations) {
			upgraded_specializations += PipelineListTool::UpgradeInt21Specialization(permutation.specialization) ? 1 : 0;
		}
		int21 = true;
	}
	const auto listed_shaders   = file.shaders.size();
	const auto listed_pipelines = file.pipelines.size();
	const bool other_state      = file.info.state_format != PipelineList::StateFormat;
	if (int21 || other_state) file.pipelines.clear();
	PipelineList::ResolveStats  resolve;
	auto                        resolved = ResolveWith(game, file, image, info.input_info_sizes, resolve);
	PipelineList::RetargetStats retarget;
	base.kept       = PipelineList::Retarget(file, resolved, retarget);
	base.kept.files = file.files; // Retarget keeps every shader's file number
	const auto key = [](const std::vector<uint32_t>& code) {
		const auto hash = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(code.data()), code.size() * 4});
		return std::pair {hash.low, hash.high};
	};
	std::vector<uint32_t> source_of(file.shaders.size(), UINT32_MAX);
	for (size_t i = 0; i < file.shaders.size(); i++) {
		if (!resolved.usable[i]) continue;
		const auto index  = static_cast<uint32_t>(file.shaders[i].file);
		const auto name   = index == 0 ? std::string() : file.files[index - 1];
		auto&      source = resolved.sources[i];
		base.where.try_emplace(key(source.code), name, resolved.offsets[i]);
		if (!source.back_code.empty()) base.where.try_emplace(key(source.back_code), name, resolved.back_offsets[i]);
		source_of[i] = static_cast<uint32_t>(base.sources.size());
		base.sources.push_back(std::move(source));
	}
	for (const auto& entry: resolved.entries) {
		base.entries.push_back({source_of[entry.source], entry.push_data_cursor, entry.specialization});
	}
	base.used = !base.sources.empty();
	std::printf("base list %s: %s %s by %s: %zu of %zu shaders found in this game, %zu permutations, %zu of %zu pipelines kept\n",
	            args.base.c_str(), file.info.title_id.c_str(), file.info.app_version.c_str(), file.info.producer.c_str(),
	            base.kept.shaders.size(), listed_shaders, base.kept.permutations.size(), base.kept.pipelines.size(), listed_pipelines);
	if (int21) {
		std::printf("  int21 layout: %llu vertex input infos and %llu specializations upgraded as the emulator upgrades them; its "
		            "pipelines were recorded with that layout and are left out\n",
		            static_cast<unsigned long long>(upgraded_infos), static_cast<unsigned long long>(upgraded_specializations));
	} else if (other_state && listed_pipelines != 0) {
		std::printf("  its pipelines have pipeline state format %u (this build: %u) and are left out\n", file.info.state_format,
		            PipelineList::StateFormat);
	}
	if (!base.used) unused("none of its shaders is in this game");
}

int Export(const Arguments& args) {
	if (args.out.empty() || args.game.empty() || args.shaders.empty() || args.shaders.size() != args.pipelines.size()) {
		std::fprintf(stderr, "export needs --out, --game and pairs of --shaders/--pipelines\n");
		return 2;
	}
	const auto game = OpenGame(args);
	if (game == nullptr) return 2;
	PipelineList::Info info;
	if (!PipelineListTool::ReadParam(*game, info.title_id, info.app_version)) {
		std::fprintf(stderr, "cannot read the title and version from %s/sce_sys/param.json\n", args.game.c_str());
		return 2;
	}
	info.producer     = args.producer;
	info.created_unix = NowUnix();
	info.state_format = PipelineList::StateFormat;
	std::vector<uint8_t> image;
	if (!ReadExecutable(args, *game, image)) {
		std::fprintf(stderr, "cannot read %s\n", ExecutableName(args).c_str());
		return 2;
	}
	std::printf("%s %s, executable %zu bytes\n", info.title_id.c_str(), info.app_version.c_str(), image.size());
	// The journals; every source's code located in one pass over the executable.
	struct Run {
		std::shared_ptr<ShaderJournal>   shaders;
		std::unique_ptr<PipelineJournal> pipelines;
	};
	std::vector<Run> runs;
	// A shader journal named in several pairs (one per level's pipeline journal) is loaded once.
	std::map<std::string, std::shared_ptr<ShaderJournal>> loaded_shaders;
	bool             have_sizes = false;
	for (size_t i = 0; i < args.shaders.size(); i++) {
		std::vector<uint8_t> shader_identity, pipeline_identity;
		// "--pipelines none": a shader journal without pipelines (its shaders still precompile).
		// "--shaders none": pipelines without a shader journal (a session whose shaders all came from the
		// base list writes none); the input layout is then this build's, which wrote the pipelines.
		const bool no_pipelines = args.pipelines[i] == "none";
		const bool no_shaders   = args.shaders[i] == "none";
		if ((!no_shaders && !ShaderJournal::ReadIdentity(args.shaders[i], shader_identity)) ||
		    (!no_pipelines && !PipelineJournal::ReadIdentity(args.pipelines[i], pipeline_identity))) {
			std::fprintf(stderr, "%s or %s is not a journal\n", args.shaders[i].c_str(), args.pipelines[i].c_str());
			return 2;
		}
		std::array<uint32_t, 3> sizes {static_cast<uint32_t>(ShaderStoredInputInfoSize<ShaderVertexInputInfo>()),
		                               static_cast<uint32_t>(ShaderStoredInputInfoSize<ShaderPixelInputInfo>()),
		                               static_cast<uint32_t>(ShaderStoredInputInfoSize<ShaderComputeInputInfo>())};
		if (!no_shaders && !InputSizes(shader_identity, sizes)) {
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
		auto& shared = loaded_shaders[args.shaders[i]];
		if (shared == nullptr) {
			ShaderJournal::Settings shader_settings;
			shader_settings.path              = no_shaders ? std::filesystem::path() : U8(args.shaders[i]);
			shader_settings.identity          = shader_identity;
			shader_settings.background_writer = false;
			shared                            = std::make_shared<ShaderJournal>(std::move(shader_settings));
		}
		run.shaders = shared;
		PipelineJournal::Settings pipeline_settings;
		pipeline_settings.path              = no_pipelines ? std::filesystem::path() : U8(args.pipelines[i]);
		pipeline_settings.identity          = pipeline_identity;
		pipeline_settings.background_writer = false;
		pipeline_settings.sealed_writes     = true; // never writes
		run.pipelines                       = std::make_unique<PipelineJournal>(std::move(pipeline_settings));
		std::printf("run %zu: %zu sources, %zu permutations, %zu pipelines\n", i, run.shaders->Sources().size(),
		            run.shaders->Entries().size(), run.pipelines->Loaded().size());
		runs.push_back(std::move(run));
	}
	PipelineList::Builder builder(info);
	BaseList              base;
	if (!args.base.empty()) LoadBase(args, *game, info, image, base);
	std::vector<PipelineList::CodeQuery> queries;
	std::map<std::pair<uint64_t, uint64_t>, size_t> query_of;
	for (const auto& [path, journal]: loaded_shaders) {
		(void)path;
		for (const auto& source: journal->Sources()) {
			for (const auto* code: {&source.code, &source.back_code}) {
				if (code->empty()) continue;
				const std::span bytes(reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4);
				const auto      query = PipelineList::QueryFor(bytes);
				if (base.where.contains({query.hash.low, query.hash.high})) continue; // the base list says where it is
				if (query_of.try_emplace({query.hash.low, query.hash.high}, queries.size()).second) queries.push_back(query);
			}
		}
	}
	const auto found = PipelineList::FindCode(image, queries);
	std::vector<PipelineList::Builder::Location> where(queries.size());
	std::vector<bool>                            have(queries.size(), false);
	size_t                                       located = 0;
	for (size_t q = 0; q < found.size(); q++) {
		if (!found[q].has_value()) continue;
		where[q] = {PipelineList::CodeFile::Executable, *found[q]};
		have[q]  = true;
		located++;
	}
	std::printf("located %zu of %zu distinct shader codes in the executable\n", located, queries.size());
	// The rest in the data files, in the order given (the first file holding a code wins).
	for (const auto& data: args.data) {
		std::vector<size_t>                  rest;
		std::vector<PipelineList::CodeQuery> rest_queries;
		for (size_t q = 0; q < queries.size(); q++) {
			if (have[q]) continue;
			rest.push_back(q);
			rest_queries.push_back(queries[q]);
		}
		if (rest.empty()) break;
		std::vector<uint8_t> data_image;
		if (!game->Read(U8(data), data_image)) {
			std::fprintf(stderr, "cannot read %s in %s\n", data.c_str(), args.game.c_str());
			return 2;
		}
		const auto data_found = PipelineList::FindCode(data_image, rest_queries);
		size_t     in_file    = 0;
		for (size_t r = 0; r < rest.size(); r++) {
			if (!data_found[r].has_value()) continue;
			if (in_file == 0) (void)builder.AddFile(data);
			where[rest[r]] = {builder.AddFile(data), *data_found[r]};
			have[rest[r]]  = true;
			in_file++;
		}
		std::printf("located %zu more in %s\n", in_file, data.c_str());
	}
	if (args.scan) {
		// Wanted: footer hash -> query; the largest code bounds the bytes kept before a footer.
		std::unordered_multimap<uint64_t, size_t> wanted;
		size_t                                    max_before = 0, no_footer = 0, rest = 0, min_code = SIZE_MAX;
		std::vector<uint64_t>                     footer_of(queries.size(), 0);
		for (const auto& [path, journal]: loaded_shaders) {
			(void)path;
			for (const auto& source: journal->Sources()) {
				for (const auto* code: {&source.code, &source.back_code}) {
					if (code->empty()) continue;
					const std::span bytes(reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4);
					const auto      hash = PipelineList::CodeHash(bytes);
					if (base.where.contains({hash.low, hash.high})) continue;
					const auto      q    = query_of.at({hash.low, hash.high});
					if (have[q] || footer_of[q] != 0) continue;
					footer_of[q] = FooterHash(bytes);
					if (footer_of[q] == 0) {
						no_footer++;
						footer_of[q] = 1; // counted once; never matches a real footer hash (they are odd but not 1)
						continue;
					}
					wanted.emplace(footer_of[q], q);
					max_before = std::max<size_t>(max_before, bytes.size() - FooterBytes);
					min_code   = std::min<size_t>(min_code, bytes.size());
					rest++;
				}
			}
		}
		std::printf("scan: %zu codes not found yet have a footer, %zu have none\n", rest, no_footer);
		// eboot.bin was searched; the other files until every code is found: a base list's data files and
		// their folders first, then the rest, smaller first.
		// (Listing a game's files takes a while on its own: not when there is nothing to look for.)
		auto files = rest == 0 ? std::vector<GameFiles::Entry>() : game->Files();
		std::erase_if(files, [](const GameFiles::Entry& file) { return file.relative.filename() == "eboot.bin"; });
		if (base.used) {
			PipelineListTool::OrderForScan(files, base.kept.files,
			                               [](const GameFiles::Entry& file) { return Common::PathToGenericString(file.relative); });
		}
		uint64_t total_bytes = 0;
		for (const auto& file: files) total_bytes += file.size;
		size_t   found_total = 0, files_read = 0;
		uint64_t read_bytes  = 0;
		// A line about every two seconds while it reads (the launcher shows the last one).
		auto       reported = std::chrono::steady_clock::now();
		const auto report   = [&] {
			const auto now = std::chrono::steady_clock::now();
			if (now - reported < std::chrono::seconds(2)) return;
			reported = now;
			std::printf("scan: %zu of %zu files, %.1f of %.1f GB read, %zu of %zu codes located\n", files_read, files.size(),
			            static_cast<double>(read_bytes) / 1e9, static_cast<double>(total_bytes) / 1e9, found_total, rest);
			std::fflush(stdout);
		};
		std::fflush(stdout);
		for (const auto& entry: files) {
			if (found_total == rest) break;
			files_read++;
			if (entry.size < min_code) { // too small to hold any of the codes (a game can have 200,000 such files)
				report();
				continue;
			}
			const auto& path     = entry.relative;
			const auto  relative = Common::PathToGenericString(path);
			size_t      in_file  = 0;
			const auto        stream = game->OpenStream(path);
			GameFiles::Stream counted;
			if (stream) {
				counted = [&](uint8_t* data, size_t bytes) {
					const auto got = stream(data, bytes);
					read_bytes += got;
					report();
					return got;
				};
			}
			ForEachFooter(counted, max_before, [&](uint64_t end, std::span<const uint8_t> before) {
				const auto footer = XXH3_64bits(before.data() + before.size() - FooterBytes, FooterBytes) | 1u;
				const auto [first, last] = wanted.equal_range(footer);
				for (auto it = first; it != last; ++it) {
					const auto  q    = it->second;
					const auto& want = queries[q];
					if (have[q] || want.size_bytes > before.size()) continue;
					const auto code = before.subspan(before.size() - want.size_bytes);
					if (!(PipelineList::CodeHash(code) == want.hash)) continue;
					where[q] = {builder.AddFile(relative), end - want.size_bytes};
					have[q]  = true;
					in_file++;
					found_total++;
				}
			});
			if (in_file != 0) std::printf("scan: located %zu in %s\n", in_file, relative.c_str());
		}
		std::printf("scan: located %zu of %zu (%zu of %zu files, %.1f GB read)\n", found_total, rest, files_read, files.size(),
		            static_cast<double>(read_bytes) / 1e9);
	}
	const PipelineList::Builder::LocateFn locate =
	    [&](std::span<const uint32_t> code) -> std::optional<PipelineList::Builder::Location> {
		const auto hash = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(code.data()), code.size_bytes()});
		if (const auto known = base.where.find({hash.low, hash.high}); known != base.where.end()) {
			const auto& [name, offset] = known->second;
			return PipelineList::Builder::Location {name.empty() ? PipelineList::CodeFile::Executable : builder.AddFile(name), offset};
		}
		const auto it   = query_of.find({hash.low, hash.high});
		if (it == query_of.end() || !have[it->second]) return std::nullopt;
		return where[it->second];
	};
	PipelineList::ExportStats base_stats, stats;
	PipelineList::File        with_base;
	if (base.used) {
		builder.AddList(base.kept, base_stats);
		with_base = builder.Finish();
	}
	for (const auto& run: runs) {
		if (!base.used) {
			builder.AddJournals(run.shaders->Sources(), run.shaders->Entries(), run.pipelines->Loaded(), locate, stats);
			continue;
		}
		// The base list's shaders and permutations go in once more as journal content, in the call that
		// takes the run's pipelines: that call looks up the pipelines' stage identities.
		auto       sources = base.sources;
		auto       entries = base.entries;
		const auto first   = static_cast<uint32_t>(sources.size());
		sources.insert(sources.end(), run.shaders->Sources().begin(), run.shaders->Sources().end());
		for (const auto& entry: run.shaders->Entries()) {
			entries.push_back({entry.source + first, entry.push_data_cursor, entry.specialization});
		}
		builder.AddJournals(sources, entries, run.pipelines->Loaded(), locate, stats);
		// They are the list's (known already), not the journals': out of the journals' counts.
		stats.sources -= std::min<uint64_t>(stats.sources, base.sources.size());
		stats.sources_duplicate -= std::min<uint64_t>(stats.sources_duplicate, base.sources.size());
		stats.entries -= std::min<uint64_t>(stats.entries, base.entries.size());
	}
	PrintExport(stats);
	// A module without a permutation identity (an expanded call program, which is not journaled) cannot be
	// named in a list: its pipelines are among those not kept.
	size_t unidentified = 0;
	for (const auto& run: runs) {
		for (const auto& record: run.pipelines->Loaded()) {
			PipelineJournalPayload payload;
			if (!DecodePipelineJournalPayload(record.payload, payload)) continue;
			unidentified += std::any_of(payload.identities.begin(), payload.identities.end(),
			                            [](const auto& stage) { return stage.first == 0; })
			                   ? 1
			                   : 0;
		}
	}
	if (unidentified != 0) std::printf("  %zu recorded pipelines name a module without an identity (they cannot be listed)\n", unidentified);
	const auto file = builder.Finish();
	if (base.used) {
		std::printf("new since the base list: %zu shaders, %zu permutations, %zu pipelines\n",
		            file.shaders.size() - with_base.shaders.size(), file.permutations.size() - with_base.permutations.size(),
		            file.pipelines.size() - with_base.pipelines.size());
	}
	return Write(file, args.out);
}

bool Load(const std::string& path, PipelineList::File& file) {
	std::vector<uint8_t> bytes;
	std::string          error;
	if (!ReadFile(U8(path), bytes)) {
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
		           file.info.input_info_sizes != info.input_info_sizes || file.info.state_format != info.state_format) {
			std::fprintf(stderr, "%s is for another title, version, input layout or pipeline state format\n", input.c_str());
			return 2;
		}
		builder->AddList(file, stats);
	}
	PrintExport(stats);
	return Write(builder->Finish(), args.out);
}

// A vertex input info of the layout before ShaderMeshInputInfo::fast_launch/passes and
// ShaderTessellationInputInfo::min/max_tess_level (8 bytes each) in this build's layout.
bool UpgradeVertexInputInfo(std::vector<uint8_t>& bytes) {
	constexpr size_t now       = Libs::Graphics::ShaderStoredInputInfoSize<Libs::Graphics::ShaderVertexInputInfo>();
	constexpr size_t mesh_at   = offsetof(Libs::Graphics::ShaderVertexInputInfo, mesh) +
	                           offsetof(Libs::Graphics::ShaderMeshInputInfo, fast_launch);
	constexpr size_t mesh_size = offsetof(Libs::Graphics::ShaderMeshInputInfo, passes) + sizeof(uint32_t) -
	                             offsetof(Libs::Graphics::ShaderMeshInputInfo, fast_launch);
	constexpr size_t tess_at   = offsetof(Libs::Graphics::ShaderVertexInputInfo, tess) +
	                           offsetof(Libs::Graphics::ShaderTessellationInputInfo, min_tess_level);
	static_assert(mesh_size == 8 && mesh_at < tess_at && tess_at + 8 <= now);
	if (bytes.size() != now - 16) return false;
	std::vector<uint8_t> out;
	out.reserve(now);
	out.insert(out.end(), bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(mesh_at));
	const uint8_t mesh_defaults[8] = {0, 0, 0, 0, 1, 0, 0, 0}; // fast_launch false (and padding), passes 1
	out.insert(out.end(), std::begin(mesh_defaults), std::end(mesh_defaults));
	out.insert(out.end(), bytes.begin() + static_cast<std::ptrdiff_t>(mesh_at),
	           bytes.begin() + static_cast<std::ptrdiff_t>(tess_at - 8));
	out.insert(out.end(), 8, uint8_t {0}); // min/max tess level 0
	out.insert(out.end(), bytes.begin() + static_cast<std::ptrdiff_t>(tess_at - 8), bytes.end());
	if (out.size() != now) return false;
	bytes = std::move(out);
	return true;
}

int Upgrade(const Arguments& args) {
	if (args.inputs.size() != 1 || args.out.empty()) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	const std::array<uint32_t, 3> now {static_cast<uint32_t>(Libs::Graphics::ShaderStoredInputInfoSize<Libs::Graphics::ShaderVertexInputInfo>()),
	                                   static_cast<uint32_t>(sizeof(Libs::Graphics::ShaderPixelInputInfo)),
	                                   static_cast<uint32_t>(sizeof(Libs::Graphics::ShaderComputeInputInfo))};
	if (file.info.input_info_sizes == now) {
		std::printf("%s already has this build's input layout\n", args.inputs[0].c_str());
		return Write(file, args.out);
	}
	if (file.info.input_info_sizes[0] != now[0] - 16 || file.info.input_info_sizes[1] != now[1] ||
	    file.info.input_info_sizes[2] != now[2]) {
		std::fprintf(stderr, "%s: input infos %u %u %u cannot be upgraded to %u %u %u\n", args.inputs[0].c_str(),
		             file.info.input_info_sizes[0], file.info.input_info_sizes[1], file.info.input_info_sizes[2], now[0],
		             now[1], now[2]);
		// Not this tool's work: the stored specializations change with that layout too.
		if (file.info.input_info_sizes[0] == Libs::Graphics::Int21VertexLayout().from_size) {
			std::fprintf(stderr, "this is the int21 layout, which the emulator upgrades itself when it loads the list (its log "
			                     "says \"Pipeline list: int21 layout: ...\"): the list can be used as it is\n");
		}
		return 2;
	}
	size_t upgraded = 0;
	for (auto& shader: file.shaders) {
		if (shader.kind != ShaderJournal::Kind::Vertex) continue;
		if (!UpgradeVertexInputInfo(shader.input_info)) {
			std::fprintf(stderr, "a vertex shader's input info has %zu bytes\n", shader.input_info.size());
			return 2;
		}
		upgraded++;
	}
	file.info.input_info_sizes = now;
	std::printf("upgraded %zu vertex input infos\n", upgraded);
	return Write(file, args.out);
}

int Info(const Arguments& args) {
	if (args.inputs.size() != 1) return 2;
	std::vector<uint8_t> bytes;
	if (!ReadFile(U8(args.inputs[0]), bytes)) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	std::printf("%s: %zu bytes\n  title %s, version %s, producer %s, input infos %u %u %u, created %llu, pipeline "
	            "state format %u (this tool: %u), %zu data files\n",
	            args.inputs[0].c_str(), bytes.size(), file.info.title_id.c_str(), file.info.app_version.c_str(),
	            file.info.producer.c_str(), file.info.input_info_sizes[0], file.info.input_info_sizes[1],
	            file.info.input_info_sizes[2], static_cast<unsigned long long>(file.info.created_unix),
	            file.info.state_format, PipelineList::StateFormat, file.files.size());
	std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;
	(void)PipelineList::DecodeSections(bytes, sections);
	for (const auto& [id, raw]: sections) std::printf("  section %u: %zu raw bytes\n", id, raw.size());
	size_t kinds[3] = {}, code_bytes = 0, merged = 0;
	for (const auto& shader: file.shaders) {
		kinds[static_cast<size_t>(shader.kind)]++;
		code_bytes += (shader.code_size + shader.back_code_size) * 4u;
		merged += shader.back_code_size != 0 ? 1 : 0;
	}
	size_t mesh = 0, stages[3] = {};
	std::vector<size_t> per_level(file.levels.size());
	for (const auto& pipeline: file.pipelines) {
		mesh += pipeline.words.size() > 1 && pipeline.words[1] != 0 ? 1 : 0;
		stages[std::min<size_t>(2, pipeline.stages.size())]++;
		for (const auto level: pipeline.levels) per_level[level]++;
	}
	std::printf("  shaders %zu (vertex %zu, pixel %zu, compute %zu; %zu merged stages; %zu code bytes referenced)\n",
	            file.shaders.size(), kinds[0], kinds[1], kinds[2], merged, code_bytes);
	std::printf("  permutations %zu, pipelines %zu (%zu mesh, %zu without a pixel stage)\n", file.permutations.size(),
	            file.pipelines.size(), mesh, stages[1]);
	for (size_t i = 0; i < file.levels.size(); i++) std::printf("  level %s: %zu pipelines\n", file.levels[i].c_str(), per_level[i]);
	return 0;
}

// A list resolved against a game copy as the emulator does at start-up: the executable from `image`, data
// files read where the list names code (never a whole file: a data file can be many gigabytes).
PipelineList::Resolved ResolveWith(GameFiles& game, const PipelineList::File& file, const std::vector<uint8_t>& image,
                                   const std::array<uint32_t, 3>& sizes, PipelineList::ResolveStats& stats) {
	const PipelineList::ReadCodeFn read = [&](PipelineList::CodeFile code_file, uint64_t offset, std::span<uint8_t> out) {
		const auto index = static_cast<uint32_t>(code_file);
		if (index != 0) return index <= file.files.size() && game.ReadAt(U8(file.files[index - 1]), offset, out);
		if (offset > image.size() || out.size() > image.size() - offset) return false;
		std::memcpy(out.data(), image.data() + offset, out.size());
		return true;
	};
	return PipelineList::Resolve(file, sizes, read, [&](PipelineList::CodeFile) { return image; }, stats);
}

PipelineList::Resolved ResolveAgainst(const Arguments& args, GameFiles& game, const PipelineList::File& file,
                                      std::vector<uint8_t>& image, PipelineList::ResolveStats& stats) {
	if (!ReadExecutable(args, game, image)) {
		std::fprintf(stderr, "cannot read %s\n", ExecutableName(args).c_str());
		image.clear();
	}
	return ResolveWith(game, file, image, file.info.input_info_sizes, stats);
}

int Check(const Arguments& args) {
	if (args.inputs.size() != 1 || args.game.empty()) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	const auto game = OpenGame(args);
	if (game == nullptr) return 2;
	std::string title, version;
	if (!PipelineListTool::ReadParam(*game, title, version)) return 2;
	std::printf("list for %s %s; game %s %s%s\n", file.info.title_id.c_str(), file.info.app_version.c_str(), title.c_str(),
	            version.c_str(), title == file.info.title_id && version == file.info.app_version ? "" : ": MISMATCH (ignored at run time)");
	if (title != file.info.title_id || version != file.info.app_version) {
		return 2; // as the emulator: another title or game version is not resolved (no game file is read)
	}
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats stats;
	const auto                 resolved = ResolveAgainst(args, *game, file, image, stats);
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

int Retarget(const Arguments& args) {
	if (args.list.empty() || args.exe.empty() || args.title.empty() || args.app_ver.empty() || args.out.empty()) {
		std::fprintf(stderr, "retarget needs --list, --exe, --title, --app-ver and --out\n");
		return 2;
	}
	PipelineList::File file;
	if (!Load(args.list, file)) return 2;
	const auto game = OpenGame(args); // the list's data files (--game, if given)
	if (game == nullptr) return 2;
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats resolve_stats;
	const auto                 resolved = ResolveAgainst(args, *game, file, image, resolve_stats);
	PipelineList::RetargetStats stats;
	auto                        out = PipelineList::Retarget(file, resolved, stats);
	const auto ull = [](uint64_t v) { return static_cast<unsigned long long>(v); };
	std::printf("list for %s %s -> %s %s\n", file.info.title_id.c_str(), file.info.app_version.c_str(), args.title.c_str(),
	            args.app_ver.c_str());
	std::printf("shaders %llu: %llu at the same offset, %llu found moved, %llu not found\n", ull(stats.shaders), ull(stats.same_offset),
	            ull(stats.moved), ull(stats.missing));
	const char* names[3]    = {"vertex", "pixel", "compute"};
	uint64_t    total[3]    = {}, kept[3] = {};
	for (size_t i = 0; i < file.shaders.size(); i++) {
		total[static_cast<size_t>(file.shaders[i].kind)]++;
		kept[static_cast<size_t>(file.shaders[i].kind)] += resolved.usable[i] ? 1 : 0;
	}
	for (size_t k = 0; k < 3; k++) {
		std::printf("  %s shaders: %llu of %llu found\n", names[k], ull(kept[k]), ull(total[k]));
	}
	std::printf("permutations %zu of %zu kept; pipelines %llu: %llu kept, %llu dropped\n", out.permutations.size(),
	            file.permutations.size(), ull(stats.pipelines), ull(stats.pipelines_kept), ull(stats.pipelines - stats.pipelines_kept));
	// Per level: pipelines kept of those the level had (a pipeline counts in every level it was seen in).
	std::vector<size_t> level_total(file.levels.size()), level_kept(file.levels.size());
	for (const auto& pipeline: file.pipelines) {
		const bool keep = std::all_of(pipeline.stages.begin(), pipeline.stages.end(), [&](const PipelineList::Stage& stage) {
			return resolved.permutation_entry[stage.permutation] != UINT32_MAX;
		});
		for (const auto level: pipeline.levels) {
			level_total[level]++;
			level_kept[level] += keep ? 1 : 0;
		}
	}
	for (size_t i = 0; i < file.levels.size(); i++) {
		std::printf("  level %s: %zu of %zu pipelines kept\n", file.levels[i].c_str(), level_kept[i], level_total[i]);
	}
	if (stats.same_offset + stats.moved == 0) {
		std::fprintf(stderr, "no shader found: nothing written\n");
		return 1;
	}
	out.info.title_id     = args.title;
	out.info.app_version  = args.app_ver;
	out.info.producer     = args.producer + " (retargeted from " + file.info.title_id + " " + file.info.app_version + ")";
	out.info.created_unix = NowUnix();
	return Write(out, args.out);
}

int Verify(const Arguments& args) {
	if (args.inputs.size() != 1 || args.game.empty()) return 2;
	std::vector<uint8_t> bytes;
	PipelineList::File   file;
	if (!ReadFile(U8(args.inputs[0]), bytes) || !Load(args.inputs[0], file)) return 2;
	const auto game = OpenGame(args);
	if (game == nullptr) return 2;
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats stats;
	const auto                 resolved = ResolveAgainst(args, *game, file, image, stats);
	PipelineList::CodeWindows  windows;
	uint64_t                   code_bytes = 0;
	for (size_t i = 0; i < resolved.sources.size(); i++) {
		if (!resolved.usable[i]) continue;
		for (const auto* code: {&resolved.sources[i].code, &resolved.sources[i].back_code}) {
			windows.Add({reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4});
			code_bytes += code->size() * 4;
		}
	}
	for (const auto& path: args.shaders) {
		std::vector<uint8_t> identity;
		if (!ShaderJournal::ReadIdentity(path, identity)) {
			std::fprintf(stderr, "%s is not a shader journal\n", path.c_str());
			return 2;
		}
		ShaderJournal::Settings settings;
		settings.path              = U8(path);
		settings.identity          = identity;
		settings.background_writer = false;
		ShaderJournal journal(std::move(settings));
		for (const auto& source: journal.Sources()) {
			for (const auto* code: {&source.code, &source.back_code}) {
				windows.Add({reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4});
				code_bytes += code->size() * 4;
			}
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
		std::fprintf(stderr, "usage: kyty_pipeline_list export|merge|info|check|retarget|verify ... (see the source header)\n");
		return 2;
	}
	// The arguments as UTF-8 (Windows: from the wide command line; the narrow argv loses characters outside the code page).
	std::vector<std::string> utf8(argv, argv + argc);
#if defined(_WIN32)
	int  wide_count = 0;
	auto wide       = CommandLineToArgvW(GetCommandLineW(), &wide_count);
	if (wide != nullptr && wide_count == argc) {
		for (int i = 0; i < argc; i++) utf8[i] = Common::PathToString(std::filesystem::path(wide[i]));
	}
	if (wide != nullptr) LocalFree(wide);
#endif
	args.command = utf8[1];
	for (int i = 2; i < argc; i++) {
		const std::string arg   = utf8[i];
		const auto        value = [&]() -> std::string { return i + 1 < argc ? utf8[++i] : std::string(); };
		if (arg == "--out") {
			args.out = value();
		} else if (arg == "--game") {
			args.game = value();
		} else if (arg == "--exe") {
			args.exe = value();
		} else if (arg == "--list") {
			args.list = value();
		} else if (arg == "--title") {
			args.title = value();
		} else if (arg == "--app-ver") {
			args.app_ver = value();
		} else if (arg == "--producer") {
			args.producer = value();
		} else if (arg == "--shaders") {
			args.shaders.push_back(value());
		} else if (arg == "--pipelines") {
			args.pipelines.push_back(value());
		} else if (arg == "--data") {
			args.data.push_back(value());
		} else if (arg == "--base") {
			args.base = value();
		} else if (arg == "--scan") {
			args.scan = true;
		} else {
			args.inputs.push_back(arg);
		}
	}
	if (args.command == "export") return Export(args);
	if (args.command == "merge") return Merge(args);
	if (args.command == "info") return Info(args);
	if (args.command == "check") return Check(args);
	if (args.command == "retarget") return Retarget(args);
	if (args.command == "verify") return Verify(args);
	if (args.command == "upgrade") return Upgrade(args);
	std::fprintf(stderr, "unknown command %s\n", args.command.c_str());
	return 2;
}
