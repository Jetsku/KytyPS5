// kyty_pipeline_list: makes and checks KYPLST1 pipeline lists (pipelineList.h, KYTY_PIPELINE_LIST).
//
//   export --out <list> --game <app0 dir> (--shaders <shader journal> --pipelines <pipeline journal>)...
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
//   merge --out <list> <list>...        Union of lists of the same title, version and input layout.
//   info <list>                         Header, counts, levels, section sizes.
//   check <list> --game <app0 dir> [--exe <executable>]
//       Resolves the list against a game copy, as the emulator does at start-up.
//   retarget --list <list> --exe <executable> --title <id> --app-ver <version> --out <list>
//       Re-stamps a list for another copy of the game (another version or SKU): each shader is looked up in
//       the other executable (at its recorded offset, then by hash), shaders not found are dropped with
//       their pipelines, the title and version are replaced. Prints the match counts.
//   upgrade <list> --out <list>
//       Rewrites the vertex stage input infos of a list recorded before the mesh pass and
//       tessellation clamp fields (stored size 16 bytes smaller: int21 and older builds) into this
//       build's layout, with those fields at their defaults; other input infos must already match.
//   verify <list> --game <app0 dir> [--exe <executable>] [--shaders <shader journal>]...
//       The no-code proof: no 64-byte window of the listed shaders' code (read from the executable)
//       or of any journaled shader's code occurs in the file or in any decompressed section.
//       Exit code 1 when one does.
#include "common/stringUtils.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineList.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
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

// Calls fn(file offset of the byte after the footer, bytes ending there (at most max_before + footer))
// for every footer in the file, reading it in 64 MiB chunks. False when the file cannot be read.
template <typename Fn>
bool ForEachFooter(const std::filesystem::path& path, size_t max_before, Fn&& fn) {
	std::ifstream in(path, std::ios::binary);
	if (!in) return false;
	constexpr size_t     Chunk = size_t {64} << 20u;
	std::vector<uint8_t> buffer;
	std::vector<uint8_t> chunk(Chunk);
	uint64_t             buffer_start = 0; // file offset of buffer[0]
	size_t               searched     = 0; // buffer positions below this were searched already
	for (;;) {
		in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
		const auto got = static_cast<size_t>(in.gcount());
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
	return args.exe.empty() ? U8(args.game) / "eboot.bin" : U8(args.exe);
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
	if (!WriteFile(U8(out), bytes)) {
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
	if (!ReadParam(U8(args.game), info.title_id, info.app_version)) {
		std::fprintf(stderr, "cannot read the title and version from %s/sce_sys/param.json\n", args.game.c_str());
		return 2;
	}
	info.producer     = args.producer;
	info.created_unix = NowUnix();
	info.state_format = PipelineList::StateFormat;
	std::vector<uint8_t> image;
	if (!ReadFile(Executable(args), image)) {
		std::fprintf(stderr, "cannot read %s\n", Common::PathToString(Executable(args)).c_str());
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
		const bool no_pipelines = args.pipelines[i] == "none";
		if (!ShaderJournal::ReadIdentity(args.shaders[i], shader_identity) ||
		    (!no_pipelines && !PipelineJournal::ReadIdentity(args.pipelines[i], pipeline_identity))) {
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
		auto& shared = loaded_shaders[args.shaders[i]];
		if (shared == nullptr) {
			ShaderJournal::Settings shader_settings;
			shader_settings.path              = U8(args.shaders[i]);
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
	std::vector<PipelineList::CodeQuery> queries;
	std::map<std::pair<uint64_t, uint64_t>, size_t> query_of;
	for (const auto& [path, journal]: loaded_shaders) {
		(void)path;
		for (const auto& source: journal->Sources()) {
			for (const auto* code: {&source.code, &source.back_code}) {
				if (code->empty()) continue;
				const std::span bytes(reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4);
				const auto      query = PipelineList::QueryFor(bytes);
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
	PipelineList::Builder builder(info);
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
		if (!ReadFile(U8(args.game) / U8(data), data_image)) {
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
		size_t                                    max_before = 0, no_footer = 0, rest = 0;
		std::vector<uint64_t>                     footer_of(queries.size(), 0);
		for (const auto& [path, journal]: loaded_shaders) {
			(void)path;
			for (const auto& source: journal->Sources()) {
				for (const auto* code: {&source.code, &source.back_code}) {
					if (code->empty()) continue;
					const std::span bytes(reinterpret_cast<const uint8_t*>(code->data()), code->size() * 4);
					const auto      hash = PipelineList::CodeHash(bytes);
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
					rest++;
				}
			}
		}
		std::printf("scan: %zu codes not found yet have a footer, %zu have none\n", rest, no_footer);
		// eboot.bin was searched; the other files, smaller first, until every code is found.
		std::vector<std::pair<uint64_t, std::filesystem::path>> files;
		std::error_code                                         ec;
		const std::filesystem::path                             root = U8(args.game);
		for (auto it = std::filesystem::recursive_directory_iterator(root, ec); !ec && it != std::filesystem::recursive_directory_iterator();
		     it.increment(ec)) {
			if (!it->is_regular_file(ec) || it->path().filename() == "eboot.bin") continue;
			files.emplace_back(it->file_size(ec), it->path());
		}
		std::sort(files.begin(), files.end());
		size_t   found_total = 0;
		uint64_t read_bytes  = 0;
		for (const auto& [size, path]: files) {
			if (found_total == rest) break;
			const auto relative = Common::PathToGenericString(std::filesystem::relative(path, root, ec));
			size_t     in_file  = 0;
			read_bytes += size;
			ForEachFooter(path, max_before, [&](uint64_t end, std::span<const uint8_t> before) {
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
		std::printf("scan: located %zu of %zu (%zu files, %.1f GB read)\n", found_total, rest, files.size(),
		            static_cast<double>(read_bytes) / 1e9);
	}
	const PipelineList::Builder::LocateFn locate =
	    [&](std::span<const uint32_t> code) -> std::optional<PipelineList::Builder::Location> {
		const auto hash = PipelineList::CodeHash({reinterpret_cast<const uint8_t*>(code.data()), code.size_bytes()});
		const auto it   = query_of.find({hash.low, hash.high});
		if (it == query_of.end() || !have[it->second]) return std::nullopt;
		return where[it->second];
	};
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

PipelineList::Resolved ResolveAgainst(const Arguments& args, const PipelineList::File& file, std::vector<uint8_t>& image,
                                      PipelineList::ResolveStats& stats) {
	if (!ReadFile(Executable(args), image)) {
		std::fprintf(stderr, "cannot read %s\n", Common::PathToString(Executable(args)).c_str());
		image.clear();
	}
	// Data files: read once when first named.
	auto data_images = std::make_shared<std::map<uint32_t, std::vector<uint8_t>>>();
	const PipelineList::ReadCodeFn read = [&, data_images](PipelineList::CodeFile code_file, uint64_t offset,
	                                                       std::span<uint8_t> out) {
		const auto index = static_cast<uint32_t>(code_file);
		const std::vector<uint8_t>* source = &image;
		if (index != 0) {
			auto [it, added] = data_images->try_emplace(index);
			if (added && (index > file.files.size() ||
			              !ReadFile(U8(args.game) / U8(file.files[index - 1]), it->second))) {
				it->second.clear();
			}
			source = &it->second;
		}
		if (offset > source->size() || out.size() > source->size() - offset) return false;
		std::memcpy(out.data(), source->data() + offset, out.size());
		return true;
	};
	return PipelineList::Resolve(file, file.info.input_info_sizes, read, [&](PipelineList::CodeFile) { return image; }, stats);
}

int Check(const Arguments& args) {
	if (args.inputs.size() != 1 || args.game.empty()) return 2;
	PipelineList::File file;
	if (!Load(args.inputs[0], file)) return 2;
	std::string title, version;
	if (!ReadParam(U8(args.game), title, version)) return 2;
	std::printf("list for %s %s; game %s %s%s\n", file.info.title_id.c_str(), file.info.app_version.c_str(), title.c_str(),
	            version.c_str(), title == file.info.title_id && version == file.info.app_version ? "" : ": MISMATCH (ignored at run time)");
	if (title != file.info.title_id || version != file.info.app_version) {
		return 2; // as the emulator: another title or game version is not resolved (no game file is read)
	}
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

int Retarget(const Arguments& args) {
	if (args.list.empty() || args.exe.empty() || args.title.empty() || args.app_ver.empty() || args.out.empty()) {
		std::fprintf(stderr, "retarget needs --list, --exe, --title, --app-ver and --out\n");
		return 2;
	}
	PipelineList::File file;
	if (!Load(args.list, file)) return 2;
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats resolve_stats;
	const auto                 resolved = ResolveAgainst(args, file, image, resolve_stats);
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
	std::vector<uint8_t>       image;
	PipelineList::ResolveStats stats;
	const auto                 resolved = ResolveAgainst(args, file, image, stats);
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
