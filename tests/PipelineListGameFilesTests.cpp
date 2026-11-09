// kyty_pipeline_list / kyty_shaderlist_export game files (tools/pipeline-list/gameFiles.h): a game
// folder and a .zar archive of it give the same param.json title and version, the same eboot.bin
// and file bytes, and the same file list in the same order (the order --scan reads them in).
#include "gameFiles.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <zarchive/zarchivewriter.h>

namespace fs = std::filesystem;

static int g_failed = 0;
#define CHECK(expr)                                                                         \
	do {                                                                                    \
		if (!(expr)) {                                                                      \
			std::printf("PipelineListGameFilesTests: FAILED: %s (line %d)\n", #expr, __LINE__); \
			++g_failed;                                                                     \
		}                                                                                   \
	} while (0)

namespace {

std::vector<uint8_t> Bytes(size_t size, uint32_t seed) {
	std::mt19937         random(seed);
	std::vector<uint8_t> bytes(size);
	for (auto& b: bytes) b = static_cast<uint8_t>(random());
	return bytes;
}

void Put(const fs::path& path, const std::vector<uint8_t>& bytes) {
	fs::create_directories(path.parent_path());
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void Put(const fs::path& path, const std::string& text) {
	Put(path, std::vector<uint8_t>(text.begin(), text.end()));
}

// Packs the folder as the launcher's "Compress game" does (common/archivePack.cpp): every entry
// under its path relative to the folder, '/' separated.
bool Pack(const fs::path& folder, const fs::path& archive) {
	struct Output {
		std::ofstream stream;
		bool          good = true;
	} output;
	output.stream.open(archive, std::ios::binary | std::ios::trunc);
	if (!output.stream) return false;
	{
		ZArchiveWriter writer(
		    [](int32_t, void*) {},
		    [](const void* data, size_t size, void* opaque) {
			    auto& out = *static_cast<Output*>(opaque);
			    out.stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
			    out.good = out.good && out.stream.good();
		    },
		    &output);
		for (const auto& entry: fs::recursive_directory_iterator(folder)) {
			const auto name = Common::PathToGenericString(entry.path().lexically_relative(folder));
			if (entry.is_directory()) {
				if (!writer.MakeDir(name.c_str())) return false;
				continue;
			}
			std::vector<uint8_t> bytes;
			std::ifstream        in(entry.path(), std::ios::binary);
			bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
			if (!writer.StartNewFile(name.c_str())) return false;
			writer.AppendData(bytes.data(), bytes.size());
		}
		writer.Finalize();
	}
	output.stream.close();
	return output.good && !output.stream.fail();
}

std::vector<uint8_t> ReadAll(const PipelineListTool::GameFiles::Stream& read, size_t chunk) {
	std::vector<uint8_t> all, buffer(chunk);
	for (;;) {
		const auto got = read(buffer.data(), buffer.size());
		all.insert(all.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(got));
		if (got < buffer.size()) break;
	}
	return all;
}

void TestZarPath() {
	using PipelineListTool::IsZarPath;
	CHECK(IsZarPath("game.zar"));
	CHECK(IsZarPath("C:/games/PPSA00000.ZAR"));
	CHECK(IsZarPath("dir.with.dots/Game.Zar"));
	CHECK(!IsZarPath("game.zip"));
	CHECK(!IsZarPath("game.zarx"));
	CHECK(!IsZarPath("zar"));
	CHECK(!IsZarPath("C:/games/PPSA00000"));
}

void TestFolderAndArchive(const fs::path& temp) {
	const auto folder  = temp / "PPSA00000-app0";
	const auto archive = temp / "PPSA00000-app0.ZAR"; // the extension in any case
	const auto eboot   = Bytes(300000, 1);            // several 64 KiB archive blocks
	Put(folder / "eboot.bin", eboot);
	Put(folder / "sce_sys" / "param.json",
	    std::string(R"({"titleId":"PPSA00000","contentVersion":"01.002.000","titleName":"Game files test"})"));
	Put(folder / "data" / "big.bin", Bytes(200000, 2));
	Put(folder / "data" / "b.bin", Bytes(4096, 3)); // three files of one size: ordered by path
	Put(folder / "data" / "a.bin", Bytes(4096, 4));
	Put(folder / "data-a.bin", Bytes(4096, 5));
	Put(folder / "data" / "sub" / "eboot.bin", Bytes(10, 6)); // not the game's eboot.bin
	Put(folder / Common::PathFromUtf8("data/\xc3\xa4\xc3\xb6-\xe6\xbc\xa2.bin"), Bytes(77, 7));
	fs::create_directories(folder / "empty");
	CHECK(Pack(folder, archive));

	const auto from_folder  = PipelineListTool::GameFiles::Open(folder);
	const auto from_archive = PipelineListTool::GameFiles::Open(archive);
	CHECK(from_folder != nullptr && !from_folder->IsArchive());
	CHECK(from_archive != nullptr && from_archive->IsArchive());
	if (from_folder == nullptr || from_archive == nullptr) return;

	for (auto* game: {from_folder.get(), from_archive.get()}) {
		std::string title, version;
		CHECK(PipelineListTool::ReadParam(*game, title, version));
		CHECK(title == "PPSA00000");
		CHECK(version == "01.002.000");
		std::vector<uint8_t> bytes;
		CHECK(game->HasFile("eboot.bin"));
		CHECK(!game->HasFile("missing.bin"));
		CHECK(!game->HasFile("data")); // a directory
		CHECK(game->Read("eboot.bin", bytes) && bytes == eboot);
		CHECK(ReadAll(game->OpenStream("eboot.bin"), 65536 + 7) == eboot);
		CHECK(!game->Read("missing.bin", bytes));
		CHECK(!game->OpenStream("missing.bin"));
	}

	const auto folder_files  = from_folder->Files();
	const auto archive_files = from_archive->Files();
	CHECK(folder_files.size() == 8);
	CHECK(folder_files.size() == archive_files.size());
	for (size_t i = 0; i < std::min(folder_files.size(), archive_files.size()); i++) {
		const auto& a = folder_files[i];
		const auto& b = archive_files[i];
		CHECK(a.size == b.size);
		CHECK(a.relative == b.relative);
		CHECK(Common::PathToGenericString(a.relative) == Common::PathToGenericString(b.relative));
		std::vector<uint8_t> x, y;
		CHECK(from_folder->Read(a.relative, x) && from_archive->Read(b.relative, y) && x == y);
		CHECK(ReadAll(from_archive->OpenStream(b.relative), 1000) == x);
		CHECK(x.size() == a.size);
	}
	if (folder_files.size() == 8 && archive_files.size() == 8) {
		// Smallest first; one size ordered by path.
		CHECK(Common::PathToGenericString(archive_files[0].relative) == "data/sub/eboot.bin");
		CHECK(Common::PathToGenericString(archive_files[3].relative) == "data/a.bin");
		CHECK(Common::PathToGenericString(archive_files[4].relative) == "data/b.bin");
		CHECK(Common::PathToGenericString(archive_files[5].relative) == "data-a.bin");
		CHECK(Common::PathToGenericString(archive_files[7].relative) == "eboot.bin");
	}
}

void TestBadArchive(const fs::path& temp) {
	CHECK(PipelineListTool::GameFiles::Open(temp / "missing.zar") == nullptr);
	Put(temp / "not-an-archive.zar", Bytes(1000, 8));
	CHECK(PipelineListTool::GameFiles::Open(temp / "not-an-archive.zar") == nullptr);
	// A missing folder opens; its reads fail.
	const auto missing = PipelineListTool::GameFiles::Open(temp / "missing-folder");
	CHECK(missing != nullptr);
	if (missing != nullptr) {
		std::string title, version;
		CHECK(!PipelineListTool::ReadParam(*missing, title, version));
		CHECK(missing->Files().empty());
	}
}

} // namespace

int main() {
	const auto temp = fs::temp_directory_path() / ("kyty-game-files-tests-" + std::to_string(std::random_device {}()));
	fs::create_directories(temp);
	TestZarPath();
	TestFolderAndArchive(temp);
	TestBadArchive(temp);
	std::error_code error;
	fs::remove_all(temp, error);
	if (g_failed != 0) {
		std::printf("PipelineListGameFilesTests: %d check(s) FAILED\n", g_failed);
		return 1;
	}
	std::printf("PipelineListGameFilesTests: all passed\n");
	return 0;
}
