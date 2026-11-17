// KYTY_PCACHE_SAVE_MAPPED (graphics/host_gpu/renderer/pipeline/pipelineCacheMappedFile.h): the driver
// cache's temporary file written through a mapping is cut to the bytes used, replaces an older file
// of the same name, and reads back as written.

#include "graphics/host_gpu/renderer/pipeline/pipelineCacheMappedFile.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using Libs::Graphics::PipelineCacheMappedFile;

namespace {

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("PipelineCacheMappedFileTests: FAILED: %s\n", what);
		g_failures++;
	}
}

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

} // namespace

int main() {
	const auto folder = std::filesystem::temp_directory_path() /
	                    ("kyty_pcache_mapped_test_" +
	                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
	std::filesystem::create_directories(folder);
	const auto path = folder / "PPSA00000.bin.tmp";

	std::vector<uint8_t> payload(100003);
	for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<uint8_t>(i * 131u + 7u);
	{
		// The driver may fill fewer bytes than it was given room for: the file keeps only those.
		PipelineCacheMappedFile file;
		Check(file.Create(path, payload.size() + 4096) && file.Data() != nullptr &&
		          file.Size() == payload.size() + 4096,
		      "create maps the requested size");
		std::memcpy(file.Data(), payload.data(), payload.size());
		Check(file.Close(payload.size()), "close cuts and flushes");
		Check(!file.Close(payload.size()), "a closed file does not close again");
	}
	Check(std::filesystem::file_size(path) == payload.size(), "the file is cut to the bytes used");
	Check(ReadAll(path) == payload, "the file reads back as written");
	{
		// A new save replaces the older file of the same name.
		PipelineCacheMappedFile file;
		Check(file.Create(path, 64), "a second create replaces the file");
		std::memset(file.Data(), 0x5a, 64);
		Check(file.Close(64), "the second file closes");
	}
	Check(std::filesystem::file_size(path) == 64 && ReadAll(path) == std::vector<uint8_t>(64, 0x5a),
	      "the replaced file holds the new bytes only");
	{
		// Destroyed while open (an error path): closed empty, no crash.
		PipelineCacheMappedFile file;
		Check(file.Create(path, 4096), "create before an early return");
		Check(!PipelineCacheMappedFile().Close(1), "a never-opened file does not close");
	}
	Check(std::filesystem::file_size(path) == 0, "a file destroyed while open is left empty");
	PipelineCacheMappedFile zero;
	Check(!zero.Create(path, 0), "a zero-sized file is refused");

	std::error_code error;
	std::filesystem::remove_all(folder, error);
	if (g_failures != 0) {
		return EXIT_FAILURE;
	}
	std::printf("PipelineCacheMappedFileTests: ok\n");
	return EXIT_SUCCESS;
}
