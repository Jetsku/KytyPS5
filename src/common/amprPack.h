#ifndef KYTY_COMMON_AMPRPACK_H_
#define KYTY_COMMON_AMPRPACK_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

typedef struct ZSTD_DCtx_s ZSTD_DCtx;

namespace Common {

class AmprPack {
public:
	struct FileInfo {
		uint64_t size;
		uint32_t firstChunk;
		uint32_t chunkCount;
	};

	struct DirEntry {
		std::string name;
		bool        is_file;
	};

	AmprPack() = default;
	~AmprPack();
	KYTY_CLASS_NO_COPY(AmprPack)

	bool Init(const std::filesystem::path& gameRoot);

	// Test-only instrumentation: counts Init() calls so tests can prove the index is parsed once per root.
	[[nodiscard]] static uint32_t InitCount();
	static void ResetInitCount();

	[[nodiscard]] bool Lookup(std::string_view guestPath, FileInfo& out) const;
	[[nodiscard]] bool Exists(std::string_view guestPath) const;

	// Directory enumeration derived from the path string table. Member paths need no "/app0/" prefix.
	[[nodiscard]] bool IsDirectory(std::string_view member) const;
	[[nodiscard]] std::vector<DirEntry> ListDirectory(std::string_view member) const;

	// One scratch buffer and one zstd context per instance: serialize reads on the same instance.
	bool Read(const FileInfo& file, uint64_t offset, size_t len, void* out);

private:
	void Reset();
	[[nodiscard]] int VolumeFd(uint32_t volume);
	[[nodiscard]] bool Fill(const uint8_t* record, FileInfo& out) const;

	int                                    m_indexFd = -1;
	const uint8_t*                         m_index = nullptr;
	size_t                                 m_indexSize = 0;
	const uint8_t*                         m_files = nullptr;
	const uint8_t*                         m_chunks = nullptr;
	const uint8_t*                         m_volumes = nullptr;
	const uint8_t*                         m_strings = nullptr;
	uint64_t                               m_fileCount = 0;
	uint64_t                               m_chunkCount = 0;
	uint32_t                               m_volumeCount = 0;
	uint64_t                               m_stringsLen = 0;
	std::filesystem::path                  m_root;
	std::vector<int>                       m_volumeFds;
	std::unordered_map<std::string, uint32_t> m_folded;
	std::vector<std::string>               m_sortedPaths; // folded file paths, sorted for prefix queries
	std::vector<uint8_t>                   m_scratch;
	ZSTD_DCtx*                             m_dctx = nullptr;
};

} // namespace Common

#endif /* KYTY_COMMON_AMPRPACK_H_ */
