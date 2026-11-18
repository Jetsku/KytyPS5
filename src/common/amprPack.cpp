#include "common/amprPack.h"

// mmap/pread and the POSIX headers below only exist on Linux and macOS. common is built on Windows too,
// where the reader stays in the target but reports every lookup/read as a miss.
#if defined(__linux__) || defined(__APPLE__)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <zstd.h>

namespace Common {

namespace {

constexpr uint32_t kHeaderSize = 128;
constexpr uint32_t kChunkShift = 18;
constexpr uint64_t kChunkSize = 1ull << kChunkShift;
constexpr uint32_t kMaxVolumes = 4096; // volume ids are 12 bits in a chunk word

uint64_t Le64(const uint8_t* p) {
	uint64_t value;
	std::memcpy(&value, p, sizeof(value));
	return value;
}

uint32_t Le32(const uint8_t* p) {
	uint32_t value;
	std::memcpy(&value, p, sizeof(value));
	return value;
}

uint16_t Le16(const uint8_t* p) {
	uint16_t value;
	std::memcpy(&value, p, sizeof(value));
	return value;
}

uint64_t Fnv1a(std::string_view text) {
	uint64_t hash = 14695981039346656037ull;
	for (unsigned char c : text) {
		hash ^= c;
		hash *= 1099511628211ull;
	}
	return hash;
}

char Fold(char c) {
	return (c >= 'A' && c <= 'Z') ? char(c - 'A' + 'a') : c;
}

std::string Fold(std::string_view text) {
	std::string result(text);
	for (char& c : result) {
		c = Fold(c);
	}
	return result;
}

std::string_view RelativePath(std::string_view path) {
	if (path.size() >= 6) {
		const auto prefix = path.substr(0, 6);
		bool is_app0 = true;
		for (size_t i = 0; i < prefix.size(); ++i) {
			if (Fold(prefix[i]) != "/app0/"[i]) {
				is_app0 = false;
				break;
			}
		}
		if (is_app0) {
			return path.substr(6);
		}
	}
	return path;
}

std::string DirectoryPrefix(std::string_view member) {
	const auto relative = RelativePath(member);
	if (relative.empty()) {
		return {};
	}
	std::string prefix = Fold(relative);
	prefix.push_back('/');
	return prefix;
}

std::atomic<uint32_t> g_initCount {0};

bool ReadExact(int fd, void* buffer, size_t size, uint64_t offset) {
	auto* bytes = static_cast<uint8_t*>(buffer);
	while (size != 0) {
		const auto read = ::pread(fd, bytes, size, static_cast<off_t>(offset));
		if (read < 0) {
			if (errno == EINTR) {
				continue;
			}
			return false;
		}
		if (read == 0) {
			return false;
		}
		bytes += read;
		size -= static_cast<size_t>(read);
		offset += static_cast<uint64_t>(read);
	}
	return true;
}

} // namespace

AmprPack::~AmprPack() {
	Reset();
}

void AmprPack::Reset() {
	for (int& fd : m_volumeFds) {
		if (fd >= 0) {
			::close(fd);
			fd = -1;
		}
	}
	m_volumeFds.clear();
	if (m_dctx != nullptr) {
		ZSTD_freeDCtx(m_dctx);
		m_dctx = nullptr;
	}
	m_scratch.clear();
	m_folded.clear();
	m_sortedPaths.clear();
	m_files = m_chunks = m_volumes = m_strings = nullptr;
	m_fileCount = m_chunkCount = m_stringsLen = 0;
	m_volumeCount = 0;
	m_root.clear();
	if (m_index != nullptr) {
		::munmap(const_cast<uint8_t*>(m_index), m_indexSize);
		m_index = nullptr;
	}
	m_indexSize = 0;
	if (m_indexFd >= 0) {
		::close(m_indexFd);
		m_indexFd = -1;
	}
}

bool AmprPack::Init(const std::filesystem::path& gameRoot) {
	g_initCount.fetch_add(1, std::memory_order_relaxed);
	Reset();
	const auto indexPath = gameRoot / "ampr_zstd.index";
	m_indexFd = ::open(indexPath.c_str(), O_RDONLY | O_CLOEXEC);
	if (m_indexFd < 0) {
		Reset();
		return false;
	}
	struct stat info {};
	if (::fstat(m_indexFd, &info) != 0 || info.st_size < kHeaderSize) {
		Reset();
		return false;
	}
	m_indexSize = static_cast<size_t>(info.st_size);
	auto* mapped = static_cast<const uint8_t*>(::mmap(nullptr, m_indexSize, PROT_READ, MAP_PRIVATE, m_indexFd, 0));
	if (mapped == MAP_FAILED) {
		Reset();
		return false;
	}
	m_index = mapped;

	const uint8_t* header = m_index;
	if (std::memcmp(header, "AMPRZPK1", 8) != 0 || Le32(header + 8) != 1 ||
	    Le32(header + 12) != kHeaderSize || header[16] != kChunkShift ||
	    header[17] < 1 || header[17] > 22 || Le16(header + 18) != 0) {
		Reset();
		return false;
	}
	m_volumeCount = Le32(header + 20);
	if (m_volumeCount == 0 || m_volumeCount > kMaxVolumes) {
		Reset();
		return false;
	}
	m_fileCount = Le64(header + 40);
	m_chunkCount = Le64(header + 48);
	const uint64_t filesOff = Le64(header + 56);
	const uint64_t chunksOff = Le64(header + 64);
	const uint64_t volumesOff = Le64(header + 72);
	const uint64_t stringsOff = Le64(header + 80);
	m_stringsLen = Le64(header + 88);
	// Check each table offset before subtracting it from the mapping size: a corrupt index
	// must not wrap the bound and hand out pointers past the end of the mapping.
	if (filesOff != kHeaderSize || m_fileCount > UINT32_MAX ||
	    m_fileCount > (m_indexSize - kHeaderSize) / 32 || chunksOff != filesOff + m_fileCount * 32) {
		Reset();
		return false;
	}
	if (m_chunkCount > (m_indexSize - chunksOff) / 8 || volumesOff != chunksOff + m_chunkCount * 8) {
		Reset();
		return false;
	}
	if (m_volumeCount > (m_indexSize - volumesOff) / 8 || stringsOff != volumesOff + m_volumeCount * 8) {
		Reset();
		return false;
	}
	if (m_stringsLen != m_indexSize - stringsOff || (m_chunkCount == 0) != (m_volumeCount == 0)) {
		Reset();
		return false;
	}

	m_files = m_index + filesOff;
	m_chunks = m_index + chunksOff;
	m_volumes = m_index + volumesOff;
	m_strings = m_index + stringsOff;
	m_root = gameRoot;
	m_folded.reserve(static_cast<size_t>(m_fileCount));
	for (uint64_t i = 0; i < m_fileCount; ++i) {
		const uint8_t* record = m_files + i * 32;
		const uint32_t pathOff = Le32(record + 20);
		const uint16_t pathLen = Le16(record + 24);
		if (pathLen >= 1024 || uint64_t(pathOff) + pathLen > m_stringsLen) {
			Reset();
			return false;
		}
		const std::string_view path(reinterpret_cast<const char*>(m_strings + pathOff), pathLen);
		m_folded.emplace(Fold(path), static_cast<uint32_t>(i));
	}
	m_sortedPaths.reserve(m_folded.size());
	for (const auto& entry: m_folded) {
		m_sortedPaths.push_back(entry.first);
	}
	std::sort(m_sortedPaths.begin(), m_sortedPaths.end());

	m_scratch.resize(2 * kChunkSize);
	m_dctx = ZSTD_createDCtx();
	if (m_dctx == nullptr || ZSTD_isError(ZSTD_DCtx_setParameter(m_dctx, ZSTD_d_windowLogMax, kChunkShift))) {
		Reset();
		return false;
	}
	m_volumeFds.assign(m_volumeCount, -1);
	return true;
}

uint32_t AmprPack::InitCount() {
	return g_initCount.load(std::memory_order_relaxed);
}

void AmprPack::ResetInitCount() {
	g_initCount.store(0, std::memory_order_relaxed);
}

bool AmprPack::Fill(const uint8_t* record, FileInfo& out) const {
	const uint64_t size = Le64(record + 8);
	const uint32_t first = Le32(record + 16);
	const uint64_t chunks = size / kChunkSize + (size % kChunkSize != 0);
	if (first > m_chunkCount || chunks > m_chunkCount - first) {
		return false;
	}
	out = {size, first, static_cast<uint32_t>(chunks)};
	return true;
}

bool AmprPack::Lookup(std::string_view guestPath, FileInfo& out) const {
	if (m_index == nullptr) {
		return false;
	}
	const auto relative = RelativePath(guestPath);
	if (relative.empty()) {
		return false;
	}
	const uint64_t hash = Fnv1a(relative);
	uint64_t low = 0;
	uint64_t high = m_fileCount;
	while (low < high) {
		const uint64_t middle = low + (high - low) / 2;
		const uint8_t* record = m_files + middle * 32;
		const uint64_t recordHash = Le64(record);
		if (recordHash != hash) {
			if (recordHash < hash) {
				low = middle + 1;
			} else {
				high = middle;
			}
			continue;
		}
		const std::string_view path(reinterpret_cast<const char*>(m_strings + Le32(record + 20)), Le16(record + 24));
		if (path != relative) {
			if (path < relative) {
				low = middle + 1;
			} else {
				high = middle;
			}
			continue;
		}
		return Fill(record, out);
	}
	const auto found = m_folded.find(Fold(relative));
	if (found == m_folded.end()) {
		return false;
	}
	return Fill(m_files + uint64_t(found->second) * 32, out);
}

bool AmprPack::Exists(std::string_view guestPath) const {
	FileInfo info {};
	return Lookup(guestPath, info);
}

bool AmprPack::IsDirectory(std::string_view member) const {
	const auto prefix = DirectoryPrefix(member);
	if (prefix.empty()) {
		return false;
	}
	const auto it = std::lower_bound(m_sortedPaths.begin(), m_sortedPaths.end(), prefix);
	return it != m_sortedPaths.end() && it->compare(0, prefix.size(), prefix) == 0;
}

std::vector<AmprPack::DirEntry> AmprPack::ListDirectory(std::string_view member) const {
	std::vector<DirEntry> result;
	const auto            prefix = DirectoryPrefix(member);
	if (prefix.empty()) {
		return result;
	}
	std::unordered_map<std::string, bool> children;
	for (auto it = std::lower_bound(m_sortedPaths.begin(), m_sortedPaths.end(), prefix);
	     it != m_sortedPaths.end() && it->compare(0, prefix.size(), prefix) == 0; ++it) {
		const auto found = m_folded.find(*it);
		if (found == m_folded.end()) {
			continue;
		}
		const uint8_t* record = m_files + uint64_t(found->second) * 32;
		const std::string_view path(
		    reinterpret_cast<const char*>(m_strings + Le32(record + 20)), Le16(record + 24));
		if (path.size() <= prefix.size()) {
			continue;
		}
		const auto remainder  = path.substr(prefix.size());
		const auto separator  = remainder.find('/');
		auto&      isFile     = children[std::string(remainder.substr(0, separator))];
		isFile                = isFile || separator == std::string_view::npos;
	}
	result.reserve(children.size());
	for (auto& child: children) {
		result.push_back({child.first, child.second});
	}
	std::sort(result.begin(), result.end(),
	          [](const DirEntry& a, const DirEntry& b) { return a.name < b.name; });
	return result;
}

int AmprPack::VolumeFd(uint32_t volume) {
	int& fd = m_volumeFds[volume];
	if (fd >= 0) {
		return fd;
	}
	char name[32];
	std::snprintf(name, sizeof(name), "ampr_zstd-%03u.pak", volume);
	fd = ::open((m_root / name).c_str(), O_RDONLY | O_CLOEXEC);
	if (fd >= 0) {
		// Chunks are scattered across volumes; readahead on a 41 KiB chunk wastes disk I/O.
		::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
	}
	return fd;
}

bool AmprPack::Read(const FileInfo& file, uint64_t offset, size_t len, void* out) {
	if (m_index == nullptr || file.firstChunk > m_chunkCount ||
	    file.chunkCount > m_chunkCount - file.firstChunk) {
		return false;
	}
	if (offset >= file.size) {
		return len == 0;
	}
	len = static_cast<size_t>(std::min<uint64_t>(len, file.size - offset));
	auto* destination = static_cast<uint8_t*>(out);
	uint8_t* const stored = m_scratch.data();
	uint8_t* const decoded = m_scratch.data() + kChunkSize;
	uint64_t left = len;
	for (uint32_t k = 0; k < file.chunkCount && left != 0; ++k) {
		const uint64_t base = uint64_t(k) * kChunkSize;
		if (base >= file.size) {
			break;
		}
		const uint64_t decodedSize = std::min<uint64_t>(kChunkSize, file.size - base);
		if (offset >= base + decodedSize) {
			continue;
		}
		const uint64_t word = Le64(m_chunks + (uint64_t(file.firstChunk) + k) * 8);
		const uint64_t chunkOffset = word & 0xffffffffu;
		const uint32_t volume = static_cast<uint32_t>((word >> 32) & 0xfffu);
		const size_t storedSize = static_cast<size_t>(((word >> 44) & 0x3ffffu) + 1);
		const uint32_t codec = static_cast<uint32_t>(word >> 62);
		if (volume >= m_volumeCount || codec > 1) {
			return false;
		}
		const uint64_t volumeSize = Le64(m_volumes + uint64_t(volume) * 8);
		if (chunkOffset > volumeSize || storedSize > volumeSize - chunkOffset) {
			return false;
		}
		const int fd = VolumeFd(volume);
		if (fd < 0 || !ReadExact(fd, stored, storedSize, chunkOffset)) {
			return false;
		}
		const uint8_t* source = stored;
		if (codec == 1) {
			if (ZSTD_getFrameContentSize(stored, storedSize) != decodedSize) {
				return false;
			}
			const size_t written = ZSTD_decompressDCtx(m_dctx, decoded, decodedSize, stored, storedSize);
			if (ZSTD_isError(written) || written != decodedSize) {
				return false;
			}
			source = decoded;
		} else if (storedSize != decodedSize) {
			return false;
		}
		const uint64_t local = offset - base;
		const size_t take = static_cast<size_t>(std::min<uint64_t>(left, decodedSize - local));
		std::memcpy(destination, source + local, take);
		destination += take;
		offset += take;
		left -= take;
	}
	return left == 0;
}

} // namespace Common

#else

namespace Common {

AmprPack::~AmprPack() = default;

bool AmprPack::Init(const std::filesystem::path&) {
	return false;
}

uint32_t AmprPack::InitCount() {
	return 0;
}

void AmprPack::ResetInitCount() {}

bool AmprPack::Lookup(std::string_view, FileInfo&) const {
	return false;
}

bool AmprPack::Exists(std::string_view) const {
	return false;
}

bool AmprPack::IsDirectory(std::string_view) const {
	return false;
}

std::vector<AmprPack::DirEntry> AmprPack::ListDirectory(std::string_view) const {
	return {};
}

bool AmprPack::Read(const FileInfo&, uint64_t, size_t, void*) {
	return false;
}

} // namespace Common

#endif
