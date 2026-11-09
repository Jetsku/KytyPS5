#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTGAMEFILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTGAMEFILE_H_

#include "common/archive.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <system_error>

namespace Libs::Graphics {

// A file of the game that the pipeline list reads shader code from (KYTY_PIPELINE_LIST,
// pipelineCache.cpp): a host file, read with a stream, or a member of a game archive
// ("game.zar!/eboot.bin", common/archive.h), which a stream cannot open.
class PipelineListGameFile {
public:
	explicit PipelineListGameFile(const std::filesystem::path& path): m_path(path) {
		if (Common::IsArchivePath(path)) {
			m_member = Common::OpenArchiveFile(path);
		} else {
			m_stream.open(path, std::ios::binary);
		}
	}

	// A host file: whether it was opened and its last read was complete (the stream's state).
	[[nodiscard]] bool Ok() const { return m_member != nullptr || (m_stream.is_open() && !m_stream.fail()); }

	// False when the size is not known.
	[[nodiscard]] bool Size(uint64_t& size) const {
		if (m_member != nullptr) {
			size = m_member->Size();
			return true;
		}
		std::error_code error;
		size = std::filesystem::file_size(m_path, error);
		return !error;
	}

	// Fills `out` with the bytes at `offset`; false when the file does not hold them all.
	bool Read(uint64_t offset, std::span<uint8_t> out) {
		if (m_member == nullptr) {
			m_stream.clear();
			m_stream.seekg(static_cast<std::streamoff>(offset));
			return static_cast<bool>(
			    m_stream.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size())));
		}
		if (!m_member->Seek(offset)) return false;
		size_t done = 0;
		while (done < out.size()) {
			const auto step = static_cast<uint32_t>(std::min<size_t>(out.size() - done, size_t {64} << 20u));
			uint32_t   got  = 0;
			m_member->Read(out.data() + done, step, &got);
			if (got == 0) return false;
			done += got;
		}
		return true;
	}

private:
	std::filesystem::path                m_path;
	std::ifstream                        m_stream;
	std::unique_ptr<Common::ArchiveFile> m_member;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELISTGAMEFILE_H_ */
