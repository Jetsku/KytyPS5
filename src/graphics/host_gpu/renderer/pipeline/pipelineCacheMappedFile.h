#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEMAPPEDFILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEMAPPEDFILE_H_

#include <cstdint>
#include <filesystem>

namespace Libs::Graphics {

// KYTY_PCACHE_SAVE_MAPPED (PipelineCache::WriteDriverCache; based on Senaxx/KytyPS5 2708521dc's
// Common::MappedFile): a new file of a fixed size, written through a shared mapping. Its pages
// belong to the file, so they are not charged to the commit limit as the same bytes on the heap are.
class PipelineCacheMappedFile final {
public:
	PipelineCacheMappedFile() = default;
	~PipelineCacheMappedFile() { (void)Close(0); }
	PipelineCacheMappedFile(const PipelineCacheMappedFile&)            = delete;
	PipelineCacheMappedFile& operator=(const PipelineCacheMappedFile&) = delete;

	// Creates or replaces the file with `size` bytes and maps it; an open file is closed empty first.
	bool Create(const std::filesystem::path& name, uint64_t size);
	// Flushes, unmaps, cuts the file to final_size bytes and closes it. False when not open or when a
	// step failed.
	bool Close(uint64_t final_size);

	[[nodiscard]] uint8_t* Data() const noexcept { return m_data; }
	[[nodiscard]] uint64_t Size() const noexcept { return m_size; }

private:
	intptr_t m_file = 0;
	uint8_t* m_data = nullptr;
	uint64_t m_size = 0;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHEMAPPEDFILE_H_ */
