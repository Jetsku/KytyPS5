#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class BufferCache;

class FaultManager {
	static constexpr size_t MaxPendingFaults = 8;

public:
	FaultManager(GraphicContext& graphics, CommandScheduler& scheduler, BufferCache& buffer_cache);
	~FaultManager();
	KYTY_CLASS_NO_COPY(FaultManager);

	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return &m_fault_buffer; }
	void                  ProcessFaultBuffer();

	// KYTY_BDA_WRITES. Before a dispatch that writes through BDA: clears the written-page bitmap and
	// the dropped-write counter the first time (the fault buffer's second half, see BufferCache).
	void PrepareBdaWrites();
	// After it: records the compaction of the written-page bitmap into page addresses (clearing
	// the bitmap) and a copy of the dropped-write counter (then clearing it), submits, waits for
	// that tick and returns what the dispatch wrote. overflow: more pages than the list holds
	// (the bits past it are lost; the caller must settle conservatively).
	struct BdaWrites {
		std::vector<uint64_t> pages;
		uint32_t              dropped  = 0;
		bool                  overflow = false;
	};
	[[nodiscard]] BdaWrites CollectBdaWrites();

private:
	GraphicContext&                            m_graphics;
	CommandScheduler&                          m_scheduler;
	BufferCache&                               m_buffer_cache;
	Buffer                                     m_fault_buffer;
	Buffer                                     m_download_buffer;
	std::array<uint64_t, MaxPendingFaults>      m_fault_areas {};
	uint32_t                                   m_current_area = 0;
	vk::DescriptorSetLayout                    m_fault_process_desc_layout = nullptr;
	vk::Pipeline                               m_fault_process_pipeline = nullptr;
	vk::PipelineLayout                         m_fault_process_pipeline_layout = nullptr;
	// KYTY_BDA_WRITES only (created on first use).
	vk::Pipeline                               m_bda_pages_pipeline = nullptr;
	std::unique_ptr<Buffer>                    m_bda_download;
	bool                                       m_bda_region_cleared = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
