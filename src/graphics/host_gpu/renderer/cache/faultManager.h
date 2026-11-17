#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_

#include "common/abi.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
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
	// mapped: the guest GPU mappings [begin, end) (KYTY_FAULT_SCAN_MAPPED scans only their bitmap
	// words; empty = the whole bitmap).
	void ProcessFaultBuffer(const std::vector<std::pair<uint64_t, uint64_t>>& mapped = {});
	// KYTY_FAULT_SCAN_MAPPED is on (the caller then passes the mappings).
	[[nodiscard]] static bool ScanMappedActive() noexcept;

	// KYTY_BDA_WRITES. Before a dispatch that writes through BDA: clears the written-page bitmap and
	// the dropped-write counter the first time (the fault buffer's second half, see BufferCache).
	void PrepareBdaWrites();
	// After it: records the compaction of the written-page bitmap into page addresses (clearing
	// the bitmap) and a copy of the dropped-write counter (then clearing it), submits, waits for
	// that tick and returns what the dispatch wrote. overflow: more pages than the list holds
	// (the bits past it are lost; the caller must settle conservatively).
	// stores (KYTY_BDA_STORE_LOG): the store log, in log order; log_overflow: stores past its
	// capacity (also counted in dropped). log_tail: entries the device log holds past the downloaded
	// prefix (BufferCache::BDA_STORE_LOG_PREFIX): CollectBdaWrites fetches them (then 0); after
	// ParseBdaWrites alone they are not in stores, and the caller counts them as dropped.
	struct BdaStore {
		uint32_t address_low  = 0;
		uint32_t address_high = 0;
		uint32_t value        = 0;
		uint32_t mask         = 0;
		[[nodiscard]] uint64_t Address() const {
			return address_low | (static_cast<uint64_t>(address_high) << 32u);
		}
	};
	struct BdaWrites {
		std::vector<uint64_t> pages;
		std::vector<BdaStore> stores;
		uint32_t              dropped      = 0;
		uint32_t              log_overflow = 0;
		uint32_t              log_tail     = 0;
		bool                  overflow     = false;
	};
	[[nodiscard]] BdaWrites CollectBdaWrites();
	// KYTY_BDA_ASYNC_SETTLE: CollectBdaWrites in two halves. RecordBdaWrites records the compaction
	// into one of BdaSlots download slots (the caller keeps at most BdaSlots unparsed) and returns
	// it; ParseBdaWrites reads a slot once the recording's tick completed.
	static constexpr uint32_t BdaSlots = 8;
	[[nodiscard]] uint32_t  RecordBdaWrites();
	[[nodiscard]] BdaWrites ParseBdaWrites(uint32_t slot);

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
	// KYTY_FAULT_SCAN_MAPPED: the same scan created with DISPATCH_BASE (created on first use).
	vk::Pipeline                               m_fault_process_pipeline_base = nullptr;
	// KYTY_BDA_WRITES only (created on first use).
	vk::Pipeline                               m_bda_pages_pipeline = nullptr;
	std::array<std::unique_ptr<Buffer>, BdaSlots> m_bda_downloads;
	// The store log past its downloaded prefix (created on first use, FetchBdaStoreLogTail).
	std::unique_ptr<Buffer>                    m_bda_tail_download;
	void                                       FetchBdaStoreLogTail(BdaWrites& writes);
	uint32_t                                   m_bda_next_slot = 0;
	bool                                       m_bda_region_cleared = false;
	// KYTY_BDA_TAGS_GPU_COUNT (diagnostic): the counters' copies per area and their last values.
	std::unique_ptr<Buffer>                    m_tag_count_download;
	std::array<uint32_t, 3>                    m_tag_count_last {};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_FAULTMANAGER_H_
