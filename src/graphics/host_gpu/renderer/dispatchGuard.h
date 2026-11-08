#pragma once

#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
class StreamBuffer;

// KYTY_INDIRECT_GUARD (default off): a DispatchIndirect reads its workgroup counts from a guard slot
// that gpu_dispatch_guard.comp filled from the guest record, zeroed when the counts are absurd.
class DispatchGuard {
public:
	explicit DispatchGuard(RenderContext& context);
	~DispatchGuard();
	DispatchGuard(const DispatchGuard&)            = delete;
	DispatchGuard& operator=(const DispatchGuard&) = delete;

	static bool Enabled();

	struct Slot {
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
	};

	// Preparation phase (may submit the recording on ring wrap).
	[[nodiscard]] Slot Reserve();
	// Records the guard into `slot`; the caller dispatches from slot.buffer at slot.offset.
	// local_size non-zero (KYTY_INDIRECT_THREADS_GPU): the record holds thread counts, converted to
	// workgroup counts; the thread counts stay at SlotThreads(slot). guard false: no limits, no
	// check (conversion only).
	void Record(CommandBuffer& buffer, const Slot& slot, vk::Buffer args, uint64_t args_offset,
	            uint64_t shader_hash, const std::array<uint32_t, 3>& local_size = {0, 0, 0},
	            bool guard = true);
	// Device address of the slot's copy of the record's raw counts (dwords 4-6).
	[[nodiscard]] vk::DeviceAddress SlotThreads(const Slot& slot) const;

private:
	void Initialize();
	void Check(const Slot& slot, uint64_t shader_hash);

	RenderContext&                m_context;
	vk::DescriptorSetLayout       m_descriptors = nullptr;
	vk::PipelineLayout            m_layout      = nullptr;
	vk::Pipeline                  m_pipeline    = nullptr;
	std::unique_ptr<StreamBuffer> m_ring;
};

} // namespace Libs::Graphics
