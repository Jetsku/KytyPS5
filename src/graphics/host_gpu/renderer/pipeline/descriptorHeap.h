#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);

	// The descriptor counts per type of a layout whose sets come from a heap (any thread), for
	// KYTY_DESCRIPTOR_FIT (descriptorHeap.cpp). Inline: layout code links without the heap.
	static void RegisterLayout(vk::DescriptorSetLayout                         layout,
	                           std::span<const vk::DescriptorSetLayoutBinding> bindings) {
		std::vector<vk::DescriptorPoolSize> sizes;
		for (const auto& binding: bindings) {
			if (binding.descriptorCount == 0) {
				continue;
			}
			bool merged = false;
			for (auto& size: sizes) {
				if (size.type == binding.descriptorType) {
					size.descriptorCount += binding.descriptorCount;
					merged = true;
					break;
				}
			}
			if (!merged) {
				sizes.push_back({binding.descriptorType, binding.descriptorCount});
			}
		}
		std::lock_guard lock(s_layout_mutex);
		s_layout_sizes[layout] = std::move(sizes);
	}

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool(std::span<const vk::DescriptorPoolSize> sizes = {});
	void               ResetRoom(std::span<const vk::DescriptorPoolSize> sizes, uint32_t sets);
	[[nodiscard]] const std::vector<vk::DescriptorPoolSize>& LayoutSizes(vk::DescriptorSetLayout layout);

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
	// KYTY_DESCRIPTOR_FIT: the room the current pool has left (descriptors per type, sets).
	std::vector<vk::DescriptorPoolSize>                                              m_free;
	uint32_t                                                                         m_free_sets = 0;
	std::unordered_map<vk::DescriptorSetLayout, std::vector<vk::DescriptorPoolSize>> m_layout_sizes;
	inline static std::mutex s_layout_mutex;
	inline static std::unordered_map<vk::DescriptorSetLayout, std::vector<vk::DescriptorPoolSize>> s_layout_sizes;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
