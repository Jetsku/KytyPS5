#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 4096},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
};

// KYTY_DESCRIPTOR_POOL_SCALE (default 4, 1..64): pools hold this many times the sets and
// descriptors above. A full pool retires with its unused batches and the next one is created (or
// an idle one reset) on the command processor. At the Sky Garden start about 1,000 sets a flip are
// written, about a pool's worth, and the driver calls under Commit (batch allocations, pool
// creation and resets) were about 1% of the command processor's samples. A larger pool only
// changes how often a pool turns over.
uint32_t DescriptorPoolScale() {
	static const uint32_t scale = [] {
		const char* value = std::getenv("KYTY_DESCRIPTOR_POOL_SCALE");
		if (value == nullptr || value[0] == '\0') {
			return 4u;
		}
		return std::clamp(static_cast<uint32_t>(std::strtoul(value, nullptr, 10)), 1u, 64u);
	}();
	return scale;
}

// KYTY_DESCRIPTOR_FIT=1 (default off; TheCruZ/KytyPS5-GTA 7da23f0a): a batch asks the pool only for
// the sets its remaining room holds (from the layout's descriptor counts), a refused allocation
// retires the pool, and a set larger than an empty default pool gets a pool sized for a few of
// them. AMD's driver keeps the room of the sets a failed batch allocation did create, so halving
// a 32-set request on failure could leave an empty pool unable to hold even one large set (GTA V
// ray tracing tables on an RX 9070: exit in Commit).
bool DescriptorFitEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DESCRIPTOR_FIT");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// Sets of a layout that does not fit a default pool that its own pool holds.
constexpr uint32_t LargeSetsPerPool = 4;

vk::DescriptorPoolSize* FindSize(std::vector<vk::DescriptorPoolSize>& sizes, vk::DescriptorType type) {
	const auto size =
	    std::find_if(sizes.begin(), sizes.end(), [&](const auto& entry) { return entry.type == type; });
	return size == sizes.end() ? nullptr : &*size;
}

} // namespace

const std::vector<vk::DescriptorPoolSize>& DescriptorHeap::LayoutSizes(vk::DescriptorSetLayout layout) {
	auto found = m_layout_sizes.find(layout);
	if (found == m_layout_sizes.end()) {
		std::vector<vk::DescriptorPoolSize> sizes;
		{
			std::lock_guard lock(s_layout_mutex);
			if (const auto registered = s_layout_sizes.find(layout); registered != s_layout_sizes.end()) {
				sizes = registered->second;
			}
		}
		found = m_layout_sizes.emplace(layout, std::move(sizes)).first;
	}
	return found->second;
}

void DescriptorHeap::ResetRoom(std::span<const vk::DescriptorPoolSize> sizes, uint32_t sets) {
	m_free.assign(sizes.begin(), sizes.end());
	m_free_sets = sets;
}

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
		if (DescriptorFitEnabled()) {
			// At least the default sizes (a large pool, below, holds more).
			const auto scale = DescriptorPoolScale();
			auto       sizes = DescriptorPoolSizes;
			for (auto& size: sizes) {
				size.descriptorCount *= scale;
			}
			ResetRoom(sizes, DescriptorHeapCount * scale);
		}
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	if (Allocate(layout, fresh_batch)) {
		return fresh_batch.sets[--fresh_batch.size];
	}
	EXIT_IF(!DescriptorFitEnabled());

	// KYTY_DESCRIPTOR_FIT: the set has more descriptors of a type than an empty default pool holds:
	// a pool with the default sizes raised to hold LargeSetsPerPool of its sets. Once retired it is
	// reused like a default pool (its room is then counted as the default sizes, which it holds).
	const auto scale = DescriptorPoolScale();
	std::vector<vk::DescriptorPoolSize> sizes(DescriptorPoolSizes.begin(), DescriptorPoolSizes.end());
	for (auto& size: sizes) {
		size.descriptorCount *= scale;
	}
	for (const auto& needed: LayoutSizes(layout)) {
		const auto count = needed.descriptorCount * LargeSetsPerPool;
		if (auto* size = FindSize(sizes, needed.type); size != nullptr) {
			size->descriptorCount = std::max(size->descriptorCount, count);
		} else {
			sizes.push_back({needed.type, count});
		}
	}
	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	CreateDescriptorPool(sizes);
	m_sets.clear();
	auto& large_batch = m_sets[layout];
	if (!Allocate(layout, large_batch)) {
		std::string counts;
		for (const auto& size: sizes) {
			counts += " " + vk::to_string(size.type) + "=" + std::to_string(size.descriptorCount);
		}
		EXIT("descriptor set does not fit a pool sized for it:%s\n", counts.c_str());
	}
	return large_batch.sets[--large_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	if (DescriptorFitEnabled()) {
		// Only the sets the pool has room for; a refusal anyway retires the pool (no retry).
		const auto& needs = LayoutSizes(layout);
		uint32_t    count = std::min(DescriptorSetBatch, m_free_sets);
		for (const auto& needed: needs) {
			const auto* free = FindSize(m_free, needed.type);
			count = std::min(count, free == nullptr ? 0u : free->descriptorCount / needed.descriptorCount);
		}
		if (count == 0) {
			return false;
		}
		allocate.descriptorSetCount = count;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result != vk::Result::eSuccess) {
			EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
			        result != vk::Result::eErrorFragmentedPool);
			m_free_sets = 0;
			return false;
		}
		m_free_sets -= count;
		for (const auto& needed: needs) {
			if (auto* free = FindSize(m_free, needed.type); free != nullptr) {
				free->descriptorCount -= needed.descriptorCount * count;
			}
		}
		batch.size = count;
		return true;
	}

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool(std::span<const vk::DescriptorPoolSize> custom) {
	const auto scale = DescriptorPoolScale();
	auto       sizes = DescriptorPoolSizes;
	for (auto& size: sizes) {
		size.descriptorCount *= scale;
	}
	const std::span<const vk::DescriptorPoolSize> used =
	    custom.empty() ? std::span<const vk::DescriptorPoolSize>(sizes) : custom;
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount * scale;
	create.poolSizeCount = static_cast<uint32_t>(used.size());
	create.pPoolSizes    = used.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
	ResetRoom(used, create.maxSets);
}

} // namespace Libs::Graphics
