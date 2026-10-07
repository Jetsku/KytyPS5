#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/alignment.h"
#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/vramBudget.h"
#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace Libs::Graphics {

namespace {
// KYTY_VRAM_BUDGET_MB=N (default 0: off): the device memory budget the renderer plans with
// (GraphicContext::GetTotalMemoryBudget: the texture and buffer caches' garbage-collection
// thresholds, the image cache pressure policy, the native image pool's limit and its flush under
// pressure) is at most N MiB, as on a card whose driver budget (VK_EXT_memory_budget) is N MiB.
// A smaller driver budget still wins. For cards whose real budget overstates what the game can
// keep resident (other applications), and to measure the caches' behaviour at a smaller budget.
uint64_t BudgetCapBytes() {
	static const uint64_t cap = [] {
		const auto* value = std::getenv("KYTY_VRAM_BUDGET_MB");
		const auto  mib   = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		if (mib != 0) {
			LOGF("VRAM budget cap (KYTY_VRAM_BUDGET_MB): %llu MiB\n", mib);
		}
		return mib * 1024ull * 1024ull;
	}();
	return cap;
}

// Streamed textures churn through a few create-info classes (e.g. 2048^2/4096^2 BC4/BC5/BC7
// with full mip chains, 2.8-22 MB each) at hundreds of images per second. The pool must
// hold roughly one retirement period of that churn or most creates miss: 128 MB kept only
// ~25 2048^2 BC7 images. KYTY_NATIVE_IMAGE_POOL=0 disables the pool (it is on by default);
// KYTY_NATIVE_IMAGE_POOL_MB (default 1024, at most an eighth of the device budget) and
// KYTY_NATIVE_IMAGE_POOL_COUNT (default 1024) bound the retained images.
uint64_t RetiredImageByteLimit(const GraphicContext& graphics) {
	static const uint64_t configured = [] {
		const auto* value = std::getenv("KYTY_NATIVE_IMAGE_POOL_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 1024ull) * 1024ull * 1024ull;
	}();
	static const uint64_t limit = [&graphics] {
		const auto budget = graphics.GetTotalMemoryBudget();
		return budget != 0 ? std::min(configured, budget / 8) : configured;
	}();
	return limit;
}

size_t RetiredImageCountLimit() {
	static const size_t limit = [] {
		const auto* value = std::getenv("KYTY_NATIVE_IMAGE_POOL_COUNT");
		return value != nullptr ? static_cast<size_t>(std::strtoull(value, nullptr, 10)) : size_t {1024};
	}();
	return limit;
}

bool NativeImagePoolEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_NATIVE_IMAGE_POOL");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_NATIVE_IMAGE_POOL_IDLE_MS=N (default 0: off): a retained native image that no create has
// reused for N ms is destroyed (TrimRetiredImages, from the garbage collector). The pool then holds
// what the current churn recycles instead of up to its whole limit of images retired in earlier
// scenes (menus, the galaxy map), which on a steady scene is never reused.
std::chrono::milliseconds RetiredImageIdleLimit() {
	static const auto limit = [] {
		const auto* value = std::getenv("KYTY_NATIVE_IMAGE_POOL_IDLE_MS");
		const auto  ms    = value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull;
		if (ms != 0) {
			LOGF("Native image pool: retained images unused for %llu ms are destroyed "
			     "(KYTY_NATIVE_IMAGE_POOL_IDLE_MS)\n",
			     ms);
		}
		return std::chrono::milliseconds(static_cast<int64_t>(std::min(ms, 3600000ull)));
	}();
	return limit;
}

// The native image pool keeps nothing while device usage is at this mark: the planning budget, or
// with KYTY_VRAM_GC_BUDGET the image collector's trigger (it frees cached images there; holding
// retired native images instead would only move the pressure).
bool PoolPressure(const GraphicContext& graphics) {
	if (!graphics.CanReportMemoryUsage()) {
		return false;
	}
	const auto budget = graphics.GetTotalMemoryBudget();
	const auto mark   = VramBudget::GcEnabled() ? VramBudget::ImageTrigger(budget) : budget;
	return graphics.GetDeviceMemoryUsage() >= mark;
}

bool CanRecycleImage(const vk::ImageCreateInfo& info) {
	const auto allowed_flags = vk::ImageCreateFlagBits::eMutableFormat |
	                           vk::ImageCreateFlagBits::eExtendedUsage |
	                           vk::ImageCreateFlagBits::eBlockTexelViewCompatible |
	                           vk::ImageCreateFlagBits::e2DArrayCompatible;
	return info.pNext == nullptr && info.pQueueFamilyIndices == nullptr &&
	       info.queueFamilyIndexCount == 0 && info.sharingMode == vk::SharingMode::eExclusive &&
	       info.tiling == vk::ImageTiling::eOptimal &&
	       info.initialLayout == vk::ImageLayout::eUndefined && !(info.flags & ~allowed_flags);
}

void DestroyNativeImage(VmaAllocator allocator, vk::Image image, VmaAllocation allocation) {
	Profiler::ScopedFrameWait timing(Profiler::FrameWait::NativeImageDestroy);
	vmaDestroyImage(allocator, image, allocation);
}

// KYTY_VMA_BUDGET_CACHE_MS=N (default 0 = off): VMA fetches the memory budget from the driver
// (vkGetPhysicalDeviceMemoryProperties2 with VkPhysicalDeviceMemoryBudgetPropertiesEXT) again once
// 30 allocations or frees have passed since its last fetch, and buffer/image churn passes that many
// times a second. With N > 0 the allocator's fetch is answered from a driver fetch younger than N ms,
// each heap's usage moved by the device memory VMA allocated and freed since (VMA's device memory
// callbacks), which is how VMA itself estimates usage between its fetches. Only other processes'
// usage and the driver's budget are then up to N ms old.
struct BudgetCache {
	std::mutex                                            mutex;
	PFN_vkGetPhysicalDeviceMemoryProperties2              fetch = nullptr;
	std::chrono::steady_clock::duration                   interval {};
	bool                                                  valid = false;
	std::chrono::steady_clock::time_point                 fetched;
	VkPhysicalDeviceMemoryProperties                      memory {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         usage {};
	std::array<VkDeviceSize, VK_MAX_MEMORY_HEAPS>         budget {};
	std::array<int64_t, VK_MAX_MEMORY_HEAPS>              allocated_at_fetch {};
	std::array<std::atomic<int64_t>, VK_MAX_MEMORY_HEAPS> allocated {};
	std::array<uint32_t, VK_MAX_MEMORY_TYPES>             type_heap {};
};

BudgetCache g_budget_cache;

uint32_t BudgetCacheIntervalMs() {
	static const uint32_t ms = [] {
		const auto* value = std::getenv("KYTY_VMA_BUDGET_CACHE_MS");
		return value != nullptr ? static_cast<uint32_t>(std::min<unsigned long long>(
		                              std::strtoull(value, nullptr, 10), 60000ull))
		                        : 0u;
	}();
	return ms;
}

void VKAPI_PTR BudgetCountAllocation(VmaAllocator /*allocator*/, uint32_t memory_type,
                                     VkDeviceMemory /*memory*/, VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_add(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR BudgetCountFree(VmaAllocator /*allocator*/, uint32_t memory_type,
                               VkDeviceMemory /*memory*/, VkDeviceSize size, void* /*user_data*/) {
	g_budget_cache.allocated[g_budget_cache.type_heap[memory_type]].fetch_sub(
	    static_cast<int64_t>(size), std::memory_order_relaxed);
}

void VKAPI_PTR CachedMemoryProperties2(VkPhysicalDevice                   physical_device,
                                       VkPhysicalDeviceMemoryProperties2* properties) {
	auto& cache  = g_budget_cache;
	auto* budget = static_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(properties->pNext);
	if (budget == nullptr ||
	    budget->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT ||
	    budget->pNext != nullptr) {
		cache.fetch(physical_device, properties);
		return;
	}
	std::scoped_lock lock(cache.mutex);
	const auto       now = std::chrono::steady_clock::now();
	if (!cache.valid || now - cache.fetched >= cache.interval) {
		cache.fetch(physical_device, properties);
		cache.memory = properties->memoryProperties;
		for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
			cache.usage[heap]              = budget->heapUsage[heap];
			cache.budget[heap]             = budget->heapBudget[heap];
			cache.allocated_at_fetch[heap] = cache.allocated[heap].load(std::memory_order_relaxed);
		}
		cache.fetched = now;
		cache.valid   = true;
		return;
	}
	properties->memoryProperties = cache.memory;
	for (uint32_t heap = 0; heap < VK_MAX_MEMORY_HEAPS; heap++) {
		const auto delta =
		    cache.allocated[heap].load(std::memory_order_relaxed) - cache.allocated_at_fetch[heap];
		const auto usage = static_cast<int64_t>(cache.usage[heap]) + delta;
		budget->heapUsage[heap]  = static_cast<VkDeviceSize>(std::max<int64_t>(usage, 0));
		budget->heapBudget[heap] = cache.budget[heap];
	}
}
} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	VmaDeviceMemoryCallbacks memory_callbacks {};
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
		if (const auto interval_ms = BudgetCacheIntervalMs(); interval_ms != 0) {
			auto& cache = g_budget_cache;
			cache.fetch = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
			    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
			        static_cast<VkInstance>(instance), "vkGetPhysicalDeviceMemoryProperties2"));
			if (cache.fetch != nullptr) {
				cache.interval = std::chrono::milliseconds(interval_ms);
				cache.valid    = false;
				const auto properties = physical_device.getMemoryProperties();
				for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
					cache.type_heap[type] = properties.memoryTypes[type].heapIndex;
				}
				functions.vkGetPhysicalDeviceMemoryProperties2KHR = CachedMemoryProperties2;
				memory_callbacks.pfnAllocate = BudgetCountAllocation;
				memory_callbacks.pfnFree     = BudgetCountFree;
				info.pDeviceMemoryCallbacks  = &memory_callbacks;
				LOGF("VMA budget cache: driver budget fetched at most every %u ms "
				     "(KYTY_VMA_BUDGET_CACHE_MS)\n",
				     interval_ms);
			}
		}
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	ClearRetiredImages();
	if (m_sparse_fence != nullptr) {
		device.destroyFence(m_sparse_fence, nullptr);
		m_sparse_fence = nullptr;
	}
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

uint64_t GraphicContext::NativeImageBytes(const VulkanImage& image) const {
	if (image.sparse) {
		return image.sparse->bytes;
	}
	if (image.allocation == nullptr || allocator == nullptr) {
		return 0;
	}
	VmaAllocationInfo info {};
	vmaGetAllocationInfo(allocator, image.allocation, &info);
	return static_cast<uint64_t>(info.size);
}

bool GraphicContext::SparseImageSupported(const vk::ImageCreateInfo& info) {
	const auto key = std::make_tuple(info.format, info.usage, info.flags);
	{
		std::scoped_lock lock(m_sparse_mutex);
		if (const auto found = m_sparse_support.find(key); found != m_sparse_support.end()) {
			return found->second;
		}
	}
	vk::ImageFormatProperties properties {};
	bool supported = physical_device.getImageFormatProperties(info.format, info.imageType, info.tiling,
	                                                          info.usage, info.flags, &properties) ==
	                     vk::Result::eSuccess &&
	                 properties.maxMipLevels >= info.mipLevels;
	if (supported) {
		uint32_t count = 0;
		physical_device.getSparseImageFormatProperties(info.format, info.imageType, info.samples,
		                                               info.usage, info.tiling, &count, nullptr);
		std::vector<vk::SparseImageFormatProperties> formats(count);
		physical_device.getSparseImageFormatProperties(info.format, info.imageType, info.samples,
		                                               info.usage, info.tiling, &count, formats.data());
		supported = std::any_of(formats.begin(), formats.end(), [](const auto& format) {
			return static_cast<bool>(format.aspectMask & vk::ImageAspectFlagBits::eColor);
		});
	}
	std::scoped_lock lock(m_sparse_mutex);
	m_sparse_support[key] = supported;
	return supported;
}

bool GraphicContext::CreateSparseImage(const vk::ImageCreateInfo& info, uint32_t first_level,
                                       VulkanImage& image) {
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);
	if (!sparse_residency_image_enabled || first_level == 0 || first_level >= info.mipLevels ||
	    info.imageType != vk::ImageType::e2D || info.arrayLayers != 1 ||
	    info.samples != vk::SampleCountFlagBits::e1 || info.tiling != vk::ImageTiling::eOptimal ||
	    info.pNext != nullptr || info.sharingMode != vk::SharingMode::eExclusive) {
		return false;
	}
	auto sparse_info = info;
	sparse_info.flags |= vk::ImageCreateFlagBits::eSparseBinding | vk::ImageCreateFlagBits::eSparseResidency;
	if (!SparseImageSupported(sparse_info)) {
		return false;
	}
	vk::Image native = nullptr;
	if (device.createImage(&sparse_info, nullptr, &native) != vk::Result::eSuccess) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	device.getImageMemoryRequirements(native, &requirements);
	uint32_t count = 0;
	device.getImageSparseMemoryRequirements(native, &count, nullptr);
	std::vector<vk::SparseImageMemoryRequirements> sparse_requirements(count);
	device.getImageSparseMemoryRequirements(native, &count, sparse_requirements.data());
	const auto color = std::find_if(sparse_requirements.begin(), sparse_requirements.end(), [](const auto& entry) {
		return static_cast<bool>(entry.formatProperties.aspectMask & vk::ImageAspectFlagBits::eColor);
	});
	if (color == sparse_requirements.end() || requirements.alignment == 0 ||
	    color->formatProperties.imageGranularity.width == 0 ||
	    color->formatProperties.imageGranularity.height == 0 ||
	    color->formatProperties.imageGranularity.depth == 0) {
		device.destroyImage(native, nullptr);
		return false;
	}
	auto state              = std::make_unique<VulkanImage::SparseState>();
	state->granularity      = color->formatProperties.imageGranularity;
	state->block_size       = requirements.alignment;
	state->memory_type_bits = requirements.memoryTypeBits;
	state->tail_first       = color->imageMipTailFirstLod;
	state->tail_offset      = color->imageMipTailOffset;
	state->tail_size        = color->imageMipTailSize;

	image.image            = native;
	image.allocation       = nullptr;
	image.sparse           = std::move(state);
	image.format           = sparse_info.format;
	image.image_type       = sparse_info.imageType;
	image.extent           = sparse_info.extent;
	image.layers           = sparse_info.arrayLayers;
	image.mip_levels       = sparse_info.mipLevels;
	image.samples          = static_cast<uint32_t>(sparse_info.samples);
	image.usage            = sparse_info.usage;
	image.flags            = sparse_info.flags;
	image.state            = {.layout = sparse_info.initialLayout};
	image.subresource_states.clear();
	image.pool_eligible    = false;
	image.pool_create_info = vk::ImageCreateInfo {};
	BindSparseImageLevels(image, first_level);
	if (HangTrace::Enabled()) {
		HangTrace::RecordNativeImage(true, false, static_cast<uint32_t>(sparse_info.format),
		                             sparse_info.extent.width, sparse_info.extent.height,
		                             sparse_info.mipLevels, static_cast<uint32_t>(sparse_info.usage),
		                             image.sparse->bytes);
	}
	return true;
}

void GraphicContext::BindSparseImageLevels(VulkanImage& image, uint32_t first_level) {
	EXIT_IF(!image.sparse || image.image == nullptr);
	auto&      state  = *image.sparse;
	const auto levels = image.mip_levels;
	const auto first  = std::min(first_level, levels - 1);
	if (first >= state.first_bound) {
		return;
	}
	// New levels: [first, end). Levels from tail_first on live in the mip tail, bound as a whole.
	const auto end        = std::min(state.first_bound, levels);
	const auto tail_first = std::min(state.tail_first, levels);
	const bool need_tail  = !state.tail_bound && state.tail_size != 0 && tail_first < levels;
	std::vector<vk::SparseImageMemoryBind> level_binds;
	uint64_t                               bytes = 0;
	const auto level_bytes = [&](uint32_t level) {
		const auto width  = std::max(image.extent.width >> level, 1u);
		const auto height = std::max(image.extent.height >> level, 1u);
		const auto depth  = std::max(image.extent.depth >> level, 1u);
		const auto blocks = uint64_t {(width + state.granularity.width - 1) / state.granularity.width} *
		                    ((height + state.granularity.height - 1) / state.granularity.height) *
		                    ((depth + state.granularity.depth - 1) / state.granularity.depth);
		return blocks * state.block_size;
	};
	for (uint32_t level = first; level < std::min(end, tail_first); ++level) {
		bytes += level_bytes(level);
	}
	const uint64_t tail_bytes = need_tail ? Common::AlignUp(state.tail_size, state.block_size) : 0;
	bytes += tail_bytes;
	if (bytes == 0) {
		state.first_bound = first;
		return;
	}
	VkMemoryRequirements requirements {};
	requirements.size           = bytes;
	requirements.alignment      = state.block_size;
	requirements.memoryTypeBits = state.memory_type_bits;
	VmaAllocationCreateInfo create {};
	create.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	VmaAllocation     allocation = nullptr;
	VmaAllocationInfo info {};
	RequireVulkanSuccess(static_cast<vk::Result>(
	                         vmaAllocateMemory(allocator, &requirements, &create, &allocation, &info)),
	                     "allocate sparse image memory");
	EXIT_IF(info.offset % state.block_size != 0);
	uint64_t offset = info.offset;
	for (uint32_t level = first; level < std::min(end, tail_first); ++level) {
		vk::SparseImageMemoryBind bind {};
		bind.subresource  = vk::ImageSubresource {vk::ImageAspectFlagBits::eColor, level, 0};
		bind.offset       = vk::Offset3D {0, 0, 0};
		bind.extent       = vk::Extent3D {std::max(image.extent.width >> level, 1u),
                                    std::max(image.extent.height >> level, 1u),
                                    std::max(image.extent.depth >> level, 1u)};
		bind.memory       = info.deviceMemory;
		bind.memoryOffset = offset;
		level_binds.push_back(bind);
		offset += level_bytes(level);
	}
	vk::SparseMemoryBind tail_bind {};
	if (need_tail) {
		tail_bind.resourceOffset = state.tail_offset;
		tail_bind.size           = state.tail_size;
		tail_bind.memory         = info.deviceMemory;
		tail_bind.memoryOffset   = offset;
	}
	vk::SparseImageMemoryBindInfo       level_info {image.image, static_cast<uint32_t>(level_binds.size()),
                                                  level_binds.data()};
	vk::SparseImageOpaqueMemoryBindInfo tail_info {image.image, 1, &tail_bind};
	vk::BindSparseInfo                  bind_info {};
	bind_info.imageBindCount       = level_binds.empty() ? 0u : 1u;
	bind_info.pImageBinds          = &level_info;
	bind_info.imageOpaqueBindCount = need_tail ? 1u : 0u;
	bind_info.pImageOpaqueBinds    = &tail_info;
	{
		std::scoped_lock lock(m_sparse_mutex);
		if (m_sparse_fence == nullptr) {
			vk::FenceCreateInfo fence_info {};
			RequireVulkanSuccess(device.createFence(&fence_info, nullptr, &m_sparse_fence),
			                     "create sparse binding fence");
		}
		vk::Result result {};
		if (side_queue != nullptr) {
			Common::LockGuard queue_lock(side_queue_mutex);
			result = side_queue.bindSparse(1, &bind_info, m_sparse_fence);
		} else {
			Common::LockGuard queue_lock(queue_mutex);
			result = queue.bindSparse(1, &bind_info, m_sparse_fence);
		}
		RequireVulkanSuccess(result, "bind sparse image memory");
		RequireVulkanSuccess(device.waitForFences(1, &m_sparse_fence, VK_TRUE, UINT64_MAX),
		                     "wait for sparse image binding");
		RequireVulkanSuccess(device.resetFences(1, &m_sparse_fence), "reset sparse binding fence");
	}
	state.allocations.push_back(allocation);
	state.bytes += bytes;
	state.first_bound = first;
	state.tail_bound |= need_tail;
	VramStats::Note(VramStats::Kind::Image, true, static_cast<int64_t>(bytes));
}

void GraphicContext::ClearRetiredImages() {
	std::scoped_lock lock(m_retired_image_mutex);
	for (const auto& retired: m_retired_images) {
		DestroyNativeImage(allocator, retired.image, retired.allocation);
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolRemovedBytes,
	                          m_retired_image_bytes);
	m_retired_images.clear();
	m_retired_image_bytes = 0;
}

uint64_t GraphicContext::ReleaseRetainedMemory() {
	uint64_t bytes = 0;
	{
		std::scoped_lock lock(m_retired_image_mutex);
		bytes = m_retired_image_bytes;
	}
	ClearRetiredImages();
	return bytes;
}

void GraphicContext::TrimRetiredImages() {
	const auto idle = RetiredImageIdleLimit();
	if (idle.count() == 0 || allocator == nullptr) {
		return;
	}
	// Called per garbage collection (every completed submission): look at most every 100 ms.
	const auto now = std::chrono::steady_clock::now();
	if (now < m_retired_trim_next) {
		return;
	}
	m_retired_trim_next = now + std::chrono::milliseconds(100);
	std::scoped_lock lock(m_retired_image_mutex);
	// Retired in order: the oldest entries are at the front.
	size_t expired = 0;
	while (expired < m_retired_images.size() && now - m_retired_images[expired].retired >= idle) {
		const auto& retired = m_retired_images[expired];
		DestroyNativeImage(allocator, retired.image, retired.allocation);
		EXIT_IF(retired.bytes > m_retired_image_bytes);
		m_retired_image_bytes -= retired.bytes;
		Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolRemovedBytes, retired.bytes);
		++expired;
	}
	if (expired != 0) {
		m_retired_images.erase(m_retired_images.begin(),
		                       m_retired_images.begin() + static_cast<std::ptrdiff_t>(expired));
	}
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

void GraphicContext::ReportVramStats() {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}
	using VramStats::ToMiB;
	const auto& properties = GetPhysicalDeviceMemoryProperties();
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	VmaTotalStatistics total {};
	vmaCalculateStatistics(allocator, &total);
	for (uint32_t heap = 0; heap < properties.memoryHeapCount; heap++) {
		const bool  local = static_cast<bool>(properties.memoryHeaps[heap].flags &
                                             vk::MemoryHeapFlagBits::eDeviceLocal);
		const auto& stats = total.memoryHeap[heap].statistics;
		const auto  usage = static_cast<uint64_t>(budgets[heap].usage);
		const auto  block = static_cast<uint64_t>(stats.blockBytes);
		VramStats::Line("heap %u %s size=%.0f: usage=%.1f budget=%.1f | VMA blocks=%.1f (%u) "
		                "allocations=%.1f (%u) slack=%.1f | outside VMA=%.1f MiB",
		                heap, local ? "device" : "system",
		                ToMiB(static_cast<uint64_t>(properties.memoryHeaps[heap].size)), ToMiB(usage),
		                ToMiB(static_cast<uint64_t>(budgets[heap].budget)), ToMiB(block), stats.blockCount,
		                ToMiB(static_cast<uint64_t>(stats.allocationBytes)), stats.allocationCount,
		                ToMiB(block - std::min(block, static_cast<uint64_t>(stats.allocationBytes))),
		                ToMiB(usage - std::min(usage, block)));
	}
	for (uint32_t type = 0; type < properties.memoryTypeCount; type++) {
		const auto& stats = total.memoryType[type].statistics;
		if (stats.blockCount == 0) {
			continue;
		}
		VramStats::Line("type %u (heap %u, flags 0x%x): VMA blocks=%.1f (%u) allocations=%.1f (%u) MiB",
		                type, properties.memoryTypes[type].heapIndex,
		                static_cast<uint32_t>(properties.memoryTypes[type].propertyFlags),
		                ToMiB(static_cast<uint64_t>(stats.blockBytes)), stats.blockCount,
		                ToMiB(static_cast<uint64_t>(stats.allocationBytes)), stats.allocationCount);
	}
	std::string kinds;
	for (size_t index = 0; index < static_cast<size_t>(VramStats::Kind::Count); index++) {
		const auto kind   = static_cast<VramStats::Kind>(index);
		const auto totals = VramStats::Totals(kind);
		char       part[160];
		std::snprintf(part, sizeof(part), "%s%s device=%.1f (%lld) system=%.1f (%lld)",
		              index == 0 ? "" : " | ", VramStats::KindName(kind), ToMiB(totals.bytes[1]),
		              static_cast<long long>(totals.count[1]), ToMiB(totals.bytes[0]),
		              static_cast<long long>(totals.count[0]));
		kinds += part;
	}
	VramStats::Line("kinds MiB (count): %s", kinds.c_str());
	uint64_t pool_bytes = 0;
	size_t   pool_count = 0;
	{
		std::scoped_lock lock(m_retired_image_mutex);
		pool_bytes = m_retired_image_bytes;
		pool_count = m_retired_images.size();
	}
	VramStats::Line("native image pool: %.1f MiB in %zu images (limit %.1f MiB / %zu, %s); "
	                "device usage %.1f of budget %.1f MiB (GetTotalMemoryBudget)",
	                ToMiB(pool_bytes), pool_count, ToMiB(RetiredImageByteLimit(*this)),
	                RetiredImageCountLimit(), NativeImagePoolEnabled() ? "on" : "off",
	                ToMiB(GetDeviceMemoryUsage()), ToMiB(GetTotalMemoryBudget()));
	// The driver's local memory (outside VMA) follows the largest per-invocation footprint.
	const auto function_storage = VramStats::FunctionStorageMax();
	VramStats::Line("Function-storage arrays: largest per invocation %llu bytes declared, %llu bytes given to "
	                "the driver (%llu modules with such arrays; KYTY_FUNCTION_ARRAY_SHRINK %s)",
	                static_cast<unsigned long long>(function_storage.declared),
	                static_cast<unsigned long long>(function_storage.created),
	                static_cast<unsigned long long>(function_storage.modules),
	                function_storage.created < function_storage.declared ? "shrank them" : "off or no effect");
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (const auto cap = BudgetCapBytes(); cap != 0) {
		budget = std::min(budget, cap);
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	const bool recycle = NativeImagePoolEnabled() && CanRecycleImage(image_info);
	if (recycle) {
		if (PoolPressure(*this)) {
			ClearRetiredImages();
		}
		std::scoped_lock lock(m_retired_image_mutex);
		for (size_t i = m_retired_images.size(); i > 0; --i) {
			const auto& retired = m_retired_images[i - 1];
			if (retired.create == image_info) {
				image.image = retired.image;
				image.allocation = retired.allocation;
				EXIT_IF(retired.bytes > m_retired_image_bytes);
				m_retired_image_bytes -= retired.bytes;
				Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolRemovedBytes,
				                          retired.bytes);
				m_retired_images.erase(m_retired_images.begin() + static_cast<std::ptrdiff_t>(i - 1));
				break;
			}
		}
		Profiler::CountFrameEvent(image.image != nullptr ? Profiler::FrameEvent::NativeImagePoolHits
		                                                 : Profiler::FrameEvent::NativeImagePoolMisses);
	}
	const bool pool_hit = image.image != nullptr;
	if (image.image == nullptr) {
		VmaAllocationCreateInfo alloc_info {};
		alloc_info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		auto allocate = [&] {
			Profiler::ScopedFrameWait timing(Profiler::FrameWait::NativeImageCreate);
			vk::Image::CType native_image = VK_NULL_HANDLE;
			VmaAllocation allocation = nullptr;
			const auto result = static_cast<vk::Result>(vmaCreateImage(
			    allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
			    &alloc_info, &native_image, &allocation, nullptr));
			if (result == vk::Result::eSuccess) {
				image.image = native_image;
				image.allocation = allocation;
			}
			return result;
		};
		auto result = allocate();
		if (result == vk::Result::eErrorOutOfDeviceMemory && NativeImagePoolEnabled()) {
			// Retained objects are optional. Release them before one allocation retry.
			ClearRetiredImages();
			result = allocate();
		}
		if (result != vk::Result::eSuccess) {
			LogMemoryBudget();
			return false;
		}
	}

	if (HangTrace::Enabled()) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		HangTrace::RecordNativeImage(true, pool_hit, static_cast<uint32_t>(image_info.format),
		                             image_info.extent.width, image_info.extent.height,
		                             image_info.mipLevels, static_cast<uint32_t>(image_info.usage),
		                             static_cast<uint64_t>(allocation_info.size));
	}
	if (VramStats::Enabled()) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		VramStats::Note(VramStats::Kind::Image, true, static_cast<int64_t>(allocation_info.size));
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();
	image.pool_eligible = recycle;
	image.pool_create_info = recycle ? image_info : vk::ImageCreateInfo {};

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	if (image.sparse) {
		// KYTY_TEXTURE_SPARSE_RESIDENCY: the image, then the memory bound to it (never pooled).
		EXIT_IF(allocator == nullptr || image.image == nullptr);
		if (HangTrace::Enabled()) {
			HangTrace::RecordNativeImage(false, false, static_cast<uint32_t>(image.format),
			                             image.extent.width, image.extent.height, image.mip_levels,
			                             static_cast<uint32_t>(image.usage), image.sparse->bytes);
		}
		VramStats::Note(VramStats::Kind::Image, true, -static_cast<int64_t>(image.sparse->bytes));
		{
			Profiler::ScopedFrameWait timing(Profiler::FrameWait::NativeImageDestroy);
			device.destroyImage(image.image, nullptr);
			for (const auto allocation: image.sparse->allocations) {
				vmaFreeMemory(allocator, allocation);
			}
		}
		image.sparse.reset();
		image.image         = nullptr;
		image.allocation    = nullptr;
		image.pool_eligible = false;
		return;
	}
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	// This is the existing destruction boundary: Image's views are already destroyed,
	// and TextureCache's deferred callback has waited for native completion/publication.
	// Only native storage is retained; no guest address, content-validity or view survives.
	if (HangTrace::Enabled()) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		HangTrace::RecordNativeImage(false, false, static_cast<uint32_t>(image.format),
		                             image.extent.width, image.extent.height, image.mip_levels,
		                             static_cast<uint32_t>(image.usage),
		                             static_cast<uint64_t>(allocation_info.size));
	}
	if (VramStats::Enabled()) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		VramStats::Note(VramStats::Kind::Image, true, -static_cast<int64_t>(allocation_info.size));
	}
	bool retained = false;
	if (image.pool_eligible && NativeImagePoolEnabled()) {
		const bool pressure = PoolPressure(*this);
		if (pressure) {
			ClearRetiredImages();
		} else {
			VmaAllocationInfo allocation_info {};
			vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
			const auto bytes       = static_cast<uint64_t>(allocation_info.size);
			const auto byte_limit  = RetiredImageByteLimit(*this);
			const auto count_limit = RetiredImageCountLimit();
			if (bytes <= byte_limit && count_limit != 0) {
				std::scoped_lock lock(m_retired_image_mutex);
				while (!m_retired_images.empty() &&
				       (m_retired_images.size() >= count_limit ||
				        bytes > byte_limit - m_retired_image_bytes)) {
					const auto oldest = m_retired_images.front();
					DestroyNativeImage(allocator, oldest.image, oldest.allocation);
					m_retired_image_bytes -= oldest.bytes;
					Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolRemovedBytes,
					                          oldest.bytes);
					m_retired_images.erase(m_retired_images.begin());
				}
				m_retired_images.push_back({image.pool_create_info, image.image, image.allocation, bytes,
				                            std::chrono::steady_clock::now()});
				m_retired_image_bytes += bytes;
				Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolRetires);
				Profiler::CountFrameEvent(Profiler::FrameEvent::NativeImagePoolAddedBytes, bytes);
				retained = true;
			}
		}
	}
	if (!retained) {
		DestroyNativeImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
	image.pool_eligible = false;
}

} // namespace Libs::Graphics
