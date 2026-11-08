#include "graphics/host_gpu/renderer/pipeline/bindlessTable.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>

namespace Libs::Graphics {

bool BindlessEngineEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_ENGINE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint32_t BindlessImagesPerArray(const GraphicContext& graphics) {
	if (!graphics.bindless_enabled) {
		return 0;
	}
	return std::min(BindlessTable::MaxImagesPerArray,
	                graphics.bindless_max_sampled_images / BindlessTable::ImageArrays);
}

vk::DescriptorSetLayout BindlessSetLayout(GraphicContext& graphics) {
	if (!graphics.bindless_enabled) {
		return nullptr;
	}
	Common::LockGuard lock(graphics.bindless_layout_mutex);
	if (graphics.bindless_layout != nullptr) {
		return graphics.bindless_layout;
	}
	const auto images_per_array = BindlessImagesPerArray(graphics);
	if (images_per_array <= BindlessTable::PlaceholderColors) {
		LOGF("Bindless table: the device allows too few update-after-bind sampled images (%u)\n",
		     graphics.bindless_max_sampled_images);
		return nullptr;
	}
	using Binding = BindlessTable::Binding;
	constexpr auto image_flags = vk::DescriptorBindingFlagBits::ePartiallyBound |
	                             vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	                             vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	// The buffers are written once, at creation, so they need no update-after-bind.
	constexpr auto buffer_flags = vk::DescriptorBindingFlags {};
	std::vector<vk::DescriptorSetLayoutBinding> bindings {{
	    {Binding::Images2D, vk::DescriptorType::eSampledImage, images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Binding::Images2DArray, vk::DescriptorType::eSampledImage, images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Binding::ImagesCube, vk::DescriptorType::eSampledImage, images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Binding::Images3D, vk::DescriptorType::eSampledImage, images_per_array,
	     vk::ShaderStageFlagBits::eAll, nullptr},
	    {Binding::Translation, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	    {Binding::Feedback, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eAll,
	     nullptr},
	}};
	std::vector<vk::DescriptorBindingFlags> binding_flags {
	    image_flags, image_flags, image_flags, image_flags, buffer_flags, buffer_flags};
	// KYTY_BINDLESS_SAMPLERS: the sampler array (update-after-bind, like the image arrays).
	if (BindlessTable::SamplersEnabled()) {
		bindings.push_back({BindlessTable::SamplersBinding, vk::DescriptorType::eSampler,
		                    BindlessTable::SamplerSlots(graphics), vk::ShaderStageFlagBits::eAll,
		                    nullptr});
		binding_flags.push_back(image_flags);
	}
	// KYTY_RUNTIME_IMAGES: the runtime image table and request buffer.
	if (BindlessTable::RuntimeImagesEnabled()) {
		for (const auto binding: {BindlessTable::RuntimeTableBinding, BindlessTable::RuntimeRequestsBinding}) {
			bindings.push_back({binding, vk::DescriptorType::eStorageBuffer, 1,
			                    vk::ShaderStageFlagBits::eAll, nullptr});
			binding_flags.push_back(buffer_flags);
		}
	}
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = static_cast<uint32_t>(binding_flags.size());
	flags_info.pBindingFlags = binding_flags.data();
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	vk::DescriptorSetLayout layout = nullptr;
	if (graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &layout) !=
	    vk::Result::eSuccess) {
		LOGF("Bindless table: creating the descriptor set layout failed\n");
		return nullptr;
	}
	graphics.bindless_layout = layout;
	return layout;
}

BindlessTable::BindlessTable(GraphicContext& graphics, CommandScheduler& scheduler,
                             PageManager& page_manager)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager) {}

BindlessTable::~BindlessTable() {
	Destroy();
	Common::LockGuard lock(m_graphics.bindless_layout_mutex);
	if (m_graphics.bindless_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_graphics.bindless_layout, nullptr);
		m_graphics.bindless_layout = nullptr;
	}
}

void BindlessTable::Destroy() {
	for (uint32_t i = 0; i < Placeholders; i++) {
		if (m_placeholder_views[i] != nullptr) {
			m_graphics.device.destroyImageView(m_placeholder_views[i], nullptr);
			m_placeholder_views[i] = nullptr;
		}
		if (m_placeholders[i].image != nullptr) {
			m_graphics.DeleteImage(m_placeholders[i]);
		}
	}
	m_translation.reset();
	m_feedback.reset();
	m_feedback_snapshots[0].reset();
	m_feedback_snapshots[1].reset();
	m_runtime_table.reset();
	m_runtime_requests.reset();
	m_runtime_requests_snapshot.reset();
	if (m_pool != nullptr) {
		m_graphics.device.destroyDescriptorPool(m_pool, nullptr);
		m_pool = nullptr;
	}
	m_set = nullptr;
}

bool BindlessTable::Supported() const noexcept {
	return m_graphics.bindless_enabled;
}

bool BindlessTable::EnsureCreated() {
	if (m_set != nullptr) {
		return true;
	}
	if (m_creation_failed || !m_graphics.bindless_enabled || !m_scheduler.Active() ||
	    m_scheduler.Current().IsInvalid()) {
		return false;
	}
	const auto layout = BindlessSetLayout(m_graphics);
	if (layout == nullptr) {
		m_creation_failed = true;
		return false;
	}
	m_images_per_array = BindlessImagesPerArray(m_graphics);

	std::vector<vk::DescriptorPoolSize> sizes {{
	    {vk::DescriptorType::eSampledImage, m_images_per_array * ImageArrays},
	    {vk::DescriptorType::eStorageBuffer, RuntimeImagesEnabled() ? 4u : 2u},
	}};
	if (SamplersEnabled()) {
		m_sampler_slots = SamplerSlots(m_graphics);
		sizes.push_back({vk::DescriptorType::eSampler, m_sampler_slots});
	}
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size());
	pool_info.pPoolSizes    = sizes.data();
	vk::DescriptorSet set   = nullptr;
	if (m_graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool) != vk::Result::eSuccess) {
		LOGF("Bindless table: creating the descriptor pool failed\n");
		m_pool            = nullptr;
		m_creation_failed = true;
		return false;
	}
	vk::DescriptorSetAllocateInfo allocate_info {};
	allocate_info.descriptorPool     = m_pool;
	allocate_info.descriptorSetCount = 1;
	allocate_info.pSetLayouts        = &layout;
	if (m_graphics.device.allocateDescriptorSets(&allocate_info, &set) != vk::Result::eSuccess) {
		LOGF("Bindless table: allocating the descriptor set failed\n");
		Destroy();
		m_creation_failed = true;
		return false;
	}

	// Both live in host-visible memory: the host writes translations and clears feedback flags
	// directly, while shaders access them at device speed where the device allows. The host
	// reads the flags from a host-cached snapshot (RecordFeedbackSnapshot).
	m_translation = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, uint64_t {TranslationCapacity()} * sizeof(uint32_t));
	m_feedback    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
	                                         AllFlags, uint64_t {TranslationCapacity()} * sizeof(uint32_t));
	// Two snapshots: the host reads the one consumed while the next is being copied.
	for (auto& snapshot: m_feedback_snapshots) {
		snapshot = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags,
		                                    uint64_t {SnapshotWords} * sizeof(uint32_t));
	}
	if (m_translation->Mapped().empty() || m_feedback->Mapped().empty() ||
	    m_feedback_snapshots[0]->Mapped().empty() || m_feedback_snapshots[1]->Mapped().empty()) {
		LOGF("Bindless table: the translation or feedback buffer is not host-visible\n");
		Destroy();
		m_creation_failed = true;
		return false;
	}
	// Written on the host before any command that reads them is submitted (Senaxx cleared them
	// in a fenced submission of its own): translation[0], the entry of keys outside their heap,
	// is placeholder slot 1; every other entry and every feedback flag is 0. A region allocated
	// after a snapshot was recorded reads as "no requests" until the next.
	std::fill(m_translation->Mapped().begin(), m_translation->Mapped().end(), uint8_t {0});
	reinterpret_cast<uint32_t*>(m_translation->Mapped().data())[0] = 1u;
	std::fill(m_feedback->Mapped().begin(), m_feedback->Mapped().end(), uint8_t {0});
	m_translation->Flush(0, m_translation->Size());
	m_feedback->Flush(0, m_feedback->Size());
	for (auto& snapshot: m_feedback_snapshots) {
		std::fill(snapshot->Mapped().begin(), snapshot->Mapped().end(), uint8_t {0});
		snapshot->Flush(0, snapshot->Size());
	}
	m_pending_chunks.clear();
	m_next_region = ShaderRecompiler::IR::BindlessFeedbackFirstRegion;
	SetVulkanObjectNameF(m_graphics.device, m_translation->Handle(), "Bindless Translation");
	SetVulkanObjectNameF(m_graphics.device, m_feedback->Handle(), "Bindless Feedback");

	std::vector<vk::DescriptorBufferInfo> buffer_infos {{
	    {m_translation->Handle(), 0, VK_WHOLE_SIZE},
	    {m_feedback->Handle(), 0, VK_WHOLE_SIZE},
	}};
	std::vector<uint32_t> buffer_bindings {Translation, Feedback};
	if (RuntimeImagesEnabled()) {
		// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): the host writes the
		// table; shaders write requests, which the host reads from a snapshot as it does the
		// feedback flags.
		using namespace ShaderRecompiler::IR;
		constexpr vk::DeviceSize table_bytes =
		    vk::DeviceSize {RuntimeImageEntries} * RuntimeImageEntryWords * sizeof(uint32_t);
		constexpr vk::DeviceSize requests_bytes =
		    vk::DeviceSize {RuntimeImageRequestBufferWords} * sizeof(uint32_t);
		m_runtime_table    = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
		                                              AllFlags, table_bytes);
		m_runtime_requests = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream,
		                                              0, AllFlags, requests_bytes);
		m_runtime_requests_snapshot = std::make_unique<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, requests_bytes);
		if (m_runtime_table->Mapped().empty() || m_runtime_requests->Mapped().empty() ||
		    m_runtime_requests_snapshot->Mapped().empty()) {
			LOGF("Bindless table: the runtime image buffers are not host-visible\n");
			Destroy();
			m_creation_failed = true;
			return false;
		}
		for (auto* buffer:
		     {m_runtime_table.get(), m_runtime_requests.get(), m_runtime_requests_snapshot.get()}) {
			std::fill(buffer->Mapped().begin(), buffer->Mapped().end(), uint8_t {0});
			buffer->Flush(0, buffer->Size());
		}
		SetVulkanObjectNameF(m_graphics.device, m_runtime_table->Handle(), "Bindless Runtime Table");
		SetVulkanObjectNameF(m_graphics.device, m_runtime_requests->Handle(),
		                     "Bindless Runtime Requests");
		buffer_infos.push_back({m_runtime_table->Handle(), 0, VK_WHOLE_SIZE});
		buffer_infos.push_back({m_runtime_requests->Handle(), 0, VK_WHOLE_SIZE});
		buffer_bindings.push_back(RuntimeTableBinding);
		buffer_bindings.push_back(RuntimeRequestsBinding);
	}
	std::vector<vk::WriteDescriptorSet> writes(buffer_infos.size());
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstSet          = set;
		writes[i].dstBinding      = buffer_bindings[i];
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &buffer_infos[i];
	}
	m_graphics.device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0,
	                                       nullptr);
	m_set = set;
	if (!CreatePlaceholders()) {
		LOGF("Bindless table: creating the placeholder textures failed\n");
		Destroy();
		m_creation_failed = true;
		return false;
	}
	LOGF("Bindless table: %u images per array, %u translation entries\n", m_images_per_array,
	     TranslationCapacity());
	return true;
}

bool BindlessTable::CreatePlaceholders() {
	struct Kind {
		vk::ImageType        type;
		vk::ImageViewType    view;
		uint32_t             layers;
		vk::ImageCreateFlags flags;
	};
	const std::array<Kind, ImageArrays> kinds {{
	    {vk::ImageType::e2D, vk::ImageViewType::e2D, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::e2DArray, 1, {}},
	    {vk::ImageType::e2D, vk::ImageViewType::eCube, 6, vk::ImageCreateFlagBits::eCubeCompatible},
	    {vk::ImageType::e3D, vk::ImageViewType::e3D, 1, {}},
	}};
	// Placeholder i: view type i % ImageArrays, colour (slot) i / ImageArrays.
	for (uint32_t i = 0; i < Placeholders; i++) {
		const auto&         kind = kinds[i % ImageArrays];
		vk::ImageCreateInfo info {};
		info.flags         = kind.flags;
		info.imageType     = kind.type;
		info.format        = vk::Format::eR8G8B8A8Unorm;
		info.extent        = vk::Extent3D {1, 1, 1};
		info.mipLevels     = 1;
		info.arrayLayers   = kind.layers;
		info.samples       = vk::SampleCountFlagBits::e1;
		info.tiling        = vk::ImageTiling::eOptimal;
		info.usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
		info.initialLayout = vk::ImageLayout::eUndefined;
		if (!m_graphics.CreateImage(info, m_placeholders[i])) {
			return false;
		}
		vk::ImageViewCreateInfo view {};
		view.image            = m_placeholders[i].image;
		view.viewType         = kind.view;
		view.format           = info.format;
		view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, kind.layers};
		if (m_graphics.device.createImageView(&view, nullptr, &m_placeholder_views[i]) !=
		    vk::Result::eSuccess) {
			m_placeholder_views[i] = nullptr;
			return false;
		}
	}

	// Cleared in the command buffer being recorded, outside rendering, before the first command
	// that can sample them (Senaxx used a fenced submission of its own).
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto sink = command.HelperSink();
	std::array<vk::ImageMemoryBarrier, Placeholders> to_transfer {};
	std::array<vk::ImageMemoryBarrier, Placeholders> to_read {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0,
		                                       kinds[i % ImageArrays].layers};
		to_transfer[i].srcAccessMask       = {};
		to_transfer[i].dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		to_transfer[i].oldLayout           = vk::ImageLayout::eUndefined;
		to_transfer[i].newLayout           = vk::ImageLayout::eTransferDstOptimal;
		to_transfer[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		to_transfer[i].image               = m_placeholders[i].image;
		to_transfer[i].subresourceRange    = range;
		to_read[i]                         = to_transfer[i];
		to_read[i].srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
		to_read[i].dstAccessMask           = vk::AccessFlagBits::eShaderRead;
		to_read[i].oldLayout               = vk::ImageLayout::eTransferDstOptimal;
		to_read[i].newLayout               = vk::ImageLayout::eShaderReadOnlyOptimal;
	}
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
	                     {}, 0, nullptr, 0, nullptr, Placeholders, to_transfer.data());
	// Transparent black by default. KYTY_BINDLESS_DEBUG_COLORS=1: slot 0 grey (not resident), slot
	// 1 red (key outside its heap), slot 2 blue (pending), to see which case a surface is in.
	// KYTY_BINDLESS_OPAQUE_PLACEHOLDER=1 (startup, default off; Erlou93/KytyPS5-Yotei v0.1.4
	// KYTY_PERF_BINDLESS_CLASSES): opaque black (0,0,0,1) instead. Ghost of Yotei never samples a
	// transparent black texture: a texture that is not ready is replaced by an opaque 1x1 default
	// before the record is written, so a pending key reading alpha 0 hid alpha-tested or blended
	// surfaces (foliage, decals) while it waited.
	const bool  debug_colors = std::getenv("KYTY_BINDLESS_DEBUG_COLORS") != nullptr;
	const char* opaque_env   = std::getenv("KYTY_BINDLESS_OPAQUE_PLACEHOLDER");
	const float opaque       = opaque_env != nullptr && opaque_env[0] == '1' ? 1.0f : 0.0f;
	const auto  color        = [debug_colors, opaque](float r, float g, float b) {
        return debug_colors ? vk::ClearColorValue {std::array<float, 4> {r, g, b, 1.0f}}
		                           : vk::ClearColorValue {std::array<float, 4> {0.0f, 0.0f, 0.0f, opaque}};
	};
	std::array<vk::ClearColorValue, PlaceholderColors> colors {
	    color(0.5f, 0.5f, 0.5f), color(0.8f, 0.1f, 0.1f), color(0.1f, 0.2f, 0.8f)};
	// KYTY_BINDLESS_PLACEHOLDER_RGBA=<r,g,b,a>[;<r,g,b,a>[;<r,g,b,a>]] (diagnostic, startup, default
	// off): the colours of placeholder slots 0 (not resident), 1 (key outside its heap) and 2
	// (pending), in that order; a slot left out keeps its colour above.
	if (const char* rgba = std::getenv("KYTY_BINDLESS_PLACEHOLDER_RGBA"); rgba != nullptr) {
		const char* p = rgba;
		for (uint32_t slot = 0; slot < PlaceholderColors && p != nullptr && *p != '\0'; slot++) {
			float v[4] {};
			if (std::sscanf(p, "%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3]) == 4) {
				colors[slot] = vk::ClearColorValue {std::array<float, 4> {v[0], v[1], v[2], v[3]}};
			}
			p = std::strchr(p, ';');
			if (p != nullptr) p++;
		}
	}
	for (uint32_t i = 0; i < Placeholders; i++) {
		sink.clearColorImage(m_placeholders[i].image, vk::ImageLayout::eTransferDstOptimal,
		                     &colors[i / ImageArrays], 1, &to_transfer[i].subresourceRange);
	}
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                     vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 0, nullptr,
	                     Placeholders, to_read.data());

	std::array<vk::DescriptorImageInfo, Placeholders> infos {};
	std::array<vk::WriteDescriptorSet, Placeholders>  writes {};
	for (uint32_t i = 0; i < Placeholders; i++) {
		infos[i]                  = {nullptr, m_placeholder_views[i],
		                             vk::ImageLayout::eShaderReadOnlyOptimal};
		writes[i].dstSet          = m_set;
		writes[i].dstBinding      = Images2D + i % ImageArrays;
		writes[i].dstArrayElement = i / ImageArrays;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[i].pImageInfo      = &infos[i];
	}
	m_graphics.device.updateDescriptorSets(Placeholders, writes.data(), 0, nullptr);
	return true;
}

bool BindlessTable::IsPlaceholder(vk::ImageView view) const noexcept {
	return std::ranges::find(m_placeholder_views, view) != m_placeholder_views.end();
}

void BindlessTable::WatchHeap(Heap& heap) {
	heap.watched_entries  = heap.guest_entries;
	heap.watch_generation = m_watch_generation.load(std::memory_order_acquire);
	// A record pointer heap watches its records (the pointers); RevalidateBindlessKeys follows
	// what they point at.
	const bool pointers = heap.record_pointer != UINT32_MAX;
	const auto table    = pointers ? heap.base : heap.base + heap.table_offset;
	const auto begin    = Common::AlignDown(table, TRACKER_PAGE_SIZE);
	const auto records  = heap.nested_keys != 0 ? heap.guest_entries / heap.nested_keys
	                                            : heap.guest_entries;
	const auto end      = Common::AlignUp(
        table + uint64_t {records} * heap.stride, TRACKER_PAGE_SIZE);
	if (end <= begin) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	auto watch = std::ranges::find_if(m_watches, [&](const Watch& w) { return w.begin == begin; });
	if (watch == m_watches.end()) {
		watch        = m_watches.insert(m_watches.end(), Watch {});
		watch->begin = begin;
		watch->end   = begin;
	}
	if (end <= watch->end) {
		return;
	}
	// The heap grew (or is new): watch the pages it now covers.
	const auto first = static_cast<size_t>((watch->end - begin) / TRACKER_PAGE_SIZE);
	const auto pages = static_cast<size_t>((end - begin) / TRACKER_PAGE_SIZE);
	watch->watched.resize(pages, 0u);
	watch->written.resize(pages, 0u);
	watch->end = end;
	for (size_t page = first; page < pages; page++) {
		watch->watched[page] = 1;
	}
	m_page_manager.UpdatePageWatchers<true>(begin + first * TRACKER_PAGE_SIZE,
	                                        (pages - first) * TRACKER_PAGE_SIZE);
	if (begin < m_watch_low.load(std::memory_order_relaxed)) {
		m_watch_low.store(begin, std::memory_order_release);
	}
	if (end > m_watch_high.load(std::memory_order_relaxed)) {
		m_watch_high.store(end, std::memory_order_release);
	}
}

void BindlessTable::OnCpuWrite(uint64_t vaddr, uint64_t size) noexcept {
	// Called for every CPU write fault: most are nowhere near a heap and must not take the lock.
	if (vaddr + size <= m_watch_low.load(std::memory_order_acquire) ||
	    vaddr >= m_watch_high.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		const auto begin = std::max(watch.begin, Common::AlignDown(vaddr, TRACKER_PAGE_SIZE));
		const auto end   = std::min(watch.end, vaddr + size);
		for (auto address = begin; address < end; address += TRACKER_PAGE_SIZE) {
			const auto page = static_cast<size_t>((address - watch.begin) / TRACKER_PAGE_SIZE);
			if (watch.watched[page] == 0) {
				continue;
			}
			// The faulting write must go through: stop watching the page until the next sync.
			watch.watched[page] = 0;
			watch.written[page] = 1;
			m_page_manager.UpdatePageWatchers<false>(address, TRACKER_PAGE_SIZE);
			m_heap_writes.store(true, std::memory_order_release);
		}
	}
}

void BindlessTable::UnwatchRange(uint64_t vaddr, uint64_t size) {
	if (vaddr + size <= m_watch_low.load(std::memory_order_acquire) ||
	    vaddr >= m_watch_high.load(std::memory_order_acquire)) {
		return;
	}
	std::scoped_lock lock {m_watch_mutex};
	for (auto& watch: m_watches) {
		if (vaddr + size <= watch.begin || vaddr >= watch.end) {
			continue;
		}
		// Part of the table is unmapped: drop the whole watch. WatchHeap builds it again the
		// next time a draw uses the heap, over what is mapped then.
		for (size_t page = 0; page < watch.watched.size(); page++) {
			if (watch.watched[page] != 0) {
				m_page_manager.UpdatePageWatchers<false>(watch.begin + page * TRACKER_PAGE_SIZE,
				                                         TRACKER_PAGE_SIZE);
			}
		}
		watch.watched.clear();
		watch.written.clear();
		watch.end = watch.begin;
		m_watch_generation.fetch_add(1, std::memory_order_acq_rel);
	}
}

void BindlessTable::TakeWrittenRanges(std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
	ranges.clear();
	std::scoped_lock lock {m_watch_mutex};
	m_heap_writes.store(false, std::memory_order_release);
	for (auto& watch: m_watches) {
		for (size_t page = 0; page < watch.written.size(); page++) {
			if (watch.written[page] == 0) {
				continue;
			}
			const auto address  = watch.begin + page * TRACKER_PAGE_SIZE;
			watch.written[page] = 0;
			// Watched again before the caller reads the entries: a later write is not lost.
			if (watch.watched[page] == 0) {
				watch.watched[page] = 1;
				m_page_manager.UpdatePageWatchers<true>(address, TRACKER_PAGE_SIZE);
			}
			if (!ranges.empty() && ranges.back().second == address) {
				ranges.back().second += TRACKER_PAGE_SIZE;
			} else {
				ranges.emplace_back(address, address + TRACKER_PAGE_SIZE);
			}
		}
	}
}

// KYTY_BINDLESS_HEAP_GROW=1 (default 0; on in the PPSA03671 preset; the idea of upstream
// 4db225008, which resizes image heap tables in place): a heap whose descriptor grows keeps its
// translation region when it is the last one allocated (it is extended where it is), and a heap
// above 64 Ki keys gets room to grow by half. Off, every growth moves the heap to a new region,
// and one above 64 Ki keys has no headroom: a heap that grows a little on most frames leaks its
// old region each time and fills the 1 Mi-entry translation buffer within a few frames, after
// which no new heap gets a region (its keys sample the placeholder).
static bool BindlessHeapGrowEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_HEAP_GROW");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_BINDLESS_WATCH_KEY=<key> (diagnostic): every translation change of that key in any heap,
// and every descriptor write to a slot it was given, is logged (at most 4000 lines).
static uint32_t BindlessWatchKey() {
	static const uint32_t key = [] {
		const char* value = std::getenv("KYTY_BINDLESS_WATCH_KEY");
		return value != nullptr && value[0] != '\0' ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0))
		                                            : UINT32_MAX;
	}();
	return key;
}
static std::vector<std::pair<uint32_t, uint32_t>> g_watched_slots; // (binding, slot), GPU thread
static bool BindlessWatchLogTake() {
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1, std::memory_order_relaxed) < 4000;
}

uint32_t BindlessTable::TranslationCapacity() {
	static const uint32_t entries = [] {
		uint32_t mi = 24;
		if (const char* value = std::getenv("KYTY_BINDLESS_TRANSLATION_MI"); value != nullptr && value[0] != '\0') {
			mi = static_cast<uint32_t>(std::clamp<unsigned long>(std::strtoul(value, nullptr, 10), 1ul, 32ul));
		}
		return mi << 20u;
	}();
	return entries;
}

bool BindlessTable::AllocateRegion(Heap& heap, uint32_t entries) {
	// Headroom so a growing heap rarely moves; a move re-writes the keys resolved so far.
	const bool grow = BindlessHeapGrowEnabled();
	// With the switch, room to double up to 64 Ki keys and by half above; without, room up to a
	// total of 64 Ki keys and none above.
	const auto capacity =
	    grow ? static_cast<uint32_t>(std::min<uint64_t>(
	               uint64_t {entries} + std::max<uint32_t>(std::min(entries, 1u << 16u), entries / 2u),
	               TranslationCapacity()))
	         : std::max(entries, std::min<uint32_t>(entries * 2u, 1u << 16u));
	const auto old_region  = heap.region;
	const auto old_entries = heap.entries;
	if (grow && old_entries != 0 && old_region + old_entries == m_next_region &&
	    capacity > old_entries && capacity - old_entries <= TranslationCapacity() - m_next_region) {
		// The last region: extend it in place. Keys below old_entries keep their entries and
		// feedback flags; frames in flight read only those.
		m_next_region = old_region + capacity;
		heap.entries  = capacity;
		m_resized_heaps.push_back(heap.serial);
		heap.slots.resize(capacity, 0u);
		heap.settled.resize(capacity, 0u);
		heap.descriptors.resize(capacity);
		heap.images.resize(capacity);
		auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
		auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
		for (uint32_t key = old_entries; key < capacity; key++) {
			translation[heap.region + key] = ShaderRecompiler::IR::BindlessPending;
			feedback[heap.region + key]    = 0;
		}
		const auto added = capacity - old_entries;
		m_translation->Flush((heap.region + old_entries) * sizeof(uint32_t), added * sizeof(uint32_t));
		m_feedback->Flush((heap.region + old_entries) * sizeof(uint32_t), added * sizeof(uint32_t));
		LOGF("Bindless heap region: base=0x%016" PRIx64 " binding=%u region=%u entries=%u (was %u, "
		     "extended in place)\n",
		     heap.base, heap.binding, heap.region, capacity, old_entries);
		return true;
	}
	if (capacity > TranslationCapacity() - m_next_region) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Bindless heap region: base=0x%016" PRIx64 " table_offset=%u binding=%u stride=%u needs %u "
			     "entries, the translation buffer is full (%u of %u used): its keys sample the placeholder\n",
			     heap.base, heap.table_offset, heap.binding, heap.stride, capacity, m_next_region,
			     TranslationCapacity());
		}
		return false;
	}
	heap.region            = m_next_region;
	heap.entries           = capacity;
	m_resized_heaps.push_back(heap.serial);
	m_next_region += capacity;
	heap.slots.resize(capacity, 0u);
	heap.settled.resize(capacity, 0u);
	heap.descriptors.resize(capacity);
	heap.images.resize(capacity);
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	auto* feedback    = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	for (uint32_t key = 0; key < capacity; key++) {
		uint32_t value = ShaderRecompiler::IR::BindlessPending;
		if (key < old_entries && heap.settled[key] != 0) {
			value = heap.slots[key];
		}
		translation[heap.region + key] = value;
		feedback[heap.region + key]    = 0;
	}
	m_translation->Flush(heap.region * sizeof(uint32_t), capacity * sizeof(uint32_t));
	m_feedback->Flush(heap.region * sizeof(uint32_t), capacity * sizeof(uint32_t));
	LOGF("Bindless heap region: base=0x%016" PRIx64 " binding=%u region=%u entries=%u (was %u at "
	     "%u)\n",
	     heap.base, heap.binding, heap.region, capacity, old_entries, old_region);
	return true;
}

uint32_t BindlessViewKeyMode() {
	static const uint32_t mode = [] {
		const char* value = std::getenv("KYTY_BINDLESS_VIEW_KEY");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	return mode;
}

uint64_t BindlessViewKey(const ShaderRecompiler::IR::ImageResource& resource) {
	uint64_t   hash = 0xcbf29ce484222325ull;
	const auto mix  = [&](uint64_t value) { hash = (hash ^ value) * 0x100000001b3ull; };
	mix(static_cast<uint64_t>(resource.numeric_class));
	mix(static_cast<uint64_t>(resource.dimension));
	mix(static_cast<uint64_t>(resource.mip_mode));
	mix(resource.mip_count);
	mix(static_cast<uint64_t>(resource.conversion_format));
	mix(resource.shader_swizzle);
	mix((resource.depth_compare ? 1u : 0u) | (resource.cube ? 2u : 0u) | (resource.r128 ? 4u : 0u));
	return hash == 0 ? 1 : hash;
}

BindlessTable::Heap* BindlessTable::FindOrCreateHeap(
    uint64_t base, uint32_t table_offset, uint32_t binding, uint32_t entries,
    const ShaderRecompiler::IR::ImageResource& resource, uint32_t stride, uint32_t record_pointer,
    uint32_t nested_vsharp, uint32_t nested_heap, uint32_t nested_keys) {
	if (entries == 0 || m_set == nullptr) {
		return nullptr;
	}
	static const auto no_heaps  = std::vector<Heap*> {};
	const auto        indexed   = m_heaps_by_base.find(base);
	const auto        view_mode = BindlessViewKeyMode();
	const uint64_t    view_key  = view_mode != 0 ? BindlessViewKey(resource) : 0;
	for (auto* candidate: indexed == m_heaps_by_base.end() ? no_heaps : indexed->second) {
		auto& heap = *candidate;
		if (heap.base == base && heap.table_offset == table_offset && heap.binding == binding &&
		    heap.stride == stride && heap.record_pointer == record_pointer &&
		    heap.nested_vsharp == nested_vsharp && heap.nested_heap == nested_heap &&
		    heap.nested_keys == nested_keys && (view_mode < 2 || heap.view_key == view_key)) {
			// KYTY_BINDLESS_VIEW_KEY=1: a user that views the heap differently than the user that
			// created it (whose resource every key is resolved with).
			if (view_mode == 1 && BindlessViewKey(heap.resource) != view_key) {
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 64u) {
					const auto& a = heap.resource;
					LOGF("Bindless view key: heap 0x%016" PRIx64 "+%u b%u entries %u viewed differently: "
					     "class %u/%u dim %u/%u mip %u:%u/%u:%u conv %u/%u swz %08x/%08x cmp %d/%d cube %d/%d "
					     "r128 %d/%d\n",
					     base, table_offset, binding, heap.entries, static_cast<uint32_t>(a.numeric_class),
					     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(a.dimension),
					     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(a.mip_mode), a.mip_count,
					     static_cast<uint32_t>(resource.mip_mode), resource.mip_count,
					     static_cast<uint32_t>(a.conversion_format), static_cast<uint32_t>(resource.conversion_format),
					     a.shader_swizzle, resource.shader_swizzle, a.depth_compare ? 1 : 0,
					     resource.depth_compare ? 1 : 0, a.cube ? 1 : 0, resource.cube ? 1 : 0, a.r128 ? 1 : 0,
					     resource.r128 ? 1 : 0);
				}
			}
			// A heap that cannot grow keeps its region (keys beyond it read as out of range).
			if (entries > heap.entries && !AllocateRegion(heap, entries) && heap.entries == 0) {
				return nullptr;
			}
			heap.guest_entries = std::max(heap.guest_entries, entries);
			return &heap;
		}
	}
	auto& heap         = m_heaps.emplace_back();
	heap.serial        = static_cast<uint32_t>(m_heaps.size() - 1u);
	heap.base          = base;
	heap.table_offset   = table_offset;
	heap.stride         = stride;
	heap.record_pointer = record_pointer;
	heap.nested_vsharp  = nested_vsharp;
	heap.nested_heap    = nested_heap;
	heap.nested_keys    = nested_keys;
	heap.binding        = binding;
	heap.resource      = resource;
	heap.view_key      = view_mode >= 2 ? view_key : 0;
	heap.guest_entries = entries;
	// KYTY_BINDLESS_SMALL_HEAPS=1 (default 0; PPSA26344 preset): a new heap reserves its guest
	// entries (at least 16) instead of at least 16 Ki. Ghost of Yotei samples T#s from hundreds
	// of small material record tables (KYTY_BINDLESS_STRIDE, 440-byte records) at changing
	// addresses; at 16 Ki keys (32 Ki with headroom) each, they filled the 24 Mi-entry translation
	// buffer before gameplay and every later heap sampled the placeholder. A heap that grows past
	// its region is extended or moved as before.
	static const bool small_heaps = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SMALL_HEAPS");
		return value != nullptr && value[0] == '1';
	}();
	if (!AllocateRegion(heap, std::max(entries, small_heaps ? 16u : 1u << 14u))) {
		m_heaps.pop_back();
		return nullptr;
	}
	m_heaps_by_base[base].push_back(&heap);
	return &heap;
}

uint32_t BindlessTable::AllocateSlot(uint32_t binding) {
	if (binding >= ImageArrays) {
		return 0;
	}
	// Keys are settled again whenever the guest rewrites a heap entry or a texture is evicted,
	// and each settle took a new slot: the array would fill up over a long session.
	if (auto& free = m_free_slots[binding];
	    !free.empty() && m_scheduler.IsFree(free.front().second)) {
		const auto slot = free.front().first;
		free.pop_front();
		m_slot_free[binding][slot] = 0;
		return slot;
	}
	// KYTY_BINDLESS_SLOT_LOG=1 (diagnostic, default off): every 4096 new slots, and on the first
	// 16 failures, the slot use per binding, the free lists, the heaps and the translation entries.
	static const bool slot_log = [] {
		const char* value = std::getenv("KYTY_BINDLESS_SLOT_LOG");
		return value != nullptr && value[0] == '1';
	}();
	static uint64_t allocations = 0;
	static uint32_t failures    = 0;
	const bool      full        = m_next_slot[binding] >= m_images_per_array;
	if (slot_log && ((!full && (++allocations % 4096u) == 0) || (full && failures++ < 16u))) {
		std::string line;
		for (uint32_t b = 0; b < ImageArrays; b++) {
			line += " b" + std::to_string(b) + "=" + std::to_string(m_next_slot[b]) + "/free" +
			        std::to_string(m_free_slots[b].size());
		}
		std::printf("Bindless slots%s: binding %u, per array %u:%s, heaps %zu, translation %u of %u\n",
		            full ? " FULL" : "", binding, m_images_per_array, line.c_str(), m_heaps.size(),
		            m_next_region, TranslationCapacity());
		std::fflush(stdout);
	}
	if (full) {
		return 0;
	}
	return m_next_slot[binding]++;
}

// KYTY_BINDLESS_SLOT_REUSE=0 (live, default 1): freed slots are not given to other keys.
static Live::Switch g_bindless_slot_reuse("KYTY_BINDLESS_SLOT_REUSE", [](const char* value) -> int64_t {
	return value == nullptr || value[0] != '0' ? 1 : 0;
});

void BindlessTable::FreeSlot(uint32_t binding, uint32_t slot) {
	// KYTY_BINDLESS_PREFILL: a shared slot stays while another key holds it; its last key frees it.
	if (!m_shared_slot_words.empty() || !m_shared_slot_refs.empty()) {
		const uint64_t shared_key = (uint64_t {binding} << 32u) | slot;
		if (const auto refs = m_shared_slot_refs.find(shared_key); refs != m_shared_slot_refs.end()) {
			if (--refs->second == 0u) {
				m_shared_slot_refs.erase(refs);
			}
			return;
		}
		if (const auto words = m_shared_slot_words.find(shared_key); words != m_shared_slot_words.end()) {
			m_shared_slots.erase(words->second);
			m_shared_slot_words.erase(words);
		}
	}
	if (g_bindless_slot_reuse.Get() == 0 || binding >= ImageArrays || slot < PlaceholderColors) {
		return;
	}
	auto& flags = m_slot_free[binding];
	if (flags.size() <= slot) {
		flags.resize(static_cast<size_t>(slot) + 1u, 0);
	}
	// A slot already waiting on the free list must not be queued twice: two keys would get it.
	if (flags[slot] != 0) {
		static std::atomic<uint32_t> reported {0};
		if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Bindless: slot %u/%u freed twice; kept once\n", binding, slot);
		}
		return;
	}
	flags[slot] = 1;
	// Command buffers recorded so far may still sample the slot.
	m_free_slots[binding].emplace_back(slot, m_scheduler.CurrentTick());
}

void BindlessTable::WriteSlot(uint32_t binding, uint32_t slot, vk::ImageView view,
                              vk::ImageLayout layout) {
	if (!g_watched_slots.empty() &&
	    std::ranges::find(g_watched_slots, std::pair {binding, slot}) != g_watched_slots.end()) [[unlikely]] {
		if (BindlessWatchLogTake()) {
			LOGF("Bindless watch: slot %u/%u <- %s view %p\n", binding, slot,
			     IsPlaceholder(view) ? "placeholder" : "image", static_cast<void*>(static_cast<VkImageView>(view)));
		}
	}
	const vk::DescriptorImageInfo info {nullptr, view, layout};
	vk::WriteDescriptorSet        write {};
	write.dstSet          = m_set;
	write.dstBinding      = binding;
	write.dstArrayElement = slot;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampledImage;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	if (binding >= ImageArrays) {
		return;
	}
	auto& views = m_slot_views[binding];
	if (views.size() <= slot) {
		views.resize(slot + 1u);
	}
	if (const auto old = views[slot]; old != nullptr) {
		if (const auto found = m_view_slots.find(static_cast<VkImageView>(old));
		    found != m_view_slots.end()) {
			std::erase(found->second, std::pair {binding, slot});
			if (found->second.empty()) {
				m_view_slots.erase(found);
			}
		}
	}
	const bool placeholder = IsPlaceholder(view);
	views[slot]            = placeholder ? vk::ImageView {} : view;
	if (!placeholder && view != nullptr) {
		m_view_slots[static_cast<VkImageView>(view)].emplace_back(binding, slot);
	}
}

uint32_t BindlessTable::ReleaseSlotKeys(uint32_t binding, uint32_t slot) {
	uint32_t keys = 0;
	for (auto& heap: m_heaps) {
		for (uint32_t key = 0; key < heap.slots.size(); key++) {
			// The slot may be in another binding than the heap's (KYTY_BINDLESS_3D_AS_ARRAY).
			const auto entry = heap.slots[key];
			if (entry == 0 || SlotBinding(heap, entry) != binding || SlotIndex(entry) != slot) {
				continue;
			}
			if (const auto id = heap.images[key]; id) {
				if (const auto refs = m_image_refs.find(ImageKey(id)); refs != m_image_refs.end()) {
					std::erase(refs->second, std::pair<Heap*, uint32_t> {&heap, key});
					if (refs->second.empty()) {
						m_image_refs.erase(refs);
					}
				}
				std::erase(heap.resolved, id);
				heap.checked_generation = 0;
			}
			heap.slots[key]   = 0;
			heap.settled[key] = 0;
			heap.images[key]  = {};
			SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
			keys++;
		}
	}
	return keys;
}

bool BindlessTable::ReleaseViews(std::span<const CachedImageView> views) {
	if (m_set == nullptr || m_view_slots.empty()) {
		return false;
	}
	bool held = false;
	for (const auto& cached: views) {
		const auto found = m_view_slots.find(static_cast<VkImageView>(cached.view));
		if (cached.view == nullptr || found == m_view_slots.end()) {
			continue;
		}
		const auto slots = found->second; // WriteSlot edits the map
		for (const auto& [binding, slot]: slots) {
			held            = true;
			const auto keys = ReleaseSlotKeys(binding, slot);
			WriteSlot(binding, slot, PlaceholderView(binding, 0),
			          vk::ImageLayout::eShaderReadOnlyOptimal);
			// Only a slot this took from keys goes to the free list: one no key uses is on it
			// already (ReleaseKey / OnImageUnregistered keep its view until it is reused).
			// KYTY_BINDLESS_PREFILL: every key of a shared slot was released here at once.
			ForgetSharedSlot(binding, slot, keys != 0);
			if (keys != 0) {
				FreeSlot(binding, slot);
			}
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 16) {
				LOGF("Bindless: binding %u slot %u still held a view of an image being dropped "
				     "(%u keys); repointed to the placeholder\n",
				     binding, slot, keys);
			}
		}
	}
	return held;
}

uint32_t BindlessTable::WatchKey() {
	return BindlessWatchKey();
}

void BindlessTable::DebugWatchReport(uint64_t frame) {
	const auto key = BindlessWatchKey();
	if (key == UINT32_MAX || m_translation == nullptr) return;
	static uint32_t calls = 0;
	if ((calls++ % 16u) != 0u) return;
	const auto* translation = reinterpret_cast<const uint32_t*>(m_translation->Mapped().data());
	std::string text;
	for (const auto& heap: m_heaps) {
		if (key >= heap.entries || heap.runtime) continue;
		const auto value = translation[heap.region + key];
		if (value == ShaderRecompiler::IR::BindlessPending) continue;
		std::string view = "-";
		if (value != 0) {
			const auto binding = SlotBinding(heap, value);
			const auto slot    = SlotIndex(value);
			const auto& views  = m_slot_views[binding < ImageArrays ? binding : 0];
			view = slot < views.size() ? (views[slot] ? "image" : "placeholder/none") : "unset";
			view += fmt::format(" heap.slots={:x}", heap.slots.size() > key ? heap.slots[key] : 0u);
		}
		text += fmt::format(" | 0x{:x}+{} b{} r{}: 0x{:x} {}", heap.base, heap.table_offset, heap.binding, heap.region,
		                    value, view);
	}
	if (BindlessWatchLogTake()) LOGF("Bindless watch report: frame=%" PRIu64 "%s\n", frame, text.c_str());
}

void BindlessTable::SetTranslation(const Heap& heap, uint32_t key, uint32_t slot) {
	if (key == BindlessWatchKey()) [[unlikely]] {
		if (BindlessWatchLogTake()) {
			LOGF("Bindless watch: heap 0x%" PRIx64 "+%u binding %u key %u translation %u -> 0x%x (slot %u/%u)\n",
			     heap.base, heap.table_offset, heap.binding, key,
			     reinterpret_cast<const uint32_t*>(m_translation->Mapped().data())[heap.region + key], slot,
			     slot == ShaderRecompiler::IR::BindlessPending ? 0u : SlotBinding(heap, slot),
			     slot == ShaderRecompiler::IR::BindlessPending ? 0u : SlotIndex(slot));
		}
		if (slot != ShaderRecompiler::IR::BindlessPending && slot != 0u) {
			const std::pair entry {SlotBinding(heap, slot), SlotIndex(slot)};
			if (std::ranges::find(g_watched_slots, entry) == g_watched_slots.end()) g_watched_slots.push_back(entry);
		}
	}
	auto* translation = reinterpret_cast<uint32_t*>(m_translation->Mapped().data());
	translation[heap.region + key] = slot;
	m_translation->Flush((heap.region + key) * sizeof(uint32_t), sizeof(uint32_t));
}

void BindlessTable::RecordFeedbackSnapshot() {
	if (m_set == nullptr || !m_scheduler.Active()) {
		return;
	}
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		return;
	}
	// Word 0 and the chunk summary words are copied every time; of the heaps' regions only the chunks
	// whose summary word was set in the snapshot consumed last (ConsumeSnapshot): a frame's flags are
	// a few chunks of a translation buffer of up to 16 Mi entries. Then the summary is cleared.
	using ShaderRecompiler::IR::BindlessFeedbackChunkShift;
	using ShaderRecompiler::IR::BindlessFeedbackFirstRegion;
	const vk::DeviceSize size = vk::DeviceSize {m_next_region} * sizeof(uint32_t);
	command.EndRendering();
	const auto              sink = command.HelperSink();
	auto&                   snapshot_buffer = *m_feedback_snapshots[m_snapshot_record];
	vk::BufferMemoryBarrier flags_written {};
	flags_written.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	flags_written.dstAccessMask       = vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;
	flags_written.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	flags_written.buffer              = m_feedback->Handle();
	flags_written.offset              = 0;
	flags_written.size                = size;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                     vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &flags_written, 0,
	                     nullptr);
	m_copy_regions.clear();
	m_copy_regions.push_back({0, 0, vk::DeviceSize {BindlessFeedbackFirstRegion} * sizeof(uint32_t)});
	// The flagged chunks are packed after the summary in the snapshot (chunk i at FirstRegion + i * 1024).
	if (m_copy_chunks.size() > MaxCopyChunks) {
		m_copy_chunks.resize(MaxCopyChunks); // the rest are flagged again by the next sample of their keys
	}
	for (size_t i = 0; i < m_copy_chunks.size(); i++) {
		const uint64_t begin = std::max<uint64_t>(uint64_t {m_copy_chunks[i]} << BindlessFeedbackChunkShift, BindlessFeedbackFirstRegion);
		const uint64_t end   = std::min<uint64_t>((uint64_t {m_copy_chunks[i]} + 1u) << BindlessFeedbackChunkShift, m_next_region);
		const uint64_t dst   = BindlessFeedbackFirstRegion + (uint64_t {i} << BindlessFeedbackChunkShift);
		if (begin < end) {
			m_copy_regions.push_back({begin * sizeof(uint32_t), dst * sizeof(uint32_t), (end - begin) * sizeof(uint32_t)});
		}
	}
	sink.copyBuffer(m_feedback->Handle(), snapshot_buffer.Handle(), static_cast<uint32_t>(m_copy_regions.size()),
	                m_copy_regions.data());
	m_copied_bytes += [&] {
		uint64_t bytes = 0;
		for (const auto& region: m_copy_regions) bytes += region.size;
		return bytes;
	}();
	// The summary is cleared after the copy read it (and before any later shader sets it again).
	vk::BufferMemoryBarrier copied = flags_written;
	copied.srcAccessMask           = vk::AccessFlagBits::eTransferRead;
	copied.dstAccessMask           = vk::AccessFlagBits::eTransferWrite;
	copied.size                    = vk::DeviceSize {BindlessFeedbackFirstRegion} * sizeof(uint32_t);
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 0,
	                     nullptr, 1, &copied, 0, nullptr);
	sink.fillBuffer(m_feedback->Handle(), sizeof(uint32_t),
	                vk::DeviceSize {BindlessFeedbackFirstRegion - 1u} * sizeof(uint32_t), 0u);
	vk::BufferMemoryBarrier cleared = copied;
	cleared.srcAccessMask           = vk::AccessFlagBits::eTransferWrite;
	cleared.dstAccessMask           = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands, {}, 0,
	                     nullptr, 1, &cleared, 0, nullptr);
	auto snapshot_written          = flags_written;
	snapshot_written.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	snapshot_written.dstAccessMask = vk::AccessFlagBits::eHostRead;
	snapshot_written.buffer        = snapshot_buffer.Handle();
	sink.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
	                     0, nullptr, 1, &snapshot_written, 0, nullptr);
	m_inflight_chunks.swap(m_copy_chunks);
	m_copy_chunks.clear();
	if (m_runtime_requests != nullptr) {
		// KYTY_RUNTIME_IMAGES: the request buffer, at the same point.
		auto requests_written   = flags_written;
		requests_written.buffer = m_runtime_requests->Handle();
		requests_written.size   = VK_WHOLE_SIZE;
		sink.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                     vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1,
		                     &requests_written, 0, nullptr);
		const vk::BufferCopy requests_copy {0, 0, m_runtime_requests->Size()};
		sink.copyBuffer(m_runtime_requests->Handle(), m_runtime_requests_snapshot->Handle(), 1,
		                &requests_copy);
		auto requests_snapshot          = snapshot_written;
		requests_snapshot.buffer        = m_runtime_requests_snapshot->Handle();
		requests_snapshot.size          = VK_WHOLE_SIZE;
		sink.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
		                     {}, 0, nullptr, 1, &requests_snapshot, 0, nullptr);
	}
	m_snapshot_tick = m_scheduler.CurrentTick();
}

bool BindlessTable::SnapshotReady() {
	return m_snapshot_tick != 0 && m_scheduler.IsFree(m_snapshot_tick);
}

void BindlessTable::ConsumeSnapshot() {
	using ShaderRecompiler::IR::BindlessFeedbackChunkShift;
	using ShaderRecompiler::IR::BindlessFeedbackFirstRegion;
	using ShaderRecompiler::IR::BindlessFeedbackSummary;
	constexpr uint32_t ChunkWords = 1u << BindlessFeedbackChunkShift;
	m_snapshot_tick = 0;
	m_snapshot_read = m_snapshot_record;
	m_snapshot_record ^= 1u;
	m_snapshot_serial++;
	auto&       snapshot = *m_feedback_snapshots[m_snapshot_read];
	const auto* words    = reinterpret_cast<const uint32_t*>(snapshot.Mapped().data());
	snapshot.Invalidate(0, snapshot.Size());
	// The chunks copied into it wait on the host until their heap takes its flags (a call table's
	// heaps are used in every other frame, and a snapshot is consumed more often).
	for (size_t i = 0; i < m_inflight_chunks.size(); i++) {
		auto& pending  = m_pending_chunks[m_inflight_chunks[i]];
		pending.serial = m_snapshot_serial;
		std::copy_n(words + BindlessFeedbackFirstRegion + i * ChunkWords, ChunkWords, pending.words.begin());
	}
	m_inflight_chunks.clear();
	std::erase_if(m_pending_chunks, [&](const auto& item) { return item.second.serial + 64u < m_snapshot_serial; });
	// The chunks flagged in this snapshot are copied with the next one.
	const uint32_t chunks = (m_next_region + ChunkWords - 1u) >> BindlessFeedbackChunkShift;
	m_copy_chunks.clear();
	for (uint32_t chunk = 0; chunk < chunks; chunk++) {
		if (words[BindlessFeedbackSummary + chunk] != 0u) {
			m_copy_chunks.push_back(chunk);
		}
	}
	m_flagged_chunks += m_copy_chunks.size();
}

void BindlessTable::TakeRequests(const Heap& heap, std::vector<uint32_t>& keys) {
	using ShaderRecompiler::IR::BindlessFeedbackChunkShift;
	if (heap.entries == 0 || m_pending_chunks.empty()) {
		return;
	}
	auto*          feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	const uint32_t first    = heap.region >> BindlessFeedbackChunkShift;
	const uint32_t last     = (heap.region + heap.entries - 1u) >> BindlessFeedbackChunkShift;
	for (uint32_t chunk = first; chunk <= last; chunk++) {
		const auto found = m_pending_chunks.find(chunk);
		if (found == m_pending_chunks.end()) {
			continue; // no flag was raised in it
		}
		auto&          words = found->second.words;
		const uint32_t base  = chunk << BindlessFeedbackChunkShift;
		const uint32_t begin = std::max(heap.region, base);
		const uint32_t end   = std::min(heap.region + heap.entries, base + (1u << BindlessFeedbackChunkShift));
		bool cleared = false;
		m_scanned_words += end - begin;
		for (uint32_t word = begin; word < end; word++) {
			const auto value = words[word - base];
			if (value != 0u) m_nonzero_words++;
			words[word - base] = 0;
			if (value == 1u) {
				// A flag the GPU sets again after the snapshot may be lost here; a key that is
				// still pending is flagged again by the next draw that samples it.
				feedback[word] = 0;
				keys.push_back(word - heap.region);
				cleared = true;
			}
		}
		if (cleared) {
			m_feedback->Flush(vk::DeviceSize {begin} * sizeof(uint32_t), vk::DeviceSize {end - begin} * sizeof(uint32_t));
		}
		if (std::ranges::all_of(words, [](uint32_t w) { return w == 0u; })) {
			m_pending_chunks.erase(found);
		}
	}
}
uint32_t BindlessTable::TakeWordZero() {
	auto&       snapshot_buffer = *m_feedback_snapshots[m_snapshot_read];
	const auto* snapshot        = reinterpret_cast<const uint32_t*>(snapshot_buffer.Mapped().data());
	auto*       feedback        = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	snapshot_buffer.Invalidate(0, sizeof(uint32_t));
	const auto value = snapshot[0];
	if (value != 0) {
		feedback[0] = 0;
		m_feedback->Flush(0, sizeof(uint32_t));
	}
	return value;
}

void BindlessTable::ArmUsageProbe() {
	if (m_feedback == nullptr) {
		return;
	}
	auto* feedback = reinterpret_cast<uint32_t*>(m_feedback->Mapped().data());
	for (const auto& heap: m_heaps) {
		if (heap.runtime) {
			continue;
		}
		uint32_t first = UINT32_MAX;
		uint32_t last  = 0;
		for (uint32_t key = 0; key < heap.entries && key < heap.images.size(); key++) {
			if (heap.images[key]) {
				feedback[heap.region + key] = ProbeMark;
				first                       = std::min(first, key);
				last                        = key;
			}
		}
		if (first != UINT32_MAX) {
			m_feedback->Flush((heap.region + first) * sizeof(uint32_t),
			                  (last - first + 1u) * sizeof(uint32_t));
		}
	}
}

void BindlessTable::CollectUnusedImages(std::vector<ImageId>& unused) {
	auto& snapshot_buffer = m_feedback_snapshots[m_snapshot_read];
	if (snapshot_buffer == nullptr) {
		return;
	}
	const auto* snapshot = reinterpret_cast<const uint32_t*>(snapshot_buffer->Mapped().data());
	snapshot_buffer->Invalidate(0, vk::DeviceSize {m_next_region} * sizeof(uint32_t));
	// A key resolved, cleared by TakeRequests, or in a moved region since the marks reads 0 or
	// 1: in use.
	for (const auto& [image, refs]: m_image_refs) {
		if (refs.empty()) {
			continue;
		}
		const bool sampled = std::any_of(refs.begin(), refs.end(), [&](const auto& ref) {
			return ref.first->runtime || snapshot[ref.first->region + ref.second] != ProbeMark;
		});
		if (!sampled) {
			unused.push_back(refs.front().first->images[refs.front().second]);
		}
	}
}

void BindlessTable::AddImageReference(ImageId id, Heap& heap, uint32_t key) {
	m_image_refs[ImageKey(id)].emplace_back(&heap, key);
}

const BindlessTable::SharedSlot* BindlessTable::FindSharedSlot(uint32_t binding,
                                                               const std::array<uint32_t, 8>& words,
                                                               uint64_t view_key) {
	const auto found = m_shared_slots.find(SharedSlotKey {binding, words, view_key});
	if (found == m_shared_slots.end()) {
		return nullptr;
	}
	// The slot must still show an image: a placeholder there (its views were dropped) would be
	// shared by every later heap that holds this T#.
	const auto& shared = found->second;
	const auto& views  = m_slot_views[shared.slot_binding];
	if (shared.slot_index >= views.size() || views[shared.slot_index] == nullptr) {
		m_shared_forgotten++;
		ForgetSharedSlot(shared.slot_binding, shared.slot_index, false);
		return nullptr;
	}
	return &shared;
}

void BindlessTable::ForgetSharedSlot(uint32_t binding, uint32_t slot, bool drop_refs) {
	const uint64_t slot_key = (uint64_t {binding} << 32u) | slot;
	if (const auto words = m_shared_slot_words.find(slot_key); words != m_shared_slot_words.end()) {
		m_shared_slots.erase(words->second);
		m_shared_slot_words.erase(words);
	}
	if (drop_refs) {
		m_shared_slot_refs.erase(slot_key);
	}
}

void BindlessTable::NoteSharedSlot(const Heap& heap, const std::array<uint32_t, 8>& words, uint32_t entry,
                                   ImageId image) {
	if (entry == 0u) {
		return;
	}
	const SharedSlotKey key {heap.binding, words, heap.view_key};
	const uint64_t      slot_key = (uint64_t {SlotBinding(heap, entry)} << 32u) | SlotIndex(entry);
	// A slot holds one T#; one T# names one slot.
	if (const auto old = m_shared_slot_words.find(slot_key); old != m_shared_slot_words.end()) {
		m_shared_slots.erase(old->second);
	}
	if (const auto old = m_shared_slots.find(key); old != m_shared_slots.end()) {
		m_shared_slot_words.erase((uint64_t {SlotBinding(heap, old->second.entry)} << 32u) |
		                          SlotIndex(old->second.entry));
	}
	m_shared_slots[key]           = SharedSlot {entry, image, SlotBinding(heap, entry), SlotIndex(entry)};
	m_shared_slot_words[slot_key] = key;
}

void BindlessTable::AddSharedSlotRef(uint32_t binding, uint32_t slot) {
	m_shared_slot_refs[(uint64_t {binding} << 32u) | slot]++;
}

bool BindlessTable::ReleaseKey(Heap& heap, uint32_t key, bool free_slot) {
	const auto id      = heap.images[key];
	bool       no_refs = false;
	if (id) {
		const auto found = m_image_refs.find(ImageKey(id));
		if (found != m_image_refs.end()) {
			std::erase(found->second, std::pair<Heap*, uint32_t> {&heap, key});
			const bool in_heap = std::any_of(found->second.begin(), found->second.end(),
			                                 [&](const auto& ref) { return ref.first == &heap; });
			if (!in_heap) {
				std::erase(heap.resolved, id);
				// Erasing shifts the images RenderExecutor::CommitBindings counted as checked.
				heap.checked_generation = 0;
			}
			if (found->second.empty()) {
				m_image_refs.erase(found);
				no_refs = true;
			}
		}
	}
	if (heap.slots[key] != 0 && free_slot) {
		FreeSlot(SlotBinding(heap, heap.slots[key]), SlotIndex(heap.slots[key]));
	}
	heap.slots[key]   = 0;
	heap.settled[key] = 0;
	heap.images[key]  = {};
	SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
	return no_refs;
}

bool BindlessTable::RemapSlotImage(uint32_t binding, uint32_t slot, ImageId old_id, ImageId new_id,
                                   vk::ImageView view, vk::ImageLayout layout) {
	WriteSlot(binding, slot, view, layout);
	if (old_id == new_id) {
		return false;
	}
	if (const auto words = m_shared_slot_words.find((uint64_t {binding} << 32u) | slot);
	    words != m_shared_slot_words.end()) {
		if (const auto shared = m_shared_slots.find(words->second); shared != m_shared_slots.end()) {
			shared->second.image = new_id;
		}
	}
	std::vector<std::pair<Heap*, uint32_t>> moved;
	bool                                    no_refs = false;
	if (const auto found = m_image_refs.find(ImageKey(old_id)); found != m_image_refs.end()) {
		auto& refs = found->second;
		std::erase_if(refs, [&](const std::pair<Heap*, uint32_t>& ref) {
			const auto entry = ref.first->slots[ref.second];
			if (entry == 0 || SlotBinding(*ref.first, entry) != binding || SlotIndex(entry) != slot) {
				return false;
			}
			moved.push_back(ref);
			return true;
		});
		for (const auto& [heap, key]: moved) {
			heap->images[key] = new_id;
			if (std::find(heap->resolved.begin(), heap->resolved.end(), new_id) == heap->resolved.end()) {
				heap->resolved.push_back(new_id);
			}
			heap->unchecked.push_back(new_id);
			if (std::none_of(refs.begin(), refs.end(), [&](const auto& ref) { return ref.first == heap; })) {
				std::erase(heap->resolved, old_id);
				heap->checked_generation = 0;
			}
		}
		if (refs.empty()) {
			m_image_refs.erase(found);
			no_refs = true;
		}
	}
	if (!moved.empty()) {
		auto& refs = m_image_refs[ImageKey(new_id)];
		refs.insert(refs.end(), moved.begin(), moved.end());
	}
	return no_refs;
}

// KYTY_BINDLESS_REPLACED_RESOLVE (default on; =0 off): see OnImageUnregistered in the header.
// Ghost of Yotei's leaf carpet samples a terrain height tile through a runtime T#; an expansion of
// that tile's image left the key on the placeholder for a few frames, and leaves spawned then kept
// the "no ground" height for good (a flat carpet).
static bool ReplacedResolveEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_BINDLESS_REPLACED_RESOLVE");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

void BindlessTable::OnImageUnregistered(ImageId id, bool replaced) {
	const auto found = m_image_refs.find(ImageKey(id));
	if (found == m_image_refs.end()) {
		return;
	}
	const bool keep = replaced && ReplacedResolveEnabled();
	for (const auto& [heap, key]: found->second) {
		if (keep) {
			m_replaced_keys.emplace_back(heap, key);
		}
		if (heap->slots[key] != 0) {
			// Placeholder 0 of the slot's own view type (the 2D-array binding for a
			// KYTY_BINDLESS_3D_AS_ARRAY entry).
			const auto binding = SlotBinding(*heap, heap->slots[key]);
			const auto slot    = SlotIndex(heap->slots[key]);
			WriteSlot(binding, slot, PlaceholderView(binding, 0),
			          vk::ImageLayout::eShaderReadOnlyOptimal);
			FreeSlot(binding, slot);
		}
		heap->slots[key]   = 0;
		heap->settled[key] = 0;
		heap->images[key]  = {};
		SetTranslation(*heap, key, ShaderRecompiler::IR::BindlessPending);
		std::erase(heap->resolved, id);
		heap->checked_generation = 0;
		m_unregistered_keys++;
		if (UnmapReresolve()) {
			std::scoped_lock lock {m_reresolve_mutex};
			if (m_reresolve.size() < (1u << 16u)) m_reresolve.emplace_back(heap, key);
		}
	}
	m_image_refs.erase(found);
}

// KYTY_BINDLESS_UNMAP_RERESOLVE=1 (live; default off since int22, on in the Miles Morales preset
// PPSA01417 where it was tested): see bindlessTable.h.
static Live::Switch g_unmap_reresolve("KYTY_BINDLESS_UNMAP_RERESOLVE", [](const char* value) -> int64_t {
	return value != nullptr && value[0] != '\0' && value[0] != '0' ? 1 : 0;
});

bool BindlessTable::UnmapReresolve() {
	return g_unmap_reresolve.Get() != 0;
}

void BindlessTable::TakeReresolve(const Heap* heap, std::vector<uint32_t>& keys) {
	keys.clear();
	std::scoped_lock lock {m_reresolve_mutex};
	std::erase_if(m_reresolve, [&](const auto& item) {
		if (item.first != heap) return false;
		keys.push_back(item.second);
		return true;
	});
}

bool BindlessTable::RuntimeImagesEnabled() {
	return ShaderRecompiler::GetCodegenOptions().runtime_images;
}

bool BindlessTable::SamplersEnabled() {
	return ShaderRecompiler::GetCodegenOptions().bindless_samplers;
}

uint32_t BindlessTable::SamplerSlots(const GraphicContext& graphics) {
	// A device without enough update-after-bind samplers still gets a one-entry array (the
	// stand-in), so every shader that indexes it sees the same layout.
	return std::max(1u, std::min(MaxSamplers, graphics.bindless_max_samplers));
}

BindlessTable::SamplerHeap& BindlessTable::FindOrCreateSamplerHeap(uint64_t base,
                                                                   uint32_t table_offset,
                                                                   uint32_t stride,
                                                                   uint32_t flags) {
	for (auto& heap: m_sampler_heaps) {
		if (heap.base == base && heap.table_offset == table_offset && heap.stride == stride &&
		    heap.flags == flags) {
			return heap;
		}
	}
	auto& heap        = m_sampler_heaps.emplace_back();
	heap.base         = base;
	heap.table_offset = table_offset;
	heap.stride       = stride;
	heap.flags        = flags;
	return heap;
}

void BindlessTable::WriteDefaultSampler(vk::Sampler sampler) {
	if (m_default_sampler_written || m_set == nullptr || m_sampler_slots == 0u) {
		return;
	}
	vk::DescriptorImageInfo info {sampler, nullptr, vk::ImageLayout::eUndefined};
	vk::WriteDescriptorSet  write {};
	write.dstSet          = m_set;
	write.dstBinding      = SamplersBinding;
	write.dstArrayElement = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eSampler;
	write.pImageInfo      = &info;
	m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	m_default_sampler_written = true;
}

bool BindlessTable::MirrorSamplerHeap(
    SamplerHeap& heap, std::span<const std::array<uint32_t, 4>> records,
    const std::function<vk::Sampler(const std::array<uint32_t, 4>&)>& make) {
	const bool same_prefix = records.size() >= heap.records.size() &&
	                         std::equal(heap.records.begin(), heap.records.end(), records.begin());
	if (same_prefix && records.size() == heap.records.size() && heap.region != 0u) {
		return true;
	}
	auto first = static_cast<uint32_t>(heap.records.size());
	if (!same_prefix || records.size() > heap.capacity || heap.region == 0u) {
		const auto capacity = std::max<uint32_t>(64u, static_cast<uint32_t>(records.size()) * 2u);
		if (m_sampler_slots <= 1u || capacity > m_sampler_slots - m_next_sampler_slot) {
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1) < 8) {
				LOGF("Bindless samplers: array full (%u of %u slots), heap 0x%016" PRIx64
				     " keeps %zu records\n",
				     m_next_sampler_slot, m_sampler_slots, heap.base, heap.records.size());
			}
			return false;
		}
		heap.region   = m_next_sampler_slot;
		heap.capacity = capacity;
		m_next_sampler_slot += capacity;
		first = 0;
	}
	std::vector<vk::DescriptorImageInfo> infos;
	infos.reserve(records.size() - first);
	for (size_t key = first; key < records.size(); key++) {
		infos.push_back({make(records[key]), nullptr, vk::ImageLayout::eUndefined});
	}
	if (!infos.empty()) {
		vk::WriteDescriptorSet write {};
		write.dstSet          = m_set;
		write.dstBinding      = SamplersBinding;
		write.dstArrayElement = heap.region + first;
		write.descriptorCount = static_cast<uint32_t>(infos.size());
		write.descriptorType  = vk::DescriptorType::eSampler;
		write.pImageInfo      = infos.data();
		m_graphics.device.updateDescriptorSets(1, &write, 0, nullptr);
	}
	static std::atomic<uint32_t> logged = 0;
	if (logged.fetch_add(1) < 32) {
		LOGF("Bindless samplers: heap 0x%016" PRIx64 "+0x%x stride=%u flags=%u region=%u "
		     "records=%zu (%zu new)\n",
		     heap.base, heap.table_offset, heap.stride, heap.flags, heap.region, records.size(),
		     infos.size());
	}
	heap.records.assign(records.begin(), records.end());
	return true;
}

BindlessTable::Heap* BindlessTable::RuntimeHeap(uint32_t binding,
                                                const ShaderRecompiler::IR::ImageResource* resource) {
	if (binding >= ImageArrays || m_runtime_table == nullptr || m_set == nullptr) {
		return nullptr;
	}
	if (m_runtime_heaps[binding] != nullptr || resource == nullptr) {
		return m_runtime_heaps[binding];
	}
	auto& heap          = m_heaps.emplace_back();
	heap.serial         = static_cast<uint32_t>(m_heaps.size() - 1u);
	heap.base           = 0;
	heap.table_offset   = 0xfffffff0u + binding; // no guest heap matches it
	heap.binding        = binding;
	heap.resource       = *resource;
	heap.runtime        = true;
	heap.guest_entries  = RuntimeKeys;
	if (!AllocateRegion(heap, RuntimeKeys)) {
		m_heaps.pop_back();
		return nullptr;
	}
	heap.runtime_dwords.resize(heap.entries);
	m_runtime_heaps[binding] = &heap;
	LOGF("Bindless runtime images: binding %u heap at region %u (%u keys)\n", binding, heap.region,
	     heap.entries);
	return &heap;
}

void BindlessTable::TakeRuntimeRequests(
    std::vector<std::pair<std::array<uint32_t, 8>, uint32_t>>& requests) {
	using namespace ShaderRecompiler::IR;
	if (m_runtime_requests == nullptr) {
		return;
	}
	const auto* snapshot =
	    reinterpret_cast<const uint32_t*>(m_runtime_requests_snapshot->Mapped().data());
	auto* live = reinterpret_cast<uint32_t*>(m_runtime_requests->Mapped().data());
	m_runtime_requests_snapshot->Invalidate(0, m_runtime_requests_snapshot->Size());
	if (snapshot[0] == 0u) {
		return;
	}
	const auto count = std::min(snapshot[0], RuntimeImageRequests);
	for (uint32_t index = 0; index < count; index++) {
		const auto* record  = snapshot + RuntimeImageRingOffset + index * RuntimeImageRequestWords;
		const auto  binding = record[8];
		if (binding == 0u || binding > ImageArrays) {
			continue; // not written yet when the snapshot was taken
		}
		std::array<uint32_t, 8> dwords {};
		std::copy_n(record, 8, dwords.begin());
		requests.emplace_back(dwords, binding - 1u);
	}
	// Cleared for the next requests. A request the GPU makes meanwhile may be lost; the T# is
	// still missing from the table, so the shader asks again.
	live[0] = 0;
	std::fill_n(live + RuntimeImageBitsOffset, RuntimeImageEntries / 32u, 0u);
	for (uint32_t index = 0; index < count; index++) {
		live[RuntimeImageRingOffset + index * RuntimeImageRequestWords + 8u] = 0;
	}
	m_runtime_requests->Flush(0, m_runtime_requests->Size());
}

void BindlessTable::PublishRuntimeEntry(const Heap& heap, uint32_t key,
                                        const std::array<uint32_t, 8>& dwords) {
	using namespace ShaderRecompiler::IR;
	auto*          table  = reinterpret_cast<uint32_t*>(m_runtime_table->Mapped().data());
	const uint32_t target = heap.region + key + 1u;
	const auto     home   = RuntimeImageHash(dwords.data(), heap.binding);
	uint32_t       chosen = UINT32_MAX;
	for (uint32_t probe = 0; probe < RuntimeImageProbes; probe++) {
		const auto  entry = (home + probe) & (RuntimeImageEntries - 1u);
		const auto* words = table + entry * RuntimeImageEntryWords;
		if (words[8] == target && words[9] == heap.binding &&
		    std::equal(dwords.begin(), dwords.end(), words)) {
			return;
		}
		if (words[8] == 0u && chosen == UINT32_MAX) {
			chosen = entry;
		}
	}
	if (chosen == UINT32_MAX) {
		// Every probe holds another T#: the last is evicted, and asks again when sampled.
		chosen = (home + RuntimeImageProbes - 1u) & (RuntimeImageEntries - 1u);
		static std::atomic<uint32_t> evicted = 0;
		if (evicted.fetch_add(1) < 16) {
			LOGF("Bindless runtime images: table entry %u evicted\n", chosen);
		}
	}
	auto* words = table + chosen * RuntimeImageEntryWords;
	words[8]    = 0; // a shader reading the entry meanwhile misses it
	std::copy(dwords.begin(), dwords.end(), words);
	words[9] = heap.binding;
	words[8] = target;
	m_runtime_table->Flush(vk::DeviceSize {chosen} * RuntimeImageEntryWords * sizeof(uint32_t),
	                       RuntimeImageEntryWords * sizeof(uint32_t));
}

uint32_t BindlessTable::RuntimeKey(Heap& heap, const std::array<uint32_t, 8>& dwords) {
	const auto map_key = std::pair {heap.binding, dwords};
	if (const auto found = m_runtime_keys.find(map_key); found != m_runtime_keys.end()) {
		PublishRuntimeEntry(heap, found->second, dwords); // evicted from the table meanwhile
		return found->second;
	}
	if (heap.runtime_next_key >= heap.entries) {
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 4) {
			LOGF("Bindless runtime images: binding %u holds %u T#s; further ones sample the "
			     "placeholder\n",
			     heap.binding, heap.entries);
		}
		return UINT32_MAX;
	}
	const auto key = heap.runtime_next_key++;
	m_runtime_keys.emplace(map_key, key);
	heap.runtime_dwords[key] = dwords;
	SetTranslation(heap, key, ShaderRecompiler::IR::BindlessPending);
	PublishRuntimeEntry(heap, key, dwords);
	return key;
}

} // namespace Libs::Graphics
