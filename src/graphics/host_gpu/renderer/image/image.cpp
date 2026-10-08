#include "graphics/host_gpu/renderer/image/image.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// KYTY_IMAGE_TRANSIT_SKIP / _VERIFY (Image::Transit).
struct TransitSkipConfig {
	bool enabled = true;
	int  verify  = 0; // 1: count and log mismatches, 2: exit on one
};
const TransitSkipConfig& TransitSkip() {
	static const TransitSkipConfig config = [] {
		TransitSkipConfig result;
		const auto* value = std::getenv("KYTY_IMAGE_TRANSIT_SKIP");
		result.enabled    = value == nullptr || std::strcmp(value, "0") != 0;
		const auto* verify = std::getenv("KYTY_IMAGE_TRANSIT_SKIP_VERIFY");
		if (verify != nullptr && *verify != '\0' && std::strcmp(verify, "0") != 0) {
			result.verify = std::strcmp(verify, "exit") == 0 ? 2 : 1;
		}
		return result;
	}();
	return config;
}

// KYTY_GUEST_STORAGE_REPEAT (Image::GuestTransitScope): default off.
bool GuestStorageRepeatEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GUEST_STORAGE_REPEAT");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}
thread_local bool g_guest_transit = false;

constexpr vk::AccessFlags2 TransitWriteAccess = vk::AccessFlagBits2::eTransferWrite |
                                                vk::AccessFlagBits2::eShaderWrite |
                                                vk::AccessFlagBits2::eMemoryWrite;

// Image::RecordedTransitions.
std::atomic<uint64_t> g_recorded_transitions {0};

// KYTY_COPY_VIA_BUFFER_BATCH=0: Image::CopyImageWithBuffer copies one region per barrier pair.
bool CopyViaBufferBatchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_COPY_VIA_BUFFER_BATCH");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

[[nodiscard]] vk::ImageType HostImageType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1D: return vk::ImageType::e1D;
		case Prospero::ImageType::kColor3D: return vk::ImageType::e3D;
		case Prospero::ImageType::kColor2D: return vk::ImageType::e2D;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(type));
	}
}

[[nodiscard]] vk::ImageCreateFlags ImageCreateFlags(const GraphicContext& graphics,
                                                   const ImageInfo& info) {
	vk::ImageCreateFlags flags {};
	if (DepthAspectTransferFormat(info.pixel_format) == vk::Format::eUndefined) {
		flags |= vk::ImageCreateFlagBits::eMutableFormat;
		flags |= vk::ImageCreateFlagBits::eExtendedUsage;
		if (info.IsBlock() && graphics.supports_block_texel_view) {
			flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
		}
	}
	if (info.IsVolume()) {
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	if (!info.IsVolume() || !info.IsBlock()) {
		return flags;
	}
	// RADV refuses a BC7 sRGB volume with block texel views (RX 9070 XT). Optional flags are
	// dropped, least needed first, until the device accepts the sampled image: uncompressed views
	// (only for guest writes, which need storage usage as well and are not used for volumes), 2D
	// views (IsValidViewType then allows 3D views only), then other formats' views. A device that
	// accepts the full set (NVIDIA, AMD's Windows driver) keeps it.
	using Bit = vk::ImageCreateFlagBits;
	const vk::ImageCreateFlags drops[] = {
	    {},
	    Bit::eBlockTexelViewCompatible,
	    Bit::eBlockTexelViewCompatible | Bit::e2DArrayCompatible,
	    Bit::e2DArrayCompatible | Bit::eBlockTexelViewCompatible | Bit::eExtendedUsage,
	    Bit::e2DArrayCompatible | Bit::eBlockTexelViewCompatible | Bit::eExtendedUsage |
	        Bit::eMutableFormat,
	};
	for (const auto drop: drops) {
		const auto candidate = flags & ~drop;
		if (graphics.GetImageFormatProperties(
		        info.pixel_format, vk::ImageType::e3D, vk::ImageTiling::eOptimal,
		        vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst |
		            vk::ImageUsageFlagBits::eSampled,
		        candidate, nullptr) == vk::Result::eSuccess) {
			if (drop) {
				static std::atomic_flag logged = ATOMIC_FLAG_INIT;
				if (!logged.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Vulkan image: block-compressed volume {} created without flags 0x{:x} "
					    "(the device does not support them)\n",
					    vk::to_string(info.pixel_format),
					    static_cast<vk::ImageCreateFlags::MaskType>(flags & drop)));
				}
			}
			return candidate;
		}
	}
	return flags;
}

[[nodiscard]] bool HasFormatFeature(vk::FormatProperties      properties,
                                    vk::FormatFeatureFlagBits feature) {
	return static_cast<bool>(properties.optimalTilingFeatures & feature);
}

[[nodiscard]] vk::ImageUsageFlags ImageUsageFlags(GraphicContext& graphics, const ImageInfo& info) {
	if (info.IsBlock()) {
		vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
		                            vk::ImageUsageFlagBits::eTransferDst |
		                            vk::ImageUsageFlagBits::eSampled;
		// Storage through uncompressed block-texel views (the block format itself has no storage
		// support): only when the device accepts that usage for this format and these flags.
		if (graphics.supports_block_texel_view && info.samples == 1 && !info.IsVolume() &&
		    ImageOps::BlockStorageUploadsEnabled()) {
			const auto          storage = usage | vk::ImageUsageFlagBits::eStorage;
			vk::ImageFormatProperties properties {};
			if (graphics.GetImageFormatProperties(info.pixel_format, HostImageType(info.type),
			                                      vk::ImageTiling::eOptimal, storage,
			                                      ImageCreateFlags(graphics, info),
			                                      &properties) == vk::Result::eSuccess) {
				usage = storage;
			}
		}
		return usage;
	}
	const auto properties = graphics.GetFormatProperties(info.pixel_format);
	auto       usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eSampledImage)) {
		usage |= vk::ImageUsageFlagBits::eSampled;
	}
	if (DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined) {
		usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
		if (graphics.attachment_feedback_loop_enabled && (usage & vk::ImageUsageFlagBits::eSampled)) {
			usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
		}
		return usage;
	}
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eColorAttachment)) {
		usage |= vk::ImageUsageFlagBits::eColorAttachment;
	}
	if (info.samples == 1) {
		usage |= vk::ImageUsageFlagBits::eStorage;
	}
	return usage;
}

// Once per format and set of drops (Image::Image).
void LogImageCreateFallback(vk::Format format, vk::ImageUsageFlags dropped_usage,
                            vk::ImageCreateFlags dropped_flags) {
	static std::mutex mutex;
	static std::set<std::tuple<vk::Format, vk::ImageUsageFlags::MaskType,
	                           vk::ImageCreateFlags::MaskType>>
	                 logged;
	std::scoped_lock lock(mutex);
	if (!logged.emplace(format, static_cast<vk::ImageUsageFlags::MaskType>(dropped_usage),
	                    static_cast<vk::ImageCreateFlags::MaskType>(dropped_flags))
	         .second) {
		return;
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Vulkan image: {} images are created without usage {} and flags {} (the device does not "
	    "support them for this format)\n",
	    vk::to_string(format), vk::to_string(dropped_usage), vk::to_string(dropped_flags)));
}

void ValidateOptionalRange(GuestRange range, const char* name) {
	if (!range.ValidOrEmpty()) {
		EXIT("invalid %s image range: address=0x%016llx size=0x%016llx\n", name,
		     static_cast<unsigned long long>(range.address),
		     static_cast<unsigned long long>(range.size));
	}
}

std::atomic<uint64_t> g_content_serial {0};

} // namespace

void Image::NoteContentWrite() noexcept {
	m_definite_writes++;
	NotePossibleWrite();
}

uint64_t Image::NextContentSerial() noexcept {
	return g_content_serial.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint64_t Image::RecordedTransitions() noexcept {
	return g_recorded_transitions.load(std::memory_order_relaxed);
}

void Image::NotePossibleWrite() noexcept {
	m_content_serial = NextContentSerial();
	// Every native write (copy, clear, draw/dispatch binding, upload) ends the guarantee that
	// clean chunks match guest memory; a guest-sourced upload sets it again afterwards.
	m_partial_valid = false;
}

vk::ImageAspectFlags Image::FullAspectMask(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

Image::GuestTransitScope::GuestTransitScope(): m_previous(g_guest_transit) {
	g_guest_transit = GuestStorageRepeatEnabled();
}

Image::GuestTransitScope::~GuestTransitScope() {
	g_guest_transit = m_previous;
}

Image::Barriers Image::GetBarriers(vk::ImageLayout                      destination_layout,
                                   vk::AccessFlags2                     destination_access,
                                   vk::PipelineStageFlags2              destination_stage,
                                   std::optional<ImageSubresourceRange> range) {
	auto& state              = backing.state;
	auto& subresource_states = backing.subresource_states;
	if (range && info.IsVolume()) {
		range->base_layer  = 0;
		range->layer_count = 1;
	}

	const bool partial =
	    range && (range->base_level != 0 || range->level_count != info.resources.levels ||
	              range->base_layer != 0 || range->layer_count != info.resources.layers);
	const bool has_subresource_states = !subresource_states.empty();
	const bool guest                  = g_guest_transit;

	Barriers barriers;
	if (partial || has_subresource_states) {
		if (!has_subresource_states) {
			subresource_states.resize(info.resources.levels * info.resources.layers, state);
		}

		const uint32_t base_level  = partial ? range->base_level : 0;
		const uint32_t level_count = partial ? range->level_count : info.resources.levels;
		const uint32_t base_layer  = partial ? range->base_layer : 0;
		const uint32_t layer_count = partial ? range->layer_count : info.resources.layers;
		for (uint32_t level = base_level; level < base_level + level_count; level++) {
			for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
				const auto index = level * info.resources.layers + layer;
				EXIT_IF(index >= subresource_states.size());
				auto& subresource_state = subresource_states[index];

				constexpr auto write_access = vk::AccessFlagBits2::eTransferWrite |
				                              vk::AccessFlagBits2::eShaderWrite |
				                              vk::AccessFlagBits2::eMemoryWrite;
				const bool     repeated_write =
				    static_cast<bool>(subresource_state.access_mask & write_access) &&
				    !(guest && subresource_state.guest);
				if (subresource_state.layout != destination_layout ||
				    subresource_state.access_mask != destination_access || repeated_write) {
					vk::ImageMemoryBarrier2 barrier {};
					barrier.srcStageMask                    = subresource_state.pl_stage;
					barrier.srcAccessMask                   = subresource_state.access_mask;
					barrier.dstStageMask                    = destination_stage;
					barrier.dstAccessMask                   = destination_access;
					barrier.oldLayout                       = subresource_state.layout;
					barrier.newLayout                       = destination_layout;
					barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.image                           = backing.image;
					barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
					barrier.subresourceRange.baseMipLevel   = level;
					barrier.subresourceRange.levelCount     = 1;
					barrier.subresourceRange.baseArrayLayer = layer;
					barrier.subresourceRange.layerCount     = 1;
					barriers.push_back(barrier);
					subresource_state = {destination_stage, destination_access, destination_layout,
					                     guest};
				}
			}
		}

		if (!partial) {
			subresource_states.clear();
		}
	} else {
		constexpr auto write_access   = vk::AccessFlagBits2::eTransferWrite |
		                                vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eMemoryWrite;
		const bool     repeated_write = static_cast<bool>(state.access_mask & write_access);
		if (state.layout == destination_layout && state.access_mask == destination_access &&
		    (!repeated_write || (guest && state.guest))) {
			return {};
		}

		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask                    = state.pl_stage;
		barrier.srcAccessMask                   = state.access_mask;
		barrier.dstStageMask                    = destination_stage;
		barrier.dstAccessMask                   = destination_access;
		barrier.oldLayout                       = state.layout;
		barrier.newLayout                       = destination_layout;
		barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                           = backing.image;
		barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
		barriers.push_back(barrier);
	}

	state = {destination_stage, destination_access, destination_layout, guest};
	return barriers;
}

bool Image::TransitIsNoOp(vk::ImageLayout                             destination_layout,
                          vk::AccessFlags2                            destination_access,
                          const std::optional<ImageSubresourceRange>& range) const noexcept {
	// GetBarriers' conditions, in its order: per-subresource states or a partial range take its
	// per-subresource path (which may create those states); otherwise it returns before changing
	// anything exactly when the layout and access match and the access includes no write.
	if (!backing.subresource_states.empty()) {
		return false;
	}
	if (range) {
		const bool     volume      = info.IsVolume();
		const uint32_t base_layer  = volume ? 0u : range->base_layer;
		const uint32_t layer_count = volume ? 1u : range->layer_count;
		if (range->base_level != 0 || range->level_count != info.resources.levels ||
		    base_layer != 0 || layer_count != info.resources.layers) {
			return false;
		}
	}
	const auto& state = backing.state;
	return state.layout == destination_layout && state.access_mask == destination_access &&
	       !static_cast<bool>(state.access_mask & TransitWriteAccess);
}

void Image::Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
                    std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer,
                    bool deferrable) {
	// KYTY_IMAGE_TRANSIT_SKIP: GetBarriers would record nothing and change nothing.
	const auto& skip       = TransitSkip();
	const bool  predicted  = skip.enabled && TransitIsNoOp(destination_layout, destination_access,
	                                                      range);
	if (predicted && skip.verify == 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageTransitSkips);
		return;
	}
	KYTY_GPU_OP_SITE("image.transition");
	const auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::PipelineStageFlags2 destination_stage {};
	if (static_cast<bool>(destination_access & transfer_access)) {
		destination_stage |= vk::PipelineStageFlagBits2::eTransfer;
	}
	if (!destination_access ||
	    static_cast<bool>(destination_access & ~vk::AccessFlags2 {transfer_access})) {
		destination_stage |=
		    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	}
	const auto barriers =
	    GetBarriers(destination_layout, destination_access, destination_stage, range);
	if (predicted) {
		// KYTY_IMAGE_TRANSIT_SKIP_VERIFY: the skip's decision against GetBarriers (whose barriers,
		// if any, are recorded below as without the skip).
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageTransitVerifyChecks);
		if (!barriers.empty()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::ImageTransitVerifyMismatches);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr,
				             "ImageTransitVerify: a skipped transition of image 0x%016llx needs %zu "
				             "barrier(s) (layout %d -> %d, access 0x%llx -> 0x%llx)\n",
				             static_cast<unsigned long long>(info.data.address), barriers.size(),
				             static_cast<int>(barriers.front().oldLayout),
				             static_cast<int>(destination_layout),
				             static_cast<unsigned long long>(
				                 static_cast<VkAccessFlags2>(barriers.front().srcAccessMask)),
				             static_cast<unsigned long long>(
				                 static_cast<VkAccessFlags2>(destination_access)));
			}
			if (skip.verify == 2) {
				EXIT("ImageTransitVerify: a transition the skip decided against needs a barrier\n");
			}
		}
	}
	if (barriers.empty()) {
		return;
	}
	g_recorded_transitions.fetch_add(1, std::memory_order_relaxed);
	// Barrier batcher (render.h): merged with pending requests, recorded now or (deferrable) at
	// the next flush point, ending an active rendering instance only when recorded.
	if (m_scheduler.Active() && m_scheduler.Current().BatchImageBarriers(barriers, command_buffer,
	                                                                     deferrable)) {
		return;
	}
	m_scheduler.EndRendering();
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command_buffer.pipelineBarrier2(dependency);
}

void Image::Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
                   uint64_t size) {
	KYTY_GPU_OP_SITE("image.upload");
	NoteContentWrite();
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyBufferToImage(buffer, backing.image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                     uint64_t offset, uint64_t size) {
	KYTY_GPU_OP_SITE("image.download");
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
}

std::pair<uint32_t, uint32_t> Image::SanitizeCopyLayers(const Image& source,
                                                        const Image& destination, uint32_t depth) {
	const auto source_type        = source.backing.image_type;
	const auto destination_type   = destination.backing.image_type;
	uint32_t   source_layers      = source.backing.layers;
	uint32_t   destination_layers = destination.backing.layers;
	if (source_type == vk::ImageType::e3D) {
		source_layers = 1;
	}
	if (destination_type == vk::ImageType::e3D) {
		destination_layers = 1;
	}
	if (source_type == destination_type) {
		source_layers = destination_layers = std::min(source_layers, destination_layers);
	} else if (source_type == vk::ImageType::e2D && destination_type == vk::ImageType::e3D) {
		source_layers = depth;
	} else if (source_type == vk::ImageType::e3D && destination_type == vk::ImageType::e2D) {
		destination_layers = depth;
	}
	return {source_layers, destination_layers};
}

void Image::CopyImage(Image& source) {
	KYTY_GPU_OP_SITE("image.copy");
	CopyImageRegions(source);
}

void Image::CopyDepthColorImage(Image& source) {
	// Same regions as CopyImage; the depth side is copied through its depth aspect, which needs
	// VK_KHR_maintenance8 (TextureCache checks the format pair). Separate profiler site.
	KYTY_GPU_OP_SITE("image.copy_depth_color");
	EXIT_IF(!m_graphics.maintenance8_enabled);
	CopyImageRegions(source);
}

void Image::CopyImageRegions(Image& source) {
	NoteContentWrite();
	EXIT_IF(source.backing.samples != backing.samples);
	m_scheduler.EndRendering();
	const uint32_t levels     = std::min(source.backing.mip_levels, backing.mip_levels);
	const uint32_t base_depth = source.backing.image_type == backing.image_type
	                                ? std::min(source.backing.extent.depth, backing.extent.depth)
	                            : backing.image_type == vk::ImageType::e3D
	                                ? backing.extent.depth
	                                : source.backing.extent.depth;
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	std::vector<vk::ImageCopy> copies;
	copies.reserve(levels);
	for (uint32_t level = 0; level < levels; level++) {
		const auto width  = std::max(source.backing.extent.width >> level, 1u);
		const auto height = std::max(source.backing.extent.height >> level, 1u);
		const auto depth  = std::max(base_depth >> level, 1u);
		const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
		vk::ImageCopy copy {};
		copy.srcSubresource = {source_aspect, level, 0, 1};
		copy.dstSubresource = {destination_aspect, level, 0, 1};
		if (source.backing.image_type == backing.image_type) {
			if (source.backing.image_type == vk::ImageType::e3D) {
				copy.extent = {width, height, depth};
			} else {
				copy.srcSubresource.layerCount = std::min(source_layers, destination_layers);
				copy.dstSubresource.layerCount = copy.srcSubresource.layerCount;
				copy.extent                    = {width, height, 1};
			}
		} else if (source.backing.image_type == vk::ImageType::e2D) {
			copy.srcSubresource.layerCount = source_layers;
			copy.extent                    = {width, height, source_layers};
		} else {
			copy.dstSubresource.layerCount = destination_layers;
			copy.extent                    = {width, height, destination_layers};
		}
		copies.push_back(copy);
	}
	if (copies.empty()) {
		return;
	}
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, static_cast<uint32_t>(copies.size()),
	                  copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Resolve(Image& source, const ImageSubresourceRange& source_range,
                    const ImageSubresourceRange& destination_range) {
	KYTY_GPU_OP_SITE("image.resolve");
	NoteContentWrite();
	EXIT_IF(backing.samples != 1 || source.backing.image_type != vk::ImageType::e2D ||
	        backing.image_type != vk::ImageType::e2D || source_range.level_count != 1 ||
	        destination_range.level_count != 1 ||
	        source_range.base_level >= source.backing.mip_levels ||
	        destination_range.base_level >= backing.mip_levels ||
	        source_range.base_layer >= source.backing.layers ||
	        destination_range.base_layer >= backing.layers);
	const auto layers       = std::min({source_range.layer_count, destination_range.layer_count,
	                                    source.backing.layers - source_range.base_layer,
	                                    backing.layers - destination_range.base_layer});
	const auto source_width = std::max(source.backing.extent.width >> source_range.base_level, 1u);
	const auto source_height =
	    std::max(source.backing.extent.height >> source_range.base_level, 1u);
	const auto destination_width =
	    std::max(backing.extent.width >> destination_range.base_level, 1u);
	const auto destination_height =
	    std::max(backing.extent.height >> destination_range.base_level, 1u);
	const bool copy = source.backing.samples == 1;
	EXIT_IF(layers == 0 || info.extent.width > source_width || info.extent.height > source_height ||
	        info.extent.width > destination_width || info.extent.height > destination_height ||
	        (copy ? !ImageViewOps::FormatsCompatible(source.backing.format, backing.format)
	              : source.backing.format != backing.format));
	auto resolved_source_range             = source_range;
	auto resolved_destination_range        = destination_range;
	resolved_source_range.layer_count      = layers;
	resolved_destination_range.layer_count = layers;
	const vk::Extent3D resolve_extent {info.extent.width, info.extent.height, 1};

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               resolved_source_range, command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	        resolved_destination_range, command);
	if (copy) {
		vk::ImageCopy region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, region);
	} else {
		vk::ImageResolve region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.resolveImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                     backing.image, vk::ImageLayout::eTransferDstOptimal, region);
	}
}

uint32_t Image::CopyRows(uint64_t row_size, uint32_t rows, uint64_t capacity) noexcept {
	if (row_size == 0 || rows == 0 || row_size > capacity) {
		return 0;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(rows, capacity / row_size));
}

void Image::CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler) {
	KYTY_GPU_OP_SITE("image.copy_via_buffer");
	NoteContentWrite();
	EXIT_IF(buffer.Handle() == nullptr || source.backing.samples != 1 || backing.samples != 1);
	m_scheduler.EndRendering();
	const uint32_t levels = std::min(source.backing.mip_levels, backing.mip_levels);
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto     source_bytes      = DepthAspectTransferBytes(source.backing.format) != 0
	                                       ? DepthAspectTransferBytes(source.backing.format)
	                                       : source.info.bytes_per_block;
	const auto     destination_bytes = DepthAspectTransferBytes(backing.format) != 0
	                                       ? DepthAspectTransferBytes(backing.format)
	                                       : info.bytes_per_block;
	const uint32_t source_block      = source.info.IsBlock() ? 4u : 1u;
	const uint32_t destination_block = info.IsBlock() ? 4u : 1u;
	EXIT_IF(levels == 0 || source_bytes == 0 || source_bytes != destination_bytes ||
	        source_block != destination_block);
	const auto source_transform = source.info.GetColorTransform();
	const auto target_transform = info.GetColorTransform();

	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcAccessMask       = vk::AccessFlagBits2::eTransferRead;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	auto command                        = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	// Regions are packed at distinct buffer offsets and copied as one round: one barrier, one
	// vkCmdCopyImageToBuffer with every region, one barrier, one vkCmdCopyBufferToImage. A new
	// round starts only when the buffer is full (each region alone always fits: rows_per_copy).
	// Offsets are multiples of the texel block size and of 4 (depth aspects); regions never
	// overlap in the buffer or in the destination image, and every region's bytes are copied
	// exactly as the former one-region-per-round loop copied them.
	// KYTY_COPY_VIA_BUFFER_BATCH=0 restores one region per round.
	const uint64_t offset_alignment = std::lcm<uint64_t>(source_bytes, 16);
	const size_t   max_regions      = CopyViaBufferBatchEnabled() ? SIZE_MAX : 1;
	std::vector<vk::BufferImageCopy> source_copies;
	std::vector<vk::BufferImageCopy> destination_copies;
	uint64_t                         used = 0;
	const auto                       flush_round = [&] {
		if (source_copies.empty()) {
			return;
		}
		barrier.size          = used;
		barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
		barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		command.pipelineBarrier2(dependency);
		command.copyImageToBuffer(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                          buffer.Handle(), static_cast<uint32_t>(source_copies.size()),
		                          source_copies.data());
		if (source_transform != target_transform) {
			// Converts the round's packed bytes in place (alignment gaps included, never copied);
			// TransformColor's barriers order it between the two copies.
			const TileManager::Result bytes {buffer.Handle(), 0, used};
			if (source_transform != ColorTransform::None) {
				tiler.TransformColor(bytes, bytes, source_transform, false);
			}
			if (target_transform != ColorTransform::None) {
				tiler.TransformColor(bytes, bytes, target_transform, true);
			}
		} else {
			barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
			barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
			command.pipelineBarrier2(dependency);
		}
		command.copyBufferToImage(buffer.Handle(), backing.image,
		                          vk::ImageLayout::eTransferDstOptimal,
		                          static_cast<uint32_t>(destination_copies.size()),
		                          destination_copies.data());
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyViaBufferRounds);
		source_copies.clear();
		destination_copies.clear();
		used = 0;
	};
	for (uint32_t level = 0; level < levels; level++) {
		const auto width             = std::max(source.backing.extent.width >> level, 1u);
		const auto height            = std::max(source.backing.extent.height >> level, 1u);
		const auto source_depth      = source.backing.image_type == vk::ImageType::e3D
		                                   ? std::max(source.backing.extent.depth >> level, 1u)
		                                   : source.backing.layers;
		const auto destination_depth = backing.image_type == vk::ImageType::e3D
		                                   ? std::max(backing.extent.depth >> level, 1u)
		                                   : backing.layers;
		const auto slices            = std::min(source_depth, destination_depth);
		const auto block_rows        = (height + source_block - 1) / source_block;
		const auto row_size =
		    static_cast<uint64_t>((width + source_block - 1) / source_block) * source_bytes;
		const auto rows_per_copy = CopyRows(row_size, block_rows, buffer.Size());
		EXIT_IF(slices == 0 || rows_per_copy == 0);
		for (uint32_t slice = 0; slice < slices; slice++) {
			for (uint32_t block_row = 0; block_row < block_rows; block_row += rows_per_copy) {
				const auto          copy_rows   = std::min(rows_per_copy, block_rows - block_row);
				const auto          y           = block_row * source_block;
				const auto          copy_height = std::min(copy_rows * source_block, height - y);
				const auto          copy_size   = row_size * copy_rows;
				vk::BufferImageCopy source_copy {};
				source_copy.imageSubresource = {
				    source_aspect, level,
				    source.backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				source_copy.imageOffset           = {0, static_cast<int32_t>(y),
				                                     source.backing.image_type == vk::ImageType::e3D
				                                         ? static_cast<int32_t>(slice)
				                                         : 0};
				source_copy.imageExtent           = {width, copy_height, 1};
				auto destination_copy             = source_copy;
				destination_copy.imageSubresource = {
				    destination_aspect, level,
				    backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				destination_copy.imageOffset.z =
				    backing.image_type == vk::ImageType::e3D ? static_cast<int32_t>(slice) : 0;
				auto offset = (used + offset_alignment - 1) / offset_alignment * offset_alignment;
				if (source_copies.size() >= max_regions || offset > buffer.Size() ||
				    copy_size > buffer.Size() - offset) {
					flush_round();
					offset = 0;
				}
				source_copy.bufferOffset      = offset;
				destination_copy.bufferOffset = offset;
				source_copies.push_back(source_copy);
				destination_copies.push_back(destination_copy);
				used = offset + copy_size;
			}
		}
	}
	flush_round();
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyMip(Image& source, uint32_t mip, uint32_t layer) {
	KYTY_GPU_OP_SITE("image.copy_mip");
	NoteContentWrite();
	EXIT_IF(source.backing.samples != backing.samples || mip >= backing.mip_levels ||
	        layer >= backing.layers);
	m_scheduler.EndRendering();
	const auto width  = std::max(backing.extent.width >> mip, 1u);
	const auto height = std::max(backing.extent.height >> mip, 1u);
	const auto depth  = std::max(backing.extent.depth >> mip, 1u);
	EXIT_IF(width != source.backing.extent.width || height != source.backing.extent.height);
	const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
	const auto aspects                             = FullAspectMask(source.backing.format);
	EXIT_IF(aspects != FullAspectMask(backing.format));
	std::array<vk::ImageCopy, 2> copies {};
	uint32_t                     copy_count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
	                         vk::ImageAspectFlagBits::eStencil}) {
		if (!static_cast<bool>(aspects & aspect)) {
			continue;
		}
		auto& copy          = copies[copy_count++];
		copy.srcSubresource = {aspect, 0, 0, source_layers};
		copy.dstSubresource = {aspect, mip, layer, destination_layers};
		copy.extent         = {width, height, depth};
	}
	auto command = m_scheduler.Current().Handle();
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy_count, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

namespace ImageOps {

void Validate(const ImageInfo& info) {
	ValidateOptionalRange(info.data, "data");
	ValidateOptionalRange(info.stencil, "stencil");

	if (info.pixel_format == vk::Format::eUndefined) {
		const bool metadata_empty =
		    info.metadata.range.Empty() && info.metadata.kind == ImageMetadataKind::None &&
		    info.metadata.control == 0 &&
		    info.metadata.compression == VideoOutCompression::Uncompressed &&
		    !info.metadata.stencil_compressed;
		if (info.data.Empty() || info.HasStencil() || !metadata_empty || info.extent.width == 0 ||
		    info.extent.height == 0 || info.extent.depth == 0 || info.resources.levels != 1 ||
		    info.resources.layers != 1 || info.samples != 1 || info.pitch != 0 ||
		    info.bytes_per_block != 0) {
			EXIT("invalid stencil association image\n");
		}
		return;
	}

	if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth == 0 ||
	    info.resources.levels == 0 || info.resources.levels > info.mip_layout.size() ||
	    info.resources.layers == 0 || info.samples == 0 ||
	    vulkan_sample_count(info.samples) == vk::SampleCountFlagBits {} ||
	    info.bytes_per_block == 0 || (info.data.address != 0 && info.pitch == 0)) {
		EXIT("invalid image geometry or format\n");
	}

	switch (info.type) {
		case Prospero::ImageType::kColor1D:
			if (info.extent.height != 1 || info.extent.depth != 1) {
				EXIT("invalid 1D image shape\n");
			}
			break;
		case Prospero::ImageType::kColor3D:
			if (info.resources.layers != 1) {
				EXIT("3D images cannot have array layers\n");
			}
			break;
		case Prospero::ImageType::kColor2D:
			if (info.extent.depth != 1) {
				EXIT("invalid 2D image shape\n");
			}
			break;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(info.type));
	}
	if (info.samples > 1 && info.resources.levels != 1) {
		EXIT("multisampled images cannot have mip levels\n");
	}

	if (info.metadata.stencil_compressed && !info.HasStencil()) {
		EXIT("compressed stencil metadata requires a stencil plane\n");
	}
	switch (info.metadata.kind) {
		case ImageMetadataKind::None:
			if (!info.metadata.range.Empty() || info.metadata.control != 0 ||
			    info.metadata.compression != VideoOutCompression::Uncompressed ||
			    info.metadata.stencil_compressed) {
				EXIT("metadata-free image has metadata state\n");
			}
			break;
		case ImageMetadataKind::Htile:
			if (!info.metadata.range.Valid() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("invalid HTILE metadata\n");
			}
			break;
		case ImageMetadataKind::Dcc:
			if (info.metadata.range.address == 0 ||
			    info.metadata.range.address >= TRACKER_ADDRESS_SIZE ||
			    (info.metadata.range.size != 0 &&
			     info.metadata.range.size > TRACKER_ADDRESS_SIZE - info.metadata.range.address) ||
			    info.metadata.compression == VideoOutCompression::Unsupported) {
				EXIT("invalid DCC metadata\n");
			}
			break;
	}
}

bool BlockStorageUploadsEnabled() {
	static const bool enabled = [] {
		const auto off = [](const char* name) {
			const auto* value = std::getenv(name);
			return value != nullptr && std::strcmp(value, "0") == 0;
		};
		return !off("KYTY_TILER_IMAGE_DIRECT") && !off("KYTY_TILER_IMAGE_DIRECT_BC");
	}();
	return enabled;
}

Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element) {
	switch (bytes_per_element) {
		case 1: return Prospero::BufferFormat::k8UNorm;
		case 2: return Prospero::BufferFormat::k16UNorm;
		case 4: return Prospero::BufferFormat::k32Float;
		case 8: return Prospero::BufferFormat::k16_16_16_16Float;
		case 16: return Prospero::BufferFormat::k32_32_32_32Float;
		default: EXIT("unsupported render-target element size: %u\n", bytes_per_element);
	}
}

} // namespace ImageOps

Image::Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& image_info,
             uint32_t sparse_first_level)
    : info(image_info), live(image_info.data),
      stencil_subresources {0, image_info.resources.levels, 0, image_info.resources.layers},
      m_graphics(graphics), m_scheduler(scheduler) {
	KYTY_PROFILER_FUNCTION();
	ImageOps::Validate(info);
	m_cpu_dirty =
	    !info.data.Empty() && info.metadata.compression == VideoOutCompression::Uncompressed;
	if (info.pixel_format == vk::Format::eUndefined) {
		return;
	}

	vk::ImageCreateInfo create {};
	create.flags         = ImageCreateFlags(graphics, info);
	create.imageType     = HostImageType(info.type);
	create.extent        = info.extent;
	create.mipLevels     = info.resources.levels;
	create.arrayLayers   = info.IsVolume() ? 1u : info.resources.layers;
	create.format        = info.pixel_format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage         = ImageUsageFlags(graphics, info);
	create.samples       = vulkan_sample_count(info.samples);

	const auto accepts = [&](vk::ImageUsageFlags usage, vk::ImageCreateFlags flags) {
		vk::ImageFormatProperties properties {};
		return graphics.GetImageFormatProperties(create.format, create.imageType, create.tiling,
		                                         usage, flags, &properties) == vk::Result::eSuccess &&
		       static_cast<bool>(properties.sampleCounts & create.samples);
	};
	if (!accepts(create.usage, create.flags)) {
		// The device refuses the image (AMD and Intel drivers, for some formats): try it without
		// the usages and flags no role of it needs here. A device that accepts it (NVIDIA) never
		// gets here, so its images are unchanged.
		const auto candidates = DeviceCompat::OptionalImageCreateFallbacks(
		    static_cast<VkImageUsageFlags>(create.usage), static_cast<VkImageCreateFlags>(create.flags),
		    static_cast<VkFormatFeatureFlags>(
		        graphics.GetFormatProperties(create.format).optimalTilingFeatures));
		bool accepted = false;
		for (uint32_t index = 1; index < candidates.count && !accepted; index++) {
			const vk::ImageUsageFlags  usage {candidates.list[index].usage};
			const vk::ImageCreateFlags flags {candidates.list[index].flags};
			if (accepts(usage, flags)) {
				m_dropped_usage = create.usage & ~usage;
				m_dropped_flags = create.flags & ~flags;
				LogImageCreateFallback(create.format, m_dropped_usage, m_dropped_flags);
				create.usage = usage;
				create.flags = flags;
				accepted     = true;
			}
		}
		if (!accepted) {
			EXIT("image format does not support required usage: the device refuses %s images "
			     "(type %s, %u sample(s), extent %ux%ux%u) with usage %s and flags %s, also without "
			     "the optional ones\n",
			     vk::to_string(create.format).c_str(), vk::to_string(create.imageType).c_str(),
			     info.samples, create.extent.width, create.extent.height, create.extent.depth,
			     vk::to_string(create.usage).c_str(), vk::to_string(create.flags).c_str());
		}
	}

	if (sparse_first_level != 0 && graphics.CreateSparseImage(create, sparse_first_level, backing)) {
		// Memory behind the resident levels only (TextureCache::EnsureResidency binds more).
	} else if (!graphics.CreateImage(create, backing)) {
		EXIT("failed to create image: extent=%ux%ux%u format=%d layers=%u levels=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth,
		     static_cast<int>(create.format), create.arrayLayers, create.mipLevels);
	}
	SetVulkanObjectNameF(
	    graphics.device, backing.image,
	    "Kyty.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} samples={}]",
	    info.data.address, info.data.size, info.extent.width, info.extent.height, info.extent.depth,
	    static_cast<uint32_t>(info.pixel_format), info.resources.levels, info.resources.layers,
	    info.samples);
}

uint64_t Image::HashGuestEdges() const {
	std::array<uint8_t, TRACKER_PAGE_SIZE * 2> bytes {};
	const auto                                 range = live;
	const uint64_t head_end =
	    std::min(range.End(), Common::AlignUp(range.address, TRACKER_PAGE_SIZE));
	const uint64_t tail_begin =
	    std::max(range.address, Common::AlignDown(range.End(), TRACKER_PAGE_SIZE));
	const uint64_t head_size    = head_end - range.address;
	const uint64_t tail_address = tail_begin < head_end ? head_end : tail_begin;
	const uint64_t tail_size    = range.End() - tail_address;
	if ((head_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(range.address, bytes.data(), head_size)) ||
	    (tail_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(tail_address, bytes.data() + head_size, tail_size))) {
		EXIT("Image: failed to hash guest backing\n");
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(head_size + tail_size));
}

Image::~Image() {
	KYTY_PROFILER_FUNCTION();
	for (const auto& cached: views) {
		if (cached.view != nullptr) {
			m_graphics.device.destroyImageView(cached.view, nullptr);
		}
	}
	if (backing.image != nullptr) {
		m_graphics.DeleteImage(backing);
	}
}

} // namespace Libs::Graphics
