#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "common/staticOnce.h"

#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/skipInvalid.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/syncEpoch.h"
#include "graphics/host_gpu/renderer/image/nanScrubber.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/watchRange.h"

#include <fmt/format.h>
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/texelImageLookup.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/drawPrep/cpGaps.h"
#include "graphics/host_gpu/renderer/image/dccClear.h"
#include "graphics/host_gpu/renderer/image/gpuWriteRanges.h"
#include "graphics/host_gpu/renderer/image/imageClearRange.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/stagingCopier.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/vramBudget.h"
#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
#include <functional>
#include <cmath>
#include <array>
#include <optional>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// KYTY_WIDE_FILL_CLEAR=1 or 2 (default off, live): see the fill-to-clear conversion below.
Live::Switch g_wide_fill_clear("KYTY_WIDE_FILL_CLEAR", [](const char* value) -> int64_t {
	if (value == nullptr) return 0;
	return std::strcmp(value, "1") == 0 ? 1 : std::strcmp(value, "2") == 0 ? 2 : 0;
});

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Largest GPU-written tiled image (in texels) that is written back to guest memory on submit.
constexpr uint64_t SmallTiledReadbackTexels = 64 * 64;

// The DccImageState* cause in the DCC CPU fallback's stderr line.
const char* DccImageStateName(Profiler::FrameEvent cause) {
	using Event = Profiler::FrameEvent;
	switch (cause) {
		case Event::DccImageStateUnregistered: return "unregistered";
		case Event::DccImageStateStencil: return "stencil";
		case Event::DccImageStateMismatch: return "mismatch";
		case Event::DccImageStateNotGpuModified: return "not-gpu-modified";
		case Event::DccImageStateBufferModified: return "buffer-modified";
		case Event::DccImageStateCpuDirty: return "cpu-dirty";
		case Event::DccImageStatePartial: return "partial";
		case Event::DccImageStateGpuDirtyBytes: return "gpu-dirty-bytes";
		default: return "other";
	}
}
// Overlapping aliases (transient render targets sharing a heap) stay registered until unused
// for this many presented guest frames; see AliasAgeByFrames.
constexpr uint64_t AliasFramesBeforeRemoval = 4;

// KYTY_IMAGE_ALIAS_AGE=ticks restores the previous rule that frees an overlapped alias after
// NumFramesBeforeRemoval scheduler ticks, which is usually within the same guest frame.
static bool AliasAgeByFrames() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr || std::strcmp(value, "ticks") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

// KYTY_STENCIL_DEPTH_ALIAS=1 (default off; Erlou93/KytyPS5-Yotei keeps both): a depth/stencil image
// whose memory a colour interpretation writes (Ghost of Yotei's "Composite depth bias" writes its
// D32S8 target 0x5002870000 as R32 storage) stays alive as a non-owner alias, as colour/colour and
// depth/colour pairs without stencil already do. Freed, its stencil planes had no image until the
// next depth bind recreated it: Deferred wetness read plane A as a null texture, and the recreated
// image's stencil was undefined. The alias syncs the depth aspect from the owner (copies never
// include stencil), so its stencil survives.
static bool StencilDepthAliasEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_STENCIL_DEPTH_ALIAS");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			std::printf("Depth/stencil images kept as aliases of their colour interpretations "
			            "(KYTY_STENCIL_DEPTH_ALIAS=1)\n");
		}
		return on;
	}();
	return enabled;
}

[[nodiscard]] bool EnvNotZero(const char* name) {
	const auto* value = std::getenv(name);
	return value == nullptr || std::strcmp(value, "0") != 0;
}

// KYTY_DIRECT_IMAGE_COPY=0 restores the image->buffer->image copy for every depth<->color
// reinterpretation. KYTY_DIRECT_IMAGE_COPY_SHADER=0 disables only the one-pass shader
// reinterpretation; KYTY_DIRECT_IMAGE_COPY_M8=0 (read at device creation) only the
// VK_KHR_maintenance8 depth<->color vkCmdCopyImage.
static bool DirectImageCopyEnabled() {
	static const bool enabled = EnvNotZero("KYTY_DIRECT_IMAGE_COPY");
	return enabled;
}

static bool DirectImageCopyShaderEnabled() {
	static const bool enabled =
	    DirectImageCopyEnabled() && EnvNotZero("KYTY_DIRECT_IMAGE_COPY_SHADER");
	return enabled;
}

// KYTY_ALIAS_SYNC_SKIP=0 copies the owner into a kept alias on every ownership switch, even when
// the alias already holds exactly the owner's native contents (see Image::ContentSerial).
static bool AliasSyncSkipEnabled() {
	static const bool enabled = EnvNotZero("KYTY_ALIAS_SYNC_SKIP");
	return enabled;
}

// KYTY_ALIAS_SYNC_DEPTH_PLANE=1 (live, default off): an alias without stencil (R32_SFLOAT or
// D32_SFLOAT) synchronized by a full copy from a D32_SFLOAT_S8_UINT owner of the same extent,
// levels and layers adopts the owner's content serial, as lossless copies do. The copy moves the
// whole depth plane, which is all the alias holds, so equal serials still prove that the alias
// holds the owner's depth bits (the only bits any alias sync copies; stencil is never copied).
// Without it (CopyImage: "combined depth/stencil images never compare equal") Ghost of Yotei
// recopies its 1920x1080 scene depth into the R32 alias before every draw that samples it through a
// colour T#, although the draws write no depth: ~45-73 copies, 16.5 ms GPU per frame (gpu-yotei
// 2026-11-05).
static Live::Switch g_alias_sync_depth_plane("KYTY_ALIAS_SYNC_DEPTH_PLANE",
                                             [](const char* value) -> int64_t {
	                                             return value != nullptr && value[0] == '1' ? 1 : 0;
                                             });

[[nodiscard]] static bool AliasSyncDepthPlane(const Image& alias, const Image& owner) {
	if (!g_alias_sync_depth_plane.On()) {
		return false;
	}
	const auto& a = alias.backing;
	const auto& o = owner.backing;
	return o.format == vk::Format::eD32SfloatS8Uint &&
	       (a.format == vk::Format::eR32Sfloat || a.format == vk::Format::eD32Sfloat) &&
	       a.extent == o.extent && a.image_type == o.image_type && a.mip_levels == o.mip_levels &&
	       a.layers == o.layers && a.samples == o.samples;
}

// Moves the texel rectangle's dimensions into [0, extent) (empty when outside).
[[nodiscard]] vk::Rect2D ClampRect(const vk::Rect2D& rect, uint32_t width, uint32_t height) {
	const int64_t left   = std::clamp<int64_t>(rect.offset.x, 0, width);
	const int64_t top    = std::clamp<int64_t>(rect.offset.y, 0, height);
	const int64_t right  = std::clamp<int64_t>(int64_t {rect.offset.x} + rect.extent.width, 0, width);
	const int64_t bottom =
	    std::clamp<int64_t>(int64_t {rect.offset.y} + rect.extent.height, 0, height);
	if (right <= left || bottom <= top) {
		return {};
	}
	return {{static_cast<int32_t>(left), static_cast<int32_t>(top)},
	        {static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)}};
}

[[nodiscard]] bool RectContains(const vk::Rect2D& outer, const vk::Rect2D& inner) {
	return inner.offset.x >= outer.offset.x && inner.offset.y >= outer.offset.y &&
	       int64_t {inner.offset.x} + inner.extent.width <=
	           int64_t {outer.offset.x} + outer.extent.width &&
	       int64_t {inner.offset.y} + inner.extent.height <=
	           int64_t {outer.offset.y} + outer.extent.height;
}

// KYTY_DIRECT_IMAGE_COPY_WIDE=0 restores the depth-only format lists of the maintenance8 copy
// and the D32-only color -> depth shader copy (no combined depth/stencil formats, no R16 SFLOAT /
// SNORM partners of D16).
static bool DirectImageCopyWideEnabled() {
	static const bool enabled = EnvNotZero("KYTY_DIRECT_IMAGE_COPY_WIDE");
	return enabled;
}

// VK_KHR_maintenance8 "compatible depth-stencil and color formats" (formats-compatible-zs-color):
// the DEPTH aspect of D32_SFLOAT and D32_SFLOAT_S8_UINT is size-compatible with R32_SFLOAT /
// R32_SINT / R32_UINT, that of D16_UNORM and D16_UNORM_S8_UINT with R16_SFLOAT / R16_UNORM /
// R16_SNORM / R16_UINT / R16_SINT. The copy moves raw depth-aspect bits (the stencil aspect of a
// combined format is neither read nor written), exactly what Image::CopyImageWithBuffer moves
// through its buffer. 24-bit depth is left out: its copies leave the padding bits undefined.
[[nodiscard]] bool Maintenance8CopyCompatible(vk::Format depth, vk::Format color) {
	const bool wide = DirectImageCopyWideEnabled();
	switch (depth) {
		case vk::Format::eD32SfloatS8Uint:
			if (!wide) {
				return false;
			}
			[[fallthrough]];
		case vk::Format::eD32Sfloat:
			return color == vk::Format::eR32Sfloat || color == vk::Format::eR32Uint ||
			       color == vk::Format::eR32Sint;
		case vk::Format::eD16UnormS8Uint:
			if (!wide) {
				return false;
			}
			[[fallthrough]];
		case vk::Format::eD16Unorm:
			return color == vk::Format::eR16Unorm || color == vk::Format::eR16Uint ||
			       color == vk::Format::eR16Sint ||
			       (wide && (color == vk::Format::eR16Sfloat || color == vk::Format::eR16Snorm));
		default: return false;
	}
}

[[nodiscard]] const char* UploadBindingName(TextureCache::BindingType binding) {
	switch (binding) {
		case TextureCache::BindingType::Texture: return "texture";
		case TextureCache::BindingType::Storage: return "storage";
		case TextureCache::BindingType::RenderTarget: return "render-target";
		case TextureCache::BindingType::DepthTarget: return "depth-target";
		case TextureCache::BindingType::VideoOut: return "video-out";
	}
	return "other";
}

// The stock Tracy CSV exporter supports -m. Keep messages free of commas/newlines,
// and bound diagnostics independently of how long a detailed capture stays connected.
std::atomic<uint32_t> g_dcc_diagnostic_messages {0};

template <typename... Args>
void TraceDccDiagnostic(const char* format, Args... args) {
	if (!Profiler::DetailedEnabled() || !tracy::ProfilerAvailable() || !TracyIsConnected) return;
	constexpr uint32_t MaxMessages = 8192;
	const auto ordinal = g_dcc_diagnostic_messages.fetch_add(1, std::memory_order_relaxed);
	if (ordinal >= MaxMessages) {
		if (ordinal == MaxMessages) TracyMessageLS("DCC_DIAG_LIMIT messages=8192", 0);
		return;
	}
	char text[768];
	const auto written = std::snprintf(text, sizeof(text), format, args...);
	if (written > 0 && static_cast<size_t>(written) < sizeof(text)) {
		TracyMessageS(text, static_cast<size_t>(written), 0);
	}
}

// KYTY_DCC_CLEAR_KEY_10=1 (default off; TheCruZ/KytyPS5-GTA eca56bfd): DCC key 0x10 is a clear to
// the CB_COLOR_CLEAR_WORD register value, like 0x20, both on the CPU path and the GPU inspection.
static bool DccClearKey10Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DCC_CLEAR_KEY_10");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

[[nodiscard]] bool DecodeDccClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	switch (code) {
		case 0x00:
		case 0x20:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		case 0x10:
			if (DccClearKey10Enabled()) {
				break;
			}
			return false;
		default: return false;
	}
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	// KYTY_DCC_CLEAR_KEY_10=1: PS5 DCC fast clears to the CB_COLOR_CLEAR_WORD value can fill the
	// keys with 0x10, not the GFX10 0x20 (GTA V's R16_FLOAT particle depth cleared to 65504).
	if (code == 0x20 || code == 0x10) {
		// Clear-to-register is a color-buffer operation; the texture pipe cannot decode it.
		// KYTY_CLEAR_REGISTER_WIDE (imageInfo.h): both CLEAR_WORDs, so 64-bit targets decode.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.dcc_clear_register_valid &&
		       (ClearRegisterWideEnabled()
		            ? DecodePackedColorClear64(
		                  format,
		                  GuestColorWordToHost(metadata.dcc_clear_word, desc.info.GetColorTransform()),
		                  metadata.dcc_clear_word1, clear)
		            : DecodePackedColorClear(
		                  format,
		                  GuestColorWordToHost(metadata.dcc_clear_word, desc.info.GetColorTransform()),
		                  clear));
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                              uint64_t slice_stride,
                                                              vk::ImageAspectFlags aspect) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {aspect, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	return copies;
}

[[nodiscard]] std::vector<GpuTileInfo> BuildDepthTiles(const ImageInfo& info) {
	TileBlockLayout block {};
	EXIT_NOT_IMPLEMENTED(
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
	const auto               full_slice_size = info.data.size / info.resources.layers;
	std::vector<GpuTileInfo> tiles;
	tiles.reserve(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		const auto offset = full_slice_size * layer;
		tiles.push_back({block.family, block.bytes_per_element, offset, full_slice_size, offset,
		                 full_slice_size, 0, info.extent.width, info.extent.height, 1, info.pitch});
		tiles.back().surface_z = layer;
	}
	return tiles;
}


// DCC/CMASK describe colour surfaces. A lookup of such a description can resolve to an image that
// holds depth (same memory reused as depth and colour, e.g. Astro Bot's D32S8 targets reused as RG16F
// DCC targets) or to a stencil plane record. Neither can take a colour clear and the description's
// metadata is not theirs, so the materialisation does nothing for them (reported a few times).
// KYTY_CLEAR_TRACE=1 (diagnostic): one stderr line per metadata address and outcome of the DCC/CMASK
// clear materialisation.
// 0 off, 1 the aliases no longer refuse the clear (decided from the CMASK bytes as usual), 2 the
// target is cleared once per GC tick without reading or writing the CMASK bytes (the aliases are
// live images sharing the memory within the frame; writing the bytes dirties them).
int CmaskAliasSupersededMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_CMASK_ALIAS_SUPERSEDED");
		return value != nullptr ? std::atoi(value) : 0;
	}();
	return mode;
}

bool CmaskAliasSupersededEnabled() {
	return CmaskAliasSupersededMode() != 0;
}

void ClearTrace(const char* site, const char* outcome, uint64_t metadata, uint64_t metadata_size,
                uint64_t data, uint64_t data_size, uint32_t width, uint32_t height, uint32_t extra) {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_CLEAR_TRACE");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	if (!enabled) {
		return;
	}
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	const uint64_t key = metadata ^ (std::hash<std::string_view> {}(site) * 31u) ^
	                     (std::hash<std::string_view> {}(outcome) << 1u) ^ (uint64_t {extra} << 48u);
	{
		std::scoped_lock lock(mutex);
		if (!seen.insert(key).second) {
			return;
		}
	}
	std::fprintf(stderr,
	             "ClearTrace: %s %s meta=0x%" PRIx64 "+0x%" PRIx64 " data=0x%" PRIx64 "+0x%" PRIx64
	             " %ux%u extra=0x%x\n",
	             site, outcome, metadata, metadata_size, data, data_size, width, height, extra);
	std::fflush(stderr);
}

// KYTY_WATCH_RANGE: texture-cache operations that write an image overlapping a watched range.
void ImageWatch(const char* what, const Image& image) {
	if (WatchRange::Enabled()) WatchRange::Note(what, image.info.data.address, image.info.data.size);
}

bool ColourMetadataTargetsDepth(const char* site, ImageId id, const Image& image) {
	if (!image.info.IsDepth() && !image.depth_id) {
		return false;
	}
	static std::atomic<uint32_t> reported {0};
	if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
		std::printf("TextureCache: %s metadata skipped for a depth image: image=%u:%u host_format=%s "
		            "depth_id=%u stencil=%u address=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		            site, id.index, id.generation, vk::to_string(image.info.pixel_format).c_str(),
		            image.depth_id ? 1u : 0u, image.info.HasStencil() ? 1u : 0u,
		            image.info.data.address, image.info.data.size);
		std::fflush(stdout);
	}
	return true;
}

// The compare functions of the binding being resolved (KYTY_NULL_COMPARE_DEPTH; set by the
// descriptor resolution right before it looks the image up on this thread).
thread_local uint32_t t_null_compare_funcs = 0;

} // namespace

// KYTY_IMAGE_CONTAINS (see ImageSubresources::Contains).
bool ImageContainsEnabled() {
	static constexpr auto on_init = [] {
		const char* value = std::getenv("KYTY_IMAGE_CONTAINS");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

bool StencilRestoreGuardEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_STENCIL_RESTORE_GUARD");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool NullTextureZeroEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_NULL_TEXTURE_ZERO");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool NullCompareDepthEnabled() {
	static constexpr auto on_init = [] {
		const char* value = std::getenv("KYTY_NULL_COMPARE_DEPTH");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

void SetNullCompareFuncs(uint32_t funcs) {
	t_null_compare_funcs = funcs;
}

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	ColorTransform                   color_transform = ColorTransform::None;
	bool                             valid           = false;

	[[nodiscard]] uint64_t LinearSize() const {
		uint64_t size = 0;
		for (const auto& tile: tiles) {
			size = std::max(size, tile.linear_offset + tile.linear_size);
		}
		return size;
	}
};

struct TextureCache::ImageDownload {
	TextureTransfer texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()),
      m_readback_tiled_images(Config::ReadbackTiledImagesEnabled()),
      m_readback_storage_images(Config::ReadbackStorageImagesEnabled()) {
	m_image_page_counts = std::make_unique<std::atomic<uint32_t>[]>(
	    size_t {1} << (ImagePageTable::kAddressSpaceBits - ImagePageTable::kPageBits));
	m_fault_fast_path = EnvNotZero("KYTY_TEXTURE_FAULT_FAST_PATH");
	m_texel_sync_skip = EnvNotZero("KYTY_TEXEL_SYNC_SKIP");
	LOGF("Image alias bytes (KYTY_ALIAS_BYTES): %s\n",
	     AliasBytesEnabled() ? "ownership follows written bytes, owned bytes reach the buffer before "
	                           "buffer reads and frees"
	                         : "off (whole-image ownership)");
	if (const auto* exact = std::getenv("KYTY_IMAGE_EXACT_RANGE_INVALIDATE");
	    exact != nullptr && *exact != '\0' && std::strcmp(exact, "0") != 0) {
		m_exact_range_mode =
		    std::strcmp(exact, "count") == 0 || std::strcmp(exact, "verify") == 0 ? 2 : 1;
		std::printf("Image exact-range invalidate (KYTY_IMAGE_EXACT_RANGE_INVALIDATE=%s): %s\n", exact,
		            m_exact_range_mode == 1
		                ? "a buffer write over exactly one image's range rebuilds only the images "
		                  "inside it"
		                : "count only (every overlapped image is rebuilt as before)");
		std::fflush(stdout);
	}
	if (const auto* partial = std::getenv("KYTY_IMAGE_PARTIAL_GPU_WRITE");
	    partial != nullptr && *partial != '\0' && std::strcmp(partial, "0") != 0) {
		m_partial_gpu_write        = 1;
		m_partial_gpu_write_verify = std::strcmp(partial, "verify") == 0;
		const auto* stencil        = std::getenv("KYTY_IMAGE_PARTIAL_GPU_WRITE_STENCIL");
		m_partial_gpu_write_stencil_keep = stencil != nullptr && std::strcmp(stencil, "keep") == 0;
		std::printf("Image partial GPU-write (KYTY_IMAGE_PARTIAL_GPU_WRITE=%s): a buffer write over "
		            "part of an image rebuilds only the subresources / block rows over it%s%s%s\n",
		            partial, m_partial_gpu_write_verify ? ", each checked against a whole rebuild" : "",
		            m_partial_gpu_write_stencil_keep ? "; stencil records trust depth-target draws" : "",
		            AliasBytesEnabled() ? "" : " (colour/depth images need KYTY_ALIAS_BYTES: stencil records only)");
		std::fflush(stdout);
	}
	if (const auto* grow = std::getenv("KYTY_DEPTH_GROW_RECREATE"); grow != nullptr && grow[0] == '1') {
		m_depth_grow_recreate = true;
		std::printf("Depth grow recreate (KYTY_DEPTH_GROW_RECREATE=1): a depth target that grows inside "
		            "its pitch gets a larger image (old depth/stencil copied, the rest from guest memory)\n");
		std::fflush(stdout);
	}
	m_gpu_write_skip  = EnvNotZero("KYTY_GPU_WRITE_IMAGE_SKIP");
	if (m_gpu_write_skip) {
		const auto* verify = std::getenv("KYTY_GPU_WRITE_IMAGE_SKIP_VERIFY");
		if (verify != nullptr && *verify != '\0' && std::strcmp(verify, "0") != 0) {
			m_gpu_write_skip_verify = std::strcmp(verify, "exit") == 0 ? 2 : 1;
		}
	}
	m_lru_touch_skip = EnvNotZero("KYTY_IMAGE_LRU_SKIP");
	if (const auto* verify = std::getenv("KYTY_IMAGE_LRU_SKIP_VERIFY");
	    verify != nullptr && *verify != '\0' && std::strcmp(verify, "0") != 0) {
		m_lru_touch_skip_verify = std::strcmp(verify, "exit") == 0 ? 2 : 1;
	}
	if (const auto* refresh = std::getenv("KYTY_DCC_GPU_REFRESH");
	    refresh != nullptr && std::strcmp(refresh, "1") == 0) {
		m_dcc_gpu_refresh = true;
		const auto* verify = std::getenv("KYTY_DCC_GPU_REFRESH_VERIFY");
		if (verify != nullptr && *verify != '\0' && std::strcmp(verify, "0") != 0) {
			m_dcc_gpu_refresh_verify = std::strcmp(verify, "exit") == 0 ? 2 : 1;
		}
	}
	m_dcc_gpu_wide      = static_cast<uint32_t>(ParseDccGpuWide(std::getenv("KYTY_DCC_GPU_WIDE")));
	if (const auto* verify = std::getenv("KYTY_DCC_GPU_WIDE_VERIFY");
	    verify != nullptr && *verify != '\0' && std::strcmp(verify, "0") != 0) {
		m_dcc_gpu_wide_verify = std::strcmp(verify, "exit") == 0 ? 2 : 1;
	}
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
	const auto* lookup = std::getenv("KYTY_IMAGE_LOOKUP");
	if (lookup != nullptr && std::strcmp(lookup, "legacy") == 0) {
		m_image_lookup_mode = ImageLookupMode::Legacy;
		LOGF("Image lookup: legacy full-region search\n");
	} else if (lookup != nullptr && std::strcmp(lookup, "verify") == 0) {
		m_image_lookup_mode = ImageLookupMode::Verify;
		LOGF("Image lookup: cross-check first-page search, use legacy results\n");
	} else {
		LOGF("Image lookup: first-page exact search\n");
	}
	const auto* dirty_query = std::getenv("KYTY_DIRTY_IMAGE_QUERY");
	m_direct_dirty_image_query = dirty_query != nullptr && std::strcmp(dirty_query, "direct") == 0;
	const auto* clean_proofs = std::getenv("KYTY_TEXTURE_CLEAN_PROOFS");
	m_clean_image_proofs = clean_proofs != nullptr && std::strcmp(clean_proofs, "1") == 0;
	if (m_direct_dirty_image_query) {
		LOGF("Dirty image query: direct page-owner search\n");
	}
	const auto* gpu_dcc = std::getenv("KYTY_DCC_GPU");
	if (gpu_dcc != nullptr && std::strcmp(gpu_dcc, "1") == 0) {
		m_dcc_clear = std::make_unique<DccClearHelper>(graphics, scheduler);
		LOGF("DCC materialization: GPU validation and conditional clear (%s)\n",
		     m_dcc_clear->Available() ? "any 8/16/32/64/128-bit color view format"
		                              : "unavailable on this device");
	}
	if (const auto* idle = std::getenv("KYTY_VRAM_IDLE_FRAMES"); idle != nullptr) {
		m_idle_frames = std::strtoull(idle, nullptr, 10);
		if (m_idle_frames != 0) {
			LOGF("Image cache: images unused for %" PRIu64 " frames are freed (KYTY_VRAM_IDLE_FRAMES)\n",
			     m_idle_frames);
		}
	}
	if (const auto* keep = std::getenv("KYTY_VRAM_IDLE_KEEP_POOLS_MB"); keep != nullptr) {
		m_idle_keep_pool_bytes = std::strtoull(keep, nullptr, 10) << 20u;
	}
	if (const auto* pressure = std::getenv("KYTY_VRAM_PRESSURE_FRAMES"); pressure != nullptr) {
		m_pressure_frames = std::strtoull(pressure, nullptr, 10);
		if (m_pressure_frames != 0) {
			LOGF("Image cache: above the collection trigger, images unused for %" PRIu64
			     " frames are freed instead of the submission-age collection "
			     "(KYTY_VRAM_PRESSURE_FRAMES)\n",
			     m_pressure_frames);
		}
	}
	const auto* policy = std::getenv("KYTY_IMAGE_CACHE_POLICY");
	m_pressure_gc_enabled = policy != nullptr && std::strcmp(policy, "pressure") == 0;
	if (m_pressure_gc_enabled) {
		m_pressure_gc_policy.SetBudget(m_graphics.GetTotalMemoryBudget());
		LOGF("Image cache policy: pressure (Approach B), low=%" PRIu64
		     " MiB high=%" PRIu64 " MiB critical=%" PRIu64 " MiB\n",
		     m_pressure_gc_policy.Low() / ImageCacheGcPolicy::MiB,
		     m_pressure_gc_policy.High() / ImageCacheGcPolicy::MiB,
		     m_pressure_gc_policy.Critical() / ImageCacheGcPolicy::MiB);
	} else {
		LOGF("Image cache policy: submission age (Approach A)\n");
	}
	// Texture streaming. KYTY_TEXTURE_PARTIAL_UPLOAD=1 enables chunk-granular CPU write tracking
	// and partial refreshes; the default is whole-image tracking and refreshes. The partial path
	// left stale coarse mips in reused streaming slots (Astro Bot, U44: distant Sky Garden trees
	// washed out or missing leaves, correct up close), so it is off until that is fixed.
	// KYTY_TEXTURE_DIRTY_CHUNK_KB (power of two, 4..4096, default 64)
	// is the write-tracking granularity. KYTY_TEXTURE_PARTIAL_BANDS=0 refreshes whole mip
	// levels instead of rows of tile blocks. KYTY_TEXTURE_PARTIAL_VERIFY=1 hashes every clean
	// chunk before a partial refresh and falls back to a full one on a mismatch (diagnostic).
	if (const auto* partial = std::getenv("KYTY_TEXTURE_PARTIAL_UPLOAD"); partial != nullptr) {
		m_partial_upload = std::strcmp(partial, "0") != 0;
	} else {
		m_partial_upload = false;
	}
	m_partial_bands  = EnvNotZero("KYTY_TEXTURE_PARTIAL_BANDS");
	// KYTY_IMAGE_GPU_KEEP (default on; =0 off): see KeepEligible (idea from chenxiao07/KytyPS5
	// e7bf421a1 / 21cfed4aa / 8fe29414f).
	m_gpu_keep = EnvNotZero("KYTY_IMAGE_GPU_KEEP");
	// KYTY_IMAGE_GC_FILTER (default on; =0 off): see RunGarbageCollector.
	m_gc_filter_candidates = EnvNotZero("KYTY_IMAGE_GC_FILTER");
	// KYTY_IMAGE_WRITEBACK_GUARD (default on; =0 off): see WriteBackImageBytes.
	m_writeback_guard = EnvNotZero("KYTY_IMAGE_WRITEBACK_GUARD");
	LOGF("Image GPU keep (KYTY_IMAGE_GPU_KEEP): %s\n",
	     m_gpu_keep ? "CPU writes and partial unmaps replace only their bytes of GPU-written images"
	                : "off (whole-image refresh)");
	if (const auto* verify = std::getenv("KYTY_TEXTURE_PARTIAL_VERIFY");
	    verify != nullptr && std::strcmp(verify, "1") == 0) {
		m_partial_verify = true;
	}
	if (const auto* chunk = std::getenv("KYTY_TEXTURE_DIRTY_CHUNK_KB"); chunk != nullptr) {
		const auto kib = std::strtoull(chunk, nullptr, 10);
		if (kib >= 4 && kib <= 4096 && std::has_single_bit(kib)) {
			m_chunk_shift = static_cast<uint32_t>(std::countr_zero(kib)) + 10u;
		} else {
			LOGF("KYTY_TEXTURE_DIRTY_CHUNK_KB=%s ignored (power of two 4..4096 expected)\n", chunk);
		}
	}
	// KYTY_TEXTURE_OVERLAP_KEEP_FRAMES=N (default 0: off) keeps a clean, texture-only image that a
	// newer overlapping image would retire as stale, while it was used in the last N frames.
	if (const auto* keep = std::getenv("KYTY_TEXTURE_OVERLAP_KEEP_FRAMES"); keep != nullptr) {
		m_overlap_keep_frames = std::strtoull(keep, nullptr, 10);
	}
	LOGF("Texture streaming: partial_upload=%u chunk=%u KiB bands=%u verify=%u overlap_keep=%" PRIu64
	     " frames\n",
	     m_partial_upload ? 1u : 0u, (1u << m_chunk_shift) >> 10u, m_partial_bands ? 1u : 0u,
	     m_partial_verify ? 1u : 0u, m_overlap_keep_frames);
	// KYTY_TEXTURE_RESIDENT_MIPS=0 registers, watches and uploads every level of every texture.
	// By default a sampled texture holds only the levels its views can sample (finer than the
	// T# MIN_LOD / BASE_LEVEL clamp is never read); =poison also fills the other levels with a
	// visible marker to check that claim on screen.
	// The Sky Garden water's normal map keeps mip 0 non-resident (zero) because its T# MIN_LOD is
	// 1.0, so no view samples it; the white water was the stale quarter-res input B (U44 bisect).
	m_residency = ResidencyMode::On;
	if (const auto* residency = std::getenv("KYTY_TEXTURE_RESIDENT_MIPS"); residency != nullptr) {
		if (std::strcmp(residency, "poison") == 0) {
			m_residency = ResidencyMode::Poison;
		} else if (std::strcmp(residency, "0") == 0) {
			m_residency = ResidencyMode::Off;
		}
	}
	// KYTY_TEXTURE_SPARSE_RESIDENCY=1 (default off; device support required, see
	// GraphicContext::sparse_residency_image_enabled): a partially resident texture gets memory
	// behind its resident levels only. Not with poison (its marker writes need the memory).
	m_sparse_textures = graphics.sparse_residency_image_enabled && m_residency == ResidencyMode::On;
	if (m_sparse_textures) {
		LOGF("Texture streaming: partially resident textures are sparse residency images\n");
	}
	if (const auto* idle = std::getenv("KYTY_TEXTURE_RESIDENT_IDLE_FRAMES"); idle != nullptr) {
		m_resident_idle_frames = std::strtoull(idle, nullptr, 10);
	}
	LOGF("Texture streaming: resident mip levels %s, idle partially resident images retired "
	     "after %" PRIu64 " frames\n",
	     m_residency == ResidencyMode::Off      ? "off (whole chains)"
	     : m_residency == ResidencyMode::Poison ? "on, non-resident levels poisoned"
	                                            : "on",
	     m_resident_idle_frames);
	// KYTY_TEXTURE_ASYNC_STAGING=0 keeps texture refresh staging copies on the GPU thread.
	if (EnvNotZero("KYTY_TEXTURE_ASYNC_STAGING")) {
		m_staging_copier = std::make_unique<StagingCopier>(graphics);
		m_scheduler.SetSubmitDependency(m_staging_copier.get());
		// With resizable BAR the worker writes straight into VRAM. Only when a device-local
		// host-visible heap is large, so the ring cannot starve other users of a 256 MiB BAR.
		const auto& memory = graphics.GetPhysicalDeviceMemoryProperties();
		uint64_t    bar_heap = 0;
		for (uint32_t index = 0; index < memory.memoryTypeCount; index++) {
			const auto flags = memory.memoryTypes[index].propertyFlags;
			if ((flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
			    (flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
				bar_heap = std::max<uint64_t>(bar_heap,
				                              memory.memoryHeaps[memory.memoryTypes[index].heapIndex].size);
			}
		}
		uint64_t ring_mib = 256;
		if (const auto* size = std::getenv("KYTY_TEXTURE_STAGING_MB"); size != nullptr) {
			ring_mib = std::strtoull(size, nullptr, 10);
		}
		constexpr uint64_t MiB = 1024ull * 1024;
		if (EnvNotZero("KYTY_TEXTURE_STAGING_REBAR") && ring_mib != 0 &&
		    bar_heap >= 2048 * MiB && ring_mib * MiB <= bar_heap / 8) {
			m_texture_staging = std::make_unique<StreamBuffer>(graphics, scheduler, MemoryUsage::Stream,
			                                                   ring_mib * MiB);
		}
		LOGF("Texture streaming: async staging copies on, staging ring %s (%" PRIu64
		     " MiB, largest device-local host-visible heap %" PRIu64 " MiB)\n",
		     m_texture_staging ? "device-local" : "shared upload ring",
		     m_texture_staging ? ring_mib : 0, bar_heap / MiB);
	} else {
		LOGF("Texture streaming: async staging copies off\n");
	}
}

bool TextureCache::AliasBytesEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return AliasAgeByFrames() && EnvNotZero("KYTY_ALIAS_BYTES");
	}));
	return enabled;
}

StreamBuffer& TextureCache::StagingRing() {
	return m_texture_staging ? *m_texture_staging
	                         : m_buffer_cache.GetUtilityBuffer(MemoryUsage::Upload);
}

TextureCache::~TextureCache() {
	if (m_staging_copier) {
		// The scheduler has drained by now; no later submission may wait on the copier.
		m_scheduler.SetSubmitDependency(nullptr);
	}
	if (m_image_lookup_mode == ImageLookupMode::Verify) {
		LOGF("Image lookup verification final: checks=%" PRIu64 " mismatches=%" PRIu64 "\n",
		     m_image_lookup_checks, m_image_lookup_mismatches);
	}
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

bool TextureCache::SafeToDownload(const Image& image) {
	// Non-resident levels hold undefined native contents and must never reach guest memory.
	// (GPU-written images are always fully resident; this only keeps downloads exact.)
	if (!image.SafeToDownload() || !image.FullyResident()) {
		return false;
	}
	const auto range = image.info.data;
	return !m_buffer_cache.HasGpuDirtyBytes(range.address, range.size);
}

// KYTY_GC_KEEP_SMALL_RT=1 (default off; TheCruZ/KytyPS5-GTA 23585d0c): the LRU and pressure collectors
// never evict a render target of at most 1024x1024. Those are the layers a game draws once and
// samples from then on (glyph atlases, cached UI); evicting one that cannot be downloaded loses its
// contents (Wolverine's missing letters). Freeing their memory (unmap) still drops them.
static bool KeepSmallRenderTarget(const Image& image) {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_GC_KEEP_SMALL_RT");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled && image.usage.render_target && image.info.extent.width <= 1024 &&
	       image.info.extent.height <= 1024;
}

// KYTY_IMAGE_SUPERSEDES_GPU_DIRTY=0 keeps refusing texel-read syncs of a GPU-modified image whose
// range holds any GPU-dirty buffer bytes (the previous behaviour).
static bool ImageSupersedesGpuDirtyEnabled() {
	static const bool enabled = EnvNotZero("KYTY_IMAGE_SUPERSEDES_GPU_DIRTY");
	return enabled;
}

bool TextureCache::SupersedesGpuDirtyBytes(const Image& image) {
	// Every bounded GPU buffer write (storage bindings, fills, copies, GPU-timeline WRITE_DATA,
	// image writebacks) clears the GPU ownership of the images it overlaps
	// (InvalidateMemoryFromGPU). An image that is GPU-modified now was therefore written after all
	// of them, and after every unbounded writer bound before its content serial was issued.
	// (An adopted serial is never newer than the image's own last write, so this stays exact
	// or conservative.) GPU-dirty bytes in its range are then stale copies: the image downloads
	// over them. Without this, one image writeback (which leaves the whole moved range GPU-dirty
	// until a readback) blocks every later texel-read sync of a newer render target there: Astro
	// Bot's water copies its scene colour target with a compute memcpy that read stale bytes.
	return ImageSupersedesGpuDirtyEnabled() && image.IsGpuModified() &&
	       image.ContentSerial() > m_buffer_cache.UnboundedWriteSerial();
}

bool TextureCache::SafeToSyncIntoBuffer(const Image& image) {
	if (!image.SafeToDownload() || !image.FullyResident()) {
		return false;
	}
	const auto range = image.info.data;
	return !m_buffer_cache.HasGpuDirtyBytes(range.address, range.size) ||
	       SupersedesGpuDirtyBytes(image);
}

ImageId TextureCache::InsertImage(const ImageInfo& info, uint32_t resident_first,
                                  uint64_t resident_prefix) {
	if (resident_first != 0 && resident_prefix == 0) {
		resident_prefix = ResidentPrefixSize(info, resident_first);
	}
	// KYTY_TEXTURE_SPARSE_RESIDENCY: memory behind the resident levels only.
	const uint32_t sparse_first = m_sparse_textures && resident_first != 0 && resident_prefix != 0
	                                  ? resident_first
	                                  : 0u;
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info, sparse_first);
	if (ZeroNewImagesEnabled()) {
		// KYTY_ZERO_NEW_IMAGES: fresh memory reads as zero, as on the PS5 (an upload after this
		// replaces what it covers).
		(void)m_slot_images[id].ZeroFresh();
	}
	if (VramStats::Enabled()) {
		if (const auto bytes = m_graphics.NativeImageBytes(m_slot_images[id].backing); bytes != 0) {
			m_vram_creates.first++;
			m_vram_creates.second += bytes;
		}
	}
	if (resident_first != 0) {
		if (resident_prefix != 0) {
			auto& image          = m_slot_images[id];
			image.resident_first = resident_first;
			NoteImageStateChange();
			image.live           = {info.data.address, resident_prefix};
			NoteImageStateChange();
			if (m_resident_idle_frames != 0) {
				m_partial_images.push_back(id);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidentImages);
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidentLevelsSkipped,
			                          resident_first);
		}
	}
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

uint32_t TextureCache::RequestedFirstLevel(const ImageDesc& desc, uint32_t levels) const {
	if (m_residency == ResidencyMode::Off || desc.type != BindingType::Texture || levels <= 1) {
		return 0;
	}
	// The finest level a sampled view can read. Its minimum LOD (the T# MIN_LOD clamp, applied
	// through VK_EXT_image_view_min_lod) bounds every level-of-detail computation: trilinear
	// filtering at the clamp reads floor(min_lod) and the next coarser level, nearest-mip
	// selection rounds only towards coarser levels, and texel fetches of finer levels are out
	// of range for the view. Levels below base_level are outside the view altogether.
	const auto& view  = desc.view_info;
	const auto  first = view.base_level + (view.min_lod >> 8u);
	return std::min(first, levels - 1u);
}

uint64_t TextureCache::ResidentPrefixSize(const ImageInfo& info, uint32_t first_level) const {
	if (first_level == 0 || m_residency == ResidencyMode::Off || info.data.Empty() ||
	    info.type != Prospero::ImageType::kColor2D || info.resources.layers != 1 ||
	    info.samples != 1 || info.resources.levels <= first_level || info.IsDepth() ||
	    info.HasStencil() || info.HasMetadata() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed ||
	    info.pixel_format == vk::Format::eUndefined) {
		return 0;
	}
	// The guest bytes of levels [first_level, levels), from the same layout the refresh detiles
	// (PS5 stores the chain smallest level first: the mip tail block, then coarser to finer).
	const auto transfer =
	    BuildTextureTransfer(info, 1, BindingType::Texture, TransferDirection::Upload);
	if (!transfer.valid) {
		return 0;
	}
	uint64_t begin = UINT64_MAX;
	uint64_t end   = 0;
	if (!transfer.tiles.empty()) {
		if (transfer.tiles.size() != transfer.regions.size()) {
			return 0;
		}
		for (size_t index = 0; index < transfer.tiles.size(); ++index) {
			if (transfer.regions[index].imageSubresource.mipLevel < first_level) {
				continue;
			}
			const auto& tile = transfer.tiles[index];
			begin            = std::min(begin, tile.tiled_offset);
			end              = std::max(end, tile.tiled_offset + tile.tiled_size);
		}
	} else {
		for (uint32_t level = first_level; level < info.resources.levels; ++level) {
			const auto& mip = transfer.layout.mips[level];
			begin           = std::min(begin, mip.offset);
			end             = std::max(end, mip.offset + mip.size);
		}
	}
	// Only a proper prefix: it keeps the first page (exact-backing lookups start there) and
	// leaves every non-resident byte after it.
	if (begin != 0 || end == 0 || end >= info.data.size) {
		return 0;
	}
	return end;
}

void TextureCache::BindSparseLevelsAfterSubmitted(Image& image, uint32_t first_level) {
	auto&          master   = m_scheduler.GetMasterSemaphore();
	const auto     current  = m_scheduler.CurrentTick();
	const uint64_t previous = current > 1 ? current - 1 : 0;
	if (previous == 0 || master.KnownGpuTick() >= previous) {
		m_graphics.BindSparseImageLevels(image.backing, first_level); // nothing in flight
		return;
	}
	// The GPU wait must never precede its signal (queueSubmission.h): hand every older tick to the
	// driver first, as BufferCache's side readbacks do (KYTY_CP_RECORDER ring, submission broker).
	m_scheduler.WaitRecorded(previous);
	const auto& progress = master.GetSubmissionProgress();
	if (progress != nullptr && progress->dispatched_tick.load(std::memory_order_acquire) < previous) {
		m_graphics.submission_queue.WaitPendingDependencies(progress.get(), previous);
		Common::LockGuard lock(m_graphics.queue_mutex);
		m_graphics.submission_queue.DrainThroughLocked(progress.get(), previous);
	}
	m_graphics.BindSparseImageLevels(image.backing, first_level, master.Handle(), previous);
}

void TextureCache::EnsureResidency(ImageId id, uint32_t first_level, bool sampling) {
	auto& image = m_slot_images[id];
	if (first_level >= image.resident_first) {
		return;
	}
	++m_lookup_side_effects;
	const auto prefix    = ResidentPrefixSize(image.info, first_level);
	const auto new_first = prefix != 0 ? first_level : 0u;
	// The registered range grows. For everything keyed on the page owner index (clean-page
	// proofs, lookups, binding identity caches) this is an unregister and a register.
	const bool registered = image.registered;
	const auto old_end    = image.live.End();
	if (registered) {
		UnregisterImage(id);
	}
	image.resident_first = new_first;
	NoteImageStateChange();
	image.live = new_first != 0 ? GuestRange {image.info.data.address, prefix} : image.info.data;
	NoteImageStateChange();
	if (image.backing.sparse) {
		// KYTY_TEXTURE_SPARSE_RESIDENCY: memory behind the newly resident levels; their refresh
		// follows (MarkResidencyDirty). Submitted ticks may still access these levels: Image::
		// GetBarriers transitions the whole chain, because sampled views span non-resident levels
		// and need them in the view's layout. Vulkan forbids such accesses while the binding
		// changes on another queue (Bug A), so the bind waits on the GPU for every tick before the
		// current recording, and the current one waits for the bind (CommandScheduler::Submit).
		BindSparseLevelsAfterSubmitted(image, new_first);
	}
	if (registered) {
		RegisterImage(id);
	}
	// The newly registered bytes were outside every overlap resolution so far: an image created
	// there meanwhile (the partially resident one was invisible to its lookup) now overlaps this
	// one unresolved. Diagnostics only: a GPU-modified one holds the newest bytes, which the
	// refresh below does not see (it reads guest memory).
	if (registered && image.live.End() > old_end) {
		for (const auto other_id: FindImagesInRegion(old_end, image.live.End() - old_end, false)) {
			const auto* other = m_slot_images.try_get(other_id);
			if (other_id == id || other == nullptr || !other->registered) {
				continue;
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidencyExtensionOverlaps);
			if (other->IsGpuModified()) {
				Profiler::CountFrameEvent(
				    Profiler::FrameEvent::TextureResidencyExtensionGpuOverlaps);
				if (++m_residency_overlap_logs <= 16) {
					LOGF("TextureCache: residency extension of 0x%016" PRIx64 " (levels from %u)"
					     " covers GPU-modified image 0x%016" PRIx64 "+0x%" PRIx64
					     "; its refresh reads guest memory there\n",
					     image.info.data.address, new_first, other->info.data.address,
					     other->info.data.size);
				}
			}
		}
	}
	// Newly resident levels hold undefined contents: refresh every resident level before use
	// (from the GPU-written buffer bytes when there are any, as for a new image).
	image.MarkResidencyDirty();
	image.residency_refresh = true;
	if (m_buffer_cache.HasGpuDirtyBytes(image.live.address, image.live.size)) {
		image.MarkBufferModified();
	}
	image.residency_poisoned = image.residency_poisoned && new_first != 0;
	Profiler::CountFrameEvent(sampling ? Profiler::FrameEvent::TextureResidencyExtensions
	                                   : Profiler::FrameEvent::TextureResidencyFullFallbacks);
}

static bool ResidentIdleKeepBindless() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_RESIDENT_IDLE_KEEP_BINDLESS");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return on;
}

void TextureCache::RetireIdlePartialImages() {
	// A partially resident image overlaps no neighbouring streamed texture, so the overlap
	// rule that used to retire unused ones never fires: its full native mip chain would stay
	// allocated until a same-slot replacement or memory pressure. Retire one unused for
	// KYTY_TEXTURE_RESIDENT_IDLE_FRAMES presented frames instead; a later use recreates it and
	// uploads only its resident levels. Once per frame, over partially resident images only.
	const auto frame = m_frame.load(std::memory_order_relaxed);
	if (m_resident_idle_frames == 0 || m_partial_images.empty() || frame == m_partial_scan_frame) {
		return;
	}
	m_partial_scan_frame = frame;
	const auto tick      = m_scheduler.CurrentTick();
	size_t     kept      = 0;
	for (size_t index = 0; index < m_partial_images.size(); ++index) {
		const auto id    = m_partial_images[index];
		auto*      image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered || image->FullyResident()) {
			continue;
		}
		// KYTY_RESIDENT_IDLE_KEEP_BINDLESS (default on): an image a bindless heap key holds is in use
		// although no binding touches it (bindless sampling never calls TouchImage); freeing it
		// repoints its slots to the placeholder and re-resolves every key each time.
		if (ResidentIdleKeepBindless() && image->bindless_pinned) {
			m_partial_images[kept++] = id;
			continue;
		}
		const bool idle =
		    frame - std::min(frame, image->frame_accessed_last) > m_resident_idle_frames &&
		    tick - std::min(tick, image->tick_accessed_last) > NumFramesBeforeRemoval &&
		    !image->binding.is_bound && !image->binding.is_target && !image->IsGpuModified();
		if (idle) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidentIdleFrees);
			FreeImage(id, HangTrace::ImageFreeReason::ResidentIdle);
			continue;
		}
		m_partial_images[kept++] = id;
	}
	m_partial_images.resize(kept);
}

void TextureCache::RequireFullResidency(ImageId id) {
	if (m_slot_images[id].FullyResident()) {
		return;
	}
	// A use other than sampling through a clamped view (render or storage target, copy,
	// clear, download, presentation): the whole chain is live from now on. Refresh it before
	// the caller records anything that reads or partially writes it.
	EnsureResidency(id, 0, false);
	RefreshImage(id);
}

void TextureCache::RestrictToResidentLevels(const Image& image, TextureTransfer& transfer) const {
	if (image.FullyResident()) {
		return;
	}
	const auto first = image.resident_first;
	const bool tiled = !transfer.tiles.empty();
	EXIT_IF(tiled && transfer.tiles.size() != transfer.regions.size());
	size_t kept = 0;
	for (size_t index = 0; index < transfer.regions.size(); ++index) {
		if (transfer.regions[index].imageSubresource.mipLevel < first) {
			continue;
		}
		transfer.regions[kept] = transfer.regions[index];
		if (tiled) {
			transfer.tiles[kept] = transfer.tiles[index];
		}
		kept++;
	}
	transfer.regions.resize(kept);
	if (tiled) {
		transfer.tiles.resize(kept);
		// The detiled scratch holds only the resident levels.
		uint64_t base = UINT64_MAX;
		for (const auto& tile: transfer.tiles) {
			base = std::min(base, tile.linear_offset);
		}
		for (size_t index = 0; index < kept; ++index) {
			transfer.tiles[index].linear_offset -= base;
			transfer.regions[index].bufferOffset -= base;
		}
	}
}

void TextureCache::PoisonNonResidentLevels(Image& image) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	// KYTY_TEXTURE_RESIDENT_MIPS=poison: fill the levels no view may sample with a marker
	// (magenta/green texels) so any read of a non-resident level shows on screen.
	auto transfer = BuildTextureTransfer(image, BindingType::Texture, TransferDirection::Upload);
	if (!transfer.valid) {
		return;
	}
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         bytes = 0;
	const auto element = std::max(image.info.bytes_per_block, 1u);
	const auto texel   = image.info.IsBlock() ? 4u : 1u;
	for (auto region: transfer.regions) {
		if (region.imageSubresource.mipLevel >= image.resident_first) {
			continue;
		}
		region.bufferOffset      = 0;
		region.bufferRowLength   = 0;
		region.bufferImageHeight = 0;
		const uint64_t size = static_cast<uint64_t>((region.imageExtent.width + texel - 1) / texel) *
		                      ((region.imageExtent.height + texel - 1) / texel) * element;
		bytes = std::max(bytes, size);
		regions.push_back(region);
	}
	if (regions.empty() || bytes == 0) {
		return;
	}
	const auto scratch = m_tiler.GetScratchBuffer(bytes);
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().fillBuffer(scratch.buffer, 0, Common::AlignUp(bytes, 4),
	                                          0xF81FF81Fu);
	image.Upload(regions, scratch.buffer, 0, scratch.size);
	image.residency_poisoned = true;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.live.address, image.live.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	// Registering an image that is already GPU-modified (EnsureResidency re-registers) makes its
	// range GPU-owned: log it as such, so the sticky GPU-touched pages (gpuTouchedPages.h) cover it.
	InvalidateCleanImageProofs(image.live.address, image.live.size,
	                           image.IsGpuModified() ? Coherence::Source::ImageGpuModified
	                                                 : Coherence::Source::ImageRegister);
	NoteStructureChange(image);
	ForEachPage(image.live.address, image.live.size, [this, id](uint64_t page) {
		// Counted before the image enters the page's owner list, so a lock-free zero count means
		// FindImagesInRegion finds nothing there (NoImagesOnPages); and before the image can
		// watch pages (TrackImage follows registration).
		m_image_page_counts[page].fetch_add(1, std::memory_order_seq_cst);
		m_image_page_table[page].push_back(id);
	});
	image.registered = true;
	NoteImageStateChange();
	image.lru_id     = m_lru_cache.Insert(id, m_gc_tick);
	image.lru_tick   = m_lru_cache.TickOf(image.lru_id);
	m_total_used_memory += image.AccountedSize();
	m_registered_image_memory += image.AccountedSize();
	// Registration precedes the first TrackImage, so the tracking mode never changes while
	// the image watches pages. Re-registration (a residency change) re-derives it.
	if (!image.IsTracked()) {
		if (m_partial_upload && ChunkTrackingEligible(image)) {
			image.EnableChunkTracking(m_chunk_shift);
		} else {
			image.chunks = {};
		}
	}
}

bool TextureCache::ChunkTrackingEligible(const Image& image) const {
	const auto& info = image.info;
	// Page-aligned ranges only: then every write fault on one of the image's pages overlaps its
	// bytes, and no same-page neighbour write can untrack a page without dirtying it.
	return info.pixel_format != vk::Format::eUndefined && image.backing.image != nullptr &&
	       !info.IsDepth() && !info.HasStencil() && !info.HasMetadata() && info.samples == 1 &&
	       info.metadata.compression == VideoOutCompression::Uncompressed &&
	       image.live.address % TRACKER_PAGE_SIZE == 0 &&
	       image.live.size % TRACKER_PAGE_SIZE == 0 &&
	       image.live.size > (uint64_t {1} << m_chunk_shift);
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	InvalidateCleanImageProofs(image.live.address, image.live.size,
	                           Coherence::Source::ImageUnregister);
	NoteStructureChange(image);
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.live.address, image.live.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.live.address, image.live.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
		// After UntrackImage above: the image no longer watches these pages.
		m_image_page_counts[page].fetch_sub(1);
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	// With VK_EXT_memory_budget the collector sets m_total_used_memory to the device-local usage,
	// which can be below the images' own bytes (allocations outside the device-local heaps when
	// VRAM runs short), so it saturates. m_registered_image_memory is the exact owned count.
	m_total_used_memory -= std::min(m_total_used_memory, accounted);
	if (accounted > m_registered_image_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_registered_image_memory -= accounted;
	image.registered = false;
	NoteImageStateChange();
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	// Its copies (SingleSampleDepth, SingleSampleStencil, SubExtentCopy) go with it.
	if (!m_single_sample_depths.empty() || !m_single_sample_stencils.empty() ||
	    !m_sub_extent_copies.empty()) [[unlikely]] {
		for (auto* copies: {&m_single_sample_depths, &m_single_sample_stencils}) {
			if (const auto copy = copies->find(id); copy != copies->end()) {
				const auto copy_id = copy->second.copy;
				copies->erase(copy);
				if (copy_id) {
					ReleaseUnregisteredImage(copy_id);
				}
			}
		}
		if (const auto copies = m_sub_extent_copies.find(id); copies != m_sub_extent_copies.end()) {
			const auto entries = std::move(copies->second);
			m_sub_extent_copies.erase(copies);
			for (const auto& entry: entries) {
				if (entry.copy) {
					ReleaseUnregisteredImage(entry.copy);
				}
			}
		}
		image = m_slot_images.try_get(id);
	}
	// KYTY_STENCIL_STORAGE_SHADOW: a depth image deleted before its shadow is copied back has no
	// stencil left to take it (Erlou crashed copying into one, slotVector.h:71). Its pending pairs
	// and its shadow image go with it, so a later image in the same slot never receives the copy.
	if (!m_stencil_shadows.empty() || !m_pending_stencil_shadows.empty()) [[unlikely]] {
		const auto dropped = std::erase_if(m_pending_stencil_shadows,
		                                   [id](const auto& pair) { return pair.second == id; });
		if (dropped != 0) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 32u) {
				LOGF("Stencil storage shadow: depth 0x%012" PRIx64
				     " deleted before the copy back, %zu pair(s) skipped\n",
				     image->info.data.address, static_cast<size_t>(dropped));
			}
		}
		if (const auto shadow = m_stencil_shadows.find(id); shadow != m_stencil_shadows.end()) {
			const auto shadow_id = shadow->second;
			m_stencil_shadows.erase(shadow);
			ReleaseUnregisteredImage(shadow_id);
			image = m_slot_images.try_get(id);
		}
	}
	if (HangTrace::Enabled()) {
		const auto tick = m_scheduler.CurrentTick();
		HangTrace::RecordImageFree(image->info.data.address, image->info.extent.width,
		                           image->info.extent.height, image->info.resources.levels,
		                           image->info.resources.layers,
		                           static_cast<uint32_t>(image->backing.format), image->info.data.size,
		                           tick - std::min(tick, image->tick_accessed_last));
		HangTrace::SetImageFreeReason(HangTrace::ImageFreeReason::Other);
	}
	// Stencil associations only ever point at depth images (AssociateStencil requires one, and
	// an image's pixel format never changes). The depth image lists them (Image::stencil_ids,
	// upstream f4692a450), so retiring one needs no full-cache scan.
	if (image->depth_id) {
		if (auto* owner = m_slot_images.try_get(image->depth_id); owner != nullptr) {
			std::erase(owner->stencil_ids, id);
		}
	} else if (image->info.IsDepth()) {
		// Detached first: deleting each association also unlinks it from its depth owner.
		for (const auto association: std::exchange(image->stencil_ids, {})) {
			const auto* associated = m_slot_images.try_get(association);
			if (associated != nullptr && associated->depth_id == id) {
				FreeImage(association, HangTrace::ImageFreeReason::DepthAssociation);
			}
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile && metadata->second.owner == id) {
			// A later binding may have reused this address for another metadata type, and only
			// the last image to acquire an HTile range releases its clear state (upstream
			// 2d84e36b4): a superseded image freed later keeps its successor's clear state.
			m_surface_metas.erase(metadata);
			NoteSurfaceMetaChange();
		}
	}
	// Bindless table (Senaxx): a pinned image's keys go back to pending, and no slot may keep a
	// view of the image once it is destroyed (IDXTRI's stale view guard): such slots are repointed
	// to the placeholder now, before the destruction below waits for the current tick.
	if (image->bindless_pinned) {
		NoteBindlessStateChange(*image);
		image->bindless_pinned = false;
		if (on_bindless_unregister) {
			on_bindless_unregister(id, m_freeing_replaced);
		}
	}
	if (on_image_views_dropped && !image->views.empty()) {
		(void)on_image_views_dropped(image->views);
	}
	UnregisterImage(id);
	if (m_scheduler.Active()) {
		uint64_t retiring_bytes = 0;
		if (m_pressure_gc_enabled) {
			retiring_bytes = m_graphics.NativeImageBytes(image->backing);
			m_pressure_retirement_bytes += retiring_bytes;
		}
		m_scheduler.DeferOperation([this, id, retiring_bytes] {
			m_slot_images.erase(id);
			NoteImageStateChange();
			EXIT_IF(retiring_bytes > m_pressure_retirement_bytes);
			m_pressure_retirement_bytes -= retiring_bytes;
		});
	} else {
		m_slot_images.erase(id);
		NoteImageStateChange();
	}
}

void TextureCache::NoteVramFree(const Image& image, HangTrace::ImageFreeReason reason) {
	const auto bytes = m_graphics.NativeImageBytes(image.backing);
	if (bytes == 0 || reason >= HangTrace::ImageFreeReason::Count) {
		return;
	}
	auto& entry = m_vram_frees[static_cast<size_t>(reason)];
	entry.first++;
	entry.second += bytes;
}

void TextureCache::FreeImage(ImageId id, HangTrace::ImageFreeReason reason) {
	HangTrace::SetImageFreeReason(reason);
	auto& image = m_slot_images[id];
	// Bindless-pinned images (a heap key samples them through set 1) are never collected.
	if (image.bindless_pinned && !m_bindless_evicting &&
	    (reason == HangTrace::ImageFreeReason::GarbageCollect ||
	                              reason == HangTrace::ImageFreeReason::PressureCollect ||
	                              reason == HangTrace::ImageFreeReason::ResidentIdle)) {
		return;
	}
	// KYTY_BINDLESS_FREE_LOG=<n> (diagnostic): the first n frees of bindless-pinned images.
	if (image.bindless_pinned) {
		static const uint32_t limit = [] {
			const char* value = std::getenv("KYTY_BINDLESS_FREE_LOG");
			return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
		}();
		static std::atomic<uint32_t> logged = 0;
		if (limit != 0 && logged.fetch_add(1, std::memory_order_relaxed) < limit) {
			LOGF("Bindless free: reason %u image %u 0x%" PRIx64 "+0x%" PRIx64 " %ux%u vk%u gpu=%d\n",
			     static_cast<uint32_t>(reason), id.index, image.info.data.address, image.info.data.size,
			     image.info.extent.width, image.info.extent.height,
			     static_cast<uint32_t>(image.backing.format), image.IsGpuModified() ? 1 : 0);
		}
	}
	// KYTY_BINDLESS_DROP_LOG=<n> (diagnostic): the first n frees of bindless-pinned images, with the
	// reason (HangTrace::ImageFreeReason) and the image.
	if (image.bindless_pinned && image.registered) {
		static const uint32_t drop_log = [] {
			const char* value = std::getenv("KYTY_BINDLESS_DROP_LOG");
			return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
		}();
		static std::atomic<uint32_t> dropped {0};
		if (drop_log != 0 && dropped.fetch_add(1, std::memory_order_relaxed) < drop_log) {
			LOGF("Bindless drop: image %u reason %u base 0x%" PRIx64 " %ux%u layers %u vk%u gpu_modified %d\n",
			     id.index, static_cast<uint32_t>(reason), image.info.data.address, image.info.extent.width,
			     image.info.extent.height, image.info.resources.layers, static_cast<uint32_t>(image.backing.format),
			     image.IsGpuModified() ? 1 : 0);
		}
	}
	if (VramStats::Enabled() && image.registered) {
		NoteVramFree(image, reason);
	}
	// The garbage collectors free a GPU-modified image that is safe to download only after
	// downloading it into guest memory.
	const bool collected = (reason == HangTrace::ImageFreeReason::GarbageCollect ||
	                        reason == HangTrace::ImageFreeReason::PressureCollect) &&
	                       SafeToDownload(image);
	if (image.IsGpuModified() && AliasBytesEnabled() && !collected &&
	    reason != HangTrace::ImageFreeReason::Unmap && image.registered && !image.depth_id) {
		// The bytes this image still owns exist nowhere else (an overlap-stale free at a layout
		// switch, a garbage collection): move them into the buffer first. An unmap discards them.
		(void)MaterializeOwnedBytes(image.info.data, id, {}, "free");
	} else if (image.IsGpuModified() && !collected && reason != HangTrace::ImageFreeReason::Unmap) {
		NoteDiscardedTarget(id, image); // KYTY_DISCARDED_TARGET_SKIP
	}
	if (image.IsGpuModified()) {
		CleanVerdict::Invalidate(image.live.address, image.live.size,
		                         Coherence::Source::ImageGpuClear);
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

// KYTY_DISCARDED_TARGET_SKIP=1 (default off; TheCruZ/KytyPS5-GTA 1b9454f7): a new render or depth
// target over memory where a GPU-written target was freed without its pixels reaching guest memory
// within the last four frames, not written by any GPU buffer and never uploaded, is not initialized
// from guest memory: those bytes are stale, and the target's first write replaces them. Games
// that scale their resolution every frame (RDR: ~10 targets recreated per frame, 70 MB of uploads)
// otherwise copy and detile every replacement.
static bool DiscardedTargetSkipEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DISCARDED_TARGET_SKIP");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

static bool IsTargetImage(const Image& image) noexcept {
	return image.usage.render_target || image.usage.depth_target;
}

void TextureCache::NoteDiscardedTarget(ImageId id, const Image& image) {
	// Images read back after every submission hold their pixels in guest memory.
	if (!DiscardedTargetSkipEnabled() || !IsTargetImage(image) || image.info.data.Empty() ||
	    m_download_images.contains(id)) {
		return;
	}
	m_discarded_targets[m_discarded_next] = {image.info.data.address,
	                                         image.info.data.address + image.info.data.size, Frame()};
	m_discarded_next                      = (m_discarded_next + 1) % DiscardedTargetCount;
}

bool TextureCache::WasDiscardedTarget(const Image& image) const {
	constexpr uint64_t MaxAgeFrames = 4;
	const auto         begin        = image.info.data.address;
	const auto         end          = begin + image.info.data.size;
	const auto         frame        = Frame();
	return std::any_of(m_discarded_targets.begin(), m_discarded_targets.end(), [&](const DiscardedTarget& target) {
		return target.end != 0 && target.begin <= begin && end <= target.end && target.frame + MaxAgeFrames >= frame;
	});
}

void TextureCache::TouchImage(Image& image) {
	if (HangTrace::CpWatch(image.info.data.address, image.info.data.size)) {
		HangTrace::CpEvent event;
		event.event   = "use-image";
		event.address = image.info.data.address;
		event.value   = static_cast<uint64_t>(image.backing.format);
		event.ref     = (uint64_t {image.info.extent.width} << 32u) | image.info.extent.height;
		event.aux     = (image.IsGpuModified() ? 1 : 0) | (image.IsBufferModified() ? 2 : 0);
		event.size    = image.info.data.size;
		HangTrace::RecordCp(event);
	}
	if (ImgSide::StatsEnabled()) [[unlikely]] {
		ImgSide::Count(ImgSide::C::Touch);
		if (image.frame_accessed_last == m_frame.load(std::memory_order_relaxed) && image.registered &&
		    m_lru_touch_skip && image.lru_tick >= m_gc_tick) {
			ImgSide::Count(ImgSide::C::TouchSame);
		}
	}
	image.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	if (!image.registered) {
		return;
	}
	const auto bump = [](std::atomic<uint64_t>& counter) {
		counter.store(counter.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
	};
	// KYTY_IMAGE_LRU_SKIP: Touch returns at once for an item whose tick is not older than
	// m_gc_tick, and image.lru_tick is that tick (textureCache.h).
	bool skip = m_lru_touch_skip && image.lru_tick >= m_gc_tick;
	if (m_lru_touch_skip_verify != 0) [[unlikely]] {
		if (!LruMirrorHolds(image, skip)) {
			skip = false; // today's path
		}
	}
	if (skip) {
		bump(m_lru_touch_totals.skips);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageLruTouchSkips);
		return;
	}
	bump(m_lru_touch_totals.touches);
	m_lru_cache.Touch(image.lru_id, m_gc_tick);
	image.lru_tick = m_lru_cache.TickOf(image.lru_id);
}

bool TextureCache::LruMirrorHolds(Image& image, bool skip) {
	const auto bump = [](std::atomic<uint64_t>& counter, uint64_t amount = 1) {
		counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
	};
	bump(m_lru_touch_totals.verify_checks);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ImageLruVerifyChecks);
	const bool linked = m_lru_cache.Linked(image.lru_id);
	const auto item   = linked ? m_lru_cache.TickOf(image.lru_id) : 0;
	if (linked && item == image.lru_tick) {
		return true;
	}
	bump(m_lru_touch_totals.verify_mismatches);
	Profiler::CountFrameEvent(Profiler::FrameEvent::ImageLruVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr,
		             "ImageLruVerify: image 0x%016" PRIx64 " size=0x%" PRIx64
		             " mirror tick %" PRIu64 ", LRU item %zu %s tick %" PRIu64 ", GC tick %" PRIu64
		             " (%s)\n",
		             image.info.data.address, image.info.data.size, image.lru_tick, image.lru_id,
		             linked ? "at" : "unlinked,", item, m_gc_tick,
		             skip ? "the mirror skipped the touch" : "touched");
	}
	if (m_lru_touch_skip_verify == 2) {
		EXIT("ImageLruVerify: an image's LRU mirror differs from its LRU item\n");
	}
	// Repair after the touch (TouchImage copies the item's tick back).
	return false;
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.ChunkTracked()) {
		TrackChunkImage(image);
		return;
	}
	if (!image.kept_holes.empty()) {
		// KYTY_IMAGE_GPU_KEEP: watch the pages kept CPU writes released again. Their bytes stay
		// dirty until the refresh; a write from here on faults and is recorded before it lands.
		for (const auto& [begin, end]: image.kept_holes) {
			m_page_manager.UpdatePageWatchers<true>(begin, end - begin);
		}
		image.kept_holes.clear();
	}
	const auto image_begin = image.live.address;
	const auto image_end   = image.live.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		NoteImageStateChange();
		image.track_addr_end = image_end;
		NoteImageStateChange();
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.live.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.live.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	NoteImageStateChange();
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.live.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	NoteImageStateChange();
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::TrackChunkImage(Image& image) {
	if (!image.IsTracked()) {
		// Nothing is watched: the untracked marks describe holes of a previous tracked period.
		std::fill(image.chunks.untracked.begin(), image.chunks.untracked.end(), 0);
		image.chunks.untracked_count = 0;
		NoteImageStateChange();
		image.track_addr     = image.live.address;
		NoteImageStateChange();
		image.track_addr_end = image.live.End();
		NoteImageStateChange();
		m_page_manager.UpdatePageWatchers<true>(image.live.address, image.live.size);
		return;
	}
	// Watch again every chunk a CPU write released. Their dirty marks stay until a refresh:
	// a write after this point faults and is recorded before it reaches memory. (Runs on every
	// bind; nothing to scan while all chunks are watched.)
	if (image.chunks.untracked_count != 0) {
		(void)UpdateChunkWatchers<true>(image, 0, image.chunks.count);
	}
}

template <bool track>
uint32_t TextureCache::UpdateChunkWatchers(Image& image, uint32_t first, uint32_t last) {
	auto&          chunks     = image.chunks;
	if (track ? chunks.untracked_count == 0 : chunks.untracked_count == chunks.count) {
		return 0;
	}
	const uint64_t chunk_size = uint64_t {1} << chunks.shift;
	const uint64_t begin      = image.live.address; // page aligned (ChunkTrackingEligible)
	const uint64_t end        = image.live.End();
	uint32_t       changed    = 0;
	uint32_t       run_first  = UINT32_MAX;
	const auto     flush      = [&](uint32_t run_last) {
		if (run_first == UINT32_MAX) {
			return;
		}
		const auto run_begin = std::max(begin, chunks.base + uint64_t {run_first} * chunk_size);
		const auto run_end   = std::min(end, chunks.base + uint64_t {run_last} * chunk_size);
		if (run_begin < run_end) {
			m_page_manager.UpdatePageWatchers<track>(run_begin, run_end - run_begin);
		}
		run_first = UINT32_MAX;
	};
	for (uint32_t index = first; index < last; index++) {
		// track: re-watch released chunks; untrack: release still-watched chunks.
		if (Image::ChunkBit(chunks.untracked, index) == track) {
			if constexpr (track) {
				Image::ClearChunkBit(chunks.untracked, index);
			} else {
				Image::SetChunkBit(chunks.untracked, index);
			}
			if (run_first == UINT32_MAX) {
				run_first = index;
			}
			changed++;
		} else {
			flush(index);
		}
	}
	flush(last);
	if constexpr (track) {
		chunks.untracked_count -= changed;
		NoteImageStateChange();
	} else {
		chunks.untracked_count += changed;
		NoteImageStateChange();
	}
	return changed;
}

void TextureCache::InvalidateChunks(Image& image, uint64_t address, uint64_t size) {
	const auto begin = std::max(Common::AlignDown(address, TRACKER_PAGE_SIZE), image.live.address);
	const auto end   = std::min(Common::AlignUp(address + size, TRACKER_PAGE_SIZE), image.live.End());
	if (begin >= end) {
		return;
	}
	auto& chunks = image.chunks;
	// Whole rewrites (see ChunkState::full_streak) release everything on the first write.
	constexpr uint32_t WholeRewriteStreak = 2;
	const bool         whole = chunks.full_streak >= WholeRewriteStreak;
	const uint32_t     first = whole ? 0u : static_cast<uint32_t>((begin - chunks.base) >> chunks.shift);
	const uint32_t     last  = whole ? chunks.count
	                                 : static_cast<uint32_t>(((end - 1 - chunks.base) >> chunks.shift) + 1);
	chunks.whole_released |= whole;
	uint32_t newly = 0;
	for (uint32_t index = first; index < last; index++) {
		newly += image.MarkChunkDirty(index) ? 1u : 0u;
	}
	if (newly != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureChunkInvalidations, newly);
	}
	// Release only the written chunks: writes elsewhere in the image keep faulting and are
	// recorded chunk by chunk, so a refresh knows every byte that may have changed.
	if (image.IsTracked()) {
		(void)UpdateChunkWatchers<false>(image, first, last);
	}
	image.NoteChunkWrite(address, size);
}

// KYTY_IMAGE_GPU_KEEP. On the console a CPU write or a remap replaces only those bytes of an image the
// GPU rendered; the whole-image path made it CPU-dirty everywhere (a partial unmap freed it), and its
// next use uploaded every byte from guest memory, which does not hold what the GPU rendered. A game
// that streams through a pool it also renders into (Demon's Souls: an 80-layer shadow-map array whose
// free layers its texture streamer remaps and fills from the CPU, chenxiao07/KytyPS5 e7bf421a1)
// lost the rendered layers and uploaded the whole array again. Kept images refresh only the
// subresources over the replaced bytes (UploadKeptRanges).
namespace {
// A kept CPU write releases this much around the fault: a streamed layer faults a few times,
// not once per page.
constexpr uint64_t KeptWriteGranule = uint64_t {64} << 10;
} // namespace

bool TextureCache::KeepEligible(const Image& image) const {
	const auto& info = image.info;
	if (!m_gpu_keep || !image.registered || image.depth_id || image.ChunkTracked() ||
	    !image.IsGpuModified() || !image.OwnsAllBytes() || image.IsBufferModified() ||
	    image.IsMaybeCpuDirty() || (image.IsDefinitelyCpuDirty() && image.kept_dirty.empty()) ||
	    !image.FullyResident() || image.backing.image == nullptr || info.samples != 1 ||
	    image.backing.samples != 1 || info.IsVolume() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed || !image.IsTracked() ||
	    image.track_addr != image.live.address || image.track_addr_end != image.live.End()) {
		return false;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		// Whole layers of one level, without a D16 promotion (UploadKeptDepthLayers).
		const auto layers = info.resources.layers;
		return layers >= 2 && info.resources.levels == 1 && info.data.size % layers == 0 &&
		       DepthAspectTransferBytes(info.pixel_format) == info.bytes_per_block;
	}
	return info.IsTiled() && info.GetColorTransform() == ColorTransform::None;
}

bool TextureCache::KeepGpuBytes(Image& image, uint64_t address, uint64_t size, bool unmap) {
	if (!KeepEligible(image)) {
		return false;
	}
	const auto live  = image.live;
	auto       begin = std::max(address, live.address);
	auto       end   = std::min(address + size, live.End());
	if (!unmap) {
		const auto span = KeptRanges::WriteSpan(address, size, live.address, live.End(),
		                                        KeptWriteGranule, TRACKER_PAGE_SIZE);
		begin = span.first;
		end   = span.second;
		const auto& info = image.info;
		if (UploadBinding(image) == BindingType::DepthTarget && begin < end) {
			// Depth arrays refresh whole layers: the granule stops at the written layers (thin
			// layers are not refreshed for a neighbour's write), or at the written pages where a
			// layer is not page-sized.
			const auto slice = info.data.size / info.resources.layers;
			const auto base  = info.data.address;
			if (slice % TRACKER_PAGE_SIZE == 0 && address < info.data.End() && address + size > base) {
				const auto first = (std::max(address, base) - base) / slice;
				const auto last  = (std::min(address + size, info.data.End()) - 1 - base) / slice;
				begin = std::max(begin, base + first * slice);
				end   = std::min(end, base + (last + 1) * slice);
			} else {
				begin = std::max(begin, Common::AlignDown(address, TRACKER_PAGE_SIZE));
				end   = std::min(end, Common::AlignUp(address + size, TRACKER_PAGE_SIZE));
			}
		}
	}
	if (begin >= end) {
		return false;
	}
	// Bytes the image keeps once this span is replaced. (A CPU rewriting nearly all of a render
	// target reuses its memory: one whole refresh instead of a fault per granule.)
	auto       covered = image.kept_dirty;
	(void)KeptRanges::Add(covered, std::max(begin, live.address), std::min(end, live.End()));
	if (KeptRanges::Bytes(covered) >= live.size - live.size / 4) {
		return false;
	}
	if (!image.NoteKeptCpuWrite(begin, end)) {
		return false;
	}
	if (!unmap) {
		// Stop watching the written pages (the faulting write must be able to complete); every
		// other page stays watched, so later writes elsewhere are recorded too.
		auto released = image.kept_holes;
		(void)KeptRanges::Add(released, begin, end);
		auto at = begin;
		for (const auto& [hole_begin, hole_end]: image.kept_holes) {
			if (hole_end <= begin || hole_begin >= end) {
				continue;
			}
			if (at < hole_begin) {
				m_page_manager.UpdatePageWatchers<false>(at, hole_begin - at);
			}
			at = std::max(at, hole_end);
		}
		if (at < end) {
			m_page_manager.UpdatePageWatchers<false>(at, end - at);
		}
		image.kept_holes = std::move(released);
	}
	// An unmap keeps the pages watched: a remap gives the watched pages their protection back
	// (memory.cpp ResyncGpuRangeProtection), so the new contents' CPU writes fault as before.
	Profiler::CountFrameEvent(unmap ? Profiler::FrameEvent::TextureKeptUnmaps
	                                : Profiler::FrameEvent::TextureKeptWrites);
	return true;
}

TextureCache::PartialUploadResult TextureCache::UploadKeptRanges(Image& image) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	PartialUploadResult result;
	const auto          base = image.live.address;
	if (!image.HasKeptDirty() || !image.IsGpuModified() || !image.OwnsAllBytes() ||
	    image.IsBufferModified() || image.depth_id || !image.FullyResident()) {
		return result;
	}
	if (UploadBinding(image) == BindingType::DepthTarget) {
		(void)UploadKeptDepthLayers(image, result);
		return result;
	}
	const auto binding  = UploadBinding(image);
	const auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
	if (!transfer.valid || transfer.color_transform != ColorTransform::None || transfer.tiles.empty() ||
	    transfer.tiles.size() != transfer.regions.size()) {
		return result;
	}
	// Whole subresources (tiles) over a replaced byte; their other bytes come from guest memory
	// too, every other subresource keeps the GPU's contents.
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         linear_total = 0;
	for (size_t index = 0; index < transfer.tiles.size(); index++) {
		auto tile   = transfer.tiles[index];
		auto region = transfer.regions[index];
		if (tile.tiled_size == 0 ||
		    !KeptRanges::Intersects(image.kept_dirty, base + tile.tiled_offset,
		                            base + tile.tiled_offset + tile.tiled_size)) {
			continue;
		}
		linear_total        = Common::AlignUp(linear_total, 256);
		tile.linear_offset  = linear_total;
		region.bufferOffset = linear_total;
		linear_total += tile.linear_size;
		tiles.push_back(tile);
		regions.push_back(region);
	}
	result.dirty_bytes = KeptRanges::Bytes(image.kept_dirty);
	if (tiles.empty()) {
		// Only bytes no texel is decoded from (gaps between levels) were replaced.
		result.done = true;
		return result;
	}
	// The selected tiled ranges, packed, from guest memory (an unmapped part reads as zeros, as the
	// console's fresh mapping would hold whatever the CPU writes later). Synchronous: this path is
	// rare and must not depend on the staging copier's mapping checks.
	struct SourceRange {
		uint64_t offset;
		uint64_t size;
		uint64_t packed;
	};
	std::vector<SourceRange> ranges;
	uint64_t                 packed_total = 0;
	uint64_t                 tiled_bytes  = 0;
	for (auto& tile: tiles) {
		const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const SourceRange& r) {
			return r.offset == tile.tiled_offset && r.size == tile.tiled_size;
		});
		if (found != ranges.end()) {
			tile.tiled_offset = found->packed;
			continue;
		}
		packed_total = Common::AlignUp(packed_total, 256);
		ranges.push_back({tile.tiled_offset, tile.tiled_size, packed_total});
		tile.tiled_offset = packed_total;
		packed_total += tile.tiled_size;
		tiled_bytes += tile.tiled_size;
	}
	auto& staging         = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Upload);
	auto [mapped, offset] = staging.Map(packed_total, 256);
	if (mapped == nullptr) {
		return result;
	}
	for (const auto& range: ranges) {
		if (!LibKernel::Memory::TryReadBacking(base + range.offset, mapped + range.packed, range.size) &&
		    !LibKernel::Memory::TryReadSparseBacking(base + range.offset, mapped + range.packed,
		                                             range.size)) {
			std::memset(mapped + range.packed, 0, range.size);
		}
	}
	staging.Commit();
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureUploadBytesStaging, packed_total);
	if (!m_tiler.DetileToImage(image, staging.Handle(), offset, packed_total, linear_total, tiles,
	                           regions)) {
		const auto linear = m_tiler.Detile(staging.Handle(), offset, packed_total, linear_total, tiles);
		for (auto& region: regions) {
			region.bufferOffset += linear.offset;
		}
		image.Upload(regions, linear.buffer, linear.offset, linear.size);
	}
	result.done  = true;
	result.bytes = tiled_bytes;
	return result;
}

bool TextureCache::UploadKeptDepthLayers(Image& image, PartialUploadResult& result) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	const auto& info   = image.info;
	const auto  layers = info.resources.layers;
	if (info.samples != 1 || image.backing.samples != 1 || layers < 2 || info.resources.levels != 1 ||
	    info.data.size % layers != 0 || DepthAspectTransferBytes(info.pixel_format) != info.bytes_per_block) {
		return false;
	}
	const uint64_t slice  = info.data.size / layers;
	const uint64_t base   = info.data.address;
	const auto     picked = KeptRanges::DirtyLayers(image.kept_dirty, base, slice, layers);
	result.dirty_bytes    = KeptRanges::Bytes(image.kept_dirty);
	if (picked.empty()) {
		result.done = true;
		return true;
	}
	const uint64_t total  = slice * picked.size();
	auto&          staging = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Upload);
	auto [mapped, offset]  = staging.Map(total, 256);
	if (mapped == nullptr) {
		return false;
	}
	for (size_t index = 0; index < picked.size(); index++) {
		const auto source = base + slice * picked[index];
		auto*      target = mapped + slice * index;
		if (!LibKernel::Memory::TryReadBacking(source, target, slice) &&
		    !LibKernel::Memory::TryReadSparseBacking(source, target, slice)) {
			std::memset(target, 0, slice);
		}
	}
	staging.Commit();
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureUploadBytesStaging, total);
	const auto                       all_copies = BuildDepthCopies(info, slice, vk::ImageAspectFlagBits::eDepth);
	std::vector<vk::BufferImageCopy> copies;
	copies.reserve(picked.size());
	for (size_t index = 0; index < picked.size(); index++) {
		auto copy         = all_copies[picked[index]];
		copy.bufferOffset = slice * index;
		copies.push_back(copy);
	}
	TileManager::Result linear {staging.Handle(), offset, total};
	if (info.IsTiled()) {
		const auto       all_tiles = BuildDepthTiles(info);
		std::vector<GpuTileInfo> tiles;
		tiles.reserve(picked.size());
		for (size_t index = 0; index < picked.size(); index++) {
			auto tile          = all_tiles[picked[index]];
			tile.tiled_offset  = slice * index;
			tile.linear_offset = slice * index;
			tiles.push_back(tile);
		}
		linear = m_tiler.Detile(staging.Handle(), offset, total, total, tiles);
	}
	for (auto& copy: copies) {
		copy.bufferOffset += linear.offset;
	}
	image.Upload(copies, linear.buffer, linear.offset, linear.size);
	result.done  = true;
	result.bytes = total;
	return true;
}

// KYTY_IMAGE_PARTIAL_GPU_WRITE (default off). A GPU buffer write over part of an image used to make
// its next use rebuild all of it from the buffer: Beyond Good & Evil writes single layers of a
// 4096x4096 layered depth/stencil target (a 256 MiB stencil plane rebuilt per write, Jetsku/KytyPS5
// issue #82), Astro Bot's compute passes write 64 KiB of 1080p targets that were rebuilt whole.
// InvalidateMemoryFromGPU records the written ranges while the image's other native contents are
// known to equal what the rebuild would produce there (GpuWriteTrackEligible, foreign_since_sync,
// the content serial); the refresh then rebuilds only the subresources, or the runs of block rows
// of 2D tiled subresources, over those ranges, moving owned bytes through the buffer for those
// ranges only. KYTY_IMAGE_PARTIAL_GPU_WRITE=verify also rebuilds whole afterwards and compares.
namespace {
std::atomic<uint64_t> g_pgw_verify_checks {0};
std::atomic<uint64_t> g_pgw_verify_mismatches {0};
} // namespace

void TextureCache::MaterializeForRebuild(ImageId id, RefreshIntent intent) {
	auto& image = m_slot_images[id];
	if (!AliasBytesEnabled() || image.depth_id) {
		return;
	}
	// The rebuild reads buffer (or guest) bytes over the whole image. Bytes it still owns itself (a
	// bounded buffer write took the others) reach the buffer first; a read also takes the bytes
	// other images own there (a write binding usually overwrites them).
	if (image.IsBufferModified() && !image.IsDefinitelyCpuDirty() && image.OwnedSet() != nullptr &&
	    !image.OwnsNoBytes()) {
		(void)MaterializeOwnedBytes(image.live, id, {}, "rebuild-own");
	}
	if (intent == RefreshIntent::Read) {
		(void)MaterializeOwnedBytes(image.live, {}, id, "rebuild");
	}
}

bool TextureCache::GpuWritePartialCandidate(const Image& image) {
	if (m_partial_gpu_write == 0 || !image.IsBufferModified()) {
		return false;
	}
	PrintPartialGpuWriteSummary();
	m_pgw_totals.refreshes++;
	if (!image.gpu_write.active || image.gpu_write.ranges.empty()) {
		m_pgw_totals.untracked++;
		return false;
	}
	const bool cpu     = image.IsCpuDirty() || image.HasKeptDirty();
	const bool serial  = !image.depth_id && image.ContentSerial() != image.gpu_write.serial;
	const bool foreign = image.foreign_since_sync;
	if (cpu || serial || foreign || !image.FullyResident() || image.info.samples != 1 ||
	    image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		m_pgw_totals.stale++;
		m_pgw_totals.stale_cpu += cpu ? 1u : 0u;
		m_pgw_totals.stale_serial += serial ? 1u : 0u;
		m_pgw_totals.stale_foreign += foreign ? 1u : 0u;
		return false;
	}
	return true;
}

void TextureCache::NoteRefreshSync(ImageId id, bool gpu_partial) {
	auto& image = m_slot_images[id];
	if (gpu_partial) {
		m_pgw_totals.partial++;
	}
	if (image.depth_id) {
		image.foreign_since_sync = false;
		// The upload wrote the depth image's stencil aspect, also behind its other records.
		if (const auto* depth = m_slot_images.try_get(image.depth_id); depth != nullptr) {
			for (const auto record: depth->stencil_ids) {
				if (record != id) {
					if (auto* other = m_slot_images.try_get(record); other != nullptr) {
						other->foreign_since_sync = true;
					}
				}
			}
		}
		return;
	}
	// Bytes other images still own (a write-intent refresh does not take them) may hold other
	// contents than this image's native ones.
	image.foreign_since_sync = AliasBytesEnabled() && OtherImagesOwnBytes(image.live, id);
}

TextureCache::PartialUploadResult TextureCache::UploadGpuWrittenRanges(ImageId id, RefreshIntent intent) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	PartialUploadResult result;
	auto&               image       = m_slot_images[id];
	const auto          live        = image.live;
	const auto          base        = image.info.data.address;
	const auto&         written     = image.gpu_write.ranges;
	auto&               destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto          binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	result.dirty_bytes              = KeptRanges::Bytes(written);

	// Selection: every subresource (tile) whose tiled bytes intersect a written range; for 2D
	// tiles outside the mip tail only the runs of block rows that do (see TryPartialUpload).
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	KeptRanges::List                 refreshed;
	uint64_t                         linear_total = 0;
	uint64_t                         tiled_bytes  = 0;
	const auto add = [&](GpuTileInfo tile, vk::BufferImageCopy region, uint64_t linear_size) {
		linear_total        = Common::AlignUp(linear_total, 256);
		tile.linear_offset  = linear_total;
		tile.linear_size    = linear_size;
		region.bufferOffset = linear_total;
		linear_total += linear_size;
		tiled_bytes += tile.tiled_size;
		(void)KeptRanges::Add(refreshed, base + tile.tiled_offset,
		                      base + tile.tiled_offset + tile.tiled_size);
		tiles.push_back(tile);
		regions.push_back(region);
	};
	const auto select = [&](const std::vector<GpuTileInfo>&         all_tiles,
	                        const std::vector<vk::BufferImageCopy>& all_regions, uint32_t texel_height) {
		for (size_t index = 0; index < all_tiles.size(); index++) {
			const auto& tile   = all_tiles[index];
			const auto& region = all_regions[index];
			const auto  begin  = base + tile.tiled_offset;
			if (tile.tiled_size == 0 || !KeptRanges::Intersects(written, begin, begin + tile.tiled_size)) {
				continue;
			}
			TileBlockLayout block {};
			const bool      family_bands = tile.family != TileBlockFamily::Standard4KB3D &&
			                          tile.family != TileBlockFamily::Standard64KB3D &&
			                          tile.family != TileBlockFamily::Prt64KB3D;
			const bool bands = family_bands && !tile.tail && tile.depth == 1 &&
			                   region.imageOffset.y == 0 && region.imageExtent.depth == 1 &&
			                   region.imageSubresource.layerCount == 1 &&
			                   TileGetBlockLayout(tile.family, tile.bytes_per_element, block) &&
			                   block.block_depth == 1 && block.block_width != 0 &&
			                   block.block_height != 0 && block.block_size != 0;
			const uint32_t tiled_width = tile.tiled_width != 0 ? tile.tiled_width : tile.pitch;
			const uint64_t row_bytes =
			    bands ? static_cast<uint64_t>((tiled_width + block.block_width - 1) / block.block_width) *
			                block.block_size
			          : 0;
			const uint32_t rows = bands ? (tile.height + block.block_height - 1) / block.block_height : 0;
			if (!bands || row_bytes == 0 || rows <= 1 || rows * row_bytes > tile.tiled_size) {
				add(tile, region, tile.linear_size);
				continue;
			}
			const uint64_t pitch_bytes = static_cast<uint64_t>(tile.pitch) * tile.bytes_per_element;
			for (const auto& [row, end_row]: GpuWriteRanges::DirtyRowRuns(written, begin, row_bytes, rows)) {
				const uint32_t y0   = row * block.block_height;
				const uint32_t y1   = std::min(end_row * block.block_height, tile.height);
				GpuTileInfo    band = tile;
				band.tiled_offset   = tile.tiled_offset + row * row_bytes;
				band.tiled_size     = (end_row - row) * row_bytes;
				band.height         = y1 - y0;
				band.tiled_height   = (end_row - row) * block.block_height;
				auto           band_region = region;
				const uint32_t texel_y0    = y0 * texel_height;
				const uint32_t texel_y1    = std::min(y1 * texel_height, region.imageExtent.height);
				band_region.imageOffset.y      = static_cast<int32_t>(texel_y0);
				band_region.imageExtent.height = texel_y1 - texel_y0;
				add(band, band_region, static_cast<uint64_t>(band.height) * pitch_bytes);
			}
		}
	};

	bool linear_depth = false; // copies read the source directly (untiled depth/stencil)
	if (binding != BindingType::DepthTarget) {
		const auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid || transfer.color_transform != ColorTransform::None ||
		    transfer.tiles.empty() || transfer.tiles.size() != transfer.regions.size()) {
			m_pgw_totals.fallback_layout++;
			return result;
		}
		select(transfer.tiles, transfer.regions,
		       std::max(transfer.layout.surface.texture.texel_height, 1u));
	} else {
		const auto info = GpuWriteDepthInfo(image);
		if (!info.has_value()) {
			m_pgw_totals.fallback_layout++;
			return result;
		}
		const auto layers = info->resources.layers;
		const auto slice  = info->data.size / layers;
		auto       copies = BuildDepthCopies(*info, slice,
		                                     image.depth_id ? vk::ImageAspectFlagBits::eStencil
		                                                    : vk::ImageAspectFlagBits::eDepth);
		if (image.depth_id) {
			for (auto& copy: copies) {
				copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
			}
		}
		if (info->IsTiled()) {
			select(BuildDepthTiles(*info), copies, 1);
		} else {
			linear_depth = true;
			for (const auto layer: KeptRanges::DirtyLayers(written, base, slice, layers)) {
				regions.push_back(copies[layer]);
				tiled_bytes += slice;
				(void)KeptRanges::Add(refreshed, base + slice * layer, base + slice * (layer + 1));
			}
		}
	}
	if (regions.empty()) {
		// The writes touched only bytes no texel is decoded from (gaps between levels).
		result.done = true;
		return result;
	}
	if (tiled_bytes >= live.size - live.size / 8) {
		// Nearly all of it: the whole rebuild needs fewer dispatches for the same bytes.
		m_pgw_totals.fallback_most++;
		return result;
	}
	if (!image.depth_id && AliasBytesEnabled()) {
		// Bytes outside the rebuilt parts that another image owns would come from it in the whole
		// rebuild (a read takes them, a write leaves the buffer's older bytes there).
		if (OtherImagesOwnBytes(live, id)) {
			for (const auto& [begin, end]: GpuWriteRanges::Complement(refreshed, live.address, live.End())) {
				if (OtherImagesOwnBytes({begin, end - begin}, id)) {
					m_pgw_totals.fallback_others++;
					return result;
				}
			}
		}
		// MaterializeForRebuild for the rebuilt parts only.
		const bool own = image.IsBufferModified() && !image.IsDefinitelyCpuDirty() &&
		                 image.OwnedSet() != nullptr && !image.OwnsNoBytes();
		for (const auto& [begin, end]: refreshed) {
			if (own && image.OwnsBytesIn(begin, end - begin)) {
				(void)MaterializeOwnedBytes({begin, end - begin}, id, {}, "partial-own");
			}
			if (intent == RefreshIntent::Read) {
				(void)MaterializeOwnedBytes({begin, end - begin}, {}, id, "partial");
			}
		}
	}

	const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(live.address, live.size);
	if (source == nullptr) {
		EXIT("TextureCache: failed to obtain image upload source\n");
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureUploadBytesBuffer, tiled_bytes);
	if (linear_depth) {
		for (auto& region: regions) {
			region.bufferOffset += source_offset;
		}
		destination.Upload(regions, source->Handle(), source_offset, source->Size() - source_offset);
	} else if (binding == BindingType::DepthTarget ||
	           !m_tiler.DetileToImage(destination, source->Handle(), source_offset, live.size,
	                                  linear_total, tiles, regions)) {
		const auto linear =
		    m_tiler.Detile(source->Handle(), source_offset, live.size, linear_total, tiles);
		for (auto& region: regions) {
			region.bufferOffset += linear.offset;
		}
		destination.Upload(regions, linear.buffer, linear.offset, linear.size);
	}
	result.done  = true;
	result.bytes = tiled_bytes;
	m_pgw_totals.partial_bytes += tiled_bytes;
	m_pgw_totals.whole_bytes += live.size;
	if (m_partial_gpu_write_verify) {
		VerifyGpuWrittenRanges(id, intent);
	}
	return result;
}

std::optional<ImageInfo> TextureCache::GpuWriteDepthInfo(const Image& image) const {
	// As UploadImage's depth path: the stencil plane has its own row pitch and shares the native
	// image with the depth plane.
	const Image* depth = image.depth_id ? m_slot_images.try_get(image.depth_id) : &image;
	if (depth == nullptr) {
		return std::nullopt;
	}
	const auto& destination = *depth;
	auto        info        = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) {
			info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
		}
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (info.samples != 1 || destination.backing.samples != 1 || info.resources.layers == 0 ||
	    info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block ||
	    transfer_bytes != info.bytes_per_block) {
		return std::nullopt;
	}
	return info;
}

void TextureCache::VerifyGpuWrittenRanges(ImageId id, RefreshIntent intent) {
	auto&      image       = m_slot_images[id];
	auto&      destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding     = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	// Every texel the whole rebuild writes, in its linear layout.
	std::vector<vk::BufferImageCopy> copies;
	uint64_t                         size = 0;
	if (binding != BindingType::DepthTarget) {
		const auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		copies              = transfer.regions;
		size                = transfer.LinearSize();
	} else if (const auto info = GpuWriteDepthInfo(image); info.has_value()) {
		const auto slice = info->data.size / info->resources.layers;
		copies           = BuildDepthCopies(*info, slice,
		                                    image.depth_id ? vk::ImageAspectFlagBits::eStencil
		                                                   : vk::ImageAspectFlagBits::eDepth);
		if (image.depth_id) {
			for (auto& copy: copies) {
				copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
			}
		}
		size = info->data.size;
	}
	if (copies.empty() || size == 0) {
		return;
	}
	const auto readback = [&]() {
		const auto capacity = Common::AlignUp(size, uint64_t {4});
		auto buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                       vk::BufferUsageFlagBits::eTransferDst, capacity);
		buffer->Fill(0, capacity, 0);
		destination.Download(copies, buffer->Handle(), 0, size);
		return buffer;
	};
	auto partial = readback();
	// The whole rebuild, exactly as InitializeImage does it without the switch.
	MaterializeForRebuild(id, intent);
	const auto [source, source_offset] =
	    m_buffer_cache.ObtainBufferForImage(image.live.address, image.live.size);
	if (source == nullptr) {
		EXIT("TextureCache: failed to obtain image upload source\n");
	}
	UploadImage(image, *source, source_offset);
	auto whole = readback();
	vk::MemoryBarrier2 to_host {};
	to_host.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	to_host.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &to_host;
	m_scheduler.Current().HelperSink().pipelineBarrier2(dependency);
	const auto address = image.info.data.address;
	const auto width   = image.info.extent.width;
	const auto height  = image.info.extent.height;
	const auto format  = static_cast<uint32_t>(destination.backing.format);
	const bool stencil = static_cast<bool>(image.depth_id);
	const auto ranges  = image.gpu_write.ranges.size();
	m_scheduler.DeferOperation([partial, whole, size, address, width, height, format, stencil, ranges] {
		partial->Invalidate(0, size);
		whole->Invalidate(0, size);
		g_pgw_verify_checks.fetch_add(1, std::memory_order_relaxed);
		const auto* a = partial->Mapped().data();
		const auto* b = whole->Mapped().data();
		if (std::memcmp(a, b, size) == 0) {
			return;
		}
		uint64_t first = 0;
		while (first < size && a[first] == b[first]) {
			first++;
		}
		uint64_t differing = 0;
		for (uint64_t i = first; i < size; i++) {
			differing += a[i] != b[i] ? 1u : 0u;
		}
		const auto count = g_pgw_verify_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count <= 32) {
			std::printf("Image partial GPU-write verify: MISMATCH image 0x%016" PRIx64
			            " %ux%u format %u%s, %zu written range(s): %" PRIu64 " of %" PRIu64
			            " linear bytes differ, first at %" PRIu64 "\n",
			            address, width, height, format, stencil ? " (stencil plane)" : "", ranges,
			            differing, size, first);
			std::fflush(stdout);
		}
	});
}

bool TextureCache::DebugDumpImage(uint64_t address, uint32_t want_width, uint32_t want_height,
                                  const std::string& prefix) {
	std::scoped_lock lock {m_lock};
	ImageId              found {};
	std::vector<ImageId> exact;
	std::string          listing;
	FindImagesInRegion(address, 1, false).ForEach([&](ImageId id) {
		const auto& image = m_slot_images[id];
		listing += fmt::format(" [id {} base 0x{:x} {}x{} vk{}{}]", id.index, image.info.data.address,
		                       image.info.extent.width, image.info.extent.height,
		                       static_cast<uint32_t>(image.backing.format), image.depth_id ? " depth" : "");
		if (image.info.data.address != address || image.depth_id) return;
		if (!found) found = id;
		if (image.info.extent.width == want_width && image.info.extent.height == want_height) exact.push_back(id);
	});
	LOGF("DebugDumpImage 0x%" PRIx64 " want %ux%u:%s\n", address, want_width, want_height, listing.c_str());
	if (exact.empty() && found) exact.push_back(found);
	if (exact.empty()) {
		return false;
	}
	bool dumped = false;
	for (const auto id: exact) {
		dumped |= DebugDumpOne(id, address, exact.size() > 1 ? fmt::format("{}_i{}", prefix, id.index) : prefix);
	}
	return dumped;
}

bool TextureCache::DebugNanScan(uint64_t address, uint32_t want_width, uint32_t want_height,
                                const std::string& label) {
	std::scoped_lock lock {m_lock};
	ImageId          found {};
	FindImagesInRegion(address, 1, false).ForEach([&](ImageId id) {
		const auto& image = m_slot_images[id];
		if (image.info.data.address != address || image.depth_id) return;
		if (!found || (image.info.extent.width == want_width && image.info.extent.height == want_height)) {
			found = id;
		}
	});
	if (!found) {
		return false;
	}
	auto&          image  = m_slot_images[found];
	const auto     format = image.backing.format;
	const auto     name   = vk::to_string(format);
	const uint32_t bytes  = vk::blockSize(format);
	const uint32_t comps  = vk::componentCount(format);
	const bool     packed = format == vk::Format::eB10G11R11UfloatPack32;
	const bool     sfloat = name.find("Sfloat") != std::string::npos && comps != 0 &&
	                    (bytes / comps == 2 || bytes / comps == 4);
	if (image.info.samples != 1 || (!packed && !sfloat) || image.info.IsDepth()) {
		return false;
	}
	const uint32_t width  = image.info.extent.width;
	const uint32_t height = image.info.extent.height;
	// Rows 30 %..60 % of the height in 8 bands of 2 rows: the torii, the hat and the village.
	constexpr uint32_t Bands = 8;
	constexpr uint32_t Rows  = 2;
	const uint64_t band_bytes = uint64_t {width} * Rows * bytes;
	std::array<vk::BufferImageCopy, Bands> copies {};
	uint32_t count = 0;
	for (uint32_t b = 0; b < Bands; b++) {
		const uint32_t y = static_cast<uint32_t>(uint64_t {height} * (30u + b * 4u) / 100u);
		if (y + Rows > height) continue;
		auto& copy            = copies[count];
		copy.bufferOffset     = band_bytes * count;
		copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageOffset      = vk::Offset3D {0, static_cast<int32_t>(y), 0};
		copy.imageExtent      = vk::Extent3D {width, Rows, 1};
		count++;
	}
	if (count == 0) {
		return false;
	}
	const uint64_t size   = band_bytes * count;
	auto           buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                       vk::BufferUsageFlagBits::eTransferDst,
	                                       Common::AlignUp(size, uint64_t {4}));
	image.Download({copies.data(), count}, buffer->Handle(), 0, size);
	vk::MemoryBarrier2 to_host {};
	to_host.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	to_host.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &to_host;
	m_scheduler.Current().HelperSink().pipelineBarrier2(dependency);
	const auto line = fmt::format("{} base 0x{:x} {}x{} {}", label, address, width, height, name);
	m_scheduler.DeferOperation([buffer, size, line, packed, bytes, comps] {
		buffer->Invalidate(0, size);
		const auto* data  = buffer->Mapped().data();
		uint64_t    nan   = 0;
		uint64_t    inf   = 0;
		uint64_t    texels = size / bytes;
		const auto  half = [&](uint16_t v) {
			if ((v & 0x7c00u) == 0x7c00u) ((v & 0x3ffu) != 0 ? nan : inf)++;
		};
		const auto single = [&](uint32_t v) {
			if ((v & 0x7f800000u) == 0x7f800000u) ((v & 0x7fffffu) != 0 ? nan : inf)++;
		};
		if (packed) {
			for (uint64_t at = 0; at + 4 <= size; at += 4) {
				uint32_t v = 0;
				std::memcpy(&v, data + at, 4);
				const uint32_t channels[3] = {v & 0x7ffu, (v >> 11u) & 0x7ffu, (v >> 22u) & 0x3ffu};
				const uint32_t mant[3]     = {0x3fu, 0x3fu, 0x1fu};
				const uint32_t shift[3]    = {6u, 6u, 5u};
				for (int c = 0; c < 3; c++) {
					if ((channels[c] >> shift[c]) == 0x1fu) ((channels[c] & mant[c]) != 0 ? nan : inf)++;
				}
			}
		} else if (bytes / comps == 2) {
			for (uint64_t at = 0; at + 2 <= size; at += 2) {
				uint16_t v = 0;
				std::memcpy(&v, data + at, 2);
				half(v);
			}
		} else {
			for (uint64_t at = 0; at + 4 <= size; at += 4) {
				uint32_t v = 0;
				std::memcpy(&v, data + at, 4);
				single(v);
			}
		}
		if (nan != 0 || inf != 0) {
			std::printf("NanScan: %s: nan %llu inf %llu of %llu texels sampled\n", line.c_str(),
			            static_cast<unsigned long long>(nan), static_cast<unsigned long long>(inf),
			            static_cast<unsigned long long>(texels));
		}
	});
	return true;
}

// KYTY_PASS_DUMP: one texel of a sampled row as float RGBA (raw integer values for integer
// formats; colour channels as stored, no sRGB decoding or swizzle).
static float PassSampleHalf(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u;
	const uint32_t exp  = (h >> 10u) & 0x1fu;
	const uint32_t man  = h & 0x3ffu;
	if (exp == 0) {
		const float v = std::ldexp(static_cast<float>(man), -24);
		return sign != 0 ? -v : v;
	}
	uint32_t bits = exp == 31 ? ((sign << 31u) | 0x7f800000u | (man << 13u))
	                          : ((sign << 31u) | ((exp + 112u) << 23u) | (man << 13u));
	float f = 0;
	std::memcpy(&f, &bits, 4);
	return f;
}

static float PassSampleSmallFloat(uint32_t v, uint32_t man_bits) {
	const uint32_t exp = v >> man_bits;
	const uint32_t man = v & ((1u << man_bits) - 1u);
	if (exp == 0) return std::ldexp(static_cast<float>(man), -14 - static_cast<int>(man_bits));
	if (exp == 31) return man != 0 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
	return std::ldexp(1.0f + static_cast<float>(man) / static_cast<float>(1u << man_bits), static_cast<int>(exp) - 15);
}

static void PassSampleDecode(vk::Format format, int aspect, const uint8_t* p, float out[4]) {
	out[0] = out[1] = out[2] = out[3] = 0.0f;
	uint32_t u32 = 0;
	if (aspect == 2) {
		out[0] = p[0];
		return;
	}
	if (aspect == 1) {
		switch (format) {
			case vk::Format::eD16Unorm:
			case vk::Format::eD16UnormS8Uint: {
				uint16_t v = 0;
				std::memcpy(&v, p, 2);
				out[0] = static_cast<float>(v) / 65535.0f;
				return;
			}
			case vk::Format::eX8D24UnormPack32:
			case vk::Format::eD24UnormS8Uint:
				std::memcpy(&u32, p, 4);
				out[0] = static_cast<float>(u32 & 0xffffffu) / 16777215.0f;
				return;
			default: std::memcpy(&out[0], p, 4); return;
		}
	}
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32:
			std::memcpy(&u32, p, 4);
			out[0] = PassSampleSmallFloat(u32 & 0x7ffu, 6);
			out[1] = PassSampleSmallFloat((u32 >> 11u) & 0x7ffu, 6);
			out[2] = PassSampleSmallFloat((u32 >> 22u) & 0x3ffu, 5);
			return;
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
			std::memcpy(&u32, p, 4);
			out[0] = static_cast<float>(u32 & 0x3ffu) / 1023.0f;
			out[1] = static_cast<float>((u32 >> 10u) & 0x3ffu) / 1023.0f;
			out[2] = static_cast<float>((u32 >> 20u) & 0x3ffu) / 1023.0f;
			out[3] = static_cast<float>(u32 >> 30u) / 3.0f;
			return;
		case vk::Format::eE5B9G9R9UfloatPack32: {
			std::memcpy(&u32, p, 4);
			const float scale = std::ldexp(1.0f, static_cast<int>(u32 >> 27u) - 15 - 9);
			out[0]            = static_cast<float>(u32 & 0x1ffu) * scale;
			out[1]            = static_cast<float>((u32 >> 9u) & 0x1ffu) * scale;
			out[2]            = static_cast<float>((u32 >> 18u) & 0x1ffu) * scale;
			return;
		}
		default: break;
	}
	const auto     name  = vk::to_string(format);
	const uint32_t comps = std::min<uint32_t>(vk::componentCount(format), 4);
	const uint32_t bytes = vk::blockSize(format);
	if (comps == 0 || bytes % comps != 0) return;
	const uint32_t width  = bytes / comps;
	const bool     sfloat = name.find("Sfloat") != std::string::npos;
	const bool     unorm  = name.find("Unorm") != std::string::npos || name.find("Srgb") != std::string::npos;
	const bool     snorm  = name.find("Snorm") != std::string::npos;
	const bool     sint   = name.find("Sint") != std::string::npos;
	for (uint32_t c = 0; c < comps; c++) {
		const uint8_t* q = p + c * width;
		if (width == 1) {
			const uint8_t v = q[0];
			out[c] = unorm   ? static_cast<float>(v) / 255.0f
			         : snorm ? std::max(-1.0f, static_cast<float>(static_cast<int8_t>(v)) / 127.0f)
			         : sint  ? static_cast<float>(static_cast<int8_t>(v))
			                 : static_cast<float>(v);
		} else if (width == 2) {
			uint16_t v = 0;
			std::memcpy(&v, q, 2);
			out[c] = sfloat  ? PassSampleHalf(v)
			         : unorm ? static_cast<float>(v) / 65535.0f
			         : snorm ? std::max(-1.0f, static_cast<float>(static_cast<int16_t>(v)) / 32767.0f)
			         : sint  ? static_cast<float>(static_cast<int16_t>(v))
			                 : static_cast<float>(v);
		} else if (width == 4) {
			std::memcpy(&u32, q, 4);
			if (sfloat) {
				std::memcpy(&out[c], &u32, 4);
			} else {
				out[c] = sint ? static_cast<float>(static_cast<int32_t>(u32)) : static_cast<float>(u32);
			}
		}
	}
}

bool TextureCache::DebugPassSample(uint64_t address, uint32_t want_width, uint32_t want_height, int aspect,
                                   uint32_t step, const std::string& thumb_path,
                                   std::function<void(std::string)> emit) {
	std::scoped_lock lock {m_lock};
	ImageId          found {};
	FindImagesInRegion(address, 1, false).ForEach([&](ImageId id) {
		const auto& image = m_slot_images[id];
		if (image.info.data.address != address || image.depth_id) return;
		if ((aspect != 0) != image.info.IsDepth()) return;
		if (!found || (image.info.extent.width == want_width && image.info.extent.height == want_height)) {
			found = id;
		}
	});
	if (!found) {
		return false;
	}
	auto&      image  = m_slot_images[found];
	const auto format = image.backing.format;
	const auto block  = vk::blockExtent(format);
	uint32_t   bytes  = vk::blockSize(format);
	if (aspect == 1) {
		bytes = format == vk::Format::eD16Unorm || format == vk::Format::eD16UnormS8Uint ? 2u : 4u;
	} else if (aspect == 2) {
		if (vk::to_string(format).find("S8") == std::string::npos) return false;
		bytes = 1;
	}
	if (image.info.samples != 1 || bytes == 0 || block[0] != 1 || block[1] != 1) {
		return false;
	}
	step                     = std::max(step, 1u);
	const uint32_t width     = image.info.extent.width;
	const uint32_t height    = image.info.extent.height;
	const uint32_t rows      = (height + step - 1) / step;
	const uint64_t row_bytes = uint64_t {width} * bytes;
	std::vector<vk::BufferImageCopy> copies(rows);
	const auto aspect_bits = aspect == 0   ? vk::ImageAspectFlagBits::eColor
	                         : aspect == 1 ? vk::ImageAspectFlagBits::eDepth
	                                       : vk::ImageAspectFlagBits::eStencil;
	for (uint32_t r = 0; r < rows; r++) {
		auto& copy            = copies[r];
		copy.bufferOffset     = row_bytes * r;
		copy.imageSubresource = {aspect_bits, 0, 0, 1};
		copy.imageOffset      = vk::Offset3D {0, static_cast<int32_t>(r * step), 0};
		copy.imageExtent      = vk::Extent3D {width, 1, 1};
	}
	const uint64_t size   = row_bytes * rows;
	auto           buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                       vk::BufferUsageFlagBits::eTransferDst,
	                                       Common::AlignUp(size, uint64_t {4}));
	image.Download(copies, buffer->Handle(), 0, size);
	vk::MemoryBarrier2 to_host {};
	to_host.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	to_host.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &to_host;
	m_scheduler.Current().HelperSink().pipelineBarrier2(dependency);
	const auto head = fmt::format("{}x{} vk{} {} img{}", width, height, static_cast<uint32_t>(format),
	                              aspect == 0 ? "color" : aspect == 1 ? "depth" : "stencil", found.index);
	m_scheduler.DeferOperation([buffer, size, head, format, aspect, bytes, width, rows, step, thumb_path,
	                            emit = std::move(emit)] {
		buffer->Invalidate(0, size);
		const auto*    data = buffer->Mapped().data();
		const uint32_t cols = (width + step - 1) / step;
		double         sum[4] {};
		float          lo[4];
		float          hi[4];
		for (int c = 0; c < 4; c++) {
			lo[c] = std::numeric_limits<float>::max();
			hi[c] = -std::numeric_limits<float>::max();
		}
		uint64_t           nan = 0, inf = 0, zero = 0, count = 0;
		constexpr uint32_t GX = 16, GY = 9;
		std::array<double, GX * GY>   grid {};
		std::array<uint32_t, GX * GY> grid_n {};
		std::vector<float>            thumb;
		if (!thumb_path.empty()) thumb.resize(uint64_t {cols} * rows * 4);
		const uint32_t comps = aspect != 0 ? 1u : std::max(1u, std::min<uint32_t>(vk::componentCount(format), 4));
		for (uint32_t r = 0; r < rows; r++) {
			for (uint32_t cx = 0; cx < cols; cx++) {
				float v[4];
				PassSampleDecode(format, aspect, data + uint64_t {r} * width * bytes + uint64_t {cx} * step * bytes, v);
				bool   all_zero = true;
				double lum      = 0;
				for (uint32_t c = 0; c < comps; c++) {
					const float x = v[c];
					if (std::isnan(x)) {
						nan++;
						continue;
					}
					if (std::isinf(x)) {
						inf++;
						continue;
					}
					sum[c] += x;
					lo[c] = std::min(lo[c], x);
					hi[c] = std::max(hi[c], x);
					if (x != 0.0f) all_zero = false;
					if (c < 3) lum += x;
				}
				zero += all_zero ? 1u : 0u;
				count++;
				const uint32_t gx = std::min(GX - 1, cx * GX / cols);
				const uint32_t gy = std::min(GY - 1, r * GY / rows);
				grid[gy * GX + gx] += lum / std::min(comps, 3u);
				grid_n[gy * GX + gx]++;
				if (!thumb.empty()) std::memcpy(&thumb[(uint64_t {r} * cols + cx) * 4], v, sizeof(v));
			}
		}
		std::string line = head;
		line += fmt::format(" hash={:016x} n={} nan={} inf={} zero={:.4f}", XXH3_64bits(data, size), count, nan, inf,
		                    count != 0 ? static_cast<double>(zero) / static_cast<double>(count) : 0.0);
		for (uint32_t c = 0; c < comps; c++) {
			line += fmt::format(" c{}={:.5g}/{:.5g}/{:.5g}", c,
			                    count != 0 ? sum[c] / static_cast<double>(count) : 0.0, lo[c], hi[c]);
		}
		line += " grid=";
		for (uint32_t i = 0; i < GX * GY; i++) {
			line += fmt::format("{}{:.4g}", i == 0 ? "" : ",", grid_n[i] != 0 ? grid[i] / grid_n[i] : 0.0);
		}
		if (!thumb.empty()) {
			const auto path = fmt::format("{}_{}x{}.f32", thumb_path, cols, rows);
			if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
				std::fwrite(thumb.data(), sizeof(float), thumb.size(), file);
				std::fclose(file);
			}
			line += " thumb=" + path;
		}
		emit(std::move(line));
	});
	return true;
}

bool TextureCache::DebugDumpOne(ImageId found, uint64_t address, const std::string& prefix) {
	auto&          image  = m_slot_images[found];
	const auto     format = image.backing.format;
	const auto     extent = vk::blockExtent(format);
	const uint32_t bytes  = vk::blockSize(format);
	if (image.info.samples != 1 || bytes == 0 || extent[0] != 1 || extent[1] != 1 ||
	    vk::componentCount(format) == 0) {
		return false;
	}
	const uint32_t width  = image.info.extent.width;
	const uint32_t height = image.info.extent.height;
	const uint64_t size   = uint64_t {width} * height * bytes;
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {width, height, 1};
	auto buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
	                                       vk::BufferUsageFlagBits::eTransferDst,
	                                       Common::AlignUp(size, uint64_t {4}));
	image.Download({&copy, 1}, buffer->Handle(), 0, size);
	vk::MemoryBarrier2 to_host {};
	to_host.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	to_host.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	to_host.dstStageMask  = vk::PipelineStageFlagBits2::eHost;
	to_host.dstAccessMask = vk::AccessFlagBits2::eHostRead;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &to_host;
	m_scheduler.Current().HelperSink().pipelineBarrier2(dependency);
	const auto path = fmt::format("{}_{:x}_{}x{}_vk{}_b{}.raw", prefix, address, width, height,
	                              static_cast<uint32_t>(format), bytes);
	m_scheduler.DeferOperation([buffer, size, path] {
		buffer->Invalidate(0, size);
		if (std::FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
			std::fwrite(buffer->Mapped().data(), 1, size, file);
			std::fclose(file);
		}
	});
	return true;
}

void TextureCache::PrintPartialGpuWriteSummary() {
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (m_pgw_print_ns == 0) {
		m_pgw_print_ns = now;
		return;
	}
	if (now - m_pgw_print_ns < 10'000'000'000ull) {
		return;
	}
	const auto& t = m_pgw_totals;
	const auto& p = m_pgw_printed;
	const auto  saved =
	    (t.whole_bytes - p.whole_bytes) - std::min(t.whole_bytes - p.whole_bytes, t.partial_bytes - p.partial_bytes);
	std::printf("Image partial GPU-write %.0fs: %" PRIu64 " buffer-write refreshes, %" PRIu64
	            " partial (%.1f MiB rebuilt, %.1f MiB saved), not tracked %" PRIu64 ", stale %" PRIu64
	            " (foreign %" PRIu64 " serial %" PRIu64 " cpu %" PRIu64 "), whole because: layout %" PRIu64 " most %" PRIu64 " other owners %" PRIu64
	            " ranges %" PRIu64 "; verify %" PRIu64 " checks %" PRIu64 " mismatches\n",
	            static_cast<double>(now - m_pgw_print_ns) * 1e-9, t.refreshes - p.refreshes,
	            t.partial - p.partial,
	            static_cast<double>(t.partial_bytes - p.partial_bytes) / 1048576.0,
	            static_cast<double>(saved) / 1048576.0, t.untracked - p.untracked, t.stale - p.stale,
	            t.stale_foreign - p.stale_foreign, t.stale_serial - p.stale_serial, t.stale_cpu - p.stale_cpu,
	            t.fallback_layout - p.fallback_layout, t.fallback_most - p.fallback_most,
	            t.fallback_others - p.fallback_others, t.overflows - p.overflows,
	            g_pgw_verify_checks.load(std::memory_order_relaxed),
	            g_pgw_verify_mismatches.load(std::memory_order_relaxed));
	std::fflush(stdout);
	m_pgw_printed  = t;
	m_pgw_print_ns = now;
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	if (image.ChunkTracked()) {
		(void)UpdateChunkWatchers<false>(image, 0, image.chunks.count);
		std::fill(image.chunks.untracked.begin(), image.chunks.untracked.end(), 0);
		image.chunks.untracked_count = 0;
		NoteImageStateChange();
		image.track_addr     = 0;
		NoteImageStateChange();
		image.track_addr_end = 0;
		NoteImageStateChange();
		// Writes are no longer observed: only a full refresh can restore partial validity.
		image.SetPartialValid(false);
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	NoteImageStateChange();
	image.track_addr_end = 0;
	NoteImageStateChange();
	if (!image.kept_holes.empty() || !image.kept_dirty.empty()) {
		// KYTY_IMAGE_GPU_KEEP: the released pages are not watched any more. Writes from now on are
		// not observed, so only a whole refresh is exact again.
		auto at = address;
		for (const auto& [begin, end]: image.kept_holes) {
			if (at < begin) {
				m_page_manager.UpdatePageWatchers<false>(at, begin - at);
			}
			at = std::max(at, end);
		}
		if (at < address + size) {
			m_page_manager.UpdatePageWatchers<false>(at, address + size - at);
		}
		image.kept_holes.clear();
		image.kept_dirty.clear();
		return;
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.live.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	NoteImageStateChange();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.live.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	NoteImageStateChange();
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

// KYTY_READBACK_SMALL_IMAGES_KB (default 0: off): GPU-written single-sample linear images up to
// this size are copied back to guest memory after each submission, as --readback-linear-images
// does for every linear image. Gran Turismo 7 draws a saved livery's picture into two 1920x540
// linear targets (4 MB each) and encodes it on the CPU as JPEG and PNG: without a read-back the
// encoders got the stale bytes of guest memory (noise). usa-ratt/KytyPS5-GT7 reads back up to
// 32768 KB (rounds 118 and 126); a race writes no GPU linear image above 64 KB.
static uint64_t SmallLinearReadbackBytes() {
	static constexpr auto bytes_init = [] {
		const char* value = std::getenv("KYTY_READBACK_SMALL_IMAGES_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull) << 10u;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(bytes_init())>> bytes_once;
	const uint64_t& bytes = bytes_once.Get(bytes_init);
	return bytes;
}

// KYTY_READBACK_LINEAR_COPIES=1 (default off; Ghost of Yotei preset): with --readback-linear-images,
// GPU-written linear depth targets and image copy destinations are read back after each
// submission too, not only storage images and color targets. Erlou93/KytyPS5-Yotei M118: the
// game's CPU occlusion test reads its hi-Z pyramid from memory, and that pyramid ends each frame
// as the D32 copy of an R32 one; a stale pyramid made distant objects vanish.
static bool ReadbackLinearCopiesEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_READBACK_LINEAR_COPIES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	// Small tiled targets are written back too: games read GPU results such as reductions and
	// simulation state from them on the CPU (PPSA04263 keeps the origin of its ocean wave grid
	// in a 1x1 R32G32 target, and its water physics read it to sample the wave heights).
	const auto& info   = image.info;
	const auto  texels = static_cast<uint64_t>(info.extent.width) * info.extent.height *
	                     info.extent.depth * info.resources.layers;
	const bool small_linear = SmallLinearReadbackBytes() != 0 && info.samples == 1 &&
	                          info.data.size <= SmallLinearReadbackBytes();
	if (!info.data.Empty() && (info.IsTiled() ? m_readback_tiled_images && texels <= SmallTiledReadbackTexels
	                                         : m_readback_linear_images || small_linear)) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	TexelImageLookup::ForEachOwnerList(m_image_page_table, address, size, [&](const ImageIds& owners) {
		owners.ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

ImageId TextureCache::FindImageWithSameBacking(const ImageInfo& requested,
                                               bool exact_format) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(requested.data.address, requested.data.size, pages)) {
		return {};
	}
	const auto* owners = m_image_page_table.Find(pages.first);
	if (owners == nullptr) {
		return {};
	}
	ImageId result {};
	// SameBacking requires an identical start address, so every possible exact match is
	// already present on the first page. Preserve the last-match ordering used by the
	// full region search without visiting the remaining pages or building a candidate list.
	// Read the live index under m_lock so unregister, alias replacement and reused slot
	// generations cannot leave a cached result pointing at a retired image.
	owners->ForEach([&](ImageId id) {
		const auto* image = m_slot_images.try_get(id);
		if (image != nullptr && image->registered &&
		    SameBacking(image->info, requested, exact_format)) {
			result = id;
		}
	});
	return result;
}

bool TextureCache::HasGpuWrittenImageOverlap(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return true;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		if (image.IsGpuModified()) {
			return true;
		}
	}
	return false;
}

bool TextureCache::HoldsImage(const ImageInfo& info) {
	if (info.data.Empty()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	return static_cast<bool>(FindImageWithSameBacking(info, false));
}

bool TextureCache::HeapExtentHeld(const ImageInfo& info, bool exact_format) {
	if (info.data.Empty()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	bool             held = false;
	for (const auto id: FindImagesInRegion(info.data.address, info.data.size, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered) {
			continue;
		}
		if (SameBacking(image->info, info, exact_format)) {
			return false;
		}
		held |= image->bindless_pinned && image->info.data.address == info.data.address &&
		        image->info.type == info.type && image->info.extent != info.extent;
	}
	return held;
}

bool TextureCache::MsDepthSample0Enabled() {
	static std::atomic<int64_t> slot {Common::OnceUnset};
	return Common::Once(slot, [] {
		       const char* value = std::getenv("KYTY_MS_DEPTH_SAMPLE0");
		       const bool  on    = value != nullptr && value[0] == '1';
		       if (on) {
			       LOGF("Multisample depth read as single-sample: from a copy of its first sample "
			            "(KYTY_MS_DEPTH_SAMPLE0=1)\n");
		       }
		       return on ? 1 : 0;
	       }) != 0;
}

void TextureCache::ReleaseUnregisteredImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || image->registered) {
		return;
	}
	if (on_image_views_dropped && !image->views.empty()) {
		(void)on_image_views_dropped(image->views);
	}
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] {
			m_slot_images.erase(id);
			NoteImageStateChange();
		});
	} else {
		m_slot_images.erase(id);
		NoteImageStateChange();
	}
}

ImageId TextureCache::SingleSampleDepth(ImageId id) {
	return SingleSampleCopyOf(id, false);
}

ImageId TextureCache::SingleSampleStencil(ImageId depth_id) {
	return SingleSampleCopyOf(depth_id, true);
}

bool TextureCache::PreserveStencilEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_PRESERVE_STENCIL");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Stencil planes: one saved copy per guest plane, swapped at depth binds "
			     "(KYTY_PRESERVE_STENCIL=1)\n");
		}
		return on;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

namespace {

uint64_t StencilPlaneKey(uint64_t depth_address, uint64_t plane, uint32_t width, uint32_t height) {
	uint64_t key = depth_address * 0x9E3779B97F4A7C15ull;
	key ^= plane + 0x632BE59BD9B4E019ull + (key << 6u) + (key >> 2u);
	key ^= (uint64_t {width} << 32u | height) + 0x9E3779B97F4A7C15ull + (key << 6u) + (key >> 2u);
	return key;
}

// The depth images whose stencil planes KYTY_PRESERVE_STENCIL swaps.
bool SwapsStencilPlanes(const Image& depth) {
	return depth.info.IsDepth() && depth.info.HasStencil() && depth.backing.image != nullptr &&
	       depth.backing.samples == 1 && depth.info.samples == 1 &&
	       depth.info.resources.layers == 1 && depth.backing.layers == 1 &&
	       !depth.info.metadata.stencil_compressed && depth.info.type == Prospero::ImageType::kColor2D;
}

vk::BufferImageCopy StencilPlaneCopy(uint32_t width, uint32_t height, vk::ImageAspectFlagBits aspect) {
	vk::BufferImageCopy copy {};
	copy.imageSubresource = {aspect, 0, 0, 1};
	copy.imageExtent      = vk::Extent3D {width, height, 1};
	return copy;
}

} // namespace

// KYTY_STENCIL_READ_DEPTH=1 (default off; Erlou93/KytyPS5-Yotei behaviour): a sampled read of a
// stencil plane binds the depth image's stencil aspect, whichever plane it holds, instead of the
// plane's saved copy, and so does a 1-byte colour image at the address of a plane a depth image
// swaps to (Ghost of Yotei's Deferred wetness 84f5a0d3: plane B resolved to a separate R8 image
// nothing writes, plane A to the saved copy; Erlou binds the D32S8 for both).
bool TextureCache::StencilReadDepthEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_READ_DEPTH");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Stencil planes: sampled reads bind the depth image's stencil (KYTY_STENCIL_READ_DEPTH=1)\n");
		}
		return on;
	}();
	return enabled;
}

// KYTY_STENCIL_READ_HELD=1 (default on; acts only with KYTY_STENCIL_READ_DEPTH=1 and
// KYTY_PRESERVE_STENCIL=1): a sampled read of a stencil plane record whose depth image holds
// another plane reads the plane's saved copy, as without KYTY_STENCIL_READ_DEPTH, instead of the
// depth image's stencil (the other plane). Ghost of Yotei's Deferred wetness 84f5a0d3 reads plane A
// (0x5004850000, the G-buffer classes) right after the volume draws swapped plane B (0x50615a0000)
// in: it read plane B there, wrote a garbage wetness mask and G-buffer (CS Lights checkerboard and
// stripes, cutscene wall blocks). Erlou93/KytyPS5-Yotei swaps plane A back in for that read.
// =0 restores the depth image's stencil for every plane.
bool TextureCache::StencilReadHeldEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_READ_HELD");
		const bool  on    = value == nullptr || value[0] != '0';
		if (on && StencilReadDepthEnabled() && PreserveStencilEnabled()) {
			LOGF("Stencil planes: a plane the depth image does not hold is read from its saved copy "
			     "(KYTY_STENCIL_READ_HELD=1)\n");
		}
		return on;
	}();
	return enabled;
}

ImageId TextureCache::StencilPlaneDepth(uint64_t plane_address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_stencil_plane_depth.find(plane_address);
	if (found == m_stencil_plane_depth.end()) {
		return {};
	}
	const auto* depth = m_slot_images.try_get(found->second);
	if (depth == nullptr || !depth->registered || !depth->info.IsDepth() || !depth->info.HasStencil() ||
	    !SwapsStencilPlanes(*depth) || depth->backing.samples != 1u) {
		return {};
	}
	return found->second;
}

// KYTY_STENCIL_READ_PARTNER=1 (default off; Erlou93/KytyPS5-Yotei parity): with KYTY_STENCIL_READ_DEPTH,
// a sampled read of a plane T# that resolves to a plain 1-byte image (no stencil association) binds
// the plane its depth image swapped away from when it took this one in, instead of this plane.
// Ghost of Yotei's Deferred wetness 84f5a0d3 reads plane B (0x50615a0000, the z-fail volume
// counts, ro[1.1]) and plane A (0x5004850000, the G-buffer classes, ro[1.2]); Erlou swaps plane A
// back into the D32S8 before the dispatch and binds the D32S8 for both, so its wetness mask is set
// on every classed pixel that has no G-buffer wetness (zero 0.795 vs ours 0.982 at the torii).
// Whether the PS5 reads the volume counts there is not known; this switch reproduces Erlou.
bool TextureCache::StencilReadPartnerEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_READ_PARTNER");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Stencil planes: a redirected plane read binds the partner plane (KYTY_STENCIL_READ_PARTNER=1)\n");
		}
		return on;
	}();
	return enabled;
}

ImageId TextureCache::StencilPlanePartnerRecord(uint64_t plane_address) {
	std::scoped_lock lock {m_lock};
	const auto       partner = m_stencil_plane_partner.find(plane_address);
	const auto       depth   = m_stencil_plane_depth.find(plane_address);
	if (partner == m_stencil_plane_partner.end() || depth == m_stencil_plane_depth.end()) {
		return {};
	}
	const auto* image = m_slot_images.try_get(depth->second);
	if (image == nullptr) {
		return {};
	}
	for (const auto record_id: image->stencil_ids) {
		if (const auto* record = m_slot_images.try_get(record_id);
		    record != nullptr && record->registered && record->depth_id == depth->second &&
		    record->info.data.address == partner->second) {
			return record_id;
		}
	}
	return {};
}

bool TextureCache::StencilPlaneHeld(const Image& record) const {
	const auto* depth = record.depth_id ? m_slot_images.try_get(record.depth_id) : nullptr;
	return depth == nullptr || !SwapsStencilPlanes(*depth) ||
	       depth->info.stencil.address == record.info.data.address;
}

bool TextureCache::SwapStencilPlane(ImageId depth_id, uint64_t next_plane) {
	auto&      depth    = m_slot_images[depth_id];
	const auto previous = depth.info.stencil.address;
	if (!SwapsStencilPlanes(depth) || previous == next_plane || previous == 0 || next_plane == 0) {
		return false;
	}
	const auto width  = depth.backing.extent.width;
	const auto height = depth.backing.extent.height;
	const auto bytes  = uint64_t {width} * height;
	const auto copy   = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
	// KYTY_STENCIL_READ_DEPTH: both planes are known to belong to this depth image.
	m_stencil_plane_depth[previous]   = depth_id;
	m_stencil_plane_depth[next_plane] = depth_id;
	m_stencil_plane_partner[next_plane] = previous;
	// The stencil aspect is saved as it is: a pending refresh of the held plane's record from guest
	// memory stays pending (its record is not held after the swap, so the refresh is applied when
	// the plane is swapped back in, after the restore).
	for (const auto record_id: depth.stencil_ids) {
		if (const auto* record = m_slot_images.try_get(record_id);
		    record != nullptr && record->info.data.address == previous) {
			NoteStructureChange(*record);
		}
	}
	auto& held = m_stencil_planes[StencilPlaneKey(depth.info.data.address, previous, width, height)];
	if (!held.buffer || held.buffer->Size() < bytes) {
		if (held.buffer) {
			m_scheduler.DeferOperation([old = held.buffer] { (void)old; });
		}
		held.buffer = std::make_shared<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
	}
	held.width          = width;
	held.height         = height;
	held.serial         = ++m_stencil_plane_serial;
	held.content_serial = Image::NextContentSerial();
	m_slot_images[depth_id].Download({&copy, 1}, held.buffer->Handle(), 0, bytes);
	const auto key   = StencilPlaneKey(depth.info.data.address, next_plane, width, height);
	const bool trace = MetaTraceAddress() != 0 && (MetaTraceAddress() == next_plane || MetaTraceAddress() == previous);
	// KYTY_STENCIL_PLANE_SYNC=2: a 1-byte colour image at the incoming plane's address, of this
	// extent, written by a shader or a clear after the plane's saved copy holds the plane now.
	const char* loaded = nullptr;
	if (StencilPlaneSyncLevel() >= 2 || trace) {
		const auto  saved_it     = m_stencil_planes.find(key);
		const auto  saved_serial = saved_it != m_stencil_planes.end() ? saved_it->second.content_serial : 0u;
		ImageId     newer_id {};
		uint64_t    newer_serial = 0;
		for (const auto id: FindImagesInRegion(next_plane, 1, false)) {
			const auto* image = m_slot_images.try_get(id);
			if (image == nullptr) continue;
			if (trace && MetaTraceLine()) {
				LOGF("MetaTrace: swap-in 0x%" PRIx64 " sees image 0x%" PRIx64 "+0x%" PRIx64 " %ux%u bpb %u depth %d record %d "
				     "samples %u write %" PRIu64 " saved %" PRIu64 "\n",
				     next_plane, image->info.data.address, image->info.data.size, image->backing.extent.width,
				     image->backing.extent.height, image->info.bytes_per_block, image->info.IsDepth() ? 1 : 0,
				     image->depth_id ? 1 : 0, image->backing.samples, image->shader_write_serial, saved_serial);
			}
			if (image->depth_id || image->info.IsDepth() || image->info.data.address != next_plane ||
			    image->info.bytes_per_block != 1u || image->backing.image == nullptr || image->backing.samples != 1u ||
			    image->backing.extent.width != width || image->backing.extent.height != height ||
			    image->shader_write_serial <= saved_serial || image->shader_write_serial <= newer_serial) {
				continue;
			}
			newer_id     = id;
			newer_serial = image->shader_write_serial;
		}
		if (newer_id && StencilPlaneSyncLevel() >= 2) {
			auto& saved = m_stencil_planes[key];
			if (!saved.buffer || saved.buffer->Size() < bytes) {
				if (saved.buffer) {
					m_scheduler.DeferOperation([old = saved.buffer] { (void)old; });
				}
				saved.buffer = std::make_shared<Buffer>(
				    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
				    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
			}
			saved.width          = width;
			saved.height         = height;
			saved.serial         = ++m_stencil_plane_serial;
			saved.content_serial = newer_serial;
			const auto color     = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eColor);
			m_slot_images[newer_id].Download({&color, 1}, saved.buffer->Handle(), 0, bytes);
			loaded = "loaded from a shader-written 1-byte image";
		}
	}
	const auto found = m_stencil_planes.find(key);
	const bool restored =
	    found != m_stencil_planes.end() && found->second.buffer && found->second.width == width &&
	    found->second.height == height;
	if (restored) {
		m_slot_images[depth_id].Upload({&copy, 1}, found->second.buffer->Handle(), 0, bytes);
	}
	for (const auto record_id: m_slot_images[depth_id].stencil_ids) {
		if (const auto* record = m_slot_images.try_get(record_id);
		    record != nullptr && record->info.data.address == next_plane) {
			NoteStructureChange(*record);
		}
	}
	NoteImageStateChange();
	static std::atomic<uint32_t> swaps {0};
	const auto n = swaps.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 8u || std::has_single_bit(n) || (trace && MetaTraceLine())) {
		LOGF("Stencil plane swap #%u (KYTY_PRESERVE_STENCIL): depth 0x%012" PRIx64
		     " %ux%u plane 0x%012" PRIx64 " -> 0x%012" PRIx64 " %s\n",
		     n, depth.info.data.address, width, height, previous, next_plane,
		     loaded != nullptr ? loaded : restored ? "restored" : "first use (guest bytes)");
	}
	if (loaded != nullptr) {
		static std::atomic<uint32_t> loads {0};
		const auto l = loads.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (l <= 8u || std::has_single_bit(l)) {
			LOGF("Stencil plane sync #%u (KYTY_STENCIL_PLANE_SYNC=2): plane 0x%012" PRIx64 " of depth 0x%012" PRIx64
			     " %ux%u %s\n",
			     l, next_plane, depth.info.data.address, width, height, loaded);
		}
	}
	return restored || loaded != nullptr;
}

// KYTY_STENCIL_PLANE_SYNC (default 2; based on Erlou93/KytyPS5-Yotei WORKAROUND_STENCIL_PLANE_SYNC):
// 2 = at swap-in, a 1-byte colour image at the incoming plane's address and the depth image's extent
// that a storage binding or a colour clear wrote after the plane's saved copy is loaded as the plane
// (Ghost of Yotei's z-fail volume plane 0x50615a0000: Erlou loads it from the R8 image the game clears
// every frame, 0 before the volume draws; ours restored the last frame's counts). 0 or 1: off (our
// StencilPlaneReader already covers Erlou's level 1, shader reads of a plane).
int TextureCache::StencilPlaneSyncLevel() {
	static constexpr auto level_init = [] {
		const char* value = std::getenv("KYTY_STENCIL_PLANE_SYNC");
		const int   on    = value == nullptr || value[0] == '\0' ? 2 : (value[0] >= '0' && value[0] <= '9' ? value[0] - '0' : 2);
		if (on >= 2 && PreserveStencilEnabled()) {
			LOGF("Stencil planes: a plane written as a 1-byte image is loaded at swap-in (KYTY_STENCIL_PLANE_SYNC=%d)\n",
			     on);
		}
		return on;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(level_init())>> level_once;
	const int& level = level_once.Get(level_init);
	return level;
}

// KYTY_STENCIL_PLANE_FILL_SAVED (default on; based on Erlou93/KytyPS5-Yotei KYTY_FIX_STENCIL_PLANE_WRITES,
// its fill part). =0: a fill of a stencil plane record clears the depth image's stencil aspect whatever
// plane it holds, as before. Not seen to fire in Ghost of Yotei gameplay (2 boots).
bool TextureCache::StencilPlaneFillSavedEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_PLANE_FILL_SAVED");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_STENCIL_PLANE_FILL_SAVED: a fill of the stencil plane `record_id` of `depth_id` while the depth
// image holds another plane goes into the plane's saved copy (the next swap-in restores it). False
// (nothing done) when the depth image holds this plane or the shapes differ. Caller holds m_lock.
bool TextureCache::FillStencilPlaneSaved(CommandBuffer& command, ImageId depth_id, ImageId record_id, uint8_t value,
                                         const char* what) {
	auto* depth  = m_slot_images.try_get(depth_id);
	auto* record = m_slot_images.try_get(record_id);
	if (depth == nullptr || record == nullptr || record->depth_id != depth_id || !SwapsStencilPlanes(*depth)) {
		return false;
	}
	const auto width  = depth->backing.extent.width;
	const auto height = depth->backing.extent.height;
	const auto bytes  = uint64_t {width} * height;
	if (depth->info.stencil.address == 0 || depth->info.stencil.address == record->info.data.address ||
	    record->info.extent.width != width || record->info.extent.height != height ||
	    record->stencil_subresources.base_layer != 0 || bytes % 4u != 0u) {
		return false;
	}
	const auto depth_address = depth->info.data.address;
	const auto held_address  = depth->info.stencil.address;
	const auto plane_address = record->info.data.address;
	auto&      held          = m_stencil_planes[StencilPlaneKey(depth_address, plane_address, width, height)];
	if (!held.buffer || held.buffer->Size() < bytes) {
		if (held.buffer) {
			m_scheduler.DeferOperation([old = held.buffer] { (void)old; });
		}
		held.buffer = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		                                       vk::BufferUsageFlagBits::eTransferSrc |
		                                           vk::BufferUsageFlagBits::eTransferDst,
		                                       bytes);
	}
	held.width          = width;
	held.height         = height;
	held.serial         = ++m_stencil_plane_serial;
	held.content_serial = Image::NextContentSerial();
	command.EndRendering();
	const auto         native = command.Handle();
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.srcAccessMask = vk::AccessFlagBits2::eMemoryWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	native.pipelineBarrier2(dependency);
	native.fillBuffer(held.buffer->Handle(), 0, bytes, uint32_t {value} * 0x01010101u);
	native.pipelineBarrier2(dependency);
	static std::atomic<uint32_t> fills {0};
	const auto n = fills.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 8u || std::has_single_bit(n)) {
		LOGF("Stencil plane fill #%u (KYTY_STENCIL_PLANE_FILL_SAVED, %s): plane 0x%012" PRIx64 " of depth 0x%012" PRIx64
		     " (holding 0x%012" PRIx64 ") %ux%u value %u into its saved copy\n",
		     n, what, plane_address, depth_address, held_address, width, height, static_cast<unsigned>(value));
	}
	return true;
}

bool TextureCache::PreserveStencilRecreateEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PRESERVE_STENCIL_RECREATE");
		const bool  on    = value != nullptr && value[0] == '1';
		if (on) {
			LOGF("Stencil of depth images replaced by colour images: saved and restored "
			     "(KYTY_PRESERVE_STENCIL_RECREATE=1)\n");
		}
		return on;
	}();
	return enabled;
}

void TextureCache::SaveRecreatedStencil(Image& depth) {
	const auto width  = depth.backing.extent.width;
	const auto height = depth.backing.extent.height;
	const auto bytes  = uint64_t {width} * height;
	if (!m_recreate_stencil || m_recreate_stencil->Size() < bytes) {
		if (m_recreate_stencil) {
			m_scheduler.DeferOperation([old = m_recreate_stencil] { (void)old; });
		}
		m_recreate_stencil = std::make_shared<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, bytes);
	}
	const auto copy = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
	// (Download transitions the image itself; the copy is recorded before the image is freed.)
	depth.Download({&copy, 1}, m_recreate_stencil->Handle(), 0, bytes);
	m_recreate_stencil_address = depth.info.data.address;
	m_recreate_stencil_plane   = depth.info.stencil.address;
	m_recreate_stencil_width   = width;
	m_recreate_stencil_height  = height;
	m_recreate_stencil_valid   = true;
	static std::atomic<uint32_t> saves {0};
	const auto n = saves.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 4u || std::has_single_bit(n)) {
		LOGF("Stencil save #%u (KYTY_PRESERVE_STENCIL_RECREATE): depth 0x%012" PRIx64 " %ux%u plane 0x%012" PRIx64
		     " replaced by a colour image\n",
		     n, depth.info.data.address, width, height, depth.info.stencil.address);
	}
}

bool TextureCache::RestoreRecreatedStencil(Image& depth) {
	const auto width  = depth.backing.extent.width;
	const auto height = depth.backing.extent.height;
	if (!m_recreate_stencil_valid || m_recreate_stencil_address != depth.info.data.address ||
	    m_recreate_stencil_width != width || m_recreate_stencil_height != height) {
		static std::atomic<uint32_t> misses {0};
		const auto n = misses.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (n <= 4u || std::has_single_bit(n)) {
			LOGF("Stencil restore miss #%u (KYTY_PRESERVE_STENCIL_RECREATE): depth 0x%012" PRIx64
			     " %ux%u, saved %s 0x%012" PRIx64 " %ux%u\n",
			     n, depth.info.data.address, width, height, m_recreate_stencil_valid ? "valid" : "none",
			     m_recreate_stencil_address, m_recreate_stencil_width, m_recreate_stencil_height);
		}
		return false;
	}
	const auto bytes = uint64_t {width} * height;
	const auto copy  = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
	depth.Upload({&copy, 1}, m_recreate_stencil->Handle(), 0, bytes);
	m_recreate_stencil_valid = false;
	// The stencil aspect holds the plane the replaced image held: with KYTY_PRESERVE_STENCIL the
	// next depth bind swaps planes from that one.
	if (PreserveStencilEnabled() && m_recreate_stencil_plane != 0) {
		depth.info.stencil.address = m_recreate_stencil_plane;
	}
	static std::atomic<uint32_t> restores {0};
	const auto n = restores.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 4u || std::has_single_bit(n)) {
		LOGF("Stencil restore #%u (KYTY_PRESERVE_STENCIL_RECREATE): depth 0x%012" PRIx64 " %ux%u plane 0x%012" PRIx64
		     "\n",
		     n, depth.info.data.address, width, height, m_recreate_stencil_plane);
	}
	return true;
}

bool TextureCache::StencilAliasCopyEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_ALIAS_COPY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool TextureCache::CopyStencilPlaneToAlias(ImageId target_id) {
	std::scoped_lock lock {m_lock};
	const auto*      target = m_slot_images.try_get(target_id);
	if (target == nullptr || !target->registered || target->depth_id || target->info.IsDepth() ||
	    target->info.bytes_per_block != 1u || target->backing.samples != 1u ||
	    target->backing.image == nullptr || target->info.IsVolume()) {
		return false;
	}
	const auto address = target->info.data.address;
	ImageId    record_id {};
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (image != nullptr && image->depth_id && image->info.data.address == address) {
			record_id = id;
		}
	}
	if (!record_id) {
		return false;
	}
	const auto  depth_id = m_slot_images[record_id].depth_id;
	const auto* depth    = m_slot_images.try_get(depth_id);
	if (depth == nullptr || !SwapsStencilPlanes(*depth) ||
	    (depth->backing.extent.width == target->backing.extent.width &&
	     depth->backing.extent.height == target->backing.extent.height)) {
		return false;
	}
	const auto depth_width  = depth->backing.extent.width;
	const auto depth_height = depth->backing.extent.height;
	const auto width        = std::min(depth_width, target->backing.extent.width);
	const auto height       = std::min(depth_height, target->backing.extent.height);
	if (width == 0 || height == 0) {
		return false;
	}
	// What the CPU or a buffer write put in the alias's memory is taken first, so no later refresh
	// replaces the copy with older guest bytes.
	RefreshImage(target_id);
	vk::Buffer source_buffer {};
	uint32_t   source_row = 0;
	bool       from_saved = false;
	if (StencilPlaneHeld(m_slot_images[record_id])) {
		const auto bytes = uint64_t {width} * height;
		if (!m_stencil_alias_staging || m_stencil_alias_staging->Size() < bytes) {
			if (m_stencil_alias_staging) {
				m_scheduler.DeferOperation([old = m_stencil_alias_staging] { (void)old; });
			}
			m_stencil_alias_staging = std::make_shared<Buffer>(
			    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
			    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
			    std::max<uint64_t>(bytes, uint64_t {1920} * 1080));
		}
		auto stencil_copy            = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
		stencil_copy.bufferRowLength = width;
		m_slot_images[depth_id].Download({&stencil_copy, 1}, m_stencil_alias_staging->Handle(), 0, bytes);
		source_buffer = m_stencil_alias_staging->Handle();
		source_row    = width;
	} else {
		const auto found = m_stencil_planes.find(
		    StencilPlaneKey(m_slot_images[depth_id].info.data.address, address, depth_width, depth_height));
		if (found == m_stencil_planes.end() || !found->second.buffer) {
			return false;
		}
		source_buffer = found->second.buffer->Handle();
		source_row    = depth_width;
		from_saved    = true;
	}
	auto color_copy            = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eColor);
	color_copy.bufferRowLength = source_row;
	auto& alias                = m_slot_images[target_id];
	alias.Upload({&color_copy, 1}, source_buffer,
	             0, uint64_t {source_row} * (height - 1u) + width);
	MarkImageGpuModified(alias);
	static std::atomic<uint32_t> logged {0};
	const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 8u || std::has_single_bit(n)) {
		LOGF("Stencil alias copy #%u (KYTY_STENCIL_ALIAS_COPY, approximation): plane 0x%012" PRIx64
		     " depth %ux%u -> alias %ux%u%s\n",
		     n, address, depth_width, depth_height, alias.backing.extent.width, alias.backing.extent.height,
		     from_saved ? " from the saved plane" : "");
	}
	return true;
}

bool TextureCache::StencilStorageShadowEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_STENCIL_STORAGE_SHADOW");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

ImageId TextureCache::StencilStorageShadow(ImageId record_id, bool& shadow) {
	shadow = false;
	std::scoped_lock lock {m_lock};
	const auto* record = m_slot_images.try_get(record_id);
	if (record == nullptr || !record->depth_id) {
		return record_id;
	}
	const auto  depth_id = record->depth_id;
	const auto* depth    = m_slot_images.try_get(depth_id);
	if (depth == nullptr || !SwapsStencilPlanes(*depth) || !StencilPlaneHeld(*record)) {
		return depth_id;
	}
	const auto width  = depth->backing.extent.width;
	const auto height = depth->backing.extent.height;
	const auto bytes  = uint64_t {width} * height;
	for (const auto& [pending_shadow, pending_depth]: m_pending_stencil_shadows) {
		if (pending_depth == depth_id) {
			shadow = true; // already holds the stencil for this dispatch
			return pending_shadow;
		}
	}
	auto  found     = m_stencil_shadows.find(depth_id);
	auto* shadow_im = found != m_stencil_shadows.end() ? m_slot_images.try_get(found->second) : nullptr;
	if (shadow_im == nullptr || shadow_im->info.extent.width != width || shadow_im->info.extent.height != height) {
		if (shadow_im != nullptr) {
			ReleaseUnregisteredImage(found->second);
		}
		ImageInfo info {};
		info.pixel_format    = vk::Format::eR8Uint;
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.type            = Prospero::ImageType::kColor2D;
		info.extent          = {width, height, 1};
		info.resources       = {1, 1};
		info.pitch           = width;
		info.bytes_per_block = 1;
		info.samples         = 1;
		info.tile_mode       = Prospero::TileMode::kLinear;
		info.mip_layout[0]   = {0, bytes, width, height};
		const auto shadow_id = InsertImage(info);
		m_stencil_shadows[depth_id] = shadow_id;
	}
	const auto shadow_id = m_stencil_shadows[depth_id];
	if (!m_stencil_alias_staging || m_stencil_alias_staging->Size() < bytes) {
		if (m_stencil_alias_staging) {
			m_scheduler.DeferOperation([old = m_stencil_alias_staging] { (void)old; });
		}
		m_stencil_alias_staging = std::make_shared<Buffer>(
		    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
		    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		    std::max<uint64_t>(bytes, uint64_t {1920} * 1080));
	}
	const auto stencil_copy = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
	const auto color_copy   = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eColor);
	m_slot_images[depth_id].Download({&stencil_copy, 1}, m_stencil_alias_staging->Handle(), 0, bytes);
	auto& target               = m_slot_images[shadow_id];
	target.tick_accessed_last  = m_scheduler.CurrentTick();
	target.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	target.Upload({&color_copy, 1}, m_stencil_alias_staging->Handle(), 0, bytes);
	m_pending_stencil_shadows.emplace_back(shadow_id, depth_id);
	static std::atomic<uint32_t> logged {0};
	const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 8u || std::has_single_bit(n)) {
		LOGF("Stencil storage shadow #%u (KYTY_STENCIL_STORAGE_SHADOW): plane 0x%012" PRIx64
		     " of depth 0x%012" PRIx64 " %ux%u\n",
		     n, m_slot_images[record_id].info.data.address, m_slot_images[depth_id].info.data.address, width,
		     height);
	}
	shadow = true;
	return shadow_id;
}

void TextureCache::FlushStencilStorageShadows() {
	std::scoped_lock lock {m_lock};
	for (const auto& [shadow_id, depth_id]: m_pending_stencil_shadows) {
		auto* shadow = m_slot_images.try_get(shadow_id);
		auto* depth  = m_slot_images.try_get(depth_id);
		if (shadow == nullptr || depth == nullptr || !depth->registered || !SwapsStencilPlanes(*depth) ||
		    depth->backing.extent.width != shadow->info.extent.width ||
		    depth->backing.extent.height != shadow->info.extent.height) {
			continue;
		}
		const auto width        = shadow->info.extent.width;
		const auto height       = shadow->info.extent.height;
		const auto bytes        = uint64_t {width} * height;
		const auto stencil_copy = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eStencil);
		const auto color_copy   = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eColor);
		shadow->Download({&color_copy, 1}, m_stencil_alias_staging->Handle(), 0, bytes);
		m_slot_images[depth_id].Upload({&stencil_copy, 1}, m_stencil_alias_staging->Handle(), 0, bytes);
		MarkImageGpuModified(m_slot_images[depth_id]);
	}
	m_pending_stencil_shadows.clear();
	NoteImageStateChange();
}

ImageId TextureCache::StencilPlaneReader(ImageId record_id, bool& copy) {
	copy = false;
	std::scoped_lock lock {m_lock};
	const auto* record = m_slot_images.try_get(record_id);
	if (record == nullptr || !record->depth_id) {
		return record_id;
	}
	const auto depth_id = record->depth_id;
	if (StencilPlaneHeld(*record)) {
		return depth_id;
	}
	const auto& depth  = m_slot_images[depth_id];
	const auto  width  = depth.backing.extent.width;
	const auto  height = depth.backing.extent.height;
	const auto  found =
	    m_stencil_planes.find(StencilPlaneKey(depth.info.data.address, record->info.data.address, width, height));
	if (found == m_stencil_planes.end() || !found->second.buffer || found->second.width != width ||
	    found->second.height != height) {
		return depth_id; // never saved: the held plane, as without the switch
	}
	auto& saved  = found->second;
	auto* reader = saved.reader ? m_slot_images.try_get(saved.reader) : nullptr;
	if (reader == nullptr || reader->info.extent.width != width || reader->info.extent.height != height) {
		if (reader != nullptr) {
			ReleaseUnregisteredImage(saved.reader);
		}
		ImageInfo info {};
		info.pixel_format    = vk::Format::eR8Uint;
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.type            = Prospero::ImageType::kColor2D;
		info.extent          = {width, height, 1};
		info.resources       = {1, 1};
		info.pitch           = width;
		info.bytes_per_block = 1;
		info.samples         = 1;
		info.tile_mode       = Prospero::TileMode::kLinear;
		info.mip_layout[0]   = {0, uint64_t {width} * height, width, height};
		const auto reader_id = InsertImage(info);
		// InsertImage may move the slots: the entry is looked up again.
		auto& fresh         = m_stencil_planes[found->first];
		fresh.reader        = reader_id;
		fresh.reader_serial = 0;
	}
	auto& entry  = m_stencil_planes[found->first];
	auto& target = m_slot_images[entry.reader];
	target.tick_accessed_last  = m_scheduler.CurrentTick();
	target.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	if (entry.reader_serial != entry.serial) {
		const auto plane_copy = StencilPlaneCopy(width, height, vk::ImageAspectFlagBits::eColor);
		target.Upload({&plane_copy, 1}, entry.buffer->Handle(), 0, uint64_t {width} * height);
		entry.reader_serial = entry.serial;
		static std::atomic<uint32_t> reads {0};
		const auto n = reads.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (n <= 8u || std::has_single_bit(n)) {
			LOGF("Stencil plane read #%u (KYTY_PRESERVE_STENCIL): plane 0x%012" PRIx64
			     " of depth 0x%012" PRIx64 " %ux%u from its saved copy\n",
			     n, m_slot_images[record_id].info.data.address, m_slot_images[depth_id].info.data.address,
			     width, height);
		}
	}
	copy = true;
	return entry.reader;
}

ImageId TextureCache::SingleSampleCopyOf(ImageId id, bool stencil) {
	if (!MsDepthSample0Enabled()) {
		return id;
	}
	std::scoped_lock lock {m_lock};
	const auto*      source = m_slot_images.try_get(id);
	const auto       format = source != nullptr ? source->backing.format : vk::Format::eUndefined;
	const bool       has_stencil = format == vk::Format::eD16UnormS8Uint ||
	                         format == vk::Format::eD24UnormS8Uint ||
	                         format == vk::Format::eD32SfloatS8Uint;
	if (source == nullptr || !source->registered || source->binding.needs_rebind ||
	    source->backing.samples <= 1 || !source->info.IsDepth() || (stencil && !has_stencil) ||
	    source->backing.image == nullptr || source->backing.image_type != vk::ImageType::e2D ||
	    source->info.resources.layers == 0 ||
	    source->info.resources.layers != source->backing.layers ||
	    source->info.resources.levels != 1 || source->depth_id) {
		return id;
	}
	// What FindTexture does with an image it binds, for the source of the copy: its use is noted
	// and what the CPU or a buffer write put in its memory is uploaded first (which moves its
	// content serial, so the copy is made again).
	TouchImage(m_slot_images[id]);
	RefreshImage(id);
	if (stencil) {
		const auto stencil_ids = m_slot_images[id].stencil_ids;
		for (const auto stencil_id: stencil_ids) {
			if (const auto* plane = m_slot_images.try_get(stencil_id);
			    plane != nullptr && plane->depth_id == id && plane->registered) {
				RefreshImage(stencil_id);
			}
		}
	}
	source = &m_slot_images[id];
	// The stencil changes with the depth image (draws, clears, copies) and its stencil planes
	// (their own uploads).
	uint64_t serial = source->ContentSerial();
	if (stencil) {
		for (const auto stencil_id: source->stencil_ids) {
			if (const auto* plane = m_slot_images.try_get(stencil_id); plane != nullptr) {
				serial = serial * 31u + plane->ContentSerial() + 1u;
			}
		}
	}
	const auto width  = source->info.extent.width;
	const auto height = source->info.extent.height;
	const auto layers = source->info.resources.layers;
	auto&      entry  = (stencil ? m_single_sample_stencils : m_single_sample_depths)[id];
	auto*      copy   = entry.copy ? m_slot_images.try_get(entry.copy) : nullptr;
	if (copy == nullptr || copy->info.extent.width != width ||
	    copy->info.extent.height != height || copy->info.resources.layers != layers) {
		if (copy != nullptr) {
			ReleaseUnregisteredImage(entry.copy);
		}
		const uint32_t bytes = stencil ? 1u : 4u;
		ImageInfo      info {};
		info.pixel_format    = stencil ? vk::Format::eR8Uint : vk::Format::eD32Sfloat;
		info.guest_format    = stencil ? Prospero::BufferFormat::k8UInt : source->info.guest_format;
		info.type            = source->info.type;
		info.extent          = {width, height, 1};
		info.resources       = {1, layers};
		info.pitch           = width;
		info.bytes_per_block = bytes;
		info.samples         = 1;
		info.tile_mode       = Prospero::TileMode::kLinear;
		info.mip_layout[0]   = {0, uint64_t {width} * height * bytes, width, height};
		const auto guest     = source->info.data.address;
		const auto samples   = source->backing.samples;
		const auto copy_id   = InsertImage(info);
		// InsertImage may move the slots: entry and source are looked up again.
		auto& fresh  = (stencil ? m_single_sample_stencils : m_single_sample_depths)[id];
		fresh.copy   = copy_id;
		fresh.serial = UINT64_MAX;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("Multisample depth read as single-sample: %s0x%012" PRIx64
			     " %ux%u, %u layers, %u samples: read from a copy of its first sample\n",
			     stencil ? "the stencil of " : "", guest, width, height, layers, samples);
		}
	}
	auto& current = (stencil ? m_single_sample_stencils : m_single_sample_depths)[id];
	auto& target  = m_slot_images[current.copy];
	target.tick_accessed_last  = m_scheduler.CurrentTick();
	target.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	if (current.serial != serial) {
		if (stencil) {
			m_blit_helper.CopyMsStencilSample0(m_slot_images[id], target);
		} else {
			m_blit_helper.CopyMsDepthSample0(m_slot_images[id], target);
		}
		current.serial = serial;
	}
	return current.copy;
}

namespace {

// KYTY_LIGHT_GRID_CAP: the shapes pixel shaders bind light grids with (NoteLightGridShape):
// width, height, row pitch and layers, packed under the top bit; 0 is a free slot.
std::array<std::atomic<uint64_t>, 16> g_light_grid_shapes {};

uint64_t LightGridShapeKey(const ImageInfo& info) {
	return (uint64_t {1} << 63u) | (info.extent.width & 0xffffu) |
	       (static_cast<uint64_t>(info.extent.height & 0xffffu) << 16u) |
	       (static_cast<uint64_t>(info.pitch & 0xffffu) << 32u) |
	       (static_cast<uint64_t>(info.resources.layers & 0x7fffu) << 48u);
}

bool LightGridFormat(const ImageInfo& info) {
	return info.pixel_format == vk::Format::eR32G32Uint && !info.IsTiled() && !info.IsVolume() &&
	       info.resources.levels == 1 && info.resources.layers >= 2 && info.samples == 1 &&
	       !info.data.Empty() && info.data.size <= 0x40000u;
}

bool IsLightGridShape(const ImageInfo& info) {
	if (!LightGridFormat(info)) {
		return false;
	}
	const auto key = LightGridShapeKey(info);
	for (const auto& shape: g_light_grid_shapes) {
		const auto value = shape.load(std::memory_order_relaxed);
		if (value == key) {
			return true;
		}
		if (value == 0) {
			return false;
		}
	}
	return false;
}

// No light grid cell of GT7's asks for this many lights (the counts seen are below 64); the
// garbage ones ask for up to 65535.
constexpr uint32_t LightGridMostLights = 1024;

// A light grid's bytes: cell (x, y) of layer l is two words at l * layer bytes + (y * row + x) * 8,
// a light-list offset and packed counts (the low 16 bits one of them). Cells that ask for more
// than LightGridMostLights lights are counted and, with `repair`, emptied in place.
uint32_t ScanLightGrid(uint8_t* bytes, uint64_t size, uint32_t width, uint32_t height,
                       uint32_t layers, uint32_t row, bool repair) {
	if (layers == 0 || size % layers != 0) {
		return 0;
	}
	const uint64_t layer_bytes = size / layers;
	if (static_cast<uint64_t>(row) * height * 8u > layer_bytes || width > row) {
		return 0;
	}
	uint32_t garbage = 0;
	for (uint32_t layer = 0; layer < layers; layer++) {
		for (uint32_t y = 0; y < height; y++) {
			for (uint32_t x = 0; x < width; x++) {
				const uint64_t at = layer * layer_bytes + (static_cast<uint64_t>(y) * row + x) * 8u;
				uint32_t       packed = 0;
				std::memcpy(&packed, bytes + at + 4u, sizeof(packed));
				if ((packed & 0xffffu) <= LightGridMostLights) {
					continue;
				}
				garbage++;
				if (repair) {
					std::memset(bytes + at, 0, 8u);
				}
			}
		}
	}
	return garbage;
}

} // namespace

bool TextureCache::LightGridCapEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LIGHT_GRID_CAP");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

void TextureCache::NoteLightGridShape(const ImageInfo& info) {
	if (!LightGridFormat(info)) {
		return;
	}
	const auto key = LightGridShapeKey(info);
	for (auto& shape: g_light_grid_shapes) {
		auto value = shape.load(std::memory_order_relaxed);
		if (value == key) {
			return;
		}
		if (value == 0) {
			if (shape.compare_exchange_strong(value, key, std::memory_order_relaxed)) {
				LOGF("LightGrid: pixel shaders read light grids of %ux%u cells, %u layers (row "
				     "pitch %u): their uploads are checked (KYTY_LIGHT_GRID_CAP)\n",
				     info.extent.width, info.extent.height, info.resources.layers, info.pitch);
				return;
			}
			if (value == key) {
				return;
			}
		}
	}
}

bool TextureCache::LightGridProbeEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LIGHT_GRID_PROBE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

void TextureCache::ProbeLightGrid(const ImageInfo& info, uint64_t shader_hash) {
	if (!LightGridFormat(info)) {
		return;
	}
	const auto range = info.data;
	thread_local std::vector<uint8_t> guest;
	guest.resize(static_cast<size_t>(range.size));
	if (!LibKernel::Memory::TryReadBacking(range.address, guest.data(), range.size)) {
		return;
	}
	const auto row     = std::max(info.pitch, info.extent.width);
	const auto layers  = info.resources.layers;
	const auto garbage = ScanLightGrid(guest.data(), range.size, info.extent.width,
	                                   info.extent.height, layers, row, false);
	// Largest count and the sum of counts (low 16 bits of each cell's second word).
	uint32_t       most = 0;
	uint64_t       sum  = 0;
	const uint64_t layer_bytes = layers != 0 ? range.size / layers : 0;
	for (uint32_t l = 0; l < layers; l++) {
		for (uint32_t y = 0; y < info.extent.height; y++) {
			for (uint32_t x = 0; x < info.extent.width; x++) {
				const uint64_t at = l * layer_bytes + (static_cast<uint64_t>(y) * row + x) * 8u + 4u;
				if (at + 4u > guest.size()) continue;
				uint32_t packed = 0;
				std::memcpy(&packed, guest.data() + at, sizeof(packed));
				most = std::max(most, packed & 0xffffu);
				sum += packed & 0xffffu;
			}
		}
	}
	static std::atomic<uint64_t> probes {0};
	static std::atomic<uint64_t> bad {0};
	const auto n = probes.fetch_add(1, std::memory_order_relaxed) + 1;
	const auto b = garbage != 0 ? bad.fetch_add(1, std::memory_order_relaxed) + 1
	                            : bad.load(std::memory_order_relaxed);
	if ((garbage != 0 && (b <= 64 || b % 256 == 0)) || n <= 16 || n % 4096 == 0) {
		LOGF("LightGridProbe: ps 0x%016" PRIx64 " grid 0x%012" PRIx64 "+0x%" PRIx64
		     " (%ux%u, %u layers, pitch %u): guest bytes %u cells > %u lights, max %u, sum %" PRIu64
		     " (probe %" PRIu64 ", %" PRIu64 " with garbage)\n",
		     shader_hash, range.address, range.size, info.extent.width, info.extent.height, layers,
		     info.pitch, garbage, LightGridMostLights, most, sum, n, b);
	}
}

void TextureCache::RepairLightGridUpload(const Image& image, Buffer& source,
                                         uint64_t source_offset) {
	const auto& info  = image.info;
	const auto  range = info.data;
	if (source.Mapped().size() < source_offset || source.Mapped().size() - source_offset < range.size) {
		return;
	}
	// The guest bytes, read again into cached memory (the staging copy is write-combined, slow to
	// read); when they hold such cells they are what is uploaded, repaired.
	thread_local std::vector<uint8_t> guest;
	guest.resize(static_cast<size_t>(range.size));
	if (!LibKernel::Memory::TryReadBacking(range.address, guest.data(), range.size)) {
		return;
	}
	const auto row = std::max(info.pitch, info.extent.width);
	const auto garbage = ScanLightGrid(guest.data(), range.size, info.extent.width,
	                                   info.extent.height, info.resources.layers, row, true);
	if (garbage == 0) {
		return;
	}
	std::memcpy(source.Mapped().data() + source_offset, guest.data(), static_cast<size_t>(range.size));
	source.Flush(source_offset, range.size);
	static std::atomic<uint64_t> found {0};
	const auto                   n = found.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 24 || n % 64 == 0) {
		LOGF("LightGrid: 0x%012" PRIx64 "+0x%" PRIx64 " (%ux%u, %u layers, pitch %u): %u cells ask "
		     "for more than %u lights; uploaded as empty cells (%" PRIu64 " so far)\n",
		     range.address, range.size, info.extent.width, info.extent.height,
		     info.resources.layers, info.pitch, garbage, LightGridMostLights, n);
	}
}

bool TextureCache::SubExtentCopyEnabled() {
	static std::atomic<int64_t> slot {Common::OnceUnset};
	return Common::Once(slot, [] {
		       const char* value = std::getenv("KYTY_SUB_EXTENT_COPY");
		       return value != nullptr && value[0] == '1' ? 1 : 0;
	       }) != 0;
}

ImageId TextureCache::SubExtentCopy(ImageId id, const ImageInfo& requested) {
	if (!SubExtentCopyEnabled()) {
		return id;
	}
	std::scoped_lock lock {m_lock};
	const auto*      source = m_slot_images.try_get(id);
	if (source == nullptr || !source->registered || source->depth_id ||
	    source->backing.image == nullptr || source->binding.needs_rebind) {
		return id;
	}
	const auto& info = source->info;
	if (info.data.Empty() || info.IsDepth() || info.HasStencil() || info.IsBlock() ||
	    info.IsVolume() || source->backing.samples != 1 ||
	    source->backing.image_type != vk::ImageType::e2D || requested.samples != 1 ||
	    requested.type != info.type || requested.data.address != info.data.address ||
	    requested.data.size != info.data.size || requested.tile_mode != info.tile_mode ||
	    requested.resources != info.resources || requested.resources.levels != 1 ||
	    requested.extent.depth != info.extent.depth || requested.extent.width == 0 ||
	    requested.extent.height == 0 || requested.extent.width > info.extent.width ||
	    requested.extent.height > info.extent.height ||
	    (requested.extent.width == info.extent.width &&
	     requested.extent.height == info.extent.height) ||
	    source->backing.extent.width != info.extent.width ||
	    source->backing.extent.height != info.extent.height ||
	    !ImageViewOps::FormatsCompatible(info.pixel_format, requested.pixel_format)) {
		return id;
	}
	// FindTexture's refresh of an image it binds, for the image the copy is made from (an upload
	// moves its content serial, so the copy is made again).
	TouchImage(m_slot_images[id]);
	RefreshImage(id);
	source            = &m_slot_images[id];
	const auto width  = requested.extent.width;
	const auto height = requested.extent.height;
	const auto tick   = m_scheduler.CurrentTick();
	auto*      copies = &m_sub_extent_copies[id];
	size_t     index  = copies->size();
	for (size_t i = 0; i < copies->size(); i++) {
		if ((*copies)[i].width == width && (*copies)[i].height == height &&
		    m_slot_images.try_get((*copies)[i].copy) != nullptr) {
			index = i;
			break;
		}
	}
	if (index == copies->size()) {
		// At most 4 extents an image: the one used longest ago goes.
		if (copies->size() >= 4) {
			size_t oldest = 0;
			for (size_t i = 1; i < copies->size(); i++) {
				if ((*copies)[i].used < (*copies)[oldest].used) {
					oldest = i;
				}
			}
			const auto old = (*copies)[oldest].copy;
			copies->erase(copies->begin() + static_cast<std::ptrdiff_t>(oldest));
			if (old) {
				ReleaseUnregisteredImage(old);
			}
		}
		ImageInfo copy_info {};
		copy_info.pixel_format    = source->info.pixel_format;
		copy_info.guest_format    = source->info.guest_format;
		copy_info.type            = source->info.type;
		copy_info.extent          = {width, height, source->info.extent.depth};
		copy_info.resources       = source->info.resources;
		copy_info.pitch           = width;
		copy_info.bytes_per_block = source->info.bytes_per_block;
		copy_info.samples         = 1;
		copy_info.tile_mode       = Prospero::TileMode::kLinear;
		copy_info.bgra16          = source->info.bgra16;
		copy_info.mip_layout[0]   = {0, uint64_t {width} * height * source->info.bytes_per_block,
		                             width, height};
		const auto guest          = source->info.data.address;
		const auto source_width   = source->info.extent.width;
		const auto source_height  = source->info.extent.height;
		const auto copy_id        = InsertImage(copy_info);
		copies                    = &m_sub_extent_copies[id];
		copies->push_back({copy_id, width, height, UINT64_MAX, tick});
		index = copies->size() - 1;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("TextureCache: 0x%012" PRIx64 " %ux%u read as a %ux%u texture (same memory and "
			     "tiling): from a copy of that corner (KYTY_SUB_EXTENT_COPY)\n",
			     guest, source_width, source_height, width, height);
		}
	}
	auto& entry             = (*copies)[index];
	auto& copy              = m_slot_images[entry.copy];
	entry.used              = tick;
	copy.tick_accessed_last = tick;
	copy.frame_accessed_last = m_frame.load(std::memory_order_relaxed);
	if (entry.serial != m_slot_images[id].ContentSerial()) {
		// Within both images: the corner of the source the copy's extent covers.
		copy.CopyImage(m_slot_images[id]);
		entry.serial = m_slot_images[id].ContentSerial();
	}
	return entry.copy;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	// One placeholder per format and image dimensionality: a 1D or 3D view cannot use a 2D image.
	const auto type = desc.info.type == Prospero::ImageType::kColor1D ||
	                          desc.info.type == Prospero::ImageType::kColor3D
	                      ? desc.info.type
	                      : Prospero::ImageType::kColor2D;
	// KYTY_NULL_COMPARE_DEPTH: the D16 null image of a depth-compare binding holds 0.0 or 1.0.
	const bool compare_depth = NullCompareDepthEnabled() && desc.info.pixel_format == vk::Format::eD16Unorm &&
	                           desc.view_info.aspect == vk::ImageAspectFlagBits::eDepth;
	bool       compare_one   = false;
	if (compare_depth) {
		// Erlou NullCompareBinding: the depth that fails the S#'s comparison (reference OP texel; D16
		// clamps the reference to [0, 1]). 0.0 for NEVER/LESS/EQUAL/LEQUAL, 1.0 for GREATER/GEQUAL.
		// NOTEQUAL/ALWAYS (or both families) pass for some reference whatever the depth: Erlou exits,
		// ours logs once and keeps 0.0.
		constexpr uint32_t FailAtZero = (1u << 0u) | (1u << 1u) | (1u << 2u) | (1u << 3u);
		constexpr uint32_t FailAtOne  = (1u << 0u) | (1u << 4u) | (1u << 6u);
		const uint32_t     funcs      = t_null_compare_funcs;
		compare_one                   = funcs != 0u && (funcs & ~FailAtOne) == 0u;
		if (!compare_one && (funcs & ~FailAtZero) != 0u) {
			static std::atomic_bool logged {false};
			if (!logged.exchange(true)) {
				std::printf("Warning: KYTY_NULL_COMPARE_DEPTH: null image on a depth-compare binding with "
				            "compare functions 0x%02x (NOTEQUAL/ALWAYS or both families): bound as 0.0 "
				            "(logged once)\n",
				            funcs);
			}
		}
	}
	const auto format = (static_cast<uint64_t>(type) << 32u) |
	                    static_cast<uint64_t>(desc.info.pixel_format) |
	                    (static_cast<uint64_t>(compare_one ? 1u : 0u) << 40u);
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = type;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(format, id);
	const bool depth_format = info.pixel_format == vk::Format::eD16Unorm ||
	                          info.pixel_format == vk::Format::eD32Sfloat;
	if ((compare_depth || (NullTextureZeroEnabled() && !depth_format)) &&
	    info.pixel_format != vk::Format::eUndefined) {
		// Before, the texel was never written: an R32F null read (x, 0, 0, 1) with x undefined.
		auto& image   = m_slot_images[id];
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto sink = command.HelperSink();
		image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
		              sink.Identity());
		if (compare_depth) {
			const vk::ClearDepthStencilValue value {compare_one ? 1.0f : 0.0f, 0u};
			const vk::ImageSubresourceRange  range {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
			sink.clearDepthStencilImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
			                            &value, 1, &range);
		} else {
			const vk::ClearColorValue       zero {std::array<float, 4> {}};
			const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
			sink.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &zero, 1,
			                     &range);
		}
		image.NoteContentWrite();
		static std::atomic_bool logged {false};
		if (!logged.exchange(true)) {
			std::printf("KYTY_NULL_TEXTURE_ZERO / KYTY_NULL_COMPARE_DEPTH: null images cleared "
			            "(colour 0,0,0,0; compare depth by function)\n");
		}
	}
	return id;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	// A copy reads every level: non-resident levels must hold guest data first.
	RequireFullResidency(id);
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	// Converts guest D16 between a four-byte host depth aspect (D24/D32, stenciled) and two-byte
	// color or native D16 depth (upstream 0685bffb2: D16 <-> D24/D32 depth copies exited).
	const bool source_wide = DepthAspectTransferBytes(source.backing.format) == sizeof(uint32_t);
	if (source_wide ==
	    (DepthAspectTransferBytes(destination.backing.format) == sizeof(uint32_t))) {
		return false;
	}
	const bool source_depth   = source_wide;
	auto&      depth          = source_wide ? source : destination;
	auto&      color          = source_wide ? destination : source;
	const auto transfer_bytes = sizeof(uint32_t);
	const auto narrow_aspect  = color.info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
	                                                 : vk::ImageAspectFlagBits::eColor;
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	std::vector<vk::BufferImageCopy> depth_copies(layers);
	std::vector<vk::BufferImageCopy> color_copies(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		depth_copies[layer].bufferOffset      = depth_slice * layer;
		depth_copies[layer].bufferRowLength   = depth.info.pitch;
		depth_copies[layer].bufferImageHeight = depth.info.extent.height;
		depth_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
		depth_copies[layer].imageExtent       = depth.info.extent;
		color_copies[layer].bufferOffset      = color_slice * layer;
		color_copies[layer].bufferRowLength   = color.info.pitch;
		color_copies[layer].bufferImageHeight = color.info.extent.height;
		color_copies[layer].imageSubresource  = {narrow_aspect, 0, layer, 1};
		color_copies[layer].imageExtent       = color.info.extent;
	}

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

const char* TextureCache::TryDirectReinterpret(Image& destination, Image& source) {
	if (!DirectImageCopyEnabled() || source.backing.samples != destination.backing.samples) {
		return nullptr;
	}
	const bool source_depth = source.info.IsDepth();
	if (source_depth == destination.info.IsDepth()) {
		return nullptr;
	}
	const auto& depth = source_depth ? source : destination;
	const auto& color = source_depth ? destination : source;
	if (m_graphics.maintenance8_enabled && depth.backing.image_type == vk::ImageType::e2D &&
	    color.backing.image_type == vk::ImageType::e2D &&
	    Maintenance8CopyCompatible(depth.backing.format, color.backing.format)) {
		destination.CopyDepthColorImage(source);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyMaintenance8);
		return "maintenance8";
	}
	if (!DirectImageCopyShaderEnabled()) {
		return nullptr;
	}
	if (source_depth && m_blit_helper.SupportsDepthToColor32(source, destination)) {
		m_blit_helper.CopyDepthToColor32(source, destination);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyShaderDepthToColor);
		return "shader-depth-to-color";
	}
	if (!source_depth && m_blit_helper.SupportsColor32ToDepth(source, destination) &&
	    (destination.backing.format == vk::Format::eD32Sfloat || DirectImageCopyWideEnabled())) {
		m_blit_helper.CopyColor32ToDepth(source, destination);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyShaderColorToDepth);
		return "shader-color-to-depth";
	}
	return nullptr;
}

bool TextureCache::CopyImage(ImageId destination_id, ImageId source_id, const char* context) {
	ImageWatch(context != nullptr ? context : "image copy destination", m_slot_images[destination_id]);
	RefreshCopySource(source_id);
	// Levels the copy does not write must stay valid, and the result is GPU-owned.
	RequireFullResidency(destination_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		HangTrace::RecordTransfer(HangTrace::TransferKind::ImageCopy, "skipped-buffer-owned",
		                          context, destination.info.data.address,
		                          static_cast<uint32_t>(destination.backing.format),
		                          destination.info.extent.width, destination.info.extent.height,
		                          0, 0);
		return false;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	// Images whose storage differs by a colour transform (BGRA16 swap, 10-11-11) copy through the
	// tiler's transform (CopyImageWithBuffer); raw texel copies would keep the source bit order.
	const bool same_transform =
	    source.info.GetColorTransform() == destination.info.GetColorTransform();
	const bool direct_copy =
	    same_transform &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	// Lossless paths copy raw texel bits; the D16 path converts through unorm16.
	bool        lossless = true;
	const char* path     = nullptr;
	if (direct_copy) {
		destination.CopyImage(source);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyDirect);
		path = "direct";
	} else if (same_transform && (path = TryDirectReinterpret(destination, source)) != nullptr) {
	} else if (CopyD16(destination, source)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyD16);
		lossless = false;
		path     = "d16-convert";
	} else {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyViaBuffer);
		path = "via-buffer";
	}
	HangTrace::RecordTransfer(HangTrace::TransferKind::ImageCopy, path, context,
	                          destination.info.data.address,
	                          static_cast<uint32_t>(destination.backing.format),
	                          destination.info.extent.width, destination.info.extent.height,
	                          destination.info.data.size, 0);
	if (source.IsGpuModified()) {
		MarkImageGpuModified(destination);
		TakeOverOwnedBytes(destination, source);
		if (ReadbackLinearCopiesEnabled()) {
			TrackImageDownload(destination_id, destination);
		}
	}
	destination.ClearBufferModified();
	// Copies never include a stencil aspect, so combined depth/stencil images never compare equal.
	const auto has_stencil = [](vk::Format format) {
		return format == vk::Format::eD16UnormS8Uint || format == vk::Format::eD24UnormS8Uint ||
		       format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eS8Uint;
	};
	return lossless && !has_stencil(source.backing.format) &&
	       !has_stencil(destination.backing.format) &&
	       source.backing.extent == destination.backing.extent &&
	       source.backing.image_type == destination.backing.image_type &&
	       source.backing.mip_levels == destination.backing.mip_levels &&
	       source.backing.layers == destination.backing.layers;
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	RequireFullResidency(destination_id);
	// A partial copy into an image whose guest buffer was written by the GPU: take those bytes
	// first, or the rest of the image would publish stale contents (upstream dad900c61).
	if (m_slot_images[destination_id].IsBufferModified()) {
		RefreshImage(destination_id);
	}
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		MarkImageGpuModified(destination);
		TakeOverOwnedBytes(destination, source);
		if (ReadbackLinearCopiesEnabled()) {
			TrackImageDownload(destination_id, destination);
		}
	}
}

// KYTY_DEPTH_EXTENT_RECREATE=1 (live, default off; based on Senaxx/KytyPS5 4b7fb3274): see
// ResolveDepthOverlap. Off keeps the narrower cached image as before (and counts the case).
static Live::Switch g_depth_extent_recreate("KYTY_DEPTH_EXTENT_RECREATE", Live::ParseDefaultOff);

// How often ResolveDepthOverlap met a depth target asked for larger than the cached image: logged
// the first 8 times and then at every power of two, with the switch's answer.
static void NoteDepthExtentGrowth(const ImageInfo& requested, const ImageInfo& cached, bool recreated) {
	static std::atomic<uint64_t> count {0};
	const auto                   n = count.fetch_add(1, std::memory_order_relaxed) + 1;
	if (n <= 8 || (n & (n - 1)) == 0) {
		std::printf("DepthExtent: depth target 0x%016" PRIx64 " asked %ux%u over a cached %ux%u (pitch %u, "
		            "levels %u, layers %u): %s (%" PRIu64 " so far, KYTY_DEPTH_EXTENT_RECREATE)\n",
		            requested.data.address, requested.extent.width, requested.extent.height,
		            cached.extent.width, cached.extent.height, requested.pitch, requested.resources.levels,
		            requested.resources.layers, recreated ? "re-created" : "kept", n);
		std::fflush(stdout);
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	m_depth_path = DepthPath::None;
	if (!cached.info.IsDepth() && !requested.IsDepth()) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264, PPSA04288
	// A partial view retains the entire matching array layout. HTile belongs to
	// the depth source, independently of the data slices copied into color storage.
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.tile_mode == cached.info.tile_mode && requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    (cached.info.metadata.kind == ImageMetadataKind::None ||
	     cached.info.metadata.kind == ImageMetadataKind::Htile);
	// Depth bytes sampled through a color format no depth view provides (16_UINT, 8_8_UNORM)
	// need a color copy of the whole single-sample depth layout (upstream 0543c3d22, which
	// generalizes the former 16_UNORM -> 16_UINT case).
	const bool raw_depth_texture =
	    cached.info.IsDepth() && !requested.IsDepth() && !cached.info.HasStencil() &&
	    !requested.HasMetadata() && cached.backing.samples == 1 && requested.samples == 1 &&
	    requested.pitch == cached.info.pitch &&
	    (requested.resources == cached.info.resources || retain_cached_layout) &&
	    !IsSupportedSampledDepthFormat(cached.info.pixel_format, requested.pixel_format);
	// KYTY_IMAGE_CONTAINS=1 (Erlou93/KytyPS5-Yotei): capacity = levels AND layers, not the
	// lexicographic compare (which reused 7 levels x 9 layers for a 1 level x 12 layers request).
	const bool too_small = ImageContainsEnabled() ? !cached.info.resources.Contains(requested.resources)
	                                              : cached.info.resources < requested.resources;
	bool recreate = too_small || requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_depth_texture;
			// Astro Bot (PPSA21567) samples the D32 memory of its 3328x1872 depth target as
			// R16G16_SFLOAT: a colour alias is filled from the depth image.
			recreate |= !requested.HasStencil() && !cached.info.HasStencil() &&
			            NeedsColorAliasForSampledDepth(cached.info.pixel_format,
			                                           cached.info.bytes_per_block,
			                                           requested.pixel_format,
			                                           requested.bytes_per_block);
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	// KYTY_DEPTH_EXTENT_RECREATE (based on Senaxx/KytyPS5 4b7fb3274): a depth target asked for
	// wider than the cached depth image at the same address, pitch and height (the caller compared
	// BlockExtent) was given the narrower image, while the render area follows the requested extent:
	// the pass would render past the image's end. Counted whatever the switch says.
	const bool depth_extent_grows = binding == BindingType::DepthTarget && !recreate &&
	                                cached.info.IsDepth() &&
	                                (requested.extent.width > cached.info.extent.width ||
	                                 requested.extent.height > cached.info.extent.height);
	if (depth_extent_grows) {
		const bool enabled = g_depth_extent_recreate.On();
		NoteDepthExtentGrowth(requested, cached.info, enabled);
		if (enabled) {
			if (IsDepthGrowth(requested, cached.info) && cached.registered && !cached.depth_id &&
			    cached.backing.samples == 1) {
				// One level and layer: the grown image keeps depth and stencil (KYTY_DEPTH_GROW_RECREATE's copy).
				m_depth_path = DepthPath::Recreate;
				return GrowDepthTarget(requested, cached_id);
			}
			recreate = true;
		}
	}
	if (!recreate) {
		// A function of the two infos and the start-up switches; a growth the live switch left alone
		// is not replayable (not Same: NoteDepthOverlap marks the lookup impure).
		m_depth_path = depth_extent_grows ? DepthPath::Recreate : DepthPath::Same;
		return cached_id;
	}
	// Guests reuse one allocation as depth and as color (e.g. D32F <-> R32F) every frame. Keep one
	// image per interpretation alive and reuse it: FindImage copies the current owner's contents
	// into a non-owner alias (SyncAliasFromOwner) instead of destroying and recreating images.
	// A partial view that retains the cached array layout (fbf6ae400) gets that layout: an alias
	// made for a smaller prefix of the same layers is the same image.
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	}
	const bool keep_aliases = AliasAgeByFrames() && !(cached.info.resources < requested.resources);
	if (keep_aliases) {
		for (const auto other_id: FindImagesInRegion(requested.data.address, requested.data.size, false)) {
			const auto* other = m_slot_images.try_get(other_id);
			if (other == nullptr || other_id == cached_id || !other->registered || other->depth_id ||
			    other->backing.samples != cached.backing.samples ||
			    !SameBacking(other->info, info, true)) {
				continue;
			}
			// KYTY_IMG_DEPTHALIAS: the branch's only effect (TextureBindingMemo replays it).
			m_depth_path           = DepthPath::Alias;
			m_depth_inner_address  = requested.data.address;
			m_depth_inner_size     = requested.data.size;
			m_depth_rebind_applied = cached.binding.is_bound || cached.binding.is_target;
			if (cached.binding.is_bound || cached.binding.is_target) {
				cached.binding.needs_rebind = true;
				NoteImageStateChange();
			}
			return other_id;
		}
	}
	m_depth_path = DepthPath::Recreate;
	RefreshImage(cached_id);
	info.htile_initial_clear  = false;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
		NoteImageStateChange();
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			// KYTY_PRESERVE_STENCIL_RECREATE: a copy never includes the stencil aspect.
			const bool recreate_stencil = PreserveStencilRecreateEnabled() && cached.backing.samples == 1;
			if (recreate_stencil && cached.info.IsDepth() && cached.info.HasStencil() &&
			    !replacement.info.IsDepth() && SwapsStencilPlanes(cached)) {
				SaveRecreatedStencil(cached);
			}
			(void)CopyImage(replacement_id, cached_id, "depth-overlap");
			if (recreate_stencil && !cached.info.IsDepth()) {
				auto& made = m_slot_images[replacement_id];
				if (made.info.IsDepth() && made.info.HasStencil() && SwapsStencilPlanes(made)) {
					(void)RestoreRecreatedStencil(made);
				}
			}
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		// The replacement holds the source's contents: it owns what the source owned.
		RangeSet   converted;
		WriteClaim claim;
		if (AliasBytesEnabled() && !cached.OwnsAllBytes()) {
			cached.ForEachOwnedRange(replacement.info.data.address, replacement.info.data.size,
			                         [&](uint64_t begin, uint64_t end) {
				                         converted.Add(begin, end - begin);
			                         });
			claim.ranges = &converted;
		}
		CommitGpuWrite(replacement, claim);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	// KYTY_STENCIL_DEPTH_ALIAS: single-sample depth/stencil pairs too.
	const bool stencil_alias = StencilDepthAliasEnabled() && cached.backing.samples == 1 &&
	                           (cached.info.HasStencil() != replacement.info.HasStencil());
	if (keep_aliases && cached.backing.samples == replacement.backing.samples &&
	    ((!cached.info.HasStencil() && !replacement.info.HasStencil()) || stencil_alias)) {
		// The replacement now holds the newest contents; the cached interpretation stays alive
		// as a non-owner alias for the next switch back.
		if (replacement.IsGpuModified()) {
			RangeSet   owned;
			WriteClaim claim;
			if (AliasBytesEnabled() && replacement.OwnedSet() != nullptr) {
				owned        = *replacement.OwnedSet();
				claim.ranges = &owned;
			}
			CommitGpuWrite(replacement, claim);
		}
		return replacement_id;
	}
	FreeImage(cached_id, HangTrace::ImageFreeReason::DepthRecreate);
	return replacement_id;
}

static bool BlockSubresourceWritesEnabled(); // KYTY_BLOCK_SUBRESOURCE_WRITES, defined below

bool TextureCache::IsDepthGrowth(const ImageInfo& requested, const ImageInfo& cached) {
	// Same guest layout apart from the extent: with an equal pitch every texel of the smaller
	// image keeps its address in the larger one. One level and one layer only (a taller target
	// moves the start of every further slice).
	if (!requested.IsDepth() || !cached.IsDepth() ||
	    requested.data.address != cached.data.address || requested.data.size < cached.data.size ||
	    requested.pixel_format != cached.pixel_format ||
	    requested.guest_format != cached.guest_format ||
	    requested.bytes_per_block != cached.bytes_per_block || requested.samples != 1 ||
	    cached.samples != 1 || requested.HasStencil() != cached.HasStencil() ||
	    requested.tile_mode != cached.tile_mode || requested.type != cached.type ||
	    requested.IsVolume() || requested.pitch != cached.pitch ||
	    requested.resources != cached.resources || requested.resources.levels != 1 ||
	    requested.resources.layers != 1 || requested.extent.depth != cached.extent.depth ||
	    requested.GetColorTransform() != cached.GetColorTransform()) {
		return false;
	}
	return requested.extent.width >= cached.extent.width &&
	       requested.extent.height >= cached.extent.height &&
	       (requested.extent.width > cached.extent.width ||
	        requested.extent.height > cached.extent.height);
}

ImageId TextureCache::GrowDepthTarget(const ImageInfo& requested, ImageId cached_id) {
	// The cached image's native depth and stencil must be current before they are copied: CPU
	// writes to its depth or stencil bytes reach it first.
	RefreshCopySource(cached_id);
	bool stencil_gpu = false;
	for (const auto record: std::vector<ImageId>(m_slot_images[cached_id].stencil_ids)) {
		if (const auto* plane = m_slot_images.try_get(record); plane != nullptr && plane->registered) {
			RefreshImage(record);
			stencil_gpu |= m_slot_images[record].IsGpuModified();
		}
	}
	auto info                = requested;
	info.htile_initial_clear = false;
	const auto grown_id      = InsertImage(info);
	{
		auto& cached = m_slot_images[cached_id];
		auto& grown  = m_slot_images[grown_id];
		grown.usage  = cached.usage;
		if (cached.binding.is_bound || cached.binding.is_target) {
			cached.binding.needs_rebind = true;
			NoteImageStateChange();
		}
		// As FindImage does for a new image: bytes the GPU wrote to buffers or other images
		// own come from there.
		if (m_buffer_cache.HasGpuDirtyBytes(grown.live.address, grown.live.size) ||
		    (AliasBytesEnabled() && OtherImagesOwnBytes(grown.live, grown_id))) {
			grown.MarkBufferModified();
		}
	}
	// Guest bytes over the whole image (the texels the cached image never had included).
	InitializeImage(grown_id, RefreshIntent::Read);
	ImageId record_id {};
	if (requested.HasStencil() && requested.stencil.Valid()) {
		// The stencil plane's guest bytes go into the new stencil aspect now, before the copy below
		// (a record made later by FindDepthTarget would upload them over the copied stencil).
		auto& grown                = m_slot_images[grown_id];
		grown.stencil_subresources = {0, requested.resources.levels, 0, requested.resources.layers};
		grown.info.stencil         = requested.stencil;
		record_id                  = AssociateStencil(grown_id, requested.stencil);
		RefreshImage(record_id, RefreshIntent::Read);
	}
	auto& cached = m_slot_images[cached_id];
	auto& grown  = m_slot_images[grown_id];
	TrackImage(grown_id);
	PrepareImageCopy(grown);
	if (cached.backing.format == grown.backing.format) {
		// Depth and stencil the GPU wrote, over the cached extent.
		grown.CopyDepthStencilImage(cached);
	} else {
		grown.CopyImage(cached);
	}
	HangTrace::RecordTransfer(HangTrace::TransferKind::ImageCopy, "depth-grow", "depth-grow",
	                          grown.info.data.address, static_cast<uint32_t>(grown.backing.format),
	                          grown.info.extent.width, grown.info.extent.height,
	                          grown.info.data.size, 0);
	if (cached.IsGpuModified()) {
		MarkImageGpuModified(grown);
		TakeOverOwnedBytes(grown, cached);
	}
	grown.ClearBufferModified();
	if (record_id && stencil_gpu) {
		// The copied stencil is newer than the stencil plane's guest bytes.
		TrackImage(record_id);
		CommitGpuWrite(m_slot_images[record_id]);
	}
	if (static std::atomic<uint32_t> logged {0}; logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		LOGF("TextureCache: depth target 0x%016" PRIx64 " grew %ux%u -> %ux%u (pitch %u%s%s)\n",
		     requested.data.address, cached.info.extent.width, cached.info.extent.height,
		     requested.extent.width, requested.extent.height, requested.pitch,
		     cached.IsGpuModified() ? ", GPU depth copied" : "",
		     stencil_gpu ? ", GPU stencil copied" : "");
	}
	FreeImage(cached_id, HangTrace::ImageFreeReason::DepthRecreate);
	return grown_id;
}

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached         = *owner;
	const bool safe_to_delete = OverlapSafeToDelete(cached);
	// KYTY_IMG_OVERLAP: a branch decided by safe_to_delete is pure while it stays false (the memo
	// re-checks it live for these candidates); taken (a free, or the keep window) it is not.
	const auto note_timed = [&] {
		auto& lookup = m_last_overlap;
		if (safe_to_delete || lookup.timed_count >= OverlapLookup::MaxTimed) {
			NoteOverlapImpure(safe_to_delete ? OverlapLookup::WhyTimedTaken : OverlapLookup::WhyTimedMany);
		} else {
			lookup.timed[lookup.timed_count++] = cached_id;
		}
	};

	// KYTY_DEPTH_GROW_RECREATE: before the pitch/height comparison below, which takes a wider
	// cached depth image as it is and never sees a taller one.
	if (binding == BindingType::DepthTarget && IsDepthGrowth(requested, cached.info) &&
	    cached.registered && !cached.depth_id && cached.backing.samples == 1) {
		NoteOverlapImpure(OverlapLookup::WhyDepth);
		if (m_depth_grow_recreate) {
			if (merged_id) {
				// A larger image was already chosen for this binding: an older, smaller one at the
				// same address must not replace it.
				return {merged_id};
			}
			return {GrowDepthTarget(requested, cached_id)};
		}
		if (static std::atomic<uint64_t> logged {0}; logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("TextureCache: depth target 0x%016" PRIx64 " grows %ux%u -> %ux%u (pitch %u); "
			     "KYTY_DEPTH_GROW_RECREATE is off, the smaller image is kept\n",
			     requested.data.address, cached.info.extent.width, cached.info.extent.height,
			     requested.extent.width, requested.extent.height, requested.pitch);
		}
		// On the console too, counted (the LOGF above goes to the emulator log only).
		static std::atomic<uint64_t> grows {0};
		if (const auto n = grows.fetch_add(1, std::memory_order_relaxed) + 1; n <= 8 || (n & (n - 1)) == 0) {
			std::printf("DepthGrow: depth target 0x%016" PRIx64 " grows %ux%u -> %ux%u (pitch %u), "
			            "KYTY_DEPTH_GROW_RECREATE off (%" PRIu64 " so far)\n",
			            requested.data.address, cached.info.extent.width, cached.info.extent.height,
			            requested.extent.width, requested.extent.height, requested.pitch, n);
			std::fflush(stdout);
		}
	}

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			NoteDepthOverlap(cached_id);
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			note_timed(); // a time-dependent decision (safe_to_delete)
			if (safe_to_delete) {
				FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapLayout);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		NoteOverlapImpure(OverlapLookup::WhyMerge); // an expansion or a merge
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			NoteOverlapImpure(OverlapLookup::WhyExpand);
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapMipMerge);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address) {
		note_timed(); // a time-dependent decision (safe_to_delete, then the keep window)
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		if (KeepOverlappedImage(cached, m_frame.load(std::memory_order_relaxed))) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureOverlapKeeps);
		} else {
			FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapStale);
		}
	}
	return {merged_id};
}

void TextureCache::NoteDepthOverlap(ImageId cached_id) {
	// KYTY_IMG_DEPTHALIAS: which ResolveDepthOverlap answers keep the lookup replayable.
	//  Same: the outcome is a function of the request, the cached image's fixed fields (info, backing
	//    sample count) and start-up switches; no effect.
	//  Alias: additionally of the inner region scan (FindImagesInRegion over the request's range:
	//    owner lists, registered flags, stencil associations - all of which move the pages' structure
	//    versions - and the found image's fixed fields); its one effect is listed for the replay.
	//  Recreate: inserts, copies, frees: impure.
	auto& lookup = m_last_overlap;
	if (m_depth_path == DepthPath::Same) {
		lookup.depth_class = true;
		return;
	}
	ImagePageTable::PageRange inner {};
	if (m_depth_path == DepthPath::Alias && lookup.rebind_count < OverlapLookup::MaxRebinds &&
	    ImagePageTable::TryGetPageRange(m_depth_inner_address, m_depth_inner_size, inner)) {
		lookup.depth_class                         = true;
		lookup.rebind[lookup.rebind_count]         = cached_id;
		lookup.rebind_applied[lookup.rebind_count] = m_depth_rebind_applied;
		lookup.rebind_count++;
		lookup.span_first = std::min(lookup.span_first, inner.first);
		lookup.span_last  = std::max(lookup.span_last, inner.last_exclusive);
		return;
	}
	NoteOverlapImpure(OverlapLookup::WhyDepth);
}

bool TextureCache::BlockSubwritesOn() {
	return BlockSubresourceWritesEnabled();
}

bool TextureCache::AliasSyncSkipOn() {
	return AliasSyncSkipEnabled();
}

void TextureCache::NoteOverlapImpure(uint8_t why) {
	if (!m_overlap_impure) {
		m_last_overlap.impure_why = why;
	}
	m_overlap_impure = true;
}

bool TextureCache::OverlapSafeToDelete(const Image& cached) const {
	const auto current_tick  = m_scheduler.CurrentTick();
	const auto current_frame = m_frame.load(std::memory_order_relaxed);
	return current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	       (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                   AliasFramesBeforeRemoval);
}

bool TextureCache::KeepOverlappedImage(const Image& cached, uint64_t current_frame) const {
	// A partially overlapped sampled texture stays valid: it watches its own pages and a later
	// use refreshes (at most) its written chunks. Freeing it only bounds the number of live
	// aliases; streamed textures that return to their slot are otherwise recreated and fully
	// uploaded. Keep only clean texture-only images used recently, and never in crowded ranges
	// (each live alias adds a page watcher; the tracker allows 127 per page).
	return m_overlap_keep_frames != 0 && !m_overlap_crowded && cached.ChunkTracked() &&
	       !cached.IsGpuModified() && !cached.IsBufferModified() && !cached.depth_id &&
	       !cached.usage.render_target && !cached.usage.depth_target && !cached.usage.storage &&
	       !cached.usage.video_out && !cached.binding.is_target &&
	       current_frame - std::min(current_frame, cached.frame_accessed_last) <=
	           m_overlap_keep_frames;
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      expanded    = m_slot_images[expanded_id];
	auto&      source      = m_slot_images[source_id];
	// A block-compressed image keeps its own guest layout and texture transfers: it does not
	// inherit render-target or storage usage from the non-block image it replaces (upstream
	// 0d4f99335, reused BC5 textures).
	if (!info.IsBlock() || source.info.IsBlock()) {
		expanded.usage = source.usage;
	}
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
		NoteImageStateChange();
	}
	// The copy below supplies the source's bytes (and their ownership): the refresh must not move
	// them into the buffer first.
	InitializeImage(expanded_id, RefreshIntent::Write);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		(void)CopyImage(expanded_id, source_id, "expand");
	}
	m_freeing_replaced = true;
	FreeImage(source_id, HangTrace::ImageFreeReason::Expand);
	m_freeing_replaced = false;
	return expanded_id;
}

TextureCache::TextureTransfer
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	return BuildTextureTransfer(image.info, image.backing.samples, binding, direction);
}

TextureCache::TextureTransfer TextureCache::BuildTextureTransfer(const ImageInfo& info,
                                                                 uint32_t backing_samples,
                                                                 BindingType binding,
                                                                 TransferDirection direction) {
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	auto        format           = info.guest_format;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	bool        allow_depth_tile = upload;
	const char* owner            = "TextureCache readback";

	TextureTransfer transfer;
	// Sampled BGRA16 uploads keep the guest channel order (our layout); every other transform
	// (10-11-11 storage) applies in both directions.
	transfer.color_transform = info.GetColorTransform();
	if (transfer.color_transform == ColorTransform::SwapBgra16 && upload && !render_target &&
	    !video_out) {
		transfer.color_transform = ColorTransform::None;
	}
	if (render_target) {
		format = ImageOps::RenderTargetTransferFormat(info.bytes_per_block);
	}
	if (video_out) {
		allow_depth_tile = false;
	} else if (render_target || binding == BindingType::Storage) {
		allow_depth_tile = true;
	}
	if (upload) {
		if ((render_target || video_out) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || backing_samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		owner = "TextureCache";
		if (render_target) {
			owner = "RenderTarget";
		} else if (binding == BindingType::Storage) {
			owner = "StorageTextureCache";
		} else if (video_out) {
			if (info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("TextureCache: invalid color-attachment upload\n");
			}
			layers = info.resources.layers;
			volume = false;
			owner  = "VideoOut";
		}
	}

	if (!upload && !TextureUploadLayoutSupported(format, info.extent.width, info.extent.height,
	                                             info.resources.levels, layers, info.tile_mode,
	                                             allow_depth_tile, volume)) {
		// Not downloadable (e.g. a colour view of depth-tiled memory): report an invalid transfer
		// so callers keep their exact fallback instead of stopping the emulator.
		return transfer;
	}
	transfer.layout  = TextureCalcUploadLayout(format, info.extent.width, info.extent.height,
	                                       info.resources.levels, layers, info.tile_mode,
	                                       info.data.size, allow_depth_tile, volume, owner);
	transfer.regions = TextureBuildImageCopies(transfer.layout);
	if (info.IsDepth()) {
		for (auto& region: transfer.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (transfer.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, transfer.regions, transfer.layout,
		                              info.resources.levels, transfer.tiles)) {
			return transfer;
		}
	}
	transfer.valid = true;
	return transfer;
}

TextureCache::ImageDownload TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	ImageDownload transfer {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return transfer;
	}
	if (transfer.depth_target) {
		transfer.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		// KYTY_SKIP_INVALID (usa-ratt/KytyPS5-GT7 28bed064f): a depth image whose rows (pitch x
		// height) do not fit in its share of its range is not downloaded (DownloadDepth would end
		// the emulator); it stays until it is safe to free, as an unsupported one does.
		if (transfer.valid && Common::SkipInvalidEnabled()) {
			const uint64_t guest_slice =
			    static_cast<uint64_t>(info.pitch) * info.extent.height * info.bytes_per_block;
			transfer.valid = guest_slice <= info.data.size / info.resources.layers;
		}
		return transfer;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return transfer;
	}
	transfer.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	transfer.valid   = transfer.texture.valid;
	return transfer;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	ImageWatch("image upload", image);
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	};

	if (binding != BindingType::DepthTarget) {
		const auto& info = image.info;
		auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(transfer.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		// Resident levels only: the source holds the resident prefix (image.live) of the chain.
		RestrictToResidentLevels(image, transfer);
		// KYTY_TILER_IMAGE_DIRECT: detile straight into the image, no linear scratch and copy.
		if (!transfer.tiles.empty() && transfer.color_transform == ColorTransform::None &&
		    m_tiler.DetileToImage(destination, source.Handle(), source_offset, image.live.size,
		                          transfer.LinearSize(), transfer.tiles, transfer.regions)) {
			return;
		}
		TileManager::Result linear {source.Handle(), source_offset, image.live.size};
		if (!transfer.tiles.empty()) {
			// The detile pass applies the colour transform.
			linear = m_tiler.Detile(source.Handle(), source_offset, image.live.size,
			                        transfer.LinearSize(), transfer.tiles, transfer.color_transform);
		} else if (transfer.color_transform != ColorTransform::None) {
			linear = m_tiler.TransformColor(linear, transfer.color_transform, true);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
	}
	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	// KYTY_SKIP_INVALID (usa-ratt/KytyPS5-GT7 8c9355135): a slice its range cannot hold (an image
	// made with more layers than its range has room for) is left as it is rather than ending the
	// emulator in the tiler.
	if (const uint64_t slice_bytes =
	        static_cast<uint64_t>(info.pitch) * info.extent.height * info.bytes_per_block;
	    (info.extent.height == 0 || slice_bytes > full_slice_size) && Common::SkipInvalidEnabled()) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("TextureCache: depth %s 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u pitch %u, %u bytes a texel, %u layers: a slice needs 0x%" PRIx64
			     " bytes and has 0x%" PRIx64 "; not uploaded\n",
			     image.depth_id ? "stencil" : "image", info.data.address, info.data.size,
			     info.extent.width, info.extent.height, info.pitch, info.bytes_per_block, layers,
			     slice_bytes, full_slice_size);
		}
		return;
	}
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		for (uint32_t layer = 0; layer < layers; layer++) {
			copies[layer].bufferOffset = transfer_slice * layer;
		}
	}
	upload(copies, linear);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureUploadBytes, linear.size);
}

static bool HashGuestBytes(uint64_t address, uint64_t size, uint64_t& hash);

void TextureCache::InitializeImage(ImageId id, RefreshIntent intent) {
	auto& image = m_slot_images[id];
	ImageWatch("image initialize", image);
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	// KYTY_DISCARDED_TARGET_SKIP: a new target over a just-discarded GPU-written target holds
	// nothing the game wrote.
	if (upload && DiscardedTargetSkipEnabled() && !image.IsBufferModified() && !image.WasEverUploaded() &&
	    IsTargetImage(image) && !m_buffer_cache.IsRegionGpuModified(image.info.data.address, image.info.data.size) &&
	    WasDiscardedTarget(image)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureDiscardedTargetSkips);
		image.RefreshComplete();
		return;
	}
	// KYTY_PENDING_FILLS: a fresh target inside a GPU fill starts as the fill.
	if (upload && !image.WasEverUploaded() && BufferCache::PendingFillsEnabled() &&
	    ClearFromKnownFill(id)) {
		return;
	}
	bool       guest_refresh = false;
	// KYTY_IMAGE_PARTIAL_GPU_WRITE: only GPU buffer writes over known ranges changed the image's
	// bytes; it moves bytes through the buffer for those ranges only (UploadGpuWrittenRanges).
	const bool gpu_partial_candidate = upload && GpuWritePartialCandidate(image);
	if (upload && !gpu_partial_candidate) {
		MaterializeForRebuild(id, intent);
	}
	if (upload) {
		Profiler::ScopedFrameWait upload_time(Profiler::FrameWait::TextureUpload);
		const CpGaps::Scope       upload_gap(CpGaps::Cat::TextureUpload);
		// Attribution only: why this refresh happens and how much of the image was dirtied.
		const bool  first  = !image.WasEverUploaded();
		const char* reason = image.IsBufferModified()
		                         ? (first ? "first-use-gpu-buffer" : "gpu-buffer-write")
		                     : first                     ? "first-use"
		                     : image.residency_refresh   ? "resident-extend"
		                     : image.DirtyFromEdgeHash() ? "cpu-edge-hash"
		                                                 : "cpu-write";
		const auto span = image.DirtySpanBytes();
		// For GPU buffer writes: which kind of recorded write last touched the dirty bytes.
		const char* writer =
		    HangTrace::Enabled() && image.IsBufferModified()
		        ? HangTrace::LastGpuWriteKind(span != 0 ? image.DirtySpanBegin()
		                                                : image.info.data.address)
		        : nullptr;
		const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
		// A refresh from guest memory of a sampled-only chunk-tracked image leaves native
		// contents equal to guest bytes everywhere; later refreshes can then be partial.
		guest_refresh = image.ChunkTracked() && !image.IsBufferModified() &&
		                !image.IsGpuModified() && !image.depth_id && binding == BindingType::Texture;
		PartialUploadResult partial {};
		bool                gpu_partial = false;
		if (gpu_partial_candidate) {
			partial     = UploadGpuWrittenRanges(id, intent);
			gpu_partial = partial.done;
			if (!partial.done) {
				// Nothing was recorded or moved: the whole rebuild as without the switch.
				MaterializeForRebuild(id, intent);
			}
		} else if (image.HasKeptDirty()) {
			// KYTY_IMAGE_GPU_KEEP: only the subresources over the replaced bytes.
			partial = UploadKeptRanges(image);
			if (!partial.done) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::TextureKeptFallbacks);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
					LOGF("TextureCache: kept GPU image 0x%016" PRIx64 " size=0x%" PRIx64
					     " refreshes whole (no partial upload for it)\n",
					     image.info.data.address, image.info.data.size);
				}
			}
		} else if (image.ChunkTracked() && image.chunks.dirty_count != 0 && !first) {
			partial = TryPartialUpload(image);
			if (!partial.done) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialFallbacks);
			}
		}
		// Sources cover the resident prefix only (image.live; the whole range when fully
		// resident): the bytes of levels no view can sample are neither read nor uploaded.
		const auto live = image.live;
		// KYTY_LIGHT_GRID_CAP: a light grid is uploaded here, from bytes it can repair.
		const bool light_grid = LightGridCapEnabled() && image.FullyResident() &&
		                        IsLightGridShape(image.info);
		// KYTY_GUEST_HASH_IMAGES_KB: the guest bytes are hashed before they are staged (a CPU write
		// between the two leaves the hash older than the upload, so the next comparison uploads once
		// more rather than missing it), and the upload takes them whatever buffer covers them.
		uint64_t   guest_hash = 0;
		const bool hash_guest = !partial.done && image.FullyResident() && ChecksGuestContent(image) &&
		                        HashGuestBytes(live.address, live.size, guest_hash);
		image.content_hash_valid = false;
		if (!partial.done && (light_grid || hash_guest || !TryAsyncFullUpload(image))) {
			const bool cached_source = m_buffer_cache.IsRegionRegistered(live.address, live.size) ||
			                           m_buffer_cache.IsRegionGpuModified(live.address, live.size);
			Profiler::CountFrameEvent(cached_source ? Profiler::FrameEvent::TextureUploadBytesBuffer
			                                        : Profiler::FrameEvent::TextureUploadBytesStaging,
			                          live.size);
			const auto [source, source_offset] =
			    hash_guest ? m_buffer_cache.StageGuestBytesForImage(live.address, live.size)
			               : m_buffer_cache.ObtainBufferForImage(live.address, live.size);
			if (source == nullptr) {
				EXIT("TextureCache: failed to obtain image upload source\n");
			}
			if (light_grid && (hash_guest || !cached_source)) {
				RepairLightGridUpload(image, *source, source_offset);
			}
			UploadImage(image, *source, source_offset);
			if (hash_guest) {
				image.content_hash             = guest_hash;
				image.content_hash_valid       = true;
				image.content_check_submission = SyncEpoch::CurrentSubmission();
			}
		}
		const auto uploaded = partial.done ? partial.bytes : live.size;
		image.residency_refresh = false;
		if (!image.FullyResident()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidentBytesSkipped,
			                          image.info.data.size - live.size);
			if (m_residency == ResidencyMode::Poison && !image.residency_poisoned) {
				PoisonNonResidentLevels(image);
			}
		}
		image.ClearBufferModified();
		if (m_partial_gpu_write != 0) {
			NoteRefreshSync(id, gpu_partial);
		}
		if (m_exact_range_mode == 1) {
			// KYTY_IMAGE_EXACT_RANGE_INVALIDATE: a refresh from memory also covers kept bytes.
			NoteExactRangeRewrite(image);
		}
		image.NoteUpload();
		image.ClearDirtySpan();
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageUploads);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageUploadBytes, uploaded);
		if (partial.done) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialUploads);
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialUploadBytes, uploaded);
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialSkippedBytes,
			                          live.size - std::min(live.size, uploaded));
		}
		if (HangTrace::Enabled()) {
			// Partial refreshes: bytes are the tiled guest bytes refreshed, span_bytes the dirty
			// chunk bytes that caused the refresh.
			// Partially resident textures: detail "texture-resident", bytes = resident bytes.
			HangTrace::RecordTransfer(HangTrace::TransferKind::ImageUpload,
			                          gpu_partial    ? "gpu-buffer-partial"
			                          : partial.done ? "cpu-write-partial"
			                                         : reason,
			                          writer != nullptr          ? writer
			                          : image.depth_id           ? "stencil-plane"
			                          : !image.FullyResident()   ? "texture-resident"
			                                                     : UploadBindingName(binding),
			                          image.info.data.address,
			                          static_cast<uint32_t>(image.backing.format),
			                          image.info.extent.width, image.info.extent.height, uploaded,
			                          partial.done ? partial.dirty_bytes : span);
		}
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	if (image.ChunkTracked()) {
		auto& chunks = image.chunks;
		if (upload && chunks.whole_released) {
			// A whole release says nothing about how much was written; probe periodically.
			if (++chunks.whole_cycles >= 8) {
				chunks.whole_cycles = 0;
				chunks.full_streak  = 0;
			}
		} else if (upload && chunks.dirty_count != 0) {
			const bool nearly_all = uint64_t {chunks.dirty_count} * 8 >= uint64_t {chunks.count} * 7;
			chunks.full_streak    = nearly_all ? std::min(chunks.full_streak + 1, 255u) : 0u;
		}
		chunks.whole_released = false;
		image.ClearChunkDirty();
		if (guest_refresh) {
			image.SetPartialValid(true);
			if (m_partial_verify) {
				RecordChunkHashes(image);
			}
		}
	}
}

TextureCache::PartialUploadResult TextureCache::TryPartialUpload(Image& image) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	PartialUploadResult result;
	const auto&         info = image.info;
	auto&               chunks = image.chunks;
	if (!image.PartialValid() || image.IsGpuModified() || image.IsBufferModified() ||
	    image.depth_id || info.samples != 1 || image.backing.samples != 1 || info.IsVolume() ||
	    !info.IsTiled() || info.IsDepth() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed ||
	    chunks.dirty_count == 0 || chunks.dirty_count >= chunks.count ||
	    UploadBinding(image) != BindingType::Texture) {
		return result;
	}
	const auto transfer = BuildTextureTransfer(image, BindingType::Texture, TransferDirection::Upload);
	if (!transfer.valid || transfer.color_transform != ColorTransform::None || transfer.tiles.empty() ||
	    transfer.tiles.size() != transfer.regions.size()) {
		return result;
	}
	if (m_partial_verify && !VerifyCleanChunks(image)) {
		return result;
	}

	// Select the parts of the chain whose tiled guest bytes overlap a dirty chunk. A detile
	// dispatch reads only its tile's [tiled_offset, tiled_offset + tiled_size) range, so every
	// changed byte that affects a texel lies in a selected part. Outside the mip tail, 2D block
	// families address a texel as (block row * blocks per row + block column) * block size plus
	// an offset that depends only on the texel's position inside its block, so a run of block
	// rows is an independent sub-surface: its own tiled range, texel rows and copy region.
	const uint64_t                   base = info.data.address;
	std::vector<GpuTileInfo>         tiles;
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         linear_total = 0;
	uint64_t                         tiled_bytes  = 0;
	const auto add = [&](GpuTileInfo tile, vk::BufferImageCopy region, uint64_t linear_size) {
		linear_total        = Common::AlignUp(linear_total, 256);
		tile.linear_offset  = linear_total;
		tile.linear_size    = linear_size;
		region.bufferOffset = linear_total;
		linear_total += linear_size;
		tiled_bytes += tile.tiled_size;
		tiles.push_back(tile);
		regions.push_back(region);
	};
	const auto texel_height = std::max(transfer.layout.surface.texture.texel_height, 1u);
	for (size_t index = 0; index < transfer.tiles.size(); index++) {
		const auto& tile   = transfer.tiles[index];
		const auto& region = transfer.regions[index];
		if (region.imageSubresource.mipLevel < image.resident_first || tile.tiled_size == 0 ||
		    !image.ChunkRangeDirty(base + tile.tiled_offset, tile.tiled_size)) {
			continue;
		}
		TileBlockLayout block {};
		const bool      family_bands = tile.family == TileBlockFamily::Standard256B ||
		                          tile.family == TileBlockFamily::Standard4KB ||
		                          tile.family == TileBlockFamily::Standard64KB ||
		                          tile.family == TileBlockFamily::Prt64KB;
		const bool bands = m_partial_bands && family_bands && !tile.tail && tile.depth == 1 &&
		                   region.imageOffset.y == 0 &&
		                   TileGetBlockLayout(tile.family, tile.bytes_per_element, block) &&
		                   block.block_depth == 1 && block.block_width != 0 &&
		                   block.block_height != 0 && block.block_size != 0;
		const uint32_t tiled_width = tile.tiled_width != 0 ? tile.tiled_width : tile.pitch;
		const uint64_t row_bytes =
		    bands ? static_cast<uint64_t>((tiled_width + block.block_width - 1) / block.block_width) *
		                block.block_size
		          : 0;
		const uint32_t rows =
		    bands ? (tile.height + block.block_height - 1) / block.block_height : 0;
		if (!bands || row_bytes == 0 || rows <= 1 || rows * row_bytes > tile.tiled_size) {
			add(tile, region, tile.linear_size);
			continue;
		}
		const uint64_t pitch_bytes = static_cast<uint64_t>(tile.pitch) * tile.bytes_per_element;
		for (uint32_t row = 0; row < rows;) {
			const auto row_dirty = [&](uint32_t r) {
				return image.ChunkRangeDirty(base + tile.tiled_offset + r * row_bytes, row_bytes);
			};
			if (!row_dirty(row)) {
				row++;
				continue;
			}
			uint32_t end_row = row + 1;
			while (end_row < rows && row_dirty(end_row)) {
				end_row++;
			}
			const uint32_t y0   = row * block.block_height;
			const uint32_t y1   = std::min(end_row * block.block_height, tile.height);
			GpuTileInfo    band = tile;
			band.tiled_offset   = tile.tiled_offset + row * row_bytes;
			band.tiled_size     = (end_row - row) * row_bytes;
			band.height         = y1 - y0;
			band.tiled_height   = (end_row - row) * block.block_height;
			auto band_region    = region;
			const uint32_t texel_y0 = y0 * texel_height;
			const uint32_t texel_y1 = std::min(y1 * texel_height, region.imageExtent.height);
			band_region.imageOffset.y      = static_cast<int32_t>(texel_y0);
			band_region.imageExtent.height = texel_y1 - texel_y0;
			add(band, band_region, static_cast<uint64_t>(band.height) * pitch_bytes);
			row = end_row;
		}
	}
	if (tiles.empty()) {
		// The writes touched only bytes no texel is decoded from (gaps between levels).
		result.done        = true;
		result.dirty_bytes = static_cast<uint64_t>(chunks.dirty_count) << chunks.shift;
		return result;
	}
	// Nearly everything is dirty: the plain refresh needs fewer dispatches for the same bytes.
	const auto live = image.live;
	if (tiled_bytes >= live.size - live.size / 8) {
		return result;
	}

	// Source: an existing cache buffer for this range (synchronized page by page) or the
	// canonical GPU-written bytes; otherwise only the selected tiled ranges of guest memory.
	vk::Buffer tiled_buffer   = nullptr;
	uint64_t   tiled_offset   = 0;
	uint64_t   tiled_capacity = 0;
	if (m_buffer_cache.IsRegionRegistered(base, live.size) ||
	    m_buffer_cache.IsRegionGpuModified(base, live.size)) {
		const auto [source, source_offset] = m_buffer_cache.ObtainBufferForImage(base, live.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		tiled_buffer   = source->Handle();
		tiled_offset   = source_offset;
		tiled_capacity = live.size;
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureUploadBytesBuffer, tiled_bytes);
	} else {
		struct SourceRange {
			uint64_t offset;
			uint64_t size;
			uint64_t packed;
		};
		std::vector<SourceRange> ranges;
		uint64_t                 packed_total = 0;
		for (auto& tile: tiles) {
			const auto found = std::find_if(ranges.begin(), ranges.end(), [&](const SourceRange& r) {
				return r.offset == tile.tiled_offset && r.size == tile.tiled_size;
			});
			if (found != ranges.end()) {
				tile.tiled_offset = found->packed;
				continue;
			}
			packed_total = Common::AlignUp(packed_total, 256);
			ranges.push_back({tile.tiled_offset, tile.tiled_size, packed_total});
			tile.tiled_offset = packed_total;
			packed_total += tile.tiled_size;
		}
		auto& staging = m_staging_copier ? StagingRing()
		                                 : m_buffer_cache.GetUtilityBuffer(MemoryUsage::Upload);
		auto [mapped, offset] = staging.Map(packed_total, 256);
		if (mapped == nullptr) {
			return result;
		}
		Profiler::CountFrameEvent(m_staging_copier ? Profiler::FrameEvent::TextureUploadBytesAsync
		                                           : Profiler::FrameEvent::TextureUploadBytesStaging,
		                          packed_total);
		if (m_staging_copier) {
			std::vector<StagingCopier::Range> copies;
			copies.reserve(ranges.size());
			for (const auto& range: ranges) {
				copies.push_back({base + range.offset, mapped + range.packed, range.size});
			}
			staging.Commit();
			m_staging_copier->Enqueue(std::move(copies), &staging, offset, packed_total);
		} else {
			for (const auto& range: ranges) {
				if (!LibKernel::Memory::TryReadBacking(base + range.offset, mapped + range.packed,
				                                       range.size) &&
				    !LibKernel::Memory::TryReadSparseBacking(base + range.offset,
				                                          mapped + range.packed, range.size) &&
				    !ReadPartlyUnmappedImageBacking(base + range.offset, mapped + range.packed,
				                                    range.size)) {
					EXIT("TextureCache: failed to read mapped guest image backing\n");
				}
			}
			staging.Commit();
		}
		tiled_buffer   = staging.Handle();
		tiled_offset   = offset;
		tiled_capacity = packed_total;
	}
	if (!m_tiler.DetileToImage(image, tiled_buffer, tiled_offset, tiled_capacity, linear_total,
	                           tiles, regions)) {
		const auto linear =
		    m_tiler.Detile(tiled_buffer, tiled_offset, tiled_capacity, linear_total, tiles);
		for (auto& region: regions) {
			region.bufferOffset += linear.offset;
		}
		image.Upload(regions, linear.buffer, linear.offset, linear.size);
	}
	result.done        = true;
	result.bytes       = tiled_bytes;
	result.dirty_bytes = static_cast<uint64_t>(chunks.dirty_count) << chunks.shift;
	return result;
}

bool TextureCache::TryAsyncFullUpload(Image& image) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	const auto& info = image.info;
	if (!m_staging_copier || image.depth_id || image.IsBufferModified() || info.samples != 1 ||
	    image.backing.samples != 1 || !info.IsTiled() || info.IsDepth() ||
	    info.metadata.compression != VideoOutCompression::Uncompressed ||
	    UploadBinding(image) != BindingType::Texture) {
		return false;
	}
	// Same source choice as ObtainBufferForImage: cache buffers and GPU-written bytes stay on
	// their synchronous paths; only plain guest memory is copied by the worker.
	const auto base = image.live.address;
	const auto size = image.live.size;
	if (m_buffer_cache.IsRegionRegistered(base, size) ||
	    m_buffer_cache.IsRegionGpuModified(base, size)) {
		return false;
	}
	auto transfer = BuildTextureTransfer(image, BindingType::Texture, TransferDirection::Upload);
	if (!transfer.valid || transfer.color_transform != ColorTransform::None || transfer.tiles.empty()) {
		return false;
	}
	RestrictToResidentLevels(image, transfer);
	auto& staging         = StagingRing();
	auto [mapped, offset] = staging.Map(size, 256);
	if (mapped == nullptr) {
		return false;
	}
	staging.Commit();
	Profiler::CountFrameEvent(Profiler::FrameEvent::TextureUploadBytesAsync, size);
	m_staging_copier->Enqueue({{base, mapped, size}}, &staging, offset, size);
	// The detile barrier includes host writes; the submission waits for the copy on the GPU.
	if (m_tiler.DetileToImage(image, staging.Handle(), offset, size, transfer.LinearSize(),
	                          transfer.tiles, transfer.regions)) {
		return true;
	}
	const auto linear =
	    m_tiler.Detile(staging.Handle(), offset, size, transfer.LinearSize(), transfer.tiles);
	for (auto& region: transfer.regions) {
		region.bufferOffset += linear.offset;
	}
	image.Upload(transfer.regions, linear.buffer, linear.offset, linear.size);
	return true;
}

void TextureCache::RecordChunkHashes(Image& image) {
	auto&          chunks     = image.chunks;
	const uint64_t chunk_size = uint64_t {1} << chunks.shift;
	chunks.hashes.assign(chunks.count, 0);
	std::vector<uint8_t> bytes(chunk_size);
	for (uint32_t index = 0; index < chunks.count; index++) {
		const auto begin = std::max(image.live.address, chunks.base + uint64_t {index} * chunk_size);
		const auto end   = std::min(image.live.End(), chunks.base + uint64_t {index + 1} * chunk_size);
		if (begin >= end ||
		    (!LibKernel::Memory::TryReadBacking(begin, bytes.data(), end - begin) &&
		     !LibKernel::Memory::TryReadSparseBacking(begin, bytes.data(), end - begin))) {
			chunks.hashes.clear();
			return;
		}
		chunks.hashes[index] = XXH3_64bits(bytes.data(), static_cast<size_t>(end - begin));
	}
}

bool TextureCache::VerifyCleanChunks(Image& image) {
	auto&          chunks     = image.chunks;
	const uint64_t chunk_size = uint64_t {1} << chunks.shift;
	if (chunks.hashes.size() != chunks.count) {
		return false;
	}
	std::vector<uint8_t> bytes(chunk_size);
	for (uint32_t index = 0; index < chunks.count; index++) {
		if (Image::ChunkBit(chunks.dirty, index)) {
			continue;
		}
		const auto begin = std::max(image.live.address, chunks.base + uint64_t {index} * chunk_size);
		const auto end   = std::min(image.live.End(), chunks.base + uint64_t {index + 1} * chunk_size);
		if (begin >= end ||
		    (!LibKernel::Memory::TryReadBacking(begin, bytes.data(), end - begin) &&
		     !LibKernel::Memory::TryReadSparseBacking(begin, bytes.data(), end - begin))) {
			return false;
		}
		if (XXH3_64bits(bytes.data(), static_cast<size_t>(end - begin)) != chunks.hashes[index]) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialVerifyMismatches);
			if (++m_partial_verify_mismatches <= 64) {
				LOGF("Texture partial verify: clean chunk changed without a recorded write: "
				     "image=0x%016" PRIx64 " size=0x%" PRIx64 " chunk=0x%016" PRIx64 "\n",
				     image.info.data.address, image.info.data.size, begin);
			}
			return false;
		}
	}
	return true;
}

// KYTY_DCC_GPU_VIDEOOUT (default off; live): TryMaterializeGpuDccClear.
static Live::Switch g_dcc_gpu_video_out("KYTY_DCC_GPU_VIDEOOUT", Live::ParseDefaultOff);

Profiler::FrameEvent TextureCache::TryMaterializeGpuDccClear(ImageId id, const ImageDesc& desc,
                                                              uint32_t metadata_base_layer) {
	using Event = Profiler::FrameEvent;
	if (!m_dcc_clear || !m_dcc_clear->Available()) {
		return Event::DccFallbackDisabled;
	}
	// The video-out path does not consume its clear key; it stays on the CPU inspection unless
	// KYTY_DCC_GPU_VIDEOOUT=1 (default off; Ghost of Yotei preset) or KYTY_DCC_GPU_WIDE 4 (GT7
	// preset): the native inspection then checks the keys in the command stream and clears the
	// image as the CPU fallback would, but leaves the keys as the guest wrote them (no consumption,
	// no retained inspection). Ghost of Yotei's compute pass 0x1ed5a1e735be2f6a rewrites its flip
	// target's keys every frame, so the CPU inspection read them back from the GPU at every flip
	// (~10.8 ms of waiting per frame at the torii). VIDEOOUT adapted from Erlou93/KytyPS5-Yotei
	// KYTY_PERF_DCC_GPU_ALL (TextureCache::MaterializeDccClearOnGpu: video-out keys checked every
	// time, never consumed). Live.
	const bool keep_keys          = desc.type == BindingType::VideoOut;
	const bool video_out_any_kind = keep_keys && g_dcc_gpu_video_out.On();
	if (keep_keys && !video_out_any_kind && (DccGpuWide() & 4u) == 0) {
		return Event::DccFallbackBinding;
	}
	// KYTY_DCC_GPU_VIDEOOUT: the CPU fallback treats a compressed video-out description like any
	// other (its keys decide). KYTY_DCC_GPU_WIDE 8 (with 4): a DCC-compressed video-out surface
	// (GT7's flips) is inspected too. The fallback decides it from the same bytes and the same way
	// (first key a decodable clear code and every byte equal to it), whatever the compression.
	const bool compressed_flip = video_out_any_kind ||
	                             (keep_keys && (DccGpuWide() & 8u) != 0 &&
	                              desc.info.metadata.compression != VideoOutCompression::Unsupported);
	if (desc.info.metadata.compression != VideoOutCompression::Uncompressed && !compressed_flip) {
		return Event::DccFallbackShape;
	}
	// The CPU decoder's result for every clear code under this binding (ClearCodes order).
	DccClearHelper::ClearValues values {};
	uint32_t                    decodable = 0;
	for (uint32_t index = 0; index < DccClearHelper::ClearCodes.size(); ++index) {
		vk::ClearColorValue color {};
		if (DecodeDccClear(desc, DccClearHelper::ClearCodes[index], color)) {
			values[index] = color;
			decodable |= 1u << index;
		}
	}
	return TryMaterializeGpuMetadataClear(*m_dcc_clear, id, desc, desc.info.metadata.range,
	                                      metadata_base_layer, values, decodable, false, keep_keys);
}

// KYTY_DCC_GPU_ALIASED=1|2 (default off, read once): images registered over a target's metadata bytes
// no longer send the metadata inspection to the CPU fallback (a download that drains the GPU: GT7
// races, ~3.6 a frame at 1-5 ms each, metadata written by a compute dispatch of the same frame) as
// long as none of them is GPU-modified (its newest bytes would be in the image, not in the buffer
// the inspection reads) and none is the target itself. The inspection reads and rewrites the
// metadata through the canonical buffer as a GPU write (ObtainBuffer), which invalidates the images
// over those bytes as any buffer write does. Other images keep the old path.
// =2 also accepts GPU-modified images over the metadata (not the target itself), as usa-ratt's GPU
// clear check does (no aliasing test): GT7 registers an 8.4 MiB image 0x60000 past a 2048x1024 target
// whose range covers that target's DCC bytes; it is GPU-modified, so =1 keeps that inspection on the
// CPU (a drain per bind). Accuracy cost: bytes of the metadata such an image wrote and has not
// written back would be missed by the inspection (the buffer copy is read).
bool TextureCache::MetadataAliasBlocks(ImageId id, GuestRange range, bool* aliased) {
	const auto images = FindImagesInRegion(range.address, range.size, false);
	if (images.size() == 0) {
		return false;
	}
	static const int relaxed_env = [] {
		const auto* value = std::getenv("KYTY_DCC_GPU_ALIASED");
		return value == nullptr ? 0 : std::clamp(std::atoi(value), 0, 2);
	}();
	// KYTY_DCC_GPU_WIDE 1 (textureCache.h): as =2.
	const int relaxed = (DccGpuWide() & 1u) != 0 ? 2 : relaxed_env;
	if (relaxed == 0) {
		return true;
	}
	for (const auto other: images) {
		const auto* image = m_slot_images.try_get(other);
		if (other == id || image == nullptr || (relaxed == 1 && image->IsGpuModified())) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::fprintf(stderr,
				             "DCC aliased metadata 0x%016" PRIx64 "+0x%" PRIx64 " kept on the CPU path: image "
				             "data 0x%016" PRIx64 "+0x%" PRIx64 " %s\n",
				             range.address, range.size, image != nullptr ? image->info.data.address : 0,
				             image != nullptr ? image->info.data.size : 0,
				             other == id ? "is the target" : image == nullptr ? "gone" : "is GPU-modified");
			}
			return true;
		}
	}
	if (aliased != nullptr) {
		*aliased = true;
	}
	static std::atomic<uint64_t> accepted {0};
	if (const auto n = accepted.fetch_add(1, std::memory_order_relaxed) + 1;
	    relaxed_env != 0 && (n & (n - 1)) == 0) {
		const auto* image = m_slot_images.try_get(*images.begin());
		std::fprintf(stderr,
		             "DCC aliased metadata inspected on the GPU (KYTY_DCC_GPU_ALIASED): %" PRIu64
		             " times; e.g. metadata 0x%016" PRIx64 "+0x%" PRIx64 " under image data 0x%016" PRIx64
		             "+0x%" PRIx64 " %ux%u\n",
		             n, range.address, range.size, image != nullptr ? image->info.data.address : 0,
		             image != nullptr ? image->info.data.size : 0, image != nullptr ? image->info.extent.width : 0,
		             image != nullptr ? image->info.extent.height : 0);
	}
	return false;
}

Profiler::FrameEvent TextureCache::TryMaterializeGpuMetadataClear(
    DccClearHelper& helper, ImageId id, const ImageDesc& desc, GuestRange range,
    uint32_t metadata_base_layer, const DccClearHelper::ClearValues& values, uint32_t decodable,
    bool cmask, bool keep_keys) {
	using Event = Profiler::FrameEvent;
	using Support = DccClearHelper::Support;
	// KYTY_DCC_GPU_WIDE (textureCache.h): what this inspection was accepted through.
	bool wide_aliased  = false;
	bool wide_mismatch = false;
	bool watch_reused  = false; // reused only through the metadata watch (KYTY_DCC_GPU_WIDE)
	const auto& view = desc.view_info;
	if (!helper.Available()) {
		return Event::DccFallbackDisabled;
	}
	const auto layers = desc.info.TransferLayers();
	const auto first  = metadata_base_layer;
	const auto count  = view.layer_count;
	if (desc.info.IsVolume() || desc.info.resources.levels != 1 || desc.info.samples != 1 ||
	    view.base_level != 0 || view.level_count != 1 ||
	    view.aspect != vk::ImageAspectFlagBits::eColor ||
	    (view.type != vk::ImageViewType::e2D && view.type != vk::ImageViewType::e2DArray) ||
	    layers == 0 || count == 0 || first >= layers || count > layers - first) {
		return Event::DccFallbackShape;
	}
	switch (helper.SupportsFormat(view.format)) {
		case Support::Ok: break;
		case Support::Disabled: return Event::DccFallbackDisabled;
		case Support::Format: return Event::DccFallbackFormat;
		case Support::Unsupported: return Event::DccFallbackUnsupported;
	}
	const auto slice_size = range.size / layers;
	const auto slice_range = [&](uint32_t slice) {
		return GuestRange {range.address + slice_size * (first + slice), slice_size};
	};
	{
		std::scoped_lock lock {m_lock};
		if (MetadataAliasBlocks(id, range, &wide_aliased)) {
			return Event::DccFallbackMetadataAliased;
		}
		// A retained inspection of an unchanged slice leaves consumed keys (0xFF), nonuniform
		// bytes or a code its interpretation rejected. An interpretation accepting no more codes
		// than that one reads the same bytes to the same no-op, for any image or layer; only the
		// metadata processing is skipped, the caller still refreshes this image normally.
		// Kept keys (video-out) are never retained: a clear they select clears again each time.
		// KYTY_DCC_GPU_WIDE also takes a slice nothing wrote since its inspection (byte-exact
		// watch, BufferCache::MetadataUnwritten) as unchanged: the buffer's content revision moves
		// with any write anywhere in the buffer, and every binding would inspect again.
		bool       reusable = !keep_keys;
		const bool watched  = DccGpuWide() != 0;
		for (uint32_t slice = 0; reusable && slice < count; ++slice) {
			const auto metadata = slice_range(slice);
			const auto revision = m_buffer_cache.GetContentRevision(metadata.address, metadata.size);
			const auto found    = m_gpu_dcc_inspections.find(metadata.address);
			const bool same     = found != m_gpu_dcc_inspections.end() && found->second.metadata == metadata &&
			                  (decodable & ~found->second.decodable_mask) == 0;
			const bool by_revision = same && revision && found->second.revision == *revision;
			const bool by_watch =
			    same && !by_revision && watched && m_buffer_cache.MetadataUnwritten(metadata.address, metadata.size);
			watch_reused |= by_watch;
			reusable = by_revision || by_watch;
		}
		// KYTY_DCC_GPU_WIDE_VERIFY: a reuse through the watch is checked below against the bytes
		// (after a drain): a retained inspection left them deciding "no clear" (consumed keys,
		// nonuniform bytes or a rejected code); a clear there means the watch missed a write.
		if (reusable && !(watch_reused && m_dcc_gpu_wide_verify != 0)) {
			if (!cmask) {
				m_gpu_dcc_reuses += count;
				Profiler::CountFrameEvent(Event::DccGpuReuses, count);
			}
			return Event::DccGpuReuses;
		}
	}
	if (watch_reused && m_dcc_gpu_wide_verify != 0) {
		m_buffer_cache.ReadMemory(range.address, range.size, false);
		std::vector<uint8_t> bytes(slice_size);
		uint32_t             wrong = 0;
		bool                 all   = true;
		for (uint32_t slice = 0; slice < count; ++slice) {
			if (!LibKernel::Memory::TryReadBacking(slice_range(slice).address, bytes.data(), bytes.size())) {
				all = false;
				break;
			}
			uint32_t index = 0;
			while (index < DccClearHelper::ClearCodes.size() && DccClearHelper::ClearCodes[index] != bytes[0]) {
				index++;
			}
			wrong += index < DccClearHelper::ClearCodes.size() && (decodable & (1u << index)) != 0 &&
			                 std::all_of(bytes.begin(), bytes.end(), [&](uint8_t byte) { return byte == bytes[0]; })
			             ? 1u
			             : 0u;
		}
		if (all) {
			m_dcc_wide_totals.verify_checks.fetch_add(count, std::memory_order_relaxed);
			if (wrong == 0) {
				if (!cmask) {
					m_gpu_dcc_reuses += count;
					Profiler::CountFrameEvent(Event::DccGpuReuses, count);
				}
				return Event::DccGpuReuses;
			}
			m_dcc_wide_totals.verify_mismatches.fetch_add(wrong, std::memory_order_relaxed);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::fprintf(stderr,
				             "DCC wide verify: %u slice(s) of metadata 0x%016" PRIx64
				             " reused through the watch hold a clear (a write the watch missed)\n",
				             wrong, range.address);
			}
			if (m_dcc_gpu_wide_verify == 2) {
				EXIT("DCC wide verify: a watched metadata reuse missed a write\n");
			}
		}
		// Not reused: inspected below as usual.
	}
	if (ImageRangeOverlaps(range, desc.info.data)) {
		return Event::DccFallbackMetadataAliased;
	}
	// KYTY_DCC_GPU_REFRESH (textureCache.h): a DCC image that is registered, matches the view and
	// is fully resident is accepted while waiting for a refresh or not GPU-owned; it is refreshed
	// right before the inspection, as the CPU fallback's ClearImage refreshes it.
	const bool refresh_allowed = m_dcc_gpu_refresh && !cmask;
	if (keep_keys && m_dcc_gpu_refresh_verify != 0) {
		// The refresh verification expects consumed keys; the video-out path keeps them.
		return Event::DccFallbackBinding;
	}
	const auto eligible = [&]() -> std::optional<Event> {
		const auto* image = m_slot_images.try_get(id);
		const bool  exact = image != nullptr && image->info.data == desc.info.data &&
		                   image->info.extent == desc.info.extent;
		// KYTY_DCC_GPU_WIDE 2: the image found for the binding, whatever its range and extent; its
		// whole layer is inspected, as the CPU fallback's ClearImage(id) clears it.
		wide_mismatch = !exact && (DccGpuWide() & 2u) != 0;
		const bool  matches = image != nullptr && image->registered && !image->depth_id &&
		                     (exact || wide_mismatch);
		if (!matches ||
		    (!SafeToDownload(*image) && !(refresh_allowed && image->FullyResident()))) {
			// Which condition refused it (DccImageState* counters, MaterializeDccClear). The
			// pending-refresh states come before "not GPU-modified", which they usually imply.
			if (image == nullptr || !image->registered) {
				m_image_state_reason = Event::DccImageStateUnregistered;
			} else if (image->depth_id) {
				m_image_state_reason = Event::DccImageStateStencil;
			} else if (image->info.data != desc.info.data || image->info.extent != desc.info.extent) {
				m_image_state_reason = Event::DccImageStateMismatch;
			} else if (image->IsBufferModified()) {
				m_image_state_reason = Event::DccImageStateBufferModified;
			} else if (image->IsCpuDirty()) {
				m_image_state_reason = Event::DccImageStateCpuDirty;
			} else if (!image->IsGpuModified()) {
				m_image_state_reason = Event::DccImageStateNotGpuModified;
			} else if (!image->FullyResident()) {
				m_image_state_reason = Event::DccImageStatePartial;
			} else {
				m_image_state_reason = Event::DccImageStateGpuDirtyBytes;
			}
			return Event::DccFallbackImageState;
		}
		if ((!wide_mismatch && (image->backing.extent.width != desc.info.extent.width ||
		                        image->backing.extent.height != desc.info.extent.height)) ||
		    view.base_layer >= image->backing.layers ||
		    count > image->backing.layers - view.base_layer) {
			return Event::DccFallbackShape;
		}
		switch (helper.SupportsImage(*image, view.format, slice_size)) {
			case Support::Ok: break;
			case Support::Disabled: return Event::DccFallbackDisabled;
			case Support::Format: return Event::DccFallbackFormat;
			case Support::Unsupported: return Event::DccFallbackUnsupported;
		}
		if (MetadataAliasBlocks(id, range, &wide_aliased)) {
			return Event::DccFallbackMetadataAliased;
		}
		return std::nullopt;
	};
	bool refreshed = false;
	{
		std::scoped_lock lock {m_lock};
		if (const auto reason = eligible()) return *reason;
		if (refresh_allowed && !SafeToDownload(m_slot_images[id])) {
			// Before the metadata buffer is acquired: the refresh may create or join the buffer
			// under the image, which can share a caching page with the metadata.
			refreshed = true;
			InitializeImage(id);
		}
	}
	// KYTY_DCC_GPU_REFRESH_VERIFY: today's decision for an image accepted through the refresh,
	// from the metadata bytes the inspection will read (after a drain, as the CPU fallback).
	std::vector<uint8_t> verify_bytes;
	std::vector<uint8_t> verify_clears;
	if (refreshed && m_dcc_gpu_refresh_verify != 0) {
		m_buffer_cache.ReadMemory(range.address, range.size, false);
		verify_bytes.resize(slice_size * count);
		verify_clears.resize(count);
		if (!LibKernel::Memory::TryReadBacking(slice_range(0).address, verify_bytes.data(),
		                                       verify_bytes.size())) {
			EXIT("TextureCache: failed to read DCC metadata for verification\n");
		}
		for (uint32_t slice = 0; slice < count; ++slice) {
			const auto* bytes = verify_bytes.data() + slice_size * slice;
			vk::ClearColorValue color {};
			verify_clears[slice] =
			    DecodeDccClear(desc, bytes[0], color) &&
			    std::all_of(bytes, bytes + slice_size, [&](uint8_t byte) { return byte == bytes[0]; });
		}
	}
	// KYTY_DCC_GPU_WIDE_VERIFY: the CPU fallback's decision per slice, from the bytes it would read
	// (after the same drain), for an inspection accepted only through KYTY_DCC_GPU_WIDE.
	const bool           wide_used = wide_aliased || wide_mismatch || keep_keys;
	std::vector<uint8_t> wide_expected;
	if (wide_used && m_dcc_gpu_wide_verify != 0) {
		m_buffer_cache.ReadMemory(range.address, range.size, false);
		wide_expected.resize(count);
		std::vector<uint8_t> bytes(slice_size);
		for (uint32_t slice = 0; slice < count; ++slice) {
			if (!LibKernel::Memory::TryReadBacking(slice_range(slice).address, bytes.data(), bytes.size())) {
				EXIT("TextureCache: failed to read colour metadata for the wide verification\n");
			}
			uint32_t index = 0;
			while (index < DccClearHelper::ClearCodes.size() && DccClearHelper::ClearCodes[index] != bytes[0]) {
				index++;
			}
			wide_expected[slice] =
			    index < DccClearHelper::ClearCodes.size() && (decodable & (1u << index)) != 0 &&
			    std::all_of(bytes.begin(), bytes.end(), [&](uint8_t byte) { return byte == bytes[0]; });
		}
	}
	// Use the canonical buffer and its normal GPU-write tracking. No CPU shadow of the
	// clear value is trusted. Acquiring it may upload/submit, so hold no texture lock.
	// Kept keys are only read.
	auto [buffer, offset] = m_buffer_cache.ObtainBuffer(range.address, range.size, !keep_keys, false);
	if (buffer == nullptr) {
		return Event::DccFallbackUnsupported;
	}
	if (offset % sizeof(uint32_t) != 0) {
		return Event::DccFallbackAlignment;
	}
	std::scoped_lock lock {m_lock};
	if (const auto reason = eligible()) return *reason;
	auto& image = m_slot_images[id];
	if (refresh_allowed && !SafeToDownload(image)) {
		// Only when something dirtied the image again since the refresh above.
		refreshed = true;
		InitializeImage(id);
	}
	TrackImage(id);
	{
		KYTY_PROFILER_DETAIL_BLOCK("DCC::GpuMaterialize");
		uint8_t*   decisions = nullptr;
		vk::Buffer decision_buffer {};
		uint64_t   decision_offset = 0;
		if (!wide_expected.empty()) {
			auto& download                 = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
			const auto [mapped, mapped_at] = download.Map(uint64_t {16} * count, 64);
			if (mapped == nullptr) {
				EXIT("TextureCache: DCC wide verification exceeds the download ring\n");
			}
			download.Commit();
			decisions       = mapped;
			decision_buffer = download.Handle();
			decision_offset = mapped_at;
		}
		for (uint32_t slice = 0; slice < count; ++slice) {
			helper.RecordSlice(image, view.format, view.base_layer + slice, buffer->Handle(),
			                   offset + slice_size * (first + slice), slice_size, values, keep_keys,
			                   decision_buffer, decision_offset + uint64_t {16} * slice);
		}
		if (decisions != nullptr) {
			RecordDccWideVerify(decisions, decision_offset, std::move(wide_expected), range,
			                    static_cast<uint32_t>(desc.type));
		}
	}
	CommitGpuWrite(image);
	if (wide_used) {
		(keep_keys       ? m_dcc_wide_totals.video_out
		 : wide_mismatch ? m_dcc_wide_totals.mismatch
		                 : m_dcc_wide_totals.aliased)
		    .fetch_add(count, std::memory_order_relaxed);
		ReportDccWide();
	}
	if (refreshed) {
		m_dcc_refresh_totals.refreshes += count;
		Profiler::CountFrameEvent(Event::DccGpuRefreshes, count);
		if (!verify_clears.empty()) {
			RecordDccRefreshVerify(*buffer, offset + slice_size * first, slice_size,
			                       std::move(verify_bytes), std::move(verify_clears), range);
		}
	}
	if (!cmask) {
		m_gpu_dcc_records += count;
		Profiler::CountFrameEvent(Event::DccGpuRecords, count);
	}
	for (uint32_t slice = 0; !keep_keys && slice < count; ++slice) {
		const auto metadata = slice_range(slice);
		if (const auto revision =
		        m_buffer_cache.GetContentRevision(metadata.address, metadata.size)) {
			// Retained metadata is independent of image lifetime; cap address reuse history.
			if (m_gpu_dcc_inspections.size() >= 256 &&
			    !m_gpu_dcc_inspections.contains(metadata.address)) {
				m_gpu_dcc_inspections.clear();
			}
			m_gpu_dcc_inspections.insert_or_assign(
			    metadata.address, GpuDccInspection {metadata, *revision, decodable});
			if (DccGpuWide() != 0) {
				m_buffer_cache.WatchMetadata(metadata.address, metadata.size);
			}
		} else {
			m_gpu_dcc_inspections.erase(metadata.address);
		}
	}
	if (!cmask && m_gpu_dcc_records <= 8) {
		LOGF("GPU DCC inspection: metadata=0x%016" PRIx64 " bytes=%" PRIu64
		     " image=0x%016" PRIx64 " format=%u type=%u slices=%u codes=0x%02x\n",
		     range.address, range.size, desc.info.data.address,
		     static_cast<uint32_t>(view.format), static_cast<uint32_t>(desc.type), count,
		     decodable);
	}
	TraceDccDiagnostic("DCC_GPU_RECORD meta=0x%" PRIx64 " bytes=%" PRIu64
	                   " data=0x%" PRIx64 " format=%u type=%u first=%u count=%u tick=%" PRIu64,
	                   range.address, range.size, desc.info.data.address,
	                   static_cast<uint32_t>(view.format), static_cast<uint32_t>(desc.type), first,
	                   count, m_scheduler.CurrentTick());
	return Event::DccGpuRecords;
}

// KYTY_DCC_GPU_WIDE (textureCache.h) is also a live switch (common/liveSwitch.h): the mask used is the
// startup value (read when the cache is created) or'ed with the live value, so live control (A/B) turns
// bits on and off when the run starts with 0. Every inspection decides with the current mask, and both
// paths leave the metadata in a state the other reads the same way, so a change between two
// inspections is safe.
int64_t TextureCache::ParseDccGpuWide(const char* value) {
	return value != nullptr ? static_cast<int64_t>(std::strtoul(value, nullptr, 0) & 15u) : 0;
}
static Live::Switch g_dcc_gpu_wide("KYTY_DCC_GPU_WIDE", &TextureCache::ParseDccGpuWide);

uint32_t TextureCache::DccGpuWide() const {
	return m_dcc_gpu_wide | static_cast<uint32_t>(g_dcc_gpu_wide.Get());
}

void TextureCache::ReportDccWide() {
	const auto aliased  = m_dcc_wide_totals.aliased.load(std::memory_order_relaxed);
	const auto mismatch = m_dcc_wide_totals.mismatch.load(std::memory_order_relaxed);
	const auto flips    = m_dcc_wide_totals.video_out.load(std::memory_order_relaxed);
	const auto n        = aliased + mismatch + flips;
	if (n != 0 && (n & (n - 1)) == 0) {
		std::fprintf(stderr,
		             "DCC wide (KYTY_DCC_GPU_WIDE=%u): %" PRIu64 " slices inspected on the GPU instead of a "
		             "drain: %" PRIu64 " under aliasing images, %" PRIu64 " of a differing image, %" PRIu64
		             " video-out; verify %" PRIu64 " checks, %" PRIu64 " mismatches\n",
		             DccGpuWide(), n, aliased, mismatch, flips,
		             m_dcc_wide_totals.verify_checks.load(std::memory_order_relaxed),
		             m_dcc_wide_totals.verify_mismatches.load(std::memory_order_relaxed));
	}
}

void TextureCache::RecordDccWideVerify(const uint8_t* decisions, uint64_t download_offset,
                                       std::vector<uint8_t> expected, GuestRange range, uint32_t type) {
	auto& download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	m_scheduler.DeferOperation([this, &download, decisions, download_offset,
	                            expected = std::move(expected), range, type] {
		download.Invalidate(download_offset, uint64_t {16} * expected.size());
		uint32_t wrong = 0;
		for (size_t slice = 0; slice < expected.size(); ++slice) {
			uint32_t groups = 0;
			std::memcpy(&groups, decisions + 16 * slice, sizeof(groups));
			wrong += (groups != 0) != (expected[slice] != 0) ? 1u : 0u;
		}
		m_dcc_wide_totals.verify_checks.fetch_add(expected.size(), std::memory_order_relaxed);
		if (wrong == 0) {
			return;
		}
		m_dcc_wide_totals.verify_mismatches.fetch_add(wrong, std::memory_order_relaxed);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::fprintf(stderr,
			             "DCC wide verify: %u of %zu slice(s) of metadata 0x%016" PRIx64 " size=0x%" PRIx64
			             " (binding type %u) decided differently from the CPU fallback\n",
			             wrong, expected.size(), range.address, range.size, type);
		}
		if (m_dcc_gpu_wide_verify == 2) {
			EXIT("DCC wide verify: an inspection differs from the CPU fallback\n");
		}
	});
}

void TextureCache::RecordDccRefreshVerify(Buffer& metadata, uint64_t offset, uint64_t slice_size,
                                          std::vector<uint8_t> before, std::vector<uint8_t> clears,
                                          GuestRange range) {
	// Caller holds m_lock, right after the inspection was recorded: copy the inspected slices
	// out behind it and, once the recording completed, check what the helper did to them against
	// the CPU fallback's decision. A cleared slice has its key consumed (every byte 0xFF); any
	// other slice is untouched.
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	const auto bytes    = static_cast<uint64_t>(before.size());
	const auto [mapped, download_offset] = download.Map(bytes, 64);
	if (mapped == nullptr) {
		EXIT("TextureCache: DCC refresh verification exceeds the download ring\n");
	}
	download.Commit();
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask       = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = metadata.Handle();
	barrier.offset              = offset;
	barrier.size                = bytes;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &barrier, 0,
	                       nullptr);
	const vk::BufferCopy copy {offset, download_offset, bytes};
	native.copyBuffer(metadata.Handle(), download.Handle(), 1, &copy);
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.buffer        = download.Handle();
	barrier.offset        = download_offset;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &barrier, 0, nullptr);
	m_scheduler.DeferOperation([this, &download, mapped = mapped, download_offset = download_offset,
	                            bytes, slice_size, before = std::move(before),
	                            clears = std::move(clears), range] {
		download.Invalidate(download_offset, bytes);
		const auto* after = mapped;
		uint32_t    wrong = 0;
		for (size_t slice = 0; slice < clears.size(); ++slice) {
			const auto* now  = after + slice_size * slice;
			const auto* then = before.data() + slice_size * slice;
			const bool  ok   = clears[slice] != 0
			                       ? std::all_of(now, now + slice_size,
			                                     [](uint8_t byte) { return byte == 0xffu; })
			                       : std::memcmp(now, then, slice_size) == 0;
			wrong += ok ? 0u : 1u;
		}
		m_dcc_refresh_totals.verify_checks++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DccGpuRefreshVerifyChecks);
		if (wrong == 0) {
			return;
		}
		m_dcc_refresh_totals.verify_mismatches += wrong;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DccGpuRefreshVerifyMismatches, wrong);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::fprintf(stderr,
			             "DccGpuRefreshVerify: %u slice(s) of metadata 0x%016" PRIx64
			             " size=0x%" PRIx64 " differ from the CPU fallback's decision\n",
			             wrong, range.address, range.size);
		}
		if (m_dcc_gpu_refresh_verify == 2) {
			EXIT("DccGpuRefreshVerify: a refreshed image's inspection differs from the CPU "
			     "fallback\n");
		}
	});
}

void TextureCache::MaterializeDccClear(ImageId id, const ImageDesc& desc,
                                       uint32_t metadata_base_layer, MetadataNoop* noop) {
	ImageWatch("image DCC clear materialize", m_slot_images[id]);
	if (noop != nullptr) {
		*noop = {};
	}
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc) {
		if (noop != nullptr) {
			noop->provable = true;
		}
		return;
	}
	KYTY_PROFILER_DETAIL_FUNCTION();
	const auto range = desc.info.metadata.range;
	// KYTY_WATCH_RANGE (diagnostic): every 32nd DCC decision of an image over the watched range
	// reads the metadata (waits for the GPU) and logs how many key bytes hold each value.
	if (WatchRange::Enabled() && range.Valid() && range.size != 0 && range.size <= (16u << 20u)) {
		static std::atomic<uint32_t> calls {0};
		if ((calls.fetch_add(1, std::memory_order_relaxed) & 31u) == 0) {
			m_buffer_cache.ReadMemory(range.address, range.size, false);
			std::vector<uint8_t> keys(range.size);
			if (LibKernel::Memory::TryReadBacking(range.address, keys.data(), range.size)) {
				std::array<uint32_t, 256> histogram {};
				for (const auto key: keys) {
					histogram[key]++;
				}
				std::string text;
				for (uint32_t value = 0; value < 256; value++) {
					if (histogram[value] != 0) {
						text += fmt::format(" {:02x}:{}", value, histogram[value]);
					}
				}
				WatchRange::Note(fmt::format("DCC keys meta 0x{:x}+0x{:x} {}x{} type {}:{}", range.address,
				                             range.size, desc.info.extent.width, desc.info.extent.height,
				                             static_cast<uint32_t>(desc.type), text)
				                     .c_str(),
				                 desc.info.data.address, desc.info.data.size);
			}
		}
	}
	ClearTrace("DCC", "enter", range.address, range.size, desc.info.data.address, desc.info.data.size,
	           desc.info.extent.width, desc.info.extent.height,
	           desc.info.resources.levels | (static_cast<uint32_t>(desc.type) << 8u) |
	               (static_cast<uint32_t>(desc.info.metadata.dcc_clear_word) << 16u));
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		if (ColourMetadataTargetsDepth("DCC", id, image)) {
			return; // not provable: every lookup decides again
		}
		image.info.metadata = desc.info.metadata;
		// A native DCC allocation must not retain a reused HTile/CMask/FMask interpretation.
		EraseSurfaceMeta(range.address);
		// A single-mip target reused from a cached mip chain clears its view's mip (upstream
		// 0ec3655f1); the image's own level count does not matter.
		if (range.size == 0 || desc.info.resources.levels != 1) {
			// Decided by the description alone.
			if (noop != nullptr) {
				noop->provable = true;
			}
			return;
		}
	}
	const auto layers = desc.info.TransferLayers();
	// These one-mip surfaces use complete 4 KiB DCC metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: DCC slices must contain aligned 4 KiB blocks\n");
	}
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: DCC view exceeds its native metadata slices\n");
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	uint64_t diagnostic_readback = 0;
	// Why this decision is not a provable no-op (instrumentation; the paths below refine it).
	m_dcc_noop_refusal = Profiler::FrameEvent::TargetRecordDccGuest;
	const bool gpu_written = m_buffer_cache.IsRegionGpuModified(range.address, range.size);
	if (gpu_written) {
		// The guest's DCC fast clear is a uniform fill of the metadata. When the whole range still
		// holds a recorded fill (nothing wrote it since), every slice's code is that byte: no
		// GPU readback is needed to decide the clear.
		uint64_t   fill_generation = 0;
		const auto known = m_buffer_cache.KnownFill(range.address, range.size, fill_generation);
		bool       unaliased = false;
		bool       pages     = false;
		MetadataNoop decided;
		if (known) {
			std::scoped_lock lock {m_lock};
			unaliased = FindImagesInRegion(range.address, range.size, false).empty();
			pages     = noop != nullptr && CaptureMetadataPages(range, decided);
		}
		const uint32_t known_byte = known ? (*known & 0xffu) : 0u;
		if (known && !(unaliased && *known == known_byte * 0x01010101u)) {
			ClearTrace("DCC", unaliased ? "known-nonuniform" : "known-aliased", range.address, range.size,
			           desc.info.data.address, desc.info.data.size, desc.info.extent.width,
			           desc.info.extent.height, *known);
		} else if (!known) {
			ClearTrace("DCC", "gpu-written-no-known-fill", range.address, range.size, desc.info.data.address,
			           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, 0);
		}
		if (known && unaliased && *known == known_byte * 0x01010101u) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DccKnownFillClears);
			vk::ClearValue clear {};
			const bool known_decoded = DecodeDccClear(desc, static_cast<uint8_t>(known_byte), clear.color);
			ClearTrace("DCC", known_decoded ? "known-clear" : "known-not-clear-code", range.address, range.size,
			           desc.info.data.address, desc.info.data.size, desc.info.extent.width,
			           desc.info.extent.height, known_byte);
			if (!known_decoded) {
				// Decided by the GPU-dirty state, the recorded fill and the images over the bytes.
				if (pages) {
					decided.fill_generation = fill_generation;
					decided.gpu_ranges[0]   = range;
					decided.gpu_range_count = 1;
					decided.provable        = true;
					*noop                   = decided;
				} else {
					m_dcc_noop_refusal = Profiler::FrameEvent::TargetRecordDccPages;
				}
				return; // e.g. 0xFF: not a clear code, nothing to materialize
			}
			m_dcc_noop_refusal    = Profiler::FrameEvent::TargetRecordDccClear;
			const auto slice_size = range.size / layers;
			for (uint32_t slice = 0; slice < count; slice++) {
				++m_dcc_decision_effects;
				bool cleared = false;
				{
					std::scoped_lock lock {m_lock};
					cleared = ClearImage(m_scheduler.Current(), id, view.format,
					                     {vk::ImageAspectFlagBits::eColor, view.base_level,
					                      view.level_count, image_first + slice, 1},
					                     clear, "dcc-known-fill");
				}
				if (!cleared) {
					// Rejected (reported): leave the metadata as the guest wrote it.
					m_dcc_noop_refusal = Profiler::FrameEvent::TargetRecordDccClear;
					return;
				}
				if (desc.type != BindingType::VideoOut) {
					m_buffer_cache.FillBuffer(range.address + slice_size * (first + slice),
					                          slice_size, UINT32_MAX, false);
				}
			}
			return;
		}
		{
			++m_gpu_dcc_attempts;
			++m_dcc_decision_effects;
			const auto outcome = TryMaterializeGpuDccClear(id, desc, metadata_base_layer);
			ClearTrace("DCC", "gpu-outcome", range.address, range.size, desc.info.data.address,
			           desc.info.data.size, desc.info.extent.width, desc.info.extent.height,
			           static_cast<uint32_t>(outcome));
			const bool native  = outcome == Profiler::FrameEvent::DccGpuRecords ||
			                    outcome == Profiler::FrameEvent::DccGpuReuses;
			m_dcc_noop_refusal = native ? Profiler::FrameEvent::TargetRecordDccNative
			                            : Profiler::FrameEvent::TargetRecordDccFallback;
			if (!native) {
				++m_gpu_dcc_fallbacks;
				Profiler::CountFrameEvent(Profiler::FrameEvent::DccCpuFallbacks);
				Profiler::CountFrameEvent(outcome);
				if (outcome == Profiler::FrameEvent::DccFallbackImageState) {
					Profiler::CountFrameEvent(m_image_state_reason);
					// The first few of every cause, also in release builds (LOGF may be off).
					static std::array<std::atomic<uint32_t>,
					                  static_cast<size_t>(Profiler::FrameEvent::Count)>
					    cause_logged {};
					if (cause_logged[static_cast<size_t>(m_image_state_reason)].fetch_add(
					        1, std::memory_order_relaxed) < 4) {
						std::fprintf(stderr,
						             "DCC CPU fallback, image state cause %s: metadata=0x%016" PRIx64
						             " bytes=%" PRIu64 " data=0x%016" PRIx64 " size=0x%" PRIx64
						             " extent=%ux%u type=%u\n",
						             DccImageStateName(m_image_state_reason), range.address,
						             range.size, desc.info.data.address, desc.info.data.size,
						             desc.info.extent.width, desc.info.extent.height,
						             static_cast<uint32_t>(desc.type));
					}
				}
				// The first few fallbacks of every reason, with what decided it.
				static std::array<std::atomic<uint32_t>, static_cast<size_t>(
				                                             Profiler::FrameEvent::Count)>
				    logged {};
				if (logged[static_cast<size_t>(outcome)].fetch_add(1, std::memory_order_relaxed) <
				    4) {
					uint32_t state = 0;
					uint32_t native_format = 0, native_layers = 0, native_usage = 0,
					         native_flags = 0;
					{
						std::scoped_lock lock {m_lock};
						if (const auto* image = m_slot_images.try_get(id)) {
							state = (image->IsCpuDirty() ? 1u : 0u) |
							        (image->IsBufferModified() ? 2u : 0u) |
							        (image->IsGpuModified() ? 4u : 0u) |
							        (image->registered ? 8u : 0u) |
							        (image->depth_id ? 16u : 0u) |
							        (m_buffer_cache.HasGpuDirtyBytes(image->info.data.address,
							                                          image->info.data.size)
							             ? 32u
							             : 0u) |
							        (!FindImagesInRegion(range.address, range.size, false).empty()
							             ? 64u
							             : 0u);
							native_format = static_cast<uint32_t>(image->backing.format);
							native_layers = image->backing.layers;
							native_usage =
							    static_cast<uint32_t>(static_cast<VkImageUsageFlags>(image->backing.usage));
							native_flags = static_cast<uint32_t>(
							    static_cast<VkImageCreateFlags>(image->backing.flags));
						}
					}
					LOGF("DCC CPU fallback (reason event %u): metadata=0x%016" PRIx64
					     " bytes=%" PRIu64 " data=0x%016" PRIx64 " type=%u view_format=%u"
					     " view_type=%u base_layer=%u layers=%u meta_first=%u meta_layers=%u"
					     " extent=%ux%ux%u native_format=%u native_layers=%u usage=0x%x"
					     " flags=0x%x state=0x%x (1 cpu-dirty 2 buffer-modified 4 gpu-modified"
					     " 8 registered 16 stencil 32 gpu-dirty-bytes 64 metadata-image)\n",
					     static_cast<uint32_t>(outcome), range.address, range.size,
					     desc.info.data.address, static_cast<uint32_t>(desc.type),
					     static_cast<uint32_t>(view.format), static_cast<uint32_t>(view.type),
					     view.base_layer, view.layer_count, first, layers, desc.info.extent.width,
					     desc.info.extent.height, desc.info.extent.depth, native_format,
					     native_layers, native_usage, native_flags, state);
				}
			}
			if ((m_gpu_dcc_attempts & 255u) == 0) {
				if (tracy::ProfilerAvailable()) {
					TracyPlot("DCC.NativeInspections", static_cast<double>(m_gpu_dcc_records));
					TracyPlot("DCC.ReusedInspections", static_cast<double>(m_gpu_dcc_reuses));
					TracyPlot("DCC.CpuFallbacks", static_cast<double>(m_gpu_dcc_fallbacks));
				}
				LOGF("GPU DCC totals: requests=%" PRIu64 " native=%" PRIu64
				     " reused=%" PRIu64 " CPU_fallback=%" PRIu64 "\n",
				     m_gpu_dcc_attempts, m_gpu_dcc_records, m_gpu_dcc_reuses, m_gpu_dcc_fallbacks);
			}
			if (native) return;
		}
		KYTY_PROFILER_DETAIL_BLOCK("DCC::Readback");
		if (Profiler::DetailedEnabled() && tracy::ProfilerAvailable() && TracyIsConnected) {
			static std::atomic<uint64_t> readback_count {0};
			diagnostic_readback = readback_count.fetch_add(1, std::memory_order_relaxed) + 1u;
			TraceDccDiagnostic(
			    "DCC_READBACK_BEGIN id=%" PRIu64 " meta=0x%" PRIx64 " bytes=%" PRIu64
			    " data=0x%" PRIx64 " data_bytes=%" PRIu64 " format=%u type=%u first=%u count=%u"
			    " layers=%u mip=%u tick=%" PRIu64,
			    diagnostic_readback, range.address, range.size, desc.info.data.address,
			    desc.info.data.size, static_cast<uint32_t>(view.format),
			    static_cast<uint32_t>(desc.type), first, count, layers, view.base_level,
			    m_scheduler.CurrentTick());
		}
		// An odd period avoids repeatedly sampling the same member of an alternating pair.
		const bool sample_fallback = m_dcc_clear && tracy::ProfilerAvailable() && TracyIsConnected &&
		                             (m_gpu_dcc_fallbacks % 257u) == 0;
		uint32_t ownership = 0;
		if (sample_fallback) {
			std::scoped_lock lock {m_lock};
			const auto& image = m_slot_images[id];
			ownership = (image.IsCpuDirty() ? 1u : 0u) |
			            (image.IsBufferModified() ? 2u : 0u) |
			            (image.IsGpuModified() ? 4u : 0u) |
			            (m_buffer_cache.HasGpuDirtyBytes(image.info.data.address,
			                                              image.info.data.size) ? 8u : 0u);
		}
		const auto readback_start = sample_fallback ? std::chrono::steady_clock::now()
		                                            : std::chrono::steady_clock::time_point {};
		{
			Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::DccFallback);
			m_buffer_cache.ReadMemory(range.address, range.size, false);
		}
		if (sample_fallback) {
			const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
			    std::chrono::steady_clock::now() - readback_start).count();
			char message[400];
			const auto length = std::snprintf(message, sizeof(message),
			    "DCC_CPU_FALLBACK meta=0x%" PRIx64 " bytes=%" PRIu64 " data=0x%" PRIx64
			    " format=%u type=%u ownership=%u layers=%u first=%u count=%u ns=%" PRIu64,
			    range.address, range.size, desc.info.data.address,
			    static_cast<uint32_t>(view.format), static_cast<uint32_t>(desc.type),
			    ownership, layers, first, count,
			    static_cast<uint64_t>(elapsed));
			if (length > 0 && static_cast<size_t>(length) < sizeof(message)) {
				TracyMessageS(message, static_cast<size_t>(length), 0);
			}
		}
		if (diagnostic_readback != 0) {
			TraceDccDiagnostic("DCC_READBACK_END id=%" PRIu64 " tick=%" PRIu64,
			                   diagnostic_readback, m_scheduler.CurrentTick());
		}
	}
	const auto slice_size = range.size / layers;
	// KYTY_CP_COMMIT=dccguest: metadata the GPU has not written, decided by guest key bytes alone
	// while no slice's first key is a clear code (MetadataNoop::guest_key_count).
	bool guest_keys = noop != nullptr && !gpu_written && count != 0 &&
	                  count <= MetadataNoop::MaxGuestKeys && CpCommit::Enabled(CpCommit::Part::DccGuest);
	for (uint32_t slice = 0; slice < count; slice++) {
		KYTY_PROFILER_DETAIL_BLOCK("DCC::InspectSlice");
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read DCC metadata backing\n");
		}
		vk::ClearValue clear {};
		const bool     decoded = DecodeDccClear(desc, code, clear.color);
		if (guest_keys) {
			if (decoded) {
				guest_keys = false; // the rest of the slice decides
			} else {
				noop->guest_key_addresses[slice] = address;
				noop->guest_key_codes[slice]     = code;
			}
		}
		if (!decoded) {
			ClearTrace("DCC", "cpu-not-clear-code", address, slice_size, desc.info.data.address,
			           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, code);
			if (diagnostic_readback != 0) {
				TraceDccDiagnostic(
				    "DCC_SLICE id=%" PRIu64 " slice=%u address=0x%" PRIx64 " bytes=%" PRIu64
				    " code=0x%02x decoded=0 uniform=-1 consumed=0",
				    diagnostic_readback, first + slice, address, slice_size, unsigned(code));
			}
			continue;
		}
		std::vector<uint8_t> bytes(slice_size);
		if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
			EXIT("TextureCache: failed to read DCC metadata slice\n");
		}
		if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
			{
				const auto diff = std::find_if(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte != code; });
				const auto same = static_cast<uint64_t>(std::count(bytes.begin(), bytes.end(), code));
				ClearTrace("DCC", "cpu-nonuniform", address, slice_size, desc.info.data.address,
				           desc.info.data.size, desc.info.extent.width, desc.info.extent.height,
				           static_cast<uint32_t>(diff - bytes.begin()) |
				               (static_cast<uint32_t>(same * 255u / bytes.size()) << 24u));
			}
			if (diagnostic_readback != 0) {
				TraceDccDiagnostic(
				    "DCC_SLICE id=%" PRIu64 " slice=%u address=0x%" PRIx64 " bytes=%" PRIu64
				    " code=0x%02x decoded=1 uniform=0 consumed=0",
				    diagnostic_readback, first + slice, address, slice_size, unsigned(code));
			}
			continue;
		}
		bool cleared = false;
		{
			KYTY_PROFILER_DETAIL_BLOCK("DCC::ClearImage");
			++m_dcc_decision_effects;
			std::scoped_lock lock {m_lock};
			cleared = ClearImage(m_scheduler.Current(), id, view.format,
			                     {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			                      image_first + slice, 1},
			                     clear, "dcc-cpu-fallback");
		}
		ClearTrace("DCC", cleared ? "cpu-cleared" : "cpu-rejected", address, slice_size,
		           desc.info.data.address, desc.info.data.size, desc.info.extent.width,
		           desc.info.extent.height, code);
		if (!cleared) {
			// Rejected (reported): the guest's clear key stays; nothing is published as cleared.
			continue;
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			KYTY_PROFILER_DETAIL_BLOCK("DCC::ConsumeClearKey");
			// Not a guest write (no write fault): images over the metadata bytes keep their
			// contents (upstream 190608ae2). Like FillBuffer, retire a metadata clear recorded here.
			(void)ClearMeta(address);
			std::fill(bytes.begin(), bytes.end(), uint8_t {0xff});
			m_buffer_cache.InvalidateMemory(address, slice_size);
			LibKernel::Memory::WriteBacking(address, bytes.data(), bytes.size());
		}
		if (diagnostic_readback != 0) {
			TraceDccDiagnostic(
			    "DCC_SLICE id=%" PRIu64 " slice=%u address=0x%" PRIx64 " bytes=%" PRIu64
			    " code=0x%02x decoded=1 uniform=1 consumed=%u",
			    diagnostic_readback, first + slice, address, slice_size, unsigned(code),
			    desc.type != BindingType::VideoOut ? 1u : 0u);
		}
	}
	if (guest_keys) {
		// Every slice's first key decoded to no clear, and nothing else was read or done: the same
		// keys on still CPU-owned metadata give the same decision (TryRepeatLookup re-reads them).
		noop->guest_range     = range;
		noop->guest_key_count = count;
		noop->provable        = true;
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpCommitDccGuestRecords);
	}
}

void TextureCache::MaterializeCmaskClear(ImageId id, const ImageDesc& desc,
                                         uint32_t metadata_base_layer, MetadataNoop* noop) {
	ImageWatch("image CMASK clear materialize", m_slot_images[id]);
	using Event = Profiler::FrameEvent;
	if (noop != nullptr) {
		*noop = {};
	}
	// Decided by the description alone.
	const auto description_noop = [noop] {
		if (noop != nullptr) {
			noop->provable = true;
		}
	};
	if (!desc.cmask.valid || desc.type != BindingType::RenderTarget || !id) {
		description_noop();
		return;
	}
	KYTY_PROFILER_DETAIL_FUNCTION();
	{
		std::scoped_lock lock {m_lock};
		if (ColourMetadataTargetsDepth("CMASK", id, m_slot_images[id])) {
			return;
		}
		// KYTY_CMASK_MSAA (usa-ratt/KytyPS5-GT7 8a0b6058f): GT7 reuses memory within a frame, and
		// a depth target's HTile can start where a (multisampled) colour target keeps its CMASK
		// (the in-car map). A compute fill there clears the depth target's HTile and must still
		// run, so this target reads the filled CMASK too (IsColorMeta, UpdateComputeMetaClears).
		if (CmaskMsaaEnabled() && desc.cmask.range.Valid()) {
			if (m_color_metas.size() >= 4096) {
				m_color_metas.clear();
			}
			m_color_metas.insert(desc.cmask.range.address);
		}
	}
	const auto  range  = desc.cmask.range;
	const auto& view   = desc.view_info;
	const auto  layers = desc.info.TransferLayers();
	if (!range.Valid() || layers == 0 || range.size % layers != 0 || (range.size / layers) % 4 != 0 ||
	    desc.info.resources.levels != 1 || desc.info.IsVolume() ||
	    (desc.info.samples != 1 && !CmaskMsaaEnabled()) ||
	    view.layer_count == 0 || metadata_base_layer >= layers ||
	    view.layer_count > layers - metadata_base_layer) {
		Profiler::CountFrameEvent(Event::CmaskFastClearShape);
		ClearTrace("CMASK", "shape", range.address, range.size, desc.info.data.address, desc.info.data.size,
		           desc.info.extent.width, desc.info.extent.height, layers);
		description_noop();
		return;
	}
	vk::ClearValue clear {};
	if (!DecodePackedColorClear64(view.format,
	                              GuestColorWordToHost(desc.cmask.clear_word0,
	                                                   desc.info.GetColorTransform()),
	                              desc.cmask.clear_word1, clear.color)) {
		Profiler::CountFrameEvent(Event::CmaskFastClearFormat);
		description_noop();
		return;
	}
	const auto slice_size = range.size / layers;
	// MetadataNoop: the images over the bound slices' bytes are captured before any slice is
	// decided (a change in between only makes the record older than what the decisions read, so a
	// repeat is refused). Every slice must be decided by aliasing or by a recorded fill of
	// GPU-owned bytes that is not a clear; reading guest bytes, an inspection or a clear refuse.
	MetadataNoop decided;
	bool         provable = false;
	if (noop != nullptr) {
		std::scoped_lock lock {m_lock};
		provable = CaptureMetadataPages(
		    {range.address + slice_size * metadata_base_layer, slice_size * view.layer_count},
		    decided);
	}
	// Slices whose GPU-owned bytes only a native inspection or a readback can decide.
	std::vector<uint8_t> native(view.layer_count, 0);
	for (uint32_t slice = 0; slice < view.layer_count; slice++) {
		const GuestRange metadata {range.address + slice_size * (metadata_base_layer + slice),
		                           slice_size};
		{
			std::scoped_lock lock {m_lock};
			// Bytes some image also covers are not proven to be this surface's CMASK.
			const auto aliases = FindImagesInRegion(metadata.address, metadata.size, false);
			bool       superseded_only = false;
			// KYTY_CMASK_ALIAS_SUPERSEDED=1 (default off; on in the GT7 preset): an image that covers
			// this target's own data as well as its CMASK is an older surface of the same reused
			// memory (GT7's course-map preview over a stale 1920x1080 target), not an owner of the
			// CMASK bytes; it no longer refuses the clear (the trails of the course-map markers).
			if (CmaskAliasSupersededEnabled()) {
				const auto data_begin = desc.info.data.address;
				const auto data_end   = data_begin + desc.info.data.size;
				superseded_only = true;
				for (const auto alias: aliases) {
					const auto& other = m_slot_images[alias].info.data;
					superseded_only &= other.address < data_end && data_begin < other.address + other.size;
				}
			}
			{
				// KYTY_CLEAR_TRACE: the images covering the CMASK bytes with their LRU ticks against the
				// target's and the GC tick (first 60 decisions), to tell stale aliases from live ones.
				static const bool trace = [] {
					const auto* value = std::getenv("KYTY_CLEAR_TRACE");
					return value != nullptr && value[0] != '\0' && value[0] != '0';
				}();
				static uint32_t traced = 0;
				if (trace && !aliases.empty() && traced < 60) {
					traced++;
					std::fprintf(stderr, "ClearTrace: CMASK alias meta=0x%llx target tick %llu gc tick %llu superseded_only %d\n",
					             static_cast<unsigned long long>(metadata.address),
					             static_cast<unsigned long long>(m_slot_images[id].lru_tick),
					             static_cast<unsigned long long>(m_gc_tick), superseded_only ? 1 : 0);
					for (const auto alias: aliases) {
						const auto& image = m_slot_images[alias];
						const auto& other = image.info;
						std::fprintf(stderr,
						             "ClearTrace:   covered by image 0x%llx+0x%llx %ux%u samples %u tick %llu\n",
						             static_cast<unsigned long long>(other.data.address),
						             static_cast<unsigned long long>(other.data.size), other.extent.width,
						             other.extent.height, static_cast<unsigned>(other.samples),
						             static_cast<unsigned long long>(image.lru_tick));
					}
				}
			}
			if (!aliases.empty() && superseded_only && CmaskAliasSupersededMode() == 2) {
				static std::unordered_map<uint64_t, uint64_t> cleared_at;
				auto& tick = cleared_at[metadata.address];
				if (tick != m_gc_tick + 1u) {
					tick = m_gc_tick + 1u;
					ClearImage(m_scheduler.Current(), id, view.format,
					           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
					            view.base_layer + slice, 1},
					           clear, "cmask-alias-frame-clear");
					ClearTrace("CMASK", "alias-frame-clear", metadata.address, metadata.size, desc.info.data.address,
					           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, slice);
				}
				continue;
			}
			if (!aliases.empty() && !superseded_only) {
				Profiler::CountFrameEvent(Event::CmaskFastClearAliased);
				ClearTrace("CMASK", "aliased", metadata.address, metadata.size, desc.info.data.address,
				           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, slice);
				continue;
			}
		}
		// The proof that every tile of the slice is fast-cleared: all CMASK bytes are 0. Guest
		// memory holds bytes the GPU does not own. GPU-owned bytes are decided here only while a
		// recorded uniform fill still covers them (nothing wrote them since), otherwise by the
		// native inspection below, which reads them on the GPU.
		std::optional<uint32_t> value;
		if (m_buffer_cache.IsRegionGpuModified(metadata.address, metadata.size)) {
			uint64_t fill_generation = 0;
			value = m_buffer_cache.KnownFill(metadata.address, metadata.size, fill_generation);
			if (!value) {
				native[slice] = 1;
				provable      = false;
				continue;
			}
			if (provable) {
				if ((decided.fill_generation != 0 && decided.fill_generation != fill_generation) ||
				    decided.gpu_range_count == MetadataNoop::MaxGpuRanges) {
					provable = false;
				} else {
					decided.fill_generation                          = fill_generation;
					decided.gpu_ranges[decided.gpu_range_count++] = metadata;
				}
			}
		} else {
			provable = false;
			std::vector<uint32_t> words(metadata.size / sizeof(uint32_t));
			if (LibKernel::Memory::TryReadBacking(metadata.address, words.data(), metadata.size) &&
			    std::all_of(words.begin(), words.end(),
			                [first = words.front()](uint32_t word) { return word == first; })) {
				value = words.front();
			}
		}
		if (!value || *value != 0) {
			ClearTrace("CMASK", !value ? "nonuniform" : (*value == UINT32_MAX ? "expanded" : "uniform-nonzero"),
			           metadata.address, metadata.size, desc.info.data.address, desc.info.data.size,
			           desc.info.extent.width, desc.info.extent.height, value ? *value : 0u);
			// 0xFFFFFFFF: every tile expanded (no pending clear). Anything else: some tiles are not
			// fast-cleared (or the bytes encode nothing a single-sample surface uses).
			Profiler::CountFrameEvent(value && *value == UINT32_MAX ? Event::CmaskFastClearExpanded
			                                                        : Event::CmaskFastClearUnproven);
			continue;
		}
		provable = false;
		bool cleared = false;
		{
			std::scoped_lock lock {m_lock};
			cleared = ClearImage(m_scheduler.Current(), id, view.format,
			                     {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			                      view.base_layer + slice, 1},
			                     clear, "cmask-fast-clear");
		}
		if (!cleared) {
			continue; // rejected (reported): the CMASK bytes stay as the guest wrote them
		}
		// After the eliminate every tile is expanded; this also keeps a later binding from
		// clearing again what the draws wrote meanwhile. FillBuffer can fault: no texture lock.
		m_buffer_cache.FillBuffer(metadata.address, metadata.size, UINT32_MAX, false);
		Profiler::CountFrameEvent(Event::CmaskFastClears);
		ClearTrace("CMASK", "cleared", metadata.address, metadata.size, desc.info.data.address,
		           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, slice);
	}
	// Why this decision is not a provable no-op, when it is not (instrumentation).
	const bool any_native = std::ranges::find(native, uint8_t {1}) != native.end();
	m_cmask_noop_refusal  = any_native ? Profiler::FrameEvent::TargetRecordCmaskNative
	                                   : Profiler::FrameEvent::TargetRecordCmaskOther;
	if (!any_native) {
		if (provable) {
			decided.provable = true;
			*noop            = decided;
		}
		return;
	}
	// GPU-owned bytes without a recorded fill (the game's own fill kernel is not a proven uniform
	// fill): the DCC helper's native inspection with CMASK semantics. Only code 0x00 (two
	// fast-cleared tiles per byte) selects a clear, to the CLEAR_WORD colour; a cleared slice's
	// bytes become 0xFF. It needs no readback; the DCC helper when KYTY_DCC_GPU=1 created it,
	// otherwise a CMASK-only instance.
	// KYTY_CMASK_NATIVE=0 decides them by readback instead.
	static constexpr auto native_enabled_init = [] {
		const auto* value = std::getenv("KYTY_CMASK_NATIVE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(native_enabled_init())>> native_enabled_once;
	const bool& native_enabled = native_enabled_once.Get(native_enabled_init);
	static_assert(DccClearHelper::ClearCodes[0] == 0x00);
	if (native_enabled && m_dcc_clear == nullptr && m_cmask_clear == nullptr) {
		m_cmask_clear = std::make_unique<DccClearHelper>(m_graphics, m_scheduler);
	}
	DccClearHelper::ClearValues values {};
	values[0]          = clear.color;
	auto* helper       = m_dcc_clear != nullptr ? m_dcc_clear.get() : m_cmask_clear.get();
	const auto outcome = native_enabled && helper != nullptr
	                         ? TryMaterializeGpuMetadataClear(*helper, id, desc, range,
	                                                          metadata_base_layer, values, 1u, true)
	                         : Event::DccFallbackDisabled;
	ClearTrace("CMASK", "clear-word0", range.address, range.size, desc.info.data.address,
	           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, desc.cmask.clear_word0);
	ClearTrace("CMASK", "clear-word1", range.address, range.size, desc.info.data.address,
	           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, desc.cmask.clear_word1);
	ClearTrace("CMASK", "view-format", range.address, range.size, desc.info.data.address,
	           desc.info.data.size, desc.info.extent.width, desc.info.extent.height, static_cast<uint32_t>(view.format));
	ClearTrace("CMASK", "native-outcome", range.address, range.size, desc.info.data.address,
	           desc.info.data.size, desc.info.extent.width, desc.info.extent.height,
	           static_cast<uint32_t>(outcome));
	if (outcome == Event::DccGpuRecords || outcome == Event::DccGpuReuses) {
		Profiler::CountFrameEvent(outcome == Event::DccGpuRecords
		                              ? Event::CmaskFastClearInspections
		                              : Event::CmaskFastClearInspectionReuses,
		                          view.layer_count);
		return;
	}
	// No native inspection: read the undecided slices back, as the DCC CPU fallback does.
	for (uint32_t slice = 0; slice < view.layer_count; slice++) {
		if (native[slice] == 0) {
			continue;
		}
		const GuestRange metadata {range.address + slice_size * (metadata_base_layer + slice),
		                           slice_size};
		Profiler::CountFrameEvent(Event::CmaskFastClearReadbacks);
		{
			Profiler::ScopedFrameWait frame_wait(Profiler::FrameWait::DccFallback);
			m_buffer_cache.ReadMemory(metadata.address, metadata.size, false);
		}
		std::vector<uint32_t> words(metadata.size / sizeof(uint32_t));
		if (!LibKernel::Memory::TryReadBacking(metadata.address, words.data(), metadata.size)) {
			EXIT("TextureCache: failed to read CMASK bytes\n");
		}
		const bool cleared = std::all_of(words.begin(), words.end(), [](uint32_t word) { return word == 0; });
		if (!cleared) {
			Profiler::CountFrameEvent(std::all_of(words.begin(), words.end(),
			                                      [](uint32_t word) { return word == UINT32_MAX; })
			                              ? Event::CmaskFastClearExpanded
			                              : Event::CmaskFastClearUnproven);
			continue;
		}
		bool applied = false;
		{
			std::scoped_lock lock {m_lock};
			applied = ClearImage(m_scheduler.Current(), id, view.format,
			                     {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			                      view.base_layer + slice, 1},
			                     clear, "cmask-readback");
		}
		if (!applied) {
			continue; // rejected (reported): the CMASK bytes stay as the guest wrote them
		}
		m_buffer_cache.FillBuffer(metadata.address, metadata.size, UINT32_MAX, false);
		Profiler::CountFrameEvent(Event::CmaskFastClears);
	}
}

// TextureBindingMemo::RefreshIsNoOp mirrors when this is a no-op; keep them in sync.
// KYTY_GUEST_HASH_IMAGES_KB=N (default 0: off; adapted from usa-ratt/KytyPS5-GT7 1a344cd80, whose
// default is 256): linear images up to N KiB that the GPU neither wrote nor covers with its writes
// are compared with their guest bytes (a hash) the first time each guest submission refreshes them,
// and uploaded again, straight from guest memory, when the bytes changed. What the game wrote
// before submitting is then visible to all of that submission's commands, as on the console. The
// page tracking alone misses CPU writes to an image's first and last pages once a write next to the
// image untracked them, and an upload from a cached buffer over the range is only as fresh as that
// buffer: GT7's light grids, rewritten by the CPU every frame in memory shared with other data,
// were found stale (dark screen tiles in tunnels and at night).
static uint64_t GuestHashImageBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GUEST_HASH_IMAGES_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull) << 10u;
	}();
	return bytes;
}

// KYTY_GUEST_HASH_TILED_KB=N (default 0: off; the experiment of usa-ratt/KytyPS5-GT7 95b1f059f):
// the same per-submission guest-byte check for tiled images up to N KiB.
static uint64_t GuestHashTiledBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_GUEST_HASH_TILED_KB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull) << 10u;
	}();
	return bytes;
}

// KYTY_GUEST_HASH_IMAGES_KB: the hash of the guest bytes of [address, address + size); false when
// they cannot be read.
static bool HashGuestBytes(uint64_t address, uint64_t size, uint64_t& hash) {
	thread_local std::vector<uint8_t> bytes;
	bytes.resize(size);
	if (size == 0 || !Libs::LibKernel::Memory::TryReadBacking(address, bytes.data(), size)) {
		return false;
	}
	hash = XXH3_64bits(bytes.data(), static_cast<size_t>(size));
	return true;
}

bool TextureCache::ClearFromKnownFill(ImageId id) {
	auto&       image = m_slot_images[id];
	const auto& info  = image.info;
	// A target the binding is about to write, made from scratch: a texture may hold what the CPU
	// wrote into memory a fill's holes left unmapped (those stores are not seen); a target's bytes
	// are the GPU's.
	if (image.depth_id || image.backing.image == nullptr || info.samples != 1 ||
	    image.backing.samples != 1 || info.data.Empty() || info.IsVolume() || info.HasStencil() ||
	    !(image.usage.render_target || image.usage.depth_target) || image.usage.texture ||
	    image.usage.storage || image.usage.video_out ||
	    info.metadata.compression != VideoOutCompression::Uncompressed) {
		return false;
	}
	uint32_t value = 0;
	if (!m_buffer_cache.FillOfNewImage(info.data.address, info.data.size, value, false)) {
		return false;
	}
	vk::ClearValue       clear {};
	vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor;
	if (info.IsDepth()) {
		aspect = ImageViewOps::DepthAspectMask(image.backing.format);
		if (info.pixel_format == vk::Format::eD16Unorm) {
			// Two texels a dword: both the same.
			if ((value >> 16u) != (value & 0xffffu)) {
				return false;
			}
			clear.depthStencil.depth = static_cast<float>(value & 0xffffu) / 65535.0f;
		} else if (!DecodePackedDepthClear(info.pixel_format, value, clear.depthStencil.depth)) {
			return false;
		}
		clear.depthStencil.stencil = 0;
	} else if (!DecodeColorDwordFill(info.pixel_format, value, clear.color)) {
		return false;
	}
	if (!ClearImage(m_scheduler.Current(), id, image.backing.format,
	                {aspect, 0, info.resources.levels, 0, image.backing.layers}, clear,
	                "pending-fill")) {
		return false;
	}
	(void)m_buffer_cache.FillOfNewImage(info.data.address, info.data.size, value, true);
	static std::atomic<uint64_t> cleared {0};
	if (const auto count = cleared.fetch_add(1, std::memory_order_relaxed) + 1;
	    count <= 32 || std::has_single_bit(count)) {
		LOGF("TextureCache: %s 0x%012" PRIx64 "+0x%" PRIx64 " %ux%u layers %u mips %u created inside "
		     "a GPU fill with 0x%08x: starts as the fill, not as its memory (%" PRIu64
		     " so far; KYTY_PENDING_FILLS)\n",
		     info.IsDepth() ? "depth target" : "render target", info.data.address, info.data.size,
		     info.extent.width, info.extent.height, info.resources.layers, info.resources.levels,
		     value, count);
	}
	return true;
}

// KYTY_NAN_SCRUB=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7 9bbca0635 and the
// scan of c5511faad, without its NaN-count diagnostics): the B10G11R11 and half-float arrays of six
// layers or more that the GPU writes (light probes, cube maps), up to KYTY_NAN_SCRUB_MB (default
// 64) in all their mips, have their texels that are not finite rewritten on the GPU at a frame end
// (one image a frame, each at most every 30 frames and only once it changed): a NaN becomes zero,
// an infinity the largest finite value, what the guest GPU stores for a value out of range. A probe
// relit every frame from its last value keeps such texels for good, and the surfaces that reflect
// it shade black or NaN, which the temporal and blur passes spread (GT7).
static bool NanScrubEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_NAN_SCRUB");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

static uint64_t NanScrubMaxBytes() {
	static const uint64_t bytes = [] {
		const char* value = std::getenv("KYTY_NAN_SCRUB_MB");
		return (value != nullptr ? std::strtoull(value, nullptr, 10) : 64ull) << 20u;
	}();
	return bytes;
}

// The scrub's packing of an image's host format and its bytes per texel; nullopt for others.
static std::optional<std::pair<NanScrubber::Packing, uint32_t>> NanScrubFormat(vk::Format format) {
	switch (format) {
		case vk::Format::eB10G11R11UfloatPack32: return std::pair {NanScrubber::Packing::B10G11R11, 4u};
		case vk::Format::eR16G16Sfloat: return std::pair {NanScrubber::Packing::TwoHalves, 4u};
		case vk::Format::eR16G16B16A16Sfloat: return std::pair {NanScrubber::Packing::TwoHalves, 8u};
		default: return std::nullopt;
	}
}

// The bytes of every mip of `image`, tightly packed (0 when it is not eligible for the scrub).
static uint64_t NanScrubBytes(const Image& image) {
	const auto&    backing      = image.backing;
	const auto     format       = NanScrubFormat(backing.format);
	constexpr auto TransferBoth = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (!format || backing.image == nullptr || backing.layers < 6u ||
	    backing.image_type != vk::ImageType::e2D || backing.samples != 1 || image.depth_id ||
	    image.info.IsDepth() || !image.FullyResident() || (backing.usage & TransferBoth) != TransferBoth) {
		return 0;
	}
	uint64_t total = 0;
	for (uint32_t mip = 0; mip < std::max(backing.mip_levels, 1u); mip++) {
		total += uint64_t {std::max(backing.extent.width >> mip, 1u)} *
		         std::max(backing.extent.height >> mip, 1u) * backing.layers * format->second;
	}
	return total <= NanScrubMaxBytes() ? total : 0;
}

void TextureCache::NoteNanScrubCandidate(const Image& image) {
	if (!NanScrubEnabled() || image.info.data.Empty() || NanScrubBytes(image) == 0) {
		return;
	}
	const auto address = image.info.data.address;
	if (std::ranges::any_of(m_nan_candidates,
	                        [address](const NanScrubCandidate& c) { return c.address == address; })) {
		return;
	}
	if (m_nan_candidates.size() >= 256) {
		m_nan_candidates.erase(m_nan_candidates.begin());
	}
	m_nan_candidates.push_back({address, image.info.data.size, UINT64_MAX, 0});
}

void TextureCache::RunNanScrub() {
	if (!NanScrubEnabled()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	const auto       frame = m_frame.load(std::memory_order_relaxed);
	if (frame == m_nan_frame || m_nan_candidates.empty() || !m_scheduler.Active()) {
		return;
	}
	m_nan_frame = frame;
	if (m_nan_scrubber == nullptr) {
		m_nan_scrubber = std::make_unique<NanScrubber>(m_graphics);
	}
	if (!m_nan_scrubber->Valid()) {
		return;
	}
	for (size_t tries = 0; tries < m_nan_candidates.size();) {
		m_nan_next      = (m_nan_next + 1) % m_nan_candidates.size();
		auto& candidate = m_nan_candidates[m_nan_next];
		Image* image    = nullptr;
		for (const auto id: FindImagesInRegion(candidate.address, candidate.size, false)) {
			auto* found = m_slot_images.try_get(id);
			if (found != nullptr && found->registered && found->info.data.address == candidate.address &&
			    NanScrubBytes(*found) != 0) {
				image = found;
				break;
			}
		}
		if (image == nullptr) {
			// Gone: forget it.
			m_nan_candidates.erase(m_nan_candidates.begin() + static_cast<std::ptrdiff_t>(m_nan_next));
			if (m_nan_candidates.empty()) {
				return;
			}
			m_nan_next = (m_nan_next + m_nan_candidates.size() - 1) % m_nan_candidates.size();
			continue;
		}
		tries++;
		if ((candidate.frame != 0 && frame - candidate.frame < 30) ||
		    image->ContentSerial() == candidate.serial) {
			continue;
		}
		const auto  total   = NanScrubBytes(*image);
		const auto  format  = NanScrubFormat(image->backing.format);
		const auto& backing = image->backing;
		std::vector<vk::BufferImageCopy> regions;
		uint64_t                         offset = 0;
		for (uint32_t mip = 0; mip < std::max(backing.mip_levels, 1u); mip++) {
			const uint32_t      width  = std::max(backing.extent.width >> mip, 1u);
			const uint32_t      height = std::max(backing.extent.height >> mip, 1u);
			vk::BufferImageCopy region {};
			region.bufferOffset     = offset;
			region.imageSubresource = {vk::ImageAspectFlagBits::eColor, mip, 0, backing.layers};
			region.imageExtent      = vk::Extent3D {width, height, 1};
			regions.push_back(region);
			offset += uint64_t {width} * height * backing.layers * format->second;
		}
		if (m_nan_scratch == nullptr || m_nan_scratch->Size() < total) {
			// The previous one stays alive in the deferred release of the scrub that used it.
			m_nan_scratch = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
			                                         AllFlags, Common::AlignUp(total, uint64_t {4} << 20u));
		}
		image->Download(regions, m_nan_scratch->Handle(), 0, total);
		m_nan_scrubber->Scrub(m_scheduler.Current(), m_nan_scratch->Handle(), 0, total, format->first);
		image->Upload(regions, m_nan_scratch->Handle(), 0, total);
		m_scheduler.DeferOperation([scratch = m_nan_scratch] {});
		candidate.serial = image->ContentSerial();
		candidate.frame  = frame;
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("TextureCache: NaN scrub of 0x%012" PRIx64 " %ux%u, %u layers, %u mips, format %d "
			     "(KYTY_NAN_SCRUB)\n",
			     candidate.address, backing.extent.width, backing.extent.height, backing.layers,
			     backing.mip_levels, static_cast<int>(backing.format));
		}
		return;
	}
}

bool TextureCache::LightGridGuestEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_LIGHT_GRID_GUEST");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

bool TextureCache::LightGridGuestApplies(const Image& image) {
	return LightGridGuestEnabled() && LightGridCapEnabled() && IsLightGridShape(image.info) &&
	       image.FullyResident() && !image.IsGpuModified() && !image.depth_id;
}

bool TextureCache::ChecksGuestContent(const Image& image) {
	const auto& info  = image.info;
	// KYTY_LIGHT_GRID_GUEST: a light grid is checked against (and uploaded from) its guest bytes
	// even where GPU writes to other data shared its memory.
	if (LightGridGuestApplies(image)) {
		return true;
	}
	const auto  limit = info.IsTiled() ? GuestHashTiledBytes() : GuestHashImageBytes();
	return limit != 0 && !info.data.Empty() && info.data.size <= limit &&
	       info.samples == 1 && !image.depth_id && !info.IsDepth() &&
	       info.metadata.compression == VideoOutCompression::Uncompressed && image.FullyResident() &&
	       !image.IsGpuModified() && !image.IsBufferModified() &&
	       !m_buffer_cache.IsRegionGpuModified(info.data.address, info.data.size);
}

void TextureCache::RefreshByContent(ImageId id, RefreshIntent intent) {
	auto&      image = m_slot_images[id];
	const auto range = image.info.data;
	TrackImage(id);
	const auto submission = SyncEpoch::CurrentSubmission();
	if (image.content_check_submission == submission && !image.IsCpuDirty()) {
		return;
	}
	uint64_t hash = 0;
	if (!HashGuestBytes(range.address, range.size, hash)) {
		image.content_hash_valid = false;
		RefreshImage(id, intent);
		return;
	}
	image.content_check_submission = submission;
	if (hash == image.content_hash) {
		// The GPU holds these bytes already, whatever the tracking marked.
		if (image.IsDefinitelyCpuDirty()) {
			image.RefreshComplete();
		} else if (image.IsMaybeCpuDirty()) {
			image.ClearMaybeCpuDirty();
		}
		return;
	}
	if (!image.IsDefinitelyCpuDirty()) {
		// What this check is for: the first few, then every 1000th.
		static std::atomic<uint64_t> unseen {0};
		if (const auto count = unseen.fetch_add(1, std::memory_order_relaxed) + 1;
		    count <= 16 || count % 1000 == 0) {
			LOGF("TextureCache: guest bytes of 0x%012" PRIx64 "+0x%" PRIx64
			     " (%ux%u, %u layers) changed unseen by the page tracking%s: uploaded again (%" PRIu64
			     " so far; KYTY_GUEST_HASH_IMAGES_KB)\n",
			     range.address, range.size, image.info.extent.width, image.info.extent.height,
			     image.info.resources.layers,
			     image.IsMaybeCpuDirty() ? " (a neighbour's write marked it maybe-dirty)" : "", count);
		}
	}
	image.MarkResidencyDirty();
	InitializeImage(id, intent);
}

void TextureCache::RefreshImage(ImageId id, RefreshIntent intent) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	// KYTY_PRESERVE_STENCIL: a plane the depth image does not hold is refreshed when it is swapped
	// back in (its guest bytes would land on the held plane).
	if (image.depth_id && PreserveStencilEnabled() && !StencilPlaneHeld(image)) {
		return;
	}
	if (image.content_hash_valid && ChecksGuestContent(image)) {
		RefreshByContent(id, intent);
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return;
		}
		(void)image.ResolveMaybeCpuHash(hash);
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id, intent);
}

ImageId TextureCache::FindStencilPlane(const Image& depth, GuestRange range) const {
	for (const auto id: depth.stencil_ids) {
		const auto* stencil = m_slot_images.try_get(id);
		if (stencil != nullptr && stencil->info.data == range) {
			return id;
		}
	}
	return {};
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	// KYTY_STENCIL_TARGETS_APART=1 (default off): an image the GPU draws to at the stencil plane's
	// range (a colour or depth target of a pool of targets reused while an area loads) is not taken
	// as the association record; a record of its own is made beside it. Taken, the target turned
	// into the depth image's from then on and the next draw to it ended the emulator ("color target
	// requires rediscovery before final acquisition"; chenxiao07/KytyPS5 issue #94). Storage images
	// stay eligible here (stencil writes through storage views resolve through the record). The
	// first such cases are logged with the switch on or off.
	static const bool targets_apart = [] {
		const char* value = std::getenv("KYTY_STENCIL_TARGETS_APART");
		return value != nullptr && value[0] == '1';
	}();
	// Not upstream 84419aef7's own-record-first order: an existing image at the stencil address
	// (e.g. one a shader sampled as the stencil plane) becomes the association, as before
	// (RenderExecutorStencilBindingDiscovery "existing stencil association selection").
	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			if (owner->usage.render_target || owner->usage.depth_target) {
				if (static std::atomic<uint32_t> reports {0};
				    reports.fetch_add(1, std::memory_order_relaxed) < 4) {
					std::printf("TextureCache: a %s target at 0x%" PRIx64
					            " shares its range with a depth image's stencil plane: %s\n",
					            owner->usage.depth_target ? "depth" : "color", stencil.address,
					            targets_apart ? "kept apart" : "taken as the stencil record");
				}
				if (targets_apart) {
					continue;
				}
			}
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	if (record.depth_id != depth_id) {
		// A binding of this image now resolves to the depth image (TextureBindingMemo).
		NoteStructureChange(record);
		// Reparented: the previous depth owner no longer retires it (Image::stencil_ids).
		if (record.depth_id) {
			if (auto* previous = m_slot_images.try_get(record.depth_id); previous != nullptr) {
				std::erase(previous->stencil_ids, association);
			}
		}
		depth.stencil_ids.push_back(association);
	}
	if (record.depth_id != depth_id || record.stencil_subresources != depth.stencil_subresources) {
		// KYTY_IMAGE_PARTIAL_GPU_WRITE: other native stencil contents behind this record.
		record.foreign_since_sync = true;
		record.gpu_write.active   = false;
	}
	record.depth_id             = depth_id;
	record.stencil_subresources = depth.stencil_subresources;
	NoteImageStateChange();
	return association;
}

// KYTY_VIEW_PLACEMENT=1 (default off; adapted from usa-ratt/KytyPS5-GT7 rounds 71 and 74,
// 509d22411 / e0200a03a): a texture or target the overlap loop finds inside an older, larger image
// is viewed from its own place in the image the loop ends with (where its memory starts there, or
// the last overlap's place when that overlap placed the binding itself), plus its descriptor's own
// first mip and layer (BASE_LEVEL / BASE_ARRAY, a target's SLICE_START). Without it the last
// overlap's place is used, whichever image that overlap was given, and the binding's own first
// layer is dropped: GT7's reflection faces read their light grids through T#s of one 48-layer
// grid with BASE_ARRAY 0, 8, ... 40, and a face whose grid lay inside an older image read the
// first face's layers or other data (light counts of 65535 a pixel). A binding with no place, or
// one its view does not fit in, gets an image of its own (logged).
bool TextureCache::ViewPlacementEnabled() {
	static constexpr auto on_init = [] {
		const char* value = std::getenv("KYTY_VIEW_PLACEMENT");
		const bool  set   = value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
		if (set) {
			LOGF("TextureCache: bindings inside larger images keep their own place and first layer "
			     "(KYTY_VIEW_PLACEMENT)\n");
		}
		return set;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

void TextureCache::PlaceBindingView(const ImageDesc& desc, ImageId& result, bool loop_own,
                                    int32_t& view_mip, int32_t& view_layer) const {
	const auto& found      = m_slot_images[result].info;
	const auto  loop_mip   = view_mip;
	const auto  loop_layer = view_layer;
	view_mip               = -1;
	view_layer             = -1;
	if (found.data.address == desc.info.data.address) {
		if (loop_own) {
			view_mip   = loop_mip;
			view_layer = loop_layer;
		}
		return;
	}
	if (const auto mip = desc.info.MipOf(found); mip >= 0) {
		view_mip   = mip;
		view_layer = desc.info.SliceOf(found, mip);
	} else if (loop_own) {
		view_mip   = loop_mip;
		view_layer = loop_layer;
	}
	const bool fits = view_mip >= 0 && view_layer >= 0 &&
	                  static_cast<uint64_t>(view_mip) + desc.view_info.base_level + desc.view_info.level_count <=
	                      found.resources.levels &&
	                  static_cast<uint64_t>(view_layer) + desc.view_info.base_layer + desc.view_info.layer_count <=
	                      found.resources.layers;
	if (fits) {
		return;
	}
	static std::atomic<uint64_t> unplaced {0};
	const auto                   count = unplaced.fetch_add(1, std::memory_order_relaxed) + 1;
	if (count <= 16u || (count & (count - 1u)) == 0u) {
		LOGF("TextureCache: 0x%012" PRIx64 " (%ux%u, mips %u, layers %u, view mips %u+%u layers %u+%u) lies "
		     "in 0x%012" PRIx64 " (%ux%u, mips %u, layers %u) at mip %d layer %d, where its view does not "
		     "fit: given an image of its own (%" PRIu64 " so far, KYTY_VIEW_PLACEMENT)\n",
		     desc.info.data.address, desc.info.extent.width, desc.info.extent.height, desc.info.resources.levels,
		     desc.info.resources.layers, desc.view_info.base_level, desc.view_info.level_count,
		     desc.view_info.base_layer, desc.view_info.layer_count, found.data.address, found.extent.width,
		     found.extent.height, found.resources.levels, found.resources.layers, view_mip, view_layer, count);
	}
	view_mip   = -1;
	view_layer = -1;
	result     = {};
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, RepeatLookup* record) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	if (record != nullptr) {
		*record = {};
	}
	// Draw preparation guarantees a valid command buffer (upstream e827c92a2 dropped the check).
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	// Instrumentation: why a record did not become valid before the metadata decisions.
	auto lookup_refusal = Profiler::FrameEvent::TargetRecordNotFirstPage;
	{
		std::scoped_lock lock {m_lock};
		ImageIds candidates;
		const bool use_legacy = m_image_lookup_mode != ImageLookupMode::FirstPage;
		// The answer is the first-page lookup's, unchanged by anything below (RepeatLookup).
		bool first_page_answer = false;
		if (use_legacy) {
			candidates = FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
			for (const auto id: candidates) {
				if (SameBacking(m_slot_images[id].info, desc.info, exact_format)) {
					result = id;
				}
			}
			if (m_image_lookup_mode == ImageLookupMode::Verify) {
				const auto fast = FindImageWithSameBacking(desc.info, exact_format);
				++m_image_lookup_checks;
				if (fast != result) {
					++m_image_lookup_mismatches;
					if (m_image_lookup_mismatches <= 32) {
						LOGF("Image lookup mismatch: address=0x%016" PRIx64
						     " size=0x%016" PRIx64 " exact=%u fast=%u:%u legacy=%u:%u\n",
						     desc.info.data.address, desc.info.data.size, exact_format ? 1u : 0u,
						     fast.index, fast.generation, result.index, result.generation);
					}
				}
				if ((m_image_lookup_checks & 0xffffu) == 0) {
					LOGF("Image lookup verification: checks=%" PRIu64 " mismatches=%" PRIu64 "\n",
					     m_image_lookup_checks, m_image_lookup_mismatches);
				}
			}
		} else {
			result            = FindImageWithSameBacking(desc.info, exact_format);
			first_page_answer = static_cast<bool>(result);
		}

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		// KYTY_IMG_STATS / KYTY_IMG_OVERLAP: whether the overlap loop below takes only branches that
		// are pure functions of the region's owner lists and the images' fixed fields.
		m_overlap_impure = false;
		m_last_overlap   = {};
		if (ImgSide::StatsEnabled()) [[unlikely]] {
			m_last_overlap.requested = desc.info; // KYTY_IMG_STATS (backing split)
		}
		bool inserted    = false;
		// Levels this lookup can sample. A new image registers only their prefix, and images that
		// overlap only levels it cannot sample are neither merge candidates nor retired by it.
		const auto wanted_first    = RequestedFirstLevel(desc, desc.info.resources.levels);
		uint64_t   wanted_prefix   = 0;
		bool       prefix_computed = false;
		if (!result) {
			if (!use_legacy) {
				wanted_prefix   = ResidentPrefixSize(desc.info, wanted_first);
				prefix_computed = true;
				candidates    = FindImagesInRegion(
				    desc.info.data.address,
				    wanted_prefix != 0 ? wanted_prefix : desc.info.data.size, false);
			}
			constexpr size_t CrowdedOverlap = 24;
			m_overlap_crowded = candidates.size() > CrowdedOverlap;
			if (!use_legacy) {
				m_last_overlap.overlap        = true;
				m_last_overlap.region_address = desc.info.data.address;
				m_last_overlap.region_size    = wanted_prefix != 0 ? wanted_prefix : desc.info.data.size;
				m_last_overlap.candidates     = static_cast<uint32_t>(candidates.size());
			}
			const bool placement  = ViewPlacementEnabled();
			ImageId    placed_in  = {};
			bool       placed_raw = false; // placed against the binding's own address
			for (const auto candidate: candidates) {
				if (!placement) {
					view_mip   = -1;
					view_layer = -1;
				}
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const bool  direct      = merged_info.data.address == desc.info.data.address;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result = overlap.image;
					if (!placement || overlap.mip >= 0 || overlap.layer >= 0) {
						view_mip   = overlap.mip;
						view_layer = overlap.layer;
					}
					placed_in  = overlap.mip >= 0 || overlap.layer >= 0 ? overlap.image : ImageId {};
					placed_raw = direct;
				}
			}
			if (result && placement) {
				PlaceBindingView(desc, result, placed_in == result && placed_raw, view_mip, view_layer);
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (ImageContainsEnabled() ? !resolved.info.resources.Contains(desc.info.resources)
			                                  : resolved.info.resources < desc.info.resources) {
				NoteOverlapImpure(OverlapLookup::WhyContains);
				FreeImage(result, HangTrace::ImageFreeReason::SmallerResources);
				result = {};
			}
		}
		if (!result) {
			first_page_answer = false;
			if (ViewPlacementEnabled()) {
				// The binding gets an image of its own: no place in another one applies.
				view_mip   = -1;
				view_layer = -1;
			}
			// A prefix computed as not applicable (0) keeps the image fully resident; InsertImage
			// computes it when this lookup did not.
			result = InsertImage(desc.info,
			                     prefix_computed && wanted_prefix == 0 ? 0u : wanted_first,
			                     wanted_prefix);
			inserted = true;
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.live.address, inserted.live.size) ||
			    (AliasBytesEnabled() && OtherImagesOwnBytes(inserted.live, result))) {
				inserted.MarkBufferModified();
			}
		}
		auto& image = m_slot_images[result];
		m_last_overlap.pure       = m_last_overlap.overlap && !inserted && !m_overlap_impure;
		m_last_overlap.inserted   = inserted;
		m_last_overlap.first_page = first_page_answer;
		m_last_overlap.result     = result;
		// KYTY_VIEW_PLACEMENT: the binding's view starts at its place plus its own first mip and
		// layer (the T#'s BASE_LEVEL and BASE_ARRAY, a target's SLICE_START) when they fit;
		// otherwise (and without the switch) at its place alone.
		const bool keep_own = ViewPlacementEnabled();
		if (view_mip >= 0) {
			auto level = static_cast<uint32_t>(view_mip);
			if (keep_own && level + desc.view_info.base_level + desc.view_info.level_count <=
			                    image.info.resources.levels) {
				level += desc.view_info.base_level;
			}
			desc.view_info.base_level = level;
		}
		if (view_layer >= 0) {
			auto layer = static_cast<uint32_t>(view_layer);
			if (keep_own && layer + desc.view_info.base_layer + desc.view_info.layer_count <=
			                    image.info.resources.layers) {
				layer += desc.view_info.base_layer;
			}
			desc.view_info.base_layer = layer;
		}
		// Every level the (adjusted) view can sample must be resident: a finer MIN_LOD than
		// before extends the image, any non-sampling binding makes the whole chain resident.
		EnsureResidency(result, RequestedFirstLevel(desc, image.info.resources.levels),
		                desc.type == BindingType::Texture);
		SyncAliasFromOwner(result);
		if (BlockSubresourceWritesEnabled()) {
			ApplyBlockSubwrites(result);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
		// RepeatLookup: the state a repeat is checked against, taken after the residency extension
		// and alias synchronization above (which may re-register the image or make it the owner).
		// Video-out lookups also check the image's contents below: never repeated.
		ImagePageTable::PageRange pages {};
		if (record != nullptr && first_page_answer && image.registered &&
		    desc.type != BindingType::VideoOut &&
		    ImagePageTable::TryGetPageRange(desc.info.data.address, desc.info.data.size, pages) &&
		    FindImageWithSameBacking(desc.info, exact_format) == result) {
			record->image           = result;
			record->page            = pages.first;
			record->page_version    = PageVersion(pages.first);
			record->requested_first = RequestedFirstLevel(desc, image.info.resources.levels);
			record->exact_format    = exact_format;
			record->has_partner     = HasAliasPartner(pages.first, result, image);
			record->valid           = true;
		} else if (first_page_answer) {
			lookup_refusal = Profiler::FrameEvent::TargetRecordChanged;
		}
	}
	const bool recording = record != nullptr && record->valid;
	MaterializeDccClear(result, desc, metadata_base_layer, recording ? &record->dcc : nullptr);
	MaterializeCmaskClear(result, desc, metadata_base_layer, recording ? &record->cmask : nullptr);
	if (recording) {
		record->valid = record->dcc.provable && record->cmask.provable;
		if (!record->valid) {
			Profiler::CountFrameEvent(!record->dcc.provable ? m_dcc_noop_refusal
			                                                : m_cmask_noop_refusal);
		}
	} else if (record != nullptr) {
		Profiler::CountFrameEvent(lookup_refusal);
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

bool TextureCache::CaptureMetadataPages(GuestRange range, MetadataNoop& noop) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(range.address, range.size, pages) ||
	    pages.last_exclusive - pages.first > MetadataNoop::MaxPages) {
		return false;
	}
	noop.first_page = pages.first;
	noop.page_count = static_cast<uint32_t>(pages.last_exclusive - pages.first);
	for (uint32_t i = 0; i < noop.page_count; i++) {
		noop.page_versions[i] = PageVersion(pages.first + i);
	}
	return true;
}

bool TextureCache::MetadataPagesHold(const MetadataNoop& noop) const {
	for (uint32_t i = 0; i < noop.page_count; i++) {
		if (PageVersion(noop.first_page + i) != noop.page_versions[i]) {
			return false;
		}
	}
	return true;
}

bool TextureCache::MetadataStateHolds(const MetadataNoop& noop) {
	if (noop.fill_generation != 0 &&
	    m_buffer_cache.KnownFillGeneration() != noop.fill_generation) {
		return false;
	}
	for (uint32_t i = 0; i < noop.gpu_range_count; i++) {
		const auto& range = noop.gpu_ranges[i];
		if (!m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
			return false;
		}
	}
	if (noop.guest_key_count != 0 && !GuestKeysHold(noop)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::CpCommitDccGuestRejects);
		return false;
	}
	return true;
}

bool TextureCache::GuestKeysHold(const MetadataNoop& noop) {
	// Still not GPU-dirty: MaterializeDccClear's CPU-owned branch. Only the GPU thread sets
	// GPU-dirty bits, so on it the lock-free mirror answers "not GPU-dirty" exactly (other threads
	// only clear them); elsewhere the locked query decides.
	const auto& range = noop.guest_range;
	if (GuestGpu::IsGpuThread() ? m_buffer_cache.IsRegionGpuModifiedRelaxed(range.address, range.size)
	                            : m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		return false;
	}
	// Each slice's first key still reads as the decision read it (so it again decodes to no clear,
	// and nothing else of its slice is read).
	for (uint32_t i = 0; i < noop.guest_key_count; i++) {
		uint8_t code = 0;
		if (!LibKernel::Memory::TryReadBacking(noop.guest_key_addresses[i], &code, sizeof(code)) ||
		    code != noop.guest_key_codes[i]) {
			return false;
		}
	}
	return true;
}

bool TextureCache::RepeatGpuRangesLost(const RepeatLookup& record) {
	for (const auto* noop: {&record.dcc, &record.cmask}) {
		for (uint32_t i = 0; i < noop->gpu_range_count; i++) {
			const auto& range = noop->gpu_ranges[i];
			if (!m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
				return true;
			}
		}
		// A guest write to a recorded key (or a GPU write of the metadata) since the repeat check
		// raced the lookup the same way.
		if (noop->guest_key_count != 0 && !GuestKeysHold(*noop)) {
			return true;
		}
	}
	return false;
}

bool TextureCache::TryRepeatLookup(const ImageDesc& desc, bool exact_format,
                                   const RepeatLookup& record, bool apply) {
	if (!record.valid || record.exact_format != exact_format) {
		return false;
	}
	if (m_scheduler.Current().IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	// The metadata decisions read these outside the texture-cache lock, as FindImage's do.
	if (!MetadataStateHolds(record.dcc) || !MetadataStateHolds(record.cmask)) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(record.image);
	if (m_image_lookup_mode != ImageLookupMode::FirstPage || image == nullptr ||
	    !image->registered || PageVersion(record.page) != record.page_version ||
	    record.requested_first < image->resident_first ||
	    (record.has_partner && !SyncAliasReturnsAtOnce(*image)) || !image->block_subwrites.empty() ||
	    !MetadataPagesHold(record.dcc) || !MetadataPagesHold(record.cmask)) {
		return false;
	}
	if (!apply) {
		return true;
	}
	// FindImage's access bookkeeping, and MaterializeDccClear's for a DCC description (it assigns
	// the metadata and drops any other interpretation of those bytes before deciding anything).
	image->tick_accessed_last = m_scheduler.CurrentTick();
	TouchImage(*image);
	if (desc.info.metadata.kind == ImageMetadataKind::Dcc) {
		image->info.metadata = desc.info.metadata;
		EraseSurfaceMeta(desc.info.metadata.range.address);
	}
	return true;
}

void TextureCache::EraseSurfaceMeta(uint64_t address) {
	// Every removal bumps the surface-metadata generation (KYTY_META_CLEAR_MEMO's validity); an
	// erase that finds nothing changes nothing.
	if (!CpCommit::Enabled(CpCommit::Part::MetaErase)) {
		if (m_surface_metas.erase(address) != 0) {
			NoteSurfaceMetaChange();
		}
		return;
	}
	// KYTY_CP_COMMIT=metaerase: erased here before, and nothing inserted since (FindDepthTarget is
	// the only insertion), so the address is still absent and the erase would do nothing.
	auto& memo = m_meta_erases[(address >> 12u) % m_meta_erases.size()];
	if (memo.address == address && memo.inserts == m_surface_meta_inserts) {
		return;
	}
	if (m_surface_metas.erase(address) != 0) {
		NoteSurfaceMetaChange();
	}
	memo = {address, m_surface_meta_inserts};
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	// External readers (presentation, depth copies) read every level.
	RequireFullResidency(id);
	RefreshImage(id);
}

namespace TexelImageLookup {

namespace {

// Live switch (common/liveSwitch.h): every call chooses between equivalent queries.
Live::Switch g_mode("KYTY_TEXEL_IMAGE_LOOKUP",
                    [](const char* value) { return static_cast<int64_t>(ParseMode(value)); });

Mode GetMode() {
	return static_cast<Mode>(g_mode.Get());
}

bool StatsEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* stats  = std::getenv("KYTY_TEXEL_IMAGE_LOOKUP_STATS");
		const auto* commit = std::getenv("KYTY_CP_COMMIT_STATS");
		return (stats != nullptr && std::strcmp(stats, "1") == 0) ||
		       (commit != nullptr && commit[0] != '\0' && std::strcmp(commit, "0") != 0) ||
		       GetMode() == Mode::Verify || GetMode() == Mode::Exit;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled || GetMode() == Mode::Verify || GetMode() == Mode::Exit;
}

struct Totals {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> range_pages {0}; // pages a walk of the whole range visits
	std::atomic<uint64_t> found {0};
	std::atomic<uint64_t> verify_checks {0};
	std::atomic<uint64_t> verify_mismatches {0};
};
Totals g_totals;

} // namespace

void PrintSummary() {
	if (!StatsEnabled()) {
		return;
	}
	static uint64_t                last_ns = 0;
	static std::array<uint64_t, 5> last {};
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (last_ns == 0) {
		last_ns = now;
		return;
	}
	if (now - last_ns < 10'000'000'000ull) {
		return;
	}
	const std::array<uint64_t, 5> values {g_totals.calls.load(std::memory_order_relaxed),
	                                      g_totals.range_pages.load(std::memory_order_relaxed),
	                                      g_totals.found.load(std::memory_order_relaxed),
	                                      g_totals.verify_checks.load(std::memory_order_relaxed),
	                                      g_totals.verify_mismatches.load(std::memory_order_relaxed)};
	static const char* const names[] = {"page", "range", "verify", "exit"};
	std::printf("TexelImageLookup %.0fs (%s): %" PRIu64 " calls, %" PRIu64
	            " pages a range walk visits (%.2f per call), %" PRIu64 " found; verify %" PRIu64
	            " checks, %" PRIu64 " mismatches\n",
	            static_cast<double>(now - last_ns) * 1e-9, names[static_cast<uint32_t>(GetMode())],
	            values[0] - last[0], values[1] - last[1],
	            values[0] != last[0] ? static_cast<double>(values[1] - last[1]) /
	                                       static_cast<double>(values[0] - last[0])
	                                 : 0.0,
	            values[2] - last[2], values[3] - last[3], values[4] - last[4]);
	std::fflush(stdout);
	last    = values;
	last_ns = now;
}

} // namespace TexelImageLookup

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid,
                                         bool buffer_sync) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	using TexelImageLookup::Mode;
	const auto mode  = TexelImageLookup::GetMode();
	const bool stats = TexelImageLookup::StatsEnabled();
	if (stats) {
		TexelImageLookup::g_totals.calls.fetch_add(1, std::memory_order_relaxed);
		TexelImageLookup::g_totals.range_pages.fetch_add(
		    TexelImageLookup::PagesVisited(address, size, ImagePageTable::kPageBits),
		    std::memory_order_relaxed);
	}
	std::scoped_lock lock {m_lock};
	if (mode == Mode::Verify || mode == Mode::Exit) {
		const auto page  = FindImageFromRangeLocked(address, size, 1, ensure_valid, buffer_sync);
		const auto range = FindImageFromRangeLocked(address, size, size, ensure_valid, buffer_sync);
		TexelImageLookup::g_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
		if (page != range) {
			const auto count =
			    TexelImageLookup::g_totals.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
			if (count < 64) {
				std::printf("TexelImageLookupVerify: 0x%" PRIx64 "+0x%" PRIx64
				            " ensure_valid=%u buffer_sync=%u: first page found image %u, the range "
				            "walk image %u\n",
				            address, size, ensure_valid ? 1u : 0u, buffer_sync ? 1u : 0u,
				            static_cast<uint32_t>(page.index), static_cast<uint32_t>(range.index));
				std::fflush(stdout);
			}
			if (mode == Mode::Exit) {
				EXIT("TexelImageLookupVerify: the first-page query found a different image\n");
			}
		}
		if (stats && range) {
			TexelImageLookup::g_totals.found.fetch_add(1, std::memory_order_relaxed);
		}
		return range;
	}
	const auto selected = FindImageFromRangeLocked(
	    address, size, TexelImageLookup::QuerySize(mode, size), ensure_valid, buffer_sync);
	if (stats && selected) {
		TexelImageLookup::g_totals.found.fetch_add(1, std::memory_order_relaxed);
	}
	return selected;
}

ImageId TextureCache::FindImageFromRangeLocked(uint64_t address, uint64_t size, uint64_t query_size,
                                               bool ensure_valid, bool buffer_sync) {
	ImageIds matches;
	for (const auto id: FindImagesInRegion(address, query_size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr ||
		    (ensure_valid && !(buffer_sync ? SafeToSyncIntoBuffer(*owner) : SafeToDownload(*owner)))) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

// KYTY_IMAGE_LAYER_DIAG=<layers> (diagnostic, default off): every 5 s, for each sampled image with
// that many array layers, log its guest range and how many 64 KiB chunks of the guest memory behind
// it are nonzero, to tell an image that missed guest writes from guest memory that never got them.
static void ImageLayerDiag(const ImageInfo& info) {
	static constexpr auto layers_init = [] {
		const auto* value = std::getenv("KYTY_IMAGE_LAYER_DIAG");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 0)) : 0u;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(layers_init())>> layers_once;
	const uint32_t& layers = layers_once.Get(layers_init);
	if (layers == 0 || info.resources.layers != layers || info.data.Empty()) {
		return;
	}
	static std::mutex                                                     mutex;
	static std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> last;
	const auto                                                            now = std::chrono::steady_clock::now();
	{
		std::scoped_lock lock {mutex};
		auto [it, inserted] = last.try_emplace(info.data.address, now);
		if (!inserted && now - it->second < std::chrono::seconds(5)) {
			return;
		}
		it->second = now;
	}
	constexpr uint64_t   Chunk = 64 * 1024;
	std::vector<uint8_t> bytes(Chunk);
	uint64_t             chunks = 0, nonzero = 0, unreadable = 0, first = UINT64_MAX;
	std::string          map;
	for (uint64_t at = info.data.address; at < info.data.End(); at += Chunk, chunks++) {
		const auto size = std::min(Chunk, info.data.End() - at);
		if (!LibKernel::Memory::TryReadBacking(at, bytes.data(), size)) {
			unreadable++;
			map += 'u';
			continue;
		}
		const bool any = std::any_of(bytes.begin(), bytes.begin() + static_cast<int64_t>(size),
		                             [](uint8_t b) { return b != 0; });
		if (any) {
			nonzero++;
			first = std::min(first, at - info.data.address);
		}
		map += any ? '#' : '.';
	}
	LOGF("Image layer diag: 0x%016" PRIx64 " size=0x%" PRIx64 " %ux%u layers=%u levels=%u fmt=%u: "
	     "%" PRIu64 "/%" PRIu64 " chunks nonzero, %" PRIu64 " unreadable, first nonzero +0x%" PRIx64
	     " map %s\n",
	     info.data.address, info.data.size, info.extent.width, info.extent.height,
	     info.resources.layers, info.resources.levels, static_cast<uint32_t>(info.guest_format),
	     nonzero, chunks, unreadable, first == UINT64_MAX ? 0 : first, map.c_str());
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	ImageLayerDiag(m_slot_images[id].info);
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
		// The view's levels must be resident (a storage binding: the whole chain). FindImage
		// already did this for the same description; binding caches may skip FindImage.
		EnsureResidency(id, RequestedFirstLevel(desc, image.info.resources.levels),
		                desc.type == BindingType::Texture);
	}
	// Storage readback (upstream 9678c59b7) also queues the first GPU write since the CPU last
	// wrote the image: an image rewritten every frame would otherwise be downloaded at every GL2
	// writeback. GPU data in the buffer cache over the whole image is not a CPU write. Read before
	// the image is marked GPU-written and before RefreshImage uploads the CPU write.
	const bool first_write =
	    m_readback_storage_images && desc.type == BindingType::Storage && !image.info.data.Empty() &&
	    (!image.IsGpuModified() || image.IsDefinitelyCpuDirty()) &&
	    !m_buffer_cache.HasOnlyGpuDirtyBytes(image.info.data.address, image.info.data.size);
	if (desc.type == BindingType::Storage) {
		m_bind_mark = true;
		MarkImageGpuModified(image);
		m_bind_mark = false;
		image.shader_write_serial = Image::NextContentSerial(); // KYTY_STENCIL_PLANE_SYNC
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id, desc.type == BindingType::Storage ? RefreshIntent::Write
		                                                   : RefreshIntent::Read);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			if (const auto stencil_id = FindStencilPlane(image, image.info.stencil)) {
				RefreshImage(stencil_id); // upstream 84419aef7
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
				if (BlockSubresourceWritesEnabled()) {
					NoteBlockSubwrite(id);
				}
			}
			TrackImageDownload(id, image);
			if (first_write) {
				m_download_images.insert(id);
			}
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc,
                                             const vk::Rect2D* written) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	RequireFullResidency(id);
	TouchImage(image);
	m_bind_mark = true;
	MarkImageGpuModified(image);
	m_bind_mark = false;
	image.usage.render_target = true;
	RefreshImage(id, RefreshIntent::Write);
	// KYTY_ALIAS_BYTES: the draw can write only inside its scissor. The target owns the 64 KiB
	// blocks under it; other images keep the rest of its range.
	WriteClaim claim;
	// KYTY_CP_COMMIT=targetalloc: the block list is a reused member (m_lock held), so an empty one
	// costs no heap allocation (a std::map allocates its head node when constructed).
	std::optional<RangeSet> local_blocks;
	RangeSet&               blocks = CpCommit::Enabled(CpCommit::Part::TargetAlloc)
	                                     ? m_claim_blocks
	                                     : local_blocks.emplace();
	blocks.Clear();
	if (written != nullptr && AliasBytesEnabled() && !image.OwnsAllBytes()) {
		auto rect = ClampRect(*written, image.info.extent.width, image.info.extent.height);
		if (image.alias_owner && image.claimed_rect_valid &&
		    RectContains(image.claimed_rect, rect)) {
			// Every block under the scissor is owned already (the draws of one pass): claim
			// nothing new, without building the block list.
			claim.ranges = &blocks;
			claim.rect   = rect;
		} else if (BlocksUnderRect(image, rect, blocks)) {
			claim.ranges = &blocks;
			claim.rect   = rect;
			Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesBoundedClaims);
		}
	}
	CommitGpuWrite(image, claim);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	RequireFullResidency(id);
	TouchImage(image);
	// KYTY_IMAGE_PARTIAL_GPU_WRITE: this binding's draws mark its stencil records below, after their
	// refresh (the draws come after it).
	m_stencil_draw_suppressed = true;
	m_bind_mark               = true;
	MarkImageGpuModified(image);
	m_bind_mark               = false;
	m_stencil_draw_suppressed = false;
	image.usage.depth_target = true;
	bool stencil_swapped_in  = false; // KYTY_STENCIL_RESTORE_GUARD
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
		if (PreserveStencilEnabled() && image.info.HasStencil() &&
		    image.info.stencil.address != desc.info.stencil.address) {
			stencil_swapped_in = SwapStencilPlane(id, desc.info.stencil.address);
		}
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		// Inserts only when the key is absent, without building and freeing a node (and its
		// 8192-slice clear mask) every draw when it is present.
		const auto [metadata, inserted] =
		    m_surface_metas.try_emplace(desc.info.metadata.range.address);
		if (inserted) {
			metadata->second.type = MetaDataInfo::Type::HTile;
			if (image.info.htile_initial_clear) {
				metadata->second.clear_mask.set();
			}
		}
		metadata->second.size = std::max(metadata->second.size, desc.info.metadata.range.size);
		metadata->second.owner = id;
		// The only insertion into m_surface_metas (EraseSurfaceMeta). Every draw with this target
		// asks again; only an added entry is a change (KYTY_META_CLEAR_MEMO's generation).
		m_surface_meta_inserts += inserted ? 1u : 0u;
		// KYTY_HTILE_SLICE_CLEAR: the per-layer HTILE size, from a binding of the whole image.
		if (HtileSliceClearEnabled()) {
			const auto layers = desc.info.resources.layers;
			const auto size   = desc.info.metadata.range.size;
			if (layers > 1 && layers == image.info.resources.layers &&
			    desc.info.data.address == image.info.data.address && size % layers == 0) {
				metadata->second.slice_size = size / layers;
				metadata->second.layers     = layers;
			}
		}
		if (inserted) {
			NoteSurfaceMetaChange();
		}
	}
	RefreshImage(id, RefreshIntent::Write);
	m_stencil_draw_suppressed = true;
	CommitGpuWrite(image);
	m_stencil_draw_suppressed = false;
	if (ReadbackLinearCopiesEnabled() && !image.info.IsTiled()) {
		TrackImageDownload(id, image);
	}
	if (desc.info.HasStencil()) {
		const auto association = AssociateStencil(id, desc.info.stencil);
		if (stencil_swapped_in && StencilRestoreGuardEnabled()) {
			// KYTY_STENCIL_RESTORE_GUARD=1 (Erlou93/KytyPS5-Yotei FindDepthTarget swap-in rule): the
			// restored plane is newer than guest memory, so only a GPU buffer write to the plane's
			// memory made while it was not held (the game's fill before drawing into it) is uploaded
			// over it; a CPU-dirty mark alone is not (a new plane record starts CPU-dirty).
			if (m_slot_images[association].IsBufferModified()) {
				RefreshImage(association, RefreshIntent::Write);
			}
			if (m_slot_images[association].IsCpuDirty()) {
				m_slot_images[association].RefreshComplete();
			}
		} else {
			RefreshImage(association, RefreshIntent::Write);
		}
	}
	if (m_partial_gpu_write != 0 && !m_partial_gpu_write_stencil_keep) {
		// The draws of this binding come after the stencil record's refresh above.
		for (const auto record: image.stencil_ids) {
			if (auto* stencil = m_slot_images.try_get(record); stencil != nullptr) {
				stencil->foreign_since_sync = true;
			}
		}
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	// Before the caller records its write: levels it does not write must hold guest data.
	RequireFullResidency(id);
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

// KYTY_BLOCK_SUBRESOURCE_WRITES=1 (default off; PPSA26344 candidate). Ghost of Yotei builds its
// BC6 reflection/ambient probe array (256x256, 6 levels, 102 layers) on the GPU: a compute
// shader encodes each level of each updated layer through its own R32G32B32A32_UINT image
// (64x64 .. 2x2, one texel per BC6 block) at that level's address inside the array. Those
// images are not subresources of the array for the cache (other format, extent in blocks), so
// the array kept its load-time contents: the composite's environment term stayed stale, the snow
// water lost its cyan sky reflection and the ground was darker. With the switch a GPU write to
// such an image is queued on the block image and copied into its (level, layer) at its next
// lookup.
static bool BlockSubresourceWritesEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_BLOCK_SUBRESOURCE_WRITES");
		const bool  on    = value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
		if (on) {
			std::printf("KYTY_BLOCK_SUBRESOURCE_WRITES: uncompressed block views written by the GPU "
			            "are copied into their block-compressed image\n");
		}
		return on;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

// The (level, layer) of `container` (block-compressed) whose blocks are the texels of `child`
// (uncompressed, one level, one layer, same bytes per block and tiling), or {-1, -1}.
static std::pair<int32_t, int32_t> BlockSubresourceOf(const ImageInfo& child,
                                                      const ImageInfo& container) {
	constexpr std::pair<int32_t, int32_t> none {-1, -1};
	if (!container.IsBlock() || child.IsBlock() || child.IsVolume() || container.IsVolume() ||
	    child.bytes_per_block != container.bytes_per_block || child.tile_mode != container.tile_mode ||
	    child.samples != 1 || container.samples != 1 || child.resources.levels != 1 ||
	    child.resources.layers != 1 || container.resources.layers == 0 ||
	    container.resources.levels > container.mip_layout.size() || !child.data.Valid() ||
	    !container.data.Valid() || child.data.address < container.data.address ||
	    child.data.address >= container.data.End() || child.HasStencil() ||
	    container.data.size % container.resources.layers != 0) {
		return none;
	}
	const auto slice_stride  = container.data.size / container.resources.layers;
	const auto address_delta = child.data.address - container.data.address;
	const auto layer         = address_delta / slice_stride;
	const auto in_slice      = address_delta % slice_stride;
	// The level is the one whose block extent the child has (unique per level). A level outside
	// the mip tail starts exactly at the child; the tail levels (here 8x8 .. 2x2 texels) are packed
	// below the smallest full level at offsets of their own, so a tail child only has to lie
	// between its level's offset and the next larger level.
	for (uint32_t level = 0; level < container.resources.levels; level++) {
		const auto width  = std::max(container.extent.width >> level, 1u);
		const auto height = std::max(container.extent.height >> level, 1u);
		if (child.extent.width != (width + 3) / 4 || child.extent.height != (height + 3) / 4) {
			continue;
		}
		const auto& layout = container.mip_layout[level];
		// The tail levels share one offset; the smallest full level starts above them.
		uint64_t next_larger = slice_stride;
		for (uint32_t larger = 0; larger < level; larger++) {
			if (container.mip_layout[larger].offset > layout.offset) {
				next_larger = std::min<uint64_t>(next_larger, container.mip_layout[larger].offset);
			}
		}
		bool shared = false;
		for (uint32_t other = 0; other < container.resources.levels; other++) {
			shared = shared || (other != level && container.mip_layout[other].offset == layout.offset);
		}
		const bool tail = level > 0 && shared && layout.offset <= in_slice && in_slice < next_larger;
		if (layout.offset != in_slice && !tail) {
			return none;
		}
		return {static_cast<int32_t>(level), static_cast<int32_t>(layer)};
	}
	return none;
}

// Caller holds m_lock. `self` is bound as a storage image (its GPU write follows).
void TextureCache::NoteBlockSubwrite(ImageId self) {
	auto& image = m_slot_images[self];
	if (image.info.IsBlock() || image.info.resources.levels != 1 ||
	    image.info.resources.layers != 1 || image.info.data.Empty() || image.depth_id ||
	    image.backing.image == nullptr || image.info.bytes_per_block != 16) {
		return;
	}
	const auto ids = FindImagesInRegion(image.info.data.address, image.info.data.size, false);
	for (const auto id: ids) {
		auto* other = m_slot_images.try_get(id);
		if (other == nullptr || other == &image || !other->registered || other->depth_id ||
		    other->backing.image == nullptr || !other->info.IsBlock()) {
			continue;
		}
		const auto [mip, layer] = BlockSubresourceOf(image.info, other->info);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 24) {
			std::printf("KYTY_BLOCK_SUBRESOURCE_WRITES: 0x%" PRIx64 " %ux%u tile %u in block image "
			            "0x%" PRIx64 " %ux%u levels %u layers %u tile %u (offsets of levels 2-5: 0x%" PRIx64
			            " 0x%" PRIx64 " 0x%" PRIx64 " 0x%" PRIx64 ") -> mip %d layer %d\n",
			            image.info.data.address, image.info.extent.width, image.info.extent.height,
			            static_cast<uint32_t>(image.info.tile_mode), other->info.data.address,
			            other->info.extent.width, other->info.extent.height,
			            other->info.resources.levels, other->info.resources.layers,
			            static_cast<uint32_t>(other->info.tile_mode),
			            static_cast<uint64_t>(other->info.mip_layout[2].offset),
			            static_cast<uint64_t>(other->info.mip_layout[3].offset),
			            static_cast<uint64_t>(other->info.mip_layout[4].offset),
			            static_cast<uint64_t>(other->info.mip_layout[5].offset), mip, layer);
		}
		if (mip < 0 || static_cast<uint32_t>(mip) >= other->backing.mip_levels ||
		    static_cast<uint32_t>(layer) >= other->backing.layers) {
			continue;
		}
		auto& pending = other->block_subwrites;
		if (std::none_of(pending.begin(), pending.end(), [&](const Image::BlockSubwrite& w) {
			    return w.source == self && w.mip == static_cast<uint32_t>(mip) &&
			           w.layer == static_cast<uint32_t>(layer);
		    })) {
			pending.push_back({self, static_cast<uint32_t>(mip), static_cast<uint32_t>(layer)});
		}
	}
}

// Caller holds m_lock: copies the queued block views into `id` (recorded after their writes).
void TextureCache::ApplyBlockSubwrites(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.block_subwrites.empty()) {
		return;
	}
	auto pending = std::move(image.block_subwrites);
	image.block_subwrites.clear();
	if (image.backing.image == nullptr) {
		return;
	}
	for (const auto& write: pending) {
		auto* source = m_slot_images.try_get(write.source);
		if (source == nullptr || !source->registered || source->backing.image == nullptr ||
		    source->IsBufferModified() || write.mip >= m_slot_images[id].backing.mip_levels ||
		    write.layer >= m_slot_images[id].backing.layers) {
			continue;
		}
		RefreshCopySource(write.source);
		RequireFullResidency(id);
		TrackImage(id);
		m_slot_images[id].CopyBlockMip(m_slot_images[write.source], write.mip, write.layer);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyDirect);
	}
}

bool TextureCache::SyncAliasReturnsAtOnce(const Image& image) {
	return !AliasAgeByFrames() || image.alias_owner || image.depth_id || image.info.data.Empty() ||
	       (image.info.HasStencil() && !StencilDepthAliasEnabled()) || image.backing.image == nullptr;
}

// Whether two images at the same guest range are interpretations of the same texel grid. Same
// extent, or (KYTY_ALIAS_BLOCK_VIEWS=1; default off in this line, on in the PPSA01474 and
// PPSA03671 title presets) a block-compressed image and an
// uncompressed one whose texels are its blocks: same block grid, block size, depth and
// subresources. Ratchet & Clank writes its BC6 reflection-probe array (256x256x384) through a
// R32G32B32A32_UINT 64x64x384 view of the same memory; with extents compared as texels the
// written view never synced into the BC6 image, every probe but the first stayed black and areas
// lit only by probes were pitch black.
static bool SameAliasGrid(const ImageInfo& a, const ImageInfo& b) {
	if (a.extent == b.extent) {
		return true;
	}
	static const bool block_views = [] {
		const auto* value = std::getenv("KYTY_ALIAS_BLOCK_VIEWS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return block_views && a.IsBlock() != b.IsBlock() && a.BlockExtent() == b.BlockExtent() &&
	       a.extent.depth == b.extent.depth && a.bytes_per_block == b.bytes_per_block &&
	       a.resources == b.resources && a.tile_mode == b.tile_mode && !a.IsVolume() &&
	       !b.IsVolume();
}

bool TextureCache::IsAliasSyncSource(const Image& other, const Image& image) {
	return other.registered && !other.depth_id && other.alias_owner && other.IsGpuModified() &&
	       !other.IsBufferModified() && !(other.info.HasStencil() && !StencilDepthAliasEnabled()) &&
	       other.info.data == image.info.data && SameAliasGrid(other.info, image.info) &&
	       other.backing.samples == image.backing.samples;
}

bool TextureCache::HasAliasPartner(uint64_t page, ImageId found, const Image& image) const {
	bool partner = false;
	if (const auto* owners = m_image_page_table.Find(page); owners != nullptr) {
		owners->ForEach([&](ImageId id) {
			const auto* other = m_slot_images.try_get(id);
			if (id != found && other != nullptr && other->registered &&
			    other->info.data == image.info.data && SameAliasGrid(other->info, image.info) &&
			    other->backing.samples == image.backing.samples) {
				partner = true;
			}
		});
	}
	return partner;
}

void TextureCache::SyncAliasFromOwner(ImageId id) {
	auto& image = m_slot_images[id];
	ImageWatch("image alias sync", image);
	// TextureCache::SyncAliasReturnsAtOnce mirrors this condition.
	if (!AliasAgeByFrames() || image.alias_owner || image.depth_id || image.info.data.Empty() ||
	    (image.info.HasStencil() && !StencilDepthAliasEnabled()) || image.backing.image == nullptr) {
		return;
	}
	if (WatchRange::Enabled()) {
		WatchRange::Note(fmt::format("image alias sync check {} {}x{}", vk::to_string(image.backing.format),
		                             image.backing.extent.width, image.backing.extent.height)
		                     .c_str(),
		                 image.info.data.address, image.info.data.size);
	}
	// A kept alias (same memory, another format) is stale when another interpretation wrote the
	// memory since. Reinterpret the owner's contents, as the recreate path used to, and take over.
	for (const auto other_id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
		const auto* other = m_slot_images.try_get(other_id);
		// IsAliasSyncSource is this condition (TextureBindingMemo's KYTY_IMG_PARTNER relies on it).
		if (other_id == id || other == nullptr || !IsAliasSyncSource(*other, image)) {
			continue;
		}
		++m_lookup_side_effects;
		// Taking over the owner's contents makes this image GPU-owned: whole chain.
		RequireFullResidency(id);
		const auto trace_sync = [&](const char* reason, uint64_t bytes) {
			HangTrace::RecordTransfer(HangTrace::TransferKind::AliasSync, reason, "",
			                          image.info.data.address,
			                          static_cast<uint32_t>(image.backing.format),
			                          image.info.extent.width, image.info.extent.height, bytes, 0);
		};
		if (AliasBytesEnabled() && !other->OwnsAllBytes()) {
			// The owner holds only some of the bytes (it wrote part of the range, or other writes
			// took the rest): its contents are current only there. Its bytes go through the
			// buffer, and this alias is rebuilt from there.
			const auto moved = MaterializeOwnedBytes(image.info.data, other_id, {}, "alias-sync");
			image.MarkBufferModified();
			trace_sync("partial-owner", moved);
			return;
		}
		// Skip the copy when this alias already holds exactly the owner's native bits: its last
		// contents came from a lossless full copy of the owner (or the other way round) and
		// neither image has been written since. Ownership moves exactly as after a copy.
		if (AliasSyncSkipEnabled() && image.ContentSerial() != 0 &&
		    image.ContentSerial() == other->ContentSerial() && !image.IsCpuDirty() &&
		    !image.IsBufferModified()) {
			// A CPU-dirty owner is refreshed first, exactly as CopyImage would; that upload is
			// a write and gives it a new serial.
			RefreshCopySource(other_id);
			const auto& owner = m_slot_images[other_id];
			if (image.ContentSerial() == owner.ContentSerial() && owner.IsGpuModified() &&
			    !owner.IsBufferModified() && !image.IsCpuDirty() && !image.IsBufferModified()) {
				const auto serial = owner.ContentSerial();
				TrackImage(id);
				CommitGpuWrite(image);
				image.AdoptContentSerial(serial);
				Profiler::CountFrameEvent(Profiler::FrameEvent::AliasSyncSkips);
				trace_sync("skip-same-contents", 0);
				return;
			}
		}
		if (WatchRange::Enabled()) {
			// Diagnostic: why the copy was not skipped (serials, dirty and buffer-owned flags).
			const auto& owner = m_slot_images[other_id];
			WatchRange::Note(fmt::format("alias sync copy: alias serial {} owner serial {} alias cpu_dirty {} "
			                             "alias buffer_mod {} owner buffer_mod {} owner definite {}",
			                             image.ContentSerial(), owner.ContentSerial(), image.IsCpuDirty(),
			                             image.IsBufferModified(), owner.IsBufferModified(),
			                             owner.DefiniteWrites())
			                     .c_str(),
			                 image.info.data.address, image.info.data.size);
		}
		const bool lossless = CopyImage(id, other_id, "alias-sync");
		// CopyImage marks the alias buffer-owned instead of copying when the owner's bytes are
		// in the buffer: then nothing was copied.
		const bool copied = !image.IsBufferModified();
		CommitGpuWrite(image);
		if ((lossless || (copied && AliasSyncDepthPlane(image, m_slot_images[other_id]))) &&
		    AliasSyncSkipEnabled()) {
			image.AdoptContentSerial(m_slot_images[other_id].ContentSerial());
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::AliasSyncCopies);
		trace_sync("copy", image.info.data.size);
		return;
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	CommitGpuWrite(image, WriteClaim {});
}

void TextureCache::CommitGpuWrite(Image& image, const WriteClaim& claim) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	NoteExactRangeRewrite(image); // KYTY_IMAGE_EXACT_RANGE_INVALIDATE bookkeeping only
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	// KYTY_GUEST_HASH_IMAGES_KB: the GPU's pixels from now on, not the guest bytes an upload took.
	image.content_hash_valid = false;
	NoteNanScrubCandidate(image); // KYTY_NAN_SCRUB
	// KYTY_PENDING_FILLS: its pixels are newer than a fill of its memory, so an image made there
	// later must not start as that fill (ClearFromKnownFill).
	if (!image.depth_id && !image.info.data.Empty()) {
		m_buffer_cache.ForgetFillsUnder(image.info.data.address, image.info.data.size);
	}
	if (AliasBytesEnabled() && !image.depth_id && !image.info.data.Empty()) {
		// Single owner per byte: this write takes the claimed bytes (the whole image, or the
		// blocks under a draw's scissor) from every other image, which keeps the rest of its
		// bytes. Scanned only when ownership changes, not on every bind.
		const bool whole   = claim.ranges == nullptr;
		bool       covered = false;
		if (whole) {
			covered = image.OwnsAllBytes();
		} else if (claim.rect && image.claimed_rect_valid &&
		           RectContains(image.claimed_rect, *claim.rect)) {
			covered = true;
		} else {
			covered = image.OwnsAllBytes() || image.OwnedSet() != nullptr;
			claim.ranges->ForEach([&](uint64_t begin, uint64_t end) {
				covered = covered && image.OwnsBytes(begin, end - begin);
			});
		}
		if (!image.alias_owner || !covered) {
			for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size,
			                                       false)) {
				auto* other = m_slot_images.try_get(id);
				if (other == nullptr || other == &image || other->depth_id || !other->registered ||
				    !other->Overlaps(image.info.data.address, image.info.data.size, false)) {
					continue;
				}
				other->alias_owner = false;
				// KYTY_IMAGE_PARTIAL_GPU_WRITE: this write replaces bytes of the other image.
				other->foreign_since_sync = true;
				if (!other->IsGpuModified()) {
					continue;
				}
				bool lost = false;
				if (whole) {
					lost = other->DisownBytes(image.info.data.address, image.info.data.size);
				} else {
					claim.ranges->ForEach([&](uint64_t begin, uint64_t end) {
						lost = other->DisownBytes(begin, end - begin) || lost;
					});
				}
				if (!other->IsGpuModified()) {
					InvalidateCleanImageProofs(other->live.address, other->live.size,
					                           Coherence::Source::ImageGpuClear);
				} else if (lost) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesKept);
				}
			}
			image.alias_owner = true;
		}
		MarkImageGpuModified(image);
		if (whole) {
			image.OwnAllBytes();
			// Its native contents are now the contents of every byte of its range.
			image.foreign_since_sync = false;
			image.claimed_rect       = {{0, 0}, {image.info.extent.width, image.info.extent.height}};
			image.claimed_rect_valid = true;
		} else {
			claim.ranges->ForEach(
			    [&](uint64_t begin, uint64_t end) { image.OwnBytes(begin, end - begin); });
			if (claim.rect && (!image.claimed_rect_valid ||
			                   RectContains(*claim.rect, image.claimed_rect))) {
				image.claimed_rect       = *claim.rect;
				image.claimed_rect_valid = true;
			}
			if (image.OwnsNoBytes()) {
				// Nothing claimed (an empty scissor): no GPU ownership.
				image.ClearGpuModified();
				InvalidateCleanImageProofs(image.live.address, image.live.size,
				                           Coherence::Source::ImageGpuClear);
			}
		}
		return;
	}
	// Single owner among live aliases: this write supersedes the bytes of every other image
	// overlapping it, so none of them may later be downloaded over it. Their native contents
	// stay as they are (the previous rule freed them without a download). Scanned only when
	// ownership changes, not on every bind.
	if (AliasAgeByFrames() && !image.alias_owner && !image.depth_id && !image.info.data.Empty()) {
		for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* other = m_slot_images.try_get(id);
			if (other == nullptr || other == &image || other->depth_id || !other->registered ||
			    !other->Overlaps(image.info.data.address, image.info.data.size, false)) {
				continue;
			}
			other->alias_owner = false;
			other->foreign_since_sync = true;
			if (other->IsGpuModified()) {
				other->ClearGpuModified();
				InvalidateCleanImageProofs(other->live.address, other->live.size,
				                           Coherence::Source::ImageGpuClear);
			}
		}
		image.alias_owner = true;
	}
	MarkImageGpuModified(image);
}

// KYTY_STENCIL_PLANE_FILL_CLEAR=1 (default off; GT7 preset): ClearImageFromBuffer also takes a fill
// that covers exactly a depth target's stencil plane as a stencil clear of that target.
bool StencilPlaneFillClearEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_STENCIL_PLANE_FILL_CLEAR");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

void TextureCache::ResetUnmappedDepth(ImageId id, float empty) {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_SPARSE_HOLES");
		return value != nullptr && value[0] == '1';
	}();
	if (!enabled || !id) {
		return;
	}
	std::scoped_lock lock {m_lock};
	auto*            found = m_slot_images.try_get(id);
	if (found == nullptr || found->sparse_checked_serial == found->ContentSerial()) {
		return;
	}
	auto&       image           = *found;
	const auto& info            = image.info;
	image.sparse_checked_serial = image.ContentSerial();
	const auto     format       = image.backing.format;
	const uint32_t bytes        = format == vk::Format::eD16Unorm    ? 2u
	                              : format == vk::Format::eD32Sfloat ? 4u
	                                                                 : 0u;
	constexpr uint64_t BlockBytes = 0x10000u;
	if (bytes == 0 || image.backing.image == nullptr || image.depth_id || info.data.Empty() ||
	    info.tile_mode != Prospero::TileMode::kDepth || info.HasStencil() || info.IsVolume() ||
	    info.resources.levels != 1 || info.resources.layers != 1 || image.backing.samples != 1 ||
	    image.backing.mip_levels != 1 || (info.data.address % BlockBytes) != 0 ||
	    !(image.backing.usage & vk::ImageUsageFlagBits::eTransferDst)) {
		return;
	}
	std::vector<std::pair<uint64_t, uint64_t>> spans;
	if (!Libs::LibKernel::Memory::CommittedSpans(info.data.address, info.data.size, 1u << 20u,
	                                             &spans)) {
		return;
	}
	uint64_t committed = 0;
	for (const auto& span: spans) {
		committed += span.second;
	}
	if (committed >= info.data.size) {
		return; // all mapped: the common case
	}
	TileBlockLayout block {};
	if (!TileGetBlockLayout(TileBlockFamily::Depth64KB, bytes, block) ||
	    block.block_size != BlockBytes || block.block_depth != 1 || block.block_width == 0 ||
	    block.block_height == 0) {
		return;
	}
	const uint32_t width          = std::max(image.backing.extent.width, 1u);
	const uint32_t height         = std::max(image.backing.extent.height, 1u);
	const uint32_t blocks_per_row = (width + block.block_width - 1u) / block.block_width;
	const uint32_t block_rows     = (height + block.block_height - 1u) / block.block_height;
	const uint64_t blocks         = uint64_t {blocks_per_row} * block_rows;
	if (blocks * BlockBytes > info.data.size) {
		return; // not the layout assumed here (64 KB blocks, row by row)
	}
	// Whether each block is mapped: any byte of it is (a block partly mapped keeps its texels).
	std::vector<bool> mapped(blocks, false);
	for (const auto& [start, size]: spans) {
		const uint64_t first = (start - info.data.address) / BlockBytes;
		const uint64_t end   = std::min<uint64_t>(
            (start + size - info.data.address + BlockBytes - 1u) / BlockBytes, blocks);
		for (uint64_t b = first; b < end; b++) {
			mapped[b] = true;
		}
	}
	std::vector<vk::BufferImageCopy> regions;
	uint64_t                         largest = 0;
	uint64_t                         holes   = 0;
	for (uint32_t row = 0; row < block_rows; row++) {
		for (uint32_t column = 0; column < blocks_per_row;) {
			if (mapped[uint64_t {row} * blocks_per_row + column]) {
				column++;
				continue;
			}
			uint32_t end = column;
			while (end < blocks_per_row && !mapped[uint64_t {row} * blocks_per_row + end]) {
				end++;
			}
			holes += end - column;
			const uint32_t      x = column * block.block_width;
			const uint32_t      y = row * block.block_height;
			vk::BufferImageCopy region {};
			region.imageSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
			region.imageOffset      = vk::Offset3D {static_cast<int32_t>(x), static_cast<int32_t>(y), 0};
			region.imageExtent      = vk::Extent3D {std::min(end * block.block_width, width) - x,
                                               std::min(block.block_height, height - y), 1};
			largest = std::max<uint64_t>(largest, uint64_t {region.imageExtent.width} *
			                                          region.imageExtent.height * bytes);
			regions.push_back(region);
			column = end;
		}
	}
	if (regions.empty()) {
		return;
	}
	// One passing-depth source for every region (each reads it from its start).
	const uint64_t source_size = Common::AlignUp(largest, uint64_t {4});
	auto           source      = std::make_shared<Buffer>(
        m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, source_size);
	const uint32_t empty_word =
	    bytes == 2u ? (empty >= 1.0f ? 0xffffffffu : 0u) : std::bit_cast<uint32_t>(empty);
	source->Fill(0, source_size, empty_word);
	image.Upload(regions, source->Handle(), 0, source_size);
	m_scheduler.DeferOperation([source] {});
	image.sparse_checked_serial = image.ContentSerial();
	static std::mutex                   logged_mutex;
	static std::unordered_set<uint64_t> logged;
	std::lock_guard                     logged_lock(logged_mutex);
	if (logged.size() < 32 &&
	    logged.insert(info.data.address ^ (uint64_t {width} << 40u) ^ (holes << 20u)).second) {
		LOGF("Sparse depth: 0x%012" PRIx64 " %ux%u: %" PRIu64 " of %" PRIu64
		     " 64 KB blocks unmapped, reset to depth %.1f (%zu regions; KYTY_SPARSE_HOLES)\n",
		     info.data.address, width, height, holes, blocks, static_cast<double>(empty),
		     regions.size());
	}
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	KYTY_GPU_OP_SITE("texcache.clear_from_buffer");
	if (WatchRange::Enabled()) WatchRange::Note("image clear from buffer", address, size);
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	const bool trace_images = address == MetaTraceAddress() && MetaTraceLine();
	// KYTY_STENCIL_PLANE_SYNC=2: a fill that matches a stencil plane record and other images (an R8
	// colour alias) goes into the plane's saved copy when its depth image holds another plane
	// (Erlou93/KytyPS5-Yotei clears the alias and loads it into the plane at swap-in).
	const bool plane_fill_first = StencilPlaneSyncLevel() >= 2 && PreserveStencilEnabled() &&
	                              StencilPlaneFillSavedEnabled();
	std::vector<std::pair<ImageId, ImageId>> plane_records; // (depth, record)
	bool                                     conflict = false;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (trace_images && owner != nullptr) {
			LOGF("MetaTrace: clear-from-buffer 0x%" PRIx64 " sees image data 0x%" PRIx64 "+0x%" PRIx64 " stencil 0x%" PRIx64
			     "+0x%" PRIx64 " depth %d depth_id %d has_stencil %d\n",
			     address, owner->info.data.address, owner->info.data.size, owner->info.stencil.address,
			     owner->info.stencil.size, owner->info.IsDepth() ? 1 : 0, owner->depth_id ? 1 : 0,
			     owner->info.HasStencil() ? 1 : 0);
		}
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		} else if (StencilPlaneFillClearEnabled() && !owner->depth_id && owner->info.IsDepth() &&
		           owner->info.HasStencil() && owner->backing.image != nullptr &&
		           owner->info.stencil.address == address && owner->info.stencil.size == size) {
			// KYTY_STENCIL_PLANE_FILL_CLEAR: a fill of exactly a depth target's stencil plane clears its
			// stencil aspect (GT7's course-map preview: CS 0xfdc68063b0791828 every frame).
			candidate = vk::ImageAspectFlagBits::eStencil;
		}
		if (!candidate) {
			continue;
		}
		if (plane_fill_first && candidate == vk::ImageAspectFlagBits::eStencil && candidate_id != id) {
			plane_records.emplace_back(candidate_id, id);
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			if (!plane_fill_first) {
				return false;
			}
			conflict = true;
			continue;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (conflict) {
		uint8_t value = 0;
		if (!DecodePackedStencilClear(packed_clear, value)) {
			return false;
		}
		for (const auto& [depth_id, record_id]: plane_records) {
			if (FillStencilPlaneSaved(command, depth_id, record_id, value, "fill+alias")) {
				TrackImage(record_id);
				CommitGpuWrite(m_slot_images[record_id]);
				if (trace_images) {
					LOGF("MetaTrace: clear-from-buffer 0x%" PRIx64 " size 0x%" PRIx64 ": plane saved copy %u\n", address,
					     size, static_cast<unsigned>(value));
				}
				return true;
			}
		}
		return false;
	}
	// KYTY_STENCIL_PLANE_FILL_CLEAR: a depth image's stencil plane is not on its registered pages; look
	// for depth images up to 64 MiB below the fill whose stencil plane is exactly the filled range.
	if (!selected && StencilPlaneFillClearEnabled() && address > 0x10000u) {
		constexpr uint64_t Window = uint64_t {64} << 20u;
		const uint64_t     begin  = address > Window ? address - Window : 0x10000u;
		for (const auto id: FindImagesInRegion(begin, address - begin, false)) {
			const auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr || owner->depth_id || !owner->info.IsDepth() || !owner->info.HasStencil() ||
			    owner->backing.image == nullptr || owner->info.stencil.address != address ||
			    owner->info.stencil.size != size) {
				continue;
			}
			const ImageSubresourceRange candidate_subresources {0, owner->info.resources.levels, 0,
			                                                    owner->info.TransferLayers()};
			if (selected && (selected != id || subresources != candidate_subresources)) {
				return false;
			}
			selected     = id;
			stencil_id   = id;
			aspect       = vk::ImageAspectFlagBits::eStencil;
			subresources = candidate_subresources;
		}
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		// KYTY_WIDE_FILL_CLEAR (default off, live): a fill of an image with 64- or 128-bit texels is a clear
		// too; otherwise only a zero RGBA16F fill and the 32-bit formats are.
		// KYTY_WIDE_FILL_CLEAR=2 (GT7 fork 83707d403): the fill words of more formats (DecodeFillWord).
		const auto wide_fill = g_wide_fill_clear.Get();
		// KYTY_10_11_11_RED: the fill word is in the guest's layout.
		const auto host_clear = GuestColorWordToHost(packed_clear, image.info.GetColorTransform());
		// KYTY_STENCIL_PLANE_SYNC=2: a uniform byte fill of an R8 image is its clear (Ghost of Yotei's CS
		// 2bd3cd12 zero-fills the R8 image at its z-fail volume plane 0x50615a0000 every frame; Erlou93/
		// KytyPS5-Yotei clears that image and loads it into the plane at swap-in).
		const uint8_t byte      = static_cast<uint8_t>(packed_clear & 0xffu);
		const bool    byte_fill = packed_clear == uint32_t {byte} * 0x01010101u;
		const auto    r8        = image.backing.format;
		if (StencilPlaneSyncLevel() >= 2 && PreserveStencilEnabled() && byte_fill &&
		    (r8 == vk::Format::eR8Uint || r8 == vk::Format::eR8Unorm)) {
			if (r8 == vk::Format::eR8Uint) {
				clear.color.uint32 = std::array<uint32_t, 4> {byte, 0u, 0u, 0u};
			} else {
				clear.color.float32 = std::array<float, 4> {static_cast<float>(byte) / 255.0f, 0.0f, 0.0f, 0.0f};
			}
			if (trace_images) {
				LOGF("MetaTrace: clear-from-buffer 0x%" PRIx64 " size 0x%" PRIx64 ": R8 byte clear %u\n", address, size,
				     static_cast<unsigned>(byte));
			}
		} else if (!(wide_fill == 2   ? DecodeFillWord(image.info.pixel_format, host_clear, clear.color)
		             : wide_fill == 1 ? DecodeFilledColorClear(image.info.pixel_format, host_clear, clear.color)
		                              : DecodeColorDwordFill(image.info.pixel_format, host_clear, clear.color))) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	// KYTY_STENCIL_PLANE_FILL_SAVED (default on, acts with KYTY_PRESERVE_STENCIL; based on
	// Erlou93/KytyPS5-Yotei KYTY_FIX_STENCIL_PLANE_WRITES): a fill of a stencil plane record whose
	// depth image holds another plane fills that plane's saved copy, which the next swap-in restores,
	// instead of clearing the depth image's stencil aspect (the plane it holds).
	if (stencil_id && aspect == vk::ImageAspectFlagBits::eStencil && PreserveStencilEnabled() &&
	    StencilPlaneFillSavedEnabled() &&
	    FillStencilPlaneSaved(command, selected, stencil_id, clear.depthStencil.stencil, "fill")) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
		return true;
	}
	// A rejected clear (reported) returns false: the guest's own fill shader then runs.
	if (!ClearImage(command, selected, image.backing.format,
	                {aspect, subresources.base_level, subresources.level_count,
	                 subresources.base_layer, subresources.layer_count},
	                clear, "clear-from-buffer")) {
		return false;
	}
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

namespace {

// A clear the target cannot take is reported once per distinct signature (printf: reaches the
// console users send; LOGF does not) and then skipped by the caller of ClearImage.
void ReportRejectedClear(const char* site, ImageId id, const Image& image, ClearRangeFault fault,
                         const ClearTarget& target, const ClearRange& range) {
	const auto& info = image.info;
	uint64_t    hash = 0xcbf29ce484222325ull;
	const auto  mix  = [&hash](uint64_t value) {
		hash = (hash ^ value) * 0x100000001b3ull;
	};
	for (const char* c = site != nullptr ? site : ""; *c != 0; ++c) {
		mix(static_cast<uint8_t>(*c));
	}
	mix(static_cast<uint64_t>(fault));
	mix(static_cast<uint64_t>(info.guest_format));
	mix(static_cast<uint64_t>(info.pixel_format));
	mix(static_cast<uint64_t>(image.backing.format));
	mix(target.aspects | (target.stencil_companion ? 0x100u : 0u) | (info.IsVolume() ? 0x200u : 0u));
	mix(target.levels);
	mix(target.layers);
	mix((static_cast<uint64_t>(info.extent.width) << 32) | info.extent.height);
	mix(range.aspect_mask);
	mix((static_cast<uint64_t>(range.base_level) << 32) | range.level_count);
	mix((static_cast<uint64_t>(range.base_layer) << 32) | range.layer_count);
	static std::mutex              mutex;
	static std::array<uint64_t, 128> seen {};
	static size_t                  seen_count = 0;
	{
		std::scoped_lock lock {mutex};
		if (std::find(seen.begin(), seen.begin() + seen_count, hash) != seen.begin() + seen_count) {
			return;
		}
		if (seen_count < seen.size()) {
			seen[seen_count++] = hash;
		}
	}
	std::printf(
	    "TextureCache: ClearImage skipped (%s): site=%s image=%u:%u guest_format=%u host_format=%s "
	    "backing_format=%s depth=%u stencil=%u volume=%u depth_id=%u stencil_companion=%u "
	    "levels=%u layers=%u backing_layers=%u target_layers=%u extent=%ux%ux%u "
	    "address=0x%016" PRIx64 " size=0x%016" PRIx64
	    " range={aspect=0x%x level=%u+%u layer=%u+%u} target_aspects=0x%x\n",
	    ClearRangeFaultName(fault), site != nullptr ? site : "?", id.index, id.generation,
	    static_cast<uint32_t>(info.guest_format), vk::to_string(info.pixel_format).c_str(),
	    vk::to_string(image.backing.format).c_str(), info.IsDepth() ? 1u : 0u,
	    info.HasStencil() ? 1u : 0u, info.IsVolume() ? 1u : 0u, image.depth_id.index,
	    image.depth_id ? 1u : 0u, info.resources.levels, info.resources.layers,
	    image.backing.layers, target.layers, info.extent.width, info.extent.height,
	    info.extent.depth, info.data.address, info.data.size, range.aspect_mask, range.base_level,
	    range.level_count, range.base_layer, range.layer_count, target.aspects);
	std::fflush(stdout);
}

} // namespace

// Returns false (and changes nothing) when the target image cannot take the clear. The guest
// sources of a skipped clear stay as they are: the image keeps its previous contents and ownership
// (nothing is committed as GPU-written), so stale pixels are never published as the cleared result,
// and callers must not consume the guest's clear key (metadata) or fill.
bool TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear,
                              const char* site) {
	KYTY_GPU_OP_SITE("texcache.clear");
	ImageWatch(site != nullptr ? site : "image clear", m_slot_images[id]);
	if (command.IsInvalid()) {
		EXIT("TextureCache: invalid command buffer for an image clear (%s)\n",
		     site != nullptr ? site : "?");
	}
	{
		// Validated before any state changes (residency, tracking) so a rejection is a pure no-op.
		auto&      target_image = m_slot_images[id];
		const auto aspects      = target_image.info.IsDepth()
		                              ? ImageViewOps::DepthAspectMask(target_image.backing.format)
		                              : vk::ImageAspectFlagBits::eColor;
		const ClearTarget target {
		    .levels = target_image.info.resources.levels,
		    .layers = ClearTargetLayers(target_image.info.IsVolume(),
		                                target_image.info.extent.depth, range.baseMipLevel,
		                                target_image.backing.layers),
		    .aspects = static_cast<uint32_t>(VkImageAspectFlags(aspects)),
		    .stencil_companion = static_cast<bool>(target_image.depth_id),
		    .format_aliased    = format != target_image.backing.format,
		    .volume            = target_image.info.IsVolume()};
		const ClearRange request {
		    .aspect_mask = static_cast<uint32_t>(VkImageAspectFlags(range.aspectMask)),
		    .base_level  = range.baseMipLevel,
		    .level_count = range.levelCount,
		    .base_layer  = range.baseArrayLayer,
		    .layer_count = range.layerCount};
		const auto fault = CheckClearRange(target, request);
		if (fault != ClearRangeFault::None) {
			ReportRejectedClear(site, id, target_image, fault, target, request);
			return false;
		}
	}
	// Levels outside the cleared range must hold guest data, and the result is GPU-owned.
	RequireFullResidency(id);
	auto& image = m_slot_images[id];
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		image.shader_write_serial = Image::NextContentSerial(); // KYTY_STENCIL_PLANE_SYNC
	}
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Recorder-aware (no drain): the transitions go through the batcher (Identity() targets).
	const auto sink = command.HelperSink();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		// Shape validated up front (ClearRangeFault::ViewClearShape).
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, sink.Identity());
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eClear;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		attachment.clearValue  = clear;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {
		    std::max(image.info.extent.width >> range.baseMipLevel, 1u),
		    std::max(image.info.extent.height >> range.baseMipLevel, 1u)};
		rendering.layerCount           = range.layerCount;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		sink.beginRendering(rendering);
		sink.endRendering();
		image.NoteContentWrite();
		CommitGpuWrite(image);
		return true;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              sink.Identity());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		sink.clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal, &clear.color,
		                     1, &native_range);
	} else {
		sink.clearDepthStencilImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                            &clear.depthStencil, 1, &native_range);
	}
	image.NoteContentWrite();
	CommitGpuWrite(image);
	return true;
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	if (m_fault_fast_path) {
		// Guest write faults on pages only buffers watch must not wait for the texture-cache
		// lock, which the GPU thread holds across refreshes. An image watches a page only while
		// registered on its 1 MiB page (counted before tracking starts, and after it ends).
		ImagePageTable::PageRange pages {};
		bool                      covered = !ImagePageTable::TryGetPageRange(address, size, pages);
		for (auto page = pages.first; !covered && page < pages.last_exclusive; ++page) {
			covered = m_image_page_counts[page].load() != 0;
		}
		// KYTY_IMAGE_WRITEBACK_GUARD: a pending image write-back watches pages of its own.
		covered = covered || m_pending_writeback_count.load(std::memory_order_acquire) != 0;
		if (!covered) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureInvalidateSkips);
			return;
		}
	}
	// Images that stop watching these pages release them when the scope ends, after m_lock is
	// released (KYTY_DEFER_UNPROTECT). Constructed before the lock guard, destroyed after it.
	const PageManager::DeferUnprotectScope defer_unprotect;
	std::scoped_lock                       lock {m_lock};
	InvalidateCpuAliases(address, size);
	NotePendingWritebackCpuWrite(address, size);
}

void TextureCache::DownloadDepth(Image& image, vk::Buffer destination, uint64_t destination_offset,
                                 uint64_t destination_capacity) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.data.size / layers;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	if (guest_slice > full_slice_size && Common::SkipInvalidEnabled()) {
		// As in UploadImage, a slice its range cannot hold is not written back.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			LOGF("TextureCache: depth image 0x%012" PRIx64 "+0x%" PRIx64
			     " %ux%u, %u layers: a slice needs 0x%" PRIx64 " bytes and has 0x%" PRIx64
			     "; not written back\n",
			     info.data.address, info.data.size, info.extent.width, info.extent.height, layers,
			     guest_slice, full_slice_size);
		}
		return;
	}
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	auto copies = BuildDepthCopies(info, full_slice_size, vk::ImageAspectFlagBits::eDepth);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			for (auto& copy: copies) {
				copy.bufferOffset += destination_offset;
			}
			image.Download(copies, destination, destination_offset, info.data.size);
			return;
		}
		const auto tiles = BuildDepthTiles(info);
		m_tiler.TileImage(image, copies, destination, destination_offset, info.data.size,
		                  info.data.size, tiles);
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	for (uint32_t layer = 0; layer < layers; layer++) {
		copies[layer].bufferOffset = transfer_slice * layer;
	}
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size)
	                                : TileManager::Result {destination, destination_offset,
	                                                       destination_capacity};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (!tiled) {
		return;
	}
	const auto tiles = BuildDepthTiles(info);
	m_tiler.Tile(guest_linear.buffer, guest_linear.offset, info.data.size, destination,
	             destination_offset, info.data.size, tiles);
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, ImageDownload transfer) {
	DownloadImageTo(image, destination.Handle(), destination_offset,
	                destination.Size() - destination_offset, destination_size, std::move(transfer));
}

void TextureCache::DownloadImageTo(Image& image, vk::Buffer destination,
                                   uint64_t destination_offset, uint64_t destination_capacity,
                                   uint64_t destination_size, ImageDownload transfer) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	if (!transfer.valid) {
		EXIT("TextureCache: invalid image download transfer\n");
	}
	if (transfer.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset, destination_capacity);
		return;
	}

	auto&      texture   = transfer.texture;
	const auto transform = texture.color_transform;
	if (texture.tiles.empty()) {
		if (transform != ColorTransform::None) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.TransformColor(linear, {destination, destination_offset, destination_size},
			                       transform, false);
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination, destination_offset, destination_size);
		return;
	}

	// KYTY_TILER_IMAGE_DIRECT: tile straight from the image, no image->buffer copy.
	if (transform == ColorTransform::None &&
	    m_tiler.TileFromImage(image, texture.regions, destination, destination_offset,
	                          destination_size, texture.LinearSize(), texture.tiles)) {
		return;
	}
	m_tiler.TileImage(image, texture.regions, destination, destination_offset, destination_size,
	                  texture.LinearSize(), texture.tiles, transform);
}

bool TextureCache::AnyImageStartsAt(uint64_t address) {
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (image != nullptr && image->info.data.address == address) {
			return true;
		}
	}
	return false;
}

bool TextureCache::MayOverlapImages(uint64_t address, uint64_t size) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return true;
	}
	for (auto page = pages.first; page < pages.last_exclusive; ++page) {
		if (m_image_page_counts[page].load() != 0) {
			return true;
		}
	}
	return false;
}

bool TextureCache::OtherImagesOwnBytes(GuestRange range, ImageId except) {
	if (range.Empty()) {
		return false;
	}
	for (const auto id: FindImagesInRegion(range.address, range.size, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (id != except && image != nullptr && image->registered && !image->depth_id &&
		    image->OwnsBytesIn(range.address, range.size)) {
			return true;
		}
	}
	return false;
}

bool TextureCache::BlocksUnderRect(const Image& image, vk::Rect2D& rect, RangeSet& blocks) const {
	const auto& info = image.info;
	TileBlockLayout block {};
	if (info.tile_mode != Prospero::TileMode::kRenderTarget || info.IsDepth() || info.IsVolume() ||
	    info.resources.levels != 1 || info.resources.layers != 1 || info.samples != 1 ||
	    image.backing.samples != 1 || info.HasStencil() ||
	    !TileGetBlockLayout(TileBlockFamily::RenderTarget64KB, info.bytes_per_block, block) ||
	    block.block_depth != 1 || block.block_width == 0 || block.block_height == 0 ||
	    info.pitch % block.block_width != 0 || info.data.address % block.block_size != 0) {
		return false;
	}
	const uint64_t columns = info.pitch / block.block_width;
	const uint64_t rows    = (info.extent.height + block.block_height - 1) / block.block_height;
	if (columns == 0 || info.data.size != columns * rows * block.block_size) {
		return false;
	}
	rect = ClampRect(rect, info.extent.width, info.extent.height);
	if (rect.extent.width == 0 || rect.extent.height == 0) {
		// An empty scissor writes nothing.
		return true;
	}
	const uint64_t first_column = rect.offset.x / block.block_width;
	const uint64_t last_column  = (rect.offset.x + rect.extent.width - 1) / block.block_width;
	const uint64_t first_row    = rect.offset.y / block.block_height;
	const uint64_t last_row     = (rect.offset.y + rect.extent.height - 1) / block.block_height;
	if (first_column == 0 && last_column + 1 >= columns && first_row == 0 && last_row + 1 >= rows) {
		// Every block: the whole image.
		return false;
	}
	for (uint64_t row = first_row; row <= last_row; ++row) {
		const uint64_t begin = (row * columns + first_column) * block.block_size;
		const uint64_t end   = (row * columns + last_column + 1) * block.block_size;
		blocks.Add(info.data.address + begin, end - begin);
	}
	return true;
}

void TextureCache::TakeOverOwnedBytes(Image& destination, Image& source) {
	ImageWatch("image take over owned bytes", destination);
	if (!AliasBytesEnabled() || &destination == &source || destination.depth_id ||
	    !source.IsGpuModified()) {
		return;
	}
	std::vector<GuestRange> taken;
	source.ForEachOwnedRange(destination.info.data.address, destination.info.data.size,
	                         [&](uint64_t begin, uint64_t end) {
		                         taken.push_back({begin, end - begin});
	                         });
	for (const auto& range: taken) {
		destination.OwnBytes(range.address, range.size);
		(void)source.DisownBytes(range.address, range.size);
	}
	if (!source.IsGpuModified()) {
		InvalidateCleanImageProofs(source.live.address, source.live.size,
		                           Coherence::Source::ImageGpuClear);
	}
	if (destination.IsGpuModified() && destination.OwnsNoBytes()) {
		destination.ClearGpuModified();
		InvalidateCleanImageProofs(destination.live.address, destination.live.size,
		                           Coherence::Source::ImageGpuClear);
	}
}

uint64_t TextureCache::MaterializeOwnedBytes(GuestRange range, ImageId only, ImageId except,
                                             const char* reason, Buffer* target) {
	// Scratch buffers from the tiler stay alive until this operation's last tick (tiler.h).
	const TileManager::ScratchHold scratch_hold(m_tiler);
	if (!AliasBytesEnabled() || range.Empty() || !m_scheduler.Active() ||
	    m_scheduler.Current().IsInvalid()) {
		return 0;
	}
	struct Owner {
		ImageId  id;
		RangeSet ranges;
	};
	std::vector<Owner> owners;
	const auto collect = [&](ImageId id) {
		const auto* image = m_slot_images.try_get(id);
		if (id == except || image == nullptr || !image->registered || image->depth_id ||
		    !image->OwnsBytesIn(range.address, range.size)) {
			return;
		}
		// CPU invalidation releases write tracking for the image (or its dirty chunks),
		// but its old GPU-owned range can remain until refresh/retirement. Those native
		// contents are no longer a source of guest bytes. In particular, retiring an old
		// render target must not tile its pixels over CPU tables that reused the memory.
		// Match SafeToSyncIntoBuffer's CPU-dirty rejection. InitializeImage also excludes
		// definitely CPU-dirty images from its own-byte preservation before a rebuild.
		if (image->IsCpuDirty()) {
			static const bool diagnose = [] {
				const auto* value = std::getenv("KYTY_ALIAS_CPU_WRITE_DIAGNOSTICS");
				return value != nullptr && std::strcmp(value, "1") == 0;
			}();
			static std::atomic<uint32_t> logged {0};
			if (diagnose && logged.fetch_add(1, std::memory_order_relaxed) < 8) {
				std::fprintf(stderr,
				             "AliasCpuWrite: rejected stale image addr=0x%016" PRIx64
				             " size=0x%" PRIx64 " dirty_begin=0x%016" PRIx64
				             " dirty_bytes=0x%" PRIx64 " reason=%s\n",
				             image->info.data.address, image->info.data.size,
				             image->DirtySpanBegin(), image->DirtySpanBytes(), reason);
			}
			return;
		}
		Owner owner {id, {}};
		image->ForEachOwnedRange(range.address, range.size, [&](uint64_t begin, uint64_t end) {
			owner.ranges.Add(begin, end - begin);
		});
		owners.push_back(std::move(owner));
	};
	if (only) {
		collect(only);
		if (!owners.empty()) {
			// Bytes another image owns too (the destination of a copy from this one) stay there.
			for (const auto id: FindImagesInRegion(range.address, range.size, false)) {
				const auto* other = m_slot_images.try_get(id);
				if (id == only || other == nullptr || !other->registered || other->depth_id ||
				    other->IsCpuDirty()) {
					continue;
				}
				other->ForEachOwnedRange(range.address, range.size,
				                         [&](uint64_t begin, uint64_t end) {
					                         owners.front().ranges.Subtract(begin, end - begin);
				                         });
			}
			if (owners.front().ranges.Empty()) {
				owners.clear();
			}
		}
	} else {
		for (const auto id: FindImagesInRegion(range.address, range.size, false)) {
			collect(id);
		}
	}
	uint64_t moved = 0;
	for (auto& owner: owners) {
		auto& image    = m_slot_images[owner.id];
		auto  transfer = BuildDownload(image);
		if (!transfer.valid || !image.FullyResident() || image.backing.image == nullptr) {
			// No download layout (e.g. multisampled, or a colour view of depth-tiled memory):
			// the bytes stay with the image (a free drops them, as before).
			Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesUnmaterialized);
			continue;
		}
		const auto data = image.info.data;
		uint64_t   bytes = 0;
		owner.ranges.ForEach([&](uint64_t begin, uint64_t end) { bytes += end - begin; });
		// The cache buffer takes the ranges as a writable binding would: their CPU-dirty pages are
		// uploaded first, then the tracker pages become GPU-owned.
		auto& buffer = m_buffer_cache.PrepareImageBytes(owner.ranges, target);
		// Scratch holding the tiled bytes at their offsets in `data`: the owned ranges first get
		// the buffer's bytes (so bytes no texel covers, such as padding, keep their values), the
		// image is tiled over them, and the owned ranges are copied back.
		const auto scratch = m_tiler.GetScratchBuffer(data.size);
		std::vector<vk::BufferCopy> to_scratch;
		std::vector<vk::BufferCopy> to_buffer;
		owner.ranges.ForEach([&](uint64_t begin, uint64_t end) {
			to_scratch.push_back({buffer.Offset(begin), scratch.offset + (begin - data.address),
			                      end - begin});
			to_buffer.push_back({scratch.offset + (begin - data.address), buffer.Offset(begin),
			                     end - begin});
		});
		m_scheduler.EndRendering();
		{
			const auto              command = m_scheduler.Current().HelperSink();
			vk::BufferMemoryBarrier before[2] {};
			before[0].srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
			before[0].dstAccessMask       = vk::AccessFlagBits::eTransferRead;
			before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			before[0].buffer              = buffer.Handle();
			before[0].offset              = 0;
			before[0].size                = VK_WHOLE_SIZE;
			before[1]                     = before[0];
			before[1].srcAccessMask =
			    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			before[1].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			before[1].buffer        = scratch.buffer;
			before[1].offset        = scratch.offset;
			before[1].size          = scratch.size;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 2, before,
			                        0, nullptr);
			command.copyBuffer(buffer.Handle(), scratch.buffer,
			                   static_cast<uint32_t>(to_scratch.size()), to_scratch.data());
		}
		DownloadImageTo(image, scratch.buffer, scratch.offset, scratch.size, data.size,
		                std::move(transfer));
		{
			// Again: the tile pass may have begun a new command buffer.
			m_scheduler.EndRendering();
			const auto              command = m_scheduler.Current().HelperSink();
			vk::BufferMemoryBarrier between[2] {};
			between[0].srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
			between[0].dstAccessMask       = vk::AccessFlagBits::eTransferRead;
			between[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			between[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			between[0].buffer              = scratch.buffer;
			between[0].offset              = scratch.offset;
			between[0].size                = scratch.size;
			between[1]                     = between[0];
			between[1].srcAccessMask =
			    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			between[1].dstAccessMask = vk::AccessFlagBits::eTransferWrite;
			between[1].buffer        = buffer.Handle();
			between[1].offset        = 0;
			between[1].size          = VK_WHOLE_SIZE;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
			                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 2, between,
			                        0, nullptr);
			command.copyBuffer(scratch.buffer, buffer.Handle(),
			                   static_cast<uint32_t>(to_buffer.size()), to_buffer.data());
			vk::BufferMemoryBarrier after {};
			after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
			after.dstAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			after.buffer              = buffer.Handle();
			after.offset              = 0;
			after.size                = VK_WHOLE_SIZE;
			command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
			                        vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 1, &after,
			                        0, nullptr);
		}
		m_buffer_cache.CommitImageBytes(buffer, owner.ranges);
		// The buffer owns these bytes now; the image's native contents still equal them.
		owner.ranges.ForEach([&](uint64_t begin, uint64_t end) {
			(void)image.DisownBytes(begin, end - begin);
		});
		if (!image.IsGpuModified()) {
			InvalidateCleanImageProofs(image.live.address, image.live.size,
			                           Coherence::Source::ImageGpuClear);
		}
		moved += bytes;
		Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesMaterializations);
		Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesMaterializedBytes, bytes);
		HangTrace::RecordTransfer(HangTrace::TransferKind::ImageCopy, "alias-bytes", reason,
		                          data.address, static_cast<uint32_t>(image.backing.format),
		                          image.info.extent.width, image.info.extent.height, bytes, 0);
	}
	return moved;
}

namespace {

// KYTY_TEXEL_IMAGE_SYNC_STATS=1 (diagnostic, default off): BufferCache::SynchronizeBufferFromImage
// accounting, one console line every 10 s: calls (GPU thread / other), time, the image lookup's
// share, images found, downloads and their bytes, unchanged skips, alias-byte scans and the
// bytes they moved.
struct TexelImageSyncStats {
	std::atomic<uint64_t> calls {0}, other_thread {0}, ns {0}, lookup_ns {0}, found {0},
	    downloads {0}, download_bytes {0}, skips {0}, alias_scans {0}, alias_ns {0},
	    alias_bytes {0}, owner_ns {0};
	std::atomic<int64_t> last_ms {0};
};
TexelImageSyncStats g_texel_image_sync;

bool TexelImageSyncStatsEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_TEXEL_IMAGE_SYNC_STATS");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

uint64_t TexelSyncNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

void ReportTexelImageSync() {
	auto&      s   = g_texel_image_sync;
	const auto now = static_cast<int64_t>(TexelSyncNowNs() / 1000000u);
	auto       last = s.last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		s.last_ms.store(now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10000 || !s.last_ms.compare_exchange_strong(last, now)) {
		return;
	}
	std::printf("TexelImageSync %.0fs: %" PRIu64 " calls (%" PRIu64 " off the GPU thread), %.1f ms "
	            "(lookup %.1f, owner sync %.1f, alias %.1f); %" PRIu64 " images found, %" PRIu64
	            " downloads %.1f MiB, %" PRIu64 " unchanged skips; %" PRIu64 " alias scans moved %.1f MiB\n",
	            static_cast<double>(now - last) / 1000.0, s.calls.exchange(0), s.other_thread.exchange(0),
	            static_cast<double>(s.ns.exchange(0)) / 1e6, static_cast<double>(s.lookup_ns.exchange(0)) / 1e6,
	            static_cast<double>(s.owner_ns.exchange(0)) / 1e6,
	            static_cast<double>(s.alias_ns.exchange(0)) / 1e6, s.found.exchange(0),
	            s.downloads.exchange(0), static_cast<double>(s.download_bytes.exchange(0)) / 1048576.0,
	            s.skips.exchange(0), s.alias_scans.exchange(0),
	            static_cast<double>(s.alias_bytes.exchange(0)) / 1048576.0);
	std::fflush(stdout);
}

} // namespace

namespace {

// KYTY_TEXEL_SYNC_MEMO (live, default off in this line, on in the Insomniac title presets; =0 off; =2 verify): a texel read whose range had no
// image starting at its address and no image page overlapping it (so it synchronized nothing:
// no image download, no alias-byte scan) is remembered with the image registration epoch
// (TextureCache::RegistrationEpoch, bumped on every image register/unregister and GPU-modified
// transition). The same range, buffer and epoch again returns at once: nothing it looked at can
// have changed. Miles Morales binds about 190,000 heap-wide (~1 GB) texel buffers a second on the
// command processor, every one of them finding nothing after a walk of ~1,000 image page counts.
// =2 runs the full synchronization on a memo hit too and reports a hit that would have synced.
Live::Switch g_texel_sync_memo("KYTY_TEXEL_SYNC_MEMO", [](const char* value) -> int64_t {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0) {
		return 2;
	}
	return std::clamp<int64_t>(std::strtoll(value, nullptr, 10), 0, 2);
});

struct TexelSyncMemoEntry {
	uint64_t      vaddr  = 0;
	uint64_t      size   = 0;
	uint64_t      epoch  = 0;
	const Buffer* buffer = nullptr;
};
constexpr size_t TexelSyncMemoSlots = 4096;
std::atomic<uint64_t> g_texel_sync_memo_hits {0}, g_texel_sync_memo_stores {0},
    g_texel_sync_memo_mismatches {0};
std::atomic<int64_t> g_texel_sync_memo_last_ms {0};

void ReportTexelSyncMemo() {
	const auto now  = static_cast<int64_t>(TexelSyncNowNs() / 1000000u);
	auto       last = g_texel_sync_memo_last_ms.load(std::memory_order_relaxed);
	if (last == 0) {
		g_texel_sync_memo_last_ms.store(now, std::memory_order_relaxed);
		return;
	}
	if (now - last < 10000 || !g_texel_sync_memo_last_ms.compare_exchange_strong(last, now)) {
		return;
	}
	std::printf("TexelSyncMemo %.0fs: %" PRIu64 " hits, %" PRIu64 " stores, %" PRIu64
	            " verify mismatches\n",
	            static_cast<double>(now - last) / 1000.0, g_texel_sync_memo_hits.exchange(0),
	            g_texel_sync_memo_stores.exchange(0), g_texel_sync_memo_mismatches.exchange(0));
	std::fflush(stdout);
}

} // namespace

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto memo_mode = g_texel_sync_memo.Get();
	if (memo_mode != 0 && GuestGpu::IsGpuThread()) {
		const auto epoch = m_texture_cache.RegistrationEpoch();
		if (epoch != 0) {
			thread_local std::vector<TexelSyncMemoEntry> memo(TexelSyncMemoSlots);
			auto& entry = memo[((vaddr >> 6u) ^ (vaddr >> 20u) ^ size) % TexelSyncMemoSlots];
			const bool hit = entry.vaddr == vaddr && entry.size == size && entry.epoch == epoch &&
			                 entry.buffer == &buffer;
			if (hit && memo_mode == 1) {
				g_texel_sync_memo_hits.fetch_add(1, std::memory_order_relaxed);
				ReportTexelSyncMemo();
				return false;
			}
			bool       nothing = false;
			const bool synced  = SynchronizeBufferFromImageCounted(buffer, vaddr, size, &nothing);
			if (hit) {
				g_texel_sync_memo_hits.fetch_add(1, std::memory_order_relaxed);
				if (synced || !nothing) {
					const auto count = g_texel_sync_memo_mismatches.fetch_add(1, std::memory_order_relaxed);
					if (count < 16) {
						std::printf("TexelSyncMemo verify: 0x%" PRIx64 "+0x%" PRIx64
						            " would have been skipped but %s\n",
						            vaddr, size, synced ? "synchronized" : "was not provably empty");
						std::fflush(stdout);
					}
				}
			}
			// The epoch read before the synchronization: a change during it makes the entry stale.
			if (nothing && !synced && m_texture_cache.RegistrationEpoch() == epoch &&
			    !m_texture_cache.AnyImageStartsAt(vaddr)) {
				entry = {vaddr, size, epoch, &buffer};
				g_texel_sync_memo_stores.fetch_add(1, std::memory_order_relaxed);
			} else if (hit) {
				entry = {};
			}
			ReportTexelSyncMemo();
			return synced;
		}
	}
	return SynchronizeBufferFromImageCounted(buffer, vaddr, size, nullptr);
}

bool BufferCache::RawReadImageSyncEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_RAW_READ_IMAGE_SYNC");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

bool BufferCache::SyncRawReadFromWrittenImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	// A stream-buffer read holds CPU-dirty guest bytes of a range without GPU-written buffer bytes
	// (as for texel reads, no image download into it). Lock-free: no image registered on the
	// range's 1 MiB pages (most raw reads).
	if (&buffer == &m_stream_buffer || !m_texture_cache.MayOverlapImages(vaddr, size)) {
		return false;
	}
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size, true, true);
	if (!selected) {
		return false;
	}
	uint32_t width = 0, height = 0, format = 0;
	uint64_t serial = 0;
	static const bool once = [] {
		const auto* value = std::getenv("KYTY_RAW_READ_IMAGE_SYNC_ONCE");
		return value != nullptr && value[0] == '1';
	}();
	static std::atomic<uint64_t> skipped_once {0};
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		const auto*      image = m_texture_cache.m_slot_images.try_get(selected);
		// Only GPU-written contents differ from the buffer's bytes (a CPU-written image was
		// uploaded from them).
		if (image == nullptr || !image->IsGpuModified()) {
			return false;
		}
		serial = image->ContentSerial();
		// KYTY_RAW_READ_IMAGE_SYNC_ONCE: this buffer already holds these contents of the range.
		if (once && serial != 0 && image->raw_sync_serial == serial && image->raw_sync_buffer == &buffer &&
		    size <= image->raw_sync_size) {
			skipped_once.fetch_add(1, std::memory_order_relaxed);
			return false;
		}
		width  = image->info.extent.width;
		height = image->info.extent.height;
		format = static_cast<uint32_t>(image->backing.format);
	}
	const bool synced = SynchronizeBufferFromImage(buffer, vaddr, size);
	if (once && synced) {
		std::scoped_lock lock {m_texture_cache.m_lock};
		if (auto* image = m_texture_cache.m_slot_images.try_get(selected);
		    image != nullptr && image->ContentSerial() == serial) {
			image->raw_sync_serial = serial;
			image->raw_sync_size   = size;
			image->raw_sync_buffer = &buffer;
		}
	}
	static std::atomic<uint64_t> syncs {0};
	if (synced) {
		const auto n = syncs.fetch_add(1, std::memory_order_relaxed) + 1;
		// The first 16, then at powers of two: how often raw reads take image contents.
		if (n <= 16 || (n & (n - 1)) == 0) {
			std::printf("KYTY_RAW_READ_IMAGE_SYNC: #%" PRIu64 " raw read 0x%016" PRIx64 "+0x%" PRIx64
			            " took a GPU-written image's contents (%ux%u, vk format %u; %" PRIu64
			            " repeated reads skipped)\n",
			            n, vaddr, size, width, height, format, skipped_once.load(std::memory_order_relaxed));
			std::fflush(stdout);
		}
	}
	return synced;
}

void BufferCache::PrepareReadRange(uint64_t vaddr, uint64_t size) {
	static constexpr auto obtains_init = [] {
		const auto* value = std::getenv("KYTY_RAW_READ_IMAGE_SYNC_OBTAINS");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(obtains_init())>> obtains_once;
	const bool& obtains = obtains_once.Get(obtains_init);
	if (!obtains || size == 0 || !RawReadImageSyncEnabled() || !m_texture_cache.MayOverlapImages(vaddr, size)) {
		return;
	}
	if (auto [buffer, offset] = ObtainBuffer(vaddr, size, false); buffer != nullptr) {
		(void)SyncRawReadFromWrittenImage(*buffer, vaddr, size);
	}
}

bool BufferCache::SynchronizeBufferFromImageCounted(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                                    bool* nothing) {
	if (TexelImageSyncStatsEnabled()) [[unlikely]] {
		auto&      s     = g_texel_image_sync;
		const auto begin = TexelSyncNowNs();
		s.calls.fetch_add(1, std::memory_order_relaxed);
		if (!GuestGpu::IsGpuThread()) {
			s.other_thread.fetch_add(1, std::memory_order_relaxed);
		}
		const bool result = SynchronizeBufferFromImageImpl(buffer, vaddr, size, nothing);
		s.ns.fetch_add(TexelSyncNowNs() - begin, std::memory_order_relaxed);
		ReportTexelImageSync();
		return result;
	}
	return SynchronizeBufferFromImageImpl(buffer, vaddr, size, nothing);
}

bool BufferCache::SynchronizeBufferFromImageImpl(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                                 bool* nothing) {
	if (nothing != nullptr) {
		*nothing = false;
	}
	const bool stats        = TexelImageSyncStatsEnabled();
	const auto lookup_begin = stats ? TexelSyncNowNs() : 0;
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size, true, true);
	if (stats) {
		g_texel_image_sync.lookup_ns.fetch_add(TexelSyncNowNs() - lookup_begin, std::memory_order_relaxed);
		g_texel_image_sync.found.fetch_add(selected ? 1 : 0, std::memory_order_relaxed);
	}
	const auto owner_begin = stats && selected ? TexelSyncNowNs() : 0;
	const bool synced   = selected && SynchronizeBufferFromOwner(buffer, selected);
	if (owner_begin != 0) {
		g_texel_image_sync.owner_ns.fetch_add(TexelSyncNowNs() - owner_begin, std::memory_order_relaxed);
	}
	// KYTY_ALIAS_BYTES: every other byte of the range that an image owns (an owner that does not
	// start at vaddr, owns only part of its bytes, or lies past the image synchronized above) is
	// read from the buffer too: move it there. Only into the cache buffer that holds the range
	// (whose GPU-dirty bytes the cache tracks).
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (!TextureCache::AliasBytesEnabled()) {
		if (nothing != nullptr) {
			*nothing = !selected;
		}
		return synced;
	}
	if (owner == nullptr || !*owner || &m_slot_buffers[*owner] != &buffer) {
		return synced;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesTexelReads);
	if (!m_texture_cache.MayOverlapImages(vaddr, size)) {
		if (nothing != nullptr) {
			*nothing = !selected;
		}
		return synced;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesTexelScans);
	const auto alias_begin = stats ? TexelSyncNowNs() : 0;
	std::scoped_lock lock {m_texture_cache.m_lock};
	const auto       moved = m_texture_cache.MaterializeOwnedBytes(
        {vaddr, size}, {}, synced ? selected : ImageId {}, "texel-read", &buffer);
	if (stats) {
		g_texel_image_sync.alias_scans.fetch_add(1, std::memory_order_relaxed);
		g_texel_image_sync.alias_bytes.fetch_add(moved, std::memory_order_relaxed);
		g_texel_image_sync.alias_ns.fetch_add(TexelSyncNowNs() - alias_begin, std::memory_order_relaxed);
	}
	return synced || moved != 0;
}

bool BufferCache::SynchronizeBufferFromOwner(Buffer& buffer, Common::SlotId selected) {
	uint64_t image_address = 0;
	uint64_t copied        = 0;
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		auto&            image = m_texture_cache.m_slot_images[selected];
		// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
		if (!m_texture_cache.SafeToSyncIntoBuffer(image)) {
			return false;
		}
		if (HasGpuDirtyBytes(image.info.data.address, image.info.data.size)) {
			// The image supersedes stale GPU-dirty bytes in its range (SupersedesGpuDirtyBytes).
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexelImageSyncOverGpuDirty);
		}
		const bool track_texel_sync = m_texture_cache.m_texel_sync_skip && GuestGpu::IsGpuThread();
		bool       skipped          = false;
		copied = RecordImageDownload(buffer, selected, track_texel_sync, &skipped);
		if (copied == 0) {
			return false;
		}
		if (skipped) {
			// Nothing was written. (A skip needs a content revision for the whole range, which no
			// CPU-dirty page, so no hot page, has: there is nothing to settle either.)
			Profiler::CountFrameEvent(Profiler::FrameEvent::TexelImageSyncSkips);
			if (TexelImageSyncStatsEnabled()) {
				g_texel_image_sync.skips.fetch_add(1, std::memory_order_relaxed);
			}
			return true;
		}
		image_address = image.info.data.address;
		Profiler::CountFrameEvent(Profiler::FrameEvent::TexelImageSyncDownloads);
		if (TexelImageSyncStatsEnabled()) {
			g_texel_image_sync.downloads.fetch_add(1, std::memory_order_relaxed);
			g_texel_image_sync.download_bytes.fetch_add(copied, std::memory_order_relaxed);
		}
		if (track_texel_sync) {
			if (const auto revision = GetContentRevision(image.info.data.address, copied);
			    revision && &m_slot_buffers[revision->id] == &buffer) {
				image.texel_sync = {revision->id, revision->write_revision, revision->global_epoch,
				                    image.ContentSerial(), copied, true};
			}
		}
	}
	// The image bytes now in the buffer are not tracked as GPU-owned: hot pages there go back to
	// the ordinary fault tracking (MemoryTracker hot pages, SettleHotPages). A page this re-dirties
	// also withdraws the content revision recorded above (CPU-dirty ranges have none).
	SettleHotPages(image_address, copied);
	MemoryStats::Count(MemoryStats::Counter::BufferFromImageSyncs);
	return true;
}

uint64_t BufferCache::RecordImageDownload(Buffer& buffer, ImageId id, bool skip_unchanged,
                                          bool* skipped) {
	auto& image = m_texture_cache.m_slot_images[id];
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return 0;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else if (image.info.IsVolume() || image.info.resources.layers > 1) {
		// Array and volume slices are strided across the full mip chain, so a mip's linear span
		// cannot prove that every retained slice fits: whole-image synchronization only.
		return 0;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return 0;
	}
	// The buffer already holds this image's bytes when neither changed since the last download:
	// the image's content serial is unchanged and the buffer's write revision (which every GPU
	// or CPU-upload write to it advances, and which CPU-dirty pages invalidate) is the one that
	// download produced.
	if (skip_unchanged && image.texel_sync.valid && image.ContentSerial() != 0 &&
	    image.texel_sync.serial == image.ContentSerial() && image.texel_sync.size == copy_size) {
		const auto revision = GetContentRevision(image.info.data.address, copy_size);
		if (revision && revision->id == image.texel_sync.buffer &&
		    revision->write_revision == image.texel_sync.revision &&
		    revision->global_epoch == image.texel_sync.epoch &&
		    &m_slot_buffers[revision->id] == &buffer) {
			if (skipped != nullptr) {
				*skipped = true;
			}
			return copy_size;
		}
	}
	image.texel_sync.valid = false;
	auto transfer = m_texture_cache.BuildDownload(image);
	if (!transfer.valid) {
		return 0;
	}
	if (transfer.depth_target && copy_size != image.info.data.size) {
		return 0;
	}
	if (!transfer.depth_target && levels < image.info.resources.levels) {
		auto& texture = transfer.texture;
		std::erase_if(texture.regions, [levels](const vk::BufferImageCopy& region) {
			return region.imageSubresource.mipLevel >= levels;
		});
		if (texture.regions.empty()) {
			return 0;
		}
		if (!texture.tiles.empty()) {
			texture.tiles.clear();
			if (!TextureBuildGpuTileInfos(copy_size, texture.regions, texture.layout, levels,
			                              texture.tiles)) {
				return 0;
			}
		}
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, std::move(transfer));
	buffer.MarkContentWritten();
	NoteBufferContentWrite(image.info.data.address, copy_size);
	return copy_size;
}

// KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE=1 enables writebacks: a GPU write binding over a GPU-modified
// image first moves the image's contents into the buffer (see PreserveImagesForGpuWrite). Default
// off, the previous behaviour (the contents are dropped): each writeback leaves the whole moved
// range GPU-dirty until a readback, and images created there later start from those bytes; the
// Sky Garden water sampled such stale data (a never-written quarter-res input) and rendered white.
bool BufferCache::ImageWritebackOnGpuWriteEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE");
		return value != nullptr && std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

void BufferCache::PreserveImagesForGpuWrite(BufferId id, uint64_t vaddr, uint64_t size) {
	if (!ImageWritebackOnGpuWriteEnabled()) {
		return;
	}
	// KYTY_GPU_WRITE_IMAGE_SKIP: no registered image whose contents could move into the buffer.
	if (m_texture_cache.SkipGpuWriteImageWalk(vaddr, size)) {
		return;
	}
	struct Candidate {
		ImageId    id;
		GuestRange range;
	};
	std::vector<Candidate> candidates;
	auto&                  buffer       = m_slot_buffers[id];
	const auto             buffer_begin = buffer.CpuAddress();
	const auto             buffer_end   = buffer_begin + buffer.Size();
	// Only an image whose native contents supersede every byte of its range (GPU-modified, no CPU
	// write since, no newer buffer write) can be moved into the buffer as a whole. It must be
	// fully resident: non-resident levels hold undefined native contents (GPU-written images are
	// fully resident anyway, TextureCache::MarkImageGpuModified). Depth-associated color aliases
	// publish through their depth image. The image must also have a supported download layout
	// (e.g. not a colour view of depth-tiled memory): otherwise keep the previous behaviour for it
	// rather than taking its whole range into GPU ownership without its contents.
	const auto movable_image = [this](const Image& image) {
		return !image.depth_id && image.SafeToDownload() && image.FullyResident() &&
		       m_texture_cache.BuildDownload(image).valid;
	};
	{
		std::scoped_lock lock {m_texture_cache.m_lock};
		// Exactly the images TextureCache::InvalidateMemoryFromGPU takes GPU ownership from.
		for (const auto image_id: m_texture_cache.FindImagesInRegion(vaddr, size, true)) {
			const auto& image = m_texture_cache.m_slot_images[image_id];
			if (!image.Overlaps(vaddr, size) || !image.IsGpuModified()) {
				continue;
			}
			const auto image_range = image.info.data;
			bool       movable     = movable_image(image) && image_range.address >= buffer_begin &&
			                image_range.address < buffer_end;
			const GuestRange range {image_range.address,
			                        std::min(image_range.End(), buffer_end) - image_range.address};
			if (movable) {
				// The buffer will own the whole moved range. Another GPU-modified image there
				// that this write does not already take ownership from would lose its contents
				// to this one's: keep the previous behaviour in that (alias) case.
				for (const auto other_id:
				     m_texture_cache.FindImagesInRegion(range.address, range.size, true)) {
					const auto& other = m_texture_cache.m_slot_images[other_id];
					if (other_id != image_id && other.Overlaps(range.address, range.size) &&
					    other.IsGpuModified() && !other.Overlaps(vaddr, size)) {
						movable = false;
						break;
					}
				}
			}
			if (!movable) {
				MemoryStats::Count(MemoryStats::Counter::ImageWritebackSkips);
				continue;
			}
			candidates.push_back({image_id, range});
		}
	}
	for (const auto& candidate: candidates) {
		const auto& range = candidate.range;
		// Own the moved range exactly like a writable binding: upload its CPU-dirty pages (bytes
		// the image download may not cover keep their guest values), take the tracker pages
		// under their locks, then overwrite the image's bytes with its native contents.
		(void)SynchronizeBuffer(buffer, range.address, range.size, true, false, nullptr,
		                        "image-writeback");
		uint64_t copied     = 0;
		uint64_t image_size = 0;
		{
			std::scoped_lock lock {m_texture_cache.m_lock};
			const auto*      image = m_texture_cache.m_slot_images.try_get(candidate.id);
			// A guest thread may have dirtied the image since the scan. Its fault then waits for
			// this GPU-thread command and reads back the buffer's (guest-valued) bytes.
			if (image != nullptr && movable_image(*image) &&
			    image->info.data.address == range.address) {
				image_size = image->info.data.size;
				copied     = RecordImageDownload(buffer, candidate.id);
			}
		}
		if (copied != 0) {
			MemoryStats::Count(MemoryStats::Counter::ImageWritebacks);
			MemoryStats::Count(MemoryStats::Counter::ImageWritebackBytes, copied);
			if (copied < image_size) {
				MemoryStats::Count(MemoryStats::Counter::ImageWritebackPartial);
			}
		} else {
			MemoryStats::Count(MemoryStats::Counter::ImageWritebackSkips);
		}
		// The tracker pages are GPU-owned now, so their bytes must be too (downloaded or not).
		buffer.MarkContentWritten();
		if (!m_gpu_modified_ranges.Contains(range.address, range.size)) {
			CleanVerdict::Invalidate(range.address, range.size, Coherence::Source::BufferDirtyAdd);
		}
		m_gpu_modified_ranges.Add(range.address, range.size);
		NoteBufferContentWrite(range.address, range.size);
		ForgetKnownFills(range.address, range.size);
		HangTrace::NoteGpuWrite(range.address, range.size);
		// As for any GPU buffer write: overlapping images are now rebuilt from the buffer.
		m_texture_cache.InvalidateMemoryFromGPU(range.address, range.size);
	}
}

bool TextureCache::DownloadImageMemory(ImageId id, std::shared_ptr<std::atomic_bool> done) {
	KYTY_GPU_OP_SITE("texcache.download");
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || !SafeToDownload(image)) {
		return false;
	}
	const auto range    = image.info.data;
	auto&      download = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	auto [mapped, offset] =
	    download.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
	// Map refuses a reservation larger than the download buffer instead of waiting for space, so
	// a whole-image readback can exceed it. Such a readback is staged in a private buffer of
	// exactly the needed size instead of exiting; the deferred write-back retires it once the copy
	// has been read, as the shared-buffer path releases its reservation.
	std::unique_ptr<Buffer> oversized;
	if (mapped == nullptr) {
		static std::atomic_bool logged {false};
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("TextureCache: image readback of %" PRIu64
			     " bytes exceeds the download buffer; staging it in a private buffer\n",
			     range.size);
		}
		oversized = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		mapped    = oversized->Mapped().data();
		offset    = 0;
	} else {
		download.Commit();
	}
	auto& destination = oversized != nullptr ? *oversized : static_cast<Buffer&>(download);
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	destination.Flush(offset, range.size);

	DownloadImage(image, destination, offset, range.size, std::move(transfer));
	Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureDownloadBytes, range.size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = destination.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	const auto publication = m_buffer_cache.BeginBackingPublication(
	    std::span<const GuestRange>(&range, 1), m_scheduler.CurrentTick());
	// KYTY_IMAGE_WRITEBACK_GUARD: the range's pages stay watched until the write-back (see
	// PendingWriteback).
	const auto pending = m_writeback_guard ? BeginPendingWriteback(range) : nullptr;
	// Storage readback (upstream 9678c59b7): guest accesses to the pages, reads too, wait until
	// the stored data lands (RenderContext::WaitWriteBacks, WriteBackTick).
	const StorageWriteBack storage_write_back {range, m_scheduler.CurrentTick()};
	const bool             trapped = m_readback_storage_images;
	if (trapped) {
		{
			std::scoped_lock write_back_lock {m_storage_write_back_mutex};
			m_storage_write_backs.push_back(storage_write_back);
		}
		m_page_manager.UpdatePageWatchers<true, true>(range.address, range.size);
	}
	const auto release_trap = [this, storage_write_back, trapped] {
		if (!trapped) {
			return;
		}
		m_page_manager.UpdatePageWatchers<false, true>(storage_write_back.range.address,
		                                               storage_write_back.range.size);
		// Matched by value, not erased from the front: our oversized-download path defers its own
		// operation (upstream f47da249b relies on queue order and drops the defaulted ==, which
		// broke the MSVC STL 14.51 build).
		std::scoped_lock write_back_lock {m_storage_write_back_mutex};
		const auto found = std::find_if(
		    m_storage_write_backs.begin(), m_storage_write_backs.end(), [&](const StorageWriteBack& entry) {
			    return entry.tick == storage_write_back.tick &&
			           entry.range.address == storage_write_back.range.address &&
			           entry.range.size == storage_write_back.range.size;
		    });
		if (found != m_storage_write_backs.end()) {
			m_storage_write_backs.erase(found);
		}
	};
	if (oversized != nullptr) {
		m_scheduler.DeferPriorityOperation(
		    [this, owner = std::move(oversized), range, mapped, publication, pending, done,
		     release_trap]() mutable {
			    owner->Invalidate(0, range.size);
			    WriteBackImageBytes(range, mapped, pending);
			    release_trap();
			    m_buffer_cache.EndBackingPublication(publication);
			    owner.reset();
			    if (done != nullptr) {
				    done->store(true, std::memory_order_release);
			    }
		    });
		return true;
	}
	m_scheduler.DeferPriorityOperation([this, &download, range, mapped, offset, publication, pending, done,
	                                    release_trap] {
		download.Invalidate(offset, range.size);
		WriteBackImageBytes(range, mapped, pending);
		release_trap();
		m_buffer_cache.EndBackingPublication(publication);
		if (done != nullptr) {
			done->store(true, std::memory_order_release);
		}
	});
	return true;
}

// KYTY_IMAGE_WRITEBACK_GUARD (default on; =0 writes every byte as before; chenxiao07/KytyPS5
// 79d57528c). An image download is recorded, and its bytes reach guest memory only when the GPU
// has finished, on the priority worker. Nothing held the CPU off the range meanwhile: a garbage
// collection frees the image (its pages are no longer watched) right after recording, and a CPU
// write to a readback image that stays registered only makes the image CPU-dirty. Writing the
// recorded bytes then put stale GPU data over what the CPU wrote since. Now the pending write-back
// watches the range's pages itself until it has written them: a CPU write meanwhile faults, the
// fault (InvalidateCpuAliases) releases the written pages and marks them CPU-written, and the
// write-back leaves those pages to the CPU (logged). A write that faults while the write-back runs
// waits for it (m_writeback_mutex) and lands after it.
std::shared_ptr<TextureCache::PendingWriteback> TextureCache::BeginPendingWriteback(GuestRange range) {
	auto       pending = std::make_shared<PendingWriteback>();
	const auto begin   = Common::AlignDown(range.address, TRACKER_PAGE_SIZE);
	const auto end     = Common::AlignUp(range.End(), TRACKER_PAGE_SIZE);
	m_page_manager.UpdatePageWatchers<true>(begin, end - begin);
	pending->watched.emplace_back(begin, end);
	std::scoped_lock lock {m_writeback_mutex};
	m_pending_writebacks.push_back(pending);
	m_pending_writeback_count.store(m_pending_writebacks.size(), std::memory_order_release);
	return pending;
}

void TextureCache::NotePendingWritebackCpuWrite(uint64_t address, uint64_t size) {
	if (m_pending_writeback_count.load(std::memory_order_acquire) == 0) {
		return;
	}
	const auto begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	std::scoped_lock lock {m_writeback_mutex};
	for (const auto& pending: m_pending_writebacks) {
		KeptRanges::List released;
		for (const auto& [watched_begin, watched_end]: pending->watched) {
			const auto from = std::max(begin, watched_begin);
			const auto to   = std::min(end, watched_end);
			if (from < to) {
				released.emplace_back(from, to);
			}
		}
		for (const auto& [from, to]: released) {
			m_page_manager.UpdatePageWatchers<false>(from, to - from);
			KeptRanges::Subtract(pending->watched, from, to);
			(void)KeptRanges::Add(pending->cpu_written, from, to);
		}
	}
}

void TextureCache::WriteBackImageBytes(GuestRange range, const uint8_t* data,
                                       const std::shared_ptr<PendingWriteback>& pending) {
	if (pending == nullptr) {
		LibKernel::Memory::WriteBacking(range.address, data, range.size);
		return;
	}
	std::scoped_lock lock {m_writeback_mutex};
	uint64_t at      = range.address;
	uint64_t skipped = 0;
	for (const auto& [written_begin, written_end]: pending->cpu_written) {
		const auto from = std::max(written_begin, range.address);
		const auto to   = std::min(written_end, range.End());
		if (from >= to) {
			continue;
		}
		if (at < from) {
			LibKernel::Memory::WriteBacking(at, data + (at - range.address), from - at);
		}
		skipped += to - from;
		at = std::max(at, to);
	}
	if (at < range.End()) {
		LibKernel::Memory::WriteBacking(at, data + (at - range.address), range.End() - at);
	}
	// Written: the pages the CPU did not take may be released now.
	for (const auto& [begin, end]: pending->watched) {
		m_page_manager.UpdatePageWatchers<false>(begin, end - begin);
	}
	pending->watched.clear();
	std::erase(m_pending_writebacks, pending);
	m_pending_writeback_count.store(m_pending_writebacks.size(), std::memory_order_release);
	if (skipped == 0) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::ImageWritebackSkippedBytes, skipped);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("TextureCache: image write-back at 0x%016" PRIx64 " size=0x%" PRIx64
		     " left 0x%" PRIx64 " bytes to CPU writes made after the download was recorded\n",
		     range.address, range.size, skipped);
	}
}

bool TextureCache::NoImagesOnPages(uint64_t address, uint64_t size) const noexcept {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return true; // FindImagesInRegion finds nothing there either
	}
	for (auto page = pages.first; page < pages.last_exclusive; ++page) {
		if (m_image_page_counts[page].load(std::memory_order_seq_cst) != 0) {
			return false;
		}
	}
	return true;
}

bool TextureCache::SkipGpuWriteImageWalk(uint64_t address, uint64_t size) {
	if (!m_gpu_write_skip || !NoImagesOnPages(address, size)) {
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuWriteImageSkips);
	m_gpu_write_skip_totals.skips.fetch_add(1, std::memory_order_relaxed);
	if (m_gpu_write_skip_verify == 0) {
		return true;
	}
	std::scoped_lock lock {m_lock};
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuWriteImageSkipVerifyChecks);
	m_gpu_write_skip_totals.verify_checks.fetch_add(1, std::memory_order_relaxed);
	if (FindImagesInRegion(address, size, true).empty()) {
		return false;
	}
	// Under m_lock a findable image's pages are counted: uncounted pages here are a broken
	// ordering, counted ones an image registered after the lock-free decision.
	if (!NoImagesOnPages(address, size)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::GpuWriteImageSkipVerifyRaces);
		m_gpu_write_skip_totals.verify_races.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::GpuWriteImageSkipVerifyMismatches);
	m_gpu_write_skip_totals.verify_mismatches.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
		std::fprintf(stderr,
		             "GpuWriteImageSkipVerify: an image over uncounted pages: addr=0x%016" PRIx64
		             " size=0x%" PRIx64 "\n",
		             address, size);
	}
	if (m_gpu_write_skip_verify == 2) {
		EXIT("GpuWriteImageSkipVerify: an image over uncounted pages: addr=0x%016" PRIx64
		     " size=0x%" PRIx64 "\n",
		     address, size);
	}
	return false;
}

void TextureCache::InvalidateMemoryFromGPUImageStarts(uint64_t address, uint64_t size) {
	m_invalidate_image_starts_only = true;
	InvalidateMemoryFromGPU(address, size);
	m_invalidate_image_starts_only = false;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	if (HangTrace::CpWatch(address, size)) {
		HangTrace::CpEvent event;
		event.event   = "gpuwrite-buffer";
		event.address = address;
		event.size    = size;
		HangTrace::RecordCp(event);
	}
	// KYTY_GPU_WRITE_IMAGE_SKIP: no registered image to take ownership from.
	if (SkipGpuWriteImageWalk(address, size)) {
		return;
	}
	std::scoped_lock lock {m_lock};
	const auto       images = FindImagesInRegion(address, size, true);
	// KYTY_IMAGE_EXACT_RANGE_INVALIDATE (after BryanKAdams/KytyPS5 18ec846): Sky Garden's compute
	// passes write several tiled render-target surfaces as raw storage buffers, each exactly one
	// image's range (e.g. 1920x1080 RGBA16F at 0x53ad00000, about three times a frame), in a heap
	// whose other surfaces (2432x1368 at 0x53aa00000, other 1080p targets) cover the same bytes at
	// other times. Every overlapped image was rebuilt from the buffer at its next use: full-size
	// detile uploads (about 1.2 GB/s at the start view) of surfaces the write did not target, whose
	// bytes outside the write had not changed. With the rule, when the range equals an image's,
	// only the images inside the range are rebuilt. An image it partly overlaps still loses GPU
	// ownership of the written bytes (the buffer holds them now; buffer reads see the write), but
	// keeps its native contents: on the console those bytes now hold another surface's layout,
	// which it does not read before it is rendered again. Needs KYTY_ALIAS_BYTES (per-byte
	// ownership); depth-associated images keep the old behaviour.
	bool exact = false;
	if (m_exact_range_mode != 0 && AliasBytesEnabled()) {
		for (const auto id: images) {
			const auto& data = m_slot_images[id].info.data;
			if (data.address == address && data.size == size) {
				exact = true;
				m_exact_range_totals.exact_writes++;
				break;
			}
		}
	}
	for (const auto id: images) {
		auto& image = m_slot_images[id];
		if (!image.Overlaps(address, size)) {
			continue;
		}
		if (m_invalidate_image_starts_only &&
		    (image.info.data.address < address || image.info.data.address >= address + size)) {
			continue;
		}
		// KYTY_IMAGE_PARTIAL_GPU_WRITE: decided before the write takes the bytes.
		const bool first_write  = !image.IsBufferModified();
		const bool track_ranges = m_partial_gpu_write != 0 &&
		                          (first_write ? GpuWriteTrackEligible(image) : image.gpu_write.active);
		if (image.IsGpuModified()) {
			// The buffer cache takes ownership of these bytes; its Add bumps as well.
			// KYTY_ALIAS_BYTES: only of these bytes; the image keeps its others (rebuilds and
			// buffer reads move them into the buffer when needed).
			if (AliasBytesEnabled()) {
				(void)image.DisownBytes(address, size);
				if (image.IsGpuModified()) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesKept);
				}
			} else {
				image.ClearGpuModified();
			}
			if (!image.IsGpuModified()) {
				CleanVerdict::Invalidate(image.live.address, image.live.size,
				                         Coherence::Source::ImageGpuClear);
			}
		}
		if (exact && !image.depth_id &&
		    (image.info.data.address < address || image.info.data.End() > address + size)) {
			m_exact_range_totals.kept++;
			image.exact_range_stale = true;
			if (m_exact_range_mode == 1) {
				continue;
			}
		}
		const auto serial = image.ContentSerial();
		image.MarkBufferModified();
		if (track_ranges) {
			if (first_write) {
				image.BeginGpuWriteTrack(serial);
			} else {
				image.gpu_write.active = true; // MarkBufferModified ended it
			}
			if (!GpuWriteRanges::Record(image.gpu_write.ranges, address, size, image.live.address,
			                            image.live.End())) {
				image.gpu_write.active = false;
				m_pgw_totals.overflows++;
			}
		}
		image.NoteDirtySpan(address, size);
	}
	if (m_exact_range_mode != 0) {
		PrintExactRangeSummary();
	}
}

bool TextureCache::GpuWriteTrackEligible(const Image& image) const {
	const auto& info = image.info;
	if (!image.registered || info.data.Empty() || !image.FullyResident() || info.samples != 1 ||
	    info.metadata.compression != VideoOutCompression::Uncompressed || image.IsCpuDirty() ||
	    image.foreign_since_sync || image.exact_range_stale || info.IsVolume()) {
		return false;
	}
	if (image.depth_id) {
		// A stencil plane record: its contents are the depth image's stencil aspect. A record the
		// GPU wrote through itself is refreshed whole (the full path never moves its bytes).
		const auto* depth = m_slot_images.try_get(image.depth_id);
		return depth != nullptr && !image.IsGpuModified() && depth->info.samples == 1 &&
		       depth->backing.samples == 1 && !depth->info.metadata.stencil_compressed;
	}
	// The whole rebuild moves owned bytes through the buffer (KYTY_ALIAS_BYTES); without per-byte
	// ownership it would not keep the image's own bytes either. Colour transforms do not round-trip.
	return AliasBytesEnabled() && image.backing.image != nullptr && image.backing.samples == 1 &&
	       info.GetColorTransform() == ColorTransform::None;
}

void TextureCache::NoteExactRangeStaleUseSlow(Image& image, bool storage) {
	m_exact_range_totals.stale_uses++;
	if (m_exact_range_logged < 16) {
		m_exact_range_logged++;
		std::printf("Image exact-range invalidate: %s use of a kept image (0x%016" PRIx64
		            " size 0x%" PRIx64 " %ux%u format %u) before its next GPU write\n",
		            storage ? "storage" : "sampled", image.info.data.address, image.info.data.size,
		            image.info.extent.width, image.info.extent.height,
		            static_cast<uint32_t>(image.backing.format));
		std::fflush(stdout);
	}
}

void TextureCache::NoteExactRangeRewrite(Image& image) {
	if (image.exact_range_stale) {
		image.exact_range_stale = false;
		m_exact_range_totals.rewrites++;
	}
}

void TextureCache::PrintExactRangeSummary() {
	const auto now = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count());
	if (m_exact_range_print_ns == 0) {
		m_exact_range_print_ns = now;
		return;
	}
	if (now - m_exact_range_print_ns < 10'000'000'000ull) {
		return;
	}
	const auto& t = m_exact_range_totals;
	const auto& p = m_exact_range_printed;
	std::printf("Image exact-range invalidate %.0fs (%s): %" PRIu64
	            " buffer writes matched an image's range, %" PRIu64
	            " partly overlapped images %s, %" PRIu64 " uses of them before a GPU write, %" PRIu64
	            " written or refreshed again\n",
	            static_cast<double>(now - m_exact_range_print_ns) * 1e-9,
	            m_exact_range_mode == 1 ? "on" : "count", t.exact_writes - p.exact_writes,
	            t.kept - p.kept, m_exact_range_mode == 1 ? "kept" : "would be kept",
	            t.stale_uses - p.stale_uses, t.rewrites - p.rewrites);
	std::fflush(stdout);
	m_exact_range_printed  = t;
	m_exact_range_print_ns = now;
}

uint32_t TextureCache::CountImagesOutsideGpuWrite(uint64_t address, uint64_t size,
                                                  std::span<const GuestRange> written) {
	if (!GuestRange {address, size}.Valid()) {
		return 0;
	}
	std::scoped_lock lock {m_lock};
	uint32_t         count = 0;
	for (const auto id: FindImagesInRegion(address, size, true)) {
		const auto& image = m_slot_images[id];
		if (!image.Overlaps(address, size)) {
			continue;
		}
		const bool hit = std::any_of(written.begin(), written.end(), [&](const GuestRange& range) {
			return image.Overlaps(range.address, range.size);
		});
		count += hit ? 0u : 1u;
	}
	return count;
}

void TextureCache::InvalidateCleanImageProofs(uint64_t address, uint64_t size,
                                              Coherence::Source source) {
	// Every image register/unregister and GPU-modified transition passes here, before the
	// change is made under m_lock. Global clean-read verdicts depend on the same state.
	CleanVerdict::Invalidate(address, size, source);
	if (!m_clean_image_proofs) {
		return;
	}
	// Saturation disables reuse permanently instead of allowing an old epoch to match.
	if (m_clean_image_epoch != UINT64_MAX) {
		++m_clean_image_epoch;
	}
}

void TextureCache::MarkImageGpuModified(Image& image) {
	// KYTY_GUEST_HASH_IMAGES_KB: the GPU's pixels from now on, not the guest bytes an upload took.
	image.content_hash_valid = false;
	if (HangTrace::CpWatch(image.info.data.address, image.info.data.size)) {
		HangTrace::CpEvent event;
		event.event   = "gpuwrite-image";
		event.address = image.info.data.address;
		event.value   = static_cast<uint64_t>(image.backing.format);
		event.ref     = (uint64_t {image.info.extent.width} << 32u) | image.info.extent.height;
		event.aux     = image.IsGpuModified() ? 1 : 0;
		event.size    = image.info.data.size;
		HangTrace::RecordCp(event);
	}
	if (!image.FullyResident()) {
		// Every GPU-write path makes the image fully resident before recording. Reaching this
		// point partially resident would expose undefined levels: count and report it.
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidencyViolations);
		if (++m_residency_violations <= 16) {
			LOGF("TextureCache: GPU write to a partially resident image 0x%016" PRIx64
			     " (resident from level %u)\n",
			     image.info.data.address, image.resident_first);
		}
	}
	if (!image.IsGpuModified()) {
		// The ownership predicate (IsRegionGpuModified) covers the registered range `live`.
		InvalidateCleanImageProofs(image.live.address, image.live.size,
		                           Coherence::Source::ImageGpuModified);
		if (AliasBytesEnabled() && !image.depth_id && !image.info.data.Empty()) {
			// Owns no byte until the write (CommitGpuWrite) or copy that claims some.
			image.MarkGpuModified();
			image.OwnNoBytes();
		}
	}
	image.MarkGpuModified();
	// KYTY_IMAGE_PARTIAL_GPU_WRITE: a write to a depth image may write its stencil aspect, the
	// native contents of its stencil plane records (FindDepthTarget marks them itself, unless
	// KYTY_IMAGE_PARTIAL_GPU_WRITE_STENCIL=keep trusts depth-target draws not to write stencil).
	if (m_partial_gpu_write != 0 && !image.stencil_ids.empty() && !m_stencil_draw_suppressed) {
		for (const auto record: image.stencil_ids) {
			if (auto* stencil = m_slot_images.try_get(record); stencil != nullptr) {
				stencil->foreign_since_sync = true;
			}
		}
	}
	// Every GPU writer (draw targets, storage bindings, clears, helper passes, copies) passes
	// here before or right after recording its write: the native contents get a new identity.
	const auto serial_before = image.ContentSerial();
	image.NotePossibleWrite();
	if (m_bind_mark && image.gpu_write.active && image.gpu_write.serial == serial_before) {
		// KYTY_IMAGE_PARTIAL_GPU_WRITE: a binding marks its image before refreshing it (RefreshImage
		// right after); nothing has written the native contents in between.
		image.gpu_write.serial = image.ContentSerial();
	}
}

TextureCache::ContentMark TextureCache::MarkContent(ImageId id) {
	std::scoped_lock lock {m_lock};
	const auto*      image = m_slot_images.try_get(id);
	return image == nullptr ? ContentMark {}
	                        : ContentMark {image->ContentSerial(), image->DefiniteWrites()};
}

void TextureCache::RestoreContentIfUnwritten(ImageId id, const ContentMark& mark) {
	if (!AliasSyncSkipEnabled() || mark.serial == 0) {
		return;
	}
	std::scoped_lock lock {m_lock};
	auto*            image = m_slot_images.try_get(id);
	// Only bind-time possible writes happened since the mark: no upload, copy or clear.
	if (image != nullptr && WatchRange::Enabled()) {
		WatchRange::Note(fmt::format("depth restore-if-unwritten: mark serial {} now {} definite {} -> {}",
		                             mark.serial, image->ContentSerial(), mark.definite_writes,
		                             image->DefiniteWrites())
		                     .c_str(),
		                 image->info.data.address, image->info.data.size);
	}
	if (image != nullptr && image->DefiniteWrites() == mark.definite_writes) {
		image->AdoptContentSerial(mark.serial);
	}
}

std::string TextureCache::DescribeGpuModifiedImages(uint64_t address, uint64_t size) {
	std::string text;
	if (!GuestRange {address, size}.Valid()) {
		return text;
	}
	std::scoped_lock lock {m_lock};
	uint32_t         listed = 0;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		if (image.depth_id || !image.IsGpuModified()) {
			continue;
		}
		if (listed++ == 4) {
			text += "; ...";
			break;
		}
		if (!text.empty()) {
			text += "; ";
		}
		text += fmt::format("image 0x{:x}+0x{:x} {} {}x{} type {} alias_owner {} owns {} all {} cpu_dirty {}/{} "
		                    "span 0x{:x}+0x{:x} buffer_mod {} registered {} resident {} live 0x{:x}+0x{:x}",
		                    image.info.data.address, image.info.data.size, vk::to_string(image.backing.format),
		                    image.backing.extent.width, image.backing.extent.height,
		                    static_cast<uint32_t>(image.info.type), image.alias_owner ? 1 : 0,
		                    image.OwnsBytesIn(address, size) ? 1 : 0, image.OwnsAllBytes() ? 1 : 0,
		                    image.IsDefinitelyCpuDirty() ? 1 : 0, image.IsMaybeCpuDirty() ? 1 : 0,
		                    image.DirtySpanBegin(), image.DirtySpanBytes(), image.IsBufferModified() ? 1 : 0,
		                    image.registered ? 1 : 0, image.FullyResident() ? 1 : 0, image.live.address,
		                    image.live.size);
	}
	return text;
}

uint64_t TextureCache::MaterializeForRead(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return 0;
	}
	std::scoped_lock lock {m_lock};
	return MaterializeOwnedBytes({address, size}, {}, {}, "srt-read");
}

bool TextureCache::IsRegionGpuOwnedForRead(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || image->depth_id || !image->IsGpuModified() ||
		    !image->Overlaps(address, size, false)) {
			continue;
		}
		if (image->IsDefinitelyCpuDirty() && !image->HasKeptDirty()) {
			continue; // superseded: refreshed whole from guest memory, its bytes are no source
		}
		if (image->OwnsBytesIn(address, size)) {
			return true;
		}
	}
	return false;
}

bool TextureCache::IsRegionGpuModified(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	// A negative proof covers one complete 4 KiB page, while the owner index uses
	// 1 MiB pages. Dirty neighboring bytes only prevent caching: the exact query
	// below must still succeed for clean subranges. No guest bytes are cached/read.
	constexpr uint64_t proof_size = 4096;
	const auto proof_begin = Common::AlignDown(address, proof_size);
	const bool use_proof = m_clean_image_proofs && m_clean_image_epoch != UINT64_MAX &&
	                       size <= proof_size - (address - proof_begin);
	if (use_proof) {
		const auto page = proof_begin / proof_size;
		auto& proof = m_clean_image_pages[(page ^ (page >> 8)) % m_clean_image_pages.size()];
		if (proof.epoch == m_clean_image_epoch && proof.page == page) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureCleanProofHits);
			return false;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureCleanProofMisses);
		const auto* owners = m_image_page_table.Find(address >> ImagePageTable::kPageBits);
		bool page_modified = false;
		if (owners != nullptr) {
			for (const auto id: *owners) {
				const auto* image = m_slot_images.try_get(id);
				if (image == nullptr || image->depth_id || !image->IsGpuModified()) {
					continue;
				}
				if (image->Overlaps(address, size, false)) {
					return true;
				}
				page_modified |= image->Overlaps(proof_begin, proof_size, false);
			}
		}
		if (!page_modified) {
			proof = {page, m_clean_image_epoch};
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureCleanProofStores);
		}
		return false;
	}
	if (m_direct_dirty_image_query) {
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
			return false;
		}
		// Scalar SRT reads stay within one page. Keep multi-page queries on the
		// deduplicated path so large clean ranges cannot revisit every owner.
		if (pages.last_exclusive - pages.first == 1) {
			const auto* owners = m_image_page_table.Find(pages.first);
			if (owners == nullptr) {
				return false;
			}
			for (const auto id: *owners) {
				const auto* image = m_slot_images.try_get(id);
				if (image != nullptr && image->Overlaps(address, size, false) &&
				    !image->depth_id && image->IsGpuModified()) {
					return true;
				}
			}
			return false;
		}
	}
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		if (!image.depth_id && image.IsGpuModified()) {
			return true;
		}
	}
	return false;
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->ChunkTracked()) {
			InvalidateChunks(*owner, address, size);
			continue;
		}
		if (!owner->kept_holes.empty() || owner->HasKeptDirty() ||
		    (owner->Overlaps(address, size) && owner->IsGpuModified())) {
			// KYTY_IMAGE_GPU_KEEP: the written pages only (also a neighbour's write on a shared
			// page: its bytes of the image are refreshed, the page is released). The head/tail
			// untracking below does not know released pages: a kept image that cannot keep the
			// write takes the whole-image path.
			if (KeepGpuBytes(*owner, address, size, false)) {
				continue;
			}
			if (!owner->kept_holes.empty() || owner->HasKeptDirty()) {
				owner->InvalidateCpuWrite(owner->live.address, owner->live.size);
				UntrackImage(id);
				continue;
			}
		}
		if (owner->Overlaps(address, size)) {
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->live.address;
		const auto image_end   = owner->live.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsColorMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	return !m_color_metas.empty() && m_color_metas.contains(address);
}

// Live switch (common/liveSwitch.h). Every change of m_surface_metas bumps m_surface_meta_generation
// whatever the switch says, so a memo kept from before an off period would only be used if nothing
// changed since; the on-change hook also starts a new memo epoch (the A/B guide's memo reset), so no
// entry recorded before a live change is used after it.
static std::atomic<uint64_t> g_meta_clear_memo_epoch {0};
static Live::Switch          g_meta_clear_memo(
    "KYTY_META_CLEAR_MEMO",
    [](const char* value) -> int64_t {
        return value != nullptr && std::strcmp(value, "1") == 0 ? 1 : 0;
    },
    [](int64_t /*previous*/, int64_t /*value*/) {
        g_meta_clear_memo_epoch.fetch_add(1, std::memory_order_relaxed);
    });

static bool MetaClearMemoEnabled() {
	return g_meta_clear_memo.On();
}

// Slices 0-31 of an entry's clear mask (what KYTY_META_CLEAR_MEMO keeps; higher slices bypass it).
static uint32_t LowSliceClearMask(const auto& entry) {
	uint32_t mask = 0;
	for (uint32_t slice = 0; slice < 32u; slice++) {
		mask |= entry.clear_mask.test(slice) ? 1u << slice : 0u;
	}
	return mask;
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	if (MetaClearMemoEnabled() && slice < 32) {
		// The generation is read before the lookup and bumped after every change: a change racing
		// with this call leaves the memo stale for the next call only (it then looks up again), and
		// this answer is what the locked lookup gave just before that change. The memo epoch rides
		// in the high bits of the stored generation (the generation never reaches 2^48).
		const auto generation = m_surface_meta_generation.load(std::memory_order_acquire) ^
		                        (g_meta_clear_memo_epoch.load(std::memory_order_relaxed) << 48u);
		auto&      memo       = m_meta_clear_memo;
		if (!memo.valid || memo.address != address || memo.generation != generation) {
			std::scoped_lock lock {m_lock};
			const auto       found = m_surface_metas.find(address);
			memo = {.address    = address,
			        .generation = generation,
			        .clear_mask = found != m_surface_metas.end() ? LowSliceClearMask(found->second) : 0u,
			        .found      = found != m_surface_metas.end(),
			        .valid      = true};
		}
		return memo.found && (memo.clear_mask & (1u << slice)) != 0;
	}
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	return found->second.clear_mask.test(slice);
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask.set();
	NoteSurfaceMetaChange();
	return true;
}

bool TextureCache::HtileSliceClearEnabled() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_HTILE_SLICE_CLEAR");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

bool TextureCache::HtileSizeRuleEnabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_HTILE_SIZE_RULE");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

bool TextureCache::OversizedMetaWrite(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_lock};
	const auto found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || found->second.type != MetaDataInfo::Type::HTile ||
	    found->second.size == 0 || size <= found->second.size) {
		return false;
	}
	// Ghost of Yotei: the UI depth target's HTILE (0xa0000 bytes) lies at the start of the contact
	// shadow mask (0x870000 bytes R8); its per-frame fill was taken as an HTILE clear and dropped.
	static std::atomic<uint32_t> logged {0};
	const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 8u || std::has_single_bit(n)) {
		LOGF("KYTY_HTILE_SIZE_RULE #%u: write 0x%012" PRIx64 " size 0x%" PRIx64
		     " larger than its HTILE (0x%" PRIx64 "): ordinary write, entry erased\n",
		     n, address, size, found->second.size);
	}
	EraseSurfaceMeta(address);
	return true;
}

bool TextureCache::ClearMetaRange(uint64_t address, uint64_t size) {
	if (!HtileSliceClearEnabled()) {
		return ClearMeta(address);
	}
	std::scoped_lock lock {m_lock};
	static std::atomic<uint32_t> logged {0};
	const auto note = [&](const char* kind, uint64_t base, uint32_t first, uint32_t count, uint32_t layers) {
		const auto n = logged.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (n <= 16u || std::has_single_bit(n)) {
			LOGF("HTILE slice clear #%u (KYTY_HTILE_SLICE_CLEAR): %s htile 0x%012" PRIx64 " at 0x%012" PRIx64
			     " size 0x%" PRIx64 " layers %u..%u of %u\n",
			     n, kind, base, address, size, first, first + count - 1u, layers);
		}
	};
	if (const auto found = m_surface_metas.find(address); found != m_surface_metas.end()) {
		auto&      meta = found->second;
		const auto ss   = meta.slice_size;
		if (meta.type == MetaDataInfo::Type::HTile && ss != 0 && meta.layers > 1 && size != 0 &&
		    size % ss == 0 && size / ss < meta.layers) {
			const auto count = static_cast<uint32_t>(size / ss);
			for (uint32_t s = 0; s < count && s < meta.clear_mask.size(); s++) {
				meta.clear_mask.set(s);
			}
			note("first layers of", found->first, 0, count, meta.layers);
		} else {
			meta.clear_mask.set();
		}
		NoteSurfaceMetaChange();
		return true;
	}
	auto inside = m_surface_metas.upper_bound(address);
	if (inside == m_surface_metas.begin()) {
		return false;
	}
	--inside;
	auto&      meta = inside->second;
	const auto ss   = meta.slice_size;
	if (meta.type != MetaDataInfo::Type::HTile || ss == 0 || meta.layers <= 1 || address <= inside->first) {
		return false;
	}
	const auto offset = address - inside->first;
	const auto total  = ss * meta.layers;
	if (offset >= total || offset % ss != 0 || size == 0 || size % ss != 0 || size > total - offset) {
		return false;
	}
	const auto first = static_cast<uint32_t>(offset / ss);
	const auto count = static_cast<uint32_t>(size / ss);
	for (uint32_t s = first; s < first + count && s < meta.clear_mask.size(); s++) {
		meta.clear_mask.set(s);
	}
	NoteSurfaceMetaChange();
	note("layers inside", inside->first, first, count, meta.layers);
	// Not a clear at the base: the caller keeps the write itself (the guest bytes stay current).
	return false;
}

// KYTY_META_CLEAR_COVER=1 (default off; GT7 course-map preview depth): a buffer clear (compute
// fill or CP fill) whose range starts below a registered HTile but covers its first byte clears it
// too. Without it only a clear that starts exactly at the HTile address counts.
bool MetaClearCoverEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_META_CLEAR_COVER");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

uint32_t TextureCache::ClearMetaCovered(uint64_t address, uint64_t size) {
	if (size == 0 || size > UINT64_MAX - address) return 0;
	std::scoped_lock lock {m_lock};
	uint32_t         cleared = 0;
	for (auto it = m_surface_metas.upper_bound(address); it != m_surface_metas.end() && it->first < address + size;
	     ++it) {
		it->second.clear_mask.set();
		cleared++;
	}
	if (cleared != 0) NoteSurfaceMetaChange();
	return cleared;
}

// KYTY_META_TRACE=<hex address> (diagnostic): logs every compute buffer write and CP fill whose range
// holds that metadata address, and the result. With KYTY_META_TRACE=auto, KYTY_DRAW_TRACE_TARGET
// sets it to the HTile of the traced target's depth buffer.
static std::atomic<uint64_t> g_meta_trace_address {[] {
	const auto* value = std::getenv("KYTY_META_TRACE");
	if (value == nullptr || std::strcmp(value, "auto") == 0) return uint64_t {0};
	return static_cast<uint64_t>(std::strtoull(value, nullptr, 16));
}()};

uint64_t MetaTraceAddress() {
	return g_meta_trace_address.load(std::memory_order_relaxed);
}

void SetMetaTraceAddressAuto(uint64_t address) {
	static const bool automatic = [] {
		const auto* value = std::getenv("KYTY_META_TRACE");
		return value != nullptr && std::strcmp(value, "auto") == 0;
	}();
	if (!automatic || address == 0) return;
	if (g_meta_trace_address.exchange(address, std::memory_order_relaxed) != address) {
		LOGF("MetaTrace: tracing metadata address 0x%" PRIx64 "\n", address);
	}
}

void SetMetaTraceAddressStencil(uint64_t address) {
	if (address == 0) return;
	if (g_meta_trace_address.exchange(address, std::memory_order_relaxed) != address) {
		LOGF("MetaTrace: tracing stencil plane 0x%" PRIx64 "\n", address);
	}
}

bool MetaTraceLine() {
	static std::atomic<uint32_t> lines {0};
	static const uint32_t        max = [] {
		const auto* value = std::getenv("KYTY_META_TRACE_LINES");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 400u;
	}();
	return lines.fetch_add(1, std::memory_order_relaxed) < max;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= found->second.clear_mask.size()) {
		return false;
	}
	const bool previous = found->second.clear_mask.test(slice);
	found->second.clear_mask.set(slice, is_clear);
	if (previous != is_clear) {
		NoteSurfaceMetaChange();
	}
	return true;
}

bool TextureCache::IsRegionRegistered(uint64_t address, uint64_t size) {
	if (!MayOverlapImages(address, size)) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	return !FindImagesInRegion(address, size, false).empty();
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	m_gpu_dcc_inspections.clear();
	// The map is ordered by base address: erase the entries with a base in [address, address +
	// size) by range instead of walking the whole map on every unmap (as upstream does; based on
	// Senaxx/KytyPS5 089bebdc2). GuestRange::Valid above rules out an overflowing end.
	m_surface_metas.erase(m_surface_metas.lower_bound(address), m_surface_metas.lower_bound(address + size));
	NoteSurfaceMetaChange();
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		// KYTY_IMAGE_GPU_KEEP: an image the GPU wrote keeps its other bytes when only part of it is
		// unmapped (a pool the game remaps one layer at a time); the unmapped bytes are refreshed
		// from guest memory at its next use.
		if ((address > owner->live.address || address + size < owner->live.End()) &&
		    KeepGpuBytes(*owner, address, size, true)) {
			continue;
		}
		FreeImage(id, HangTrace::ImageFreeReason::Unmap);
	}
	// A partially resident image is registered on its resident prefix only. Its non-resident
	// bytes must stay mapped for a later residency extension: retire it with the mapping.
	// (Unmaps drain the GPU first; a scan of the live images is negligible next to that.)
	std::vector<ImageId> partial;
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered && !image.FullyResident() &&
		    ImageRangeOverlaps(image.info.data, GuestRange {address, size})) {
			partial.push_back(id);
		}
	});
	for (const auto id: partial) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TextureResidencyUnmapFrees);
		FreeImage(id, HangTrace::ImageFreeReason::Unmap);
	}
}

// KYTY_VRAM_GC_FIX=1 (live, default off; adapted from Erlou93/KytyPS5-Yotei's collector of the same
// name): with KYTY_VRAM_GC_BUDGET, when freeing clean images (RetireUnusedImages) does not bring the
// usage back to the trigger, GPU-written colour images unused for the same age are downloaded into
// guest memory (the tiler writes tiled images back) and freed at a later collection, once the
// write-back has published the bytes and only when the image was neither written (content serial)
// nor used (frame) since: a later use re-creates it from those bytes, as for any image that left the
// cache. Without it, GPU-written images stay however long they go unused (IdleRetirable). Never a
// depth/stencil image (HTILE and stencil planes are not in the bytes a download writes back), a
// stencil association, a multisampled, sparse, bound, target, presentation or bindless-pinned image,
// or one whose download would not be exact (SafeToDownload: CPU-dirty, buffer-modified, partly
// resident or with GPU-dirty buffer bytes in its range). At most 128 MiB of downloads are recorded
// per frame (256 MiB above the critical mark).
// KYTY_VRAM_GC_FIX_VERIFY=1 (live, diagnostic, stalls the GPU thread): before each free, an image is
// rebuilt from the published guest bytes and its native texels are compared with the original's; a
// mismatch keeps the image (logged, counted). KYTY_VRAM_GC_FIX_LOG=<frames> (default 600, 0 = off):
// a stats line while the switch is on.
static Live::Switch g_vram_gc_fix("KYTY_VRAM_GC_FIX", Live::ParseDefaultOff);
static Live::Switch g_vram_gc_fix_verify("KYTY_VRAM_GC_FIX_VERIFY", Live::ParseDefaultOff);

uint64_t TextureCache::LimitTickForAge(uint64_t frame, uint64_t min_age) const {
	if (frame <= min_age) {
		return 0;
	}
	const auto limit_frame = frame - min_age;
	const auto newer       = std::upper_bound(m_frame_ticks.begin(), m_frame_ticks.end(), limit_frame,
	                                          [](uint64_t value, const auto& entry) { return value < entry.first; });
	return newer == m_frame_ticks.begin() ? 0 : std::prev(newer)->second;
}

bool TextureCache::GcFixEligible(const Image& image, uint64_t frame, uint64_t min_age) {
	if (!image.registered || !image.IsGpuModified() || image.depth_id || image.info.IsDepth() ||
	    image.info.HasStencil() || image.usage.video_out || image.usage.depth_target ||
	    image.bindless_pinned || image.binding.is_bound || image.binding.is_target || image.info.samples != 1 ||
	    image.backing.samples != 1 || image.backing.image == nullptr || image.backing.sparse ||
	    frame - std::min(frame, image.frame_accessed_last) <= min_age) {
		return false;
	}
	return SafeToDownload(image) && BuildDownload(image).valid;
}

uint64_t TextureCache::CollectGpuWrittenImages(uint64_t frame, uint64_t min_age, uint64_t enough, bool critical) {
	const auto limit_tick = LimitTickForAge(frame, min_age);
	if (limit_tick == 0 || enough == 0) {
		return 0;
	}
	const uint64_t       max_download = critical ? (256ull << 20u) : (128ull << 20u);
	uint64_t             planned      = 0;
	size_t               scanned      = 0;
	std::vector<ImageId> downloads;
	m_lru_cache.ForEachItemBelow(limit_tick, [&](ImageId id) {
		if (++scanned > 16384 || planned >= enough) {
			return true;
		}
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || m_gc_fix_pending_ids.contains(id.index) || !GcFixEligible(*image, frame, min_age)) {
			return false;
		}
		const auto bytes = image->AccountedSize();
		if (planned + bytes > max_download && !downloads.empty()) {
			m_gc_fix_stats.skip_budget++;
			return false;
		}
		planned += bytes;
		downloads.push_back(id);
		return false;
	});
	for (const auto id: downloads) {
		auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered) {
			continue;
		}
		auto done = std::make_shared<std::atomic_bool>(false);
		if (!DownloadImageMemory(id, done)) {
			m_gc_fix_stats.failed++;
			continue;
		}
		image            = &m_slot_images[id];
		const auto bytes = image->AccountedSize();
		m_gc_fix_pending.push_back({id, image->ContentSerial(), frame, bytes, std::move(done)});
		m_gc_fix_pending_ids.insert(id.index);
		m_gc_fix_stats.downloads++;
		m_gc_fix_stats.download_bytes += bytes;
	}
	return planned;
}

void TextureCache::FinishGcFixEvictions(uint64_t frame) {
	size_t kept = 0;
	for (size_t i = 0; i < m_gc_fix_pending.size(); i++) {
		auto& pending = m_gc_fix_pending[i];
		auto* image   = m_slot_images.try_get(pending.id);
		if (image != nullptr && image->registered && !pending.done->load(std::memory_order_acquire)) {
			if (kept != i) {
				m_gc_fix_pending[kept] = std::move(pending);
			}
			kept++;
			continue;
		}
		m_gc_fix_pending_ids.erase(pending.id.index);
		if (image == nullptr || !image->registered) {
			continue; // freed meanwhile (unmap, overlap)
		}
		// Written or used since the download was recorded: it stays (guest memory holds an older copy,
		// as after any download). Also when the switch went off meanwhile.
		if (!g_vram_gc_fix.On() || image->ContentSerial() != pending.serial || !image->IsGpuModified() ||
		    image->frame_accessed_last > pending.frame || image->bindless_pinned || image->binding.is_bound ||
		    image->binding.is_target || !SafeToDownload(*image)) {
			m_gc_fix_stats.cancelled++;
			continue;
		}
		if (g_vram_gc_fix_verify.On()) {
			const int verdict = VerifyGcFixRoundTrip(pending.id);
			if (verdict < 0) {
				m_gc_fix_stats.verify_skipped++;
			} else if (verdict == 0) {
				m_gc_fix_stats.verify_mismatch++;
				continue; // kept: its bytes do not come back exactly
			} else {
				m_gc_fix_stats.verify_exact++;
			}
			image = m_slot_images.try_get(pending.id);
			if (image == nullptr || !image->registered) {
				continue;
			}
		}
		m_gc_fix_stats.freed++;
		m_gc_fix_stats.freed_bytes += image->AccountedSize();
		FreeImage(pending.id, HangTrace::ImageFreeReason::GarbageCollect);
	}
	m_gc_fix_pending.resize(kept);
	static const uint64_t log_frames = [] {
		const char* value = std::getenv("KYTY_VRAM_GC_FIX_LOG");
		return value != nullptr ? std::strtoull(value, nullptr, 10) : 600ull;
	}();
	if (log_frames != 0 && g_vram_gc_fix.On() && frame >= m_gc_fix_log_frame + log_frames) {
		m_gc_fix_log_frame = frame;
		const auto& s      = m_gc_fix_stats;
		LOGF("VramGcFix: frame=%" PRIu64 " downloads=%" PRIu64 " (%" PRIu64 " MiB) freed=%" PRIu64 " (%" PRIu64
		     " MiB) cancelled=%" PRIu64 " failed=%" PRIu64 " over_frame_budget=%" PRIu64
		     " pending=%zu verify exact=%" PRIu64 " mismatch=%" PRIu64 " unchecked=%" PRIu64 " usage=%" PRIu64 " MiB\n",
		     frame, s.downloads, s.download_bytes >> 20u, s.freed, s.freed_bytes >> 20u, s.cancelled, s.failed,
		     s.skip_budget, m_gc_fix_pending.size(), s.verify_exact, s.verify_mismatch, s.verify_skipped,
		     m_total_used_memory >> 20u);
	}
}

int TextureCache::VerifyGcFixRoundTrip(ImageId id) {
	const auto transfer = BuildDownload(m_slot_images[id]);
	const auto info     = m_slot_images[id].info;
	const auto usage    = m_slot_images[id].usage;
	const auto range    = info.data;
	if (!transfer.valid || transfer.depth_target || range.Empty()) {
		return -1;
	}
	// Native texels of both images, in the transfer's linear layout (before tiling and colour
	// transforms): the original, and an image rebuilt from the guest bytes the download published.
	const uint64_t linear_size = transfer.texture.tiles.empty() ? range.size : transfer.texture.LinearSize();
	if (linear_size == 0) {
		return -1;
	}
	auto upload = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Upload, 0, AllFlags, range.size);
	if (!LibKernel::Memory::TryReadBacking(range.address, upload->Mapped().data(), range.size)) {
		return -1;
	}
	upload->Flush(0, range.size);
	const auto copy = m_slot_images.insert(m_graphics, m_scheduler, info, 0u);
	m_slot_images[copy].usage = usage;
	UploadImage(m_slot_images[copy], *upload, 0);
	auto a = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, linear_size);
	auto b = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0, AllFlags, linear_size);
	std::memset(a->Mapped().data(), 0, linear_size);
	std::memset(b->Mapped().data(), 0, linear_size);
	a->Flush(0, linear_size);
	b->Flush(0, linear_size);
	m_scheduler.EndRendering();
	m_slot_images[id].Download(transfer.texture.regions, a->Handle(), 0, linear_size);
	m_slot_images[copy].Download(transfer.texture.regions, b->Handle(), 0, linear_size);
	m_scheduler.FlushAndWait();
	a->Invalidate(0, linear_size);
	b->Invalidate(0, linear_size);
	const auto* pa     = a->Mapped().data();
	const auto* pb     = b->Mapped().data();
	uint64_t    first  = UINT64_MAX;
	uint64_t    differ = 0;
	for (uint64_t i = 0; i < linear_size; i++) {
		if (pa[i] != pb[i]) {
			first = std::min(first, i);
			differ++;
		}
	}
	ReleaseUnregisteredImage(copy);
	// The GPU is done with the buffers (FlushAndWait), but they are released like the download path's.
	m_scheduler.DeferOperation([up = std::shared_ptr<Buffer>(std::move(upload)), da = std::shared_ptr<Buffer>(std::move(a)),
	                            db = std::shared_ptr<Buffer>(std::move(b))] {});
	static std::atomic<uint32_t> reports {0};
	if (differ != 0 && reports.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("VramGcFix verify: MISMATCH image 0x%" PRIx64 "+0x%" PRIx64 " %ux%ux%u levels=%u layers=%u vkfmt=%d "
		     "guest_fmt=%u tile=%u tiled=%d: %" PRIu64 " of %" PRIu64 " linear bytes differ, first at %" PRIu64 "\n",
		     range.address, range.size, info.extent.width, info.extent.height, info.extent.depth,
		     info.resources.levels, info.resources.layers, static_cast<int>(info.pixel_format),
		     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
		     transfer.texture.tiles.empty() ? 0 : 1, differ, linear_size, first);
	}
	return differ == 0 ? 1 : 0;
}

bool TextureCache::IdleRetirable(const Image& image, uint64_t frame, uint64_t min_age) const {
	// What the submission-age collector frees without a download: clean of GPU writes (the guest
	// bytes or the buffer cache hold the contents, a later use recreates and refreshes it), and
	// not a stencil association, a bound resource, a render target or a presentation surface.
	// GPU-written images stay however long they go unused (chenxiao07/KytyPS5 45beec32a frees them
	// after 600 flips): their native contents are the only copy of what the GPU rendered (guest
	// memory holds older bytes; a tiled image has no download path, and KYTY_ALIAS_BYTES would only
	// move the bytes into a cache buffer of the same size). An age says nothing about a later read:
	// a probe or impostor rendered once and sampled minutes later would come back as stale guest
	// bytes. Freeing them needs a download into guest memory first, not an age.
	return image.registered && !image.depth_id && !image.IsGpuModified() && !image.bindless_pinned &&
	       !image.binding.is_bound && !image.binding.is_target && !image.usage.video_out &&
	       frame - std::min(frame, image.frame_accessed_last) > min_age;
}

bool TextureCache::RunBudgetGarbageCollector(uint64_t frame) {
	if (!m_graphics.CanReportMemoryUsage()) {
		return false; // no driver budget: the stock collection
	}
	// Once per frame (this runs on the command-processor thread after every completed submission;
	// images are retired at most once per frame anyway).
	if (frame == m_budget_frame) {
		return true;
	}
	m_budget_frame    = frame;
	const auto budget = m_graphics.GetTotalMemoryBudget();
	m_budget_last     = budget;
	const auto plan   = VramBudget::PlanImages(budget, m_total_used_memory, m_pressure_frames);
	if (!plan.retire) {
		return true;
	}
	const auto before  = m_idle_freed;
	const auto freed   = RetireUnusedImages(frame, plan.age_frames, 256, plan.bytes);
	m_budget_freed_bytes += freed;
	m_budget_freed += m_idle_freed - before;
	// KYTY_VRAM_GC_FIX: the rest from GPU-written colour images (downloaded now, freed later).
	if (freed < plan.bytes && g_vram_gc_fix.On()) {
		(void)CollectGpuWrittenImages(frame, plan.age_frames, plan.bytes - freed,
		                              m_total_used_memory >= VramBudget::Critical(budget));
	}
	return true;
}

void TextureCache::NoteFrameTick(uint64_t frame, uint64_t tick) {
	if (m_frame_ticks.empty() || m_frame_ticks.back().first != frame) {
		m_frame_ticks.emplace_back(frame, tick);
	}
	const auto keep = std::max({m_idle_frames, m_pressure_frames, VramBudget::DefaultAgeFrames}) + 64;
	while (m_frame_ticks.size() > keep) {
		m_frame_ticks.pop_front();
	}
}

// The game's texture pools (one-level arrays it samples through views of moving layer ranges) can
// go unused for minutes while the camera is elsewhere; retired by age, the next view rebuilds the
// whole pool from guest memory (chenxiao07/KytyPS5 928366303: 1.3-2 GB uploads, 150-460 ms
// frames, 2-3 times a map in Demon's Souls). Kept only with plenty of video memory: pressure and
// budget collection still free them. Adapted from chenxiao07/KytyPS5 928366303 (no scene tags).
bool TextureCache::KeptIdlePool(const Image& image) const {
	if (m_idle_keep_pool_bytes == 0 || image.info.resources.levels != 1 ||
	    image.info.resources.layers <= 1 || image.AccountedSize() < m_idle_keep_pool_bytes ||
	    image.usage.storage || image.usage.render_target || image.usage.depth_target) {
		return false;
	}
	static const bool big_budget = [this] {
		return m_graphics.CanReportMemoryUsage() &&
		       m_graphics.GetTotalMemoryBudget() >= (uint64_t {12} << 30u);
	}();
	return big_budget;
}

uint64_t TextureCache::RetireUnusedImages(uint64_t frame, uint64_t min_age, size_t max_count,
                                          uint64_t enough, bool idle_pass) {
	if (frame <= min_age) {
		return 0;
	}
	// The GC tick the newest recorded frame up to frame - min_age began at: an LRU item below it
	// was last used before that frame (LRU ticks are GC ticks). IdleRetirable checks the exact age.
	const auto limit_frame = frame - min_age;
	const auto newer       = std::upper_bound(m_frame_ticks.begin(), m_frame_ticks.end(), limit_frame,
	                                          [](uint64_t value, const auto& entry) { return value < entry.first; });
	if (newer == m_frame_ticks.begin()) {
		return 0;
	}
	const auto           limit_tick = std::prev(newer)->second;
	std::vector<ImageId> candidates;
	size_t               scanned = 0;
	// Oldest first, bounded per pass. Entries it may not free (GPU-modified render targets at the
	// LRU head) are passed over, not counted as deletions.
	m_lru_cache.ForEachItemBelow(limit_tick, [&](ImageId id) {
		const auto* image = m_slot_images.try_get(id);
		if (image != nullptr && IdleRetirable(*image, frame, min_age) &&
		    !(idle_pass && KeptIdlePool(*image))) {
			candidates.push_back(id);
		}
		return ++scanned >= 16384 || candidates.size() >= max_count;
	});
	uint64_t freed = 0;
	for (const auto id: candidates) {
		if (freed >= enough) {
			break;
		}
		// Freeing a depth image frees its stencil associations: check again.
		const auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !IdleRetirable(*image, frame, min_age)) {
			continue;
		}
		freed += image->AccountedSize();
		// (Large retirements logged, switch on or off: a log shows whether a game's pools go.)
		if (idle_pass && image->info.resources.layers > 1 &&
		    image->AccountedSize() >= (uint64_t {128} << 20u)) {
			static std::atomic<uint32_t> reports {0};
			if (reports.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::printf("TextureCache: idle retirement of a %ux%u x%u-layer image (%" PRIu64
				            " MiB, %u levels) at 0x%" PRIx64 "\n",
				            image->info.extent.width, image->info.extent.height,
				            image->info.resources.layers, image->AccountedSize() >> 20u,
				            image->info.resources.levels, image->info.data.address);
			}
		}
		FreeImage(id, HangTrace::ImageFreeReason::GarbageCollect);
		++m_idle_freed;
	}
	return freed;
}

// KYTY_BINDLESS_GUEST_DROP_FRAMES=<n> (live, default 0 = off; adapted from Senaxx 89b188d74 via
// usa-ratt/KytyPS5-GT7, 300 there): the guest signals that a scene no longer needs a texture by
// replacing or clearing its texture-heap entries (it never unregisters resources). When no heap
// key refers to a bindless image any more, it is freed once no draw has used it, and it has not
// been dropped again, for n presented frames, whatever the cache's memory threshold. Only images
// the idle collector could free (IdleRetirable: clean of GPU writes, not bound, not a target, not
// pinned again by a heap key) go. The guest drops entries and takes the same texture back
// shortly after: a short grace frees and recreates textures (the fork's jungle slash stalled
// 13 s with 3 frames).
static Live::Switch g_guest_drop_frames("KYTY_BINDLESS_GUEST_DROP_FRAMES", [](const char* value) -> int64_t {
	return value != nullptr ? static_cast<int64_t>(std::strtoull(value, nullptr, 10)) : 0;
});

void TextureCache::ReleaseDroppedByGuest(ImageId id) {
	if (g_guest_drop_frames.Get() == 0) {
		return;
	}
	std::scoped_lock lock {m_lock};
	if (const auto* image = m_slot_images.try_get(id); image != nullptr && image->registered) {
		m_guest_dropped.emplace_back(id, m_frame.load(std::memory_order_relaxed));
	}
}

void TextureCache::FreeGuestDroppedImages(uint64_t frame) {
	const auto grace = static_cast<uint64_t>(g_guest_drop_frames.Get());
	if (m_guest_dropped.empty()) {
		return;
	}
	if (grace == 0) {
		m_guest_dropped.clear();
		return;
	}
	std::erase_if(m_guest_dropped, [&](const std::pair<ImageId, uint64_t>& entry) {
		const auto* image = m_slot_images.try_get(entry.first);
		if (image == nullptr || !image->registered || image->bindless_pinned ||
		    image->binding.is_target || image->IsGpuModified()) {
			return true; // gone, pinned again, or kept for the regular collector
		}
		if (entry.second + grace > frame || !IdleRetirable(*image, frame, grace)) {
			return false; // dropped or used lately (or bound): look again later
		}
		m_guest_dropped_freed++;
		m_guest_dropped_freed_bytes += image->AccountedSize();
		FreeImage(entry.first, HangTrace::ImageFreeReason::GarbageCollect);
		return true;
	});
	static uint64_t last_log = 0;
	if (frame >= last_log + 1000) {
		last_log = frame;
		LOGF("TexGuestDrop: frame=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 " MB) waiting=%zu "
		     "(KYTY_BINDLESS_GUEST_DROP_FRAMES=%" PRIu64 ")\n",
		     frame, m_guest_dropped_freed, m_guest_dropped_freed_bytes >> 20u,
		     m_guest_dropped.size(), grace);
	}
}

uint64_t TextureCache::SetBindlessEvictable(std::span<const ImageId> ids) {
	std::scoped_lock lock {m_lock};
	m_bindless_evictable.assign(ids.begin(), ids.end());
	uint64_t bytes = 0;
	for (const auto id: m_bindless_evictable) {
		if (const auto* image = m_slot_images.try_get(id); image != nullptr && image->registered) {
			bytes += image->AccountedSize();
		}
	}
	return bytes;
}

void TextureCache::FreeBindlessEvictable(uint64_t frame) {
	if (m_bindless_evictable.empty() || !BindlessUnderPressure()) {
		return;
	}
	constexpr size_t     PerCall = 64;
	constexpr uint64_t   MinAge  = 16; // frames since a bound (non-bindless) use
	size_t               freed   = 0;
	std::vector<ImageId> later;
	for (const auto id: m_bindless_evictable) {
		auto* image = m_slot_images.try_get(id);
		if (image == nullptr || !image->registered || !image->bindless_pinned) {
			continue; // gone, or no key refers to it any more (the regular collectors' now)
		}
		if (freed == PerCall || !BindlessUnderPressure()) {
			later.push_back(id);
			continue;
		}
		if (image->depth_id || image->IsGpuModified() || image->binding.is_bound ||
		    image->binding.is_target || image->usage.video_out ||
		    frame - std::min(frame, image->frame_accessed_last) <= MinAge) {
			continue; // its contents are not in guest memory, or it is in use otherwise
		}
		m_bindless_evicted++;
		m_bindless_evicted_bytes += image->AccountedSize();
		m_bindless_evicting = true;
		FreeImage(id, HangTrace::ImageFreeReason::PressureCollect);
		m_bindless_evicting = false;
		freed++;
	}
	m_bindless_evictable = std::move(later);
	if (freed != 0) {
		static uint64_t last_log = 0;
		if (frame >= last_log + 600) {
			last_log = frame;
			LOGF("TexBindlessEvict: frame=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64
			     " MB) (KYTY_BINDLESS_EVICT)\n",
			     frame, m_bindless_evicted, m_bindless_evicted_bytes >> 20u);
		}
	}
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	RetireIdlePartialImages();
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	const auto frame = m_frame.load(std::memory_order_relaxed);
	FreeGuestDroppedImages(frame);
	FreeBindlessEvictable(frame);
	if (!m_gc_fix_pending.empty() || g_vram_gc_fix.On()) {
		FinishGcFixEvictions(frame);
	}
	// Always (one entry per frame): KYTY_VRAM_GC_BUDGET is a live switch and needs recent frames.
	NoteFrameTick(frame, tick);
	if (m_idle_frames != 0 && frame >= m_idle_next_frame) {
		m_idle_next_frame = frame + 32;
		(void)RetireUnusedImages(frame, m_idle_frames, 512, UINT64_MAX, true);
	}
	m_tiler.TrimScratchPool();
	if ((tick & 63u) == 0 && Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
		if (m_graphics.CanReportMemoryUsage()) {
			TracyPlot("ImageCache.DeviceMemoryBytes", static_cast<double>(m_total_used_memory));
		}
		TracyPlot("ImageCache.RegisteredGuestBytes", static_cast<double>(m_registered_image_memory));
		TracyPlot("ImageCache.PendingRetirementBytes", static_cast<double>(m_pressure_retirement_bytes));
	}
	if (VramBudget::GcEnabled() && RunBudgetGarbageCollector(frame)) {
		return;
	}
	if (m_pressure_gc_enabled) {
		RunPressureGarbageCollector(tick);
		return;
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	if (m_pressure_frames != 0) {
		// KYTY_VRAM_PRESSURE_FRAMES: once per frame, retire images unused for that many frames,
		// oldest first, until usage is back under the trigger (a quarter of the age, at least two
		// frames, above the critical mark). Never an image used in the last frames, never a
		// download: no submission-age collection below.
		if (frame != m_pressure_frame) {
			m_pressure_frame       = frame;
			const auto critical    = m_total_used_memory >= m_critical_gc_memory;
			const auto age         = critical ? std::max<uint64_t>(m_pressure_frames / 4, 2)
			                                  : m_pressure_frames;
			const auto excess      = m_total_used_memory - m_trigger_gc_memory;
			m_pressure_freed_bytes += RetireUnusedImages(frame, age, 256, excess);
		}
		return;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool           pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool           aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		const uint64_t age       = std::min<uint64_t>(aggressive ? 160 : pressured ? 80 : 16, tick);
		size_t         deletions = aggressive ? 40 : pressured ? 20 : 10;
		std::vector<ImageId> candidates;
		candidates.reserve(deletions);
		// KYTY_IMAGE_GC_FILTER (default on; =0 counts every entry; chenxiao07/KytyPS5 45beec32a):
		// the walk passes over the images the loop below keeps, without counting them against the
		// budget. Those are never used again either, so they stay at the front of the LRU list:
		// counted, every pass examined the same kept images and deleted nothing behind them.
		const auto kept = [&](ImageId id) {
			const auto* owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				return true; // a stencil association goes with its depth image
			}
			if (KeepSmallRenderTarget(*owner)) {
				return true;
			}
			if (!owner->IsGpuModified()) {
				return false;
			}
			// Under pressure a GPU-written image is downloaded and freed, tiled or not (upstream
			// 16a7bbe3c). Unlike upstream, depth/stencil images are not kept: ours retires them with a
			// deferred download (UnifiedTextureCacheFlow "depth/stencil deferred pressure retirement").
			const bool safe = SafeToDownload(*owner);
			return safe && !pressured;
		};
		size_t passed = 0;
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			if (m_gc_filter_candidates && kept(id)) {
				return ++passed >= 4096; // bounded per pass
			}
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id || KeepSmallRenderTarget(*owner)) {
				continue;
			}
			if (owner->IsGpuModified() && SafeToDownload(*owner) &&
			    (!pressured || !DownloadImageMemory(id))) {
				continue; // upstream 16a7bbe3c: tiled GPU-written images are no longer skipped
			}
			--deletions; // only images actually freed spend the budget (16a7bbe3c)
			FreeImage(id, HangTrace::ImageFreeReason::GarbageCollect);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::TextureEvictions);
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
}

void TextureCache::RunPressureGarbageCollector(uint64_t tick) {
	// Driver heap usage includes images awaiting deferred destruction. Do not evict
	// a second cache's worth of resources while those bytes are already retiring.
	const auto pending_bytes = m_graphics.CanReportMemoryUsage() ? m_pressure_retirement_bytes : 0;
	const auto effective_usage = m_total_used_memory - std::min(m_total_used_memory, pending_bytes);
	const auto plan = m_pressure_gc_policy.Begin(effective_usage, tick);
	if (plan.deletions == 0) {
		return;
	}
	KYTY_PROFILER_BLOCK("TextureCache::PressureGarbageCollection");
	m_scheduler.GetMasterSemaphore().Refresh();
	const auto completed_tick = m_scheduler.GetMasterSemaphore().KnownGpuTick();
	std::vector<ImageId> candidates;
	candidates.reserve(std::min<size_t>(plan.candidates, 256));
	// Complete traversal before deleting depth images and their stencil associations.
	m_lru_cache.ScanItemsBelow(tick, m_pressure_gc_cursor, plan.candidates,
	                          [&](ImageId id) { candidates.push_back(id); });

	uint64_t reclaimed_bytes = 0;
	size_t deletions = 0;
	for (const auto id: candidates) {
		if (deletions >= plan.deletions || reclaimed_bytes >= plan.bytes ||
		    effective_usage - std::min(effective_usage, reclaimed_bytes) <= m_pressure_gc_policy.Low()) {
			break;
		}
		auto* owner = m_slot_images.try_get(id);
		if (owner == nullptr || !owner->registered || owner->depth_id || KeepSmallRenderTarget(*owner)) {
			continue;
		}
		// Retain resources still in use whenever pressure leaves room to do so.
		// The critical path retains the existing deferred timeline retirement.
		if (!plan.critical && owner->tick_accessed_last > completed_tick) {
			continue;
		}
		if (owner->IsGpuModified()) {
			const bool safe = SafeToDownload(*owner);
			if (safe && !plan.critical) {
				continue;
			}
			if (safe && owner->info.IsTiled()) {
				continue;
			}
			if (safe && !DownloadImageMemory(id)) {
				continue;
			}
		}
		const auto before = m_total_used_memory;
		FreeImage(id, HangTrace::ImageFreeReason::PressureCollect);
		const auto accounted = before - m_total_used_memory;
		reclaimed_bytes += accounted;
		++deletions;
	}
	// A download can drain the scheduler mid-collection, running destruction
	// callbacks. Refresh actual heap usage instead of inferring release by tick.
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	const auto pending_after = m_graphics.CanReportMemoryUsage() ? m_pressure_retirement_bytes : 0;
	m_pressure_gc_policy.Complete(tick,
	    m_total_used_memory - std::min(m_total_used_memory, pending_after), deletions != 0);
}

void TextureCache::ReportVram() {
	using VramStats::ToMiB;
	struct Bucket {
		uint64_t bytes = 0;
		uint64_t count = 0;
		void     Add(uint64_t value) {
            bytes += value;
            ++count;
		}
	};
	enum : size_t { KindTexture, KindRenderTarget, KindDepth, KindStorage, KindVideoOut, KindCount };
	static constexpr std::array<const char*, KindCount> kind_names {"texture", "render-target", "depth",
	                                                                "storage", "video-out"};
	// Presented frames since the image was last used: <=1, 2-10, 11-60, 61-600, >600.
	static constexpr std::array<uint64_t, 4>    age_limits {1, 10, 60, 600};
	static constexpr std::array<const char*, 5> age_names {"<=1", "2-10", "11-60", "61-600", ">600"};
	const auto age_index = [](uint64_t age) {
		size_t index = 0;
		while (index < age_limits.size() && age > age_limits[index]) {
			++index;
		}
		return index;
	};
	struct Entry {
		uint64_t bytes = 0;
		ImageId  id;
		size_t   kind = 0;
		uint64_t age  = 0;
	};

	std::scoped_lock lock {m_lock};
	const auto       frame = m_frame.load(std::memory_order_relaxed);
	std::array<std::array<Bucket, age_names.size()>, KindCount> by_kind_age {};
	std::array<Bucket, KindCount>                               by_kind {};
	Bucket   registered;
	Bucket   unregistered;
	Bucket   gpu_modified;
	Bucket   gpu_modified_idle;
	Bucket   partial;
	uint64_t partial_skipped = 0;
	uint64_t no_backing      = 0;
	std::map<std::pair<uint32_t, uint32_t>, Bucket>                         target_sizes;
	std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> by_address;
	std::vector<Entry>                                                      entries;
	entries.reserve(m_slot_images.size());
	Bucket sparse;
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.backing.image == nullptr) {
			++no_backing;
			return;
		}
		const auto   bytes = m_graphics.NativeImageBytes(image.backing);
		if (image.backing.sparse) {
			sparse.Add(bytes);
		}
		const size_t kind  = image.info.IsDepth() || image.usage.depth_target ? KindDepth
		                     : image.usage.video_out                          ? KindVideoOut
		                     : image.usage.render_target                      ? KindRenderTarget
		                     : image.usage.storage                            ? KindStorage
		                                                                      : KindTexture;
		const auto   age   = frame - std::min(frame, image.frame_accessed_last);
		by_kind[kind].Add(bytes);
		by_kind_age[kind][age_index(age)].Add(bytes);
		(image.registered ? registered : unregistered).Add(bytes);
		if (image.IsGpuModified()) {
			gpu_modified.Add(bytes);
			if (age > 60) {
				gpu_modified_idle.Add(bytes);
			}
		}
		if (!image.FullyResident()) {
			partial.Add(bytes);
			// The native bytes of levels no view samples (proportional to their texels).
			double all     = 0.0;
			double skipped = 0.0;
			for (uint32_t level = 0; level < image.info.resources.levels; ++level) {
				const double texels = static_cast<double>(std::max(image.info.extent.width >> level, 1u)) *
				                      static_cast<double>(std::max(image.info.extent.height >> level, 1u)) *
				                      static_cast<double>(std::max(image.info.extent.depth >> level, 1u));
				all += texels;
				if (level < image.resident_first) {
					skipped += texels;
				}
			}
			if (all > 0.0 && !image.backing.sparse) {
				partial_skipped += static_cast<uint64_t>(static_cast<double>(bytes) * (skipped / all));
			}
		}
		if (kind == KindRenderTarget || kind == KindDepth) {
			target_sizes[{image.info.extent.width, image.info.extent.height}].Add(bytes);
		}
		if (image.registered) {
			by_address[image.info.data.address].emplace_back(image.frame_accessed_last, bytes);
		}
		entries.push_back({bytes, id, kind, age});
	});

	VramStats::Line("images %zu: registered %.1f MiB (%llu), awaiting deletion %.1f MiB (%llu), "
	                "no native image %llu | GPU-modified %.1f MiB (%llu), unused >60 frames %.1f MiB "
	                "(%llu) | frame %llu",
	                entries.size(), ToMiB(registered.bytes),
	                static_cast<unsigned long long>(registered.count), ToMiB(unregistered.bytes),
	                static_cast<unsigned long long>(unregistered.count),
	                static_cast<unsigned long long>(no_backing), ToMiB(gpu_modified.bytes),
	                static_cast<unsigned long long>(gpu_modified.count), ToMiB(gpu_modified_idle.bytes),
	                static_cast<unsigned long long>(gpu_modified_idle.count),
	                static_cast<unsigned long long>(frame));
	for (size_t kind = 0; kind < KindCount; ++kind) {
		std::string ages;
		for (size_t age = 0; age < age_names.size(); ++age) {
			char part[64];
			std::snprintf(part, sizeof(part), " %s:%.1f(%llu)", age_names[age],
			              ToMiB(by_kind_age[kind][age].bytes),
			              static_cast<unsigned long long>(by_kind_age[kind][age].count));
			ages += part;
		}
		VramStats::Line("  %-13s %.1f MiB (%llu) | frames since use:%s", kind_names[kind],
		                ToMiB(by_kind[kind].bytes), static_cast<unsigned long long>(by_kind[kind].count),
		                ages.c_str());
	}
	VramStats::Line("partially resident: %.1f MiB (%llu), about %.1f MiB of it in levels no view "
	                "samples; %zu tracked for idle retirement | sparse (KYTY_TEXTURE_SPARSE_RESIDENCY): "
	                "%.1f MiB bound in %llu images",
	                ToMiB(partial.bytes), static_cast<unsigned long long>(partial.count),
	                ToMiB(partial_skipped), m_partial_images.size(), ToMiB(sparse.bytes),
	                static_cast<unsigned long long>(sparse.count));
	{
		std::vector<std::pair<std::pair<uint32_t, uint32_t>, Bucket>> sizes(target_sizes.begin(),
		                                                                    target_sizes.end());
		std::sort(sizes.begin(), sizes.end(),
		          [](const auto& a, const auto& b) { return a.second.bytes > b.second.bytes; });
		std::string line;
		for (size_t index = 0; index < std::min<size_t>(sizes.size(), 12); ++index) {
			char part[64];
			std::snprintf(part, sizeof(part), " %ux%u:%.1f(%llu)", sizes[index].first.first,
			              sizes[index].first.second, ToMiB(sizes[index].second.bytes),
			              static_cast<unsigned long long>(sizes[index].second.count));
			line += part;
		}
		VramStats::Line("target sizes (render+depth, MiB):%s", line.c_str());
	}
	{
		uint64_t groups      = 0;
		uint64_t older       = 0;
		uint64_t older_bytes = 0;
		for (auto& [address, list]: by_address) {
			(void)address;
			if (list.size() < 2) {
				continue;
			}
			++groups;
			const auto newest = std::max_element(list.begin(), list.end());
			for (auto it = list.begin(); it != list.end(); ++it) {
				if (it != newest) {
					++older;
					older_bytes += it->second;
				}
			}
		}
		VramStats::Line("aliases: %llu guest addresses hold 2+ registered images; all but the most "
		                "recently used one: %.1f MiB (%llu)",
		                static_cast<unsigned long long>(groups), ToMiB(older_bytes),
		                static_cast<unsigned long long>(older));
	}
	const size_t top = std::min<size_t>(entries.size(), 16);
	std::partial_sort(entries.begin(), entries.begin() + static_cast<std::ptrdiff_t>(top),
	                  entries.end(), [](const Entry& a, const Entry& b) { return a.bytes > b.bytes; });
	for (size_t index = 0; index < top; ++index) {
		const auto& entry = entries[index];
		const auto& image = m_slot_images[entry.id];
		VramStats::Line("  top %.1f MiB %ux%ux%u mips=%u layers=%u samples=%u vkfmt=%d %s%s%s "
		                "unused=%llu frames resident_first=%u guest=0x%" PRIx64 "+0x%" PRIx64,
		                ToMiB(entry.bytes), image.info.extent.width, image.info.extent.height,
		                image.info.extent.depth, image.info.resources.levels,
		                image.info.resources.layers, image.info.samples,
		                static_cast<int>(image.info.pixel_format), kind_names[entry.kind],
		                image.IsGpuModified() ? " gpu-modified" : "",
		                image.registered ? "" : " awaiting-deletion",
		                static_cast<unsigned long long>(entry.age), image.resident_first,
		                image.info.data.address, image.info.data.size);
	}
	{
		static constexpr std::array<const char*, static_cast<size_t>(HangTrace::ImageFreeReason::Count)>
		            reason_names {"other",        "depth-association", "depth-recreate", "overlap-layout",
		                          "overlap-mip-merge", "overlap-stale", "expand", "smaller-resources",
		                          "unmap",        "gc",                "pressure-gc",    "resident-idle"};
		std::string frees;
		for (size_t reason = 0; reason < m_vram_frees.size(); ++reason) {
			if (m_vram_frees[reason].first == 0) {
				continue;
			}
			char part[80];
			std::snprintf(part, sizeof(part), " %s:%.1f(%llu)", reason_names[reason],
			              ToMiB(m_vram_frees[reason].second),
			              static_cast<unsigned long long>(m_vram_frees[reason].first));
			frees += part;
		}
		VramStats::Line("since last report: created %.1f MiB (%llu) | freed MiB (count):%s | "
		                "frame-aged retirement (KYTY_VRAM_IDLE_FRAMES=%llu, KYTY_VRAM_PRESSURE_FRAMES=%llu) "
		                "freed %llu images in total, %.1f MiB under pressure",
		                ToMiB(m_vram_creates.second),
		                static_cast<unsigned long long>(m_vram_creates.first), frees.c_str(),
		                static_cast<unsigned long long>(m_idle_frames),
		                static_cast<unsigned long long>(m_pressure_frames),
		                static_cast<unsigned long long>(m_idle_freed), ToMiB(m_pressure_freed_bytes));
		m_vram_frees   = {};
		m_vram_creates = {};
	}
	const auto [scratch_idle, scratch_limit] = m_tiler.ScratchPoolBytes();
	const bool budget_mode = VramBudget::GcEnabled() && m_graphics.CanReportMemoryUsage();
	VramStats::Line("GC: %s, usage %.1f MiB, trigger %.1f pressure %.1f critical %.1f MiB, registered "
	                "guest bytes %.1f MiB, pending retirement %.1f MiB, GC tick %llu | tiler scratch "
	                "pool %.1f of %.1f MiB idle",
	                budget_mode            ? "budget (KYTY_VRAM_GC_BUDGET)"
	                : m_pressure_gc_enabled ? "pressure policy"
	                                        : "submission age",
	                ToMiB(m_total_used_memory),
	                ToMiB(budget_mode ? VramBudget::ImageTrigger(m_budget_last) : m_trigger_gc_memory),
	                ToMiB(budget_mode ? VramBudget::ImageTrigger(m_budget_last) : m_pressure_gc_memory),
	                ToMiB(budget_mode ? VramBudget::Critical(m_budget_last) : m_critical_gc_memory),
	                ToMiB(m_registered_image_memory), ToMiB(m_pressure_retirement_bytes),
	                static_cast<unsigned long long>(m_gc_tick), ToMiB(scratch_idle), ToMiB(scratch_limit));
	if (budget_mode) {
		VramStats::Line("budget GC: planning budget %.1f MiB, freed %llu images / %.1f MiB in total (age %llu "
		                "frames, a quarter above the critical mark)",
		                ToMiB(m_budget_last), static_cast<unsigned long long>(m_budget_freed),
		                ToMiB(m_budget_freed_bytes),
		                static_cast<unsigned long long>(m_pressure_frames != 0 ? m_pressure_frames
		                                                                       : VramBudget::DefaultAgeFrames));
	}
}

void TextureCache::ProcessDownloadImages() {
	if (!m_readback_linear_images && !m_readback_tiled_images && SmallLinearReadbackBytes() == 0) {
		return;
	}
	std::scoped_lock lock {m_lock};
	// KYTY_READBACK_LOG=<n> (diagnostic, default off): the first n images enrolled for a readback here, with the
	// result (downloaded, refused, not GPU-owned), and one count line every 1024 calls.
	static const uint32_t log_limit = [] {
		const char* value = std::getenv("KYTY_READBACK_LOG");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
	}();
	static uint32_t logged = 0, calls = 0;
	static uint64_t done = 0, refused = 0, skipped = 0;
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		const bool owned = owner != nullptr && owner->registered && owner->IsGpuModified();
		const bool ok    = owned && DownloadImageMemory(id);
		if (log_limit == 0) {
			continue;
		}
		(ok ? done : owned ? refused : skipped)++;
		if (logged < log_limit && owner != nullptr) {
			logged++;
			std::printf("KYTY_READBACK_LOG: frame %" PRIu64 " image 0x%012" PRIx64 " size 0x%" PRIx64
			            " %ux%ux%u fmt %u tile %u depth %d %s\n",
			            m_frame.load(std::memory_order_relaxed), owner->info.data.address, owner->info.data.size,
			            owner->info.extent.width, owner->info.extent.height, owner->info.extent.depth,
			            static_cast<uint32_t>(owner->info.pixel_format), static_cast<uint32_t>(owner->info.tile_mode),
			            owner->info.IsDepth() ? 1 : 0, ok ? "downloaded" : owned ? "refused" : "not GPU-owned");
		}
	}
	if (log_limit != 0 && (++calls & 1023u) == 0) {
		std::printf("KYTY_READBACK_LOG: frame %" PRIu64 " downloads %" PRIu64 " refused %" PRIu64 " skipped %" PRIu64 "\n",
		            m_frame.load(std::memory_order_relaxed), done, refused, skipped);
	}
	m_download_images.clear();
}

uint64_t TextureCache::WriteBackTick(uint64_t address, uint64_t size) {
	if (!m_readback_storage_images) {
		return 0;
	}
	const auto begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	uint64_t   tick  = 0;

	std::scoped_lock write_back_lock {m_storage_write_back_mutex};
	for (const auto& write_back: m_storage_write_backs) {
		if (write_back.range.address < end && begin < write_back.range.End()) {
			tick = std::max(tick, write_back.tick);
		}
	}
	return tick;
}

bool TextureCache::WriteBackImages() {
	return m_readback_storage_images && DownloadQueuedImages();
}

// Storage readback at a GL2 writeback (upstream 9678c59b7): the queued GPU-written images.
bool TextureCache::DownloadQueuedImages() {
	std::scoped_lock lock {m_lock};
	bool             downloaded = false;
	for (const auto id: m_download_images) {
		const auto* owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			downloaded |= DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
	return downloaded;
}

} // namespace Libs::Graphics
