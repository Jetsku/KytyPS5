#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/hangTrace.h"
#include "common/emulatorConfig.h"
#include "common/liveSwitch.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/cleanVerdictCache.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/memoryStats.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/image/dccClear.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/stagingCopier.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"
#include "graphics/host_gpu/vramStats.h"

#include <algorithm>
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
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;

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
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr || std::strcmp(value, "ticks") != 0;
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

[[nodiscard]] bool DecodeDccClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	switch (code) {
		case 0x00:
		case 0x20:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	if (code == 0x20) {
		// Clear-to-register is a color-buffer operation; the texture pipe cannot decode it.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.dcc_clear_register_valid &&
		       DecodePackedColorClear(format, metadata.dcc_clear_word, clear);
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

} // namespace

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	bool                             swap_bgra16 = false;
	bool                             valid       = false;

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
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
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
	static const bool enabled = AliasAgeByFrames() && EnvNotZero("KYTY_ALIAS_BYTES");
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
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    (cached.type != requested.type && requested.extent != vk::Extent3D {1, 1, 1})) {
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
			image.live           = {info.data.address, resident_prefix};
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
	image.live = new_first != 0 ? GuestRange {image.info.data.address, prefix} : image.info.data;
	if (image.backing.sparse) {
		// KYTY_TEXTURE_SPARSE_RESIDENCY: memory behind the newly resident levels, bound (and waited
		// for) before anything records a write of them; their refresh follows (MarkResidencyDirty).
		m_graphics.BindSparseImageLevels(image.backing, new_first);
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
	InvalidateCleanImageProofs(image.live.address, image.live.size,
	                           Coherence::Source::ImageRegister);
	NoteStructureChange(image);
	ForEachPage(image.live.address, image.live.size, [this, id](uint64_t page) {
		// Counted before the image enters the page's owner list, so a lock-free zero count means
		// FindImagesInRegion finds nothing there (NoImagesOnPages); and before the image can
		// watch pages (TrackImage follows registration).
		m_image_page_counts[page].fetch_add(1, std::memory_order_seq_cst);
		m_image_page_table[page].push_back(id);
	});
	image.registered = true;
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
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	EXIT_IF(accounted > m_registered_image_memory);
	m_registered_image_memory -= accounted;
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
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
	// an image's pixel format never changes), so retiring a color image skips the full scan.
	if (!image->depth_id && image->info.IsDepth()) {
		std::vector<ImageId> associations;
		m_slot_images.ForEach([&](ImageId candidate, const Image& associated) {
			if (associated.depth_id == id) {
				associations.push_back(candidate);
			}
		});
		for (const auto association: associations) {
			FreeImage(association, HangTrace::ImageFreeReason::DepthAssociation);
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
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
			NoteSurfaceMetaChange();
		}
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
			EXIT_IF(retiring_bytes > m_pressure_retirement_bytes);
			m_pressure_retirement_bytes -= retiring_bytes;
		});
	} else {
		m_slot_images.erase(id);
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
	}
	if (image.IsGpuModified()) {
		CleanVerdict::Invalidate(image.live.address, image.live.size,
		                         Coherence::Source::ImageGpuClear);
		image.ClearGpuModified();
	}
	DeleteImage(id);
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
	const auto image_begin = image.live.address;
	const auto image_end   = image.live.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
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
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::TrackChunkImage(Image& image) {
	if (!image.IsTracked()) {
		// Nothing is watched: the untracked marks describe holes of a previous tracked period.
		std::fill(image.chunks.untracked.begin(), image.chunks.untracked.end(), 0);
		image.chunks.untracked_count = 0;
		image.track_addr     = image.live.address;
		image.track_addr_end = image.live.End();
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
	} else {
		chunks.untracked_count += changed;
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

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	if (image.ChunkTracked()) {
		(void)UpdateChunkWatchers<false>(image, 0, image.chunks.count);
		std::fill(image.chunks.untracked.begin(), image.chunks.untracked.end(), 0);
		image.chunks.untracked_count = 0;
		image.track_addr     = 0;
		image.track_addr_end = 0;
		// Writes are no longer observed: only a full refresh can restore partial validity.
		image.SetPartialValid(false);
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
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
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (m_readback_linear_images && !image.info.IsTiled() && !image.info.data.Empty()) {
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
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
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

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto format = desc.info.pixel_format;
	if (const auto found = m_null_images.find(format); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = Prospero::ImageType::kColor2D;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	// A native image starts with undefined texels; give fallback reads a defined value.
	vk::ClearValue clear {};
	ClearImage(m_scheduler.Current(), id, format,
	           {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, clear);
	m_null_images.emplace(format, id);
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
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
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
		color_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
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
	const bool direct_copy =
	    source.backing.format == destination.backing.format ||
	    (!source_depth && !dest_depth &&
	     vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format));
	// Lossless paths copy raw texel bits; the D16 path converts through unorm16.
	bool        lossless = true;
	const char* path     = nullptr;
	if (direct_copy) {
		destination.CopyImage(source);
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyDirect);
		path = "direct";
	} else if ((path = TryDirectReinterpret(destination, source)) != nullptr) {
	} else if (CopyD16(destination, source)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::ImageCopyD16);
		lossless = false;
		path     = "d16-convert";
	} else {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer);
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
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if (!cached.info.IsDepth() && !requested.IsDepth()) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.tile_mode == cached.info.tile_mode && !requested.HasStencil() &&
	    !cached.info.HasStencil() && !requested.HasMetadata() && !cached.info.HasMetadata();
	// PPSA04264
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
	    !cached.info.HasMetadata();
	bool recreate = cached.info.resources < requested.resources;
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	// Guests reuse one allocation as depth and as color (e.g. D32F <-> R32F) every frame. Keep one
	// image per interpretation alive and reuse it: FindImage copies the current owner's contents
	// into a non-owner alias (SyncAliasFromOwner) instead of destroying and recreating images.
	const bool keep_aliases = AliasAgeByFrames() && !(cached.info.resources < requested.resources);
	if (keep_aliases) {
		for (const auto other_id: FindImagesInRegion(requested.data.address, requested.data.size, false)) {
			const auto* other = m_slot_images.try_get(other_id);
			if (other == nullptr || other_id == cached_id || !other->registered || other->depth_id ||
			    other->backing.samples != cached.backing.samples ||
			    !SameBacking(other->info, requested, true)) {
				continue;
			}
			if (cached.binding.is_bound || cached.binding.is_target) {
				cached.binding.needs_rebind = true;
			}
			return other_id;
		}
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	} else {
		info.resources = std::max(requested.resources, cached.info.resources);
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			(void)CopyImage(replacement_id, cached_id, "depth-overlap");
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
	if (keep_aliases && cached.backing.samples == replacement.backing.samples &&
	    !cached.info.HasStencil() && !replacement.info.HasStencil()) {
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

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick = m_scheduler.CurrentTick();
	const auto current_frame = m_frame.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                AliasFramesBeforeRemoval);

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.data.size == cached.info.data.size &&
		    (requested.IsVolume() || cached.info.IsVolume())) {
			return {ExpandImage(requested, cached_id)};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			if (safe_to_delete) {
				FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapLayout);
			}
			return {merged_id};
		}
		// PPSA08394
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources && requested.type == cached.info.type &&
		    requested.extent.width > cached.info.extent.width &&
		    requested.extent.height >= cached.info.extent.height &&
		    requested.extent.depth >= cached.info.extent.depth &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format)) {
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
			return {ExpandImage(requested, cached_id)};
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
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapMipMerge);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		if (KeepOverlappedImage(cached, current_frame)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TextureOverlapKeeps);
		} else {
			FreeImage(cached_id, HangTrace::ImageFreeReason::OverlapStale);
		}
	}
	return {merged_id};
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
	expanded.usage         = source.usage;
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
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
	FreeImage(source_id, HangTrace::ImageFreeReason::Expand);
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
	transfer.swap_bgra16 = info.bgra16 && (!upload || render_target || video_out);
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
		if (!transfer.tiles.empty() && !transfer.swap_bgra16 &&
		    m_tiler.DetileToImage(destination, source.Handle(), source_offset, image.live.size,
		                          transfer.LinearSize(), transfer.tiles, transfer.regions)) {
			return;
		}
		TileManager::Result linear {source.Handle(), source_offset, image.live.size};
		if (!transfer.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, image.live.size,
			                        transfer.LinearSize(), transfer.tiles);
		}
		if (transfer.swap_bgra16) {
			linear = m_tiler.SwapBgra16(linear);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
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
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
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
}

void TextureCache::InitializeImage(ImageId id, RefreshIntent intent) {
	auto& image = m_slot_images[id];
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
	bool       guest_refresh = false;
	if (upload && AliasBytesEnabled() && !image.depth_id) {
		// The rebuild reads buffer (or guest) bytes over the whole image. Bytes it still owns
		// itself (a bounded buffer write took the others) reach the buffer first; a read also
		// takes the bytes other images own there (a write binding usually overwrites them).
		if (image.IsBufferModified() && !image.IsDefinitelyCpuDirty() &&
		    image.OwnedSet() != nullptr && !image.OwnsNoBytes()) {
			(void)MaterializeOwnedBytes(image.live, id, {}, "rebuild-own");
		}
		if (intent == RefreshIntent::Read) {
			(void)MaterializeOwnedBytes(image.live, {}, id, "rebuild");
		}
	}
	if (upload) {
		Profiler::ScopedFrameWait upload_time(Profiler::FrameWait::TextureUpload);
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
		if (image.ChunkTracked() && image.chunks.dirty_count != 0 && !first) {
			partial = TryPartialUpload(image);
			if (!partial.done) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::TexturePartialFallbacks);
			}
		}
		// Sources cover the resident prefix only (image.live; the whole range when fully
		// resident): the bytes of levels no view can sample are neither read nor uploaded.
		const auto live = image.live;
		if (!partial.done && !TryAsyncFullUpload(image)) {
			const bool cached_source = m_buffer_cache.IsRegionRegistered(live.address, live.size) ||
			                           m_buffer_cache.IsRegionGpuModified(live.address, live.size);
			Profiler::CountFrameEvent(cached_source ? Profiler::FrameEvent::TextureUploadBytesBuffer
			                                        : Profiler::FrameEvent::TextureUploadBytesStaging,
			                          live.size);
			const auto [source, source_offset] =
			    m_buffer_cache.ObtainBufferForImage(live.address, live.size);
			if (source == nullptr) {
				EXIT("TextureCache: failed to obtain image upload source\n");
			}
			UploadImage(image, *source, source_offset);
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
			                          partial.done ? "cpu-write-partial" : reason,
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
	if (!transfer.valid || transfer.swap_bgra16 || transfer.tiles.empty() ||
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
				    !LibKernel::Memory::TryReadPrtBacking(base + range.offset,
				                                          mapped + range.packed, range.size)) {
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
	if (!transfer.valid || transfer.swap_bgra16 || transfer.tiles.empty()) {
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
		     !LibKernel::Memory::TryReadPrtBacking(begin, bytes.data(), end - begin))) {
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
		     !LibKernel::Memory::TryReadPrtBacking(begin, bytes.data(), end - begin))) {
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

Profiler::FrameEvent TextureCache::TryMaterializeGpuDccClear(ImageId id, const ImageDesc& desc,
                                                              uint32_t metadata_base_layer) {
	using Event = Profiler::FrameEvent;
	if (!m_dcc_clear || !m_dcc_clear->Available()) {
		return Event::DccFallbackDisabled;
	}
	// The video-out path does not consume its clear key; keep it on the CPU inspection.
	if (desc.type == BindingType::VideoOut) {
		return Event::DccFallbackBinding;
	}
	if (desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
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
	                                      metadata_base_layer, values, decodable, false);
}

Profiler::FrameEvent TextureCache::TryMaterializeGpuMetadataClear(
    DccClearHelper& helper, ImageId id, const ImageDesc& desc, GuestRange range,
    uint32_t metadata_base_layer, const DccClearHelper::ClearValues& values, uint32_t decodable,
    bool cmask) {
	using Event = Profiler::FrameEvent;
	using Support = DccClearHelper::Support;
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
		if (!FindImagesInRegion(range.address, range.size, false).empty()) {
			return Event::DccFallbackMetadataAliased;
		}
		// A retained inspection of an unchanged slice leaves consumed keys (0xFF), nonuniform
		// bytes or a code its interpretation rejected. An interpretation accepting no more codes
		// than that one reads the same bytes to the same no-op, for any image or layer; only the
		// metadata processing is skipped, the caller still refreshes this image normally.
		bool reusable = true;
		for (uint32_t slice = 0; reusable && slice < count; ++slice) {
			const auto metadata = slice_range(slice);
			const auto revision = m_buffer_cache.GetContentRevision(metadata.address, metadata.size);
			const auto found    = m_gpu_dcc_inspections.find(metadata.address);
			reusable = revision && found != m_gpu_dcc_inspections.end() &&
			           found->second.metadata == metadata && found->second.revision == *revision &&
			           (decodable & ~found->second.decodable_mask) == 0;
		}
		if (reusable) {
			if (!cmask) {
				m_gpu_dcc_reuses += count;
				Profiler::CountFrameEvent(Event::DccGpuReuses, count);
			}
			return Event::DccGpuReuses;
		}
	}
	if (ImageRangeOverlaps(range, desc.info.data)) {
		return Event::DccFallbackMetadataAliased;
	}
	// KYTY_DCC_GPU_REFRESH (textureCache.h): a DCC image that is registered, matches the view and
	// is fully resident is accepted while waiting for a refresh or not GPU-owned; it is refreshed
	// right before the inspection, as the CPU fallback's ClearImage refreshes it.
	const bool refresh_allowed = m_dcc_gpu_refresh && !cmask;
	const auto eligible = [&]() -> std::optional<Event> {
		const auto* image = m_slot_images.try_get(id);
		const bool  matches = image != nullptr && image->registered && !image->depth_id &&
		                     image->info.data == desc.info.data &&
		                     image->info.extent == desc.info.extent;
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
		if (image->backing.extent.width != desc.info.extent.width ||
		    image->backing.extent.height != desc.info.extent.height ||
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
		if (!FindImagesInRegion(range.address, range.size, false).empty()) {
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
	// Use the canonical buffer and its normal GPU-write tracking. No CPU shadow of the
	// clear value is trusted. Acquiring it may upload/submit, so hold no texture lock.
	auto [buffer, offset] = m_buffer_cache.ObtainBuffer(range.address, range.size, true, false);
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
		for (uint32_t slice = 0; slice < count; ++slice) {
			helper.RecordSlice(image, view.format, view.base_layer + slice, buffer->Handle(),
			                         offset + slice_size * (first + slice), slice_size, values);
		}
	}
	CommitGpuWrite(image);
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
	for (uint32_t slice = 0; slice < count; ++slice) {
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
	{
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// A native DCC allocation must not retain a reused HTile/CMask/FMask interpretation.
		EraseSurfaceMeta(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1 || image.info.resources.levels != 1) {
			// Decided by the description and the image's level count (fixed for its lifetime).
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
		if (known && unaliased && *known == known_byte * 0x01010101u) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DccKnownFillClears);
			vk::ClearValue clear {};
			if (!DecodeDccClear(desc, static_cast<uint8_t>(known_byte), clear.color)) {
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
				{
					std::scoped_lock lock {m_lock};
					ClearImage(m_scheduler.Current(), id, view.format,
					           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
					            image_first + slice, 1}, clear);
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
			if (diagnostic_readback != 0) {
				TraceDccDiagnostic(
				    "DCC_SLICE id=%" PRIu64 " slice=%u address=0x%" PRIx64 " bytes=%" PRIu64
				    " code=0x%02x decoded=1 uniform=0 consumed=0",
				    diagnostic_readback, first + slice, address, slice_size, unsigned(code));
			}
			continue;
		}
		{
			KYTY_PROFILER_DETAIL_BLOCK("DCC::ClearImage");
			++m_dcc_decision_effects;
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
		}
		// Native expanded keys own consumption. Existing buffer tracking publishes this CPU
		// write to future GPU readers; FillBuffer can fault and must run outside the texture lock.
		if (desc.type != BindingType::VideoOut) {
			KYTY_PROFILER_DETAIL_BLOCK("DCC::ConsumeClearKey");
			m_buffer_cache.FillBuffer(address, slice_size, UINT32_MAX, false);
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
	const auto  range  = desc.cmask.range;
	const auto& view   = desc.view_info;
	const auto  layers = desc.info.TransferLayers();
	if (!range.Valid() || layers == 0 || range.size % layers != 0 || (range.size / layers) % 4 != 0 ||
	    desc.info.resources.levels != 1 || desc.info.IsVolume() || desc.info.samples != 1 ||
	    view.layer_count == 0 || metadata_base_layer >= layers ||
	    view.layer_count > layers - metadata_base_layer) {
		Profiler::CountFrameEvent(Event::CmaskFastClearShape);
		description_noop();
		return;
	}
	vk::ClearValue clear {};
	if (!DecodePackedColorClear64(view.format, desc.cmask.clear_word0, desc.cmask.clear_word1,
	                              clear.color)) {
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
			if (!FindImagesInRegion(metadata.address, metadata.size, false).empty()) {
				Profiler::CountFrameEvent(Event::CmaskFastClearAliased);
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
			// 0xFFFFFFFF: every tile expanded (no pending clear). Anything else: some tiles are not
			// fast-cleared (or the bytes encode nothing a single-sample surface uses).
			Profiler::CountFrameEvent(value && *value == UINT32_MAX ? Event::CmaskFastClearExpanded
			                                                        : Event::CmaskFastClearUnproven);
			continue;
		}
		provable = false;
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            view.base_layer + slice, 1},
			           clear);
		}
		// After the eliminate every tile is expanded; this also keeps a later binding from
		// clearing again what the draws wrote meanwhile. FillBuffer can fault: no texture lock.
		m_buffer_cache.FillBuffer(metadata.address, metadata.size, UINT32_MAX, false);
		Profiler::CountFrameEvent(Event::CmaskFastClears);
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
	static const bool native_enabled = [] {
		const auto* value = std::getenv("KYTY_CMASK_NATIVE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
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
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            view.base_layer + slice, 1},
			           clear);
		}
		m_buffer_cache.FillBuffer(metadata.address, metadata.size, UINT32_MAX, false);
		Profiler::CountFrameEvent(Event::CmaskFastClears);
	}
}

// TextureBindingMemo::RefreshIsNoOp mirrors when this is a no-op; keep them in sync.
void TextureCache::RefreshImage(ImageId id, RefreshIntent intent) {
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
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

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
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
	}
	record.depth_id = depth_id;
	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format, RepeatLookup* record) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	if (record != nullptr) {
		*record = {};
	}
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
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
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result, HangTrace::ImageFreeReason::SmallerResources);
				result = {};
			}
		}
		if (!result) {
			first_page_answer = false;
			// A prefix computed as not applicable (0) keeps the image fully resident; InsertImage
			// computes it when this lookup did not.
			result = InsertImage(desc.info,
			                     prefix_computed && wanted_prefix == 0 ? 0u : wanted_first,
			                     wanted_prefix);
			auto& inserted = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted.live.address, inserted.live.size) ||
			    (AliasBytesEnabled() && OtherImagesOwnBytes(inserted.live, result))) {
				inserted.MarkBufferModified();
			}
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		// Every level the (adjusted) view can sample must be resident: a finer MIN_LOD than
		// before extends the image, any non-sampling binding makes the whole chain resident.
		EnsureResidency(result, RequestedFirstLevel(desc, image.info.resources.levels),
		                desc.type == BindingType::Texture);
		SyncAliasFromOwner(result);
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
	    (record.has_partner && !SyncAliasReturnsAtOnce(*image)) ||
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

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid,
                                         bool buffer_sync) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	// Only images starting at address can match below. Their live prefix starts there too,
	// so every candidate is indexed on its first page; size still selects the exact match.
	for (const auto id: FindImagesInRegion(address, 1, false)) {
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

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
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
	if (desc.type == BindingType::Storage) {
		MarkImageGpuModified(image);
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id, desc.type == BindingType::Storage ? RefreshIntent::Write
		                                                   : RefreshIntent::Read);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
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
	MarkImageGpuModified(image);
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
	MarkImageGpuModified(image);
	image.usage.depth_target = true;
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		const MetaDataInfo entry {.type       = MetaDataInfo::Type::HTile,
		                          .clear_mask = image.info.htile_clear_mask};
		bool inserted = false;
		if (CpCommit::Enabled(CpCommit::Part::TargetAlloc)) {
			// Inserts exactly when emplace would (the key is absent), without building and freeing
			// a node every draw when it is present.
			inserted = m_surface_metas.try_emplace(desc.info.metadata.range.address, entry).second;
		} else {
			inserted = m_surface_metas.emplace(desc.info.metadata.range.address, entry).second;
		}
		// The only insertion into m_surface_metas (EraseSurfaceMeta). Every draw with this target
		// asks again; only an added entry is a change (KYTY_META_CLEAR_MEMO's generation).
		m_surface_meta_inserts += inserted ? 1u : 0u;
		if (inserted) {
			NoteSurfaceMetaChange();
		}
	}
	RefreshImage(id, RefreshIntent::Write);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil), RefreshIntent::Write);
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

bool TextureCache::SyncAliasReturnsAtOnce(const Image& image) {
	return !AliasAgeByFrames() || image.alias_owner || image.depth_id || image.info.data.Empty() ||
	       image.info.HasStencil() || image.backing.image == nullptr;
}

bool TextureCache::HasAliasPartner(uint64_t page, ImageId found, const Image& image) const {
	bool partner = false;
	if (const auto* owners = m_image_page_table.Find(page); owners != nullptr) {
		owners->ForEach([&](ImageId id) {
			const auto* other = m_slot_images.try_get(id);
			if (id != found && other != nullptr && other->registered &&
			    other->info.data == image.info.data && other->info.extent == image.info.extent &&
			    other->backing.samples == image.backing.samples) {
				partner = true;
			}
		});
	}
	return partner;
}

void TextureCache::SyncAliasFromOwner(ImageId id) {
	auto& image = m_slot_images[id];
	// TextureCache::SyncAliasReturnsAtOnce mirrors this condition.
	if (!AliasAgeByFrames() || image.alias_owner || image.depth_id || image.info.data.Empty() ||
	    image.info.HasStencil() || image.backing.image == nullptr) {
		return;
	}
	// A kept alias (same memory, another format) is stale when another interpretation wrote the
	// memory since. Reinterpret the owner's contents, as the recreate path used to, and take over.
	for (const auto other_id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
		const auto* other = m_slot_images.try_get(other_id);
		if (other_id == id || other == nullptr || !other->registered || other->depth_id ||
		    !other->alias_owner || !other->IsGpuModified() || other->IsBufferModified() ||
		    other->info.HasStencil() || other->info.data != image.info.data ||
		    other->info.extent != image.info.extent ||
		    other->backing.samples != image.backing.samples) {
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
		const bool lossless = CopyImage(id, other_id, "alias-sync");
		CommitGpuWrite(image);
		if (lossless && AliasSyncSkipEnabled()) {
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

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	KYTY_GPU_OP_SITE("texcache.clear_from_buffer");
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock     lock {m_lock};
	ImageId              selected {};
	vk::ImageAspectFlags aspect {};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate    = vk::ImageAspectFlagBits::eStencil;
			candidate_id = owner->depth_id;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && selected != candidate_id) {
			return false;
		}
		selected = candidate_id;
		aspect   = candidate;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
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
	ClearImage(command, selected, image.backing.format,
	           {aspect, 0, image.info.resources.levels, 0, image.info.TransferLayers()}, clear);
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	KYTY_GPU_OP_SITE("texcache.clear");
	// Levels outside the cleared range must hold guest data, and the result is GPU-owned.
	RequireFullResidency(id);
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
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
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Handle());
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
		command.Handle().beginRendering(&rendering);
		command.Handle().endRendering();
		image.NoteContentWrite();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Handle());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Handle().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Handle().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	image.NoteContentWrite();
	CommitGpuWrite(image);
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
}

void TextureCache::DownloadDepth(Image& image, vk::Buffer destination, uint64_t destination_offset,
                                 uint64_t destination_capacity) {
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
	const auto transform = texture.swap_bgra16 ? TileManager::ColorTransform::SwapBgra16
	                                           : TileManager::ColorTransform::None;
	if (texture.tiles.empty()) {
		if (transform == TileManager::ColorTransform::SwapBgra16) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.SwapBgra16(linear, {destination, destination_offset, destination_size});
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination, destination_offset, destination_size);
		return;
	}

	// KYTY_TILER_IMAGE_DIRECT: tile straight from the image, no image->buffer copy.
	if (transform == TileManager::ColorTransform::None &&
	    m_tiler.TileFromImage(image, texture.regions, destination, destination_offset,
	                          destination_size, texture.LinearSize(), texture.tiles)) {
		return;
	}
	m_tiler.TileImage(image, texture.regions, destination, destination_offset, destination_size,
	                  texture.LinearSize(), texture.tiles, transform);
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
			const auto              command = m_scheduler.Current().Handle();
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
			// Handle() again: the tile pass may have begun a new command buffer.
			m_scheduler.EndRendering();
			const auto              command = m_scheduler.Current().Handle();
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

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size, true, true);
	const bool synced   = selected && SynchronizeBufferFromOwner(buffer, selected);
	// KYTY_ALIAS_BYTES: every other byte of the range that an image owns (an owner that does not
	// start at vaddr, owns only part of its bytes, or lies past the image synchronized above) is
	// read from the buffer too: move it there. Only into the cache buffer that holds the range
	// (whose GPU-dirty bytes the cache tracks).
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (!TextureCache::AliasBytesEnabled() || owner == nullptr || !*owner ||
	    &m_slot_buffers[*owner] != &buffer) {
		return synced;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesTexelReads);
	if (!m_texture_cache.MayOverlapImages(vaddr, size)) {
		return synced;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::AliasBytesTexelScans);
	std::scoped_lock lock {m_texture_cache.m_lock};
	const auto       moved = m_texture_cache.MaterializeOwnedBytes(
        {vaddr, size}, {}, synced ? selected : ImageId {}, "texel-read", &buffer);
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
			return true;
		}
		image_address = image.info.data.address;
		Profiler::CountFrameEvent(Profiler::FrameEvent::TexelImageSyncDownloads);
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
	if (image.info.IsVolume()) {
		// Volume mips contain strided block slices, so a mip's linear span cannot prove that
		// every retained slice fits. Keep volume synchronization whole-image only.
		if (!buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
			return 0;
		}
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
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
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_WRITEBACK_ON_GPU_WRITE");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
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

bool TextureCache::DownloadImageMemory(ImageId id) {
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
	if (mapped == nullptr) {
		EXIT("TextureCache: failed to map reusable download buffer\n");
	}
	download.Commit();
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	download.Flush(offset, range.size);

	DownloadImage(image, download, offset, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download.Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	const auto publication = m_buffer_cache.BeginBackingPublication(
	    std::span<const GuestRange>(&range, 1), m_scheduler.CurrentTick());
	m_scheduler.DeferPriorityOperation([this, &download, range, mapped, offset, publication] {
		download.Invalidate(offset, range.size);
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
		m_buffer_cache.EndBackingPublication(publication);
	});
	return true;
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
		image.MarkBufferModified();
		image.NoteDirtySpan(address, size);
	}
	if (m_exact_range_mode != 0) {
		PrintExactRangeSummary();
	}
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
	// Every GPU writer (draw targets, storage bindings, clears, helper passes, copies) passes
	// here before or right after recording its write: the native contents get a new identity.
	image.NotePossibleWrite();
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
	if (image != nullptr && image->DefiniteWrites() == mark.definite_writes) {
		image->AdoptContentSerial(mark.serial);
	}
}

bool TextureCache::ReleaseCpuOverwrittenImages(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return false;
	}
	std::scoped_lock lock {m_lock};
	bool released = false;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto& image = m_slot_images[id];
		// A maybe-dirty image may still hold the only copy of its bytes: only a CPU write into the
		// image's own bytes (InvalidateCpuWrite) releases it. Same rule as the alias
		// materialization and SafeToDownload, which never take a CPU-dirty image's contents.
		if (image.depth_id || !image.IsGpuModified() || !image.IsDefinitelyCpuDirty()) {
			continue;
		}
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("TextureCache: released CPU-overwritten GPU image 0x%016" PRIx64 "+0x%" PRIx64
			            " (%s %ux%u) for a guest read at 0x%016" PRIx64 "\n",
			            image.info.data.address, image.info.data.size,
			            vk::to_string(image.info.pixel_format).c_str(), image.info.extent.width,
			            image.info.extent.height, address);
		}
		InvalidateCleanImageProofs(image.live.address, image.live.size,
		                           Coherence::Source::ImageGpuClear);
		image.ClearGpuModified();
		released = true;
	}
	return released;
}

void TextureCache::LogGpuModifiedImages(uint64_t address, uint64_t size) {
	static std::atomic<uint32_t> logged {0};
	if (!GuestRange {address, size}.Valid() || logged.fetch_add(1, std::memory_order_relaxed) >= 32) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, false)) {
		const auto& image = m_slot_images[id];
		if (image.depth_id || !image.IsGpuModified()) {
			continue;
		}
		const auto& info = image.info;
		std::fprintf(stderr,
		             "  gpu-modified image id=%u live=0x%016" PRIx64 "+0x%" PRIx64 " data=0x%016" PRIx64
		             "+0x%" PRIx64 " %s %ux%ux%u tiled=%d metadata=%d owns_all=%d buffer_modified=%d"
		             " cpu_dirty=%d alias_owner=%d tick=%" PRIu64 " frame=%" PRIu64 "\n",
		             static_cast<uint32_t>(id.index), image.live.address, image.live.size,
		             info.data.address, info.data.size, vk::to_string(info.pixel_format).c_str(),
		             info.extent.width, info.extent.height, info.extent.depth, info.IsTiled() ? 1 : 0,
		             static_cast<int>(info.metadata.kind), image.OwnsAllBytes() ? 1 : 0,
		             image.IsBufferModified() ? 1 : 0, image.IsCpuDirty() ? 1 : 0,
		             image.alias_owner ? 1 : 0, image.tick_accessed_last, image.frame_accessed_last);
	}
	std::fflush(stderr);
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

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	if (MetaClearMemoEnabled()) {
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
			        .clear_mask = found != m_surface_metas.end() ? found->second.clear_mask : 0u,
			        .found      = found != m_surface_metas.end(),
			        .valid      = true};
		}
		return memo.found && slice < 32 && (memo.clear_mask & (1u << slice)) != 0;
	}
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	NoteSurfaceMetaChange();
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	const auto previous = found->second.clear_mask;
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	if (found->second.clear_mask != previous) {
		NoteSurfaceMetaChange();
	}
	return true;
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	m_gpu_dcc_inspections.clear();
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	NoteSurfaceMetaChange();
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
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

bool TextureCache::IdleRetirable(const Image& image, uint64_t frame, uint64_t min_age) const {
	// What the submission-age collector frees without a download: clean of GPU writes (the guest
	// bytes or the buffer cache hold the contents, a later use recreates and refreshes it), and
	// not a stencil association, a bound resource, a render target or a presentation surface.
	return image.registered && !image.depth_id && !image.IsGpuModified() &&
	       !image.binding.is_bound && !image.binding.is_target && !image.usage.video_out &&
	       frame - std::min(frame, image.frame_accessed_last) > min_age;
}

void TextureCache::NoteFrameTick(uint64_t frame, uint64_t tick) {
	if (m_frame_ticks.empty() || m_frame_ticks.back().first != frame) {
		m_frame_ticks.emplace_back(frame, tick);
	}
	const auto keep = std::max(m_idle_frames, m_pressure_frames) + 64;
	while (m_frame_ticks.size() > keep) {
		m_frame_ticks.pop_front();
	}
}

uint64_t TextureCache::RetireUnusedImages(uint64_t frame, uint64_t min_age, size_t max_count,
                                          uint64_t enough) {
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
		if (image != nullptr && IdleRetirable(*image, frame, min_age)) {
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
		FreeImage(id, HangTrace::ImageFreeReason::GarbageCollect);
		++m_idle_freed;
	}
	return freed;
}

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	RetireIdlePartialImages();
	const uint64_t   tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	const auto frame = m_frame.load(std::memory_order_relaxed);
	if (m_idle_frames != 0 || m_pressure_frames != 0) {
		NoteFrameTick(frame, tick);
	}
	if (m_idle_frames != 0 && frame >= m_idle_next_frame) {
		m_idle_next_frame = frame + 32;
		(void)RetireUnusedImages(frame, m_idle_frames, 512, UINT64_MAX);
	}
	m_tiler.TrimScratchPool();
	if ((tick & 63u) == 0 && Profiler::DetailedEnabled() && tracy::ProfilerAvailable()) {
		if (m_graphics.CanReportMemoryUsage()) {
			TracyPlot("ImageCache.DeviceMemoryBytes", static_cast<double>(m_total_used_memory));
		}
		TracyPlot("ImageCache.RegisteredGuestBytes", static_cast<double>(m_registered_image_memory));
		TracyPlot("ImageCache.PendingRetirementBytes", static_cast<double>(m_pressure_retirement_bytes));
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
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(tick - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() == deletions;
		});
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			--deletions;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered || owner->depth_id) {
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = SafeToDownload(*owner);
				if (safe && owner->info.IsTiled()) {
					continue;
				}
				if (safe && !pressured) {
					continue;
				}
				if (safe && !DownloadImageMemory(id)) {
					continue;
				}
			}
			FreeImage(id, HangTrace::ImageFreeReason::GarbageCollect);
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
		if (owner == nullptr || !owner->registered || owner->depth_id) {
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
	VramStats::Line("GC: %s, usage %.1f MiB, trigger %.1f pressure %.1f critical %.1f MiB, registered "
	                "guest bytes %.1f MiB, pending retirement %.1f MiB, GC tick %llu | tiler scratch "
	                "pool %.1f of %.1f MiB idle",
	                m_pressure_gc_enabled ? "pressure policy" : "submission age", ToMiB(m_total_used_memory),
	                ToMiB(m_trigger_gc_memory), ToMiB(m_pressure_gc_memory), ToMiB(m_critical_gc_memory),
	                ToMiB(m_registered_image_memory), ToMiB(m_pressure_retirement_bytes),
	                static_cast<unsigned long long>(m_gc_tick), ToMiB(scratch_idle), ToMiB(scratch_limit));
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
