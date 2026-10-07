#include "graphics/host_gpu/renderer/colorRenderTarget.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

static std::atomic<uint32_t> g_render_color_log_count = 0;

bool TargetDescMemoEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_TARGET_DESC_MEMO");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_CMASK_FAST_CLEAR=0 ignores CMASK fast clears as before: the target keeps its previous
// contents wherever the game fast-cleared it.
bool CmaskFastClearEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CMASK_FAST_CLEAR");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool DrawSequenceEnabled(DrawSequencePart part) {
	static const uint32_t parts = [] {
		constexpr uint32_t all = static_cast<uint32_t>(DrawSequencePart::Targets) |
		                         static_cast<uint32_t>(DrawSequencePart::Textures);
		const auto* value = std::getenv("KYTY_DRAW_SEQUENCE_FAST");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) {
			return all;
		}
		if (std::strcmp(value, "0") == 0) {
			return 0u;
		}
		uint32_t selected = 0;
		if (std::strstr(value, "targets") != nullptr) {
			selected |= static_cast<uint32_t>(DrawSequencePart::Targets);
		}
		if (std::strstr(value, "textures") != nullptr) {
			selected |= static_cast<uint32_t>(DrawSequencePart::Textures);
		}
		return selected;
	}();
	return (parts & static_cast<uint32_t>(part)) != 0;
}

int DrawSequenceVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_DRAW_SEQUENCE_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

void ReportDrawSequenceMismatch(const char* what) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr, "DrawSequenceVerify: %s differs from the full path\n", what);
	}
	if (DrawSequenceVerifyMode() == 2) {
		EXIT("DrawSequenceVerify: %s differs from the full path\n", what);
	}
}

bool RenderStateFastEnabled(RenderStatePart part) {
	static const uint32_t parts = [] {
		constexpr uint32_t all = static_cast<uint32_t>(RenderStatePart::VertexCopy) |
		                         static_cast<uint32_t>(RenderStatePart::PrepSwap) |
		                         static_cast<uint32_t>(RenderStatePart::Reset);
		const auto* value = std::getenv("KYTY_RENDER_STATE_FAST");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "1") == 0) {
			return all;
		}
		if (std::strcmp(value, "0") == 0) {
			return 0u;
		}
		uint32_t selected = 0;
		if (std::strstr(value, "vertex") != nullptr) {
			selected |= static_cast<uint32_t>(RenderStatePart::VertexCopy);
		}
		if (std::strstr(value, "swap") != nullptr) {
			selected |= static_cast<uint32_t>(RenderStatePart::PrepSwap);
		}
		if (std::strstr(value, "reset") != nullptr) {
			selected |= static_cast<uint32_t>(RenderStatePart::Reset);
		}
		return selected;
	}();
	return (parts & static_cast<uint32_t>(part)) != 0;
}

int RenderStateVerifyMode() {
	static const int mode = [] {
		const auto* value = std::getenv("KYTY_RENDER_STATE_VERIFY");
		if (value == nullptr || *value == '\0' || std::strcmp(value, "0") == 0) {
			return 0;
		}
		return std::strcmp(value, "exit") == 0 ? 2 : 1;
	}();
	return mode;
}

void ReportRenderStateMismatch(const char* what) {
	Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyMismatches);
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
		std::fprintf(stderr, "RenderStateVerify: %s differs from the full one\n", what);
	}
	if (RenderStateVerifyMode() == 2) {
		EXIT("RenderStateVerify: %s differs from the full one\n", what);
	}
}

bool SameImageDesc(const TextureCache::ImageDesc& a, const TextureCache::ImageDesc& b) {
	// The bindings name every field in declaration order: a new field stops this compiling until
	// it is compared below.
	const auto& [a_info, a_view, a_type, a_cmask] = a;
	const auto& [b_info, b_view, b_type, b_cmask] = b;
	const auto& [a_data, a_stencil, a_meta, a_htile_clear_mask, a_pixel_format, a_guest_format,
	             a_image_type, a_extent, a_resources, a_pitch, a_bytes_per_block, a_samples,
	             a_tile_mode, a_bgra16, a_mip_layout] = a_info;
	const auto& [b_data, b_stencil, b_meta, b_htile_clear_mask, b_pixel_format, b_guest_format,
	             b_image_type, b_extent, b_resources, b_pitch, b_bytes_per_block, b_samples,
	             b_tile_mode, b_bgra16, b_mip_layout] = b_info;
	const auto& [a_range, a_kind, a_control, a_dcc_clear_word, a_dcc_clear_word1, a_compression,
	             a_stencil_compressed, a_dcc_clear_register_valid, a_dcc_alpha_msb] = a_meta;
	const auto& [b_range, b_kind, b_control, b_dcc_clear_word, b_dcc_clear_word1, b_compression,
	             b_stencil_compressed, b_dcc_clear_register_valid, b_dcc_alpha_msb] = b_meta;
	const auto& [a_cmask_range, a_clear_word0, a_clear_word1, a_cmask_valid] = a_cmask;
	const auto& [b_cmask_range, b_clear_word0, b_clear_word1, b_cmask_valid] = b_cmask;
	// ImageViewInfo::operator== compares all of its fields.
	return a_data == b_data && a_stencil == b_stencil && a_range == b_range && a_kind == b_kind &&
	       a_control == b_control && a_dcc_clear_word == b_dcc_clear_word &&
	       a_dcc_clear_word1 == b_dcc_clear_word1 &&
	       a_compression == b_compression && a_stencil_compressed == b_stencil_compressed &&
	       a_dcc_clear_register_valid == b_dcc_clear_register_valid &&
	       a_dcc_alpha_msb == b_dcc_alpha_msb && a_htile_clear_mask == b_htile_clear_mask &&
	       a_pixel_format == b_pixel_format && a_guest_format == b_guest_format &&
	       a_image_type == b_image_type && a_extent == b_extent && a_resources == b_resources &&
	       a_pitch == b_pitch && a_bytes_per_block == b_bytes_per_block &&
	       a_samples == b_samples && a_tile_mode == b_tile_mode && a_bgra16 == b_bgra16 &&
	       a_mip_layout == b_mip_layout && a_view == b_view && a_type == b_type &&
	       a_cmask_range == b_cmask_range && a_clear_word0 == b_clear_word0 &&
	       a_clear_word1 == b_clear_word1 && a_cmask_valid == b_cmask_valid;
}

bool SameRenderColorInfo(const RenderColorInfo& a, const RenderColorInfo& b) {
	// Every field, named in declaration order (see ResolveRenderColorTarget).
	const auto& [a_desc, a_image, a_slot, a_mip, a_layer, a_mapping] = a;
	const auto& [b_desc, b_image, b_slot, b_mip, b_layer, b_mapping] = b;
	return SameImageDesc(a_desc, b_desc) && a_image == b_image && a_slot == b_slot &&
	       a_mip == b_mip && a_layer == b_layer && a_mapping == b_mapping;
}

ImageId RenderExecutor::FindTargetImage(TextureCache::ImageDesc& desc, bool exact_format,
                                        TextureCache::RepeatLookup* record) {
	auto& cache = m_context.GetTextureCache();
	if (record == nullptr || !DrawSequenceEnabled(DrawSequencePart::Targets)) {
		return cache.FindImage(desc, exact_format);
	}
	auto&      totals = m_draw_sequence_totals;
	const auto full   = [&] {
		const auto id = cache.FindImage(desc, exact_format, record);
		if (record->valid) {
			totals.target_records++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTargetRecords);
		}
		return id;
	};
	if (!record->valid) {
		return full();
	}
	const bool verify  = DrawSequenceVerifyMode() != 0;
	// Read before the check: a guest write fault changing the recorded fills after it (on another
	// thread) races the lookup (verify below).
	const auto foreign = m_context.GetBufferCache().ForeignKnownFillChanges();
	if (!cache.TryRepeatLookup(desc, exact_format, *record, !verify)) {
		totals.target_misses++;
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTargetMisses);
		return full();
	}
	totals.target_repeats++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceTargetRepeats);
	if (!verify) {
		return record->image;
	}
	// KYTY_DRAW_SEQUENCE_VERIFY: the full lookup must return the same image and do nothing but its
	// bookkeeping: no residency extension or alias synchronization, and metadata decisions that
	// are again provable no-ops. It provides the result (and the next record).
	totals.verify_checks++;
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceVerifyChecks);
	const auto claimed = *record; // the full lookup records again
	const auto effects = cache.LookupSideEffects();
	const auto id      = full();
	if (id != claimed.image || cache.LookupSideEffects() != effects || !record->valid) {
		if (m_context.GetBufferCache().ForeignKnownFillChanges() != foreign ||
		    cache.RepeatGpuRangesLost(claimed)) {
			// Another thread forgot a recorded fill or ended GPU ownership of the metadata
			// meanwhile: the full lookup decided on other metadata than the check.
			totals.verify_races++;
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawSequenceVerifyRaces);
		} else {
			totals.verify_mismatches++;
			ReportDrawSequenceMismatch("a repeated target lookup");
		}
	}
	return id;
}

static bool DccAlphaOnMsb(const HW::ColorInfo& info) {
	switch (info.format) {
		case Prospero::ChannelLayout::k10_10_10_2:
		case Prospero::ChannelLayout::k10_10_10_2Float:
		case Prospero::ChannelLayout::k5_5_5_1: return true;
		case Prospero::ChannelLayout::k2_10_10_10:
		case Prospero::ChannelLayout::k1_5_5_5: return false;
		default: break;
	}
	const auto components =
	    Prospero::ResolveRenderTargetFormat(info.format, info.channel_type).components;
	if (components == 1) {
		return info.channel_order != Prospero::ChannelOrder::kStandard;
	}
	return components == 3 || info.channel_order == Prospero::ChannelOrder::kStandard ||
	       info.channel_order == Prospero::ChannelOrder::kAlt;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& r,
                                              uint32_t         render_target_slice_offset,
                                              uint32_t rt_slot, bool ignore_target_mask,
                                              bool exact_format) {
	KYTY_PROFILER_FUNCTION();
	const auto& hw = buffer.GetRegisters();

	const auto& rt      = hw.GetRenderTarget(rt_slot);
	auto        mask    = render_target_mask_slot(hw.GetRenderTargetMask(), rt_slot);
	if (ignore_target_mask && rt.base.addr != 0 && mask == 0) {
		mask = 0x0f;
	}

	// KYTY_RENDER_STATE_FAST reset: both paths below that resolve a target assign every field of
	// the entry (desc, image_id, target_slot, guest_mip_level, guest_array_layer, export_mapping;
	// SameRenderColorInfo names them all), so only an entry without a target is value-initialised.
	const bool output = rt.base.addr != 0 && mask != 0;
	if (!output || !RenderStateFastEnabled(RenderStatePart::Reset)) {
		r = {};
	}
	r.target_slot = rt_slot;

	if (!output) {
		if (graphics_debug_dump_enabled()) {
			static std::atomic_uint log_count = 0;
			const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
			if (log_id < 128) {
				LOGF("RenderColorTarget: no color output slot=%" PRIu32 " base=0x%010" PRIx64
				     " slot_mask=0x%01" PRIx32 " target_mask=0x%08" PRIx32
				     " rt_slice_offset=%" PRIu32 "\n",
				     rt_slot, rt.base.addr, mask, hw.GetRenderTargetMask(),
				     render_target_slice_offset);
			}
		}

		return;
	}
	// KYTY_TARGET_DESC_MEMO: the description below is a pure function of these register bytes,
	// the effective mask and the slice offset. It is bypassed while decision logging is still
	// active so that the first logged decisions stay identical.
	auto* memo = rt_slot < m_color_target_memo.size() && TargetDescMemoEnabled() &&
	                     !graphics_debug_dump_enabled() &&
	                     g_render_color_log_count.load(std::memory_order_relaxed) >= 128
	                 ? &m_color_target_memo[rt_slot]
	                 : nullptr;
	if (memo != nullptr) {
		if (memo->valid && memo->mask == mask && memo->slice_offset == render_target_slice_offset &&
		    std::memcmp(&memo->registers, &rt, sizeof(rt)) == 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TargetDescMemoHits);
			r.desc              = memo->desc;
			r.guest_mip_level   = memo->guest_mip_level;
			r.guest_array_layer = memo->guest_array_layer;
			r.export_mapping    = memo->export_mapping;
			r.image_id          = FindTargetImage(r.desc, exact_format, &memo->lookup);
			BindRenderTarget(r.image_id);
			return;
		}
		Profiler::CountFrameEvent(Profiler::FrameEvent::TargetDescMemoMisses);
	}
	const auto samples = render_sample_count(rt.attrib.num_fragments);
	if (samples == 0 || rt.attrib.num_samples != rt.attrib.num_fragments) {
		EXIT("unsupported render-target sample configuration: samples=%u fragments=%u\n",
		     rt.attrib.num_samples, rt.attrib.num_fragments);
	}
	const uint32_t levels = rt.attrib2.num_mip_levels + 1u;
	if (levels == 0 || levels > 16 || rt.view.current_mip_level >= levels) {
		EXIT("unsupported render-target mip range: current=%u levels=%u\n",
		     rt.view.current_mip_level, levels);
	}
	static constexpr std::array image_types {Prospero::ImageType::kColor1D,
	                                         Prospero::ImageType::kColor2D,
	                                         Prospero::ImageType::kColor3D};
	if (rt.attrib3.dimension >= image_types.size()) {
		EXIT("unsupported render-target dimension: %u\n", rt.attrib3.dimension);
	}
	const auto image_type = image_types[rt.attrib3.dimension];
	const bool is_1d      = image_type == Prospero::ImageType::kColor1D;
	const bool volume     = image_type == Prospero::ImageType::kColor3D;
	if (is_1d && rt.attrib2.height != 0) {
		EXIT("1D render target has nonzero height: %u\n", rt.attrib2.height);
	}
	if (!volume && rt.attrib3.depth != 0) {
		EXIT("non-3D render target has nonzero depth: %u\n", rt.attrib3.depth);
	}
	if (is_1d && samples != 1) {
		EXIT("multisampled 1D render targets are unsupported\n");
	}
	if (volume && samples != 1) {
		EXIT("multisampled 3D render targets are unsupported\n");
	}
	const uint32_t depth = volume ? rt.attrib3.depth + 1u : 1u;
	// For volumes, CB_COLOR_VIEW bounds exported slices; ATTRIB3 defines storage depth.
	// The host attachment contains only the selected slices that exist in this mip.
	const uint32_t last_layer = volume
	                                ? std::min(rt.view.last_array_slice_index,
	                                           std::max(depth >> rt.view.current_mip_level, 1u) - 1u)
	                                : rt.view.last_array_slice_index;
	const auto view = ResolveTargetViewInfo(
	    rt.view.base_array_slice_index, last_layer, render_target_slice_offset);
	switch (view.type) {
		case TargetViewType::Image2D:
		case TargetViewType::Image2DArray: break;
		case TargetViewType::Unsupported:
			EXIT("invalid render-target view: base=%u last=%u draw_offset=%u\n",
			     rt.view.base_array_slice_index, rt.view.last_array_slice_index,
			     render_target_slice_offset);
	}
	if (graphics_debug_dump_enabled()) {
		static std::atomic_uint log_count = 0;
		const auto              log_id    = log_count.fetch_add(1, std::memory_order_relaxed);
		if (log_id < 128) {
			LOGF("RenderColorTarget: inspect slot=%" PRIu32 " base=0x%010" PRIx64
			     " mask=0x%01" PRIx32 " attrib2_width=%" PRIu32 " attrib2_height=%" PRIu32
			     " attrib3_tile=0x%08" PRIx32 " attrib3_dim=0x%08" PRIx32 " fmt=0x%08" PRIx32
			     " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32 "\n",
			     rt_slot, rt.base.addr, mask, rt.attrib2.width, rt.attrib2.height,
			     static_cast<uint32_t>(rt.attrib3.tile_mode), rt.attrib3.dimension,
			     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
			     static_cast<uint32_t>(rt.info.channel_order));
		}
	}

	// Color-control state selects the color-buffer operation and logical blend operation.
	// The normal copy operation is a regular color write, not an attachment clear.
	// Nonlinear clear values are still stored as normalized components.
	// Fast color clears are metadata driven and must be handled explicitly when
	// that metadata path is implemented; render-pass load must preserve contents.
	uint32_t   width  = 0;
	uint32_t   height = 0;
	uint32_t   pitch  = 0;
	uint64_t   size   = 0;
	bool       tile   = false;
	const bool     standard4    = rt.attrib3.tile_mode == Prospero::TileMode::kStandard4KB;
	const bool     standard64   = rt.attrib3.tile_mode == Prospero::TileMode::kStandard64KB;
	const bool     depth_tile   = rt.attrib3.tile_mode == Prospero::TileMode::kDepth;
	const bool     texture_tile = standard4 || standard64 || depth_tile;

	switch (rt.attrib3.tile_mode) {
		case Prospero::TileMode::kLinear:
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
		case Prospero::TileMode::kDepth:
		case Prospero::TileMode::kRenderTarget:
			tile = !RenderIsColorTileModeLinear(rt.attrib3.tile_mode);
			break;
		default: EXIT("unknown tile mode: %u\n", static_cast<uint32_t>(rt.attrib3.tile_mode));
	}
	if (!tile && levels > 1) {
		EXIT("linear mipmapped render targets are unsupported\n");
	}
	if (samples > 1 && (!tile || levels != 1)) {
		EXIT("multisampled render targets require a single-mip tiled surface\n");
	}
	if (texture_tile && samples != 1) {
		EXIT("texture-tiled color render targets do not support multisampling\n");
	}

	width  = rt.attrib2.width + 1;
	height = rt.attrib2.height + 1;
	const auto target_format =
	    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order);
	const auto bytes_per_element = target_format.bytes_per_element;
	if (bytes_per_element == 0) {
		EXIT("render-target format has no valid element size\n");
	}
	const auto transfer_format = ImageOps::RenderTargetTransferFormat(bytes_per_element);
	TileTextureBlockLayout texture_tile_layout {};
	if (texture_tile &&
	    (!TileGetTextureBlockLayout(transfer_format, rt.attrib3.tile_mode, volume,
	                                texture_tile_layout) ||
	     (rt.base.addr & (texture_tile_layout.block.block_size - 1u)) != 0 ||
	     rt.info.fmask_compression_enable || rt.info.fmask_data_compression_disable ||
	     rt.info.fmask_one_frag_mode || rt.info.cmask_fast_clear_enable ||
	     rt.info.dcc_compression_enable || rt.cmask.addr != 0 || rt.fmask.addr != 0 ||
	     rt.dcc_addr.addr != 0 || rt.dcc.data_write_on_dcc_clear_to_reg)) {
		EXIT("unsupported texture-tiled render target: addr=0x%016" PRIx64 " tile=%u"
		     " dimension=%u depth=%u levels=%u layer=%u/%u samples=%u fragments=%u bpe=%u"
		     " cmask=0x%016" PRIx64 " fmask=0x%016" PRIx64 " dcc=0x%016" PRIx64 "\n",
		     rt.base.addr, static_cast<uint32_t>(rt.attrib3.tile_mode), rt.attrib3.dimension,
		     rt.attrib3.depth, levels, view.base_layer, view.image_layers, rt.attrib.num_samples,
		     rt.attrib.num_fragments, bytes_per_element, rt.cmask.addr, rt.fmask.addr,
		     rt.dcc_addr.addr);
	}
	if ((standard64 || depth_tile) &&
	    (rt.attrib3.dimension != 1 || rt.attrib3.depth != 0 ||
	     (depth_tile && (view.base_layer != 0 || view.image_layers != 1)))) {
		EXIT("unsupported 64KB texture-tiled render-target view: dimension=%u depth=%u"
		     " layer=%u/%u\n",
		     rt.attrib3.dimension, rt.attrib3.depth, view.base_layer, view.image_layers);
	}
	// PPSA28068: the linear EULA target and its sampled view must share the padded pitch.
	if (!tile || volume || texture_tile) {
		pitch = TileGetTexturePitch(transfer_format, width, rt.attrib3.tile_mode);
	} else {
		pitch = TileGetRenderTargetPitch(width, bytes_per_element, rt.attrib.num_fragments);
	}
	if (pitch == 0) {
		EXIT("unsupported render-target pitch: width=%u bytes=%u\n", width, bytes_per_element);
	}

	TileSizeOffset    mip_sizes[16] {};
	TilePaddedSize    mip_padded[16] {};
	TileSurfaceLayout volume_layout {};
	uint64_t          backing_size = 0;
	if (volume) {
		const TileSurfaceDescription description {transfer_format,
		                                          rt.attrib3.tile_mode,
		                                          TileSurfaceDimension::Dim3D,
		                                          width,
		                                          height,
		                                          depth,
		                                          levels,
		                                          1};
		if (!tile || !TileGetTiledTextureLayout(description, volume_layout)) {
			EXIT("unsupported 3D render-target layout: %ux%ux%u levels=%u tile=%u\n", width, height,
			     depth, levels, static_cast<uint32_t>(rt.attrib3.tile_mode));
		}
		size         = volume_layout.block_slice_size;
		backing_size = volume_layout.total_size;
	} else if (tile) {
		TileSizeAlign layout {};
		bool          valid_layout = false;
		if (texture_tile) {
			TileGetTextureSize(transfer_format, width, height, levels, rt.attrib3.tile_mode,
			                   &layout, mip_sizes, mip_padded);
			valid_layout = layout.size != 0 && layout.align == texture_tile_layout.block.block_size;
		} else {
			valid_layout =
			    levels == 1 ? TileGetRenderTargetSize(width, height, pitch, bytes_per_element,
			                                          layout, rt.attrib.num_fragments)
			                : TileGetRenderTargetMipLayout(width, height, pitch, bytes_per_element,
			                                               levels, layout, mip_sizes, mip_padded);
		}
		if (!valid_layout) {
			EXIT("unsupported render-target layout: %ux%u pitch=%u bytes=%u levels=%u\n", width,
			     height, pitch, bytes_per_element, levels);
		}
		size = layout.size;
		EXIT_IF(size > UINT32_MAX);
		if (levels == 1) {
			mip_sizes[0]  = {static_cast<uint32_t>(size), 0, 0, 0, 0, 0};
			mip_padded[0] = {pitch, height};
		}
	} else {
		size = static_cast<uint64_t>(pitch) * height * bytes_per_element * samples;
		if (size > UINT32_MAX) {
			EXIT("linear render-target slice exceeds the supported layout size\n");
		}
		mip_sizes[0]  = {static_cast<uint32_t>(size), 0, 0, 0, 0, 0};
		mip_padded[0] = {pitch, height};
	}
	if (size == 0 || (!volume && size > UINT64_MAX / view.image_layers)) {
		EXIT("render-target memory footprint is invalid\n");
	}
	if (!volume) {
		backing_size = size * view.image_layers;
	}
	if (backing_size == 0) {
		EXIT("render-target backing is empty\n");
	}
	if (!GuestRange {rt.base.addr, backing_size}.Valid()) {
		EXIT("render-target backing range is invalid\n");
	}

	const vk::Extent2D view_extent = {std::max(width >> rt.view.current_mip_level, 1u),
	                                  std::max(height >> rt.view.current_mip_level, 1u)};

	// Only the first 128 decisions are logged; skip the atomic increment afterwards.
	const auto decision_log_id = g_render_color_log_count.load(std::memory_order_relaxed) < 128
	                                 ? g_render_color_log_count.fetch_add(1)
	                                 : 128u;
	if (decision_log_id < 128) {
		LOGF("RenderColorTarget: slot=%" PRIu32 " addr=0x%010" PRIx64 " size=0x%016" PRIx64
		     " extent=%ux%ux%u view_mip=%u view_extent=%ux%u levels=%u pitch=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32 " samples=%u tile=%s\n",
		     rt_slot, rt.base.addr, backing_size, width, height, depth, rt.view.current_mip_level,
		     view_extent.width, view_extent.height, levels, pitch,
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), samples, tile ? "tiled" : "linear");
	}

	TextureCache::ImageDesc desc {};
	desc.type              = TextureCache::BindingType::RenderTarget;
	desc.info.data         = {rt.base.addr, backing_size};
	desc.info.pixel_format = target_format.format;
	desc.info.guest_format = transfer_format;
	desc.info.type         = image_type;
	desc.info.extent       = {width, height, depth};
	desc.info.resources    = {levels, volume ? 1u : view.image_layers};
	desc.info.pitch        = pitch;
	desc.info.bytes_per_block = bytes_per_element;
	desc.info.samples         = samples;
	desc.info.tile_mode       = rt.attrib3.tile_mode;
	const bool has_dcc        = rt.info.dcc_compression_enable && rt.dcc_addr.addr != 0;
	if (has_dcc) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : view.image_layers, bytes_per_element,
		                     levels, rt.attrib3.tile_mode, metadata_size, rt.attrib.num_fragments);
		desc.info.metadata.kind                     = ImageMetadataKind::Dcc;
		desc.info.metadata.range                    = {rt.dcc_addr.addr, metadata_size.size};
		desc.info.metadata.dcc_clear_word           = rt.clear_word0.word0;
		desc.info.metadata.dcc_clear_word1          = rt.clear_word1.word1;
		desc.info.metadata.dcc_clear_register_valid = true;
		desc.info.metadata.dcc_alpha_msb            = DccAlphaOnMsb(rt.info);
	}
	// CMASK fast clears (CB_COLORn_INFO.FAST_CLEAR): a CMASK tile marked cleared reads as the
	// CLEAR_WORD colour to the colour block and is written with it by the fast-clear eliminate.
	// The host image holds expanded texels, so the texture cache applies a pending clear when the
	// target is bound (TextureCache::MaterializeCmaskClear). Single-sample, single-mip 2D targets.
	if (!has_dcc && CmaskFastClearEnabled() && rt.info.cmask_fast_clear_enable &&
	    rt.cmask.addr != 0 && tile && !texture_tile && !volume && samples == 1 && levels == 1) {
		TileSizeAlign cmask_size {};
		if (TileGetCmaskSize(width, height, view.image_layers, cmask_size)) {
			desc.cmask.range       = {rt.cmask.addr, cmask_size.size};
			desc.cmask.clear_word0 = rt.clear_word0.word0;
			desc.cmask.clear_word1 = rt.clear_word1.word1;
			desc.cmask.valid       = true;
		}
	}
	for (uint32_t level = 0; level < levels; level++) {
		if (volume) {
			const auto& mip             = volume_layout.mips[level];
			desc.info.mip_layout[level] = {
			    mip.offset,
			    mip.size,
			    mip.padded_width,
			    mip.padded_height,
			};
			continue;
		}
		const auto level_offset =
		    mip_sizes[level].src_size != 0 ? mip_sizes[level].src_offset : mip_sizes[level].offset;
		const auto level_size =
		    static_cast<uint64_t>(mip_sizes[level].src_size != 0 ? mip_sizes[level].src_size
		                                                         : mip_sizes[level].size) *
		    view.image_layers;
		desc.info.mip_layout[level] = {
		    level_offset,
		    level_size,
		    mip_padded[level].width,
		    mip_padded[level].height,
		};
	}
	desc.view_info.format = target_format.format;
	if (is_1d) {
		desc.view_info.type =
		    view.layer_count == 1 ? vk::ImageViewType::e1D : vk::ImageViewType::e1DArray;
	} else {
		desc.view_info.type =
		    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	}
	desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	desc.view_info.base_level  = rt.view.current_mip_level;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eColorAttachment;
	if (memo != nullptr) {
		std::memcpy(&memo->registers, &rt, sizeof(rt));
		memo->mask              = mask;
		memo->slice_offset      = render_target_slice_offset;
		memo->desc              = desc;
		memo->guest_mip_level   = rt.view.current_mip_level;
		memo->guest_array_layer = view.base_layer;
		memo->export_mapping    = target_format.export_mapping;
		memo->lookup            = {};
		memo->valid             = true;
	}
	r.desc                     = std::move(desc);
	r.guest_mip_level          = rt.view.current_mip_level;
	r.guest_array_layer        = view.base_layer;
	r.image_id = FindTargetImage(r.desc, exact_format, memo != nullptr ? &memo->lookup : nullptr);
	r.export_mapping           = target_format.export_mapping;
	BindRenderTarget(r.image_id);
}

} // namespace Libs::Graphics
