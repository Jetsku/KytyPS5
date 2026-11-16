#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "common/staticOnce.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/skipInvalid.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_set>
#include <limits>

namespace Libs::Graphics {

[[noreturn]] static void DepthFatal(const char* format, ...) {
	// The details go into the error (and so the log) too, not only to the console.
	char    details[512] {};
	va_list args;
	va_start(args, format);
	std::vsnprintf(details, sizeof(details), format, args);
	va_end(args);
	std::fprintf(stderr, "Depth target fatal: %s\n", details);
	std::fflush(stderr);
	EXIT("unsupported render state: %s\n", details);
}

// KYTY_SKIP_INVALID (usa-ratt/KytyPS5-GT7 9d26551c4, a GT7 time trial ended after 16 minutes): a
// stencil or depth state Vulkan cannot express exactly is drawn with the nearest one it can, or its
// draw left out, instead of ending the emulator. False (and nothing logged) with the switch off:
// the caller then calls DepthFatal. The first 16 are logged.
static bool DepthApproximation(const char* format, ...) {
	if (!Common::SkipInvalidEnabled()) {
		return false;
	}
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) >= 16) {
		return true;
	}
	char    details[512] {};
	va_list args;
	va_start(args, format);
	std::vsnprintf(details, sizeof(details), format, args);
	va_end(args);
	LOGF("DepthTarget: %s\n", details);
	return true;
}

static vk::StencilOp ConvertStencilOp(uint8_t value, uint8_t write_mask, uint8_t op_value) {
	if (write_mask == 0) {
		return vk::StencilOp::eKeep;
	}
	switch (static_cast<Prospero::StencilOp>(value)) {
		case Prospero::StencilOp::kKeep: return vk::StencilOp::eKeep;
		case Prospero::StencilOp::kZero: return vk::StencilOp::eZero;
		case Prospero::StencilOp::kReplaceTest: return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kReplaceOp:
			if ((op_value & write_mask) == 0) {
				return vk::StencilOp::eZero;
			}
			return vk::StencilOp::eReplace;
		case Prospero::StencilOp::kAddClamp: return vk::StencilOp::eIncrementAndClamp;
		case Prospero::StencilOp::kSubClamp: return vk::StencilOp::eDecrementAndClamp;
		case Prospero::StencilOp::kInvert: return vk::StencilOp::eInvert;
		case Prospero::StencilOp::kAddWrap: return vk::StencilOp::eIncrementAndWrap;
		case Prospero::StencilOp::kSubWrap: return vk::StencilOp::eDecrementAndWrap;
		case Prospero::StencilOp::kXor:
			if ((write_mask & op_value) == 0) {
				return vk::StencilOp::eKeep;
			}
			if ((write_mask & ~op_value) != 0 &&
			    !DepthApproximation("stencil XOR of only some written bits (write mask=0x%02" PRIx8
			                        ", operation value=0x%02" PRIx8 "): inverted as a whole",
			                        write_mask, op_value)) {
				DepthFatal("unsupported stencil XOR operands: write mask=0x%02" PRIx8
				           ", operation value=0x%02" PRIx8,
				           write_mask, op_value);
			}
			return vk::StencilOp::eInvert;
		default:
			if (!DepthApproximation("unknown stencil operation 0x%02" PRIx8 ": kept", value)) {
				DepthFatal("unsupported stencil operation: 0x%02" PRIx8, value);
			}
			return vk::StencilOp::eKeep;
	}
}

static vk::StencilOpState ConvertStencilState(
    uint8_t compare, const std::array<uint8_t, 3>& operations, uint8_t op_value,
    const vk::StencilOpState& state) {
	const auto test_value = state.reference;
	auto reference       = test_value;
	auto required_bits   = state.compareMask;
	if (compare == static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    compare == static_cast<uint8_t>(vk::CompareOp::eNever)) {
		required_bits = 0;
	}
	std::array<vk::StencilOp, 3> converted {};
	for (size_t i = 0; i < operations.size(); i++) {
		converted[i] = ConvertStencilOp(operations[i], state.writeMask, op_value);
		if (converted[i] != vk::StencilOp::eReplace) {
			continue;
		}
		auto replacement = test_value;
		if (static_cast<Prospero::StencilOp>(operations[i]) == Prospero::StencilOp::kReplaceOp) {
			replacement = op_value;
		}
		if (((reference ^ replacement) & required_bits & state.writeMask) != 0 &&
		    // Vulkan's one reference cannot be both: the replacement wins the written bits.
		    !DepthApproximation("stencil replacement other than the test value (compare=%u, "
		                        "compare mask=0x%02" PRIx32 ", write mask=0x%02" PRIx32
		                        ", operation value=0x%02" PRIx8 ", test value=0x%02" PRIx32
		                        "): the test compares with the replacement",
		                        compare, state.compareMask, state.writeMask, op_value, test_value)) {
			DepthFatal("unsupported stencil replacement: compare=%u, compare mask=0x%02" PRIx32
			           ", write mask=0x%02" PRIx32 ", operation value=0x%02" PRIx8
			           ", test value=0x%02" PRIx32,
			           compare, state.compareMask, state.writeMask, op_value, test_value);
		}
		// Vulkan shares one reference between comparison and every replacement on this face.
		reference = (reference & ~state.writeMask) | (replacement & state.writeMask);
		required_bits |= state.writeMask;
	}
	return {converted[0], converted[1], converted[2], static_cast<vk::CompareOp>(compare),
	        state.compareMask, state.writeMask, reference};
}

// Any thread: GetImageFormatProperties is a cached query guarded by its own mutex.
[[nodiscard]] static vk::Format ResolveHostDepthAttachmentFormat(const GraphicContext&    graphics,
                                                                 const DepthFormatPolicy& policy,
                                                                 bool     has_stencil,
                                                                 uint32_t samples) {
	const auto required_samples = vulkan_sample_count(samples);
	const auto supports         = [&](vk::Format format) {
		vk::ImageFormatProperties properties {};
		return format != vk::Format::eUndefined &&
		       graphics.GetImageFormatProperties(
		           format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, DepthTargetImageUsage(),
		           vk::ImageCreateFlags {}, &properties) == vk::Result::eSuccess &&
		       static_cast<bool>(properties.sampleCounts & required_samples);
	};
	if (!has_stencil) {
		return supports(policy.depth_attachment_format) ? policy.depth_attachment_format
		                                                : vk::Format::eUndefined;
	}
	for (const auto format: policy.stencil_attachment_formats) {
		if (supports(format)) {
			return format;
		}
	}
	return vk::Format::eUndefined;
}

static TextureCache::ImageDesc MakeDepthTargetDesc(const CommandBuffer& buffer,
                                                  const HW::DepthRenderTarget& z,
                                                  bool write_buffer = false) {
	const bool has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const auto depth_address = write_buffer ? z.z_write_base_addr : z.z_read_base_addr;
	const auto stencil_address =
	    write_buffer ? z.stencil_write_base_addr : z.stencil_read_base_addr;
	if (!z.z_info.HasValidTextureCompatibility() ||
	    !z.stencil_info.HasValidTextureCompatibility()) {
		DepthFatal("invalid PS5 depth texture-compatibility encoding");
	}
	const bool has_htile = z.z_info.htile_acceleration;
	const bool unsupported_shading_rate_encoding =
	    z.shading_rate_encoding > 1 || (z.shading_rate_encoding != 0 && !has_htile);
	const auto samples   = render_sample_count(z.z_info.num_samples);
	if (samples == 0) {
		DepthFatal("unsupported depth fragment count: %u", z.z_info.num_samples);
	}
	const auto view = ResolveTargetViewInfo(z.depth_view.slice_start, z.depth_view.slice_max);
	switch (view.type) {
		case TargetViewType::Image2D:
		case TargetViewType::Image2DArray: break;
		case TargetViewType::Unsupported:
			DepthFatal("invalid depth view: base=%u last=%u", z.depth_view.slice_start,
			           z.depth_view.slice_max);
	}
	// EXPCLEAR permits an HTile acceleration state; the host attachment is already expanded.
	// A selected mip of a single-level target is drawn below (upstream d5b70d8).
	if (z.z_info.max_mip_level != 0 || unsupported_shading_rate_encoding || depth_address == 0 ||
	    (depth_address & 0xffffu) != 0) {
		DepthFatal("unsupported depth register state");
	}
	// A stencil plane without HTile is accepted while Hi-Stencil is enabled (upstream ec94db2).
	if (has_stencil) {
		if (z.stencil_info.format != Prospero::StencilFormat::k8UInt || stencil_address == 0 ||
		    (stencil_address & 0xffffu) != 0) {
			DepthFatal("unsupported stencil attachment state");
		}
	} else if (z.stencil_read_base_addr != 0 || z.stencil_write_base_addr != 0) {
		DepthFatal("stencil state without an active stencil attachment");
	}
	if (has_htile) {
		if (z.htile_data_base_addr == 0 || (z.htile_data_base_addr & 0x7fffu) != 0) {
			DepthFatal("invalid HTile metadata address");
		}
	}
	if (!z.size.valid) {
		DepthFatal("missing depth extent");
	}
	// With a single level, the selected mip is a surface of that mip's extent at the base address.
	const auto     mip    = z.depth_view.current_mip_level;
	const uint32_t width  = std::max((static_cast<uint32_t>(z.size.x_max) + 1u) >> mip, 1u);
	const uint32_t height = std::max((static_cast<uint32_t>(z.size.y_max) + 1u) >> mip, 1u);
	if (width > 16384 || height > 16384) {
		DepthFatal("invalid depth extent");
	}
	const auto* policy = FindDepthFormatPolicy(z.z_info.format);
	if (policy == nullptr) {
		DepthFatal("unsupported depth/stencil format pair");
	}
	const auto ideal_format = DepthAttachmentFormat(*policy, has_stencil);
	const auto format =
	    ResolveHostDepthAttachmentFormat(buffer.GetGraphics(), *policy, has_stencil, samples);
	if (format == vk::Format::eUndefined) {
		DepthFatal("no host depth/stencil format supports required usage for %s",
		           vk::to_string(ideal_format).c_str());
	}
	const auto     guest_format = policy->guest_format;
	const uint32_t bytes        = policy->bytes_per_element;
	const auto     pitch        = TileGetDepthPitch(width, bytes, z.z_info.num_samples);
	TileSizeAlign depth_size {};
	TileSizeAlign stencil_size {};
	TileSizeAlign htile_size {};
	if (!TileGetDepthSize(width, height, 0, z.z_info.format, z.stencil_info.format, has_htile,
	                      stencil_size, htile_size, depth_size, z.z_info.num_samples) ||
	    depth_size.align != 65536 || depth_size.size == 0 ||
	    (has_stencil != (stencil_size.align == 65536 && stencil_size.size != 0)) ||
	    (has_htile != (htile_size.align == 32768 && htile_size.size != 0))) {
		DepthFatal("unsupported depth/stencil/HTile footprint");
	}
	if (depth_size.size > UINT64_MAX / view.image_layers ||
	    stencil_size.size > UINT64_MAX / view.image_layers ||
	    htile_size.size > UINT64_MAX / view.image_layers) {
		DepthFatal("layered depth footprint overflow");
	}
	const auto depth_backing_size   = depth_size.size * view.image_layers;
	const auto stencil_backing_size = stencil_size.size * view.image_layers;
	const auto htile_backing_size   = htile_size.size * view.image_layers;
	if (!GuestRange {depth_address, depth_backing_size}.Valid() ||
	    (has_stencil && !GuestRange {stencil_address, stencil_backing_size}.Valid()) ||
	    (has_htile && !GuestRange {z.htile_data_base_addr, htile_backing_size}.Valid())) {
		DepthFatal("layered depth backing range is invalid");
	}
	TextureCache::ImageDesc desc {};
	desc.type                 = TextureCache::BindingType::DepthTarget;
	desc.info.data            = {depth_address, depth_backing_size};
	desc.info.stencil =
	    has_stencil ? GuestRange {stencil_address, stencil_backing_size} : GuestRange {};
	desc.info.pixel_format    = format;
	desc.info.guest_format    = guest_format;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {width, height, 1};
	desc.info.resources       = {1, view.image_layers};
	desc.info.pitch           = pitch;
	desc.info.bytes_per_block = bytes;
	desc.info.samples         = samples;
	desc.info.tile_mode       = Prospero::TileMode::kDepth;
	desc.info.mip_layout[0]   = {0, depth_backing_size, pitch, height};
	desc.info.metadata.range =
	    has_htile ? GuestRange {z.htile_data_base_addr, htile_backing_size} : GuestRange {};
	desc.info.metadata.kind   = has_htile ? ImageMetadataKind::Htile : ImageMetadataKind::None;
	desc.info.metadata.stencil_compressed =
	    has_stencil && has_htile && !z.stencil_info.htile_stencil_disabled;
	desc.view_info.format = format;
	desc.view_info.type =
	    view.layer_count == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
	desc.view_info.aspect      = ImageViewOps::DepthAspectMask(format);
	desc.view_info.base_level  = 0;
	desc.view_info.level_count = 1;
	desc.view_info.base_layer  = view.base_layer;
	desc.view_info.layer_count = view.layer_count;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eDepthStencilAttachment;
	return desc;
}

// ResolveRenderDepthTarget's decision whether depth or stencil is used at all, and whether the
// registers describe an unbound attachment (then the draw has no depth target either).
// KYTY_DEPTH_COMPRESS_DISABLE_BIND=1 (default off; upstream b64e430b9 for another title): an HTile
// depth target whose pass sets DB_RENDER_CONTROL.DEPTH_COMPRESS_DISABLE is bound even with the
// depth and stencil tests off, so its pending HTile clear is materialised.
static bool DepthCompressDisableBind() {
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_DEPTH_COMPRESS_DISABLE_BIND");
		return value != nullptr && std::strcmp(value, "1") == 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

static bool DepthTargetActive(const HW::Context& hw) {
	const auto& z           = hw.GetDepthRenderTarget();
	const auto& rc          = hw.GetRenderControl();
	const auto& dc          = hw.GetDepthControl();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	// With HTile, DEPTH_COMPRESS_DISABLE expands touched tiles even with depth/stencil tests off
	// (upstream b64e430b9): a title fast-clears HTile each frame, runs such a pass, then samples the
	// depth as raw memory; unbound, the clear never lands.
	const bool depth_active = dc.z_enable || dc.depth_bounds_enable || rc.depth_clear_enable ||
	                          rc.copy_depth_to_color ||
	                          (rc.depth_compress_disable && z.z_info.htile_acceleration &&
	                           DepthCompressDisableBind());
	const bool stencil_active =
	    has_stencil && (dc.stencil_enable || rc.stencil_clear_enable || rc.copy_stencil_to_color);
	return depth_active || stencil_active;
}

static bool DepthAttachmentUnbound(const HW::DepthRenderTarget& z) {
	return z.z_info.format == Prospero::DepthFormat::kInvalid &&
	       z.stencil_info.format == Prospero::StencilFormat::kInvalid && z.z_info.num_samples == 0 &&
	       z.z_info.texture_compatibility == Prospero::TextureCompatiblePlaneCompression::kDisable &&
	       !z.z_info.expclear_enabled && !z.z_info.partially_resident && z.z_info.max_mip_level == 0 &&
	       z.stencil_info.texture_compatibility == Prospero::TextureCompatibleStencil::kDisable &&
	       !z.stencil_info.expclear_enabled && !z.stencil_info.partially_resident &&
	       z.depth_view.slice_start == 0 && z.depth_view.slice_max == 0 &&
	       z.depth_view.current_mip_level == 0 && !z.depth_view.depth_write_disable &&
	       !z.depth_view.stencil_write_disable && z.z_read_base_addr == 0 &&
	       z.z_write_base_addr == 0 && z.stencil_read_base_addr == 0 &&
	       z.stencil_write_base_addr == 0 && z.htile_data_base_addr == 0 &&
	       // DB_DEPTH_SIZE_XY is independent state and may remain programmed after the attachment
	       // formats and addresses are unbound. A zero encoding is the valid 1x1 value, so its
	       // presence alone must not manufacture a depth attachment.
	       !z.z_info.htile_acceleration && z.shading_rate_encoding == 0 && z.size.x_max == 0 &&
	       z.size.y_max == 0;
}

bool PredictRenderDepthTarget(const GraphicContext& graphics, const HW::Context& hw,
                              DepthTargetPrediction& prediction) {
	prediction = {};
	const auto& z = hw.GetDepthRenderTarget();
	if (!DepthTargetActive(hw) || DepthAttachmentUnbound(z)) {
		return true; // no target: RenderDepthInfo {}
	}
	const auto* policy = FindDepthFormatPolicy(z.z_info.format);
	const auto  samples = render_sample_count(z.z_info.num_samples);
	if (policy == nullptr || samples == 0) {
		return false; // the resolution stops the emulator
	}
	const bool has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	const auto format = ResolveHostDepthAttachmentFormat(graphics, *policy, has_stencil, samples);
	if (format == vk::Format::eUndefined) {
		return false;
	}
	const auto& dc                = hw.GetDepthControl();
	prediction.with_depth         = true;
	prediction.format             = format;
	prediction.samples            = samples;
	prediction.bounds_test_enable = dc.depth_bounds_enable;
	prediction.min_bounds         = hw.GetDepthBoundsMin();
	prediction.max_bounds         = hw.GetDepthBoundsMax();
	return true;
}

// KYTY_STENCIL_REF_TRACE=<ref> (diagnostic, default off): prints the guest depth/stencil state of
// draws testing or writing that stencil reference, once per distinct state.
namespace {
struct StencilTraceDraw {
	uint32_t index_count = 0;
	uint64_t ps_addr     = 0;
	uint32_t color_count = 0;
	uint32_t mrt_mask    = 0;
};
thread_local StencilTraceDraw g_stencil_trace_draw;
} // namespace

void SetStencilTraceDraw(uint32_t index_count, uint64_t ps_addr, uint32_t color_count,
                         uint32_t mrt_mask) {
	g_stencil_trace_draw = {index_count, ps_addr, color_count, mrt_mask};
}

static void TraceStencilRefState(const HW::Context& hw, const RenderDepthInfo& r) {
	static constexpr auto ref_init = [] {
		const auto* value = std::getenv("KYTY_STENCIL_REF_TRACE");
		return value != nullptr && value[0] != '\0' ? std::atoi(value) : -1;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(ref_init())>> ref_once;
	const int& ref = ref_once.Get(ref_init);
	const auto& sm = hw.GetStencilMask();
	if (ref < 0 || (sm.stencil_testval != ref && sm.stencil_opval != ref)) {
		return;
	}
	const auto& dc = hw.GetDepthControl();
	const auto& sc = hw.GetStencilControl();
	const auto& rc = hw.GetRenderControl();
	const auto& ro = hw.GetDepthRenderOverride();
	const auto& z  = hw.GetDepthRenderTarget();
	const auto& sh = hw.GetShaderRegisters().db_shader_control;
	char line[1024];
	std::snprintf(line, sizeof(line),
	              "StencilRefTrace: dc z=%d zw=%d zfunc=%u bounds=%d st=%d sfunc=%u/%u bf=%d | sc "
	              "fail=%u zpass=%u zfail=%u bf=%u/%u/%u | sm test=%u mask=0x%02x wmask=0x%02x "
	              "opval=%u bf test=%u mask=0x%02x wmask=0x%02x opval=%u | rc dclr=%d sclr=%d | ro "
	              "zv=%d zd=%d sv=%d sd=%d | view zro=%d sro=%d | shctl kill=%d zexp=%d mask=%d "
	              "zbeh=%u other=0x%08x a2m_dis=%d | host dw=%d dfunc=%d sload=%d",
	              dc.z_enable, dc.z_write_enable, dc.zfunc, dc.depth_bounds_enable, dc.stencil_enable,
	              dc.stencilfunc, dc.stencilfunc_bf, dc.backface_enable, sc.stencil_fail,
	              sc.stencil_zpass, sc.stencil_zfail, sc.stencil_fail_bf, sc.stencil_zpass_bf,
	              sc.stencil_zfail_bf, sm.stencil_testval, sm.stencil_mask, sm.stencil_writemask,
	              sm.stencil_opval, sm.stencil_testval_bf, sm.stencil_mask_bf, sm.stencil_writemask_bf,
	              sm.stencil_opval_bf, rc.depth_clear_enable, rc.stencil_clear_enable,
	              ro.force_z_valid, ro.force_z_dirty, ro.force_stencil_valid, ro.force_stencil_dirty,
	              z.depth_view.depth_write_disable, z.depth_view.stencil_write_disable,
	              sh.shader_kill_enable, sh.shader_z_export_enable, sh.shader_mask_export_enable,
	              sh.shader_z_behavior, sh.other_bits, sh.alpha_to_mask_disable,
	              r.depth_write_enable, static_cast<int>(r.depth_compare_op),
	              r.stencil_load_clear_enable);
	// Colour side of the same draw: blend per MRT 0-3, target and shader masks, colour control
	// and the export formats.
	const auto& shr = hw.GetShaderRegisters();
	const auto& cc  = hw.GetColorControl();
	auto        used = std::strlen(line);
	used += static_cast<size_t>(std::snprintf(
	    line + used, sizeof(line) - used, " | cb tmask=0x%08x smask=0x%08x mode=%u rop=0x%02x fmt=%u/%u/%u/%u",
	    hw.GetRenderTargetMask(), shr.m_cbShaderMask, cc.mode, cc.op, shr.target_output_mode[0],
	    shr.target_output_mode[1], shr.target_output_mode[2], shr.target_output_mode[3]));
	used += static_cast<size_t>(std::snprintf(
	    line + used, sizeof(line) - used, " | draw idx=%u ps=0x%010llx colors=%u mrt=0x%x",
	    g_stencil_trace_draw.index_count,
	    static_cast<unsigned long long>(g_stencil_trace_draw.ps_addr),
	    g_stencil_trace_draw.color_count, g_stencil_trace_draw.mrt_mask));
	for (uint32_t slot = 0; slot < 4 && used < sizeof(line); slot++) {
		const auto& b = hw.GetBlendControl(slot);
		used += static_cast<size_t>(std::snprintf(
		    line + used, sizeof(line) - used, " b%u=%d:%u,%u,%u/%u,%u,%u,%d,byp%d,f%u/%u", slot, b.enable,
		    b.color_srcblend, b.color_comb_fcn, b.color_destblend, b.alpha_srcblend,
		    b.alpha_comb_fcn, b.alpha_destblend, b.separate_alpha_blend,
		    hw.GetRenderTarget(slot).info.blend_bypass,
		    static_cast<uint32_t>(hw.GetRenderTarget(slot).info.format),
		    static_cast<uint32_t>(hw.GetRenderTarget(slot).info.channel_type)));
	}
	static std::mutex                     mutex;
	static std::unordered_set<std::string> seen;
	std::scoped_lock                       lock {mutex};
	if (seen.insert(line).second) {
		std::printf("%s\n", line);
		LOGF("%s\n", line);
		if (auto* file = std::fopen("_StencilRefTrace.txt", "a"); file != nullptr) {
			std::fprintf(file, "%s\n", line);
			std::fclose(file);
		}
	}
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& r) {
	KYTY_PROFILER_FUNCTION();
	const auto& hw          = buffer.GetRegisters();
	const auto& z           = hw.GetDepthRenderTarget();
	const auto& rc          = hw.GetRenderControl();
	const auto& dc          = hw.GetDepthControl();
	const auto& sc          = hw.GetStencilControl();
	const auto& sm          = hw.GetStencilMask();
	const bool  has_stencil = z.stencil_info.format != Prospero::StencilFormat::kInvalid;
	// KYTY_RENDER_STATE_FAST reset: the draw state's entry is not value-initialised before the
	// draw. The paths without a target do it here; the one with a target assigns every field
	// (SameRenderDepthInfo names them all), the stencil states included.
	const bool reset_here = RenderStateFastEnabled(RenderStatePart::Reset);
	m_depth_unsupported   = false;
	if (!DepthTargetActive(hw)) {
		if (reset_here) {
			r = {};
		}
		return;
	}
	if (DepthAttachmentUnbound(z)) {
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			LOGF("DepthTarget: ignoring enabled depth/stencil state without a bound attachment\n");
		}
		if (reset_here) {
			r = {};
		}
		return;
	}
	if (rc.copy_stencil_to_color ||
	    dc.zfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
	    (!z.depth_view.depth_write_disable && z.z_write_base_addr != z.z_read_base_addr) ||
	    (has_stencil && !z.depth_view.stencil_write_disable &&
	     z.stencil_write_base_addr != z.stencil_read_base_addr)) {
		// Stencil copied to colour (a41d85085), a depth compare out of range, separate read and
		// write bases being written: the draw is left out (m_depth_unsupported).
		if (!DepthApproximation(
		        "unsupported depth register state, the draw left out: copy depth %d stencil %d "
		        "centroid %d sample %u, zfunc %u, depth write %d read 0x%012" PRIx64
		        " write 0x%012" PRIx64 ", stencil write %d read 0x%012" PRIx64 " write 0x%012" PRIx64,
		        rc.copy_depth_to_color ? 1 : 0, rc.copy_stencil_to_color ? 1 : 0,
		        rc.copy_centroid ? 1 : 0, static_cast<uint32_t>(rc.copy_sample),
		        static_cast<uint32_t>(dc.zfunc), z.depth_view.depth_write_disable ? 0 : 1,
		        static_cast<uint64_t>(z.z_read_base_addr), static_cast<uint64_t>(z.z_write_base_addr),
		        (has_stencil && !z.depth_view.stencil_write_disable) ? 1 : 0,
		        static_cast<uint64_t>(z.stencil_read_base_addr),
		        static_cast<uint64_t>(z.stencil_write_base_addr))) {
			DepthFatal("unsupported depth register state");
		}
		r                   = {};
		m_depth_unsupported = true;
		return;
	}
	if (rc.copy_depth_to_color &&
	    (z.z_info.num_samples != 0 || rc.copy_sample != 0 ||
	     z.z_info.format != Prospero::DepthFormat::kZ32F ||
	     dc.z_enable || dc.z_write_enable || dc.depth_bounds_enable || dc.stencil_enable ||
	     rc.depth_clear_enable || rc.stencil_clear_enable)) {
		if (!DepthApproximation("unsupported depth-to-color copy state (samples %u, copy sample "
		                        "%u, format %u), the draw left out",
		                        static_cast<uint32_t>(z.z_info.num_samples),
		                        static_cast<uint32_t>(rc.copy_sample),
		                        static_cast<uint32_t>(z.z_info.format))) {
			DepthFatal("unsupported depth-to-color copy state");
		}
		r                   = {};
		m_depth_unsupported = true;
		return;
	}
	// KYTY_TARGET_DESC_MEMO: MakeDepthTargetDesc is a pure function of the depth-target
	// registers (and constant device format support), so equal register bytes give an equal
	// description. FindImage below still runs for every draw.
	auto& memo = m_depth_target_memo;
	if (TargetDescMemoEnabled() && memo.valid &&
	    std::memcmp(&memo.registers, &z, sizeof(z)) == 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::TargetDescMemoHits);
		CommitStats::AddItems(CommitStats::Item::DepthMemoHit, 1);
		r.desc = memo.desc;
	} else {
		r.desc = MakeDepthTargetDesc(buffer, z);
		CommitStats::AddItems(CommitStats::Item::DepthMemoMiss, 1);
		if (TargetDescMemoEnabled()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::TargetDescMemoMisses);
			std::memcpy(&memo.registers, &z, sizeof(z));
			memo.desc   = r.desc;
			memo.lookup = {};
			memo.valid  = true;
		}
	}
	r.depth_clear_enable      = rc.depth_clear_enable;
	r.depth_clear_value       = hw.GetDepthClearValue();
	r.depth_test_enable       = dc.z_enable;
	r.depth_write_enable      = r.depth_test_enable && dc.z_write_enable &&
	                            !z.depth_view.depth_write_disable && !r.depth_clear_enable;
	r.depth_compare_op        = static_cast<vk::CompareOp>(dc.zfunc);

	r.depth_bounds_test_enable = dc.depth_bounds_enable;
	r.depth_min_bounds         = hw.GetDepthBoundsMin();
	r.depth_max_bounds         = hw.GetDepthBoundsMax();

	r.stencil_clear_enable =
	    has_stencil && rc.stencil_clear_enable && !z.depth_view.stencil_write_disable;
	r.stencil_load_clear_enable = r.stencil_clear_enable;
	r.stencil_clear_value = hw.GetStencilClearValue();
	// Explicit clears until AcquireRenderTargets adds a pending HTile clear (upstream 57791645e).
	r.load_clear_aspects = {};
	if (r.depth_clear_enable) r.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
	if (r.stencil_clear_enable) r.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
	r.stencil_test_enable = has_stencil && dc.stencil_enable;
	if (r.stencil_test_enable) {
		const bool stencil_ops_disabled =
		    rc.stencil_clear_enable || z.depth_view.stencil_write_disable;
		const uint8_t front_write_mask = stencil_ops_disabled ? 0 : sm.stencil_writemask;
		const uint8_t back_write_mask  = stencil_ops_disabled ? 0 : sm.stencil_writemask_bf;
		if (dc.stencilfunc > static_cast<uint8_t>(vk::CompareOp::eAlways) ||
		    (dc.backface_enable &&
		     dc.stencilfunc_bf > static_cast<uint8_t>(vk::CompareOp::eAlways))) {
			if (!DepthApproximation("unsupported stencil compare %u / %u, the draw left out",
			                        static_cast<uint32_t>(dc.stencilfunc),
			                        static_cast<uint32_t>(dc.stencilfunc_bf))) {
				DepthFatal("unsupported stencil compare state");
			}
			r                   = {};
			m_depth_unsupported = true;
			return;
		}
		r.stencil_front = ConvertStencilState(
		    dc.stencilfunc, {sc.stencil_fail, sc.stencil_zpass, sc.stencil_zfail},
		    sm.stencil_opval, {.compareMask = sm.stencil_mask, .writeMask = front_write_mask,
		                       .reference = sm.stencil_testval});
		if (dc.backface_enable) {
			r.stencil_back = ConvertStencilState(
			    dc.stencilfunc_bf, {sc.stencil_fail_bf, sc.stencil_zpass_bf, sc.stencil_zfail_bf},
			    sm.stencil_opval_bf, {.compareMask = sm.stencil_mask_bf, .writeMask = back_write_mask,
			                          .reference = sm.stencil_testval_bf});
		} else {
			r.stencil_back = r.stencil_front;
		}
	} else if (reset_here) {
		r.stencil_front = vk::StencilOpState {};
		r.stencil_back  = vk::StencilOpState {};
	}
	TraceStencilRefState(hw, r);
	// r.desc is the memo's description (KYTY_DRAW_SEQUENCE_FAST: its last lookup may repeat).
	r.image_id = FindTargetImage(r.desc, false, TargetDescMemoEnabled() ? &memo.lookup : nullptr);
	BindRenderTarget(r.image_id);
}

bool RenderExecutor::DepthStencilCopy(CommandBuffer& buffer) {
	KYTY_GPU_OP_SITE("depth.stencil_copy");
	const auto& hw       = buffer.GetRegisters();
	const auto& z        = hw.GetDepthRenderTarget();
	const auto& override = hw.GetDepthRenderOverride();
	if (hw.GetColorControl().mode != 0) {
		return false;
	}
	const bool depth_copy = override.force_z_dirty && override.force_z_valid &&
	                        z.z_info.format != Prospero::DepthFormat::kInvalid &&
	                        z.z_read_base_addr != 0 && z.z_write_base_addr != 0 &&
	                        z.z_read_base_addr != z.z_write_base_addr;
	const bool stencil_copy = override.force_stencil_dirty && override.force_stencil_valid &&
	                          z.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	                          z.stencil_read_base_addr != 0 && z.stencil_write_base_addr != 0 &&
	                          z.stencil_read_base_addr != z.stencil_write_base_addr;
	if (!depth_copy && !stencil_copy) {
		return false;
	}

	auto  read_desc  = MakeDepthTargetDesc(buffer, z);
	auto  write_desc = MakeDepthTargetDesc(buffer, z, true);
	auto& cache      = m_context.GetTextureCache();
	const auto read_id = cache.FindImage(read_desc);
	BindRenderTarget(read_id);
	const auto write_id = cache.FindImage(write_desc);
	BindRenderTarget(write_id);
	cache.UpdateImage(read_id);
	cache.UpdateImage(write_id);
	cache.MarkGpuWritten(write_id);
	auto& source      = cache.GetImage(read_id);
	auto& destination = cache.GetImage(write_id);
	EXIT_IF(read_id == write_id || source.backing.format != destination.backing.format);

	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();
	const auto command = scheduler.Current().Handle();
	const ImageSubresourceRange range {read_desc.view_info.base_level, 1,
	                                  read_desc.view_info.base_layer,
	                                  read_desc.view_info.layer_count};
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               range, command);
	destination.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                    range, command);
	std::array<vk::ImageCopy, 2> regions {};
	uint32_t                   count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil}) {
		if ((aspect == vk::ImageAspectFlagBits::eDepth && !depth_copy) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil && !stencil_copy)) {
			continue;
		}
		auto& region = regions[count++];
		region.srcSubresource = {aspect, range.base_level, range.base_layer,
		                         range.layer_count};
		region.dstSubresource = region.srcSubresource;
		region.extent = write_desc.info.extent;
	}
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
	                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
	                  count, regions.data());
	destination.NoteContentWrite();
	return true;
}

bool SameRenderDepthInfo(const RenderDepthInfo& a, const RenderDepthInfo& b) {
	// Every field, named in declaration order (see ResolveRenderDepthTarget): a new field stops
	// this compiling until it is compared below.
	const auto& [a_desc, a_clear, a_load_clear, a_clear_value, a_test, a_write, a_compare,
	             a_bounds, a_min_bounds, a_max_bounds, a_stencil_clear, a_stencil_load_clear,
	             a_stencil_clear_value,
	             a_stencil_test, a_front, a_back, a_image] = a;
	const auto& [b_desc, b_clear, b_load_clear, b_clear_value, b_test, b_write, b_compare,
	             b_bounds, b_min_bounds, b_max_bounds, b_stencil_clear, b_stencil_load_clear,
	             b_stencil_clear_value,
	             b_stencil_test, b_front, b_back, b_image] = b;
	const auto same_float = [](float x, float y) {
		return std::bit_cast<uint32_t>(x) == std::bit_cast<uint32_t>(y);
	};
	return SameImageDesc(a_desc, b_desc) && a_clear == b_clear && a_load_clear == b_load_clear &&
	       same_float(a_clear_value, b_clear_value) && a_test == b_test && a_write == b_write &&
	       a_compare == b_compare && a_bounds == b_bounds &&
	       same_float(a_min_bounds, b_min_bounds) && same_float(a_max_bounds, b_max_bounds) &&
	       a_stencil_clear == b_stencil_clear && a_stencil_load_clear == b_stencil_load_clear &&
	       a_stencil_clear_value == b_stencil_clear_value &&
	       a_stencil_test == b_stencil_test && a_front == b_front && a_back == b_back &&
	       a_image == b_image;
}

vk::ImageAspectFlags RenderDepthInfo::AttachmentWriteAspects() const {
	const auto format = desc.view_info.format;
	if (format == vk::Format::eUndefined) {
		return {};
	}

	const auto           available = ImageViewOps::DepthAspectMask(format);
	vk::ImageAspectFlags writes = load_clear_aspects & available;
	if ((available & vk::ImageAspectFlagBits::eDepth) &&
	    depth_write_enable) {
		writes |= vk::ImageAspectFlagBits::eDepth;
	}
	if (!(available & vk::ImageAspectFlagBits::eStencil)) {
		return writes;
	}

	const auto face_writes = [&](const vk::StencilOpState& state) {
		if (state.writeMask == 0) {
			return false;
		}
		bool can_pass = state.compareOp != vk::CompareOp::eNever;
		bool can_fail = state.compareOp != vk::CompareOp::eAlways;
		if (state.compareMask == 0) {
			switch (state.compareOp) {
				case vk::CompareOp::eEqual:
				case vk::CompareOp::eLessOrEqual:
				case vk::CompareOp::eGreaterOrEqual:
				case vk::CompareOp::eAlways:
					can_pass = true;
					can_fail = false;
					break;
				case vk::CompareOp::eNever:
				case vk::CompareOp::eLess:
				case vk::CompareOp::eGreater:
				case vk::CompareOp::eNotEqual:
					can_pass = false;
					can_fail = true;
					break;
				default: break;
			}
		}
		const bool depth_pass = !depth_test_enable || depth_compare_op != vk::CompareOp::eNever;
		const bool depth_fail = depth_test_enable && depth_compare_op != vk::CompareOp::eAlways;
		return (can_fail && state.failOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_pass && state.passOp != vk::StencilOp::eKeep) ||
		       (can_pass && depth_fail && state.depthFailOp != vk::StencilOp::eKeep);
	};
	if (stencil_test_enable && (face_writes(stencil_front) || face_writes(stencil_back))) {
		writes |= vk::ImageAspectFlagBits::eStencil;
	}
	return writes;
}

} // namespace Libs::Graphics
