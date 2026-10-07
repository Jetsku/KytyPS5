#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"
#include "graphics/host_gpu/renderer/gpuOpProfiler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

static bool MeshRestartEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_MESH_RESTART");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

struct MeshDrawSegment {
	uint32_t first;
	uint32_t count;
	uint32_t groups;
};

// Instances one mesh dispatch of `groups` workgroups per instance can carry: instances go in Y,
// so the Y limit and the total workgroup limit both apply. Zero when one instance does not fit.
[[nodiscard]] static uint32_t MeshInstancesPerDispatch(uint32_t groups,
                                                       const vk::PhysicalDeviceMeshShaderPropertiesEXT& limits) {
	if (groups == 0 || groups > limits.maxMeshWorkGroupCount[0]) {
		return 0;
	}
	return std::min(limits.maxMeshWorkGroupCount[1], limits.maxMeshWorkGroupTotalCount / groups);
}

// Workgroups one mesh dispatch takes in X: all of them, or with split_groups (a device whose X
// limit is below its total, RADV) at most that limit; the program adds each part's first group
// (draw dword IR::PushData::MeshFirstGroupDword).
[[nodiscard]] static uint32_t MeshGroupsPerDispatch(uint32_t groups, const ShaderMeshInputInfo& mesh,
                                                    const vk::PhysicalDeviceMeshShaderPropertiesEXT& limits) {
	if (mesh.split_groups == 0) {
		return groups;
	}
	return std::min({groups, limits.maxMeshWorkGroupCount[0], limits.maxMeshWorkGroupTotalCount});
}

static void SplitMeshRestartIndices(uint64_t address, uint32_t count, uint32_t element_size,
                                    uint32_t marker, const ShaderMeshInputInfo& mesh,
                                    std::vector<MeshDrawSegment>& segments) {
	EXIT_IF(address == 0 || (element_size != 1 && element_size != 2 && element_size != 4));
	segments.clear();
	std::array<uint8_t, 4096> bytes;
	uint32_t start = 0;
	uint64_t markers = 0;
	const auto add = [&](uint32_t end) {
		const auto primitives = mesh.InputPrimitiveCount(end - start);
		if (primitives != 0) segments.push_back({start, end - start,
		    (primitives - 1u) / mesh.primitives_per_group + 1u});
	};
	for (uint32_t base = 0; base < count;) {
		const auto chunk = std::min<uint32_t>(count - base, bytes.size() / element_size);
		const auto size = static_cast<size_t>(chunk) * element_size;
		const auto source = address + static_cast<uint64_t>(base) * element_size;
		if (!LibKernel::Memory::TryReadGpuCleanBacking(source, bytes.data(), size)) {
			// The ordinary mapped read follows the existing page-fault/readback path.
			std::memcpy(bytes.data(), reinterpret_cast<const void*>(source), size);
		}
		for (uint32_t i = 0; i < chunk; ++i) {
			uint32_t value = 0;
			std::memcpy(&value, bytes.data() + i * element_size, element_size);
			if (value == marker) {
				add(base + i);
				start = base + i + 1u;
				++markers;
			}
		}
		base += chunk;
	}
	add(count);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartMarkers, markers);
	Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartSegments, segments.size());
}
std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u attr_num=%d\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index, b.attr_num);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < b.attr_num; ai++) {
					const auto  res_index = b.attr_indices[ai];
					const auto& r         = vs_input_info.resources[res_index];
					const auto& rd        = vs_input_info.resources_dst[res_index];
					const auto  offset    = b.attr_offsets[ai];
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < b.attr_num; ai++) {
			const auto  res_index = b.attr_indices[ai];
			const auto& r         = vs_input_info.resources[res_index];
			const auto& rd        = vs_input_info.resources_dst[res_index];
			LOGF("DrawInputState[%u]: attr[%d] res=%d offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, res_index, b.attr_offsets[ai], rd.register_start, rd.registers_num,
			     rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static bool DynamicStateShadowEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DYNAMIC_STATE_SHADOW");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// Dynamic-state commands recorded by draws (GPU thread; KYTY_DRAW_RUN=verify reads the difference
// around a continuation's dynamic state).
static uint64_t g_dynamic_state_emitted = 0;

// Records one dynamic-state group unless the shadow proves the command buffer already holds
// exactly these values (see GraphicsDynamicStateShadow for when that holds).
class DynamicStateRecorder {
public:
	DynamicStateRecorder(GraphicsDynamicStateShadow& shadow, const CommandBuffer& buffer,
	                     vk::CommandBuffer vk_buffer /* identity */)
	    : m_shadow(shadow) {
		m_reuse = DynamicStateShadowEnabled() && shadow.valid && shadow.command == vk_buffer &&
		          shadow.pipeline != nullptr &&
		          buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) == shadow.pipeline;
		if (!m_reuse) {
			shadow         = {};
			shadow.command = vk_buffer;
		}
	}
	~DynamicStateRecorder() {
		m_shadow.valid = DynamicStateShadowEnabled();
		if (m_emitted != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsEmitted, m_emitted);
			g_dynamic_state_emitted += m_emitted;
		}
		if (m_avoided != 0) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsAvoided, m_avoided);
		}
	}
	DynamicStateRecorder(const DynamicStateRecorder&)            = delete;
	DynamicStateRecorder& operator=(const DynamicStateRecorder&) = delete;

	// Whether the group must be recorded; `current` is the shadow's value for the group and
	// `valid` whether the shadow holds it at all. Stores the new value when recording. Values
	// are compared bitwise (types without padding), so a skipped command would have set
	// bit-identical state.
	template <typename T>
	bool Update(T& current, const T& value, bool valid = true) {
		static_assert(std::is_trivially_copyable_v<T>);
		if (m_reuse && valid && std::memcmp(&current, &value, sizeof(T)) == 0) {
			++m_avoided;
			return false;
		}
		current = value;
		++m_emitted;
		return true;
	}
	[[nodiscard]] bool Reuse() const { return m_reuse; }
	void Emitted(uint64_t count) { m_emitted += count; }
	void Avoided(uint64_t count) { m_avoided += count; }

private:
	GraphicsDynamicStateShadow& m_shadow;
	bool                        m_reuse   = false;
	uint64_t                    m_emitted = 0;
	uint64_t                    m_avoided = 0;
};

// The union of the scissors a draw rasterizes with (SetGraphicsDynamicParams), not yet clamped to
// the framebuffer: every pixel the draw can write lies inside it (KYTY_ALIAS_BYTES claims only
// the render-target blocks under it).
// quiet: on a draw-prep thread, where calc_final_scissor must not log: false for a clip-rect rule
// it does not support (the command processor computes the union then, logging it).
static bool ScissorUnion(const HW::Context& ctx, bool indexed_viewports, bool quiet,
                         vk::Rect2D& result) {
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	const auto&        vp             = ctx.GetScreenViewport();
	const vk::Extent2D unbounded {16384, 16384};
	int64_t            left   = INT64_MAX;
	int64_t            top    = INT64_MAX;
	int64_t            right  = INT64_MIN;
	int64_t            bottom = INT64_MIN;
	for (uint32_t i = 0; i < (indexed_viewports ? viewport_slots : 1u); i++) {
		if (!ctx.GetClipControl().clip_disable && vp.viewports[i].xscale == 0.0f) {
			continue; // an empty slot: zero scissor
		}
		ScissorRect scissor;
		if (quiet) {
			if (!calc_scissor_unclamped(vp, ctx.GetScanModeControl(), i, scissor)) {
				return false;
			}
			scissor = clamp_scissor(scissor, unbounded);
		} else {
			scissor = calc_final_scissor(vp, ctx.GetScanModeControl(), unbounded, i);
		}
		if (scissor.right <= scissor.left || scissor.bottom <= scissor.top) {
			continue;
		}
		left   = std::min<int64_t>(left, scissor.left);
		top    = std::min<int64_t>(top, scissor.top);
		right  = std::max<int64_t>(right, scissor.right);
		bottom = std::max<int64_t>(bottom, scissor.bottom);
	}
	if (right <= left || bottom <= top) {
		result = {};
		return true;
	}
	result = {{static_cast<int32_t>(left), static_cast<int32_t>(top)},
	          {static_cast<uint32_t>(right - left), static_cast<uint32_t>(bottom - top)}};
	return true;
}

vk::Rect2D DrawScissorUnion(const HW::Context& ctx, bool indexed_viewports) {
	vk::Rect2D result {};
	(void)ScissorUnion(ctx, indexed_viewports, false, result);
	return result;
}

bool DrawScissorUnionQuiet(const HW::Context& ctx, bool indexed_viewports, vk::Rect2D& result) {
	return ScissorUnion(ctx, indexed_viewports, true, result);
}

// SetGraphicsDynamicParams' viewport for slot `i`, before an empty slot's width is made positive.
static vk::Viewport MakeDynamicViewport(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                        uint32_t i) {
	const auto&  guest = ctx.GetScreenViewport().viewports[i];
	vk::Viewport viewport {};
	if (ctx.GetClipControl().clip_disable) {
		viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
		viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
	} else {
		viewport.x      = guest.xoffset - guest.xscale;
		viewport.y      = guest.yoffset - guest.yscale;
		viewport.width  = guest.xscale * 2.0f;
		viewport.height = guest.yscale * 2.0f;
	}
	viewport.minDepth = guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
	viewport.maxDepth = guest.zscale + guest.zoffset;
	return viewport;
}

bool DrawPrep::PlanDynamicViewports(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                    bool indexed_viewports, DynamicViewportPlan& plan) {
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	static_assert(viewport_slots <= DynamicViewportPlan::MaxViewports);
	plan.valid      = false;
	plan.count      = indexed_viewports ? viewport_slots : 1;
	plan.empty_mask = 0;
	for (uint32_t i = 0; i < plan.count; i++) {
		auto& viewport = plan.viewports[i];
		viewport       = MakeDynamicViewport(ctx, limits, i);
		if (!calc_scissor_unclamped(ctx.GetScreenViewport(), ctx.GetScanModeControl(), i,
		                            plan.scissors[i])) {
			return false;
		}
		if (viewport.width == 0.0f) {
			viewport.width = 1.0f;
			plan.empty_mask |= 1u << i;
		}
	}
	plan.valid = true;
	return true;
}

// planned: the viewports and unclamped scissors of the draw's binding plan (KYTY_DRAW_PREP_BINDINGS
// dynamic, computed from the same registers and vertex program), or null.
static void SetGraphicsDynamicParams(const CommandBuffer& buffer, const CommandSink& vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering,
                                     GraphicsDynamicStateShadow&            shadow,
                                     const DrawPrep::DynamicViewportPlan* planned) {
	KYTY_PROFILER_FUNCTION();
	DynamicStateRecorder recorder(shadow, buffer, vk_buffer.Identity());

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const auto scissor_rect = [](const ScissorRect& final_scissor) {
		return vk::Rect2D {{final_scissor.left, final_scissor.top},
		                   {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                    static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)}};
	};
	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	uint32_t   viewport_count = 0;
	if (planned != nullptr && !verify) {
		viewport_count = planned->count;
		for (uint32_t i = 0; i < viewport_count; i++) {
			viewports[i] = planned->viewports[i];
			scissors[i]  = scissor_rect(clamp_scissor(planned->scissors[i], framebuffer_extent));
			if ((planned->empty_mask & (1u << i)) != 0) {
				scissors[i].extent = vk::Extent2D {0, 0};
			}
		}
		DrawPrep::GetBindingTotals().viewports_used.fetch_add(1, std::memory_order_relaxed);
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingViewportsUsed);
	} else {
		const auto& outputs = vs_input_info.stage.program->info.outputs;
		const bool  indexed_viewports =
		    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
			    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		    });
		viewport_count = indexed_viewports ? viewport_slots : 1;
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		for (uint32_t i = 0; i < viewport_count; i++) {
			auto& viewport = viewports[i];
			viewport       = MakeDynamicViewport(ctx, limits, i);

			const auto final_scissor =
			    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
			auto& scissor = scissors[i];
			scissor       = scissor_rect(final_scissor);
			if (viewport.width == 0.0f) {
				// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
				viewport.width = 1.0f;
				scissor.extent = vk::Extent2D {0, 0};
			}
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			bool same = planned->count == viewport_count;
			for (uint32_t i = 0; same && i < viewport_count; i++) {
				auto scissor = scissor_rect(clamp_scissor(planned->scissors[i], framebuffer_extent));
				if ((planned->empty_mask & (1u << i)) != 0) {
					scissor.extent = vk::Extent2D {0, 0};
				}
				same = std::memcmp(&planned->viewports[i], &viewports[i], sizeof(vk::Viewport)) == 0 &&
				       std::memcmp(&scissor, &scissors[i], sizeof(vk::Rect2D)) == 0;
			}
			if (!same) {
				DrawPrep::ReportBindingMismatch("dynamic viewports", viewport_count);
			}
		}
	}
	static_assert(viewport_slots <= GraphicsDynamicStateShadow::MaxViewports);
	// Bitwise comparisons: a skipped command must leave bit-identical state.
	const bool same_count = recorder.Reuse() && shadow.viewport_count == viewport_count;
	if (same_count && std::memcmp(shadow.viewports.data(), viewports.data(),
	                              sizeof(vk::Viewport) * viewport_count) == 0) {
		recorder.Avoided(1);
	} else {
		vk_buffer.setViewportWithCount(viewport_count, viewports.data());
		std::copy_n(viewports.begin(), viewport_count, shadow.viewports.begin());
		recorder.Emitted(1);
	}
	if (same_count && std::memcmp(shadow.scissors.data(), scissors.data(),
	                              sizeof(vk::Rect2D) * viewport_count) == 0) {
		recorder.Avoided(1);
	} else {
		vk_buffer.setScissorWithCount(viewport_count, scissors.data());
		std::copy_n(scissors.begin(), viewport_count, shadow.scissors.begin());
		recorder.Emitted(1);
	}
	shadow.viewport_count = viewport_count;

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	if (recorder.Update(shadow.line_width, line_width)) {
		vk_buffer.setLineWidth(line_width);
	}
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	if (recorder.Update(shadow.blend_constants, blend_constants)) {
		vk_buffer.setBlendConstants(blend_constants.data());
	}
	const vk::Bool32 depth_test_enable  = depth.depth_test_enable ? VK_TRUE : VK_FALSE;
	const vk::Bool32 depth_write_enable = depth.depth_write_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.depth_test_enable, depth_test_enable)) {
		vk_buffer.setDepthTestEnable(depth_test_enable);
	}
	if (recorder.Update(shadow.depth_write_enable, depth_write_enable)) {
		vk_buffer.setDepthWriteEnable(depth_write_enable);
	}
	if (recorder.Update(shadow.depth_compare_op, depth.depth_compare_op)) {
		vk_buffer.setDepthCompareOp(depth.depth_compare_op);
	}

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	const vk::Bool32 depth_bias_flag = depth_bias_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.depth_bias_enable, depth_bias_flag)) {
		vk_buffer.setDepthBiasEnable(depth_bias_flag);
	}
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		const std::array<float, 3> bias {constant_factor, poly_offset.clamp, slope_factor};
		if (recorder.Update(shadow.depth_bias, bias, shadow.depth_bias_valid)) {
			vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
			shadow.depth_bias_valid = true;
		}
	}

	const vk::Bool32 stencil_test_enable = depth.stencil_test_enable ? VK_TRUE : VK_FALSE;
	if (recorder.Update(shadow.stencil_test_enable, stencil_test_enable)) {
		vk_buffer.setStencilTestEnable(stencil_test_enable);
	}
	if (depth.stencil_test_enable) {
		const auto set_stencil = [&](vk::StencilFaceFlagBits face, const vk::StencilOpState& state) {
			vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp, state.compareOp);
			vk_buffer.setStencilCompareMask(face, state.compareMask);
			vk_buffer.setStencilWriteMask(face, state.writeMask);
			vk_buffer.setStencilReference(face, state.reference);
		};
		// Each face is recorded as its group of four commands.
		const bool valid = shadow.stencil_valid;
		if (recorder.Update(shadow.stencil_front, depth.stencil_front, valid)) {
			set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
		}
		if (recorder.Update(shadow.stencil_back, depth.stencil_back, valid)) {
			set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
		}
		shadow.stencil_valid = true;
	}

	if (PipelineDynamicRasterStateEnabled()) {
		// Formerly pipeline-key fields (PipelineCache::GetGraphicsPipeline), from the same
		// registers: rect lists are drawn without culling.
		const bool        rect_list = Prospero::IsRectList(buffer.GetUserConfig().GetPrimType());
		vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eNone;
		if (!rect_list && mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
		if (!rect_list && mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (recorder.Update(shadow.cull_mode, cull_mode)) {
			vk_buffer.setCullMode(cull_mode);
		}
		const auto front_face =
		    mode.face ? vk::FrontFace::eClockwise : vk::FrontFace::eCounterClockwise;
		if (recorder.Update(shadow.front_face, front_face)) {
			vk_buffer.setFrontFace(front_face);
		}
#if !defined(__APPLE__)
		const vk::Bool32 bounds_test = depth.depth_bounds_test_enable ? VK_TRUE : VK_FALSE;
		if (recorder.Update(shadow.depth_bounds_test_enable, bounds_test)) {
			vk_buffer.setDepthBoundsTestEnable(bounds_test);
		}
		if (depth.depth_bounds_test_enable) {
			const std::array<float, 2> bounds {depth.depth_min_bounds, depth.depth_max_bounds};
			if (recorder.Update(shadow.depth_bounds, bounds, shadow.depth_bounds_valid)) {
				vk_buffer.setDepthBounds(bounds[0], bounds[1]);
				shadow.depth_bounds_valid = true;
			}
		}
#endif
	}

	// Without VK_EXT_color_write_enable (MoltenVK, older drivers) the pipeline is created without
	// the eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead; an
	// attachment without an image view discards its writes.
	if (buffer.GetGraphics().color_write_enable_enabled && rendering.num_color_attachments != 0) {
		std::array<vk::Bool32, RENDER_COLOR_ATTACHMENTS_MAX> enable {};
		for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
			enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
		}
		const bool same = recorder.Reuse() && shadow.color_write_valid &&
		                  shadow.color_write_count == rendering.num_color_attachments &&
		                  std::memcmp(shadow.color_write.data(), enable.data(),
		                              sizeof(vk::Bool32) * rendering.num_color_attachments) == 0;
		if (same) {
			recorder.Avoided(1);
		} else {
			vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable.data());
			shadow.color_write       = enable;
			shadow.color_write_count = rendering.num_color_attachments;
			shadow.color_write_valid = true;
			recorder.Emitted(1);
		}
	}
}

// uc_check and hw_check of DrawIndex/DrawAuto. KYTY_DRAW_PREP_BINDINGS hwcheck: skipped when the
// committed draw's binding plan found they would neither stop the emulator nor log for these
// registers, the draw's snapshot bound for its commit (the verdict needs no validated
// preparation: it is a function of the registers and of log state that only moves one way).
static void RunDrawChecks(const CommandBuffer& buffer, const HW::UserConfig& ucfg,
                          const DrawPrep::BindingPlan* plan) {
	if (plan != nullptr && plan->hw_checks_quiet) {
		if (DrawPrep::BindingsVerifyMode() == 0) {
			DrawPrep::GetBindingTotals().hw_checks_skipped.fetch_add(1, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingHwChecksSkipped);
			return;
		}
		DrawPrep::CountBindingVerifyCheck();
		if (!hw_checks_quiet(buffer.GetRegisters(), ucfg)) {
			DrawPrep::ReportBindingMismatch("hw checks");
		}
	}
	uc_check(ucfg);

	hw_check(buffer);
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

// One per RenderExecutor, reused by every draw instead of being value-initialised (tens of KB)
// each time. Reset() returns it to the value-initialised state for every field a draw reads
// before writing: the fields below `programs` are bookkeeping for that.
struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
	// Per-draw program preparation. The vertex_info/ps_input_info stage runtimes point into it;
	// it is overwritten by the next preparation, and its vectors keep their capacity.
	PipelineCache::GraphicsStagePreps stage_preps;
	// color_info entries and vertex_info stages the current draw may have written.
	uint32_t color_slots_written   = 0;
	uint32_t vertex_stages_written = 0;

	void Reset() {
		// KYTY_RENDER_STATE_FAST reset: a draw reads color_info[0, color_count) and depth_info, and
		// its target resolution assigns every field of those (ResolveRenderColorTarget,
		// ResolveRenderDepthTarget). Debug dumps also log color_info[0] and depth_info of draws
		// without such targets, so they still get value-initialised entries.
		if (!RenderStateFastEnabled(RenderStatePart::Reset) || graphics_debug_dump_enabled()) {
			depth_info = {};
			for (uint32_t slot = 0; slot < color_slots_written; slot++) {
				color_info[slot] = {};
			}
		} else {
			Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMinimalResets);
		}
		color_slots_written = 0;
		color_count         = 0;
		ps_active           = true;
		// PrepareProgram/PrepareTessellationPrograms value-initialise every stage they prepare,
		// and a draw always prepares vertex_info[0]. Only the tessellation stages of an earlier
		// draw can be left behind.
		for (uint32_t stage = 1; stage < vertex_stages_written; stage++) {
			vertex_info[stage] = {};
		}
		vertex_stages_written = 0;
		// RefreshShaders resets ps_input_info and programs itself.
	}
};

RenderExecutor::RenderExecutor(RenderContext& context)
    : m_context(context), m_draw_state(std::make_unique<DrawRenderState>()) {}

RenderExecutor::~RenderExecutor() = default;

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages,
                                                 const vk::Rect2D* written) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects       = {};
	m_depth_feedback.valid = false;
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc,
		                                               target.guest_mip_level == 0 ? written
		                                                                           : nullptr);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		// FindDepthTarget treats the binding as a write. Restored below for draws that write
		// neither aspect, so an alias synchronized from this image stays provably identical.
		const auto  content_mark = cache.MarkContent(depth.image_id);
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		if (!draw_writes) {
			// Load-op LOAD, store-op STORE, no depth/stencil writes or clears: contents unchanged.
			cache.RestoreContentIfUnwritten(depth.image_id, content_mark);
		}
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Without VK_EXT_attachment_feedback_loop_layout/dynamic_state (AMD's Windows driver)
			// the draw samples the depth target it writes in the GENERAL layout, chosen below: not
			// defined by Vulkan, but what the hardware does, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog("Warning: depth attachment feedback loop without host "
				                          "support; using the GENERAL layout\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (!sampled_aspects && DepthLayoutStableEnabled()) {
			// Nothing in this draw samples the image, so its attachment layout is not observable:
			// keep the current one while it allows the draw's writes (no transition, no new
			// rendering instance).
			const auto stable = depth_stable_attachment_layout(
			    image.backing.state.layout, image.backing.subresource_states.empty(),
			    ImageViewOps::DepthAspectMask(depth.desc.view_info.format), draw_writes);
			if (stable != layout && stable == image.backing.state.layout) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DepthLayoutTransitionsAvoided);
			}
			layout = stable;
		}
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		auto        access = vk::AccessFlags2 {vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                                vk::AccessFlagBits2::eDepthStencilAttachmentWrite};
		const auto& view   = depth.desc.view_info;
		// KYTY_DEPTH_FEEDBACK_KEEP: inside the active rendering instance, with no attachment
		// write (draw writes, load clears) since it began - the content serial recorded then is
		// unchanged and this draw writes nothing - every access to the image in the instance is a
		// read (depth/stencil tests, sampling). Reads need no ordering among themselves, so the
		// attachment <-> attachment+shader-read toggles of sampling draws, a barrier and a new
		// instance per draw, are left out: the image keeps the union of both scopes.
		const auto sampled = vk::AccessFlags2 {access | vk::AccessFlagBits2::eShaderRead};
		const bool whole   = view.base_level == 0 && view.base_layer == 0 &&
		                   view.level_count == image.info.resources.levels &&
		                   view.layer_count == image.info.resources.layers;
		const auto& tracked = image.backing.state;
		const bool  proof_instance = image.feedback_instance != 0 &&
		                            image.feedback_instance == buffer.ActiveRenderingSerial();
		const bool  proof_serial   = image.feedback_serial == image.ContentSerial();
		const bool  state_matches  = whole && image.backing.subresource_states.empty() &&
		                            tracked.layout == layout &&
		                            (tracked.access_mask == access || tracked.access_mask == sampled);
		if (DepthFeedbackKeepEnabled() && sampled_aspects &&
		    (draw_writes || !proof_instance || !proof_serial || !state_matches)) {
			// Aggregate profile: why a sampled depth attachment keeps its per-draw toggles.
			Profiler::CountFrameEvent(draw_writes      ? Profiler::FrameEvent::DepthFeedbackKeepMissWrite
			                          : !proof_instance ? Profiler::FrameEvent::DepthFeedbackKeepMissInstance
			                          : !proof_serial   ? Profiler::FrameEvent::DepthFeedbackKeepMissSerial
			                                            : Profiler::FrameEvent::DepthFeedbackKeepMissState);
		}
		const bool keep = DepthFeedbackKeepEnabled() && !draw_writes && proof_instance &&
		                  proof_serial && state_matches;
		if (keep && !DepthFeedbackAdoptsUnion(true, DepthFeedbackLazyEnabled(),
		                                      static_cast<bool>(sampled_aspects),
		                                      tracked.access_mask == access)) {
			// KYTY_DEPTH_FEEDBACK_LAZY: nothing samples the image yet; it keeps the attachment access.
			DrawRun::GetTotals().depth_promotions_deferred.fetch_add(1, std::memory_order_relaxed);
		} else if (keep) {
			// Without the keep: a barrier back to the attachment scope when the previous draw
			// sampled, and one to the sampled scope when this draw samples.
			const uint64_t avoided = (tracked.access_mask == sampled ? 1u : 0u) +
			                         (sampled_aspects ? 1u : 0u);
			if (avoided != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DepthFeedbackBarriersAvoided,
				                          avoided);
			}
			vk::ImageMemoryBarrier2 ordering {};
			ordering.srcStageMask        = tracked.pl_stage;
			ordering.srcAccessMask       = sampled;
			ordering.dstStageMask        = tracked.pl_stage;
			ordering.dstAccessMask       = sampled;
			ordering.oldLayout           = layout;
			ordering.newLayout           = layout;
			ordering.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			ordering.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			ordering.image               = image.backing.image;
			ordering.subresourceRange    = {ImageViewOps::DepthAspectMask(image.backing.format), 0,
			                                VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
			buffer.NoteFeedbackKeep(ordering);
			image.AdoptState(tracked.pl_stage, sampled, layout);
			access = sampled;
		}
		m_depth_feedback = {depth.image_id, static_cast<bool>(draw_writes), true};
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

void RenderExecutor::NoteDepthFeedback(const CommandBuffer& buffer) {
	m_depth_feedback.valid = false;
	auto* image = m_context.GetTextureCache().m_slot_images.try_get(m_depth_feedback.id);
	if (image == nullptr) {
		return;
	}
	const auto instance      = buffer.ActiveRenderingSerial();
	const bool first_draw    = image->feedback_attached != instance;
	image->feedback_attached = instance;
	if (m_depth_feedback.writes || instance == 0) {
		// This draw writes the attachment (or clears it on load): the instance is not read-only.
		image->feedback_instance = 0;
		image->feedback_serial   = 0;
		return;
	}
	if (first_draw) {
		// The first draw of this instance attaching the image, and it writes nothing: later draws
		// of the instance compare the content serial against this one. An instance whose earlier
		// draw wrote the image keeps feedback_instance 0 until it ends.
		image->feedback_instance = instance;
		image->feedback_serial   = image->ContentSerial();
	}
}

static bool DrawHasActivePixelShader(const CommandBuffer& buffer) {
	const auto& ctx     = buffer.GetRegisters();
	const auto& sh_regs = ctx.GetShaderRegisters();
	// KYTY_SKIP_INACTIVE_PS also requires a nonzero export format (DrawColorOutputFilter).
	const bool has_color_output = SkipInactivePixelShadersEnabled()
	                                  ? DrawColorOutputFilter(ctx) != 0
	                                  : (ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask) != 0;
	return buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	       (has_color_output || PixelShaderHasDepthOrCoverageSideEffects(sh_regs));
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

// These special modes run color-buffer metadata or decompression operations. The shader is a
// vehicle for that operation, and its exported color must not be applied as a normal draw.
// Kyty stores expanded Vulkan images rather than compressed guest surfaces. DCC and CMASK
// clear state must be materialized here even when no ordinary draw ever binds the attachment;
// a later sampled image or buffer reader need not carry its metadata. FMask is not implemented.
static bool IsMetadataColorMode(uint8_t mode) {
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// KYTY_CB_METADATA_MATERIALIZE=0 restores the old diagnostic behavior of dropping the operation.
// By default a fast-clear-eliminate or DCC-decompress draw first resolves each supported
// render-target-tiled colour target it names (FindImage), which materializes a pending uniform
// DCC or CMASK fast clear into the image and consumes the
// clear key, exactly as a later attachment bind would. On hardware these draws exist to write the
// cleared values into the surface memory for readers without DCC metadata (a T# without META,
// buffer reads); dropping them left such readers with the surface's previous contents whenever
// no draw bound the target in between. Astro Bot's water input uses exactly this CMASK route:
// dropping the eliminate exposes stale scene texels at normal and top-down camera angles.
static bool MetadataColorMaterializeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CB_METADATA_MATERIALIZE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t      render_target_slice_offset) {
	const auto& hw   = buffer.GetRegisters();
	const auto  mode = hw.GetColorControl().mode;
	if (!IsMetadataColorMode(mode)) {
		return false;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::MetadataColorOps);
	const bool materialize =
	    MetadataColorMaterializeEnabled() &&
	    (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	     mode == static_cast<uint8_t>(CbColorMode::DccDecompress));
	const auto target_mask = hw.GetRenderTargetMask();
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& rt = hw.GetRenderTarget(slot);
		if (rt.base.addr == 0 || (slot != 0 && render_target_mask_slot(target_mask, slot) == 0)) {
			continue;
		}
		const bool dcc = rt.info.dcc_compression_enable && rt.dcc_addr.addr != 0;
		// A pending CMASK fast clear is written by the eliminate (TextureCache::MaterializeCmaskClear).
		const bool cmask = mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) &&
		                   rt.info.cmask_fast_clear_enable && rt.cmask.addr != 0 &&
		                   CmaskFastClearEnabled();
		// Only the layouts ResolveRenderColorTarget supports with DCC: render-target tiled,
		// single sample.
		const bool eligible = materialize && (dcc || cmask) &&
		                      rt.attrib3.tile_mode == Prospero::TileMode::kRenderTarget &&
		                      rt.attrib.num_samples == 0 && rt.attrib.num_fragments == 0;
		uint64_t image_size = 0;
		if (eligible) {
			RenderColorInfo target {};
			ResolveRenderColorTarget(buffer, target, render_target_slice_offset, slot, true, false);
			if (target.image_id) {
				image_size = target.desc.info.data.size;
				Profiler::CountFrameEvent(Profiler::FrameEvent::MetadataColorOpMaterializations);
			}
		}
		if (HangTrace::CpTraceEnabled()) {
			// cp.csv: address = colour base, value = CB_COLOR_CONTROL.MODE, ref = DCC base,
			// mask = CMASK base, aux = slot | DCC enable << 8 | tile mode << 16 | resolved << 24,
			// size = the resolved image's guest size (0 when not resolved).
			HangTrace::CpEvent event;
			event.event   = "cb-meta-op";
			event.address = rt.base.addr;
			event.value   = mode;
			event.ref     = rt.dcc_addr.addr;
			event.mask    = rt.cmask.addr;
			event.aux     = static_cast<int64_t>(slot) | (static_cast<int64_t>(dcc ? 1 : 0) << 8) |
			            (static_cast<int64_t>(rt.attrib3.tile_mode) << 16) |
			            (static_cast<int64_t>(image_size != 0 ? 1 : 0) << 24);
			event.size = image_size;
			HangTrace::RecordCp(event);
		}
	}
	return true;
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
	// Non-null: counts and offsets are GPU data in these records, emitted with
	// vkCmdDraw*Indirect*; the fields above are unused.
	const DrawIndirectSource* indirect = nullptr;
	// KYTY_NATIVE_INDIRECT_MESH: the indirect record feeds a mesh draw through a GPU conversion
	// (meshIndirect.h) with these inputs.
	bool                 mesh_indirect = false;
	MeshIndirect::Inputs mesh_inputs;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

struct PreparedIndirectBuffers {
	vk::Buffer     args         = nullptr;
	vk::DeviceSize args_offset  = 0;
	vk::Buffer     count        = nullptr;
	vk::DeviceSize count_offset = 0;
};

// Pure superset of the ConsumeMetadataColorOperation, DepthStencilCopy and ResolveColorTargets
// conditions. Those draws run an operation instead of drawing, but only when their CPU-visible
// counts are nonzero, so indirect draws in these modes keep the CPU-read arguments.
static bool MayRunTargetOperation(const HW::Context& hw) {
	const auto mode = hw.GetColorControl().mode;
	if (IsMetadataColorMode(mode) || mode == static_cast<uint8_t>(CbColorMode::Resolve)) {
		return true;
	}
	const auto& override = hw.GetDepthRenderOverride();
	return mode == static_cast<uint8_t>(CbColorMode::Disable) &&
	       ((override.force_z_dirty && override.force_z_valid) ||
	        (override.force_stencil_dirty && override.force_stencil_valid));
}

static bool DrawMayRunTargetOperation(const CommandBuffer& buffer) {
	return MayRunTargetOperation(buffer.GetRegisters());
}

// ResolvePrimitiveRestart without its index scan. nullopt when only a scan of CPU-visible
// indices could decide (a custom reset value), or when the CPU path would reject the control.
static std::optional<bool> ResolveNativePrimitiveRestart(const CommandBuffer& buffer,
                                                         uint32_t element_size) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) {
		return std::nullopt;
	}
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}
	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	if ((reset_index & index_mask) == index_mask) {
		return true;
	}
	return std::nullopt;
}

// Argument and count records stay in GPU memory. The stream-buffer path of ObtainBuffer only
// applies to CPU-written ranges, whose snapshot is then equally current.
static PreparedIndirectBuffers ObtainIndirectBuffers(CommandBuffer&            buffer,
                                                     const DrawIndirectSource& source) {
	auto&                   cache = buffer.GetContext().GetBufferCache();
	PreparedIndirectBuffers prepared;
	const auto [args, args_offset] = cache.ObtainBuffer(source.args_addr, source.ArgsSize(), false);
	EXIT_IF(args == nullptr || (args_offset & 3u) != 0);
	prepared.args        = args->Handle();
	prepared.args_offset = args_offset;
	if (source.count_addr != 0) {
		const auto [count, count_offset] =
		    cache.ObtainBuffer(source.count_addr, sizeof(uint32_t), false);
		EXIT_IF(count == nullptr || (count_offset & 3u) != 0);
		prepared.count        = count->Handle();
		prepared.count_offset = count_offset;
	}
	return prepared;
}

// Shader and transfer writes (compute culling, DMA, cache uploads) -> indirect command fetch.
// Must be recorded outside a rendering instance.
static void IndirectArgumentsBarrier(const CommandBuffer& buffer, const CommandSink& vk_buffer) {
	if (BarrierBatchEnabled()) {
		// Recorded (outside rendering) by the BeginRendering() that precedes the draw.
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllGraphics |
		                                vk::PipelineStageFlagBits2::eComputeShader |
		                                vk::PipelineStageFlagBits2::eTransfer,
		                            vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
		return;
	}
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0,
	                          nullptr, 0, nullptr);
}

static void EmitIndirectDraw(const CommandSink& vk_buffer, const DrawIndirectSource& source,
                             const PreparedIndirectBuffers& prepared) {
	if (source.count_addr != 0) {
		if (source.indexed) {
			vk_buffer.drawIndexedIndirectCount(prepared.args, prepared.args_offset, prepared.count,
			                                   prepared.count_offset, source.max_count,
			                                   source.stride);
		} else {
			vk_buffer.drawIndirectCount(prepared.args, prepared.args_offset, prepared.count,
			                            prepared.count_offset, source.max_count, source.stride);
		}
	} else if (source.indexed) {
		vk_buffer.drawIndexedIndirect(prepared.args, prepared.args_offset, source.max_count,
		                              source.stride);
	} else {
		vk_buffer.drawIndirect(prepared.args, prepared.args_offset, source.max_count,
		                       source.stride);
	}
}

static uint64_t VertexBufferDescriptorSize(const ShaderVertexInputBuffer& buffer,
                                           const ShaderVertexInputInfo& info) {
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < buffer.attr_num; i++) {
		const auto& resource = info.resources[buffer.attr_indices[i]];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? static_cast<uint64_t>(buffer.attr_offsets[i]) +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

bool DrawPrep::CollectVertexRanges(const ShaderVertexInputInfo& vs_input_info,
                                   VertexRangePlan& ranges, uint32_t& invalid) {
	// Collect the non-empty guest vertex ranges.
	using Merged = VertexRangePlan::Merged;
	std::array<Merged, VertexRangePlan::MaxBuffers> collected {};
	uint32_t                                        range_count = 0;
	ranges.buffer_count = static_cast<uint32_t>(vs_input_info.buffers_num);
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(vertex, vs_input_info);
		ranges.sizes[i]    = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			invalid = static_cast<uint32_t>(i);
			return false;
		}
		collected[range_count++] = {vertex.addr, vertex.addr + size, 0};
	}

	std::sort(collected.begin(), collected.begin() + range_count,
	          [](const Merged& left, const Merged& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	ranges.merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = collected[i];
		if (ranges.merged_count != 0 &&
		    ranges.merged[ranges.merged_count - 1].requested_end >= range.base_address) {
			ranges.merged[ranges.merged_count - 1].requested_end =
			    std::max(ranges.merged[ranges.merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		ranges.merged[ranges.merged_count++] = {range.base_address, range.requested_end, 0};
	}
	return true;
}

bool DrawPrep::AssignVertexRanges(const ShaderVertexInputInfo& vs_input_info,
                                  VertexRangePlan& ranges, uint32_t& unassigned) {
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		if (ranges.sizes[i] == 0) {
			continue;
		}
		const auto address = vs_input_info.buffers[i].addr;
		uint32_t   index   = 0;
		while (index < ranges.merged_count && !(address >= ranges.merged[index].base_address &&
		                                        address < ranges.merged[index].acquired_end)) {
			index++;
		}
		if (index == ranges.merged_count) {
			unassigned = static_cast<uint32_t>(i);
			return false;
		}
		ranges.merged_index[i] = static_cast<uint8_t>(index);
	}
	return true;
}

// KYTY_DRAW_PREP_BINDINGS_VERIFY: a plan's vertex ranges against the serial ones.
static bool SameVertexRanges(const DrawPrep::VertexRangePlan& a, const DrawPrep::VertexRangePlan& b) {
	if (a.buffer_count != b.buffer_count || a.merged_count != b.merged_count) {
		return false;
	}
	for (uint32_t i = 0; i < a.buffer_count; i++) {
		if (a.sizes[i] != b.sizes[i] || (a.sizes[i] != 0 && a.merged_index[i] != b.merged_index[i])) {
			return false;
		}
	}
	for (uint32_t i = 0; i < a.merged_count; i++) {
		if (a.merged[i].base_address != b.merged[i].base_address ||
		    a.merged[i].requested_end != b.merged[i].requested_end ||
		    a.merged[i].acquired_end != b.merged[i].acquired_end) {
			return false;
		}
	}
	return true;
}

// planned: the draw's guest ranges from its binding plan (KYTY_DRAW_PREP_BINDINGS, clamped by the
// preparing thread while the guest virtual ranges were as they are now), or null.
static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&                   buffer,
                                                  const ShaderVertexInputInfo&     vs_input_info,
                                                  const DrawPrep::VertexRangePlan* planned) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const DrawPrep::VertexRangePlan* ranges = planned;
	DrawPrep::VertexRangePlan        collected;
	if (planned == nullptr || verify) {
		uint32_t invalid = 0;
		if (!DrawPrep::CollectVertexRanges(vs_input_info, collected, invalid)) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[invalid].addr, collected.sizes[invalid]);
		}
		for (uint32_t i = 0; i < collected.merged_count; i++) {
			auto& range = collected.merged[i];
			// PPSA20298
			const auto size = Libs::LibKernel::Memory::ClampRangeSize(
			    range.base_address, range.requested_end - range.base_address);
			range.acquired_end = range.base_address + size;
		}
		uint32_t unassigned = 0;
		if (!DrawPrep::AssignVertexRanges(vs_input_info, collected, unassigned)) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[unassigned].addr);
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			if (!SameVertexRanges(*planned, collected)) {
				DrawPrep::ReportBindingMismatch("vertex ranges", collected.merged_count);
			}
		}
		ranges = &collected;
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	std::array<std::pair<Buffer*, uint64_t>, ShaderVertexInputInfo::RES_MAX> acquired {};
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range = ranges->merged[i];
		const auto  size  = range.acquired_end - range.base_address;
		acquired[i]       = cache.ObtainBuffer(range.base_address, size, false);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, acquired[i].first->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = ranges->sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto  index = ranges->merged_index[i];
		const auto& range = ranges->merged[index];
		prepared.buffers[i] = acquired[index].first->Handle();
		prepared.offsets[i] = acquired[index].second + vertex.addr - range.base_address;
		prepared.sizes[i]   = std::min(size, range.acquired_end - vertex.addr);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}

	return prepared;
}

// KYTY_CP_COMMIT=draws: AcquireVertexBuffers into the draw's value-initialised bindings, without
// its dead work: the collected range plan (about 1 KiB value-initialised) exists only when the
// draw has no binding plan (or verifies it), the acquired ranges are not value-initialised (only
// the merged ones are written and read), and no second PreparedVertexBuffers is built and copied.
// Every value it computes and every call it makes, in order, are AcquireVertexBuffers'.
static void AcquireVertexBuffersInto(CommandBuffer&                   buffer,
                                     const ShaderVertexInputInfo&     vs_input_info,
                                     const DrawPrep::VertexRangePlan* planned,
                                     PreparedVertexBuffers&           prepared) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);
	EXIT_IF(prepared.count != 0); // the draw's bindings, value-initialised

	const bool verify = planned != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const DrawPrep::VertexRangePlan*          ranges = planned;
	std::optional<DrawPrep::VertexRangePlan> collected_storage;
	if (planned == nullptr || verify) {
		auto&    collected = collected_storage.emplace();
		uint32_t invalid   = 0;
		if (!DrawPrep::CollectVertexRanges(vs_input_info, collected, invalid)) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[invalid].addr, collected.sizes[invalid]);
		}
		for (uint32_t i = 0; i < collected.merged_count; i++) {
			auto& range = collected.merged[i];
			// PPSA20298
			const auto size = Libs::LibKernel::Memory::ClampRangeSize(
			    range.base_address, range.requested_end - range.base_address);
			range.acquired_end = range.base_address + size;
		}
		uint32_t unassigned = 0;
		if (!DrawPrep::AssignVertexRanges(vs_input_info, collected, unassigned)) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vs_input_info.buffers[unassigned].addr);
		}
		if (verify) {
			DrawPrep::CountBindingVerifyCheck();
			if (!SameVertexRanges(*planned, collected)) {
				DrawPrep::ReportBindingMismatch("vertex ranges", collected.merged_count);
			}
		}
		ranges = &collected;
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	struct Acquired {
		Buffer*  buffer;
		uint64_t offset;
	};
	std::array<Acquired, ShaderVertexInputInfo::RES_MAX> acquired; // [0, merged_count) written
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range           = ranges->merged[i];
		const auto  size            = range.acquired_end - range.base_address;
		const auto [handle, offset] = cache.ObtainBuffer(range.base_address, size, false);
		acquired[i]                 = {handle, offset};
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, handle->Handle(),
		    "Kyty.VertexBufferRange[guest=0x{:016x} size=0x{:x}]", range.base_address, size);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = ranges->sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto  index = ranges->merged_index[i];
		const auto& range = ranges->merged[index];
		prepared.buffers[i] = acquired[index].buffer->Handle();
		prepared.offsets[i] = acquired[index].offset + vertex.addr - range.base_address;
		prepared.sizes[i]   = std::min(size, range.acquired_end - vertex.addr);
		SetVulkanObjectNameF(
		    buffer.GetContext().GetGraphics().device, prepared.buffers[i],
		    "Kyty.VertexBuffer[slot={} guest=0x{:016x} size=0x{:x} stride={} records={}]", i,
		    vertex.addr, size, vertex.stride, vertex.num_records);
	}
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              const DrawEmitInfo& emit, uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count,
	                    emit.indirect != nullptr ? emit.indirect->args_addr : draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

// Draw-prep (drawPrep.h): the early returns of DrawIndex/DrawAuto before PrepareDrawRenderState,
// evaluated on a register snapshot (target operations as their pure superset).
bool DrawPrep::DrawReachesPrograms(const HW::Context& context, const HW::UserConfig& user_config,
                                   const HW::Shader& shaders, uint32_t count,
                                   uint32_t instance_count) {
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	return count != 0 && instance_count != 0 && !MayRunTargetOperation(context) &&
	       DrawHasValidVertexShader(shaders) && GetDrawTopology(user_config, topology);
}

bool DrawPrep::DrawTopology(const HW::UserConfig& user_config, vk::PrimitiveTopology& topology) {
	return GetDrawTopology(user_config, topology);
}

bool DrawPrep::MeshPrimitiveRestartEnabled() {
	return MeshRestartEnabled();
}

DrawPrep::RestartDecision DrawPrep::DecidePrimitiveRestart(const HW::UserConfig& user_config,
                                                           const HW::Context&    registers,
                                                           uint32_t element_size, bool allow_custom) {
	const auto control = user_config.GetPrimitiveResetControl();
	if ((control & ~0x3u) != 0) {
		return RestartDecision::Unsupported;
	}
	if ((control & 0x1u) == 0) {
		return RestartDecision::Disabled;
	}
	switch (user_config.GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return RestartDecision::Disabled;
	}

	const auto index_mask  = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index = registers.GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return RestartDecision::Disabled;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask || allow_custom) {
		// Native assembly uses the maximum marker; the mesh path also accepts custom values.
		return RestartDecision::Enabled;
	}
	return RestartDecision::Scan;
}

uint32_t DrawPrep::PrimitiveRestartIndex(const HW::Context& registers, uint32_t element_size) {
	return registers.GetPrimitiveResetIndex() & (UINT32_MAX >> ((4 - element_size) * 8));
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source, bool allow_custom = false) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	const auto element_size = source.guest_element_size;
	switch (DrawPrep::DecidePrimitiveRestart(buffer.GetUserConfig(), buffer.GetRegisters(),
	                                         element_size, allow_custom)) {
		case DrawPrep::RestartDecision::Disabled:
		case DrawPrep::RestartDecision::Unsupported: return false;
		case DrawPrep::RestartDecision::Enabled: return true;
		case DrawPrep::RestartDecision::Scan: break;
	}
	const auto restart_index = DrawPrep::PrimitiveRestartIndex(buffer.GetRegisters(), element_size);

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before final binding preparation: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

// KYTY_RENDER_STATE_FAST vertex (renderDraw.h): copies a vertex input as the used prefixes of its
// arrays (and of each buffer's attribute list) and the fields after the arrays, and resets the
// entries the destination used past the source's counts. Entries past the counts hold their
// default values in every vertex input this copies from or into: the program preparations
// value-initialise the whole structure and fill only these prefixes
// (ShaderGetStaticVertexInputInfo; a failed preparation is not used), the draw state starts
// value-initialised, and this copy keeps it so. The destination then holds the bytes of a full
// copy (KYTY_RENDER_STATE_VERIFY compares them).
bool CopyVertexInputPrefixes(ShaderVertexInputInfo& dst, const ShaderVertexInputInfo& src) {
	static_assert(std::is_trivially_copyable_v<ShaderVertexInputInfo>);
	constexpr int MaxEntries    = ShaderVertexInputInfo::RES_MAX;
	constexpr int MaxAttributes = ShaderVertexInputBuffer::ATTR_MAX;
	const auto counts_valid = [](const ShaderVertexInputInfo& info) {
		if (info.resources_num < 0 || info.resources_num > MaxEntries || info.buffers_num < 0 ||
		    info.buffers_num > MaxEntries) {
			return false;
		}
		for (int i = 0; i < info.buffers_num; i++) {
			if (info.buffers[i].attr_num < 0 || info.buffers[i].attr_num > MaxAttributes) {
				return false;
			}
		}
		return true;
	};
	if (!counts_valid(src) || !counts_valid(dst)) {
		return false;
	}
	// The three arrays lead the structure; everything from `stage` on is copied as one block.
	const auto* base = reinterpret_cast<const std::byte*>(&src);
	const auto  tail = sizeof(src.resources) + sizeof(src.resources_dst) + sizeof(src.buffers);
	EXIT_IF(reinterpret_cast<const std::byte*>(&src.resources) != base ||
	        reinterpret_cast<const std::byte*>(&src.resources_dst) != base + sizeof(src.resources) ||
	        reinterpret_cast<const std::byte*>(&src.buffers) !=
	            base + sizeof(src.resources) + sizeof(src.resources_dst) ||
	        reinterpret_cast<const std::byte*>(&src.stage) != base + tail);

	const int resources     = src.resources_num;
	const int old_resources = dst.resources_num;
	std::copy_n(src.resources, resources, dst.resources);
	std::copy_n(src.resources_dst, resources, dst.resources_dst);
	for (int i = resources; i < old_resources; i++) {
		dst.resources[i]     = {};
		dst.resources_dst[i] = {};
	}
	const int buffers     = src.buffers_num;
	const int old_buffers = dst.buffers_num;
	for (int i = 0; i < std::max(buffers, old_buffers); i++) {
		auto&     to        = dst.buffers[i];
		const int old_attrs = i < old_buffers ? to.attr_num : 0;
		int       new_attrs = 0;
		if (i < buffers) {
			const auto& from = src.buffers[i];
			new_attrs        = from.attr_num;
			to.addr          = from.addr;
			to.stride        = from.stride;
			to.num_records   = from.num_records;
			to.fetch_index   = from.fetch_index;
			to.attr_num      = from.attr_num;
			std::copy_n(from.attr_indices, new_attrs, to.attr_indices);
			std::copy_n(from.attr_offsets, new_attrs, to.attr_offsets);
		} else {
			to.addr        = 0;
			to.stride      = 0;
			to.num_records = 0;
			to.fetch_index = 0;
			to.attr_num    = 0;
		}
		if (old_attrs > new_attrs) {
			std::fill(to.attr_indices + new_attrs, to.attr_indices + old_attrs, 0);
			std::fill(to.attr_offsets + new_attrs, to.attr_offsets + old_attrs, 0u);
		}
	}
	std::memcpy(reinterpret_cast<std::byte*>(&dst) + tail, base + tail,
	            sizeof(ShaderVertexInputInfo) - tail);
	return true;
}

static void CopyPreparedVertexInfo(ShaderVertexInputInfo& dst, const ShaderVertexInputInfo& src) {
	if (!RenderStateFastEnabled(RenderStatePart::VertexCopy)) {
		dst = src;
		return;
	}
	if (CopyVertexInputPrefixes(dst, src)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVertexPartialCopies);
	} else {
		dst = src;
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVertexFullCopies);
	}
	if (RenderStateVerifyMode() != 0) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
		if (std::memcmp(&dst, &src, sizeof(dst)) != 0) {
			ReportRenderStateMismatch("a partial vertex input copy");
			dst = src;
		}
	}
}

// KYTY_RENDER_STATE_FAST swap: exchanges two stage preparations member by member. The bindings name
// every member in declaration order, so a new member stops this compiling until it is swapped.
static void SwapStagePrepMembers(PipelineCache::StagePrep& a, PipelineCache::StagePrep& b) noexcept {
	auto& [a_resources, a_specialization, a_permutation] = a;
	auto& [b_resources, b_specialization, b_permutation] = b;
	auto& [a_buffers, a_images, a_samplers, a_srt, a_user_data, a_fill] = a_resources;
	auto& [b_buffers, b_images, b_samplers, b_srt, b_user_data, b_fill] = b_resources;
	auto& [a_spec_buffers, a_spec_images] = a_specialization;
	auto& [b_spec_buffers, b_spec_images] = b_specialization;
	a_buffers.swap(b_buffers);
	a_images.swap(b_images);
	a_samplers.swap(b_samplers);
	a_srt.swap(b_srt);
	a_user_data.swap(b_user_data);
	std::swap(a_fill, b_fill);
	a_spec_buffers.swap(b_spec_buffers);
	a_spec_images.swap(b_spec_images);
	std::swap(a_permutation, b_permutation);
}

static bool SameStagePrep(const PipelineCache::StagePrep& a, const PipelineCache::StagePrep& b) {
	const auto& x = a.resources;
	const auto& y = b.resources;
	return x.buffers == y.buffers && x.images == y.images && x.samplers == y.samplers &&
	       x.flattened_srt == y.flattened_srt && x.user_data == y.user_data &&
	       x.uniform_fill == y.uniform_fill && a.specialization == b.specialization &&
	       a.permutation == b.permutation;
}

static void SwapPreparedStage(PipelineCache::StagePrep& state, PipelineCache::StagePrep& prepared) {
	if (!RenderStateFastEnabled(RenderStatePart::PrepSwap)) {
		std::swap(state, prepared);
		return;
	}
	if (RenderStateVerifyMode() == 0) {
		SwapStagePrepMembers(state, prepared);
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMemberSwaps);
		return;
	}
	const auto state_before    = state;
	const auto prepared_before = prepared;
	SwapStagePrepMembers(state, prepared);
	Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateMemberSwaps);
	Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
	if (!SameStagePrep(state, prepared_before) || !SameStagePrep(prepared, state_before)) {
		ReportRenderStateMismatch("a member-wise stage preparation swap");
		state    = prepared_before;
		prepared = state_before;
	}
}

// Draw-prep: installs a validated preparation as if GetGraphicsPrograms had produced it. The
// stage preps are swapped (the preparation keeps the old vectors' capacity) and the stage
// runtimes re-pointed at the draw state's copies. Like GetGraphicsPrograms, a draw without an
// active pixel shader leaves the pixel prep untouched.
static void ApplyPreparedDraw(DrawPrep::PreparedDraw& prepared, DrawRenderState& state) {
	state.programs = prepared.programs;
	CopyPreparedVertexInfo(state.vertex_info[0], prepared.vertex_info);
	state.ps_input_info = prepared.pixel_info;
	SwapPreparedStage(state.stage_preps.vertex[0], prepared.vertex_prep);
	state.vertex_info[0].stage.resources = &state.stage_preps.vertex[0].resources;
	if (prepared.pixel_active) {
		SwapPreparedStage(state.stage_preps.pixel, prepared.pixel_prep);
		state.ps_input_info.stage.resources = &state.stage_preps.pixel.resources;
	}
}

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	// KYTY_RENDER_STATE_FAST reset: a validated preparation assigns both in full
	// (ApplyPreparedDraw), and the serial path below assigns the programs; it only needs the pixel
	// interface value-initialised (it leaves it alone without an active pixel shader).
	const bool reset_fast = RenderStateFastEnabled(RenderStatePart::Reset);
	if (!reset_fast) {
		state.programs      = {};
		state.ps_input_info = {};
	}
	// Draw-prep: the draw's speculative preparation, used below when its certificate holds now.
	auto&       executor = buffer.GetContext().GetRenderExecutor();
	auto* const prepared = executor.TakePreparedDraw();
	// KYTY_DRAW_PREP_BINDINGS statics: the preparation computed the export mapping from these same
	// registers (the draw's snapshot, bound for its commit).
	const auto* plan            = executor.PendingBindingPlan();
	const bool  planned_mapping = prepared != nullptr && plan != nullptr && plan->statics_valid;
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	if (planned_mapping && DrawPrep::BindingsVerifyMode() == 0) {
		std::copy(prepared->target_export_mapping.begin(), prepared->target_export_mapping.end(),
		          target_export_mapping.begin());
	} else {
		const auto output_filter = DrawColorOutputFilter(ctx);
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = ctx.GetRenderTarget(slot);
			if (rt.base.addr != 0 && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
			    (output_filter & (1u << slot)) != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		if (planned_mapping) {
			DrawPrep::CountBindingVerifyCheck();
			for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
				if (target_export_mapping[slot].packed != prepared->target_export_mapping[slot].packed) {
					DrawPrep::ReportBindingMismatch("target export mapping", slot);
					break;
				}
			}
		}
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.vertex_stages_written = PipelineCache::TessellationActive(buffer.GetUserConfig()) ? 3u : 1u;
	// Draw-prep: use the draw's speculative preparation when its certificate holds now.
	if (prepared != nullptr && DrawPrep::Validate(*prepared, state.ps_active, target_export_mapping)) {
		ApplyPreparedDraw(*prepared, state);
		// The binding plan was computed from this preparation (KYTY_DRAW_PREP_BINDINGS).
		executor.ActivateBindingPlan();
		executor.NotePreparedValidated();
		if (DrawPrep::VerifyMode() != 0) {
			// The serial preparation on copies, from the same (snapshot) registers.
			auto vertex_copy = std::make_unique<std::array<ShaderVertexInputInfo, 3>>();
			auto pixel_copy  = std::make_unique<ShaderPixelInputInfo>();
			auto prep_copy   = std::make_unique<PipelineCache::GraphicsStagePreps>();
			const auto serial = pipeline_cache.GetGraphicsPrograms(
			    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
			    target_export_mapping, state.ps_active, *vertex_copy, *pixel_copy, *prep_copy);
			(void)DrawPrep::VerifyCommitted(state.ps_active, state.programs, state.vertex_info[0],
			                                state.ps_input_info, state.stage_preps, serial,
			                                (*vertex_copy)[0], *pixel_copy, *prep_copy);
		}
		return;
	}
	if (reset_fast) {
		state.ps_input_info = {};
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    state.stage_preps);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	state.ps_active = DrawHasActivePixelShader(buffer);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::RefreshShaders");
		RefreshShaders(buffer, draw, state);
	}
	CommitStats::Mark(CommitStats::Phase::Programs);
	// A stage whose shader the recompiler skipped (an unresolvable runtime descriptor,
	// KYTY_SRT_VARIANT_READS) leaves its program empty: drop the draw.
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		// Counted and reported (the first few, then every 2048th) so a dropped draw is never silent.
		static std::atomic_uint64_t dropped {0};
		const auto                  count = dropped.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (count <= 4u || (count & 2047u) == 0u) {
			std::printf("Warning: draw #%" PRIu64 " dropped, %s program missing (%s)\n", count,
			            !state.programs.vertex[0] ? "vertex" : "pixel", draw.Name());
		}
		return false;
	}
	// KYTY_DRAW_RUN (drawPrep/drawRun.h): a continuation keeps the previous draw's targets (the
	// entries are still in `state`: the fast reset leaves them, DrawRunCandidate requires it).
	m_run_slice_offset = render_target_slice_offset;
	if (DrawRun::Enabled() && DrawRunCandidate(buffer, state, render_target_slice_offset)) {
		if (DrawRun::GetMode() == DrawRun::Mode::On) {
			state.color_count         = m_run.color_count;
			state.color_slots_written = m_run.color_slots;
			m_run_active              = true;
			DrawRunMarkBindings();
			CommitStats::Mark(CommitStats::Phase::Targets);
			return true;
		}
		m_run_verify = true;
	}
	KYTY_PROFILER_DETAIL_BLOCK("Draw::ResolveTargets");
	DrawRunTargets(buffer, draw, render_target_slice_offset, state);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	CommitStats::Mark(CommitStats::Phase::Targets);
	return true;
}

// The draw's colour and depth target resolution (PrepareDrawRenderState; a KYTY_DRAW_RUN
// continuation whose images changed resolves them late, from the same registers).
void RenderExecutor::DrawRunTargets(CommandBuffer& buffer, const DrawCallInfo& draw,
                                    uint32_t render_target_slice_offset, DrawRenderState& state) {
	state.color_count         = 0;
	state.color_slots_written = 0;
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	if (SkipInactivePixelShadersEnabled()) {
		mrt_mask &= DrawColorOutputFilter(buffer.GetRegisters());
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	// KYTY_RENDER_STATE_VERIFY with the reset part: each entry resolved over the previous draw's is
	// compared with the same resolution into a value-initialised entry (run after it, so the
	// draw's own resolution is the one without verification), which then provides the result.
	const bool verify_entries = RenderStateVerifyMode() != 0 &&
	                            RenderStateFastEnabled(RenderStatePart::Reset) &&
	                            !graphics_debug_dump_enabled();
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			state.color_slots_written = std::max(state.color_slots_written, state.color_count + 1u);
			auto& entry = state.color_info[state.color_count];
			ResolveRenderColorTarget(buffer, entry, render_target_slice_offset, slot);
			if (verify_entries) {
				RenderColorInfo reference {};
				ResolveRenderColorTarget(buffer, reference, render_target_slice_offset, slot);
				Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
				if (!SameRenderColorInfo(entry, reference)) {
					ReportRenderStateMismatch("a colour target entry resolved over the last draw's");
					entry = reference;
				}
			}
			if (entry.image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);
	if (verify_entries) {
		RenderDepthInfo reference {};
		ResolveRenderDepthTarget(buffer, reference);
		Profiler::CountFrameEvent(Profiler::FrameEvent::RenderStateVerifyChecks);
		if (!SameRenderDepthInfo(state.depth_info, reference)) {
			ReportRenderStateMismatch("the depth target entry resolved over the last draw's");
			state.depth_info = reference;
		}
	}
}

// ------------------------------------------------------------------------------------------------
// KYTY_DRAW_RUN (drawPrep/drawRun.h): run-level commit of structure-sharing draws.

// A program whose draws can form a run: it writes no guest memory and no image (buffer or image
// writes and atomics, address writes, GDS, the fault buffer), so a draw of it changes nothing a
// later draw's structure depends on and needs no barrier against the next one. Its mip-statistics
// counters (an emulator buffer, unordered atomics) are allowed.
static bool DrawRunProgramEligible(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (program.has_address_writes) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || binding.kind == Kind::FaultBuffer) {
			return false;
		}
	}
	return true;
}

void RenderExecutor::BeginDrawRun() {
	m_run_prev_valid     = m_run.valid;
	m_run.valid          = false;
	m_run_active         = false;
	m_run_verify         = false;
	m_prepared_validated = false;
	if (!m_in_engine_commit && DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity("draw outside an engine commit");
	}
}

bool RenderExecutor::DrawRunCommandUnchanged(const CommandBuffer& buffer) const {
	return buffer.Identity() == m_run.command &&
	       m_context.GetCommandScheduler().CurrentTick() == m_run.tick &&
	       buffer.ActiveRenderingSerial() != 0 &&
	       buffer.ActiveRenderingSerial() == m_run.rendering_serial;
}

RenderExecutor::DrawRunImage RenderExecutor::MakeDrawRunImage(ImageId id, bool texture) const {
	DrawRunImage mark;
	mark.id           = id;
	mark.texture      = texture;
	const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
	if (image == nullptr) {
		return mark;
	}
	mark.image          = image->backing.image;
	mark.stage          = image->backing.state.pl_stage;
	mark.access         = image->backing.state.access_mask;
	mark.layout         = image->backing.state.layout;
	mark.serial         = image->ContentSerial();
	mark.address        = image->info.data.address;
	mark.size           = image->info.data.size;
	mark.resident_first = image->resident_first;
	mark.registered     = image->registered;
	mark.single_state   = image->backing.subresource_states.empty();
	return mark;
}

// Whether a guest range lies over one of the record's attachments (either may be a null image).
bool RenderExecutor::DrawRunOverAttachment(std::span<const DrawRunImage> marks, uint64_t address,
                                           uint64_t size) {
	if (size == 0) {
		return false;
	}
	for (const auto& mark: marks) {
		if (!mark.texture && mark.size != 0 && address < mark.address + mark.size &&
		    mark.address < address + size) {
			return true;
		}
	}
	return false;
}

// A sampled image whose view the normal path would take as it is: FindTexture's rediscovery checks
// and a no-op RefreshImage (TextureBindingMemo::ViewImageReady and RefreshIsNoOp; any new refresh
// trigger added there must be added here). Null images are never refreshed.
static bool DrawRunTextureReady(const Image& image) {
	if (image.info.data.Empty()) {
		return true;
	}
	if (image.depth_id || image.info.HasStencil() || !image.IsTracked()) {
		return false;
	}
	if (image.ChunkTracked()) {
		return image.chunks.untracked_count == 0;
	}
	return image.track_addr == image.live.address && image.track_addr_end == image.live.End();
}

// Whether every image the recorded structure refers to still has the identity and state the
// previous draw left it in, with nothing the normal path's resolution would refresh (CPU-dirty or
// buffer-modified contents, a released tracking range, a rebind request) (GPU thread; images change
// only on it or, for CPU dirtiness, under the texture-cache lock on a faulting thread).
// 0 when unchanged; otherwise which mark differs and how (verify-mode detail):
// (mark index + 1) << 8 | 0x80 for a texture | the field (1 slot freed, 2 native image,
// 3 registration, 4 rebind request, 5 residency, 6 layout, 7 access, 8 stage, 9 per-subresource
// states, 10 content serial, 11 unregistered, 12 CPU-dirty, 13 buffer-modified, 14 texture refresh).
uint32_t RenderExecutor::DrawRunImagesChange(bool compare_serials, bool attachments_only,
                                           bool log_change) const {
	auto&            cache = m_context.GetTextureCache();
	std::scoped_lock lock {cache.m_lock};
	const auto&      images = cache.m_slot_images;
	for (uint32_t index = 0; index < m_run.images.size(); index++) {
		const auto& mark = m_run.images[index];
		if (attachments_only && mark.texture) {
			continue;
		}
		const auto* image = images.try_get(mark.id);
		uint32_t    field = 0;
		if (image == nullptr) {
			field = 1;
		} else if (image->backing.image != mark.image) {
			field = 2;
		} else if (image->registered != mark.registered) {
			field = 3;
		} else if (image->binding.needs_rebind) {
			field = 4;
		} else if (image->resident_first != mark.resident_first) {
			field = 5;
		} else if (image->backing.state.layout != mark.layout) {
			field = 6;
		} else if (image->backing.state.access_mask != mark.access) {
			field = 7;
		} else if (image->backing.state.pl_stage != mark.stage) {
			field = 8;
		} else if (image->backing.subresource_states.empty() != mark.single_state) {
			field = 9;
		} else if (compare_serials && image->ContentSerial() != mark.serial) {
			field = 10;
		} else if (!image->info.data.Empty() && !image->registered) {
			field = 11;
		} else if (!image->info.data.Empty() && image->IsCpuDirty()) {
			field = 12;
		} else if (!image->info.data.Empty() && image->IsBufferModified()) {
			field = 13;
		} else if (mark.texture && !DrawRunTextureReady(*image)) {
			field = 14;
		}
		if (field != 0) {
			static uint32_t logged = 0;
			if (log_change && logged++ < DrawRun::MismatchLogLimit()) {
				static const char* names[] = {"none", "slot freed", "native image", "registration",
				    "rebind", "residency", "layout", "access", "stage", "subresource state",
				    "content serial", "unregistered", "CPU dirty", "buffer modified", "texture refresh"};
				std::printf("DrawRunVerifyImage: mark=%u role=%s field=%s guest=0x%llx size=0x%llx "
				            "layout=%u->%u access=0x%llx->0x%llx stage=0x%llx->0x%llx "
				            "serial=%llu->%llu subresources=%zu feedback=%llu/%llu attachment=%llu run=%llu active=%llu\n",
				            index, mark.texture ? "texture" : "attachment", names[field],
				            (unsigned long long)mark.address, (unsigned long long)mark.size,
				            (unsigned)mark.layout, image ? (unsigned)image->backing.state.layout : 0u,
				            (unsigned long long)(VkAccessFlags2)mark.access,
				            image ? (unsigned long long)(VkAccessFlags2)image->backing.state.access_mask : 0ull,
				            (unsigned long long)(VkPipelineStageFlags2)mark.stage,
				            image ? (unsigned long long)(VkPipelineStageFlags2)image->backing.state.pl_stage : 0ull,
				            (unsigned long long)mark.serial, image ? (unsigned long long)image->ContentSerial() : 0ull,
				            image ? image->backing.subresource_states.size() : 0u,
				            image ? (unsigned long long)image->feedback_instance : 0ull,
				            image ? (unsigned long long)image->feedback_serial : 0ull,
				            image ? (unsigned long long)image->feedback_attached : 0ull,
				            (unsigned long long)m_run.rendering_serial,
				            (unsigned long long)m_context.GetCommandScheduler().Current().ActiveRenderingSerial());
				std::fflush(stdout);
			}
			return ((index + 1u) << 8u) | (mark.texture ? 0x80u : 0u) | field;
		}
	}
	return 0;
}

void RenderExecutor::DrawRunMarkBindings() {
	auto& images = m_context.GetTextureCache().m_slot_images;
	for (const auto& mark: m_run.images) {
		auto* image = images.try_get(mark.id);
		if (image == nullptr) {
			continue; // the late check falls back
		}
		if (mark.texture) {
			// BindImage (no storage bindings in a run).
			if (image->info.data.Empty()) {
				continue;
			}
			if (image->binding.is_bound) {
				image->binding.force_general |= image->binding.shader_write;
			}
			if (!Common::RendererBatchEnabled() ||
			    (!image->binding.is_bound && !image->binding.is_target)) {
				m_bound_images.push_back(mark.id);
			}
			image->binding.is_bound = true;
		} else {
			// BindRenderTarget.
			if (!Common::RendererBatchEnabled() ||
			    (!image->binding.is_bound && !image->binding.is_target)) {
				m_bound_images.push_back(mark.id);
			}
			image->binding.is_target = true;
		}
	}
}

bool RenderExecutor::DrawRunCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                      uint32_t render_target_slice_offset) {
	using DrawRun::Miss;
	auto& totals = DrawRun::GetTotals();
	totals.draws.fetch_add(1, std::memory_order_relaxed);
	if (!m_run_prev_valid || m_run_key == 0 || m_run_key != m_run.key) {
		return false;
	}
	totals.key_matches.fetch_add(1, std::memory_order_relaxed);
	if (DrawRun::ActivityEpoch() != m_run.activity) {
		DrawRun::CountMiss(Miss::Activity);
		return false;
	}
	if (buffer.Identity() != m_run.command ||
	    m_context.GetCommandScheduler().CurrentTick() != m_run.tick) {
		DrawRun::CountMiss(Miss::Command);
		return false;
	}
	if (buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != m_run.rendering_serial) {
		DrawRun::CountMiss(Miss::Instance);
		return false;
	}
	if (!m_prepared_validated) {
		DrawRun::CountMiss(Miss::Validation);
		return false;
	}
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	if (state.ps_active != m_run.ps_active || render_target_slice_offset != m_run.slice_offset ||
	    state.vertex_stages_written != 1 || state.vertex_info[0].stage.program != m_run.vertex_program ||
	    pixel != m_run.pixel_program) {
		DrawRun::CountMiss(Miss::Programs);
		return false;
	}
	totals.continued.fetch_add(1, std::memory_order_relaxed);
	return true;
}

bool RenderExecutor::DrawRunAcquireCandidate(const CommandBuffer&               buffer,
                                             const DrawRenderState&             state,
                                             std::span<PreparedBindings* const> stages,
                                             const vk::Rect2D*                  written) const {
	const auto& run = m_run;
	if (!m_run_prev_valid || !m_in_engine_commit || !run.acquire_valid || run.depth.size() != 1 ||
	    run.colors.size() != run.color_count || state.color_count != run.color_count ||
	    DrawRun::ActivityEpoch() != run.activity || buffer.Identity() != run.command ||
	    m_context.GetCommandScheduler().CurrentTick() != run.tick ||
	    buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != run.rendering_serial ||
	    (written != nullptr) != run.bounded || (written != nullptr && *written != run.written)) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], run.colors[i])) {
			return false;
		}
	}
	if (!SameRenderDepthInfo(state.depth_info, run.depth[0])) {
		return false;
	}
	// No texture of the draw is an attachment or lies over one (AcquireRenderTargets would detect a
	// feedback loop or choose other layouts; an alias is synchronized from the attachment).
	{
		auto&            cache = m_context.GetTextureCache();
		std::scoped_lock lock {cache.m_lock};
		// The record's attachments follow its textures.
		const auto attachments = std::span<const DrawRunImage>(run.images).subspan(
		    std::min<size_t>(run.attachments_begin, run.images.size()));
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				const auto* image = cache.m_slot_images.try_get(binding.image_id);
				if (image == nullptr) {
					return false;
				}
				for (const auto& mark: attachments) {
					if (!mark.texture && mark.id == binding.image_id) {
						return false;
					}
				}
				if (DrawRunOverAttachment(attachments, image->info.data.address,
				                          image->info.data.size)) {
					return false;
				}
			}
		}
	}
	return DrawRunImagesUnchanged(true, true);
}

bool RenderExecutor::DrawRunPartialPush(const CommandBuffer&           buffer,
                                        const PipelineCache::Pipeline& pipeline) const {
	return DrawRun::PushPartialEnabled() && pipeline.uses_push_descriptors && m_run.push_valid &&
	       m_run.push_layout == pipeline.pipeline_layout &&
	       buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics) == m_run.push_epoch;
}

// After an eligible draw was recorded: what a continuation of it may reuse, and its certificate.
void RenderExecutor::DrawRunRecordDraw(const CommandBuffer& buffer, const DrawRenderState& state,
                                       uint32_t render_target_slice_offset,
                                       const RenderState& rendering,
                                       std::span<PreparedBindings* const> stages,
                                       const vk::Rect2D*                  written,
                                       const PipelineCache::Pipeline&     pipeline) {
	auto& run            = m_run;
	run.key              = m_run_key;
	run.activity         = DrawRun::ActivityEpoch();
	run.command          = buffer.Identity();
	run.tick             = m_context.GetCommandScheduler().CurrentTick();
	run.rendering_serial = buffer.ActiveRenderingSerial();
	// The draw's push (whole or partial) is the last descriptor command at the bind point.
	run.push_epoch = buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics);
	if (m_run_active) {
		// A continuation changed none of the recorded images, targets or bindings.
		run.valid = true;
		return;
	}
	run.push_valid  = pipeline.uses_push_descriptors;
	run.push_layout = pipeline.pipeline_layout;
	run.slice_offset   = render_target_slice_offset;
	run.vertex_program = state.vertex_info[0].stage.program;
	run.pixel_program  = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	run.ps_active      = state.ps_active;
	run.color_count    = state.color_count;
	run.color_slots    = state.color_slots_written;
	run.rendering      = rendering;
	run.images.clear();
	{
		std::scoped_lock lock {m_context.GetTextureCache().m_lock};
		// KYTY_DEPTH_FEEDBACK_LAZY: an acquisition that does not sample the image (as a
		// continuation's or a reused acquisition's, whose textures are never an attachment) keeps
		// the attachment access, so nothing is excluded.
		if (state.depth_info.image_id && DepthFeedbackKeepEnabled() && !DepthFeedbackLazyEnabled() &&
		    !state.depth_info.AttachmentWriteAspects()) {
			const auto* depth = m_context.GetTextureCache().m_slot_images.try_get(state.depth_info.image_id);
			const auto attachment_access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
			                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
			// NoteDepthFeedback just established an unwritten-instance proof. The next acquisition
			// can now adopt attachment_access | ShaderRead even for an unsampled attachment
			// (AcquireRenderTargets). It must run normally to publish that union and its deferred
			// ordering; this first draw cannot seed either a continuation or acquisition reuse.
			if (depth != nullptr && depth->backing.state.access_mask == attachment_access &&
			    depth->feedback_instance == run.rendering_serial &&
			    depth->feedback_serial == depth->ContentSerial()) {
				DrawRun::GetTotals().depth_promotions_excluded.fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				run.images.push_back(MakeDrawRunImage(binding.image_id, true));
			}
		}
		run.attachments_begin = static_cast<uint32_t>(run.images.size());
		for (uint32_t i = 0; i < state.color_count; i++) {
			run.images.push_back(MakeDrawRunImage(state.color_info[i].image_id, false));
		}
		if (state.depth_info.image_id) {
			run.images.push_back(MakeDrawRunImage(state.depth_info.image_id, false));
		}
	}
	// A texture over an attachment's memory (an alias of it) is synchronized from the attachment by
	// each draw's texture resolution (SyncAliasFromOwner), which a continuation skips: no run.
	const auto attachments = std::span<const DrawRunImage>(run.images).subspan(run.attachments_begin);
	for (const auto& mark: run.images) {
		if (mark.texture && DrawRunOverAttachment(attachments, mark.address, mark.size)) {
			DrawRun::GetTotals().alias_excluded.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	const bool verify = DrawRun::GetMode() == DrawRun::Mode::Verify;
	run.acquire_valid = verify || DrawRun::AcquireReuseEnabled();
	run.verify_valid  = verify;
	if (run.acquire_valid) {
		run.colors.assign(state.color_info, state.color_info + state.color_count);
		run.depth.assign(1, state.depth_info);
		run.bounded = written != nullptr;
		run.written = written != nullptr ? *written : vk::Rect2D {};
	}
	if (verify) {
		run.pushed_images.assign(m_descriptor_images.begin(), m_descriptor_images.end());
		run.textures.clear();
		run.samplers.clear();
		for (const auto* stage: stages) {
			run.textures.insert(run.textures.end(), stage->images.begin(), stage->images.end());
			run.samplers.insert(run.samplers.end(), stage->samplers.begin(), stage->samplers.end());
		}
	}
	run.valid = true;
}

// KYTY_DRAW_RUN=verify: a draw that would have continued the run took the normal path; what the
// continuation would have reused must be what it computed.
void RenderExecutor::DrawRunVerify(const DrawRenderState& state, const RenderState& rendering,
                                   vk::ImageAspectFlags               feedback_aspects,
                                   std::span<PreparedBindings* const> stages) {
	const auto& run = m_run;
	if (!run.verify_valid) {
		return; // recorded before verify mode was switched on
	}
	DrawRun::CountVerifyCheck();
	if (state.color_count != run.color_count || state.color_slots_written != run.color_slots ||
	    run.depth.size() != 1) {
		DrawRun::ReportMismatch("colour target count", state.color_count);
		return;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], run.colors[i])) {
			DrawRun::ReportMismatch("colour target", i);
		}
	}
	if (!SameRenderDepthInfo(state.depth_info, run.depth[0])) {
		DrawRun::ReportMismatch("depth target");
	}
	size_t texture = 0;
	size_t sampler = 0;
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			if (texture >= run.textures.size()) {
				DrawRun::ReportMismatch("texture count", texture);
				return;
			}
			const auto& old = run.textures[texture++];
			if (binding.image_id != old.image_id || binding.image_view != old.image_view ||
			    binding.layout != old.layout || binding.desc.type != old.desc.type ||
			    binding.mip_views != old.mip_views) {
				DrawRun::ReportMismatch(binding.image_view != old.image_view || binding.mip_views != old.mip_views
				                            ? "texture views" : "texture binding", texture - 1);
			}
		}
		for (const auto handle: stage->samplers) {
			if (sampler >= run.samplers.size() || handle != run.samplers[sampler++]) {
				DrawRun::ReportMismatch("sampler", sampler);
			}
		}
	}
	if (texture != run.textures.size() || sampler != run.samplers.size()) {
		DrawRun::ReportMismatch("binding count", texture);
	}
	if (!(rendering == run.rendering)) {
		DrawRun::ReportMismatch("rendering state");
	}
	if (feedback_aspects) {
		DrawRun::ReportMismatch("attachment feedback", static_cast<uint32_t>(feedback_aspects));
	}
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=transient size=0x{:x} type={}]", source.size,
		                     static_cast<uint32_t>(source.type));
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

static void CommitVertexBuffers(const CommandSink&           vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(const CommandSink& vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, const CommandSink& vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

// CommandBuffer::DrawScope: the draw's only memory writes are its own color/depth attachments,
// and it samples none of them. Conservative: any shader write path (storage buffers/images,
// atomics, address writes, GDS, fault/LOD counters), indirect arguments or a feedback loop makes
// the draw unsafe for sinking a pending barrier past it.
// plain_pixel: the pixel stage runs its feedback-free variant (KYTY_LOD_STATS_PLAIN_VARIANT), so
// its mip-statistics binding is not written.
bool DrawPrep::ProgramBarrierSafe(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                  bool                                            plain_pixel) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (program.has_address_writes) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || binding.kind == Kind::FaultBuffer ||
		    (binding.kind == Kind::MipStats && !(plain_pixel && program.stage == ShaderType::Pixel))) {
			return false;
		}
	}
	return true;
}

// programs_safe: DrawPrep::ProgramBarrierSafe of every stage's program (a binding plan's), or null
// to check them here.
static bool DrawIsBarrierSafe(std::span<PreparedBindings* const> stages,
                              const RenderColorInfo* colors, uint32_t color_count,
                              const RenderDepthInfo& depth, vk::ImageAspectFlags feedback_aspects,
                              bool indirect, bool plain_pixel, const bool* programs_safe = nullptr) {
	if (indirect || feedback_aspects || !BarrierSinkEnabled()) {
		return false;
	}
	if (programs_safe != nullptr && !*programs_safe) {
		return false;
	}
	for (const auto* stage: stages) {
		if (stage == nullptr || stage->runtime == nullptr || !*stage->runtime) {
			return false;
		}
		if (stage->gds.buffer != nullptr ||
		    (programs_safe == nullptr &&
		     !DrawPrep::ProgramBarrierSafe(*stage->runtime->program, plain_pixel))) {
			return false;
		}
		for (const auto& image: stage->images) {
			if (depth.image_id && image.image_id == depth.image_id) {
				return false;
			}
			for (uint32_t i = 0; i < color_count; i++) {
				if (image.image_id == colors[i].image_id) {
					return false;
				}
			}
		}
	}
	return true;
}

// KYTY_DRAW_BINDING_REPEAT_STATS=1 (opt-in diagnostics, counted with aggregates only; it copies
// every draw's descriptor words): how often a draw binds the same programs and descriptors as the
// draw before it, the room a draw-level binding reuse would have for "stamps"
// (FrameEvent.DrawBindingRepeat*: same programs; also the same images and samplers; also the same
// buffers; also the same user data and flattened SRT words).
static bool BindingRepeatStatsEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_DRAW_BINDING_REPEAT_STATS");
		return Profiler::AggregateEnabled() && value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static void CountBindingRepeats(std::span<const ShaderStageRuntime* const> stages) {
	using ShaderRecompiler::IR::DescriptorValue;
	struct Previous {
		const void*                  program = nullptr;
		std::vector<DescriptorValue> buffers;
		std::vector<DescriptorValue> images;
		std::vector<DescriptorValue> samplers;
		std::vector<uint32_t>        user_data;
		std::vector<uint32_t>        flattened_srt;
	};
	static thread_local std::vector<Previous> previous;
	bool same_programs  = previous.size() == stages.size();
	bool same_textures  = same_programs;
	bool same_buffers   = same_programs;
	bool same_constants = same_programs;
	previous.resize(stages.size());
	for (size_t i = 0; i < stages.size(); i++) {
		const auto& stage     = *stages[i];
		const auto& resources = *stage.resources;
		auto&       old       = previous[i];
		same_programs &= old.program == stage.program;
		same_textures &= old.images == resources.images && old.samplers == resources.samplers;
		same_buffers &= old.buffers == resources.buffers;
		same_constants &=
		    old.user_data == resources.user_data && old.flattened_srt == resources.flattened_srt;
		old.program = stage.program;
		old.buffers.assign(resources.buffers.begin(), resources.buffers.end());
		old.images.assign(resources.images.begin(), resources.images.end());
		old.samplers.assign(resources.samplers.begin(), resources.samplers.end());
		old.user_data.assign(resources.user_data.begin(), resources.user_data.end());
		old.flattened_srt.assign(resources.flattened_srt.begin(), resources.flattened_srt.end());
	}
	if (!same_programs) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatPrograms);
	if (!same_textures) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatTextures);
	if (!same_buffers) {
		return;
	}
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatResources);
	if (same_constants) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawBindingRepeatAll);
	}
}

// KYTY_CP_COMMIT_STATS (commitStats.h): what a recorded draw bound, as hashes.
namespace {

class ShapeWords {
public:
	void Add(uint64_t word) { m_words.push_back(word); }
	void AddFloat(float value) { m_words.push_back(std::bit_cast<uint32_t>(value)); }
	template <typename Handle>
	void AddHandle(Handle handle) {
		m_words.push_back(static_cast<uint64_t>(
		    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle))));
	}
	void AddBufferInfo(const vk::DescriptorBufferInfo& info) {
		AddHandle(info.buffer);
		Add(info.offset);
		Add(info.range);
	}
	void AddValue(const ShaderRecompiler::IR::DescriptorValue& value, bool mask_address) {
		Add(value.dword_count);
		for (uint32_t i = 0; i < value.dword_count && i < value.dwords.size(); i++) {
			auto word = value.dwords[i];
			if (mask_address && i == 0) {
				word = 0;
			} else if (mask_address && i == 1) {
				word &= 0xffff0000u;
			}
			Add(word);
		}
	}
	[[nodiscard]] uint64_t Finish() {
		const auto hash = CommitStats::Hash(m_words.data(), m_words.size() * sizeof(uint64_t));
		m_words.clear();
		return hash;
	}

private:
	std::vector<uint64_t> m_words;
};

} // namespace

static CommitStats::DrawShape MakeCommitShape(
    const CommandBuffer& buffer, std::span<PreparedBindings* const> stages,
    const PipelineCache::Pipeline& pipeline, const RenderState& rendering,
    const GraphicsDynamicStateShadow& dynamic, const PreparedVertexBuffers& vertex,
    const PreparedIndexBuffer& index, const DrawCallInfo& draw, const DrawEmitInfo& emit,
    const DrawIndexBufferSource& index_source, vk::PipelineStageFlags shader_write_stages) {
	static thread_local ShapeWords words;
	CommitStats::DrawShape shape;
	shape.rendering_serial = buffer.ActiveRenderingSerial();
	shape.command = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
	    static_cast<VkCommandBuffer>(buffer.Identity())));
	shape.pipeline = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
	    static_cast<VkPipeline>(pipeline.pipeline)));
	shape.shader_writes = static_cast<bool>(shader_write_stages);
	// Structure: programs, texture views and layouts, samplers, attachments.
	for (const auto* stage: stages) {
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& image: stage->images) {
			words.AddHandle(image.image_view);
			words.Add(static_cast<uint64_t>(image.layout));
			for (const auto view: image.mip_views) {
				words.AddHandle(view);
			}
		}
		for (const auto sampler: stage->samplers) {
			words.AddHandle(sampler);
		}
	}
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		words.AddHandle(rendering.color_attachments[i].image_view);
		words.Add(static_cast<uint64_t>(rendering.color_attachments[i].image_layout));
	}
	words.AddHandle(rendering.depth_stencil_attachment.image_view);
	words.Add(static_cast<uint64_t>(rendering.depth_stencil_attachment.image_layout));
	shape.structure = words.Finish();
	// Dynamic state.
	words.Add(dynamic.viewport_count);
	for (uint32_t i = 0; i < dynamic.viewport_count && i < dynamic.viewports.size(); i++) {
		const auto& v = dynamic.viewports[i];
		words.AddFloat(v.x);
		words.AddFloat(v.y);
		words.AddFloat(v.width);
		words.AddFloat(v.height);
		words.AddFloat(v.minDepth);
		words.AddFloat(v.maxDepth);
		const auto& s = dynamic.scissors[i];
		words.Add(static_cast<uint32_t>(s.offset.x));
		words.Add(static_cast<uint32_t>(s.offset.y));
		words.Add(s.extent.width);
		words.Add(s.extent.height);
	}
	words.AddFloat(dynamic.line_width);
	for (const auto value: dynamic.blend_constants) {
		words.AddFloat(value);
	}
	words.Add(dynamic.depth_test_enable);
	words.Add(dynamic.depth_write_enable);
	words.Add(static_cast<uint64_t>(dynamic.depth_compare_op));
	words.Add(dynamic.depth_bias_enable);
	for (const auto value: dynamic.depth_bias) {
		words.AddFloat(value);
	}
	words.Add(dynamic.stencil_test_enable);
	for (const auto* op: {&dynamic.stencil_front, &dynamic.stencil_back}) {
		words.Add(static_cast<uint64_t>(op->failOp));
		words.Add(static_cast<uint64_t>(op->passOp));
		words.Add(static_cast<uint64_t>(op->depthFailOp));
		words.Add(static_cast<uint64_t>(op->compareOp));
		words.Add(op->compareMask);
		words.Add(op->writeMask);
		words.Add(op->reference);
	}
	words.Add(dynamic.color_write_count);
	for (uint32_t i = 0; i < dynamic.color_write_count && i < dynamic.color_write.size(); i++) {
		words.Add(dynamic.color_write[i]);
	}
	words.Add(static_cast<uint32_t>(dynamic.feedback));
	words.Add(static_cast<uint32_t>(dynamic.cull_mode));
	words.Add(static_cast<uint64_t>(dynamic.front_face));
	words.Add(dynamic.depth_bounds_test_enable);
	words.AddFloat(dynamic.depth_bounds[0]);
	words.AddFloat(dynamic.depth_bounds[1]);
	shape.dynamic = words.Finish();
	// Per-draw data.
	for (const auto* stage: stages) {
		for (const auto& info: stage->buffers) {
			words.AddBufferInfo(info);
		}
	}
	shape.buffers = words.Finish();
	for (const auto* stage: stages) {
		for (const auto dword: stage->shader_data) {
			words.Add(dword);
		}
	}
	shape.push = words.Finish();
	for (const auto* stage: stages) {
		words.AddBufferInfo(stage->flattened_srt);
		words.AddBufferInfo(stage->shader_data_buffer);
	}
	shape.tables = words.Finish();
	words.AddHandle(index.buffer);
	words.Add(index.offset);
	words.Add(static_cast<uint64_t>(index.type));
	shape.index = words.Finish();
	for (uint32_t i = 0; i < vertex.count; i++) {
		words.AddHandle(vertex.buffers[i]);
		words.Add(vertex.offsets[i]);
		words.Add(vertex.sizes[i]);
	}
	shape.vertex = words.Finish();
	// Cross-frame keys: the complete binding inputs, and the structure alone.
	for (const auto* stage: stages) {
		const auto& resources = *stage->runtime->resources;
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& value: resources.buffers) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.images) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.samplers) {
			words.AddValue(value, false);
		}
		for (const auto dword: resources.flattened_srt) {
			words.Add(dword);
		}
		for (const auto dword: resources.user_data) {
			words.Add(dword);
		}
	}
	words.Add(draw.index_count);
	words.Add(draw.instance_count);
	words.Add(static_cast<uint32_t>(emit.vertex_offset));
	words.Add(emit.first_vertex);
	words.Add(emit.first_instance);
	words.Add(index_source.address);
	shape.bind_input = words.Finish();
	words.Add(shape.pipeline);
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		words.AddHandle(rendering.color_attachments[i].image_view);
	}
	words.AddHandle(rendering.depth_stencil_attachment.image_view);
	for (const auto* stage: stages) {
		const auto& resources = *stage->runtime->resources;
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& value: resources.buffers) {
			words.AddValue(value, true);
		}
		for (const auto& value: resources.images) {
			words.AddValue(value, false);
		}
		for (const auto& value: resources.samplers) {
			words.AddValue(value, false);
		}
	}
	words.Add(draw.index_count);
	words.Add(draw.instance_count);
	shape.struct_input = words.Finish();
	return shape;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	KYTY_GPU_OP_SITE("draw.execute");
	KYTY_PROFILER_DETAIL_FUNCTION();
	auto& ucfg = buffer.GetUserConfig();
	if (m_context.GetOcclusionCounter().DebugLogActive()) {
		OcclusionCounter::DebugLog("e n=%u inst=%u vs=0x%" PRIx64 "\n", draw.index_count, draw.instance_count,
		                           buffer.GetShaders().GetVs().es_regs.data_addr);
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	if (m_context.GetPipelineCache().PipelinePrefetchEnabled() &&
	    (m_binding_plan == nullptr || m_binding_plan->pipeline == nullptr)) {
		m_context.GetPipelineCache().PrefetchGraphicsPipeline(
		    {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
		    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
		    state.programs);
	}
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	// Reused per thread (draws run on the GPU thread under the render mutex).
	static thread_local std::vector<MeshDrawSegment> mesh_segments;
	mesh_segments.clear();
	// KYTY_NATIVE_INDIRECT_MESH: the counts are GPU data, converted into the dispatches right
	// before the draw (meshIndirect.h); nothing below may read them.
	const bool mesh_indirect = mesh_active && emit.mesh_indirect;
	EXIT_IF(emit.mesh_indirect && (!mesh_active || emit.indirect == nullptr));
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !MeshRestartEnabled() && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
	}
	if (mesh_active && !mesh_indirect) {
		const auto& mesh = state.vertex_info[0].mesh;
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (primitive_restart_enable && draw.IsIndexed() && MeshRestartEnabled()) {
			const auto mask = UINT32_MAX >> ((4u - index_source.guest_element_size) * 8u);
			SplitMeshRestartIndices(index_source.address, draw.index_count,
			    index_source.guest_element_size, buffer.GetRegisters().GetPrimitiveResetIndex() & mask, mesh,
			    mesh_segments);
			if (mesh_segments.empty()) return;
		} else {
			mesh_segments.push_back({0, draw.index_count, mesh_groups});
		}
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		uint64_t total_groups = 0;
		for (const auto& segment: mesh_segments) {
			// More instances than one dispatch carries are split when the draw is recorded, and
			// with split_groups (MeshGroupsPerDispatch) so are more groups than X allows.
			if (MeshInstancesPerDispatch(MeshGroupsPerDispatch(segment.groups, mesh, limits),
			                             limits) == 0) {
				EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", segment.groups, draw.instance_count);
			}
			total_groups += segment.groups;
		}
		if (Profiler::AggregateEnabled()) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshDraws);
			if (primitive_restart_enable) {
				// Enabled state only; this does not inspect indices for actual restart markers.
				Profiler::CountFrameEvent(Profiler::FrameEvent::MeshRestartEnabledDraws);
			}
			if (draw.IsIndexed()) {
				Profiler::CountFrameEvent(
				    Profiler::FrameEvent::MeshInputIndices,
				    static_cast<uint64_t>(draw.index_count) * draw.instance_count);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshWorkgroups,
			                          total_groups * draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed. A native
		// indirect draw registers the bound range [INDEX_BASE, +INDEX_BUFFER_SIZE) (its count and
		// start index are GPU data).
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, mesh_indirect ? index_source.size
		                                        : static_cast<uint64_t>(draw.index_count) *
		                                              index_source.guest_element_size);
	}
	const auto* indirect = emit.indirect;
	EXIT_IF(indirect != nullptr && mesh_active && !mesh_indirect);
	if (indirect != nullptr) {
		// ObtainBuffer can merge cache buffers, which must not retire buffers that the shader
		// bindings below already resolved. Acquire the argument, count and whole index ranges
		// first; acquiring them again after the bindings then finds the same or a merged,
		// still covering, buffer and merges nothing. Mesh shaders read indices by address (the
		// range registered above), not through an index buffer binding.
		(void)ObtainIndirectBuffers(buffer, *indirect);
		if (index_source.size != 0 && !mesh_active) {
			(void)m_context.GetBufferCache().ObtainBuffer(index_source.address, index_source.size,
			                                              false);
		}
	}
	if (BindingRepeatStatsEnabled()) [[unlikely]] {
		std::array<const ShaderStageRuntime*, 4> runtimes {};
		uint32_t                                 runtime_count = 0;
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			runtimes[runtime_count++] = &state.vertex_info[i].stage;
		}
		if (state.ps_active) {
			runtimes[runtime_count++] = &state.ps_input_info.stage;
		}
		CountBindingRepeats(std::span {runtimes.data(), runtime_count});
	}
	// KYTY_DRAW_PREP_BINDINGS: the committed draw's binding plan (RefreshShaders activated it once
	// DrawPrep::Validate accepted the preparation it was computed from). Its stage plans cover a
	// single vertex stage and the pixel stage.
	auto* const plan        = m_binding_plan_active ? m_binding_plan : nullptr;
	const bool  plan_verify = plan != nullptr && DrawPrep::BindingsVerifyMode() != 0;
	const bool  plan_stages = plan != nullptr && vertex_stages.size() == 1;
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareBindings");
		// KYTY_DRAW_RUN continuation: the stages keep the previous draw's textures and samplers.
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i],
			                plan_stages ? &plan->vertex : nullptr, m_run_active);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel,
			                plan_stages && plan->pixel_active ? &plan->pixel : nullptr,
			                m_run_active);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
	}
	CommitStats::Mark(CommitStats::Phase::PrepareBindings);
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareGraphicsBindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count},
		                        m_run_active);
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		// KYTY_DRAW_PREP_BINDINGS buffers: the plan's ranges while the guest virtual ranges are the
		// ones its clamps saw.
		const DrawPrep::VertexRangePlan* vertex_ranges = nullptr;
		if (plan_stages && plan->vertex_ranges.valid &&
		    plan->vertex_ranges.buffer_count ==
		        static_cast<uint32_t>(state.vertex_info[0].buffers_num)) {
			if (plan->vm_generation == LibKernel::Memory::VirtualRangesGeneration()) {
				vertex_ranges = &plan->vertex_ranges;
				DrawPrep::GetBindingTotals().vertex_ranges_used.fetch_add(1, std::memory_order_relaxed);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingFallbackVm);
			}
		}
		if (CpCommit::Enabled(CpCommit::Part::Draws)) {
			AcquireVertexBuffersInto(buffer, state.vertex_info[0], vertex_ranges, vertex_bindings);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], vertex_ranges);
		}
		index_binding = PrepareIndexBuffer(buffer, index_source);
	}
	PreparedIndirectBuffers indirect_buffers;
	MeshIndirect::Converter::Slot mesh_slot;
	if (mesh_indirect) {
		// Still preparation: a ring wrap may submit the recording here.
		if (m_mesh_indirect == nullptr) {
			m_mesh_indirect = std::make_unique<MeshIndirect::Converter>(m_context);
		}
		mesh_slot = m_mesh_indirect->Reserve();
	}
	if (indirect != nullptr) {
		indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
	}
	CommitStats::Mark(CommitStats::Phase::VertexIndex);
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// All per-draw buffer work is now complete, including vertex/index/indirect acquisition. It
	// can end rendering without changing a command buffer, tick or image state (buffer uploads /
	// barriers). The rendering instance is part of the certificate, not just image identity.
	// Unless every kept image is as the previous draw left it, the structure is resolved now, in the normal order relative to the buffer
	// bindings, which are made again after it (their reservations follow the image identities).
	// KYTY_DRAW_RUN_INDIRECT: an indirect continuation also needs the argument barrier of this
	// rendering instance (recording one ends rendering, which a continuation never does).
	const bool run_indirect_barrier =
	    indirect != nullptr && buffer.ActiveRenderingSerial() != m_indirect_barrier_rendering;
	if (m_run_active && (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true) ||
	                     run_indirect_barrier)) {
		DrawRun::GetTotals().late_fallbacks.fetch_add(1, std::memory_order_relaxed);
		DrawRun::CountMiss(DrawRun::Miss::Images);
		m_run_active = false;
		DrawRunTargets(buffer, draw, m_run_slice_offset, state);
		// No plan: its shader data was taken by the first preparation above.
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i], nullptr);
		}
		if (state.ps_active) {
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel, nullptr);
		}
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
		// The repeated binding work may merge/retire cache buffers or restart the command buffer.
		// Reacquire the per-draw vertex/index reservations after it, just as on the normal path,
		// into fresh bindings: the first acquisition above filled them (AcquireVertexBuffersInto
		// requires value-initialised bindings and stopped the emulator for a draw with vertex
		// buffers).
		vertex_bindings = PreparedVertexBuffers {};
		if (CpCommit::Enabled(CpCommit::Part::Draws)) {
			AcquireVertexBuffersInto(buffer, state.vertex_info[0], nullptr, vertex_bindings);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], nullptr);
		}
		index_binding = PrepareIndexBuffer(buffer, index_source);
		if (indirect != nullptr) {
			indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
		}
	}
	// KYTY_DRAW_RUN=verify: whether the kept images would have passed the check above (the normal
	// path's own texture resolution ran before it here). A draw that would have fallen back is
	// not compared.
	if (m_run_verify && (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true) ||
	                     run_indirect_barrier)) {
		DrawRun::GetTotals().late_fallbacks.fetch_add(1, std::memory_order_relaxed);
		DrawRun::CountMiss(DrawRun::Miss::Images);
		m_run_verify = false;
	}

	// KYTY_LOD_STATS_PLAIN_VARIANT: without a mip-statistics counter on any image the pixel
	// program's feedback records nothing; its feedback-free variant then gives the same results
	// without forcing the depth/stencil tests after a shader that can discard.
	const auto* programs    = &state.programs;
	bool        plain_pixel = false;
	PipelineCache::GraphicsPrograms plain_programs;
	if (state.ps_active && bindings.pixel.has_value()) {
		const auto plain_mode = LodStatsCounter::PlainVariant();
		const auto plain      = plain_mode != LodStatsCounter::Plain::Off
		                            ? PipelineCache::PlainPixelProgram(state.stage_preps.pixel)
		                            : ShaderProgram {};
		if (plain) {
			if (bindings.pixel->mip_stats_active) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsInstrumentedDraws);
			} else if (plain_mode == LodStatsCounter::Plain::On) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsPlainDraws);
				plain_programs       = state.programs;
				plain_programs.pixel = plain;
				programs             = &plain_programs;
				plain_pixel          = true;
			} else if (m_context.GetLodStats().CanaryReady()) {
				// Verify: the instrumented program records into the canary, which must stay reset.
				Profiler::CountFrameEvent(Profiler::FrameEvent::LodStatsPlainDraws);
				bindings.pixel->mip_stats_canary = true;
			}
		}
	}
	// KYTY_DRAW_PREP_BINDINGS pipeline: the object the preparing thread found for this key, when
	// the draw's targets, topology, restart flag and plain-pixel choice are the ones the key was
	// built for and no pipeline was replaced since (DrawPrep::PlannedPipeline).
	auto&       pipeline_cache = m_context.GetPipelineCache();
	const auto  colors_span    = std::span<const RenderColorInfo> {state.color_info, state.color_count};
	const auto* ps_input       = state.ps_active ? &state.ps_input_info : nullptr;
	const auto* planned_pipeline =
	    plan_stages && plan->pipeline != nullptr
	        ? DrawPrep::PlannedPipeline(pipeline_cache, *plan, colors_span, state.depth_info,
	                                    topology, primitive_restart_enable, plain_pixel)
	        : nullptr;
	const PipelineCache::Pipeline* pipeline_object = planned_pipeline;
	if (planned_pipeline != nullptr && !plan_verify) {
		pipeline_cache.NotePlannedPipeline(state.depth_info, ps_input);
	} else {
		pipeline_object = &pipeline_cache.GetGraphicsPipeline(
		    colors_span, state.depth_info, vertex_stages, buffer, ps_input, topology,
		    primitive_restart_enable, *programs);
		if (planned_pipeline != nullptr) {
			DrawPrep::CountBindingVerifyCheck();
			// A pipeline replaced after the certificate check is a race, not a difference.
			if (pipeline_object != planned_pipeline &&
			    pipeline_cache.PipelineGeneration() == plan->pipeline_generation) {
				DrawPrep::ReportBindingMismatch("pipeline");
			}
		}
	}
	const auto& pipeline = *pipeline_object;
	CommitStats::Mark(CommitStats::Phase::Pipeline);
	if (mesh_indirect) {
		// The conversion of the argument record into this draw's dispatches and draw dwords:
		// outside rendering (it ends the active instance, before the targets are acquired for the
		// new one), after every earlier write, before the draw (MeshIndirect::Converter::Record).
		m_mesh_indirect->Record(buffer, mesh_slot, emit.mesh_inputs, indirect_buffers.args,
		                        indirect_buffers.args_offset);
	}
	vk::ImageAspectFlags feedback_aspects;
	// KYTY_DRAW_PREP_BINDINGS statics: the plan's program flags and scissor union.
	const bool plan_statics = plan_stages && plan->statics_valid;
	RenderState rendering;
	// KYTY_DRAW_RUN=verify: the continuation's skipped acquisition and transitions must be no-ops.
	const bool run_verify_images   = m_run_verify && DrawRunImagesUnchanged(false);
	const auto run_transitions     = m_run_verify ? Image::RecordedTransitions() : uint64_t {0};
	// Where this draw can write its colour targets (KYTY_ALIAS_BYTES ownership claims); a
	// continuation's are the recorded draw's.
	vk::Rect2D written = m_run_active ? m_run.written : vk::Rect2D {};
	const bool bounded = TextureCache::AliasBytesEnabled();
	if (bounded && !m_run_active) {
		if (plan_statics && plan->written_valid && !plan_verify) {
			written = plan->written;
		} else {
			const auto& outputs = vertex_stages.back().stage.program->info.outputs;
			written             = DrawScissorUnion(
	            buffer.GetRegisters(), std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
	                return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	            }));
			if (plan_statics && plan->written_valid) {
				DrawPrep::CountBindingVerifyCheck();
				if (written != plan->written) {
					DrawPrep::ReportBindingMismatch("scissor union");
				}
			}
		}
	}
	if (m_run_active) {
		// KYTY_DRAW_RUN continuation: the attachments were acquired, claimed and transitioned by the
		// previous draw of the run, for the same targets, scissors and textures (no feedback), and
		// are still in that state (DrawRunImagesUnchanged).
		rendering              = m_run.rendering;
		m_depth_feedback.valid = false;
	} else {
		// KYTY_DRAW_RUN_ACQUIRE: the same targets and scissor union right after the recorded draw, in
		// its rendering instance, with the attachments as it left them: its acquisition, whose
		// transitions, claims, refreshes and depth-feedback bookkeeping would all be repeats.
		// (Verify mode checks a would-be continuation's acquisition as the continuation's.)
		const bool acquire_reuse =
		    DrawRun::Enabled() && DrawRun::AcquireReuseEnabled() && !m_run_verify &&
		    DrawRunAcquireCandidate(buffer, state, stages, bounded ? &written : nullptr);
		if (acquire_reuse) {
			DrawRun::GetTotals().acquire_reused.fetch_add(1, std::memory_order_relaxed);
		}
		if (acquire_reuse && DrawRun::GetMode() == DrawRun::Mode::On) {
			rendering              = m_run.rendering;
			m_depth_feedback.valid = false;
		} else {
			const auto acquire_transitions =
			    acquire_reuse ? Image::RecordedTransitions() : uint64_t {0};
			rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
			                                 state.depth_info, feedback_aspects, stages,
			                                 bounded ? &written : nullptr);
			if (acquire_reuse) {
				// Verify mode: the acquisition a reuse skips must have been a repeat.
				DrawRun::CountVerifyCheck();
				if (!(rendering == m_run.rendering) || feedback_aspects) {
					DrawRun::ReportMismatch("reused acquisition's rendering state");
				}
				if (Image::RecordedTransitions() != acquire_transitions) {
					DrawRun::ReportMismatch("reused acquisition's transitions",
					                        DrawRunImagesChange(false, true, true));
				}
				if (const auto change = DrawRunImagesChange(false, true, true); change != 0) {
					DrawRun::ReportMismatch("reused acquisition's attachment state", change);
				}
			}
			if (run_verify_images) {
				if (const auto change = DrawRunImagesChange(false, false, true); change != 0) {
					DrawRun::CountVerifyCheck();
					DrawRun::ReportMismatch("attachment state after acquisition", change);
				}
			}
		}
	}
	CommitStats::Mark(CommitStats::Phase::AcquireTargets);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Until BeginRendering() only state commands are recorded through vk_buffer; image
	// transitions and dependencies go through the barrier batcher, which BeginRendering() flushes
	// (or, for a safe draw continuing the same rendering instance, may sink past the draw).
	const bool* programs_safe = plan_statics && plan->plain_pixel == plain_pixel
	                                ? &plan->programs_barrier_safe
	                                : nullptr;
	const bool barrier_safe =
	    DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
	                      feedback_aspects, indirect != nullptr, plain_pixel,
	                      plan_verify ? nullptr : programs_safe);
	if (plan_verify && programs_safe != nullptr) {
		DrawPrep::CountBindingVerifyCheck();
		if (DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
		                      feedback_aspects, indirect != nullptr, plain_pixel,
		                      programs_safe) != barrier_safe) {
			DrawPrep::ReportBindingMismatch("barrier safety");
		}
	}
	const CommandBuffer::DrawScope draw_scope(buffer, barrier_safe);
	// Emission safe point (KYTY_CP_RECORDER): no native handle obtained during the preparation above
	// is alive; from here the draw's commands go to the recorder (render.h, CommandBuffer::Sink).
	const auto vk_buffer = buffer.EmissionSink();
	SetDrawDebugPhase(buffer, submit_id, draw, emit, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x300u);
	}
	// KYTY_DRAW_RUN_PUSH verify: a would-be continuation that would push only its per-draw
	// descriptors must push the previous draw's image and sampler descriptors on the normal path.
	const bool run_verify_push =
	    run_verify_images && m_run.verify_valid && DrawRunPartialPush(buffer, pipeline);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::CommitBindings");
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages, m_run_active);
	}
	if (run_verify_push) {
		DrawRun::GetTotals().partial_pushes.fetch_add(1, std::memory_order_relaxed);
		DrawRun::CountVerifyCheck();
		if (m_descriptor_images != m_run.pushed_images) {
			DrawRun::ReportMismatch("pushed image or sampler descriptors",
			                        m_descriptor_images.size());
		}
	}
	if (run_verify_images && Image::RecordedTransitions() != run_transitions) {
		DrawRun::CountVerifyCheck();
		// Detail: which recorded image's state the transition changed (0: another image's).
		DrawRun::ReportMismatch("image transition", DrawRunImagesChange(false, false, true));
	}
	CommitStats::Mark(CommitStats::Phase::CommitBindings);
	if (!mesh_active) CommitIndexBuffer(vk_buffer, index_binding);

	// KYTY_DRAW_RUN continuation: the dynamic state is a function of the same registers, targets and
	// vertex program as the previous draw's, which recorded it into this command buffer; the
	// shadow shows nothing replaced it since (another pipeline bind or command buffer).
	const bool run_dynamic = (m_run_active || m_run_verify) && DynamicStateShadowEnabled() &&
	                         m_dynamic_state.valid && m_dynamic_state.command == vk_buffer.Identity() &&
	                         m_dynamic_state.pipeline != nullptr &&
	                         buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) ==
	                             m_dynamic_state.pipeline &&
	                         (!m_context.GetGraphics().attachment_feedback_loop_enabled ||
	                          (m_dynamic_state.feedback_valid && !m_dynamic_state.feedback));
	if (m_run_active && !run_dynamic) {
		DrawRun::GetTotals().dynamic_emitted.fetch_add(1, std::memory_order_relaxed);
	}
	const auto run_dynamic_emitted = g_dynamic_state_emitted;
	if (!(m_run_active && run_dynamic)) {
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DynamicState");
		SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info,
		                         rendering, m_dynamic_state,
		                         plan_stages && plan->viewports.valid ? &plan->viewports : nullptr);
		if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Declared dynamic by every renderer pipeline; the shadow (valid for this command
			// buffer and bound pipeline, established just above) covers it too.
			auto& shadow = m_dynamic_state;
			if (DynamicStateShadowEnabled() && shadow.feedback_valid &&
			    shadow.feedback == feedback_aspects) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsAvoided);
			} else {
				vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
				shadow.feedback       = feedback_aspects;
				shadow.feedback_valid = true;
				Profiler::CountFrameEvent(Profiler::FrameEvent::DynamicStateCommandsEmitted);
				g_dynamic_state_emitted++;
			}
		}
	}
	if (m_run_verify && run_verify_images && run_dynamic &&
	    g_dynamic_state_emitted != run_dynamic_emitted) {
		DrawRun::CountVerifyCheck();
		DrawRun::ReportMismatch("dynamic state", g_dynamic_state_emitted - run_dynamic_emitted);
	}

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x400u);
	}
	if (indirect != nullptr) {
		// Pipeline barriers cannot be recorded inside dynamic rendering. Every buffer write is
		// recorded outside rendering or ends it (ShaderWriteBarrier), so the rendering instance
		// begun right after the previous barrier needs no new one while it remains active.
		const auto active = buffer.ActiveRenderingSerial();
		if (active == 0 || active != m_indirect_barrier_rendering) {
			{
				KYTY_GPU_OP_SITE("draw.indirect_args");
				m_context.GetCommandScheduler().EndRendering();
			}
			IndirectArgumentsBarrier(buffer, vk_buffer);
		}
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	if (m_depth_feedback.valid) {
		NoteDepthFeedback(buffer);
	}
	if (indirect != nullptr) {
		// A state change above only switched instances; no buffer write came in between.
		m_indirect_barrier_rendering = buffer.ActiveRenderingSerial();
	}
	buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	// The shadow now describes this command buffer with this pipeline bound. A pipeline without
	// color attachments declares no dynamic color-write enable, so binding it invalidated that
	// state for later pipelines.
	m_dynamic_state.command  = vk_buffer.Identity();
	m_dynamic_state.pipeline = pipeline.pipeline;
	if (state.color_count == 0) {
		m_dynamic_state.color_write_valid = false;
	}
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x500u);
	}
	// Mesh programs with split_groups read a seventh draw dword (MeshFirstGroupDword).
	const uint32_t mesh_draw_dwords =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwords(
	                      state.vertex_info[0].mesh.split_groups != 0)
	                : 0u;
	if (mesh_indirect) {
		// One indirect dispatch per conversion record, each reading its draw dwords from its own
		// parameter block (a record the draw does not need has no workgroups). Indirect draws are
		// never split: the block's dword 6 is 0.
		for (uint32_t record = 0; record < MeshIndirect::Records; record++) {
			const auto     address = mesh_slot.ParamsAddress(record);
			const uint32_t draw_data[] {static_cast<uint32_t>(address),
			                            static_cast<uint32_t>(address >> 32u), 0u,
			                            ShaderRecompiler::IR::PushData::MeshIndirectSentinel, 0u, 0u,
			                            0u};
			static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwords(true));
			vk_buffer.pushConstants(pipeline.pipeline_layout,
			    vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment, 0,
			    mesh_draw_dwords * sizeof(uint32_t), draw_data);
			vk_buffer.drawMeshTasksIndirectEXT(mesh_slot.buffer, mesh_slot.CommandOffset(record), 1,
			                                   MeshIndirect::CommandDwords * 4u);
		}
	} else if (mesh_active) {
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		const auto& mesh   = state.vertex_info[0].mesh;
		for (const auto& segment: mesh_segments) {
			const auto address = index_source.address +
			    static_cast<uint64_t>(segment.first) * index_source.guest_element_size;
			// A segment with more groups than X allows is dispatched in parts (split_groups only),
			// each pushing its first group. The shader's instance index is the pushed first instance
			// plus WorkgroupId.y, so a part with more instances than one dispatch carries is split
			// into instance ranges.
			const auto groups_per_dispatch = MeshGroupsPerDispatch(segment.groups, mesh, limits);
			for (uint32_t first_group = 0; first_group < segment.groups;
			     first_group += groups_per_dispatch) {
				const auto groups = std::min(groups_per_dispatch, segment.groups - first_group);
				const auto instances_per_dispatch = MeshInstancesPerDispatch(groups, limits);
				for (uint32_t base = 0; base < draw.instance_count; base += instances_per_dispatch) {
					const uint32_t draw_data[] {
					    segment.count,
					    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
					    emit.first_instance + base, index_source.guest_element_size,
					    static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u),
					    first_group};
					static_assert(std::size(draw_data) ==
					              ShaderRecompiler::IR::PushData::MeshDrawDwords(true));
					vk_buffer.pushConstants(pipeline.pipeline_layout,
					    vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eFragment,
					    0, mesh_draw_dwords * sizeof(uint32_t), draw_data);
					vk_buffer.drawMeshTasksEXT(groups,
					                           std::min(instances_per_dispatch, draw.instance_count - base),
					                           1);
				}
			}
		}
	} else if (indirect != nullptr) {
		EmitIndirectDraw(vk_buffer, *indirect, indirect_buffers);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	static const bool occlusion_draw_rows = [] {
		const char* value = std::getenv("KYTY_HANG_TRACE_OCCLUSION_DRAWS");
		return value != nullptr && value[0] == '1';
	}();
	const bool occlusion_debug_log = m_context.GetOcclusionCounter().DebugLogActive();
	if ((occlusion_draw_rows && HangTrace::Enabled() && m_context.GetOcclusionCounter().Active()) ||
	    occlusion_debug_log) {
		// Occlusion diagnostics: the state deciding whether this draw's samples pass.
		const auto&               regs = buffer.GetRegisters();
		const auto&               vp   = regs.GetScreenViewport().viewports[0];
		const auto&               mc   = regs.GetModeControl();
		const auto&               di   = state.depth_info;
		HangTrace::OcclusionEvent event;
		event.event  = "draw";
		event.value  = draw.index_count;
		event.scopes = draw.instance_count;
		event.colors = state.color_count;
		event.has_depth = static_cast<bool>(di.image_id);
		event.detail = fmt::format(
		    "prim={} indexed={} mesh={} indirect={} ps={} zt={} zw={} zop={} db={}[{:g},{:g}] "
		    "st={} cull={}{} face={} vp=[{:g},{:g}] zs={:g} zo={:g} vpxy={:g}x{:g}",
		    static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed() ? 1 : 0,
		    mesh_active ? 1 : 0, indirect != nullptr ? 1 : 0, state.ps_active ? 1 : 0,
		    di.depth_test_enable ? 1 : 0, di.depth_write_enable ? 1 : 0,
		    static_cast<uint32_t>(di.depth_compare_op), di.depth_bounds_test_enable ? 1 : 0,
		    di.depth_min_bounds, di.depth_max_bounds, di.stencil_test_enable ? 1 : 0,
		    mc.cull_front ? "F" : "", mc.cull_back ? "B" : "", mc.face ? 1 : 0, vp.zmin,
		    vp.zmax, vp.zscale, vp.zoffset, vp.xscale, vp.yscale);
		if (occlusion_debug_log) {
			const auto& clip = regs.GetClipControl();
			OcclusionCounter::DebugLog(
			    "d counted=%u n=%u inst=%u vs=0x%" PRIx64 " ps=0x%" PRIx64 " colors=%u zclip=%u/%u %s\n",
			    m_context.GetOcclusionCounter().Active() ? 1u : 0u, draw.index_count, draw.instance_count,
			    buffer.GetShaders().GetVs().es_regs.data_addr, buffer.GetShaders().GetPs().ps_regs.data_addr,
			    state.color_count, clip.min_z_clip_disable ? 1u : 0u, clip.max_z_clip_disable ? 1u : 0u,
			    event.detail.c_str());
		}
		if (occlusion_draw_rows && HangTrace::Enabled() && m_context.GetOcclusionCounter().Active()) {
			HangTrace::RecordOcclusion(event);
		}
	}

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	if (plan_statics && !plan_verify) {
		shader_write_stages = plan->shader_write_stages;
		DrawPrep::GetBindingTotals().statics_used.fetch_add(1, std::memory_order_relaxed);
	} else {
		for (const auto& stage: vertex_stages) {
			if (HasShaderBufferWrites(stage.stage)) {
				shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
			}
		}
		if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
			shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
		}
		if (plan_statics) {
			DrawPrep::CountBindingVerifyCheck();
			if (shader_write_stages != plan->shader_write_stages) {
				DrawPrep::ReportBindingMismatch("shader write stages");
			}
		}
	}
	if (shader_write_stages) {
		if (DrawWriteSinkEnabled()) {
			// Queued without ending the instance (render.h, KYTY_DRAW_WRITE_SINK): a later draw of
			// this instance may pass it, anything else records it first.
			ShaderWriteBarrier(buffer, shader_write_stages);
			// An indirect draw can no longer rely on "no buffer write since the argument barrier
			// while this instance is active": it must end the instance and record a new one.
			m_indirect_barrier_rendering = 0;
		} else {
			{
				KYTY_GPU_OP_SITE("draw.write_barrier");
				m_context.GetCommandScheduler().EndRendering();
			}
			ShaderWriteBarrier(buffer, shader_write_stages);
		}
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x700u);
	}
	// KYTY_DRAW_RUN: whether the next draw may continue this one (drawPrep/drawRun.h), and in verify
	// mode what this draw's normal path computed against what a continuation would have reused.
	if (DrawRun::Enabled()) {
		if (m_run_verify) {
			DrawRunVerify(state, rendering, feedback_aspects, stages);
		}
		const auto is_attachment = [&state](ImageId id) {
			for (uint32_t i = 0; i < state.color_count; i++) {
				if (state.color_info[i].image_id == id) {
					return true;
				}
			}
			return static_cast<bool>(state.depth_info.image_id) && state.depth_info.image_id == id;
		};
		bool eligible = m_in_engine_commit && m_prepared_validated && m_run_key != 0 &&
		                !mesh_active &&
		                (indirect == nullptr || DrawRun::IndirectRunsEnabled()) &&
		                vertex_stages.size() == 1 &&
		                !shader_write_stages && !feedback_aspects &&
		                buffer.ActiveRenderingSerial() != 0 &&
		                RenderStateFastEnabled(RenderStatePart::Reset) &&
		                !graphics_debug_dump_enabled() &&
		                !rendering.depth_stencil_attachment.depth_clear &&
		                !rendering.depth_stencil_attachment.stencil_clear;
		for (uint32_t i = 0; eligible && i < rendering.num_color_attachments; i++) {
			eligible = !rendering.color_attachments[i].is_clear;
		}
		for (const auto* stage: stages) {
			eligible = eligible && stage->gds.buffer == nullptr &&
			           DrawRunProgramEligible(*stage->runtime->program);
			for (const auto& binding: stage->images) {
				// Storage bindings are transitioned with write access (a barrier every draw).
				eligible = eligible && !is_attachment(binding.image_id) &&
				           binding.desc.type != TextureCache::BindingType::Storage;
			}
		}
		if (eligible) {
			DrawRun::GetTotals().eligible.fetch_add(1, std::memory_order_relaxed);
			DrawRunRecordDraw(buffer, state, m_run_slice_offset, rendering, stages,
			                  bounded ? &written : nullptr, pipeline);
		}
	}
	if (CommitStats::Enabled()) [[unlikely]] {
		CommitStats::Mark(CommitStats::Phase::Emit);
		CommitStats::NoteRecorded(MakeCommitShape(buffer, stages, pipeline, rendering,
		                                          m_dynamic_state, vertex_bindings, index_binding,
		                                          draw, emit, index_source, shader_write_stages));
		// The hashing above is the diagnostic's own cost: not counted in any phase.
		CommitStats::Skip();
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_GPU_OP_SITE("draw.index");
	KYTY_PROFILER_FUNCTION();
	Profiler::CountFrameWork(Profiler::FrameWork::DrawIndex);

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	// Self time here includes renderer-lock acquisition and setup outside the
	// separately timed preparation/execution phases.
	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	Common::LockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	RunDrawChecks(buffer, ucfg, m_binding_plan);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::Mark(CommitStats::Phase::Setup);
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source,
	    MeshRestartEnabled() && state.vertex_info[0].stage.program->stage == ShaderType::Mesh);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_GPU_OP_SITE("draw.auto");
	KYTY_PROFILER_FUNCTION();
	Profiler::CountFrameWork(Profiler::FrameWork::DrawAuto);

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	Common::LockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	RunDrawChecks(buffer, ucfg, m_binding_plan);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::Mark(CommitStats::Phase::Setup);
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

// Mirrors DrawIndex/DrawAuto with DrawOffsetSource::IndirectArgs, except that the counts, first
// index/vertex, vertex offset and first instance stay GPU data. Every check that needs a
// CPU-visible count returns false before a command is recorded.
// KYTY_CP_OP_STATS=1 (diagnostic): where a native indirect draw's time goes, one console line
// every 10 s ("IndirectDraws 10s: ...").
namespace IndirectStats {
enum class Part : uint8_t { Prepare, Execute, Count };
static bool Enabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_CP_OP_STATS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
static uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
struct Totals {
	std::array<uint64_t, 2> ns {};
	std::array<uint64_t, 2> count {};
	uint64_t                last_print = 0;
};
static Totals g_totals;
class Timer {
public:
	explicit Timer(Part part): m_part(part), m_start(Enabled() ? NowNs() : 0) {}
	void Stop() {
		if (m_start == 0) {
			return;
		}
		const auto now = NowNs();
		g_totals.ns[static_cast<size_t>(m_part)] += now - m_start;
		g_totals.count[static_cast<size_t>(m_part)]++;
		m_start = 0;
		if (g_totals.last_print == 0) {
			g_totals.last_print = now;
		} else if (now - g_totals.last_print >= 10'000'000'000ull) {
			std::printf("IndirectDraws 10s: prepare %" PRIu64 " in %.1f ms, execute %" PRIu64 " in %.1f ms\n",
			            g_totals.count[0], static_cast<double>(g_totals.ns[0]) / 1e6, g_totals.count[1],
			            static_cast<double>(g_totals.ns[1]) / 1e6);
			std::fflush(stdout);
			g_totals = {};
			g_totals.last_print = now;
		}
	}

private:
	Part     m_part;
	uint64_t m_start;
};
} // namespace IndirectStats

bool RenderExecutor::DrawIndirectNative(uint64_t submit_id, CommandBuffer& buffer,
                                        const DrawIndirectSource& source) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid() || source.args_addr == 0 || source.max_count == 0);
	const auto& graphics = m_context.GetGraphics();
	const auto& limits   = graphics.GetPhysicalDeviceProperties().limits;
	// Host command forms. firstInstance is GPU data, so its feature is always required.
	if (!graphics.draw_indirect_first_instance_enabled || (source.args_addr & 3u) != 0 ||
	    source.stride < source.RecordSize() || (source.stride & 3u) != 0 ||
	    source.max_count > limits.maxDrawIndirectCount ||
	    (source.max_count > 1 && !graphics.multi_draw_indirect_enabled) ||
	    (source.count_addr != 0 &&
	     (!graphics.draw_indirect_count_enabled || (source.count_addr & 3u) != 0)) ||
	    !m_context.IsMapped(source.args_addr, source.ArgsSize()) ||
	    (source.count_addr != 0 && !m_context.IsMapped(source.count_addr, sizeof(uint32_t)))) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackHost);
		return false;
	}

	DrawIndexBufferSource index_source {};
	if (source.indexed) {
		switch (static_cast<Prospero::IndexType>(source.index_type_and_size)) {
			case Prospero::IndexType::kIndex16:
				index_source.type               = vk::IndexType::eUint16;
				index_source.guest_element_size = 2;
				break;
			case Prospero::IndexType::kIndex32:
				index_source.type               = vk::IndexType::eUint32;
				index_source.guest_element_size = 4;
				break;
			case Prospero::IndexType::kIndex8:
				// The CPU path widens 8-bit indices to 16 bits on the host.
				if (!graphics.index_type_uint8_enabled) {
					Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackIndexBuffer);
					return false;
				}
				index_source.type               = vk::IndexType::eUint8;
				index_source.guest_element_size = 1;
				break;
			default: EXIT("unknown index_type_and_size: %u\n", source.index_type_and_size);
		}
		// The bound range is [INDEX_BASE, +INDEX_BUFFER_SIZE); firstIndex comes from each record.
		// The CPU path clamps the index count to INDEX_BUFFER_SIZE instead. Past that range the
		// host reads the neighbouring cached guest bytes, or robustBufferAccess values past the
		// host buffer (KYTY_INDIRECT_VALIDATE=1 reports such records). Without a size there is
		// no range to bind.
		if (source.index_base_addr == 0 || source.index_buffer_size == 0 ||
		    !m_context.IsMapped(source.index_base_addr, index_source.guest_element_size)) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackIndexBuffer);
			return false;
		}
		index_source.address = source.index_base_addr;
		index_source.size    = Libs::LibKernel::Memory::ClampRangeSize(
		    source.index_base_addr,
		    static_cast<uint64_t>(source.index_buffer_size) * index_source.guest_element_size);
	}

	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::DrainPendingOperations");
		m_context.GetCommandScheduler().PopReadyOperations();
	}
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	const auto debug_op =
	    source.indexed ? CommandBufferDebugOp::DrawIndex : CommandBufferDebugOp::DrawIndexAuto;
	// Counts are unknown on the CPU: 0xFFFFFFFF, with the argument address in place of the index
	// or first-instance detail.
	buffer.SetDebugInfo(static_cast<uint32_t>(debug_op), submit_id, UINT32_MAX, 0, 1, UINT32_MAX,
	                    source.args_addr);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	Common::LockGuard lock(m_context.GetMutex());
	// KYTY_DRAW_RUN: an engine commit with KYTY_DRAW_RUN_INDIRECT and a prepared slot, otherwise
	// other command-processor work for the run certificate.
	BeginDrawRun();
	if (DrawMayRunTargetOperation(buffer)) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackTargetOp);
		return false;
	}
	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndirectNative():Shader:\n");
		uc_print("GraphicsRenderDrawIndirectNative():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndirectNative():Parameters:\n"
		     "\t args_addr           = 0x%016" PRIx64 "\n"
		     "\t stride              = 0x%08" PRIx32 "\n"
		     "\t max_count           = 0x%08" PRIx32 "\n"
		     "\t count_addr          = 0x%016" PRIx64 "\n"
		     "\t indexed             = %u\n"
		     "\t index_base_addr     = 0x%016" PRIx64 "\n"
		     "\t index_buffer_size   = 0x%08" PRIx32 "\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n",
		     source.args_addr, source.stride, source.max_count, source.count_addr,
		     source.indexed ? 1u : 0u, source.index_base_addr, source.index_buffer_size,
		     source.index_type_and_size);
	}

	uc_check(ucfg);

	hw_check(buffer);

	// Legacy quads are split into per-quad host draws from the vertex count.
	if (ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy) {
		Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackQuadList);
		return false;
	}
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return true;
	}
	bool primitive_restart = false;
	if (source.indexed) {
		const auto restart =
		    ResolveNativePrimitiveRestart(buffer, index_source.guest_element_size);
		if (!restart) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackRestart);
			return false;
		}
		primitive_restart = *restart;
	}

	const DrawCallInfo draw {debug_op, UINT32_MAX, UINT32_MAX, 0};
	// The member state is reused (as in DrawIndex/DrawAuto) instead of value-initialising about
	// 37 KB and allocating fresh preparation vectors; the render mutex makes it exclusive.
	auto& state = *m_draw_state;
	IndirectStats::Timer prep_timer(IndirectStats::Part::Prepare);
	state.Reset();
	if (!PrepareDrawRenderState(buffer, draw, 0, state)) {
		ResetBindings();
		return true;
	}
	prep_timer.Stop();
	DrawEmitInfo emit {};
	emit.indirect = &source;
	// Mesh draws derive group counts, restart segments and push data from the counts: on the GPU
	// (KYTY_NATIVE_INDIRECT_MESH, meshIndirect.h), or on the CPU path.
	if (state.vertex_info[0].stage.program->stage == ShaderType::Mesh) {
		const auto decline = [&](std::optional<Profiler::FrameEvent> reason) {
			if (reason) {
				Profiler::CountFrameEvent(*reason);
			}
			Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectFallbackMesh);
			ResetBindings();
			return false;
		};
		if (MeshIndirect::GetMode() == MeshIndirect::Mode::Off) {
			return decline(std::nullopt);
		}
		const auto& program         = *state.vertex_info[0].stage.program;
		const auto& mesh            = state.vertex_info[0].mesh;
		const auto& limits          = graphics.mesh_shader_properties;
		auto&       inputs          = emit.mesh_inputs;
		inputs.indexed              = source.indexed;
		inputs.index_base           = index_source.address;
		inputs.index_size           = index_source.guest_element_size;
		inputs.index_buffer_size    = source.indexed ? source.index_buffer_size : 0u;
		inputs.primitive_size       = mesh.InputPrimitiveSize();
		inputs.primitive_step       = mesh.InputPrimitiveStep();
		inputs.primitives_per_group = mesh.primitives_per_group;
		inputs.max_groups_x         = limits.maxMeshWorkGroupCount[0];
		inputs.max_groups_y         = limits.maxMeshWorkGroupCount[1];
		inputs.max_groups_total     = limits.maxMeshWorkGroupTotalCount;
		if (MeshIndirect::AlwaysEmpty(inputs)) {
			// No record can reach a primitive (every record's count is clamped below one input
			// primitive): the CPU path would read the records and draw nothing, returning before
			// the render targets are acquired (ExecutePreparedDraw) or, for zero counts, before
			// the render state is resolved (DrawIndex). Target operations and restart values only
			// an index scan decides took the CPU path above.
			MeshIndirect::GetTotals().always_empty.fetch_add(1, std::memory_order_relaxed);
			Profiler::CountFrameEvent(Profiler::FrameEvent::MeshIndirectAlwaysEmpty);
			ResetBindings();
			return true;
		}
		if (!MeshIndirect::ConversionEnabled()) {
			return decline(std::nullopt);
		}
		// One record without a count: the CPU path draws a multi-draw record by record.
		if (source.max_count != 1 || source.count_addr != 0) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedMulti);
		}
		// Restart segments come from a CPU scan of the indices (KYTY_MESH_RESTART=1).
		if (primitive_restart && MeshRestartEnabled()) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedRestart);
		}
		// A clear-enable draw clears its depth/stencil attachment when its rendering instance
		// begins, which a draw with zero counts on the CPU path never does.
		if (state.depth_info.image_id &&
		    (state.depth_info.depth_clear_enable || state.depth_info.stencil_clear_enable)) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedClear);
		}
		// Programs generated with the parameter-block loads (the codegen option follows the
		// mode); the CPU path reports an unusable primitive group itself.
		if (!ShaderRecompiler::GetCodegenOptions().mesh_indirect_params ||
		    program.stage != ShaderType::Mesh || mesh.primitives_per_group == 0) {
			return decline(Profiler::FrameEvent::MeshIndirectDeclinedProgram);
		}
		emit.mesh_indirect = true;
	}
	if (!source.indexed && Prospero::IsRectList(ucfg.GetPrimType()) &&
	    state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, source.index_type_and_size,
	                     reinterpret_cast<const void*>(source.index_base_addr));
	IndirectStats::Timer exec_timer(IndirectStats::Part::Execute);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	exec_timer.Stop();
	ResetBindings();
	Profiler::CountFrameEvent(Profiler::FrameEvent::DrawIndirectNative);
	return true;
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
