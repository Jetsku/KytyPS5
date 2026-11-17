#include "graphics/host_gpu/renderer/renderDraw.h"
#include "common/staticOnce.h"

#include "graphics/host_gpu/renderer/drawPrep/ownedCounters.h"
#include "graphics/host_gpu/renderer/drawPrep/imgSide.h"
#include "graphics/guest_gpu/drawShortcutCounters.h"

#include "common/assert.h"
#include "common/liveSwitch.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/hangTrace.h"
#include "common/logging/log.h"
#include "common/onceValue.h"
#include "common/profiler.h"
#include "common/rendererBatch.h"
#include "common/skipInvalid.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/cpCommit.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawPrep/bindingPlan.h"
#include "graphics/host_gpu/renderer/drawPrep/commitStats.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/drawPrep/drawRun.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/dispatchGuard.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pixelRateVariant.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/commandStreamLog.h"
#include "graphics/host_gpu/renderer/encodeOffload.h"
#include "graphics/guest_gpu/command_processor/passCensus.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/watchRange.h"
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
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <set>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <string>
#include <string_view>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// KYTY_DEPTH_EQUAL_RELAX: see SetGraphicsDynamicParams.
static Live::Switch g_depth_equal_relax("KYTY_DEPTH_EQUAL_RELAX", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtol(value, nullptr, 10) : 0;
});
// KYTY_FRAME_TRACE (renderCompute.cpp).
bool FrameTraceTake();
// KYTY_NAN_SCAN (renderCompute.cpp).
uint32_t NanScanTake();
// KYTY_PASS_DUMP (renderCompute.cpp).
namespace PassDump {
bool Before(RenderContext& context);
void Flush(RenderContext& context);
bool Capturing();
void NoteOp(char kind, uint64_t hash);
void NoteOutput(uint64_t address, uint32_t width, uint32_t height, int aspect);
} // namespace PassDump

// KYTY_GPU_INSTANCE_COUNT=1 (default off, live; GT7 fork dcac2bb4f): a draw without its own
// instance count inherits the one a native indirect draw left in GPU memory. The command processor
// read it there (NumInstances), which drains the GPU; with the switch, when the newest native
// indirect draw decides the count alone, a vertex-pipeline draw becomes a one-record indirect draw
// whose instance count the GPU copies from the guest record. Mesh pipelines and quad lists still
// read it on the CPU.
static Live::Switch g_gpu_instance_count("KYTY_GPU_INSTANCE_COUNT", Live::ParseDefaultOff);
static std::atomic<uint64_t> g_gpu_instance_draws {0};

bool RenderExecutor::GpuInstanceCountEnabled() {
	return g_gpu_instance_count.On();
}

static bool MeshRestartEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_MESH_RESTART");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}));
	return enabled;
}

// KYTY_HTILE_FULL_RANGE=1 (adapted from Erlou93/KytyPS5-Yotei WORKAROUND_HTILE_FULL_RANGE, a logged
// approximation; PPSA26344 preset, default off): the host never writes HTILE contents (a consumed
// clear leaves the guest bytes as they were), but shaders read the per-tile min/max depth (Ghost of
// Yotei: to enqueue shadow-resolve tiles). Once per frame and HTILE address (and at each depth
// clear) the guest HTILE of a bound depth target is filled with "expanded, full depth range"
// (ZMask 0xF, MinZ 0, MaxZ 0x3FFF), so such tests are conservative instead of using stale tiles.
static bool HtileFullRangeEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_HTILE_FULL_RANGE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_HTILE_STENCIL_CLEAR (default on): an HTile clear also clears a stencil plane it tracks.
static bool HtileStencilClearEnabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_HTILE_STENCIL_CLEAR");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
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
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [] {
		const auto* value = std::getenv("KYTY_DYNAMIC_STATE_SHADOW");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}));
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
	constexpr uint32_t viewport_slots = std::extent_v<decltype(HW::ScreenViewport::viewports)>;
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
	// KYTY_VTE_CNTL: the scales and offsets PA_CL_VTE_CNTL disables are 1 and 0.
	const uint32_t control = ShaderRecompiler::GetCodegenOptions().vte_cntl
	                             ? ctx.GetScreenViewport().transform_control
	                             : ViewportTransform::DefaultControl;
	const float xscale  = ViewportTransform::Scale<0>(control, guest.xscale);
	const float yscale  = ViewportTransform::Scale<1>(control, guest.yscale);
	const float zscale  = ViewportTransform::Scale<2>(control, guest.zscale);
	const float xoffset = ViewportTransform::Offset<0>(control, guest.xoffset);
	const float yoffset = ViewportTransform::Offset<1>(control, guest.yoffset);
	const float zoffset = ViewportTransform::Offset<2>(control, guest.zoffset);
	if (ctx.GetClipControl().clip_disable) {
		viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
		viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
	} else {
		viewport.x      = xoffset - xscale;
		viewport.y      = yoffset - yscale;
		viewport.width  = xscale * 2.0f;
		viewport.height = yscale * 2.0f;
	}
	viewport.minDepth = zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : zscale);
	viewport.maxDepth = zscale + zoffset;
	return viewport;
}

bool DrawPrep::PlanDynamicViewports(const HW::Context& ctx, const vk::PhysicalDeviceLimits& limits,
                                    bool indexed_viewports, DynamicViewportPlan& plan) {
	constexpr uint32_t viewport_slots = std::extent_v<decltype(HW::ScreenViewport::viewports)>;
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
	constexpr uint32_t viewport_slots = std::extent_v<decltype(HW::ScreenViewport::viewports)>;
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
		OwnedCounters::Add(DrawPrep::GetBindingTotals().viewports_used, 1);
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
	// KYTY_DEPTH_EQUAL_RELAX (live diagnostic, default 0): an EQUAL depth test as LESS_OR_EQUAL (1)
	// or GREATER_OR_EQUAL (2), to test whether a pass's EQUAL test fails against a depth pre-pass.
	auto depth_compare_op = depth.depth_compare_op;
	if (depth_compare_op == vk::CompareOp::eEqual && g_depth_equal_relax.Get() != 0) {
		depth_compare_op = g_depth_equal_relax.Get() == 1 ? vk::CompareOp::eLessOrEqual
		                                                   : vk::CompareOp::eGreaterOrEqual;
	}
	if (recorder.Update(shadow.depth_compare_op, depth_compare_op)) {
		vk_buffer.setDepthCompareOp(depth_compare_op);
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
			OwnedCounters::Add(DrawPrep::GetBindingTotals().hw_checks_skipped, 1);
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
	       db.shader_dual_export_enable || db.shader_execute_on_noop || sh_regs.AlphaToMask();
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
	// KYTY_BK_ADOPT (drawPrep/bookkeep.h): while a kernel continuation runs, its stage runtimes point
	// at the committed slot's preparations instead of `stage_preps` (null otherwise).
	DrawPrep::PreparedDraw* adopted = nullptr;
	// The pixel preparation the draw reads (PlainPixelProgram): the adopted one or `stage_preps`.
	[[nodiscard]] const PipelineCache::StagePrep& PixelPrep() const {
		return adopted != nullptr ? adopted->pixel_prep : stage_preps.pixel;
	}
	// Points the stage runtimes back at `stage_preps` (after an adopted draw, or when it declines).
	void Unadopt() {
		if (adopted == nullptr) {
			return;
		}
		for (uint32_t i = 0; i < adopted->vertex_stages && i < vertex_info.size(); i++) {
			vertex_info[i].stage.resources = &stage_preps.vertex[i].resources;
		}
		if (adopted->pixel_active) {
			ps_input_info.stage.resources = &stage_preps.pixel.resources;
		}
		adopted = nullptr;
	}

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


struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

// The render area must lie inside every attachment. The guest extent comes from discovery, the
// attachment is a view of a cache image: should the image be smaller (an alias of a transient pool
// recreated at another size, seen in GT7), rendering the guest extent would write past it. With
// KYTY_SKIP_INVALID the area is bounded to the image, and logged.
static vk::Extent2D ClampToBacking(const Image& image, uint32_t base_level, vk::Extent2D extent,
                                   uint64_t address, const char* kind) {
	const vk::Extent2D backing {std::max(image.info.extent.width >> base_level, 1u),
	                            std::max(image.info.extent.height >> base_level, 1u)};
	if ((backing.width >= extent.width && backing.height >= extent.height) ||
	    !Common::SkipInvalidEnabled()) {
		return extent;
	}
	static std::atomic<uint32_t> reports {0};
	if (reports.fetch_add(1, std::memory_order_relaxed) < 32u) {
		LOGF("Render target: %s 0x%016" PRIx64 " is %ux%u but its image is %ux%u (level %u): "
		     "render area bounded to the image\n",
		     kind, address, extent.width, extent.height, backing.width, backing.height,
		     base_level);
	}
	return {std::min(extent.width, backing.width), std::min(extent.height, backing.height)};
}

RenderExecutor::DepthAccessDecision RenderExecutor::DecideDepthAccess(
    const CommandBuffer& buffer, const RenderDepthInfo& depth, const Image& image,
    std::span<PreparedBindings* const> stages) const {
	// AcquireRenderTargets' depth decisions, without effects (its profile counters are its own): what
	// the draw samples of the image (its own texture bindings' native views), the attachment layout
	// and access, and KYTY_DEPTH_FEEDBACK_KEEP. Reads: the depth info, the draw's bindings, the
	// image's views, tracked state, content serial and feedback proofs, the active rendering serial.
	DepthAccessDecision decision;
	const auto draw_writes = depth.AttachmentWriteAspects();
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			if (binding.image_id != depth.image_id ||
			    binding.desc.type != TextureCache::BindingType::Texture) continue;
			const auto native =
			    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
			if (native == image.views.end()) {
				decision.views_ok = false;
				return decision;
			}
			decision.sampled |= native->info.aspect;
			decision.feedback |= DepthFeedbackAspects(draw_writes, depth.desc.view_info, native->info);
		}
	}
	const auto sampled_aspects = decision.sampled;
	auto       layout          = depth_attachment_layout(depth);
	if (!sampled_aspects && DepthLayoutStableEnabled()) {
		// Nothing in this draw samples the image, so its attachment layout is not observable:
		// keep the current one while it allows the draw's writes (no transition, no new
		// rendering instance).
		const auto stable = depth_stable_attachment_layout(
		    image.backing.state.layout, image.backing.subresource_states.empty(),
		    ImageViewOps::DepthAspectMask(depth.desc.view_info.format), draw_writes);
		decision.stable_avoided = stable != layout && stable == image.backing.state.layout;
		layout                  = stable;
	}
	if (sampled_aspects & ~DepthReadableAspects(layout)) {
		layout = m_context.GetGraphics().attachment_feedback_loop_enabled
		             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
		             : vk::ImageLayout::eGeneral;
	}
	// The attachment store writes even when guest depth/stencil tests do not.
	const auto access = vk::AccessFlags2 {vk::AccessFlagBits2::eDepthStencilAttachmentRead |
	                                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite};
	const auto& view  = depth.desc.view_info;
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
	const auto& tracked        = image.backing.state;
	const bool  proof_instance = image.feedback_instance != 0 &&
	                            image.feedback_instance == buffer.ActiveRenderingSerial();
	const bool  proof_serial   = image.feedback_serial == image.ContentSerial();
	const bool  state_matches  = whole && image.backing.subresource_states.empty() &&
	                            tracked.layout == layout &&
	                            (tracked.access_mask == access || tracked.access_mask == sampled);
	decision.miss  = draw_writes      ? Profiler::FrameEvent::DepthFeedbackKeepMissWrite
	                 : !proof_instance ? Profiler::FrameEvent::DepthFeedbackKeepMissInstance
	                 : !proof_serial   ? Profiler::FrameEvent::DepthFeedbackKeepMissSerial
	                                   : Profiler::FrameEvent::DepthFeedbackKeepMissState;
	decision.keep  = DepthFeedbackKeepEnabled() && !draw_writes && proof_instance && proof_serial &&
	                state_matches;
	decision.adopt = decision.keep && DepthFeedbackAdoptsUnion(true, DepthFeedbackLazyEnabled(),
	                                                           static_cast<bool>(sampled_aspects),
	                                                           tracked.access_mask == access);
	decision.layout         = layout;
	decision.sampled_access = sampled;
	decision.access         = decision.adopt ? sampled : access;
	return decision;
}

vk::ImageMemoryBarrier2 RenderExecutor::FeedbackKeepOrdering(const Image& image, vk::ImageLayout layout,
                                                             vk::AccessFlags2 sampled) {
	// The ordering KYTY_DEPTH_FEEDBACK_KEEP leaves for the end of the instance.
	const auto&             tracked = image.backing.state;
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
	return ordering;
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages,
                                                 const vk::Rect2D* written) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects       = {};
	m_depth_feedback.valid = false;
	m_acq_has_depth        = false; // KYTY_IMG_ACQFB's seed
	m_acq_color_bound      = {};
	auto&       cache = m_context.GetTextureCache();
	if (ImgSide::StatsEnabled()) [[unlikely]] {
		// The same targets as the previous acquisition: image ids, slots, levels/layers, the depth
		// image and the command buffer.
		static uint64_t last_signature = 0;
		uint64_t        signature      = 0x9e3779b97f4a7c15ull ^ color_count;
		const auto      mix            = [&](uint64_t v) { signature = (signature ^ v) * 0x100000001b3ull; };
		for (uint32_t i = 0; i < color_count; i++) {
			mix(colors[i].image_id.index | (uint64_t {colors[i].image_id.generation} << 32u));
			mix(colors[i].target_slot | (uint64_t {colors[i].guest_mip_level} << 16u) |
			    (uint64_t {colors[i].guest_array_layer} << 32u));
		}
		mix(depth.image_id.index | (uint64_t {depth.image_id.generation} << 32u));
		mix(reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(buffer.Identity())));
		ImgSide::Count(ImgSide::C::Acquire);
		if (signature == last_signature) {
			ImgSide::Count(ImgSide::C::AcquireSame);
			ImgSide::Count(static_cast<ImgSide::C>(static_cast<uint8_t>(ImgSide::C::AcqWhyCaller) + m_acq_why));
			if (m_acq_why == 2) {
				ImgSide::Count(static_cast<ImgSide::C>(static_cast<uint8_t>(ImgSide::C::SeedWhyNotReached) +
				                                       m_seed_why_prev));
			}
		}
		last_signature = signature;
		m_acq_why      = 0;
	}
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	CommitStats::SubScope color_scope(CommitStats::Sub::AcquireColor);
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
		if (WatchRange::Enabled()) {
			WatchRange::Note(fmt::format("color target slot {} image {}:{} {} {}x{} written {} meta {} 0x{:x}+0x{:x}",
			                             target.target_slot, target.image_id.index, target.image_id.generation,
			                             vk::to_string(image.backing.format), image.backing.extent.width,
			                             image.backing.extent.height, written != nullptr ? 1 : 0,
			                             static_cast<uint32_t>(target.desc.info.metadata.kind),
			                             target.desc.info.metadata.range.address,
			                             target.desc.info.metadata.range.size)
			                     .c_str(),
			                 target.desc.info.data.address, target.desc.info.data.size);
		}
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		m_acq_color_bound[i] = image.binding.is_bound; // KYTY_IMG_ACQFB's seed
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		const auto extent       = ClampToBacking(image, view.base_level, target.Extent(),
		                                         target.desc.info.data.address, "color");
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	color_scope.Stop();
	CommitStats::SubScope depth_scope(CommitStats::Sub::AcquireDepth);
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
		if (HtileFullRangeEnabled() && metadata.kind == ImageMetadataKind::Htile &&
		    metadata.range.size >= 4u && (metadata.range.address & 3u) == 0) {
			static std::unordered_map<uint64_t, uint64_t> filled_frame;
			const auto frame = static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum());
			const auto [entry, inserted] = filled_frame.try_emplace(metadata.range.address, frame);
			if (inserted || entry->second != frame || depth.depth_clear_enable) {
				entry->second   = frame;
				const auto size = metadata.range.size & ~uint64_t {3};
				auto [htile, htile_offset] =
				    buffer.GetContext().GetBufferCache().ObtainBuffer(metadata.range.address, size, true);
				if (htile != nullptr) {
					htile->Fill(htile_offset, size, 0xFFFC000Fu);
				}
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1, std::memory_order_relaxed) < 4u) {
					LOGF("HTILE full range (KYTY_HTILE_FULL_RANGE, approximation): 0x%012" PRIx64
					     " size 0x%" PRIx64 "\n",
					     metadata.range.address, size);
				}
			}
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		if (metadata.kind == ImageMetadataKind::Htile && metadata.range.address == MetaTraceAddress() &&
		    MetaTraceLine()) {
			LOGF("MetaTrace: draw frame %d depth 0x%" PRIx64 " %ux%u samples %u layer %u meta_clear %d reg_clear %d "
			     "(%g) test %d write %d func %u stencil_compressed %d stencil_clear %d (%u)\n",
			     static_cast<int>(m_context.GetGpu().GetFrameNum()), depth.desc.info.data.address,
			     depth.desc.info.extent.width, depth.desc.info.extent.height,
			     static_cast<unsigned>(depth.desc.info.samples), depth.desc.view_info.base_layer, meta_clear ? 1 : 0,
			     depth.depth_clear_enable ? 1 : 0, static_cast<double>(depth.depth_clear_value),
			     depth.depth_test_enable ? 1 : 0, depth.depth_write_enable ? 1 : 0,
			     static_cast<unsigned>(depth.depth_compare_op), metadata.stencil_compressed ? 1 : 0,
			     depth.stencil_clear_enable ? 1 : 0, static_cast<unsigned>(depth.stencil_clear_value));
		}
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth (upstream 5bf37f52a).
		// KYTY_HTILE_STENCIL_CLEAR=0 restores the depth-only clear of a compressed stencil.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed && HtileStencilClearEnabled()))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
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
		// The draw's sampling of its depth attachment, its layout and the KYTY_DEPTH_FEEDBACK_KEEP decision
		// (DecideDepthAccess; KYTY_IMG_ACQFB re-evaluates the same function for a reuse).
		const auto decision = DecideDepthAccess(buffer, depth, image, stages);
		EXIT_IF(!decision.views_ok);
		m_acq_decision  = decision; // KYTY_IMG_ACQFB's seed
		m_acq_has_depth = true;
		const auto sampled_aspects = decision.sampled;
		feedback_aspects |= decision.feedback;
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
		const auto layout = decision.layout;
		if (decision.stable_avoided) {
			Profiler::CountFrameEvent(Profiler::FrameEvent::DepthLayoutTransitionsAvoided);
		}
		auto        access  = decision.access;
		const auto& view    = depth.desc.view_info;
		const auto& tracked = image.backing.state;
		if (DepthFeedbackKeepEnabled() && sampled_aspects && !decision.keep) {
			// Aggregate profile: why a sampled depth attachment keeps its per-draw toggles.
			Profiler::CountFrameEvent(decision.miss);
		}
		if (decision.keep && !decision.adopt) {
			// KYTY_DEPTH_FEEDBACK_LAZY: nothing samples the image yet; it keeps the attachment access.
			DrawRun::GetTotals().depth_promotions_deferred.fetch_add(1, std::memory_order_relaxed);
		} else if (decision.adopt) {
			// Without the keep: a barrier back to the attachment scope when the previous draw
			// sampled, and one to the sampled scope when this draw samples.
			const uint64_t avoided = (tracked.access_mask == decision.sampled_access ? 1u : 0u) +
			                         (sampled_aspects ? 1u : 0u);
			if (avoided != 0) {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DepthFeedbackBarriersAvoided,
				                          avoided);
			}
			buffer.NoteFeedbackKeep(FeedbackKeepOrdering(image, layout, decision.sampled_access));
			m_acq_keeps++; // KYTY_IMG_ACQFB counts
			image.AdoptState(tracked.pl_stage, decision.sampled_access, layout);
		}
		m_depth_feedback = {depth.image_id, static_cast<bool>(draw_writes), true};
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Identity(), true);
		const auto depth_extent =
		    ClampToBacking(image, view.base_level,
		                   {depth.desc.info.extent.width, depth.desc.info.extent.height},
		                   depth.desc.info.data.address, "depth");
		state.width               = std::min(state.width, depth_extent.width);
		state.height              = std::min(state.height, depth_extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	depth_scope.Stop();
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
		if (!render_target_bound(rt) || (slot != 0 && render_target_mask_slot(target_mask, slot) == 0)) {
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
		                      ((rt.attrib.num_samples == 0 && rt.attrib.num_fragments == 0) ||
		                       (cmask && !dcc && CmaskMsaaEnabled() &&
		                        rt.attrib.num_samples == rt.attrib.num_fragments));
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
	// KYTY_GPU_INSTANCE_COUNT: nonzero, a vertex-pipeline draw whose instance count the GPU
	// copies from this guest dword into a one-record indirect draw (draw.instance_count unused).
	uint64_t gpu_instance_address = 0;
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
	if (IsMetadataColorMode(mode) || mode == static_cast<uint8_t>(CbColorMode::Resolve) ||
	    hw.GetRenderControl().copy_depth_to_color) { // RenderExecutor::CopyDepthToColor
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
	// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS (BufferCache::PrepareReadRange), before either obtain.
	cache.PrepareReadRange(source.args_addr, source.ArgsSize());
	if (source.count_addr != 0) {
		cache.PrepareReadRange(source.count_addr, sizeof(uint32_t));
	}
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
// KYTY_INDIRECT_ARGS_COVERED (live, default off): an indirect draw in a rendering instance that began
// after other work (a submission, a target change, a barrier split) does not end it for its own
// argument barrier when no draw of that instance queued a shader-write barrier without ending it
// (KYTY_DRAW_WRITE_SINK). Every argument writer's own barrier already reaches indirect command reads
// and was recorded before the instance began (BeginRendering flushes the batch of a new instance):
// dispatches ShaderAccess (compute -> all commands, memory read), draws ShaderWrite (-> all graphics,
// indirect command read), buffer fills, copies and uploads (transfer -> all commands, memory read).
// Not with KYTY_PM4_BARRIERS=guest (dispatches and draws record no barriers there).
static Live::Switch g_indirect_args_covered("KYTY_INDIRECT_ARGS_COVERED", Live::ParseDefaultOff);

// KYTY_INDIRECT_ARGS_RANGES (render.h): live, default off.
static Live::Switch g_indirect_args_ranges("KYTY_INDIRECT_ARGS_RANGES", Live::ParseDefaultOff);

bool RenderExecutor::IndirectArgsBarrierNeeded(const CommandBuffer& buffer, const DrawIndirectSource* source,
                                               bool count, std::span<const Hazard::Range> host) const {
	const auto active = buffer.ActiveRenderingSerial();
	if (active == 0) {
		return true; // a new instance begins for this draw: its batch carries the request
	}
	if (active == m_indirect_barrier_rendering) {
		return false;
	}
	if (count) {
		GpuHazardCountIndirect(HazIndirect::Split);
	}
	if (g_indirect_args_ranges.On() && source != nullptr && BarrierBatchEnabled() && !GuestSyncBarriersEnabled()) {
		// Writers before this instance recorded their barriers before it began (a new instance flushes
		// the batch); within it only draws with sunk shader-write barriers wrote, at these ranges.
		if (active == m_haz_write_instance) {
			if (m_haz_write_imprecise) {
				if (count) GpuHazardCountIndirect(HazIndirect::Imprecise);
				return true;
			}
			Hazard::Range reads[2] {{0, source->args_addr, source->args_addr + source->ArgsSize(), false},
			                        {0, source->count_addr, source->count_addr + sizeof(uint32_t), false}};
			const uint32_t read_count = source->count_addr != 0 ? 2u : 1u;
			for (const auto& write: m_haz_instance_writes) {
				for (uint32_t i = 0; i < read_count; i++) {
					if (Hazard::Overlap(write, reads[i])) {
						if (count) GpuHazardCountIndirect(HazIndirect::Conflict);
						return true;
					}
				}
			}
		}
		// Verify: the same question on host buffer ranges (the instance's written bindings against the
		// argument/count buffers); a disagreement keeps the barrier.
		if (active == m_haz_write_instance) {
			for (const auto& write: m_haz_instance_host_writes) {
				for (const auto& read: host) {
					if (Hazard::Overlap(write, read)) {
						if (count) GpuHazardCountIndirect(HazIndirect::HostMismatch);
						return true;
					}
				}
			}
		}
		if (count) GpuHazardCountIndirect(HazIndirect::Skip);
		return false;
	}
	return !(g_indirect_args_covered.On() && active != m_indirect_write_rendering &&
	         BarrierBatchEnabled() && !GuestSyncBarriersEnabled());
}

void RenderExecutor::NoteInstanceWrites(uint64_t instance, std::span<PreparedBindings* const> stages) {
	if (instance != m_haz_write_instance) {
		m_haz_write_instance  = instance;
		m_haz_write_imprecise = false;
		m_haz_instance_writes.clear();
		m_haz_instance_host_writes.clear();
	}
	if (m_haz_write_imprecise) {
		return;
	}
	auto& textures = m_context.GetTextureCache();
	for (const auto* stage: stages) {
		const auto& program = *stage->runtime->program;
		const auto& info    = program.info;
		if (program.has_address_writes || info.uses_dma || info.bda_writes || stage->buffer_sources.size() != info.buffers.size() ||
		    stage->images.size() != info.images.size()) {
			m_haz_write_imprecise = true;
			return;
		}
		for (uint32_t i = 0; i < info.buffers.size(); i++) {
			if (!(info.buffers[i].written || info.buffers[i].atomic)) {
				continue;
			}
			if (info.buffers[i].indirect_root != ShaderRecompiler::IR::BufferResource::NoIndirectBuffer ||
			    !info.buffers[i].indirect_resources.empty()) {
				m_haz_write_imprecise = true;
				return;
			}
			const auto& source = stage->buffer_sources[i];
			if (source.address != 0 && source.size != 0) {
				m_haz_instance_writes.push_back({0, source.address, source.address + source.size, true});
			}
			if (i < stage->buffers.size() && stage->buffers[i].buffer) {
				const auto& host = stage->buffers[i];
				m_haz_instance_host_writes.push_back(
				    {std::bit_cast<uint64_t>(static_cast<VkBuffer>(host.buffer)), host.offset,
				     host.range == VK_WHOLE_SIZE ? UINT64_MAX : host.offset + host.range, true});
			}
		}
		for (uint32_t i = 0; i < info.images.size(); i++) {
			if (!(info.images[i].written || info.images[i].atomic || info.images[i].atomic64)) {
				continue;
			}
			auto* image = textures.TryGetImage(stage->images[i].image_id);
			if (image == nullptr || info.images[i].bindless || info.images[i].runtime ||
			    info.images[i].indirect_root != ShaderRecompiler::IR::ImageResource::NoIndirectImage) {
				m_haz_write_imprecise = true;
				return;
			}
			for (const auto& range: {image->info.data, image->info.stencil, image->info.metadata.range}) {
				if (range.size != 0) {
					m_haz_instance_writes.push_back({0, range.address, range.address + range.size, true});
				}
			}
		}
	}
	if (m_haz_instance_writes.size() > 4096) {
		m_haz_write_imprecise = true;
	}
}

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

// ------------------------------------------------------------------------------------------------
// KYTY_ENC_OFFLOAD (live; default 0; 1 on; verify): the emission of draws on one in-order emit
// thread (encodeOffload.h; the C1 cut of the whole-workload plan at the draw's emission point). A
// draw the value packets cover posts two parts: its bindings (vertex
// buffers, push constants, push descriptors, bindless set, index buffer) after the commit thread
// built its descriptor writes, and its draw (pipeline bind, clip planes, object id, draw command)
// after BeginRendering. Anything else the commit thread records, and any access to the shadow
// contents, settles first (the hooks), so the stream order is the inline one.
static int64_t ParseEncOffload(const char* value) {
	if (value == nullptr || value[0] == '\0') {
		return 0;
	}
	if (std::strcmp(value, "verify") == 0 || std::strcmp(value, "2") == 0) {
		return 2;
	}
	return std::strcmp(value, "1") == 0 ? 1 : 0;
}
static Live::Switch g_enc_offload("KYTY_ENC_OFFLOAD", ParseEncOffload);

struct RenderExecutor::OffloadJob {
	bool                    draw_part = false;
	bool                    verify    = false;
	DrawEncode::Shadow*     shadow    = nullptr;
	DrawEncode::Shadow      verify_shadow;
	DrawEncode::ShadowFlags flags;
	vk::PipelineLayout      layout = nullptr;
	// Bindings part.
	bool                          vertex = false;
	PreparedVertexBuffers         vb;
	vk::PipelineBindPoint         point = vk::PipelineBindPoint::eGraphics;
	DrawEncode::DescriptorScratch scratch;
	DrawEncode::BuiltDescriptors  built;
	bool                          push       = false;
	bool                          known_miss = false;
	bool                          bindless   = false;
	vk::DescriptorSet             bindless_set = nullptr;
	bool                          index        = false;
	PreparedIndexBuffer           ib;
	// Draw part.
	bool                     bind     = false;
	vk::Pipeline             pipeline = nullptr;
	uint32_t                 clip_bytes = 0;
	vk::ShaderStageFlags     clip_stages;
	std::array<uint32_t, 24> clip {};
	bool                     objprim = false;
	vk::ShaderStageFlags     objprim_stages;
	uint32_t                 objprim_offset = 0;
	uint32_t                 object_id      = 0;
	DrawEncode::DirectDraw   direct;

	template <typename Sink>
	void Run(Sink& sink) {
		auto& shadow_ref = *shadow;
		if (!draw_part) {
			if (vertex) {
				DrawEncode::EmitVertexBuffers(sink, vb.count, vb.buffers.data(), vb.offsets.data(), vb.sizes.data());
			}
			DrawEncode::EmitPushConstants(sink, shadow_ref, flags, layout, built);
			if (push && !scratch.writes.empty()) {
				const auto result =
				    DrawEncode::EmitPushDescriptors(sink, shadow_ref, flags, point, layout, scratch, known_miss);
				if (!verify) {
					DrawEncode::CountPushMiss(result, scratch.writes);
				}
			}
			if (bindless) {
				DrawEncode::EmitBindlessSet(sink, point, layout, bindless_set);
			}
			if (index) {
				DrawEncode::EmitIndexBuffer(sink, ib.buffer, ib.offset, ib.type);
			}
			return;
		}
		if (bind) {
			sink.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		}
		if (clip_bytes != 0) {
			sink.pushConstants(layout, clip_stages, ShaderRecompiler::IR::PushData::UserClipPlaneDword * sizeof(uint32_t),
			                   clip_bytes, clip.data());
		}
		if (objprim) {
			sink.pushConstants(layout, objprim_stages, objprim_offset, sizeof(object_id), &object_id);
			shadow_ref.InvalidatePushConstants();
		}
		DrawEncode::EmitDirectDraw(sink, direct);
	}
	static void RunOn(void* job, CommandStream::Encoder& out) { static_cast<OffloadJob*>(job)->Run(out); }
};

// Why a draw (with the switch on) was emitted inline.
enum class OffloadMiss : uint32_t {
	NoStream,      // the command buffer records natively (no recorder, or a direct window)
	Sites,         // capture-mode GpuOpProfiler sites on every packet
	BatchOff,      // barrier batcher off (a GDS dependency would record at once)
	SetPath,       // descriptor sets (allocation and reuse cache stay on the commit thread)
	PushCompare,   // a push the shadow must compare (its epoch would move on the encode thread)
	DrawRunVerify, // KYTY_DRAW_RUN=verify reads the built writes
	Mesh,
	DrawDiag,      // the draw part: labels, draw statistics, skip lists, GPU instances, indirect
	Count,
};
static constexpr std::array<const char*, static_cast<size_t>(OffloadMiss::Count)> g_offload_miss_names {
    "no_stream", "sites", "batch_off", "set_path", "push_compare", "drawrun_verify", "mesh", "draw_diag"};

struct RenderExecutor::OffloadState {
	EncodeOffload::Engine                                     engine;
	std::array<OffloadJob, EncodeOffload::Engine::MaxJobs>    jobs;
	uint64_t                                                  next   = 0;
	CommandStream::Encoder*                                   target = nullptr;
	const CommandBuffer*                                      buffer = nullptr;
	// Counters (commit thread; printed by OffloadReport).
	uint64_t candidates = 0;
	uint64_t kernel     = 0; // of them kernel-taken continuations
	uint64_t bindings   = 0;
	uint64_t draws      = 0;
	std::array<uint64_t, static_cast<size_t>(OffloadMiss::Count)> misses {};
	uint64_t settle_stream = 0;
	uint64_t settle_shadow = 0;
	uint64_t settle_full   = 0;
	uint64_t settle_switch = 0;
	// Verify.
	uint64_t verify_parts       = 0;
	uint64_t verify_stream_b    = 0;
	uint64_t verify_stream_d    = 0;
	uint64_t verify_shadow_b    = 0;
	uint64_t verify_shadow_d    = 0;
	uint64_t verify_uncompared  = 0;
	uint64_t verify_packets     = 0;
	uint64_t reported           = 0;
};

void RenderExecutor::OffloadSettleNow(OffloadState& state, uint64_t& reason) {
	reason++;
	state.target->SetOpenHook(nullptr, nullptr);
	state.buffer->SetSettleHook(nullptr, nullptr);
	state.engine.Settle(*state.target);
}

void RenderExecutor::OffloadSettle(uint32_t reason) {
	auto& state = *m_offload;
	if (state.engine.Pending() == 0) {
		return;
	}
	OffloadSettleNow(state, reason == 0 ? state.settle_full : state.settle_switch);
}

RenderExecutor::OffloadJob* RenderExecutor::OffloadAcquire(CommandBuffer& buffer) {
	if (m_offload == nullptr) {
		m_offload = new OffloadState();
	}
	auto&      state   = *m_offload;
	auto*      encoder = buffer.StreamEncoder();
	const bool verify  = g_enc_offload.Get() == 2;
	if (state.engine.Pending() != 0 && (verify || state.target != encoder || state.buffer != &buffer)) {
		// Verify jobs run on a copy of the shadow and are taken at once: nothing may be pending.
		// (Unreachable: a new command buffer begins with a packet, which settles.)
		OffloadSettle(1);
	}
	if (state.engine.Pending() >= EncodeOffload::Engine::MaxJobs) {
		OffloadSettle(0);
	}
	state.engine.Configure(encoder->GetOptions());
	auto& job     = state.jobs[state.next % EncodeOffload::Engine::MaxJobs];
	job.verify    = verify;
	job.flags     = CommandBuffer::EncodeFlags();
	job.shadow    = job.verify ? &job.verify_shadow : &buffer.EncodeShadowUnsettled();
	if (job.verify) {
		job.verify_shadow = buffer.EncodeShadowUnsettled();
	}
	return &job;
}

void RenderExecutor::OffloadPost(OffloadJob* job, CommandBuffer& buffer) {
	auto& state = *m_offload;
	EXIT_IF(!state.engine.Post(job, OffloadJob::RunOn));
	state.next++;
	if (job->verify) {
		return;
	}
	state.target = buffer.StreamEncoder();
	state.buffer = &buffer;
	state.target->SetOpenHook(
	    [](void* context) {
		    auto& s = *static_cast<OffloadState*>(context);
		    OffloadSettleNow(s, s.settle_stream);
	    },
	    &state);
	buffer.SetSettleHook(
	    [](void* context) {
		    auto& s = *static_cast<OffloadState*>(context);
		    OffloadSettleNow(s, s.settle_shadow);
	    },
	    &state);
}

RenderExecutor::OffloadJob* RenderExecutor::OffloadBindings(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
                                                            std::span<PreparedBindings* const> stages, bool keep_images,
                                                            bool mesh_active, bool run_verify_push,
                                                            const PreparedVertexBuffers& vertex,
                                                            const PreparedIndexBuffer&   index) {
	if (m_offload == nullptr) {
		m_offload = new OffloadState();
	}
	auto& state = *m_offload;
	if ((++state.candidates & 0x3ffffu) == 1) {
		OffloadReport();
	}
	state.kernel += m_kernel ? 1u : 0u;
	// The commit's cache and barrier work runs here (the caller then encodes inline without it).
	CommitBindingsSideEffects(buffer, pipeline, stages, keep_images);
	auto miss = OffloadMiss::Count;
	auto* encoder = buffer.StreamEncoder();
	if (encoder == nullptr) {
		miss = OffloadMiss::NoStream;
	} else if (encoder->GetOptions().all_sites) {
		miss = OffloadMiss::Sites;
	} else if (!BarrierBatchEnabled()) {
		miss = OffloadMiss::BatchOff;
	} else if (mesh_active) {
		miss = OffloadMiss::Mesh;
	} else if (!pipeline.uses_push_descriptors) {
		miss = OffloadMiss::SetPath;
	} else if (run_verify_push || DrawRun::GetMode() == DrawRun::Mode::Verify) {
		miss = OffloadMiss::DrawRunVerify;
	} else if (!DrawEncode::FreshSkipEnabled() ||
	           !std::ranges::any_of(stages, [](const auto* prepared) { return prepared->fresh_upload; })) {
		// Without a fresh upload the push is compared with the shadow (and may move its epoch).
		bool any_descriptors = false;
		for (const auto* prepared: stages) {
			any_descriptors |= !prepared->runtime->program->bindings.descriptors.empty();
		}
		if (any_descriptors) {
			miss = OffloadMiss::PushCompare;
		}
	}
	if (miss != OffloadMiss::Count) {
		state.misses[static_cast<size_t>(miss)]++;
		return nullptr;
	}
	auto* job      = OffloadAcquire(buffer);
	job->draw_part = false;
	job->layout    = pipeline.pipeline_layout;
	job->point     = vk::PipelineBindPoint::eGraphics;
	job->vertex    = true;
	job->vb        = vertex;
	job->index     = true;
	job->ib        = index;
	// KYTY_DRAW_RUN_PUSH: a continuation pushes only its per-draw descriptors.
	const bool partial = keep_images && DrawRunPartialPush(buffer, pipeline);
	if (partial && !job->verify) {
		OwnedCounters::Add(DrawRun::GetTotals().partial_pushes, 1);
	}
	DrawEncode::BuildDescriptors(stages, vk::PipelineBindPoint::eGraphics, partial, MakeDescriptorEnv(stages),
	                             job->scratch, job->built);
	if (!job->verify) {
		CommitStats::AddItems(CommitStats::Item::DescWrites, job->scratch.writes.size());
	}
	job->push       = true;
	job->known_miss = DrawEncode::FreshSkipEnabled() && job->built.fresh;
	EXIT_IF(!job->scratch.writes.empty() && (!job->known_miss || pipeline.descriptor_set_layout == nullptr));
	if (!job->scratch.writes.empty() && !job->verify && DrawEncode::OffsetAuditEnabled()) [[unlikely]] {
		NoteDescriptorOffsetAudit(pipeline, job->scratch.writes);
	}
	job->bindless     = pipeline.uses_bindless;
	job->bindless_set = pipeline.uses_bindless ? m_context.GetBindlessTable().Set() : vk::DescriptorSet {};
	OffloadPost(job, buffer);
	if (!job->verify) {
		state.bindings++;
	}
	return job;
}

// Verify: the oldest job's packets and shadow against the inline encode recorded since real_begin.
void RenderExecutor::OffloadVerifyPart(OffloadJob* job, CommandBuffer& buffer, const CommandStream::Ring* ring,
                                       uint64_t real_begin) {
	auto&                  state = *m_offload;
	CommandStream::CallLog job_log;
	state.engine.TakeOldest(job_log);
	state.verify_parts++;
	const auto* after = buffer.EncoderRing();
	bool        stream_bad = false;
	if (ring == nullptr || after != ring) {
		state.verify_uncompared++;
	} else {
		CommandStream::CallLog inline_log;
		CommandStream::LogPackets(*ring, real_begin, ring->WritePosition(), inline_log);
		state.verify_packets += inline_log.calls.size();
		stream_bad = inline_log.calls != job_log.calls;
	}
	const auto& real  = buffer.EncodeShadowUnsettled();
	const auto& mine  = job->verify_shadow;
	const auto& p     = real.push_constants;
	const auto& q     = mine.push_constants;
	bool        shadow_bad = p.valid != q.valid ||
	                  (p.valid && (p.layout != q.layout || p.stages != q.stages || p.size != q.size ||
	                               std::memcmp(p.dwords.data(), q.dwords.data(),
	                                           std::min<size_t>(p.size, sizeof(p.dwords))) != 0));
	if (!job->draw_part) {
		const auto& x = real.descriptor_states[0];
		const auto& y = mine.descriptor_states[0];
		shadow_bad |= x.layout != y.layout || x.bound_set != y.bound_set || x.buffers != y.buffers ||
		              x.images != y.images || x.writes.size() != y.writes.size() ||
		              real.descriptor_epochs != mine.descriptor_epochs;
	}
	if (stream_bad) {
		(job->draw_part ? state.verify_stream_d : state.verify_stream_b)++;
	}
	if (shadow_bad) {
		(job->draw_part ? state.verify_shadow_d : state.verify_shadow_b)++;
	}
	if ((stream_bad || shadow_bad) && state.reported < 16) {
		state.reported++;
		std::printf("EncOffload verify: part %s mismatch (stream %d shadow %d, job packets %zu)\n",
		            job->draw_part ? "draw" : "bindings", stream_bad ? 1 : 0, shadow_bad ? 1 : 0, job_log.calls.size());
		std::fflush(stdout);
	}
}

void RenderExecutor::OffloadReport() {
	auto& state = *m_offload;
	auto& es    = state.engine.GetStats();
	const auto jobs    = es.jobs.load(std::memory_order_relaxed);
	const auto settles = es.settles.load(std::memory_order_relaxed);
	const auto timed   = es.timed.load(std::memory_order_relaxed);
	const auto wtimed  = es.worker_timed.load(std::memory_order_relaxed);
	std::string misses;
	for (size_t i = 0; i < g_offload_miss_names.size(); i++) {
		misses += fmt::format(" {}={}", g_offload_miss_names[i], state.misses[i]);
	}
	std::printf("EncOffload (%s): draws seen %" PRIu64 " (kernel %" PRIu64 ") bindings %" PRIu64 " draws %" PRIu64 " jobs %" PRIu64
	            " settles %" PRIu64 " (stream %" PRIu64 " shadow %" PRIu64 " full %" PRIu64 " switch %" PRIu64
	            ") jobs/settle %.2f spliced %" PRIu64 " B (%.0f B/job) waits %" PRIu64
	            " sampled splice %.0f ns wait %.0f ns per settle, worker %.0f ns per job, parks %" PRIu64 ";"
	            " misses%s; verify parts %" PRIu64 " packets %" PRIu64 " uncompared %" PRIu64
	            " mismatches stream_b=%" PRIu64 " stream_d=%" PRIu64 " shadow_b=%" PRIu64 " shadow_d=%" PRIu64 "\n",
	            g_enc_offload.Get() == 2 ? "verify" : "on", state.candidates, state.kernel, state.bindings, state.draws, jobs, settles,
	            state.settle_stream, state.settle_shadow, state.settle_full, state.settle_switch,
	            settles != 0 ? static_cast<double>(es.spliced_jobs.load()) / static_cast<double>(settles) : 0.0,
	            es.spliced_bytes.load(),
	            es.spliced_jobs.load() != 0
	                ? static_cast<double>(es.spliced_bytes.load()) / static_cast<double>(es.spliced_jobs.load())
	                : 0.0,
	            es.waits.load(), timed != 0 ? static_cast<double>(es.splice_ns.load()) / static_cast<double>(timed) : 0.0,
	            timed != 0 ? static_cast<double>(es.wait_ns.load()) / static_cast<double>(timed) : 0.0,
	            wtimed != 0 ? static_cast<double>(es.worker_ns.load()) / static_cast<double>(wtimed) : 0.0,
	            es.parks.load(), misses.c_str(), state.verify_parts, state.verify_packets, state.verify_uncompared,
	            state.verify_stream_b, state.verify_stream_d, state.verify_shadow_b, state.verify_shadow_d);
	std::fflush(stdout);
}

RenderExecutor::~RenderExecutor() {
	delete m_offload;
}

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

// KYTY_KEEP_DRAW_BUFFERS (live, default 0; 1 on, 2 verify): a draw's vertex buffer range or index
// buffer whose guest range equals the one kept at its position (merged vertex range i, the index
// buffer) takes the kept cache buffer and offset while BufferCache::KeepValid holds - the binding
// memo would hit the same entry (the KYTY_KEEP_BINDINGS rule for these read-only obtains). Not with
// KYTY_RAW_READ_IMAGE_SYNC (its obtains synchronize images first). 2 = verify: the normal obtain
// runs and its result is compared ("KeepDrawBuffers verify"). GPU thread.
static Live::Switch g_keep_draw_buffers("KYTY_KEEP_DRAW_BUFFERS", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtoll(value, nullptr, 10) : 0;
});
namespace KeepDrawBuffers {
struct State {
	std::array<BufferCache::KeptRead, ShaderVertexInputInfo::RES_MAX> vertex {};
	BufferCache::KeptRead                                            index {};
	uint64_t obtains = 0, kept = 0, checks = 0, mismatches = 0;
};
State& Get() {
	static thread_local State state;
	return state;
}
int Mode() {
	const auto mode = static_cast<int>(g_keep_draw_buffers.Get());
	if (mode == 0 || BufferCache::RawReadImageSyncEnabled()) {
		return 0;
	}
	if (!RegionManager::AnySerialEnabled()) {
		RegionManager::EnableAnySerial();
	}
	return mode;
}
// ObtainBuffer(vaddr, size, false) through the kept result.
std::pair<Buffer*, uint64_t> Obtain(BufferCache& cache, int mode, BufferCache::KeptRead& kept, uint64_t vaddr,
                                    uint64_t size) {
	if (mode == 0) {
		return cache.ObtainBuffer(vaddr, size, false);
	}
	auto&      state = Get();
	const bool valid = cache.KeepValid(kept, vaddr, size, BufferId {});
	if ((++state.obtains & 0x3fffffu) == 0) {
		std::printf("KeepDrawBuffers: obtains %" PRIu64 " kept %" PRIu64 " verify checks %" PRIu64
		            " mismatches %" PRIu64 "\n",
		            state.obtains, state.kept, state.checks, state.mismatches);
		std::fflush(stdout);
	}
	if (valid && mode == 1) {
		state.kept++;
		return {kept.buffer, kept.offset};
	}
	const auto stamp  = cache.KeepBegin();
	const auto result = cache.ObtainBuffer(vaddr, size, false);
	if (valid) {
		state.checks++;
		if ((result.first != kept.buffer || result.second != kept.offset) && state.mismatches++ < 8) {
			std::printf("KeepDrawBuffers verify: mismatch guest=0x%016" PRIx64 " size=0x%" PRIx64 "\n", vaddr,
			            size);
		}
	}
	(void)cache.KeepCapture(kept, stamp, vaddr, size, BufferId {}, result);
	return result;
}
} // namespace KeepDrawBuffers

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
		// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS (BufferCache::PrepareReadRange), before every obtain.
		cache.PrepareReadRange(ranges->merged[i].base_address,
		                       ranges->merged[i].acquired_end - ranges->merged[i].base_address);
	}
	const int keep_mode = KeepDrawBuffers::Mode();
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range = ranges->merged[i];
		const auto  size  = range.acquired_end - range.base_address;
		acquired[i]       = KeepDrawBuffers::Obtain(cache, keep_mode, KeepDrawBuffers::Get().vertex[i],
		                                            range.base_address, size);
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
		// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS (BufferCache::PrepareReadRange), before every obtain.
		cache.PrepareReadRange(ranges->merged[i].base_address,
		                       ranges->merged[i].acquired_end - ranges->merged[i].base_address);
	}
	const int keep_mode = KeepDrawBuffers::Mode();
	for (uint32_t i = 0; i < ranges->merged_count; i++) {
		const auto& range           = ranges->merged[i];
		const auto  size            = range.acquired_end - range.base_address;
		const auto [handle, offset] = KeepDrawBuffers::Obtain(cache, keep_mode, KeepDrawBuffers::Get().vertex[i],
		                                                      range.base_address, size);
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

// Whether `count` indices of `element_size` bytes hold `value`: blocks of 256 without an early exit
// inside, which the compiler vectorizes.
static bool HoldsIndex(const uint8_t* bytes, uint64_t count, uint32_t element_size, uint32_t value) {
	constexpr uint64_t Block = 256;
	for (uint64_t at = 0; at < count; at += Block) {
		const uint64_t n   = std::min(Block, count - at);
		bool           hit = false;
		if (element_size == 2) {
			const auto wanted = static_cast<uint16_t>(value);
			for (uint64_t i = 0; i < n; i++) {
				uint16_t index = 0;
				std::memcpy(&index, bytes + (at + i) * 2u, 2);
				hit |= index == wanted;
			}
		} else {
			for (uint64_t i = 0; i < n; i++) {
				uint32_t index = 0;
				std::memcpy(&index, bytes + (at + i) * 4u, 4);
				hit |= index == value;
			}
		}
		if (hit) {
			return true;
		}
	}
	return false;
}

// KYTY_LIST_RESET=1 (default off, read once; adapted from usa-ratt/KytyPS5-GT7 589d2d821): an
// indexed point, line, triangle or patch list draw with the guest's primitive reset on, whose
// indices hold the reset index, is drawn as the console assembles it: the primitive the reset
// index falls in is dropped and the next starts after it (GE_MULTI_PRIM_IB_RESET_EN applies to
// lists as to strips). Vulkan restarts strips and fans only, so the index was drawn as a vertex
// past the mesh (a suspect for spikes on GT7's glass roofs). The kept indices are drawn from the
// stream buffer. Indirect draws, 8-bit indices, mesh-shader vertex stages (they read the guest's
// indices themselves), lists over 16 MB and indices the GPU wrote (or whose download is still to
// land) are left as they are.
//
// Returns the count of indices kept (written to `rewritten`, in the guest's element size), or
// UINT32_MAX when the draw is left as it is.
static uint32_t RewriteListReset(const CommandBuffer& buffer, BufferCache& cache,
                                 const DrawIndexBufferSource& source, uint32_t index_count,
                                 std::vector<uint8_t>& rewritten) {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_LIST_RESET");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	if (!enabled) {
		return UINT32_MAX;
	}
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	if ((control & 0x1u) == 0 || (control & ~0x3u) != 0) {
		return UINT32_MAX;
	}
	const auto prim_type     = buffer.GetUserConfig().GetPrimType();
	uint32_t   per_primitive = 0;
	switch (prim_type) {
		case Prospero::PrimitiveType::kPointList: per_primitive = 1; break;
		case Prospero::PrimitiveType::kLineList: per_primitive = 2; break;
		case Prospero::PrimitiveType::kTriList: per_primitive = 3; break;
		case Prospero::PrimitiveType::kPatch:
			// The patch's input control points (VGT_LS_HS_CONFIG.HS_NUM_INPUT_CP).
			per_primitive = (buffer.GetRegisters().GetShaderRegisters().m_vgtLsHsConfig >> 8u) & 0x3fu;
			if (per_primitive == 0 || per_primitive > 32) {
				return UINT32_MAX;
			}
			break;
		default: return UINT32_MAX;
	}
	const auto         element_size = source.guest_element_size;
	constexpr uint64_t MaxBytes     = 16ull << 20u;
	if ((element_size != 2 && element_size != 4) || source.address == 0 || source.size == 0 ||
	    source.size > MaxBytes || source.host_data != nullptr || index_count == 0) {
		return UINT32_MAX;
	}
	const uint32_t index_mask = element_size == 2 ? 0xffffu : 0xffffffffu;
	const auto     reset      = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset & ~index_mask) != 0) {
		return UINT32_MAX; // matches no index of this size
	}
	const uint32_t restart = reset & index_mask;

	// What each index buffer held when it was last looked at. One without the reset index is
	// looked at again once its pages are CPU-dirty, and a quarter of a second after it was last
	// looked at anyway. One with it keeps its bytes and the indices kept, used again while the
	// guest bytes are the same.
	struct Key {
		uint64_t address;
		uint64_t size;
		uint32_t element_size;
		uint32_t restart;
		uint32_t per_primitive;
		bool     operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const noexcept {
			return static_cast<size_t>((key.address * 0x9e3779b97f4a7c15ull) ^ (key.size << 7u) ^
			                           key.restart ^ (uint64_t {key.per_primitive} << 48u) ^
			                           (static_cast<uint64_t>(key.element_size) << 60u));
		}
	};
	struct Seen {
		bool                                  holds_reset = false;
		std::chrono::steady_clock::time_point scanned {};
		std::vector<uint8_t>                  original;  // holds_reset: the guest's bytes
		std::vector<uint8_t>                  rewritten; // holds_reset: the indices kept
	};
	constexpr auto     RescanAfter    = std::chrono::milliseconds(250);
	constexpr uint64_t MaxCachedBytes = 64ull << 20u;
	thread_local std::unordered_map<Key, Seen, KeyHash> seen;
	thread_local uint64_t                               cached_bytes = 0;
	thread_local std::vector<uint8_t>                   guest;
	const Key  key {source.address, source.size, element_size, restart, per_primitive};
	auto       found = seen.find(key);
	const auto now   = std::chrono::steady_clock::now();
	if (found != seen.end() && !found->second.holds_reset &&
	    now - found->second.scanned < RescanAfter &&
	    !cache.IsRegionCpuModified(source.address, source.size)) {
		return UINT32_MAX;
	}
	// The guest's bytes, as the console reads them: not when the GPU wrote them (they would need
	// a download first).
	if (cache.HasGpuDirtyBytes(source.address, source.size) ||
	    cache.HasPendingBackingPublication(source.address, source.size)) {
		return UINT32_MAX;
	}
	const uint64_t count      = std::min<uint64_t>(index_count, source.size / element_size);
	const uint64_t byte_count = count * element_size;
	guest.resize(byte_count);
	if (!LibKernel::Memory::TryReadBacking(source.address, guest.data(), byte_count)) {
		return UINT32_MAX;
	}
	const uint8_t* bytes = guest.data();
	if (found != seen.end() && found->second.holds_reset && found->second.original.size() == byte_count &&
	    std::memcmp(found->second.original.data(), bytes, byte_count) == 0) {
		rewritten = found->second.rewritten;
		return static_cast<uint32_t>(rewritten.size() / element_size);
	}
	if (seen.size() >= 65536 || cached_bytes > MaxCachedBytes) {
		seen.clear();
		cached_bytes = 0;
	}
	auto& entry = seen[key];
	cached_bytes -= std::min<uint64_t>(cached_bytes, entry.original.size() + entry.rewritten.size());
	std::vector<uint8_t>().swap(entry.original);
	std::vector<uint8_t>().swap(entry.rewritten);
	entry.scanned     = now;
	entry.holds_reset = HoldsIndex(bytes, count, element_size, restart);
	if (!entry.holds_reset) {
		return UINT32_MAX;
	}
	entry.original.assign(bytes, bytes + byte_count);

	// A reset index drops the primitive it falls in and the next starts after it; each complete
	// primitive is copied whole.
	const auto read = [&](uint64_t i) -> uint32_t {
		if (element_size == 2) {
			uint16_t value = 0;
			std::memcpy(&value, bytes + i * 2u, 2);
			return value;
		}
		uint32_t value = 0;
		std::memcpy(&value, bytes + i * 4u, 4);
		return value;
	};
	rewritten.resize(byte_count);
	uint64_t kept   = 0;
	uint64_t resets = 0;
	uint64_t filled = 0;
	for (uint64_t i = 0; i < count; i++) {
		if (read(i) == restart) {
			resets++;
			filled = 0;
			continue;
		}
		if (++filled < per_primitive) {
			continue;
		}
		std::memcpy(rewritten.data() + kept * element_size,
		            bytes + (i + 1 - per_primitive) * element_size, per_primitive * element_size);
		kept += per_primitive;
		filled = 0;
	}
	rewritten.resize(kept * element_size);
	entry.rewritten = rewritten;
	cached_bytes += entry.original.size() + entry.rewritten.size();

	static std::atomic<uint64_t> lists {0};
	if (const auto n = lists.fetch_add(1, std::memory_order_relaxed) + 1; n <= 16 || std::has_single_bit(n)) {
		LOGF("Draw: an indexed list (primitive type %u, %u indices a primitive) at 0x%016" PRIx64
		     " holds the primitive reset index 0x%x %" PRIu64 " times in %" PRIu64
		     " indices: drawn with the %" PRIu64 " indices of the primitives it keeps (%" PRIu64
		     " such lists; KYTY_LIST_RESET)\n",
		     static_cast<uint32_t>(prim_type), per_primitive, source.address, restart, resets, count,
		     kept, n);
	}
	return static_cast<uint32_t>(kept);
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

// Draws that waited at least 2 ms for their graphics pipeline on the command processor (a new
// pipeline built there, whichever path builds it: fast-first, libraries, the optimized build): one
// console line a minute while there are any, "Pipeline waits 60s: ...". KYTY_PIPELINE_WAIT_STATS=1
// (startup, default off): the timing and the line cost two clock reads per unplanned draw.
static bool PipelineWaitStats() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PIPELINE_WAIT_STATS");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

static void NotePipelineWait(std::chrono::steady_clock::duration elapsed) {
	const auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
	if (us < 2000) {
		return;
	}
	struct Totals {
		uint64_t count = 0;
		uint64_t us    = 0;
		uint64_t max   = 0;
		uint64_t over_50ms = 0;
		std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
		uint64_t all_count = 0;
		uint64_t all_us    = 0;
	};
	static Totals totals; // command processor thread only
	totals.count++;
	totals.us += static_cast<uint64_t>(us);
	totals.max = std::max<uint64_t>(totals.max, static_cast<uint64_t>(us));
	totals.over_50ms += us >= 50000 ? 1u : 0u;
	totals.all_count++;
	totals.all_us += static_cast<uint64_t>(us);
	const auto now = std::chrono::steady_clock::now();
	if (now - totals.last >= std::chrono::seconds(60)) {
		std::printf("Pipeline waits 60s: %" PRIu64 " draws waited >= 2 ms for their graphics pipeline, "
		            "%.1f ms in total (max %.1f ms, %" PRIu64 " >= 50 ms); since start %" PRIu64
		            " draws, %.1f ms\n",
		            totals.count, static_cast<double>(totals.us) / 1000.0,
		            static_cast<double>(totals.max) / 1000.0, totals.over_50ms, totals.all_count,
		            static_cast<double>(totals.all_us) / 1000.0);
		std::fflush(stdout);
		totals.count = totals.us = totals.max = totals.over_50ms = 0;
		totals.last  = now;
	}
}

// KYTY_RENDER_STATE_FAST swap: exchanges two stage preparations member by member. The bindings name
// every member in declaration order, so a new member stops this compiling until it is swapped.
static void SwapStagePrepMembers(PipelineCache::StagePrep& a, PipelineCache::StagePrep& b) noexcept {
	auto& [a_resources, a_specialization, a_permutation] = a;
	auto& [b_resources, b_specialization, b_permutation] = b;
	auto& [a_buffers, a_images, a_samplers, a_srt, a_user_data, a_fill, a_gpu_fills,
	       a_bindless_heaps, a_bindless_sampler_heaps] = a_resources;
	auto& [b_buffers, b_images, b_samplers, b_srt, b_user_data, b_fill, b_gpu_fills,
	       b_bindless_heaps, b_bindless_sampler_heaps] = b_resources;
	auto& [a_spec_buffers, a_spec_images] = a_specialization;
	auto& [b_spec_buffers, b_spec_images] = b_specialization;
	a_buffers.swap(b_buffers);
	a_images.swap(b_images);
	a_samplers.swap(b_samplers);
	a_srt.swap(b_srt);
	a_user_data.swap(b_user_data);
	std::swap(a_fill, b_fill);
	a_gpu_fills.swap(b_gpu_fills);
	a_bindless_heaps.swap(b_bindless_heaps);
	a_bindless_sampler_heaps.swap(b_bindless_sampler_heaps);
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
// KYTY_CONT_KERNEL_INPLACE (renderDraw.cpp, ContinuationKernel): 1 while a kernel continuation is
// applied - its programs and pixel input are functions of the run key (programs, registers), so the
// draw state already holds them from the previous draw of the run and they are not copied again;
// 2 verify: copied, and compared with what the state held. GPU thread only.
static int g_apply_keyed_skip = 0;
// KYTY_BK_ADOPT: true while ContinuationKernel applies a preparation it may adopt by pointer.
static bool g_apply_adopt = false;
static std::atomic<uint64_t> g_inplace_checks {0};
static std::atomic<uint64_t> g_inplace_mismatches {0};

static void ApplyPreparedDraw(DrawPrep::PreparedDraw& prepared, DrawRenderState& state) {
	if (g_apply_keyed_skip == 2) {
		auto pixel            = prepared.pixel_info;
		pixel.stage.resources = state.ps_input_info.stage.resources;
		const bool same =
		    std::memcmp(&state.programs, &prepared.programs, sizeof(state.programs)) == 0 &&
		    (!prepared.pixel_active || std::memcmp(&state.ps_input_info, &pixel, sizeof(pixel)) == 0);
		const auto n = g_inplace_checks.fetch_add(1, std::memory_order_relaxed) + 1;
		const auto m = same ? g_inplace_mismatches.load(std::memory_order_relaxed)
		                    : g_inplace_mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
		if ((!same && m <= 8) || (n & 0x7fffffu) == 0) {
			std::printf("ContKernel inplace verify: checks %" PRIu64 " mismatches %" PRIu64 "\n", n, m);
		}
	}
	if (g_apply_keyed_skip != 1) {
		state.programs      = prepared.programs;
		state.ps_input_info = prepared.pixel_info;
	}
	CopyPreparedVertexInfo(state.vertex_info[0], prepared.vertex_info);
	state.adopted = nullptr;
	if (g_apply_adopt) {
		// KYTY_BK_ADOPT: the runtimes read the slot's preparations, which stay this thread's until
		// the slot retires after the draw (ContinuationKernel points them back first).
		state.vertex_info[0].stage.resources = &prepared.vertex_prep.resources;
		for (uint32_t i = 1; i < prepared.vertex_stages; i++) {
			CopyPreparedVertexInfo(state.vertex_info[i], prepared.tess_info[i - 1]);
			state.vertex_info[i].stage.resources = &prepared.tess_prep[i - 1].resources;
		}
		if (prepared.pixel_active) {
			state.ps_input_info.stage.resources = &prepared.pixel_prep.resources;
		}
		state.adopted = &prepared;
		return;
	}
	SwapPreparedStage(state.stage_preps.vertex[0], prepared.vertex_prep);
	state.vertex_info[0].stage.resources = &state.stage_preps.vertex[0].resources;
	// KYTY_DRAW_PREP_TESS: a tessellated draw's HS and TES stages.
	for (uint32_t i = 1; i < prepared.vertex_stages; i++) {
		CopyPreparedVertexInfo(state.vertex_info[i], prepared.tess_info[i - 1]);
		SwapPreparedStage(state.stage_preps.vertex[i], prepared.tess_prep[i - 1]);
		state.vertex_info[i].stage.resources = &state.stage_preps.vertex[i].resources;
	}
	if (prepared.pixel_active) {
		SwapPreparedStage(state.stage_preps.pixel, prepared.pixel_prep);
		state.ps_input_info.stage.resources = &state.stage_preps.pixel.resources;
	}
}

static bool AsyncTranslationDrawAllowed(CommandBuffer& buffer);
static void NoteAsyncTranslationRefusals(const DrawRenderState& state);

// KYTY_SKIP_OCCLUSION_VISIBLE=1 (default off; adapted from usa-ratt/KytyPS5-GT7 round 141,
// 3adca0037): draws inside a counting occlusion query may be skipped while their pipelines build
// (KYTY_PIPELINE_SKIP) or their shaders translate (KYTY_ASYNC_TRANSLATION), and such a skipped draw
// counts as visible: it adds 1024 samples to what later dumps publish (OcclusionCounter::
// AddSkippedSamples). With 0 samples the game would take the object for occluded and might stop
// drawing it, so it stayed missing after its pipeline was ready. Without the switch such draws are
// not skipped (they wait for their pipeline). A fixed count, not the draw's coverage: enough for
// visibility tests; a flare sized by samples flickers until the pipeline is ready.
static bool SkipOcclusionVisibleEnabled() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_SKIP_OCCLUSION_VISIBLE");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

// KYTY_CP_PIPELINE_PREFETCH_SKIP=1 (default off): the command processor does not request the
// draw's pipeline early (PipelineCache::PrefetchGraphicsPipeline before the bindings) when the
// draw has no planned pipeline. The prefetch is only a compile hint: the lookup after the bindings
// (TryGetGraphicsPipeline) builds the same key, finds the pipeline or requests its compile. With a
// warm pipeline cache the early request only built and looked up the key a second time (GT7 title:
// ~0.34 us of a ~6.3 us commit). DrawPrep workers still prefetch while planning (PlanPipeline).
static bool CpPipelinePrefetchSkipped() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_CP_PIPELINE_PREFETCH_SKIP");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

static void NoteSkippedOcclusionDraw(RenderContext& context, CommandBuffer& buffer) {
	if (!SkipOcclusionVisibleEnabled()) return;
	auto& occlusion = context.GetOcclusionCounter();
	if (occlusion.WouldCount(buffer.GetRegisters().GetDepthCountControl())) {
		constexpr uint64_t SkippedSamples = 1024;
		occlusion.AddSkippedSamples(SkippedSamples);
		const auto skipped = occlusion.SkippedDraws();
		if (skipped == 1 || (skipped & (skipped - 1u)) == 0) {
			LOGF("Occlusion: %" PRIu64 " draws skipped while compiling inside counting queries, counted "
			     "visible (KYTY_SKIP_OCCLUSION_VISIBLE)\n",
			     skipped);
		}
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
			if (render_target_bound(rt) && render_target_mask_slot(ctx.GetRenderTargetMask(), slot) != 0 &&
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
	CommitStats::StepMark(CommitStats::Step::TakePrepared);
	// KYTY_DRAW_PREP_TESS: the preparation has the draw's vertex stages (both come from the draw's
	// register snapshot, so they always agree once Validate holds).
	const bool prepared_valid =
	    prepared != nullptr && DrawPrep::Validate(*prepared, state.ps_active, target_export_mapping) &&
	    prepared->vertex_stages == state.vertex_stages_written;
	CommitStats::StepMark(CommitStats::Step::Validate);
	if (prepared_valid) {
		ApplyPreparedDraw(*prepared, state);
		// The binding plan was computed from this preparation (KYTY_DRAW_PREP_BINDINGS).
		executor.ActivateBindingPlan();
		executor.NotePreparedValidated();
		CommitStats::StepMark(CommitStats::Step::Apply);
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
	// KYTY_ASYNC_TRANSLATION: new VS/PS of a draw that may be skipped translate on workers.
	const bool async_translation = PipelineCache::AsyncTranslationEnabled();
	if (async_translation) {
		PipelineCache::AllowDeferredTranslation(AsyncTranslationDrawAllowed(buffer));
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    state.stage_preps);
	if (async_translation && state.programs.translation_taken) {
		NoteAsyncTranslationRefusals(state);
	}
}

// KYTY_OCCLUSION_NO_OUTPUT_DRAWS=1 (live, default off; PPSA26344 preset with KYTY_GPU_OCCLUSION=1; ported from
// Erlou93/KytyPS5-Yotei v0.1.4 ClassifyNoOutputDraw): a draw that writes no colour target, has no depth/stencil
// target and no active pixel shader still runs while a guest occlusion query counts it (KYTY_GPU_OCCLUSION): with
// nothing to test, every covered sample passes and the open query counts it. Ghost of Yotei's bloom draws each
// sun/moon flare twice, depth-tested and then with the depth test off, and a CS divides the two query results into
// the flare's visibility; the second draw was dropped here, so the ratio was wrong (flares through walls). Without
// an open query such a draw is still dropped.
static Live::Switch g_occlusion_no_output_draws("KYTY_OCCLUSION_NO_OUTPUT_DRAWS", Live::ParseDefaultOff);

// KYTY_EMPTY_SCISSOR_SKIP (default on, upstream b727a1faa; "0" draws them as before): a draw whose
// SCREEN scissor is empty rasterizes nothing in any viewport, so it is skipped unless a vertex stage
// writes buffers (descriptor writes, address writes or BDA writes); another title ends mip
// reductions with a 0x0 pass past the last level. Decided in PrepareDrawRenderState once the
// programs are known, before target resolution and the draw-run candidate: every commit path
// (DrawIndex, DrawAuto, both MRT split passes, DrawIndirectNative, a draw-prep slot's commit, a
// draw-run continuation) passes there. The continuation kernel (KYTY_CONT_KERNEL) does not, and needs
// not: it continues only a committed draw with the same structure key, which hashes every context
// register, the screen scissor included, and a skipped draw never records a run.
static bool EmptyScissorSkipEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_EMPTY_SCISSOR_SKIP");
		return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

static bool DrawHasEmptyScreenScissor(const CommandBuffer& buffer, const DrawRenderState& state) {
	const auto& vp = buffer.GetRegisters().GetScreenViewport();
	if (vp.screen_scissor_right > vp.screen_scissor_left &&
	    vp.screen_scissor_bottom > vp.screen_scissor_top) {
		return false;
	}
	const auto vertex_stages = std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	// A stage without a runtime counts as writing (the draw is not skipped).
	return std::ranges::none_of(vertex_stages, [](const ShaderVertexInputInfo& info) {
		return !info.stage || HasShaderBufferWrites(info.stage) ||
		       info.stage.program->info.bda_writes;
	});
}

// The empty-screen-scissor skip's counter site (one call per skipped draw; a draw's MRT split second
// pass is never reached for one).
static void NoteEmptyScissorSkip() {
	DrawShortcutCounters::Add(DrawShortcutCounters::g_empty_scissor_skips, 1);
	PassCensus::Drop('d', 0, "skipped: empty screen scissor (KYTY_EMPTY_SCISSOR_SKIP)");
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	KYTY_PROFILER_DETAIL_FUNCTION();
	state.ps_active = DrawHasActivePixelShader(buffer);
	CommitStats::StepMark(CommitStats::Step::PsActive);
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::RefreshShaders");
		RefreshShaders(buffer, draw, state);
	}
	CommitStats::Mark(CommitStats::Phase::Programs);
	if (state.programs.translation_pending) {
		// KYTY_ASYNC_TRANSLATION: skipped while a stage translates, as a draw is skipped while its
		// pipeline builds (counted for KYTY_HOLD_SKIPPED_DRAWS); no draw-run record may be continued
		// or recalled across it.
		g_frame_pipeline_skips.fetch_add(1, std::memory_order_relaxed);
		PassCensus::Drop('d', 0, "skipped: stage translation pending (KYTY_ASYNC_TRANSLATION)");
		NoteSkippedOcclusionDraw(m_context, buffer);
		m_run.valid        = false;
		m_run_active       = false;
		m_run_targets_kept = false;
		for (auto& record: m_run_kept) {
			record.valid = false;
		}
		return false;
	}
	// A stage whose shader the recompiler skipped (an unresolvable runtime descriptor,
	// KYTY_SRT_VARIANT_READS) leaves its program empty: drop the draw. A tessellated draw needs
	// all three vertex stages (one without a runtime made PrepareBindings exit: Wolverine
	// HS 0x3ff475220c3b4e7b + TES 0xb39fc591165dbc5f).
	const bool tess_missing =
	    state.vertex_stages_written == 3u &&
	    (!state.programs.vertex[1] || !state.programs.vertex[2] || !state.vertex_info[0].stage ||
	     !state.vertex_info[1].stage || !state.vertex_info[2].stage);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel) || tess_missing) {
		// Counted and reported (the first few, then every 2048th) so a dropped draw is never silent.
		static std::atomic_uint64_t dropped {0};
		const auto                  count = dropped.fetch_add(1, std::memory_order_relaxed) + 1u;
		if (count <= 4u || (count & 2047u) == 0u) {
			std::printf("Warning: draw #%" PRIu64 " dropped, %s program missing (%s)\n", count,
			            !state.programs.vertex[0] ? "vertex"
			            : tess_missing           ? "tessellation"
			                                     : "pixel",
			            draw.Name());
		}
		PassCensus::Drop('d', 0,
		                 !state.programs.vertex[0] ? "dropped: vertex program missing"
		                 : tess_missing           ? "dropped: tessellation program missing"
		                                          : "dropped: pixel program missing");
		return false;
	}
	if (EmptyScissorSkipEnabled() && DrawHasEmptyScreenScissor(buffer, state)) {
		NoteEmptyScissorSkip();
		return false;
	}
	if (PassCensus::Enabled()) {
		PassCensus::Exec('d', state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
		                 state.ps_active && state.ps_input_info.stage.program ? state.ps_input_info.stage.program->shader_hash : 0);
	}
	// KYTY_DRAW_RUN (drawPrep/drawRun.h): a continuation keeps the previous draw's targets (the
	// entries are still in `state`: the fast reset leaves them, DrawRunCandidate requires it).
	m_run_slice_offset = render_target_slice_offset;
	// Bindless images (descriptors.cpp): their per-frame feedback copy and image transitions may end
	// the rendering instance while the bindings are prepared; such draws never continue a run.
	const auto bindless_stage = [](const ShaderStageRuntime& stage) {
		return stage.resources != nullptr && !stage.resources->bindless_heaps.empty();
	};
	const auto bindless = [&] {
		return (state.ps_active && bindless_stage(state.ps_input_info.stage)) ||
		       std::ranges::any_of(state.vertex_info,
		                           [&](const auto& info) { return bindless_stage(info.stage); });
	};
	const bool run_candidate =
	    DrawRun::Enabled() && !bindless() &&
	    (DrawRunCandidate(buffer, state, render_target_slice_offset) ||
	     (DrawRun::RecallEnabled() &&
	      DrawRunRecallCandidate(buffer, state, render_target_slice_offset)));
	if (m_run_recall) {
		m_texdelta_verify = false; // KYTY_DRAW_RUN_TEXDELTA=2: a recall is not compared
	}
	CommitStats::StepMark(CommitStats::Step::RunCandidate);
	if (run_candidate) {
		if (DrawRun::GetMode() == DrawRun::Mode::On) {
			state.color_count         = m_run.color_count;
			state.color_slots_written = m_run.color_slots;
			m_run_active              = true;
			DrawRunMarkBindings(m_run_texdelta);
			CommitStats::StepMark(CommitStats::Step::MarkBindings);
			CommitStats::Mark(CommitStats::Phase::Targets);
			return true;
		}
		m_run_verify = true;
	}
	KYTY_PROFILER_DETAIL_BLOCK("Draw::ResolveTargets");
	// KYTY_DRAW_RUN_TARGETS (drawRun.h): the record's targets, when their inputs are the record's.
	const auto targets_mode = DrawRun::TargetsKeepMode();
	const bool keep_targets = targets_mode != 0 && !m_run_verify && !m_texdelta_verify &&
	                          DrawRunTargetsCandidate(buffer, state, render_target_slice_offset);
	if (keep_targets) {
		OwnedCounters::Add(DrawRun::GetTotals().targets_kept, 1);
	}
	if (keep_targets && targets_mode == 1) {
		const CommitStats::SubScope color_scope(CommitStats::Sub::ColorTargets);
		state.color_count         = m_run.color_count;
		state.color_slots_written = m_run.color_slots;
		// As the resolution does: only a resolved target is marked (a draw without a depth target,
		// or a colour entry without an image, has an invalid id).
		for (uint32_t i = 0; i < m_run.color_count; i++) {
			state.color_info[i] = m_run.colors[i];
			if (state.color_info[i].image_id) {
				BindRenderTarget(state.color_info[i].image_id);
			}
		}
		state.depth_info = m_run.depth[0];
		if (state.depth_info.image_id) {
			BindRenderTarget(state.depth_info.image_id);
		}
		m_run_targets_kept = true;
	} else {
		DrawRunTargets(buffer, draw, render_target_slice_offset, state);
		if (keep_targets) {
			// Verify: the resolution must have given the record's targets.
			DrawRun::CountVerifyCheck();
			bool same = state.color_count == m_run.color_count &&
			            state.color_slots_written == m_run.color_slots &&
			            SameRenderDepthInfo(state.depth_info, m_run.depth[0]);
			for (uint32_t i = 0; same && i < state.color_count; i++) {
				same = SameRenderColorInfo(state.color_info[i], m_run.colors[i]);
			}
			if (!same) {
				DrawRun::ReportMismatch("kept targets");
				if (targets_mode == 3) {
					EXIT("DrawRunVerify: kept targets differ from the resolved ones\n");
				}
			}
		}
	}

	if (std::exchange(m_depth_unsupported, false)) {
		// A depth or stencil state the host cannot draw with (logged by ResolveRenderDepthTarget).
		return false;
	}
	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		auto& occlusion = m_context.GetOcclusionCounter();
		if (g_occlusion_no_output_draws.On() && OcclusionCounter::Enabled() &&
		    (occlusion.Active() || occlusion.WouldCount(buffer.GetRegisters().GetDepthCountControl()))) {
			static std::atomic<uint64_t> counted {0};
			const auto                   n = counted.fetch_add(1, std::memory_order_relaxed);
			if (n < 4 || (n & 4095u) == 0) {
				LOGF("Occlusion: no-output draw %s run for an open query (KYTY_OCCLUSION_NO_OUTPUT_DRAWS, %" PRIu64
				     " so far)\n",
				     draw.Name(), n + 1u);
			}
		} else {
			LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
			                   draw.index_count, 0);
			return false;
		}
	}

	CommitStats::StepMark(CommitStats::Step::MarkBindings);
	CommitStats::Mark(CommitStats::Phase::Targets);
	return true;
}

// KYTY_MRT_SPLIT (default on; YOTEI 44ea334d4): slots of mask that write the same surface as a
// lower slot of it, each with channels of its own. Astro Bot's foliage-interaction stamps write RG
// (blended) through MRT0 and BA through MRT1 of one 1024x1024 surface; one image bound as two
// Vulkan colour attachments is undefined, and NVIDIA lost the stamps (grass ignored the hover laser
// or bent ahead of Astro). Such a draw (DrawIndex, DrawAuto) runs twice: once without these slots,
// once with only them, each pass keeping its slots' masks and blends. KYTY_MRT_SPLIT=0 binds them
// together as before.
static Live::Switch g_mrt_split("KYTY_MRT_SPLIT", Live::ParseDefaultOn); // live
// KYTY_SPIRV_VARIANT_LIVE (PipelineCache::SpirvVariantLiveGroup): draws run that group's copies.
static std::atomic<uint64_t> g_spirv_variant_draws {0};
// KYTY_DRAW_SKIP_VS / KYTY_DRAW_SKIP_PS=<hex hash>[,<hex hash>...] (diagnostic, live, default off, up
// to 64 hashes each): draws with one of these vertex / pixel programs emit no draw command.
struct DrawSkipSet {
	static constexpr uint32_t             Max = 64;
	std::array<std::atomic<uint64_t>, Max> hashes {};
	std::atomic<uint32_t>                 count {0};

	// Stores the list, returns its size (the switch's value).
	int64_t Parse(const char* value) {
		uint32_t n = 0;
		while (value != nullptr && *value != '\0' && n < Max) {
			char*      end  = nullptr;
			const auto hash = std::strtoull(value, &end, 16);
			if (end == value) break;
			if (hash != 0) hashes[n++].store(hash, std::memory_order_relaxed);
			value = (*end == ',') ? end + 1 : end;
			if (*end != ',') break;
		}
		count.store(n, std::memory_order_release);
		return n;
	}
	[[nodiscard]] bool Contains(uint64_t hash) const {
		const auto n = count.load(std::memory_order_acquire);
		for (uint32_t i = 0; i < n; i++) {
			if (hashes[i].load(std::memory_order_relaxed) == hash) return true;
		}
		return false;
	}
};
static DrawSkipSet  g_draw_skip_vs_set;
static DrawSkipSet  g_draw_skip_ps_set;
static Live::Switch g_draw_skip_vs("KYTY_DRAW_SKIP_VS", [](const char* value) -> int64_t {
	return g_draw_skip_vs_set.Parse(value);
}); // live, diagnostic
static Live::Switch g_draw_skip_ps("KYTY_DRAW_SKIP_PS", [](const char* value) -> int64_t {
	return g_draw_skip_ps_set.Parse(value);
}); // live, diagnostic

static bool MrtSplitEnabled() {
	return g_mrt_split.Get() != 0;
}

static uint32_t AliasedColorSlots(const HW::Context& ctx, uint32_t mask) {
	uint32_t aliased = 0;
	for (uint32_t later = 1; later < RENDER_COLOR_ATTACHMENTS_MAX; later++) {
		if ((mask & (1u << later)) == 0) {
			continue;
		}
		const auto later_addr = ctx.GetRenderTarget(later).base.addr;
		const auto later_mask = render_target_mask_slot(ctx.GetRenderTargetMask(), later);
		if (later_addr == 0 || later_mask == 0) {
			continue;
		}
		for (uint32_t first = 0; first < later; first++) {
			if ((mask & (1u << first)) != 0 && (aliased & (1u << first)) == 0 &&
			    ctx.GetRenderTarget(first).base.addr == later_addr &&
			    (render_target_mask_slot(ctx.GetRenderTargetMask(), first) & later_mask) == 0) {
				aliased |= 1u << later;
				break;
			}
		}
	}
	return aliased;
}

// KYTY_MRT_SPLIT accounting: the first splits are logged, then a count every 4096.
static void NoteMrtSplit(uint32_t slots) {
	static std::atomic<uint64_t> count {0};
	const auto                   n = count.fetch_add(1, std::memory_order_relaxed) + 1u;
	if (n <= 4u || (n & 4095u) == 0u) {
		std::printf("MRT split: draw #%llu runs a second pass for colour slots 0x%x (one surface bound "
		            "to several slots)\n",
		            static_cast<unsigned long long>(n), slots);
	}
}
// KYTY_MRT_DIAG=<ps hash>[,<ps hash>...] (diagnostic, default off): for draws with those pixel
// shaders, one stdout line per distinct register state (at most 64): CB_SHADER_MASK,
// CB_TARGET_MASK, the export formats, the program's colour outputs and the bound targets.
static void NoteMrtDiag(const CommandBuffer& buffer, const DrawRenderState& state, uint32_t mrt_mask) {
	static constexpr auto hashes_init = [] {
		std::vector<uint64_t> list;
		const char*           value = std::getenv("KYTY_MRT_DIAG");
		for (const char* p = value; p != nullptr && *p != '\0';) {
			char* end = nullptr;
			list.push_back(std::strtoull(p, &end, 16));
			if (end == p) break;
			p = (*end == ',') ? end + 1 : end;
		}
		return list;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(hashes_init())>> hashes_once;
	const std::vector<uint64_t>& hashes = hashes_once.Get(hashes_init);
	if (hashes.empty() || !state.ps_active || state.ps_input_info.stage.program == nullptr) {
		return;
	}
	const auto hash = state.ps_input_info.stage.program->shader_hash;
	if (std::find(hashes.begin(), hashes.end(), hash) == hashes.end()) {
		return;
	}
	const auto& ctx     = buffer.GetRegisters();
	const auto& sh_regs = ctx.GetShaderRegisters();
	std::string modes;
	for (uint32_t i = 0; i < 8; i++) {
		modes += fmt::format("{}{}", i == 0 ? "" : ",", static_cast<uint32_t>(sh_regs.target_output_mode[i]));
	}
	std::string outputs;
	for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
		if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
			outputs += fmt::format(" mrt@{}", output.location);
		} else if (output.kind == ShaderRecompiler::IR::StageOutputKind::ObjPrimId) {
			outputs += fmt::format(" objprim@{}", output.location);
		}
	}
	std::string targets;
	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt = ctx.GetRenderTarget(i);
		if (rt.base.addr != 0) {
			targets += fmt::format(" rt{}=0x{:x}/fmt{}/mask{:x}", i, rt.base.addr, static_cast<uint32_t>(rt.info.format),
			                       render_target_mask_slot(ctx.GetRenderTargetMask(), i));
		}
	}
	const auto line = fmt::format("MrtDiag: ps {:016x} cb_shader_mask 0x{:08x} target_mask 0x{:08x} col_format {} "
	                              "key_mask 0x{:08x} outputs{} mrt_mask 0x{:x} targets{}\n",
	                              hash, sh_regs.m_cbShaderMask, ctx.GetRenderTargetMask(), modes,
	                              state.ps_input_info.target_shader_mask, outputs, mrt_mask, targets);
	static std::mutex            mutex;
	static std::set<std::string> seen;
	std::lock_guard              lock(mutex);
	if (seen.size() < 64 && seen.insert(line).second) {
		std::printf("%s", line.c_str());
		std::fflush(stdout);
	}
}

// The draw's colour and depth target resolution (PrepareDrawRenderState; a KYTY_DRAW_RUN
// continuation whose images changed resolves them late, from the same registers).
void RenderExecutor::DrawRunTargets(CommandBuffer& buffer, const DrawCallInfo& draw,
                                    uint32_t render_target_slice_offset, DrawRenderState& state) {
	ImgSide::Count(ImgSide::C::TargetRun);
	state.color_count         = 0;
	state.color_slots_written = 0;
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.location; // the colour target (KYTY_CB_SHADER_MASK_EXPORTS)
			}
		}
	}
	if (SkipInactivePixelShadersEnabled()) {
		mrt_mask &= DrawColorOutputFilter(buffer.GetRegisters());
	}
	// KYTY_OBJPRIM_ID: the slot of the hardware ObjPrimID write (not a pixel-shader export).
	if (state.ps_active && state.ps_input_info.HasObjprimOutput()) {
		mrt_mask |= 1u << state.ps_input_info.objprim_slot;
	}
	// KYTY_MRT_SPLIT: the first pass of a draw that binds one surface to several slots leaves the
	// later ones for a second pass (m_mrt_split_slots); the second pass binds only those.
	mrt_mask &= m_mrt_filter;
	if (m_mrt_split_allowed && m_mrt_filter == ~0u && (mrt_mask & (mrt_mask - 1u)) != 0 &&
	    MrtSplitEnabled()) {
		m_mrt_split_slots = AliasedColorSlots(buffer.GetRegisters(), mrt_mask);
		mrt_mask &= ~m_mrt_split_slots;
	}
	NoteMrtDiag(buffer, state, mrt_mask);
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	// KYTY_RENDER_STATE_VERIFY with the reset part: each entry resolved over the previous draw's is
	// compared with the same resolution into a value-initialised entry (run after it, so the
	// draw's own resolution is the one without verification), which then provides the result.
	const bool verify_entries = RenderStateVerifyMode() != 0 &&
	                            RenderStateFastEnabled(RenderStatePart::Reset) &&
	                            !graphics_debug_dump_enabled();
	CommitStats::SubScope color_scope(CommitStats::Sub::ColorTargets);
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
	color_scope.Stop();
	const CommitStats::SubScope depth_scope(CommitStats::Sub::DepthTarget);
	SetStencilTraceDraw(draw.index_count, buffer.GetShaders().GetPs().ps_regs.data_addr,
	                    state.color_count, mrt_mask);
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
	// KYTY_TRACE_DRS_EXTENT=<hex address> (diagnostic; =1: 0x5002870000, Ghost of Yotei's main depth target;
	// idea from Erlou93/KytyPS5-Yotei's DRS_EXTENT): one stdout line "DRS_EXTENT frame=<n> depth=<w>x<h>"
	// whenever the extent of the depth target at that address changes (the game's internal resolution).
	{
		static constexpr auto drs_address_init = [] {
			const char* value = std::getenv("KYTY_TRACE_DRS_EXTENT");
			if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) return uint64_t {0};
			if (std::strcmp(value, "1") == 0) return uint64_t {0x5002870000ull};
			return static_cast<uint64_t>(std::strtoull(value, nullptr, 16));
		};
		static constinit Common::StaticOnce<std::remove_cvref_t<decltype(drs_address_init())>> drs_address_once;
		const uint64_t& drs_address = drs_address_once.Get(drs_address_init);
		if (drs_address != 0 && state.depth_info.image_id && state.depth_info.desc.info.data.address == drs_address) {
			static uint32_t last = 0;
			const auto&     e    = state.depth_info.desc.info.extent;
			const uint32_t  key  = e.width * 65536u + e.height;
			if (key != last) {
				last = key;
				std::printf("DRS_EXTENT frame=%llu depth=%ux%u\n",
				            static_cast<unsigned long long>(buffer.GetContext().GetGpu().GetFrameNum()), e.width,
				            e.height);
				std::fflush(stdout);
			}
		}
	}
	// Upstream 7b7294aa0 (TODO PPSA09243): Vulkan passes take one sample count. Where PS5
	// attachments mix counts, keep those matching the raster count instead of resolving
	// multisampled shadows. Draws with one sample count (all Astro Bot draws seen) skip this.
	if (state.color_count != 0 || state.depth_info.image_id) {
		const auto colors       = std::span {state.color_info, state.color_count};
		const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
		                                                    : state.color_info[0].desc.info.samples;
		// KYTY_EQAA_MIXED (usa-ratt/KytyPS5-GT7 18979e65b): colours of one count over depth of
		// another, which the device's mixed attachment samples take (EQAA: GT7's countdown and
		// replay passes with 2 depth samples and 1 colour), keep every attachment.
		const auto first_color_samples =
		    state.color_count != 0 ? state.color_info[0].desc.info.samples : 0u;
		const bool mixed_ok =
		    state.depth_info.image_id && state.color_count != 0 &&
		    buffer.GetGraphics().mixed_attachment_samples !=
		        GraphicContext::MixedAttachmentSamples::None &&
		    std::ranges::all_of(colors,
		                        [&](const RenderColorInfo& color) {
			                        return color.desc.info.samples == first_color_samples;
		                        }) &&
		    buffer.GetGraphics().AcceptsMixedSamples(first_color_samples, pass_samples);
		if (!mixed_ok && std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
			    return color.desc.info.samples != pass_samples;
		    })) {
			const auto raster =
			    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
			const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
				return color.desc.info.samples != raster;
			});
			state.color_count -= static_cast<uint32_t>(dropped.size());
			if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
				state.depth_info = {};
			}
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping attachments whose sample count differs from the raster.\n");
			}
		}
	}
}

// ------------------------------------------------------------------------------------------------
// KYTY_DRAW_RUN (drawPrep/drawRun.h): run-level commit of structure-sharing draws.

// A program whose draws can form a run: it writes no guest memory and no image (buffer or image
// writes and atomics, address writes, GDS, the fault buffer), so a draw of it changes nothing a
// later draw's structure depends on and needs no barrier against the next one. Its mip-statistics
// counters (an emulator buffer, unordered atomics) are allowed.
// buffer_writes (KYTY_DRAW_RUN_WRITES): V# buffer writes and atomics are allowed as well.
// fault_buffer_ok (the pipeline-skip decision with KYTY_PIPELINE_SKIP_FAULT_BUFFER): the fault
// buffer binding is allowed too.
static bool DrawRunProgramEligible(const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                   bool buffer_writes = false, bool fault_buffer_ok = false) {
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	if (program.has_address_writes) {
		return false;
	}
	for (const auto& resource: program.info.buffers) {
		if (!buffer_writes && (resource.written || resource.atomic)) {
			return false;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			return false;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || (binding.kind == Kind::FaultBuffer && !fault_buffer_ok)) {
			return false;
		}
	}
	return true;
}

// KYTY_DRAW_RUN_FAULT_BUFFER (live, default 0; Ghost of Yotei preset candidate): programs whose
// only "write" is the fault buffer (every program reading memory through BDA has the binding) may
// form runs. The binding is the same buffer for every draw, its bits are OR-ed without order and
// read after the submission (RenderContext::RunGarbageCollector), and a continuation still prepares
// its buffers (PrepareGraphicsBindings: PrefaultDmaPointers, PrepareBda), so a continued draw reads
// and reports what the full path's draw would. At Ghost of Yotei's torii ~1,250 draws per frame
// left no run record for this reason, ~740 of them followed by a draw with the same key.
static Live::Switch g_draw_run_fault_buffer("KYTY_DRAW_RUN_FAULT_BUFFER", Live::ParseDefaultOff);

// KYTY_PIPELINE_SKIP_FAULT_BUFFER=1 (default off; on in the GT7 and Ghost of Yotei presets): a
// program whose only "write" is the fault buffer may have its draws skipped while their pipeline
// builds. Every program that reads memory through BDA gets that binding (BindingLayout,
// info.uses_dma): it reports pages that must be faulted in, and a skipped draw loses only that
// report (the next draw that runs reports them again). Most of GT7's draws have it, so they all
// waited for their pipelines; in Ghost of Yotei 881 of 883 refusing programs refused only for it,
// and on a cold cache the command processor built their pipelines unoptimized for 0.1-0.7 s each
// while the game stood still.
static bool SkipFaultBufferEnabled() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_SKIP_FAULT_BUFFER");
		return value != nullptr && value[0] == '1';
	}();
	return on;
}

// KYTY_PIPELINE_SKIP: whether a draw may be skipped while the driver builds its new pipeline
// because the picture is all it changes: no stage writes guest memory or images (DrawRunProgramEligible),
// no guest occlusion query counts its samples, it clears no depth or stencil, it is no mesh draw,
// and it renders to tiled images only (the CPU reads linear ones back).
// KYTY_PIPELINE_SKIP_RECURRENT (default on): a skipped draw is lost for good when its target is
// not drawn again, e.g. a sky or probe texture rendered once at level start (Sky Garden's sky
// stayed black after a shader-cache invalidation). Only draws whose every target was already a
// render target in at least two earlier presented frames may be skipped; the others wait.
static bool SkipRecurrentTargetsOnly() {
	static constexpr auto on_init = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_SKIP_RECURRENT");
		return value == nullptr || std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(on_init())>> on_once;
	const bool& on = on_once.Get(on_init);
	return on;
}

// Notes the draw's render targets for the rule above (once per image and frame).
// KYTY_SKIP_TARGETS_MEMO (live, default 0): a draw whose targets are the ones the previous noted draw
// had, in the same frame, notes nothing (each image is noted once per frame anyway); ~10,000 draws a
// flip at the Astro's Playroom GPU Jungle mostly repeat their targets.
static Live::Switch g_skip_targets_memo("KYTY_SKIP_TARGETS_MEMO", Live::ParseDefaultOff);

static void NoteSkipTargets(RenderContext& context, const DrawRenderState& state) {
	auto&      cache = context.GetTextureCache();
	const auto frame = cache.Frame() + 1u; // 0 = never
	if (g_skip_targets_memo.On()) {
		// Commit thread only (ExecutePreparedDraw).
		static struct {
			uint64_t frame = 0;
			uint32_t count = UINT32_MAX;
			ImageId  depth {};
			std::array<ImageId, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
		} last;
		bool same = last.frame == frame && last.count == state.color_count && last.depth == state.depth_info.image_id;
		for (uint32_t i = 0; same && i < state.color_count; i++) {
			same = last.colors[i] == state.color_info[i].image_id;
		}
		if (same) {
			return;
		}
		last.frame = frame;
		last.count = state.color_count;
		last.depth = state.depth_info.image_id;
		for (uint32_t i = 0; i < state.color_count && i < last.colors.size(); i++) {
			last.colors[i] = state.color_info[i].image_id;
		}
	}
	const auto note  = [&](ImageId id) {
		if (!id) {
			return;
		}
		auto* image = cache.TryGetImage(id);
		if (image != nullptr && image->skip_target_frame != frame) {
			image->skip_target_frame  = frame;
			image->skip_target_frames = std::min<uint32_t>(image->skip_target_frames + 1u, 1000u);
		}
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		note(state.color_info[i].image_id);
	}
	note(state.depth_info.image_id);
}

static bool TargetsRecurrent(RenderContext& context, const DrawRenderState& state) {
	auto&      cache     = context.GetTextureCache();
	const auto frame     = cache.Frame() + 1u;
	const auto recurrent = [&](ImageId id) {
		if (!id) {
			return true;
		}
		const auto* image = cache.TryGetImage(id);
		if (image == nullptr) {
			return false;
		}
		// Earlier frames only: the current frame was noted before this check.
		const auto earlier = image->skip_target_frames - (image->skip_target_frame == frame ? 1u : 0u);
		return earlier >= 2u;
	};
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!recurrent(state.color_info[i].image_id)) {
			return false;
		}
	}
	return recurrent(state.depth_info.image_id);
}

// KYTY_SKIP_EXEMPT=1 (default off; on in the GT7 preset): draws into a reflection cube (a color
// target of 6n square layers) and G-buffer passes (two color targets or more, each at least
// 256x256, with a depth test that can fail and a depth write) may be skipped while their pipelines
// build from their first frame, not only once the targets recur (adapted from usa-ratt/KytyPS5-GT7
// rounds 91 and 146, a5a933c6d / 013717a4e: GT7's race start drew its cube and the car view's
// G-buffer for the first time and compiled 33 and 64 pipelines on the GPU thread).
static bool SkipExemptEnabled() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_SKIP_EXEMPT");
		return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
	}();
	return on;
}

static bool SkipExemptDraw(const DrawRenderState& state) {
	if (state.color_count == 0) return false;
	const auto& first = state.color_info[0].desc.info;
	if (first.resources.layers >= 6u && first.resources.layers % 6u == 0u &&
	    first.extent.width == first.extent.height) {
		return true; // a reflection cube
	}
	const auto& depth = state.depth_info;
	if (state.color_count < 2u || !static_cast<bool>(depth.image_id) || !depth.depth_test_enable ||
	    !depth.depth_write_enable || depth.depth_compare_op == vk::CompareOp::eAlways) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto extent = state.color_info[i].Extent();
		if (extent.width < 256u || extent.height < 256u) return false;
	}
	return true; // a G-buffer pass
}

static bool SkipStatsEnabled();
// KYTY_PIPELINE_SKIP_STATS: why a program keeps its draws from being skipped, once per shader.
static void NoteProgramRefusal(const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	if (!SkipStatsEnabled()) return;
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> noted;
	{
		std::scoped_lock lock(mutex);
		if (noted.size() > 512 || !noted.insert(program.shader_hash).second) return;
	}
	using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
	std::string why;
	if (program.has_address_writes) why += " address-writes";
	for (const auto& resource: program.info.buffers) {
		if (resource.written || resource.atomic) {
			why += resource.atomic ? " buffer-atomic" : " buffer-write";
			break;
		}
	}
	for (const auto& resource: program.info.images) {
		if (resource.written || resource.atomic) {
			why += resource.atomic ? " image-atomic" : " image-write";
			break;
		}
	}
	for (const auto& binding: program.bindings.descriptors) {
		if (binding.kind == Kind::Gds || binding.kind == Kind::FaultBuffer) {
			why += binding.kind == Kind::Gds ? " gds" : " fault-buffer";
			break;
		}
	}
	Log::WriteToConsoleAndLog(fmt::format("Pipeline skip: shader 0x{:016x} refuses skipping:{}\n",
	                                      program.shader_hash, why));
}

// Why a draw may not be skipped while its pipeline builds (0: it may), for KYTY_PIPELINE_SKIP_STATS.
enum class SkipRefusal : uint8_t { None, Stages, Occlusion, NotRecurrent, Program, LinearTarget, DepthClear, Count };
constexpr const char* SkipRefusalNames[] = {"skippable", "stages", "occlusion", "not-recurrent",
                                            "program", "linear-target", "depth-clear"};

static SkipRefusal DrawSkipRefusal(RenderContext& context, const DrawRenderState& state,
                                   std::span<const ShaderVertexInputInfo> vertex_stages, bool mesh_active) {
	// KYTY_PIPELINE_SKIP_TESS: tessellated draws (LS/HS/TES) may be skipped too. GT7's car bodies
	// and scenery are tessellated; their cold pipelines (200k-word stages) were built on the GPU
	// thread while the game stood still (usa-ratt/KytyPS5-GT7 skips every draw that writes no memory).
	const bool tess_ok = vertex_stages.size() == 3 && PipelineCache::SkipTessellatedEnabled();
	if (mesh_active || (vertex_stages.size() != 1 && !tess_ok)) {
		return SkipRefusal::Stages;
	}
	if (context.GetOcclusionCounter().Active() && !SkipOcclusionVisibleEnabled()) {
		return SkipRefusal::Occlusion;
	}
	if (SkipRecurrentTargetsOnly() && !TargetsRecurrent(context, state) &&
	    !(SkipExemptEnabled() && SkipExemptDraw(state))) {
		return SkipRefusal::NotRecurrent;
	}
	for (const auto& stage: vertex_stages) {
		if (stage.stage.program == nullptr ||
		    !DrawRunProgramEligible(*stage.stage.program, false, SkipFaultBufferEnabled())) {
			if (stage.stage.program != nullptr) NoteProgramRefusal(*stage.stage.program);
			return SkipRefusal::Program;
		}
	}
	if (state.ps_active && (state.ps_input_info.stage.program == nullptr ||
	                        !DrawRunProgramEligible(*state.ps_input_info.stage.program, false,
	                                                SkipFaultBufferEnabled()))) {
		if (state.ps_input_info.stage.program != nullptr) NoteProgramRefusal(*state.ps_input_info.stage.program);
		return SkipRefusal::Program;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!state.color_info[i].desc.info.IsTiled()) {
			return SkipRefusal::LinearTarget;
		}
	}
	const auto& depth = state.depth_info;
	if (static_cast<bool>(depth.image_id) &&
	    (!depth.desc.info.IsTiled() || depth.depth_clear_enable || depth.load_clear_aspects ||
	     depth.stencil_clear_enable)) {
		return SkipRefusal::DepthClear;
	}
	return SkipRefusal::None;
}

// KYTY_ASYNC_TRANSLATION (pipelineCache.cpp; adapted from Senaxx's GT7 fork a3d1eba5c): whether a
// draw may be skipped while a new VS/PS of it translates on a worker. The programs, and so the
// resolved targets, do not exist yet, so this is DrawSkipRefusal from the draw's registers (the
// stage rule is GetGraphicsPrograms'), plus the fork's rule: the depth test is on (UI and layer
// draws, which a game may draw once, translate at once). The target recurrence uses the targets'
// guest addresses, noted here for the draws it is asked about. The program rule (no stage writes
// memory) can only be checked once the translation is ready: a shader whose program refuses is
// remembered (NoteAsyncTranslationRefusals), and its later sources translate at once.
static bool AsyncTranslationDrawAllowed(CommandBuffer& buffer) {
	auto&       ctx   = buffer.GetRegisters();
	auto&       rc    = buffer.GetContext();
	const auto& depth = ctx.GetDepthControl();
	if (!depth.z_enable || (rc.GetOcclusionCounter().Active() && !SkipOcclusionVisibleEnabled())) return false;
	const auto& render = ctx.GetRenderControl();
	if (render.depth_clear_enable || render.stencil_clear_enable) return false;
	struct Seen {
		uint64_t frame  = 0;
		uint32_t frames = 0;
	};
	thread_local std::unordered_map<uint64_t, Seen> seen;
	if (seen.size() > 8192u) seen.clear();
	const auto frame     = rc.GetTextureCache().Frame() + 1u;
	bool       recurrent = true;
	const auto note      = [&](uint64_t address) {
		auto& entry = seen[address];
		if (entry.frame != frame) {
			entry.frame  = frame;
			entry.frames = std::min<uint32_t>(entry.frames + 1u, 1000u);
		}
		recurrent &= entry.frames - 1u >= 2u; // earlier frames only
	};
	uint32_t colors      = 0;
	bool     small_color = false;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if (!render_target_bound(rt) || render_target_mask_slot(ctx.GetRenderTargetMask(), slot) == 0) {
			continue;
		}
		if (RenderIsColorTileModeLinear(rt.attrib3.tile_mode)) return false; // the CPU reads it back
		colors++;
		small_color |= rt.attrib2.width < 256u || rt.attrib2.height < 256u;
		note(rt.base.addr);
	}
	const auto& z         = ctx.GetDepthRenderTarget();
	const bool  has_depth = z.z_info.format != Prospero::DepthFormat::kInvalid && z.z_write_base_addr != 0;
	if (has_depth) note(z.z_write_base_addr);
	if (recurrent || !SkipRecurrentTargetsOnly()) return true;
	// KYTY_SKIP_EXEMPT's G-buffer pass from the registers (its cube rule needs the resolved image).
	return SkipExemptEnabled() && colors >= 2u && !small_color && has_depth && depth.z_write_enable &&
	       static_cast<vk::CompareOp>(depth.zfunc) != vk::CompareOp::eAlways;
}

static void NoteAsyncTranslationRefusals(const DrawRenderState& state) {
	const auto check = [](const ShaderStageRuntime& stage) {
		if (stage.program != nullptr && !DrawRunProgramEligible(*stage.program, false, SkipFaultBufferEnabled())) {
			PipelineCache::NoteAsyncTranslationRefused(stage.program->shader_hash);
		}
	};
	for (uint32_t i = 0; i < state.vertex_stages_written && i < state.vertex_info.size(); i++) {
		check(state.vertex_info[i].stage);
	}
	if (state.ps_active) check(state.ps_input_info.stage);
}

// KYTY_PIPELINE_SKIP_STATS=1 (diagnostics): the time draws spent in the pipeline lookup by why they
// could or could not be skipped, logged every 10 s (where a cold start's CP pipeline waits remain).
static bool SkipStatsEnabled() {
	static const bool on = [] {
		const auto* value = std::getenv("KYTY_PIPELINE_SKIP_STATS");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	return on;
}

static void NoteSkipLookup(SkipRefusal reason, uint64_t ns) {
	static uint64_t totals_ns[static_cast<size_t>(SkipRefusal::Count)] {};
	static uint64_t slow[static_cast<size_t>(SkipRefusal::Count)] {};
	static auto     last = std::chrono::steady_clock::now();
	totals_ns[static_cast<size_t>(reason)] += ns;
	if (ns > 1'000'000) slow[static_cast<size_t>(reason)]++;
	const auto now = std::chrono::steady_clock::now();
	if (now - last < std::chrono::seconds(10)) return;
	last = now;
	std::string line = "Pipeline skip stats (lookup ms, lookups over 1 ms):";
	for (size_t i = 0; i < static_cast<size_t>(SkipRefusal::Count); i++) {
		line += fmt::format(" {} {:.1f}/{}", SkipRefusalNames[i], static_cast<double>(totals_ns[i]) / 1e6, slow[i]);
	}
	LOGF("%s\n", line.c_str());
}

void RenderExecutor::BeginDrawRun() {
	m_acq_seed_prev      = m_acq_seed.valid; // KYTY_IMG_ACQSEED
	m_acq_seed.valid     = false;
	m_acqfb_prev         = m_acqfb.valid; // KYTY_IMG_ACQFB
	m_acqfb.valid        = false;
	m_seed_why_prev      = m_seed_why;
	m_seed_why           = 0;
	m_run_prev_valid     = m_run.valid;
	m_run_keepable       = m_run.valid;
	m_run.valid          = false;
	m_run_active         = false;
	m_run_recall         = false;
	m_run_verify         = false;
	m_run_targets_kept   = false;
	m_run_pipeswap       = false;
	m_prepared_validated = false;
	m_kernel_verify      = false;
	m_run_texdelta       = false;
	m_texdelta_verify    = false;
	if (CommitStats::Enabled()) [[unlikely]] {
		m_stats_kernel    = CommitStats::KernelOutcome::Off;
		m_stats_candidate = CommitStats::CandidateOutcome::NotReached;
	}
	if (!m_in_engine_commit && DrawRun::Enabled()) {
		DrawRun::NoteForeignActivity("draw outside an engine commit");
	}
	if (DrawRun::KeyDiagEnabled()) [[unlikely]] {
		m_key_diag.prev_reason = m_key_diag.reason;
		m_key_diag.reason      = m_in_engine_commit ? DrawRun::KeyDiag::Inelig::NotReached
		                                            : DrawRun::KeyDiag::Inelig::NotEngine;
		m_key_diag.combo       = -2;
	}
	if (CommitStats::Enabled()) [[unlikely]] {
		m_stats_prev_inelig = m_stats_inelig;
		m_stats_inelig      = static_cast<uint8_t>(m_in_engine_commit ? DrawRun::KeyDiag::Inelig::NotReached
		                                                                   : DrawRun::KeyDiag::Inelig::NotEngine);
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
                                           bool log_change, bool attachment_serials) const {
	return DrawRunMarksChange(m_run.images, compare_serials, attachments_only, log_change, attachment_serials);
}

uint32_t RenderExecutor::DrawRunMarksChange(std::span<const DrawRunImage> marks, bool compare_serials,
                                          bool attachments_only, bool log_change,
                                          bool attachment_serials) const {
	auto&            cache = m_context.GetTextureCache();
	std::scoped_lock lock {cache.m_lock};
	const auto&      images = cache.m_slot_images;
	for (uint32_t index = 0; index < marks.size(); index++) {
		const auto& mark = marks[index];
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
		} else if (compare_serials && (mark.texture || attachment_serials) &&
		           image->ContentSerial() != mark.serial) {
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

// KYTY_DRAW_RUN_IMAGE_EPOCH (drawRun.h): DrawRunImagesChange is skipped while no image property it
// reads changed since the record's marks were proven current.
bool RenderExecutor::DrawRunImagesUnchanged(bool compare_serials, bool attachments_only) const {
	const auto mode  = DrawRun::ImageEpochMode();
	const auto epoch = ImageStateEpoch();
	if (mode != 0 && m_run.image_epoch != 0 && epoch == m_run.image_epoch) {
		auto& totals = DrawRun::GetTotals();
		if (mode == 1) {
			OwnedCounters::Add(totals.epoch_skips, 1);
			return true;
		}
		totals.epoch_checks.fetch_add(1, std::memory_order_relaxed);
		const auto change =
		    DrawRunImagesChange(compare_serials, attachments_only, false, !m_run_recall);
		if (change != 0) {
			totals.epoch_mismatches.fetch_add(1, std::memory_order_relaxed);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				std::printf("DrawRunImageEpochVerify: an image changed without an epoch bump "
				            "(change 0x%x)\n",
				            change);
				std::fflush(stdout);
			}
			if (mode == 3) {
				EXIT("DrawRunImageEpochVerify: an image changed without an epoch bump\n");
			}
		}
		return change == 0;
	}
	const bool unchanged =
	    DrawRunImagesChange(compare_serials, attachments_only, false, !m_run_recall) == 0;
	// A full check that passed proves the marks current as of `epoch` (read before it).
	if (unchanged && compare_serials && !attachments_only && !m_run_recall) {
		m_run.image_epoch = epoch;
	}
	return unchanged;
}

void RenderExecutor::DrawRunMarkBindings(bool attachments_only) {
	auto& images = m_context.GetTextureCache().m_slot_images;
	for (const auto& mark: m_run.images) {
		if (attachments_only && mark.texture) {
			continue; // KYTY_DRAW_RUN_TEXDELTA: the draw's own textures are bound by their resolution
		}
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

// KYTY_DRAW_RUN_TARGETS (drawPrep/drawRun.h): the record's targets are what this draw's resolution
// would give (the same inputs, the run certificate and unchanged attachments).
bool RenderExecutor::DrawRunTargetsCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                             uint32_t render_target_slice_offset) const {
	const auto& run   = m_run;
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	return m_run_prev_valid && !m_run_recall && m_in_engine_commit && run.acquire_valid &&
	       run.depth.size() == 1 && run.colors.size() == run.color_count &&
	       m_run_targets_key != 0 && m_run_targets_key == run.targets_key &&
	       state.ps_active == run.ps_active && pixel == run.pixel_program &&
	       render_target_slice_offset == run.slice_offset &&
	       DrawRun::ActivityEpoch() == run.activity && DrawRunCommandUnchanged(buffer) &&
	       DrawRunImagesUnchanged(true, true);
}

// KYTY_DRAW_RUN_KEY_DIAG: a draw whose key differed from the record's compares its resolved
// pipeline and structure with the record's; a draw that recorded becomes the record.
void RenderExecutor::DrawRunKeyDiagResolved(const CommandBuffer& buffer,
                                            std::span<PreparedBindings* const> stages,
                                            const PipelineCache::Pipeline&     pipeline,
                                            const RenderState& rendering, bool recorded) {
	auto&       kd     = m_key_diag;
	const auto  handle = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(static_cast<VkPipeline>(pipeline.pipeline)));
	std::vector<uint64_t> words;
	words.reserve(64);
	const auto add_handle = [&words](auto native) {
		using Native = typename decltype(native)::NativeType;
		words.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<Native>(native))));
	};
	for (const auto* stage: stages) {
		words.push_back(reinterpret_cast<uintptr_t>(stage->runtime->program));
		for (const auto& image: stage->images) {
			add_handle(image.image_view);
			words.push_back(static_cast<uint64_t>(image.layout));
			for (const auto view: image.mip_views) {
				add_handle(view);
			}
		}
		for (const auto sampler: stage->samplers) {
			add_handle(sampler);
		}
		words.push_back(0xffffffffull);
	}
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		add_handle(rendering.color_attachments[i].image_view);
		words.push_back(static_cast<uint64_t>(rendering.color_attachments[i].image_layout));
	}
	add_handle(rendering.depth_stencil_attachment.image_view);
	words.push_back(static_cast<uint64_t>(rendering.depth_stencil_attachment.image_layout));
	const uint64_t structure = XXH3_64bits(words.data(), words.size() * sizeof(uint64_t));
	if (kd.combo != -2 && kd.record_resolved) {
		DrawRun::KeyDiag::CountResolved(kd.combo, handle == kd.record_pipeline,
		                                structure == kd.record_structure);
	}
	if (recorded && m_run_parts != nullptr) {
		kd.record_parts       = *m_run_parts;
		kd.record_parts_valid = true;
		kd.record_pipeline    = handle;
		kd.record_structure   = structure;
		kd.record_resolved    = true;
		auto& last = kd.recent[(kd.recent_next + kd.recent.size() - 1) % kd.recent.size()];
		if (last.key != m_run.key || last.command != m_run.command || last.serial != m_run.rendering_serial) {
			auto& entry   = kd.recent[kd.recent_next];
			kd.recent_next = (kd.recent_next + 1) % static_cast<uint32_t>(kd.recent.size());
			entry         = {m_run.key, m_run.activity, m_run.command, m_run.tick, m_run.rendering_serial};
		} else {
			last.activity = m_run.activity;
			last.tick     = m_run.tick;
		}
	} else if (!recorded) {
		kd.record_parts_valid = false;
		kd.record_resolved    = false;
	}
}

// KYTY_DRAW_RUN_TEXDELTA (drawPrep/texDelta.h): counters, one console line every 2^18 deltas taken
// or verified ("TexDelta: ..."), the first verify mismatches by field.
namespace TexDeltaStats {
enum Field : uint32_t { Targets, Rendering, Feedback, AttachmentState, Transitions, Dynamic, FieldCount };
constexpr const char* kFieldNames[FieldCount] = {"targets",     "rendering", "feedback", "attachments",
                                                 "transitions", "dynamic"};
enum Late : uint32_t { LateImages, LateAttachment, LateStorage, LateOverAttachment, LateCount };
constexpr const char* kLateNames[LateCount] = {"images", "attachment", "storage", "overattachment"};
struct Counters {
	std::array<std::atomic<uint64_t>, 5>          decisions {}; // TexDelta::Decision
	std::atomic<uint64_t>                         taken {0};
	std::array<std::atomic<uint64_t>, LateCount>  late {};
	std::atomic<uint64_t>                         verify_draws {0};
	std::atomic<uint64_t>                         verify_fallback {0}; // would have fallen back late
	std::atomic<uint64_t>                         verify_checks {0};
	std::array<std::atomic<uint64_t>, FieldCount> mismatches {};
};
Counters g_counters;

void Print() {
	auto&       c    = g_counters;
	std::string line = fmt::format(
	    "TexDelta: candidates off {} keymatched {} other {} writes {} delta {} | taken {} late", c.decisions[0].load(),
	    c.decisions[1].load(), c.decisions[2].load(), c.decisions[3].load(), c.decisions[4].load(), c.taken.load());
	for (uint32_t i = 0; i < LateCount; i++) {
		line += fmt::format(" {}={}", kLateNames[i], c.late[i].load());
	}
	line += fmt::format(" | verify draws {} fallback {} checks {} mismatches", c.verify_draws.load(),
	                    c.verify_fallback.load(), c.verify_checks.load());
	uint64_t total = 0;
	for (uint32_t i = 0; i < FieldCount; i++) {
		total += c.mismatches[i].load();
		line += fmt::format(" {}={}", kFieldNames[i], c.mismatches[i].load());
	}
	line += fmt::format(" total={}", total);
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

void Tick(uint64_t n) {
	if ((n & 0x3ffffu) == 0) {
		Print();
	}
}

void Check(Field field, bool same) {
	auto& c = g_counters;
	c.verify_checks.fetch_add(1, std::memory_order_relaxed);
	if (!same) {
		if (c.mismatches[field].fetch_add(1, std::memory_order_relaxed) < 8) {
			std::printf("TexDelta verify: %s mismatch\n", kFieldNames[field]);
			std::fflush(stdout);
		}
	}
}
} // namespace TexDeltaStats

void RenderExecutor::TexDeltaCount(DrawRun::TexDelta::Decision decision, bool taken) {
	auto& c = TexDeltaStats::g_counters;
	if (taken) {
		TexDeltaStats::Tick(c.taken.fetch_add(1, std::memory_order_relaxed) + 1);
		return;
	}
	c.decisions[static_cast<uint32_t>(decision)].fetch_add(1, std::memory_order_relaxed);
}

// KYTY_DRAW_RUN_TEXDELTA: 0 when the delta's resolved textures may keep the record's acquisition,
// else 1 + TexDeltaStats::Late (a texture is an attachment, a storage binding, or lies over an
// attachment's memory).
uint32_t RenderExecutor::TexDeltaTextureVerdict(std::span<PreparedBindings* const> stages) const {
	using DrawRun::TexDelta::AttachmentFacts;
	using DrawRun::TexDelta::TextureFacts;
	const auto id_of = [](ImageId id) {
		return id ? (static_cast<uint64_t>(id.generation) << 32u) | id.index : uint64_t {0};
	};
	static thread_local std::vector<AttachmentFacts> attachments;
	static thread_local std::vector<TextureFacts>    textures;
	attachments.clear();
	textures.clear();
	for (uint32_t i = m_run.attachments_begin; i < m_run.images.size(); i++) {
		const auto& mark = m_run.images[i];
		attachments.push_back({id_of(mark.id), mark.address, mark.size});
	}
	auto&            cache = m_context.GetTextureCache();
	std::scoped_lock lock {cache.m_lock};
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			TextureFacts facts;
			facts.id      = id_of(binding.image_id);
			facts.storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (const auto* image = cache.m_slot_images.try_get(binding.image_id); image != nullptr) {
				facts.address = image->info.data.address;
				facts.size    = image->info.data.size;
			}
			textures.push_back(facts);
		}
	}
	const auto verdict = DrawRun::TexDelta::CheckTextures(textures, attachments);
	// TextureVerdict Attachment/Storage/OverAttachment are TexDeltaStats::Late 1..3.
	return verdict == DrawRun::TexDelta::TextureVerdict::Ok ? 0u : 1u + static_cast<uint32_t>(verdict);
}

// KYTY_DRAW_RUN_TEXDELTA=2: a would-be delta took the normal path (no target keep, no acquisition
// reuse); what the delta keeps from the record must be what that path computed.
void RenderExecutor::TexDeltaVerify(const DrawRenderState& state, const RenderState& rendering,
                                    vk::ImageAspectFlags feedback_aspects, bool /*dynamic_emitted*/,
                                    std::span<PreparedBindings* const> /*stages*/) {
	using TexDeltaStats::Check;
	const auto& run = m_run;
	if (run.acquire_valid && run.depth.size() == 1 && run.colors.size() == run.color_count) {
		bool same = state.color_count == run.color_count && state.color_slots_written == run.color_slots &&
		            SameRenderDepthInfo(state.depth_info, run.depth[0]);
		for (uint32_t i = 0; same && i < state.color_count; i++) {
			same = SameRenderColorInfo(state.color_info[i], run.colors[i]);
		}
		Check(TexDeltaStats::Targets, same);
	}
	Check(TexDeltaStats::Rendering, rendering == run.rendering);
	Check(TexDeltaStats::Feedback, !feedback_aspects);
	Check(TexDeltaStats::AttachmentState, DrawRunImagesChange(false, true, false) == 0);
}
bool RenderExecutor::DrawRunCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                      uint32_t render_target_slice_offset) {
	using DrawRun::Miss;
	using Cand = CommitStats::CandidateOutcome;
	// KYTY_CP_COMMIT_STATS: the outcome for the change classes.
	const auto note = [this](Cand outcome) {
		if (CommitStats::Enabled()) [[unlikely]] {
			m_stats_candidate = outcome;
		}
	};
	auto& totals = DrawRun::GetTotals();
	OwnedCounters::Add(totals.draws, 1);
	if (m_run_parts != nullptr) [[unlikely]] {
		// KYTY_DRAW_RUN_KEY_DIAG: why the key check fails.
		auto& kd = m_key_diag;
		if (!m_run_prev_valid) {
			DrawRun::KeyDiag::CountPrevInvalid(kd.prev_reason, m_run_key != 0 && m_run_key == kd.last_key);
		} else if (m_run_key != 0 && m_run_key != m_run.key && kd.record_parts_valid) {
			uint32_t mask = 0;
			for (uint32_t i = 0; i < DrawRun::KpCount; i++) {
				mask |= (*m_run_parts)[i] != kd.record_parts[i] ? (1u << i) : 0u;
			}
			bool recallable = false;
			bool any        = false;
			for (const auto& recent: kd.recent) {
				if (recent.key == m_run_key && recent.command == buffer.Identity() &&
				    recent.tick == m_context.GetCommandScheduler().CurrentTick() &&
				    recent.serial != 0 && recent.serial == buffer.ActiveRenderingSerial()) {
					any        = true;
					recallable = recallable || recent.activity == DrawRun::ActivityEpoch();
				}
			}
			kd.combo = DrawRun::KeyDiag::CountKeyMismatch(mask, recallable, any);
		}
		kd.last_key = m_run_key;
	}
	bool pipeswap = false;
	bool texdelta = false;
	if (!m_run_prev_valid || m_run_key == 0 || m_run_key != m_run.key) {
		// KYTY_DRAW_RUN_TEXDELTA (texDelta.h): the key differs in the T#/S# words alone.
		const auto td_mode = DrawRun::TexDeltaMode();
		if (td_mode != 0 && DrawRun::GetMode() == DrawRun::Mode::On) {
			DrawRun::TexDelta::Inputs in;
			in.mode            = td_mode;
			in.record_valid    = m_run_prev_valid;
			in.key             = m_run_key;
			in.record_key      = m_run.key;
			in.free_key        = m_run_texfree_key;
			in.record_free_key = m_run.texfree_key;
			in.record_writes   = m_run.writes;
			auto decision      = DrawRun::TexDelta::Decide(in);
			if (decision == DrawRun::TexDelta::Decision::Delta) {
				const auto writes = [](const ShaderStageRuntime& runtime) {
					return !DrawRunProgramEligible(*runtime.program, false, g_draw_run_fault_buffer.On());
				};
				in.draw_writes = writes(state.vertex_info[0].stage) ||
				                 (state.ps_active && writes(state.ps_input_info.stage));
				decision       = DrawRun::TexDelta::Decide(in);
			}
			TexDeltaCount(decision);
			texdelta = decision == DrawRun::TexDelta::Decision::Delta;
		}
		// KYTY_DRAW_RUN_PIPESWAP: the key differs in the vertex program alone.
		if (!texdelta && (!m_run_prev_valid || m_run_key == 0 || m_run_swap_key == 0 ||
		                  m_run_swap_key != m_run.swap_key || !DrawRun::PipeSwapEnabled())) {
			note(m_run_prev_valid && m_run_key != 0 ? Cand::Key : Cand::NoRecord);
			return false;
		}
		if (!texdelta) {
			pipeswap = true;
			OwnedCounters::Add(totals.swap_matches, 1);
		}
	} else {
		OwnedCounters::Add(totals.key_matches, 1);
	}
	if (DrawRun::ActivityEpoch() != m_run.activity) {
		DrawRun::CountMiss(Miss::Activity);
		note(Cand::Activity);
		return false;
	}
	if (buffer.Identity() != m_run.command ||
	    m_context.GetCommandScheduler().CurrentTick() != m_run.tick) {
		DrawRun::CountMiss(Miss::Command);
		note(Cand::Command);
		return false;
	}
	if (buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != m_run.rendering_serial) {
		DrawRun::CountMiss(Miss::Instance);
		note(Cand::Instance);
		return false;
	}
	if (!m_prepared_validated) {
		DrawRun::CountMiss(Miss::Validation);
		note(Cand::Validation);
		return false;
	}
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	if (state.ps_active != m_run.ps_active || render_target_slice_offset != m_run.slice_offset ||
	    state.vertex_stages_written != 1 ||
	    (!pipeswap && state.vertex_info[0].stage.program != m_run.vertex_program) ||
	    pixel != m_run.pixel_program) {
		DrawRun::CountMiss(Miss::Programs);
		note(Cand::Programs);
		return false;
	}
	bool swap_writes = false;
	if (pipeswap) {
		const auto* recorded = static_cast<const ShaderRecompiler::IR::CompiledShaderInfo*>(
		    m_run.vertex_program);
		const auto& program = *state.vertex_info[0].stage.program;
		if (recorded == nullptr || recorded == &program || !DrawRunPipeSwapCompatible(*recorded, program)) {
			OwnedCounters::Add(totals.swap_incompatible, 1);
			note(Cand::SwapIncompatible);
			return false;
		}
		// The new vertex program: run-eligible, and its buffer writes (if allowed) off the run's images.
		if (!DrawRunProgramEligible(program, false, g_draw_run_fault_buffer.On())) {
			if (!m_run.record_writes_ok ||
			    !DrawRunProgramEligible(program, true, g_draw_run_fault_buffer.On())) {
				DrawRun::CountMiss(Miss::Writes);
				note(Cand::Writes);
				return false;
			}
			switch (DrawRunWritesOverRun(state.vertex_info[0].stage, m_run.images)) {
				case 0: break;
				case 1: swap_writes = true; break;
				default:
					DrawRun::CountMiss(Miss::Writes);
					note(Cand::Writes);
					return false;
			}
		}
	}
	// KYTY_DRAW_RUN_WRITES: a record of a writing program exists only while the switch is on (a
	// change ends the run). The candidate's own writes must not lie over an image of the run.
	bool writes = false;
	if (m_run.writes) {
		const ShaderStageRuntime* runtimes[] = {&state.vertex_info[0].stage,
		                                        state.ps_active ? &state.ps_input_info.stage : nullptr};
		for (const auto* runtime: runtimes) {
			if (runtime == nullptr) {
				continue;
			}
			switch (DrawRunWritesOverRun(*runtime, m_run.images)) {
				case 0: break;
				case 1: writes = true; break;
				default:
					DrawRun::CountMiss(Miss::Writes);
					note(Cand::Writes);
					return false;
			}
		}
	}
	if (texdelta) {
		// KYTY_DRAW_RUN_TEXDELTA=2: the normal path runs and is compared (TexDeltaVerify).
		if (DrawRun::TexDeltaMode() == 2) {
			m_texdelta_verify = true;
			return false;
		}
		m_run_texdelta = true;
		TexDeltaCount(DrawRun::TexDelta::Decision::Delta, true);
	}
	note(Cand::Continued);
	OwnedCounters::Add(totals.continued, 1);
	if (writes || swap_writes) {
		OwnedCounters::Add(totals.write_continued, 1);
	}
	if (pipeswap) {
		OwnedCounters::Add(totals.swapped, 1);
		m_run_pipeswap = true;
	}
	return true;
}

bool RenderExecutor::DrawRunPipeSwapCompatible(const ShaderRecompiler::IR::CompiledShaderInfo& recorded,
                                               const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	const auto& a = recorded.info;
	const auto& b = program.info;
	if (a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size()) {
		return false;
	}
	for (size_t i = 0; i < a.images.size(); i++) {
		auto x = a.images[i];
		auto y = b.images[i];
		x.source = y.source = 0;
		x.first_use_pc = y.first_use_pc = 0;
		if (!(x == y)) {
			return false;
		}
	}
	for (size_t i = 0; i < a.samplers.size(); i++) {
		auto x = a.samplers[i];
		auto y = b.samplers[i];
		x.source = y.source = 0;
		x.first_use_pc = y.first_use_pc = 0;
		if (!(x == y)) {
			return false;
		}
	}
	const auto viewport_index = [](const auto& outputs) {
		return std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
			return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
		});
	};
	return viewport_index(a.outputs) == viewport_index(b.outputs);
}

// KYTY_DRAW_RUN_RECALL: a hash of a binding's mip views (RecallTexture::mips).
static uint64_t DrawRunMipViewsHash(const std::vector<vk::ImageView>& views) {
	return views.empty() ? 0 : XXH3_64bits(views.data(), views.size() * sizeof(vk::ImageView));
}

// KYTY_DRAW_RUN_RECALL: the stage's texture set of these words (-1: the current one, 0..2 its
// history; -2: none) when it holds exactly the bindings the record committed.
template <typename RecallTextures>
static int DrawRunRecallSet(const PreparedBindings& prepared, const ShaderStageRuntime& runtime,
                            const RecallTextures& textures) {
	const auto& program = *runtime.program;
	const auto& words   = runtime.resources->images;
	if (program.info.images.size() != textures.size()) {
		return -2;
	}
	const auto same = [&](const std::vector<TextureBinding>& images) {
		if (images.size() != textures.size()) {
			return false;
		}
		for (size_t i = 0; i < images.size(); i++) {
			const auto& binding = images[i];
			const auto& kept    = textures[i];
			if (binding.image_id != kept.id || binding.image_view != kept.view ||
			    binding.layout != kept.layout || DrawRunMipViewsHash(binding.mip_views) != kept.mips) {
				return false;
			}
		}
		return true;
	};
	if (textures.empty()) {
		return -1; // nothing to take back: the stage's images are cleared
	}
	if (prepared.texture_program == &program && std::ranges::equal(prepared.texture_words, words)) {
		return same(prepared.images) ? -1 : -2;
	}
	for (size_t i = 0; i < prepared.texture_history.size(); i++) {
		const auto& set = prepared.texture_history[i];
		if (set.program == &program && std::ranges::equal(set.words, words)) {
			return same(set.images) ? static_cast<int>(i) : -2;
		}
	}
	return -2;
}

// KYTY_DRAW_RUN_RECALL (drawPrep/drawRun.h): whether the committed draw continues a kept record of
// its rendering instance (PrepareDrawRenderState, after DrawRunCandidate declined a plain
// continuation). On success the record is m_run (the previous one kept in its place) and, unless in
// verify mode, the stages hold the record's texture sets and samplers and `state` its targets.
bool RenderExecutor::DrawRunRecallCandidate(const CommandBuffer& buffer, DrawRenderState& state,
                                            uint32_t render_target_slice_offset) {
	if (m_run_key == 0 || (m_run_prev_valid && m_run_key == m_run.key)) {
		return false;
	}
	DrawRunRecord* kept = nullptr;
	for (auto& record: m_run_kept) {
		if (record.valid && record.key == m_run_key) {
			kept = &record;
			break;
		}
	}
	if (kept == nullptr) {
		return false;
	}
	auto& totals = DrawRun::GetTotals();
	totals.recall_matches.fetch_add(1, std::memory_order_relaxed);
	auto& record = *kept;
	if (DrawRun::ActivityEpoch() != record.activity || buffer.Identity() != record.command ||
	    m_context.GetCommandScheduler().CurrentTick() != record.tick ||
	    buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != record.rendering_serial) {
		record.valid = false; // it can never be recalled again
		return false;
	}
	const void* pixel = state.ps_active ? state.ps_input_info.stage.program : nullptr;
	if (!record.recall_valid || !record.acquire_valid || record.depth.size() != 1 ||
	    record.colors.size() != record.color_count || !m_prepared_validated ||
	    state.ps_active != record.ps_active || render_target_slice_offset != record.slice_offset ||
	    state.vertex_stages_written != 1 ||
	    state.vertex_info[0].stage.program != record.vertex_program || pixel != record.pixel_program ||
	    (state.ps_active && !m_graphics_bindings.pixel.has_value())) {
		return false;
	}
	const auto vertex_textures = std::span(record.recall_textures).first(record.recall_vertex_textures);
	const auto pixel_textures  = std::span(record.recall_textures).subspan(record.recall_vertex_textures);
	auto&      vertex          = m_graphics_bindings.vertex[0];
	const auto vertex_set      = DrawRunRecallSet(vertex, state.vertex_info[0].stage, vertex_textures);
	const auto pixel_set       = state.ps_active ? DrawRunRecallSet(*m_graphics_bindings.pixel,
	                                                                state.ps_input_info.stage, pixel_textures)
	                                             : (pixel_textures.empty() ? -1 : -2);
	const auto pixel_samplers  = record.recall_samplers.size() - record.recall_vertex_samplers;
	if (vertex_set == -2 || pixel_set == -2 ||
	    state.vertex_info[0].stage.program->info.samplers.size() != record.recall_vertex_samplers ||
	    (state.ps_active ? state.ps_input_info.stage.program->info.samplers.size() : 0) !=
	        pixel_samplers) {
		totals.recall_sets_missed.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	// KYTY_DRAW_RUN_WRITES: the candidate's own writes must not lie over an image of the record.
	if (record.writes) {
		const ShaderStageRuntime* runtimes[] = {&state.vertex_info[0].stage,
		                                        state.ps_active ? &state.ps_input_info.stage : nullptr};
		for (const auto* runtime: runtimes) {
			if (runtime != nullptr && DrawRunWritesOverRun(*runtime, record.images) > 1) {
				DrawRun::CountMiss(DrawRun::Miss::Writes);
				return false;
			}
		}
	}
	// The record becomes the run; the previous one (complete or not) takes its place.
	std::swap(m_run, record);
	record.valid     = m_run_keepable && record.key != m_run.key;
	m_run.valid      = false; // the draw's own record (DrawRunRecordDraw) validates it again
	m_run_prev_valid = true;
	m_run_keepable   = true;
	m_run_recall     = true;
	totals.recalled.fetch_add(1, std::memory_order_relaxed);
	if (DrawRun::GetMode() != DrawRun::Mode::On) {
		return true; // verify: the normal path runs and is compared with the record
	}
	// The record's bindings: each stage's texture set back from its history, its samplers, and the
	// record's targets.
	const auto restore = [](PreparedBindings& prepared, const ShaderStageRuntime& runtime, int set,
	                        std::span<const vk::Sampler> samplers) {
		if (runtime.program->info.images.empty()) {
			prepared.images.clear();
			prepared.texture_program = nullptr;
		} else if (set >= 0) {
			auto& found = prepared.texture_history[static_cast<size_t>(set)];
			std::swap(found.program, prepared.texture_program);
			std::swap(found.words, prepared.texture_words);
			std::swap(found.images, prepared.images);
		}
		prepared.samplers.assign(samplers.begin(), samplers.end());
	};
	const auto samplers = std::span<const vk::Sampler>(m_run.recall_samplers);
	restore(vertex, state.vertex_info[0].stage, vertex_set,
	        samplers.first(m_run.recall_vertex_samplers));
	if (state.ps_active) {
		restore(*m_graphics_bindings.pixel, state.ps_input_info.stage, pixel_set,
		        samplers.subspan(m_run.recall_vertex_samplers));
	}
	for (uint32_t i = 0; i < m_run.color_count; i++) {
		state.color_info[i] = m_run.colors[i];
	}
	state.depth_info = m_run.depth[0];
	return true;
}

// KYTY_DRAW_RUN_RECALL: before a draw records a new run, the record it replaces is kept (in the
// entry of the same key, else the next one in turn).
void RenderExecutor::DrawRunKeep() {
	if (!m_run_keepable || !m_run.recall_valid || m_run.key == m_run_key || !DrawRun::RecallEnabled()) {
		return;
	}
	m_run_keepable = false;
	auto slot      = m_run_kept_next;
	for (uint32_t i = 0; i < m_run_kept.size(); i++) {
		if (m_run_kept[i].key == m_run.key) {
			slot = i;
			break;
		}
	}
	if (slot == m_run_kept_next) {
		m_run_kept_next = (m_run_kept_next + 1) % static_cast<uint32_t>(m_run_kept.size());
	}
	std::swap(m_run, m_run_kept[slot]);
	m_run_kept[slot].valid = true;
}

// KYTY_DRAW_RUN_WRITES: 0 when the stage writes no buffer, 1 when it writes only outside the run's
// images, 2 when a written V# range (as the descriptor states it, before any clamp) lies over an
// image of the run (a texture or an attachment) or the run has no image marks to compare.
uint32_t RenderExecutor::DrawRunWritesOverRun(const ShaderStageRuntime&     runtime,
                                              std::span<const DrawRunImage> images) {
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	uint32_t    result   = 0;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource = program.info.buffers[i];
		if (!resource.written && !resource.atomic) {
			continue;
		}
		if (i >= snapshot.buffers.size() || snapshot.buffers[i].dword_count < 4) {
			return 2;
		}
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
		const auto address    = descriptor.Base48();
		const auto size       = descriptor.GetSize();
		if (address == 0 || size == 0) {
			continue; // a null V#: nothing is written (HasShaderBufferWrites)
		}
		result = 1;
		const auto end = address + size < address ? UINT64_MAX : address + size;
		for (const auto& mark: images) {
			if (mark.size != 0 && address < mark.address + mark.size && mark.address < end) {
				return 2;
			}
		}
	}
	return result;
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

uint8_t RenderExecutor::DrawRunSeedAcquire(const CommandBuffer& buffer, const DrawRenderState& state,
                                        const RenderState& rendering, std::span<PreparedBindings* const> stages,
                                        const vk::Rect2D* written) {
	auto& seed = m_acq_seed;
	seed.valid = false;
	auto& cache = m_context.GetTextureCache();
	{
		std::scoped_lock lock {cache.m_lock};
		// As DrawRunRecordDraw: a draw that just established an unwritten-instance depth proof seeds
		// nothing (the next acquisition must run to publish the union and its deferred ordering).
		if (state.depth_info.image_id && DepthFeedbackKeepEnabled() && !DepthFeedbackLazyEnabled() &&
		    !state.depth_info.AttachmentWriteAspects()) {
			const auto* depth = cache.m_slot_images.try_get(state.depth_info.image_id);
			const auto attachment_access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
			                               vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
			if (depth != nullptr && depth->backing.state.access_mask == attachment_access &&
			    depth->feedback_instance == buffer.ActiveRenderingSerial() &&
			    depth->feedback_serial == depth->ContentSerial()) {
				return 11;
			}
		}
		seed.marks.clear();
		for (uint32_t i = 0; i < state.color_count; i++) {
			seed.marks.push_back(MakeDrawRunImage(state.color_info[i].image_id, false));
		}
		if (state.depth_info.image_id) {
			seed.marks.push_back(MakeDrawRunImage(state.depth_info.image_id, false));
		}
		// As DrawRunRecordDraw: no texture of the draw over an attachment's memory (an alias of it).
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				const auto* image = cache.m_slot_images.try_get(binding.image_id);
				if (image == nullptr || DrawRunOverAttachment(seed.marks, image->info.data.address,
				                                              image->info.data.size)) {
					return 12;
				}
			}
		}
	}
	// Buffer writes (V# writes and atomics) of the draw: none over an attachment's memory (the
	// written ranges as the descriptors state them; KYTY_DRAW_RUN_WRITES's check).
	for (const auto* stage: stages) {
		if (stage->runtime != nullptr && DrawRunWritesOverRun(*stage->runtime, seed.marks) > 1) {
			return 7;
		}
	}
	seed.activity         = DrawRun::ActivityEpoch();
	seed.command          = buffer.Identity();
	seed.tick             = m_context.GetCommandScheduler().CurrentTick();
	seed.rendering_serial = buffer.ActiveRenderingSerial();
	seed.rendering        = rendering;
	seed.colors.assign(state.color_info, state.color_info + state.color_count);
	seed.depth.assign(1, state.depth_info);
	seed.bounded = written != nullptr;
	seed.written = written != nullptr ? *written : vk::Rect2D {};
	seed.valid   = true;
	ImgSide::Bump(ImgSide::S::AcqSeedRecords);
	return 13;
}

bool RenderExecutor::DrawRunSeedCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                          std::span<PreparedBindings* const> stages,
                                          const vk::Rect2D* written) const {
	// DrawRunAcquireCandidate's conditions, against the seed.
	const auto& seed = m_acq_seed;
	// (BeginDrawRun moved the seed's validity into m_acq_seed_prev.)
	if (!m_acq_seed_prev || !m_in_engine_commit || seed.colors.size() != state.color_count ||
	    DrawRun::ActivityEpoch() != seed.activity || buffer.Identity() != seed.command ||
	    m_context.GetCommandScheduler().CurrentTick() != seed.tick || buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != seed.rendering_serial || (written != nullptr) != seed.bounded ||
	    (written != nullptr && *written != seed.written)) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], seed.colors[i])) {
			return false;
		}
	}
	if (seed.depth.size() != 1 || !SameRenderDepthInfo(state.depth_info, seed.depth[0])) {
		return false;
	}
	{
		auto&            cache = m_context.GetTextureCache();
		std::scoped_lock lock {cache.m_lock};
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				const auto* image = cache.m_slot_images.try_get(binding.image_id);
				if (image == nullptr) {
					return false;
				}
				for (const auto& mark: seed.marks) {
					if (mark.id == binding.image_id) {
						return false;
					}
				}
				if (DrawRunOverAttachment(seed.marks, image->info.data.address, image->info.data.size)) {
					return false;
				}
			}
		}
	}
	return DrawRunMarksChange(seed.marks, true, true, false, !m_run_recall) == 0;
}

RenderExecutor::AcqFbProbe RenderExecutor::AcqFbBefore(const CommandBuffer&   buffer,
                                                       const DrawRenderState& state) const {
	// The targets (images, slots, levels/layers, depth image, command buffer), as KYTY_IMG_STATS's
	// AcquireSame signature, and the attachments' marks before the acquisition.
	AcqFbProbe probe;
	uint64_t   signature = 0x9e3779b97f4a7c15ull ^ state.color_count;
	const auto mix       = [&](uint64_t v) { signature = (signature ^ v) * 0x100000001b3ull; };
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color = state.color_info[i];
		mix(color.image_id.index | (uint64_t {color.image_id.generation} << 32u));
		mix(color.target_slot | (uint64_t {color.guest_mip_level} << 16u) |
		    (uint64_t {color.guest_array_layer} << 32u));
	}
	mix(state.depth_info.image_id.index | (uint64_t {state.depth_info.image_id.generation} << 32u));
	mix(reinterpret_cast<uint64_t>(static_cast<VkCommandBuffer>(buffer.Identity())));
	probe.signature   = signature;
	probe.instance    = buffer.ActiveRenderingSerial() != 0;
	probe.transitions = Image::RecordedTransitions();
	probe.keeps       = m_acq_keeps;
	auto&            cache = m_context.GetTextureCache();
	std::scoped_lock lock {cache.m_lock};
	for (uint32_t i = 0; i < state.color_count && probe.count < probe.marks.size(); i++) {
		probe.marks[probe.count++] = MakeDrawRunImage(state.color_info[i].image_id, false);
	}
	if (state.depth_info.image_id && probe.count < probe.marks.size()) {
		probe.marks[probe.count++] = MakeDrawRunImage(state.depth_info.image_id, false);
	}
	return probe;
}

void RenderExecutor::AcqFbAfter(const AcqFbProbe& probe, const RenderState& rendering,
                                vk::ImageAspectFlags feedback) {
	if (m_acq_last.valid && m_acq_last.signature == probe.signature && !m_run_prev_valid) {
		ImgSide::Bump(ImgSide::S::AcqFbSame);
		const auto marks = std::span<const DrawRunImage>(probe.marks.data(), probe.count);
		if (!(rendering == m_acq_last.rendering) || feedback != m_acq_last.feedback) {
			ImgSide::Bump(ImgSide::S::AcqFbState);
		} else if (m_acq_keeps != probe.keeps) {
			ImgSide::Bump(ImgSide::S::AcqFbKeep);
		} else if (Image::RecordedTransitions() != probe.transitions) {
			ImgSide::Bump(ImgSide::S::AcqFbTransit);
		} else if (DrawRunMarksChange(marks, false, false, false, true) != 0) {
			ImgSide::Bump(ImgSide::S::AcqFbMarks);
		} else {
			ImgSide::Bump(ImgSide::S::AcqFbRepeat);
			if (DrawRunMarksChange(marks, true, false, false, true) == 0) {
				ImgSide::Bump(ImgSide::S::AcqFbRepeatSerial);
			}
			if (feedback) {
				ImgSide::Bump(ImgSide::S::AcqFbFeedback);
			}
			if (!probe.instance) {
				ImgSide::Bump(ImgSide::S::AcqFbNoInstance);
			}
		}
	}
	m_acq_last.valid     = true;
	m_acq_last.signature = probe.signature;
	m_acq_last.rendering = rendering;
	m_acq_last.feedback  = feedback;
}

void RenderExecutor::AcqFbRecord(const CommandBuffer& buffer, const DrawRenderState& state,
                                 const RenderState& rendering, vk::ImageAspectFlags feedback,
                                 std::span<PreparedBindings* const> stages, const vk::Rect2D* written) {
	// KYTY_IMG_ACQFB: the seed, taken after the draw (its attachments as it left them).
	auto& seed = m_acqfb;
	seed.valid = false;
	if (HtileFullRangeEnabled() || state.color_count > RENDER_COLOR_ATTACHMENTS_MAX) {
		return; // a per-frame HTILE fill decided inside the acquisition
	}
	{
		auto&            cache = m_context.GetTextureCache();
		std::scoped_lock lock {cache.m_lock};
		seed.marks.clear();
		for (uint32_t i = 0; i < state.color_count; i++) {
			seed.marks.push_back(MakeDrawRunImage(state.color_info[i].image_id, false));
		}
		if (state.depth_info.image_id) {
			seed.marks.push_back(MakeDrawRunImage(state.depth_info.image_id, false));
		}
	}
	// Buffer writes (V# writes and atomics) of the draw: none over an attachment's memory.
	for (const auto* stage: stages) {
		if (stage->runtime != nullptr && DrawRunWritesOverRun(*stage->runtime, seed.marks) > 1) {
			return;
		}
	}
	seed.activity         = DrawRun::ActivityEpoch();
	seed.command          = buffer.Identity();
	seed.tick             = m_context.GetCommandScheduler().CurrentTick();
	seed.rendering_serial = buffer.ActiveRenderingSerial();
	seed.rendering        = rendering;
	seed.feedback         = feedback;
	seed.colors.assign(state.color_info, state.color_info + state.color_count);
	seed.depth.assign(1, state.depth_info);
	seed.bounded     = written != nullptr;
	seed.written     = written != nullptr ? *written : vk::Rect2D {};
	seed.has_depth   = m_acq_has_depth && static_cast<bool>(state.depth_info.image_id);
	seed.decision    = m_acq_decision;
	seed.color_bound = m_acq_color_bound;
	seed.valid       = true;
}

bool RenderExecutor::AcqFbCandidate(const CommandBuffer& buffer, const DrawRenderState& state,
                                    std::span<PreparedBindings* const> stages, const vk::Rect2D* written,
                                    DepthAccessDecision& decision) const {
	// KYTY_IMG_ACQFB: AcquireRenderTargets would repeat the seed's acquisition (imgside DESIGN 9.4).
	const auto& seed = m_acqfb;
	// (BeginDrawRun moved the seed's validity into m_acqfb_prev.)
	if (!m_acqfb_prev || !m_in_engine_commit || seed.colors.size() != state.color_count || seed.depth.size() != 1 ||
	    DrawRun::ActivityEpoch() != seed.activity || buffer.Identity() != seed.command ||
	    m_context.GetCommandScheduler().CurrentTick() != seed.tick || buffer.ActiveRenderingSerial() == 0 ||
	    buffer.ActiveRenderingSerial() != seed.rendering_serial || (written != nullptr) != seed.bounded ||
	    (written != nullptr && *written != seed.written) || HtileFullRangeEnabled()) {
		return false;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], seed.colors[i])) {
			return false;
		}
	}
	const auto& depth = state.depth_info;
	if (!SameRenderDepthInfo(depth, seed.depth[0]) || static_cast<bool>(depth.image_id) != seed.has_depth ||
	    depth.depth_clear_enable || depth.stencil_clear_enable) {
		return false;
	}
	auto&      cache        = m_context.GetTextureCache();
	{
		std::scoped_lock lock {cache.m_lock};
		const auto&      images = cache.m_slot_images;
		// Each colour target's layout follows its is_bound (the draw's own texture bindings).
		for (uint32_t i = 0; i < state.color_count; i++) {
			const auto* image = images.try_get(state.color_info[i].image_id);
			if (image == nullptr || image->binding.is_bound != seed.color_bound[i]) {
				return false;
			}
		}
		if (seed.has_depth) {
			const auto* image = images.try_get(depth.image_id);
			if (image == nullptr) {
				return false;
			}
			decision = DecideDepthAccess(buffer, depth, *image, stages);
			const auto& d = seed.decision;
			if (!decision.views_ok || decision.sampled != d.sampled || decision.feedback != d.feedback ||
			    decision.layout != d.layout || decision.access != d.access || decision.keep != d.keep ||
			    decision.adopt != d.adopt) {
				return false;
			}
		}
	}
	// No HTILE clear waits for the depth target (the acquisition would consume it: a load clear).
	const auto& metadata = depth.desc.info.metadata;
	if (seed.has_depth && metadata.kind == ImageMetadataKind::Htile &&
	    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer)) {
		return false;
	}
	// Every attachment as the seeding draw left it, content serials included.
	return DrawRunMarksChange(seed.marks, true, false, false, true) == 0;
}

void RenderExecutor::AcqFbApply(CommandBuffer& buffer, DrawRenderState& state, const DepthAccessDecision& decision) {
	// KYTY_IMG_ACQFB on: the effects of the skipped acquisition that are not repeats of the seed's, in
	// its order: per colour target FindRenderTarget's touch and GPU-write mark (a fresh content
	// serial, as the binding gives it) and the attachment layout/access; for the depth target
	// FindDepthTarget's touch and mark, the serial restored when the draw writes nothing
	// (RestoreContentIfUnwritten), the feedback-keep ordering and union scope (adopt), the
	// attachment layout/access and the depth-feedback note.
	const auto& seed  = m_acqfb;
	auto&       cache = m_context.GetTextureCache();
	{
		std::scoped_lock lock {cache.m_lock};
		for (uint32_t i = 0; i < state.color_count; i++) {
			auto& image = cache.m_slot_images[state.color_info[i].image_id];
			cache.TouchImage(image);
			cache.m_bind_mark = true;
			cache.MarkImageGpuModified(image);
			cache.m_bind_mark = false;
			image.binding.attachment_layout = seed.rendering.color_attachments[state.color_info[i].target_slot].image_layout;
			image.binding.attachment_access =
			    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		}
	}
	if (seed.has_depth) {
		auto&      depth        = state.depth_info;
		const auto content_mark = cache.MarkContent(depth.image_id);
		{
			std::scoped_lock lock {cache.m_lock};
			auto&            image = cache.m_slot_images[depth.image_id];
			cache.TouchImage(image);
			cache.m_stencil_draw_suppressed = true;
			cache.m_bind_mark               = true;
			cache.MarkImageGpuModified(image);
			cache.m_bind_mark               = false;
			cache.m_stencil_draw_suppressed = false;
		}
		depth.load_clear_aspects = {}; // no clear enables, no HTILE clear (AcqFbCandidate)
		const bool writes        = static_cast<bool>(depth.AttachmentWriteAspects());
		if (!writes) {
			cache.RestoreContentIfUnwritten(depth.image_id, content_mark);
		}
		std::scoped_lock lock {cache.m_lock};
		auto&            image = cache.m_slot_images[depth.image_id];
		if (decision.adopt) {
			buffer.NoteFeedbackKeep(FeedbackKeepOrdering(image, decision.layout, decision.sampled_access));
			m_acq_keeps++;
			image.AdoptState(image.backing.state.pl_stage, decision.sampled_access, decision.layout);
		}
		m_depth_feedback                = {depth.image_id, writes, true};
		image.binding.attachment_layout = decision.layout;
		image.binding.attachment_access = decision.access;
	}
	// What the seed records next (a chain of repeats records the same decisions).
	m_acq_decision    = decision;
	m_acq_has_depth   = seed.has_depth;
	m_acq_color_bound = seed.color_bound;
}

uint8_t RenderExecutor::DrawRunAcquireRefusal(const CommandBuffer&               buffer,
                                              const DrawRenderState&             state,
                                              std::span<PreparedBindings* const> stages,
                                              const vk::Rect2D*                  written) const {
	// DrawRunAcquireCandidate's conditions in its order, by reason.
	const auto& run = m_run;
	if (!m_run_prev_valid) {
		return 2;
	}
	if (!m_in_engine_commit) {
		return 3;
	}
	if (!run.acquire_valid || run.depth.size() != 1) {
		return 4;
	}
	if (run.colors.size() != run.color_count || state.color_count != run.color_count) {
		return 5;
	}
	if (DrawRun::ActivityEpoch() != run.activity) {
		return 6;
	}
	if (buffer.Identity() != run.command) {
		return 7;
	}
	if (m_context.GetCommandScheduler().CurrentTick() != run.tick) {
		return 8;
	}
	if (buffer.ActiveRenderingSerial() == 0 || buffer.ActiveRenderingSerial() != run.rendering_serial) {
		return 9;
	}
	if ((written != nullptr) != run.bounded || (written != nullptr && *written != run.written)) {
		return 10;
	}
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (!SameRenderColorInfo(state.color_info[i], run.colors[i])) {
			return 11;
		}
	}
	if (!SameRenderDepthInfo(state.depth_info, run.depth[0])) {
		return 11;
	}
	{
		auto&            cache = m_context.GetTextureCache();
		std::scoped_lock lock {cache.m_lock};
		const auto attachments = std::span<const DrawRunImage>(run.images).subspan(
		    std::min<size_t>(run.attachments_begin, run.images.size()));
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				const auto* image = cache.m_slot_images.try_get(binding.image_id);
				if (image == nullptr) {
					return 12;
				}
				for (const auto& mark: attachments) {
					if (!mark.texture && mark.id == binding.image_id) {
						return 12;
					}
				}
				if (DrawRunOverAttachment(attachments, image->info.data.address, image->info.data.size)) {
					return 12;
				}
			}
		}
	}
	return DrawRunImagesChange(true, true, false, !m_run_recall) != 0 ? 13 : 14;
}

bool RenderExecutor::DrawRunPartialPush(const CommandBuffer&           buffer,
                                        const PipelineCache::Pipeline& pipeline) const {
	return DrawRun::PushPartialEnabled() && !m_run_recall && !m_run_pipeswap &&
	       pipeline.uses_push_descriptors &&
	       m_run.push_valid &&
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
	if (!m_run_active) {
		DrawRunKeep();
	}
	auto& run = m_run;
	// KYTY_BK_RECORD (drawPrep/bookkeep.h): a kernel continuation of the record (same key, no pipeline
	// swap or texture delta) proved activity, command buffer, tick and rendering instance equal to the
	// record's (DrawRunCandidate; the late check keeps the command and images); only the descriptor
	// epoch moves with its push. Verify: those fields are compared with the full update's values.
	if (const auto record_mode = DrawPrep::BkRecordMode();
	    record_mode != DrawPrep::Bookkeep::Mode::Off && m_run_active) {
		auto& bk = DrawPrep::BkCounters();
		if (!m_kernel || m_run_pipeswap || m_run_texdelta) {
			bk.Fall(DrawPrep::Bookkeep::Fallback::RecordNotKernel);
		} else if (record_mode == DrawPrep::Bookkeep::Mode::On) {
			run.key         = m_run_key;
			run.targets_key = m_run_targets_key;
			run.swap_key    = m_run_swap_key;
			run.texfree_key = m_run_texfree_key;
			run.push_epoch  = buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics);
			run.valid       = true;
			bk.Add(bk.record_lean);
			return;
		} else {
			bk.Add(bk.record_checks);
			const char* field = run.key != m_run_key                                          ? "key"
			                    : run.activity != DrawRun::ActivityEpoch()                    ? "activity"
			                    : run.command != buffer.Identity()                            ? "command"
			                    : run.tick != m_context.GetCommandScheduler().CurrentTick()   ? "tick"
			                    : run.rendering_serial != buffer.ActiveRenderingSerial()      ? "rendering_serial"
			                                                                                  : nullptr;
			if (field != nullptr) {
				DrawPrep::Bookkeep::ReportMismatch(bk.record_mismatches, "record", field);
			}
		}
	}
	run.key              = m_run_key;
	run.targets_key      = m_run_targets_key;
	run.swap_key         = m_run_swap_key;
	run.texfree_key      = m_run_texfree_key;
	run.activity         = DrawRun::ActivityEpoch();
	run.command          = buffer.Identity();
	run.tick             = m_context.GetCommandScheduler().CurrentTick();
	run.rendering_serial = buffer.ActiveRenderingSerial();
	// The draw's push (whole or partial) is the last descriptor command at the bind point.
	run.push_epoch = buffer.DescriptorEpoch(vk::PipelineBindPoint::eGraphics);
	if (m_run_active) {
		// A continuation changed none of the recorded images, targets or bindings; a pipeline swap
		// changed the vertex program and the pipeline (layout) the next draw is compared with.
		if (m_run_pipeswap) {
			run.vertex_program = state.vertex_info[0].stage.program;
			run.push_valid     = pipeline.uses_push_descriptors;
			run.push_layout    = pipeline.pipeline_layout;
		}
		if (m_run_texdelta) {
			// KYTY_DRAW_RUN_TEXDELTA: the record's attachments stay; its textures are the draw's.
			run.push_valid  = pipeline.uses_push_descriptors;
			run.push_layout = pipeline.pipeline_layout;
			DrawRunRecordTextures(stages);
		}
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
				m_key_diag.reason = DrawRun::KeyDiag::Inelig::DepthPromotion;
				m_stats_inelig    = static_cast<uint8_t>(DrawRun::KeyDiag::Inelig::DepthPromotion);
				return;
			}
		}
		// KYTY_DRAW_RUN_IMAGE_EPOCH: the epoch before the marks, valid when every marked image also
		// passes the check's flag tests now (what the marks do not record).
		const auto epoch = ImageStateEpoch();
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
		bool clean = true;
		for (const auto& mark: run.images) {
			const auto* image = m_context.GetTextureCache().m_slot_images.try_get(mark.id);
			clean = clean && image != nullptr && !image->binding.needs_rebind &&
			        (image->info.data.Empty() ||
			         (image->registered && !image->IsCpuDirty() && !image->IsBufferModified())) &&
			        (!mark.texture || DrawRunTextureReady(*image));
		}
		run.image_epoch = clean ? epoch : 0;
	}
	// A texture over an attachment's memory (an alias of it) is synchronized from the attachment by
	// each draw's texture resolution (SyncAliasFromOwner), which a continuation skips: no run.
	const auto attachments = std::span<const DrawRunImage>(run.images).subspan(run.attachments_begin);
	for (const auto& mark: run.images) {
		if (mark.texture && DrawRunOverAttachment(attachments, mark.address, mark.size)) {
			DrawRun::GetTotals().alias_excluded.fetch_add(1, std::memory_order_relaxed);
			m_key_diag.reason = DrawRun::KeyDiag::Inelig::Alias;
			m_stats_inelig    = static_cast<uint8_t>(DrawRun::KeyDiag::Inelig::Alias);
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
	run.recall_valid = DrawRun::RecallEnabled();
	if (run.recall_valid) {
		run.recall_textures.clear();
		run.recall_samplers.clear();
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				run.recall_textures.push_back({binding.image_id, binding.image_view, binding.layout,
				                               DrawRunMipViewsHash(binding.mip_views)});
			}
			run.recall_samplers.insert(run.recall_samplers.end(), stage->samplers.begin(),
			                           stage->samplers.end());
			if (stage == stages.front()) {
				run.recall_vertex_textures = static_cast<uint32_t>(run.recall_textures.size());
				run.recall_vertex_samplers = static_cast<uint32_t>(run.recall_samplers.size());
			}
		}
	}
	if (verify) {
		run.pushed_images.assign(m_descriptor_scratch.images.begin(), m_descriptor_scratch.images.end());
		run.textures.clear();
		run.samplers.clear();
		for (const auto* stage: stages) {
			run.textures.insert(run.textures.end(), stage->images.begin(), stage->images.end());
			run.samplers.insert(run.samplers.end(), stage->samplers.begin(), stage->samplers.end());
		}
	}
	run.valid = true;
}

// KYTY_DRAW_RUN_TEXDELTA: a delta's record keeps its attachment marks and takes the draw's texture
// marks; the image epoch is proven again over all marks (as a full record does). The record is not
// recalled later (its recall copies would be the previous textures').
void RenderExecutor::DrawRunRecordTextures(std::span<PreparedBindings* const> stages) {
	auto&                                     run = m_run;
	static thread_local std::vector<DrawRunImage> textures;
	textures.clear();
	std::scoped_lock lock {m_context.GetTextureCache().m_lock};
	const auto       epoch = ImageStateEpoch();
	for (const auto* stage: stages) {
		for (const auto& binding: stage->images) {
			textures.push_back(MakeDrawRunImage(binding.image_id, true));
		}
	}
	run.attachments_begin = DrawRun::TexDelta::ReplaceTextureMarks<DrawRunImage>(run.images, run.attachments_begin, textures);
	bool clean = true;
	for (const auto& mark: run.images) {
		const auto* image = m_context.GetTextureCache().m_slot_images.try_get(mark.id);
		clean = clean && image != nullptr && !image->binding.needs_rebind &&
		        (image->info.data.Empty() || (image->registered && !image->IsCpuDirty() && !image->IsBufferModified())) &&
		        (!mark.texture || DrawRunTextureReady(*image));
	}
	run.image_epoch  = clean ? epoch : 0;
	run.recall_valid = false;
	run.verify_valid = false;
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
		// KYTY_RAW_READ_IMAGE_SYNC_OBTAINS (BufferCache::PrepareReadRange).
		buffer.GetContext().GetBufferCache().PrepareReadRange(source.address, source.size);
		auto [buffer_ptr, offset] = KeepDrawBuffers::Obtain(buffer.GetContext().GetBufferCache(),
		                                                    KeepDrawBuffers::Mode(), KeepDrawBuffers::Get().index,
		                                                    source.address, source.size);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
		SetVulkanObjectNameF(buffer.GetContext().GetGraphics().device, prepared.buffer,
		                     "Kyty.IndexBuffer[guest=0x{:016x} size=0x{:x} type={}]",
		                     source.address, source.size, static_cast<uint32_t>(source.type));
	}
	return prepared;
}

template <typename Sink>
static void CommitVertexBuffers(Sink& vk_buffer, const PreparedVertexBuffers& prepared) {
	DrawEncode::EmitVertexBuffers(vk_buffer, prepared.count, prepared.buffers.data(), prepared.offsets.data(),
	                              prepared.sizes.data());
}

template <typename Sink>
static void CommitIndexBuffer(Sink& vk_buffer, const PreparedIndexBuffer& prepared) {
	DrawEncode::EmitIndexBuffer(vk_buffer, prepared.buffer, prepared.offset, prepared.type);
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

// The draw command of a direct draw (DrawEncode::EmitDirectDraw), by primitive type.
static DrawEncode::DirectDraw MakeDirectDraw(const HW::UserConfig& ucfg, const DrawCallInfo& draw,
                                            const DrawEmitInfo& emit) {
	DrawEncode::DirectDraw direct;
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
		case Prospero::PrimitiveType::kPatch: break;
		case Prospero::PrimitiveType::kQuadListLegacy: direct.quad_list = true; break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
	direct.indexed        = draw.IsIndexed();
	direct.count          = draw.index_count;
	direct.instance_count = draw.instance_count;
	direct.vertex_offset  = emit.vertex_offset;
	direct.first_vertex   = emit.first_vertex;
	direct.first_instance = emit.first_instance;
	return direct;
}

template <typename Sink>
static void EmitDrawPrimitives(const HW::UserConfig& ucfg, Sink& vk_buffer, const DrawCallInfo& draw,
                               const DrawEmitInfo& emit) {
	DrawEncode::EmitDirectDraw(vk_buffer, MakeDirectDraw(ucfg, draw, emit));
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

// DB_RENDER_CONTROL.COPY_DEPTH_TO_COLOR (upstream 1029511bb): a rect-list draw that copies the
// depth target into the R32F colour target (VK_KHR_maintenance8 depth/colour image copy).
bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.first_vertex != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		m_context.GetCommandScheduler().EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
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
	static constexpr auto enabled_init = [] {
		const auto* value = std::getenv("KYTY_DRAW_BINDING_REPEAT_STATS");
		return Profiler::AggregateEnabled() && value != nullptr && std::strcmp(value, "0") != 0;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
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
	// Structure: programs, texture views and layouts, samplers, attachments (each part on its own).
	for (const auto* stage: stages) {
		words.Add(reinterpret_cast<uintptr_t>(stage->runtime->program));
	}
	shape.programs = words.Finish();
	for (const auto* stage: stages) {
		words.Add(stage->images.size());
		for (const auto& image: stage->images) {
			words.AddHandle(image.image_view);
			words.Add(static_cast<uint64_t>(image.layout));
			for (const auto view: image.mip_views) {
				words.AddHandle(view);
			}
		}
	}
	shape.textures = words.Finish();
	for (const auto* stage: stages) {
		words.Add(stage->samplers.size());
		for (const auto sampler: stage->samplers) {
			words.AddHandle(sampler);
		}
	}
	shape.samplers = words.Finish();
	for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
		words.AddHandle(rendering.color_attachments[i].image_view);
		words.Add(static_cast<uint64_t>(rendering.color_attachments[i].image_layout));
	}
	words.AddHandle(rendering.depth_stencil_attachment.image_view);
	words.Add(static_cast<uint64_t>(rendering.depth_stencil_attachment.image_layout));
	shape.attachments = words.Finish();
	words.Add(shape.programs);
	words.Add(shape.textures);
	words.Add(shape.samplers);
	words.Add(shape.attachments);
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
	// KYTY_GPU_INSTANCE_COUNT: instance_count is a placeholder; the GPU reads the count there.
	words.Add(static_cast<uint32_t>(emit.gpu_instance_address));
	words.Add(static_cast<uint32_t>(emit.gpu_instance_address >> 32u));
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

// KYTY_CUBE_EVERY=<n> (live, 1 to 8, default 1 = every frame; usa-ratt/KytyPS5-GT7 fed07dcc3,
// round 85): the draws into a reflection cube run every nth frame only; on the other frames the
// cube keeps the last pictures drawn into it. GT7 draws its surroundings into a 512x512 cube of
// six layers every frame for the cars' reflections: 550 to 750 draws a frame, 30 to 40 % of the
// fork's GPU-thread draw time in a race. A cube is a color target of six square layers of at most
// 2048 pixels; its address is learned at its first draw while the switch is above 1 (a
// "Reflection cube:" line), and a direct draw (DrawIndex, DrawAuto) is left out, before anything
// is prepared for it, while that address is its first color target and the draw writes it.
// Command-processor thread only, but the frame count (NoteCubeFrame).
static Live::Switch g_cube_every("KYTY_CUBE_EVERY", [](const char* value) -> int64_t {
	const auto n = value != nullptr ? std::strtol(value, nullptr, 10) : 1l;
	return std::clamp<int64_t>(n, 1, 8);
});
static std::array<uint64_t, 4> g_cube_targets {};
static uint32_t                g_cube_target_count = 0;
static uint64_t                g_cube_draws_left   = 0;
static std::atomic<uint64_t>   g_cube_frame {0};

void NoteCubeFrame() noexcept {
	g_cube_frame.fetch_add(1, std::memory_order_relaxed);
}

static void LearnCubeTarget(const CommandBuffer& buffer, const DrawRenderState& state) {
	if (g_cube_every.Get() <= 1 || state.color_count == 0 ||
	    g_cube_target_count >= g_cube_targets.size()) {
		return;
	}
	const auto& color = state.color_info[0];
	const auto& info  = color.desc.info;
	if (color.target_slot != 0 || info.resources.layers != 6 ||
	    info.extent.width != info.extent.height || info.extent.width > 2048u) {
		return;
	}
	const auto address = buffer.GetRegisters().GetRenderTarget(0).base.addr;
	const auto known   = std::span(g_cube_targets.data(), g_cube_target_count);
	if (address == 0 || std::find(known.begin(), known.end(), address) != known.end()) {
		return;
	}
	g_cube_targets[g_cube_target_count++] = address;
	LOGF("Reflection cube: color target 0x%" PRIx64 ", %ux%u, 6 layers: its draws run every "
	     "%" PRId64 " frames (KYTY_CUBE_EVERY)\n",
	     address, info.extent.width, info.extent.height, g_cube_every.Get());
}

static bool SkipCubeDraw(const CommandBuffer& buffer) {
	if (g_cube_target_count == 0) {
		return false;
	}
	const auto every = static_cast<uint64_t>(g_cube_every.Get());
	if (every <= 1 || g_cube_frame.load(std::memory_order_relaxed) % every == 0) {
		return false;
	}
	const auto& regs    = buffer.GetRegisters();
	const auto  address = regs.GetRenderTarget(0).base.addr;
	const auto  known   = std::span(g_cube_targets.data(), g_cube_target_count);
	if (address == 0 || std::find(known.begin(), known.end(), address) == known.end() ||
	    (regs.GetRenderTargetMask() & 0xfu) == 0) {
		return false;
	}
	const auto left = ++g_cube_draws_left;
	if ((left & (left - 1u)) == 0 && left >= 1024u) {
		LOGF("Reflection cube: %" PRIu64 " draws left out so far (KYTY_CUBE_EVERY=%" PRIu64 ")\n",
		     left, every);
	}
	return true;
}

// KYTY_DRAW_PEEK=<vs hash>[:<max draws>] (hexadecimal hash, diagnostic, default off; default 200000
// draws): after every draw with this vertex program, copies the first 16 bytes of VS buffer 0, the
// draw's first indirect argument record and its count (indirect draws) into a host-visible ring and
// logs them once the tick completed ("DrawPeek", with the frame, the draw's ordinal in the frame and
// steady-clock ns at recording and at completion, to match screenshots). Ends the rendering
// instance after each such draw. videoOut.cpp logs a "DrawPeekFlip" marker per guest flip with it.
static const std::array<uint64_t, 2>& DrawPeekRequest() {
	static constexpr auto request_init = [] {
		std::array<uint64_t, 2> parsed {0, 200000};
		if (const auto* value = std::getenv("KYTY_DRAW_PEEK"); value != nullptr) {
			char* end = nullptr;
			parsed[0] = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') parsed[1] = std::strtoull(end + 1, &end, 10);
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(request_init())>> request_once;
	const std::array<uint64_t, 2>& request = request_once.Get(request_init);
	return request;
}

static uint64_t DrawPeekNowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Records the copies after the draw (rendering already ended by the caller).
static void DrawPeek(RenderContext& context, CommandBuffer& buffer, const DrawCallInfo& draw,
                     const DrawRenderState& state, const DrawIndirectSource* indirect) {
	constexpr uint32_t SlotBytes = 64;
	constexpr uint32_t Slots     = 4096;
	struct Range {
		uint64_t address = 0;
		uint32_t bytes   = 0;
		uint32_t offset  = 0;
	};
	static std::mutex              mutex;
	static std::unique_ptr<Buffer> download;
	static uint32_t                next_slot  = 0;
	static uint64_t                peeked     = 0;
	static int64_t                 last_frame = -1;
	static uint32_t                ordinal    = 0;
	std::scoped_lock               lock(mutex);
	if (peeked >= DrawPeekRequest()[1]) return;
	peeked++;
	const auto frame = static_cast<int64_t>(context.GetGpu().GetFrameNum());
	if (frame != last_frame) {
		last_frame = frame;
		ordinal    = 0;
	}
	const auto index = ordinal++;
	if (download == nullptr) {
		download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
		                                    MemoryUsage::Download, 0, AllFlags, uint64_t {SlotBytes} * Slots);
	}
	const auto slot = next_slot++ % Slots;
	std::array<Range, 3> ranges {};
	const auto* program   = state.vertex_info[0].stage.program;
	const auto* resources = state.vertex_info[0].stage.resources;
	if (program != nullptr && resources != nullptr && !program->info.buffers.empty() && !resources->buffers.empty()) {
		ranges[0] = {DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[0]).Base48(), 16, 0};
	}
	if (indirect != nullptr) {
		ranges[1] = {indirect->args_addr, indirect->RecordSize(), 16};
		if (indirect->count_addr != 0) ranges[2] = {indirect->count_addr, 4, 40};
	}
	auto& cache = context.GetBufferCache();
	for (auto& range: ranges) {
		if (range.address == 0 || range.bytes == 0) continue;
		const auto [source, offset] = cache.ObtainBuffer(range.address, range.bytes, false);
		if (source == nullptr) {
			range.bytes = 0;
			continue;
		}
		download->CopyFrom(buffer, *source, offset, uint64_t {slot} * SlotBytes + range.offset, range.bytes,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
	}
	// KYTY_DRAW_PEEK_WINDOW=<hex address>+<hex bytes>[,...] (at most 2 windows of at most 64 KiB, with
	// KYTY_DRAW_PEEK): at the frame's first peeked draw, the GPU's bytes of each window are appended to
	// _drawpeek-window.bin in the working directory ({u32 'DPKW', i32 frame, u64 address, u32 bytes,
	// data} per window and frame), e.g. the instance records the draws read.
	struct Window {
		uint64_t address = 0;
		uint32_t bytes   = 0;
	};
	static const std::vector<Window> windows = [] {
		std::vector<Window> parsed;
		if (const auto* value = std::getenv("KYTY_DRAW_PEEK_WINDOW"); value != nullptr) {
			std::string_view text(value);
			while (!text.empty() && parsed.size() < 2) {
				const auto comma = text.find(',');
				const auto item  = std::string(text.substr(0, comma));
				const auto plus  = item.find('+');
				if (plus != std::string::npos) {
					Window window {std::strtoull(item.substr(0, plus).c_str(), nullptr, 16),
					               static_cast<uint32_t>(std::min<uint64_t>(
					                   std::strtoull(item.substr(plus + 1).c_str(), nullptr, 16), 0x10000u)) & ~3u};
					if (window.address != 0 && window.bytes != 0) parsed.push_back(window);
				}
				if (comma == std::string_view::npos) break;
				text.remove_prefix(comma + 1);
			}
		}
		return parsed;
	}();
	if (index == 0 && !windows.empty()) {
		constexpr uint32_t WindowSlots     = 32;
		constexpr uint32_t WindowSlotBytes = 2 * 0x10000;
		static std::unique_ptr<Buffer> window_download;
		static uint32_t                next_window_slot = 0;
		if (window_download == nullptr) {
			window_download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
			                                           MemoryUsage::Download, 0, AllFlags,
			                                           uint64_t {WindowSlotBytes} * WindowSlots);
		}
		const auto wslot  = next_window_slot++ % WindowSlots;
		auto       copied = windows;
		for (uint32_t i = 0; i < copied.size(); i++) {
			const auto [source, offset] = cache.ObtainBuffer(copied[i].address, copied[i].bytes, false);
			if (source == nullptr) {
				copied[i].bytes = 0;
				continue;
			}
			window_download->CopyFrom(buffer, *source, offset, uint64_t {wslot} * WindowSlotBytes + i * 0x10000u,
			                          copied[i].bytes, vk::AccessFlagBits::eMemoryWrite,
			                          vk::AccessFlagBits::eHostRead,
			                          vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
			                          vk::AccessFlagBits::eHostRead);
		}
		context.GetCommandScheduler().DeferOperation([wslot, copied, frame] {
			static std::FILE* file = std::fopen("_drawpeek-window.bin", "wb");
			if (file == nullptr) return;
			window_download->Invalidate(uint64_t {wslot} * WindowSlotBytes, WindowSlotBytes);
			const auto* bytes = window_download->Mapped().data() + uint64_t {wslot} * WindowSlotBytes;
			for (uint32_t i = 0; i < copied.size(); i++) {
				if (copied[i].bytes == 0) continue;
				const uint32_t magic = 0x574b5044u; // "DPKW"
				const auto     f32   = static_cast<int32_t>(frame);
				std::fwrite(&magic, 4, 1, file);
				std::fwrite(&f32, 4, 1, file);
				std::fwrite(&copied[i].address, 8, 1, file);
				std::fwrite(&copied[i].bytes, 4, 1, file);
				std::fwrite(bytes + i * 0x10000u, 1, copied[i].bytes, file);
			}
			std::fflush(file);
		});
	}
	const auto tick    = context.GetCommandScheduler().CurrentTick();
	const auto queue   = static_cast<int32_t>(HangTrace::CpContextQueue());
	const auto seq     = HangTrace::CpContextSequence();
	const auto t_rec   = DrawPeekNowNs();
	const auto n       = draw.index_count;
	const auto inst    = draw.instance_count;
	const auto args    = indirect != nullptr ? indirect->args_addr : 0;
	const auto count   = indirect != nullptr ? indirect->count_addr : 0;
	const auto max_cnt = indirect != nullptr ? indirect->max_count : 0;
	// The VS buffers' bases, strides and record counts (where each draw's instance data lives).
	std::string vs_buffers;
	if (program != nullptr && resources != nullptr) {
		for (uint32_t i = 0; i < program->info.buffers.size() && i < resources->buffers.size() && i < 8; i++) {
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[i]);
			vs_buffers += fmt::format(" vsb{}=0x{:x}/s{}/n{}", i, r.Base48(), r.Stride(), r.NumRecords());
		}
	}
	context.GetCommandScheduler().DeferOperation(
	    [slot, ranges, frame, index, tick, queue, seq, t_rec, n, inst, args, count, max_cnt,
	     vs_buffers = std::move(vs_buffers)] {
		    download->Invalidate(uint64_t {slot} * SlotBytes, SlotBytes);
		    const auto* bytes = download->Mapped().data() + uint64_t {slot} * SlotBytes;
		    std::string text  = fmt::format(
                "DrawPeek: frame={} #{} q={} seq={} tick={} t_rec={} t_done={} n={} inst={} b0=0x{:x} args=0x{:x} "
                "count=0x{:x} max={}",
                frame, index, queue, seq, tick, t_rec, DrawPeekNowNs(), n, inst, ranges[0].address, args, count,
                max_cnt);
		    static constexpr const char* Names[] = {"blk", "arg", "cnt"};
		    for (uint32_t i = 0; i < ranges.size(); i++) {
			    if (ranges[i].bytes == 0) continue;
			    text += fmt::format(" | {}:", Names[i]);
			    for (uint32_t word = 0; word < ranges[i].bytes / 4; word++) {
				    uint32_t value = 0;
				    std::memcpy(&value, bytes + ranges[i].offset + word * 4u, sizeof(value));
				    text += fmt::format(" {:08x}", value);
			    }
		    }
		    text += " |" + vs_buffers;
		    LOGF("%s\n", text.c_str());
	    });
}

// KYTY_DRAW_STATS=<file> (diagnostic, default off): every draw with a depth target it writes is
// wrapped in a pipeline-statistics query (input-assembly primitives, clipping primitives, fragment
// shader invocations). The results are read when the query slot's frame comes round again (a ring
// of 8 frames) and appended to <file> as binary records: kind 1 = draw (frame, ordinal in the frame,
// flags, vs/ps hashes, counts, depth and colour-0 addresses, the three statistics), kind 2 = the
// first recorded draw of a frame (frame, steady_clock ns: QPC-based, comparable to a window
// capture's SystemRelativeTime). Finds the draw whose geometry blows up in one frame (a draw that
// suddenly covers the screen shows as a jump of its fragment count). Needs pipelineStatisticsQuery
// and hostQueryReset (enabled at device creation when the variable is set). Local files only.
namespace DrawStats {
struct Pending {
	uint32_t frame          = 0;
	uint32_t ordinal        = 0;
	uint32_t flags          = 0;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	uint64_t vs             = 0;
	uint64_t ps             = 0;
	uint64_t depth          = 0;
	uint64_t color0         = 0;
};
#pragma pack(push, 1)
struct Record {
	uint32_t kind;
	uint32_t frame;
	uint32_t ordinal;
	uint32_t flags;
	uint64_t vs;
	uint64_t ps;
	uint32_t index_count;
	uint32_t instance_count;
	uint64_t depth;
	uint64_t color0;
	uint64_t stats[3];
};
#pragma pack(pop)
static_assert(sizeof(Record) == 80);
constexpr uint32_t Slots    = 8;
constexpr uint32_t Capacity = 16384;
struct Slot {
	vk::QueryPool        pool = nullptr;
	std::vector<Pending> pending;
};
struct State {
	std::mutex  mutex;
	std::FILE*  file = nullptr;
	vk::Device  device;
	Slot        slots[Slots];
	uint32_t    current = 0;
	int64_t     frame   = -1;
	uint64_t    dropped = 0;
	bool        failed  = false;
};
static const char* Path() {
	static constexpr auto path_init = [] {
		const auto* value = std::getenv("KYTY_DRAW_STATS");
		return value != nullptr && value[0] != '\0' ? value : static_cast<const char*>(nullptr);
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(path_init())>> path_once;
	const char* const& path = path_once.Get(path_init);
	return path;
}
static State& Get() {
	static State s;
	return s;
}
static void Flush(State& s, Slot& slot) {
	const auto used = static_cast<uint32_t>(slot.pending.size());
	if (used == 0) return;
	std::vector<uint64_t> results(static_cast<size_t>(used) * 3u);
	const auto result = s.device.getQueryPoolResults(slot.pool, 0, used, results.size() * sizeof(uint64_t), results.data(),
	                                                 3u * sizeof(uint64_t),
	                                                 vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
	std::vector<Record> records(used);
	for (uint32_t i = 0; i < used; i++) {
		const auto& p = slot.pending[i];
		auto&       r = records[i];
		r.kind           = 1;
		r.frame          = p.frame;
		r.ordinal        = p.ordinal;
		r.flags          = p.flags | (result == vk::Result::eSuccess ? 0u : 0x80000000u);
		r.vs             = p.vs;
		r.ps             = p.ps;
		r.index_count    = p.index_count;
		r.instance_count = p.instance_count;
		r.depth          = p.depth;
		r.color0         = p.color0;
		for (uint32_t k = 0; k < 3; k++) r.stats[k] = results[i * 3u + k];
	}
	std::fwrite(records.data(), sizeof(Record), records.size(), s.file);
	std::fflush(s.file);
	s.device.resetQueryPool(slot.pool, 0, Capacity);
	slot.pending.clear();
}
// The query index for this draw (UINT32_MAX: not recorded).
static uint32_t Begin(RenderContext& context, const CommandSink& sink, const Pending& draw) {
	auto& s = Get();
	std::scoped_lock lock(s.mutex);
	if (s.failed) return UINT32_MAX;
	if (s.file == nullptr) {
		s.device = context.GetGraphics().device;
		s.file   = std::fopen(Path(), "wb");
		vk::QueryPoolCreateInfo create {};
		create.queryType          = vk::QueryType::ePipelineStatistics;
		create.queryCount         = Capacity;
		create.pipelineStatistics = vk::QueryPipelineStatisticFlagBits::eInputAssemblyPrimitives |
		                            vk::QueryPipelineStatisticFlagBits::eClippingPrimitives |
		                            vk::QueryPipelineStatisticFlagBits::eFragmentShaderInvocations;
		for (auto& slot: s.slots) {
			if (s.file == nullptr || s.device.createQueryPool(&create, nullptr, &slot.pool) != vk::Result::eSuccess) {
				s.failed = true;
				std::printf("DrawStats: setup failed\n");
				return UINT32_MAX;
			}
			s.device.resetQueryPool(slot.pool, 0, Capacity);
			slot.pending.reserve(Capacity);
		}
		std::printf("DrawStats: recording to %s\n", Path());
	}
	if (static_cast<int64_t>(draw.frame) != s.frame) {
		s.frame   = draw.frame;
		s.current = (s.current + 1u) % Slots;
		Flush(s, s.slots[s.current]);
		Record r {};
		r.kind     = 2;
		r.frame    = draw.frame;
		r.stats[0] = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
		        .count());
		r.stats[1] = s.dropped;
		std::fwrite(&r, sizeof(r), 1, s.file);
	}
	auto& slot = s.slots[s.current];
	if (slot.pending.size() >= Capacity) {
		s.dropped++;
		return UINT32_MAX;
	}
	const auto query = static_cast<uint32_t>(slot.pending.size());
	slot.pending.push_back(draw);
	slot.pending.back().ordinal = query;
	sink.beginQuery(slot.pool, query, {});
	return query;
}
static void End(const CommandSink& sink, uint32_t query) {
	if (query == UINT32_MAX) return;
	auto& s = Get();
	sink.endQuery(s.slots[s.current].pool, query);
}
} // namespace DrawStats

// KYTY_DRAW_VERIFY_VSB=<vs hash>:<buffer index>[,<buffer index>] (hexadecimal hash, diagnostic,
// default off): after every draw with this vertex program, the GPU's bytes of those VS buffers
// (stride x records, at most 256 KiB each) are copied to a host-visible ring and, once the tick
// completed, compared with the guest's bytes of the same range read when the draw was recorded
// (the guest backing, as the CPU wrote it). "VsbVerify" logs per draw the records that differ, so
// an instance buffer the GPU read stale (or that the guest rewrote while the draw was pending)
// shows. Ends the rendering instance after each such draw.
struct VerifyVsbRequest {
	uint64_t              hash = 0;
	std::vector<uint32_t> buffers;
};
static const VerifyVsbRequest& VerifyVsb() {
	static constexpr auto request_init = [] {
		VerifyVsbRequest parsed;
		if (const auto* value = std::getenv("KYTY_DRAW_VERIFY_VSB"); value != nullptr) {
			char* end   = nullptr;
			parsed.hash = std::strtoull(value, &end, 16);
			while (end != nullptr && (*end == ':' || *end == ',')) {
				parsed.buffers.push_back(static_cast<uint32_t>(std::strtoul(end + 1, &end, 10)));
			}
			if (parsed.buffers.empty()) parsed.buffers.push_back(0);
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(request_init())>> request_once;
	const VerifyVsbRequest& request = request_once.Get(request_init);
	return request;
}

static void VerifyVsbDraw(RenderContext& context, CommandBuffer& buffer, const DrawCallInfo& draw,
                          const DrawRenderState& state) {
	constexpr uint64_t MaxBytes = 256 * 1024;
	constexpr uint32_t Slots    = 64;
	static std::mutex              mutex;
	static std::unique_ptr<Buffer> download;
	static uint32_t                next_slot  = 0;
	static int64_t                 last_frame = -1;
	static uint32_t                ordinal    = 0;
	const auto* program   = state.vertex_info[0].stage.program;
	const auto* resources = state.vertex_info[0].stage.resources;
	if (program == nullptr || resources == nullptr) return;
	std::scoped_lock lock(mutex);
	const auto       frame = static_cast<int64_t>(context.GetGpu().GetFrameNum());
	if (frame != last_frame) {
		last_frame = frame;
		ordinal    = 0;
	}
	const auto draw_index = ordinal++;
	if (download == nullptr) {
		download = std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(),
		                                    MemoryUsage::Download, 0, AllFlags, MaxBytes * Slots);
	}
	auto& cache = context.GetBufferCache();
	for (const auto index: VerifyVsb().buffers) {
		if (index >= program->info.buffers.size() || index >= resources->buffers.size()) continue;
		const auto     r      = DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[index]);
		const uint64_t base   = r.Base48();
		const uint32_t stride = r.Stride();
		const uint64_t bytes  = std::min<uint64_t>(stride != 0 ? uint64_t {stride} * r.NumRecords() : r.NumRecords(),
		                                          MaxBytes) & ~uint64_t {3};
		if (base == 0 || bytes == 0) continue;
		auto       guest = std::make_shared<std::vector<uint8_t>>(bytes);
		const bool clean = LibKernel::Memory::TryReadGpuCleanBacking(base, guest->data(), bytes);
		if (!clean) {
			const auto* alias = LibKernel::Memory::GuestBackingAlias(base, bytes);
			if (alias == nullptr) continue;
			std::memcpy(guest->data(), alias, bytes);
		}
		const auto [source, offset] = cache.ObtainBuffer(base, bytes, false);
		if (source == nullptr) continue;
		const auto slot = next_slot++ % Slots;
		download->CopyFrom(buffer, *source, offset, uint64_t {slot} * MaxBytes, bytes,
		                   vk::AccessFlagBits::eMemoryWrite, vk::AccessFlagBits::eHostRead,
		                   vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
		                   vk::AccessFlagBits::eHostRead);
		const auto inst = draw.instance_count;
		context.GetCommandScheduler().DeferOperation([slot, guest, base, bytes, stride, clean, frame, draw_index,
		                                              index, inst] {
			download->Invalidate(uint64_t {slot} * MaxBytes, bytes);
			const auto*    gpu    = download->Mapped().data() + uint64_t {slot} * MaxBytes;
			// KYTY_DRAW_VERIFY_VSB_DIR=<folder>: the GPU bytes are also written to a file per draw and buffer.
			if (const auto* dir = std::getenv("KYTY_DRAW_VERIFY_VSB_DIR"); dir != nullptr && dir[0] != '\0') {
				const auto path = fmt::format("{}/f{}_d{}_vsb{}_{:x}_s{}.bin", dir, frame, draw_index, index, base, stride);
				if (auto* f = std::fopen(path.c_str(), "wb"); f != nullptr) {
					std::fwrite(gpu, 1, bytes, f);
					std::fclose(f);
				}
			}
			const uint32_t record = stride != 0 ? stride : 16u;
			uint32_t       diff = 0, first = UINT32_MAX;
			const uint64_t records = bytes / record;
			for (uint64_t k = 0; k < records; k++) {
				if (std::memcmp(gpu + k * record, guest->data() + k * record, record) != 0) {
					if (first == UINT32_MAX) first = static_cast<uint32_t>(k);
					diff++;
				}
			}
			// Guest bytes now (at completion): whether the guest rewrote the range meanwhile.
			uint32_t    now_diff = 0;
			const auto* alias    = LibKernel::Memory::GuestBackingAlias(base, bytes);
			if (alias != nullptr) {
				for (uint64_t k = 0; k < records; k++) {
					if (std::memcmp(static_cast<const uint8_t*>(alias) + k * record, guest->data() + k * record,
					                record) != 0) {
						now_diff++;
					}
				}
			}
			// The GPU bytes as floats: the largest magnitude and the NaN/Inf count (garbage instance data).
			float    max_abs    = 0.0f;
			uint32_t non_finite = 0;
			for (uint64_t w = 0; w < bytes / 4u; w++) {
				float value = 0.0f;
				std::memcpy(&value, gpu + w * 4u, 4);
				if (!std::isfinite(value)) {
					non_finite++;
				} else {
					max_abs = std::max(max_abs, std::fabs(value));
				}
			}
			std::string text = fmt::format(
			    "VsbVerify: frame={} #{} vsb{} base=0x{:x} stride={} records={} inst={} clean={} diff={} "
			    "guest_changed_by_completion={} gpu_max_abs={:g} gpu_non_finite={}",
			    frame, draw_index, index, base, stride, records, inst, clean ? 1 : 0, diff, now_diff, max_abs,
			    non_finite);
			if (diff != 0) {
				text += fmt::format(" first={} gpu:", first);
				for (uint32_t w = 0; w < std::min<uint32_t>(record, 32u) / 4u; w++) {
					uint32_t value = 0;
					std::memcpy(&value, gpu + uint64_t {first} * record + w * 4u, 4);
					text += fmt::format(" {:08x}", value);
				}
				text += " guest:";
				for (uint32_t w = 0; w < std::min<uint32_t>(record, 32u) / 4u; w++) {
					uint32_t value = 0;
					std::memcpy(&value, guest->data() + uint64_t {first} * record + w * 4u, 4);
					text += fmt::format(" {:08x}", value);
				}
			}
			LOGF("%s\n", text.c_str());
		});
	}
}

// KYTY_CONT_KERNEL (live, default 0): the continuation kernel. A DrawIndex whose run structure key is
// the current record's (DrawRun::StructureKey: every context register, the user-config registers but
// the index offset and object id, both programs and their T#/S# words), in the record's command
// buffer, tick, rendering instance and activity epoch, with the record's index element size, takes a
// short path (RenderExecutor::ContinuationKernel): the decisions that are functions of that key are
// the record's instead of being computed again (target operations, vertex-shader and hardware checks,
// topology, primitive restart, the pixel shader's activity - which the preparation's Validate also
// proves -, the pipeline while no pipeline was replaced and the plain-pixel choice is the record's,
// barrier safety), the copy-to-colour / EQAA / prefetch steps of a first draw are left out, and the run
// record keeps only what a continuation changes. What depends on more than the key is still done every
// draw: the preparation's certificate, the run candidate check, the binding marks, the buffer work, the
// late image / command check (its fallback leaves the kernel), the rebind after a join, the dynamic-state
// shadow check and the packet encode. 2 = verify: the normal path runs, and every key-derived decision a
// kernel draw would have reused is compared with the computed one ("ContKernel verify" lines).
static Live::Switch g_cont_kernel("KYTY_CONT_KERNEL", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtoll(value, nullptr, 10) : 0; // 0, 1 or 2
});
// KYTY_CONT_KERNEL_INPLACE (live, default 0; 2 = verify): ApplyPreparedDraw (above) of a kernel
// continuation keeps the programs and pixel input the draw state holds.
static Live::Switch g_cont_kernel_inplace("KYTY_CONT_KERNEL_INPLACE", [](const char* value) -> int64_t {
	return value != nullptr ? std::strtoll(value, nullptr, 10) : 0;
});

namespace ContKernel {
enum Item : uint32_t { TargetOps, VsCheck, Topology, Restart, Pipeline, BarrierSafe, Stage, Eligible, ItemCount };
constexpr const char* kItemNames[ItemCount] = {"targetops", "vscheck", "topology", "restart",
                                               "pipeline",  "barrier", "stage",    "eligible"};
struct Counters {
	std::atomic<uint64_t>                          draws {0};
	std::atomic<uint64_t>                          checks {0};
	std::array<std::atomic<uint64_t>, ItemCount>   mismatches {};
	std::atomic<uint64_t>                          taken {0};
	std::atomic<uint64_t>                          declined {0};
	std::atomic<uint64_t>                          left {0}; // left the kernel after the buffer work
};
Counters g_counters;

void Print(const char* why) {
	auto& c = g_counters;
	std::string line = fmt::format("ContKernel {}: taken {} declined {} left {} | verify draws {} checks {} mismatches",
	                               why, c.taken.load(), c.declined.load(), c.left.load(), c.draws.load(),
	                               c.checks.load());
	uint64_t total = 0;
	for (uint32_t i = 0; i < ItemCount; i++) {
		total += c.mismatches[i].load();
		line += fmt::format(" {}={}", kItemNames[i], c.mismatches[i].load());
	}
	line += fmt::format(" total={}", total);
	std::printf("%s\n", line.c_str());
	std::fflush(stdout);
}

void Check(Item item, bool same) {
	auto& c = g_counters;
	const auto n = c.checks.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!same) {
		const auto m = c.mismatches[item].fetch_add(1, std::memory_order_relaxed);
		if (m < 8) {
			std::printf("ContKernel verify: %s mismatch\n", kItemNames[item]);
		}
	}
	if ((n & 0x7fffffu) == 0) {
		Print("verify");
	}
}

// The candidate's precondition (PrepareDrawRenderState): no stage with bindless heaps.
bool Bindless(const DrawRenderState& state) {
	const auto bindless_stage = [](const ShaderStageRuntime& stage) {
		return stage.resources != nullptr && !stage.resources->bindless_heaps.empty();
	};
	return (state.ps_active && bindless_stage(state.ps_input_info.stage)) ||
	       std::ranges::any_of(state.vertex_info, [&](const auto& info) { return bindless_stage(info.stage); });
}

// KYTY_BK_LEAN: the commit thread is the counter's only writer; a plain add (no locked RMW per draw).
void NoteTaken(bool plain) {
	auto&      taken = g_counters.taken;
	const auto n     = plain ? taken.load(std::memory_order_relaxed) + 1 : taken.fetch_add(1, std::memory_order_relaxed) + 1;
	if (plain) {
		taken.store(n, std::memory_order_relaxed);
	}
	if ((n & 0x1fffffu) == 0) {
		Print("kernel");
	}
}
} // namespace ContKernel

// The register test of DepthStencilCopy (depthRenderTarget.cpp): whether it would copy.
static bool DepthStencilCopyApplies(const HW::Context& hw) {
	if (hw.GetColorControl().mode != 0) {
		return false;
	}
	const auto& z        = hw.GetDepthRenderTarget();
	const auto& override = hw.GetDepthRenderOverride();
	const bool depth_copy = override.force_z_dirty && override.force_z_valid &&
	                        z.z_info.format != Prospero::DepthFormat::kInvalid && z.z_read_base_addr != 0 &&
	                        z.z_write_base_addr != 0 && z.z_read_base_addr != z.z_write_base_addr;
	const bool stencil_copy = override.force_stencil_dirty && override.force_stencil_valid &&
	                          z.stencil_info.format != Prospero::StencilFormat::kInvalid &&
	                          z.stencil_read_base_addr != 0 && z.stencil_write_base_addr != 0 &&
	                          z.stencil_read_base_addr != z.stencil_write_base_addr;
	return depth_copy || stencil_copy;
}

static bool ListResetEnabled() {
	static constexpr auto enabled_init = [] {
		const char* value = std::getenv("KYTY_LIST_RESET");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(enabled_init())>> enabled_once;
	const bool& enabled = enabled_once.Get(enabled_init);
	return enabled;
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	KYTY_GPU_OP_SITE("draw.execute");
	KYTY_PROFILER_DETAIL_FUNCTION();
	m_bda_mesh_index = {}; // KYTY_BDA_REACH: set below for this draw only
	m_bda_mesh_known = false;
	auto& ucfg = buffer.GetUserConfig();
	LearnCubeTarget(buffer, state);
	if (m_context.GetOcclusionCounter().DebugLogActive()) {
		OcclusionCounter::DebugLog("e n=%u inst=%u vs=0x%" PRIx64 "\n", draw.index_count, draw.instance_count,
		                           buffer.GetShaders().GetVs().es_regs.data_addr);
	}
	// KYTY_DRAW_SHADER_MAP=1 (diagnostic): logs once per guest VS/PS address pair the program hashes
	// (device-loss checkpoints name the guest addresses only).
	static constexpr auto shader_map_init = [] {
		const auto* value = std::getenv("KYTY_DRAW_SHADER_MAP");
		return value != nullptr && value[0] != '\0' && value[0] != '0';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(shader_map_init())>> shader_map_once;
	const bool& shader_map = shader_map_once.Get(shader_map_init);
	if (shader_map) {
		static std::mutex                                 map_mutex;
		static std::set<std::pair<uint64_t, uint64_t>>    seen;
		const auto vs_addr = buffer.GetShaders().GetVs().es_regs.data_addr;
		const auto ps_addr = state.ps_active ? buffer.GetShaders().GetPs().ps_regs.data_addr : 0;
		std::scoped_lock map_lock(map_mutex);
		if (seen.emplace(vs_addr, ps_addr).second) {
			LOGF("DrawShaderMap: VS=0x%016" PRIx64 " hash=0x%016" PRIx64 " PS=0x%016" PRIx64
			     " hash=0x%016" PRIx64 "\n",
			     vs_addr, state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
			     ps_addr, state.ps_active && state.ps_input_info.stage.program ? state.ps_input_info.stage.program->shader_hash : 0);
		}
	}
	// KYTY_DRAW_TRACE_TARGET=<hex address>[:<count>] (diagnostic): logs the executed draws whose colour
	// target 0 starts at that guest address (default 200): frame, programs, counts, depth state.
	static constexpr auto trace_target_init = [] {
		std::pair<uint64_t, uint32_t> parsed {0, 200};
		if (const auto* value = std::getenv("KYTY_DRAW_TRACE_TARGET"); value != nullptr) {
			char* end    = nullptr;
			parsed.first = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') parsed.second = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(trace_target_init())>> trace_target_once;
	const auto& trace_target = trace_target_once.Get(trace_target_init);
	if (trace_target.first != 0 && state.color_count > 0 &&
	    state.color_info[0].desc.info.data.address == trace_target.first) {
		static std::atomic<uint32_t> traced {0};
		if (traced.fetch_add(1, std::memory_order_relaxed) < trace_target.second) {
			const auto& d = state.depth_info;
			LOGF("DrawTraceTarget: frame %d VS 0x%016" PRIx64 " PS 0x%016" PRIx64
			     " n=%u inst=%u colors=%u depth 0x%" PRIx64 " test %d write %d clear %d (%g) samples %u\n",
			     static_cast<int>(m_context.GetGpu().GetFrameNum()),
			     state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
			     state.ps_active && state.ps_input_info.stage.program ? state.ps_input_info.stage.program->shader_hash : 0,
			     draw.index_count, draw.instance_count, state.color_count, d.desc.info.data.address,
			     d.depth_test_enable ? 1 : 0, d.depth_write_enable ? 1 : 0, d.depth_clear_enable ? 1 : 0,
			     static_cast<double>(d.depth_clear_value), static_cast<unsigned>(state.color_info[0].desc.info.samples));
			const auto& meta = d.desc.info.metadata;
			LOGF("DrawTraceTarget: depth metadata kind %u at 0x%" PRIx64 " size 0x%" PRIx64 "\n",
			     static_cast<unsigned>(meta.kind), meta.range.address, meta.range.size);
			{
				const auto& ctx = buffer.GetRegisters();
				const auto& bc  = ctx.GetBlendControl(state.color_info[0].target_slot);
				const auto& vp0 = ctx.GetScreenViewport().viewports[0];
				const auto  sc  = calc_final_scissor(ctx.GetScreenViewport(), ctx.GetScanModeControl(),
				                                     state.color_info[0].Extent(), 0);
				LOGF("DrawTraceTarget: slot %u blend %d src %u dst %u comb %u a %u/%u/%u sep %d rt_mask 0x%08x "
				     "cb_shader_mask 0x%08x viewport (%g,%g %gx%g z %g/%g) scissor (%d,%d)-(%d,%d) extent %ux%u fmt %u\n",
				     state.color_info[0].target_slot, bc.enable ? 1 : 0, bc.color_srcblend, bc.color_destblend,
				     bc.color_comb_fcn, bc.alpha_srcblend, bc.alpha_destblend, bc.alpha_comb_fcn,
				     bc.separate_alpha_blend ? 1 : 0, ctx.GetRenderTargetMask(),
				     ctx.GetShaderRegisters().m_cbShaderMask, static_cast<double>(vp0.xoffset - vp0.xscale),
				     static_cast<double>(vp0.yoffset - vp0.yscale), static_cast<double>(vp0.xscale * 2.0f),
				     static_cast<double>(vp0.yscale * 2.0f), static_cast<double>(vp0.zoffset),
				     static_cast<double>(vp0.zscale), sc.left, sc.top, sc.right, sc.bottom,
				     state.color_info[0].Extent().width, state.color_info[0].Extent().height,
				     static_cast<unsigned>(state.color_info[0].desc.info.pixel_format));
			}
			if (meta.kind == ImageMetadataKind::Htile) SetMetaTraceAddressAuto(meta.range.address);
		}
	}
	// KYTY_DRAW_TRACE_DEPTH=<hex address>[:<count>] (diagnostic): the same for the draws whose depth
	// target starts at that address, with their colour target 0 and the PS's depth/kill outputs.
	static constexpr auto trace_depth_init = [] {
		std::pair<uint64_t, uint32_t> parsed {0, 200};
		if (const auto* value = std::getenv("KYTY_DRAW_TRACE_DEPTH"); value != nullptr) {
			char* end    = nullptr;
			parsed.first = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') parsed.second = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(trace_depth_init())>> trace_depth_once;
	const auto& trace_depth = trace_depth_once.Get(trace_depth_init);
	if (trace_depth.first != 0 && state.depth_info.desc.info.data.address == trace_depth.first) {
		static std::atomic<uint32_t> traced {0};
		if (traced.fetch_add(1, std::memory_order_relaxed) < trace_depth.second) {
			const auto& d = state.depth_info;
			LOGF("DrawTraceDepth: frame %d VS 0x%016" PRIx64 " PS 0x%016" PRIx64
			     " n=%u inst=%u colors=%u color0 0x%" PRIx64 " test %d write %d func %u stencil %d (func %u ref %u cmask 0x%x wmask 0x%x pass %u fail %u zfail %u) clear %d (%g) sclear %d (%u)\n",
			     static_cast<int>(m_context.GetGpu().GetFrameNum()),
			     state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
			     state.ps_active && state.ps_input_info.stage.program ? state.ps_input_info.stage.program->shader_hash : 0,
			     draw.index_count, draw.instance_count, state.color_count,
			     state.color_count > 0 ? state.color_info[0].desc.info.data.address : 0, d.depth_test_enable ? 1 : 0,
			     d.depth_write_enable ? 1 : 0, static_cast<unsigned>(d.depth_compare_op), d.stencil_test_enable ? 1 : 0,
			     static_cast<unsigned>(d.stencil_front.compareOp), d.stencil_front.reference, d.stencil_front.compareMask,
			     d.stencil_front.writeMask, static_cast<unsigned>(d.stencil_front.passOp),
			     static_cast<unsigned>(d.stencil_front.failOp), static_cast<unsigned>(d.stencil_front.depthFailOp),
			     d.depth_clear_enable ? 1 : 0, static_cast<double>(d.depth_clear_value), d.stencil_clear_enable ? 1 : 0,
			     static_cast<unsigned>(d.stencil_clear_value));
			LOGF("DrawTraceDepth: stencil plane 0x%" PRIx64 " size 0x%" PRIx64 "\n", d.desc.info.stencil.address,
			     d.desc.info.stencil.size);
			// KYTY_META_TRACE=stencil: trace the writers of this target's stencil plane.
			if (const auto* mode = std::getenv("KYTY_META_TRACE"); mode != nullptr && std::strcmp(mode, "stencil") == 0) {
				SetMetaTraceAddressStencil(d.desc.info.stencil.address);
			}
		}
	}
	// KYTY_DRAW_TRACE_PS=<hash>[:<sgpr>:<offset>] (diagnostic): logs the first 16 draws with this
	// pixel program: counts and user SGPRs; with <sgpr>:<offset> also the V# at that byte offset of
	// the pointer in user SGPRs <sgpr>/<sgpr>+1 and the first 64 dwords of its buffer, from guest
	// memory (with whether a GPU write of them is pending).
	static constexpr auto trace_ps_init = [] {
		std::array<uint64_t, 3> parsed {0, UINT64_MAX, 0};
		if (const auto* value = std::getenv("KYTY_DRAW_TRACE_PS"); value != nullptr) {
			char* end = nullptr;
			parsed[0] = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') {
				parsed[1] = std::strtoull(end + 1, &end, 10);
				if (end != nullptr && *end == ':') parsed[2] = std::strtoull(end + 1, &end, 0);
			}
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(trace_ps_init())>> trace_ps_once;
	const auto& trace_ps = trace_ps_once.Get(trace_ps_init);
	if (trace_ps[0] != 0 && state.ps_active && state.ps_input_info.stage.program != nullptr &&
	    state.ps_input_info.stage.program->shader_hash == trace_ps[0]) {
		static std::atomic<uint32_t> traced {0};
		if (traced.fetch_add(1, std::memory_order_relaxed) < 16) {
			const auto& user_data = state.ps_input_info.stage.resources->user_data;
			std::string text = fmt::format("DrawTracePs: hash=0x{:016x} index_count={} instances={} ud:",
			                               trace_ps[0], draw.index_count, draw.instance_count);
			for (const auto word: user_data) text += fmt::format(" {:08x}", word);
			if (trace_ps[1] != UINT64_MAX && trace_ps[1] + 1 < user_data.size()) {
				const uint64_t pointer = user_data[trace_ps[1]] |
				                         (static_cast<uint64_t>(user_data[trace_ps[1] + 1] & 0xffffu) << 32u);
				uint32_t vsharp[4] {};
				const bool read = LibKernel::Memory::TryReadBacking(pointer + trace_ps[2], vsharp, sizeof(vsharp));
				const uint64_t base = vsharp[0] | (static_cast<uint64_t>(vsharp[1] & 0xffffu) << 32u);
				text += fmt::format(" | V#@0x{:x}{}: {:08x} {:08x} {:08x} {:08x} base=0x{:x} stride={} records={}",
				                    pointer + trace_ps[2], read ? "" : "?", vsharp[0], vsharp[1], vsharp[2],
				                    vsharp[3], base, (vsharp[1] >> 16u) & 0x3fffu, vsharp[2]);
				// The same pointer read as an image + sampler table (T# 8 dwords, S# 4 dwords), and the first
				// bytes of the T#'s 256-byte-aligned base (and of its 0x100-spaced neighbours).
				uint32_t table[12] {};
				if (LibKernel::Memory::TryReadBacking(pointer + trace_ps[2], table, sizeof(table))) {
					text += " | T#S#:";
					for (const auto word: table) text += fmt::format(" {:08x}", word);
					const uint64_t tbase = (table[0] | (static_cast<uint64_t>(table[1] & 0xffu) << 32u)) << 8u;
					text += fmt::format(" tbase=0x{:x} gpu_dirty={}", tbase,
					                    m_context.GetBufferCache().HasGpuDirtyBytes(tbase, 256) ? 1 : 0);
					for (uint64_t k = 0; k < 16 && tbase >= 0x400; k++) {
						uint32_t probe[4] {};
						if (LibKernel::Memory::TryReadBacking(tbase - 0x400 + k * 0x100, probe, sizeof(probe))) {
							text += fmt::format(" [{:x}: {:08x} {:08x}]", tbase - 0x400 + k * 0x100, probe[0], probe[1]);
						}
					}
				}
				uint32_t words[64] {};
				if (base != 0 && LibKernel::Memory::TryReadBacking(base, words, sizeof(words))) {
					text += fmt::format(" gpu_dirty={} data:",
					                    m_context.GetBufferCache().HasGpuDirtyBytes(base, sizeof(words)) ? 1 : 0);
					for (const auto word: words) text += fmt::format(" {:08x}", word);
				}
			}
			LOGF("%s\n", text.c_str());
		}
	}
	// KYTY_DRAW_TRACE_CB=<vs hash>[:<max lines>[:<every n frames>]] (diagnostic; default 300 lines, every
	// 10 frames): for the draws with this vertex program, the first 64 dwords of each PS and VS buffer
	// from guest memory (with whether a GPU write of them is pending), to compare per-draw constants
	// (fade/dissolve values) between boots.
	static constexpr auto trace_cb_init = [] {
		std::array<uint64_t, 3> parsed {0, 300, 10};
		if (const auto* value = std::getenv("KYTY_DRAW_TRACE_CB"); value != nullptr) {
			char* end = nullptr;
			parsed[0] = std::strtoull(value, &end, 16);
			if (end != nullptr && *end == ':') {
				parsed[1] = std::strtoull(end + 1, &end, 10);
				if (end != nullptr && *end == ':') parsed[2] = std::max<uint64_t>(1, std::strtoull(end + 1, &end, 10));
			}
		}
		return parsed;
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(trace_cb_init())>> trace_cb_once;
	const auto& trace_cb = trace_cb_once.Get(trace_cb_init);
	if (trace_cb[0] != 0 && state.vertex_info[0].stage.program != nullptr &&
	    state.vertex_info[0].stage.program->shader_hash == trace_cb[0]) {
		static std::atomic<uint32_t> traced {0};
		static std::atomic<int64_t>  last_frame {-1000000};
		const auto                   frame = static_cast<int64_t>(m_context.GetGpu().GetFrameNum());
		const auto                   last  = last_frame.load(std::memory_order_relaxed);
		if ((frame == last || frame - last >= static_cast<int64_t>(trace_cb[2])) &&
		    traced.fetch_add(1, std::memory_order_relaxed) < trace_cb[1]) {
			if (frame != last) last_frame.store(frame, std::memory_order_relaxed);
			std::string text = fmt::format("DrawTraceCb: frame={} q={} seq={} tick={} vs=0x{:016x} ps=0x{:016x} n={} inst={}", frame,
			                               static_cast<int32_t>(HangTrace::CpContextQueue()), HangTrace::CpContextSequence(),
			                               m_context.GetCommandScheduler().CurrentTick(),
			                               trace_cb[0],
			                               state.ps_active && state.ps_input_info.stage.program != nullptr
			                                   ? state.ps_input_info.stage.program->shader_hash
			                                   : 0,
			                               draw.index_count, draw.instance_count);
			const auto dump = [&](const char* stage, const auto* program, const auto* resources) {
				if (program == nullptr || resources == nullptr) return;
				text += fmt::format(" | {} ud:", stage);
				for (const auto word: resources->user_data) text += fmt::format(" {:08x}", word);
				for (uint32_t i = 0; i < program->info.buffers.size() && i < resources->buffers.size(); i++) {
					const auto r    = DecodeNativeDescriptor<ShaderBufferResource>(resources->buffers[i]);
					const auto base = r.Base48();
					text += fmt::format(" | {}b{}{}=0x{:x}/s{}/n{}", stage, i, program->info.buffers[i].written ? "w" : "",
					                    base, r.Stride(), r.NumRecords());
					uint32_t words[64] {};
					if (base != 0 && LibKernel::Memory::TryReadBacking(base, words, sizeof(words))) {
						text += fmt::format(" d{}:", m_context.GetBufferCache().HasGpuDirtyBytes(base, sizeof(words)) ? 1 : 0);
						for (const auto word: words) text += fmt::format(" {:08x}", word);
					}
					if (base != 0) text += WatchRange::DescribeWriters(base);
					// The newest writer's first user-data pointer (an upload-copy table): its first 64 dwords, and
					// 24 dwords at each 0x20xxxxxxxx pointer in it (the copy sources).
					uint32_t wud0 = 0;
					uint32_t wud1 = 0;
					if (base != 0 && WatchRange::NewestWriterUd(base, &wud0, &wud1)) {
						const uint64_t table = wud0 | (static_cast<uint64_t>(wud1 & 0xffffu) << 32u);
						uint32_t       tw[64] {};
						if (table != 0 && LibKernel::Memory::TryReadBacking(table, tw, sizeof(tw))) {
							text += fmt::format(" table@0x{:x}:", table);
							for (const auto word: tw) text += fmt::format(" {:08x}", word);
							uint32_t shown = 0;
							for (uint32_t k = 0; k + 1 < 64 && shown < 8; k++) {
								const uint64_t ptr = tw[k] | (static_cast<uint64_t>(tw[k + 1]) << 32u);
								if ((ptr >> 32u) != 0x20u) continue;
								uint32_t src[24] {};
								if (LibKernel::Memory::TryReadBacking(ptr, src, sizeof(src))) {
									text += fmt::format(" src[{}]@0x{:x}:", k, ptr);
									for (const auto word: src) text += fmt::format(" {:08x}", word);
									shown++;
								}
							}
						}
					}
				}
			};
			if (state.ps_active) dump("ps", state.ps_input_info.stage.program, state.ps_input_info.stage.resources);
			dump("vs", state.vertex_info[0].stage.program, state.vertex_info[0].stage.resources);
			LOGF("%s\n", text.c_str());
		}
	}
	if (WatchRange::Enabled() || WatchRange::CpFaultWritersEnabled()) {
		WatchRange::SetCurrentOp("draw", state.ps_active && state.ps_input_info.stage.program != nullptr
		                                     ? state.ps_input_info.stage.program->shader_hash
		                                     : 0);
	}
	// KYTY_FRAME_TRACE (renderCompute.cpp): one line per draw while a frame trace runs.
	if (FrameTraceTake()) {
		const auto* ps = state.ps_active ? state.ps_input_info.stage.program : nullptr;
		std::string text = fmt::format(
		    "FrameTrace DRAW: frame={} vs=0x{:016x} ps=0x{:016x} n={} inst={}",
		    static_cast<int>(m_context.GetGpu().GetFrameNum()),
		    state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
		    ps != nullptr ? ps->shader_hash : 0, draw.index_count, draw.instance_count);
		text += fmt::format(" q={} seq={}", static_cast<int32_t>(HangTrace::CpContextQueue()), HangTrace::CpContextSequence());
		// The vertex program's images (vt<i>[w]: written).
		if (const auto* vs = state.vertex_info[0].stage.program;
		    vs != nullptr && state.vertex_info[0].stage.resources != nullptr) {
			const auto& vs_resources = *state.vertex_info[0].stage.resources;
			for (uint32_t i = 0; i < vs->info.images.size() && i < vs_resources.images.size(); i++) {
				const auto r = DecodeNativeDescriptor<ShaderTextureResource>(vs_resources.images[i]);
				text += fmt::format(" | vt{}{}=0x{:x}/f{}/{}x{}", i, vs->info.images[i].written ? "w" : "", r.Base40(),
				                    static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
				                    static_cast<uint32_t>(r.Height5()) + 1u);
			}
		}
		for (uint32_t i = 0; i < state.color_count; i++) {
			const auto& info = state.color_info[i].desc.info;
			text += fmt::format(" | c{}=0x{:x}/vk{}/{}x{}", i, info.data.address,
			                    static_cast<uint32_t>(info.pixel_format), info.extent.width, info.extent.height);
		}
		if (state.depth_info.desc.info.data.address != 0) {
			text += fmt::format(" | z=0x{:x} test{} write{}", state.depth_info.desc.info.data.address,
			                    state.depth_info.depth_test_enable ? 1 : 0, state.depth_info.depth_write_enable ? 1 : 0);
		}
		if (ps != nullptr && state.ps_input_info.stage.resources != nullptr) {
			const auto& resources = *state.ps_input_info.stage.resources;
			for (uint32_t i = 0; i < ps->info.images.size() && i < resources.images.size(); i++) {
				const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
				text += fmt::format(" | t{}{}@{:x}=0x{:x}/f{}/{}x{}", i, ps->info.images[i].written ? "w" : "", ps->info.images[i].first_use_pc, r.Base40(),
				                    static_cast<uint32_t>(r.Format()), static_cast<uint32_t>(r.Width5()) + 1u,
				                    static_cast<uint32_t>(r.Height5()) + 1u);
			}
			for (uint32_t i = 0; i < ps->info.images.size() && i < resources.images.size(); i++) {
				const auto& raw = resources.images[i];
				if (DecodeNativeDescriptor<ShaderTextureResource>(raw).Base40() == 0) {
					text += fmt::format(" | t{}raw(n{})={:08x} {:08x} {:08x} {:08x}", i, raw.dword_count, raw.dwords[0],
					                    raw.dwords[1], raw.dwords[2], raw.dwords[3]);
				}
			}
			if (!resources.gpu_fills.empty()) {
				text += fmt::format(" | gpufills={}:", resources.gpu_fills.size());
				for (size_t i = 0; i < resources.gpu_fills.size() && i < 8; i++) {
					text += fmt::format(" {}@0x{:x}", resources.gpu_fills[i].flat_offset, resources.gpu_fills[i].address);
				}
			}
			for (uint32_t i = 0; i < ps->info.buffers.size() && i < resources.buffers.size(); i++) {
				const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				text += fmt::format(" | b{}{}=0x{:x}/n{}", i, ps->info.buffers[i].written ? "w" : "", r.Base48(),
				                    r.NumRecords());
			}
		}
		LOGF("%s\n", text.c_str());
	}
	// KYTY_CONT_KERNEL: both tests below are functions of the run key (registers and programs) and
	// passed for the record.
	if (!m_kernel && CopyDepthToColor(buffer, draw, emit, state)) {
		m_run_active = false; // the copy is not a draw of the run
		return;
	}
	// KYTY_EQAA_MIXED through NV mixed samples (colour samples below depth samples): no per-sample
	// shading. A pixel shader that shades per sample runs its per-pixel copy (shaders.cpp); one
	// without such a copy (the rewrite gave up) is skipped instead of losing the device.
	if (!m_kernel && state.ps_active && state.color_count != 0 && state.depth_info.image_id &&
	    state.color_info[0].desc.info.samples < state.depth_info.desc.info.samples &&
	    buffer.GetGraphics().mixed_attachment_samples ==
	        GraphicContext::MixedAttachmentSamples::Nv &&
	    !EqaaPerSampleShading()) {
		if (const auto variant = PixelRate::Find(state.programs.pixel.id);
		    variant.shades_per_sample && variant.module == nullptr) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("Pipeline: EQAA draw skipped: pixel shader 0x%016" PRIx64
				     " shades per sample and has no per-pixel copy (KYTY_EQAA_MIXED)\n",
				     state.ps_input_info.stage.program != nullptr
				         ? state.ps_input_info.stage.program->shader_hash
				         : uint64_t {0});
			}
			m_run_active = false;
			return;
		}
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	// CommitStats: image transitions recorded while this draw is committed.
	const auto stats_transitions = CommitStats::Enabled() ? Image::RecordedTransitions() : uint64_t {0};
	if (!m_kernel && m_context.GetPipelineCache().PipelinePrefetchEnabled() && !CpPipelinePrefetchSkipped() &&
	    (m_binding_plan == nullptr || m_binding_plan->pipeline == nullptr)) {
		const CommitStats::SubScope prefetch_scope(CommitStats::Sub::Prefetch);
		m_context.GetPipelineCache().PrefetchGraphicsPipeline(
		    {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
		    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
		    state.programs);
	}
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	// Reused per thread (draws run on the GPU thread under the render mutex). KYTY_BK_LEAN: one
	// function-local vector instead of the thread_local one (no TLS lookup per draw; one thread draws).
	static thread_local std::vector<MeshDrawSegment> mesh_segments_tls;
	static std::vector<MeshDrawSegment>              mesh_segments_one;
	auto& mesh_segments = DrawPrep::BkLeanMode() != DrawPrep::Bookkeep::Mode::Off ? mesh_segments_one : mesh_segments_tls;
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

	// KYTY_BDA_REACH: the index bytes the mesh prologue reads.
	m_bda_mesh_index = {};
	m_bda_mesh_known = false;
	if (mesh_active && !mesh_indirect) {
		// A non-indexed draw pushes index size 0: the prologue reads no index (empty reach).
		m_bda_mesh_known = true;
		if (draw.IsIndexed()) {
			m_bda_mesh_index = {index_source.address,
			                    index_source.address + static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size};
		}
	} else if (mesh_indirect) {
		// Native indirect mesh draw: the counts are GPU data. A non-indexed record's parameter
		// block holds index size 0 (empty reach). An indexed one reads inside the bound range
		// [INDEX_BASE, +INDEX_BUFFER_SIZE) (DrawIndirectNative requires a size), the same range the
		// full pass relies on: it is the only one registered for these reads (above). A record whose
		// start index runs past it is counted by the conversion (StatusIndexRange, logged as
		// "MeshIndirect: status 0x4"); verify mode traps such reads too.
		m_bda_mesh_known = true;
		if (emit.mesh_inputs.indexed) {
			m_bda_mesh_index = {index_source.address, index_source.address + index_source.size};
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
	EXIT_IF(emit.gpu_instance_address != 0 && (mesh_active || indirect != nullptr));
	if (emit.gpu_instance_address != 0) {
		// Acquired before the bindings resolve, as the indirect records below.
		(void)m_context.GetBufferCache().ObtainBuffer(emit.gpu_instance_address, sizeof(uint32_t),
		                                              false);
	}
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
	CommitStats::StepMark(CommitStats::Step::ExecHead);
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareBindings");
		// KYTY_DRAW_RUN continuation: the stages keep the previous draw's textures and samplers.
		// KYTY_CONT_KERNEL: the same programs as the record's, which had no GDS (run eligibility) and
		// no bindless heaps (the candidate's precondition): the lean preparation.
		for (uint32_t i = 0; m_kernel && i < 1; i++) {
			PrepareBindingsContinuation(state.vertex_info[0].stage, bindings.vertex[0],
			                            plan_stages ? &plan->vertex : nullptr);
			descriptor_stages[stage_count++] = &bindings.vertex[0];
			if (state.ps_active) {
				PrepareBindingsContinuation(state.ps_input_info.stage, *bindings.pixel,
				                            plan_stages && plan->pixel_active ? &plan->pixel : nullptr);
				descriptor_stages[stage_count++] = &*bindings.pixel;
			}
		}
		for (uint32_t i = 0; !m_kernel && i < vertex_stages.size(); i++) {
			// KYTY_UNNORMALIZED_FIX: a hull shader's normalized-LOD copy binds normalized samplers.
			bindings.vertex[i].normalize_unnormalized = state.programs.vertex[i].normalized_samplers;
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i],
			                plan_stages ? &plan->vertex : nullptr, m_run_active && !m_run_texdelta);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (!m_kernel && state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel,
			                plan_stages && plan->pixel_active ? &plan->pixel : nullptr,
			                m_run_active && !m_run_texdelta);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
		if (m_kernel_verify && m_run_active && !m_run_pipeswap) {
			// What the lean preparation leaves out must be absent.
			bool lean = vertex_stages.size() == 1;
			for (uint32_t i = 0; i < stage_count; i++) {
				lean = lean && descriptor_stages[i]->gds.buffer == nullptr &&
				       descriptor_stages[i]->bindless_heaps.empty();
			}
			ContKernel::Check(ContKernel::Stage, lean);
		}
	}
	CommitStats::StepMark(CommitStats::Step::PrepStages);
	CommitStats::Mark(CommitStats::Phase::PrepareBindings);
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	{
		KYTY_PROFILER_DETAIL_BLOCK("Draw::PrepareGraphicsBindings");
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count},
		                        m_run_active && !m_run_texdelta);
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
				OwnedCounters::Add(DrawPrep::GetBindingTotals().vertex_ranges_used, 1);
			} else {
				Profiler::CountFrameEvent(Profiler::FrameEvent::DrawPrepBindingFallbackVm);
			}
		}
		if (CpCommit::Enabled(CpCommit::Part::Draws)) {
			AcquireVertexBuffersInto(buffer, state.vertex_info[0], vertex_ranges, vertex_bindings);
		} else {
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], vertex_ranges);
		}
		CommitStats::StepMark(CommitStats::Step::VertexAcquire);
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
	// KYTY_GPU_INSTANCE_COUNT: a one-record indirect draw in the stream ring with the known counts
	// and offsets (still preparation: the ring may wait here); the GPU copies the instance count in
	// right before the draw.
	vk::Buffer     instance_record        = nullptr;
	vk::DeviceSize instance_record_offset = 0;
	if (emit.gpu_instance_address != 0) {
		auto&          stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const uint32_t record[5] {draw.index_count, 0u,
		                          draw.IsIndexed() ? 0u : emit.first_vertex,
		                          draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset)
		                                           : emit.first_instance,
		                          emit.first_instance};
		instance_record        = stream.Handle();
		instance_record_offset = stream.Copy(record, draw.IsIndexed() ? 20u : 16u, 16u);
	}
	CommitStats::StepMark(CommitStats::Step::IndexPrepare);
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
	// KYTY_GPU_INSTANCE_COUNT: its copy ends rendering too.
	const bool run_indirect_barrier =
	    (indirect != nullptr && buffer.ActiveRenderingSerial() != m_indirect_barrier_rendering &&
	     IndirectArgsBarrierNeeded(buffer, indirect)) ||
	    instance_record;
	// KYTY_DRAW_RUN_TARGETS: kept targets whose attachments changed during the buffer work are
	// resolved now, with the same repeated binding work as a continuation's fallback.
	const bool targets_late = m_run_targets_kept &&
	                          (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true, true));
	if (targets_late) {
		m_run_targets_kept = false;
		OwnedCounters::Add(DrawRun::GetTotals().targets_late, 1);
	}
	// KYTY_DRAW_RUN_TEXDELTA: a delta's textures are its own (resolved above); the record's attachments
	// must be unchanged and no texture may be an attachment, a storage binding or over one.
	uint32_t texdelta_late = 0;
	if (m_run_active && m_run_texdelta) {
		texdelta_late = !DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true, true) || run_indirect_barrier
		                    ? 1u + TexDeltaStats::LateImages
		                    : TexDeltaTextureVerdict(stages);
	}
	if (targets_late ||
	    (m_run_active && (m_run_texdelta ? texdelta_late != 0
	                                     : (!DrawRunCommandUnchanged(buffer) || !DrawRunImagesUnchanged(true) ||
	                                        run_indirect_barrier)))) {
		const CommitStats::SubScope late_scope(CommitStats::Sub::LateFallback);
		if (!targets_late) {
			DrawRun::GetTotals().late_fallbacks.fetch_add(1, std::memory_order_relaxed);
			DrawRun::CountMiss(DrawRun::Miss::Images);
		}
		if (texdelta_late != 0) {
			TexDeltaStats::g_counters.late[texdelta_late - 1].fetch_add(1, std::memory_order_relaxed);
		}
		m_run_texdelta = false;
		m_run_active   = false;
		if (std::exchange(m_kernel, false)) {
			ContKernel::g_counters.left.fetch_add(1, std::memory_order_relaxed);
			if (CommitStats::Enabled()) [[unlikely]] {
				m_stats_kernel = CommitStats::KernelOutcome::Left;
			}
		}
		DrawRunRecallFallback();
		DrawRunTargets(buffer, draw, m_run_slice_offset, state);
		// No plan: its shader data was taken by the first preparation above.
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			bindings.vertex[i].normalize_unnormalized = state.programs.vertex[i].normalized_samplers;
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
	// KYTY_REBIND_AFTER_JOIN: a buffer the bindings resolved was retired since (a later lookup of
	// this draw - another binding, a vertex, index or argument range - joined it into a new
	// buffer): the recorded draw would read and write a buffer nothing uses again, its writes lost
	// while the range stays GPU-owned. Resolve the buffers and the per-draw ranges again while the
	// count moves (at most 4 times).
	if (RebindAfterJoinEnabled() || std::exchange(m_lazy_find_draw, false)) {
		auto& cache = m_context.GetBufferCache();
		for (uint32_t pass = 0; pass < 4 && cache.UnregisterEpoch() != m_rebind_epoch; pass++) {
			NoteRebindAfterJoin("draw", draw.Name());
			m_rebind_epoch = cache.UnregisterEpoch();
			{
				const BufferCache::UploadBatch upload_batch(cache);
				for (auto* stage: stages) {
					RebindBuffers(*stage);
				}
			}
			if (!mesh_active) {
				vertex_bindings = PreparedVertexBuffers {};
				if (CpCommit::Enabled(CpCommit::Part::Draws)) {
					AcquireVertexBuffersInto(buffer, state.vertex_info[0], nullptr, vertex_bindings);
				} else {
					vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0], nullptr);
				}
				index_binding = PrepareIndexBuffer(buffer, index_source);
			}
			if (indirect != nullptr) {
				indirect_buffers = ObtainIndirectBuffers(buffer, *indirect);
			}
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
		DrawRunRecallFallback();
	}
	// KYTY_DRAW_RUN_TEXDELTA=2: whether the would-be delta would have passed its late check (the normal
	// path resolved its textures above). One that would have fallen back is not compared.
	bool texdelta_compare = false;
	if (m_texdelta_verify) {
		TexDeltaStats::Tick(TexDeltaStats::g_counters.verify_draws.fetch_add(1, std::memory_order_relaxed) + 1);
		texdelta_compare = DrawRunCommandUnchanged(buffer) && DrawRunImagesUnchanged(true, true) &&
		                   !run_indirect_barrier && TexDeltaTextureVerdict(stages) == 0;
		if (!texdelta_compare) {
			TexDeltaStats::g_counters.verify_fallback.fetch_add(1, std::memory_order_relaxed);
		}
	}
	CommitStats::StepMark(CommitStats::Step::LateCheck);

	// KYTY_LOD_STATS_PLAIN_VARIANT: without a mip-statistics counter on any image the pixel
	// program's feedback records nothing; its feedback-free variant then gives the same results
	// without forcing the depth/stencil tests after a shader that can discard.
	const auto* programs    = &state.programs;
	bool        plain_pixel = false;
	PipelineCache::GraphicsPrograms plain_programs;
	if (state.ps_active && bindings.pixel.has_value()) {
		const auto plain_mode = LodStatsCounter::PlainVariant();
		const auto plain      = plain_mode != LodStatsCounter::Plain::Off
		                            ? PipelineCache::PlainPixelProgram(state.PixelPrep())
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
	// KYTY_SPIRV_VARIANT_LIVE: the edited copies of the programs (same bindings and layout); the
	// prepared plan's pipeline is then not used.
	PipelineCache::GraphicsPrograms variant_programs;
	bool                            variant_used = false;
	if (const auto variant_group = PipelineCache::SpirvVariantLiveGroup();
	    variant_group != 0 && PipelineCache::SpirvVariantsConfigured()) [[unlikely]] {
		variant_programs = *programs;
		if (state.ps_active) {
			if (const auto v = PipelineCache::SpirvVariantProgram(variant_programs.pixel, variant_group)) {
				variant_programs.pixel = v;
				variant_used           = true;
			}
		}
		for (auto& vertex: variant_programs.vertex) {
			if (const auto v = PipelineCache::SpirvVariantProgram(vertex, variant_group)) {
				vertex       = v;
				variant_used = true;
			}
		}
		if (variant_used) {
			programs = &variant_programs;
			const auto n = g_spirv_variant_draws.fetch_add(1, std::memory_order_relaxed) + 1;
			if ((n & (n - 1)) == 0 && n >= 1024) {
				std::printf("KYTY_SPIRV_VARIANT_LIVE: %" PRIu64 " draws ran a variant so far\n", n);
			}
		}
	}
	// KYTY_DRAW_PREP_BINDINGS pipeline: the object the preparing thread found for this key, when
	// the draw's targets, topology, restart flag and plain-pixel choice are the ones the key was
	// built for and no pipeline was replaced since (DrawPrep::PlannedPipeline).
	auto&       pipeline_cache = m_context.GetPipelineCache();
	const auto  colors_span    = std::span<const RenderColorInfo> {state.color_info, state.color_count};
	if (pipeline_cache.PipelineSkipEnabled()) {
		pipeline_cache.NoteSkipFrame(m_context.GetTextureCache().Frame());
		if (SkipRecurrentTargetsOnly()) NoteSkipTargets(m_context, state);
	}
	const auto* ps_input       = state.ps_active ? &state.ps_input_info : nullptr;
	// KYTY_CONT_KERNEL: the pipeline is selected as for any draw. The record's is NOT reused: the
	// pipeline's vertex input comes from the vertex buffers' V# words (per-draw data, outside the run
	// key); verify mode counts how often the record's would have been wrong ("pipeline").
	const bool kernel_valid = m_kernel_record.plain_pixel == plain_pixel &&
	                          m_kernel_record.generation == pipeline_cache.PipelineGeneration();
	if (m_kernel && !kernel_valid) {
		m_kernel = false; // barrier safety depends on the plain-pixel choice
		ContKernel::g_counters.left.fetch_add(1, std::memory_order_relaxed);
		if (CommitStats::Enabled()) [[unlikely]] {
			m_stats_kernel = CommitStats::KernelOutcome::Left;
		}
	}
	const auto* planned_pipeline =
	    plan_stages && plan->pipeline != nullptr && !variant_used
	        ? DrawPrep::PlannedPipeline(pipeline_cache, *plan, colors_span, state.depth_info,
	                                    topology, primitive_restart_enable, plain_pixel)
	        : nullptr;
	const PipelineCache::Pipeline* pipeline_object = planned_pipeline;
	if (planned_pipeline != nullptr && !plan_verify) {
		pipeline_cache.NotePlannedPipeline(state.depth_info, ps_input);
	} else {
		const bool wait_stats     = PipelineWaitStats();
		const auto pipeline_begin = wait_stats ? std::chrono::steady_clock::now()
		                                       : std::chrono::steady_clock::time_point {};
		const bool skip_on    = planned_pipeline == nullptr && pipeline_cache.PipelineSkipEnabled();
		const auto refusal    = skip_on ? DrawSkipRefusal(m_context, state, vertex_stages, mesh_active)
		                                : SkipRefusal::Stages;
		const bool skippable  = skip_on && refusal == SkipRefusal::None;
		const bool skip_stats = skip_on && SkipStatsEnabled();
		const auto lookup_begin =
		    skip_stats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		pipeline_object = pipeline_cache.TryGetGraphicsPipeline(
		    colors_span, state.depth_info, vertex_stages, buffer, ps_input, topology,
		    primitive_restart_enable, *programs, skippable);
		if (skip_stats) {
			NoteSkipLookup(refusal, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                                  std::chrono::steady_clock::now() - lookup_begin)
			                                                  .count()));
		}
		if (wait_stats) {
			NotePipelineWait(std::chrono::steady_clock::now() - pipeline_begin);
		}
		if (pipeline_object == nullptr) {
			NoteSkippedOcclusionDraw(m_context, buffer);
			// KYTY_PIPELINE_SKIP: skipped until its pipeline is built. Nothing was recorded; no
			// draw-run record may be continued or recalled across it (its bindings replaced the
			// record's).
			m_run.valid        = false;
			m_run_active       = false;
			m_run_targets_kept = false;
			for (auto& record: m_run_kept) {
				record.valid = false;
			}
			return;
		}
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
	const bool  kernel_verify = m_kernel_verify && m_run_active && !m_run_pipeswap;
	if (kernel_verify) {
		ContKernel::Check(ContKernel::Topology, topology == m_kernel_record.topology);
		ContKernel::Check(ContKernel::Restart, primitive_restart_enable == m_kernel_record.primitive_restart);
		if (kernel_valid) {
			ContKernel::Check(ContKernel::Pipeline, pipeline_object == m_kernel_record.pipeline);
		}
	}
	CommitStats::StepMark(CommitStats::Step::PipelineSel);
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
	// KYTY_BK_LEAN (drawPrep/bookkeep.h): a continuation's rendering state is its record's, used by
	// reference; any other draw's is built here (not constructed when unused).
	std::optional<RenderState> rendering_own;
	const RenderState*         rendering_ref = nullptr;
	const auto                 lean_mode     = DrawPrep::BkLeanMode();
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
		if (lean_mode == DrawPrep::Bookkeep::Mode::Off) {
			rendering_own = m_run.rendering;
		} else {
			rendering_ref = &m_run.rendering;
			if (lean_mode == DrawPrep::Bookkeep::Mode::Verify) {
				rendering_own = m_run.rendering; // compared with the record after the draw
			}
			auto& bk = DrawPrep::BkCounters();
			bk.Add(bk.lean_rendering);
		}
		m_depth_feedback.valid = false;
		m_acq_last.valid       = false; // KYTY_IMG_ACQFB counts full acquisitions only
		m_acqfb_this           = false;
	} else {
		// KYTY_DRAW_RUN_ACQUIRE: the same targets and scissor union right after the recorded draw, in
		// its rendering instance, with the attachments as it left them: its acquisition, whose
		// transitions, claims, refreshes and depth-feedback bookkeeping would all be repeats.
		// (Verify mode checks a would-be continuation's acquisition as the continuation's.)
		if (ImgSide::StatsEnabled()) [[unlikely]] {
			m_acq_why = !DrawRun::Enabled() || !DrawRun::AcquireReuseEnabled() || m_run_verify || m_texdelta_verify
			                ? uint8_t {1}
			                : DrawRunAcquireRefusal(buffer, state, stages, bounded ? &written : nullptr);
		}
		CommitStats::SubScope candidate_scope(CommitStats::Sub::AcqCandidate);
		const bool acquire_reuse =
		    DrawRun::Enabled() && DrawRun::AcquireReuseEnabled() && !m_run_verify && !m_texdelta_verify &&
		    DrawRunAcquireCandidate(buffer, state, stages, bounded ? &written : nullptr);
		candidate_scope.Stop();
		if (acquire_reuse) {
			OwnedCounters::Add(DrawRun::GetTotals().acquire_reused, 1);
		}
		// KYTY_IMG_ACQSEED: no run record, but the previous draw's acquisition was seeded.
		const auto seed_mode = ImgSide::AcqSeedMode();
		const bool seed_reuse = !acquire_reuse && seed_mode != ImgSide::Mode::Off && !m_run_prev_valid &&
		                        DrawRun::Enabled() && DrawRun::AcquireReuseEnabled() && !m_run_verify &&
		                        !m_texdelta_verify &&
		                        DrawRunSeedCandidate(buffer, state, stages, bounded ? &written : nullptr);
		// KYTY_IMG_ACQFB: no run record and no seed, but the previous draw's acquisition (also one that
		// samples its attachments) was recorded and would be repeated (AcqFbCandidate).
		const auto          fb_mode = ImgSide::AcqFbMode();
		DepthAccessDecision fb_decision;
		const bool          fb_reuse = !acquire_reuse && !seed_reuse && fb_mode != ImgSide::Mode::Off &&
		                      !m_run_prev_valid && DrawRun::Enabled() && DrawRun::AcquireReuseEnabled() &&
		                      !m_run_verify && !m_texdelta_verify &&
		                      AcqFbCandidate(buffer, state, stages, bounded ? &written : nullptr, fb_decision);
		m_acqfb_this = false;
		if (fb_reuse && fb_mode == ImgSide::Mode::On) {
			ImgSide::Bump(ImgSide::S::AcqFbHit);
			AcqFbApply(buffer, state, fb_decision);
			rendering_own    = m_acqfb.rendering;
			feedback_aspects = m_acqfb.feedback;
			m_acq_last.valid = false; // the count's "previous acquisition" is a full one
			m_acqfb_this     = true;
		} else if (seed_reuse && seed_mode == ImgSide::Mode::On) {
			ImgSide::Bump(ImgSide::S::AcqSeedHit);
			rendering_own          = m_acq_seed.rendering;
			m_depth_feedback.valid = false;
			m_acq_last.valid       = false; // KYTY_IMG_ACQFB counts full acquisitions only
		} else if (acquire_reuse && DrawRun::GetMode() == DrawRun::Mode::On) {
			rendering_own          = m_run.rendering;
			m_depth_feedback.valid = false;
			m_acq_last.valid       = false;
		} else if (seed_reuse) {
			m_acq_last.valid = false;
			// KYTY_IMG_ACQSEED=verify: the acquisition the seed would skip must be a repeat (the checks
			// of KYTY_DRAW_RUN_ACQUIRE's verify mode); it provides the result.
			const auto transitions = Image::RecordedTransitions();
			rendering_own = AcquireRenderTargets(buffer, state.color_info, state.color_count,
			                                     state.depth_info, feedback_aspects, stages,
			                                     bounded ? &written : nullptr);
			ImgSide::Bump(ImgSide::S::AcqSeedChecks);
			if (!(*rendering_own == m_acq_seed.rendering) || feedback_aspects) {
				ImgSide::ReportMismatch(ImgSide::S::AcqSeedMismatch, "acqseed: the acquisition's rendering state differs");
			} else if (Image::RecordedTransitions() != transitions) {
				ImgSide::ReportMismatch(ImgSide::S::AcqSeedMismatch, "acqseed: the acquisition recorded transitions");
			} else if (DrawRunMarksChange(m_acq_seed.marks, false, true, false, true) != 0) {
				ImgSide::ReportMismatch(ImgSide::S::AcqSeedMismatch, "acqseed: the acquisition changed an attachment's state");
			}
		} else {
			// KYTY_IMG_ACQFB (count build): is this acquisition a repeat of the previous one?
			const bool fb_count = !acquire_reuse && !seed_reuse && ImgSide::AcqFbMode() != ImgSide::Mode::Off;
			AcqFbProbe fb_probe;
			if (fb_count) {
				fb_probe = AcqFbBefore(buffer, state);
			}
			const auto acquire_transitions =
			    acquire_reuse || texdelta_compare ? Image::RecordedTransitions() : uint64_t {0};
			const auto fb_transitions = fb_reuse ? Image::RecordedTransitions() : uint64_t {0};
			const auto fb_keeps       = m_acq_keeps;
			rendering_own = AcquireRenderTargets(buffer, state.color_info, state.color_count,
			                                 state.depth_info, feedback_aspects, stages,
			                                 bounded ? &written : nullptr);
			m_acqfb_this = true;
			if (fb_reuse) {
				// KYTY_IMG_ACQFB=verify: the acquisition a reuse would skip must be a repeat of the seed's,
				// with the effects the reuse performs (AcqFbApply) and no other.
				ImgSide::Bump(ImgSide::S::AcqFbChecks);
				const auto& d = m_acq_decision;
				const bool  same_decision =
				    m_acq_has_depth == m_acqfb.has_depth &&
				    (!m_acq_has_depth ||
				     (d.sampled == fb_decision.sampled && d.feedback == fb_decision.feedback &&
				      d.layout == fb_decision.layout && d.access == fb_decision.access && d.keep == fb_decision.keep &&
				      d.adopt == fb_decision.adopt));
				const bool depth_writes = m_acqfb.has_depth && static_cast<bool>(state.depth_info.AttachmentWriteAspects());
				// Content serials as AcqFbApply leaves them: fresh for every colour target, the depth target's
				// restored when the draw writes nothing (with KYTY_ALIAS_SYNC_SKIP) and fresh otherwise.
				bool serial_kept = true;
				{
					auto&            cache = m_context.GetTextureCache();
					std::scoped_lock lock {cache.m_lock};
					for (uint32_t i = 0; i < state.color_count && i < m_acqfb.marks.size(); i++) {
						const auto* color = cache.m_slot_images.try_get(state.color_info[i].image_id);
						serial_kept = serial_kept && color != nullptr &&
						              (color->ContentSerial() == m_acqfb.marks[i].serial) ==
						                  ImgSide::AcqFbSerialKept(false, true, TextureCache::AliasSyncSkipOn());
					}
					if (m_acqfb.has_depth) {
						const auto* depth = cache.m_slot_images.try_get(state.depth_info.image_id);
						const bool  same  = depth != nullptr && depth->ContentSerial() == m_acqfb.marks.back().serial;
						serial_kept = serial_kept && depth != nullptr &&
						              same == ImgSide::AcqFbSerialKept(true, depth_writes, TextureCache::AliasSyncSkipOn());
					}
				}
				if (!(*rendering_own == m_acqfb.rendering) || feedback_aspects != m_acqfb.feedback) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition's rendering state differs");
				} else if (!same_decision || m_acq_color_bound != m_acqfb.color_bound) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition decided differently");
				} else if (Image::RecordedTransitions() != fb_transitions) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition recorded transitions");
				} else if (m_acq_keeps - fb_keeps != (fb_decision.adopt ? 1u : 0u)) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition's feedback keep differs");
				} else if (DrawRunMarksChange(m_acqfb.marks, false, false, false, true) != 0) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition changed an attachment's state");
				} else if (!serial_kept) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition's content serials differ from the replay's");
				} else if (m_acqfb.has_depth && (!m_depth_feedback.valid || m_depth_feedback.id != state.depth_info.image_id ||
				                                 m_depth_feedback.writes != depth_writes)) {
					ImgSide::ReportMismatch(ImgSide::S::AcqFbMismatch, "acqfb: the acquisition's depth feedback note differs");
				}
			}
			if (fb_count) {
				AcqFbAfter(fb_probe, *rendering_own, feedback_aspects);
			}
			if (acquire_reuse) {
				// Verify mode: the acquisition a reuse skips must have been a repeat.
				DrawRun::CountVerifyCheck();
				if (!(*rendering_own == m_run.rendering) || feedback_aspects) {
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
			if (texdelta_compare) {
				TexDeltaStats::Check(TexDeltaStats::Transitions, Image::RecordedTransitions() == acquire_transitions);
				TexDeltaVerify(state, *rendering_own, feedback_aspects, false, stages);
			}
			if (run_verify_images) {
				if (const auto change = DrawRunImagesChange(false, false, true); change != 0) {
					DrawRun::CountVerifyCheck();
					DrawRun::ReportMismatch("attachment state after acquisition", change);
				}
			}
		}
	}
	if (rendering_ref == nullptr) {
		rendering_ref = &*rendering_own;
	}
	const RenderState& rendering = *rendering_ref;
	CommitStats::StepMark(CommitStats::Step::Acquire);
	CommitStats::Mark(CommitStats::Phase::AcquireTargets);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. Until BeginRendering() only state commands are recorded through vk_buffer; image
	// transitions and dependencies go through the barrier batcher, which BeginRendering() flushes
	// (or, for a safe draw continuing the same rendering instance, may sink past the draw).
	const bool* programs_safe = plan_statics && plan->plain_pixel == plain_pixel
	                                ? &plan->programs_barrier_safe
	                                : nullptr;
	// KYTY_CONT_KERNEL: a function of the programs, plain-pixel choice, images and targets (the
	// record's) and the sink switch.
	const bool kernel_barrier = m_kernel && m_kernel_record.barrier_sink == BarrierSinkEnabled();
	const bool barrier_safe =
	    kernel_barrier ? m_kernel_record.barrier_safe
	                   : DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
	                                       feedback_aspects, indirect != nullptr, plain_pixel,
	                                       plan_verify ? nullptr : programs_safe);
	if (kernel_verify && kernel_valid && m_kernel_record.barrier_sink == BarrierSinkEnabled()) {
		ContKernel::Check(ContKernel::BarrierSafe, barrier_safe == m_kernel_record.barrier_safe);
	}
	if (plan_verify && programs_safe != nullptr) {
		DrawPrep::CountBindingVerifyCheck();
		if (DrawIsBarrierSafe(stages, state.color_info, state.color_count, state.depth_info,
		                      feedback_aspects, indirect != nullptr, plain_pixel,
		                      programs_safe) != barrier_safe) {
			DrawPrep::ReportBindingMismatch("barrier safety");
		}
	}
	CommitStats::StepMark(CommitStats::Step::BarrierSafe);
	const CommandBuffer::DrawScope draw_scope(buffer, barrier_safe);
	// Emission safe point (KYTY_CP_RECORDER): no native handle obtained during the preparation above
	// is alive; from here the draw's commands go to the recorder (render.h, CommandBuffer::Sink).
	const auto vk_buffer = buffer.EmissionSink();
	SetDrawDebugPhase(buffer, submit_id, draw, emit, draw.IsIndexed() ? 0x100u : 0x200u);
	CommitStats::StepMark(CommitStats::Step::DrawScope);
	// KYTY_DRAW_RUN_PUSH verify: a would-be continuation that would push only its per-draw
	// descriptors must push the previous draw's image and sampler descriptors on the normal path.
	const bool run_verify_push =
	    run_verify_images && m_run.verify_valid && DrawRunPartialPush(buffer, pipeline);
	const bool keep_commit_images = m_run_active && !m_run_texdelta;
	// KYTY_EMIT_SPLIT (descriptors.cpp): a continuation that pushes only its per-draw descriptors is
	// committed as one resolved draw (1), or inline followed by a verify-only one (2).
	const auto split_mode = EmitSplitMode();
	bool       split      = false;
	if (split_mode != 0) [[unlikely]] {
		split = keep_commit_images && !mesh_active && !mesh_indirect && indirect == nullptr && !instance_record &&
		        g_enc_offload.Get() == 0 && !run_verify_push && !PassCensus::LabelsEnabled() &&
		        g_draw_skip_vs.Get() == 0 && g_draw_skip_ps.Get() == 0 && DrawStats::Path() == nullptr &&
		        ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy && DrawRunPartialPush(buffer, pipeline) &&
		        EmitSplitPrepare(buffer, pipeline, stages);
	}
	const bool split_on = split && split_mode == 1;
	// KYTY_ENC_OFFLOAD: the draw's bindings part goes to the emit thread (its side effects ran in
	// OffloadBindings either way); verify encodes inline as well and compares.
	const bool  offload_tried = g_enc_offload.Get() != 0;
	OffloadJob* offload_b     = nullptr;
	if (offload_tried) [[unlikely]] {
		offload_b = OffloadBindings(buffer, pipeline, stages, keep_commit_images, mesh_active, run_verify_push,
		                            vertex_bindings, index_binding);
	}
	const bool  inline_b    = (offload_b == nullptr || offload_b->verify) && !split_on;
	const auto* verify_ring = offload_b != nullptr ? buffer.EncoderRing() : nullptr;
	const auto  verify_from = verify_ring != nullptr ? verify_ring->WritePosition() : 0;
	if (!mesh_active && inline_b) {
		const CommitStats::SubScope vertex_scope(CommitStats::Sub::VertexCommit);
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	CommitStats::StepMark(CommitStats::Step::VertexCommit);
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x300u);
	}
	if (inline_b) {
		KYTY_PROFILER_DETAIL_BLOCK("Draw::CommitBindings");
		const CommitStats::SubScope push_scope(CommitStats::Sub::Push);
		if (offload_tried) {
			CommitBindingsEncode(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages, keep_commit_images);
		} else {
			CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages, keep_commit_images);
		}
	} else if (split_on) {
		// KYTY_EMIT_SPLIT: the commit's cache and barrier work here; its encode is the resolved draw's.
		CommitBindingsSideEffects(buffer, pipeline, stages, keep_commit_images);
	}
	if (run_verify_push) {
		OwnedCounters::Add(DrawRun::GetTotals().partial_pushes, 1);
		DrawRun::CountVerifyCheck();
		if (m_descriptor_scratch.images != m_run.pushed_images) {
			DrawRun::ReportMismatch("pushed image or sampler descriptors",
			                        m_descriptor_scratch.images.size());
		}
	}
	if (run_verify_images && Image::RecordedTransitions() != run_transitions) {
		DrawRun::CountVerifyCheck();
		// Detail: which recorded image's state the transition changed (0: another image's).
		DrawRun::ReportMismatch("image transition", DrawRunImagesChange(false, false, true));
	}
	CommitStats::StepMark(CommitStats::Step::CbTail);
	CommitStats::Mark(CommitStats::Phase::CommitBindings);
	if (!mesh_active && inline_b) CommitIndexBuffer(vk_buffer, index_binding);
	if (offload_b != nullptr && offload_b->verify) {
		OffloadVerifyPart(offload_b, buffer, verify_ring, verify_from);
	}
	CommitStats::StepMark(CommitStats::Step::IndexCommit);

	// KYTY_DRAW_RUN continuation: the dynamic state is a function of the same registers, targets and
	// vertex program as the previous draw's, which recorded it into this command buffer; the
	// shadow shows nothing replaced it since (another pipeline bind or command buffer).
	// A recall records it again: the draws in between left their own.
	const bool run_dynamic = (m_run_active || m_run_verify || texdelta_compare) && !m_run_recall && !m_run_pipeswap &&
	                         DynamicStateShadowEnabled() &&
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
		const CommitStats::SubScope dynamic_scope(CommitStats::Sub::Dynamic);
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
	if (texdelta_compare && run_dynamic) {
		TexDeltaStats::Check(TexDeltaStats::Dynamic, g_dynamic_state_emitted == run_dynamic_emitted);
	}
	if (m_run_verify && run_verify_images && run_dynamic &&
	    g_dynamic_state_emitted != run_dynamic_emitted) {
		DrawRun::CountVerifyCheck();
		DrawRun::ReportMismatch("dynamic state", g_dynamic_state_emitted - run_dynamic_emitted);
	}

	CommitStats::StepMark(CommitStats::Step::DynamicState);
	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x400u);
	}
	// KYTY_GPU_INSTANCE_COUNT: the GPU copies the instance count into the prepared record (outside
	// rendering).
	if (instance_record) {
		const auto [source, source_offset] = m_context.GetBufferCache().ObtainBuffer(
		    emit.gpu_instance_address, sizeof(uint32_t), false);
		EXIT_IF(source == nullptr);
		m_context.GetCommandScheduler().EndRendering();
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eAllCommands,
		                            vk::AccessFlagBits2::eMemoryWrite,
		                            vk::PipelineStageFlagBits2::eCopy,
		                            vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
		                            BarrierOrigin::IndirectArgs);
		const vk::BufferCopy copy {source_offset, instance_record_offset + sizeof(uint32_t),
		                           sizeof(uint32_t)};
		buffer.Sink().copyBuffer(source->Handle(), instance_record, 1, &copy);
		buffer.RequestMemoryBarrier(vk::PipelineStageFlagBits2::eCopy,
		                            vk::AccessFlagBits2::eTransferWrite,
		                            vk::PipelineStageFlagBits2::eDrawIndirect,
		                            vk::AccessFlagBits2::eIndirectCommandRead,
		                            BarrierOrigin::IndirectArgs);
		g_gpu_instance_draws.fetch_add(1, std::memory_order_relaxed);
	}
	if (indirect != nullptr) {
		// Pipeline barriers cannot be recorded inside dynamic rendering. Every buffer write is
		// recorded outside rendering or ends it (ShaderWriteBarrier), so the rendering instance
		// begun right after the previous barrier needs no new one while it remains active.
		const Hazard::Range indirect_host[2] {
		    {std::bit_cast<uint64_t>(static_cast<VkBuffer>(indirect_buffers.args)), indirect_buffers.args_offset,
		     indirect_buffers.args_offset + indirect->ArgsSize(), false},
		    {std::bit_cast<uint64_t>(static_cast<VkBuffer>(indirect_buffers.count)), indirect_buffers.count_offset,
		     indirect_buffers.count_offset + sizeof(uint32_t), false}};
		if (IndirectArgsBarrierNeeded(buffer, indirect, true,
		                              std::span<const Hazard::Range>(indirect_host, indirect_buffers.count ? 2u : 1u))) {
			const CommitStats::SubScope barrier_scope(CommitStats::Sub::Barriers);
			{
				KYTY_GPU_OP_SITE("draw.indirect_args");
				m_context.GetCommandScheduler().EndRendering();
			}
			IndirectArgumentsBarrier(buffer, vk_buffer);
		}
	}
	{
		const CommitStats::SubScope begin_scope(CommitStats::Sub::Begin);
		m_context.GetCommandScheduler().BeginRendering(rendering);
	}
	if (m_depth_feedback.valid) {
		NoteDepthFeedback(buffer);
	}
	if (indirect != nullptr) {
		// A state change above only switched instances; no buffer write came in between.
		m_indirect_barrier_rendering = buffer.ActiveRenderingSerial();
	}
	CommitStats::StepMark(CommitStats::Step::Begin);
	// Vulkan debug labels (--rd / KYTY_TRACE_VK_LABELS): the draw's shaders and game pass.
	const bool draw_label = PassCensus::LabelsEnabled();
	// KYTY_DRAW_SKIP_VS / KYTY_DRAW_SKIP_PS (diagnostic, live, default off): draws with one of these
	// programs record their state but emit no draw command (to see which object a draw renders).
	const bool skip_draw =
	    (g_draw_skip_vs.Get() != 0 && state.vertex_info[0].stage.program != nullptr &&
	     g_draw_skip_vs_set.Contains(state.vertex_info[0].stage.program->shader_hash)) ||
	    (g_draw_skip_ps.Get() != 0 && state.ps_active && state.ps_input_info.stage.program != nullptr &&
	     g_draw_skip_ps_set.Contains(state.ps_input_info.stage.program->shader_hash));
	const bool draw_stats = DrawStats::Path() != nullptr && !skip_draw && state.depth_info.desc.info.data.address != 0 &&
	                        state.depth_info.depth_write_enable;
	// KYTY_ENC_OFFLOAD: the draw part follows the bindings part (BeginRendering settled it if it
	// recorded anything).
	OffloadJob* offload_d = nullptr;
	if (offload_b != nullptr) [[unlikely]] {
		if (draw_label || skip_draw || draw_stats || mesh_active || mesh_indirect || indirect != nullptr ||
		    instance_record) {
			m_offload->misses[static_cast<size_t>(OffloadMiss::DrawDiag)]++;
		} else {
			offload_d            = OffloadAcquire(buffer);
			offload_d->draw_part = true;
			offload_d->layout    = pipeline.pipeline_layout;
			offload_d->pipeline  = pipeline.pipeline;
			const auto flags     = CommandBuffer::EncodeFlags();
			const auto bound     = buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics);
			offload_d->bind      = !(flags.renderer_batch && bound == pipeline.pipeline);
			offload_d->clip_bytes = 0;
			if (const auto planes = state.vertex_info[0].user_clip_planes; planes != 0) {
				uint32_t count = 0;
				for (uint32_t mask = planes; mask != 0; mask >>= 1u) {
					count++;
				}
				offload_d->clip_stages = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex;
				offload_d->clip_bytes  = count * 4u * sizeof(uint32_t);
				offload_d->clip        = buffer.GetRegisters().GetUserClipPlanes();
			}
			offload_d->objprim = state.ps_active && state.ps_input_info.HasObjprimOutput();
			if (offload_d->objprim) {
				vk::ShaderStageFlags objprim_stages = vk::ShaderStageFlagBits::eFragment;
				for (const auto& stage: vertex_stages) {
					objprim_stages |= NativeShaderStage(stage.logical_stage);
				}
				offload_d->objprim_stages = objprim_stages;
				offload_d->objprim_offset = uint32_t {state.ps_input_info.objprim_push_dword} * 4u;
				offload_d->object_id      = buffer.GetUserConfig().GetObjectId();
			}
			if (m_segment_splice != nullptr && draw.IsIndexed()) {
				const auto& spliced               = *m_segment_splice;
				offload_d->direct                 = {};
				offload_d->direct.indexed         = true;
				offload_d->direct.count           = spliced.index_count;
				offload_d->direct.instance_count  = spliced.instance_count;
				offload_d->direct.vertex_offset   = spliced.vertex_offset;
				offload_d->direct.first_instance  = spliced.first_instance;
				if (!offload_d->verify) {
					DrawPrep::SegmentCounters().spliced.fetch_add(1, std::memory_order_relaxed);
				}
			} else {
				offload_d->direct = MakeDirectDraw(ucfg, draw, emit);
			}
			if (!offload_d->verify) {
				// What the inline bind leaves behind, decided here (the commit thread owns these).
				if (CommitStats::Enabled() && offload_d->bind) [[unlikely]] {
					CommitStats::AddItems(CommitStats::Item::PipelineNew, 1);
				}
				(void)buffer.EncodeShadowUnsettled().TakePipelineBind(flags, vk::PipelineBindPoint::eGraphics,
				                                                      pipeline.pipeline);
				m_dynamic_state.command  = vk_buffer.Identity();
				m_dynamic_state.pipeline = pipeline.pipeline;
				if (state.color_count == 0) {
					m_dynamic_state.color_write_valid = false;
				}
				m_offload->draws++;
			}
			OffloadPost(offload_d, buffer);
		}
	}
	const bool  inline_d      = (offload_d == nullptr || offload_d->verify) && !split_on;
	const auto* verify_ring_d = offload_d != nullptr ? buffer.EncoderRing() : nullptr;
	const auto  verify_from_d = verify_ring_d != nullptr ? verify_ring_d->WritePosition() : 0;
	if (inline_d) {
	{
		const CommitStats::SubScope bind_scope(CommitStats::Sub::BindPipeline);
		if (CommitStats::Enabled() &&
		    buffer.BoundPipeline(vk::PipelineBindPoint::eGraphics) != pipeline.pipeline) [[unlikely]] {
			CommitStats::AddItems(CommitStats::Item::PipelineNew, 1);
		}
		buffer.BindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		// The shadow now describes this command buffer with this pipeline bound. A pipeline
		// without color attachments declares no dynamic color-write enable, so binding it
		// invalidated that state for later pipelines.
		m_dynamic_state.command  = vk_buffer.Identity();
		m_dynamic_state.pipeline = pipeline.pipeline;
		if (state.color_count == 0) {
			m_dynamic_state.color_write_valid = false;
		}
	}
	CommitStats::StepMark(CommitStats::Step::BindPipeline);
	CommitStats::SubScope draw_cmd_scope(CommitStats::Sub::DrawCmd);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, emit, 0x500u);
	}
	// Mesh programs with split_groups read a seventh draw dword (MeshFirstGroupDword).
	const uint32_t mesh_draw_dwords =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwords(
	                      state.vertex_info[0].mesh.split_groups != 0)
	                : 0u;
	// PA_CL_UCP planes (upstream feed075a9): four float dwords per plane up to the highest enabled.
	if (const auto planes = state.vertex_info[0].user_clip_planes; planes != 0) {
		uint32_t count = 0;
		for (uint32_t mask = planes; mask != 0; mask >>= 1u) {
			count++;
		}
		vk_buffer.pushConstants(
		    pipeline.pipeline_layout,
		    vk::ShaderStageFlagBits::eFragment |
		        (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT : vk::ShaderStageFlagBits::eVertex),
		    ShaderRecompiler::IR::PushData::UserClipPlaneDword * sizeof(uint32_t),
		    count * 4u * sizeof(uint32_t), buffer.GetRegisters().GetUserClipPlanes().data());
	}
	// KYTY_OBJPRIM_ID (based on Erlou93/KytyPS5-Yotei): this draw's VGT_OBJECT_ID, after
	// CommitBindings (which pushes the whole block). The stage flags are the layout's push range
	// (fragment plus every vertex stage of the pipeline). The raw push bypasses the shadow, which
	// is then invalidated: the next whole-block push must not be skipped as unchanged.
	if (state.ps_active && state.ps_input_info.HasObjprimOutput()) {
		vk::ShaderStageFlags objprim_stages = vk::ShaderStageFlagBits::eFragment;
		for (const auto& stage: vertex_stages) {
			objprim_stages |= NativeShaderStage(stage.logical_stage);
		}
		const uint32_t object_id = buffer.GetUserConfig().GetObjectId();
		vk_buffer.pushConstants(pipeline.pipeline_layout, objprim_stages,
		                        uint32_t {state.ps_input_info.objprim_push_dword} * 4u,
		                        sizeof(object_id), &object_id);
		buffer.InvalidatePushConstants();
	}
	if (draw_label) {
		char text[256];
		PassCensus::LabelText(text, sizeof(text), 'd',
		                      state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
		                      state.ps_active && state.ps_input_info.stage.program
		                          ? state.ps_input_info.stage.program->shader_hash
		                          : 0);
		vk_buffer.insertDebugLabel(text);
	}
	uint32_t stats_query = UINT32_MAX;
	if (draw_stats) [[unlikely]] {
		DrawStats::Pending p;
		p.frame          = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		p.flags          = (indirect != nullptr ? 1u : 0u) | (mesh_active ? 2u : 0u) | (mesh_indirect ? 4u : 0u) |
		                   (state.ps_active ? 8u : 0u) | (state.color_count << 8u);
		p.index_count    = draw.index_count;
		p.instance_count = draw.instance_count;
		p.vs             = state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0;
		p.ps             = state.ps_active && state.ps_input_info.stage.program ? state.ps_input_info.stage.program->shader_hash : 0;
		p.depth          = state.depth_info.desc.info.data.address;
		p.color0         = state.color_count != 0 ? state.color_info[0].desc.info.data.address : 0;
		stats_query      = DrawStats::Begin(m_context, vk_buffer, p);
	}
	if (skip_draw) [[unlikely]] {
	} else if (mesh_indirect) {
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
	} else if (instance_record) {
		if (draw.IsIndexed()) {
			vk_buffer.drawIndexedIndirect(instance_record, instance_record_offset, 1, 20u);
		} else {
			vk_buffer.drawIndirect(instance_record, instance_record_offset, 1, 16u);
		}
	} else if (m_segment_splice != nullptr && draw.IsIndexed()) {
		// KYTY_SPEC_SEGMENTS=1: the draw command the preparing thread encoded (a plain primitive
		// type and an indexed draw, which the encode required; the same command EmitDrawPrimitives
		// records).
		const auto& spliced = *m_segment_splice;
		auto&       counters = DrawPrep::SegmentCounters();
		const auto  n        = counters.spliced.fetch_add(1, std::memory_order_relaxed);
		const bool  timed    = (n & 63u) == 0;
		const auto  start    = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		vk_buffer.drawIndexed(spliced.index_count, spliced.instance_count, 0, spliced.vertex_offset,
		                      spliced.first_instance);
		if (timed) {
			counters.splice_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                 std::chrono::steady_clock::now() - start)
			                                                       .count()),
			                             std::memory_order_relaxed);
			counters.splice_timed.fetch_add(1, std::memory_order_relaxed);
		}
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	DrawStats::End(vk_buffer, stats_query);
	draw_cmd_scope.Stop();
	}
	if (offload_d != nullptr && offload_d->verify) {
		OffloadVerifyPart(offload_d, buffer, verify_ring_d, verify_from_d);
	}
	if (split) [[unlikely]] {
		// KYTY_EMIT_SPLIT: the resolved draw (on: the draw's commands; verify: after the inline ones).
		using P = CommandStream::ResolvedDrawPacket;
		P                                 packet;
		CommandStream::ResolvedDrawArrays arrays;
		if (vertex_bindings.count != 0) {
			for (uint32_t i = 0; i < vertex_bindings.count; i++) {
				EXIT_IF(vertex_bindings.buffers[i] == nullptr);
			}
			packet.flags |= P::FlagVertex;
			packet.vertex_count   = vertex_bindings.count;
			arrays.vertex_buffers = vertex_bindings.buffers.data();
			arrays.vertex_offsets = vertex_bindings.offsets.data();
			arrays.vertex_sizes   = vertex_bindings.sizes.data();
		}
		if (index_binding.buffer != nullptr) {
			packet.flags |= P::FlagIndex;
			packet.index_buffer = index_binding.buffer;
			packet.index_offset = index_binding.offset;
			packet.index_type   = static_cast<uint32_t>(index_binding.type);
		}
		if (const auto planes = state.vertex_info[0].user_clip_planes; planes != 0) {
			uint32_t count = 0;
			for (uint32_t mask = planes; mask != 0; mask >>= 1u) {
				count++;
			}
			packet.clip_offset = ShaderRecompiler::IR::PushData::UserClipPlaneDword * sizeof(uint32_t);
			packet.clip_bytes  = count * 4u * sizeof(uint32_t);
			packet.clip_stages = static_cast<uint32_t>(
			    static_cast<VkShaderStageFlags>(vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex));
			arrays.clip = reinterpret_cast<const uint32_t*>(buffer.GetRegisters().GetUserClipPlanes().data());
		}
		if (state.ps_active && state.ps_input_info.HasObjprimOutput()) {
			vk::ShaderStageFlags objprim_stages = vk::ShaderStageFlagBits::eFragment;
			for (const auto& stage: vertex_stages) {
				objprim_stages |= NativeShaderStage(stage.logical_stage);
			}
			packet.flags |= P::FlagObjprim;
			packet.objprim_stages = static_cast<uint32_t>(static_cast<VkShaderStageFlags>(objprim_stages));
			packet.objprim_offset = uint32_t {state.ps_input_info.objprim_push_dword} * 4u;
			packet.object_id      = buffer.GetUserConfig().GetObjectId();
		}
		DrawEncode::DirectDraw direct;
		if (m_segment_splice != nullptr && draw.IsIndexed()) {
			const auto& spliced   = *m_segment_splice;
			direct.indexed        = true;
			direct.count          = spliced.index_count;
			direct.instance_count = spliced.instance_count;
			direct.vertex_offset  = spliced.vertex_offset;
			direct.first_instance = spliced.first_instance;
			if (split_on) {
				DrawPrep::SegmentCounters().spliced.fetch_add(1, std::memory_order_relaxed);
			}
		} else {
			direct = MakeDirectDraw(ucfg, draw, emit);
		}
		EXIT_IF(direct.quad_list);
		packet.flags |= direct.indexed ? P::FlagIndexed : 0u;
		packet.count          = direct.count;
		packet.instance_count = direct.instance_count;
		packet.vertex_offset  = direct.vertex_offset;
		packet.first_vertex   = direct.first_vertex;
		packet.first_instance = direct.first_instance;
		EmitSplitWrite(buffer, pipeline, stages, packet, arrays, !split_on);
		if (split_on) {
			// What the inline bind leaves behind (the commit thread owns these).
			m_dynamic_state.command  = vk_buffer.Identity();
			m_dynamic_state.pipeline = pipeline.pipeline;
			if (state.color_count == 0) {
				m_dynamic_state.color_write_valid = false;
			}
		}
	}
	// KYTY_DRAW_PEEK (above): the draw's per-draw block and indirect arguments, after it.
	if (const auto peek_hash = DrawPeekRequest()[0];
	    peek_hash != 0 && state.vertex_info[0].stage.program != nullptr &&
	    state.vertex_info[0].stage.program->shader_hash == peek_hash) [[unlikely]] {
		buffer.EndRendering();
		m_run_active = false;
		DrawPeek(m_context, buffer, draw, state, indirect);
	}
	// KYTY_DRAW_VERIFY_VSB (above): the draw's VS buffers on the GPU vs the guest's bytes.
	if (const auto verify_hash = VerifyVsb().hash;
	    verify_hash != 0 && state.vertex_info[0].stage.program != nullptr &&
	    state.vertex_info[0].stage.program->shader_hash == verify_hash) [[unlikely]] {
		buffer.EndRendering();
		m_run_active = false;
		VerifyVsbDraw(m_context, buffer, draw, state);
	}
	// KYTY_PASS_DUMP (renderCompute.cpp): the draw and the targets it wrote.
	if (PassDump::Capturing()) {
		PassDump::NoteOp('d', state.ps_active && state.ps_input_info.stage.program
		                          ? state.ps_input_info.stage.program->shader_hash
		                      : state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash
		                                                           : 0);
		for (uint32_t i = 0; i < state.color_count; i++) {
			const auto& target = state.color_info[i];
			PassDump::NoteOutput(target.desc.info.data.address, target.desc.info.extent.width,
			                     target.desc.info.extent.height, 0);
		}
		const auto& depth = state.depth_info;
		if (depth.image_id && depth.desc.info.data.address != 0) {
			const auto aspects = depth.AttachmentWriteAspects();
			if (aspects & vk::ImageAspectFlagBits::eDepth) {
				PassDump::NoteOutput(depth.desc.info.data.address, depth.desc.info.extent.width,
				                     depth.desc.info.extent.height, 1);
			}
			if (aspects & vk::ImageAspectFlagBits::eStencil) {
				PassDump::NoteOutput(depth.desc.info.data.address, depth.desc.info.extent.width,
				                     depth.desc.info.extent.height, 2);
			}
		}
	}
	// KYTY_NAN_SCAN (renderCompute.cpp): the draw's colour targets, after it (ends the rendering).
	if (const auto op = NanScanTake(); op != 0) {
		buffer.EndRendering();
		m_run_active = false;
		for (uint32_t i = 0; i < state.color_count; i++) {
			const auto& target = state.color_info[i];
			(void)m_context.GetTextureCache().DebugNanScan(
			    target.desc.info.data.address, target.desc.info.extent.width, target.desc.info.extent.height,
			    fmt::format("op {} DRAW vs 0x{:016x} ps 0x{:016x} target {}", op,
			                state.vertex_info[0].stage.program ? state.vertex_info[0].stage.program->shader_hash : 0,
			                state.ps_active && state.ps_input_info.stage.program
			                    ? state.ps_input_info.stage.program->shader_hash
			                    : 0,
			                i));
		}
	}
	// KYTY_GRAPHICS_BDA_WRITES: the pages a stage wrote through BDA, settled as after a dispatch
	// (the settle ends rendering, records the written-page pass, submits and waits).
	for (auto* stage: stages) {
		if (stage->runtime->program->info.bda_writes) {
			m_context.GetBufferCache().SettleBdaWrites(stage->runtime->program->shader_hash);
			break;
		}
	}
	CommitStats::StepMark(CommitStats::Step::DrawCmd);
	static constexpr auto occlusion_draw_rows_init = [] {
		const char* value = std::getenv("KYTY_HANG_TRACE_OCCLUSION_DRAWS");
		return value != nullptr && value[0] == '1';
	};
	static constinit Common::StaticOnce<std::remove_cvref_t<decltype(occlusion_draw_rows_init())>> occlusion_draw_rows_once;
	const bool& occlusion_draw_rows = occlusion_draw_rows_once.Get(occlusion_draw_rows_init);
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
		OwnedCounters::Add(DrawPrep::GetBindingTotals().statics_used, 1);
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
	// KYTY_PM4_BARRIERS=guest (render.h): the guest's own sync points order the draw's writes.
	// A later indirect draw of this instance still records its own argument barrier.
	if (shader_write_stages && GuestSyncBarriersEnabled()) {
		m_indirect_barrier_rendering = 0;
	} else if (shader_write_stages) {
		const CommitStats::SubScope barrier_scope(CommitStats::Sub::Barriers);
		if (DrawWriteSinkEnabled()) {
			// Queued without ending the instance (render.h, KYTY_DRAW_WRITE_SINK): a later draw of
			// this instance may pass it, anything else records it first.
			ShaderWriteBarrier(buffer, shader_write_stages);
			// An indirect draw can no longer rely on "no buffer write since the argument barrier
			// while this instance is active": it must end the instance and record a new one.
			m_indirect_barrier_rendering = 0;
			m_indirect_write_rendering   = buffer.ActiveRenderingSerial();
			if (g_indirect_args_ranges.On()) {
				NoteInstanceWrites(buffer.ActiveRenderingSerial(),
				                   std::span<PreparedBindings* const>(descriptor_stages.data(), stage_count));
			} else {
				m_haz_write_imprecise = true; // the switch was off for this write: not recorded
				m_haz_write_instance  = buffer.ActiveRenderingSerial();
			}
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
	CommitStats::StepMark(CommitStats::Step::Occlusion);
	// KYTY_DRAW_RUN: whether the next draw may continue this one (drawPrep/drawRun.h), and in verify
	// mode what this draw's normal path computed against what a continuation would have reused.
	// KYTY_CONT_KERNEL: a kernel continuation's eligibility is its record's but for the terms that
	// can change after the key matched (ContinuationKernelEntry checks the switches' terms).
	if (rendering_own.has_value() && rendering_ref != &*rendering_own) {
		// KYTY_BK_LEAN verify: the record's rendering state did not change while the draw used it.
		auto& bk = DrawPrep::BkCounters();
		bk.Add(bk.lean_checks);
		if (!(*rendering_own == *rendering_ref)) {
			DrawPrep::Bookkeep::ReportMismatch(bk.lean_mismatches, "lean", "record rendering state changed");
		}
	}
	const bool kernel_eligible = (m_kernel || kernel_verify) && buffer.ActiveRenderingSerial() != 0 &&
	                             RenderStateFastEnabled(RenderStatePart::Reset) &&
	                             !graphics_debug_dump_enabled();
	if (m_kernel && DrawRun::Enabled()) {
		const CommitStats::SubScope record_scope(CommitStats::Sub::RunRecord);
		m_seed_why = 1; // KYTY_IMG_STATS: the kernel path seeds nothing
		if (kernel_eligible) {
			DrawRunRecordDraw(buffer, state, m_run_slice_offset, rendering, stages,
			                  bounded ? &written : nullptr, pipeline);
		}
	} else if (DrawRun::Enabled()) {
		const CommitStats::SubScope record_scope(CommitStats::Sub::RunRecord);
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
		// KYTY_DRAW_RUN_WRITES: buffer writes (shader_write_stages only ever holds those) may seed a run.
		const bool writes_ok = DrawRun::WritesEnabled();
		bool eligible = m_in_engine_commit && m_prepared_validated && m_run_key != 0 &&
		                !mesh_active &&
		                (indirect == nullptr || DrawRun::IndirectRunsEnabled()) &&
		                vertex_stages.size() == 1 &&
		                (!shader_write_stages || writes_ok) && !feedback_aspects &&
		                buffer.ActiveRenderingSerial() != 0 &&
		                RenderStateFastEnabled(RenderStatePart::Reset) &&
		                !graphics_debug_dump_enabled() &&
		                !rendering.depth_stencil_attachment.depth_clear &&
		                !rendering.depth_stencil_attachment.stencil_clear;
		for (uint32_t i = 0; eligible && i < rendering.num_color_attachments; i++) {
			eligible = !rendering.color_attachments[i].is_clear;
		}
		// KYTY_DRAW_RUN_CONT_ELIGIBLE (drawRun.h): a continuation's per-stage part is its record's.
		const auto cont_mode  = DrawRun::ContEligibleMode();
		const bool cont_known = cont_mode != 0 && m_run_active && !m_run_pipeswap && !m_run_texdelta &&
		                        m_run.record_writes_ok == writes_ok;
		const auto stages_eligible = [&] {
			bool ok = true;
			for (const auto* stage: stages) {
				ok = ok && stage->gds.buffer == nullptr &&
				     DrawRunProgramEligible(*stage->runtime->program, writes_ok,
				                            g_draw_run_fault_buffer.On());
				for (const auto& binding: stage->images) {
					// Storage bindings are transitioned with write access (a barrier every draw).
					ok = ok && !is_attachment(binding.image_id) &&
					     binding.desc.type != TextureCache::BindingType::Storage;
				}
			}
			return ok;
		};
		const auto stages_write = [&] {
			return writes_ok && std::any_of(stages.begin(), stages.end(), [](const auto* stage) {
				       return !DrawRunProgramEligible(*stage->runtime->program, false,
				                                      g_draw_run_fault_buffer.On());
			       });
		};
		if (cont_known && cont_mode == 1) {
			// The record's: eligible (it exists), and its writes flag below.
		} else {
			eligible = eligible && stages_eligible();
		}
		if (cont_known && cont_mode >= 2 && eligible) {
			DrawRun::CountVerifyCheck();
			if (!stages_eligible() || stages_write() != m_run.writes) {
				DrawRun::ReportMismatch("continuation's record eligibility");
				if (cont_mode == 3) {
					EXIT("DrawRunVerify: a continuation's record eligibility differs\n");
				}
			}
		}
		if (DrawRun::KeyDiagEnabled() || CommitStats::Enabled()) [[unlikely]] {
			using DrawRun::KeyDiag::Inelig;
			auto reason = Inelig::None;
			if (!eligible) {
				bool color_clear = false;
				for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
					color_clear = color_clear || rendering.color_attachments[i].is_clear;
				}
				reason = !m_in_engine_commit                                          ? Inelig::NotEngine
				         : !m_prepared_validated                                      ? Inelig::NotValidated
				         : m_run_key == 0                                             ? Inelig::NoKey
				         : mesh_active                                                ? Inelig::Mesh
				         : (indirect != nullptr && !DrawRun::IndirectRunsEnabled())   ? Inelig::Indirect
				         : vertex_stages.size() != 1                                  ? Inelig::VertexStages
				         : (shader_write_stages && !writes_ok)                        ? Inelig::Writes
				         : feedback_aspects                                           ? Inelig::Feedback
				         : buffer.ActiveRenderingSerial() == 0                        ? Inelig::NoRendering
				         : !RenderStateFastEnabled(RenderStatePart::Reset)            ? Inelig::RenderState
				         : graphics_debug_dump_enabled()                              ? Inelig::DebugDump
				         : rendering.depth_stencil_attachment.depth_clear             ? Inelig::DepthClear
				         : rendering.depth_stencil_attachment.stencil_clear           ? Inelig::StencilClear
				         : color_clear                                                ? Inelig::ColorClear
				                                                                      : Inelig::Count;
				if (reason == Inelig::Count) {
					for (const auto* stage: stages) {
						if (reason != Inelig::Count) {
							break;
						}
						if (stage->gds.buffer != nullptr) {
							reason = Inelig::Gds;
						} else if (!DrawRunProgramEligible(*stage->runtime->program, writes_ok,
						                                   g_draw_run_fault_buffer.On())) {
							reason = Inelig::Program;
						} else {
							for (const auto& binding: stage->images) {
								if (is_attachment(binding.image_id)) {
									reason = Inelig::TexAttachment;
									break;
								}
								if (binding.desc.type == TextureCache::BindingType::Storage) {
									reason = Inelig::Storage;
									break;
								}
							}
						}
					}
					if (reason == Inelig::Count) {
						reason = Inelig::NotReached; // not found: shown as "notreached"
					}
				}
				if (DrawRun::KeyDiagEnabled()) {
					DrawRun::KeyDiag::CountIneligible(reason);
				}
			}
			m_key_diag.reason = reason;
			m_stats_inelig    = static_cast<uint8_t>(reason); // KYTY_CP_COMMIT_STATS
		}
		if (eligible) {
			OwnedCounters::Add(DrawRun::GetTotals().eligible, 1);
			const bool program_writes =
			    cont_known && cont_mode == 1 ? m_run.writes : stages_write();
			if (program_writes) {
				OwnedCounters::Add(DrawRun::GetTotals().write_records, 1);
			}
			const bool continued = m_run_active;
			DrawRunRecordDraw(buffer, state, m_run_slice_offset, rendering, stages,
			                  bounded ? &written : nullptr, pipeline);
			// A continuation keeps its record's programs (DrawRunCandidate), so the flag holds.
			m_run.writes = program_writes;
			if (!continued) {
				m_run.record_writes_ok = writes_ok;
			}
		}
		if (eligible) {
			m_seed_why = 2;
		} else {
			// KYTY_IMG_ACQSEED: an ordinary acquisition of a draw that may not seed a run seeds the next
			// draw's acquisition reuse (m_seed_why: which condition refused, for KYTY_IMG_STATS).
			bool clears = rendering.depth_stencil_attachment.depth_clear ||
			              rendering.depth_stencil_attachment.stencil_clear;
			for (uint32_t i = 0; i < rendering.num_color_attachments; i++) {
				clears = clears || rendering.color_attachments[i].is_clear;
			}
			// What the acquisition needs of the seeding draw (6.4 of the imgside design): its writes cannot
			// reach the attachments unseen. Buffer writes are checked against the attachments' ranges
			// (DrawRunSeedAcquire), storage images by the texture-over-attachment check; address writes
			// (unbounded) and GDS refuse; the fault buffer (OR-ed bits read after the submission) does not.
			bool program_ok  = true;
			bool textures_ok = true;
			for (const auto* stage: stages) {
				program_ok = program_ok && stage->gds.buffer == nullptr && !stage->runtime->program->has_address_writes;
				for (const auto& binding: stage->runtime->program->bindings.descriptors) {
					program_ok = program_ok && binding.kind != ShaderRecompiler::IR::DescriptorBindingKind::Gds;
				}
				for (const auto& binding: stage->images) {
					textures_ok = textures_ok && !is_attachment(binding.image_id);
				}
			}
			m_seed_why = ImgSide::AcqSeedMode() == ImgSide::Mode::Off ? 3
			             : !m_in_engine_commit                         ? 4
			             : feedback_aspects                            ? 5
			             : buffer.ActiveRenderingSerial() == 0 || !RenderStateFastEnabled(RenderStatePart::Reset) ||
			                       graphics_debug_dump_enabled()
			                 ? 6
			             : clears       ? 8
			             : !program_ok  ? 9
			             : !textures_ok ? 10
			                            : 0;
			if (m_seed_why == 0) {
				m_seed_why = DrawRunSeedAcquire(buffer, state, rendering, stages, bounded ? &written : nullptr);
			}
			if (m_seed_why != 13 && m_seed_why != 3) {
				ImgSide::Bump(m_seed_why == 10  ? ImgSide::S::SeedRefAttachment
				              : m_seed_why == 9 ? ImgSide::S::SeedRefProgram
				              : m_seed_why == 7 ? ImgSide::S::SeedRefWrites
				                                : ImgSide::S::SeedRefOther);
			}
			// KYTY_IMG_ACQFB: an acquisition (full, or itself an ACQFB reuse) that ACQSEED did not take
			// seeds the next draw's reuse; the seeding draw's writes cannot reach the attachments unseen
			// (no clears, no address writes or GDS, buffer writes checked in AcqFbRecord).
			if (m_seed_why != 13 && m_acqfb_this && ImgSide::AcqFbMode() != ImgSide::Mode::Off &&
			    m_in_engine_commit && !clears && program_ok && buffer.ActiveRenderingSerial() != 0 &&
			    RenderStateFastEnabled(RenderStatePart::Reset) && !graphics_debug_dump_enabled()) {
				AcqFbRecord(buffer, state, rendering, feedback_aspects, stages, bounded ? &written : nullptr);
			}
		}
		if (kernel_verify) {
			// A kernel continuation would have recorded exactly when this one did.
			ContKernel::Check(ContKernel::Eligible, kernel_eligible == (eligible && m_run.valid));
		}
		if (m_run.valid && g_cont_kernel.Get() != 0) {
			// KYTY_CONT_KERNEL: what a kernel continuation of this record reuses.
			const auto& hw        = buffer.GetRegisters();
			const auto  mode      = hw.GetColorControl().mode;
			auto&       record    = m_kernel_record;
			record.pipeline          = &pipeline;
			record.generation        = pipeline_cache.PipelineGeneration();
			record.key               = m_run.key;
			record.serial            = m_run.rendering_serial;
			record.topology          = topology;
			record.element_size      = draw.IsIndexed() && index_source.host_data == nullptr &&
			                                   (index_source.guest_element_size == 2 ||
			                                    index_source.guest_element_size == 4)
			                               ? index_source.guest_element_size
			                               : 0u;
			record.primitive_restart = primitive_restart_enable;
			record.plain_pixel       = plain_pixel;
			record.barrier_safe      = barrier_safe;
			record.barrier_sink      = BarrierSinkEnabled();
			record.target_ops_quiet  = (mode == 0 || mode == 1) && !DepthStencilCopyApplies(hw);
		} else {
			m_kernel_record.pipeline = nullptr; // switched off: no record from before switching on
		}
		if (DrawRun::KeyDiagEnabled()) [[unlikely]] {
			DrawRunKeyDiagResolved(buffer, stages, pipeline, rendering, eligible && m_run.valid);
		}
	}
	CommitStats::StepMark(CommitStats::Step::RunRecord);
	if (CommitStats::Enabled()) [[unlikely]] {
		CommitStats::Mark(CommitStats::Phase::Emit);
		uint64_t textures = 0;
		uint64_t samplers = 0;
		for (const auto* stage: stages) {
			textures += stage->images.size();
			samplers += stage->samplers.size();
		}
		CommitStats::AddItems(CommitStats::Item::Textures, textures);
		CommitStats::AddItems(CommitStats::Item::Samplers, samplers);
		CommitStats::AddItems(CommitStats::Item::Transitions,
		                      Image::RecordedTransitions() - stats_transitions);
		auto shape = MakeCommitShape(buffer, stages, pipeline, rendering, m_dynamic_state,
		                             vertex_bindings, index_binding, draw, emit, index_source,
		                             shader_write_stages);
		shape.run_continued = m_run_active;
		shape.kernel        = m_kernel ? CommitStats::KernelOutcome::Taken : m_stats_kernel;
		shape.candidate     = m_stats_candidate;
		shape.prev_inelig   = m_stats_prev_inelig;
		CommitStats::NoteRecorded(shape);
		// The hashing above is the diagnostic's own cost: not counted in any phase.
		CommitStats::Skip();
	}
}

// KYTY_GPU_INSTANCE_COUNT: the inherited instance count read on the CPU after all (mesh pipelines,
// quad lists), synchronizing its GPU write; 0 when it cannot be read.
static uint32_t ReadInheritedInstances(RenderContext& context, uint64_t address) {
	uint32_t instances = 0;
	if (!context.SynchronizeGpuBackingForRead(address, sizeof(instances)) ||
	    !LibKernel::Memory::TryReadBacking(address, &instances, sizeof(instances))) {
		return 0;
	}
	return instances;
}

// KYTY_GPU_INSTANCE_COUNT: whether a draw inheriting its instance count reads it on the GPU (vertex
// pipelines, primitive types drawn with one command, not a depth-to-colour copy).
static bool GpuInstanceCountApplies(CommandBuffer& buffer, const HW::UserConfig& ucfg,
                                    const DrawRenderState& state) {
	return state.vertex_info[0].stage.program->stage != ShaderType::Mesh &&
	       ucfg.GetPrimType() != Prospero::PrimitiveType::kQuadListLegacy &&
	       !buffer.GetRegisters().GetRenderControl().copy_depth_to_color;
}

// KYTY_CONT_KERNEL (above ExecutePreparedDraw): whether the committed DrawIndex may take the kernel.
bool RenderExecutor::ContinuationKernelEntry(const DrawIndexArgs& args) const {
	const auto& record = m_kernel_record;
	const auto  type   = static_cast<Prospero::IndexType>(args.index_type_and_size);
	const uint32_t element_size = type == Prospero::IndexType::kIndex16   ? 2u
	                              : type == Prospero::IndexType::kIndex32 ? 4u
	                                                                       : 0u;
	return m_in_engine_commit && m_run_prev_valid && m_run_key != 0 && m_run_key == m_run.key &&
	       record.pipeline != nullptr && record.key == m_run.key && record.serial == m_run.rendering_serial &&
	       record.target_ops_quiet && element_size != 0 && record.element_size == element_size &&
	       args.offset_source == DrawOffsetSource::DrawState && args.gpu_instance_address == 0 &&
	       DrawRun::Enabled() && DrawRun::GetMode() == DrawRun::Mode::On && DrawRun::ContEligibleMode() == 1 &&
	       !DrawRun::KeyDiagEnabled() && DrawRun::WritesEnabled() == m_run.record_writes_ok &&
	       g_cube_every.Get() <= 1 && !graphics_debug_dump_enabled() && !ListResetEnabled();
}

// KYTY_SPEC_SEGMENTS (drawPrep/segment.h): every 64th validation (and its splice) is timed.
const DrawPrep::Segment::EncodedDraw* RenderExecutor::SegmentValidate(const DrawIndexArgs& args) {
	(void)args;
	m_segment_validated = false;
	if (m_segment == nullptr) {
		return nullptr;
	}
	const auto mode     = DrawPrep::SpecSegmentsMode();
	auto&      counters = DrawPrep::SegmentCounters();
	if (mode == DrawPrep::Segment::Mode::Off) {
		return nullptr;
	}
	const auto taken = counters.kernel_taken.fetch_add(1, std::memory_order_relaxed);
	const bool timed = (taken & 63u) == 0;
	const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	const auto miss  = DrawPrep::Segment::Validate(*m_segment, m_run_key, m_kernel_record.element_size);
	if (timed) {
		counters.validate_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                   std::chrono::steady_clock::now() - start)
		                                                         .count()),
		                               std::memory_order_relaxed);
		counters.timed.fetch_add(1, std::memory_order_relaxed);
	}
	if (miss != DrawPrep::Segment::Miss::None) {
		counters.misses[static_cast<size_t>(miss)].fetch_add(1, std::memory_order_relaxed);
		return nullptr;
	}
	counters.validated.fetch_add(1, std::memory_order_relaxed);
	m_segment_validated = true;
	return mode == DrawPrep::Segment::Mode::On ? m_segment : nullptr;
}

void RenderExecutor::SegmentVerify(const DrawIndexArgs& args, const DrawEmitInfo& emit,
                                   const DrawIndexBufferSource& index_source) {
	if (DrawPrep::SpecSegmentsMode() != DrawPrep::Segment::Mode::Verify || m_segment == nullptr) {
		return;
	}
	auto&                          counters = DrawPrep::SegmentCounters();
	DrawPrep::Segment::EncodedDraw serial;
	serial.miss           = DrawPrep::Segment::Miss::None;
	serial.index_count    = args.index_count;
	serial.instance_count = args.instance_count;
	serial.vertex_offset  = emit.vertex_offset;
	serial.first_instance = emit.first_instance;
	serial.index_address  = index_source.address;
	serial.index_bytes    = index_source.size;
	const auto mask       = DrawPrep::Segment::Compare(*m_segment, serial);
	const auto draws      = counters.verify_draws.fetch_add(1, std::memory_order_relaxed);
	if (mask == 0) {
		return;
	}
	uint64_t reported = 0;
	for (size_t i = 0; i < DrawPrep::Segment::kFieldNames.size(); i++) {
		if ((mask & (1u << i)) != 0) {
			reported += counters.mismatches[i].fetch_add(1, std::memory_order_relaxed);
		}
	}
	if (reported < 16) {
		std::printf("SpecSegments verify: draw %" PRIu64 " mismatch mask 0x%x (encoded vo %d fi %u ic %u; serial vo %d fi %u ic %u)\n",
		            draws, mask, m_segment->vertex_offset, m_segment->first_instance, m_segment->index_count,
		            serial.vertex_offset, serial.first_instance, serial.index_count);
		std::fflush(stdout);
	}
}

// KYTY_CONT_KERNEL=1: the continuation kernel's front. The preparation is taken, validated (with the
// record's pixel-shader activity: Validate compares it with the preparation's own) and applied; the
// run candidate check decides. A draw that is not a continuation gets its preparation back and runs
// the normal path from the start. False then.
bool RenderExecutor::ContinuationKernel(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args) {
	auto&       ucfg           = buffer.GetUserConfig();
	auto&       state          = *m_draw_state;
	auto* const saved_prepared = m_prepared_draw;
	auto&       counters       = ContKernel::g_counters;
	if (saved_prepared == nullptr || !saved_prepared->ok) {
		counters.declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
	                   args.first_instance};
	state.ps_active = m_run.ps_active;
	g_apply_keyed_skip = static_cast<int>(g_cont_kernel_inplace.Get());
	// KYTY_BK_ADOPT (drawPrep/bookkeep.h): the stage preparations by pointer. Not with the verify
	// paths that read `stage_preps` or compare the swap itself.
	const auto adopt_mode = DrawPrep::BkAdoptMode();
	auto&      bk         = DrawPrep::BkCounters();
	if (adopt_mode != DrawPrep::Bookkeep::Mode::Off) {
		g_apply_adopt = DrawPrep::VerifyMode() == 0 && RenderStateVerifyMode() == 0;
		if (!g_apply_adopt) {
			bk.Fall(DrawPrep::Bookkeep::Fallback::AdoptVerifyPaths);
		}
	}
	RefreshShaders(buffer, draw, state);
	g_apply_adopt      = false;
	g_apply_keyed_skip = 0;
	if (!m_prepared_validated || state.programs.translation_pending || ContKernel::Bindless(state) ||
	    !DrawRunCandidate(buffer, state, args.render_target_slice_offset) || m_run_pipeswap) {
		// ApplyPreparedDraw swapped the stage preps: swapping again restores both sides. An adopted
		// preparation was not moved: the runtimes only point back at the draw state's.
		if (state.adopted != nullptr) {
			state.Unadopt();
		} else if (m_prepared_validated) {
			SwapPreparedStage(state.stage_preps.vertex[0], saved_prepared->vertex_prep);
			for (uint32_t i = 1; i < saved_prepared->vertex_stages; i++) {
				SwapPreparedStage(state.stage_preps.vertex[i], saved_prepared->tess_prep[i - 1]);
			}
			if (saved_prepared->pixel_active) {
				SwapPreparedStage(state.stage_preps.pixel, saved_prepared->pixel_prep);
			}
		}
		m_prepared_draw       = saved_prepared;
		m_binding_plan_active = false;
		m_prepared_validated  = false;
		m_run_pipeswap        = false;
		counters.declined.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	ContKernel::NoteTaken(DrawPrep::BkLeanMode() != DrawPrep::Bookkeep::Mode::Off);
	m_bk_kernel_taken         = true;
	m_run_slice_offset        = args.render_target_slice_offset;
	state.color_count         = m_run.color_count;
	state.color_slots_written = m_run.color_slots;
	m_run_active              = true;
	DrawRunMarkBindings();

	DrawIndexBufferSource index_source {};
	index_source.address            = reinterpret_cast<uint64_t>(args.index_addr);
	index_source.guest_element_size = m_kernel_record.element_size;
	index_source.type = m_kernel_record.element_size == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	DrawEmitInfo emit {};
	// KYTY_SPEC_SEGMENTS (drawPrep/segment.h): the draw the preparing thread encoded, validated
	// against this run; mode 1 takes its offsets and splices its draw command, verify compares.
	const auto* spliced = SegmentValidate(args);
	if (spliced != nullptr) {
		emit.vertex_offset  = spliced->vertex_offset;
		emit.first_instance = spliced->first_instance;
	} else {
		const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
		emit.vertex_offset  = vertex_offset + args.base_vertex;
		emit.first_instance = instance_offset;
		if (m_segment_validated) {
			SegmentVerify(args, emit, index_source);
		}
	}
	m_segment_splice = spliced;
	// KYTY_BK_ADOPT verify: the adopted preparations, as they were before the draw.
	static PipelineCache::GraphicsStagePreps adopt_copy; // (the GPU thread only)
	auto* const adopted = state.adopted;
	if (adopted != nullptr) {
		bk.Add(bk.adopted);
		if (adopt_mode == DrawPrep::Bookkeep::Mode::Verify) {
			adopt_copy.vertex[0] = adopted->vertex_prep;
			if (adopted->pixel_active) {
				adopt_copy.pixel = adopted->pixel_prep;
			}
		}
	}
	m_kernel = true;
	CommitStats::StepMark(CommitStats::Step::Restart);
	ExecutePreparedDraw(submit_id, buffer, draw, state, m_kernel_record.topology, emit, index_source,
	                    m_kernel_record.primitive_restart);
	m_kernel = false;
	ResetBindings();
	if (adopted != nullptr) {
		if (adopt_mode == DrawPrep::Bookkeep::Mode::Verify) {
			// The draw reads the preparations through the runtimes only; it must not have changed them
			// (the swap would have left such a change in the draw state, the adoption in the slot).
			bk.Add(bk.adopt_checks);
			if (!SameStagePrep(adopt_copy.vertex[0], adopted->vertex_prep) ||
			    (adopted->pixel_active && !SameStagePrep(adopt_copy.pixel, adopted->pixel_prep))) {
				DrawPrep::Bookkeep::ReportMismatch(bk.adopt_mismatches, "adopt", "preparation changed by the draw");
			}
		}
		// The slot retires after this commit: nothing may point into it.
		state.Unadopt();
	}
	CommitStats::StepMark(CommitStats::Step::ResetBindings);
	return true;
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
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	// Self time here includes renderer-lock acquisition and setup outside the
	// separately timed preparation/execution phases.
	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	CommitStats::StepMark(CommitStats::Step::DebugInfo);
	RenderLockGuard lock(m_context.GetMutex());
	if (PassDump::Before(m_context)) { // KYTY_PASS_DUMP: the previous pass segment's outputs
		buffer.EndRendering();
		m_run.valid        = false;
		m_run_active       = false;
		m_run_targets_kept = false;
		for (auto& record: m_run_kept) {
			record.valid = false;
		}
		PassDump::Flush(m_context);
	}
	CommitStats::StepMark(CommitStats::Step::Lock);
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	CommitStats::StepMark(CommitStats::Step::BeginRun);
	if (args.index_count == 0 || args.instance_count == 0) {
		PassCensus::Drop('d', 0, "skipped: zero indices or instances");
		return;
	}
	// KYTY_CONT_KERNEL (above ExecutePreparedDraw).
	const auto kernel_mode  = g_cont_kernel.Get();
	const bool kernel_entry = kernel_mode != 0 && ContinuationKernelEntry(args);
	if (CommitStats::Enabled()) [[unlikely]] {
		m_stats_kernel = kernel_mode == 0 ? CommitStats::KernelOutcome::Off
		                 : kernel_entry   ? CommitStats::KernelOutcome::Declined
		                                  : CommitStats::KernelOutcome::NoEntry;
	}
	if (kernel_entry) {
		if (kernel_mode == 1) {
			CommitStats::StepMark(CommitStats::Step::KernelEntry);
			if (ContinuationKernel(submit_id, buffer, args)) {
				return;
			}
		} else {
			m_kernel_verify = true;
			ContKernel::g_counters.draws.fetch_add(1, std::memory_order_relaxed);
		}
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		PassCensus::Drop('d', 0, "consumed: metadata colour op / depth-stencil copy / colour resolve");
		if (m_kernel_verify) {
			ContKernel::Check(ContKernel::TargetOps, false); // an upper bound: the kernel skips these
		}
		ResetBindings();
		return;
	}
	CommitStats::StepMark(CommitStats::Step::TargetOps);

	if (!DrawHasValidVertexShader(sh_ctx) || SkipCubeDraw(buffer)) {
		PassCensus::Drop('d', 0, "skipped: no valid vertex shader / SkipCubeDraw");
		if (m_kernel_verify) {
			ContKernel::Check(ContKernel::VsCheck, false); // an upper bound, as above
		}
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

	CommitStats::StepMark(CommitStats::Step::VsCheck);
	RunDrawChecks(buffer, ucfg, m_binding_plan);
	CommitStats::StepMark(CommitStats::Step::DrawChecks);

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
	DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
	                   args.first_instance};
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::StepMark(CommitStats::Step::StateReset);
	CommitStats::Mark(CommitStats::Phase::Setup);
	m_mrt_split_slots   = 0;
	m_mrt_split_allowed = true;
	const bool prepared = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	m_mrt_split_allowed = false;
	if (!prepared) {
		m_mrt_split_slots = 0;
		ResetBindings();
		return;
	}
	// KYTY_GPU_INSTANCE_COUNT: the inherited instance count, on the GPU or read here after all.
	uint64_t gpu_instance_address = args.gpu_instance_address;
	if (gpu_instance_address != 0 && !GpuInstanceCountApplies(buffer, ucfg, state)) {
		draw.instance_count  = ReadInheritedInstances(m_context, gpu_instance_address);
		gpu_instance_address = 0;
		if (draw.instance_count == 0) {
			m_mrt_split_slots = 0;
			ResetBindings();
			return;
		}
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

	// KYTY_LIST_RESET (RewriteListReset): after the draw's programs are known, as a mesh-shader
	// vertex stage reads the guest's indices itself.
	std::vector<uint8_t> list_reset_indices;
	if (args.offset_source != DrawOffsetSource::IndirectArgs &&
	    state.vertex_info[0].stage.program->stage != ShaderType::Mesh) {
		if (const auto kept = RewriteListReset(buffer, m_context.GetBufferCache(), index_source,
		                                       args.index_count, list_reset_indices);
		    kept != UINT32_MAX) {
			if (kept == 0) {
				ResetBindings();
				return;
			}
			index_source.host_data = list_reset_indices.data();
			index_source.size      = list_reset_indices.size();
			draw.index_count       = kept;
		}
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
	emit.gpu_instance_address = gpu_instance_address;

	CommitStats::StepMark(CommitStats::Step::Restart);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
	CommitStats::StepMark(CommitStats::Step::ResetBindings);
	if (const auto split = std::exchange(m_mrt_split_slots, 0u); split != 0u) {
		// KYTY_MRT_SPLIT: the second pass, with only the slots the first left out.
		NoteMrtSplit(split);
		m_run.valid = false;
		BeginDrawRun();
		state.Reset();
		m_mrt_filter = split;
		if (PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
			ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
			                    primitive_restart);
		}
		m_mrt_filter = ~0u;
		ResetBindings();
		m_run.valid = false;
		BeginDrawRun();
	}
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
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	CommitStats::StepMark(CommitStats::Step::DebugInfo);
	RenderLockGuard lock(m_context.GetMutex());
	if (PassDump::Before(m_context)) { // KYTY_PASS_DUMP: the previous pass segment's outputs
		buffer.EndRendering();
		m_run.valid        = false;
		m_run_active       = false;
		m_run_targets_kept = false;
		for (auto& record: m_run_kept) {
			record.valid = false;
		}
		PassDump::Flush(m_context);
	}
	CommitStats::StepMark(CommitStats::Step::Lock);
	// KYTY_DRAW_RUN: only a draw that records through the eligible path seeds the next continuation.
	BeginDrawRun();
	CommitStats::StepMark(CommitStats::Step::BeginRun);
	if (args.vertex_count == 0 || args.instance_count == 0) {
		PassCensus::Drop('d', 0, "skipped: zero vertices or instances");
		return;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		PassCensus::Drop('d', 0, "consumed: metadata colour op / depth-stencil copy / colour resolve");
		ResetBindings();
		return;
	}
	CommitStats::StepMark(CommitStats::Step::TargetOps);

	if (!DrawHasValidVertexShader(sh_ctx)) {
		PassCensus::Drop('d', 0, "skipped: no valid vertex shader");
		return;
	}
	if (SkipCubeDraw(buffer)) {
		PassCensus::Drop('d', 0, "skipped: SkipCubeDraw");
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

	CommitStats::StepMark(CommitStats::Step::VsCheck);
	RunDrawChecks(buffer, ucfg, m_binding_plan);
	CommitStats::StepMark(CommitStats::Step::DrawChecks);

	DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	// The member state is reused; the render mutex held above makes it exclusive to this draw.
	auto& state = *m_draw_state;
	state.Reset();
	CommitStats::StepMark(CommitStats::Step::StateReset);
	CommitStats::Mark(CommitStats::Phase::Setup);
	m_mrt_split_slots   = 0;
	m_mrt_split_allowed = true;
	const bool prepared = PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state);
	m_mrt_split_allowed = false;
	if (!prepared) {
		m_mrt_split_slots = 0;
		ResetBindings();
		return;
	}
	// KYTY_GPU_INSTANCE_COUNT: the inherited instance count, on the GPU or read here after all.
	uint64_t gpu_instance_address = args.gpu_instance_address;
	if (gpu_instance_address != 0 && !GpuInstanceCountApplies(buffer, ucfg, state)) {
		draw.instance_count  = ReadInheritedInstances(m_context, gpu_instance_address);
		gpu_instance_address = 0;
		if (draw.instance_count == 0) {
			m_mrt_split_slots = 0;
			ResetBindings();
			return;
		}
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
		m_mrt_split_slots = 0;
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
	emit.gpu_instance_address = gpu_instance_address;

	DrawIndexBufferSource index_source {};
	CommitStats::StepMark(CommitStats::Step::Restart);
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
	CommitStats::StepMark(CommitStats::Step::ResetBindings);
	if (const auto split = std::exchange(m_mrt_split_slots, 0u); split != 0u) {
		// KYTY_MRT_SPLIT: the second pass, with only the slots the first left out.
		NoteMrtSplit(split);
		m_run.valid = false;
		BeginDrawRun();
		state.Reset();
		m_mrt_filter = split;
		if (PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
			ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
		}
		m_mrt_filter = ~0u;
		ResetBindings();
		m_run.valid = false;
		BeginDrawRun();
	}
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
	CommitStats::StepMark(CommitStats::Step::PopReady);
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	const auto debug_op =
	    source.indexed ? CommandBufferDebugOp::DrawIndex : CommandBufferDebugOp::DrawIndexAuto;
	// Counts are unknown on the CPU: 0xFFFFFFFF, with the argument address in place of the index
	// or first-instance detail.
	buffer.SetDebugInfo(static_cast<uint32_t>(debug_op), submit_id, UINT32_MAX, 0, 1, UINT32_MAX,
	                    source.args_addr);

	KYTY_PROFILER_DETAIL_BLOCK("Draw::SetupAndExecution");
	RenderLockGuard lock(m_context.GetMutex());
	if (PassDump::Before(m_context)) { // KYTY_PASS_DUMP: the previous pass segment's outputs
		buffer.EndRendering();
		m_run.valid        = false;
		m_run_active       = false;
		m_run_targets_kept = false;
		for (auto& record: m_run_kept) {
			record.valid = false;
		}
		PassDump::Flush(m_context);
	}
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
	// KYTY_RESOLVE_CACHE_VIEW=1 (default off; the GT7 preset; usa-ratt/KytyPS5-GT7 e1134351f from
	// the GTA V fork 6bb0811e): a target can lie inside a larger image (a face of a cube, a slice of
	// an array the guest points a colour target at); FindImage then moved the view to that level
	// and layer, which is where the resolve reads and writes. Off: the register view's level and
	// layer (every face lands in the first).
	static const bool cache_view = [] {
		const char* value = std::getenv("KYTY_RESOLVE_CACHE_VIEW");
		return value != nullptr && std::strcmp(value, "0") != 0;
	}();
	if (cache_view) {
		destination.Resolve(source,
		                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
		                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
		return true;
	}
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
