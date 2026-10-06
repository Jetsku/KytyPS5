#include "graphics/host_gpu/renderer/drawPrep/repeatTrace.h"

#include "common/hangTrace.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/drawPrep/drawPrep.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "kernel/memory.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace Libs::Graphics::RepeatTrace {

namespace {

const char* EnvValue(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && *value != '\0' ? value : nullptr;
}

uint64_t EnvU64(const char* name, uint64_t fallback) {
	const auto* value = EnvValue(name);
	return value == nullptr ? fallback : std::strtoull(value, nullptr, 10);
}

bool EnvFlag(const char* name, bool fallback) {
	const auto* value = EnvValue(name);
	if (value == nullptr) {
		return fallback;
	}
	return !(value[0] == '0' || value[0] == 'n' || value[0] == 'N' || value[0] == 'f' ||
	         value[0] == 'F');
}

uint64_t NowMs() {
	return HangTrace::NowNs() / 1'000'000u;
}

// Field-wise hashing: the register structs have padding, which a raw byte hash would include.
class Words {
public:
	void Clear() { m_words.clear(); }
	void U32(uint32_t value) { m_words.push_back(value); }
	void U64(uint64_t value) {
		m_words.push_back(static_cast<uint32_t>(value));
		m_words.push_back(static_cast<uint32_t>(value >> 32u));
	}
	void B(bool value) { m_words.push_back(value ? 1u : 0u); }
	void F(float value) {
		uint32_t bits = 0;
		std::memcpy(&bits, &value, sizeof(bits));
		m_words.push_back(bits);
	}
	template <typename E>
	void E32(E value) {
		m_words.push_back(static_cast<uint32_t>(value));
	}
	void Span(std::span<const uint32_t> words) {
		m_words.push_back(static_cast<uint32_t>(words.size()));
		m_words.insert(m_words.end(), words.begin(), words.end());
	}
	[[nodiscard]] uint64_t Hash() const {
		return XXH3_64bits(m_words.data(), m_words.size() * sizeof(uint32_t));
	}

private:
	std::vector<uint32_t> m_words;
};

// Streaming hash states of this thread (created once; XXH3_state_t is opaque here).
XXH3_state_t* ThreadHashState(uint32_t index) {
	thread_local std::array<XXH3_state_t*, 3> states {};
	if (states[index] == nullptr) {
		states[index] = XXH3_createState();
	}
	XXH3_64bits_reset(states[index]);
	return states[index];
}

Words& ThreadWords() {
	thread_local Words words;
	words.Clear();
	return words;
}

void PutRenderTarget(Words& w, const HW::RenderTarget& rt) {
	w.U64(rt.base.addr);
	w.U32(rt.view.base_array_slice_index);
	w.U32(rt.view.last_array_slice_index);
	w.U32(rt.view.current_mip_level);
	const auto& info = rt.info;
	w.U32((info.fmask_compression_enable ? 1u : 0u) | (info.fmask_data_compression_disable ? 2u : 0u) |
	      (info.fmask_one_frag_mode ? 4u : 0u) | (info.cmask_fast_clear_enable ? 8u : 0u) |
	      (info.dcc_compression_enable ? 16u : 0u) | (info.blend_clamp ? 32u : 0u) |
	      (info.blend_bypass ? 64u : 0u) | (info.round_mode ? 128u : 0u));
	w.E32(info.format);
	w.E32(info.channel_type);
	w.E32(info.channel_order);
	w.B(rt.attrib.force_dest_alpha_to_one);
	w.U32(rt.attrib.num_samples);
	w.U32(rt.attrib.num_fragments);
	w.U32(rt.attrib2.height);
	w.U32(rt.attrib2.width);
	w.U32(rt.attrib2.num_mip_levels);
	w.U32(rt.attrib3.depth);
	w.E32(rt.attrib3.tile_mode);
	w.U32(rt.attrib3.dimension);
	w.B(rt.attrib3.metadata_pipe_aligned);
	w.B(rt.attrib3.write_vrs_rate_hint_to_cmask);
	w.U32(rt.dcc.max_uncompressed_block_size);
	w.U32(rt.dcc.max_compressed_block_size);
	w.U32(rt.dcc.color_transform);
	w.B(rt.dcc.dcc_clear_key_enable);
	w.B(rt.dcc.overwrite_combiner_disable);
	w.E32(rt.dcc.independent_block_size);
	w.B(rt.dcc.data_write_on_dcc_clear_to_reg);
	w.U64(rt.cmask.addr);
	w.U64(rt.fmask.addr);
	w.U32(rt.clear_word0.word0);
	w.U32(rt.clear_word1.word1);
	w.U64(rt.dcc_addr.addr);
}

void PutDepthTarget(Words& w, const HW::DepthRenderTarget& d) {
	w.E32(d.z_info.format);
	w.U32(d.z_info.num_samples);
	w.E32(d.z_info.texture_compatibility);
	w.E32(d.z_info.z_compare_base);
	w.B(d.z_info.htile_acceleration);
	w.B(d.z_info.expclear_enabled);
	w.B(d.z_info.partially_resident);
	w.U32(d.z_info.max_mip_level);
	w.E32(d.stencil_info.format);
	w.E32(d.stencil_info.texture_compatibility);
	w.B(d.stencil_info.expclear_enabled);
	w.B(d.stencil_info.htile_stencil_disabled);
	w.B(d.stencil_info.partially_resident);
	w.U32(d.depth_view.slice_start);
	w.U32(d.depth_view.slice_max);
	w.U32(d.depth_view.current_mip_level);
	w.B(d.depth_view.depth_write_disable);
	w.B(d.depth_view.stencil_write_disable);
	w.U32(d.size.x_max);
	w.U32(d.size.y_max);
	w.B(d.size.valid);
	w.U64(d.z_read_base_addr);
	w.U64(d.stencil_read_base_addr);
	w.U64(d.z_write_base_addr);
	w.U64(d.stencil_write_base_addr);
	w.U64(d.htile_data_base_addr);
	w.U32(d.shading_rate_encoding);
}

uint64_t HashTargets(const HW::Context& ctx) {
	auto& w = ThreadWords();
	w.U32(ctx.GetRenderTargetMask());
	for (uint32_t slot = 0; slot < 8; slot++) {
		PutRenderTarget(w, ctx.GetRenderTarget(slot));
	}
	PutDepthTarget(w, ctx.GetDepthRenderTarget());
	return w.Hash();
}

uint64_t HashViewport(const HW::Context& ctx) {
	auto&       w  = ThreadWords();
	const auto& sv = ctx.GetScreenViewport();
	for (const auto& vp: sv.viewports) {
		w.F(vp.zmin);
		w.F(vp.zmax);
		w.F(vp.xscale);
		w.F(vp.xoffset);
		w.F(vp.yscale);
		w.F(vp.yoffset);
		w.F(vp.zscale);
		w.F(vp.zoffset);
		w.U32(static_cast<uint32_t>(vp.viewport_scissor_left));
		w.U32(static_cast<uint32_t>(vp.viewport_scissor_top));
		w.U32(static_cast<uint32_t>(vp.viewport_scissor_right));
		w.U32(static_cast<uint32_t>(vp.viewport_scissor_bottom));
		w.B(vp.viewport_scissor_window_offset_enable);
	}
	w.U32(sv.transform_control);
	for (const int value: {sv.screen_scissor_left, sv.screen_scissor_top, sv.screen_scissor_right,
	                       sv.screen_scissor_bottom, sv.window_scissor_left, sv.window_scissor_top,
	                       sv.window_scissor_right, sv.window_scissor_bottom,
	                       sv.generic_scissor_left, sv.generic_scissor_top,
	                       sv.generic_scissor_right, sv.generic_scissor_bottom, sv.window_offset_x,
	                       sv.window_offset_y}) {
		w.U32(static_cast<uint32_t>(value));
	}
	w.B(sv.window_scissor_window_offset_enable);
	w.B(sv.generic_scissor_window_offset_enable);
	w.U32(sv.hw_offset_x);
	w.U32(sv.hw_offset_y);
	w.F(sv.guard_band_horz_clip);
	w.F(sv.guard_band_vert_clip);
	w.F(sv.guard_band_horz_discard);
	w.F(sv.guard_band_vert_discard);
	w.U32(sv.clip_rect_rule);
	for (uint32_t i = 0; i < 4; i++) {
		w.U32(static_cast<uint32_t>(sv.clip_rect_left[i]));
		w.U32(static_cast<uint32_t>(sv.clip_rect_top[i]));
		w.U32(static_cast<uint32_t>(sv.clip_rect_right[i]));
		w.U32(static_cast<uint32_t>(sv.clip_rect_bottom[i]));
		w.B(sv.clip_rect_window_offset_enable[i]);
	}
	return w.Hash();
}

uint64_t HashState(const HW::Context& ctx, const HW::UserConfig& ucfg) {
	auto& w = ThreadWords();
	for (uint32_t slot = 0; slot < 8; slot++) {
		const auto& b = ctx.GetBlendControl(slot);
		w.U32(b.color_srcblend | (b.color_comb_fcn << 8u) | (b.color_destblend << 16u));
		w.U32(b.alpha_srcblend | (b.alpha_comb_fcn << 8u) | (b.alpha_destblend << 16u));
		w.B(b.separate_alpha_blend);
		w.B(b.enable);
	}
	const auto& bc = ctx.GetBlendColor();
	w.F(bc.red);
	w.F(bc.green);
	w.F(bc.blue);
	w.F(bc.alpha);
	w.U32(ctx.GetColorControl().mode | (ctx.GetColorControl().op << 8u));
	const auto& smc = ctx.GetScanModeControl();
	w.U32((smc.msaa_enable ? 1u : 0u) | (smc.vport_scissor_enable ? 2u : 0u) |
	      (smc.line_stipple_enable ? 4u : 0u));
	const auto& asc = ctx.GetAaSampleControl();
	w.U64(asc.centroid_priority);
	for (const auto location: asc.locations) {
		w.U32(location);
	}
	const auto& aa = ctx.GetAaConfig();
	w.U32(aa.msaa_num_samples | (aa.max_sample_dist << 8u) | (aa.msaa_exposed_samples << 16u));
	w.B(aa.aa_mask_centroid_dtmn);
	w.U32(ctx.GetShaderStages());
	const auto& dc = ctx.GetDepthControl();
	w.U32((dc.stencil_enable ? 1u : 0u) | (dc.z_enable ? 2u : 0u) | (dc.z_write_enable ? 4u : 0u) |
	      (dc.depth_bounds_enable ? 8u : 0u) | (dc.backface_enable ? 16u : 0u));
	w.U32(dc.zfunc | (dc.stencilfunc << 8u) | (dc.stencilfunc_bf << 16u));
	const auto& sc = ctx.GetStencilControl();
	w.U32(sc.stencil_fail | (sc.stencil_zpass << 8u) | (sc.stencil_zfail << 16u));
	w.U32(sc.stencil_fail_bf | (sc.stencil_zpass_bf << 8u) | (sc.stencil_zfail_bf << 16u));
	const auto& sm = ctx.GetStencilMask();
	w.U32(sm.stencil_testval | (sm.stencil_mask << 8u) | (sm.stencil_writemask << 16u) |
	      (static_cast<uint32_t>(sm.stencil_opval) << 24u));
	w.U32(sm.stencil_testval_bf | (sm.stencil_mask_bf << 8u) | (sm.stencil_writemask_bf << 16u) |
	      (static_cast<uint32_t>(sm.stencil_opval_bf) << 24u));
	const auto& mc = ctx.GetModeControl();
	w.U32((mc.cull_front ? 1u : 0u) | (mc.cull_back ? 2u : 0u) | (mc.face ? 4u : 0u) |
	      (mc.poly_offset_front_enable ? 8u : 0u) | (mc.poly_offset_back_enable ? 16u : 0u) |
	      (mc.vtx_window_offset_enable ? 32u : 0u) | (mc.provoking_vtx_last ? 64u : 0u) |
	      (mc.persp_corr_dis ? 128u : 0u));
	w.U32(mc.poly_mode | (mc.polymode_front_ptype << 8u) | (mc.polymode_back_ptype << 16u));
	const auto& po = ctx.GetPolyOffset();
	w.U32(static_cast<uint32_t>(static_cast<int32_t>(po.neg_num_db_bits)));
	w.B(po.db_is_float_fmt);
	w.F(po.clamp);
	w.F(po.front_scale);
	w.F(po.front_offset);
	w.F(po.back_scale);
	w.F(po.back_offset);
	const auto& cc = ctx.GetClipControl();
	w.U32(cc.user_clip_planes | (cc.user_clip_plane_mode << 8u));
	w.U32((cc.dx_clip_space ? 1u : 0u) | (cc.vertex_kill_any ? 2u : 0u) |
	      (cc.min_z_clip_disable ? 4u : 0u) | (cc.max_z_clip_disable ? 8u : 0u) |
	      (cc.user_clip_plane_negate_y ? 16u : 0u) | (cc.clip_disable ? 32u : 0u) |
	      (cc.user_clip_plane_cull_only ? 64u : 0u) |
	      (cc.cull_on_clipping_error_disable ? 128u : 0u) |
	      (cc.linear_attribute_clip_enable ? 256u : 0u) |
	      (cc.force_viewport_index_from_vs_enable ? 512u : 0u));
	const auto& eq = ctx.GetEqaaControl();
	w.U32(eq.max_anchor_samples | (eq.ps_iter_samples << 8u) | (eq.mask_export_num_samples << 16u) |
	      (static_cast<uint32_t>(eq.alpha_to_mask_num_samples) << 24u));
	w.U32((eq.high_quality_intersections ? 1u : 0u) | (eq.incoherent_eqaa_reads ? 2u : 0u) |
	      (eq.interpolate_comp_z ? 4u : 0u) | (eq.static_anchor_associations ? 8u : 0u));
	const auto& rc = ctx.GetRenderControl();
	w.U32((rc.depth_clear_enable ? 1u : 0u) | (rc.stencil_clear_enable ? 2u : 0u) |
	      (rc.resummarize_enable ? 4u : 0u) | (rc.stencil_compress_disable ? 8u : 0u) |
	      (rc.depth_compress_disable ? 16u : 0u) | (rc.copy_depth_to_color ? 32u : 0u) |
	      (rc.copy_stencil_to_color ? 64u : 0u) | (rc.copy_centroid ? 128u : 0u));
	w.U32(rc.copy_sample);
	const auto& ro = ctx.GetDepthRenderOverride();
	w.U32((ro.force_z_valid ? 1u : 0u) | (ro.force_z_dirty ? 2u : 0u) |
	      (ro.force_stencil_valid ? 4u : 0u) | (ro.force_stencil_dirty ? 8u : 0u));
	w.U32(ctx.GetDepthCountControl());
	w.F(ctx.GetDepthClearValue());
	w.F(ctx.GetDepthBoundsMin());
	w.F(ctx.GetDepthBoundsMax());
	w.U32(ctx.GetStencilClearValue());
	w.F(ctx.GetLineWidth());
	w.U32(ctx.GetPrimitiveResetIndex());
	const auto& sh = ctx.GetShaderRegisters();
	for (const uint32_t value:
	     {sh.m_spiVsOutConfig, sh.m_spiShaderPosFormat, sh.m_paClVsOutCntl, sh.m_spiShaderIdxFormat,
	      sh.m_geNggSubgrpCntl, sh.m_vgtGsInstanceCnt, sh.m_vgtGsOnchipCntl,
	      sh.m_vgtHosMaxTessLevel, sh.m_vgtHosMinTessLevel, sh.m_geMaxOutputPerSubgroup,
	      sh.m_vgtEsgsRingItemsize, sh.m_vgtGsMaxVertOut, sh.m_vgtGsOutPrimType,
	      sh.m_vgtPrimitiveIdEn, sh.m_vgtReuseOff, sh.m_vgtTessDistribution, sh.m_vgtLsHsConfig,
	      sh.m_vgtTfParam, sh.shader_z_format, sh.ps_input_ena, sh.ps_input_addr,
	      sh.ps_in_control, sh.baryc_cntl, sh.m_cbShaderMask, sh.m_paScShaderControl}) {
		w.U32(value);
	}
	for (const auto value: sh.ps_interpolator_settings) {
		w.U32(value);
	}
	for (const auto mode: sh.target_output_mode) {
		w.U32(mode);
	}
	const auto& db = sh.db_shader_control;
	w.U32(db.other_bits);
	w.U32(db.conservative_z_export_value | (db.shader_z_behavior << 8u));
	w.U32((db.shader_kill_enable ? 1u : 0u) | (db.shader_z_export_enable ? 2u : 0u) |
	      (db.shader_mask_export_enable ? 4u : 0u) | (db.shader_dual_export_enable ? 8u : 0u) |
	      (db.shader_execute_on_noop ? 16u : 0u) | (db.alpha_to_mask_disable ? 32u : 0u));
	// User config (the index offset and object id are draw arguments).
	w.E32(ucfg.GetPrimType());
	w.U32(ucfg.GetPrimitiveResetControl());
	w.U32(ucfg.GetGeControl().primitive_group_size | (ucfg.GetGeControl().vertex_group_size << 16u));
	const auto& en = ucfg.GetGeUserVgprEn();
	w.U32((en.vgpr1 ? 1u : 0u) | (en.vgpr2 ? 2u : 0u) | (en.vgpr3 ? 4u : 0u));
	const auto& oa = ucfg.GetGdsOaState();
	w.U32(oa.cntl);
	for (const auto& counter: oa.counters) {
		w.U32(counter.counter);
		w.U32(counter.address);
	}
	const auto& fsr = ucfg.GetFsrView();
	for (const auto& axis: fsr.control_points) {
		for (const auto value: axis) {
			w.U32(value);
		}
	}
	for (const auto& axis: fsr.alphas) {
		for (const auto value: axis) {
			w.U32(value);
		}
	}
	w.U32(fsr.window[0]);
	w.U32(fsr.window[1]);
	return w.Hash();
}

void PutUserSgpr(Words& w, const HW::UserSgprInfo& info) {
	w.U32(info.count);
	for (uint32_t i = 0; i < info.count && i < HW::UserSgprInfo::SGPRS_MAX; i++) {
		w.U32(info.value[i]);
		w.E32(info.type[i]);
	}
}

uint64_t HashShader(const HW::Shader& shaders) {
	auto&       w  = ThreadWords();
	const auto& vs = shaders.GetVs();
	const auto& ps = shaders.GetPs();
	w.U64(vs.es_regs.data_addr);
	w.U64(vs.ls_regs.data_addr);
	w.U64(vs.hs_regs.data_addr);
	const auto& hr1 = vs.hs_regs.rsrc1;
	w.U32(hr1.vgprs | (hr1.priority << 8u) | (hr1.float_mode << 16u) |
	      (static_cast<uint32_t>(hr1.ls_vgpr_component_count) << 24u));
	w.U32((hr1.dx10_clamp ? 1u : 0u) | (hr1.debug_mode ? 2u : 0u) | (hr1.ieee_mode ? 4u : 0u) |
	      (hr1.require_forward_progress ? 8u : 0u) | (hr1.threadgroup_configuration ? 16u : 0u) |
	      (hr1.fp16_overflow ? 32u : 0u));
	const auto& hr2 = vs.hs_regs.rsrc2;
	w.U32((hr2.scratch_en ? 1u : 0u) | (hr2.user_sgpr << 8u) | (hr2.shared_vgprs << 16u));
	w.U32(hr2.lds_size);
	w.U64(vs.gs_regs.data_addr);
	const auto& gr1 = vs.gs_regs.rsrc1;
	w.U32(gr1.vgprs | (gr1.priority << 8u) | (gr1.float_mode << 16u) |
	      (static_cast<uint32_t>(gr1.gs_vgpr_component_count) << 24u));
	w.U32((gr1.dx10_clamp ? 1u : 0u) | (gr1.debug_mode ? 2u : 0u) | (gr1.ieee_mode ? 4u : 0u) |
	      (gr1.cu_group_enable ? 8u : 0u) | (gr1.require_forward_progress ? 16u : 0u) |
	      (gr1.threadgroup_configuration ? 32u : 0u) | (gr1.fp16_overflow ? 64u : 0u));
	const auto& gr2 = vs.gs_regs.rsrc2;
	w.U32((gr2.scratch_en ? 1u : 0u) | (gr2.user_sgpr << 8u) | (gr2.es_vgpr_component_count << 16u) |
	      (static_cast<uint32_t>(gr2.lds_size) << 24u));
	w.U32((gr2.offchip_lds ? 1u : 0u) | (gr2.shared_vgprs << 8u));
	w.U64(ps.ps_regs.data_addr);
	const auto& pr1 = ps.ps_regs.rsrc1;
	w.U32(pr1.vgprs | (pr1.priority << 8u) | (pr1.float_mode << 16u));
	w.U32((pr1.dx10_clamp ? 1u : 0u) | (pr1.debug_mode ? 2u : 0u) | (pr1.ieee_mode ? 4u : 0u) |
	      (pr1.cu_group_disable ? 8u : 0u) | (pr1.require_forward_progress ? 16u : 0u) |
	      (pr1.fp16_overflow ? 32u : 0u));
	const auto& pr2 = ps.ps_regs.rsrc2;
	w.U32((pr2.scratch_en ? 1u : 0u) | (pr2.user_sgpr << 8u) | (pr2.wave_cnt_en ? 0x10000u : 0u));
	w.U32(pr2.extra_lds_size | (pr2.raster_ordered_shading << 8u) | (pr2.shared_vgprs << 16u));
	return w.Hash();
}

uint64_t HashUserSgpr(const HW::Shader& shaders) {
	auto&       w  = ThreadWords();
	const auto& vs = shaders.GetVs();
	w.U64(vs.hs_regs.user_data_addr);
	w.U64(vs.gs_regs.user_data_addr);
	PutUserSgpr(w, vs.hs_user_sgpr);
	PutUserSgpr(w, vs.gs_user_sgpr);
	w.U32(static_cast<uint32_t>(vs.start_instance_user_sgpr));
	PutUserSgpr(w, shaders.GetPs().ps_user_sgpr);
	return w.Hash();
}

void PutDescriptors(Words& w, const std::vector<ShaderRecompiler::IR::DescriptorValue>& values,
                    int mask_kind) {
	w.U32(static_cast<uint32_t>(values.size()));
	for (const auto& value: values) {
		w.U32(value.dword_count);
		for (uint32_t i = 0; i < value.dword_count && i < value.dwords.size(); i++) {
			auto word = value.dwords[i];
			if (mask_kind == 1 && value.dword_count >= 4) {
				// V# shape: drop BASE_ADDRESS (dword 0 and the low 16 bits of dword 1).
				word = i == 0 ? 0u : i == 1 ? (word & 0xffff0000u) : word;
			} else if (mask_kind == 2 && value.dword_count >= 8) {
				// T# shape: drop BASE_ADDRESS (dword 0 and the low 8 bits of dword 1).
				word = i == 0 ? 0u : i == 1 ? (word & 0xffffff00u) : word;
			}
			w.U32(word);
		}
	}
}

// Stage order: vertex stage 0 then pixel (when active).
template <typename F>
void ForStages(const DrawPrep::PreparedDraw& prepared, F&& f) {
	f(prepared.vertex_prep);
	if (prepared.pixel_active) {
		f(prepared.pixel_prep);
	}
}

struct Settings {
	bool     enabled        = false;
	bool     payload        = true;
	bool     dcb_dump       = true;
	uint64_t burst_frames   = 4;
	uint64_t burst_period   = 20000;
	uint64_t max_bursts     = 30;
	uint64_t dump_bursts    = 12;
	uint64_t payload_per_vs = 4096;
	uint64_t payload_limit  = 16384;
};

const Settings& GetSettings() {
	static const Settings settings = [] {
		Settings s;
		s.enabled        = EnvFlag("KYTY_CP_REPEAT_TRACE", false);
		s.payload        = EnvFlag("KYTY_CP_REPEAT_PAYLOAD", true);
		s.dcb_dump       = EnvFlag("KYTY_CP_REPEAT_DCB_DUMP", true);
		s.burst_frames   = std::max<uint64_t>(1, EnvU64("KYTY_CP_REPEAT_BURST_FRAMES", 4));
		s.burst_period   = EnvU64("KYTY_CP_REPEAT_BURST_PERIOD_MS", 20000);
		s.max_bursts     = EnvU64("KYTY_CP_REPEAT_BURSTS", 30);
		s.dump_bursts    = EnvU64("KYTY_CP_REPEAT_DCB_DUMP_BURSTS", 12);
		s.payload_per_vs = EnvU64("KYTY_CP_REPEAT_PAYLOAD_PER_VSHARP", 4096);
		s.payload_limit  = EnvU64("KYTY_CP_REPEAT_PAYLOAD_LIMIT", 16384);
		return s;
	}();
	return settings;
}

std::atomic<RenderContext*> g_renderer {nullptr};

// ---------------------------------------------------------------------------------------------
// GPU-thread frame accumulation and the background writer.

struct DispatchRecord {
	uint64_t cs_addr = 0;
	uint64_t h_regs  = 0; // CS registers and group counts
	uint64_t h_user  = 0; // user SGPRs
	uint32_t queue   = 0;
	uint32_t groups[3] {};
};

struct SubmitRecord {
	uint64_t address  = 0;
	uint64_t sequence = 0;
	uint64_t hash     = 0;
	uint32_t dwords   = 0;
	uint32_t queue    = 0;
	uint32_t kind     = 0; // 0 dcb, 1 ce, 2 ib call, 3 ib chain
};

struct Frame {
	uint64_t                    index = 0;
	uint64_t                    t_ms  = 0;
	bool                        burst = false;
	bool                        dump  = false;
	uint32_t                    unprepared_draws = 0;
	std::vector<DrawRecord>     draws;
	std::vector<DispatchRecord> dispatches;
	std::vector<SubmitRecord>   submits;
	std::vector<uint8_t>        raw; // burst dumps
};

// Composite keys (repeat-frames.csv), built from the components.
enum Composite : uint32_t { Struct, Bind, PrepInputs, Full, DispatchStruct, DispatchFull, CompositeCount };
constexpr const char* kComponentNames[Component::Count] = {
    "args",   "argshape", "targets", "viewport", "state",    "shader",   "usgpr",     "programs",
    "vsharp", "vshape",   "tsharp",  "ssharp",   "srt",      "userdata", "reads",     "readbytes",
    "payload"};
constexpr const char* kCompositeNames[CompositeCount] = {"struct", "bind", "prepin",
                                                         "full",   "dstruct", "dfull"};

uint64_t Combine(std::initializer_list<uint64_t> values) {
	return XXH3_64bits(std::data(values), values.size() * sizeof(uint64_t));
}

struct Keys {
	// [key][draw or dispatch], in stream order and sorted.
	std::vector<std::vector<uint64_t>> k;
	std::vector<std::vector<uint64_t>> sorted;
};

constexpr uint32_t kComponentCount = static_cast<uint32_t>(Component::Count);
constexpr uint32_t kKeyCount       = kComponentCount + static_cast<uint32_t>(CompositeCount);

constexpr uint32_t KeyIndex(Composite composite) {
	return kComponentCount + static_cast<uint32_t>(composite);
}

Keys BuildKeys(const Frame& frame) {
	Keys keys;
	keys.k.resize(kKeyCount);
	for (auto& v: keys.k) {
		v.reserve(frame.draws.size());
	}
	for (const auto& d: frame.draws) {
		for (uint32_t c = 0; c < Component::Count; c++) {
			keys.k[c].push_back(d.h[c]);
		}
		const auto& h = d.h;
		const auto  s = Combine({h[ArgShape], h[Targets], h[Viewport], h[State], h[Shader],
		                         h[Programs], h[VShape], h[TSharp], h[SSharp]});
		const auto  b = Combine({s, h[Args], h[VSharp], h[Srt], h[UserData], h[UserSgpr]});
		const auto  p = Combine({h[Shader], h[UserSgpr], h[Reads], h[State], h[Targets]});
		const auto  f = Combine({b, h[Reads], h[Payload]});
		keys.k[KeyIndex(Struct)].push_back(s);
		keys.k[KeyIndex(Bind)].push_back(b);
		keys.k[KeyIndex(PrepInputs)].push_back(p);
		keys.k[KeyIndex(Full)].push_back(f);
	}
	auto& ds = keys.k[KeyIndex(DispatchStruct)];
	auto& df = keys.k[KeyIndex(DispatchFull)];
	for (const auto& d: frame.dispatches) {
		ds.push_back(Combine({d.cs_addr, d.h_regs, d.queue}));
		df.push_back(Combine({d.cs_addr, d.h_regs, d.queue, d.h_user}));
	}
	keys.sorted = keys.k;
	for (auto& v: keys.sorted) {
		std::sort(v.begin(), v.end());
	}
	return keys;
}

// Multiset intersection size of two sorted key lists, as a fraction of `current`.
double MatchFraction(const std::vector<uint64_t>& current, const std::vector<uint64_t>& previous) {
	if (current.empty()) {
		return 0.0;
	}
	uint64_t hits = 0;
	size_t   i    = 0;
	size_t   j    = 0;
	while (i < current.size() && j < previous.size()) {
		if (current[i] == previous[j]) {
			hits++;
			i++;
			j++;
		} else if (current[i] < previous[j]) {
			i++;
		} else {
			j++;
		}
	}
	return static_cast<double>(hits) / static_cast<double>(current.size());
}

// Fraction of `current` (sorted) whose key occurs anywhere in `pool` (sorted, not consumed).
double PresentFraction(const std::vector<uint64_t>& current, const std::vector<uint64_t>& pool) {
	if (current.empty()) {
		return 0.0;
	}
	uint64_t hits = 0;
	size_t   j    = 0;
	for (const auto key: current) {
		while (j < pool.size() && pool[j] < key) {
			j++;
		}
		hits += j < pool.size() && pool[j] == key ? 1u : 0u;
	}
	return static_cast<double>(hits) / static_cast<double>(current.size());
}

class Writer {
public:
	Writer() {
		std::filesystem::path dir;
		if (const auto* value = EnvValue("KYTY_CP_REPEAT_DIR"); value != nullptr) {
			dir = value;
		} else if (auto hang = HangTrace::OutputDirectory(); !hang.empty()) {
			dir = hang;
		} else {
#ifdef _WIN32
			const auto pid = static_cast<uint64_t>(_getpid());
#else
			const auto pid = static_cast<uint64_t>(getpid());
#endif
			dir = std::filesystem::path("_HangTrace") / ("cprepeat-" + std::to_string(pid));
		}
		std::error_code error;
		std::filesystem::create_directories(dir, error);
		m_dir = dir;
		m_thread = std::thread([this] { Run(); });
		std::printf("Kyty CP repeat trace (KYTY_CP_REPEAT_TRACE=1): writing %s\n",
		            m_dir.string().c_str());
	}

	~Writer() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_condition.notify_all();
		if (m_thread.joinable()) {
			m_thread.join();
		}
		for (auto* file: {m_frames, m_draws, m_dispatches, m_submits}) {
			if (file != nullptr) {
				std::fclose(file);
			}
		}
	}

	Writer(const Writer&)            = delete;
	Writer& operator=(const Writer&) = delete;

	void Push(std::unique_ptr<Frame> frame) {
		{
			std::lock_guard lock(m_mutex);
			if (m_queue.size() > 64) {
				m_dropped++;
				return;
			}
			m_queue.push_back(std::move(frame));
		}
		m_condition.notify_one();
	}

private:
	FILE* Open(const char* name, const char* header) {
		const auto path = m_dir / name;
		FILE*      file = std::fopen(path.string().c_str(), "wb");
		if (file != nullptr) {
			std::fputs(header, file);
			std::fputc('\n', file);
		}
		return file;
	}

	void Run() {
		for (;;) {
			std::unique_ptr<Frame> frame;
			{
				std::unique_lock lock(m_mutex);
				m_condition.wait(lock, [this] { return m_stop || !m_queue.empty(); });
				if (m_queue.empty()) {
					return;
				}
				frame = std::move(m_queue.front());
				m_queue.pop_front();
			}
			Write(*frame);
		}
	}

	void Write(const Frame& frame) {
		if (m_frames == nullptr) {
			std::string header = "frame,t_ms,burst,draws,unprepared_draws,prepared_ok,committed,"
			                     "dispatches,submits,submit_bytes,submits_m1,submits_m3";
			for (uint32_t k = 0; k < kKeyCount; k++) {
				const char* name = k < kComponentCount ? kComponentNames[k]
				                                       : kCompositeNames[k - kComponentCount];
				for (const char* suffix: {"_m1", "_m2", "_m3", "_many", "_p1"}) {
					header += ",";
					header += name;
					header += suffix;
				}
			}
			header += ",dropped";
			m_frames = Open("repeat-frames.csv", header.c_str());
			m_draws  = Open("repeat-draws.csv",
			                "frame,idx,packet,kind,eligible,prep_ok,committed,failure,hashed,count,"
			                "instances,vs_addr,ps_addr,index_addr,rt0_addr,depth_addr,n_vsharp,"
			                "n_tsharp,n_ssharp,srt_words,read_ranges,read_bytes,payload_bytes,"
			                "payload_gpu,args,argshape,targets,viewport,state,shader,usgpr,programs,"
			                "vsharp,vshape,tsharp,ssharp,srt,userdata,reads,readbytes,payload");
			m_dispatches = Open("repeat-dispatches.csv",
			                    "frame,idx,queue,cs_addr,regs,user,groups_x,groups_y,groups_z");
			m_submits = Open("repeat-submits.csv",
			                 "frame,t_ms,queue,seq,kind,address,dwords,hash");
		}
		auto keys = BuildKeys(frame);
		// Submissions: content hashes of guest command buffers (with their kind).
		std::vector<uint64_t> submit_keys;
		uint64_t              submit_bytes = 0;
		for (const auto& s: frame.submits) {
			submit_keys.push_back(Combine({s.hash, s.kind, s.dwords}));
			submit_bytes += uint64_t {s.dwords} * 4u;
		}
		uint32_t prepared_ok = 0;
		uint32_t committed   = 0;
		for (const auto& d: frame.draws) {
			prepared_ok += d.prep_ok;
			committed += d.committed;
		}
		if (m_frames != nullptr) {
			std::string row = std::to_string(frame.index) + "," + std::to_string(frame.t_ms) + "," +
			                  (frame.burst ? "1" : "0") + "," + std::to_string(frame.draws.size()) +
			                  "," + std::to_string(frame.unprepared_draws) + "," +
			                  std::to_string(prepared_ok) + "," + std::to_string(committed) + "," +
			                  std::to_string(frame.dispatches.size()) + "," +
			                  std::to_string(frame.submits.size()) + "," +
			                  std::to_string(submit_bytes);
			char buf[64];
			std::vector<uint64_t> submit_sorted = submit_keys;
			std::sort(submit_sorted.begin(), submit_sorted.end());
			const auto submit_m = [&](size_t back) {
				if (m_submit_history.size() < back) {
					return std::string(",");
				}
				const auto& previous = m_submit_history[m_submit_history.size() - back];
				std::snprintf(buf, sizeof(buf), ",%.4f", MatchFraction(submit_sorted, previous));
				return std::string(buf);
			};
			row += submit_m(1);
			row += submit_m(3);
			for (uint32_t k = 0; k < kKeyCount; k++) {
				const auto& current = keys.sorted[k];
				std::array<std::string, 3> m;
				std::vector<uint64_t>      pool;
				for (size_t back = 1; back <= 3; back++) {
					if (m_history.size() >= back) {
						const auto& previous = m_history[m_history.size() - back].sorted[k];
						std::snprintf(buf, sizeof(buf), "%.4f", MatchFraction(current, previous));
						m[back - 1] = buf;
						pool.insert(pool.end(), previous.begin(), previous.end());
					}
				}
				row += "," + m[0] + "," + m[1] + "," + m[2];
				if (m_history.empty()) {
					row += ",,";
					continue;
				}
				std::sort(pool.begin(), pool.end());
				std::snprintf(buf, sizeof(buf), ",%.4f", PresentFraction(current, pool));
				row += buf;
				const auto& ordered  = keys.k[k];
				const auto& previous = m_history.back().k[k];
				uint64_t    same     = 0;
				for (size_t i = 0; i < ordered.size() && i < previous.size(); i++) {
					same += ordered[i] == previous[i] ? 1u : 0u;
				}
				std::snprintf(buf, sizeof(buf), ",%.4f",
				              ordered.empty() ? 0.0
				                              : static_cast<double>(same) /
				                                    static_cast<double>(ordered.size()));
				row += buf;
			}
			uint64_t dropped = 0;
			{
				std::lock_guard lock(m_mutex);
				dropped = m_dropped;
			}
			row += "," + std::to_string(dropped) + "\n";
			std::fputs(row.c_str(), m_frames);
			std::fflush(m_frames);
		}
		if (frame.burst && m_draws != nullptr) {
			const auto h48 = [](uint64_t value) { return value & 0xffffffffffffull; };
			for (size_t i = 0; i < frame.draws.size(); i++) {
				const auto& d = frame.draws[i];
				std::fprintf(m_draws,
				             "%llu,%zu,%llx,%u,%u,%u,%u,%u,%u,%u,%u,%llx,%llx,%llx,%llx,%llx,%u,%u,%u,%u,"
				             "%u,%u,%u,%u",
				             static_cast<unsigned long long>(frame.index), i,
				             static_cast<unsigned long long>(d.packet), d.kind, d.eligible,
				             d.prep_ok, d.committed, d.failure, d.hashed, d.count, d.instances,
				             static_cast<unsigned long long>(d.vs_addr),
				             static_cast<unsigned long long>(d.ps_addr),
				             static_cast<unsigned long long>(d.index_addr),
				             static_cast<unsigned long long>(d.rt0_addr),
				             static_cast<unsigned long long>(d.depth_addr), d.n_vsharp, d.n_tsharp,
				             d.n_ssharp, d.srt_words, d.read_ranges, d.read_bytes, d.payload_bytes,
				             d.payload_gpu);
				for (uint32_t c = 0; c < Component::Count; c++) {
					std::fprintf(m_draws, ",%llx", static_cast<unsigned long long>(h48(d.h[c])));
				}
				std::fputc('\n', m_draws);
			}
			for (size_t i = 0; i < frame.dispatches.size() && m_dispatches != nullptr; i++) {
				const auto& d = frame.dispatches[i];
				std::fprintf(m_dispatches, "%llu,%zu,%u,%llx,%llx,%llx,%u,%u,%u\n",
				             static_cast<unsigned long long>(frame.index), i, d.queue,
				             static_cast<unsigned long long>(d.cs_addr),
				             static_cast<unsigned long long>(h48(d.h_regs)),
				             static_cast<unsigned long long>(h48(d.h_user)), d.groups[0],
				             d.groups[1], d.groups[2]);
			}
			std::fflush(m_draws);
			if (m_dispatches != nullptr) {
				std::fflush(m_dispatches);
			}
		}
		if (m_submits != nullptr) {
			for (const auto& s: frame.submits) {
				std::fprintf(m_submits, "%llu,%llu,%u,%llu,%u,%llx,%u,%llx\n",
				             static_cast<unsigned long long>(frame.index),
				             static_cast<unsigned long long>(frame.t_ms), s.queue,
				             static_cast<unsigned long long>(s.sequence), s.kind,
				             static_cast<unsigned long long>(s.address), s.dwords,
				             static_cast<unsigned long long>(s.hash));
			}
			std::fflush(m_submits);
		}
		if (frame.dump && !frame.raw.empty()) {
			const auto path = m_dir / ("repeat-dcb-" + std::to_string(frame.index) + ".bin");
			if (FILE* file = std::fopen(path.string().c_str(), "wb"); file != nullptr) {
				std::fwrite(frame.raw.data(), 1, frame.raw.size(), file);
				std::fclose(file);
			}
		}
		m_history.push_back(std::move(keys));
		if (m_history.size() > 3) {
			m_history.pop_front();
		}
		std::sort(submit_keys.begin(), submit_keys.end());
		m_submit_history.push_back(std::move(submit_keys));
		if (m_submit_history.size() > 3) {
			m_submit_history.pop_front();
		}
	}

	std::filesystem::path                  m_dir;
	std::thread                            m_thread;
	std::mutex                             m_mutex;
	std::condition_variable                m_condition;
	std::deque<std::unique_ptr<Frame>>     m_queue;
	bool                                   m_stop    = false;
	uint64_t                               m_dropped = 0;
	std::deque<Keys>                       m_history;
	std::deque<std::vector<uint64_t>>      m_submit_history;
	FILE*                                  m_frames     = nullptr;
	FILE*                                  m_draws      = nullptr;
	FILE*                                  m_dispatches = nullptr;
	FILE*                                  m_submits    = nullptr;
};

// GPU thread state.
struct TraceState {
	std::unique_ptr<Writer> writer;
	std::unique_ptr<Frame>  current;
	uint64_t                frame_index      = 0;
	uint64_t                next_burst_ms    = 0;
	uint64_t                bursts_started   = 0;
	uint64_t                burst_remaining  = 0;
	uint64_t                pending_packet   = 0;
};

// Never destroyed: the GPU thread may still report while the process exits. Every frame's rows
// are flushed when written, so nothing is lost with the writer thread left running.
TraceState& GetState() {
	static auto* state = new TraceState();
	return *state;
}

void AppendRaw(Frame& frame, uint32_t queue, uint64_t sequence, uint64_t address,
               std::span<const uint32_t> words, uint32_t kind) {
	struct Header {
		uint32_t magic;
		uint32_t queue;
		uint64_t sequence;
		uint64_t address;
		uint32_t dwords;
		uint32_t kind;
	};
	const Header header {0x4244434bu, queue, sequence, address,
	                     static_cast<uint32_t>(words.size()), kind};
	const auto   offset = frame.raw.size();
	frame.raw.resize(offset + sizeof(header) + words.size_bytes());
	std::memcpy(frame.raw.data() + offset, &header, sizeof(header));
	if (!words.empty()) {
		std::memcpy(frame.raw.data() + offset + sizeof(header), words.data(), words.size_bytes());
	}
}

// Guest command memory is read through the backing view (never faults, whatever the page
// protection), not the guest mapping the command processor reads later. Empty when unavailable.
std::span<const uint32_t> ReadCommandWords(std::span<const uint32_t> words) {
	thread_local std::vector<uint32_t> scratch;
	scratch.resize(words.size());
	if (words.empty() ||
	    !LibKernel::Memory::TryReadBacking(reinterpret_cast<uint64_t>(words.data()), scratch.data(),
	                                       words.size_bytes())) {
		return {};
	}
	return scratch;
}

Frame& CurrentFrame() {
	auto& state = GetState();
	if (state.current == nullptr) {
		state.current        = std::make_unique<Frame>();
		state.current->index = state.frame_index;
		state.current->t_ms  = NowMs();
		state.current->draws.reserve(8192);
	}
	return *state.current;
}

} // namespace

bool Enabled() {
	return GetSettings().enabled;
}

void SetRenderContext(RenderContext* renderer) {
	g_renderer.store(renderer, std::memory_order_release);
}

void HashDraw(const DrawPrep::RegisterSnapshot& registers, const DrawIndexArgs* index_args,
              const DrawAutoArgs* auto_args, bool eligible, const DrawPrep::PreparedDraw& prepared,
              DrawRecord& record) {
	const auto& settings = GetSettings();
	record          = {};
	record.eligible = eligible ? 1u : 0u;
	record.prep_ok  = prepared.ok ? 1u : 0u;
	record.failure  = static_cast<uint8_t>(prepared.failure);
	const auto& ctx  = registers.context;
	const auto& ucfg = registers.user_config;
	const auto& sh   = registers.shaders;
	{
		auto& w = ThreadWords();
		if (index_args != nullptr) {
			record.kind       = 0;
			record.count      = index_args->index_count;
			record.instances  = index_args->instance_count;
			record.index_addr = reinterpret_cast<uint64_t>(index_args->index_addr);
			w.U32(0);
			w.U32(index_args->index_count);
			w.U32(index_args->instance_count);
			w.U32(index_args->index_type_and_size);
			w.E32(index_args->offset_source);
			w.U32(index_args->render_target_slice_offset);
		} else if (auto_args != nullptr) {
			record.kind      = 1;
			record.count     = auto_args->vertex_count;
			record.instances = auto_args->instance_count;
			w.U32(1);
			w.U32(auto_args->vertex_count);
			w.U32(auto_args->instance_count);
			w.U32(0);
			w.E32(auto_args->offset_source);
			w.U32(auto_args->render_target_slice_offset);
		}
		record.h[ArgShape] = w.Hash();
		if (index_args != nullptr) {
			w.U64(record.index_addr);
			w.U32(static_cast<uint32_t>(index_args->base_vertex));
			w.U32(index_args->first_instance);
		} else if (auto_args != nullptr) {
			w.U32(auto_args->first_vertex);
			w.U32(auto_args->first_instance);
		}
		w.U32(ucfg.GetIndexOffset());
		w.U32(ucfg.GetObjectId());
		record.h[Args] = w.Hash();
	}
	record.h[Targets]  = HashTargets(ctx);
	record.h[Viewport] = HashViewport(ctx);
	record.h[State]    = HashState(ctx, ucfg);
	record.h[Shader]   = HashShader(sh);
	record.h[UserSgpr] = HashUserSgpr(sh);
	record.vs_addr     = sh.GetVs().gs_regs.data_addr != 0 ? sh.GetVs().gs_regs.data_addr
	                                                      : sh.GetVs().es_regs.data_addr;
	record.ps_addr     = sh.GetPs().ps_regs.data_addr;
	record.rt0_addr    = ctx.GetRenderTarget(0).base.addr;
	record.depth_addr  = ctx.GetDepthRenderTarget().z_write_base_addr;
	record.hashed      = 1;
	if (!prepared.ok) {
		return;
	}
	{
		auto& w = ThreadWords();
		w.U64(prepared.programs.vertex[0].id);
		w.U64(prepared.programs.vertex[1].id);
		w.U64(prepared.programs.vertex[2].id);
		w.U64(prepared.pixel_active ? prepared.programs.pixel.id : 0u);
		w.B(prepared.pixel_active);
		record.h[Programs] = w.Hash();
	}
	const auto& vi = prepared.vertex_info;
	for (int mask_kind: {0, 1}) {
		auto& w = ThreadWords();
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			PutDescriptors(w, stage.resources.buffers, mask_kind);
		});
		w.U32(static_cast<uint32_t>(vi.resources_num));
		for (int i = 0; i < vi.resources_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
			const auto* f = vi.resources[i].fields;
			w.U32(mask_kind == 1 ? 0u : f[0]);
			w.U32(mask_kind == 1 ? (f[1] & 0xffff0000u) : f[1]);
			w.U32(f[2]);
			w.U32(f[3]);
		}
		record.h[mask_kind == 0 ? VSharp : VShape] = w.Hash();
	}
	{
		auto& w = ThreadWords();
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			PutDescriptors(w, stage.resources.images, 0);
		});
		record.h[TSharp] = w.Hash();
	}
	{
		auto& w = ThreadWords();
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			PutDescriptors(w, stage.resources.samplers, 0);
		});
		record.h[SSharp] = w.Hash();
	}
	{
		auto& w = ThreadWords();
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			w.Span(stage.resources.flattened_srt);
			record.srt_words += static_cast<uint32_t>(stage.resources.flattened_srt.size());
		});
		record.h[Srt] = w.Hash();
	}
	{
		auto& w = ThreadWords();
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			w.Span(stage.resources.user_data);
			record.n_vsharp += static_cast<uint32_t>(stage.resources.buffers.size());
			record.n_tsharp += static_cast<uint32_t>(stage.resources.images.size());
			record.n_ssharp += static_cast<uint32_t>(stage.resources.samplers.size());
		});
		record.h[UserData] = w.Hash();
	}
	if (prepared.reads.Finished()) {
		const auto ranges = prepared.reads.Ranges();
		XXH3_state_t* with_addresses = ThreadHashState(0);
		XXH3_state_t* bytes_only     = ThreadHashState(1);
		for (size_t i = 0; i < ranges.size(); i++) {
			const auto bytes = prepared.reads.RangeBytes(i);
			XXH3_64bits_update(with_addresses, &ranges[i].begin, sizeof(ranges[i].begin));
			XXH3_64bits_update(with_addresses, &ranges[i].end, sizeof(ranges[i].end));
			XXH3_64bits_update(with_addresses, bytes.data(), bytes.size());
			XXH3_64bits_update(bytes_only, bytes.data(), bytes.size());
			record.read_bytes += static_cast<uint32_t>(bytes.size());
		}
		record.read_ranges  = static_cast<uint32_t>(ranges.size());
		record.h[Reads]     = XXH3_64bits_digest(with_addresses);
		record.h[ReadBytes] = XXH3_64bits_digest(bytes_only);
	}
	if (settings.payload) {
		thread_local std::vector<uint8_t> scratch;
		XXH3_state_t*                     state = ThreadHashState(2);
		auto*    renderer = g_renderer.load(std::memory_order_acquire);
		uint64_t budget   = settings.payload_limit;
		ForStages(prepared, [&](const PipelineCache::StagePrep& stage) {
			for (const auto& value: stage.resources.buffers) {
				if (value.dword_count < 4) {
					continue;
				}
				ShaderBufferResource v;
				std::memcpy(v.fields, value.dwords.data(), sizeof(v.fields));
				const auto base = v.Base48();
				auto       size = std::min<uint64_t>(v.GetSize(), settings.payload_per_vs);
				size            = std::min<uint64_t>(size, budget);
				// Content only: the addresses are in the V# hash.
				XXH3_64bits_update(state, &size, sizeof(size));
				if (!GuestRange {base, size}.Valid()) {
					continue;
				}
				if (renderer != nullptr &&
				    (renderer->GetBufferCache().IsRegionGpuModified(base, size) ||
				     renderer->GetBufferCache().HasPendingBackingPublication(base, size))) {
					record.payload_gpu++;
					continue;
				}
				scratch.resize(size);
				if (!LibKernel::Memory::TryReadBacking(base, scratch.data(), size)) {
					continue;
				}
				XXH3_64bits_update(state, scratch.data(), size);
				record.payload_bytes += static_cast<uint32_t>(size);
				budget -= size;
			}
		});
		record.h[Payload] = XXH3_64bits_digest(state);
	}
}

void NoteDrawPacket(const void* packet) {
	GetState().pending_packet = reinterpret_cast<uint64_t>(packet);
}

uint64_t TakeDrawPacket() {
	auto&      state  = GetState();
	const auto packet = state.pending_packet;
	state.pending_packet = 0;
	return packet;
}

void OnDraw(const DrawRecord& record) {
	CurrentFrame().draws.push_back(record);
}

void OnUnpreparedDraw() {
	CurrentFrame().unprepared_draws++;
}

void OnDispatch(uint32_t queue, const HW::ComputeShaderInfo& cs, uint32_t groups_x,
                uint32_t groups_y, uint32_t groups_z, uint32_t mode) {
	DispatchRecord record;
	record.queue     = queue;
	record.cs_addr   = cs.cs_regs.data_addr;
	record.groups[0] = groups_x;
	record.groups[1] = groups_y;
	record.groups[2] = groups_z;
	{
		auto&       w = ThreadWords();
		const auto& r = cs.cs_regs;
		w.U64(r.data_addr);
		w.U32(r.num_thread_x);
		w.U32(r.num_thread_y);
		w.U32(r.num_thread_z);
		w.U32(r.vgprs | (r.priority << 8u) | (r.float_mode << 16u) |
		      (static_cast<uint32_t>(r.wave_size) << 24u));
		w.U32((r.dx10_clamp ? 1u : 0u) | (r.debug_mode ? 2u : 0u) | (r.ieee_mode ? 4u : 0u) |
		      (r.require_forward_progress ? 8u : 0u) | (r.fp16_overflow ? 16u : 0u) |
		      (r.threadgroup_configuration ? 32u : 0u) | (r.scratch_en ? 64u : 0u) |
		      (r.tgid_x_en ? 128u : 0u) | (r.tgid_y_en ? 256u : 0u) | (r.tgid_z_en ? 512u : 0u) |
		      (r.tg_size_en ? 1024u : 0u));
		w.U32(r.user_sgpr | (r.tidig_comp_cnt << 8u) | (r.shared_vgprs << 16u));
		w.U32(r.lds_size);
		w.U32(groups_x);
		w.U32(groups_y);
		w.U32(groups_z);
		w.U32(mode);
		record.h_regs = w.Hash();
	}
	{
		auto& w = ThreadWords();
		PutUserSgpr(w, cs.cs_user_sgpr);
		record.h_user = w.Hash();
	}
	CurrentFrame().dispatches.push_back(record);
}

void OnSubmission(uint32_t queue, uint64_t sequence, std::span<const uint32_t> commands,
                  std::span<const uint32_t> constant_commands) {
	auto& frame = CurrentFrame();
	for (uint32_t kind: {0u, 1u}) {
		const auto words = kind == 0 ? commands : constant_commands;
		if (words.empty()) {
			continue;
		}
		SubmitRecord record;
		record.address   = reinterpret_cast<uint64_t>(words.data());
		record.sequence  = sequence;
		record.dwords    = static_cast<uint32_t>(words.size());
		record.queue     = queue;
		record.kind      = kind;
		const auto bytes = ReadCommandWords(words);
		record.hash      = bytes.empty() ? 0u : XXH3_64bits(bytes.data(), bytes.size_bytes());
		frame.submits.push_back(record);
		if (frame.dump && !bytes.empty()) {
			AppendRaw(frame, queue, sequence, record.address, bytes, kind);
		}
	}
}

void OnIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	auto&        frame = CurrentFrame();
	SubmitRecord record;
	record.address = reinterpret_cast<uint64_t>(commands.data());
	record.dwords  = static_cast<uint32_t>(commands.size());
	record.kind      = chain ? 3u : 2u;
	const auto bytes = ReadCommandWords(commands);
	record.hash      = bytes.empty() ? 0u : XXH3_64bits(bytes.data(), bytes.size_bytes());
	record.queue     = UINT32_MAX;
	frame.submits.push_back(record);
	if (frame.dump && !bytes.empty()) {
		AppendRaw(frame, UINT32_MAX, 0, record.address, bytes, record.kind);
	}
}

void OnFrameBoundary() {
	auto&       state    = GetState();
	const auto& settings = GetSettings();
	if (state.writer == nullptr) {
		state.writer = std::make_unique<Writer>();
		state.next_burst_ms = NowMs() + std::min<uint64_t>(settings.burst_period, 10000u);
	}
	if (state.current != nullptr) {
		state.writer->Push(std::move(state.current));
	}
	state.frame_index++;
	auto& frame = CurrentFrame();
	const auto now = NowMs();
	if (state.burst_remaining == 0 && state.bursts_started < settings.max_bursts &&
	    now >= state.next_burst_ms) {
		state.burst_remaining = settings.burst_frames;
		state.bursts_started++;
		state.next_burst_ms = now + settings.burst_period;
	}
	if (state.burst_remaining != 0) {
		frame.burst = true;
		frame.dump  = settings.dcb_dump && state.bursts_started <= settings.dump_bursts;
		state.burst_remaining--;
	}
}

} // namespace Libs::Graphics::RepeatTrace
