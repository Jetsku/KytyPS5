#include "graphics/shader/shaderCompiler.h"

#include "common/assert.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <iterator>
#include <vector>

// The static part of a program key (PipelineCache's ProgramKey): every stage input field that
// translating a program reads. Programs are cached per key in memory and, through the persistent
// program cache, across runs; this file is therefore part of the codegen source set that the
// codegen version hashes (src/codegen_version.cmake), so a change to a key layout never matches
// entries written with another one.

namespace Libs::Graphics {

void BuildStageStaticKey(const ShaderVertexInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.resources_num < 0 || info.resources_num > ShaderVertexInputInfo::RES_MAX);
	key.clear();
	key.push_back(static_cast<uint32_t>(info.fetch_embedded));
	key.push_back(static_cast<uint32_t>(info.fetch_attrib_reg));
	key.push_back(static_cast<uint32_t>(info.fetch_buffer_reg));
	key.push_back(info.resources_num);
	key.push_back(info.wave_size);
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.pa_cl_vs_out_cntl);
	key.push_back(static_cast<uint32_t>(info.start_instance_sgpr));
	// Per-invocation LDS size (upstream fa8216c4d); only keys with LDS get the word.
	if (info.lds_size_dwords != 0) {
		key.push_back(0x1d500000u | info.lds_size_dwords);
	}
	// PA_CL_UCP planes (upstream feed075a9); only keys with planes get the word, so other keys and
	// shipped pipeline lists stay as before.
	if (info.user_clip_planes != 0) {
		key.push_back(0x55c90000u | info.user_clip_planes);
	}
	// Relocated PS parameters (ShaderLinkVertexPixelParameters). Only keys with aliases get these
	// words, so the keys of every other vertex shader (and the shipped pipeline lists) stay as before;
	// the tag cannot be mistaken for the clip-space flag that follows otherwise.
	if (info.param_alias_mask != 0) {
		key.push_back(0x7A11A5u);
		key.push_back(info.param_alias_mask);
		for (uint32_t destination = 0; destination < 32u; ++destination) {
			if ((info.param_alias_mask & (1u << destination)) != 0) {
				key.push_back(info.param_alias_source[destination]);
			}
		}
	}
	if (info.indirect_vertex_base) {
		// KYTY_INDIRECT_VERTEX_BASE: only when set, so other keys stay as they were.
		key.push_back(0x1B5E0000u | (static_cast<uint32_t>(info.base_vertex_sgpr) & 0xffffu));
	}
	if (info.clip_space.vte_format != 0) {
		// KYTY_VTE_CNTL: only when set, so other keys stay as they were.
		key.push_back(0x7E7E0000u | info.clip_space.vte_format);
	}
	key.push_back(static_cast<uint32_t>(info.clip_space.enabled));
	if (info.clip_space.enabled) {
		for (const float value: info.clip_space.scale) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.offset) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
		for (const float value: info.clip_space.half_extent) {
			key.push_back(std::bit_cast<uint32_t>(value));
		}
	}

	key.push_back(info.mesh.threads_num[0]);
	if (info.mesh.threads_num[0] != 0) {
		const auto& mesh = info.mesh;
		// The translator reads all three workgroup dimensions (PrepareProgram sets y and z to 1).
		key.insert(key.end(), {mesh.threads_num[1], mesh.threads_num[2]});
		key.insert(key.end(), {mesh.wave_size, mesh.host_subgroup_size, mesh.lds_size_dwords,
		                       mesh.scratch_size_dwords, mesh.input_primitive,
		                       mesh.primitives_per_group, mesh.vertices_per_group,
		                       mesh.max_vertices, mesh.max_primitives, mesh.provoking_vertex,
		                       mesh.split_groups, static_cast<uint32_t>(mesh.fast_launch),
		                       mesh.passes});
	}
	key.push_back(info.tess.input_control_points);
	if (info.tess.input_control_points != 0) {
		const auto& tess = info.tess;
		key.insert(key.end(), {tess.output_control_points, tess.ls_stride, tess.hs_stride,
		                       tess.domain, tess.partitioning, tess.output_topology, tess.min_tess_level,
		                       tess.max_tess_level});
		// KYTY_TESS_INDEXED: the patch-constant range both stages declare (0 without the switch).
		if (tess.patch_size != 0u) key.insert(key.end(), {tess.patch_base, tess.patch_size});
		// KYTY_TESS_MERGED_CARRY: the carried VGPR set (empty for every pair without it).
		if (std::ranges::any_of(tess.carried_vgprs, [](uint32_t word) { return word != 0u; }))
			key.insert(key.end(), tess.carried_vgprs.begin(), tess.carried_vgprs.end());
	}

	for (int i = 0; i < info.resources_num; i++) {
		const auto& resource    = info.resources[i];
		const auto& destination = info.resources_dst[i];
		key.push_back(destination.registers_num);
		key.push_back(static_cast<uint32_t>(destination.attr_id));
		key.push_back(resource.fields[3] & 0x7ffffu); // Embedded fetch channels and format.
	}
}

void BuildStageStaticKey(const ShaderPixelInputInfo& info, std::vector<uint32_t>& key) {
	EXIT_IF(info.input_num > std::size(info.interpolator_settings));
	key.clear();
	key.push_back(info.scratch_size_dwords);
	key.push_back(info.input_num);
	key.push_back(info.wave_size);
	key.push_back(info.ps_system_input_base);
	// EXTRA_LDS_SIZE (upstream fa8216c4d); only keys with LDS get the word.
	if (info.lds_size_dwords != 0) {
		key.push_back(0x1d500000u | info.lds_size_dwords);
	}
	key.push_back(info.custom_interpolation_mask);
	key.push_back(info.ps_perspective_center_vgpr);
	key.push_back(info.ps_perspective_centroid_vgpr);
	key.push_back(info.ps_perspective_sample_vgpr);
	key.push_back(info.ps_linear_sample_vgpr);
	key.push_back(info.ps_linear_center_vgpr);
	key.push_back(info.ps_linear_centroid_vgpr);
	key.push_back(static_cast<uint32_t>(info.ps_pos_x));
	key.push_back(static_cast<uint32_t>(info.ps_pos_y));
	key.push_back(static_cast<uint32_t>(info.ps_pos_z));
	key.push_back(static_cast<uint32_t>(info.ps_pos_w));
	// Bit 1: FRONT_FACE_ALL_BITS (0, float +/-1, keeps older keys).
	key.push_back(static_cast<uint32_t>(info.ps_front_face) |
	              (static_cast<uint32_t>(info.ps_front_face_all_bits) << 1u));
	key.push_back(static_cast<uint32_t>(info.ps_ancillary));
	// Bits 8+: the provoking-vertex mode of flat parameters (0, first vertex, keeps older keys).
	key.push_back(static_cast<uint32_t>(info.ps_no_perspective) |
	              (static_cast<uint32_t>(info.parameter_mode) << 8u));
	key.push_back(static_cast<uint32_t>(info.ps_pixel_kill_enable));
	key.push_back(static_cast<uint32_t>(info.ps_depth_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_sample_mask_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_early_z));
	key.push_back(static_cast<uint32_t>(info.dual_source_blending));
	key.push_back(static_cast<uint32_t>(info.alpha_blend_source));
	key.insert(key.end(), std::begin(info.target_output_mode), std::end(info.target_output_mode));
	if (info.target_half_float_mask != 0) {
		// KYTY_FLOAT_STORE_CLAMP: only when set, so other keys stay as they were.
		key.push_back(0x48460000u | info.target_half_float_mask);
	}
	if ((info.target_uint_mask | info.target_sint_mask) != 0) {
		// KYTY_INT_MRT_OUTPUTS (upstream 26ca60923): integer-typed MRT outputs; only when set.
		key.push_back(0x494e0000u | (uint32_t {info.target_sint_mask} << 8u) | info.target_uint_mask);
	}
	if (const auto mask = info.target_shader_mask; mask != UINT32_MAX) {
		// KYTY_CB_SHADER_MASK_EXPORTS: which targets are written decides export locations.
		key.push_back((mask | (mask >> 1u) | (mask >> 2u) | (mask >> 3u)) & 0x11111111u);
	}
	if (info.HasObjprimOutput()) {
		// KYTY_OBJPRIM_ID: only with an ObjPrimID output, so every other key stays as it was.
		key.push_back(0x0b1d0000u | info.objprim_slot |
		              (static_cast<uint32_t>(info.objprim_push_dword) << 4u) |
		              (static_cast<uint32_t>(info.objprim_add_prim_id) << 12u));
	}
	for (uint32_t base = 0; base < info.target_export_mapping.size(); base += 4u) {
		uint32_t packed = 0;
		for (uint32_t i = 0; i < 4u; i++) {
			packed |= static_cast<uint32_t>(info.target_export_mapping[base + i].packed)
			          << (i * 8u);
		}
		key.push_back(packed);
	}
	key.insert(key.end(), std::begin(info.interpolator_settings),
	           std::begin(info.interpolator_settings) + info.input_num);
}

void BuildStageStaticKey(const ShaderComputeInputInfo& info, std::vector<uint32_t>& key) {
	key.clear();
	key.push_back(info.workgroup_register);
	// FLOAT_MODE selects FP32 compare denormal flushing and FP64 support (translator, emitter).
	key.push_back(info.wave_size | (static_cast<uint32_t>(info.float_mode) << 8u));
	key.push_back(info.host_subgroup_size);
	key.push_back(info.thread_ids_num);
	// Bit 31: LDS in the device buffer (KYTY_LDS_DEVICE_BUFFER); sizes stay below 2^16 dwords.
	key.push_back(info.lds_size_dwords | (info.lds_storage ? 0x80000000u : 0u));
	key.push_back(info.scratch_size_dwords);
	// Bit 1: an async compute queue (S_GETREG_B32 HW_ID2 answers only for graphics compute).
	// Bit 2: thread counts read through a device address (KYTY_INDIRECT_THREADS_GPU).
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions) |
	              (static_cast<uint32_t>(info.async_compute) << 1u) |
	              (static_cast<uint32_t>(info.dispatch_dimensions_indirect) << 2u));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
}

} // namespace Libs::Graphics
