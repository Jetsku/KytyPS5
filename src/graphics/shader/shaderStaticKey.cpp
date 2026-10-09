#include "graphics/shader/shaderCompiler.h"

#include "common/assert.h"

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
		                       mesh.split_groups});
	}
	key.push_back(info.tess.input_control_points);
	if (info.tess.input_control_points != 0) {
		const auto& tess = info.tess;
		key.insert(key.end(), {tess.output_control_points, tess.ls_stride, tess.hs_stride,
		                       tess.domain, tess.partitioning, tess.output_topology});
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
	key.push_back(static_cast<uint32_t>(info.ps_front_face));
	key.push_back(static_cast<uint32_t>(info.ps_ancillary));
	key.push_back(static_cast<uint32_t>(info.ps_no_perspective));
	key.push_back(static_cast<uint32_t>(info.ps_pixel_kill_enable));
	key.push_back(static_cast<uint32_t>(info.ps_depth_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_sample_mask_export_enable));
	key.push_back(static_cast<uint32_t>(info.ps_early_z));
	key.push_back(static_cast<uint32_t>(info.dual_source_blending));
	key.push_back(static_cast<uint32_t>(info.alpha_blend_source));
	key.insert(key.end(), std::begin(info.target_output_mode), std::end(info.target_output_mode));
	if (const auto mask = info.target_shader_mask; mask != UINT32_MAX) {
		// KYTY_CB_SHADER_MASK_EXPORTS: which targets are written decides export locations.
		key.push_back((mask | (mask >> 1u) | (mask >> 2u) | (mask >> 3u)) & 0x11111111u);
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
	key.push_back(info.lds_size_dwords);
	key.push_back(info.scratch_size_dwords);
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
}

} // namespace Libs::Graphics
