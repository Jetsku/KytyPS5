#include "graphics/shader/recompiler/CodegenOptions.h"

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <mutex>
#include <tuple>
#include <utility>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace Libs::Graphics::ShaderRecompiler {
namespace {

// Unset or empty keeps the default; "0" disables; anything else enables.
bool EnvFlag(const char* name, bool default_value) {
	const auto* value = std::getenv(name);
	if (value == nullptr || value[0] == '\0') {
		return default_value;
	}
	return std::strcmp(value, "0") != 0;
}

// The value "nvidia" of a switch that ApplyHostVendor resolves.
bool EnvNamesNvidia(const char* name) {
	const auto* value = std::getenv(name);
	return value != nullptr && std::strcmp(value, "nvidia") == 0;
}

// EnvFlag for those switches: "nvidia" stays off until ApplyHostVendor has seen the device.
bool EnvFlagVendor(const char* name, bool default_value) {
	return !EnvNamesNvidia(name) && EnvFlag(name, default_value);
}

CodegenOptions FromEnvironment() {
	CodegenOptions options;
	options.movrel_range = EnvFlag("KYTY_MOVREL_RANGE", options.movrel_range);
	options.movrel_known_zeros = EnvFlag("KYTY_MOVREL_KNOWN_ZEROS", options.movrel_known_zeros);
	options.movrel_switch      = EnvFlag("KYTY_MOVREL_SWITCH", options.movrel_switch);
	options.uniform_lane_reads = EnvFlag("KYTY_UNIFORM_LANE_READS", options.uniform_lane_reads);
	options.short_f32_helpers  = EnvFlag("KYTY_SHORT_F32_HELPERS", options.short_f32_helpers);
	options.vs_lane_wrap32     = EnvFlag("KYTY_VS_LANE_WRAP32", options.vs_lane_wrap32);
	options.fast_float_min_max = EnvFlag("KYTY_FAST_FMINMAX", options.fast_float_min_max);
	options.native_float_min_max = EnvFlag("KYTY_NATIVE_FMINMAX", options.native_float_min_max);
	options.relaxed_float        = EnvFlag("KYTY_RELAXED_FLOAT", options.relaxed_float);
	options.dcache_invalidate = EnvFlag("KYTY_DCACHE_INV", options.dcache_invalidate);
	options.readfirstlane_exec_zero =
	    EnvFlag("KYTY_READFIRSTLANE_EXEC0", options.readfirstlane_exec_zero);
	options.uniform_buffers      = EnvFlag("KYTY_UNIFORM_BUFFERS", options.uniform_buffers);
	options.wave64_warps         = EnvFlag("KYTY_WAVE64_WARPS", options.wave64_warps);
	options.fast_pkrtz         = EnvFlag("KYTY_FAST_PKRTZ", options.fast_pkrtz);
	options.single_f2i_saturation =
	    EnvFlag("KYTY_SINGLE_F2I_SATURATION", options.single_f2i_saturation);
	options.lod_stats_gate = EnvFlag("KYTY_LOD_STATS_GATE", options.lod_stats_gate);
	options.layer_clamp    = EnvFlag("KYTY_LAYER_CLAMP", options.layer_clamp);
	options.robust_buffer_loads = EnvFlag("KYTY_ROBUST_BUFFER_LOADS", options.robust_buffer_loads);
	options.robust_buffer_stores = EnvFlag("KYTY_ROBUST_BUFFER_STORES", options.robust_buffer_stores);
	options.branchless_loads    = EnvFlag("KYTY_BRANCHLESS_LOADS", options.branchless_loads);
	options.branchless_lds      = EnvFlag("KYTY_BRANCHLESS_LDS", options.branchless_lds);
	options.flat_aperture_bvh   = EnvFlag("KYTY_FLAT_APERTURE_BVH", options.flat_aperture_bvh);
	options.lane_local          = EnvFlag("KYTY_LANE_LOCAL", options.lane_local);
	options.readonly_buffers    = EnvFlag("KYTY_READONLY_BUFFERS", options.readonly_buffers);
	options.native_f16     = EnvFlag("KYTY_NATIVE_F16", options.native_f16);
	options.f32_flush_host = EnvFlagVendor("KYTY_F32_FLUSH_HOST", options.f32_flush_host);
	if (const auto* glc = std::getenv("KYTY_COHERENT_GLC_ONLY"); glc != nullptr && glc[0] != '\0') {
		options.coherent_glc_only = std::strcmp(glc, "2") == 0 ? 2u : std::strcmp(glc, "0") == 0 ? 0u : 1u;
	}
	options.interp_modes = EnvFlag("KYTY_INTERP_MODES", options.interp_modes);
	options.sample_offsets = EnvFlag("KYTY_SAMPLE_OFFSETS", options.sample_offsets);
	options.sample_lod_clamp = EnvFlag("KYTY_SAMPLE_LOD_CLAMP", options.sample_lod_clamp);
	options.host_ftz_inputs  = EnvFlag("KYTY_HOST_FTZ_INPUTS", options.host_ftz_inputs);
	options.exec_selects     = EnvFlag("KYTY_EXEC_SELECTS", options.exec_selects);
	options.ps_append_live_election =
	    EnvFlag("KYTY_PS_APPEND_LIVE_ELECTION", options.ps_append_live_election);
	if (const auto* mode = std::getenv("KYTY_PS_LIVE_EXEC"); mode != nullptr && mode[0] != '\0') {
		options.ps_live_exec = std::strcmp(mode, "all") == 0 ? PsLiveExec::All
		                       : std::strcmp(mode, "0") == 0 ? PsLiveExec::Off
		                                                     : PsLiveExec::AppendConsume;
	}
	if (const auto* budget = std::getenv("KYTY_LOOP_GUARD"); budget != nullptr) {
		options.loop_guard_budget = static_cast<uint32_t>(std::strtoul(budget, nullptr, 0));
	}
	options.ps_helper_atomics_skip =
	    EnvFlag("KYTY_PS_HELPER_ATOMICS_SKIP", options.ps_helper_atomics_skip);
	options.volatile_loads         = EnvFlag("KYTY_VOLATILE_LOADS", options.volatile_loads);
	options.indirect_coherent = EnvFlag("KYTY_INDIRECT_COHERENT", options.indirect_coherent);
	options.buffer_pair_atomic =
	    EnvFlag("KYTY_BUFFER_PAIR_ATOMIC", options.buffer_pair_atomic);
	options.indirect_format_compact =
	    EnvFlag("KYTY_INDIRECT_FORMAT_COMPACT", options.indirect_format_compact);
	options.indirect_format_inline =
	    EnvFlag("KYTY_INDIRECT_FORMAT_INLINE", options.indirect_format_inline);
	if (const char* fast = std::getenv("KYTY_INDIRECT_FORMAT_FAST"); fast != nullptr) {
		options.indirect_format_fast = fast[0] != '\0' && std::strcmp(fast, "0") != 0;
	}
	options.cfg_clone_exits   = EnvFlag("KYTY_CFG_CLONE_EXITS", options.cfg_clone_exits);
	options.cfg_clone_regions = EnvFlag("KYTY_CFG_CLONE_REGIONS", options.cfg_clone_regions);
	options.indirect_raw_outline = EnvFlag("KYTY_INDIRECT_RAW_OUTLINE", options.indirect_raw_outline);
	options.huge_helper_outline  = EnvFlag("KYTY_HUGE_HELPER_OUTLINE", options.huge_helper_outline);
	options.null_buffer_share    = EnvFlag("KYTY_NULL_BUFFER_SHARE", options.null_buffer_share);
	options.bounded_key_images   = EnvFlag("KYTY_BOUNDED_KEY_IMAGES", options.bounded_key_images);
	options.drop_graphics_indirect_stores =
	    EnvFlag("KYTY_DROP_GRAPHICS_INDIRECT_STORES", options.drop_graphics_indirect_stores);
	options.indirect_format_stores = EnvFlag("KYTY_INDIRECT_FORMAT_STORES", options.indirect_format_stores);
	options.graphics_bda_writes    = EnvFlag("KYTY_GRAPHICS_BDA_WRITES", options.graphics_bda_writes);
	options.indirect_format_stores_log =
	    EnvFlag("KYTY_INDIRECT_FORMAT_STORES_LOG", options.indirect_format_stores_log);
	options.r128_table_images    = EnvFlag("KYTY_R128_TABLE_IMAGES", options.r128_table_images);
	options.indirect_format_d16  = EnvFlag("KYTY_INDIRECT_FORMAT_D16", options.indirect_format_d16);
	options.heap_key_mask     = EnvFlag("KYTY_HEAP_KEY_MASK", options.heap_key_mask);
	options.wave64_active_zero =
	    EnvFlag("KYTY_WAVE64_ACTIVE_ZERO", options.wave64_active_zero);
	options.lane2_share = EnvFlag("KYTY_LANE2_SHARE", options.lane2_share);
	options.known_bits_masks = EnvFlag("KYTY_KNOWN_BITS_MASKS", options.known_bits_masks);
	options.gds_m0_base      = EnvFlag("KYTY_GDS_M0_BASE", options.gds_m0_base);
	options.lds_atomic_order = EnvFlag("KYTY_LDS_ATOMIC_ORDER", options.lds_atomic_order);
	options.split_lds_write_sync = EnvFlag("KYTY_SPLIT_LDS_WRITE_SYNC", options.split_lds_write_sync);
	options.storage_table_loose  = EnvFlag("KYTY_STORAGE_TABLE_LOOSE", options.storage_table_loose);
	options.sampled_table_reads  = EnvFlag("KYTY_SAMPLED_TABLE_READS", options.sampled_table_reads);
	options.scalar_load_groups   = EnvFlag("KYTY_SCALAR_LOAD_GROUPS", options.scalar_load_groups);
	options.srt_uniform          = EnvFlag("KYTY_SRT_UNIFORM", options.srt_uniform);
	options.null_image_keep_dim  = EnvFlag("KYTY_NULL_IMAGE_KEEP_DIM", options.null_image_keep_dim);
	options.storage_swizzle_canon = EnvFlag("KYTY_STORAGE_SWIZZLE_CANON", options.storage_swizzle_canon);
	options.raw_small_sint_storage =
	    EnvFlag("KYTY_RAW_SMALL_SINT_STORAGE", options.raw_small_sint_storage);
	options.point_integer_images = EnvFlag("KYTY_POINT_INTEGER_IMAGES", options.point_integer_images);
	options.compare_sampler_split = EnvFlag("KYTY_COMPARE_SAMPLER_SPLIT", options.compare_sampler_split);
	options.waterfall_active_leader = EnvFlag("KYTY_WATERFALL_ACTIVE_LEADER", options.waterfall_active_leader);
	options.wave64_present_exec = EnvFlag("KYTY_WAVE64_PRESENT_EXEC", options.wave64_present_exec);
	options.embedded_fetch_no_sgpr_offset = EnvFlag("KYTY_EMBEDDED_FETCH_NO_SGPR_OFFSET", options.embedded_fetch_no_sgpr_offset);
	options.mov_no_float_mods = EnvFlag("KYTY_MOV_NO_FLOAT_MODS", options.mov_no_float_mods);
	options.pops_interlock = EnvFlag("KYTY_POPS_INTERLOCK", options.pops_interlock);
	options.barrier_acquire_stores =
	    EnvFlag("KYTY_BARRIER_ACQUIRE_STORES", options.barrier_acquire_stores);
	options.format_missing_canonical =
	    EnvFlag("KYTY_FORMAT_MISSING_CANON", options.format_missing_canonical);
	options.indirect_format_all = EnvFlag("KYTY_INDIRECT_FORMAT_ALL", options.indirect_format_all);
	options.format_store_dst_sel = EnvFlag("KYTY_FORMAT_STORE_DST_SEL", options.format_store_dst_sel);
	options.image_glc_coherent = EnvFlag("KYTY_IMAGE_GLC_COHERENT", options.image_glc_coherent);
	options.implicit_lod_stages = EnvFlag("KYTY_IMPLICIT_LOD_STAGES", options.implicit_lod_stages);
	options.buffer_oob_modes = EnvFlag("KYTY_BUFFER_OOB_MODES", options.buffer_oob_modes);
	options.srgb8_decode = EnvFlag("KYTY_SRGB8_DECODE", options.srgb8_decode);
	options.depth_bits_read = EnvFlag("KYTY_DEPTH_BITS_READ", options.depth_bits_read);
	options.bindless_3d_as_array =
	    EnvFlag("KYTY_BINDLESS_3D_AS_ARRAY", options.bindless_3d_as_array);
	options.runtime_images   = EnvFlag("KYTY_RUNTIME_IMAGES", options.runtime_images);
	options.runtime_int_images =
	    options.runtime_images && EnvFlag("KYTY_RUNTIME_INT_IMAGES", options.runtime_int_images);
	options.bindless_strided = EnvFlag("KYTY_BINDLESS_STRIDE", options.bindless_strided);
	options.bindless_samplers = EnvFlag("KYTY_BINDLESS_SAMPLERS", options.bindless_samplers);
	options.workgroup_images = EnvFlag("KYTY_WORKGROUP_IMAGES", options.workgroup_images);
	options.table_key_bounds = EnvFlag("KYTY_TABLE_KEY_BOUNDS", options.table_key_bounds);
	{
		const auto selector_bit = [&](const char* name, uint8_t bit) {
			if (EnvFlag(name, (options.selector_keys & bit) != 0u)) {
				options.selector_keys |= bit;
			} else {
				options.selector_keys &= static_cast<uint8_t>(~bit);
			}
		};
		selector_bit("KYTY_SELECTOR_BITSCAN", SelectorKeysBitscan);
		selector_bit("KYTY_SELECTOR_EXCLUDED", SelectorKeysExcluded);
		selector_bit("KYTY_SELECTOR_MASK_RANKS", SelectorKeysMaskRanks);
		selector_bit("KYTY_SELECTOR_SHARED_LIST", SelectorKeysSharedList);
		selector_bit("KYTY_SELECTOR_BITSET", SelectorKeysBitset);
	}
	options.native_wave64_branch =
	    EnvFlag("KYTY_NATIVE_WAVE64_BRANCH", options.native_wave64_branch);
	options.sampler_reads_deferred =
	    EnvFlag("KYTY_SAMPLER_READS_DEFER", options.sampler_reads_deferred);
	options.call_trace_reach = EnvFlag("KYTY_CALL_TRACE_REACH", options.call_trace_reach);
	if (const auto* cap = std::getenv("KYTY_CALL_LOOP_CAP"); cap != nullptr && cap[0] != '\0') {
		options.call_loop_cap = static_cast<uint32_t>(std::strtoul(cap, nullptr, 0));
	}
	options.call_loops           = EnvFlag("KYTY_CALL_LOOPS", options.call_loops);
	options.call_waterfall_clear = EnvFlag("KYTY_CALL_WATERFALL_CLEAR", options.call_waterfall_clear);
	options.storage_length_zero =
	    EnvFlag("KYTY_STORAGE_LENGTH_ZERO", options.storage_length_zero);
	if (const auto* list = std::getenv("KYTY_BDA_FORCE_SHADERS"); list != nullptr) {
		std::string_view text(list);
		while (!text.empty()) {
			const auto comma = text.find(',');
			const auto token = std::string(text.substr(0, comma));
			if (!token.empty()) {
				options.bda_force_shaders.push_back(std::strtoull(token.c_str(), nullptr, 16));
			}
			if (comma == std::string_view::npos) {
				break;
			}
			text.remove_prefix(comma + 1);
		}
	}
	if (const auto* list = std::getenv("KYTY_INDIRECT_FORMAT_STORES_SHADERS"); list != nullptr) {
		std::string_view text(list);
		while (!text.empty()) {
			const auto comma = text.find(',');
			const auto token = std::string(text.substr(0, comma));
			if (!token.empty()) {
				options.indirect_format_stores_shaders.push_back(
				    std::strtoull(token.c_str(), nullptr, 16));
			}
			if (comma == std::string_view::npos) {
				break;
			}
			text.remove_prefix(comma + 1);
		}
	}
	if (const auto* list = std::getenv("KYTY_LOOP_GUARD_SHADERS"); list != nullptr) {
		options.loop_guard_all = std::strcmp(list, "all") == 0 || std::strcmp(list, "*") == 0;
		std::string_view text(options.loop_guard_all ? "" : list);
		while (!text.empty()) {
			const auto comma = text.find(',');
			const auto token = std::string(text.substr(0, comma));
			if (!token.empty()) {
				options.loop_guard_shaders.push_back(std::strtoull(token.c_str(), nullptr, 16));
			}
			if (comma == std::string_view::npos) {
				break;
			}
			text.remove_prefix(comma + 1);
		}
	}
	options.srt_variant_reads = EnvFlag("KYTY_SRT_VARIANT_READS", options.srt_variant_reads);
	options.srt_image_reach   = EnvFlag("KYTY_SRT_IMAGE_REACH", options.srt_image_reach);
	options.shader_calls      = EnvFlag("KYTY_SHADER_CALLS", options.shader_calls);
	options.call_trace_wave32 = EnvFlag("KYTY_CALL_TRACE_WAVE32", options.call_trace_wave32);
	options.realtime_clock   = EnvFlag("KYTY_REALTIME_CLOCK", options.realtime_clock);
	options.dpp_skip_inactive = EnvFlag("KYTY_DPP_SKIP_INACTIVE", options.dpp_skip_inactive);
	options.split_dpp_lane    = EnvFlag("KYTY_SPLIT_DPP_LANE", options.split_dpp_lane);
	options.mov_int_mods      = EnvFlag("KYTY_MOV_INT_MODS", options.mov_int_mods);
	options.mad_flush         = EnvFlag("KYTY_MAD_FLUSH", options.mad_flush);
	options.f16_exact_round   = EnvFlag("KYTY_F16_EXACT_ROUND", options.f16_exact_round);
	options.movrel_full_m0    = EnvFlag("KYTY_MOVREL_FULL_M0", options.movrel_full_m0);
	options.omod_zero         = EnvFlag("KYTY_OMOD_ZERO", options.omod_zero);
	options.cvt_f32_f64_inf   = EnvFlag("KYTY_CVT_F32_F64_INF", options.cvt_f32_f64_inf);
	options.bfm_defined       = EnvFlag("KYTY_BFM_DEFINED", options.bfm_defined);
	options.wave64_high_lanes = EnvFlag("KYTY_WAVE64_HIGH_LANES", options.wave64_high_lanes);
	options.lane_reductions   = EnvFlag("KYTY_LANE_REDUCTIONS", options.lane_reductions);
	options.ir_linear_uses    = EnvFlag("KYTY_IR_LINEAR_USES", options.ir_linear_uses);
	options.fold_lane_masks   = EnvFlag("KYTY_FOLD_LANE_MASKS", options.fold_lane_masks);
	options.vs_full_wave      = EnvFlagVendor("KYTY_VS_FULL_WAVE", options.vs_full_wave);
	options.int_mrt_outputs   = EnvFlag("KYTY_INT_MRT_OUTPUTS", options.int_mrt_outputs);
	options.ufloat_store_rtz = EnvFlag("KYTY_UFLOAT_STORE_RTZ", options.ufloat_store_rtz);
	if (const auto* cap = std::getenv("KYTY_DISPATCHER_CAP"); cap != nullptr && cap[0] != '\0') {
		options.dispatcher_cap = static_cast<uint32_t>(std::strtoul(cap, nullptr, 0));
	}
	// KYTY_NATIVE_INDIRECT_MESH=1|on|verify|exit, or unset/empty (the default, renderer/meshIndirect.h:
	// GPU-converted indirect mesh draws); 0 and "empty" keep the pushed-dword-only mesh draw
	// parameters.
	{
		const auto* mode = std::getenv("KYTY_NATIVE_INDIRECT_MESH");
		options.mesh_indirect_params =
		    mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "1") == 0 ||
		    std::strcmp(mode, "on") == 0 || std::strcmp(mode, "verify") == 0 ||
		    std::strcmp(mode, "exit") == 0;
	}
	{
		const auto* mode        = std::getenv("KYTY_BDA_ACCESS_TAGS");
		options.bda_access_tags = mode != nullptr && mode[0] != '\0' && std::strcmp(mode, "0") != 0;
		options.bda_tags_gpu_count = options.bda_access_tags && EnvFlag("KYTY_BDA_TAGS_GPU_COUNT", false);
	}
	options.rt_stub      = EnvFlag("KYTY_RT_STUB", options.rt_stub);
	options.fract_clamp  = EnvFlag("KYTY_FRACT_CLAMP", options.fract_clamp);
	if (const auto* mode = std::getenv("KYTY_RT_SOFTWARE");
	    mode != nullptr && mode[0] != '\0' && std::strcmp(mode, "auto") != 0) {
		options.rt_software      = std::strcmp(mode, "0") != 0;
		options.rt_software_auto = false;
	}
	if (const auto* mode = std::getenv("KYTY_RT_FUNCTION"); mode != nullptr && mode[0] != '\0') {
		options.rt_function = std::min<uint32_t>(static_cast<uint32_t>(std::strtoul(mode, nullptr, 10)), 2u);
	}
	options.rt_type6     = EnvFlag("KYTY_RT_TYPE6", options.rt_type6);
	options.bda_writes        = EnvFlag("KYTY_BDA_WRITES", options.bda_writes);
	options.bda_store_log     = EnvFlag("KYTY_BDA_STORE_LOG", options.bda_store_log);
	if (const auto* budget = std::getenv("KYTY_RT_NODE_BUDGET");
	    budget != nullptr && budget[0] != '\0') {
		options.rt_node_budget = static_cast<uint32_t>(std::strtoul(budget, nullptr, 0));
	}
	options.rt_node_stats = EnvFlag("KYTY_RT_NODE_STATS", options.rt_node_stats);
	options.rt_hw         = EnvFlag("KYTY_RT_HW", options.rt_hw);
	options.rt_hw_verify  = options.rt_hw && EnvFlag("KYTY_RT_HW_VERIFY", options.rt_hw_verify);
	if (const auto* mode = std::getenv("KYTY_LDS_WAITCNT_BARRIER"); mode != nullptr && mode[0] != '\0') {
		options.lds_waitcnt_barrier = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                                  ? LdsWaitcntBarrier::Off
		                              : std::strcmp(mode, "workgroup") == 0 ? LdsWaitcntBarrier::Workgroup
		                                                                    : LdsWaitcntBarrier::Subgroup;
	}
	if (const auto* mode = std::getenv("KYTY_WAVE32_CLUSTERS"); mode != nullptr && mode[0] != '\0') {
		options.wave32_clusters = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                              ? Wave32Clusters::Off
		                          : std::strcmp(mode, "force") == 0 ? Wave32Clusters::Force
		                                                            : Wave32Clusters::Auto;
	}
	if (const auto* mode = std::getenv("KYTY_LDS_DEVICE_BUFFER"); mode != nullptr && mode[0] != '\0') {
		options.lds_device_buffer = std::strcmp(mode, "0") == 0 || std::strcmp(mode, "off") == 0
		                                ? LdsDeviceBuffer::Off
		                            : std::strcmp(mode, "force") == 0 ? LdsDeviceBuffer::Force
		                                                              : LdsDeviceBuffer::Auto;
	}
	if (const auto* limit = std::getenv("KYTY_LDS_LIMIT_OVERRIDE"); limit != nullptr && limit[0] != '\0') {
		options.lds_limit_override = static_cast<uint32_t>(std::strtoul(limit, nullptr, 0));
	}
	options.ps_per_vertex = EnvFlag("KYTY_PS_PER_VERTEX", options.ps_per_vertex);
	options.clip_guard    = EnvFlag("KYTY_CLIP_GUARD", options.clip_guard);
	options.lds_slots     = EnvFlagVendor("KYTY_LDS_SLOTS", options.lds_slots);
	options.float_store_clamp = EnvFlag("KYTY_FLOAT_STORE_CLAMP", options.float_store_clamp);
	options.vte_cntl          = EnvFlag("KYTY_VTE_CNTL", options.vte_cntl);
	options.snapshot_written_guard =
	    EnvFlag("KYTY_SNAPSHOT_WRITTEN_GUARD", options.snapshot_written_guard);
	options.tess_indexed = EnvFlag("KYTY_TESS_INDEXED", options.tess_indexed);
	options.tess_single_cp = EnvFlag("KYTY_TESS_SINGLE_CP", options.tess_single_cp);
	options.ps_exec_words = EnvFlag("KYTY_PS_EXEC_WORDS", options.ps_exec_words);
	// "off" (or "0") skips that stage's shader-side denormal flushes; unset, "exact" or anything
	// else keeps them.
	const auto flush_off = [](const char* name) {
		const auto* value = std::getenv(name);
		return value != nullptr && (std::strcmp(value, "off") == 0 || std::strcmp(value, "0") == 0);
	};
	options.denorm_flush_skip =
	    static_cast<uint8_t>((flush_off("KYTY_PS_FLUSH") ? DenormFlushSkipPixel : 0u) |
	                         (flush_off("KYTY_CS_FLUSH") ? DenormFlushSkipCompute : 0u) |
	                         (flush_off("KYTY_VS_FLUSH") ? DenormFlushSkipGeometry : 0u));
	if (const auto* mode = std::getenv("KYTY_MAD_MODE"); mode != nullptr) {
		if (std::strcmp(mode, "exact") == 0) {
			options.mad_mode = MadMode::Exact;
		} else if (std::strcmp(mode, "fused") == 0) {
			options.mad_mode = MadMode::Fused;
		} else if (std::strcmp(mode, "position") == 0) {
			options.mad_mode = MadMode::Position;
		}
	}
	return options;
}

CodegenOptions& Storage() {
	static CodegenOptions options = FromEnvironment();
	return options;
}

} // namespace

namespace {
std::atomic<bool> g_host_compute_subgroups_32 {false};
} // namespace

void SetHostComputeSubgroups32(bool fixed_32) {
	g_host_compute_subgroups_32.store(fixed_32, std::memory_order_relaxed);
}

bool GetHostComputeSubgroups32() {
	return g_host_compute_subgroups_32.load(std::memory_order_relaxed);
}

namespace {
std::atomic<bool> g_host_uniform_buffer_robust {false};
} // namespace

void SetHostUniformBufferRobust(bool robust) {
	g_host_uniform_buffer_robust.store(robust, std::memory_order_relaxed);
}

bool GetHostUniformBufferRobust() {
	return g_host_uniform_buffer_robust.load(std::memory_order_relaxed);
}

namespace {
std::atomic<bool> g_host_native_half {false};
} // namespace

void SetHostNativeHalf(bool supported) {
	g_host_native_half.store(supported, std::memory_order_relaxed);
}

bool GetHostNativeHalf() {
	return g_host_native_half.load(std::memory_order_relaxed);
}

bool NativeHalfActive() {
	return Storage().native_f16 && GetHostNativeHalf();
}

const CodegenOptions& GetCodegenOptions() {
	return Storage();
}

bool LoopGuardApplies(uint64_t shader_hash) {
	const auto& options = Storage();
	return options.loop_guard_budget != 0 &&
	       (options.loop_guard_all || std::ranges::find(options.loop_guard_shaders, shader_hash) !=
	                                      options.loop_guard_shaders.end());
}

namespace {
MadExactFilter g_mad_exact_filter;
} // namespace

void SetMadExactFilter(MadExactFilter filter) {
	g_mad_exact_filter = std::move(filter);
}

bool MadExactProgram(uint32_t stage, uint64_t shader_hash) {
	const auto& filter = g_mad_exact_filter;
	if (!filter.active || stage >= 32 || (filter.stage_mask & (1u << stage)) == 0) {
		return false;
	}
	if (filter.hashes.empty() && filter.hash_mask == 0) {
		return true;
	}
	return std::ranges::find(filter.hashes, shader_hash) != filter.hashes.end() ||
	       (filter.hash_mask != 0 && (shader_hash & filter.hash_mask) == filter.hash_value);
}

void SetCodegenOptions(const CodegenOptions& options) {
	Storage() = options;
}

namespace {
bool g_f32_flush_host_self_test = false;
} // namespace

void ApplyHostVendor(uint32_t vendor_id) {
	constexpr uint32_t NvidiaVendorId = 0x10deu;
	const bool         nvidia         = vendor_id == NvidiaVendorId;
	auto&              options        = Storage();
	const auto         apply          = [nvidia, vendor_id](const char* name, bool& option, bool deferred) {
		if (!EnvNamesNvidia(name)) {
			return false;
		}
		option = nvidia;
		// On the console, with the other "Kyty GPU capabilities:" lines of a player's log (a
		// deferred switch's line comes with its self-test verdict).
		if (!(deferred && nvidia)) {
			std::printf("Kyty GPU capabilities: %s=nvidia is %s (device vendor 0x%04x)\n", name,
			            nvidia ? "on" : "off", vendor_id);
		}
		return nvidia;
	};
	g_f32_flush_host_self_test = apply("KYTY_F32_FLUSH_HOST", options.f32_flush_host, true);
	apply("KYTY_VS_FULL_WAVE", options.vs_full_wave, false);
	apply("KYTY_LDS_SLOTS", options.lds_slots, false);
}

bool F32FlushHostAwaitsSelfTest() {
	return g_f32_flush_host_self_test;
}

void SettleF32FlushHost(uint8_t classes) {
	Storage().f32_flush_host         = (classes & F32FlushClassAll) != 0u;
	Storage().f32_flush_host_classes =
	    static_cast<uint8_t>(classes & (F32FlushClassAll | F32FlushClassNoPreserve16));
	g_f32_flush_host_self_test       = false;
}

namespace {
thread_local int g_rt_function_override = -1;
} // namespace

uint32_t RtFunctionMode() {
	return g_rt_function_override >= 0 ? static_cast<uint32_t>(g_rt_function_override)
	                                   : Storage().rt_function;
}

void SetThreadRtFunctionOverride(int mode) {
	g_rt_function_override = mode;
}

void NoteFormattedSelectorMissing(uint64_t shader_hash, uint32_t format, uint32_t swizzle,
                                  const char* where) {
	// Counted per compiled formatted channel; the first 32 distinct (program, format, dst_sel)
	// combinations are named.
	static std::atomic<uint64_t> total {0};
	static std::atomic<uint32_t> named {0};
	static std::mutex            mutex;
	static std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> seen;
	const auto count = total.fetch_add(1, std::memory_order_relaxed) + 1;
	std::scoped_lock lock(mutex);
	const auto key = std::make_tuple(shader_hash, format, swizzle);
	if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
		return;
	}
	if (named.load(std::memory_order_relaxed) >= 32u) {
		if ((count & (count - 1u)) == 0u) {
			LOGF("Format: dst_sel past the format's channels, %" PRIu64 " formatted channels so far\n",
			     count);
		}
		return;
	}
	seen.push_back(key);
	named.fetch_add(1, std::memory_order_relaxed);
	LOGF("Format: %s dst_sel 0x%03x names a channel format %u lacks (program 0x%016" PRIx64
	     ", %s; %" PRIu64 " formatted channels so far)\n",
	     where, swizzle & 0xfffu, format, shader_hash,
	     GetCodegenOptions().format_missing_canonical ? "reads 0/1" : "wraps", count);
}

ComputeLdsPlan PlanComputeLds(uint32_t requested_dwords, uint32_t device_limit_bytes) {
	const auto& options = Storage();
	ComputeLdsPlan plan;
	plan.limit_bytes = options.lds_limit_override != 0
	                       ? std::min(device_limit_bytes, options.lds_limit_override)
	                       : device_limit_bytes;
	const auto limit_dwords = plan.limit_bytes / 4u;
	plan.dwords             = requested_dwords;
	switch (options.lds_device_buffer) {
		case LdsDeviceBuffer::Force: plan.storage = requested_dwords != 0; break;
		case LdsDeviceBuffer::Auto: plan.storage = requested_dwords > limit_dwords; break;
		case LdsDeviceBuffer::Off: break;
	}
	if (!plan.storage && requested_dwords > limit_dwords) {
		plan.dwords  = limit_dwords;
		plan.clamped = true;
	}
	return plan;
}

} // namespace Libs::Graphics::ShaderRecompiler
