#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
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

CodegenOptions FromEnvironment() {
	CodegenOptions options;
	options.movrel_range = EnvFlag("KYTY_MOVREL_RANGE", options.movrel_range);
	options.movrel_known_zeros = EnvFlag("KYTY_MOVREL_KNOWN_ZEROS", options.movrel_known_zeros);
	options.movrel_switch      = EnvFlag("KYTY_MOVREL_SWITCH", options.movrel_switch);
	options.uniform_lane_reads = EnvFlag("KYTY_UNIFORM_LANE_READS", options.uniform_lane_reads);
	options.short_f32_helpers  = EnvFlag("KYTY_SHORT_F32_HELPERS", options.short_f32_helpers);
	options.fast_float_min_max = EnvFlag("KYTY_FAST_FMINMAX", options.fast_float_min_max);
	options.fast_pkrtz         = EnvFlag("KYTY_FAST_PKRTZ", options.fast_pkrtz);
	options.single_f2i_saturation =
	    EnvFlag("KYTY_SINGLE_F2I_SATURATION", options.single_f2i_saturation);
	options.lod_stats_gate = EnvFlag("KYTY_LOD_STATS_GATE", options.lod_stats_gate);
	options.robust_buffer_loads = EnvFlag("KYTY_ROBUST_BUFFER_LOADS", options.robust_buffer_loads);
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
	if (const auto* list = std::getenv("KYTY_LOOP_GUARD_SHADERS"); list != nullptr) {
		std::string_view text(list);
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
	options.realtime_clock    = EnvFlag("KYTY_REALTIME_CLOCK", options.realtime_clock);
	options.dpp_skip_inactive = EnvFlag("KYTY_DPP_SKIP_INACTIVE", options.dpp_skip_inactive);
	options.lane_reductions   = EnvFlag("KYTY_LANE_REDUCTIONS", options.lane_reductions);
	options.ir_linear_uses    = EnvFlag("KYTY_IR_LINEAR_USES", options.ir_linear_uses);
	options.fold_lane_masks   = EnvFlag("KYTY_FOLD_LANE_MASKS", options.fold_lane_masks);
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
	options.rt_stub      = EnvFlag("KYTY_RT_STUB", options.rt_stub);
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
	if (const auto* budget = std::getenv("KYTY_RT_NODE_BUDGET");
	    budget != nullptr && budget[0] != '\0') {
		options.rt_node_budget = static_cast<uint32_t>(std::strtoul(budget, nullptr, 0));
	}
	options.rt_node_stats = EnvFlag("KYTY_RT_NODE_STATS", options.rt_node_stats);
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

const CodegenOptions& GetCodegenOptions() {
	return Storage();
}

bool LoopGuardApplies(uint64_t shader_hash) {
	const auto& options = Storage();
	return options.loop_guard_budget != 0 &&
	       std::ranges::find(options.loop_guard_shaders, shader_hash) !=
	           options.loop_guard_shaders.end();
}

void SetCodegenOptions(const CodegenOptions& options) {
	Storage() = options;
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

} // namespace Libs::Graphics::ShaderRecompiler
