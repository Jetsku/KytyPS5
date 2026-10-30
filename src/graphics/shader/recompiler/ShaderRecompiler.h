#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/CallSplit.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	// VK_KHR_maintenance8 is enabled (dynamic image sample texel offsets are valid SPIR-V). Only the
	// host's SPIR-V validation reads it; the emitted code does not depend on it.
	bool                        maintenance8_enabled       = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	ShaderStageInputInfo        input_info;
	// Also emit CompileResult::spirv_plain for a GET_LOD_STATS-instrumented pixel shader
	// (KYTY_LOD_STATS_PLAIN_VARIANT).
	bool                        plain_mip_stats_variant = false;
};

// A function an S_SWAPPC_B64 call jumps to, as the dispatch's user data gives it: `address` is the
// value of user SGPRs s[user_sgpr:user_sgpr+1]; the callee returns through s[return_sgpr:+1].
struct CallTarget {
	uint64_t address     = 0;
	uint32_t user_sgpr   = 0;
	uint32_t return_sgpr = 0;
};

// An S_SWAPPC_B64 call whose target comes from a table in guest memory (Insomniac's ray-tracing
// hit callbacks): s_buffer_load_dwordxN s[n:...], <table V#>, <index << record_shift>, then the
// target pair copied from s[n+target_word:+1]. The table V# is reached from user SGPRs through
// `steps`: each loads `dwords` dwords at `offset` from the 64-bit address (S_LOAD) or the buffer
// descriptor (S_BUFFER_LOAD) at dword `base_index` of the previous value (of the user SGPRs from
// `user_sgpr` for the first step). The last step's dwords from `table_index` are the table V#.
struct CallTableStep {
	bool     buffer     = false;
	uint32_t base_index = 0;
	int32_t  offset     = 0;
	uint32_t dwords     = 0;
};

struct CallTable {
	uint32_t                   call_pc      = 0;
	uint32_t                   user_sgpr    = 0;
	uint32_t                   return_sgpr  = 0;
	uint32_t                   record_shift = 0;
	uint32_t                   target_word  = 0;
	uint32_t                   record_offset = 0; // the record load's immediate byte offset
	uint32_t                   table_index  = 0;
	std::vector<CallTableStep> steps;
};

// An S_SWAPPC_B64 call of a program and where its target comes from: user SGPRs
// s[user_sgpr:+1] (user_sgpr != UINT32_MAX), a table (`table`), or neither (not traced).
struct ShaderCall {
	uint32_t                 pc         = 0;
	uint32_t                 source_reg = 0;
	uint32_t                 return_reg = 0;
	uint32_t                 user_sgpr  = UINT32_MAX;
	std::optional<CallTable> table;
	// KYTY_RAY_HIT_CALLS_RETURN: an untraced call of a ray-tracing program that returns at once.
	bool                     null_callee = false;

	[[nodiscard]] bool Traced() const {
		return user_sgpr != UINT32_MAX || table.has_value() || null_callee;
	}
};

struct ShaderCallAnalysis {
	std::vector<ShaderCall> calls;
	std::string             failures; // the calls that are not traced, empty when all are
};

// The calls of a program and the origin of their targets (the traces the S_SWAPPC_B64 skip
// diagnostics log), from the code and the user-data count alone.
[[nodiscard]] ShaderCallAnalysis AnalyzeShaderCalls(std::span<const uint32_t> code,
                                                    const CompileOptions& options);

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
	bool        skip_dispatch = false;
	// A skipped program's first unresolved instruction (pc) and why (diagnostics; UINT32_MAX: none).
	uint32_t    skip_pc = UINT32_MAX;
	std::string skip_reason;
	// Only for a program skipped because it calls through S_SWAPPC_B64 (KYTY_SRT_VARIANT_READS).
	std::vector<CallTarget> call_targets;
	std::vector<CallTable>  call_tables;
	// A call-split planning translation (CallSplit::FlagLiveness or FlagProbe): what it found. Such a
	// translation is skipped (never compiled).
	std::shared_ptr<const CallSplit::Report> call_split;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	// The same program without GET_LOD_STATS feedback (Spirv::EmitProgram mip_stats_records=false):
	// its only difference is the absent per-sample recording. Empty unless requested and the
	// program is instrumented.
	std::vector<uint32_t>  spirv_plain;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
