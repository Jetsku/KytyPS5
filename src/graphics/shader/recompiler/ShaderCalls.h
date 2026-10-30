#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERCALLS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERCALLS_H_

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

// Programs that call functions through S_SWAPPC_B64 (KYTY_SHADER_CALLS, default on): the calls are
// traced (AnalyzeShaderCalls), their targets read from the dispatch's user data or the call tables
// in guest memory, and the callees expanded into the program (Decoder::ExpandShaderCalls), which
// is then compiled like any other.
namespace Libs::Graphics::ShaderRecompiler {

// A call table as the dispatch's user data reaches it: `bytes` of records of `record_bytes` from
// guest address `base`; each record's target pair is at byte `target_offset`.
struct CallTableView {
	uint64_t base          = 0;
	uint64_t bytes         = 0;
	uint32_t record_bytes  = 0;
	uint32_t target_offset = 0;

	bool operator==(const CallTableView&) const = default;
};

constexpr uint64_t MaxCallTableBytes   = 16ull << 20u;
constexpr size_t   MaxCallSiteTargets  = 256;
constexpr size_t   MaxExpansionsPerProgram = 8;

// Follows the table's load steps from the user data. False with `failure` when a step reads
// outside the loaded values or unreadable memory, or the table is not a usable buffer.
bool EvaluateCallTable(const CallTable& table, std::span<const uint32_t> user_data,
                       uint32_t user_data_base, const Decoder::ShaderCodeReader& read,
                       CallTableView& view, std::string& failure);

// The table's words (at most MaxCallTableBytes).
bool ReadCallTable(const CallTableView& view, const Decoder::ShaderCodeReader& read,
                   std::vector<uint32_t>& words, std::string& failure);
// The distinct target pairs of the table's records with a nonzero address, as the program compares
// them (the raw 64-bit values), added to `targets`.
void CallTableTargets(const CallTableView& view, std::span<const uint32_t> words,
                      std::set<uint64_t>& targets);

// Expands the calls of compute programs for the targets a dispatch can reach: each dispatch
// evaluates them from user data and the table descriptors (a few small reads). A table is read
// when its descriptor is new and every `rescan_interval` dispatches (and parsed only when its
// digest changed). A program keeps one expansion per target set (at most MaxExpansionsPerProgram,
// least recently used out), so tables that alternate do not expand it again; an expansion is
// dropped when the callee code it placed changes (checked every `rescan_interval` dispatches). A
// target that leaves the set between two scans takes the skip path of the expansion.
// KYTY_SHADER_CALLS_STICKY (default on): ShaderCallExpander::sticky.
[[nodiscard]] bool StickyExpansionsDefault();

// KYTY_CALL_PARTITIONS=1 (default off; on in the GT7 / PPSA03671 presets): a compute program whose
// one call is a table call with at least KYTY_CALL_PARTITION_MIN targets (default 16), and whose
// code outside the call cannot see lanes leave the wave (no LDS, barriers, lane reads, DPP,
// implicit derivatives, atomics or messages), is dispatched once per target ("partition"), each
// dispatch's program inlining that one target. Lanes calling another target report it (an open
// hash set in a small device buffer the renderer reads back) and leave the wave, so each lane runs
// its material in exactly one dispatch. A program none of whose targets is known yet runs one
// "probe" partition that only reports. Targets stay known (KYTY_CALL_PARTITION_MAX, default 64,
// partitions per dispatch). After usa-ratt/KytyPS5-GT7 rounds 99 and 118 (799e327cc, 3d2a02104):
// GT7's material passes call one of about 300 functions; one program with all of them does not
// compile in minutes, and our split dispatch needs wave64 one-wave workgroups.
[[nodiscard]] bool     CallPartitionsEnabled();
[[nodiscard]] uint32_t CallPartitionMinTargets();
[[nodiscard]] uint32_t CallPartitionMaxParts();
// KYTY_CALL_PARTITION_MEMORY=1 (default off; on in the GT7 / PPSA03671 presets): the targets each
// partitioned shader (by hash) reported are remembered in a file (_PipelineCache/<title>.partitions,
// "shader_hash target" in hex per line, appended as they are reported). The next start makes the
// remembered targets that the shader's table still holds known at its first dispatch, so their
// partitions (cached programs by then) run from the start instead of after a probe and readbacks.
[[nodiscard]] bool CallPartitionMemoryEnabled();
// KYTY_CALL_LEARN=1 (default off; adapted from usa-ratt/KytyPS5-GT7 8b76a992a, which learns
// table variants from the GPU's miss reports): a compute program whose one call is a table call
// with more targets than an expansion places (over MaxCallSiteTargets, or over
// KYTY_SHADER_CALLS_TOPK when that is set) and that is not partitioned is expanded with the targets
// its dispatches reported missing, at most KYTY_CALL_VARIANTS (default 32), in the order they were
// reported. Every other target reports itself into the program's report set (the partitions' set
// and readback, CallStateOp::MissReport) and takes the skip path as before; a new report makes the
// next dispatch use an expansion with that target too. Off: such a table call does not expand
// ("more than the bound") or keeps its TOPK targets, as before.
[[nodiscard]] bool     CallLearnEnabled();
[[nodiscard]] uint32_t CallLearnVariants();
// Why the code outside the call keeps a program from being partitioned (empty when it does not).
[[nodiscard]] std::string CallPartitionUnsafe(std::span<const uint32_t> code);

struct CallPartitionPart {
	std::shared_ptr<const std::vector<uint32_t>> code;
	uint64_t                                     digest = 0;
	uint64_t                                     target = 0; // 0: the probe partition
};
struct CallPartitionRun {
	uint64_t                       program     = 0; // report-set key (ShaderCallExpander program)
	uint64_t                       shader_hash = 0;
	std::vector<CallPartitionPart> parts;
};

class ShaderCallExpander {
public:
	struct Expanded {
		std::shared_ptr<const std::vector<uint32_t>> code;
		uint64_t                                     digest = 0; // XXH3-64 of the code
		// Where each placed callee body is in `code`.
		std::shared_ptr<const std::vector<Decoder::ShaderCallBody>> bodies;
	};

	// A callee whose body keeps an expansion from translating (diagnosed after the translation of
	// an expansion): it is left out of every later expansion, its calls take the skip path.
	void DropTarget(uint64_t address, std::string_view why);

	// False (logged once per program) when the program does not expand: it is compiled as is.
	bool Expand(std::span<const uint32_t> code, uint64_t base, const CompileOptions& options,
	            const Decoder::ShaderCodeReader& read, Expanded& expanded);

	// KYTY_CALL_PARTITIONS: false when the program is not partitioned (then Expand applies).
	bool Partition(std::span<const uint32_t> code, uint64_t base, const CompileOptions& options,
	               const Decoder::ShaderCodeReader& read, CallPartitionRun& run);
	// The keys a partition's report set holds (target low word | 1): their targets become known.
	void ReportPartitionKeys(uint64_t program, std::span<const uint32_t> keys);
	// KYTY_CALL_PARTITION_MEMORY: loads the remembered targets from `path` (once) and appends new
	// reports to it. An empty path or a disabled switch does nothing.
	void SetPartitionMemory(const std::string& path);

	uint32_t rescan_interval = 64;
	// KYTY_CALL_LEARN (tests set these): the learned mode, the variant bound, and the table size
	// above which it applies (0: the bound Expand applies, MaxCallSiteTargets or the TOPK cap).
	bool     learn          = CallLearnEnabled();
	uint32_t learn_variants = CallLearnVariants();
	size_t   learn_over     = 0;
	// Once every call of a program reaches two targets, that expansion is kept for every table
	// (targets outside it take the skip path): changing tables compile no new pipelines.
	bool sticky = StickyExpansionsDefault();

private:
	struct TableState {
		uint64_t              digest     = 0;
		uint32_t              scanned_at = 0;
		std::vector<uint64_t> targets; // ascending
	};
	struct Call {
		ShaderCall trace;
		std::map<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>, TableState> tables; // by view
	};
	struct Cached {
		Expanded                     expanded;
		Decoder::ShaderCallExpansion expansion; // its reads (the callee code)
		uint32_t                     checked_at = 0;
		uint32_t                     used_at    = 0;
	};
	struct Program {
		bool                       usable = true;
		std::vector<Call>          calls;
		uint32_t                   dispatches = 0;
		Expanded                   current;
		std::map<uint64_t, Cached> expansions; // by target set
		uint64_t                   pinned = 0; // KYTY_SHADER_CALLS_STICKY: the expansion kept for every table
		std::string                last_failure;
		// KYTY_CALL_PARTITIONS: 0 not decided, 1 partitioned, -1 not.
		int                                    partition_state = 0;
		std::vector<uint64_t>                  partition_targets; // the table's, ascending
		std::vector<uint64_t>                  partition_known;   // in the order they were reported
		std::map<uint64_t, CallPartitionPart>  partition_parts;   // by target (0: the probe)
		uint64_t                               shader_hash     = 0;
		bool                                   memory_seeded   = false; // KYTY_CALL_PARTITION_MEMORY
		// KYTY_CALL_LEARN: the expansion of the learned targets and the set it holds (by key).
		CallPartitionPart                      learn_part;
		uint64_t                               learn_key = 0;
	};
	// KYTY_CALL_LEARN: the run of a program in the learned mode (Partition takes it).
	bool Learn(Program& program, uint64_t key, std::span<const uint32_t> code, uint64_t base,
	           const CompileOptions& options, const Decoder::ShaderCodeReader& read, CallPartitionRun& run);

	bool Refresh(Program& program, std::span<const uint32_t> code, uint64_t base,
	             const CompileOptions& options, const Decoder::ShaderCodeReader& read);

	std::mutex                            m_mutex;
	std::unordered_map<uint64_t, Program> m_programs; // by code address ^ hash
	std::vector<uint32_t>                 m_table_words;
	std::set<uint64_t>                    m_dropped; // guest addresses (48 bits)
	// KYTY_CALL_PARTITION_MEMORY: remembered (shader hash, target) pairs and the file they go to.
	std::set<std::pair<uint64_t, uint64_t>> m_memory;
	std::string                             m_memory_path;
	// Every expansion handed out stays alive: a compile may still read one that was replaced.
	std::vector<std::shared_ptr<const std::vector<uint32_t>>> m_retained;
};

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERCALLS_H_ */
