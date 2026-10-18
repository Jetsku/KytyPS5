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

	uint32_t rescan_interval = 64;
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
	};

	bool Refresh(Program& program, std::span<const uint32_t> code, uint64_t base,
	             const CompileOptions& options, const Decoder::ShaderCodeReader& read);

	std::mutex                            m_mutex;
	std::unordered_map<uint64_t, Program> m_programs; // by code address ^ hash
	std::vector<uint32_t>                 m_table_words;
	std::set<uint64_t>                    m_dropped; // guest addresses (48 bits)
	// Every expansion handed out stays alive: a compile may still read one that was replaced.
	std::vector<std::shared_ptr<const std::vector<uint32_t>>> m_retained;
};

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERCALLS_H_ */
