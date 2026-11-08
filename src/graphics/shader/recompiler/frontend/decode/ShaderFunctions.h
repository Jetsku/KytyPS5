#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_SHADERFUNCTIONS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_SHADERFUNCTIONS_H_

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// S_SWAPPC_B64 calls expanded into the calling program (after Senaxx's ShaderFunctions expander,
// senaxx/wolverine c26ad60c8..f5cbe7866): the callees are read from guest memory and their code is
// placed inline, so the expanded words are an ordinary program for the decoder, CFG and IR.
namespace Libs::Graphics::ShaderRecompiler::Decoder {

// Reads guest words at a guest address; false when they are not readable.
using ShaderCodeReader = std::function<bool(uint64_t, std::span<uint32_t>)>;

// S_SWAPPC_B64 with a destination pair (a call); with the null destination it is a plain jump
// (decoded as S_SETPC_B64). A literal can look like one: callers that need certainty decode.
[[nodiscard]] constexpr bool IsShaderCallWord(uint32_t word) {
	return (word & 0xff80ff00u) == 0xbe802100u && ((word >> 16u) & 0x7fu) != 125u;
}
// Whether the code may call (any word looks like a call): the cheap test before an expansion.
[[nodiscard]] bool MayCallFunctions(std::span<const uint32_t> code);

// One call instruction of the program and the functions it can reach. `source_reg` is the target
// pair's first SGPR, `return_reg` the destination pair's (the callee returns through it with
// S_SETPC_B64). The expansion compares the target pair with every address at run time; a target
// outside `addresses` (or none at all) skips the call: the return pair is written and the program
// continues after it, as if the function returned at once.
struct ShaderCallSite {
	uint32_t              pc         = 0;
	uint32_t              source_reg = 0;
	uint32_t              return_reg = 0;
	std::vector<uint64_t> addresses; // ascending, distinct, nonzero
	// KYTY_CALL_PARTITIONS (ShaderCalls.h): one partition of the call. At most one address, whose
	// body runs when the target pair equals the target the dispatch passes in (PartitionCompare);
	// every other target is reported and its lanes dropped (PartitionDrop).
	bool partition = false;
	// Call-split callee programs (CallSplitPlan.cpp): one address, whose body runs when the target
	// pair equals the callee the split dispatch passes in (SplitTargetCompare); the return pair and
	// the program's own S_GETPC_B64 values are written relative to `shader_base` (ShaderAddress).
	bool     split_callee = false;
	uint64_t shader_base  = 0;
	// KYTY_CALL_LEARN (ShaderCalls.h): a target outside `addresses` is reported (MissReport) before
	// its call takes the skip path, so a later expansion can place it.
	bool report_misses = false;

	bool operator==(const ShaderCallSite&) const = default;
};

// Where a placed callee body is in the expanded program (byte offsets, as instruction pcs).
struct ShaderCallBody {
	uint32_t begin   = 0;
	uint32_t end     = 0;
	uint64_t address = 0;
};

struct ShaderCallExpansion {
	std::vector<ShaderCallBody> bodies;
	std::vector<uint32_t> code;
	// The callee code as it was read: (guest address, words). The expansion is valid while these
	// read the same.
	std::vector<std::pair<uint64_t, std::vector<uint32_t>>> reads;
	uint32_t              inlined = 0; // callee bodies placed
	// Why targets were left out (unreadable, not a placeable function): their calls are skipped.
	std::vector<std::string> dropped;
};

// Expands every call of `code` (at guest address `base`). `sites` must name each call instruction
// (by pc). Each callee runs from its address to its last S_SETPC_B64 through the return pair (at
// most MaxShaderFunctionWords), with no call, S_ENDPGM, S_SETPC_B64 through another pair, write
// of the return pair or branch outside itself; a target that is not such a function is left out
// (`dropped`) and its calls are skipped. A body is placed once per call site; the SCC
// value the call keeps (S_SWAPPC_B64 does not change it) is restored after the comparisons.
// S_GETPC_B64 (in the program and the callees) becomes moves of its guest value, so the expanded
// program does not depend on where its words are. False with `reason` when it cannot expand.
bool ExpandShaderCalls(std::span<const uint32_t> code, uint64_t base,
                       std::span<const ShaderCallSite> sites, const ShaderCodeReader& read,
                       uint32_t wave_size, ShaderCallExpansion& expansion, std::string& reason);

// The code of one function as the expansion reads it (from `address` to its last S_SETPC_B64
// through s[return_reg:+1]); false with `reason` when it is not a function the expansion places.
bool ReadShaderFunction(uint64_t address, uint32_t return_reg, const ShaderCodeReader& read,
                        uint32_t wave_size, std::vector<uint32_t>& words, std::string& reason);

// KYTY_CALL_LOOP_CAP (CodegenOptions::call_loop_cap): a placed body with a back edge and no barrier
// starts with this S_NOP, and this one stands before its last return; the translation budgets the
// loops whose headers lie between them. S_NOP immediates real code does not use (the wait count is
// the low bits). After usa-ratt/KytyPS5-GT7 08197df0b.
constexpr uint32_t ShaderCallLoopBodyBegin = 0xbf80c0deu;
constexpr uint32_t ShaderCallLoopBodyEnd   = 0xbf80c0dfu;

constexpr uint32_t MaxShaderFunctionWords = 16384;
constexpr uint32_t MaxExpandedShaderWords = 1u << 20u;

} // namespace Libs::Graphics::ShaderRecompiler::Decoder

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_FRONTEND_DECODE_SHADERFUNCTIONS_H_ */
