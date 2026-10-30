#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLIT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLIT_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Call splitting (KYTY_SHADER_CALLS_SPLIT): a compute program whose S_SWAPPC_B64 calls sit in
// waterfall loops (Insomniac's ray-tracing kernels: for each distinct record of the active lanes,
// call the record's hit function) runs as a chain of dispatches instead of one program with every
// callee inlined, which the driver cannot compile in reasonable time:
//
//   R_0    the program from its start. The first instruction of each call loop (site k, guest
//          pcs [a_k, b_k)) suspends the wave: its live registers go to the call-state buffer and
//          the wave's header word becomes k.
//   C_k,F  the call loop of site k alone with only callee F expanded (ExpandShaderCalls): it
//          loads the state at a_k, runs the loop (calls to other targets take the skip path) and
//          stores the VGPR lanes it changed. Lanes of other callees are unchanged, so the C passes
//          of a site touch disjoint lanes and need no ordering among themselves.
//   R_ak   the program entered at a_k: it loads the state, runs the call loop with every call
//          skipped (which recreates the loop's scalar state), and continues to the end or the
//          next site.
//
// Every program is the guest code with S_SWAPPC_B64 replaced by S_GETPC_B64 (the skip path: the
// return pair gets the return address) plus S_KYTY_CALL_STATE markers. Liveness comes from the
// IR: a translation that loads every register at a_k keeps only the loads its code reads (VGPR
// writes are exec selects, so a partial write keeps the old value live). Scalar values that are a
// function of the user data and memory alone (descriptors) are not saved: their IR at a_k (a probe
// translation of the whole program) is rebuilt in the loading program, so resource tracking sees
// them as in the original.
namespace Libs::Graphics::ShaderRecompiler::CallSplit {

constexpr uint32_t HeaderMagic = 0x3153434bu; // "KCS1"

// Register codes: scalar operand codes (s0-s105, VCC_LO 106, VCC_HI 107, M0 124, EXEC_LO 126,
// EXEC_HI 127, SCC 253) and VectorCode + n for v[n].
constexpr uint16_t CodeVccLo  = 106;
constexpr uint16_t CodeVccHi  = 107;
constexpr uint16_t CodeM0     = 124;
constexpr uint16_t CodeExecLo = 126;
constexpr uint16_t CodeExecHi = 127;
constexpr uint16_t CodeScc    = 253;
constexpr uint16_t VectorCode = 256;

// Push-data parameters (IR::BindingLayout::call_state_dword).
enum Param : uint32_t {
	ParamStateLo  = 0, // the wave blocks
	ParamStateHi  = 1,
	ParamListLo   = 2, // the wave list of this dispatch (programs entered at a site)
	ParamListHi   = 3,
	ParamHeaderLo = 4, // one word per wave: the site it suspended at
	ParamHeaderHi = 5,
	ParamStride   = 6, // dwords per wave block
	ParamTargetLo = 7, // callee programs: the guest address of the callee this dispatch runs
	ParamTargetHi = 8,
};

// The registers kept for a site: uniform words (one per wave) then 64 lanes per vector register.
struct SiteLayout {
	std::vector<uint16_t> uniform;
	std::vector<uint16_t> vector;

	[[nodiscard]] uint32_t VectorBase() const {
		return (static_cast<uint32_t>(uniform.size()) + 63u) & ~63u;
	}
	[[nodiscard]] uint32_t WaveDwords() const {
		return VectorBase() + static_cast<uint32_t>(vector.size()) * 64u;
	}
	bool operator==(const SiteLayout&) const = default;
};

enum HeaderFlag : uint32_t {
	FlagLiveness = 1u << 8u, // a planning translation: entry loads are markers, saves store nothing
	FlagProbe    = 1u << 9u, // a planning translation of the whole program with probes at each site
};

struct Header {
	uint32_t                entry_site = 0; // 0: from the program start
	uint32_t                flags      = 0;
	std::array<uint64_t, 2> digest {};      // of the canonical program (recipe key)
	uint32_t                vgprs      = 0; // registers a liveness translation loads
	std::vector<SiteLayout> sites;

	[[nodiscard]] uint32_t WaveDwords() const;
};

// The marker word followed by the encoded header (the first instruction of every split program).
void EncodeHeader(const Header& header, std::vector<uint32_t>& out);
bool DecodeHeader(std::span<const uint32_t> payload, Header& header);

// A scalar value rebuilt in the loading program: nodes in dependency order. An argument is a
// node index (>= 0) or the immediate imms[-1 - arg].
struct RecipeNode {
	IR::ValueOpcode       opcode = IR::ValueOpcode::Void;
	uint64_t              flags  = 0;
	std::vector<int32_t>  args;
	bool                  has_memory = false;
	uint32_t              memory_pc  = 0;
	IR::MemoryInfo        memory {};
};
struct SiteRecipe {
	std::vector<RecipeNode>     nodes;
	std::vector<IR::Value>      imms;
	std::map<uint16_t, int32_t> roots; // scalar code -> argument (node or immediate)
};
struct Recipe {
	std::vector<SiteRecipe> sites; // index k - 1
};

void                          RegisterRecipe(const std::array<uint64_t, 2>& digest,
                                             std::shared_ptr<const Recipe> recipe);
std::shared_ptr<const Recipe> FindRecipe(const std::array<uint64_t, 2>& digest);

// What a planning translation found (TranslateResult::call_split).
struct Report {
	std::vector<uint16_t> live; // liveness: the codes the program reads at its entry
	Recipe                recipe;
};

// Translation-time state (the translator thread of one program).
struct Translation {
	Header                        header;
	std::shared_ptr<const Recipe> recipe;
	IR::Value                     state_base;  // U64
	IR::Value                     header_base; // U64
	IR::Value                     wave;        // U32
	IR::Value                     wave_dword;  // U32
	std::vector<IR::Value>        loaded;      // per vector slot of the entry site
	std::vector<uint32_t>         marker_codes; // liveness: CallStateParam(MarkerBase + i) -> code
	std::map<uint32_t, uint32_t>  site_at_pc;   // Save / Probe markers: guest pc -> site
	// Bytes in front of the guest code in this program (the header and, for a resume program, its
	// branch to a_k): an S_GETPC_B64 of the canonical code returns its guest address, so PC-relative
	// data after the guest code (Miles' 0x6d9e reads its pixel jitter table this way) is found.
	uint32_t                      guest_pc_offset = 0;
	bool                          prepared = false;
};
constexpr uint32_t MarkerBase = 0x1000u;
[[nodiscard]] Translation* Current();
void                       SetCurrent(Translation* translation);

// Collects the planning report of a translated program (after the early IR passes).
void CollectReport(const Translation& translation, const IR::Program& program, Report& report);

} // namespace Libs::Graphics::ShaderRecompiler::CallSplit

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_CALLSPLIT_H_ */
