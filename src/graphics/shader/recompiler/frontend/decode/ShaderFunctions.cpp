#include "graphics/shader/recompiler/frontend/decode/ShaderFunctions.h"

#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"

#include <algorithm>
#include <array>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <map>
#include <set>

namespace Libs::Graphics::ShaderRecompiler::Decoder {
namespace {

constexpr uint32_t SBranch      = 0xbf820000u; // s_branch (simm16 patched)
constexpr uint32_t SCbranchScc0 = 0xbf840000u;
constexpr uint32_t SCbranchScc1 = 0xbf850000u;
constexpr uint32_t SEndpgm      = 0xbf810000u;
constexpr uint32_t SNop         = 0xbf800000u;
constexpr uint32_t SetScc1      = 0xbf068080u; // s_cmp_eq_u32 0, 0
constexpr uint32_t SetScc0      = 0xbf068180u; // s_cmp_eq_u32 0, 1

bool IsCall(const Instruction& inst) {
	return inst.family == Family::SOP1 && IsShaderCallWord(inst.raw[0]);
}

// The SGPRs an instruction's destinations cover (VOPC and other lane masks: the wave's mask width).
uint32_t DestinationWords(const Instruction& inst, uint32_t wave_size) {
	const auto name = magic_enum::enum_name(inst.opcode);
	if (name.starts_with("V_CMP")) return wave_size / 32u;
	// A 32-bit result of a 64-bit source (S_FF1_I32_B64, S_BCNT1_I32_B64): one SGPR.
	const bool pair = name.find("64") != std::string_view::npos && !name.ends_with("32_B64");
	return std::max(inst.data_dwords, pair ? 2u : 1u);
}

bool Writes(const Instruction& inst, uint32_t reg, uint32_t wave_size) {
	if (IsCall(inst)) {
		const auto dst = (inst.raw[0] >> 16u) & 0x7fu;
		return reg == dst || reg == dst + 1u;
	}
	return (inst.dst.kind == OperandKind::Sgpr && reg >= inst.dst.reg &&
	        reg < inst.dst.reg + DestinationWords(inst, wave_size)) ||
	       (inst.dst2.kind == OperandKind::Sgpr && reg >= inst.dst2.reg &&
	        reg < inst.dst2.reg + wave_size / 32u);
}

void EmitMove(std::vector<uint32_t>& code, uint32_t reg, uint32_t value) {
	code.push_back(0xbe8003ffu | (reg << 16u)); // s_mov_b32 s[reg], literal
	code.push_back(value);
}

// A scalar pair set to `value`: its moves, or with a shader base (call-split callee programs) the
// ShaderAddress marker of the offset from it, so the words do not hold the guest address.
void EmitPair(std::vector<uint32_t>& code, uint32_t reg, uint64_t value, uint64_t shader_base) {
	if (shader_base != 0 && value >= shader_base && value - shader_base <= 0xffffffffull) {
		code.push_back(EncodeCallStateWord(CallStateOp::ShaderAddress, 2u));
		code.push_back(reg);
		code.push_back(static_cast<uint32_t>(value - shader_base));
		return;
	}
	EmitMove(code, reg, static_cast<uint32_t>(value));
	EmitMove(code, reg + 1u, static_cast<uint32_t>(value >> 32u));
}

// S_GETPC_B64 at guest address `address`: moves of its value (S_MOV_B32 keeps SCC, as GETPC).
bool EmitGetPc(std::vector<uint32_t>& code, const Instruction& inst, uint64_t address,
               std::string& reason, uint64_t shader_base = 0) {
	if (inst.dst.kind == OperandKind::Null) {
		code.push_back(SNop);
		return true;
	}
	const auto reg = (inst.raw[0] >> 16u) & 0x7fu;
	if (!((inst.dst.kind == OperandKind::Sgpr && inst.dst.reg <= 104u) ||
	      inst.dst.kind == OperandKind::VccLo)) {
		reason = "S_GETPC_B64 destination is not a scalar pair";
		return false;
	}
	const auto pc = address + inst.word_count * sizeof(uint32_t);
	EmitPair(code, reg, pc, shader_base);
	return true;
}

struct Function {
	uint64_t                 address = 0;
	std::vector<uint32_t>    code;
	std::vector<Instruction> instructions;
};

// The callee's code from `address` up to its last return through `return_reg`.
bool ReadFunction(uint64_t address, uint32_t return_reg, const ShaderCodeReader& read,
                  uint32_t wave_size, Function& function, std::string& reason) {
	function.address = address;
	const auto guest = address & 0x0000ffffffffffffull;
	if ((guest & 3u) != 0 || guest == 0) {
		reason = fmt::format("callee 0x{:x} is not a dword-aligned address", address);
		return false;
	}
	std::vector<uint32_t> words;
	bool                  at_end = false;
	// Makes `count` words readable when guest memory has them; a chunk that fails is retried
	// word by word, so a callee that ends near the end of its mapping still reads.
	const auto ensure = [&](size_t count) {
		while (!at_end && words.size() < count && words.size() < MaxShaderFunctionWords) {
			std::array<uint32_t, 64> chunk {};
			const auto           at = guest + words.size() * sizeof(uint32_t);
			if (read(at, chunk)) {
				words.insert(words.end(), chunk.begin(), chunk.end());
				continue;
			}
			std::array<uint32_t, 1> one {};
			if (!read(at, one)) {
				at_end = true;
				break;
			}
			words.push_back(one[0]);
		}
		return words.size() >= count;
	};
	uint32_t next            = 0;
	uint32_t furthest_target = 0;
	bool     returned        = false;
	for (;;) {
		if (next >= MaxShaderFunctionWords || !ensure(next + 1u)) {
			reason = next >= MaxShaderFunctionWords
			             ? fmt::format("callee 0x{:x} is longer than {} words", address,
			                           MaxShaderFunctionWords)
			             : fmt::format("callee 0x{:x} code is not readable at +0x{:x}", address,
			                           next * 4u);
			return false;
		}
		(void)ensure(next + MaxInstructionRawWords);
		if (GetInstructionFamily(words[next]) == Family::Unknown) {
			reason = fmt::format("callee 0x{:x} has an unknown encoding at +0x{:x}", address, next * 4u);
			return false;
		}
		std::vector<uint32_t> window(words.begin() + next,
		                             words.begin() + std::min<size_t>(words.size(),
		                                                              next + MaxInstructionRawWords));
		window.resize(MaxInstructionRawWords, 0u);
		Instruction inst;
		DecodeInstruction(window, 0, inst);
		if (next + inst.word_count > words.size()) {
			reason = fmt::format("callee 0x{:x} code is not readable at +0x{:x}", address, next * 4u);
			return false;
		}
		// Decoded at pc 0: move it and its branch target to its place in the callee.
		inst.pc = next * sizeof(uint32_t);
		if (IsDirectBranch(inst.opcode)) {
			inst.branch_target += inst.pc;
		}
		if (IsCall(inst)) {
			reason = fmt::format("callee 0x{:x} calls another function (+0x{:x})", address, inst.pc);
			return false;
		}
		if (inst.opcode == Opcode::S_ENDPGM || inst.opcode == Opcode::UNSUPPORTED ||
		    inst.opcode == Opcode::UNKNOWN) {
			reason = fmt::format("callee 0x{:x} has {} at +0x{:x}", address,
			                     magic_enum::enum_name(inst.opcode), inst.pc);
			return false;
		}
		if (Writes(inst, return_reg, wave_size) || Writes(inst, return_reg + 1u, wave_size)) {
			reason = fmt::format("callee 0x{:x} writes its return pair s{} (+0x{:x})", address,
			                     return_reg, inst.pc);
			return false;
		}
		function.instructions.push_back(inst);
		next += inst.word_count;
		if (IsDirectBranch(inst.opcode)) {
			furthest_target = std::max(furthest_target, inst.branch_target);
		}
		if (inst.opcode == Opcode::S_SETPC_B64) {
			if (inst.src0.kind != OperandKind::Sgpr || inst.src0.reg != return_reg) {
				reason = fmt::format("callee 0x{:x} jumps through another pair (+0x{:x})", address,
				                     inst.pc);
				return false;
			}
			returned = true;
			if (next * sizeof(uint32_t) > furthest_target) break;
		}
	}
	if (!returned) {
		reason = fmt::format("callee 0x{:x} does not return", address);
		return false;
	}
	words.resize(next);
	function.code = std::move(words);
	std::set<uint32_t> starts;
	for (const auto& inst: function.instructions) {
		starts.insert(inst.pc);
	}
	for (const auto& inst: function.instructions) {
		if (IsDirectBranch(inst.opcode) && !starts.contains(inst.branch_target)) {
			reason = fmt::format("callee 0x{:x} branches outside itself (+0x{:x})", address, inst.pc);
			return false;
		}
	}
	return true;
}

bool ReadsScc(const Instruction& inst) {
	const auto name = magic_enum::enum_name(inst.opcode);
	return name.starts_with("S_CSELECT") || name.starts_with("S_CMOV") ||
	       name.starts_with("S_CBRANCH_SCC") || name == "S_ADDC_U32" || name == "S_SUBB_U32";
}

bool WritesScc(const Instruction& inst) {
	if (inst.family == Family::SOPC) return true;
	if (inst.family != Family::SOP2 && inst.family != Family::SOP1) return false;
	const auto name = magic_enum::enum_name(inst.opcode);
	for (const std::string_view prefix: {"S_ADD_", "S_SUB_", "S_AND_", "S_OR_", "S_XOR_", "S_NAND_",
	                                     "S_NOR_", "S_XNOR_", "S_ANDN2_", "S_ORN2_", "S_LSHL_",
	                                     "S_LSHR_", "S_ASHR_", "S_BFE_", "S_MIN_", "S_MAX_",
	                                     "S_ABSDIFF_", "S_NOT_", "S_ABS_", "S_BCNT"}) {
		if (name.starts_with(prefix)) return true;
	}
	return false;
}

// Whether the callee can observe the SCC its call keeps: it reads SCC, or reaches control flow or
// its return, before anything writes it (then the caller sees the SCC of the call after it).
bool NeedsCallScc(const Function& function) {
	for (const auto& inst: function.instructions) {
		if (ReadsScc(inst)) return true;
		if (WritesScc(inst)) return false;
		if (IsDirectBranch(inst.opcode) || inst.opcode == Opcode::S_SETPC_B64) return true;
	}
	return true;
}

bool HasBackEdge(const Function& function) {
	return std::ranges::any_of(function.instructions, [](const Instruction& inst) {
		return IsDirectBranch(inst.opcode) && inst.branch_target <= inst.pc;
	});
}

bool HasBarrier(const Function& function) {
	return std::ranges::any_of(function.instructions,
	                           [](const Instruction& inst) { return inst.opcode == Opcode::S_BARRIER; });
}

// KYTY_CALL_WATERFALL_CLEAR (after usa-ratt/KytyPS5-GT7 655faf07e): the VGPR a waterfall around the
// call at instructions[call] keeps its "lanes left" flag in, or UINT32_MAX. GT7's material passes
// call each material's function for its lanes in turn:
//   loop: V_READFIRSTLANE_B32 s, vKey; S_MOV_B64 sE, exec; V_CMPX_EQ_U32 s, vKey
//         V_MOV_B32 vF, 0            (these lanes are done)
//         ... scalar setup ... S_SWAPPC_B64
//         S_MOV_B64 exec, sE; V_CMPX_NE_U32 0, vF; S_CBRANCH_EXECNZ loop
// A callee that left vF set would have its lanes called again, forever. Only this shape counts: the
// exec restore, the test and the branch back to (or before) the clear right after the call, and
// nothing but scalar instructions, none of them a branch target, between the clear and the call.
uint32_t WaterfallFlag(const std::vector<Instruction>& instructions, size_t call) {
	const auto zero = [](const Operand& operand) {
		return operand.kind == OperandKind::IntegerInlineConstant && operand.value == 0;
	};
	const auto next = [&](size_t k) {
		while (k < instructions.size() &&
		       (instructions[k].opcode == Opcode::S_WAITCNT || instructions[k].opcode == Opcode::S_NOP)) {
			k++;
		}
		return k;
	};
	const auto restore = next(call + 1u);
	if (restore >= instructions.size() || instructions[restore].opcode != Opcode::S_MOV_B64 ||
	    instructions[restore].dst.kind != OperandKind::ExecLo) {
		return UINT32_MAX;
	}
	const auto test = next(restore + 1u);
	if (test >= instructions.size() || instructions[test].opcode != Opcode::V_CMPX_NE_U32 ||
	    !zero(instructions[test].src0) || instructions[test].src1.kind != OperandKind::Vgpr) {
		return UINT32_MAX;
	}
	const auto flag   = instructions[test].src1.reg;
	const auto branch = next(test + 1u);
	if (branch >= instructions.size() || instructions[branch].opcode != Opcode::S_CBRANCH_EXECNZ) {
		return UINT32_MAX;
	}
	for (size_t k = call; k-- > 0 && k + 24u > call;) {
		const auto& inst = instructions[k];
		if (inst.opcode == Opcode::V_MOV_B32 && inst.dst.kind == OperandKind::Vgpr && inst.dst.reg == flag &&
		    zero(inst.src0)) {
			const auto clear   = inst.pc;
			const auto end     = instructions[call].pc;
			const bool entered = std::ranges::any_of(instructions, [&](const Instruction& other) {
				return IsDirectBranch(other.opcode) && other.branch_target > clear && other.branch_target <= end;
			});
			return instructions[branch].branch_target <= clear && !entered ? flag : UINT32_MAX;
		}
		if (!magic_enum::enum_name(inst.opcode).starts_with("S_") || IsDirectBranch(inst.opcode) || IsCall(inst)) {
			break;
		}
	}
	return UINT32_MAX;
}

using Branches = std::vector<std::pair<uint32_t, uint32_t>>; // (branch word, target word)

// Patches the SOPP branches. An edge beyond simm16 goes through an island of two S_BRANCH placed
// at an instruction boundary between its ends: the first skips the island for code that falls
// into it, the second continues the long edge. No guest SGPR or SCC is touched.
bool RelocateBranches(std::vector<uint32_t>& code, Branches& branches, std::vector<ShaderCallBody>& bodies,
                      std::string& reason) {
	for (size_t index = 0; index < branches.size();) {
		const auto [at, target] = branches[index];
		const auto delta        = int64_t(target) - int64_t(at) - 1;
		if (delta >= std::numeric_limits<int16_t>::min() &&
		    delta <= std::numeric_limits<int16_t>::max()) {
			++index;
			continue;
		}
		if (code.size() + 2 > MaxExpandedShaderWords) {
			reason = "the expanded program's branch islands exceed the code bound";
			return false;
		}
		const auto desired   = delta > 0 ? at + 16384u : at - 16384u;
		uint32_t   insertion = 0;
		for (uint32_t word = 0; word <= desired && word < code.size();) {
			insertion = word;
			Instruction inst;
			DecodeInstruction(code, word, inst);
			if (inst.word_count == 0 || inst.word_count > code.size() - word) {
				reason = "the expanded program has an invalid instruction boundary";
				return false;
			}
			word += inst.word_count;
		}
		if (insertion <= std::min(at, target) || insertion >= std::max(at, target)) {
			reason = "no instruction boundary for a branch island";
			return false;
		}
		code.insert(code.begin() + insertion, {SBranch, SBranch});
		for (auto& [source, destination]: branches) {
			if (source >= insertion) source += 2;
			if (destination >= insertion) destination += 2;
		}
		for (auto& body: bodies) {
			if (body.begin >= insertion * 4u) body.begin += 8u;
			if (body.end > insertion * 4u) body.end += 8u;
		}
		const auto onward      = branches[index].second;
		branches[index].second = insertion + 1u;
		branches.emplace_back(insertion, insertion + 2u); // falls past the island
		branches.emplace_back(insertion + 1u, onward);    // the long edge continues
		index = 0; // an insertion can lengthen an earlier edge
	}
	for (const auto [at, target]: branches) {
		const auto delta = int64_t(target) - int64_t(at) - 1;
		code[at]         = (code[at] & 0xffff0000u) | (static_cast<uint32_t>(delta) & 0xffffu);
	}
	return true;
}

} // namespace

bool ReadShaderFunction(uint64_t address, uint32_t return_reg, const ShaderCodeReader& read,
                        uint32_t wave_size, std::vector<uint32_t>& words, std::string& reason) {
	Function function;
	if (!ReadFunction(address, return_reg, read, wave_size, function, reason)) return false;
	words = std::move(function.code);
	return true;
}
bool MayCallFunctions(std::span<const uint32_t> code) {
	return std::ranges::any_of(code, [](uint32_t word) { return IsShaderCallWord(word); });
}

bool ExpandShaderCalls(std::span<const uint32_t> code, uint64_t base,
                       std::span<const ShaderCallSite> sites, const ShaderCodeReader& read,
                       uint32_t wave_size, ShaderCallExpansion& expansion, std::string& reason) {
	expansion = {};
	reason.clear();
	if (wave_size != 32u && wave_size != 64u) {
		reason = "unknown wave size";
		return false;
	}
	Program program;
	DecodeProgram(code, program, true);
	std::map<uint32_t, const ShaderCallSite*> calls;
	for (const auto& site: sites) {
		calls[site.pc] = &site;
	}
	size_t call_count = 0;
	for (const auto& inst: program.instructions) {
		if (inst.opcode == Opcode::S_SETPC_B64) {
			reason = fmt::format("the program jumps through S_SETPC_B64 (pc 0x{:x})", inst.pc);
			return false;
		}
		if (!IsCall(inst)) continue;
		call_count++;
		const auto found = calls.find(inst.pc);
		if (found == calls.end()) {
			reason = fmt::format("no targets for the call at pc 0x{:x}", inst.pc);
			return false;
		}
		const auto source = inst.raw[0] & 0xffu;
		const auto dest   = (inst.raw[0] >> 16u) & 0x7fu;
		if (source >= 104u || (source & 1u) != 0 || dest >= 104u || (dest & 1u) != 0 ||
		    found->second->source_reg != source || found->second->return_reg != dest) {
			reason = fmt::format("the call at pc 0x{:x} uses a special or odd scalar pair", inst.pc);
			return false;
		}
	}
	if (call_count == 0) {
		reason = "the program has no call";
		return false;
	}
	// Each callee is read once, whatever number of sites reach it. A target that is not a function
	// this can place (a table entry that is not code, or a callee with a nested call) is left out:
	// its calls take the skip path.
	std::map<std::pair<uint64_t, uint32_t>, Function> functions; // (address, return pair)
	std::set<std::pair<uint64_t, uint32_t>>           unusable;
	std::map<uint32_t, std::vector<uint64_t>>         reachable; // per call pc
	for (const auto& site: sites) {
		auto& addresses = reachable[site.pc];
		for (const auto address: site.addresses) {
			const auto key = std::pair {address, site.return_reg};
			if (!functions.contains(key) && !unusable.contains(key)) {
				Function    function;
				std::string why;
				const bool readable = ReadFunction(address, site.return_reg, read, wave_size, function, why);
				if (readable && !GetCodegenOptions().call_loops && HasBackEdge(function)) {
					// KYTY_CALL_LOOPS=0: looping callees take the skip path.
					unusable.insert(key);
					expansion.dropped.push_back(
					    fmt::format("callee 0x{:x} has a loop (KYTY_CALL_LOOPS=0)", address));
				} else if (readable) {
					functions.emplace(key, std::move(function));
				} else {
					unusable.insert(key);
					expansion.dropped.push_back(std::move(why));
				}
			}
			if (functions.contains(key)) addresses.push_back(address);
		}
	}
	for (const auto& [key, function]: functions) {
		expansion.reads.emplace_back(function.address & 0x0000ffffffffffffull, function.code);
	}

	// Call-split callee programs: addresses in the program's own code relative to its shader base.
	uint64_t relative_base = 0;
	for (const auto& site: sites) {
		if (site.split_callee) relative_base = site.shader_base;
	}
	std::vector<uint32_t>        result;
	std::map<uint32_t, uint32_t> main_pc; // guest pc -> word in `result`
	Branches                     main_branches;
	Branches                     relocations;
	result.reserve(code.size() * 2);
	for (const auto& inst: program.instructions) {
		main_pc[inst.pc] = static_cast<uint32_t>(result.size());
		const auto found = IsCall(inst) ? calls.find(inst.pc) : calls.end();
		if (found == calls.end()) {
			if (inst.opcode == Opcode::S_ENDPGM) {
				result.push_back(SEndpgm);
			} else if (inst.opcode == Opcode::S_GETPC_B64) {
				if (!EmitGetPc(result, inst, base + inst.pc, reason, relative_base)) return false;
			} else {
				if (IsDirectBranch(inst.opcode)) {
					main_branches.emplace_back(static_cast<uint32_t>(result.size()), inst.branch_target);
				}
				result.insert(result.end(), inst.raw, inst.raw + inst.word_count);
			}
		} else {
			auto site      = *found->second;
			site.addresses = reachable.at(site.pc);
			const uint64_t return_pc = base + inst.pc + inst.word_count * sizeof(uint32_t);
			const auto     set_return = [&] { EmitPair(result, site.return_reg, return_pc, relative_base); };
			// Selection, shaped for the structurizer: every selection of it has its own merge block (an
			// S_BRANCH to the next instruction) and no block is entered from two places, so the CFG
			// needs no merge splitting. The targets of a high address word are found by a balanced
			// search over their low words (s_cmp_lt_u32), each leaf comparing for equality; a match
			// runs its own copy of the body, a miss its own skip path (the call returns at once), and
			// every path leaves through the merges to the continuation. The comparisons change SCC and
			// S_SWAPPC_B64 does not: when a callee may see the SCC of the call (NeedsCallScc), there is
			// one selection per incoming SCC value, each restoring it.
			std::map<uint32_t, std::vector<size_t>> groups; // high word -> targets, by low word
			for (size_t t = 0; t < site.addresses.size(); t++) {
				groups[static_cast<uint32_t>(site.addresses[t] >> 32u)].push_back(t);
			}
			for (auto& [high, targets]: groups) {
				std::ranges::sort(targets, {}, [&](size_t t) { return static_cast<uint32_t>(site.addresses[t]); });
			}
			const bool dual = std::ranges::any_of(site.addresses, [&](uint64_t address) {
				return NeedsCallScc(functions.at({address, site.return_reg}));
			});
			const auto branch = [&] {
				const auto at = static_cast<uint32_t>(result.size());
				result.push_back(SBranch);
				return at;
			};
			// The merge block of a selection: an S_BRANCH to the instruction after it.
			const auto merge = [&](std::vector<uint32_t>& exits) {
				for (const auto at: exits) {
					relocations.emplace_back(at, static_cast<uint32_t>(result.size()));
				}
				const auto at = branch();
				relocations.emplace_back(at, at + 1u);
			};
			const auto emit_body = [&](const Function& function, std::vector<uint32_t>& exits) {
				const auto body_begin = static_cast<uint32_t>(result.size());
				// KYTY_CALL_LOOP_CAP: a looping body without a barrier is bracketed for the loop budget
				// (the end marker before its last return, so that it is reached).
				const bool bounded = GetCodegenOptions().call_loop_cap != 0u && HasBackEdge(function) &&
				                     !HasBarrier(function);
				if (bounded) result.push_back(ShaderCallLoopBodyBegin);
				set_return();
				std::map<uint32_t, uint32_t> local_pc;
				Branches                     local_branches;
				for (const auto& body: function.instructions) {
					const auto at     = static_cast<uint32_t>(result.size());
					local_pc[body.pc] = at;
					if (bounded && &body == &function.instructions.back()) {
						result.push_back(ShaderCallLoopBodyEnd);
					}
					if (body.opcode == Opcode::S_SETPC_B64) {
						exits.push_back(branch());
					} else if (body.opcode == Opcode::S_GETPC_B64) {
						if (!EmitGetPc(result, body, (function.address & 0x0000ffffffffffffull) + body.pc,
						               reason)) {
							return false;
						}
					} else {
						if (IsDirectBranch(body.opcode)) local_branches.emplace_back(at, body.branch_target);
						result.insert(result.end(), body.raw, body.raw + body.word_count);
					}
				}
				for (const auto& [at, target]: local_branches) {
					relocations.emplace_back(at, local_pc.at(target));
				}
				expansion.inlined++;
				expansion.bodies.push_back({.begin   = body_begin * 4u,
				                            .end     = static_cast<uint32_t>(result.size()) * 4u,
				                            .address = function.address});
				return result.size() <= MaxExpandedShaderWords;
			};
			// Code that enters with the target pair in s[source_reg:+1] and leaves by falling through.
			// KYTY_CALL_LEARN: a miss reports the target first (the marker keeps SCC).
			const auto skip = [&](uint32_t restore) {
				if (site.report_misses && !site.partition) {
					result.push_back(EncodeCallStateWord(CallStateOp::MissReport, 1u));
					result.push_back(site.source_reg);
				}
				if (restore != 0) result.push_back(restore);
				set_return();
			};
			std::function<bool(const std::vector<size_t>&, size_t, size_t, uint32_t)> emit_range;
			emit_range = [&](const std::vector<size_t>& targets, size_t first, size_t last,
			                 uint32_t restore) -> bool {
				std::vector<uint32_t> exits;
				if (last - first == 1u) {
					const auto t = targets[first];
					result.push_back(0xbf06ff00u | site.source_reg); // s_cmp_eq_u32 lo, lit
					result.push_back(static_cast<uint32_t>(site.addresses[t]));
					const auto miss = static_cast<uint32_t>(result.size());
					result.push_back(SCbranchScc0);
					if (restore != 0) result.push_back(restore);
					if (!emit_body(functions.at({site.addresses[t], site.return_reg}), exits)) return false;
					relocations.emplace_back(miss, static_cast<uint32_t>(result.size()));
					skip(restore);
				} else {
					const auto middle = first + (last - first) / 2u;
					result.push_back(0xbf0aff00u | site.source_reg); // s_cmp_lt_u32 lo, lit
					result.push_back(static_cast<uint32_t>(site.addresses[targets[middle]]));
					const auto lower = static_cast<uint32_t>(result.size());
					result.push_back(SCbranchScc1);
					if (!emit_range(targets, middle, last, restore)) return false;
					exits.push_back(branch());
					relocations.emplace_back(lower, static_cast<uint32_t>(result.size()));
					if (!emit_range(targets, first, middle, restore)) return false;
				}
				merge(exits);
				return true;
			};
			const auto emit_groups = [&](uint32_t restore) -> bool {
				std::vector<std::vector<uint32_t>> exits;
				for (const auto& [high, targets]: groups) {
					result.push_back(0xbf06ff00u | (site.source_reg + 1u)); // s_cmp_eq_u32 hi, lit
					result.push_back(high);
					const auto other = static_cast<uint32_t>(result.size());
					result.push_back(SCbranchScc0);
					if (!emit_range(targets, 0, targets.size(), restore)) return false;
					exits.push_back({branch()});
					relocations.emplace_back(other, static_cast<uint32_t>(result.size()));
				}
				skip(restore);
				for (auto it = exits.rbegin(); it != exits.rend(); ++it) {
					merge(*it);
				}
				return true;
			};
			// A partition (ShaderCallSite::partition): the target the dispatch passes in runs its body,
			// every other one is reported and its lanes leave the wave; then the skip path.
			const auto marker = [&](CallStateOp op) {
				result.push_back(EncodeCallStateWord(op, 1u));
				result.push_back(site.source_reg);
			};
			const auto emit_partition = [&](uint32_t restore) -> bool {
				if (site.addresses.empty()) {
					marker(CallStateOp::PartitionDrop);
					skip(0);
					return true;
				}
				std::vector<uint32_t> exits;
				marker(CallStateOp::PartitionCompare);
				const auto miss = static_cast<uint32_t>(result.size());
				result.push_back(SCbranchScc0);
				if (restore != 0) result.push_back(restore);
				if (!emit_body(functions.at({site.addresses.front(), site.return_reg}), exits)) return false;
				relocations.emplace_back(miss, static_cast<uint32_t>(result.size()));
				marker(CallStateOp::PartitionDrop);
				skip(restore);
				merge(exits);
				return true;
			};
			// A call-split callee program: the callee the dispatch runs (bin pass) against the target
			// pair, compared at run time instead of with its guest address.
			const auto emit_split = [&](uint32_t restore) -> bool {
				if (site.addresses.empty()) {
					skip(0);
					return true;
				}
				std::vector<uint32_t> exits;
				marker(CallStateOp::SplitTargetCompare);
				const auto miss = static_cast<uint32_t>(result.size());
				result.push_back(SCbranchScc0);
				if (restore != 0) result.push_back(restore);
				if (!emit_body(functions.at({site.addresses.front(), site.return_reg}), exits)) return false;
				relocations.emplace_back(miss, static_cast<uint32_t>(result.size()));
				skip(restore);
				merge(exits);
				return true;
			};
			bool emitted = false;
			if (site.split_callee && (site.partition || site.addresses.size() > 1u)) {
				reason = "a call-split callee with more than one target";
				return false;
			}
			if (site.split_callee && !dual) {
				emitted = emit_split(0);
			} else if (site.split_callee) {
				const auto select = static_cast<uint32_t>(result.size());
				result.push_back(SCbranchScc1);
				std::vector<uint32_t> exits;
				emitted = emit_split(SetScc0);
				exits.push_back(branch());
				relocations.emplace_back(select, static_cast<uint32_t>(result.size()));
				emitted = emitted && emit_split(SetScc1);
				merge(exits);
			} else if (site.partition && site.addresses.size() > 1u) {
				reason = "a partition with more than one target";
				return false;
			}
			if (site.split_callee) {
				// emitted above
			} else if (site.partition && !dual) {
				emitted = emit_partition(0);
			} else if (site.partition) {
				const auto select = static_cast<uint32_t>(result.size());
				result.push_back(SCbranchScc1);
				std::vector<uint32_t> exits;
				emitted = emit_partition(SetScc0);
				exits.push_back(branch());
				relocations.emplace_back(select, static_cast<uint32_t>(result.size()));
				emitted = emitted && emit_partition(SetScc1);
				merge(exits);
			} else if (dual) {
				const auto select = static_cast<uint32_t>(result.size());
				result.push_back(SCbranchScc1);
				std::vector<uint32_t> exits;
				emitted = emit_groups(SetScc0);
				exits.push_back(branch());
				relocations.emplace_back(select, static_cast<uint32_t>(result.size()));
				emitted = emitted && emit_groups(SetScc1);
				merge(exits);
			} else {
				emitted = emit_groups(0);
			}
			if (!emitted) {
				if (reason.empty()) reason = "the expanded program exceeds the code bound";
				return false;
			}
			// KYTY_CALL_WATERFALL_CLEAR: every callee leaves a waterfall's flag clear on the console
			// (its lanes would be called forever otherwise); a body run on data the emulator does not
			// reproduce may not, so the flag is cleared again where the call returns.
			if (GetCodegenOptions().call_waterfall_clear && !site.addresses.empty()) {
				const auto flag =
				    WaterfallFlag(program.instructions, static_cast<size_t>(&inst - program.instructions.data()));
				if (flag != UINT32_MAX) {
					result.push_back(0x7e000280u | (flag << 17u)); // V_MOV_B32 v[flag], 0
				}
			}
		}
		if (result.size() > MaxExpandedShaderWords) {
			reason = "the expanded program exceeds the code bound";
			return false;
		}
	}
	for (const auto& [at, target]: main_branches) {
		const auto found = main_pc.find(target);
		if (found == main_pc.end()) {
			reason = fmt::format("a branch targets pc 0x{:x} outside the decoded program", target);
			return false;
		}
		relocations.emplace_back(at, found->second);
	}
	if (!RelocateBranches(result, relocations, expansion.bodies, reason)) return false;
	expansion.code = std::move(result);
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
