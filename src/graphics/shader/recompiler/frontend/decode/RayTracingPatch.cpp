#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include "common/logging/log.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

namespace Libs::Graphics::ShaderRecompiler::Decoder {

namespace {

Operand Sgpr(uint32_t reg) {
	Operand operand;
	operand.kind  = OperandKind::Sgpr;
	operand.reg   = reg;
	operand.value = reg;
	return operand;
}

Operand Vgpr(uint32_t reg) {
	Operand operand;
	operand.kind  = OperandKind::Vgpr;
	operand.reg   = reg;
	operand.value = reg;
	return operand;
}

uint32_t EndPc(const Instruction& inst) {
	return inst.pc + inst.word_count * 4u;
}

const Instruction* FindInstruction(const Program& program, uint32_t pc) {
	const auto found = std::ranges::lower_bound(program.instructions, pc, {}, &Instruction::pc);
	return found != program.instructions.end() && found->pc == pc ? &*found : nullptr;
}

bool CheckSite(const Program& program, const RayTracingSite& site) {
	const auto expect = [&](uint32_t pc, Opcode opcode) {
		const auto* inst = FindInstruction(program, pc);
		return inst != nullptr && (opcode == Opcode::UNKNOWN || inst->opcode == opcode);
	};
	const auto* entry = FindInstruction(program, site.entry_pc);
	return entry != nullptr && EndPc(*entry) == site.loop_pc && !IsDirectBranch(entry->opcode) &&
	       expect(site.loop_pc, Opcode::S_MOV_B64) && expect(site.setup_pc, Opcode::UNKNOWN) &&
	       expect(site.cull_exit_pc, Opcode::UNKNOWN) &&
	       expect(site.blas_pc, Opcode::IMAGE_BVH_INTERSECT_RAY) && expect(site.leaf_pc, Opcode::UNKNOWN) &&
	       expect(site.leaf_end_pc, Opcode::S_MOV_B64) && expect(site.exit_pc, Opcode::UNKNOWN);
}

// The rewritten program before its pcs are assigned: branch targets are guest pcs (resolved
// through the new position of the guest instruction, or a redirect) or labels of inserted
// instructions.
class Layout {
public:
	enum class Target { None, Guest, Label };

	struct Item {
		Instruction inst;
		Target      target = Target::None;
		uint32_t    value  = 0;
	};

	uint32_t NewLabel() { return m_next_label++; }
	// The label refers to the next item added.
	void Bind(uint32_t label) { m_bindings.emplace_back(label, m_items.size()); }
	// Branches of guest code to `pc` go to `label` instead (inserted code before the guest
	// instruction at `pc` that every guest path to it runs first).
	void Redirect(uint32_t pc, uint32_t label) { m_redirects[pc] = label; }

	Instruction& AddGuest(const Instruction& inst) {
		Item item {inst};
		// The guest code stays where it is in memory: PC-relative addresses (S_GETPC_B64 reads
		// of data after the program) keep their guest pc.
		if (item.inst.guest_pc == UINT32_MAX) {
			item.inst.guest_pc = inst.pc;
		}
		if (IsDirectBranch(inst.opcode)) {
			item.target = Target::Guest;
			item.value  = inst.branch_target;
		}
		m_guest_items.emplace(inst.pc, m_items.size());
		m_items.push_back(std::move(item));
		return m_items.back().inst;
	}
	// A guest instruction placed again elsewhere (its branches set by the caller).
	Instruction& AddCopy(const Instruction& inst) {
		Item item {inst};
		if (item.inst.guest_pc == UINT32_MAX) {
			item.inst.guest_pc = inst.pc;
		}
		m_items.push_back(std::move(item));
		return m_items.back().inst;
	}
	void SetBranch(Target target, uint32_t value) {
		m_items.back().target = target;
		m_items.back().value  = value;
	}
	void AddBranch(Opcode opcode, Target target, uint32_t value) {
		Instruction inst;
		inst.family     = Family::SOPP;
		inst.opcode     = opcode;
		inst.word_count = 1;
		m_items.push_back({inst, target, value});
	}
	Instruction& AddPseudo(Opcode opcode, uint32_t site) {
		Instruction inst;
		inst.family     = Family::Unknown;
		inst.opcode     = opcode;
		inst.word_count = 1;
		inst.offset     = site;
		m_items.push_back({inst});
		return m_items.back().inst;
	}

	// Assigns the pcs from `first_pc`; false when a target is unknown.
	bool Finish(uint32_t first_pc, uint32_t guest_end_pc, std::vector<Instruction>& out) {
		std::vector<uint32_t> pcs(m_items.size() + 1);
		uint32_t              pc = first_pc;
		for (size_t i = 0; i < m_items.size(); i++) {
			pcs[i] = pc;
			pc += m_items[i].inst.word_count * 4u;
		}
		pcs[m_items.size()] = pc;
		std::unordered_map<uint32_t, uint32_t> labels;
		for (const auto& [label, index]: m_bindings) {
			labels.emplace(label, pcs[index]);
		}
		out.clear();
		out.reserve(m_items.size());
		for (size_t i = 0; i < m_items.size(); i++) {
			auto inst = m_items[i].inst;
			inst.pc   = pcs[i];
			if (m_items[i].target == Target::Guest) {
				const auto target = m_items[i].value;
				if (const auto redirect = m_redirects.find(target); redirect != m_redirects.end()) {
					const auto found = labels.find(redirect->second);
					if (found == labels.end()) {
						return false;
					}
					inst.branch_target = found->second;
				} else if (target == guest_end_pc) {
					inst.branch_target = pc;
				} else if (const auto found = m_guest_items.find(target); found != m_guest_items.end()) {
					inst.branch_target = pcs[found->second];
				} else {
					return false;
				}
			} else if (m_items[i].target == Target::Label) {
				const auto found = labels.find(m_items[i].value);
				if (found == labels.end()) {
					return false;
				}
				inst.branch_target = found->second;
			}
			out.push_back(std::move(inst));
		}
		return true;
	}

private:
	std::vector<Item>                        m_items;
	std::unordered_map<uint32_t, size_t>     m_guest_items;
	std::vector<std::pair<uint32_t, size_t>> m_bindings;
	std::unordered_map<uint32_t, uint32_t>   m_redirects;
	uint32_t                                 m_next_label = 0;
};

// Places a copy of the guest instructions in [begin, end): branches within the range go to the
// copy; `outside` gives the label of any other target and of the fall through past the end.
template <typename Outside>
bool AddRangeCopy(Layout& layout, const Program& program, uint32_t begin, uint32_t end,
                  Outside&& outside) {
	const auto first = std::ranges::lower_bound(program.instructions, begin, {}, &Instruction::pc);
	if (first == program.instructions.end() || first->pc != begin) {
		return false;
	}
	std::unordered_map<uint32_t, uint32_t> copy_labels;
	std::vector<const Instruction*>        range;
	for (auto it = first; it != program.instructions.end() && it->pc < end; ++it) {
		if (it->opcode == Opcode::S_SETPC_B64 ||
		    it->opcode == Opcode::S_GETPC_B64 || it->opcode == Opcode::S_ENDPGM ||
		    it->opcode == Opcode::S_KYTY_CALL_STATE || it->opcode == Opcode::UNSUPPORTED) {
			return false;
		}
		copy_labels.emplace(it->pc, layout.NewLabel());
		range.push_back(&*it);
	}
	if (range.empty() || EndPc(*range.back()) != end) {
		return false;
	}
	for (const auto* inst: range) {
		layout.Bind(copy_labels.at(inst->pc));
		layout.AddCopy(*inst);
		if (!IsDirectBranch(inst->opcode)) {
			continue;
		}
		const auto target = inst->branch_target;
		if (const auto found = copy_labels.find(target); found != copy_labels.end()) {
			layout.SetBranch(Layout::Target::Label, found->second);
			continue;
		}
		const auto label = outside(target);
		if (label == UINT32_MAX) {
			return false;
		}
		layout.SetBranch(Layout::Target::Label, label);
	}
	if (range.back()->opcode != Opcode::S_BRANCH) {
		const auto label = outside(end);
		if (label == UINT32_MAX) {
			return false;
		}
		layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, label);
	}
	return true;
}

// The result registers RT_VERIFY_SAVE / RT_VERIFY_CHECK compare (NoRegister: none).
std::array<uint32_t, 8> VerifyRegisters(const RayTracingSite& site) {
	return {site.result_t,        site.result_u,           site.result_v,    site.result_word,
	        site.result_node,     site.result_instance,    site.result_instance + 1u,
	        site.alpha_counter};
}

// The native traversal of a PsrClosest site, placed between its entry instruction and the loop
// header: RT_INIT falls back to the loop header, else the native loop runs and exits after the
// loop (or, verifying, runs the guest loop again from the same state).
bool AddNativeTraversal(Layout& layout, const Program& program, const RayTracingSite& site,
                        uint32_t index, bool verify, uint32_t verify_label) {
	if (verify) {
		// dmask 2: the registers' values at the entry, for the guest loop's run after the native one.
		auto&      snapshot = layout.AddPseudo(Opcode::RT_VERIFY_SAVE, index);
		const auto regs     = VerifyRegisters(site);
		for (uint32_t i = 0; i < regs.size(); i++) {
			snapshot.image_nsa_addr[i] = regs[i];
		}
		snapshot.dmask = 2;
	}
	auto& init = layout.AddPseudo(Opcode::RT_INIT, index);
	for (uint32_t i = 0; i < 3; i++) {
		init.image_nsa_addr[i]     = site.origin[i];
		init.image_nsa_addr[3 + i] = site.direction[i];
	}
	init.image_nsa_addr[6] = site.t_max;
	init.image_nsa_addr[7] = site.cull_mask;
	init.src0              = Sgpr(site.header_sgpr);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Guest, site.loop_pc);

	const auto loop_label = layout.NewLabel(), exit_label = layout.NewLabel();
	const auto leaf_label = layout.NewLabel(), tail_label = layout.NewLabel();
	const auto cull_label = layout.NewLabel();

	// The wave's next instance (the smallest any lane has a candidate in), or the exit.
	layout.Bind(loop_label);
	auto& select = layout.AddPseudo(Opcode::RT_SELECT, index);
	select.dst   = Sgpr(site.node_sgpr);
	layout.AddBranch(Opcode::S_CBRANCH_SCC0, Layout::Target::Label, exit_label);
	// The guest instance setup; it continues with the lanes' candidates instead of the BLAS stack.
	if (!AddRangeCopy(layout, program, site.setup_pc, site.blas_pc, [&](uint32_t target) {
		    return target == site.blas_pc ? leaf_label : target == site.cull_exit_pc ? cull_label : UINT32_MAX;
	    })) {
		return false;
	}
	layout.Bind(leaf_label);
	auto& leaf = layout.AddPseudo(Opcode::RT_LEAF, index);
	leaf.dst   = Vgpr(site.leaf_node);
	leaf.dst2  = Vgpr(site.leaf_word);
	leaf.image_nsa_addr[0] = site.result_t;
	leaf.image_nsa_addr[1] = site.result_node;
	leaf.image_nsa_addr[2] = site.result_instance;
	// The guest test of the lane's triangle node (the hardware found the triangle; the guest's
	// exact test gives the leaf code its distance and barycentrics).
	const auto* test = FindInstruction(program, site.blas_pc);
	if (test == nullptr) {
		return false;
	}
	layout.AddCopy(*test);
	// The guest leaf code of one triangle; it ends where the BLAS loop pops its next node.
	if (!AddRangeCopy(layout, program, site.leaf_pc, site.leaf_end_pc, [&](uint32_t target) {
		    return target == site.leaf_end_pc ? tail_label : UINT32_MAX;
	    })) {
		return false;
	}
	// The end of an iteration: lanes whose leaf code accepted the hit confirm their candidate.
	layout.Bind(tail_label);
	auto& resolve = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	resolve.image_nsa_addr[0] = site.result_t;
	resolve.image_nsa_addr[1] = site.result_node;
	resolve.image_nsa_addr[2] = site.result_instance;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);
	// The instance setup culled the instance (its mask test): the candidates are dropped.
	layout.Bind(cull_label);
	auto& cull = layout.AddPseudo(Opcode::RT_RESOLVE, index);
	cull.image_nsa_addr[0] = site.result_t;
	cull.image_nsa_addr[1] = site.result_node;
	cull.image_nsa_addr[2] = site.result_instance;
	cull.dmask             = 1;
	layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Label, loop_label);

	layout.Bind(exit_label);
	layout.AddPseudo(Opcode::RT_EXIT, index);
	if (verify) {
		// The native results are kept, the registers get their values at RT_INIT back and the
		// guest loop runs from the same state; RT_VERIFY_CHECK at the exit compares.
		auto& save = layout.AddPseudo(Opcode::RT_VERIFY_SAVE, index);
		const auto regs = VerifyRegisters(site);
		for (uint32_t i = 0; i < regs.size(); i++) {
			save.image_nsa_addr[i] = regs[i];
		}
		// The guest loop starts at the root with the stack pointer's entry value.
		save.dst              = Sgpr(site.node_sgpr);
		save.secondary_offset = site.root;
		save.dst2             = site.stack_sgpr != NoRegister ? Sgpr(site.stack_sgpr) : Operand {};
		save.data_format      = site.stack_value;
		layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.loop_pc);
		(void)verify_label;
	} else {
		layout.AddBranch(Opcode::S_BRANCH, Layout::Target::Guest, site.exit_pc);
	}
	return true;
}

} // namespace

// ---- Structural derivation of the traversal sites (see the top of RayTracingPatch.h). ----
//
// PsrClosest sites (Miles Morales' kernels; pcs of 0x6d9e's first traversal):
//   loop header   S_MOV_B64 <e>, exec (0x710, the target of the stack pop's backward branch),
//                 S_AND_B32 vcc_lo, <node>, 7, S_ADD_I32 vcc_lo, vcc_lo, -4, S_CMP_LE_U32 vcc_lo, 2,
//                 S_CBRANCH_SCC1 <TLAS box test>
//   instance      S_CMP_LG_U32 -1, <node> ... S_AND_SAVEEXEC_B64 <s>, vcc, S_CBRANCH_EXECZ <skip>;
//                 the setup follows (record = instances + (node >> 3) * 128, mask test
//                 S_CBRANCH_SCC0 <skip>)
//   BLAS test     the first IMAGE_BVH_INTERSECT_RAY after the setup (node VGPR, object space ray)
//   leaf          V_AND_B32 <x>, 7, <node>, V_CMP_LT_U32 vcc, 3, <x>, S_AND_SAVEEXEC_B64 <b>, vcc,
//                 S_CBRANCH_EXECZ, ..., S_ANDN2_B64 exec, <b>, exec, S_CBRANCH_EXECZ <leaf end>;
//                 the leaf code follows, up to S_MOV_B64 exec, <b> at the leaf end
//   accepted hit  after V_CMPX_NE_U32 exec, 0, <v> in the leaf code: V_MOV_B32s of the distance
//                 (into the TLAS test's t max), the barycentrics, the info word, the node and the
//                 instance record pair
//   next entry    BUFFER_LOAD_DWORDX2 <a> (the leaf entry), V_MOV_B32 <node>, <a>: the info word is
//                 <a> + 1
//   TLAS test     the IMAGE_BVH_INTERSECT_RAY after the box branch target (NSA: t max, origin,
//                 direction); its T# from S_BFE_U64 <t>, <header>, 0x280008
//   exit          the stack pop S_CMP_LG_U32 0, <sp>, S_CBRANCH_SCC0 <exit> before the backward branch
// The TLAS header address is S_BUFFER_LOAD_DWORDX2 <header>, <v#> @k from the V#
// S_LOAD_DWORDX4 <v#>, <user> @j of a user data pointer nothing writes before; the root is the
// S_MOV_B32 <node>, 0x25 before the loop.
namespace {

bool IsSgpr(const Operand& operand, uint32_t reg) {
	return operand.kind == OperandKind::Sgpr && operand.reg == reg;
}

bool IsVgpr(const Operand& operand, uint32_t reg) {
	return operand.kind == OperandKind::Vgpr && operand.reg == reg;
}

bool IsVccLo(const Operand& operand) {
	return operand.kind == OperandKind::VccLo;
}

bool IsExecLo(const Operand& operand) {
	return operand.kind == OperandKind::ExecLo;
}

bool IsConstant(const Operand& operand, uint32_t value) {
	return (operand.kind == OperandKind::IntegerInlineConstant ||
	        operand.kind == OperandKind::LiteralConstant) &&
	       operand.value == value;
}

bool IsImmediate(const Operand& operand) {
	return operand.kind == OperandKind::IntegerInlineConstant ||
	       operand.kind == OperandKind::LiteralConstant;
}

class SiteFinder {
public:
	explicit SiteFinder(const Program& program): m_inst(program.instructions) {}

	// The sites of the program and its TLAS source.
	bool Find(RayTracingPatch& patch) {
		std::vector<RayTracingSite> sites;
		uint32_t user_sgpr = NoRegister, vsharp_offset = 0, header_offset = 0, root = 0;
		for (size_t header = 1; header + 5 < m_inst.size(); header++) {
			if (!IsClosestHeader(header)) {
				continue;
			}
			const auto back = BackwardBranchTo(header);
			if (back == npos) {
				continue;
			}
			RayTracingSite site;
			std::string    why;
			if (!Derive(header, back, site, why)) {
				patch.unmatched++;
				patch.report += fmt::format(" loop@0x{:x}: {};", m_inst[header].pc, why);
				continue;
			}
			uint32_t site_user = NoRegister, site_vsharp = 0, site_offset = 0;
			if (!FindTlasSource(header, site.header_sgpr, site_user, site_vsharp, site_offset)) {
				patch.unmatched++;
				patch.report += fmt::format(" loop@0x{:x}: TLAS source not traced;", m_inst[header].pc);
				continue;
			}
			if (user_sgpr != NoRegister &&
			    (site_user != user_sgpr || site_vsharp != vsharp_offset || site_offset != header_offset ||
			     site.root != root)) {
				patch.unmatched++;
				patch.report += fmt::format(" loop@0x{:x}: another TLAS;", m_inst[header].pc);
				continue;
			}
			user_sgpr     = site_user;
			vsharp_offset = site_vsharp;
			header_offset = site_offset;
			root          = site.root;
			patch.report += fmt::format(" site@0x{:x} (blas 0x{:x}, leaf 0x{:x}-0x{:x}, exit 0x{:x});",
			                            site.loop_pc, site.blas_pc, site.leaf_pc, site.leaf_end_pc,
			                            site.exit_pc);
			sites.push_back(site);
		}
		if (sites.empty() || sites.size() > 4) {
			return false;
		}
		patch.sites         = std::move(sites);
		patch.user_sgpr     = user_sgpr;
		patch.vsharp_offset = vsharp_offset;
		patch.header_offset = header_offset;
		patch.root          = root;
		return true;
	}

private:
	static constexpr size_t npos = SIZE_MAX;

	size_t IndexOf(uint32_t pc) const {
		const auto found = std::ranges::lower_bound(m_inst, pc, {}, &Instruction::pc);
		return found != m_inst.end() && found->pc == pc ? static_cast<size_t>(found - m_inst.begin()) : npos;
	}

	// The last backward S_BRANCH to instruction `target`, or npos.
	size_t BackwardBranchTo(size_t target) const {
		size_t found = npos;
		for (size_t i = target + 1; i < m_inst.size(); i++) {
			if (m_inst[i].opcode == Opcode::S_BRANCH && m_inst[i].branch_target == m_inst[target].pc) {
				found = i;
			}
		}
		return found;
	}

	size_t Next(size_t from, Opcode opcode, size_t limit = SIZE_MAX) const {
		for (size_t i = from; i < m_inst.size() && i < limit; i++) {
			if (m_inst[i].opcode == opcode) {
				return i;
			}
		}
		return npos;
	}

	// The address VGPR of component `k` of an IMAGE_BVH_INTERSECT_RAY (node, t max, origin,
	// direction, inverse direction).
	static uint32_t AddressReg(const Instruction& inst, uint32_t k) {
		if (inst.src0.kind != OperandKind::Vgpr) {
			return NoRegister;
		}
		if (k == 0) {
			return inst.src0.reg;
		}
		return inst.image_nsa_dwords != 0 ? inst.image_nsa_addr[k - 1] : inst.src0.reg + k;
	}

	// S_AND_B32 vcc_lo, <node>, 7 at `i` followed by S_ADD_I32 vcc_lo, vcc_lo, -4.
	bool IsNodeType(size_t i) const {
		if (i + 1 >= m_inst.size()) {
			return false;
		}
		const auto& a = m_inst[i];
		const auto& b = m_inst[i + 1];
		return a.opcode == Opcode::S_AND_B32 && IsVccLo(a.dst) && a.src0.kind == OperandKind::Sgpr &&
		       IsConstant(a.src1, 7) && b.opcode == Opcode::S_ADD_I32 && IsVccLo(b.dst) &&
		       IsVccLo(b.src0) && b.src1.kind == OperandKind::IntegerInlineConstant && b.src1.signed_val == -4;
	}

	bool IsClosestHeader(size_t i) const {
		return i + 4 < m_inst.size() && m_inst[i].opcode == Opcode::S_MOV_B64 &&
		       m_inst[i].dst.kind == OperandKind::Sgpr && IsExecLo(m_inst[i].src0) && IsNodeType(i + 1) &&
		       m_inst[i + 3].opcode == Opcode::S_CMP_LE_U32 && IsVccLo(m_inst[i + 3].src0) &&
		       IsConstant(m_inst[i + 3].src1, 2) && m_inst[i + 4].opcode == Opcode::S_CBRANCH_SCC1;
	}

	bool Derive(size_t header, size_t back, RayTracingSite& site, std::string& why) const {
		const auto fail = [&](const char* reason) {
			why = reason;
			return false;
		};
		site.kind      = RayTracingKind::PsrClosest;
		site.entry_pc  = m_inst[header - 1].pc;
		site.loop_pc   = m_inst[header].pc;
		site.node_sgpr = m_inst[header + 1].src0.reg;
		if (IsDirectBranch(m_inst[header - 1].opcode)) {
			return fail("the entry is a branch");
		}
		const auto box = IndexOf(m_inst[header + 4].branch_target);
		if (box == npos || box <= header || box >= back) {
			return fail("no TLAS box block");
		}
		// The instance: S_AND_SAVEEXEC_B64, S_CBRANCH_EXECZ <skip>, then the setup.
		const auto saveexec = Next(header + 5, Opcode::S_AND_SAVEEXEC_B64, header + 12);
		if (saveexec == npos || m_inst[saveexec + 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return fail("no instance branch");
		}
		const auto skip_pc = m_inst[saveexec + 1].branch_target;
		const auto setup   = saveexec + 2;
		site.setup_pc      = m_inst[setup].pc;
		// The BLAS test: the first BVH test after the setup, before the TLAS box block.
		const auto blas = Next(setup, Opcode::IMAGE_BVH_INTERSECT_RAY, box);
		if (blas == npos || m_inst[blas].src0.kind != OperandKind::Vgpr || m_inst[blas].image_nsa_dwords != 0) {
			return fail("no BLAS test");
		}
		site.blas_pc   = m_inst[blas].pc;
		site.leaf_node = m_inst[blas].src0.reg;
		// The setup's only branch leaves to the skip (the instance mask culls it).
		site.cull_exit_pc = skip_pc;
		for (size_t i = setup; i < blas; i++) {
			if (IsDirectBranch(m_inst[i].opcode) && m_inst[i].branch_target != skip_pc) {
				return fail("a setup branch other than the cull");
			}
		}
		// The leaf: the node type test of the BLAS test's node.
		if (blas + 4 >= back || m_inst[blas + 1].opcode != Opcode::V_AND_B32 ||
		    !IsConstant(m_inst[blas + 1].src0, 7) || !IsVgpr(m_inst[blas + 1].src1, site.leaf_node) ||
		    m_inst[blas + 2].opcode != Opcode::V_CMP_LT_U32 || !IsConstant(m_inst[blas + 2].src0, 3) ||
		    m_inst[blas + 3].opcode != Opcode::S_AND_SAVEEXEC_B64 || m_inst[blas + 3].dst.kind != OperandKind::Sgpr ||
		    m_inst[blas + 4].opcode != Opcode::S_CBRANCH_EXECZ) {
			return fail("no node type test after the BLAS test");
		}
		const auto box_save = m_inst[blas + 3].dst.reg;
		const auto triangle = IndexOf(m_inst[blas + 4].branch_target);
		if (triangle == npos || triangle + 2 >= back || m_inst[triangle].opcode != Opcode::S_ANDN2_B64 ||
		    !IsExecLo(m_inst[triangle].dst) || !IsSgpr(m_inst[triangle].src0, box_save) ||
		    !IsExecLo(m_inst[triangle].src1) || m_inst[triangle + 1].opcode != Opcode::S_CBRANCH_EXECZ) {
			return fail("no triangle branch");
		}
		site.leaf_pc     = m_inst[triangle + 2].pc;
		site.leaf_end_pc = m_inst[triangle + 1].branch_target;
		const auto leaf_end = IndexOf(site.leaf_end_pc);
		if (leaf_end == npos || leaf_end <= triangle || leaf_end >= back ||
		    m_inst[leaf_end].opcode != Opcode::S_MOV_B64 || !IsExecLo(m_inst[leaf_end].dst) ||
		    !IsSgpr(m_inst[leaf_end].src0, box_save)) {
			return fail("no leaf end");
		}
		// The TLAS test.
		const auto tlas = Next(box, Opcode::IMAGE_BVH_INTERSECT_RAY, std::min(back, box + 24));
		if (tlas == npos || m_inst[tlas].src1.kind != OperandKind::Sgpr) {
			return fail("no TLAS test");
		}
		const auto& test = m_inst[tlas];
		site.t_max       = AddressReg(test, 1);
		for (uint32_t c = 0; c < 3; c++) {
			site.origin[c]    = AddressReg(test, 2 + c);
			site.direction[c] = AddressReg(test, 5 + c);
		}
		site.header_sgpr = NoRegister;
		for (size_t i = header; i-- > 0;) {
			const auto& bfe = m_inst[i];
			if (bfe.opcode == Opcode::S_BFE_U64 && IsSgpr(bfe.dst, test.src1.reg)) {
				if (bfe.src0.kind == OperandKind::Sgpr && IsConstant(bfe.src1, 0x280008)) {
					site.header_sgpr = bfe.src0.reg;
				}
				break;
			}
		}
		if (site.header_sgpr == NoRegister || site.t_max == NoRegister) {
			return fail("no TLAS T# from the header");
		}
		// The accepted hit in the leaf code.
		size_t accept = npos;
		for (size_t i = triangle + 2; i + 2 < leaf_end && accept == npos; i++) {
			if (m_inst[i].opcode == Opcode::V_CMPX_NE_U32 && IsConstant(m_inst[i].src0, 0) &&
			    m_inst[i + 1].opcode == Opcode::S_CBRANCH_EXECZ && m_inst[i + 2].opcode == Opcode::V_MOV_B32) {
				accept = i + 2;
			}
		}
		if (accept == npos) {
			return fail("no accepted hit");
		}
		site.result_t = site.result_node = site.result_instance = NoRegister;
		uint32_t bary = 0;
		for (size_t i = accept; i < leaf_end && m_inst[i].opcode == Opcode::V_MOV_B32; i++) {
			const auto& move = m_inst[i];
			if (move.dst.kind != OperandKind::Vgpr) {
				return fail("an accepted hit move to no VGPR");
			}
			if (move.src0.kind == OperandKind::Sgpr) {
				if (site.result_instance == NoRegister) {
					site.result_instance = move.dst.reg;
					if (i + 1 >= leaf_end || m_inst[i + 1].opcode != Opcode::V_MOV_B32 ||
					    !IsVgpr(m_inst[i + 1].dst, move.dst.reg + 1) ||
					    !IsSgpr(m_inst[i + 1].src0, move.src0.reg + 1)) {
						return fail("no instance record pair");
					}
					i++;
				}
				continue;
			}
			if (move.src0.kind != OperandKind::Vgpr) {
				return fail("an accepted hit move of no register");
			}
			if (move.dst.reg == site.t_max) {
				site.result_t = move.dst.reg;
			} else if (move.src0.reg == site.leaf_node) {
				site.result_node = move.dst.reg;
			} else if (bary < 2) {
				(bary++ == 0 ? site.result_u : site.result_v) = move.dst.reg;
			} else {
				site.result_word = move.dst.reg;
			}
		}
		if (site.result_t == NoRegister || site.result_node == NoRegister ||
		    site.result_instance == NoRegister) {
			return fail("the accepted hit lacks the distance, node or instance");
		}
		// The alpha test counter: V_ADD_NC_U32 <c>, 1, <c> in the leaf code.
		for (size_t i = triangle + 2; i < leaf_end; i++) {
			const auto& add = m_inst[i];
			if (add.opcode == Opcode::V_ADD_NC_U32 && add.dst.kind == OperandKind::Vgpr && IsConstant(add.src0, 1) &&
			    IsVgpr(add.src1, add.dst.reg)) {
				site.alpha_counter = add.dst.reg;
				break;
			}
		}
		// The next leaf entry: BUFFER_LOAD_DWORDX2 <a>, then V_MOV_B32 <node>, <a>.
		site.leaf_word = NoRegister;
		for (size_t i = leaf_end; i < back && site.leaf_word == NoRegister; i++) {
			const auto& move = m_inst[i];
			if (move.opcode == Opcode::V_MOV_B32 && IsVgpr(move.dst, site.leaf_node) &&
			    move.src0.kind == OperandKind::Vgpr) {
				for (size_t j = i; j-- > leaf_end;) {
					if (m_inst[j].opcode == Opcode::BUFFER_LOAD_DWORDX2 && IsVgpr(m_inst[j].dst, move.src0.reg)) {
						site.leaf_word = move.src0.reg + 1u;
						break;
					}
				}
				if (site.leaf_word == NoRegister) {
					return fail("the next node is not a leaf entry load");
				}
			}
		}
		if (site.leaf_word == NoRegister) {
			return fail("no next leaf entry");
		}
		// The stack pop before the backward branch and the exit.
		size_t pop = npos;
		for (size_t i = back; i-- > box && i + 6 > back && pop == npos;) {
			if (m_inst[i].opcode == Opcode::S_CMP_LG_U32 && IsConstant(m_inst[i].src0, 0) &&
			    m_inst[i].src1.kind == OperandKind::Sgpr && m_inst[i + 1].opcode == Opcode::S_CBRANCH_SCC0) {
				pop = i;
			}
		}
		if (pop == npos) {
			return fail("no stack pop");
		}
		site.exit_pc    = m_inst[pop + 1].branch_target;
		site.stack_sgpr = m_inst[pop].src1.reg;
		if (IndexOf(site.exit_pc) == npos || site.exit_pc <= m_inst[back].pc) {
			return fail("the exit is not after the loop");
		}
		// The root and the stack pointer's value at the entry.
		if (!FindConstant(header, site.node_sgpr, site.root) || site.root != 0x25u) {
			return fail("no root 0x25");
		}
		if (!FindConstant(header, site.stack_sgpr, site.stack_value)) {
			site.stack_sgpr = NoRegister;
		}
		return true;
	}

	// The immediate an S_MOV_B32 gives `reg` last before the loop (false: another write).
	bool FindConstant(size_t header, uint32_t reg, uint32_t& value) const {
		for (size_t i = header; i-- > 0;) {
			const auto& inst = m_inst[i];
			if (!IsSgpr(inst.dst, reg)) {
				continue;
			}
			if (inst.opcode != Opcode::S_MOV_B32 || !IsImmediate(inst.src0)) {
				return false;
			}
			value = inst.src0.value;
			return true;
		}
		return false;
	}

	// The TLAS header pair: S_BUFFER_LOAD_DWORDX2 <pair>, <v#> @k, the V# S_LOAD_DWORDX4 <v#>,
	// <user> @j from a pair nothing writes before (user data).
	bool FindTlasSource(size_t header, uint32_t header_sgpr, uint32_t& user_sgpr, uint32_t& vsharp_offset,
	                    uint32_t& offset) const {
		size_t load = npos;
		for (size_t i = header; i-- > 0;) {
			if (IsSgpr(m_inst[i].dst, header_sgpr)) {
				load = i;
				break;
			}
		}
		if (load == npos || m_inst[load].opcode != Opcode::S_BUFFER_LOAD_DWORDX2 ||
		    m_inst[load].src0.kind != OperandKind::Sgpr ||
		    (m_inst[load].src1.kind != OperandKind::Unknown && m_inst[load].src1.kind != OperandKind::Null)) {
			return false;
		}
		const auto vsharp = m_inst[load].src0.reg;
		offset            = m_inst[load].offset;
		size_t source     = npos;
		for (size_t i = load; i-- > 0;) {
			const auto& dst = m_inst[i].dst;
			if (dst.kind == OperandKind::Sgpr && dst.reg <= vsharp + 3u && dst.reg + 3u >= vsharp) {
				source = i;
				break;
			}
		}
		if (source == npos || m_inst[source].opcode != Opcode::S_LOAD_DWORDX4 ||
		    !IsSgpr(m_inst[source].dst, vsharp) || m_inst[source].src0.kind != OperandKind::Sgpr ||
		    (m_inst[source].src1.kind != OperandKind::Unknown && m_inst[source].src1.kind != OperandKind::Null)) {
			return false;
		}
		const auto base = m_inst[source].src0.reg;
		for (size_t j = 0; j < source; j++) {
			const auto& dst = m_inst[j].dst;
			if (dst.kind == OperandKind::Sgpr && (dst.reg == base || dst.reg == base + 1u)) {
				return false;
			}
		}
		user_sgpr     = base;
		vsharp_offset = m_inst[source].offset;
		return true;
	}

	const std::vector<Instruction>& m_inst;
};

} // namespace

bool DeriveRayTracingPatch(const Program& program, RayTracingPatch& patch) {
	if (program.instructions.empty() ||
	    std::ranges::none_of(program.instructions,
	                         [](const Instruction& inst) { return inst.opcode == Opcode::IMAGE_BVH_INTERSECT_RAY; })) {
		return false;
	}
	SiteFinder finder(program);
	if (!finder.Find(patch)) {
		return false;
	}
	return std::ranges::all_of(patch.sites, [&](const RayTracingSite& site) { return CheckSite(program, site); });
}

bool ApplyRayTracingPatch(const RayTracingPatch& patch, Program& program, bool verify) {
	if (program.instructions.empty()) {
		return false;
	}
	const auto fail = [&](uint32_t pc, const char* reason) {
		LOGF("RayTracingPatch: site at 0x%x: %s; not patched\n", pc, reason);
		return false;
	};
	for (const auto& site: patch.sites) {
		if (!CheckSite(program, site)) {
			return fail(site.entry_pc, "the code does not match");
		}
	}
	// The native code goes right after each entry instruction, in program order, so the
	// structured control flow sees it as a region of the shader; the guest instructions move.
	Layout                layout;
	std::vector<uint32_t> verify_labels(patch.sites.size(), UINT32_MAX);
	if (verify) {
		for (uint32_t index = 0; index < patch.sites.size(); index++) {
			verify_labels[index] = layout.NewLabel();
			layout.Redirect(patch.sites[index].exit_pc, verify_labels[index]);
		}
	}
	for (const auto& inst: program.instructions) {
		for (uint32_t index = 0; index < patch.sites.size(); index++) {
			if (verify && patch.sites[index].exit_pc == inst.pc) {
				// Every path to the exit (the guest loop's) compares first.
				layout.Bind(verify_labels[index]);
				auto& check     = layout.AddPseudo(Opcode::RT_VERIFY_CHECK, index);
				const auto regs = VerifyRegisters(patch.sites[index]);
				for (uint32_t i = 0; i < regs.size(); i++) {
					check.image_nsa_addr[i] = regs[i];
				}
			}
		}
		layout.AddGuest(inst);
		for (uint32_t index = 0; index < patch.sites.size(); index++) {
			if (patch.sites[index].entry_pc == inst.pc &&
			    !AddNativeTraversal(layout, program, patch.sites[index], index, verify, verify_labels[index])) {
				return fail(inst.pc, "the instance setup or the leaf code cannot be copied");
			}
		}
	}
	std::vector<Instruction> patched;
	if (!layout.Finish(program.instructions.front().pc, EndPc(program.instructions.back()), patched)) {
		return fail(0, "a branch target is unknown");
	}
	program.instructions       = std::move(patched);
	program.ray_tracing_source = {.user_sgpr     = patch.user_sgpr,
	                              .vsharp_offset = patch.vsharp_offset,
	                              .header_offset = patch.header_offset,
	                              .root          = patch.root,
	                              .valid         = true};
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::Decoder
