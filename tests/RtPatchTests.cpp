// rt_patch_tests: the native ray tracing patch (KYTY_RT_HW, RayTracingPatch.h) on a synthetic
// decoded program with the structure of Miles Morales' sce::Psr 2.0 traversal loops (a scalar
// TLAS walk, a per-lane BLAS walk, a leaf with a closest-hit update). Checks what the site finder
// reads from the structure (the ray, the TLAS source, the registers of the instance setup, the
// leaf and the accepted hit) and that the rewrite keeps every branch target valid, in both the
// plain and the verifying form.

#include "graphics/shader/recompiler/frontend/decode/RayTracingPatch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <vector>

namespace D = Libs::Graphics::ShaderRecompiler::Decoder;

namespace {

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "RtPatchTests: failed: %s\n", text);
		g_failures++;
	}
}

D::Operand S(uint32_t reg) {
	D::Operand o;
	o.kind  = D::OperandKind::Sgpr;
	o.reg   = reg;
	o.value = reg;
	return o;
}

D::Operand V(uint32_t reg) {
	D::Operand o;
	o.kind  = D::OperandKind::Vgpr;
	o.reg   = reg;
	o.value = reg;
	return o;
}

D::Operand K(int32_t value) {
	D::Operand o;
	o.kind       = value >= -16 && value <= 64 ? D::OperandKind::IntegerInlineConstant : D::OperandKind::LiteralConstant;
	o.value      = static_cast<uint32_t>(value);
	o.signed_val = value;
	return o;
}

D::Operand Kind(D::OperandKind kind) {
	D::Operand o;
	o.kind = kind;
	return o;
}

// Builds a program: instructions get consecutive pcs (4 bytes each); branch targets are labels
// resolved at the end.
class Builder {
public:
	D::Instruction& Add(D::Opcode opcode, D::Operand dst = {}, D::Operand src0 = {}, D::Operand src1 = {}) {
		D::Instruction inst;
		inst.opcode = opcode;
		inst.dst    = dst;
		inst.src0   = src0;
		inst.src1   = src1;
		inst.family = D::Family::SOP2;
		m_insts.push_back(inst);
		return m_insts.back();
	}
	void Branch(D::Opcode opcode, int label) {
		Add(opcode).family = D::Family::SOPP;
		m_branches.push_back({m_insts.size() - 1, label});
	}
	void Label(int label) { m_labels.push_back({label, m_insts.size()}); }
	D::Instruction& Last() { return m_insts.back(); }

	D::Program Finish() {
		D::Program program;
		for (size_t i = 0; i < m_insts.size(); i++) {
			m_insts[i].pc         = static_cast<uint32_t>(i * 4);
			m_insts[i].word_count = 1;
		}
		for (const auto& [index, label]: m_branches) {
			const auto found = std::find_if(m_labels.begin(), m_labels.end(), [&](const auto& l) { return l.first == label; });
			m_insts[index].branch_target = static_cast<uint32_t>(found->second * 4);
		}
		program.instructions = m_insts;
		return program;
	}

private:
	std::vector<D::Instruction>         m_insts;
	std::vector<std::pair<size_t, int>> m_branches;
	std::vector<std::pair<int, size_t>> m_labels;
};

enum Labels { Header = 1, Box, Skip, Pop, Exit, Blas, Triangle, LeafEnd, Push };

D::Program MilesLikeProgram() {
	using O = D::Opcode;
	Builder b;
	const auto vcc  = Kind(D::OperandKind::VccLo);
	const auto exec = Kind(D::OperandKind::ExecLo);
	b.Add(O::S_LOAD_DWORDX4, S(64), S(12)).offset = 544;   // the V# of the TLAS header
	b.Add(O::S_BUFFER_LOAD_DWORDX2, S(76), S(64)).offset = 80; // the header address
	b.Add(O::S_BFE_U64, S(28), S(76), K(0x280008));        // the TLAS T#
	b.Add(O::S_MOV_B32, S(37), K(37));                     // the root
	b.Add(O::S_MOV_B32, S(80), K(0));                      // the stack pointer
	b.Add(O::V_MOV_B32, V(21), K(0x501502f9));             // t max 1e10 (the entry instruction)
	b.Label(Header);
	b.Add(O::S_MOV_B64, S(8), exec);
	b.Label(Push);
	b.Add(O::S_AND_B32, vcc, S(37), K(7));
	b.Add(O::S_ADD_I32, vcc, vcc, K(-4));
	b.Add(O::S_CMP_LE_U32, {}, vcc, K(2));
	b.Branch(O::S_CBRANCH_SCC1, Box);
	b.Add(O::S_CMP_LG_U32, {}, K(-1), S(37));
	b.Add(O::S_AND_SAVEEXEC_B64, S(14), vcc);
	b.Branch(O::S_CBRANCH_EXECZ, Skip);
	// The instance setup: mask test, BLAS root.
	b.Add(O::S_LOAD_DWORDX2, S(8), S(40)).offset = 56;
	b.Branch(O::S_CBRANCH_SCC0, Skip);
	b.Add(O::V_MOV_B32, V(20), S(8));
	// The BLAS walk.
	b.Label(Blas);
	b.Add(O::IMAGE_BVH_INTERSECT_RAY, V(16), V(20), S(32));
	b.Add(O::V_AND_B32, V(1), K(7), V(20));
	b.Add(O::V_CMP_LT_U32, vcc, K(3), V(1));
	b.Add(O::S_AND_SAVEEXEC_B64, S(42), vcc);
	b.Branch(O::S_CBRANCH_EXECZ, Triangle);
	b.Add(O::S_NOP); // the box children pushes
	b.Label(Triangle);
	b.Add(O::S_ANDN2_B64, exec, S(42), exec);
	b.Branch(O::S_CBRANCH_EXECZ, LeafEnd);
	// The leaf code: alpha test counter, accepted hit.
	b.Add(O::V_ADD_NC_U32, V(73), K(1), V(73));
	b.Add(O::V_CMPX_NE_U32, exec, K(0), V(18));
	b.Branch(O::S_CBRANCH_EXECZ, LeafEnd);
	b.Add(O::V_MOV_B32, V(21), V(1));
	b.Add(O::V_MOV_B32, V(14), V(3));
	b.Add(O::V_MOV_B32, V(15), V(65));
	b.Add(O::V_MOV_B32, V(54), V(2));
	b.Add(O::V_MOV_B32, V(55), V(20));
	b.Add(O::V_MOV_B32, V(7), S(40));
	b.Add(O::V_MOV_B32, V(8), S(41));
	b.Label(LeafEnd);
	b.Add(O::S_MOV_B64, exec, S(42));
	b.Add(O::BUFFER_LOAD_DWORDX2, V(1), V(3), S(8));
	b.Add(O::V_MOV_B32, V(20), V(1));
	b.Branch(O::S_CBRANCH_EXECNZ, Blas);
	b.Label(Skip);
	b.Add(O::S_MOV_B64, exec, S(14));
	b.Branch(O::S_BRANCH, Pop);
	// The TLAS box test.
	b.Label(Box);
	b.Add(O::V_MOV_B32, V(1), S(37));
	{
		auto& test = b.Add(O::IMAGE_BVH_INTERSECT_RAY, V(16), V(1), S(28));

		const uint32_t nsa[] = {21, 60, 62, 61, 81, 88, 9, 56, 57, 58};
		test.image_nsa_dwords = 3;
		std::copy(std::begin(nsa), std::end(nsa), test.image_nsa_addr);
	}
	b.Branch(O::S_BRANCH, Push);
	// The stack pop.
	b.Label(Pop);
	b.Add(O::S_CMP_LG_U32, {}, K(0), S(80));
	b.Branch(O::S_CBRANCH_SCC0, Exit);
	b.Add(O::S_ADD_I32, S(80), S(80), K(-1));
	b.Add(O::V_READLANE_B32, S(37), V(59), S(80));
	b.Branch(O::S_BRANCH, Header);
	b.Label(Exit);
	b.Add(O::V_CMP_NE_U32, vcc, K(-1), V(54));
	b.Add(O::S_ENDPGM);
	return b.Finish();
}

size_t CountOpcode(const D::Program& program, D::Opcode opcode) {
	return static_cast<size_t>(std::count_if(program.instructions.begin(), program.instructions.end(),
	                                         [&](const D::Instruction& inst) { return inst.opcode == opcode; }));
}

bool BranchesValid(const D::Program& program) {
	for (size_t i = 0; i < program.instructions.size(); i++) {
		const auto& inst = program.instructions[i];
		if (i + 1 < program.instructions.size() && program.instructions[i + 1].pc != inst.pc + inst.word_count * 4) {
			return false;
		}
		if (!D::IsDirectBranch(inst.opcode)) {
			continue;
		}
		const bool found = std::any_of(program.instructions.begin(), program.instructions.end(),
		                               [&](const D::Instruction& other) { return other.pc == inst.branch_target; });
		if (!found) {
			return false;
		}
	}
	return true;
}

const D::Instruction* At(const D::Program& program, uint32_t pc) {
	for (const auto& inst: program.instructions) {
		if (inst.pc == pc) {
			return &inst;
		}
	}
	return nullptr;
}

void TestFinder() {
	const auto      program = MilesLikeProgram();
	D::RayTracingPatch patch;
	Check(D::DeriveRayTracingPatch(program, patch), "the Miles-like traversal is recognized");
	Check(patch.sites.size() == 1, "one site");
	if (patch.sites.size() != 1) {
		std::fprintf(stderr, "report:%s\n", patch.report.c_str());
		return;
	}
	const auto& site = patch.sites[0];
	Check(patch.user_sgpr == 12 && patch.vsharp_offset == 544 && patch.header_offset == 80 && patch.root == 37,
	      "TLAS source: V# at user s[12:13] + 544, header at its base + 80, root 37");
	Check(site.loop_pc == 6 * 4 && site.entry_pc == 5 * 4, "loop header and entry");
	Check(site.node_sgpr == 37 && site.header_sgpr == 76, "node and header SGPRs");
	Check(site.t_max == 21 && site.origin[0] == 60 && site.origin[1] == 62 && site.origin[2] == 61 &&
	          site.direction[0] == 81 && site.direction[1] == 88 && site.direction[2] == 9,
	      "ray registers from the TLAS test's NSA addresses");
	Check(site.leaf_node == 20 && site.leaf_word == 2, "BLAS node VGPR and leaf entry info word");
	Check(site.result_t == 21 && site.result_u == 14 && site.result_v == 15 && site.result_word == 54 &&
	          site.result_node == 55 && site.result_instance == 7,
	      "accepted hit registers");
	Check(site.alpha_counter == 73, "alpha test counter");
	Check(site.stack_sgpr == 80 && site.stack_value == 0, "stack pointer and its entry value");
	Check(site.cull_exit_pc == At(program, site.cull_exit_pc)->pc &&
	          At(program, site.cull_exit_pc)->opcode == D::Opcode::S_MOV_B64,
	      "cull exit at the instance skip");
	Check(At(program, site.exit_pc) != nullptr && At(program, site.exit_pc)->opcode == D::Opcode::V_CMP_NE_U32,
	      "exit after the loop");

	// A traversal without the leaf's accepted hit is not a site.
	auto broken = program;
	for (auto& inst: broken.instructions) {
		if (inst.opcode == D::Opcode::V_CMPX_NE_U32) {
			inst.opcode = D::Opcode::V_CMP_NE_U32;
		}
	}
	D::RayTracingPatch none;
	Check(!D::DeriveRayTracingPatch(broken, none) && none.unmatched == 1, "a loop without an accepted hit is rejected");
}

void TestRewrite(bool verify) {
	auto            program = MilesLikeProgram();
	const auto      before  = program.instructions.size();
	D::RayTracingPatch patch;
	if (!D::DeriveRayTracingPatch(program, patch)) {
		Check(false, "derive for the rewrite");
		return;
	}
	Check(D::ApplyRayTracingPatch(patch, program, verify), "the patch applies");
	Check(BranchesValid(program), "pcs are consecutive and every branch target exists");
	Check(program.ray_tracing_source.valid && program.ray_tracing_source.user_sgpr == 12 &&
	          program.ray_tracing_source.vsharp_offset == 544 && program.ray_tracing_source.header_offset == 80,
	      "the TLAS source is recorded");
	Check(CountOpcode(program, D::Opcode::RT_INIT) == 1 && CountOpcode(program, D::Opcode::RT_SELECT) == 1 &&
	          CountOpcode(program, D::Opcode::RT_LEAF) == 1 && CountOpcode(program, D::Opcode::RT_RESOLVE) == 2 &&
	          CountOpcode(program, D::Opcode::RT_EXIT) == 1,
	      "the pseudo instructions of one site");
	// The guest loop stays (fallback), the BLAS test gets one copy, the setup and leaf ranges one each.
	Check(CountOpcode(program, D::Opcode::IMAGE_BVH_INTERSECT_RAY) == 3, "two guest BVH tests plus the leaf copy");
	Check(program.instructions.size() > before, "native code inserted");
	// Every moved guest instruction keeps its guest pc.
	Check(std::all_of(program.instructions.begin(), program.instructions.end(),
	                  [](const D::Instruction& inst) {
		                  return inst.family == D::Family::Unknown || inst.guest_pc != UINT32_MAX ||
		                         (inst.family == D::Family::SOPP && inst.guest_pc == UINT32_MAX);
	                  }),
	      "guest instructions carry their guest pc");
	if (verify) {
		Check(CountOpcode(program, D::Opcode::RT_VERIFY_SAVE) == 2 && CountOpcode(program, D::Opcode::RT_VERIFY_CHECK) == 1,
		      "verify: snapshot, save and check");
		// The guest loop's exit branch goes to the check, which precedes the guest exit instruction.
		const auto check = std::find_if(program.instructions.begin(), program.instructions.end(),
		                                [](const D::Instruction& inst) { return inst.opcode == D::Opcode::RT_VERIFY_CHECK; });
		Check(check != program.instructions.end() && (check + 1)->opcode == D::Opcode::V_CMP_NE_U32,
		      "verify: the check is right before the exit");
		bool redirected = false;
		for (const auto& inst: program.instructions) {
			redirected |= inst.opcode == D::Opcode::S_CBRANCH_SCC0 && check != program.instructions.end() &&
			              inst.branch_target == check->pc;
		}
		Check(redirected, "verify: the stack pop's exit branch reaches the check");
	} else {
		Check(CountOpcode(program, D::Opcode::RT_VERIFY_SAVE) == 0, "no verify code without verify");
	}
}

} // namespace

int main() {
	TestFinder();
	TestRewrite(false);
	TestRewrite(true);
	if (g_failures != 0) {
		std::fprintf(stderr, "RtPatchTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("RtPatchTests: ok\n");
	return 0;
}
