#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"

#include "graphics/shader/recompiler/CodegenOptions.h"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

void RemoveIdentities(const BlockList& blocks) {
	if (GetCodegenOptions().ir_linear_uses) {
		// KYTY_IR_LINEAR_USES (Senaxx 5145dc1f9): each identity's uses move to its argument in
		// program order, as below; its own entry in the argument's use list is dropped for all
		// identities at once at the end. Removing it one by one searched and shifted lists that
		// grow with every identity folded into the same value.
		std::unordered_set<const Inst*> removed;
		std::vector<Inst*>              touched;
		for (auto* block: blocks) {
			auto& instructions = block->Instructions();
			for (auto inst = instructions.begin(); inst != instructions.end();) {
				if (inst->GetOpcode() != ValueOpcode::Identity) {
					inst++;
					continue;
				}
				const auto replacement = inst->Arg(0);
				inst->ReplaceUsesForRemoval(replacement, removed, touched);
				removed.insert(&*inst);
				inst = instructions.erase(inst);
			}
		}
		Inst::DropRemovedUses(touched, removed);
		return;
	}
	for (auto* block: blocks) {
		auto& instructions = block->Instructions();
		for (auto inst = instructions.begin(); inst != instructions.end();) {
			if (inst->GetOpcode() != ValueOpcode::Identity) {
				inst++;
				continue;
			}
			const auto replacement = inst->Arg(0);
			inst->ReplaceUsesWith(replacement, false);
			inst = instructions.erase(inst);
		}
	}
}

static bool RemoveDeadPhiWebs(const BlockList& blocks, const std::unordered_set<const Inst*>* keep) {
	std::vector<Inst*> dead;
	std::vector<Inst*> web;
	std::vector<Inst*> stack;
	for (auto* block: blocks) {
		for (auto& inst: block->Instructions()) {
			if (inst.GetOpcode() != ValueOpcode::Phi ||
			    std::ranges::find(dead, &inst) != dead.end()) {
				continue;
			}
			web.clear();
			stack.clear();
			stack.push_back(&inst);
			bool external = false;
			while (!stack.empty() && !external) {
				auto* current = stack.back();
				stack.pop_back();
				if (std::ranges::find(web, current) != web.end()) {
					continue;
				}
				web.push_back(current);
				if (keep != nullptr && keep->contains(current)) {
					external = true;
					break;
				}
				for (const auto& use: current->Uses()) {
					if (use.user->GetOpcode() != ValueOpcode::Phi) {
						external = true;
						break;
					}
					stack.push_back(use.user);
				}
			}
			if (external) {
				continue;
			}
			dead.insert(dead.end(), web.begin(), web.end());
		}
	}
	if (dead.empty()) {
		return false;
	}
	for (auto* inst: dead) {
		inst->Invalidate();
	}
	for (auto* block: blocks) {
		auto& instructions = block->Instructions();
		for (auto inst = instructions.begin(); inst != instructions.end();) {
			if (std::ranges::find(dead, &*inst) != dead.end()) {
				inst = instructions.erase(inst);
			} else {
				inst++;
			}
		}
	}
	return true;
}

static void EliminateDeadCodeImpl(const BlockList& blocks, const std::unordered_set<const Inst*>* keep) {
	do {
		bool changed;
		do {
			changed = false;
			for (auto block = blocks.rbegin(); block != blocks.rend(); block++) {
				auto& instructions = (*block)->Instructions();
				auto  inst         = instructions.end();
				while (inst != instructions.begin()) {
					--inst;
					if (inst->HasUses() || inst->MayHaveSideEffects() ||
					    (keep != nullptr && keep->contains(&*inst))) {
						continue;
					}
					inst->Invalidate();
					inst    = instructions.erase(inst);
					changed = true;
				}
			}
		} while (changed);
	} while (RemoveDeadPhiWebs(blocks, keep));
}

void EliminateDeadCode(const BlockList& blocks) {
	EliminateDeadCodeImpl(blocks, nullptr);
}

void EliminateDeadCode(const BlockList& blocks, const std::unordered_set<const Inst*>& keep) {
	EliminateDeadCodeImpl(blocks, keep.empty() ? nullptr : &keep);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
