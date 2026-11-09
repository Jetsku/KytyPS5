#pragma once

#include "graphics/shader/recompiler/ir/Block.h"

#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

void RemoveIdentities(const BlockList& blocks);
void EliminateDeadCode(const BlockList& blocks);
// The same, but the instructions in `keep` (and so everything they use) stay: values that tables
// outside the blocks name, such as the resource plan's (PlanTableInstructions, ShaderIR.h).
void EliminateDeadCode(const BlockList& blocks, const std::unordered_set<const Inst*>& keep);

} // namespace Libs::Graphics::ShaderRecompiler::IR
