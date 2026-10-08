#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

// Deep copy of a translated program for the pipeline cache's translation reuse
// (PipelineCache::ProgramCache, KYTY_TRANSLATION_CACHE): a program source is translated once and
// every further permutation specializes a copy, instead of decoding, structurizing and
// translating the guest code again.
//
// The copy must be indistinguishable from a fresh TranslateProgram result to everything that runs
// afterwards (CompileProgram: specialization, identity removal, dead-code elimination, info
// collection, binding layout, write ranges, SPIR-V emission). It is therefore member-by-member:
// the same blocks in the same storage and list order, the same instructions in the same order
// with the same opcode, flags, arguments, phi blocks, use lists (in their exact order) and
// evaluation indices, and every Value, Inst* and Block* remapped into the copy.
//
// Not copied: each block's SSA scratch arrays (ssa_*_values, ssa_sreg_mask_tags). Only
// RewriteToSsa reads them, and it runs inside TranslateProgram; afterwards they may still name
// instructions that dead-code elimination has since erased, so they are not remappable either.
//
// A program that references an instruction or block it does not own is not copied (false); the
// cache then translates again. The layout checks below break the build when a member is added to
// one of the copied types, so this function cannot silently fall behind.

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

#if defined(_MSC_VER) && defined(_WIN64) && defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL == 0
// Update CloneProgram, then these sizes, when one of these types changes.
// Members: Inst 7 (opcode, flags, parent, args, phi_blocks, uses, evaluation_index); Block 4 plus
// the SSA scratch arrays; ResourcePlan 28; Program 14 beyond ResourcePlan.
static_assert(sizeof(Inst) == 104, "IR::Inst changed: update CloneProgram");
static_assert(sizeof(Block) == 9256, "IR::Block changed: update CloneProgram");
static_assert(sizeof(ResourcePlan) == 704, "IR::ResourcePlan changed: update CloneProgram");
static_assert(sizeof(Program) == 1008, "IR::Program changed: update CloneProgram");
static_assert(sizeof(BlockInfo) == 192, "IR::BlockInfo changed: update CloneProgram");
static_assert(sizeof(DescriptorSource) == 240, "IR::DescriptorSource changed: update CloneProgram");
static_assert(sizeof(ResourceBlock) == 64, "IR::ResourceBlock changed: update CloneProgram");
static_assert(sizeof(SrtRead) == 24, "IR::SrtRead changed: update CloneProgram");
static_assert(sizeof(ResourcePlan::EvaluationRecipe) == 120,
              "IR::ResourcePlan::EvaluationRecipe changed: update CloneProgram");
static_assert(sizeof(ResourcePlan::ArithmeticTapeInstruction) == 80,
              "IR::ResourcePlan::ArithmeticTapeInstruction changed: update CloneProgram");
static_assert(sizeof(UniformFillPlan) == 96, "IR::UniformFillPlan changed: update CloneProgram");
#endif

class Remapper {
public:
	void Own(const Inst* source, Inst* target) { m_insts.emplace(source, target); }
	void Own(const Block* source, Block* target) { m_blocks.emplace(source, target); }

	[[nodiscard]] bool Owns(const Inst* inst) const {
		return inst == nullptr || m_insts.contains(inst);
	}
	[[nodiscard]] bool Owns(const Block* block) const {
		return block == nullptr || m_blocks.contains(block);
	}
	[[nodiscard]] bool Owns(Value value) const { return Owns(value.TryInstruction()); }

	[[nodiscard]] Inst* Map(const Inst* inst) const {
		return inst == nullptr ? nullptr : m_insts.at(inst);
	}
	[[nodiscard]] Block* Map(const Block* block) const {
		return block == nullptr ? nullptr : m_blocks.at(block);
	}
	[[nodiscard]] Value Map(Value value) const {
		// Every instruction value is made by Value(Inst*) (type Opaque); other kinds are data.
		const auto* inst = value.TryInstruction();
		return inst == nullptr ? value : Value(Map(inst));
	}

private:
	std::unordered_map<const Inst*, Inst*>   m_insts;
	std::unordered_map<const Block*, Block*> m_blocks;
};

} // namespace

bool CloneProgram(const Program& source, Program& target) {
	target = Program {};
	Remapper map;
	// Blocks, then every instruction (fresh, unlinked), so that all owned addresses are known
	// before any reference is checked or copied.
	std::vector<std::unique_ptr<Block>> blocks;
	blocks.reserve(source.block_storage.size());
	for (const auto& block: source.block_storage) {
		if (block == nullptr) return false;
		blocks.push_back(std::make_unique<Block>());
		map.Own(block.get(), blocks.back().get());
	}
	for (size_t index = 0; index < source.block_storage.size(); ++index) {
		auto& instructions = blocks[index]->instructions;
		for (const auto& inst: source.block_storage[index]->instructions) {
			map.Own(&inst, &instructions.emplace_back(inst.opcode, inst.flags));
		}
	}
	std::list<Inst> value_storage;
	for (const auto& inst: source.value_storage) {
		map.Own(&inst, &value_storage.emplace_back(inst.opcode, inst.flags));
	}

	// Pass 1: every instruction and block the source references must be one it owns. Nothing
	// references the fresh instructions yet, so they destroy cleanly on failure.
	const auto inst_owned = [&](const Inst& inst) {
		if (!map.Owns(inst.parent)) return false;
		for (const auto& arg: inst.args) {
			if (!map.Owns(arg)) return false;
		}
		for (const auto* block: inst.phi_blocks) {
			if (!map.Owns(block)) return false;
		}
		for (const auto& use: inst.uses) {
			if (use.user == nullptr || !map.Owns(use.user)) return false;
		}
		return true;
	};
	const auto references_owned = [&] {
		for (const auto& block: source.block_storage) {
			for (const auto& inst: block->instructions) {
				if (!inst_owned(inst)) return false;
			}
			for (const auto* other: block->predecessors) {
				if (other == nullptr || !map.Owns(other)) return false;
			}
			for (const auto* other: block->successors) {
				if (other == nullptr || !map.Owns(other)) return false;
			}
		}
		for (const auto& inst: source.value_storage) {
			if (!inst_owned(inst)) return false;
		}
		for (const auto* block: source.blocks) {
			if (block == nullptr || !map.Owns(block)) return false;
		}
		for (const auto& descriptor: source.descriptor_sources) {
			for (const auto& value: descriptor.dwords) {
				if (!map.Owns(value)) return false;
			}
			if (descriptor.indirect_descriptor &&
			    (!map.Owns(descriptor.indirect_descriptor->key_count) ||
			     !map.Owns(descriptor.indirect_descriptor->selector_first) ||
			     !map.Owns(descriptor.indirect_descriptor->selector_mask))) {
				return false;
			}
		}
		for (const auto& block: source.control_flow) {
			if (!map.Owns(block.condition)) return false;
		}
		for (const auto& read: source.srt_reads) {
			if (!map.Owns(read.value)) return false;
		}
		for (const auto& recipe: source.evaluation_recipes) {
			if (!map.Owns(recipe.instruction) || !map.Owns(recipe.selection_mask)) return false;
		}
		for (const auto& instruction: source.arithmetic_tape_instructions) {
			if (!map.Owns(instruction.instruction)) return false;
		}
		for (const auto& value: source.uniform_fill.values) {
			if (!map.Owns(value)) return false;
		}
		for (const auto& info: source.block_info) {
			if (!map.Owns(info.condition) || !map.Owns(info.indirect_target)) return false;
		}
		for (const auto& value: source.dynamic_reads) {
			if (!map.Owns(value)) return false;
		}
		return true;
	};
	if (!references_owned()) {
		return false;
	}

	// Pass 2: every member. Inst(opcode, flags) sized `args` for fixed-arity opcodes; all members
	// are overwritten. Use lists are copied as they are (not rebuilt through SetArg), so their
	// order matches the source.
	const auto copy_inst = [&](const Inst& from, Inst& to) {
		to.parent = map.Map(from.parent);
		to.args.resize(from.args.size());
		for (size_t i = 0; i < from.args.size(); ++i) {
			to.args[i] = map.Map(from.args[i]);
		}
		to.phi_blocks.resize(from.phi_blocks.size());
		for (size_t i = 0; i < from.phi_blocks.size(); ++i) {
			to.phi_blocks[i] = map.Map(from.phi_blocks[i]);
		}
		to.uses.resize(from.uses.size());
		for (size_t i = 0; i < from.uses.size(); ++i) {
			to.uses[i] = {map.Map(from.uses[i].user), from.uses[i].operand};
		}
		to.evaluation_index = from.evaluation_index;
	};
	for (size_t index = 0; index < source.block_storage.size(); ++index) {
		const auto& from = *source.block_storage[index];
		auto&       to   = *blocks[index];
		auto        copy = to.instructions.begin();
		for (const auto& inst: from.instructions) {
			copy_inst(inst, *copy++);
		}
		to.predecessors.reserve(from.predecessors.size());
		for (const auto* other: from.predecessors) {
			to.predecessors.push_back(map.Map(other));
		}
		to.successors.reserve(from.successors.size());
		for (const auto* other: from.successors) {
			to.successors.push_back(map.Map(other));
		}
		to.ssa_sealed = from.ssa_sealed;
	}
	{
		auto copy = value_storage.begin();
		for (const auto& inst: source.value_storage) {
			copy_inst(inst, *copy++);
		}
	}

	// ResourcePlan members.
	target.stage           = source.stage;
	target.shader_hash     = source.shader_hash;
	target.user_data_base  = source.user_data_base;
	target.user_data_count = source.user_data_count;
	target.value_storage   = std::move(value_storage); // list nodes keep their addresses
	target.memory_info     = source.memory_info;
	target.descriptor_sources.reserve(source.descriptor_sources.size());
	for (const auto& descriptor: source.descriptor_sources) {
		auto& copy = target.descriptor_sources.emplace_back(descriptor);
		for (auto& value: copy.dwords) {
			value = map.Map(value);
		}
		if (copy.indirect_descriptor) {
			copy.indirect_descriptor->key_count     = map.Map(copy.indirect_descriptor->key_count);
			copy.indirect_descriptor->selector_first = map.Map(copy.indirect_descriptor->selector_first);
			copy.indirect_descriptor->selector_mask = map.Map(copy.indirect_descriptor->selector_mask);
		}
	}
	target.control_flow.reserve(source.control_flow.size());
	for (const auto& block: source.control_flow) {
		auto& copy     = target.control_flow.emplace_back(block);
		copy.condition = map.Map(block.condition);
	}
	target.srt_reads.reserve(source.srt_reads.size());
	for (const auto& read: source.srt_reads) {
		auto& copy = target.srt_reads.emplace_back(read);
		copy.value = map.Map(read.value);
	}
	target.srt_read_run_ends = source.srt_read_run_ends;
	target.evaluation_recipes.reserve(source.evaluation_recipes.size());
	for (const auto& recipe: source.evaluation_recipes) {
		auto& copy          = target.evaluation_recipes.emplace_back(recipe);
		copy.instruction    = map.Map(recipe.instruction);
		copy.selection_mask = map.Map(recipe.selection_mask);
	}
	target.descriptor_roots       = source.descriptor_roots;
	target.flat_read_roots        = source.flat_read_roots;
	target.condition_roots        = source.condition_roots;
	target.initial_active_sources = source.initial_active_sources;
	target.flow_aliases           = source.flow_aliases;
	target.flow_initial_sources   = source.flow_initial_sources;
	target.arithmetic_tapes       = source.arithmetic_tapes;
	target.arithmetic_tape_instructions.reserve(source.arithmetic_tape_instructions.size());
	for (const auto& instruction: source.arithmetic_tape_instructions) {
		auto& copy       = target.arithmetic_tape_instructions.emplace_back(instruction);
		copy.instruction = map.Map(instruction.instruction);
	}
	target.clean_flat_slots               = source.clean_flat_slots;
	target.requires_specialization_memory = source.requires_specialization_memory;
	target.has_address_writes             = source.has_address_writes;
	target.srt_plan_complete              = source.srt_plan_complete;
	target.resource_tracking_complete     = source.resource_tracking_complete;
	target.info                           = source.info;
	target.uniform_fill.fill              = source.uniform_fill.fill;
	for (size_t i = 0; i < source.uniform_fill.values.size(); ++i) {
		target.uniform_fill.values[i] = map.Map(source.uniform_fill.values[i]);
	}
	target.evaluation_value_count = source.evaluation_value_count;
	target.evaluation_sealed      = source.evaluation_sealed;

	// Program members.
	target.block_storage = std::move(blocks);
	target.blocks.reserve(source.blocks.size());
	for (const auto* block: source.blocks) {
		target.blocks.push_back(map.Map(block));
	}
	target.wave_size           = source.wave_size;
	target.scratch_dwords      = source.scratch_dwords;
	target.dispatcher_fallback = source.dispatcher_fallback;
	target.cfg_failure_kind    = source.cfg_failure_kind;
	target.fallback_reason     = source.fallback_reason;
	target.block_info.reserve(source.block_info.size());
	for (const auto& info: source.block_info) {
		auto& copy           = target.block_info.emplace_back(info);
		copy.condition       = map.Map(info.condition);
		copy.indirect_target = map.Map(info.indirect_target);
	}
	target.export_info = source.export_info;
	target.dynamic_reads.reserve(source.dynamic_reads.size());
	for (const auto& value: source.dynamic_reads) {
		target.dynamic_reads.push_back(map.Map(value));
	}
	target.shader_info_complete    = source.shader_info_complete;
	target.bindings                = source.bindings;
	target.binding_layout_complete = source.binding_layout_complete;
	target.write_ranges            = source.write_ranges;
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
