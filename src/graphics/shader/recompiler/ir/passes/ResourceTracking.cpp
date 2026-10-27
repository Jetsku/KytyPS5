#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

static std::atomic_bool g_bindless_images {false};

void SetBindlessImagesEnabled(bool enabled) {
	g_bindless_images.store(enabled, std::memory_order_release);
}

bool BindlessImagesEnabled() {
	return g_bindless_images.load(std::memory_order_acquire);
}

namespace {

constexpr uint32_t SamplerBorderClampMask    = (1u << 2u) | (1u << 5u) | (1u << 8u);
constexpr uint32_t SamplerDword3ReservedMask = 0x3ffff000u;

uint32_t PossibleU32Bits(Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U32 ? value.U32() : UINT32_MAX;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return UINT32_MAX;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::BitwiseAnd32:
			return PossibleU32Bits(inst->Arg(0)) & PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::BitwiseOr32:
			return PossibleU32Bits(inst->Arg(0)) | PossibleU32Bits(inst->Arg(1));
		case ValueOpcode::ShiftLeftLogical32: {
			const auto shift = inst->Arg(1).Resolve();
			return shift.IsImmediate() && shift.GetType() == Type::U32
			           ? PossibleU32Bits(inst->Arg(0)) << (shift.U32() & 31u)
			           : UINT32_MAX;
		}
		default: return UINT32_MAX;
	}
}

// Whether `active` being true proves `condition` is true (same value, or an AND of such parts).
bool MaskImplies(Value active, Value condition, uint32_t depth = 0) {
	active    = active.Resolve();
	condition = condition.Resolve();
	if (active == condition) return true;
	if (depth > 6) return false;
	const auto* wanted = condition.TryInstruction();
	if (wanted != nullptr && wanted->GetOpcode() == ValueOpcode::LogicalAnd) {
		return MaskImplies(active, wanted->Arg(0), depth + 1) &&
		       MaskImplies(active, wanted->Arg(1), depth + 1);
	}
	const auto* have = active.TryInstruction();
	if (have != nullptr && have->GetOpcode() == ValueOpcode::LogicalAnd) {
		return MaskImplies(have->Arg(0), condition, depth + 1) ||
		       MaskImplies(have->Arg(1), condition, depth + 1);
	}
	return false;
}

// Whether the high dword of a FLAT address is provably outside the canonical user address range
// (a PS5 shared/private aperture tag), so that access can only be local stack memory.
bool IsNoncanonicalFlatAddress(Value high, Value active) {
	high   = high.Resolve();
	active = active.Resolve();
	// The aperture-routed access adds the carry of its low-dword offset to the high dword.
	if (const auto* add = high.TryInstruction();
	    add != nullptr && add->GetOpcode() == ValueOpcode::IAdd32) {
		const auto is_carry = [](Value value) {
			const auto* select = value.Resolve().TryInstruction();
			return select != nullptr && select->GetOpcode() == ValueOpcode::SelectU32 &&
			       select->Arg(1).Resolve().IsImmediate() && select->Arg(2).Resolve().IsImmediate() &&
			       select->Arg(1).Resolve().GetType() == Type::U32 &&
			       select->Arg(2).Resolve().GetType() == Type::U32 &&
			       select->Arg(1).Resolve().U32() <= 1u && select->Arg(2).Resolve().U32() <= 1u;
		};
		if (is_carry(add->Arg(1))) {
			high = add->Arg(0).Resolve();
		} else if (is_carry(add->Arg(0))) {
			high = add->Arg(1).Resolve();
		}
	}
	// A predicated VGPR write supplies its new value on exactly the lanes which execute this
	// access. Do not discard a selection under a different mask.
	while (const auto* select = high.TryInstruction()) {
		if (select->GetOpcode() != ValueOpcode::SelectU32) {
			break;
		}
		// The select must be this access's own predicated write. Instrumented copies of the mask
		// (the loop guard duplicates it) are not the same value, so a select whose other side is the
		// register's previous value (a phi) is accepted too.
		const auto* other = select->Arg(2).Resolve().TryInstruction();
		if (!MaskImplies(active, select->Arg(0)) &&
		    (other == nullptr || other->GetOpcode() != ValueOpcode::Phi)) {
			break;
		}
		high = select->Arg(1).Resolve();
	}
	uint32_t lower = 0;
	uint32_t upper = UINT32_MAX;
	if (high.IsImmediate() && high.GetType() == Type::U32) {
		lower = upper = high.U32();
	} else if (const auto* inst = high.TryInstruction();
	           inst != nullptr && inst->GetOpcode() == ValueOpcode::UMedTri32) {
		// The median lies between any two constant inputs, regardless of the third.
		uint32_t count = 0;
		for (size_t index = 0; index < inst->NumArgs(); ++index) {
			const auto value = inst->Arg(index).Resolve();
			if (!value.IsImmediate() || value.GetType() != Type::U32) continue;
			if (count++ == 0) {
				lower = upper = value.U32();
			} else {
				lower = std::min(lower, value.U32());
				upper = std::max(upper, value.U32());
			}
		}
		if (count < 2) return false;
	}
	// Leave one high DWORD on each side for the signed instruction offset/carry.
	return lower > 0x00008000u && upper < 0xffff7fffu;
}

Value CanonicalizeSampleAdjustDword3(Value value) {
	for (;;) {
		value            = value.Resolve();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::BitwiseOr32) {
			if ((PossibleU32Bits(value) & ~SamplerDword3ReservedMask) == 0)
				return Value(0u);
			return value;
		}
		const auto left           = inst->Arg(0).Resolve();
		const auto right          = inst->Arg(1).Resolve();
		const bool left_reserved  = (PossibleU32Bits(left) & ~SamplerDword3ReservedMask) == 0;
		const bool right_reserved = (PossibleU32Bits(right) & ~SamplerDword3ReservedMask) == 0;
		if (left_reserved && right_reserved) {
			return Value(0u);
		}
		if (left_reserved) {
			value = right;
		} else if (right_reserved) {
			value = left;
		} else {
			return value;
		}
	}
}

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

uint32_t ByteExtent(const MemoryInfo& memory) {
	const auto bytes = std::max((memory.data_bits + 7u) / 8u, 1u);
	const auto count = std::max(memory.data_dwords, 1u);
	const auto end   = static_cast<uint64_t>(memory.offset) + static_cast<uint64_t>(bytes) * count;
	return end > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(end);
}

class Tracker {
public:
	Tracker(Program& program, bool indirect_scalar_buffers, bool bda_writes)
	    : m_program(program), m_info(program.info),
	      m_indirect_scalar_buffers(indirect_scalar_buffers), m_bda_writes(bda_writes) {
		m_info.buffers.clear();
		m_info.images.clear();
		m_info.samplers.clear();
		m_info.sampled_pairs.clear();
		m_info.uses_dma   = false;
		m_info.bda_writes = false;
		m_shader_writes   = HasShaderMemoryWrites(program);
	}

	void Run() {
		if (m_program.resource_tracking_complete) {
			Fail(0, "resources already tracked");
		}
		if (!m_program.srt_plan_complete) {
			Fail(0, "SRT plan is not ready");
		}
		PlanIndirectImages();
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				Collect(inst);
			}
		}
		if (!m_dead_stores.empty()) {
			// Global legs of aperture-routed stores that can never run: erase them.
			for (auto* block: m_program.blocks) {
				auto& instructions = block->Instructions();
				for (auto inst = instructions.begin(); inst != instructions.end();) {
					if (std::ranges::find(m_dead_stores, &*inst) != m_dead_stores.end()) {
						inst->Invalidate();
						inst = instructions.erase(inst);
					} else {
						++inst;
					}
				}
			}
		}
		LinkImageAliases();
		for (const auto& patch: m_handle_patches) {
			patch.handle->SetFlags<uint32_t>(patch.resource);
		}
		for (const auto& patch: m_memory_patches) {
			auto& memory    = m_program.memory_info[patch.index];
			memory.resource = patch.resource;
			if (patch.has_sampler) {
				memory.sampler = patch.sampler;
			}
		}
		for (const auto& plan: m_indirect_images) {
			plan.handle->SetArg(0, plan.key);
			for (uint32_t dword = 0; dword < 4u; dword++) {
				plan.handle->SetArg(dword + 1u, plan.roots[dword + 4u]);
			}
			for (uint32_t dword = 5u; dword < plan.roots.size(); dword++) {
				plan.handle->SetArg(dword, plan.key);
			}
			for (const auto index: plan.memory) {
				if (index != UINT32_MAX) {
					m_program.memory_info[index].planning_only = true;
				}
			}
		}
		std::erase_if(m_program.dynamic_reads, [&](Value value) {
			const auto* inst = value.Resolve().TryInstruction();
			return std::any_of(m_indirect_images.begin(), m_indirect_images.end(),
			                   [&](const IndirectImagePlan& plan) {
				return std::ranges::find(plan.reads, inst) != plan.reads.end();
			});
		});
		m_program.descriptor_sources         = std::move(m_sources);
		m_program.info                       = std::move(m_info);
		m_program.resource_tracking_complete = true;
	}

private:
	std::vector<Inst*> m_dead_stores;

	struct HandlePatch {
		Inst*    handle   = nullptr;
		uint32_t resource = 0;
	};

	struct MemoryPatch {
		uint32_t index       = 0;
		uint32_t resource    = 0;
		uint32_t sampler     = 0;
		bool     has_sampler = false;
	};

	struct IndirectImagePlan {
		Inst*                      handle = nullptr;
		uint32_t                   source = 0;
		Value                      key;
		std::array<Value, 8>       roots {};
		std::array<uint32_t, 8>    memory {};
		std::array<const Inst*, 8> reads {};
	};

	[[noreturn]] void Fail(uint32_t pc, const std::string& reason) const {
		const auto message =
		    fmt::format("shader resource tracking: hash=0x{:016x} stage={} pc=0x{:08x} {}",
		                m_program.shader_hash, StageName(m_program.stage), pc, reason);
		EXIT("%s", message.c_str());
		std::abort();
	}

	Value LowerDescriptorPhi(Value value, const Block* use) {
		value           = value.Resolve();
		const auto* phi = value.TryInstruction();
		if (m_shader_writes || phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u || phi->GetType() != Type::U32 ||
		    m_program.blocks.size() != m_program.block_info.size()) {
			return value;
		}
		const auto* merge = phi->Parent();
		const auto* branch = phi->PhiBlock(0);
		if (merge == nullptr || branch == nullptr || phi->PhiBlock(1) == nullptr ||
		    branch == phi->PhiBlock(1)) {
			return value;
		}
		// Structurization can merge the descriptor and its use predicate in parallel Phis.
		// Match their incoming blocks to exclude only edges that cannot reach this use.
		if (use != nullptr && use->ImmPredecessors().size() == 1u &&
		    use->ImmPredecessors()[0] == merge) {
			const auto merge_it = BlockIter(merge);
			const auto use_it   = BlockIter(use);
			if (merge_it != m_program.blocks.end() && use_it != m_program.blocks.end()) {
				const auto& info = m_program.block_info[merge_it - m_program.blocks.begin()];
				const auto& term = info.terminator;
				const auto id    = m_program.block_info[use_it - m_program.blocks.begin()].id;
				const auto* condition = info.condition.Resolve().TryInstruction();
				if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
				    term.true_block != term.false_block &&
				    (id == term.true_block || id == term.false_block) && condition != nullptr &&
				    condition->GetOpcode() == ValueOpcode::Phi && condition->GetType() == Type::U1 &&
				    condition->Parent() == merge && condition->NumArgs() == 2u &&
				    condition->NumPhiBlocks() == 2u) {
					const bool taken = id == term.true_block;
					for (uint32_t skipped = 0; skipped < 2u; skipped++) {
						const auto excluded = condition->Arg(skipped).Resolve();
						const auto included = condition->Arg(skipped ^ 1u).Resolve();
						if (!excluded.IsImmediate() || excluded.GetType() != Type::U1 ||
						    excluded.U1() == taken ||
						    (included.IsImmediate() && included.U1() != taken)) {
							continue;
						}
						for (uint32_t selected = 0; selected < 2u; selected++) {
							if (phi->PhiBlock(selected) == condition->PhiBlock(skipped ^ 1u) &&
							    phi->PhiBlock(selected ^ 1u) == condition->PhiBlock(skipped)) {
								return phi->Arg(selected);
							}
						}
					}
				}
			}
		}
		for (const auto& [original, selected]: m_descriptor_selections) {
			if (original == phi) {
				return selected;
			}
		}
		if (branch->ImmSuccessors().size() != 2u) {
			if (branch->ImmPredecessors().size() != 1u) {
				return value;
			}
			branch = branch->ImmPredecessors()[0];
		}
		if (branch == merge || branch->ImmSuccessors().size() != 2u) {
			return value;
		}
		std::array<uint32_t, 2> target_ids;
		for (uint32_t arm = 0; arm < 2; arm++) {
			const auto* incoming = phi->PhiBlock(arm);
			if (incoming == merge ||
			    (incoming != branch &&
			     (incoming->ImmPredecessors().size() != 1u ||
			      incoming->ImmPredecessors()[0] != branch ||
			      incoming->ImmSuccessors().size() != 1u ||
			      incoming->ImmSuccessors()[0] != merge))) {
				return value;
			}
			const auto* target = incoming == branch ? merge : incoming;
			const auto  it     = BlockIter(target);
			if (it == m_program.blocks.end()) {
				return value;
			}
			target_ids[arm] = m_program.block_info[it - m_program.blocks.begin()].id;
		}
		const auto branch_it = BlockIter(branch);
		if (branch_it == m_program.blocks.end()) {
			return value;
		}
		const auto& info = m_program.block_info[branch_it - m_program.blocks.begin()];
		const auto& term = info.terminator;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    !((term.true_block == target_ids[0] && term.false_block == target_ids[1]) ||
		      (term.false_block == target_ids[0] && term.true_block == target_ids[1])) ||
		    !ValidateRuntimeValue(m_program, info.condition, RuntimeValueType::Integer) ||
		    !ValidateRuntimeValue(m_program, phi->Arg(0)) ||
		    !ValidateRuntimeValue(m_program, phi->Arg(1))) {
			return value;
		}
		// Retain a host expression; replacing the GPU Phi would break SSA dominance.
		const auto true_arg = term.true_block == target_ids[0] ? 0u : 1u;
		auto&      selected = m_program.value_storage.emplace_back(ValueOpcode::SelectU32);
		selected.SetArg(0, info.condition);
		selected.SetArg(1, phi->Arg(true_arg));
		selected.SetArg(2, phi->Arg(true_arg ^ 1u));
		m_descriptor_selections.emplace_back(phi, Value(&selected));
		return Value(&selected);
	}

	void MakeSource(const Inst& handle, uint32_t width, bool sampler, bool sample_adjust,
	                DescriptorSource& descriptor, uint32_t pc) {
		if (handle.NumArgs() != width) {
			Fail(pc, fmt::format("{} has {} descriptor dwords, expected {}",
			                     ValueOpcodeName(handle.GetOpcode()), handle.NumArgs(), width));
		}
		descriptor.dword_count = width;
		for (uint32_t i = 0; i < width; i++) {
			descriptor.dwords[i] = LowerDescriptorPhi(handle.Arg(i), handle.Parent());
		}
		if (sample_adjust) {
			descriptor.dwords[3] = CanonicalizeSampleAdjustDword3(descriptor.dwords[3]);
		}
		const auto dword0 = descriptor.dwords[0].Resolve();
		if (sampler && dword0.IsImmediate() && dword0.GetType() == Type::U32 &&
		    (dword0.U32() & SamplerBorderClampMask) == 0) {
			// Border color and its table index are unused unless a clamp axis selects border mode.
			descriptor.dwords[3] = Value(0u);
		}
	}

	bool ValidateSource(const DescriptorSource& descriptor, uint32_t& bad_dword) const {
		for (uint32_t i = 0; i < descriptor.dword_count; i++) {
			bad_dword = i;
			if (descriptor.dwords[i].Resolve().GetType() != Type::U32) {
				return false;
			}
			if (!ValidateRuntimeValue(m_program, descriptor.dwords[i])) {
				return false;
			}
		}
		return true;
	}

	uint32_t InternSource(const DescriptorSource& descriptor) {
		for (uint32_t candidate = 0; candidate < m_sources.size(); candidate++) {
			const auto& current = m_sources[candidate];
			if (current.dword_count != descriptor.dword_count ||
			    current.indirect_descriptor.has_value() != descriptor.indirect_descriptor.has_value()) {
				continue;
			}
			if (current.indirect_descriptor.has_value()) {
				const auto& a = *current.indirect_descriptor;
				const auto& b = *descriptor.indirect_descriptor;
				if (a.selector != b.selector || a.table_source != b.table_source ||
				    a.table_offset != b.table_offset || a.table_immediate != b.table_immediate ||
				    a.table_stride != b.table_stride || a.table_record_bytes != b.table_record_bytes ||
				    a.table_scalar != b.table_scalar || a.workgroup_axis != b.workgroup_axis ||
				    a.bindless != b.bindless || a.record_pointer != b.record_pointer ||
				    a.nested_vsharp != b.nested_vsharp || a.nested_heap != b.nested_heap ||
				    a.nested_keys != b.nested_keys ||
				    !EquivalentValue(m_program, a.key_count, b.key_count) ||
				    a.selector_first.IsEmpty() != b.selector_first.IsEmpty() ||
				    (!a.selector_first.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_first, b.selector_first)) ||
				    a.selector_mask.IsEmpty() != b.selector_mask.IsEmpty() ||
				    (!a.selector_mask.IsEmpty() &&
				     !EquivalentValue(m_program, a.selector_mask, b.selector_mask))) continue;
			}
			bool same = true;
			for (uint32_t i = 0; i < descriptor.dword_count; i++) {
				same = same && EquivalentValue(m_program, current.dwords[i], descriptor.dwords[i]);
			}
			if (same) {
				return candidate;
			}
		}
		m_sources.push_back(descriptor);
		return static_cast<uint32_t>(m_sources.size() - 1);
	}

	static bool ImmediateU32(Value value, uint32_t& result) {
		value = value.Resolve();
		if (!value.IsImmediate() || value.GetType() != Type::U32) {
			return false;
		}
		result = value.U32();
		return true;
	}

	static bool UsesOnly(const Inst& value, std::span<const Inst* const> users) {
		return !value.Uses().empty() && std::ranges::all_of(value.Uses(), [&](const Use& use) {
			return std::ranges::find(users, use.user) != users.end();
		});
	}

	const MemoryInfo* ScalarReadMemory(const Inst& read, uint32_t& index) const {
		const bool address = read.GetOpcode() == ValueOpcode::LoadAddressU32;
		if (!(address ? read.NumArgs() == 4u
		              : read.GetOpcode() == ValueOpcode::ReadConstBuffer && read.NumArgs() == 2u)) {
			return nullptr;
		}
		if (address) {
			const auto high = read.Arg(2).Resolve();
			const auto enabled = read.Arg(3).Resolve();
			if (!high.IsImmediate() || high.GetType() != Type::U32 || high.U32() != 0u ||
			    !enabled.IsImmediate() || enabled.GetType() != Type::U1 || !enabled.U1()) {
				return nullptr;
			}
		}
		index = read.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) {
			return nullptr;
		}
		const auto& memory = m_program.memory_info[index];
		return memory.kind == (address ? ResourceKind::ScalarAddress : ResourceKind::ScalarBuffer) &&
		               memory.data_bits == 32u && memory.data_dwords == 1u
		           ? &memory
		           : nullptr;
	}

	static bool IsMemoryOperation(const Inst& inst) {
		const auto op = inst.GetOpcode();
		return BufferAccessOf(op) != BufferAccess::None ||
		       AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		       ImageOpcodeInfoOf(op).access != ImageAccess::None;
	}

	// No memory operation but `owner` uses metadata entry `index`. The uses per entry are
	// counted once (the program's memory operations do not change while resources are planned).
	bool MemoryIndexBelongsTo(uint32_t index, const Inst& owner) const {
		if (!m_memory_users_built) {
			m_memory_users_built = true;
			for (const auto* block: m_program.blocks) {
				for (const auto& inst: *block) {
					if (IsMemoryOperation(inst)) {
						const auto user = inst.Flags<MemoryFlags>().index;
						if (user >= m_memory_users.size()) m_memory_users.resize(user + 1u, 0u);
						m_memory_users[user]++;
					}
				}
			}
		}
		const uint32_t users = index < m_memory_users.size() ? m_memory_users[index] : 0u;
		const uint32_t own   = IsMemoryOperation(owner) && owner.Flags<MemoryFlags>().index == index;
		return users == own;
	}

	bool MakeRuntimeTableSource(const Inst& handle, uint32_t pc,
	                            DescriptorSource& descriptor) {
		const auto width = handle.GetOpcode() == ValueOpcode::GetBufferResource ? 4u
		                 : handle.GetOpcode() == ValueOpcode::GetAddressResource ? 2u : 0u;
		if (width == 0u) {
			return false;
		}
		MakeSource(handle, width, false, false, descriptor, pc);
		uint32_t bad_dword = 0;
		return ValidateSource(descriptor, bad_dword);
	}

	bool MatchMaterialOffset(Value value, Value& selector, uint32_t& stride,
	                         uint32_t& offset) const {
		value           = value.Resolve();
		offset          = 0;
		auto* candidate = value.TryInstruction();
		if (candidate != nullptr && candidate->GetOpcode() == ValueOpcode::IAdd32 &&
		    candidate->NumArgs() == 2u) {
			uint32_t immediate = 0;
			if (ImmediateU32(candidate->Arg(0), immediate)) {
				value = candidate->Arg(1).Resolve();
			} else if (ImmediateU32(candidate->Arg(1), immediate)) {
				value = candidate->Arg(0).Resolve();
			} else {
				return false;
			}
			offset = immediate;
		}
		const auto* multiply = value.TryInstruction();
		if (multiply == nullptr || multiply->GetOpcode() != ValueOpcode::IMul32 ||
		    multiply->NumArgs() != 2u) {
			return false;
		}
		if (ImmediateU32(multiply->Arg(0), stride)) {
			selector = multiply->Arg(1).Resolve();
		} else if (ImmediateU32(multiply->Arg(1), stride)) {
			selector = multiply->Arg(0).Resolve();
		} else {
			return false;
		}
		const auto* selector_inst = selector.TryInstruction();
		return stride != 0u && selector_inst != nullptr &&
		       selector_inst->GetOpcode() == ValueOpcode::ReadFirstLane;
	}

	bool NonzeroOnEntry(Value value, const Block* block) const {
		if (m_program.blocks.size() != m_program.block_info.size()) {
			return false;
		}
		// Each unique predecessor must execute before this use. Stop at joins: an
		// unrelated comparison is not a bound on FindILsb's zero-input sentinel.
		for (size_t depth = 0; block != nullptr && depth < m_program.blocks.size(); ++depth) {
			if (block->ImmPredecessors().size() != 1u) {
				return false;
			}
			const auto* previous = block->ImmPredecessors()[0];
			const auto current_it = BlockIter(block);
			const auto previous_it = BlockIter(previous);
			if (current_it == m_program.blocks.end() || previous_it == m_program.blocks.end()) {
				return false;
			}
			const auto& info = m_program.block_info[previous_it - m_program.blocks.begin()];
			const auto id = m_program.block_info[current_it - m_program.blocks.begin()].id;
			const auto& term = info.terminator;
			if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
			    (term.true_block == id) != (term.false_block == id)) {
				bool positive = term.true_block == id;
				const auto* test = info.condition.Resolve().TryInstruction();
				while (test != nullptr && test->GetOpcode() == ValueOpcode::LogicalNot) {
					positive = !positive;
					test = test->Arg(0).Resolve().TryInstruction();
				}
				if (test != nullptr && test->NumArgs() == 2u &&
				    ((test->GetOpcode() == ValueOpcode::INotEqual32 && positive) ||
				     (test->GetOpcode() == ValueOpcode::IEqual32 && !positive))) {
					for (uint32_t arg = 0; arg < 2u; ++arg) {
						uint32_t immediate;
						if (ImmediateU32(test->Arg(arg), immediate) && immediate == 0u &&
						    EquivalentValue(m_program, test->Arg(arg ^ 1u), value)) {
							return true;
						}
					}
				}
			}
			block = previous;
		}
		return false;
	}

	// KYTY_HEAP_KEY_MASK=1 (CodegenOptions::heap_key_mask, default off): a heap offset
	// (key << 5) & mask, with the mask's low five bits clear, is the record of key & (mask >> 5)
	// (PPSA03671's material pixel shaders: a V_READFIRSTLANE key, S_LSHL_B32 5, S_AND_B32 0x01ffffe0,
	// S_BUFFER_LOAD_DWORDX8). Only for a key the host cannot evaluate (a lane value): such a program
	// was skipped before, and a key the SRT walk evaluates keeps its flat slot.
	// The masked form: `masked` is the S_AND whose result the offset adds to, `key_mask` the mask
	// to apply to the key (UINT32_MAX when the offset is a plain key << 5).
	bool MatchTableOffset(Value value, Value& key, uint32_t& offset, uint32_t* key_mask = nullptr,
	                      Inst** masked = nullptr) const {
		offset = 0;
		if (key_mask != nullptr) *key_mask = UINT32_MAX;
		if (masked != nullptr) *masked = nullptr;
		for (;;) {
			auto* inst = value.Resolve().TryInstruction();
			if (inst == nullptr || inst->NumArgs() != 2u) {
				return false;
			}
			uint32_t immediate;
			if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    ImmediateU32(inst->Arg(1), immediate) && immediate == 5u) {
				key = inst->Arg(0).Resolve();
				return key.GetType() == Type::U32;
			}
			if (inst->GetOpcode() == ValueOpcode::BitwiseAnd32 && key_mask != nullptr &&
			    masked != nullptr && GetCodegenOptions().heap_key_mask) {
				uint32_t mask = 0;
				const auto operand = ImmediateU32(inst->Arg(1), mask) ? 0u
				                     : ImmediateU32(inst->Arg(0), mask) ? 1u
				                                                        : 2u;
				if (operand == 2u || mask == 0u || (mask & 31u) != 0u) {
					return false;
				}
				const auto* shift = inst->Arg(operand).Resolve().TryInstruction();
				if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
				    shift->NumArgs() != 2u || !ImmediateU32(shift->Arg(1), immediate) ||
				    immediate != 5u) {
					return false;
				}
				key = shift->Arg(0).Resolve();
				if (key.GetType() != Type::U32 || ValidateRuntimeValue(m_program, key)) {
					return false;
				}
				*key_mask = mask >> 5u;
				*masked   = inst;
				return true;
			}
			if (inst->GetOpcode() != ValueOpcode::IAdd32) {
				return false;
			}
			if (ImmediateU32(inst->Arg(0), immediate)) {
				value = inst->Arg(1);
			} else if (ImmediateU32(inst->Arg(1), immediate)) {
				value = inst->Arg(0);
			} else {
				return false;
			}
			// These additions are shader U32 arithmetic, before the scalar memory offset.
			offset += immediate;
		}
	}

	// Block lookups by pointer and by id, built on first use (the block list does not change
	// while resources are tracked). Big programs (an S_SWAPPC_B64 expansion of 7,000 blocks)
	// spent their tracking time in linear searches of the block list.
	void BuildBlockMaps() const {
		if (m_block_maps_built) {
			return;
		}
		m_block_maps_built = true;
		m_block_index.reserve(m_program.blocks.size());
		for (size_t index = 0; index < m_program.blocks.size(); ++index) {
			m_block_index.emplace(m_program.blocks[index], static_cast<uint32_t>(index));
		}
		m_block_by_id.reserve(m_program.block_info.size());
		for (size_t index = m_program.block_info.size(); index-- > 0;) {
			if (index < m_program.blocks.size()) {
				m_block_by_id[m_program.block_info[index].id] = m_program.blocks[index];
			}
		}
	}

	// std::ranges::find(m_program.blocks, block), in constant time.
	BlockList::iterator BlockIter(const Block* block) const {
		BuildBlockMaps();
		const auto found = m_block_index.find(block);
		return found == m_block_index.end() ? m_program.blocks.end()
		                                    : m_program.blocks.begin() + found->second;
	}

	const Block* FindBlock(uint32_t id) const {
		BuildBlockMaps();
		const auto found = m_block_by_id.find(id);
		return found == m_block_by_id.end() ? nullptr : found->second;
	}

	bool CanReach(const Block* start, const Block* target, const Block* avoid) const {
		BuildBlockMaps();
		std::vector<const Block*>       pending {start};
		std::vector<uint8_t>            visited(m_program.blocks.size(), 0u);
		std::unordered_set<const Block*> other; // blocks outside the list, if any
		while (!pending.empty()) {
			const auto* block = pending.back();
			pending.pop_back();
			if (block == nullptr || block == avoid) continue;
			const auto index = m_block_index.find(block);
			if (index != m_block_index.end()) {
				if (visited[index->second] != 0u) continue;
				visited[index->second] = 1u;
			} else if (!other.insert(block).second) {
				continue;
			}
			if (block == target) return true;
			for (const auto* next: block->ImmSuccessors()) pending.push_back(next);
		}
		return false;
	}

	bool ConditionalEdge(const Block* from, const Block* to, Value& condition,
	                     bool& positive) const {
		const auto position = BlockIter(from);
		const auto target = BlockIter(to);
		if (position == m_program.blocks.end() || target == m_program.blocks.end()) return false;
		const auto& info = m_program.block_info[position - m_program.blocks.begin()];
		const auto& term = info.terminator;
		const auto id = m_program.block_info[target - m_program.blocks.begin()].id;
		if (term.kind != CFG::TerminatorKind::ConditionalBranch ||
		    (term.true_block == id) == (term.false_block == id)) return false;
		condition = info.condition;
		positive = term.true_block == id;
		while (const auto* inst = condition.Resolve().TryInstruction()) {
			if (inst->GetOpcode() != ValueOpcode::LogicalNot) break;
			condition = inst->Arg(0);
			positive = !positive;
		}
		condition = condition.Resolve();
		return true;
	}

	Value PositiveUseGuard(const Block* use) const {
		if (use == nullptr || use->ImmPredecessors().size() != 1u) return {};
		Value condition;
		bool positive = false;
		return ConditionalEdge(use->ImmPredecessors()[0], use, condition, positive) && positive
		           ? condition : Value {};
	}

	Value SimplifyGuard(Value guard) const {
		const auto invariant = ResolveInvariantPhi(m_program, guard);
		return (invariant.IsEmpty() ? guard : invariant).Resolve();
	}

	bool Implies(Value guard, Value required) const {
		guard = SimplifyGuard(guard);
		required = required.Resolve();
		if (EquivalentValue(m_program, guard, required)) return true;
		const auto* inst = guard.TryInstruction();
		return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalAnd &&
		       (Implies(inst->Arg(0), required) || Implies(inst->Arg(1), required));
	}

	Value EqualLocalKey(Value guard, Value key) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return {};
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			const auto left = EqualLocalKey(inst->Arg(0), key);
			return left.IsEmpty() ? EqualLocalKey(inst->Arg(1), key) : left;
		}
		if (inst->GetOpcode() != ValueOpcode::IEqual32 || inst->NumArgs() != 2u) return {};
		if (EquivalentValue(m_program, inst->Arg(0), key)) return inst->Arg(1).Resolve();
		if (EquivalentValue(m_program, inst->Arg(1), key)) return inst->Arg(0).Resolve();
		return {};
	}

	struct AffineOffset {
		Value    index;
		uint64_t stride = 0;
		uint64_t offset = 0;
	};

	bool MatchAffineOffset(Value value, Value guard, AffineOffset& out,
	                       uint32_t depth = 0) const {
		if (depth > 16u) return false;
		value = value.Resolve();
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) {
			out.offset = immediate;
			return true;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::SelectU32 && inst->NumArgs() == 3u &&
		    Implies(guard, inst->Arg(0))) {
			return MatchAffineOffset(inst->Arg(1), guard, out, depth + 1u);
		}
		if (inst->GetOpcode() == ValueOpcode::IAdd32 && inst->NumArgs() == 2u) {
			AffineOffset left, right;
			if (!MatchAffineOffset(inst->Arg(0), guard, left, depth + 1u) ||
			    !MatchAffineOffset(inst->Arg(1), guard, right, depth + 1u) ||
			    (!left.index.IsEmpty() && !right.index.IsEmpty() &&
			     !EquivalentValue(m_program, left.index, right.index))) return false;
			out.index = left.index.IsEmpty() ? right.index : left.index;
			out.stride = left.stride + right.stride;
			out.offset = left.offset + right.offset;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		if (inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 && inst->NumArgs() == 2u &&
		    ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
			if (!MatchAffineOffset(inst->Arg(0), guard, out, depth + 1u)) return false;
			out.stride <<= immediate;
			out.offset <<= immediate;
			return out.stride <= UINT32_MAX && out.offset <= UINT32_MAX;
		}
		out.index = value;
		out.stride = 1u;
		return value.GetType() == Type::U32;
	}

	bool ImpliesNonzero(Value guard, Value value) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return ImpliesNonzero(inst->Arg(0), value) ||
			       ImpliesNonzero(inst->Arg(1), value);
		}
		if (inst->GetOpcode() != ValueOpcode::INotEqual32 || inst->NumArgs() != 2u)
			return false;
		uint32_t zero = 1u;
		return (ImmediateU32(inst->Arg(0), zero) && zero == 0u &&
		        EquivalentValue(m_program, inst->Arg(1), value)) ||
		       (ImmediateU32(inst->Arg(1), zero) && zero == 0u &&
		        EquivalentValue(m_program, inst->Arg(0), value));
	}

	bool ImpliesIndexBelow32(Value guard, Value index) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return ImpliesIndexBelow32(inst->Arg(0), index) ||
			       ImpliesIndexBelow32(inst->Arg(1), index);
		}
		if (inst->NumArgs() != 2u) return false;
		uint32_t limit = 0;
		return (inst->GetOpcode() == ValueOpcode::SLessThan32 &&
		        EquivalentValue(m_program, inst->Arg(0), index) &&
		        ImmediateU32(inst->Arg(1), limit) && limit == 32u) ||
		       (inst->GetOpcode() == ValueOpcode::SGreaterThan32 &&
		        ImmediateU32(inst->Arg(0), limit) && limit == 32u &&
		        EquivalentValue(m_program, inst->Arg(1), index));
	}

	bool ClearsFirstSetBit(Value value, Value mask) const {
		const auto* update = value.Resolve().TryInstruction();
		if (update == nullptr || update->GetOpcode() != ValueOpcode::BitwiseXor32 ||
		    update->NumArgs() != 2u) return false;
		Value bit;
		if (EquivalentValue(m_program, update->Arg(0), mask)) bit = update->Arg(1);
		else if (EquivalentValue(m_program, update->Arg(1), mask)) bit = update->Arg(0);
		else return false;
		const auto* shift = bit.Resolve().TryInstruction();
		uint32_t one = 0;
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    shift->NumArgs() != 2u || !ImmediateU32(shift->Arg(0), one) || one != 1u)
			return false;
		Value position = shift->Arg(1).Resolve();
		const auto* masked = position.TryInstruction();
		uint32_t lane_mask = 0;
		if (masked != nullptr && masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
		    masked->NumArgs() == 2u) {
			if (ImmediateU32(masked->Arg(0), lane_mask) && lane_mask == 31u)
				position = masked->Arg(1);
			else if (ImmediateU32(masked->Arg(1), lane_mask) && lane_mask == 31u)
				position = masked->Arg(0);
		}
		const auto* first = position.Resolve().TryInstruction();
		return first != nullptr && first->GetOpcode() == ValueOpcode::FindILsb32 &&
		       first->NumArgs() == 1u && EquivalentValue(m_program, first->Arg(0), mask);
	}

	Value InitialClearingMask(Value value, const Block* update_block) const {
		const auto* phi = value.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 2u || phi->GetType() != Type::U32) return {};
		for (uint32_t back = 0; back < 2u; ++back) {
			if (phi->PhiBlock(back) != update_block ||
			    !ClearsFirstSetBit(phi->Arg(back), value)) continue;
			const auto initial = phi->Arg(back ^ 1u).Resolve();
			return ValidateRuntimeValue(m_program, initial, RuntimeValueType::Integer)
			           ? initial : Value {};
		}
		return {};
	}

	bool EdgeOn(const Block* from, const Block* to, Value predicate, bool positive) const {
		Value condition;
		bool taken = false;
		return ConditionalEdge(from, to, condition, taken) && taken == positive &&
		       EquivalentValue(m_program, condition, predicate);
	}

	Value BoundedSetBitMask(Value index, Value guard) const {
		const auto* phi = index.Resolve().TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->NumArgs() != 3u || phi->GetType() != Type::U32 ||
		    !ImpliesIndexBelow32(guard, index)) return {};
		for (uint32_t bit_arm = 0; bit_arm < 3u; ++bit_arm) {
			const auto bit_guard = PositiveUseGuard(phi->PhiBlock(bit_arm));
			const auto* selected = phi->Arg(bit_arm).Resolve().TryInstruction();
			if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
			    selected->NumArgs() != 3u ||
			    !Implies(bit_guard, selected->Arg(0))) continue;
			const auto* first = selected->Arg(1).Resolve().TryInstruction();
			if (first == nullptr || first->GetOpcode() != ValueOpcode::FindILsb32 ||
			    first->NumArgs() != 1u || !ImpliesNonzero(bit_guard, first->Arg(0)))
				continue;
			const auto initial_mask = InitialClearingMask(first->Arg(0), phi->PhiBlock(bit_arm));
			if (initial_mask.IsEmpty()) continue;
			for (uint32_t sentinel_arm = 0; sentinel_arm < 3u; ++sentinel_arm) {
				if (sentinel_arm == bit_arm) continue;
				const auto sentinel_guard = PositiveUseGuard(phi->PhiBlock(sentinel_arm));
				const auto* sentinel = phi->Arg(sentinel_arm).Resolve().TryInstruction();
				uint32_t bound = 0;
				if (sentinel == nullptr || sentinel->GetOpcode() != ValueOpcode::SelectU32 ||
				    sentinel->NumArgs() != 3u ||
				    !Implies(sentinel_guard, sentinel->Arg(0)) ||
				    !ImmediateU32(sentinel->Arg(1), bound) || bound != 32u) continue;
				const auto loop_active = sentinel->Arg(0).Resolve();
				const auto* active_phi = loop_active.TryInstruction();
				if (active_phi == nullptr || active_phi->GetOpcode() != ValueOpcode::Phi ||
				    active_phi->NumArgs() != 2u) continue;
				bool invariant = false;
				for (uint32_t initial = 0; initial < 2u; ++initial) {
					invariant = Implies(guard, active_phi->Arg(initial)) &&
					            EdgeOn(active_phi->PhiBlock(initial ^ 1u), active_phi->Parent(),
					                   active_phi->Arg(initial ^ 1u), true);
					if (invariant) break;
				}
				if (!invariant) continue;
				const auto other_arm = 3u - bit_arm - sentinel_arm;
				if (EdgeOn(phi->PhiBlock(other_arm), phi->Parent(), loop_active, false))
					return initial_mask;
			}
		}
		return {};
	}

	bool MatchUniformizedMaterialKey(Value key, const Inst& image,
	                                DescriptorSource::IndirectDescriptor& indirect,
	                                DescriptorSource& material_source, uint32_t pc) {
		const auto guard = PositiveUseGuard(image.Parent());
		if (guard.IsEmpty()) return false;
		const auto local = EqualLocalKey(guard, key);
		const auto* selected = local.Resolve().TryInstruction();
		if (selected == nullptr || selected->GetOpcode() != ValueOpcode::SelectU32 ||
		    selected->NumArgs() != 3u || !Implies(guard, selected->Arg(0))) return false;
		const auto active = selected->Arg(0).Resolve();
		const auto* read = selected->Arg(1).Resolve().TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadAddressU32 ||
		    read->NumArgs() != 4u || !EquivalentValue(m_program, read->Arg(3), active))
			return false;
		uint32_t high = 1u;
		if (!ImmediateU32(read->Arg(2), high) || high != 0u) return false;
		const auto memory_index = read->Flags<MemoryFlags>().index;
		if (memory_index >= m_program.memory_info.size()) return false;
		const auto& memory = m_program.memory_info[memory_index];
		if (memory.kind != ResourceKind::Global || memory.data_bits != 32u ||
		    memory.data_dwords != 1u || !MemoryIndexBelongsTo(memory_index, *read)) return false;
		const auto* material_handle = read->Arg(0).Resolve().TryInstruction();
		if (material_handle == nullptr ||
		    material_handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    !MakeRuntimeTableSource(*material_handle, pc, material_source)) return false;
		AffineOffset offset;
		if (!MatchAffineOffset(read->Arg(1), active, offset) || offset.index.IsEmpty() ||
		    offset.stride == 0u || offset.offset + memory.offset > UINT32_MAX ||
		    offset.offset + memory.offset + 31u * offset.stride + 4u >
		        static_cast<uint64_t>(UINT32_MAX) + 1u) return false;
		const auto mask = BoundedSetBitMask(offset.index, active);
		if (mask.IsEmpty()) return false;
		indirect.selector = DescriptorSource::IndirectDescriptor::SelectorRead {
		    .source = InternSource(material_source),
		    .stride = static_cast<uint32_t>(offset.stride),
		    .offset = static_cast<uint32_t>(offset.offset + memory.offset)};
		indirect.key_count = Value(32u);
		indirect.selector_mask = mask;
		return true;
	}

	Value BoundedLoopCount(Value key, const Block* use) const {
		const auto* phi = key.Resolve().TryInstruction();
		if (m_shader_writes || phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi ||
		    phi->GetType() != Type::U32 || phi->NumArgs() != 2u ||
		    m_program.blocks.size() != m_program.block_info.size()) {
			return {};
		}
		bool induction = false;
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto zero = phi->Arg(initial).Resolve();
			const auto* step = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			if (!zero.IsImmediate() || zero.GetType() != Type::U32 || zero.U32() != 0u ||
			    step == nullptr || step->GetOpcode() != ValueOpcode::IAdd32 ||
			    step->Parent() != phi->PhiBlock(initial ^ 1u)) {
				continue;
			}
			uint32_t increment = 0;
			induction = (step->Arg(0).Resolve() == key &&
			             ImmediateU32(step->Arg(1), increment) && increment == 1u) ||
			            (step->Arg(1).Resolve() == key &&
			             ImmediateU32(step->Arg(0), increment) && increment == 1u);
			if (induction) break;
		}
		if (!induction) return {};

		const auto contains = [&](auto&& self, Value value, const Inst* comparison) -> bool {
			value = value.Resolve();
			if (value.TryInstruction() == comparison) return true;
			const auto* inst = value.TryInstruction();
			return inst != nullptr && inst->GetOpcode() == ValueOpcode::LogicalAnd &&
			       (self(self, inst->Arg(0), comparison) || self(self, inst->Arg(1), comparison));
		};
		for (const auto& use_of_key: phi->Uses()) {
			const auto* compare = use_of_key.user;
			if (compare->GetOpcode() != ValueOpcode::SLessThan32 || use_of_key.operand != 0u ||
			    !ValidateRuntimeValue(m_program, compare->Arg(1))) continue;
			for (uint32_t i = 0; i < m_program.block_info.size(); ++i) {
				const auto& info = m_program.block_info[i];
				const auto* negated = info.condition.Resolve().TryInstruction();
				if (info.terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
				    compare->Parent() != m_program.blocks[i] || negated == nullptr ||
				    negated->GetOpcode() != ValueOpcode::LogicalNot ||
				    !contains(contains, negated->Arg(0), compare) ||
				    use == m_program.blocks[i] ||
				    CanReach(m_program.blocks.front(), use, m_program.blocks[i]) ||
				    CanReach(phi->Parent(), use, m_program.blocks[i]) ||
				    CanReach(FindBlock(info.terminator.true_block), use, phi->Parent()) ||
				    !CanReach(FindBlock(info.terminator.false_block), use, phi->Parent())) {
					continue;
				}
				return compare->Arg(1);
			}
		}
		return {};
	}

	// KYTY_HEAP_KEY_MASK (adapted from KytyPS5-Frank's ResolveDescriptorPhi): a descriptor word that
	// is a Phi web where the register held the T# word on the sampling paths and a data value on
	// others (PPSA03671's 0xbf3eac34fedbad2d restores a spilled heap T# with V_READLANE after loops
	// that reuse the spill lanes). The one leaf read from the descriptor's own table (`reference`,
	// the table of its plain words) is the word; every other leaf must be plainly data (an
	// immediate, a value with uses outside Phis and image handles, or a computed value), so two
	// real descriptors still count as selected.
	Value ResolveDescriptorPhi(Value value, const Inst* reference) const {
		if (const auto invariant = ResolveInvariantPhi(m_program, value); !invariant.IsEmpty()) {
			return invariant;
		}
		value            = value.Resolve();
		const auto* root = value.TryInstruction();
		if (reference == nullptr || root == nullptr || root->GetOpcode() != ValueOpcode::Phi) {
			return {};
		}
		Value                           found;
		std::vector<const Inst*>        pending {root};
		std::unordered_set<const Inst*> seen;
		while (!pending.empty()) {
			const auto* phi = pending.back();
			pending.pop_back();
			if (!seen.insert(phi).second) {
				continue;
			}
			for (size_t i = 0; i < phi->NumArgs(); i++) {
				const auto  leaf = phi->Arg(i).Resolve();
				const auto* inst = leaf.TryInstruction();
				if (inst != nullptr && inst->GetOpcode() == ValueOpcode::Phi) {
					pending.push_back(inst);
					continue;
				}
				const bool data =
				    leaf.IsImmediate() ||
				    (inst != nullptr && std::ranges::any_of(inst->Uses(), [](const Use& use) {
					     const auto op = use.user->GetOpcode();
					     return op != ValueOpcode::Phi && op != ValueOpcode::GetImageResource;
				     }));
				if (data) {
					continue;
				}
				uint32_t    memory_index = 0;
				const auto* memory = inst != nullptr ? ScalarReadMemory(*inst, memory_index) : nullptr;
				if (memory == nullptr) {
					if (inst != nullptr && inst->GetOpcode() == ValueOpcode::GetUserData) {
						return {};
					}
					continue;
				}
				const auto* table = inst->Arg(0).Resolve().TryInstruction();
				if (table == nullptr || !EquivalentValue(m_program, Value(table), Value(reference)) ||
				    (!found.IsEmpty() && !EquivalentValue(m_program, found, leaf))) {
					return {};
				}
				found = leaf;
			}
		}
		return found;
	}

	bool TryMakeIndirectImage(Inst& handle, uint32_t pc, IndirectImagePlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetImageResource || handle.NumArgs() != 8u) {
			return false;
		}
		Inst* table_handle = nullptr;
		Value key;
		uint32_t table_offset = 0;
		uint32_t key_mask     = UINT32_MAX;
		Inst*    masked       = nullptr;
		bool     word0_shared = false;
		// KYTY_HEAP_KEY_MASK: the table of the descriptor's plain (non-Phi) words, for
		// ResolveDescriptorPhi.
		const Inst* reference_table = nullptr;
		if (GetCodegenOptions().heap_key_mask) {
			for (uint32_t dword = 0; dword < plan.reads.size() && reference_table == nullptr; ++dword) {
				const auto* plain = handle.Arg(dword).Resolve().TryInstruction();
				uint32_t    memory_index = 0;
				if (plain != nullptr && plain->GetOpcode() != ValueOpcode::Phi &&
				    ScalarReadMemory(*plain, memory_index) != nullptr) {
					reference_table = plain->Arg(0).Resolve().TryInstruction();
				}
			}
		}
		for (uint32_t dword = 0; dword < plan.reads.size(); ++dword) {
			auto* read = handle.Arg(dword).Resolve().TryInstruction();
			if (read != nullptr && read->GetOpcode() == ValueOpcode::Phi && reference_table != nullptr) {
				read = ResolveDescriptorPhi(Value(read), reference_table).TryInstruction();
			}
			if (read == nullptr) {
				return false;
			}
			uint32_t current_mask   = UINT32_MAX;
			Inst*    current_masked = nullptr;
			uint32_t memory_index = 0;
			const auto* memory = ScalarReadMemory(*read, memory_index);
			if (memory == nullptr || memory->offset > INT32_MAX || (memory->offset & 3u) != 0u ||
			    !MemoryIndexBelongsTo(memory_index, *read)) {
				return false;
			}
			auto* current_handle = read->Arg(0).Resolve().TryInstruction();
			Value current_key;
			uint32_t offset = 0;
			if (current_handle == nullptr ||
			    current_handle->GetOpcode() != (memory->kind == ResourceKind::ScalarAddress
			                                      ? ValueOpcode::GetAddressResource
			                                      : ValueOpcode::GetBufferResource) ||
			    (memory->kind == ResourceKind::ScalarAddress && read->Parent() != handle.Parent()) ||
			    (table_handle != nullptr &&
			     !EquivalentValue(m_program, Value(table_handle), Value(current_handle))) ||
			    !MatchTableOffset(read->Arg(1), current_key, offset, &current_mask, &current_masked) ||
			    memory->offset > UINT32_MAX - offset) {
				return false;
			}
			offset += memory->offset;
			if (dword == 0u) {
				key = current_key;
				table_offset = offset;
				key_mask     = current_mask;
				masked       = current_masked;
			} else if (!EquivalentValue(m_program, key, current_key) || key_mask != current_mask ||
			           static_cast<uint64_t>(table_offset) + dword * sizeof(uint32_t) != offset) {
				return false;
			}
			table_handle = current_handle;
			// Several image instructions may sample the same T# (each has its own handle over
			// the same reads): every user must be an image handle with equivalent dwords.
			std::vector<const Inst*> image_users {&handle};
			for (const auto& use: read->Uses()) {
				const Inst* other = use.user;
				if (other == &handle || other->GetOpcode() != ValueOpcode::GetImageResource ||
				    other->NumArgs() != 8u) {
					continue;
				}
				bool same = true;
				for (uint32_t i = 0; same && i < 8u; i++) {
					same = EquivalentValue(m_program, handle.Arg(i), other->Arg(i));
				}
				if (same) image_users.push_back(other);
			}
			if (!UsesOnly(*read, image_users)) {
				// A T# word the shader also uses as a value stays a real read through the table;
				// only the handle's own uses are planned. An explicit-LOD gather passes words 1 and
				// 3 by value (its mip selection reads them on the GPU, Translator::IMAGE_GATHER:
				// Wolverine's HiZ CS 0x37ab7d3ab9bff0b9), and Wolverine's material pixel shaders
				// (0x487054eeb2d7f3e7 and five more) take their sampler key from T# word 4.
				// KYTY_TSHARP_DATA_WORDS=0: only the gather words (the first fix).
				static const bool data_words = [] {
					const char* value = std::getenv("KYTY_TSHARP_DATA_WORDS");
					return value == nullptr || std::strcmp(value, "0") != 0;
				}();
				const bool lod_words_only = std::ranges::all_of(read->Uses(), [&](const Use& use) {
					if (std::ranges::find(image_users, use.user) != image_users.end()) return true;
					return use.user->GetOpcode() == ValueOpcode::ImageGatherRaw && use.operand >= 3u &&
					       std::ranges::find(image_users, use.user->Arg(0).Resolve().TryInstruction()) !=
					           image_users.end();
				});
				// KYTY_HEAP_KEY_MASK: word 0 too, for a heap key only the shader knows (checked below,
				// the plan is then bindless). PPSA03671's material pixel shader 0x885fa47aec73f696
				// spills word 0 with V_WRITELANE and restores it with V_READLANE before the sample.
				if ((!lod_words_only && !data_words) ||
				    (dword == 0u && !GetCodegenOptions().heap_key_mask)) {
					return false;
				}
				if (dword == 0u) {
					word0_shared = true;
				}
				plan.memory[dword] = UINT32_MAX;
				continue;
			}
			plan.memory[dword] = memory_index;
			plan.reads[dword] = read;
		}

		// A shared word 0 (KYTY_HEAP_KEY_MASK) only for a key the host cannot evaluate: a key the
		// SRT walk evaluates keeps its flat slot, as before.
		if (word0_shared && ValidateRuntimeValue(m_program, key)) {
			return false;
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source)) {
			return false;
		}
		DescriptorSource material_source;
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset = table_offset;
		// Enumerate the keys the table can reach (upstream's plan). When that fails and the device
		// supports bindless images, a sampled-only image over a buffer heap looks its key up at run
		// time instead (Senaxx's bindless images): a program the enumeration covers translates as
		// before.
		const auto enumerable = [&]() -> bool {
			if (table_source.dword_count == 2u) {
				const auto* selector = key.Resolve().TryInstruction();
				const bool bitscan = selector != nullptr && selector->GetOpcode() == ValueOpcode::FindILsb32 &&
				    selector->NumArgs() == 1u && !m_shader_writes &&
				    NonzeroOnEntry(selector->Arg(0), handle.Parent());
				if (bitscan) {
					indirect.key_count = Value(32u);
				} else {
					indirect.key_count = BoundedLoopCount(key, handle.Parent());
				}
				if (indirect.key_count.IsEmpty() &&
				    !MatchUniformizedMaterialKey(key, handle, indirect, material_source, pc)) {
					return false;
				}
				if ((table_offset & 3u) != 0u ||
				    (bitscan && table_offset > UINT32_MAX - (32u * 32u - 1u))) return false;
			} else {
				auto* material_read = key.Resolve().TryInstruction();
				uint32_t material_memory_index = 0;
				const auto* memory = material_read != nullptr
				                         ? ScalarReadMemory(*material_read, material_memory_index) : nullptr;
				// The record read may carry its own immediate offset (S_BUFFER_LOAD offset=4 reads the
				// second dword of a per-material record): it adds to the selector offset.
				if (table_offset != 0u || memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
				    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
				    !MemoryIndexBelongsTo(material_memory_index, *material_read)) {
					return false;
				}
				Value selector;
				DescriptorSource::IndirectDescriptor::SelectorRead selector_read;
				if (!MatchMaterialOffset(material_read->Arg(1), selector, selector_read.stride,
				                         selector_read.offset) ||
				    selector_read.offset > UINT32_MAX - memory->offset) {
					return false;
				}
				selector_read.offset += memory->offset;
				const auto* shift = plan.reads[0]->Arg(1).Resolve().TryInstruction();
				const std::array<const Inst*, 1> material_users {shift};
				if (!UsesOnly(*material_read, material_users) || !UsesOnly(*shift, plan.reads)) {
					return false;
				}
				const auto* material_handle = material_read->Arg(0).Resolve().TryInstruction();
				if (material_handle == nullptr ||
				    material_handle->GetOpcode() != ValueOpcode::GetBufferResource ||
				    !MakeRuntimeTableSource(*material_handle, pc, material_source)) {
					return false;
				}
				selector_read.source = InternSource(material_source);
				indirect.selector    = selector_read;
			}
			return true;
		};
		// A masked key is a lane value (MatchTableOffset): no enumeration covers it. A shared
		// word 0 has no planning read for the material selector either.
		if (masked != nullptr || word0_shared || !enumerable()) {
			const bool sampled_only = std::ranges::all_of(handle.Uses(), [](const Use& use) {
				return ImageOpcodeInfoOf(use.user->GetOpcode()).resource_class ==
				       ImageResourceClass::Sampled;
			});
			if (!BindlessImagesEnabled() || !sampled_only || table_source.dword_count != 4u) {
				return false;
			}
			indirect              = {};
			indirect.table_offset = table_offset;
			indirect.table_stride = 32u;
			indirect.bindless     = true;
			material_source       = {};
		}
		if (masked != nullptr) {
			// The shader looks up key & (mask >> 5), computed where the S_AND computes the offset
			// (the key dominates it, and it dominates the T# reads).
			auto*      block = masked->Parent();
			const auto where = std::ranges::find_if(block->Instructions(),
			                                        [&](const Inst& inst) { return &inst == masked; });
			if (where == block->Instructions().end()) {
				return false;
			}
			key = Value(&*block->PrependNewInst(where, ValueOpcode::BitwiseAnd32, {key, Value(key_mask)}));
		}
		indirect.table_source = InternSource(table_source);
		DescriptorSource image_source;
		image_source.dword_count = 8u;
		image_source.dwords.fill(Value(0u));
		std::copy_n(material_source.dwords.begin(), material_source.dword_count,
		            image_source.dwords.begin());
		std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		            image_source.dwords.begin() + 4u);
		image_source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(image_source);
		plan.key = key;
		plan.roots = image_source.dwords;
		return true;
	}

	// offset = key << shift (+ immediate): a record of 1 << shift bytes selected by a key.
	static bool MatchRecordOffset(Value value, Value& key, uint32_t& shift, uint32_t& immediate) {
		immediate       = 0;
		const auto* inst = value.Resolve().TryInstruction();
		if (inst != nullptr && inst->GetOpcode() == ValueOpcode::IAdd32 && inst->NumArgs() == 2u) {
			const auto operand = ImmediateU32(inst->Arg(1), immediate) ? 0u
			                     : ImmediateU32(inst->Arg(0), immediate) ? 1u
			                                                            : 2u;
			if (operand == 2u) {
				return false;
			}
			inst = inst->Arg(operand).Resolve().TryInstruction();
		}
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    inst->NumArgs() != 2u || !ImmediateU32(inst->Arg(1), shift) || shift < 3u ||
		    shift > 12u) {
			return false;
		}
		key = inst->Arg(0).Resolve();
		return key.GetType() == Type::U32;
	}

	// A bindless image whose T# is read through a pointer that a keyed record holds: Spider-Man
	// Miles Morales' hit callbacks (an S_BUFFER_LOAD_DWORDX4 of record key << 4 from a V#
	// table, whose dwords 2-3 point at the material; the T# is an S_LOAD_DWORDX8 from there).
	// Only for a sampled-only handle, when the device has bindless images; the host reads the
	// pointer and then the T# for each key it resolves.
	bool TryMakeRecordPointerImage(Inst& handle, uint32_t pc, IndirectImagePlan& plan) {
		if (!BindlessImagesEnabled() || handle.GetOpcode() != ValueOpcode::GetImageResource ||
		    handle.NumArgs() != 8u ||
		    !std::ranges::all_of(handle.Uses(), [](const Use& use) {
			    return ImageOpcodeInfoOf(use.user->GetOpcode()).resource_class ==
			           ImageResourceClass::Sampled;
		    })) {
			return false;
		}
		plan = {};
		plan.memory.fill(UINT32_MAX);
		const Inst* pointer      = nullptr;
		uint32_t    tsharp_offset = 0;
		for (uint32_t dword = 0; dword < plan.reads.size(); ++dword) {
			auto*      read         = handle.Arg(dword).Resolve().TryInstruction();
			uint32_t   memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			uint32_t   dynamic = 1u;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarAddress ||
			    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !ImmediateU32(read->Arg(1), dynamic) || dynamic != 0u) {
				return false;
			}
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetAddressResource ||
			    current->NumArgs() != 2u ||
			    (pointer != nullptr && !EquivalentValue(m_program, Value(pointer), Value(current)))) {
				return false;
			}
			if (dword == 0u) {
				tsharp_offset = memory->offset;
			} else if (static_cast<uint64_t>(tsharp_offset) + dword * sizeof(uint32_t) !=
			           memory->offset) {
				return false;
			}
			pointer = current;
			// T# words the shader also uses otherwise (another sample of the same material, or a
			// V_WRITELANE spill the callee restores them from) stay real reads through BDA; only
			// the handle's own words become planning-only.
			const std::array<const Inst*, 1> image_users {&handle};
			if (UsesOnly(*read, image_users)) {
				plan.memory[dword] = memory_index;
				plan.reads[dword]  = read;
			}
		}
		// The pointer: two dwords of one keyed record of a buffer table.
		Inst*    table_handle   = nullptr;
		Value    key;
		uint32_t shift          = 0;
		uint32_t record_pointer = 0;
		for (uint32_t half = 0; half < 2u; ++half) {
			const auto* read         = pointer->Arg(half).Resolve().TryInstruction();
			uint32_t    memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			Value       current_key;
			uint32_t    current_shift = 0;
			uint32_t    immediate     = 0;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
			    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !MatchRecordOffset(read->Arg(1), current_key, current_shift, immediate)) {
				return false;
			}
			auto*      current = read->Arg(0).Resolve().TryInstruction();
			const auto offset  = static_cast<uint64_t>(memory->offset) + immediate;
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource) {
				return false;
			}
			if (half == 0u) {
				table_handle   = current;
				key            = current_key;
				shift          = current_shift;
				record_pointer = static_cast<uint32_t>(offset);
			} else if (!EquivalentValue(m_program, Value(table_handle), Value(current)) ||
			           !EquivalentValue(m_program, key, current_key) || shift != current_shift ||
			           static_cast<uint64_t>(record_pointer) + 4u != offset) {
				return false;
			}
		}
		if (record_pointer + 8u > (1u << shift)) {
			return false;
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return false;
		}
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset   = tsharp_offset;
		indirect.table_stride   = 1u << shift;
		indirect.record_pointer = record_pointer;
		indirect.bindless       = true;
		indirect.table_source   = InternSource(table_source);
		DescriptorSource image_source;
		image_source.dword_count = 8u;
		image_source.dwords.fill(Value(0u));
		std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		            image_source.dwords.begin() + 4u);
		image_source.indirect_descriptor = indirect;
		plan.handle = &handle;
		plan.source = InternSource(image_source);
		plan.key    = key;
		plan.roots  = image_source.dwords;
		return true;
	}

	// The pointer of a keyed record: two dwords of one record (key << shift) of a buffer table.
	bool MatchRecordPointer(const Inst& address, Inst*& table_handle, Value& key, uint32_t& shift,
	                        uint32_t& record_pointer) const {
		if (address.GetOpcode() != ValueOpcode::GetAddressResource || address.NumArgs() != 2u) {
			return false;
		}
		for (uint32_t half = 0; half < 2u; ++half) {
			const auto* read         = address.Arg(half).Resolve().TryInstruction();
			uint32_t    memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			Value       current_key;
			uint32_t    current_shift = 0;
			uint32_t    immediate     = 0;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
			    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !MatchRecordOffset(read->Arg(1), current_key, current_shift, immediate)) {
				return false;
			}
			auto*      current = read->Arg(0).Resolve().TryInstruction();
			const auto offset  = static_cast<uint64_t>(memory->offset) + immediate;
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource) {
				return false;
			}
			if (half == 0u) {
				table_handle   = current;
				key            = current_key;
				shift          = current_shift;
				record_pointer = static_cast<uint32_t>(offset);
			} else if (!EquivalentValue(m_program, Value(table_handle), Value(current)) ||
			           !EquivalentValue(m_program, key, current_key) || shift != current_shift ||
			           static_cast<uint64_t>(record_pointer) + 4u != offset) {
				return false;
			}
		}
		return record_pointer + 8u <= (1u << shift);
	}

	// Four dwords that are one scalar read each of `handle_opcode`'s handle at consecutive byte
	// offsets with no dynamic part: the handle and the first offset.
	bool MatchConsecutiveWords(const Inst& owner, ValueOpcode handle_opcode, const Inst*& source,
	                           uint32_t& offset) const {
		source = nullptr;
		for (uint32_t dword = 0; dword < 4u; ++dword) {
			const auto* read         = owner.Arg(dword).Resolve().TryInstruction();
			uint32_t    memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			uint32_t    dynamic = 1u;
			if (memory == nullptr || (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !ImmediateU32(read->Arg(1), dynamic) || dynamic != 0u) {
				return false;
			}
			const auto* handle = read->Arg(0).Resolve().TryInstruction();
			if (handle == nullptr || handle->GetOpcode() != handle_opcode ||
			    (source != nullptr && !EquivalentValue(m_program, Value(source), Value(handle)))) {
				return false;
			}
			if (dword == 0u) {
				offset = memory->offset;
			} else if (static_cast<uint64_t>(offset) + dword * 4u != memory->offset) {
				return false;
			}
			source = handle;
		}
		return true;
	}

	// The nested heap of Miles' hit callbacks (8 samples in 2 of the 218 callees): the T# is an
	// S_BUFFER_LOAD_DWORDX8 at a waterfall key << 5 from a heap whose V# is S_BUFFER_LOAD_DWORDX4
	// at a fixed offset of a buffer whose V# the material holds (S_LOAD_DWORDX4 through the record
	// pointer). One bindless heap keyed by record * NestedKeys + the inner key; an inner key past
	// NestedKeys samples the placeholder.
	static constexpr uint32_t NestedKeys = 64;

	bool TryMakeNestedRecordImage(Inst& handle, uint32_t pc, IndirectImagePlan& plan) {
		if (!BindlessImagesEnabled() || handle.GetOpcode() != ValueOpcode::GetImageResource ||
		    handle.NumArgs() != 8u ||
		    !std::ranges::all_of(handle.Uses(), [](const Use& use) {
			    return ImageOpcodeInfoOf(use.user->GetOpcode()).resource_class ==
			           ImageResourceClass::Sampled;
		    })) {
			return false;
		}
		plan = {};
		plan.memory.fill(UINT32_MAX);
		const Inst* heap_handle  = nullptr;
		Value       inner_key;
		uint32_t    table_offset = 0;
		for (uint32_t dword = 0; dword < plan.reads.size(); ++dword) {
			auto*      read         = handle.Arg(dword).Resolve().TryInstruction();
			uint32_t   memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			Value      current_key;
			uint32_t   offset = 0;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
			    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !MatchTableOffset(read->Arg(1), current_key, offset) ||
			    memory->offset > UINT32_MAX - offset) {
				return false;
			}
			offset += memory->offset;
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource ||
			    (heap_handle != nullptr &&
			     !EquivalentValue(m_program, Value(heap_handle), Value(current)))) {
				return false;
			}
			if (dword == 0u) {
				inner_key    = current_key;
				table_offset = offset;
			} else if (!EquivalentValue(m_program, inner_key, current_key) ||
			           static_cast<uint64_t>(table_offset) + dword * 4u != offset) {
				return false;
			}
			heap_handle = current;
			const std::array<const Inst*, 1> image_users {&handle};
			if (UsesOnly(*read, image_users)) {
				plan.memory[dword] = memory_index;
				plan.reads[dword]  = read;
			}
		}
		// The heap's V#: S_BUFFER_LOAD_DWORDX4 at a fixed offset of a buffer ...
		const Inst* outer_handle = nullptr;
		uint32_t    nested_heap  = 0;
		if (!MatchConsecutiveWords(*heap_handle, ValueOpcode::GetBufferResource, outer_handle,
		                           nested_heap)) {
			return false;
		}
		// ... whose V# is S_LOAD_DWORDX4 through the record pointer.
		const Inst* pointer       = nullptr;
		uint32_t    nested_vsharp = 0;
		if (!MatchConsecutiveWords(*outer_handle, ValueOpcode::GetAddressResource, pointer,
		                           nested_vsharp)) {
			return false;
		}
		Inst*    table_handle   = nullptr;
		Value    record_key;
		uint32_t shift          = 0;
		uint32_t record_pointer = 0;
		if (!MatchRecordPointer(*pointer, table_handle, record_key, shift, record_pointer)) {
			return false;
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return false;
		}
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset   = table_offset;
		indirect.table_stride   = 1u << shift;
		indirect.record_pointer = record_pointer;
		indirect.nested_vsharp  = nested_vsharp;
		indirect.nested_heap    = nested_heap;
		indirect.nested_keys    = NestedKeys;
		indirect.bindless       = true;
		indirect.table_source   = InternSource(table_source);
		DescriptorSource image_source;
		image_source.dword_count = 8u;
		image_source.dwords.fill(Value(0u));
		std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		            image_source.dwords.begin() + 4u);
		image_source.indirect_descriptor = indirect;
		// key = record * NestedKeys + inner, or UINT32_MAX (outside every heap) for an inner key
		// past NestedKeys.
		auto*      block = handle.Parent();
		const auto where = std::ranges::find_if(block->Instructions(),
		                                        [&](const Inst& inst) { return &inst == &handle; });
		const auto scaled = Value(&*block->PrependNewInst(where, ValueOpcode::IMul32,
		                                                  {record_key, Value(NestedKeys)}));
		const auto sum = Value(&*block->PrependNewInst(where, ValueOpcode::IAdd32, {scaled, inner_key}));
		const auto inside = Value(&*block->PrependNewInst(where, ValueOpcode::ULessThan32,
		                                                  {inner_key, Value(NestedKeys)}));
		const auto key = Value(&*block->PrependNewInst(where, ValueOpcode::SelectU32,
		                                               {inside, sum, Value(UINT32_MAX)}));
		plan.handle = &handle;
		plan.source = InternSource(image_source);
		plan.key    = key;
		plan.roots  = image_source.dwords;
		return true;
	}

	// KYTY_HEAP_KEY_MASK: the heap of PPSA03671's material callbacks (expanded S_SWAPPC_B64 callees
	// of the tiled material kernels, e.g. 0xe6d76d24f59f8015): the T# is an S_BUFFER_LOAD_DWORDX8 at
	// key << 5 from a heap whose V# is S_LOAD_DWORDX4 at a fixed offset through the record pointer
	// (a 16-byte record {function, data} at a waterfall key << 4; the callee gets the data pointer).
	// The nested form without the middle buffer: nested_heap = UINT32_MAX tells the host that the V#
	// at nested_vsharp past the pointer is the heap itself.
	bool TryMakeRecordHeapImage(Inst& handle, uint32_t pc, IndirectImagePlan& plan) {
		if (!GetCodegenOptions().heap_key_mask || !BindlessImagesEnabled() ||
		    handle.GetOpcode() != ValueOpcode::GetImageResource || handle.NumArgs() != 8u ||
		    !std::ranges::all_of(handle.Uses(), [](const Use& use) {
			    return ImageOpcodeInfoOf(use.user->GetOpcode()).resource_class ==
			           ImageResourceClass::Sampled;
		    })) {
			return false;
		}
		plan = {};
		plan.memory.fill(UINT32_MAX);
		const Inst* heap_handle  = nullptr;
		Value       inner_key;
		uint32_t    table_offset = 0;
		for (uint32_t dword = 0; dword < plan.reads.size(); ++dword) {
			auto*      read         = handle.Arg(dword).Resolve().TryInstruction();
			uint32_t   memory_index = 0;
			const auto* memory = read != nullptr ? ScalarReadMemory(*read, memory_index) : nullptr;
			Value      current_key;
			uint32_t   offset = 0;
			if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
			    (memory->offset & 3u) != 0u || memory->offset > INT32_MAX ||
			    !MatchTableOffset(read->Arg(1), current_key, offset) ||
			    memory->offset > UINT32_MAX - offset) {
				return false;
			}
			offset += memory->offset;
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource ||
			    (heap_handle != nullptr &&
			     !EquivalentValue(m_program, Value(heap_handle), Value(current)))) {
				return false;
			}
			if (dword == 0u) {
				inner_key    = current_key;
				table_offset = offset;
			} else if (!EquivalentValue(m_program, inner_key, current_key) ||
			           static_cast<uint64_t>(table_offset) + dword * 4u != offset) {
				return false;
			}
			heap_handle = current;
			const std::array<const Inst*, 1> image_users {&handle};
			if (UsesOnly(*read, image_users)) {
				plan.memory[dword] = memory_index;
				plan.reads[dword]  = read;
			}
		}
		// The heap's V#: S_LOAD_DWORDX4 at a fixed offset through the record pointer.
		const Inst* pointer       = nullptr;
		uint32_t    nested_vsharp = 0;
		if (!MatchConsecutiveWords(*heap_handle, ValueOpcode::GetAddressResource, pointer,
		                           nested_vsharp)) {
			return false;
		}
		Inst*    table_handle   = nullptr;
		Value    record_key;
		uint32_t shift          = 0;
		uint32_t record_pointer = 0;
		if (!MatchRecordPointer(*pointer, table_handle, record_key, shift, record_pointer)) {
			return false;
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return false;
		}
		DescriptorSource::IndirectDescriptor indirect;
		indirect.table_offset   = table_offset;
		indirect.table_stride   = 1u << shift;
		indirect.record_pointer = record_pointer;
		indirect.nested_vsharp  = nested_vsharp;
		indirect.nested_heap    = UINT32_MAX;
		indirect.nested_keys    = NestedKeys;
		indirect.bindless       = true;
		indirect.table_source   = InternSource(table_source);
		DescriptorSource image_source;
		image_source.dword_count = 8u;
		image_source.dwords.fill(Value(0u));
		std::copy_n(table_source.dwords.begin(), table_source.dword_count,
		            image_source.dwords.begin() + 4u);
		image_source.indirect_descriptor = indirect;
		// key = record * NestedKeys + inner, or UINT32_MAX (outside every heap) for an inner key
		// past NestedKeys, as for the nested heap.
		auto*      block = handle.Parent();
		const auto where = std::ranges::find_if(block->Instructions(),
		                                        [&](const Inst& inst) { return &inst == &handle; });
		const auto scaled = Value(&*block->PrependNewInst(where, ValueOpcode::IMul32,
		                                                  {record_key, Value(NestedKeys)}));
		const auto sum = Value(&*block->PrependNewInst(where, ValueOpcode::IAdd32, {scaled, inner_key}));
		const auto inside = Value(&*block->PrependNewInst(where, ValueOpcode::ULessThan32,
		                                                  {inner_key, Value(NestedKeys)}));
		const auto key = Value(&*block->PrependNewInst(where, ValueOpcode::SelectU32,
		                                               {inside, sum, Value(UINT32_MAX)}));
		plan.handle = &handle;
		plan.source = InternSource(image_source);
		plan.key    = key;
		plan.roots  = image_source.dwords;
		return true;
	}

	bool IsBindlessPlan(const IndirectImagePlan& plan) const {
		const auto* source = Source(plan.source);
		return source != nullptr && source->indirect_descriptor.has_value() &&
		       source->indirect_descriptor->bindless;
	}

	// Stopgap (Senaxx's default sampler): the S# beside a bindless T# that is read per key too
	// (from the same material record) has no plan. Sample with one fixed sampler (trilinear,
	// wrap, full LOD range) rather than skipping the program; only filtering and edge addressing
	// can differ from the material's own sampler.
	void UseDefaultSamplerIfUnresolved(Value value, bool sample_adjust, uint32_t pc) {
		auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetSamplerResource ||
		    handle->NumArgs() != 4u) {
			return;
		}
		DescriptorSource descriptor;
		MakeSource(*handle, 4u, true, sample_adjust, descriptor, pc);
		uint32_t bad_dword = 0;
		if (ValidateSource(descriptor, bad_dword)) {
			return;
		}
		constexpr std::array<uint32_t, 4> DefaultSampler {
		    0x00000000u,                             // wrap on every axis
		    0xfffu << 12u,                           // max_lod 255.9
		    (1u << 20u) | (1u << 22u) | (2u << 26u), // bilinear mag/min, linear mip
		    0x00000000u};
		for (uint32_t dword = 0; dword < 4u; dword++) {
			handle->SetArg(dword, Value(DefaultSampler[dword]));
		}
	}

	const IndirectImagePlan* FindIndirectImage(const Inst& handle) const {
		const auto found = m_indirect_by_handle.find(&handle);
		return found == m_indirect_by_handle.end() ? nullptr : &m_indirect_images[found->second];
	}

	// Memory entries the indirect plans read only for planning (a flag per entry: an RT kernel
	// expansion has thousands of plans and checks every memory operation against them).
	bool IsIndirectPlanningMemory(uint32_t index) const {
		return index < m_planning_memory.size() && m_planning_memory[index] != 0u;
	}

	void PlanIndirectImages() {
		for (auto* block: m_program.blocks) {
			for (auto& inst: *block) {
				if (ImageOpcodeInfoOf(inst.GetOpcode()).access == ImageAccess::None ||
				    inst.NumArgs() == 0u) {
					continue;
				}
				auto* handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || FindIndirectImage(*handle) != nullptr) {
					continue;
				}
				IndirectImagePlan plan;
				if (TryMakeIndirectImage(*handle, inst.Flags<MemoryFlags>().pc, plan) ||
				    TryMakeRecordPointerImage(*handle, inst.Flags<MemoryFlags>().pc, plan) ||
				    TryMakeNestedRecordImage(*handle, inst.Flags<MemoryFlags>().pc, plan) ||
				    TryMakeRecordHeapImage(*handle, inst.Flags<MemoryFlags>().pc, plan)) {
					for (const auto index: plan.memory) {
						if (index != UINT32_MAX) {
							if (index >= m_planning_memory.size()) m_planning_memory.resize(index + 1u, 0u);
							m_planning_memory[index] = 1u;
						}
					}
					m_indirect_by_handle.emplace(plan.handle, m_indirect_images.size());
					m_indirect_images.push_back(std::move(plan));
				}
			}
		}
	}

	bool GetHandle(Value value, ValueOpcode expected, uint32_t width, uint32_t pc, Inst*& handle,
	               uint32_t& source, bool sampler = false, bool sample_adjust = false) {
		handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != expected) {
			Fail(pc, fmt::format("memory operation requires {}", ValueOpcodeName(expected)));
		}
		DescriptorSource descriptor;
		MakeSource(*handle, width, sampler, sample_adjust, descriptor, pc);
		uint32_t bad_dword = 0;
		if (!ValidateSource(descriptor, bad_dword)) {
			if (expected == ValueOpcode::GetBufferResource &&
			    std::all_of(descriptor.dwords.begin(), descriptor.dwords.begin() + width,
			                [](Value word) { return word.Resolve().GetType() == Type::U32; })) {
				return false;
			}
			if (m_indirect_scalar_buffers) {
				MarkUnresolved(pc);
				return false;
			}
			Fail(pc, fmt::format("{} dword {} is not a valid runtime value",
			                     ValueOpcodeName(expected), bad_dword));
		}
		source = InternSource(descriptor);
		return true;
	}

	void ValidateAddressHandle(Value value, uint32_t pc) const {
		const auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource) {
			Fail(pc, "address operation requires GetAddressResource");
		}
		if (handle->NumArgs() != 2) {
			Fail(pc, "GetAddressResource must have two address dwords");
		}
	}

	uint32_t AddBuffer(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.buffers.size(); i++) {
			if (m_info.buffers[i].source == source) {
				Merge(m_info.buffers[i], memory, op, pc);
				return i;
			}
		}
		if (m_info.buffers.size() >= ShaderInfo::MaxBuffers) {
			return UINT32_MAX;
		}
		BufferResource resource;
		resource.source       = source;
		resource.first_use_pc = pc;
		Merge(resource, memory, op, pc);
		m_info.buffers.push_back(resource);
		return static_cast<uint32_t>(m_info.buffers.size() - 1);
	}

	static void Merge(BufferResource& resource, const MemoryInfo& memory, ValueOpcode op,
	                  uint32_t pc) {
		const auto access        = BufferAccessOf(op);
		const bool atomic        = access == BufferAccess::Atomic;
		const bool write         = access == BufferAccess::Write || atomic;
		resource.first_use_pc    = std::min(resource.first_use_pc, pc);
		resource.max_byte_extent = std::max(resource.max_byte_extent, ByteExtent(memory));
		resource.read            = resource.read || !write || atomic;
		resource.written         = resource.written || write;
		resource.atomic          = resource.atomic || atomic;
		resource.formatted       = resource.formatted || memory.formatted;
		resource.scalar          = resource.scalar || op == ValueOpcode::ReadConstBuffer ||
		                           memory.kind == ResourceKind::ScalarBuffer;
	}

	uint32_t AddImage(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc) {
		const auto resource_class = ImageOpcodeInfoOf(op).resource_class;
		const bool atomic64 = ImageOpcodeInfoOf(op).access == ImageAccess::Atomic &&
		                      memory.data_bits == 64u;
		const bool dynamic_mip =
		    (resource_class == ImageResourceClass::Storage && memory.image_has_mip) ||
		    (op == ValueOpcode::ImageGatherRaw &&
		     (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u);
		const auto mip = dynamic_mip ? ImageMipMode::Dynamic : ImageMipMode::None;
		const bool depth = (memory.image_sample_flags & Decoder::ImageSampleFlagCompare) != 0;
		for (uint32_t i = 0; i < m_info.images.size(); i++) {
			auto& image = m_info.images[i];
			if (image.source == source && image.resource_class == resource_class &&
			    image.dimension == memory.image_dimension && image.mip_mode == mip &&
			    image.depth_compare == depth && image.r128 == memory.image_r128 &&
			    image.atomic64 == atomic64) {
				Merge(image, op, pc);
				return i;
			}
		}
		if (m_info.images.size() >= ShaderInfo::MaxImages) {
			return UINT32_MAX;
		}
		ImageResource image;
		image.source         = source;
		image.first_use_pc   = pc;
		image.resource_class = resource_class;
		image.dimension      = memory.image_dimension;
		image.mip_mode       = mip;
		image.depth_compare  = depth;
		image.r128           = memory.image_r128;
		image.atomic64       = atomic64;
		Merge(image, op, pc);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	static void Merge(ImageResource& image, ValueOpcode op, uint32_t pc) {
		const auto access  = ImageOpcodeInfoOf(op).access;
		const bool atomic  = access == ImageAccess::Atomic;
		const bool write   = access == ImageAccess::Write || atomic;
		image.first_use_pc = std::min(image.first_use_pc, pc);
		image.read         = image.read || !write || atomic;
		image.written      = image.written || write;
		image.atomic       = image.atomic || atomic;
	}

	uint32_t AddSampler(uint32_t source, uint32_t pc) {
		for (uint32_t i = 0; i < m_info.samplers.size(); i++) {
			if (m_info.samplers[i].source == source) {
				m_info.samplers[i].first_use_pc = std::min(m_info.samplers[i].first_use_pc, pc);
				return i;
			}
		}
		if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
			return UINT32_MAX;
		}
		m_info.samplers.push_back({source, pc});
		return static_cast<uint32_t>(m_info.samplers.size() - 1);
	}

	void AddSampledPair(uint32_t image, uint32_t sampler, uint32_t pc) {
		for (auto& pair: m_info.sampled_pairs) {
			if (pair.image == image && pair.sampler == sampler) {
				pair.first_use_pc = std::min(pair.first_use_pc, pc);
				return;
			}
		}
		m_info.sampled_pairs.push_back({image, sampler, pc});
	}

	void AddHandlePatch(Inst* handle, uint32_t resource, uint32_t pc) {
		for (const auto& patch: m_handle_patches) {
			if (patch.handle == handle) {
				if (patch.resource != resource) {
					Fail(pc, fmt::format("{} is reused with incompatible resource classes",
					                     ValueOpcodeName(handle->GetOpcode())));
				}
				return;
			}
		}
		m_handle_patches.push_back({handle, resource});
	}

	void AddMemoryPatch(uint32_t index, uint32_t resource, uint32_t sampler, bool has_sampler,
	                    uint32_t pc) {
		for (auto& patch: m_memory_patches) {
			if (patch.index != index) {
				continue;
			}
			if (patch.resource != resource ||
			    (has_sampler && patch.has_sampler && patch.sampler != sampler)) {
				Fail(pc, "memory metadata is reused with incompatible resources");
			}
			if (has_sampler) {
				patch.sampler     = sampler;
				patch.has_sampler = true;
			}
			return;
		}
		m_memory_patches.push_back({index, resource, sampler, has_sampler});
	}

	void Collect(Inst& inst) {
		const auto op           = inst.GetOpcode();
		const auto buffer       = BufferAccessOf(op);
		const auto address_info = AddressOpcodeInfoOf(op);
		const auto image_info   = ImageOpcodeInfoOf(op);
		if (buffer == BufferAccess::None && address_info.access == AddressAccess::None &&
		    image_info.access == ImageAccess::None) {
			return;
		}
		const auto flags = inst.Flags<MemoryFlags>();
		if (flags.index >= m_program.memory_info.size()) {
			Fail(flags.pc, fmt::format("memory metadata index {} is out of range", flags.index));
		}
		if (inst.NumArgs() == 0) {
			Fail(flags.pc, "memory operation has no resource handle");
		}
		const auto& memory = m_program.memory_info[flags.index];
		if (memory.planning_only || IsIndirectPlanningMemory(flags.index)) {
			return;
		}
		Inst*    handle   = nullptr;
		uint32_t source   = 0;
		uint32_t resource = 0;

		if (buffer != BufferAccess::None) {
			if (!GetHandle(inst.Arg(0), ValueOpcode::GetBufferResource, 4, flags.pc, handle,
			               source)) {
				// KYTY_SRT_VARIANT_READS: an S_BUFFER_LOAD through a V# the shader computed (e.g. a
				// BVH instance's record) or any raw vector load through one reads through BDA like
				// the raw DWORD x2-x4 loads below.
				const bool indirect_variant =
				    m_indirect_scalar_buffers &&
				    ((memory.kind == ResourceKind::ScalarBuffer &&
				      memory.SupportsIndirectBufferLoad(op)) ||
				     (memory.kind == ResourceKind::Buffer && memory.SupportsIndirectRawLoad(op)));
				// KYTY_BDA_WRITES: a raw store or atomic through one writes through BDA (compute
				// only: the renderer settles each dispatch of such a program synchronously).
				const bool indirect_write = m_bda_writes && memory.kind == ResourceKind::Buffer &&
				                            m_program.stage == ShaderType::Compute &&
				                            memory.SupportsIndirectRawWrite(op);
				if (!indirect_variant && !indirect_write &&
				    (memory.kind != ResourceKind::Buffer || !memory.SupportsIndirectBufferLoad(op))) {
					if (m_indirect_scalar_buffers) {
						MarkUnresolved(flags.pc);
						return;
					}
					Fail(flags.pc,
					     "buffer descriptor is not a valid runtime value; GPU-selected access "
					     "requires a raw DWORD x2/x3/x4 load");
				}
				auto& indirect = m_program.memory_info[flags.index];
				indirect.kind  = ResourceKind::IndirectBuffer;
				if (m_indirect_scalar_buffers || indirect_write) {
					// No bound buffer: resource-control-flow planning skips it (it has no source).
					indirect.resource = NoIndirectBufferResource;
				}
				if (indirect_write) {
					// An unbounded writer: every memo and verdict that trusts buffer contents is
					// retired when it is recorded (CommitBindings, has_address_writes).
					m_info.bda_writes             = true;
					m_program.has_address_writes = true;
				}
				m_info.uses_dma = true;
				return;
			}
			resource = AddBuffer(source, memory, op, flags.pc);
			if (resource == UINT32_MAX) {
				Fail(flags.pc, "buffer resource limit exceeded");
			}
			AddHandlePatch(handle, resource, flags.pc);
			AddMemoryPatch(flags.index, resource, 0, false, flags.pc);
			return;
		}
		if (address_info.access != AddressAccess::None) {
			if (!IsAddressResourceKind(memory.kind)) {
				Fail(flags.pc, "address operation has invalid resource kind");
			}
			if (memory.kind == ResourceKind::Scratch) {
				handle = inst.Arg(0).Resolve().TryInstruction();
				if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetScratchResource ||
				    handle->NumArgs() != 0) {
					Fail(flags.pc, "scratch operation requires GetScratchResource");
				}
				if (m_program.scratch_dwords == 0) {
					Fail(flags.pc, "scratch operation requires a nonzero AGC per-thread size");
				}
				return;
			}
			ValidateAddressHandle(inst.Arg(0), flags.pc);
			if (address_info.access == AddressAccess::Write && memory.kind == ResourceKind::Global &&
			    inst.NumArgs() >= 5) {
				// The global leg of an aperture-routed FLAT store: the predicate is exec AND NOT
				// (shared OR private). When the address is provably an aperture tag the leg never
				// runs, so it is switched off rather than requiring ownership tracking.
				const auto last      = inst.NumArgs() - 1;
				const auto predicate = inst.Arg(last).Resolve();
				const auto* source   = predicate.TryInstruction();
				if (source != nullptr && source->GetOpcode() == ValueOpcode::LogicalAnd &&
				    IsNoncanonicalFlatAddress(inst.Arg(2), source->Arg(0))) {
					m_dead_stores.push_back(&inst);
					return;
				}
			}
			if (address_info.access == AddressAccess::Write) {
				m_program.has_address_writes = true;
			}
			m_info.uses_dma = true;
			return;
		}

		if (memory.kind != ResourceKind::Image ||
		    image_info.resource_class == ImageResourceClass::None) {
			Fail(flags.pc, "image operation has invalid resource kind");
		}
		handle               = inst.Arg(0).Resolve().TryInstruction();
		const auto* indirect = handle != nullptr ? FindIndirectImage(*handle) : nullptr;
		if (indirect != nullptr) {
			source = indirect->source;
		} else if (!GetHandle(inst.Arg(0), ValueOpcode::GetImageResource, 8, flags.pc, handle,
		                      source)) {
			return; // unresolved (KYTY_SRT_VARIANT_READS): the shader is skipped
		}
		resource = AddImage(source, memory, op, flags.pc);
		if (resource == UINT32_MAX) {
			Fail(flags.pc, "image resource limit exceeded");
		}
		AddHandlePatch(handle, resource, flags.pc);
		uint32_t sampler = 0;
		if (image_info.needs_sampler) {
			if (inst.NumArgs() < 2) {
				Fail(flags.pc, "sampled image operation has no sampler handle");
			}
			Inst*      sampler_handle = nullptr;
			uint32_t   sampler_source = 0;
			const bool sample_adjust =
			    (memory.image_sample_flags & Decoder::ImageSampleFlagAdjust) != 0;
			if (indirect != nullptr && IsBindlessPlan(*indirect)) {
				UseDefaultSamplerIfUnresolved(inst.Arg(1), sample_adjust, flags.pc);
			}
			if (!GetHandle(inst.Arg(1), ValueOpcode::GetSamplerResource, 4, flags.pc,
			               sampler_handle, sampler_source, true, sample_adjust)) {
				return; // unresolved (KYTY_SRT_VARIANT_READS): the shader is skipped
			}
			sampler = AddSampler(sampler_source, flags.pc);
			if (sampler == UINT32_MAX) {
				Fail(flags.pc, "sampler resource limit exceeded");
			}
			m_info.samplers[sampler].gather_lod |=
			    op == ValueOpcode::ImageGatherRaw &&
			    (memory.image_sample_flags & Decoder::ImageSampleFlagLod) != 0u;
			AddHandlePatch(sampler_handle, sampler, flags.pc);
			AddSampledPair(resource, sampler, flags.pc);
		}
		AddMemoryPatch(flags.index, resource, sampler, image_info.needs_sampler, flags.pc);
	}

	const DescriptorSource* Source(uint32_t source) const {
		return source < m_sources.size() ? &m_sources[source] : nullptr;
	}

	void LinkImageAliases() {
		for (auto& buffer: m_info.buffers) {
			const auto* buffer_source = Source(buffer.source);
			if (buffer_source == nullptr || buffer_source->dword_count != 4) {
				continue;
			}
			for (uint32_t image = 0; image < m_info.images.size(); image++) {
				const auto* image_source = Source(m_info.images[image].source);
				if (image_source == nullptr || image_source->dword_count != 8 ||
				    image_source->indirect_descriptor.has_value()) {
					continue;
				}
				bool alias = true;
				for (uint32_t dword = 0; dword < 4; dword++) {
					alias = alias && EquivalentValue(m_program, buffer_source->dwords[dword],
					                                 image_source->dwords[dword]);
				}
				if (alias) {
					buffer.image_alias = image;
					break;
				}
			}
		}
	}

	Program&                                   m_program;
	ShaderInfo                                 m_info;
	mutable bool                               m_block_maps_built = false;
	mutable std::unordered_map<const Block*, uint32_t> m_block_index;
	mutable std::unordered_map<uint32_t, const Block*> m_block_by_id;
	std::vector<uint8_t>                       m_planning_memory;
	std::unordered_map<const Inst*, size_t>    m_indirect_by_handle;
	mutable bool                               m_memory_users_built = false;
	mutable std::vector<uint32_t>              m_memory_users;
	std::vector<DescriptorSource>              m_sources;
	std::vector<HandlePatch>                   m_handle_patches;
	std::vector<MemoryPatch>                   m_memory_patches;
	std::vector<IndirectImagePlan>             m_indirect_images;
	std::vector<std::pair<const Inst*, Value>> m_descriptor_selections;
	bool                                       m_shader_writes = false;
	bool                                       m_indirect_scalar_buffers = false;
	bool                                       m_bda_writes              = false;
	uint32_t                                   m_unresolved_pc = UINT32_MAX;

public:
	// KYTY_SRT_VARIANT_READS: a descriptor the shader computes at runtime that has no BDA path
	// (an image, a sampler, a formatted access, or a write without KYTY_BDA_WRITES). Planning it at runtime removed the flat
	// slot that would have failed to evaluate and dropped the dispatch, so the program is dropped
	// here instead of failing: never worse than without the switch.
	// KYTY_SRT_UNRESOLVED_ALL=1 (diagnostics): every unresolved access is logged, not only
	// the first one the skip line names, so one translation lists all that a program lacks.
	void MarkUnresolved(uint32_t pc) {
		static const bool all = [] {
			const char* value = std::getenv("KYTY_SRT_UNRESOLVED_ALL");
			return value != nullptr && value[0] == '1';
		}();
		if (all) {
			// Console and log (the emulator's log file), as the skip line itself.
			Log::WriteToConsoleAndLog(fmt::format(
			    "KYTY_SRT_UNRESOLVED_ALL: {} shader 0x{:016x} pc=0x{:08x}\n",
			    StageName(m_program.stage), m_program.shader_hash, pc));
		}
		m_unresolved_pc = std::min(m_unresolved_pc, pc);
	}
	[[nodiscard]] uint32_t UnresolvedPc() const { return m_unresolved_pc; }
};

} // namespace

uint32_t TrackResources(Program& program, bool indirect_scalar_buffers, bool bda_writes) {
	Tracker tracker(program, indirect_scalar_buffers, bda_writes);
	tracker.Run();
	return tracker.UnresolvedPc();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
