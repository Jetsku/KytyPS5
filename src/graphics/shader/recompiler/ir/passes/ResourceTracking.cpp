#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/frontend/decode/ImageOps.h"
#include "graphics/shader/recompiler/ir/BindlessBindings.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cinttypes>
#include <bit>
#include <cmath>
#include <limits>
#include <optional>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <bit>
#include <fmt/format.h>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
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
// KYTY_TEXTURE_LAYER_BOUND=1 (default off; chenxiao07/KytyPS5 d44e93d81, 6b57b4939, c965985ff): the
// largest layer each sampled 1D/2D array's samples and gathers can reach (ImageResource::
// max_array_index); its view then holds no layer past it (descriptors.cpp).
std::atomic<int> g_layer_bound_test {-1}; // SetLayerBoundForTest: -1 the switch decides
bool LayerBoundEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_TEXTURE_LAYER_BOUND");
		return value != nullptr && value[0] == '1';
	}();
	const int test = g_layer_bound_test.load(std::memory_order_relaxed);
	return test >= 0 ? test != 0 : enabled;
}

// KYTY_IMAGE_CUBE_TRACK=1 (default off; upstream 9a29dc62c): a tracked image is a cube when its
// MIMG instructions declare DIM CUBE (MemoryInfo::image_cube). A null T# with
// KYTY_NULL_IMAGE_KEEP_DIM and a bindless image then keep the cube their instructions sample;
// without it the tracked image stays non-cube as before.
bool ImageCubeTrackEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_IMAGE_CUBE_TRACK");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}
} // namespace

void SetLayerBoundForTest(int enabled) {
	g_layer_bound_test.store(enabled, std::memory_order_relaxed);
}

namespace {

// KYTY_TABLE_KEY_BOUNDS: matches MaxIndirectImageProbes (ResourceMaterialization.cpp).
constexpr uint32_t MaxTableKeyProbes = 65536u;
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

// KYTY_SELECTOR_* (upstream KytyPS5's LoopBoundProof, adapted): prove a loop cannot continue after
// its bound fails, both on entry and after an arbitrary previous iteration. Header Phis are
// substituted simultaneously; all unsupported expressions remain unconstrained (a small BDD over
// U1 values). This state exists only while tracking.
class LoopBoundProof {
public:
	LoopBoundProof(const Program& program, const Inst& induction, const Inst& bound)
	    : m_program(program), m_induction(induction), m_bound(bound),
	      m_values(1u + induction.NumArgs()) {}

	bool Excludes(Value condition, bool positive) {
		// Upstream 4dc15e27c: the current-iteration values of each incoming edge stay valid
		// across the guard conditions tested with this proof.
		for (uint32_t incoming = 0; incoming < m_induction.NumArgs(); ++incoming) {
			m_incoming = m_induction.PhiBlock(incoming);
			m_current  = 1u + incoming;
			if (Evaluate(condition, true) != (positive ? 0u : 1u)) return false;
		}
		return true;
	}

private:
	struct Node {
		uint32_t variable = UINT32_MAX;
		uint32_t low      = 0;
		uint32_t high     = 0;
	};

	uint32_t NodeFor(uint32_t variable, uint32_t low, uint32_t high) {
		if (low == high) return low;
		const auto [it, inserted] = m_nodes_by_key.try_emplace(std::array {variable, low, high},
		                                                       static_cast<uint32_t>(m_nodes.size()));
		if (inserted) m_nodes.push_back({variable, low, high});
		return it->second;
	}

	uint32_t Unknown(Type type) {
		if (type == Type::U1) return NodeFor(m_variables++, 0u, 1u);
		m_nodes.emplace_back();
		return static_cast<uint32_t>(m_nodes.size() - 1u);
	}

	uint32_t Select(uint32_t condition, uint32_t yes, uint32_t no) {
		if (condition == 0u) return no;
		if (condition == 1u || yes == no) return yes;
		if (yes == 1u && no == 0u) return condition;
		const std::array key {condition, yes, no};
		if (const auto found = m_choices.find(key); found != m_choices.end()) return found->second;
		const auto variable =
		    std::min({m_nodes[condition].variable, m_nodes[yes].variable, m_nodes[no].variable});
		const auto arm = [&](uint32_t value, bool high) {
			const auto node = m_nodes[value];
			return node.variable == variable ? (high ? node.high : node.low) : value;
		};
		const auto low    = Select(arm(condition, false), arm(yes, false), arm(no, false));
		const auto high   = Select(arm(condition, true), arm(yes, true), arm(no, true));
		const auto result = NodeFor(variable, low, high);
		m_choices.emplace(key, result);
		return result;
	}

	uint32_t Compare(const Inst& inst, uint32_t left, uint32_t right) {
		if (left == right) {
			if (inst.GetOpcode() == ValueOpcode::IEqual32) return 1u;
			if (inst.GetOpcode() == ValueOpcode::INotEqual32) return 0u;
		}
		const auto key = std::tuple {inst.GetOpcode(), inst.Flags<uint64_t>(), left, right};
		if (const auto found = m_predicates.find(key); found != m_predicates.end()) return found->second;
		const auto variable = std::min(m_nodes[left].variable, m_nodes[right].variable);
		uint32_t   result   = 0;
		if (variable == UINT32_MAX) {
			result = Unknown(Type::U1);
		} else {
			const auto lhs  = m_nodes[left];
			const auto rhs  = m_nodes[right];
			const auto low  = Compare(inst, lhs.variable == variable ? lhs.low : left,
			                          rhs.variable == variable ? rhs.low : right);
			const auto high = Compare(inst, lhs.variable == variable ? lhs.high : left,
			                          rhs.variable == variable ? rhs.high : right);
			result          = Select(NodeFor(variable, 0u, 1u), high, low);
		}
		m_predicates.emplace(key, result);
		return result;
	}

	uint32_t Evaluate(Value value, bool current) {
		value = value.Resolve();
		if (value.IsImmediate()) {
			if (value.GetType() == Type::U1) return value.U1() ? 1u : 0u;
			for (const auto& [literal, node]: m_literals) {
				if (literal == value) return node;
			}
			const auto node = Unknown(value.GetType());
			m_literals.emplace_back(value, node);
			return node;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return Unknown(value.GetType());
		if (current && inst == &m_bound) return 0u;
		auto& values = m_values[current ? m_current : 0u];
		if (const auto found = values.find(inst); found != values.end()) {
			if (found->second == UINT32_MAX) found->second = Unknown(value.GetType());
			return found->second;
		}
		const auto cached = values.emplace(inst, UINT32_MAX).first;
		auto       result = UINT32_MAX;
		const auto arg    = [&](uint32_t index) { return Evaluate(inst->Arg(index), current); };
		switch (inst->GetOpcode()) {
			case ValueOpcode::Phi: {
				const auto invariant = ResolveInvariantPhi(m_program, value);
				if (!invariant.IsEmpty()) {
					result = Evaluate(invariant, current);
				} else if (inst->Parent() == m_induction.Parent()) {
					if (current) {
						for (uint32_t i = 0; i < inst->NumArgs(); ++i) {
							if (inst->PhiBlock(i) == m_incoming) result = Evaluate(inst->Arg(i), false);
						}
					}
				} else if (inst->NumArgs() != 0u) {
					result = arg(0);
					for (uint32_t i = 1; i < inst->NumArgs(); ++i) {
						if (arg(i) != result) {
							result = UINT32_MAX;
							break;
						}
					}
				}
				break;
			}
			case ValueOpcode::LogicalNot: result = Select(arg(0), 0u, 1u); break;
			case ValueOpcode::LogicalAnd: {
				const auto left = arg(0);
				result          = left == 0u ? 0u : Select(left, arg(1), 0u);
				break;
			}
			case ValueOpcode::LogicalOr: {
				const auto left = arg(0);
				result          = left == 1u ? 1u : Select(left, 1u, arg(1));
				break;
			}
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectU32: {
				const auto condition = arg(0);
				result = condition == 0u   ? arg(2)
				         : condition == 1u ? arg(1)
				                           : Select(condition, arg(1), arg(2));
				break;
			}
			default:
				if (inst->GetType() == Type::U1 && inst->NumArgs() == 2u &&
				    inst->Arg(0).GetType() == Type::U32 && inst->Arg(1).GetType() == Type::U32) {
					result = Compare(*inst, arg(0), arg(1));
				}
				break;
		}
		// Only unsupported values and cycles need free variables.
		if (result == UINT32_MAX) {
			if (cached->second == UINT32_MAX) cached->second = Unknown(value.GetType());
			return cached->second;
		}
		cached->second = result;
		return result;
	}

	const Program&                                                              m_program;
	const Inst&                                                                 m_induction;
	const Inst&                                                                 m_bound;
	const Block*                                                                m_incoming  = nullptr;
	uint32_t                                                                    m_current   = 1;
	uint32_t                                                                    m_variables = 0;
	std::vector<Node>                                                           m_nodes {{}, {}};
	std::vector<std::pair<Value, uint32_t>>                                     m_literals;
	// Previous-iteration values, then current ones per incoming edge, kept across conditions.
	std::vector<std::map<const Inst*, uint32_t>>                                m_values;
	std::map<std::array<uint32_t, 3>, uint32_t>                                 m_nodes_by_key;
	std::map<std::array<uint32_t, 3>, uint32_t>                                 m_choices;
	std::map<std::tuple<ValueOpcode, uint64_t, uint32_t, uint32_t>, uint32_t> m_predicates;
};

class Tracker {
public:
	Tracker(Program& program, bool indirect_scalar_buffers, bool bda_writes)
	    : m_program(program), m_info(program.info),
	      m_indirect_scalar_buffers(indirect_scalar_buffers), m_bda_writes(bda_writes) {
		const auto& force = GetCodegenOptions().bda_force_shaders;
		m_force_bda       = std::ranges::find(force, program.shader_hash) != force.end();
		m_info.buffers.clear();
		m_info.images.clear();
		m_info.samplers.clear();
		m_info.sampled_pairs.clear();
		m_info.address_bases.clear();
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
		CollectDeferredSamplerReads();
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
		// KYTY_BINDLESS_SAMPLERS: the handle carries its key (which the shader reads) and the heap
		// V#'s first three dwords (which keep the plan's values alive).
		for (const auto& plan: m_bindless_samplers) {
			plan.handle->SetArg(0, plan.key);
			for (uint32_t dword = 0; dword < 3u; dword++) {
				plan.handle->SetArg(dword + 1u, plan.table_roots[dword]);
			}
		}
		std::erase_if(m_program.dynamic_reads, [&](Value value) {
			const auto* inst = value.Resolve().TryInstruction();
			return std::any_of(m_indirect_images.begin(), m_indirect_images.end(),
			                   [&](const IndirectImagePlan& plan) {
				return inst != nullptr && std::ranges::find(plan.reads, inst) != plan.reads.end();
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

	// KYTY_BINDLESS_SAMPLERS: a sampler handle whose S# is read from a sampler heap at a key.
	struct BindlessSamplerPlan {
		Inst*                handle = nullptr;
		uint32_t             source = 0;
		Value                key;
		std::array<Value, 3> table_roots {};
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
			    current.indirect_descriptor.has_value() != descriptor.indirect_descriptor.has_value() ||
			    current.bindless_sampler != descriptor.bindless_sampler) {
				continue;
			}
			if (current.indirect_descriptor.has_value()) {
				const auto& a = *current.indirect_descriptor;
				const auto& b = *descriptor.indirect_descriptor;
				if (a.selector != b.selector || a.table_source != b.table_source ||
				    a.table_offset != b.table_offset || a.table_immediate != b.table_immediate ||
				    a.table_stride != b.table_stride || a.table_record_bytes != b.table_record_bytes ||
				    a.table_scalar != b.table_scalar || a.workgroup_axis != b.workgroup_axis ||
				    a.workgroup_multiplier != b.workgroup_multiplier ||
				    a.workgroup_shift != b.workgroup_shift ||
				    a.bindless != b.bindless || a.runtime != b.runtime ||
				    a.record_pointer != b.record_pointer ||
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

	// any_selector (KYTY_SELECTOR_BITSCAN / KYTY_SELECTOR_SHARED_LIST): the selector need not be a
	// V_READFIRSTLANE, and selector << n (a power-of-two record) matches too.
	bool MatchMaterialOffset(Value value, Value& selector, uint32_t& stride, uint32_t& offset,
	                         bool any_selector = false) const {
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
		uint32_t    shift    = 0;
		if (any_selector && multiply != nullptr &&
		    multiply->GetOpcode() == ValueOpcode::ShiftLeftLogical32 && multiply->NumArgs() == 2u &&
		    ImmediateU32(multiply->Arg(1), shift) && shift < 32u) {
			stride   = 1u << shift;
			selector = multiply->Arg(0).Resolve();
		} else if (multiply == nullptr || multiply->GetOpcode() != ValueOpcode::IMul32 ||
		           multiply->NumArgs() != 2u) {
			return false;
		} else if (ImmediateU32(multiply->Arg(0), stride)) {
			selector = multiply->Arg(1).Resolve();
		} else if (ImmediateU32(multiply->Arg(1), stride)) {
			selector = multiply->Arg(0).Resolve();
		} else {
			return false;
		}
		const auto* selector_inst = selector.TryInstruction();
		return stride != 0u && selector_inst != nullptr &&
		       (any_selector || selector_inst->GetOpcode() == ValueOpcode::ReadFirstLane);
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
	// stride (KYTY_BINDLESS_STRIDE, adapted from usa-ratt/KytyPS5-GT7 a6cd035f0): when not null,
	// key * stride (an immediate multiply, or a shift of 3 to 12) matches too, and the stride is
	// returned (32 for the plain key << 5); without it only key << 5 matches.
	// KYTY_WORKGROUP_IMAGES (adapted from upstream KytyPS5 0ee880090, nmzik): the axis of a key that
	// is a compute program's workgroup id, or UINT32_MAX.
	uint32_t WorkgroupAxis(Value key) const {
		const auto* builtin = key.Resolve().TryInstruction();
		uint32_t    axis    = UINT32_MAX;
		if (m_program.stage != ShaderType::Compute || builtin == nullptr ||
		    builtin->GetOpcode() != ValueOpcode::GetBuiltin || builtin->NumArgs() != 2u ||
		    builtin->Arg(0) != Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) ||
		    !ImmediateU32(builtin->Arg(1), axis) || axis >= 3u) {
			return UINT32_MAX;
		}
		return axis;
	}

	// KYTY_WORKGROUP_IMAGES (upstream de3ed905c): key = (first + workgroup ID) * multiplier >> shift
	// with shift >= 32 (a UMulHi and a shift), where first is a runtime value; first and the
	// division are optional. Fills the indirect descriptor's workgroup fields on a match.
	bool MatchWorkgroupKey(Value key, DescriptorSource::IndirectDescriptor* indirect) const {
		uint32_t    bits = 0, multiplier = 0;
		const auto* shift = key.Resolve().TryInstruction();
		if (shift != nullptr && shift->GetOpcode() == ValueOpcode::ShiftRightLogical32) {
			const auto* high = shift->Arg(0).Resolve().TryInstruction();
			if (high != nullptr && high->GetOpcode() == ValueOpcode::UMulHi &&
			    ImmediateU32(shift->Arg(1), bits) && bits < 32u &&
			    ImmediateU32(high->Arg(1), multiplier)) {
				key = high->Arg(0);
				bits += 32u;
			} else {
				bits = 0;
			}
		} else if (shift != nullptr && shift->GetOpcode() == ValueOpcode::UMulHi &&
		           ImmediateU32(shift->Arg(1), multiplier)) {
			key  = shift->Arg(0);
			bits = 32u;
		}
		Value    first;
		uint32_t axis = WorkgroupAxis(key);
		if (axis == UINT32_MAX) {
			const auto* sum = key.Resolve().TryInstruction();
			if (sum == nullptr || sum->GetOpcode() != ValueOpcode::IAdd32) return false;
			for (uint32_t arm = 0; arm < 2u && axis == UINT32_MAX; ++arm) {
				const auto base = sum->Arg(arm ^ 1u);
				if (WorkgroupAxis(sum->Arg(arm)) == UINT32_MAX ||
				    !ValidateRuntimeValue(m_program, base, RuntimeValueType::Integer)) {
					continue;
				}
				axis  = WorkgroupAxis(sum->Arg(arm));
				first = base;
			}
			if (axis == UINT32_MAX) return false;
		}
		if (indirect != nullptr) {
			indirect->workgroup_axis       = axis;
			indirect->selector_first       = first;
			indirect->workgroup_multiplier = bits != 0u ? multiplier : 0u;
			indirect->workgroup_shift      = static_cast<uint8_t>(bits);
		}
		return true;
	}

	bool MatchTableOffset(Value value, Value& key, uint32_t& offset, uint32_t* key_mask = nullptr,
	                      Inst** masked = nullptr, uint32_t* stride = nullptr) const {
		offset = 0;
		if (key_mask != nullptr) *key_mask = UINT32_MAX;
		if (masked != nullptr) *masked = nullptr;
		if (stride != nullptr) *stride = 32u;
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
			if (stride != nullptr && inst->GetOpcode() == ValueOpcode::ShiftLeftLogical32 &&
			    ImmediateU32(inst->Arg(1), immediate) && immediate >= 3u && immediate <= 12u) {
				key     = inst->Arg(0).Resolve();
				*stride = 1u << immediate;
				return key.GetType() == Type::U32;
			}
			if (stride != nullptr && inst->GetOpcode() == ValueOpcode::IMul32) {
				uint32_t factor = 0;
				if (ImmediateU32(inst->Arg(0), factor)) {
					key = inst->Arg(1).Resolve();
				} else if (ImmediateU32(inst->Arg(1), factor)) {
					key = inst->Arg(0).Resolve();
				} else {
					return false;
				}
				*stride = factor;
				return factor != 0u && key.GetType() == Type::U32;
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
		return EqualLocalValue(guard, [&](Value at) { return EquivalentValue(m_program, at, key); });
	}

	// The value guard proves equal to one that matches (upstream 145578ac0's generalisation).
	template <typename Matches>
	Value EqualLocalValue(Value guard, const Matches& matches) const {
		guard = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return {};
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			const auto left = EqualLocalValue(inst->Arg(0), matches);
			return left.IsEmpty() ? EqualLocalValue(inst->Arg(1), matches) : left;
		}
		if (inst->GetOpcode() != ValueOpcode::IEqual32 || inst->NumArgs() != 2u) return {};
		if (matches(inst->Arg(0))) return inst->Arg(1).Resolve();
		if (matches(inst->Arg(1))) return inst->Arg(0).Resolve();
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

	// KYTY_TABLE_KEY_BOUNDS (adapted from upstream dfcf1c0b0, 20bb268a4, a3ac6d0d3 and b2053c286):
	// a constant maximum for an address-table image key that neither a bit scan nor a bounded loop
	// covers, so the table enumerates keys 0..maximum instead of the program being skipped (or
	// going bindless).

	// Bounds of an F32 value; -inf / +inf when a side is unknown.
	std::pair<float, float> KeyFloatRange(Value value, uint32_t depth = 0) const {
		constexpr std::pair<float, float> unknown {-INFINITY, INFINITY};
		value = value.Resolve();
		if (value.IsImmediate()) {
			if (value.GetType() == Type::F32) return {value.F32Value(), value.F32Value()};
			return unknown;
		}
		const auto* inst = value.TryInstruction();
		// A multiply-add evaluates three operands; keep the search shallow.
		if (depth > 12u || inst == nullptr) return unknown;
		const auto range  = [&](uint32_t arg) { return KeyFloatRange(inst->Arg(arg), depth + 1u); };
		const auto finite = [](const std::pair<float, float>& r) {
			return std::isfinite(r.first) && std::isfinite(r.second);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::BitCastF32U32:
			case ValueOpcode::BitCastU32F32: return range(0);
			// Clamping maps NaN to zero.
			case ValueOpcode::FPSaturate32: return {0.0f, 1.0f};
			case ValueOpcode::FPFloor32: {
				const auto [low, high] = range(0);
				return {std::floor(low), std::floor(high)};
			}
			case ValueOpcode::FPMin32: {
				const auto a = range(0), b = range(1);
				return {std::min(a.first, b.first), std::min(a.second, b.second)};
			}
			case ValueOpcode::FPMax32: {
				const auto a = range(0), b = range(1);
				return {std::max(a.first, b.first), std::max(a.second, b.second)};
			}
			case ValueOpcode::FPAdd32: {
				const auto a = range(0), b = range(1);
				return {a.first + b.first, a.second + b.second};
			}
			case ValueOpcode::FPMul32:
			case ValueOpcode::FPMad32:
			case ValueOpcode::FPFma32: {
				// The rounded multiply and add are monotonic, so the bounds lie at the corners.
				const auto a = range(0), b = range(1);
				if (!finite(a) || !finite(b)) return unknown;
				const auto c = inst->GetOpcode() == ValueOpcode::FPMul32
				                   ? std::pair<float, float> {0.0f, 0.0f}
				                   : range(2);
				const auto [low, high] = std::minmax({a.first * b.first, a.first * b.second,
				                                      a.second * b.first, a.second * b.second});
				return {low + c.first, high + c.second};
			}
			default: return unknown;
		}
	}

	// The largest value key can hold where guard (an EXEC or branch predicate, true on the lanes
	// that use key) holds: key < C, key <= C (inclusive, a3ac6d0d3), key == C and mirrored forms.
	uint32_t GuardKeyMaximum(Value guard, Value key, bool positive = true, uint32_t depth = 0) const {
		if (guard.IsEmpty() || depth > 8u) return UINT32_MAX;
		const auto* test = SimplifyGuard(guard).TryInstruction();
		if (test == nullptr) return UINT32_MAX;
		const auto op = test->GetOpcode();
		if (op == ValueOpcode::LogicalNot) {
			return GuardKeyMaximum(test->Arg(0), key, !positive, depth + 1u);
		}
		if ((op == ValueOpcode::LogicalAnd && positive) || (op == ValueOpcode::LogicalOr && !positive)) {
			return std::min(GuardKeyMaximum(test->Arg(0), key, positive, depth + 1u),
			                GuardKeyMaximum(test->Arg(1), key, positive, depth + 1u));
		}
		if (test->NumArgs() != 2u) return UINT32_MAX;
		uint32_t   limit    = 0;
		const bool key_left = EquivalentValue(m_program, test->Arg(0), key) &&
		                      ImmediateU32(test->Arg(1), limit);
		const bool key_right = !key_left && EquivalentValue(m_program, test->Arg(1), key) &&
		                       ImmediateU32(test->Arg(0), limit);
		if (!key_left && !key_right) return UINT32_MAX;
		// Normalise to "key OP limit" on the lanes where the test holds.
		auto opcode = op;
		if (key_right) {
			switch (op) {
				case ValueOpcode::ULessThan32: opcode = ValueOpcode::UGreaterThan32; break;
				case ValueOpcode::ULessThanEqual32: opcode = ValueOpcode::UGreaterThanEqual32; break;
				case ValueOpcode::UGreaterThan32: opcode = ValueOpcode::ULessThan32; break;
				case ValueOpcode::UGreaterThanEqual32: opcode = ValueOpcode::ULessThanEqual32; break;
				default: break;
			}
		}
		if (!positive) {
			switch (opcode) {
				case ValueOpcode::ULessThan32: opcode = ValueOpcode::UGreaterThanEqual32; break;
				case ValueOpcode::ULessThanEqual32: opcode = ValueOpcode::UGreaterThan32; break;
				case ValueOpcode::UGreaterThan32: opcode = ValueOpcode::ULessThanEqual32; break;
				case ValueOpcode::UGreaterThanEqual32: opcode = ValueOpcode::ULessThan32; break;
				case ValueOpcode::INotEqual32: opcode = ValueOpcode::IEqual32; break;
				default: return UINT32_MAX;
			}
		}
		switch (opcode) {
			case ValueOpcode::ULessThan32: return limit == 0u ? 0u : limit - 1u;
			case ValueOpcode::ULessThanEqual32:
			case ValueOpcode::IEqual32: return limit;
			default: return UINT32_MAX;
		}
	}

	// GuardKeyMaximum over every unique-predecessor edge that leads to block after key's definition.
	uint32_t EntryKeyMaximum(Value key, const Block* block) const {
		uint32_t    result     = UINT32_MAX;
		const auto* definition = key.Resolve().TryInstruction();
		for (size_t depth = 0; block != nullptr && depth < m_program.blocks.size(); ++depth) {
			if (block->ImmPredecessors().size() != 1u ||
			    (definition != nullptr && definition->Parent() == block)) {
				break;
			}
			const auto* previous = block->ImmPredecessors()[0];
			Value       condition;
			bool        positive = false;
			if (ConditionalEdge(previous, block, condition, positive)) {
				result = std::min(result, GuardKeyMaximum(condition, key, positive));
			}
			block = previous;
		}
		return result;
	}

	uint32_t KeyMaximum(Value value, Value guard, uint32_t depth = 0) const {
		value              = value.Resolve();
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) return immediate;
		const auto* inst = value.TryInstruction();
		if (depth > 24u || inst == nullptr) return UINT32_MAX;
		uint32_t   result  = std::min(PossibleU32Bits(value), GuardKeyMaximum(guard, value));
		const auto maximum = [&](uint32_t arg, Value context) {
			return KeyMaximum(inst->Arg(arg), context, depth + 1u);
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::ReadFirstLane:
			case ValueOpcode::ReadLane: {
				// The value is some lane's: a context-free bound on every lane applies, and a
				// waterfall guard (local == key) bounds it by the executing lanes' local value.
				result           = std::min(result, maximum(0, {}));
				const auto local = EqualLocalKey(guard, value);
				if (!local.IsEmpty()) result = std::min(result, KeyMaximum(local, guard, depth + 1u));
				break;
			}
			case ValueOpcode::UMin32:
			case ValueOpcode::BitwiseAnd32:
				result = std::min(result, std::min(maximum(0, guard), maximum(1, guard)));
				break;
			case ValueOpcode::SMin32: {
				const auto left = maximum(0, guard), right = maximum(1, guard);
				// Signed and unsigned minimums agree on nonnegative operands.
				if (std::max(left, right) <= static_cast<uint32_t>(INT32_MAX)) {
					result = std::min(result, std::min(left, right));
					break;
				}
				// A signed clamp to a constant (20bb268a4): a negative key finds no enumerated
				// row and reads the first candidate.
				for (uint32_t arg = 0; arg < 2u; ++arg) {
					uint32_t row = 0;
					if (ImmediateU32(inst->Arg(arg), row) && row <= static_cast<uint32_t>(INT32_MAX)) {
						result = std::min(result, row);
					}
				}
				break;
			}
			case ValueOpcode::ShiftRightLogical32:
				if (ImmediateU32(inst->Arg(1), immediate)) {
					result = std::min(result, maximum(0, guard) >> (immediate & 31u));
				}
				break;
			case ValueOpcode::BitFieldUExtract:
				if (inst->NumArgs() == 3u && ImmediateU32(inst->Arg(2), immediate) && immediate < 32u) {
					result = std::min(result, (1u << immediate) - 1u);
				}
				break;
			case ValueOpcode::IAdd32: {
				const uint64_t sum = uint64_t {maximum(0, guard)} + maximum(1, guard);
				if (sum <= UINT32_MAX) result = std::min(result, static_cast<uint32_t>(sum));
				break;
			}
			case ValueOpcode::SelectU32:
				if (inst->NumArgs() == 3u) {
					result = std::min(result, !guard.IsEmpty() && Implies(guard, inst->Arg(0))
					                              ? maximum(1, guard)
					                              : std::max(maximum(1, guard), maximum(2, guard)));
				}
				break;
			case ValueOpcode::ConvertS32F32:
			case ValueOpcode::ConvertU32F32: {
				// The conversion truncates toward zero (dfcf1c0b0).
				const auto [low, high] = KeyFloatRange(inst->Arg(0));
				if (low > -1.0f && high < 0x1p31f) {
					result = std::min(result, static_cast<uint32_t>(high));
				}
				break;
			}
			default: break;
		}
		return result;
	}

	// The EXEC predicate of an image access through use, else the branch predicate of its block.
	Value KeyUseGuard(const Use& use) const {
		const auto& access = *use.user;
		if (use.operand == 0u && access.NumArgs() != 0u &&
		    ImageOpcodeInfoOf(access.GetOpcode()).access != ImageAccess::None) {
			const auto mask = access.Arg(access.NumArgs() - 1u);
			if (mask.GetType() == Type::U1 && mask.Resolve() != Value(true)) return mask;
		}
		return PositiveUseGuard(access.Parent());
	}

	uint32_t TableKeyMaximum(Value key, const Inst& handle) const {
		uint32_t result = 0;
		bool     any    = false;
		for (const auto& use: handle.Uses()) {
			if (ImageOpcodeInfoOf(use.user->GetOpcode()).access == ImageAccess::None) {
				return UINT32_MAX;
			}
			result = std::max(result, std::min(EntryKeyMaximum(key, use.user->Parent()),
			                                   KeyMaximum(key, KeyUseGuard(use))));
			any = true;
		}
		return any ? result : UINT32_MAX;
	}

	static bool ZeroBasedInduction(const Inst* phi) {
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U32 ||
		    phi->NumArgs() != 2u) {
			return false;
		}
		const Value key(const_cast<Inst*>(phi));
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto  zero      = phi->Arg(initial).Resolve();
			const auto* step      = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			uint32_t    increment = 0;
			if (zero.IsImmediate() && zero.GetType() == Type::U32 && zero.U32() == 0u &&
			    step != nullptr && step->GetOpcode() == ValueOpcode::IAdd32 &&
			    ((step->Arg(0).Resolve() == key && ImmediateU32(step->Arg(1), increment) &&
			      increment == 1u) ||
			     (step->Arg(1).Resolve() == key && ImmediateU32(step->Arg(0), increment) &&
			      increment == 1u))) {
				return true;
			}
		}
		return false;
	}

	// b2053c286: a zero-based loop key that only per-lane data bounds takes the runtime count that
	// every uniformly bounded zero-based loop of the program shares, assuming each lane's count
	// stays within it.
	Value ProgramLoopCount(Value key) const {
		key = key.Resolve();
		if (m_shader_writes || !ZeroBasedInduction(key.TryInstruction())) return {};
		Value count;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				if ((inst.GetOpcode() != ValueOpcode::SLessThan32 &&
				     inst.GetOpcode() != ValueOpcode::ULessThan32) ||
				    inst.NumArgs() != 2u) {
					continue;
				}
				const auto induction = inst.Arg(0).Resolve();
				if (induction == key || !ZeroBasedInduction(induction.TryInstruction()) ||
				    !ValidateRuntimeValue(m_program, inst.Arg(1))) {
					continue;
				}
				if (!count.IsEmpty() && !EquivalentValue(m_program, count, inst.Arg(1))) return {};
				count = inst.Arg(1);
			}
		}
		if (!count.IsEmpty()) {
			std::printf("Warning: shader 0x%016" PRIx64 " bounds a per-lane table loop by its uniform "
			            "loops' count (KYTY_TABLE_KEY_BOUNDS)\n",
			            m_program.shader_hash);
		}
		return count;
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

	// KYTY_INDIRECT_FORMAT_STORES: BUFFER_STORE_FORMAT_X..XYZW of 32-bit data.
	static bool IndirectFormattedStore(const MemoryInfo& memory, ValueOpcode op) {
		return memory.formatted && !memory.typed && memory.data_bits == 32u &&
		       (op == ValueOpcode::StoreBufferU32 || op == ValueOpcode::StoreBufferU32x2 ||
		        op == ValueOpcode::StoreBufferU32x3 || op == ValueOpcode::StoreBufferU32x4);
	}

	// Every image instruction on the handle takes a 128-bit T# (R128=1).
	bool AllUsersR128(const Inst& handle) const {
		bool any = false;
		for (const auto& use: handle.Uses()) {
			if (ImageOpcodeInfoOf(use.user->GetOpcode()).access == ImageAccess::None ||
			    use.operand != 0u) {
				return false;
			}
			const auto index = use.user->Flags<MemoryFlags>().index;
			if (index >= m_program.memory_info.size() || !m_program.memory_info[index].image_r128) {
				return false;
			}
			any = true;
		}
		return any;
	}

	// ---------------------------------------------------------------------------------------------
	// KYTY_SELECTOR_* (CodegenOptions::selector_keys): GPU-selected key proofs adapted from upstream
	// KytyPS5's selector framework (49ebbfd62, 0e9c77b0a, 145578ac0, f297c0bfe, ffb6ad6ca, 914a3bb4b,
	// 99f4eba22) onto this tracker's plans. Nothing below runs with every bit off.

	bool SelectorKeys(uint8_t bit) const { return (GetCodegenOptions().selector_keys & bit) != 0u; }

	// Upstream's wider ImpliesNonzero / Implies, used only by the selector proofs (the shared helpers
	// keep their narrower forms so plans with every switch off do not change): 0 < value proves
	// value != 0, a guard proves value != 0 that way, and a disjunction implies what both arms do.
	bool SelectorNonzero(Value guard, Value value, uint32_t depth = 0) const {
		if (depth > 16u) return false;
		guard            = SimplifyGuard(guard);
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return SelectorNonzero(inst->Arg(0), value, depth + 1u) ||
			       SelectorNonzero(inst->Arg(1), value, depth + 1u);
		}
		if (inst->NumArgs() != 2u) return false;
		if (inst->GetOpcode() == ValueOpcode::ULessThan32) {
			return inst->Arg(0).Resolve() == Value(0u) && EquivalentValue(m_program, inst->Arg(1), value);
		}
		if (inst->GetOpcode() == ValueOpcode::UGreaterThan32) {
			return inst->Arg(1).Resolve() == Value(0u) && EquivalentValue(m_program, inst->Arg(0), value);
		}
		return ImpliesNonzero(guard, value);
	}

	bool SelectorImplies(Value guard, Value required, uint32_t depth = 0) const {
		if (guard.IsEmpty() || depth > 16u) return false;
		guard    = SimplifyGuard(guard);
		required = SimplifyGuard(required);
		if (required == Value(true) || Implies(guard, required)) return true;
		const auto* test = required.TryInstruction();
		if (test != nullptr && test->GetOpcode() == ValueOpcode::INotEqual32 && test->NumArgs() == 2u) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				if (test->Arg(arg).Resolve() == Value(0u) && SelectorNonzero(guard, test->Arg(arg ^ 1u))) {
					return true;
				}
			}
		}
		const auto* inst = guard.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return SelectorImplies(inst->Arg(0), required, depth + 1u) ||
			       SelectorImplies(inst->Arg(1), required, depth + 1u);
		}
		if (inst->GetOpcode() == ValueOpcode::LogicalOr) {
			return SelectorImplies(inst->Arg(0), required, depth + 1u) &&
			       SelectorImplies(inst->Arg(1), required, depth + 1u);
		}
		return false;
	}

	enum class LaneQuantifier { Any, All };
	struct EdgePredicate {
		Value          condition;
		bool           positive = true;
		LaneQuantifier lanes    = LaneQuantifier::All;
	};

	// Branch conditions on lane masks are AnyLane(mask) here (VCCZ/VCCNZ/EXECZ/EXECNZ); SCC branches
	// are uniform. A taken AnyLane edge proves the mask on some lane, a not-taken one on none.
	std::optional<EdgePredicate> EdgeOf(const Block* from, const Block* to) const {
		Value condition;
		bool  positive = false;
		if (from == nullptr || to == nullptr || !ConditionalEdge(from, to, condition, positive)) return {};
		EdgePredicate edge {condition, positive, LaneQuantifier::All};
		bool          reduced = false;
		while (const auto* inst = edge.condition.Resolve().TryInstruction()) {
			if (inst->GetOpcode() == ValueOpcode::LogicalNot) {
				edge.positive = !edge.positive;
			} else if (inst->GetOpcode() == ValueOpcode::AnyLane && !reduced) {
				reduced    = true;
				edge.lanes = edge.positive ? LaneQuantifier::Any : LaneQuantifier::All;
			} else {
				break;
			}
			edge.condition = inst->Arg(0);
		}
		edge.condition = edge.condition.Resolve();
		return edge;
	}

	// Every path into block passes an edge accepts takes before it reaches a block stops names.
	template <typename Stops, typename Accepts>
	bool GuardedOnEntry(const Block* block, const Stops& stops, const Accepts& accepts) const {
		std::vector<const Block*> pending {block};
		for (size_t i = 0; i < pending.size(); ++i) {
			const auto* current = pending[i];
			if (current == nullptr || stops(current) || current->ImmPredecessors().empty()) return false;
			for (const auto* previous: current->ImmPredecessors()) {
				const auto edge = EdgeOf(previous, current);
				if (edge && accepts(*edge)) continue;
				if (std::ranges::find(pending, previous) == pending.end()) pending.push_back(previous);
			}
		}
		return true;
	}

	Value ActiveValue(Value value, Value guard) const {
		value = value.Resolve();
		while (const auto* select = value.TryInstruction()) {
			if (select->GetOpcode() != ValueOpcode::SelectU32 || guard.IsEmpty() ||
			    !SelectorImplies(guard, select->Arg(0))) {
				break;
			}
			value = select->Arg(1).Resolve();
		}
		return value;
	}

	bool Reaches(const Block* from, const Block* to, const Block* stop = nullptr) const {
		std::vector<const Block*> pending {to};
		for (size_t index = 0; index < pending.size(); ++index) {
			if (pending[index] == stop) continue;
			if (pending[index] == from) return true;
			for (const auto* previous: pending[index]->ImmPredecessors()) {
				if (std::ranges::find(pending, previous) == pending.end()) pending.push_back(previous);
			}
		}
		return false;
	}

	bool InCycle(const Block* block) const {
		return block != nullptr && std::ranges::any_of(block->ImmPredecessors(), [&](const Block* previous) {
			       return Reaches(block, previous);
		       });
	}

	static bool BeforeInBlock(const Inst& before, const Inst& after) {
		if (before.Parent() != after.Parent()) return false;
		for (const auto& inst: *before.Parent()) {
			if (&inst == &before) return true;
			if (&inst == &after) return false;
		}
		return false;
	}

	// initial runs before use on every path; a witness (a lane-local zero proof, ffb6ad6ca) rules
	// out the all-lane bypass edges it excludes.
	bool Dominates(const Block* initial, const Block* use, Value witness = {}) const {
		const auto* negated = SimplifyGuard(witness).TryInstruction();
		const bool  nonzero = negated != nullptr && negated->GetOpcode() == ValueOpcode::LogicalNot;
		std::vector<const Block*> pending {use};
		for (size_t index = 0; index < pending.size(); ++index) {
			const auto* current = pending[index];
			if (current == initial) continue;
			if (current->ImmPredecessors().empty()) return false;
			for (const auto* previous: current->ImmPredecessors()) {
				if (!witness.IsEmpty()) {
					const auto edge = EdgeOf(previous, current);
					if (edge && edge->lanes == LaneQuantifier::All &&
					    (edge->positive ? nonzero && SelectorImplies(edge->condition, negated->Arg(0))
					                    : SelectorImplies(witness, edge->condition))) {
						continue;
					}
				}
				if (std::ranges::find(pending, previous) == pending.end()) pending.push_back(previous);
			}
		}
		return true;
	}

	bool AvailableBefore(const Inst& initial, const Inst& use, Value witness = {}) const {
		return initial.Parent() == use.Parent() ? BeforeInBlock(initial, use)
		                                       : Dominates(initial.Parent(), use.Parent(), witness);
	}

	// Some lane of mask is active wherever use runs.
	bool HasActiveLane(Value mask, const Block* use) const {
		mask = SimplifyGuard(mask);
		if (mask.IsImmediate()) return mask.GetType() == Type::U1 && mask.U1();
		const auto incoming_is_nonempty = [&](const Block* from, const Block* to, Value incoming,
		                                      const Block* header) {
			const auto accepts = [&](const EdgePredicate& edge) {
				return edge.positive && SelectorImplies(edge.condition, incoming);
			};
			const auto edge = EdgeOf(from, to);
			if (edge && accepts(*edge)) return true;
			const auto* definition = incoming.Resolve().TryInstruction();
			return GuardedOnEntry(
			    from,
			    [&](const Block* block) {
				    // A witness before this definition may belong to an earlier loop iteration.
				    return block == header || (definition != nullptr && block == definition->Parent());
			    },
			    accepts);
		};
		const auto* phi = mask.TryInstruction();
		if (phi != nullptr && phi->GetOpcode() == ValueOpcode::Phi && phi->GetType() == Type::U1 &&
		    phi->NumArgs() != 0u) {
			for (size_t arm = 0; arm < phi->NumArgs(); ++arm) {
				if (!incoming_is_nonempty(phi->PhiBlock(arm), phi->Parent(), phi->Arg(arm), phi->Parent())) {
					return false;
				}
			}
			return true;
		}
		return use != nullptr && use->ImmPredecessors().size() == 1u &&
		       incoming_is_nonempty(use->ImmPredecessors()[0], use, mask, nullptr);
	}

	struct Induction {
		const Inst* phi     = nullptr;
		const Inst* step    = nullptr;
		uint32_t    initial = 0;
	};

	template <typename AcceptsStep>
	std::optional<Induction> MatchInduction(Value key, const AcceptsStep& accepts_step) const {
		key              = key.Resolve();
		const auto* phi  = key.TryInstruction();
		if (phi == nullptr || phi->GetOpcode() != ValueOpcode::Phi || phi->GetType() != Type::U32 ||
		    phi->NumArgs() != 2u || phi->NumPhiBlocks() != 2u || phi->Parent() == nullptr) {
			return {};
		}
		for (uint32_t initial = 0; initial < 2u; ++initial) {
			const auto  initial_value = phi->Arg(initial).Resolve();
			const auto* step          = phi->Arg(initial ^ 1u).Resolve().TryInstruction();
			if (!initial_value.IsImmediate() || initial_value.GetType() != Type::U32 || step == nullptr ||
			    step->GetOpcode() != ValueOpcode::IAdd32 || step->Parent() == nullptr) {
				continue;
			}
			uint32_t increment = 0;
			if ((step->Arg(0).Resolve() == key && ImmediateU32(step->Arg(1), increment) &&
			     accepts_step(initial_value.U32(), increment)) ||
			    (step->Arg(1).Resolve() == key && ImmediateU32(step->Arg(0), increment) &&
			     accepts_step(initial_value.U32(), increment))) {
				return Induction {phi, step, initial};
			}
		}
		return {};
	}

	// The comparison of an induction that bounds it at use (or at its increment), with the loop
	// proven unable to continue past the bound (LoopBoundProof).
	template <typename AcceptsStep, typename AcceptsBound>
	const Inst* BoundedInduction(Value key, const Block* use, const AcceptsStep& accepts_step,
	                             const AcceptsBound& accepts_bound,
	                             LaneQuantifier lanes = LaneQuantifier::Any) const {
		const auto induction = MatchInduction(key, accepts_step);
		if (!induction) return nullptr;
		const auto* phi             = induction->phi;
		const auto* increment_block = induction->step->Parent();
		// Structurization can forward the native increment through a synthetic continue block.
		const Block* incoming = phi->PhiBlock(induction->initial ^ 1u);
		const Block* next     = phi->Parent();
		for (size_t edges = 0; incoming != increment_block && edges < m_program.blocks.size(); ++edges) {
			if (incoming == nullptr || incoming->ImmSuccessors().size() != 1u ||
			    incoming->ImmSuccessors()[0] != next || incoming->ImmPredecessors().size() != 1u) {
				break;
			}
			next     = incoming;
			incoming = incoming->ImmPredecessors()[0];
		}
		if (incoming != increment_block) return nullptr;
		const auto stops = [&](const Block* block) { return block == phi->Parent(); };
		for (const auto& use_of_key: phi->Uses()) {
			const auto* compare = use_of_key.user;
			if (use_of_key.operand != 0u || !accepts_bound(*compare, induction->initial)) continue;
			if (!GuardedOnEntry(use != nullptr ? use : increment_block, stops, [&](const EdgePredicate& edge) {
				    return edge.positive && (lanes == LaneQuantifier::Any || edge.lanes == lanes) &&
				           SelectorImplies(edge.condition, Value(use_of_key.user));
			    })) {
				continue;
			}
			LoopBoundProof proof(m_program, *phi, *compare);
			// The image bound and the increment guard are separate obligations.
			if (GuardedOnEntry(increment_block, stops, [&](const EdgePredicate& edge) {
				    return proof.Excludes(edge.condition, edge.positive);
			    })) {
				return compare;
			}
		}
		return nullptr;
	}

	template <typename AcceptsBound>
	const Inst* ZeroBasedLoop(Value key, const Block* use, const AcceptsBound& accepts_bound) const {
		return BoundedInduction(
		    key, use, [](uint32_t initial, uint32_t step) { return initial == 0u && step == 1u; },
		    [&](const Inst& compare, uint32_t initial) {
			    return (compare.GetOpcode() == ValueOpcode::SLessThan32 ||
			            compare.GetOpcode() == ValueOpcode::ULessThan32) &&
			           accepts_bound(compare, initial);
		    });
	}

	// i = maximum; i >= 0; --i (49ebbfd62).
	const Inst* DescendingLoop(Value key, const Block* use, uint32_t& maximum) const {
		return BoundedInduction(
		    key, use,
		    [&](uint32_t initial, uint32_t step) {
			    maximum = initial;
			    return initial <= INT32_MAX && step == UINT32_MAX;
		    },
		    [](const Inst& compare, uint32_t) {
			    return compare.GetOpcode() == ValueOpcode::SGreaterThanEqual32 &&
			           compare.Arg(1).Resolve() == Value(0u);
		    });
	}

	using SelectorBounds = std::span<const std::pair<const Inst*, uint32_t>>;

	bool SelectorPredicateFalse(Value value, SelectorBounds bounds, uint32_t depth = 0,
	                            Value guard = {}) const {
		if (depth > 32u) return false;
		value = value.Resolve();
		if (value.IsImmediate()) return value.GetType() == Type::U1 && !value.U1();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return false;
		if (inst->GetOpcode() == ValueOpcode::LogicalAnd) {
			return SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u, guard) ||
			       SelectorPredicateFalse(inst->Arg(1), bounds, depth + 1u, guard);
		}
		if (inst->GetOpcode() == ValueOpcode::IEqual32) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				uint32_t target = 0;
				if (ImmediateU32(inst->Arg(arg), target) &&
				    SelectorMaximum(inst->Arg(arg ^ 1u), bounds, depth + 1u, guard) < target) {
					return true;
				}
			}
		}
		return false;
	}

	// The largest value `value` holds on the lanes where guard holds (upstream SelectorMaximum);
	// bounds give the maxima of descending-loop phis.
	uint32_t SelectorMaximum(Value value, SelectorBounds bounds = {}, uint32_t depth = 0,
	                         Value guard = {}) const {
		if (depth > 32u) return UINT32_MAX;
		value = value.Resolve();
		if (value.IsImmediate() && value.GetType() == Type::U32) return value.U32();
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return UINT32_MAX;
		for (const auto& [phi, maximum]: bounds) {
			if (inst == phi) return maximum;
		}
		const auto maximum = [&](uint32_t arg) {
			return SelectorMaximum(inst->Arg(arg), bounds, depth + 1u, guard);
		};
		uint32_t result = PossibleU32Bits(value);
		switch (inst->GetOpcode()) {
			case ValueOpcode::ReadFirstLane: {
				// Some lane of the mask runs: its value is one of the mask lanes' values.
				if (guard.IsEmpty() ? !HasActiveLane(inst->Arg(1), inst->Parent())
				                    : !SelectorImplies(guard, inst->Arg(1))) {
					return UINT32_MAX;
				}
				return std::min(result, SelectorMaximum(inst->Arg(0), bounds, depth + 1u, inst->Arg(1)));
			}
			case ValueOpcode::ReadLane: {
				// ReadLane ignores EXEC: only a waterfall guard bounds it by the executing lanes' value.
				const auto local = EqualLocalKey(guard, value);
				return local.IsEmpty() ? PossibleU32Bits(inst->Arg(0))
				                       : SelectorMaximum(local, bounds, depth + 1u, guard);
			}
			case ValueOpcode::Phi: {
				if (inst->NumArgs() == 0u || InCycle(inst->Parent())) return result;
				uint32_t arms = 0u;
				for (uint32_t arm = 0; arm < inst->NumArgs() && arms != UINT32_MAX; ++arm) {
					arms = std::max(arms, maximum(arm));
				}
				return std::min(result, arms);
			}
			case ValueOpcode::BitFieldUExtract: {
				uint32_t width = 0;
				if (inst->NumArgs() == 3u && ImmediateU32(inst->Arg(2), width) && width < 32u) {
					return std::min(result, (1u << width) - 1u);
				}
				break;
			}
			case ValueOpcode::FindILsb32:
			case ValueOpcode::FindUMsb32:
				return (!guard.IsEmpty() && SelectorNonzero(guard, inst->Arg(0))) ||
				               NonzeroOnEntry(inst->Arg(0), inst->Parent())
				           ? 31u
				           : UINT32_MAX;
			case ValueOpcode::UMin32:
			case ValueOpcode::BitwiseAnd32: return std::min({result, maximum(0), maximum(1)});
			case ValueOpcode::SMin32: {
				const auto left = maximum(0), right = maximum(1);
				if (std::max(left, right) <= static_cast<uint32_t>(INT32_MAX)) {
					return std::min({result, left, right});
				}
				break;
			}
			case ValueOpcode::ConvertS32F32: {
				const auto [low, high] = KeyFloatRange(inst->Arg(0));
				if (low >= 0.0f && high < 0x1p31f) return std::min(result, static_cast<uint32_t>(high));
				break;
			}
			case ValueOpcode::SelectU32:
				if (!guard.IsEmpty() && SelectorImplies(guard, inst->Arg(0))) return std::min(result, maximum(1));
				return std::min(result, SelectorPredicateFalse(inst->Arg(0), bounds, depth + 1u, guard)
				                            ? maximum(2)
				                            : std::max(maximum(1), maximum(2)));
			case ValueOpcode::ISub32: {
				uint32_t left = 0;
				if (ImmediateU32(inst->Arg(0), left) && maximum(1) <= left) return std::min(result, left);
				break;
			}
			case ValueOpcode::IAdd32: {
				const auto sum = uint64_t {maximum(0)} + maximum(1);
				if (sum <= UINT32_MAX) return std::min(result, static_cast<uint32_t>(sum));
				break;
			}
			case ValueOpcode::IMul32: {
				const auto result = uint64_t {maximum(0)} * maximum(1);
				if (result <= UINT32_MAX) return static_cast<uint32_t>(result);
				break;
			}
			case ValueOpcode::ShiftRightLogical32:
			case ValueOpcode::ShiftLeftLogical32: {
				uint32_t shift = 0;
				if (!ImmediateU32(inst->Arg(1), shift) || shift >= 32u) break;
				if (inst->GetOpcode() == ValueOpcode::ShiftRightLogical32) {
					return std::min(result, maximum(0) >> shift);
				}
				const auto shifted = uint64_t {maximum(0)} << shift;
				if (shifted <= UINT32_MAX) return std::min(result, static_cast<uint32_t>(shifted));
				break;
			}
			default: break;
		}
		return result;
	}

	// f297c0bfe: the immediate an all-lane edge proves value differs from.
	std::optional<uint32_t> ExcludedBy(const EdgePredicate& edge, Value value) const {
		if (edge.lanes != LaneQuantifier::All) return {};
		const auto* test = edge.condition.TryInstruction();
		if (test != nullptr && test->NumArgs() == 2u &&
		    ((test->GetOpcode() == ValueOpcode::INotEqual32 && edge.positive) ||
		     (test->GetOpcode() == ValueOpcode::IEqual32 && !edge.positive))) {
			for (uint32_t arg = 0; arg < 2u; ++arg) {
				uint32_t immediate = 0;
				if (ImmediateU32(test->Arg(arg), immediate) &&
				    EquivalentValue(m_program, test->Arg(arg ^ 1u), value)) {
					return immediate;
				}
			}
		}
		return {};
	}

	// f297c0bfe: the key every path from its definition to block rules out, if any.
	std::optional<uint32_t> ExcludedKeyOnEntry(Value key, const Block* block) const {
		const auto* definition = key.Resolve().TryInstruction();
		if (definition == nullptr || definition->Parent() == nullptr) return {};
		std::optional<uint32_t> excluded;
		const bool              guarded = GuardedOnEntry(
            block, [&](const Block* current) { return current == definition->Parent(); },
            [&](const EdgePredicate& edge) {
                const auto immediate = ExcludedBy(edge, key);
                if (!immediate || (excluded && *excluded != *immediate)) return false;
                excluded = immediate;
                return true;
            });
		return guarded ? excluded : std::nullopt;
	}

	// 145578ac0 / 914a3bb4b / 99f4eba22: mask keeps only bits of the words reads load (each a
	// four-word buffer load, or one scalar load). Each equation is an inductive subset obligation:
	// phis require every incoming value, AND either operand. Grounding rejects cycles without an
	// initial value, then the greatest fixed point checks every initial and carried edge. Adapted:
	// mask ^ (1 << FindILsb(mask)) (ClearsFirstSetBit) keeps a subset of mask too.
	bool BitsetMaskSubset(Value mask, Value active, Value word, std::span<const Inst* const> reads) const {
		struct State {
			Value    value;
			Value    guard;
			uint32_t word      = UINT32_MAX;
			uint32_t component = UINT32_MAX; // UINT32_MAX: a U1 implication of the load guard.
			bool     operator==(const State&) const = default;
		};
		struct Equation {
			bool                all = true;
			std::vector<size_t> terms;
		};
		if (reads.empty()) return false;
		const bool scalar     = reads[0]->GetOpcode() == ValueOpcode::ReadConstBuffer;
		const auto load_guard = scalar ? Value(true) : reads[0]->Arg(4).Resolve();
		const auto words      = static_cast<uint32_t>(reads.size()) * (scalar ? 1u : 4u);
		std::vector<State>    states(2);
		std::vector<Equation> equations(2);
		const auto            add = [&](State state) {
            state.value      = state.value.Resolve();
            const auto found = std::find(states.begin() + 2, states.end(), state);
            if (found != states.end()) return static_cast<size_t>(found - states.begin());
            states.push_back(state);
            equations.emplace_back();
            return states.size() - 1u;
		};
		std::vector<size_t> roots {add({.value = active})};
		for (uint32_t component = 0; component < words; ++component) {
			roots.push_back(add({mask, active, component, component}));
		}
		for (size_t index = 2; index < states.size() && states.size() < 4096u; ++index) {
			const auto  state = states[index];
			const auto* inst  = state.value.TryInstruction();
			Equation    equation;
			const auto  term = [&](Value value, Value guard, uint32_t selected) {
                return add({value, guard, selected, state.component});
			};
			if (state.component == UINT32_MAX) {
				if (SelectorImplies(state.value, load_guard)) {
					equation.terms = {1u};
				} else if (inst != nullptr && (inst->GetOpcode() == ValueOpcode::LogicalAnd ||
				                               inst->GetOpcode() == ValueOpcode::Phi)) {
					equation.all = inst->GetOpcode() == ValueOpcode::Phi;
					for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
						equation.terms.push_back(add({.value = inst->Arg(arg)}));
					}
				}
			} else if (state.value == Value(0u)) {
				equation.terms = {1u};
			} else if (inst != nullptr) {
				const auto constant = [&](const auto& self, Value value) -> std::optional<uint32_t> {
					uint32_t literal = 0;
					if (ImmediateU32(value, literal)) return literal;
					if (state.word != UINT32_MAX && !word.IsEmpty() && EquivalentValue(m_program, value, word)) {
						return state.word;
					}
					const auto* operation = value.Resolve().TryInstruction();
					if (operation == nullptr || operation->GetOpcode() != ValueOpcode::BitwiseAnd32) return {};
					const auto left  = self(self, operation->Arg(0));
					const auto right = self(self, operation->Arg(1));
					return left && right ? std::optional {*left & *right} : std::nullopt;
				};
				const auto condition = [&](const auto& self, Value value) -> std::optional<bool> {
					value = value.Resolve();
					if (value.IsImmediate() && value.GetType() == Type::U1) return value.U1();
					if (SelectorImplies(state.guard, value) || SelectorImplies(load_guard, value)) return true;
					const auto* test = value.TryInstruction();
					if (test == nullptr) return {};
					if (test->GetOpcode() == ValueOpcode::LogicalAnd) {
						const auto left  = self(self, test->Arg(0));
						const auto right = self(self, test->Arg(1));
						if ((left && !*left) || (right && !*right)) return false;
						if (left && right) return *left && *right;
					} else if (test->GetOpcode() == ValueOpcode::IEqual32) {
						const auto left  = constant(constant, test->Arg(0));
						const auto right = constant(constant, test->Arg(1));
						if (left && right) return *left == *right;
					}
					return {};
				};
				switch (inst->GetOpcode()) {
					case ValueOpcode::CompositeExtractU32x4: {
						uint32_t component = 0;
						if (!scalar && inst->Arg(0).Resolve().TryInstruction() == reads[state.component / 4u] &&
						    ImmediateU32(inst->Arg(1), component) && component == state.component % 4u) {
							equation.terms = {1u};
						}
						break;
					}
					case ValueOpcode::ReadConstBuffer:
						if (scalar && inst == reads[state.component]) equation.terms = {1u};
						break;
					case ValueOpcode::Phi:
						// The predecessor may be an earlier outer iteration: establish this register's
						// invariant without the current word or EXEC.
						for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
							equation.terms.push_back(term(inst->Arg(arg), load_guard, UINT32_MAX));
						}
						break;
					case ValueOpcode::BitwiseAnd32:
						equation.all   = false;
						equation.terms = {term(inst->Arg(0), state.guard, state.word),
						                  term(inst->Arg(1), state.guard, state.word)};
						break;
					case ValueOpcode::BitwiseXor32:
						for (uint32_t side = 0; side < 2u; ++side) {
							if (ClearsFirstSetBit(state.value, inst->Arg(side))) {
								equation.terms = {term(inst->Arg(side), state.guard, state.word)};
							}
						}
						break;
					case ValueOpcode::SelectU32: {
						const auto selected = condition(condition, inst->Arg(0));
						if (!selected || *selected) {
							auto       selected_word = state.word;
							const auto equality      = word.IsEmpty() ? Value {} : EqualLocalValue(inst->Arg(0), [&](Value key) {
                                key                = key.Resolve();
                                const auto* masked = key.TryInstruction();
                                uint32_t    bound  = 0;
                                if (masked != nullptr && masked->GetOpcode() == ValueOpcode::BitwiseAnd32 &&
                                    DescendingLoop(word, masked->Parent(), bound) && bound == words - 1u) {
                                    for (uint32_t side = 0; side < 2u; ++side) {
                                        uint32_t bits = 0;
                                        if (ImmediateU32(masked->Arg(side), bits) && (bits & bound) == bound) {
                                            key = masked->Arg(side ^ 1u).Resolve();
                                        }
                                    }
                                }
                                return EquivalentValue(m_program, key, word);
                            });
							uint32_t literal = 0;
							if (ImmediateU32(equality, literal) && literal < words) selected_word = literal;
							equation.terms.push_back(term(inst->Arg(1), inst->Arg(0), selected_word));
						}
						if (!selected || !*selected) {
							equation.terms.push_back(term(inst->Arg(2), state.guard, state.word));
						}
						break;
					}
					default: break;
				}
			}
			if (equation.terms.empty()) equation.terms = {0u};
			equations[index] = std::move(equation);
		}
		if (states.size() >= 4096u) return false;
		std::vector<bool> valid(states.size(), false);
		valid[1] = true;
		bool changed = false;
		do {
			changed = false;
			for (size_t index = 2; index < equations.size(); ++index) {
				if (!valid[index] && std::ranges::any_of(equations[index].terms, [&](size_t t) { return valid[t]; })) {
					valid[index] = changed = true;
				}
			}
		} while (changed);
		do {
			changed = false;
			for (size_t index = 2; index < equations.size(); ++index) {
				const auto& equation = equations[index];
				const auto  proven   = [&](size_t t) { return static_cast<bool>(valid[t]); };
				if (valid[index] && !(equation.all ? std::ranges::all_of(equation.terms, proven)
				                                   : std::ranges::any_of(equation.terms, proven))) {
					valid[index] = false;
					changed      = true;
				}
			}
		} while (changed);
		return std::ranges::all_of(roots, [&](size_t root) { return static_cast<bool>(valid[root]); });
	}

	// 99f4eba22: the mask words whose set bits are every record a bit scan of index selects: the
	// scanned value keeps only bits of one scalar load, which reads a word its address can reach.
	std::optional<DescriptorSource::IndirectDescriptor::SelectorRead::MaskRanks> BitScanRanks(Value index,
	                                                                                        uint32_t pc) {
		const auto* scan = index.Resolve().TryInstruction();
		if (scan == nullptr || scan->GetOpcode() != ValueOpcode::FindILsb32) return {};
		const auto* masked = scan->Arg(0).Resolve().TryInstruction();
		if (masked == nullptr) return {};
		std::vector<const Inst*> values {masked};
		const Inst*              load = nullptr;
		for (size_t i = 0; i < values.size() && load == nullptr && values.size() < 256u; ++i) {
			const auto* value = values[i];
			if (value->GetOpcode() == ValueOpcode::ReadConstBuffer) load = value;
			if (value->GetOpcode() != ValueOpcode::Phi && value->GetOpcode() != ValueOpcode::BitwiseAnd32 &&
			    value->GetOpcode() != ValueOpcode::BitwiseXor32) {
				continue;
			}
			for (size_t arg = 0; arg < value->NumArgs(); ++arg) {
				const auto* next = value->Arg(arg).Resolve().TryInstruction();
				if (next != nullptr && std::ranges::find(values, next) == values.end()) values.push_back(next);
			}
		}
		uint32_t    memory_index = 0;
		const auto* memory       = load != nullptr ? ScalarReadMemory(*load, memory_index) : nullptr;
		const std::array<const Inst*, 1> reads {load};
		if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer || memory->offset > INT32_MAX ||
		    !MemoryIndexBelongsTo(memory_index, *load) ||
		    !BitsetMaskSubset(scan->Arg(0), Value(true), Value {}, reads)) {
			return {};
		}
		auto        address = load->Arg(1);
		auto        active  = Value(true);
		const auto* lane    = address.Resolve().TryInstruction();
		if (lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane) {
			if (!HasActiveLane(lane->Arg(1), lane->Parent())) return {};
			active  = lane->Arg(1);
			address = lane->Arg(0);
		}
		// A left shift keeps the address a multiple of its scale; other addresses read any word.
		const auto* shift = ActiveValue(address, active).TryInstruction();
		uint32_t    bits  = 0;
		if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
		    !ImmediateU32(shift->Arg(1), bits) || bits >= 32u) {
			bits = 0;
		}
		const auto*      mask_handle = load->Arg(0).Resolve().TryInstruction();
		DescriptorSource source;
		if (mask_handle == nullptr || mask_handle->GetOpcode() != ValueOpcode::GetBufferResource ||
		    !MakeRuntimeTableSource(*mask_handle, pc, source)) {
			return {};
		}
		return DescriptorSource::IndirectDescriptor::SelectorRead::MaskRanks {
		    .source = InternSource(source), .stride = std::max(1u << bits, 4u), .offset = memory->offset & ~3u};
	}

	// 145578ac0 / 914a3bb4b: rank = (word << 5) + (31 - (31 - FindUMsb(bits))) in a descending loop
	// over four or eight bitmap words of one record (two four-word loads for eight).
	bool MatchBitsetSelector(Value rank, Value active, SelectorBounds bounds, uint32_t words, uint32_t pc,
	                         DescriptorSource::IndirectDescriptor& indirect, DescriptorSource& material_source) {
		const auto selected_value = [&](Value value) {
			for (size_t depth = 0; depth < 64u; ++depth) {
				value              = ActiveValue(value, active);
				const auto* select = value.TryInstruction();
				if (select == nullptr || select->GetOpcode() != ValueOpcode::SelectU32 ||
				    !SelectorPredicateFalse(select->Arg(0), bounds, 0u, active)) {
					return value;
				}
				value = select->Arg(2);
			}
			return value;
		};
		const auto* sum = selected_value(rank).TryInstruction();
		if (sum == nullptr || sum->GetOpcode() != ValueOpcode::IAdd32) return false;
		Value word;
		Value bit;
		for (uint32_t arg = 0; arg < 2u; ++arg) {
			const auto* shift = selected_value(sum->Arg(arg)).TryInstruction();
			if (shift == nullptr || shift->GetOpcode() != ValueOpcode::ShiftLeftLogical32 ||
			    shift->Arg(1).Resolve() != Value(5u)) {
				continue;
			}
			word = selected_value(shift->Arg(0));
			bit  = selected_value(sum->Arg(arg ^ 1u));
		}
		const auto* word_inst = word.TryInstruction();
		if (word_inst == nullptr || std::ranges::none_of(bounds, [&](const auto& bound) {
			    return bound.first == word_inst && bound.second == words - 1u;
		    })) {
			return false;
		}
		// FFBH's leading-zero count is inverted again to recover the set-bit rank.
		for (uint32_t inverse = 0; inverse < 2u; ++inverse) {
			const auto* subtract = bit.TryInstruction();
			if (subtract == nullptr || subtract->GetOpcode() != ValueOpcode::ISub32 ||
			    subtract->Arg(0).Resolve() != Value(31u)) {
				return false;
			}
			bit = selected_value(subtract->Arg(1));
		}
		const auto* scan = bit.TryInstruction();
		if (scan == nullptr || scan->GetOpcode() != ValueOpcode::FindUMsb32 ||
		    !SelectorNonzero(active, scan->Arg(0))) {
			return false;
		}
		const auto bitmap_load = [&](const Inst& read, uint32_t offset) {
			const auto index = read.Flags<MemoryFlags>().index;
			if (read.GetOpcode() != ValueOpcode::LoadBufferU32x4 || InCycle(read.Parent()) ||
			    index >= m_program.memory_info.size()) {
				return false;
			}
			const auto& memory = m_program.memory_info[index];
			return memory.kind == ResourceKind::Buffer && !memory.typed && !memory.formatted && memory.idxen &&
			       !memory.offen && memory.offset == offset && memory.data_bits == 32u &&
			       memory.data_dwords == 4u && read.Arg(2).Resolve() == Value(0u) &&
			       read.Arg(3).Resolve() == Value(0u);
		};
		for (const auto* block: m_program.blocks) {
			for (const auto& read: *block) {
				if (!bitmap_load(read, 0u)) continue;
				// An eight-word record continues in a second four-word load of the same buffer and EXEC.
				std::array<const Inst*, 2> reads {&read, nullptr};
				if (words == 8u) {
					for (const auto* other: m_program.blocks) {
						for (const auto& load: *other) {
							if (reads[1] == nullptr && bitmap_load(load, 16u) &&
							    EquivalentValue(m_program, load.Arg(0), read.Arg(0)) &&
							    EquivalentValue(m_program, load.Arg(4), read.Arg(4))) {
								reads[1] = &load;
							}
						}
					}
				}
				const auto       parts  = std::span<const Inst* const>(reads).first(words / 4u);
				const auto*      handle = read.Arg(0).Resolve().TryInstruction();
				DescriptorSource source;
				if (std::ranges::find(parts, nullptr) != parts.end() || handle == nullptr ||
				    handle->GetOpcode() != ValueOpcode::GetBufferResource ||
				    !BitsetMaskSubset(scan->Arg(0), active, word, parts) ||
				    !MakeRuntimeTableSource(*handle, pc, source)) {
					continue;
				}
				material_source = source;
				indirect.selector.emplace(DescriptorSource::IndirectDescriptor::SelectorRead {
				    .source = InternSource(source), .stride = words * 4u, .offset = 0u, .bitset = true});
				return true;
			}
		}
		return false;
	}

	// 49ebbfd62 (reversed ranges, the part 914a3bb4b builds on): key = base - V_READFIRSTLANE(rank)
	// with rank bounded through descending loops: keys base - maximum .. base.
	bool MatchReversedImageRange(Value key, uint32_t pc, DescriptorSource::IndirectDescriptor& indirect,
	                             DescriptorSource& material_source) {
		const auto* subtract = key.Resolve().TryInstruction();
		if (subtract == nullptr || subtract->GetOpcode() != ValueOpcode::ISub32 ||
		    !ValidateRuntimeValue(m_program, subtract->Arg(0), RuntimeValueType::Integer)) {
			return false;
		}
		const auto* lane = subtract->Arg(1).Resolve().TryInstruction();
		if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
		    !HasActiveLane(lane->Arg(1), lane->Parent())) {
			return false;
		}
		std::vector<std::pair<const Inst*, uint32_t>> bounds;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != ValueOpcode::Phi) continue;
				uint32_t maximum = 0;
				if (DescendingLoop(Value(const_cast<Inst*>(&inst)), lane->Parent(), maximum)) {
					bounds.emplace_back(&inst, maximum);
				}
			}
		}
		const auto maximum = SelectorMaximum(lane->Arg(0), bounds, 0u, lane->Arg(1));
		if (maximum >= MaxTableKeyProbes) return false;
		auto& first = m_program.value_storage.emplace_back(ValueOpcode::ISub32);
		first.SetArg(0, subtract->Arg(0));
		first.SetArg(1, Value(maximum));
		indirect.selector_first = Value(&first);
		indirect.key_count      = Value(maximum + 1u);
		if (maximum == 127u || maximum == 255u) {
			MatchBitsetSelector(lane->Arg(0), lane->Arg(1), bounds, (maximum + 1u) / 32u, pc, indirect,
			                    material_source);
		}
		return true;
	}

	// 0e9c77b0a: value = c0 + sum c_axis * local_id[axis] (+ c4 * one induction phi).
	std::optional<std::array<uint64_t, 5>> LocalIndex(Value value, Value active, Value& induction,
	                                                  bool allow_induction, uint32_t depth = 0) const {
		if (depth > 16u || m_workgroup == nullptr) return {};
		value              = ActiveValue(value, active);
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) return std::array<uint64_t, 5> {immediate, 0u, 0u, 0u, 0u};
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return {};
		const auto scale = [](std::array<uint64_t, 5> index, uint64_t factor) -> std::optional<std::array<uint64_t, 5>> {
			for (auto& coefficient: index) {
				coefficient *= factor;
				if (coefficient > UINT32_MAX) return {};
			}
			return index;
		};
		switch (inst->GetOpcode()) {
			case ValueOpcode::IAdd32: {
				const auto left  = LocalIndex(inst->Arg(0), active, induction, allow_induction, depth + 1u);
				const auto right = left ? LocalIndex(inst->Arg(1), active, induction, allow_induction, depth + 1u)
				                        : std::nullopt;
				if (!left || !right) return {};
				std::array<uint64_t, 5> sum {};
				for (size_t i = 0; i < sum.size(); ++i) {
					sum[i] = (*left)[i] + (*right)[i];
					if (sum[i] > UINT32_MAX) return {};
				}
				return sum;
			}
			case ValueOpcode::IMul32:
				for (uint32_t side = 0; side < 2u; ++side) {
					if (ImmediateU32(inst->Arg(side), immediate)) {
						const auto other = LocalIndex(inst->Arg(side ^ 1u), active, induction, allow_induction, depth + 1u);
						return other ? scale(*other, immediate) : std::nullopt;
					}
				}
				return {};
			case ValueOpcode::ShiftLeftLogical32:
				if (ImmediateU32(inst->Arg(1), immediate) && immediate < 32u) {
					const auto other = LocalIndex(inst->Arg(0), active, induction, allow_induction, depth + 1u);
					return other ? scale(*other, uint64_t {1} << immediate) : std::nullopt;
				}
				return {};
			default: break;
		}
		uint32_t    bits = 32u;
		const auto* leaf = inst;
		if (leaf->GetOpcode() == ValueOpcode::BitFieldUExtract && leaf->NumArgs() == 3u &&
		    leaf->Arg(1).Resolve() == Value(0u) && ImmediateU32(leaf->Arg(2), bits) && bits <= 32u) {
			leaf = leaf->Arg(0).Resolve().TryInstruction();
			if (leaf == nullptr) return {};
		}
		uint32_t axis = 0;
		if (leaf->GetOpcode() == ValueOpcode::GetBuiltin && leaf->NumArgs() == 2u &&
		    leaf->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::LocalInvocationId)) &&
		    ImmediateU32(leaf->Arg(1), axis) && axis < 3u &&
		    uint64_t {m_workgroup->threads_num[axis]} <= (uint64_t {1} << bits)) {
			std::array<uint64_t, 5> index {};
			index[axis + 1u] = 1u;
			return index;
		}
		if (!allow_induction || inst->GetOpcode() != ValueOpcode::Phi ||
		    (!induction.IsEmpty() && induction != value)) {
			return {};
		}
		induction = value;
		return std::array<uint64_t, 5> {0u, 0u, 0u, 0u, 1u};
	}

	bool LinearLocalIndex(const std::array<uint64_t, 5>& index) const {
		uint64_t stride = 1u;
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			if (m_workgroup->threads_num[axis] != 1u && index[axis + 1u] != stride) return false;
			stride *= m_workgroup->threads_num[axis];
		}
		return index[0] == 0u;
	}

	// The lanes of guard include the invocation whose linear local index is zero (ffb6ad6ca: also
	// the negated form !(local != 0)).
	bool LocalZeroWitness(Value guard) const {
		const auto* equal   = SimplifyGuard(guard).TryInstruction();
		const bool  negated = equal != nullptr && equal->GetOpcode() == ValueOpcode::LogicalNot;
		if (negated) equal = SimplifyGuard(equal->Arg(0)).TryInstruction();
		const auto expected = negated ? ValueOpcode::INotEqual32 : ValueOpcode::IEqual32;
		if (equal == nullptr || equal->GetOpcode() != expected) return false;
		for (uint32_t arg = 0; arg < 2u; ++arg) {
			if (equal->Arg(arg).Resolve() != Value(0u)) continue;
			Value      induction;
			const auto index = LocalIndex(equal->Arg(arg ^ 1u), guard, induction, false);
			if (index && LinearLocalIndex(*index)) return true;
		}
		return false;
	}

	// The runtime value guard proves value below (unsigned).
	Value UnsignedBound(Value guard, Value value, bool positive = true, Value witness = {},
	                    uint32_t depth = 0) const {
		if (depth > 16u) return {};
		const auto* test = SimplifyGuard(guard).TryInstruction();
		if (test == nullptr) return {};
		const auto op = test->GetOpcode();
		if (op == ValueOpcode::LogicalNot) return UnsignedBound(test->Arg(0), value, !positive, witness, depth + 1u);
		if ((op == ValueOpcode::LogicalAnd && positive) || (op == ValueOpcode::LogicalOr && !positive)) {
			const auto left = UnsignedBound(test->Arg(0), value, positive, witness, depth + 1u);
			return left.IsEmpty() ? UnsignedBound(test->Arg(1), value, positive, witness, depth + 1u) : left;
		}
		if (op == ValueOpcode::LogicalAnd && !positive && !witness.IsEmpty()) {
			// !(A & Q) proves !Q only for accesses whose EXEC also proves A.
			for (uint32_t arm = 0; arm < 2u; ++arm) {
				if (!SelectorImplies(witness, test->Arg(arm))) continue;
				const auto bound = UnsignedBound(test->Arg(arm ^ 1u), value, false, witness, depth + 1u);
				if (!bound.IsEmpty()) return bound;
			}
		}
		if (test->NumArgs() != 2u) return {};
		if ((positive ? op == ValueOpcode::ULessThan32 : op == ValueOpcode::UGreaterThanEqual32) &&
		    EquivalentValue(m_program, test->Arg(0), value)) {
			return test->Arg(1);
		}
		if ((positive ? op == ValueOpcode::UGreaterThan32 : op == ValueOpcode::ULessThanEqual32) &&
		    EquivalentValue(m_program, test->Arg(1), value)) {
			return test->Arg(0);
		}
		uint32_t limit = 0;
		if ((positive ? op == ValueOpcode::ULessThanEqual32 : op == ValueOpcode::UGreaterThan32) &&
		    ImmediateU32(test->Arg(1), limit) && limit != UINT32_MAX &&
		    EquivalentValue(m_program, test->Arg(0), value)) {
			return Value(limit + 1u);
		}
		return {};
	}

	// 0e9c77b0a + ffb6ad6ca: a record index read from a one-wave LDS append list (the producer
	// appends candidate payloads below its loop bound with SharedAtomicIAdd32 counters that one
	// zero-index invocation initialized) is below the producers' bound.
	Value SharedLoadBound(Value value, Value active) {
		if (m_workgroup == nullptr || m_program.stage != ShaderType::Compute) return {};
		const auto* read = ActiveValue(value, active).TryInstruction();
		if (read == nullptr || read->GetOpcode() != ValueOpcode::LoadSharedU32 ||
		    !SelectorImplies(active, read->Arg(1))) {
			return {};
		}
		uint64_t group_size = 1u;
		for (const auto dimension: m_workgroup->threads_num) {
			if (dimension == 0u || dimension > m_program.wave_size) return {};
			group_size *= dimension;
		}
		// Indexed LDS operations within one native wave execute in order. Multiwave append lists
		// require a separate workgroup synchronization proof.
		if (group_size != m_program.wave_size || (group_size != 32u && group_size != 64u)) return {};
		const auto memory = [&](const Inst& inst) -> const MemoryInfo* {
			const auto index = inst.Flags<MemoryFlags>().index;
			if (index >= m_program.memory_info.size()) return nullptr;
			const auto& info = m_program.memory_info[index];
			return info.kind == ResourceKind::Lds && info.data_bits == 32u && info.secondary_offset == 0u &&
			               info.component_index == 0u
			           ? &info
			           : nullptr;
		};
		const auto constant_address = [&](const Inst& inst, uint32_t& address) {
			const auto* info = memory(inst);
			if (info == nullptr ||
			    !ImmediateU32(ActiveValue(inst.Arg(0), inst.Arg(inst.NumArgs() - 1u)), address) ||
			    uint64_t {address} + info->offset > UINT32_MAX) {
				return false;
			}
			address += info->offset;
			return (address & 3u) == 0u;
		};
		const auto*  read_memory = memory(*read);
		AffineOffset read_offset;
		if (read_memory == nullptr || !MatchAffineOffset(read->Arg(0), active, read_offset) ||
		    read_offset.index.IsEmpty() || read_offset.stride != 4u ||
		    read_offset.offset + read_memory->offset > UINT32_MAX) {
			return {};
		}
		const auto* limit = ZeroBasedLoop(read_offset.index, read->Parent(), [](const Inst& compare, uint32_t) {
			return compare.GetOpcode() == ValueOpcode::ULessThan32;
		});
		if (limit == nullptr) return {};
		auto        counter_active = active;
		const auto* counter_read   = ActiveValue(limit->Arg(1), active).TryInstruction();
		// Every lane reads the same counter word, so a selected lane carries it.
		if (counter_read != nullptr && counter_read->GetOpcode() == ValueOpcode::ReadFirstLane) {
			counter_active = counter_read->Arg(1);
			counter_read   = ActiveValue(counter_read->Arg(0), counter_active).TryInstruction();
		}
		uint32_t consumed_counter = 0;
		if (counter_read == nullptr || counter_read->GetOpcode() != ValueOpcode::LoadSharedU32 ||
		    !SelectorImplies(counter_active, counter_read->Arg(1)) ||
		    !constant_address(*counter_read, consumed_counter)) {
			return {};
		}
		struct Initializer {
			const Inst* inst;
			uint32_t    address;
			uint32_t    words;
		};
		struct Append {
			const Inst* atomic;
			const Inst* store   = nullptr;
			uint32_t    counter = 0;
			uint32_t    base    = 0;
			Value       bound;
		};
		std::vector<Initializer> initializers;
		std::vector<Append>      appends;
		std::vector<const Inst*> stores;
		for (const auto* block: m_program.blocks) {
			for (const auto& inst: *block) {
				const auto op = inst.GetOpcode();
				if (AddressOpcodeInfoOf(op).access == AddressAccess::Write) {
					const auto index = inst.Flags<MemoryFlags>().index;
					if (index >= m_program.memory_info.size() ||
					    m_program.memory_info[index].kind == ResourceKind::Flat) {
						return {};
					}
				}
				const auto access = SharedAccessOf(op);
				if (access == SharedAccess::None || access == SharedAccess::Read) continue;
				const auto* info = memory(inst);
				if (info == nullptr) return {};
				const auto guard   = inst.Arg(inst.NumArgs() - 1u);
				uint32_t   address = 0;
				if (access == SharedAccess::Atomic) {
					if (op != ValueOpcode::SharedAtomicIAdd32 || ActiveValue(inst.Arg(1), guard) != Value(1u) ||
					    !constant_address(inst, address) ||
					    uint64_t {address} + 4u > uint64_t {m_workgroup->lds_size_dwords} * 4u ||
					    address >= 0x10000u ||
					    std::ranges::any_of(appends, [&](const Append& at) { return at.counter == address; })) {
						return {};
					}
					appends.push_back({.atomic = &inst, .counter = address});
				} else if (access == SharedAccess::Write) {
					const auto words = SharedComponentCount(op);
					bool       zero  = words != 0u && LocalZeroWitness(guard) && constant_address(inst, address) &&
					            address % (words * 4u) == 0u;
					for (uint32_t w = 0; w < words; ++w) {
						zero = zero && ActiveValue(inst.Arg(w + 1u), guard) == Value(0u);
					}
					if (zero) {
						initializers.push_back({&inst, address, words});
					} else if (op == ValueOpcode::WriteSharedU32) {
						stores.push_back(&inst);
					} else {
						return {};
					}
				} else {
					return {};
				}
			}
		}
		if (appends.empty() || stores.size() != appends.size() || initializers.empty()) return {};
		for (const auto& initial: initializers) {
			if (InCycle(initial.inst->Parent())) return {};
			// Every initialized word is a counter; arbitrary shared writes are not append provenance.
			for (uint32_t w = 0; w < initial.words; ++w) {
				if (std::ranges::none_of(appends, [&](const Append& at) {
					    return uint64_t {initial.address} + w * 4u == at.counter;
				    })) {
					return {};
				}
			}
		}
		for (auto& append: appends) {
			const auto guard = append.atomic->Arg(2);
			if (Reaches(counter_read->Parent(), append.atomic->Parent())) return {};
			uint32_t initialized = 0;
			for (const auto& initial: initializers) {
				const auto witness = initial.inst->Arg(initial.inst->NumArgs() - 1u);
				if (!AvailableBefore(*initial.inst, *append.atomic, witness) ||
				    !AvailableBefore(*initial.inst, *counter_read, witness)) {
					return {};
				}
				if (append.counter >= initial.address &&
				    uint64_t {append.counter} < uint64_t {initial.address} + initial.words * 4u) {
					++initialized;
				}
			}
			if (initialized != 1u) return {};
			for (const auto* store: stores) {
				if (!EquivalentValue(m_program, guard, store->Arg(2)) || !BeforeInBlock(*append.atomic, *store)) {
					continue;
				}
				AffineOffset address;
				const auto*  info = memory(*store);
				if (!MatchAffineOffset(store->Arg(0), guard, address) ||
				    address.index != Value(const_cast<Inst*>(append.atomic)) || address.stride != 4u ||
				    (address.offset + info->offset) % 4u != 0u ||
				    address.offset + info->offset + uint64_t {MaxTableKeyProbes} * 4u > UINT32_MAX ||
				    std::ranges::any_of(appends, [&](const Append& at) {
					    return address.offset + info->offset <= at.counter;
				    })) {
					continue;
				}
				if (append.store != nullptr) return {};
				const auto payload = ActiveValue(store->Arg(1), guard);
				Value      induction;
				const auto index = LocalIndex(payload, guard, induction, true);
				if (!index || !LinearLocalIndex(*index) || (*index)[4] != 1u || induction.IsEmpty()) return {};
				const auto* bound = BoundedInduction(
				    induction, append.atomic->Parent(),
				    [&](uint32_t initial, uint32_t step) { return initial == 0u && step == group_size; },
				    [&](const Inst& compare, uint32_t initial) {
					    return compare.GetOpcode() == ValueOpcode::ULessThan32 &&
					           ValidateRuntimeValue(m_program, compare.Arg(1), RuntimeValueType::Integer) &&
					           !InCycle(induction.Resolve().TryInstruction()->PhiBlock(initial));
				    });
				if (bound == nullptr ||
				    !EquivalentValue(m_program, UnsignedBound(guard, payload), bound->Arg(1))) {
					return {};
				}
				// Each re-entry must advance the producer induction; an inner append cycle could
				// otherwise increment the counter twice for one candidate.
				const auto* producer = induction.Resolve().TryInstruction()->Parent();
				if (std::ranges::any_of(append.atomic->Parent()->ImmPredecessors(), [&](const Block* previous) {
					    return Reaches(append.atomic->Parent(), previous, producer);
				    })) {
					return {};
				}
				append.store = store;
				append.base  = static_cast<uint32_t>(address.offset + info->offset);
				append.bound = bound->Arg(1);
			}
			if (append.store == nullptr) return {};
		}
		const auto consumed = std::ranges::find_if(appends, [&](const Append& at) {
			return at.counter == consumed_counter && at.base == read_offset.offset + read_memory->offset;
		});
		if (consumed == appends.end()) return {};
		// Built from opcodes the SRT evaluators support: max(a, b) = a < b ? b : a.
		const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
			auto&  node  = m_program.value_storage.emplace_back(op);
			size_t index = 0;
			for (const auto arg: args) node.SetArg(index++, arg);
			return Value(&node);
		};
		const auto umax = [&](Value a, Value b) {
			return emit(ValueOpcode::SelectU32, {emit(ValueOpcode::ULessThan32, {a, b}), b, a});
		};
		Value bound      = consumed->bound;
		Value all_bounds = bound;
		for (const auto& append: appends) {
			if (EquivalentValue(m_program, consumed->bound, append.bound)) continue;
			all_bounds        = umax(all_bounds, append.bound);
			auto contribution = append.bound;
			if (append.base != consumed->base) {
				// Each list occupies at most its producer bound in DWORDs. Only a list whose possible
				// payload span overlaps the consumed span can supply its values.
				const auto gap     = (std::max(append.base, consumed->base) - std::min(append.base, consumed->base)) / 4u;
				const auto overlap = emit(ValueOpcode::ULessThan32,
				                          {Value(gap), append.base < consumed->base ? append.bound : consumed->bound});
				contribution       = emit(ValueOpcode::SelectU32, {overlap, contribution, Value(0u)});
			}
			bound = umax(bound, contribution);
		}
		if (all_bounds == consumed->bound) return bound;
		bound = emit(ValueOpcode::SelectU32,
		             {emit(ValueOpcode::ULessThan32, {Value(0u), consumed->bound}), bound, Value(0u)});
		// Every bound must fit the cap, including when the consumer is empty: a larger one makes
		// the count fail the materialization's limit (all_bounds > cap).
		return emit(ValueOpcode::SelectU32,
		            {emit(ValueOpcode::UGreaterThan32, {all_bounds, Value(MaxTableKeyProbes)}), all_bounds,
		             bound});
	}

	bool TryMakeIndirectImage(Inst& handle, uint32_t pc, IndirectImagePlan& plan) {
		if (handle.GetOpcode() != ValueOpcode::GetImageResource || handle.NumArgs() != 8u) {
			return WhyNot(__LINE__);
		}
		Inst* table_handle = nullptr;
		Value key;
		uint32_t table_offset = 0;
		uint32_t key_mask     = UINT32_MAX;
		Inst*    masked       = nullptr;
		bool     word0_shared = false;
		// KYTY_BINDLESS_STRIDE: bytes from one key's T# to the next (32 for a plain T# array).
		const bool strided      = GetCodegenOptions().bindless_strided && BindlessImagesEnabled();
		// KYTY_WORKGROUP_IMAGES: a key that is the workgroup id may select records of any stride.
		const bool workgroup    = GetCodegenOptions().workgroup_images &&
		                          m_program.stage == ShaderType::Compute;
		uint32_t   table_stride = 32u;
		// KYTY_HEAP_KEY_MASK: the table of the descriptor's plain (non-Phi) words, for
		// ResolveDescriptorPhi.
		const Inst* reference_table = nullptr;
		if (GetCodegenOptions().heap_key_mask) {
			for (uint32_t dword = 0; dword < plan.reads.size() && reference_table == nullptr; ++dword) {
				// A plain word may itself be carried through a Phi web whose inputs agree
				// (Senaxx/KytyPS5-GT7 5ebc89daa resolves it the same way).
				const auto* plain =
				    ResolveInvariantPhi(m_program, handle.Arg(dword)).TryInstruction();
				uint32_t    memory_index = 0;
				if (plain != nullptr && plain->GetOpcode() != ValueOpcode::Phi &&
				    ScalarReadMemory(*plain, memory_index) != nullptr) {
					reference_table = plain->Arg(0).Resolve().TryInstruction();
				}
			}
		}
		// KYTY_R128_TABLE_IMAGES: image instructions that all take a 128-bit T# read four words
		// from the table; the translator passes the upper four as constants.
		const bool     r128 = GetCodegenOptions().r128_table_images && AllUsersR128(handle);
		const uint32_t tsharp_dwords = r128 ? 4u : static_cast<uint32_t>(plan.reads.size());
		for (uint32_t dword = tsharp_dwords; dword < plan.reads.size(); ++dword) {
			plan.memory[dword] = UINT32_MAX;
			plan.reads[dword]  = nullptr;
		}
		for (uint32_t dword = 0; dword < tsharp_dwords; ++dword) {
			auto* read = handle.Arg(dword).Resolve().TryInstruction();
			// KYTY_HEAP_KEY_MASK: a Phi web whose inputs agree resolves even when no word is plain
			// (ResolveDescriptorPhi then needs no reference table).
			if (read != nullptr && read->GetOpcode() == ValueOpcode::Phi &&
			    GetCodegenOptions().heap_key_mask) {
				read = ResolveDescriptorPhi(Value(read), reference_table).TryInstruction();
			}
			if (read == nullptr) {
				return WhyNot(__LINE__);
			}
			uint32_t current_mask   = UINT32_MAX;
			Inst*    current_masked = nullptr;
			uint32_t memory_index = 0;
			const auto* memory = ScalarReadMemory(*read, memory_index);
			if (memory == nullptr || memory->offset > INT32_MAX || (memory->offset & 3u) != 0u ||
			    !MemoryIndexBelongsTo(memory_index, *read)) {
				return WhyNot(__LINE__);
			}
			auto* current_handle = read->Arg(0).Resolve().TryInstruction();
			Value current_key;
			uint32_t offset = 0;
			uint32_t current_stride = 32u;
			const auto match_offset = [&] {
				if (MatchTableOffset(read->Arg(1), current_key, offset, &current_mask, &current_masked,
				                     strided ? &current_stride : nullptr)) {
					return true;
				}
				// KYTY_WORKGROUP_IMAGES: without KYTY_BINDLESS_STRIDE, a strided form only for a
				// workgroup-id key (every other key is rejected as before).
				return workgroup && !strided &&
				       MatchTableOffset(read->Arg(1), current_key, offset, &current_mask,
				                        &current_masked, &current_stride) &&
				       current_masked == nullptr && MatchWorkgroupKey(current_key, nullptr);
			};
			if (current_handle == nullptr ||
			    current_handle->GetOpcode() != (memory->kind == ResourceKind::ScalarAddress
			                                      ? ValueOpcode::GetAddressResource
			                                      : ValueOpcode::GetBufferResource) ||
			    (memory->kind == ResourceKind::ScalarAddress && read->Parent() != handle.Parent()) ||
			    (table_handle != nullptr &&
			     !EquivalentValue(m_program, Value(table_handle), Value(current_handle))) ||
			    !match_offset() || memory->offset > UINT32_MAX - offset) {
				return WhyNot(__LINE__);
			}
			offset += memory->offset;
			if (dword == 0u) {
				key = current_key;
				table_offset = offset;
				key_mask     = current_mask;
				masked       = current_masked;
				table_stride = current_stride;
			} else if (current_stride != table_stride ||
			           !EquivalentValue(m_program, key, current_key) || key_mask != current_mask ||
			           static_cast<uint64_t>(table_offset) + dword * sizeof(uint32_t) != offset) {
				return WhyNot(__LINE__);
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
					return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source)) {
			return WhyNot(__LINE__);
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
				// KYTY_BOUNDED_KEY_IMAGES: every key below a constant field width or low-bit mask is
				// a candidate; the table words are snapshot dependencies, whatever the key's source.
				if (indirect.key_count.IsEmpty() && selector != nullptr &&
				    GetCodegenOptions().bounded_key_images) {
					uint32_t bound = 0;
					uint32_t width = 0;
					uint32_t mask  = 0;
					if (selector->GetOpcode() == ValueOpcode::BitFieldUExtract && selector->NumArgs() == 3u &&
					    ImmediateU32(selector->Arg(2), width) && width != 0u && width <= 6u) {
						bound = 1u << width;
					} else if (selector->GetOpcode() == ValueOpcode::BitwiseAnd32 && selector->NumArgs() == 2u &&
					           (ImmediateU32(selector->Arg(1), mask) || ImmediateU32(selector->Arg(0), mask)) &&
					           mask != 0u && mask < 64u && (mask & (mask + 1u)) == 0u) {
						bound = mask + 1u;
					}
					if (bound != 0u && table_offset <= UINT32_MAX - bound * 32u) {
						indirect.key_count = Value(bound);
					}
				}
				if (indirect.key_count.IsEmpty() &&
				    !MatchUniformizedMaterialKey(key, handle, indirect, material_source, pc) &&
				    !(SelectorKeys(SelectorKeysBitset) && !m_shader_writes &&
				      MatchReversedImageRange(key, pc, indirect, material_source))) {
					if (!GetCodegenOptions().table_key_bounds) return WhyNot(__LINE__);
					const auto maximum = TableKeyMaximum(key, handle);
					if (maximum < MaxTableKeyProbes) {
						indirect.key_count = Value(maximum + 1u);
					} else {
						indirect.key_count = ProgramLoopCount(key);
						if (indirect.key_count.IsEmpty()) return WhyNot(__LINE__);
					}
				}
				if ((table_offset & 3u) != 0u ||
				    (bitscan && table_offset > UINT32_MAX - (32u * 32u - 1u))) return WhyNot(__LINE__);
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
					return WhyNot(__LINE__);
				}
				Value selector;
				DescriptorSource::IndirectDescriptor::SelectorRead selector_read;
				// KYTY_SELECTOR_BITSCAN / KYTY_SELECTOR_SHARED_LIST: a bounded selector of any kind
				// (else only a V_READFIRSTLANE selector, whose records are every offset, as before).
				const bool bounded_selectors =
				    SelectorKeys(SelectorKeysBitscan) || SelectorKeys(SelectorKeysSharedList);
				// 0e9c77b0a: V_READFIRSTLANE of a whole affine record offset (lane_index * stride +
				// offset under lane_mask) selects the record of the first active lane's index.
				Value lane_index, lane_mask;
				const auto match_lane_offset = [&] {
					uint32_t    immediate = 0;
					Value       address   = material_read->Arg(1).Resolve();
					const auto* add       = address.TryInstruction();
					if (add != nullptr && add->GetOpcode() == ValueOpcode::IAdd32 && add->NumArgs() == 2u) {
						for (uint32_t side = 0; side < 2u; ++side) {
							if (ImmediateU32(add->Arg(side), immediate)) {
								address = add->Arg(side ^ 1u).Resolve();
								break;
							}
						}
					}
					const auto*  lane = address.TryInstruction();
					AffineOffset affine;
					if (lane == nullptr || lane->GetOpcode() != ValueOpcode::ReadFirstLane ||
					    !MatchAffineOffset(lane->Arg(0), lane->Arg(1), affine) || affine.index.IsEmpty() ||
					    affine.stride == 0u || ((affine.stride | affine.offset) & 3u) != 0u ||
					    uint64_t {immediate} + affine.offset > UINT32_MAX) {
						return false;
					}
					selector_read.stride = static_cast<uint32_t>(affine.stride);
					selector_read.offset = static_cast<uint32_t>(immediate + affine.offset);
					lane_index           = affine.index;
					lane_mask            = lane->Arg(1);
					selector             = Value(const_cast<Inst*>(lane));
					return HasActiveLane(lane_mask, lane->Parent());
				};
				if (!(MatchMaterialOffset(material_read->Arg(1), selector, selector_read.stride,
				                          selector_read.offset, bounded_selectors) ||
				      (SelectorKeys(SelectorKeysSharedList) && match_lane_offset())) ||
				    selector_read.offset > UINT32_MAX - memory->offset) {
					return WhyNot(__LINE__);
				}
				selector_read.offset += memory->offset;
				const auto* lane = selector.TryInstruction();
				const bool  read_first_lane =
				    lane_index.IsEmpty() && lane != nullptr && lane->GetOpcode() == ValueOpcode::ReadFirstLane;
				if (!lane_index.IsEmpty()) {
					// No unbounded fallback for this form: only the append-list bound covers it.
					if (m_shader_writes) return false;
					indirect.key_count = SharedLoadBound(lane_index, lane_mask);
					if (indirect.key_count.IsEmpty()) return false;
				} else if (bounded_selectors && !m_shader_writes) {
					// f297c0bfe: other bounded indices, such as bit scans, read the records up to their
					// maximum; 99f4eba22: a bit scan of a mask load reads only the ranks its words set.
					if (SelectorKeys(SelectorKeysBitscan)) {
						const auto maximum = SelectorMaximum(selector, {}, 0u, PositiveUseGuard(material_read->Parent()));
						if (maximum < MaxTableKeyProbes) indirect.key_count = Value(maximum + 1u);
						if (maximum == 31u && SelectorKeys(SelectorKeysMaskRanks)) {
							selector_read.ranks = BitScanRanks(selector, pc);
						}
					}
					// 0e9c77b0a / ffb6ad6ca: a record selected from a shared append list is below the
					// list's bound.
					if (indirect.key_count.IsEmpty() && read_first_lane && SelectorKeys(SelectorKeysSharedList) &&
					    HasActiveLane(lane->Arg(1), lane->Parent())) {
						indirect.key_count = SharedLoadBound(lane->Arg(0), lane->Arg(1));
					}
				}
				if (indirect.key_count.IsEmpty() && !read_first_lane) {
					return false;
				}
				if (SelectorKeys(SelectorKeysExcluded)) {
					selector_read.excluded = ExcludedKeyOnEntry(key, handle.Parent());
				}
				const auto* shift = plan.reads[0]->Arg(1).Resolve().TryInstruction();
				const std::array<const Inst*, 1> material_users {shift};
				// KYTY_SELECTOR_EXCLUDED: the guard's own comparison of the key against an
				// immediate may use it too (the read stays in the shader).
				const auto guard_compare = [&](const Use& use) {
					uint32_t immediate = 0;
					return SelectorKeys(SelectorKeysExcluded) && use.user->NumArgs() == 2u &&
					       (use.user->GetOpcode() == ValueOpcode::IEqual32 ||
					        use.user->GetOpcode() == ValueOpcode::INotEqual32) &&
					       ImmediateU32(use.user->Arg(use.operand ^ 1u), immediate);
				};
				const bool key_uses = std::ranges::all_of(material_read->Uses(), [&](const Use& use) {
					return use.user == shift || guard_compare(use);
				});
				if (shift == nullptr || !(UsesOnly(*material_read, material_users) || key_uses) ||
				    !UsesOnly(*shift, plan.reads)) {
					return WhyNot(__LINE__);
				}
				const auto* material_handle = material_read->Arg(0).Resolve().TryInstruction();
				if (material_handle == nullptr ||
				    material_handle->GetOpcode() != ValueOpcode::GetBufferResource ||
				    !MakeRuntimeTableSource(*material_handle, pc, material_source)) {
					return WhyNot(__LINE__);
				}
				selector_read.source = InternSource(material_source);
				indirect.selector    = selector_read;
			}
			return true;
		};
		// A masked key is a lane value (MatchTableOffset): no enumeration covers it. A shared
		// word 0 has no planning read for the material selector either.
		// A strided key (KYTY_BINDLESS_STRIDE) is not enumerated: the enumeration reads 32-byte T#s.
		// KYTY_WORKGROUP_IMAGES (adapted from upstream KytyPS5 0ee880090 and c23de6218): a key that is
		// the workgroup id enumerates that axis' workgroup count at each dispatch, reading key k's
		// T# at table_offset + k * table_stride (MaterializeIndirectImage); any user, storage
		// images too. Words the shader also uses as values stay real reads (above).
		// The key may also be a runtime base plus the workgroup id, divided by a constant
		// (MatchWorkgroupKey, upstream de3ed905c).
		if (workgroup && masked == nullptr && !word0_shared && MatchWorkgroupKey(key, &indirect)) {
			if (table_stride == 0u || (table_stride & 3u) != 0u || (table_offset & 3u) != 0u) {
				return WhyNot(__LINE__);
			}
			indirect.table_stride = table_stride;
		} else if (masked != nullptr || word0_shared || table_stride != 32u || !enumerable()) {
			const bool sampled_only = std::ranges::all_of(handle.Uses(), [](const Use& use) {
				return ImageOpcodeInfoOf(use.user->GetOpcode()).resource_class ==
				       ImageResourceClass::Sampled;
			});
			// The host reads key k's T# at table_offset + k * table_stride: any 4-byte-aligned
			// stride that holds a T# (GT7's 188-byte material records).
			if (!BindlessImagesEnabled() || !sampled_only || table_source.dword_count != 4u ||
			    table_stride < (r128 ? 16u : 32u) || (table_stride & 3u) != 0u) {
				return WhyNot(__LINE__);
			}
			indirect              = {};
			indirect.table_offset = table_offset;
			indirect.table_stride = table_stride;
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
				return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetAddressResource ||
			    current->NumArgs() != 2u ||
			    (pointer != nullptr && !EquivalentValue(m_program, Value(pointer), Value(current)))) {
				return WhyNot(__LINE__);
			}
			if (dword == 0u) {
				tsharp_offset = memory->offset;
			} else if (static_cast<uint64_t>(tsharp_offset) + dword * sizeof(uint32_t) !=
			           memory->offset) {
				return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			auto*      current = read->Arg(0).Resolve().TryInstruction();
			const auto offset  = static_cast<uint64_t>(memory->offset) + immediate;
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource) {
				return WhyNot(__LINE__);
			}
			if (half == 0u) {
				table_handle   = current;
				key            = current_key;
				shift          = current_shift;
				record_pointer = static_cast<uint32_t>(offset);
			} else if (!EquivalentValue(m_program, Value(table_handle), Value(current)) ||
			           !EquivalentValue(m_program, key, current_key) || shift != current_shift ||
			           static_cast<uint64_t>(record_pointer) + 4u != offset) {
				return WhyNot(__LINE__);
			}
		}
		if (record_pointer + 8u > (1u << shift)) {
			return WhyNot(__LINE__);
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			auto*      current = read->Arg(0).Resolve().TryInstruction();
			const auto offset  = static_cast<uint64_t>(memory->offset) + immediate;
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource) {
				return WhyNot(__LINE__);
			}
			if (half == 0u) {
				table_handle   = current;
				key            = current_key;
				shift          = current_shift;
				record_pointer = static_cast<uint32_t>(offset);
			} else if (!EquivalentValue(m_program, Value(table_handle), Value(current)) ||
			           !EquivalentValue(m_program, key, current_key) || shift != current_shift ||
			           static_cast<uint64_t>(record_pointer) + 4u != offset) {
				return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			const auto* handle = read->Arg(0).Resolve().TryInstruction();
			if (handle == nullptr || handle->GetOpcode() != handle_opcode ||
			    (source != nullptr && !EquivalentValue(m_program, Value(source), Value(handle)))) {
				return WhyNot(__LINE__);
			}
			if (dword == 0u) {
				offset = memory->offset;
			} else if (static_cast<uint64_t>(offset) + dword * 4u != memory->offset) {
				return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			offset += memory->offset;
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource ||
			    (heap_handle != nullptr &&
			     !EquivalentValue(m_program, Value(heap_handle), Value(current)))) {
				return WhyNot(__LINE__);
			}
			if (dword == 0u) {
				inner_key    = current_key;
				table_offset = offset;
			} else if (!EquivalentValue(m_program, inner_key, current_key) ||
			           static_cast<uint64_t>(table_offset) + dword * 4u != offset) {
				return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
		}
		// ... whose V# is S_LOAD_DWORDX4 through the record pointer.
		const Inst* pointer       = nullptr;
		uint32_t    nested_vsharp = 0;
		if (!MatchConsecutiveWords(*outer_handle, ValueOpcode::GetAddressResource, pointer,
		                           nested_vsharp)) {
			return WhyNot(__LINE__);
		}
		Inst*    table_handle   = nullptr;
		Value    record_key;
		uint32_t shift          = 0;
		uint32_t record_pointer = 0;
		if (!MatchRecordPointer(*pointer, table_handle, record_key, shift, record_pointer)) {
			return WhyNot(__LINE__);
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
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
				return WhyNot(__LINE__);
			}
			offset += memory->offset;
			const auto* current = read->Arg(0).Resolve().TryInstruction();
			if (current == nullptr || current->GetOpcode() != ValueOpcode::GetBufferResource ||
			    (heap_handle != nullptr &&
			     !EquivalentValue(m_program, Value(heap_handle), Value(current)))) {
				return WhyNot(__LINE__);
			}
			if (dword == 0u) {
				inner_key    = current_key;
				table_offset = offset;
			} else if (!EquivalentValue(m_program, inner_key, current_key) ||
			           static_cast<uint64_t>(table_offset) + dword * 4u != offset) {
				return WhyNot(__LINE__);
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
			return WhyNot(__LINE__);
		}
		Inst*    table_handle   = nullptr;
		Value    record_key;
		uint32_t shift          = 0;
		uint32_t record_pointer = 0;
		if (!MatchRecordPointer(*pointer, table_handle, record_key, shift, record_pointer)) {
			return WhyNot(__LINE__);
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return WhyNot(__LINE__);
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

	// With bindless images, the S# beside a bindless or runtime T# may be read from a sampler heap
	// with S_BUFFER_LOAD (per key, like the T#). The sampler plan (TryMakeBindlessSampler or the
	// stand-in, UseDefaultSamplerIfUnresolved) is made at the image operation, after the walk has
	// passed those reads: collected first, they became a buffer resource whose V# words dead code
	// elimination then removed with the stranded reads, and materialization failed. A scalar
	// buffer read that feeds nothing but sampler handles (and the dead phis a waterfall loop
	// leaves) is collected after the walk instead, and only if an ordinary sampler still uses it;
	// otherwise it is planning-only, like an indirect image plan's T# reads (Senaxx 59cb8a83e
	// marks a bindless sampler's exclusive reads the same way).
	bool IsSamplerOnlyRead(const Inst& inst) const {
		if (!GetCodegenOptions().sampler_reads_deferred || !BindlessImagesEnabled() ||
		    inst.GetOpcode() != ValueOpcode::ReadConstBuffer) {
			return false;
		}
		uint32_t index = 0;
		if (ScalarReadMemory(inst, index) == nullptr || inst.Uses().empty()) {
			return false;
		}
		bool sampler = false;
		if (!FeedsOnlySamplers(inst, sampler) || !sampler) {
			return false;
		}
		return true;
	}

	// Whether every use of value (through phis) is a sampler handle; sampler: one was found.
	static bool FeedsOnlySamplers(const Inst& value, bool& sampler) {
		std::vector<const Inst*> pending {&value};
		std::unordered_set<const Inst*> visited {&value};
		while (!pending.empty()) {
			const auto* current = pending.back();
			pending.pop_back();
			for (const auto& use: current->Uses()) {
				const auto op = use.user->GetOpcode();
				if (op == ValueOpcode::GetSamplerResource) {
					sampler = true;
				} else if (op == ValueOpcode::Phi) {
					if (visited.insert(use.user).second) {
						pending.push_back(use.user);
					}
				} else {
					return false;
				}
			}
		}
		return true;
	}

	void CollectDeferredSamplerReads() {
		if (m_deferred_sampler_reads.empty()) {
			return;
		}
		auto deferred = std::move(m_deferred_sampler_reads);
		m_deferred_sampler_reads.clear();
		std::vector<const Inst*> stranded;
		m_collecting_deferred = true;
		for (auto* read: deferred) {
			// Still the S# of a sampler without a heap plan: an ordinary read.
			bool live = false;
			std::vector<const Inst*>        pending {read};
			std::unordered_set<const Inst*> visited {read};
			while (!pending.empty() && !live) {
				const auto* current = pending.back();
				pending.pop_back();
				for (const auto& use: current->Uses()) {
					if (use.user->GetOpcode() == ValueOpcode::Phi) {
						if (visited.insert(use.user).second) {
							pending.push_back(use.user);
						}
					} else if (FindBindlessSampler(use.user) == nullptr) {
						live = true;
						break;
					}
				}
			}
			if (live) {
				Collect(*read);
				continue;
			}
			m_program.memory_info[read->Flags<MemoryFlags>().index].planning_only = true;
			stranded.push_back(read);
		}
		m_collecting_deferred = false;
		if (!stranded.empty()) {
			std::erase_if(m_program.dynamic_reads, [&](Value value) {
				return std::ranges::find(stranded, value.Resolve().TryInstruction()) != stranded.end();
			});
		}
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

	const BindlessSamplerPlan* FindBindlessSampler(const Inst* handle) const {
		if (handle == nullptr) {
			return nullptr;
		}
		const auto found = std::ranges::find(m_bindless_samplers, handle, &BindlessSamplerPlan::handle);
		return found == m_bindless_samplers.end() ? nullptr : &*found;
	}

	// KYTY_BINDLESS_SAMPLERS (a minimal port of Senaxx's mirrored sampler heaps, 59cb8a83e): the S#
	// beside a bindless or runtime image whose four dwords are scalar reads of one sampler heap at
	// key * stride + offset (a key the shader computes, a stride of 16 or more, a heap V# the host
	// can evaluate) is planned as a bindless sampler: the shader keeps the key and indexes the
	// bindless sampler array, which the host mirrors from the heap. The reads stay ordinary loads.
	// False: the caller falls back to the stand-in sampler (UseDefaultSamplerIfUnresolved).
	bool TryMakeBindlessSampler(Value value, bool sample_adjust, uint32_t pc) {
		auto* handle = value.Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetSamplerResource ||
		    handle->NumArgs() != 4u) {
			return false;
		}
		if (FindBindlessSampler(handle) != nullptr) {
			return true;
		}
		DescriptorSource descriptor;
		MakeSource(*handle, 4u, true, sample_adjust, descriptor, pc);
		uint32_t bad_dword = 0;
		if (ValidateSource(descriptor, bad_dword)) {
			return false; // an ordinary S#
		}
		Inst*    table_handle = nullptr;
		Value    key;
		uint32_t table_offset = 0;
		uint32_t table_stride = 0;
		for (uint32_t dword = 0; dword < 4u; ++dword) {
			const auto* read = ResolveInvariantPhi(m_program, handle->Arg(dword)).TryInstruction();
			if (read == nullptr) {
				return false;
			}
			uint32_t    memory_index = 0;
			const auto* memory       = ScalarReadMemory(*read, memory_index);
			if (memory == nullptr || memory->kind != ResourceKind::ScalarBuffer ||
			    memory->offset > INT32_MAX || (memory->offset & 3u) != 0u) {
				return false;
			}
			auto*    current_handle = read->Arg(0).Resolve().TryInstruction();
			Value    current_key;
			uint32_t offset = 0;
			uint32_t stride = 0;
			if (current_handle == nullptr ||
			    current_handle->GetOpcode() != ValueOpcode::GetBufferResource ||
			    (table_handle != nullptr &&
			     !EquivalentValue(m_program, Value(table_handle), Value(current_handle))) ||
			    !MatchTableOffset(read->Arg(1), current_key, offset, nullptr, nullptr, &stride) ||
			    stride < 16u || (stride & 3u) != 0u || memory->offset > UINT32_MAX - offset) {
				return false;
			}
			offset += memory->offset;
			if (dword == 0u) {
				key          = current_key;
				table_offset = offset;
				table_stride = stride;
			} else if (stride != table_stride || !EquivalentValue(m_program, key, current_key) ||
			           static_cast<uint64_t>(table_offset) + dword * sizeof(uint32_t) != offset) {
				return false;
			}
			table_handle = current_handle;
		}
		// A key the host can evaluate has an ordinary plan (the S# would be a valid value).
		if (ValidateRuntimeValue(m_program, key)) {
			return false;
		}
		DescriptorSource table_source;
		if (!MakeRuntimeTableSource(*table_handle, pc, table_source) ||
		    table_source.dword_count != 4u) {
			return false;
		}
		DescriptorSource source;
		source.dword_count = 4u;
		source.dwords.fill(Value(0u));
		BindlessSamplerPlan plan;
		for (uint32_t dword = 0; dword < 3u; dword++) {
			source.dwords[dword]    = table_source.dwords[dword];
			plan.table_roots[dword] = table_source.dwords[dword];
		}
		source.bindless_sampler =
		    DescriptorSource::BindlessSampler {.table_offset = table_offset, .stride = table_stride};
		plan.handle = handle;
		plan.key    = key;
		plan.source = InternSource(source);
		m_bindless_samplers.push_back(plan);
		return true;
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
				m_why.clear();
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
				} else {
					m_why_by_handle[handle] = m_why;
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
				// KYTY_SRT_UNRESOLVED_ALL: which descriptor kind and which dword (and the opcode that
				// produced it) has no runtime plan.
				static const bool why = [] {
					const char* value = std::getenv("KYTY_SRT_UNRESOLVED_ALL");
					return value != nullptr && value[0] == '1';
				}();
				if (why) {
					const auto* producer = descriptor.dwords[bad_dword].Resolve().TryInstruction();
					Log::WriteToConsoleAndLog(fmt::format(
					    "KYTY_SRT_UNRESOLVED_ALL: {} shader 0x{:016x} pc=0x{:08x} {} dword {} from {} why=[{}] off={}\n",
					    StageName(m_program.stage), m_program.shader_hash, pc, ValueOpcodeName(expected),
					    bad_dword, producer != nullptr ? ValueOpcodeName(producer->GetOpcode()) : "a constant",
					    m_why_by_handle.contains(handle) ? m_why_by_handle[handle] : std::string("-"),
					    producer != nullptr && producer->NumArgs() >= 2u ? DescribeValue(producer->Arg(1), 0)
					                                                     : std::string("-")));
				}
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

	uint32_t AddImage(uint32_t source, const MemoryInfo& memory, ValueOpcode op, uint32_t pc,
	                  uint32_t array_index_bound = ImageResource::NoArrayIndexBound) {
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
				image.cube |= ImageCubeTrackEnabled() && memory.image_cube;
				image.max_array_index = std::max(image.max_array_index, array_index_bound);
				Merge(image, op, pc);
				image.table_access |= TableAccess(memory, op);
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
		image.cube           = ImageCubeTrackEnabled() && memory.image_cube;
		image.atomic64       = atomic64;
		if (const auto* descriptor = Source(source); descriptor != nullptr &&
		                                             descriptor->indirect_descriptor.has_value() &&
		                                             descriptor->indirect_descriptor->runtime) {
			image.bindless = true;
			image.runtime  = true;
		}
		image.max_array_index = array_index_bound;
		Merge(image, op, pc);
		image.table_access |= TableAccess(memory, op);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	// KYTY_SAMPLED_TABLE_READS: the access kind ImageResource::table_access records (0 without the
	// switch, so programs and their cache entries stay as they were).
	static uint32_t TableAccess(const MemoryInfo& memory, ValueOpcode op) {
		if (!GetCodegenOptions().sampled_table_reads) {
			return 0u;
		}
		switch (op) {
			case ValueOpcode::ImageRead:
			case ValueOpcode::ImageReadStorage: return ImageTableAccessRead;
			case ValueOpcode::ImageWrite: return ImageTableAccessWrite;
			case ValueOpcode::ImageSampleRaw: {
				// Gradients and offsets take the instruction's dimension (not a candidate's); an
				// explicit LOD or clamp follows the coordinates (a candidate with more coordinates
				// than the instruction would read it as one).
				if ((memory.image_sample_flags &
				     (Decoder::ImageSampleFlagDerivative | Decoder::ImageSampleFlagOffset)) != 0u) {
					return ImageTableAccessOther;
				}
				return (memory.image_sample_flags &
				        (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagLodClamp)) != 0u
				           ? ImageTableAccessSampleExtra
				           : ImageTableAccessSample;
			}
			default: return ImageTableAccessOther;
		}
	}

	// The bound of the layer coordinate of a sample or gather of a sampled array image (ImageResource::max_array_index):
	// the layer it reads is the coordinate rounded to the nearest, clamped to the view's layers (the hardware's rule and
	// Vulkan's), so a coordinate of a known range reaches no layer past its rounded maximum. Demon's Souls' texture pools
	// are sampled through views of a few thousand layers whose shaders take the layer from a byte: 256 of them (a
	// 1,023-layer image of 1 GiB on an 8 GB card, issue #174). Any other operation on the image, a cube, a 16-bit
	// coordinate or an operand of unknown range: no bound.
	// Only the lanes whose result the shader keeps count (ResultLanes): the translator writes a register as
	// select(exec, new, old) and the old value of a lane the exec leaves out is the register's earlier use (an
	// interpolated value, a product). Another lane's layer is clamped to the view as well, its texel never seen.
	uint32_t ArrayIndexBound(const Inst& inst, const MemoryInfo& memory, ValueOpcode op) {
		const auto dimension = memory.image_dimension;
		if ((op != ValueOpcode::ImageSampleRaw && op != ValueOpcode::ImageGatherRaw) ||
		    (dimension != Decoder::ImageDimension::Dim1DArray && dimension != Decoder::ImageDimension::Dim2DArray) ||
		    inst.NumArgs() < 3)
			return ImageResource::NoArrayIndexBound;
		const auto* address = inst.Arg(2).Resolve().TryInstruction();
		if (address == nullptr || address->GetOpcode() != ValueOpcode::MakeImageAddress)
			return ImageResource::NoArrayIndexBound;
		// The address's components in SpirvEmitter's order (Layout): offset, bias, compare, derivatives, the coordinates
		// (the layer after the spatial ones).
		const uint32_t flags   = memory.image_sample_flags;
		const uint32_t spatial = dimension == Decoder::ImageDimension::Dim1DArray ? 1u : 2u;
		uint32_t       layer   = spatial;
		if ((flags & Decoder::ImageSampleFlagOffset) != 0u) layer++;
		if ((flags & Decoder::ImageSampleFlagBias) != 0u) layer++;
		if ((flags & Decoder::ImageSampleFlagCompare) != 0u) layer++;
		if ((flags & Decoder::ImageSampleFlagDerivative) != 0u) layer += 2u * spatial;
		// (16-bit coordinates, A16: no bound. Otherwise every component is one 32-bit word, as
		// Decoder::ImageAddressComponentLayout lays them out. A component past the address's words
		// reads 0: SpirvEmitter's AddressU32.)
		if ((flags & Decoder::ImageSampleFlagA16) != 0u) return ImageResource::NoArrayIndexBound;
		const auto word = layer;
		if (word >= address->NumArgs()) return 0;
		m_lanes          = ResultLanes(inst);
		CollectLaneTerms();
		const auto bound = SolveBound(address->Arg(word));
		if (!(bound < 65536.0)) return ImageResource::NoArrayIndexBound;
		return static_cast<uint32_t>(std::floor(bound + 0.5));
	}

	// The lanes an image operation's result is kept for: every component of it taken out and written to a register by
	// a select on one condition (the exec mask the translator gives the operation's destination registers), that
	// condition. None (an empty value: every lane) when a use is anything else or the conditions differ.
	static Value ResultLanes(const Inst& image) {
		Value lanes;
		for (const auto& use: image.Uses()) {
			const auto* extract = use.user;
			if (extract == nullptr || extract->GetOpcode() != ValueOpcode::CompositeExtractU32x4) return {};
			for (const auto& kept: extract->Uses()) {
				const auto* select = kept.user;
				if (select == nullptr || select->GetOpcode() != ValueOpcode::SelectU32 || kept.operand != 1) return {};
				const auto condition = select->Arg(0).Resolve();
				if (lanes.IsEmpty()) {
					lanes = condition;
				} else if (!(lanes == condition)) {
					return {};
				}
			}
		}
		return lanes;
	}

	// The terms of the result's lanes (ResultLanes), each true in all of them: the condition and the operands of its ands;
	// and the values a term is the not of, false in all of them. Per ArrayIndexBound call, as the answers below.
	void CollectLaneTerms() {
		m_lane_terms.clear();
		m_lane_negated.clear();
		m_lanes_imply.clear();
		m_lanes_exclude.clear();
		if (m_lanes.IsEmpty()) return;
		std::vector<Value> pending {m_lanes};
		while (!pending.empty()) {
			const auto* term = pending.back().Resolve().TryInstruction();
			pending.pop_back();
			if (term == nullptr || !m_lane_terms.insert(term).second) continue;
			if (term->GetOpcode() == ValueOpcode::LogicalAnd) {
				pending.push_back(term->Arg(0));
				pending.push_back(term->Arg(1));
			} else if (term->GetOpcode() == ValueOpcode::LogicalNot) {
				if (const auto* negated = term->Arg(0).Resolve().TryInstruction(); negated != nullptr)
					m_lane_negated.insert(negated);
			}
		}
	}

	// Whether `condition` holds in every lane the result is kept for: the lanes' condition, a term of it, or an and of
	// such. (Each and answered once: the conditions share their terms, the exec masks of nested branches.)
	bool LanesImply(Value condition, uint32_t depth) {
		condition = condition.Resolve();
		if (m_lanes.IsEmpty() || depth > 16) return false;
		if (condition == m_lanes.Resolve()) return true;
		const auto* inst = condition.TryInstruction();
		if (inst == nullptr) return false;
		if (m_lane_terms.contains(inst)) return true;
		if (inst->GetOpcode() != ValueOpcode::LogicalAnd) return false;
		if (const auto known = m_lanes_imply.find(inst); known != m_lanes_imply.end()) return known->second;
		const bool implied   = LanesImply(inst->Arg(0), depth + 1) && LanesImply(inst->Arg(1), depth + 1);
		m_lanes_imply[inst] = implied;
		return implied;
	}

	// Whether `condition` fails in every lane the result is kept for: a term of the lanes is its not, it is the not of a
	// condition they imply, or an and with such an operand.
	bool LanesExclude(Value condition, uint32_t depth) {
		condition = condition.Resolve();
		if (m_lanes.IsEmpty() || depth > 16) return false;
		const auto* inst = condition.TryInstruction();
		if (inst == nullptr) return false;
		if (m_lane_negated.contains(inst)) return true;
		if (inst->GetOpcode() == ValueOpcode::LogicalNot) return LanesImply(inst->Arg(0), depth + 1);
		if (inst->GetOpcode() != ValueOpcode::LogicalAnd) return false;
		if (const auto known = m_lanes_exclude.find(inst); known != m_lanes_exclude.end()) return known->second;
		const bool excluded   = LanesExclude(inst->Arg(0), depth + 1) || LanesExclude(inst->Arg(1), depth + 1);
		m_lanes_exclude[inst] = excluded;
		return excluded;
	}

	// The bound analysis: each value it meets, read as a float or as an unsigned integer, for the lanes the result is
	// kept for or for any lane (past a loop's phi), is a node whose bound follows from its operands' by a rule. Every
	// rule grows with its operands, so the bounds, all 0 at first and recomputed in rounds until none grows, are the
	// least ones the rules allow for every value a loop's iterations give as well. (A value read as a float may also be
	// negative: the first layer. NoBound: none.)
	enum class BoundRule : uint8_t {
		Constant,        // its value
		Copy,            // its operand's
		Max,             // its largest operand's (0 without one)
		Min,             // its smallest operand's (an operand of no bound gives the others')
		RoundEven,       // to the nearest integer
		Floor,           // (a truncation too: the bounds are not negative)
		Ceil,
		SignedToFloat,   // an integer below the sign bit
		BitsToFloat,     // an integer's bits: below the sign bit, a larger integer is a larger float
		FloatToUnsigned, // a float's integer part
		ShiftRight,      // by `shift`
		BitField,        // `bits` bits from bit `shift` (32: all there are)
	};
	struct BoundNode {
		BoundRule rule  = BoundRule::Constant;
		uint8_t   shift = 0;
		uint8_t   bits  = 32;
		uint32_t  first = 0; // its operands: m_bound_operands[first, first + count), met as m_bound_specs
		uint32_t  count = 0;
		double    value = 0.0;
	};
	struct BoundOperand {
		Value value;
		bool  is_float = false;
		bool  lanes    = false;
	};

	static constexpr double NoBound = std::numeric_limits<double>::infinity();
	// (A shader of more values than that, or whose bounds still grow after that many rounds: no bound.)
	static constexpr uint32_t MaxBoundNodes  = 4096;
	static constexpr uint32_t MaxBoundRounds = 16;

	// A 32-bit value's bits read as a float: a negative one counts as 0 (the first layer), an infinity or NaN as no bound.
	static double FloatOfBits(uint32_t bits) {
		if ((bits & 0x7f800000u) == 0x7f800000u) return NoBound;
		if ((bits & 0x80000000u) != 0u) return 0.0;
		return static_cast<double>(std::bit_cast<float>(bits));
	}

	// Whether a phi may take a value along a back edge: a predecessor that is not before its block (the blocks are in
	// the order of their code; an unknown one counts as such). Past one the result's lanes no longer select the values:
	// they are other iterations', whose conditions are not the ones the lanes are made of.
	bool IsLoopPhi(const Inst& phi) {
		if (m_block_order.empty())
			for (size_t i = 0; i < m_program.blocks.size(); i++) m_block_order[m_program.blocks[i]] = i;
		const auto block = m_block_order.find(phi.Parent());
		if (block == m_block_order.end()) return true;
		for (size_t i = 0; i < phi.NumPhiBlocks(); i++) {
			const auto from = m_block_order.find(phi.PhiBlock(i));
			if (from == m_block_order.end() || from->second >= block->second) return true;
		}
		return false;
	}

	// The bound of `root`, a 32-bit value read as a float, for the result's lanes (m_lanes): the nodes of the values it
	// is made of, each after its operands' (but around a loop), then the rounds.
	double SolveBound(Value root) {
		m_bound_nodes.clear();
		m_bound_specs.clear();
		m_bound_operands.clear();
		m_bound_order.clear();
		m_bound_index.clear();
		m_bound_stack.clear();
		bool       fresh = false;
		const auto top   = BoundNodeOf({root, true, true}, fresh);
		if (!top) return NoBound;
		if (fresh) m_bound_stack.emplace_back(*top, 0u);
		while (!m_bound_stack.empty()) {
			const auto [id, next] = m_bound_stack.back();
			if (next == m_bound_nodes[id].count) {
				m_bound_order.push_back(id);
				m_bound_stack.pop_back();
				continue;
			}
			m_bound_stack.back().second++;
			const uint32_t     slot    = m_bound_nodes[id].first + next;
			const BoundOperand operand = m_bound_specs[slot];
			const auto         node    = BoundNodeOf(operand, fresh);
			if (!node) return NoBound;
			m_bound_operands[slot] = *node;
			if (fresh) m_bound_stack.emplace_back(*node, 0u);
		}
		for (uint32_t round = 0; round < MaxBoundRounds; round++) {
			bool grew = false;
			for (const auto id: m_bound_order) {
				auto& node = m_bound_nodes[id];
				if (node.rule == BoundRule::Constant) continue;
				if (const double bound = BoundOf(node); bound != node.value) {
					node.value = bound;
					grew       = true;
				}
			}
			if (!grew) return m_bound_nodes[*top].value;
		}
		return NoBound;
	}

	// The node of a value read as a float or as an integer for the lanes or not: the one it has, or a new one (`fresh`,
	// its operands still to meet). None past MaxBoundNodes.
	std::optional<uint32_t> BoundNodeOf(const BoundOperand& operand, bool& fresh) {
		fresh            = false;
		const auto value = operand.value.Resolve();
		if (m_bound_nodes.size() >= MaxBoundNodes) return std::nullopt;
		const auto* inst = value.IsImmediate() ? nullptr : value.TryInstruction();
		if (inst == nullptr) {
			BoundNode node;
			node.value = NoBound;
			if (value.IsImmediate() && operand.is_float && value.GetType() == Type::F32)
				node.value = FloatOfBits(std::bit_cast<uint32_t>(value.F32Value()));
			else if (value.IsImmediate() && value.GetType() == Type::U32)
				node.value = operand.is_float ? FloatOfBits(value.U32()) : static_cast<double>(value.U32());
			m_bound_nodes.push_back(node);
			return static_cast<uint32_t>(m_bound_nodes.size() - 1);
		}
		const auto key = reinterpret_cast<uintptr_t>(inst) | (operand.is_float ? 2u : 0u) | (operand.lanes ? 1u : 0u);
		const auto id  = static_cast<uint32_t>(m_bound_nodes.size());
		if (const auto [known, added] = m_bound_index.try_emplace(key, id); !added) return known->second;
		BoundNode node;
		node.first = static_cast<uint32_t>(m_bound_specs.size());
		MakeBoundRule(node, *inst, operand.is_float, operand.lanes);
		node.count = static_cast<uint32_t>(m_bound_specs.size()) - node.first;
		m_bound_operands.resize(m_bound_specs.size());
		m_bound_nodes.push_back(node);
		fresh = true;
		return id;
	}

	// The rule of `inst` read as a float (is_float) or as an unsigned integer for the lanes or not, its operands added
	// to m_bound_specs. Any other operation: no bound.
	void MakeBoundRule(BoundNode& node, const Inst& inst, bool is_float, bool lanes) {
		const auto add = [&](Value value, bool as_float, bool for_lanes) {
			m_bound_specs.push_back({value, as_float, for_lanes});
		};
		const auto args = [&](size_t count, bool as_float, BoundRule rule) {
			for (size_t i = 0; i < count; i++) add(inst.Arg(i), as_float, lanes);
			node.rule = rule;
		};
		const auto constant = [&](size_t index) -> std::optional<uint32_t> {
			const auto value = inst.Arg(index).Resolve();
			return value.IsImmediate() && value.GetType() == Type::U32 ? std::optional<uint32_t>(value.U32()) : std::nullopt;
		};
		const auto op = inst.GetOpcode();
		// A select of a condition the lanes decide: the operand they take.
		if (op == ValueOpcode::SelectU32 || (is_float && op == ValueOpcode::SelectF32)) {
			node.rule = BoundRule::Copy;
			if (lanes && LanesImply(inst.Arg(0), 0)) return add(inst.Arg(1), is_float, lanes);
			if (lanes && LanesExclude(inst.Arg(0), 0)) return add(inst.Arg(2), is_float, lanes);
			node.rule = BoundRule::Max;
			add(inst.Arg(1), is_float, lanes);
			return add(inst.Arg(2), is_float, lanes);
		}
		if (op == ValueOpcode::Phi) {
			const bool through = lanes && !IsLoopPhi(inst);
			for (size_t i = 0; i < inst.NumArgs(); i++) add(inst.Arg(i), is_float, through);
			node.rule = BoundRule::Max;
			return;
		}
		if (is_float) {
			switch (op) {
				case ValueOpcode::BitCastU32F32:
				case ValueOpcode::BitCastF32U32: return args(1, true, BoundRule::Copy);
				case ValueOpcode::FPRoundEven32: return args(1, true, BoundRule::RoundEven);
				case ValueOpcode::FPFloor32:
				case ValueOpcode::FPTrunc32: return args(1, true, BoundRule::Floor);
				case ValueOpcode::FPCeil32: return args(1, true, BoundRule::Ceil);
				case ValueOpcode::FPMax32: return args(2, true, BoundRule::Max);
				case ValueOpcode::FPMaxTri32: return args(3, true, BoundRule::Max);
				// (No larger than any operand of a bound: a NaN operand gives the other one, and no value of a bound
				// here is a NaN.)
				case ValueOpcode::FPMin32: return args(2, true, BoundRule::Min);
				case ValueOpcode::FPMinTri32: return args(3, true, BoundRule::Min);
				case ValueOpcode::ConvertF32U32: return args(1, false, BoundRule::Copy);
				case ValueOpcode::ConvertF32S32: return args(1, false, BoundRule::SignedToFloat);
				default:
					// The instruction read as an integer.
					node.rule = BoundRule::BitsToFloat;
					return add(Value(const_cast<Inst*>(&inst)), false, lanes);
			}
		}
		switch (op) {
			case ValueOpcode::BitwiseAnd32:
			case ValueOpcode::UMin32: return args(2, false, BoundRule::Min);
			case ValueOpcode::UMax32: return args(2, false, BoundRule::Max);
			case ValueOpcode::ShiftRightLogical32: {
				const auto shift = constant(1);
				node.shift       = static_cast<uint8_t>(shift.value_or(0) & 31u);
				return args(1, false, shift ? BoundRule::ShiftRight : BoundRule::Copy);
			}
			case ValueOpcode::BitFieldUExtract: {
				const auto offset = constant(1), count = constant(2);
				node.shift        = static_cast<uint8_t>(offset.value_or(0) & 31u);
				node.bits         = static_cast<uint8_t>(count && *count < 32u ? *count : 32u);
				return args(1, false, BoundRule::BitField);
			}
			case ValueOpcode::ConvertU32U8: node.value = 0xffu; return;
			case ValueOpcode::ConvertU32U16: node.value = 0xffffu; return;
			case ValueOpcode::ConvertU32F32: return args(1, true, BoundRule::FloatToUnsigned);
			default: node.value = NoBound; return;
		}
	}

	// A node's bound from its operands' now.
	double BoundOf(const BoundNode& node) const {
		const auto operand = [&](uint32_t index) { return m_bound_nodes[m_bound_operands[node.first + index]].value; };
		switch (node.rule) {
			case BoundRule::Constant: return node.value;
			case BoundRule::Copy: return operand(0);
			case BoundRule::Max: {
				double bound = 0.0;
				for (uint32_t i = 0; i < node.count; i++) bound = std::max(bound, operand(i));
				return bound;
			}
			case BoundRule::Min: {
				double bound = NoBound;
				for (uint32_t i = 0; i < node.count; i++) bound = std::min(bound, operand(i));
				return bound;
			}
			case BoundRule::RoundEven: return std::floor(operand(0) + 0.5);
			case BoundRule::Floor: return std::floor(operand(0));
			case BoundRule::Ceil: return std::ceil(operand(0));
			case BoundRule::SignedToFloat: return operand(0) <= 2147483647.0 ? operand(0) : NoBound;
			case BoundRule::BitsToFloat:
				return operand(0) <= 2139095039.0 ? FloatOfBits(static_cast<uint32_t>(operand(0))) : NoBound;
			case BoundRule::FloatToUnsigned: {
				const double bound = operand(0);
				if (bound == NoBound) return NoBound;
				return bound >= 4294967295.0 ? 4294967295.0 : std::floor(bound);
			}
			case BoundRule::ShiftRight: {
				const double bound = operand(0);
				return static_cast<double>((bound == NoBound ? UINT32_MAX : static_cast<uint32_t>(bound)) >> node.shift);
			}
			case BoundRule::BitField: {
				double bound = operand(0);
				if (bound != NoBound) bound = static_cast<double>(static_cast<uint32_t>(bound) >> node.shift);
				if (node.bits < 32u) bound = std::min(bound, static_cast<double>((1u << node.bits) - 1u));
				return bound;
			}
		}
		return NoBound;
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
		SamplerResource sampler {.source = source, .first_use_pc = pc};
		const auto*     descriptor = Source(source);
		sampler.bindless = descriptor != nullptr && descriptor->bindless_sampler.has_value();
		m_info.samplers.push_back(sampler);
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
		if (!m_collecting_deferred && IsSamplerOnlyRead(inst)) {
			m_deferred_sampler_reads.push_back(&inst);
			return;
		}
		Inst*    handle   = nullptr;
		uint32_t source   = 0;
		uint32_t resource = 0;

		if (buffer != BufferAccess::None) {
			// KYTY_BDA_FORCE_SHADERS: the access takes the BDA path even where the V# is a value the
			// command processor could evaluate.
			const bool forced = m_force_bda && memory.kind == ResourceKind::Buffer &&
			                    m_program.stage == ShaderType::Compute &&
			                    ((m_indirect_scalar_buffers && memory.SupportsIndirectRawLoad(op)) ||
			                     (m_bda_writes && memory.SupportsIndirectRawWrite(op)));
			if (forced || !GetHandle(inst.Arg(0), ValueOpcode::GetBufferResource, 4, flags.pc, handle,
			                         source)) {
				// KYTY_SRT_VARIANT_READS: an S_BUFFER_LOAD through a V# the shader computed (e.g. a
				// BVH instance's record) or any raw vector load through one reads through BDA like
				// the raw DWORD x2-x4 loads below.
				// KYTY_INDIRECT_FORMAT_D16: a D16 formatted XY/XYZW load (packed into dwords).
				const bool d16_formatted = GetCodegenOptions().indirect_format_d16 &&
				                           memory.kind == ResourceKind::Buffer && memory.formatted &&
				                           !memory.typed && memory.data_bits == 16u &&
				                           op == ValueOpcode::LoadBufferU32x2;
				const bool indirect_variant =
				    m_indirect_scalar_buffers &&
				    ((memory.kind == ResourceKind::ScalarBuffer &&
				      memory.SupportsIndirectBufferLoad(op)) ||
				     (memory.kind == ResourceKind::Buffer && memory.SupportsIndirectRawLoad(op)) ||
				     d16_formatted);
				// KYTY_BDA_WRITES: a raw store or atomic through one writes through BDA (compute
				// only: the renderer settles each dispatch of such a program synchronously).
				// KYTY_INDIRECT_FORMAT_STORES: a formatted 32-bit store too, in the V#'s run-time format.
				// KYTY_GRAPHICS_BDA_WRITES: raw stores and atomics in graphics stages too (the
				// renderer settles such a draw right after it).
				const bool graphics_raw = GetCodegenOptions().graphics_bda_writes &&
				                          m_program.stage != ShaderType::Compute &&
				                          memory.SupportsIndirectRawWrite(op);
				const bool indirect_write = m_bda_writes && memory.kind == ResourceKind::Buffer &&
				                            (m_program.stage == ShaderType::Compute || graphics_raw) &&
				                            (memory.SupportsIndirectRawWrite(op) ||
				                             (GetCodegenOptions().IndirectFormatStoresFor(m_program.shader_hash) &&
				                              IndirectFormattedStore(memory, op)));
				if (!indirect_variant && !indirect_write && !d16_formatted &&
				    (memory.kind != ResourceKind::Buffer || !memory.SupportsIndirectBufferLoad(op))) {
					// KYTY_DROP_GRAPHICS_INDIRECT_STORES: a graphics-stage raw store through a
					// computed V# is removed (an approximation) rather than skipping the program.
					if (GetCodegenOptions().drop_graphics_indirect_stores &&
					    m_program.stage != ShaderType::Compute && memory.kind == ResourceKind::Buffer &&
					    BufferAccessOf(op) == BufferAccess::Write && memory.SupportsIndirectRawWrite(op)) {
						Log::WriteToConsoleAndLog(fmt::format(
						    "KYTY_DROP_GRAPHICS_INDIRECT_STORES: {} shader 0x{:016x} pc=0x{:08x} {} "
						    "through a computed V# removed\n",
						    StageName(m_program.stage), m_program.shader_hash, flags.pc,
						    ValueOpcodeName(op)));
						m_dead_stores.push_back(&inst);
						return;
					}
					if (m_indirect_scalar_buffers) {
						// KYTY_SRT_UNRESOLVED_ALL: which buffer access has no BDA path.
						if (const char* value = std::getenv("KYTY_SRT_UNRESOLVED_ALL");
						    value != nullptr && value[0] == '1') {
							Log::WriteToConsoleAndLog(fmt::format(
							    "KYTY_SRT_UNRESOLVED_ALL: shader 0x{:016x} pc=0x{:08x} buffer op {} kind {} "
							    "typed {} formatted {} writes {}\n",
							    m_program.shader_hash, flags.pc, ValueOpcodeName(op),
							    static_cast<uint32_t>(memory.kind), memory.typed ? 1 : 0,
							    memory.formatted ? 1 : 0, m_bda_writes ? 1 : 0));
						}
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
					Fail(flags.pc, "scratch operation requires a nonzero per-thread scratch size");
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
			// Upstream c174ade18: the flattened-SRT slots of a base read straight from the SRT, so
			// the renderer can back its pages before a compute dispatch (KYTY_BACK_ADDRESS_BASES).
			if (const auto* base_handle = inst.Arg(0).Resolve().TryInstruction();
			    base_handle != nullptr && base_handle->GetOpcode() == ValueOpcode::GetAddressResource &&
			    base_handle->NumArgs() >= 2u) {
				std::array<uint32_t, 2> base {UINT32_MAX, UINT32_MAX};
				for (uint32_t word = 0; word < base.size(); ++word) {
					const auto* read = base_handle->Arg(word).Resolve().TryInstruction();
					if (read != nullptr && read->GetOpcode() == ValueOpcode::ReadConst &&
					    read->Arg(1).Resolve().IsImmediate()) {
						base[word] = read->Arg(1).Resolve().U32();
					}
				}
				auto& bases = m_info.address_bases;
				if (std::ranges::max(base) != UINT32_MAX && std::ranges::find(bases, base) == bases.end()) {
					bases.push_back(base);
				}
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
		bool runtime = false;
		if (indirect != nullptr) {
			source = indirect->source;
		} else if (handle != nullptr && TryRuntimeImage(*handle, memory, op, flags.pc, source)) {
			runtime = true;
		} else if (!GetHandle(inst.Arg(0), ValueOpcode::GetImageResource, 8, flags.pc, handle,
		                      source)) {
			return; // unresolved (KYTY_SRT_VARIANT_READS): the shader is skipped
		}
		// KYTY_TEXTURE_LAYER_BOUND: the layer coordinate's bound (chenxiao07/KytyPS5 d44e93d81,
		// 6b57b4939); computed only with the switch on (no cost, no plan change otherwise).
		resource = AddImage(source, memory, op, flags.pc,
		                    LayerBoundEnabled() ? ArrayIndexBound(inst, memory, op)
		                                        : ImageResource::NoArrayIndexBound);
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
			if (((indirect != nullptr && IsBindlessPlan(*indirect)) || runtime) &&
			    !(GetCodegenOptions().bindless_samplers &&
			      TryMakeBindlessSampler(inst.Arg(1), sample_adjust, flags.pc))) {
				UseDefaultSamplerIfUnresolved(inst.Arg(1), sample_adjust, flags.pc);
			}
			if (const auto* bindless = FindBindlessSampler(inst.Arg(1).Resolve().TryInstruction());
			    bindless != nullptr) {
				sampler_handle = bindless->handle;
				sampler_source = bindless->source;
			} else if (!GetHandle(inst.Arg(1), ValueOpcode::GetSamplerResource, 4, flags.pc,
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

	// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): a T# that is neither a
	// value the host can evaluate nor a heap entry at a key, but eight dwords the shader loads at
	// run time (GT7's material functions read theirs through the call record's data pointer). With
	// bindless images, a handle only read by size queries, loads and samples becomes the shared
	// runtime image (ImageResource::runtime): the shader looks its dwords up in the runtime image
	// table, which the host fills as shaders request T#s.
	bool TryRuntimeImage(Inst& handle, const MemoryInfo& memory, ValueOpcode op, uint32_t pc,
	                     uint32_t& source) {
		if (!GetCodegenOptions().runtime_images || !BindlessImagesEnabled() ||
		    handle.GetOpcode() != ValueOpcode::GetImageResource || handle.NumArgs() != 8u ||
		    memory.image_r128) {
			return false;
		}
		const auto readable = [](ValueOpcode opcode) {
			const auto info = ImageOpcodeInfoOf(opcode);
			return info.access == ImageAccess::Read &&
			       info.resource_class == ImageResourceClass::Sampled;
		};
		if (!readable(op) || memory.image_dimension == Decoder::ImageDimension::Unknown ||
		    memory.image_dimension == Decoder::ImageDimension::Dim2DMsaa ||
		    memory.image_dimension == Decoder::ImageDimension::Dim2DMsaaArray) {
			return false;
		}
		for (const auto& use: handle.Uses()) {
			if (use.operand != 0u || !readable(use.user->GetOpcode())) {
				return false;
			}
		}
		DescriptorSource descriptor;
		MakeSource(handle, 8u, false, false, descriptor, pc);
		uint32_t bad_dword = 0;
		if (ValidateSource(descriptor, bad_dword)) {
			return false; // the host evaluates it
		}
		for (uint32_t dword = 0; dword < 8u; dword++) {
			if (handle.Arg(dword).Resolve().GetType() != Type::U32) {
				return false;
			}
		}
		DescriptorSource runtime;
		runtime.dword_count = 8u;
		runtime.dwords.fill(Value(0u));
		runtime.indirect_descriptor.emplace();
		runtime.indirect_descriptor->runtime = true;
		source                               = InternSource(runtime);
		if (!m_logged_runtime_image) {
			m_logged_runtime_image = true;
			LOGF("shader resource tracking: hash=0x%016" PRIx64 " pc=0x%08x runtime image: the T# "
			     "is loaded at run time and looked up in the runtime image table\n",
			     m_program.shader_hash, pc);
		}
		return true;
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
	// (ArrayIndexBound's, per call: the nodes of the values met (SolveBound), the lanes whose result counts.)
	std::vector<BoundNode>                     m_bound_nodes;
	std::vector<BoundOperand>                  m_bound_specs;
	std::vector<uint32_t>                      m_bound_operands;
	std::vector<uint32_t>                      m_bound_order;
	std::vector<std::pair<uint32_t, uint32_t>> m_bound_stack;
	std::unordered_map<uintptr_t, uint32_t>    m_bound_index;
	Value                                      m_lanes;
	std::unordered_set<const Inst*>            m_lane_terms;
	std::unordered_set<const Inst*>            m_lane_negated;
	std::unordered_map<const Inst*, bool>      m_lanes_imply;
	std::unordered_map<const Inst*, bool>      m_lanes_exclude;
	std::unordered_map<const Block*, size_t>   m_block_order;
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
	std::vector<BindlessSamplerPlan>           m_bindless_samplers;
	std::vector<Inst*>                         m_deferred_sampler_reads; // IsSamplerOnlyRead
	bool                                       m_collecting_deferred = false;
	std::vector<std::pair<const Inst*, Value>> m_descriptor_selections;
	bool                                       m_shader_writes = false;
	bool                                       m_logged_runtime_image = false;
	bool                                       m_indirect_scalar_buffers = false;
	bool                                       m_bda_writes              = false;
	bool                                       m_force_bda               = false;
	uint32_t                                   m_unresolved_pc = UINT32_MAX;
	// KYTY_SRT_UNRESOLVED_ALL diagnostics: source lines at which the image planners gave up.
	mutable std::string                          m_why;
	std::unordered_map<const Inst*, std::string> m_why_by_handle;
	bool WhyNot(uint32_t line) const {
		m_why += std::to_string(line);
		m_why += ' ';
		return false;
	}
	// A small expression tree for the diagnostics (opcodes and immediates, depth 5).
	static std::string DescribeValue(Value value, uint32_t depth) {
		value = value.Resolve();
		uint32_t immediate = 0;
		if (ImmediateU32(value, immediate)) return fmt::format("0x{:x}", immediate);
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) return "?";
		std::string out(ValueOpcodeName(inst->GetOpcode()));
		if (depth >= 5u || inst->GetOpcode() == ValueOpcode::Phi) return out;
		out += '(';
		for (uint32_t i = 0; i < inst->NumArgs() && i < 3u; i++) {
			if (i != 0u) out += ',';
			out += DescribeValue(inst->Arg(i), depth + 1u);
		}
		return out + ')';
	}
	const ShaderWorkgroupInputInfo*            m_workgroup     = nullptr;

public:
	void SetWorkgroup(const ShaderWorkgroupInputInfo* workgroup) { m_workgroup = workgroup; }

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

uint32_t TrackResources(Program& program, bool indirect_scalar_buffers, bool bda_writes,
                        const ShaderWorkgroupInputInfo* workgroup) {
	Tracker tracker(program, indirect_scalar_buffers, bda_writes);
	tracker.SetWorkgroup(workgroup);
	tracker.Run();
	return tracker.UnresolvedPc();
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
