#include "graphics/shader/recompiler/ir/passes/ExecSelectElimination.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr int      ImplicationDepth  = 16;
constexpr uint32_t MaxFixpointPasses = 8;

bool IsSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

// Side-effect free, and each lane's result depends only on that lane's operands.
bool IsLaneLocalPure(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::BitCastU16F16:
		case ValueOpcode::BitCastF16U16:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU16U32:
		case ValueOpcode::ConvertU32U16:
		case ValueOpcode::ConvertU8U32:
		case ValueOpcode::ConvertU32U8:
		case ValueOpcode::ConvertF32F16:
		case ValueOpcode::ConvertF16F32:
		case ValueOpcode::ConvertS32F32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32S32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeConstructU32x3:
		case ValueOpcode::CompositeConstructF32x2:
		case ValueOpcode::CompositeConstructU32x4:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::CompositeExtractU32x3:
		case ValueOpcode::CompositeExtractU32x4:
		case ValueOpcode::PackHalf2x16:
		case ValueOpcode::PackSnorm2x16:
		case ValueOpcode::PackUnorm2x16:
		case ValueOpcode::PackFloat2x16Rtz:
		case ValueOpcode::FPAbs32:
		case ValueOpcode::FPNeg32:
		case ValueOpcode::FPSaturate32:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32:
		case ValueOpcode::SelectU32:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UDiv32:
		case ValueOpcode::SMulHi:
		case ValueOpcode::UMulHi:
		case ValueOpcode::IAbs32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::BitReverse32:
		case ValueOpcode::BitCount32:
		case ValueOpcode::BitCount64:
		case ValueOpcode::FindUMsb32:
		case ValueOpcode::FindUMsb64:
		case ValueOpcode::FindILsb32:
		case ValueOpcode::SMin32:
		case ValueOpcode::UMin32:
		case ValueOpcode::SMax32:
		case ValueOpcode::UMax32:
		case ValueOpcode::SMinTri32:
		case ValueOpcode::UMinTri32:
		case ValueOpcode::SMaxTri32:
		case ValueOpcode::UMaxTri32:
		case ValueOpcode::SMedTri32:
		case ValueOpcode::UMedTri32:
		case ValueOpcode::SLessThan32:
		case ValueOpcode::SLessThan64:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThan64:
		case ValueOpcode::IEqual32:
		case ValueOpcode::IEqual64:
		case ValueOpcode::SLessThanEqual32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::SGreaterThan32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::UGreaterThan64:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::INotEqual64:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::UGreaterThanEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdEqual32:
		case ValueOpcode::FPUnordEqual32:
		case ValueOpcode::FPOrdNotEqual32:
		case ValueOpcode::FPUnordNotEqual32:
		case ValueOpcode::FPOrdLessThan32:
		case ValueOpcode::FPUnordLessThan32:
		case ValueOpcode::FPOrdGreaterThan32:
		case ValueOpcode::FPUnordGreaterThan32:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPUnordLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPUnordGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPCmpClass32:
		case ValueOpcode::FPAdd32:
		case ValueOpcode::FPSub32:
		case ValueOpcode::FPFma32:
		case ValueOpcode::FPMad32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPMin32:
		case ValueOpcode::FPMax32:
		case ValueOpcode::FPMinTri32:
		case ValueOpcode::FPMaxTri32:
		case ValueOpcode::FPMedTri32:
		case ValueOpcode::FPRecip32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPRecipSqrt32:
		case ValueOpcode::FPSqrt:
		case ValueOpcode::FPSin:
		case ValueOpcode::FPCos:
		case ValueOpcode::FPExp2:
		case ValueOpcode::FPLog2:
		case ValueOpcode::FPLdexp:
		case ValueOpcode::FPRoundEven32:
		case ValueOpcode::FPFloor32:
		case ValueOpcode::FPCeil32:
		case ValueOpcode::FPTrunc32:
		case ValueOpcode::FPFract32:
		case ValueOpcode::MakeImageAddress: return true;
		default: return false;
	}
}

// Accesses the backend performs only in lanes whose last (U1) operand is set: loads then
// return zero, stores, atomics and exports do nothing (EmitIfCondition and friends).
bool IsGuardedAccess(ValueOpcode op) {
	const auto count = NumArgsOf(op);
	if (count == 0 || ArgTypeOf(op, count - 1) != Type::U1) {
		return false;
	}
	if (BufferAccessOf(op) != BufferAccess::None ||
	    AddressOpcodeInfoOf(op).access != AddressAccess::None) {
		return true;
	}
	switch (SharedAccessOf(op)) {
		case SharedAccess::Read:
		case SharedAccess::Write:
		case SharedAccess::Atomic: return true;
		default: break;
	}
	const auto image = ImageOpcodeInfoOf(op).access;
	return op == ValueOpcode::ImageRead || image == ImageAccess::Write ||
	       image == ImageAccess::Atomic || op == ValueOpcode::SetAttribute;
}

const Inst* ConditionInstruction(Value value) {
	return value.Resolve().TryInstruction();
}

// Whether `value` can only be true in lanes where `condition` is true.
bool Implies(Value value, const Inst* condition, int depth) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		return value.GetType() == Type::U1 && !value.U1();
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (inst == condition) {
		return true;
	}
	if (depth == 0) {
		return false;
	}
	switch (inst->GetOpcode()) {
		case ValueOpcode::LogicalAnd:
			return Implies(inst->Arg(0), condition, depth - 1) ||
			       Implies(inst->Arg(1), condition, depth - 1);
		case ValueOpcode::SelectU1:
			return Implies(inst->Arg(1), condition, depth - 1) &&
			       Implies(inst->Arg(2), condition, depth - 1);
		default: return false;
	}
}

// Instructions referenced from outside the instruction lists.
struct ExternalReferences {
	// Branch conditions and indirect-branch selectors: never replaced or removed. Their operands
	// may change, as every rewrite here keeps each lane's value of what it touches.
	std::unordered_set<const Inst*> pinned;
	// The resource plan's descriptor, SRT, fill, evaluation and dynamic-read values with every
	// operand they depend on (a closed set): left exactly as they are.
	std::unordered_set<const Inst*> frozen;

	[[nodiscard]] bool Keeps(const Inst* inst) const {
		return pinned.contains(inst) || frozen.contains(inst);
	}
};

ExternalReferences CollectExternalReferences(const Program& program) {
	ExternalReferences       refs;
	std::vector<const Inst*> pending;
	const auto               freeze = [&](const Inst* inst) {
        if (inst != nullptr && refs.frozen.insert(inst).second) {
            pending.push_back(inst);
        }
	};
	const auto freeze_value = [&](Value value) {
		if (!value.IsEmpty()) {
			freeze(value.Resolve().TryInstruction());
		}
	};
	const auto pin = [&](const Inst* inst) {
		if (inst != nullptr) {
			refs.pinned.insert(inst);
		}
	};
	const auto pin_value = [&](Value value) {
		if (!value.IsEmpty()) {
			pin(value.Resolve().TryInstruction());
		}
	};
	for (const auto& inst: program.value_storage) {
		freeze(&inst);
	}
	for (const auto& descriptor: program.descriptor_sources) {
		for (const auto& value: descriptor.dwords) {
			freeze_value(value);
		}
		if (descriptor.indirect_descriptor) {
			freeze_value(descriptor.indirect_descriptor->key_count);
			freeze_value(descriptor.indirect_descriptor->selector_first);
			freeze_value(descriptor.indirect_descriptor->selector_mask);
		}
	}
	for (const auto& read: program.srt_reads) {
		freeze_value(read.value);
	}
	for (const auto& recipe: program.evaluation_recipes) {
		freeze(recipe.instruction);
		freeze_value(recipe.selection_mask);
	}
	for (const auto& instruction: program.arithmetic_tape_instructions) {
		freeze(instruction.instruction);
	}
	for (const auto& value: program.uniform_fill.values) {
		freeze_value(value);
	}
	for (const auto& value: program.dynamic_reads) {
		freeze_value(value);
	}
	for (const auto& block: program.control_flow) {
		pin_value(block.condition);
	}
	for (const auto& info: program.block_info) {
		pin_value(info.condition);
		pin_value(info.indirect_target);
	}
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			freeze_value(inst->Arg(index));
		}
	}
	return refs;
}

// Immediate dominators of the blocks reachable from the entry (Cooper, Harvey and Kennedy).
class DominatorTree {
public:
	static constexpr size_t None = static_cast<size_t>(-1);

	explicit DominatorTree(const Program& program) {
		const auto count = program.blocks.size();
		for (size_t index = 0; index < count; index++) {
			m_index.emplace(program.blocks[index], index);
		}
		m_idom.assign(count, None);
		m_children.assign(count, {});
		if (count == 0) {
			return;
		}
		std::vector<size_t>                    postorder;
		std::vector<uint8_t>                   visited(count, 0);
		std::vector<std::pair<size_t, size_t>> stack;
		stack.push_back({0, 0});
		visited[0] = 1;
		while (!stack.empty()) {
			auto& [block, next]   = stack.back();
			const auto successors = program.blocks[block]->ImmSuccessors();
			if (next < successors.size()) {
				const auto successor = IndexOf(successors[next++]);
				if (successor != None && visited[successor] == 0) {
					visited[successor] = 1;
					stack.push_back({successor, 0});
				}
				continue;
			}
			postorder.push_back(block);
			stack.pop_back();
		}
		std::vector<size_t> order(count, None);
		for (size_t index = 0; index < postorder.size(); index++) {
			order[postorder[index]] = index;
		}
		m_idom[0]            = 0;
		const auto intersect = [&](size_t a, size_t b) {
			while (a != b) {
				while (order[a] < order[b]) {
					a = m_idom[a];
				}
				while (order[b] < order[a]) {
					b = m_idom[b];
				}
			}
			return a;
		};
		bool changed = true;
		while (changed) {
			changed = false;
			for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
				const auto block = *it;
				if (block == 0) {
					continue;
				}
				size_t candidate = None;
				for (const auto* predecessor: program.blocks[block]->ImmPredecessors()) {
					const auto index = IndexOf(predecessor);
					if (index == None || m_idom[index] == None) {
						continue;
					}
					candidate = candidate == None ? index : intersect(index, candidate);
				}
				if (candidate != None && m_idom[block] != candidate) {
					m_idom[block] = candidate;
					changed       = true;
				}
			}
		}
		for (size_t block = 1; block < count; block++) {
			if (m_idom[block] != None) {
				m_children[m_idom[block]].push_back(block);
			}
		}
	}

	[[nodiscard]] size_t IndexOf(const Block* block) const {
		const auto found = m_index.find(block);
		return found == m_index.end() ? None : found->second;
	}

	[[nodiscard]] bool Reachable(size_t block) const {
		return block < m_idom.size() && m_idom[block] != None;
	}

	[[nodiscard]] const std::vector<size_t>& Children(size_t block) const {
		return m_children[block];
	}

	[[nodiscard]] bool Dominates(size_t a, size_t b) const {
		if (!Reachable(a) || !Reachable(b)) {
			return false;
		}
		while (b != a) {
			if (b == 0) {
				return false;
			}
			b = m_idom[b];
		}
		return true;
	}

	[[nodiscard]] bool Dominates(const Block* a, const Block* b) const {
		const auto ia = IndexOf(a);
		const auto ib = IndexOf(b);
		return ia != None && ib != None && Dominates(ia, ib);
	}

private:
	std::unordered_map<const Block*, size_t> m_index;
	std::vector<size_t>                      m_idom;
	std::vector<std::vector<size_t>>         m_children;
};

class MaskedUseAnalysis {
public:
	// `assumed`: selects taken to be unobserved where their condition is false (the fixpoint
	// candidate set), or nullptr to count every phi use as an observation.
	MaskedUseAnalysis(const Program& program, const DominatorTree& dom,
	                  const std::unordered_set<const Inst*>* assumed)
	    : m_program(program), m_dom(dom), m_assumed(assumed) {}

	// Whether lanes where `condition` is false can never observe `root`'s value.
	bool ObservedOnlyUnder(const Inst* root, const Inst* condition) {
		auto& root_verdict = Lookup(root, condition);
		if (root_verdict != Verdict::Unknown) {
			return root_verdict == Verdict::Yes;
		}
		struct Frame {
			const Inst* inst;
			size_t      next_use;
		};
		std::vector<Frame> stack;
		root_verdict = Verdict::Pending;
		stack.push_back({root, 0});
		while (!stack.empty()) {
			auto&       frame      = stack.back();
			const auto& uses       = frame.inst->Uses();
			bool        descended  = false;
			bool        observed   = false;
			while (frame.next_use < uses.size()) {
				const auto use  = uses[frame.next_use];
				const auto kind = Classify(use, condition);
				if (kind == UseKind::Masked) {
					frame.next_use++;
					continue;
				}
				if (kind == UseKind::Observed) {
					observed = true;
					break;
				}
				auto& verdict = Lookup(use.user, condition);
				if (verdict == Verdict::Yes) {
					frame.next_use++;
					continue;
				}
				if (verdict != Verdict::Unknown) {
					// No, or a use cycle (only possible through a phi, which never gets here).
					observed = true;
					break;
				}
				verdict = Verdict::Pending;
				stack.push_back({use.user, 0});
				descended = true;
				break;
			}
			if (descended) {
				continue;
			}
			if (observed) {
				// Each frame waits on the one above it, so they all fail.
				for (const auto& failed: stack) {
					Lookup(failed.inst, condition) = Verdict::No;
				}
				return false;
			}
			Lookup(frame.inst, condition) = Verdict::Yes;
			stack.pop_back();
			if (!stack.empty()) {
				stack.back().next_use++;
			}
		}
		return true;
	}

private:
	enum class Verdict : uint8_t { Unknown, Pending, Yes, No };
	enum class UseKind : uint8_t { Masked, Recurse, Observed };

	struct Key {
		const Inst* inst;
		const Inst* condition;
		bool        operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const {
			const auto a = reinterpret_cast<uintptr_t>(key.inst);
			const auto b = reinterpret_cast<uintptr_t>(key.condition);
			return static_cast<size_t>(a * 0x9e3779b97f4a7c15ull ^ (b + (a << 6u) + (a >> 2u)));
		}
	};

	Verdict& Lookup(const Inst* inst, const Inst* condition) {
		return m_memo[{inst, condition}];
	}

	[[nodiscard]] bool UsesImplicitDerivatives(const Inst& sample) const {
		if (m_program.stage != ShaderType::Pixel) {
			return false; // Samples outside pixel shaders always take an explicit LOD.
		}
		const auto index = sample.Flags<MemoryFlags>().index;
		if (index >= m_program.memory_info.size()) {
			return true;
		}
		const auto flags = m_program.memory_info[index].image_sample_flags;
		return (flags & (Decoder::ImageSampleFlagLod | Decoder::ImageSampleFlagLevelZero |
		                 Decoder::ImageSampleFlagDerivative)) == 0u;
	}

	// Every use of `phi` is the false arm of a select assumed unobserved where it takes that arm.
	[[nodiscard]] bool FeedsOnlyAssumedFalseArms(const Inst& phi) const {
		for (const auto& use: phi.Uses()) {
			if (use.operand != 2 || !m_assumed->contains(use.user)) {
				return false;
			}
		}
		return true;
	}

	[[nodiscard]] UseKind Classify(const Use& use, const Inst* condition) const {
		const auto* user = use.user;
		const auto  op   = user->GetOpcode();
		switch (op) {
			case ValueOpcode::SelectU1:
			case ValueOpcode::SelectU32:
			case ValueOpcode::SelectF32:
				return use.operand == 1 && Implies(user->Arg(0), condition, ImplicationDepth)
				           ? UseKind::Masked
				           : UseKind::Recurse;
			case ValueOpcode::LogicalAnd:
				return Implies(user->Arg(use.operand == 0 ? 1 : 0), condition, ImplicationDepth)
				           ? UseKind::Masked
				           : UseKind::Recurse;
			case ValueOpcode::ImageSampleRaw:
				// Implicit derivatives read the coordinates of the other lanes of the quad.
				return UsesImplicitDerivatives(*user) ? UseKind::Observed : UseKind::Recurse;
			case ValueOpcode::ImageGatherRaw:
			case ValueOpcode::ImageQueryDimensions: return UseKind::Recurse;
			default: break;
		}
		if (IsLaneLocalPure(op)) {
			return UseKind::Recurse;
		}
		if (IsGuardedAccess(op)) {
			const auto guard = user->NumArgs() - 1;
			return use.operand != guard && Implies(user->Arg(guard), condition, ImplicationDepth)
			           ? UseKind::Masked
			           : UseKind::Observed;
		}
		if (op == ValueOpcode::Phi) {
			if (m_assumed != nullptr && FeedsOnlyAssumedFalseArms(*user)) {
				// Whatever the phi carries (into the next iteration, too) only reaches lanes
				// where those selects take their other arm, which nobody observes.
				return UseKind::Masked;
			}
			if (!m_dom.Dominates(user->Parent(), user->PhiBlock(use.operand))) {
				// A join forwards the value of the lanes that came along this edge. Without a
				// back edge in between, its users see the same instance of the condition.
				return UseKind::Recurse;
			}
			return UseKind::Observed;
		}
		// Cross-lane operations, references kept for the resource plan, unguarded side effects.
		return UseKind::Observed;
	}

	const Program&                               m_program;
	const DominatorTree&                         m_dom;
	const std::unordered_set<const Inst*>*       m_assumed;
	std::unordered_map<Key, Verdict, KeyHash>    m_memo;
};

class BranchFacts {
public:
	BranchFacts(const Program& program, const DominatorTree& dom)
	    : m_program(program), m_dom(dom) {
		CollectEdgeFacts();
	}

	// Calls `visit(block, known)` for every reachable block in dominator-tree preorder, with
	// `known(value)` giving what the branches leading into the block decided about `value`.
	template <typename Visit>
	void Walk(Visit&& visit) {
		if (m_program.blocks.empty()) {
			return;
		}
		struct Frame {
			size_t block;
			size_t next_child;
			size_t undo_mark;
		};
		std::vector<Frame> stack;
		const auto         enter = [&](size_t block) {
            const auto mark = m_undo.size();
            for (const auto& [inst, value]: m_edge_facts[block]) {
                AddFact(inst, value);
            }
            visit(m_program.blocks[block],
                  [&](Value value) { return Known(value, ImplicationDepth); });
            stack.push_back({block, 0, mark});
		};
		enter(0);
		while (!stack.empty()) {
			auto& frame = stack.back();
			if (frame.next_child < m_dom.Children(frame.block).size()) {
				const auto child = m_dom.Children(frame.block)[frame.next_child++];
				enter(child);
				continue;
			}
			while (m_undo.size() > frame.undo_mark) {
				m_facts.erase(m_undo.back());
				m_undo.pop_back();
			}
			stack.pop_back();
		}
	}

private:
	static constexpr size_t None = DominatorTree::None;

	[[nodiscard]] size_t IndexOf(const Block* block) const {
		return block == nullptr ? None : m_dom.IndexOf(block);
	}

	[[nodiscard]] const Block* TargetBlock(uint32_t id) const {
		for (size_t index = 0; index < m_program.block_info.size(); index++) {
			if (m_program.block_info[index].id == id) {
				return index < m_program.blocks.size() ? m_program.blocks[index] : nullptr;
			}
		}
		return nullptr;
	}

	void CollectEdgeFacts() {
		const auto count = m_program.blocks.size();
		m_edge_facts.assign(count, {});
		for (size_t index = 0; index < count && index < m_program.block_info.size(); index++) {
			const auto& info = m_program.block_info[index];
			if (info.terminator.kind != CFG::TerminatorKind::ConditionalBranch ||
			    info.condition.IsEmpty() || !m_dom.Reachable(index)) {
				continue;
			}
			const auto* condition = ConditionInstruction(info.condition);
			if (condition == nullptr) {
				continue;
			}
			const auto true_block  = IndexOf(TargetBlock(info.terminator.true_block));
			const auto false_block = IndexOf(TargetBlock(info.terminator.false_block));
			if (true_block == None || false_block == None || true_block == false_block) {
				continue;
			}
			for (const auto [target, value]: {std::pair {true_block, true}, {false_block, false}}) {
				if (EdgeDominatesTarget(index, target)) {
					Decompose(condition, value, m_edge_facts[target], ImplicationDepth);
				}
			}
		}
	}

	// Every path into `target` comes through the edge from `source` (its other predecessors are
	// back edges from blocks it dominates).
	[[nodiscard]] bool EdgeDominatesTarget(size_t source, size_t target) const {
		size_t from_source = 0;
		for (const auto* predecessor: m_program.blocks[target]->ImmPredecessors()) {
			const auto index = IndexOf(predecessor);
			if (index == source) {
				from_source++;
			} else if (index == None || !m_dom.Dominates(target, index)) {
				return false;
			}
		}
		return from_source == 1 && target != 0;
	}

	static void Decompose(const Inst* inst, bool value,
	                      std::vector<std::pair<const Inst*, bool>>& facts, int depth) {
		facts.emplace_back(inst, value);
		if (depth == 0) {
			return;
		}
		const auto operand = [&](size_t index) { return ConditionInstruction(inst->Arg(index)); };
		switch (inst->GetOpcode()) {
			case ValueOpcode::LogicalNot:
				if (const auto* source = operand(0); source != nullptr) {
					Decompose(source, !value, facts, depth - 1);
				}
				break;
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr: {
				// a && b true, or a || b false, decides both operands.
				if (value != (inst->GetOpcode() == ValueOpcode::LogicalAnd)) {
					break;
				}
				for (size_t index = 0; index < 2; index++) {
					if (const auto* source = operand(index); source != nullptr) {
						Decompose(source, value, facts, depth - 1);
					}
				}
				break;
			}
			default: break;
		}
	}

	void AddFact(const Inst* inst, bool value) {
		if (m_facts.emplace(inst, value).second) {
			m_undo.push_back(inst);
		}
	}

	[[nodiscard]] std::optional<bool> Known(Value value, int depth) const {
		value = value.Resolve();
		if (value.IsImmediate()) {
			return value.GetType() == Type::U1 ? std::optional<bool>(value.U1()) : std::nullopt;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			return std::nullopt;
		}
		if (const auto found = m_facts.find(inst); found != m_facts.end()) {
			return found->second;
		}
		if (depth == 0) {
			return std::nullopt;
		}
		switch (inst->GetOpcode()) {
			case ValueOpcode::LogicalNot:
				if (const auto source = Known(inst->Arg(0), depth - 1)) {
					return !*source;
				}
				return std::nullopt;
			case ValueOpcode::LogicalAnd:
			case ValueOpcode::LogicalOr: {
				const bool is_and = inst->GetOpcode() == ValueOpcode::LogicalAnd;
				const auto a      = Known(inst->Arg(0), depth - 1);
				const auto b      = Known(inst->Arg(1), depth - 1);
				// A false operand decides an AND, a true one an OR.
				if ((a && *a != is_and) || (b && *b != is_and)) {
					return !is_and;
				}
				if (a && b) {
					return is_and;
				}
				return std::nullopt;
			}
			default: return std::nullopt;
		}
	}

	const Program&                                          m_program;
	const DominatorTree&                                    m_dom;
	std::vector<std::vector<std::pair<const Inst*, bool>>> m_edge_facts;
	std::unordered_map<const Inst*, bool>                   m_facts;
	std::vector<const Inst*>                                m_undo;
};

void Replace(Inst& select, Value replacement) {
	// Rewire the users only: the select loses its last use and dead-code elimination drops it.
	const auto uses = select.Uses();
	for (const auto& use: uses) {
		use.user->SetArg(use.operand, replacement);
	}
}

// Select(c, Select(e, t, f), z) with c => e reads t, and Select(c, x, Select(e, t, f)) with
// e => c reads f: the inner select can only take the other arm where the outer one ignores it.
// This drops the old-value chains of VGPRs written several times under the same EXEC.
uint32_t FoldNestedSelects(Program& program, const ExternalReferences& refs) {
	uint32_t folded = 0;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (!IsSelect(inst.GetOpcode()) || refs.frozen.contains(&inst)) {
				continue;
			}
			const auto* condition = ConditionInstruction(inst.Arg(0));
			if (condition == nullptr) {
				continue;
			}
			for (size_t arm = 1; arm <= 2; arm++) {
				for (;;) {
					const auto* nested = inst.Arg(arm).Resolve().TryInstruction();
					if (nested == nullptr || !IsSelect(nested->GetOpcode())) {
						break;
					}
					const auto* nested_condition = ConditionInstruction(nested->Arg(0));
					if (nested_condition == nullptr) {
						break;
					}
					const bool applies =
					    arm == 1 ? Implies(inst.Arg(0), nested_condition, ImplicationDepth)
					             : Implies(nested->Arg(0), condition, ImplicationDepth);
					if (!applies) {
						break;
					}
					inst.SetArg(arm, nested->Arg(arm));
					folded++;
				}
			}
		}
	}
	return folded;
}

// EliminateDeadCode that keeps what the resource plan references.
void EliminateDeadCode(const BlockList& blocks, const ExternalReferences& refs) {
	bool changed;
	do {
		changed = false;
		for (auto block = blocks.rbegin(); block != blocks.rend(); block++) {
			auto& instructions = (*block)->Instructions();
			auto  inst         = instructions.end();
			while (inst != instructions.begin()) {
				--inst;
				if (inst->HasUses() || inst->MayHaveSideEffects() || refs.Keeps(&*inst)) {
					continue;
				}
				inst->Invalidate();
				inst    = instructions.erase(inst);
				changed = true;
			}
		}
	} while (changed);
}

} // namespace

ExecSelectStats EliminateExecSelects(Program& program, bool per_invocation_branches) {
	ExecSelectStats stats;
	const auto      refs      = CollectExternalReferences(program);
	const auto      candidate = [&](const Inst& inst) {
        return IsSelect(inst.GetOpcode()) && inst.HasUses() && !refs.Keeps(&inst);
	};

	const DominatorTree dom(program);
	if (per_invocation_branches) {
		std::vector<std::pair<Inst*, bool>> decided;
		BranchFacts                         facts(program, dom);
		facts.Walk([&](Block* block, const auto& known) {
			for (auto& inst: *block) {
				if (!candidate(inst)) {
					continue;
				}
				if (const auto value = known(inst.Arg(0))) {
					decided.emplace_back(&inst, *value);
				}
			}
		});
		// Arms are read at replacement time: an arm may be a select replaced before it.
		for (const auto& [inst, value]: decided) {
			Replace(*inst, inst->Arg(value ? 1 : 2));
			stats.branch_known++;
		}
	}

	stats.nested_folds = FoldNestedSelects(program, refs);
	if (stats.branch_known + stats.nested_folds != 0u) {
		// Selects left without users would still count as users of their operands below.
		EliminateDeadCode(program.blocks, refs);
	}

	std::vector<std::pair<Inst*, const Inst*>> candidates;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (!candidate(inst)) {
				continue;
			}
			if (const auto* condition = ConditionInstruction(inst.Arg(0)); condition != nullptr) {
				candidates.emplace_back(&inst, condition);
			}
		}
	}
	// Greatest fixpoint: assume every candidate unobserved where its condition is false and drop
	// those with an observation there until the set is stable. Loop phis between members then
	// carry nothing observable. Without convergence, fall back to counting phis as observations.
	std::unordered_set<const Inst*> assumed;
	for (const auto& [inst, condition]: candidates) {
		assumed.insert(inst);
	}
	bool converged = false;
	for (uint32_t pass = 0; pass < MaxFixpointPasses && !converged; pass++) {
		MaskedUseAnalysis analysis(program, dom, &assumed);
		converged = true;
		for (const auto& [inst, condition]: candidates) {
			if (assumed.contains(inst) && !analysis.ObservedOnlyUnder(inst, condition)) {
				assumed.erase(inst);
				converged = false;
			}
		}
	}
	if (!converged) {
		assumed.clear();
		MaskedUseAnalysis analysis(program, dom, nullptr);
		for (const auto& [inst, condition]: candidates) {
			if (analysis.ObservedOnlyUnder(inst, condition)) {
				assumed.insert(inst);
			}
		}
	}
	for (const auto& [inst, condition]: candidates) {
		if (assumed.contains(inst)) {
			Replace(*inst, inst->Arg(1));
			stats.masked_uses++;
		}
	}
	if (stats.masked_uses != 0u) {
		EliminateDeadCode(program.blocks, refs);
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
