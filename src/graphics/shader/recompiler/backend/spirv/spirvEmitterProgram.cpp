#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayTracing.h"
#include "graphics/shader/recompiler/ir/passes/BindingLayout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <functional>
#include <optional>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

// KYTY_OBJPRIM_ID (based on Erlou93/KytyPS5-Yotei): what the hardware writes to the ObjPrimID colour
// slot, VGT_OBJECT_ID (push-constant dword) plus the primitive index of the draw when
// ADD_PIPED_PRIM_ID is set. Stored in the entry block: a fragment the guest code discards writes
// nothing, like any colour output.
void EmitObjprimIdOutput(EmitterState& state) {
	const auto& pixel   = *state.input_info.pixel;
	const auto  pointer = state.builder.AllocateId();
	auto        value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 0),
	                          ConstantU32(state, pixel.objprim_push_dword));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	if (pixel.objprim_add_prim_id) {
		const auto variable = InputVariableForKind(state, IR::StageInputKind::PrimitiveId);
		EXIT_IF(variable == 0);
		const auto primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeI32(state), primitive, variable);
		const auto primitive_u32 = Unary(state, spv::OpBitcast, TypeU32(state), primitive);
		value = Binary(state, spv::OpIAdd, TypeU32(state), value, primitive_u32);
	}
	state.builder.AddFunction(spv::OpStore, state.objprim_output_variable, value);
}

void EmitKillIfBoolFalse(EmitterState& state, uint32_t active) {
	const auto kill_label  = state.builder.AllocateId();
	const auto merge_label = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, active, merge_label, kill_label);
	EmitLabel(state, kill_label);
	state.builder.AddFunction(spv::OpKill);
	EmitLabel(state, merge_label);
}

void EmitKillIfPixelValidMaskInactive(EmitterState& state) {
	if (state.pixel_valid_mask_variable == 0) {
		return;
	}

	const auto mask_value = state.builder.AllocateId();
	const auto active     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), mask_value,
	                          state.pixel_valid_mask_variable);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, mask_value,
	                          ConstantU32(state, 0));
	EmitKillIfBoolFalse(state, active);
}

uint32_t SpillPointerType(ValueEmitContext& ctx, IR::Type type) {
	const auto value_type = TypeId(ctx.state, type);
	return value_type == 0 ? 0 : TypePointer(ctx.state, spv::StorageClassFunction, value_type);
}

struct DeferredPhiPatch {
	DeferredPhi     phi;
	const IR::Inst* instruction = nullptr;
	uint32_t        half        = 0;
};

// An incoming edge of a bool phi from a block emitted at or after the phi's own block (a loop's
// back edge); see CollectLateBoolPhiEdges.
struct LateBoolPhiEdge {
	const IR::Inst* phi   = nullptr;
	size_t          index = 0;
};

struct StructuredFunctionState {
	std::unordered_map<const IR::Block*, uint32_t> block_exit_labels;
	std::vector<DeferredPhiPatch>                  deferred_phis;
	// The late bool phi edges by predecessor, and the copy each phi takes on them (by phi, then
	// incoming index and half; 0 for the other edges).
	std::unordered_map<const IR::Block*, std::vector<LateBoolPhiEdge>>        late_bool_phi_edges;
	std::unordered_map<const IR::Inst*, std::vector<std::array<uint32_t, 2>>> late_bool_phi_copies;
};

struct DispatcherFunctionState {
	std::array<std::unordered_map<const IR::Inst*, uint32_t>, 2> spills;
	uint32_t                                      header_label       = 0;
	uint32_t                                      select_label       = 0;
	uint32_t                                      after_switch_label = 0;
	uint32_t                                      continue_label     = 0;
	uint32_t                                      merge_label        = 0;
};

void StoreDispatcherPhiEdge(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                            const IR::Block* from, const IR::Block* to) {
	if (to == nullptr) {
		return;
	}
	for (const auto& phi: *to) {
		if (phi.GetOpcode() != IR::ValueOpcode::Phi) {
			break;
		}
		for (size_t index = 0; index < phi.NumArgs(); index++) {
			if (phi.PhiBlock(index) == from) {
				ctx.state.builder.AddFunction(spv::OpStore, dispatcher.spills[ctx.half].at(&phi),
				                              ctx.Def(phi.Arg(index)));
				break;
			}
		}
	}
}

const IR::Block* TargetBlock(const IR::Program& program, uint32_t id) {
	const auto found = std::ranges::find_if(
	    program.block_info, [&](const IR::BlockInfo& info) { return info.id == id; });
	if (found == program.block_info.end()) {
		return nullptr;
	}
	return program.blocks[static_cast<size_t>(found - program.block_info.begin())];
}

// KYTY_LOOP_GUARD: true once the invocation has run more iterations than the budget.
uint32_t LoopGuardExhausted(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.loop_guard_variable);
	const auto exhausted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThan, TypeBool(state), exhausted, count,
	                          ConstantU32(state, GetCodegenOptions().loop_guard_budget));
	return exhausted;
}

void SpendLoopGuard(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.loop_guard_variable);
	state.builder.AddFunction(spv::OpStore, state.loop_guard_variable,
	                          EmitAddU32(state, count, ConstantU32(state, 1)));
}

// KYTY_RT_NODE_BUDGET: the node tests one invocation may run (both halves of a two-lane invocation
// count), or 0 without a budget.
uint32_t BvhNodeLimit(const EmitterState& state) {
	const uint64_t budget = GetCodegenOptions().rt_node_budget;
	return static_cast<uint32_t>(std::min<uint64_t>(budget * state.lane_count, UINT32_MAX - 1u));
}

// True once a node test of the invocation was refused (the count passed the limit).
uint32_t BvhNodeBudgetExhausted(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.bvh_node_count_variable);
	const auto exhausted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThan, TypeBool(state), exhausted, count,
	                          ConstantU32(state, BvhNodeLimit(state)));
	return exhausted;
}

bool BvhNodeBudgetActive(const EmitterState& state) {
	return state.bvh_node_count_variable != 0 && GetCodegenOptions().rt_node_budget != 0;
}

// KYTY_LOOP_GUARD or KYTY_RT_NODE_BUDGET: whether this invocation leaves its loops.
uint32_t GuardExhausted(EmitterState& state) {
	uint32_t exhausted = 0;
	if (state.loop_guard_variable != 0) {
		exhausted = LoopGuardExhausted(state);
	}
	if (BvhNodeBudgetActive(state)) {
		const auto bvh = BvhNodeBudgetExhausted(state);
		if (exhausted == 0) {
			exhausted = bvh;
		} else {
			const auto both = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), both, exhausted, bvh);
			exhausted = both;
		}
	}
	return exhausted;
}

uint32_t GdsDwordFromEnd(EmitterState& state, uint32_t from_end) {
	const auto index =
	    EmitBinaryU32(state, spv::OpISub, state.gds_length, ConstantU32(state, from_end));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.gds_variable, ConstantU32(state, 0), index);
	return pointer;
}

void AddToGds(EmitterState& state, uint32_t pointer) {
	const auto previous = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), previous, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone),
	                          ConstantU32(state, 1));
}

// At return: an invocation that exhausted KYTY_RT_NODE_BUDGET adds one to its GDS dword, and with
// KYTY_RT_NODE_STATS an invocation that ran node tests adds one to its count's histogram bin.
void EmitBvhNodeReport(EmitterState& state) {
	if (state.bvh_node_count_variable == 0 || state.gds_variable == 0 || state.gds_length == 0) {
		return;
	}
	if (BvhNodeBudgetActive(state)) {
		const auto exhausted    = BvhNodeBudgetExhausted(state);
		const auto report_label = state.builder.AllocateId();
		const auto merge_label  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
		state.builder.AddFunction(spv::OpBranchConditional, exhausted, report_label, merge_label);
		EmitLabel(state, report_label);
		AddToGds(state, GdsDwordFromEnd(state, RtNodeBudgetGdsFromEnd));
		state.builder.AddFunction(spv::OpBranch, merge_label);
		EmitLabel(state, merge_label);
	}
	if (GetCodegenOptions().rt_node_stats) {
		const auto count = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.bvh_node_count_variable);
		// Per guest lane: both halves of a two-lane invocation count.
		const auto per_lane =
		    state.lane_count == 2
		        ? EmitBinaryU32(state, spv::OpShiftRightLogical, count, ConstantU32(state, 1))
		        : count;
		const auto ran = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), ran, per_lane,
		                          ConstantU32(state, 0));
		const auto report_label = state.builder.AllocateId();
		const auto merge_label  = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
		state.builder.AddFunction(spv::OpBranchConditional, ran, report_label, merge_label);
		EmitLabel(state, report_label);
		const auto msb = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), msb, GlslStd450(state),
		                          GLSLstd450FindUMsb, per_lane);
		const auto last_bin = ConstantU32(state, RtNodeStatsBins - 1u);
		const auto clamp    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), clamp, GlslStd450(state),
		                          GLSLstd450UMin, msb, last_bin);
		const auto from_end =
		    EmitBinaryU32(state, spv::OpIAdd, clamp, ConstantU32(state, RtNodeStatsGdsFromEnd));
		const auto index   = EmitBinaryU32(state, spv::OpISub, state.gds_length, from_end);
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state),
		                          pointer, state.gds_variable, ConstantU32(state, 0), index);
		AddToGds(state, pointer);
		state.builder.AddFunction(spv::OpBranch, merge_label);
		EmitLabel(state, merge_label);
	}
}

bool IsLoopMergeBlock(const IR::Program& program, uint32_t target) {
	return std::ranges::any_of(program.block_info, [=](const IR::BlockInfo& info) {
		return info.terminator.loop_header && info.terminator.merge_block == target;
	});
}

// KYTY_CALL_LOOP_CAP: the merge of a loop of a bracketed callee body (CFG::Terminator::call_loop).
bool IsCallLoopMergeBlock(const IR::Program& program, uint32_t target) {
	return std::ranges::any_of(program.block_info, [=](const IR::BlockInfo& info) {
		return info.terminator.loop_header && info.terminator.call_loop && info.terminator.merge_block == target;
	});
}

// True once the invocation has run more rounds of such loops than the cap.
uint32_t CallLoopExhausted(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.call_loop_variable);
	const auto exhausted = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpUGreaterThan, TypeBool(state), exhausted, count,
	                          ConstantU32(state, GetCodegenOptions().call_loop_cap));
	return exhausted;
}

void SpendCallLoop(EmitterState& state) {
	const auto count = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), count, state.call_loop_variable);
	state.builder.AddFunction(spv::OpStore, state.call_loop_variable,
	                          EmitAddU32(state, count, ConstantU32(state, 1)));
}

// An exhausted invocation reports once, at return, by adding one to the last GDS dword.
void EmitLoopGuardReport(EmitterState& state) {
	if (state.loop_guard_variable == 0 || state.gds_variable == 0 || state.gds_length == 0) {
		return;
	}
	const auto exhausted    = LoopGuardExhausted(state);
	const auto report_label = state.builder.AllocateId();
	const auto merge_label  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, exhausted, report_label, merge_label);
	EmitLabel(state, report_label);
	const auto last    = EmitBinaryU32(state, spv::OpISub, state.gds_length, ConstantU32(state, 1));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.gds_variable, ConstantU32(state, 0), last);
	const auto previous = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), previous, pointer,
	                          ConstantU32(state, spv::ScopeDevice),
	                          ConstantU32(state, spv::MemorySemanticsMaskNone),
	                          ConstantU32(state, 1));
	// Which shader: its hash, low dword at LoopGuardHashGdsFromEnd, high dword in the next one.
	const auto hash = state.program.shader_hash;
	for (uint32_t word = 0; word < 2; word++) {
		const auto replaced = state.builder.AllocateId();
		state.builder.AddFunction(
		    spv::OpAtomicExchange, TypeU32(state), replaced,
		    GdsDwordFromEnd(state, LoopGuardHashGdsFromEnd - word), ConstantU32(state, spv::ScopeDevice),
		    ConstantU32(state, spv::MemorySemanticsMaskNone),
		    ConstantU32(state, static_cast<uint32_t>(hash >> (32u * word))));
	}
	state.builder.AddFunction(spv::OpBranch, merge_label);
	EmitLabel(state, merge_label);
}

void EmitReturn(ValueEmitContext& ctx) {
	EmitLoopGuardReport(ctx.state);
	EmitBvhNodeReport(ctx.state);
	if (PopsInterlockActive(ctx.state)) {
		ctx.state.builder.AddFunction(spv::OpEndInvocationInterlockEXT);
	}
	EmitKillIfPixelValidMaskInactive(ctx.state);
	ctx.state.builder.AddFunction(spv::OpReturn);
}

uint32_t BranchCondition(ValueEmitContext& ctx, const IR::BlockInfo& info) {
	const auto kind = info.terminator.condition;
	const bool zero = kind == CFG::BranchCondition::ExecZero ||
	                  kind == CFG::BranchCondition::VccZero || kind == CFG::BranchCondition::SccZero;
	if (ctx.state.split_wave && kind != CFG::BranchCondition::ScalarInstruction &&
	    kind != CFG::BranchCondition::GotoVariable &&
	    (ctx.state.wave_probe || ctx.state.wave_local_branches.contains(ctx.state.current_block))) {
		// No block this branch controls synchronizes the halves, so each half branches on its own
		// lanes: a half whose lanes are all inactive skips code the guest ran with them masked.
		const auto word = EmitHostBallotWord(ctx.state, ctx.Def(info.condition));
		return Binary(ctx.state, zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(ctx.state), word,
		              ConstantU32(ctx.state, zero ? ~0u : 0u));
	}
	// KYTY_NATIVE_WAVE64_BRANCH (Senaxx 49723097f): a wave64 workgroup shader on a 64-wide host
	// subgroup (compute and mesh shaders on AMD) runs natively, one guest lane per invocation. Its
	// VCC/EXEC/SCC branch decides for the wave, as the two emulated halves below do; a lane's own
	// bit would split the wave. Lanes a partial workgroup lacks are not in the ballot, so a zero
	// condition holds when no present lane fails it.
	if (ctx.other_half == nullptr && !ctx.state.split_wave &&
	    kind != CFG::BranchCondition::ScalarInstruction &&
	    kind != CFG::BranchCondition::GotoVariable && GetCodegenOptions().native_wave64_branch &&
	    ctx.state.program.wave_size == 64u) {
		const auto* workgroup = ShaderWorkgroupInput(ctx.state.program.stage, ctx.state.input_info);
		if (workgroup != nullptr && workgroup->host_subgroup_size == 64u) {
			auto& state     = ctx.state;
			auto  predicate = ctx.Def(info.condition);
			if (zero) {
				predicate = Unary(state, spv::OpLogicalNot, TypeBool(state), predicate);
			}
			state.builder.RequireCapability(spv::CapabilityGroupNonUniform);
			state.builder.RequireCapability(spv::CapabilityGroupNonUniformBallot);
			const auto ballot = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
			                          ConstantU32(state, spv::ScopeSubgroup), predicate);
			const auto low  = state.builder.AllocateId();
			const auto high = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
			const auto any = Binary(state, spv::OpBitwiseOr, TypeU32(state), low, high);
			return Binary(state, zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(state), any,
			              ConstantU32(state, 0));
		}
	}
	// Scalar-instruction conditions already test the full wave's raw register values.
	if ((ctx.other_half == nullptr && !ctx.state.split_wave) ||
	    info.terminator.condition == CFG::BranchCondition::ScalarInstruction ||
	    info.terminator.condition == CFG::BranchCondition::GotoVariable) {
		return ctx.Def(info.condition);
	}
	const auto ballot = ctx.Ballot(info.condition);
	const auto low    = ctx.state.builder.AllocateId();
	const auto high   = ctx.state.builder.AllocateId();
	const auto result = ctx.state.builder.AllocateId();
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), low, ballot, 0);
	ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), high, ballot, 1);
	if (zero && ctx.other_half != nullptr && !ctx.state.split_wave &&
	    GetCodegenOptions().wave64_active_zero) {
		// KYTY_WAVE64_ACTIVE_ZERO: "zero" holds when every running host invocation agrees in both
		// halves; invocations that are not running have no ballot bit (CodegenOptions.h).
		const auto active      = ctx.Ballot(IR::Value(true));
		const auto active_word = ctx.state.builder.AllocateId();
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), active_word,
		                              active, 0);
		const auto low_all  = Binary(ctx.state, spv::OpIEqual, TypeBool(ctx.state), low, active_word);
		const auto high_all = Binary(ctx.state, spv::OpIEqual, TypeBool(ctx.state), high, active_word);
		return Binary(ctx.state, spv::OpLogicalAnd, TypeBool(ctx.state), low_all, high_all);
	}
	const auto combined =
	    EmitBinaryU32(ctx.state, zero ? spv::OpBitwiseAnd : spv::OpBitwiseOr, low, high);
	ctx.state.builder.AddFunction(zero ? spv::OpIEqual : spv::OpINotEqual, TypeBool(ctx.state),
	                              result, combined, ConstantU32(ctx.state, zero ? ~0u : 0u));
	return result;
}

void EmitStructuredTerminator(ValueEmitContext& ctx, const IR::Block* block,
                              const IR::BlockInfo& info) {
	const auto& program = ctx.state.program;
	const auto& term       = info.terminator;
	const bool  degenerate_branch = term.kind == CFG::TerminatorKind::ConditionalBranch &&
	                               !term.loop_header && term.true_block == term.false_block;
	const auto  emit_merge = [&]() {
		if (term.loop_header) {
			if (ctx.state.loop_guard_variable != 0) {
				SpendLoopGuard(ctx.state);
			}
			if (ctx.state.call_loop_variable != 0 && term.call_loop) {
				SpendCallLoop(ctx.state);
			}
			const auto* merge = TargetBlock(program, term.merge_block);
			const auto* cont  = TargetBlock(program, term.continue_block);
			if (merge != nullptr && cont != nullptr) {
				ctx.state.builder.AddFunction(spv::OpLoopMerge, ctx.Label(merge), ctx.Label(cont),
				                              spv::LoopControlMaskNone);
			}
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           term.merge_block != UINT32_MAX) {
			if (const auto* merge = TargetBlock(program, term.merge_block); merge != nullptr) {
				ctx.state.builder.AddFunction(spv::OpSelectionMerge, ctx.Label(merge),
				                              spv::SelectionControlMaskNone);
			}
		}
	};

	switch (term.kind) {
		case CFG::TerminatorKind::Branch: {
			const auto* target = TargetBlock(program, term.true_block);
			if (target == nullptr) {
				EmitReturn(ctx);
				return;
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(target));
			return;
		}
		case CFG::TerminatorKind::ConditionalBranch: {
			const auto* true_block  = TargetBlock(program, term.true_block);
			const auto* false_block = TargetBlock(program, term.false_block);
			if (true_block == nullptr || false_block == nullptr || info.condition.IsEmpty()) {
				EmitReturn(ctx);
				return;
			}
			if (degenerate_branch) {
				// SPIRV-Cross discards all code after a selection whose branch targets equal its merge.
				ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(true_block));
				return;
			}
			auto condition = BranchCondition(ctx, info);
			if (ctx.state.loop_guard_variable != 0 || BvhNodeBudgetActive(ctx.state)) {
				// KYTY_LOOP_GUARD, KYTY_RT_NODE_BUDGET: an exhausted invocation takes whichever edge
				// leaves a loop.
				const bool true_exits  = IsLoopMergeBlock(program, term.true_block);
				const bool false_exits = IsLoopMergeBlock(program, term.false_block);
				if (true_exits != false_exits) {
					const auto exhausted = GuardExhausted(ctx.state);
					const auto forced    = ctx.state.builder.AllocateId();
					if (true_exits) {
						ctx.state.builder.AddFunction(spv::OpLogicalOr, TypeBool(ctx.state), forced,
						                              condition, exhausted);
					} else {
						const auto alive = ctx.state.builder.AllocateId();
						ctx.state.builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), alive,
						                              exhausted);
						ctx.state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(ctx.state), forced,
						                              condition, alive);
					}
					condition = forced;
				}
			}
			if (ctx.state.call_loop_variable != 0) {
				// KYTY_CALL_LOOP_CAP: an exhausted invocation leaves the bracketed bodies' loops.
				const bool true_exits  = IsCallLoopMergeBlock(program, term.true_block);
				const bool false_exits = IsCallLoopMergeBlock(program, term.false_block);
				if (true_exits != false_exits) {
					const auto exhausted = CallLoopExhausted(ctx.state);
					const auto forced    = ctx.state.builder.AllocateId();
					if (true_exits) {
						ctx.state.builder.AddFunction(spv::OpLogicalOr, TypeBool(ctx.state), forced, condition,
						                              exhausted);
					} else {
						const auto alive = ctx.state.builder.AllocateId();
						ctx.state.builder.AddFunction(spv::OpLogicalNot, TypeBool(ctx.state), alive, exhausted);
						ctx.state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(ctx.state), forced, condition,
						                              alive);
					}
					condition = forced;
				}
			}
			emit_merge();
			ctx.state.builder.AddFunction(spv::OpBranchConditional, condition,
			                              ctx.Label(true_block), ctx.Label(false_block));
			return;
		}
		default: EmitReturn(ctx); return;
	}
}

void EmitDispatcherTarget(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                          const IR::Block* from, uint32_t target) {
	const auto* block = TargetBlock(ctx.state.program, target);
	if (block != nullptr) {
		StoreDispatcherPhiEdge(ctx, dispatcher, from, block);
		if (ctx.other_half != nullptr) {
			StoreDispatcherPhiEdge(*ctx.other_half, dispatcher, from, block);
		}
	}
}

uint32_t EmitDispatcherNextPc(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                              const IR::Block* block, const IR::BlockInfo& info) {
	const auto& term = info.terminator;
	switch (term.kind) {
		case CFG::TerminatorKind::Branch:
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			return ConstantU32(ctx.state, term.true_block);
		case CFG::TerminatorKind::ConditionalBranch: {
			EmitDispatcherTarget(ctx, dispatcher, block, term.true_block);
			EmitDispatcherTarget(ctx, dispatcher, block, term.false_block);
			const auto selected = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(
			    spv::OpSelect, TypeU32(ctx.state), selected, BranchCondition(ctx, info),
			    ConstantU32(ctx.state, term.true_block), ConstantU32(ctx.state, term.false_block));
			return selected;
		}
		case CFG::TerminatorKind::IndirectBranch: {
			for (const auto target: term.indirect_targets) {
				EmitDispatcherTarget(ctx, dispatcher, block, target);
			}
			uint32_t selected = ConstantU32(ctx.state, UINT32_MAX);
			if (!info.indirect_target.IsEmpty()) {
				const auto  selector = ctx.Def(info.indirect_target);
				const auto& values   = term.indirect_selector_code != UINT32_MAX
				                           ? term.indirect_selector_values
				                           : term.indirect_target_pcs;
				const auto& targets  = term.indirect_selector_code != UINT32_MAX
				                           ? term.indirect_selector_targets
				                           : term.indirect_targets;
				for (size_t index = 0; index < std::min(values.size(), targets.size()); index++) {
					const auto match = ctx.state.builder.AllocateId();
					const auto next  = ctx.state.builder.AllocateId();
					ctx.state.builder.AddFunction(spv::OpIEqual, TypeBool(ctx.state), match,
					                              selector, ConstantU32(ctx.state, values[index]));
					ctx.state.builder.AddFunction(spv::OpSelect, TypeU32(ctx.state), next, match,
					                              ConstantU32(ctx.state, targets[index]), selected);
					selected = next;
				}
			}
			return selected;
		}
		default: return ConstantU32(ctx.state, UINT32_MAX);
	}
}

template <typename T>
decltype(auto) Arg(ValueEmitContext& ctx, const IR::Inst& inst, size_t index) {
	if constexpr (std::is_same_v<T, const IR::Inst&>) {
		return inst;
	} else if constexpr (std::is_same_v<T, IR::Value>) {
		return inst.Arg(index);
	} else if constexpr (std::is_same_v<T, IR::ScalarReg>) {
		return inst.Arg(index).ScalarRegister();
	} else {
		static_assert(std::is_same_v<T, uint32_t>);
		return ctx.Def(inst.Arg(index));
	}
}

template <typename Context, typename Return, typename... Args>
void Invoke(Return (*emit)(Context&, Args...), ValueEmitContext& ctx, const IR::Inst& inst) {
	// A full instruction keeps metadata and predicated/lane operand loads lazy.
	static_assert(std::is_same_v<Context, ValueEmitContext> ||
	              std::is_same_v<Context, EmitterState>);
	auto& context = [&]() -> Context& {
		if constexpr (std::is_same_v<Context, EmitterState>)
			return ctx.state;
		else
			return ctx;
	}();
	constexpr bool has_inst = (std::is_same_v<Args, const IR::Inst&> || ...);
	[&]<size_t... I>(std::index_sequence<I...>) {
		static_assert(((!std::is_same_v<Args, const IR::Inst&> || I == 0) && ...));
		const auto call = [&] {
			return emit(context, Arg<Args>(ctx, inst, I - (has_inst && I != 0))...);
		};
		if constexpr (std::is_void_v<Return>) {
			call();
		} else {
			static_assert(std::is_same_v<Return, uint32_t>);
			ctx.Define(inst, call());
		}
	}(std::index_sequence_for<Args...> {});
}

// KYTY_MOVREL_SWITCH (from BryanKAdams/KytyPS5 55a7540). A link select(index == constant, value,
// rest) of an indexed select chain: a V_MOVRELS read is one chain over every register above the
// source, on a uniform index (M0 & 0xff).
struct IndexedSelectLink {
	IR::Value index;
	uint32_t  constant = 0;
	IR::Value value;
	IR::Value rest;
};

bool IndexedSelectLinkOf(const IR::Inst& inst, IndexedSelectLink& link) {
	if (inst.GetOpcode() != IR::ValueOpcode::SelectU32 || inst.NumArgs() != 3) {
		return false;
	}
	const auto* compare = inst.Arg(0).Resolve().TryInstruction();
	if (compare == nullptr || compare->GetOpcode() != IR::ValueOpcode::IEqual32) {
		return false;
	}
	auto index    = compare->Arg(0).Resolve();
	auto constant = compare->Arg(1).Resolve();
	if (index.IsImmediate()) {
		std::swap(index, constant);
	}
	if (index.IsImmediate() || !constant.IsImmediate() || constant.GetType() != IR::Type::U32) {
		return false;
	}
	link = {index, constant.U32(), inst.Arg(1), inst.Arg(2)};
	return true;
}

// The link below `inst` on the same index, when only `inst` uses it and it is in the same block.
const IR::Inst* NextIndexedSelect(const IR::Inst& inst, const IndexedSelectLink& link,
                                  IndexedSelectLink& next) {
	const auto* rest = link.rest.Resolve().TryInstruction();
	if (rest == nullptr || rest->Parent() != inst.Parent() || rest->UseCount() != 1 ||
	    !IndexedSelectLinkOf(*rest, next) || !(next.index == link.index)) {
		return nullptr;
	}
	return rest;
}

// Shorter chains stay selects.
constexpr size_t MinIndexedSelects = 16;

void FindIndexedSelects(EmitterState& state) {
	state.indexed_selects_found = true;
	if (!GetCodegenOptions().movrel_switch) {
		return;
	}
	std::unordered_set<const IR::Inst*> inner;
	IndexedSelectLink                   link;
	IndexedSelectLink                   next;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (IndexedSelectLinkOf(inst, link)) {
				if (const auto* rest = NextIndexedSelect(inst, link, next)) {
					inner.insert(rest);
				}
			}
		}
	}
	std::vector<const IR::Inst*> members;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inner.contains(&inst) || !IndexedSelectLinkOf(inst, link)) {
				continue;
			}
			members.clear();
			for (const auto* current = &inst;;) {
				const auto* rest = NextIndexedSelect(*current, link, next);
				if (rest == nullptr) {
					break;
				}
				members.push_back(rest);
				current = rest;
				link    = next;
			}
			if (members.size() + 1u >= MinIndexedSelects) {
				state.indexed_select_heads.insert(&inst);
				state.indexed_select_members.insert(members.begin(), members.end());
			}
		}
	}
	// The compares of converted links emit nothing either, unless something else uses them.
	const auto converted = [&](const IR::Inst* select) {
		return state.indexed_select_heads.contains(select) ||
		       state.indexed_select_members.contains(select);
	};
	for (const auto* set: {&state.indexed_select_heads, &state.indexed_select_members}) {
		for (const auto* select: *set) {
			const auto* compare = select->Arg(0).Resolve().TryInstruction();
			if (compare != nullptr && compare->HasUses() &&
			    std::ranges::all_of(compare->Uses(), [&](const IR::Use& use) {
				    return use.operand == 0u && converted(use.user);
			    })) {
				state.indexed_select_compares.insert(compare);
			}
		}
	}
}

// Emits a long indexed select chain at its outermost select as an OpSwitch on index / 16 into
// short select chains: the host branches on the uniform index and runs one chain of at most 16
// selects, while the code stays about as large as the whole chain (one case per index value
// compiled about twice as slowly on AMD, for little more speed). Returns false for other
// instructions; the chain's inner selects emit nothing.
bool EmitIndexedSelect(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& state = ctx.state;
	if (!state.indexed_selects_found) {
		FindIndexedSelects(state);
	}
	if (state.indexed_select_members.contains(&inst) ||
	    state.indexed_select_compares.contains(&inst)) {
		return true;
	}
	if (!state.indexed_select_heads.contains(&inst)) {
		return false;
	}
	IndexedSelectLink link;
	IndexedSelectLinkOf(inst, link);
	const auto selector = ctx.Def(link.index);
	// Outer links take precedence over inner ones with the same constant, as in the chain.
	std::vector<std::pair<uint32_t, uint32_t>> cases; // Constant, value.
	std::unordered_set<uint32_t>               constants;
	for (const auto* current = &inst;;) {
		if (constants.insert(link.constant).second) {
			cases.emplace_back(link.constant, ctx.Def(link.value));
		}
		IndexedSelectLink next;
		const auto*       rest = NextIndexedSelect(*current, link, next);
		if (rest == nullptr) {
			break;
		}
		current = rest;
		link    = next;
	}
	const auto         fallback   = ctx.Def(link.rest);
	constexpr uint32_t BucketBits = 4;
	std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> buckets;
	for (const auto& [constant, value]: cases) {
		if (value != fallback) {
			buckets[constant >> BucketBits].emplace_back(constant, value);
		}
	}
	if (buckets.empty()) {
		ctx.Define(inst, fallback);
		return true;
	}
	const auto bucket_selector =
	    EmitBinaryU32(state, spv::OpShiftRightLogical, selector, ConstantU32(state, BucketBits));
	const auto            default_label = state.builder.AllocateId();
	const auto            merge_label   = state.builder.AllocateId();
	std::vector<uint32_t> labels;
	std::vector<uint32_t> switch_words {spv::OpSwitch, bucket_selector, default_label};
	for (const auto& [bucket, entries]: buckets) {
		labels.push_back(state.builder.AllocateId());
		switch_words.push_back(bucket);
		switch_words.push_back(labels.back());
	}
	state.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
	state.builder.AddFunction(switch_words);
	std::vector<uint32_t> phi_words {spv::OpPhi, TypeU32(state), state.builder.AllocateId(),
	                                 fallback, default_label};
	EmitLabel(state, default_label);
	state.builder.AddFunction(spv::OpBranch, merge_label);
	size_t label_index = 0;
	for (const auto& [bucket, entries]: buckets) {
		const auto label = labels[label_index++];
		EmitLabel(state, label);
		// The constants are distinct, so the order of the selects does not matter.
		auto selected = fallback;
		for (const auto& [constant, value]: entries) {
			const auto match = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpIEqual, TypeBool(state), match, selector,
			                          ConstantU32(state, constant));
			const auto chosen = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpSelect, TypeU32(state), chosen, match, value, selected);
			selected = chosen;
		}
		state.builder.AddFunction(spv::OpBranch, merge_label);
		phi_words.push_back(selected);
		phi_words.push_back(label);
	}
	EmitLabel(state, merge_label);
	state.builder.AddFunction(phi_words);
	ctx.Define(inst, phi_words[2]);
	return true;
}

void EmitDirectInstruction(ValueEmitContext& ctx, const IR::Inst& inst) {
	if (ctx.half != 0 &&
	    (inst.GetOpcode() == IR::ValueOpcode::Ballot ||
	     inst.GetOpcode() == IR::ValueOpcode::ReadFirstLane ||
	     // Native ray tracing: the wave's next instance is chosen once for both halves.
	     inst.GetOpcode() == IR::ValueOpcode::RayQueryNext ||
	     // Upstream 8cc5f5364: SMEM loads (SGPR operands, no EXEC) are identical in both halves.
	     inst.GetOpcode() == IR::ValueOpcode::ReadConstBuffer ||
	     (inst.GetOpcode() == IR::ValueOpcode::LoadAddressU32 &&
	      ctx.Memory(inst).kind == IR::ResourceKind::ScalarAddress))) {
		// Both operations already combine both emulated halves into one whole-wave result.
		ctx.Define(inst, ctx.other_half->Result(inst));
		return;
	}
	if (ctx.half != 0 && inst.GetOpcode() == IR::ValueOpcode::RayQueryCount) {
		// Counted once for both halves (EmitRayQueryCount).
		return;
	}
	if ((inst.GetOpcode() == IR::ValueOpcode::SelectU32 ||
	     inst.GetOpcode() == IR::ValueOpcode::IEqual32) &&
	    EmitIndexedSelect(ctx, inst)) {
		return;
	}
	switch (inst.GetOpcode()) {
#define VALUE_OPCODE(name, ...)                                                                    \
	case IR::ValueOpcode::name: return Invoke(Emit##name, ctx, inst);
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.inc"
#undef VALUE_OPCODE
		default: ctx.Fail(inst, "has no direct SPIR-V emitter");
	}
}

void EmitStructuredInstruction(ValueEmitContext& ctx, StructuredFunctionState& structured,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0 || inst.NumArgs() == 0) {
			ctx.Fail(inst, "has no native SPIR-V representation");
		}
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			const auto* predecessor = inst.PhiBlock(index);
			if (predecessor == nullptr || !ctx.state.labels.contains(predecessor)) {
				ctx.Fail(inst, "has a predecessor outside the structured function");
			}
		}
		structured.deferred_phis.push_back(
		    {ctx.state.builder.AddDeferredPhi(type, ctx.Result(inst), inst.NumArgs()), &inst,
		     ctx.half});
		return;
	}
	EmitDirectInstruction(ctx, inst);
}

void EmitDispatcherInstruction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher,
                               const IR::Inst& inst) {
	if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
		const auto type = TypeId(ctx.state, inst.GetType());
		if (type == 0) {
			ctx.Fail(inst, "cannot be loaded by the dispatcher");
		}
		ctx.state.builder.AddFunction(spv::OpLoad, type, ctx.Result(inst),
		                              dispatcher.spills[ctx.half].at(&inst));
		return;
	}
	EmitDirectInstruction(ctx, inst);
	if (const auto found = dispatcher.spills[ctx.half].find(&inst);
	    found != dispatcher.spills[ctx.half].end()) {
		ctx.state.builder.AddFunction(spv::OpStore, found->second,
		                              ctx.Def(IR::Value(const_cast<IR::Inst*>(&inst))));
	}
}

// Whether the instruction accesses the workgroup's LDS through a DS instruction.
bool AccessesLds(const IR::Program& program, const IR::Inst& inst) {
	if (IR::SharedAccessOf(inst.GetOpcode()) == IR::SharedAccess::None) {
		return false;
	}
	const auto index = inst.Flags<IR::MemoryFlags>().index;
	return index < program.memory_info.size() &&
	       program.memory_info[index].kind == IR::ResourceKind::Lds;
}

// Whether a split wave's halves must synchronize before the instruction: an LDS access that
// another lane's LDS write may conflict with.
bool SyncsLdsAccess(const EmitterState& state, const IR::Inst& inst) {
	return AccessesLds(state.program, inst) &&
	       (state.ds_writes_lds || IR::SharedAccessOf(inst.GetOpcode()) != IR::SharedAccess::Read);
}

template <typename EmitInstruction>
void EmitBlock(ValueEmitContext& ctx, const IR::Block* block, EmitInstruction&& emit_instruction) {
	ctx.state.current_block = block;
	EmitLabel(ctx.state, ctx.Label(block));
	// KYTY_SPLIT_LDS_WRITE_SYNC: a read may reach this block on another path.
	ctx.state.lds_split_synced = false;
	bool emitted_non_phi = false;
	for (const auto& inst: *block) {
		if (inst.GetOpcode() == IR::ValueOpcode::Phi) {
			if (emitted_non_phi) {
				ctx.Fail(inst, "appears after a non-Phi instruction");
			}
		} else {
			emitted_non_phi = true;
		}
		const bool lds_write_sync = ctx.state.split_wave && GetCodegenOptions().split_lds_write_sync &&
		                            AccessesLds(ctx.state.program, inst);
		const bool lds_write = lds_write_sync && IR::SharedAccessOf(inst.GetOpcode()) != IR::SharedAccess::Read;
		if (lds_write_sync) {
			// KYTY_SPLIT_LDS_WRITE_SYNC: a write waits for the other half only when this half read the
			// LDS since the last barrier (it may overwrite what the other half has yet to read).
			if (lds_write && !ctx.state.lds_split_synced) {
				EmitWaveBarrier(ctx.state);
			}
		} else if (ctx.state.split_wave && SyncsLdsAccess(ctx.state, inst)) {
			// The guest's lanes ran in lockstep: a lane may read what another lane of its wave
			// wrote, or overwrite what it read, without waiting. A split wave's halves do not.
			EmitWaveBarrier(ctx.state);
		}
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto& lane          = half == 0 ? ctx : *ctx.other_half;
			ctx.state.lane_half = half;
			if (half == 0 || (inst.GetOpcode() != IR::ValueOpcode::Barrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::SharedMemoryBarrier &&
			                  inst.GetOpcode() != IR::ValueOpcode::StoreCompletion &&
			                  inst.GetOpcode() != IR::ValueOpcode::DataCacheInvalidate &&
			                  inst.GetOpcode() != IR::ValueOpcode::MeshAllocate)) {
				if (half != 0 && ctx.state.lane_share.active && LaneShareReuse(ctx, lane, inst)) {
					continue;
				}
				emit_instruction(lane, inst);
			}
		}
		ctx.state.lane_half = 0;
		if (lds_write_sync) {
			// A write is followed by a barrier, so the other half's later reads see it; a read
			// leaves the halves unsynchronized for the next write.
			if (lds_write) {
				EmitWaveBarrier(ctx.state);
			}
			ctx.state.lds_split_synced = lds_write;
		}
	}
}

// NVIDIA's SPIR-V compiler (driver 580.178) crashes on a bool OpPhi whose incoming values are all
// defined before it when one of its predecessors comes later in the function, as a loop header's
// back edge does: it retypes such a phi and converts each value at the end of its predecessor,
// whose terminator it has not translated yet (a null pointer read at
// libnvidia-glvkspirv.so+0x157c88). Astro's Playroom's ray-traced lighting kernels have such phis;
// the emulator closed once the game left its menu. On those edges the phi takes a copy of the value
// made at the end of the later predecessor: not yet defined where the phi is, it keeps the phi a
// plain bool.
void CollectLateBoolPhiEdges(const IR::Program& program, StructuredFunctionState& structured) {
	std::unordered_map<const IR::Block*, size_t> order;
	for (size_t index = 0; index < program.blocks.size(); index++) {
		order.emplace(program.blocks[index], index);
	}
	for (size_t index = 0; index < program.blocks.size(); index++) {
		for (const auto& inst: *program.blocks[index]) {
			if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
				break;
			}
			if (inst.GetType() != IR::Type::U1) {
				continue;
			}
			for (size_t arg = 0; arg < inst.NumArgs(); arg++) {
				const auto found = order.find(inst.PhiBlock(arg));
				if (found != order.end() && found->second >= index) {
					structured.late_bool_phi_edges[found->first].push_back({&inst, arg});
				}
			}
		}
	}
}

// At the end of `block`, before its terminator: the copies its late bool phi edges carry.
void EmitLateBoolPhiCopies(ValueEmitContext& ctx, StructuredFunctionState& structured,
                           const IR::Block* block) {
	const auto edges = structured.late_bool_phi_edges.find(block);
	if (edges == structured.late_bool_phi_edges.end()) {
		return;
	}
	for (const auto& edge: edges->second) {
		auto& copies = structured.late_bool_phi_copies[edge.phi];
		copies.resize(edge.phi->NumArgs());
		for (uint32_t half = 0; half < ctx.state.lane_count; half++) {
			auto&      lane = half == 0 ? ctx : *ctx.other_half;
			const auto copy = ctx.state.builder.AllocateId();
			ctx.state.builder.AddFunction(spv::OpCopyObject, TypeBool(ctx.state), copy,
			                              lane.Def(edge.phi->Arg(edge.index)));
			copies[edge.index][half] = copy;
		}
	}
}

void PatchStructuredPhis(ValueEmitContext& ctx, StructuredFunctionState& structured) {
	for (const auto& deferred: structured.deferred_phis) {
		auto&      lane   = deferred.half == 0 ? ctx : *ctx.other_half;
		const auto copies = structured.late_bool_phi_copies.find(deferred.instruction);
		for (size_t index = 0; index < deferred.instruction->NumArgs(); index++) {
			const auto* predecessor = deferred.instruction->PhiBlock(index);
			const auto  found       = structured.block_exit_labels.find(predecessor);
			if (found == structured.block_exit_labels.end()) {
				ctx.Fail(*deferred.instruction, "has a predecessor that was not emitted");
			}
			auto value = copies != structured.late_bool_phi_copies.end()
			                 ? copies->second[index][deferred.half]
			                 : 0u;
			if (value == 0) {
				value = lane.Def(deferred.instruction->Arg(index));
			}
			ctx.state.builder.PatchDeferredPhi(deferred.phi, index, value, found->second);
		}
	}
}

void EmitStructuredFunction(ValueEmitContext& ctx) {
	const auto& program = ctx.state.program;
	StructuredFunctionState structured;
	CollectLateBoolPhiEdges(program, structured);
	ctx.state.builder.AddFunction(spv::OpBranch, ctx.Label(program.blocks.front()));
	for (size_t index = 0; index < program.blocks.size(); index++) {
		const auto* block = program.blocks[index];
		EmitBlock(ctx, block, [&](ValueEmitContext& lane, const IR::Inst& inst) {
			EmitStructuredInstruction(lane, structured, inst);
		});
		structured.block_exit_labels.emplace(block, ctx.state.current_label);
		EmitLateBoolPhiCopies(ctx, structured, block);
		EmitStructuredTerminator(ctx, block, program.block_info[index]);
	}
	PatchStructuredPhis(ctx, structured);
}

// A guest barrier where a mesh subgroup that runs in passes is cut (see EmitMeshEntryPoint).
struct MeshSplit {
	size_t block    = 0; // index in program.blocks
	size_t position = 0; // the barrier's position in that block
};

// The program's barriers as split points, or why the program cannot run in passes. A barrier
// qualifies at the top level only: no edge bypasses or re-enters its block and no loop or
// selection spans it, so the code before it is a whole region. The guest must also not end
// before its last barrier, which would leave a later segment running for a finished wave.
const char* FindMeshSplits(const IR::Program& program, std::vector<MeshSplit>& splits) {
	splits.clear();
	if (program.dispatcher_fallback) {
		return "the control flow needs the dispatcher";
	}
	const auto count = program.blocks.size();
	std::unordered_map<uint32_t, size_t> index_of;
	for (size_t i = 0; i < count; i++) {
		index_of.emplace(program.block_info[i].id, i);
	}
	const auto index = [&](uint32_t id) {
		const auto found = index_of.find(id);
		return found == index_of.end() ? SIZE_MAX : found->second;
	};
	// Successors in program order; a block whose terminator returns (as the structured emitter
	// treats it) is an exit.
	std::vector<std::vector<size_t>> successors(count);
	std::vector<bool>                exits(count, false);
	for (size_t i = 0; i < count; i++) {
		const auto& info = program.block_info[i];
		const auto& term = info.terminator;
		if (term.kind == CFG::TerminatorKind::Branch && index(term.true_block) != SIZE_MAX) {
			successors[i] = {index(term.true_block)};
		} else if (term.kind == CFG::TerminatorKind::ConditionalBranch &&
		           index(term.true_block) != SIZE_MAX && index(term.false_block) != SIZE_MAX &&
		           !info.condition.IsEmpty()) {
			successors[i] = {index(term.true_block), index(term.false_block)};
		} else if (term.kind == CFG::TerminatorKind::IndirectBranch ||
		           term.kind == CFG::TerminatorKind::Unsupported) {
			return "an indirect branch";
		} else {
			exits[i] = true;
		}
	}
	const auto top_level = [&](size_t k) {
		for (size_t i = 0; i < count; i++) {
			for (const auto j: successors[i]) {
				if ((i < k && j > k) || (i >= k && j <= k)) {
					return false;
				}
			}
			const auto& term = program.block_info[i].terminator;
			if (i < k && ((term.merge_block != UINT32_MAX && index(term.merge_block) > k) ||
			              (term.loop_header && index(term.continue_block) >= k))) {
				return false;
			}
		}
		return true;
	};
	for (size_t k = 0; k < count; k++) {
		size_t position = 0;
		for (const auto& inst: *program.blocks[k]) {
			if (inst.GetOpcode() == IR::ValueOpcode::Barrier) {
				if (!top_level(k)) {
					return "a barrier inside a loop or branch";
				}
				splits.push_back({k, position});
			}
			position++;
		}
	}
	for (size_t i = 0; !splits.empty() && i < splits.back().block; i++) {
		if (exits[i]) {
			return "the program can end before a barrier";
		}
	}
	for (const auto* block: program.blocks) {
		for (const auto& inst: *block) {
			// Scratch lives in Function variables, which do not outlive a segment.
			if (IR::AddressOpcodeInfoOf(inst.GetOpcode()).access != IR::AddressAccess::None &&
			    program.memory_info.at(inst.Flags<IR::MemoryFlags>().index).kind ==
			        IR::ResourceKind::Scratch) {
				return "scratch memory";
			}
		}
	}
	return nullptr;
}

// Which values cross a split: each is kept in a Private array with one element per lane of each
// pass, stored at the end of the segment that defines it and loaded by every later segment that
// uses it.
struct MeshSegmentPlan {
	std::vector<MeshSplit>                              splits;
	std::unordered_map<const IR::Inst*, uint32_t>       spills;
	std::vector<std::vector<const IR::Inst*>>           stores;
	std::vector<std::vector<const IR::Inst*>>           loads;
};

MeshSegmentPlan PlanMeshSegments(EmitterState& state, std::vector<MeshSplit> splits) {
	const auto&     program = state.program;
	MeshSegmentPlan plan;
	plan.splits = std::move(splits);
	plan.stores.resize(plan.splits.size() + 1);
	plan.loads.resize(plan.splits.size() + 1);
	std::unordered_map<const IR::Inst*, uint32_t> segment_of;
	std::vector<uint32_t>                         terminator_segment(program.blocks.size());
	uint32_t                                      segment = 0;
	size_t                                        next    = 0;
	for (size_t block = 0; block < program.blocks.size(); block++) {
		size_t position = 0;
		for (const auto& inst: *program.blocks[block]) {
			if (next < plan.splits.size() && plan.splits[next].block == block &&
			    plan.splits[next].position == position) {
				segment++;
				next++;
			}
			segment_of.emplace(&inst, segment);
			position++;
		}
		terminator_segment[block] = segment;
	}
	std::set<std::pair<const IR::Inst*, uint32_t>> visited;
	std::function<void(IR::Value, uint32_t)>       use = [&](IR::Value value, uint32_t user) {
        value            = value.Resolve();
        const auto* inst = value.TryInstruction();
        if (inst == nullptr || !visited.emplace(inst, user).second) {
            return;
        }
        const auto found = segment_of.find(inst);
        if (found == segment_of.end() || found->second >= user) {
            return;
        }
        if (TypeId(state, inst->GetType()) == 0) {
            // A handle the emitter reads through its producer: the producer's operands are
            // what the user's segment loads.
            for (size_t index = 0; index < inst->NumArgs(); index++) {
                use(inst->Arg(index), user);
            }
            return;
        }
        if (!plan.spills.contains(inst)) {
            const auto slots  = state.lane_count * state.mesh_passes;
            const auto array  = state.builder.Type(spv::OpTypeArray, TypeId(state, inst->GetType()),
                                                   ConstantU32(state, slots));
            plan.spills.emplace(inst, state.builder.DefineGlobalVariable(
                                          TypePointer(state, spv::StorageClassPrivate, array),
                                          spv::StorageClassPrivate));
            plan.stores[found->second].push_back(inst);
        }
        plan.loads[user].push_back(inst);
	};
	for (size_t block = 0; block < program.blocks.size(); block++) {
		for (const auto& inst: *program.blocks[block]) {
			for (size_t index = 0; index < inst.NumArgs(); index++) {
				use(inst.Arg(index), segment_of.at(&inst));
			}
		}
		const auto& info = program.block_info[block];
		if (!info.condition.IsEmpty()) {
			use(info.condition, terminator_segment[block]);
		}
		if (!info.indirect_target.IsEmpty()) {
			use(info.indirect_target, terminator_segment[block]);
		}
	}
	return plan;
}

// The element of a crossing value's array for this lane in the current pass.
uint32_t MeshSpillPointer(ValueEmitContext& lane, uint32_t variable, const IR::Inst& inst) {
	auto&      state = lane.state;
	const auto pass  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), pass, state.mesh_pass_variable);
	const auto slot = Binary(state, spv::OpIAdd, TypeU32(state),
	                         Binary(state, spv::OpIMul, TypeU32(state), pass,
	                                ConstantU32(state, state.lane_count)),
	                         ConstantU32(state, lane.half));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassPrivate,
	                                      TypeId(state, inst.GetType())),
	                          pointer, variable, slot);
	return pointer;
}

// The guest program as one function per segment between its barriers. Each segment emits its
// blocks once; the block a barrier cuts continues under a fresh label in the next segment.
void EmitMeshSegmentFunctions(ValueEmitContext& ctx, const MeshSegmentPlan& plan) {
	auto&       state   = ctx.state;
	const auto& program = state.program;
	size_t      block   = 0;
	size_t      start   = 0;
	for (size_t segment = 0; segment <= plan.splits.size(); segment++) {
		const auto function = state.builder.AllocateId();
		state.mesh_segment_funcs.push_back(function);
		state.builder.AddFunction(spv::OpFunction, TypeVoid(state), function,
		                          spv::FunctionControlMaskNone, TypeFunction(state));
		EmitLabel(state, state.builder.AllocateId());
		for (uint32_t half = 0; half < state.lane_count; half++) {
			auto& lane = half == 0 ? ctx : *ctx.other_half;
			if (lane.scratch_u32_variable != 0) {
				lane.scratch_u32_variable = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpVariable, TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
				    lane.scratch_u32_variable, spv::StorageClassFunction);
			}
		}
		if (state.gds_variable != 0) {
			state.gds_length = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
			                          state.gds_variable, 0);
		}
		EmitMemoryOffsets(state);
		for (const auto* inst: plan.loads[segment]) {
			for (uint32_t half = 0; half < state.lane_count; half++) {
				auto&      lane  = half == 0 ? ctx : *ctx.other_half;
				const auto value = state.builder.AllocateId();
				state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), value,
				                          MeshSpillPointer(lane, plan.spills.at(inst), *inst));
				lane.definitions.insert_or_assign(inst, value);
			}
		}
		StructuredFunctionState structured;
		if (start == 0) {
			state.builder.AddFunction(spv::OpBranch, ctx.Label(program.blocks[block]));
		} else {
			const auto resume = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBranch, resume);
			EmitLabel(state, resume);
		}
		const auto* end = segment < plan.splits.size() ? &plan.splits[segment] : nullptr;
		while (block < program.blocks.size()) {
			const auto* ir_block = program.blocks[block];
			state.current_block  = ir_block;
			if (start == 0) {
				EmitLabel(state, ctx.Label(ir_block));
			}
			const auto stop     = end != nullptr && end->block == block ? end->position : SIZE_MAX;
			size_t     position = 0;
			for (const auto& inst: *ir_block) {
				if (position >= start && position < stop &&
				    inst.GetOpcode() != IR::ValueOpcode::Barrier) {
					for (uint32_t half = 0; half < state.lane_count; half++) {
						state.lane_half = half;
						if (half == 0 || inst.GetOpcode() != IR::ValueOpcode::MeshAllocate) {
							EmitStructuredInstruction(half == 0 ? ctx : *ctx.other_half, structured,
							                          inst);
						}
					}
					state.lane_half = 0;
				}
				position++;
			}
			if (stop != SIZE_MAX) {
				start = stop + 1;
				break;
			}
			structured.block_exit_labels.emplace(ir_block, state.current_label);
			EmitStructuredTerminator(ctx, ir_block, program.block_info[block]);
			block++;
			start = 0;
		}
		if (end != nullptr) {
			for (const auto* inst: plan.stores[segment]) {
				for (uint32_t half = 0; half < state.lane_count; half++) {
					auto& lane = half == 0 ? ctx : *ctx.other_half;
					state.builder.AddFunction(spv::OpStore,
					                          MeshSpillPointer(lane, plan.spills.at(inst), *inst),
					                          lane.Def(IR::Value(const_cast<IR::Inst*>(inst))));
				}
			}
			state.builder.AddFunction(spv::OpReturn);
		}
		PatchStructuredPhis(ctx, structured);
		state.builder.AddFunction(spv::OpFunctionEnd);
	}
}

void EmitDispatcherFunction(ValueEmitContext& ctx, const DispatcherFunctionState& dispatcher) {
	auto&       state = ctx.state;
	const auto* entry = state.program.blocks.front();
	state.builder.AddFunction(spv::OpBranch, ctx.Label(entry));
	EmitBlock(ctx, entry, [&](ValueEmitContext& lane, const IR::Inst& inst) {
		EmitDispatcherInstruction(lane, dispatcher, inst);
	});
	const auto initial_pc =
	    EmitDispatcherNextPc(ctx, dispatcher, entry, state.program.block_info.front());
	const auto initial_parent = state.current_label;
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);

	EmitLabel(state, dispatcher.header_label);
	const auto pc      = state.builder.AllocateId();
	const auto next_pc = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpPhi, TypeU32(state), pc, initial_pc, initial_parent, next_pc,
	                          dispatcher.continue_label);
	// KYTY_DISPATCHER_CAP (from Senaxx's wolverine branch): a guest loop whose trip count comes from
	// bad data (memory an unemulated pass left unwritten) hung the GPU until the driver reset it.
	// The invocation leaves the dispatcher after a bounded number of block transitions instead:
	// its results are wrong, the device survives. 0 emits the loop without the bound.
	const auto cap            = GetCodegenOptions().dispatcher_cap;
	uint32_t   iteration      = 0;
	uint32_t   next_iteration = 0;
	if (cap != 0) {
		iteration      = state.builder.AllocateId();
		next_iteration = state.builder.AllocateId();
	}
	const auto finished = state.builder.AllocateId();
	if (cap != 0) {
		state.builder.AddFunction(spv::OpPhi, TypeU32(state), iteration, ConstantU32(state, 0u),
		                          initial_parent, next_iteration, dispatcher.continue_label);
	}
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), finished, pc,
	                          ConstantU32(ctx.state, UINT32_MAX));
	auto done = finished;
	if (cap != 0) {
		const auto exhausted = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), exhausted, iteration,
		                          ConstantU32(state, cap));
		done = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLogicalOr, TypeBool(state), done, finished, exhausted);
	}
	state.builder.AddFunction(spv::OpLoopMerge, dispatcher.merge_label, dispatcher.continue_label,
	                          spv::LoopControlMaskNone);
	state.builder.AddFunction(spv::OpBranchConditional, done, dispatcher.merge_label,
	                          dispatcher.select_label);

	EmitLabel(state, dispatcher.select_label);
	state.builder.AddFunction(spv::OpSelectionMerge, dispatcher.after_switch_label,
	                          spv::SelectionControlMaskNone);
	std::vector<uint32_t> words {spv::OpSwitch, pc, dispatcher.after_switch_label};
	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		words.push_back(state.program.block_info[index].id);
		words.push_back(ctx.Label(state.program.blocks[index]));
	}
	state.builder.AddFunction(words);
	std::vector<uint32_t> next_pc_words {spv::OpPhi, TypeU32(state), next_pc,
	                                     ConstantU32(state, UINT32_MAX), dispatcher.select_label};

	for (size_t index = 1; index < state.program.blocks.size(); index++) {
		EmitBlock(ctx, state.program.blocks[index],
		          [&](ValueEmitContext& lane, const IR::Inst& inst) {
			          EmitDispatcherInstruction(lane, dispatcher, inst);
		          });
		const auto selected = EmitDispatcherNextPc(ctx, dispatcher, state.program.blocks[index],
		                                           state.program.block_info[index]);
		next_pc_words.push_back(selected);
		next_pc_words.push_back(state.current_label);
		state.builder.AddFunction(spv::OpBranch, dispatcher.after_switch_label);
	}
	EmitLabel(state, dispatcher.after_switch_label);
	state.builder.AddFunction(next_pc_words);
	state.builder.AddFunction(spv::OpBranch, dispatcher.continue_label);
	EmitLabel(state, dispatcher.continue_label);
	if (cap != 0) {
		state.builder.AddFunction(spv::OpIAdd, TypeU32(state), next_iteration, iteration,
		                          ConstantU32(state, 1u));
	}
	state.builder.AddFunction(spv::OpBranch, dispatcher.header_label);
	EmitLabel(state, dispatcher.merge_label);
	EmitReturn(ctx);
}

} // namespace

uint32_t TypeId(EmitterState& state, IR::Type type) {
	switch (type) {
		case IR::Type::U1: return TypeBool(state);
		case IR::Type::U8:
		case IR::Type::U16:
		case IR::Type::U32:
		case IR::Type::F16: return TypeU32(state);
		case IR::Type::U64: return TypeU64(state);
		case IR::Type::U32x2: return TypeU32Pair(state);
		case IR::Type::F32: return TypeF32(state);
		case IR::Type::F64: return TypeF64(state);
		case IR::Type::Bvh8Result: return TypeBvh8Result(state);
		case IR::Type::U32x3: return TypeU32Vector(state, 3);
		case IR::Type::U32x4: return TypeU32Vector(state, 4);
		case IR::Type::F32x2: return TypeF32Vector(state, 2);
		default: return 0;
	}
}

uint32_t ValueEmitContext::Def(IR::Value value) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case IR::Type::U1: return ConstantBool(state, value.U1());
			case IR::Type::U8: return ConstantU32(state, value.U8());
			case IR::Type::U16: return ConstantU32(state, value.U16());
			case IR::Type::U32: return ConstantU32(state, value.U32());
			case IR::Type::U64: return ConstantU64(state, value.U64());
			case IR::Type::F16: return ConstantU32(state, value.F16Bits());
			case IR::Type::F32:
				return ConstantF32(state, std::bit_cast<uint32_t>(value.F32Value()));
			default: break;
		}
	}
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr) {
		Fail("direct SPIR-V emitter received a non-value argument");
	}
	if (dispatcher_spills != nullptr && state.current_block != nullptr &&
	    inst->Parent() != state.current_block) {
		if (const auto found = dispatcher_spills->find(inst); found != dispatcher_spills->end()) {
			if (const auto loaded = dispatcher_block_loads.find(inst);
			    loaded != dispatcher_block_loads.end() &&
			    loaded->second.first == state.current_label) {
				return loaded->second.second;
			}
			const auto id = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpLoad, TypeId(state, inst->GetType()), id,
			                          found->second);
			dispatcher_block_loads.insert_or_assign(inst, std::pair {state.current_label, id});
			return id;
		}
	}
	return Result(*inst);
}

uint32_t ValueEmitContext::Arg(const IR::Inst& inst, size_t index) {
	return Def(inst.Arg(index));
}

uint32_t ValueEmitContext::HalfArg(const IR::Inst& inst, size_t index, uint32_t lane_half) {
	return lane_half == half ? Arg(inst, index) : other_half->Arg(inst, index);
}

uint32_t ValueEmitContext::Ballot(IR::Value predicate) {
	if (state.split_wave) {
		// Every lane of a split wave exists, so a constant-true ballot needs no exchange.
		if (predicate.IsImmediate() && predicate.GetType() == IR::Type::U1 && predicate.U1()) {
			return state.builder.Constant(spv::OpConstantComposite, TypeU32Vector(state, 4),
			                              ConstantU32(state, ~0u), ConstantU32(state, ~0u),
			                              ConstantU32(state, 0), ConstantU32(state, 0));
		}
		return ExchangeWaveBallot(state, Def(predicate));
	}
	const auto ballot_type = TypeU32Vector(state, 4);
	const auto scope       = ConstantU32(state, spv::ScopeSubgroup);
	const auto low_value   = other_half == nullptr || half == 0 ? Def(predicate) : other_half->Def(predicate);
	const auto low         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, low, scope, low_value);
	if (other_half == nullptr) {
		return EmitWaveBallot(state, low);
	}
	const auto high_value = half == 1 ? Def(predicate) : other_half->Def(predicate);
	const auto low_word   = state.builder.AllocateId();
	const auto ballot     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low_word, low, 0);
	// KYTY_LANE2_SHARE: both halves hold the same id (a constant, or a value proven equal in both
	// halves), so the high half's ballot is the low half's.
	uint32_t high_word = low_word;
	if (high_value != low_value || !GetCodegenOptions().lane2_share) {
		const auto high = state.builder.AllocateId();
		high_word       = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallot, ballot_type, high, scope, high_value);
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high_word, high, 0);
	}
	state.builder.AddFunction(spv::OpCompositeConstruct, ballot_type, ballot, low_word, high_word,
	                          ConstantU32(state, 0), ConstantU32(state, 0));
	return ballot;
}

uint32_t ValueEmitContext::FirstLane(uint32_t ballot) {
	if (state.wave_clusters) {
		// The wave's own word (EmitWaveBallot); FindILsb of 0 is -1, as FindLSB of an empty ballot.
		const auto bits   = state.builder.AllocateId();
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, ballot, 0);
		state.builder.AddFunction(spv::OpExtInst, TypeU32(state), result, GlslStd450(state),
		                          GLSLstd450FindILsb, bits);
		return result;
	}
	if (other_half == nullptr && !state.split_wave && GetCodegenOptions().readfirstlane_exec_zero) {
		// KYTY_READFIRSTLANE_EXEC0 (upstream 919fac13d): an empty ballot (EXEC zero) gives lane 0,
		// which V_READFIRSTLANE reads on hardware; BallotFindLSB is undefined for it.
		const auto first = [&](uint32_t word) {
			const auto bits = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), bits, ballot, word);
			const auto lsb = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpExtInst, TypeU32(state), lsb, GlslStd450(state),
			                          GLSLstd450FindILsb, bits);
			return std::pair {bits, lsb};
		};
		const auto [low, low_first] = first(0);
		const auto clamp_zero = [&](uint32_t value) {
			const auto result = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpExtInst, TypeU32(state), result, GlslStd450(state),
			                          GLSLstd450SMax, value, ConstantU32(state, 0));
			return result;
		};
		if (state.program.wave_size == 32u) {
			return clamp_zero(low_first);
		}
		const auto high_first = first(1).second;
		const auto high_lane  = clamp_zero(EmitBinaryU32(state, spv::OpBitwiseOr, high_first,
		                                                 ConstantU32(state, 32)));
		return Select(state, TypeU32(state),
		              Binary(state, spv::OpINotEqual, TypeBool(state), low, ConstantU32(state, 0)),
		              low_first, high_lane);
	}
	if (other_half == nullptr && !state.split_wave) {
		const auto result = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpGroupNonUniformBallotFindLSB, TypeU32(state), result,
		                          ConstantU32(state, spv::ScopeSubgroup), ballot);
		return result;
	}
	const auto low        = state.builder.AllocateId();
	const auto high       = state.builder.AllocateId();
	const auto low_first  = state.builder.AllocateId();
	const auto high_first = state.builder.AllocateId();
	const auto low_active = state.builder.AllocateId();
	const auto result     = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, ballot, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, ballot, 1);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), low_first, GlslStd450(state),
	                          GLSLstd450FindILsb, low);
	state.builder.AddFunction(spv::OpExtInst, TypeU32(state), high_first, GlslStd450(state),
	                          GLSLstd450FindILsb, high);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), low_active, low,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, TypeU32(state), result, low_active, low_first,
	                          EmitAddU32(state, high_first, ConstantU32(state, 32)));
	return result;
}

uint32_t ValueEmitContext::Shuffle(const IR::Inst& inst, size_t index, uint32_t lane) {
	const auto type  = TypeId(state, inst.Arg(index).GetType());
	if (state.split_wave) {
		// The lane may be in the other host subgroup: exchange through workgroup memory.
		const auto u32   = TypeU32(state);
		const auto value = Arg(inst, index);
		if (type == TypeBool(state)) {
			const auto bits = ExchangeWaveValue(
			    state, Select(state, u32, value, ConstantU32(state, 1), ConstantU32(state, 0)), lane);
			return Binary(state, spv::OpINotEqual, type, bits, ConstantU32(state, 0));
		}
		if (type == u32) {
			return ExchangeWaveValue(state, value, lane);
		}
		if (type != TypeF32(state) && type != TypeI32(state)) {
			Fail(inst, "split-wave lane read of a value wider than a dword");
		}
		return Unary(state, spv::OpBitcast, type,
		             ExchangeWaveValue(state, Unary(state, spv::OpBitcast, u32, value), lane));
	}
	const auto scope = ConstantU32(state, spv::ScopeSubgroup);
	const auto low   = state.builder.AllocateId();
	if (other_half == nullptr) {
		state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope, Arg(inst, index),
		                          EmitHostLane(state, lane));
		return low;
	}
	const auto physical_lane =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31));
	const auto high          = state.builder.AllocateId();
	const auto in_high       = state.builder.AllocateId();
	const auto value         = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, low, scope,
	                          HalfArg(inst, index, 0), physical_lane);
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, high, scope,
	                          HalfArg(inst, index, 1), physical_lane);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), in_high,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 32)),
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpSelect, type, value, in_high, high, low);
	return value;
}

uint32_t ValueEmitContext::Result(const IR::Inst& inst) {
	if (const auto found = definitions.find(&inst); found != definitions.end()) {
		return found->second;
	}
	const auto id = state.builder.AllocateId();
	definitions.emplace(&inst, id);
	return id;
}

uint32_t ValueEmitContext::Define(const IR::Inst& inst, uint32_t value) {
	if (const auto found = definitions.find(&inst); found != definitions.end()) {
		if (found->second != value) {
			state.builder.AddFunction(spv::OpCopyObject, TypeId(state, inst.GetType()),
			                          found->second, value);
		}
		return found->second;
	}
	definitions.emplace(&inst, value);
	return value;
}

uint32_t ValueEmitContext::ResourceIndex(IR::Value value, IR::ValueOpcode opcode) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != opcode) {
		Fail("typed resource handle has the wrong producer");
	}
	return inst->Flags<uint32_t>();
}

const IR::Inst* ValueEmitContext::ImageAddress(IR::Value value) {
	const auto* inst = value.ResolveInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::MakeImageAddress) {
		Fail("typed image address was not constructed by MakeImageAddress");
	}
	return inst;
}

const IR::MemoryInfo& ValueEmitContext::Memory(const IR::Inst& inst) const {
	return state.program.memory_info.at(inst.Flags<IR::MemoryFlags>().index);
}

const IR::ExportInfo& ValueEmitContext::Export(const IR::Inst& inst) const {
	return state.program.export_info.at(inst.Flags<IR::ExportFlags>().index);
}

uint32_t ValueEmitContext::Label(const IR::Block* block) const {
	return state.labels.at(block);
}

[[noreturn]] void ValueEmitContext::Fail(const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage), reason);
	std::abort();
}

[[noreturn]] void ValueEmitContext::Fail(const IR::Inst& inst, const char* reason) const {
	EXIT("SPIR-V emission failed: hash=0x%016" PRIx64 " stage=%u opcode=%s reason=%s\n",
	     state.program.shader_hash, static_cast<unsigned>(state.program.stage),
	     IR::ValueOpcodeName(inst.GetOpcode()), reason);
	std::abort();
}

void EmitProgram(EmitterState& state) {
	const auto&      program = state.program;
	ValueEmitContext ctx(state);
	ValueEmitContext high(state);
	if (state.lane_count == 2) {
		ctx.other_half  = &high;
		high.other_half = &ctx;
		high.half       = 1;
	}
	LaneShareClassify(state);
	std::optional<DispatcherFunctionState> dispatcher;
	if (state.program.stage == ShaderType::Pixel && state.requirements.pixel_valid_mask) {
		state.pixel_valid_mask_variable = state.builder.AllocateId();
		state.builder.AddName(state.pixel_valid_mask_variable, "pixel_valid_mask_active");
	}
	if (GetCodegenOptions().call_loop_cap != 0u && !program.dispatcher_fallback &&
	    std::ranges::any_of(program.block_info,
	                        [](const IR::BlockInfo& info) { return info.terminator.call_loop; })) {
		state.call_loop_variable = state.builder.AllocateId();
		state.builder.AddName(state.call_loop_variable, "call_loop_rounds");
	}
	if (LoopGuardApplies(program.shader_hash) && !program.dispatcher_fallback) {
		state.loop_guard_variable = state.builder.AllocateId();
		state.builder.AddName(state.loop_guard_variable, "loop_guard");
	}
	if (IR::UsesBvhNodeCount(program)) {
		state.bvh_node_count_variable = state.builder.AllocateId();
		state.builder.AddName(state.bvh_node_count_variable, "bvh_node_count");
	}
	for (const auto* block: program.blocks) {
		const auto label = state.builder.AllocateId();
		state.labels.emplace(block, label);
	}
	if (state.program.dispatcher_fallback) {
		auto& dispatch = dispatcher.emplace();
		// Insertion order of spills[0]. The high half's ids are allocated in this program order:
		// iterating the pointer-keyed map made them depend on instruction addresses, so the same
		// shader produced different SPIR-V on every run and missed the pipeline caches.
		std::vector<const IR::Inst*> spill_order;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				if (inst.GetOpcode() != IR::ValueOpcode::Phi) {
					continue;
				}
				if (SpillPointerType(ctx, inst.GetType()) == 0) {
					ctx.Fail(inst, "cannot be stored by the dispatcher");
					break;
				}
				dispatch.spills[0].emplace(&inst, state.builder.AllocateId());
				spill_order.push_back(&inst);
			}
		}
		const auto mark_cross_block = [&](IR::Value value, const IR::Block* consumer) {
			value                  = value.Resolve();
			const auto* definition = value.TryInstruction();
			if (definition == nullptr || definition->Parent() == consumer ||
			    definition->Parent() == program.blocks.front()) {
				return;
			}
			if (SpillPointerType(ctx, definition->GetType()) == 0) {
				ctx.Fail(*definition, "cannot be stored by the dispatcher");
				return;
			}
			if (!dispatch.spills[0].contains(definition)) {
				dispatch.spills[0].emplace(definition, state.builder.AllocateId());
				spill_order.push_back(definition);
			}
		};
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				for (size_t index = 0; index < inst.NumArgs(); index++) {
					const auto* consumer =
					    inst.GetOpcode() == IR::ValueOpcode::Phi ? inst.PhiBlock(index) : block;
					mark_cross_block(inst.Arg(index), consumer);
				}
			}
		}
		for (size_t index = 0; index < program.blocks.size(); index++) {
			mark_cross_block(program.block_info[index].condition, program.blocks[index]);
			mark_cross_block(program.block_info[index].indirect_target, program.blocks[index]);
		}
		dispatch.header_label       = state.builder.AllocateId();
		dispatch.select_label       = state.builder.AllocateId();
		dispatch.after_switch_label = state.builder.AllocateId();
		dispatch.continue_label     = state.builder.AllocateId();
		dispatch.merge_label        = state.builder.AllocateId();
		ctx.dispatcher_spills       = &dispatch.spills[0];
		if (state.lane_count == 2) {
			for (const auto* inst: spill_order) {
				dispatch.spills[1].emplace(inst, state.builder.AllocateId());
			}
			high.dispatcher_spills = &dispatch.spills[1];
		}
	}
	DefineGetBdaPointer(state);
	DefineBvhNodeTestFunction(state);
	DefineBvh8IntersectFunction(state);
	DefineIndirectFormatFunction(state);
	for (const auto* block: program.blocks) {
		if (std::ranges::any_of(*block, [](const IR::Inst& inst) {
			    return inst.GetOpcode() == IR::ValueOpcode::SwizzleU32;
		    })) {
			ctx.scratch_u32_variable = state.builder.AllocateId();
			if (state.lane_count == 2) {
				high.scratch_u32_variable = state.builder.AllocateId();
			}
			break;
		}
	}
	if (state.mesh_passes > 1u) {
		// TranslateProgram gave up on programs that cannot be cut.
		std::vector<MeshSplit> splits;
		if (const auto* reason = FindMeshSplits(program, splits); reason != nullptr) {
			ctx.Fail(reason);
		}
		EmitMeshSegmentFunctions(ctx, PlanMeshSegments(state, std::move(splits)));
		EmitMeshEntryPoint(state);
		return;
	}
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state),
	                          state.mesh_guest_func != 0 ? state.mesh_guest_func : state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.entry_label);
	if (state.split_wave) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.wave_exchange_parity, spv::StorageClassFunction,
		                          ConstantU32(state, 0));
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				state.ds_writes_lds |= AccessesLds(program, inst) &&
				                       IR::SharedAccessOf(inst.GetOpcode()) != IR::SharedAccess::Read;
			}
		}
	}
	if (state.requirements.function_lds && state.lds_slot_plan) {
		for (const auto& [key, variable]: state.lds_slot_variables) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          variable, spv::StorageClassFunction);
		}
	} else if (state.requirements.function_lds) {
		state.builder.AddFunction(
		    spv::OpVariable,
		    TypeU32ArrayPointer(state, spv::StorageClassFunction, LdsDwordCount(state)),
		    state.lds_variable, spv::StorageClassFunction);
	}
	if (state.requirements.function_scratch) {
		for (uint32_t half = 0; half < state.lane_count; half++) {
			state.builder.AddFunction(
			    spv::OpVariable,
			    TypeU32ArrayPointer(state, spv::StorageClassFunction, state.program.scratch_dwords),
			    state.scratch_variable[half], spv::StorageClassFunction);
		}
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.pixel_valid_mask_variable, spv::StorageClassFunction);
	}
	if (state.loop_guard_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.loop_guard_variable, spv::StorageClassFunction);
	}
	if (state.call_loop_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.call_loop_variable, spv::StorageClassFunction);
	}
	if (state.bvh_node_count_variable != 0) {
		state.builder.AddFunction(spv::OpVariable,
		                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
		                          state.bvh_node_count_variable, spv::StorageClassFunction);
	}
	DeclareRayQueryVariables(state);
	for (uint32_t half = 0; half < state.lane_count; half++) {
		auto& lane = half == 0 ? ctx : high;
		if (state.program.dispatcher_fallback) {
			for (const auto* block: program.blocks) {
				for (const auto& inst: *block) {
					if (const auto found = dispatcher->spills[half].find(&inst);
					    found != dispatcher->spills[half].end()) {
						state.builder.AddFunction(spv::OpVariable,
						                          SpillPointerType(lane, inst.GetType()),
						                          found->second, spv::StorageClassFunction);
					}
				}
			}
		}
		if (lane.scratch_u32_variable != 0) {
			state.builder.AddFunction(spv::OpVariable,
			                          TypePointer(state, spv::StorageClassFunction, TypeU32(state)),
			                          lane.scratch_u32_variable, spv::StorageClassFunction);
		}
	}
	if (state.gds_variable != 0) {
		state.gds_length = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), state.gds_length,
		                          state.gds_variable, 0);
	}
	if (state.pixel_valid_mask_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.pixel_valid_mask_variable,
		                          ConstantU32(state, 1));
	}
	if (state.bvh_node_count_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.bvh_node_count_variable,
		                          ConstantU32(state, 0));
	}
	if (state.loop_guard_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.loop_guard_variable, ConstantU32(state, 0));
	}
	if (state.call_loop_variable != 0) {
		state.builder.AddFunction(spv::OpStore, state.call_loop_variable, ConstantU32(state, 0));
	}
	InitializeRayQueryVariables(state);
	EmitGeometryOutputDefaults(state);
	if (state.lds_storage_class == spv::StorageClassStorageBuffer && state.lds_variable != 0) {
		// The workgroup's LDS region is the one of its index in the dispatch, x + nx * (y + ny * z),
		// LdsStorageRegionDwords apart. A workgroup the bound range holds no region for (more groups
		// than the renderer could allocate for, RenderExecutor::BindComputeLds) gets length 0: its
		// LDS reads return 0 and its writes are dropped, as for any out-of-range LDS address,
		// instead of landing in another workgroup's region or past the buffer.
		const auto group_x = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
		const auto group_y = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1);
		const auto group_z = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 2);
		const auto count_x = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0);
		const auto count_y = EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 1);
		const auto row = EmitAddU32(state, group_y, EmitBinaryU32(state, spv::OpIMul, group_z, count_y));
		const auto index = EmitAddU32(state, group_x, EmitBinaryU32(state, spv::OpIMul, row, count_x));
		const auto region = ConstantU32(state, LdsStorageRegionDwords(LdsDwordCount(state)));
		const auto dwords = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), dwords, state.lds_variable, 0);
		const auto held    = EmitBinaryU32(state, spv::OpUDiv, dwords, region);
		const auto present = Binary(state, spv::OpULessThan, TypeBool(state), index, held);
		state.lds_base_dwords = EmitSelectValueU32(
		    state, present, EmitBinaryU32(state, spv::OpIMul, index, region), ConstantU32(state, 0));
		state.lds_length = EmitSelectValueU32(state, present, ConstantU32(state, LdsDwordCount(state)),
		                                      ConstantU32(state, 0));
	}
	EmitMemoryOffsets(state);
	if (PopsInterlockActive(state)) {
		// KYTY_POPS_INTERLOCK: the guest's POPS section (S_SETREG POPS_PACKER .. S_SENDMSG
		// ORDERED_PS_DONE) widened to the whole program, so Begin and End each run exactly once in
		// uniform control flow; EmitReturn ends it.
		state.builder.AddFunction(spv::OpBeginInvocationInterlockEXT);
	}
	if (state.objprim_output_variable != 0) {
		EmitObjprimIdOutput(state);
	}
	if (program.blocks.empty()) {
		EmitReturn(ctx);
	} else if (state.program.dispatcher_fallback) {
		EmitDispatcherFunction(ctx, *dispatcher);
	} else {
		EmitStructuredFunction(ctx);
	}
	state.builder.AddFunction(spv::OpFunctionEnd);
	if (state.program.stage == ShaderType::Mesh) {
		EmitMeshEntryPoint(state);
	}
	LaneShareReport(state);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter

namespace Libs::Graphics::ShaderRecompiler::Spirv {

const char* MeshPassesUnsupported(const IR::Program& program) {
	std::vector<Emitter::MeshSplit> splits;
	return Emitter::FindMeshSplits(program, splits);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
