#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

#include "graphics/shader/recompiler/CodegenOptions.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

DppTargetLane EmitDppPermTargetLane(EmitterState& state, uint32_t subid, uint32_t control,
                                    uint32_t lane_bits) {
	// Each lane of a group takes its lane_bits-wide source index from control.
	const auto lane = Binary(state, spv::OpBitwiseAnd, TypeU32(state), subid,
	                         ConstantU32(state, (1u << lane_bits) - 1u));
	const auto selected = EmitNative<spv::OpBitFieldUExtract, IR::Type::U32>(
	    state, ConstantU32(state, control),
	    Binary(state, spv::OpIMul, TypeU32(state), lane, ConstantU32(state, lane_bits)),
	    ConstantU32(state, lane_bits));
	return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
	            state, subid, selected, ConstantU32(state, 0), ConstantU32(state, lane_bits)),
	        ConstantBool(state, true)};
}

DppTargetLane EmitDppRowShiftTargetLane(EmitterState& state, uint32_t subid, uint32_t amount,
                                        bool left) {
	const auto row          = state.builder.AllocateId();
	const auto lane         = state.builder.AllocateId();
	const auto lane_shifted = state.builder.AllocateId();
	const auto target       = state.builder.AllocateId();
	const auto valid        = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), row, subid,
	                          ConstantU32(state, 0xfffffff0u));
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane, subid,
	                          ConstantU32(state, 15));
	if (left) {
		state.builder.AddFunction(spv::OpIAdd, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), valid, lane,
		                          ConstantU32(state, 16u - amount));
	} else {
		state.builder.AddFunction(spv::OpISub, TypeU32(state), lane_shifted, lane,
		                          ConstantU32(state, amount));
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), valid, lane,
		                          ConstantU32(state, amount));
	}
	state.builder.AddFunction(spv::OpBitwiseOr, TypeU32(state), target, row, lane_shifted);
	return {target, valid};
}

DppTargetLane EmitDppRowRotateRightTargetLane(EmitterState& state, uint32_t subid,
                                              uint32_t amount) {
	// Rotation wraps within the row, so only the low four lane bits change.
	const auto rotated = Binary(state, spv::OpISub, TypeU32(state), subid, ConstantU32(state, amount));
	return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(state, subid, rotated,
	                                                        ConstantU32(state, 0),
	                                                        ConstantU32(state, 4)),
	        ConstantBool(state, true)};
}

DppTargetLane EmitDppMirrorTargetLane(EmitterState& state, uint32_t subid, bool half_row) {
	// Mirroring lane i of a row or half row reads lane (size - 1 - i), which flips its low bits.
	return {Binary(state, spv::OpBitwiseXor, TypeU32(state), subid,
	               ConstantU32(state, half_row ? 7u : 15u)),
	        ConstantBool(state, true)};
}

} // namespace

DppTargetLane EmitDppTargetLane(EmitterState& state, const IR::DppMoveFlags& flags) {
	const auto subid = EmitSubgroupLocalInvocationId(state);
	const auto control = flags.control;
	if (flags.dpp8) {
		return EmitDppPermTargetLane(state, subid, control, 3u);
	}
	if (control <= 0xffu) {
		return EmitDppPermTargetLane(state, subid, control, 2u);
	}
	if (control >= 0x101u && control <= 0x10fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, true);
	}
	if (control >= 0x111u && control <= 0x11fu) {
		return EmitDppRowShiftTargetLane(state, subid, control & 0xfu, false);
	}
	if (control >= 0x121u && control <= 0x12fu) {
		return EmitDppRowRotateRightTargetLane(state, subid, control & 0xfu);
	}
	if (control == 0x140u) {
		return EmitDppMirrorTargetLane(state, subid, false);
	}
	if (control == 0x141u) {
		return EmitDppMirrorTargetLane(state, subid, true);
	}
	// DPP_ROW_SHARE (GFX10, 0x150-0x15F): every lane of a row reads lane `control & 15` of it.
	if (control >= 0x150u && control <= 0x15fu) {
		return {EmitNative<spv::OpBitFieldInsert, IR::Type::U32>(
		            state, subid, ConstantU32(state, control & 0xfu), ConstantU32(state, 0),
		            ConstantU32(state, 4)),
		        ConstantBool(state, true)};
	}
	if (control >= 0x160u && control <= 0x16fu) {
		const auto target = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitwiseXor, TypeU32(state), target, subid,
		                          ConstantU32(state, control & 0xfu));
		return {target, ConstantBool(state, true)};
	}
	return {subid, ConstantBool(state, true)};
}

uint32_t EmitSubgroupLocalInvocationId(EmitterState& state) {
	if (state.split_wave) {
		// The workgroup is the wave: its invocation index is the guest lane.
		return EmitLocalInvocationIndex(state);
	}
	if (state.subgroup_local_invocation_id_variable == 0) {
		EXIT("SubgroupLocalInvocationId was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value,
	                          state.subgroup_local_invocation_id_variable);
	if (state.wave_clusters) {
		return EmitBinaryU32(state, spv::OpBitwiseAnd, value, ConstantU32(state, 31));
	}
	return state.lane_half == 0 ? value : EmitAddU32(state, value, ConstantU32(state, 32));
}

namespace {

// The raw SubgroupLocalInvocationId (KYTY_WAVE32_CLUSTERS only).
uint32_t LoadHostSubgroupLane(EmitterState& state) {
	if (state.subgroup_local_invocation_id_variable == 0) {
		EXIT("SubgroupLocalInvocationId was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value,
	                          state.subgroup_local_invocation_id_variable);
	return value;
}

} // namespace

uint32_t EmitHostSubgroupLane(EmitterState& state) {
	return LoadHostSubgroupLane(state);
}

void NoteWaveSync(EmitterState& state) {
	if (state.wave_sync_blocks != nullptr && state.current_block != nullptr) {
		state.wave_sync_blocks->insert(state.current_block);
	}
}

void EmitWaveBarrier(EmitterState& state) {
	NoteWaveSync(state);
	state.builder.AddFunction(spv::OpControlBarrier, ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::ScopeWorkgroup),
	                          ConstantU32(state, spv::MemorySemanticsAcquireReleaseMask |
	                                                 spv::MemorySemanticsWorkgroupMemoryMask));
}

namespace {

// The exchange slot of `lane` in the half of the exchange buffer this exchange uses. Exchanges
// alternate halves, so a host subgroup that runs ahead into the next exchange does not overwrite
// slots the other one still reads: it waits at that exchange's barrier first.
uint32_t WaveExchangeSlot(EmitterState& state, uint32_t parity, uint32_t lane) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(
	    spv::OpAccessChain, TypePointer(state, spv::StorageClassWorkgroup, TypeU32(state)), pointer,
	    state.wave_exchange_variable,
	    EmitAddU32(state, EmitBinaryU32(state, spv::OpShiftLeftLogical, parity, ConstantU32(state, 6)),
	               EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 63))));
	return pointer;
}

uint32_t NextWaveExchangeParity(EmitterState& state) {
	const auto parity = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), parity, state.wave_exchange_parity);
	state.builder.AddFunction(spv::OpStore, state.wave_exchange_parity,
	                          EmitBinaryU32(state, spv::OpBitwiseXor, parity, ConstantU32(state, 1)));
	return parity;
}

} // namespace

uint32_t ExchangeWaveValue(EmitterState& state, uint32_t value, uint32_t lane) {
	const auto parity = NextWaveExchangeParity(state);
	state.builder.AddFunction(spv::OpStore,
	                          WaveExchangeSlot(state, parity, EmitLocalInvocationIndex(state)), value);
	EmitWaveBarrier(state);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), result,
	                          WaveExchangeSlot(state, parity, lane));
	return result;
}

uint32_t EmitHostBallotWord(EmitterState& state, uint32_t predicate) {
	const auto ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), predicate);
	const auto word = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), word, ballot, 0);
	return word;
}

uint32_t EmitHalfBallot(EmitterState& state, uint32_t predicate) {
	const auto word   = EmitHostBallotWord(state, predicate);
	const auto zero   = ConstantU32(state, 0);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, word, word,
	                          zero, zero);
	return result;
}

uint32_t ExchangeWaveBallot(EmitterState& state, uint32_t predicate) {
	const auto word = EmitHostBallotWord(state, predicate);
	// Each lane publishes its half's ballot word; lanes 0 and 32 hold the two halves'.
	const auto parity = NextWaveExchangeParity(state);
	state.builder.AddFunction(spv::OpStore,
	                          WaveExchangeSlot(state, parity, EmitLocalInvocationIndex(state)), word);
	EmitWaveBarrier(state);
	std::array<uint32_t, 2> halves {};
	for (uint32_t half = 0; half < 2; half++) {
		halves[half] = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), halves[half],
		                          WaveExchangeSlot(state, parity, ConstantU32(state, half * 32)));
	}
	const auto zero   = ConstantU32(state, 0);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, halves[0],
	                          halves[1], zero, zero);
	return result;
}

uint32_t ReadHostSubgroupLane(EmitterState& state, uint32_t type, uint32_t value, uint32_t lane) {
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformShuffle, type, result,
	                          ConstantU32(state, spv::ScopeSubgroup), value,
	                          EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31)));
	return result;
}

uint32_t EmitHostLane(EmitterState& state, uint32_t lane) {
	if (!state.wave_clusters) {
		return lane;
	}
	const auto wave_base =
	    EmitBinaryU32(state, spv::OpBitwiseAnd, LoadHostSubgroupLane(state), ConstantU32(state, ~31u));
	return EmitBinaryU32(state, spv::OpBitwiseOr, wave_base,
	                     EmitBinaryU32(state, spv::OpBitwiseAnd, lane, ConstantU32(state, 31)));
}

uint32_t EmitWaveBallot(EmitterState& state, uint32_t ballot) {
	if (!state.wave_clusters) {
		return ballot;
	}
	const auto word = EmitBinaryU32(state, spv::OpShiftRightLogical, LoadHostSubgroupLane(state),
	                                ConstantU32(state, 5));
	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpVectorExtractDynamic, TypeU32(state), bits, ballot, word);
	const auto zero   = ConstantU32(state, 0);
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result, bits, zero,
	                          zero, zero);
	return result;
}

bool PixelHelperAtomicsSkipped(const EmitterState& state) {
	return state.program.stage == ShaderType::Pixel && GetCodegenOptions().ps_helper_atomics_skip &&
	       state.helper_invocation_variable != 0;
}

uint32_t EmitIsHelperInvocation(EmitterState& state) {
	if (state.helper_invocation_variable == 0) {
		EXIT("HelperInvocation was not declared before SPIR-V function emission\n");
	}
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeBool(state), value,
	                          state.helper_invocation_variable);
	return value;
}

uint32_t InputVariableForKind(const EmitterState& state, IR::StageInputKind kind) {
	for (const auto& input: state.inputs) {
		if (input.kind == kind) {
			return input.variable_id;
		}
	}
	return 0;
}

const InputBinding* InputBindingForParameter(const EmitterState& state, uint32_t location) {
	for (const auto& input: state.inputs) {
		if (input.kind == IR::StageInputKind::Parameter && input.location == location) {
			return &input;
		}
	}
	return nullptr;
}

uint32_t EmitInputComponentU32(EmitterState& state, IR::StageInputKind kind, uint32_t component) {
	const auto variable = InputVariableForKind(state, kind);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	const auto pointer = state.builder.AllocateId();
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	                          TypePointer(state, spv::StorageClassInput, TypeU32(state)), pointer,
	                          variable, ConstantU32(state, component));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t EmitLocalInvocationIndex(EmitterState& state) {
	const auto variable = InputVariableForKind(state, IR::StageInputKind::LocalInvocationIndex);
	if (variable == 0) {
		return ConstantU32(state, 0);
	}
	auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, variable);
	if (state.lane_count == 2) {
		const auto wave_base =
		    EmitBinaryU32(state, spv::OpBitwiseAnd, value, ConstantU32(state, ~31u));
		value = EmitAddU32(state, EmitAddU32(state, value, wave_base),
		                   ConstantU32(state, state.lane_half * 32));
	}
	if (state.mesh_pass_variable != 0) {
		// Each pass runs the next Waves() / passes waves of the subgroup.
		const auto& mesh = state.input_info.vertex->mesh;
		const auto  pass = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), pass, state.mesh_pass_variable);
		value = EmitAddU32(
		    state, value,
		    EmitBinaryU32(state, spv::OpIMul, pass,
		                  ConstantU32(state, mesh.Waves() / state.mesh_passes * mesh.wave_size)));
	}
	return value;
}

uint32_t EmitVertexParameterComponentU32(EmitterState& state, const InputBinding& input,
                                         uint32_t component) {
	const auto count = VertexParameterComponentCount(input);
	const auto kind  = VertexParameterScalarKind(state, input.location);
	const auto scalar_type = VertexParameterScalarType(state, kind);
	uint32_t   raw         = state.builder.AllocateId();
	if (count == 1u) {
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, input.variable_id);
	} else {
		const auto pointer_type = TypePointer(state, spv::StorageClassInput, scalar_type);
		const auto pointer      = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, pointer_type, pointer, input.variable_id,
		                          ConstantU32(state, component));
		state.builder.AddFunction(spv::OpLoad, scalar_type, raw, pointer);
	}

	if (kind == Format::ComponentType::Uint) {
		return raw;
	}

	const auto bits = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, raw);
	return bits;
}

uint32_t EmitSubgroupLaneActiveBool(EmitterState& state, uint32_t lane) {
	const auto active_ballot = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4), active_ballot,
	                          ConstantU32(state, spv::ScopeSubgroup), ConstantBool(state, true));
	return EmitBallotLaneActiveBool(state, EmitWaveBallot(state, active_ballot), lane);
}
uint32_t EmitBallotLaneActiveBool(EmitterState& state, uint32_t active_ballot, uint32_t lane) {
	const auto low = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), low, active_ballot, 0);
	uint32_t mask = low;
	if (state.program.wave_size == 64u) {
		const auto high     = state.builder.AllocateId();
		const auto in_high  = state.builder.AllocateId();
		const auto selected = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), high, active_ballot, 1);
		state.builder.AddFunction(spv::OpUGreaterThanEqual, TypeBool(state), in_high, lane,
		                          ConstantU32(state, 32));
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), selected, in_high, high, low);
		mask = selected;
	}

	const auto lane_low = state.builder.AllocateId();
	const auto bit      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), lane_low, lane,
	                          ConstantU32(state, 31));
	state.builder.AddFunction(spv::OpShiftLeftLogical, TypeU32(state), bit, ConstantU32(state, 1),
	                          lane_low);

	const auto hit      = state.builder.AllocateId();
	const auto active   = state.builder.AllocateId();
	const auto in_range = state.builder.AllocateId();
	const auto ret      = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitwiseAnd, TypeU32(state), hit, mask, bit);
	state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), active, hit,
	                          ConstantU32(state, 0));
	state.builder.AddFunction(spv::OpULessThan, TypeBool(state), in_range, lane,
	                          ConstantU32(state, state.program.wave_size));
	state.builder.AddFunction(spv::OpLogicalAnd, TypeBool(state), ret, active, in_range);
	return ret;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
