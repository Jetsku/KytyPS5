#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <functional>

// Call-split programs (CallSplit.h): the call state is a plain device buffer reached through
// push-data device addresses. Its accesses ignore EXEC and the BDA page table: every lane of a
// wave saves and restores its registers, active or not.
namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t CallStatePointer(EmitterState& state, uint32_t address) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpConvertUToPtr, TypePhysicalU32Pointer(state), pointer, address);
	return pointer;
}

void StoreCallState(EmitterState& state, uint32_t address, uint32_t value) {
	state.builder.AddFunction(spv::OpStore, CallStatePointer(state, address), value,
	                          spv::MemoryAccessAlignedMask, uint32_t {4});
}

} // namespace

uint32_t EmitCallStateParam(EmitterState& state, IR::Value index) {
	const auto start = state.program.bindings.call_state_dword;
	EXIT_IF(start == IR::PushData::NoStart || !index.IsImmediate() ||
	        index.U32() >= IR::BindingLayout::CallStateDwords);
	return EmitShaderDataDwordLoad(state, start + index.U32());
}

uint32_t EmitCallStateLoad(EmitterState& state, uint32_t address) {
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, CallStatePointer(state, address),
	                          spv::MemoryAccessAlignedMask, uint32_t {4});
	return value;
}

void EmitCallStateStore(EmitterState& state, uint32_t address, uint32_t value) {
	StoreCallState(state, address, value);
}

void EmitCallStateStoreChanged(EmitterState& state, uint32_t address, uint32_t value,
                               uint32_t original) {
	if (value == original) {
		return;
	}
	const auto changed = Binary(state, spv::OpINotEqual, TypeBool(state), value, original);
	EmitIfCondition(state, changed, [&] { StoreCallState(state, address, value); });
}

void EmitCallStateProbe(ValueEmitContext& ctx, const IR::Inst& inst) {
	ctx.Fail(inst, "is a call-split planning probe");
}

// KYTY_CALL_PARTITIONS: the leader lane inserts `key` into the report set at `base` (an open hash
// set of IR::CallPartitionReportSlots words, zero = free) with up to eight linear probes.
void EmitCallStateReport(EmitterState& state, uint32_t base, uint32_t key, uint32_t leader) {
	EmitIfCondition(state, leader, [&] {
		const auto hash = EmitBinaryU32(
		    state, spv::OpShiftRightLogical,
		    EmitBinaryU32(state, spv::OpIMul, key, ConstantU32(state, 0x9e3779b1u)), ConstantU32(state, 23u));
		std::function<void(uint32_t)> probe = [&](uint32_t i) {
			const auto slot = EmitBinaryU32(
			    state, spv::OpBitwiseAnd, EmitBinaryU32(state, spv::OpIAdd, hash, ConstantU32(state, i)),
			    ConstantU32(state, IR::CallPartitionReportSlots - 1u));
			const auto bytes = Unary(state, spv::OpUConvert, TypeU64(state),
			                         EmitBinaryU32(state, spv::OpShiftLeftLogical, slot, ConstantU32(state, 2u)));
			const auto address = Binary(state, spv::OpIAdd, TypeU64(state), base, bytes);
			const auto old     = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpAtomicCompareExchange, TypeU32(state), old,
			                          CallStatePointer(state, address), ConstantU32(state, spv::ScopeDevice),
			                          ConstantU32(state, spv::MemorySemanticsMaskNone),
			                          ConstantU32(state, spv::MemorySemanticsMaskNone), key, ConstantU32(state, 0u));
			if (i + 1u >= 8u) return;
			const auto taken = Binary(state, spv::OpLogicalAnd, TypeBool(state),
			                          Binary(state, spv::OpINotEqual, TypeBool(state), old, ConstantU32(state, 0u)),
			                          Binary(state, spv::OpINotEqual, TypeBool(state), old, key));
			EmitIfCondition(state, taken, [&] { probe(i + 1u); });
		};
		probe(0);
	});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
