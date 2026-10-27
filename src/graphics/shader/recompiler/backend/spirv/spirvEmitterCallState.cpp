#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

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

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
