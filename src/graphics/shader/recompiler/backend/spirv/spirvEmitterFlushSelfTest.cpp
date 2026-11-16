// The KYTY_F32_FLUSH_HOST start-up self-test module (recompiler/F32FlushSelfTest.h), made with the
// emitter's own module header (DeclareFloatControls) and instruction helpers, so that it runs in
// the float mode and with the instruction forms of a guest program without f64.
#include "graphics/shader/recompiler/F32FlushSelfTest.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest {

namespace {

using Spirv::Emitter::EmitterState;
namespace E = Spirv::Emitter;

using CompareEmit = uint32_t (*)(EmitterState&, uint32_t, uint32_t);

// EmitFloatCompare32 on two register operands with KYTY_F32_FLUSH_HOST: the bare opcode
// (spirvEmitterInstructions.h), in CompareOp order.
template <spv::Op opcode>
uint32_t Compare(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return E::EmitNative<opcode, IR::Type::U1>(state, lhs, rhs);
}

constexpr CompareEmit CompareForms[CompareOps] = {
    Compare<spv::OpFOrdEqual>,           Compare<spv::OpFUnordEqual>,
    Compare<spv::OpFOrdNotEqual>,        Compare<spv::OpFUnordNotEqual>,
    Compare<spv::OpFOrdLessThan>,        Compare<spv::OpFUnordLessThan>,
    Compare<spv::OpFOrdGreaterThan>,     Compare<spv::OpFUnordGreaterThan>,
    Compare<spv::OpFOrdLessThanEqual>,   Compare<spv::OpFUnordLessThanEqual>,
    Compare<spv::OpFOrdGreaterThanEqual>, Compare<spv::OpFUnordGreaterThanEqual>,
};

static_assert(F32FlushClassRcp == 1u << static_cast<uint32_t>(UnaryOp::Rcp) &&
              F32FlushClassRsq == 1u << static_cast<uint32_t>(UnaryOp::Rsq) &&
              F32FlushClassSqrt == 1u << static_cast<uint32_t>(UnaryOp::Sqrt) &&
              F32FlushClassExp == 1u << static_cast<uint32_t>(UnaryOp::Exp) &&
              F32FlushClassLog == 1u << static_cast<uint32_t>(UnaryOp::Log) &&
              F32FlushClassCompare == ClassCompare && F32FlushClassAll == ClassAll &&
              F32FlushClassNoPreserve16 == ClassNoPreserve16);

using UnaryEmit = uint32_t (*)(EmitterState&, uint32_t);
// V_RCP/RSQ/SQRT/EXP/LOG_F32 as the translator lowers them (UnaryOp order).
constexpr UnaryEmit UnaryForms[UnaryOps] = {E::EmitFPRecip32, E::EmitFPRecipSqrt32, E::EmitFPSqrt,
                                            E::EmitFPExp2, E::EmitFPLog2};

} // namespace

std::vector<uint32_t> EmitModule(std::string* error) {
	IR::Program program;
	program.stage = ShaderType::Compute;
	ShaderComputeInputInfo compute {};
	compute.threads_num[0] = WorkgroupSize;
	compute.threads_num[1] = 1;
	compute.threads_num[2] = 1;
	ShaderStageInputInfo input {};
	input.compute = &compute;
	EmitterState state(program, input);
	if (!GetCodegenOptions().f32_flush_host || GetCodegenOptions().f32_flush_host_classes != F32FlushClassAll ||
	    !E::F32FlushHost(state, F32FlushClassAll)) {
		if (error != nullptr) {
			*error = "the emitter's host-flush path is not on for every op class (KYTY_F32_FLUSH_HOST)";
		}
		return {};
	}
	const auto flushes_before = Spirv::EmittedDenormFlushes(1);
	auto&      b              = state.builder;

	const auto u32      = E::TypeU32(state);
	const auto f32      = E::TypeF32(state);
	const auto uvec3    = E::TypeU32Vector(state, 3);
	const auto words    = b.DecoratedType(spv::OpTypeRuntimeArray,
	                                      {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, u32);
	const auto block    = b.DecoratedType(spv::OpTypeStruct,
	                                      {{spv::OpDecorate, {spv::DecorationBlock}},
	                                       {spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}}},
	                                      words);
	const auto buffer   = b.DefineGlobalVariable(E::TypePointer(state, spv::StorageClassStorageBuffer, block),
	                                             spv::StorageClassStorageBuffer);
	const auto word_ptr = E::TypePointer(state, spv::StorageClassStorageBuffer, u32);
	const auto global_id =
	    b.DefineGlobalVariable(E::TypePointer(state, spv::StorageClassInput, uvec3), spv::StorageClassInput);
	b.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationDescriptorSet, 0u);
	b.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationBinding, 0u);
	b.AddAnnotation(spv::OpDecorate, global_id, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);

	state.main_func = b.AllocateId();
	b.RequireCapability(spv::CapabilityShader);
	b.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	b.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", {global_id});
	// The float mode of a program without f64 (SignedZeroInfNanPreserve 32, and the denormal modes
	// this device takes: DenormFlushToZero 32 only where the device declares it).
	E::DeclareFloatControls(state);
	b.AddExecutionMode(state.main_func, spv::ExecutionModeLocalSize, WorkgroupSize, 1u, 1u);
	b.AddName(state.main_func, "main");

	b.AddFunction(spv::OpFunction, E::TypeVoid(state), state.main_func, spv::FunctionControlMaskNone,
	              E::TypeFunction(state));
	E::EmitLabel(state, b.AllocateId());
	const auto id    = E::Unary(state, spv::OpLoad, uvec3, global_id);
	const auto index = b.AllocateId();
	b.AddFunction(spv::OpCompositeExtract, u32, index, id, 0u);

	const auto pointer = [&](uint32_t word) {
		const auto result = b.AllocateId();
		b.AddFunction(spv::OpAccessChain, word_ptr, result, buffer, E::ConstantU32(state, 0u), word);
		return result;
	};
	const auto load = [&](uint32_t word) { return E::Unary(state, spv::OpLoad, u32, pointer(word)); };
	const auto plane_word = [&](uint32_t plane) {
		return E::Binary(state, spv::OpIAdd, u32, index, E::ConstantU32(state, PlaneWord(plane, 0)));
	};
	const auto store = [&](uint32_t plane, uint32_t value) {
		b.AddFunction(spv::OpStore, pointer(plane_word(plane)), value);
	};
	// The reference operand: the zero of the same sign for a denormal, by integer ops (not the
	// emitter's flush, which KYTY_CS_FLUSH=off or KYTY_RELAXED_FLOAT leave out).
	const auto flushed = [&](uint32_t bits) {
		const auto magnitude = E::Binary(state, spv::OpBitwiseAnd, u32, bits, E::ConstantU32(state, 0x7fffffffu));
		const auto sign      = E::Binary(state, spv::OpBitwiseAnd, u32, bits, E::ConstantU32(state, 0x80000000u));
		const auto small =
		    E::Binary(state, spv::OpULessThan, E::TypeBool(state), magnitude, E::ConstantU32(state, 0x00800000u));
		const auto selected = b.AllocateId();
		b.AddFunction(spv::OpSelect, u32, selected, small, sign, bits);
		return E::Unary(state, spv::OpBitcast, f32, selected);
	};

	const auto x_bits = load(E::Binary(state, spv::OpIAdd, u32, index, E::ConstantU32(state, InputBase)));
	const auto x      = E::Unary(state, spv::OpBitcast, f32, x_bits);
	const auto x_zero = flushed(x_bits);
	for (uint32_t op = 0; op < UnaryOps; op++) {
		const auto unary    = static_cast<UnaryOp>(op);
		const auto actual   = UnaryForms[op](state, x);
		const auto expected = UnaryForms[op](state, x_zero);
		store(UnaryActualPlane(unary), E::Unary(state, spv::OpBitcast, u32, actual));
		store(UnaryExpectedPlane(unary), E::Unary(state, spv::OpBitcast, u32, expected));
	}

	std::array<uint32_t, OperandCount> y {};
	std::array<uint32_t, OperandCount> y_zero {};
	for (uint32_t j = 0; j < OperandCount; j++) {
		const auto bits = load(E::ConstantU32(state, OperandBase + j));
		y[j]            = E::Unary(state, spv::OpBitcast, f32, bits);
		y_zero[j]       = flushed(bits);
	}
	for (uint32_t op = 0; op < CompareOps; op++) {
		const auto compare = static_cast<CompareOp>(op);
		for (uint32_t order = 0; order < 2; order++) {
			for (const bool expected: {false, true}) {
				const auto lhs  = expected ? x_zero : x;
				auto       mask = E::ConstantU32(state, 0u);
				for (uint32_t j = 0; j < OperandCount; j++) {
					const auto rhs    = expected ? y_zero[j] : y[j];
					const auto result = order == 0 ? CompareForms[op](state, lhs, rhs) : CompareForms[op](state, rhs, lhs);
					const auto bit    = b.AllocateId();
					b.AddFunction(spv::OpSelect, u32, bit, result, E::ConstantU32(state, 1u << j),
					              E::ConstantU32(state, 0u));
					mask = E::Binary(state, spv::OpBitwiseOr, u32, mask, bit);
				}
				store(ComparePlane(compare, order, expected), mask);
			}
		}
	}
	b.AddFunction(spv::OpReturn);
	b.AddFunction(spv::OpFunctionEnd);

	if (Spirv::EmittedDenormFlushes(1) != flushes_before) {
		if (error != nullptr) {
			*error = "the emitter added a shader-side flush to the tested ops";
		}
		return {};
	}
	return b.Build();
}

std::vector<uint32_t> EmitF16Module() {
	IR::Program program;
	program.stage = ShaderType::Compute;
	ShaderComputeInputInfo compute {};
	compute.threads_num[0] = WorkgroupSize;
	compute.threads_num[1] = 1;
	compute.threads_num[2] = 1;
	ShaderStageInputInfo input {};
	input.compute = &compute;
	EmitterState state(program, input);
	auto&        b = state.builder;

	const auto u32   = E::TypeU32(state);
	const auto f32   = E::TypeF32(state);
	const auto uvec3 = E::TypeU32Vector(state, 3);
	const auto words = b.DecoratedType(spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, u32);
	const auto block = b.DecoratedType(spv::OpTypeStruct,
	                                   {{spv::OpDecorate, {spv::DecorationBlock}},
	                                    {spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}}},
	                                   words);
	const auto buffer =
	    b.DefineGlobalVariable(E::TypePointer(state, spv::StorageClassStorageBuffer, block), spv::StorageClassStorageBuffer);
	const auto word_ptr = E::TypePointer(state, spv::StorageClassStorageBuffer, u32);
	const auto global_id =
	    b.DefineGlobalVariable(E::TypePointer(state, spv::StorageClassInput, uvec3), spv::StorageClassInput);
	b.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationDescriptorSet, 0u);
	b.AddAnnotation(spv::OpDecorate, buffer, spv::DecorationBinding, 0u);
	b.AddAnnotation(spv::OpDecorate, global_id, spv::DecorationBuiltIn, spv::BuiltInGlobalInvocationId);

	state.main_func = b.AllocateId();
	b.RequireCapability(spv::CapabilityShader);
	b.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
	b.AddEntryPoint(spv::ExecutionModelGLCompute, state.main_func, "main", {global_id});
	E::DeclareFloatControls(state);
	b.AddExecutionMode(state.main_func, spv::ExecutionModeLocalSize, WorkgroupSize, 1u, 1u);
	b.AddName(state.main_func, "main");
	b.AddFunction(spv::OpFunction, E::TypeVoid(state), state.main_func, spv::FunctionControlMaskNone,
	              E::TypeFunction(state));
	E::EmitLabel(state, b.AllocateId());
	const auto id    = E::Unary(state, spv::OpLoad, uvec3, global_id);
	const auto index = b.AllocateId();
	b.AddFunction(spv::OpCompositeExtract, u32, index, id, 0u);
	const auto pointer = [&](uint32_t word) {
		const auto result = b.AllocateId();
		b.AddFunction(spv::OpAccessChain, word_ptr, result, buffer, E::ConstantU32(state, 0u), word);
		return result;
	};
	const auto at = [&](uint32_t base, uint32_t scale, uint32_t add) {
		auto word = index;
		if (scale != 1u) {
			word = E::Binary(state, spv::OpIMul, u32, word, E::ConstantU32(state, scale));
		}
		return E::Binary(state, spv::OpIAdd, u32, word, E::ConstantU32(state, base + add));
	};
	const auto load  = [&](uint32_t word) { return E::Unary(state, spv::OpLoad, u32, pointer(word)); };
	const auto store = [&](uint32_t word, uint32_t value) { b.AddFunction(spv::OpStore, pointer(word), value); };

	// Unpack: EmitF16BitsToF32 of the low half and of the high half (shifted down, as the
	// translator reads an f16 operand's high half).
	const auto word = load(at(F16UnpackBase, 1u, 0u));
	const auto low  = E::EmitF16BitsToF32(state, word);
	const auto high = E::EmitF16BitsToF32(
	    state, E::Binary(state, spv::OpShiftRightLogical, u32, word, E::ConstantU32(state, 16u)));
	store(at(F16UnpackOutBase, 2u, 0u), E::Unary(state, spv::OpBitcast, u32, low));
	store(at(F16UnpackOutBase, 2u, 1u), E::Unary(state, spv::OpBitcast, u32, high));
	// Pack: EmitGlslPackHalf2x16 of (x, y).
	const auto x    = E::Unary(state, spv::OpBitcast, f32, load(at(F16PackBase, 2u, 0u)));
	const auto y    = E::Unary(state, spv::OpBitcast, f32, load(at(F16PackBase, 2u, 1u)));
	const auto pair = b.AllocateId();
	b.AddFunction(spv::OpCompositeConstruct, E::TypeF32Vector(state, 2), pair, x, y);
	store(at(F16PackOutBase, 1u, 0u), E::EmitGlslPackHalf2x16(state, pair));
	b.AddFunction(spv::OpReturn);
	b.AddFunction(spv::OpFunctionEnd);
	return b.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::FlushSelfTest
