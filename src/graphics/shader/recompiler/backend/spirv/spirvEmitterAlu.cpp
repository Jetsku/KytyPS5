#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <array>
#include <bit>
#include <optional>

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

struct Pair {
	uint32_t low  = 0;
	uint32_t high = 0;
};

Pair ExtractPair(EmitterState& state, uint32_t value) {
	Pair result {state.builder.AllocateId(), state.builder.AllocateId()};
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result.low, value, 0);
	state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), result.high, value, 1);
	return result;
}

uint32_t EmitMinMaxF64(EmitterState& state, uint32_t lhs, uint32_t rhs, bool max_value) {
	const auto bits_type = TypeU32Vector(state, 2);
	const auto lhs_bits = Unary(state, spv::OpBitcast, bits_type, lhs);
	const auto rhs_bits = Unary(state, spv::OpBitcast, bits_type, rhs);
	const auto ordered = Binary(state, max_value ? spv::OpFOrdGreaterThan : spv::OpFOrdLessThan,
	                            TypeBool(state), lhs, rhs);
	auto result = Select(state, TypeF64(state), ordered, lhs, rhs);

	// Equal values have identical bits except signed zero: min chooses -0, max chooses +0.
	const auto equal = Binary(state, spv::OpFOrdEqual, TypeBool(state), lhs, rhs);
	const auto equal_bits = Binary(state, max_value ? spv::OpBitwiseAnd : spv::OpBitwiseOr,
	                               bits_type, lhs_bits, rhs_bits);
	result = Select(state, TypeF64(state), equal,
	                Unary(state, spv::OpBitcast, TypeF64(state), equal_bits), result);

	// Non-IEEE mode selects the other operand for NaN, including rhs when both are NaN.
	result = Select(state, TypeF64(state), Unary(state, spv::OpIsNan, TypeBool(state), rhs),
	                lhs, result);
	return Select(state, TypeF64(state), Unary(state, spv::OpIsNan, TypeBool(state), lhs),
	              rhs, result);
}

uint32_t EmitMulHigh(EmitterState& state, uint32_t lhs, uint32_t rhs, bool signed_value) {
	const auto operand_type = signed_value ? TypeI32(state) : TypeU32(state);
	const auto pair_type    = signed_value ? TypeI32Pair(state) : TypeU32Pair(state);
	uint32_t   lhs_operand  = lhs;
	uint32_t   rhs_operand  = rhs;
	if (signed_value) {
		lhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), lhs);
		rhs_operand = Unary(state, spv::OpBitcast, TypeI32(state), rhs);
	}
	const auto extended = state.builder.AllocateId();
	state.builder.AddFunction(signed_value ? spv::OpSMulExtended : spv::OpUMulExtended, pair_type,
	                          extended, lhs_operand, rhs_operand);
	const auto high = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, operand_type, high, extended, 1);
	return signed_value ? Unary(state, spv::OpBitcast, TypeU32(state), high) : high;
}

uint32_t EmitMinMax3(EmitterState& state, uint32_t a, uint32_t b, uint32_t c, bool signed_value,
                     bool max_value) {
	const auto ab = signed_value ? EmitMinMaxI32Value(state, a, b, max_value)
	                             : EmitMinMaxU32Value(state, a, b, max_value);
	return signed_value ? EmitMinMaxI32Value(state, ab, c, max_value)
	                    : EmitMinMaxU32Value(state, ab, c, max_value);
}

uint32_t EmitMed3(EmitterState& state, uint32_t a, uint32_t b, uint32_t c, bool signed_value) {
	const auto minimum = EmitMinMax3(state, a, b, c, signed_value, false);
	const auto maximum = EmitMinMax3(state, a, b, c, signed_value, true);
	const auto ab      = Binary(state, spv::OpIAdd, TypeU32(state), a, b);
	const auto abc     = Binary(state, spv::OpIAdd, TypeU32(state), ab, c);
	return Binary(state, spv::OpISub, TypeU32(state),
	              Binary(state, spv::OpISub, TypeU32(state), abc, minimum), maximum);
}

uint32_t EmitFMinMax3(EmitterState& state, uint32_t a, uint32_t b, uint32_t c, bool max_value) {
	return EmitMinMaxF32Value(state, EmitMinMaxF32Value(state, a, b, max_value), c, max_value);
}

// An f32 operand of a min/max, with its bits when it is a compile-time constant.
struct F32Operand {
	uint32_t                id = 0;
	std::optional<uint32_t> bits;
};

F32Operand OperandF32(ValueEmitContext& ctx, IR::Value value) {
	F32Operand operand {.id = ctx.Def(value)};
	const auto resolved = value.Resolve();
	if (resolved.IsImmediate() && resolved.GetType() == IR::Type::F32) {
		operand.bits = std::bit_cast<uint32_t>(resolved.F32Value());
	}
	return operand;
}

bool IsNonNanConstant(const F32Operand& operand) {
	return operand.bits.has_value() && (*operand.bits & 0x7fffffffu) <= 0x7f800000u;
}

// A constant that is neither NaN nor a zero: it never pairs with another zero.
bool IsPlainConstant(const F32Operand& operand) {
	return IsNonNanConstant(operand) && (*operand.bits & 0x7fffffffu) != 0u;
}

// V_MIN/V_MAX_F32 exactly as EmitMinMaxF32Value defines them: a NaN lhs yields rhs, a NaN rhs
// yields lhs, two zeros (by bits) yield lhs|rhs for min and lhs&rhs for max, and otherwise
// FOrdLessThan(lhs, rhs) / FOrdGreaterThanEqual(lhs, rhs) picks lhs. The very same compare
// instruction decides here, while the four classification chains disappear:
//  - pick lhs = (Ord(lhs op rhs) || isnan(rhs)) && !isnan(lhs).
//  - two zeros: one integer test on (lhs|rhs) & 0x7fffffff selects lhs|rhs or lhs&rhs.
// Keeping the identical compare matters for denormals: the NVIDIA driver flushes them in
// register-register compares but may lower a compare against a constant to an integer test that
// does not, so any other formulation (GLSL NMin/NMax, unordered compares) can pick the other
// operand when a zero meets a denormal.
// A constant operand drops what cannot apply: a non-NaN constant cannot be the NaN operand, and a
// non-zero constant never forms two zeros.
F32Operand EmitFastMinMaxF32(EmitterState& state, F32Operand lhs, F32Operand rhs,
                             bool max_value) {
	auto pick = Binary(state, max_value ? spv::OpFOrdGreaterThanEqual : spv::OpFOrdLessThan,
	                   TypeBool(state), lhs.id, rhs.id);
	if (!IsNonNanConstant(rhs)) {
		const auto rhs_nan = Unary(state, spv::OpIsNan, TypeBool(state), rhs.id);
		pick               = Binary(state, spv::OpLogicalOr, TypeBool(state), pick, rhs_nan);
		if (!IsNonNanConstant(lhs)) {
			const auto lhs_ordered = Binary(state, spv::OpFOrdEqual, TypeBool(state), lhs.id, lhs.id);
			pick = Binary(state, spv::OpLogicalAnd, TypeBool(state), pick, lhs_ordered);
		}
	}
	// Select raw bits, as the legacy code does, so the driver cannot turn the compare-and-select
	// back into a min/max instruction with its own denormal and signed-zero rules.
	const auto lhs_bits = EmitBitcastF32ToU32(state, lhs.id);
	const auto rhs_bits = EmitBitcastF32ToU32(state, rhs.id);
	auto       value    = EmitSelectValueU32(state, pick, lhs_bits, rhs_bits);
	if (!IsPlainConstant(lhs) && !IsPlainConstant(rhs)) {
		const auto either = Binary(state, spv::OpBitwiseOr, TypeU32(state), lhs_bits, rhs_bits);
		const auto both_zero = EmitCompareU32Constant(
		    state, spv::OpIEqual, EmitAndConstant(state, either, 0x7fffffffu), 0u);
		const auto combined =
		    max_value ? Binary(state, spv::OpBitwiseAnd, TypeU32(state), lhs_bits, rhs_bits) : either;
		value = EmitSelectValueU32(state, both_zero, combined, value);
	}
	return {.id = EmitBitcastU32ToF32(state, value)};
}

F32Operand EmitFastMinMax3F32(EmitterState& state, F32Operand a, F32Operand b, F32Operand c,
                              bool max_value) {
	return EmitFastMinMaxF32(state, EmitFastMinMaxF32(state, a, b, max_value), c, max_value);
}

// V_MED3_F32 exactly as EmitFPMedTri32 defines it (any NaN operand gives min3, otherwise
// max(min(a,b), min(max(a,b),c))), from the exact min/max above. Constant non-NaN operands drop
// out of the NaN test.
uint32_t EmitFastMedTri32(EmitterState& state, F32Operand a, F32Operand b, F32Operand c) {
	const auto min_ab   = EmitFastMinMaxF32(state, a, b, false);
	const auto min3     = EmitFastMinMaxF32(state, min_ab, c, false);
	const auto max_ab   = EmitFastMinMaxF32(state, a, b, true);
	const auto high_min = EmitFastMinMaxF32(state, max_ab, c, false);
	const auto median   = EmitFastMinMaxF32(state, min_ab, high_min, true);
	uint32_t   any_nan  = 0;
	for (const auto* operand: {&a, &b, &c}) {
		if (IsNonNanConstant(*operand)) {
			continue;
		}
		const auto nan = Unary(state, spv::OpIsNan, TypeBool(state), operand->id);
		any_nan = any_nan == 0 ? nan : Binary(state, spv::OpLogicalOr, TypeBool(state), any_nan, nan);
	}
	if (any_nan == 0) {
		return median.id;
	}
	return Select(state, TypeF32(state), any_nan, min3.id, median.id);
}

bool FastFloatMinMax() {
	return GetCodegenOptions().fast_float_min_max;
}

uint32_t EmitExt(EmitterState& state, uint32_t type, uint32_t opcode,
                 std::initializer_list<uint32_t> args) {
	const auto            result = state.builder.AllocateId();
	std::vector<uint32_t> words {spv::OpExtInst, type, result, GlslStd450(state), opcode};
	words.insert(words.end(), args.begin(), args.end());
	state.builder.AddFunction(words);
	return result;
}

uint32_t EmitF32ToU32(EmitterState& state, uint32_t src, bool signed_value) {
	const auto trunc         = EmitTruncF32Value(state, src);
	const auto converted_raw = state.builder.AllocateId();
	if (signed_value) {
		const auto converted_i = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpConvertFToS, TypeI32(state), converted_i, trunc);
		state.builder.AddFunction(spv::OpBitcast, TypeU32(state), converted_raw, converted_i);
	} else {
		state.builder.AddFunction(spv::OpConvertFToU, TypeU32(state), converted_raw, trunc);
	}
	const auto nan = EmitClassifyF32(state, src).nan;
	if (signed_value) {
		const auto below = Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), src,
		                          ConstantF32(state, 0xcf000000u));
		const auto above = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), src,
		                          ConstantF32(state, 0x4f000000u));
		const auto high =
		    Select(state, TypeU32(state), above, ConstantU32(state, 0x7fffffffu), converted_raw);
		const auto low =
		    Select(state, TypeU32(state), below, ConstantU32(state, 0x80000000u), high);
		return Select(state, TypeU32(state), nan, ConstantU32(state, 0), low);
	}
	const auto below =
	    Binary(state, spv::OpFOrdLessThanEqual, TypeBool(state), src, ConstantF32(state, 0));
	const auto above = Binary(state, spv::OpFOrdGreaterThanEqual, TypeBool(state), src,
	                          ConstantF32(state, 0x4f800000u));
	const auto zero  = Binary(state, spv::OpLogicalOr, TypeBool(state), nan, below);
	const auto high =
	    Select(state, TypeU32(state), above, ConstantU32(state, 0xffffffffu), converted_raw);
	return Select(state, TypeU32(state), zero, ConstantU32(state, 0), high);
}

} // namespace
uint32_t EmitFPMedTri32(ValueEmitContext& ctx, IR::Value arg0, IR::Value arg1, IR::Value arg2) {
	auto& state = ctx.state;
	if (FastFloatMinMax()) {
		return EmitFastMedTri32(state, OperandF32(ctx, arg0), OperandF32(ctx, arg1),
		                          OperandF32(ctx, arg2));
	}
	const auto a        = ctx.Def(arg0);
	const auto b        = ctx.Def(arg1);
	const auto c        = ctx.Def(arg2);
	const auto min_ab   = EmitMinMaxF32Value(state, a, b, false);
	const auto min3     = EmitMinMaxF32Value(state, min_ab, c, false);
	const auto max_ab   = EmitMinMaxF32Value(state, a, b, true);
	const auto high_min = EmitMinMaxF32Value(state, max_ab, c, false);
	const auto median   = EmitMinMaxF32Value(state, min_ab, high_min, true);
	const auto nan_ab   = Binary(state, spv::OpLogicalOr, TypeBool(state),
	                             EmitClassifyF32(state, a).nan, EmitClassifyF32(state, b).nan);
	const auto any_nan =
	    Binary(state, spv::OpLogicalOr, TypeBool(state), nan_ab, EmitClassifyF32(state, c).nan);
	return Select(state, TypeF32(state), any_nan, min3, median);
}

uint32_t EmitFindUMsb64(EmitterState& state, uint32_t value) {
	// GLSL FindUMsb is restricted to 32-bit components.
	const auto unpacked = Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), value);
	const auto pair =
	    ExtractPair(state, EmitExt(state, TypeU32Vector(state, 2), GLSLstd450FindUMsb, {unpacked}));
	const auto high_found =
	    Binary(state, spv::OpINotEqual, TypeBool(state), pair.high, ConstantU32(state, UINT32_MAX));
	return Select(state, TypeU32(state), high_found,
	              Binary(state, spv::OpIAdd, TypeU32(state), pair.high, ConstantU32(state, 32)),
	              pair.low);
}

uint32_t EmitConvertU16U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U16>(state, arg0, ConstantU32(state, 0xffffu));
}

uint32_t EmitConvertU8U32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpBitwiseAnd, IR::Type::U8>(state, arg0, ConstantU32(state, 0xffu));
}

uint32_t EmitConvertF16F32(EmitterState& state, uint32_t arg0) {
	const auto pair = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 2), pair, arg0,
	                          ConstantF32(state, 0));
	return EmitPackHalf2x16(state, pair);
}

// ConvertS32F32/ConvertU32F32 are only produced by Translator::ConvertF32ToI32Saturated and
// ConvertF32ToU32Saturated, which already replace NaN, truncate, and clamp the operand to an
// integral value inside the destination range (at most 2147483520 / 4294967040, at least
// -2^31 / 0), and select the saturated result themselves. For such operands a bare conversion is
// exact and EmitF32ToU32's second trunc/NaN/range layer never changes the value.
// KYTY_SINGLE_F2I_SATURATION=0 keeps that second layer.
uint32_t EmitConvertS32F32(EmitterState& state, uint32_t arg0) {
	if (GetCodegenOptions().single_f2i_saturation) {
		const auto converted = Unary(state, spv::OpConvertFToS, TypeI32(state), arg0);
		return Unary(state, spv::OpBitcast, TypeU32(state), converted);
	}
	return EmitF32ToU32(state, arg0, true);
}

uint32_t EmitConvertU32F32(EmitterState& state, uint32_t arg0) {
	if (GetCodegenOptions().single_f2i_saturation) {
		return Unary(state, spv::OpConvertFToU, TypeU32(state), arg0);
	}
	return EmitF32ToU32(state, arg0, false);
}

uint32_t EmitConvertF32F64(EmitterState& state, uint32_t arg0) {
	const auto converted = Unary(state, spv::OpFConvert, TypeF32(state), arg0);
	const auto source    = EmitNative<spv::OpCompositeExtract, IR::Type::U32>(
	    state, Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), arg0), 1u);
	const auto exponent = EmitAndConstant(state, source, 0x7ff00000u);
	const auto overflow =
	    Binary(state, spv::OpLogicalAnd, TypeBool(state),
	           EmitCompareU32Constant(state, spv::OpUGreaterThan, exponent, 0x47e00000u),
	           EmitCompareU32Constant(state, spv::OpULessThan, exponent, 0x7ff00000u));
	const auto clamped = EmitOrU32(state, EmitAndConstant(state, source, 0x80000000u),
	                               ConstantU32(state, 0x7f7fffffu));
	return EmitFlushF32DenormToSignedZero(
	    state, Select(state, TypeF32(state), overflow,
	                  Unary(state, spv::OpBitcast, TypeF32(state), clamped), converted));
}

uint32_t EmitConvertF64F32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpFConvert, IR::Type::F64>(state,
	                                                  EmitFlushF32DenormToSignedZero(state, arg0));
}

uint32_t EmitCompositeConstructU64(ValueEmitContext& ctx, uint32_t arg0, IR::Value arg1) {
	arg1 = arg1.Resolve();
	return arg1.IsImmediate() && arg1.U32() == 0u
	           ? Unary(ctx.state, spv::OpUConvert, TypeU64(ctx.state), arg0)
	           : PackU64(ctx.state, arg0, ctx.Def(arg1));
}

uint32_t EmitCompositeExtractU64(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	const auto value = arg1.U32() == 0u ? arg0
	                                    : Binary(state, spv::OpShiftRightLogical, TypeU64(state),
	                                             arg0, ConstantU32(state, 32));
	return Unary(state, spv::OpUConvert, TypeU32(state), value);
}

uint32_t EmitCompositeExtractU32x2(EmitterState& state, uint32_t arg0, IR::Value arg1) {
	return EmitNative<spv::OpCompositeExtract, IR::Type::U32>(state, arg0, arg1.U32());
}

namespace {

// One V_CVT_PKRTZ_F16_F32 half, exactly as EmitF32ToF16RtzBits computes it, with fewer selects:
//  - |x| >= 2^-14 (f32 exponent >= 113): the truncated f16 magnitude is (|bits| >> 13) - (112 << 10).
//    Finite values past the f16 range (exponent >= 143) give at least 0x7c00 there, so
//    UMin(., 0x7bff) produces the RTZ saturation.
//  - |x| < 2^-14 (exponent <= 112): the f16 subnormal (mantissa | hidden) >> (126 - exponent),
//    shift limited to 31; exponents <= 102, including f32 zeros and denormals, shift everything
//    out and leave the signed zero.
//  - Infinity and NaN (exponent 255): (|bits| >> 13) & 0x7fff is 0x7c00 plus the top ten payload
//    bits; NaNs also get the quiet bit 0x0200.
// Pure integer arithmetic on the input bits, so the result does not depend on host rounding,
// denormal handling or NaN canonicalization. (A PackHalf2x16-plus-correction variant was not
// exact on NVIDIA: the driver folds UnpackHalf2x16(PackHalf2x16(x)) back to x, which hides the
// rounding direction.)
uint32_t EmitFastF32ToF16RtzBits(EmitterState& state, uint32_t f32) {
	const auto bits   = EmitBitcastF32ToU32(state, f32);
	const auto abs    = EmitAndConstant(state, bits, 0x7fffffffu);
	const auto sign   = EmitAndConstant(state, EmitShiftRightConstant(state, bits, 16), 0x8000u);
	const auto high13 = EmitShiftRightConstant(state, abs, 13);
	const auto normal = EmitMinMaxU32Value(
	    state, Binary(state, spv::OpISub, TypeU32(state), high13, ConstantU32(state, 112u << 10u)),
	    ConstantU32(state, 0x7bffu), false);
	const auto exponent = EmitShiftRightConstant(state, abs, 23);
	const auto shift    = EmitMinMaxU32Value(state, EmitSubConstantMinusU32(state, 126, exponent),
	                                         ConstantU32(state, 31), false);
	const auto subnormal = Binary(
	    state, spv::OpShiftRightLogical, TypeU32(state),
	    EmitOrU32(state, EmitAndConstant(state, abs, 0x007fffffu), ConstantU32(state, 0x00800000u)),
	    shift);
	auto result =
	    EmitSelectValueU32(state, EmitCompareU32Constant(state, spv::OpULessThan, abs, 0x38800000u),
	                       subnormal, normal);
	const auto quiet = EmitSelectValueU32(
	    state, EmitCompareU32Constant(state, spv::OpUGreaterThan, abs, 0x7f800000u),
	    ConstantU32(state, 0x0200u), ConstantU32(state, 0u));
	const auto special = EmitOrU32(state, EmitAndConstant(state, high13, 0x7fffu), quiet);
	result = EmitSelectValueU32(
	    state, EmitCompareU32Constant(state, spv::OpUGreaterThanEqual, abs, 0x7f800000u), special,
	    result);
	return EmitOrU32(state, result, sign);
}

uint32_t EmitFastPackFloat2x16Rtz(EmitterState& state, uint32_t x, uint32_t y) {
	const auto low  = EmitFastF32ToF16RtzBits(state, x);
	const auto high = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
	                         EmitFastF32ToF16RtzBits(state, y), ConstantU32(state, 16));
	return EmitOrU32(state, low, high);
}

} // namespace

uint32_t EmitPackFloat2x16Rtz(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	if (GetCodegenOptions().fast_pkrtz) {
		return EmitFastPackFloat2x16Rtz(state, arg0, arg1);
	}
	const auto low  = EmitF32ToF16RtzBits(state, arg0);
	const auto high = Binary(state, spv::OpShiftLeftLogical, TypeU32(state),
	                         EmitF32ToF16RtzBits(state, arg1), ConstantU32(state, 16));
	return Binary(state, spv::OpBitwiseOr, TypeU32(state), low, high);
}

uint32_t EmitFPSaturate32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450FClamp,
	               {arg0, ConstantF32(state, 0), ConstantF32(state, 0x3f800000u)});
}

uint32_t EmitSMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, true);
}

uint32_t EmitUMulHi(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMulHigh(state, arg0, arg1, false);
}

uint32_t EmitIAbs32(EmitterState& state, uint32_t arg0) {
	const auto value = arg0;
	const auto neg   = Unary(state, spv::OpSNegate, TypeU32(state), value);
	const auto negative =
	    Binary(state, spv::OpSLessThan, TypeBool(state), value, ConstantU32(state, 0));
	return Select(state, TypeU32(state), negative, neg, value);
}

uint32_t EmitBitCount64(EmitterState& state, uint32_t arg0) {
	// Vulkan bit-count operands remain 32-bit, including when counting a U64.
	const auto unpacked = Unary(state, spv::OpBitcast, TypeU32Vector(state, 2), arg0);
	const auto pair =
	    ExtractPair(state, Unary(state, spv::OpBitCount, TypeU32Vector(state, 2), unpacked));
	return Binary(state, spv::OpIAdd, TypeU32(state), pair.low, pair.high);
}

uint32_t EmitFindILsb32(EmitterState& state, uint32_t arg0) {
	const auto value = EmitExt(state, TypeI32(state), GLSLstd450FindILsb, {arg0});
	return Unary(state, spv::OpBitcast, TypeU32(state), value);
}

uint32_t EmitFindUMsb32(EmitterState& state, uint32_t arg0) {
	const auto value = EmitExt(state, TypeI32(state), GLSLstd450FindUMsb, {arg0});
	return Unary(state, spv::OpBitcast, TypeU32(state), value);
}

uint32_t EmitSMin32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxI32Value(state, arg0, arg1, false);
}

uint32_t EmitSMax32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxI32Value(state, arg0, arg1, true);
}

uint32_t EmitUMin32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxU32Value(state, arg0, arg1, false);
}

uint32_t EmitUMax32(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxU32Value(state, arg0, arg1, true);
}

uint32_t EmitSMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMinMax3(state, arg0, arg1, arg2, true, false);
}

uint32_t EmitSMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMinMax3(state, arg0, arg1, arg2, true, true);
}

uint32_t EmitUMinTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMinMax3(state, arg0, arg1, arg2, false, false);
}

uint32_t EmitUMaxTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMinMax3(state, arg0, arg1, arg2, false, true);
}

uint32_t EmitSMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMed3(state, arg0, arg1, arg2, true);
}

uint32_t EmitUMedTri32(EmitterState& state, uint32_t arg0, uint32_t arg1, uint32_t arg2) {
	return EmitMed3(state, arg0, arg1, arg2, false);
}

namespace {

void CollectPositionSlice(EmitterState& state) {
	state.position_slice_ready = true;
	std::vector<const IR::Inst*> pending;
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() != IR::ValueOpcode::SetAttribute) {
				continue;
			}
			const auto index = inst.Flags<IR::ExportFlags>().index;
			if (index < state.program.export_info.size() &&
			    state.program.export_info[index].kind == IR::ExportTargetKind::Position) {
				pending.push_back(&inst);
			}
		}
	}
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		for (size_t index = 0; index < inst->NumArgs(); index++) {
			const auto* producer = inst->Arg(index).Resolve().TryInstruction();
			if (producer != nullptr && state.position_slice.insert(producer).second) {
				pending.push_back(producer);
			}
		}
	}
}

uint32_t EmitFloatBinary(ValueEmitContext& ctx, const IR::Inst& inst, spv::Op opcode) {
	auto&      state  = ctx.state;
	const auto result = Binary(state, opcode, TypeF32(state), ctx.Def(inst.Arg(0)),
	                           ctx.Def(inst.Arg(1)));
	if (NoContraction(state, inst)) {
		state.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
	}
	return result;
}

} // namespace

// MadMode: guest float arithmetic is never contracted by the guest's hardware (it contracts only
// where the program uses FMA/FMAC explicitly), so a host FMA contraction changes bits. Exact
// forbids it everywhere; Position only where the value reaches a position export, so that two
// shaders computing the same position (depth pre-pass and main pass) cannot be compiled to
// different roundings.
bool NoContraction(EmitterState& state, const IR::Inst& inst) {
	switch (GetCodegenOptions().mad_mode) {
		case MadMode::Exact: return true;
		case MadMode::Fused: return false;
		case MadMode::Position:
			if (!state.position_slice_ready) {
				CollectPositionSlice(state);
			}
			return state.position_slice.contains(&inst);
	}
	return false;
}

uint32_t EmitFPAdd32(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitFloatBinary(ctx, inst, spv::OpFAdd);
}

uint32_t EmitFPSub32(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitFloatBinary(ctx, inst, spv::OpFSub);
}

uint32_t EmitFPMul32(ValueEmitContext& ctx, const IR::Inst& inst) {
	return EmitFloatBinary(ctx, inst, spv::OpFMul);
}

// V_MAD_F32/V_MAC_F32/V_MADMK_F32/V_MADAK_F32. PS5 executes them unfused (as GCN/RDNA1 did): the
// product is rounded to f32, then added, so v_mad_f32 equals v_mul_f32 + v_add_f32 bit for bit.
// An FMul and an FAdd, both NoContraction, reproduce that exactly; MadMode::Fused (and
// MadMode::Position off the position data flow) keep the cheaper fused FMA of the old code.
uint32_t EmitFPMad32(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto a     = ctx.Def(inst.Arg(0));
	const auto b     = ctx.Def(inst.Arg(1));
	const auto c     = ctx.Def(inst.Arg(2));
	if (!NoContraction(state, inst)) {
		return EmitExt(state, TypeF32(state), GLSLstd450Fma, {a, b, c});
	}
	const auto product = Binary(state, spv::OpFMul, TypeF32(state), a, b);
	state.builder.AddAnnotation(spv::OpDecorate, product, spv::DecorationNoContraction);
	const auto sum = Binary(state, spv::OpFAdd, TypeF32(state), product, c);
	state.builder.AddAnnotation(spv::OpDecorate, sum, spv::DecorationNoContraction);
	return sum;
}

uint32_t EmitFPIsNan32(EmitterState& state, uint32_t arg0) {
	return EmitNative<spv::OpFUnordNotEqual, IR::Type::U1>(state, arg0, arg0);
}

uint32_t EmitFPMin32(ValueEmitContext& ctx, IR::Value arg0, IR::Value arg1) {
	if (FastFloatMinMax()) {
		return EmitFastMinMaxF32(ctx.state, OperandF32(ctx, arg0), OperandF32(ctx, arg1), false)
		    .id;
	}
	return EmitMinMaxF32Value(ctx.state, ctx.Def(arg0), ctx.Def(arg1), false);
}

uint32_t EmitFPMax32(ValueEmitContext& ctx, IR::Value arg0, IR::Value arg1) {
	if (FastFloatMinMax()) {
		return EmitFastMinMaxF32(ctx.state, OperandF32(ctx, arg0), OperandF32(ctx, arg1), true)
		    .id;
	}
	return EmitMinMaxF32Value(ctx.state, ctx.Def(arg0), ctx.Def(arg1), true);
}

uint32_t EmitFPMin64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, false);
}

uint32_t EmitFPMax64(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	return EmitMinMaxF64(state, arg0, arg1, true);
}

uint32_t EmitFPMinTri32(ValueEmitContext& ctx, IR::Value arg0, IR::Value arg1, IR::Value arg2) {
	if (FastFloatMinMax()) {
		return EmitFastMinMax3F32(ctx.state, OperandF32(ctx, arg0), OperandF32(ctx, arg1),
		                            OperandF32(ctx, arg2), false)
		    .id;
	}
	return EmitFMinMax3(ctx.state, ctx.Def(arg0), ctx.Def(arg1), ctx.Def(arg2), false);
}

uint32_t EmitFPMaxTri32(ValueEmitContext& ctx, IR::Value arg0, IR::Value arg1, IR::Value arg2) {
	if (FastFloatMinMax()) {
		return EmitFastMinMax3F32(ctx.state, OperandF32(ctx, arg0), OperandF32(ctx, arg1),
		                            OperandF32(ctx, arg2), true)
		    .id;
	}
	return EmitFMinMax3(ctx.state, ctx.Def(arg0), ctx.Def(arg1), ctx.Def(arg2), true);
}

// FLOAT_MODE 0xC0 flushes f32 denormal inputs to zeros of the same sign (so rcp(-denorm) is
// -inf). KYTY_HOST_FTZ_INPUTS=1 leaves that to a module declaring DenormFlushToZero 32
// (CodegenTranscendentalDenormInputs checks the host keeps the sign).
static uint32_t FlushTranscendentalInput(EmitterState& state, uint32_t value) {
	if (GetCodegenOptions().host_ftz_inputs && GetHostFloatControls().denorm_flush_f32) {
		return value;
	}
	return EmitFlushF32DenormToSignedZero(state, value);
}

uint32_t EmitFPRecip32(EmitterState& state, uint32_t arg0) {
	const auto source = FlushTranscendentalInput(state, arg0);
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), source);
}

uint32_t EmitFPRecip64(EmitterState& state, uint32_t arg0) {
	const auto one = state.builder.Constant(spv::OpConstant, TypeF64(state), 0u, 0x3ff00000u);
	return EmitNative<spv::OpFDiv, IR::Type::F64>(state, one, arg0);
}

uint32_t EmitFPRecipIFlag32(EmitterState& state, uint32_t arg0) {
	// Integer-to-float inputs used by IFLAG cannot be denormal.
	return Binary(state, spv::OpFDiv, TypeF32(state), ConstantF32(state, 0x3f800000u), arg0);
}

uint32_t EmitFPRecipSqrt32(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450InverseSqrt,
	               {FlushTranscendentalInput(state, arg0)});
}

uint32_t EmitFPSqrt(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Sqrt, {FlushTranscendentalInput(state, arg0)});
}

uint32_t EmitFPExp2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Exp2, {FlushTranscendentalInput(state, arg0)});
}

uint32_t EmitFPLog2(EmitterState& state, uint32_t arg0) {
	return EmitExt(state, TypeF32(state), GLSLstd450Log2, {FlushTranscendentalInput(state, arg0)});
}

uint32_t EmitFPLdexp(EmitterState& state, uint32_t arg0, uint32_t arg1) {
	const auto exponent = Unary(state, spv::OpBitcast, TypeI32(state), arg1);
	return EmitExt(state, TypeF32(state), GLSLstd450Ldexp, {arg0, exponent});
}

uint32_t EmitFPSin(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, true);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Sin, {source});
}

uint32_t EmitFPCos(EmitterState& state, uint32_t arg0) {
	auto source = EmitTrigCycleF32(state, arg0, false);
	source = Binary(state, spv::OpFMul, TypeF32(state), source, ConstantF32(state, 0x40c90fdbu));
	return EmitExt(state, TypeF32(state), GLSLstd450Cos, {source});
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
