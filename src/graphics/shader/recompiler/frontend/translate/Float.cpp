#include "graphics/shader/recompiler/frontend/translate/Translator.h"
#include "graphics/shader/recompiler/CodegenOptions.h"

#include <array>
#include <bit>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::Frontend {

void Translator::PackedFloat16(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool accumulator, bool quiet_snan) {
	const auto translate_lane = [&](bool high) {
		const auto lhs = ReadF16LaneAsF32(inst.src0, high);
		const auto rhs = ReadF16LaneAsF32(inst.src1, high);
		IR::F32    result;
		if (accumulator) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.dst, high)}));
		} else if (inst.src_count == 3u) {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs, ReadF16LaneAsF32(inst.src2, high)}));
		} else {
			result = IR::F32(ir.Emit(opcode, {lhs, rhs}));
		}
		return ApplyF32ResultModifiers(inst.dst, result);
	};
	auto raw    = DestinationOperand(inst);
	raw.omod    = 0u;
	raw.clamp   = false;
	// KYTY_NATIVE_F16: add/mul/min/max of two sources without result modifiers as one f16vec2
	// operation on the packed source halves (selection, abs and neg already applied per lane).
	IR::ValueOpcode native = IR::ValueOpcode::Void;
	if (!accumulator && inst.src_count == 2u && !inst.dst.clamp && inst.dst.omod == 0u &&
	    NativeHalfActive()) {
		switch (opcode) {
			case IR::ValueOpcode::FPAdd32: native = IR::ValueOpcode::HalfPkAdd; break;
			case IR::ValueOpcode::FPMul32: native = IR::ValueOpcode::HalfPkMul; break;
			case IR::ValueOpcode::FPMin32: native = IR::ValueOpcode::HalfPkMin; break;
			case IR::ValueOpcode::FPMax32: native = IR::ValueOpcode::HalfPkMax; break;
			default: break;
		}
	}
	// V_PK_FMA_F16 / V_PK_FMAC_F16 (the accumulator is the destination's halves) as one f16vec2 Fma.
	if (opcode == IR::ValueOpcode::FPFma32 && (accumulator || inst.src_count == 3u) &&
	    !inst.dst.clamp && inst.dst.omod == 0u && NativeHalfActive()) {
		native = IR::ValueOpcode::HalfPkFma;
	}
	const auto packed = [&](const Decoder::Operand& operand) {
		return PackU16Lanes(Read16LaneBits(operand, false), Read16LaneBits(operand, true));
	};
	IR::U32 result;
	if (native == IR::ValueOpcode::HalfPkFma) {
		result = IR::U32(ir.Emit(native, {packed(inst.src0), packed(inst.src1),
		                                  packed(accumulator ? inst.dst : inst.src2)}));
	} else if (native != IR::ValueOpcode::Void) {
		result = IR::U32(ir.Emit(native, {packed(inst.src0), packed(inst.src1)}));
	} else {
		result = PackHalf2x16(translate_lane(false), translate_lane(true));
	}
	if (quiet_snan) {
		const auto quiet_snan_lane = [&](const Decoder::Operand& operand, bool high) {
			const auto bits     = ReadU16LaneAsU32(operand, high, false);
			const auto exponent = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x7e00u)));
			const auto payload  = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x01ffu)));
			const auto snan     = ir.LogicalAnd(ir.IEqual(exponent, IR::U32(IR::Value(0x7c00u))),
			                                    ir.INotEqual(payload, IR::U32(IR::Value(0u))));
			return std::pair {snan, ir.BitwiseOr(bits, IR::U32(IR::Value(0x0200u)))};
		};
		const auto override_lane = [&](bool high) {
			const auto [lhs_snan, lhs_quiet] = quiet_snan_lane(inst.src0, high);
			const auto [rhs_snan, rhs_quiet] = quiet_snan_lane(inst.src1, high);
			const auto normal = high ? ir.ShiftRightLogical(result, IR::U32(IR::Value(16u)))
			                         : ir.BitwiseAnd(result, IR::U32(IR::Value(0xffffu)));
			return ir.Select(lhs_snan, lhs_quiet, ir.Select(rhs_snan, rhs_quiet, normal));
		};
		result = PackU16Lanes(override_lane(false), override_lane(true));
	}
	WriteOperand(raw, result);
}

void Translator::Float16Unary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool invalid_negative) {
	const auto argument = ReadF16AsF32(inst.src0);
	auto       result   = IR::F32(ir.Emit(opcode, {argument}));
	if (!invalid_negative) {
		WriteF16(DestinationOperand(inst), result);
		return;
	}
	const auto negative =
	    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {argument, IR::Value::F32(0.0f)}));
	result             = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(negative, invalid, bits));
}

void Translator::Float16Trig(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto argument = ReadF16AsF32(inst.src0);
	const auto magnitude =
	    ir.BitwiseAnd(ir.BitCastU32(argument), IR::U32(IR::Value(0x7fffffffu)));
	auto result = IR::F32(ir.Emit(opcode, {argument}));
	if (opcode == IR::ValueOpcode::FPSin) {
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto whole = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.0f)}));
		const auto half = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.5f)}));
		const auto nonzero = ir.INotEqual(magnitude, IR::U32(IR::Value(0u)));
		const auto cardinal = ir.LogicalOr(ir.LogicalAnd(whole, nonzero), half);
		result = IR::F32(ir.Emit(
		    IR::ValueOpcode::SelectF32, {cardinal, IR::Value::F32(0.0f), result}));
	} else {
		// The architectural quarter-cycle result is exactly zero. Evaluating cos(pi/2) with a
		// rounded F32 PI can otherwise produce a half-precision subnormal instead.
		const auto fraction = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {argument}));
		const auto quarter = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.25f)}));
		const auto three_quarters = IR::U1(
		    ir.Emit(IR::ValueOpcode::FPOrdEqual32, {fraction, IR::Value::F32(0.75f)}));
		result = IR::F32(
		    ir.Emit(IR::ValueOpcode::SelectF32,
		            {ir.LogicalOr(quarter, three_quarters), IR::Value::F32(0.0f), result}));
	}

	const auto infinite = ir.IEqual(magnitude, IR::U32(IR::Value(0x7f800000u)));
	result              = ApplyF32ResultModifiers(inst.dst, result);
	const auto bits    = PackHalf2x16(result, IR::F32(IR::Value::F32(0.0f)));
	const auto invalid = IR::U32(IR::Value(inst.dst.clamp ? 0u : 0xfe00u));
	Write16Bits(DestinationOperand(inst), ir.Select(infinite, invalid, bits));
}

void Translator::Float16Binary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                               bool reverse) {
	const auto lhs = ReadF16AsF32(reverse ? inst.src1 : inst.src0);
	const auto rhs = ReadF16AsF32(reverse ? inst.src0 : inst.src1);
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {lhs, rhs})));
}

void Translator::Float16Ternary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                                bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		args[index] = mix ? IR::Value(ReadMixF32(operand)) : IR::Value(ReadF16AsF32(operand));
	}
	if (opcode == IR::ValueOpcode::FPFma32 && GetCodegenOptions().f16_exact_round &&
	    float64_allowed) {
		// KYTY_F16_EXACT_ROUND: V_FMA_F16/V_FMAC_F16/V_FMA_MIX*_F16 round once to f16 (ISA p.165).
		// The product of two f32s is exact in f64, the f64 Fma rounds once to 53 bits, and a
		// round-to-odd step to f32 keeps that rounding from deciding the f16 tie (24 >= 11 + 2), so
		// the final round to nearest even is the single rounding; it is off only when the addend is so
		// small that the 53-bit rounding itself lands on a 24-bit value.
		std::array<IR::Value, 3> wide;
		for (uint32_t index = 0; index < wide.size(); index++) {
			wide[index] = ir.Emit(IR::ValueOpcode::ConvertF64F32, {args[index]});
		}
		const auto exact  = ir.Emit(IR::ValueOpcode::FPFma64, {wide[0], wide[1], wide[2]});
		const auto narrow = IR::F32(ir.Emit(IR::ValueOpcode::ConvertF32F64, {exact}));
		const auto back   = ir.Emit(IR::ValueOpcode::ConvertF64F32, {narrow});
		const auto inexact =
		    IR::U1(ir.Emit(IR::ValueOpcode::FPOrdNotEqual64, {back, exact}));
		const auto above = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThan64, {exact, back}));
		const auto bits  = ir.BitCastU32(narrow);
		const auto negative =
		    ir.INotEqual(ir.ShiftRightLogical(bits, IR::U32(IR::Value(31u))), IR::U32(IR::Value(0u)));
		const auto even =
		    ir.IEqual(ir.BitwiseAnd(bits, IR::U32(IR::Value(1u))), IR::U32(IR::Value(0u)));
		// One ulp away from zero when the exact value has the larger magnitude, else toward it.
		const auto grow = IR::U1(ir.Emit(IR::ValueOpcode::SelectU1, {negative, ir.LogicalNot(above), above}));
		const auto odd  = ir.Select(grow, ir.IAdd(bits, IR::U32(IR::Value(1u))),
		                            ir.ISub(bits, IR::U32(IR::Value(1u))));
		const auto rounded = ir.Select(ir.LogicalAnd(inexact, even), odd, bits);
		WriteF16(DestinationOperand(inst), ir.BitCastF32(rounded));
		return;
	}
	WriteF16(DestinationOperand(inst), IR::F32(ir.Emit(opcode, {args[0], args[1], args[2]})));
}

void Translator::FloatUnary(const Decoder::Instruction& inst, IR::ValueOpcode opcode) {
	const auto type = IR::ArgTypeOf(opcode, 0);
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {ReadOperand(inst.src0, type)}));
}

void Translator::V_FRACT(const Decoder::Instruction& inst, IR::Type type) {
	// x - floor(x) rounds to 1.0 for a tiny negative x; v_fract returns the largest value below
	// one instead. The ordered compare lets NaN through. KYTY_FRACT_CLAMP=1 only (default off:
	// the plain fract as before).
	const auto destination = DestinationOperand(inst);
	if (!GetCodegenOptions().fract_clamp) {
		if (type == IR::Type::F16) {
			return Float16Unary(inst, IR::ValueOpcode::FPFract32, false);
		}
		return FloatUnary(inst, type == IR::Type::F64 ? IR::ValueOpcode::FPFract64
		                                              : IR::ValueOpcode::FPFract32);
	}
	if (type == IR::Type::F64) {
		const auto result    = ir.Emit(IR::ValueOpcode::FPFract64, {ReadOperand(inst.src0, type)});
		const auto below_one = ir.Emit(IR::ValueOpcode::BitCastF64U64,
		                               {IR::Value(uint64_t {0x3fefffffffffffff})});
		const auto above     = ir.Emit(IR::ValueOpcode::FPOrdGreaterThan64, {result, below_one});
		WriteOperand(destination, ir.Emit(IR::ValueOpcode::SelectF64, {above, below_one, result}));
		return;
	}
	const bool half   = type == IR::Type::F16;
	const auto source = half ? ReadF16AsF32(inst.src0) : IR::F32(ReadOperand(inst.src0, type));
	const auto result = IR::F32(ir.Emit(IR::ValueOpcode::FPFract32, {source}));
	// Clamp in the destination precision: 0x3f7fe000 is the half 0x3bff.
	const auto below_one =
	    IR::F32(IR::Value::F32(std::bit_cast<float>(half ? 0x3f7fe000u : 0x3f7fffffu)));
	const auto above   = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThan32, {result, below_one}));
	const auto clamped = SelectF32(above, below_one, result);
	if (half) {
		WriteF16(destination, clamped);
	} else {
		WriteOperand(destination, clamped);
	}
}

void Translator::FloatBinary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                             bool reverse) {
	std::array<IR::Value, 2> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = SourceAt(inst, reverse ? 1u - index : index);
		args[index]        = ReadOperand(operand, IR::ArgTypeOf(opcode, index));
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1]}));
}

void Translator::FloatTernary(const Decoder::Instruction& inst, IR::ValueOpcode opcode,
                              bool accumulator, bool mix) {
	std::array<IR::Value, 3> args;
	for (uint32_t index = 0; index < args.size(); index++) {
		const auto& operand = accumulator && index == 2u ? inst.dst : SourceAt(inst, index);
		const auto type    = IR::ArgTypeOf(opcode, index);
		args[index]        = type == IR::Type::F32 && mix ? IR::Value(ReadMixF32(operand))
		                                                  : ReadOperand(operand, type);
	}
	WriteOperand(DestinationOperand(inst), ir.Emit(opcode, {args[0], args[1], args[2]}));
}

void Translator::V_FREXP_MANT_F32(const Decoder::Instruction& inst) {
	const auto bits = ReadU32(inst.src0);
	const auto exponent =
	    IR::U32(ir.Emit(IR::ValueOpcode::BitFieldUExtract, {bits, IR::Value(23u), IR::Value(8u)}));
	const auto mantissa = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x007fffffu)));
	const auto sign     = ir.BitwiseAnd(bits, IR::U32(IR::Value(0x80000000u)));
	const auto base     = ir.BitwiseOr(sign, IR::U32(IR::Value(0x3f000000u)));
	const auto normal   = ir.BitwiseOr(base, mantissa);
	const auto msb      = IR::U32(ir.Emit(IR::ValueOpcode::FindUMsb32, {mantissa}));
	const auto shift    = ir.ISub(IR::U32(IR::Value(23u)), msb);
	const auto fraction =
	    ir.BitwiseAnd(ir.ShiftLeftLogical(mantissa, shift), IR::U32(IR::Value(0x007fffffu)));
	const auto subnormal = ir.BitwiseOr(base, fraction);
	const auto zero      = ir.IEqual(mantissa, IR::U32(IR::Value(0u)));
	const auto finite    = ir.Select(ir.INotEqual(exponent, IR::U32(IR::Value(0u))), normal,
	                                 ir.Select(zero, bits, subnormal));
	const auto result    = ir.Select(ir.IEqual(exponent, IR::U32(IR::Value(0xffu))), bits, finite);
	WriteOperand(DestinationOperand(inst), ir.BitCastF32(result));
}

void Translator::V_DOT2C_F32_F16(const Decoder::Instruction& inst) {
	auto a          = inst.src0;
	a.op_sel        = false;
	a.op_sel_hi     = true;
	auto b          = inst.src1;
	b.op_sel        = false;
	b.op_sel_hi     = true;
	const auto a_lo = ReadF16LaneAsF32(a, false);
	const auto a_hi = ReadF16LaneAsF32(a, true);
	const auto b_lo = ReadF16LaneAsF32(b, false);
	const auto b_hi = ReadF16LaneAsF32(b, true);
	const auto acc  = ReadMixF32(inst.dst);
	const auto lo   = IR::F32(ir.Emit(IR::ValueOpcode::FPFma32, {a_lo, b_lo, acc}));
	WriteOperand(DestinationOperand(inst), ir.Emit(IR::ValueOpcode::FPFma32, {a_hi, b_hi, lo}));
}

void Translator::V_CUBEID_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 0u);
}

void Translator::V_CUBESC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 1u);
}

void Translator::V_CUBETC_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 2u);
}

void Translator::V_CUBEMA_F32(const Decoder::Instruction& inst) {
	return FloatCube(inst, 3u);
}

void Translator::FloatCube(const Decoder::Instruction& inst, uint32_t result_kind) {
	const auto x  = ReadMixF32(inst.src0);
	const auto y  = ReadMixF32(inst.src1);
	const auto z  = ReadMixF32(inst.src2);
	const auto nx = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {x}));
	const auto ny = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {y}));
	const auto nz = IR::F32(ir.Emit(IR::ValueOpcode::FPNeg32, {z}));
	const auto ax = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {x}));
	const auto ay = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {y}));
	const auto az = IR::F32(ir.Emit(IR::ValueOpcode::FPAbs32, {z}));
	const auto z_face =
	    ir.LogicalAnd(IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ax})),
	                  IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {az, ay})));
	const auto y_face = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdGreaterThanEqual32, {ay, ax}));
	const auto x_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {x, IR::Value::F32(0.0f)}));
	const auto y_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {y, IR::Value::F32(0.0f)}));
	const auto z_neg = IR::U1(ir.Emit(IR::ValueOpcode::FPOrdLessThan32, {z, IR::Value::F32(0.0f)}));
	const auto select_face = [&](IR::F32 x_value, IR::F32 y_value, IR::F32 z_value) {
		return SelectF32(z_face, z_value, SelectF32(y_face, y_value, x_value));
	};
	IR::F32 result;
	switch (result_kind) {
		case 0u:
			result = select_face(
			    SelectF32(x_neg, IR::F32(IR::Value::F32(1.0f)), IR::F32(IR::Value::F32(0.0f))),
			    SelectF32(y_neg, IR::F32(IR::Value::F32(3.0f)), IR::F32(IR::Value::F32(2.0f))),
			    SelectF32(z_neg, IR::F32(IR::Value::F32(5.0f)), IR::F32(IR::Value::F32(4.0f))));
			break;
		case 1u: result = select_face(SelectF32(x_neg, z, nz), x, SelectF32(z_neg, nx, x)); break;
		case 2u: result = select_face(ny, SelectF32(y_neg, nz, z), ny); break;
		case 3u:
			result = IR::F32(
			    ir.Emit(IR::ValueOpcode::FPMul32, {select_face(x, y, z), IR::Value::F32(2.0f)}));
			break;
		default: EXIT("invalid cube result kind");
	}
	WriteOperand(DestinationOperand(inst), result);
}

} // namespace Libs::Graphics::ShaderRecompiler::Frontend
