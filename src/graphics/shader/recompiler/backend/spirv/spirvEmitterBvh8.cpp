#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayTracing.h"

#include <algorithm>
#include <array>
#include <span>
#include <utility>
#include <vector>

// IMAGE_BVH8_INTERSECT_RAY (PS5 Pro / Trinity; IR Bvh8IntersectRay), ported from upstream 5eab0ec73
// (spirvEmitterTrinityBvh.cpp and the triangle test it shared with upstream's BVH4 emitter). Upstream
// called it through its BvhIntersect/ImageAddress design; here the IR value carries the ray as plain
// operands like our BVH4 BvhIntersectRay, and node memory is read through our BDA page lookup
// (EmitBdaDeviceAddress; an unmapped page reads as absent). The node formats, flags, instance
// transform and sorting network are upstream's unchanged.
//
// Node pointer types: 0/1 a triangle pair (slot 0, or both slots), 2/5 a full 128-byte box pair
// (two 4-child boxes, f32 or compressed), 3/4 a 64-byte box pair (f16 or compressed), 6 an instance.
// Result: dwords 0-7 two groups of four (box: child pointers; triangle: t_num, t_denom, I/ID, J/hit),
// 8-9 the triangles' shape IDs, 10-15 the ray (transformed by an instance), 16 the instance flag.

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

uint32_t TypeBvh8Result(EmitterState& s) {
	if (s.bvh8_result_type == 0) {
		const auto v       = TypeU32Vector(s, 4);
		s.bvh8_result_type = s.builder.Type(spv::OpTypeStruct, v, v, v, v, TypeU32(s));
	}
	return s.bvh8_result_type;
}

namespace {

using Vec3 = std::array<uint32_t, 3>;

// Upstream's watertight triangle test (EmitBvhTriangle): the ray's largest direction axis becomes
// z, the vertices are sheared into ray space and the three edge functions decide the hit. Returns
// {t_num or +inf, t_denom or 1, bary ? I : triangle_id, bary ? J : hit}.
uint32_t EmitBvh8Triangle(EmitterState& s, const std::array<Vec3, 3>& positions, Vec3 origin,
                          Vec3 direction, uint32_t remap, uint32_t triangle_id, uint32_t bary) {
	const auto f    = TypeF32(s);
	const auto u    = TypeU32(s);
	const auto b    = TypeBool(s);
	const auto zero = ConstantF32Value(s, 0.0f);
	const auto op   = [&](spv::Op code, uint32_t x, uint32_t y) {
        const auto value = Binary(s, code, f, x, y);
        s.builder.AddAnnotation(spv::OpDecorate, value, spv::DecorationNoContraction);
        return value;
	};
	const auto cmp    = [&](spv::Op code, uint32_t x, uint32_t y) { return Binary(s, code, b, x, y); };
	const auto either = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalOr, b, x, y); };
	const auto both   = [&](uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalAnd, b, x, y); };
	const auto abs    = [&](uint32_t x) { return EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(s, x); };
	const auto y_largest = cmp(spv::OpFOrdLessThan, abs(direction[0]), abs(direction[1]));
	const auto z_largest = cmp(spv::OpFOrdLessThan,
	                           Select(s, f, y_largest, abs(direction[1]), abs(direction[0])),
	                           abs(direction[2]));
	const auto rotate    = [&](Vec3 value) {
        Vec3 result;
        for (uint32_t i = 0; i < 3; ++i) {
            result[i] = Select(s, f, z_largest, value[i],
                               Select(s, f, y_largest, value[(i + 2) % 3], value[(i + 1) % 3]));
        }
        return result;
	};
	origin    = rotate(origin);
	direction = rotate(direction);
	std::array<Vec3, 3> projected;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto position = rotate(positions[i]);
		Vec3       relative;
		for (uint32_t axis = 0; axis < 3; ++axis) {
			relative[axis] = op(spv::OpFSub, position[axis], origin[axis]);
		}
		projected[i] = {op(spv::OpFSub, op(spv::OpFMul, relative[0], direction[2]),
		                   op(spv::OpFMul, direction[0], relative[2])),
		                op(spv::OpFSub, op(spv::OpFMul, relative[1], direction[2]),
		                   op(spv::OpFMul, direction[1], relative[2])),
		                relative[2]};
	}
	Vec3 edge;
	Vec3 weighted;
	for (uint32_t i = 0; i < 3; ++i) {
		const auto& first  = projected[(i + 1) % 3];
		const auto& second = projected[(i + 2) % 3];
		edge[i]     = op(spv::OpFSub, op(spv::OpFMul, second[0], first[1]),
		                 op(spv::OpFMul, second[1], first[0]));
		weighted[i] = op(spv::OpFMul, edge[i], direction[2]);
	}
	const auto numerator =
	    op(spv::OpFAdd,
	       op(spv::OpFAdd, op(spv::OpFMul, edge[0], projected[0][2]),
	          op(spv::OpFMul, edge[1], projected[1][2])),
	       op(spv::OpFMul, edge[2], projected[2][2]));
	const auto denominator = op(spv::OpFAdd, op(spv::OpFAdd, weighted[0], weighted[1]), weighted[2]);
	uint32_t   negative    = ConstantBool(s, false);
	uint32_t   positive    = negative;
	for (const auto value: edge) {
		negative = either(negative, cmp(spv::OpFOrdLessThan, value, zero));
		positive = either(positive, cmp(spv::OpFOrdGreaterThan, value, zero));
	}
	const auto winding = cmp(spv::OpFOrdGreaterThan, denominator, zero);
	auto missed = either(both(negative, positive), cmp(spv::OpFOrdEqual, denominator, zero));
	missed      = either(missed, Unary(s, spv::OpIsNan, b, numerator));
	missed      = either(missed, cmp(spv::OpFOrdLessThan,
	                                 Select(s, f, winding, numerator, Unary(s, spv::OpFNegate, f, numerator)),
	                                 zero));
	for (uint32_t i = 0; i < 3; ++i) {
		const auto first      = projected[(i + 1) % 3][1];
		const auto second     = projected[(i + 2) % 3][1];
		const auto first_zero = cmp(spv::OpFOrdEqual, first, zero);
		const auto horizontal = both(first_zero, cmp(spv::OpFOrdEqual, second, zero));
		const auto below      = either(cmp(spv::OpFOrdLessThan, first, zero),
		                               both(first_zero, cmp(spv::OpFOrdGreaterThan, second, zero)));
		const auto right      = Binary(s, spv::OpLogicalNotEqual, b, below, winding);
		const auto excluded =
		    Select(s, b, horizontal, cmp(spv::OpFOrdGreaterThan, projected[i][1], zero), right);
		missed = either(missed, both(cmp(spv::OpFOrdEqual, edge[i], zero), excluded));
	}
	const auto coordinate = [&](uint32_t shift) {
		const auto index =
		    Binary(s, spv::OpBitwiseAnd, u,
		           Binary(s, spv::OpShiftRightLogical, u, remap, ConstantU32(s, shift)),
		           ConstantU32(s, 3));
		return Unary(s, spv::OpBitcast, u,
		             Select(s, f, cmp(spv::OpIEqual, index, ConstantU32(s, 1)), weighted[1],
		                    Select(s, f, cmp(spv::OpIEqual, index, ConstantU32(s, 2)), weighted[2],
		                           weighted[0])));
	};
	const std::array<uint32_t, 4> result {
	    Unary(s, spv::OpBitcast, u, Select(s, f, missed, ConstantF32(s, 0x7f800000), numerator)),
	    Unary(s, spv::OpBitcast, u,
	          Select(s, f, missed, ConstantF32Value(s, 1.0f), denominator)),
	    Select(s, u, bary, coordinate(0), triangle_id),
	    Select(s, u, bary, coordinate(2),
	           Select(s, u, missed, ConstantU32(s, 0), ConstantU32(s, 1)))};
	const auto vector = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(s, 4), vector,
	                      std::span<const uint32_t>(result));
	return vector;
}

struct Bvh8Emitter {
	EmitterState&     s;
	ValueEmitContext* ctx = nullptr;
	const uint32_t    u   = TypeU32(s);
	const uint32_t    f   = TypeF32(s);
	const uint32_t    b   = TypeBool(s);
	const uint32_t    w   = TypeU64(s);
	const uint32_t    v4  = TypeU32Vector(s, 4);
	const uint32_t    v3  = TypeF32Vector(s, 3);
	uint32_t          triangle_function   = 0;
	uint32_t          boxes_function      = 0;
	uint32_t          decompress_function = 0;

	uint32_t C(uint32_t x) { return ConstantU32(s, x); }
	uint32_t And(uint32_t x, uint32_t y) { return Binary(s, spv::OpBitwiseAnd, u, x, y); }
	uint32_t Or(uint32_t x, uint32_t y) { return Binary(s, spv::OpBitwiseOr, u, x, y); }
	uint32_t Add(uint32_t x, uint32_t y) { return EmitAddU32(s, x, y); }
	uint32_t Sub(uint32_t x, uint32_t y) { return Binary(s, spv::OpISub, u, x, y); }
	uint32_t Shl(uint32_t x, uint32_t n) { return Binary(s, spv::OpShiftLeftLogical, u, x, C(n)); }
	uint32_t Shr(uint32_t x, uint32_t n) { return Binary(s, spv::OpShiftRightLogical, u, x, C(n)); }
	uint32_t Eq(uint32_t x, uint32_t y) { return Binary(s, spv::OpIEqual, b, x, y); }
	uint32_t Both(uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalAnd, b, x, y); }
	uint32_t Either(uint32_t x, uint32_t y) { return Binary(s, spv::OpLogicalOr, b, x, y); }
	uint32_t Not(uint32_t x) { return Unary(s, spv::OpLogicalNot, b, x); }
	uint32_t Bit(uint32_t x, uint32_t n) { return Not(Eq(And(x, C(1u << n)), C(0))); }
	uint32_t Choose(uint32_t condition, uint32_t x, uint32_t y) { return Select(s, u, condition, x, y); }
	uint32_t Float(uint32_t x) { return Unary(s, spv::OpBitcast, f, x); }
	uint32_t Raw(uint32_t x) { return Unary(s, spv::OpBitcast, u, x); }
	uint32_t Fop(spv::Op op, uint32_t x, uint32_t y) { return Binary(s, op, f, x, y); }
	uint32_t Fcmp(spv::Op op, uint32_t x, uint32_t y) { return Binary(s, op, b, x, y); }
	uint32_t Min(uint32_t x, uint32_t y) { return EmitGlsl<GLSLstd450FMin, IR::Type::F32>(s, x, y); }
	uint32_t Max(uint32_t x, uint32_t y) { return EmitGlsl<GLSLstd450FMax, IR::Type::F32>(s, x, y); }
	uint32_t UMin(uint32_t x, uint32_t y) { return EmitGlsl<GLSLstd450UMin, IR::Type::U32>(s, x, y); }
	uint32_t Extract(uint32_t type, uint32_t value, uint32_t index) {
		const auto id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, type, id, value, index);
		return id;
	}
	uint32_t Make(uint32_t type, std::span<const uint32_t> values) {
		const auto id = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, type, id, values);
		return id;
	}
	uint32_t Make4(uint32_t x, uint32_t y, uint32_t z, uint32_t t) {
		return Make(v4, std::array {x, y, z, t});
	}
	Vec3 Unpack3(uint32_t value) {
		return {Extract(f, value, 0), Extract(f, value, 1), Extract(f, value, 2)};
	}
	// One dword at a device address (a node block from Fetch) plus a dword index.
	uint32_t Word(uint32_t block, uint32_t index) {
		const auto offset  = Unary(s, spv::OpUConvert, w, Shl(index, 2));
		const auto address = Binary(s, spv::OpIAdd, w, block, offset);
		const auto pointer = Unary(s, spv::OpConvertUToPtr, TypePhysicalU32Pointer(s), address);
		const auto id      = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpLoad, u, id, pointer, spv::MemoryAccessAlignedMask, 4u);
		return id;
	}
	uint32_t Call(uint32_t type, uint32_t function, std::span<const uint32_t> args) {
		const auto            id = s.builder.AllocateId();
		std::vector<uint32_t> call {spv::OpFunctionCall, type, id, function};
		call.insert(call.end(), args.begin(), args.end());
		s.builder.AddFunction(std::span<const uint32_t>(call));
		return id;
	}
	std::vector<uint32_t> Begin(uint32_t& function, const char* name, uint32_t result,
	                            std::span<const uint32_t> types) {
		function = s.builder.AllocateId();
		s.builder.AddName(function, name);
		s.builder.AddFunction(spv::OpFunction, result, function, spv::FunctionControlMaskNone,
		                      s.builder.Type(spv::OpTypeFunction, result, types));
		std::vector<uint32_t> args;
		for (const auto type: types) {
			args.push_back(s.builder.AllocateId());
			s.builder.AddFunction(spv::OpFunctionParameter, type, args.back());
		}
		EmitLabel(s, s.builder.AllocateId());
		return args;
	}
	void End(uint32_t result) {
		s.builder.AddFunction(spv::OpReturnValue, result);
		s.builder.AddFunction(spv::OpFunctionEnd);
	}
	uint32_t Result(const std::array<uint32_t, 10>& dst, const Vec3& org, const Vec3& dir,
	                uint32_t instance) {
		const std::array groups {Make4(dst[0], dst[1], dst[2], dst[3]),
		                         Make4(dst[4], dst[5], dst[6], dst[7]),
		                         Make4(dst[8], dst[9], Raw(org[0]), Raw(org[1])),
		                         Make4(Raw(org[2]), Raw(dir[0]), Raw(dir[1]), Raw(dir[2])), instance};
		return Make(TypeBvh8Result(s), groups);
	}
	uint32_t NormalizeFlags(uint32_t flags) {
		flags = Choose(Bit(flags, 2), And(flags, C(~0x30u)), flags);
		return Or(flags, Choose(Eq(And(flags, C(0x30)), C(0x30)), C(0x100), C(0)));
	}

	void     DefineTriangle();
	uint32_t Instance(uint32_t block, uint32_t flags, uint32_t mask, Vec3 org, Vec3 dir);
	void     DefineDecompress();
	void     DefineBoxes();
	uint32_t Boxes(uint32_t first, uint32_t second, uint32_t full, uint32_t compressed,
	               uint32_t flags, uint32_t descriptor, uint32_t extent, uint32_t org, uint32_t dir);
	void     DefineIntersect();
};

void Bvh8Emitter::DefineTriangle() {
	const auto a = Begin(triangle_function, "trinity_triangle", v4, std::array {w, u, u, u, v3, v3});
	const auto block = a[0], slot = a[1], flags = a[2], descriptor = a[3];
	const auto origin = Unpack3(a[4]), direction = Unpack3(a[5]);
	const auto flag_word     = Word(block, C(15));
	const auto remap         = Binary(s, spv::OpShiftRightLogical, u, flag_word, Shl(slot, 3));
	const auto bary          = Bit(descriptor, 24);
	const auto pointer_flags = Both(bary, Bit(descriptor, 23));
	const auto tri_flags     = Choose(pointer_flags, remap, C(0));
	const auto procedural    = Bit(tri_flags, 6);
	const auto raw           = EmitValueOrDefaultIfCondition(
        s, Not(procedural), v4, Make4(C(0), C(0x3f800000), C(0x3f800000), C(0x3f800000)), [&] {
            std::array<Vec3, 3> positions;
            const std::array    vertices {slot, Choose(Eq(slot, C(0)), C(1), C(3)), C(2)};
            for (uint32_t vertex = 0; vertex < 3; ++vertex) {
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    positions[vertex][axis] = Float(
                        Word(block, Add(Binary(s, spv::OpIMul, u, vertices[vertex], C(3)), C(axis))));
                }
            }
            return EmitBvh8Triangle(s, positions, origin, direction, remap, Add(flag_word, slot),
                                    bary);
        });
	std::array<uint32_t, 4> result;
	for (uint32_t i = 0; i < 4; ++i) {
		result[i] = Extract(u, raw, i);
	}
	auto       missed = Eq(result[0], C(0x7f800000));
	const auto flip   = Both(Both(Not(procedural), Not(missed)),
	                         Binary(s, spv::OpLogicalNotEqual, b, Bit(flags, 3), Bit(tri_flags, 5)));
	for (uint32_t i = 0; i < 4; ++i) {
		result[i] = Binary(s, spv::OpBitwiseXor, u, result[i],
		                   Choose(i < 2 ? flip : Both(flip, bary), C(0x80000000), C(0)));
	}
	const auto front  = Fcmp(spv::OpFOrdGreaterThan, Float(result[1]), ConstantF32Value(s, 0.0f));
	const auto opaque = Both(Either(Bit(tri_flags, 7), Bit(flags, 0)), Not(Bit(flags, 1)));
	missed = Either(missed, Either(Both(opaque, Bit(flags, 6)), Both(Not(opaque), Bit(flags, 7))));
	missed = Either(missed, Both(Both(Not(Bit(tri_flags, 4)), Not(procedural)),
	                             Select(s, b, front, Bit(flags, 5), Bit(flags, 4))));
	missed = Either(missed, Select(s, b, procedural, Bit(flags, 9), Bit(flags, 8)));
	const auto sign = [&](uint32_t value, uint32_t enabled) {
		return Choose(pointer_flags,
		              Or(And(value, C(0x7fffffff)), Choose(enabled, C(0x80000000), C(0))), value);
	};
	result[2] = sign(result[2], Either(Not(opaque), procedural));
	result[3] = sign(result[3], Not(opaque));
	const std::array miss {C(0x7f800000), C(0x3f800000), Choose(bary, C(0), C(~0u)), C(0)};
	for (uint32_t i = 0; i < 4; ++i) {
		result[i] = Choose(missed, miss[i], result[i]);
	}
	End(Make(v4, result));
}

uint32_t Bvh8Emitter::Instance(uint32_t block, uint32_t flags, uint32_t mask, Vec3 org, Vec3 dir) {
	const auto lo         = Word(block, C(12));
	const auto hi         = Word(block, C(13));
	const auto inst_flags = Shr(hi, 22);
	auto       merged     = Or(And(Or(flags, inst_flags), C(0x3f0)), And(inst_flags, C(4)));
	merged = Or(merged, And(Binary(s, spv::OpBitwiseXor, u, flags, inst_flags), C(8)));
	merged = Or(merged, Choose(Either(Bit(flags, 0), Both(Bit(inst_flags, 0), Not(Bit(flags, 1)))),
	                           C(1), C(0)));
	merged = Or(merged, Choose(Either(Bit(flags, 1), Both(Bit(inst_flags, 1), Not(Bit(flags, 0)))),
	                           C(2), C(0)));
	const auto               user = Word(block, C(15));
	std::array<uint32_t, 10> dst;
	dst.fill(C(~0u));
	dst[2] = lo;
	dst[3] = Or(And(hi, C(0x3fffff)), Shl(merged, 22));
	dst[6] = And(user, C(0xffffff));
	dst[7] = Choose(Eq(And(mask, Shr(user, 24)), C(0)), C(~0u), Word(block, C(14)));
	Vec3 transformed_org;
	Vec3 transformed_dir;
	for (uint32_t axis = 0; axis < 3; ++axis) {
		Vec3 row;
		for (uint32_t i = 0; i < 3; ++i) {
			row[i] = Float(Word(block, C(axis * 4 + i)));
		}
		const auto dot = [&](const Vec3& value) {
			return Fop(spv::OpFAdd,
			           Fop(spv::OpFAdd, Fop(spv::OpFMul, row[0], value[0]),
			               Fop(spv::OpFMul, row[1], value[1])),
			           Fop(spv::OpFMul, row[2], value[2]));
		};
		transformed_org[axis] = Fop(spv::OpFAdd, dot(org), Float(Word(block, C(axis * 4 + 3))));
		transformed_dir[axis] = dot(dir);
	}
	return Result(dst, transformed_org, transformed_dir, C(1));
}

void Bvh8Emitter::DefineDecompress() {
	const auto a = Begin(decompress_function, "trinity_shared_exponent", f, std::array {u, u, b});
	const auto exponent = a[0], packed = a[1], minimum = a[2];
	const auto negative = Bit(packed, 17), mantissa = And(packed, C(0x1ffff));
	const auto outward  = Eq(Choose(negative, C(1), C(0)), Choose(minimum, C(1), C(0)));
	const auto nonzero  = Not(Eq(mantissa, C(0)));
	const auto highest  = EmitGlsl<GLSLstd450FindUMsb, IR::Type::U32>(s, mantissa);
	const auto shift    = UMin(Sub(C(16), highest), exponent);
	const auto decoded_exp =
	    Choose(nonzero, Sub(exponent, shift),
	           Choose(Both(outward, Binary(s, spv::OpUGreaterThan, b, exponent, C(17))),
	                  Sub(exponent, C(17)), C(0)));
	// Zero mantissas take a separate result, but their unused shifts must still be defined.
	const auto amount = Add(C(7), Choose(nonzero, shift, C(0)));
	const auto trail  = Choose(Both(outward, Not(Eq(decoded_exp, C(255)))),
	                           Sub(Binary(s, spv::OpShiftLeftLogical, u, C(1), amount), C(1)), C(0));
	const auto fraction =
	    Choose(nonzero,
	           And(Or(Binary(s, spv::OpShiftLeftLogical, u, mantissa, amount), trail), C(0x7fffff)),
	           Choose(outward, C(0x7fffff), C(0)));
	End(Float(Or(Or(fraction, Shl(decoded_exp, 23)), Choose(negative, C(0x80000000), C(0)))));
}

void Bvh8Emitter::DefineBoxes() {
	const auto result_type = s.builder.Type(spv::OpTypeStruct, v4, v4);
	const auto a           = Begin(boxes_function, "trinity_box_slot", result_type,
	                               std::array {w, w, b, b, u, u, f, v3, v3});
	const auto first = a[0], second = a[1], full = a[2], compressed = a[3];
	const auto flags = a[4], descriptor = a[5], extent = a[6];
	const auto org = Unpack3(a[7]), dir = Unpack3(a[8]);
	const auto zero     = ConstantF32Value(s, 0.0f);
	const auto infinity = ConstantF32(s, 0x7f800000);
	Vec3       inverse;
	for (uint32_t i = 0; i < 3; ++i) {
		inverse[i] = Fop(spv::OpFDiv, ConstantF32Value(s, 1.0f), dir[i]);
	}
	const auto packed = [&](uint32_t offset, uint32_t width) {
		const auto word  = Shr(offset, 5);
		const auto shift = And(offset, C(31));
		const auto lo    = Binary(s, spv::OpShiftRightLogical, u, Word(first, word), shift);
		const auto next  = UMin(Add(word, C(1)), C(15));
		const auto hi    = Binary(s, spv::OpShiftLeftLogical, u, Word(first, next),
		                          And(Sub(C(32), shift), C(31)));
		return And(Or(lo, Choose(Eq(shift, C(0)), C(0), hi)), C((1u << width) - 1));
	};
	const auto format        = Choose(compressed, C(0), Choose(full, C(1), C(2)));
	const auto entry_label   = s.current_label;
	const auto header        = s.builder.AllocateId();
	const auto body          = s.builder.AllocateId();
	const auto cont          = s.builder.AllocateId();
	const auto merge         = s.builder.AllocateId();
	const auto initial_nodes = Make4(C(~0u), C(~0u), C(~0u), C(~0u));
	const auto initial_keys  = Make4(C(0x7f800000), C(0x7f800000), C(0x7f800000), C(0x7f800000));
	const auto base          = And(Word(first, C(0)), C(0x1fffffff));
	s.builder.AddFunction(spv::OpBranch, header);
	EmitLabel(s, header);
	const auto child      = s.builder.AllocateId();
	const auto prefix     = s.builder.AllocateId();
	const auto nodes      = s.builder.AllocateId();
	const auto keys       = s.builder.AllocateId();
	const auto child_phi  = s.builder.AddDeferredPhi(u, child, 2);
	const auto prefix_phi = s.builder.AddDeferredPhi(u, prefix, 2);
	const auto nodes_phi  = s.builder.AddDeferredPhi(v4, nodes, 2);
	const auto keys_phi   = s.builder.AddDeferredPhi(v4, keys, 2);
	const auto more       = Binary(s, spv::OpULessThan, b, child, C(4));
	s.builder.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlDontUnrollMask);
	s.builder.AddFunction(spv::OpBranchConditional, more, body, merge);
	EmitLabel(s, body);
	const auto kind        = packed(Add(C(29), Binary(s, spv::OpIMul, u, child, C(3))), 3);
	const auto direct_node = Word(first, child);
	const auto node        = Choose(compressed, Or(Shl(prefix, 3), kind), direct_node);
	const auto child_flags =
	    Choose(compressed,
	           And(packed(Add(C(41), Shl(child, 2)), 4), Choose(Eq(child, C(3)), C(7), C(15))), C(0));
	const auto step =
	    Choose(Eq(kind, C(2)), C(4),
	           Choose(Either(Either(Eq(kind, C(3)), Eq(kind, C(5))), Eq(kind, C(6))), C(2), C(1)));
	const auto next_prefix = Add(prefix, step);
	const auto bounds_type = s.builder.Type(spv::OpTypeStruct, v3, v3);
	const auto box         = EmitIndexSwitch(s, format, 3, bounds_type, [&](uint32_t encoding) {
        std::array<uint32_t, 6> bounds;
        if (encoding == 0) {
            const auto       yz = Word(first, C(2));
            const std::array exponents {Shr(Word(first, C(1)), 24), And(yz, C(255)),
                                        And(Shr(yz, 8), C(255))};
            for (uint32_t component = 0; component < 6; ++component) {
                const auto offset =
                    Add(C(80 + component * 18), Binary(s, spv::OpIMul, u, child, C(108)));
                bounds[component] =
                    Call(f, decompress_function,
                         std::array {exponents[component % 3], packed(offset, 18),
                                     ConstantBool(s, component < 3)});
            }
        } else if (encoding == 1) {
            for (uint32_t component = 0; component < 6; ++component) {
                const auto word  = Add(C(4 + component), Binary(s, spv::OpIMul, u, child, C(6)));
                const auto block = Select(s, w, Binary(s, spv::OpULessThan, b, word, C(16)), first,
                                          second);
                bounds[component] = Float(Word(block, And(word, C(15))));
            }
        } else {
            for (uint32_t pair = 0; pair < 3; ++pair) {
                const auto word  = Add(C(4 + pair), Binary(s, spv::OpIMul, u, child, C(3)));
                const auto value = s.builder.AllocateId();
                s.builder.AddFunction(spv::OpExtInst, TypeF32Vector(s, 2), value, GlslStd450(s),
                                      GLSLstd450UnpackHalf2x16, Word(first, word));
                bounds[pair * 2]     = Extract(f, value, 0);
                bounds[pair * 2 + 1] = Extract(f, value, 1);
            }
        }
        return Make(bounds_type, std::array {Make(v3, std::span(bounds).first(3)),
                                             Make(v3, std::span(bounds).last(3))});
	});
	const auto lo_bound = Unpack3(Extract(v3, box, 0));
	const auto hi_bound = Unpack3(Extract(v3, box, 1));
	Vec3       near_t;
	Vec3       far_t;
	auto       valid = Not(Eq(And(child_flags, C(3)), C(3)));
	for (uint32_t axis = 0; axis < 3; ++axis) {
		valid         = Both(valid, Fcmp(spv::OpFOrdLessThanEqual, lo_bound[axis], hi_bound[axis]));
		const auto lo = Fop(spv::OpFMul, Fop(spv::OpFSub, lo_bound[axis], org[axis]), inverse[axis]);
		const auto hi = Fop(spv::OpFMul, Fop(spv::OpFSub, hi_bound[axis], org[axis]), inverse[axis]);
		const auto forward = Fcmp(spv::OpFOrdGreaterThanEqual, inverse[axis], zero);
		near_t[axis]       = Select(s, f, forward, lo, hi);
		far_t[axis]        = Select(s, f, forward, hi, lo);
	}
	const auto entry     = Max(Max(near_t[0], near_t[1]), near_t[2]);
	const auto exit      = Min(Min(far_t[0], far_t[1]), far_t[2]);
	const auto exit_bits = Raw(Select(s, f, Fcmp(spv::OpFOrdEqual, exit, zero), zero, exit));
	const auto grown =
	    Float(UMin(Add(exit_bits, And(Shr(descriptor, 23), C(255))), C(0x7f800000)));
	valid = Both(valid, Both(Fcmp(spv::OpFOrdGreaterThanEqual, exit, zero),
	                         Both(Fcmp(spv::OpFOrdLessThan, entry, extent),
	                              Fcmp(spv::OpFOrdLessThanEqual, entry, grown))));
	const auto opaque = Both(Either(Bit(child_flags, 0), Bit(flags, 0)), Not(Bit(flags, 1)));
	const auto nonopaque = Both(Either(Bit(child_flags, 1), Bit(flags, 1)), Not(Bit(flags, 0)));
	const auto culled =
	    Either(Either(Both(opaque, Bit(flags, 6)), Both(nonopaque, Bit(flags, 7))),
	           Either(Both(Bit(child_flags, 2), Bit(flags, 8)), Both(Bit(child_flags, 3), Bit(flags, 9))));
	valid                = Both(valid, Not(culled));
	const auto heuristic = And(Shr(descriptor, 21), C(3));
	auto       key       = Max(entry, zero);
	key = Select(s, f, Eq(heuristic, C(1)),
	             Unary(s, spv::OpFNegate, f, Fop(spv::OpFSub, exit, entry)), key);
	key = Select(s, f, Eq(heuristic, C(2)), Fop(spv::OpFAdd, entry, exit), key);
	const auto next_nodes = s.builder.AllocateId();
	const auto next_keys  = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpVectorInsertDynamic, v4, next_nodes, nodes,
	                      Choose(valid, node, C(~0u)), child);
	s.builder.AddFunction(spv::OpVectorInsertDynamic, v4, next_keys, keys,
	                      Raw(Select(s, f, valid, key, infinity)), child);
	s.builder.AddFunction(spv::OpBranch, cont);
	EmitLabel(s, cont);
	const auto next_child = Add(child, C(1));
	s.builder.AddFunction(spv::OpBranch, header);
	s.builder.PatchDeferredPhi(child_phi, 0, C(0), entry_label);
	s.builder.PatchDeferredPhi(child_phi, 1, next_child, cont);
	s.builder.PatchDeferredPhi(prefix_phi, 0, base, entry_label);
	s.builder.PatchDeferredPhi(prefix_phi, 1, next_prefix, cont);
	s.builder.PatchDeferredPhi(nodes_phi, 0, initial_nodes, entry_label);
	s.builder.PatchDeferredPhi(nodes_phi, 1, next_nodes, cont);
	s.builder.PatchDeferredPhi(keys_phi, 0, initial_keys, entry_label);
	s.builder.PatchDeferredPhi(keys_phi, 1, next_keys, cont);
	EmitLabel(s, merge);
	End(Make(result_type, std::array {nodes, keys}));
}

uint32_t Bvh8Emitter::Boxes(uint32_t first, uint32_t second, uint32_t full, uint32_t compressed,
                            uint32_t flags, uint32_t descriptor, uint32_t extent, uint32_t org,
                            uint32_t dir) {
	return Call(s.builder.Type(spv::OpTypeStruct, v4, v4), boxes_function,
	            std::array {first, second, full, compressed, flags, descriptor, extent, org, dir});
}

void Bvh8Emitter::DefineIntersect() {
	const auto result_type = TypeBvh8Result(s);
	const auto a = Begin(s.bvh8_intersect_function, "trinity_bvh_intersect", result_type,
	                     std::array {v4, w, u, f, u, v3, v3});
	const auto descriptor = a[0], encoded_base = a[1], node = a[2], extent = a[3], mask = a[4];
	const auto org = Unpack3(a[5]), dir = Unpack3(a[6]);
	std::array<uint32_t, 4> d;
	for (uint32_t i = 0; i < 4; ++i) {
		d[i] = Extract(u, descriptor, i);
	}
	const auto kind       = And(node, C(7));
	const auto triangle   = Binary(s, spv::OpULessThan, b, kind, C(2));
	const auto instance   = Eq(kind, C(6));
	const auto full       = Either(Eq(kind, C(2)), Eq(kind, C(5)));
	const auto big        = Either(Eq(kind, C(2)), Eq(kind, C(3)));
	const auto compressed = Both(Not(full), Bit(d[3], 20));
	const auto flags      = NormalizeFlags(
        Choose(Bit(d[3], 23),
               Unary(s, spv::OpUConvert, u,
                     Binary(s, spv::OpShiftRightLogical, w, encoded_base, ConstantU64(s, 54))),
               C(0)));
	const auto base =
	    Binary(s, spv::OpShiftLeftLogical, w, PackU64(s, d[0], And(d[1], C(255))), ConstantU64(s, 8));
	const auto lane_base = Binary(
	    s, spv::OpShiftLeftLogical, w,
	    Binary(s, spv::OpBitwiseAnd, w, encoded_base, ConstantU64(s, (uint64_t {1} << 54) - 1)),
	    ConstantU64(s, 3));
	const auto offset = Binary(s, spv::OpShiftLeftLogical, w,
	                           Unary(s, spv::OpUConvert, w, Shr(node, 3)), ConstantU64(s, 6));
	const auto address =
	    Binary(s, spv::OpIAdd, w, base, Binary(s, spv::OpIAdd, w, lane_base, offset));
	const auto capacity =
	    Binary(s, spv::OpIAdd, w, PackU64(s, d[2], And(d[3], C(0x3ff))), ConstantU64(s, 1));
	const auto limit =
	    Binary(s, spv::OpIAdd, w, base,
	           Binary(s, spv::OpShiftLeftLogical, w, capacity, ConstantU64(s, 6)));
	auto valid = Both(Eq(Shr(d[3], 28), C(8)),
	                  Eq(And(Unary(s, spv::OpUConvert, u, encoded_base), C(7)), C(0)));
	valid      = Both(valid, Both(Not(Eq(kind, C(7))), Either(Not(instance), Bit(d[3], 22))));
	valid      = Both(valid, Not(Unary(s, spv::OpIsNan, b, extent)));
	for (uint32_t i = 0; i < 3; ++i) {
		valid = Both(valid, Not(Either(Unary(s, spv::OpIsNan, b, org[i]),
		                               Unary(s, spv::OpIsInf, b, org[i]))));
		valid = Both(valid, Not(Unary(s, spv::OpIsNan, b, dir[i])));
	}
	const auto opaque    = Bit(flags, 0);
	const auto nonopaque = Bit(flags, 1);
	valid = Both(valid, Not(Either(Either(Both(opaque, Bit(flags, 6)), Both(nonopaque, Bit(flags, 7))),
	                               Both(Bit(flags, 6), Bit(flags, 7)))));
	const auto fetch_size = Select(s, w, full, ConstantU64(s, 128), ConstantU64(s, 64));
	// The device address of a 64-byte guest block (it never crosses a BDA page); zero when it is
	// outside the descriptor's node range, the 48-bit space or unmapped.
	const auto fetch = [&](uint32_t at, uint32_t enabled) {
		const auto end       = Binary(s, spv::OpIAdd, w, at, fetch_size);
		const auto in_bounds = Both(Binary(s, spv::OpULessThanEqual, b, end, limit),
		                            Binary(s, spv::OpULessThanEqual, b, end,
		                                   ConstantU64(s, (uint64_t {1} << 48) - 1)));
		return EmitValueOrDefaultIfCondition(s, Both(Both(valid, enabled), in_bounds), w,
		                                     ConstantU64(s, 0),
		                                     [&] { return EmitBdaDeviceAddress(*ctx, at); });
	};
	const auto first   = fetch(address, ConstantBool(s, true));
	const auto present = [&](uint32_t pointer) {
		return Not(Binary(s, spv::OpIEqual, b, pointer, ConstantU64(s, 0)));
	};
	const auto tail = [&](uint32_t at, uint32_t pointer) {
		return EmitValueOrDefaultIfCondition(s, Both(full, present(pointer)), w, ConstantU64(s, 0),
		                                     [&] {
			                                     return EmitBdaDeviceAddress(
			                                         *ctx, Binary(s, spv::OpIAdd, w, at,
			                                                      ConstantU64(s, 64)));
		                                     });
	};
	const auto second    = tail(address, first);
	const auto available = Both(present(first), Either(Not(full), present(second)));
	std::array<uint32_t, 10> invalid;
	invalid.fill(C(~0u));
	for (uint32_t slot = 0; slot < 2; ++slot) {
		invalid[slot * 4]     = Choose(triangle, C(0x7f800000), C(~0u));
		invalid[slot * 4 + 1] = Choose(triangle, C(0x3f800000), C(~0u));
		invalid[slot * 4 + 3] = Choose(triangle, C(0), C(~0u));
	}
	const auto default_result = Result(invalid, org, dir, C(0));
	const auto value = EmitValueOrDefaultIfCondition(s, Both(instance, available), result_type,
	                                                 default_result,
	                                                 [&] { return Instance(first, flags, mask, org, dir); });
	const auto result = EmitValueOrDefaultIfCondition(s, Both(Not(instance), valid), result_type, value, [&] {
		const auto second_address = Binary(s, spv::OpIAdd, w, address, fetch_size);
		const auto next_first     = fetch(second_address, big);
		const auto next_second    = tail(second_address, next_first);
		const auto next_available = Both(present(next_first), Either(Not(full), present(next_second)));
		const auto triangles = EmitValueOrDefaultIfCondition(s, triangle, result_type, default_result, [&] {
			auto dst = invalid;
			for (uint32_t slot = 0; slot < 2; ++slot) {
				const auto active = Both(available, slot == 0 ? ConstantBool(s, true) : Eq(kind, C(1)));
				const auto absent = Both(available, Both(Eq(kind, C(0)), Bit(d[3], 24)));
				const auto miss   = Make4(C(0x7f800000), C(0x3f800000),
				                          slot == 1 ? Choose(absent, C(0), C(~0u)) : C(~0u), C(0));
				const auto hit    = EmitValueOrDefaultIfCondition(s, active, v4, miss, [&] {
                    return Call(v4, triangle_function,
                                std::array {first, C(slot), flags, d[3], a[5], a[6]});
				});
				for (uint32_t i = 0; i < 4; ++i) {
					dst[slot * 4 + i] = Extract(u, hit, i);
				}
				dst[8 + slot] = EmitValueOrDefaultIfCondition(s, active, u, C(~0u),
				                                              [&] { return Word(first, C(14)); });
			}
			return Result(dst, org, dir, C(0));
		});
		return EmitValueOrDefaultIfCondition(s, Not(triangle), result_type, triangles, [&] {
			auto       dst       = invalid;
			const auto slot_type = s.builder.Type(spv::OpTypeStruct, v4, v4);
			const auto miss      = Make(slot_type,
			                            std::array {Make4(C(~0u), C(~0u), C(~0u), C(~0u)),
			                                        Make4(C(0x7f800000), C(0x7f800000), C(0x7f800000),
			                                              C(0x7f800000))});
			std::array<uint32_t, 8> keys;
			for (uint32_t slot = 0; slot < 2; ++slot) {
				const auto hit = EmitValueOrDefaultIfCondition(
				    s, slot == 0 ? available : next_available, slot_type, miss, [&] {
					    return Boxes(slot == 0 ? first : next_first, slot == 0 ? second : next_second,
					                 full, compressed, flags, d[1], extent, a[5], a[6]);
				    });
				const auto indices = Extract(v4, hit, 0);
				const auto depths  = Extract(v4, hit, 1);
				for (uint32_t i = 0; i < 4; ++i) {
					dst[slot * 4 + i]  = Extract(u, indices, i);
					keys[slot * 4 + i] = Extract(u, depths, i);
				}
			}
			const auto sort      = Bit(d[1], 31);
			const auto wide      = Both(sort, Bit(d[3], 21));
			const auto tri_first = Bit(d[1], 20);
			constexpr std::array<std::pair<uint32_t, uint32_t>, 19> network {
			    {{0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {4, 5}, {6, 7}, {4, 6}, {5, 7}, {5, 6},
			     {0, 4}, {1, 5}, {1, 4}, {2, 6}, {3, 7}, {3, 6}, {2, 4}, {3, 5}, {3, 4}}};
			const auto unsorted = Make(slot_type, std::array {Make(v4, std::span(dst).first(4)),
			                                                  Make(v4, std::span(dst).subspan(4, 4))});
			const auto sorted_result = EmitValueOrDefaultIfCondition(s, sort, slot_type, unsorted, [&] {
				auto sorted_dst = dst;
				for (size_t step = 0; step < network.size(); ++step) {
					const auto [x, y]  = network[step];
					const auto is_tri  = [&](uint32_t index) {
                        return Binary(s, spv::OpULessThan, b, And(index, C(7)), C(2));
					};
					const auto xt   = is_tri(sorted_dst[x]);
					const auto yt   = is_tri(sorted_dst[y]);
					auto       swap = Fcmp(spv::OpFOrdGreaterThan, Float(keys[x]), Float(keys[y]));
					swap = Select(s, b, Both(tri_first, Binary(s, spv::OpLogicalNotEqual, b, xt, yt)),
					              yt, swap);
					if (step >= 10) {
						swap = Both(wide, swap);
					}
					const auto key   = keys[x];
					const auto index = sorted_dst[x];
					const auto later = [&](uint32_t position) {
						return std::ranges::any_of(std::span(network).subspan(step + 1),
						                           [=](auto pair) {
							                           return pair.first == position ||
							                                  pair.second == position;
						                           });
					};
					if (later(x)) {
						keys[x] = Choose(swap, keys[y], key);
					}
					if (later(y)) {
						keys[y] = Choose(swap, key, keys[y]);
					}
					sorted_dst[x] = Choose(swap, sorted_dst[y], index);
					sorted_dst[y] = Choose(swap, index, sorted_dst[y]);
				}
				return Make(slot_type, std::array {Make(v4, std::span(sorted_dst).first(4)),
				                                   Make(v4, std::span(sorted_dst).subspan(4, 4))});
			});
			for (uint32_t slot = 0; slot < 2; ++slot) {
				const auto indices = Extract(v4, sorted_result, slot);
				for (uint32_t i = 0; i < 4; ++i) {
					dst[slot * 4 + i] = Extract(u, indices, i);
				}
			}
			return Result(dst, org, dir, C(0));
		});
	});
	End(result);
}

bool HasBvh8(const EmitterState& state) {
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::Bvh8IntersectRay) {
				return true;
			}
		}
	}
	return false;
}

} // namespace

void DefineBvh8IntersectFunction(EmitterState& state) {
	if (!state.program.info.uses_dma || !HasBvh8(state)) {
		return;
	}
	ValueEmitContext ctx(state);
	Bvh8Emitter      e {state, &ctx};
	e.DefineDecompress();
	e.DefineTriangle();
	e.DefineBoxes();
	e.DefineIntersect();
}

uint32_t EmitBvh8Extract(ValueEmitContext& ctx, const IR::Inst& inst) {
	const auto index = inst.Arg(1).U32();
	if (index > 16u) {
		ctx.Fail(inst, "invalid BVH8 result component");
	}
	const auto result = ctx.state.builder.AllocateId();
	if (index == 16u) {
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), result,
		                              ctx.Arg(inst, 0), 4u);
	} else {
		ctx.state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(ctx.state), result,
		                              ctx.Arg(inst, 0), index / 4u, index % 4u);
	}
	return result;
}

uint32_t EmitBvh8IntersectRay(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	if (s.bvh8_intersect_function == 0) {
		ctx.Fail(inst, "BVH8 intersection requires its module function (BDA access)");
	}
	// Every operand is read before the first branch.
	Bvh8Emitter e {s, &ctx};
	std::array<uint32_t, 11> ray;
	for (uint32_t i = 0; i < ray.size(); i++) {
		ray[i] = ctx.Arg(inst, 2u + i);
	}
	const auto descriptor = ctx.Arg(inst, 1);
	const auto active     = ctx.Arg(inst, 13);
	const auto vector     = [&](uint32_t first) {
        return e.Make(e.v3, Vec3 {e.Float(ray[first]), e.Float(ray[first + 1]), e.Float(ray[first + 2])});
	};
	const std::array args {descriptor, PackU64(s, ray[0], ray[1]), ray[10], e.Float(ray[2]),
	                       ray[3],     vector(4),                  vector(7)};
	const auto type = TypeBvh8Result(s);
	return EmitValueOrDefaultIfCondition(s, active, type, s.builder.Constant(spv::OpConstantNull, type),
	                                     [&] { return e.Call(type, s.bvh8_intersect_function, args); });
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
