#include "graphics/shader/recompiler/backend/spirv/spirvEmitterRayTracing.h"

#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/shader/recompiler/CodegenOptions.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"

#include <algorithm>
#include <array>
#include <span>
#include <vector>

// Software IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY (IR BvhIntersectRay, built by
// KYTY_RT_SOFTWARE). The node test follows AMD GPURT's emulation of the RTIP 1.1 instruction
// (fast_intersect_bbox, IntersectNodeBvh4, fast_intersect_triangle, SwizzleBarycentrics) with
// IEEE f32 arithmetic: separately rounded operations (NoContraction), NaN-ignoring max/min and
// correctly rounded division. Node layouts are RDNA2's (Mesa RADV bvh.h, GPURT), plus the PS5's
// type-6 shared-exponent box (Psr's decoder). The full specification and the CPU reference are
// Profiling/analysis/RT-SOFTWARE-DESIGN.md and graphics/shader/recompiler/BvhReference.h.

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

constexpr uint32_t InvalidNode       = 0xffffffffu;
constexpr uint32_t PlusInfBits       = 0x7f800000u;
constexpr uint32_t MinusInfBits      = 0xff800000u;
constexpr uint32_t OneBits           = 0x3f800000u;
constexpr uint32_t ZeroBits          = 0u;
constexpr uint32_t QuietNanBits      = 0x7fc00000u;
constexpr uint32_t TwoPowMinus24Bits = 0x33800000u; // 2^-24, GPURT's box growing "eps"
// Offset bits within one BDA page.
constexpr uint64_t BufferCachePageMask = BufferCache::CACHING_PAGESIZE - 1u;

using Vec3 = std::array<uint32_t, 3>;

class BvhLowering {
public:
	BvhLowering(ValueEmitContext& ctx_, const BvhOperands& operands)
	    : ctx(ctx_), s(ctx_.state), flush_f32(GetHostFloatControls().denorm_flush_f32),
	      tsharp(operands.tsharp), node_lo(operands.node_lo), node_hi(operands.node_hi),
	      extent(operands.extent), origin(operands.origin), direction(operands.direction),
	      inverse(operands.inverse) {}

	uint32_t NodeTest();

private:
	uint32_t F32() { return TypeF32(s); }
	uint32_t U32() { return TypeU32(s); }
	uint32_t U64() { return TypeU64(s); }
	uint32_t Bool() { return TypeBool(s); }
	uint32_t U32x4() { return TypeU32Vector(s, 4); }
	uint32_t Cu(uint32_t value) { return ConstantU32(s, value); }
	uint32_t Cf(uint32_t bits) { return ConstantF32(s, bits); }
	uint32_t Cu64(uint64_t value) { return EmitDeviceAddressConstant(s, value); }

	uint32_t Op(spv::Op op, uint32_t type, uint32_t a, uint32_t b) {
		return Binary(s, op, type, a, b);
	}
	uint32_t Op1(spv::Op op, uint32_t type, uint32_t a) { return Unary(s, op, type, a); }
	uint32_t Sel(uint32_t type, uint32_t cond, uint32_t a, uint32_t b) {
		return Select(s, type, cond, a, b);
	}
	uint32_t And(uint32_t a, uint32_t b) { return Op(spv::OpLogicalAnd, Bool(), a, b); }
	uint32_t Or(uint32_t a, uint32_t b) { return Op(spv::OpLogicalOr, Bool(), a, b); }
	uint32_t Not(uint32_t a) { return Op1(spv::OpLogicalNot, Bool(), a); }
	uint32_t Extract(uint32_t type, uint32_t composite, uint32_t index) {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, type, result, composite, index);
		return result;
	}
	uint32_t ToF32(uint32_t bits) { return Op1(spv::OpBitcast, F32(), bits); }
	uint32_t ToU32(uint32_t value) { return Op1(spv::OpBitcast, U32(), value); }

	// Separately rounded f32 arithmetic: the host compiler must not fuse these.
	uint32_t FArith(spv::Op op, uint32_t a, uint32_t b) {
		const auto result = Op(op, F32(), a, b);
		s.builder.AddAnnotation(spv::OpDecorate, result, spv::DecorationNoContraction);
		return result;
	}
	uint32_t FAdd(uint32_t a, uint32_t b) { return FArith(spv::OpFAdd, a, b); }
	uint32_t FSub(uint32_t a, uint32_t b) { return FArith(spv::OpFSub, a, b); }
	uint32_t FMul(uint32_t a, uint32_t b) { return FArith(spv::OpFMul, a, b); }
	uint32_t IsNan(uint32_t a) { return Op1(spv::OpIsNan, Bool(), a); }
	// IEEE maxNum/minNum: a NaN operand is ignored; a tie returns the second operand. Explicit
	// compares and selects rather than GLSL NMax/NMin: those were no faster here, and under a
	// flushing float mode the driver may treat a denormal against a zero differently from the
	// compares around it (see EmitFastMinMaxF32).
	uint32_t MaxNum(uint32_t a, uint32_t b) {
		const auto take_a = Or(IsNan(b), Op(spv::OpFOrdGreaterThan, Bool(), a, b));
		return Sel(F32(), IsNan(a), b, Sel(F32(), take_a, a, b));
	}
	uint32_t MinNum(uint32_t a, uint32_t b) {
		const auto take_a = Or(IsNan(b), Op(spv::OpFOrdLessThan, Bool(), a, b));
		return Sel(F32(), IsNan(a), b, Sel(F32(), take_a, a, b));
	}
	Vec3 Sub3(const Vec3& a, const Vec3& b) {
		return {FSub(a[0], b[0]), FSub(a[1], b[1]), FSub(a[2], b[2])};
	}
	Vec3 Cross(const Vec3& p, const Vec3& q) {
		return {FSub(FMul(p[1], q[2]), FMul(p[2], q[1])), FSub(FMul(p[2], q[0]), FMul(p[0], q[2])),
		        FSub(FMul(p[0], q[1]), FMul(p[1], q[0]))};
	}
	uint32_t Dot(const Vec3& p, const Vec3& q) {
		return FAdd(FAdd(FMul(p[0], q[0]), FMul(p[1], q[1])), FMul(p[2], q[2]));
	}
	uint32_t U64Of(uint32_t low, uint32_t high) {
		return Op(spv::OpBitwiseOr, U64(), Op1(spv::OpUConvert, U64(), low),
		          Op(spv::OpShiftLeftLogical, U64(), Op1(spv::OpUConvert, U64(), high), Cu64(32)));
	}

	template <typename Then, typename Else>
	uint32_t IfElse(uint32_t condition, uint32_t type, Then&& then_fn, Else&& else_fn) {
		const auto then_label  = s.builder.AllocateId();
		const auto else_label  = s.builder.AllocateId();
		const auto merge_label = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpSelectionMerge, merge_label, spv::SelectionControlMaskNone);
		s.builder.AddFunction(spv::OpBranchConditional, condition, then_label, else_label);
		EmitLabel(s, then_label);
		const auto then_value = then_fn();
		const auto then_exit  = s.current_label;
		s.builder.AddFunction(spv::OpBranch, merge_label);
		EmitLabel(s, else_label);
		const auto else_value = else_fn();
		const auto else_exit  = s.current_label;
		s.builder.AddFunction(spv::OpBranch, merge_label);
		EmitLabel(s, merge_label);
		const auto value = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpPhi, type, value, then_value, then_exit, else_value,
		                      else_exit);
		return value;
	}

	// Bits [first, first + count) of a node's little-endian bit stream (count < 32).
	uint32_t NodeBits(const std::array<uint32_t, 16>& dwords, uint32_t first, uint32_t count) {
		const uint32_t word  = first / 32u;
		const uint32_t shift = first % 32u;
		auto value = shift == 0u ? dwords[word]
		                         : Op(spv::OpShiftRightLogical, U32(), dwords[word], Cu(shift));
		if (shift + count > 32u) {
			value = Op(spv::OpBitwiseOr, U32(), value,
			           Op(spv::OpShiftLeftLogical, U32(), dwords[word + 1u], Cu(32u - shift)));
		}
		return Op(spv::OpBitwiseAnd, U32(), value, Cu((1u << count) - 1u));
	}

	std::array<uint32_t, 16> LoadBlock(uint32_t device);
	uint32_t                 ExactDiv(uint32_t x, uint32_t y);
	uint32_t BoxResult(const std::array<uint32_t, 4>& children, const std::array<Vec3, 4>& lo,
	                   const std::array<Vec3, 4>& hi);
	uint32_t Box16(uint32_t address);
	uint32_t Box32(uint32_t address);
	uint32_t SharedExpBound(uint32_t field, uint32_t exponent, bool is_min);
	uint32_t BoxPs5(uint32_t address);
	uint32_t Triangle(uint32_t address);

	ValueEmitContext&       ctx;
	EmitterState&           s;
	const bool              flush_f32;
	std::array<uint32_t, 4> tsharp {};
	uint32_t                node_lo   = 0;
	uint32_t                node_hi   = 0;
	uint32_t                node_type = 0;
	uint32_t                extent    = 0;
	Vec3                    origin {};
	Vec3                    direction {};
	Vec3                    inverse {};
};

// 64 bytes at the device address of a 64-byte aligned guest address (a node or half a box32 node),
// from one BDA page lookup: 64-byte aligned blocks never cross a page. An unmapped page (device
// address 0; the lookup recorded a fault) reads as zeros, like every other guest address load.
std::array<uint32_t, 16> BvhLowering::LoadBlock(uint32_t device) {
	const auto present      = Op(spv::OpINotEqual, Bool(), device, Cu64(0));
	const auto array        = s.builder.Type(spv::OpTypeArray, U32x4(), Cu(4));
	const auto zero         = s.builder.Constant(spv::OpConstantNull, array);
	const auto pointer_type = TypePointer(s, spv::StorageClassPhysicalStorageBuffer, U32x4());
	const auto load         = [&](uint32_t alignment) {
		std::array<uint32_t, 4> vectors {};
		for (uint32_t index = 0; index < 4u; index++) {
			const auto address =
			    index == 0u ? device : Op(spv::OpIAdd, U64(), device, Cu64(index * 16u));
			const auto pointer = Op1(spv::OpConvertUToPtr, pointer_type, address);
			vectors[index]     = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpLoad, U32x4(), vectors[index], pointer,
			                      spv::MemoryAccessAlignedMask, alignment);
		}
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, array, result, vectors[0], vectors[1],
		                      vectors[2], vectors[3]);
		return result;
	};
	const auto               block = EmitValueOrDefaultIfCondition(s, present, array, zero, [&]() {
		// Buffer cache buffers start on guest pages, so a node keeps its 16-byte alignment in
		// device memory wherever the device aligns storage buffers to 16 bytes or more; 16-byte
		// loads are about 4x faster than 4-byte ones. The 4-byte path keeps any mapping correct.
		const auto aligned =
		    Op(spv::OpIEqual, Bool(), Op(spv::OpBitwiseAnd, U64(), device, Cu64(15u)), Cu64(0u));
		return IfElse(aligned, array, [&]() { return load(16u); }, [&]() { return load(4u); });
	});
	std::array<uint32_t, 16> dwords {};
	for (uint32_t index = 0; index < 16u; index++) {
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, U32(), result, block, index / 4u,
		                      index % 4u);
		dwords[index] = result;
	}
	return dwords;
}

// Correctly rounded f32 x / y (IEEE round-to-nearest-even), computed exactly with integers so the
// result does not depend on the host's division precision. Denormal inputs count as zero and
// denormal results flush to signed zero when the module flushes f32 denormals, as the float
// operations around it do.
uint32_t BvhLowering::ExactDiv(uint32_t x_f32, uint32_t y_f32) {
	const auto x = ToU32(x_f32);
	const auto y = ToU32(y_f32);
	const auto sign =
	    Op(spv::OpBitwiseAnd, U32(), Op(spv::OpBitwiseXor, U32(), x, y), Cu(0x80000000u));
	const auto ax         = Op(spv::OpBitwiseAnd, U32(), x, Cu(0x7fffffffu));
	const auto ay         = Op(spv::OpBitwiseAnd, U32(), y, Cu(0x7fffffffu));
	const auto x_nan      = Op(spv::OpUGreaterThan, Bool(), ax, Cu(PlusInfBits));
	const auto y_nan      = Op(spv::OpUGreaterThan, Bool(), ay, Cu(PlusInfBits));
	const auto x_inf      = Op(spv::OpIEqual, Bool(), ax, Cu(PlusInfBits));
	const auto y_inf      = Op(spv::OpIEqual, Bool(), ay, Cu(PlusInfBits));
	const auto zero_limit = Cu(flush_f32 ? 0x00800000u : 1u);
	const auto x_zero     = Op(spv::OpULessThan, Bool(), ax, zero_limit);
	const auto y_zero     = Op(spv::OpULessThan, Bool(), ay, zero_limit);
	const auto nan        = Or(Or(x_nan, y_nan), Or(And(x_zero, y_zero), And(x_inf, y_inf)));
	const auto inf        = Or(x_inf, y_zero);
	const auto zero       = Or(x_zero, y_inf);
	const auto special    = Or(Or(nan, inf), zero);

	// Significands in [2^23, 2^24) and effective biased exponents (a denormal is normalized).
	const auto normalize = [&](uint32_t a) {
		const auto exponent = Op(spv::OpShiftRightLogical, U32(), a, Cu(23u));
		const auto fraction = Op(spv::OpBitwiseAnd, U32(), a, Cu(0x7fffffu));
		const auto denormal = Op(spv::OpIEqual, Bool(), exponent, Cu(0u));
		const auto msb      = EmitGlsl<GLSLstd450FindUMsb, IR::Type::U32>(s, fraction);
		const auto shift    = Op(spv::OpISub, U32(), Cu(23u), msb);
		const auto mant = Sel(U32(), denormal, Op(spv::OpShiftLeftLogical, U32(), fraction, shift),
		                      Op(spv::OpBitwiseOr, U32(), fraction, Cu(0x800000u)));
		const auto exp  = Sel(U32(), denormal, Op(spv::OpISub, U32(), Cu(1u), shift), exponent);
		return std::pair {mant, exp};
	};
	const auto [mx, ex]     = normalize(ax);
	const auto [my_raw, ey] = normalize(ay);
	const auto my           = Sel(U32(), special, Cu(0x800000u), my_raw);
	const auto num = Op(spv::OpShiftLeftLogical, U64(), Op1(spv::OpUConvert, U64(), mx), Cu64(40));
	const auto den = Op1(spv::OpUConvert, U64(), my);
	const auto q   = Op(spv::OpUDiv, U64(), num, den);
	const auto r   = Op(spv::OpISub, U64(), num, Op(spv::OpIMul, U64(), q, den));
	// mx/my is in (1/2, 2), so q is in (2^39, 2^41).
	const auto big   = Op(spv::OpUGreaterThanEqual, Bool(), q, Cu64(uint64_t {1} << 40u));
	const auto shift = Sel(U32(), big, Cu(17u), Cu(16u));
	const auto biased =
	    Op(spv::OpISub, U32(), Op(spv::OpIAdd, U32(), Op(spv::OpISub, U32(), ex, ey), Cu(127u)),
	       Sel(U32(), big, Cu(0u), Cu(1u)));
	const auto subnormal = Op(spv::OpSLessThanEqual, Bool(), biased, Cu(0u));
	const auto extra     = Sel(U32(), subnormal, Op(spv::OpISub, U32(), Cu(1u), biased), Cu(0u));
	const auto total     = Op(spv::OpIAdd, U32(), shift, extra);
	const auto clamp = Sel(U32(), Op(spv::OpUGreaterThan, Bool(), total, Cu(62u)), Cu(62u), total);
	const auto shift64 = Op1(spv::OpUConvert, U64(), clamp);
	const auto kept    = Op(spv::OpShiftRightLogical, U64(), q, shift64);
	const auto one     = Cu64(1);
	const auto rem =
	    Op(spv::OpBitwiseAnd, U64(), q,
	       Op(spv::OpISub, U64(), Op(spv::OpShiftLeftLogical, U64(), one, shift64), one));
	const auto half = Op(spv::OpShiftLeftLogical, U64(), one, Op(spv::OpISub, U64(), shift64, one));
	const auto odd = Op(spv::OpINotEqual, Bool(), Op(spv::OpBitwiseAnd, U64(), kept, one), Cu64(0));
	const auto sticky = Op(spv::OpINotEqual, Bool(), r, Cu64(0));
	const auto up     = Or(Op(spv::OpUGreaterThan, Bool(), rem, half),
	                       And(Op(spv::OpIEqual, Bool(), rem, half), Or(sticky, odd)));
	const auto rounded =
	    Op1(spv::OpUConvert, U32(), Op(spv::OpIAdd, U64(), kept, Sel(U64(), up, one, Cu64(0))));
	// Normal result: rounding may carry into the next binade, or overflow to infinity.
	const auto carry    = Op(spv::OpIEqual, Bool(), rounded, Cu(0x1000000u));
	const auto mant_n   = Sel(U32(), carry, Cu(0x800000u), rounded);
	const auto exp_n    = Op(spv::OpIAdd, U32(), biased, Sel(U32(), carry, Cu(1u), Cu(0u)));
	const auto overflow = Op(spv::OpSGreaterThanEqual, Bool(), exp_n, Cu(255u));
	const auto normal_bits =
	    Sel(U32(), overflow, Cu(PlusInfBits),
	        Op(spv::OpBitwiseOr, U32(), Op(spv::OpShiftLeftLogical, U32(), exp_n, Cu(23u)),
	           Op(spv::OpBitwiseAnd, U32(), mant_n, Cu(0x7fffffu))));
	// Subnormal result: the significand already carries the exponent (2^23 is the smallest
	// normal); it flushes to zero under a flushing float mode.
	auto subnormal_bits = rounded;
	if (flush_f32) {
		subnormal_bits =
		    Sel(U32(), Op(spv::OpULessThan, Bool(), rounded, Cu(0x800000u)), Cu(0u), rounded);
	}
	const auto magnitude = Sel(U32(), subnormal, subnormal_bits, normal_bits);
	const auto finite    = Op(spv::OpBitwiseOr, U32(), sign, magnitude);
	const auto result    = Sel(U32(), nan, Cu(QuietNanBits),
	                           Sel(U32(), inf, Op(spv::OpBitwiseOr, U32(), sign, Cu(PlusInfBits)),
	                               Sel(U32(), zero, sign, finite)));
	return ToF32(result);
}

// GPURT fast_intersect_bbox per child, then IntersectNodeBvh4's hit test and sorting network.
uint32_t BvhLowering::BoxResult(const std::array<uint32_t, 4>& children,
                                const std::array<Vec3, 4>& lo, const std::array<Vec3, 4>& hi) {
	const auto grow = Op(spv::OpBitwiseAnd, U32(),
	                     Op(spv::OpShiftRightLogical, U32(), tsharp[1], Cu(23u)), Cu(0xffu));
	const auto factor =
	    FAdd(Cf(OneBits), FMul(Op1(spv::OpConvertUToF, F32(), grow), Cf(TwoPowMinus24Bits)));
	std::array<uint32_t, 4> pointers {};
	std::array<uint32_t, 4> keys {};
	for (uint32_t child = 0; child < 4u; child++) {
		Vec3 near_t {};
		Vec3 far_t {};
		for (uint32_t axis = 0; axis < 3u; axis++) {
			const auto plane_lo = FMul(FSub(lo[child][axis], origin[axis]), inverse[axis]);
			const auto plane_hi = FMul(FSub(hi[child][axis], origin[axis]), inverse[axis]);
			// NaN inverse direction takes the second arm, as GPURT's `inv >= 0 ? a : b`.
			const auto forward =
			    Op(spv::OpFOrdGreaterThanEqual, Bool(), inverse[axis], Cf(ZeroBits));
			near_t[axis] = Sel(F32(), forward, plane_lo, plane_hi);
			far_t[axis]  = Sel(F32(), forward, plane_hi, plane_lo);
		}
		const auto near3   = MaxNum(MaxNum(near_t[0], near_t[1]), near_t[2]);
		const auto far3    = MinNum(MinNum(far_t[0], far_t[1]), far_t[2]);
		const auto invalid = Or(IsNan(near3), IsNan(far3));
		const auto min_t   = Sel(F32(), invalid, Cf(PlusInfBits), MaxNum(near3, Cf(ZeroBits)));
		const auto max_t   = Sel(F32(), invalid, Cf(MinusInfBits), MinNum(far3, extent));
		const auto hit     = Op(spv::OpFOrdLessThanEqual, Bool(), min_t, FMul(max_t, factor));
		pointers[child]    = Sel(U32(), hit, children[child], Cu(InvalidNode));
		keys[child]        = min_t;
	}
	auto sorted_pointers = pointers;
	auto sorted_keys     = keys;
	// SORT(a, b): swap when b is a hit nearer than a, or a is empty.
	for (const auto& [a, b]: std::array<std::pair<uint32_t, uint32_t>, 5> {
	         {{0u, 2u}, {1u, 3u}, {0u, 1u}, {2u, 3u}, {1u, 2u}}}) {
		const auto b_valid = Op(spv::OpINotEqual, Bool(), sorted_pointers[b], Cu(InvalidNode));
		const auto a_empty = Op(spv::OpIEqual, Bool(), sorted_pointers[a], Cu(InvalidNode));
		const auto nearer  = Op(spv::OpFOrdLessThan, Bool(), sorted_keys[b], sorted_keys[a]);
		const auto swap    = Or(And(b_valid, nearer), a_empty);
		const auto pa      = sorted_pointers[a];
		const auto pb      = sorted_pointers[b];
		const auto ka      = sorted_keys[a];
		const auto kb      = sorted_keys[b];
		sorted_pointers[a] = Sel(U32(), swap, pb, pa);
		sorted_pointers[b] = Sel(U32(), swap, pa, pb);
		sorted_keys[a]     = Sel(F32(), swap, kb, ka);
		sorted_keys[b]     = Sel(F32(), swap, ka, kb);
	}
	const auto sort = Op(spv::OpINotEqual, Bool(),
	                     Op(spv::OpBitwiseAnd, U32(), tsharp[1], Cu(0x80000000u)), Cu(0u));
	std::array<uint32_t, 4> out {};
	for (uint32_t child = 0; child < 4u; child++) {
		out[child] = Sel(U32(), sort, sorted_pointers[child], pointers[child]);
	}
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, U32x4(), result, out[0], out[1], out[2],
	                      out[3]);
	return result;
}

// box16: child pointers in dwords 0-3; child i's f16 bounds in dwords 4+3i..6+3i as
// {min.x, min.y}, {min.z, max.x}, {max.y, max.z} (low half first).
uint32_t BvhLowering::Box16(uint32_t address) {
	const auto              dwords = LoadBlock(EmitBdaDeviceAddress(ctx, address));
	std::array<uint32_t, 4> children {dwords[0], dwords[1], dwords[2], dwords[3]};
	std::array<Vec3, 4>     lo {};
	std::array<Vec3, 4>     hi {};
	for (uint32_t child = 0; child < 4u; child++) {
		std::array<uint32_t, 6> halves {};
		for (uint32_t pair = 0; pair < 3u; pair++) {
			const auto unpacked = s.builder.AllocateId();
			s.builder.AddFunction(spv::OpExtInst, TypeF32Vector(s, 2), unpacked, GlslStd450(s),
			                      GLSLstd450UnpackHalf2x16, dwords[4u + child * 3u + pair]);
			halves[pair * 2u]      = Extract(F32(), unpacked, 0u);
			halves[pair * 2u + 1u] = Extract(F32(), unpacked, 1u);
		}
		lo[child] = {halves[0], halves[1], halves[2]};
		hi[child] = {halves[3], halves[4], halves[5]};
	}
	return BoxResult(children, lo, hi);
}

// box32 (128 bytes): child pointers in dwords 0-3; child i's f32 bounds in dwords 4+6i..9+6i.
uint32_t BvhLowering::Box32(uint32_t address) {
	const auto device = EmitBdaDeviceAddress(ctx, address);
	// The second half is on the same page unless the node starts 64 bytes before a page end.
	const auto page_end = Op(spv::OpIEqual, Bool(),
	                         Op(spv::OpBitwiseAnd, U64(), address, Cu64(BufferCachePageMask - 63u)),
	                         Cu64(BufferCachePageMask - 63u));
	const auto device_high = IfElse(
	    page_end, U64(),
	    [&]() { return EmitBdaDeviceAddress(ctx, Op(spv::OpIAdd, U64(), address, Cu64(64))); },
	    [&]() {
		    return Sel(U64(), Op(spv::OpINotEqual, Bool(), device, Cu64(0)),
		               Op(spv::OpIAdd, U64(), device, Cu64(64)), Cu64(0));
	    });
	const auto first  = LoadBlock(device);
	const auto second = LoadBlock(device_high);
	const auto dword  = [&](uint32_t index) {
		return ToF32(index < 16u ? first[index] : second[index - 16u]);
	};
	std::array<uint32_t, 4> children {first[0], first[1], first[2], first[3]};
	std::array<Vec3, 4>     lo {};
	std::array<Vec3, 4>     hi {};
	for (uint32_t child = 0; child < 4u; child++) {
		const auto base = 4u + child * 6u;
		lo[child]       = {dword(base), dword(base + 1u), dword(base + 2u)};
		hi[child]       = {dword(base + 3u), dword(base + 4u), dword(base + 5u)};
	}
	return BoxResult(children, lo, hi);
}

// One bound of a PS5 type-6 node: an 18-bit sign-magnitude `field` (bit 17 = sign) on the fixed
// grid of the axis' shared biased `exponent` (magnitude bit 16 weighs 2^(exponent - 127)).
// Renormalized to f32; the truncated low mantissa bits are filled with ones when the bound rounds
// away from zero (minima towards -inf, maxima towards +inf), so the box only grows. A zero
// magnitude rounding away becomes one grid step (just below 2^(exponent - 143)). Psr's decoder,
// eboot kernel 0x8ffdb60 (RT-HW, psrBvh.cpp DecodeSharedExpBound). Integer operations only.
uint32_t BvhLowering::SharedExpBound(uint32_t field, uint32_t exponent, bool is_min) {
	const auto sign      = Op(spv::OpShiftRightLogical, U32(), field, Cu(17u));
	const auto magnitude = Op(spv::OpBitwiseAnd, U32(), field, Cu(0x1ffffu));
	const auto negative  = Op(spv::OpINotEqual, Bool(), sign, Cu(0u));
	const auto away      = is_min ? negative : Not(negative);
	// Nonzero magnitude: shift its leading one to the implicit bit, lowering the exponent (not
	// below zero: then the result is denormal).
	const auto msb   = EmitGlsl<GLSLstd450FindUMsb, IR::Type::U32>(s, magnitude);
	const auto lead  = Op(spv::OpISub, U32(), Cu(16u), msb);
	const auto down  = Sel(U32(), Op(spv::OpULessThan, Bool(), lead, exponent), lead, exponent);
	const auto exp_n = Op(spv::OpISub, U32(), exponent, down);
	const auto shift = Op(spv::OpIAdd, U32(), Cu(7u), down);
	const auto fill  = Sel(
	    U32(), And(away, Op(spv::OpINotEqual, Bool(), exp_n, Cu(0xffu))),
	    Op(spv::OpISub, U32(), Op(spv::OpShiftLeftLogical, U32(), Cu(1u), shift), Cu(1u)), Cu(0u));
	const auto mant_n =
	    Op(spv::OpBitwiseAnd, U32(),
	       Op(spv::OpBitwiseOr, U32(), Op(spv::OpShiftLeftLogical, U32(), magnitude, shift), fill),
	       Cu(0x7fffffu));
	// Zero magnitude (FindUMsb gives -1, so down = min(17, exponent)): rounding away yields
	// mantissa 0x7fffff at exponent max(exponent - 17, 0), which is exp_n; otherwise a signed zero.
	const auto zero = Op(spv::OpIEqual, Bool(), magnitude, Cu(0u));
	const auto mant = Sel(U32(), And(zero, away), Cu(0x7fffffu), mant_n);
	const auto exp  = Sel(U32(), And(zero, Not(away)), Cu(0u), exp_n);
	return ToF32(Op(spv::OpBitwiseOr, U32(),
	                Op(spv::OpBitwiseOr, U32(), Op(spv::OpShiftLeftLogical, U32(), sign, Cu(31u)),
	                   Op(spv::OpShiftLeftLogical, U32(), exp, Cu(23u))),
	                mant));
}

// PS5 type-6 box node (64 bytes, Psr's "shared exponent" node; spec 4.4), in node bits:
// 0-28 child base (64-byte units), 29+3c child c's type, 41+2c child c's field (3 = empty),
// 56/64/72 the x/y/z shared exponents, 80+18(6c+k) bound k (min xyz, max xyz) of child c.
// Child pointers are implicit: ((base + units of the earlier children) << 3) | type, where a
// child of type 2 spans 4 units, of type 3, 5 or 6 spans 2, any other 1 (empty ones included).
uint32_t BvhLowering::BoxPs5(uint32_t address) {
	const auto              dwords = LoadBlock(EmitBdaDeviceAddress(ctx, address));
	std::array<uint32_t, 3> exponent {};
	for (uint32_t axis = 0; axis < 3u; axis++) {
		exponent[axis] = NodeBits(dwords, 56u + 8u * axis, 8u);
	}
	auto                    running = Op(spv::OpBitwiseAnd, U32(), dwords[0], Cu(0x1fffffffu));
	std::array<uint32_t, 4> children {};
	std::array<Vec3, 4>     lo {};
	std::array<Vec3, 4>     hi {};
	for (uint32_t child = 0; child < 4u; child++) {
		const auto type = NodeBits(dwords, 29u + 3u * child, 3u);
		const auto empty =
		    Op(spv::OpIEqual, Bool(), NodeBits(dwords, 41u + 2u * child, 2u), Cu(3u));
		children[child] = Sel(
		    U32(), empty, Cu(InvalidNode),
		    Op(spv::OpBitwiseOr, U32(), Op(spv::OpShiftLeftLogical, U32(), running, Cu(3u)), type));
		const auto two_units =
		    Or(Or(Op(spv::OpIEqual, Bool(), type, Cu(3u)), Op(spv::OpIEqual, Bool(), type, Cu(5u))),
		       Op(spv::OpIEqual, Bool(), type, Cu(6u)));
		const auto units = Sel(U32(), Op(spv::OpIEqual, Bool(), type, Cu(2u)), Cu(4u),
		                       Sel(U32(), two_units, Cu(2u), Cu(1u)));
		running          = Op(spv::OpIAdd, U32(), running, units);
		for (uint32_t k = 0; k < 6u; k++) {
			const auto field                  = NodeBits(dwords, 80u + 18u * (6u * child + k), 18u);
			const auto value                  = SharedExpBound(field, exponent[k % 3u], k < 3u);
			(k < 3u ? lo : hi)[child][k % 3u] = value;
		}
	}
	return BoxResult(children, lo, hi);
}

// Triangle node (64 bytes): v0..v4 at 0/12/24/36/48, triangle ID at 60. The pointer type picks
// one of four triangles sharing v2: 0 = (v0, v1, v2), 1 = (v1, v3, v2), 2 = (v2, v3, v4),
// 3 = (v2, v4, v0) (GPURT CalcTriangleVertexOffsets for 0/1; Astro Bot's GI hit-normal code for
// all four). GPURT fast_intersect_triangle + SwizzleBarycentrics.
uint32_t BvhLowering::Triangle(uint32_t address) {
	const auto dwords = LoadBlock(EmitBdaDeviceAddress(ctx, address));
	const auto is1    = Op(spv::OpIEqual, Bool(), node_type, Cu(1u));
	const auto is2    = Op(spv::OpIEqual, Bool(), node_type, Cu(2u));
	const auto is3    = Op(spv::OpIEqual, Bool(), node_type, Cu(3u));
	Vec3       a {};
	Vec3       b {};
	Vec3       c {};
	for (uint32_t axis = 0; axis < 3u; axis++) {
		std::array<uint32_t, 5> v {};
		for (uint32_t vertex = 0; vertex < 5u; vertex++) {
			v[vertex] = ToF32(dwords[vertex * 3u + axis]);
		}
		const auto pick = [&](uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
			return Sel(F32(), is3, v[t3], Sel(F32(), is2, v[t2], Sel(F32(), is1, v[t1], v[t0])));
		};
		a[axis] = pick(0u, 1u, 2u, 2u);
		b[axis] = pick(1u, 3u, 3u, 4u);
		c[axis] = pick(2u, 2u, 4u, 0u);
	}
	const auto e1      = Sub3(b, a);
	const auto e2      = Sub3(c, a);
	const auto e3      = Sub3(origin, a);
	const auto s1      = Cross(direction, e2);
	const auto s2      = Cross(e3, e1);
	const auto t_num   = Dot(e2, s2);
	const auto t_den   = Dot(s1, e1);
	const auto i_num   = Dot(e3, s1);
	const auto j_num   = Dot(direction, s2);
	const auto zero    = Cf(ZeroBits);
	const auto one     = Cf(OneBits);
	const auto miss_of = [&](uint32_t t, uint32_t u, uint32_t v, uint32_t u_plus_v) {
		return Or(Or(Or(Op(spv::OpFOrdLessThan, Bool(), u, zero),
		                Op(spv::OpFOrdGreaterThan, Bool(), u, one)),
		             Or(Op(spv::OpFOrdLessThan, Bool(), v, zero),
		                Op(spv::OpFOrdGreaterThan, Bool(), u_plus_v, one))),
		          Op(spv::OpFOrdLessThan, Bool(), t, zero));
	};
	// The miss decision needs only the signs of t, u and v and where u and u + v lie against 1. Far
	// from those thresholds, quotients through the host's reciprocal decide exactly as the
	// correctly rounded ones (RT-HW's PsrTriangleMiss): Vulkan bounds OpFDiv by 2.5 ulp for a
	// divisor in [2^-126, 2^126], so num * (1 / t_den) is within about 3.5 ulp of the correctly
	// rounded quotient, 2^-21.2 relative. The decision is taken from them when:
	// - |t_den| is a normal in [2^-126, 2^126] and every approximation is finite;
	// - |t|, |u|, |v| > 2^-100, so none of the correctly rounded quotients is a zero;
	// - |u - 1| > 2^-18 and |u + v - 1| > 2^-18 (1 + |u| + |v|).
	// Otherwise the exact integer divisions decide; near the thresholds that is rare.
	const auto abs_bits = [&](uint32_t value) {
		return Op(spv::OpBitwiseAnd, U32(), ToU32(value), Cu(0x7fffffffu));
	};
	const auto finite = [&](uint32_t value) {
		return Op(spv::OpULessThan, Bool(), abs_bits(value), Cu(PlusInfBits));
	};
	const auto away_from_zero = [&](uint32_t value) {
		return Op(spv::OpUGreaterThan, Bool(), abs_bits(value), Cu(0x0d800000u)); // 2^-100
	};
	const auto fabs = [&](uint32_t value) {
		return EmitGlsl<GLSLstd450FAbs, IR::Type::F32>(s, value);
	};
	const auto margin     = Cf(0x36800000u); // 2^-18
	const auto den_bits   = abs_bits(t_den);
	const auto den_ok     = And(Op(spv::OpUGreaterThanEqual, Bool(), den_bits, Cu(0x00800000u)),
	                            Op(spv::OpULessThanEqual, Bool(), den_bits, Cu(0x7e800000u)));
	const auto reciprocal = Op(spv::OpFDiv, F32(), one, t_den);
	const auto t_approx   = FMul(t_num, reciprocal);
	const auto u_approx   = FMul(i_num, reciprocal);
	const auto v_approx   = FMul(j_num, reciprocal);
	const auto sum_approx = FAdd(u_approx, v_approx);
	const auto u_clear    = Op(spv::OpFOrdGreaterThan, Bool(), fabs(FSub(u_approx, one)), margin);
	const auto sum_clear  = Op(spv::OpFOrdGreaterThan, Bool(), fabs(FSub(sum_approx, one)),
	                           FMul(margin, FAdd(FAdd(one, fabs(u_approx)), fabs(v_approx))));
	const auto certain    = And(
	    And(And(den_ok, And(And(finite(t_approx), finite(u_approx)),
	                        And(finite(v_approx), finite(sum_approx)))),
	        And(And(away_from_zero(t_approx), away_from_zero(u_approx)), away_from_zero(v_approx))),
	    And(u_clear, sum_clear));
	const auto miss = IfElse(
	    certain, Bool(), [&]() { return miss_of(t_approx, u_approx, v_approx, sum_approx); },
	    [&]() {
		    const auto t = ExactDiv(t_num, t_den);
		    const auto u = ExactDiv(i_num, t_den);
		    const auto v = ExactDiv(j_num, t_den);
		    return miss_of(t, u, v, FAdd(u, v));
	    });
	const auto                    result_t   = Sel(F32(), miss, Cf(PlusInfBits), t_num);
	const auto                    result_den = Sel(F32(), miss, one, t_den);
	const std::array<uint32_t, 3> barycentric {FSub(FSub(result_den, i_num), j_num), i_num, j_num};
	const auto                    triangle_id = dwords[15];
	const auto                    shift = Op(spv::OpShiftLeftLogical, U32(), node_type, Cu(3u));
	const auto                    pick  = [&](uint32_t field_shift) {
		const auto selector = Op(spv::OpBitwiseAnd, U32(),
		                         Op(spv::OpShiftRightLogical, U32(), triangle_id,
		                            Op(spv::OpIAdd, U32(), shift, Cu(field_shift))),
		                         Cu(3u));
		auto       value    = zero; // index 3 selects nothing
		for (uint32_t index = 0; index < 3u; index++) {
			value = Sel(F32(), Op(spv::OpIEqual, Bool(), selector, Cu(index)), barycentric[index],
			            value);
		}
		return ToU32(value);
	};
	const auto mode_ij = Op(spv::OpINotEqual, Bool(),
	                        Op(spv::OpBitwiseAnd, U32(), tsharp[3], Cu(1u << 24u)), Cu(0u));
	const auto dword2  = Sel(U32(), mode_ij, pick(0u), triangle_id);
	const auto dword3  = Sel(U32(), mode_ij, pick(2u), Sel(U32(), miss, Cu(0u), Cu(1u)));
	const auto result  = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpCompositeConstruct, U32x4(), result, ToU32(result_t),
	                      ToU32(result_den), dword2, dword3);
	return result;
}

uint32_t BvhLowering::NodeTest() {
	node_type       = Op(spv::OpBitwiseAnd, U32(), node_lo, Cu(7u));
	const auto node = U64Of(node_lo, node_hi);
	// Node address = T# base (bits 39:0 = address bits 47:8) + (pointer & ~7) << 3.
	const auto base =
	    Op(spv::OpBitwiseOr, U64(),
	       Op(spv::OpShiftLeftLogical, U64(), Op1(spv::OpUConvert, U64(), tsharp[0]), Cu64(8)),
	       Op(spv::OpShiftLeftLogical, U64(),
	          Op1(spv::OpUConvert, U64(), Op(spv::OpBitwiseAnd, U32(), tsharp[1], Cu(0xffu))),
	          Cu64(40)));
	const auto address = Op(spv::OpIAdd, U64(), base,
	                        Op(spv::OpShiftLeftLogical, U64(),
	                           Op(spv::OpBitwiseAnd, U64(), node, Cu64(~uint64_t {7})), Cu64(3)));
	// Bounds check against the T# size (number of 64-byte nodes minus 1).
	const auto index    = Op(spv::OpShiftRightLogical, U64(), node, Cu64(3));
	const auto size     = U64Of(tsharp[2], Op(spv::OpBitwiseAnd, U32(), tsharp[3], Cu(0x3ffu)));
	const auto in_range = Op(spv::OpULessThanEqual, Bool(), index, size);
	const auto triangle = Op(spv::OpULessThan, Bool(), node_type, Cu(4u));
	const auto box16    = Op(spv::OpIEqual, Bool(), node_type, Cu(4u));
	const auto box32    = Op(spv::OpIEqual, Bool(), node_type, Cu(5u));
	// Type 6 is the PS5's shared-exponent box; KYTY_RT_TYPE6=0 misses it like RDNA2 (GPURT).
	const auto box_ps5 = GetCodegenOptions().rt_type6 ? Op(spv::OpIEqual, Bool(), node_type, Cu(6u))
	                                                  : ConstantBool(s, false);
	const auto miss    = [&]() {
		const auto t      = Sel(U32(), triangle, Cu(PlusInfBits), Cu(InvalidNode));
		const auto den    = Sel(U32(), triangle, Cu(OneBits), Cu(InvalidNode));
		const auto rest   = Sel(U32(), triangle, Cu(0u), Cu(InvalidNode));
		const auto result = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeConstruct, U32x4(), result, t, den, rest, rest);
		return result;
	};
	return IfElse(
	    And(in_range, triangle), U32x4(), [&]() { return Triangle(address); },
	    [&]() {
		    return IfElse(
		        And(in_range, box16), U32x4(), [&]() { return Box16(address); },
		        [&]() {
			        return IfElse(
			            And(in_range, box32), U32x4(), [&]() { return Box32(address); },
			            [&]() {
				            return IfElse(
				                And(in_range, box_ps5), U32x4(), [&]() { return BoxPs5(address); },
				                miss);
			            });
		        });
	    });
}

} // namespace

uint32_t EmitBvhNodeTest(ValueEmitContext& ctx, const BvhOperands& operands) {
	return BvhLowering(ctx, operands).NodeTest();
}

namespace {

// The node test at one instruction: a call of the module's function (DefineBvhNodeTestFunction), or
// the whole lowering inline (KYTY_RT_FUNCTION=0).
uint32_t EmitBvhNodeTestSite(ValueEmitContext& ctx, const BvhOperands& operands) {
	auto& s = ctx.state;
	if (s.bvh_node_test_function == 0) {
		return EmitBvhNodeTest(ctx, operands);
	}
	std::vector<uint32_t> arguments(operands.tsharp.begin(), operands.tsharp.end());
	arguments.push_back(operands.node_lo);
	arguments.push_back(operands.node_hi);
	arguments.push_back(operands.extent);
	arguments.insert(arguments.end(), operands.origin.begin(), operands.origin.end());
	arguments.insert(arguments.end(), operands.direction.begin(), operands.direction.end());
	arguments.insert(arguments.end(), operands.inverse.begin(), operands.inverse.end());
	const auto result = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpFunctionCall, TypeU32Vector(s, 4), result,
	                      s.bvh_node_test_function, std::span<const uint32_t>(arguments));
	return result;
}

bool HasSoftwareBvh(const EmitterState& state) {
	for (const auto* block: state.program.blocks) {
		for (const auto& inst: *block) {
			if (inst.GetOpcode() == IR::ValueOpcode::BvhIntersectRay) {
				return true;
			}
		}
	}
	return false;
}

} // namespace

void DefineBvhNodeTestFunction(EmitterState& state) {
	const auto& options = GetCodegenOptions();
	const auto  mode    = RtFunctionMode();
	if (mode == 0 || !options.rt_software || !state.program.info.uses_dma ||
	    !HasSoftwareBvh(state)) {
		return;
	}
	const auto u32    = TypeU32(state);
	const auto f32    = TypeF32(state);
	const auto result = TypeU32Vector(state, 4);
	// tsharp[4], node_lo, node_hi, extent, origin[3], direction[3], inverse[3]
	std::vector<uint32_t> parameter_types(6u, u32);
	parameter_types.insert(parameter_types.end(), 10u, f32);
	const auto function_type =
	    state.builder.Type(spv::OpTypeFunction, result, std::span<const uint32_t>(parameter_types));
	state.bvh_node_test_function = state.builder.AllocateId();
	state.builder.AddName(state.bvh_node_test_function, "bvh_node_test");
	state.builder.AddFunction(spv::OpFunction, result, state.bvh_node_test_function,
	                          mode >= 2u ? spv::FunctionControlDontInlineMask
	                                                    : spv::FunctionControlMaskNone,
	                          function_type);
	std::vector<uint32_t> parameters;
	for (const auto type: parameter_types) {
		const auto id = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpFunctionParameter, type, id);
		parameters.push_back(id);
	}
	EmitLabel(state, state.builder.AllocateId());
	BvhOperands operands;
	size_t      next = 0;
	for (auto& dword: operands.tsharp) {
		dword = parameters[next++];
	}
	operands.node_lo = parameters[next++];
	operands.node_hi = parameters[next++];
	operands.extent  = parameters[next++];
	for (auto& value: operands.origin) {
		value = parameters[next++];
	}
	for (auto& value: operands.direction) {
		value = parameters[next++];
	}
	for (auto& value: operands.inverse) {
		value = parameters[next++];
	}
	ValueEmitContext ctx(state);
	const auto       value = EmitBvhNodeTest(ctx, operands);
	state.builder.AddFunction(spv::OpReturnValue, value);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

uint32_t EmitBvhIntersectRay(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto& s = ctx.state;
	// Every operand is read before the first branch.
	BvhOperands operands;
	const auto  descriptor = ctx.Arg(inst, 1);
	for (uint32_t index = 0; index < 4u; index++) {
		operands.tsharp[index] = s.builder.AllocateId();
		s.builder.AddFunction(spv::OpCompositeExtract, TypeU32(s), operands.tsharp[index],
		                      descriptor, index);
	}
	const auto to_f32 = [&](uint32_t bits) { return Unary(s, spv::OpBitcast, TypeF32(s), bits); };
	operands.node_lo  = ctx.Arg(inst, 2);
	operands.node_hi  = ctx.Arg(inst, 3);
	operands.extent   = to_f32(ctx.Arg(inst, 4));
	for (uint32_t axis = 0; axis < 3u; axis++) {
		operands.origin[axis]    = to_f32(ctx.Arg(inst, 5u + axis));
		operands.direction[axis] = to_f32(ctx.Arg(inst, 8u + axis));
		operands.inverse[axis]   = to_f32(ctx.Arg(inst, 11u + axis));
	}
	const auto active = ctx.Arg(inst, 14);
	if (s.bvh_node_count_variable == 0) {
		return EmitValueOrDefaultIfCondition(s, active, TypeU32Vector(s, 4),
		                                     ConstantU32CompositeZero(s, 4),
		                                     [&]() { return EmitBvhNodeTestSite(ctx, operands); });
	}
	// KYTY_RT_NODE_BUDGET / KYTY_RT_NODE_STATS: every invocation counts every execution, whatever
	// EXEC holds (the wave's traversal length; a two-lane invocation counts once per half).
	const auto count = s.builder.AllocateId();
	s.builder.AddFunction(spv::OpLoad, TypeU32(s), count, s.bvh_node_count_variable);
	const auto saturated = Binary(s, spv::OpIEqual, TypeBool(s), count, ConstantU32(s, UINT32_MAX));
	s.builder.AddFunction(spv::OpStore, s.bvh_node_count_variable,
	                      Select(s, TypeU32(s), saturated, count,
	                             Binary(s, spv::OpIAdd, TypeU32(s), count, ConstantU32(s, 1u))));
	const uint64_t budget = GetCodegenOptions().rt_node_budget;
	if (budget == 0) {
		return EmitValueOrDefaultIfCondition(s, active, TypeU32Vector(s, 4),
		                                     ConstantU32CompositeZero(s, 4),
		                                     [&]() { return EmitBvhNodeTestSite(ctx, operands); });
	}
	// Past the budget a node test misses without reading memory: a box has no hit children and a
	// triangle is not hit, so the guest's traversal stack only drains.
	const auto limit =
	    static_cast<uint32_t>(std::min<uint64_t>(budget * s.lane_count, UINT32_MAX - 1u));
	const auto allowed = Binary(s, spv::OpULessThan, TypeBool(s), count, ConstantU32(s, limit));
	return EmitValueOrDefaultIfCondition(
	    s, active, TypeU32Vector(s, 4), ConstantU32CompositeZero(s, 4), [&]() {
		    return EmitValueOrDefaultIfCondition(s, allowed, TypeU32Vector(s, 4),
		                                         EmitBvhIntersectRayStub(s, operands.node_lo),
		                                         [&]() { return EmitBvhNodeTestSite(ctx, operands); });
	    });
}

// KYTY_RT_STUB: a triangle node (type 0-3) misses with t_num = +inf, t_denom = 1.0 and zero
// dwords 2-3 (a cleared hit_status in return mode 0); any other node returns four invalid child
// pointers.
uint32_t EmitBvhIntersectRayStub(EmitterState& state, uint32_t node_lo) {
	const auto type =
	    Binary(state, spv::OpBitwiseAnd, TypeU32(state), node_lo, ConstantU32(state, 7u));
	const auto triangle =
	    Binary(state, spv::OpULessThan, TypeBool(state), type, ConstantU32(state, 4u));
	const auto select = [&](uint32_t triangle_value) {
		return Select(state, TypeU32(state), triangle, ConstantU32(state, triangle_value),
		              ConstantU32(state, InvalidNode));
	};
	const auto result = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4), result,
	                          select(PlusInfBits), select(OneBits), select(0u), select(0u));
	return result;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
