#ifndef EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BVHREFERENCE_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_RECOMPILER_BVHREFERENCE_H_

// CPU reference of one IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY node test: the exact
// model that KYTY_RT_SOFTWARE lowers to SPIR-V (backend/spirv/spirvEmitterRayTracing.cpp), with
// the same operations in the same order. Header-only, standard library only, for tests and CPU
// tools (the GPU harness uses it as the oracle). Specification:
// Profiling/analysis/RT-SOFTWARE-DESIGN.md.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::BvhReference {

constexpr uint32_t InvalidNode = 0xffffffffu;
constexpr uint32_t PlusInf     = 0x7f800000u;
constexpr uint32_t MinusInf    = 0xff800000u;
constexpr uint32_t OneF        = 0x3f800000u;
// A triangle the ray misses (and an out-of-range triangle node); a box with no hit child.
constexpr std::array<uint32_t, 4> TriangleMiss {PlusInf, OneF, 0u, 0u};
constexpr std::array<uint32_t, 4> BoxMiss {InvalidNode, InvalidNode, InvalidNode, InvalidNode};

inline float F(uint32_t bits) {
	return std::bit_cast<float>(bits);
}
inline uint32_t B(float value) {
	return std::bit_cast<uint32_t>(value);
}

// Exact binary16 -> binary32 (box16 bounds, A16 ray operands).
inline float HalfToFloat(uint32_t half) {
	const uint32_t sign     = (half & 0x8000u) << 16u;
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	const uint32_t mantissa = half & 0x3ffu;
	if (exponent == 0u) {
		const float magnitude = static_cast<float>(mantissa) * 0x1p-24f;
		return F(B(magnitude) | sign);
	}
	if (exponent == 31u) {
		return F(sign | 0x7f800000u | (mantissa << 13u));
	}
	return F(sign | ((exponent - 15u + 127u) << 23u) | (mantissa << 13u));
}

// IEEE f32 with the host's float mode: with `flush`, denormal operands count as zero and denormal
// results become signed zero (Kyty declares DenormFlushToZero for f32 where the device supports
// it, HostFloatControls::denorm_flush_f32); without it, IEEE denormals.
struct Math {
	bool flush = true;

	[[nodiscard]] float Fl(float x) const {
		if (!flush) return x;
		const uint32_t bits = B(x);
		return (bits & 0x7f800000u) == 0u ? F(bits & 0x80000000u) : x;
	}
	// Separate statements: nothing may contract into an FMA.
	[[nodiscard]] float Add(float a, float b) const {
		const float r = Fl(a) + Fl(b);
		return Fl(r);
	}
	[[nodiscard]] float Sub(float a, float b) const {
		const float r = Fl(a) - Fl(b);
		return Fl(r);
	}
	[[nodiscard]] float Mul(float a, float b) const {
		const float r = Fl(a) * Fl(b);
		return Fl(r);
	}
	// Correctly rounded (x86 SSE division), then the float mode.
	[[nodiscard]] float Div(float a, float b) const {
		const float r = Fl(a) / Fl(b);
		return Fl(r);
	}
	[[nodiscard]] bool Lt(float a, float b) const { return Fl(a) < Fl(b); }
	[[nodiscard]] bool Gt(float a, float b) const { return Fl(a) > Fl(b); }
	[[nodiscard]] bool Le(float a, float b) const { return Fl(a) <= Fl(b); }
	[[nodiscard]] bool Ge(float a, float b) const { return Fl(a) >= Fl(b); }
	// maxNum/minNum: a NaN operand is ignored; a tie returns the second operand.
	[[nodiscard]] float MaxNum(float a, float b) const {
		if (std::isnan(a)) return b;
		return std::isnan(b) || Gt(a, b) ? a : b;
	}
	[[nodiscard]] float MinNum(float a, float b) const {
		if (std::isnan(a)) return b;
		return std::isnan(b) || Lt(a, b) ? a : b;
	}
};

struct Ray {
	float                extent = 0;
	std::array<float, 3> origin {};
	std::array<float, 3> dir {};
	std::array<float, 3> inv {};
};

// Dword at a 4-byte aligned guest address.
using Memory = std::function<uint32_t(uint64_t)>;
using Vec3   = std::array<float, 3>;

inline std::array<uint32_t, 4> BoxResult(const std::array<uint32_t, 4>& tsharp,
                                         const std::array<uint32_t, 4>& children,
                                         const std::array<Vec3, 4>&     lo,
                                         const std::array<Vec3, 4>& hi, const Ray& ray,
                                         const Math& m) {
	const uint32_t          grow   = (tsharp[1] >> 23u) & 0xffu;
	const float             factor = m.Add(1.0f, m.Mul(static_cast<float>(grow), 0x1p-24f));
	std::array<uint32_t, 4> pointers {};
	std::array<float, 4>    keys {};
	for (uint32_t child = 0; child < 4u; ++child) {
		Vec3 near_t {};
		Vec3 far_t {};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const float plane_lo = m.Mul(m.Sub(lo[child][axis], ray.origin[axis]), ray.inv[axis]);
			const float plane_hi = m.Mul(m.Sub(hi[child][axis], ray.origin[axis]), ray.inv[axis]);
			const bool  forward  = m.Ge(ray.inv[axis], 0.0f);
			near_t[axis]         = forward ? plane_lo : plane_hi;
			far_t[axis]          = forward ? plane_hi : plane_lo;
		}
		const float near3   = m.MaxNum(m.MaxNum(near_t[0], near_t[1]), near_t[2]);
		const float far3    = m.MinNum(m.MinNum(far_t[0], far_t[1]), far_t[2]);
		const bool  invalid = std::isnan(near3) || std::isnan(far3);
		const float min_t   = invalid ? F(PlusInf) : m.MaxNum(near3, 0.0f);
		const float max_t   = invalid ? F(MinusInf) : m.MinNum(far3, ray.extent);
		const bool  hit     = m.Le(min_t, m.Mul(max_t, factor));
		pointers[child]     = hit ? children[child] : InvalidNode;
		keys[child]         = min_t;
	}
	if ((tsharp[1] & 0x80000000u) == 0u) {
		return pointers;
	}
	constexpr std::array<std::pair<uint32_t, uint32_t>, 5> Network {
	    {{0u, 2u}, {1u, 3u}, {0u, 1u}, {2u, 3u}, {1u, 2u}}};
	for (const auto& [a, b]: Network) {
		const bool swap =
		    (pointers[b] != InvalidNode && m.Lt(keys[b], keys[a])) || pointers[a] == InvalidNode;
		if (swap) {
			std::swap(pointers[a], pointers[b]);
			std::swap(keys[a], keys[b]);
		}
	}
	return pointers;
}

inline std::array<uint32_t, 4> Triangle(const std::array<uint32_t, 4>& tsharp, uint32_t type,
                                        const std::array<uint32_t, 16>& node, const Ray& ray,
                                        const Math& m) {
	// Fan of four triangles sharing v2 (spec 4.3).
	constexpr std::array<std::array<uint32_t, 3>, 4> Fan {
	    {{0u, 1u, 2u}, {1u, 3u, 2u}, {2u, 3u, 4u}, {2u, 4u, 0u}}};
	const auto vertex = [&](uint32_t index) {
		return Vec3 {F(node[index * 3u]), F(node[index * 3u + 1u]), F(node[index * 3u + 2u])};
	};
	const Vec3 a   = vertex(Fan[type][0]);
	const Vec3 b   = vertex(Fan[type][1]);
	const Vec3 c   = vertex(Fan[type][2]);
	const auto sub = [&](const Vec3& p, const Vec3& q) {
		return Vec3 {m.Sub(p[0], q[0]), m.Sub(p[1], q[1]), m.Sub(p[2], q[2])};
	};
	const auto cross = [&](const Vec3& p, const Vec3& q) {
		return Vec3 {m.Sub(m.Mul(p[1], q[2]), m.Mul(p[2], q[1])),
		             m.Sub(m.Mul(p[2], q[0]), m.Mul(p[0], q[2])),
		             m.Sub(m.Mul(p[0], q[1]), m.Mul(p[1], q[0]))};
	};
	const auto dot = [&](const Vec3& p, const Vec3& q) {
		return m.Add(m.Add(m.Mul(p[0], q[0]), m.Mul(p[1], q[1])), m.Mul(p[2], q[2]));
	};
	const Vec3  e1    = sub(b, a);
	const Vec3  e2    = sub(c, a);
	const Vec3  e3    = sub(ray.origin, a);
	const Vec3  s1    = cross(ray.dir, e2);
	const Vec3  s2    = cross(e3, e1);
	const float t_num = dot(e2, s2);
	const float t_den = dot(s1, e1);
	const float i_num = dot(e3, s1);
	const float j_num = dot(ray.dir, s2);
	const float t     = m.Div(t_num, t_den);
	const float u     = m.Div(i_num, t_den);
	const float v     = m.Div(j_num, t_den);
	const bool  miss =
	    m.Lt(u, 0.0f) || m.Gt(u, 1.0f) || m.Lt(v, 0.0f) || m.Gt(m.Add(u, v), 1.0f) || m.Lt(t, 0.0f);
	const float                result_t   = miss ? F(PlusInf) : t_num;
	const float                result_den = miss ? 1.0f : t_den;
	const std::array<float, 3> barycentric {m.Sub(m.Sub(result_den, i_num), j_num), i_num, j_num};
	const uint32_t             triangle_id = node[15];
	const auto                 pick        = [&](uint32_t shift) {
		const uint32_t selector = (triangle_id >> (type * 8u + shift)) & 3u;
		return B(selector < 3u ? barycentric[selector] : 0.0f);
	};
	if ((tsharp[3] & (1u << 24u)) != 0u) {
		return {B(result_t), B(result_den), pick(0u), pick(2u)};
	}
	return {B(result_t), B(result_den), triangle_id, miss ? 0u : 1u};
}

// PS5 type-6 box (spec 4.4). Bits [first, first + count) of the node (count < 32).
inline uint32_t NodeBits(const std::array<uint32_t, 16>& node, uint32_t first, uint32_t count) {
	const uint32_t word   = first / 32u;
	uint64_t       window = node[word];
	if (word + 1u < 16u) window |= uint64_t {node[word + 1u]} << 32u;
	return static_cast<uint32_t>((window >> (first % 32u)) & ((uint64_t {1} << count) - 1u));
}

// 64-byte units a type-6 node's child spans, by its type.
inline uint32_t SharedExpUnits(uint32_t type) {
	return type == 2u ? 4u : (type == 3u || type == 5u || type == 6u) ? 2u : 1u;
}

// One bound: an 18-bit sign-magnitude field on the axis' shared-exponent grid, renormalized and
// rounded outwards (minima towards -inf, maxima towards +inf). Returns the f32 bits.
inline uint32_t SharedExpBound(uint32_t field, uint32_t exponent, bool is_min) {
	const uint32_t sign      = (field >> 17u) & 1u;
	const uint32_t magnitude = field & 0x1ffffu;
	const bool     away      = is_min ? sign != 0u : sign == 0u;
	uint32_t       exp       = 0;
	uint32_t       mantissa  = 0;
	if (magnitude != 0u) {
		const uint32_t msb   = static_cast<uint32_t>(std::bit_width(magnitude)) - 1u;
		const uint32_t down  = std::min(16u - msb, exponent);
		exp                  = exponent - down;
		const uint32_t shift = 7u + down;
		const uint32_t fill  = (away && exp != 0xffu) ? (1u << shift) - 1u : 0u;
		mantissa             = ((magnitude << shift) | fill) & 0x7fffffu;
	} else {
		mantissa = away ? 0x7fffffu : 0u;
		exp      = (away && exponent > 17u) ? exponent - 17u : 0u;
	}
	return (sign << 31u) | (exp << 23u) | mantissa;
}

struct DecodedBox {
	std::array<uint32_t, 4> children {};
	std::array<Vec3, 4>     lo {};
	std::array<Vec3, 4>     hi {};
};

inline DecodedBox DecodeBoxPs5(const std::array<uint32_t, 16>& node) {
	DecodedBox box;
	uint32_t   running = node[0] & 0x1fffffffu;
	for (uint32_t child = 0; child < 4u; ++child) {
		const uint32_t type  = NodeBits(node, 29u + 3u * child, 3u);
		const bool     empty = NodeBits(node, 41u + 2u * child, 2u) == 3u;
		box.children[child]  = empty ? InvalidNode : (running << 3u) | type;
		running += SharedExpUnits(type);
		for (uint32_t k = 0; k < 6u; ++k) {
			const uint32_t exponent = NodeBits(node, 56u + 8u * (k % 3u), 8u);
			const uint32_t field    = NodeBits(node, 80u + 18u * (6u * child + k), 18u);
			(k < 3u ? box.lo : box.hi)[child][k % 3u] = F(SharedExpBound(field, exponent, k < 3u));
		}
	}
	return box;
}

// `ps5_type6`: type 6 is the PS5 box (KYTY_RT_TYPE6, default); otherwise it misses like RDNA2.
inline std::array<uint32_t, 4> Intersect(const std::array<uint32_t, 4>& tsharp,
                                         uint64_t node_pointer, const Ray& ray,
                                         const Memory& memory, const Math& m,
                                         bool ps5_type6 = true) {
	const uint32_t type     = static_cast<uint32_t>(node_pointer & 7u);
	const uint64_t base     = (uint64_t {tsharp[0]} << 8u) | (uint64_t {tsharp[1] & 0xffu} << 40u);
	const uint64_t address  = base + ((node_pointer & ~uint64_t {7}) << 3u);
	const uint64_t index    = node_pointer >> 3u;
	const uint64_t size     = uint64_t {tsharp[2]} | (uint64_t {tsharp[3] & 0x3ffu} << 32u);
	const bool     triangle = type < 4u;
	const bool     box_ps5  = type == 6u && ps5_type6;
	if (index > size || (type > 5u && !box_ps5)) {
		return triangle ? TriangleMiss : BoxMiss;
	}
	const auto load = [&](uint64_t offset) {
		std::array<uint32_t, 16> dwords {};
		for (uint32_t i = 0; i < 16u; ++i)
			dwords[i] = memory(address + offset + i * 4u);
		return dwords;
	};
	if (triangle) {
		return Triangle(tsharp, type, load(0), ray, m);
	}
	if (box_ps5) {
		const auto box = DecodeBoxPs5(load(0));
		return BoxResult(tsharp, box.children, box.lo, box.hi, ray, m);
	}
	const auto              first = load(0);
	std::array<uint32_t, 4> children {first[0], first[1], first[2], first[3]};
	std::array<Vec3, 4>     lo {};
	std::array<Vec3, 4>     hi {};
	if (type == 4u) {
		for (uint32_t child = 0; child < 4u; ++child) {
			std::array<float, 6> halves {};
			for (uint32_t k = 0; k < 6u; ++k) {
				const uint32_t word = first[4u + child * 3u + k / 2u];
				halves[k] = HalfToFloat((k % 2u) == 0u ? (word & 0xffffu) : (word >> 16u));
			}
			lo[child] = {halves[0], halves[1], halves[2]};
			hi[child] = {halves[3], halves[4], halves[5]};
		}
	} else {
		const auto second = load(64);
		const auto word   = [&](uint32_t index) {
			return F(index < 16u ? first[index] : second[index - 16u]);
		};
		for (uint32_t child = 0; child < 4u; ++child) {
			const uint32_t at = 4u + child * 6u;
			lo[child]         = {word(at), word(at + 1u), word(at + 2u)};
			hi[child]         = {word(at + 3u), word(at + 4u), word(at + 5u)};
		}
	}
	return BoxResult(tsharp, children, lo, hi, ray, m);
}

} // namespace Libs::Graphics::ShaderRecompiler::BvhReference

#endif
