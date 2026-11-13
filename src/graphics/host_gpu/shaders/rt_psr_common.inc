// Psr BVH access and reference node tests for the hardware ray-tracing kernels. Mirrors
// graphics/host_gpu/rt/psrBvh.{h,cpp}: same layout facts, same operation order (precise, no
// contraction) so decisions match the CPU reference bit for bit.

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer PsrWords {
	uint w[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer RwWords {
	uint w[];
};
layout(buffer_reference, std430, buffer_reference_align = 4) buffer RwFloats {
	float f[];
};
layout(buffer_reference, std430, buffer_reference_align = 8) buffer RwU64 {
	uint64_t q[];
};

const uint PSR_INVALID            = 0xffffffffu;
const uint PSR_ROOT               = 37u;
const uint PSR_FAKE_INSTANCE_OFF  = 0x80u;
const uint PSR_INSTANCE_STRIDE    = 128u;
const uint PSR_HEADER_FLAGS       = 24u;
const uint PSR_HEADER_TRAP_BIT    = 1u << 18u;
const uint PSR_HEADER_INSTANCES   = 48u;
const uint PSR_CULL_DISABLE       = 1u << 24u;
const uint PSR_FACING             = 1u << 25u;
const uint PSR_BLAS_HIGH_MASK     = 0x003fffffu;
const uint PSR_LEAF_LAST          = 1u << 31u;
const uint PSR_FAKE_INDEX         = 0x00ffffffu;

// Guest addresses reach device memory through a page table like the recompiler's BDA path
// (BufferCache: 16 KiB pages, one device address per page). With no table, guest addresses are
// device addresses. A node never straddles a page here; the emulator path must split reads at
// page boundaries the way LoadBda does.
const uint PSR_PAGE_BITS = 14u;
uint64_t   g_page_table  = 0ul;
uint64_t   g_page_first  = 0ul;

layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer PsrPageTable {
	uint64_t page[];
};

uint64_t PsrTranslate(uint64_t guest) {
	if (g_page_table == 0ul) {
		return guest;
	}
	uint64_t page = (guest >> PSR_PAGE_BITS) - g_page_first;
	// Bit 0 of an entry is the KYTY_BDA_ACCESS_TAGS accessed tag (BufferCache::BDA_ACCESSED).
	return (PsrPageTable(g_page_table).page[uint(page)] & ~1ul) +
	       (guest & ((1ul << PSR_PAGE_BITS) - 1ul));
}

float PsrInf() {
	return uintBitsToFloat(0x7f800000u);
}

bool PsrIsBox(uint ptr) {
	uint t = ptr & 7u;
	return ptr != PSR_INVALID && t >= 4u && t <= 6u;
}
uint64_t PsrNodeAddress(uint64_t base, uint ptr) {
	return base + (uint64_t(ptr & ~7u) << 3);
}
uint64_t PsrLeafAddress(uint64_t base, uint ptr) {
	return base + uint64_t(ptr & ~7u);
}
uint PsrLoad(uint64_t guest) {
	return PsrWords(PsrTranslate(guest)).w[0];
}
uint64_t PsrLoad64(uint64_t guest) {
	PsrWords p = PsrWords(PsrTranslate(guest));
	return uint64_t(p.w[0]) | (uint64_t(p.w[1]) << 32);
}
// Device view of a guest object (node, record, leaf list).
PsrWords PsrObject(uint64_t guest) {
	return PsrWords(PsrTranslate(guest));
}

struct PsrBox {
	uint child[4];
	vec3 bmin[4];
	vec3 bmax[4];
};

// Bits [first, first + count) of a node (count <= 18 here).
uint PsrBits(PsrWords n, uint first, uint count) {
	uint word  = first >> 5u;
	uint shift = first & 31u;
	uint value = n.w[word] >> shift;
	if (shift + count > 32u) {
		value |= n.w[word + 1u] << (32u - shift);
	}
	return value & ((1u << count) - 1u);
}

// One bound of a type-6 node (psrBvh.cpp, DecodeSharedExpBound): float bits, rounded outwards.
uint PsrSharedExpBound(uint field, uint exponent, bool is_min) {
	uint sign      = (field >> 17u) & 1u;
	uint magnitude = field & 0x1ffffu;
	bool away      = is_min ? sign != 0u : sign == 0u;
	uint e         = 0u;
	uint mantissa  = 0u;
	if (magnitude != 0u) {
		uint down  = min(16u - uint(findMSB(magnitude)), exponent);
		e          = exponent - down;
		uint shift = 7u + down;
		uint fill  = (away && e != 0xffu) ? (1u << shift) - 1u : 0u;
		mantissa   = ((magnitude << shift) | fill) & 0x7fffffu;
	} else {
		mantissa = away ? 0x7fffffu : 0u;
		e        = (away && exponent > 17u) ? exponent - 17u : 0u;
	}
	return (sign << 31u) | (e << 23u) | mantissa;
}

// Box16 (type 4), box32 (type 5) and the Psr shared-exponent box (type 6).
bool PsrReadBox(uint64_t base, uint ptr, out PsrBox box) {
	uint     type = ptr & 7u;
	PsrWords n    = PsrObject(PsrNodeAddress(base, ptr));
	if (type == 6u) {
		uint running = PsrBits(n, 0u, 29u);
		uint ex      = PsrBits(n, 56u, 8u);
		uint ey      = PsrBits(n, 64u, 8u);
		uint ez      = PsrBits(n, 72u, 8u);
		for (uint c = 0u; c < 4u; c++) {
			uint child_type  = PsrBits(n, 29u + 3u * c, 3u);
			uint child_field = PsrBits(n, 41u + 2u * c, 2u);
			box.child[c]     = child_field == 3u ? PSR_INVALID : ((running << 3u) | child_type);
			running += child_type == 2u ? 4u
			           : (child_type == 3u || child_type == 5u || child_type == 6u) ? 2u
			                                                                        : 1u;
			uint b      = 80u + 108u * c;
			box.bmin[c] = vec3(uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b, 18u), ex, true)),
			                   uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b + 18u, 18u), ey, true)),
			                   uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b + 36u, 18u), ez, true)));
			box.bmax[c] = vec3(uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b + 54u, 18u), ex, false)),
			                   uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b + 72u, 18u), ey, false)),
			                   uintBitsToFloat(PsrSharedExpBound(PsrBits(n, b + 90u, 18u), ez, false)));
		}
		return true;
	}
	for (uint i = 0u; i < 4u; i++) {
		box.child[i] = n.w[i];
	}
	if (type == 5u) {
		for (uint i = 0u; i < 4u; i++) {
			uint b      = 4u + i * 6u;
			box.bmin[i] = vec3(uintBitsToFloat(n.w[b + 0u]), uintBitsToFloat(n.w[b + 1u]),
			                   uintBitsToFloat(n.w[b + 2u]));
			box.bmax[i] = vec3(uintBitsToFloat(n.w[b + 3u]), uintBitsToFloat(n.w[b + 4u]),
			                   uintBitsToFloat(n.w[b + 5u]));
		}
		return true;
	}
	if (type == 4u) {
		for (uint i = 0u; i < 4u; i++) {
			uint b      = 4u + i * 3u;
			vec2 a0     = unpackHalf2x16(n.w[b + 0u]);
			vec2 a1     = unpackHalf2x16(n.w[b + 1u]);
			vec2 a2     = unpackHalf2x16(n.w[b + 2u]);
			box.bmin[i] = vec3(a0.x, a0.y, a1.x);
			box.bmax[i] = vec3(a1.y, a2.x, a2.y);
		}
		return true;
	}
	return false;
}

struct PsrRay {
	vec3  o;
	vec3  d;
	vec3  inv;
	float extent;
};

// Child pointers in entry-distance order (stable for ties), misses as PSR_INVALID.
void PsrBoxTest(PsrBox box, PsrRay ray, out uint order[4], out uint count) {
	float t[4];
	uint  s[4];
	count = 0u;
	for (uint slot = 0u; slot < 4u; slot++) {
		if (box.child[slot] == PSR_INVALID) {
			continue;
		}
		float tnear = -PsrInf();
		float tfar  = PsrInf();
		for (int axis = 0; axis < 3; axis++) {
			precise float lo = box.bmin[slot][axis] - ray.o[axis];
			precise float hi = box.bmax[slot][axis] - ray.o[axis];
			precise float t1 = lo * ray.inv[axis];
			precise float t2 = hi * ray.inv[axis];
			tnear            = max(tnear, min(t1, t2));
			tfar             = min(tfar, max(t1, t2));
		}
		if (tfar >= tnear && tfar >= 0.0 && tnear < ray.extent) {
			t[count] = max(tnear, 0.0);
			s[count] = slot;
			count++;
		}
	}
	// Stable insertion sort on at most four entries.
	for (uint i = 1u; i < count; i++) {
		float tk = t[i];
		uint  sk = s[i];
		uint  j  = i;
		while (j > 0u && tk < t[j - 1u]) {
			t[j] = t[j - 1u];
			s[j] = s[j - 1u];
			j--;
		}
		t[j] = tk;
		s[j] = sk;
	}
	for (uint i = 0u; i < 4u; i++) {
		order[i] = i < count ? box.child[s[i]] : PSR_INVALID;
	}
}

// Triangle nodes: five float3 vertices; pointer types 0..3 select one of four fan triangles
// around v2 (psrBvh.h, Psr::FanTriangles).
const uint PSR_FAN[12] = uint[](0u, 1u, 2u, 1u, 3u, 2u, 2u, 3u, 4u, 2u, 4u, 0u);

vec3 PsrVertex(PsrWords n, uint slot) {
	uint b = slot * 3u;
	return vec3(uintBitsToFloat(n.w[b]), uintBitsToFloat(n.w[b + 1u]), uintBitsToFloat(n.w[b + 2u]));
}

bool PsrReadTriangle(uint64_t base, uint ptr, out vec3 v0, out vec3 v1, out vec3 v2) {
	uint t = ptr & 7u;
	if (t > 3u) {
		return false;
	}
	PsrWords n = PsrObject(PsrNodeAddress(base, ptr));
	v0         = PsrVertex(n, PSR_FAN[t * 3u + 0u]);
	v1         = PsrVertex(n, PSR_FAN[t * 3u + 1u]);
	v2         = PsrVertex(n, PSR_FAN[t * 3u + 2u]);
	return true;
}

vec3 PsrSub(vec3 a, vec3 b) {
	precise vec3 r = a - b;
	return r;
}
vec3 PsrCross(vec3 a, vec3 b) {
	precise float x0 = a.y * b.z;
	precise float x1 = a.z * b.y;
	precise float y0 = a.z * b.x;
	precise float y1 = a.x * b.z;
	precise float z0 = a.x * b.y;
	precise float z1 = a.y * b.x;
	precise vec3  r  = vec3(x0 - x1, y0 - y1, z0 - z1);
	return r;
}
float PsrDot(vec3 a, vec3 b) {
	precise float x = a.x * b.x;
	precise float y = a.y * b.y;
	precise float z = a.z * b.z;
	precise float s = x + y;
	precise float r = s + z;
	return r;
}

// Correctly rounded f32 division (round to nearest even, IEEE denormals), the CPU's `/`: GLSL's
// `/` is only required to be within 2.5 ULP. Every NaN result is the canonical quiet NaN; NaN
// payloads never decide anything the tests compare.
uint PsrDivBits(uint a, uint b) {
	uint sign = (a ^ b) & 0x80000000u;
	uint ea   = (a >> 23u) & 0xffu;
	uint eb   = (b >> 23u) & 0xffu;
	uint ma   = a & 0x7fffffu;
	uint mb   = b & 0x7fffffu;
	if ((ea == 0xffu && ma != 0u) || (eb == 0xffu && mb != 0u)) {
		return 0x7fc00000u;
	}
	bool inf_a  = ea == 0xffu;
	bool inf_b  = eb == 0xffu;
	bool zero_a = (a & 0x7fffffffu) == 0u;
	bool zero_b = (b & 0x7fffffffu) == 0u;
	if ((inf_a && inf_b) || (zero_a && zero_b)) {
		return 0x7fc00000u;
	}
	if (inf_a || zero_b) {
		return sign | 0x7f800000u;
	}
	if (inf_b || zero_a) {
		return sign;
	}
	// 24-bit significands; denormals are normalized with a lower exponent.
	int xa = int(ea);
	int xb = int(eb);
	if (ea == 0u) {
		int s = 23 - findMSB(ma);
		ma <<= uint(s);
		xa = 1 - s;
	} else {
		ma |= 0x800000u;
	}
	if (eb == 0u) {
		int s = 23 - findMSB(mb);
		mb <<= uint(s);
		xb = 1 - s;
	} else {
		mb |= 0x800000u;
	}
	// A 27-bit quotient (24 bits, guard, round and one more) plus a sticky remainder.
	bool     ge     = ma >= mb;
	uint64_t n      = uint64_t(ma) << (ge ? 26u : 27u);
	uint     q      = uint(n / uint64_t(mb));
	bool     sticky = (n % uint64_t(mb)) != 0ul;
	int      e      = xa - xb + 127 - (ge ? 0 : 1);
	// Results below the normal range shift further right before rounding.
	uint total    = 3u + uint(e >= 1 ? 0 : min(1 - e, 27));
	uint mant     = q >> total;
	uint lost     = q & ((1u << total) - 1u);
	uint halfway  = 1u << (total - 1u);
	if (lost > halfway || (lost == halfway && (sticky || (mant & 1u) != 0u))) {
		mant += 1u;
	}
	// The implicit bit (and a rounding carry) lands in the exponent field.
	uint bits = (e >= 1 ? uint(e - 1) << 23u : 0u) + mant;
	if (bits >= 0x7f800000u) {
		return sign | 0x7f800000u;
	}
	return sign | bits;
}
float PsrDiv(float a, float b) {
	return uintBitsToFloat(PsrDivBits(floatBitsToUint(a), floatBitsToUint(b)));
}

// The instruction's miss decision on the correctly rounded quotients t, u, v (and the rounded u + v).
bool PsrTriangleMissExact(float t_num, float t_den, float i_num, float j_num) {
	float         t  = PsrDiv(t_num, t_den);
	float         u  = PsrDiv(i_num, t_den);
	float         v  = PsrDiv(j_num, t_den);
	precise float uv = u + v;
	return u < 0.0 || u > 1.0 || v < 0.0 || uv > 1.0 || t < 0.0;
}

// The same decision without the exact divisions when it is certain: GLSL's division is within
// 2.5 ULP for divisors in [2^-126, 2^126], so an approximate quotient more than 2^-18 (relative)
// away from 1, larger than 2^-100 in magnitude (no underflow to a signed zero), and an approximate
// u + v clear of 1 by a margin give the exact quotients' answers. Everything else (near a
// threshold, special values, extreme divisors) takes the exact path.
bool PsrTriangleMiss(float t_num, float t_den, float i_num, float j_num) {
	float min_den = uintBitsToFloat(0x00800000u); // 2^-126
	float max_den = uintBitsToFloat(0x7e800000u); // 2^126
	float tiny    = uintBitsToFloat(0x0d800000u); // 2^-100
	float margin  = uintBitsToFloat(0x36800000u); // 2^-18
	float abs_den = abs(t_den);
	if (!(abs_den >= min_den && abs_den <= max_den)) {
		return PsrTriangleMissExact(t_num, t_den, i_num, j_num);
	}
	precise float r  = 1.0 / t_den;
	precise float t  = t_num * r;
	precise float u  = i_num * r;
	precise float v  = j_num * r;
	precise float uv = u + v;
	precise float du = u - 1.0;
	precise float dv = v - 1.0;
	precise float ds = uv - 1.0;
	bool certain = !isinf(t) && !isnan(t) && !isinf(uv) && !isnan(uv) && abs(t) > tiny &&
	               abs(u) > tiny && abs(v) > tiny && abs(du) > margin && abs(dv) > margin &&
	               abs(ds) > margin * (1.0 + abs(u) + abs(v));
	if (!certain) {
		return PsrTriangleMissExact(t_num, t_den, i_num, j_num);
	}
	return u < 0.0 || u > 1.0 || v < 0.0 || uv > 1.0 || t < 0.0;
}

// IMAGE_BVH_INTERSECT_RAY on a triangle, return mode 1: the exact model of RT-SW's
// BvhReference::Triangle (graphics/shader/recompiler/BvhReference.h), same operations in the same
// order. A miss returns t_num = +inf and t_den = 1.0.
void PsrTriangleExact(vec3 a, vec3 b, vec3 c, PsrRay ray, out float t_num_out,
                      out float t_den_out) {
	vec3  e1    = PsrSub(b, a);
	vec3  e2    = PsrSub(c, a);
	vec3  e3    = PsrSub(ray.o, a);
	vec3  s1    = PsrCross(ray.d, e2);
	vec3  s2    = PsrCross(e3, e1);
	float t_num = PsrDot(e2, s2);
	float t_den = PsrDot(s1, e1);
	float i_num = PsrDot(e3, s1);
	float j_num = PsrDot(ray.d, s2);
#ifdef PSR_EXACT_DIVISIONS_ONLY
	bool miss = PsrTriangleMissExact(t_num, t_den, i_num, j_num);
#else
	bool miss = PsrTriangleMiss(t_num, t_den, i_num, j_num);
#endif
	t_num_out = miss ? PsrInf() : t_num;
	t_den_out = miss ? 1.0 : t_den;
}

// The guest's use of that result (traversal routine, section 2.3 of the design): a hit unless
// t_num is +inf, t = t_num * V_RCP_F32(t_den) (a correctly rounded reciprocal in this model), and
// the facing is the sign bit of t_num. Mirrors ExactTriangleTest in psrBvh.cpp.
bool PsrTriangleTest(vec3 v0, vec3 v1, vec3 v2, PsrRay ray, out float t, out bool positive) {
	float t_num;
	float t_den;
	PsrTriangleExact(v0, v1, v2, ray, t_num, t_den);
	positive = (floatBitsToUint(t_num) >> 31u) == 0u;
	if (floatBitsToUint(t_num) == 0x7f800000u) {
		t = 0.0;
		return false;
	}
	precise float tt = t_num * PsrDiv(1.0, t_den);
	t                = tt;
	return true;
}

struct PsrInstance {
	uint64_t address;
	uint     index;
	float    m[12];
	uint64_t blas;
	uint     flags;
	uint     root;
	uint     id_mask;
};

bool PsrReadInstance(uint64_t tlas_base, uint leaf_ptr, out PsrInstance inst) {
	uint type = leaf_ptr & 7u;
	if (leaf_ptr == PSR_INVALID || (type >= 4u && type <= 6u)) {
		return false;
	}
	if (type == 7u) {
		inst.address = tlas_base + PSR_FAKE_INSTANCE_OFF;
		inst.index   = PSR_FAKE_INDEX;
	} else {
		uint64_t instances = PsrLoad64(tlas_base + PSR_HEADER_INSTANCES);
		inst.index         = leaf_ptr >> 3u;
		inst.address = tlas_base + instances + uint64_t(inst.index) * uint64_t(PSR_INSTANCE_STRIDE);
	}
	PsrWords r = PsrObject(inst.address);
	for (uint i = 0u; i < 12u; i++) {
		inst.m[i] = uintBitsToFloat(r.w[i]);
	}
	uint lo      = r.w[12];
	uint hi      = r.w[13];
	inst.blas    = (uint64_t(lo) | (uint64_t(hi & PSR_BLAS_HIGH_MASK) << 32)) << 3;
	inst.flags   = hi & ~PSR_BLAS_HIGH_MASK;
	inst.root    = type == 7u ? PSR_ROOT : r.w[14];
	inst.id_mask = r.w[15];
	return true;
}

// The guest's object-space ray: d' = (m0*dx + m1*dy) + m2*dz, o' = ((m0*ox + m3) + m1*oy) + m2*oz.
PsrRay PsrTransformRay(PsrRay world, float m[12]) {
	PsrRay obj;
	for (int row = 0; row < 3; row++) {
		precise float d = m[row * 4 + 0] * world.d.x;
		d               = d + m[row * 4 + 1] * world.d.y;
		d               = d + m[row * 4 + 2] * world.d.z;
		precise float o = m[row * 4 + 0] * world.o.x;
		o               = o + m[row * 4 + 3];
		o               = o + m[row * 4 + 1] * world.o.y;
		o               = o + m[row * 4 + 2] * world.o.z;
		obj.d[row]      = d;
		obj.o[row]      = o;
		obj.inv[row]    = PsrDiv(1.0, d);
	}
	obj.extent = world.extent;
	return obj;
}

PsrRay PsrMakeRay(vec3 o, vec3 d, vec3 inv, float extent) {
	PsrRay ray;
	ray.o      = o;
	ray.d      = d;
	ray.inv    = inv;
	ray.extent = extent;
	return ray;
}

// Ray (48 bytes; the reciprocal direction comes from the host so world-space box tests match the
// CPU reference exactly) and result (32 bytes) records shared by the trace kernels.
struct TraceRay {
	float ox, oy, oz, extent;
	float dx, dy, dz, pad0;
	float ix, iy, iz, pad1;
};
struct TraceOut {
	float t;
	uint  info;
	uint  instance_index;
	uint  tri_ptr;
	uint  hit;
	uint  nodes;
	uint  candidates;
	uint  status;
};
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer TraceRays {
	TraceRay r[];
};
layout(buffer_reference, std430, buffer_reference_align = 16) buffer TraceOuts {
	TraceOut r[];
};
