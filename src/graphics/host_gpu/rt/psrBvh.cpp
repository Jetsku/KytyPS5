#include "graphics/host_gpu/rt/psrBvh.h"

#include "graphics/shader/recompiler/BvhReference.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

// The reference node tests must not fuse multiplies into adds: the guest's V_MUL/V_MAD/V_MAC and
// the GPU test kernels (precise) round every product.
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace Libs::Graphics::Rt {

namespace {

uint32_t LoadU32(std::span<const uint8_t> bytes, size_t offset) {
	uint32_t value = 0;
	std::memcpy(&value, bytes.data() + offset, sizeof(value));
	return value;
}

float LoadF32(std::span<const uint8_t> bytes, size_t offset) {
	return std::bit_cast<float>(LoadU32(bytes, offset));
}

float HalfToFloat(uint16_t h) {
	const uint32_t sign     = static_cast<uint32_t>(h & 0x8000u) << 16u;
	const uint32_t exponent = (h >> 10u) & 0x1fu;
	uint32_t       mantissa = h & 0x3ffu;
	if (exponent == 0u) {
		if (mantissa == 0u) {
			return std::bit_cast<float>(sign);
		}
		// Subnormal half: renormalise.
		uint32_t e = 0;
		while ((mantissa & 0x400u) == 0u) {
			mantissa <<= 1u;
			e++;
		}
		mantissa &= 0x3ffu;
		return std::bit_cast<float>(sign | (127u - 15u + 1u - e) << 23u | mantissa << 13u);
	}
	if (exponent == 0x1fu) {
		return std::bit_cast<float>(sign | 0x7f800000u | mantissa << 13u);
	}
	return std::bit_cast<float>(sign | (exponent + 127u - 15u) << 23u | mantissa << 13u);
}

bool Read(const ReadMemory& read, uint64_t address, void* dst, size_t size) {
	return read && read(address, dst, size);
}

bool ReadTriangle(const ReadMemory& read, uint64_t base, uint32_t pointer, Triangle& tri,
                  WalkStatus& status) {
	std::array<uint8_t, Psr::TriangleNodeBytes> bytes {};
	if (!Read(read, Psr::HardwareNodeAddress(base, pointer), bytes.data(), bytes.size())) {
		status = WalkStatus::ReadFailed;
		return false;
	}
	if (!DecodeTriangle(Psr::TypeOf(pointer), bytes, tri)) {
		status = WalkStatus::UnsupportedNode;
		return false;
	}
	return true;
}

} // namespace

void Aabb::Grow(const std::array<float, 3>& point) {
	for (int axis = 0; axis < 3; axis++) {
		min[axis] = std::min(min[axis], point[axis]);
		max[axis] = std::max(max[axis], point[axis]);
	}
}

void Aabb::Grow(const Aabb& other) {
	for (int axis = 0; axis < 3; axis++) {
		min[axis] = std::min(min[axis], other.min[axis]);
		max[axis] = std::max(max[axis], other.max[axis]);
	}
}

const char* ToString(WalkStatus status) {
	switch (status) {
		case WalkStatus::Ok: return "ok";
		case WalkStatus::ReadFailed: return "read failed";
		case WalkStatus::UnsupportedNode: return "unsupported node type";
		case WalkStatus::TooManyNodes: return "too many nodes";
		case WalkStatus::BadPointer: return "bad pointer";
		case WalkStatus::TrapFlag: return "header trap flag";
	}
	return "?";
}

namespace {

// Bits [first, first + count) of a little-endian bit stream (count <= 32).
uint32_t Bits(std::span<const uint8_t> bytes, uint32_t first, uint32_t count) {
	uint64_t window = 0;
	std::memcpy(&window, bytes.data() + (first / 32u) * 4u,
	            std::min<size_t>(8u, bytes.size() - (first / 32u) * 4u));
	return static_cast<uint32_t>((window >> (first % 32u)) & ((uint64_t {1} << count) - 1u));
}

void PutBits(std::span<uint8_t> bytes, uint32_t first, uint32_t count, uint32_t value) {
	for (uint32_t i = 0; i < count; i++) {
		const uint32_t bit = first + i;
		const auto     b   = static_cast<uint8_t>(1u << (bit % 8u));
		if ((value >> i) & 1u) {
			bytes[bit / 8u] |= b;
		} else {
			bytes[bit / 8u] &= static_cast<uint8_t>(~b);
		}
	}
}

uint32_t Msb(uint32_t value) {
	return 31u - static_cast<uint32_t>(std::countl_zero(value));
}

} // namespace

uint32_t DecodeSharedExpBound(uint32_t field, uint32_t exponent, bool is_min) {
	const uint32_t sign      = (field >> 17u) & 1u;
	const uint32_t magnitude = field & 0x1ffffu;
	const bool     away      = is_min ? sign != 0u : sign == 0u;
	uint32_t       exp       = 0;
	uint32_t       mantissa  = 0;
	if (magnitude != 0u) {
		const uint32_t down  = std::min(16u - Msb(magnitude), exponent);
		exp                  = exponent - down;
		const uint32_t shift = 7u + down;
		const uint32_t fill  = (away && exp != 0xffu) ? (1u << shift) - 1u : 0u;
		mantissa             = ((magnitude << shift) | fill) & 0x7fffffu;
	} else {
		mantissa = away ? 0x7fffffu : 0u;
		exp      = (away && exponent > 17u) ? exponent - 17u : 0u;
	}
	return sign << 31u | exp << 23u | mantissa;
}

uint32_t EncodeSharedExpBound(float value, uint32_t exponent) {
	const uint32_t bits     = std::bit_cast<uint32_t>(value);
	const uint32_t sign     = bits >> 31u;
	const uint32_t exp      = (bits >> 23u) & 0xffu;
	const uint32_t mantissa = bits & 0x7fffc0u;
	uint32_t       aligned  = exp != 0u ? (mantissa >> 1u) | 0x400000u : mantissa;
	const uint32_t shift    = exponent - exp;
	aligned                 = shift < 23u ? aligned >> shift : 0u;
	return sign << 17u | (aligned >> 6u);
}

std::array<uint8_t, 64> EncodeSharedExpBox(uint32_t                       child_base,
                                           const std::array<uint32_t, 4>& child_types,
                                           const std::array<uint32_t, 4>& child_fields,
                                           const std::array<Aabb, 4>&     bounds) {
	std::array<uint8_t, 64> node {};
	PutBits(node, 0, 29, child_base);
	std::array<uint32_t, 3> exponent {};
	for (uint32_t c = 0; c < 4; c++) {
		PutBits(node, 29u + 3u * c, 3, child_types[c]);
		PutBits(node, 41u + 2u * c, 2, child_fields[c]);
		if (child_fields[c] == Psr::EmptyChildField) {
			continue;
		}
		for (uint32_t axis = 0; axis < 3; axis++) {
			for (const float v: {bounds[c].min[axis], bounds[c].max[axis]}) {
				exponent[axis] =
				    std::max(exponent[axis], (std::bit_cast<uint32_t>(v) >> 23u) & 0xffu);
			}
		}
	}
	for (uint32_t axis = 0; axis < 3; axis++) {
		PutBits(node, 56u + 8u * axis, 8, exponent[axis]);
	}
	for (uint32_t c = 0; c < 4; c++) {
		for (uint32_t k = 0; k < 6; k++) {
			const uint32_t axis  = k % 3u;
			const float    value = k < 3u ? bounds[c].min[axis] : bounds[c].max[axis];
			const uint32_t field = child_fields[c] == Psr::EmptyChildField
			                           ? 0u
			                           : EncodeSharedExpBound(value, exponent[axis]);
			PutBits(node, 80u + 18u * (6u * c + k), 18, field);
		}
	}
	return node;
}

bool DecodeBox(Psr::NodeType type, std::span<const uint8_t> bytes, BoxNode& out) {
	if (type == Psr::NodeType::BoxPs5) {
		if (bytes.size() < Psr::BoxPs5Bytes) {
			return false;
		}
		uint32_t running = Bits(bytes, 0, 29);
		for (uint32_t c = 0; c < 4; c++) {
			const uint32_t child_type = Bits(bytes, 29u + 3u * c, 3);
			out.fields[c]             = Bits(bytes, 41u + 2u * c, 2);
			out.children[c]           = out.fields[c] == Psr::EmptyChildField
			                                ? Psr::InvalidNode
			                                : ((running << 3u) | child_type);
			running += Psr::SharedExpChildUnits(child_type);
			for (uint32_t k = 0; k < 6; k++) {
				const uint32_t axis     = k % 3u;
				const uint32_t exponent = Bits(bytes, 56u + 8u * axis, 8);
				const uint32_t field    = Bits(bytes, 80u + 18u * (6u * c + k), 18);
				const float    value =
				    std::bit_cast<float>(DecodeSharedExpBound(field, exponent, k < 3u));
				(k < 3u ? out.bounds[c].min : out.bounds[c].max)[axis] = value;
			}
		}
		return true;
	}
	out.fields = {};
	if (type == Psr::NodeType::Box32) {
		if (bytes.size() < Psr::Box32Bytes) {
			return false;
		}
		for (uint32_t i = 0; i < 4; i++) {
			out.children[i]   = LoadU32(bytes, i * 4u);
			const size_t base = 16u + i * 24u;
			for (uint32_t axis = 0; axis < 3; axis++) {
				out.bounds[i].min[axis] = LoadF32(bytes, base + axis * 4u);
				out.bounds[i].max[axis] = LoadF32(bytes, base + 12u + axis * 4u);
			}
		}
		return true;
	}
	if (type == Psr::NodeType::Box16) {
		if (bytes.size() < Psr::Box16Bytes) {
			return false;
		}
		for (uint32_t i = 0; i < 4; i++) {
			out.children[i]   = LoadU32(bytes, i * 4u);
			const size_t base = 16u + i * 12u;
			for (uint32_t axis = 0; axis < 3; axis++) {
				uint16_t lo = 0;
				uint16_t hi = 0;
				std::memcpy(&lo, bytes.data() + base + axis * 2u, 2);
				std::memcpy(&hi, bytes.data() + base + 6u + axis * 2u, 2);
				out.bounds[i].min[axis] = HalfToFloat(lo);
				out.bounds[i].max[axis] = HalfToFloat(hi);
			}
		}
		return true;
	}
	return false;
}

bool DecodeTriangle(Psr::NodeType type, std::span<const uint8_t> bytes, Triangle& out) {
	const auto index = static_cast<uint32_t>(type);
	if (index > 3u || bytes.size() < Psr::TriangleNodeBytes) {
		return false;
	}
	for (uint32_t vertex = 0; vertex < 3; vertex++) {
		const uint32_t slot = Psr::FanTriangles[index][vertex];
		for (uint32_t axis = 0; axis < 3; axis++) {
			out.v[vertex][axis] = LoadF32(bytes, (slot * 3u + axis) * 4u);
		}
	}
	return true;
}

bool ReadBox(const ReadMemory& read, uint64_t base, uint32_t pointer, BoxNode& out,
             WalkStatus& status) {
	const auto type = Psr::TypeOf(pointer);
	if (!Psr::IsBox(pointer)) {
		status = WalkStatus::UnsupportedNode;
		return false;
	}
	std::array<uint8_t, Psr::Box32Bytes> bytes {};
	const size_t                         size = type == Psr::NodeType::Box32    ? Psr::Box32Bytes
	                                            : type == Psr::NodeType::BoxPs5 ? Psr::BoxPs5Bytes
	                                                                            : Psr::Box16Bytes;
	if (!Read(read, Psr::HardwareNodeAddress(base, pointer), bytes.data(), size)) {
		status = WalkStatus::ReadFailed;
		return false;
	}
	DecodeBox(type, std::span<const uint8_t>(bytes.data(), size), out);
	return true;
}

WalkStatus ReadLeafEntries(const ReadMemory& read, uint64_t blas_base, uint32_t leaf_pointer,
                           std::vector<LeafEntry>& out, uint32_t max_entries) {
	out.clear();
	const uint64_t list = Psr::LeafListAddress(blas_base, leaf_pointer);
	for (uint32_t index = 0; index < max_entries; index++) {
		LeafEntry entry {};
		if (!Read(read, list + index * Psr::LeafEntryBytes, &entry, sizeof(entry))) {
			return WalkStatus::ReadFailed;
		}
		if (Psr::TypeOf(entry.triangle_pointer) == Psr::NodeType::Special) {
			return WalkStatus::Ok;
		}
		out.push_back(entry);
		if ((entry.info & Psr::LeafLastFlag) != 0u) {
			return WalkStatus::Ok;
		}
	}
	return WalkStatus::TooManyNodes;
}

WalkStatus ReadInstance(const ReadMemory& read, uint64_t tlas_base, uint32_t leaf_pointer,
                        InstanceRecord& out) {
	out             = {};
	const auto type = Psr::TypeOf(leaf_pointer);
	out.fake        = type == Psr::NodeType::Special;
	uint64_t addr   = 0;
	if (leaf_pointer == Psr::InvalidNode) {
		return WalkStatus::BadPointer;
	}
	if (out.fake) {
		addr      = tlas_base + Psr::FakeInstanceOffset;
		out.index = Psr::FakeInstanceIndex;
	} else if (static_cast<uint32_t>(type) <= 3u) {
		uint64_t instances = 0;
		if (!Read(read, tlas_base + Psr::HeaderInstancesAt, &instances, sizeof(instances))) {
			return WalkStatus::ReadFailed;
		}
		out.index = leaf_pointer >> 3u;
		addr      = tlas_base + instances + static_cast<uint64_t>(out.index) * Psr::InstanceStride;
	} else {
		return WalkStatus::BadPointer;
	}
	std::array<uint32_t, 16> dwords {};
	if (!Read(read, addr, dwords.data(), sizeof(dwords))) {
		return WalkStatus::ReadFailed;
	}
	out.address = addr;
	for (int i = 0; i < 12; i++) {
		out.world_to_object[i] = std::bit_cast<float>(dwords[i]);
	}
	const uint64_t lo = dwords[12];
	const uint64_t hi = dwords[13] & Psr::BlasAddressHighMask;
	out.blas_address  = (lo | hi << 32u) << 3u;
	out.flags         = dwords[13] & ~Psr::BlasAddressHighMask;
	out.root          = out.fake ? Psr::RootPointer : dwords[14];
	out.id_mask       = dwords[15];
	return WalkStatus::Ok;
}

WalkStatus WalkBlas(const ReadMemory& read, uint64_t blas_base, uint32_t root, BlasWalk& out,
                    const WalkLimits& limits) {
	out = {};
	if (limits.reject_trap_header) {
		uint32_t flags = 0;
		if (!Read(read, blas_base + Psr::HeaderFlagsOffset, &flags, sizeof(flags))) {
			return WalkStatus::ReadFailed;
		}
		if ((flags & Psr::HeaderTrapBit) != 0u) {
			return WalkStatus::TrapFlag;
		}
	}
	if (root == Psr::InvalidNode) {
		return WalkStatus::Ok;
	}
	if (!Psr::IsBox(root)) {
		// A leaf root has no parent box: take the bounds of its triangles.
		std::vector<LeafEntry> entries;
		if (const auto status =
		        ReadLeafEntries(read, blas_base, root, entries, limits.max_leaf_entries);
		    status != WalkStatus::Ok) {
			return status;
		}
		BlasLeaf leaf {root, Aabb::Empty(), true};
		for (const auto& entry: entries) {
			Triangle   tri {};
			WalkStatus status = WalkStatus::Ok;
			if (!ReadTriangle(read, blas_base, entry.triangle_pointer, tri, status)) {
				return status;
			}
			for (const auto& v: tri.v) {
				leaf.bounds.Grow(v);
			}
		}
		out.leaves.push_back(leaf);
		return WalkStatus::Ok;
	}
	struct Item {
		uint32_t pointer;
		uint32_t depth;
	};
	std::vector<Item> stack {{root, 1}};
	while (!stack.empty()) {
		const auto item = stack.back();
		stack.pop_back();
		if (++out.box_nodes > limits.max_nodes) {
			return WalkStatus::TooManyNodes;
		}
		out.max_depth = std::max(out.max_depth, item.depth);
		if (Psr::TypeOf(item.pointer) == Psr::NodeType::BoxPs5) {
			out.shared_exp_nodes++;
		}
		BoxNode    node {};
		WalkStatus status = WalkStatus::Ok;
		if (!ReadBox(read, blas_base, item.pointer, node, status)) {
			return status;
		}
		for (uint32_t slot = 0; slot < 4; slot++) {
			const auto child = node.children[slot];
			if (child != Psr::InvalidNode && !Psr::IsBox(child)) {
				out.leaves.push_back({child, node.bounds[slot], false});
			}
		}
		for (int slot = 3; slot >= 0; slot--) {
			if (Psr::IsBox(node.children[slot])) {
				stack.push_back({node.children[slot], item.depth + 1});
			}
		}
	}
	return WalkStatus::Ok;
}

WalkStatus ExpandTriangles(const ReadMemory& read, uint64_t blas_base, const BlasWalk& walk,
                           BlasTriangles& out, const WalkLimits& limits) {
	out = {};
	std::vector<LeafEntry> entries;
	for (uint32_t leaf = 0; leaf < walk.leaves.size(); leaf++) {
		if (const auto status = ReadLeafEntries(read, blas_base, walk.leaves[leaf].pointer, entries,
		                                        limits.max_leaf_entries);
		    status != WalkStatus::Ok) {
			return status;
		}
		for (const auto& entry: entries) {
			Triangle   tri {};
			WalkStatus status = WalkStatus::Ok;
			if (!ReadTriangle(read, blas_base, entry.triangle_pointer, tri, status)) {
				return status;
			}
			out.triangles.push_back(tri);
			out.entries.push_back(entry);
			out.leaf_of.push_back(leaf);
		}
	}
	return WalkStatus::Ok;
}

WalkStatus WalkTlas(const ReadMemory& read, uint64_t tlas_base, TlasWalk& out,
                    const WalkLimits& limits) {
	out            = {};
	uint32_t flags = 0;
	if (!Read(read, tlas_base + Psr::HeaderFlagsOffset, &flags, sizeof(flags))) {
		return WalkStatus::ReadFailed;
	}
	if (limits.reject_trap_header && (flags & Psr::HeaderTrapBit) != 0u) {
		return WalkStatus::TrapFlag;
	}
	std::vector<uint32_t> stack {Psr::RootPointer};
	while (!stack.empty()) {
		const auto pointer = stack.back();
		stack.pop_back();
		if (Psr::IsBox(pointer)) {
			if (++out.box_nodes > limits.max_nodes) {
				return WalkStatus::TooManyNodes;
			}
			BoxNode    node {};
			WalkStatus status = WalkStatus::Ok;
			if (!ReadBox(read, tlas_base, pointer, node, status)) {
				return status;
			}
			for (int slot = 3; slot >= 0; slot--) {
				if (node.children[slot] != Psr::InvalidNode) {
					stack.push_back(node.children[slot]);
				}
			}
			continue;
		}
		if (out.instances.size() >= limits.max_instances) {
			return WalkStatus::TooManyNodes;
		}
		InstanceRecord record {};
		if (const auto status = ReadInstance(read, tlas_base, pointer, record);
		    status != WalkStatus::Ok) {
			return status;
		}
		out.instances.push_back(record);
	}
	return WalkStatus::Ok;
}

bool InvertAffine(const std::array<float, 12>& m, std::array<float, 12>& out) {
	const double a00 = m[0], a01 = m[1], a02 = m[2], t0 = m[3];
	const double a10 = m[4], a11 = m[5], a12 = m[6], t1 = m[7];
	const double a20 = m[8], a21 = m[9], a22 = m[10], t2 = m[11];
	const double c00 = a11 * a22 - a12 * a21;
	const double c01 = a12 * a20 - a10 * a22;
	const double c02 = a10 * a21 - a11 * a20;
	const double det = a00 * c00 + a01 * c01 + a02 * c02;
	if (det == 0.0 || !std::isfinite(det)) {
		return false;
	}
	const double inv = 1.0 / det;
	const double r00 = c00 * inv;
	const double r01 = (a02 * a21 - a01 * a22) * inv;
	const double r02 = (a01 * a12 - a02 * a11) * inv;
	const double r10 = c01 * inv;
	const double r11 = (a00 * a22 - a02 * a20) * inv;
	const double r12 = (a02 * a10 - a00 * a12) * inv;
	const double r20 = c02 * inv;
	const double r21 = (a01 * a20 - a00 * a21) * inv;
	const double r22 = (a00 * a11 - a01 * a10) * inv;
	out = {static_cast<float>(r00), static_cast<float>(r01),
	       static_cast<float>(r02), static_cast<float>(-(r00 * t0 + r01 * t1 + r02 * t2)),
	       static_cast<float>(r10), static_cast<float>(r11),
	       static_cast<float>(r12), static_cast<float>(-(r10 * t0 + r11 * t1 + r12 * t2)),
	       static_cast<float>(r20), static_cast<float>(r21),
	       static_cast<float>(r22), static_cast<float>(-(r20 * t0 + r21 * t1 + r22 * t2))};
	return true;
}

// ---------------------------------------------------------------------------------------------
// Reference node tests. The GLSL copies in the test kernels use the same operation order.

Ray MakeRay(const std::array<float, 3>& origin, const std::array<float, 3>& dir, float extent) {
	Ray ray {};
	ray.origin = origin;
	ray.dir    = dir;
	for (int axis = 0; axis < 3; axis++) {
		ray.inv_dir[axis] = 1.0f / dir[axis];
	}
	ray.extent = extent;
	return ray;
}

Ray TransformRay(const Ray& world, const std::array<float, 12>& m) {
	// The guest order: d' = (m0*dx + m1*dy) + m2*dz and o' = ((m0*ox + m3) + m1*oy) + m2*oz.
	Ray obj {};
	for (int row = 0; row < 3; row++) {
		const float* r   = &m[static_cast<size_t>(row) * 4u];
		float        d   = r[0] * world.dir[0];
		d                = d + r[1] * world.dir[1];
		d                = d + r[2] * world.dir[2];
		float o          = r[0] * world.origin[0];
		o                = o + r[3];
		o                = o + r[1] * world.origin[1];
		o                = o + r[2] * world.origin[2];
		obj.dir[row]     = d;
		obj.origin[row]  = o;
		obj.inv_dir[row] = 1.0f / d;
	}
	obj.extent = world.extent;
	return obj;
}

std::array<uint32_t, 4> ReferenceBoxTest(const BoxNode& node, const Ray& ray) {
	struct Hit {
		float    t;
		uint32_t slot;
	};
	std::array<Hit, 4> hits {};
	uint32_t           count = 0;
	for (uint32_t slot = 0; slot < 4; slot++) {
		if (node.children[slot] == Psr::InvalidNode) {
			continue;
		}
		const auto& b     = node.bounds[slot];
		float       tnear = -std::numeric_limits<float>::infinity();
		float       tfar  = std::numeric_limits<float>::infinity();
		for (int axis = 0; axis < 3; axis++) {
			const float t1 = (b.min[axis] - ray.origin[axis]) * ray.inv_dir[axis];
			const float t2 = (b.max[axis] - ray.origin[axis]) * ray.inv_dir[axis];
			tnear          = std::max(tnear, std::min(t1, t2));
			tfar           = std::min(tfar, std::max(t1, t2));
		}
		if (tfar >= tnear && tfar >= 0.0f && tnear < ray.extent) {
			hits[count++] = {std::max(tnear, 0.0f), slot};
		}
	}
	std::stable_sort(hits.begin(), hits.begin() + count,
	                 [](const Hit& a, const Hit& b) { return a.t < b.t; });
	std::array<uint32_t, 4> result {Psr::InvalidNode, Psr::InvalidNode, Psr::InvalidNode,
	                                Psr::InvalidNode};
	for (uint32_t i = 0; i < count; i++) {
		result[i] = node.children[hits[i].slot];
	}
	return result;
}

std::optional<TriangleHit> ReferenceTriangleTest(const Triangle& tri, const Ray& ray) {
	// Moller-Trumbore with every product rounded; the inside test compares sign-corrected
	// numerators so no division decides whether there is a hit.
	using V3       = std::array<float, 3>;
	const auto sub = [](const V3& a, const V3& b) {
		return V3 {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
	};
	const auto cross = [](const V3& a, const V3& b) {
		return V3 {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
	};
	const auto dot = [](const V3& a, const V3& b) {
		return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
	};
	const V3    e1  = sub(tri.v[1], tri.v[0]);
	const V3    e2  = sub(tri.v[2], tri.v[0]);
	const V3    p   = cross(ray.dir, e2);
	const float det = dot(e1, p);
	if (det == 0.0f || !std::isfinite(det)) {
		return std::nullopt;
	}
	const V3    s        = sub(ray.origin, tri.v[0]);
	const float u_num    = dot(s, p);
	const V3    q        = cross(s, e1);
	const float v_num    = dot(ray.dir, q);
	const float t_num    = dot(e2, q);
	const bool  positive = det > 0.0f;
	const float u_s      = positive ? u_num : -u_num;
	const float v_s      = positive ? v_num : -v_num;
	const float t_s      = positive ? t_num : -t_num;
	const float det_s    = positive ? det : -det;
	if (u_s < 0.0f || v_s < 0.0f || u_s + v_s > det_s || t_s < 0.0f) {
		return std::nullopt;
	}
	const float inv = 1.0f / det;
	TriangleHit hit {};
	hit.t        = t_num * inv;
	hit.u        = u_num * inv;
	hit.v        = v_num * inv;
	hit.positive = positive;
	if (!std::isfinite(hit.t)) {
		return std::nullopt;
	}
	return hit;
}

namespace {

namespace Exact = ShaderRecompiler::BvhReference;

// The RTX 3090 has no shaderDenormFlushToZeroFloat32, so the recompiled code keeps IEEE
// denormals (RT-SW); the model follows it.
constexpr Exact::Math ExactMath {false};

Exact::Ray ToExactRay(const Ray& ray, float extent) {
	return Exact::Ray {extent, ray.origin, ray.dir, ray.inv_dir};
}

// The guest's use of a return-mode-1 triangle result (section 2.3 of the design).
std::optional<TriangleHit> GuestTriangleHit(const std::array<uint32_t, 4>& result) {
	if (result[0] == Exact::PlusInf) {
		return std::nullopt;
	}
	const float t_num = std::bit_cast<float>(result[0]);
	const float t_den = std::bit_cast<float>(result[1]);
	const float rcp   = 1.0f / t_den; // V_RCP_F32, correctly rounded in this model
	TriangleHit hit {};
	hit.t        = t_num * rcp;
	hit.positive = (result[0] >> 31u) == 0u;
	return hit;
}

} // namespace

std::optional<TriangleHit> ExactTriangleTest(const Triangle& tri, const Ray& ray) {
	// The fan's type-0 triangle is (v0, v1, v2): put the three vertices there.
	std::array<uint32_t, 16> node {};
	for (uint32_t v = 0; v < 3; v++) {
		for (uint32_t axis = 0; axis < 3; axis++) {
			node[v * 3u + axis] = std::bit_cast<uint32_t>(tri.v[v][axis]);
		}
	}
	return GuestTriangleHit(
	    Exact::Triangle(GuestTsharp(0, 0), 0u, node, ToExactRay(ray, ray.extent), ExactMath));
}

NodeTests DefaultNodeTests() {
	NodeTests tests {};
	tests.box      = ReferenceBoxTest;
	tests.triangle = ExactTriangleTest;
	return tests;
}

std::array<uint32_t, 4> GuestTsharp(uint64_t bvh_base, uint64_t size) {
	// S_BFE_U64 base, header, 0x280008; dword 1 | bit 31 (sort); dword 3 = size[41:32] | 0x81000000
	// (type 8, triangle return mode 1).
	const uint64_t base = bvh_base >> 8u;
	return {static_cast<uint32_t>(base), static_cast<uint32_t>((base >> 32u) & 0xffu) | 0x80000000u,
	        static_cast<uint32_t>(size),
	        static_cast<uint32_t>((size >> 32u) & 0x3ffu) | 0x81000000u};
}

WalkStatus TraceReference(const ReadMemory& read, uint64_t tlas_base, const Ray& world,
                          const NodeTests& tests, TraceResult& out, uint32_t stack_limit) {
	out          = {};
	float best_t = world.extent;

	struct Entry {
		uint32_t pointer;
		bool     blas;
	};
	std::vector<Entry> stack;
	stack.reserve(64);
	const auto push = [&](uint32_t pointer, bool blas) {
		// The guest clamps its lane stack at 60 entries; later pushes are lost.
		if (stack.size() < stack_limit) {
			stack.push_back({pointer, blas});
		}
	};

	// The BLAS being walked. The guest keeps one at a time: its entries sit above the TLAS
	// entries on the shared stack, so popping a TLAS entry means the BLAS is finished.
	InstanceRecord instance {};
	Ray            obj {};

	uint32_t pointer = Psr::RootPointer;
	bool     blas    = false;
	for (;;) {
		if (++out.nodes_visited > (1u << 24u)) {
			return WalkStatus::TooManyNodes;
		}
		const uint64_t base      = blas ? instance.blas_address : tlas_base;
		bool           have_next = false;
		if (Psr::IsBox(pointer)) {
			BoxNode    node {};
			WalkStatus status = WalkStatus::Ok;
			if (!ReadBox(read, base, pointer, node, status)) {
				return status;
			}
			Ray bounded      = blas ? obj : world;
			bounded.extent   = best_t;
			const auto order = tests.box(node, bounded);
			int        count = 0;
			while (count < 4 && order[count] != Psr::InvalidNode) {
				count++;
			}
			// Nearest child next, the others pushed farthest first.
			for (int i = count - 1; i >= 1; i--) {
				push(order[i], blas);
			}
			if (count > 0) {
				pointer   = order[0];
				have_next = true;
			}
		} else if (pointer != Psr::InvalidNode && !blas) {
			if (const auto status = ReadInstance(read, tlas_base, pointer, instance);
			    status != WalkStatus::Ok) {
				return status;
			}
			if (instance.Mask() != 0u) {
				obj       = TransformRay(world, instance.world_to_object);
				pointer   = instance.root;
				blas      = true;
				have_next = pointer != Psr::InvalidNode;
			}
		} else if (pointer != Psr::InvalidNode) {
			std::vector<LeafEntry> entries;
			if (const auto status = ReadLeafEntries(read, base, pointer, entries);
			    status != WalkStatus::Ok) {
				return status;
			}
			const bool cull_off = (instance.flags & Psr::CullDisableFlag) != 0u;
			const bool keep_neg = (instance.flags & Psr::FacingFlag) != 0u;
			for (const auto& entry: entries) {
				Triangle   tri {};
				WalkStatus status = WalkStatus::Ok;
				if (!ReadTriangle(read, base, entry.triangle_pointer, tri, status)) {
					return status;
				}
				const auto hit = tests.triangle(tri, obj);
				if (!hit) {
					continue;
				}
				const bool facing_ok = cull_off || (hit->positive != keep_neg);
				if (facing_ok && !(hit->t > best_t)) {
					best_t               = hit->t;
					out.hit              = true;
					out.t                = hit->t;
					out.info             = entry.info;
					out.instance_index   = instance.index;
					out.instance_address = instance.address;
					out.triangle_pointer = entry.triangle_pointer;
				}
			}
		}
		if (have_next) {
			continue;
		}
		if (stack.empty()) {
			break;
		}
		pointer = stack.back().pointer;
		blas    = stack.back().blas;
		stack.pop_back();
	}
	return WalkStatus::Ok;
}

WalkStatus TraceGuestExact(const ReadMemory& read, uint64_t tlas_base, const Ray& world,
                           TraceResult& out, uint32_t stack_limit) {
	out              = {};
	bool read_failed = false;
	// The instruction reads its node dwords itself; an unreadable one fails the trace.
	const Exact::Memory memory = [&](uint64_t address) {
		uint32_t value = 0;
		if (!Read(read, address, &value, sizeof(value))) {
			read_failed = true;
		}
		return value;
	};
	// The guest's T# of a BVH, after its trap check on the header (S_BITCMP1 bit 18, S_TRAP).
	const auto tsharp_of = [&](uint64_t base, bool blas, std::array<uint32_t, 4>& tsharp) {
		uint32_t flags = 0;
		uint64_t size  = 0;
		if (!Read(read, base + Psr::HeaderFlagsOffset, &flags, sizeof(flags)) ||
		    !Read(read, base + Psr::HeaderNodeSizeAt, &size, sizeof(size))) {
			return WalkStatus::ReadFailed;
		}
		if ((flags & Psr::HeaderTrapBit) != 0u) {
			return WalkStatus::TrapFlag;
		}
		tsharp = GuestTsharp(base, blas ? size - 1u : size);
		return WalkStatus::Ok;
	};
	std::array<uint32_t, 4> tlas_tsharp {};
	std::array<uint32_t, 4> blas_tsharp {};
	if (const auto status = tsharp_of(tlas_base, false, tlas_tsharp); status != WalkStatus::Ok) {
		return status;
	}
	float best_t = world.extent;

	struct Entry {
		uint32_t pointer;
		bool     blas;
	};
	std::vector<Entry> stack;
	stack.reserve(64);
	const auto push = [&](uint32_t pointer, bool blas) {
		if (stack.size() < stack_limit) {
			stack.push_back({pointer, blas});
		}
	};
	InstanceRecord instance {};
	Ray            obj {};
	uint32_t       pointer = Psr::RootPointer;
	bool           blas    = false;
	for (;;) {
		if (++out.nodes_visited > (1u << 24u)) {
			return WalkStatus::TooManyNodes;
		}
		const uint64_t base      = blas ? instance.blas_address : tlas_base;
		bool           have_next = false;
		if (Psr::IsBox(pointer)) {
			// Children sorted by entry distance, misses (0xffffffff) last.
			const auto order =
			    Exact::Intersect(blas ? blas_tsharp : tlas_tsharp, pointer,
			                     ToExactRay(blas ? obj : world, best_t), memory, ExactMath);
			if (read_failed) {
				return WalkStatus::ReadFailed;
			}
			int count = 0;
			while (count < 4 && order[count] != Psr::InvalidNode) {
				count++;
			}
			for (int i = count - 1; i >= 1; i--) {
				push(order[i], blas);
			}
			if (count > 0) {
				pointer   = order[0];
				have_next = true;
			}
		} else if (pointer != Psr::InvalidNode && !blas) {
			if (const auto status = ReadInstance(read, tlas_base, pointer, instance);
			    status != WalkStatus::Ok) {
				return status;
			}
			if (instance.Mask() != 0u) {
				if (const auto status = tsharp_of(instance.blas_address, true, blas_tsharp);
				    status != WalkStatus::Ok) {
					return status;
				}
				obj       = TransformRay(world, instance.world_to_object);
				pointer   = instance.root;
				blas      = true;
				have_next = pointer != Psr::InvalidNode;
			}
		} else if (pointer != Psr::InvalidNode) {
			std::vector<LeafEntry> entries;
			if (const auto status = ReadLeafEntries(read, base, pointer, entries);
			    status != WalkStatus::Ok) {
				return status;
			}
			const bool cull_off = (instance.flags & Psr::CullDisableFlag) != 0u;
			const bool keep_neg = (instance.flags & Psr::FacingFlag) != 0u;
			for (const auto& entry: entries) {
				const auto result = Exact::Intersect(blas_tsharp, entry.triangle_pointer,
				                                     ToExactRay(obj, best_t), memory, ExactMath);
				if (read_failed) {
					return WalkStatus::ReadFailed;
				}
				const auto hit = GuestTriangleHit(result);
				if (hit && (cull_off || hit->positive != keep_neg) && !(hit->t > best_t)) {
					best_t               = hit->t;
					out.hit              = true;
					out.t                = hit->t;
					out.info             = entry.info;
					out.instance_index   = instance.index;
					out.instance_address = instance.address;
					out.triangle_pointer = entry.triangle_pointer;
				}
			}
		}
		if (have_next) {
			continue;
		}
		if (stack.empty()) {
			break;
		}
		pointer = stack.back().pointer;
		blas    = stack.back().blas;
		stack.pop_back();
	}
	return WalkStatus::Ok;
}

} // namespace Libs::Graphics::Rt
