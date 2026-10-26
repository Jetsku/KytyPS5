// Builds BVHs in the Psr layout the guest traversal reads (graphics/host_gpu/rt/psrBvh.h), for
// the ray-tracing hardware tests and benchmarks. Only the fields the traversal reads are written;
// the rest of the 128-byte header stays zero.

#ifndef KYTY_TESTS_RTPSRBVHBUILDER_H_
#define KYTY_TESTS_RTPSRBVHBUILDER_H_

#include "graphics/host_gpu/rt/psrBvh.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <vector>

namespace RtTest {

using namespace Libs::Graphics::Rt;

// Guest-like memory: a set of blobs at fixed addresses.
class GuestMemory {
public:
	uint64_t Allocate(uint64_t size, uint64_t align = 256) {
		m_next             = (m_next + align - 1) & ~(align - 1);
		const auto address = m_next;
		m_next += size;
		m_blobs[address].resize(size);
		return address;
	}
	std::vector<uint8_t>& Blob(uint64_t address) { return m_blobs.at(address); }
	[[nodiscard]] const std::map<uint64_t, std::vector<uint8_t>>& Blobs() const { return m_blobs; }
	bool Read(uint64_t address, void* dst, size_t size) const {
		auto it = m_blobs.upper_bound(address);
		if (it == m_blobs.begin()) {
			return false;
		}
		--it;
		if (address + size > it->first + it->second.size()) {
			return false;
		}
		std::memcpy(dst, it->second.data() + (address - it->first), size);
		return true;
	}
	[[nodiscard]] ReadMemory Reader() const {
		return
		    [this](uint64_t address, void* dst, size_t size) { return Read(address, dst, size); };
	}
	// A blob at a fixed address (a loaded BVH dump's pages); blobs must not overlap.
	void Insert(uint64_t address, std::vector<uint8_t> bytes) {
		m_begin          = std::min(m_begin, address);
		m_next           = std::max<uint64_t>(m_next, address + bytes.size());
		m_blobs[address] = std::move(bytes);
	}
	[[nodiscard]] uint64_t Begin() const { return m_begin; }
	[[nodiscard]] uint64_t End() const { return m_next; }

	static constexpr uint64_t Base = 0x0000000140000000ull;

private:
	uint64_t                                 m_begin = Base;
	uint64_t                                 m_next  = Base;
	std::map<uint64_t, std::vector<uint8_t>> m_blobs;
};

struct BuildOptions {
	uint32_t leaf_size      = 4;     // triangles per leaf list
	bool     box16_interior = false; // non-root box nodes as box16 (conservative fp16 bounds)
	bool     leaf_root      = false; // BLAS only: the instance root is a leaf (no box node)
	// Non-root nodes whose children are all box nodes become Psr shared-exponent boxes (type 6,
	// implicit child pointers). Children are always laid out contiguously for this.
	bool shared_exp_interior = false;
};

namespace Detail {

inline void Put32(std::vector<uint8_t>& blob, uint64_t offset, uint32_t value) {
	std::memcpy(blob.data() + offset, &value, 4);
}
inline void PutF32(std::vector<uint8_t>& blob, uint64_t offset, float value) {
	Put32(blob, offset, std::bit_cast<uint32_t>(value));
}
inline void Put64(std::vector<uint8_t>& blob, uint64_t offset, uint64_t value) {
	std::memcpy(blob.data() + offset, &value, 8);
}

// Float to half, rounding towards -inf (down) or +inf (up), for conservative box16 bounds.
inline uint16_t HalfDirected(float value, bool up) {
	// Largest finite half magnitudes clamp; the test scenes stay far inside the fp16 range.
	const float clamped = std::clamp(value, -65504.0f, 65504.0f);
	// Search the two neighbouring halves around a round-to-nearest conversion.
	const auto to_half = [](float f) -> uint16_t {
		const uint32_t x    = std::bit_cast<uint32_t>(f);
		const uint32_t sign = (x >> 16u) & 0x8000u;
		const int      e    = static_cast<int>((x >> 23u) & 0xffu) - 127 + 15;
		uint32_t       m    = x & 0x7fffffu;
		if (e <= 0) {
			if (e < -10) {
				return static_cast<uint16_t>(sign);
			}
			m |= 0x800000u;
			const uint32_t shift = static_cast<uint32_t>(14 - e);
			uint32_t       h     = m >> shift;
			if ((m >> (shift - 1u)) & 1u) {
				h++;
			}
			return static_cast<uint16_t>(sign | h);
		}
		if (e >= 31) {
			return static_cast<uint16_t>(sign | 0x7bffu);
		}
		uint32_t h = sign | static_cast<uint32_t>(e) << 10u | m >> 13u;
		if (m & 0x1000u) {
			h++;
		}
		return static_cast<uint16_t>(h);
	};
	const auto to_float = [](uint16_t h) {
		// Reuse the product decoder for the exact half->float mapping.
		Libs::Graphics::Rt::BoxNode decoded {};
		std::array<uint8_t, 64>     bytes {};
		std::memcpy(bytes.data() + 16, &h, 2);
		Libs::Graphics::Rt::DecodeBox(Libs::Graphics::Rt::Psr::NodeType::Box16, bytes, decoded);
		return decoded.bounds[0].min[0];
	};
	uint16_t h = to_half(clamped);
	for (int step = 0; step < 4; step++) {
		const float f = to_float(h);
		if (up ? (f >= value) : (f <= value)) {
			break;
		}
		// Step one half ulp in the needed direction.
		const bool negative = (h & 0x8000u) != 0u;
		if (up) {
			if (negative) {
				h = (h & 0x7fffu) == 0u ? static_cast<uint16_t>(0x0001u)
				                        : static_cast<uint16_t>(h - 1u);
			} else {
				h = static_cast<uint16_t>(h + 1u);
			}
		} else {
			if (negative) {
				h = static_cast<uint16_t>(h + 1u);
			} else {
				h = h == 0u ? static_cast<uint16_t>(0x8001u) : static_cast<uint16_t>(h - 1u);
			}
		}
	}
	return h;
}

struct Item {
	Aabb                 bounds;
	std::array<float, 3> centroid;
	uint32_t             index;
};

} // namespace Detail

// Writes one Psr BVH (header, fake instance, box nodes, leaves) into a fresh guest blob.
class PsrBvhWriter {
public:
	PsrBvhWriter(GuestMemory& memory, const BuildOptions& options)
	    : m_memory(memory), m_options(options) {}

	struct Result {
		uint64_t              base      = 0;
		uint32_t              root      = Psr::RootPointer;
		Aabb                  bounds    = Aabb::Empty();
		uint32_t              box_nodes = 0;
		uint32_t              leaves    = 0;
		std::vector<Triangle> triangles; // primitive i as the node decodes it (info word = i)
	};

	using TriNode = std::array<std::array<float, 3>, Psr::TriangleNodeVertices>;

	// BLAS over triangles, one triangle node (type 0) each. Info word of triangle i = i.
	Result BuildBlas(const std::vector<Triangle>& triangles) {
		m_tri_nodes.clear();
		m_prims.clear();
		for (const auto& tri: triangles) {
			TriNode node {};
			node[0] = tri.v[0];
			node[1] = tri.v[1];
			node[2] = tri.v[2];
			m_prims.push_back({static_cast<uint32_t>(m_tri_nodes.size()), 0u});
			m_tri_nodes.push_back(node);
		}
		return BuildTriangleBlas();
	}

	// BLAS over fan nodes: each node stores five vertices and all four of its triangles are
	// primitives (pointer types 0..3).
	Result BuildBlasFans(const std::vector<TriNode>& fans) {
		m_tri_nodes = fans;
		m_prims.clear();
		for (uint32_t n = 0; n < fans.size(); n++) {
			for (uint32_t t = 0; t < 4; t++) {
				m_prims.push_back({n, t});
			}
		}
		return BuildTriangleBlas();
	}

	struct InstanceDesc {
		std::array<float, 12> object_to_world {};
		uint64_t              blas_address = 0;
		uint32_t              blas_root    = Psr::RootPointer;
		Aabb                  blas_bounds {};
		uint32_t              flags = Psr::CullDisableFlag;
		uint32_t              mask  = 0xffu;
		uint32_t              id    = 0;
	};

	// TLAS over instances; instance i is record i of the instance array.
	Result BuildTlas(const std::vector<InstanceDesc>& instances) {
		std::vector<Detail::Item> items(instances.size());
		for (uint32_t i = 0; i < instances.size(); i++) {
			auto& item  = items[i];
			item.bounds = TransformBounds(instances[i].blas_bounds, instances[i].object_to_world);
			for (int a = 0; a < 3; a++) {
				item.centroid[a] = 0.5f * (item.bounds.min[a] + item.bounds.max[a]);
			}
			item.index = i;
		}
		m_instances = &instances;
		m_leaf_kind = LeafKind::Instances;
		return Build(items);
	}

	Triangle PrimTriangle(uint32_t prim) const {
		const auto [node, type] = m_prims[prim];
		Triangle tri {};
		for (int v = 0; v < 3; v++) {
			tri.v[v] = m_tri_nodes[node][Psr::FanTriangles[type][v]];
		}
		return tri;
	}

	static Aabb TransformBounds(const Aabb& b, const std::array<float, 12>& m) {
		Aabb out = Aabb::Empty();
		for (int corner = 0; corner < 8; corner++) {
			const std::array<float, 3> p {(corner & 1) ? b.max[0] : b.min[0],
			                              (corner & 2) ? b.max[1] : b.min[1],
			                              (corner & 4) ? b.max[2] : b.min[2]};
			std::array<float, 3>       q {};
			for (int r = 0; r < 3; r++) {
				q[r] = m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3];
			}
			out.Grow(q);
		}
		// One float ulp of slack either way for the rounding of the transform.
		for (int a = 0; a < 3; a++) {
			out.min[a] = std::nextafter(out.min[a], -INFINITY);
			out.max[a] = std::nextafter(out.max[a], INFINITY);
		}
		return out;
	}

private:
	enum class LeafKind { Triangles, Instances };

	Result BuildTriangleBlas() {
		std::vector<Detail::Item> items(m_prims.size());
		std::vector<Triangle>     decoded(m_prims.size());
		for (uint32_t i = 0; i < m_prims.size(); i++) {
			decoded[i]  = PrimTriangle(i);
			auto& item  = items[i];
			item.bounds = Aabb::Empty();
			for (const auto& v: decoded[i].v) {
				item.bounds.Grow(v);
			}
			for (int a = 0; a < 3; a++) {
				item.centroid[a] = 0.5f * (item.bounds.min[a] + item.bounds.max[a]);
			}
			item.index = i;
		}
		m_leaf_kind      = LeafKind::Triangles;
		Result result    = Build(items);
		result.triangles = std::move(decoded);
		return result;
	}

	struct Node {
		std::array<int32_t, 4> child_node {-1, -1, -1, -1}; // index into m_nodes
		std::array<int32_t, 4> child_leaf {-1, -1, -1, -1}; // index into m_leaf_groups
		std::array<Aabb, 4>    child_bounds {};
		uint32_t               count      = 0;
		bool                   box16      = false;
		bool                   shared_exp = false;
		uint64_t               offset     = 0;

		[[nodiscard]] Psr::NodeType Type() const {
			return shared_exp ? Psr::NodeType::BoxPs5
			       : box16    ? Psr::NodeType::Box16
			                  : Psr::NodeType::Box32;
		}
		// Bytes the node occupies: the type-6 implicit child sizes count it as 2 units.
		[[nodiscard]] uint64_t Size() const {
			return shared_exp ? 2u * 64u : box16 ? Psr::Box16Bytes : Psr::Box32Bytes;
		}
	};

	Result Build(std::vector<Detail::Item>& items) {
		m_nodes.clear();
		m_leaf_groups.clear();
		Result result {};
		for (const auto& item: items) {
			result.bounds.Grow(item.bounds);
		}
		// The root is always a box32 node at byte 256 (pointer 37).
		const int root = static_cast<int>(m_nodes.size());
		m_nodes.emplace_back();
		const bool leaf_root = m_options.leaf_root && m_leaf_kind == LeafKind::Triangles &&
		                       items.size() <= m_options.leaf_size;
		if (!items.empty() && !leaf_root) {
			Split(root, items, 0, static_cast<uint32_t>(items.size()), true);
		} else if (leaf_root) {
			m_leaf_groups.emplace_back();
			for (const auto& item: items) {
				m_leaf_groups.back().push_back(item.index);
			}
		}
		// Layout: header, fake instance, nodes, leaf lists, triangle nodes, instances.
		uint64_t offset = Psr::RootNodeOffset;
		for (auto& node: m_nodes) {
			node.offset = offset;
			offset += node.Size();
		}
		std::vector<uint64_t> leaf_offsets(m_leaf_groups.size());
		if (m_leaf_kind == LeafKind::Triangles) {
			for (size_t g = 0; g < m_leaf_groups.size(); g++) {
				leaf_offsets[g] = offset;
				offset += m_leaf_groups[g].size() * Psr::LeafEntryBytes;
			}
			offset = (offset + 63u) & ~uint64_t(63u);
		}
		const uint64_t triangles_at = offset;
		if (m_leaf_kind == LeafKind::Triangles) {
			offset += m_tri_nodes.size() * Psr::TriangleNodeBytes;
		}
		offset                      = (offset + 127u) & ~uint64_t(127u);
		const uint64_t instances_at = offset;
		if (m_leaf_kind == LeafKind::Instances) {
			offset += m_instances->size() * Psr::InstanceStride;
		}
		const uint64_t size = std::max<uint64_t>(offset, Psr::RootNodeOffset + Psr::Box32Bytes);
		result.base         = m_memory.Allocate(size, 256);
		auto& blob          = m_memory.Blob(result.base);
		using namespace Detail;

		// Header: only what the traversal reads.
		Put32(blob, Psr::HeaderFlagsOffset, 0);
		Put64(blob, Psr::HeaderInstancesAt, instances_at);
		Put64(blob, Psr::HeaderNodeSizeAt, size / 64u);
		Put32(blob, Psr::HeaderLeafRecordsAt, static_cast<uint32_t>(size / 8u));
		// Fake instance: identity world-to-object, the BVH itself as its BLAS.
		WriteInstanceRecord(blob, Psr::FakeInstanceOffset, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
		                    result.base, Psr::CullDisableFlag, Psr::RootPointer, 0xffu, 0);

		if (leaf_root) {
			result.root = Psr::LeafPointer(leaf_offsets[0], Psr::NodeType::Leaf0);
		}
		for (auto& node: m_nodes) {
			WriteNode(blob, node, leaf_offsets);
		}
		if (m_leaf_kind == LeafKind::Triangles) {
			for (size_t g = 0; g < m_leaf_groups.size(); g++) {
				const auto& group = m_leaf_groups[g];
				for (size_t e = 0; e < group.size(); e++) {
					const uint32_t tri            = group[e];
					const auto [node_index, type] = m_prims[tri];
					const uint64_t node =
					    triangles_at + uint64_t(node_index) * Psr::TriangleNodeBytes;
					const uint32_t ptr =
					    Psr::HardwarePointer(node, static_cast<Psr::NodeType>(type));
					uint32_t info = tri;
					if (e + 1 == group.size()) {
						info |= Psr::LeafLastFlag;
					}
					Put32(blob, leaf_offsets[g] + e * 8u, ptr);
					Put32(blob, leaf_offsets[g] + e * 8u + 4u, info);
				}
			}
			for (uint32_t n = 0; n < m_tri_nodes.size(); n++) {
				const uint64_t node = triangles_at + uint64_t(n) * Psr::TriangleNodeBytes;
				for (uint32_t v = 0; v < Psr::TriangleNodeVertices; v++) {
					for (int a = 0; a < 3; a++) {
						PutF32(blob, node + (v * 3 + a) * 4u, m_tri_nodes[n][v][a]);
					}
				}
			}
		} else {
			for (uint32_t i = 0; i < m_instances->size(); i++) {
				const auto&           desc = (*m_instances)[i];
				std::array<float, 12> w2o {};
				InvertAffine(desc.object_to_world, w2o);
				WriteInstanceRecord(blob, instances_at + uint64_t(i) * Psr::InstanceStride, w2o,
				                    desc.blas_address, desc.flags, desc.blas_root, desc.mask,
				                    desc.id);
			}
		}
		result.box_nodes = static_cast<uint32_t>(m_nodes.size());
		result.leaves    = static_cast<uint32_t>(m_leaf_groups.size());
		return result;
	}

	static void WriteInstanceRecord(std::vector<uint8_t>& blob, uint64_t at,
	                                const std::array<float, 12>& w2o, uint64_t blas, uint32_t flags,
	                                uint32_t root, uint32_t mask, uint32_t id) {
		using namespace Detail;
		for (int i = 0; i < 12; i++) {
			PutF32(blob, at + i * 4u, w2o[i]);
		}
		const uint64_t shifted = blas >> 3u;
		Put32(blob, at + 48u, static_cast<uint32_t>(shifted));
		Put32(blob, at + 52u,
		      (static_cast<uint32_t>(shifted >> 32u) & Psr::BlasAddressHighMask) |
		          (flags & ~Psr::BlasAddressHighMask));
		Put32(blob, at + 56u, root);
		Put32(blob, at + 60u, (mask << 24u) | (id & 0xffffffu));
	}

	void WriteNode(std::vector<uint8_t>& blob, const Node& node,
	               const std::vector<uint64_t>& leaf_offsets) {
		using namespace Detail;
		if (node.shared_exp) {
			// Implicit pointers: children are contiguous, the first one is the base.
			std::array<uint32_t, 4> types {};
			std::array<uint32_t, 4> fields {Psr::EmptyChildField, Psr::EmptyChildField,
			                                Psr::EmptyChildField, Psr::EmptyChildField};
			std::array<Aabb, 4>     bounds {};
			const auto&             first = m_nodes[node.child_node[0]];
			for (uint32_t slot = 0; slot < node.count; slot++) {
				types[slot]  = static_cast<uint32_t>(m_nodes[node.child_node[slot]].Type());
				fields[slot] = 0;
				bounds[slot] = node.child_bounds[slot];
			}
			const auto bytes = EncodeSharedExpBox(static_cast<uint32_t>(first.offset / 64u), types,
			                                      fields, bounds);
			std::memcpy(blob.data() + node.offset, bytes.data(), bytes.size());
			return;
		}
		for (uint32_t slot = 0; slot < 4; slot++) {
			uint32_t pointer = Psr::InvalidNode;
			if (slot < node.count) {
				if (node.child_node[slot] >= 0) {
					const auto& child = m_nodes[node.child_node[slot]];
					pointer           = Psr::HardwarePointer(child.offset, child.Type());
				} else if (m_leaf_kind == LeafKind::Triangles) {
					pointer =
					    Psr::LeafPointer(leaf_offsets[node.child_leaf[slot]], Psr::NodeType::Leaf0);
				} else {
					// TLAS leaf: instance index in bits 3.., type 0.
					pointer = m_leaf_groups[node.child_leaf[slot]][0] << 3u;
				}
			}
			Put32(blob, node.offset + slot * 4u, pointer);
			const Aabb b = slot < node.count ? node.child_bounds[slot] : Aabb {};
			if (node.box16) {
				for (int a = 0; a < 3; a++) {
					const uint16_t lo = HalfDirected(b.min[a], false);
					const uint16_t hi = HalfDirected(b.max[a], true);
					std::memcpy(blob.data() + node.offset + 16u + slot * 12u + a * 2u, &lo, 2);
					std::memcpy(blob.data() + node.offset + 22u + slot * 12u + a * 2u, &hi, 2);
				}
			} else {
				for (int a = 0; a < 3; a++) {
					PutF32(blob, node.offset + 16u + slot * 24u + a * 4u, b.min[a]);
					PutF32(blob, node.offset + 28u + slot * 24u + a * 4u, b.max[a]);
				}
			}
		}
	}

	// Splits items[begin, end) into up to four children of node `index`.
	void Split(int index, std::vector<Detail::Item>& items, uint32_t begin, uint32_t end,
	           bool root) {
		std::array<std::pair<uint32_t, uint32_t>, 4> ranges {};
		uint32_t                                     count = 0;
		const auto leaf_capacity = m_leaf_kind == LeafKind::Triangles ? m_options.leaf_size : 1u;
		if (end - begin <= 4u * leaf_capacity && end - begin > 0) {
			// Small enough: distribute into leaves directly.
			for (uint32_t at = begin; at < end; at += leaf_capacity) {
				ranges[count++] = {at, std::min(end, at + leaf_capacity)};
			}
		} else {
			const uint32_t mid = SplitMedian(items, begin, end);
			const uint32_t q1  = SplitMedian(items, begin, mid);
			const uint32_t q3  = SplitMedian(items, mid, end);
			ranges             = {{{begin, q1}, {q1, mid}, {mid, q3}, {q3, end}}};
			count              = 4;
		}
		m_nodes[index].box16 = !root && m_options.box16_interior;
		m_nodes[index].count = 0;
		// Create every child node first so siblings are contiguous (type-6 parents need it),
		// then recurse.
		std::array<std::pair<uint32_t, uint32_t>, 4> child_ranges {};
		bool                                         all_nodes = true;
		for (uint32_t c = 0; c < count; c++) {
			const auto [b, e] = ranges[c];
			if (b == e) {
				continue;
			}
			Aabb bounds = Aabb::Empty();
			for (uint32_t i = b; i < e; i++) {
				bounds.Grow(items[i].bounds);
			}
			const uint32_t slot               = m_nodes[index].count++;
			m_nodes[index].child_bounds[slot] = bounds;
			child_ranges[slot]                = {b, e};
			if (e - b <= leaf_capacity) {
				all_nodes                       = false;
				m_nodes[index].child_leaf[slot] = static_cast<int32_t>(m_leaf_groups.size());
				m_leaf_groups.emplace_back();
				for (uint32_t i = b; i < e; i++) {
					m_leaf_groups.back().push_back(items[i].index);
				}
			} else {
				m_nodes[index].child_node[slot] = static_cast<int32_t>(m_nodes.size());
				m_nodes.emplace_back();
			}
		}
		if (!root && m_options.shared_exp_interior && all_nodes && m_nodes[index].count > 0) {
			m_nodes[index].shared_exp = true;
			m_nodes[index].box16      = false;
		}
		for (uint32_t slot = 0; slot < m_nodes[index].count; slot++) {
			if (m_nodes[index].child_node[slot] >= 0) {
				Split(m_nodes[index].child_node[slot], items, child_ranges[slot].first,
				      child_ranges[slot].second, false);
			}
		}
	}

	static uint32_t SplitMedian(std::vector<Detail::Item>& items, uint32_t begin, uint32_t end) {
		if (end - begin < 2) {
			return end;
		}
		Aabb c = Aabb::Empty();
		for (uint32_t i = begin; i < end; i++) {
			c.Grow(items[i].centroid);
		}
		int axis = 0;
		for (int a = 1; a < 3; a++) {
			if (c.max[a] - c.min[a] > c.max[axis] - c.min[axis]) {
				axis = a;
			}
		}
		const uint32_t mid = begin + (end - begin) / 2;
		std::nth_element(items.begin() + begin, items.begin() + mid, items.begin() + end,
		                 [axis](const Detail::Item& a, const Detail::Item& b) {
			                 return a.centroid[axis] < b.centroid[axis];
		                 });
		return mid;
	}

	GuestMemory&                               m_memory;
	BuildOptions                               m_options;
	LeafKind                                   m_leaf_kind = LeafKind::Triangles;
	std::vector<TriNode>                       m_tri_nodes;
	std::vector<std::pair<uint32_t, uint32_t>> m_prims; // (triangle node, pointer type)
	const std::vector<InstanceDesc>*           m_instances = nullptr;
	std::vector<Node>                          m_nodes;
	std::vector<std::vector<uint32_t>>         m_leaf_groups;
};

// Random scene generator for tests and benchmarks.
struct SceneParams {
	uint32_t blas_count         = 4;
	uint32_t triangles_per_blas = 256;
	uint32_t instance_count     = 32;
	float    world_size         = 40.0f;
	float    triangle_size      = 0.25f; // relative to the BLAS unit cube
	uint32_t seed               = 1;
	bool     box16_interior     = false;
	bool     culling_instances  = true;  // some instances cull by facing
	bool     masked_instances   = true;  // some instances have mask 0
	bool     leaf_root_blas     = false; // add one tiny BLAS whose root is a leaf
	bool     fan_nodes          = false; // four triangles per triangle node (types 0..3)
	bool     shared_exp         = false; // type-6 interior nodes where all children are boxes
	// Surface scene: every BLAS is a displaced grid of quads, each quad one fan node of four
	// triangles around its centre (like mesh geometry), and instances tile the XZ plane with a
	// random yaw instead of overlapping at random. Rays then behave like camera and GI rays.
	bool surface = false;
};

struct Scene {
	GuestMemory                             memory;
	std::vector<PsrBvhWriter::Result>       blas;
	std::vector<std::vector<Triangle>>      blas_triangles;
	std::vector<PsrBvhWriter::InstanceDesc> instances;
	PsrBvhWriter::Result                    tlas;
	uint64_t                                total_triangles = 0;
};

inline std::array<float, 12> RandomTransform(std::mt19937& rng, float world, float& scale_out) {
	std::uniform_real_distribution<float> u(-1.0f, 1.0f);
	std::uniform_real_distribution<float> s(0.5f, 4.0f);
	// Random rotation from a random unit quaternion.
	float q[4];
	float n = 0;
	do {
		n = 0;
		for (auto& c: q) {
			c = u(rng);
			n += c * c;
		}
	} while (n < 1e-3f || n > 1.0f);
	n = std::sqrt(n);
	for (auto& c: q) {
		c /= n;
	}
	const float x = q[0], y = q[1], z = q[2], w = q[3];
	const float r[9]  = {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
	                     2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
	                     2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
	const float scale = s(rng) * world / 20.0f;
	scale_out         = scale;
	std::array<float, 12> m {};
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			m[row * 4 + col] = r[row * 3 + col] * scale;
		}
		m[row * 4 + 3] = u(rng) * world * 0.5f;
	}
	return m;
}

inline Scene MakeScene(const SceneParams& params) {
	Scene                                 scene;
	std::mt19937                          rng(params.seed);
	std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
	BuildOptions                          options {};
	options.box16_interior      = params.box16_interior;
	options.shared_exp_interior = params.shared_exp;
	scene.blas_triangles.resize(params.blas_count + (params.leaf_root_blas ? 1 : 0));
	for (uint32_t b = 0; b < params.blas_count; b++) {
		PsrBvhWriter writer(scene.memory, options);
		if (params.surface) {
			// cells x cells quads over [-1, 1]^2 in XZ, heights from a few random waves.
			const uint32_t cells =
			    std::max(1u, static_cast<uint32_t>(
			                     std::sqrt(static_cast<float>(params.triangles_per_blas) / 4.0f)));
			float amp[3], fx[3], fz[3], ph[3];
			for (int k = 0; k < 3; k++) {
				amp[k] = 0.15f * (unit(rng) + 1.2f) / (k + 1);
				fx[k]  = 1.5f * (k + 1) * (unit(rng) + 1.5f);
				fz[k]  = 1.5f * (k + 1) * (unit(rng) + 1.5f);
				ph[k]  = 3.0f * unit(rng);
			}
			const auto height = [&](float x, float z) {
				float y = 0;
				for (int k = 0; k < 3; k++) {
					y += amp[k] * std::sin(fx[k] * x + ph[k]) * std::cos(fz[k] * z - ph[k]);
				}
				return y;
			};
			std::vector<PsrBvhWriter::TriNode> fans(static_cast<size_t>(cells) * cells);
			for (uint32_t j = 0; j < cells; j++) {
				for (uint32_t i = 0; i < cells; i++) {
					const float x0 = -1.0f + 2.0f * static_cast<float>(i) / cells;
					const float x1 = -1.0f + 2.0f * static_cast<float>(i + 1) / cells;
					const float z0 = -1.0f + 2.0f * static_cast<float>(j) / cells;
					const float z1 = -1.0f + 2.0f * static_cast<float>(j + 1) / cells;
					const float xc = 0.5f * (x0 + x1);
					const float zc = 0.5f * (z0 + z1);
					// Rim v0, v1, v3, v4 around the centre v2 (Psr::FanTriangles).
					auto& fan = fans[static_cast<size_t>(j) * cells + i];
					fan[0]    = {x0, height(x0, z0), z0};
					fan[1]    = {x1, height(x1, z0), z0};
					fan[2]    = {xc, height(xc, zc), zc};
					fan[3]    = {x1, height(x1, z1), z1};
					fan[4]    = {x0, height(x0, z1), z1};
				}
			}
			scene.blas.push_back(writer.BuildBlasFans(fans));
		} else if (params.fan_nodes) {
			std::vector<PsrBvhWriter::TriNode> fans(std::max(1u, params.triangles_per_blas / 4u));
			for (auto& fan: fans) {
				std::array<float, 3> c {unit(rng), unit(rng), unit(rng)};
				for (auto& v: fan) {
					for (int a = 0; a < 3; a++) {
						v[a] = c[a] + unit(rng) * params.triangle_size;
					}
				}
			}
			scene.blas.push_back(writer.BuildBlasFans(fans));
		} else {
			std::vector<Triangle> tris(params.triangles_per_blas);
			for (auto& tri: tris) {
				std::array<float, 3> c {unit(rng), unit(rng), unit(rng)};
				for (auto& v: tri.v) {
					for (int a = 0; a < 3; a++) {
						v[a] = c[a] + unit(rng) * params.triangle_size;
					}
				}
			}
			scene.blas.push_back(writer.BuildBlas(tris));
		}
		scene.blas_triangles[b] = scene.blas.back().triangles;
		scene.total_triangles += scene.blas_triangles[b].size();
	}
	if (params.leaf_root_blas) {
		auto& tris = scene.blas_triangles[params.blas_count];
		tris.resize(3);
		for (auto& tri: tris) {
			std::array<float, 3> c {unit(rng) * 0.5f, unit(rng) * 0.5f, unit(rng) * 0.5f};
			for (auto& v: tri.v) {
				for (int a = 0; a < 3; a++) {
					v[a] = c[a] + unit(rng) * 0.4f;
				}
			}
		}
		BuildOptions leaf_options = options;
		leaf_options.leaf_root    = true;
		PsrBvhWriter writer(scene.memory, leaf_options);
		scene.blas.push_back(writer.BuildBlas(tris));
		scene.blas_triangles[params.blas_count] = scene.blas.back().triangles;
		scene.total_triangles += tris.size();
	}
	std::uniform_int_distribution<uint32_t> pick(0, static_cast<uint32_t>(scene.blas.size()) - 1);
	std::uniform_int_distribution<uint32_t> coin(0, 7);
	const uint32_t                          side = std::max(
	    1u, static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<float>(params.instance_count)))));
	for (uint32_t i = 0; i < params.instance_count; i++) {
		PsrBvhWriter::InstanceDesc desc {};
		float                      scale = 1;
		if (params.surface) {
			// Tile the XZ plane: cell (i % side, i / side), random yaw, slight overlap.
			const float cell = params.world_size / side;
			const float yaw  = 3.14159265f * unit(rng);
			const float s    = 0.55f * cell;
			const float cx =
			    (static_cast<float>(i % side) + 0.5f) * cell - 0.5f * params.world_size;
			const float cz =
			    (static_cast<float>(i / side) + 0.5f) * cell - 0.5f * params.world_size;
			desc.object_to_world = {
			    std::cos(yaw) * s,  0, std::sin(yaw) * s, cx, 0, s, 0, 0.05f * cell * unit(rng),
			    -std::sin(yaw) * s, 0, std::cos(yaw) * s, cz};
		} else {
			desc.object_to_world = RandomTransform(rng, params.world_size, scale);
		}
		const auto& blas  = scene.blas[pick(rng)];
		desc.blas_address = blas.base;
		desc.blas_root    = blas.root;
		desc.blas_bounds  = blas.bounds;
		desc.flags        = Psr::CullDisableFlag;
		if (params.culling_instances) {
			const auto c = coin(rng);
			if (c == 1) {
				desc.flags = 0; // keep positive t_num
			} else if (c == 2) {
				desc.flags = Psr::FacingFlag; // keep negative t_num
			}
		}
		desc.mask = params.masked_instances && coin(rng) == 3 ? 0u : 0xffu;
		desc.id   = i;
		scene.instances.push_back(desc);
	}
	PsrBvhWriter writer(scene.memory, options);
	scene.tlas = writer.BuildTlas(scene.instances);
	return scene;
}

struct RayDesc {
	std::array<float, 3> origin;
	std::array<float, 3> dir;
	float                extent;
};

// Incoherent rays through the scene volume; no zero direction components.
inline std::vector<RayDesc> RandomRays(uint32_t count, float world, uint32_t seed) {
	std::mt19937                          rng(seed);
	std::uniform_real_distribution<float> u(-1.0f, 1.0f);
	std::vector<RayDesc>                  rays(count);
	for (auto& ray: rays) {
		for (int a = 0; a < 3; a++) {
			ray.origin[a] = u(rng) * world * 0.6f;
			float d       = u(rng);
			if (std::fabs(d) < 1e-3f) {
				d = 1e-3f;
			}
			ray.dir[a] = d;
		}
		ray.extent = 1e10f;
	}
	return rays;
}

// Camera rays looking down at the surface scene, and short hemisphere rays from points just above
// the surface (like GI or reflection rays).
inline std::vector<RayDesc> SurfaceCameraRays(uint32_t width, uint32_t height, float world) {
	std::vector<RayDesc> rays(static_cast<size_t>(width) * height);
	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			auto&       ray = rays[static_cast<size_t>(y) * width + x];
			const float fx  = (static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2 - 1;
			const float fy  = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2 - 1;
			ray.origin      = {0.013f * world, 0.08f * world, -0.45f * world};
			// Looking along +z, pitched down; fx/fy span the view.
			ray.dir    = {fx * 0.8f + 1e-4f, -0.35f + fy * 0.3f, 1.0f};
			ray.extent = 1e10f;
		}
	}
	return rays;
}

inline std::vector<RayDesc> SurfaceBounceRays(uint32_t count, float world, uint32_t seed) {
	std::mt19937                          rng(seed);
	std::uniform_real_distribution<float> u(-1.0f, 1.0f);
	std::vector<RayDesc>                  rays(count);
	for (auto& ray: rays) {
		ray.origin = {u(rng) * world * 0.45f, 0.02f * world, u(rng) * world * 0.45f};
		float d[3];
		do {
			d[0] = u(rng);
			d[1] = u(rng);
			d[2] = u(rng);
		} while (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > 1.0f || std::fabs(d[0]) < 1e-3f ||
		         std::fabs(d[1]) < 1e-3f || std::fabs(d[2]) < 1e-3f);
		ray.dir    = {d[0], d[1], d[2]};
		ray.extent = 1e10f;
	}
	return rays;
}

// Coherent camera-like rays from outside the scene towards it.
inline std::vector<RayDesc> CameraRays(uint32_t width, uint32_t height, float world) {
	std::vector<RayDesc> rays(static_cast<size_t>(width) * height);
	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			auto&       ray = rays[static_cast<size_t>(y) * width + x];
			const float fx  = (static_cast<float>(x) + 0.5f) / static_cast<float>(width) * 2 - 1;
			const float fy  = (static_cast<float>(y) + 0.5f) / static_cast<float>(height) * 2 - 1;
			ray.origin      = {0.013f * world, 0.021f * world, -1.2f * world};
			ray.dir         = {fx * 0.45f + 1e-4f, fy * 0.45f + 1e-4f, 1.0f};
			ray.extent      = 1e10f;
		}
	}
	return rays;
}

} // namespace RtTest

#endif /* KYTY_TESTS_RTPSRBVHBUILDER_H_ */
