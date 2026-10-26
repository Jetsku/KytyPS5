#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_PSRBVH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_PSRBVH_H_

// Sony Psr ("sce::Psr") BVHs as Astro Bot's ray-tracing shaders walk them, and a CPU walker that
// turns one into the inputs of host (Vulkan) acceleration structures.
//
// Everything modelled here is what the guest traversal routine itself reads. The routine is
// inlined in all 36 BVH shaders of the game (Profiling/analysis/RT-HARDWARE-DESIGN.md, section 2):
//
// - A BVH starts with a 128-byte sce::Psr::internal::qbvh_header. The traversal reads the dword
//   at +24 (bit 18 set: S_TRAP), the u64 at +48 (byte offset of the instance array), the u64 at
//   +88 (T# size field) and the u32 at +96 (V# num_records of the BLAS leaf buffer, stride 8).
// - +0x80 holds a 128-byte "fake instance" record; the root is always pointer 37 (a box32 node
//   at byte 256).
// - Node pointers carry the node type in bits 0..2. Types 4, 5 and 6 are box nodes that go to
//   the hardware box test (hardware encoding: address = base + (ptr & ~7) * 8). 0xffffffff is an
//   empty child.
// - In a BLAS, types 0..3 are leaves: base + (ptr & ~7) is a *byte* offset of a list of 8-byte
//   entries {hardware triangle node pointer, info word}; info bit 31 ends the list and an entry
//   whose pointer has type 7 ends it early. Type 7 leaf pointers are walked the same way.
// - In a TLAS, leaf types 0..3 select instance record (ptr >> 3) of the instance array and type
//   7 selects the fake instance. A record starts with a row-major 3x4 world-to-object matrix,
//   then the BLAS address >> 3 in dwords 12..13 (dword 13 bits 22..31 are flags: bit 24 culling
//   off, bit 25 the facing that is kept), the BLAS root pointer in dword 14 (37 for the fake
//   instance) and the instance mask in the top byte of dword 15 (0 = never visited).
//
// Box node payloads follow RDNA2 for types 4 (box16) and 5 (box32). Type 6 is the PS5 "shared
// exponent" box that Psr builds (psr_scatter_shared_exp_node_*). Its layout and decode are read
// from Psr's own GPU kernels (decoder: eboot raw kernel 0x8ffdb60, pc 0x1a84..; child pointers:
// 0x8ff3730, pc 0x2bc..0x4c0; encoder: 0x8fad010, pc 0xd9c..0x1258). Bit numbers are within the
// 64-byte node:
//   0..28    child base, in 64-byte units from the BVH base
//   29..40   four 3-bit child types
//   41..48   four 2-bit child fields; 3 = empty child (pointer 0xffffffff)
//   49..55   unknown (the encoder fills them from a per-node parameter)
//   56..79   8-bit shared exponents x, y, z (the largest biased exponent of any bound on the axis)
//   80..511  per child: 18-bit min x, min y, min z, max x, max y, max z; sign at bit 17, then a
//            17-bit magnitude aligned to the shared exponent (truncated by the encoder)
// Child pointers are implicit: child i = ((base + units of children 0..i-1) << 3) | type_i, where
// a type-2 child spans 4 units, types 3, 5 and 6 span 2, all others 1. The decode rounds every
// bound outwards (see DecodeSharedExpBound). Whether the hardware decodes exactly like Psr's
// software is not proven; the exact instruction emulation uses this same decode.
//
// A triangle node holds five float3 vertices at +0, +12, +24, +36 and +48; pointer types 0..3
// select one of four triangles that all share v2 (a fan). The vertex sets come from the GI probe
// shader, which rebuilds the hit triangle's normal from the node itself: t0 = (v0,v1,v2),
// t1 = (v1,v3,v2), t2 = (v2,v3,v4), t3 = (v2,v4,v0) in the winding of that normal. The winding the
// hardware uses for the sign of t_num is not proven by this.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::Rt {

namespace Psr {

inline constexpr uint32_t InvalidNode          = 0xffffffffu;
inline constexpr uint32_t RootPointer          = 37u;
inline constexpr uint64_t HeaderBytes          = 128u;
inline constexpr uint64_t FakeInstanceOffset   = 0x80u;
inline constexpr uint64_t RootNodeOffset       = 256u;
inline constexpr uint64_t InstanceStride       = 128u;
inline constexpr uint64_t LeafEntryBytes       = 8u;
inline constexpr uint64_t Box16Bytes           = 64u;
inline constexpr uint64_t Box32Bytes           = 128u;
inline constexpr uint64_t BoxPs5Bytes          = 64u; // decoded bytes; the node spans 2 units
inline constexpr uint32_t EmptyChildField      = 3u;
inline constexpr uint64_t TriangleNodeBytes    = 64u;
inline constexpr uint64_t HeaderFlagsOffset    = 24u;
inline constexpr uint32_t HeaderTrapBit        = 1u << 18u;
inline constexpr uint64_t HeaderInstancesAt    = 48u;
inline constexpr uint64_t HeaderNodeSizeAt     = 88u;
inline constexpr uint64_t HeaderLeafRecordsAt  = 96u;
inline constexpr uint32_t CullDisableFlag      = 1u << 24u;
inline constexpr uint32_t FacingFlag           = 1u << 25u;
inline constexpr uint32_t BlasAddressHighMask  = 0x003fffffu;
inline constexpr uint32_t LeafLastFlag         = 1u << 31u;
inline constexpr uint32_t FakeInstanceIndex    = 0x00ffffffu;
inline constexpr uint32_t TriangleNodeVertices = 5u;

// Vertex slots of the four triangles of a triangle node (pointer types 0..3).
inline constexpr std::array<std::array<uint32_t, 3>, 4> FanTriangles {
    {{0u, 1u, 2u}, {1u, 3u, 2u}, {2u, 3u, 4u}, {2u, 4u, 0u}}};

enum class NodeType : uint32_t {
	Leaf0   = 0,
	Leaf1   = 1,
	Leaf2   = 2,
	Leaf3   = 3,
	Box16   = 4,
	Box32   = 5,
	BoxPs5  = 6,
	Special = 7,
};

constexpr NodeType TypeOf(uint32_t pointer) {
	return static_cast<NodeType>(pointer & 7u);
}
constexpr bool IsBox(uint32_t pointer) {
	const auto type = pointer & 7u;
	return pointer != InvalidNode && type >= 4u && type <= 6u;
}
// Hardware node pointer: byte offset / 8 with the type in bits 0..2 (64-byte aligned nodes).
constexpr uint64_t HardwareNodeAddress(uint64_t base, uint32_t pointer) {
	return base + (static_cast<uint64_t>(pointer & ~7u) << 3u);
}
constexpr uint32_t HardwarePointer(uint64_t byte_offset, NodeType type) {
	return static_cast<uint32_t>(byte_offset >> 3u) | static_cast<uint32_t>(type);
}
// Leaf pointer: the byte offset of the entry list itself, 8-byte aligned, type in bits 0..2.
constexpr uint64_t LeafListAddress(uint64_t base, uint32_t pointer) {
	return base + static_cast<uint64_t>(pointer & ~7u);
}
constexpr uint32_t LeafPointer(uint64_t byte_offset, NodeType type) {
	return static_cast<uint32_t>(byte_offset) | static_cast<uint32_t>(type);
}
// 64-byte units a child of a type-6 node spans, by its child type.
constexpr uint32_t SharedExpChildUnits(uint32_t type) {
	return type == 2u ? 4u : (type == 3u || type == 5u || type == 6u) ? 2u : 1u;
}

} // namespace Psr

struct Aabb {
	std::array<float, 3> min {};
	std::array<float, 3> max {};

	static Aabb Empty() {
		constexpr float inf = std::numeric_limits<float>::infinity();
		return {{inf, inf, inf}, {-inf, -inf, -inf}};
	}
	void               Grow(const std::array<float, 3>& point);
	void               Grow(const Aabb& other);
	[[nodiscard]] bool IsEmpty() const { return min[0] > max[0]; }
};

struct BoxNode {
	std::array<uint32_t, 4> children {};
	std::array<Aabb, 4>     bounds {};
	std::array<uint32_t, 4> fields {}; // type 6: the 2-bit child fields (0 for box16/box32)
};

struct Triangle {
	std::array<std::array<float, 3>, 3> v {};
};

struct LeafEntry {
	uint32_t triangle_pointer = 0;
	uint32_t info             = 0;
};

// One 128-byte instance record as the traversal decodes it.
struct InstanceRecord {
	uint64_t              address = 0;
	uint32_t              index   = 0; // TLAS leaf pointer >> 3, or Psr::FakeInstanceIndex
	std::array<float, 12> world_to_object {};
	uint64_t              blas_address = 0;
	uint32_t              flags        = 0; // dword 13 bits 22..31
	uint32_t              root         = 0;
	uint32_t              id_mask      = 0; // dword 15; the mask is the top byte
	bool                  fake         = false;

	[[nodiscard]] uint32_t Mask() const { return id_mask >> 24u; }
};

// Reads guest bytes; returns false when the range is not readable.
using ReadMemory = std::function<bool(uint64_t address, void* dst, size_t size)>;

enum class WalkStatus : uint32_t {
	Ok = 0,
	ReadFailed,
	UnsupportedNode,
	TooManyNodes,
	BadPointer,
	TrapFlag,
};
const char* ToString(WalkStatus status);

// Decodes a box node payload (types 4, 5 and 6). Returns false for other types.
bool DecodeBox(Psr::NodeType type, std::span<const uint8_t> bytes, BoxNode& out);

// One bound of a type-6 node: 18-bit sign-magnitude `field` with the axis' shared `exponent`.
// Minima round towards -inf and maxima towards +inf (Psr's decoder fills the truncated mantissa
// bits with ones when rounding away from zero). Returns the float bits.
uint32_t DecodeSharedExpBound(uint32_t field, uint32_t exponent, bool is_min);
// Psr's encoder for one bound (truncates the magnitude); `exponent` is the axis' shared exponent.
uint32_t EncodeSharedExpBound(float value, uint32_t exponent);
// Builds a 64-byte type-6 node the way Psr's encoder does (for test fixtures).
std::array<uint8_t, 64> EncodeSharedExpBox(uint32_t                       child_base,
                                           const std::array<uint32_t, 4>& child_types,
                                           const std::array<uint32_t, 4>& child_fields,
                                           const std::array<Aabb, 4>&     bounds);
// Decodes the triangle a pointer of type 0..3 selects from a triangle node.
bool DecodeTriangle(Psr::NodeType type, std::span<const uint8_t> bytes, Triangle& out);

bool ReadBox(const ReadMemory& read, uint64_t base, uint32_t pointer, BoxNode& out,
             WalkStatus& status);

// Reads the entry list of one leaf, stopping after the entry with bit 31 set or before an entry
// whose pointer has type 7, like the traversal does.
WalkStatus ReadLeafEntries(const ReadMemory& read, uint64_t blas_base, uint32_t leaf_pointer,
                           std::vector<LeafEntry>& out, uint32_t max_entries = 1024);

WalkStatus ReadInstance(const ReadMemory& read, uint64_t tlas_base, uint32_t leaf_pointer,
                        InstanceRecord& out);

// A BLAS as a list of leaves. Each leaf keeps the bounds its parent box node stores for it, so a
// host procedural BLAS over these boxes finds every leaf the guest box tests can reach.
struct BlasLeaf {
	uint32_t pointer = 0;
	Aabb     bounds {};
	bool     bounds_from_triangles = false; // root leaf: bounds of its decoded triangles
};

struct BlasWalk {
	std::vector<BlasLeaf> leaves;
	uint32_t              box_nodes        = 0;
	uint32_t              shared_exp_nodes = 0; // type 6
	uint32_t              max_depth        = 0;
};

struct WalkLimits {
	uint32_t max_nodes          = 1u << 24u;
	uint32_t max_leaf_entries   = 1024u;
	uint32_t max_instances      = 1u << 20u;
	bool     reject_trap_header = true;
};

WalkStatus WalkBlas(const ReadMemory& read, uint64_t blas_base, uint32_t root, BlasWalk& out,
                    const WalkLimits& limits = {});

// Expands every leaf entry of a walked BLAS into triangles (triangle conversion mode).
struct BlasTriangles {
	std::vector<Triangle>  triangles;
	std::vector<LeafEntry> entries; // same order: the entry that produced each triangle
	std::vector<uint32_t>  leaf_of; // index into BlasWalk::leaves
};
WalkStatus ExpandTriangles(const ReadMemory& read, uint64_t blas_base, const BlasWalk& walk,
                           BlasTriangles& out, const WalkLimits& limits = {});

struct TlasWalk {
	std::vector<InstanceRecord> instances; // reachable instances, in first-reached order
	uint32_t                    box_nodes = 0;
};
WalkStatus WalkTlas(const ReadMemory& read, uint64_t tlas_base, TlasWalk& out,
                    const WalkLimits& limits = {});

// Row-major 3x4 inverse (object-to-world from world-to-object), in double precision.
bool InvertAffine(const std::array<float, 12>& m, std::array<float, 12>& out);

// ---------------------------------------------------------------------------------------------
// CPU reference traversal with Psr semantics (per ray; the guest's wave-coherent packet
// traversal gives each lane the same closest hit up to ties). Node tests are pluggable so the
// exact software implementation of the instruction can be dropped in; the defaults are a float
// slab test and a Moller-Trumbore triangle test that the GPU test kernels mirror operation for
// operation.

struct Ray {
	std::array<float, 3> origin {};
	std::array<float, 3> dir {};
	std::array<float, 3> inv_dir {};
	float                extent = 1e10f;
};

struct TriangleHit {
	float t        = 0.0f;
	bool  positive = false; // sign of the hardware t_num: compared with the instance facing flag
	float u        = 0.0f;
	float v        = 0.0f;
};

struct NodeTests {
	// Child pointers ordered by entry distance, misses as Psr::InvalidNode.
	std::function<std::array<uint32_t, 4>(const BoxNode&, const Ray&)>     box;
	std::function<std::optional<TriangleHit>(const Triangle&, const Ray&)> triangle;
};
NodeTests DefaultNodeTests();

std::array<uint32_t, 4> ReferenceBoxTest(const BoxNode& node, const Ray& ray);
// Moller-Trumbore with sign-corrected numerators (the prototype's first test; kept for comparison).
std::optional<TriangleHit> ReferenceTriangleTest(const Triangle& tri, const Ray& ray);
// The instruction's exact triangle test (RT-SW's BvhReference::Triangle, return mode 1, IEEE
// denormals as on the RTX 3090) followed by the guest's use of it: a hit unless t_num is +inf,
// t = t_num * V_RCP_F32(t_den) with a correctly rounded reciprocal, positive = sign bit of t_num
// clear. This is DefaultNodeTests().triangle.
std::optional<TriangleHit> ExactTriangleTest(const Triangle& tri, const Ray& ray);
Ray MakeRay(const std::array<float, 3>& origin, const std::array<float, 3>& dir, float extent);
// The guest's object-space ray: the same multiply/add order as the traversal routine.
Ray TransformRay(const Ray& world, const std::array<float, 12>& world_to_object);

struct TraceResult {
	bool     hit              = false;
	float    t                = 0.0f;
	uint32_t info             = 0xffffffffu;
	uint32_t instance_index   = 0xffffffffu;
	uint64_t instance_address = 0;
	uint32_t triangle_pointer = 0;
	uint32_t nodes_visited    = 0;
};

WalkStatus TraceReference(const ReadMemory& read, uint64_t tlas_base, const Ray& ray,
                          const NodeTests& tests, TraceResult& out, uint32_t stack_limit = 60);

// The T# the traversal builds for a Psr BVH: base = header >> 8, box sorting on, growing 0,
// triangle return mode 1, type 8. `size` is header +88 for the TLAS and header +88 - 1 for a BLAS.
std::array<uint32_t, 4> GuestTsharp(uint64_t bvh_base, uint64_t size);

// TraceReference with every node test done by the instruction itself (RT-SW's
// BvhReference::Intersect on the raw node, with the guest's T#s): the hardware box tests
// (box16/box32/type 6 decode, slab test, sorting network) and the exact triangle test. Per ray it
// is what the software path computes, up to the guest's wave-union of children.
WalkStatus TraceGuestExact(const ReadMemory& read, uint64_t tlas_base, const Ray& ray,
                           TraceResult& out, uint32_t stack_limit = 60);

} // namespace Libs::Graphics::Rt

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RT_PSRBVH_H_ */
