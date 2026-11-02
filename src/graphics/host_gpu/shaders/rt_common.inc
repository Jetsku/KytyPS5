// Native ray tracing (KYTY_RT_HW, see nativeRayTracing.h; adapted from TheCruZ's
// native-hardware-raytracing branch): shared definitions of the passes that read the
// guest's AMD BVHs through the BDA page table and write the host acceleration structures.

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require

#ifndef CACHING_PAGEBITS
#error CACHING_PAGEBITS must be defined
#endif

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Ref { uint v; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer U32x4Ref { uvec4 v; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer U64Ref { uint64_t v; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32Ref { float v; };

// The BufferCache device-address page table: the host address of each caching page (0 when the
// page has no cached buffer).
layout(set = 0, binding = 0, std430) readonly buffer PageTable {
	uint64_t pages[];
};

const uint     kPageMask    = (1u << CACHING_PAGEBITS) - 1u;
// The 40-bit guest space the table covers (BufferCache::CACHING_NUMPAGES); the entries after it
// (the native ray tracing context) are no guest pages.
const uint64_t kGuestPages  = uint64_t(1) << (40 - CACHING_PAGEBITS);
const uint     kInvalidNode = 0xffffffffu;

// The host address of guest address `va`, 0 when it has no cached buffer.
uint64_t HostAddress(uint64_t va) {
	const uint64_t page = va >> CACHING_PAGEBITS;
	if (page >= kGuestPages || page >= uint64_t(pages.length())) {
		return 0;
	}
	const uint64_t base = pages[uint(page)];
	return base == 0 ? 0 : base + (va & uint64_t(kPageMask));
}

// Guest loads (zero where the memory is not cached). Aligned 16-byte loads never cross a caching
// page.
uint LoadGuestU32(uint64_t va) {
	const uint64_t host = HostAddress(va);
	return host == 0 ? 0u : U32Ref(host).v;
}

uvec4 LoadGuestU32x4(uint64_t va) {
	const uint64_t host = HostAddress(va);
	return host == 0 ? uvec4(0) : U32x4Ref(host).v;
}

uint64_t LoadGuestU64(uint64_t va) {
	return uint64_t(LoadGuestU32(va)) | (uint64_t(LoadGuestU32(va + 4)) << 32);
}

// An AMD BVH node pointer: the node type in the low 3 bits, the 64-byte node index above.
uint NodeType(uint node) {
	return node & 7u;
}

uint64_t NodeAddress(uint64_t base, uint node) {
	return base + (uint64_t(node >> 3) << 6);
}

bool IsBoxNode(uint node) {
	return node != kInvalidNode && (NodeType(node) == 4u || NodeType(node) == 5u);
}

// A box node: four child pointers and their bounds (box16: fp16 pairs, box32: floats).
struct BoxNode {
	uint children[4];
	vec3 lo[4];
	vec3 hi[4];
};

BoxNode LoadBoxNode(uint64_t base, uint node) {
	const uint64_t address = NodeAddress(base, node);
	BoxNode        box;
	const uvec4    first = LoadGuestU32x4(address);
	box.children[0] = first.x;
	box.children[1] = first.y;
	box.children[2] = first.z;
	box.children[3] = first.w;
	if (NodeType(node) == 4u) {
		const uvec4 a = LoadGuestU32x4(address + 16);
		const uvec4 b = LoadGuestU32x4(address + 32);
		const uvec4 c = LoadGuestU32x4(address + 48);
		const uint words[12] = uint[12](a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w);
		for (uint child = 0; child < 4; child++) {
			const vec2 xy = unpackHalf2x16(words[child * 3]);
			const vec2 zx = unpackHalf2x16(words[child * 3 + 1]);
			const vec2 yz = unpackHalf2x16(words[child * 3 + 2]);
			box.lo[child] = vec3(xy.x, xy.y, zx.x);
			box.hi[child] = vec3(zx.y, yz.x, yz.y);
		}
	} else {
		uint words[24];
		for (uint i = 0; i < 6; i++) {
			const uvec4 v = LoadGuestU32x4(address + 16 + i * 16);
			words[i * 4]     = v.x;
			words[i * 4 + 1] = v.y;
			words[i * 4 + 2] = v.z;
			words[i * 4 + 3] = v.w;
		}
		for (uint child = 0; child < 4; child++) {
			box.lo[child] = uintBitsToFloat(uvec3(words[child * 6], words[child * 6 + 1], words[child * 6 + 2]));
			box.hi[child] = uintBitsToFloat(uvec3(words[child * 6 + 3], words[child * 6 + 4], words[child * 6 + 5]));
		}
	}
	return box;
}

// The state of one preparation (NativeRayTracing::Prepare), cleared before the walks.
layout(buffer_reference, std430, buffer_reference_align = 8) buffer RtState {
	uint64_t header_va;
	uint64_t instance_base;
	uint     node_limit;
	uint     instance_count;
	uint     box_count;
	uint     overflow;
	uint     missing_count;
	uint     dispatch_x;
	uint     dispatch_y;
	uint     dispatch_z;
	uint     unknown_blas;
	uint     inactive;
	uint     max_depth;
	uint     verify_count;
	uint     verify_x;
	uint     verify_y;
	uint     verify_z;
	uint     changed_count;
	uint     pool_cursor;
	uint     pool_overflow;
	uint     pad0;
	uint     pad1;
};

// A host BLAS: the guest BLAS it mirrors and its acceleration structure of triangles, in two
// geometries: 0 = opaque, 1 = alpha tested. Per primitive (geometry 1 after the capacity of
// geometry 0), the entry table holds (16 bytes) the guest leaf pointer of its list entry, its primitive
// word, its triangle node and the barycentric remap of the node (see rt_blas_walk.comp).
layout(buffer_reference, std430, buffer_reference_align = 8) buffer SlotInfo {
	uint64_t as_address;
	uint64_t entry_table;
	uint     capacity0;
	uint     capacity1;
	uint     count0;
	uint     count1;
	uint     signature;
	uint     state;
	uint64_t blas_va;
	uint     root;
	uint     stamp;
	uint     leaf_count;
	// Set by a gathered build (see MODE_GATHER in rt_blas_walk.comp) to bit 31 and the preparation:
	// the triangles follow the guest's refit boxes, so the check of that same preparation takes their
	// signature instead of reporting a change. A later check (the guest may have changed the BLAS
	// since) compares as usual.
	uint     rebase;
};

// A guest BLAS the top-level walk did not find, counted by the BLAS walk.
layout(buffer_reference, std430, buffer_reference_align = 8) buffer MissingEntry {
	uint64_t blas_va;
	uint     root;
	uint     table_index;
	uint     leaf_count;
	uint     box_count;
	uint     opaque_count;
	uint     alpha_count;
	uint     node_limit;
	uint     flags;
	uint     pad0;
	uint     pad1;
};

const uint kSlotStride    = 64;
const uint kMissingStride = 48;
const uint kSlotBuilt     = 1u;
// Entries of one guest leaf list at most.
const uint kMaxLeafEntries = 64;

// The vertices of triangle `type` (0-3) of an AMD triangle node with five vertices: (0,1,2),
// (1,3,2), (2,3,4), (2,4,0).
uvec3 TriangleVertexIds(uint type) {
	return uvec3(min(type, 2u), type == 0u ? 1u : (type == 3u ? 4u : 3u),
	             type == 2u ? 4u : (type == 3u ? 0u : 2u));
}

vec3 LoadVertex(uint64_t node, uint id) {
	return uintBitsToFloat(uvec3(LoadGuestU32(node + id * 12), LoadGuestU32(node + id * 12 + 4),
	                             LoadGuestU32(node + id * 12 + 8)));
}

// A cheap mix of 32-bit words for signatures.
uint Mix(uint hash, uint value) {
	hash ^= value + 0x9e3779b9u + (hash << 6) + (hash >> 2);
	return hash;
}
