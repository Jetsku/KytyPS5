// Tests for the optional hardware ray-tracing path: Psr BVH walking and conversion,
// host acceleration structures and ray queries, checked hit for hit against a CPU reference of
// the guest traversal.
//
//   rt_hardware_tests              CPU tests, then GPU tests (skipped without an RT device)
//   rt_hardware_tests --cpu-only   CPU tests only
//   rt_hardware_tests --bench      benchmark scenes (build and trace timings)
//   rt_hardware_tests --bvh-dump <file.kbvh> [--rays N]
//                                  offline analysis of a KYTY_RT_DUMP_BVH dump of the game's BVHs
//   rt_hardware_tests --write-test-dump <file.kbvh>
//                                  a dump of a synthetic scene, to exercise --bvh-dump

#include "RtPsrBvhBuilder.h"
#include "graphics/host_gpu/rt/bvhDump.h"
#include "graphics/host_gpu/rt/psrBvh.h"
#include "graphics/host_gpu/rt/rtAccel.h"
#include "graphics/shader/recompiler/BvhReference.h"
#include "rt_psr_convert_spv.h"
#include "rt_psr_exact_test_spv.h"
#include "rt_psr_trace_hw_spv.h"
#include "rt_psr_trace_hw_tri_spv.h"
#include "rt_psr_trace_sw_spv.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {

using namespace Libs::Graphics::Rt;
using RtTest::GuestMemory;
using RtTest::RayDesc;
using RtTest::Scene;
using RtTest::SceneParams;

int g_failures = 0;

void Check(bool condition, const char* what) {
	if (!condition) {
		std::printf("FAILED: %s\n", what);
		++g_failures;
	}
}

Ray ToRay(const RayDesc& desc) {
	return MakeRay(desc.origin, desc.dir, desc.extent);
}

// ---------------------------------------------------------------------------------------------
// CPU tests

void TestPointers() {
	using namespace Psr;
	Check(HardwarePointer(RootNodeOffset, NodeType::Box32) == RootPointer,
	      "root pointer is box32 at byte 256");
	Check(HardwareNodeAddress(0x1000, RootPointer) == 0x1000 + 256, "hardware node address");
	Check(IsBox(RootPointer) && IsBox(HardwarePointer(64, NodeType::Box16)) &&
	          IsBox(HardwarePointer(64, NodeType::BoxPs5)),
	      "types 4..6 are boxes");
	Check(!IsBox(InvalidNode) && !IsBox(LeafPointer(8, NodeType::Leaf0)) &&
	          !IsBox(LeafPointer(8, NodeType::Special)),
	      "leaves and the empty child are not boxes");
	Check(LeafListAddress(0x1000, LeafPointer(0x238, NodeType::Leaf1)) == 0x1238,
	      "leaf pointers are byte offsets");
}

void TestDecode() {
	std::array<uint8_t, 128> box32 {};
	const uint32_t           children[4] = {0x25u, 0xffffffffu, 0x48u, 0x1234u};
	std::memcpy(box32.data(), children, sizeof(children));
	const float bounds[6] = {-1.0f, -2.0f, -3.0f, 4.0f, 5.0f, 6.0f};
	std::memcpy(box32.data() + 16 + 24 * 2, bounds, sizeof(bounds));
	BoxNode node {};
	Check(DecodeBox(Psr::NodeType::Box32, box32, node), "box32 decodes");
	Check(node.children[2] == 0x48u && node.bounds[2].min[1] == -2.0f &&
	          node.bounds[2].max[2] == 6.0f,
	      "box32 child 2 fields");

	std::array<uint8_t, 64> box16 {};
	std::memcpy(box16.data(), children, sizeof(children));
	// Child 1: min (1, -2, 0.5), max (2, 65504, -0).
	const uint16_t halves[6] = {0x3c00u, 0xc000u, 0x3800u, 0x4000u, 0x7bffu, 0x8000u};
	std::memcpy(box16.data() + 16 + 12, halves, sizeof(halves));
	Check(DecodeBox(Psr::NodeType::Box16, box16, node), "box16 decodes");
	Check(node.bounds[1].min[0] == 1.0f && node.bounds[1].min[1] == -2.0f &&
	          node.bounds[1].min[2] == 0.5f && node.bounds[1].max[0] == 2.0f &&
	          node.bounds[1].max[1] == 65504.0f && std::signbit(node.bounds[1].max[2]),
	      "box16 fp16 bounds");
	Check(!DecodeBox(Psr::NodeType::Special, box32, node), "type 7 is not a box");

	std::array<uint8_t, 64> tri {};
	const float             verts[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	std::memcpy(tri.data(), verts, sizeof(verts));
	Triangle decoded {};
	Check(DecodeTriangle(Psr::NodeType::Leaf0, tri, decoded) && decoded.v[1][2] == 5.0f,
	      "fan triangle 0 is (v0, v1, v2)");
	std::array<float, 15> fan {};
	for (int i = 0; i < 15; i++) {
		fan[i] = static_cast<float>(i);
	}
	std::memcpy(tri.data(), fan.data(), sizeof(fan));
	// Vertex k has x = 3k: triangle t must be made of the slots in Psr::FanTriangles[t].
	const float expect[4][3] = {{0, 3, 6}, {3, 9, 6}, {6, 9, 12}, {6, 12, 0}};
	bool        fan_ok       = true;
	for (uint32_t t = 0; t < 4; t++) {
		fan_ok = fan_ok && DecodeTriangle(static_cast<Psr::NodeType>(t), tri, decoded);
		for (int v = 0; v < 3; v++) {
			fan_ok = fan_ok && decoded.v[v][0] == expect[t][v];
		}
	}
	Check(fan_ok, "fan triangles 0..3 select the guest's vertex slots");
	Check(!DecodeTriangle(Psr::NodeType::Box16, tri, decoded), "box types are not triangles");
}

// Type-6 (Psr shared exponent) bounds: vectors worked by hand from Psr's decoder kernel.
void TestSharedExpBounds() {
	struct Vector {
		uint32_t field, exponent;
		bool     is_min;
		uint32_t bits;
	};
	const Vector vectors[] = {
	    {0x10000u, 127, true, 0x3f800000u},  // 1.0, min: exact
	    {0x10000u, 127, false, 0x3f80007fu}, // 1.0, max: truncated bits filled
	    {0x30000u, 127, true, 0xbf80007fu},  // -1.0, min: rounds away from zero
	    {0x30000u, 127, false, 0xbf800000u}, // -1.0, max: exact
	    {0x00001u, 127, true, 0x37800000u},  // 2^-16, renormalised
	    {0x00001u, 127, false, 0x37ffffffu},
	    {0x00000u, 127, true, 0x00000000u},  // zero magnitude
	    {0x00000u, 127, false, 0x377fffffu}, // just below one quantum
	    {0x20000u, 127, true, 0xb77fffffu},  // negative zero magnitude, min: rounds away
	    {0x20000u, 127, false, 0x80000000u},
	    {0x00001u, 10, true, 0x00020000u}, // exponent too small: denormal
	    {0x00001u, 10, false, 0x0003ffffu},
	    {0x00000u, 10, false, 0x007fffffu},
	    {0x10000u, 255, false, 0x7f800000u}, // exponent 255: no fill
	    {0x1ffffu, 130, false, 0x417fffffu}, // largest magnitude: 1.99998 * 8, filled
	    {0x1ffffu, 130, true, 0x417fff80u},
	};
	bool ok = true;
	for (const auto& v: vectors) {
		const uint32_t got = DecodeSharedExpBound(v.field, v.exponent, v.is_min);
		if (got != v.bits) {
			std::printf("  shared-exp field %05x exp %u %s: got %08x want %08x\n", v.field,
			            v.exponent, v.is_min ? "min" : "max", got, v.bits);
			ok = false;
		}
	}
	Check(ok, "shared-exponent bound vectors");

	// Encoder then decoder: conservative, within a quantum, implicit child pointers.
	std::mt19937                            rng(11);
	std::uniform_real_distribution<float>   coord(-200.0f, 200.0f);
	std::uniform_real_distribution<float>   extent(0.0f, 30.0f);
	std::uniform_int_distribution<uint32_t> type_pick(0, 7);
	uint32_t                                bad_bounds = 0, bad_children = 0, loose = 0;
	for (int n = 0; n < 20000; n++) {
		std::array<Aabb, 4>     boxes {};
		std::array<uint32_t, 4> types {};
		std::array<uint32_t, 4> fields {};
		for (uint32_t c = 0; c < 4; c++) {
			for (int a = 0; a < 3; a++) {
				const float lo  = coord(rng) * (n % 3 == 0 ? 1e-3f : 1.0f);
				boxes[c].min[a] = lo;
				boxes[c].max[a] = lo + extent(rng);
			}
			types[c]  = type_pick(rng);
			fields[c] = (n + c) % 7 == 0 ? Psr::EmptyChildField : (n + c) % 3;
		}
		const uint32_t base  = (static_cast<uint32_t>(n) * 7919u) & 0x1fffffu;
		const auto     bytes = EncodeSharedExpBox(base, types, fields, boxes);
		BoxNode        node {};
		DecodeBox(Psr::NodeType::BoxPs5, bytes, node);
		uint32_t running  = base;
		float    scale[3] = {0, 0, 0};
		for (uint32_t c = 0; c < 4; c++) {
			if (fields[c] != Psr::EmptyChildField) {
				for (int a = 0; a < 3; a++) {
					scale[a] = std::max(
					    {scale[a], std::fabs(boxes[c].min[a]), std::fabs(boxes[c].max[a])});
				}
			}
		}
		for (uint32_t c = 0; c < 4; c++) {
			const uint32_t want =
			    fields[c] == Psr::EmptyChildField ? Psr::InvalidNode : ((running << 3u) | types[c]);
			running += Psr::SharedExpChildUnits(types[c]);
			if (node.children[c] != want || node.fields[c] != fields[c]) {
				bad_children++;
			}
			if (fields[c] == Psr::EmptyChildField) {
				continue;
			}
			for (int a = 0; a < 3; a++) {
				if (!(node.bounds[c].min[a] <= boxes[c].min[a] &&
				      node.bounds[c].max[a] >= boxes[c].max[a])) {
					bad_bounds++;
				}
				// A quantum is 2^-16 of the axis' largest magnitude (2^-15 for the fill).
				const float q = scale[a] * (1.0f / 16384.0f) + 1e-30f;
				if (boxes[c].min[a] - node.bounds[c].min[a] > q ||
				    node.bounds[c].max[a] - boxes[c].max[a] > q) {
					loose++;
				}
			}
		}
	}
	Check(bad_children == 0, "shared-exponent implicit child pointers and fields");
	Check(bad_bounds == 0, "shared-exponent decode is conservative");
	Check(loose == 0, "shared-exponent decode stays within a few quanta");
}

void TestConservativeHalves() {
	std::mt19937                          rng(7);
	std::uniform_real_distribution<float> u(-3000.0f, 3000.0f);
	uint32_t                              bad = 0;
	for (int i = 0; i < 20000; i++) {
		const float             value = i < 10 ? static_cast<float>(i) * 1e-6f : u(rng);
		const auto              lo    = RtTest::Detail::HalfDirected(value, false);
		const auto              hi    = RtTest::Detail::HalfDirected(value, true);
		std::array<uint8_t, 64> bytes {};
		std::memcpy(bytes.data() + 16, &lo, 2);
		std::memcpy(bytes.data() + 22, &hi, 2);
		BoxNode node {};
		DecodeBox(Psr::NodeType::Box16, bytes, node);
		if (!(node.bounds[0].min[0] <= value && value <= node.bounds[0].max[0])) {
			bad++;
		}
	}
	Check(bad == 0, "box16 builder bounds are conservative");
}

// Brute force over every instance and triangle with the guest's acceptance rule.
TraceResult BruteForce(const Scene& scene, const std::vector<BlasTriangles>& expanded,
                       const std::map<uint64_t, size_t>& blas_index, const TlasWalk& tlas,
                       const Ray& world) {
	TraceResult best {};
	float       best_t = world.extent;
	for (const auto& inst: tlas.instances) {
		if (inst.Mask() == 0u) {
			continue;
		}
		const Ray   obj      = TransformRay(world, inst.world_to_object);
		const auto& tris     = expanded[blas_index.at(inst.blas_address)];
		const bool  cull_off = (inst.flags & Psr::CullDisableFlag) != 0u;
		const bool  keep_neg = (inst.flags & Psr::FacingFlag) != 0u;
		for (size_t i = 0; i < tris.triangles.size(); i++) {
			const auto hit = ExactTriangleTest(tris.triangles[i], obj);
			if (!hit || !(cull_off || hit->positive != keep_neg) || hit->t > best_t) {
				continue;
			}
			if (hit->t < best_t || !best.hit) {
				best_t                = hit->t;
				best.hit              = true;
				best.t                = hit->t;
				best.info             = tris.entries[i].info;
				best.instance_index   = inst.index;
				best.triangle_pointer = tris.entries[i].triangle_pointer;
			}
		}
	}
	(void)scene;
	return best;
}

// Rays aimed at triangle vertices, at points on edges and just beside edges, from random
// directions: they stress the exact test's boundary decisions and the leaf boxes' margins.
std::vector<RayDesc> EdgeRays(const Scene& scene, uint32_t count, uint32_t seed) {
	std::mt19937                          rng(seed);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);
	std::normal_distribution<float>       normal(0.0f, 1.0f);
	std::map<uint64_t, size_t>            index;
	for (size_t b = 0; b < scene.blas.size(); b++) {
		index[scene.blas[b].base] = b;
	}
	std::vector<RayDesc> rays;
	rays.reserve(count);
	while (rays.size() < count) {
		const auto& inst = scene.instances[rng() % scene.instances.size()];
		const auto  it   = index.find(inst.blas_address);
		if (it == index.end() || scene.blas_triangles[it->second].empty()) {
			continue;
		}
		const auto&          tris = scene.blas_triangles[it->second];
		const auto&          tri  = tris[rng() % tris.size()];
		const uint32_t       kind = rng() % 3u; // vertex, on an edge, beside an edge
		const uint32_t       edge = rng() % 3u;
		const auto&          p    = tri.v[edge];
		const auto&          q    = tri.v[(edge + 1u) % 3u];
		const float          s    = kind == 0u ? 0.0f : unit(rng);
		std::array<float, 3> target {};
		for (int a = 0; a < 3; a++) {
			target[a] = p[a] + s * (q[a] - p[a]);
			if (kind == 2u) {
				target[a] += (unit(rng) - 0.5f) * 1e-5f * (1.0f + std::fabs(target[a]));
			}
		}
		const auto&          m = inst.object_to_world;
		std::array<float, 3> world {};
		for (int row = 0; row < 3; row++) {
			world[row] = m[row * 4 + 0] * target[0] + m[row * 4 + 1] * target[1] +
			             m[row * 4 + 2] * target[2] + m[row * 4 + 3];
		}
		std::array<float, 3> dir {};
		float                length = 0.0f;
		for (int a = 0; a < 3; a++) {
			dir[a] = normal(rng);
			length += dir[a] * dir[a];
		}
		length           = std::sqrt(std::max(length, 1e-12f));
		const float dist = 1.0f + 19.0f * unit(rng);
		RayDesc     ray {};
		for (int a = 0; a < 3; a++) {
			ray.dir[a]    = dir[a] / length;
			ray.origin[a] = world[a] - ray.dir[a] * dist;
		}
		ray.extent = 1e4f;
		rays.push_back(ray);
	}
	return rays;
}

void TestWalkAndReference(const SceneParams& params, uint32_t ray_count, const char* label) {
	const Scene scene  = RtTest::MakeScene(params);
	const auto  reader = scene.memory.Reader();

	TlasWalk tlas;
	Check(WalkTlas(reader, scene.tlas.base, tlas) == WalkStatus::Ok, "TLAS walk");
	Check(tlas.instances.size() == scene.instances.size(), "TLAS walk reaches every instance");
	std::set<uint32_t> seen;
	for (const auto& inst: tlas.instances) {
		seen.insert(inst.index);
		const auto& desc = scene.instances.at(inst.index);
		Check(inst.blas_address == desc.blas_address && inst.root == desc.blas_root &&
		          inst.flags == desc.flags && inst.Mask() == desc.mask &&
		          (inst.id_mask & 0xffffffu) == desc.id && !inst.fake,
		      "instance record fields");
	}
	Check(seen.size() == scene.instances.size(), "instances are distinct");

	std::vector<BlasTriangles> expanded(scene.blas.size());
	std::map<uint64_t, size_t> blas_index;
	for (size_t b = 0; b < scene.blas.size(); b++) {
		blas_index[scene.blas[b].base] = b;
		BlasWalk walk;
		Check(WalkBlas(reader, scene.blas[b].base, scene.blas[b].root, walk) == WalkStatus::Ok,
		      "BLAS walk");
		Check(walk.leaves.size() == scene.blas[b].leaves, "BLAS walk finds every leaf");
		Check(!params.shared_exp || b >= params.blas_count || walk.shared_exp_nodes > 0,
		      "shared-exp scene contains type-6 nodes");
		Check(ExpandTriangles(reader, scene.blas[b].base, walk, expanded[b]) == WalkStatus::Ok,
		      "BLAS triangle expansion");
		const auto& source = scene.blas_triangles[b];
		Check(expanded[b].triangles.size() == source.size(), "every triangle expanded once");
		std::vector<bool> used(source.size());
		for (size_t i = 0; i < expanded[b].triangles.size(); i++) {
			const uint32_t prim = expanded[b].entries[i].info & ~Psr::LeafLastFlag;
			Check(prim < source.size() && !used[prim], "leaf entries are a permutation");
			if (prim >= source.size() || used[prim]) {
				continue;
			}
			used[prim]     = true;
			const auto& lb = walk.leaves[expanded[b].leaf_of[i]].bounds;
			for (const auto& v: expanded[b].triangles[i].v) {
				for (int a = 0; a < 3; a++) {
					Check(lb.min[a] <= v[a] && v[a] <= lb.max[a], "leaf bounds hold the triangle");
				}
			}
			Check(std::memcmp(&expanded[b].triangles[i], &source[prim], sizeof(Triangle)) == 0,
			      "triangle vertices round-trip");
		}
	}

	// Reference traversal against brute force, both with the instruction's exact triangle test,
	// on random rays and on rays aimed at vertices and edges. The traversal with the instruction's
	// own box tests too (TraceGuestExact) may miss a grazing hit where the hardware box rounding
	// rejects a leaf; that count is reported, not checked.
	auto       rays      = RtTest::RandomRays(ray_count, params.world_size, params.seed + 11);
	const auto edge_rays = EdgeRays(scene, ray_count / 2u, params.seed + 13);
	rays.insert(rays.end(), edge_rays.begin(), edge_rays.end());
	const auto tests      = DefaultNodeTests();
	uint32_t   hits       = 0;
	uint32_t   mismatch   = 0;
	uint32_t   ties       = 0;
	uint32_t   guest_diff = 0;
	uint64_t   nodes      = 0;
	for (const auto& desc: rays) {
		const Ray   ray = ToRay(desc);
		TraceResult traced {};
		Check(TraceReference(reader, scene.tlas.base, ray, tests, traced, 64) == WalkStatus::Ok,
		      "reference trace");
		TraceResult guest {};
		Check(TraceGuestExact(reader, scene.tlas.base, ray, guest, 64) == WalkStatus::Ok,
		      "guest-exact trace");
		const auto brute = BruteForce(scene, expanded, blas_index, tlas, ray);
		nodes += traced.nodes_visited;
		if (traced.hit != brute.hit || (traced.hit && traced.t != brute.t)) {
			mismatch++;
		} else if (traced.hit &&
		           (traced.info != brute.info || traced.instance_index != brute.instance_index)) {
			ties++;
		}
		guest_diff += (guest.hit != brute.hit || (guest.hit && guest.t != brute.t)) ? 1u : 0u;
		hits += traced.hit ? 1u : 0u;
	}
	const auto total = static_cast<uint32_t>(rays.size());
	std::printf("  %-28s %5u rays (%u at edges): %5u hits, %u mismatches, %u equal-t ties, %.1f "
	            "nodes/ray; hardware box tests differ on %u\n",
	            label, total, static_cast<uint32_t>(edge_rays.size()), hits, mismatch, ties,
	            static_cast<double>(nodes) / std::max<uint32_t>(total, 1), guest_diff);
	Check(mismatch == 0, "reference traversal equals brute force");
	Check(hits > total / 20, "scene is hit often enough to be meaningful");
}

void TestErrors() {
	SceneParams params {};
	params.blas_count         = 1;
	params.instance_count     = 2;
	params.triangles_per_blas = 40;
	Scene scene               = RtTest::MakeScene(params);
	auto  reader              = scene.memory.Reader();

	BlasWalk walk;
	Check(WalkBlas(reader, 0x10, Psr::RootPointer, walk) == WalkStatus::ReadFailed,
	      "unmapped BLAS reports a read failure");

	auto& blob = scene.memory.Blob(scene.blas[0].base);
	RtTest::Detail::Put32(blob, Psr::HeaderFlagsOffset, Psr::HeaderTrapBit);
	Check(WalkBlas(reader, scene.blas[0].base, Psr::RootPointer, walk) == WalkStatus::TrapFlag,
	      "trap bit is reported");
	RtTest::Detail::Put32(blob, Psr::HeaderFlagsOffset, 0);

	// A child pointer with type 7 in a box node's slot is a leaf, but one with an unknown box
	// payload is not: point the root's first child at the header (garbage for a box16).
	uint32_t child = 0;
	std::memcpy(&child, blob.data() + Psr::RootNodeOffset, 4);
	if (Psr::IsBox(child)) {
		RtTest::Detail::Put32(blob, Psr::RootNodeOffset,
		                      Psr::HardwarePointer(0x40000000u, Psr::NodeType::Box32));
		Check(WalkBlas(reader, scene.blas[0].base, Psr::RootPointer, walk) ==
		          WalkStatus::ReadFailed,
		      "a child outside the BVH reports a read failure");
	}
}

// A fetcher that treats the scene's arena as mapped guest memory (gaps between blobs read as
// zero) and reports the pages of [gpu_begin, gpu_end) as GPU-written.
GuestSnapshot::Fetch SceneFetch(const GuestMemory& memory, uint64_t gpu_begin, uint64_t gpu_end,
                                uint64_t* fetches) {
	return [&memory, gpu_begin, gpu_end, fetches](uint64_t address, std::span<uint8_t> dst,
	                                              std::span<uint32_t> flags) {
		constexpr uint64_t page  = GuestSnapshot::PageBytes;
		const uint64_t     begin = memory.Begin() & ~(page - 1);
		const uint64_t     end   = (memory.End() + page - 1) & ~(page - 1);
		if (address < begin || address + dst.size() > end) {
			return false;
		}
		if (fetches != nullptr) {
			++*fetches;
		}
		std::fill(dst.begin(), dst.end(), uint8_t {0});
		for (const auto& [base, blob]: memory.Blobs()) {
			const uint64_t lo = std::max(base, address);
			const uint64_t hi = std::min<uint64_t>(base + blob.size(), address + dst.size());
			if (lo < hi) {
				std::memcpy(dst.data() + (lo - address), blob.data() + (lo - base), hi - lo);
			}
		}
		for (size_t i = 0; i < flags.size(); i++) {
			const uint64_t at = address + i * page;
			flags[i] = at < gpu_end && at + page > gpu_begin ? GuestSnapshot::GpuWritten : 0u;
		}
		return true;
	};
}

// KYTY_RT_DUMP_BVH's capture and file format on a synthetic scene: the capture reaches every
// instance, leaf and triangle; a saved and reloaded dump is self-contained and traces exactly like
// the original memory; windows and the byte budget bound what is fetched.
void TestBvhDump(const SceneParams& params, const char* label) {
	const Scene    scene     = RtTest::MakeScene(params);
	const auto     reader    = scene.memory.Reader();
	const uint64_t gpu_begin = scene.blas.front().base;
	const uint64_t gpu_end   = gpu_begin + scene.memory.Blobs().at(gpu_begin).size();
	uint64_t       fetches   = 0;

	Check(LooksLikePsrBvh(reader, scene.tlas.base), "the scene TLAS looks like a Psr BVH");
	Check(!LooksLikePsrBvh(reader, scene.tlas.base + 128), "an unaligned address is not a BVH");
	Check(!LooksLikePsrBvh(reader, 0x1000), "unmapped memory is not a BVH");

	GuestSnapshot snapshot(SceneFetch(scene.memory, gpu_begin, gpu_end, &fetches));
	const auto    capture = CaptureBvh(snapshot, scene.tlas.base, BvhCaptureLimits {});
	Check(capture.tlas_walk == WalkStatus::Ok, "dump: TLAS walk");
	Check(capture.instances == scene.instances.size(), "dump: every instance");
	std::set<std::pair<uint64_t, uint32_t>> keys;
	for (const auto& inst: scene.instances) {
		keys.insert({inst.blas_address, inst.blas_root});
	}
	Check(capture.blas.size() == keys.size(), "dump: every distinct BLAS");
	for (const auto& b: capture.blas) {
		size_t index = scene.blas.size();
		for (size_t i = 0; i < scene.blas.size(); i++) {
			index = scene.blas[i].base == b.base ? i : index;
		}
		Check(index < scene.blas.size(), "dump: BLAS base is a scene BLAS");
		if (index >= scene.blas.size()) {
			continue;
		}
		Check(b.walk == WalkStatus::Ok && b.expand == WalkStatus::Ok, "dump: BLAS walk");
		Check(b.leaves == scene.blas[index].leaves, "dump: BLAS leaves");
		Check(b.triangles == scene.blas_triangles[index].size(), "dump: BLAS triangles");
		Check(b.outside_extent == 0, "dump: BLAS reads stay inside the declared extent");
		Check(b.header_read && b.extent >= scene.memory.Blobs().at(b.base).size(),
		      "dump: declared extent covers the BLAS");
	}
	Check(snapshot.BytesWithFlag(GuestSnapshot::GpuWritten) > 0 || keys.empty(),
	      "dump: GPU-written pages are flagged");
	// Only fetch-ahead past the end of the arena may fail.
	const uint64_t arena_end_page =
	    (scene.memory.End() + GuestSnapshot::PageBytes - 1) / GuestSnapshot::PageBytes;
	Check(snapshot.Refused() == 0 &&
	          std::all_of(snapshot.UnreadablePages().begin(), snapshot.UnreadablePages().end(),
	                      [&](uint64_t page) { return page >= arena_end_page; }),
	      "dump: nothing refused, nothing unreadable inside the arena");
	snapshot.meta.emplace_back("test", label);
	AppendCaptureMeta(capture, snapshot, snapshot.meta);

	// Save, load, compare.
	std::error_code ec;
	const auto      path = std::filesystem::temp_directory_path(ec) /
	                       ("rt_hardware_tests_" + std::to_string(params.seed) + ".kbvh");
	Check(snapshot.Save(path), "dump: save");
	GuestSnapshot loaded;
	Check(loaded.Load(path), "dump: load");
	std::filesystem::remove(path, ec);
	Check(loaded.meta == snapshot.meta, "dump: meta round-trips");
	bool same_pages = loaded.Pages().size() == snapshot.Pages().size();
	for (auto a = loaded.Pages().begin(), b = snapshot.Pages().begin();
	     same_pages && a != loaded.Pages().end(); ++a, ++b) {
		same_pages = a->first == b->first && a->second.flags == b->second.flags &&
		             a->second.bytes == b->second.bytes;
	}
	Check(same_pages, "dump: pages round-trip");
	char tlas_text[24];
	std::snprintf(tlas_text, sizeof(tlas_text), "0x%" PRIx64, scene.tlas.base);
	Check(loaded.MetaValue("capture.tlas") == tlas_text, "dump: TLAS address in the meta");

	// The loaded dump alone (no fetcher) walks and traces like the scene memory.
	const auto again = CaptureBvh(loaded, scene.tlas.base, BvhCaptureLimits {});
	Check(again.tlas_walk == WalkStatus::Ok && again.instances == capture.instances &&
	          again.blas.size() == capture.blas.size() && loaded.Refused() == 0,
	      "dump: self-contained");
	const auto loaded_reader = loaded.Reader();
	const auto rays          = RtTest::RandomRays(500, params.world_size, params.seed + 7);
	const auto tests         = DefaultNodeTests();
	uint32_t   differ        = 0;
	for (const auto& desc: rays) {
		TraceResult a {};
		TraceResult b {};
		TraceReference(reader, scene.tlas.base, ToRay(desc), tests, a, 64);
		TraceReference(loaded_reader, scene.tlas.base, ToRay(desc), tests, b, 64);
		differ += (a.hit != b.hit || a.t != b.t || a.info != b.info ||
		           a.instance_index != b.instance_index)
		              ? 1u
		              : 0u;
	}
	Check(differ == 0, "dump: traces equal the scene's");

	// Windows and the budget.
	GuestSnapshot closed(SceneFetch(scene.memory, 0, 0, nullptr));
	uint32_t      word = 0;
	Check(!closed.Read(scene.tlas.base, &word, 4) && closed.Refused() == 1,
	      "dump: no fetch outside a window");
	GuestSnapshot small(SceneFetch(scene.memory, 0, 0, nullptr), 16 * GuestSnapshot::PageBytes);
	CaptureBvh(small, scene.tlas.base, BvhCaptureLimits {});
	Check(small.Bytes() <= 16 * GuestSnapshot::PageBytes && small.Refused() > 0,
	      "dump: the byte budget bounds the snapshot");
	std::printf("  %-28s dump: %zu BLAS, %" PRIu64 " KiB (%" PRIu64 " KiB GPU-written), %" PRIu64
	            " fetches, %.1f ms\n",
	            label, capture.blas.size(), snapshot.Bytes() / 1024,
	            snapshot.BytesWithFlag(GuestSnapshot::GpuWritten) / 1024, fetches, capture.ms);
}

void RunCpuTests() {
	std::printf("CPU tests\n");
	TestPointers();
	TestDecode();
	TestConservativeHalves();
	TestSharedExpBounds();
	TestErrors();
	SceneParams params {};
	TestWalkAndReference(params, 3000, "box32, culling, masks");
	params.box16_interior = true;
	params.seed           = 2;
	TestWalkAndReference(params, 3000, "box16 interior nodes");
	params.box16_interior = false;
	params.leaf_root_blas = true;
	params.seed           = 3;
	TestWalkAndReference(params, 3000, "leaf-root BLAS");
	params.leaf_root_blas = false;
	params.fan_nodes      = true;
	params.seed           = 4;
	TestWalkAndReference(params, 3000, "fan triangle nodes");
	params.shared_exp         = true;
	params.triangles_per_blas = 2048;
	params.seed               = 5;
	TestWalkAndReference(params, 3000, "type-6 shared-exp nodes");
	TestBvhDump(params, "type-6 shared-exp nodes");
	TestBvhDump(SceneParams {}, "box32, culling, masks");
}

// ---------------------------------------------------------------------------------------------
// Vulkan harness

void CheckVk(vk::Result result, const char* what) {
	if (result != vk::Result::eSuccess) {
		std::printf("FATAL: %s failed (%d)\n", what, static_cast<int>(result));
		std::exit(2);
	}
}

struct Buffer {
	vk::Buffer        buffer  = nullptr;
	vk::DeviceMemory  memory  = nullptr;
	vk::DeviceAddress address = 0;
	vk::DeviceSize    size    = 0;
	void*             mapped  = nullptr;
};

class Gpu {
public:
	bool Init() {
		static vk::detail::DynamicLoader loader;
		const auto                       get_proc =
		    loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (get_proc == nullptr) {
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(get_proc);
		vk::ApplicationInfo app {};
		app.pApplicationName = "rt_hardware_tests";
		app.apiVersion       = VK_API_VERSION_1_3;
		vk::InstanceCreateInfo instance_info {};
		instance_info.pApplicationInfo = &app;
		if (vk::createInstance(&instance_info, nullptr, &m_instance) != vk::Result::eSuccess) {
			return false;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance);
		uint32_t count = 0;
		CheckVk(m_instance.enumeratePhysicalDevices(&count, nullptr), "enumerate devices");
		std::vector<vk::PhysicalDevice> devices(count);
		CheckVk(m_instance.enumeratePhysicalDevices(&count, devices.data()), "enumerate devices");
		for (const auto& candidate: devices) {
			vk::PhysicalDeviceProperties properties {};
			candidate.getProperties(&properties);
			if (QueryRtSupport(candidate).Usable() &&
			    (!m_physical || properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)) {
				m_physical   = candidate;
				m_properties = properties;
			}
		}
		if (!m_physical) {
			return false;
		}
		m_support = QueryRtSupport(m_physical);

		uint32_t family_count = 0;
		m_physical.getQueueFamilyProperties(&family_count, nullptr);
		std::vector<vk::QueueFamilyProperties> families(family_count);
		m_physical.getQueueFamilyProperties(&family_count, families.data());
		for (uint32_t i = 0; i < family_count; i++) {
			if ((families[i].queueFlags & vk::QueueFlagBits::eCompute) &&
			    families[i].timestampValidBits > 0) {
				m_family = i;
				break;
			}
		}
		const float               priority = 1.0f;
		vk::DeviceQueueCreateInfo queue_info {};
		queue_info.queueFamilyIndex = m_family;
		queue_info.queueCount       = 1;
		queue_info.pQueuePriorities = &priority;

		vk::PhysicalDeviceRayQueryFeaturesKHR              rq {};
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR as {};
		vk::PhysicalDeviceVulkan13Features                 f13 {};
		vk::PhysicalDeviceVulkan12Features                 f12 {};
		vk::PhysicalDeviceFeatures2                        f {};
		rq.rayQuery                     = VK_TRUE;
		as.accelerationStructure        = VK_TRUE;
		as.pNext                        = &rq;
		f13.synchronization2            = VK_TRUE;
		f13.pNext                       = &as;
		f12.bufferDeviceAddress         = VK_TRUE;
		f12.hostQueryReset              = VK_TRUE;
		f12.pNext                       = &f13;
		f.features.shaderInt64          = VK_TRUE;
		f.features.shaderFloat64        = VK_TRUE;
		f.pNext                         = &f12;
		const auto           extensions = RequiredRtDeviceExtensions();
		vk::DeviceCreateInfo device_info {};
		device_info.pNext                   = &f;
		device_info.queueCreateInfoCount    = 1;
		device_info.pQueueCreateInfos       = &queue_info;
		device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		device_info.ppEnabledExtensionNames = extensions.data();
		CheckVk(m_physical.createDevice(&device_info, nullptr, &m_device), "create device");
		VULKAN_HPP_DEFAULT_DISPATCHER.init(m_device);
		m_device.getQueue(m_family, 0, &m_queue);
		m_physical.getMemoryProperties(&m_memory);

		vk::CommandPoolCreateInfo pool {};
		pool.flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		pool.queueFamilyIndex = m_family;
		CheckVk(m_device.createCommandPool(&pool, nullptr, &m_pool), "command pool");
		vk::QueryPoolCreateInfo query {};
		query.queryType  = vk::QueryType::eTimestamp;
		query.queryCount = 64;
		CheckVk(m_device.createQueryPool(&query, nullptr, &m_queries), "query pool");

		vk::DescriptorSetLayoutBinding binding {};
		binding.binding         = 0;
		binding.descriptorType  = vk::DescriptorType::eAccelerationStructureKHR;
		binding.descriptorCount = 1;
		binding.stageFlags      = vk::ShaderStageFlagBits::eCompute;
		vk::DescriptorSetLayoutCreateInfo layout {};
		layout.bindingCount = 1;
		layout.pBindings    = &binding;
		CheckVk(m_device.createDescriptorSetLayout(&layout, nullptr, &m_tlas_layout), "set layout");
		vk::DescriptorPoolSize pool_size {};
		pool_size.type            = vk::DescriptorType::eAccelerationStructureKHR;
		pool_size.descriptorCount = 256;
		vk::DescriptorPoolCreateInfo dpool {};
		dpool.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
		dpool.maxSets       = 256;
		dpool.poolSizeCount = 1;
		dpool.pPoolSizes    = &pool_size;
		CheckVk(m_device.createDescriptorPool(&dpool, nullptr, &m_descriptors), "descriptor pool");
		return true;
	}

	~Gpu() {
		if (m_device) {
			(void)m_device.waitIdle();
			for (auto& p: m_pipelines) {
				m_device.destroyPipeline(p.pipeline);
				m_device.destroyPipelineLayout(p.layout);
			}
			for (auto as: m_structures) {
				m_device.destroyAccelerationStructureKHR(as);
			}
			for (auto& b: m_buffers) {
				m_device.destroyBuffer(b.buffer);
				m_device.freeMemory(b.memory);
			}
			m_device.destroyDescriptorPool(m_descriptors);
			m_device.destroyDescriptorSetLayout(m_tlas_layout);
			m_device.destroyQueryPool(m_queries);
			m_device.destroyCommandPool(m_pool);
			m_device.destroy();
		}
		if (m_instance) {
			m_instance.destroy();
		}
	}

	const char* Name() const { return m_properties.deviceName.data(); }

	// Frees the buffers and acceleration structures created after Mark().
	struct Marker {
		size_t buffers    = 0;
		size_t structures = 0;
	};
	Marker Mark() const { return {m_buffers.size(), m_structures.size()}; }
	void   Release(const Marker& marker) {
		(void)m_device.waitIdle();
		while (m_structures.size() > marker.structures) {
			m_device.destroyAccelerationStructureKHR(m_structures.back());
			m_structures.pop_back();
		}
		while (m_buffers.size() > marker.buffers) {
			m_device.destroyBuffer(m_buffers.back().buffer);
			m_device.freeMemory(m_buffers.back().memory);
			m_buffers.pop_back();
		}
	}
	const RtDeviceSupport& Support() const { return m_support; }
	vk::Device             Device() const { return m_device; }

	Buffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, bool host) {
		Buffer b {};
		b.size = std::max<vk::DeviceSize>(size, 16);
		vk::BufferCreateInfo info {};
		info.size  = b.size;
		info.usage = usage | vk::BufferUsageFlagBits::eShaderDeviceAddress |
		             vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
		             vk::BufferUsageFlagBits::eStorageBuffer;
		CheckVk(m_device.createBuffer(&info, nullptr, &b.buffer), "create buffer");
		vk::MemoryRequirements req {};
		m_device.getBufferMemoryRequirements(b.buffer, &req);
		const auto want = host ? (vk::MemoryPropertyFlagBits::eHostVisible |
		                          vk::MemoryPropertyFlagBits::eHostCoherent)
		                       : vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eDeviceLocal);
		uint32_t   type = UINT32_MAX;
		for (uint32_t i = 0; i < m_memory.memoryTypeCount; i++) {
			if ((req.memoryTypeBits & (1u << i)) &&
			    (m_memory.memoryTypes[i].propertyFlags & want) == want) {
				type = i;
				break;
			}
		}
		if (type == UINT32_MAX) {
			std::printf("FATAL: no memory type\n");
			std::exit(2);
		}
		vk::MemoryAllocateFlagsInfo flags {};
		flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
		vk::MemoryAllocateInfo alloc {};
		alloc.pNext           = &flags;
		alloc.allocationSize  = req.size;
		alloc.memoryTypeIndex = type;
		CheckVk(m_device.allocateMemory(&alloc, nullptr, &b.memory), "allocate memory");
		CheckVk(m_device.bindBufferMemory(b.buffer, b.memory, 0), "bind memory");
		vk::BufferDeviceAddressInfo address {};
		address.buffer = b.buffer;
		b.address      = m_device.getBufferAddress(&address);
		if (host) {
			CheckVk(m_device.mapMemory(b.memory, 0, VK_WHOLE_SIZE, {}, &b.mapped), "map");
		}
		m_buffers.push_back(b);
		return b;
	}

	// Device-local buffer filled from host data through a staging copy.
	Buffer Upload(const void* data, vk::DeviceSize size, vk::BufferUsageFlags usage = {}) {
		Buffer staging = CreateBuffer(size, {}, true);
		std::memcpy(staging.mapped, data, size);
		Buffer dst = CreateBuffer(size, usage, false);
		Run([&](vk::CommandBuffer cmd) {
			vk::BufferCopy copy {};
			copy.size = size;
			cmd.copyBuffer(staging.buffer, dst.buffer, 1, &copy);
		});
		return dst;
	}

	void Download(const Buffer& src, void* dst, vk::DeviceSize size) {
		Buffer staging = CreateBuffer(size, {}, true);
		Run([&](vk::CommandBuffer cmd) {
			vk::BufferCopy copy {};
			copy.size = size;
			cmd.copyBuffer(src.buffer, staging.buffer, 1, &copy);
		});
		std::memcpy(dst, staging.mapped, size);
	}

	vk::AccelerationStructureKHR CreateAccel(const Buffer& storage, vk::DeviceSize size, bool top) {
		vk::AccelerationStructureCreateInfoKHR info {};
		info.buffer                     = storage.buffer;
		info.size                       = size;
		info.type                       = top ? vk::AccelerationStructureTypeKHR::eTopLevel
		                                      : vk::AccelerationStructureTypeKHR::eBottomLevel;
		vk::AccelerationStructureKHR as = nullptr;
		CheckVk(m_device.createAccelerationStructureKHR(&info, nullptr, &as), "create AS");
		m_structures.push_back(as);
		return as;
	}

	vk::DeviceAddress AccelAddress(vk::AccelerationStructureKHR as) {
		vk::AccelerationStructureDeviceAddressInfoKHR info {};
		info.accelerationStructure = as;
		return m_device.getAccelerationStructureAddressKHR(&info);
	}

	struct Pipeline {
		vk::Pipeline       pipeline = nullptr;
		vk::PipelineLayout layout   = nullptr;
		bool               tlas     = false;
	};

	Pipeline CreatePipeline(std::span<const uint32_t> spirv, uint32_t push_bytes, bool tlas) {
		Pipeline p {};
		p.tlas = tlas;
		vk::PushConstantRange range {};
		range.stageFlags = vk::ShaderStageFlagBits::eCompute;
		range.size       = push_bytes;
		vk::PipelineLayoutCreateInfo layout {};
		layout.pushConstantRangeCount = 1;
		layout.pPushConstantRanges    = &range;
		if (tlas) {
			layout.setLayoutCount = 1;
			layout.pSetLayouts    = &m_tlas_layout;
		}
		CheckVk(m_device.createPipelineLayout(&layout, nullptr, &p.layout), "pipeline layout");
		vk::ShaderModuleCreateInfo module_info {};
		module_info.codeSize    = spirv.size_bytes();
		module_info.pCode       = spirv.data();
		vk::ShaderModule module = nullptr;
		CheckVk(m_device.createShaderModule(&module_info, nullptr, &module), "shader module");
		vk::ComputePipelineCreateInfo info {};
		info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
		info.stage.module = module;
		info.stage.pName  = "main";
		info.layout       = p.layout;
		CheckVk(m_device.createComputePipelines(nullptr, 1, &info, nullptr, &p.pipeline),
		        "compute pipeline");
		m_device.destroyShaderModule(module);
		m_pipelines.push_back(p);
		return p;
	}

	vk::DescriptorSet TlasSet(vk::AccelerationStructureKHR tlas) {
		vk::DescriptorSetAllocateInfo alloc {};
		alloc.descriptorPool     = m_descriptors;
		alloc.descriptorSetCount = 1;
		alloc.pSetLayouts        = &m_tlas_layout;
		vk::DescriptorSet set    = nullptr;
		CheckVk(m_device.allocateDescriptorSets(&alloc, &set), "descriptor set");
		vk::WriteDescriptorSetAccelerationStructureKHR as_write {};
		as_write.accelerationStructureCount = 1;
		as_write.pAccelerationStructures    = &tlas;
		vk::WriteDescriptorSet write {};
		write.pNext           = &as_write;
		write.dstSet          = set;
		write.dstBinding      = 0;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eAccelerationStructureKHR;
		m_device.updateDescriptorSets(1, &write, 0, nullptr);
		return set;
	}

	void Dispatch(vk::CommandBuffer cmd, const Pipeline& p, const void* push, uint32_t push_bytes,
	              uint32_t groups, vk::DescriptorSet set = nullptr, uint32_t groups_y = 1) {
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, p.pipeline);
		if (p.tlas) {
			cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, p.layout, 0, 1, &set, 0,
			                       nullptr);
		}
		cmd.pushConstants(p.layout, vk::ShaderStageFlagBits::eCompute, 0, push_bytes, push);
		cmd.dispatch(std::max(groups, 1u), std::max(groups_y, 1u), 1);
	}

	static void ComputeBarrier(vk::CommandBuffer cmd) { RecordAccelBarrier(cmd, true); }

	// Records `record` into a one-shot command buffer, submits and waits.
	void Run(const std::function<void(vk::CommandBuffer)>& record) {
		vk::CommandBufferAllocateInfo alloc {};
		alloc.commandPool        = m_pool;
		alloc.level              = vk::CommandBufferLevel::ePrimary;
		alloc.commandBufferCount = 1;
		vk::CommandBuffer cmd    = nullptr;
		CheckVk(m_device.allocateCommandBuffers(&alloc, &cmd), "allocate command buffer");
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		CheckVk(cmd.begin(&begin), "begin");
		record(cmd);
		CheckVk(cmd.end(), "end");
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &cmd;
		CheckVk(m_queue.submit(1, &submit, nullptr), "submit");
		CheckVk(m_queue.waitIdle(), "wait idle");
		m_device.freeCommandBuffers(m_pool, 1, &cmd);
	}

	// Timestamps: Stamp(cmd, i) inside Run, then Millis(a, b) afterwards.
	void ResetStamps() { m_device.resetQueryPool(m_queries, 0, 64); }
	void Stamp(vk::CommandBuffer cmd, uint32_t index) {
		cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, m_queries, index);
	}
	double Millis(uint32_t a, uint32_t b) {
		uint64_t first  = 0;
		uint64_t second = 0;
		CheckVk(m_device.getQueryPoolResults(
		            m_queries, a, 1, sizeof(uint64_t), &first, sizeof(uint64_t),
		            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
		        "query a");
		CheckVk(m_device.getQueryPoolResults(
		            m_queries, b, 1, sizeof(uint64_t), &second, sizeof(uint64_t),
		            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait),
		        "query b");
		return static_cast<double>(second - first) * m_properties.limits.timestampPeriod / 1e6;
	}

private:
	vk::Instance                              m_instance = nullptr;
	vk::PhysicalDevice                        m_physical = nullptr;
	vk::PhysicalDeviceProperties              m_properties {};
	vk::PhysicalDeviceMemoryProperties        m_memory {};
	RtDeviceSupport                           m_support {};
	vk::Device                                m_device      = nullptr;
	vk::Queue                                 m_queue       = nullptr;
	uint32_t                                  m_family      = 0;
	vk::CommandPool                           m_pool        = nullptr;
	vk::QueryPool                             m_queries     = nullptr;
	vk::DescriptorSetLayout                   m_tlas_layout = nullptr;
	vk::DescriptorPool                        m_descriptors = nullptr;
	std::vector<Buffer>                       m_buffers;
	std::vector<Pipeline>                     m_pipelines;
	std::vector<vk::AccelerationStructureKHR> m_structures;
};

// Push constant blocks and records (must match the kernels).
struct ConvertPush {
	uint64_t page_table, page_first, base, out0, out1, queue, counters, aux;
	uint32_t root, capacity, queue_capacity, mode, aux_count;
};
struct ConvertJob {
	uint64_t base, aabbs, leaves, triangles, records, counters, queue;
	uint32_t root, leaf_capacity, tri_capacity, queue_capacity;
};
static_assert(sizeof(ConvertJob) == 72);
struct TraceSwPush {
	uint64_t tlas_base, page_table, page_first, rays, results;
	uint32_t ray_count, stack_limit;
};
struct TraceHwPush {
	uint64_t page_table, page_first, rays, results, slots;
	uint32_t ray_count, pad;
};
struct GpuRay {
	float o[3], extent, d[3], pad0, inv[3], pad1;
};
struct GpuResult {
	float    t;
	uint32_t info, instance_index, tri_ptr, hit, nodes, candidates, status;
};
static_assert(sizeof(GpuRay) == 48 && sizeof(GpuResult) == 32);

// Guest memory on the device behind a 16 KiB page table, like the recompiler's BDA path.
struct DeviceGuest {
	Buffer   arena;
	Buffer   page_table;
	uint64_t page_first = 0;
};

DeviceGuest UploadGuest(Gpu& gpu, const GuestMemory& memory) {
	// Only the 16 KiB pages that hold blob bytes go to the device (a BVH dump's pages lie far
	// apart in the guest address space); the pages in between map to one zero page.
	constexpr uint64_t           page  = 1ull << 14u;
	const uint64_t               begin = memory.Begin() & ~(page - 1);
	const uint64_t               end   = (memory.End() + page - 1) & ~(page - 1);
	std::map<uint64_t, uint64_t> slot; // page number -> arena page (arena page 0: the zero page)
	for (const auto& [address, blob]: memory.Blobs()) {
		if (blob.empty()) continue;
		for (uint64_t p = address / page; p <= (address + blob.size() - 1u) / page; p++) {
			slot.try_emplace(p, slot.size() + 1u);
		}
	}
	std::vector<uint8_t> bytes((slot.size() + 1u) * page);
	for (const auto& [address, blob]: memory.Blobs()) {
		for (uint64_t at = 0; at < blob.size();) {
			const uint64_t a = address + at;
			const uint64_t n = std::min<uint64_t>(blob.size() - at, page - (a % page));
			std::memcpy(bytes.data() + slot.at(a / page) * page + (a % page), blob.data() + at, n);
			at += n;
		}
	}
	DeviceGuest guest {};
	guest.arena      = gpu.Upload(bytes.data(), bytes.size());
	guest.page_first = begin >> 14u;
	std::vector<uint64_t> table((end - begin) / page, guest.arena.address);
	for (const auto& [p, s]: slot) {
		table[p - begin / page] = guest.arena.address + s * page;
	}
	guest.page_table = gpu.Upload(table.data(), table.size() * sizeof(uint64_t));
	return guest;
}

struct HostBlas {
	uint64_t                     psr_base       = 0;
	uint32_t                     root           = 0;
	uint32_t                     capacity       = 0; // primitives of the host BLAS
	uint32_t                     leaf_capacity  = 0;
	uint32_t                     tri_capacity   = 0;
	uint32_t                     queue_capacity = 0;
	Buffer                       aabbs;
	Buffer                       leaves;
	Buffer                       triangles;
	Buffer                       records;
	Buffer                       queue;
	Buffer                       geometry; // AABBs or triangle vertices
	Buffer                       table;    // leaf pointers or (tri ptr, info) records
	Buffer                       counters;
	Buffer                       storage;
	vk::AccelerationStructureKHR accel = nullptr;
	AccelBuildDesc               desc {};
	vk::DeviceSize               scratch_size        = 0;
	vk::DeviceSize               update_scratch_size = 0;
	vk::DeviceAddress            scratch             = 0;
};

struct HostScene {
	DeviceGuest                  guest;
	std::vector<HostBlas>        blas;
	Buffer                       blas_table;
	Buffer                       instances;
	Buffer                       slots;
	Buffer                       tlas_counters;
	Buffer                       tlas_storage;
	Buffer                       scratch;
	Buffer                       blas_scratch;
	vk::AccelerationStructureKHR tlas = nullptr;
	AccelBuildDesc               tlas_desc {};
	uint32_t                     instance_capacity = 0;
	double   convert_ms = 0, blas_build_ms = 0, tlas_ms = 0, tlas_convert_ms = 0;
	uint64_t accel_bytes = 0, input_bytes = 0;
	uint32_t status_bits = 0;
	uint64_t primitives  = 0;
};

// Converts the scene on the GPU (walks, AS inputs), builds the BLASes and the TLAS.
HostScene BuildHostScene(Gpu& gpu, const Scene& scene, bool triangles, bool timed) {
	HostScene host {};
	host.guest          = UploadGuest(gpu, scene.memory);
	static auto convert = gpu.CreatePipeline(RT_PSR_CONVERT_SPV, sizeof(ConvertPush), false);
	const auto  reader  = scene.memory.Reader();

	// One host BLAS per (Psr BVH, root) used by an instance. Capacities come from a CPU walk in
	// this test; the emulator bounds them from the BVH size instead.
	std::set<std::pair<uint64_t, uint32_t>> keys;
	for (const auto& inst: scene.instances) {
		keys.insert({inst.blas_address, inst.blas_root});
	}
	std::vector<ConvertJob> jobs;
	uint32_t                max_leaves = 1;
	uint32_t                max_tris   = 1;
	for (const auto& [base, root]: keys) {
		HostBlas b {};
		b.psr_base = base;
		b.root     = root;
		BlasWalk walk;
		Check(WalkBlas(reader, base, root, walk) == WalkStatus::Ok, "host BLAS capacity walk");
		b.leaf_capacity  = std::max<uint32_t>(static_cast<uint32_t>(walk.leaves.size()), 1u);
		b.queue_capacity = walk.box_nodes + 4u;
		if (triangles) {
			BlasTriangles tris;
			ExpandTriangles(reader, base, walk, tris);
			b.tri_capacity = std::max<uint32_t>(static_cast<uint32_t>(tris.triangles.size()), 1u);
		}
		const auto as_input = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		b.aabbs             = gpu.CreateBuffer(uint64_t(b.leaf_capacity) * 24u, as_input, false);
		b.leaves            = gpu.CreateBuffer(uint64_t(b.leaf_capacity) * 4u, {}, false);
		b.queue             = gpu.CreateBuffer(uint64_t(b.queue_capacity) * 4u, {}, false);
		b.counters          = gpu.CreateBuffer(64, {}, true);
		std::memset(b.counters.mapped, 0, 64);
		if (triangles) {
			b.triangles = gpu.CreateBuffer(uint64_t(b.tri_capacity) * 36u, as_input, false);
			b.records   = gpu.CreateBuffer(uint64_t(b.tri_capacity) * 8u, {}, false);
			b.geometry  = b.triangles;
			b.table     = b.records;
			b.capacity  = b.tri_capacity;
		} else {
			b.geometry = b.aabbs;
			b.table    = b.leaves;
			b.capacity = b.leaf_capacity;
		}
		max_leaves = std::max(max_leaves, b.leaf_capacity);
		max_tris   = std::max(max_tris, b.tri_capacity);
		ConvertJob job {};
		job.base           = base;
		job.aabbs          = b.aabbs.address;
		job.leaves         = b.leaves.address;
		job.triangles      = triangles ? b.triangles.address : 0;
		job.records        = triangles ? b.records.address : 0;
		job.counters       = b.counters.address;
		job.queue          = b.queue.address;
		job.root           = root;
		job.leaf_capacity  = b.leaf_capacity;
		job.tri_capacity   = b.tri_capacity;
		job.queue_capacity = b.queue_capacity;
		jobs.push_back(job);
		host.blas.push_back(b);
	}
	Buffer     job_buffer = gpu.Upload(jobs.data(), jobs.size() * sizeof(ConvertJob));
	const auto job_count  = static_cast<uint32_t>(jobs.size());

	// Conversion: every BLAS walk in one dispatch (a workgroup each), then the fill passes.
	gpu.ResetStamps();
	gpu.Run([&](vk::CommandBuffer cmd) {
		gpu.Stamp(cmd, 0);
		ConvertPush p {};
		p.page_table = host.guest.page_table.address;
		p.page_first = host.guest.page_first;
		p.aux        = job_buffer.address;
		p.aux_count  = job_count;
		p.mode       = 0;
		gpu.Dispatch(cmd, convert, &p, sizeof(p), job_count);
		Gpu::ComputeBarrier(cmd);
		if (triangles) {
			p.mode = 1;
			gpu.Dispatch(cmd, convert, &p, sizeof(p), (max_leaves + 255) / 256, nullptr, job_count);
			Gpu::ComputeBarrier(cmd);
			p.mode = 3;
			gpu.Dispatch(cmd, convert, &p, sizeof(p), (max_tris + 255) / 256, nullptr, job_count);
		} else {
			p.mode = 2;
			gpu.Dispatch(cmd, convert, &p, sizeof(p), (max_leaves + 255) / 256, nullptr, job_count);
		}
		Gpu::ComputeBarrier(cmd);
		gpu.Stamp(cmd, 1);
	});
	host.convert_ms = gpu.Millis(0, 1);
	for (auto& b: host.blas) {
		const auto* counters = static_cast<const uint32_t*>(b.counters.mapped);
		host.status_bits |= counters[2];
		host.primitives += triangles ? counters[4] : counters[0];
	}

	// BLAS builds.
	vk::DeviceSize scratch_size = 0;
	for (auto& b: host.blas) {
		b.desc.geometry        = triangles ? AccelGeometry::Triangles : AccelGeometry::Aabbs;
		b.desc.data            = b.geometry.address;
		b.desc.stride          = triangles ? 12u : 24u;
		b.desc.primitive_count = b.capacity;
		b.desc.opaque          = triangles;
		b.desc.allow_update    = true;
		const auto sizes       = QueryBuildSizes(gpu.Device(), b.desc);
		b.storage =
		    gpu.CreateBuffer(sizes.accelerationStructureSize,
		                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, false);
		b.accel = gpu.CreateAccel(b.storage, sizes.accelerationStructureSize, false);
		host.accel_bytes += sizes.accelerationStructureSize;
		host.input_bytes += b.geometry.size + b.table.size;
		b.scratch_size        = sizes.buildScratchSize;
		b.update_scratch_size = sizes.updateScratchSize;
		scratch_size =
		    std::max(scratch_size, std::max(sizes.buildScratchSize, sizes.updateScratchSize));
	}
	host.instance_capacity         = static_cast<uint32_t>(scene.instances.size());
	host.tlas_desc.geometry        = AccelGeometry::Instances;
	host.tlas_desc.primitive_count = host.instance_capacity;
	host.instances                 = gpu.CreateBuffer(
	    uint64_t(host.instance_capacity) * 64u,
	    vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, false);
	host.tlas_desc.data   = host.instances.address;
	const auto tlas_sizes = QueryBuildSizes(gpu.Device(), host.tlas_desc);
	host.tlas_storage =
	    gpu.CreateBuffer(tlas_sizes.accelerationStructureSize,
	                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR, false);
	host.tlas = gpu.CreateAccel(host.tlas_storage, tlas_sizes.accelerationStructureSize, true);
	host.accel_bytes += tlas_sizes.accelerationStructureSize;
	scratch_size                    = std::max(scratch_size, tlas_sizes.buildScratchSize);
	const auto align                = std::max<uint32_t>(gpu.Support().scratch_alignment, 256u);
	host.scratch                    = gpu.CreateBuffer(scratch_size + align, {}, false);
	const vk::DeviceAddress scratch = (host.scratch.address + align - 1) & ~uint64_t(align - 1);

	// All BLAS builds in one batch, each with its own scratch range (they run concurrently).
	vk::DeviceSize total_scratch = 0;
	for (auto& b: host.blas) {
		total_scratch +=
		    (std::max(b.scratch_size, b.update_scratch_size) + align - 1) / align * align;
	}
	host.blas_scratch = gpu.CreateBuffer(total_scratch + align, {}, false);
	{
		vk::DeviceAddress at = (host.blas_scratch.address + align - 1) & ~uint64_t(align - 1);
		for (auto& b: host.blas) {
			b.scratch = at;
			at += (std::max(b.scratch_size, b.update_scratch_size) + align - 1) / align * align;
		}
	}
	gpu.ResetStamps();
	gpu.Run([&](vk::CommandBuffer cmd) {
		gpu.Stamp(cmd, 0);
		for (auto& b: host.blas) {
			RecordBuild(cmd, b.desc, b.accel, nullptr, b.scratch);
		}
		RecordAccelBarrier(cmd, false);
		gpu.Stamp(cmd, 1);
	});
	host.blas_build_ms = gpu.Millis(0, 1);

	// BLAS table for the TLAS conversion: (psr base, root, AS address, per-primitive table).
	std::vector<uint64_t> table;
	for (auto& b: host.blas) {
		table.push_back(b.psr_base);
		table.push_back(b.root);
		table.push_back(gpu.AccelAddress(b.accel));
		table.push_back(b.table.address);
	}
	host.blas_table    = gpu.Upload(table.data(), table.size() * sizeof(uint64_t));
	host.slots         = gpu.CreateBuffer(uint64_t(host.instance_capacity) * 32u, {}, false);
	host.tlas_counters = gpu.CreateBuffer(64, {}, true);
	std::memset(host.tlas_counters.mapped, 0, 64);
	uint32_t tlas_nodes = 0;
	{
		TlasWalk walk;
		WalkTlas(reader, scene.tlas.base, walk);
		tlas_nodes = walk.box_nodes + 4;
	}
	// Node queue, then the instance leaf list (rt_psr_convert.comp, InstanceList).
	Buffer tlas_queue = gpu.CreateBuffer(
	    uint64_t(tlas_nodes) * 4u + uint64_t(host.instance_capacity) * 4u, {}, false);

	gpu.ResetStamps();
	gpu.Run([&](vk::CommandBuffer cmd) {
		gpu.Stamp(cmd, 0);
		ConvertPush p {};
		p.page_table     = host.guest.page_table.address;
		p.page_first     = host.guest.page_first;
		p.base           = scene.tlas.base;
		p.out0           = host.instances.address;
		p.out1           = host.slots.address;
		p.queue          = tlas_queue.address;
		p.counters       = host.tlas_counters.address;
		p.aux            = host.blas_table.address;
		p.root           = Psr::RootPointer;
		p.capacity       = host.instance_capacity;
		p.queue_capacity = tlas_nodes;
		p.mode           = 4;
		p.aux_count      = static_cast<uint32_t>(host.blas.size());
		gpu.Dispatch(cmd, convert, &p, sizeof(p), 1);
		Gpu::ComputeBarrier(cmd);
		p.mode = 5;
		gpu.Dispatch(cmd, convert, &p, sizeof(p), (host.instance_capacity + 255) / 256);
		Gpu::ComputeBarrier(cmd);
		gpu.Stamp(cmd, 1);
		RecordBuild(cmd, host.tlas_desc, host.tlas, nullptr, scratch);
		RecordAccelBarrier(cmd, false);
		gpu.Stamp(cmd, 2);
	});
	host.tlas_ms         = gpu.Millis(0, 2);
	host.tlas_convert_ms = gpu.Millis(0, 1);
	const auto* tc       = static_cast<const uint32_t*>(host.tlas_counters.mapped);
	host.status_bits |= tc[2];
	Check(tc[0] == scene.instances.size(), "GPU TLAS conversion emits every instance");
	if (timed) {
		std::printf("    BLAS convert %.3f ms (%" PRIu64
		            " %s), BLAS builds %.3f ms, TLAS convert %.3f "
		            "ms + build %.3f ms, AS memory %.1f MiB, inputs %.1f MiB\n",
		            host.convert_ms, host.primitives, triangles ? "triangles" : "leaf AABBs",
		            host.blas_build_ms, host.tlas_convert_ms, host.tlas_ms - host.tlas_convert_ms,
		            host.accel_bytes / 1048576.0, host.input_bytes / 1048576.0);
	}
	return host;
}

std::vector<GpuRay> ToGpuRays(const std::vector<RayDesc>& rays) {
	std::vector<GpuRay> out(rays.size());
	for (size_t i = 0; i < rays.size(); i++) {
		const Ray r = ToRay(rays[i]);
		for (int a = 0; a < 3; a++) {
			out[i].o[a]   = r.origin[a];
			out[i].d[a]   = r.dir[a];
			out[i].inv[a] = r.inv_dir[a];
		}
		out[i].extent = r.extent;
	}
	return out;
}

struct TraceRun {
	std::vector<GpuResult> results;
	double                 ms = 0;
};

enum class TraceKind { Software, Hybrid, Triangles };

// Traces in chunks of at most 256k rays, one submission each, so no single dispatch runs long
// enough to risk a driver timeout. Time = sum over chunks of the fastest of `repeats` runs.
TraceRun Trace(Gpu& gpu, const HostScene& host, const Scene& scene, const Buffer& rays,
               uint32_t ray_count, TraceKind kind, uint32_t repeats = 1) {
	static auto        sw  = gpu.CreatePipeline(RT_PSR_TRACE_SW_SPV, sizeof(TraceSwPush), false);
	static auto        hw  = gpu.CreatePipeline(RT_PSR_TRACE_HW_SPV, sizeof(TraceHwPush), true);
	static auto        tri = gpu.CreatePipeline(RT_PSR_TRACE_HW_TRI_SPV, sizeof(TraceHwPush), true);
	constexpr uint32_t chunk = 256u * 1024u;
	Buffer results           = gpu.CreateBuffer(uint64_t(ray_count) * sizeof(GpuResult), {}, false);
	vk::DescriptorSet set    = kind == TraceKind::Software ? nullptr : gpu.TlasSet(host.tlas);
	TraceRun          run {};
	for (uint32_t first = 0; first < ray_count; first += chunk) {
		const uint32_t count = std::min(chunk, ray_count - first);
		double         best  = 1e30;
		for (uint32_t r = 0; r < repeats; r++) {
			gpu.ResetStamps();
			gpu.Run([&](vk::CommandBuffer cmd) {
				gpu.Stamp(cmd, 0);
				if (kind == TraceKind::Software) {
					TraceSwPush p {};
					p.tlas_base   = scene.tlas.base;
					p.page_table  = host.guest.page_table.address;
					p.page_first  = host.guest.page_first;
					p.rays        = rays.address + uint64_t(first) * sizeof(GpuRay);
					p.results     = results.address + uint64_t(first) * sizeof(GpuResult);
					p.ray_count   = count;
					p.stack_limit = 60;
					gpu.Dispatch(cmd, sw, &p, sizeof(p), (count + 63) / 64);
				} else {
					TraceHwPush p {};
					p.page_table = host.guest.page_table.address;
					p.page_first = host.guest.page_first;
					p.rays       = rays.address + uint64_t(first) * sizeof(GpuRay);
					p.results    = results.address + uint64_t(first) * sizeof(GpuResult);
					p.slots      = host.slots.address;
					p.ray_count  = count;
					gpu.Dispatch(cmd, kind == TraceKind::Hybrid ? hw : tri, &p, sizeof(p),
					             (count + 63) / 64, set);
				}
				gpu.Stamp(cmd, 1);
			});
			const double ms = gpu.Millis(0, 1);
			best            = std::min(best, ms);
			if (ms > 400.0) {
				break; // slow enough; do not repeat
			}
		}
		run.ms += best;
	}
	run.results.resize(ray_count);
	gpu.Download(results, run.results.data(), run.results.size() * sizeof(GpuResult));
	return run;
}

struct Compare {
	uint32_t hits = 0, mismatches = 0, ties = 0, statuses = 0;
};

// `ref` vs `got`: hit flag and t must agree (t exactly when `exact_t`); equal t with a different
// primitive or instance is a tie (the guest keeps the later of two hits at the same distance).
Compare CompareResults(const std::vector<TraceResult>& ref, const std::vector<GpuResult>& got,
                       bool exact_t, const char* label, bool verbose) {
	Compare c {};
	for (size_t i = 0; i < ref.size(); i++) {
		const auto& a = ref[i];
		const auto& b = got[i];
		if (b.status != 0) {
			c.statuses++;
		}
		c.hits += a.hit ? 1u : 0u;
		// NaN payloads differ between the CPU and the GPU; a NaN t is the same result.
		bool same_t = a.t == b.t || (std::isnan(a.t) && std::isnan(b.t));
		if (!exact_t && a.hit && b.hit) {
			same_t = std::fabs(a.t - b.t) <= 1e-4f * std::max(1.0f, std::fabs(a.t));
		}
		if (a.hit != (b.hit != 0) || (a.hit && !same_t)) {
			if (verbose && c.mismatches < 5) {
				std::printf(
				    "    %s ray %zu: ref hit=%d t=%.9g info=%08x inst=%u | got hit=%u t=%.9g "
				    "info=%08x inst=%u\n",
				    label, i, a.hit ? 1 : 0, a.t, a.info, a.instance_index, b.hit, b.t, b.info,
				    b.instance_index);
			}
			c.mismatches++;
		} else if (a.hit && (a.info != b.info || a.instance_index != b.instance_index)) {
			c.ties++;
		}
	}
	return c;
}

void RunGpuScene(Gpu& gpu, const SceneParams& params, uint32_t ray_count, const char* label) {
	const Scene scene  = RtTest::MakeScene(params);
	const auto  reader = scene.memory.Reader();
	// Random rays, then as many again aimed at triangle vertices and edges.
	const uint32_t random_count = ray_count;
	auto           rays = RtTest::RandomRays(random_count, params.world_size, params.seed + 101);
	const auto     edge_rays = EdgeRays(scene, random_count, params.seed + 103);
	rays.insert(rays.end(), edge_rays.begin(), edge_rays.end());
	ray_count = static_cast<uint32_t>(rays.size());
	std::vector<TraceResult> ref(rays.size());
	const auto               tests = DefaultNodeTests();
	for (size_t i = 0; i < rays.size(); i++) {
		TraceReference(reader, scene.tlas.base, ToRay(rays[i]), tests, ref[i], 60);
	}
	const auto gpu_rays = ToGpuRays(rays);
	Buffer     ray_buf  = gpu.Upload(gpu_rays.data(), gpu_rays.size() * sizeof(GpuRay));

	// Brute force (the exact triangle test on every triangle of every instance) settles the rays
	// where the hybrid and the reference disagree: a box test on tight bounds can reject a leaf
	// that a grazing ray hits exactly, and the hybrid's inflated leaf AABBs keep it.
	TlasWalk                   tlas_walk;
	std::vector<BlasTriangles> expanded(scene.blas.size());
	std::map<uint64_t, size_t> blas_index;
	WalkTlas(reader, scene.tlas.base, tlas_walk);
	for (size_t b = 0; b < scene.blas.size(); b++) {
		blas_index[scene.blas[b].base] = b;
		BlasWalk walk;
		WalkBlas(reader, scene.blas[b].base, scene.blas[b].root, walk);
		ExpandTriangles(reader, scene.blas[b].base, walk, expanded[b]);
	}
	const auto same = [](bool hit_a, float t_a, bool hit_b, float t_b) {
		return hit_a == hit_b && (!hit_a || t_a == t_b || (std::isnan(t_a) && std::isnan(t_b)));
	};

	std::printf("  %s: %u BLAS x %u tris, %u instances, %u rays (%u aimed at edges)\n", label,
	            params.blas_count, params.triangles_per_blas, params.instance_count, ray_count,
	            ray_count - random_count);
	{
		HostScene host = BuildHostScene(gpu, scene, false, false);
		Check(host.status_bits == 0, "GPU conversion reports no errors (AABB mode)");
		const auto sw     = Trace(gpu, host, scene, ray_buf, ray_count, TraceKind::Software);
		const auto hybrid = Trace(gpu, host, scene, ray_buf, ray_count, TraceKind::Hybrid);
		const auto c_sw   = CompareResults(ref, sw.results, false, "sw", true);
		std::printf("    GPU software trace vs CPU reference: %u hits, %u mismatches, %u ties, %u "
		            "status\n",
		            c_sw.hits, c_sw.mismatches, c_sw.ties, c_sw.statuses);
		Check(c_sw.statuses == 0, "software trace kernel reports no errors");
		Check(c_sw.mismatches == 0, "GPU software trace equals the CPU reference");

		uint32_t ties = 0, box_misses = 0, bad = 0, statuses = 0;
		for (size_t i = 0; i < rays.size(); i++) {
			const auto& a = ref[i];
			const auto& b = hybrid.results[i];
			statuses += b.status != 0u ? 1u : 0u;
			if (same(a.hit, a.t, b.hit != 0u, b.t)) {
				ties +=
				    a.hit && (a.info != b.info || a.instance_index != b.instance_index) ? 1u : 0u;
				continue;
			}
			const auto bf = BruteForce(scene, expanded, blas_index, tlas_walk, ToRay(rays[i]));
			if (same(bf.hit, bf.t, b.hit != 0u, b.t)) {
				box_misses++;
				continue;
			}
			if (bad < 5u) {
				std::printf(
				    "    hybrid ray %zu: reference hit=%d t=%.9g | brute force hit=%d t=%.9g "
				    "| hybrid hit=%u t=%.9g\n",
				    i, a.hit ? 1 : 0, a.t, bf.hit ? 1 : 0, bf.t, b.hit, b.t);
			}
			bad++;
		}
		std::printf(
		    "    hybrid (RT cores + exact leaf tests): equals the CPU reference except on %u "
		    "rays, where the reference's box tests rejected a grazing hit and the hybrid "
		    "equals brute force; %u mismatches, %u equal-t ties\n",
		    box_misses, bad, ties);
		// What the software path computes per ray (the instruction's own box tests): how often it
		// differs from the hybrid is what a verify mode would see on such geometry.
		uint32_t guest_random = 0, guest_edge = 0;
		for (size_t i = 0; i < rays.size(); i++) {
			TraceResult guest {};
			TraceGuestExact(reader, scene.tlas.base, ToRay(rays[i]), guest, 60);
			const auto& b = hybrid.results[i];
			if (!same(guest.hit, guest.t, b.hit != 0u, b.t)) {
				(i < random_count ? guest_random : guest_edge)++;
			}
		}
		std::printf("    software path model (the instruction's box tests too) vs hybrid: %u of %u "
		            "random rays, %u of %u rays aimed at edges differ\n",
		            guest_random, random_count, guest_edge, ray_count - random_count);
		Check(statuses == 0, "hybrid trace kernel reports no errors");
		Check(bad == 0, "hybrid trace equals the CPU reference, or brute force where they differ");
	}
	if (!params.culling_instances) {
		HostScene host = BuildHostScene(gpu, scene, true, false);
		Check(host.status_bits == 0, "GPU conversion reports no errors (triangle mode)");
		const auto tri = Trace(gpu, host, scene, ray_buf, ray_count, TraceKind::Triangles);
		// Host watertightness and precision differ from the exact test at edges: on random rays
		// anything beyond a trace would be a conversion error; rays aimed at edges are reported.
		uint32_t random_bad = 0, edge_bad = 0;
		for (size_t i = 0; i < rays.size(); i++) {
			const auto& a = ref[i];
			const auto& b = tri.results[i];
			const bool  ok =
			    a.hit == (b.hit != 0u) &&
			    (!a.hit || std::fabs(a.t - b.t) <= 1e-4f * std::max(1.0f, std::fabs(a.t)));
			(i < random_count ? random_bad : edge_bad) += ok ? 0u : 1u;
		}
		std::printf("    triangle mode (approximate) vs CPU reference: %u of %u random rays differ "
		            "(%.3f%%), %u of %u rays aimed at edges\n",
		            random_bad, random_count, 100.0 * random_bad / std::max(1u, random_count),
		            edge_bad, ray_count - random_count);
		Check(random_bad * 1000u <= random_count,
		      "triangle mode agrees on >= 99.9% of random rays");
	}
}

struct ExactTestPush {
	uint64_t cases, results;
	uint32_t count, mode;
};

bool SameBits(uint32_t a, uint32_t b) {
	const bool nan_a = (a & 0x7fffffffu) > 0x7f800000u;
	const bool nan_b = (b & 0x7fffffffu) > 0x7f800000u;
	return a == b || (nan_a && nan_b);
}

// The GLSL twins in rt_psr_common.inc against the CPU, bit for bit:
// - the correctly rounded division on random bits, special values, normal operands with nearby
//   exponents, quotients near 1, and results in and below the denormal range (with exact ties);
// - the triangle test against RT-SW's BvhReference::Triangle (t_num, t_den) and ExactTriangleTest
//   (hit, t, facing) on random, vertex, edge, just-outside, degenerate and parallel cases.
void RunExactTwinTests(Gpu& gpu) {
	static auto  pipeline = gpu.CreatePipeline(RT_PSR_EXACT_TEST_SPV, sizeof(ExactTestPush), false);
	std::mt19937 rng(77);
	std::uniform_int_distribution<uint32_t> any;
	std::uniform_real_distribution<float>   mant(1.0f, 2.0f);
	std::uniform_real_distribution<float>   unit(0.0f, 1.0f);
	const auto bits    = [](float f) { return std::bit_cast<uint32_t>(f); };
	const auto from    = [](uint32_t u) { return std::bit_cast<float>(u); };
	const auto signed_ = [&](float f) { return (any(rng) & 1u) != 0u ? -f : f; };

	const auto run = [&](const std::vector<uint32_t>& cases, uint32_t count, uint32_t mode,
	                     uint32_t out_words) {
		Buffer        in  = gpu.Upload(cases.data(), cases.size() * sizeof(uint32_t));
		Buffer        out = gpu.CreateBuffer(uint64_t {count} * out_words * 4u, {}, false);
		ExactTestPush p {in.address, out.address, count, mode};
		gpu.Run([&](vk::CommandBuffer cmd) {
			gpu.Dispatch(cmd, pipeline, &p, sizeof(p), (count + 63u) / 64u);
		});
		std::vector<uint32_t> result(uint64_t {count} * out_words);
		gpu.Download(out, result.data(), result.size() * sizeof(uint32_t));
		return result;
	};

	// Division.
	std::vector<uint32_t> divs;
	const uint32_t specials[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu,
	                             0x00400000u, 0x00800000u, 0x3f800000u, 0xbf800000u, 0x3f7fffffu,
	                             0x3f800001u, 0x40400000u, 0x3eaaaaabu, 0x7f7fffffu, 0x7f800000u,
	                             0xff800000u, 0x7fc00000u, 0x7f800001u};
	for (const uint32_t a: specials) {
		for (const uint32_t b: specials) {
			divs.push_back(a);
			divs.push_back(b);
		}
	}
	for (int i = 0; i < 65536; i++) {
		divs.push_back(any(rng));
		divs.push_back(any(rng));
	}
	std::uniform_int_distribution<int> near_exp(-20, 20);
	for (int i = 0; i < 65536; i++) {
		divs.push_back(bits(signed_(std::ldexp(mant(rng), near_exp(rng)))));
		divs.push_back(bits(signed_(std::ldexp(mant(rng), near_exp(rng)))));
	}
	for (int i = 0; i < 32768; i++) {
		// Quotients next to 1 and to other rounding boundaries.
		const uint32_t b     = bits(std::ldexp(mant(rng), near_exp(rng)));
		const int32_t  delta = static_cast<int32_t>(any(rng) % 9u) - 4;
		divs.push_back(static_cast<uint32_t>(static_cast<int32_t>(b) + delta));
		divs.push_back(b);
	}
	std::uniform_int_distribution<uint32_t> tiny_exp(0, 27);
	std::uniform_int_distribution<int>      shift(0, 30);
	for (int i = 0; i < 65536; i++) {
		// Denormal and near-denormal results; a power-of-two divisor gives exact ties.
		const uint32_t a = (any(rng) & 0x807fffffu) | (tiny_exp(rng) << 23u);
		const uint32_t b = (i % 2 == 0) ? bits(std::ldexp(1.0f, shift(rng)))
		                                : bits(std::ldexp(mant(rng), shift(rng)));
		divs.push_back(a);
		divs.push_back(b);
	}
	const auto div_count = static_cast<uint32_t>(divs.size() / 2u);
	const auto div_gpu   = run(divs, div_count, 0u, 1u);
	uint32_t   div_bad   = 0;
	for (uint32_t i = 0; i < div_count; i++) {
		const uint32_t expected = bits(from(divs[i * 2u]) / from(divs[i * 2u + 1u]));
		if (!SameBits(expected, div_gpu[i])) {
			if (div_bad < 5u) {
				std::printf("    div %08x / %08x: CPU %08x GPU %08x\n", divs[i * 2u],
				            divs[i * 2u + 1u], expected, div_gpu[i]);
			}
			div_bad++;
		}
	}

	// Triangles: 16 dwords per case (a, b, c, origin, direction, unused).
	std::vector<uint32_t>           tri;
	const uint32_t                  tri_count = 131072;
	const float                     scales[]  = {1e-3f, 1.0f, 1e3f};
	std::normal_distribution<float> normal(0.0f, 1.0f);
	for (uint32_t i = 0; i < tri_count; i++) {
		const float          s    = scales[any(rng) % 3u];
		const uint32_t       kind = any(rng) % 8u;
		std::array<float, 3> a {}, b {}, c {};
		for (int k = 0; k < 3; k++) {
			a[k] = (unit(rng) * 2.0f - 1.0f) * s;
			b[k] = (unit(rng) * 2.0f - 1.0f) * s;
			c[k] = (unit(rng) * 2.0f - 1.0f) * s;
		}
		if (kind == 6u) { // degenerate: collinear or repeated vertex
			const float f = (any(rng) & 1u) != 0u ? 0.0f : unit(rng);
			for (int k = 0; k < 3; k++) {
				c[k] = a[k] + f * (b[k] - a[k]);
			}
		}
		// Target: inside, a vertex, on an edge, or just outside an edge.
		float u = unit(rng);
		float v = unit(rng) * (1.0f - u);
		if (kind == 1u) {
			u = 0.0f;
			v = 0.0f;
		} else if (kind == 2u) {
			v = 0.0f;
		} else if (kind == 3u) {
			v = 1.0f - u;
		} else if (kind == 4u) {
			u = -1e-6f * unit(rng);
		}
		std::array<float, 3> target {}, dir {};
		float                length = 0.0f;
		for (int k = 0; k < 3; k++) {
			target[k] = a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]);
			dir[k]    = normal(rng);
			length += dir[k] * dir[k];
		}
		if (kind == 5u) { // parallel to the plane: remove the normal component
			const std::array<float, 3> e1 {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
			for (int k = 0; k < 3; k++) {
				dir[k] = e1[k];
			}
			length = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
		}
		length           = std::sqrt(std::max(length, 1e-30f));
		const float dist = (0.1f + 10.0f * unit(rng)) * s;
		for (int k = 0; k < 3; k++) {
			dir[k] /= length;
		}
		if (kind == 7u) { // a zero direction component
			dir[any(rng) % 3u] = 0.0f;
		}
		for (const auto& v3: {a, b, c}) {
			for (const float f: v3) {
				tri.push_back(bits(f));
			}
		}
		for (int k = 0; k < 3; k++) {
			tri.push_back(bits(target[k] - dir[k] * dist));
		}
		for (int k = 0; k < 3; k++) {
			tri.push_back(bits(dir[k]));
		}
		tri.push_back(0u);
	}
	const auto tri_gpu  = run(tri, tri_count, 1u, 4u);
	uint32_t   tri_bad  = 0;
	uint32_t   tri_hits = 0;
	for (uint32_t i = 0; i < tri_count; i++) {
		const uint32_t*          c = tri.data() + uint64_t {i} * 16u;
		std::array<uint32_t, 16> node {};
		Triangle                 t3 {};
		for (int k = 0; k < 9; k++) {
			node[k]            = c[k];
			t3.v[k / 3][k % 3] = from(c[k]);
		}
		Ray ray {};
		for (int k = 0; k < 3; k++) {
			ray.origin[k] = from(c[9 + k]);
			ray.dir[k]    = from(c[12 + k]);
		}
		namespace Exact   = Libs::Graphics::ShaderRecompiler::BvhReference;
		const auto  ref   = Exact::Triangle(GuestTsharp(0, 0), 0u, node,
		                                    Exact::Ray {0.0f, ray.origin, ray.dir, ray.inv_dir},
		                                    Exact::Math {false});
		const auto  hit   = ExactTriangleTest(t3, ray);
		const auto* g     = tri_gpu.data() + uint64_t {i} * 4u;
		const bool  g_hit = (g[3] & 1u) != 0u;
		bool ok = SameBits(ref[0], g[0]) && SameBits(ref[1], g[1]) && hit.has_value() == g_hit;
		if (ok && hit) {
			ok = SameBits(bits(hit->t), g[2]) && hit->positive == ((g[3] & 2u) != 0u);
		}
		tri_hits += hit ? 1u : 0u;
		if (!ok) {
			if (tri_bad < 5u) {
				std::printf(
				    "    triangle case %u: CPU t_num %08x t_den %08x hit %d | GPU %08x %08x "
				    "hit %d\n",
				    i, ref[0], ref[1], hit ? 1 : 0, g[0], g[1], g_hit ? 1 : 0);
			}
			tri_bad++;
		}
	}
	std::printf("  exact twins: division %u cases, %u differ; triangle %u cases (%u hits), %u "
	            "differ\n",
	            div_count, div_bad, tri_count, tri_hits, tri_bad);
	Check(div_bad == 0, "GLSL division is correctly rounded");
	Check(tri_bad == 0, "GLSL triangle test equals BvhReference::Triangle");
}

void PrintSupport(Gpu& gpu) {
	const auto& s = gpu.Support();
	std::printf("  indirect AS builds: %s, scratch alignment %u, max primitives %" PRIu64
	            ", max instances %" PRIu64 "\n",
	            s.indirect_build ? "yes" : "no", s.scratch_alignment, s.max_primitive_count,
	            s.max_instance_count);
}

void RunGpuTests(Gpu& gpu) {
	std::printf("GPU tests on %s\n", gpu.Name());
	PrintSupport(gpu);
	RunExactTwinTests(gpu);
	SceneParams params {};
	params.culling_instances = true;
	RunGpuScene(gpu, params, 20000, "box32, culling, masks");
	params.box16_interior = true;
	params.seed           = 5;
	RunGpuScene(gpu, params, 20000, "box16 interior nodes");
	params.box16_interior    = false;
	params.culling_instances = false;
	params.leaf_root_blas    = true;
	params.seed              = 6;
	RunGpuScene(gpu, params, 20000, "no culling, leaf-root BLAS");
	params.leaf_root_blas    = false;
	params.culling_instances = true;
	params.fan_nodes         = true;
	params.seed              = 8;
	RunGpuScene(gpu, params, 20000, "fan triangle nodes");
	params.shared_exp         = true;
	params.triangles_per_blas = 2048;
	params.seed               = 10;
	RunGpuScene(gpu, params, 20000, "type-6 shared-exp nodes");
	params.shared_exp         = false;
	params.triangles_per_blas = 256;
	// Mesh-like geometry: shared edges between neighbouring triangles.
	params.surface            = true;
	params.culling_instances  = false;
	params.triangles_per_blas = 1024;
	params.instance_count     = 64;
	params.seed               = 9;
	RunGpuScene(gpu, params, 20000, "surface meshes");
}

// ---------------------------------------------------------------------------------------------
// Benchmarks

void RunBench(Gpu& gpu) {
	std::printf("Benchmarks on %s\n", gpu.Name());
	PrintSupport(gpu);
	struct Case {
		const char* name;
		uint32_t    blas, tris, instances;
		bool        surface;
	};
	const Case cases[] = {
	    {"surface, 1M tris (256 BLAS x 4k)", 256, 4096, 1024, true},
	    {"surface, 4M tris (256 BLAS x 16k)", 256, 16384, 4096, true},
	    {"triangle soup, 1M tris (stress)", 64, 16384, 2048, false},
	};
	for (const auto& c: cases) {
		SceneParams params {};
		params.blas_count         = c.blas;
		params.triangles_per_blas = c.tris;
		params.instance_count     = c.instances;
		params.world_size         = 400.0f;
		params.triangle_size      = 0.02f;
		params.culling_instances  = false;
		params.masked_instances   = false;
		params.surface            = c.surface;
		params.fan_nodes          = true;
		params.seed               = 42;
		const auto  t0            = std::chrono::steady_clock::now();
		const Scene scene         = RtTest::MakeScene(params);
		const auto  t1            = std::chrono::steady_clock::now();
		std::printf("  %s: %u BLAS x %u, %u instances (scene build %.0f ms CPU)\n", c.name, c.blas,
		            c.tris, c.instances,
		            std::chrono::duration<double, std::milli>(t1 - t0).count());
		const auto     marker = gpu.Mark();
		const uint32_t w = 1920, h = 1080;
		const uint32_t n       = w * h;
		const uint32_t n_rnd   = 512u * 1024u;
		const auto     camera  = c.surface ? RtTest::SurfaceCameraRays(w, h, params.world_size)
		                                   : RtTest::CameraRays(w, h, params.world_size);
		const auto     random  = c.surface ? RtTest::SurfaceBounceRays(n_rnd, params.world_size, 9)
		                                   : RtTest::RandomRays(n_rnd, params.world_size, 9);
		const auto     cam_gpu = ToGpuRays(camera);
		const auto     rnd_gpu = ToGpuRays(random);
		Buffer         cam_buf = gpu.Upload(cam_gpu.data(), cam_gpu.size() * sizeof(GpuRay));
		Buffer         rnd_buf = gpu.Upload(rnd_gpu.data(), rnd_gpu.size() * sizeof(GpuRay));
		for (int mode = 0; mode < 2; mode++) {
			const bool triangles = mode == 1;
			std::printf("   %s:\n", triangles ? "triangle mode (approximate)"
			                                  : "hybrid mode (exact leaf tests)");
			// The first build of a case warms clocks and fresh allocations up; time the second.
			(void)BuildHostScene(gpu, scene, triangles, false);
			HostScene host = BuildHostScene(gpu, scene, triangles, true);
			// In-place update (refit) of every BLAS after the conversion rewrote its inputs.
			{
				gpu.ResetStamps();
				gpu.Run([&](vk::CommandBuffer cmd) {
					gpu.Stamp(cmd, 0);
					for (auto& b: host.blas) {
						RecordBuild(cmd, b.desc, b.accel, b.accel, b.scratch);
					}
					RecordAccelBarrier(cmd, false);
					gpu.Stamp(cmd, 1);
				});
				std::printf("    BLAS updates (refit) %.3f ms\n", gpu.Millis(0, 1));
			}
			const auto kinds =
			    triangles ? std::vector<TraceKind> {TraceKind::Triangles}
			              : std::vector<TraceKind> {TraceKind::Software, TraceKind::Hybrid};
			for (const auto kind: kinds) {
				const char* name = kind == TraceKind::Software ? "software (GLSL, per-ray stack)"
				                   : kind == TraceKind::Hybrid ? "hybrid ray query"
				                                               : "triangle ray query";
				const auto  cam  = Trace(gpu, host, scene, cam_buf, n, kind, 3);
				const auto  rnd  = Trace(gpu, host, scene, rnd_buf, n_rnd, kind, 3);
				uint64_t    hits_cam = 0, hits_rnd = 0, cand_cam = 0, cand_rnd = 0, nodes = 0;
				for (const auto& r: cam.results) {
					hits_cam += r.hit;
					cand_cam += r.candidates;
				}
				for (const auto& r: rnd.results) {
					hits_rnd += r.hit;
					cand_rnd += r.candidates;
					nodes += r.nodes;
				}
				const char* what = kind == TraceKind::Software ? "tri tests" : "leaf candidates";
				std::printf(
				    "    %-30s camera 1080p %8.3f ms %6.0f Mrays/s %3u%% hit %6.1f %s/ray | "
				    "random 512k %8.3f ms %6.0f Mrays/s %3u%% hit %6.1f %s/ray",
				    name, cam.ms, n / cam.ms / 1e3, static_cast<uint32_t>(hits_cam * 100 / n),
				    static_cast<double>(cand_cam) / n, what, rnd.ms, n_rnd / rnd.ms / 1e3,
				    static_cast<uint32_t>(hits_rnd * 100 / n_rnd),
				    static_cast<double>(cand_rnd) / n_rnd, what);
				if (kind == TraceKind::Software) {
					std::printf(", %.0f nodes/ray", static_cast<double>(nodes) / n_rnd);
				}
				std::printf("\n");
			}
		}
		gpu.Release(marker);
	}
}

// Writes bit-exact type-6 vectors (decoded by DecodeBox) as a C++ include for other test suites.
bool EmitType6Vectors(const char* path) {
	std::FILE* f = std::fopen(path, "w");
	if (f == nullptr) {
		return false;
	}
	std::fprintf(
	    f, "// Psr shared-exponent (type 6) box nodes and their decode (psrBvh.cpp, DecodeBox).\n"
	       "// Generated by rt_hardware_tests --emit-type6-vectors. Bounds are float bits,\n"
	       "// child-major: min x, min y, min z, max x, max y, max z.\n"
	       "struct PsrType6Vector {\n"
	       "\tuint32_t node[16];\n"
	       "\tuint32_t children[4];\n"
	       "\tuint32_t bounds[24];\n"
	       "};\n"
	       "inline constexpr PsrType6Vector kPsrType6Vectors[] = {\n");
	std::mt19937                            rng(2026);
	std::uniform_real_distribution<float>   coord(-500.0f, 500.0f);
	std::uniform_real_distribution<float>   extent(0.0f, 40.0f);
	std::uniform_int_distribution<uint32_t> type_pick(0, 7);
	for (int n = 0; n < 64; n++) {
		std::array<Aabb, 4>     boxes {};
		std::array<uint32_t, 4> types {};
		std::array<uint32_t, 4> fields {};
		for (uint32_t c = 0; c < 4; c++) {
			for (int a = 0; a < 3; a++) {
				const float scale = n % 4 == 0 ? 1e-4f : n % 4 == 1 ? 1e3f : 1.0f;
				boxes[c].min[a]   = coord(rng) * scale;
				boxes[c].max[a]   = boxes[c].min[a] + extent(rng) * scale;
			}
			types[c]  = type_pick(rng);
			fields[c] = (n + c) % 5 == 0 ? Psr::EmptyChildField : (n + c) % 3;
		}
		const auto bytes = EncodeSharedExpBox((static_cast<uint32_t>(n) * 104729u) & 0x1fffffffu,
		                                      types, fields, boxes);
		BoxNode    node {};
		DecodeBox(Psr::NodeType::BoxPs5, bytes, node);
		uint32_t words[16];
		std::memcpy(words, bytes.data(), sizeof(words));
		std::fprintf(f, "    {{");
		for (int i = 0; i < 16; i++) {
			std::fprintf(f, "0x%08xu%s", words[i], i < 15 ? ", " : "");
		}
		std::fprintf(f, "},\n     {");
		for (int c = 0; c < 4; c++) {
			std::fprintf(f, "0x%08xu%s", node.children[c], c < 3 ? ", " : "");
		}
		std::fprintf(f, "},\n     {");
		for (int c = 0; c < 4; c++) {
			for (int k = 0; k < 6; k++) {
				const float v = k < 3 ? node.bounds[c].min[k] : node.bounds[c].max[k - 3];
				std::fprintf(f, "0x%08xu%s", std::bit_cast<uint32_t>(v),
				             c * 6 + k < 23 ? ", " : "");
			}
		}
		std::fprintf(f, "}},\n");
	}
	std::fprintf(f, "};\n");
	std::fclose(f);
	return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// Offline analysis of a KYTY_RT_DUMP_BVH dump (the game's own BVHs)

// Whether a leaf list at `list` is plausible for a child with parent-stored `bounds`: 1..64
// entries ending with the last flag or a type-7 entry, triangle pointers of type 0..3 whose nodes
// lie in [blas_base, extent_end), and every vertex finite and inside the bounds (small slack).
bool PlausibleLeafList(const ReadMemory& read, uint64_t blas_base, uint64_t list,
                       uint64_t extent_end, const Aabb& bounds) {
	for (uint32_t i = 0; i < 64u; i++) {
		LeafEntry entry {};
		if (!read(list + uint64_t {i} * Psr::LeafEntryBytes, &entry, sizeof(entry))) {
			return false;
		}
		const auto type = Psr::TypeOf(entry.triangle_pointer);
		if (type == Psr::NodeType::Special) {
			return i > 0;
		}
		if (static_cast<uint32_t>(type) > 3u) {
			return false;
		}
		const uint64_t node = Psr::HardwareNodeAddress(blas_base, entry.triangle_pointer);
		std::array<uint8_t, Psr::TriangleNodeBytes> bytes {};
		Triangle                                    tri {};
		if (node + Psr::TriangleNodeBytes > extent_end || !read(node, bytes.data(), bytes.size()) ||
		    !DecodeTriangle(type, bytes, tri)) {
			return false;
		}
		for (const auto& v: tri.v) {
			for (int a = 0; a < 3; a++) {
				const float slack =
				    1e-3f * std::max({1.0f, std::fabs(bounds.min[a]), std::fabs(bounds.max[a])});
				if (!std::isfinite(v[a]) || v[a] < bounds.min[a] - slack ||
				    v[a] > bounds.max[a] + slack) {
					return false;
				}
			}
		}
		if ((entry.info & Psr::LeafLastFlag) != 0u) {
			return true;
		}
	}
	return false;
}

struct ChildTally {
	uint64_t leaves = 0, leaf_bytes = 0, leaf_units = 0, leaf_both = 0, leaf_neither = 0;
	uint64_t boxes = 0, boxes_contained = 0;
};

// Checks, per parent box type, how leaf children address their entry lists (byte offset, as the
// traversal reads them under box16/box32 parents, or 64-byte units like node pointers) and whether
// box children's own bounds lie inside the bounds their parent stores for them.
void AnalyzeChildren(GuestSnapshot& snapshot, const BvhCapture& capture,
                     std::array<ChildTally, 3>& tally) {
	const auto read = snapshot.Reader();
	for (const auto& b: capture.blas) {
		if (!b.header_read || !Psr::IsBox(b.root)) {
			continue;
		}
		const uint64_t        end = b.base + b.extent;
		std::vector<uint32_t> stack {b.root};
		for (uint32_t visited = 0; !stack.empty() && visited < (1u << 22u); visited++) {
			const uint32_t pointer = stack.back();
			stack.pop_back();
			BoxNode    node {};
			WalkStatus status = WalkStatus::Ok;
			if (!ReadBox(read, b.base, pointer, node, status)) {
				continue;
			}
			auto& t = tally[(pointer & 7u) - 4u];
			for (uint32_t c = 0; c < 4; c++) {
				const uint32_t child = node.children[c];
				if (child == Psr::InvalidNode) {
					continue;
				}
				const auto& bounds = node.bounds[c];
				if (Psr::IsBox(child)) {
					t.boxes++;
					BoxNode inner {};
					if (ReadBox(read, b.base, child, inner, status)) {
						bool inside = true;
						for (uint32_t k = 0; k < 4; k++) {
							if (inner.children[k] == Psr::InvalidNode) {
								continue;
							}
							for (int a = 0; a < 3; a++) {
								const float slack =
								    1e-3f * std::max({1.0f, std::fabs(bounds.min[a]),
								                      std::fabs(bounds.max[a])});
								inside = inside &&
								         inner.bounds[k].min[a] >= bounds.min[a] - slack &&
								         inner.bounds[k].max[a] <= bounds.max[a] + slack;
							}
						}
						t.boxes_contained += inside ? 1u : 0u;
					}
					stack.push_back(child);
					continue;
				}
				t.leaves++;
				const uint64_t offset = child & ~7u;
				const bool bytes = PlausibleLeafList(read, b.base, b.base + offset, end, bounds);
				const bool units =
				    PlausibleLeafList(read, b.base, b.base + (offset << 3u), end, bounds);
				t.leaf_bytes += bytes && !units ? 1u : 0u;
				t.leaf_units += units && !bytes ? 1u : 0u;
				t.leaf_both += bytes && units ? 1u : 0u;
				t.leaf_neither += !bytes && !units ? 1u : 0u;
			}
		}
	}
}

int AnalyzeDump(const char* path, uint32_t ray_count) {
	GuestSnapshot snapshot;
	if (!snapshot.Load(path)) {
		std::printf("cannot load %s\n", path);
		return 2;
	}
	std::printf("%s: %zu pages\n", path, snapshot.Pages().size());
	uint32_t blas_lines = 0;
	for (const auto& [key, value]: snapshot.meta) {
		if (key.rfind("blas.", 0) == 0) {
			blas_lines++;
		} else {
			std::printf("  %s = %s\n", key.c_str(), value.c_str());
		}
	}
	std::printf("  (%u blas.* lines)\n", blas_lines);
	const uint64_t tlas =
	    std::strtoull(snapshot.MetaValue("capture.tlas", "0").c_str(), nullptr, 0);

	// The same walks over the loaded pages only (a loaded snapshot fetches nothing).
	const auto capture = CaptureBvh(snapshot, tlas, BvhCaptureLimits {});
	std::printf("Offline walk:\n%s", DescribeCapture(capture, snapshot).c_str());

	std::array<ChildTally, 3> tally {};
	AnalyzeChildren(snapshot, capture, tally);
	const char* names[3] = {"box16 (4)", "box32 (5)", "type 6"};
	std::printf(
	    "Children by parent type (leaf lists: byte offset / 64-byte units / both / neither; "
	    "box children inside the parent's bounds):\n");
	for (int i = 0; i < 3; i++) {
		const auto& t = tally[i];
		std::printf("  %-10s leaves %" PRIu64 ": %" PRIu64 " / %" PRIu64 " / %" PRIu64 " / %" PRIu64
		            "; boxes %" PRIu64 ": %" PRIu64 " contained\n",
		            names[i], t.leaves, t.leaf_bytes, t.leaf_units, t.leaf_both, t.leaf_neither,
		            t.boxes, t.boxes_contained);
	}

	// Rays inside the TLAS root bounds: the reference traversal with the guest's 60-entry stack,
	// without the limit, and against brute force for the first rays.
	const auto read = snapshot.Reader();
	BoxNode    root {};
	WalkStatus status = WalkStatus::Ok;
	if (!ReadBox(read, tlas, Psr::RootPointer, root, status)) {
		std::printf("TLAS root unreadable: %s\n", ToString(status));
		return 1;
	}
	Aabb world = Aabb::Empty();
	for (uint32_t c = 0; c < 4; c++) {
		if (root.children[c] != Psr::InvalidNode) {
			world.Grow(root.bounds[c]);
		}
	}
	if (world.IsEmpty()) {
		std::printf("TLAS is empty\n");
		return 0;
	}
	TlasWalk tlas_walk;
	WalkTlas(read, tlas, tlas_walk);
	std::map<std::pair<uint64_t, uint32_t>, BlasTriangles> expanded;
	uint64_t                                               total_triangles = 0;
	for (const auto& inst: tlas_walk.instances) {
		const auto key = std::make_pair(inst.blas_address, inst.root);
		if (expanded.contains(key)) {
			continue;
		}
		BlasWalk walk;
		WalkBlas(read, inst.blas_address, inst.root, walk);
		ExpandTriangles(read, inst.blas_address, walk, expanded[key]);
		total_triangles += expanded[key].triangles.size();
	}
	std::mt19937                          rng(1234);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);
	std::normal_distribution<float>       normal(0.0f, 1.0f);
	float                                 diagonal = 0.0f;
	for (int a = 0; a < 3; a++) {
		diagonal += (world.max[a] - world.min[a]) * (world.max[a] - world.min[a]);
	}
	diagonal               = std::sqrt(diagonal);
	const auto     tests   = DefaultNodeTests();
	const uint32_t brute_n = std::min<uint32_t>(ray_count, total_triangles > 2000000 ? 16u : 64u);
	uint32_t       hits = 0, stack_differ = 0, brute_mismatch = 0, failures = 0;
	uint64_t       nodes = 0;
	for (uint32_t i = 0; i < ray_count; i++) {
		std::array<float, 3> origin {};
		std::array<float, 3> dir {};
		float                length = 0.0f;
		for (int a = 0; a < 3; a++) {
			origin[a] = world.min[a] + (0.05f + 0.9f * unit(rng)) * (world.max[a] - world.min[a]);
			dir[a]    = normal(rng);
			length += dir[a] * dir[a];
		}
		for (auto& d: dir) {
			d /= std::sqrt(std::max(length, 1e-12f));
		}
		const Ray   ray = MakeRay(origin, dir, diagonal);
		TraceResult guest {};
		TraceResult deep {};
		failures += TraceReference(read, tlas, ray, tests, guest, 60) != WalkStatus::Ok ? 1u : 0u;
		TraceReference(read, tlas, ray, tests, deep, 1u << 16u);
		hits += guest.hit ? 1u : 0u;
		nodes += guest.nodes_visited;
		stack_differ += (guest.hit != deep.hit || guest.t != deep.t) ? 1u : 0u;
		if (i < brute_n) {
			TraceResult best {};
			float       best_t = ray.extent;
			for (const auto& inst: tlas_walk.instances) {
				if (inst.Mask() == 0u) {
					continue;
				}
				const Ray   obj      = TransformRay(ray, inst.world_to_object);
				const auto& tris     = expanded.at({inst.blas_address, inst.root});
				const bool  cull_off = (inst.flags & Psr::CullDisableFlag) != 0u;
				const bool  keep_neg = (inst.flags & Psr::FacingFlag) != 0u;
				for (const auto& tri: tris.triangles) {
					const auto hit = ExactTriangleTest(tri, obj);
					if (hit && (cull_off || hit->positive != keep_neg) && !(hit->t > best_t)) {
						best_t   = hit->t;
						best.hit = true;
						best.t   = hit->t;
					}
				}
			}
			brute_mismatch += (best.hit != deep.hit || (best.hit && best.t != deep.t)) ? 1u : 0u;
		}
	}
	std::printf(
	    "Rays: %u inside the TLAS bounds (%" PRIu64 " triangles over %zu distinct BLAS): "
	    "%u hits, %.1f nodes/ray, %u trace failures; the 60-entry stack changes %u results; "
	    "brute force differs on %u of %u\n",
	    ray_count, total_triangles, expanded.size(), hits,
	    static_cast<double>(nodes) / std::max(ray_count, 1u), failures, stack_differ,
	    brute_mismatch, brute_n);
	return 0;
}

// A dump of a synthetic scene (type-6 interiors), to exercise --bvh-dump without a game run.
bool WriteTestDump(const char* path) {
	SceneParams params {};
	params.shared_exp         = true;
	params.triangles_per_blas = 2048;
	params.seed               = 5;
	const Scene   scene       = RtTest::MakeScene(params);
	GuestSnapshot snapshot(SceneFetch(scene.memory, 0, 0, nullptr));
	const auto    capture = CaptureBvh(snapshot, scene.tlas.base, BvhCaptureLimits {});
	snapshot.meta.emplace_back("dump", "rt_hardware_tests --write-test-dump");
	AppendCaptureMeta(capture, snapshot, snapshot.meta);
	return snapshot.Save(path);
}

// --bvh-bench <file.kbvh> [--rays N]: the hardware path on a game's own BVHs. The dump's pages
// become the scene memory; the GPU converts every BLAS (one AABB per leaf) and the TLAS, builds
// the acceleration structures, then traces the same random rays inside the TLAS bounds with the
// GLSL software traversal (a lower bound on the guest routine's cost) and the hybrid ray query
// (RT cores for the boxes, the instruction's exact triangle test per leaf candidate). Reports
// times, Mrays/s, AS memory and how often the two disagree.
int BenchDump(const char* path, uint32_t ray_count) {
	GuestSnapshot snapshot;
	if (!snapshot.Load(path)) {
		std::printf("cannot load %s\n", path);
		return 2;
	}
	Scene scene;
	{
		uint64_t             run_begin = 0;
		std::vector<uint8_t> run;
		uint64_t             last_page = ~0ull;
		const auto           flush     = [&] {
            if (!run.empty()) scene.memory.Insert(run_begin, std::move(run));
            run.clear();
		};
		for (const auto& [number, page]: snapshot.Pages()) {
			if (number != last_page + 1u) {
				flush();
				run_begin = number * GuestSnapshot::PageBytes;
			}
			run.insert(run.end(), page.bytes.begin(), page.bytes.end());
			last_page = number;
		}
		flush();
	}
	scene.tlas.base = std::strtoull(snapshot.MetaValue("capture.tlas", "0").c_str(), nullptr, 0);
	const auto read = scene.memory.Reader();
	TlasWalk   tlas_walk;
	if (WalkTlas(read, scene.tlas.base, tlas_walk) != WalkStatus::Ok) {
		std::printf("TLAS walk failed\n");
		return 1;
	}
	for (const auto& inst: tlas_walk.instances) {
		RtTest::PsrBvhWriter::InstanceDesc desc {};
		desc.blas_address = inst.blas_address;
		desc.blas_root    = inst.root;
		scene.instances.push_back(desc);
	}
	BoxNode    root {};
	WalkStatus status = WalkStatus::Ok;
	if (!ReadBox(read, scene.tlas.base, Psr::RootPointer, root, status)) {
		std::printf("TLAS root unreadable\n");
		return 1;
	}
	Aabb world = Aabb::Empty();
	for (uint32_t c = 0; c < 4; c++) {
		if (root.children[c] != Psr::InvalidNode) world.Grow(root.bounds[c]);
	}
	float diagonal = 0.0f;
	for (int a = 0; a < 3; a++) diagonal += (world.max[a] - world.min[a]) * (world.max[a] - world.min[a]);
	diagonal = std::sqrt(diagonal);
	// Random rays from inside the TLAS bounds (reflection- and GI-like: any direction, full length).
	std::mt19937                          rng(1234);
	std::uniform_real_distribution<float> unit(0.0f, 1.0f);
	std::normal_distribution<float>       normal(0.0f, 1.0f);
	std::vector<GpuRay>                   rays(ray_count);
	for (auto& g: rays) {
		std::array<float, 3> origin {};
		std::array<float, 3> dir {};
		float                length = 0.0f;
		for (int a = 0; a < 3; a++) {
			origin[a] = world.min[a] + (0.05f + 0.9f * unit(rng)) * (world.max[a] - world.min[a]);
			dir[a]    = normal(rng);
			length += dir[a] * dir[a];
		}
		for (auto& d: dir) d /= std::sqrt(std::max(length, 1e-12f));
		const Ray r = MakeRay(origin, dir, diagonal);
		for (int a = 0; a < 3; a++) {
			g.o[a]   = r.origin[a];
			g.d[a]   = r.dir[a];
			g.inv[a] = r.inv_dir[a];
		}
		g.extent = r.extent;
	}
	Gpu gpu;
	if (!gpu.Init()) {
		std::printf("no device with acceleration structures and ray query\n");
		return 1;
	}
	std::printf("%s on %s: %zu instances, %zu guest pages\n", path, gpu.Name(), scene.instances.size(),
	            snapshot.Pages().size());
	PrintSupport(gpu);
	(void)BuildHostScene(gpu, scene, false, false);
	HostScene host = BuildHostScene(gpu, scene, false, true);
	std::printf("  hybrid conversion: BLAS walk+convert %.3f ms, BLAS builds %.3f ms (%zu BLAS, %" PRIu64
	            " leaves), TLAS convert %.3f ms + build %.3f ms, AS %.1f MiB, inputs %.1f MiB, status 0x%x\n",
	            host.convert_ms, host.blas_build_ms, host.blas.size(), host.primitives, host.tlas_convert_ms,
	            host.tlas_ms - host.tlas_convert_ms, host.accel_bytes / 1048576.0, host.input_bytes / 1048576.0,
	            host.status_bits);
	Buffer     ray_buf = gpu.Upload(rays.data(), rays.size() * sizeof(GpuRay));
	const auto sw      = Trace(gpu, host, scene, ray_buf, ray_count, TraceKind::Software, 3);
	const auto hw      = Trace(gpu, host, scene, ray_buf, ray_count, TraceKind::Hybrid, 3);
	uint64_t   hits = 0, nodes = 0, cand = 0, differ = 0, statuses = 0;
	for (uint32_t i = 0; i < ray_count; i++) {
		const auto& a = sw.results[i];
		const auto& b = hw.results[i];
		hits += a.hit;
		nodes += a.nodes;
		cand += b.candidates;
		statuses += (a.status | b.status) != 0u ? 1u : 0u;
		differ += (a.hit != b.hit || (a.hit != 0u && a.t != b.t)) ? 1u : 0u;
	}
	std::printf("  %u rays: software %.3f ms (%.0f Mrays/s, %.1f nodes/ray) | hybrid %.3f ms (%.0f Mrays/s, "
	            "%.1f leaf candidates/ray) | speed-up %.1fx | %" PRIu64 "%% hit, %" PRIu64
	            " rays differ, %" PRIu64 " with status\n",
	            ray_count, sw.ms, ray_count / sw.ms / 1e3, static_cast<double>(nodes) / ray_count, hw.ms,
	            ray_count / hw.ms / 1e3, static_cast<double>(cand) / ray_count, sw.ms / std::max(hw.ms, 1e-6),
	            hits * 100 / std::max<uint64_t>(ray_count, 1u), differ, statuses);
	return 0;
}

int main(int argc, char** argv) {
	bool        cpu_only = false;
	bool        bench    = false;
	const char* dump       = nullptr;
	const char* dump_bench = nullptr;
	uint32_t    rays       = 2000;
	for (int i = 1; i < argc; i++) {
		if (std::strcmp(argv[i], "--cpu-only") == 0) {
			cpu_only = true;
		} else if (std::strcmp(argv[i], "--bench") == 0) {
			bench = true;
		} else if (std::strcmp(argv[i], "--emit-type6-vectors") == 0 && i + 1 < argc) {
			return EmitType6Vectors(argv[i + 1]) ? 0 : 1;
		} else if (std::strcmp(argv[i], "--write-test-dump") == 0 && i + 1 < argc) {
			return WriteTestDump(argv[i + 1]) ? 0 : 1;
		} else if (std::strcmp(argv[i], "--bvh-dump") == 0 && i + 1 < argc) {
			dump = argv[++i];
		} else if (std::strcmp(argv[i], "--bvh-bench") == 0 && i + 1 < argc) {
			dump_bench = argv[++i];
		} else if (std::strcmp(argv[i], "--rays") == 0 && i + 1 < argc) {
			rays = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
		}
	}
	if (dump != nullptr) {
		return AnalyzeDump(dump, rays);
	}
	if (dump_bench != nullptr) {
		return BenchDump(dump_bench, rays);
	}
	if (!bench) {
		RunCpuTests();
	}
	if (!cpu_only) {
		Gpu gpu;
		if (!gpu.Init()) {
			std::printf(
			    "GPU tests skipped: no device with acceleration structures and ray query\n");
		} else if (bench) {
			RunBench(gpu);
		} else {
			RunGpuTests(gpu);
		}
	}
	if (g_failures != 0) {
		std::printf("%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("all checks passed\n");
	return 0;
}
