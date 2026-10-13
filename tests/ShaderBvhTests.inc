// IMAGE_BVH_INTERSECT_RAY / IMAGE_BVH64_INTERSECT_RAY (MIMG 0xe6/0xe7) tests (included by
// ShaderRecompilerComputeTests.cpp after ShaderCodegenTests.inc). Specification:
// Profiling/analysis/RT-SOFTWARE-DESIGN.md.
//
// - Ref: a CPU reference of one node test (spec sections 3-7), checked against hand-computed
//   results, then used as the oracle for the GPU (KYTY_RT_SOFTWARE) in directed and fuzzed cases.
// - Stub: KYTY_RT_STUB returns misses.
// - Benchmark (--bvh-bench): GPU cost per million node tests.

namespace BvhTests {

using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;

// ----------------------------------------------------------------------------------------------
// Encoding

// RDNA2 ISA 8.2.10: DMASK=0xf, UNRM=1, R128=1, DIM=0 (1D), no sampler.
constexpr u32 EncodeBvh0(bool bvh64, u32 nsa_dwords = 0) {
  const u32 opcode = bvh64 ? 0xe7u : 0xe6u;
  return (0x3cu << 26u) | ((opcode >> 7u) & 0x1u) | ((nsa_dwords & 0x3u) << 1u) |
         (0xfu << 8u) | (1u << 12u) | (1u << 15u) | ((opcode & 0x7fu) << 18u);
}

constexpr u32 EncodeBvh1(u32 vdata, u32 vaddr, u32 srsrc_sgpr, bool a16) {
  return (((srsrc_sgpr / 4u) & 0x1fu) << 16u) | ((vdata & 0xffu) << 8u) | (vaddr & 0xffu) |
         (a16 ? (1u << 30u) : 0u);
}

constexpr u32 BvhAddressDwords(bool bvh64, bool a16) {
  return (bvh64 ? 2u : 1u) + 4u + (a16 ? 3u : 6u);
}

// Appends the instruction with its result in v[vdata:vdata+3]. Address dword 0 is VADDR; with
// NSA, dword i >= 1 comes from nsa_regs[i - 1].
void AppendBvh(std::vector<u32> *code, bool bvh64, bool a16, u32 vdata, u32 vaddr,
               u32 srsrc_sgpr, const std::vector<u32> &nsa_regs = {}) {
  const u32 nsa_dwords = static_cast<u32>((nsa_regs.size() + 3u) / 4u);
  code->push_back(EncodeBvh0(bvh64, nsa_dwords));
  code->push_back(EncodeBvh1(vdata, vaddr, srsrc_sgpr, a16));
  for (u32 dword = 0; dword < nsa_dwords; ++dword) {
    u32 packed = 0;
    for (u32 byte = 0; byte < 4u; ++byte) {
      const u32 index = dword * 4u + byte;
      if (index < nsa_regs.size()) packed |= (nsa_regs[index] & 0xffu) << (byte * 8u);
    }
    code->push_back(packed);
  }
}

// CPU reference (spec sections 3-7): src/graphics/shader/recompiler/BvhReference.h
namespace Ref = ::Libs::Graphics::ShaderRecompiler::BvhReference;
using Ref::B;
using Ref::BoxMiss;
using Ref::F;
using Ref::HalfToFloat;
using Ref::InvalidNode;
using Ref::MinusInf;
using Ref::OneF;
using Ref::PlusInf;
using Ref::TriangleMiss;

// Exact binary32 -> binary16 for test data (bounds and A16 rays are chosen representable).
inline u32 FloatToHalf(float value) {
  const u32 bits = B(value);
  const u32 sign = (bits >> 16u) & 0x8000u;
  const u32 abs = bits & 0x7fffffffu;
  if (abs >= 0x7f800000u) return sign | 0x7c00u | (abs > 0x7f800000u ? 0x200u : 0u);
  if (abs == 0u) return sign;
  const int exponent = static_cast<int>(abs >> 23u) - 127;
  Require("BvhTests", "FloatToHalf", exponent >= -14 && exponent <= 15 &&
              (abs & 0x1fffu) == 0u,
          "test value is not an exact normal binary16");
  return sign | (static_cast<u32>(exponent + 15) << 10u) | ((abs >> 13u) & 0x3ffu);
}

#include "ShaderBvhType6Vectors.inc"

// ----------------------------------------------------------------------------------------------
// Test BVH images

// The whole test buffer is guest memory at GuestBase (BDA mapping at buffer offset 0):
// [0, 4 KiB) results, [RecordOffset, +64 per lane) ray records, [NodeOffset, ...) nodes.
constexpr uint64_t GuestBase = 0x0000000110000000ull;
constexpr u32 RecordOffset = 4096;
constexpr u32 RecordDwords = 16;
constexpr u32 NodeOffset = 16384;
// The T# base covers the node region: node byte offset X is pointer (X >> 3) | type.
constexpr u32 TsharpBase0 = static_cast<u32>((GuestBase + NodeOffset) >> 8u);
constexpr u32 TsharpBase1 = static_cast<u32>(((GuestBase + NodeOffset) >> 40u) & 0xffu);

struct Image {
  std::vector<u32> nodes; // dwords at NodeOffset

  void Put(u32 byte_offset, u32 value) {
    const u32 index = byte_offset / 4u;
    if (nodes.size() <= index) nodes.resize(index + 1u, 0u);
    nodes[index] = value;
  }
  [[nodiscard]] u32 Get(u32 byte_offset) const {
    const u32 index = byte_offset / 4u;
    return index < nodes.size() ? nodes[index] : 0u;
  }
  void Box32(u32 offset, const std::array<u32, 4> &children,
             const std::array<std::array<float, 6>, 4> &bounds) {
    for (u32 i = 0; i < 4u; ++i) Put(offset + i * 4u, children[i]);
    for (u32 child = 0; child < 4u; ++child) {
      for (u32 k = 0; k < 6u; ++k) Put(offset + 16u + child * 24u + k * 4u, B(bounds[child][k]));
    }
    for (u32 k = 112u; k < 128u; k += 4u) Put(offset + k, 0u);
  }
  void Box16(u32 offset, const std::array<u32, 4> &children,
             const std::array<std::array<float, 6>, 4> &bounds) {
    for (u32 i = 0; i < 4u; ++i) Put(offset + i * 4u, children[i]);
    for (u32 child = 0; child < 4u; ++child) {
      for (u32 pair = 0; pair < 3u; ++pair) {
        Put(offset + 16u + child * 12u + pair * 4u,
            FloatToHalf(bounds[child][pair * 2u]) |
                (FloatToHalf(bounds[child][pair * 2u + 1u]) << 16u));
      }
    }
  }
  void Triangles(u32 offset, const std::array<std::array<float, 3>, 5> &vertices,
                 u32 triangle_id) {
    for (u32 v = 0; v < 5u; ++v) {
      for (u32 axis = 0; axis < 3u; ++axis) Put(offset + (v * 3u + axis) * 4u, B(vertices[v][axis]));
    }
    Put(offset + 60u, triangle_id);
  }
  // Guest memory view for the reference: the node region, zero elsewhere.
  [[nodiscard]] Ref::Memory Memory() const {
    return [this](uint64_t address) -> u32 {
      const uint64_t start = GuestBase + NodeOffset;
      if (address < start || address - start >= nodes.size() * 4u) return 0u;
      return nodes[static_cast<size_t>((address - start) / 4u)];
    };
  }
};

constexpr uint64_t Pointer(u32 byte_offset, u32 type) {
  return (uint64_t{byte_offset} >> 3u) | type;
}

// T#: base = node region, size = nodes - 1, type 8.
constexpr std::array<u32, 4> Tsharp(bool sort = true, bool ij = true, u32 grow = 0,
                                    uint64_t size = 0xffffffull) {
  return {TsharpBase0, TsharpBase1 | ((grow & 0xffu) << 23u) | (sort ? 0x80000000u : 0u),
          static_cast<u32>(size),
          static_cast<u32>((size >> 32u) & 0x3ffu) | (ij ? (1u << 24u) : 0u) | 0x80000000u};
}

// One lane: raw register contents (A16 keeps f16 direction halves).
struct Lane {
  uint64_t node = 0;
  u32 extent = 0;
  std::array<u32, 3> origin{};
  std::array<u32, 3> dir{};
  std::array<u32, 3> inv{};
};

Lane MakeLane(uint64_t node, float extent, std::array<float, 3> origin,
              std::array<float, 3> dir, std::array<float, 3> inv) {
  Lane lane;
  lane.node = node;
  lane.extent = B(extent);
  for (u32 axis = 0; axis < 3u; ++axis) {
    lane.origin[axis] = B(origin[axis]);
    lane.dir[axis] = B(dir[axis]);
    lane.inv[axis] = B(inv[axis]);
  }
  return lane;
}

// A lane as the instruction sees it: A16 narrows direction and inverse direction to f16.
Ref::Ray RayOf(const Lane &lane, bool a16) {
  Ref::Ray ray;
  ray.extent = F(lane.extent);
  for (u32 axis = 0; axis < 3u; ++axis) {
    ray.origin[axis] = F(lane.origin[axis]);
    ray.dir[axis] = a16 ? HalfToFloat(FloatToHalf(F(lane.dir[axis]))) : F(lane.dir[axis]);
    ray.inv[axis] = a16 ? HalfToFloat(FloatToHalf(F(lane.inv[axis]))) : F(lane.inv[axis]);
  }
  return ray;
}

// v1 = RecordOffset + lane * 64: the lane's record (MUBUF immediate offsets have only 12 bits).
void AppendRecordAddress(std::vector<u32> *code) {
  static_assert(RecordOffset % 64u == 0u && RecordOffset + 64u * 64u <= NodeOffset);
  code->push_back(EncodeVop2(0x1a, 1, InlineU32(6), 0)); // v_lshlrev_b32 v1, 6, v0
  code->push_back(EncodeVop2(0x25, 1, 255u, 1));         // v_add_nc_u32 v1, RecordOffset, v1
  code->push_back(RecordOffset);
}

struct Scenario {
  std::string name;
  std::array<u32, 4> tsharp{};
  Image image;
  std::vector<Lane> lanes;
  bool ps5_type6 = true; // KYTY_RT_TYPE6 (the reference must match the codegen option)
  // Guest memory from SplitGuest on lives SplitGap bytes further into the buffer, so the halves of
  // a node that spans that page boundary are not contiguous in device memory (the gap holds
  // garbage): a lowering that reuses the first half's page lookup for the second would fail.
  bool split_pages = false;
};

constexpr u32 SplitGuest = 0x8000; // buffer offset = guest offset below this
constexpr u32 SplitGap = 0x4000;   // one BDA page

// Destination contents of lanes that EXEC turns off (the instruction must leave them).
constexpr u32 Sentinel = 0xdead0000u;

// Builds the GPU case: each lane loads its record, runs the instruction and stores the four
// result dwords at [dword * wave + lane]. Expected values come from the reference. Lanes whose
// bit in `exec` is clear run the instruction inactive and keep Sentinel + dword.
TestCase MakeCase(const Scenario &scenario, u32 wave_size, bool bvh64, bool a16, bool nsa,
                  bool flush, uint64_t exec = ~uint64_t{0}) {
  using O = ShaderOpcode;
  TestCase test;
  test.name = scenario.name.c_str();
  std::vector<u32> &code = test.code;
  const u32 dwords = BvhAddressDwords(bvh64, a16);
  AppendRecordAddress(&code);
  // Record layout: node_lo, node_hi, extent, origin xyz, then dir xyz + inv xyz, or with A16
  // {dir.x, dir.y}, {dir.z, inv.x}, {inv.y, inv.z}.
  std::vector<u32> nsa_regs;
  for (u32 k = 0; k < dwords; ++k) {
    const u32 record = bvh64 ? k : (k == 0u ? 0u : k + 1u);
    const u32 reg = (k == 0u || !nsa) ? 40u + k : 90u - k;
    code.push_back(EncodeMubuf0(0x0cu, record * 4u));
    code.push_back(EncodeMubuf1(reg, 0, 1));
    if (nsa && k != 0u) nsa_regs.push_back(reg);
  }
  for (u32 i = 0; i < 4u; ++i) AppendSMovLiteral(&code, 32u + i, scenario.tsharp[i]);
  const uint64_t all_lanes = wave_size == 64u ? ~uint64_t{0} : 0xffffffffull;
  exec &= all_lanes;
  if (exec != all_lanes) {
    for (u32 i = 0; i < 4u; ++i) AppendVMovLiteral(&code, 16u + i, Sentinel + i);
    AppendSMovLiteral(&code, 126, static_cast<u32>(exec)); // exec_lo
    AppendSMovLiteral(&code, 127, static_cast<u32>(exec >> 32u));
  }
  AppendBvh(&code, bvh64, a16, 16, 40, 32, nsa_regs);
  if (exec != all_lanes) {
    AppendSMovLiteral(&code, 126, static_cast<u32>(all_lanes));
    AppendSMovLiteral(&code, 127, static_cast<u32>(all_lanes >> 32u));
  }
  for (u32 dword = 0; dword < 4u; ++dword) {
    AppendStoreVgprAtLaneDwordOffset(&code, 16u + dword, 0, dword * wave_size);
  }
  AppendEnd(&code);

  test.initial.assign(NodeOffset / 4u + scenario.image.nodes.size(), 0u);
  std::copy(scenario.image.nodes.begin(), scenario.image.nodes.end(),
            test.initial.begin() + NodeOffset / 4u);
  test.bda_mappings = {{GuestBase, 0}};
  if (scenario.split_pages && test.initial.size() > SplitGuest / 4u) {
    const std::vector<u32> moved(test.initial.begin() + SplitGuest / 4u, test.initial.end());
    test.initial.resize((SplitGuest + SplitGap) / 4u + moved.size());
    std::fill(test.initial.begin() + SplitGuest / 4u,
              test.initial.begin() + (SplitGuest + SplitGap) / 4u, 0xbad0bad0u);
    std::copy(moved.begin(), moved.end(), test.initial.begin() + (SplitGuest + SplitGap) / 4u);
    test.bda_mappings.push_back({GuestBase + SplitGuest, SplitGuest + SplitGap});
  }
  const Ref::Math math{.flush = flush};
  const auto memory = scenario.image.Memory();
  test.expected.assign(4u * wave_size, 0u);
  for (u32 lane = 0; lane < wave_size; ++lane) {
    const Lane &input = scenario.lanes[lane % scenario.lanes.size()];
    auto *record = &test.initial[RecordOffset / 4u + lane * RecordDwords];
    record[0] = static_cast<u32>(input.node);
    record[1] = static_cast<u32>(input.node >> 32u);
    record[2] = input.extent;
    for (u32 axis = 0; axis < 3u; ++axis) record[3 + axis] = input.origin[axis];
    if (a16) {
      std::array<u32, 6> halves{};
      for (u32 axis = 0; axis < 3u; ++axis) {
        halves[axis] = FloatToHalf(F(input.dir[axis]));
        halves[3u + axis] = FloatToHalf(F(input.inv[axis]));
      }
      for (u32 pair = 0; pair < 3u; ++pair) {
        record[6 + pair] = halves[pair * 2u] | (halves[pair * 2u + 1u] << 16u);
      }
    } else {
      for (u32 axis = 0; axis < 3u; ++axis) {
        record[6 + axis] = input.dir[axis];
        record[9 + axis] = input.inv[axis];
      }
    }
    const uint64_t node = bvh64 ? input.node : (input.node & 0xffffffffull);
    auto result = Ref::Intersect(scenario.tsharp, node, RayOf(input, a16), memory, math,
                                 scenario.ps5_type6);
    if (((exec >> lane) & 1u) == 0u) {
      result = {Sentinel, Sentinel + 1u, Sentinel + 2u, Sentinel + 3u};
    }
    for (u32 dword = 0; dword < 4u; ++dword) {
      test.expected[dword * wave_size + lane] = result[dword];
    }
  }
  test.opcodes = {O::V_LSHLREV_B32, O::BUFFER_LOAD_DWORD, O::S_MOV_B32,
                  bvh64 ? O::IMAGE_BVH64_INTERSECT_RAY : O::IMAGE_BVH_INTERSECT_RAY,
                  O::V_ADD_NC_U32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  if (exec != all_lanes) test.opcodes.push_back(O::V_MOV_B32);
  test.compute_info.threads_num[0] = wave_size;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = wave_size;
  test.has_compute_info = true;
  return test;
}

// ----------------------------------------------------------------------------------------------
// Directed scenarios

constexpr float Inf = std::numeric_limits<float>::infinity();

// Four unit-thick boxes along +x, stored far to near: x in [7,8], [5,6], [3,4], [1,2].
constexpr std::array<std::array<float, 6>, 4> RowBounds{{{7, -1, -1, 8, 1, 1},
                                                         {5, -1, -1, 6, 1, 1},
                                                         {3, -1, -1, 4, 1, 1},
                                                         {1, -1, -1, 2, 1, 1}}};
constexpr std::array<u32, 4> RowChildren{0x1000u | 5u, 0x2000u | 4u, 0x3000u | 0u, 0x4000u | 1u};

// `offset`: the node's byte offset in the node region (a multiple of 64).
Scenario BoxScenario(bool box16, std::array<u32, 4> tsharp, const char *name, u32 offset = 0) {
  Scenario scenario;
  scenario.name = name;
  scenario.tsharp = tsharp;
  if (box16) {
    scenario.image.Box16(offset, RowChildren, RowBounds);
  } else {
    scenario.image.Box32(offset, RowChildren, RowBounds);
  }
  const uint64_t node = Pointer(offset, box16 ? 4u : 5u);
  auto &lanes = scenario.lanes;
  // 0: +x from the origin through all four; 1: extent 4.5 stops before x=5; 2: -x misses all;
  // 3: starting inside [3,4]; 4: y offset above every box; 5: grazing the top face (y = 1);
  // 6: diagonal; 7: zero extent.
  lanes.push_back(MakeLane(node, 100, {0, 0, 0}, {1, 0, 0}, {1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 4.5f, {0, 0, 0}, {1, 0, 0}, {1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 100, {0, 0, 0}, {-1, 0, 0}, {-1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 100, {3.5f, 0, 0}, {1, 0, 0}, {1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 100, {0, 2, 0}, {1, 0, 0}, {1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 100, {0, 1, 0}, {1, 0, 0}, {1, Inf, Inf}));
  lanes.push_back(MakeLane(node, 100, {0, -0.5f, -0.5f}, {1, 0.125f, 0.125f}, {1, 8, 8}));
  lanes.push_back(MakeLane(node, 0, {1.5f, 0, 0}, {1, 0, 0}, {1, Inf, Inf}));
  // 8: negative zero direction components (inverse -inf); 9: NaN inverse on y.
  lanes.push_back(MakeLane(node, 100, {0, 0, 0}, {1, -0.0f, -0.0f}, {1, -Inf, -Inf}));
  lanes.push_back(MakeLane(node, 100, {0, 0, 0}, {1, 0, 0}, {1, std::nanf(""), Inf}));
  return scenario;
}

// A fan node at 0: v0..v4 around v2 = (0, 0, 5). Types 0..3 select the four triangles.
constexpr std::array<std::array<float, 3>, 5> FanVertices{
    {{-1, -1, 5}, {1, -1, 5}, {0, 0, 5}, {1, 1, 5}, {-1, 1, 5}}};
// Byte per type: I source | J source << 2. Type 0: identity (9); others exercise the swizzle.
constexpr u32 FanTriangleId = 9u | ((2u | (1u << 2u)) << 8u) | ((0u | (2u << 2u)) << 16u) |
                              ((3u | (0u << 2u)) << 24u);

Scenario TriangleScenario(std::array<u32, 4> tsharp, const char *name) {
  Scenario scenario;
  scenario.name = name;
  scenario.tsharp = tsharp;
  scenario.image.Triangles(0, FanVertices, FanTriangleId);
  auto &lanes = scenario.lanes;
  const std::array<std::array<float, 2>, 4> inside{
      {{0, -0.5f}, {0.5f, 0}, {0, 0.5f}, {-0.5f, 0}}};
  for (u32 type = 0; type < 4u; ++type) {
    const uint64_t node = Pointer(0, type);
    const auto [x, y] = inside[type];
    // Hit from below, hit from above (other facing), miss beside, miss behind.
    lanes.push_back(MakeLane(node, 100, {x, y, 0}, {0, 0, 1}, {Inf, Inf, 1}));
    lanes.push_back(MakeLane(node, 100, {x, y, 10}, {0, 0, -1}, {Inf, Inf, -1}));
    lanes.push_back(MakeLane(node, 100, {x + 5, y, 0}, {0, 0, 1}, {Inf, Inf, 1}));
    lanes.push_back(MakeLane(node, 100, {x, y, 6}, {0, 0, 1}, {Inf, Inf, 1}));
  }
  // Exactly on the shared edge v2-v0 of triangles 0 and 3, and through the shared vertex v2.
  lanes.push_back(MakeLane(Pointer(0, 0), 100, {-0.5f, -0.5f, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  lanes.push_back(MakeLane(Pointer(0, 3), 100, {-0.5f, -0.5f, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  lanes.push_back(MakeLane(Pointer(0, 1), 100, {0, 0, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  // Ray parallel to the plane (degenerate determinant).
  lanes.push_back(MakeLane(Pointer(0, 0), 100, {0, -0.5f, 5}, {1, 0, 0}, {1, Inf, Inf}));
  // Out of range (T# size) and user node types 6/7.
  lanes.push_back(MakeLane(Pointer(0x100000u, 0), 100, {0, -0.5f, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  lanes.push_back(MakeLane(Pointer(0, 6), 100, {0, -0.5f, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  lanes.push_back(MakeLane(Pointer(0, 7), 100, {0, -0.5f, 0}, {0, 0, 1}, {Inf, Inf, 1}));
  return scenario;
}

// Hand-computed results check the reference before it becomes the GPU oracle.
void CheckReferenceDirected() {
  const Ref::Math m{.flush = true};
  const auto check = [](bool ok, const char *what) {
    Require("BvhReference", "directed", ok, what);
  };
  {
    const auto scenario = BoxScenario(false, Tsharp(), "box32");
    const auto memory = scenario.image.Memory();
    const auto run = [&](u32 lane, std::array<u32, 4> tsharp) {
      return Ref::Intersect(tsharp, scenario.lanes[lane].node, RayOf(scenario.lanes[lane], false),
                            memory, m);
    };
    const auto c = RowChildren;
    check(run(0, Tsharp()) == std::array<u32, 4>{c[3], c[2], c[1], c[0]},
          "box32: sorted near to far");
    check(run(1, Tsharp()) == std::array<u32, 4>{c[3], c[2], InvalidNode, InvalidNode},
          "box32: ray extent cuts the far boxes");
    check(run(2, Tsharp()) == BoxMiss, "box32: backwards ray misses");
    check(run(3, Tsharp()) == std::array<u32, 4>{c[2], c[1], c[0], InvalidNode},
          "box32: origin inside a box sorts it first");
    check(run(4, Tsharp()) == BoxMiss, "box32: ray above the boxes misses");
    check(run(5, Tsharp()) == std::array<u32, 4>{c[3], c[2], c[1], c[0]},
          "box32: a grazing ray hits (closed slabs)");
    check(run(7, Tsharp()) == std::array<u32, 4>{c[3], InvalidNode, InvalidNode, InvalidNode},
          "box32: zero extent keeps only the box around the origin");
    check(run(8, Tsharp()) == std::array<u32, 4>{c[3], c[2], c[1], c[0]},
          "box32: -0 direction (inverse -inf) behaves like +0");
    check(run(0, Tsharp(false)) == c, "box32: unsorted keeps slot order");
    check(run(1, Tsharp(false)) == std::array<u32, 4>{InvalidNode, InvalidNode, c[2], c[3]},
          "box32: unsorted misses stay in their slots");
    // The box at x = 5: an extent of exactly 5 reaches it (closed test). One ulp less (5 - 2^-21)
    // misses; a growing amount of 2 multiplies max_t by 1 + 2^-23, which rounds back to 5.
    Ref::Ray ray = RayOf(scenario.lanes[0], false);
    ray.extent = 5.0f;
    const auto grow_run = [&](u32 grow) {
      return Ref::Intersect(Tsharp(true, true, grow), scenario.lanes[0].node, ray, memory, m);
    };
    check(grow_run(0)[2] == c[1], "box32: extent 5 reaches the box starting at 5");
    ray.extent = std::nextafter(5.0f, 0.0f);
    check(grow_run(0)[2] == InvalidNode, "box32: one ulp short misses without growing");
    check(grow_run(2)[2] == c[1], "box32: growing by 2 ulps (2^-23) reaches it");
  }
  {
    const auto scenario = BoxScenario(true, Tsharp(), "box16");
    const auto memory = scenario.image.Memory();
    const auto c = RowChildren;
    check(Ref::Intersect(Tsharp(), scenario.lanes[0].node, RayOf(scenario.lanes[0], false),
                         memory, m) == std::array<u32, 4>{c[3], c[2], c[1], c[0]},
          "box16: sorted near to far");
  }
  {
    const auto scenario = TriangleScenario(Tsharp(), "triangle");
    const auto memory = scenario.image.Memory();
    const auto run = [&](const Lane &lane, std::array<u32, 4> tsharp) {
      return Ref::Intersect(tsharp, lane.node, RayOf(lane, false), memory, m);
    };
    // Type 0 = (v0, v1, v2) = ((-1,-1,5), (1,-1,5), (0,0,5)); ray (0,-0.5,0) +z.
    // e1 = (2,0,0), e2 = (1,1,0), e3 = (1,0.5,-5); s1 = dir x e2 = (-1,1,0);
    // s2 = e3 x e1 = (0,-10,-1); t_num = e2.s2 = -10; t_den = s1.e1 = -2; i_num = e3.s1 = -0.5;
    // j_num = dir.s2 = -1 -> t = 5, u = 0.25, v = 0.5. ID 9: I = i_num, J = j_num.
    check(run(scenario.lanes[0], Tsharp()) ==
              std::array<u32, 4>{B(-10.0f), B(-2.0f), B(-0.5f), B(-1.0f)},
          "triangle 0: hit from below, return mode 1");
    check(run(scenario.lanes[0], Tsharp(true, false)) ==
              std::array<u32, 4>{B(-10.0f), B(-2.0f), FanTriangleId, 1u},
          "triangle 0: return mode 0 gives the triangle ID and hit status");
    // From above: dir (0,0,-1), origin z = 10: t_num and t_den change sign: t = 5.
    const auto above = run(scenario.lanes[1], Tsharp());
    check(F(above[0]) / F(above[1]) == 5.0f && F(above[1]) > 0.0f,
          "triangle 0: hit from above has a positive denominator");
    check(run(scenario.lanes[2], Tsharp())[0] == PlusInf &&
              run(scenario.lanes[2], Tsharp())[1] == OneF,
          "triangle 0: miss beside returns +inf over 1");
    check(run(scenario.lanes[3], Tsharp())[0] == PlusInf, "triangle 0: behind the origin misses");
    // Type 1 = (v1, v3, v2), I source 2, J source 1 (swapped).
    const auto second = run(scenario.lanes[4], Tsharp());
    check(second[0] != PlusInf && F(second[0]) / F(second[1]) == 5.0f,
          "triangle 1: hit at t = 5");
    // Degenerate (parallel) ray: t_den = 0 -> quotients inf/NaN: u = -0/0 = NaN... not a miss.
    const auto parallel = run(scenario.lanes[19], Tsharp());
    check(F(parallel[1]) == 0.0f, "parallel ray: zero determinant is returned as is");
    check(run(scenario.lanes[20], Tsharp(true, true, 0, 3)) == TriangleMiss,
          "node beyond the T# size misses");
    check(Ref::Intersect(Tsharp(), scenario.lanes[21].node, RayOf(scenario.lanes[21], false),
                         memory, m, false) == BoxMiss,
          "type 6 returns invalid children without the PS5 decode");
    check(run(scenario.lanes[22], Tsharp()) == BoxMiss, "type 7 returns invalid children");
  }
  {
    // PS5 type-6 bounds: RT-HW's hand vectors (tests/RtHardwareTests.cpp on claude/rt-hardware).
    struct BoundVector {
      u32 field, exponent;
      bool is_min;
      u32 bits;
    };
    constexpr std::array<BoundVector, 16> Bounds{{
        {0x10000u, 127, true, 0x3f800000u},  // 1.0, min: exact
        {0x10000u, 127, false, 0x3f80007fu}, // 1.0, max: truncated bits filled
        {0x30000u, 127, true, 0xbf80007fu},  // -1.0, min: rounds away from zero
        {0x30000u, 127, false, 0xbf800000u}, // -1.0, max: exact
        {0x00001u, 127, true, 0x37800000u},  // 2^-16, renormalized
        {0x00001u, 127, false, 0x37ffffffu},
        {0x00000u, 127, true, 0x00000000u},  // zero magnitude
        {0x00000u, 127, false, 0x377fffffu}, // just below one grid step
        {0x20000u, 127, true, 0xb77fffffu},  // negative zero magnitude, min: rounds away
        {0x20000u, 127, false, 0x80000000u},
        {0x00001u, 10, true, 0x00020000u}, // exponent too small: denormal
        {0x00001u, 10, false, 0x0003ffffu},
        {0x00000u, 10, false, 0x007fffffu},
        {0x10000u, 255, false, 0x7f800000u}, // exponent 255: no fill
        {0x1ffffu, 130, false, 0x417fffffu}, // largest magnitude, filled
        {0x1ffffu, 130, true, 0x417fff80u},
    }};
    for (const auto &v : Bounds) {
      check(Ref::SharedExpBound(v.field, v.exponent, v.is_min) == v.bits,
            "type 6: shared-exponent bound hand vector");
    }
    // Whole nodes: RT-HW's decoder output for 64 encoder-built nodes.
    for (const auto &vector : kPsrType6Vectors) {
      std::array<u32, 16> node{};
      std::copy(std::begin(vector.node), std::end(vector.node), node.begin());
      const auto box = Ref::DecodeBoxPs5(node);
      bool same = true;
      for (u32 child = 0; child < 4u; ++child) {
        same = same && box.children[child] == vector.children[child];
        for (u32 k = 0; k < 6u; ++k) {
          const float value = (k < 3u ? box.lo : box.hi)[child][k % 3u];
          same = same && B(value) == vector.bounds[child * 6u + k];
        }
      }
      check(same, "type 6: node decode matches RT-HW's vectors");
    }
    // A hand-built node: base 0x100, children of types 5, 2 and 4, then an empty type-6 child.
    // Children 0-2 span [1, 2) on every axis at exponent 127 (1.0 = 0x10000, 0x1ffff below 2.0).
    std::array<u32, 16> node{};
    const auto put = [&](u32 first, u32 count, u32 value) {
      for (u32 i = 0; i < count; ++i) {
        const u32 bit = first + i;
        const u32 mask = 1u << (bit % 32u);
        node[bit / 32u] = ((value >> i) & 1u) != 0u ? node[bit / 32u] | mask
                                                    : node[bit / 32u] & ~mask;
      }
    };
    put(0, 29, 0x100u);
    const std::array<u32, 4> types{5u, 2u, 4u, 6u};
    for (u32 c = 0; c < 4u; ++c) put(29u + 3u * c, 3, types[c]);
    for (u32 c = 0; c < 4u; ++c) put(41u + 2u * c, 2, c == 3u ? 3u : c);
    for (u32 axis = 0; axis < 3u; ++axis) put(56u + 8u * axis, 8, 127u);
    for (u32 c = 0; c < 3u; ++c) {
      for (u32 k = 0; k < 6u; ++k) put(80u + 18u * (6u * c + k), 18, k < 3u ? 0x10000u : 0x1ffffu);
    }
    const auto box = Ref::DecodeBoxPs5(node);
    check(box.children == std::array<u32, 4>{(0x100u << 3u) | 5u, (0x102u << 3u) | 2u,
                                             (0x106u << 3u) | 4u, InvalidNode},
          "type 6: implicit child pointers (units 2, 4, 1; empty child)");
    check(box.lo[0][0] == 1.0f && B(box.hi[0][0]) == 0x3fffffffu,
          "type 6: the maximum 0x1ffff rounds up with the filled bits");
  }
  std::printf("[bvh]     %-32s ok\n", "ReferenceDirected");
}

// ----------------------------------------------------------------------------------------------
// Fuzzing

struct Rng {
  std::mt19937 engine;
  explicit Rng(u32 seed) : engine(seed) {}
  float Uniform(float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(engine);
  }
  u32 Below(u32 n) { return std::uniform_int_distribution<u32>(0, n - 1u)(engine); }
  bool Chance(u32 percent) { return Below(100) < percent; }
  // Exactly representable in binary16 (for box16 bounds and A16 rays).
  float HalfExact(float lo, float hi) { return HalfToFloat(FloatToHalf(QuantizeHalf(Uniform(lo, hi)))); }
  static float QuantizeHalf(float value) {
    // Round to 10 fraction bits by truncation, keeping the exponent in binary16 normal range.
    if (value == 0.0f) return 0.0f;
    const float magnitude = std::min(std::max(std::fabs(value), 0x1p-14f), 60000.0f);
    const u32 bits = B(magnitude) & ~0x1fffu;
    return value < 0 ? -F(bits) : F(bits);
  }
  float Special() {
    constexpr std::array<u32, 8> Values{0x7f800000u, 0xff800000u, 0x7fc00000u, 0x80000000u,
                                        0x00000000u, 0x00000001u, 0x7f7fffffu, 0x3f800000u};
    return F(Values[Below(static_cast<u32>(Values.size()))]);
  }
};

// PS5 type-6 nodes: RT-HW's vectors (Psr's encoder) in 64-byte slots. Half the lanes aim at a
// child's centre; the others start on a decoded bound, or one ulp either side of it, and leave the
// box along that axis, so they hit only if the GPU decodes that bound bit-exactly.
Scenario Type6Scenario(std::array<u32, 4> tsharp, const char *name, u32 seed) {
  Rng rng(seed);
  Scenario scenario;
  scenario.name = name;
  scenario.tsharp = tsharp;
  constexpr u32 Count = static_cast<u32>(std::size(kPsrType6Vectors));
  for (u32 slot = 0; slot < Count; ++slot) {
    for (u32 i = 0; i < 16u; ++i) {
      scenario.image.Put(slot * 64u + i * 4u, kPsrType6Vectors[slot].node[i]);
    }
  }
  for (u32 lane = 0; lane < 64u; ++lane) {
    const u32 slot = rng.Below(Count);
    const auto &vector = kPsrType6Vectors[slot];
    const u32 child = rng.Below(4);
    std::array<float, 3> lo{};
    std::array<float, 3> hi{};
    std::array<float, 3> center{};
    for (u32 axis = 0; axis < 3u; ++axis) {
      lo[axis] = F(vector.bounds[child * 6u + axis]);
      hi[axis] = F(vector.bounds[child * 6u + 3u + axis]);
      center[axis] = lo[axis] * 0.5f + hi[axis] * 0.5f;
    }
    std::array<float, 3> origin{};
    std::array<float, 3> dir{};
    std::array<float, 3> inv{};
    if ((lane % 2u) == 0u) {
      for (u32 axis = 0; axis < 3u; ++axis) {
        origin[axis] = center[axis] + rng.Uniform(-30, 30);
        dir[axis] = center[axis] - origin[axis];
        inv[axis] = 1.0f / dir[axis];
      }
    } else {
      const u32 axis = rng.Below(3);
      const bool low_side = rng.Chance(50);
      const float bound = low_side ? lo[axis] : hi[axis];
      const u32 nudge = rng.Below(3);
      origin = center;
      origin[axis] = nudge == 0u ? bound : std::nextafter(bound, nudge == 1u ? Inf : -Inf);
      dir = {0, 0, 0};
      inv = {Inf, Inf, Inf};
      dir[axis] = low_side ? -1.0f : 1.0f;
      inv[axis] = dir[axis];
    }
    scenario.lanes.push_back(
        MakeLane(Pointer(slot * 64u, 6u), rng.Chance(20) ? 0.0f : 1000.0f, origin, dir, inv));
  }
  return scenario;
}

// Random nodes of every kind in 64 slots of 128 bytes, and rays aimed at their contents.
Scenario FuzzScenario(u32 seed, bool a16) {
  Rng rng(seed);
  Scenario scenario;
  scenario.name = "BvhFuzz";
  const bool sort = rng.Chance(75);
  const bool ij = rng.Chance(75);
  const u32 grow = rng.Chance(70) ? 0u : rng.Below(256);
  const uint64_t size = rng.Chance(85) ? 0xffffffull : rng.Below(128);
  scenario.tsharp = Tsharp(sort, ij, grow, size);
  struct Target {
    uint64_t pointer;
    std::array<float, 3> point;
  };
  std::vector<Target> targets;
  for (u32 slot = 0; slot < 64u; ++slot) {
    const u32 offset = slot * 128u;
    const u32 kind = rng.Below(4);
    std::array<float, 3> point{};
    if (kind == 3u) {
      // A PS5 type-6 node of random bits, mostly with exponents that make usable boxes.
      std::array<u32, 16> node{};
      for (auto &word : node) word = static_cast<u32>(rng.engine());
      for (u32 axis = 0; axis < 3u; ++axis) {
        if (rng.Chance(90)) {
          const u32 shift = 56u + 8u * axis;
          const u32 exponent = 118u + rng.Below(16);
          node[shift / 32u] = (node[shift / 32u] & ~(0xffu << (shift % 32u))) |
                              (exponent << (shift % 32u));
        }
      }
      for (u32 i = 0; i < 16u; ++i) scenario.image.Put(offset + i * 4u, node[i]);
      const auto box = Ref::DecodeBoxPs5(node);
      const u32 child = rng.Below(4);
      for (u32 axis = 0; axis < 3u; ++axis) {
        point[axis] = box.lo[child][axis] * 0.5f + box.hi[child][axis] * 0.5f;
      }
      targets.push_back({Pointer(offset, 6u), point});
    } else if (kind == 2u) {
      std::array<std::array<float, 3>, 5> vertices{};
      for (auto &v : vertices) {
        for (auto &c : v) c = rng.Chance(3) ? rng.Special() : rng.Uniform(-10, 10);
      }
      const u32 id = static_cast<u32>(rng.engine());
      scenario.image.Triangles(offset, vertices, id);
      const u32 type = rng.Below(4);
      constexpr std::array<std::array<u32, 3>, 4> Fan{
          {{0u, 1u, 2u}, {1u, 3u, 2u}, {2u, 3u, 4u}, {2u, 4u, 0u}}};
      // A random point in the triangle (barycentric). Often it is instead within 2^-26..2^-12 of an
      // edge (u = 0, v = 0 or u + v = 1) or of u = 1, on either side, where the miss decision
      // leaves the reciprocal fast path for the exact divisions.
      float wa = rng.Uniform(0, 1), wb = rng.Uniform(0, 1);
      if (wa + wb > 1) {
        wa = 1 - wa;
        wb = 1 - wb;
      }
      if (rng.Chance(40)) {
        // ("near" is a Windows macro.)
        const float step = std::ldexp(rng.Chance(50) ? 1.0f : -1.0f,
                                      -12 - static_cast<int>(rng.Below(15)));
        switch (rng.Below(4)) {
        case 0: wa = step; break;
        case 1: wb = step; break;
        case 2: wb = 1 - wa + step; break;
        default: wa = 1 + step; wb = rng.Uniform(-0.001f, 0.001f); break;
        }
      }
      for (u32 axis = 0; axis < 3u; ++axis) {
        point[axis] = vertices[Fan[type][0]][axis] * (1 - wa - wb) +
                      vertices[Fan[type][1]][axis] * wa + vertices[Fan[type][2]][axis] * wb;
      }
      targets.push_back({Pointer(offset, type), point});
    } else {
      std::array<u32, 4> children{};
      std::array<std::array<float, 6>, 4> bounds{};
      for (u32 child = 0; child < 4u; ++child) {
        children[child] = rng.Chance(10) ? InvalidNode : static_cast<u32>(rng.engine());
        for (u32 axis = 0; axis < 3u; ++axis) {
          float lo = kind == 0u ? rng.HalfExact(-50, 50) : rng.Uniform(-50, 50);
          float extent = kind == 0u ? rng.HalfExact(0, 20) : rng.Uniform(0, 20);
          if (rng.Chance(5)) extent = -extent; // inverted box
          float hi = kind == 0u ? Rng::QuantizeHalf(lo + extent) : lo + extent;
          if (kind == 1u && rng.Chance(2)) lo = rng.Special();
          bounds[child][axis] = lo;
          bounds[child][3u + axis] = hi;
        }
      }
      if (kind == 0u) {
        scenario.image.Box16(offset, children, bounds);
        // Raw binary16 specials: infinities, NaN, -0 and denormals.
        constexpr std::array<u32, 6> Halves{0x7c00u, 0xfc00u, 0x7e00u, 0x8000u, 0x0001u, 0x83ffu};
        for (u32 k = 0; k < 24u; ++k) {
          if (!rng.Chance(2)) continue;
          const u32 at = offset + 16u + (k / 2u) * 4u;
          const u32 shift = (k % 2u) * 16u;
          const u32 half = Halves[rng.Below(static_cast<u32>(Halves.size()))];
          scenario.image.Put(at, (scenario.image.Get(at) & ~(0xffffu << shift)) | (half << shift));
        }
      } else {
        scenario.image.Box32(offset, children, bounds);
      }
      const u32 child = rng.Below(4);
      for (u32 axis = 0; axis < 3u; ++axis) {
        point[axis] = (bounds[child][axis] + bounds[child][3u + axis]) * 0.5f;
      }
      targets.push_back({Pointer(offset, kind == 0u ? 4u : 5u), point});
    }
  }
  for (u32 lane = 0; lane < 64u; ++lane) {
    const auto &target = targets[rng.Below(static_cast<u32>(targets.size()))];
    uint64_t node = target.pointer;
    if (rng.Chance(3)) node = (node & ~uint64_t{7}) | rng.Below(8);    // wrong/user type
    if (rng.Chance(2)) node |= uint64_t{rng.Below(4)} << 32u;          // high bits (BVH64)
    std::array<float, 3> origin{};
    std::array<float, 3> dir{};
    std::array<float, 3> inv{};
    for (u32 axis = 0; axis < 3u; ++axis) {
      origin[axis] = a16 ? rng.Uniform(-30, 30) : rng.Uniform(-30, 30);
    }
    for (u32 axis = 0; axis < 3u; ++axis) {
      dir[axis] = std::isfinite(target.point[axis]) ? target.point[axis] - origin[axis]
                                                    : rng.Uniform(-1, 1);
      if (rng.Chance(5)) dir[axis] = 0.0f;
      if (rng.Chance(2)) dir[axis] = -0.0f;
    }
    for (u32 axis = 0; axis < 3u; ++axis) {
      if (a16) dir[axis] = Rng::QuantizeHalf(dir[axis]);
      inv[axis] = 1.0f / dir[axis];
      if (rng.Chance(3)) inv[axis] = rng.Special();
      if (a16) inv[axis] = std::isfinite(inv[axis]) ? Rng::QuantizeHalf(inv[axis]) : inv[axis];
      if (a16 && std::isnan(inv[axis])) inv[axis] = Inf;
    }
    float extent = rng.Chance(20) ? Inf : rng.Uniform(0, 200);
    if (rng.Chance(3)) extent = rng.Special();
    scenario.lanes.push_back(MakeLane(node, extent, origin, dir, inv));
  }
  return scenario;
}

// ----------------------------------------------------------------------------------------------
// Runners

class ScopedFloatControls {
public:
  explicit ScopedFloatControls(const ShaderRecompiler::Spirv::HostFloatControls &controls)
      : m_saved(ShaderRecompiler::Spirv::GetHostFloatControls()) {
    ShaderRecompiler::Spirv::SetHostFloatControls(controls);
  }
  ~ScopedFloatControls() { ShaderRecompiler::Spirv::SetHostFloatControls(m_saved); }
  ScopedFloatControls(const ScopedFloatControls &) = delete;
  ScopedFloatControls &operator=(const ScopedFloatControls &) = delete;

private:
  ShaderRecompiler::Spirv::HostFloatControls m_saved;
};

// KYTY_RT_STUB: every lane misses. Lane l tests node pointer l (type l & 7).
TestCase StubReturnsMiss(u32 wave_size, bool bvh64, bool a16, bool nsa) {
  using O = ShaderOpcode;
  TestCase test;
  test.name = "BvhStubReturnsMiss";
  std::vector<u32> &code = test.code;
  const u32 dwords = BvhAddressDwords(bvh64, a16);
  constexpr u32 AddrBase = 40;
  constexpr u32 NsaBase = 60;
  constexpr u32 Result = 16;
  for (u32 i = 0; i < 4u; ++i) AppendSMovLiteral(&code, 32u + i, Tsharp()[i]);
  std::vector<u32> nsa_regs;
  for (u32 i = 0; i < dwords; ++i) {
    const u32 reg = i == 0 || !nsa ? AddrBase + i : NsaBase + (dwords - 1u - i);
    if (i == 0) {
      code.push_back(EncodeVop1(0x01, reg, Vgpr(0)));
    } else {
      AppendVMovLiteral(&code, reg, 0x3f800000u + i);
      if (nsa) nsa_regs.push_back(reg);
    }
  }
  AppendBvh(&code, bvh64, a16, Result, AddrBase, 32, nsa_regs);
  for (u32 dword = 0; dword < 4u; ++dword) {
    AppendStoreVgprAtLaneDwordOffset(&code, Result + dword, 0, dword * wave_size);
  }
  AppendEnd(&code);
  for (u32 dword = 0; dword < 4u; ++dword) {
    for (u32 lane = 0; lane < wave_size; ++lane) {
      test.expected.push_back((lane & 7u) < 4u ? TriangleMiss[dword] : InvalidNode);
    }
  }
  test.initial.resize(test.expected.size());
  test.opcodes = {O::S_MOV_B32, O::V_MOV_B32,
                  bvh64 ? O::IMAGE_BVH64_INTERSECT_RAY : O::IMAGE_BVH_INTERSECT_RAY,
                  O::V_LSHLREV_B32, O::V_ADD_NC_U32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.compute_info.threads_num[0] = wave_size;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = wave_size;
  test.has_compute_info = true;
  return test;
}

void RunStub(VulkanHarness *vulkan) {
  auto options = GetCodegenOptions();
  options.rt_stub = true;
  options.rt_software = false;
  const ScopedCodegenOptions scoped(options);
  for (const u32 wave_size : {32u, 64u}) {
    for (const bool bvh64 : {false, true}) {
      for (const bool a16 : {false, true}) {
        for (const bool nsa : {false, true}) {
          const auto test = StubReturnsMiss(wave_size, bvh64, a16, nsa);
          const auto compiled = CompileCase(test, vulkan->SubgroupSize());
          Require(test.name, "program info", compiled.program.info.uses_bvh,
                  "stubbed BVH program was not marked as using BVH instructions");
          RunCase(vulkan, test);
        }
      }
    }
  }
}

namespace Recompiler = ::Libs::Graphics::ShaderRecompiler;

// GDS size of the node-count tests: room for the budget word and the whole histogram.
constexpr u32 GdsDwords = 32;
static_assert(GdsDwords >= Recompiler::RtNodeStatsGdsFromEnd + Recompiler::RtNodeStatsBins);

// Runs a MakeCase test and reports mismatching lanes with their inputs.
void RunChecked(VulkanHarness *vulkan, const TestCase &test, const Scenario &scenario, u32 wave_size,
                bool bvh64, bool a16) {
  using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
  const auto compiled = CompileCase(test, vulkan->SubgroupSize());
  Require(test.name, "program info", compiled.program.info.uses_bvh,
          "BVH program was not marked as using BVH instructions");
  if (std::getenv("KYTY_BVH_TEST_DUMP") != nullptr) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string text;
    tools.Disassemble(compiled.spirv, &text);
    std::fprintf(stderr, "%s IR:\n%s\nSPIR-V:\n%s\n", test.name,
                 ShaderRecompiler::IR::ProgramToString(compiled.program).c_str(), text.c_str());
  }
  const bool bda =
      ShaderRecompiler::IR::FindBinding(compiled.program.bindings, Kind::BdaPagetable) != nullptr;
  auto buffer = vulkan->CreateStorageBuffer(test.name, test.initial, test.initial.size(), bda);
  // The BVH node count (KYTY_RT_NODE_BUDGET, on by default) reports through GDS.
  const bool gds =
      ShaderRecompiler::IR::FindBinding(compiled.program.bindings, Kind::Gds) != nullptr;
  auto gds_buffer = vulkan->CreateStorageBuffer(test.name, std::vector<u32>(GdsDwords, 0u),
                                                GdsDwords);
  vulkan->Dispatch(test, compiled, buffer, gds ? &gds_buffer : nullptr);
  const auto actual = vulkan->ReadBuffer(test.name, buffer, test.expected.size());
  const auto gds_words = vulkan->ReadBuffer(test.name, gds_buffer, GdsDwords);
  vulkan->DestroyBuffer(&gds_buffer);
  vulkan->DestroyBuffer(&buffer);
  // (EXEC = 0 folds the instruction away, and with it the count.)
  Require(test.name, "node budget", gds == ShaderRecompiler::IR::UsesBvhNodeCount(compiled.program),
          "the BVH node count's GDS binding does not follow KYTY_RT_NODE_BUDGET");
  Require(test.name, "node budget", gds_words[GdsDwords - Recompiler::RtNodeBudgetGdsFromEnd] == 0u,
          "an ordinary node test exhausted the default BVH node budget");
  // Float results match any NaN with any NaN: the encoding of a NaN made by an invalid operation
  // is the host's (x86 0xffc00000, NVIDIA 0x7fffffff), as in every other translated float op.
  const auto is_nan = [](u32 bits) { return (bits & 0x7fffffffu) > PlusInf; };
  const bool ij = (scenario.tsharp[3] & (1u << 24u)) != 0u;
  u32 bad = 0;
  for (u32 lane = 0; lane < wave_size; ++lane) {
    const Lane &input = scenario.lanes[lane % scenario.lanes.size()];
    const bool triangle = (input.node & 7u) < 4u;
    bool same = true;
    for (u32 dword = 0; dword < 4u; ++dword) {
      const u32 got = actual[dword * wave_size + lane];
      const u32 want = test.expected[dword * wave_size + lane];
      const bool is_float = triangle && (dword < 2u || ij);
      same = same && (got == want || (is_float && is_nan(got) && is_nan(want)));
    }
    if (same) continue;
    if (++bad > 8u) continue;
    std::fprintf(stderr,
                 "%s: wave%u bvh64=%d a16=%d lane %u node=0x%llx extent=%08x origin=%08x %08x "
                 "%08x dir=%08x %08x %08x inv=%08x %08x %08x T#=%08x %08x %08x %08x\n"
                 "  expected %08x %08x %08x %08x actual %08x %08x %08x %08x\n",
                 test.name, wave_size, bvh64, a16, lane,
                 static_cast<unsigned long long>(input.node), input.extent, input.origin[0],
                 input.origin[1], input.origin[2], input.dir[0], input.dir[1], input.dir[2],
                 input.inv[0], input.inv[1], input.inv[2], scenario.tsharp[0], scenario.tsharp[1],
                 scenario.tsharp[2], scenario.tsharp[3], test.expected[lane],
                 test.expected[wave_size + lane], test.expected[2 * wave_size + lane],
                 test.expected[3 * wave_size + lane], actual[lane], actual[wave_size + lane],
                 actual[2 * wave_size + lane], actual[3 * wave_size + lane]);
    // The node's dwords (box32: 32, others: 16) for offline analysis.
    const uint64_t node = bvh64 ? input.node : (input.node & 0xffffffffull);
    const uint64_t offset = (node & ~uint64_t{7}) << 3u;
    if (offset < scenario.image.nodes.size() * 4u) {
      std::string words;
      for (u32 i = 0; i < ((node & 7u) == 5u ? 32u : 16u); ++i) {
        char text[12];
        std::snprintf(text, sizeof(text), " %08x",
                      scenario.image.Get(static_cast<u32>(offset) + i * 4u));
        words += text;
      }
      std::fprintf(stderr, "  node dwords:%s\n", words.c_str());
    }
  }
  Require(test.name, "readback", bad == 0u,
          std::to_string(bad) + " lane(s) differ from the reference");
  std::printf("[bvh]     %-24s wave%-2u bvh64=%d a16=%d ok\n", test.name, wave_size, bvh64, a16);
}

// Software path against the reference: every scenario in both wave sizes and every encoding.
void RunSoftware(VulkanHarness *vulkan, u32 fuzz_iterations, u32 first_iteration = 0) {
  auto options = GetCodegenOptions();
  options.rt_software = true;
  const ScopedCodegenOptions scoped(options);
  // The emulator's float mode on this device. The guest flushes f32 denormals (FLOAT_MODE 0xC0),
  // but a device without shaderDenormFlushToZeroFloat32 (the RTX 3090 on driver 610.74) keeps
  // its default, IEEE denormals, and the reference follows the device.
  const auto controls = vulkan->DeviceFloatControls();
  const ScopedFloatControls float_controls(controls);
  const bool flush = controls.denorm_flush_f32;
  std::printf("[bvh]     f32 denormals: %s\n", flush ? "flushed" : "kept (no device FTZ)");
  std::vector<Scenario> scenarios;
  scenarios.push_back(BoxScenario(false, Tsharp(), "BvhBox32Sorted"));
  scenarios.push_back(BoxScenario(false, Tsharp(false, true, 7), "BvhBox32UnsortedGrow"));
  scenarios.push_back(BoxScenario(true, Tsharp(), "BvhBox16Sorted"));
  scenarios.push_back(TriangleScenario(Tsharp(), "BvhTriangleIJ"));
  scenarios.push_back(TriangleScenario(Tsharp(true, false), "BvhTriangleIdHit"));
  scenarios.push_back(TriangleScenario(Tsharp(true, true, 0, 3), "BvhTriangleSmallSize"));
  // A box32 node whose second half is on the next BDA page, mapped elsewhere.
  scenarios.push_back(BoxScenario(false, Tsharp(), "BvhBox32PageSplit", SplitGuest - NodeOffset - 64u));
  scenarios.back().split_pages = true;
  for (const auto &scenario : scenarios) {
    for (const u32 wave_size : {32u, 64u}) {
      for (const bool bvh64 : {false, true}) {
        for (const bool nsa : {false, true}) {
          RunChecked(vulkan, MakeCase(scenario, wave_size, bvh64, false, nsa, flush), scenario,
                     wave_size, bvh64, false);
        }
      }
      RunChecked(vulkan, MakeCase(scenario, wave_size, false, true, false, flush), scenario,
                 wave_size, false, true);
    }
  }
  // PS5 type-6 boxes (not A16: the rays are not binary16 values), and KYTY_RT_TYPE6=0.
  for (const auto &tsharp : {Tsharp(), Tsharp(false, true, 200)}) {
    const auto scenario = Type6Scenario(tsharp, "BvhType6", 0x7e600000u + tsharp[1]);
    // The probes must discriminate: both hits and misses among them.
    u32 hits = 0;
    for (const auto &lane : scenario.lanes) {
      const auto result = Ref::Intersect(tsharp, lane.node, RayOf(lane, false),
                                         scenario.image.Memory(), Ref::Math{.flush = flush});
      hits += std::any_of(result.begin(), result.end(), [](u32 p) { return p != InvalidNode; })
                  ? 1u
                  : 0u;
    }
    Require("BvhType6", "scenario", hits >= 16u && hits <= 60u,
            std::to_string(hits) + " of 64 lanes hit a child");
    std::printf("[bvh]     BvhType6 T#[1]=%08x: %u of 64 lanes hit\n", tsharp[1], hits);
    for (const u32 wave_size : {32u, 64u}) {
      for (const bool bvh64 : {false, true}) {
        RunChecked(vulkan, MakeCase(scenario, wave_size, bvh64, false, bvh64, flush), scenario,
                   wave_size, bvh64, false);
      }
    }
  }
  {
    auto rdna2 = GetCodegenOptions();
    rdna2.rt_type6 = false;
    const ScopedCodegenOptions scoped_rdna2(rdna2);
    auto scenario = Type6Scenario(Tsharp(), "BvhType6Off", 0x7e610000u);
    scenario.ps5_type6 = false;
    RunChecked(vulkan, MakeCase(scenario, 32, false, false, false, flush), scenario, 32, false,
               false);
  }
  // Inactive lanes keep their destination registers (and read no memory).
  for (const uint64_t exec : {0xc3a5f00f5a3c0ff0ull, 0x0ull}) {
    for (const u32 wave_size : {32u, 64u}) {
      for (const u32 index : {0u, 3u}) {
        RunChecked(vulkan, MakeCase(scenarios[index], wave_size, false, false, true, flush, exec),
                   scenarios[index], wave_size, false, false);
      }
    }
  }
  for (u32 iteration = first_iteration; iteration < first_iteration + fuzz_iterations;
       ++iteration) {
    const bool a16 = (iteration % 4u) == 3u;
    const bool bvh64 = (iteration % 3u) == 1u;
    const u32 wave_size = (iteration % 2u) == 0u ? 64u : 32u;
    const auto scenario = FuzzScenario(0x5eed0000u + iteration, a16);
    RunChecked(vulkan, MakeCase(scenario, wave_size, bvh64, a16, (iteration % 5u) == 2u, flush),
               scenario, wave_size, bvh64, a16);
  }
}

// ----------------------------------------------------------------------------------------------
// KYTY_RT_NODE_BUDGET / KYTY_RT_NODE_STATS

// The scenario must outlive the case (its name). A traversal that follows child 0 until a node
// test returns no hit child, counting its node tests in s2; then a scalar loop counts s3 up to 0x10000 (the loop an exhausted invocation must
// leave at once). Stores s2, s3 and the last result's dword 0 at [lane], [wave + lane] and
// [2 * wave + lane].
TestCase MakeWalkCase(const Scenario &scenario, u32 wave_size, u32 expected_tests,
                      u32 expected_s3) {
  using O = ShaderOpcode;
  auto test = MakeCase(scenario, wave_size, false, false, false, true);
  std::vector<u32> code;
  AppendRecordAddress(&code);
  const u32 dwords = BvhAddressDwords(false, false);
  for (u32 k = 0; k < dwords; ++k) {
    code.push_back(EncodeMubuf0(0x0cu, (k == 0u ? 0u : k + 1u) * 4u));
    code.push_back(EncodeMubuf1(40u + k, 0, 1));
  }
  for (u32 i = 0; i < 4u; ++i) AppendSMovLiteral(&code, 32u + i, scenario.tsharp[i]);
  code.push_back(EncodeSMovB32(2, InlineU32(0)));
  const auto walk = static_cast<u32>(code.size());
  AppendBvh(&code, false, false, 16, 40, 32);
  code.push_back(EncodeSop2(0x00, 2, 2, InlineU32(1)));   // s_add_u32 s2, s2, 1
  code.push_back(EncodeVopc(0xc2, 193u, 16));             // v_cmp_eq_u32 vcc, -1, v16
  code.push_back(EncodeSopp(0x07, 2));                    // s_cbranch_vccnz done
  code.push_back(EncodeVop1(0x01, 40, Vgpr(16)));         // v_mov_b32 v40, v16
  const auto back = static_cast<int32_t>(code.size());
  code.push_back(EncodeSopp(0x02, static_cast<u32>(static_cast<int32_t>(walk) - (back + 1)) &
                                      0xffffu));        // s_branch walk
  code.push_back(EncodeSMovB32(3, InlineU32(0)));
  const auto count = static_cast<u32>(code.size());
  code.push_back(EncodeSop2(0x00, 3, 3, InlineU32(1)));   // s_add_u32 s3, s3, 1
  code.push_back(EncodeSopc(0x0a, 3, 255u));              // s_cmp_lt_u32 s3, 0x10000
  code.push_back(0x10000u);
  const auto again = static_cast<int32_t>(code.size());
  code.push_back(EncodeSopp(0x05, static_cast<u32>(static_cast<int32_t>(count) - (again + 1)) &
                                      0xffffu));        // s_cbranch_scc1 count
  code.push_back(EncodeVop1(0x01, 20, 2));                // v_mov_b32 v20, s2
  code.push_back(EncodeVop1(0x01, 21, 3));                // v_mov_b32 v21, s3
  AppendStoreVgprAtLaneDwordOffset(&code, 20, 0, 0);
  AppendStoreVgprAtLaneDwordOffset(&code, 21, 0, wave_size);
  AppendStoreVgprAtLaneDwordOffset(&code, 16, 0, 2u * wave_size);
  AppendEnd(&code);
  test.code = std::move(code);
  test.expected.assign(3u * wave_size, 0u);
  for (u32 lane = 0; lane < wave_size; ++lane) {
    test.expected[lane] = expected_tests;
    test.expected[wave_size + lane] = expected_s3;
    test.expected[2u * wave_size + lane] = InvalidNode;
  }
  test.opcodes = {O::V_LSHLREV_B32, O::BUFFER_LOAD_DWORD, O::S_MOV_B32,
                  O::IMAGE_BVH_INTERSECT_RAY, O::S_ADD_U32, O::V_CMP_EQ_U32,
                  O::S_CBRANCH_VCCNZ, O::V_MOV_B32, O::S_BRANCH, O::S_CMP_LT_U32,
                  O::S_CBRANCH_SCC1, O::V_ADD_NC_U32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  return test;
}

// `links` box32 nodes, 128 bytes apart, each with child 0 = the next node and no other child; the
// last one points back to the first when `cycle`, else it has no child. One ray through them.
Scenario WalkScenario(u32 links, bool cycle, const char *name) {
  Scenario scenario;
  scenario.name = name;
  scenario.tsharp = Tsharp();
  constexpr std::array<float, 6> Everywhere{-1000, -1000, -1000, 1000, 1000, 1000};
  for (u32 link = 0; link < links; ++link) {
    const bool last = link + 1u == links;
    const u32 next = last ? (cycle ? static_cast<u32>(Pointer(0, 5)) : InvalidNode)
                          : static_cast<u32>(Pointer((link + 1u) * 128u, 5));
    scenario.image.Box32(link * 128u, {next, InvalidNode, InvalidNode, InvalidNode},
                         {Everywhere, Everywhere, Everywhere, Everywhere});
  }
  scenario.lanes.push_back(MakeLane(Pointer(0, 5), Inf, {0, 0, 0}, {1, 0, 0}, {1, Inf, Inf}));
  return scenario;
}

std::vector<u32> RunWalk(VulkanHarness *vulkan, const TestCase &test) {
  const auto compiled = CompileCase(test, vulkan->SubgroupSize());
  Require(test.name, "GDS", ShaderRecompiler::IR::FindBinding(
                                compiled.program.bindings,
                                ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr,
          "a counted BVH program does not bind GDS");
  auto buffer = vulkan->CreateStorageBuffer(test.name, test.initial, test.initial.size(), true);
  auto gds = vulkan->CreateStorageBuffer(test.name, std::vector<u32>(GdsDwords, 0u), GdsDwords);
  vulkan->Dispatch(test, compiled, buffer, &gds);
  const auto actual = vulkan->ReadBuffer(test.name, buffer, test.expected.size());
  auto gds_words = vulkan->ReadBuffer(test.name, gds, GdsDwords);
  vulkan->DestroyBuffer(&gds);
  vulkan->DestroyBuffer(&buffer);
  Require(test.name, "result", actual == test.expected,
          "walk results differ (lane 0: tests " + std::to_string(actual[0]) + ", second loop " +
              std::to_string(actual[test.expected.size() / 3u]) + ")");
  return gds_words;
}

// CPU only: the count and its GDS binding exist exactly when the budget or the statistics are on.
void CheckNodeCountCodegen() {
  constexpr const char *name = "BvhNodeCountCodegen";
  const auto scenario = WalkScenario(1, false, name);
  const auto test = MakeCase(scenario, 32, false, false, false, true);
  for (const u32 mode : {0u, 1u, 2u}) {
    auto options = GetCodegenOptions();
    options.rt_software = true;
    options.rt_node_budget = mode == 1u ? 64u : 0u;
    options.rt_node_stats = mode == 2u;
    const ScopedCodegenOptions scoped(options);
    const auto compiled = CompileCase(test, 32u);
    const auto names = CodegenTests::CountSpirv(compiled.spirv, "OpName %bvh_node_count");
    const bool gds = ShaderRecompiler::IR::FindBinding(
                         compiled.program.bindings,
                         ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr;
    Require(name, "SPIR-V", (names == 1u) == (mode != 0u) && gds == (mode != 0u),
            "the BVH node count or its GDS binding does not follow the switches");
  }
  std::printf("[bvh]     %-32s ok\n", name);
}

// GPU: a cyclic BVH ends at the budget, the invocation leaves its next loop at once, and each
// invocation reports once; with the statistics, a 5-node walk lands in bin [4, 8).
void RunNodeBudget(VulkanHarness *vulkan) {
  CheckNodeCountCodegen();
  const auto controls = vulkan->DeviceFloatControls();
  const ScopedFloatControls float_controls(controls);
  for (const u32 wave_size : {32u, 64u}) {
    const u32 lanes = wave_size == 64u && vulkan->SubgroupSize() == 32u ? 2u : 1u;
    const u32 invocations = wave_size / lanes;
    {
      auto options = GetCodegenOptions();
      options.rt_software = true;
      options.rt_node_budget = 100;
      options.rt_node_stats = false;
      const ScopedCodegenOptions scoped(options);
      // The 101st node test misses; the walk stops there and the counting loop is left at once.
      const auto scenario = WalkScenario(1, true, "BvhBudgetCycle");
      const auto test = MakeWalkCase(scenario, wave_size, 101, 1);
      const auto gds = RunWalk(vulkan, test);
      for (u32 word = 0; word < GdsDwords; ++word) {
        const u32 want = word == GdsDwords - Recompiler::RtNodeBudgetGdsFromEnd ? invocations : 0u;
        Require(test.name, "report", gds[word] == want,
                "GDS dword " + std::to_string(word) + " is " + std::to_string(gds[word]) +
                    ", expected " + std::to_string(want));
      }
      std::printf("[bvh]     %-24s wave%-2u ok (%u reports)\n", test.name, wave_size,
                  gds[GdsDwords - Recompiler::RtNodeBudgetGdsFromEnd]);
    }
    {
      auto options = GetCodegenOptions();
      options.rt_software = true;
      options.rt_node_budget = 0;
      options.rt_node_stats = true;
      const ScopedCodegenOptions scoped(options);
      const auto scenario = WalkScenario(5, false, "BvhNodeStats");
      const auto test = MakeWalkCase(scenario, wave_size, 5, 0x10000);
      const auto gds = RunWalk(vulkan, test);
      for (u32 word = 0; word < GdsDwords; ++word) {
        const u32 want = word == GdsDwords - Recompiler::RtNodeStatsGdsFromEnd - 2u ? invocations : 0u;
        Require(test.name, "histogram", gds[word] == want,
                "GDS dword " + std::to_string(word) + " is " + std::to_string(gds[word]) +
                    ", expected " + std::to_string(want));
      }
      std::printf("[bvh]     %-24s wave%-2u ok\n", test.name, wave_size);
    }
  }
}

// ----------------------------------------------------------------------------------------------
// The traversal-loop shape (see TestTraversalLoopBreakRegion in shader_cfg_tests) runs correctly:
// the outer loop counts s0 to 4 and then leaves through the break region, whose loop counts s1
// to 3. Every lane stores s0 and s1.
TestCase TraversalLoopBreakRegion(u32 wave_size) {
  using O = ShaderOpcode;
  TestCase test;
  test.name = wave_size == 64u ? "TraversalLoopBreakRegionWave64" : "TraversalLoopBreakRegionWave32";
  test.code = {
      EncodeSMovB32(0, InlineU32(0)),           // s_mov_b32 s0, 0
      EncodeSopc(0x0a, 0, InlineU32(4)),        // outer: s_cmp_lt_u32 s0, 4
      EncodeSopp(0x05, 5),                      // s_cbranch_scc1 body
      EncodeSMovB32(1, InlineU32(0)),           // s_mov_b32 s1, 0
      EncodeSop2(0x00, 1, 1, InlineU32(1)),     // inner: s_add_u32 s1, s1, 1
      EncodeSopc(0x0a, 1, InlineU32(3)),        // s_cmp_lt_u32 s1, 3
      EncodeSopp(0x05, 0xfffdu),                // s_cbranch_scc1 inner
      EncodeSopp(0x02, 4),                      // s_branch done
      EncodeSop2(0x00, 0, 0, InlineU32(1)),     // body: s_add_u32 s0, s0, 1
      EncodeSopc(0x06, 0, InlineU32(50)),       // s_cmp_eq_u32 s0, 50
      EncodeSopp(0x05, 1),                      // s_cbranch_scc1 done
      EncodeSopp(0x02, 0xfff5u),                // s_branch outer
      EncodeVop1(0x01, 1, 0),                   // done: v_mov_b32 v1, s0
      EncodeVop1(0x01, 2, 1),                   // v_mov_b32 v2, s1
  };
  AppendStoreVgprAtLaneDwordOffset(&test.code, 1, 0, 0);
  AppendStoreVgprAtLaneDwordOffset(&test.code, 2, 0, wave_size);
  AppendEnd(&test.code);
  test.initial.assign(2u * wave_size, 0u);
  test.expected.assign(2u * wave_size, 0u);
  for (u32 lane = 0; lane < wave_size; ++lane) {
    test.expected[lane] = 4u;
    test.expected[wave_size + lane] = 3u;
  }
  test.opcodes = {O::S_MOV_B32, O::S_CMP_LT_U32, O::S_CBRANCH_SCC1, O::S_ADD_U32,
                  O::S_BRANCH,  O::S_CMP_EQ_U32, O::V_MOV_B32,      O::V_LSHLREV_B32,
                  O::V_ADD_NC_U32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.compute_info.threads_num[0] = wave_size;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = wave_size;
  test.has_compute_info = true;
  return test;
}

void RunAll(VulkanHarness *vulkan) {
  CheckReferenceDirected();
  RunStub(vulkan);
  RunSoftware(vulkan, 64);
  RunNodeBudget(vulkan);
  for (const u32 wave_size : {32u, 64u}) {
    RunCase(vulkan, TraversalLoopBreakRegion(wave_size));
  }
}

// ----------------------------------------------------------------------------------------------
// Benchmark (--bvh-bench): each lane repeats a node test `loop` times on the 64 node slots in
// turn, XOR-accumulating the four result dwords. The GPU time over groups * 64 lanes * loop
// tests gives microseconds per million node tests; the same loop without the instruction is the
// baseline that is subtracted.

enum class BenchMode { Baseline, Stub, Software };

// 64 slots of 128 bytes of one node kind (4 = box16, 5 = box32, 6 = PS5 box, 0 = triangle), and
// 64 rays aimed at them.
Scenario BenchScenario(u32 kind, u32 seed) {
  Rng rng(seed);
  Scenario scenario;
  scenario.name = "BvhBench";
  scenario.tsharp = Tsharp();
  for (u32 slot = 0; slot < 64u; ++slot) {
    const u32 offset = slot * 128u;
    if (kind == 6u) {
      const auto &vector = kPsrType6Vectors[slot % std::size(kPsrType6Vectors)];
      for (u32 i = 0; i < 16u; ++i) scenario.image.Put(offset + i * 4u, vector.node[i]);
    } else if (kind == 0u) {
      std::array<std::array<float, 3>, 5> vertices{};
      for (auto &v : vertices) {
        for (auto &c : v) c = rng.Uniform(-10, 10);
      }
      scenario.image.Triangles(offset, vertices, 9u);
    } else {
      std::array<u32, 4> children{};
      std::array<std::array<float, 6>, 4> bounds{};
      for (u32 child = 0; child < 4u; ++child) {
        children[child] = (slot * 4u + child) << 4u;
        for (u32 axis = 0; axis < 3u; ++axis) {
          const float lo = rng.HalfExact(-20, 20);
          bounds[child][axis] = lo;
          bounds[child][3u + axis] = Rng::QuantizeHalf(lo + rng.HalfExact(1, 10));
        }
      }
      if (kind == 4u) {
        scenario.image.Box16(offset, children, bounds);
      } else {
        scenario.image.Box32(offset, children, bounds);
      }
    }
  }
  for (u32 lane = 0; lane < 64u; ++lane) {
    std::array<float, 3> origin{};
    std::array<float, 3> dir{};
    std::array<float, 3> inv{};
    for (u32 axis = 0; axis < 3u; ++axis) {
      origin[axis] = rng.Uniform(-30, -20);
      dir[axis] = rng.Uniform(0.2f, 1.0f);
      inv[axis] = 1.0f / dir[axis];
    }
    scenario.lanes.push_back(MakeLane(Pointer(lane % 64u * 128u, kind), 1000, origin, dir, inv));
  }
  return scenario;
}

TestCase BenchCase(const Scenario &scenario, BenchMode mode, u32 loop, u32 groups, bool flush) {
  using O = ShaderOpcode;
  auto test = MakeCase(scenario, 64, false, false, false, flush);
  test.name = "BvhBench";
  std::vector<u32> code;
  AppendRecordAddress(&code);
  const u32 dwords = BvhAddressDwords(false, false);
  for (u32 k = 0; k < dwords; ++k) {
    code.push_back(EncodeMubuf0(0x0cu, (k == 0u ? 0u : k + 1u) * 4u));
    code.push_back(EncodeMubuf1(40u + k, 0, 1));
  }
  for (u32 i = 0; i < 4u; ++i) AppendSMovLiteral(&code, 32u + i, scenario.tsharp[i]);
  AppendSMovLiteral(&code, 60, loop);
  AppendVMovU32(&code, 20, 0);
  const auto loop_start = static_cast<u32>(code.size());
  if (mode == BenchMode::Baseline) {
    // Same dependency shape without the instruction: four XORs of the node pointer.
    for (u32 k = 0; k < 4u; ++k) code.push_back(EncodeVop2(0x1d, 16u + k, Vgpr(40u), 40u + k));
  } else {
    AppendBvh(&code, false, false, 16, 40, 32);
  }
  for (u32 k = 0; k < 4u; ++k) code.push_back(EncodeVop2(0x1d, 20, Vgpr(16u + k), 20));
  // Next slot: pointer += 16 (128 bytes), wrapping at 64 slots; the type bits stay.
  code.push_back(EncodeVop2(0x25, 40, InlineU32(16), 40)); // v_add_nc_u32
  code.push_back(EncodeVop2(0x1b, 40, 255u, 40));          // v_and_b32 v40, 0x3ff, v40
  code.push_back(0x3ffu | 7u);
  code.push_back(EncodeSop2(0x01, 60, 60, InlineU32(1)));  // s_sub_u32 s60, s60, 1
  code.push_back(EncodeSopc(0x07, 60, InlineU32(0)));      // s_cmp_lg_u32 s60, 0
  const auto branch = static_cast<int32_t>(code.size());
  code.push_back(EncodeSopp(0x05, static_cast<u32>(static_cast<int32_t>(loop_start) - (branch + 1)) &
                                      0xffffu));        // s_cbranch_scc1 loop
  AppendStoreVgprAtLaneDwordOffset(&code, 20, 0, 0);
  AppendEnd(&code);
  test.code = std::move(code);
  test.expected.clear();
  test.dispatch_x = groups;
  test.opcodes = {O::V_LSHLREV_B32, O::BUFFER_LOAD_DWORD, O::S_MOV_B32, O::V_XOR_B32,
                  O::V_ADD_NC_U32, O::V_AND_B32, O::S_SUB_U32, O::S_CMP_LG_U32,
                  O::S_CBRANCH_SCC1, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  return test;
}

// GPU nanoseconds for one dispatch of the case (median of `samples`, each averaging `repeats`
// back-to-back dispatches).
double TimeCase(VulkanHarness *vulkan, const TestCase &test, u32 repeats, u32 samples) {
  using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
  const auto compiled = CompileCase(test, vulkan->SubgroupSize());
  const bool bda =
      ShaderRecompiler::IR::FindBinding(compiled.program.bindings, Kind::BdaPagetable) != nullptr;
  auto buffer = vulkan->CreateStorageBuffer(test.name, test.initial, test.initial.size(), bda);
  const bool gds =
      ShaderRecompiler::IR::FindBinding(compiled.program.bindings, Kind::Gds) != nullptr;
  auto gds_buffer = vulkan->CreateStorageBuffer(test.name, std::vector<u32>(GdsDwords, 0u),
                                                GdsDwords);
  std::vector<double> times;
  for (u32 sample = 0; sample < samples; ++sample) {
    double elapsed_us = 0;
    vulkan->Dispatch(test, compiled, buffer, gds ? &gds_buffer : nullptr, nullptr, nullptr,
                     nullptr, nullptr, {}, repeats, &elapsed_us);
    times.push_back(elapsed_us * 1000.0);
  }
  vulkan->DestroyBuffer(&gds_buffer);
  vulkan->DestroyBuffer(&buffer);
  std::sort(times.begin(), times.end());
  return times[times.size() / 2u];
}

void RunBenchmark(VulkanHarness *vulkan) {
  constexpr u32 Loop = 256;
  constexpr u32 Groups = 2048;
  constexpr double Tests = double(Loop) * Groups * 64.0;
  const auto controls = vulkan->DeviceFloatControls();
  const ScopedFloatControls float_controls(controls);
  for (const u32 kind : {5u, 4u, 6u, 0u}) {
    const auto scenario = BenchScenario(kind, 0xbe7c0000u + kind);
    double ns[3] = {};
    for (const auto mode : {BenchMode::Baseline, BenchMode::Stub, BenchMode::Software}) {
      auto options = GetCodegenOptions();
      options.rt_stub = mode == BenchMode::Stub;
      options.rt_software = mode == BenchMode::Software;
      const ScopedCodegenOptions scoped(options);
      const auto test = BenchCase(scenario, mode, Loop, Groups, controls.denorm_flush_f32);
      ns[static_cast<u32>(mode)] = TimeCase(vulkan, test, 4, 7);
    }
    const auto per_million = [&](double value) { return value / Tests * 1e6 / 1e3; };
    std::printf("[bvh-bench] %s: baseline %.1f us/M loop, stub %.2f us/M tests, software %.2f "
                "us/M tests (%.0f M tests in %.3f ms)\n",
                kind == 5u   ? "box32"
                : kind == 4u ? "box16"
                : kind == 6u ? "type6"
                             : "triangle",
                per_million(ns[0]), per_million(ns[1] - ns[0]), per_million(ns[2] - ns[0]),
                Tests / 1e6, ns[2] / 1e6);
  }
}

} // namespace BvhTests
