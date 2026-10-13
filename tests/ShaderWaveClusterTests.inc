// KYTY_WAVE32_CLUSTERS tests (included by ShaderRecompilerComputeTests.cpp after
// ShaderCodegenTests.inc).
//
// The AMD proprietary driver runs vertex, mesh and pixel shaders on 64-wide subgroups and lets only
// compute pipelines require 32, so a wave32 graphics program shares a host subgroup with a second
// guest wave. Its waterfall loops (V_READFIRSTLANE, V_CMP_EQ, S_AND_SAVEEXEC_B32, S_XOR_B32 EXEC,
// S_CBRANCH_EXECNZ) then read the other wave's EXEC word and key, never remove their own lanes,
// and spin until a device reset. Clusters keep every guest wave to its own 32 host lanes. A 32-wide
// host (the RTX 3090) cannot show the hang; these tests check that the cluster code is chosen only
// where it is needed, leaves the SPIR-V unchanged elsewhere, validates, and gives the guest's
// results when forced (KYTY_WAVE32_CLUSTERS=force) on a 32-wide host.

namespace WaveClusterTests {

using CodegenTests::CountSpirv;
using CodegenTests::RequireEqualWords;
using CodegenTests::RunCollect;
using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;
using ShaderRecompiler::Wave32Clusters;
namespace Spirv = ShaderRecompiler::Spirv;

// Restores the device layer's wave-cluster stages.
class ScopedHostWaveClusters {
public:
  explicit ScopedHostWaveClusters(const Spirv::HostWaveClusters &clusters)
      : m_saved(Spirv::GetHostWaveClusters()) {
    Spirv::SetHostWaveClusters(clusters);
  }
  ~ScopedHostWaveClusters() { Spirv::SetHostWaveClusters(m_saved); }
  ScopedHostWaveClusters(const ScopedHostWaveClusters &) = delete;
  ScopedHostWaveClusters &operator=(const ScopedHostWaveClusters &) = delete;

private:
  Spirv::HostWaveClusters m_saved;
};

ShaderRecompiler::CodegenOptions WithClusters(Wave32Clusters mode) {
  auto options = GetCodegenOptions();
  options.wave32_clusters = mode;
  return options;
}

// A wave32 waterfall loop over eight keys (lane / 8) in a 64-thread workgroup (two waves): every
// lane stores the key it was served with, then its wave's iteration count (4: one per key).
TestCase MakeWaterfallWave32() {
  using O = ShaderOpcode;
  constexpr u32 Threads = 64;
  std::vector<u32> code;
  code.push_back(EncodeVop2(0x16u, 1u, InlineU32(3u), 0u)); // v_lshrrev_b32 v1, 3, v0
  code.push_back(EncodeSop1(0x03u, 10u, 126u));              // s_mov_b32 s10, exec_lo
  AppendVMovLiteral(&code, 2u, 0xdeadbeefu);
  code.push_back(EncodeSMovB32(11u, InlineU32(0u)));
  const auto loop = code.size();
  code.push_back(EncodeVop1(0x02u, 0u, Vgpr(1u)));               // v_readfirstlane_b32 s0, v1
  code.push_back(EncodeVopc(0xc2u, 0u, 1u));                      // v_cmp_eq_u32 vcc_lo, s0, v1
  code.push_back(EncodeSop1(0x3cu, 2u, 106u));                    // s_and_saveexec_b32 s2, vcc_lo
  code.push_back(EncodeVop1(0x01u, 2u, 0u));                      // v_mov_b32 v2, s0
  code.push_back(EncodeSop2(0x00u, 11u, 11u, InlineU32(1u)));     // s_add_u32 s11, s11, 1
  code.push_back(EncodeSop2(0x12u, 126u, 126u, 2u));              // s_xor_b32 exec_lo, exec_lo, s2
  const auto branch = code.size();
  code.push_back(EncodeSopp(0x09u, static_cast<u32>(static_cast<int32_t>(loop) -
                                                     static_cast<int32_t>(branch + 1u)) &
                                       0xffffu)); // s_cbranch_execnz loop
  code.push_back(EncodeSop1(0x03u, 126u, 10u));   // s_mov_b32 exec_lo, s10
  code.push_back(EncodeVop1(0x01u, 3u, 11u));     // v_mov_b32 v3, s11
  AppendStoreVgprAtLaneDwordOffset(&code, 2u, 0u, 0u);
  AppendStoreVgprAtLaneDwordOffset(&code, 3u, 0u, Threads);
  AppendEnd(&code);

  TestCase test;
  test.name = "Wave32ClusterWaterfall";
  test.code = std::move(code);
  test.initial = std::vector<u32>(2u * Threads, 0u);
  test.expected.resize(2u * Threads);
  for (u32 lane = 0; lane < Threads; lane++) {
    test.expected[lane] = lane / 8u;
    test.expected[Threads + lane] = 4u;
  }
  test.opcodes = {O::V_LSHRREV_B32,       O::S_MOV_B32,   O::V_MOV_B32,         O::V_READFIRSTLANE_B32,
                  O::V_CMP_EQ_U32,        O::S_AND_SAVEEXEC_B32, O::S_ADD_U32, O::S_XOR_B32,
                  O::S_CBRANCH_EXECNZ,    O::V_LSHLREV_B32, O::V_ADD_NC_U32,  O::BUFFER_STORE_DWORD,
                  O::S_ENDPGM};
  test.compute_info.threads_num[0] = Threads;
  test.compute_info.threads_num[1] = 1;
  test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = 32;
  test.has_compute_info = true;
  return test;
}

// A wave32 pixel shader that reads its first lane's V_MBCNT lane index (V_READFIRSTLANE).
GraphicsCase MakePixelFirstLaneWave32() {
  GraphicsCase test;
  test.name = "Wave32ClusterPixelFirstLane";
  test.wave_size = 32;
  AppendVop3(&test.fragment_code, 0x365u, 1u, 193u, InlineU32(0u)); // v_mbcnt_lo_u32_b32 v1, -1, 0
  test.fragment_code.push_back(EncodeVop1(0x02u, 0u, Vgpr(1u)));     // v_readfirstlane_b32 s0, v1
  test.fragment_code.push_back(EncodeVop1(0x01u, 2u, 0u));           // v_mov_b32 v2, s0
  test.fragment_code.push_back(EncodeVop1(0x06u, 2u, Vgpr(2u)));     // v_cvt_f32_u32 v2, v2
  test.fragment_code.push_back(EncodeExp0(0u, 0xfu, true, false, true));
  test.fragment_code.push_back(EncodeExp1(2u, 2u, 2u, 2u));
  AppendEnd(&test.fragment_code);
  return test;
}

// CPU only: which programs take clusters, and that every other program's SPIR-V is unchanged.
void CheckWaveClusterSelection() {
  constexpr const char *name = "Wave32ClusterSelection";
  namespace IR = ShaderRecompiler::IR;
  struct Row {
    ShaderType stage;
    u32 wave_size;
    Spirv::HostWaveClusters host;
    Wave32Clusters mode;
    bool active;
  };
  const Spirv::HostWaveClusters none{};
  const Spirv::HostWaveClusters amd{.vertex = true, .pixel = true, .mesh = true};
  const Spirv::HostWaveClusters radv{.vertex = true, .pixel = false, .mesh = false};
  const Row rows[] = {
      {ShaderType::Pixel, 32, amd, Wave32Clusters::Auto, true},
      {ShaderType::Vertex, 32, amd, Wave32Clusters::Auto, true},
      {ShaderType::Local, 32, amd, Wave32Clusters::Auto, true},
      {ShaderType::TessellationEvaluation, 32, amd, Wave32Clusters::Auto, true},
      {ShaderType::Mesh, 32, amd, Wave32Clusters::Auto, true},
      {ShaderType::Pixel, 64, amd, Wave32Clusters::Auto, false},
      {ShaderType::Mesh, 64, amd, Wave32Clusters::Force, false},
      {ShaderType::Compute, 32, amd, Wave32Clusters::Auto, false},
      {ShaderType::Compute, 32, none, Wave32Clusters::Force, true},
      {ShaderType::TessellationControl, 32, amd, Wave32Clusters::Force, false},
      {ShaderType::Pixel, 32, amd, Wave32Clusters::Off, false},
      {ShaderType::Pixel, 32, none, Wave32Clusters::Auto, false},
      {ShaderType::Pixel, 32, none, Wave32Clusters::Force, true},
      {ShaderType::Pixel, 32, radv, Wave32Clusters::Auto, false},
      {ShaderType::Vertex, 32, radv, Wave32Clusters::Auto, true},
      {ShaderType::Mesh, 32, radv, Wave32Clusters::Auto, false},
  };
  for (const auto &row : rows) {
    ScopedHostWaveClusters host(row.host);
    ScopedCodegenOptions scoped(WithClusters(row.mode));
    IR::Program program;
    program.stage = row.stage;
    program.wave_size = row.wave_size;
    Require(name, "selection", Spirv::WaveClustersActive(program) == row.active,
            "stage " + std::to_string(static_cast<u32>(row.stage)) + " wave" +
                std::to_string(row.wave_size) + " mode " +
                std::to_string(static_cast<u32>(row.mode)) + ": expected " +
                (row.active ? "clusters" : "no clusters"));
  }

  // NVIDIA (no cluster stages) and Off: bit-identical SPIR-V; Force: the cluster code, valid.
  const auto waterfall = MakeWaterfallWave32();
  std::vector<u32> base;
  {
    ScopedHostWaveClusters host(none);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Off));
    base = CompileCase(waterfall, 32u).spirv;
  }
  for (const auto &clusters : {none, amd}) {
    // Compute never takes clusters outside Force (its pipelines require the wave size).
    ScopedHostWaveClusters host(clusters);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Auto));
    Require(name, "auto compute", CompileCase(waterfall, 32u).spirv == base,
            "KYTY_WAVE32_CLUSTERS=auto changed a compute program");
  }
  std::vector<u32> forced;
  {
    ScopedHostWaveClusters host(none);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Force));
    forced = CompileCase(waterfall, 32u).spirv;
  }
  ValidateSpirv(name, forced);
  Require(name, "force",
          CountSpirv(forced, "OpVectorExtractDynamic") > CountSpirv(base, "OpVectorExtractDynamic") &&
              CountSpirv(forced, "BuiltIn SubgroupLocalInvocationId") == 1u,
          "forced clusters did not select the wave's ballot word by its subgroup lane");
  {
    // A wave64 program is never clustered.
    auto wave64 = waterfall;
    wave64.compute_info.wave_size = 64;
    std::vector<u32> off;
    {
      ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Off));
      off = CompileCase(wave64, 32u).spirv;
    }
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Force));
    Require(name, "wave64", CompileCase(wave64, 32u).spirv == off,
            "KYTY_WAVE32_CLUSTERS=force changed a wave64 program");
  }

  // A wave32 pixel shader on an AMD-like host: clustered; on NVIDIA: unchanged.
  const auto pixel = MakePixelFirstLaneWave32();
  std::vector<u32> pixel_nvidia;
  std::vector<u32> pixel_off;
  std::vector<u32> pixel_amd;
  {
    ScopedHostWaveClusters host(none);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Auto));
    pixel_nvidia = CompileFragmentCase(pixel).spirv;
  }
  {
    ScopedHostWaveClusters host(amd);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Off));
    pixel_off = CompileFragmentCase(pixel).spirv;
  }
  {
    ScopedHostWaveClusters host(amd);
    ScopedCodegenOptions scoped(WithClusters(Wave32Clusters::Auto));
    pixel_amd = CompileFragmentCase(pixel).spirv;
  }
  ValidateSpirv(name, pixel_amd);
  Require(name, "pixel", pixel_nvidia == pixel_off && pixel_amd != pixel_off &&
                             CountSpirv(pixel_amd, "OpVectorExtractDynamic") >
                                 CountSpirv(pixel_off, "OpVectorExtractDynamic"),
          "a wave32 pixel shader is not clustered exactly where the host needs it");
  std::printf("[codegen] %-32s ok (%zu selection rows)\n", name, std::size(rows));
}

// GPU: the waterfall gives the guest's results with clusters off and forced (on a 32-wide host the
// cluster is the whole subgroup, so both must match the ISA).
void CheckWaveClusterWaterfall(VulkanHarness *vulkan) {
  constexpr const char *name = "Wave32ClusterWaterfall";
  const auto test = MakeWaterfallWave32();
  for (const auto mode : {Wave32Clusters::Off, Wave32Clusters::Force}) {
    ScopedCodegenOptions scoped(WithClusters(mode));
    const auto actual = RunCollect(vulkan, test, test.expected.size());
    RequireEqualWords(name, mode == Wave32Clusters::Off ? "off" : "force", test.expected, actual);
  }
  std::printf("[compute] %-32s ok (subgroup %u)\n", name, vulkan->SubgroupSize());
}

// KYTY_LDS_WAITCNT_BARRIER: S_WAITCNT lgkmcnt(0) after an LDS write is a subgroup-scope memory
// barrier by default (never a workgroup-scope one, which the AMD driver turns into a workgroup
// barrier in mesh shaders: Astro Bot's title-screen particle mesh shader then hung at its next
// S_BARRIER), workgroup scope with =workgroup, none with =0. CPU: the emitted barrier; GPU (with
// `vulkan`): the intra-wave LDS exchange (a wave64 split over two lanes per invocation on a
// 32-wide host) reads its neighbours' writes at either scope.
void CheckLdsWaitcntBarrierScope(VulkanHarness *vulkan) {
  constexpr const char *name = "LdsWaitcntBarrierScope";
  using ShaderRecompiler::LdsWaitcntBarrier;
  auto test = LdsWaitcntOrdersIntraWaveExchange();
  test.ir_counts.clear(); // checked per mode below
  test.required_spirv.clear();
  struct Mode {
    LdsWaitcntBarrier mode;
    const char *label;
    const char *barrier; // AcquireRelease | WorkgroupMemory = 264
  };
  const Mode modes[] = {
      {LdsWaitcntBarrier::Subgroup, "subgroup", "OpMemoryBarrier %uint_3 %uint_264"},
      {LdsWaitcntBarrier::Workgroup, "workgroup", "OpMemoryBarrier %uint_2 %uint_264"},
      {LdsWaitcntBarrier::Off, "off", nullptr},
  };
  {
    // The default is subgroup scope.
    const ShaderRecompiler::CodegenOptions defaults;
    Require(name, "default", defaults.lds_waitcnt_barrier == LdsWaitcntBarrier::Subgroup,
            "the default LDS waitcnt barrier is not subgroup-scoped");
  }
  for (const auto &mode : modes) {
    auto options = GetCodegenOptions();
    options.lds_waitcnt_barrier = mode.mode;
    ScopedCodegenOptions scoped(options);
    const auto spirv = CompileCase(test, vulkan != nullptr ? vulkan->SubgroupSize() : 32u).spirv;
    ValidateSpirv(name, spirv);
    const auto barriers = CountSpirv(spirv, "OpMemoryBarrier");
    if (mode.barrier != nullptr) {
      Require(name, mode.label, barriers == 1u && CountSpirv(spirv, mode.barrier) == 1u,
              std::string("expected one ") + mode.barrier + ", found " + std::to_string(barriers) +
                  " OpMemoryBarrier");
    } else {
      Require(name, mode.label, barriers == 0u, "KYTY_LDS_WAITCNT_BARRIER=0 left a barrier");
    }
    Require(name, mode.label, CountSpirv(spirv, "OpControlBarrier") == 0u,
            "an LDS waitcnt became an execution barrier");
    if (vulkan != nullptr && mode.mode != LdsWaitcntBarrier::Off) {
      const auto actual = RunCollect(vulkan, test, test.expected.size());
      RequireEqualWords(name, mode.label, test.expected, actual);
    }
  }
  std::printf("[%s] %-32s ok\n", vulkan != nullptr ? "compute" : "codegen", name);
}

} // namespace WaveClusterTests
