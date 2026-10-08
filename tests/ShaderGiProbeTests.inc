// Astro Bot GI-probe path tests (included by ShaderRecompilerComputeTests.cpp after
// ShaderCodegenTests.inc).
//
// The ray-bundle GI passes (GfxGiBox PS 0x4a55033774c8f01e and the 222 material ray-bundle pixel
// shaders) build per-pixel fragment linked lists: DS_APPEND on a GDS counter gives each fragment
// a node index (base + V_MBCNT of EXEC), IMAGE_ATOMIC_SWAP links it into the pixel's list, and
// GfxGiProbeTraceLinkedList (CS 0x657ad04626bf9d55) walks the lists until index 0. A duplicate
// node index can link a list to itself, and the unbounded walk then hangs the GPU.

namespace GiProbeTests {

using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;
using ShaderRecompiler::PsLiveExec;

// The GfxGiBox-shaped append: M0 = GDS range, DS_APPEND (GDS) at offset 0, index = base +
// MBCNT(EXEC), exported as a float so the target holds it exactly.
std::vector<u32> MakePixelAppendIndexShader(u32 gds_m0) {
  std::vector<u32> code;
  AppendSMovLiteral(&code, 124u, gds_m0);
  code.push_back(EncodeDs0(0x3eu, 0u, true));
  code.push_back(EncodeDs1(1u, 0u, 0u));
  code.push_back(EncodeVop1(0x01u, 3u, InlineU32(0u)));
  code.push_back(EncodeVop2(0x23u, 2u, 126u, 3u));
  code.push_back(EncodeVop2(0x24u, 2u, 127u, 2u));
  code.push_back(EncodeVop2(0x25u, 1u, Vgpr(1u), 2u));
  code.push_back(EncodeVop1(0x06u, 1u, Vgpr(1u)));
  code.push_back(EncodeExp0(0u, 0xfu, true, false, true));
  code.push_back(EncodeExp1(1u, 1u, 1u, 1u));
  AppendEnd(&code);
  return code;
}

// One small triangle around the centre of every pixel (x, y) in `pixels`, in NDC for a
// width x height viewport. Each triangle covers exactly that pixel's centre.
std::vector<u32> MakePixelTriangles(std::span<const std::pair<u32, u32>> pixels, u32 width,
                                    u32 height) {
  std::vector<u32> vertices;
  const auto push = [&](float x, float y) {
    const float ndc_x = 2.0f * x / static_cast<float>(width) - 1.0f;
    const float ndc_y = 2.0f * y / static_cast<float>(height) - 1.0f;
    for (const float value : {ndc_x, ndc_y, 0.0f, 0.0f, 0.0f, 1.0f}) {
      vertices.push_back(std::bit_cast<u32>(value));
    }
  };
  for (const auto &[x, y] : pixels) {
    const auto fx = static_cast<float>(x);
    const auto fy = static_cast<float>(y);
    push(fx + 0.2f, fy + 0.2f);
    push(fx + 0.8f, fy + 0.2f);
    push(fx + 0.5f, fy + 0.8f);
  }
  return vertices;
}

struct AppendOutcome {
  u32 covered = 0;
  u32 duplicates = 0;
  u32 out_of_range = 0;
  u32 unexpected_coverage = 0;
  u32 counter = 0;
};

AppendOutcome RunPixelAppend(VulkanHarness *vulkan, const char *name, bool live_election,
                             std::span<const std::pair<u32, u32>> pixels, u32 width,
                             u32 height, u32 initial_counter,
                             PsLiveExec live_exec = PsLiveExec::Off) {
  auto options = GetCodegenOptions();
  options.ps_append_live_election = live_election;
  options.ps_live_exec = live_exec;
  ScopedCodegenOptions scoped(options);

  GraphicsCase test;
  test.name = name;
  test.fragment_code = MakePixelAppendIndexShader(0x00000100u);
  const auto compiled = CompileFragmentCase(test);
  const auto helper_decorations = CodegenTests::CountSpirv(compiled.spirv, "BuiltIn HelperInvocation");
  Require(name, "SPIR-V",
          helper_decorations == (live_election || live_exec != PsLiveExec::Off ? 1u : 0u),
          "gl_HelperInvocation declaration does not follow KYTY_PS_APPEND_LIVE_ELECTION and "
          "KYTY_PS_LIVE_EXEC");

  std::vector<u32> gds_initial(64u, 0u);
  gds_initial[0] = initial_counter;
  auto gds = vulkan->CreateStorageBuffer(name, gds_initial, gds_initial.size());
  const auto vertices = MakePixelTriangles(pixels, width, height);
  const auto words = vulkan->RenderFragmentWithGds(name, compiled, width, height, vertices, gds);
  AppendOutcome outcome;
  outcome.counter = vulkan->ReadBuffer(name, gds, 1u)[0];
  vulkan->DestroyBuffer(&gds);

  std::set<u32> seen;
  for (u32 y = 0; y < height; y++) {
    for (u32 x = 0; x < width; x++) {
      const float value = std::bit_cast<float>(words[(y * width + x) * 4u]);
      const bool expected = std::ranges::any_of(
          pixels, [&](const auto &pixel) { return pixel.first == x && pixel.second == y; });
      if (value == -1.0f) {
        outcome.unexpected_coverage += expected ? 1u : 0u;
        continue;
      }
      if (!expected) {
        outcome.unexpected_coverage++;
        continue;
      }
      outcome.covered++;
      const auto index = static_cast<u32>(value);
      if (static_cast<float>(index) != value || index < initial_counter ||
          index >= outcome.counter) {
        outcome.out_of_range++;
      }
      if (!seen.insert(index).second) {
        outcome.duplicates++;
      }
    }
  }
  return outcome;
}

// Only the lower-right pixel of every 2x2 quad is covered, so the first EXEC lane of every quad
// (and of every host subgroup) is a helper invocation.
void CheckPixelAppendHelperElection(VulkanHarness *vulkan) {
  constexpr const char *name = "PixelAppendHelperElection";
  constexpr u32 width = 16;
  constexpr u32 height = 16;
  constexpr u32 initial = 100;
  std::vector<std::pair<u32, u32>> pixels;
  for (u32 y = 1; y < height; y += 2) {
    for (u32 x = 1; x < width; x += 2) {
      pixels.emplace_back(x, y);
    }
  }
  const auto off = RunPixelAppend(vulkan, name, false, pixels, width, height, initial);
  std::printf("[graphics] %-31s switch off: covered %u, duplicate indices %u, out of range %u, "
              "counter %u -> %u\n",
              name, off.covered, off.duplicates, off.out_of_range, initial, off.counter);
  const auto on = RunPixelAppend(vulkan, name, true, pixels, width, height, initial);
  Require(name, "coverage", on.covered == pixels.size() && on.unexpected_coverage == 0,
          "the test triangles did not cover exactly one pixel each");
  Require(name, "append", on.duplicates == 0 && on.out_of_range == 0,
          "a pixel-shader DS_APPEND returned duplicate or out-of-range indices");
  Require(name, "counter", on.counter >= initial + pixels.size(),
          "the GDS append counter advanced by less than the live lane count");
  std::printf("[graphics] %-31s ok (switch on: %u unique indices, counter %u -> %u)\n", name,
              on.covered, initial, on.counter);
}

// KYTY_PS_LIVE_EXEC: with EXEC starting at the non-helper invocations, the append counts and
// indexes exactly the covered pixels, as on the PS5 (no slot is left for a helper lane).
void CheckPixelAppendLiveExec(VulkanHarness *vulkan) {
  constexpr const char *name = "PixelAppendLiveExec";
  constexpr u32 width = 16;
  constexpr u32 height = 16;
  constexpr u32 initial = 100;
  std::vector<std::pair<u32, u32>> pixels;
  for (u32 y = 1; y < height; y += 2) {
    for (u32 x = 1; x < width; x += 2) {
      pixels.emplace_back(x, y);
    }
  }
  const auto count = static_cast<u32>(pixels.size());
  for (const bool live_election : {false, true}) {
    const auto on = RunPixelAppend(vulkan, name, live_election, pixels, width, height, initial,
                                   PsLiveExec::AppendConsume);
    Require(name, "coverage", on.covered == count && on.unexpected_coverage == 0,
            "the test triangles did not cover exactly one pixel each");
    Require(name, "append", on.duplicates == 0 && on.out_of_range == 0,
            "a pixel-shader DS_APPEND returned duplicate or out-of-range indices");
    Require(name, "counter", on.counter == initial + count,
            "the GDS append counter did not advance by exactly the covered pixel count");
  }
  std::printf("[graphics] %-31s ok (%u unique indices, counter %u -> %u)\n", name, count, initial,
              initial + count);
}

// Saves EXEC, switches to whole quad mode and exports (S_BCNT1 of the WQM EXEC, S_BCNT1 of the
// saved EXEC) as floats under the saved EXEC.
std::vector<u32> MakePixelWqmCountShader() {
  return {
      EncodeSop1(0x04u, 0u, 126u),   // s_mov_b64 s[0:1], exec
      EncodeSop1(0x0au, 126u, 126u), // s_wqm_b64 exec, exec
      EncodeSop1(0x10u, 2u, 126u),   // s_bcnt1_i32_b64 s2, exec
      EncodeSop1(0x10u, 3u, 0u),     // s_bcnt1_i32_b64 s3, s[0:1]
      EncodeVop1(0x01u, 1u, 2u),     // v_mov_b32 v1, s2
      EncodeVop1(0x06u, 1u, Vgpr(1u)),
      EncodeVop1(0x01u, 2u, 3u),     // v_mov_b32 v2, s3
      EncodeVop1(0x06u, 2u, Vgpr(2u)),
      EncodeSop1(0x04u, 126u, 0u),   // s_mov_b64 exec, s[0:1]
      EncodeExp0(0u, 0xfu, true, false, true),
      EncodeExp1(1u, 2u, 1u, 1u),
      0xbf810000u,
  };
}

// KYTY_PS_LIVE_EXEC=all: every quad of the test has exactly one covered pixel, so the saved
// (live) EXEC counts one lane per quad of the subgroup and S_WQM_B64 four; with the switch off
// both count every lane.
void CheckPixelWqmLiveExec(VulkanHarness *vulkan) {
  constexpr const char *name = "PixelWqmLiveExec";
  constexpr u32 width = 16;
  constexpr u32 height = 16;
  std::vector<std::pair<u32, u32>> pixels;
  for (u32 y = 0; y < height; y += 2) {
    for (u32 x = 0; x < width; x += 2) {
      pixels.emplace_back(x + ((x / 2u + y / 2u) & 1u), y + ((y / 2u) & 1u));
    }
  }
  const auto vertices = MakePixelTriangles(pixels, width, height);
  for (const auto mode : {PsLiveExec::Off, PsLiveExec::All}) {
    auto options = GetCodegenOptions();
    options.ps_live_exec = mode;
    ScopedCodegenOptions scoped(options);
    GraphicsCase test;
    test.name = name;
    test.fragment_code = MakePixelWqmCountShader();
    const auto compiled = CompileFragmentCase(test);
    const std::vector<u32> gds_initial(16u, 0u);
    auto gds = vulkan->CreateStorageBuffer(name, gds_initial, gds_initial.size());
    const auto words = vulkan->RenderFragmentWithGds(name, compiled, width, height, vertices, gds);
    vulkan->DestroyBuffer(&gds);
    u32 covered = 0;
    u32 exact = 0;
    u32 min_live = ~0u;
    u32 max_live = 0;
    for (const auto &[x, y] : pixels) {
      const float wqm_value = std::bit_cast<float>(words[(y * width + x) * 4u + 0u]);
      const float live_value = std::bit_cast<float>(words[(y * width + x) * 4u + 1u]);
      if (wqm_value == -1.0f) {
        continue;
      }
      covered++;
      const auto wqm = static_cast<u32>(wqm_value);
      const auto live = static_cast<u32>(live_value);
      exact += (live >= 1u && wqm == 4u * live) ? 1u : 0u;
      min_live = std::min(min_live, live);
      max_live = std::max(max_live, live);
    }
    Require(name, "coverage", covered == pixels.size(),
            "the test triangles did not cover exactly one pixel each");
    if (mode == PsLiveExec::Off) {
      std::printf("[graphics] %-31s switch off: %u of %u pixels see WQM lanes = 4 x live lanes "
                  "(live %u..%u)\n",
                  name, exact, covered, min_live, max_live);
      continue;
    }
    Require(name, "exec", exact == covered,
            "the initial EXEC did not hold exactly the covered pixels, or S_WQM_B64 did not add "
            "their quads");
    std::printf("[graphics] %-31s ok (switch all: WQM lanes = 4 x live lanes, live %u..%u per "
                "subgroup)\n",
                name, min_live, max_live);
  }
}

// CPU only: KYTY_PS_LIVE_EXEC=1 reads gl_HelperInvocation in pixel shaders with DS_APPEND or
// DS_CONSUME, =all in every pixel shader.
void CheckPixelLiveExecCodegen() {
  constexpr const char *name = "PixelLiveExecCodegen";
  const ShaderRecompiler::CodegenOptions defaults;
  Require(name, "defaults",
          defaults.ps_live_exec == PsLiveExec::AppendConsume && defaults.ps_append_live_election,
          "KYTY_PS_LIVE_EXEC=1 and KYTY_PS_APPEND_LIVE_ELECTION=1 are not the defaults");
  for (const auto mode : {PsLiveExec::Off, PsLiveExec::AppendConsume, PsLiveExec::All}) {
    auto options = GetCodegenOptions();
    options.ps_append_live_election = false;
    options.ps_live_exec = mode;
    ScopedCodegenOptions scoped(options);
    GraphicsCase append;
    append.name = name;
    append.fragment_code = MakePixelAppendIndexShader(0x00000100u);
    GraphicsCase plain;
    plain.name = name;
    plain.fragment_code = MakePixelWqmCountShader();
    const auto append_helpers =
        CodegenTests::CountSpirv(CompileFragmentCase(append).spirv, "BuiltIn HelperInvocation");
    const auto plain_helpers =
        CodegenTests::CountSpirv(CompileFragmentCase(plain).spirv, "BuiltIn HelperInvocation");
    Require(name, "SPIR-V", append_helpers == (mode == PsLiveExec::Off ? 0u : 1u),
            "the append pixel shader does not follow KYTY_PS_LIVE_EXEC");
    Require(name, "SPIR-V", plain_helpers == (mode == PsLiveExec::All ? 1u : 0u),
            "a pixel shader without DS_APPEND does not follow KYTY_PS_LIVE_EXEC");
  }
  std::printf("[host]    %-32s ok\n", name);
}

// CPU only: the switch adds gl_HelperInvocation to pixel shaders with DS_APPEND/DS_CONSUME and
// to nothing else.
void CheckPixelAppendElectionCodegen() {
  constexpr const char *name = "PixelAppendElectionCodegen";
  for (const bool live_election : {false, true}) {
    auto options = GetCodegenOptions();
    options.ps_append_live_election = live_election;
    options.ps_live_exec = PsLiveExec::Off;
    ScopedCodegenOptions scoped(options);
    GraphicsCase test;
    test.name = name;
    test.fragment_code = MakePixelAppendIndexShader(0x00000100u);
    const auto compiled = CompileFragmentCase(test);
    Require(name, "SPIR-V",
            CodegenTests::CountSpirv(compiled.spirv, "BuiltIn HelperInvocation") ==
                (live_election ? 1u : 0u),
            "gl_HelperInvocation declaration does not follow KYTY_PS_APPEND_LIVE_ELECTION");
    GraphicsCase plain;
    plain.name = name;
    plain.fragment_code = {EncodeVop1(0x01u, 1u, InlineU32(0u)), EncodeExp0(0u, 0xfu, true, false, true),
                           EncodeExp1(1u, 1u, 1u, 1u), 0xbf810000u};
    const auto plain_compiled = CompileFragmentCase(plain);
    Require(name, "SPIR-V",
            CodegenTests::CountSpirv(plain_compiled.spirv, "BuiltIn HelperInvocation") == 0u,
            "a pixel shader without DS_APPEND declared gl_HelperInvocation");
  }
  std::printf("[host]    %-32s ok\n", name);
}

// A scalar loop whose exit (s_cbranch_scc1 on s0 < 0) is never taken, then every thread
// stores 7 at its lane. Only ever dispatched with KYTY_LOOP_GUARD applying to it.
TestCase MakeEndlessLoopCase() {
  using O = ShaderOpcode;
  std::vector<u32> code = {
      EncodeSMovB32(0, InlineU32(0)),
      EncodeSopc(0x0a, 0, InlineU32(0)),
      EncodeSopp(0x05, 2),
      EncodeSop2(0x00, 0, 0, InlineU32(1)),
      EncodeSopp(0x02, 0xfffcu),
      EncodeVop1(0x01, 1, InlineU32(7)),
  };
  AppendStoreVgprAtLaneDwordOffset(&code, 1, 0, 0);
  AppendEnd(&code);
  TestCase test;
  test.name = "LoopGuardEndlessLoop";
  test.code = std::move(code);
  test.initial.assign(64, 0u);
  test.expected.assign(64, 7u);
  test.opcodes = {O::S_MOV_B32, O::S_CMP_LT_U32, O::S_CBRANCH_SCC1, O::S_ADD_U32,
                  O::S_BRANCH,  O::V_MOV_B32,    O::V_LSHLREV_B32,  O::BUFFER_STORE_DWORD,
                  O::S_ENDPGM};
  test.compute_info.threads_num[0] = 64;
  test.compute_info.threads_num[1] = 1;
  test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.has_compute_info = true;
  return test;
}

// CPU only: KYTY_LOOP_GUARD instruments exactly the listed shaders (test programs hash to 0).
void CheckLoopGuardCodegen() {
  constexpr const char *name = "LoopGuardCodegen";
  const auto test = MakeEndlessLoopCase();
  size_t atomics_off = 0;
  for (const bool guarded : {false, true}) {
    auto options = GetCodegenOptions();
    options.loop_guard_budget = guarded ? 64u : 0u;
    options.loop_guard_shaders = {guarded ? 0u : 1u};
    ScopedCodegenOptions scoped(options);
    const auto compiled = CompileCase(test, 32u);
    const auto names = CodegenTests::CountSpirv(compiled.spirv, "OpName %loop_guard");
    const auto atomics = CodegenTests::CountSpirv(compiled.spirv, "OpAtomicIAdd");
    const bool gds = ShaderRecompiler::IR::FindBinding(
                         compiled.program.bindings,
                         ShaderRecompiler::IR::DescriptorBindingKind::Gds) != nullptr;
    if (!guarded) {
      Require(name, "SPIR-V", names == 0u && !gds,
              "an unguarded shader carried the loop guard or a GDS binding");
      atomics_off = atomics;
    } else {
      Require(name, "SPIR-V", names == 1u && gds && atomics == atomics_off + 1u,
              "a guarded shader lacks the loop-guard counter, its GDS binding or its report");
    }
  }
  // KYTY_LOOP_GUARD_SHADERS=all: an unlisted shader is guarded too, and its report also records
  // the shader hash (two GDS exchanges).
  {
    auto options = GetCodegenOptions();
    options.loop_guard_budget = 64u;
    options.loop_guard_shaders = {1u};
    options.loop_guard_all = true;
    ScopedCodegenOptions scoped(options);
    const auto compiled = CompileCase(test, 32u);
    Require(name, "SPIR-V",
            CodegenTests::CountSpirv(compiled.spirv, "OpName %loop_guard") == 1u &&
                CodegenTests::CountSpirv(compiled.spirv, "OpAtomicIAdd") == atomics_off + 1u &&
                CodegenTests::CountSpirv(compiled.spirv, "OpAtomicExchange") == 2u,
            "KYTY_LOOP_GUARD_SHADERS=all did not guard the shader or record its hash");
  }
  std::printf("[host]    %-32s ok\n", name);
}

// CPU only: KYTY_PS_HELPER_ATOMICS_SKIP. A pixel shader's BUFFER_ATOMIC_INC is an emulated
// read-modify-write (a compare-exchange loop); helper invocations skip it by default, which declares
// gl_HelperInvocation, and with the switch off the shader is as before. (BUFFER_STORE_BYTE was the
// case here until byte stores became native 8-bit stores, upstream c3f496271.)
void CheckPixelHelperAtomicsCodegen() {
  constexpr const char *name = "PixelHelperAtomicsCodegen";
  const ShaderRecompiler::CodegenOptions defaults;
  Require(name, "defaults", defaults.ps_helper_atomics_skip,
          "KYTY_PS_HELPER_ATOMICS_SKIP=1 is not the default");
  for (const bool skip : {false, true}) {
    auto options = GetCodegenOptions();
    options.ps_helper_atomics_skip = skip;
    options.ps_append_live_election = false;
    options.ps_live_exec = PsLiveExec::Off;
    ScopedCodegenOptions scoped(options);
    GraphicsCase test;
    test.name = name;
    std::vector<u32> code;
    AppendVMovU32(&code, 1u, 0x5au);
    AppendVMovU32(&code, 31u, 0u);
    AppendBufferStoreOpcode(&code, 0x3cu, 1u, 31u); // buffer_atomic_inc v1, v31, s[48:51] offen
    code.push_back(EncodeVop1(0x01u, 2u, InlineU32(0u)));
    code.push_back(EncodeExp0(0u, 0xfu, true, false, true));
    code.push_back(EncodeExp1(2u, 2u, 2u, 2u));
    code.push_back(0xbf810000u);
    test.fragment_code = code;
    const auto compiled = CompileFragmentCase(test);
    const auto helpers = CodegenTests::CountSpirv(compiled.spirv, "BuiltIn HelperInvocation");
    const auto cas = CodegenTests::CountSpirv(compiled.spirv, "OpAtomicCompareExchange");
    Require(name, "SPIR-V", cas == 1u, "the atomic increment is not one compare-exchange loop");
    Require(name, "SPIR-V", helpers == (skip ? 1u : 0u),
            "gl_HelperInvocation declaration does not follow KYTY_PS_HELPER_ATOMICS_SKIP");
  }
  std::printf("[host]    %-32s ok\n", name);
}

// GPU: the guard ends an endless loop, the threads finish, and every host invocation reports
// once in the last GDS dword.
void CheckLoopGuardEndsEndlessLoop(VulkanHarness *vulkan) {
  constexpr const char *name = "LoopGuardEndlessLoop";
  const auto test = MakeEndlessLoopCase();
  auto options = GetCodegenOptions();
  options.loop_guard_budget = 64u;
  options.loop_guard_shaders = {0u};
  ScopedCodegenOptions scoped(options);
  const auto compiled = CompileCase(test, vulkan->SubgroupSize());
  auto buffer = vulkan->CreateStorageBuffer(name, test.initial, test.initial.size());
  const std::vector<u32> gds_initial(16u, 0u);
  auto gds = vulkan->CreateStorageBuffer(name, gds_initial, gds_initial.size());
  vulkan->Dispatch(test, compiled, buffer, &gds);
  const auto actual = vulkan->ReadBuffer(name, buffer, test.expected.size());
  const auto gds_words = vulkan->ReadBuffer(name, gds, gds_initial.size());
  vulkan->DestroyBuffer(&gds);
  vulkan->DestroyBuffer(&buffer);
  Require(name, "result", actual == test.expected,
          "the guarded endless loop did not exit into the store");
  const u32 lanes = test.compute_info.wave_size == 64u && vulkan->SubgroupSize() == 32u ? 2u : 1u;
  Require(name, "report", gds_words.back() == 64u / lanes &&
                              std::all_of(gds_words.begin(), gds_words.end() - 1,
                                          [](u32 word) { return word == 0u; }),
          "the exhausted invocations were not counted once each in the last GDS dword");
  std::printf("[compute] %-32s ok (%u reports)\n", name, gds_words.back());
}

// CPU only (--gi-decode-inventory <file.bin> [out.txt]): decodes a whole guest shader with the
// recompiler's decoder, continuing past BVH instructions (DecodeProgram stops at the first one),
// and prints one line "file,instructions,unsupported,bvh" plus, with out.txt, the listing.
int GiDecodeInventory(const char *path, const char *listing_path) {
  std::FILE *file = std::fopen(path, "rb");
  if (file == nullptr) {
    std::printf("%s,open\n", path);
    return 1;
  }
  std::vector<u32> code;
  u32 word = 0;
  while (std::fread(&word, sizeof(word), 1, file) == 1) {
    code.push_back(word);
  }
  std::fclose(file);
  namespace D = ShaderRecompiler::Decoder;
  D::Program program;
  program.code = code;
  std::vector<bool> targets(code.size() + 1u, false);
  size_t unsupported = 0;
  size_t bvh = 0;
  for (u32 index = 0; index < code.size();) {
    auto &inst = program.instructions.emplace_back();
    D::DecodeInstruction(code, index, inst);
    index += inst.word_count;
    if (inst.opcode == D::Opcode::UNSUPPORTED) {
      unsupported++;
    }
    if (inst.family == D::Family::MIMG && (inst.opcode_id == 0xe6u || inst.opcode_id == 0xe7u)) {
      bvh++;
    }
    if (D::IsDirectBranch(inst.opcode) && inst.branch_target / 4u < targets.size()) {
      targets[inst.branch_target / 4u] = true;
    }
    if (inst.opcode == D::Opcode::S_ENDPGM &&
        std::none_of(targets.begin() + index, targets.end(), [](bool t) { return t; })) {
      break;
    }
  }
  if (listing_path != nullptr) {
    if (std::FILE *out = std::fopen(listing_path, "wb"); out != nullptr) {
      const auto text = D::ProgramToString(program);
      std::fwrite(text.data(), 1, text.size(), out);
      std::fclose(out);
    }
  }
  std::printf("%s,%zu,%zu,%zu\n", path, program.instructions.size(), unsupported, bvh);
  return 0;
}

} // namespace GiProbeTests
