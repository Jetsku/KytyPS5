// Code-generation equivalence tests (included by ShaderRecompilerComputeTests.cpp).
//
// Each change to emitted SPIR-V that must keep results bit-exact is exercised here twice: once with
// the new code generation and once with its KYTY_* switch turned off, on the same inputs. The
// outputs must match each other (and, where one exists, an exact CPU reference).

namespace CodegenTests {

using ShaderRecompiler::CodegenOptions;
using ShaderRecompiler::GetCodegenOptions;
using ShaderRecompiler::SetCodegenOptions;

class ScopedCodegenOptions {
public:
  explicit ScopedCodegenOptions(const CodegenOptions &options)
      : m_saved(GetCodegenOptions()) {
    SetCodegenOptions(options);
  }
  ~ScopedCodegenOptions() { SetCodegenOptions(m_saved); }
  ScopedCodegenOptions(const ScopedCodegenOptions &) = delete;
  ScopedCodegenOptions &operator=(const ScopedCodegenOptions &) = delete;

private:
  CodegenOptions m_saved;
};

size_t CountSpirv(const std::vector<u32> &spirv, const std::string &needle) {
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
  std::string text;
  Require("CodegenTests", "SPIR-V disassembly", tools.Disassemble(spirv, &text),
          "failed to disassemble emitted SPIR-V");
  return CountText(text, needle);
}

// Compiles and dispatches a case, returning the first `dwords` words of its buffer.
std::vector<u32> RunCollect(VulkanHarness *vulkan, const TestCase &test,
                            size_t dwords, std::vector<u32> *spirv = nullptr) {
  auto compiled = CompileCase(test, vulkan->SubgroupSize());
  ValidateSpirv(test.name, compiled.spirv);
  if (spirv != nullptr) {
    *spirv = compiled.spirv;
  }
  const auto size = std::max<size_t>({test.initial.size(), dwords, 1u});
  auto buffer = vulkan->CreateStorageBuffer(test.name, test.initial, size);
  vulkan->Dispatch(test, compiled, buffer, nullptr, nullptr, nullptr, nullptr,
                   nullptr);
  auto actual = vulkan->ReadBuffer(test.name, buffer, dwords);
  vulkan->DestroyBuffer(&buffer);
  return actual;
}

void RequireEqualWords(const char *name, const char *stage,
                       const std::vector<u32> &expected,
                       const std::vector<u32> &actual, size_t first = 0) {
  Require(name, stage, expected.size() == actual.size(), "size mismatch");
  for (size_t i = first; i < expected.size(); i++) {
    if (expected[i] != actual[i]) {
      Fail(name, stage,
           "word " + std::to_string(i) + ": expected " + Hex(expected[i]) +
               ", got " + Hex(actual[i]));
    }
  }
}

// V_MOVRELS with M0 = BFE(s, 24, 3) * 5, loaded from memory so it is not a compile-time
// constant: the value set {0, 5, ..., 35} folds all other compares of the select chain.
void CheckMoveRelValueSet(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenMoveRelValueSet";
  using O = ShaderOpcode;
  constexpr u32 OutBase = 16;
  TestCase test;
  test.name = name;
  AppendVMovLiteral(&test.code, 0, 0xdeadbeefu);
  for (u32 reg = 10; reg <= 50; reg++) {
    AppendVMovLiteral(&test.code, reg, 0xa0000000u + reg);
  }
  for (u32 k = 0; k < 8; k++) {
    AppendSmemLoadOpcode(&test.code, 0x08, 33, k * 4u); // s_buffer_load_dword s33
    test.code.push_back(EncodeSopp(0x0c, 0));             // s_waitcnt 0
    test.code.push_back(EncodeSop2(0x27, 34, 33, 255));   // s_bfe_u32 s34, s33, lit
    test.code.push_back(0x00030018u);
    test.code.push_back(EncodeSop2(0x26, 34, 34, InlineU32(5))); // s_mul_i32 s34, 5
    test.code.push_back(EncodeSMovB32(124, 34));                 // s_mov_b32 m0, s34
    test.code.push_back(EncodeVop1(0x43, 60, Vgpr(12)));         // v_movrels v60, v12
    test.code.push_back(EncodeVop1(0x43, 61, Vgpr(45)));         // v_movrels v61, v45
    AppendStoreVgpr(&test.code, 60, OutBase + 2u * k);
    AppendStoreVgpr(&test.code, 61, OutBase + 2u * k + 1u);
  }
  AppendEnd(&test.code);
  test.initial.assign(OutBase + 16u, 0u);
  for (u32 k = 0; k < 8; k++) {
    test.initial[k] = (k << 24u) | (0x00f0f0f0u + k) | (k == 3u ? 0xf8000000u : 0u);
  }
  test.opcodes = {O::V_MOV_B32, O::S_BUFFER_LOAD_DWORD, O::S_WAITCNT,
                  O::S_BFE_U32, O::S_MUL_I32,           O::S_MOV_B32,
                  O::V_MOVRELS_B32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};

  // The highest named VGPR is v61, so offsets from v45 at or above 17 are out of range.
  const auto expected = [&](bool isa_out_of_range) {
    std::vector<u32> words(OutBase + 16u, 0u);
    for (u32 k = 0; k < 8; k++) {
      words[k] = test.initial[k];
      const u32 index = 5u * k;
      words[OutBase + 2u * k] = 0xa0000000u + 12u + index;
      const u32 reg = 45u + index;
      u32 value = 0;
      if (reg <= 50u) {
        value = 0xa0000000u + reg;
      } else if (reg == 60u) {
        value = words[OutBase + 2u * k]; // written by the first V_MOVRELS
      } else if (reg < 62u) {
        value = 0u; // never written
      } else {
        value = isa_out_of_range ? 0xdeadbeefu : 0xa0000000u + 45u;
      }
      words[OutBase + 2u * k + 1u] = value;
    }
    return words;
  };

  CodegenOptions fast = GetCodegenOptions();
  fast.movrel_range = true;
  CodegenOptions slow = fast;
  slow.movrel_range = false;
  std::vector<u32> fast_spirv;
  std::vector<u32> slow_spirv;
  std::vector<u32> fast_words;
  std::vector<u32> slow_words;
  {
    ScopedCodegenOptions scope(fast);
    fast_words = RunCollect(vulkan, test, OutBase + 16u, &fast_spirv);
  }
  {
    ScopedCodegenOptions scope(slow);
    slow_words = RunCollect(vulkan, test, OutBase + 16u, &slow_spirv);
  }
  RequireEqualWords(name, "value-set readback", expected(true), fast_words);
  RequireEqualWords(name, "legacy readback", expected(false), slow_words);
  const auto fast_selects = CountSpirv(fast_spirv, "OpSelect");
  const auto slow_selects = CountSpirv(slow_spirv, "OpSelect");
  Require(name, "select count", fast_selects * 3u < slow_selects,
          "value-set folding did not shrink the select chains: " +
              std::to_string(fast_selects) + " vs " +
              std::to_string(slow_selects));
  std::printf("[codegen] %-32s ok (OpSelect %zu -> %zu, words %zu -> %zu)\n", name,
              slow_selects, fast_selects, slow_spirv.size(), fast_spirv.size());
}

// V_MOVRELS with M0 = s << 2, s loaded from memory (a loop counter times 4 in Astro Bot's foliage
// vertex shaders): the value set is unknown, but the two low bits are zero. KYTY_MOVREL_KNOWN_ZEROS
// keeps every fourth link of each select chain, and KYTY_MOVREL_SWITCH branches on M0 / 16 into
// short chains; the results must equal the ISA reference in every combination, including index 0,
// out-of-range indices (VGPR0) and never-written registers.
void CheckMoveRelKnownZeros(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenMoveRelKnownZeros";
  using O = ShaderOpcode;
  constexpr u32 OutBase = 16;
  constexpr u32 Count = 8;
  constexpr u32 Limit = 102; // the highest named VGPR is v101
  constexpr u32 Bases[] = {12u, 45u};
  const u32 sources[Count] = {0u, 1u, 17u, 0x40u, 13u, 0x3fu, 5u, 0xffffffffu};
  TestCase test;
  test.name = name;
  AppendVMovLiteral(&test.code, 0, 0xdeadbeefu);
  for (u32 reg = 10; reg <= 90; reg++) {
    AppendVMovLiteral(&test.code, reg, 0xa0000000u + reg);
  }
  for (u32 k = 0; k < Count; k++) {
    AppendSmemLoadOpcode(&test.code, 0x08, 33, k * 4u); // s_buffer_load_dword s33
    test.code.push_back(EncodeSopp(0x0c, 0));             // s_waitcnt 0
    test.code.push_back(EncodeSop2(0x1e, 34, 33, InlineU32(2))); // s_lshl_b32 s34, s33, 2
    test.code.push_back(EncodeSMovB32(124, 34));                 // s_mov_b32 m0, s34
    test.code.push_back(EncodeVop1(0x43, 100, Vgpr(Bases[0])));  // v_movrels v100, v12
    test.code.push_back(EncodeVop1(0x43, 101, Vgpr(Bases[1])));  // v_movrels v101, v45
    AppendStoreVgpr(&test.code, 100, OutBase + 2u * k);
    AppendStoreVgpr(&test.code, 101, OutBase + 2u * k + 1u);
  }
  AppendEnd(&test.code);
  test.initial.assign(OutBase + 2u * Count, 0u);
  std::copy(std::begin(sources), std::end(sources), test.initial.begin());
  test.opcodes = {O::V_MOV_B32, O::S_BUFFER_LOAD_DWORD, O::S_WAITCNT,
                  O::S_LSHL_B32, O::S_MOV_B32, O::V_MOVRELS_B32, O::BUFFER_STORE_DWORD,
                  O::S_ENDPGM};

  // RDNA2 3.6.1: a source past the last VGPR reads VGPR0. v31 holds store addresses, but no
  // index reaches it (it is not base + a multiple of 4); v91-v99 are never written.
  std::vector<u32> expected(OutBase + 2u * Count, 0u);
  std::copy(std::begin(sources), std::end(sources), expected.begin());
  for (u32 k = 0; k < Count; k++) {
    for (u32 i = 0; i < 2u; i++) {
      const u32 index = (sources[k] << 2u) & 0xffu;
      const u32 reg = Bases[i] + index;
      u32 value = 0u;
      if (index >= Limit - Bases[i]) {
        value = 0xdeadbeefu;
      } else if (reg >= 10u && reg <= 90u) {
        Require(name, "reference", reg != 31u, "an index reads the address register");
        value = 0xa0000000u + reg;
      }
      expected[OutBase + 2u * k + i] = value;
    }
  }

  // [known_zeros][switch]: KYTY_MOVREL_KNOWN_ZEROS and KYTY_MOVREL_SWITCH. With known zeros the
  // v45 chain keeps 14 links (below the switch threshold of 16) and the v12 chain 22.
  CodegenOptions base = GetCodegenOptions();
  base.movrel_range = true;
  size_t selects[2][2] = {};
  size_t switches[2][2] = {};
  size_t words[2][2] = {};
  for (const bool known_zeros : {false, true}) {
    for (const bool use_switch : {false, true}) {
      CodegenOptions options = base;
      options.movrel_known_zeros = known_zeros;
      options.movrel_switch = use_switch;
      std::vector<u32> spirv;
      std::vector<u32> actual;
      {
        ScopedCodegenOptions scope(options);
        actual = RunCollect(vulkan, test, OutBase + 2u * Count, &spirv);
      }
      const std::string stage = std::string(known_zeros ? "known-zeros" : "value-set") +
                                (use_switch ? " switch" : " select") + " readback";
      RequireEqualWords(name, stage.c_str(), expected, actual);
      selects[known_zeros][use_switch] = CountSpirv(spirv, "OpSelect");
      switches[known_zeros][use_switch] = CountSpirv(spirv, "OpSwitch");
      words[known_zeros][use_switch] = spirv.size();
    }
  }
  Require(name, "select count", selects[1][0] * 2u < selects[0][0],
          "known zero bits did not shrink the select chains: " + std::to_string(selects[1][0]) +
              " vs " + std::to_string(selects[0][0]));
  for (const bool known_zeros : {false, true}) {
    Require(name, "switch count", switches[known_zeros][1] >= switches[known_zeros][0] + Count,
            "KYTY_MOVREL_SWITCH did not turn the long chains into switches");
  }
  std::printf("[codegen] %-32s ok (OpSelect %zu -> %zu, words %zu -> %zu; switch: OpSwitch +%zu/+%zu, "
              "words %zu/%zu)\n",
              name, selects[0][0], selects[1][0], words[0][0], words[1][0],
              switches[0][1] - switches[0][0], switches[1][1] - switches[1][0], words[0][1],
              words[1][1]);
}

// KYTY_UNIFORM_LANE_READS: V_READFIRSTLANE and V_READLANE results go through
// OpGroupNonUniformBroadcastFirst, which must not change them (wave32: one guest lane per host
// invocation; wave64: two per invocation on 32-lane hosts).
void CheckUniformLaneReads(VulkanHarness *vulkan) {
  for (const u32 wave_size : {32u, 64u}) {
  const std::string case_name = "CodegenUniformLaneReads/wave" + std::to_string(wave_size);
  const char *name = case_name.c_str();
  using O = ShaderOpcode;
  constexpr u32 OutBase = 8;
  TestCase test;
  test.name = name;
  test.compute_info.threads_num[0] = wave_size;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = wave_size;
  test.has_compute_info = true;
  test.code.push_back(EncodeVop2(0x1a, 1, InlineU32(1), 0));        // v_lshlrev_b32 v1, 1, v0
  test.code.push_back(EncodeVop2(0x25, 1, InlineU32(7), 1));        // v_add_nc_u32 v1, 7, v1
  test.code.push_back(EncodeVop1(0x02, 33, Vgpr(1)));               // v_readfirstlane_b32 s33, v1
  AppendVop3(&test.code, 0x360, 34, Vgpr(1), InlineU32(5));         // v_readlane_b32 s34, v1, 5
  test.code.push_back(EncodeSop2(0x00, 35, 33, 34));                // s_add_u32 s35, s33, s34
  AppendStoreSgpr(&test.code, 33, OutBase);
  AppendStoreSgpr(&test.code, 34, OutBase + 1u);
  AppendStoreSgpr(&test.code, 35, OutBase + 2u);
  AppendEnd(&test.code);
  test.initial.assign(OutBase + 3u, 0u);
  test.opcodes = {O::V_LSHLREV_B32,  O::V_ADD_NC_U32, O::V_READFIRSTLANE_B32,
                  O::V_READLANE_B32, O::S_ADD_U32,    O::V_MOV_B32,
                  O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  std::vector<u32> expected(OutBase + 3u, 0u);
  expected[OutBase] = 7u;       // lane 0: 2 * 0 + 7
  expected[OutBase + 1u] = 17u; // lane 5: 2 * 5 + 7
  expected[OutBase + 2u] = 24u;

  size_t broadcasts[2] = {};
  for (const bool uniform : {false, true}) {
    CodegenOptions options = GetCodegenOptions();
    options.uniform_lane_reads = uniform;
    std::vector<u32> spirv;
    std::vector<u32> actual;
    {
      ScopedCodegenOptions scope(options);
      actual = RunCollect(vulkan, test, OutBase + 3u, &spirv);
    }
    RequireEqualWords(name, uniform ? "uniform readback" : "shuffle readback", expected, actual);
    broadcasts[uniform ? 1 : 0] = CountSpirv(spirv, "OpGroupNonUniformBroadcastFirst");
  }
  Require(name, "broadcast count", broadcasts[0] == 0u && broadcasts[1] >= 2u,
          "KYTY_UNIFORM_LANE_READS did not mark both lane reads uniform: " +
              std::to_string(broadcasts[0]) + " -> " + std::to_string(broadcasts[1]));
  std::printf("[codegen] %-32s ok (OpGroupNonUniformBroadcastFirst %zu -> %zu)\n", name,
              broadcasts[0], broadcasts[1]);
  }
}

// A 64-thread workgroup that loops over `count` elements (a multiple of 64). Input i of element
// e is at dword i * count + e; output o at (inputs + o) * count + e. The body reads inputs from
// v2.. and writes outputs to v10.. (v1 holds the lane byte offset, s40 the element byte offset).
struct BulkCase {
  TestCase test;
  u32 inputs = 0;
  u32 outputs = 0;
  u32 count = 0;
};

BulkCase MakeBulkCase(const char *name, u32 inputs, u32 outputs,
                      const std::vector<std::vector<u32>> &input_values,
                      const std::function<void(std::vector<u32> *)> &body,
                      std::vector<ShaderOpcode> opcodes) {
  BulkCase bulk;
  bulk.inputs = inputs;
  bulk.outputs = outputs;
  bulk.count = static_cast<u32>(input_values.front().size());
  Require(name, "bulk layout", bulk.count % 64u == 0u && input_values.size() == inputs,
          "bulk inputs must be one vector per input, a multiple of 64 long");
  auto &test = bulk.test;
  auto &code = test.code;
  test.name = name;
  code.push_back(EncodeVop2(0x1a, 1, InlineU32(2), 0)); // v_lshlrev_b32 v1, 2, v0
  code.push_back(EncodeSMovB32(40, InlineU32(0)));      // s_mov_b32 s40, 0
  const auto loop = static_cast<u32>(code.size());
  for (u32 i = 0; i < inputs; i++) {
    code.push_back(EncodeSop2(0x00, 41 + i, 40, 255)); // s_add_u32 s(41+i), s40, lit
    code.push_back(i * bulk.count * 4u);
    code.push_back(EncodeMubuf0(0x0c)); // buffer_load_dword v(2+i), v1, s[0:3], s(41+i)
    code.push_back(EncodeMubuf1(2 + i, 0, 1, 41 + i));
  }
  code.push_back(EncodeSopp(0x0c, 0)); // s_waitcnt 0
  body(&code);
  for (u32 o = 0; o < outputs; o++) {
    code.push_back(EncodeSop2(0x00, 50 + o, 40, 255)); // s_add_u32 s(50+o), s40, lit
    code.push_back((inputs + o) * bulk.count * 4u);
    code.push_back(EncodeMubuf0(0x1c)); // buffer_store_dword v(10+o), v1, s[0:3], s(50+o)
    code.push_back(EncodeMubuf1(10 + o, 0, 1, 50 + o));
  }
  code.push_back(EncodeSop2(0x00, 40, 40, 255)); // s_add_u32 s40, s40, 256
  code.push_back(256u);
  code.push_back(EncodeSopc(0x0a, 40, 255)); // s_cmp_lt_u32 s40, count * 4
  code.push_back(bulk.count * 4u);
  const auto branch = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size()) - 1;
  code.push_back(EncodeSopp(0x05, static_cast<u32>(branch) & 0xffffu)); // s_cbranch_scc1
  AppendEnd(&code);
  test.initial.assign(static_cast<size_t>(inputs + outputs) * bulk.count, 0u);
  for (u32 i = 0; i < inputs; i++) {
    std::copy(input_values[i].begin(), input_values[i].end(),
              test.initial.begin() + static_cast<size_t>(i) * bulk.count);
  }
  test.compute_info.threads_num[0] = 64;
  test.compute_info.threads_num[1] = 1;
  test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = 64;
  test.has_compute_info = true;
  opcodes.insert(opcodes.end(),
                 {ShaderOpcode::V_LSHLREV_B32, ShaderOpcode::S_MOV_B32,
                  ShaderOpcode::S_ADD_U32, ShaderOpcode::BUFFER_LOAD_DWORD,
                  ShaderOpcode::S_WAITCNT, ShaderOpcode::BUFFER_STORE_DWORD,
                  ShaderOpcode::S_CMP_LT_U32, ShaderOpcode::S_CBRANCH_SCC1,
                  ShaderOpcode::S_ENDPGM});
  test.opcodes = std::move(opcodes);
  return bulk;
}

// Output o of every element.
std::vector<u32> BulkOutput(const BulkCase &bulk, const std::vector<u32> &words, u32 output) {
  const auto begin = words.begin() + static_cast<size_t>(bulk.inputs + output) * bulk.count;
  return {begin, begin + bulk.count};
}

struct BulkRun {
  std::vector<u32> words;
  std::vector<u32> spirv;
};

BulkRun RunBulk(VulkanHarness *vulkan, const BulkCase &bulk, const CodegenOptions &options) {
  ScopedCodegenOptions scope(options);
  BulkRun run;
  run.words = RunCollect(vulkan, bulk.test, bulk.test.initial.size(), &run.spirv);
  return run;
}

// Compares every output of the new and the legacy code generation with each other and with an
// optional CPU reference (reference(output, element) -> expected bits). Elements for which\n// skip returns true are not compared; the number skipped is returned.
size_t CompareBulk(const char *name, const BulkCase &bulk, const BulkRun &fast,
                   const BulkRun &slow, const std::function<u32(u32, u32)> &reference,
                   const char *const *output_names,
                   const std::function<bool(u32, u32)> &skip = {}) {
  size_t skipped = 0;
  for (u32 o = 0; o < bulk.outputs; o++) {
    const auto fast_out = BulkOutput(bulk, fast.words, o);
    const auto slow_out = BulkOutput(bulk, slow.words, o);
    for (u32 e = 0; e < bulk.count; e++) {
      const auto input = [&](u32 i) {
        return i < bulk.inputs ? bulk.test.initial[static_cast<size_t>(i) * bulk.count + e] : 0u;
      };
      if (skip && skip(o, e)) {
        skipped++;
        continue;
      }
      const auto describe = [&]() {
        return std::string(output_names[o]) + " element " + std::to_string(e) + " inputs " +
               Hex(input(0)) + " " + Hex(input(1)) + " " + Hex(input(2));
      };
      if (fast_out[e] != slow_out[e]) {
        Fail(name, "new vs legacy",
             describe() + ": new " + Hex(fast_out[e]) + ", legacy " + Hex(slow_out[e]));
      }
      if (reference) {
        const auto expected = reference(o, e);
        if (fast_out[e] != expected) {
          Fail(name, "CPU reference",
               describe() + ": got " + Hex(fast_out[e]) + ", expected " + Hex(expected));
        }
      }
    }
  }
  return skipped;
}

// f32 bit patterns covering zeros, denormals, normals, infinities and NaNs with payloads.
const std::vector<u32> &SpecialF32() {
  static const std::vector<u32> values = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x007fffffu, 0x807fffffu,
      0x00400000u, 0x80400000u, 0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u,
      0x3fc00000u, 0xbfc00000u, 0x3f000000u, 0xbf000000u, 0x40000000u, 0xc0000000u,
      0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u,
      0x7f800001u, 0xff800001u, 0x7fa00000u, 0x7fffffffu, 0xffffffffu, 0x7fc12345u,
      0x3eaaaaabu, 0x42c80000u,
  };
  return values;
}

uint32_t NextRandom(uint64_t &state) {
  state = state * 6364136223846793005ull + 1442695040888963407ull;
  return static_cast<uint32_t>(state >> 32u);
}

// Mostly random bit patterns, a quarter drawn from the special set.
u32 RandomF32(uint64_t &state) {
  const auto r = NextRandom(state);
  if ((r & 3u) == 0u) {
    return SpecialF32()[NextRandom(state) % SpecialF32().size()];
  }
  return NextRandom(state);
}

// CPU model of the legacy f32 min/max emulation (EmitMinMaxF32Value). The NVIDIA driver flushes
// f32 denormals in float compares by default (the module declares no f32 denorm mode there), so
// the ordered compare sees denormals as zeros while the selected bits stay raw.
float FlushedF32(u32 bits) {
  return (bits & 0x7f800000u) == 0u ? std::bit_cast<float>(bits & 0x80000000u)
                                    : std::bit_cast<float>(bits);
}

u32 RefMinMaxF32(u32 lhs, u32 rhs, bool max_value) {
  const auto nan = [](u32 bits) { return (bits & 0x7fffffffu) > 0x7f800000u; };
  const auto zero = [](u32 bits) { return (bits & 0x7fffffffu) == 0u; };
  if (nan(lhs)) {
    return rhs;
  }
  if (nan(rhs)) {
    return lhs;
  }
  if (zero(lhs) && zero(rhs)) {
    return max_value ? (lhs & rhs) : (lhs | rhs);
  }
  const auto a = FlushedF32(lhs);
  const auto b = FlushedF32(rhs);
  return max_value ? (a >= b ? lhs : rhs) : (a < b ? lhs : rhs);
}

u32 RefMedTriF32(u32 a, u32 b, u32 c) {
  const auto nan = [](u32 bits) { return (bits & 0x7fffffffu) > 0x7f800000u; };
  const auto min_ab = RefMinMaxF32(a, b, false);
  const auto min3 = RefMinMaxF32(min_ab, c, false);
  const auto max_ab = RefMinMaxF32(a, b, true);
  const auto high_min = RefMinMaxF32(max_ab, c, false);
  const auto median = RefMinMaxF32(min_ab, high_min, true);
  return nan(a) || nan(b) || nan(c) ? min3 : median;
}

void CheckFloatMinMax(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenFloatMinMaxMed3";
  using O = ShaderOpcode;
  // Every triple of special values, then random triples.
  const auto &special = SpecialF32();
  std::vector<std::vector<u32>> inputs(3);
  for (const auto a : special) {
    for (const auto b : special) {
      for (const auto c : special) {
        inputs[0].push_back(a);
        inputs[1].push_back(b);
        inputs[2].push_back(c);
      }
    }
  }
  uint64_t seed = 0x1234567890abcdefull;
  while (inputs[0].size() < 65536u) {
    for (auto &input : inputs) {
      input.push_back(RandomF32(seed));
    }
  }
  const auto bulk = MakeBulkCase(
      name, 3, 7, inputs,
      [](std::vector<u32> *code) {
        code->push_back(EncodeVop2(0x0f, 10, Vgpr(2), 3)); // v_min_f32 v10, v2, v3
        code->push_back(EncodeVop2(0x10, 11, Vgpr(2), 3)); // v_max_f32 v11, v2, v3
        AppendVop3(code, 0x151, 12, Vgpr(2), Vgpr(3), Vgpr(4)); // v_min3_f32
        AppendVop3(code, 0x154, 13, Vgpr(2), Vgpr(3), Vgpr(4)); // v_max3_f32
        AppendVop3(code, 0x157, 14, Vgpr(2), Vgpr(3), Vgpr(4)); // v_med3_f32
        AppendVop3(code, 0x157, 15, Vgpr(4), Vgpr(2), Vgpr(3)); // v_med3_f32 (rotated)
        code->push_back(EncodeVop2(0x0f, 16, Vgpr(3), 2)); // v_min_f32 v16, v3, v2
      },
      {O::V_MIN_F32, O::V_MAX_F32, O::V_MIN3_F32, O::V_MAX3_F32, O::V_MED3_F32});
  CodegenOptions fast = GetCodegenOptions();
  fast.fast_float_min_max = true;
  CodegenOptions slow = fast;
  slow.fast_float_min_max = false;
  const auto fast_run = RunBulk(vulkan, bulk, fast);
  const auto slow_run = RunBulk(vulkan, bulk, slow);
  const auto in = [&](u32 i, u32 e) { return inputs[i][e]; };
  static const char *const names[] = {"min", "max", "min3", "max3", "med3", "med3(c,a,b)",
                                      "min(b,a)"};
  CompareBulk(name, bulk, fast_run, slow_run,
              [&](u32 o, u32 e) -> u32 {
                const auto a = in(0, e), b = in(1, e), c = in(2, e);
                switch (o) {
                case 0: return RefMinMaxF32(a, b, false);
                case 1: return RefMinMaxF32(a, b, true);
                case 2: return RefMinMaxF32(RefMinMaxF32(a, b, false), c, false);
                case 3: return RefMinMaxF32(RefMinMaxF32(a, b, true), c, true);
                case 4: return RefMedTriF32(a, b, c);
                case 5: return RefMedTriF32(c, a, b);
                default: return RefMinMaxF32(b, a, false);
                }
              },
              names);
  std::printf("[codegen] %-32s ok (%u elements, SPIR-V %zu -> %zu words, OpSelect %zu -> %zu)\n",
              name, bulk.count, slow_run.spirv.size(), fast_run.spirv.size(),
              CountSpirv(slow_run.spirv, "OpSelect"), CountSpirv(fast_run.spirv, "OpSelect"));
}

// KYTY_SHORT_F32_HELPERS: NaN tests by OpIsNan (float-to-int conversions, the legacy min/max) must
// give the same bits as the integer tests they replace. sin/cos stay in the set: their large-value
// select is kept on purpose (the RTX 3090's Fract of -FLT_MAX is not +0), so they must not change.
void CheckShortF32Helpers(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenShortF32Helpers";
  using O = ShaderOpcode;
  auto special = SpecialF32();
  // Whole numbers around 2^23 and 2^24 of both signs, and a value just below 2^23.
  for (const u32 bits : {0x4b000000u, 0xcb000000u, 0x4b000001u, 0xcb000001u, 0x4affffffu,
                         0xcaffffffu, 0x4b800000u, 0xcb800000u, 0x4f000000u, 0xdf000000u}) {
    special.push_back(bits);
  }
  std::vector<std::vector<u32>> inputs(2);
  for (const auto a : special) {
    for (const auto b : special) {
      inputs[0].push_back(a);
      inputs[1].push_back(b);
    }
  }
  uint64_t seed = 0x5eed5eed12345678ull;
  while (inputs[0].size() < 65536u) {
    for (auto &input : inputs) {
      input.push_back(RandomF32(seed));
    }
  }
  const auto bulk = MakeBulkCase(
      name, 2, 6, inputs,
      [](std::vector<u32> *code) {
        code->push_back(EncodeVop1(0x08, 10, Vgpr(2)));    // v_cvt_i32_f32 v10, v2
        code->push_back(EncodeVop1(0x07, 11, Vgpr(2)));    // v_cvt_u32_f32 v11, v2
        code->push_back(EncodeVop1(0x35, 12, Vgpr(2)));    // v_sin_f32 v12, v2
        code->push_back(EncodeVop1(0x36, 13, Vgpr(2)));    // v_cos_f32 v13, v2
        code->push_back(EncodeVop2(0x0f, 14, Vgpr(2), 3)); // v_min_f32 v14, v2, v3
        code->push_back(EncodeVop2(0x10, 15, Vgpr(2), 3)); // v_max_f32 v15, v2, v3
      },
      {O::V_CVT_I32_F32, O::V_CVT_U32_F32, O::V_SIN_F32, O::V_COS_F32, O::V_MIN_F32,
       O::V_MAX_F32});
  // The legacy min/max classifies both operands; the fast one uses OpIsNan anyway.
  CodegenOptions legacy = GetCodegenOptions();
  legacy.fast_float_min_max = false;
  legacy.short_f32_helpers = false;
  CodegenOptions shorter = legacy;
  shorter.short_f32_helpers = true;
  const auto short_run = RunBulk(vulkan, bulk, shorter);
  const auto legacy_run = RunBulk(vulkan, bulk, legacy);
  static const char *const names[] = {"cvt_i32", "cvt_u32", "sin", "cos", "min", "max"};
  // cvt/sin/cos: bit-identical to the legacy sequences (checked by CompareBulk); min/max: also
  // the CPU model of the legacy emulation.
  CompareBulk(name, bulk, short_run, legacy_run,
              [&](u32 o, u32 e) -> u32 {
                if (o == 4u || o == 5u) {
                  return RefMinMaxF32(inputs[0][e], inputs[1][e], o == 5u);
                }
                return legacy_run.words[static_cast<size_t>(bulk.inputs + o) * bulk.count + e];
              },
              names);
  const auto nan_tests = [](const std::vector<u32> &spirv) { return CountSpirv(spirv, "OpIsNan"); };
  Require(name, "OpIsNan count", nan_tests(short_run.spirv) > nan_tests(legacy_run.spirv),
          "KYTY_SHORT_F32_HELPERS did not classify NaNs with OpIsNan");
  std::printf("[codegen] %-32s ok (%u elements, SPIR-V %zu -> %zu words, OpIsNan %zu -> %zu)\n",
              name, bulk.count, legacy_run.spirv.size(), short_run.spirv.size(),
              nan_tests(legacy_run.spirv), nan_tests(short_run.spirv));
}

// KYTY_SHORT_F32_HELPERS with NaN inputs through every path that takes its NaN test: the legacy
// min3/max3/med3, the emitter's float-to-int layer (KYTY_SINGLE_F2I_SATURATION=0) and buffer float
// atomics (the compare-and-swap replacement). Every triple of special values (quiet, signaling and
// negative NaNs included); the min/max results must also equal the CPU model, the atomics their
// fixed expected memory and return values.
void CheckShortF32HelpersNanPaths(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenShortF32HelpersNan";
  using O = ShaderOpcode;
  const auto &special = SpecialF32();
  std::vector<std::vector<u32>> inputs(3);
  for (const auto a : special) {
    for (const auto b : special) {
      for (const auto c : special) {
        inputs[0].push_back(a);
        inputs[1].push_back(b);
        inputs[2].push_back(c);
      }
    }
  }
  const auto bulk = MakeBulkCase(
      name, 3, 6, inputs,
      [](std::vector<u32> *code) {
        AppendVop3(code, 0x151, 10, Vgpr(2), Vgpr(3), Vgpr(4)); // v_min3_f32
        AppendVop3(code, 0x154, 11, Vgpr(2), Vgpr(3), Vgpr(4)); // v_max3_f32
        AppendVop3(code, 0x157, 12, Vgpr(2), Vgpr(3), Vgpr(4)); // v_med3_f32
        AppendVop3(code, 0x157, 13, Vgpr(4), Vgpr(2), Vgpr(3)); // v_med3_f32 (rotated)
        code->push_back(EncodeVop1(0x08, 14, Vgpr(2)));         // v_cvt_i32_f32
        code->push_back(EncodeVop1(0x07, 15, Vgpr(3)));         // v_cvt_u32_f32
      },
      {O::V_MIN3_F32, O::V_MAX3_F32, O::V_MED3_F32, O::V_CVT_I32_F32, O::V_CVT_U32_F32});
  CodegenOptions legacy = GetCodegenOptions();
  legacy.fast_float_min_max = false;
  legacy.single_f2i_saturation = false;
  legacy.short_f32_helpers = false;
  CodegenOptions shorter = legacy;
  shorter.short_f32_helpers = true;
  const auto short_run = RunBulk(vulkan, bulk, shorter);
  const auto legacy_run = RunBulk(vulkan, bulk, legacy);
  const auto nan = [](u32 bits) { return (bits & 0x7fffffffu) > 0x7f800000u; };
  const auto cvt = [&](u32 bits, bool is_signed) -> u32 {
    if (nan(bits)) {
      return 0u;
    }
    const double f = std::bit_cast<float>(bits);
    if (is_signed) {
      return f <= -2147483648.0  ? 0x80000000u
             : f >= 2147483648.0 ? 0x7fffffffu
                                 : static_cast<u32>(static_cast<int32_t>(std::trunc(f)));
    }
    return f <= 0.0 ? 0u : f >= 4294967296.0 ? 0xffffffffu : static_cast<u32>(std::trunc(f));
  };
  static const char *const names[] = {"min3", "max3", "med3", "med3(c,a,b)", "cvt_i32",
                                      "cvt_u32"};
  CompareBulk(name, bulk, short_run, legacy_run,
              [&](u32 o, u32 e) -> u32 {
                const auto a = inputs[0][e], b = inputs[1][e], c = inputs[2][e];
                switch (o) {
                case 0: return RefMinMaxF32(RefMinMaxF32(a, b, false), c, false);
                case 1: return RefMinMaxF32(RefMinMaxF32(a, b, true), c, true);
                case 2: return RefMedTriF32(a, b, c);
                case 3: return RefMedTriF32(c, a, b);
                case 4: return cvt(a, true);
                default: return cvt(b, false);
                }
              },
              names);
  const auto nan_tests = [](const std::vector<u32> &spirv) { return CountSpirv(spirv, "OpIsNan"); };
  Require(name, "OpIsNan count (ALU)", nan_tests(short_run.spirv) > nan_tests(legacy_run.spirv),
          "KYTY_SHORT_F32_HELPERS did not classify the min/max/med3/cvt NaNs with OpIsNan");

  // Buffer float atomics: old and incoming NaNs (quiet and signaling), infinities, signed zeros.
  size_t atomic_short_nan = 0;
  size_t atomic_legacy_nan = 0;
  for (const auto &test : {BufferAtomicFMinSpecialValues(), BufferAtomicFMaxSpecialValues()}) {
    for (const bool use_short : {true, false}) {
      CodegenOptions options = GetCodegenOptions();
      options.short_f32_helpers = use_short;
      ScopedCodegenOptions scope(options);
      std::vector<u32> spirv;
      const auto words = RunCollect(vulkan, test, test.expected.size(), &spirv);
      RequireEqualWords(name, use_short ? "atomic readback (short)" : "atomic readback (legacy)",
                        test.expected, words);
      (use_short ? atomic_short_nan : atomic_legacy_nan) += nan_tests(spirv);
    }
  }
  Require(name, "OpIsNan count (atomics)", atomic_short_nan > atomic_legacy_nan,
          "KYTY_SHORT_F32_HELPERS did not classify the atomic NaNs with OpIsNan");
  std::printf("[codegen] %-32s ok (%u triples, OpIsNan %zu -> %zu; atomics OpIsNan %zu -> %zu)\n",
              name, bulk.count, nan_tests(legacy_run.spirv), nan_tests(short_run.spirv),
              atomic_legacy_nan, atomic_short_nan);
}

// med3/min/max with constant operands (the clamp idiom): x from the special set and random bits.
void CheckFloatClampConstants(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenFloatClampConstants";
  using O = ShaderOpcode;
  constexpr u32 Zero = 128, One = 242, MinusOne = 243, Half = 240, MinusHalf = 241, Four = 246,
                Literal = 255;
  std::vector<std::vector<u32>> inputs(1);
  inputs[0] = SpecialF32();
  uint64_t seed = 0xfeedfacecafebeefull;
  while (inputs[0].size() < 4096u) {
    inputs[0].push_back(RandomF32(seed));
  }
  struct Op {
    u32 opcode;
    u32 src0, src1, src2;
    u32 literal;
  };
  const std::vector<Op> ops = {
      {0x157, Vgpr(2), Zero, One, 0},        {0x157, Zero, Vgpr(2), One, 0},
      {0x157, One, Zero, Vgpr(2), 0},        {0x157, Vgpr(2), MinusOne, Zero, 0},
      {0x157, MinusOne, Vgpr(2), Zero, 0},   {0x157, Zero, MinusOne, Vgpr(2), 0},
      {0x157, Vgpr(2), Literal, One, 0x80000000u},
      {0x157, Vgpr(2), MinusOne, Literal, 0x80000000u},
      {0x157, Vgpr(2), Half, MinusHalf, 0},  {0x157, Four, Vgpr(2), Four, 0},
      {0x157, Vgpr(2), Literal, One, 0x7fc00000u},
      {0x10f, Vgpr(2), Zero, 0, 0},          {0x110, Vgpr(2), Zero, 0, 0},
      {0x110, One, Vgpr(2), 0, 0},           {0x10f, Vgpr(2), Literal, 0, 0x7fd00001u},
      {0x151, Vgpr(2), Zero, One, 0},        {0x154, MinusOne, Vgpr(2), Literal, 0x80000000u},
  };
  const auto bulk = MakeBulkCase(
      name, 1, static_cast<u32>(ops.size()), inputs,
      [&](std::vector<u32> *code) {
        for (u32 i = 0; i < ops.size(); i++) {
          AppendVop3(code, ops[i].opcode, 10 + i, ops[i].src0, ops[i].src1, ops[i].src2);
          if (ops[i].src0 == Literal || ops[i].src1 == Literal || ops[i].src2 == Literal) {
            code->push_back(ops[i].literal);
          }
        }
      },
      {O::V_MIN_F32, O::V_MAX_F32, O::V_MIN3_F32, O::V_MAX3_F32, O::V_MED3_F32});
  const auto operand = [&](const Op &op, u32 src, u32 x) -> u32 {
    switch (src) {
    case Zero: return 0x00000000u;
    case One: return 0x3f800000u;
    case MinusOne: return 0xbf800000u;
    case Half: return 0x3f000000u;
    case MinusHalf: return 0xbf000000u;
    case Four: return 0x40800000u;
    case Literal: return op.literal;
    default: return x;
    }
  };
  CodegenOptions fast = GetCodegenOptions();
  fast.fast_float_min_max = true;
  CodegenOptions slow = fast;
  slow.fast_float_min_max = false;
  const auto fast_run = RunBulk(vulkan, bulk, fast);
  const auto slow_run = RunBulk(vulkan, bulk, slow);
  std::vector<std::string> labels;
  std::vector<const char *> names;
  for (u32 i = 0; i < ops.size(); i++) {
    labels.push_back("op" + std::to_string(i));
  }
  for (const auto &label : labels) {
    names.push_back(label.c_str());
  }
  // A denormal x compared with a constant is where the NVIDIA compiler is inconsistent: it
  // flushes denormals in register compares but may lower a compare against an immediate to an
  // integer test that does not flush, per instance. The legacy code itself then returns
  // different bits for the same values depending on whether an operand is a constant (for
  // example med3(0, 0x00000001, 1.0) gives 0x00000001 with registers and 0 with constants), so
  // those elements have no single "current result" to match and are not compared.
  const auto denormal_x = [&](u32, u32 e) {
    const auto x = inputs[0][e];
    return (x & 0x7f800000u) == 0u && (x & 0x007fffffu) != 0u;
  };
  const auto skipped = CompareBulk(
      name, bulk, fast_run, slow_run,
      [&](u32 o, u32 e) -> u32 {
        const auto &op = ops[o];
        const auto x = inputs[0][e];
        const auto a = operand(op, op.src0, x), b = operand(op, op.src1, x),
                   c = operand(op, op.src2, x);
        switch (op.opcode) {
        case 0x10f: return RefMinMaxF32(a, b, false);
        case 0x110: return RefMinMaxF32(a, b, true);
        case 0x151: return RefMinMaxF32(RefMinMaxF32(a, b, false), c, false);
        case 0x154: return RefMinMaxF32(RefMinMaxF32(a, b, true), c, true);
        default: return RefMedTriF32(a, b, c);
        }
      },
      names.data(), denormal_x);
  std::printf("[codegen] %-32s ok (%u elements x %zu ops, %zu denormal skipped, SPIR-V %zu -> "
              "%zu words)\n",
              name, bulk.count, ops.size(), skipped, slow_run.spirv.size(),
              fast_run.spirv.size());
}

// CPU port of EmitF32ToF16RtzBits (the legacy V_CVT_PKRTZ_F16_F32 half conversion).
u32 RefF32ToF16Rtz(u32 bits) {
  const u32 sign = (bits >> 16u) & 0x8000u;
  const u32 exp = (bits >> 23u) & 0xffu;
  const u32 mant = bits & 0x007fffffu;
  if (exp == 255u) {
    return mant == 0u ? (sign | 0x7c00u) : (sign | 0x7c00u | (mant >> 13u) | 0x0200u);
  }
  if (exp >= 143u) {
    return sign | 0x7bffu;
  }
  if (exp < 103u) {
    return sign;
  }
  if (exp <= 112u) {
    const u32 shift = exp > 112u ? 14u : 126u - exp;
    return sign | ((mant | 0x00800000u) >> shift);
  }
  return (sign | ((exp - 112u) << 10u) | (mant >> 13u)) & 0xffffu;
}

// CPU model of EmitFastF32ToF16RtzBits, op for op.
u32 FastF32ToF16Rtz(u32 bits) {
  const u32 abs = bits & 0x7fffffffu;
  const u32 sign = (bits >> 16u) & 0x8000u;
  const u32 high13 = abs >> 13u;
  const u32 normal = std::min(high13 - (112u << 10u), 0x7bffu);
  const u32 exponent = abs >> 23u;
  const u32 shift = std::min(126u - exponent, 31u);
  const u32 subnormal = ((abs & 0x007fffffu) | 0x00800000u) >> shift;
  u32 result = abs < 0x38800000u ? subnormal : normal;
  const u32 special = (high13 & 0x7fffu) | (abs > 0x7f800000u ? 0x0200u : 0u);
  result = abs >= 0x7f800000u ? special : result;
  return result | sign;
}

// The integer formulation against the legacy one for all 2^32 inputs (CPU, a few seconds).
void CheckPackRtzExhaustive() {
  constexpr const char *name = "CodegenCvtPkrtzExhaustive";
  constexpr u32 Threads = 8;
  std::array<std::atomic<uint64_t>, Threads> mismatch{};
  std::vector<std::thread> workers;
  for (u32 t = 0; t < Threads; t++) {
    workers.emplace_back([t, &mismatch] {
      const uint64_t begin = (uint64_t{1} << 32u) / Threads * t;
      const uint64_t end = (uint64_t{1} << 32u) / Threads * (t + 1u);
      uint64_t first = UINT64_MAX;
      for (uint64_t value = begin; value < end; value++) {
        const auto bits = static_cast<u32>(value);
        if (FastF32ToF16Rtz(bits) != RefF32ToF16Rtz(bits)) {
          first = value;
          break;
        }
      }
      mismatch[t] = first;
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  for (const auto &value : mismatch) {
    if (value != UINT64_MAX) {
      const auto bits = static_cast<u32>(value.load());
      Fail(name, "exhaustive", "input " + Hex(bits) + ": fast " + Hex(FastF32ToF16Rtz(bits)) +
                                   ", legacy " + Hex(RefF32ToF16Rtz(bits)));
    }
  }
  std::printf("[codegen] %-32s ok (2^32 inputs)\n", name);
}

void CheckPackRtz(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenCvtPkrtzF16F32";
  using O = ShaderOpcode;
  std::vector<std::vector<u32>> inputs(2);
  // Every exponent and sign with mantissas around the f16 rounding points, then random bits.
  const u32 mantissas[] = {0x000000u, 0x000001u, 0x000fffu, 0x001000u, 0x001001u, 0x001fffu,
                           0x002000u, 0x002001u, 0x3fe000u, 0x3fffffu, 0x400000u, 0x7fe000u,
                           0x7fefffu, 0x7ff000u, 0x7fffffu, 0x555555u};
  uint64_t seed = 0x0badc0ffee123457ull;
  for (u32 sign = 0; sign < 2u; sign++) {
    for (u32 exp = 0; exp < 256u; exp++) {
      for (const auto mant : mantissas) {
        inputs[0].push_back((sign << 31u) | (exp << 23u) | mant);
        inputs[1].push_back(RandomF32(seed));
      }
    }
  }
  for (const auto value : SpecialF32()) {
    inputs[0].push_back(value);
    inputs[1].push_back(value ^ 0x80000000u);
  }
  while (inputs[0].size() < (1u << 19u)) {
    inputs[0].push_back(RandomF32(seed));
    inputs[1].push_back(NextRandom(seed));
  }
  inputs[0].resize(inputs[0].size() / 64u * 64u);
  inputs[1].resize(inputs[0].size());
  const auto bulk = MakeBulkCase(
      name, 2, 2, inputs,
      [](std::vector<u32> *code) {
        code->push_back(EncodeVop2(0x2f, 10, Vgpr(2), 3)); // v_cvt_pkrtz_f16_f32 v10, v2, v3
        code->push_back(EncodeVop2(0x2f, 11, Vgpr(3), 2)); // v_cvt_pkrtz_f16_f32 v11, v3, v2
      },
      {O::V_CVT_PKRTZ_F16_F32});
  CodegenOptions fast = GetCodegenOptions();
  fast.fast_pkrtz = true;
  CodegenOptions slow = fast;
  slow.fast_pkrtz = false;
  const auto fast_run = RunBulk(vulkan, bulk, fast);
  const auto slow_run = RunBulk(vulkan, bulk, slow);
  static const char *const names[] = {"pkrtz(x,y)", "pkrtz(y,x)"};
  CompareBulk(name, bulk, fast_run, slow_run,
              [&](u32 o, u32 e) -> u32 {
                const auto lo = inputs[o][e];
                const auto hi = inputs[1u - o][e];
                return RefF32ToF16Rtz(lo) | (RefF32ToF16Rtz(hi) << 16u);
              },
              names);
  std::printf("[codegen] %-32s ok (%u pairs, SPIR-V %zu -> %zu words)\n", name, bulk.count,
              slow_run.spirv.size(), fast_run.spirv.size());
}

// Float-to-int conversions with the emitter's second saturation layer on and off.
void CheckFloatToIntSaturation(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenFloatToIntSaturation";
  using O = ShaderOpcode;
  std::vector<std::vector<u32>> inputs(1);
  inputs[0] = SpecialF32();
  // Around the integer range limits and the rounding points.
  const float edges[] = {2147483520.0f, 2147483648.0f, 4294967040.0f, 4294967296.0f,
                         -2147483648.0f, -2147483904.0f, 65535.0f,    65536.0f,
                         32767.0f,       32768.0f,       -32768.0f,   -32769.0f,
                         255.0f,         255.5f,         256.0f,      0.5f,
                         -0.5f,          1.5f,           -1.5f,       2.5f};
  for (const auto edge : edges) {
    const auto bits = std::bit_cast<u32>(edge);
    for (int delta = -2; delta <= 2; delta++) {
      inputs[0].push_back(bits + static_cast<u32>(delta));
    }
  }
  uint64_t seed = 0x5eed5eed5eed5eedull;
  while (inputs[0].size() < 65536u) {
    // Random magnitudes spread over the whole exponent range.
    inputs[0].push_back(RandomF32(seed));
  }
  const auto bulk = MakeBulkCase(
      name, 1, 7, inputs,
      [](std::vector<u32> *code) {
        code->push_back(EncodeVop1(0x07, 10, Vgpr(2))); // v_cvt_u32_f32
        code->push_back(EncodeVop1(0x08, 11, Vgpr(2))); // v_cvt_i32_f32
        code->push_back(EncodeVop1(0x0c, 12, Vgpr(2))); // v_cvt_rpi_i32_f32
        code->push_back(EncodeVop1(0x0d, 13, Vgpr(2))); // v_cvt_flr_i32_f32
        code->push_back(EncodeVop1(0x52, 14, Vgpr(2))); // v_cvt_u16_f16 (low half as f16)
        code->push_back(EncodeVop1(0x53, 15, Vgpr(2))); // v_cvt_i16_f16
        AppendVop3(code, 0x15e, 16, Vgpr(2), InlineU32(1), InlineU32(0)); // v_cvt_pk_u8_f32
      },
      {O::V_CVT_U32_F32, O::V_CVT_I32_F32, O::V_CVT_RPI_I32_F32, O::V_CVT_FLR_I32_F32,
       O::V_CVT_U16_F16, O::V_CVT_I16_F16, O::V_CVT_PK_U8_F32});
  CodegenOptions fast = GetCodegenOptions();
  fast.single_f2i_saturation = true;
  CodegenOptions slow = fast;
  slow.single_f2i_saturation = false;
  const auto fast_run = RunBulk(vulkan, bulk, fast);
  const auto slow_run = RunBulk(vulkan, bulk, slow);
  static const char *const names[] = {"cvt_u32", "cvt_i32", "cvt_rpi", "cvt_flr",
                                      "cvt_u16_f16", "cvt_i16_f16", "cvt_pk_u8"};
  CompareBulk(name, bulk, fast_run, slow_run, {}, names);
  std::printf("[codegen] %-32s ok (%u values, SPIR-V %zu -> %zu words)\n", name, bulk.count,
              slow_run.spirv.size(), fast_run.spirv.size());
}

// GET_LOD_STATS instrumentation of a pixel-shader sample: compile-only (the graphics harness has
// no mip-statistics buffer), SPIR-V validation and structure in both modes. Needs no GPU
// (--lod-stats-codegen-only).
void CheckLodStatsGate() {
  constexpr const char *name = "CodegenLodStatsGate";
  GraphicsCase test;
  test.name = name;
  AppendVMovLiteral(&test.fragment_code, 20, 0x3f000000u);
  AppendVMovLiteral(&test.fragment_code, 21, 0x3f000000u);
  test.fragment_code.push_back(EncodeMimg0(0x20, 0xf)); // image_sample v[0:3], v[20:21]
  test.fragment_code.push_back(EncodeMimg1(0, 20));
  test.fragment_code.push_back(EncodeExp0(0, 0xf, true, false, true));
  test.fragment_code.push_back(EncodeExp1(0, 1, 2, 3));
  AppendEnd(&test.fragment_code);
  test.user_data = MakeSampledTextureData(Prospero::BufferFormat::k32Float);
  test.has_user_data = true;
  test.opcodes = {ShaderOpcode::V_MOV_B32, ShaderOpcode::IMAGE_SAMPLE, ShaderOpcode::EXP,
                  ShaderOpcode::S_ENDPGM};
  CodegenOptions gated = GetCodegenOptions();
  gated.lod_stats_gate = true;
  CodegenOptions legacy = gated;
  legacy.lod_stats_gate = false;
  std::vector<u32> gated_spirv;
  std::vector<u32> legacy_spirv;
  {
    ScopedCodegenOptions scope(gated);
    gated_spirv = CompileFragmentCase(test).spirv;
  }
  {
    ScopedCodegenOptions scope(legacy);
    legacy_spirv = CompileFragmentCase(test).spirv;
  }
  ValidateSpirv(name, gated_spirv);
  ValidateSpirv(name, legacy_spirv);
  const auto count = [](const std::vector<u32> &spirv, const char *op) {
    return CountSpirv(spirv, op);
  };
  // Both modes: the finest level (subgroup UMin, one AtomicUMin) and the count, added only when a
  // lane's LOD is below the field's threshold (FOrdLessThan, subgroup UMax, one AtomicIAdd in its
  // own branch). The gate adds the has-counter and smaller-than-stored branches.
  Require(name, "instrumentation",
          count(legacy_spirv, "OpImageQueryLod") == 1u &&
              count(legacy_spirv, "OpAtomicUMin") == 1u &&
              count(legacy_spirv, "OpAtomicIAdd") == 1u &&
              count(legacy_spirv, "OpAtomicLoad") == 0u &&
              count(legacy_spirv, "OpGroupNonUniformUMin") == 1u &&
              count(legacy_spirv, "OpGroupNonUniformUMax") == 1u &&
              count(legacy_spirv, "OpFOrdLessThan") >= 1u &&
              count(gated_spirv, "OpImageQueryLod") == 1u &&
              count(gated_spirv, "OpAtomicUMin") == 1u &&
              count(gated_spirv, "OpAtomicIAdd") == 1u &&
              count(gated_spirv, "OpAtomicLoad") == 1u &&
              count(gated_spirv, "OpGroupNonUniformUMin") == 1u &&
              count(gated_spirv, "OpGroupNonUniformUMax") == 1u &&
              count(gated_spirv, "OpSelectionMerge") ==
                  count(legacy_spirv, "OpSelectionMerge") + 2u,
          "unexpected GET_LOD_STATS instrumentation structure");
  std::printf("[codegen] %-32s ok (SPIR-V %zu -> %zu words)\n", name, legacy_spirv.size(),
              gated_spirv.size());
}

#include "ShaderIndirectImageCode.inc"

// A bindless sample (the guest picks one of several images from a descriptor table by a key
// found with a waterfall loop) of a GET_LOD_STATS-instrumented pixel shader: the recompiler
// emits one sample per candidate image under an OpSwitch and merges the results with an OpPhi.
// Each sample's instrumentation adds branches, so the block that ends a case is not the block it
// started in, and the phi must name that last block. Demon's Souls pixel shader
// 0xc509ed46b415549b (character creation): the phi named the case's first block and NVIDIA's
// shader compiler crashed on the invalid module. Compile-only, needs no GPU
// (--lod-stats-codegen-only).
void CheckIndirectImageLodStats() {
  constexpr const char *name = "CodegenIndirectImageLodStats";
  constexpr u32 srt = 0x1000u;      // user data 0..1: the root table
  constexpr u32 table = 0x2000u;    // root table dwords 0..1: the image table pointer
  constexpr u32 images = table + 0x158u; // the shader reads image `key` at images + key * 32
  constexpr u32 first_pointer = 0x6000u; // root table dwords 2..: pointers to constant data
  GraphicsCase test;
  test.name = name;
  test.fragment_code.assign(std::begin(kBindlessWaterfallPixelShader),
                            std::end(kBindlessWaterfallPixelShader));
  test.pixel_interpolator_settings = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  test.memory.assign(0x4000u, 0u);
  test.user_data[0] = srt;
  test.has_user_data = true;
  test.memory[srt / 4u] = table;
  for (u32 i = 0; i < 2u; i++) {
    test.memory[srt / 4u + 2u + i * 2u] = first_pointer + i * 0x1000u;
  }
  // Eight distinct 2D array images (the shader samples 2d_array) in the table.
  for (u32 key = 0; key < 8u; key++) {
    const u32 word = (images + key * 32u) / 4u;
    test.memory[word] = 0x100u + key;
    test.memory[word + 1u] =
        static_cast<u32>(Prospero::BufferFormat::k32_32_32_32Float) << 20u;
    test.memory[word + 2u] = 63u | (63u << 14u);
    test.memory[word + 3u] = DstSel(4, 5, 6, 7) |
                             (static_cast<u32>(Prospero::ImageType::kColor2DArray) << 28u);
  }
  std::vector<u32> spirv[2];
  for (u32 mode = 0; mode < 2u; mode++) {
    CodegenOptions options = GetCodegenOptions();
    options.lod_stats_gate = mode == 0u;
    ScopedCodegenOptions scope(options);
    // CompileFragmentCase validates the module (spirv-val), the instrumented one and the plain one.
    spirv[mode] = CompileFragmentCase(test, true).spirv;
  }
  for (const auto &module : spirv) {
    Require(name, "bindless switch",
            CountSpirv(module, "OpSwitch") >= 1u && CountSpirv(module, "OpPhi") >= 1u &&
                CountSpirv(module, "OpImageQueryLod") >= 2u,
            "the shader did not take the indirect image switch with LOD instrumentation (OpSwitch " +
                std::to_string(CountSpirv(module, "OpSwitch")) + ", OpPhi " +
                std::to_string(CountSpirv(module, "OpPhi")) + ", OpImageQueryLod " +
                std::to_string(CountSpirv(module, "OpImageQueryLod")) + ")");
  }
  std::printf("[codegen] %-32s ok (SPIR-V %zu / %zu words)\n", name, spirv[0].size(),
              spirv[1].size());
}

// KYTY_LOD_STATS_PLAIN_VARIANT: the plain variant of an instrumented pixel shader keeps the
// mip_stats binding and every guest instruction but has no GET_LOD_STATS feedback: no LOD query,
// subgroup reduction or atomic, so no memory side effect that would force late depth tests.
void CheckLodStatsPlainVariant() {
  constexpr const char *name = "CodegenLodStatsPlainVariant";
  GraphicsCase test;
  test.name = name;
  AppendVMovLiteral(&test.fragment_code, 20, 0x3f000000u);
  AppendVMovLiteral(&test.fragment_code, 21, 0x3f000000u);
  test.fragment_code.push_back(EncodeMimg0(0x20, 0xf)); // image_sample v[0:3], v[20:21]
  test.fragment_code.push_back(EncodeMimg1(0, 20));
  test.fragment_code.push_back(EncodeExp0(0, 0xf, true, false, true));
  test.fragment_code.push_back(EncodeExp1(0, 1, 2, 3));
  AppendEnd(&test.fragment_code);
  test.user_data = MakeSampledTextureData(Prospero::BufferFormat::k32Float);
  test.has_user_data = true;
  test.opcodes = {ShaderOpcode::V_MOV_B32, ShaderOpcode::IMAGE_SAMPLE, ShaderOpcode::EXP,
                  ShaderOpcode::S_ENDPGM};
  const auto without = CompileFragmentCase(test);
  const auto with = CompileFragmentCase(test, true);
  Require(name, "only on request", without.spirv_plain.empty() && !with.spirv_plain.empty(),
          "the plain variant was emitted without being requested, or not at all");
  Require(name, "instrumented module unchanged", with.spirv == without.spirv,
          "requesting the plain variant changed the instrumented module");
  const auto &plain = with.spirv_plain;
  const auto count = [](const std::vector<u32> &spirv, const char *op) {
    return CountSpirv(spirv, op);
  };
  Require(name, "no feedback",
          count(plain, "OpImageQueryLod") == 0u && count(plain, "OpAtomicUMin") == 0u &&
              count(plain, "OpAtomicIAdd") == 0u && count(plain, "OpAtomicLoad") == 0u &&
              count(plain, "OpGroupNonUniformUMin") == 0u &&
              count(plain, "OpImageSampleImplicitLod") ==
                  count(with.spirv, "OpImageSampleImplicitLod") &&
              count(with.spirv, "OpAtomicUMin") == 1u,
          "the plain variant still records GET_LOD_STATS feedback, or lost the guest sample");
  // Same bindings: the mip_stats storage buffer stays declared, so both modules fit one layout.
  const auto has_name = [](const std::vector<u32> &spirv, const char *wanted) {
    for (size_t i = 5; i < spirv.size();) {
      const u32 words = spirv[i] >> 16u;
      if (words == 0u) {
        break;
      }
      if ((spirv[i] & 0xffffu) == 5u /* OpName */ && words > 2u &&
          std::strcmp(reinterpret_cast<const char *>(&spirv[i + 2]), wanted) == 0) {
        return true;
      }
      i += words;
    }
    return false;
  };
  Require(name, "binding kept", has_name(plain, "mip_stats") && has_name(with.spirv, "mip_stats"),
          "the plain variant dropped the mip_stats binding declaration");
  std::printf("[codegen] %-32s ok (SPIR-V %zu -> %zu words)\n", name, with.spirv.size(),
              plain.size());
}

// Buffer and scalar-buffer loads that straddle the end of a guest descriptor range whose size is
// not a multiple of 4 bytes, with the shader's bounds check and with the device's
// robustBufferAccess2 check. Out-of-range dwords (any byte past the range) must read 0, and so
// must every byte and short of the dword that is only partly inside.
void CheckRobustBufferLoads(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenRobustBufferLoads";
  using O = ShaderOpcode;
  if (vulkan->RobustStorageAlignment() != 1u) {
    std::printf("[codegen] %-32s skipped (robustStorageBufferAccessSizeAlignment %u)\n", name,
                static_cast<u32>(vulkan->RobustStorageAlignment()));
    return;
  }
  constexpr u32 Lanes = 16;
  constexpr u32 PerLane = 16;
  constexpr u32 Data = Lanes * PerLane;  // first data dword
  constexpr u32 RangeBytes = 4u * (Data + 8u) + 2u; // dword Data+8 is half inside
  TestCase test;
  test.name = name;
  auto &code = test.code;
  code.push_back(EncodeVop2(0x1a, 1, InlineU32(2), 0)); // v1 = lane * 4
  code.push_back(EncodeVop2(0x1a, 2, InlineU32(1), 0)); // v2 = lane * 2
  code.push_back(EncodeVop2(0x1a, 5, InlineU32(6), 0)); // v5 = lane * 64 (output base)
  const auto mubuf = [&](u32 opcode, u32 vdata, u32 vaddr, u32 offset) {
    code.push_back(EncodeMubuf0(opcode, offset));
    code.push_back(EncodeMubuf1(vdata, 0, vaddr));
  };
  mubuf(0x0c, 10, 1, 4u * (Data + 4u));        // dword: Data+4 .. Data+19
  mubuf(0x0a, 11, 2, 4u * (Data + 6u));        // ushort: 2-byte steps across the end
  mubuf(0x0b, 12, 2, 4u * (Data + 5u) + 2u);   // sshort
  mubuf(0x08, 13, 0, 4u * (Data + 8u) - 6u);   // ubyte: 1-byte steps
  mubuf(0x09, 14, 0, 4u * (Data + 8u) - 6u);   // sbyte
  mubuf(0x0e, 16, 1, 4u * (Data + 2u));        // dwordx4 v[16:19]
  AppendSmemLoadOpcode(&code, 0x0a, 20, 4u * (Data + 6u)); // s_buffer_load_dwordx4 s[20:23]
  AppendSmemLoadOpcode(&code, 0x08, 24, 4u * (Data + 8u)); // s_buffer_load_dword s24 (partial)
  AppendSmemLoadOpcode(&code, 0x08, 25, 4u * (Data + 7u)); // s_buffer_load_dword s25
  code.push_back(EncodeSopp(0x0c, 0));
  for (u32 s = 0; s < 6; s++) {
    code.push_back(EncodeVop1(0x01, 30 + s, 20 + s)); // v_mov_b32 v(30+s), s(20+s)
  }
  const u32 outputs[] = {10, 11, 12, 13, 14, 16, 17, 18, 19, 30, 31, 32, 33, 34, 35};
  for (u32 k = 0; k < std::size(outputs); k++) {
    code.push_back(EncodeMubuf0(0x1c, 4u * k)); // buffer_store_dword v, v5, s[48:51]
    code.push_back(EncodeMubuf1(outputs[k], 12, 5));
  }
  AppendEnd(&code);
  test.initial.assign(Data + 24u, 0u);
  for (u32 i = 0; i < Data + 24u; i++) {
    test.initial[i] = i < Data ? 0u : 0x80a0c0e0u + (i - Data) * 0x01010101u;
  }
  test.compute_info.threads_num[0] = Lanes;
  test.compute_info.threads_num[1] = 1;
  test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.has_compute_info = true;
  test.opcodes = {O::V_LSHLREV_B32, O::BUFFER_LOAD_DWORD, O::BUFFER_LOAD_USHORT,
                  O::BUFFER_LOAD_SSHORT, O::BUFFER_LOAD_UBYTE, O::BUFFER_LOAD_SBYTE,
                  O::BUFFER_LOAD_DWORDX4, O::S_BUFFER_LOAD_DWORDX4, O::S_BUFFER_LOAD_DWORD,
                  O::S_WAITCNT, O::V_MOV_B32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};

  // Reference: a dword reads its data when all four bytes are inside the range, else 0.
  const auto dword = [&](u32 index) -> u32 {
    return 4u * index + 4u <= RangeBytes && index < test.initial.size() ? test.initial[index]
                                                                          : 0u;
  };
  const auto sub = [&](u32 byte, u32 bits, bool sign) -> u32 {
    const auto word = dword(byte / 4u) >> ((byte % 4u) * 8u);
    const auto mask = bits == 8u ? 0xffu : 0xffffu;
    auto value = word & mask;
    if (sign && (value & (1u << (bits - 1u))) != 0u) {
      value |= ~mask;
    }
    return value;
  };
  std::vector<u32> expected(Data, 0u);
  for (u32 lane = 0; lane < Lanes; lane++) {
    const std::array<u32, 15> values = {
        dword(Data + 4u + lane),
        sub(4u * (Data + 6u) + 2u * lane, 16, false),
        sub(4u * (Data + 5u) + 2u + 2u * lane, 16, true),
        sub(4u * (Data + 8u) - 6u + lane, 8, false),
        sub(4u * (Data + 8u) - 6u + lane, 8, true),
        dword(Data + 2u + lane),
        dword(Data + 3u + lane),
        dword(Data + 4u + lane),
        dword(Data + 5u + lane),
        dword(Data + 6u),
        dword(Data + 7u),
        dword(Data + 8u),
        dword(Data + 9u),
        dword(Data + 8u),
        dword(Data + 7u)};
    std::copy(values.begin(), values.end(), expected.begin() + lane * PerLane);
  }
  CodegenOptions device = GetCodegenOptions();
  device.robust_buffer_loads = true;
  CodegenOptions shader = device;
  shader.robust_buffer_loads = false;
  std::vector<u32> device_spirv;
  std::vector<u32> shader_spirv;
  std::vector<u32> device_words;
  std::vector<u32> shader_words;
  // The host binds storage buffers in whole dwords (descriptors.cpp NativeStorageBuffer):
  // NVIDIA returns data for a dword only partly inside the range despite reporting a 1-byte
  // robustness alignment. Both runs bind that rounded range, as the emulator does. Subword loads
  // use native 8/16-bit storage views, whose OpArrayLength counts bytes or shorts, so with the
  // unrounded range the shader check would return the short at byte 4 * (Data + 8).
  test.storage_buffer_range_bytes = RangeBytes & ~3u;
  {
    ScopedCodegenOptions scope(device);
    device_words = RunCollect(vulkan, test, Data, &device_spirv);
  }
  {
    ScopedCodegenOptions scope(shader);
    shader_words = RunCollect(vulkan, test, Data, &shader_spirv);
  }
  RequireEqualWords(name, "shader-checked loads", expected, shader_words);
  RequireEqualWords(name, "device-checked loads", expected, device_words);
  std::printf("[codegen] %-32s ok (OpArrayLength %zu -> %zu, OpBranchConditional %zu -> %zu)\n",
              name, CountSpirv(shader_spirv, "OpArrayLength"),
              CountSpirv(device_spirv, "OpArrayLength"),
              CountSpirv(shader_spirv, "OpBranchConditional"),
              CountSpirv(device_spirv, "OpBranchConditional"));
}

// Random finite f32 with a moderate exponent, so products and sums stay normal.
u32 RandomModerateF32(uint64_t &state) {
  const auto bits = NextRandom(state);
  const auto exponent = 127u - 20u + (NextRandom(state) % 41u);
  return (bits & 0x807fffffu) | (exponent << 23u);
}

float RoundedF32(float value) {
  volatile float stored = value;
  return stored;
}

// V_MAD/MAC/MADMK/MADAK (unfused on PS5) and V_FMA (fused) in a compute shader under every
// MadMode: Exact must give round(round(a*b)+c), Fused and Position (compute has no position
// export) the fused result.
void CheckMadModes(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenMadModes";
  using O = ShaderOpcode;
  std::vector<std::vector<u32>> inputs(3);
  uint64_t seed = 0x3141592653589793ull;
  while (inputs[0].size() < 16384u) {
    for (auto &input : inputs) {
      input.push_back(RandomModerateF32(seed));
    }
  }
  static constexpr u32 MadmkConstant = 0x3fa66666u; // 1.3f
  const auto bulk = MakeBulkCase(
      name, 3, 6, inputs,
      [](std::vector<u32> *code) {
        AppendVop3(code, 0x141, 10, Vgpr(2), Vgpr(3), Vgpr(4)); // v_mad_f32
        code->push_back(EncodeVop1(0x01, 11, Vgpr(4)));         // v_mov_b32 v11, v4
        code->push_back(EncodeVop2(0x1f, 11, Vgpr(2), 3));      // v_mac_f32 v11, v2, v3
        code->push_back(EncodeVop2(0x20, 12, Vgpr(2), 4));      // v_madmk_f32 v12, v2, K, v4
        code->push_back(MadmkConstant);
        code->push_back(EncodeVop2(0x21, 13, Vgpr(2), 3));      // v_madak_f32 v13, v2, v3, K
        code->push_back(MadmkConstant);
        AppendVop3(code, 0x14b, 14, Vgpr(2), Vgpr(3), Vgpr(4)); // v_fma_f32
        code->push_back(EncodeVop1(0x01, 15, Vgpr(4)));         // v_mov_b32 v15, v4
        code->push_back(EncodeVop2(0x2b, 15, Vgpr(2), 3));      // v_fmac_f32 v15, v2, v3
      },
      {O::V_MAD_F32, O::V_MOV_B32, O::V_MAC_F32, O::V_MADMK_F32, O::V_MADAK_F32,
       O::V_FMA_F32});
  const auto k = std::bit_cast<float>(MadmkConstant);
  const auto reference = [&](u32 output, u32 e, bool fused) -> u32 {
    const auto a = std::bit_cast<float>(inputs[0][e]);
    const auto b = std::bit_cast<float>(inputs[1][e]);
    const auto c = std::bit_cast<float>(inputs[2][e]);
    const auto mad = [&](float x, float y, float z) {
      return fused ? std::fma(x, y, z) : RoundedF32(RoundedF32(x * y) + z);
    };
    switch (output) {
    case 0:
    case 1: return std::bit_cast<u32>(mad(a, b, c));
    case 2: return std::bit_cast<u32>(mad(a, k, c));
    case 3: return std::bit_cast<u32>(mad(a, b, k));
    default: return std::bit_cast<u32>(std::fma(a, b, c));
    }
  };
  static const char *const names[] = {"mad", "mac", "madmk", "madak", "fma", "fmac"};
  size_t differing = 0;
  for (const auto mode : {ShaderRecompiler::MadMode::Exact, ShaderRecompiler::MadMode::Fused,
                          ShaderRecompiler::MadMode::Position}) {
    CodegenOptions options = GetCodegenOptions();
    options.mad_mode = mode;
    const auto run = RunBulk(vulkan, bulk, options);
    const bool fused = mode != ShaderRecompiler::MadMode::Exact;
    for (u32 o = 0; o < bulk.outputs; o++) {
      const auto out = BulkOutput(bulk, run.words, o);
      for (u32 e = 0; e < bulk.count; e++) {
        const auto expected = reference(o, e, fused);
        if (out[e] != expected) {
          Fail(name, names[o],
               "mode " + std::to_string(static_cast<int>(mode)) + " element " +
                   std::to_string(e) + ": got " + Hex(out[e]) + ", expected " +
                   Hex(expected));
        }
        if (o == 0 && !fused && expected != reference(o, e, true)) {
          differing++;
        }
      }
    }
  }
  {
    // KYTY_MAD_FLUSH on normal operands: still round(round(a*b)+c), a denormal sum flushed.
    CodegenOptions options = GetCodegenOptions();
    options.mad_mode = ShaderRecompiler::MadMode::Exact;
    options.mad_flush = true;
    const auto run = RunBulk(vulkan, bulk, options);
    for (u32 o = 0; o < 4u; o++) {
      const auto out = BulkOutput(bulk, run.words, o);
      for (u32 e = 0; e < bulk.count; e++) {
        auto expected = reference(o, e, false);
        if ((expected & 0x7f800000u) == 0u) {
          expected &= 0x80000000u;
        }
        if (out[e] != expected) {
          Fail(name, names[o], "KYTY_MAD_FLUSH element " + std::to_string(e) + ": got " +
                                   Hex(out[e]) + ", expected " + Hex(expected));
        }
      }
    }
  }
  std::printf("[codegen] %-32s ok (%u elements, %zu where fused and unfused differ)\n", name,
              bulk.count, differing);
}

// KYTY_WAVE64_HIGH_LANES: a wave64 vertex shader (32-wide host subgroups, no link between them)
// spills an SGPR to v13 lane 40 and reads it back, and reads lane 41 that nothing wrote. With the
// switch both reads are resolved at compile time (the spilled value, and 0), so no subgroup
// shuffle of a lane >= 32 remains; without it the reads stay shuffles.
void CheckWave64HighLanes() {
  constexpr const char *name = "CodegenWave64HighLanes";
  std::vector<u32> code;
  AppendSMovLiteral(&code, 10, 0x40490fdbu);
  AppendVop3(&code, 0x361, 13, 10, InlineU32(40));   // v_writelane_b32 v13, s10, 40
  AppendVop3(&code, 0x360, 11, Vgpr(13), InlineU32(40)); // v_readlane_b32 s11, v13, 40
  AppendVop3(&code, 0x360, 12, Vgpr(13), InlineU32(41)); // v_readlane_b32 s12, v13, 41
  code.push_back(EncodeVop1(0x01, 4, 11));            // v_mov_b32 v4, s11
  code.push_back(EncodeVop1(0x01, 5, 12));            // v_mov_b32 v5, s12
  code.push_back(EncodeExp0(12, 0xf, true));          // exp pos0 v4, v5, v4, v5 done
  code.push_back(EncodeExp1(4, 5, 4, 5));
  AppendEnd(&code);
  const auto compile = [&](bool enabled) {
    CodegenOptions options = GetCodegenOptions();
    options.wave64_high_lanes = enabled;
    ScopedCodegenOptions scope(options);
    std::array<u32, 16> user_data{};
    ShaderVertexInputInfo vertex;
    ShaderRecompiler::CompileOptions compile_options;
    compile_options.stage = ShaderType::Vertex;
    compile_options.wave_size = 64;
    compile_options.user_data = user_data;
    compile_options.input_info.vertex = &vertex;
    auto translated = ShaderRecompiler::TranslateProgram(code, compile_options);
    auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSnapshot resources;
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    const ShaderRecompiler::IR::SrtRuntime runtime{
        .user_data = compile_options.user_data,
        .shader_base = reinterpret_cast<uint64_t>(code.data()),
    };
    Require(name, "resource materialization",
            ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources,
                                                       specialization),
            "vertex resources could not be materialized");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), compile_options,
                                                   specialization);
    ValidateSpirv(name, result.spirv);
    return result.spirv;
  };
  const auto on = compile(true);
  const auto off = compile(false);
  const auto on_shuffles = CountSpirv(on, "OpGroupNonUniformShuffle ");
  const auto off_shuffles = CountSpirv(off, "OpGroupNonUniformShuffle ");
  Require(name, "high-lane reads resolved", on_shuffles == 0u,
          "KYTY_WAVE64_HIGH_LANES left " + std::to_string(on_shuffles) + " shuffles");
  std::printf("[codegen] %-32s ok (switch off: %zu shuffles)\n", name, off_shuffles);
}

// The f16 bits of s + e (|e| <= ulp(s) / 2, both exact), rounded once to nearest even; f16
// subnormals kept, overflow to INF. Reference for KYTY_F16_EXACT_ROUND.
u32 RoundPairToF16(double s, double e) {
  if (std::isnan(s)) {
    return 0x7e00u;
  }
  const u32 sign = std::signbit(s) ? 0x8000u : 0u;
  if (std::isinf(s)) {
    return sign | 0x7c00u;
  }
  if (sign != 0u) {
    s = -s;
    e = -e;
  }
  if (s == 0.0) {
    return sign;
  }
  int exponent = 0;
  (void)std::frexp(s, &exponent); // s in [2^(exponent-1), 2^exponent)
  const int ulp_exp = s < std::ldexp(1.0, -14) ? -24 : exponent - 11;
  const double ulp = std::ldexp(1.0, ulp_exp);
  const double low = std::floor(s / ulp) * ulp;
  const double d = (s - low) - ulp / 2.0;
  const int side = d != 0.0 ? (d > 0.0 ? 1 : -1) : (e > 0.0 ? 1 : (e < 0.0 ? -1 : 0));
  auto units = static_cast<uint64_t>(low / ulp);
  if (side > 0 || (side == 0 && (units & 1u) != 0u)) {
    units++;
  }
  const double value = static_cast<double>(units) * ulp;
  if (value >= 65536.0) {
    return sign | 0x7c00u;
  }
  if (value == 0.0) {
    return sign;
  }
  if (value < std::ldexp(1.0, -14)) {
    return sign | static_cast<u32>(units);
  }
  int vexp = 0;
  const double mant = std::frexp(value, &vexp); // [0.5, 1)
  const auto fraction = static_cast<u32>((mant * 2.0 - 1.0) * 1024.0);
  return sign | (static_cast<u32>(vexp - 1 + 15) << 10u) | fraction;
}

u32 F16BitsToF32Bits(u32 h) {
  const u32 sign = (h & 0x8000u) << 16u;
  const u32 exp = (h >> 10u) & 0x1fu;
  const u32 frac = h & 0x3ffu;
  if (exp == 0x1fu) {
    return sign | 0x7f800000u | (frac << 13u) | (frac != 0u ? 0x400000u : 0u);
  }
  if (exp == 0u) {
    const float magnitude = std::ldexp(static_cast<float>(frac), -24);
    return sign | std::bit_cast<u32>(magnitude);
  }
  return sign | ((exp + 112u) << 23u) | (frac << 13u);
}

// KYTY_F16_EXACT_ROUND: V_CVT_F16_F32 rounds to nearest even (ties, subnormals, overflow to INF);
// V_CVT_F16_F32 then V_CVT_F32_F16 quantises (a host that folds the pair keeps f32 precision);
// V_FMA_MIXLO_F16 with f32 sources rounds a*b+c once (every fourth element is built to land next
// to an f16 tie, where an f32 Fma rounded again to f16 can differ).
void CheckF16ExactRound(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenF16ExactRound";
  using O = ShaderOpcode;
  std::vector<std::vector<u32>> inputs(3);
  uint64_t seed = 0x2718281828459045ull;
  const auto next = [&]() {
    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<u32>(seed >> 32u);
  };
  while (inputs[0].size() < 16384u) {
    const float a = std::bit_cast<float>(F16BitsToF32Bits(0x3c00u + next() % 0x0800u));
    const float b = std::bit_cast<float>(F16BitsToF32Bits(0x3800u + next() % 0x0800u)) *
                    (1.0f + std::ldexp(static_cast<float>(next() % 4096u), -23));
    inputs[1].push_back(std::bit_cast<u32>(a));
    inputs[2].push_back(std::bit_cast<u32>(b));
    if (inputs[0].size() % 4u == 3u) {
      // c = f32(tie - a*b): the exact sum sits next to an f16 tie.
      const double tie = static_cast<double>(std::bit_cast<float>(
                             F16BitsToF32Bits(0x3c00u + next() % 0x0400u))) +
                         std::ldexp(1.0, -11);
      const float c = static_cast<float>(tie - static_cast<double>(a) * static_cast<double>(b));
      inputs[0].push_back(std::bit_cast<u32>(c));
      continue;
    }
    // A random finite f16 h, then a value at, next to or past the tie above it, as f32.
    const u32 h = next() % 0x7c00u;
    const u32 base = F16BitsToF32Bits(h);
    const u32 tie_step = h >= 0x400u ? 0x1000u : 0u;
    u32 x = base;
    switch (next() % 6u) {
    case 0: x = base + tie_step; break;
    case 1: x = base + tie_step + 1u; break;
    case 2: x = base + (tie_step != 0u ? tie_step - 1u : 0u); break;
    case 3: x = next() & 0x7fffffffu; break;
    case 4: x = 0x477fe000u + (next() & 0x3fffu); break;
    default: x = (next() & 0x00ffffffu) | 0x33000000u; break;
    }
    if ((next() & 1u) != 0u) {
      x |= 0x80000000u;
    }
    inputs[0].push_back(x);
  }
  const auto bulk = MakeBulkCase(
      name, 3, 3, inputs,
      [](std::vector<u32> *code) {
        AppendVMovU32(code, 10, 0u);
        code->push_back(EncodeVop1(0x0a, 10, Vgpr(2)));         // v_cvt_f16_f32 v10, v2
        code->push_back(EncodeVop1(0x0a, 11, Vgpr(2)));         // v_cvt_f16_f32 v11, v2
        code->push_back(EncodeVop1(0x0b, 11, Vgpr(11)));        // v_cvt_f32_f16 v11, v11
        AppendVMovU32(code, 12, 0u);
        AppendVop3p(code, 0x21, 12, Vgpr(3), Vgpr(4), Vgpr(2)); // v_fma_mixlo_f16 v12, v3, v4, v2
      },
      {O::V_MOV_B32, O::V_CVT_F16_F32, O::V_CVT_F32_F16, O::V_MAD_MIXLO_F16});
  CodegenOptions options = GetCodegenOptions();
  options.f16_exact_round = true;
  options.native_f16 = false;
  const auto run = RunBulk(vulkan, bulk, options);
  options.f16_exact_round = false;
  const auto legacy = RunBulk(vulkan, bulk, options);
  const auto out_cvt = BulkOutput(bulk, run.words, 0);
  const auto out_rt = BulkOutput(bulk, run.words, 1);
  const auto out_mix = BulkOutput(bulk, run.words, 2);
  const auto old_cvt = BulkOutput(bulk, legacy.words, 0);
  const auto old_rt = BulkOutput(bulk, legacy.words, 1);
  const auto old_mix = BulkOutput(bulk, legacy.words, 2);
  size_t legacy_cvt = 0;
  size_t legacy_round_trip = 0;
  size_t legacy_mix = 0;
  size_t mix_checked = 0;
  for (u32 e = 0; e < bulk.count; e++) {
    const u32 x = inputs[0][e];
    const float xf = std::bit_cast<float>(x);
    const bool denormal = (x & 0x7f800000u) == 0u;
    const bool nan = std::isnan(xf);
    const u32 half =
        nan ? 0x7e00u | ((x >> 13u) & 0x3ffu) | ((x >> 16u) & 0x8000u)
            : RoundPairToF16(denormal ? (std::signbit(xf) ? -0.0 : 0.0) : static_cast<double>(xf),
                             0.0);
    const u32 trip = F16BitsToF32Bits(half);
    const auto cvt = out_cvt[e];
    const auto rt = out_rt[e];
    if (cvt != half) {
      Fail(name, "v_cvt_f16_f32",
           "element " + std::to_string(e) + " (" + Hex(x) + "): got " + Hex(cvt) +
               ", expected " + Hex(half));
    }
    if (!nan && rt != trip) {
      Fail(name, "cvt round trip",
           "element " + std::to_string(e) + " (" + Hex(x) + "): got " + Hex(rt) +
               ", expected " + Hex(trip));
    }
    legacy_cvt += old_cvt[e] != half ? 1u : 0u;
    legacy_round_trip += !nan && old_rt[e] != trip ? 1u : 0u;
    if (nan || std::isinf(xf) || denormal) {
      continue;
    }
    const double a = std::bit_cast<float>(inputs[1][e]);
    const double b = std::bit_cast<float>(inputs[2][e]);
    const double c = xf;
    const double p = a * b; // exact: two 24-bit significands
    const double s2 = p + c;
    const double bb = s2 - p;
    const double err = (p - (s2 - bb)) + (c - bb);
    const u32 mix = RoundPairToF16(s2, err);
    const auto got = out_mix[e];
    mix_checked++;
    if ((got & 0xffffu) != mix) {
      Fail(name, "v_fma_mixlo_f16",
           "element " + std::to_string(e) + ": got " + Hex(got) + ", expected " + Hex(mix));
    }
    legacy_mix += (old_mix[e] & 0xffffu) != mix ? 1u : 0u;
  }
  std::printf("[codegen] %-32s ok (%u elements, %zu mix; switch off differs: cvt %zu, round trip "
              "%zu, mix %zu)\n",
              name, bulk.count, mix_checked, legacy_cvt, legacy_round_trip, legacy_mix);
}

// RDNA2 ISA check small items, each with its switch on and off: S_BFM_B32 with offset + count > 32
// (KYTY_BFM_DEFINED), OMOD of a -0 result (KYTY_OMOD_ZERO), V_CVT_F32_F64 overflow
// (KYTY_CVT_F32_F64_INF), and V_MOVRELS/V_MOVRELD with M0 = 257 (KYTY_MOVREL_FULL_M0: out of range,
// so the source reads v0 and the destination is not written; the legacy M0[7:0] = 1).
void CheckIsaSmallFixes(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenIsaSmallFixes";
  TestCase test;
  test.name = name;
  auto &code = test.code;
  code.push_back(EncodeSop2(0x24, 10, InlineU32(20), InlineU32(16))); // s_bfm_b32 s10, 20, 16
  AppendVop3(&code, 0x108, 14, 128, 243, 0, 0, 0, false, 1);        // v_mul_f32 v14, 0, -1.0 mul:2
  AppendVMovU32(&code, 16, 0u);
  AppendVMovU32(&code, 17, 0x7e300000u);                             // v[16:17] = 2^996
  code.push_back(EncodeVop1(0x0f, 15, Vgpr(16)));                    // v_cvt_f32_f64 v15, v[16:17]
  AppendVMovU32(&code, 2, 0x22u);
  AppendVMovU32(&code, 3, 0x33u);
  AppendVMovU32(&code, 4, 0x44u);
  AppendVMovU32(&code, 5, 0x55u);
  AppendVMovU32(&code, 6, 0x66u);
  AppendSMovLiteral(&code, 124, 257u);                                // s_mov_b32 m0, 257
  code.push_back(EncodeVop1(0x43, 20, Vgpr(2)));                     // v_movrels_b32 v20, v2
  code.push_back(EncodeVop1(0x42, 4, Vgpr(6)));                      // v_movreld_b32 v4, v6
  AppendStoreSgpr(&code, 10, 0);
  AppendStoreVgpr(&code, 14, 1);
  AppendStoreVgpr(&code, 15, 2);
  AppendStoreVgpr(&code, 20, 3);
  AppendStoreVgpr(&code, 5, 4);
  AppendEnd(&code);
  test.initial.assign(5, 0xdeadbeefu);
  test.compute_info.threads_num[0] = 1;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = 32;
  test.has_compute_info = true;
  const auto run = [&](bool enabled) {
    CodegenOptions options = GetCodegenOptions();
    options.bfm_defined = enabled;
    options.omod_zero = enabled;
    options.cvt_f32_f64_inf = enabled;
    options.movrel_full_m0 = enabled;
    ScopedCodegenOptions scope(options);
    return RunCollect(vulkan, test, test.initial.size());
  };
  const auto on = run(true);
  const auto off = run(false);
  const std::vector<u32> expected = {0xffff0000u, 0x00000000u, 0x7f800000u, 0x00000000u, 0x55u};
  RequireEqualWords(name, "switches on", expected, on);
  std::printf("[codegen] %-32s ok (off: bfm %08x, omod %08x, cvt %08x, movrels %08x, movreld "
              "%08x)\n",
              name, off[0], off[1], off[2], off[3], off[4]);
}

// A vertex shader with one MAD feeding the position and one feeding a parameter: which of them
// are fused, which carry NoContraction, and the Invariant position, per MadMode.
void CheckMadPositionSlice() {
  constexpr const char *name = "CodegenMadPositionSlice";
  std::vector<u32> code;
  AppendVop3(&code, 0x141, 4, Vgpr(0), Vgpr(1), Vgpr(2)); // v_mad_f32 v4 (position)
  AppendVop3(&code, 0x141, 5, Vgpr(0), Vgpr(2), Vgpr(1)); // v_mad_f32 v5 (parameter)
  AppendVMovLiteral(&code, 6, 0x3f800000u);
  code.push_back(EncodeExp0(12, 0xf, true)); // exp pos0 v4, v4, v4, v6 done
  code.push_back(EncodeExp1(4, 4, 4, 6));
  code.push_back(EncodeExp0(32, 0xf, false)); // exp param0 v5, v5, v5, v6
  code.push_back(EncodeExp1(5, 5, 5, 6));
  AppendEnd(&code);
  const auto compile = [&](ShaderRecompiler::MadMode mode) {
    CodegenOptions options = GetCodegenOptions();
    options.mad_mode = mode;
    ScopedCodegenOptions scope(options);
    std::array<u32, 16> user_data{};
    ShaderVertexInputInfo vertex;
    ShaderRecompiler::CompileOptions compile_options;
    compile_options.stage = ShaderType::Vertex;
    compile_options.user_data = user_data;
    compile_options.input_info.vertex = &vertex;
    auto translated = ShaderRecompiler::TranslateProgram(code, compile_options);
    auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSnapshot resources;
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    const ShaderRecompiler::IR::SrtRuntime runtime{
        .user_data = compile_options.user_data,
        .shader_base = reinterpret_cast<uint64_t>(code.data()),
    };
    Require(name, "resource materialization",
            ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources,
                                                       specialization),
            "vertex resources could not be materialized");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), compile_options,
                                                   specialization);
    ValidateSpirv(name, result.spirv);
    return result.spirv;
  };
  struct Expectation {
    ShaderRecompiler::MadMode mode;
    size_t fma;
    size_t no_contraction;
    size_t invariant;
  };
  for (const auto &expect : {Expectation{ShaderRecompiler::MadMode::Exact, 0, 4, 1},
                             Expectation{ShaderRecompiler::MadMode::Position, 1, 2, 1},
                             Expectation{ShaderRecompiler::MadMode::Fused, 2, 0, 0}}) {
    const auto spirv = compile(expect.mode);
    const auto fma = CountSpirv(spirv, "Fma ");
    const auto no_contraction = CountSpirv(spirv, "NoContraction");
    const auto invariant = CountSpirv(spirv, "Invariant");
    Require(name, "decorations",
            fma == expect.fma && no_contraction == expect.no_contraction &&
                invariant == expect.invariant,
            "mode " + std::to_string(static_cast<int>(expect.mode)) + ": Fma " +
                std::to_string(fma) + ", NoContraction " + std::to_string(no_contraction) +
                ", Invariant " + std::to_string(invariant));
  }
  std::printf("[codegen] %-32s ok\n", name);
}

std::vector<u32> CompilePixelShader(const char *name, const std::vector<u32> &code,
                                    const ShaderPixelInputInfo &pixel_info) {
  auto pixel = pixel_info;
  // CompileOptions::user_data is a span: keep its backing array alive for the whole
  // translation, materialization and compilation below.
  const auto user_data = MakeNativeUserData(nullptr);
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Pixel;
  options.input_info.pixel = &pixel;
  options.user_data = user_data;
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(code.data()),
  };
  Require(name, "resource materialization",
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization),
          "pixel resources could not be materialized");
  auto result =
      ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ValidateSpirv(name, result.spirv);
  return result.spirv;
}

// V_INTERP_P2 reads through different hardware I/J pairs: a perspective-center input, a
// linear-center input and an input read at the centroid and at the center. KYTY_INTERP_MODES
// decorates per input; the legacy path made every input NoPerspective because LINEAR_CENTER is
// enabled and ignored the centroid pair.
void CheckInterpolationModes() {
  constexpr const char *name = "CodegenInterpolationModes";
  std::vector<u32> code;
  code.push_back(EncodeVintrp(0, 10, 0, 0, 0)); // v_interp_p1_f32 v10, v0, attr0.x
  code.push_back(EncodeVintrp(1, 10, 0, 0, 1)); // v_interp_p2_f32 v10, v1, attr0.x
  code.push_back(EncodeVintrp(0, 11, 1, 0, 4)); // linear center pair v4/v5
  code.push_back(EncodeVintrp(1, 11, 1, 0, 5));
  code.push_back(EncodeVintrp(0, 12, 2, 0, 2)); // perspective centroid pair v2/v3
  code.push_back(EncodeVintrp(1, 12, 2, 0, 3));
  code.push_back(EncodeVintrp(0, 13, 2, 1, 0)); // attr2.y at the center
  code.push_back(EncodeVintrp(1, 13, 2, 1, 1));
  code.push_back(EncodeVintrp(0, 14, 3, 0, 2)); // attr3 only at the centroid
  code.push_back(EncodeVintrp(1, 14, 3, 0, 3));
  code.push_back(EncodeVop2(0x03, 13, Vgpr(13), 14)); // v_add_f32 v13, v13, v14
  code.push_back(EncodeExp0(0, 0xf, true, false, true));
  code.push_back(EncodeExp1(10, 11, 12, 13));
  AppendEnd(&code);
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 4;
  for (u32 i = 0; i < std::size(pixel.interpolator_settings); i++) {
    pixel.interpolator_settings[i] = i;
  }
  pixel.ps_perspective_center_vgpr = 0;
  pixel.ps_perspective_centroid_vgpr = 2;
  pixel.ps_linear_center_vgpr = 4;
  pixel.ps_no_perspective = true;
  pixel.ps_system_input_base = 6;
  struct Counts {
    size_t no_perspective, centroid, interpolate_at_centroid;
  };
  const auto compile = [&](bool modes) {
    CodegenOptions options = GetCodegenOptions();
    options.interp_modes = modes;
    ScopedCodegenOptions scope(options);
    const auto spirv = CompilePixelShader(name, code, pixel);
    return Counts{CountSpirv(spirv, " NoPerspective"), CountSpirv(spirv, " Centroid\n"),
                  CountSpirv(spirv, "InterpolateAtCentroid")};
  };
  const auto modes = compile(true);
  const auto legacy = compile(false);
  Require(name, "per-input interpolation",
          modes.no_perspective == 1u && modes.centroid == 1u &&
              modes.interpolate_at_centroid == 1u,
          "modes: NoPerspective " + std::to_string(modes.no_perspective) + ", Centroid " +
              std::to_string(modes.centroid) + ", InterpolateAtCentroid " +
              std::to_string(modes.interpolate_at_centroid));
  Require(name, "legacy interpolation",
          legacy.no_perspective == 4u && legacy.centroid == 0u &&
              legacy.interpolate_at_centroid == 0u,
          "legacy: NoPerspective " + std::to_string(legacy.no_perspective) + ", Centroid " +
              std::to_string(legacy.centroid) + ", InterpolateAtCentroid " +
              std::to_string(legacy.interpolate_at_centroid));
  std::printf("[codegen] %-32s ok\n", name);
}

// KYTY_LDS_SLOTS: a pixel shader whose LDS accesses are all dwords at (LaneId << 2) + c (the
// guest compiler's per-lane VGPR spill) keeps one Function variable per element offset instead of
// the dynamically indexed function-scope array; any other access keeps the array.
void CheckLdsSlots() {
  constexpr const char *name = "CodegenLdsSlots";
  const auto build = [](bool foreign_address) {
    std::vector<u32> code;
    code.push_back(EncodeVop1(0x01, 4, InlineU32(0)));       // v_mov_b32 v4, 0
    code.push_back(EncodeVop2(0x23, 5, 193u, 4));            // v_mbcnt_lo_u32_b32 v5, -1, v4
    code.push_back(EncodeVop2(0x24, 6, 193u, 5));            // v_mbcnt_hi_u32_b32 v6, -1, v5
    code.push_back(EncodeVop2(0x1a, 7, InlineU32(2), 6));    // v_lshlrev_b32 v7, 2, v6
    code.push_back(EncodeDs0(0x0d, 0));                      // ds_write_b32 v7, v0
    code.push_back(EncodeDs1(0, 0, 7));
    code.push_back(EncodeDs0(0x0d, 256));                    // ds_write_b32 v7, v1 offset:256
    code.push_back(EncodeDs1(0, 1, 7));
    if (foreign_address) {
      // An address that is not the lane's own: ds_write_b32 v4(=0), v1.
      code.push_back(EncodeDs0(0x0d, 0));
      code.push_back(EncodeDs1(0, 1, 4));
    }
    code.push_back(EncodeVop2(0x03, 0, Vgpr(0), 1));         // v_add_f32 v0, v0, v1
    code.push_back(EncodeDs0(0x36, 256));                    // ds_read_b32 v10, v7 offset:256
    code.push_back(EncodeDs1(10, 0, 7));
    code.push_back(EncodeDs0(0x36, 0));                      // ds_read_b32 v11, v7
    code.push_back(EncodeDs1(11, 0, 7));
    code.push_back(EncodeVop1(0x01, 12, InlineU32(0)));
    code.push_back(EncodeVop1(0x01, 13, InlineU32(0)));
    code.push_back(EncodeExp0(0, 0xf, true, false, true));
    code.push_back(EncodeExp1(10, 11, 12, 13));
    AppendEnd(&code);
    return code;
  };
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 1;
  pixel.interpolator_settings[0] = 0;
  pixel.ps_perspective_center_vgpr = 0;
  pixel.ps_system_input_base = 2;
  pixel.lds_size_dwords = 1024;
  struct Counts {
    size_t slots, array;
  };
  const auto compile = [&](bool slots, bool foreign) {
    CodegenOptions options = GetCodegenOptions();
    options.lds_slots = slots;
    ScopedCodegenOptions scope(options);
    const auto spirv = CompilePixelShader(name, build(foreign), pixel);
    return Counts{CountSpirv(spirv, "\"lds_slot_"), CountSpirv(spirv, "\"lds_dwords\"")};
  };
  const auto on = compile(true, false);
  const auto off = compile(false, false);
  const auto foreign = compile(true, true);
  Require(name, "own-lane accesses take the slots", on.slots == 2u && on.array == 0u,
          "slots " + std::to_string(on.slots) + ", array " + std::to_string(on.array));
  Require(name, "switch off keeps the array", off.slots == 0u && off.array == 1u,
          "slots " + std::to_string(off.slots) + ", array " + std::to_string(off.array));
  Require(name, "a foreign address keeps the array", foreign.slots == 0u && foreign.array == 1u,
          "slots " + std::to_string(foreign.slots) + ", array " + std::to_string(foreign.array));
  std::printf("[codegen] %-32s ok\n", name);
}

// KYTY_PS_FLUSH=off: a pixel shader's transcendental inputs are not flushed in the shader (no
// flush sequence emitted); the default keeps them.
void CheckDenormFlushSkip() {
  constexpr const char *name = "CodegenDenormFlushSkip";
  std::vector<u32> code;
  code.push_back(EncodeVop1(0x2a, 10, Vgpr(0)));             // v_rcp_f32 v10, v0
  code.push_back(EncodeVop1(0x2e, 11, Vgpr(0)));             // v_rsq_f32 v11, v0
  code.push_back(EncodeVop1(0x01, 12, InlineU32(0)));
  code.push_back(EncodeVop1(0x01, 13, InlineU32(0)));
  code.push_back(EncodeExp0(0, 0xf, true, false, true));
  code.push_back(EncodeExp1(10, 11, 12, 13));
  AppendEnd(&code);
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 1;
  pixel.interpolator_settings[0] = 0;
  pixel.ps_perspective_center_vgpr = 0;
  pixel.ps_system_input_base = 2;
  const auto saved = ShaderRecompiler::Spirv::GetHostFloatControls();
  auto controls = saved;
  controls.denorm_flush_f32 = false;
  ShaderRecompiler::Spirv::SetHostFloatControls(controls);
  const auto flushes = [&](uint8_t skip) {
    CodegenOptions options = GetCodegenOptions();
    options.denorm_flush_skip = skip;
    options.host_ftz_inputs = false;
    ScopedCodegenOptions scope(options);
    const auto before = ShaderRecompiler::Spirv::EmittedDenormFlushes(0);
    (void)CompilePixelShader(name, code, pixel);
    return ShaderRecompiler::Spirv::EmittedDenormFlushes(0) - before;
  };
  const auto exact = flushes(0);
  const auto skipped = flushes(ShaderRecompiler::DenormFlushSkipPixel);
  const auto compute_only = flushes(ShaderRecompiler::DenormFlushSkipCompute);
  ShaderRecompiler::Spirv::SetHostFloatControls(saved);
  Require(name, "default flushes", exact >= 2u, "flushes " + std::to_string(exact));
  Require(name, "KYTY_PS_FLUSH=off", skipped == 0u, "flushes " + std::to_string(skipped));
  Require(name, "KYTY_CS_FLUSH=off leaves pixel shaders", compute_only == exact,
          "flushes " + std::to_string(compute_only) + " vs " + std::to_string(exact));
  std::printf("[codegen] %-32s ok (%llu flushes by default)\n", name,
              static_cast<unsigned long long>(exact));
}

// V_RCP/RSQ/SQRT/EXP/LOG_F32 flush denormal inputs to zeros of the same sign ("denormals are
// flushed"; rcp(-0) = -INF, sqrt(-0) = -0). With DenormFlushToZero 32 declared, the shader-side
// flush is left to the host (KYTY_HOST_FTZ_INPUTS); both variants must agree bit for bit, and
// denormal inputs must give the signed-zero results.
void CheckTranscendentalDenormInputs(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenTranscendentalDenormInputs";
  using O = ShaderOpcode;
  const auto saved = ShaderRecompiler::Spirv::GetHostFloatControls();
  auto controls = saved;
  controls.denorm_flush_f32 = vulkan->DenormFlushF32Supported();
  ShaderRecompiler::Spirv::SetHostFloatControls(controls);
  std::vector<std::vector<u32>> inputs(1);
  for (const auto value : SpecialF32()) {
    inputs[0].push_back(value);
  }
  uint64_t state = 0x7f3a1c5bu;
  while (inputs[0].size() < 8192u) {
    const auto r = NextRandom(state);
    // Half denormals of either sign, half arbitrary bit patterns.
    inputs[0].push_back((r & 1u) != 0u ? (r & 0x807fffffu) : NextRandom(state));
  }
  const auto body = [](std::vector<u32> *code) {
    code->push_back(EncodeVop1(0x2a, 10, Vgpr(2))); // v_rcp_f32
    code->push_back(EncodeVop1(0x2e, 11, Vgpr(2))); // v_rsq_f32
    code->push_back(EncodeVop1(0x33, 12, Vgpr(2))); // v_sqrt_f32
    code->push_back(EncodeVop1(0x25, 13, Vgpr(2))); // v_exp_f32
    code->push_back(EncodeVop1(0x27, 14, Vgpr(2))); // v_log_f32
  };
  const auto bulk = MakeBulkCase(name, 1, 5, inputs, body,
                                 {O::V_RCP_F32, O::V_RSQ_F32, O::V_SQRT_F32, O::V_EXP_F32,
                                  O::V_LOG_F32});
  CodegenOptions host = GetCodegenOptions();
  host.host_ftz_inputs = true;
  CodegenOptions shader = host;
  shader.host_ftz_inputs = false;
  const auto fast = RunBulk(vulkan, bulk, host);
  const auto slow = RunBulk(vulkan, bulk, shader);
  static const char *const names[] = {"rcp", "rsq", "sqrt", "exp", "log"};
  // Only denormal inputs have an exact reference; the rest must match the shader-side flush.
  const auto denormal = [&](u32 e) {
    const auto bits = bulk.test.initial[e];
    return (bits & 0x7fffffffu) != 0u && (bits & 0x7fffffffu) < 0x00800000u;
  };
  const auto reference = [&](u32 output, u32 e) -> u32 {
    const auto negative = (bulk.test.initial[e] & 0x80000000u) != 0u;
    switch (output) {
      case 0:
      case 1: return negative ? 0xff800000u : 0x7f800000u;
      case 2: return negative ? 0x80000000u : 0u;
      case 3: return 0x3f800000u;
      default: return 0xff800000u;
    }
  };
  size_t checked = 0;
  for (u32 e = 0; e < bulk.count; e++) {
    checked += denormal(e) ? 1u : 0u;
  }
  CompareBulk(name, bulk, fast, slow, {}, names);
  CompareBulk(name, bulk, fast, slow, reference, names,
              [&](u32, u32 e) { return !denormal(e); });
  const auto flushes = [](const std::vector<u32> &spirv) {
    return CountSpirv(spirv, "OpULessThan");
  };
  ShaderRecompiler::Spirv::SetHostFloatControls(saved);
  std::printf("[codegen] %-32s ok (%u elements, %zu denormal, host FTZ %s, OpULessThan %zu -> %zu)\n",
              name, bulk.count, checked, controls.denorm_flush_f32 ? "declared" : "unavailable",
              flushes(slow.spirv), flushes(fast.spirv));
}

// KYTY_F32_FLUSH_HOST=nvidia start-up self-test (recompiler/F32FlushSelfTest.h,
// presentation/window/f32FlushSelfTest.cpp) on this device: no module with the switch off; the
// module validates, has the float modes the emitter declares here and the host-flush forms; the
// shader's reference equals the IEEE model; on an NVIDIA device the job passes, and the forced
// mismatch (KYTY_F32_FLUSH_SELFTEST=fail) fails it with that one result.
void CheckF32FlushSelfTest(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenF32FlushSelfTest";
  namespace F = ShaderRecompiler::FlushSelfTest;
  const auto saved = ShaderRecompiler::Spirv::GetHostFloatControls();
  const auto controls = vulkan->DeviceFloatControls();
  ShaderRecompiler::Spirv::SetHostFloatControls(controls);
  CodegenOptions off = GetCodegenOptions();
  off.f32_flush_host = false;
  {
    ScopedCodegenOptions scope(off);
    std::string error;
    const auto none = F::EmitModule(&error);
    Require(name, "no module with the switch off", none.empty() && !error.empty(), error);
  }
  CodegenOptions on = off;
  on.f32_flush_host = true;
  on.f32_flush_host_classes = ShaderRecompiler::F32FlushClassAll;
  // The emitter's own flush switches must not change the module (its reference flush is its own).
  on.denorm_flush_skip = ShaderRecompiler::DenormFlushSkipCompute;
  ScopedCodegenOptions scope(on);
  std::string error;
  const auto module = F::EmitModule(&error);
  Require(name, "module", !module.empty(), error);
  {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string messages;
    tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &,
                                 const char *message) { messages += message; });
    Require(name, "module validates", tools.Validate(module), messages);
  }
  const auto count = [&](const char *needle) { return CountSpirv(module, needle); };
  Require(name, "SignedZeroInfNanPreserve 32", count("SignedZeroInfNanPreserve 32") == 1u,
          std::to_string(count("SignedZeroInfNanPreserve 32")));
  Require(name, "DenormFlushToZero 32 as the device declares it",
          count("DenormFlushToZero 32") == (controls.denorm_flush_f32 ? 1u : 0u),
          std::to_string(count("DenormFlushToZero 32")));
  Require(name, "a program without f64", count("Float64") == 0u, "");
  Require(name, "rcp/rsq/exp/log forms",
          count("OpFDiv") == 2u && count("InverseSqrt") == 2u && count("Exp2") == 2u &&
              count("Log2") == 2u,
          "");
  const auto compares = size_t{2} * 2u * F::OperandCount;
  for (const char *opcode : {"OpFOrdEqual ", "OpFUnordEqual ", "OpFOrdNotEqual ", "OpFUnordNotEqual ",
                             "OpFOrdLessThan ", "OpFUnordLessThan ", "OpFOrdGreaterThan ",
                             "OpFUnordGreaterThan ", "OpFOrdLessThanEqual ", "OpFUnordLessThanEqual ",
                             "OpFOrdGreaterThanEqual ", "OpFUnordGreaterThanEqual "}) {
    Require(name, opcode, count(opcode) == compares, std::to_string(count(opcode)));
  }
  const auto properties = vulkan->DeviceProperties();
  const auto memory = vulkan->MemoryProperties();
  const auto run = RunF32FlushSelfTest(vulkan->Device(), memory, vulkan->Queue(),
                                       vulkan->QueueFamily(), false);
  Require(name, "the job ran", run.ran, run.error);
  Require(name, "the shader's reference is the IEEE model", run.reference.Passed(),
          run.reference.first);
  std::printf("[codegen] %-32s self-test: %llu of %llu results differ (%s), first %s\n", name,
              static_cast<unsigned long long>(run.verdict.mismatches),
              static_cast<unsigned long long>(run.verdict.checked), run.verdict.ByOpText().c_str(),
              run.verdict.first.c_str());
  // Diagnostic: the same job with no denormal modes declared (KYTY_SHADER_FLOAT_CONTROLS=0).
  {
    ShaderRecompiler::Spirv::SetHostFloatControls({});
    const auto bare = RunF32FlushSelfTest(vulkan->Device(), memory, vulkan->Queue(),
                                          vulkan->QueueFamily(), false);
    ShaderRecompiler::Spirv::SetHostFloatControls(controls);
    std::printf("[codegen] %-32s self-test without denormal modes: %llu mismatches (%s)\n", name,
                static_cast<unsigned long long>(bare.verdict.mismatches), bare.verdict.ByOpText().c_str());
  }
  // The same inputs through a translated guest program (V_RCP/RSQ/SQRT/EXP/LOG_F32 and
  // V_CMP_LT/EQ_F32 on two VGPRs, KYTY_F32_FLUSH_HOST on): the self-test must agree with it.
  {
    using O = ShaderOpcode;
    const auto inputs = F::Inputs();
    const auto operands = F::Operands();
    std::vector<std::vector<u32>> values(2);
    values[0] = inputs;
    for (u32 i = 0; i < F::InputCount; i++) {
      values[1].push_back(operands[i % F::OperandCount]);
    }
    const auto body = [](std::vector<u32> *code) {
      code->push_back(EncodeVop1(0x2a, 10, Vgpr(2)));           // v_rcp_f32
      code->push_back(EncodeVop1(0x2e, 11, Vgpr(2)));           // v_rsq_f32
      code->push_back(EncodeVop1(0x33, 12, Vgpr(2)));           // v_sqrt_f32
      code->push_back(EncodeVop1(0x25, 13, Vgpr(2)));           // v_exp_f32
      code->push_back(EncodeVop1(0x27, 14, Vgpr(2)));           // v_log_f32
      code->push_back(EncodeVop1(0x01, 20, InlineU32(1)));      // v_mov_b32 v20, 1
      code->push_back(EncodeVopc(0x01, Vgpr(2), 3));            // v_cmp_lt_f32 vcc, v2, v3
      code->push_back(EncodeVop2(0x01, 15, InlineU32(0), 20));  // v_cndmask_b32 v15, 0, v20
      code->push_back(EncodeVopc(0x02, Vgpr(2), 3));            // v_cmp_eq_f32 vcc, v2, v3
      code->push_back(EncodeVop2(0x01, 16, InlineU32(0), 20));  // v_cndmask_b32 v16, 0, v20
    };
    const auto bulk = MakeBulkCase("CodegenF32FlushGuest", 2, 7, values, body,
                                   {O::V_RCP_F32, O::V_RSQ_F32, O::V_SQRT_F32, O::V_EXP_F32,
                                    O::V_LOG_F32, O::V_MOV_B32, O::V_CMP_LT_F32, O::V_CMP_EQ_F32,
                                    O::V_CNDMASK_B32});
    const auto run_guest = [&](uint8_t classes) {
      CodegenOptions guest = on;
      guest.host_ftz_inputs = false;
      guest.denorm_flush_skip = 0;
      guest.f32_flush_host_classes = classes;
      const auto result = RunBulk(vulkan, bulk, guest);
      std::array<u32, 7> counts{};
      for (u32 e = 0; e < bulk.count; e++) {
        const auto x = values[0][e];
        const auto y = values[1][e];
        for (u32 o = 0; o < 7; o++) {
          const auto got = result.words[static_cast<size_t>(2 + o) * bulk.count + e];
          const auto expected =
              o < 5 ? F::ExpectedUnary(static_cast<F::UnaryOp>(o), x)
                    : (F::ExpectedCompare(o == 5 ? F::CompareOp::OrdLessThan : F::CompareOp::OrdEqual, x, y)
                           ? 1u : 0u);
          counts[o] += F::ResultsEqual(got, expected) ? 0u : 1u;
        }
      }
      std::printf("[codegen] %-32s guest program, host flush for %s: differing results rcp %u rsq %u "
                  "sqrt %u exp %u log %u lt %u eq %u of %u\n",
                  name, F::ClassesText(classes).c_str(), counts[0], counts[1], counts[2], counts[3],
                  counts[4], counts[5], counts[6], bulk.count);
      return counts;
    };
    const auto differ = run_guest(F::ClassAll);
    // KYTY_F32_FLUSH_HOST=nvidia's per-class mode: the classes the self-test passes give exactly
    // the flushed results; the others keep the shader-side flush.
    const auto passing = F::PassingClasses(run.verdict);
    const auto exact = run_guest(passing);
    for (u32 o = 0; o < 7; o++) {
      Require(name, "the per-class mode gives the flushed results", exact[o] == 0u,
              "output " + std::to_string(o) + ": " + std::to_string(exact[o]) + " differ");
    }
    for (u32 o = 0; o < 5; o++) {
      Require(name, "guest program and self-test agree on the unary op",
              (differ[o] == 0u) == (run.verdict.by_op[o] == 0u),
              F::UnaryName(static_cast<F::UnaryOp>(o)));
    }
  }
  const bool nvidia = properties.vendorID == 0x10deu;
  if (nvidia && std::getenv("KYTY_TEST_F32_FLUSH_REQUIRE") != nullptr) {
    Require(name, "this NVIDIA device flushes", run.Passed(),
            std::to_string(run.verdict.mismatches) + " mismatches, first " + run.verdict.first);
  }
  const auto forced = RunF32FlushSelfTest(vulkan->Device(), memory, vulkan->Queue(),
                                          vulkan->QueueFamily(), true);
  Require(name, "the forced mismatch fails", forced.ran && !forced.Passed(), forced.verdict.first);
  if (run.Passed()) {
    Require(name, "the forced mismatch is that one result",
            forced.verdict.mismatches == 1u &&
                forced.verdict.first == "rsq(0x80000001) = 0x7fc00000, expected 0xff800000",
            forced.verdict.first);
  }
  ShaderRecompiler::Spirv::SetHostFloatControls(saved);
  std::printf("[codegen] %-32s ok (%s: %llu results, %llu mismatches%s%s; %.1f ms, %zu words)\n",
              name, properties.deviceName.data(),
              static_cast<unsigned long long>(run.verdict.checked),
              static_cast<unsigned long long>(run.verdict.mismatches),
              run.verdict.first.empty() ? "" : ", first ", run.verdict.first.c_str(),
              run.milliseconds, module.size());
}

// KYTY_F32_FLUSH_HOST=nvidia stage B (F32FlushSelfTest.h): the f16 module validates and carries
// the float modes; the f32 job and the f16 block run with the device's modes minus
// DenormPreserve 16 (and the f16 block also with them); the decision those runs give. Prints the
// verdicts; KYTY_TEST_F32_FLUSH_REQUIRE=1 makes "stage B passes" a requirement on NVIDIA.
void CheckF32FlushStageB(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenF32FlushStageB";
  namespace F = ShaderRecompiler::FlushSelfTest;
  const auto saved = ShaderRecompiler::Spirv::GetHostFloatControls();
  const auto declared = vulkan->DeviceFloatControls();
  auto without = declared;
  without.denorm_preserve_f16 = false;
  CodegenOptions on = GetCodegenOptions();
  on.f32_flush_host = true;
  on.f32_flush_host_classes = ShaderRecompiler::F32FlushClassAll;
  ScopedCodegenOptions scope(on);
  const auto memory = vulkan->MemoryProperties();
  const auto device = vulkan->Device();
  {
    ShaderRecompiler::Spirv::SetHostFloatControls(without);
    const auto module = F::EmitF16Module();
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string messages;
    tools.SetMessageConsumer([&](spv_message_level_t, const char *, const spv_position_t &,
                                 const char *message) { messages += message; });
    Require(name, "f16 module validates", tools.Validate(module), messages);
    Require(name, "f16 module forms",
            CountSpirv(module, "UnpackHalf2x16") == 2u && CountSpirv(module, "PackHalf2x16") == 1u &&
                CountSpirv(module, "DenormPreserve 16") == 0u &&
                CountSpirv(module, "SignedZeroInfNanPreserve 32") == 1u,
            "");
  }
  const auto f32_a = [&] {
    ShaderRecompiler::Spirv::SetHostFloatControls(declared);
    return RunF32FlushSelfTest(device, memory, vulkan->Queue(), vulkan->QueueFamily(), false);
  }();
  ShaderRecompiler::Spirv::SetHostFloatControls(without);
  const auto f32_b = RunF32FlushSelfTest(device, memory, vulkan->Queue(), vulkan->QueueFamily(), false);
  const auto f16_b = RunF16SelfTest(device, memory, vulkan->Queue(), vulkan->QueueFamily());
  ShaderRecompiler::Spirv::SetHostFloatControls(declared);
  const auto f16_a = RunF16SelfTest(device, memory, vulkan->Queue(), vulkan->QueueFamily());
  ShaderRecompiler::Spirv::SetHostFloatControls(saved);
  Require(name, "the jobs ran", f32_a.ran && f32_b.ran && f16_a.ran && f16_b.ran,
          f32_a.error + f32_b.error + f16_a.error + f16_b.error);
  const auto line = [&](const char *what, const F32FlushSelfTestRun &run) {
    std::printf("[codegen] %-32s %s: %llu of %llu differ%s%s%s%s (%.1f ms)\n", name, what,
                static_cast<unsigned long long>(run.verdict.mismatches),
                static_cast<unsigned long long>(run.verdict.checked),
                run.verdict.mismatches != 0 ? " (" : "", run.verdict.ByOpText().c_str(),
                run.verdict.mismatches != 0 ? "), first " : "", run.verdict.first.c_str(), run.milliseconds);
  };
  line("stage A f32 job (declared modes)", f32_a);
  line("stage B f32 job (no DenormPreserve 16)", f32_b);
  line("f16 block, no DenormPreserve 16", f16_b);
  line("f16 block, declared modes", f16_a);
  F::DecisionInput in;
  in.stage_a_classes = F::PassingClasses(f32_a.verdict);
  in.preserve16_declared = declared.denorm_preserve_f16;
  in.native_f16_active = ShaderRecompiler::NativeHalfActive();
  in.stage_b_ran = true;
  in.stage_b_classes = F::PassingClasses(f32_b.verdict);
  in.f16_passed = f16_b.verdict.Passed();
  const auto outcome = F::Decide(in);
  std::printf("[codegen] %-32s decision on this device: %s (stage A classes %s, stage B classes %s, f16 %s)\n",
              name,
              outcome == F::Outcome::AllDeclared ? "all classes, modes as declared"
              : outcome == F::Outcome::PerClass  ? "per class, modes as declared"
                                                 : "all classes, no DenormPreserve 16",
              F::ClassesText(in.stage_a_classes).c_str(), F::ClassesText(in.stage_b_classes).c_str(),
              in.f16_passed ? "pass" : "fail");
  if (vulkan->DeviceProperties().vendorID == 0x10deu && std::getenv("KYTY_TEST_F32_FLUSH_REQUIRE") != nullptr) {
    Require(name, "stage B passes on this NVIDIA device", outcome != F::Outcome::PerClass,
            f32_b.verdict.first + " / " + f16_b.verdict.first);
  }
  // A guest program in the stage B outcome (no DenormPreserve 16, every class on) gives the
  // flushed results of rcp/rsq/sqrt/exp/log and v_cmp_lt/eq when stage B passed.
  if (in.stage_b_classes == F::ClassAll) {
    using O = ShaderOpcode;
    const auto inputs = F::Inputs();
    const auto operands = F::Operands();
    std::vector<std::vector<u32>> values(2);
    values[0] = inputs;
    for (u32 i = 0; i < F::InputCount; i++) {
      values[1].push_back(operands[i % F::OperandCount]);
    }
    const auto body = [](std::vector<u32> *code) {
      code->push_back(EncodeVop1(0x2a, 10, Vgpr(2)));
      code->push_back(EncodeVop1(0x2e, 11, Vgpr(2)));
      code->push_back(EncodeVop1(0x33, 12, Vgpr(2)));
      code->push_back(EncodeVop1(0x25, 13, Vgpr(2)));
      code->push_back(EncodeVop1(0x27, 14, Vgpr(2)));
      code->push_back(EncodeVop1(0x01, 20, InlineU32(1)));
      code->push_back(EncodeVopc(0x01, Vgpr(2), 3));
      code->push_back(EncodeVop2(0x01, 15, InlineU32(0), 20));
      code->push_back(EncodeVopc(0x02, Vgpr(2), 3));
      code->push_back(EncodeVop2(0x01, 16, InlineU32(0), 20));
    };
    const auto bulk = MakeBulkCase("CodegenF32FlushStageBGuest", 2, 7, values, body,
                                   {O::V_RCP_F32, O::V_RSQ_F32, O::V_SQRT_F32, O::V_EXP_F32,
                                    O::V_LOG_F32, O::V_MOV_B32, O::V_CMP_LT_F32, O::V_CMP_EQ_F32,
                                    O::V_CNDMASK_B32});
    CodegenOptions guest = on;
    guest.host_ftz_inputs = false;
    guest.denorm_flush_skip = 0;
    guest.f32_flush_host_classes = F::ClassAll | F::ClassNoPreserve16;
    ShaderRecompiler::Spirv::SetHostFloatControls(without);
    const auto result = RunBulk(vulkan, bulk, guest);
    ShaderRecompiler::Spirv::SetHostFloatControls(saved);
    u32 differ = 0;
    for (u32 e = 0; e < bulk.count; e++) {
      for (u32 o = 0; o < 7; o++) {
        const auto got = result.words[static_cast<size_t>(2 + o) * bulk.count + e];
        const auto x = values[0][e];
        const auto y = values[1][e];
        const auto expected =
            o < 5 ? F::ExpectedUnary(static_cast<F::UnaryOp>(o), x)
                  : (F::ExpectedCompare(o == 5 ? F::CompareOp::OrdLessThan : F::CompareOp::OrdEqual, x, y)
                         ? 1u : 0u);
        differ += F::ResultsEqual(got, expected) ? 0u : 1u;
      }
    }
    std::printf("[codegen] %-32s guest program in the stage B outcome: %u of %u results differ\n", name,
                differ, bulk.count * 7u);
    Require(name, "the stage B outcome gives the flushed results", differ == 0u, std::to_string(differ));
  }
}

// Diagnostic for DenormPreserve 16 (declared where the device takes it): V_CVT_F32_F16 of every
// f16 denormal and V_CVT_F16_F32 of f32 values that round to f16 denormals (exact ones and ties),
// with the device's float controls and with DenormPreserve 16 left out, bit-exact against the CPU
// (f16 denormals kept, round to nearest even). Prints the differing counts; fails nothing.
void CheckF16DenormConversions(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenF16DenormConversions";
  using O = ShaderOpcode;
  std::vector<std::vector<u32>> values(1);
  std::vector<u32> expect_f32;  // V_CVT_F32_F16 of input (low half)
  std::vector<u32> expect_f16;  // V_CVT_F16_F32 of the f32 bits in the input
  std::vector<u32> f32_inputs;
  for (u32 sign = 0; sign < 2; sign++) {
    for (u32 m = 1; m < 1024u; m++) {
      const u32 h = (sign << 15u) | m;
      values[0].push_back(h);
      const float v = static_cast<float>(m) * 0x1p-24f;  // exact
      expect_f32.push_back(std::bit_cast<u32>(v) | (sign << 31u));
      // Exact f16 denormal as f32, and the tie above it (k + 0.5) * 2^-24 -> even.
      const u32 tie = (m % 2u) == 0u ? m : m + 1u;
      f32_inputs.push_back(std::bit_cast<u32>(v) | (sign << 31u));
      expect_f16.push_back(h);
      const float t = (static_cast<float>(m) + 0.5f) * 0x1p-24f;
      f32_inputs.push_back(std::bit_cast<u32>(t) | (sign << 31u));
      expect_f16.push_back((sign << 15u) | tie);
    }
  }
  while (values[0].size() % 64u != 0u) {
    values[0].push_back(0u);
    expect_f32.push_back(0u);
  }
  while (f32_inputs.size() % 64u != 0u) {
    f32_inputs.push_back(0u);
    expect_f16.push_back(0u);
  }
  const auto to_f32 = MakeBulkCase(name, 1, 1, values,
                                   [](std::vector<u32> *code) {
                                     code->push_back(EncodeVop1(0x0b, 10, Vgpr(2)));  // v_cvt_f32_f16
                                   },
                                   {O::V_CVT_F32_F16});
  std::vector<std::vector<u32>> f32_values(1, f32_inputs);
  const auto to_f16 = MakeBulkCase(name, 1, 1, f32_values,
                                   [](std::vector<u32> *code) {
                                     code->push_back(EncodeVop1(0x0a, 10, Vgpr(2)));  // v_cvt_f16_f32
                                   },
                                   {O::V_CVT_F16_F32});
  const auto saved = ShaderRecompiler::Spirv::GetHostFloatControls();
  const auto device = vulkan->DeviceFloatControls();
  for (const bool preserve16 : {true, false}) {
    auto controls = device;
    controls.denorm_preserve_f16 = preserve16 && device.denorm_preserve_f16;
    if (!preserve16 && !device.denorm_preserve_f16) {
      continue;
    }
    ShaderRecompiler::Spirv::SetHostFloatControls(controls);
    const auto a = RunBulk(vulkan, to_f32, GetCodegenOptions());
    const auto b = RunBulk(vulkan, to_f16, GetCodegenOptions());
    u32 bad_f32 = 0;
    u32 bad_f16 = 0;
    std::string first;
    for (u32 e = 0; e < to_f32.count; e++) {
      const auto got = a.words[static_cast<size_t>(to_f32.count) + e];
      if (got != expect_f32[e]) {
        if (bad_f32++ == 0u) {
          first = "f32(f16 " + Hex(values[0][e]) + ") = " + Hex(got) + ", expected " + Hex(expect_f32[e]);
        }
      }
    }
    for (u32 e = 0; e < to_f16.count; e++) {
      const auto got = b.words[static_cast<size_t>(to_f16.count) + e] & 0xffffu;
      if (got != expect_f16[e]) {
        if (bad_f16++ == 0u && first.empty()) {
          first = "f16(f32 " + Hex(f32_inputs[e]) + ") = " + Hex(got) + ", expected " + Hex(expect_f16[e]);
        }
      }
    }
    std::printf("[codegen] %-32s DenormPreserve 16 %s: v_cvt_f32_f16 %u of %u differ, v_cvt_f16_f32 %u of "
                "%u differ%s%s\n",
                name, controls.denorm_preserve_f16 ? "declared" : "left out", bad_f32, to_f32.count, bad_f16,
                to_f16.count, first.empty() ? "" : ", first ", first.c_str());
  }
  ShaderRecompiler::Spirv::SetHostFloatControls(saved);
}

// A compressed export to a SNORM16_ABGR target (SPI_SHADER_COL_FORMAT 6) carries
// V_CVT_PKNORM_I16_F32 pairs; they were unpacked as FP16.
void CheckCompressedSnormExport() {
  constexpr const char *name = "CodegenCompressedSnormExport";
  std::vector<u32> code;
  code.push_back(EncodeExp0(1, 0xf, true, true, true)); // exp mrt1 compr done vm
  code.push_back(EncodeExp1(0, 1, 2, 3));
  AppendEnd(&code);
  const auto compile = [&](uint8_t mode) {
    ShaderPixelInputInfo pixel{};
    pixel.target_output_mode[1] = mode;
    return CompilePixelShader(name, code, pixel);
  };
  const auto snorm = compile(6);
  const auto half = compile(4);
  Require(name, "SNORM16 compressed export",
          CountSpirv(snorm, "UnpackSnorm2x16") == 2u && CountSpirv(snorm, "UnpackHalf2x16") == 0u &&
              CountSpirv(half, "UnpackHalf2x16") == 2u,
          "SNORM16: UnpackSnorm2x16 " + std::to_string(CountSpirv(snorm, "UnpackSnorm2x16")) +
              ", UnpackHalf2x16 " + std::to_string(CountSpirv(snorm, "UnpackHalf2x16")));
  std::printf("[codegen] %-32s ok\n", name);
}

// EXEC-masked VGPR merges (IR::EliminateExecSelects, KYTY_EXEC_SELECTS). Region A is entered
// through s_cbranch_execz, region B is a plain EXEC-masked stretch. Both write VGPRs twice under
// one EXEC, keep temporaries and leave values live after EXEC is restored (inactive lanes keep
// their old values). Region B reads lane 5 of a merged VGPR with V_READLANE_B32, which sees the
// old value when lane 5 is inactive. Wave64 runs two lanes per host invocation (wave-uniform
// branches), wave32 one (branches taken per invocation).
void CheckExecSelects(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenExecSelects";
  using O = ShaderOpcode;
  constexpr u32 Count = 4096;
  std::vector<std::vector<u32>> inputs(2);
  uint64_t state = 0x5eed0e5eu;
  for (u32 e = 0; e < Count; e++) {
    inputs[0].push_back(NextRandom(state));
    // Equal pairs take neither region.
    inputs[1].push_back((NextRandom(state) & 7u) == 0u ? inputs[0].back() : NextRandom(state));
  }
  std::string summary;
  for (const u32 wave : {64u, 32u}) {
    const bool w64 = wave == 64u;
    const auto save_exec = [&](u32 sdst) {
      return w64 ? EncodeSop1(0x24, sdst, 106)  // s_and_saveexec_b64 s[sdst:sdst+1], vcc
                 : EncodeSop1(0x3c, sdst, 106); // s_and_saveexec_b32 sdst, vcc_lo
    };
    const auto restore_exec = [&](u32 ssrc) {
      return w64 ? EncodeSop2(0x11, 126, 126, ssrc)  // s_or_b64 exec, exec, s[ssrc:ssrc+1]
                 : EncodeSop2(0x10, 126, 126, ssrc); // s_or_b32 exec_lo, exec_lo, ssrc
    };
    const auto body = [&](std::vector<u32> *code) {
      code->push_back(EncodeVop1(0x01, 10, Vgpr(2))); // v_mov_b32 v10, v2
      code->push_back(EncodeVop1(0x01, 11, Vgpr(3))); // v_mov_b32 v11, v3
      AppendVMovLiteral(code, 12, 0x12345678u);
      AppendVMovLiteral(code, 13, 0x0badf00du);
      AppendVMovLiteral(code, 14, 0x55aa55aau);
      // Region A: x > y.
      code->push_back(EncodeVopc(0xc4, Vgpr(2), 3)); // v_cmp_gt_u32 vcc, v2, v3
      code->push_back(save_exec(60));
      const auto skip = code->size();
      code->push_back(0);                                  // s_cbranch_execz, patched below
      code->push_back(EncodeVop2(0x25, 20, Vgpr(2), 3));   // v_add_nc_u32 v20, v2, v3
      code->push_back(EncodeVop2(0x1d, 20, Vgpr(20), 2));  // v_xor_b32 v20, v20, v2
      AppendVop3(code, 0x169, 21, Vgpr(20), Vgpr(3));      // v_mul_lo_u32 v21, v20, v3
      code->push_back(EncodeVop2(0x26, 10, Vgpr(21), 2));  // v_sub_nc_u32 v10, v21, v2
      code->push_back(EncodeVop2(0x25, 21, Vgpr(21), 20)); // v_add_nc_u32 v21, v21, v20
      code->push_back(EncodeVop2(0x1d, 11, Vgpr(21), 11)); // v_xor_b32 v11, v21, v11
      (*code)[skip] = EncodeSopp(0x08, static_cast<u32>(code->size() - skip - 1));
      code->push_back(restore_exec(60));
      // Region B: x < y, no branch.
      code->push_back(EncodeVopc(0xc1, Vgpr(2), 3)); // v_cmp_lt_u32 vcc, v2, v3
      code->push_back(save_exec(62));
      code->push_back(EncodeVop2(0x25, 13, InlineU32(7), 2));   // v_add_nc_u32 v13, 7, v2
      code->push_back(EncodeVop2(0x1d, 13, Vgpr(13), 3));       // v_xor_b32 v13, v13, v3
      AppendVop3(code, 0x360, 70, Vgpr(13), InlineU32(5));      // v_readlane_b32 s70, v13, 5
      code->push_back(EncodeVop1(0x01, 12, 70));                // v_mov_b32 v12, s70
      code->push_back(EncodeVop2(0x25, 22, Vgpr(13), 2));       // v_add_nc_u32 v22, v13, v2
      code->push_back(EncodeVop2(0x1a, 14, InlineU32(3), 22));  // v_lshlrev_b32 v14, 3, v22
      code->push_back(restore_exec(62));
    };
    auto bulk = MakeBulkCase(
        name, 2, 5, inputs, body,
        {O::V_MOV_B32, O::V_CMP_GT_U32, O::V_CMP_LT_U32,
         w64 ? O::S_AND_SAVEEXEC_B64 : O::S_AND_SAVEEXEC_B32, O::S_CBRANCH_EXECZ,
         O::V_ADD_NC_U32, O::V_XOR_B32, O::V_MUL_LO_U32, O::V_SUB_NC_U32,
         w64 ? O::S_OR_B64 : O::S_OR_B32, O::V_READLANE_B32, O::V_LSHLREV_B32});
    bulk.test.compute_info.wave_size = wave;
    CodegenOptions merged = GetCodegenOptions();
    merged.exec_selects   = true;
    CodegenOptions legacy = merged;
    legacy.exec_selects   = false;
    const auto fast = RunBulk(vulkan, bulk, merged);
    const auto slow = RunBulk(vulkan, bulk, legacy);
    const auto region_b = [&](u32 i) { return inputs[0][i] < inputs[1][i]; };
    const auto t13 = [&](u32 i) { return (inputs[0][i] + 7u) ^ inputs[1][i]; };
    const auto reference = [&](u32 output, u32 e) -> u32 {
      const u32 x = inputs[0][e];
      const u32 y = inputs[1][e];
      const u32 t20 = (x + y) ^ x;
      const u32 t21 = t20 * y;
      // Lane 5 of the wave that runs element e (the workgroup's 64 threads are 64/wave waves).
      const u32 source = e - e % wave + 5u;
      switch (output) {
        case 0: return x > y ? t21 - x : x;
        case 1: return x > y ? (t21 + t20) ^ y : y;
        case 2: return region_b(e) ? (region_b(source) ? t13(source) : 0x0badf00du) : 0x12345678u;
        case 3: return region_b(e) ? t13(e) : 0x0badf00du;
        default: return region_b(e) ? (t13(e) + x) << 3u : 0x55aa55aau;
      }
    };
    static const char *const names[] = {"region A v10", "region A v11", "readlane v12",
                                        "region B v13", "region B v14"};
    CompareBulk(name, bulk, fast, slow, reference, names);
    const auto fast_selects = CountSpirv(fast.spirv, "OpSelect");
    const auto slow_selects = CountSpirv(slow.spirv, "OpSelect");
    Require(name, "select count", fast_selects < slow_selects,
            "wave" + std::to_string(wave) + ": OpSelect " + std::to_string(slow_selects) +
                " -> " + std::to_string(fast_selects));
    summary += (summary.empty() ? "" : ", ") + std::string("wave") + std::to_string(wave) +
               " OpSelect " + std::to_string(slow_selects) + " -> " +
               std::to_string(fast_selects);
  }
  std::printf("[codegen] %-32s ok (%u elements, %s)\n", name, Count, summary.c_str());
}

// --exec-selects-bench: GPU time of an EXEC region covering half the lanes (a straight-line
// chain of 4096 integer VALU ops over eight VGPR temporaries, folded into one value that is
// stored only when it matches a magic number) with KYTY_EXEC_SELECTS on and off, in wave64 (two
// lanes per host invocation) and wave32. Every workgroup reads the same 64 dwords, so memory
// stays out of the timing. Timing only: CodegenExecSelects checks the results.
void BenchExecSelects(VulkanHarness *vulkan) {
  constexpr const char *name = "ExecSelectsBench";
  constexpr u32 Ops = 4096;
  constexpr u32 Groups = 8192;
  constexpr u32 Repeats = 20;
  constexpr u32 Rounds = 6;
  for (const u32 wave : {64u, 32u}) {
    const bool w64 = wave == 64u;
    TestCase test;
    test.name = name;
    auto &code = test.code;
    code.push_back(EncodeVop2(0x1a, 1, InlineU32(2), 0)); // v_lshlrev_b32 v1, 2, v0
    code.push_back(EncodeMubuf0(0x0c));                   // buffer_load_dword v2, v1, s[0:3], 0
    code.push_back(EncodeMubuf1(2, 0, 1));
    code.push_back(EncodeSopp(0x0c, 0)); // s_waitcnt 0
    for (u32 r = 0; r < 8; r++) {
      code.push_back(EncodeVop2(0x25, 20 + r, InlineU32(r + 1), 2)); // v_add_nc_u32
    }
    code.push_back(EncodeVop2(0x1b, 3, InlineU32(1), 2)); // v_and_b32 v3, 1, v2
    code.push_back(EncodeVopc(0xc5, InlineU32(0), 3));    // v_cmp_ne_u32 vcc, 0, v3
    code.push_back(w64 ? EncodeSop1(0x24, 60, 106) : EncodeSop1(0x3c, 60, 106));
    const auto skip = code.size();
    code.push_back(0); // s_cbranch_execz, patched below
    static constexpr u32 opcodes[] = {0x25, 0x1d, 0x26, 0x1d}; // add, xor, sub, xor
    for (u32 i = 0; i < Ops; i++) {
      code.push_back(EncodeVop2(opcodes[i % 4u], 20 + i % 8u, Vgpr(20 + (i * 3u + 1u) % 8u),
                                20 + (i * 5u + 2u) % 8u));
    }
    for (u32 r = 1; r < 8; r++) {
      code.push_back(EncodeVop2(0x1d, 20, Vgpr(20), 20 + r)); // v_xor_b32 v20, v20, v(20+r)
    }
    // Store only lanes whose result is a (practically never seen) magic value: every lane has to
    // compute the chain, and the host compiler cannot sink it into the store.
    code.push_back(EncodeVopc(0xc2, 255, 20)); // v_cmp_eq_u32 vcc, lit, v20
    code.push_back(0x13579bdfu);
    code.push_back(w64 ? EncodeSop1(0x24, 62, 106) : EncodeSop1(0x3c, 62, 106));
    code.push_back(EncodeMubuf0(0x1c, 256)); // buffer_store_dword v20, v1, s[0:3], 0 offset:256
    code.push_back(EncodeMubuf1(20, 0, 1));
    code.push_back(w64 ? EncodeSop2(0x11, 126, 126, 62) : EncodeSop2(0x10, 126, 126, 62));
    code[skip] = EncodeSopp(0x08, static_cast<u32>(code.size() - skip - 1));
    code.push_back(w64 ? EncodeSop2(0x11, 126, 126, 60) : EncodeSop2(0x10, 126, 126, 60));
    AppendEnd(&code);
    uint64_t state = 0xbe7c4u + wave;
    test.initial.resize(128);
    for (u32 i = 0; i < 64; i++) {
      test.initial[i] = NextRandom(state);
    }
    test.compute_info.threads_num[0] = 64;
    test.compute_info.threads_num[1] = 1;
    test.compute_info.threads_num[2] = 1;
    test.compute_info.thread_ids_num = 1;
    test.compute_info.wave_size = wave;
    test.has_compute_info = true;
    test.dispatch_x = Groups;
    struct Variant {
      CompiledShader compiled;
      std::vector<u32> result;
      double best_us = 1e30;
    };
    std::array<Variant, 2> variants;
    for (u32 v = 0; v < 2; v++) {
      CodegenOptions options = GetCodegenOptions();
      options.exec_selects = v == 1;
      ScopedCodegenOptions scope(options);
      variants[v].compiled = CompileCase(test, vulkan->SubgroupSize());
    }
    // The first round only raises the clocks.
    for (u32 round = 0; round <= Rounds; round++) {
      for (auto &variant : variants) {
        auto buffer = vulkan->CreateStorageBuffer(name, test.initial, test.initial.size());
        double us = 0.0;
        vulkan->Dispatch(test, variant.compiled, buffer, nullptr, nullptr, nullptr, nullptr,
                         nullptr, {}, Repeats, &us);
        if (round != 0) {
          variant.best_us = std::min(variant.best_us, us);
        }
        variant.result = vulkan->ReadBuffer(name, buffer, test.initial.size());
        vulkan->DestroyBuffer(&buffer);
      }
    }
    RequireEqualWords(name, "results", variants[0].result, variants[1].result);
    std::printf("[bench]   %-32s wave%u: OpSelect %zu -> %zu, SPIR-V %zu -> %zu words, "
                "%.1f us -> %.1f us per dispatch (%u groups)\n",
                name, wave, CountSpirv(variants[0].compiled.spirv, "OpSelect"),
                CountSpirv(variants[1].compiled.spirv, "OpSelect"),
                variants[0].compiled.spirv.size(), variants[1].compiled.spirv.size(),
                variants[0].best_us, variants[1].best_us, Groups);
  }
}

// --corpus-pipeline-times <dir>: for every compute shader dump in `dir` (CompileCorpusShader),
// SPIR-V size and vkCreateComputePipelines time with KYTY_EXEC_SELECTS off and on, best of three
// cache-proof compiles each, as CSV plus totals.
int CorpusPipelineTimes(VulkanHarness *vulkan, const char *directory) {
  std::vector<std::filesystem::path> files;
  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    const auto file = entry.path().filename().string();
    if (entry.path().extension() == ".bin" && file.find("_cs_") != std::string::npos) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  std::printf("shader,words_off,words_on,ms_off,ms_on\n");
  double total_off = 0.0;
  double total_on = 0.0;
  size_t words_off = 0;
  size_t words_on = 0;
  u32 nonce = 1;
  for (const auto &file : files) {
    const auto path = file.string();
    std::array<ShaderRecompiler::CompileResult, 2> results;
    bool ok = true;
    for (u32 v = 0; v < 2 && ok; v++) {
      CodegenOptions options = GetCodegenOptions();
      options.exec_selects = v == 1;
      ScopedCodegenOptions scope(options);
      ok = std::string(CompileCorpusShader(path.c_str(), &results[v])) == "ok";
    }
    if (!ok) {
      continue;
    }
    std::array<double, 2> best{1e30, 1e30};
    for (u32 round = 0; round < 3; round++) {
      for (u32 v = 0; v < 2; v++) {
        const auto index = (round % 2u == 0u) ? v : 1u - v;
        best[index] = std::min(best[index],
                               vulkan->TimeComputePipeline("CorpusPipelineTimes",
                                                           results[index].spirv,
                                                           results[index].program.bindings,
                                                           nonce++));
      }
    }
    std::printf("%s,%zu,%zu,%.2f,%.2f\n", file.filename().string().c_str(),
                results[0].spirv.size(), results[1].spirv.size(), best[0], best[1]);
    std::fflush(stdout);
    total_off += best[0];
    total_on += best[1];
    words_off += results[0].spirv.size();
    words_on += results[1].spirv.size();
  }
  std::printf("total,%zu,%zu,%.1f,%.1f\n", words_off, words_on, total_off, total_on);
  return 0;
}

// KYTY_DPP_SKIP_INACTIVE: a DPP source lane EXEC disables leaves the receiving lane unchanged
// (default); with the switch off the lane reads zero from it, the behaviour before.
void CheckDppInactiveSource(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenDppInactiveSource";
  const auto test = VectorDppInactiveSourcePreservesDestination();
  std::vector<u32> skip_words;
  std::vector<u32> zero_words;
  {
    CodegenOptions options = GetCodegenOptions();
    options.dpp_skip_inactive = true;
    ScopedCodegenOptions scope(options);
    skip_words = RunCollect(vulkan, test, test.expected.size());
  }
  {
    CodegenOptions options = GetCodegenOptions();
    options.dpp_skip_inactive = false;
    ScopedCodegenOptions scope(options);
    zero_words = RunCollect(vulkan, test, test.expected.size());
  }
  RequireEqualWords(name, "inactive source kept", test.expected, skip_words);
  auto legacy = test.expected;
  legacy[4] = 100u; // lane 4 read zero from the disabled lane 3
  RequireEqualWords(name, "KYTY_DPP_SKIP_INACTIVE=0", legacy, zero_words);
  std::printf("[codegen] %-32s ok\n", name);
}

// KYTY_LANE_REDUCTIONS: a v_max_u32 DPP row_shr scan (1, 2, 4, 8) read by v_readlane at each
// row's last lane (wave32: 15, 31) or, after V_PERMLANEX16, at each row pair's last lane (wave64:
// 31, 63) on full waves of distinct data, with every lane enabled by s_mov exec, -1 or as EXEC =
// m | ~m (s_orn2_saveexec of a copy of EXEC). The native subgroup reduction and the emulated scan
// must both give each row's (pair's) maximum.
void CheckLaneReductions(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenLaneReductions";
  for (const u32 variant : {0u, 1u, 2u, 3u}) {
    const u32 wave = (variant & 1u) != 0u ? 64u : 32u;
    const bool complement = (variant & 2u) != 0u;
    TestCase test;
    test.name = name;
    auto &code = test.code;
    code.push_back(EncodeVop2(0x1a, 3, InlineU32(2), 0)); // v_lshlrev_b32 v3, 2, v0
    AppendBufferLoadDword(&code, 1, 3);                  // v1 = data[lane]
    code.push_back(EncodeSopp(0x0c, 0));                 // s_waitcnt 0
    if (!complement) {
      code.push_back(wave == 64u ? EncodeSop1(0x04, 126, 193)  // s_mov_b64 exec, -1
                                 : EncodeSMovB32(126, 193));   // s_mov_b32 exec_lo, -1
    } else if (wave == 64u) {
      code.push_back(EncodeSop1(0x04, 10, 126)); // s_mov_b64 s[10:11], exec
      code.push_back(EncodeSop1(0x28, 12, 10));  // s_orn2_saveexec_b64 s[12:13], s[10:11]
    } else {
      code.push_back(EncodeSMovB32(10, 126));    // s_mov_b32 s10, exec_lo
      code.push_back(EncodeSop1(0x40, 12, 10));  // s_orn2_saveexec_b32 s12, s10
    }
    for (const u32 shift : {1u, 2u, 4u, 8u}) {
      code.push_back(EncodeVop2(0x14, 1, 250, 1));       // v_max_u32 v1, v1 row_shr:n, v1
      code.push_back(EncodeVop2Dpp(1, 0x110u + shift));
    }
    if (wave == 64u) {
      AppendVop3(&code, 0x378, 2, Vgpr(1), 193, 193);   // v_permlanex16_b32 v2, v1, -1, -1
      code.push_back(EncodeVop2(0x14, 1, Vgpr(1), 2));   // v_max_u32 v1, v1, v2
    }
    const u32 first = wave == 64u ? 31u : 15u;
    const u32 second = wave == 64u ? 63u : 31u;
    AppendVop3(&code, 0x360, 20, Vgpr(1), InlineU32(first));  // v_readlane_b32 s20, v1, first
    AppendVop3(&code, 0x360, 21, Vgpr(1), InlineU32(second)); // v_readlane_b32 s21, v1, second
    AppendStoreSgpr(&code, 20, wave);
    AppendStoreSgpr(&code, 21, wave + 1u);
    AppendEnd(&code);
    test.initial.resize(wave + 2u);
    for (u32 lane = 0; lane < wave; lane++) {
      test.initial[lane] = (lane * 2654435761u) ^ 0x5a5a5a5au;
    }
    test.compute_info.threads_num[0] = wave;
    test.compute_info.threads_num[1] = 1;
    test.compute_info.threads_num[2] = 1;
    test.compute_info.thread_ids_num = 1;
    test.compute_info.wave_size = wave;
    test.has_compute_info = true;
    const u32 span = wave == 64u ? 32u : 16u;
    std::vector<u32> expected = test.initial;
    expected[wave] = *std::max_element(test.initial.begin(), test.initial.begin() + span);
    expected[wave + 1u] = *std::max_element(test.initial.begin() + (wave - span),
                                            test.initial.begin() + wave);
    std::vector<u32> native_spirv;
    std::vector<u32> scan_spirv;
    std::vector<u32> native_words;
    std::vector<u32> scan_words;
    {
      CodegenOptions options = GetCodegenOptions();
      options.lane_reductions = true;
      ScopedCodegenOptions scope(options);
      native_words = RunCollect(vulkan, test, wave + 2u, &native_spirv);
    }
    {
      CodegenOptions options = GetCodegenOptions();
      options.lane_reductions = false;
      ScopedCodegenOptions scope(options);
      scan_words = RunCollect(vulkan, test, wave + 2u, &scan_spirv);
    }
    const auto stage = wave == 64u ? (complement ? "wave64 row pairs, EXEC = m | ~m"
                                                 : "wave64 row pairs")
                                   : (complement ? "wave32 rows, EXEC = m | ~m" : "wave32 rows");
    if (CountSpirv(native_spirv, "OpGroupNonUniformUMax") != 2u) {
      // The IR the match saw, for the failure below.
      CodegenOptions options = GetCodegenOptions();
      options.lane_reductions = true;
      ScopedCodegenOptions scope(options);
      std::fprintf(stderr, "%s (%s): IR\n%s\n", name, stage,
                   ShaderRecompiler::IR::ProgramToString(
                       CompileCase(test, vulkan->SubgroupSize()).program)
                       .c_str());
    }
    Require(name, stage,
            CountSpirv(native_spirv, "OpGroupNonUniformUMax") == 2u &&
                CountSpirv(scan_spirv, "OpGroupNonUniformUMax") == 0u,
            "the row scan reads did not become two native reductions, or did with the switch off");
    RequireEqualWords(name, stage, expected, native_words);
    RequireEqualWords(name, stage, expected, scan_words);
  }
  std::printf("[codegen] %-32s ok (wave32 rows, wave64 row pairs; both EXEC forms)\n", name);
}

// KYTY_DISPATCHER_CAP: a guest loop that never ends in an irreducible CFG (the dispatcher
// fallback; entry -> B or A, A -> C, C -> A or B, B -> C, never S_ENDPGM) after a store of 7.
// With the cap the invocation leaves the dispatcher loop and the dispatch completes; codegen with
// and without the cap. Without a cap this dispatch would never end, so the GPU part needs one.
void CheckDispatcherCap(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenDispatcherCap";
  using O = ShaderOpcode;
  TestCase test;
  test.name = name;
  auto &code = test.code;
  AppendVMovU32(&code, 2, 7);
  AppendStoreVgpr(&code, 2, 0);
  code.push_back(EncodeSopp(0x05, 2));       // entry -> B, fallthrough A
  code.push_back(EncodeSopp(0x02, 0));       // A -> C
  code.push_back(EncodeSopp(0x05, 0xfffeu)); // C -> A, fallthrough B
  code.push_back(EncodeSopp(0x02, 0xfffeu)); // B -> C
  AppendEnd(&code);
  test.initial = {0u};
  test.opcodes = {O::V_MOV_B32, O::BUFFER_STORE_DWORD, O::S_CBRANCH_SCC1, O::S_BRANCH,
                  O::S_ENDPGM};
  const auto compile = [&](uint32_t cap) {
    CodegenOptions options = GetCodegenOptions();
    options.dispatcher_cap = cap;
    ScopedCodegenOptions scope(options);
    return CompileCase(test, vulkan->SubgroupSize());
  };
  const auto capped = compile(4096u);
  const auto uncapped = compile(0u);
  Require(name, "dispatcher", capped.program.dispatcher_fallback,
          "the irreducible loop did not take the dispatcher fallback");
  Require(name, "cap",
          CountSpirv(capped.spirv, "OpUGreaterThanEqual") ==
                  CountSpirv(uncapped.spirv, "OpUGreaterThanEqual") + 1u &&
              CountSpirv(capped.spirv, "OpLogicalOr") ==
                  CountSpirv(uncapped.spirv, "OpLogicalOr") + 1u,
          "the dispatcher loop does not leave at KYTY_DISPATCHER_CAP");
  if (GetCodegenOptions().dispatcher_cap == 0u) {
    std::printf("[codegen] %-32s ok (codegen only: KYTY_DISPATCHER_CAP=0)\n", name);
    return;
  }
  const auto words = RunCollect(vulkan, test, 1);
  Require(name, "endless loop ends", words[0] == 7u,
          "the dispatch with an endless dispatcher loop did not complete with its store");
  std::printf("[codegen] %-32s ok (an endless dispatcher loop ended after %u transitions)\n",
              name, GetCodegenOptions().dispatcher_cap);
}

// The Scope operand of every OpReadClockKHR (Device 1, Subgroup 3), from the 32-bit constants.
std::vector<u32> ReadClockScopes(const std::vector<u32> &spirv) {
  constexpr u32 OpConstant = 43;
  constexpr u32 OpReadClockKHR = 5056;
  std::unordered_map<u32, u32> constants;
  std::vector<u32> scope_ids;
  for (size_t at = 5; at < spirv.size();) {
    const u32 words = spirv[at] >> 16u;
    const u32 opcode = spirv[at] & 0xffffu;
    if (words == 0 || at + words > spirv.size()) {
      break;
    }
    if (opcode == OpConstant && words == 4) {
      constants[spirv[at + 2]] = spirv[at + 3];
    } else if (opcode == OpReadClockKHR && words == 4) {
      scope_ids.push_back(spirv[at + 3]);
    }
    at += words;
  }
  std::vector<u32> scopes;
  for (const auto id : scope_ids) {
    const auto found = constants.find(id);
    scopes.push_back(found != constants.end() ? found->second : UINT32_MAX);
  }
  return scopes;
}

// S_MEMREALTIME (KYTY_REALTIME_CLOCK, default on): a guest spin-wait timed by it. t0 = clock; the
// loop reads the clock until it is 1000 ticks (10 us at 100 MHz) past t0 or 2^24 reads were made,
// then stores whether it advanced, the ticks and the reads. Codegen for every host clock (CPU),
// then the wait on the GPU: with the device's clock it ends by time; with KYTY_REALTIME_CLOCK=0
// (the placeholder UINT64_MAX) the clock never advances and only the read limit ends it.
void CheckRealtimeClock(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenRealtimeClock";
  namespace Spirv = ShaderRecompiler::Spirv;
  using O = ShaderOpcode;
  Require(name, "shift",
          Spirv::RealtimeClockShift(1.0) == 3 && Spirv::RealtimeClockShift(10.0) == 0 &&
              Spirv::RealtimeClockShift(0.5) == 4 && Spirv::RealtimeClockShift(52.083) == -2 &&
              Spirv::RealtimeClockShift(83.333) == -3 && Spirv::RealtimeClockShift(0.0) == 3,
          "unexpected shift toward the 100 MHz S_MEMREALTIME rate");

  TestCase test;
  test.name = name;
  auto &code = test.code;
  code.push_back(EncodeSmem0(0x25, 12));            // s_memrealtime s[12:13]
  code.push_back(EncodeSmem1(0, 0x7d));
  code.push_back(EncodeSopp(0x0c, 0));              // s_waitcnt 0
  code.push_back(EncodeSMovB32(20, InlineU32(0)));  // s_mov_b32 s20, 0
  const size_t loop = code.size();
  code.push_back(EncodeSmem0(0x25, 14));            // s_memrealtime s[14:15]
  code.push_back(EncodeSmem1(0, 0x7d));
  code.push_back(EncodeSopp(0x0c, 0));              // s_waitcnt 0
  code.push_back(EncodeSop2(0x01, 16, 14, 12));     // s_sub_u32 s16, s14, s12
  code.push_back(EncodeSop2(0x05, 17, 15, 13));     // s_subb_u32 s17, s15, s13
  code.push_back(EncodeSop2(0x00, 20, 20, InlineU32(1))); // s_add_u32 s20, s20, 1
  code.push_back(EncodeSopc(0x07, 17, InlineU32(0))); // s_cmp_lg_u32 s17, 0
  const size_t exit_high = code.size();
  code.push_back(0);                                // s_cbranch_scc1 done
  code.push_back(EncodeSopc(0x09, 16, 255));        // s_cmp_ge_u32 s16, 1000
  code.push_back(1000u);
  const size_t exit_low = code.size();
  code.push_back(0);                                // s_cbranch_scc1 done
  code.push_back(EncodeSopc(0x0a, 20, 255));        // s_cmp_lt_u32 s20, 1 << 24
  code.push_back(1u << 24u);
  const size_t back = code.size();
  code.push_back(0);                                // s_cbranch_scc1 loop
  const size_t done = code.size();
  const auto branch = [&](size_t at, size_t target) {
    const auto offset = static_cast<int32_t>(target) - static_cast<int32_t>(at + 1);
    code[at] = EncodeSopp(0x05, static_cast<u32>(offset) & 0xffffu);
  };
  branch(exit_high, done);
  branch(exit_low, done);
  branch(back, loop);
  code.push_back(EncodeSopc(0x09, 16, 255));        // s_cmp_ge_u32 s16, 1000
  code.push_back(1000u);
  code.push_back(EncodeSop2(0x0a, 22, InlineU32(1), InlineU32(0))); // s_cselect_b32 s22, 1, 0
  code.push_back(EncodeSopc(0x07, 17, InlineU32(0)));               // s_cmp_lg_u32 s17, 0
  code.push_back(EncodeSop2(0x0a, 23, InlineU32(1), InlineU32(0))); // s_cselect_b32 s23, 1, 0
  code.push_back(EncodeSop2(0x10, 22, 22, 23));                     // s_or_b32 s22, s22, s23
  AppendStoreSgpr(&code, 22, 0);
  AppendStoreSgpr(&code, 16, 1);
  AppendStoreSgpr(&code, 20, 2);
  AppendEnd(&code);
  test.initial = {0u, 0u, 0u};
  test.opcodes = {O::S_MEMREALTIME, O::S_WAITCNT,     O::S_MOV_B32,      O::S_SUB_U32,
                  O::S_SUBB_U32,    O::S_ADD_U32,     O::S_CMP_LG_U32,   O::S_CBRANCH_SCC1,
                  O::S_CMP_GE_U32,  O::S_CMP_LT_U32,  O::S_CSELECT_B32,  O::S_OR_B32,
                  O::V_MOV_B32,     O::BUFFER_STORE_DWORD, O::S_ENDPGM};

  // Codegen: the clock read for each host clock, with the shift toward 100 MHz.
  const auto saved_clock = Spirv::GetHostShaderClock();
  const auto compile = [&](Spirv::HostShaderClock clock, bool realtime_clock) {
    CodegenOptions options = GetCodegenOptions();
    options.realtime_clock = realtime_clock;
    ScopedCodegenOptions scope(options);
    Spirv::SetHostShaderClock(clock);
    auto spirv = CompileCase(test, 32).spirv;
    Spirv::SetHostShaderClock(saved_clock);
    ValidateSpirv(name, spirv);
    return spirv;
  };
  using Scope = Spirv::HostClockScope;
  const auto device3 = compile({Scope::Device, 3}, true);
  const auto device0 = compile({Scope::Device, 0}, true);
  const auto left2 = compile({Scope::Device, -2}, true);
  const auto subgroup = compile({Scope::Subgroup, 3}, true);
  const auto none = compile({Scope::None, 3}, true);
  const auto off = compile({Scope::Device, 3}, false);
  const auto right_shifts = [](const std::vector<u32> &spirv) {
    return CountSpirv(spirv, "OpShiftRightLogical");
  };
  const auto left_shifts = [](const std::vector<u32> &spirv) {
    return CountSpirv(spirv, "OpShiftLeftLogical");
  };
  Require(name, "device clock",
          ReadClockScopes(device3) == std::vector<u32>{1u, 1u} &&
              CountSpirv(device3, "OpGroupNonUniformBroadcastFirst") ==
                  CountSpirv(off, "OpGroupNonUniformBroadcastFirst") + 2u &&
              CountSpirv(device3, "ShaderClockKHR") >= 1u &&
              ReadClockScopes(device0) == std::vector<u32>{1u, 1u},
          "S_MEMREALTIME did not read the device clock once per instruction, wave-uniform");
  Require(name, "shift",
          right_shifts(device3) == right_shifts(device0) + 4u &&
              left_shifts(device3) == left_shifts(device0) + 2u &&
              right_shifts(left2) == right_shifts(device0) + 2u &&
              left_shifts(left2) == left_shifts(device0) + 4u,
          "the clock was not scaled by the host clock's shift");
  Require(name, "subgroup clock", ReadClockScopes(subgroup) == std::vector<u32>{3u, 3u},
          "the subgroup-clock fallback did not read at subgroup scope");
  Require(name, "placeholder",
          ReadClockScopes(none).empty() && CountSpirv(none, "ShaderClockKHR") == 0u &&
              ReadClockScopes(off).empty() && CountSpirv(off, "ShaderClockKHR") == 0u,
          "no host clock or KYTY_REALTIME_CLOCK=0 still read a clock");

  // GPU: the wait ends by time with this device's clock; KYTY_REALTIME_CLOCK=0 never advances.
  const auto host = Spirv::GetHostShaderClock();
  std::vector<u32> clock_words;
  {
    CodegenOptions options = GetCodegenOptions();
    options.realtime_clock = true;
    ScopedCodegenOptions scope(options);
    clock_words = RunCollect(vulkan, test, 3);
  }
  std::vector<u32> placeholder_words;
  {
    CodegenOptions options = GetCodegenOptions();
    options.realtime_clock = false;
    ScopedCodegenOptions scope(options);
    placeholder_words = RunCollect(vulkan, test, 3);
  }
  Require(name, "placeholder wait",
          placeholder_words[0] == 0u && placeholder_words[1] == 0u &&
              placeholder_words[2] == (1u << 24u),
          "with the placeholder the clock advanced or the read limit did not end the wait");
  if (host.scope == Scope::None) {
    Require(name, "no host clock", clock_words == placeholder_words,
            "without a host clock S_MEMREALTIME did not keep the placeholder");
  } else {
    Require(name, "clock wait",
            clock_words[0] == 1u && clock_words[2] < (1u << 24u),
            "S_MEMREALTIME did not advance 1000 ticks within 2^24 reads");
  }
  std::printf("[codegen] %-32s ok (%s clock, shift %d: 1000 ticks after %u reads, %u ticks)\n",
              name,
              host.scope == Scope::Device     ? "device"
              : host.scope == Scope::Subgroup ? "subgroup"
                                              : "no",
              host.shift, clock_words[2], clock_words[1]);
}

// KYTY_LANE2_SHARE: a wave64 program (two lanes per host invocation on a 32-wide host) with a
// scalar inner loop whose counter and running sum are the same in both halves, feeding per-lane
// VALU work and a uniform output. Results must match with and without the sharing and the CPU
// reference, and the shared module must be smaller.
void CheckLane2Share(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenLane2Share";
  using O = ShaderOpcode;
  constexpr u32 Count = 2048;
  constexpr u32 Seed = 0x9e3779b9u;
  std::vector<std::vector<u32>> inputs(2);
  uint64_t state = 0x1a2e5aa5u;
  for (u32 e = 0; e < Count; e++) {
    inputs[0].push_back(NextRandom(state));
    inputs[1].push_back(NextRandom(state));
  }
  const auto body = [&](std::vector<u32> *code) {
    code->push_back(EncodeVop1(0x01, 10, Vgpr(2)));     // v_mov_b32 v10, v2
    code->push_back(EncodeSMovB32(70, InlineU32(3)));   // s_mov_b32 s70, 3
    code->push_back(EncodeSMovB32(71, 255));            // s_mov_b32 s71, lit
    code->push_back(Seed);
    const auto inner = static_cast<u32>(code->size());
    code->push_back(EncodeSop2(0x00, 71, 71, 70));      // s_add_u32 s71, s71, s70
    code->push_back(EncodeVop2(0x25, 10, 71, 10));      // v_add_nc_u32 v10, s71, v10
    code->push_back(EncodeVop2(0x1d, 10, Vgpr(10), 3)); // v_xor_b32 v10, v10, v3
    code->push_back(EncodeSop2(0x01, 70, 70, InlineU32(1))); // s_sub_u32 s70, s70, 1
    code->push_back(EncodeSopc(0x07, 70, InlineU32(0)));      // s_cmp_lg_u32 s70, 0
    const auto branch = static_cast<int32_t>(inner) - static_cast<int32_t>(code->size()) - 1;
    code->push_back(EncodeSopp(0x05, static_cast<u32>(branch) & 0xffffu)); // s_cbranch_scc1
    code->push_back(EncodeVop1(0x01, 11, 71));          // v_mov_b32 v11, s71
  };
  auto bulk = MakeBulkCase(name, 2, 2, inputs, body,
                           {O::V_MOV_B32, O::S_SUB_U32, O::S_CMP_LG_U32, O::V_ADD_NC_U32,
                            O::V_XOR_B32});
  CodegenOptions shared = GetCodegenOptions();
  shared.lane2_share = true;
  shared.wave64_warps = false;
  shared.lane_local = false;
  CodegenOptions legacy = shared;
  legacy.lane2_share = false;
  const auto fast = RunBulk(vulkan, bulk, shared);
  const auto slow = RunBulk(vulkan, bulk, legacy);
  const auto reference = [&](u32 output, u32 e) -> u32 {
    u32 sum = Seed;
    u32 value = inputs[0][e];
    for (u32 counter = 3; counter != 0; counter--) {
      sum += counter;
      value = (value + sum) ^ inputs[1][e];
    }
    return output == 0 ? value : sum;
  };
  static const char *const names[] = {"per-lane v10", "uniform v11"};
  CompareBulk(name, bulk, fast, slow, reference, names);
  const bool host32 = vulkan->SubgroupSize() == 32u;
  if (host32) {
    Require(name, "module size", fast.spirv.size() < slow.spirv.size(),
            "SPIR-V " + std::to_string(slow.spirv.size()) + " -> " +
                std::to_string(fast.spirv.size()) + " words");
  }
  std::printf("[codegen] %-32s ok (SPIR-V %zu -> %zu words, OpPhi %zu -> %zu)\n", name,
              slow.spirv.size(), fast.spirv.size(), CountSpirv(slow.spirv, "OpPhi"),
              CountSpirv(fast.spirv, "OpPhi"));
}

// KYTY_KNOWN_BITS_MASKS: v_and_b32 0xffff of a value shifted right by 16, and v_bfe_u32 0, 8 of a
// value already masked to 0xff, are their sources. Same results, fewer masks.
void CheckKnownBitsMasks(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenKnownBitsMasks";
  using O = ShaderOpcode;
  constexpr u32 Count = 1024;
  std::vector<std::vector<u32>> inputs(2);
  uint64_t state = 0x6b17b175u;
  for (u32 e = 0; e < Count; e++) {
    inputs[0].push_back(NextRandom(state));
    inputs[1].push_back(NextRandom(state));
  }
  const auto body = [&](std::vector<u32> *code) {
    code->push_back(EncodeVop2(0x16, 20, InlineU32(16), 2)); // v_lshrrev_b32 v20, 16, v2
    code->push_back(EncodeVop2(0x1b, 10, 255, 20));          // v_and_b32 v10, lit, v20
    code->push_back(0xffffu);
    code->push_back(EncodeVop2(0x1b, 21, 255, 3));           // v_and_b32 v21, lit, v3
    code->push_back(0xffu);
    AppendVop3(code, 0x148, 11, Vgpr(21), InlineU32(0), InlineU32(8)); // v_bfe_u32 v11, v21, 0, 8
  };
  auto bulk = MakeBulkCase(name, 2, 2, inputs, body,
                           {O::V_LSHRREV_B32, O::V_AND_B32, O::V_BFE_U32});
  for (const u32 wave : {64u, 32u}) {
    bulk.test.compute_info.wave_size = wave;
    CodegenOptions merged = GetCodegenOptions();
    merged.movrel_range = true;
    merged.known_bits_masks = true;
    CodegenOptions legacy = merged;
    legacy.known_bits_masks = false;
    const auto fast = RunBulk(vulkan, bulk, merged);
    const auto slow = RunBulk(vulkan, bulk, legacy);
    const auto reference = [&](u32 output, u32 e) -> u32 {
      return output == 0 ? inputs[0][e] >> 16u : inputs[1][e] & 0xffu;
    };
    static const char *const names[] = {"and of shift v10", "bfe of and v11"};
    CompareBulk(name, bulk, fast, slow, reference, names);
    const auto masks_fast = CountSpirv(fast.spirv, "OpBitwiseAnd") +
                            CountSpirv(fast.spirv, "OpBitFieldUExtract");
    const auto masks_slow = CountSpirv(slow.spirv, "OpBitwiseAnd") +
                            CountSpirv(slow.spirv, "OpBitFieldUExtract");
    Require(name, "mask count", masks_fast + 2u <= masks_slow,
            "wave" + std::to_string(wave) + ": masks " + std::to_string(masks_slow) + " -> " +
                std::to_string(masks_fast));
    std::printf("[codegen] %-32s ok (wave%u, AND+BFE %zu -> %zu)\n", name, wave, masks_slow,
                masks_fast);
  }
}

// KYTY_GDS_M0_BASE: ds_write_b32/ds_read_b32 on GDS with M0 = {base 16 bytes, size 0x1000} address
// GDS word 4 + lane; without the switch the base is ignored (word lane).
void CheckGdsM0Base(VulkanHarness *vulkan) {
  using O = ShaderOpcode;
  for (const bool enabled : {true, false}) {
    for (const u32 wave_size : {32u, 64u}) {
      TestCase test;
      test.name = enabled ? "CodegenGdsM0Base" : "CodegenGdsM0BaseOff";
      auto &code = test.code;
      code.push_back(EncodeVop2(0x1a, 1, InlineU32(2), 0)); // v_lshlrev_b32 v1, 2, v0
      code.push_back(EncodeVop2(0x25, 2, InlineU32(1), 0)); // v_add_nc_u32 v2, 1, v0
      AppendSMovLiteral(&code, 124, 0x00101000u);           // s_mov_b32 m0, base 16, size 4096
      code.insert(code.end(), {EncodeDs0(0x0d, 0, true), EncodeDs1(0, 2, 1),   // ds_write_b32 gds
                               EncodeDs0(0x36, 0, true), EncodeDs1(5, 0, 1)}); // ds_read_b32 gds
      AppendStoreVgprAtLaneDwordOffset(&code, 5, 0, 0);
      AppendEnd(&code);
      test.initial.assign(wave_size, 0xdeadbeefu);
      test.expected.resize(wave_size);
      std::iota(test.expected.begin(), test.expected.end(), 1u);
      test.compute_info.threads_num[0] = wave_size;
      test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
      test.compute_info.thread_ids_num = 1;
      test.compute_info.wave_size = wave_size;
      test.has_compute_info = true;
      const u32 base_words = enabled ? 4u : 0u;
      test.gds_initial.assign(wave_size + 4u, 0u);
      test.expected_gds.assign(wave_size + 4u, 0u);
      for (u32 lane = 0; lane < wave_size; lane++) {
        test.expected_gds[base_words + lane] = lane + 1u;
      }
      test.opcodes = {O::V_LSHLREV_B32, O::V_ADD_NC_U32, O::S_MOV_B32, O::DS_WRITE_B32,
                      O::DS_READ_B32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
      CodegenOptions options = GetCodegenOptions();
      options.gds_m0_base = enabled;
      ScopedCodegenOptions scope(options);
      RunCase(vulkan, test);
    }
  }
}

// KYTY_LDS_ATOMIC_ORDER: every lane of one wave does ds_add_rtn_u32 (lane + 1) and ds_sub_rtn_u32
// (lane + 1) on one LDS dword; the returned old values follow lane order (lane L sees the sum of the
// lanes below it), in wave32 and in wave64 (two lanes per host invocation).
void CheckLdsAtomicOrder(VulkanHarness *vulkan) {
  using O = ShaderOpcode;
  for (const u32 wave_size : {64u, 32u}) {
    TestCase test;
    test.name = wave_size == 64u ? "CodegenLdsAtomicOrder64" : "CodegenLdsAtomicOrder32";
    auto &code = test.code;
    AppendVMovU32(&code, 1, 0);                           // v1 = 0 (LDS address)
    AppendVMovU32(&code, 4, 0);                           // v4 = 0
    code.push_back(EncodeVop2(0x25, 2, InlineU32(1), 0)); // v_add_nc_u32 v2, 1, v0
    code.insert(code.end(), {EncodeDs0(0x0d), EncodeDs1(0, 4, 1)});     // ds_write_b32 v1, v4
    code.insert(code.end(), {EncodeDs0(0x0d, 4), EncodeDs1(0, 4, 1)});  // ds_write_b32 v1, v4 offset:4
    code.insert(code.end(), {EncodeDs0(0x20), EncodeDs1(3, 2, 1)});     // ds_add_rtn_u32 v3, v1, v2
    code.insert(code.end(), {EncodeDs0(0x21, 4), EncodeDs1(5, 2, 1)});  // ds_sub_rtn_u32 v5, v1, v2 offset:4
    AppendStoreVgprAtLaneDwordOffset(&code, 3, 0, 0);
    AppendStoreVgprAtLaneDwordOffset(&code, 5, 0, wave_size);
    AppendEnd(&code);
    test.initial.assign(wave_size * 2u, 0xdeadbeefu);
    test.expected.resize(wave_size * 2u);
    for (u32 lane = 0; lane < wave_size; lane++) {
      const u32 below = lane * (lane + 1u) / 2u;
      test.expected[lane] = below;
      test.expected[wave_size + lane] = 0u - below;
    }
    test.compute_info.threads_num[0] = wave_size;
    test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
    test.compute_info.thread_ids_num = 1;
    test.compute_info.wave_size = wave_size;
    test.compute_info.lds_size_dwords = 2;
    test.has_compute_info = true;
    test.opcodes = {O::V_MOV_B32, O::V_ADD_NC_U32, O::DS_WRITE_B32, O::DS_ADD_RTN_U32,
                    O::DS_SUB_RTN_U32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
    test.required_spirv = {"OpGroupNonUniformIAdd"};
    CodegenOptions options = GetCodegenOptions();
    options.lds_atomic_order = true;
    options.wave64_warps = false;
    options.lds_slots = false;
    ScopedCodegenOptions scope(options);
    RunCase(vulkan, test);
  }
}

// KYTY_SPLIT_LDS_WRITE_SYNC: a split wave64 (KYTY_WAVE64_WARPS, one 64-thread workgroup) writes its
// lane id to LDS, reads the mirrored lane (the other host subgroup) twice, overwrites its dword after
// those reads and reads the mirrored lane again: every read must see the other half's write in guest
// order, with and without the switch.
void CheckSplitLdsWriteSync(VulkanHarness *vulkan) {
  using O = ShaderOpcode;
  for (const bool enabled : {true, false}) {
    TestCase test;
    test.name = enabled ? "CodegenSplitLdsWriteSync" : "CodegenSplitLdsWriteSyncOff";
    auto &code = test.code;
    code.push_back(EncodeVop2(0x1a, 1, InlineU32(2), 0));       // v_lshlrev_b32 v1, 2, v0
    code.push_back(EncodeVop2(0x26, 2, 255, 0));                // v_sub_nc_u32 v2, 63, v0
    code.push_back(63u);
    code.push_back(EncodeVop2(0x1a, 2, InlineU32(2), 2));       // v_lshlrev_b32 v2, 2, v2
    code.insert(code.end(), {EncodeDs0(0x0d), EncodeDs1(0, 0, 1)});   // ds_write_b32 v1, v0
    code.insert(code.end(), {EncodeDs0(0x36), EncodeDs1(3, 0, 2)});   // ds_read_b32 v3, v2
    code.insert(code.end(), {EncodeDs0(0x36), EncodeDs1(4, 0, 2)});   // ds_read_b32 v4, v2
    code.push_back(EncodeVop2(0x25, 5, 255, 0));                // v_add_nc_u32 v5, 100, v0
    code.push_back(100u);
    code.insert(code.end(), {EncodeDs0(0x0d), EncodeDs1(0, 5, 1)});   // ds_write_b32 v1, v5
    code.insert(code.end(), {EncodeDs0(0x36), EncodeDs1(6, 0, 2)});   // ds_read_b32 v6, v2
    code.push_back(EncodeVop2(0x25, 3, Vgpr(3), 4));            // v_add_nc_u32 v3, v3, v4
    AppendStoreVgprAtLaneDwordOffset(&code, 3, 0, 0);
    AppendStoreVgprAtLaneDwordOffset(&code, 6, 0, 64);
    AppendEnd(&code);
    test.initial.assign(128, 0xdeadbeefu);
    test.expected.resize(128);
    for (u32 lane = 0; lane < 64; lane++) {
      test.expected[lane] = 2u * (63u - lane);
      test.expected[64 + lane] = 100u + 63u - lane;
    }
    test.compute_info.threads_num[0] = 64;
    test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
    test.compute_info.thread_ids_num = 1;
    test.compute_info.wave_size = 64;
    test.compute_info.lds_size_dwords = 64;
    test.has_compute_info = true;
    test.opcodes = {O::V_LSHLREV_B32, O::V_SUB_NC_U32, O::V_ADD_NC_U32, O::DS_WRITE_B32,
                    O::DS_READ_B32, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
    CodegenOptions options = GetCodegenOptions();
    options.wave64_warps = true;
    options.split_lds_write_sync = enabled;
    options.lds_slots = false;
    options.lane_local = false;
    ScopedCodegenOptions scope(options);
    RunCase(vulkan, test);
  }
}

// KYTY_SPLIT_DPP_LANE: a split wave64 (KYTY_WAVE64_WARPS, one 64-thread workgroup) runs DPP16
// row_shr:1,2,4,8 scans (BC=0, FI=0, every lane enabled) with v_add_nc_u32, v_add_f32, v_min_f32 and
// v_max_f32: each lane must get the inclusive scan of its row of 16, in both host subgroups. Before
// the fix the upper subgroup tested bits 32-63 of its 32-lane ballot and never wrote.
void CheckSplitDppScan(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenSplitDppScan";
  // VOP2 opcode, then how the guest combines two lanes (D = src0(shifted) op src1(own)).
  struct Op {
    u32 opcode;
    u32 (*apply)(u32, u32);
  };
  const Op ops[] = {
      {0x25, [](u32 a, u32 b) { return a + b; }},
      {0x03, [](u32 a, u32 b) { return std::bit_cast<u32>(std::bit_cast<float>(a) + std::bit_cast<float>(b)); }},
      {0x0f, [](u32 a, u32 b) { return std::bit_cast<u32>(std::min(std::bit_cast<float>(a), std::bit_cast<float>(b))); }},
      {0x10, [](u32 a, u32 b) { return std::bit_cast<u32>(std::max(std::bit_cast<float>(a), std::bit_cast<float>(b))); }},
  };
  constexpr u32 Wave = 64;
  TestCase test;
  test.name = name;
  auto &code = test.code;
  code.push_back(EncodeVop2(0x1a, 3, InlineU32(2), 0)); // v_lshlrev_b32 v3, 2, v0
  AppendBufferLoadDword(&code, 1, 3);                  // v1 = data[lane] (integer op input)
  code.push_back(EncodeVop2(0x25, 3, 255, 3));          // v3 += 256 (float inputs at 64..127)
  code.push_back(256u);
  AppendBufferLoadDword(&code, 2, 3);                  // v2 = data[64 + lane]
  code.push_back(EncodeSopp(0x0c, 0));                 // s_waitcnt 0
  for (u32 i = 0; i < 4; i++) {
    const u32 reg = 4u + i;
    code.push_back(EncodeVop1(0x01, reg, Vgpr(i == 0 ? 1u : 2u))); // v_mov_b32 v(4+i), input
    for (const u32 shift : {1u, 2u, 4u, 8u}) {
      code.push_back(EncodeVop2(ops[i].opcode, reg, 250, reg)); // op v, v row_shr:n, v
      code.push_back(EncodeVop2Dpp(reg, 0x110u + shift));
    }
  }
  for (u32 i = 0; i < 4; i++) {
    AppendStoreVgprAtLaneDwordOffset(&code, 4u + i, 0, 128u + i * Wave);
  }
  AppendEnd(&code);
  test.initial.resize(128u + 4u * Wave, 0xdeadbeefu);
  for (u32 lane = 0; lane < Wave; lane++) {
    test.initial[lane] = (lane * 2654435761u) >> 4u;
    test.initial[Wave + lane] =
        std::bit_cast<u32>((static_cast<float>((lane * 37u) % 101u) - 50.0f) * 0.371f);
  }
  test.expected = test.initial;
  for (u32 i = 0; i < 4; i++) {
    std::vector<u32> v(test.initial.begin() + (i == 0 ? 0 : Wave),
                       test.initial.begin() + (i == 0 ? Wave : 2u * Wave));
    for (const u32 shift : {1u, 2u, 4u, 8u}) {
      auto next = v;
      for (u32 lane = 0; lane < Wave; lane++) {
        if ((lane & 15u) >= shift) {
          next[lane] = ops[i].apply(v[lane - shift], v[lane]);
        }
      }
      v = next;
    }
    std::copy(v.begin(), v.end(), test.expected.begin() + 128 + i * Wave);
  }
  test.compute_info.threads_num[0] = Wave;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = Wave;
  test.has_compute_info = true;
  std::vector<u32> fixed_words;
  std::vector<u32> legacy_words;
  {
    CodegenOptions options = GetCodegenOptions();
    options.wave64_warps = true;
    options.split_dpp_lane = true;
    options.dpp_skip_inactive = true;
    options.lane_reductions = false;
    options.lane_local = false;
    options.lds_slots = false;
    ScopedCodegenOptions scope(options);
    fixed_words = RunCollect(vulkan, test, test.expected.size());
  }
  {
    CodegenOptions options = GetCodegenOptions();
    options.wave64_warps = true;
    options.split_dpp_lane = false;
    options.dpp_skip_inactive = true;
    options.lane_reductions = false;
    options.lane_local = false;
    options.lds_slots = false;
    ScopedCodegenOptions scope(options);
    legacy_words = RunCollect(vulkan, test, test.expected.size());
  }
  RequireEqualWords(name, "split wave64 row_shr scans", test.expected, fixed_words);
  u32 wrong = 0;
  for (size_t i = 128; i < test.expected.size(); i++) {
    wrong += legacy_words[i] != test.expected[i] ? 1u : 0u;
  }
  std::printf("[codegen] %-32s ok (KYTY_SPLIT_DPP_LANE=0: %u of %u scan results differ)\n", name,
              wrong, 4u * Wave);
}

// KYTY_MOV_INT_MODS: V_MOV_B32 and V_CNDMASK_B32 neg/abs flip or clear the sign bit (ISA p.137,
// p.131), bit exact for f32 denormals, which a host flushing f32 denormals in OpFNegate/OpFAbs
// would turn into signed zeros.
void CheckMovIntMods(VulkanHarness *vulkan) {
  constexpr const char *name = "CodegenMovIntMods";
  TestCase test;
  test.name = name;
  auto &code = test.code;
  AppendVMovU32(&code, 1, 0x00000001u);
  AppendVMovU32(&code, 2, 0x80000005u);
  AppendVMovU32(&code, 3, 0x00400000u);
  AppendVMovU32(&code, 4, 0x3f800000u);
  AppendVop3(&code, 0x181, 10, Vgpr(1), 0, 0, 0, 0, false, 0, 1); // v_mov_b32 v10, -v1
  AppendVop3(&code, 0x181, 11, Vgpr(2), 0, 0, 1);                 // v_mov_b32 v11, |v2|
  AppendVop3(&code, 0x181, 12, Vgpr(2), 0, 0, 1, 0, false, 0, 1); // v_mov_b32 v12, -|v2|
  AppendVop3(&code, 0x181, 13, Vgpr(4), 0, 0, 0, 0, false, 0, 1); // v_mov_b32 v13, -v4 (-1.0)
  code.push_back(EncodeSMovB32(20, 193));                         // s_mov_b32 s20, -1
  code.push_back(EncodeSMovB32(21, 128));                         // s_mov_b32 s21, 0
  AppendVop3(&code, 0x101, 14, Vgpr(1), Vgpr(3), 20, 0, 0, false, 0, 2); // v_cndmask v14, v1, -v3, s20
  AppendVop3(&code, 0x101, 15, Vgpr(2), Vgpr(3), 21, 1);                 // v_cndmask v15, |v2|, v3, s21
  for (u32 i = 0; i < 6; i++) {
    AppendStoreVgpr(&code, 10u + i, i);
  }
  AppendEnd(&code);
  test.initial.assign(6, 0xdeadbeefu);
  test.expected = {0x80000001u, 0x00000005u, 0x80000005u, 0xbf800000u, 0x80400000u, 0x00000005u};
  test.compute_info.threads_num[0] = 32;
  test.compute_info.threads_num[1] = test.compute_info.threads_num[2] = 1u;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.wave_size = 32;
  test.has_compute_info = true;
  std::vector<u32> on_words;
  std::vector<u32> off_words;
  {
    CodegenOptions options = GetCodegenOptions();
    options.mov_int_mods = true;
    options.mov_no_float_mods = false;
    ScopedCodegenOptions scope(options);
    on_words = RunCollect(vulkan, test, test.expected.size());
  }
  {
    CodegenOptions options = GetCodegenOptions();
    options.mov_int_mods = false;
    options.mov_no_float_mods = false;
    ScopedCodegenOptions scope(options);
    off_words = RunCollect(vulkan, test, test.expected.size());
  }
  RequireEqualWords(name, "integer sign modifiers", test.expected, on_words);
  std::printf("[codegen] %-32s ok (float path: -denormal %08x, cndmask -denormal %08x)\n", name,
              off_words[0], off_words[4]);
}

// KYTY_SCALAR_LOAD_GROUPS: S_LOAD_DWORDX4 through a scalar address, once inside one BDA page
// (one page lookup for the four dwords) and once straddling a page boundary (per-dword path); the
// results match the memory with and without the switch.
void CheckScalarLoadGroups(VulkanHarness *vulkan) {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000110000000ull;
  constexpr u32 Page = static_cast<u32>(BufferCache::CACHING_PAGESIZE);
  constexpr u32 Inside = 256u;
  constexpr u32 Straddle = Page - 8u;
  for (const bool enabled : {true, false}) {
    TestCase test;
    test.name = enabled ? "CodegenScalarLoadGroups" : "CodegenScalarLoadGroupsOff";
    test.initial.assign((2u * Page) / 4u, 0u);
    for (u32 word = 0; word < 4; word++) {
      test.initial[Inside / 4 + word] = 0x11110000u + word;
      test.initial[Straddle / 4 + word] = 0x22220000u + word;
    }
    auto &code = test.code;
    AppendSMovLiteral(&code, 2, static_cast<u32>(GuestBase));
    AppendSMovLiteral(&code, 3, static_cast<u32>(GuestBase >> 32u));
    code.push_back(EncodeSmem0(0x02, 8, 1));   // s_load_dwordx4 s[8:11], s[2:3], Inside
    code.push_back(EncodeSmem1(Inside));
    code.push_back(EncodeSmem0(0x02, 12, 1));  // s_load_dwordx4 s[12:15], s[2:3], Straddle
    code.push_back(EncodeSmem1(Straddle));
    code.push_back(EncodeSopp(0x0c, 0));       // s_waitcnt 0
    for (u32 i = 0; i < 8; i++) {
      code.push_back(EncodeVop1(0x01, 20 + i, 8 + i)); // v_mov_b32 v(20+i), s(8+i)
      AppendStoreVgpr(&code, 20 + i, i);
    }
    AppendEnd(&code);
    test.expected = {0x11110000u, 0x11110001u, 0x11110002u, 0x11110003u,
                     0x22220000u, 0x22220001u, 0x22220002u, 0x22220003u};
    test.bda_mappings = {{GuestBase, 0}};
    test.opcodes = {O::S_MOV_B32, O::S_LOAD_DWORDX4, O::S_WAITCNT, O::V_MOV_B32,
                    O::BUFFER_STORE_DWORD, O::S_ENDPGM};
    test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer"};
    CodegenOptions options = GetCodegenOptions();
    options.scalar_load_groups = enabled;
    ScopedCodegenOptions scope(options);
    RunCase(vulkan, test);
    // S_BUFFER_LOAD_DWORDXn through GPU-selected V#s (the indirect scalar group form) and the
    // scalar load variants.
    RunCase(vulkan, BufferLoadsGpuSelectedDescriptors());
    RunCase(vulkan, ScalarMemoryLoadVariants());
  }
}

void RunAll(VulkanHarness *vulkan) {
  CheckScalarLoadGroups(vulkan);
  CheckSplitLdsWriteSync(vulkan);
  CheckSplitDppScan(vulkan);
  CheckMovIntMods(vulkan);
  CheckWave64HighLanes();
  CheckF16ExactRound(vulkan);
  CheckIsaSmallFixes(vulkan);
  CheckLdsAtomicOrder(vulkan);
  CheckGdsM0Base(vulkan);
  CheckKnownBitsMasks(vulkan);
  CheckLane2Share(vulkan);
  CheckCompressedSnormExport();
  CheckTranscendentalDenormInputs(vulkan);
  CheckF32FlushSelfTest(vulkan);
  CheckF16DenormConversions(vulkan);
  CheckF32FlushStageB(vulkan);
  CheckExecSelects(vulkan);
  CheckInterpolationModes();
  CheckLdsSlots();
  CheckDenormFlushSkip();
  CheckMadPositionSlice();
  CheckMadModes(vulkan);
  CheckLodStatsGate();
  CheckLodStatsPlainVariant();
  CheckIndirectImageLodStats();
  CheckMoveRelValueSet(vulkan);
  CheckMoveRelKnownZeros(vulkan);
  CheckUniformLaneReads(vulkan);
  CheckFloatMinMax(vulkan);
  CheckShortF32Helpers(vulkan);
  CheckShortF32HelpersNanPaths(vulkan);
  CheckFloatClampConstants(vulkan);
  CheckPackRtzExhaustive();
  CheckPackRtz(vulkan);
  CheckFloatToIntSaturation(vulkan);
  CheckRobustBufferLoads(vulkan);
}

} // namespace CodegenTests
