void TestRayTracingDispatchDetection() {
  namespace D = ShaderRecompiler::Decoder;
  // The skip behaviour without a BVH mode (KYTY_RT_SOFTWARE=0, KYTY_RT_STUB=0); the software path is
  // the default.
  struct NoBvhMode {
    ShaderRecompiler::CodegenOptions saved = ShaderRecompiler::GetCodegenOptions();
    NoBvhMode() {
      auto options = saved;
      options.rt_stub = false;
      options.rt_software = false;
      ShaderRecompiler::SetCodegenOptions(options);
    }
    ~NoBvhMode() { ShaderRecompiler::SetCodegenOptions(saved); }
  } no_bvh_mode;
  const auto check_decoded = [](std::span<const uint32_t> code, bool expected,
                                const char *message) {
    D::Program decoded;
    D::DecodeProgram(code, decoded);
    Check(decoded.has_bvh == expected, message);
    if (expected) {
      const auto &last = decoded.instructions.back();
      Check(last.family == D::Family::MIMG &&
                (last.opcode_id == 0xe6u || last.opcode_id == 0xe7u),
            "decoder continued past the first BVH instruction");
      const auto result = ShaderRecompiler::TranslateProgram(
          code, MakeCompileOptions(ShaderType::Compute));
      Check(result.skip_dispatch && result.program.blocks.empty() && result.cfg_dump.empty(),
            "BVH compute shader reached IR translation");
    }
  };
  const auto check = [&](std::initializer_list<uint32_t> code, bool expected,
                         const char *message) {
    check_decoded(std::span<const uint32_t>{code.begin(), code.size()}, expected, message);
  };

  check({0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "captured contiguous BVH instruction did not skip dispatch");
  check({0xf1989f07u, 0x00071001u, 0x3d3e3c15u, 0x38095851u, 0x00003a39u,
         EncodeSopp(0x01)}, true,
        "captured NSA BVH instruction did not skip dispatch");
  check({0xf19c9f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "RDNA2 BVH64 instruction did not skip dispatch");
  check({0xbe8e210eu, 0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "unsupported SWAPPC before BVH prevented dispatch detection");
  check({0xf1989f01u, 0x00081014u, 0xc0000000u, EncodeSopp(0x01)}, true,
        "unknown instruction after BVH prevented dispatch skipping");

  for (uint32_t opcode : {0xe6u, 0xe7u}) {
    for (uint32_t nsa = 0; nsa < 4; ++nsa) {
      for (uint32_t slc = 0; slc < 2; ++slc) {
        std::array<uint32_t, 6> code{
            EncodeMimg0(opcode, 0xf, true) | (nsa << 1u) | (slc << 25u),
            EncodeMimg1(0, 0, 0, 0)};
        code[2 + nsa] = EncodeSopp(0x01);
        check_decoded(std::span(code).first(3 + nsa), true,
                      "BVH cache flags or NSA length prevented dispatch detection");
      }
    }
  }
  for (uint32_t opcode : {0x66u, 0x67u, 0xe4u, 0xe5u, 0xe8u, 0xe9u}) {
    check({EncodeMimg0(opcode, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
          false, "non-BVH image opcode was incorrectly skipped");
  }

  check({EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "scalar literal was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf) | 6u, EncodeMimg1(0, 0, 0, 0),
         0xf1989f01u, 0xf19c9f01u, 0u, EncodeSopp(0x01)}, false,
        "NSA address words were mistaken for BVH instructions");
  check({EncodeMimg0(0x20, 0xf), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "MIMG operand word was mistaken for a BVH instruction");
  check({EncodeSopp(0x01), 0xf1989f01u, 0x00081014u}, false,
        "data after the shader end was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
        false, "ordinary image shader was incorrectly skipped");

  // An end instruction on one branch does not end another branch's code.
  check({EncodeSopp(0x04, 1), EncodeSopp(0x01), 0xf1989f01u, 0x00081014u,
         EncodeSopp(0x01)}, true,
        "BVH at a branch target after an early end was not detected");

  const std::array bvh_code{0xf1989f01u, 0x00081014u, 0xf19c9f01u, 0x00081014u,
                            EncodeSopp(0x01)};
  const std::array ordinary_code{EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)};
  D::Program reused;
  D::DecodeProgram(bvh_code, reused);
  Check(reused.has_bvh && reused.instructions.size() == 1,
        "decoder did not stop at the first BVH instruction");
  D::DecodeProgram(ordinary_code, reused);
  Check(!reused.has_bvh, "decoder retained stale BVH metadata");
  for (const auto stage : {ShaderType::Compute, ShaderType::Vertex, ShaderType::Pixel}) {
    const auto options = MakeCompileOptions(stage);
    Check(!ShaderRecompiler::TranslateProgram(ordinary_code, options).skip_dispatch,
          "BVH-shaped literal marked an ordinary shader for skipping");
    const auto compiled = RecompileForTest(ordinary_code, options);
    Check(!compiled.spirv.empty(), "ordinary shader did not compile");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
}

class ScopedRtStub {
public:
  explicit ScopedRtStub(bool enabled, bool software = false)
      : m_saved(ShaderRecompiler::GetCodegenOptions()) {
    auto options = m_saved;
    options.rt_stub = enabled;
    options.rt_software = software;
    ShaderRecompiler::SetCodegenOptions(options);
  }
  ~ScopedRtStub() { ShaderRecompiler::SetCodegenOptions(m_saved); }
  ScopedRtStub(const ScopedRtStub &) = delete;
  ScopedRtStub &operator=(const ScopedRtStub &) = delete;

private:
  ShaderRecompiler::CodegenOptions m_saved;
};

void TestRayTracingStub() {
  namespace D = ShaderRecompiler::Decoder;
  using O = D::Opcode;
  const auto decode = [](std::initializer_list<uint32_t> words) {
    const std::vector<uint32_t> code(words);
    D::Instruction inst;
    D::DecodeInstruction(code, 0, inst);
    return inst;
  };
  const auto a16 = [](const D::Instruction &inst) {
    return (inst.image_sample_flags & D::ImageSampleFlagA16) != 0u;
  };

  // Captured contiguous form: image_bvh_intersect_ray v[16:19], v[20:30], s[32:35].
  const auto contiguous = decode({0xf1989f01u, 0x00081014u});
  Check(contiguous.opcode == O::IMAGE_BVH_INTERSECT_RAY && contiguous.word_count == 2 &&
            contiguous.image_address_components == 11 && contiguous.data_dwords == 4 &&
            contiguous.dmask == 0xfu && contiguous.image_r128 && !a16(contiguous) &&
            contiguous.dst.kind == D::OperandKind::Vgpr && contiguous.dst.reg == 16 &&
            contiguous.src0.kind == D::OperandKind::Vgpr && contiguous.src0.reg == 20 &&
            contiguous.src1.kind == D::OperandKind::Sgpr && contiguous.src1.reg == 32,
        "captured contiguous BVH instruction decoded wrongly");
  Check(D::InstructionToString(contiguous).find("IMAGE_BVH_INTERSECT_RAY") !=
            std::string::npos,
        "BVH instruction text lost its opcode name");

  // Captured NSA form: node pointer in v1, then v21, v60, v62, v61, v81, v88, v9, v56-v58.
  const auto nsa = decode({0xf1989f07u, 0x00071001u, 0x3d3e3c15u, 0x38095851u, 0x00003a39u});
  constexpr uint32_t nsa_regs[] = {21, 60, 62, 61, 81, 88, 9, 56, 57, 58};
  bool nsa_ok = nsa.opcode == O::IMAGE_BVH_INTERSECT_RAY && nsa.word_count == 5 &&
                nsa.image_nsa_dwords == 3 && nsa.image_address_components == 11 &&
                nsa.src0.reg == 1 && nsa.dst.reg == 16 && nsa.src1.reg == 28;
  for (uint32_t i = 0; i < std::size(nsa_regs); ++i) {
    nsa_ok = nsa_ok && nsa.image_nsa_addr[i] == nsa_regs[i];
  }
  Check(nsa_ok, "captured NSA BVH instruction decoded wrongly");

  // Address dword counts from RDNA2 ISA Table 48.
  const auto bvh64 = decode({0xf19c9f01u, 0x00081014u});
  const auto a16_32 = decode({0xf1989f01u, 0x40081014u});
  const auto a16_64 = decode({0xf19c9f01u, 0x40081014u});
  Check(bvh64.opcode == O::IMAGE_BVH64_INTERSECT_RAY && bvh64.image_address_components == 12 &&
            a16_32.opcode == O::IMAGE_BVH_INTERSECT_RAY && a16(a16_32) &&
            a16_32.image_address_components == 8 &&
            a16_64.opcode == O::IMAGE_BVH64_INTERSECT_RAY && a16(a16_64) &&
            a16_64.image_address_components == 9,
        "BVH address dword counts do not follow the ISA");

  // Encodings the ISA forbids are rejected rather than guessed.
  for (const auto &[word0, word1, nsa_word, text] :
       {std::tuple{0xf1989701u, 0x00081014u, 0u, "DMASK=0x7"},
        std::tuple{0xf1981f01u, 0x00081014u, 0u, "R128=0"},
        std::tuple{0xf1999f01u, 0x00081014u, 0u, "TFE=1"},
        std::tuple{0xf19a9f01u, 0x00081014u, 0u, "LWE=1"},
        std::tuple{0xf1989f01u, 0x80081014u, 0u, "D16=1"},
        std::tuple{0xf1989f03u, 0x00081014u, 0x04030201u, "one NSA dword"}}) {
    const auto inst = decode({word0, word1, nsa_word});
    if (inst.opcode != O::UNSUPPORTED) {
      std::fprintf(stderr, "BVH encoding with %s was accepted\n", text);
      Check(false, "invalid BVH encoding was accepted");
    }
  }

  // decode_bvh=true decodes through BVH instructions to the end of the shader.
  const std::array code{0xf1989f01u, 0x00081014u, 0xf19c9f01u, 0x00081014u,
                        EncodeSopp(0x01)};
  D::Program program;
  D::DecodeProgram(code, program, true);
  Check(program.has_bvh && program.instructions.size() == 3 &&
            program.instructions[0].opcode == O::IMAGE_BVH_INTERSECT_RAY &&
            program.instructions[1].opcode == O::IMAGE_BVH64_INTERSECT_RAY &&
            program.instructions[2].opcode == O::S_ENDPGM,
        "BVH decode mode did not decode the whole shader");
  D::DecodeProgram(code, program, false);
  Check(program.has_bvh && program.instructions.size() == 1 &&
            program.instructions[0].opcode == O::UNSUPPORTED,
        "default decode did not stop at the first BVH instruction");

  // KYTY_RT_STUB: BVH shaders translate and compile in every stage instead of being skipped.
  const std::array compute_code{0xf1989f01u, 0x00081014u, EncodeSopp(0x01)};
  {
    const ScopedRtStub stub(false);
    Check(ShaderRecompiler::TranslateProgram(compute_code,
                                             MakeCompileOptions(ShaderType::Compute))
              .skip_dispatch,
          "BVH compute shader was not skipped without KYTY_RT_STUB");
  }
  // KYTY_RT_SOFTWARE: BVH shaders compile to valid SPIR-V. A pixel shader that exports the
  // result keeps the node test alive (an unused result is dead code), so spirv-val checks the whole
  // lowering, BDA node fetch included.
  {
    const ScopedRtStub software(false, true);
    const auto compute = RecompileForTest(compute_code, MakeCompileOptions(ShaderType::Compute));
    Check(!compute.spirv.empty() && compute.program.info.uses_bvh,
          "KYTY_RT_SOFTWARE compute program did not compile or lost its BVH flag");
    CheckSpirvBinaryValidates(compute.spirv);
    for (const bool type6 : {true, false}) {
      auto options = ShaderRecompiler::GetCodegenOptions();
      options.rt_type6 = type6;
      ShaderRecompiler::SetCodegenOptions(options);
      const std::array pixel_code{0xf1989f01u, 0x00081014u,
                                  EncodeExp0(0x00, 0xf, true, false, true),
                                  EncodeExp1(16, 17, 18, 19), EncodeSopp(0x01)};
      const auto pixel = RecompileForTest(pixel_code, MakeCompileOptions(ShaderType::Pixel));
      Check(!pixel.spirv.empty() && pixel.program.info.uses_bvh && pixel.program.info.uses_dma,
            "KYTY_RT_SOFTWARE pixel program did not compile or lost its BVH or BDA flag");
      CheckSpirvBinaryValidates(pixel.spirv);
    }
  }
  const ScopedRtStub stub(true);
  for (const auto stage : {ShaderType::Compute, ShaderType::Pixel}) {
    const auto options = MakeCompileOptions(stage);
    Check(!ShaderRecompiler::TranslateProgram(compute_code, options).skip_dispatch,
          "KYTY_RT_STUB still skipped a BVH shader");
    const auto compiled = RecompileForTest(compute_code, options);
    Check(!compiled.spirv.empty() && compiled.program.info.uses_bvh,
          "KYTY_RT_STUB program did not compile or lost its BVH flag");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
  const auto ordinary = RecompileForTest(std::array{EncodeSopp(0x01)},
                                         MakeCompileOptions(ShaderType::Compute));
  Check(!ordinary.program.info.uses_bvh, "program without BVH was marked as using BVH");
}

// The traversal-loop shape of Astro Bot's BVH shaders (380bb9d636390bae and 26 more failed spirv-val
// with "Selection must be structured"): the loop body's first conditional either stays in the
// loop or enters a region with its own loop that leaves through the loop's merge, and the body
// has another exit. The conditional must get a selection merge (a synthetic block on the region's
// exit edge), not be treated as loop control.
//   s0 = 0
//   outer: if (s0 < 4) goto body
//          s1 = 0; inner: s1 += 1; if (s1 < 3) goto inner; goto done   <- the break region
//   body:  s0 += 1; if (s0 == 50) goto done; goto outer                <- the other exit
//   done:  ...
std::vector<uint32_t> TraversalLoopBreakRegionCode() {
  return {
      EncodeSMovB32(0, 128),            // 0: s_mov_b32 s0, 0
      EncodeSopc(0x0a, 0, 132),         // 1: outer: s_cmp_lt_u32 s0, 4
      EncodeSopp(0x05, 5),              // 2: s_cbranch_scc1 body
      EncodeSMovB32(1, 128),            // 3: s_mov_b32 s1, 0
      EncodeSop2(0x00, 1, 1, 129),      // 4: inner: s_add_u32 s1, s1, 1
      EncodeSopc(0x0a, 1, 131),         // 5: s_cmp_lt_u32 s1, 3
      EncodeSopp(0x05, 0xfffdu),        // 6: s_cbranch_scc1 inner
      EncodeSopp(0x02, 4),              // 7: s_branch done
      EncodeSop2(0x00, 0, 0, 129),      // 8: body: s_add_u32 s0, s0, 1
      EncodeSopc(0x06, 0, 178),         // 9: s_cmp_eq_u32 s0, 50
      EncodeSopp(0x05, 1),              // 10: s_cbranch_scc1 done
      EncodeSopp(0x02, 0xfff5u),        // 11: s_branch outer
      EncodeSopp(0x01),                 // 12: done: s_endpgm
  };
}

// KYTY_RT_FUNCTION: the software node test is one SPIR-V function per module that every BVH
// instruction calls, not a copy at each instruction (a kernel with three instructions was 322k
// words). A pixel shader with three instructions whose results all reach the export keeps all
// three alive.
struct BvhModuleShape {
  uint32_t named_functions = 0; // OpFunction results named "bvh_node_test"
  uint32_t calls = 0;           // OpFunctionCall instructions of that function
  uint32_t functions = 0;       // every OpFunction
  size_t words = 0;
};

BvhModuleShape InspectBvhModule(const std::vector<uint32_t> &spirv) {
  BvhModuleShape shape;
  shape.words = spirv.size();
  uint32_t function_id = 0;
  for (size_t at = 5; at < spirv.size();) {
    const uint32_t opcode = spirv[at] & 0xffffu;
    const uint32_t count = spirv[at] >> 16u;
    if (count == 0) break;
    if (opcode == 5 /* OpName */ && count >= 3) {
      const char *text = reinterpret_cast<const char *>(&spirv[at + 2]);
      if (std::strcmp(text, "bvh_node_test") == 0) function_id = spirv[at + 1];
    } else if (opcode == 54 /* OpFunction */) {
      shape.functions++;
      if (function_id != 0 && spirv[at + 2] == function_id) shape.named_functions++;
    } else if (opcode == 57 /* OpFunctionCall */ && function_id != 0 &&
               spirv[at + 3] == function_id) {
      shape.calls++;
    }
    at += count;
  }
  return shape;
}

void TestRayTracingSharedFunction() {
  const std::array pixel_code{
      0xf1989f01u, 0x00081014u, // image_bvh_intersect_ray v[16:19]
      0xf1989f01u, 0x00081414u, // image_bvh_intersect_ray v[20:23]
      0xf1989f01u, 0x00081814u, // image_bvh_intersect_ray v[24:27]
      EncodeExp0(0x00, 0xf, true, false, true), EncodeExp1(16, 20, 24, 17),
      EncodeSopp(0x01)};
  const auto compile = [&](uint32_t function_mode) {
    const ScopedRtStub software(false, true);
    auto options = ShaderRecompiler::GetCodegenOptions();
    options.rt_function = function_mode;
    ShaderRecompiler::SetCodegenOptions(options);
    auto compiled = RecompileForTest(pixel_code, MakeCompileOptions(ShaderType::Pixel));
    Check(!compiled.spirv.empty() && compiled.program.info.uses_bvh,
          "three-instruction BVH pixel program did not compile");
    CheckSpirvBinaryValidates(compiled.spirv);
    return InspectBvhModule(compiled.spirv);
  };
  const auto inlined = compile(0);
  const auto shared = compile(1);
  const auto no_inline = compile(2);
  Check(inlined.named_functions == 0 && inlined.calls == 0,
        "KYTY_RT_FUNCTION=0 emitted the node test function");
  Check(shared.named_functions == 1, "the BVH node test function is not defined exactly once");
  Check(shared.calls == 3, "three BVH instructions did not make three calls of the function");
  Check(shared.functions == inlined.functions + 1,
        "the BVH node test function is not the only added function");
  Check(no_inline.named_functions == 1 && no_inline.calls == 3,
        "KYTY_RT_FUNCTION=2 changed the function structure");
  // The shared module carries about one node test, the inline one three.
  Check(shared.words * 2 < inlined.words,
        "the shared BVH module is not much smaller than the inlined one");
  std::printf("BVH node test: inline %zu words, one function %zu words (3 sites)\n", inlined.words,
              shared.words);
}

void TestTraversalLoopBreakRegion() {
  namespace D = ShaderRecompiler::Decoder;
  namespace CFG = ShaderRecompiler::CFG;
  const auto code = TraversalLoopBreakRegionCode();
  D::Program decoded;
  D::DecodeProgram(code, decoded);
  auto graph = CFG::BuildGraph(decoded);
  Check(!graph.unsupported && !graph.irreducible && CFG::Structurize(graph),
        "traversal-loop break region did not structurize");
  // The conditional at pc 4 (the loop's first body block) has a merge: a synthetic block that
  // breaks to the loop merge.
  const auto conditional = std::ranges::find_if(graph.blocks, [](const CFG::BasicBlock &block) {
    return block.start_pc == 4u && block.inst_begin != block.inst_end &&
           block.terminator.kind == CFG::TerminatorKind::ConditionalBranch;
  });
  Check(conditional != graph.blocks.end(), "traversal-loop conditional not found");
  const auto *merge = graph.FindBlock(conditional->terminator.merge_block);
  Check(merge != nullptr && merge->inst_begin == merge->inst_end &&
            merge->terminator.kind == CFG::TerminatorKind::Branch &&
            graph.FindBlock(merge->terminator.true_block) != nullptr &&
            graph.FindBlock(merge->terminator.true_block)->start_pc == 48u,
        "traversal-loop conditional lacks its synthetic merge before the loop exit");
  for (const auto stage : {ShaderType::Compute, ShaderType::Pixel}) {
    const auto compiled = RecompileForTest(code, MakeCompileOptions(stage));
    Check(!compiled.program.dispatcher_fallback,
          "traversal-loop shader fell back to the dispatcher");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
}

// Development check on real guest code (not part of the default run): each file is a raw
// shader binary (code first; data after its end is ignored). Decodes it with the BVH decoder,
// then translates it as a wave32 compute shader (Astro Bot's RT kernels set CS_W32_EN) with
// KYTY_RT_STUB and with KYTY_RT_SOFTWARE. Returns the number of failures.
int TranslateBvhShaderFiles(int argc, char **argv) {
  namespace D = ShaderRecompiler::Decoder;
  static const ShaderComputeInputInfo wave32 = [] {
    ShaderComputeInputInfo info{};
    info.wave_size = 32;
    info.host_subgroup_size = 32;
    return info;
  }();
  auto options = MakeCompileOptions(ShaderType::Compute);
  options.input_info.compute = &wave32;
  options.wave_size = 32;
  int failures = 0;
  for (int arg = 2; arg < argc; ++arg) {
    std::FILE *file = std::fopen(argv[arg], "rb");
    if (file == nullptr) {
      std::fprintf(stderr, "%s: cannot open\n", argv[arg]);
      failures++;
      continue;
    }
    std::vector<uint32_t> code;
    uint32_t word = 0;
    while (std::fread(&word, sizeof(word), 1, file) == 1) {
      code.push_back(word);
    }
    std::fclose(file);
    D::Program program;
    D::DecodeProgram(code, program, true);
    uint32_t bvh = 0;
    uint32_t bvh64 = 0;
    uint32_t unsupported = 0;
    for (const auto &inst : program.instructions) {
      bvh += inst.opcode == D::Opcode::IMAGE_BVH_INTERSECT_RAY ? 1u : 0u;
      bvh64 += inst.opcode == D::Opcode::IMAGE_BVH64_INTERSECT_RAY ? 1u : 0u;
      if (inst.opcode == D::Opcode::UNSUPPORTED) {
        unsupported++;
        std::fprintf(stderr, "%s: %s\n", argv[arg], D::InstructionToString(inst).c_str());
      }
    }
    bool ok[2] = {};
    for (const bool software : {false, true}) {
      const ScopedRtStub mode(!software, software);
      const auto translated = ShaderRecompiler::TranslateProgram(code, options);
      uint32_t ops = 0;
      for (const auto *block : translated.program.blocks) {
        for (const auto &inst : *block) {
          const auto op = inst.GetOpcode();
          ops += op == ShaderRecompiler::IR::ValueOpcode::BvhIntersectRay ||
                         op == ShaderRecompiler::IR::ValueOpcode::BvhIntersectRayStub
                     ? 1u
                     : 0u;
        }
      }
      ok[software ? 1 : 0] = unsupported == 0 && !translated.skip_dispatch &&
                             !translated.program.blocks.empty() && ops == bvh + bvh64;
    }
    std::printf("%s: instructions=%zu bvh=%u bvh64=%u unsupported=%u stub=%s software=%s\n",
                argv[arg], program.instructions.size(), bvh, bvh64, unsupported,
                ok[0] ? "yes" : "NO", ok[1] ? "yes" : "NO");
    failures += ok[0] && ok[1] ? 0 : 1;
  }
  return failures;
}

// Development check (not part of the default run): shader_cfg_tests --bvh-validate <file|@list>...
// compiles each raw guest compute shader (code first; data after its end is ignored) the way the
// emulator does -- TranslateProgram, ExtractResourcePlan, MaterializeResources, CompileProgram --
// and validates the SPIR-V as pipelineCache.cpp does (spirv-val, Vulkan 1.3). A failure prints
// the validator's messages with the disassembly around the failing instruction and writes the
// module next to the input (<file>.w<wave>.<mode>.spv).
// - The codegen switches come from the environment, as in the game (KYTY_SRT_VARIANT_READS,
//   KYTY_BDA_WRITES, KYTY_RT_STUB, KYTY_RT_SOFTWARE, ...). KYTY_HARNESS_RT_MODES=stub,software
//   (any of stub, software, env) runs each file once per BVH mode; the default is env.
// - KYTY_HARNESS_WAVE=32|64|32,64 (default 32,64) with a host subgroup of 32 (the RTX 3090).
// - Resources materialize against synthetic memory in which every read succeeds with plausible
//   pointers (RT-TILED's --srt-plan-files memory); the float controls are the 3090's.
// - @list reads one path per line (UTF-8, with or without a BOM).
namespace BvhValidate {

bool ReadSynthetic(void *, uint64_t address, std::span<uint32_t> values) {
  for (size_t i = 0; i < values.size(); i++) {
    const uint64_t word_address = address + i * 4u;
    values[i] = (word_address & 4u) != 0
                    ? 0x1u
                    : (0x00100000u | static_cast<uint32_t>(word_address & 0xff0u));
  }
  return true;
}

std::vector<std::string> SplitList(const char *text) {
  std::vector<std::string> items;
  std::string item;
  for (const char *c = text; ; ++c) {
    if (*c == ',' || *c == '\0') {
      if (!item.empty()) items.push_back(item);
      item.clear();
      if (*c == '\0') break;
    } else if (*c != ' ') {
      item += *c;
    }
  }
  return items;
}

std::vector<std::string> ExpandFiles(int argc, char **argv) {
  std::vector<std::string> files;
  for (int arg = 2; arg < argc; ++arg) {
    if (argv[arg][0] != '@') {
      files.emplace_back(argv[arg]);
      continue;
    }
    std::FILE *list = std::fopen(argv[arg] + 1, "rb");
    if (list == nullptr) {
      std::fprintf(stderr, "%s: cannot open\n", argv[arg] + 1);
      continue;
    }
    std::string text;
    char buffer[4096];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), list)) != 0) text.append(buffer, read);
    std::fclose(list);
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xefu &&
        static_cast<unsigned char>(text[1]) == 0xbbu && static_cast<unsigned char>(text[2]) == 0xbfu) {
      text.erase(0, 3);
    }
    std::string line;
    for (const char c : text) {
      if (c == '\n' || c == '\r') {
        if (!line.empty()) files.push_back(line);
        line.clear();
      } else {
        line += c;
      }
    }
    if (!line.empty()) files.push_back(line);
  }
  return files;
}

// The disassembly lines around instruction `index` (the validator's instruction position).
std::string Context(const std::vector<uint32_t> &spirv, size_t index) {
  spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
  std::string text;
  tools.Disassemble(spirv, &text,
                    static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
                        static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
  std::vector<std::string> lines;
  std::string line;
  for (const char c : text) {
    if (c == '\n') {
      lines.push_back(line);
      line.clear();
    } else {
      line += c;
    }
  }
  std::string out;
  const size_t first = index > 12u ? index - 12u : 0u;
  for (size_t i = first; i < lines.size() && i <= index + 6u; ++i) {
    out += (i == index ? ">> " : "   ") + lines[i] + "\n";
  }
  return out;
}

} // namespace BvhValidate

int ValidateBvhShaderFiles(int argc, char **argv) {
  namespace IRN = ShaderRecompiler::IR;
  const auto *wave_text = std::getenv("KYTY_HARNESS_WAVE");
  const auto *mode_text = std::getenv("KYTY_HARNESS_RT_MODES");
  const auto waves = BvhValidate::SplitList(wave_text != nullptr ? wave_text : "32,64");
  const auto modes = BvhValidate::SplitList(mode_text != nullptr ? mode_text : "env");
  const auto saved_controls = ShaderRecompiler::Spirv::GetHostFloatControls();
  ShaderRecompiler::Spirv::SetHostFloatControls(
      {.denorm_flush_f32 = false, .denorm_preserve_f16 = true, .denorm_preserve_f64 = false});
  std::array<uint32_t, 64> user_data{};
  for (uint32_t i = 0; i < user_data.size(); i++) {
    user_data[i] = (i & 1u) != 0 ? 0x1u : (0x00200000u + i * 0x100u);
  }
  int failures = 0;
  for (const auto &path : BvhValidate::ExpandFiles(argc, argv)) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
      std::printf("%s: cannot open\n", path.c_str());
      failures++;
      continue;
    }
    std::vector<uint32_t> code;
    uint32_t word = 0;
    while (std::fread(&word, sizeof(word), 1, file) == 1) code.push_back(word);
    std::fclose(file);
    for (const auto &wave_name : waves) {
      const auto wave = static_cast<uint32_t>(std::strtoul(wave_name.c_str(), nullptr, 10));
      for (const auto &mode : modes) {
        auto codegen = ShaderRecompiler::GetCodegenOptions();
        if (mode == "stub" || mode == "software") {
          codegen.rt_stub = mode == "stub";
          codegen.rt_software = mode == "software";
        }
        const auto saved_codegen = ShaderRecompiler::GetCodegenOptions();
        ShaderRecompiler::SetCodegenOptions(codegen);
        ShaderComputeInputInfo info{};
        info.wave_size = wave;
        info.host_subgroup_size = 32;
        info.threads_num[0] = 64;
        info.threads_num[1] = 1;
        info.threads_num[2] = 1;
        info.thread_ids_num = 3;
        info.group_id[0] = info.group_id[1] = info.group_id[2] = true;
        info.workgroup_register = 16;
        auto options = MakeCompileOptions(ShaderType::Compute);
        options.input_info.compute = &info;
        options.wave_size = wave;
        options.user_data = user_data;
        const auto label = path + " w" + wave_name + " " + mode;
        std::fprintf(stderr, "%s: compiling\n", label.c_str());
        std::fflush(stderr);
        auto translated = ShaderRecompiler::TranslateProgram(code, options);
        std::string result;
        if (translated.skip_dispatch || translated.program.blocks.empty()) {
          result = "skipped";
        } else {
          const auto plan = IRN::ExtractResourcePlan(translated.program);
          IRN::SrtRuntime runtime{};
          runtime.user_data = user_data;
          runtime.read_memory = BvhValidate::ReadSynthetic;
          runtime.read_specialization_memory = BvhValidate::ReadSynthetic;
          IRN::ResourceSnapshot snapshot;
          IRN::ResourceSpecialization specialization;
          if (!IRN::MaterializeResources(plan, runtime, snapshot, specialization)) {
            result = "not materialized";
          } else {
            const auto compiled =
                ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
            spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
            std::string messages;
            size_t first_index = SIZE_MAX;
            tools.SetMessageConsumer([&](spv_message_level_t, const char *,
                                         const spv_position_t &position, const char *message) {
              messages += "  spirv-val: " + std::string(message) + "\n";
              first_index = std::min<size_t>(first_index, position.index);
            });
            if (tools.Validate(compiled.spirv)) {
              result = "valid (" + std::to_string(compiled.spirv.size()) + " words)";
            } else {
              result = "INVALID";
              failures++;
              std::printf("%s: INVALID\n%s", label.c_str(), messages.c_str());
              if (first_index != SIZE_MAX) {
                std::printf("%s", BvhValidate::Context(compiled.spirv, first_index).c_str());
              }
              const auto out_path = path + ".w" + wave_name + "." + mode + ".spv";
              if (std::FILE *out = std::fopen(out_path.c_str(), "wb"); out != nullptr) {
                std::fwrite(compiled.spirv.data(), sizeof(uint32_t), compiled.spirv.size(), out);
                std::fclose(out);
              }
            }
          }
        }
        ShaderRecompiler::SetCodegenOptions(saved_codegen);
        if (result != "INVALID") {
          std::printf("%s: %s\n", label.c_str(), result.c_str());
        }
        std::fflush(stdout);
      }
    }
  }
  ShaderRecompiler::Spirv::SetHostFloatControls(saved_controls);
  return failures;
}
