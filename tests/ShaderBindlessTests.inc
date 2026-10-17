// Bindless images (Senaxx's engine method; ir/BindlessBindings.h), included by
// ShaderRecompilerComputeTests.cpp after ShaderCodegenTests.inc. CPU only: translation, resource
// tracking, materialization and SPIR-V validation (--bindless-codegen-only).
//
// Spider-Man: Miles Morales' ray-tracing kernels sample a texture whose T# they read from a
// descriptor heap at a key each lane loaded from a buffer: a waterfall over the lanes picks one
// key per iteration (S_FF1_I32_B64 over the remaining lanes, V_READLANE_B32 of the key), scales
// it by the 32-byte T# stride (S_LSHL_B32 5), reads the T# with S_BUFFER_LOAD_DWORDX8 from the
// heap's V#, and samples with IMAGE_SAMPLE_L. No enumeration covers such a key (it comes from
// memory, not a bounded loop or a material table), so without bindless images the program is
// skipped; with them the shader looks the key up in the bindless translation table.

namespace BindlessTests {

using CodegenTests::CountSpirv;
using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;

// The Miles pattern, written out: s[0:3] the key buffer, s[4:7] the heap V#, s[8:11] the S#,
// s[48:51] the output buffer.
std::vector<u32> MilesHeapSampleCode() {
  std::vector<u32> code;
  // v1 = key[lane]: buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false));
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0)); // s_waitcnt 0
  AppendVMovLiteral(&code, 3, 0x3f000000u); // u
  AppendVMovLiteral(&code, 4, 0x3f000000u); // v
  AppendVMovLiteral(&code, 5, 0x00000000u); // lod
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeSop1(0x14u, 22, 20)); // s_ff1_i32_b64 s22, s[20:21]
  // v_readlane_b32 s23, v1, s22
  code.push_back(EncodeVop3Word0(0x360u, 23));
  code.push_back(EncodeVop3Word1(Vgpr(1), 22));
  code.push_back(EncodeVopc(0xc2u, 23, 1));    // v_cmp_eq_u32 vcc, s23, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));  // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 23, InlineU32(5))); // s_lshl_b32 vcc_lo, s23, 5
  // s_buffer_load_dwordx8 s[32:39], s[4:7], vcc_lo
  code.push_back(EncodeSmem0(0x0bu, 32, 2));
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0)); // s_waitcnt 0
  // image_sample_l v2, v[3:5], s[32:39], s[8:11] dmask 0x1
  code.push_back(EncodeMimg0(0x24u, 0x1u));
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc1 loop
  code.push_back(EncodeSopp(0x0cu, 0)); // s_waitcnt 0
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

std::array<u32, 64> MilesHeapUserData() {
  std::array<u32, 64> user_data{};
  // Key buffer: 64 keys of 4 bytes.
  user_data[0] = 0x200u;
  user_data[1] = 4u << 16u;
  user_data[2] = 64u;
  user_data[3] = 0x5204u;
  // The heap: 0x1000 bytes from 0x4000, T# records from byte 0 (128 keys).
  user_data[4] = 0x4000u;
  user_data[5] = 0u;
  user_data[6] = 0x1000u;
  user_data[7] = 0x5204u;
  // Output buffer.
  user_data[48] = 0x8000u;
  user_data[50] = 256u;
  user_data[51] = 0x5204u;
  return user_data;
}

void CheckMilesHeapSample() {
  constexpr const char *name = "BindlessMilesHeapSample";
  const auto code = MilesHeapSampleCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;

  // Without bindless images the key has no enumeration: with variant reads the program is
  // skipped, as before.
  {
    auto codegen = GetCodegenOptions();
    codegen.srt_variant_reads = true;
    ScopedCodegenOptions scope(codegen);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
    const auto skipped = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "without bindless images", skipped.skip_dispatch,
            "the heap-indexed T# was not skipped without bindless images");
  }

  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the heap-indexed T# was skipped with bindless images");
  const auto &images = translated.program.info.images;
  Require(name, "bindless plan",
          images.size() == 1u && images[0].resource_class ==
                                     ShaderRecompiler::IR::ImageResourceClass::Sampled,
          "expected one sampled image, got " + std::to_string(images.size()));
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "bindless plan",
          source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
              source.indirect_descriptor->table_offset == 0u &&
              source.indirect_descriptor->table_stride == 32u,
          "the image is not a bindless heap lookup");

  const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  std::vector<u32> memory(0x4000u, 0u);
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(code.data()),
      .read_memory = ReadTestMemory,
      .userdata = &memory,
      .read_specialization_memory = ReadTestMemory,
  };
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  Require(name, "resource materialization",
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization),
          "the bindless image could not be materialized");
  Require(name, "bindless heap",
          resources.bindless_heaps.size() == 1u && resources.bindless_heaps[0].base == 0x4000u &&
              resources.bindless_heaps[0].size == 0x1000u &&
              resources.bindless_heaps[0].table_offset == 0u &&
              resources.bindless_heaps[0].image == 0u &&
              static_cast<size_t>(resources.bindless_heaps[0].mapping_offset) + 2u <=
                  resources.flattened_srt.size() &&
              specialization.images.size() == 1u && specialization.images[0].bindless,
          "the snapshot does not describe the heap");

  // The resource plan and specialization survive the program cache's encoding.
  std::vector<uint8_t> plan_bytes;
  ShaderRecompiler::IR::ResourcePlan decoded;
  Require(name, "program cache",
          ShaderRecompiler::IR::EncodeResourcePlan(plan, plan_bytes) &&
              ShaderRecompiler::IR::DecodeResourcePlan(plan_bytes, decoded),
          "the bindless resource plan did not round-trip");
  ShaderRecompiler::IR::ResourceSnapshot decoded_resources;
  ShaderRecompiler::IR::ResourceSpecialization decoded_specialization;
  Require(name, "program cache",
          ShaderRecompiler::IR::MaterializeResources(decoded, runtime, decoded_resources,
                                                     decoded_specialization) &&
              decoded_resources.bindless_heaps == resources.bindless_heaps &&
              decoded_specialization == specialization,
          "the decoded plan materialized another bindless heap");
  std::vector<uint8_t> specialization_bytes;
  ShaderRecompiler::IR::EncodeSpecialization(specialization, specialization_bytes);
  ShaderRecompiler::IR::ResourceSpecialization specialization_back;
  Require(name, "program cache",
          ShaderRecompiler::IR::DecodeSpecialization(specialization_bytes, specialization_back) &&
              specialization_back == specialization,
          "the bindless specialization did not round-trip");

  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  Require(name, "SPIR-V emit", !result.spirv.empty(), "no SPIR-V");
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V",
          CountSpirv(result.spirv, "OpCapability RuntimeDescriptorArray") == 1u &&
              CountSpirv(result.spirv, "\"bindless_translation\"") == 1u &&
              CountSpirv(result.spirv, "\"bindless_feedback\"") == 1u &&
              CountSpirv(result.spirv, "OpImageSampleExplicitLod") >= 1u,
          "the sample does not go through the bindless arrays");
  Require(name, "bindless image",
          result.program.info.images.size() == 1u && result.program.info.images[0].bindless,
          "the compiled image is not bindless");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

void RunCodegen() { CheckMilesHeapSample(); }

} // namespace BindlessTests
