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

// The hit-callback form: the waterfall key selects a 16-byte record of a V# table whose dwords
// 2-3 point at a material; the T# is S_LOAD_DWORDX8 from the material, the S# S_LOAD_DWORDX4 at
// +32. The image becomes a record pointer heap; the S# has no plan and takes the default sampler.
// BUFFER_LOAD_FORMAT_X and _XYZW through a V# of the material (+48) decode the V#'s format in
// the shader.
std::vector<u32> MilesRecordPointerCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeVop1(0x02, 22, Vgpr(1)));  // v_readfirstlane_b32 s22, v1
  code.push_back(EncodeVopc(0xc2u, 22, 1));       // v_cmp_eq_u32 vcc, s22, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));     // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 22, InlineU32(4))); // s_lshl_b32 vcc_lo, s22, 4
  // s_buffer_load_dwordx4 s[12:15], s[4:7], vcc_lo: the record
  code.push_back(EncodeSmem0(0x0au, 12, 2));
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  // s_load_dwordx8 s[32:39], s[14:15], 64 and s_load_dwordx4 s[40:43], s[14:15], 304: the T# at a
  // non-zero offset of the material, the S# at an unrelated one
  code.push_back(EncodeSmem0(0x03u, 32, 7));
  code.push_back(EncodeSmem1(64, 125));
  code.push_back(EncodeSmem0(0x02u, 40, 7));
  code.push_back(EncodeSmem1(304, 125));
  // s_load_dwordx4 s[44:47], s[14:15], 48: the material's vertex V#
  code.push_back(EncodeSmem0(0x02u, 44, 7));
  code.push_back(EncodeSmem1(48, 125));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[32:39], s[40:43]
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 40 / 4));
  // A second sample of the same T# (its words have two users)
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v7, v[3:5], s[32:39], s[40:43]
  code.push_back(EncodeMimg1(7, 3, 32 / 4, 40 / 4));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeVop2(0x03u, 2, Vgpr(7), 2)); // v_add_f32 v2, v7, v2
  // buffer_load_format_x v6, v0, s[44:47] idxen: formatted through the record's V#
  code.push_back(EncodeMubuf0(0x00u, 0, true, false));
  code.push_back(EncodeMubuf1(6, 44 / 4, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeVop2(0x03u, 2, Vgpr(6), 2)); // v_add_f32 v2, v6, v2
  // buffer_load_format_xyzw v[8:11], v0, s[44:47] idxen: four channels through the same V#
  code.push_back(EncodeMubuf0(0x03u, 0, true, false));
  code.push_back(EncodeMubuf1(8, 44 / 4, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeVop2(0x03u, 2, Vgpr(11), 2)); // v_add_f32 v2, v11, v2
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckMilesRecordPointer() {
  constexpr const char *name = "BindlessMilesRecordPointer";
  const auto code = MilesRecordPointerCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.dump_ir = true;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ScopedCodegenOptions scope(codegen);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  Require(name, "without bindless images",
          ShaderRecompiler::TranslateProgram(code, options).skip_dispatch,
          "the record-pointer T# was not skipped without bindless images");
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the record-pointer T# was skipped with bindless images");
  const auto &images = translated.program.info.images;
  Require(name, "bindless plan", images.size() == 1u, "expected one image");
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "bindless plan",
          source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
              source.indirect_descriptor->record_pointer == 8u &&
              source.indirect_descriptor->table_stride == 16u &&
              source.indirect_descriptor->table_offset == 64u,
          "the image is not a record pointer heap");
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
          "the record pointer image could not be materialized");
  Require(name, "bindless heap",
          resources.bindless_heaps.size() == 1u && resources.bindless_heaps[0].base == 0x4000u &&
              resources.bindless_heaps[0].stride == 16u &&
              resources.bindless_heaps[0].record_pointer == 8u &&
              resources.bindless_heaps[0].table_offset == 64u,
          "the snapshot does not describe the record pointer heap");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V",
          CountSpirv(result.spirv, "\"bindless_translation\"") == 1u,
          "the sample does not go through the bindless arrays");
  // Both formatted loads call the module's decoder (one function, DefineIndirectFormatFunction).
  if (const char* path = std::getenv("KYTY_BINDLESS_TEST_SPV"); path != nullptr) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::string text;
    if (tools.Disassemble(result.spirv, &text) && !text.empty()) {
      if (std::FILE* out = std::fopen(path, "wb"); out != nullptr) {
        std::fwrite(text.data(), 1, text.size(), out);
        std::fclose(out);
      }
    }
    if (std::FILE* out = std::fopen((std::string(path) + ".ir").c_str(), "wb"); out != nullptr) {
      std::fwrite(result.ir_dump.data(), 1, result.ir_dump.size(), out);
      std::fclose(out);
    }
  }
  std::printf("[bindless] formatted channel selects: %zu\n", CountSpirv(result.spirv, "OpUMod"));
  Require(name, "formatted loads", CountSpirv(result.spirv, "OpFunctionCall") >= 2u &&
              CountSpirv(result.spirv, "OpUMod") >= 1u,
          "expected the X and XYZW formatted loads to call the decoder (OpUMod " +
              std::to_string(CountSpirv(result.spirv, "OpUMod")) + ")");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// The nested form (2 of the 218 callees): the material's V# at +48 points at a buffer whose
// dwords 48-63 are the V# of a T# heap; the T# is S_BUFFER_LOAD_DWORDX8 at an inner key << 5.
std::vector<u32> MilesNestedHeapCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeVop1(0x02, 22, Vgpr(1)));  // v_readfirstlane_b32 s22, v1
  code.push_back(EncodeVopc(0xc2u, 22, 1));       // v_cmp_eq_u32 vcc, s22, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));     // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 22, InlineU32(4))); // s_lshl_b32 vcc_lo, s22, 4
  code.push_back(EncodeSmem0(0x0au, 12, 2));      // s_buffer_load_dwordx4 s[12:15], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeSmem0(0x02u, 44, 7));      // s_load_dwordx4 s[44:47], s[14:15], 48
  code.push_back(EncodeSmem1(48, 125));
  code.push_back(EncodeSmem0(0x02u, 40, 7));      // s_load_dwordx4 s[40:43], s[14:15], 304 (S#)
  code.push_back(EncodeSmem1(304, 125));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeSmem0(0x0au, 60, 22));     // s_buffer_load_dwordx4 s[60:63], s[44:47], 48
  code.push_back(EncodeSmem1(48, 125));
  code.push_back(EncodeVop1(0x02, 23, Vgpr(0)));  // v_readfirstlane_b32 s23, v0: the inner key
  code.push_back(EncodeSop2(0x1eu, 26, 23, InlineU32(5))); // s_lshl_b32 s26, s23, 5
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeSmem0(0x0bu, 64, 30));     // s_buffer_load_dwordx8 s[64:71], s[60:63], s26
  code.push_back(EncodeSmem1(0, 26));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[64:71], s[40:43]
  code.push_back(EncodeMimg1(2, 3, 64 / 4, 40 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckMilesNestedHeap() {
  constexpr const char *name = "BindlessMilesNestedHeap";
  const auto code = MilesNestedHeapCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ScopedCodegenOptions scope(codegen);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the nested heap T# was skipped with bindless images");
  const auto &images = translated.program.info.images;
  Require(name, "bindless plan", images.size() == 1u, "expected one image");
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "bindless plan",
          source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
              source.indirect_descriptor->record_pointer == 8u &&
              source.indirect_descriptor->nested_vsharp == 48u &&
              source.indirect_descriptor->nested_heap == 48u &&
              source.indirect_descriptor->nested_keys != 0u,
          "the image is not a nested heap");
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
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              resources.bindless_heaps.size() == 1u &&
              resources.bindless_heaps[0].nested_vsharp == 48u &&
              resources.bindless_heaps[0].nested_heap == 48u,
          "the nested heap did not materialize");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V", CountSpirv(result.spirv, "\"bindless_translation\"") == 1u,
          "the sample does not go through the bindless arrays");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// PPSA03671's material pixel shaders (KYTY_HEAP_KEY_MASK): the heap key is a V_READFIRSTLANE of a
// lane value, and the T# offset is masked after the scale: S_LSHL_B32 5, then S_AND_B32 0x01ffffe0,
// then S_BUFFER_LOAD_DWORDX8 from the heap's V#. `uniform_key` takes the key from user data s12
// instead (a value the SRT walk evaluates before the draw), which must keep its flat slot.
// `word0_data`: no mask, but T# word 0 is also stored as a value (0x885fa47aec73f696 spills it with
// V_WRITELANE): before, a word 0 with another use had no plan and the program was skipped.
std::vector<u32> MaskedHeapKeyCode(bool uniform_key, bool word0_data = false) {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  if (uniform_key) {
    code.push_back(EncodeSop1(0x03u, 22, 12)); // s_mov_b32 s22, s12
  } else {
    code.push_back(EncodeVop1(0x02, 22, Vgpr(1))); // v_readfirstlane_b32 s22, v1
  }
  code.push_back(EncodeSop2(0x1eu, 23, 22, InlineU32(5))); // s_lshl_b32 s23, s22, 5
  if (!word0_data) {
    code.push_back(EncodeSop2(0x0eu, 23, 23, 255)); // s_and_b32 s23, s23, 0x01ffffe0
    code.push_back(0x01ffffe0u);
  }
  code.push_back(EncodeSmem0(0x0bu, 32, 2)); // s_buffer_load_dwordx8 s[32:39], s[4:7], s23
  code.push_back(EncodeSmem1(0, 23));
  code.push_back(EncodeSopp(0x0cu, 0));
  // image_sample_l v2, v[3:5], s[32:39], s[8:11] dmask 0x1
  code.push_back(EncodeMimg0(0x24u, 0x1u));
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  if (word0_data) {
    code.push_back(EncodeVop1(0x01, 6, 32)); // v_mov_b32 v6, s32
    AppendVMovU32(&code, 7, 4);
    AppendBufferStoreDword(&code, 6, 7);
  }
  AppendEnd(&code);
  return code;
}

void CheckMaskedHeapKey() {
  constexpr const char *name = "BindlessMaskedHeapKey";
  const auto code = MaskedHeapKeyCode(false);
  auto user_data = MilesHeapUserData();
  user_data[12] = 3u;
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  const auto is_bindless = [](const ShaderRecompiler::TranslateResult &translated) {
    const auto &images = translated.program.info.images;
    if (images.size() != 1u) return false;
    const auto &source = translated.program.descriptor_sources.at(images[0].source);
    return source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
           source.indirect_descriptor->table_offset == 0u &&
           source.indirect_descriptor->table_stride == 32u;
  };
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);

  // Switch off: the masked offset has no plan and the program is skipped, as before.
  {
    codegen.heap_key_mask = false;
    ScopedCodegenOptions scope(codegen);
    const auto skipped = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "switch off", skipped.skip_dispatch,
            "the masked heap key was planned without KYTY_HEAP_KEY_MASK");
  }
  codegen.heap_key_mask = true;
  ScopedCodegenOptions scope(codegen);
  // Without bindless images there is still no plan.
  {
    ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
    const auto skipped = ShaderRecompiler::TranslateProgram(code, options);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
    Require(name, "without bindless images", skipped.skip_dispatch,
            "the masked heap key was planned without bindless images");
  }
  // A key from user data keeps its flat slot (no indirect plan), switch on or off.
  {
    const auto uniform_code = MaskedHeapKeyCode(true);
    const auto translated = ShaderRecompiler::TranslateProgram(uniform_code, options);
    Require(name, "uniform key",
            !translated.skip_dispatch && translated.program.info.images.size() == 1u &&
                !translated.program.descriptor_sources.at(translated.program.info.images[0].source)
                     .indirect_descriptor.has_value(),
            "a key the SRT walk evaluates no longer has its flat slot");
  }
  // T# word 0 also used as a value: skipped without the switch, a bindless lookup with it, and a
  // flat slot for a key the SRT walk evaluates.
  {
    const auto word0_code = MaskedHeapKeyCode(false, true);
    auto off = codegen;
    off.heap_key_mask = false;
    {
      ScopedCodegenOptions off_scope(off);
      Require(name, "word 0 switch off",
              ShaderRecompiler::TranslateProgram(word0_code, options).skip_dispatch,
              "a shared T# word 0 was planned without KYTY_HEAP_KEY_MASK");
    }
    const auto shared = ShaderRecompiler::TranslateProgram(word0_code, options);
    Require(name, "word 0", !shared.skip_dispatch && is_bindless(shared),
            "a shared T# word 0 with a lane key is not a bindless heap lookup");
    const auto uniform = ShaderRecompiler::TranslateProgram(MaskedHeapKeyCode(true, true), options);
    Require(name, "word 0 uniform key",
            !uniform.skip_dispatch && uniform.program.info.images.size() == 1u &&
                !uniform.program.descriptor_sources.at(uniform.program.info.images[0].source)
                     .indirect_descriptor.has_value(),
            "a shared T# word 0 with a key the SRT walk evaluates no longer has its flat slot");
  }

  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the masked heap key was skipped with KYTY_HEAP_KEY_MASK");
  Require(name, "bindless plan", is_bindless(translated),
          "the image is not a bindless heap lookup at stride 32");
  // The shader looks up key & 0xfffff: (key << 5) & 0x01ffffe0 == (key & 0xfffff) << 5.
  u32 masks = 0;
  for (const auto *block : translated.program.blocks) {
    for (const auto &inst : *block) {
      if (inst.GetOpcode() != ShaderRecompiler::IR::ValueOpcode::BitwiseAnd32) continue;
      const auto mask = inst.Arg(1).Resolve();
      masks += mask.IsImmediate() && mask.U32() == 0x000fffffu;
    }
  }
  Require(name, "masked key", masks >= 1u, "no key & 0xfffff in the program");

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
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              resources.bindless_heaps.size() == 1u && resources.bindless_heaps[0].base == 0x4000u &&
              resources.bindless_heaps[0].table_offset == 0u,
          "the masked heap did not materialize");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V",
          CountSpirv(result.spirv, "\"bindless_translation\"") == 1u &&
              CountSpirv(result.spirv, "OpImageSampleExplicitLod") >= 1u,
          "the sample does not go through the bindless arrays");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// PPSA03671's material callbacks (KYTY_HEAP_KEY_MASK): the waterfall record {function, data} at
// key << 4 gives the data pointer s[14:15]; the heap V# is S_LOAD_DWORDX4 at +16 through it (no
// middle buffer, unlike Miles' nested heap) and the T# S_BUFFER_LOAD_DWORDX8 at the inner key << 5.
std::vector<u32> RecordHeapCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeVop1(0x02, 22, Vgpr(1)));  // v_readfirstlane_b32 s22, v1
  code.push_back(EncodeVopc(0xc2u, 22, 1));       // v_cmp_eq_u32 vcc, s22, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));     // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 22, InlineU32(4))); // s_lshl_b32 vcc_lo, s22, 4
  code.push_back(EncodeSmem0(0x0au, 12, 2));      // s_buffer_load_dwordx4 s[12:15], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeSmem0(0x02u, 60, 7));      // s_load_dwordx4 s[60:63], s[14:15], 16 (heap V#)
  code.push_back(EncodeSmem1(16, 125));
  code.push_back(EncodeSmem0(0x02u, 40, 7));      // s_load_dwordx4 s[40:43], s[14:15], 32 (S#)
  code.push_back(EncodeSmem1(32, 125));
  code.push_back(EncodeVop1(0x02, 23, Vgpr(0)));  // v_readfirstlane_b32 s23, v0: the inner key
  code.push_back(EncodeSop2(0x1eu, 26, 23, InlineU32(5))); // s_lshl_b32 s26, s23, 5
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeSmem0(0x0bu, 64, 30));     // s_buffer_load_dwordx8 s[64:71], s[60:63], s26
  code.push_back(EncodeSmem1(0, 26));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[64:71], s[40:43]
  code.push_back(EncodeMimg1(2, 3, 64 / 4, 40 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckRecordHeap() {
  constexpr const char *name = "BindlessRecordHeap";
  const auto code = RecordHeapCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  {
    codegen.heap_key_mask = false;
    ScopedCodegenOptions scope(codegen);
    Require(name, "switch off", ShaderRecompiler::TranslateProgram(code, options).skip_dispatch,
            "the record heap was planned without KYTY_HEAP_KEY_MASK");
  }
  codegen.heap_key_mask = true;
  ScopedCodegenOptions scope(codegen);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the record heap T# was skipped with KYTY_HEAP_KEY_MASK");
  const auto &images = translated.program.info.images;
  Require(name, "bindless plan", images.size() == 1u, "expected one image");
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "bindless plan",
          source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
              source.indirect_descriptor->record_pointer == 8u &&
              source.indirect_descriptor->table_stride == 16u &&
              source.indirect_descriptor->nested_vsharp == 16u &&
              source.indirect_descriptor->nested_heap == UINT32_MAX &&
              source.indirect_descriptor->nested_keys != 0u,
          "the image is not a record heap");
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
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              resources.bindless_heaps.size() == 1u &&
              resources.bindless_heaps[0].nested_vsharp == 16u &&
              resources.bindless_heaps[0].nested_heap == UINT32_MAX,
          "the record heap did not materialize");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V", CountSpirv(result.spirv, "\"bindless_translation\"") == 1u,
          "the sample does not go through the bindless arrays");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// KYTY_BINDLESS_3D_AS_ARRAY (PPSA03671's forward pixel shader bf3e, adapted from KytyPS5-Frank
// fed216ce8): the Miles waterfall with a 3D IMAGE_SAMPLE_L, v[3:6] = (u, v, w, lod). A 3D heap's
// key may hold a 2D or 2D-array T#; with the switch the shader declares the 2D-array view too,
// samples both and selects by the translation entry's flag.
std::vector<u32> VolumeHeapSampleCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u); // u
  AppendVMovLiteral(&code, 4, 0x3f000000u); // v
  AppendVMovLiteral(&code, 5, 0x3f800000u); // w (layer 1 of a 2D array)
  AppendVMovLiteral(&code, 6, 0x00000000u); // lod
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeSop1(0x14u, 22, 20)); // s_ff1_i32_b64 s22, s[20:21]
  code.push_back(EncodeVop3Word0(0x360u, 23)); // v_readlane_b32 s23, v1, s22
  code.push_back(EncodeVop3Word1(Vgpr(1), 22));
  code.push_back(EncodeVopc(0xc2u, 23, 1));    // v_cmp_eq_u32 vcc, s23, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));  // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 23, InlineU32(5))); // s_lshl_b32 vcc_lo, s23, 5
  code.push_back(EncodeSmem0(0x0bu, 32, 2)); // s_buffer_load_dwordx8 s[32:39], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  // image_sample_l v2, v[3:6], s[32:39], s[8:11] dmask 0x1 dim:3D
  code.push_back(EncodeMimg0(0x24u, 0x1u, 0, false, 2));
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc1 loop
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckVolumeOverArray() {
  constexpr const char *name = "BindlessVolumeOverArray";
  const auto code = VolumeHeapSampleCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  const auto compile = [&](bool as_array) {
    codegen.bindless_3d_as_array = as_array;
    ScopedCodegenOptions scope(codegen);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "resource tracking", !translated.skip_dispatch,
            "the 3D heap-indexed T# was skipped with bindless images");
    const auto &images = translated.program.info.images;
    Require(name, "bindless plan",
            images.size() == 1u &&
                images[0].dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D,
            "expected one 3D image");
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
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
    Require(name, "SPIR-V emit", !result.spirv.empty(), "no SPIR-V");
    ValidateSpirv(name, result.spirv);
    return result.spirv;
  };
  // Switch off: one 3D view, one sample, as before.
  const auto off = compile(false);
  Require(name, "switch off",
          CountSpirv(off, "\"bindless_images\"") == 1u &&
              CountSpirv(off, "OpImageSampleExplicitLod") == 1u,
          "the 3D sample declares another view without KYTY_BINDLESS_3D_AS_ARRAY");
  // Switch on: the 3D and 2D-array views, both sampled, the flag selects.
  const auto on = compile(true);
  Require(name, "both views",
          CountSpirv(on, "\"bindless_images\"") == 2u &&
              CountSpirv(on, "OpImageSampleExplicitLod") == 2u &&
              CountSpirv(on, "1073741824") >= 1u,
          "the 3D sample does not read the 2D-array view selected by BindlessSlotArrayFlag");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, on.size());
}

// KYTY_RUNTIME_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): GT7's material functions
// load their T#s through the call record's data pointer, which no heap plan covers. Here the
// pointer is a lane's buffer value (V_READFIRSTLANE_B32): s_load_dwordx8 s[32:39], s[40:41], 0
// then IMAGE_SAMPLE_L. Without the switch the program is skipped; with it the image is the
// runtime image, looked up by its T# dwords in the runtime image table.
std::vector<u32> RuntimeImageCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeVop1(0x02, 40, Vgpr(1)));     // v_readfirstlane_b32 s40, v1
  code.push_back(EncodeSop1(0x03u, 41, InlineU32(0))); // s_mov_b32 s41, 0
  code.push_back(EncodeSmem0(0x03u, 32, 20));        // s_load_dwordx8 s[32:39], s[40:41], 0
  code.push_back(EncodeSmem1(0, 125));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[32:39], s[8:11]
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckRuntimeImage() {
  constexpr const char *name = "BindlessRuntimeImage";
  const auto code = RuntimeImageCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  {
    codegen.runtime_images = false;
    ScopedCodegenOptions scope(codegen);
    Require(name, "switch off", ShaderRecompiler::TranslateProgram(code, options).skip_dispatch,
            "the runtime T# was planned without KYTY_RUNTIME_IMAGES");
  }
  codegen.runtime_images = true;
  ScopedCodegenOptions scope(codegen);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the runtime T# was skipped with KYTY_RUNTIME_IMAGES");
  const auto &images = translated.program.info.images;
  Require(name, "runtime image",
          images.size() == 1u && images[0].runtime && images[0].bindless,
          "expected one runtime image");
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
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              resources.bindless_heaps.size() == 1u && resources.bindless_heaps[0].runtime &&
              specialization.images.size() == 1u && specialization.images[0].runtime,
          "the runtime image did not materialize as a runtime heap use");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "runtime SPIR-V",
          CountSpirv(result.spirv, "\"runtime_image_table\"") == 1u &&
              CountSpirv(result.spirv, "\"runtime_image_requests\"") == 1u &&
              CountSpirv(result.spirv, "OpAtomicIAdd") >= 1u,
          "the sample does not look its T# up in the runtime image table");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// KYTY_RUNTIME_INT_IMAGES (adapted from usa-ratt/KytyPS5-GT7 2fa6c6514): an IMAGE_LOAD of a runtime
// image reads its slot through uint / sint aliases of the bindless array when the T# format is an
// integer one. Off: one float array, as before; on: three arrays and a valid module.
void CheckRuntimeIntImage() {
  constexpr const char *name = "BindlessRuntimeIntImage";
  auto code = RuntimeImageCode();
  // image_load v2, v[3:4], s[32:39] (the sample's place; integer coordinates 0, 0).
  for (size_t i = 0; i + 1u < code.size(); ++i) {
    if (code[i] == EncodeMimg0(0x24u, 0x1u)) {
      code[i] = EncodeMimg0(0x00u, 0x1u);
      code[i + 1u] = EncodeMimg1(2, 3, 32 / 4, 0);
      break;
    }
  }
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  const auto compile = [&](bool integer) {
    auto codegen = GetCodegenOptions();
    codegen.srt_variant_reads = true;
    codegen.runtime_images = true;
    codegen.runtime_int_images = integer;
    ScopedCodegenOptions scope(codegen);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "resource tracking",
            !translated.skip_dispatch && translated.program.info.images.size() == 1u &&
                translated.program.info.images[0].runtime,
            "the runtime T# of the load was not a runtime image");
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
            "the runtime image did not materialize");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
    ValidateSpirv(name, result.spirv);
    return result.spirv;
  };
  const auto off = compile(false);
  Require(name, "switch off", CountSpirv(off, "\"bindless_images\"") == 1u,
          "the load declares integer aliases without KYTY_RUNTIME_INT_IMAGES");
  const auto on = compile(true);
  Require(name, "integer aliases",
          CountSpirv(on, "\"bindless_images\"") == 3u &&
              CountSpirv(on, "OpImageFetch") + CountSpirv(on, "OpImageRead") >= 3u,
          "the load does not read the uint and sint aliases selected by the T# format");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, on.size());
}

// KYTY_BINDLESS_STRIDE (adapted from usa-ratt/KytyPS5-GT7 a6cd035f0): GT7's material shaders
// select their T#s from 188-byte material records, key * 188 (S_MUL_I32) instead of key << 5. The
// Miles waterfall with that multiply: without the switch the program is skipped; with it the heap
// is a bindless heap whose keys are 188 bytes apart.
std::vector<u32> StridedHeapSampleCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeSop1(0x14u, 22, 20));   // s_ff1_i32_b64 s22, s[20:21]
  code.push_back(EncodeVop3Word0(0x360u, 23)); // v_readlane_b32 s23, v1, s22
  code.push_back(EncodeVop3Word1(Vgpr(1), 22));
  code.push_back(EncodeVopc(0xc2u, 23, 1));    // v_cmp_eq_u32 vcc, s23, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));  // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x26u, 106, 23, 255u)); // s_mul_i32 vcc_lo, s23, 188
  code.push_back(188u);
  code.push_back(EncodeSmem0(0x0bu, 32, 2)); // s_buffer_load_dwordx8 s[32:39], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[32:39], s[8:11]
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc1 loop
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckStridedHeap() {
  constexpr const char *name = "BindlessStridedHeap";
  const auto code = StridedHeapSampleCode();
  const auto user_data = MilesHeapUserData();
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  {
    codegen.bindless_strided = false;
    ScopedCodegenOptions scope(codegen);
    Require(name, "switch off", ShaderRecompiler::TranslateProgram(code, options).skip_dispatch,
            "the 188-byte heap record was planned without KYTY_BINDLESS_STRIDE");
  }
  codegen.bindless_strided = true;
  ScopedCodegenOptions scope(codegen);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the 188-byte heap record was skipped with KYTY_BINDLESS_STRIDE");
  const auto &images = translated.program.info.images;
  Require(name, "bindless plan", images.size() == 1u, "expected one image");
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "bindless plan",
          source.indirect_descriptor.has_value() && source.indirect_descriptor->bindless &&
              source.indirect_descriptor->record_pointer == UINT32_MAX &&
              source.indirect_descriptor->table_stride == 188u,
          "the image is not a strided bindless heap");
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
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              resources.bindless_heaps.size() == 1u && resources.bindless_heaps[0].stride == 188u &&
              resources.bindless_heaps[0].record_pointer == UINT32_MAX,
          "the strided heap did not materialize with its stride");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless SPIR-V", CountSpirv(result.spirv, "\"bindless_translation\"") == 1u,
          "the sample does not go through the bindless arrays");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// KYTY_WORKGROUP_IMAGES (adapted from upstream KytyPS5 0ee880090, nmzik; CPU only, so it lives with
// the bindless cases): GT7's CS 0xe042c5de1f18d724 reads a storage T# per Z workgroup from
// 148-byte records (S_MUL_I32 of the workgroup id, S_BUFFER_LOAD_DWORDX8) and stores to it. Without
// the switch the program is skipped; with it the T# table is enumerated over the dispatch's Z
// workgroups and each group stores to its own image.
void CheckWorkgroupStorageImage() {
  constexpr const char *name = "WorkgroupStorageImage";
  std::vector<u32> code;
  code.push_back(EncodeSop2(0x26u, 106, 12, 255u)); // s_mul_i32 vcc_lo, s12, 148
  code.push_back(148u);
  code.push_back(EncodeSmem0(0x0bu, 32, 2)); // s_buffer_load_dwordx8 s[32:39], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 2, 0u); // x
  AppendVMovLiteral(&code, 3, 0u); // y
  for (u32 component = 0; component < 4u; component++) {
    AppendVMovLiteral(&code, 4 + component, 0x3f800000u);
  }
  // image_store v[4:7], v[2:3], s[32:39] dmask 0xf dim 2D
  code.push_back(EncodeMimg0(0x08u, 0xfu, 0, false, 1));
  code.push_back(EncodeMimg1(4, 2, 32 / 4, 0));
  AppendEnd(&code);
  // s[4:7]: the record table, 0x400 bytes from 0x4000 (eight user SGPRs; s12 is the group id).
  std::array<u32, 8> user_data{};
  user_data[4] = 0x4000u;
  user_data[6] = 0x400u;
  user_data[7] = 0x5204u;
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  compute_info.group_id[2] = true;
  compute_info.workgroup_register = 12;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  {
    codegen.workgroup_images = false;
    ScopedCodegenOptions scope(codegen);
    Require(name, "switch off", ShaderRecompiler::TranslateProgram(code, options).skip_dispatch,
            "the workgroup-indexed T# was planned without KYTY_WORKGROUP_IMAGES");
  }
  codegen.workgroup_images = true;
  ScopedCodegenOptions scope(codegen);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the workgroup-indexed T# was skipped with KYTY_WORKGROUP_IMAGES");
  const auto &images = translated.program.info.images;
  Require(name, "plan",
          images.size() == 1u && images[0].written &&
              images[0].resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage,
          "expected one storage image");
  const auto &source = translated.program.descriptor_sources.at(images[0].source);
  Require(name, "plan",
          source.indirect_descriptor.has_value() && !source.indirect_descriptor->bindless &&
              source.indirect_descriptor->workgroup_axis == 2u &&
              source.indirect_descriptor->table_stride == 148u &&
              source.indirect_descriptor->table_offset == 0u,
          "the image is not a table indexed by workgroup Z");
  const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  Require(name, "plan", ShaderRecompiler::IR::PlanReadsWorkgroupCounts(plan),
          "the plan does not report that it reads the workgroup counts");
  // Two Z groups, two images (32-bit float RGBA, 1x1) in records 148 bytes apart.
  std::vector<u32> memory(0x4000u, 0u);
  for (u32 group = 0; group < 2u; group++) {
    auto *tsharp = memory.data() + (0x4000u + group * 148u) / 4u;
    tsharp[0] = 0x100u * (group + 1u);
    tsharp[1] = static_cast<u32>(Prospero::BufferFormat::k32_32_32_32Float) << 20u;
    tsharp[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u);
  }
  const std::array<u32, 3> groups{1u, 1u, 2u};
  ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(code.data()),
      .read_memory = ReadTestMemory,
      .userdata = &memory,
      .read_specialization_memory = ReadTestMemory,
  };
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  // No counts (an indirect dispatch whose arguments were not read): no materialization.
  (void)ShaderRecompiler::IR::TakeWorkgroupCountsMissing();
  Require(name, "no counts",
          !ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              ShaderRecompiler::IR::TakeWorkgroupCountsMissing(),
          "the table materialized without workgroup counts");
  runtime.workgroup_counts = groups;
  Require(name, "resource materialization",
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
              specialization.images.size() == 2u && resources.images.size() == 2u &&
              resources.images[0].dwords[0] == 0x100u && resources.images[1].dwords[0] == 0x200u &&
              specialization.images[0].indirect_search_iterations != 0u &&
              specialization.images[1].indirect_root == 0u,
          "the two Z groups did not materialize one image each");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ValidateSpirv(name, result.spirv);
  Require(name, "SPIR-V", CountSpirv(result.spirv, "OpImageWrite") == 2u,
          "the store does not write each group's image");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

// KYTY_CALL_TRACE_REACH (adapted from usa-ratt/KytyPS5-GT7 a6cd035f0; CPU only, so it lives with
// the bindless cases): GT7's CS 0x2218a07ab01d4f46 loads its call table V# from the user-data
// pointer s[12:13] after a branch, and writes s12 again only after the call, in a loop that does
// not reach the load. Without the switch (and without KYTY_CALL_TRACE_WAVE32) s12 has no unique
// definition; with it no writer reaches the load, so s[12:13] holds its entry value.
void CheckCallTraceReach() {
  constexpr const char *name = "CallTraceReach";
  std::vector<u32> code;
  code.push_back(EncodeSopp(0x04, 0));                    // s_cbranch_scc0 to the next block
  code.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4))); // s_lshl_b32 vcc_lo, s6, 4
  code.push_back(EncodeSmem0(0x02, 0, 6));                // s_load_dwordx4 s[0:3], s[12:13], 144
  code.push_back(EncodeSmem1(144, 125));
  code.push_back(EncodeSmem0(0x0a, 0, 0));                // s_buffer_load_dwordx4 s[0:3], s[0:3], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSMovB32(14, 0));
  code.push_back(EncodeSMovB32(15, 1));
  code.push_back(0xbe8e210eu);                            // s_swappc_b64 s[14:15], s[14:15]
  const auto loop = code.size();
  code.push_back(EncodeSop2(0x00, 12, 12, InlineU32(1))); // s_add_u32 s12, s12, 1
  code.push_back(EncodeSmem0(0x03, 8, 6));                // s_load_dwordx8 s[8:15], s[12:13], 64
  code.push_back(EncodeSmem1(64, 125));
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc1 loop
  AppendEnd(&code);
  std::array<u32, 16> user_data{};
  ShaderComputeInputInfo compute_info{};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  codegen.shader_calls = false;
  codegen.call_trace_wave32 = false;
  codegen.call_trace_reach = false;
  {
    ScopedCodegenOptions scope(codegen);
    const auto before = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "switch off", before.skip_dispatch && before.call_tables.empty(),
            "the table was traced without KYTY_CALL_TRACE_REACH");
  }
  codegen.call_trace_reach = true;
  {
    ScopedCodegenOptions scope(codegen);
    const auto result = ShaderRecompiler::TranslateProgram(code, options);
    const auto &tables = result.call_tables;
    Require(name, "table",
            result.skip_dispatch && tables.size() == 1u && tables[0].user_sgpr == 12u &&
                tables[0].steps.size() == 1u && !tables[0].steps[0].buffer &&
                tables[0].steps[0].offset == 144,
            "the call table was not traced by reachability (" + std::to_string(tables.size()) +
                " found)");
    // A loop back to the load (through the call) keeps the range unknown.
    std::vector<u32> looping = code;
    looping.pop_back(); // s_endpgm
    const auto to_lshl = 1 - static_cast<int32_t>(looping.size() + 1u);
    looping.push_back(EncodeSopp(0x04, static_cast<u32>(to_lshl) & 0xffffu)); // s_cbranch_scc0
    AppendEnd(&looping);
    const auto again = ShaderRecompiler::TranslateProgram(looping, options);
    Require(name, "loop back", again.call_tables.empty(),
            "a write that can loop back to the load was ignored");
  }
  std::printf("[bindless] %-32s ok\n", name);
}

// Miles Morales' sky material samples a cube-array T# (384 faces) with a 2D-array instruction,
// computing face and slice itself; the GPU reads the cube as its array of faces. Such a heap key
// must reach the 2D-array binding (it sampled the placeholder: a black reflected sky). The
// translation buffer must hold the RT call tables' record heaps (1 Mi filled up).
void CheckCubeOverArrayKey() {
  const char *name = "cube T# in a 2D-array heap";
  ShaderTextureResource cube{};
  cube.fields[0] = 0x11368000u;
  cube.fields[1] = static_cast<u32>(Prospero::BufferFormat::k16_16_16_16Float) << 20u;
  cube.fields[3] = static_cast<u32>(Prospero::ImageType::kCube) << 28u;
  cube.fields[4] = 383u;
  Require(name, "2D-array binding",
          BindlessKeyCompatible(cube, ShaderRecompiler::IR::BindlessImages2DArray),
          "a cube T# was refused by the 2D-array binding");
  Require(name, "cube binding", BindlessKeyCompatible(cube, ShaderRecompiler::IR::BindlessImagesCube),
          "a cube T# was refused by the cube binding");
  Require(name, "2D binding", !BindlessKeyCompatible(cube, ShaderRecompiler::IR::BindlessImages2D),
          "a cube T# was accepted by the 2D binding");
  Require(name, "3D binding", !BindlessKeyCompatible(cube, ShaderRecompiler::IR::BindlessImages3D),
          "a cube T# was accepted by the 3D binding");
  ShaderTextureResource flat = cube;
  flat.fields[3] = static_cast<u32>(Prospero::ImageType::kColor2D) << 28u;
  Require(name, "2D T# in the cube binding",
          !BindlessKeyCompatible(flat, ShaderRecompiler::IR::BindlessImagesCube),
          "a 2D T# was accepted by the cube binding");
  if (std::getenv("KYTY_BINDLESS_TRANSLATION_MI") == nullptr) {
    Require(name, "translation capacity", BindlessTable::TranslationCapacity() == (24u << 20u),
            "the default translation buffer is not 24 Mi entries");
  }
  std::printf("[bindless] %-32s ok\n", name);
}

// KYTY_BINDLESS_SAMPLERS (a minimal port of Senaxx's mirrored sampler heaps, 59cb8a83e): the Miles
// waterfall whose S# is read from a sampler heap at the same key (S_LSHL_B32 4,
// S_BUFFER_LOAD_DWORDX4 from the heap V# in s[12:15]). Without the switch the S# takes the
// linear/wrap stand-in, as before; with it the sampler is bindless: the snapshot names the heap,
// two flattened-SRT words follow the SRT slots, and the shader indexes the set-1 sampler array.
std::vector<u32> SamplerHeapSampleCode() {
  std::vector<u32> code;
  code.push_back(EncodeMubuf0(0x0cu, 0, true, false)); // buffer_load_dword v1, v0, s[0:3] idxen
  code.push_back(EncodeMubuf1(1, 0, 0));
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendVMovLiteral(&code, 3, 0x3f000000u);
  AppendVMovLiteral(&code, 4, 0x3f000000u);
  AppendVMovLiteral(&code, 5, 0x00000000u);
  code.push_back(EncodeSop1(0x04u, 20, 126)); // s_mov_b64 s[20:21], exec
  const auto loop = code.size();
  code.push_back(EncodeSop1(0x14u, 22, 20));   // s_ff1_i32_b64 s22, s[20:21]
  code.push_back(EncodeVop3Word0(0x360u, 23)); // v_readlane_b32 s23, v1, s22
  code.push_back(EncodeVop3Word1(Vgpr(1), 22));
  code.push_back(EncodeVopc(0xc2u, 23, 1));    // v_cmp_eq_u32 vcc, s23, v1
  code.push_back(EncodeSop1(0x24u, 24, 106));  // s_and_saveexec_b64 s[24:25], vcc
  code.push_back(EncodeSop2(0x1eu, 106, 23, InlineU32(5))); // s_lshl_b32 vcc_lo, s23, 5
  code.push_back(EncodeSmem0(0x0bu, 32, 2)); // s_buffer_load_dwordx8 s[32:39], s[4:7], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSop2(0x1eu, 40, 23, InlineU32(4))); // s_lshl_b32 s40, s23, 4
  code.push_back(EncodeSmem0(0x0au, 8, 6)); // s_buffer_load_dwordx4 s[8:11], s[12:15], s40
  code.push_back(EncodeSmem1(0, 40));
  code.push_back(EncodeSopp(0x0cu, 0));
  code.push_back(EncodeMimg0(0x24u, 0x1u)); // image_sample_l v2, v[3:5], s[32:39], s[8:11]
  code.push_back(EncodeMimg1(2, 3, 32 / 4, 8 / 4));
  code.push_back(EncodeSop2(0x15u, 20, 20, 126)); // s_andn2_b64 s[20:21], s[20:21], exec
  code.push_back(EncodeSop1(0x04u, 126, 24));     // s_mov_b64 exec, s[24:25]
  const auto back = static_cast<int32_t>(loop) - static_cast<int32_t>(code.size() + 1u);
  code.push_back(EncodeSopp(0x05u, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc1 loop
  code.push_back(EncodeSopp(0x0cu, 0));
  AppendBufferStoreDword(&code, 2, 0);
  AppendEnd(&code);
  return code;
}

void CheckSamplerHeap() {
  constexpr const char *name = "BindlessSamplerHeap";
  const auto code = SamplerHeapSampleCode();
  auto user_data = MilesHeapUserData();
  // The sampler heap: 0x400 bytes (64 S# records) from 0x6000.
  user_data[12] = 0x6000u;
  user_data[13] = 0u;
  user_data[14] = 0x400u;
  user_data[15] = 0x5204u;
  ShaderComputeInputInfo compute_info{};
  compute_info.lds_size_dwords = 1024;
  compute_info.host_subgroup_size = 64;
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  std::vector<u32> memory(0x4000u, 0u);
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = options.user_data,
      .shader_base = reinterpret_cast<uint64_t>(code.data()),
      .read_memory = ReadTestMemory,
      .userdata = &memory,
      .read_specialization_memory = ReadTestMemory,
  };
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = true;
  ShaderRecompiler::IR::SetBindlessImagesEnabled(true);
  {
    // Both switches off (the default): the walk collects the S# reads as a heap buffer resource,
    // and the stand-in sampler strands them. The plan must still own every value it names (the
    // heap V# words): extracting it read erased instructions, a draw-prep thread crash.
    codegen.sampler_reads_deferred = false;
    codegen.bindless_samplers = false;
    ScopedCodegenOptions scope(codegen);
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "stranded reads", !translated.skip_dispatch,
            "the bindless image was skipped without KYTY_SAMPLER_READS_DEFER");
    Require(name, "stranded reads",
            ShaderRecompiler::IR::CountDeadPlanReferences(translated.program) == 0u,
            "the plan tables name erased instructions");
    const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSnapshot resources;
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    Require(name, "stranded reads",
            ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
                resources.samplers.size() == 1u &&
                resources.samplers[0].dwords[2] == ShaderRecompiler::IR::BindlessDefaultSampler[2],
            "the stand-in sampler did not materialize beside the stranded heap buffer");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    ValidateSpirv(name, result.spirv);
    std::printf("[bindless] %-32s stranded reads ok (%zu buffers)\n", name, resources.buffers.size());
  }
  codegen.sampler_reads_deferred = true; // KYTY_SAMPLER_READS_DEFER
  {
    // Switch off: the stand-in sampler, no sampler array, no sampler heap in the snapshot.
    codegen.bindless_samplers = false;
    ScopedCodegenOptions scope(codegen);
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "switch off", !translated.skip_dispatch,
            "the bindless image was skipped without KYTY_BINDLESS_SAMPLERS");
    const auto &samplers = translated.program.info.samplers;
    Require(name, "switch off",
            samplers.size() == 1u && !samplers[0].bindless &&
                !translated.program.descriptor_sources.at(samplers[0].source)
                     .bindless_sampler.has_value(),
            "the sampler was planned as bindless without KYTY_BINDLESS_SAMPLERS");
    const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
    ShaderRecompiler::IR::ResourceSnapshot resources;
    ShaderRecompiler::IR::ResourceSpecialization specialization;
    Require(name, "switch off",
            ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) &&
                resources.bindless_sampler_heaps.empty() && resources.samplers.size() == 1u &&
                resources.samplers[0].dwords[2] == ShaderRecompiler::IR::BindlessDefaultSampler[2],
            "the stand-in sampler did not materialize");
    // The key and output buffers only: the stranded S# reads are no buffer resource.
    Require(name, "switch off", resources.buffers.size() == 2u,
            "the sampler heap's S# reads became a buffer resource");
    auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
    ValidateSpirv(name, result.spirv);
    Require(name, "switch off",
            CountSpirv(result.spirv, "\"bindless_samplers\"") == 0u &&
                CountSpirv(result.spirv, "\"bindless_translation\"") == 1u,
            "the switch-off SPIR-V has a bindless sampler array");
    std::printf("[bindless] %-32s off ok (SPIR-V %zu words)\n", name, result.spirv.size());
  }
  codegen.bindless_samplers = true;
  ScopedCodegenOptions scope(codegen);
  auto translated = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "resource tracking", !translated.skip_dispatch,
          "the bindless image was skipped with KYTY_BINDLESS_SAMPLERS");
  const auto &samplers = translated.program.info.samplers;
  Require(name, "bindless sampler plan", samplers.size() == 1u && samplers[0].bindless,
          "expected one bindless sampler, got " + std::to_string(samplers.size()));
  const auto &source = translated.program.descriptor_sources.at(samplers[0].source);
  Require(name, "bindless sampler plan",
          source.bindless_sampler.has_value() && source.bindless_sampler->table_offset == 0u &&
              source.bindless_sampler->stride == 16u,
          "the sampler is not a 16-byte sampler heap lookup");
  const auto plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  Require(name, "resource materialization",
          ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization),
          "the bindless sampler could not be materialized");
  Require(name, "resource materialization", resources.buffers.size() == 2u,
          "the sampler heap's S# reads became a buffer resource");
  const auto expected_offset = static_cast<u32>(plan.srt_reads.size());
  Require(name, "sampler heap",
          resources.bindless_sampler_heaps.size() == 1u &&
              resources.bindless_sampler_heaps[0].base == 0x6000u &&
              resources.bindless_sampler_heaps[0].size == 0x400u &&
              resources.bindless_sampler_heaps[0].stride == 16u &&
              resources.bindless_sampler_heaps[0].sampler == 0u &&
              resources.bindless_sampler_heaps[0].mapping_offset == expected_offset &&
              resources.flattened_srt.size() >= expected_offset + 2u &&
              resources.flattened_srt[expected_offset] == 0u &&
              resources.bindless_heaps.size() == 1u &&
              resources.bindless_heaps[0].mapping_offset >= expected_offset + 2u,
          "the snapshot does not describe the sampler heap");
  std::vector<uint8_t> plan_bytes;
  ShaderRecompiler::IR::ResourcePlan decoded;
  ShaderRecompiler::IR::ResourceSnapshot decoded_resources;
  ShaderRecompiler::IR::ResourceSpecialization decoded_specialization;
  Require(name, "program cache",
          ShaderRecompiler::IR::EncodeResourcePlan(plan, plan_bytes) &&
              ShaderRecompiler::IR::DecodeResourcePlan(plan_bytes, decoded) &&
              ShaderRecompiler::IR::MaterializeResources(decoded, runtime, decoded_resources,
                                                         decoded_specialization) &&
              decoded_resources.bindless_sampler_heaps == resources.bindless_sampler_heaps &&
              decoded_specialization == specialization,
          "the decoded plan materialized another sampler heap");
  auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization);
  ShaderRecompiler::IR::SetBindlessImagesEnabled(false);
  ValidateSpirv(name, result.spirv);
  Require(name, "bindless sampler SPIR-V",
          CountSpirv(result.spirv, "\"bindless_samplers\"") == 1u &&
              CountSpirv(result.spirv, "Binding 6") >= 1u &&
              CountSpirv(result.spirv, "OpTypeRuntimeArray") >= 2u &&
              CountSpirv(result.spirv, "OpImageSampleExplicitLod") >= 1u,
          "the sample does not go through the bindless sampler array");
  Require(name, "bindless sampler",
          !result.program.info.samplers.empty() && result.program.info.samplers[0].bindless &&
              result.program.info.samplers[0].bindless_mapping_offset == expected_offset,
          "the compiled sampler does not read the materialized words");
  std::printf("[bindless] %-32s ok (SPIR-V %zu words)\n", name, result.spirv.size());
}

void RunCodegen() {
  CheckCubeOverArrayKey();
  CheckMilesHeapSample();
  CheckMilesRecordPointer();
  CheckMilesNestedHeap();
  CheckMaskedHeapKey();
  CheckRecordHeap();
  CheckVolumeOverArray();
  CheckRuntimeImage();
  CheckRuntimeIntImage();
  CheckStridedHeap();
  CheckSamplerHeap();
  CheckWorkgroupStorageImage();
  CheckCallTraceReach();
}

} // namespace BindlessTests
