// KYTY_LDS_DEVICE_BUFFER tests (included by ShaderRecompilerComputeTests.cpp after
// ShaderWaveClusterTests.inc).
//
// Astro Bot asks for 48 KiB of LDS in some compute shaders (0x7db3d8515f263f0f in the RX 9060 XT
// log of issue 16). NVIDIA allows 48 KiB of compute shared memory, AMD's Windows driver 32 KiB;
// int16.1 and AMD tests 1-2 clamped the allocation, so every LDS access above 32 KiB read zero and
// dropped its write on AMD. Since upstream b0bbaef1e such LDS lives in a device buffer, one region
// per workgroup of the dispatch. KYTY_LDS_LIMIT_OVERRIDE=32768 makes a 48 KiB host plan as AMD
// does, so these tests run the AMD path on the RTX 3090 and compare it with native shared memory.

namespace LdsDeviceBufferTests {

using CodegenTests::CountSpirv;
using CodegenTests::RequireEqualWords;
using CodegenTests::RunCollect;
using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;
using ShaderRecompiler::LdsDeviceBuffer;

constexpr u32 Threads = 256;     // four wave64 waves per workgroup
constexpr u32 Slices = 12;       // 4 KiB apart: thread t's k-th dword is LDS dword 1024k + t
constexpr u32 LdsDwords = 12288; // 48 KiB, Astro Bot's request
constexpr u32 SliceStride = 64u * Threads; // output words between slices (room for 64 groups)
constexpr u32 OutputDwords = Slices * SliceStride;

ShaderRecompiler::CodegenOptions WithLds(LdsDeviceBuffer mode, u32 limit_override) {
  auto options = GetCodegenOptions();
  options.lds_device_buffer = mode;
  options.lds_limit_override = limit_override;
  return options;
}

// Every thread writes Slices dwords (group << 20 | thread << 8 | k) at LDS byte 4096k + 4t, then
// after S_BARRIER reads the slices of thread 255 - t and stores them at k * SliceStride +
// group * 256 + t. Slices 8-11 lie above 32 KiB. The V# is s[0:3], the group id s4.
std::vector<u32> MakeLdsExchangeCode() {
  std::vector<u32> code;
  code.push_back(EncodeVop1(0x01u, 3u, 4u));                   // v_mov_b32 v3, s4
  code.push_back(EncodeVop2(0x1au, 3u, InlineU32(20u), 3u));   // v_lshlrev_b32 v3, 20, v3
  code.push_back(EncodeVop2(0x1au, 4u, InlineU32(8u), 0u));    // v_lshlrev_b32 v4, 8, v0
  code.push_back(EncodeVop2(0x25u, 5u, Vgpr(3u), 4u));         // v_add_nc_u32 v5, v3, v4
  code.push_back(EncodeVop2(0x1au, 1u, InlineU32(2u), 0u));    // v_lshlrev_b32 v1, 2, v0
  for (u32 k = 0; k < Slices; k++) {
    code.push_back(EncodeVop2(0x25u, 6u, InlineU32(k), 5u));   // v_add_nc_u32 v6, k, v5
    code.push_back(EncodeDs0(0x0du, k * 4096u));                // ds_write_b32 v1, v6 offset:4096k
    code.push_back(EncodeDs1(0u, 6u, 1u));
  }
  code.push_back(EncodeSopp(0x0cu, 0xc07fu));                   // s_waitcnt lgkmcnt(0)
  code.push_back(EncodeSopp(0x0au));                            // s_barrier
  code.push_back(EncodeVop2(0x1du, 7u, 255u, 0u));              // v_xor_b32 v7, 0xff, v0
  code.push_back(0xffu);
  code.push_back(EncodeVop2(0x1au, 8u, InlineU32(2u), 7u));    // v_lshlrev_b32 v8, 2, v7
  AppendVop3(&code, 0x346u, 12u, 4u, InlineU32(8u), Vgpr(0u)); // v_lshl_add_u32 v12, s4, 8, v0
  code.push_back(EncodeVop2(0x1au, 12u, InlineU32(2u), 12u));  // v_lshlrev_b32 v12, 2, v12
  for (u32 k = 0; k < Slices; k++) {
    code.push_back(EncodeDs0(0x36u, k * 4096u));                // ds_read_b32 v10, v8 offset:4096k
    code.push_back(EncodeDs1(10u, 0u, 8u));
    code.push_back(EncodeVop2(0x25u, 11u, 255u, 12u));          // v_add_nc_u32 v11, lit, v12
    code.push_back(k * SliceStride * 4u);
    code.push_back(EncodeSopp(0x0cu, 0xc07fu));                 // s_waitcnt lgkmcnt(0)
    code.push_back(EncodeMubuf0(0x1cu));                        // buffer_store_dword v10, v11, s[0:3]
    code.push_back(EncodeMubuf1(10u, 0u, 11u));
  }
  AppendEnd(&code);
  return code;
}

// The words MakeLdsExchangeCode stores for `groups` workgroups; `lds_dwords` < LdsDwords: the
// clamped allocation, whose accesses above it read zero and drop their writes.
std::vector<u32> ExpectedExchange(u32 groups, u32 lds_dwords = LdsDwords) {
  std::vector<u32> expected(OutputDwords, 0u);
  for (u32 g = 0; g < groups; g++) {
    for (u32 t = 0; t < Threads; t++) {
      for (u32 k = 0; k < Slices; k++) {
        const u32 partner = Threads - 1u - t;
        const bool inside = k * 1024u + partner < lds_dwords;
        expected[k * SliceStride + g * Threads + t] = inside ? (g << 20u) | (partner << 8u) | k : 0u;
      }
    }
  }
  return expected;
}

TestCase MakeLdsExchange(u32 groups) {
  using O = ShaderOpcode;
  TestCase test;
  test.name = "LdsDeviceBufferExchange";
  test.code = MakeLdsExchangeCode();
  test.initial = std::vector<u32>(OutputDwords, 0u);
  test.expected = ExpectedExchange(groups);
  test.opcodes = {O::V_MOV_B32,      O::V_LSHLREV_B32, O::V_ADD_NC_U32,      O::DS_WRITE_B32,
                  O::S_WAITCNT,      O::S_BARRIER,     O::V_XOR_B32,         O::V_LSHL_ADD_U32,
                  O::DS_READ_B32,    O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.compute_info.threads_num[0] = Threads;
  test.compute_info.threads_num[1] = 1;
  test.compute_info.threads_num[2] = 1;
  test.compute_info.thread_ids_num = 1;
  test.compute_info.group_id[0] = true;
  test.compute_info.workgroup_register = 4;
  test.compute_info.wave_size = 64;
  test.compute_info.lds_size_dwords = LdsDwords;
  test.has_compute_info = true;
  test.dispatch_x = groups;
  return test;
}

// CPU only: the placement each mode plans, the device-buffer program's SPIR-V, and that the
// workgroup-memory program is the one compiled before the switch existed.
void CheckLdsDeviceBufferCodegen() {
  constexpr const char *name = "LdsDeviceBufferCodegen";
  namespace IR = ShaderRecompiler::IR;
  {
    const ShaderRecompiler::CodegenOptions defaults;
    Require(name, "default",
            defaults.lds_device_buffer == LdsDeviceBuffer::Auto && defaults.lds_limit_override == 0,
            "KYTY_LDS_DEVICE_BUFFER does not default to auto without a limit override");
  }
  struct Row {
    LdsDeviceBuffer mode;
    u32 override_bytes;
    u32 device_bytes;
    u32 requested;
    u32 dwords;
    bool storage;
    bool clamped;
  };
  const Row rows[] = {
      {LdsDeviceBuffer::Auto, 0, 32768, 12288, 12288, true, false},   // AMD, 48 KiB
      {LdsDeviceBuffer::Auto, 0, 32768, 8192, 8192, false, false},    // AMD, 32 KiB
      {LdsDeviceBuffer::Auto, 0, 49152, 12288, 12288, false, false},  // NVIDIA
      {LdsDeviceBuffer::Auto, 32768, 49152, 12288, 12288, true, false},
      {LdsDeviceBuffer::Auto, 32768, 49152, 4096, 4096, false, false},
      {LdsDeviceBuffer::Auto, 65536, 49152, 12288, 12288, false, false}, // never raises the limit
      {LdsDeviceBuffer::Off, 0, 32768, 12288, 8192, false, true},     // int16.1
      {LdsDeviceBuffer::Off, 32768, 49152, 12288, 8192, false, true},
      {LdsDeviceBuffer::Off, 0, 49152, 12288, 12288, false, false},
      {LdsDeviceBuffer::Force, 0, 49152, 256, 256, true, false},
      {LdsDeviceBuffer::Force, 0, 49152, 0, 0, false, false},
  };
  for (const auto &row : rows) {
    ScopedCodegenOptions scoped(WithLds(row.mode, row.override_bytes));
    const auto plan = ShaderRecompiler::PlanComputeLds(row.requested, row.device_bytes);
    Require(name, "plan",
            plan.dwords == row.dwords && plan.storage == row.storage && plan.clamped == row.clamped,
            "mode " + std::to_string(static_cast<u32>(row.mode)) + " override " +
                std::to_string(row.override_bytes) + " device " + std::to_string(row.device_bytes) +
                " request " + std::to_string(row.requested) + ": dwords " +
                std::to_string(plan.dwords) + (plan.storage ? " storage" : "") +
                (plan.clamped ? " clamped" : ""));
  }
  Require(name, "binding ABI",
          IR::NativeBinding(ShaderType::Compute, IR::DescriptorBindingKind::SharedMemory) == 56u &&
              IR::NativeBinding(ShaderType::Pixel, IR::DescriptorBindingKind::Buffers) == 56u &&
              IR::NativeBinding(ShaderType::Pixel, IR::DescriptorBindingKind::MipStats) == 111u,
          "the SharedMemory binding moved the graphics stages' bindings");
  {
    // The static key tells the placements apart (the program cache keeps both).
    ShaderComputeInputInfo info{};
    info.lds_size_dwords = LdsDwords;
    std::vector<uint32_t> workgroup_key;
    std::vector<uint32_t> storage_key;
    BuildStageStaticKey(info, workgroup_key);
    info.lds_storage = true;
    BuildStageStaticKey(info, storage_key);
    Require(name, "static key", workgroup_key != storage_key,
            "the program key does not include the LDS placement");
  }

  const auto test = MakeLdsExchange(4);
  std::vector<u32> native;
  {
    // The NVIDIA plan before and after the switch: workgroup memory, no workgroup-count input.
    ScopedCodegenOptions scoped(WithLds(LdsDeviceBuffer::Auto, 0));
    const auto compiled = CompileCase(test, 32u);
    native = compiled.spirv;
    Require(name, "native",
            !compiled.lds_storage && compiled.lds_dwords == LdsDwords &&
                CountSpirv(native, "BuiltIn NumWorkgroups") == 0u &&
                CountSpirv(native, "_ptr_Workgroup_") != 0u &&
                CountSpirv(native, "OpControlBarrier %uint_2 %uint_2 %uint_264") == 1u,
            "a 48 KiB program on a 64 KiB plan did not keep workgroup memory");
  }
  std::vector<u32> storage;
  {
    ScopedCodegenOptions scoped(WithLds(LdsDeviceBuffer::Auto, 32768));
    const auto compiled = CompileCase(test, 32u);
    storage = compiled.spirv;
    ValidateSpirv(name, storage);
    Require(name, "device buffer",
            compiled.lds_storage && compiled.lds_dwords == LdsDwords &&
                IR::FindBinding(compiled.program.bindings, IR::DescriptorBindingKind::SharedMemory) !=
                    nullptr &&
                CountSpirv(storage, "BuiltIn NumWorkgroups") == 1u &&
                CountSpirv(storage, "BuiltIn WorkgroupId") == 1u &&
                CountSpirv(storage, "Binding 56") == 1u &&
                CountSpirv(storage, "%lds_dwords Coherent") == 1u &&
                CountSpirv(storage, "_ptr_Workgroup_") == 0u &&
                CountSpirv(storage, "OpArrayLength") >= 1u &&
                // AcquireRelease | UniformMemory: the barrier orders the device-buffer LDS.
                CountSpirv(storage, "OpControlBarrier %uint_2 %uint_2 %uint_72") == 1u,
            "a 48 KiB program on a 32 KiB plan did not move its LDS to the device buffer");
  }
  {
    ScopedCodegenOptions scoped(WithLds(LdsDeviceBuffer::Off, 32768));
    const auto compiled = CompileCase(test, 32u);
    Require(name, "clamp",
            !compiled.lds_storage && compiled.lds_dwords == 8192u &&
                CountSpirv(compiled.spirv, "BuiltIn NumWorkgroups") == 0u,
            "KYTY_LDS_DEVICE_BUFFER=0 did not clamp to the limit");
  }
  {
    ScopedCodegenOptions scoped(WithLds(LdsDeviceBuffer::Force, 0));
    const auto compiled = CompileCase(test, 32u);
    Require(name, "force", compiled.lds_storage && compiled.spirv == storage,
            "KYTY_LDS_DEVICE_BUFFER=force differs from the over-limit program");
  }
  std::printf("[codegen] %-32s ok (%zu plan rows)\n", name, std::size(rows));
}

// GPU (harness): the exchange gives the guest's results in workgroup memory, in the device buffer
// (KYTY_LDS_LIMIT_OVERRIDE=32768, forced) and, with the clamp, loses slices 8-11.
void CheckLdsDeviceBufferExchange(VulkanHarness *vulkan) {
  constexpr const char *name = "LdsDeviceBufferExchange";
  struct Mode {
    const char *label;
    LdsDeviceBuffer mode;
    u32 limit_override;
    u32 expected_lds;
  };
  const Mode modes[] = {
      {"native", LdsDeviceBuffer::Auto, 0, LdsDwords},
      {"fallback", LdsDeviceBuffer::Auto, 32768, LdsDwords},
      {"force", LdsDeviceBuffer::Force, 0, LdsDwords},
      {"clamp", LdsDeviceBuffer::Off, 32768, 8192},
  };
  for (const u32 groups : {1u, 4u}) {
    const auto test = MakeLdsExchange(groups);
    for (const auto &mode : modes) {
      ScopedCodegenOptions scoped(WithLds(mode.mode, mode.limit_override));
      const auto actual = RunCollect(vulkan, test, OutputDwords);
      RequireEqualWords(name, mode.label, ExpectedExchange(groups, mode.expected_lds), actual);
    }
  }
  std::printf("[compute] %-32s ok (native, fallback, force; clamp loses 16 KiB)\n", name);
}

// GPU (renderer): the same exchange through RenderExecutor::DispatchDirect, so the SharedMemory
// binding, its per-dispatch sizing and growth (4, then 64 workgroups in one recording) and the
// barriers between the two dispatches are the renderer's. Fallback and force must match native.
void CheckLdsDeviceBufferRenderer(VulkanHarness &harness) {
  constexpr const char *name = "LdsDeviceBufferRenderer";
  constexpr uintptr_t base = 0x0000000208800000ull;
  constexpr uint64_t output_bytes = uint64_t{OutputDwords} * sizeof(u32);
  constexpr uint64_t allocation_size = 0x200000;
  static_assert(2u * output_bytes <= allocation_size);
  static const std::vector<u32> code = MakeLdsExchangeCode();
  ShaderMapUserData(reinterpret_cast<uint64_t>(code.data()),
                    {.type = Prospero::ShaderBinaryType::kCs,
                     .code_size_bytes = static_cast<uint32_t>(code.size() * sizeof(u32))});
  int64_t direct_offset = -1;
  Require(name, "direct allocation",
          Libs::LibKernel::Memory::KernelAllocateDirectMemory(
              0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size,
              allocation_size, 0, &direct_offset) == 0,
          "output allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "direct mapping",
          Libs::LibKernel::Memory::KernelMapDirectMemory(&mapped, allocation_size, 0x3, 0x10,
                                                         direct_offset, allocation_size) == 0 &&
              mapped == reinterpret_cast<void *>(base),
          "output mapping failed");
  struct Mode {
    const char *label;
    LdsDeviceBuffer mode;
    u32 limit_override;
  };
  const Mode modes[] = {
      {"native", LdsDeviceBuffer::Auto, 0},
      {"fallback", LdsDeviceBuffer::Auto, 32768},
      {"force", LdsDeviceBuffer::Force, 0},
  };
  constexpr std::array<u32, 2> dispatch_groups{4u, 64u};
  for (const auto &mode : modes) {
    ScopedCodegenOptions scoped(WithLds(mode.mode, mode.limit_override));
    std::memset(mapped, 0, allocation_size);
    std::array<std::vector<u32>, 2> outputs;
    {
      const auto context_owner = harness.MakeRenderContext();
      auto &context = *context_owner;
      context.InitializeGpu(nullptr);
      LibKernel::Memory::InstallGpuResources(&context);
      context.GetGpu().SendCommandSync([&] {
        CommandProcessor processor(context, 0);
        processor.BufferInit();
        auto &scheduler = context.GetCommandScheduler();
        auto &cache = context.GetBufferCache();
        auto &registers = processor.GetShCtx();
        context.MapMemory(base, allocation_size);
        for (size_t i = 0; i < dispatch_groups.size(); i++) {
          const auto output = base + i * output_bytes;
          ShaderBufferResource descriptor{};
          descriptor.UpdateAddress48(output);
          descriptor.fields[2] = static_cast<u32>(output_bytes);
          descriptor.fields[3] =
              DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
          registers.SetCsShader({.data_addr = reinterpret_cast<uint64_t>(code.data()),
                                 .num_thread_x = Threads, .num_thread_y = 1, .num_thread_z = 1,
                                 .wave_size = 64, .user_sgpr = 4, .tgid_x_en = true,
                                 .lds_size = static_cast<uint16_t>(LdsDwords * 4u / 512u)});
          for (u32 word = 0; word < 4; word++) {
            registers.SetCsUserSgpr(word, descriptor.fields[word], HW::UserSgprType::Unknown);
          }
          processor.DispatchDirect(dispatch_groups[i], 1, 1, 0x41u);
        }
        for (size_t i = 0; i < dispatch_groups.size(); i++) {
          const auto output = base + i * output_bytes;
          cache.ReadMemory(output, output_bytes);
          outputs[i].resize(OutputDwords);
          Require(name, "output readback",
                  LibKernel::Memory::TryReadBacking(output, outputs[i].data(), output_bytes),
                  "the output could not be read back");
        }
        context.UnmapMemory(base, allocation_size);
        scheduler.Finish();
      });
      LibKernel::Memory::InstallGpuResources(nullptr);
      context.ShutdownGpu();
    }
    for (size_t i = 0; i < dispatch_groups.size(); i++) {
      RequireEqualWords(name, mode.label, ExpectedExchange(dispatch_groups[i]), outputs[i]);
    }
  }
  Require(name, "unmap direct backing",
          Libs::LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
          "output mapping release failed");
  Require(name, "release direct backing",
          Libs::LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
          "output allocation release failed");
  std::printf("[gpu]     %-32s ok (4 and 64 workgroups: native, fallback, force)\n", name);
}

} // namespace LdsDeviceBufferTests
