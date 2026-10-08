// KYTY_BDA_WRITES GPU tests (included by ShaderRecompilerComputeTests.cpp after
// ShaderSrtVariantTests.inc).
//
// Psr's BVH builders store and do atomics through V#s they compute (loop-carried per-geometry
// bases, indirect counts). With the switch those writes go through BDA: the recompiler checks the
// RDNA2 range (ISA 8.1.5, by OOB_SELECT), translates the guest address with the BDA page table and
// records each written page in the fault buffer's second bitmap for the renderer's settle.

namespace BdaWriteTests {

using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;

// Stores and atomics through V#s selected with GPU data (buffer load + readfirstlane), each on its
// own 128-byte region: per-component range checks of DWORDX2/X4 stores, byte and short stores
// that keep their neighbours, atomics with return (pre-op value), all-or-nothing 64-bit atomics
// and out-of-range writes that change nothing.
TestCase WritesThroughGpuSelectedDescriptors() {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000130000000ull;
  enum Op : u32 {
    StoreByte = 0x18,
    StoreShort = 0x1a,
    StoreDword = 0x1c,
    StoreDwordX2 = 0x1d,
    StoreDwordX4 = 0x1e,
    AtomicCmpSwap = 0x31,
    AtomicAdd = 0x32,
    AtomicOr = 0x3a,
    AtomicInc = 0x3c,
    AtomicSwapX2 = 0x50,
    AtomicOrX2 = 0x5a,
  };
  struct WriteCase {
    u32 op, mode, stride, records, index, voffset, soffset;
    bool idxen;
    std::array<u32, 4> data;
    bool invalid_format = false;
  };
  const WriteCase cases[] = {
      {StoreDword, 3, 0, 16, 0, 8, 0, false, {0x11111111u}},            // 8 + 4 <= 16: written
      {StoreDword, 3, 0, 16, 0, 14, 0, false, {0x22222222u}},           // 14 + 4 > 16: dropped
      {StoreDwordX4, 3, 0, 12, 0, 4, 0, false,                          // per component: 4, 8 in;
       {0x30000001u, 0x30000002u, 0x30000003u, 0x30000004u}},           // 12, 16 out
      {StoreByte, 3, 0, 32, 0, 5, 0, false, {0xabcdef5au}},             // byte 5 only
      {StoreShort, 3, 0, 32, 0, 6, 0, false, {0x12345678u}},            // bytes 6-7 only
      {StoreDwordX2, 0, 8, 3, 1, 0, 0, true, {0x60000001u, 0x60000002u}}, // structured, index 1
      {StoreDwordX2, 0, 8, 3, 3, 0, 0, true, {0x70000001u, 0x70000002u}}, // index 3 >= 3: dropped
      {StoreDword, 3, 0, 32, 0, 4, 0, false, {0x80000008u}, true},      // invalid format: dropped
      {AtomicAdd, 3, 0, 16, 0, 4, 0, false, {5u}},                      // returns old, adds 5
      {AtomicAdd, 3, 0, 16, 0, 16, 0, false, {5u}},                     // out of range: returns 0
      {AtomicOr, 3, 0, 16, 0, 0, 4, false, {0x00ff0000u}},              // SOFFSET 4: dword at 4
      {AtomicCmpSwap, 3, 0, 32, 0, 8, 0, false, {0xc0ffee00u, 0u}},     // compare filled below
      {AtomicCmpSwap, 3, 0, 32, 0, 12, 0, false, {0xdeadbeefu, 1u}},    // mismatch: unchanged
      {AtomicInc, 3, 0, 32, 0, 16, 0, false, {0xffffffffu}},            // old + 1
      {AtomicSwapX2, 3, 0, 32, 0, 8, 0, false, {0x0a0b0c0du, 0x01020304u}}, // 8 + 8 <= 32
      {AtomicSwapX2, 3, 0, 32, 0, 28, 0, false, {0x11u, 0x22u}},        // 28 + 8 > 32: all out
      {AtomicOrX2, 0, 16, 2, 1, 0, 0, true, {0x0000f000u, 0x80000000u}}, // structured, index 1
  };
  constexpr u32 Count = static_cast<u32>(std::size(cases));
  constexpr u32 DataDword = 2048;
  constexpr u32 RegionDwords = 32;
  TestCase test;
  test.name = "WritesThroughGpuSelectedDescriptors";
  test.initial.resize(4096);
  for (u32 i = 0; i < Count; ++i) {
    const u32 data_offset = (DataDword + i * RegionDwords) * 4u;
    const uint64_t base = GuestBase + data_offset;
    const u32 format = cases[i].invalid_format ? 0u : 0x16u;
    const std::array<u32, 4> descriptor{
        static_cast<u32>(base), (cases[i].stride << 16u) | static_cast<u32>(base >> 32u),
        cases[i].records, (cases[i].mode << 28u) | (format << 12u) | 0xfacu};
    std::copy(descriptor.begin(), descriptor.end(), test.initial.begin() + 130 + i * 30);
    for (u32 word = 0; word < RegionDwords; ++word) {
      test.initial[DataDword + i * RegionDwords + word] = 0x40000000u + i * 0x100u + word;
    }
  }
  for (u32 i = 0; i < Count; ++i) {
    test.initial[64 + i] = Count - 1 - i; // the selector read through a buffer load
  }
  test.expected = test.initial;
  const auto region = [&](u32 i) { return test.expected.begin() + DataDword + i * RegionDwords; };
  for (u32 i = 0; i < Count; ++i) {
    const u32 selected = Count - 1 - i;
    auto input = cases[selected];
    if (input.op == AtomicCmpSwap && input.data[1] == 0u) {
      // Compare with the current value: the swap happens.
      input.data[1] = test.initial[DataDword + selected * RegionDwords + input.voffset / 4];
    }
    AppendVMovU32(&test.code, 30, (64 + i) * 4);
    AppendBufferLoadDword(&test.code, 0, 30);
    test.code.push_back(EncodeVop1(0x02, 20, Vgpr(0))); // v_readfirstlane_b32 s20, v0
    test.code.push_back(EncodeSop2(0x26, 20, 20, 255)); // s_mul_i32 s20, s20, 120
    test.code.push_back(120);
    test.code.push_back(EncodeSmem0(0x0a, 8, 0)); // s_buffer_load_dwordx4 s[8:11], s[0:3]
    test.code.push_back(EncodeSmem1(520, 20));
    AppendSMovLiteral(&test.code, 22, input.soffset);
    AppendVMovU32(&test.code, 2, input.idxen ? input.index : input.voffset);
    AppendVMovU32(&test.code, 3, input.voffset);
    for (u32 component = 0; component < 4; ++component) {
      AppendVMovU32(&test.code, 4 + component, input.data[component]);
    }
    const bool atomic = input.op >= AtomicCmpSwap;
    test.code.push_back(EncodeMubuf0(input.op, 0, input.idxen, true, atomic));
    test.code.push_back(EncodeMubuf1(4, 2, 2, 22)); // v[4..], s[8:11], v[2:3] or v2, s22
    // Returned pre-op values (GLC) go to dwords 2i and 2i+1.
    AppendStoreVgpr(&test.code, 4, 2 * i);
    AppendStoreVgpr(&test.code, 5, 2 * i + 1);

    // Reference: RDNA2 8.1.5 linear addressing and range checks.
    const bool wide = input.op >= AtomicSwapX2;
    const u32 components = input.op == StoreDwordX4 ? 4u : input.op == StoreDwordX2 ? 2u : 1u;
    const u32 payload = input.op == StoreByte ? 1u : input.op == StoreShort ? 2u : wide ? 8u : 4u;
    const u32 record = input.idxen ? input.index * input.stride : 0u;
    const auto in_range = [&](u32 offset, u32 bytes) {
      if (input.invalid_format) return false;
      switch (input.mode) {
        case 0: return input.index < input.records && offset < input.stride;
        case 1: return input.index < input.records;
        case 2: return input.records != 0;
        default:
          return input.soffset <= input.records && offset + bytes <= input.records - input.soffset;
      }
    };
    auto *memory = &*region(selected);
    u32 returned[2] = {input.data[0], input.data[1]};
    if (!atomic) {
      for (u32 component = 0; component < components; ++component) {
        const u32 offset = input.voffset + component * 4u;
        if (!in_range(offset, payload)) continue;
        const u32 byte = input.soffset + record + offset;
        auto &word = memory[byte / 4];
        if (payload == 4u) {
          word = input.data[component];
        } else {
          const u32 shift = (byte % 4u) * 8u;
          const u32 mask = (payload == 1u ? 0xffu : 0xffffu) << shift;
          word = (word & ~mask) | ((input.data[0] << shift) & mask);
        }
      }
    } else if (!in_range(input.voffset, payload)) {
      returned[0] = 0;
      if (wide) returned[1] = 0;
    } else {
      const u32 byte = input.soffset + record + input.voffset;
      auto &word = memory[byte / 4];
      if (wide) {
        auto &high = memory[byte / 4 + 1];
        returned[0] = word;
        returned[1] = high;
        if (input.op == AtomicSwapX2) {
          word = input.data[0];
          high = input.data[1];
        } else {
          word |= input.data[0];
          high |= input.data[1];
        }
      } else {
        returned[0] = word;
        switch (input.op) {
          case AtomicAdd: word += input.data[0]; break;
          case AtomicOr: word |= input.data[0]; break;
          case AtomicInc: word = word >= input.data[0] ? 0u : word + 1u; break;
          default: // CmpSwap: DATA[0] is stored when memory equals DATA[1]
            if (word == input.data[1]) word = input.data[0];
            break;
        }
      }
    }
    test.expected[2 * i] = returned[0];
    test.expected[2 * i + 1] = returned[1];
  }
  AppendEnd(&test.code);
  test.bda_mappings = {{GuestBase, 0}};
  test.opcodes = {O::V_MOV_B32,           O::S_MOV_B32,          O::BUFFER_LOAD_DWORD,
                  O::V_READFIRSTLANE_B32, O::S_MUL_I32,          O::S_BUFFER_LOAD_DWORDX4,
                  O::BUFFER_STORE_BYTE,   O::BUFFER_STORE_SHORT, O::BUFFER_STORE_DWORD,
                  O::BUFFER_STORE_DWORDX2, O::BUFFER_STORE_DWORDX4, O::BUFFER_ATOMIC_CMPSWAP,
                  O::BUFFER_ATOMIC_ADD,   O::BUFFER_ATOMIC_OR,   O::BUFFER_ATOMIC_INC,
                  O::BUFFER_ATOMIC_SWAP_X2, O::BUFFER_ATOMIC_OR_X2, O::S_ENDPGM};
  test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer", "OpAtomicOr"};
  return test;
}

void RunAll(VulkanHarness *vulkan) {
  {
    auto options = GetCodegenOptions();
    options.srt_variant_reads = true;
    options.bda_writes = true;
    ScopedCodegenOptions scope(options);
    RunCase(vulkan, WritesThroughGpuSelectedDescriptors());
  }
  // The renderer's settle reads its mode from the environment (ctest bda_writes sets it).
  if (BdaWritesEnabled()) {
    vulkan->CheckBdaWritesSettle();
  } else {
    std::printf("[gpu]     BdaWritesSettle                  skipped (KYTY_BDA_WRITES unset)\n");
  }
}

} // namespace BdaWriteTests
