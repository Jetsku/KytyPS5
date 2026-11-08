// KYTY_SRT_VARIANT_READS GPU tests (included by ShaderRecompilerComputeTests.cpp after
// ShaderCodegenTests.inc).
//
// Astro Bot's tiled deferred-lighting kernels walk a BVH: they load an instance record through a
// loop-carried pointer (S_LOAD) and read the record's V# with S_BUFFER_LOAD. Without the switch
// such a record read is a flat SRT slot that cannot be evaluated before the dispatch (the pipeline
// cache skips the program's dispatches with one log line), and an S_BUFFER_LOAD through a runtime
// V# fails resource tracking. With it, both are read in the shader through BDA, with the RDNA2
// bounds of a bound scalar buffer.

namespace SrtVariantTests {

using CodegenTests::ScopedCodegenOptions;
using ShaderRecompiler::GetCodegenOptions;

// S_BUFFER_LOAD_DWORDX4 through V#s the shader selects at runtime. RDNA2 ISA 7.2.1: only base,
// stride and num_records are used, addr = (base + OFFSET + SOFFSET) & ~3. The bound matches a bound
// scalar buffer (stride 0: num_records bytes, else stride * num_records), and a dword reads 0
// unless all four of its bytes are below it.
TestCase ScalarBufferLoadsGpuSelectedDescriptors() {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000118000000ull;
  struct DescriptorCase {
    u32 stride, records, offset, soffset;
  };
  const DescriptorCase cases[] = {
      {0, 16, 8, 0},  // dwords 2-3 in, 4-5 past the 16-byte bound
      {0, 24, 8, 0},  // all four in
      {12, 2, 12, 0}, // stride * records = 24 bytes: dwords 3-5 in, 6 out
      {0, 0, 0, 0},   // empty
      {0, 20, 4, 8},  // SOFFSET is added: dwords 3-4 in, 5-6 out
      {0, 32, 10, 0}, // the two LSBs of the offset are ignored: reads from byte 8
      {4, 5, 8, 0},   // 20 bytes: dwords 2-4 in, 5 out
      {0, 15, 8, 0},  // 15 bytes: dword 2 in, dword 3 (bytes 12-15) out
  };
  constexpr u32 Count = static_cast<u32>(std::size(cases));
  TestCase test;
  test.name = "ScalarBufferLoadsGpuSelectedDescriptors";
  test.initial.resize(2048);
  for (u32 i = 0; i < Count; ++i) {
    const u32 data_offset = 4096 + i * 128;
    const uint64_t base = GuestBase + data_offset;
    const std::array<u32, 4> descriptor{static_cast<u32>(base),
                                        (cases[i].stride << 16u) | static_cast<u32>(base >> 32u),
                                        cases[i].records, 0x5204u};
    std::copy(descriptor.begin(), descriptor.end(), test.initial.begin() + 130 + i * 30);
    for (u32 word = 0; word < 32; ++word) {
      test.initial[data_offset / 4 + word] = i * 100 + word + 1;
    }
  }
  for (u32 i = 0; i < Count; ++i) {
    // Select the table entry through a GPU load and readfirstlane, not host constants.
    const u32 selected = Count - 1 - i;
    const auto &input = cases[selected];
    test.initial[64 + i] = selected;
    AppendVMovU32(&test.code, 30, (64 + i) * 4);
    AppendBufferLoadDword(&test.code, 0, 30);
    test.code.push_back(EncodeVop1(0x02, 20, Vgpr(0))); // v_readfirstlane_b32 s20, v0
    test.code.push_back(EncodeSop2(0x26, 20, 20, 255)); // s_mul_i32 s20, s20, 120
    test.code.push_back(120);
    test.code.push_back(EncodeSmem0(0x0a, 8, 0)); // s_buffer_load_dwordx4 s[8:11], s[0:3]
    test.code.push_back(EncodeSmem1(520, 20));
    AppendSMovLiteral(&test.code, 22, input.soffset);
    test.code.push_back(EncodeSmem0(0x0a, 12, 4)); // s_buffer_load_dwordx4 s[12:15], s[8:11]
    test.code.push_back(EncodeSmem1(input.offset, 22));
    const u32 size = input.stride == 0 ? input.records : input.stride * input.records;
    const u32 first = (input.offset + input.soffset) & ~3u;
    for (u32 component = 0; component < 4; ++component) {
      AppendStoreSgpr(&test.code, 12 + component, i * 4 + component);
      const u32 byte = first + component * 4;
      test.expected.push_back(byte + 4 <= size ? selected * 100 + byte / 4 + 1 : 0);
    }
  }
  AppendEnd(&test.code);
  test.bda_mappings = {{GuestBase, 0}};
  test.opcodes = {O::V_MOV_B32, O::S_MOV_B32, O::BUFFER_LOAD_DWORD, O::V_READFIRSTLANE_B32,
                  O::S_MUL_I32, O::S_BUFFER_LOAD_DWORDX4, O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer"};
  return test;
}

// The BVH traversal shape: a loop follows `next` pointers through records (s_load_dwordx2 through
// the loop-carried pointer) and reads each record's V# (s_load_dwordx4 at +16) to
// s_buffer_load_dwordx4 its data. Record 0: 16-byte buffer, record 1: stride 4 x 6 records,
// record 2: empty; the walk ends at a null `next`.
TestCase ScalarLoadsThroughLoopCarriedRecords() {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000120000000ull;
  constexpr u32 Records = 3;
  constexpr u32 RecordBytes = 4096;
  constexpr u32 DataBytes = 6144;
  struct Buffer {
    u32 stride, records;
  };
  const Buffer buffers[Records] = {{0, 16}, {4, 6}, {0, 0}};
  TestCase test;
  test.name = "ScalarLoadsThroughLoopCarriedRecords";
  test.initial.resize(2048);
  for (u32 n = 0; n < Records; ++n) {
    const u32 record = (RecordBytes + n * 64) / 4;
    const uint64_t next = n + 1 < Records ? GuestBase + RecordBytes + (n + 1) * 64 : 0;
    const uint64_t data = GuestBase + DataBytes + n * 256;
    test.initial[record + 0] = static_cast<u32>(next);
    test.initial[record + 1] = static_cast<u32>(next >> 32u);
    test.initial[record + 4] = static_cast<u32>(data);
    test.initial[record + 5] = (buffers[n].stride << 16u) | static_cast<u32>(data >> 32u);
    test.initial[record + 6] = buffers[n].records;
    test.initial[record + 7] = 0x5204u;
    for (u32 word = 0; word < 64; ++word) {
      test.initial[(DataBytes + n * 256) / 4 + word] = 1000 + n * 100 + word;
    }
  }
  auto &code = test.code;
  const uint64_t root = GuestBase + RecordBytes;
  AppendSMovLiteral(&code, 20, static_cast<u32>(root));
  AppendSMovLiteral(&code, 21, static_cast<u32>(root >> 32u));
  AppendSMovLiteral(&code, 30, 0); // output byte offset
  const auto loop = code.size();
  code.push_back(EncodeSopc(0x13, 20, InlineU32(0))); // s_cmp_lg_u64 s[20:21], 0
  const auto exit_branch = code.size();
  code.push_back(EncodeSopp(0x04, 0)); // s_cbranch_scc0 done (patched below)
  code.push_back(EncodeSmem0(0x02, 24, 10)); // s_load_dwordx4 s[24:27], s[20:21], 16
  code.push_back(EncodeSmem1(16, 125));
  code.push_back(EncodeSmem0(0x01, 22, 10)); // s_load_dwordx2 s[22:23], s[20:21], 0
  code.push_back(EncodeSmem1(0, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f)); // s_waitcnt lgkmcnt(0)
  code.push_back(EncodeSmem0(0x0a, 36, 12)); // s_buffer_load_dwordx4 s[36:39], s[24:27], 8
  code.push_back(EncodeSmem1(8, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f));
  code.push_back(EncodeVop1(0x01, 31, 30)); // v_mov_b32 v31, s30
  for (u32 component = 0; component < 4; ++component) {
    code.push_back(EncodeVop1(0x01, 1, 36 + component)); // v_mov_b32 v1, s[36 + component]
    code.push_back(EncodeVop2(0x25, 32, InlineU32(component * 4), 31)); // v32 = v31 + 4k
    AppendBufferStoreDword(&code, 1, 32);
  }
  code.push_back(EncodeSop2(0x00, 30, 30, InlineU32(16))); // s_add_u32 s30, s30, 16
  code.push_back(EncodeSop1(0x04, 20, 22)); // s_mov_b64 s[20:21], s[22:23]
  const auto back_branch = code.size();
  code.push_back(EncodeSopp(0x02, static_cast<u32>(static_cast<int32_t>(loop) -
                                                   static_cast<int32_t>(back_branch + 1)) &
                                      0xffffu)); // s_branch loop
  code[exit_branch] = EncodeSopp(0x04, static_cast<u32>(code.size() - (exit_branch + 1)));
  AppendEnd(&code);
  for (u32 n = 0; n < Records; ++n) {
    const u32 size = buffers[n].stride == 0 ? buffers[n].records : buffers[n].stride * buffers[n].records;
    for (u32 component = 0; component < 4; ++component) {
      const u32 byte = 8 + component * 4;
      test.expected.push_back(byte + 4 <= size ? 1000 + n * 100 + byte / 4 : 0);
    }
  }
  test.bda_mappings = {{GuestBase, 0}};
  test.opcodes = {O::S_MOV_B32, O::S_CMP_LG_U64, O::S_CBRANCH_SCC0, O::S_LOAD_DWORDX4,
                  O::S_LOAD_DWORDX2, O::S_WAITCNT, O::S_BUFFER_LOAD_DWORDX4, O::V_MOV_B32,
                  O::V_ADD_NC_U32, O::BUFFER_STORE_DWORD, O::S_ADD_U32, O::S_MOV_B64,
                  O::S_BRANCH, O::S_ENDPGM};
  test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer"};
  return test;
}

// BUFFER_LOAD_DWORD/UBYTE/SBYTE/USHORT/SSHORT through V#s the shader selects at runtime (the
// BDA path, KYTY_SRT_VARIANT_READS). RDNA2 ISA 8.1.5 range checks by OOB_SELECT (V# dword3
// bits 29:28): 0 structured (index >= num_records or offset >= stride), 1 index only, 2 only
// num_records == 0, 3 raw (offset + payload > num_records - SOFFSET); an invalid FORMAT reads 0.
TestCase RawLoadsThroughGpuSelectedDescriptors() {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000128000000ull;
  enum Load : u32 {
    Ubyte = 0x08, Sbyte = 0x09, Ushort = 0x0a, Sshort = 0x0b, Dword = 0x0c, X2 = 0x0d, X4 = 0x0e,
    X3 = 0x0f
  };
  struct LoadCase {
    u32 op, mode, stride, records, index, voffset, soffset, inst_offset;
    bool idxen, invalid_format = false;
  };
  const LoadCase cases[] = {
      {Dword, 3, 0, 16, 0, 8, 0, 0, false},   // 8 + 4 <= 16: in
      {Dword, 3, 0, 16, 0, 14, 0, 0, false},  // 14 + 4 > 16: 0
      {Ubyte, 3, 0, 13, 0, 12, 0, 0, false},  // 12 + 1 <= 13: in
      {Ubyte, 3, 0, 12, 0, 12, 0, 0, false},  // 12 + 1 > 12: 0
      {Ushort, 3, 0, 14, 0, 12, 0, 0, false}, // 12 + 2 <= 14: in
      {Ushort, 3, 0, 13, 0, 12, 0, 0, false}, // 12 + 2 > 13: 0
      {Sbyte, 3, 0, 32, 0, 5, 0, 0, false},   // sign-extended byte
      {Sshort, 3, 0, 32, 0, 6, 0, 0, false},  // sign-extended short
      {Dword, 3, 0, 16, 0, 4, 4, 4, false},   // SOFFSET 4 + offset 8: 8 + 4 <= 16 - 4: in
      {Dword, 3, 0, 16, 0, 8, 4, 0, false},   // 8 + 4 > 16 - 4: 0
      {Dword, 0, 8, 3, 2, 4, 0, 0, true},     // structured: index 2 < 3, offset 4 < 8: in
      {Dword, 0, 8, 3, 3, 0, 0, 0, true},     // index 3 >= 3: 0
      {Dword, 0, 8, 3, 1, 8, 0, 0, true},     // offset 8 >= stride 8: 0
      {Dword, 1, 8, 3, 1, 12, 0, 0, true},    // index only: offset 12 past the stride still reads
      {Dword, 1, 4, 2, 2, 0, 0, 0, true},     // index 2 >= 2: 0
      {Dword, 2, 0, 1, 0, 20, 0, 0, false},   // num_records != 0: no bounds
      {Dword, 2, 0, 0, 0, 0, 0, 0, false},    // num_records == 0: 0
      {Dword, 3, 0, 32, 0, 0, 0, 0, false, true}, // invalid format: 0
      // Multi-dword loads: each dword has its own range check.
      {X4, 3, 0, 16, 0, 0, 0, 0, false},      // all four in
      {X4, 3, 0, 16, 0, 8, 0, 0, false},      // dwords 0-1 in, 2-3 out
      {X2, 3, 0, 16, 0, 4, 4, 0, false},      // SOFFSET 4: 4 + 8 <= 16 - 4: both in
      {X3, 0, 8, 3, 1, 0, 0, 4, true},        // structured: offset 4 in, 8 and 12 >= stride: 0
      {X2, 3, 0, 32, 0, 0, 0, 0, false, true}, // invalid format: 0
  };
  constexpr u32 Count = static_cast<u32>(std::size(cases));
  const auto byte_at = [](u32 i, u32 b) { return static_cast<uint8_t>(0x80u + i * 16u + b * 5u); };
  TestCase test;
  test.name = "RawLoadsThroughGpuSelectedDescriptors";
  test.initial.resize(4096);
  for (u32 i = 0; i < Count; ++i) {
    const u32 data_offset = 8192 + i * 128;
    const uint64_t base = GuestBase + data_offset;
    const u32 format = cases[i].invalid_format ? 0u : 0x16u;
    const std::array<u32, 4> descriptor{
        static_cast<u32>(base), (cases[i].stride << 16u) | static_cast<u32>(base >> 32u),
        cases[i].records, (cases[i].mode << 28u) | (format << 12u) | 0xfacu};
    std::copy(descriptor.begin(), descriptor.end(), test.initial.begin() + 130 + i * 30);
    for (u32 b = 0; b < 128; ++b) {
      auto &word = test.initial[(data_offset + b) / 4];
      word |= static_cast<u32>(byte_at(i, b)) << ((b % 4u) * 8u);
    }
  }
  u32 slot = 0;
  for (u32 i = 0; i < Count; ++i) {
    const u32 selected = Count - 1 - i;
    const auto &input = cases[selected];
    const u32 dwords = input.op == X2 ? 2u : input.op == X3 ? 3u : input.op == X4 ? 4u : 1u;
    test.initial[64 + i] = selected;
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
    test.code.push_back(EncodeMubuf0(input.op, input.inst_offset, input.idxen, true));
    // v1 (one dword) or v[4..7], s[8:11], v[2:3] or v2, s22
    test.code.push_back(EncodeMubuf1(dwords == 1u ? 1u : 4u, 2, 2, 22));
    for (u32 c = 0; c < dwords; ++c) {
      AppendStoreVgpr(&test.code, dwords == 1u ? 1u : 4u + c, slot++);
    }
    // Reference: RDNA2 8.1.5 linear addressing and range checks, per dword.
    const u32 bytes = dwords > 1u || input.op == Dword ? 4u
                      : (input.op == Ushort || input.op == Sshort) ? 2u : 1u;
    for (u32 c = 0; c < dwords; ++c) {
      const u32 offset = input.voffset + input.inst_offset + c * 4u;
      const u32 record = input.idxen ? input.index * input.stride : 0u;
      bool in = !input.invalid_format;
      switch (input.mode) {
        case 0: in = in && input.index < input.records && offset < input.stride; break;
        case 1: in = in && input.index < input.records; break;
        case 2: in = in && input.records != 0; break;
        default:
          in = in && input.soffset <= input.records && offset + bytes <= input.records - input.soffset;
          break;
      }
      const u32 first = input.soffset + record + offset;
      u32 value = 0;
      if (in) {
        for (u32 b = 0; b < bytes; ++b) {
          value |= static_cast<u32>(byte_at(selected, first + b)) << (b * 8u);
        }
        if (input.op == Sbyte) value = static_cast<u32>(static_cast<int32_t>(static_cast<int8_t>(value)));
        if (input.op == Sshort) value = static_cast<u32>(static_cast<int32_t>(static_cast<int16_t>(value)));
      }
      test.expected.push_back(value);
    }
  }
  AppendEnd(&test.code);
  test.bda_mappings = {{GuestBase, 0}};
  test.opcodes = {O::V_MOV_B32, O::S_MOV_B32, O::BUFFER_LOAD_DWORD, O::BUFFER_LOAD_UBYTE,
                  O::BUFFER_LOAD_SBYTE, O::BUFFER_LOAD_USHORT, O::BUFFER_LOAD_SSHORT,
                  O::BUFFER_LOAD_DWORDX2, O::BUFFER_LOAD_DWORDX3, O::BUFFER_LOAD_DWORDX4,
                  O::V_READFIRSTLANE_B32, O::S_MUL_I32, O::S_BUFFER_LOAD_DWORDX4,
                  O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer"};
  return test;
}

// BUFFER_LOAD_FORMAT_XYZW through V#s the shader selects at runtime (the module's formatted-load
// decoder: the per-format switch, or KYTY_INDIRECT_FORMAT_COMPACT's table-driven one). Every
// format the decoder decodes, with its destination selects, element alignment (8/16-bit
// formats at odd / 2-byte offsets), out-of-range elements (0, but a select of 1 still reads one),
// formats it does not decode (0) and an element across a 16 KiB page-table page.
TestCase FormattedLoadsThroughGpuSelectedDescriptors() {
  using O = ShaderOpcode;
  constexpr uint64_t GuestBase = 0x0000000130000000ull;
  enum Kind : u32 { Uint, Sint, Unorm, Snorm, Float };
  struct FormatDef {
    u32 format;
    Kind kind;
    std::array<u32, 4> widths;
  };
  // The formats the decoder decodes (IndirectFormatDecoded), with their component widths.
  const FormatDef formats[] = {
      {1, Unorm, {8}}, {2, Snorm, {8}}, {5, Uint, {8}}, {6, Sint, {8}},
      {7, Unorm, {16}}, {8, Snorm, {16}}, {11, Uint, {16}}, {12, Sint, {16}}, {13, Float, {16}},
      {14, Unorm, {8, 8}}, {15, Snorm, {8, 8}}, {18, Uint, {8, 8}}, {19, Sint, {8, 8}},
      {20, Uint, {32}}, {21, Sint, {32}}, {22, Float, {32}},
      {23, Unorm, {16, 16}}, {24, Snorm, {16, 16}}, {27, Uint, {16, 16}}, {28, Sint, {16, 16}},
      {29, Float, {16, 16}}, {36, Float, {11, 11, 10}}, {43, Float, {10, 11, 11}},
      {44, Unorm, {2, 10, 10, 10}}, {45, Snorm, {2, 10, 10, 10}},
      {50, Unorm, {10, 10, 10, 2}}, {51, Snorm, {10, 10, 10, 2}},
      {56, Unorm, {8, 8, 8, 8}}, {57, Snorm, {8, 8, 8, 8}}, {60, Uint, {8, 8, 8, 8}},
      {61, Sint, {8, 8, 8, 8}}, {62, Uint, {32, 32}}, {63, Sint, {32, 32}}, {64, Float, {32, 32}},
      {65, Unorm, {16, 16, 16, 16}}, {66, Snorm, {16, 16, 16, 16}}, {69, Uint, {16, 16, 16, 16}},
      {70, Sint, {16, 16, 16, 16}}, {71, Float, {16, 16, 16, 16}},
      {72, Uint, {32, 32, 32}}, {73, Sint, {32, 32, 32}}, {74, Float, {32, 32, 32}},
      {75, Uint, {32, 32, 32, 32}}, {76, Sint, {32, 32, 32, 32}}, {77, Float, {32, 32, 32, 32}},
  };
  struct LoadCase {
    u32 def;          // index into formats, or UINT32_MAX: undecoded `format`
    u32 format;       // for undecoded cases
    u32 dst_sel;      // 12 bits
    u32 offset;       // byte offset of the access in the case's data block
    u32 records;      // OOB_SELECT 3 bound in bytes
    u32 data_offset;  // the data block's byte offset in the buffer (0: default slot)
  };
  constexpr u32 Xyzw = 4u | (5u << 3u) | (6u << 6u) | (7u << 9u);
  std::vector<LoadCase> cases;
  for (u32 n = 0; n < std::size(formats); ++n) {
    u32 bits = 0;
    for (const auto w : formats[n].widths) bits += w;
    const u32 bytes = bits / 8u;
    cases.push_back({n, 0, Xyzw, bytes >= 4u ? 8u : bytes == 2u ? 6u : 5u, 64, 0});
  }
  const auto def_of = [&](u32 format) {
    for (u32 n = 0; n < std::size(formats); ++n) {
      if (formats[n].format == format) return n;
    }
    return UINT32_MAX;
  };
  // Destination selects: one/zero/wrapping past the format's components.
  cases.push_back({def_of(22), 0, 1u | (0u << 3u) | (7u << 6u) | (4u << 9u), 4, 64, 0});
  cases.push_back({def_of(64), 0, 5u | (4u << 3u) | (1u << 6u) | (6u << 9u), 8, 64, 0});
  cases.push_back({def_of(56), 0, 7u | (6u << 3u) | (5u << 6u) | (1u << 9u), 12, 64, 0});
  cases.push_back({def_of(36), 0, 6u | (1u << 3u) | (4u << 6u) | (7u << 9u), 0, 64, 0});
  // Unaligned access: 8-bit at byte 7, 16-bit at byte 3 (element aligned down to 2).
  cases.push_back({def_of(1), 0, Xyzw, 7, 64, 0});
  cases.push_back({def_of(11), 0, Xyzw, 3, 64, 0});
  cases.push_back({def_of(14), 0, Xyzw, 9, 64, 0});
  // Out of range (offset + element size > records): zeros, a select of 1 reads one.
  cases.push_back({def_of(71), 0, 1u | (5u << 3u) | (6u << 6u) | (7u << 9u), 8, 15, 0});
  cases.push_back({def_of(20), 0, 4u | (1u << 3u) | (0u << 6u) | (1u << 9u), 8, 11, 0});
  // Formats the decoder does not decode: 0 (invalid), 3 (8 uscaled), 30 (11_11_10 unorm).
  cases.push_back({UINT32_MAX, 0, 1u | (4u << 3u) | (5u << 6u) | (6u << 9u), 0, 64, 0});
  cases.push_back({UINT32_MAX, 3, Xyzw, 0, 64, 0});
  cases.push_back({UINT32_MAX, 30, 1u | (4u << 3u) | (5u << 6u) | (6u << 9u), 0, 64, 0});
  // A 16-byte element across the 16 KiB page-table page at buffer byte 16384.
  cases.push_back({def_of(77), 0, Xyzw, 8, 64, 16384 - 16});
  const u32 Count = static_cast<u32>(cases.size());
  const auto byte_at = [](u32 i, u32 b) {
    return static_cast<uint8_t>(0x37u + i * 29u + b * 11u + (b * b) / 3u);
  };
  TestCase test;
  test.name = "FormattedLoadsThroughGpuSelectedDescriptors";
  test.initial.resize(16384 / 4 + 64);
  std::vector<uint8_t> bytes(test.initial.size() * 4u, 0);
  const auto data_of = [&](u32 i) { return cases[i].data_offset != 0 ? cases[i].data_offset : 8192u + i * 64u; };
  for (u32 i = 0; i < Count; ++i) {
    const u32 data_offset = data_of(i);
    for (u32 b = 0; b < 64; ++b) bytes[data_offset + b] = byte_at(i, b);
    if (cases[i].def == UINT32_MAX) continue;
    const auto &def = formats[cases[i].def];
    u32 total = 0;
    for (const auto w : def.widths) total += w;
    const u32 size = total / 8u;
    const u32 element = cases[i].offset & ~(std::min(4u, size) - 1u);
    if (def.kind != Float) continue;
    // Finite float components (random bits would hold NaNs).
    u32 bit = 0;
    for (u32 c = 0; c < 4 && def.widths[c] != 0; ++c) {
      const u32 w = def.widths[c];
      uint64_t value = 0;
      if (w == 32) value = std::bit_cast<u32>(-1.25f * static_cast<float>(c + 1) + 0.0625f * i);
      if (w == 16) value = 0x3800u + c * 0x0123u + i * 3u;       // 0.5 .. ~2
      if (w == 11) value = ((13u + c) << 6u) | ((c * 9u + i) & 63u);
      if (w == 10) value = ((14u + c) << 5u) | ((c * 7u + i) & 31u);
      for (u32 k = 0; k < w; ++k) {
        const u32 at = element * 8u + bit + k;
        auto &byte = bytes[data_offset + at / 8u];
        byte = static_cast<uint8_t>((byte & ~(1u << (at % 8u))) | (((value >> k) & 1u) << (at % 8u)));
      }
      bit += w;
    }
  }
  const auto half_to_float = [](u32 h) {
    const u32 sign = (h >> 15u) & 1u, e = (h >> 10u) & 31u, m = h & 1023u;
    float value = e == 0 ? std::ldexp(static_cast<float>(m), -24)
                         : std::ldexp(1.0f + static_cast<float>(m) / 1024.0f, static_cast<int>(e) - 15);
    return sign != 0 ? -value : value;
  };
  const auto ufloat_to_float = [](u32 raw, u32 mantissa_bits) {
    const u32 e = raw >> mantissa_bits, m = raw & ((1u << mantissa_bits) - 1u);
    return e == 0 ? std::ldexp(static_cast<float>(m), 1 - 15 - static_cast<int>(mantissa_bits))
                  : std::ldexp(1.0f + static_cast<float>(m) / static_cast<float>(1u << mantissa_bits),
                               static_cast<int>(e) - 15);
  };
  for (u32 i = 0; i < Count; ++i) {
    const u32 data_offset = data_of(i);
    const auto &input = cases[i];
    const uint64_t base = GuestBase + data_offset;
    const u32 format = input.def != UINT32_MAX ? formats[input.def].format : input.format;
    const std::array<u32, 4> descriptor{static_cast<u32>(base), static_cast<u32>(base >> 32u),
                                        input.records, (3u << 28u) | (format << 12u) | input.dst_sel};
    for (u32 k = 0; k < 4; ++k) {
      for (u32 b = 0; b < 4; ++b) bytes[4096 + i * 16 + k * 4 + b] = static_cast<uint8_t>(descriptor[k] >> (8u * b));
    }
  }
  for (u32 w = 0; w < test.initial.size(); ++w) {
    test.initial[w] = bytes[w * 4] | (bytes[w * 4 + 1] << 8u) | (bytes[w * 4 + 2] << 16u) |
                      (static_cast<u32>(bytes[w * 4 + 3]) << 24u);
  }
  for (u32 i = 0; i < Count; ++i) {
    const u32 selected = Count - 1 - i;
    const auto &input = cases[selected];
    test.initial[512 + i] = selected;
    AppendVMovU32(&test.code, 30, (512 + i) * 4);
    AppendBufferLoadDword(&test.code, 0, 30);
    test.code.push_back(EncodeVop1(0x02, 20, Vgpr(0))); // v_readfirstlane_b32 s20, v0
    test.code.push_back(EncodeSop2(0x26, 20, 20, 255)); // s_mul_i32 s20, s20, 16
    test.code.push_back(16);
    test.code.push_back(EncodeSmem0(0x0a, 8, 0)); // s_buffer_load_dwordx4 s[8:11], s[0:3]
    test.code.push_back(EncodeSmem1(4096, 20));
    AppendSMovLiteral(&test.code, 22, 0);
    AppendVMovU32(&test.code, 2, input.offset);
    test.code.push_back(EncodeMubuf0(0x03, 0, false, true)); // buffer_load_format_xyzw
    test.code.push_back(EncodeMubuf1(4, 2, 2, 22));          // v[4:7], s[8:11], v2, s22
    for (u32 c = 0; c < 4; ++c) AppendStoreVgpr(&test.code, 4 + c, i * 4 + c);
    // Reference decode.
    std::array<u32, 4> channels{};
    std::array<uint8_t, 4> is_float{};
    if (input.def != UINT32_MAX) {
      const auto &def = formats[input.def];
      u32 count = 0, total = 0;
      for (const auto w : def.widths) {
        count += w != 0 ? 1u : 0u;
        total += w;
      }
      const u32 size = total / 8u;
      const u32 element = input.offset & ~(std::min(4u, size) - 1u);
      const bool in = input.offset + size <= input.records;
      const u32 data_offset = data_of(selected);
      std::array<u32, 4> components{};
      u32 bit = 0;
      for (u32 c = 0; c < count; ++c) {
        const u32 w = def.widths[c];
        uint64_t raw = 0;
        for (u32 k = 0; k < w && in; ++k) {
          const u32 at = element * 8u + bit + k;
          raw |= static_cast<uint64_t>((bytes[data_offset + at / 8u] >> (at % 8u)) & 1u) << k;
        }
        bit += w;
        const int64_t signed_raw = w == 32 ? static_cast<int32_t>(raw)
                                           : (raw & (1ull << (w - 1))) ? static_cast<int64_t>(raw) - (1ll << w)
                                                                       : static_cast<int64_t>(raw);
        switch (def.kind) {
          case Uint: components[c] = static_cast<u32>(raw); break;
          case Sint: components[c] = static_cast<u32>(static_cast<int32_t>(signed_raw)); break;
          case Unorm:
            components[c] = std::bit_cast<u32>(static_cast<float>(raw) / static_cast<float>((1u << w) - 1u));
            break;
          case Snorm:
            components[c] = std::bit_cast<u32>(std::max(
                static_cast<float>(signed_raw) / static_cast<float>((1u << (w - 1u)) - 1u), -1.0f));
            break;
          case Float:
            components[c] = w == 32 ? static_cast<u32>(raw)
                            : w == 16 ? std::bit_cast<u32>(half_to_float(static_cast<u32>(raw)))
                                      : std::bit_cast<u32>(ufloat_to_float(static_cast<u32>(raw), w == 11 ? 6u : 5u));
            break;
        }
      }
      const u32 one = def.kind == Uint || def.kind == Sint ? 1u : 0x3f800000u;
      for (u32 channel = 0; channel < 4; ++channel) {
        const u32 sel = (input.dst_sel >> (3u * channel)) & 7u;
        channels[channel] = sel >= 4 ? components[(sel - 4u) % count] : sel == 1 ? one : 0u;
        is_float[channel] = (def.kind == Unorm || def.kind == Snorm) && sel >= 4 ? 1u : 0u;
      }
    }
    for (u32 channel = 0; channel < 4; ++channel) {
      test.expected.push_back(channels[channel]);
      test.float_words.push_back(is_float[channel]);
    }
  }
  AppendEnd(&test.code);
  test.bda_mappings = {{GuestBase, 0}};
  test.opcodes = {O::V_MOV_B32, O::S_MOV_B32, O::BUFFER_LOAD_DWORD, O::BUFFER_LOAD_FORMAT_XYZW,
                  O::V_READFIRSTLANE_B32, O::S_MUL_I32, O::S_BUFFER_LOAD_DWORDX4,
                  O::BUFFER_STORE_DWORD, O::S_ENDPGM};
  test.required_spirv = {"OpConvertUToPtr", "PhysicalStorageBuffer"};
  return test;
}

// Without KYTY_SRT_VARIANT_READS the loop-carried record reads above stay flat SRT slots, and no
// evaluation before the dispatch can produce their addresses. FindVariantFlatRead names one of
// them, which is how the pipeline cache tells such a plan (skipped with one log line) from any
// other that fails to materialize (still an exit). A pointer loaded once and followed is not
// loop-carried.
void VariantPlanIsFound() {
  constexpr const char *name = "VariantPlanIsFound";
  auto codegen = GetCodegenOptions();
  codegen.srt_variant_reads = false;
  ScopedCodegenOptions scope(codegen);
  ShaderComputeInputInfo compute_info{};
  const auto translate = [&](const TestCase &test, std::array<u32, 64> &user_data) {
    user_data = MakeNativeUserData(nullptr);
    user_data[2] = static_cast<u32>(test.initial.size() * sizeof(u32));
    ShaderRecompiler::CompileOptions options;
    options.stage = ShaderType::Compute;
    options.input_info.compute = &compute_info;
    options.user_data = user_data;
    auto translated = ShaderRecompiler::TranslateProgram(test.code, options);
    Require(name, "translate", !translated.skip_dispatch && !translated.program.blocks.empty(),
            std::string(test.name) + " did not translate");
    return ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
  };

  const auto records = ScalarLoadsThroughLoopCarriedRecords();
  std::array<u32, 64> user_data{};
  const auto plan = translate(records, user_data);
  std::vector<uint32_t> loop_reads;
  {
    ShaderRecompiler::Decoder::Program decoded;
    ShaderRecompiler::Decoder::DecodeProgram(records.code, decoded);
    for (const auto &inst : decoded.instructions) {
      if (inst.opcode == ShaderOpcode::S_LOAD_DWORDX2 ||
          inst.opcode == ShaderOpcode::S_LOAD_DWORDX4 ||
          inst.opcode == ShaderOpcode::S_BUFFER_LOAD_DWORDX4) {
        loop_reads.push_back(inst.pc);
      }
    }
  }
  uint32_t pc = UINT32_MAX;
  Require(name, "loop-carried read",
          ShaderRecompiler::IR::FindVariantFlatRead(plan, pc) &&
              std::ranges::find(loop_reads, pc) != loop_reads.end(),
          "no flat read through the loop-carried pointer was found (pc " + Hex(pc) + ")");
  ShaderRecompiler::IR::ResourceSnapshot resources;
  ShaderRecompiler::IR::ResourceSpecialization specialization;
  const ShaderRecompiler::IR::SrtRuntime runtime{
      .user_data = user_data,
      .shader_base = reinterpret_cast<uint64_t>(records.code.data()),
      .read_memory = ReadTestMemory,
      .userdata = const_cast<std::vector<u32> *>(&records.initial),
  };
  Require(name, "no materialization",
          !ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization),
          "a plan with a loop-carried flat read materialized");

  // A pointer the shader loads once and follows (no loop) is evaluated before the dispatch as
  // usual: not reported.
  TestCase chase;
  chase.name = "PointerChase";
  chase.initial.resize(64);
  chase.code.push_back(EncodeSmem0(0x01, 8, 0)); // s_load_dwordx2 s[8:9], s[0:1], 0
  chase.code.push_back(EncodeSmem1(0, 125));
  chase.code.push_back(EncodeSopp(0x0c, 0xc07f)); // s_waitcnt lgkmcnt(0)
  chase.code.push_back(EncodeSmem0(0x01, 10, 4)); // s_load_dwordx2 s[10:11], s[8:9], 0x30
  chase.code.push_back(EncodeSmem1(0x30, 125));
  chase.code.push_back(EncodeSopp(0x0c, 0xc07f));
  AppendStoreSgpr(&chase.code, 10, 0);
  AppendEnd(&chase.code);
  std::array<u32, 64> chase_user_data{};
  const auto chase_plan = translate(chase, chase_user_data);
  Require(name, "pointer chase", !ShaderRecompiler::IR::FindVariantFlatRead(chase_plan, pc),
          "a pointer loaded outside any loop was reported as loop-carried");
  std::printf("[compute] %-32s ok\n", name);
}

// S_SWAPPC_B64 (a call through a function pointer) is skipped, not a CFG-build exit.
void CallIsSkipped() {
  std::vector<u32> code;
  AppendSMovLiteral(&code, 4, 0x1000u);
  AppendSMovLiteral(&code, 5, 0);
  code.push_back(0xbe8e2104u); // s_swappc_b64 s[14:15], s[4:5]
  AppendEnd(&code);
  ShaderComputeInputInfo compute_info{};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  const auto result = ShaderRecompiler::TranslateProgram(code, options);
  Require("CallIsSkipped", "translate", result.skip_dispatch,
          "S_SWAPPC_B64 program was not skipped under KYTY_SRT_VARIANT_READS");
  std::printf("[compute] CallIsSkipped                     ok\n");
}

// KYTY_HUGE_HELPER_OUTLINE: a helper with 16 call sites in a module of 100,000 words becomes
// DontInline; fewer calls or a smaller module leave its function control alone.
void HugeHelperOutlineControl() {
  const auto module = [](size_t words, u32 calls) {
    std::vector<u32> m = {0x07230203u, 0x00010300u, 0u, 100u, 0u};
    m.insert(m.end(), {(5u << 16) | 54u, 1u, 10u, 0u, 2u}); // %10 = OpFunction %1 None %2
    m.insert(m.end(), {(5u << 16) | 54u, 1u, 11u, 0u, 2u}); // %11, never called
    for (u32 c = 0; c < calls; c++) {
      m.insert(m.end(), {(5u << 16) | 57u, 1u, 20u + c, 10u, 3u}); // OpFunctionCall %1 %10 %3
    }
    m.resize(std::max(words, m.size()), (1u << 16) | 0u); // OpNop padding
    return m;
  };
  const std::array<u32, 2> helpers{10u, 11u};
  auto huge = module(100000, 16);
  ShaderRecompiler::Spirv::OutlineHugeModuleHelpers(huge, helpers);
  Require("HugeHelperOutline", "outlined", huge[8] == 2u && huge[13] == 0u,
          "a helper with 16 calls in a huge module was not made DontInline (or an uncalled one was)");
  auto few = module(100000, 15);
  ShaderRecompiler::Spirv::OutlineHugeModuleHelpers(few, helpers);
  Require("HugeHelperOutline", "few calls", few[8] == 0u, "a helper with 15 calls was changed");
  auto little = module(99999, 16);
  ShaderRecompiler::Spirv::OutlineHugeModuleHelpers(little, helpers);
  Require("HugeHelperOutline", "small module", little[8] == 0u, "a helper of a small module was changed");
  std::printf("[compute] HugeHelperOutlineControl          ok\n");
}

// Flat SRT reads through a pointer that is null, or points nowhere, when the plan is evaluated
// before the dispatch: Astro Bot's tiled lighting (78af8e26) loads a TLAS pointer from its SRT and
// header fields through it (s_load_dwordx2 s[10:11], s[8:9], 0x30), on a path only lanes that
// trace a shadow ray take, and the pointer is null while no TLAS exists. The in-place reader (no
// read_memory, as in game) must not touch such an address: it reads 0 and the plan materializes.
// Without the check the second read dereferences 0x30 and the process dies. A mapped pointer
// still reads its data, and is_guest_mapped extends the check to every address outside the guest
// mappings (the game passes it with KYTY_SRT_VARIANT_READS).
namespace NullBasedReads {

constexpr uint64_t Unmapped = 0x0000010000000000ull; // 1 TiB: outside every test mapping
uint32_t refused = 0;

bool GuestMapped(uint64_t address, uint64_t size) {
  if (address + size > Unmapped && address < Unmapped + 0x1000u) {
    refused++;
    return false;
  }
  return true;
}

// The game's clean-backing probe shape: fails (touching nothing) where no backing exists.
bool HostProbe(void *, uint64_t address, std::span<uint32_t> values) {
  const auto size = values.size_bytes();
  if (ShaderRecompiler::IR::NeverMappedAddress(address, size) ||
      (address + size > Unmapped && address < Unmapped + 0x1000u)) {
    return false;
  }
  std::memcpy(values.data(), reinterpret_cast<const void *>(address), size);
  return true;
}

} // namespace NullBasedReads

void NullBasedFlatReadReadsZero() {
  constexpr const char *name = "NullBasedFlatReadReadsZero";
  namespace IRN = ShaderRecompiler::IR;
  std::vector<u32> code;
  code.push_back(EncodeSmem0(0x01, 8, 0)); // s_load_dwordx2 s[8:9], s[0:1], 0
  code.push_back(EncodeSmem1(0, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f)); // s_waitcnt lgkmcnt(0)
  const u32 header_pc = static_cast<u32>(code.size() * sizeof(u32));
  code.push_back(EncodeSmem0(0x01, 10, 4)); // s_load_dwordx2 s[10:11], s[8:9], 0x30
  code.push_back(EncodeSmem1(0x30, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f));
  AppendStoreSgpr(&code, 10, 0);
  AppendStoreSgpr(&code, 11, 1);
  AppendEnd(&code);

  static std::array<uint64_t, 16> table{};  // table[0]: the pointer the shader follows
  static std::array<u32, 32> header{};       // a mapped target: dwords 12-13 at +0x30
  header[12] = 0x1234u;
  header[13] = 0x5678u;
  // Materializes the plan with the pointer set to `pointer` and returns the two values the
  // second load's flat slots hold.
  const auto run = [&](uint64_t pointer, IRN::SrtMappedRange mapped, IRN::SrtMemoryReader probe,
                       const char *stage) {
    table[0] = pointer;
    auto user_data = MakeNativeUserData(nullptr);
    const auto table_address = reinterpret_cast<uint64_t>(table.data());
    user_data[0] = static_cast<u32>(table_address);
    user_data[1] = static_cast<u32>(table_address >> 32u);
    ShaderComputeInputInfo compute_info{};
    ShaderRecompiler::CompileOptions options;
    options.stage = ShaderType::Compute;
    options.input_info.compute = &compute_info;
    options.user_data = user_data;
    auto translated = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, stage, !translated.skip_dispatch && !translated.program.blocks.empty(),
            "the pointer-chasing program did not translate");
    const auto plan = IRN::ExtractResourcePlan(translated.program);
    IRN::ResourceSnapshot resources;
    IRN::ResourceSpecialization specialization;
    const IRN::SrtRuntime runtime{
        .user_data = user_data,
        .shader_base = reinterpret_cast<uint64_t>(code.data()),
        .try_read_clean_backing = probe,
        .is_guest_mapped = mapped,
    };
    Require(name, stage, IRN::MaterializeResources(plan, runtime, resources, specialization),
            "the plan did not materialize");
    std::vector<u32> values;
    for (const auto &read : plan.srt_reads) {
      const auto *inst = read.value.Resolve().TryInstruction();
      if (inst != nullptr && inst->Flags<IRN::MemoryFlags>().pc == header_pc &&
          read.flat_offset < resources.flattened_srt.size()) {
        values.push_back(resources.flattened_srt[read.flat_offset]);
      }
    }
    std::ranges::sort(values);
    Require(name, stage, values.size() == 2u,
            "the header load did not become two flat slots (" +
                std::to_string(values.size()) + ")");
    return std::pair{values[0], values[1]};
  };
  // Without and with the clean-backing probe (KYTY_SRT_READ_RUNS: the flat-run path of R2's crash).
  for (const auto probe : {IRN::SrtMemoryReader{}, IRN::SrtMemoryReader{NullBasedReads::HostProbe}})
  for (const bool variant_reads : {true, false}) {
    auto codegen = GetCodegenOptions();
    codegen.srt_variant_reads = variant_reads;
    ScopedCodegenOptions scope(codegen);
    const auto null_read = run(0, nullptr, probe, "null pointer");
    Require(name, "null pointer", null_read.first == 0u && null_read.second == 0u,
            "a read of null + 0x30 did not read 0: " + Hex(null_read.first) + " " +
                Hex(null_read.second));
    const auto header_address = reinterpret_cast<uint64_t>(header.data());
    const auto mapped_read = run(header_address, nullptr, probe, "mapped pointer");
    const auto checked_read = run(header_address, NullBasedReads::GuestMapped, probe,
                                  "mapped pointer, checked");
    Require(name, "mapped pointer",
            mapped_read == std::pair{0x1234u, 0x5678u} && checked_read == mapped_read,
            "a mapped header read " + Hex(mapped_read.first) + " " + Hex(checked_read.first));
    NullBasedReads::refused = 0;
    const auto unmapped_read =
        run(NullBasedReads::Unmapped, NullBasedReads::GuestMapped, probe, "unmapped pointer");
    Require(name, "unmapped pointer",
            unmapped_read.first == 0u && unmapped_read.second == 0u &&
                NullBasedReads::refused >= 2u,
            "a read outside the guest mappings was not refused and read as 0 (refused " +
                std::to_string(NullBasedReads::refused) + ")");
  }
  std::printf("[compute] %-32s ok\n", name);
}

// The callees a skipped S_SWAPPC_B64 program jumps to, from the dispatch's user data: Psr's
// shader-mesh builders copy a callback record from user SGPRs into s[14:17] before each
// s_swappc_b64 s[14:15], s[14:15] (the swappc writes the return address into its destination pair,
// which does not make a target unknown). A call directly through user SGPRs names them. A target
// the program computes some other way is only counted as unresolved in the log line.
void CallTargetsFromUserData() {
  constexpr const char *name = "CallTargetsFromUserData";
  std::vector<u32> code;
  for (u32 i = 0; i < 4; ++i) {
    code.push_back(EncodeSMovB32(14 + i, 4 + i)); // s_mov_b32 s[14 + i], s[4 + i]
  }
  code.push_back(0xbe8e210eu); // s_swappc_b64 s[14:15], s[14:15]
  code.push_back(EncodeSMovB32(14, 8));
  code.push_back(EncodeSMovB32(15, 9));
  code.push_back(0xbe8e210eu); // s_swappc_b64 s[14:15], s[14:15]
  code.push_back(0xbe94210au); // s_swappc_b64 s[20:21], s[10:11]
  AppendEnd(&code);
  std::array<u32, 16> user_data{};
  user_data[4] = 0x00001000u;
  user_data[5] = 0x00000002u;
  user_data[8] = 0x00003000u;
  user_data[9] = 0x00000004u;
  user_data[10] = 0x00005000u;
  user_data[11] = 0x00010006u; // bits above 48 are not part of the address
  ShaderComputeInputInfo compute_info{};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  const auto result = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "skipped", result.skip_dispatch, "the calling program was not skipped");
  const auto &targets = result.call_targets;
  const auto has = [&](uint64_t address, u32 user_sgpr, u32 return_sgpr) {
    return std::ranges::any_of(targets, [&](const ShaderRecompiler::CallTarget &target) {
      return target.address == address && target.user_sgpr == user_sgpr &&
             target.return_sgpr == return_sgpr;
    });
  };
  Require(name, "targets",
          targets.size() == 3u && has(0x0000000200001000ull, 4, 14) &&
              has(0x0000000400003000ull, 8, 14) && has(0x0000000600005000ull, 10, 20),
          "the callees were not read from user data (" + std::to_string(targets.size()) +
              " found)");

  // A target the program computes from a user-data copy has no known callee.
  std::vector<u32> computed;
  computed.push_back(EncodeSMovB32(14, 4));
  computed.push_back(EncodeSMovB32(15, 5));
  computed.push_back(EncodeSop2(0x00, 14, 14, InlineU32(4))); // s_add_u32 s14, s14, 4
  computed.push_back(0xbe8e210eu);
  AppendEnd(&computed);
  const auto partial = ShaderRecompiler::TranslateProgram(computed, options);
  Require(name, "computed target", partial.skip_dispatch && partial.call_targets.empty(),
          "a computed target was reported as a user-data callee");

  // Across control flow: a copy in the entry block reaches a call behind a branch target; a copy
  // on one path only does not.
  std::vector<u32> branchy;
  branchy.push_back(EncodeSMovB32(14, 4));
  branchy.push_back(EncodeSMovB32(15, 5));
  branchy.push_back(EncodeSopp(0x04, 1)); // s_cbranch_scc0 over the next instruction
  branchy.push_back(EncodeSMovB32(20, 6));
  branchy.push_back(0xbe8e210eu); // s_swappc_b64 s[14:15], s[14:15] (a branch target)
  AppendEnd(&branchy);
  const auto dominated = ShaderRecompiler::TranslateProgram(branchy, options);
  Require(name, "entry copy", dominated.skip_dispatch && dominated.call_targets.size() == 1u &&
                                  dominated.call_targets[0].address == 0x0000000200001000ull,
          "an entry-block copy did not reach a call behind a branch target");
  std::vector<u32> one_path;
  one_path.push_back(EncodeSopp(0x04, 2)); // s_cbranch_scc0 over the two copies
  one_path.push_back(EncodeSMovB32(14, 4));
  one_path.push_back(EncodeSMovB32(15, 5));
  one_path.push_back(0xbe8e210eu);
  AppendEnd(&one_path);
  const auto conditional = ShaderRecompiler::TranslateProgram(one_path, options);
  Require(name, "one-path copy", conditional.skip_dispatch && conditional.call_targets.empty(),
          "a copy on one path only was reported as the callee");
  std::printf("[compute] %-32s ok\n", name);
}

// S_SWAPPC_B64 calls through a function table (ShaderRecompiler CallTable), as Miles Morales'
// ray-tracing shaders pick a hit callback per lane: the table V# is loaded through the SRT before a
// branch, the call block loads a 16-byte record at index << 4 and copies its first pair into the
// link registers. The table descriptor register is also used as a temporary before its load and
// after the call; the trace still finds the load that reaches the call.
void CallTablesTraced() {
  constexpr const char *name = "CallTablesTraced";
  std::vector<u32> code;
  code.push_back(EncodeSMovB32(72, 4));                        // s_mov_b32 s72, s4 (a temporary)
  code.push_back(EncodeSmem0(0x02, 8, 2));                     // s_load_dwordx4 s[8:11], s[4:5], 544
  code.push_back(EncodeSmem1(544, 125));
  code.push_back(EncodeSmem0(0x0a, 72, 4));                    // s_buffer_load_dwordx4 s[72:75], s[8:11], 64
  code.push_back(EncodeSmem1(64, 125));
  code.push_back(EncodeSopp(0x04, 0));                         // s_cbranch_scc0 to the next block
  code.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4)));      // s_lshl_b32 vcc_lo, s6, 4
  code.push_back(EncodeSmem0(0x0a, 8, 36));                    // s_buffer_load_dwordx4 s[8:11], s[72:75], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSMovB32(14, 8));
  code.push_back(EncodeSMovB32(15, 9));
  code.push_back(0xbe8e210eu);                                 // s_swappc_b64 s[14:15], s[14:15]
  code.push_back(EncodeSMovB32(72, 5));                        // a later temporary
  AppendEnd(&code);
  std::array<u32, 16> user_data{};
  ShaderComputeInputInfo compute_info{};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  const auto result = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "skipped", result.skip_dispatch, "the calling program was not skipped");
  const auto &tables = result.call_tables;
  const auto step = [](const ShaderRecompiler::CallTableStep &s, bool buffer, int32_t offset) {
    return s.buffer == buffer && s.base_index == 0u && s.offset == offset && s.dwords == 4u;
  };
  Require(name, "table",
          tables.size() == 1u && tables[0].user_sgpr == 4u && tables[0].return_sgpr == 14u &&
              tables[0].record_shift == 4u && tables[0].target_word == 0u &&
              tables[0].table_index == 0u && tables[0].steps.size() == 2u &&
              step(tables[0].steps[0], false, 544) && step(tables[0].steps[1], true, 64),
          "the call table was not traced (" + std::to_string(tables.size()) + " found)");

  // A second, different load of the descriptor that can reach the call is not a table.
  std::vector<u32> ambiguous = code;
  ambiguous.insert(ambiguous.begin() + 5, {EncodeSopp(0x04, 2),       // s_cbranch_scc0 past the reload
                                           EncodeSmem0(0x0a, 72, 4),  // s_buffer_load_dwordx4 s[72:75], s[8:11], 80
                                           EncodeSmem1(80, 125)});
  const auto other = ShaderRecompiler::TranslateProgram(ambiguous, options);
  Require(name, "two loads", other.skip_dispatch && other.call_tables.empty(),
          "a descriptor with two reaching loads was traced");
  std::printf("[compute] %-32s ok\n", name);
}

// KYTY_CALL_TRACE_WAVE32: PPSA03671's tiled material kernels (wave32) write a V_CMP mask to s11
// before their callback loop, load the table V# from the SRT pointer s[12:13] inside it, and reload
// s[8:15] from s[12:13] only after it. A wave32 V_CMP writes one SGPR, and s[12:13] holds its entry
// value at the call: the table is the user-data pointer's +144.
void CallTablesTracedWave32() {
  constexpr const char *name = "CallTablesTracedWave32";
  std::vector<u32> code;
  code.push_back(EncodeVop3Word0(0x004u, 11));           // v_cmp_gt_f32 s11, v0, v1
  code.push_back(EncodeVop3Word1(Vgpr(0), Vgpr(1)));
  code.push_back(EncodeSopp(0x04, 0));                   // s_cbranch_scc0 to the next block
  code.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4))); // s_lshl_b32 vcc_lo, s6, 4
  code.push_back(EncodeSmem0(0x02, 0, 6));               // s_load_dwordx4 s[0:3], s[12:13], 144
  code.push_back(EncodeSmem1(144, 125));
  code.push_back(EncodeSmem0(0x0a, 0, 0));               // s_buffer_load_dwordx4 s[0:3], s[0:3], vcc_lo
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSMovB32(14, 0));
  code.push_back(EncodeSMovB32(15, 1));
  code.push_back(0xbe8e210eu);                           // s_swappc_b64 s[14:15], s[14:15]
  code.push_back(EncodeSmem0(0x03, 8, 6));               // s_load_dwordx8 s[8:15], s[12:13], 64
  code.push_back(EncodeSmem1(64, 125));
  AppendEnd(&code);
  std::array<u32, 16> user_data{};
  ShaderComputeInputInfo compute_info{};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.input_info.compute = &compute_info;
  options.user_data = user_data;
  options.wave_size = 32;
  auto codegen = GetCodegenOptions();
  codegen.call_trace_wave32 = false;
  {
    ScopedCodegenOptions scope(codegen);
    const auto before = ShaderRecompiler::TranslateProgram(code, options);
    Require(name, "switch off", before.skip_dispatch && before.call_tables.empty(),
            "the table was traced without KYTY_CALL_TRACE_WAVE32");
  }
  codegen.call_trace_wave32 = true;
  ScopedCodegenOptions scope(codegen);
  const auto result = ShaderRecompiler::TranslateProgram(code, options);
  const auto &tables = result.call_tables;
  Require(name, "table",
          result.skip_dispatch && tables.size() == 1u && tables[0].user_sgpr == 12u &&
              tables[0].return_sgpr == 14u && tables[0].record_shift == 4u &&
              tables[0].target_word == 0u && tables[0].table_index == 0u &&
              tables[0].steps.size() == 1u && !tables[0].steps[0].buffer &&
              tables[0].steps[0].offset == 144 && tables[0].steps[0].dwords == 4u,
          "the wave32 call table was not traced (" + std::to_string(tables.size()) + " found)");
  // Wave64: the V_CMP writes s[11:12], so s12 has no single reaching definition.
  options.wave_size = 64;
  const auto wave64 = ShaderRecompiler::TranslateProgram(code, options);
  Require(name, "wave64", wave64.call_tables.empty(),
          "a wave64 V_CMP pair over s12 was ignored");
  // Upstream 7524e8f79's case: a lane read writes one SGPR in wave64 too, so V_READLANE_B32 s11
  // does not define s12 (WritesSgpr takes the width from the _B32 name).
  {
    auto lane_read = code;
    lane_read[0] = EncodeVop3Word0(0x360u, 11); // v_readlane_b32 s11, v0, s1
    lane_read[1] = EncodeVop3Word1(Vgpr(0), 1);
    const auto traced = ShaderRecompiler::TranslateProgram(lane_read, options);
    Require(name, "wave64 lane read",
            traced.call_tables.size() == 1u && traced.call_tables[0].user_sgpr == 12u,
            "a V_READLANE_B32 of s11 was taken as a write of s12");
  }
  // The table V# in user data itself (0xfc0537b7278db60c: s[8:11]): no load step.
  options.wave_size = 32;
  {
    std::vector<u32> direct;
    direct.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4))); // s_lshl_b32 vcc_lo, s6, 4
    direct.push_back(EncodeSmem0(0x0a, 0, 4));                // s_buffer_load_dwordx4 s[0:3], s[8:11], vcc_lo
    direct.push_back(EncodeSmem1(0, 106));
    direct.push_back(EncodeSMovB32(14, 0));
    direct.push_back(EncodeSMovB32(15, 1));
    direct.push_back(0xbe8e210eu); // s_swappc_b64 s[14:15], s[14:15]
    AppendEnd(&direct);
    const auto traced = ShaderRecompiler::TranslateProgram(direct, options);
    Require(name, "user-data table",
            traced.call_tables.size() == 1u && traced.call_tables[0].user_sgpr == 8u &&
                traced.call_tables[0].steps.empty() && traced.call_tables[0].table_index == 0u,
            "a table V# held in user data was not traced");
  }
  // A later write that can loop back to the call keeps the range unknown.
  std::vector<u32> looping = code;
  looping.pop_back(); // s_endpgm
  // The lshl is word 3; the branch offset counts from the word after the branch.
  const auto back = 3 - static_cast<int32_t>(looping.size() + 1u);
  looping.push_back(EncodeSopp(0x04, static_cast<u32>(back) & 0xffffu)); // s_cbranch_scc0 to the lshl
  AppendEnd(&looping);
  const auto loop = ShaderRecompiler::TranslateProgram(looping, options);
  Require(name, "loop back", loop.call_tables.empty(),
          "a reload that can loop back to the call was taken as the entry value");
  std::printf("[compute] %-32s ok\n", name);
}

// KYTY_SHADER_CALLS: guest memory for the expansion side of the call tests: (guest address, words).
struct FakeGuestMemory {
  std::vector<std::pair<uint64_t, std::vector<u32> *>> regions;

  bool Read(uint64_t address, std::span<u32> words) const {
    for (const auto &[base, data] : regions) {
      const uint64_t end = base + data->size() * 4u;
      if (address >= base && address + words.size_bytes() <= end && (address & 3u) == 0) {
        std::copy_n(data->begin() + static_cast<std::ptrdiff_t>((address - base) / 4u), words.size(),
                    words.begin());
        return true;
      }
    }
    return false;
  }
  ShaderRecompiler::Decoder::ShaderCodeReader Reader() const {
    return [this](uint64_t address, std::span<u32> words) { return Read(address, words); };
  }
};

// The call shape of Miles Morales' ray-tracing kernels, expanded (ShaderCallExpander) and run: the
// table V# is loaded through the user-data V#, each call loads a 16-byte record {function, data} at
// index << 4, copies it into s[14:16] and calls s[14:15]. Records 0-2 name callees A-C; record 3 is
// null and record 4 names D only in the GPU's copy (the expansion did not see it): both skip the
// call. Each call stores SCC after the call (s_cselect 1 : 2), v1 (999 before the call) and v2 (0
// before). Calls alternate the SCC they enter with; B keeps it, A and C set it.
TestCase CallTableExpanded() {
  using O = ShaderOpcode;
  // The harness binds every buffer the program reads to its one test buffer, from its start, and
  // its user-data V# s[0:3] has base 0: the table is at the start of the buffer (a table V# with
  // base 0), the stores follow it.
  constexpr uint64_t GuestBase = 0;
  constexpr u32 Out = 20;
  constexpr uint64_t CalleeBase = 0x0000000140000000ull;
  constexpr u32 TableByte = 0;
  static TestCallees callees;
  TestCase test;
  test.name = "CallTableExpanded";
  test.initial.resize(2048);
  const uint64_t table = GuestBase + TableByte;
  const std::array<u32, 4> table_vsharp{static_cast<u32>(table), static_cast<u32>(table >> 32u),
                                        5u * 16u, 0x5204u};
  std::copy(table_vsharp.begin(), table_vsharp.end(), test.initial.begin() + 130);
  const auto record = [&](std::vector<u32> &words, u32 index, uint64_t function, u32 data) {
    words[TableByte / 4 + index * 4 + 0] = static_cast<u32>(function);
    words[TableByte / 4 + index * 4 + 1] = static_cast<u32>(function >> 32u);
    words[TableByte / 4 + index * 4 + 2] = data;
  };
  record(test.initial, 0, TestCallees::Address(CalleeBase, 0), 10);
  record(test.initial, 1, TestCallees::Address(CalleeBase, 1), 20);
  record(test.initial, 2, TestCallees::Address(CalleeBase, 2), 5);
  const std::array<u32, 6> indices{0, 1, 2, 3, 4, 1};
  for (u32 i = 0; i < indices.size(); ++i) {
    test.initial[64 + i] = indices[i];
  }
  // What the expansion reads: the same buffer without record 4.
  static std::vector<u32> seen;
  seen = test.initial;
  record(test.initial, 4, TestCallees::Address(CalleeBase, 3), 0);

  auto &code = test.code;
  code.push_back(EncodeSmem0(0x0a, 72, 0)); // s_buffer_load_dwordx4 s[72:75], s[0:3], 520
  code.push_back(EncodeSmem1(520, 125));
  for (u32 i = 0; i < indices.size(); ++i) {
    AppendVMovU32(&code, 30, (64 + i) * 4);
    AppendBufferLoadDword(&code, 0, 30);
    code.push_back(EncodeVop1(0x02, 20, Vgpr(0)));                       // v_readfirstlane_b32 s20, v0
    AppendVMovLiteral(&code, 1, 999);
    code.push_back(EncodeVop1(0x01, 2, InlineU32(0)));                  // v_mov_b32 v2, 0
    code.push_back(EncodeSop2(0x1e, 106, 20, InlineU32(4)));            // s_lshl_b32 vcc_lo, s20, 4
    code.push_back(EncodeSmem0(0x0a, 8, 36));                           // s_buffer_load_dwordx4 s[8:11], s[72:75], vcc_lo
    code.push_back(EncodeSmem1(0, 106));
    code.push_back(EncodeSopp(0x0c, 0xc07f));                           // s_waitcnt lgkmcnt(0)
    code.push_back(EncodeSopc(0x06, InlineU32(i & 1u), InlineU32(1)));  // SCC = odd call
    code.push_back(EncodeSMovB32(14, 8));
    code.push_back(EncodeSMovB32(15, 9));
    code.push_back(EncodeSMovB32(16, 10));
    code.push_back(0xbe8e210eu);                                        // s_swappc_b64 s[14:15], s[14:15]
    code.push_back(EncodeSop2(0x0a, 31, InlineU32(1), InlineU32(2)));   // s_cselect_b32 s31, 1, 2
    AppendStoreSgpr(&code, 31, Out + i * 3 + 0);
    code.push_back(EncodeVop1(0x02, 32, Vgpr(1)));                      // v_readfirstlane_b32 s32, v1
    AppendStoreSgpr(&code, 32, Out + i * 3 + 1);
    code.push_back(EncodeVop1(0x02, 33, Vgpr(2)));
    AppendStoreSgpr(&code, 33, Out + i * 3 + 2);
  }
  AppendEnd(&code);
  const auto callee_c_pc = static_cast<u32>(TestCallees::Address(CalleeBase, 2) + 32u);
  test.expected.assign(test.initial.begin(), test.initial.begin() + Out); // the table
  test.expected.insert(test.expected.end(), {2, 17, 0,           // A: 10 + 7; its s_add clears SCC
                   1, 20, 0,           // B: SCC kept (odd call)
                   1, 0x555, callee_c_pc, // C: s16 == 5; its compare sets SCC
                   1, 999, 0,          // null record: skipped, SCC kept
                   2, 999, 0,          // D, outside the expanded set: skipped, SCC kept
                   1, 20, 0});

  // The expansion, as the pipeline cache does it with this user data and memory.
  static std::vector<u32> callee_words;
  callee_words = callees.words;
  FakeGuestMemory memory;
  memory.regions = {{GuestBase, &seen}, {CalleeBase, &callee_words}};
  std::array<u32, 4> user_data{0u, 0u, 8192u, 3u << 28u};
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.shader_hash = 0x5ca11ab1e0000001ull;
  options.user_data = user_data;
  ShaderRecompiler::ShaderCallExpander expander;
  ShaderRecompiler::ShaderCallExpander::Expanded expanded;
  Require(test.name, "expanded",
          expander.Expand(code, reinterpret_cast<uint64_t>(code.data()), options, memory.Reader(),
                          expanded) &&
              expanded.code != nullptr,
          "the call-table program did not expand");
  Require(test.name, "no call left",
          !ShaderRecompiler::Decoder::MayCallFunctions(*expanded.code),
          "the expansion still holds an S_SWAPPC_B64");
  code = *expanded.code;
  test.opcodes = {O::S_BUFFER_LOAD_DWORDX4, O::S_CMP_EQ_U32, O::S_CBRANCH_SCC0, O::S_CSELECT_B32,
                  O::S_ENDPGM};
  return test;
}

// ShaderCallExpander: an expansion is kept while the targets stay the same; a target that appears
// in the table is picked up when the descriptor changes or at the periodic rescan, and a changed
// callee at the rescan. A program without calls, or with a call whose target is not traced, does
// not expand. A callee that cannot be placed (here: a nested call) is left out.
void CallExpanderRefreshes() {
  constexpr const char *name = "CallExpanderRefreshes";
  constexpr uint64_t DataBase = 0x0000000150000000ull;
  constexpr uint64_t CalleeBase = 0x0000000160000000ull;
  static TestCallees callees;
  static std::vector<u32> callee_words;
  callee_words = callees.words;
  // A fifth callee with a nested call.
  callee_words.resize(callee_words.size() + TestCallees::Stride / 4, 0xbf810000u);
  callee_words[4 * TestCallees::Stride / 4] = 0xbe94210au; // s_swappc_b64 s[20:21], s[10:11]
  callee_words[4 * TestCallees::Stride / 4 + 1] = EncodeSop1(0x20, 0, 14);
  // Data: the table V# at +0, the table at +0x100 (4 records).
  static std::vector<u32> data;
  data.assign(0x200 / 4, 0u);
  const uint64_t table = DataBase + 0x100;
  data[0] = static_cast<u32>(table);
  data[1] = static_cast<u32>(table >> 32u);
  data[2] = 4u * 16u;
  const auto record = [&](u32 index, uint64_t function) {
    data[0x100 / 4 + index * 4 + 0] = static_cast<u32>(function);
    data[0x100 / 4 + index * 4 + 1] = static_cast<u32>(function >> 32u);
  };
  record(0, TestCallees::Address(CalleeBase, 0));
  record(1, TestCallees::Address(CalleeBase, 4)); // the nested call: left out
  FakeGuestMemory memory;
  memory.regions = {{DataBase, &data}, {CalleeBase, &callee_words}};

  std::vector<u32> code;
  code.push_back(EncodeSmem0(0x02, 72, 2)); // s_load_dwordx4 s[72:75], s[4:5], 0
  code.push_back(EncodeSmem1(0, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f));
  code.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4))); // s_lshl_b32 vcc_lo, s6, 4
  code.push_back(EncodeSmem0(0x0a, 8, 36));
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSMovB32(14, 8));
  code.push_back(EncodeSMovB32(15, 9));
  code.push_back(0xbe8e210eu);
  AppendEnd(&code);
  std::array<u32, 8> user_data{};
  user_data[4] = static_cast<u32>(DataBase);
  user_data[5] = static_cast<u32>(DataBase >> 32u);
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.shader_hash = 0x5ca11ab1e0000002ull;
  options.user_data = user_data;
  const uint64_t base = 0x0000000170000000ull;
  ShaderRecompiler::ShaderCallExpander expander;
  expander.rescan_interval = 4;
  expander.sticky = false; // every table change is followed
  ShaderRecompiler::ShaderCallExpander::Expanded first, again, grown, patched, moved;
  const auto expand = [&](ShaderRecompiler::ShaderCallExpander::Expanded &out) {
    return expander.Expand(code, base, options, memory.Reader(), out) && out.code != nullptr;
  };
  Require(name, "first", expand(first), "the program did not expand");
  Require(name, "kept", expand(again) && again.code == first.code,
          "an unchanged dispatch expanded the program again");
  record(2, TestCallees::Address(CalleeBase, 1));
  // Dispatches 3 and 4: no rescan yet (every 4th dispatch from the first rescans).
  Require(name, "before the rescan",
          expand(again) && again.code == first.code && expand(again) && again.code == first.code,
          "a table change was picked up before the rescan without a descriptor change");
  Require(name, "rescan", expand(grown) && grown.digest != first.digest,
          "the periodic rescan missed a new target");
  callee_words[TestCallees::Stride / 4] = EncodeVop1(0x01, 1, InlineU32(3)); // patch B
  for (int i = 0; i < 3; ++i) {
    Require(name, "patched callee", expand(patched) && patched.code == grown.code,
            "a changed callee was picked up before the rescan");
  }
  Require(name, "patched callee", expand(patched) && patched.digest != grown.digest,
          "a changed callee was not expanded again at the rescan");
  // A new table (the descriptor changes): read at once.
  data[0] = static_cast<u32>(table + 0x40);
  data[2] = 2u * 16u;
  record(4, TestCallees::Address(CalleeBase, 2));
  Require(name, "moved table", expand(moved) && moved.digest != patched.digest,
          "a changed table descriptor was not read at once");

  // The expansion of the nested-call callee is left out, and the program still has no call.
  Require(name, "nested call left out", !ShaderRecompiler::Decoder::MayCallFunctions(*moved.code),
          "a callee with a nested call was placed");

  // No calls; an untraced target (s[14:15] computed).
  std::vector<u32> plain;
  plain.push_back(EncodeSMovB32(20, 4));
  AppendEnd(&plain);
  ShaderRecompiler::ShaderCallExpander::Expanded none;
  Require(name, "no calls",
          !expander.Expand(plain, base + 0x1000, options, memory.Reader(), none),
          "a program without calls expanded");
  std::vector<u32> computed;
  computed.push_back(EncodeSMovB32(14, 4));
  computed.push_back(EncodeSMovB32(15, 5));
  computed.push_back(EncodeSop2(0x00, 14, 14, InlineU32(4))); // s_add_u32 s14, s14, 4
  computed.push_back(0xbe8e210eu);
  AppendEnd(&computed);
  Require(name, "untraced",
          !expander.Expand(computed, base + 0x2000, options, memory.Reader(), none),
          "a call with an untraced target expanded");
  std::printf("[compute] %-32s ok\n", name);
}

// KYTY_CALL_LEARN (adapted from usa-ratt/KytyPS5-GT7 8b76a992a): a table call over the bound is
// expanded with the targets its dispatches reported (at most learn_variants, in report order);
// every other target reports itself (MissReport) and is skipped. A table within the bound, or
// the mode off, leaves the program to Expand.
void CallExpanderLearns() {
  constexpr const char *name = "CallExpanderLearns";
  constexpr uint64_t DataBase = 0x0000000150000000ull;
  constexpr uint64_t CalleeBase = 0x0000000160000000ull;
  static TestCallees callees;
  static std::vector<u32> callee_words;
  callee_words = callees.words;
  static std::vector<u32> data;
  data.assign(0x200 / 4, 0u);
  const uint64_t table = DataBase + 0x100;
  data[0] = static_cast<u32>(table);
  data[1] = static_cast<u32>(table >> 32u);
  data[2] = 3u * 16u;
  for (u32 index = 0; index < 3u; ++index) {
    const auto function = TestCallees::Address(CalleeBase, index);
    data[0x100 / 4 + index * 4 + 0] = static_cast<u32>(function);
    data[0x100 / 4 + index * 4 + 1] = static_cast<u32>(function >> 32u);
  }
  FakeGuestMemory memory;
  memory.regions = {{DataBase, &data}, {CalleeBase, &callee_words}};
  std::vector<u32> code;
  code.push_back(EncodeSmem0(0x02, 72, 2)); // s_load_dwordx4 s[72:75], s[4:5], 0
  code.push_back(EncodeSmem1(0, 125));
  code.push_back(EncodeSopp(0x0c, 0xc07f));
  code.push_back(EncodeSop2(0x1e, 106, 6, InlineU32(4))); // s_lshl_b32 vcc_lo, s6, 4
  code.push_back(EncodeSmem0(0x0a, 8, 36));
  code.push_back(EncodeSmem1(0, 106));
  code.push_back(EncodeSMovB32(14, 8));
  code.push_back(EncodeSMovB32(15, 9));
  code.push_back(0xbe8e210eu);
  AppendEnd(&code);
  std::array<u32, 8> user_data{};
  user_data[4] = static_cast<u32>(DataBase);
  user_data[5] = static_cast<u32>(DataBase >> 32u);
  ShaderRecompiler::CompileOptions options;
  options.stage = ShaderType::Compute;
  options.shader_hash = 0x5ca11ab1e0000003ull;
  options.user_data = user_data;
  const uint64_t base = 0x0000000180000000ull;
  const u32 marker = ShaderRecompiler::Decoder::EncodeCallStateWord(
      ShaderRecompiler::Decoder::CallStateOp::MissReport, 1u);
  const auto reports = [&](const std::vector<u32> &words) {
    for (size_t i = 0; i + 1u < words.size(); ++i) {
      if (words[i] == marker && words[i + 1u] == 14u) return true;
    }
    return false;
  };
  const auto key = [](uint64_t target) { return static_cast<u32>(target) | 1u; };

  ShaderRecompiler::ShaderCallExpander expander;
  expander.learn = true;
  expander.learn_over = 1;
  expander.learn_variants = 2;
  ShaderRecompiler::CallPartitionRun probe, first, again, capped;
  Require(name, "probe",
          expander.Partition(code, base, options, memory.Reader(), probe) && probe.parts.size() == 1u &&
              probe.parts[0].target == 0 && probe.parts[0].code != nullptr &&
              reports(*probe.parts[0].code) &&
              !ShaderRecompiler::Decoder::MayCallFunctions(*probe.parts[0].code),
          "a table over the bound did not run the reporting expansion first");
  const std::array<u32, 1> a{key(TestCallees::Address(CalleeBase, 1))};
  expander.ReportPartitionKeys(probe.program, a);
  Require(name, "learned",
          expander.Partition(code, base, options, memory.Reader(), first) && first.parts.size() == 1u &&
              first.parts[0].target == TestCallees::Address(CalleeBase, 1) &&
              first.parts[0].code->size() > probe.parts[0].code->size() && reports(*first.parts[0].code),
          "a reported target was not placed in the next expansion");
  Require(name, "kept",
          expander.Partition(code, base, options, memory.Reader(), again) &&
              again.parts[0].code == first.parts[0].code,
          "an unchanged learned set expanded the program again");
  // One report set at a time, so the report order is callee 1, 2, 0.
  const std::array<u32, 1> second{key(TestCallees::Address(CalleeBase, 2))};
  const std::array<u32, 1> third{key(TestCallees::Address(CalleeBase, 0))};
  expander.ReportPartitionKeys(probe.program, second);
  expander.ReportPartitionKeys(probe.program, third);
  Require(name, "capped",
          expander.Partition(code, base, options, memory.Reader(), capped) &&
              capped.parts[0].digest != first.parts[0].digest &&
              capped.parts[0].target == TestCallees::Address(CalleeBase, 1) &&
              capped.parts[0].code->size() > first.parts[0].code->size(),
          "the second reported target was not added, or the bound was ignored");
  {
    // learn_variants 2: the third reported target (callee 0) stays out. A run with all three placed
    // is longer still.
    ShaderRecompiler::ShaderCallExpander all;
    all.learn = true;
    all.learn_over = 1;
    all.learn_variants = 3;
    ShaderRecompiler::CallPartitionRun run;
    (void)all.Partition(code, base, options, memory.Reader(), run);
    const std::array<u32, 3> every{key(TestCallees::Address(CalleeBase, 1)),
                                   key(TestCallees::Address(CalleeBase, 2)),
                                   key(TestCallees::Address(CalleeBase, 0))};
    all.ReportPartitionKeys(run.program, every); // one set: added in table order
    Require(name, "bound",
            all.Partition(code, base, options, memory.Reader(), run) &&
                run.parts[0].code->size() > capped.parts[0].code->size(),
            "KYTY_CALL_VARIANTS did not bound the learned targets");
  }
  {
    ShaderRecompiler::ShaderCallExpander off;
    off.learn = false;
    ShaderRecompiler::CallPartitionRun run;
    Require(name, "off",
            ShaderRecompiler::CallPartitionsEnabled() ||
                !off.Partition(code, base, options, memory.Reader(), run),
            "the learned mode ran while off");
    ShaderRecompiler::ShaderCallExpander within;
    within.learn = true;
    within.learn_over = 3;
    Require(name, "within the bound",
            ShaderRecompiler::CallPartitionsEnabled() ||
                !within.Partition(code, base, options, memory.Reader(), run),
            "a table within the bound took the learned mode");
  }
  std::printf("[compute] %-32s ok\n", name);
}

void RunAll(VulkanHarness *vulkan) {
  auto options = GetCodegenOptions();
  options.srt_variant_reads = true;
  {
    // The skip and dump behaviour of programs with calls (KYTY_SHADER_CALLS=0).
    auto skipping = options;
    skipping.shader_calls = false;
    ScopedCodegenOptions scope(skipping);
    RunCase(vulkan, ScalarBufferLoadsGpuSelectedDescriptors());
    RunCase(vulkan, ScalarLoadsThroughLoopCarriedRecords());
    RunCase(vulkan, RawLoadsThroughGpuSelectedDescriptors());
    {
      // KYTY_INDIRECT_RAW_OUTLINE: the same loads through the module's outlined functions.
      auto outline = skipping;
      outline.indirect_raw_outline = true;
      ScopedCodegenOptions outline_scope(outline);
      auto test = RawLoadsThroughGpuSelectedDescriptors();
      test.name = "RawLoadsOutlined";
      test.required_spirv.push_back("indirect_raw_x4");
      test.required_spirv.push_back("indirect_raw_u8");
      RunCase(vulkan, test);
    }
    RunCase(vulkan, FormattedLoadsThroughGpuSelectedDescriptors());
    {
      // KYTY_INDIRECT_FORMAT_COMPACT: the table-driven decoder gives the same channels.
      auto compact = skipping;
      compact.indirect_format_compact = true;
      ScopedCodegenOptions compact_scope(compact);
      auto test = FormattedLoadsThroughGpuSelectedDescriptors();
      test.name = "FormattedLoadsCompactDecoder";
      test.required_spirv.push_back("indirect_formatted_load_compact");
      RunCase(vulkan, test);
    }
    CallIsSkipped();
    HugeHelperOutlineControl();
    CallTargetsFromUserData();
    CallTablesTraced();
    CallTablesTracedWave32();
    vulkan->CheckCallTargetDump();
    NullBasedFlatReadReadsZero();
    VariantPlanIsFound();
    vulkan->CheckVariantPlanSkipped();
  }
  options.shader_calls = true;
  ScopedCodegenOptions scope(options);
  CallExpanderRefreshes();
  CallExpanderLearns();
  vulkan->CheckCallTableDispatch();
  RunCase(vulkan, CallTableExpanded());
}

} // namespace SrtVariantTests
