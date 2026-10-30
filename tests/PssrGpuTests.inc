  // Experimental Pro mode: the R_PSSR packet's host pass (renderPssr.cpp, upstream 651ef73a0's
  // PssrGpuTests.inc on our harness). Inputs are one texel row (linear, so no pitch); outputs are
  // 256B-tiled 11/11/10 images read back from the texture cache.
  void CheckPssr() {
    constexpr const char *name = "PssrLinear";
    constexpr uintptr_t base = 0x0000000206a00000ull;
    constexpr uint64_t size = 0x100000;
    constexpr auto packed = Prospero::BufferFormat::k11_11_10Float;
    constexpr auto linear = Prospero::TileMode::kLinear;
    constexpr auto tiled = Prospero::TileMode::kStandard256B;
    constexpr u32 red = 0x000003c0, green = 0x001e0000, blue = 0x78000000;
    const auto descriptor = [](uint64_t address, u32 width, u32 height,
                               Prospero::BufferFormat format, Prospero::TileMode tile,
                               u32 swizzle = DstSel(4, 5, 6, 7)) {
      ShaderTextureResource result{};
      result.fields[0] = static_cast<u32>(address >> 8);
      result.fields[1] = static_cast<u32>(address >> 40) |
                         (static_cast<u32>(format) << 20) | (((width - 1) & 3) << 30);
      result.fields[2] = ((width - 1) >> 2) | ((height - 1) << 14);
      result.fields[3] = swizzle | (static_cast<u32>(tile) << 20) |
                         (static_cast<u32>(Prospero::ImageType::kColor2D) << 28);
      result.fields[5] = 0x00700000;
      return result;
    };
    EnsureRuntimeContext();
    int64_t physical = -1;
    Require(name, "allocate",
            Libs::LibKernel::Memory::KernelAllocateDirectMemory(
                0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), size, 0x10000, 0,
                &physical) == 0,
            "upscaling allocation failed");
    void *mapped = reinterpret_cast<void *>(base);
    Require(name, "map",
            Libs::LibKernel::Memory::KernelMapDirectMemory(&mapped, size, 0x3, 0x10, physical,
                                                           0x10000) == 0 &&
                mapped == reinterpret_cast<void *>(base),
            "upscaling mapping failed");
    std::memset(mapped, 0, size);
    const auto context_owner = MakeRenderContext();
    auto &context = *context_owner;
    context.InitializeGpu(nullptr);
    LibKernel::Memory::InstallGpuResources(&context);
    context.GetGpu().SendCommandSync([&] {
      GraphicsInitJmpTables();
      CommandProcessor processor(context, 0);
      processor.BufferInit();
      context.MapMemory(base, size);
      auto &executor = context.GetRenderExecutor();
      auto &scheduler = context.GetCommandScheduler();
      const auto resolve = [&](const ShaderTextureResource &texture) {
        ShaderRecompiler::IR::ImageResource resource{};
        resource.resource_class = ShaderRecompiler::IR::ImageResourceClass::Sampled;
        resource.numeric_class = Prospero::TextureNumericClass::Float;
        resource.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
        resource.read = true;
        ShaderRecompiler::IR::DescriptorValue value{};
        value.dword_count = 8;
        std::copy_n(texture.fields, 8, value.dwords.begin());
        return RenderExecutorTestAccess::ResolveTexture(executor, resource, value);
      };
      // A one-row linear image: its texels from the start of its memory, as the CPU writes them.
      const auto write_row = [&](const ShaderTextureResource &texture, std::span<const u32> words) {
        const auto bytes = words.size() * sizeof(u32);
        Require(name, "invalidate source", context.InvalidateMemory(texture.Base40(), bytes),
                "source fixture is outside mapped memory");
        LibKernel::Memory::WriteBacking(texture.Base40(), words.data(), bytes);
      };
      const auto run = [&](const ShaderTextureResource &input, const ShaderTextureResource &output,
                           u32 width, u32 height) {
        Pssr::Request request{};
        std::copy_n(input.fields, 8, request.input);
        std::copy_n(output.fields, 8, request.output);
        request.width = width;
        request.height = height;
        // The reserved packet includes trailing words that the parser must consume.
        constexpr u32 reserved_words = Pssr::RequestDwords + 9;
        std::vector<u32> packet(reserved_words, 0xdeadbeef);
        packet[0] = KYTY_PM4(reserved_words, Pm4::IT_NOP, Pm4::R_PSSR);
        std::memcpy(packet.data() + 1, &request, sizeof(request));
        alignas(u32) u32 sentinel = 0;
        const auto address = reinterpret_cast<uint64_t>(&sentinel);
        packet.insert(packet.end(), {KYTY_PM4(5, Pm4::IT_WRITE_DATA, 0), 0,
                                     static_cast<u32>(address), static_cast<u32>(address >> 32),
                                     33});
        Pm4Execution execution;
        Require(name, "packet and suffix",
                processor.Process(execution, packet) == Pm4ProcessResult::Complete,
                "upscaling packet did not complete");
        processor.BufferFlush();
        scheduler.Finish();
        Require(name, "suffix", sentinel == 33,
                "upscaling packet did not consume its reserved payload before the suffix");
        const auto binding = resolve(output);
        return ReadCachedTexel(name, context, binding.image_id, {},
                               {output.Width5() + 1u, output.Height5() + 1u, 1},
                               binding.desc.view_info.base_layer, binding.desc.view_info.base_level);
      };

      auto input = descriptor(base, 2, 1, packed, linear);
      auto output = descriptor(base + 0x10000, 4, 1, packed, tiled);
      write_row(input, std::array<u32, 2>{red, green});
      Require(name, "CPU-first bilinear source",
              run(input, output, 2, 1) ==
                  std::vector<u32>{red, 0x3a0 | (0x340 << 11), 0x340 | (0x3a0 << 11), green},
              "first-use source upload or linear interpolation is wrong");
      write_row(input, std::array<u32, 2>{blue, blue});
      Require(name, "CPU source refresh", run(input, output, 2, 1) == std::vector<u32>(4, blue),
              "reused source retained stale GPU texels");

      input = descriptor(base + 0x20000, 2, 1, packed, linear);
      output = descriptor(base + 0x30000, 2, 1, packed, tiled);
      write_row(input, std::array<u32, 2>{red, green});
      Require(name, "same-size copy", run(input, output, 2, 1) == std::vector<u32>{red, green},
              "same-size upscaling lost source pixels");

      // Only the request's active rectangle is sampled: the two red texels of an eight-texel row.
      input = descriptor(base + 0x40000, 8, 1, packed, linear);
      output = descriptor(base + 0x50000, 8, 1, packed, tiled);
      std::vector<u32> cropped(8, blue);
      cropped[0] = cropped[1] = red;
      write_row(input, cropped);
      Require(name, "active rectangle", run(input, output, 2, 1) == std::vector<u32>(8, red),
              "equal allocation extents bypassed scaling or sampled outside the active rectangle");

      input = descriptor(base + 0x60000, 2, 1, packed, linear, DstSel(6, 5, 4, 7));
      output = descriptor(base + 0x70000, 2, 1, packed, tiled);
      write_row(input, std::array<u32, 2>{red, green});
      Require(name, "sampled view swizzle", run(input, output, 2, 1) == std::vector<u32>{blue, green},
              "sampled descriptor swizzle was ignored");
      output.fields[3] = (output.fields[3] & ~0xfffu) | DstSel(6, 5, 4, 7);
      Require(name, "storage swizzle", run(input, output, 2, 1) == std::vector<u32>{red, green},
              "output channel mapping was ignored");

      input = descriptor(base + 0x80000, 2, 1, Prospero::BufferFormat::k16_16_16_16Float, linear);
      output = descriptor(base + 0x90000, 2, 1, packed, tiled);
      write_row(input, std::array<u32, 4>{0x00003c00, 0x3c000000, 0x3c000000, 0x3c000000});
      Require(name, "float16 input", run(input, output, 2, 1) == std::vector<u32>{red, green},
              "float16 input conversion failed");

      // A request the pass cannot run is skipped (logged), not fatal: an unsupported output format.
      input = descriptor(base + 0xa0000, 2, 1, packed, linear);
      output = descriptor(base + 0xb0000, 2, 1, Prospero::BufferFormat::k8_8_8_8UNorm, tiled);
      write_row(input, std::array<u32, 2>{red, green});
      {
        Pssr::Request request{};
        std::copy_n(input.fields, 8, request.input);
        std::copy_n(output.fields, 8, request.output);
        request.width = 2;
        request.height = 1;
        std::vector<u32> packet(1 + Pssr::RequestDwords);
        packet[0] = KYTY_PM4(static_cast<u32>(packet.size()), Pm4::IT_NOP, Pm4::R_PSSR);
        std::memcpy(packet.data() + 1, &request, sizeof(request));
        Pm4Execution execution;
        Require(name, "unsupported request",
                processor.Process(execution, packet) == Pm4ProcessResult::Complete,
                "an unsupported upscaling request did not complete");
        processor.BufferFlush();
        scheduler.Finish();
      }
      context.UnmapMemory(base, size);
      scheduler.Finish();
    });
    context.ShutdownGpu();
    Require(name, "unmap", Libs::LibKernel::Memory::KernelMunmap(base, size) == 0,
            "upscaling mapping release failed");
    Require(name, "release",
            Libs::LibKernel::Memory::KernelReleaseDirectMemory(physical, size) == 0,
            "upscaling allocation release failed");
    std::printf("[gpu]     %-32s ok\n", name);
  }
