// Included after ShaderProgramCacheTests.inc in ShaderRecompilerComputeTests.cpp.
// --async-pipeline-only: uses the real renderer, driver and cache. The driver gate
// only delays vkCreateGraphicsPipelines; it forwards unchanged to the real driver.
// Run in a fresh process with KYTY_ASYNC_PIPELINES=1, KYTY_PIPELINE_LIBRARY=0,
// KYTY_PROGRAM_CACHE=0, KYTY_PIPELINE_CACHE_SAVE=0 and KYTY_DRAW_PREP=off.
namespace AsyncPipelineTests {

constexpr const char *Name = "AsyncPipelinePublication";

class DriverGate;
DriverGate *active_gate = nullptr;

class DriverGate {
public:
  DriverGate() : original(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines) {
    Require(Name, "driver gate", active_gate == nullptr, "nested driver gate");
    active_gate = this;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines = Create;
  }
  ~DriverGate() {
    // The caller drains the cache before this scope ends: no compiler may still
    // be using either the dispatch pointer or the gate's bookkeeping.
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines = original;
    active_gate = nullptr;
  }
  DriverGate(const DriverGate &) = delete;
  DriverGate &operator=(const DriverGate &) = delete;

  void WaitEntered() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Require(Name, "compiler entered driver", entered.load(), "background compile never started");
  }
  void Release() { released.store(true, std::memory_order_release); }

  std::atomic<bool> entered{false}, released{false}, timed_out{false};
  std::atomic<unsigned> calls{0};

private:
  static VKAPI_ATTR VkResult VKAPI_CALL Create(VkDevice device, VkPipelineCache cache,
      uint32_t count, const VkGraphicsPipelineCreateInfo *infos,
      const VkAllocationCallbacks *allocator, VkPipeline *pipelines) {
    auto &gate = *active_gate;
    gate.calls.fetch_add(count, std::memory_order_relaxed);
    gate.entered.store(true, std::memory_order_release);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!gate.released.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        // A sync-only regression reaches this call on the recording thread.
        // Return through the real driver after a deadline, so the test fails
        // its null-return assertion instead of hanging indefinitely.
        gate.timed_out.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return gate.original(device, cache, count, infos, allocator, pipelines);
  }
  PFN_vkCreateGraphicsPipelines original;
};

std::vector<u32> VertexCode(bool writes) {
  std::vector<u32> code;
  if (writes) {
    // Each of the three vertex invocations writes its own buffer word. The
    // native vertex index is v5 in this fixture (same as DrawPrepEngineDraw).
    AppendVMovLiteral(&code, 7, 0x12345678u);
    code.push_back(EncodeMubuf0(0x1cu, 0, true, false));
    // GS/vertex user SGPRs begin at s8 (s0..s7 are ABI inputs). MUBUF's
    // srsrc field is measured in groups of four: 2 selects s[8:11].
    code.push_back(EncodeMubuf1(7, 2, 5));
  }
  AppendVMovLiteral(&code, 1, 0xbf800000u);
  AppendVMovLiteral(&code, 2, 0x40400000u);
  code.push_back(EncodeVopc(0xc2, InlineU32(1), 5));
  code.push_back(EncodeVop2(0x01, 3, Vgpr(1), 2));
  code.push_back(EncodeVopc(0xc2, InlineU32(2), 5));
  code.push_back(EncodeVop2(0x01, 4, Vgpr(1), 2));
  AppendVMovU32(&code, 0, 0);
  AppendVMovLiteral(&code, 6, 0x3f800000u);
  code.push_back(EncodeExp0(0x0c, 0xf));
  code.push_back(EncodeExp1(3, 4, 0, 6));
  for (u32 parameter = 0; parameter < 8; ++parameter) {
    code.push_back(EncodeExp0(0x20 + parameter, 0xf));
    code.push_back(EncodeExp1(0, 0, 0, 0));
  }
  AppendEnd(&code);
  return code;
}

std::vector<u32> PixelCode(bool writes) {
  std::vector<u32> code;
  if (writes) {
    // The writer draw has a one-pixel scissor, so this store has one real
    // fragment invocation (helper invocations must not write storage).
    AppendVMovLiteral(&code, 7, 0xabcdef01u);
    code.push_back(EncodeMubuf0(0x1cu, 0, false, false));
    code.push_back(EncodeMubuf1(7, 0, 0));
  }
  AppendVMovU32(&code, 0, 0);
  AppendVMovLiteral(&code, 1, 0x3f800000u);
  code.push_back(EncodeExp0(0x00, 0xf));
  code.push_back(EncodeExp1(0, 1, 0, 1));
  AppendEnd(&code);
  return code;
}

void Check(VulkanHarness *vulkan) {
  Require(Name, "async configuration", PipelineCache::AsyncPipelinesEnabled(),
          "run this case with KYTY_ASYNC_PIPELINES=1 and KYTY_PIPELINE_LIBRARY=0");
  constexpr uintptr_t base = 0x000000020b000000ull;
  constexpr uint64_t size = 0x80000, alignment = 0x10000;
  constexpr uint64_t color_address = base + 0x20000, output_address = base + 0x40000;
  constexpr u32 extent = 32;
  const std::array<std::vector<u32>, 4> code{
      VertexCode(false), PixelCode(false), VertexCode(true), PixelCode(true)};
  const std::array<uint64_t, 4> addresses{base + 0x1000, base + 0x2000,
                                         base + 0x3000, base + 0x4000};
  (void)vulkan->RuntimeContext();
  int64_t direct_offset = -1;
  Require(Name, "direct allocation", LibKernel::Memory::KernelAllocateDirectMemory(
      0, LibKernel::Memory::KernelGetDirectMemorySize(), size, alignment, 0, &direct_offset) == 0,
      "fixture allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(Name, "direct mapping", LibKernel::Memory::KernelMapDirectMemory(
      &mapped, size, 0x3, 0x10, direct_offset, alignment) == 0 &&
      mapped == reinterpret_cast<void *>(base), "fixture mapping failed");
  std::memset(mapped, 0, size);
  auto *user_data = reinterpret_cast<ShaderUserData *>(base + 0x5000);
  for (size_t i = 0; i < code.size(); ++i) {
    std::memcpy(reinterpret_cast<void *>(addresses[i]), code[i].data(), code[i].size() * sizeof(u32));
    ShaderMapUserData(addresses[i], {
        .type = i % 2 == 0 ? Prospero::ShaderBinaryType::kGs : Prospero::ShaderBinaryType::kPs,
        .user_data = user_data, .code_size_bytes = static_cast<uint32_t>(code[i].size() * sizeof(u32))});
  }
  auto context_owner = vulkan->MakeRenderContext();
  auto &context = *context_owner;
  context.InitializeGpu(nullptr);
  LibKernel::Memory::InstallGpuResources(&context);
  DriverGate gate;
  context.GetGpu().SendCommandSync([&] {
    GraphicsInitJmpTables();
    CommandProcessor processor(context, 0);
    processor.Reset();
    processor.BufferInit();
    context.MapMemory(base, size);
    auto &scheduler = context.GetCommandScheduler();
    auto &executor = context.GetRenderExecutor();
    auto &cache = context.GetPipelineCache();
    auto &registers = processor.GetCtx();
    auto &shaders = processor.GetShCtx();
    registers.SetViewportTransformControl(0x300);
    registers.SetViewportScaleOffset(0, extent / 2, extent / 2, extent / 2, extent / 2, 1, 0);
    registers.SetViewportZMax(0, 1);
    registers.SetScreenScissor(0, 0, extent, extent);
    registers.SetWindowScissor(0, 0, extent, extent, false);
    registers.SetGenericScissor(0, 0, extent, extent, false);
    registers.SetViewportScissor(0, 0, 0, extent, extent, false);
    registers.SetRenderTargetMask(0xf);
    registers.SetShaderMask(0xf);
    registers.SetPsInControl(0x8000);
    registers.SetColorBase(0, {.addr = color_address});
    registers.SetColorInfo(0, {.format = Prospero::ChannelLayout::k32_32_32_32,
        .channel_type = Prospero::ChannelType::kFloat, .channel_order = Prospero::ChannelOrder::kStandard});
    registers.SetColorAttrib2(0, {.height = extent - 1, .width = extent - 1});
    registers.SetColorAttrib3(0, {.tile_mode = Prospero::TileMode::kLinear, .dimension = 1});
    registers.SetTargetOutputMode(0, 4);
    processor.GetUcfg().SetPrimitiveType(Prospero::PrimitiveType::kTriList);
    shaders.SetEsShaderBase(addresses[0]);
    shaders.SetPsShaderBase(addresses[1]);
    RenderColorInfo color{};
    RenderDepthInfo depth{};
    RenderExecutorTestAccess::ResolveRenderColorTarget(executor, scheduler.Current(), color, 0);
    Require(Name, "color fixture", static_cast<bool>(color.image_id), "missing render target");
    std::array<Prospero::ColorComponentMapping, 8> mappings{};
    mappings[0] = color.export_mapping;
    struct Query {
      std::array<ShaderVertexInputInfo, 3> vertex{};
      ShaderPixelInputInfo pixel{};
      PipelineCache::GraphicsStagePreps preps;
      PipelineCache::GraphicsPrograms programs;
    };
    // The shader inputs are large; do not put them on Windows' 1 MiB thread stack.
    auto query = std::make_unique<Query>();
    const auto prepare = [&] {
      query->programs = cache.GetGraphicsPrograms(shaders.GetVs(), shaders.GetPs(),
          registers.GetShaderRegisters(), registers, processor.GetUcfg(), mappings, true,
          query->vertex, query->pixel, query->preps);
    };
    const auto lookup = [&](bool defer) {
      return cache.TryGetGraphicsPipeline(std::span{&color, 1u}, depth,
          std::span{query->vertex.data(), 1u}, scheduler.Current(), &query->pixel,
          vk::PrimitiveTopology::eTriangleList, false, query->programs, defer);
    };
    PipelineCache::PipelineTargets targets{};
    targets.color_count = 1;
    targets.colors[0] = {.slot = color.target_slot, .format = color.desc.view_info.format,
        .samples = color.desc.info.samples, .export_mapping = color.export_mapping};
    const PipelineCache::Pipeline *planned = nullptr;
    uint64_t generation = 0;
    const auto plan = [&] {
      return cache.FindGraphicsPipelineForPlan(targets, registers, processor.GetUcfg(),
          query->vertex[0], &query->pixel, vk::PrimitiveTopology::eTriangleList, false,
          query->programs, planned, generation);
    };
    prepare();
    Require(Name, "cold defer", lookup(true) == nullptr,
            "a new eligible pipeline was created synchronously instead of deferred");
    gate.WaitEntered();
    for (unsigned i = 0; i < 3; ++i) {
      Require(Name, "pending defer", lookup(true) == nullptr, "pending pipeline was exposed");
      Require(Name, "pending plan", plan() == PipelineCache::PlanLookup::Absent,
              "draw-prep exposed an unpublished pipeline");
    }
    Require(Name, "one compile per key", gate.calls.load() == 1,
            "repeated pending lookups queued duplicate driver compiles");
    // A non-deferrable lookup must return the published object. Release from a
    // helper so the recording thread must handle an already pending pipeline.
    std::thread release([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      gate.Release();
    });
    auto *published = lookup(false);
    release.join();
    Require(Name, "published pipeline", published != nullptr && published->pipeline != nullptr &&
        !published->pending && !gate.timed_out.load(), "pending lookup did not wait for a real pipeline");
    Require(Name, "published reuse", lookup(true) == published &&
        plan() == PipelineCache::PlanLookup::Found && planned == published && gate.calls.load() == 1,
        "published pipeline was not reused by draw-prep and recording lookups");
    TextureCacheTestAccess::ClearImage(context.GetTextureCache(), scheduler.Current(), color.image_id,
        {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}, {});
    RenderExecutorTestAccess::DrawAuto(executor, scheduler.Current(), {.vertex_count = 3, .instance_count = 1});
    const auto pixels = vulkan->ReadCachedTexel(Name, context, color.image_id, {}, {extent, extent, 1});
    Require(Name, "published draw", pixels.size() == extent * extent * 4u, "missing color readback");
    for (size_t i = 0; i < pixels.size(); ++i)
      Require(Name, "published draw pixels", pixels[i] == (i % 2 ? 0x3f800000u : 0),
              "published pipeline did not render the expected green triangle");

    // Cold pipelines, one draw each: do not retry. A skipped writer cannot be
    // rescued by the compile finishing before the readback; its write was lost.
    registers.SetScreenScissor(0, 0, 1, 1);
    for (bool vertex_writer : {false, true}) {
      const auto output = output_address + (vertex_writer ? 0x10000u : 0u);
      shaders.SetEsShaderBase(addresses[vertex_writer ? 2 : 0]);
      shaders.SetPsShaderBase(addresses[vertex_writer ? 1 : 3]);
      ShaderBufferResource descriptor{};
      descriptor.UpdateAddress48(output);
      descriptor.fields[1] |= 4u << 16u; // one dword per vertex index
      descriptor.fields[2] = 3;
      descriptor.fields[3] = DstSel(4, 5, 6, 7) | (1u << 24u) |
          (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
      shaders.SetGsShaderResource2({.user_sgpr = static_cast<uint8_t>(vertex_writer ? 4 : 0)});
      shaders.SetPsShaderResource2({.user_sgpr = static_cast<uint8_t>(vertex_writer ? 0 : 4)});
      for (u32 i = 0; i < 4; ++i) {
        if (vertex_writer) shaders.SetGsUserSgpr(i, descriptor.fields[i], HW::UserSgprType::Vsharp);
        else shaders.SetPsUserSgpr(i, descriptor.fields[i], HW::UserSgprType::Vsharp);
      }
      prepare();
      const auto &info = vertex_writer ? query->vertex[0].stage.program->info : query->pixel.stage.program->info;
      Require(Name, "writer fixture", std::any_of(info.buffers.begin(), info.buffers.end(),
          [](const auto &resource) { return resource.written; }), "fixture did not compile a buffer write");
      const auto before = gate.calls.load();
      RenderExecutorTestAccess::DrawAuto(executor, scheduler.Current(), {.vertex_count = 3, .instance_count = 1});
      context.GetBufferCache().ReadMemory(output, 3 * sizeof(u32));
      std::array<u32, 3> actual{};
      Require(Name, "writer readback", LibKernel::Memory::TryReadBacking(output, actual.data(), sizeof(actual)),
              "writer output is inaccessible");
      const std::array<u32, 3> expected = vertex_writer
          ? std::array<u32, 3>{0x12345678u, 0x12345678u, 0x12345678u}
          : std::array<u32, 3>{0xabcdef01u, 0, 0};
      Require(Name, vertex_writer ? "vertex writer never skipped" : "pixel writer never skipped",
              actual == expected, "a cold memory-writing draw lost its storage write: " +
                  Hex(actual[0]) + ", " + Hex(actual[1]) + ", " + Hex(actual[2]));
      auto *writer_pipeline = lookup(true);
      Require(Name, "writer pipeline ready", writer_pipeline != nullptr && !writer_pipeline->pending &&
          lookup(false) == writer_pipeline && gate.calls.load() == before + 1,
          "writer pipeline was deferred or compiled more than once");
    }
    scheduler.Finish();
    cache.Save(); // joins every compiler before the dispatch gate is removed
    context.UnmapMemory(base, size);
  });
  context.ShutdownGpu();
  Require(Name, "fixture unmap", LibKernel::Memory::KernelMunmap(base, size) == 0,
          "could not unmap fixture");
  Require(Name, "fixture release", LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, size) == 0,
          "could not release fixture");
  context_owner.reset();
  std::printf("[gpu]     %-32s ok\n", Name);
}

} // namespace AsyncPipelineTests
