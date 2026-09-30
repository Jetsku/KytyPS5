// Included by ShaderRecompilerComputeTests.cpp. Independently authored regression cases using
// the packet facts documented by PS5_Vulkan d609d71, jobs/r70-gpu-barrier (PIDs 614/615).
// These check emulated buffer visibility and packet ordering, not the probe's rendered image.
namespace GuestSyncTests {

uint32_t Low(const void *pointer) {
  return static_cast<uint32_t>(reinterpret_cast<uint64_t>(pointer));
}

uint32_t High(const void *pointer) {
  return static_cast<uint32_t>(reinterpret_cast<uint64_t>(pointer) >> 32u);
}

// R70's prefetch-parser memory equality wait. The high dword is deliberately different from
// the label: only the low dword is compared, although the packet and address are 64-bit.
std::array<uint32_t, 9> WaitPacket(const uint64_t *label, uint32_t reference) {
  return {0xc0079300u, 0x06000113u, Low(label), High(label), reference, 0u,
          UINT32_MAX, 0u, 4u};
}

void MakeHostReadable(CommandScheduler &scheduler, const Buffer &buffer) {
  vk::BufferMemoryBarrier barrier{};
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer.Handle();
  barrier.size = buffer.Size();
  scheduler.Current().Handle().pipelineBarrier(
      vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
      0, nullptr, 1, &barrier, 0, nullptr);
}

void CheckBdaPacketBoundaries(VulkanHarness &vulkan) {
  constexpr const char *name = "BdaPacketBoundaries";
  constexpr uintptr_t base = 0x0000000206C00000ull;
  constexpr uint64_t allocation_size = 0x100000;
  constexpr uint64_t alignment = 0x10000;
  constexpr uint64_t buffer_offset = 0x20000;
  constexpr uint64_t buffer_size = 0x10000;
  constexpr uint64_t value_offset = 0x100;
  constexpr uint32_t initial = 0x10203040u;
  constexpr uint32_t first_write = 0xa1b2c3d4u;
  constexpr size_t snapshot_count = 7;

  auto context_owner = vulkan.MakeRenderContext();
  auto &context = *context_owner;
  auto &scheduler = context.GetCommandScheduler();
  auto &cache = context.GetBufferCache();
  GraphicsInitJmpTables();
  CommandProcessor processor(context, 0);
  processor.BufferInit();
  context.InitializeGpu(nullptr);

  int64_t direct_offset = -1;
  Require(name, "allocate", LibKernel::Memory::KernelAllocateDirectMemory(
              0, LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size, alignment, 0,
              &direct_offset) == 0, "could not allocate guest backing");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "map", LibKernel::Memory::KernelMapDirectMemory(
              &mapped, allocation_size, 0x3, 0x10, direct_offset, alignment) == 0 &&
              mapped == reinterpret_cast<void *>(base), "could not map guest backing");
  auto *memory = static_cast<uint8_t *>(mapped);
  std::memset(memory, 0, allocation_size);
  std::memcpy(memory + buffer_offset + value_offset, &initial, sizeof(initial));
  context.MapMemory(base, allocation_size);

  Buffer readback(vulkan.RuntimeContext(), scheduler, MemoryUsage::Download, 0,
                  vk::BufferUsageFlagBits::eTransferDst, snapshot_count * sizeof(uint32_t));
  std::array<uint32_t, snapshot_count> expected{initial, first_write, first_write,
                                              0x22334455u, 0x33445566u, 0x44556677u,
                                              0x55667788u};
  if (BufferCacheTestAccess::BdaEpochSkip(cache) &&
      BufferCacheTestAccess::BdaEpochVerify(cache) == 0) {
    expected[1] = initial;
  }

  // One service command: separate callbacks advance SyncEpoch themselves and would hide a
  // missing packet boundary. There is deliberately no direct SyncEpoch::Advance in this test.
  VulkanHarness::OnGpuThread(context, [&] {
    const auto [buffer, offset] = cache.ObtainBuffer(base + buffer_offset, buffer_size, false, false);
    Require(name, "cached source", buffer != nullptr, "could not obtain BDA buffer");
    const auto snapshot = [&](size_t index) {
      auto native = scheduler.Current().Handle();
      vk::MemoryBarrier before{};
      before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
      before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
      native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                             vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr,
                             0, nullptr);
      const vk::BufferCopy copy{offset + value_offset, index * sizeof(uint32_t), sizeof(uint32_t)};
      native.copyBuffer(buffer->Handle(), readback.Handle(), 1, &copy);
      // Later uploads may overwrite the source. Preserve each observation before that write.
      vk::MemoryBarrier after{};
      after.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      after.dstAccessMask = vk::AccessFlagBits::eMemoryWrite;
      native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                             vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr,
                             0, nullptr);
    };
    const auto cpu_write = [&](uint32_t value) {
      Require(name, "track CPU write", context.HandleFault(
                  PageFaultAccess::Write, base + buffer_offset + value_offset),
              "guest write was not tracked");
      std::memcpy(memory + buffer_offset + value_offset, &value, sizeof(value));
    };
    const auto execute = [&](std::span<const uint32_t> words) {
      Pm4Execution execution;
      Require(name, "execute boundary", processor.Process(execution, words) ==
                  Pm4ProcessResult::Complete, "a satisfied boundary did not complete");
    };
    const auto synchronize = [&](size_t index) {
      const auto before = BufferCacheTestAccess::BdaEpochTotals(cache).passes;
      context.PrepareBda();
      Require(name, "packet invalidates BDA proof",
              BufferCacheTestAccess::BdaEpochTotals(cache).passes == before + 1,
              "BDA reused a proof across an ordering packet");
      snapshot(index);
    };

    context.PrepareBda();
    snapshot(0);
    const auto first = BufferCacheTestAccess::BdaEpochTotals(cache);
    cpu_write(first_write);
    const std::array<uint32_t, 2> register_only{KYTY_PM4(2, Pm4::IT_NUM_INSTANCES, 0), 1u};
    execute(register_only);
    context.PrepareBda();
    const auto same_epoch = BufferCacheTestAccess::BdaEpochTotals(cache);
    Require(name, "reuse between ordering points",
            !BufferCacheTestAccess::BdaEpochSkip(cache) ||
                (same_epoch.passes == first.passes && same_epoch.skips == first.skips + 1),
            "a register-only packet discarded the BDA proof");
    snapshot(1);

    alignas(8) uint64_t label = 0xdeadbeef00000001ull;
    auto wait = WaitPacket(&label, 1);
    execute(wait);
    synchronize(2);

    cpu_write(expected[3]);
    const std::array<uint32_t, 7> acquire{0xc0055800u, 0x00400000u, 1u, 0u, 0u, 0u, 10u};
    execute(acquire);
    synchronize(3);

    cpu_write(expected[4]);
    const std::array<uint32_t, 8> private_acquire{
        0xc0061050u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    execute(private_acquire);
    synchronize(4);

    // A suspended wait must re-establish visibility when the same execution resumes.
    wait = WaitPacket(&label, 2);
    Pm4Execution suspended;
    Require(name, "wait blocks", processor.Process(suspended, wait) == Pm4ProcessResult::Blocked,
            "a stale label satisfied WAIT_REG_MEM64");
    cpu_write(expected[5]);
    label = 0xdeadbeef00000002ull;
    Require(name, "wait resumes", processor.Process(suspended, wait) == Pm4ProcessResult::Complete,
            "the updated low dword did not release the masked wait");
    synchronize(5);

    cpu_write(expected[6]);
    // Event 20, GCR 12, event index 5; no data write is needed for this epoch boundary.
    const std::array<uint32_t, 8> release{0xc0064900u, 0x0000c514u, 0u, 0u, 0u, 0u, 0u, 0u};
    execute(release);
    synchronize(6);

    MakeHostReadable(scheduler, readback);
    scheduler.Finish();
  });
  readback.Invalidate(0, readback.Size());
  std::array<uint32_t, snapshot_count> actual{};
  std::memcpy(actual.data(), readback.Mapped().data(), sizeof(actual));
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::printf("BdaPacketBoundaries: snapshot %zu got %08x expected %08x\n", i, actual[i], expected[i]);
    }
    Require(name, "GPU bytes", actual[i] == expected[i], "a boundary left stale native buffer bytes");
  }
  context.UnmapMemory(base, allocation_size);
  context.ShutdownGpu();
  Require(name, "unmap", LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
          "could not unmap guest backing");
  Require(name, "release", LibKernel::Memory::KernelReleaseDirectMemory(direct_offset,
              allocation_size) == 0, "could not release guest backing");
  std::printf("[gpu]     %-32s ok (7 native snapshots)\n", name);
}

void CheckReleaseWait(VulkanHarness &vulkan) {
  constexpr const char *name = "ReleaseWait64";
  auto context_owner = vulkan.MakeRenderContext();
  auto &context = *context_owner;
  auto &scheduler = context.GetCommandScheduler();
  HW::Context registers{};
  HW::UserConfig user_config{};
  HW::Shader shaders{};
  scheduler.Begin(registers, user_config, shaders);
  context.InitializeGpu(nullptr);
  auto &gpu = context.GetGpu();
  Buffer source(vulkan.RuntimeContext(), scheduler, MemoryUsage::DeviceLocal, 0,
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, 256);
  Buffer readback(vulkan.RuntimeContext(), scheduler, MemoryUsage::Download, 0,
                  vk::BufferUsageFlagBits::eTransferDst, source.Size());
  alignas(8) uint64_t label = 0xdeadbeef00000000ull;
  uint32_t suffix = 0;

  for (uint32_t token = 1; token <= 3; ++token) {
    const auto pattern = 0x12340000u | token;
    VulkanHarness::OnGpuThread(context, [&] {
      suffix = 0;
      // No post-fill barrier here: RELEASE_MEM must provide the native memory dependency.
      scheduler.Current().Handle().fillBuffer(source.Handle(), 0, source.Size(), pattern);
    });
    const auto wait = WaitPacket(&label, token);
    std::vector<uint32_t> commands{
        0xc0064900u, 0x0000c514u, 0x23000000u, Low(&label), High(&label), token, 0u, 0u};
    commands.insert(commands.end(), wait.begin(), wait.end());
    commands.insert(commands.end(), {KYTY_PM4(5, Pm4::IT_WRITE_DATA, 0), 0u,
                                     Low(&suffix), High(&suffix), token});
    gpu.Submit(commands, {});
    // As in CheckCpSeqGuestGpu: two Done calls cover the latest frame with bounded Done too.
    // CTest gives this check a timeout so a lost completion cannot hang the suite indefinitely.
    gpu.Done();
    gpu.Done();
    VulkanHarness::OnGpuThread(context, [&] {
      Require(name, "ordered label and suffix", label == (0xdeadbeef00000000ull | token) &&
                  suffix == token && !gpu.HasDeferredLabels(),
              "release/wait lost the new token, overwrote the high word or left a pending label");
      auto native = scheduler.Current().Handle();
      const vk::BufferCopy copy{0, 0, source.Size()};
      native.copyBuffer(source.Handle(), readback.Handle(), 1, &copy);
      MakeHostReadable(scheduler, readback);
      scheduler.Finish();
    });
    readback.Invalidate(0, readback.Size());
    for (size_t offset = 0; offset < readback.Size(); offset += sizeof(uint32_t)) {
      uint32_t actual = 0;
      std::memcpy(&actual, readback.Mapped().data() + offset, sizeof(actual));
      Require(name, "GPU data after release", actual == pattern,
              "release/wait consumer read an earlier buffer pattern");
    }
  }
  context.ShutdownGpu();
  std::printf("[gpu]     %-32s ok (3 tokens, masked high word)\n", name);
}

} // namespace GuestSyncTests
