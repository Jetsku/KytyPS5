// Included inside VulkanHarness (ShaderRecompilerComputeTests.cpp). --bda-new-buffer-only.
//
// BDA synchronization of buffers registered between two passes. A shader reads guest memory
// through device addresses, so after RenderContext::PrepareBda every registered buffer must hold
// the guest bytes, whichever pass strategy runs (full scan, dirty log, hot ranges, or a pass
// limited to the buffers registered since the previous one). Two cases, both created through
// FindBuffer, which registers a buffer without the binding path's own synchronization, so only
// the BDA pass can upload them:
//  - a new buffer over guest memory the CPU wrote while no buffer covered it;
//  - a buffer grown over an existing one (the old one is joined and deleted) whose new pages
//    the CPU wrote, and whose old pages the CPU wrote again after the previous pass.
// Checks contents only (native bytes read back from the GPU), never pass counts, so it holds for
// any implementation.
// KYTY_TEST_BDA_SAME_EPOCH=1 omits the explicit epoch advances, covering registrations before
// epoch/submission skips. KYTY_TEST_BDA_REMAP=1 also remaps the existing new buffer's GPU range.
void CheckBdaNewBufferSync() {
  constexpr const char *name = "BdaNewBufferSync";
  constexpr uintptr_t base = 0x0000000207400000ull;
  constexpr uint64_t allocation_size = 0x100000;
  constexpr uint64_t allocation_alignment = 0x10000;
  constexpr uint64_t old_offset = 0x20000;   // the buffer present before the passes
  constexpr uint64_t old_size = 0x10000;
  constexpr uint64_t grown_offset = 0x28000; // overlaps the old buffer and extends past it
  constexpr uint64_t grown_size = 0x20000;
  constexpr uint64_t new_offset = 0x80000;   // no buffer covers it before the test creates one
  constexpr uint64_t new_size = 0x10000;
  constexpr uint64_t control_offset = 0xc0000; // created last, with no pass after it
  const bool same_epoch = std::getenv("KYTY_TEST_BDA_SAME_EPOCH") != nullptr;
  const bool remap = std::getenv("KYTY_TEST_BDA_REMAP") != nullptr;

  EnsureRuntimeContext();
  const auto context_owner = MakeRenderContext();
  auto &context = *context_owner;
  auto &scheduler = context.GetCommandScheduler();
  HW::Context registers{};
  HW::UserConfig user_config{};
  HW::Shader shaders{};
  scheduler.Begin(registers, user_config, shaders);
  context.InitializeGpu(nullptr);

  int64_t direct_offset = -1;
  Require(name, "direct allocation",
          Libs::LibKernel::Memory::KernelAllocateDirectMemory(
              0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size,
              allocation_alignment, 0, &direct_offset) == 0,
          "direct-memory allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "direct mapping",
          Libs::LibKernel::Memory::KernelMapDirectMemory(&mapped, allocation_size, 0x3, 0x10,
                                                         direct_offset, allocation_alignment) == 0 &&
              mapped == reinterpret_cast<void *>(base),
          "fixed direct-memory mapping failed");
  auto *memory = static_cast<uint8_t *>(mapped);
  for (uint64_t index = 0; index < allocation_size; index++) {
    memory[index] = static_cast<uint8_t>((index * 37 + index / 4096) & 0xffu);
  }

  {
    auto &cache = context.GetBufferCache();
    context.MapMemory(base, allocation_size);
    const auto read_native = [&](const Libs::Graphics::Buffer &buffer, uint64_t offset,
                                 uint64_t bytes) {
      auto readback = CreateHostBuffer(name, bytes, vk::BufferUsageFlagBits::eTransferDst, {0});
      const vk::BufferCopy copy{offset, 0, bytes};
      scheduler.Current().Handle().copyBuffer(buffer.Handle(), readback.buffer, 1, &copy);
      vk::BufferMemoryBarrier barrier{};
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
      barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.buffer = readback.buffer;
      barrier.size = readback.size;
      scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                   vk::PipelineStageFlagBits::eHost, {}, 0,
                                                   nullptr, 1, &barrier, 0, nullptr);
      scheduler.Finish();
      auto words = ReadBuffer(name, readback, static_cast<uint32_t>(bytes / 4));
      DestroyBuffer(&readback);
      return words;
    };
    // A guest CPU write as the game makes it: the tracker sees the fault first.
    const auto cpu_write = [&](uint64_t offset, uint32_t value) {
      Require(name, "write fault", context.HandleFault(PageFaultAccess::Write, base + offset),
              "a CPU write to a mapped page did not reach the tracker");
      std::memcpy(memory + offset, &value, sizeof(value));
    };
    // The whole buffer, compared with the guest bytes it covers.
    const auto first_difference_of = [&](BufferId id) {
      const auto &buffer = cache.GetBuffer(id);
      const auto bytes = buffer.Size();
      const auto words = read_native(buffer, 0, bytes);
      const auto *guest = memory + (buffer.CpuAddress() - base);
      size_t first_difference = bytes;
      for (size_t i = 0; i < bytes / 4; i++) {
        uint32_t expected = 0;
        std::memcpy(&expected, guest + i * 4, sizeof(expected));
        if (words[i] != expected) {
          first_difference = i * 4;
          break;
        }
      }
      return std::pair{first_difference, bytes};
    };
    const auto matches = [&](BufferId id, const char *stage) {
      const auto &buffer = cache.GetBuffer(id);
      const auto [first_difference, bytes] = first_difference_of(id);
      if (first_difference != bytes) {
        std::printf("%s: %s differs at guest 0x%016llx (buffer 0x%016llx + 0x%zx)\n", name, stage,
                    static_cast<unsigned long long>(buffer.CpuAddress() + first_difference),
                    static_cast<unsigned long long>(buffer.CpuAddress()), first_difference);
      }
      Require(name, stage, first_difference == bytes,
              "the BDA pass left a registered buffer without the guest bytes");
    };

    // Everything in one GPU-thread command: each command ends with an epoch advance, and the
    // cases below must not depend on one.
    BufferId new_id{};
    BufferId grown_id{};
    BufferId control_id{};
    OnGpuThread(context, [&] {
      const auto old_id = cache.FindBuffer(base + old_offset, old_size);
      context.PrepareBda(); // the first pass: the old buffer holds the guest bytes
      (void)old_id;
      const auto first_epoch = SyncEpoch::Current();
      const auto first_submission = SyncEpoch::CurrentSubmission();

      // A new buffer over memory the CPU wrote while no buffer covered it.
      cpu_write(new_offset + 0x100, 0x5eed0001u);
      cpu_write(new_offset + 0x9104, 0x5eed0002u);
      if (!same_epoch) SyncEpoch::Advance();
      new_id = cache.FindBuffer(base + new_offset, new_size);
      if (same_epoch) {
        Require(name, "new buffer without epoch/submission change",
                SyncEpoch::Current() == first_epoch &&
                    SyncEpoch::CurrentSubmission() == first_submission,
                "the fixture advanced an epoch/submission before the new buffer's BDA read");
      }
      context.PrepareBda();

      // A grown buffer: new pages the CPU wrote, and old pages written again after the pass.
      cpu_write(old_offset + 0x2008, 0x5eed0003u);
      cpu_write(grown_offset + 0x1c000, 0x5eed0004u);
      if (!same_epoch) SyncEpoch::Advance();
      grown_id = cache.FindBuffer(base + grown_offset, grown_size);
      if (same_epoch) {
        Require(name, "grown buffer without epoch/submission change",
                SyncEpoch::Current() == first_epoch &&
                    SyncEpoch::CurrentSubmission() == first_submission,
                "the fixture advanced an epoch/submission before the joined buffer's BDA read");
      }
      context.PrepareBda();

      if (remap) {
        // Keep the native buffer and the kernel backing. Only its renderer GPU mapping changes:
        // registration cannot rescue this case, so Map/Unmap must invalidate the BDA proof.
        const auto native_handle = cache.GetBuffer(new_id).Handle();
        context.UnmapMemory(base + new_offset, new_size);
        Require(name, "GPU range unmapped", !context.IsMapped(base + new_offset, new_size),
                "the remap fixture left the GPU range mapped");
        const uint32_t rewritten = 0x5eed00f0u;
        std::memcpy(memory + new_offset + 0x3000, &rewritten, sizeof(rewritten));
        context.MapMemory(base + new_offset, new_size);
        Require(name, "remap keeps the registered buffer",
                cache.FindBuffer(base + new_offset, new_size) == new_id &&
                    cache.GetBuffer(new_id).Handle() == native_handle,
                "remapping unexpectedly replaced the buffer: this case needs the mapping proof");
        context.PrepareBda();
      }

      if (!same_epoch) {
        // Two registered buffers the CPU writes after their passes: the next pass is a dirty-log
        // pass over both (KYTY_BDA_BATCH_PROTECT: their pages are collected, write-protected in
        // one call per region, then copied). Both must hold the bytes, and the written pages
        // must be write-protected again on the host, so a later write still faults.
        context.PrepareBda();
        cpu_write(new_offset + 0x4004, 0x5eed0007u);
        cpu_write(grown_offset + 0x6008, 0x5eed0008u);
        cpu_write(grown_offset + 0x7008, 0x5eed0009u);
        SyncEpoch::Advance();
        context.PrepareBda();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
        for (const uint64_t offset : {new_offset + 0x4004, grown_offset + 0x6008,
                                      grown_offset + 0x7008}) {
          MEMORY_BASIC_INFORMATION info{};
          Require(name, "written pages protected again",
                  VirtualQuery(memory + offset, &info, sizeof(info)) != 0 &&
                      (info.Protect & 0xffu) == PAGE_READONLY,
                  "a page the pass uploaded is still writable: its next write would be missed");
        }
#endif
      }

      // Control: the same creation with no pass after it. Its contents must differ from the
      // guest bytes, or the cases above would not depend on the BDA pass at all.
      control_id = cache.FindBuffer(base + control_offset, new_size);
    });
    Require(name, "grown buffer covers the old one",
            cache.GetBuffer(grown_id).CpuAddress() <= base + old_offset &&
                cache.GetBuffer(grown_id).CpuAddress() + cache.GetBuffer(grown_id).Size() >=
                    base + grown_offset + grown_size,
            "the overlapping request did not join the old buffer into one grown buffer");
    const auto [control_difference, control_bytes] = first_difference_of(control_id);
    Require(name, "control depends on the pass", control_difference != control_bytes,
            "a buffer created without a BDA pass already held the guest bytes: this check cannot "
            "tell whether the pass synchronizes new buffers");
    matches(new_id, "new buffer contents");
    matches(grown_id, "grown buffer contents");
    scheduler.Finish();
    context.UnmapMemory(base, allocation_size);
  }

  Require(name, "unmap direct backing",
          Libs::LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
          "direct mapping release failed");
  Require(name, "release direct backing",
          Libs::LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
          "direct-memory allocation release failed");
  std::printf("[host]    %-32s ok\n", name);
}
