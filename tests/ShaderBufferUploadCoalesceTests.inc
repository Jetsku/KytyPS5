// Include inside VulkanHarness's public section. This check uses its private
// CreateHostBuffer helper and the real BufferCache/MemoryTracker upload path.
// Dispatch in a fresh VulkanHarness(false) process with KYTY_CP_RECORDER=0,
// KYTY_UPLOAD_DMA=0, KYTY_FAULT_AHEAD_KB=0, KYTY_HOT_PAGES=1,
// KYTY_HOT_PAGE_FRAMES=2, KYTY_HOT_PAGE_CHECK_LIMIT=0 and
// KYTY_HOT_PAGE_QUIET_FRAMES=1000. KYTY_UPLOAD_BATCH may be 0 or 1.
// Required friend adapter: BufferCacheTestAccess::SyncRead(cache, id, address,
// bytes) forwards to cache.SynchronizeBuffer(cache.GetBuffer(id), address,
// bytes, false, false); it must not implement upload collection itself.

class UploadCoalesceCapture {
public:
  struct Call {
    VkBuffer source = VK_NULL_HANDLE;
    VkBuffer destination = VK_NULL_HANDLE;
    std::vector<VkBufferCopy> regions;
  };

  UploadCoalesceCapture()
      : m_original(VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer) {
    Require("BufferUploadCoalesce", "copy capture", s_active == nullptr && m_original != nullptr,
            "nested capture or missing driver copy dispatcher");
    s_active = this;
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer = Copy;
  }
  ~UploadCoalesceCapture() {
    // The caller has completed every synchronous recording command before this
    // scope ends. The wrapper only observes recording, never GPU execution.
    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdCopyBuffer = m_original;
    s_active = nullptr;
  }
  UploadCoalesceCapture(const UploadCoalesceCapture &) = delete;
  UploadCoalesceCapture &operator=(const UploadCoalesceCapture &) = delete;

  // Arm/disarm and all wrapper writes run on the same guest GPU thread. The
  // caller reads calls only after SendCommandSync has returned.
  void Arm(vk::Buffer destination) {
    calls.clear();
    m_destination = static_cast<VkBuffer>(destination);
    m_armed = true;
  }
  void Disarm() { m_armed = false; }
  std::vector<Call> calls;

private:
  static VKAPI_ATTR void VKAPI_CALL Copy(VkCommandBuffer command, VkBuffer source,
      VkBuffer destination, uint32_t count, const VkBufferCopy *regions) {
    auto &capture = *s_active;
    if (capture.m_armed && destination == capture.m_destination) {
      Call call;
      call.source = source;
      call.destination = destination;
      if (count != 0) call.regions.assign(regions, regions + count);
      capture.calls.push_back(std::move(call));
    }
    // Forward the exact regions to the real dispatcher. No helper algorithm is
    // duplicated here, and the Vulkan driver still executes the upload.
    capture.m_original(command, source, destination, count, regions);
  }
  inline static UploadCoalesceCapture *s_active = nullptr;
  PFN_vkCmdCopyBuffer m_original;
  VkBuffer m_destination = VK_NULL_HANDLE;
  bool m_armed = false;
};

void CheckBufferUploadCoalesce() {
  constexpr const char *name = "BufferUploadCoalesce";
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  constexpr uintptr_t boundary = 0x000000020C400000ull;
  constexpr uintptr_t base = boundary - 0x10000;
  constexpr uint64_t allocation_size = 0x20000;
  constexpr uint64_t alignment = 0x10000;
  constexpr uintptr_t target = boundary - 0x8000;
  constexpr uint64_t span = 0x10000;
  constexpr uintptr_t normal = target + page;
  constexpr uintptr_t hot0 = normal + page;
  constexpr uintptr_t hot1 = hot0 + page;
  constexpr uintptr_t hot2 = hot1 + page;
  static_assert(boundary % TRACKER_REGION_SIZE == 0 &&
                target < boundary && target + span > boundary);
  static_assert(span > BufferCache::CACHING_PAGESIZE && page == 4096);

  const auto context_owner = MakeRenderContext();
  auto &context = *context_owner;
  auto &scheduler = context.GetCommandScheduler();
  auto &cache = context.GetBufferCache();
  auto &tracker = BufferCacheTestAccess::Tracker(cache);
  Require(name, "fault policy", tracker.GetFaultPolicy().ahead_pages == 1 &&
              tracker.GetFaultPolicy().hot_frames == 2 &&
              tracker.GetFaultPolicy().hot_max >= 3 &&
              BufferCacheTestAccess::HotCheckLimit(cache) == 0,
          "the dedicated coalesce fixture requires its documented fault-policy environment");
  HW::Context registers{};
  HW::UserConfig user_config{};
  HW::Shader shaders{};
  scheduler.Begin(registers, user_config, shaders);
  context.InitializeGpu(nullptr);

  int64_t direct_offset = -1;
  Require(name, "direct allocation", Libs::LibKernel::Memory::KernelAllocateDirectMemory(
              0, Libs::LibKernel::Memory::KernelGetDirectMemorySize(), allocation_size,
              alignment, 0, &direct_offset) == 0,
          "coalesce direct allocation failed");
  void *mapped = reinterpret_cast<void *>(base);
  Require(name, "direct mapping", Libs::LibKernel::Memory::KernelMapDirectMemory(
              &mapped, allocation_size, 0x3, 0x10, direct_offset, alignment) == 0 &&
              mapped == reinterpret_cast<void *>(base),
          "coalesce fixed direct mapping failed");
  auto *memory = static_cast<uint8_t *>(mapped);
  for (uint64_t index = 0; index < allocation_size; ++index)
    memory[index] = static_cast<uint8_t>((index * 19 + index / page * 37 + 11) & 0xffu);

  bool normal_coalesced = false;
  bool hot_coalesced = false;
  bool hole_preserved = false;
  bool origins_preserved = false;
  {
    OnGpuThread(context, [&] { context.MapMemory(base, allocation_size); });
    const auto id = OnGpuThread(context, [&] { return cache.FindBuffer(target, span); });
    auto &buffer = cache.GetBuffer(id);
    Require(name, "registered buffer", buffer.IsInBounds(target, span),
            "registered native buffer does not cover the coalesce fixture");

    // Install after device/context initialization so it cannot overwrite our
    // dispatcher. A dedicated process with the recorder disabled keeps the
    // original pointer and the one recording thread stable for this scope.
    UploadCoalesceCapture capture;
    const auto sync = [&](uint64_t address, uint64_t bytes, bool observe) {
      OnGpuThread(context, [&] {
        if (observe) capture.Arm(buffer.Handle());
        (void)BufferCacheTestAccess::SyncRead(cache, id, address, bytes);
        scheduler.Current().FlushBarriers();
        if (observe) capture.Disarm();
      });
    };
    const auto fault_write = [&](uintptr_t address, uint8_t value, uint64_t bytes) {
      OnGpuThread(context, [&] {
        Require(name, "real write fault", context.HandleFault(PageFaultAccess::Write, address),
                "tracked write did not route through RenderContext::HandleFault");
        std::memset(reinterpret_cast<void *>(address), value, bytes);
      });
    };
    const auto matches = [&](const char *stage) {
      auto readback = CreateHostBuffer(name, span, vk::BufferUsageFlagBits::eTransferDst, {0});
      OnGpuThread(context, [&] {
        const vk::BufferCopy copy{buffer.Offset(target), 0, span};
        scheduler.Current().Handle().copyBuffer(buffer.Handle(), readback.buffer, 1, &copy);
        vk::BufferMemoryBarrier barrier{};
        barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
        barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = readback.buffer;
        barrier.size = readback.size;
        scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &barrier, 0, nullptr);
        scheduler.Finish();
      });
      const auto words = ReadBuffer(name, readback, static_cast<size_t>(span / sizeof(u32)));
      DestroyBuffer(&readback);
      Require(name, stage, std::memcmp(words.data(), reinterpret_cast<const void *>(target), span) == 0,
              "real native upload corrupted or omitted a byte of the fixture");
    };
    const auto report = [&](const char *stage) {
      size_t regions = 0;
      uint64_t bytes = 0;
      for (const auto &call : capture.calls) {
        regions += call.regions.size();
        for (const auto &region : call.regions) bytes += region.size;
      }
      std::printf("[gpu]     %-32s %-24s calls=%zu regions=%zu bytes=%llu\n", name, stage,
                  capture.calls.size(), regions, static_cast<unsigned long long>(bytes));
    };

    // A same-region dirty run is already one tracker callback. Crossing the
    // 4-MiB boundary creates two real callbacks that can become one copy.
    sync(target, span, true);
    report("normal tracker boundary");
    matches("normal upload bytes");
    normal_coalesced = capture.calls.size() == 1 && capture.calls[0].regions.size() == 1 &&
        capture.calls[0].regions[0].dstOffset == buffer.Offset(target) &&
        capture.calls[0].regions[0].size == span;
    Require(name, "normal protection", !tracker.IsRegionCpuModified(target, span) &&
                tracker.HotPageCount() == 0,
            "normal upload left CPU-dirty or hot pages");
    fault_write(target + 0x100, 0x4b, 64);
    fault_write(boundary + 0x100, 0x9d, 64);
    sync(target, span, false);
    matches("writes after normal protection");

    // Promote through actual faults in two consecutive frames. Each upload
    // creates/updates production shadows; no private tracker bits are set.
    for (uint32_t frame = 0; frame < 2; ++frame) {
      fault_write(hot0, static_cast<uint8_t>(0x31 + frame), page);
      fault_write(hot1, static_cast<uint8_t>(0x51 + frame), page);
      fault_write(hot2, static_cast<uint8_t>(0x71 + frame), page);
      sync(target, span, false);
      OnGpuThread(context, [&] { cache.AdvanceFrame(); });
    }
    Require(name, "hot promotion", tracker.HotPageCount() == 3 &&
                tracker.IsRegionHot(hot0, page) && tracker.IsRegionHot(hot1, page) &&
                tracker.IsRegionHot(hot2, page) && !tracker.IsRegionHot(normal, page),
            "real fault promotion did not select exactly the three intended hot pages");

    const auto fault_epoch = tracker.FaultMutationEpoch();
    OnGpuThread(context, [&] {
      std::memset(reinterpret_cast<void *>(hot0), 0xa3, page);
      std::memset(reinterpret_cast<void *>(hot1), 0xc5, page);
      std::memset(reinterpret_cast<void *>(hot2), 0xe7, page);
    });
    sync(target, span, true);
    report("three hot snapshots");
    matches("adjacent hot upload bytes");
    hot_coalesced = capture.calls.size() == 1 && capture.calls[0].regions.size() == 1 &&
        capture.calls[0].regions[0].dstOffset == buffer.Offset(hot0) &&
        capture.calls[0].regions[0].size == 3 * page;
    Require(name, "hot epoch", tracker.FaultMutationEpoch() == fault_epoch &&
                tracker.HotPageCount() == 3,
            "fault-free hot writes or their upload changed the fault epoch/hot count");
    sync(target, span, true);
    Require(name, "unchanged hot upload", capture.calls.empty(),
            "unchanged production hot shadows still caused an upload");

    OnGpuThread(context, [&] {
      std::memset(reinterpret_cast<void *>(hot0), 0x29, page);
      std::memset(reinterpret_cast<void *>(hot2), 0x6d, page);
    });
    sync(target, span, true);
    report("unchanged hot hole");
    matches("hot gap byte preservation");
    hole_preserved = capture.calls.size() == 1 && capture.calls[0].regions.size() == 2 &&
        capture.calls[0].regions[0].dstOffset == buffer.Offset(hot0) &&
        capture.calls[0].regions[0].size == page &&
        capture.calls[0].regions[1].dstOffset == buffer.Offset(hot2) &&
        capture.calls[0].regions[1].size == page;
    OnGpuThread(context, [&] { std::memset(reinterpret_cast<void *>(hot1), 0x8f, page); });
    sync(target, span, true);
    matches("previous hot hole changes");
    Require(name, "only middle hot page", capture.calls.size() == 1 &&
                capture.calls[0].regions.size() == 1 &&
                capture.calls[0].regions[0].dstOffset == buffer.Offset(hot1) &&
                capture.calls[0].regions[0].size == page,
            "a changed middle hot page was missed or copied its unchanged neighbors");

    // The normal page's guest bytes and the two following host snapshots are
    // contiguous in BOTH the packed staging ring and destination. Their input
    // pointers differ, so the guest_copies/host_base boundary must survive.
    fault_write(normal, 0x17, page);
    OnGpuThread(context, [&] {
      std::memset(reinterpret_cast<void *>(hot0), 0xb9, page);
      std::memset(reinterpret_cast<void *>(hot1), 0xdb, page);
    });
    Require(name, "mixed source states", tracker.IsRegionCpuModified(normal, page) &&
                !tracker.IsRegionHot(normal, page) && tracker.IsRegionHot(hot0, 2 * page),
            "mixed upload did not contain normal guest bytes followed by hot snapshots");
    sync(target, span, true);
    report("guest then host boundary");
    matches("mixed guest/host source bytes");
    origins_preserved = capture.calls.size() == 1 && capture.calls[0].regions.size() == 2 &&
        capture.calls[0].regions[0].dstOffset == buffer.Offset(normal) &&
        capture.calls[0].regions[0].size == page &&
        capture.calls[0].regions[1].dstOffset == buffer.Offset(hot0) &&
        capture.calls[0].regions[1].size == 2 * page &&
        capture.calls[0].regions[0].srcOffset + page == capture.calls[0].regions[1].srcOffset;

    // Exercise the BDA full/logged/hot paths too. The full pass records a three-page hot run;
    // subsequent logged subranges overlap it. Fault-free writes must still reach native bytes
    // after compacting those runs, including the middle page previously left unchanged.
    OnGpuThread(context, [&] { context.PrepareBda(); });
    for (uint64_t changed = 0; changed < 3; ++changed) {
      OnGpuThread(context, [&] {
        tracker.MarkRegionAsCpuModified(hot0 + changed * page, page);
        std::memset(reinterpret_cast<void *>(hot0 + changed * page),
                    static_cast<int>(0x40 + changed), page);
        SyncEpoch::Advance();
        context.PrepareBda();
      });
      matches("BDA overlapping logged hot subrange");
    }
    OnGpuThread(context, [&] {
      std::memset(reinterpret_cast<void *>(hot1), 0xf2, page);
      SyncEpoch::Advance();
      context.PrepareBda();
    });
    matches("BDA fault-free hot write after compacting runs");

    // The observer ignores readback copies, but stays installed until recording
    // is finished. No compilation, DMA or replay worker uses this test gate.
    OnGpuThread(context, [&] {
      scheduler.Finish();
      context.UnmapMemory(base, allocation_size);
    });
  }
  context.ShutdownGpu();
  Require(name, "unmap direct backing",
          Libs::LibKernel::Memory::KernelMunmap(base, allocation_size) == 0,
          "coalesce direct mapping release failed");
  Require(name, "release direct backing",
          Libs::LibKernel::Memory::KernelReleaseDirectMemory(direct_offset, allocation_size) == 0,
          "coalesce direct allocation release failed");

  // Check every real byte case before the region-count expectations. On the old
  // implementation all cases execute and the diagnostic reports show 2/3/2/3
  // regions; this assertion then supplies the intended RED result.
  Require(name, "driver regions", normal_coalesced && hot_coalesced && hole_preserved && origins_preserved,
          "native copy regions were not coalesced, bridged a hot gap, or crossed guest/host origins");
  std::printf("[gpu]     %-32s ok\n", name);
  // Reuse the existing real GPU/CPU ownership regression with this minimal
  // VulkanHarness(false) device. It needs buffer transfers, not rasterization.
  // This exercises omission of unpublished GPU bytes during a CPU upload.
  CheckFalseSharingWrites();
}
