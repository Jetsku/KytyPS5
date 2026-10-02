#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/eagerReadbackPages.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/writeTickMap.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#include <immintrin.h>
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::EagerReadbackPages;
using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::WriteTickMap;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  g_protection_calls++;
  g_protection_log.push_back({vaddr, size, mode});
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *AllocateFixedGuestRange(uint64_t size, uintptr_t offset) {
  // Keep the mapping inside the tracker's guest address space and preserve the
  // requested region alignment. A fixed address can be occupied by the host
  // process (notably by the macOS runner's ASLR layout).
  constexpr uintptr_t first_base = 0x0000000200000000ull;
  constexpr uintptr_t stride = 0x0000000100000000ull;
  for (uintptr_t attempt = 0; attempt < 256; attempt++) {
    auto *wanted = reinterpret_cast<void *>(first_base + offset + attempt * stride);
    auto *memory = static_cast<uint8_t *>(
        VirtualAlloc(wanted, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (memory == wanted) {
      return memory;
    }
    if (memory != nullptr) {
      Check(VirtualFree(memory, 0, MEM_RELEASE) != 0,
            "release unexpected fixed allocation failed");
    }
  }
  Check(false, "no free fixed guest address found");
  return nullptr;
}

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  const auto size = manager.GetPageSize() * pages;
  return AllocateFixedGuestRange(size, 0x10000);
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr GuestRange last_byte{TRACKER_ADDRESS_SIZE - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == TRACKER_ADDRESS_SIZE);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestRangeInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = AllocateFixedGuestRange(size, 0x1000000);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestWriteTickMap() {
  WriteTickMap ticks;
  Check(ticks.MaxTick(0x1000, 0x1000) == 0, "empty write-tick map reported a writer");
  ticks.Assign(0x1000, 0x4000, 5);
  ticks.Assign(0x2000, 0x1000, 7);
  Check(ticks.MaxTick(0x1000, 0x1000) == 5 && ticks.MaxTick(0x2000, 0x10) == 7 &&
            ticks.MaxTick(0x3000, 0x2000) == 5 && ticks.MaxTick(0x1000, 0x4000) == 7 &&
            ticks.MaxTick(0x5000, 0x1000) == 0 && ticks.Size() == 3,
        "write-tick map did not split an overwritten interval");
  ticks.Assign(0x1000, 0x4000, 9);
  Check(ticks.Size() == 1 && ticks.MaxTick(0x2000, 1) == 9,
        "write-tick map did not replace covered intervals");
  ticks.Assign(0x5000, 0x1000, 9);
  Check(ticks.Size() == 1 && ticks.MaxTick(0x5000, 1) == 9,
        "write-tick map did not coalesce equal-tick neighbours");
  ticks.Assign(0x8000, 0x1000, 3);
  ticks.Prune(4);
  Check(ticks.Size() == 1 && ticks.MaxTick(0x8000, 1) == 0 && ticks.MaxTick(0x1000, 1) == 9,
        "write-tick prune removed the wrong entries");
  // Covered by an entry of the same tick (the early return): nothing changes.
  ticks.Assign(0x2000, 0x800, 9);
  ticks.Assign(0x1000, 0x5000, 9);
  Check(ticks.Size() == 1 && ticks.MaxTick(0x1000, 0x5000) == 9 && ticks.MaxTick(0x6000, 1) == 0,
        "a covered same-tick assignment changed the map");
  ticks.Assign(0x2000, 0x800, 10);
  Check(ticks.Size() == 3 && ticks.MaxTick(0x2000, 1) == 10 && ticks.MaxTick(0x1fff, 1) == 9 &&
            ticks.MaxTick(0x2800, 1) == 9,
        "a covered newer-tick assignment did not split its entry");
}

uint64_t NextRandom(uint64_t &state) {
  state ^= state << 13u;
  state ^= state >> 7u;
  state ^= state << 17u;
  return state;
}

// RangeSet against a byte map: random adds and subtractions keep exactly the covered bytes, as
// ascending, disjoint and non-touching ranges (the form Add's early return relies on).
void TestRangeSetModel() {
  constexpr uint64_t span = 512;
  constexpr uint64_t origin = 0x1000;
  RangeSet ranges;
  std::vector<bool> covered(span, false);
  uint64_t state = 0x9e3779b97f4a7c15ull;
  std::vector<std::pair<uint64_t, uint64_t>> listed;
  std::vector<std::pair<uint64_t, uint64_t>> runs;
  for (int step = 0; step < 20000; step++) {
    const uint64_t begin = NextRandom(state) % span;
    const uint64_t size = 1 + NextRandom(state) % std::min<uint64_t>(48, span - begin);
    const bool add = NextRandom(state) % 3 != 0;
    if (add) {
      ranges.Add(origin + begin, size);
    } else {
      ranges.Subtract(origin + begin, size);
    }
    for (uint64_t unit = begin; unit < begin + size; unit++) {
      covered[unit] = add;
    }
    listed.clear();
    ranges.ForEach([&](uint64_t first, uint64_t last) { listed.emplace_back(first, last); });
    runs.clear();
    for (uint64_t unit = 0; unit < span;) {
      if (!covered[unit]) {
        unit++;
        continue;
      }
      uint64_t last = unit;
      while (last < span && covered[last]) {
        last++;
      }
      runs.emplace_back(origin + unit, origin + last);
      unit = last;
    }
    Check(listed == runs, "range set diverged from its byte model");
    const uint64_t query = NextRandom(state) % span;
    const uint64_t query_size = 1 + NextRandom(state) % std::min<uint64_t>(64, span - query);
    bool all = true;
    for (uint64_t unit = query; unit < query + query_size; unit++) {
      all = all && covered[unit];
    }
    Check(ranges.Contains(origin + query, query_size) == all,
          "range set containment diverged from its byte model");
  }
}

// WriteTickMap against a per-byte tick model: random assignments and prunes keep every byte's
// newest tick, one entry per maximal run of an equal tick (the coalesced form Assign's early
// return relies on).
void TestWriteTickMapModel() {
  constexpr uint64_t span = 256;
  constexpr uint64_t origin = 0x10000;
  WriteTickMap ticks;
  std::vector<uint64_t> model(span, 0);
  uint64_t state = 0x2545f4914f6cdd1dull;
  uint64_t tick = 1;
  for (int step = 0; step < 8000; step++) {
    const auto op = NextRandom(state) % 16;
    if (op == 0) {
      const uint64_t completed = tick > 3 ? tick - 1 - NextRandom(state) % 3 : 0;
      ticks.Prune(completed);
      for (auto &value : model) {
        if (value <= completed) {
          value = 0;
        }
      }
    } else if (op < 5) {
      tick++;
    } else {
      const uint64_t begin = NextRandom(state) % span;
      const uint64_t size = 1 + NextRandom(state) % std::min<uint64_t>(40, span - begin);
      ticks.Assign(origin + begin, size, tick);
      for (uint64_t unit = begin; unit < begin + size; unit++) {
        model[unit] = tick;
      }
    }
    size_t runs = 0;
    for (uint64_t unit = 0; unit < span; unit++) {
      if (model[unit] != 0 && (unit == 0 || model[unit - 1] != model[unit])) {
        runs++;
      }
      Check(ticks.MaxTick(origin + unit, 1) == model[unit],
            "write-tick map diverged from its per-byte model");
    }
    Check(ticks.Size() == runs, "write-tick map was not one entry per equal-tick run");
    const uint64_t query = NextRandom(state) % span;
    const uint64_t query_size = 1 + NextRandom(state) % std::min<uint64_t>(64, span - query);
    uint64_t newest = 0;
    for (uint64_t unit = query; unit < query + query_size; unit++) {
      newest = std::max(newest, model[unit]);
    }
    Check(ticks.MaxTick(origin + query, query_size) == newest,
          "write-tick range query diverged from its per-byte model");
  }
}

void TestEagerReadbackPages() {
  using Result = EagerReadbackPages::IssueResult;
  constexpr uint64_t page = TRACKER_PAGE_SIZE;
  EagerReadbackPages::Limits limits;
  limits.capacity = 3;
  limits.idle_frames = 10;
  limits.frame_budget = 2;
  EagerReadbackPages pages(limits);

  // Writes of pages no reader needed are ignored.
  Check(!pages.NoteWrite(0x10000, page) && pages.Candidates() == 0,
        "eager: a write of a cold page became a candidate");
  pages.NoteRead(0x10000, 1, false);
  pages.NoteRead(0x20000, 1, true);
  Check(pages.Size() == 2 && pages.IsHot(0x10000) && pages.IsHot(0x20000),
        "eager: readbacks did not make their pages hot");

  // A write covering part of a hot page makes it a candidate; only GPU-thread-read pages ask
  // for an early submission.
  Check(!pages.NoteWrite(0x10000 + 0x800, 8) && pages.IsCandidate(0x10000) &&
            !pages.IsCandidate(0x20000) && pages.Candidates() == 1,
        "eager: a partial write did not mark exactly its hot page");
  Check(pages.NoteWrite(0x1f000, 2 * page) && pages.IsCandidate(0x20000) &&
            pages.Candidates() == 2,
        "eager: a write of a GPU-thread-read page did not request an early submission");
  // A second write of a candidate does not count twice.
  (void)pages.NoteWrite(0x10000, page);
  Check(pages.Candidates() == 2, "eager: a repeated write counted a candidate twice");

  // Retry keeps a candidate, Drop and Issued clear it.
  std::vector<uint64_t> offered;
  pages.IssueCandidates(1, [&](uint64_t address) {
    offered.push_back(address);
    return address == 0x10000 ? Result::Retry : Result::Drop;
  });
  Check(offered.size() == 2 && pages.IsCandidate(0x10000) &&
            !pages.IsCandidate(0x20000) && pages.Candidates() == 1,
        "eager: issue results did not update the candidates");

  // At most frame_budget copies per page and frame; a spent budget keeps the candidate for the
  // next frame.
  uint32_t issued = 0;
  for (int round = 0; round < 3; round++) {
    (void)pages.NoteWrite(0x10000, 4);
    pages.IssueCandidates(2, [&](uint64_t) {
      issued++;
      return Result::Issued;
    });
  }
  Check(issued == 2 && pages.IsCandidate(0x10000),
        "eager: the per-frame copy budget was not applied");
  pages.IssueCandidates(3, [&](uint64_t) {
    issued++;
    return Result::Issued;
  });
  Check(issued == 3 && !pages.IsCandidate(0x10000) && pages.Candidates() == 0,
        "eager: a candidate held back by the budget was not issued next frame");

  // Capacity: the least recently read page is evicted (with its candidate).
  pages.NoteRead(0x30000, 4, false);
  (void)pages.NoteWrite(0x20000, 4);
  pages.NoteRead(0x10000, 5, false);
  pages.NoteRead(0x40000, 5, false);
  Check(pages.Size() == 3 && !pages.IsHot(0x20000) && pages.IsHot(0x10000) &&
            pages.IsHot(0x30000) && pages.IsHot(0x40000) && pages.Candidates() == 0,
        "eager: capacity eviction did not drop the least recently read page");

  // Pages no reader needed for more than idle_frames expire (0x30000 was read in frame 4, the
  // others in frame 5).
  pages.Sweep(14);
  Check(pages.Size() == 3, "eager: sweep expired pages read within idle_frames");
  (void)pages.NoteWrite(0x30000, 4);
  pages.Sweep(15);
  Check(pages.Size() == 2 && !pages.IsHot(0x30000) && pages.IsHot(0x10000) &&
            pages.IsHot(0x40000) && pages.Candidates() == 0,
        "eager: sweep did not expire exactly the idle page and its candidate");

  // Frame counters wrap.
  EagerReadbackPages wrap(limits);
  wrap.NoteRead(0x50000, UINT32_MAX - 2, false);
  wrap.Sweep(3);
  Check(wrap.IsHot(0x50000), "eager: sweep mishandled a wrapped frame counter");
  wrap.Sweep(20);
  Check(!wrap.IsHot(0x50000), "eager: a wrapped idle page did not expire");

  // Capacity 0 disables hot pages.
  EagerReadbackPages none(EagerReadbackPages::Limits{0, 10, 2});
  none.NoteRead(0x10000, 1, true);
  Check(none.Empty() && !none.NoteWrite(0x10000, 4),
        "eager: capacity 0 kept a hot page");
}

void TestReadbackPendingUnmark() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  auto *second = reinterpret_cast<void *>(address + page_size);

  // Two GPU-dirty pages handed to one side readback.
  tracker.ForEachUploadRange(
      address, page_size * 2, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkReadbackPending(address, page_size * 3);
  auto result = tracker.UnmarkReadbackPending(address, page_size * 3);
  Check(result.unmarked_pages == 2 && result.retained_pages == 0 &&
            !tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory) == PAGE_READONLY && Protection(second) == PAGE_READONLY,
        "readback completion did not release its pending pages");

  // A newer writer re-dirties the second page while the copy is in flight.
  tracker.ForEachUploadRange(
      address, page_size * 2, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkReadbackPending(address, page_size * 2);
  tracker.ForEachUploadRange(
      address + page_size, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  result = tracker.UnmarkReadbackPending(address, page_size * 2);
  Check(result.unmarked_pages == 1 && result.retained_pages == 1 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_READONLY && Protection(second) == PAGE_NOACCESS,
        "readback completion released a page a newer writer re-owned");

  // An explicit download/unmark in between cancels the pending mark.
  tracker.MarkReadbackPending(address + page_size, page_size);
  tracker.UnmarkRegionAsGpuModified(address + page_size, page_size);
  tracker.ForEachUploadRange(
      address + page_size, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  result = tracker.UnmarkReadbackPending(address, page_size * 2);
  Check(result.unmarked_pages == 0 && result.retained_pages == 1 &&
            tracker.IsRegionGpuModified(address + page_size, page_size),
        "readback pending mark survived a GPU ownership transition");
  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  tracker.MarkRegionAsCpuModified(address, page_size * 3);
  tracker.UntrackMemory(address, page_size * 3);
  Release(memory);
}

// MemoryTracker::IsRangeGpuOwned (KYTY_WRITTEN_SYNC_SKIP): every page GPU-dirty, none
// readback-pending, no region missing; a written upload of such a range collects nothing and
// leaves its signature alone.
void TestRangeGpuOwned() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(!tracker.IsRangeGpuOwned(address, page_size) &&
            tracker.RangeSignature(address, page_size) == 0,
        "a range without a tracker region was owned (or the query created the region)");
  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(!tracker.IsRangeGpuOwned(address, page_size), "a clean range was owned");
  const auto written = [&](uint64_t offset, uint64_t size) {
    uint64_t collected = 0;
    tracker.ForEachUploadRange(
        address + offset, size, true,
        [&](uint64_t, uint64_t bytes) noexcept { collected += bytes; }, []() noexcept {});
    return collected;
  };
  Check(written(page_size, page_size * 2) == 0, "a clean range had pages to upload");
  Check(tracker.IsRangeGpuOwned(address + page_size, page_size * 2) &&
            tracker.IsRangeGpuOwned(address + page_size + 16, 32) &&
            !tracker.IsRangeGpuOwned(address, page_size * 2) &&
            !tracker.IsRangeGpuOwned(address + page_size * 2, page_size * 2),
        "ownership did not require every page of the range");
  const auto signature = tracker.RangeSignature(address, page_size * 4);
  Check(written(page_size, page_size * 2) == 0 &&
            tracker.RangeSignature(address, page_size * 4) == signature,
        "a written upload of an owned range collected pages or changed tracker state");
  // A readback mark ends ownership until a newer writer takes the page again.
  tracker.MarkReadbackPending(address + page_size * 2, page_size);
  Check(!tracker.IsRangeGpuOwned(address + page_size, page_size * 2) &&
            tracker.IsRangeGpuOwned(address + page_size, page_size),
        "a readback-pending page was owned");
  (void)written(page_size * 2, page_size);
  Check(tracker.IsRangeGpuOwned(address + page_size, page_size * 2),
        "a page a newer writer took again was not owned");
  // A download ends ownership.
  tracker.UnmarkRegionAsGpuModified(address + page_size, page_size);
  Check(!tracker.IsRangeGpuOwned(address + page_size, page_size * 2) &&
            tracker.IsRangeGpuOwned(address + page_size * 2, page_size),
        "a downloaded page was owned");
  tracker.UnmarkRegionAsGpuModified(address, page_size * 4);
  tracker.MarkRegionAsCpuModified(address, page_size * 4);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

struct PolicyHarness {
  explicit PolicyHarness(MemoryTracker::FaultPolicy policy,
                         bool track_cpu_mutations = false)
      : tracker(page_manager, track_cpu_mutations, policy) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

void UploadAll(MemoryTracker &tracker, uint64_t address, uint64_t size) {
  tracker.ForEachUploadRange(
      address, size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
}

// Write fault as RenderContext::HandleFault reports it (one byte).
uint32_t WriteFault(MemoryTracker &tracker, uint64_t address,
                    std::vector<std::pair<uint64_t, uint64_t>> *ahead = nullptr) {
  uint32_t flushes = 0;
  tracker.InvalidateRegionOnWriteFault(
      address, 1, [&] { flushes++; },
      [&](uint64_t run, uint64_t bytes) noexcept {
        Check(!IsWritable(reinterpret_cast<const void *>(run)),
              "fault-ahead run was reported after it became writable");
        if (ahead != nullptr) {
          ahead->push_back({run, bytes});
        }
      });
  return flushes;
}

// Hot-aware read upload: returns {normal pages, hot pages} reported.
std::pair<uint64_t, uint64_t> UploadHotAware(MemoryTracker &tracker,
                                             uint64_t address, uint64_t size) {
  uint64_t normal = 0;
  uint64_t hot = 0;
  tracker.ForEachUploadRange(
      address, size, false,
      [&](uint64_t, uint64_t bytes, bool is_hot) noexcept {
        (is_hot ? hot : normal) += bytes / 4096;
      },
      []() noexcept {});
  return {normal, hot};
}

void TestFaultAheadWindow() {
  MemoryTracker::FaultPolicy policy;
  policy.ahead_pages = 8;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadAll(tracker, address, page_size * 16);
  // Page 5 becomes GPU-owned: fault-ahead must leave it alone.
  tracker.ForEachUploadRange(
      address + page_size * 5, page_size, true,
      [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(!IsWritable(memory) && Protection(memory + page_size * 5) == PAGE_NOACCESS,
        "fault-ahead setup did not protect the range");

  ResetProtectionLog();
  std::vector<std::pair<uint64_t, uint64_t>> ahead_runs;
  Check(WriteFault(tracker, address + page_size * 2 + 16, &ahead_runs) == 0,
        "clean fault-ahead write fault requested a GPU flush");
  Check(ahead_runs.size() == 3 && ahead_runs[0].first == address &&
            ahead_runs[0].second == page_size * 2 &&
            ahead_runs[1].first == address + page_size * 3 &&
            ahead_runs[1].second == page_size * 2 &&
            ahead_runs[2].first == address + page_size * 6 &&
            ahead_runs[2].second == page_size * 2,
        "fault-ahead did not report exactly the runs it opened");
  for (uint64_t page = 0; page < 16; page++) {
    const bool in_window = page < 8 && page != 5;
    Check(tracker.IsRegionCpuModified(address + page * page_size, page_size) ==
                  in_window &&
              IsWritable(memory + page * page_size) == in_window,
          "fault-ahead window state is wrong");
  }
  Check(tracker.IsRegionGpuModified(address + page_size * 5, page_size) &&
            Protection(memory + page_size * 5) == PAGE_NOACCESS &&
            g_protection_calls <= 2,
        "fault-ahead disturbed a GPU-owned page or used too many protection calls");

  // A write fault on the GPU-owned page takes the flush path, without fault-ahead.
  UploadAll(tracker, address, page_size * 16);
  uint32_t flushes = 0;
  tracker.InvalidateRegionOnWriteFault(
      address + page_size * 5, 1,
      [&] {
        flushes++;
        tracker.ForEachDownloadRange<true>(address + page_size * 5, page_size,
                                           [](uint64_t, uint64_t) noexcept {});
        tracker.MarkRegionAsCpuModified(address + page_size * 5, 1);
      },
      [](uint64_t, uint64_t) noexcept {
        Check(false, "GPU-owned write fault reported fault-ahead pages");
      });
  Check(flushes == 1 && IsWritable(memory + page_size * 5) &&
            !IsWritable(memory + page_size * 4) &&
            !tracker.IsRegionCpuModified(address + page_size * 4, page_size),
        "GPU-owned write fault used fault-ahead");

  tracker.UntrackMemory(address, page_size * 16);
  Release(memory);
}

// The lock-free dirty mirrors are republished only for the pages a transition changes
// (RegionManager::Publish*Mirror): transitions spanning mirror words (64 pages each) and a
// 128-page fault-ahead window keep every word equal to the locked bits. The allocation starts 16
// pages into its region, so its pages 48 and 112 begin new words.
void TestMirrorsAcrossWords() {
  MemoryTracker::FaultPolicy policy;
  policy.ahead_pages = 128;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  constexpr uint64_t pages = 160;
  auto *memory = Allocate(harness.page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto mirrors_agree = [&](const char *stage) {
    for (uint64_t page = 0; page < pages; page++) {
      const auto start = address + page * page_size;
      MemoryTracker::DirtyState relaxed;
      if (!tracker.QueryDirtyRelaxed(start, page_size, relaxed) ||
          relaxed.cpu != tracker.IsRegionCpuModified(start, page_size) ||
          relaxed.gpu != tracker.IsRegionGpuModified(start, page_size)) {
        std::fprintf(stderr, "mirror differs at page %llu after %s\n",
                     static_cast<unsigned long long>(page), stage);
        Check(false, "a lock-free dirty mirror differs from the locked bits");
      }
    }
  };
  UploadAll(tracker, address, page_size * pages);
  mirrors_agree("the upload");
  // Region page 66: the fault-ahead window is region pages [0, 128), two mirror words.
  Check(WriteFault(tracker, address + page_size * 50) == 0, "a clean write fault flushed");
  Check(tracker.IsRegionCpuModified(address, page_size) &&
            tracker.IsRegionCpuModified(address + page_size * 111, page_size) &&
            !tracker.IsRegionCpuModified(address + page_size * 112, page_size),
        "the fault-ahead window is not region pages [0, 128)");
  mirrors_agree("a fault-ahead window over two words");
  UploadAll(tracker, address + page_size * 40, page_size * 20);
  mirrors_agree("an upload across a word boundary");
  UploadAll(tracker, address, page_size * pages); // GPU ownership needs clean pages
  tracker.MarkRegionAsGpuModified(address + page_size * 44, page_size * 80);
  mirrors_agree("a GPU mark across two word boundaries");
  tracker.MarkReadbackPending(address + page_size * 100, page_size * 20);
  (void)tracker.UnmarkReadbackPending(address + page_size * 100, page_size * 20);
  mirrors_agree("a completed readback across a word boundary");
  tracker.UnmarkRegionAsGpuModified(address + page_size * 44, page_size * 80);
  mirrors_agree("a GPU unmark across two word boundaries");
  tracker.MarkRegionAsCpuModified(address + page_size * 60, page_size * 60);
  mirrors_agree("a CPU mark across two word boundaries");
  tracker.UntrackMemory(address, page_size * pages);
  Release(memory);
}

void TestHotPagePromotionAndUpload() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 2;
  policy.hot_max = 8;
  PolicyHarness harness(policy, true);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadAll(tracker, address, page_size * 4);
  // Frame 1: fault, upload. Two faults in one frame count once.
  WriteFault(tracker, address + 8);
  UploadAll(tracker, address, page_size * 4);
  WriteFault(tracker, address + 8);
  UploadAll(tracker, address, page_size * 4);
  Check(tracker.HotPageCount() == 0, "one frame of faults promoted a page");
  // Frame 3 (a skipped frame) restarts the streak.
  tracker.AdvanceFrame();
  tracker.AdvanceFrame();
  WriteFault(tracker, address + 8);
  UploadAll(tracker, address, page_size * 4);
  Check(tracker.HotPageCount() == 0, "non-consecutive frames promoted a page");
  tracker.AdvanceFrame();
  const auto epoch_before = tracker.CpuMutationEpoch();
  WriteFault(tracker, address + 8);
  Check(tracker.HotPageCount() == 1 && tracker.IsRegionHot(address, page_size) &&
            !tracker.IsRegionHot(address + page_size, page_size * 3) &&
            epoch_before != UINT64_MAX && tracker.CpuMutationEpoch() == UINT64_MAX,
        "consecutive faulting frames did not promote exactly the page");

  // Hot-aware read uploads report the hot page and keep it dirty and writable.
  const auto [normal, hot] = UploadHotAware(tracker, address, page_size * 4);
  Check(normal == 0 && hot == 1 && tracker.IsRegionCpuModified(address, page_size) &&
            IsWritable(memory),
        "hot page did not stay CPU-dirty and writable across its upload");
  memory[8] = 0x5a; // no fault: the page is writable
  const auto [normal2, hot2] = UploadHotAware(tracker, address, page_size * 4);
  Check(normal2 == 0 && hot2 == 1, "hot page was not reported on every upload");

  // A caller that does not understand hot pages returns them to normal tracking.
  UploadAll(tracker, address, page_size);
  Check(tracker.HotPageCount() == 0 && !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory) && tracker.CpuMutationEpoch() != UINT64_MAX,
        "hot-unaware upload did not demote and protect the page");

  // Promote again (two consecutive frames), then a GPU writer takes it.
  tracker.AdvanceFrame();
  WriteFault(tracker, address + 8);
  UploadAll(tracker, address, page_size);
  tracker.AdvanceFrame();
  WriteFault(tracker, address + 8);
  Check(tracker.HotPageCount() == 1, "page was not promoted again");
  uint64_t written_ranges = 0;
  tracker.ForEachUploadRange(
      address, page_size, true,
      [&](uint64_t, uint64_t, bool is_hot) noexcept {
        Check(!is_hot, "written upload reported a hot range");
        written_ranges++;
      },
      []() noexcept {});
  Check(written_ranges == 1 && tracker.HotPageCount() == 0 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU writer did not take a hot page back to GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestHotPageDemotionPaths() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 1;
  policy.hot_max = 1;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadAll(tracker, address, page_size * 4);
  WriteFault(tracker, address);
  WriteFault(tracker, address + page_size);
  Check(tracker.HotPageCount() == 1 && tracker.IsRegionHot(address, page_size) &&
            !tracker.IsRegionHot(address + page_size, page_size) &&
            tracker.IsRegionCpuModified(address + page_size, page_size),
        "hot page budget was exceeded");

  // Explicit demotion keeps the page dirty until its next upload protects it.
  tracker.DemoteHotPages(address, page_size * 4);
  Check(tracker.HotPageCount() == 0 && tracker.IsRegionCpuModified(address, page_size) &&
            IsWritable(memory),
        "demotion lost the CPU-dirty state");
  const auto [normal, hot] = UploadHotAware(tracker, address, page_size * 4);
  Check(hot == 0 && normal == 2 && !IsWritable(memory) && !IsWritable(memory + page_size),
        "demoted page was not uploaded and protected normally");

  // Idle sweep: a hot page no upload visits returns to normal tracking.
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  Check(tracker.HotPageCount() == 1, "page was not promoted for the sweep");
  tracker.SweepHotPages(3);
  Check(tracker.HotPageCount() == 1, "sweep demoted a recently used hot page");
  for (int frame = 0; frame < 5; frame++) {
    tracker.AdvanceFrame();
  }
  tracker.SweepHotPages(3);
  Check(tracker.HotPageCount() == 0 && tracker.IsRegionCpuModified(address, page_size),
        "sweep did not demote an idle hot page");

  // Untracking demotes as well.
  UploadAll(tracker, address, page_size * 4);
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  Check(tracker.HotPageCount() == 1, "page was not promoted for untracking");
  tracker.UntrackMemory(address, page_size * 4);
  Check(tracker.HotPageCount() == 0 && tracker.IsRegionCpuModified(address, page_size * 4),
        "untracking kept a hot page");
  Release(memory);
}

// The BDA hot pass (BufferCache::SynchronizeBdaBuffers, KYTY_BDA_HOT_SYNC) skips every page but
// the hot ones while FaultMutationEpoch() holds: it must stay usable while hot pages exist, move on
// every transition that leaves a page CPU-dirty and writable outside the hot set, and stay put
// while hot pages are only written and uploaded.
void TestFaultMutationEpochWithHotPages() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 1;
  policy.hot_max = 8;
  PolicyHarness harness(policy, true);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadAll(tracker, address, page_size * 4);
  const auto clean = tracker.FaultMutationEpoch();
  Check(clean != UINT64_MAX && clean == tracker.CpuMutationEpoch(),
        "fault epoch differs from the CPU mutation epoch without hot pages");

  // The fault that promotes a page moves the epoch; the hot page then saturates only the old
  // token.
  WriteFault(tracker, address);
  const auto promoted = tracker.FaultMutationEpoch();
  Check(tracker.HotPageCount() == 1 && promoted != clean && promoted != UINT64_MAX &&
            tracker.CpuMutationEpoch() == UINT64_MAX,
        "promotion did not move the fault epoch, or hot pages saturated it");

  // Hot-aware uploads and fault-free writes of the hot page leave it alone.
  const auto [normal, hot] = UploadHotAware(tracker, address, page_size * 4);
  memory[16] = 0x3c;
  const auto [normal2, hot2] = UploadHotAware(tracker, address, page_size * 4);
  Check(normal == 0 && hot == 1 && normal2 == 0 && hot2 == 1 &&
            tracker.FaultMutationEpoch() == promoted,
        "hot-page uploads or writes moved the fault epoch");

  // Demotion leaves the page CPU-dirty and writable outside the hot set: the epoch moves.
  tracker.DemoteHotPages(address, page_size * 4);
  const auto demoted = tracker.FaultMutationEpoch();
  Check(tracker.HotPageCount() == 0 && demoted != promoted && IsWritable(memory) &&
            tracker.IsRegionCpuModified(address, page_size),
        "demotion did not move the fault epoch");
  // Demoting a range without hot pages changes nothing.
  UploadAll(tracker, address, page_size * 4);
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  const auto repromoted = tracker.FaultMutationEpoch();
  tracker.DemoteHotPages(address + page_size, page_size * 3);
  Check(tracker.HotPageCount() == 1 && tracker.FaultMutationEpoch() == repromoted,
        "demoting a range without hot pages moved the fault epoch");

  // The idle sweep demotes as well.
  UploadHotAware(tracker, address, page_size * 4);
  for (int frame = 0; frame < 5; frame++) {
    tracker.AdvanceFrame();
  }
  tracker.SweepHotPages(3);
  Check(tracker.HotPageCount() == 0 && tracker.FaultMutationEpoch() != repromoted &&
            tracker.IsRegionCpuModified(address, page_size),
        "idle sweep did not move the fault epoch");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// KYTY_BDA_DIRTY_LOG (BufferCache::SynchronizeBdaBuffersNow): every transition FaultMutationEpoch()
// covers also records the range it can make CPU-dirty, and a take returns the ranges with the
// epoch they account for.
void TestDirtiedLog() {
  MemoryTracker::FaultPolicy policy;
  policy.ahead_pages = 4;
  PolicyHarness harness(policy, true);
  auto &tracker = harness.tracker;
  tracker.EnableDirtiedLog();
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);
  RangeSet ranges;
  uint64_t epoch = 0;

  // The region is created (entirely CPU-dirty) by the first upload.
  UploadAll(tracker, address, page_size * 16);
  Check(tracker.TakeDirtiedRanges(ranges, epoch) && epoch == tracker.FaultMutationEpoch() &&
            ranges.Contains(address, page_size * 16),
        "a new region was not logged with the epoch it produced");
  Check(tracker.TakeDirtiedRanges(ranges, epoch) && ranges.Empty() &&
            epoch == tracker.FaultMutationEpoch(),
        "a take without transitions returned ranges or another epoch");

  // A write fault logs its fault-ahead window (pages 4-7 around page 5), nothing else.
  const auto before = tracker.FaultMutationEpoch();
  WriteFault(tracker, address + page_size * 5 + 8);
  Check(tracker.TakeDirtiedRanges(ranges, epoch) && epoch != before &&
            epoch == tracker.FaultMutationEpoch() &&
            ranges.Contains(address + page_size * 4, page_size * 4) &&
            !ranges.Intersects(address, page_size * 4) &&
            !ranges.Intersects(address + page_size * 8, page_size * 8),
        "a write fault did not log exactly its fault-ahead window");

  // Explicit CPU-dirty marks log their range.
  UploadAll(tracker, address, page_size * 16);
  tracker.MarkRegionAsCpuModified(address + page_size * 12, page_size);
  Check(tracker.TakeDirtiedRanges(ranges, epoch) && epoch == tracker.FaultMutationEpoch() &&
            ranges.Contains(address + page_size * 12, page_size) &&
            !ranges.Intersects(address, page_size * 12),
        "an explicit CPU-dirty mark was not logged");

  tracker.UntrackMemory(address, page_size * 16);
  Release(memory);
}

void TestForeignWatcherFaultsDoNotPromote() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 2;
  policy.hot_max = 8;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  // The page is never uploaded, so it stays CPU-dirty: faults there belong to other watchers.
  Check(tracker.IsRegionCpuModified(address, page_size), "new page was not CPU-dirty");
  for (int frame = 0; frame < 4; frame++) {
    WriteFault(tracker, address);
    tracker.AdvanceFrame();
  }
  Check(tracker.HotPageCount() == 0, "faults on a CPU-dirty page promoted it");
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestWrittenUploadCopiesOutsideLock() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address, page_size * 3),
        "new pages were not CPU-dirty");

  // A racing guest write fault during the unlocked copy re-dirties page 1; the late pass must
  // protect it again before copying it, and the range still ends GPU-owned.
  uint64_t main_pages = 0;
  uint64_t late_pages = 0;
  bool late_protected = false;
  bool racer_done = false;
  tracker.ForEachWrittenUploadRange(
      address, page_size * 3,
      [&](uint64_t, uint64_t bytes) noexcept { main_pages += bytes / page_size; },
      [&]() noexcept {
        Check(!IsWritable(memory + page_size),
              "main copy ran on an unprotected page");
        // Would deadlock if the region lock were held here.
        std::jthread racer([&] {
          tracker.InvalidateRegionOnWriteFault(address + page_size + 8, 1, [] {},
                                               [](uint64_t, uint64_t) noexcept {});
          memory[page_size + 8] = 0x77;
          racer_done = true;
        });
        racer.join();
      },
      [&](uint64_t range_address, uint64_t bytes) noexcept {
        late_pages += bytes / page_size;
        late_protected = range_address == address + page_size &&
                         !IsWritable(memory + page_size);
      },
      []() noexcept {});
  Check(racer_done && main_pages == 3 && late_pages == 1 && late_protected &&
            tracker.IsRegionGpuModified(address, page_size * 3) &&
            !tracker.IsRegionCpuModified(address, page_size * 3) &&
            Protection(memory + page_size) == PAGE_NOACCESS,
        "unlocked written upload lost a racing write or GPU ownership");

  // Without a racing write the late pass reports nothing.
  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  tracker.MarkRegionAsCpuModified(address, page_size);
  main_pages = 0;
  late_pages = 0;
  tracker.ForEachWrittenUploadRange(
      address, page_size * 3,
      [&](uint64_t, uint64_t bytes) noexcept { main_pages += bytes / page_size; },
      []() noexcept {},
      [&](uint64_t, uint64_t bytes) noexcept { late_pages += bytes / page_size; },
      []() noexcept {});
  Check(main_pages == 1 && late_pages == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "unlocked written upload copied clean pages or reported late pages");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  tracker.UntrackMemory(address, page_size * 3);
  Release(memory);
}

// ---------------------------------------------------------------------------------------------
// Deferred write-unprotect (KYTY_DEFER_UNPROTECT) through the tracker

void TestDeferredFaultUnprotect() {
  PageManager::SetDeferModeForTests(PageManager::DeferMode::On);
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  UploadAll(tracker, address, page_size * 2);
  Check(!IsWritable(memory) && !IsWritable(memory + page_size),
        "deferred fault setup did not protect");

  // Inside an enclosing scope (an emulator write between that scope's releases) the fault's
  // release waits for the scope; reconciling the faulting page makes it writable at once.
  {
    const PageManager::DeferUnprotectScope outer;
    WriteFault(tracker, address + 8);
    Check(tracker.IsRegionCpuModified(address, page_size),
          "fault did not make its page CPU-dirty");
    Check(!IsWritable(memory), "fault inside an enclosing scope unprotected before it ended");
    page_manager.Reconcile(address, 1, true);
    Check(IsWritable(memory), "reconciling the faulting page left it protected");
    memory[8] = 1;
  }
  // Alone, a fault returns with its page writable.
  WriteFault(tracker, address + page_size + 8);
  Check(IsWritable(memory + page_size), "fault returned with its page protected");
  memory[page_size + 8] = 2;

  // An upload takes the page back while a fault's release is still pending: the late update
  // leaves it protected, and the retried write faults again and is tracked.
  UploadAll(tracker, address, page_size * 2);
  std::binary_semaphore faulted{0};
  std::binary_semaphore uploaded{0};
  bool stale_unprotect = false;
  std::thread writer([&] {
    {
      const PageManager::DeferUnprotectScope scope;
      WriteFault(tracker, address + 16);
      faulted.release();
      uploaded.acquire();
    }
    stale_unprotect = IsWritable(memory);
    WriteFault(tracker, address + 16);
    memory[16] = 3;
  });
  faulted.acquire();
  uint64_t copied_pages = 0;
  tracker.ForEachUploadRange(
      address, page_size, false,
      [&](uint64_t, uint64_t bytes) noexcept { copied_pages += bytes / page_size; },
      []() noexcept {});
  Check(copied_pages == 1 && !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "upload during a pending release did not take the page back");
  uploaded.release();
  writer.join();
  Check(!stale_unprotect, "a stale release unprotected an uploaded page");
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory) &&
            memory[16] == 3,
        "the retried write after a pending release was not tracked");

  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
struct FaultStressState {
  PageManager *page_manager = nullptr;
  MemoryTracker *tracker = nullptr;
  uint64_t begin = 0;
  uint64_t end = 0;
  std::atomic<uint64_t> faults{0};
};
FaultStressState g_fault_stress;

// RenderContext::HandleFault's write path, on the stress test's memory.
bool FaultStressHandler(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  const auto address = info.access_violation_vaddr;
  auto *tracker = g_fault_stress.tracker;
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write || tracker == nullptr ||
      address < g_fault_stress.begin || address >= g_fault_stress.end) {
    return false;
  }
  g_fault_stress.faults.fetch_add(1, std::memory_order_relaxed);
  const bool nested = PageManager::InDeferUnprotectScope();
  {
    const PageManager::DeferUnprotectScope scope;
    tracker->InvalidateRegionOnWriteFault(address, 1, [] {},
                                          [](uint64_t, uint64_t) noexcept {});
    g_fault_stress.page_manager->Reconcile(address & ~(TRACKER_PAGE_SIZE - 1),
                                           TRACKER_PAGE_SIZE, nested);
  }
  return true;
}

// Writer threads store to tracked pages (real faults, resolved as HandleFault does) while an
// uploader keeps taking the pages back. After the last upload the uploaded copy must equal memory:
// a write that landed on a page the tracker considered clean would be missing from it.
void TestFaultStressNoLostWrites(PageManager::DeferMode mode) {
  PageManager::SetDeferModeForTests(mode);
  MemoryTracker::FaultPolicy policy;
  policy.ahead_pages = 4;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  constexpr uint64_t pages = 32;
  auto *memory = Allocate(page_manager, pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto size = page_size * pages;
  std::vector<uint8_t> shadow(size);
  uint64_t uploads = 0;
  const auto upload = [&] {
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    tracker.ForEachUploadRange(
        address, size, false,
        [&](uint64_t range, uint64_t bytes) noexcept { ranges.push_back({range, bytes}); },
        [&]() noexcept {
          for (const auto &[range, bytes] : ranges) {
            std::memcpy(shadow.data() + (range - address),
                        reinterpret_cast<const void *>(range), bytes);
          }
        });
    uploads++;
  };
  upload();
  Check(std::memcmp(shadow.data(), memory, size) == 0 && !IsWritable(memory),
        "fault stress setup did not upload and protect");

  g_fault_stress.page_manager = &page_manager;
  g_fault_stress.begin = address;
  g_fault_stress.end = address + size;
  g_fault_stress.tracker = &tracker;
  const auto faults_before = g_fault_stress.faults.load();
  const auto stats_before = PageManager::GetDeferStats();
  std::atomic<bool> writing{true};
  std::vector<std::thread> writers;
  std::atomic<uint64_t> writer_cpu_100ns{0};
  const auto start = std::chrono::steady_clock::now();
  for (uint32_t t = 0; t < 4; t++) {
    writers.emplace_back([&, t] {
      std::mt19937_64 rng(77 + t);
      for (uint32_t i = 1; i <= 20000; i++) {
        const auto offset = (rng() % (size / 8)) * 8;
        *reinterpret_cast<volatile uint64_t *>(memory + offset) = (uint64_t{t + 1} << 56) | i;
      }
      FILETIME creation{};
      FILETIME exit{};
      FILETIME kernel{};
      FILETIME user{};
      if (GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user) != 0) {
        const auto ticks = [](FILETIME time) {
          return (uint64_t{time.dwHighDateTime} << 32u) | time.dwLowDateTime;
        };
        writer_cpu_100ns.fetch_add(ticks(kernel) + ticks(user));
      }
    });
  }
  std::thread uploader([&] {
    while (writing.load(std::memory_order_acquire)) {
      upload();
    }
  });
  for (auto &writer : writers) {
    writer.join();
  }
  const auto wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  writing.store(false, std::memory_order_release);
  uploader.join();
  upload();
  g_fault_stress.tracker = nullptr;
  const auto faults = g_fault_stress.faults.load() - faults_before;
  const auto stats = PageManager::GetDeferStats();
  Check(faults != 0, "fault stress took no faults");
  Check(std::memcmp(shadow.data(), memory, size) == 0,
        "a write was not uploaded: it landed on a page the tracker considered clean");
  Check(stats.verify_mismatches == stats_before.verify_mismatches,
        "verify mode found protection mismatches under the fault stress");
  std::printf("  fault stress (%s): %llu faults in %.0f ms (writers %.0f ms CPU, %.1f us CPU "
              "per fault), %llu uploads, %llu deferred spans, %llu settled, %llu verify checks\n",
              mode == PageManager::DeferMode::Off   ? "deferral off"
              : mode == PageManager::DeferMode::On ? "deferred"
                                                   : "verify",
              static_cast<unsigned long long>(faults), wall_ms,
              static_cast<double>(writer_cpu_100ns.load()) / 1e4,
              faults != 0 ? static_cast<double>(writer_cpu_100ns.load()) / 10.0 /
                                static_cast<double>(faults)
                          : 0.0,
              static_cast<unsigned long long>(uploads),
              static_cast<unsigned long long>(stats.spans - stats_before.spans),
              static_cast<unsigned long long>(stats.settled - stats_before.settled),
              static_cast<unsigned long long>(stats.verify_checks - stats_before.verify_checks));
  tracker.UntrackMemory(address, size);
  Release(memory);
}

// --fault-bench: round-based contention on one tracking region. Each round the range is uploaded
// (clean and write-protected); then 8 writers each write once to each of their own pages, so
// every write takes exactly one fault, while a querier thread (standing in for the CP) keeps
// asking the tracker about the range. Reports the fault phase's wall time and the querier's
// latency. Run it with KYTY_TRACKER_LOCK_PARK=0 and =1.
void FaultBench() {
  for (const auto mode : {PageManager::DeferMode::Off, PageManager::DeferMode::On}) {
    PageManager::SetDeferModeForTests(mode);
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    auto &page_manager = harness.page_manager;
    const auto page_size = page_manager.GetPageSize();
    constexpr uint32_t writers = 8;
    constexpr uint64_t pages = 256;
    constexpr uint32_t rounds = 200;
    auto *memory = Allocate(page_manager, pages);
    const auto address = reinterpret_cast<uint64_t>(memory);
    const auto size = page_size * pages;
    UploadAll(tracker, address, size);
    g_fault_stress.page_manager = &page_manager;
    g_fault_stress.begin = address;
    g_fault_stress.end = address + size;
    g_fault_stress.tracker = &tracker;
    const auto faults_before = g_fault_stress.faults.load();

    std::atomic<uint32_t> phase{0}; // round * 2 + 1 while writing
    std::atomic<uint32_t> done{0};
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    for (uint32_t w = 0; w < writers; w++) {
      threads.emplace_back([&, w] {
        for (uint32_t round = 0; round < rounds; round++) {
          while (phase.load(std::memory_order_acquire) != round * 2 + 1) {
            _mm_pause();
          }
          for (uint64_t page = w; page < pages; page += writers) {
            *reinterpret_cast<volatile uint64_t *>(memory + page * page_size + 64) = round;
          }
          done.fetch_add(1, std::memory_order_acq_rel);
        }
      });
    }
    std::vector<uint32_t> query_ns;
    query_ns.reserve(1 << 20);
    std::thread querier([&] {
      while (!stop.load(std::memory_order_acquire)) {
        const auto start = std::chrono::steady_clock::now();
        (void)tracker.QueryDirty(address, size);
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        if (query_ns.size() < query_ns.capacity()) {
          query_ns.push_back(static_cast<uint32_t>(std::min<int64_t>(ns, UINT32_MAX)));
        }
      }
    });
    double fault_ms = 0;
    for (uint32_t round = 0; round < rounds; round++) {
      UploadAll(tracker, address, size);
      const auto start = std::chrono::steady_clock::now();
      phase.store(round * 2 + 1, std::memory_order_release);
      while (done.load(std::memory_order_acquire) != writers * (round + 1)) {
        _mm_pause();
      }
      fault_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            start)
                      .count();
      phase.store(round * 2 + 2, std::memory_order_release);
    }
    for (auto &thread : threads) {
      thread.join();
    }
    stop.store(true, std::memory_order_release);
    querier.join();
    g_fault_stress.tracker = nullptr;
    const auto faults = g_fault_stress.faults.load() - faults_before;
    std::sort(query_ns.begin(), query_ns.end());
    const auto percentile = [&](double p) {
      return query_ns.empty()
                 ? 0.0
                 : query_ns[static_cast<size_t>(p * static_cast<double>(query_ns.size() - 1))] /
                       1000.0;
    };
    std::printf("  fault bench (%s, lock park %s): %llu faults, %.1f us wall per fault; "
                "tracker query p50 %.1f us, p99 %.1f us, p99.9 %.1f us, max %.1f us\n",
                mode == PageManager::DeferMode::Off ? "deferral off" : "deferred",
                Libs::Graphics::TrackerLockParkEnabled() ? "on" : "off",
                static_cast<unsigned long long>(faults),
                faults != 0 ? fault_ms * 1000.0 / static_cast<double>(faults) : 0.0,
                percentile(0.5), percentile(0.99), percentile(0.999), percentile(1.0));
    tracker.UntrackMemory(address, size);
    Release(memory);
  }
}
#endif

void TestHotPageSettle() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 1;
  policy.hot_max = 8;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  UploadAll(tracker, address, page_size * 4);
  WriteFault(tracker, address);
  WriteFault(tracker, address + page_size * 2);
  Check(tracker.HotPageCount() == 2, "pages were not promoted for settling");

  // A ranged settle only touches its hot pages: they become clean and write-protected.
  const auto settled = tracker.SettleHotPages(address, page_size * 2);
  Check(settled.size() == 1 && settled[0] == address && tracker.HotPageCount() == 1 &&
            !tracker.IsRegionCpuModified(address, page_size) && !IsWritable(memory) &&
            tracker.IsRegionHot(address + page_size * 2, page_size) &&
            IsWritable(memory + page_size * 2),
        "ranged settle did not return exactly its hot page to clean tracking");
  // The caller found the contents changed: dirty and writable again.
  tracker.MarkRegionAsCpuModified(address, page_size);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "settled page could not be re-dirtied");

  // Settling everything (size 0) reaches every region.
  const auto all = tracker.SettleHotPages(0, 0);
  Check(all.size() == 1 && all[0] == address + page_size * 2 &&
            tracker.HotPageCount() == 0 && !IsWritable(memory + page_size * 2),
        "global settle missed a hot page");
  Check(tracker.SettleHotPages(0, 0).empty(), "settle without hot pages reported pages");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// MemoryTracker::RangeSignature (BufferCache::RangeMemo, KYTY_BUFFER_RANGE_MEMO): it must move on
// every change of a dirty, hot or readback-pending bit of the range, and stay put for queries and
// for transitions that change nothing (a clean range synchronized again, a hot page kept hot).
void TestRangeSignature() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 1;
  policy.hot_max = 8;
  PolicyHarness harness(policy, true);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto locked_signature = [&] { return tracker.RangeSignature(address, page_size * 4); };
  // Every transition below also keeps the lock-free dirty mirrors equal to the locked bits
  // (MemoryTracker::QueryDirtyRelaxed), page by page and over the whole range.
  const auto mirrors_agree = [&] {
    if (locked_signature() == 0) {
      return;
    }
    for (uint64_t page = 0; page <= 4; page++) {
      const auto start = page == 4 ? address : address + page * page_size;
      const auto size = page == 4 ? page_size * 4 : page_size;
      MemoryTracker::DirtyState relaxed;
      Check(tracker.QueryDirtyRelaxed(start, size, relaxed) &&
                relaxed.cpu == tracker.IsRegionCpuModified(start, size) &&
                relaxed.gpu == tracker.IsRegionGpuModified(start, size),
            "a lock-free dirty mirror differs from the locked bits");
    }
  };
  const auto signature = [&] {
    mirrors_agree();
    return locked_signature();
  };

  Check(signature() == 0 && tracker.RangeSignature(address, 0) == 0 &&
            tracker.RangeSignature(TRACKER_ADDRESS_SIZE - 1, 2) == 0,
        "a range without a region (or an invalid range) has a signature");
  Check(tracker.IsRegionCpuModified(address, page_size), "new region was not CPU-dirty");
  const auto created = signature();
  Check(created != 0, "an existing region has no signature");

  // Queries change nothing; uploads change the bits once.
  (void)tracker.IsRegionGpuModified(address, page_size * 4);
  Check(signature() == created, "a query moved the signature");
  UploadAll(tracker, address, page_size * 4);
  const auto uploaded = signature();
  Check(uploaded > created, "an upload that cleared pages kept the signature");
  UploadAll(tracker, address, page_size * 4);
  const auto [normal, hot] = UploadHotAware(tracker, address, page_size * 4);
  Check(normal == 0 && hot == 0 && signature() == uploaded,
        "synchronizing a clean range again moved the signature");

  // A write fault dirties a page.
  WriteFault(tracker, address + page_size);
  const auto faulted = signature();
  Check(faulted > uploaded, "a write fault kept the signature");
  // Marking a dirty page dirty again changes nothing; the upload does.
  tracker.MarkRegionAsCpuModified(address + page_size, 8);
  Check(signature() == faulted, "a no-op CPU-dirty mark moved the signature");
  UploadAll(tracker, address, page_size * 4);
  const auto clean_again = signature();
  Check(clean_again > faulted, "clearing the fault's pages kept the signature");

  // GPU ownership, readback marks and downloads.
  tracker.ForEachUploadRange(
      address + page_size * 3, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  const auto gpu_owned = signature();
  Check(gpu_owned > clean_again, "a written upload kept the signature");
  tracker.MarkRegionAsGpuModified(address + page_size * 3, page_size);
  Check(signature() == gpu_owned, "a no-op GPU-dirty mark moved the signature");
  tracker.MarkReadbackPending(address + page_size * 3, page_size);
  const auto pending = signature();
  Check(pending > gpu_owned, "a readback mark kept the signature");
  const auto unmarked = tracker.UnmarkReadbackPending(address + page_size * 3, page_size);
  const auto published = signature();
  Check(unmarked.unmarked_pages == 1 && published > pending &&
            !tracker.IsRegionGpuModified(address + page_size * 3, page_size),
        "a completed readback kept the signature");
  tracker.UnmarkRegionAsGpuModified(address + page_size * 3, page_size);
  tracker.ForEachDownloadRange<true>(address, page_size * 4,
                                     [](uint64_t, uint64_t) noexcept {});
  Check(signature() == published, "no-op GPU transitions moved the signature");
  tracker.MarkRegionAsGpuModified(address + page_size * 2, page_size);
  const auto remarked = signature();
  tracker.ForEachDownloadRange<false>(address, page_size * 4,
                                      [](uint64_t, uint64_t) noexcept {});
  Check(remarked > published && signature() == remarked,
        "a GPU-dirty mark kept, or a non-clearing download moved, the signature");
  tracker.ForEachDownloadRange<true>(address, page_size * 4,
                                     [](uint64_t, uint64_t) noexcept {});
  const auto downloaded = signature();
  Check(downloaded > remarked, "a clearing download kept the signature");

  // Hot pages: promotion moves it; hot-aware uploads and fault-free writes do not; settling,
  // demotion and the idle sweep do.
  UploadAll(tracker, address, page_size * 4);
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  const auto promoted = signature();
  Check(tracker.HotPageCount() == 1 && promoted > downloaded, "promotion kept the signature");
  const auto [normal2, hot2] = UploadHotAware(tracker, address, page_size * 4);
  memory[24] = 0x42;
  const auto [normal3, hot3] = UploadHotAware(tracker, address, page_size * 4);
  Check(normal2 == 0 && hot2 == 1 && normal3 == 0 && hot3 == 1 && signature() == promoted,
        "keeping a hot page hot (or writing it) moved the signature");
  Check(tracker.SettleHotPages(address, page_size).size() == 1 && signature() > promoted,
        "settling a hot page kept the signature");
  const auto settled = signature();
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  tracker.DemoteHotPages(address, page_size * 4);
  const auto demoted = signature();
  Check(tracker.HotPageCount() == 0 && demoted > settled, "demotion kept the signature");
  tracker.DemoteHotPages(address, page_size * 4);
  Check(signature() == demoted, "demoting a range without hot pages moved the signature");
  UploadAll(tracker, address, page_size * 4);
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  UploadHotAware(tracker, address, page_size * 4);
  const auto before_sweep = signature();
  for (int frame = 0; frame < 5; frame++) {
    tracker.AdvanceFrame();
  }
  tracker.SweepHotPages(3);
  Check(tracker.HotPageCount() == 0 && signature() > before_sweep,
        "the idle sweep kept the signature");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// The dirtying signature (MemoryTracker::RangeDirtiedSignature) moves exactly when a page can turn
// CPU-dirty: write faults (with hot promotion and fault-ahead) and CPU-dirty marks that change
// something. Uploads, hot-page settles and demotions, GPU transitions and no-op marks leave it.
void TestRangeDirtiedSignature() {
  MemoryTracker::FaultPolicy policy;
  policy.hot_frames = 1;
  policy.hot_max = 8;
  policy.ahead_pages = 2;
  PolicyHarness harness(policy, true);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto dirtied = [&] { return tracker.RangeDirtiedSignature(address, page_size * 4); };

  Check(dirtied() == 0, "a range without a region has a dirtying signature");
  UploadAll(tracker, address, page_size * 4);
  const auto created = dirtied();
  Check(created != 0, "an existing region has no dirtying signature");
  UploadAll(tracker, address, page_size * 4);
  Check(dirtied() == created, "an upload moved the dirtying signature");

  WriteFault(tracker, address + page_size * 2); // with fault-ahead of its pair
  const auto faulted = dirtied();
  Check(faulted > created, "a write fault kept the dirtying signature");
  tracker.MarkRegionAsCpuModified(address + page_size * 2, 8);
  Check(dirtied() == faulted, "a no-op CPU-dirty mark moved the dirtying signature");
  UploadAll(tracker, address, page_size * 4);
  Check(dirtied() == faulted, "clearing the fault's pages moved the dirtying signature");
  tracker.MarkRegionAsCpuModified(address, 8);
  const auto marked = dirtied();
  Check(marked > faulted, "a CPU-dirty mark kept the dirtying signature");
  UploadAll(tracker, address, page_size * 4);

  // GPU ownership and its readback do not dirty the CPU side.
  tracker.ForEachUploadRange(
      address + page_size * 3, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkReadbackPending(address + page_size * 3, page_size);
  (void)tracker.UnmarkReadbackPending(address + page_size * 3, page_size);
  Check(dirtied() == marked, "GPU transitions moved the dirtying signature");

  // Hot pages: promotion is a fault (moves it); settling and demotion do not.
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  const auto promoted = dirtied();
  Check(tracker.HotPageCount() == 1 && promoted > marked, "hot promotion kept the signature");
  Check(tracker.SettleHotPages(address, page_size).size() == 1 && dirtied() == promoted,
        "settling a hot page moved the dirtying signature");
  tracker.AdvanceFrame();
  WriteFault(tracker, address);
  const auto repromoted = dirtied();
  tracker.DemoteHotPages(address, page_size * 4);
  Check(tracker.HotPageCount() == 0 && dirtied() == repromoted,
        "demoting a hot page moved the dirtying signature");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// A signature over two regions moves when either region changes.
void TestRangeSignatureAcrossRegions() {
  constexpr uintptr_t base = 0x0000000204000000ull;
  constexpr uint64_t region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), region_size * 2,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base), "fixed VirtualAlloc failed");
  const auto boundary = base + region_size;
  const auto span = [&] { return tracker.RangeSignature(boundary - page_size, page_size * 2); };
  (void)tracker.IsRegionCpuModified(boundary - page_size, page_size);
  Check(span() == 0, "a range with a missing region has a signature");
  UploadAll(tracker, boundary - page_size, page_size * 2);
  const auto both = span();
  Check(both != 0, "a range over two existing regions has no signature");
  const auto first_only = tracker.RangeSignature(boundary - page_size, page_size);
  WriteFault(tracker, boundary + 8);
  const auto second = span();
  Check(second > both && tracker.RangeSignature(boundary - page_size, page_size) == first_only,
        "a change in the second region kept the two-region signature or moved the first's");
  WriteFault(tracker, boundary - page_size + 8);
  Check(span() > second, "a change in the first region kept the signature");
  tracker.UntrackMemory(base, region_size * 2);
  Release(memory);
}

// MemoryTracker::QueryDirty decides as `!IsRegionGpuModified && IsRegionCpuModified` does (it
// creates missing regions exactly when that expression would), and the lock-free GPU-dirty mirror
// (IsRegionGpuModifiedRelaxed) follows every GPU-bit transition, across regions and bit words.
void TestDirtyQueryAndGpuMirror() {
  constexpr uintptr_t base = 0x0000000204800000ull;
  constexpr uint64_t region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(base), region_size * 2,
                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  Check(memory == reinterpret_cast<void *>(base), "fixed VirtualAlloc failed");
  const auto boundary = base + region_size;
  const auto second_exists = [&] { return tracker.RangeSignature(boundary, page_size) != 0; };

  // First region only, with a GPU-dirty page right below the boundary.
  UploadAll(tracker, boundary - page_size * 80, page_size * 80);
  tracker.ForEachUploadRange(
      boundary - page_size, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(!second_exists(), "the second region exists too early");
  MemoryTracker::DirtyState relaxed_missing;
  Check(!tracker.QueryDirtyRelaxed(boundary - page_size, page_size * 2, relaxed_missing) &&
            !second_exists(),
        "a relaxed query decided (or created) a range with a missing region");
  bool cpu_missing = false;
  Check(!tracker.QueryCpuDirtyRelaxed(boundary - page_size, page_size * 2, cpu_missing) &&
            !second_exists(),
        "a CPU-only query decided (or created) a range with a missing region");
  const auto across = tracker.QueryDirty(boundary - page_size, page_size * 2);
  Check(across.gpu && !second_exists() &&
            tracker.IsRegionGpuModifiedRelaxed(boundary - page_size, page_size * 2),
        "a GPU-dirty query created the missing region or missed the GPU-dirty page");
  tracker.UnmarkRegionAsGpuModified(boundary - page_size, page_size);
  const auto created = tracker.QueryDirty(boundary - page_size, page_size * 2);
  Check(!created.gpu && created.cpu && second_exists(),
        "a clean query did not create the missing region CPU-dirty");
  UploadAll(tracker, boundary, page_size * 80);

  // Every query agrees with the locked ones after each GPU-bit transition.
  const std::array<std::pair<uint64_t, uint64_t>, 8> queries = {{
      {boundary - page_size * 70, page_size},
      {boundary - page_size * 70, page_size * 8},
      {boundary - page_size * 66, page_size * 3},
      {boundary - page_size * 65, page_size * 66},
      {boundary - page_size, page_size * 2},
      {boundary - 16, 32},
      {boundary + page_size * 63, page_size * 2},
      {boundary + page_size * 5, 64},
  }};
  const auto check_queries = [&](const char *what) {
    for (const auto &[address, size] : queries) {
      const bool gpu = tracker.IsRegionGpuModified(address, size);
      Check(tracker.IsRegionGpuModifiedRelaxed(address, size) == gpu &&
                tracker.GpuMirrorMatches(address, size),
            what);
      const auto state = tracker.QueryDirty(address, size);
      Check(state.gpu == gpu && (gpu || state.cpu == tracker.IsRegionCpuModified(address, size)),
            what);
      // The lock-free snapshot (every region exists here) equals the locked one exactly.
      MemoryTracker::DirtyState relaxed;
      Check(tracker.QueryDirtyRelaxed(address, size, relaxed) && relaxed.gpu == gpu &&
                relaxed.cpu == tracker.IsRegionCpuModified(address, size),
            what);
      bool cpu_dirty = false;
      Check(tracker.QueryCpuDirtyRelaxed(address, size, cpu_dirty) && cpu_dirty == relaxed.cpu,
            what);
    }
  };
  check_queries("queries diverged on a clean range");
  // A GPU-dirty run crossing a 64-page bit word (pages 958..962 of the first region).
  tracker.MarkRegionAsGpuModified(boundary - page_size * 66, page_size * 5);
  check_queries("queries diverged after a GPU-dirty mark across a bit word");
  tracker.ForEachDownloadRange<false>(boundary - page_size * 66, page_size * 5,
                                      [](uint64_t, uint64_t) noexcept {});
  check_queries("queries diverged after a non-clearing download");
  tracker.ForEachDownloadRange<true>(boundary - page_size * 66, page_size * 2,
                                     [](uint64_t, uint64_t) noexcept {});
  check_queries("queries diverged after a clearing download");
  tracker.MarkReadbackPending(boundary - page_size * 64, page_size * 3);
  check_queries("queries diverged after a readback mark");
  (void)tracker.UnmarkReadbackPending(boundary - page_size * 64, page_size * 3);
  check_queries("queries diverged after a completed readback");
  tracker.ForEachWrittenUploadRange(
      boundary + page_size * 62, page_size * 4, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {}, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  check_queries("queries diverged after a written upload in the second region");
  WriteFault(tracker, boundary - page_size * 70 + 8);
  check_queries("queries diverged after a write fault");
  tracker.MarkRegionAsCpuModified(boundary + page_size * 5, 64);
  check_queries("queries diverged after a CPU-dirty mark");

  tracker.ForEachDownloadRange<true>(base, region_size * 2, [](uint64_t, uint64_t) noexcept {});
  check_queries("queries diverged after clearing every GPU-dirty page");
  Check(!tracker.IsRegionGpuModifiedRelaxed(base, region_size * 2),
        "the mirror kept a cleared GPU-dirty page");
  tracker.UntrackMemory(base, region_size * 2);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  uint32_t ranges = 0;
  tracker.ForEachUploadRange(
      boundary - page_size, page_size * 2, false,
      [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
  Check(ranges == 2 &&
            !tracker.IsRegionCpuModified(boundary - page_size, page_size * 2) &&
            !IsWritable(reinterpret_cast<void *>(boundary - page_size)) &&
            !IsWritable(reinterpret_cast<void *>(boundary)),
        "cross-region upload did not clear and protect both regions");
  tracker.MarkRegionAsCpuModified(boundary - page_size, page_size * 2);
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

namespace CleanVerdict = Libs::Graphics::CleanVerdict;

void TestCleanVerdictQuery() {
  constexpr uint64_t page = 0x0000000200010000ull;
  constexpr uint64_t page_size = CleanVerdict::PAGE_SIZE;
  CleanVerdict::Table table;
  RangeSet dirty;
  uint32_t probes = 0;
  const auto is_clean = [&](uint64_t vaddr, uint64_t size) {
    probes++;
    return !dirty.Intersects(vaddr, size);
  };

  auto result = CleanVerdict::Query(table, page + 0x40, 16, is_clean);
  Check(result.clean && !result.hit && result.stores == 1 && probes == 1,
        "first clean read did not prove its whole page");
  result = CleanVerdict::Query(table, page + page_size - 16, 16, is_clean);
  Check(result.clean && result.hit && probes == 1,
        "second read in a proven page was not answered from the table");

  // Dirtying transitions bump before the state changes.
  CleanVerdict::Invalidate();
  dirty.Add(page + 0x800, 4);
  result = CleanVerdict::Query(table, page + 0x800, 4, is_clean);
  Check(!result.clean && !result.hit,
        "a cached clean verdict survived a generation bump");
  probes = 0;
  result = CleanVerdict::Query(table, page + 0x40, 16, is_clean);
  Check(result.clean && !result.hit && result.stores == 0 && probes == 1,
        "clean bytes of a partially dirty page did not use one exact query");

  // Dirty-to-clean needs no bump: a stale Dirty verdict only forces exact
  // queries until the next bump.
  dirty.Subtract(page + 0x800, 4);
  result = CleanVerdict::Query(table, page + 0x800, 4, is_clean);
  Check(result.clean && !result.hit && result.stores == 0,
        "a stale Dirty verdict hid clean bytes");
  CleanVerdict::Invalidate();
  result = CleanVerdict::Query(table, page + 0x800, 4, is_clean);
  Check(result.clean && result.stores == 1, "a cleaned page was not reproven");

  // A read crossing into a partially dirty page falls back to its exact range.
  CleanVerdict::Invalidate();
  dirty.Add(page + page_size + 0x100, 4);
  result = CleanVerdict::Query(table, page + page_size - 8, 16, is_clean);
  Check(result.clean && !result.hit && result.stores == 1,
        "cross-page read did not combine page and exact verdicts");
  result = CleanVerdict::Query(table, page + page_size + 0xfc, 8, is_clean);
  Check(!result.clean, "cross-page dirty bytes were reported clean");

  // Page 0 and long reads keep exact queries and never populate the table.
  probes = 0;
  result = CleanVerdict::Query(table, 0x100, 16, is_clean);
  Check(result.clean && result.stores == 0 && probes == 1,
        "page 0 used the verdict table");
  result = CleanVerdict::Query(
      table, page + page_size * 8,
      page_size * CleanVerdict::MAX_CACHED_PAGES + 1, is_clean);
  Check(result.clean && result.stores == 0 && probes == 2,
        "long read used the verdict table");
  dirty.Clear();
}

void TestCleanVerdictTrackerTransitionsBump() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  // New regions start CPU dirty; upload them before marking GPU ownership.
  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  auto generation = CleanVerdict::Generation();
  tracker.MarkRegionAsGpuModified(address, page_size);
  Check(CleanVerdict::Generation() > generation,
        "tracker GPU marking did not retire clean verdicts");
  generation = CleanVerdict::Generation();
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(CleanVerdict::Generation() > generation,
        "tracker GPU unmarking did not retire clean verdicts");
  generation = CleanVerdict::Generation();
  tracker.ForEachDownloadRange<true>(address, page_size,
                                     [](uint64_t, uint64_t) noexcept {});
  Check(CleanVerdict::Generation() > generation,
        "clearing GPU download ranges did not retire clean verdicts");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

// Protocol check: a writer bumps, dirties the page, then publishes that fact.
// A reader that observes the publication must never answer "clean" from its
// table, even though it proved the page clean earlier on its own thread.
void TestCleanVerdictCrossThreadInvalidation() {
  constexpr uint64_t page = 0x0000000200020000ull;
  constexpr uint32_t rounds = 2000;
  std::atomic<bool> dirty{false};
  std::atomic<uint32_t> published{0};
  std::atomic<uint32_t> consumed{0};
  std::atomic<uint32_t> failures{0};
  const auto is_clean = [&](uint64_t, uint64_t) {
    return !dirty.load(std::memory_order_acquire);
  };

  std::jthread reader([&] {
    CleanVerdict::Table table;
    for (uint32_t round = 1; round <= rounds; round++) {
      while (!CleanVerdict::Query(table, page, 16, is_clean).clean) {
        std::this_thread::yield();
      }
      consumed.store(round, std::memory_order_release);
      while (published.load(std::memory_order_acquire) != round) {
        std::this_thread::yield();
      }
      if (CleanVerdict::Query(table, page + 0x80, 16, is_clean).clean) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
      consumed.store(round + rounds, std::memory_order_release);
    }
  });
  for (uint32_t round = 1; round <= rounds; round++) {
    while (consumed.load(std::memory_order_acquire) != round) {
      std::this_thread::yield();
    }
    CleanVerdict::Invalidate();
    dirty.store(true, std::memory_order_release);
    published.store(round, std::memory_order_release);
    while (consumed.load(std::memory_order_acquire) != round + rounds) {
      std::this_thread::yield();
    }
    dirty.store(false, std::memory_order_release);
    CleanVerdict::Invalidate();
  }
  reader.join();
  Check(failures.load() == 0,
        "a published dirtying transition was hidden by a cached verdict");
}

// Draw-prep S4 audit of the tracker (fault-ahead, hot pages, unlocked written uploads): every
// transition of the tracker's GPU ownership logs its exact range and source to the coherence log
// (whose generation is the clean-verdict one), and the CPU-side policies move no GPU ownership at
// all: they log nothing and leave the GPU state as it was. The written upload's GPU bits are, as
// for ForEachUploadRange, not read by clean verdicts; its clean-verdict transition is the buffer
// cache's dirty-range Add, which logs the range itself. main() enables a log reader
// (KYTY_DRAW_PREP_LOG_AUDIT) so that entries are recorded.
std::vector<std::pair<Libs::Graphics::Coherence::Range, Libs::Graphics::Coherence::Source>>
LoggedSince(uint64_t generation) {
  namespace Coherence = Libs::Graphics::Coherence;
  std::vector<std::pair<Coherence::Range, Coherence::Source>> entries;
  const auto newest = Coherence::Generation();
  for (auto g = generation + 1; g <= newest; g++) {
    Coherence::Range range;
    Coherence::Source source{};
    Check(Coherence::g_log.Read(g, range, source),
          "a logged coherence transition is unreadable");
    entries.push_back({range, source});
  }
  return entries;
}

bool LoggedExactly(uint64_t generation, uint64_t address, uint64_t size,
                   Libs::Graphics::Coherence::Source source) {
  const auto entries = LoggedSince(generation);
  return entries.size() == 1 &&
         entries[0].first ==
             Libs::Graphics::Coherence::MakeRange(address, size) &&
         entries[0].second == source;
}

void TestCoherenceLogTrackerTransitions() {
  namespace Coherence = Libs::Graphics::Coherence;
  using Coherence::Source;
  Check(Coherence::LogReadersEnabled(), "coherence log entries are not recorded");
  MemoryTracker::FaultPolicy policy;
  policy.ahead_pages = 8;
  policy.hot_frames = 1;
  policy.hot_max = 8;
  PolicyHarness harness(policy);
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 16);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto page = [&](uint64_t index) { return address + index * page_size; };

  auto generation = Coherence::Generation();
  UploadAll(tracker, address, page_size * 16);
  Check(LoggedSince(generation).empty(), "a read upload logged a coherence transition");

  // Hooked GPU-ownership transitions: exact range and source.
  generation = Coherence::Generation();
  tracker.MarkRegionAsGpuModified(page(12), page_size);
  Check(LoggedExactly(generation, page(12), page_size, Source::TrackerGpuMark),
        "GPU marking did not log its exact range");

  // Fault-ahead and hot promotion (one faulting frame suffices here): CPU state only.
  generation = Coherence::Generation();
  std::vector<std::pair<uint64_t, uint64_t>> ahead;
  Check(WriteFault(tracker, page(2) + 16, &ahead) == 0 && !ahead.empty() &&
            tracker.HotPageCount() == 1,
        "audit setup: the write fault did not open a window and promote its page");
  Check(LoggedSince(generation).empty() &&
            tracker.IsRegionGpuModified(page(12), page_size) &&
            !tracker.IsRegionGpuModified(address, page_size * 12) &&
            !tracker.IsRegionGpuModified(page(13), page_size * 3),
        "fault-ahead or hot promotion changed GPU ownership or logged a transition");
  // Hot-aware uploads, demotion, sweep and settle: CPU state only as well.
  generation = Coherence::Generation();
  (void)UploadHotAware(tracker, address, page_size * 8);
  tracker.DemoteHotPages(address, page_size * 16);
  UploadAll(tracker, address, page_size * 12);
  tracker.AdvanceFrame();
  (void)WriteFault(tracker, page(3));
  Check(tracker.HotPageCount() == 1, "audit setup: the page was not promoted again");
  (void)tracker.SettleHotPages(0, 0);
  tracker.AdvanceFrame();
  (void)WriteFault(tracker, page(3));
  for (int frame = 0; frame < 4; frame++) {
    tracker.AdvanceFrame();
  }
  tracker.SweepHotPages(1);
  Check(LoggedSince(generation).empty() &&
            tracker.IsRegionGpuModified(page(12), page_size) &&
            !tracker.IsRegionGpuModified(address, page_size * 12),
        "hot-page maintenance changed GPU ownership or logged a transition");

  // The unlocked written upload makes its range GPU-owned in the tracker at its end, without a
  // log entry: the buffer cache's dirty-range Add that follows it is the logged transition.
  generation = Coherence::Generation();
  tracker.ForEachWrittenUploadRange(
      page(8), page_size * 2, [](uint64_t, uint64_t) noexcept {}, []() noexcept {},
      [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
  Check(LoggedSince(generation).empty() &&
            tracker.IsRegionGpuModified(page(8), page_size * 2),
        "the written upload logged a transition or did not take GPU ownership");

  // The remaining hooked transitions.
  generation = Coherence::Generation();
  tracker.UnmarkRegionAsGpuModified(page(8), page_size * 2);
  Check(LoggedExactly(generation, page(8), page_size * 2, Source::TrackerGpuUnmark),
        "GPU unmarking did not log its exact range");
  generation = Coherence::Generation();
  tracker.ForEachDownloadRange<true>(page(12), page_size,
                                     [](uint64_t, uint64_t) noexcept {});
  Check(LoggedExactly(generation, page(12), page_size, Source::TrackerDownload) &&
            !tracker.IsRegionGpuModified(page(12), page_size),
        "a clearing download did not log its exact range");
  tracker.MarkRegionAsGpuModified(page(12), page_size);
  generation = Coherence::Generation();
  tracker.MarkReadbackPending(page(12), page_size);
  Check(LoggedSince(generation).empty(), "marking a readback pending logged a transition");
  (void)tracker.UnmarkReadbackPending(page(12), page_size);
  Check(LoggedExactly(generation, page(12), page_size, Source::TrackerReadbackUnmark) &&
            !tracker.IsRegionGpuModified(page(12), page_size),
        "a readback unmark did not log its exact range");

  tracker.UntrackMemory(address, page_size * 16);
  Release(memory);
}

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload"}) {
    CheckDeathCase(name);
  }
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  // Record coherence-log entries (read by TestCoherenceLogTrackerTransitions). Set before the
  // first transition: the choice is made once per process.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  _putenv_s("KYTY_DRAW_PREP_LOG_AUDIT", "1");
#else
  setenv("KYTY_DRAW_PREP_LOG_AUDIT", "1", 1);
#endif
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  if (argc == 2 && std::strcmp(argv[1], "--fault-bench") == 0) {
    Check(Common::HostException::InstallFirstAccessHandler(FaultStressHandler),
          "install fault bench handler failed");
    FaultBench();
    return 0;
  }
#endif
  TestGuestRange();
  TestRangeSet();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestRangeInvalidation();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestWriteTickMap();
  TestRangeSetModel();
  TestWriteTickMapModel();
  TestEagerReadbackPages();
  TestReadbackPendingUnmark();
  TestRangeGpuOwned();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestCleanVerdictQuery();
  TestCleanVerdictTrackerTransitionsBump();
  TestCleanVerdictCrossThreadInvalidation();
  TestFaultAheadWindow();
  TestMirrorsAcrossWords();
  TestHotPagePromotionAndUpload();
  TestHotPageDemotionPaths();
  TestFaultMutationEpochWithHotPages();
  TestDirtiedLog();
  TestForeignWatcherFaultsDoNotPromote();
  TestWrittenUploadCopiesOutsideLock();
  TestHotPageSettle();
  TestRangeSignature();
  TestRangeDirtiedSignature();
  TestRangeSignatureAcrossRegions();
  TestDirtyQueryAndGpuMirror();
  TestCoherenceLogTrackerTransitions();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
#endif

  // Everything above ran with deferred write-unprotect (the default). The fault and upload cases
  // again with it off (the previous synchronous releases) and in verify mode.
  TestDeferredFaultUnprotect();
  for (const auto mode : {PageManager::DeferMode::Off, PageManager::DeferMode::Verify}) {
    PageManager::SetDeferModeForTests(mode);
    TestRangeInvalidation();
    TestGpuReacquisitionAfterInvalidation();
    TestFaultAheadWindow();
    TestHotPagePromotionAndUpload();
    TestForeignWatcherFaultsDoNotPromote();
    TestWrittenUploadCopiesOutsideLock();
    TestHotPageSettle();
  }
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  Check(Common::HostException::InstallFirstAccessHandler(FaultStressHandler),
        "install fault stress handler failed");
  for (const auto mode : {PageManager::DeferMode::Off, PageManager::DeferMode::On,
                          PageManager::DeferMode::Verify}) {
    TestFaultStressNoLostWrites(mode);
  }
#endif
  Check(PageManager::GetDeferStats().verify_mismatches == 0,
        "verify mode found protection mismatches");
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
