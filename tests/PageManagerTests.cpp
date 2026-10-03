#include "common/virtualMemory.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/parkingLock.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#else
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::PageManager;
using Libs::Graphics::RegionBits;
using Libs::Graphics::TRACKER_PAGE_SIZE;
using Libs::Graphics::TRACKER_REGION_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "PageManagerTests: failed: %s\n", text);
    std::abort();
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
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
};
std::vector<ProtectionCall> g_protection_ranges;

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
  g_protection_ranges.push_back({vaddr, size});
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

uint8_t *Allocate(uint64_t size, uint32_t protection = PAGE_READWRITE) {
  constexpr uintptr_t test_address = 0x0000000200010000ull;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  auto *memory = static_cast<uint8_t *>(
      VirtualAlloc(reinterpret_cast<void *>(test_address), size,
                   MEM_RESERVE | MEM_COMMIT, protection));
  Check(memory == reinterpret_cast<void *>(test_address),
        "fixed low VirtualAlloc failed");
#else
  void *raw = ::mmap(reinterpret_cast<void *>(test_address), size,
                     ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
  Check(raw == reinterpret_cast<void *>(test_address), "fixed low mmap failed");
  auto *memory = static_cast<uint8_t *>(raw);
  AllocationSizes()[raw] = static_cast<size_t>(size);
#endif
  return memory;
}

void TestWatchAndUnwatch() {
  g_protection_calls = 0;
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size * 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  manager.UpdatePageWatchers<true>(address, page_size);
  Check(Protection(memory) == PAGE_READONLY && IsWritable(memory + page_size),
        "write watch installed incorrect protections");
  Check(g_protection_calls != 0,
        "watch protection bypassed the address-space owner callback");
  manager.UpdatePageWatchers<false>(address, page_size);
  Check(IsWritable(memory), "write unwatch did not restore access");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestSharedWatcherCounts() {
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size);
  const auto address = reinterpret_cast<uint64_t>(memory);

  manager.UpdatePageWatchers<true>(address + 8, 32);
  manager.UpdatePageWatchers<true>(address + 128, 64);
  manager.UpdatePageWatchers<false>(address + 8, 32);
  Check(Protection(memory) == PAGE_READONLY,
        "first unwatch released a shared watcher");
  manager.UpdatePageWatchers<false>(address + 128, 64);
  Check(IsWritable(memory), "last unwatch did not restore access");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestCrossRegionRange() {
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  auto *memory = Allocate(region_size * 2);
  const auto base = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (base + region_size - 1) & ~(region_size - 1);
  Check(boundary >= base + page_size &&
            boundary + page_size <= base + region_size * 2,
        "test allocation does not contain a region boundary");

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchers<true>(boundary - page_size, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_ranges.size() == 2 &&
            g_protection_ranges[0].address == boundary - page_size &&
            g_protection_ranges[0].size == page_size &&
            g_protection_ranges[1].address == boundary &&
            g_protection_ranges[1].size == page_size,
        "cross-region watch was not split only at the region boundary");
  Check(!IsWritable(reinterpret_cast<void *>(boundary - page_size)) &&
            !IsWritable(reinterpret_cast<void *>(boundary)),
        "cross-region watch did not protect both pages");
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchers<false>(boundary - page_size, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_ranges.size() == 2 &&
            g_protection_ranges[0].address == boundary - page_size &&
            g_protection_ranges[0].size == page_size &&
            g_protection_ranges[1].address == boundary &&
            g_protection_ranges[1].size == page_size,
        "cross-region unwatch was not split only at the region boundary");
  Check(IsWritable(reinterpret_cast<void *>(boundary - page_size)) &&
            IsWritable(reinterpret_cast<void *>(boundary)),
        "cross-region unwatch did not restore both pages");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestBatchedWatcherRanges() {
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  constexpr uint64_t region_size = 4ull * 1024ull * 1024ull;
  constexpr uint64_t allocation_size = region_size * 3;
  auto *memory = Allocate(allocation_size);
  const auto address = reinterpret_cast<uint64_t>(memory);

  manager.UpdatePageWatchers<true>(address + page_size, page_size);
  manager.UpdatePageWatchers<true>(address + page_size * 3, page_size);
  manager.UpdatePageWatchers<true>(address, page_size * 5);
  manager.UpdatePageWatchers<false>(address, page_size * 5);
  Check(IsWritable(memory) && Protection(memory + page_size) == PAGE_READONLY &&
            IsWritable(memory + page_size * 2) &&
            Protection(memory + page_size * 3) == PAGE_READONLY &&
            IsWritable(memory + page_size * 4),
        "fragmented unwatch lost overlapping watcher counts");
  manager.UpdatePageWatchers<false>(address + page_size, page_size);
  manager.UpdatePageWatchers<false>(address + page_size * 3, page_size);

  g_protection_calls = 0;
  manager.UpdatePageWatchers<true>(address, allocation_size);
  Check(g_protection_calls == 4 && Protection(memory) == PAGE_READONLY &&
            Protection(memory + region_size) == PAGE_READONLY &&
            Protection(memory + region_size * 2) == PAGE_READONLY &&
            Protection(memory + allocation_size - page_size) == PAGE_READONLY,
        "large watch was not batched and protected by tracking region");
  manager.UpdatePageWatchers<false>(address, allocation_size);
  Check(IsWritable(memory) && IsWritable(memory + region_size) &&
            IsWritable(memory + region_size * 2) &&
            IsWritable(memory + allocation_size - page_size),
        "large cross-region unwatch did not restore the full range");

  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRegionMaskWatcherRanges() {
  PageManager manager;
  constexpr auto page_size = TRACKER_PAGE_SIZE;
  constexpr auto region_size = TRACKER_REGION_SIZE;
  auto *memory = Allocate(region_size * 2);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracking region");

  RegionBits full_mask;
  full_mask.Fill();
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, full_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base &&
            g_protection_ranges[0].size == region_size,
        "full region mask did not use one protection span");
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false>(region_base, full_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base &&
            g_protection_ranges[0].size == region_size,
        "full region unmask did not use one protection span");

  RegionBits sparse_mask;
  sparse_mask.Set(1);
  sparse_mask.Set(3);
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, sparse_mask);
  Check(g_protection_calls == 2 &&
            Protection(reinterpret_cast<void *>(region_base + page_size)) ==
                PAGE_READONLY &&
            IsWritable(reinterpret_cast<void *>(region_base + page_size * 2)) &&
            Protection(reinterpret_cast<void *>(region_base + page_size * 3)) ==
                PAGE_READONLY,
        "sparse region mask installed incorrect write watchers");
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false>(region_base, sparse_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base + page_size &&
            g_protection_ranges[0].size == page_size * 3,
        "sparse unmask did not bridge a compatible gap");

  manager.UpdatePageWatchersForRegion<true>(region_base, sparse_mask);
  g_protection_calls = 0;
  manager.UpdatePageWatchersForRegion<true>(region_base, sparse_mask);
  Check(g_protection_calls == 0,
        "duplicate sparse watch changed an already protected range");
  manager.UpdatePageWatchersForRegion<false>(region_base, sparse_mask);
  Check(g_protection_calls == 0,
        "first sparse unwatch released a duplicate watcher");
  manager.UpdatePageWatchersForRegion<false>(region_base, sparse_mask);

  manager.UpdatePageWatchers<true>(region_base + page_size * 2, page_size);
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, sparse_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base + page_size &&
            g_protection_ranges[0].size == page_size * 3,
        "sparse mask did not bridge a compatible protected gap");
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false>(region_base, sparse_mask);
  Check(g_protection_calls == 2 && g_protection_ranges.size() == 2 &&
            g_protection_ranges[0].address == region_base + page_size &&
            g_protection_ranges[0].size == page_size &&
            g_protection_ranges[1].address == region_base + page_size * 3 &&
            g_protection_ranges[1].size == page_size,
        "sparse unmask crossed an incompatible protected gap");
  manager.UpdatePageWatchers<false>(region_base + page_size * 2, page_size);

  g_protection_calls = 0;
  manager.UpdatePageWatchersForRegion<true, true>(region_base, sparse_mask);
  Check(g_protection_calls == 2 &&
            Protection(reinterpret_cast<void *>(region_base + page_size)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + page_size * 3)) ==
                PAGE_NOACCESS,
        "sparse read mask did not deny access");
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false, true>(region_base, sparse_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].size == page_size * 3,
        "sparse read unmask did not bridge a compatible gap");

  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// The tracker-gap detector's view (PageManager::CountWatchedPages): write-watched pages, pages
// watched for every access, sub-page ranges, and nothing once unwatched.
void TestCountWatchedPages() {
  PageManager manager;
  constexpr auto page_size = TRACKER_PAGE_SIZE;
  constexpr auto region_size = TRACKER_REGION_SIZE;
  auto *memory = Allocate(region_size * 2);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  const auto span = page_size * 6;
  auto counts = manager.CountWatchedPages(region_base, span);
  Check(counts.write == 0 && counts.access == 0,
        "pages no one watches were counted as watched");
  manager.UpdatePageWatchers<true>(region_base + page_size, page_size * 2);
  RegionBits access_mask;
  access_mask.Set(4);
  manager.UpdatePageWatchersForRegion<true, true>(region_base, access_mask);
  counts = manager.CountWatchedPages(region_base, span);
  Check(counts.write == 2 && counts.access == 1,
        "write- and access-watched pages were not told apart");
  counts = manager.CountWatchedPages(region_base + page_size * 2 + 16, 16);
  Check(counts.write == 1 && counts.access == 0,
        "a sub-page range did not count its page");
  counts = manager.CountWatchedPages(region_base + region_size - page_size, page_size * 2);
  Check(counts.write == 0 && counts.access == 0,
        "a range into a region no one tracks was counted as watched");
  manager.UpdatePageWatchersForRegion<false, true>(region_base, access_mask);
  manager.UpdatePageWatchers<false>(region_base + page_size, page_size * 2);
  counts = manager.CountWatchedPages(region_base, span);
  Check(counts.write == 0 && counts.access == 0, "unwatched pages were still counted");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRegionEndpointBatching() {
  PageManager manager;
  constexpr auto page_size = TRACKER_PAGE_SIZE;
  constexpr auto region_size = TRACKER_REGION_SIZE;
  constexpr auto last_page = region_size / page_size - 1;
  auto *memory = Allocate(region_size * 2);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracking region");

  RegionBits endpoints;
  endpoints.Set(0);
  endpoints.Set(last_page);
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, endpoints);
  Check(g_protection_calls == 2 && g_protection_ranges.size() == 2 &&
            g_protection_ranges[0].address == region_base &&
            g_protection_ranges[0].size == page_size &&
            g_protection_ranges[1].address ==
                region_base + region_size - page_size &&
            g_protection_ranges[1].size == page_size,
        "endpoint watch did not protect only the selected pages");

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false>(region_base, endpoints);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base &&
            g_protection_ranges[0].size == region_size,
        "endpoint unwatch did not coalesce the compatible 4 MiB span");

  RegionBits full_mask;
  full_mask.Fill();
  manager.UpdatePageWatchersForRegion<true>(region_base, full_mask);
  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, full_mask);
  Check(g_protection_calls == 0,
        "duplicate full-region watch issued a redundant protection call");
  manager.UpdatePageWatchersForRegion<false>(region_base, full_mask);
  Check(g_protection_calls == 0,
        "first full-region unwatch released overlapping watcher counts");
  manager.UpdatePageWatchersForRegion<false>(region_base, full_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base &&
            g_protection_ranges[0].size == region_size,
        "last full-region unwatch did not use one 4 MiB protection call");

  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestReadWriteWatcherInteractions() {
  PageManager manager;
  constexpr auto page_size = TRACKER_PAGE_SIZE;
  constexpr auto region_size = TRACKER_REGION_SIZE;
  auto *memory = Allocate(region_size * 2);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracking region");

  RegionBits write_mask;
  write_mask.SetRange(10, 15);
  RegionBits read_mask;
  read_mask.Set(11);
  read_mask.Set(13);

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true>(region_base, write_mask);
  Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
            g_protection_ranges[0].address == region_base + page_size * 10 &&
            g_protection_ranges[0].size == page_size * 5,
        "contiguous write watch was not batched");

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<true, true>(region_base, read_mask);
  Check(
      g_protection_calls == 2 &&
          Protection(reinterpret_cast<void *>(region_base + page_size * 11)) ==
              PAGE_NOACCESS &&
          Protection(reinterpret_cast<void *>(region_base + page_size * 12)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + page_size * 13)) ==
              PAGE_NOACCESS,
      "read watchers did not compose with write-only watchers");

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false>(region_base, write_mask);
  Check(
      g_protection_calls == 3 &&
          IsWritable(reinterpret_cast<void *>(region_base + page_size * 10)) &&
          Protection(reinterpret_cast<void *>(region_base + page_size * 11)) ==
              PAGE_NOACCESS &&
          IsWritable(reinterpret_cast<void *>(region_base + page_size * 12)) &&
          Protection(reinterpret_cast<void *>(region_base + page_size * 13)) ==
              PAGE_NOACCESS &&
          IsWritable(reinterpret_cast<void *>(region_base + page_size * 14)),
      "write unwatch changed pages still owned by read watchers");

  g_protection_calls = 0;
  g_protection_ranges.clear();
  manager.UpdatePageWatchersForRegion<false, true>(region_base, read_mask);
  Check(
      g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
          g_protection_ranges[0].address == region_base + page_size * 11 &&
          g_protection_ranges[0].size == page_size * 3 &&
          IsWritable(reinterpret_cast<void *>(region_base + page_size * 11)) &&
          IsWritable(reinterpret_cast<void *>(region_base + page_size * 13)),
      "read unwatch did not coalesce through compatible writable state");

  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// ---------------------------------------------------------------------------------------------
// Deferred write-unprotect (KYTY_DEFER_UNPROTECT) and the parking lock

using DeferMode = PageManager::DeferMode;

void SetDeferMode(DeferMode mode) { PageManager::SetDeferModeForTests(mode); }

// A release inside a scope changes the counts at once and the host only when the outermost scope
// ends; outside a scope, and with the switch off, releases stay synchronous.
void TestDeferredReleaseWaitsForScope(DeferMode mode) {
  SetDeferMode(mode);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size * 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  manager.UpdatePageWatchers<true>(address, page_size * 2);
  Check(Protection(memory) == PAGE_READONLY, "watch did not protect");
  {
    const PageManager::DeferUnprotectScope outer;
    {
      const PageManager::DeferUnprotectScope inner;
      g_protection_calls = 0;
      manager.UpdatePageWatchers<false>(address, page_size);
      if (mode == DeferMode::Off) {
        Check(g_protection_calls == 1 && IsWritable(memory),
              "release with deferral off was not synchronous");
      } else {
        Check(g_protection_calls == 0 && Protection(memory) == PAGE_READONLY,
              "deferred release reached the host before its scope ended");
      }
    }
    if (mode != DeferMode::Off) {
      Check(Protection(memory) == PAGE_READONLY,
            "an inner scope applied its enclosing scope's release");
      // A watch is always synchronous, even inside a scope.
      manager.UpdatePageWatchers<true>(address + page_size, page_size);
      Check(Protection(memory + page_size) == PAGE_READONLY, "watch inside a scope was deferred");
      manager.UpdatePageWatchers<false>(address + page_size, page_size);
    }
  }
  Check(IsWritable(memory), "outermost scope end did not apply the release");
  Check(Protection(memory + page_size) == PAGE_READONLY,
        "scope end released a page still watched");
  manager.UpdatePageWatchers<false>(address + page_size, page_size);
  Check(IsWritable(memory + page_size), "synchronous release did not apply");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// A pending release racing a synchronous watch/release on another thread: the host always ends
// at what the counts ask, and the scope's late update never loosens a page watched meanwhile.
void TestDeferredReleaseRacingWatch(DeferMode mode) {
  SetDeferMode(mode);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size);
  const auto address = reinterpret_cast<uint64_t>(memory);

  // 1. Pending release; another thread watches again; the scope ends: still protected.
  manager.UpdatePageWatchers<true>(address, page_size);
  {
    const PageManager::DeferUnprotectScope scope;
    manager.UpdatePageWatchers<false>(address, page_size);
    std::thread([&] { manager.UpdatePageWatchers<true>(address, page_size); }).join();
    Check(Protection(memory) == PAGE_READONLY, "re-watch during a pending release failed");
  }
  Check(Protection(memory) == PAGE_READONLY,
        "a stale deferred release loosened a page watched again");

  // 2. Pending release; another thread watches and releases; the scope ends: writable.
  const auto settled_before = PageManager::GetDeferStats().settled;
  {
    const PageManager::DeferUnprotectScope scope;
    manager.UpdatePageWatchers<false>(address, page_size);
    std::thread([&] {
      manager.UpdatePageWatchers<true>(address, page_size);
      manager.UpdatePageWatchers<false>(address, page_size);
    }).join();
    Check(IsWritable(memory), "synchronous release during a pending one did not apply");
  }
  Check(IsWritable(memory), "pending release ended protected");
  if (mode != DeferMode::Off) {
    Check(PageManager::GetDeferStats().settled > settled_before,
          "a span another thread had applied was not reported settled");
  }

  // 3. A pending release reconciled at once by another thread (a faulting page).
  manager.UpdatePageWatchers<true>(address, page_size);
  std::binary_semaphore released{0};
  std::binary_semaphore reconciled{0};
  std::thread owner([&] {
    const PageManager::DeferUnprotectScope scope;
    manager.UpdatePageWatchers<false>(address, page_size);
    released.release();
    reconciled.acquire();
  });
  released.acquire();
  if (mode != DeferMode::Off) {
    Check(Protection(memory) == PAGE_READONLY, "release applied before its scope ended");
  }
  manager.Reconcile(address, 1, true);
  Check(IsWritable(memory), "reconcile did not apply another thread's pending release");
  reconciled.release();
  owner.join();
  Check(IsWritable(memory), "pending release scope changed a reconciled page");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// KernelMprotect can change the host protection without updating PageManager's `applied`.
// Rewatch must tighten such pages even while an old release remains deferred.
void TestReuseAfterExternalProtection() {
  SetDeferMode(DeferMode::Verify);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size * 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  manager.UpdatePageWatchers<true>(address, 4 * page_size);
  {
    const PageManager::DeferUnprotectScope scope;
    manager.UpdatePageWatchers<false>(address, 4 * page_size);
    DWORD old_protection = 0;
    Check(VirtualProtect(memory + page_size, page_size, PAGE_READWRITE,
                         &old_protection) != 0,
          "external protection change failed");
    g_protection_calls = 0;
    g_protection_ranges.clear();
    manager.UpdatePageWatchers<true>(address, 4 * page_size);
    for (uint64_t page = 0; page < 4; page++)
      Check(Protection(memory + page * page_size) == PAGE_READONLY,
            "protection reuse trusted stale applied state after an external change");
    Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
              g_protection_ranges[0].address == address + page_size &&
              g_protection_ranges[0].size == page_size,
          "external change was not re-protected independently of its unchanged neighbours");

    manager.UpdatePageWatchers<false>(address, 4 * page_size);
    Check(VirtualProtect(memory, 4 * page_size, PAGE_READWRITE, &old_protection) != 0,
          "second external protection change failed");
    g_protection_calls = 0;
    g_protection_ranges.clear();
    manager.UpdatePageWatchers<true>(address, 4 * page_size);
    Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
              g_protection_ranges[0].address == address &&
              g_protection_ranges[0].size == 4 * page_size,
          "host protection proof survived into the next watcher update");
    for (uint64_t page = 0; page < 4; page++)
      Check(Protection(memory + page * page_size) == PAGE_READONLY,
            "external protection change left a watched page writable");
  }
  Check(g_protection_calls == 1, "a pending release undid externally restored protection");
  manager.UpdatePageWatchers<false>(address, 4 * page_size);
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// Rewatching before a deferred release reaches the host must reuse the read-only protection.
// A newly watched neighbouring page still needs a real host call before this method returns.
void TestReuseAppliedProtection() {
  SetDeferMode(DeferMode::Verify);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  auto *memory = Allocate(page_size * 3);
  const auto address = reinterpret_cast<uint64_t>(memory);
  manager.UpdatePageWatchers<true>(address, page_size);
  manager.UpdatePageWatchers<true>(address + 2 * page_size, page_size);
  {
    const PageManager::DeferUnprotectScope scope;
    manager.UpdatePageWatchers<false>(address, page_size);
    manager.UpdatePageWatchers<false>(address + 2 * page_size, page_size);
    g_protection_calls = 0;
    g_protection_ranges.clear();
    manager.UpdatePageWatchers<true>(address, page_size);
    Check(g_protection_calls == 0 && Protection(memory) == PAGE_READONLY,
          "re-watching a pending release repeated an already applied host protection");
    manager.UpdatePageWatchers<false>(address, page_size);
    manager.UpdatePageWatchers<true>(address, 3 * page_size);
    Check(g_protection_calls == 1 && g_protection_ranges.size() == 1 &&
              g_protection_ranges[0].address == address + page_size &&
              g_protection_ranges[0].size == page_size,
          "protection reuse either missed a new watch or re-protected unchanged neighbours");
    Check(Protection(memory) == PAGE_READONLY &&
              Protection(memory + page_size) == PAGE_READONLY &&
              Protection(memory + 2 * page_size) == PAGE_READONLY,
          "a watched page was left writable before returning from a reused protection");
  }
  Check(g_protection_calls == 1, "a stale deferred release undid protection reuse");
  manager.UpdatePageWatchers<false>(address, 3 * page_size);
  for (uint64_t page = 0; page < 3; page++)
    Check(IsWritable(memory + page * page_size), "reuse left a released page protected");
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}
#endif

// More pending spans than the batch holds: the extra ones apply at once, the rest at scope end.
void TestDeferredBatchOverflow() {
  SetDeferMode(DeferMode::On);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  constexpr uint64_t pages = 40 * 80; // 40 spans, 80 pages apart (beyond the merge gap)
  auto *memory = Allocate(page_size * pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  for (uint64_t span = 0; span < 40; span++) {
    manager.UpdatePageWatchers<true>(address + span * 80 * page_size, page_size);
  }
  const auto before = PageManager::GetDeferStats();
  {
    const PageManager::DeferUnprotectScope scope;
    for (uint64_t span = 0; span < 40; span++) {
      manager.UpdatePageWatchers<false>(address + span * 80 * page_size, page_size);
    }
    uint32_t applied = 0;
    for (uint64_t span = 0; span < 40; span++) {
      applied += IsWritable(memory + span * 80 * page_size) ? 1u : 0u;
    }
    const auto during = PageManager::GetDeferStats();
    Check(during.overflows - before.overflows == applied && applied != 0 && applied < 40,
          "overflowing spans were not applied at once, or all were");
  }
  for (uint64_t span = 0; span < 40; span++) {
    Check(IsWritable(memory + span * 80 * page_size), "overflowed batch lost a pending span");
  }
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// Stress: threads watch and release pages at random, half their releases deferred. While a thread
// holds a watcher the host page must never be writable (never looser than the counts), and
// everything ends writable. In verify mode the manager checks itself after every change.
void TestDeferredReleaseStress(DeferMode mode) {
  SetDeferMode(mode);
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  constexpr uint64_t pages = 16;
  auto *memory = Allocate(page_size * pages);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto before = PageManager::GetDeferStats();
  std::atomic<uint32_t> looser{0};
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 6; t++) {
    threads.emplace_back([&, t] {
      std::mt19937 rng(1234 + t);
      for (uint32_t i = 0; i < 3000; i++) {
        const auto page = rng() % pages;
        const auto run = 1 + rng() % 3;
        const auto count = std::min<uint64_t>(run, pages - page);
        const auto begin = address + page * page_size;
        manager.UpdatePageWatchers<true>(begin, count * page_size);
        for (uint64_t p = 0; p < count; p++) {
          if (IsWritable(memory + (page + p) * page_size)) {
            looser++;
          }
        }
        if ((rng() & 1u) != 0) {
          const PageManager::DeferUnprotectScope scope;
          manager.UpdatePageWatchers<false>(begin, count * page_size);
          if ((rng() & 3u) == 0) {
            std::this_thread::yield();
          }
        } else {
          manager.UpdatePageWatchers<false>(begin, count * page_size);
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  Check(looser.load() == 0, "a watched page was writable on the host");
  for (uint64_t page = 0; page < pages; page++) {
    Check(IsWritable(memory + page * page_size), "stress left a page protected");
  }
  if (mode == DeferMode::Verify) {
    const auto after = PageManager::GetDeferStats();
    Check(after.verify_checks > before.verify_checks, "verify mode made no checks");
    Check(after.verify_mismatches == before.verify_mismatches, "verify mode found mismatches");
  }
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

// The spin-then-park lock: mutual exclusion under contention with long holds, and parking.
void TestParkingLock() {
  Libs::Graphics::ParkingSpinLock lock;
  uint64_t counter = 0;
  std::atomic<uint32_t> inside{0};
  std::atomic<uint32_t> overlaps{0};
  std::atomic<uint32_t> parks{0};
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < 8; t++) {
    threads.emplace_back([&, t] {
      for (uint32_t i = 0; i < 2000; i++) {
        if (!lock.try_lock()) {
          parks += lock.LockContended() ? 1u : 0u;
        }
        if (inside.fetch_add(1) != 0) {
          overlaps++;
        }
        counter++;
        if (t == 0 && i % 64 == 0) {
          // A long hold, past the spin budget: waiters park.
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        inside.fetch_sub(1);
        lock.unlock();
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  Check(overlaps.load() == 0 && counter == 8u * 2000u,
        "parking lock let two holders in");
  Check(!Libs::Graphics::TrackerLockParkEnabled() || parks.load() != 0,
        "long holds never made a waiter park");
  std::printf("  parking lock: %u parks over %llu acquisitions\n", parks.load(),
              static_cast<unsigned long long>(counter));
}

[[noreturn]] void RunDeathCase(const char *name) {
  PageManager manager;
  const auto page_size = manager.GetPageSize();
  if (std::strcmp(name, "invalid-range") == 0) {
    manager.UpdatePageWatchers<true>((1ull << 40u) - 1, 2);
  } else if (std::strcmp(name, "unknown-untrack") == 0) {
    manager.UpdatePageWatchers<false>(0x1000, page_size);
  } else if (std::strcmp(name, "destructor-watch") == 0) {
    auto doomed = std::make_unique<PageManager>();
    auto *memory = Allocate(page_size);
    const auto address = reinterpret_cast<uint64_t>(memory);
    doomed->UpdatePageWatchers<true>(address, page_size);
    doomed.reset();
  } else if (std::strcmp(name, "known-write-underflow") == 0) {
    auto *memory = Allocate(page_size);
    const auto address = reinterpret_cast<uint64_t>(memory);
    manager.UpdatePageWatchers<true>(address, page_size);
    manager.UpdatePageWatchers<false>(address, page_size);
    manager.UpdatePageWatchers<false>(address, page_size);
  } else if (std::strcmp(name, "read-overflow") == 0) {
    auto *memory = Allocate(page_size);
    const auto address = reinterpret_cast<uint64_t>(memory);
    RegionBits mask;
    const auto region_base = address & ~(TRACKER_REGION_SIZE - 1);
    const auto page = static_cast<size_t>((address - region_base) / page_size);
    mask.Set(page);
    manager.UpdatePageWatchersForRegion<true, true>(region_base, mask);
    manager.UpdatePageWatchersForRegion<true, true>(region_base, mask);
  } else if (std::strcmp(name, "write-overflow") == 0) {
    auto *memory = Allocate(page_size);
    const auto address = reinterpret_cast<uint64_t>(memory);
    for (uint32_t count = 0; count < 128; count++) {
      manager.UpdatePageWatchers<true>(address, page_size);
    }
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
        "death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 322 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "death case did not use the PageManager fatal exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
    ::execl("/proc/self/exe", "PageManagerTests", "--death", name, nullptr);
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (322 & 0xff);
  const bool fatal_signal = WIFSIGNALED(status);
  Check(fatal_exit || fatal_signal,
        "death case did not use the PageManager fatal exit");
#endif
}

void TestFatalPaths() {
  for (const char *name :
       {"invalid-range", "unknown-untrack", "destructor-watch",
        "known-write-underflow", "read-overflow", "write-overflow"}) {
    CheckDeathCase(name);
  }
}

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  if (const auto *reuse = std::getenv("KYTY_PAGE_PROTECT_REUSE");
      reuse != nullptr && std::strcmp(reuse, "1") == 0) {
    TestReuseAfterExternalProtection();
    TestReuseAppliedProtection();
  }
#endif
  TestWatchAndUnwatch();
  TestSharedWatcherCounts();
  TestCrossRegionRange();
  TestBatchedWatcherRanges();
  TestRegionMaskWatcherRanges();
  TestRegionEndpointBatching();
  TestCountWatchedPages();
  TestReadWriteWatcherInteractions();
  TestFatalPaths();

  // The synchronous paths again, checked by verify mode after every change.
  SetDeferMode(DeferMode::Verify);
  const auto verify_before = PageManager::GetDeferStats();
  TestWatchAndUnwatch();
  TestSharedWatcherCounts();
  TestCrossRegionRange();
  TestBatchedWatcherRanges();
  TestRegionMaskWatcherRanges();
  TestRegionEndpointBatching();
  TestReadWriteWatcherInteractions();
  const auto verify_after = PageManager::GetDeferStats();
  Check(verify_after.verify_checks > verify_before.verify_checks &&
            verify_after.verify_mismatches == verify_before.verify_mismatches,
        "verify mode rejected a synchronous watcher change");

  for (const auto mode : {DeferMode::Off, DeferMode::On, DeferMode::Verify}) {
    TestDeferredReleaseWaitsForScope(mode);
    TestDeferredReleaseRacingWatch(mode);
    TestDeferredReleaseStress(mode);
  }
  TestDeferredBatchOverflow();
  TestParkingLock();
  const auto stats = PageManager::GetDeferStats();
  Check(stats.verify_mismatches == 0, "verify mode found mismatches");
  std::printf("  deferred unprotect: %llu spans, %llu host calls, %llu settled, %llu overflows, "
              "%llu verify checks\n",
              static_cast<unsigned long long>(stats.spans),
              static_cast<unsigned long long>(stats.calls),
              static_cast<unsigned long long>(stats.settled),
              static_cast<unsigned long long>(stats.overflows),
              static_cast<unsigned long long>(stats.verify_checks));
  std::puts("PageManagerTests: all cases passed");
  return 0;
}
