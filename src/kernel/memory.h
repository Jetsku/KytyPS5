#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/virtualMemory.h"

#include <string>
#include <utility>
#include <vector>

namespace Libs::Graphics {
class RenderContext;
enum class PageFaultAccess;
} // namespace Libs::Graphics

namespace Libs::LibKernel::Memory {

void Initialize();
void Shutdown();

// Guest memory usage for the performance panel. Takes only bookkeeping locks, so it never
// waits on the GPU; call it only while the memory subsystem is initialized.
struct DebugStats {
	uint64_t direct_total        = 0;
	uint64_t direct_allocated    = 0; // sceKernelAllocateDirectMemory.
	uint64_t pooled_allocated    = 0; // Memory pool expansions.
	uint64_t automatic_allocated = 0; // Extended memory.
	uint64_t direct_mapped       = 0;
	uint64_t flexible_total      = 0;
	uint64_t flexible_used       = 0;
	uint64_t pool_committed      = 0;
	uint64_t cpu_page_entries       = 0; // 2 MiB page-table entries in use.
	uint64_t gpu_page_entries       = 0;
	uint64_t cpu_page_entries_total = 0;
	uint64_t gpu_page_entries_total = 0;
};
[[nodiscard]] DebugStats GetDebugStats();

struct Lifecycle {
	static constexpr const char* name       = "Memory";
	static constexpr auto        initialize = Libs::LibKernel::Memory::Initialize;
	static constexpr auto        shutdown   = Libs::LibKernel::Memory::Shutdown;
};

using callback_func_t = void (*)(uintptr_t addr, size_t size);

constexpr uint32_t KERNEL_MAXIMUM_NAME_LENGTH = 32;

// End (exclusive) of the guest address space that Initialize() reserves on the host
// (memoryAddressSpace.inc: HOST_USER_MAX + 1). Code that must place a host reservation outside the
// guest range before it is reserved (main.cpp's early CUDA initialization) derives its bound from
// this, so the two cannot drift apart.
constexpr uint64_t GUEST_ADDRESS_SPACE_END = 0xfc00000000ull;

// The AMM virtual-address window (sceAmprAmmGetVirtualAddressRanges): the top 64 GiB of the guest
// user range, below the 1 TiB the GPU tables address. Ordinary mappings never take addresses here
// (FindGuestFreeRange, ReserveAligned), so AMM map commands at fixed addresses cannot replace a
// mapping the kernel placed there. With the window at 0x1000000000 Miles Morales' streaming
// pages replaced its own direct-memory heap at that address.
constexpr uint64_t kAmmWindowBase = 0x00ec00000000ull;
constexpr uint64_t kAmmWindowSize = 0x001000000000ull;

struct VirtualQueryInfo {
	uintptr_t start;
	uintptr_t end;
	uint64_t  offset;
	int32_t   protection;
	int32_t   memory_type;
	uint32_t  is_flexible  : 1;
	uint32_t  is_direct    : 1;
	uint32_t  is_stack     : 1;
	uint32_t  is_pooled    : 1;
	uint32_t  is_committed : 1;
	uint32_t  is_gpu_prt   : 1;
	uint32_t  amm_usage    : 1;
	uint32_t  reserved     : 1;
	char      name[KERNEL_MAXIMUM_NAME_LENGTH];
	uint8_t   gpu_mask_id;
	uint8_t   reserved2;
};

static_assert(sizeof(VirtualQueryInfo) == 72, "VirtualQueryInfo struct size is incorrect");

struct KernelBatchMapEntry {
	void*         start;
	uint64_t      offset;
	uint64_t      length;
	unsigned char protection;
	unsigned char type;
	int16_t       reserved;
	int32_t       operation;
};

static_assert(sizeof(KernelBatchMapEntry) == 32, "KernelBatchMapEntry struct size is incorrect");

struct KernelMemoryPoolBatchEntry {
	uint32_t op;
	uint32_t flags;
	union {
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} commit;
		struct {
			void*    addr;
			uint64_t len;
		} decommit;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
		} protect;
		struct {
			void*    addr;
			uint64_t len;
			uint8_t  prot;
			uint8_t  type;
		} type_protect;
		struct {
			void*    dst;
			void*    src;
			uint64_t len;
		} move;
		uintptr_t padding[3];
	};
};

static_assert(sizeof(KernelMemoryPoolBatchEntry) == 32,
              "KernelMemoryPoolBatchEntry struct size is incorrect");

struct KernelMemoryPoolBlockStats {
	int32_t available_flushed_blocks;
	int32_t available_cached_blocks;
	int32_t allocated_flushed_blocks;
	int32_t allocated_cached_blocks;
};

static_assert(sizeof(KernelMemoryPoolBlockStats) == 16,
              "KernelMemoryPoolBlockStats struct size is incorrect");

void                   RegisterCallbacks(callback_func_t alloc_func, callback_func_t free_func);
void                   SetFlexibleMemorySize(uint64_t size);
// Upstream 259a40abf: an AMM unmap that accepts holes (an AMPR AMM unmap command), 16 KiB
// aligned. Upstream's MapAutomaticMemory/automatic AllocateDirectMemory are not in this line.
int UnmapAmmMemory(uint64_t vaddr, size_t size);
bool                   TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size);
bool                   TryReadBacking(uint64_t vaddr, void* data, uint64_t size);
// The AMM engine's pending maps (libAmpr.cpp): whether a queued or running AMM map command still has
// to map the address (the host fault handler waits for it, KYTY_FAULT_AMM_WAIT_MS).
void                   SetPendingAmmMapHook(bool (*covers)(uint64_t vaddr));
[[nodiscard]] bool     PendingAmmMapCovers(uint64_t vaddr);
// TryReadBacking whose destination may hold partial bytes when it fails (the caller discards it
// then): any size through a per-thread mapping record, without the mapping lock (see
// GuestBackingStore::TryReadBackingDirect). Any thread.
bool                   TryReadBackingDirect(uint64_t vaddr, void* data, uint64_t size);
// The direct-memory backing alias of [vaddr, vaddr + size) when one mapping holds the whole range,
// or nullptr (not direct memory, or spanning mappings). The alias stays mapped and writable for
// the process lifetime: reading it never faults, whatever the guest view's protection, and after
// the guest unmaps the range it shows whatever that backing then holds. Any thread.
[[nodiscard]] const void* GuestBackingAlias(uint64_t vaddr, uint64_t size);
// The always-RW backing alias as one host range (KYTY_INPLACE imports windows of it): its base
// address and size, 0 before the address space exists.
[[nodiscard]] uint64_t GuestBackingBase();
[[nodiscard]] uint64_t GuestBackingSize();
bool                   TryReadGpuCleanBacking(uint64_t vaddr, void* data, uint64_t size);
// TryReadGpuCleanBacking that also returns the XXH3-64 digest of the bytes read. Inside a
// draw-prep preparation the read is certified by that digest instead of its bytes
// (DrawPrep::ReadSet::RecordDigest): only for bytes the preparation merely hashes.
bool TryReadGpuCleanBackingDigest(uint64_t vaddr, void* data, uint64_t size, uint64_t& digest);
// In-place variants (KYTY_BACKING_INPLACE, default on): the same clean gate as
// TryReadGpuCleanBacking, then the backing bytes are compared or hashed where they are, through a
// lock-free mapping lookup, instead of being copied into a destination first. The bytes seen are
// exactly those a TryReadGpuCleanBacking at the same moment would have returned.
[[nodiscard]] bool BackingInPlaceEnabled();
enum class BackingCompare : uint8_t { Unavailable, Equal, Different };
struct InPlaceStats {
	uint32_t inspected = 0; // ranges inspected in place
	uint32_t locked    = 0; // of those, inspected under the mapping lock
};
// Outside draw-prep preparations only (no active recorder). Unavailable when the range is not
// clean for a backing read or has no backing.
[[nodiscard]] BackingCompare CompareGpuCleanBacking(uint64_t vaddr, const void* expected,
                                                    uint64_t size, InPlaceStats* stats = nullptr);
// The XXH3-64 digest of the clean backing bytes. Inside a draw-prep preparation the read is gated
// and certified by that digest exactly as TryReadGpuCleanBackingDigest does, without a copy.
bool HashGpuCleanBacking(uint64_t vaddr, uint64_t size, uint64_t& digest,
                         InPlaceStats* stats = nullptr);
// KYTY_DRAW_PREP_LEAN digest cache (graphics/host_gpu/renderer/drawPrep/codeDigestCache.h):
// whether a digest of [vaddr, vaddr + size) may be cached: the GPU-touched map is maintained
// (KYTY_CP_SEQ=1), no page of the range was ever GPU-touched, and the range lies in at most two
// backing mapping regions, whose mapping tags it returns. Any thread.
[[nodiscard]] bool CodeDigestCacheable(uint64_t vaddr, uint64_t size, uint64_t& tag_begin,
                                       uint64_t& tag_end);
// The XXH3-64 digest of the backing bytes, not gated and not recorded (KYTY_DRAW_PREP_LEAN_VERIFY).
[[nodiscard]] bool HashBackingUnrecorded(uint64_t vaddr, uint64_t size, uint64_t& digest);
// The clean verdict of TryReadGpuCleanBacking without reading bytes (GPU thread; true for
// ranges outside GPU memory).
[[nodiscard]] bool     IsGpuCleanForRead(uint64_t vaddr, uint64_t size);
// After a clean coherence-log check of the range over [since, now] (DrawPrep::Validate,
// KYTY_DRAW_PREP_CLEAN_SINCE): IsGpuCleanForRead with page verdicts kept since `since`.
[[nodiscard]] bool     IsGpuCleanForReadSince(uint64_t vaddr, uint64_t size, uint64_t since);
// Whether [vaddr, vaddr + size) lies in guest memory mapped for the GPU (the renderer's mapped
// ranges, the same ones its fault handler resolves). False before InstallGpuResources.
[[nodiscard]] bool     IsGpuMapped(uint64_t vaddr, uint64_t size);
// Whether every byte of [vaddr, vaddr + size) is mapped (committed guest ranges, no reservation or gap).
[[nodiscard]] bool     IsCommittedSpan(uint64_t vaddr, uint64_t size);
// The unhandled-fault report: what the kernel and the GPU tracker know about an address.
[[nodiscard]] std::string DescribeAddressForCrash(uint64_t vaddr);
// The guest virtual range containing vaddr: start, size and the name the guest gave it (diagnostics).
[[nodiscard]] bool     DescribeVirtualRange(uint64_t vaddr, uint64_t* start, uint64_t* size, char* name,
                                            uint64_t name_size);
// The committed, non-private, GPU-mapped allocation containing vaddr; false for anything else.
[[nodiscard]] bool     GetGpuBufferMapping(uint64_t vaddr, uint64_t* start, uint64_t* size);
// May submit/wait only at GPU preparation boundaries, outside texture-cache/tracker locks.
bool                   SynchronizeGpuBackingForRead(uint64_t vaddr, uint64_t size);
// KYTY_SRT_STALE_IMAGE_READS=1 (default off): a resource-table read on the GPU thread that the clean
// gate refused only because GPU-modified images overlap the range, none of which is still a source
// of these guest bytes (TextureCache::IsRegionGpuOwnedForRead: a CPU write superseded it), reads the
// backing. False otherwise (the caller keeps its normal synchronization path).
[[nodiscard]] bool     SrtStaleImageReadsEnabled();
bool                   TryReadGpuBackingOverStaleImages(uint64_t vaddr, void* data, uint64_t size);
// Diagnostics: RenderContext::DescribeGpuBackingForRead (GPU thread; "no GPU" without a renderer).
std::string            DescribeGpuBackingForRead(uint64_t vaddr, uint64_t size);
bool                   TryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
// Upstream's descriptor-dependency reader (57cf08688/783d9f8d6): on the GPU thread, GPU-dirty
// buffer bytes are read back first, then the backing is read.
bool                   TryReadBufferBacking(uint64_t vaddr, void* data, uint64_t size);
// GPU thread (KYTY_SRT_GPU_FILL): the guest dword at vaddr is GPU-written bytes one registered
// buffer holds (BufferCache::GpuFillSource), so a GPU copy reads its current value where a host
// read would wait for the GPU.
[[nodiscard]] bool     IsGpuFillableWord(uint64_t vaddr);
bool                   CopySparseMemory(uint64_t vaddr, void* data, uint64_t size);
[[nodiscard]] uint64_t ClampRangeSize(uint64_t vaddr, uint64_t size);
// ClampRangeSize's answer without its log or exit (0: not committed, or no ranges yet). Any thread.
[[nodiscard]] uint64_t ClampRangeSizeQuiet(uint64_t vaddr, uint64_t size);
// The committed parts of [vaddr, vaddr + size), adjacent ones merged, in order, as (start, size):
// where a GPU write over a range with holes in its mappings (sparse memory) lands. At most
// max_spans; false (and none) when there are more. Any thread.
[[nodiscard]] bool CommittedSpans(uint64_t vaddr, uint64_t size, size_t max_spans,
                                  std::vector<std::pair<uint64_t, uint64_t>>* spans);
// From vaddr to the end of the last committed byte of [vaddr, vaddr + size); 0 when none is.
[[nodiscard]] uint64_t CommittedExtent(uint64_t vaddr, uint64_t size);
// The guest virtual ranges' change generation (advanced before every change starts; 0 before
// they exist). Unchanged since a read made before a ClampRangeSize call: the call's answer holds.
[[nodiscard]] uint64_t VirtualRangesGeneration() noexcept;
void                   WriteBacking(uint64_t vaddr, const void* data, uint64_t size) noexcept;
void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
// RenderContext::PrepareHostOverwrite (KYTY_APR_PREPARE_WRITE): false when nothing was watched.
bool                   PrepareHostOverwrite(uint64_t vaddr, uint64_t size);
void                   InstallGpuResources(Graphics::RenderContext* renderer) noexcept;
[[nodiscard]] bool HandleGpuFault(Graphics::PageFaultAccess access, uint64_t fault_vaddr) noexcept;
// KYTY_TRANSIENT_UNMAP_RETRY (default off): for a fault in a guest range briefly unmapped while the
// emulator replaced its mapping (MAP_FIXED over a mapping, or a Windows file view split by a
// partial unmap), waits until it is mapped again and returns the range's id. 0 for any other fault
// and while the switch is off.
[[nodiscard]] uint64_t WaitForTransientUnmap(uint64_t fault_vaddr) noexcept;
// Whether such a fault can be retried (false once the same page keeps faulting).
[[nodiscard]] bool RetryTransientUnmapFault(uint64_t fault_vaddr, uint64_t window) noexcept;
void               SetTransientUnmapRetryForTest(bool enabled) noexcept;
// Store in place (KYTY_STORE_IN_PLACE / KYTY_STORE_CLEAN; RenderContext::TryStoreInPlace): whether a
// game thread's write fault at fault_vaddr may be completed by writing the store's bytes to both
// copies, and that write (false to decline: the fault then takes the normal path).
[[nodiscard]] bool IsStoreInPlaceCandidate(uint64_t fault_vaddr) noexcept;
[[nodiscard]] bool TryStoreInPlace(uint64_t pc, uint64_t fault_vaddr, uint64_t vaddr, const void* data,
                                   uint64_t size) noexcept;

int KYTY_SYSV_ABI KernelMapNamedFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags,
                                               const char* name);
int KYTY_SYSV_ABI KernelMapFlexibleMemory(void** addr_in_out, size_t len, int prot, int flags);
int KYTY_SYSV_ABI KernelSetVirtualRangeName(const void* addr, uint64_t len, const char* name);
int KYTY_SYSV_ABI KernelClearVirtualRangeName(const void* addr, uint64_t len);
int KYTY_SYSV_ABI KernelMunmap(uint64_t vaddr, size_t len);
size_t KYTY_SYSV_ABI KernelGetDirectMemorySize();
int KYTY_SYSV_ABI    KernelAvailableDirectMemorySize(int64_t search_start, int64_t search_end,
                                                     size_t alignment, int64_t* phys_addr_out,
                                                     size_t* size_out);
int KYTY_SYSV_ABI    KernelGetPageTableStats(int* cpu_total, int* cpu_available, int* gpu_total,
                                             int* gpu_available);
int KYTY_SYSV_ABI KernelAllocateDirectMemory(int64_t search_start, int64_t search_end, size_t len,
                                             size_t alignment, int memory_type,
                                             int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelAllocateMainDirectMemory(size_t len, size_t alignment, int memory_type,
                                                 int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelCheckedReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelReleaseDirectMemory(int64_t start, size_t len);
int KYTY_SYSV_ABI KernelMapDirectMemory(void** addr, size_t len, int prot, int flags,
                                        int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapDirectMemory2(void** addr, size_t len, int type, int prot, int flags,
                                         int64_t direct_memory_start, size_t alignment);
int KYTY_SYSV_ABI KernelMapNamedDirectMemory(void** addr, size_t len, int prot, int flags,
                                             int64_t direct_memory_start, size_t alignment,
                                             const char* name);
int KYTY_SYSV_ABI KernelSetPrtAperture(int index, void* addr, size_t len);
int KYTY_SYSV_ABI KernelGetPrtAperture(int index, void** addr, size_t* len);
int KYTY_SYSV_ABI KernelIsAddressSanitizerEnabled();
int KYTY_SYSV_ABI KernelQueryMemoryProtection(void* addr, void** start, void** end, int* prot);
int KYTY_SYSV_ABI KernelDirectMemoryQuery(int64_t offset, int flags, void* info, size_t info_size);
int KYTY_SYSV_ABI KernelVirtualQuery(const void* addr, int flags, VirtualQueryInfo* info,
                                     uint64_t info_size);
int KYTY_SYSV_ABI KernelIsStack(void* addr, void** start, void** end);
int KYTY_SYSV_ABI KernelReserveVirtualRange(void** addr, size_t len, int flags, size_t alignment);
int KYTY_SYSV_ABI KernelAvailableFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelConfiguredFlexibleMemorySize(size_t* size);
int KYTY_SYSV_ABI KernelMprotect(const void* addr, size_t len, int prot);
int KYTY_SYSV_ABI KernelMtypeprotect(const void* addr, size_t len, int type, int prot);
int KYTY_SYSV_ABI KernelBatchMap(KernelBatchMapEntry* entries, int num_entries,
                                 int* num_entries_out);
int KYTY_SYSV_ABI KernelBatchMap2(KernelBatchMapEntry* entries, int num_entries,
                                  int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolExpand(int64_t search_start, int64_t search_end, size_t len,
                                         size_t alignment, int64_t* phys_addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolReserve(void* addr_in, size_t len, size_t alignment, int flags,
                                          void** addr_out);
int KYTY_SYSV_ABI KernelMemoryPoolCommit(void* addr, size_t len, int type, int prot, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolDecommit(void* addr, size_t len, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolBatch(const KernelMemoryPoolBatchEntry* entries, int num_entries,
                                        int* num_entries_out, int flags);
int KYTY_SYSV_ABI KernelMemoryPoolGetBlockStats(KernelMemoryPoolBlockStats* output,
                                                size_t                      output_size);

uint64_t AllocateProgramMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name);
// Loaded modules' segments use flexible memory without taking its backing.
// Charged only with KYTY_UPSTREAM_MEMORY_DEFAULTS=1; returns the size charged (0 otherwise).
uint64_t ChargeFlexibleMemory(uint64_t size);
void ReleaseFlexibleMemory(uint64_t size);

void SetProgramMemoryProtection(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
uint64_t AllocateRuntimeMemory(uint64_t search_addr, uint64_t size,
                               Common::VirtualMemory::Mode mode, const char* name,
                               bool fixed = false);
uint64_t AllocateGuestStackMemory(uint64_t search_addr, uint64_t size,
                                  Common::VirtualMemory::Mode mode, const char* name);
bool     ProtectGuestMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode,
                            Common::VirtualMemory::Mode* old_mode = nullptr);
// Transient PageManager watch state; does not change the guest mapping's semantic protection.
bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size, Common::VirtualMemory::Mode mode);
// KYTY_TRACKER_SPARE: a live thread's stack in memory the game gave it; the page tracker's
// protections leave it writable until it is removed (GuestAddressSpace::AddLiveStack).
void AddLiveThreadStack(uint64_t vaddr, uint64_t size);
void RemoveLiveThreadStack(uint64_t vaddr, uint64_t size);
bool FreeGuestMemory(uint64_t vaddr, uint64_t size);

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
// KYTY_CLAMP_RANGE_MEMO outcomes of the calling thread's ClampRangeSize calls.
struct TestClampTotals {
	uint64_t hits              = 0;
	uint64_t misses            = 0;
	uint64_t verify_checks     = 0;
	uint64_t verify_mismatches = 0;
	uint64_t verify_races      = 0;
};
TestClampTotals TestClampRangeMemoTotals();
// KYTY_CLAMP_RANGE_BITMAP for the following map changes and lookups.
void TestSetClampRangeBitmap(bool enabled);
void     TestSetGpuRangeCallback(void (*callback)(bool mapped, uint64_t addr, size_t size));
void     TestBeforeNextBackingMap(callback_func_t callback);
void     TestSetBackingReadCallback(callback_func_t callback);
// The backing store's sparse read without TryReadSparseBacking's range checks.
bool     TestTryReadSparseBacking(uint64_t vaddr, void* data, uint64_t size);
bool     TestPlaceholderRangeIsFree(uint64_t vaddr, uint64_t size);
bool     TestGuestAddressRangeIsOwned(uint64_t vaddr, uint64_t size);
bool     TestGuestBackingOutsideAddressSpace();
uint64_t TestGuestBackingSize();
uint64_t TestGuestBackingBase();
bool     TestGuestFreeRangeBounds();
#endif

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_MEMORY_H_ */
