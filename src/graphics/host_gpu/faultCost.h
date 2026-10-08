#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTCOST_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTCOST_H_

#include <cstdint>

// What guest-memory write tracking costs on this PC.
//
// The GPU caches write-protect the guest pages they mirror and catch the guest's first write to
// each: a host exception (Windows: a vectored exception handler; Linux: SIGSEGV, or SIGBUS with
// userfaultfd write-protection) and a protection change (VirtualProtect, mprotect or
// UFFDIO_WRITEPROTECT) per fault, and another protection change when the page is uploaded and
// protected again. At the Astro Bot Sky Garden that is ~1,300-1,700 faults and ~800-900 protection
// calls per frame. Their cost differs by more than 10x between PCs: ~5 us per fault here (Windows,
// RTX 3090 + 7950X3D), ~80 us on a Linux PC (mprotect takes the mmap lock for writing, which every
// page fault of the process then waits for).
//
// Startup benchmark (always on, a few ms; Windows and Linux): a write-fault round trip on a
// write-protected page of a shared-memory view (the kind of mapping guest memory is), and the
// protection calls (read-only and read-write, 1 and 64 pages). Logged as "Kyty fault cost: ...",
// with the platform facts that decide these costs ("Kyty platform: ...": Windows code integrity /
// hypervisor / hooked ntdll entry points / injected modules; Linux kernel, vm.max_map_count,
// mappings, userfaultfd features).
//
// Live numbers (always on): every guest fault the tracker resolves and every protection call is
// timed (two clock reads). KYTY_FAULT_COST_LOG=<seconds> (default 60, 0 = off) logs one line with
// the counts and average costs of the last period. Model() combines the startup benchmark (the part
// of a fault outside the handler) with the live in-handler averages.
//
// Slow-PC simulation (diagnostic, default off, live switches): busy-waits added to this PC's costs.
//   KYTY_SIM_FAULT_US=<us>          every guest fault the tracker resolves, before any lock
//   KYTY_SIM_PROTECT_US=<us>        every protection call that tightens (read-only, no-access)
//   KYTY_SIM_PROTECT_PAGE_NS=<ns>   ... plus this per page
//   KYTY_SIM_UNPROTECT_US=<us>      every protection call that makes pages writable
//   KYTY_SIM_UNPROTECT_PAGE_NS=<ns> ... plus this per page
// The protection waits hold one process-wide lock with the real call, as the OS serializes the
// protection changes of an address space (mmap_lock for writing on Linux).
//   KYTY_SIM_SERIAL=0               the fault wait runs in parallel on each thread instead of
//                                   holding that lock too (default 1: Linux, where the fault waits
//                                   for the mmap lock that every mprotect holds for writing)
//
// Fault map (diagnostic, KYTY_FAULT_MAP=1): which guest ranges fault, from which threads (the
// command processor, guest threads, other host threads) and code (guest code or the emulator), how
// often the same page faults again (same frame, next frame, every N frames), the protection calls
// per thread and direction, and how the in-handler time splits between the buffer cache, the
// texture cache and the deferred protection update. One block of "FaultMap" lines every
// KYTY_FAULT_MAP_SECONDS (default 10) on the command processor's flip.
namespace Libs::Graphics::FaultCost {

struct Benchmark {
	bool   valid             = false;
	double fault_us          = 0; // write to a protected page, handler makes it writable, retry
	double fault_handler_us  = 0; // of which the protection call inside the handler
	double protect_call_us   = 0; // read-only, one page
	double protect_page_us   = 0; // read-only, each further page
	double unprotect_call_us = 0; // read-write, one page
	double unprotect_page_us = 0; // read-write, each further page
	double clock_ns          = 0; // one steady_clock read
};

// Runs the startup benchmark once and logs it with the platform facts (later calls do nothing).
// Call it on one thread before the guest runs.
void RunStartupBenchmark();
[[nodiscard]] const Benchmark& StartupBenchmark();

// The cost model the adaptive tracking decisions use (microseconds).
struct CostModel {
	double fault_us        = 5.0; // one write fault, everything included
	double protect_call_us = 2.0; // one protection call
	double protect_page_us = 0.05;
	double tighten_fixed_us = -1.0; // live: a tightening call beyond its pages (-1: not yet)
};
[[nodiscard]] CostModel Model();

// How slow write tracking is on this PC, for decisions that must not feed back on themselves
// (KYTY_FAULT_AHEAD_ADAPT): 0 normal, 1 slow, 2 very slow. The fault handler's own time grows with
// the fault-ahead window (this PC: ~5 us per fault at 32 KiB, ~12 us at 256 KiB, ~72 us at 1 MiB),
// so it cannot choose the window. The tightening protection calls (read-only or no-access, made
// when uploads re-protect pages) keep their size whatever the window: their fixed part (time per
// call minus the startup benchmark's per-page cost) is ~4-8 us here at every window, ~18 us on the
// Linux PC whose mmap lock convoys and ~35 us in the slow-PC simulation.
//   level 1: above 10 us, level 2: above 16 us, each for 5 consecutive ~2 s periods (32+ calls);
//   seeded by the startup benchmark (uncontended tightening call above 2 / 5 us, or a fault round
//   trip above 10 / 20 us: memory integrity, a hooked VirtualProtect, a slow kernel).
//   Linux with mprotect tracking (KYTY_UFFD_WP off) starts at 2: every mprotect takes the mmap
//   lock for writing, the larger window shortens that convoy enough to hide it from the measure
//   above, and in the WSL2 benchmark 1 MiB windows stall the writers least (0.9 ms per frame
//   against 2.1 ms at 256 KiB and 35-44 ms at 32 KiB).
// The level only ever rises; "Kyty fault cost: write tracking is slow ..." logs each step.
[[nodiscard]] int SlowLevel() noexcept;

// The level logic on its own (SlowLevel uses one, on the command processor thread; tests).
class SlowLevelTracker {
public:
	static constexpr int PeriodsNeeded = 5;
	// Startup benchmark: uncontended tightening call and fault round trip (us).
	void Seed(double tighten_call_us, double fault_round_trip_us) noexcept;
	// A platform's floor (Linux mprotect: 2).
	void Raise(int level) noexcept {
		if (level > m_level) {
			m_level = level > 2 ? 2 : level;
		}
	}
	// One period's fixed tightening cost (us); returns the level.
	int  Update(double tighten_fixed_us) noexcept;
	[[nodiscard]] int Level() const noexcept { return m_level; }

private:
	int m_level     = 0;
	int m_streak[3] = {0, 0, 0};
};

// Time source of the live numbers and the fault map (steady_clock).
[[nodiscard]] uint64_t NowNs() noexcept;

// Live numbers (any thread).
void NoteFault(uint64_t handler_ns) noexcept;
void NoteProtect(bool unprotect, uint64_t pages, uint64_t ns) noexcept;

// BDA synchronization passes (BufferCache::SynchronizeBdaBuffersNow) and their upload bytes:
// 0 nothing to do, 1 hot runs only, 2 dirty-log pass, 3 full scan.
void NoteBdaPass(int kind, uint64_t bytes) noexcept;

// One more line for the periodic live log (other modules' counters; the loader's VRSQRTPS traps).
using PeriodicReporter = void (*)(double seconds, uint64_t frames);
void SetPeriodicReporter(PeriodicReporter reporter) noexcept;

// Once per guest flip on the command processor (live log period, fault map frames and output).
void AdvanceFrame() noexcept;
[[nodiscard]] uint32_t Frame() noexcept;

// Slow-PC simulation.
[[nodiscard]] bool SimEnabled() noexcept;
void               SimulateFault() noexcept;
// Busy-waits the simulated cost of a protection call, holding the simulation's address-space lock
// (released by SimProtectEnd).
void SimProtectBegin(bool unprotect, uint64_t pages) noexcept;
void SimProtectEnd() noexcept;

// Fault map.
[[nodiscard]] bool MapEnabled() noexcept;
// Who the calling thread is: 0 the command processor, 1 a guest thread, 2 another host thread;
// `name` gets a short name (guest threads). Set by the renderer (this file has no emulator deps).
using ThreadDescriber = int (*)(char* name, uint64_t size);
void SetThreadDescriber(ThreadDescriber describer) noexcept;
// The faulting instruction of the fault the calling thread is resolving (host exception handler).
void SetFaultInstruction(uint64_t rip) noexcept;
// The calling thread resolves a host exception in the exception handler: the faulting instruction
// runs again when the handler returns, so a write that leaves its page protected faults once more
// (KYTY_READBACK_SIDE_WRITES relies on that second fault). Direct HandleFault calls are not.
class RetriedFaultScope {
public:
	RetriedFaultScope() noexcept;
	~RetriedFaultScope();
	RetriedFaultScope(const RetriedFaultScope&)            = delete;
	RetriedFaultScope& operator=(const RetriedFaultScope&) = delete;

private:
	bool m_previous = false;
};
[[nodiscard]] bool InRetriedFault() noexcept;
enum class Part : uint8_t { Buffer = 0, Texture = 1, Reconcile = 2, Count = 3 };
struct FaultRecord {
	uint64_t address     = 0;
	uint64_t handler_ns  = 0;
	uint64_t part_ns[static_cast<int>(Part::Count)] {};
	bool     write       = true;
	bool     found_dirty = false; // the buffer tracker found the page CPU-dirty already
};
void MapFault(const FaultRecord& record) noexcept;
void MapProtect(bool unprotect, bool no_access, uint64_t pages, uint64_t ns) noexcept;
// Inside the tracker's write-fault resolution on this thread (protection calls count as "in fault").
void MapInFault(bool inside) noexcept;

} // namespace Libs::Graphics::FaultCost

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_FAULTCOST_H_
