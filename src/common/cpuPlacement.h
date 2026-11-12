#ifndef EMULATOR_SRC_COMMON_CPUPLACEMENT_H_
#define EMULATOR_SRC_COMMON_CPUPLACEMENT_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

// Where the emulator's threads run (KYTY_CPU_RESERVE).
//
// The command processor (CP) is the frame's critical path. In U54 its SMT sibling was busy 21.7% of
// the time, mostly with guest job threads, and a woken CP often waited for a CPU.
//
// KYTY_CPU_RESERVE=off (default): nothing changes.
// cp: the CP gets one physical core to itself, both logical processors, through Windows CPU sets
// (SetThreadSelectedCpuSets). Every other thread of the process keeps off that core through the
// process default CPU sets: guest threads, draw-prep workers, service and completion threads, the
// CP recorder, and driver threads alike.
// cp+recorder: the CP recorder thread (KYTY_CP_RECORDER=1) gets a second core the same way.
// stages (also 3): every GPU stage thread gets a whole core: the CP (resolver and execution), the
// CP recorder, and the CP sequencer (KYTY_CP_SEQ=1, the PM4 front end), as Erlou93/TheCruZ's
// gpuStageCores places their CP, resolve, execution and recorder threads (+18% at Yotei's gate on
// theirs). Draw-prep workers stay on the general processors.
//
// Details:
// - The allowed processors are read at runtime: the process affinity mask (Process Lasso pins
//   Astro Bot to 0-15, the V-cache CCD) and any process default CPU sets.
// - Only whole cores, every logical processor allowed, are reserved. They are chosen in the last
//   level cache that holds most allowed processors, by SchedulingClass (the CPPC preference),
//   avoiding the core of logical processor 0 (interrupts). KYTY_CPU_RESERVE_CORE=<logical
//   processor> picks the CP's core instead.
// - Reservation is skipped (and logged) when fewer than 6 logical processors would remain.
// - A monitor thread re-applies the layout when the process affinity changes (Process Lasso sets it
//   after the process has started), checking once a second.
// - CPU sets never override a hard thread affinity (SetThreadAffinityMask). Kyty sets none: guest
//   pthread affinities are only stored. The monitor scans the process's threads every 5 s for one
//   anyway (drivers, overlays): a hard affinity that meets the general processors is already
//   narrowed to them by the CPU sets; one confined to the reserved cores runs there regardless.
//   KYTY_CPU_RESERVE_REPIN=1 moves such threads to the general processors (default: counted and
//   logged only).
//
// KYTY_CPU_SETS=<logical processors of group 0> (default unset: no restriction), e.g. 16-31 or
// 0-7,16-23: every thread of the process runs on these processors (within the affinity mask),
// through the process default CPU sets, a soft affinity. With KYTY_CPU_RESERVE the CP's core is
// chosen among them and the other threads get the rest of them. Unlike a process affinity mask,
// Process Lasso's affinity rule does not reset it.
// KYTY_CPU_RESERVE_REASSERT=1 (default off): the monitor also checks once a second whether another
// process replaced the process default CPU sets this module applied, and applies its layout again.
// Process Lasso's CPU-set rule (SetProcessDefaultCpuSets, about 0.6 s after the process starts)
// otherwise silently undoes KYTY_CPU_RESERVE and KYTY_CPU_SETS for every thread but the CP and the
// recorder: they keep their thread-selected sets, so the CP stays confined to its core while other
// threads run there again. Without KYTY_CPU_SETS the other process's sets become the constraint the
// layout is computed in (Process Lasso's (0-31) leaves it unchanged).
//
// KYTY_VCACHE_PLACEMENT=auto|0|1 (default auto): dual-CCD X3D processors (7950X3D, 7900X3D,
// 9950X3D, ...; one CCD has 3D V-cache) without a Process Lasso-like rule.
// - The V-cache CCD is the L3 cache group that measures largest (GetLogicalProcessorInformationEx
//   cache relations), not the one with most cores or the lowest index. A chip counts as dual-CCD X3D
//   only when there are two or more L3 groups and the largest is at least 1.5 times every other
//   (96 against 32 MiB). One L3 (7800X3D, 9800X3D, Intel hybrid) or equal L3s (7950X): no change.
// - auto: only when no external CPU-set or affinity rule is active: the process affinity mask is
//   the whole system's, no process default CPU sets were set before startup, KYTY_CPU_SETS is
//   unset, and no other process replaces the process default CPU sets later. When one appears
//   (Process Lasso applies its rule about 0.6 s after the process starts) the placement is dropped
//   for good and the KYTY_CPU_RESERVE behaviour without it applies. 1: also with an external rule
//   (still inside the process affinity mask). 0: off.
// - Hot threads keep to the V-cache CCD through the process default CPU sets: guest threads (the
//   guest pthreads inherit them), the main guest thread, the CP, the recorder, draw-prep workers,
//   service, completion and present threads, and driver threads. KYTY_CPU_RESERVE's cores are taken
//   inside the V-cache CCD. Cold threads (PlaceCurrentThreadCold: shader and pipeline compile
//   workers, pipeline journal/cache savers, file I/O, audio3d playback, diagnostics and monitors)
//   get the other CCD through their thread-selected CPU sets.
// - KYTY_VCACHE_GUEST=all (default): guest threads (PlaceCurrentThreadGuest) run on every allowed
//   processor outside the reserved cores, both CCDs, through their thread-selected CPU sets.
//   hot: only the hot processors (int19.3-int20); cold: only the other CCD. With hot, Astro Bot's
//   threads got 14 logical processors and its Draw Geometry thread fell behind (Bug C).
// - One "Kyty V-cache placement" line logs the decision at startup and whenever it changes.
//
// Placement samples, with the profiler's aggregates, the hang trace or
// KYTY_CPU_PLACEMENT_SAMPLES=1: the CP, the recorder, guest threads, draw-prep workers and service
// threads note where they run.
// - FrameEvents: CpuPlacement{Cp,Guest,Host}Samples; CpuPlacementCpOffCore and
//   CpuPlacementRecorderOffCore (outside their reserved cores); CpuPlacement{Guest,Host}OnCpCore
//   (on the physical core of the CP's latest sample: SMT sharing); CpuPlacementHardAffinity
//   (threads found with a hard affinity other than the process's), CpuPlacementHardAffinityReserved
//   (of those, confined to reserved cores), CpuPlacementRepinned.
// - Hang trace: placement.csv, samples per role and logical processor each second.
namespace Common {

enum class ThreadRole : uint8_t { Cp = 0, Recorder = 1, Guest = 2, Host = 3, Count = 4 };
enum class CpuReserveMode : uint8_t { Off, Cp, CpRecorder, Stages };

// One logical processor as GetSystemCpuSetInformation reports it.
struct CpuSetInfo {
	uint32_t id               = 0;
	uint16_t group            = 0;
	uint8_t  logical          = 0; // index within the group
	uint8_t  core             = 0; // CoreIndex: shared by SMT siblings
	uint8_t  cache            = 0; // LastLevelCacheIndex
	uint8_t  scheduling_class = 0; // higher = preferred (CPPC)
	uint8_t  efficiency_class = 0; // higher = faster core class (hybrid CPUs: P above E); else 0
	bool     allowed          = false;
};

struct CpuLayout {
	bool                  reserved      = false; // a CP core is reserved
	int                   cp_core       = -1;
	int                   recorder_core = -1;
	int                   sequencer_core = -1;
	std::vector<uint32_t> cp_ids;            // CPU set ids of the CP's core
	std::vector<uint32_t> recorder_ids;      // of the recorder's core (cp+recorder, stages)
	std::vector<uint32_t> sequencer_ids;     // of the sequencer's core (stages)
	std::vector<uint32_t> general_ids;       // every other allowed processor
	uint64_t              cp_mask       = 0; // group 0 logical processors of the above
	uint64_t              recorder_mask = 0;
	uint64_t              sequencer_mask = 0;
	uint64_t              general_mask  = 0;
	std::string           description; // the choice, or why nothing was reserved
	// KYTY_VCACHE_PLACEMENT: general_ids/general_mask are the hot processors (the V-cache CCD
	// without the reserved cores) and the cold threads get these.
	bool                  vcache    = false;
	std::vector<uint32_t> cold_ids;
	uint64_t              cold_mask = 0;
};

// One L3 cache group: the logical processors sharing it (group-relative mask) and its size.
struct L3Cache {
	uint16_t group = 0;
	uint64_t mask  = 0;
	uint64_t bytes = 0;
};

// The L3 groups seen as a V-cache topology (group 0 only).
struct VCacheTopology {
	bool        dual         = false; // two or more L3 groups, the largest clearly larger
	uint32_t    groups       = 0;     // L3 groups in group 0
	uint64_t    vcache_mask  = 0;     // the logical processors of the largest L3
	uint64_t    other_mask   = 0;     // those of every smaller L3
	uint64_t    vcache_bytes = 0;
	uint64_t    other_bytes  = 0; // the largest of the smaller ones
	std::string description;
};

enum class VCacheMode : uint8_t { Auto, Off, On };

struct VCacheDecision {
	bool        engage = false;
	std::string reason; // why not, or what applies
};

inline constexpr uint32_t kMinGeneralProcessors = 6;

// KYTY_CPU_RESERVE_SCALE=1 (live, default off; KYTY_CORE_BUDGET_PROFILE turns it on when unset):
// KYTY_CPU_RESERVE's rules scaled to the CPU, from the core budget (common/coreBudget.h: the
// physical cores and logical processors the process may use; with the V-cache placement the whole
// process's, both CCDs).
// - A whole core is reserved for the CP only while at least KYTY_CPU_RESERVE_WHOLE_MIN_CORES
//   (default 8) cores are in the budget (one more per extra reserved core of cp+recorder and
//   stages, which otherwise fall back to cp). Below that the CP gets one logical processor to
//   itself (its SMT sibling stays general) when the budget has at least KYTY_CPU_RESERVE_LOGICAL_MIN
//   (default 8) logical processors; otherwise nothing is reserved. 6 cores with SMT: one logical;
//   4 cores / 8 threads: one logical; 6 cores without SMT: nothing; 8 cores and more: a whole core.
//   One logical processor needs no whole core, so an odd affinity mask still gets one.
// - The CP's core is chosen by the CPU's preferred-core ranking (CPPC SchedulingClass) first; the
//   core of logical processor 0 only loses ties.
// - Hybrid CPUs: only cores of the fastest efficiency class are candidates (never an E-core).
// - The full 7950X3D, the Process Lasso mask and the 8-core masks decide exactly as without it
//   (tested).
struct ReserveRules {
	bool     scaled          = false;
	uint32_t whole_min_cores = 8;
	uint32_t logical_min     = 8;
	// The budget the rules read; 0: computed from the table passed (ComputeVCacheLayout passes the
	// whole process's for its V-cache CCD table).
	uint32_t budget_cores   = 0;
	uint32_t budget_logical = 0;
};
[[nodiscard]] ReserveRules CurrentReserveRules(); // the switches now

// Pure (tested): the reserved cores for `sets` (only group 0 is considered). The cache holding most
// allowed processors is used; ties go to the cache with processors in `preferred_mask` (the V-cache
// CCD), then to the lower index.
[[nodiscard]] CpuLayout ComputeCpuLayout(const std::vector<CpuSetInfo>& sets, CpuReserveMode mode,
                                         int      forced_cp_logical,
                                         uint32_t min_general    = kMinGeneralProcessors,
                                         uint64_t preferred_mask = 0,
                                         const ReserveRules& rules = {});
// Pure (tested): the L3 groups as a V-cache topology (KYTY_VCACHE_PLACEMENT).
[[nodiscard]] VCacheTopology DetectVCache(const std::vector<L3Cache>& caches);
// Pure (tested): whether the V-cache placement applies. `external_rule`: an affinity or CPU-set
// rule of another process was seen; `cpu_sets`: KYTY_CPU_SETS is set.
[[nodiscard]] VCacheDecision DecideVCachePlacement(VCacheMode mode, const VCacheTopology& topology,
                                                   bool external_rule, bool cpu_sets);
// Pure (tested): the layout with the V-cache placement: the reserved cores (`mode`) and the hot
// processors in the V-cache CCD, the cold ones in the other CCDs. `vcache` false (and the reason in
// the description) when it cannot apply: no allowed processor on either side, fewer than
// `min_general` hot ones, or the reservation does not fit in the V-cache CCD.
[[nodiscard]] CpuLayout ComputeVCacheLayout(const std::vector<CpuSetInfo>& sets, CpuReserveMode mode,
                                            int forced_cp_logical, const VCacheTopology& topology,
                                            uint32_t min_general = kMinGeneralProcessors,
                                            const ReserveRules& rules = {});
// KYTY_VCACHE_PLACEMENT=auto|0|1 (also off|on); nullptr is auto. `valid` false: unknown text.
[[nodiscard]] VCacheMode ParseVCacheMode(const char* text, bool* valid = nullptr);
[[nodiscard]] VCacheMode GetVCacheMode(); // evaluated once
// Pure (tested): the hard affinity a thread with `affinity` should get so that it stays off the
// reserved cores: unchanged when nothing is reserved or it already meets the general processors
// (the CPU sets narrow it), otherwise every general processor.
[[nodiscard]] uint64_t GeneralPoolAffinity(uint64_t affinity, const CpuLayout& layout);

// KYTY_CPU_RESERVE=off|cp|cp+recorder|stages (also 0|1|2|3); nullptr is off. `valid` false: unknown text.
[[nodiscard]] CpuReserveMode ParseCpuReserveMode(const char* text, bool* valid = nullptr);
[[nodiscard]] CpuReserveMode GetCpuReserveMode(); // KYTY_CPU_RESERVE, evaluated once
// Pure (tested): "0-15", "16-31", "0-7,16-23", "5" as a mask of logical processors 0..63; 0 with
// `valid` false for empty or malformed text, a descending range or a processor above 63.
[[nodiscard]] uint64_t ParseLogicalMask(const char* text, bool* valid = nullptr);
[[nodiscard]] uint64_t CpuSetsMask();     // KYTY_CPU_SETS, evaluated once; 0: unset (no restriction)
[[nodiscard]] bool     ReassertEnabled(); // KYTY_CPU_RESERVE_REASSERT, evaluated once
// How often the monitor found the process default CPU sets replaced and applied its layout again.
[[nodiscard]] uint32_t PlacementReasserts();

// Startup (Common::InitializeThreads, or the first PlaceCurrentThread): computes and applies the
// layout, and starts the monitor thread when a reservation or placement samples are on. Idempotent.
void InitCpuPlacement();
// Re-applies the layout if the process affinity mask changed since it was computed (the monitor
// calls it once a second).
void MaintainCpuPlacement();
// The calling thread's CPU sets for its role. The CP gets its reserved core. The recorder gets its
// core in cp+recorder mode, otherwise the general processors. Other roles keep the process
// default. No-op when nothing is reserved.
void PlaceCurrentThread(ThreadRole role);
// The CP sequencer thread (KYTY_CP_SEQ=1): its own core with KYTY_CPU_RESERVE=stages, otherwise the
// process default. No-op when nothing is reserved.
void PlaceCurrentThreadSequencer();
// A cold thread (compile and pipeline workers, file I/O, audio, diagnostics): with the V-cache
// placement it runs on the other CCD, otherwise nothing changes. Idempotent per thread and cheap
// after the first call; the thread is forgotten when it exits. Never call it from a thread that
// may run hot work (the CP, guest threads, draw-prep workers).
void PlaceCurrentThreadCold();
// The calling thread runs below normal priority (background compile workers: more of them then
// cost the game threads nothing they need). Windows only; a no-op elsewhere.
void LowerCurrentThreadPriority();
// A guest thread (guest pthreads, the guest main thread): with the V-cache placement and
// KYTY_VCACHE_GUEST=all (default) it runs on every allowed processor outside the reserved cores,
// otherwise on the process default. Idempotent per thread; forgotten when the thread exits.
void PlaceCurrentThreadGuest();
// The current layout (a copy; tests and logs).
[[nodiscard]] CpuLayout CurrentCpuLayout();

// One pass over the process's threads (the monitor runs it every 5 s): threads whose hard affinity
// differs from the process mask, those confined to reserved cores, and those moved to the general
// processors (`repin`, KYTY_CPU_RESERVE_REPIN). FrameEvents count each thread once per process.
struct AffinityScan {
	uint32_t threads       = 0;
	uint32_t hard          = 0;
	uint32_t reserved_only = 0;
	uint32_t repinned      = 0;
};
AffinityScan       ScanThreadAffinities(bool repin);
[[nodiscard]] bool RepinEnabled(); // KYTY_CPU_RESERVE_REPIN

// Placement samples (see above). Cheap: one processor-number read and a few relaxed increments.
[[nodiscard]] bool PlacementSamplingEnabled();
void               SamplePlacement(ThreadRole role);
// The logical processors the process may run on now (popcount of its affinity mask; the machine's
// count where that is unavailable). KYTY_DRAW_PREP_ADAPT's worker bound. With KYTY_CORE_BUDGET
// (common/coreBudget.h) the budget's logical processors instead (affinity and CPU sets).
[[nodiscard]] uint32_t ProcessAffinityProcessorCount();
// The CPU set table as the placement reads it: `allowed` = in the process affinity mask, not
// allocated to another process, and inside KYTY_CPU_SETS or another process's default CPU sets
// (never this module's own). `affinity` receives the process mask. Empty off Windows.
// KYTY_CORE_BUDGET's source (common/coreBudget.h).
[[nodiscard]] std::vector<CpuSetInfo> ReadProcessCpuSets(uint64_t* affinity = nullptr);
// The threads this module keeps handles of (PlaceCurrentThreadGuest: guest threads;
// PlaceCurrentThreadCold: compile, I/O, audio and monitor threads): `visit(context, tid, handle,
// guest)` for each, under the placement lock (keep it short). KYTY_SPIN_BUDGET's scan reads their
// cycle counts. Nothing off Windows.
void VisitPlacedThreads(void (*visit)(void* context, uint32_t tid, void* handle, bool guest),
                        void* context);
// group * 64 + number of the processor the calling thread runs on now.
[[nodiscard]] uint32_t CurrentLogicalProcessor();
// The physical core (group * 256 + CoreIndex) of a logical processor, UINT32_MAX if unknown.
[[nodiscard]] uint32_t CoreOfLogicalProcessor(uint32_t logical);

// Samples per role and logical processor (up to 64) since the previous call (hang trace).
struct PlacementHistogram {
	static constexpr uint32_t Processors = 64;
	std::array<std::array<uint32_t, Processors>, static_cast<size_t>(ThreadRole::Count)> samples {};
	std::array<uint32_t, static_cast<size_t>(ThreadRole::Count)> on_cp_core {};
	uint32_t cp_core = UINT32_MAX; // the CP's last sampled core
};
void TakePlacementHistogram(PlacementHistogram& out);

} // namespace Common

#endif // EMULATOR_SRC_COMMON_CPUPLACEMENT_H_
