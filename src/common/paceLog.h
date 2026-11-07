#ifndef COMMON_INCLUDE_PACELOG_H_
#define COMMON_INCLUDE_PACELOG_H_

#include <cstdint>

// KYTY_PACE_LOG=<file> (default off): one CSV line per presented guest flip with the timeline of
// that frame, for classifying frames that miss a vblank (GPU late, CP late, guest late, present):
//   present_us     present thread entered the flip (the vblank tick), steady clock
//   vblank         guest vblank count at that tick
//   reserve_us     the CP executed the flip op (request reserved)
//   ready_us       the GPU work of the flip completed and was observed (request ready)
//   flipsub_us     the guest submitted the graphics submission holding the flip
//   framesub_us    the guest submitted the first graphics submission of that frame
//   cpstart_us     the CP started that first submission
//   done_us        the last sceAgcSuspendPoint of the guest before the flip
//   donewait_us    guest time blocked in sceAgcSuspendPoint since the previous flip
//   present_cost_us  time spent in the presenter for this flip
//   gpu_busy_us, gpu_idle_us, gpu_starved_us, gpu_cmdbufs, gpu_last_end_us   (KYTY_GPU_TIMING=1,
//                  folded at this flip; covers the command buffers collected since the last one)
// All *_us columns except the durations are microseconds since the first logged event.
namespace PaceLog {

[[nodiscard]] bool Enabled();
[[nodiscard]] uint64_t NowNs() noexcept; // steady_clock ns (absolute, same as GpuTiming::NowNs)

// Guest sceAgcSuspendPoint: call time and blocked time.
void NoteDone(uint64_t call_ns, uint64_t wait_ns);
// CP thread: a graphics submission begins (enqueue_ns = guest submit time). frame_start = the
// submission resets the processor (first of a guest frame).
void NoteCpSubmission(uint64_t enqueue_ns, bool frame_start);
// Flip request reserved by the CP (SubmitFlipFromGpu) / completed by the GPU.
void NoteReserve(uint64_t request_id);
void NoteReady(uint64_t request_id);
// GPU timing fold (GpuTiming::OnGuestFlip).
void NoteGpuFrame(uint64_t busy_ns, uint64_t idle_ns, uint64_t starved_ns, uint64_t cmdbufs,
                  uint64_t last_end_ns);
// A guest blocking wait (equeue) that took >= 100 us: one line in <KYTY_PACE_LOG>.waits.csv
// (begin_us,end_us,tid,filter,ident,events,name).
void NoteWait(uint64_t begin_ns, uint64_t end_ns, int tid, int filter, uint64_t ident, int events,
              const char* name);
// Presented flip: writes the line.
void NotePresent(uint64_t request_id, uint64_t vblank, uint64_t tick_ns, uint64_t present_ns);

// KYTY_STALL_TRACE=1 (diagnostic, default off; needs KYTY_PACE_LOG): host-side blocking spans of at
// least KYTY_STALL_TRACE_MIN_MS (default 20) go to the same .waits.csv, so a late frame can be
// matched with what blocked the command processor. Rows: filter = the span code below, ident = the
// shader hash (or 0), events = the CompileWatch kind (or 0), name = a static description.
enum StallCode : int {
	StallCompileWait   = -100, // a thread waited for a shader or pipeline build (CompileWatch::Wait)
	StallBuild         = -101, // a build ran on this thread (CompileWatch::Build)
	StallOptimize      = -102, // a background optimized pipeline compile (fast-first)
	StallPipelineMutex = -103, // waited for the pipeline cache lock
	StallProbe         = -104, // a driver-cache probe (fast-first, known key)
	StallFastBuild     = -105, // an unoptimized pipeline build (fast-first)
};
[[nodiscard]] bool StallTraceEnabled();
// begin_ns/end_ns from NowNs(); spans shorter than the minimum are dropped.
void NoteStall(uint64_t begin_ns, uint64_t end_ns, StallCode code, uint64_t hash, int kind, const char* name);

} // namespace PaceLog

#endif
