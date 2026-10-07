#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWRUN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWRUN_H_

#include "common/liveSwitch.h"

#include <atomic>
#include <cstdint>

// KYTY_DRAW_RUN (default 0 = off): run-level commit of consecutive draws that share their
// structure (Profiling analysis: cp-batching-report.md).
//
// At the Sky Garden start view about 55% of the committed draws continue the previous draw: the
// same rendering instance, registers (apart from the per-draw index offset and object id),
// programs, T# and S# words. Only per-draw data differs (V# buffers, user data, flattened SRT,
// index/vertex buffers, counts). Such a draw is committed as a delta of the run it continues:
//  - its programs are validated (DrawPrep::Validate) and all per-draw data is resolved, uploaded
//    and bound as for any draw (shader data, V# buffers, BDA, index/vertex buffers, descriptors,
//    push constants), in the same order;
//  - the structure is not resolved again: colour/depth target resolution, texture and sampler
//    resolution and views, the pipeline lookup, attachment acquisition (AcquireRenderTargets),
//    image layout transitions and dynamic state are those of the previous draw.
// A continuation is taken only while the run's certificate holds:
//  - the previous committed draw recorded through this path's eligible shape: one vertex stage,
//    no mesh or indirect draw, no shader writes, GDS or attachment feedback, no attachment clear
//    in its rendering state, its preparation validated;
//  - nothing else ran on the command processor since (DrawRun::ActivityEpoch: every operation other
//    than a committed draw, service commands and ready operations advance it), the command buffer,
//    its tick and the rendering instance are still the previous draw's;
//  - the draw's structure key equals the previous draw's (DrawRun::StructureKey, computed by the
//    preparing thread: every context register, the user-config registers but the index offset and
//    object id, the programs, the T# and S# words), as do its render-target slice offset;
//  - after the draw's buffer discovery and binding (which can download images for texel reads,
//    submit, or create buffers), every image the reused structure refers to (the textures of both
//    stages and the attachments) still has the identity and state (native image, registration,
//    rebind flag, residency, layout, access, stage, dirtiness, content serial) it had when the
//    previous draw finished, and the tick is unchanged. Otherwise the draw resolves its structure
//    the normal way at that point (and binds its buffers again after it).
// Everything a continuation skips is a function of those inputs and of that image state, and the
// previous draw left every skipped transition, claim and binding in the state they establish; its
// skipped side effects (LRU touches, content-identity bumps of the attachments, repeated claims)
// have no observer between the two draws. A guest CPU write racing the draws (another thread)
// changes nothing a draw of the same command buffer could rely on.
//
// KYTY_DRAW_RUN=1: continuations take the delta path.
// KYTY_DRAW_RUN=verify|exit: every draw takes the normal path; for each draw that would have been a
// continuation, what the delta path would have reused is compared with what the normal path
// computed (targets, texture bindings and views, samplers, pipeline, rendering state, barrier
// safety, shader writes), and the normal path's skipped work must have been a no-op (no image
// transition recorded, no dynamic state emitted, no attachment state changed but content serials
// and LRU marks). Differences are counted, the first N logged (KYTY_DRAW_RUN_LOG_LIMIT, default 64) ("exit" stops at the first).
// One console line every 10 s: "DrawRun 10s: ...".
namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
} // namespace HW

namespace DrawPrep {
struct PreparedDraw;
} // namespace DrawPrep

namespace DrawRun {

enum class Mode : uint8_t { Off, On, Verify };

// The three flags are live switches (common/liveSwitch.h): read per use,
// they may change at a guest flip. Live-safe: the structure key is computed by the preparing thread
// only while the flag is on (0 otherwise, which never continues), every record is reset by the next
// draw, and the parts a record holds only under one value (acquisition targets, verify copies) are
// marked valid only by the record that stored them.
namespace Detail {
// KYTY_DRAW_RUN: 0 off, 1 on, 2 verify, 3 verify stopping at the first difference ("exit").
extern Live::Switch g_mode;
extern Live::Switch g_acquire; // KYTY_DRAW_RUN_ACQUIRE
extern Live::Switch g_push;    // KYTY_DRAW_RUN_PUSH
} // namespace Detail

[[nodiscard]] inline Mode GetMode() {
	const auto mode = Detail::g_mode.Get();
	return mode == 0 ? Mode::Off : (mode == 1 ? Mode::On : Mode::Verify);
}
[[nodiscard]] inline bool Enabled() {
	return Detail::g_mode.Get() != 0;
}
// verify mode stops at the first difference.
[[nodiscard]] inline bool VerifyExit() {
	return Detail::g_mode.Get() == 3;
}

// KYTY_DRAW_RUN_ACQUIRE=1 (default 0; with KYTY_DRAW_RUN=1 or verify): a committed draw that does
// not continue the run keeps the attachment acquisition of the previous committed draw when that
// draw recorded through the run's eligible shape, nothing else ran on the command processor since,
// the command buffer, tick and rendering instance are the same, the draw resolved the same colour
// and depth targets (every field) for the same scissor union, no texture of the draw lies over an
// attachment's memory, and every attachment is unchanged since (as for a continuation). Its
// textures, samplers, bindings and dynamic state take the normal path. Verify mode runs the
// acquisition and checks it returned the recorded rendering state, recorded no transition and
// changed no attachment state.
[[nodiscard]] inline bool AcquireReuseEnabled() {
	return Detail::g_acquire.On();
}

// KYTY_DRAW_RUN_PUSH=1 (default 0; with KYTY_DRAW_RUN=1 or verify): a continuation pushes only its
// per-draw descriptors (buffers, shader data, flattened SRT, BDA, GDS, mip statistics); its image
// and sampler descriptors are the previous draw's, still in effect: the same pipeline layout, and
// no descriptor command recorded at the graphics bind point since that draw's push
// (CommandBuffer::DescriptorEpoch; push descriptors may be updated incrementally until the set is
// disturbed). Verify mode checks that the image and sampler descriptors the normal path pushes for
// a would-be continuation are the previous draw's.
[[nodiscard]] inline bool PushPartialEnabled() {
	return Detail::g_push.On();
}

// Command-processor work other than committed draws (GPU thread). Relaxed: written and read by the
// GPU thread; other threads only bump it.
// KYTY_DRAW_RUN_QUIET_OPS (default on): two kinds of command-processor work between draws do not
// end a run. An occlusion dump that kept the rendering instance (KYTY_OCCLUSION_SPLIT, no wait)
// records only query commands. A WRITE_DATA done by the CPU writes guest memory like a guest
// thread would: a write to a texture's or attachment's memory reaches it through write tracking
// (its state, which the continuation checks after the buffer work, changes) and the per-draw data
// is resolved by every draw. A WRITE_DATA done on the GPU timeline still ends the run. Astro Bot's
// clock tower level brackets most draws with an occlusion dump pair and a WRITE_DATA.
// KYTY_DRAW_RUN_QUIET_OPS=0: every such operation ends the run again.
[[nodiscard]] bool QuietOpsEnabled();
// KYTY_DRAW_RUN_INDIRECT (default on): a native indirect draw with a prepared slot
// (KYTY_DRAW_PREP_INDIRECT) is an engine commit. It may continue a run and seed the next one, and
// it takes its slot's binding plan without the vertex ranges (planned from placeholder counts).
// A continuation that would need the argument barrier (which ends rendering) falls back late.
// Astro Bot's Ape Escape level issues about 2,300 single-record indirect draws per frame, each of
// which used to take the whole serial path and end the run for the draw after it.
// KYTY_DRAW_RUN_INDIRECT=0: an indirect draw is never a commit and ends the run.
[[nodiscard]] bool IndirectRunsEnabled();
// KYTY_DRAW_RUN_QUIET_SYNTHETIC (live, default on; needs KYTY_DRAW_RUN_QUIET_OPS): with
// KYTY_GPU_OCCLUSION=0 an occlusion dump is a CPU write of an always-visible result (and
// NoteCpWrite), which is what a quiet CPU WRITE_DATA does, so it does not end a run either. Ape
// Escape with occlusion off issues about 1,500 such dumps per frame between otherwise continuable
// draws. DRAW_RUN=verify: 0 mismatches (Ape Escape, snow). 0: a synthetic dump ends the run.
[[nodiscard]] bool QuietSyntheticEnabled();
// what: a static name of the work, for the 10-second line's "activity by" attribution.
void NoteForeignActivity(const char* what = "other") noexcept;
[[nodiscard]] uint64_t ActivityEpoch() noexcept;

// The structure key of a prepared draw (any thread; the preparing thread in parallel mode). 0 when
// the preparation failed.
[[nodiscard]] uint64_t StructureKey(const HW::Context& context, const HW::UserConfig& user_config,
                                    const DrawPrep::PreparedDraw& prepared);

// Why a candidate did not continue its run (after the key matched).
enum class Miss : uint8_t {
	Activity,
	Command,
	Instance,
	Programs,
	Validation,
	Images,
	Pipeline,
	Count
};

struct Totals {
	std::atomic<uint64_t> draws {0};           // committed draws that reached PrepareDrawRenderState
	std::atomic<uint64_t> eligible {0};        // recorded draws that can seed a continuation
	std::atomic<uint64_t> key_matches {0};     // the previous draw was eligible and the key matched
	std::atomic<uint64_t> continued {0};       // took the delta path (verify: would have)
	std::atomic<uint64_t> late_fallbacks {0};  // images changed during the buffer work
	std::atomic<uint64_t> pipeline_lookups {0};// continuation whose pipeline was looked up again
	std::atomic<uint64_t> dynamic_emitted {0}; // continuation whose dynamic state was recorded again
	std::atomic<uint64_t> depth_promotions_excluded {0}; // next depth acquisition broadens its access
	std::atomic<uint64_t> depth_promotions_deferred {0}; // KYTY_DEPTH_FEEDBACK_LAZY kept the access
	std::atomic<uint64_t> alias_excluded {0};  // eligible, but a texture lies over an attachment
	std::atomic<uint64_t> acquire_reused {0};  // KYTY_DRAW_RUN_ACQUIRE (verify: would have)
	std::atomic<uint64_t> partial_pushes {0};  // KYTY_DRAW_RUN_PUSH (verify: would have)
	std::atomic<uint64_t> verify_checks {0};
	std::atomic<uint64_t> verify_mismatches {0};
	std::atomic<uint64_t> misses[static_cast<uint32_t>(Miss::Count)] {};
};
[[nodiscard]] Totals& GetTotals();

void CountMiss(Miss miss) noexcept;
// Verify mode: one comparison, and a difference (counted, logged; exit mode stops).
void CountVerifyCheck() noexcept;
[[nodiscard]] uint32_t MismatchLogLimit();
void ReportMismatch(const char* what, uint64_t detail = 0);
// GPU thread: the 10-second console line.
void PrintSummary();

} // namespace DrawRun
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_DRAWRUN_H_
