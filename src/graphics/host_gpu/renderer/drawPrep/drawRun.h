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
extern Live::Switch g_writes;  // KYTY_DRAW_RUN_WRITES
extern Live::Switch g_recall;  // KYTY_DRAW_RUN_RECALL
extern Live::Switch g_image_epoch; // KYTY_DRAW_RUN_IMAGE_EPOCH
extern Live::Switch g_cont_eligible; // KYTY_DRAW_RUN_CONT_ELIGIBLE
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

// KYTY_DRAW_RUN_WRITES=1 (live, default 0; with KYTY_DRAW_RUN=1 or verify): draws whose programs
// write or atomically update V# buffers may form runs too. At the Sky Garden 27% of the draws that
// start a run (about 550 a frame) only differ from a continuation in that the previous draw wrote a
// buffer (same instance, pipeline, structure and dynamic state). A buffer write changes nothing a
// continuation reuses unless it lies over one of the run's images:
//  - each draw's buffers, including the written ones, are resolved, synchronized, invalidated in the
//    texture cache (InvalidateMemoryFromGPU) and bound as on the normal path, in the same order;
//  - the previous draw's post-draw shader-write barrier stays pending exactly as on the normal path
//    (CommandBuffer::CanSinkDrawWrites: draws of one rendering instance pass it, as the hardware
//    does not order buffer accesses between draws either; anything else records it first);
//  - a write by the previous draw over an image of the run marks it buffer-modified or moves its
//    content serial, which the check after the buffer work sees (the draw falls back);
//  - a candidate whose own written V# ranges overlap one of the run's images (textures or
//    attachments) is not continued at all (Miss::Writes), so its textures are never resolved after
//    its own invalidation, which the normal path would not do.
// Programs with address (BDA) writes, written or atomic images, GDS or the fault buffer still never
// form runs. A switch change ends the current run.
[[nodiscard]] inline bool WritesEnabled() {
	return Detail::g_writes.On();
}

// KYTY_DRAW_RUN_RECALL=1 (live, default 0; with KYTY_DRAW_RUN=1 or verify; needs
// KYTY_DRAW_RUN_ACQUIRE or verify mode, whose records keep their targets): a draw whose structure
// key differs from the previous draw's may continue an EARLIER record of the same rendering
// instance. A record is kept (RenderExecutor::m_run_kept, 32 entries) when the next draw starts
// a new one. At the Sky Garden about half of the draws that start a run repeat the structure of an
// earlier run of their instance (stages alternating between a few materials).
// Recalling record R is a continuation of R, with these differences:
//  - the certificate is R's: the same activity epoch (so only engine-committed draws ran since R),
//    command buffer, tick and rendering instance, programs, slice offset and structure key;
//  - each stage's texture set is taken back from the stage's texture history (the sets
//    RepeatStageTextures keeps); its image ids, views, layouts and mip views must be exactly the
//    ones R recorded, and the samplers are R's, so the bindings are R's;
//  - the targets are R's recorded targets; the image check after the buffer work compares the
//    attachments without their content serials (the draws in between drew into them, which is what
//    a continuation of them would also skip), the textures with them;
//  - the descriptors are pushed whole, the dynamic state is recorded again and the pipeline is
//    bound as for any draw (the draws in between changed all three).
// Verify mode: a would-be recall takes the normal path, which is compared with R as for any
// continuation (targets, textures and views, samplers, rendering state, no transition).
[[nodiscard]] inline bool RecallEnabled() {
	return Detail::g_recall.On();
}

// KYTY_DRAW_RUN_IMAGE_EPOCH (live, default 0; 1 on, verify, exit): a continuation's image check
// (RenderExecutor::DrawRunImagesUnchanged) is skipped while the image-state epoch
// (g_image_state_epoch, image.h) is the one read when the record's marks were taken and every
// marked image then passed the check's flag tests (registered, no rebind request, not CPU-dirty or
// buffer-modified, a texture's view ready): no image property the check reads changed since. A
// check that passes refreshes the record's epoch. verify: the check runs anyway and a skip that
// would have hidden a change is a mismatch (an epoch bump missing at some image mutation).
// "exit" stops at the first.
[[nodiscard]] inline int64_t ImageEpochMode() {
	return Detail::g_image_epoch.Get();
}

// KYTY_DRAW_RUN_CONT_ELIGIBLE (live, default 0; 1 on, verify, exit): a continuation's record
// eligibility takes its per-stage part from the record it continues. That part (no GDS, eligible
// programs, no texture that is an attachment or a storage binding, and whether a program writes
// buffers) reads only the programs, the texture bindings and the attachments, which a continuation
// keeps from its record (DrawRunCandidate: same programs; keep_images; the record's targets), so it
// is what the record computed: eligible, with the record's writes flag, under the same
// KYTY_DRAW_RUN_WRITES value (a change of it ends every run). The other conditions (engine commit,
// validation, run key, mesh, indirect, stage count, shader writes, feedback, rendering instance,
// clears) are still checked per draw. verify: the full check runs as well and a difference is a
// mismatch (DrawRunVerify line); "exit" stops at the first.
[[nodiscard]] inline int64_t ContEligibleMode() {
	return Detail::g_cont_eligible.Get();
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
	Writes, // KYTY_DRAW_RUN_WRITES: the candidate's written ranges overlap an image of the run
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
	std::atomic<uint64_t> write_records {0};   // KYTY_DRAW_RUN_WRITES: eligible draws with buffer writes
	std::atomic<uint64_t> write_continued {0}; // continuations (verify: would-be) whose programs write
	std::atomic<uint64_t> recall_matches {0};  // KYTY_DRAW_RUN_RECALL: a kept record had the key
	std::atomic<uint64_t> recalled {0};        // recalled (verify: would have)
	std::atomic<uint64_t> recall_sets_missed {0}; // a stage's texture set was not R's any more
	std::atomic<uint64_t> epoch_skips {0};         // KYTY_DRAW_RUN_IMAGE_EPOCH: checks skipped
	std::atomic<uint64_t> epoch_checks {0};        // verify: skips checked
	std::atomic<uint64_t> epoch_mismatches {0};    // verify: a skip would have hidden a change
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
