#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_COMMITSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_COMMITSTATS_H_

#include "common/onceValue.h"
#include <cstdint>

// KYTY_CP_COMMIT_STATS=1 (diagnostic, default off): where the command processor's time goes when it
// commits a draw (DrawPrep::Engine::Commit), phase by phase, measured with the TSC, and how much of
// it a run-level commit or a cross-frame reuse could skip. Nothing it measures feeds back into
// emulation; with the switch unset every hook is one predictable branch.
//
// Every committed draw that records a draw command is classified against the previous one:
//  - continuation: same rendering instance (no end, split or barrier flush in between), pipeline,
//    programs, attachments, texture views/layouts, samplers and dynamic state, and the previous
//    draw had no shader buffer writes. Only per-draw data differs (V# buffers, user data, SRT
//    tables, index/vertex buffers, counts): a draw a run-level commit could record as a delta.
//  - start: the first draw of such a run.
// Draws that return before recording (no targets, metadata operations) count as "other".
// Cross-frame (frames start at the graphics processor reset after sceAgcSuspendPoint): the share
// of draws whose complete binding inputs (every stage's V#/T#/S# words, flattened SRT, user data,
// draw arguments) occurred in the previous frame or the one before (exact reuse), and the same for
// their structure (programs, pipeline, attachments, T#, S#, V# words without base addresses).
// One console line every 10 s (flushed), "CommitStats 10s: ...".
// A second line, "StartPhases 10s: ...", splits the draws by the actual KYTY_DRAW_RUN decision:
// continued, or started for the first shape difference to the previous draw (first of a frame,
// rendering instance or command buffer, pipeline, structure, dynamic state, a previous writing
// draw the run did not continue, or none: the run missed). Per reason: phases, sub-phases, items
// and, with KYTY_CP_GAP_STATS=1, the gap before the draw split by the command processor's
// categories (cpGaps.h; the time of scopes that closed in the gap, the rest "unscoped").
namespace Libs::Graphics::CommitStats {

enum class Phase : uint8_t {
	Setup,           // Commit's register binding, DrawIndex/DrawAuto checks before the render state
	Programs,        // RefreshShaders: certificate check and binding-plan activation
	Targets,         // colour/depth target resolution
	PrepareBindings, // per stage: textures, samplers, shader data (and mesh/indirect set-up before)
	FindBuffers,     // V# range discovery (FindBuffer) and the BDA preparation
	RebindImages,    // image views, colour-target rediscovery
	RebindBuffers,   // V# bindings (ObtainBuffer and synchronization), shader-table uploads
	VertexIndex,     // vertex/index/indirect buffers
	Pipeline,        // pipeline selection
	AcquireTargets,  // attachment acquisition and transitions
	CommitBindings,  // image transitions, descriptor pushes/sets, push constants, vertex binds
	Emit,            // index bind, dynamic state, rendering begin, pipeline bind, draw, barriers
	Post,            // ResetBindings, the after-commit hook, register restore, early returns
	Count
};

// Sub-phases: parts of the phases above, timed inside them (SubScope), reported per class as
// "sub[...]" in microseconds per draw of the class (the phases still include them).
enum class Sub : uint8_t {
	Textures,     // PrepareBindings: texture resolution (memo runs, repeats, full resolutions)
	Samplers,     // PrepareBindings: sampler handles
	Views,        // RebindImages: texture views
	ColorTargets, // colour target resolution
	DepthTarget,  // depth target resolution
	AcquireColor, // AcquireRenderTargets: colour attachments
	AcquireDepth, // AcquireRenderTargets: depth attachment
	Push,         // CommitBindings: descriptor writes, pushes or sets, push constants
	Dynamic,      // dynamic state
	Begin,        // BeginRendering (with the barrier flush)
	AcqCandidate, // KYTY_DRAW_RUN_ACQUIRE: the reuse check (DrawRunAcquireCandidate)
	RunRecord,    // KYTY_DRAW_RUN: the eligibility check and the record (DrawRunRecordDraw)
	// Start-draw breakdown ("StartPhases" line):
	Prefetch,     // PipelineCache::PrefetchGraphicsPipeline (draws without a planned pipeline)
	LateFallback, // a draw-run continuation that fell back late (images/command changed)
	VertexCommit, // CommitVertexBuffers
	Transitions,  // CommitBindings: image transitions (barrier batcher requests)
	DescBuild,    // CommitBindings: descriptor write list
	PushConst,    // CommitBindings: push constants
	DescPush,     // CommitBindings: push descriptors
	DescSet,      // CommitBindings: descriptor set (reuse lookup, allocation, update, bind)
	BindPipeline, // pipeline bind and dynamic-state shadow
	Barriers,     // indirect-argument and shader-write barriers (with their EndRendering)
	DrawCmd,      // the draw command(s)
	// KYTY_CP_COMMIT_STATS=steps: BufferCache::ObtainReadBinding calls inside the commit.
	ReadHit,      // memo hits (whole call)
	ReadMiss,     // memo misses (whole call, with ObtainBufferNow and the record)
	ReadSig,      // of which: the range signature read (MemoryTracker::RangeSignature)
	ValLog,       // DrawPrep::Validate: the coherence log check
	ValClean,     // DrawPrep::Validate: the clean verdicts after it
	Count
};
// Per-draw counts (averaged per draw of the class): "items[...]".
enum class Item : uint8_t {
	Textures,
	Samplers,
	TexSetCurrent,  // a stage's texture set: its current set's words (RepeatStageTextures)
	TexSetHistory,  // an earlier set of its history
	TexSetMiss,     // new words
	TexSetRepeated, // the found set passed TryRepeatResolve
	ColorMemoHit,   // colour target description memo
	ColorMemoMiss,
	DepthMemoHit,
	DepthMemoMiss,
	TargetRepeat,   // a target lookup repeated (TryRepeatLookup)
	TargetLookup,   // a full target FindImage
	DescWrites,     // descriptor writes pushed or written (graphics)
	Transitions,    // image transitions recorded while the draw was committed
	PipelineNew,    // the draw's pipeline differs from the one bound before it
	ReadHits,       // KYTY_CP_COMMIT_STATS=steps: ObtainReadBinding memo hits
	ReadMisses,     // and misses
	LogEntries,     // KYTY_CP_COMMIT_STATS=steps: coherence log entries a certificate check read
	CertRanges,     // and the certificate's ranges
	Count
};

// KYTY_CP_COMMIT_STATS=steps (adds to =1): a third line, "Steps 10s: ...", splits every draw's
// commit and the gap before it into consecutive steps (one TSC read per step boundary, so the
// steps add up to the commit and the gap), per KYTY_DRAW_RUN decision as in StartPhases. A step's
// time is from the previous boundary to its own; "calib" is an empty step (the cost of one
// boundary, included in every step). Gap steps run on the GPU thread between draws: what a
// boundary closes after the previous draw's end belongs to the next draw's gap.
enum class Step : uint8_t {
	// The gap, from the previous commit's end (after its statistics) to BeginDraw.
	GapCommitTail,  // Engine::Commit after EndDraw: periodic summaries
	GapRetire,      // CommitHead: window retire; ExecDrawIndex tail (MaybeYieldSlice)
	GapOpExit,      // ExecuteOp's scope exit, register restore, op pop, NoteExecuted, loop top
	GapLoop,        // resolver loop: commands check, Peek, placement, epoch, snapshot, bind
	GapOpHead,      // ExecuteOp: scope, run activity checks, the switch
	GapDrawOp,      // ExecDrawIndex/ExecDrawAuto up to CommitPublished
	GapHead,        // ReadyHead: claim/prepare/wait for the head
	GapCommitIn,    // Engine::Commit entry up to BeginDraw
	// The commit (BeginDraw to EndDraw).
	BindRegs,       // Engine::Commit: BindRegisters, executor fields
	PopReady,       // DrawIndex: PopReadyOperations
	DebugInfo,      // SetDebugInfo
	Lock,           // the render mutex
	BeginRun,       // BeginDrawRun
	TargetOps,      // ConsumeMetadataColorOperation, DepthStencilCopy, ResolveColorTargets
	VsCheck,        // DrawHasValidVertexShader, debug dump check
	DrawChecks,     // RunDrawChecks
	StateReset,     // topology, index source, state.Reset
	PsActive,       // DrawHasActivePixelShader
	TakePrepared,   // RefreshShaders up to Validate (prepared draw, export mapping)
	Validate,       // DrawPrep::Validate (certificate)
	Apply,          // ApplyPreparedDraw, plan activation
	RunCandidate,   // DrawRunCandidate (and recall)
	MarkBindings,   // DrawRunMarkBindings / target resolution
	Restart,        // ResolvePrimitiveRestart, 8-bit indices, LogDrawState, offsets
	ExecHead,       // ExecutePreparedDraw up to PrepareBindings (prefetch, mesh, indirect)
	PrepStages,     // PrepareBindings of every stage
	FindBuffers,    // FindBuffers of every stage
	Bda,            // PrepareBda
	Images,         // RebindImages, colour targets
	RbHead,         // RebindBuffers: tick, shader-data reset, upload batch, memo prefetch
	RbBindings,     // RebindBuffers: write ranges, NativeStorageBuffer per V#
	RbMipStats,     // RebindBuffers: WriteMipStatsFields
	RbTables,       // RebindBuffers: flattened SRT and shader-data uploads
	RbBatchEnd,     // the stages' upload batch end
	VertexAcquire,  // AcquireVertexBuffers
	IndexPrepare,   // PrepareIndexBuffer, indirect buffers
	LateCheck,      // the continuation's late check (command, images), verify
	PipelineSel,    // plain-pixel choice, planned pipeline / lookup
	Acquire,        // written rectangle, rendering state (acquisition or reuse)
	BarrierSafe,    // DrawIsBarrierSafe
	DrawScope,      // DrawScope, EmissionSink, debug phase
	VertexCommit,   // CommitVertexBuffers
	CbHead,         // CommitBindings: retick, address writes, counts, vector resets
	CbTransitions,  // CommitBindings: image transitions (per stage)
	CbBuild,        // CommitBindings: descriptor write list and push data (per stage)
	CbPushConst,    // CommitBindings: push constants
	CbDescriptors,  // CommitBindings: descriptor push or set
	CbTail,         // the run push verify, transition check
	IndexCommit,    // CommitIndexBuffer
	DynamicState,   // the dynamic-state check and emission
	Begin,          // BeginRendering, depth feedback
	BindPipeline,   // pipeline bind and shadow
	DrawCmd,        // the draw command
	Occlusion,      // occlusion diagnostics checks, shader write stages and barrier
	RunRecord,      // the run eligibility and record
	ResetBindings,  // DrawIndex: ResetBindings
	CommitTail,     // Engine::Commit: plan count, repeat trace, register restore, after-commit
	Calib,          // an empty step (the boundary's own cost)
	Count
};

namespace Detail {
[[nodiscard]] bool ReadEnabled();
[[nodiscard]] bool ReadIndirectOnly();
[[nodiscard]] bool ReadSteps();
void               StepAt(Step step);
} // namespace Detail

[[nodiscard]] inline bool StepsEnabled() {
	static std::atomic<int64_t> steps_once {Common::OnceUnset};
	const auto steps = static_cast<bool>(Common::Once(steps_once, [&] {
		return Detail::ReadSteps();
	}));
	return steps;
}

// A step boundary (GPU thread): the time since the previous boundary belongs to `step`.
inline void StepMark(Step step) {
	if (StepsEnabled()) [[unlikely]] {
		Detail::StepAt(step);
	}
}

[[nodiscard]] inline bool Enabled() {
	static std::atomic<int64_t> enabled_once {Common::OnceUnset};
	const auto enabled = static_cast<bool>(Common::Once(enabled_once, [&] {
		return Detail::ReadEnabled();
	}));
	return enabled;
}

// KYTY_CP_COMMIT_STATS=indirect: only native indirect draws committed from a prepared slot
// (DrawPrep::Engine::ExecuteIndirect) are measured.
[[nodiscard]] inline bool IndirectOnly() {
	static std::atomic<int64_t> indirect_once {Common::OnceUnset};
	const auto indirect = static_cast<bool>(Common::Once(indirect_once, [&] {
		return Detail::ReadIndirectOnly();
	}));
	return indirect;
}

// What one recorded draw bound, for the classification (hashes; 0 where not applicable).
struct DrawShape {
	uint64_t rendering_serial = 0; // CommandBuffer::ActiveRenderingSerial after BeginRendering
	uint64_t command          = 0; // native command buffer identity
	uint64_t pipeline         = 0; // native pipeline handle
	uint64_t structure        = 0; // programs, attachment views, texture views/layouts, samplers
	uint64_t dynamic          = 0; // dynamic-state shadow
	uint64_t buffers          = 0; // V# descriptor buffer infos (host buffer, offset, range)
	uint64_t push             = 0; // shader data dwords of every stage
	uint64_t tables           = 0; // flattened SRT / shader-data allocations
	uint64_t index            = 0; // index buffer handle, offset, type
	uint64_t vertex           = 0; // vertex buffer handles, offsets, sizes
	uint64_t bind_input       = 0; // cross-frame: complete binding inputs
	uint64_t struct_input     = 0; // cross-frame: structure only
	bool     shader_writes    = false;
	bool     run_continued    = false; // KYTY_DRAW_RUN continued the previous draw's run
};

// GPU thread (the command processor's resolver), around DrawPrep::Engine::Commit.
void BeginDraw();
// Time since the previous mark (or BeginDraw) belongs to `phase`.
void Mark(Phase phase);
// Time since the previous mark belongs to no phase (the diagnostic's own work).
void Skip();
// The draw recorded a draw command; its shape (ExecutePreparedDraw, after the draw).
void NoteRecorded(const DrawShape& shape);
void EndDraw();
// The graphics processor reset that starts a guest frame.
void OnFrameBoundary();
// Gap attribution (before BeginDraw): the draw-prep head this commit takes was prepared on this
// thread (self) or waited for (wait), in TSC cycles; added to the next BeginDraw's gap split.
[[nodiscard]] uint64_t Now();
void NoteHead(uint64_t self_cycles, uint64_t wait_cycles);

void AddSub(Sub sub, uint64_t cycles);
void AddItems(Item item, uint64_t count);

class SubScope {
public:
	explicit SubScope(Sub sub): m_sub(sub), m_start(Enabled() ? Now() : 0) {}
	~SubScope() { Stop(); }
	void Stop() {
		if (m_start != 0) {
			AddSub(m_sub, Now() - m_start);
			m_start = 0;
		}
	}
	SubScope(const SubScope&)            = delete;
	SubScope& operator=(const SubScope&) = delete;

private:
	Sub      m_sub;
	uint64_t m_start;
};

// Hash helpers for building a DrawShape.
[[nodiscard]] uint64_t Hash(const void* data, uint64_t size, uint64_t seed = 0);

} // namespace Libs::Graphics::CommitStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWPREP_COMMITSTATS_H_
