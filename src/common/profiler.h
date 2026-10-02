#ifndef KYTY_COMMON_PROFILER_H_
#define KYTY_COMMON_PROFILER_H_

#include "common/common.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <optional>
#include <tracy/Tracy.hpp> // IWYU pragma: export

namespace profiler::colors {

inline constexpr uint32_t RedA100        = 0xff8a80;
inline constexpr uint32_t Blue300        = 0x64b5f6;
inline constexpr uint32_t CyanA700       = 0x00b8d4;
inline constexpr uint32_t Green100       = 0xc8e6c9;
inline constexpr uint32_t Green200       = 0xa5d6a7;
inline constexpr uint32_t Green300       = 0x81c784;
inline constexpr uint32_t Green400       = 0x66bb6a;
inline constexpr uint32_t Amber300       = 0xffd54f;
inline constexpr uint32_t DeepOrangeA200 = 0xff6e40;

} // namespace profiler::colors

namespace Profiler {

// The instrumentation below sits on hot paths (hundreds of thousands of calls per guest flip on
// the command processor), so every "is it on?" check is an inlined load of a flag, and counters
// are per thread (see CountFrameEvent).
namespace Detail {
// Environment switches read on first use by any thread: -1 until then, else 0 or 1. They are
// constant-initialized, so they are correct even before this module's dynamic initialization.
extern std::atomic<int8_t> g_frames_only;
extern std::atomic<int8_t> g_aggregate;
extern std::atomic<int8_t> g_detailed;
extern std::atomic<int8_t> g_shared_counters;
[[nodiscard]] bool ReadFramesOnly() noexcept;
[[nodiscard]] bool ReadAggregate() noexcept;
[[nodiscard]] bool ReadDetailed() noexcept;
[[nodiscard]] bool ReadSharedCounters() noexcept;
// ScopedBlock emits Tracy zones: the profiler is started and not in frame-only mode. Set by
// Initialize, cleared by Shutdown.
extern std::atomic<bool> g_zones;
} // namespace Detail

class ScopedBlock {
public:
	explicit ScopedBlock(const tracy::SourceLocationData* source_location, bool active = true) {
		if (active && Detail::g_zones.load(std::memory_order_relaxed)) [[unlikely]] {
			Begin(source_location);
		}
	}
	ScopedBlock(const ScopedBlock&)            = delete;
	ScopedBlock& operator=(const ScopedBlock&) = delete;
	ScopedBlock(ScopedBlock&&)                 = delete;
	ScopedBlock& operator=(ScopedBlock&&)      = delete;
	// Always inlined: one flag test per scope (with zones off it is the whole cost).
#if defined(__clang__) || defined(__GNUC__)
	__attribute__((always_inline))
#endif
	~ScopedBlock() {
		if (m_active) [[unlikely]] {
			End();
		}
	}

	void End() noexcept;

private:
	void Begin(const tracy::SourceLocationData* source_location);

	// The zone, constructed in place by Begin while zones are on. Not a std::optional: its
	// destructor would inline Tracy's zone-end code into every scope, so compilers kept this
	// destructor out of line (a call per scope with zones off).
	alignas(tracy::ScopedZone) unsigned char m_zone[sizeof(tracy::ScopedZone)];
	bool m_active = false;
};

void EndBlock();
void SetThreadName(const char* name);
// Keep frame markers and aggregate workload counters, without per-call zones.
// Set KYTY_PROFILE_FRAMES_ONLY=1 before launching; it overrides detailed profiling.
[[nodiscard]] inline bool FramesOnlyEnabled() {
	const auto value = Detail::g_frames_only.load(std::memory_order_relaxed);
	return value >= 0 ? value != 0 : Detail::ReadFramesOnly();
}
// Opt in before launching with KYTY_PROFILE_DETAILS=1. High-frequency nested
// zones remain disabled for measurements comparable to the normal instrumentation.
[[nodiscard]] inline bool DetailedEnabled() {
	const auto value = Detail::g_detailed.load(std::memory_order_relaxed);
	return value >= 0 ? value != 0 : Detail::ReadDetailed();
}
// Additional diagnostic counters/timers; requires KYTY_PROFILE_AGGREGATES=1 and frame-only
// mode. Collection separately requires a connected profiler. Keep off for performance runs.
[[nodiscard]] inline bool AggregateEnabled() {
	const auto value = Detail::g_aggregate.load(std::memory_order_relaxed);
	return value >= 0 ? value != 0 : Detail::ReadAggregate();
}

enum class FrameWork : uint32_t {
	DrawIndex,
	DrawAuto,
	DrawIndirectCommands,
	DrawIndirectMultiCommands,
	DispatchDirectCommands,
	DispatchIndirectCommands,
	Count,
};

// DrawIndex/DrawAuto include expanded indirect draws. The command categories
// describe their origins and must not be added to those renderer-call totals.
// Counted in frame-only mode (defined below).
inline void CountFrameWork(FrameWork kind);

// The first four categories are mutually exclusive per original scheduler submission,
// not per driver call or merged VkSubmitInfo. Generic also includes forced completion
// and caller-supplied wait/signal semaphores. Remaining categories count separate work.
enum class FrameEvent : uint32_t {
	SubmitBoundaryUnprotected,
	SubmitBoundaryEopOnly,
	SubmitBoundaryEopMixed,
	SubmitBoundaryGeneric,
	MeshDraws,
	MeshRestartEnabledDraws,
	MeshInputIndices,
	MeshWorkgroups,
	ShaderProgramsCreated,
	GraphicsPipelinesCreated,
	SrtProbeHits,
	SrtProbeMisses,
	SrtProbeBytes,
	SrtProbeBatchHits,
	ResourceReuseHits,
	ResourceReuseMisses,
	ResourceReuseValidationBytes,
	ResourceReuseRejectedCaptures,
	SrtRecipeSessions,
	SrtRecipeCompiledNodes,
	SrtRecipeFallbackNodes,
	SrtRecipeMemoHits,
	SrtTapeExecutions,
	SrtTapeOperations,
	SrtTapeBoundaryCalls,
	BdaSyncPasses,
	BdaSyncSkips,
	BdaSyncScannedBuffers,
	BdaSyncUploadBytes,
	BdaSyncUploadCopies,
	ShaderHeaderProbeHits,
	ShaderHeaderProbeMisses,
	VertexMetadataProbeHits,
	VertexMetadataProbeMisses,
	NativeImagePoolHits,
	NativeImagePoolMisses,
	NativeImagePoolRetires,
	NativeImagePoolAddedBytes,
	DccKnownFillClears,
	NativeImagePoolRemovedBytes,
	TextureCleanProofHits,
	TextureCleanProofMisses,
	TextureCleanProofStores,
	ResourceCacheKeyMatches,
	ResourceCacheBackingRejects,
	ResourceCacheHistoryHits,
	ResourceCacheBudgetRejects,
	ResourceCachePermutationHits,
	SrtSharedCleanMemoHits,
	UploadReservationsOutsideLocks,
	ShaderUploadReuseHits,
	ShaderUploadReuseMisses,
	ShaderUploadBytesAvoided,
	TextureDescriptionHits,
	TextureDescriptionMisses,
	PipelineBindsAvoided,
	DescriptorPushesAvoided,
	PredicatedPackets,
	PredicatedPacketsSkipped,
	OcclusionPredicates,
	OcclusionPredicatesPending,
	OcclusionCounterDumps,
	NativeOcclusionScopes,
	NativeOcclusionDumps,
	NativeOcclusionReductions,
	MeshRestartMarkers,
	MeshRestartSegments,
	// Indirect draws whose GPU-written arguments were consumed by vkCmdDraw*Indirect*, and those
	// that fell back to reading (and so synchronizing) the arguments on the CPU.
	DrawIndirectNative,
	DrawIndirectFallback,
	// Later draws that inherited a native indirect draw's instance count and read it back.
	DrawIndirectInstanceReads,
	// KYTY_GPU_TIMING: command buffers without a usable timestamp pair (ring full, results
	// unavailable, ambiguous wrap, or pending samples over capacity). Busy/idle exclude them.
	GpuTimingDropped,
	// TryReadGpuCleanBacking calls answered from (hits) or not fully from (misses) the
	// per-page clean verdict cache; stores count 4 KiB pages newly proven clean.
	CleanVerdictHits,
	CleanVerdictMisses,
	CleanVerdictStores,
	// Direct-backing reads translated by the per-thread mapping cache without m_mutex.
	BackingMapCacheHits,
	BackingMapCacheMisses,
	// Guest-fault readbacks (BufferCache::ReadMemory, KYTY_READBACK_SIDE_COPY). SideCopies are
	// copies recorded on the side command buffer without draining the current recording; bytes
	// are the dirty bytes copied. DuplicateWaits are faults served by another thread's pending
	// side copy. Fallbacks took the drain path: the current recording wrote a dirty byte of the
	// faulting page, it holds an unbounded (address) writer, or another reason (no buffer, no
	// dirty bytes, an overlapping non-side publication, no free slot). PagesUnmarked counts
	// pages unprotected at completion; PagesRetained counts pages a newer writer re-dirtied.
	ReadbackSideCopies,
	ReadbackSideCopyBytes,
	ReadbackSideDuplicateWaits,
	ReadbackSideFallbackCurrentWriter,
	ReadbackSideFallbackUnbounded,
	ReadbackSideFallbackOther,
	ReadbackSidePagesUnmarked,
	ReadbackSidePagesRetained,
	// Guest GPU recording counters (KYTY_GPU_OP_COUNTERS, gpuOpProfiler.h), added per guest
	// flip: dynamic-rendering begins, pipeline barrier calls, image barriers that change layout
	// and guest command buffers begun. Per-site barrier plots are GpuOps.Barriers.<site>.
	GpuRenderPassBegins,
	GpuPipelineBarriers,
	GpuImageLayoutTransitions,
	GpuGuestCommandBuffers,
	// Barrier batcher (KYTY_BARRIER_BATCH, render.h), added per guest flip with the counters
	// above: barrier requests, requests merged into an already pending batch, requests elided
	// as covered by the previous barrier, pending batches kept across a same-instance draw
	// (sunk), and barrier flushes that had to end an active rendering instance.
	GpuBarrierRequests,
	GpuBarriersMerged,
	GpuBarriersElided,
	GpuBarriersSunk,
	GpuBarrierRenderSplits,
	// Texture-cache reinterpretation copies (TextureCache::CopyImage) by path: same-size color
	// vkCmdCopyImage, VK_KHR_maintenance8 depth<->color vkCmdCopyImage, one-pass shader
	// reinterpretation (depth->color compute, color->depth draw), the image->buffer->image
	// fallback and the D16 download/convert/upload path. AliasSync*: SyncAliasFromOwner copies
	// and copies skipped because both aliases already hold identical native contents.
	ImageCopyDirect,
	ImageCopyMaintenance8,
	ImageCopyShaderDepthToColor,
	ImageCopyShaderColorToDepth,
	ImageCopyViaBuffer,
	ImageCopyD16,
	AliasSyncCopies,
	AliasSyncSkips,
	// KYTY_ALIAS_BYTES (TextureCache): images that kept part of their GPU ownership through a write
	// of other bytes (another alias, a bounded buffer write), render-target claims bounded by the
	// draw's scissor, owners whose bytes were moved into the buffer (and the bytes), and owned
	// bytes that could not be moved (no download layout).
	AliasBytesKept,
	AliasBytesBoundedClaims,
	AliasBytesMaterializations,
	AliasBytesMaterializedBytes,
	AliasBytesUnmaterialized,
	// Texel-buffer reads checked for image-owned bytes, and those whose 1 MiB pages hold a
	// registered image (they take the texture-cache lock to look).
	AliasBytesTexelReads,
	AliasBytesTexelScans,
	// Guest memory -> native image refreshes (TextureCache::InitializeImage) and bytes, and
	// CPU-dirty bytes copied into device buffers (BufferCache::SynchronizeBuffer).
	ImageUploads,
	ImageUploadBytes,
	BufferUploadBytes,
	// Draw-prep S3 (KYTY_STAGE_PREP_PARALLEL): draws whose PS and VS were materialized in
	// parallel (PS on the DrawPrep helper), forks declined because the helper slept (it is woken
	// for the next draw), and parallel attempts that fell back to the serial path because a
	// speculative (probe-only) materialization failed, a source/permutation was missing, or the
	// KYTY_STAGE_PREP_VERIFY oracle disagreed.
	StagePrepParallelDraws,
	StagePrepForkDeclined,
	StagePrepSpeculativeFailures,
	StagePrepLookupFallbacks,
	StagePrepVerifyMismatches,
	// Per-thread last-lookup memos (KYTY_PROGRAM_LOOKUP_MEMO): program source entry per stage,
	// and the permutation last matched for that source.
	ProgramSourceMemoHits,
	ProgramSourceMemoMisses,
	PermutationMemoHits,
	PermutationMemoMisses,
	// GetGraphicsPipeline last-key memo (KYTY_PIPELINE_MEMO).
	PipelineMemoHits,
	PipelineMemoMisses,
	// RenderExecutor sampler memo keyed on the final sampler dwords (KYTY_SAMPLER_MEMO).
	SamplerMemoHits,
	SamplerMemoMisses,
	// Color/depth target descriptions memoized on the raw target registers
	// (KYTY_TARGET_DESC_MEMO).
	TargetDescMemoHits,
	TargetDescMemoMisses,
	// Graphics dynamic-state commands recorded or elided by the per-command-buffer shadow
	// (KYTY_DYNAMIC_STATE_SHADOW).
	DynamicStateCommandsEmitted,
	DynamicStateCommandsAvoided,
	// Vertex attribute/V# table reads served by one batched probe per table, or falling back
	// to per-attribute probes (KYTY_SHADER_METADATA_BATCH).
	VertexTableBatchHits,
	VertexTableBatchMisses,
	// Shader map lookups answered by the per-thread memo (KYTY_SHADER_MAP_MEMO).
	ShaderMapMemoHits,
	ShaderMapMemoMisses,
	// Precise write ranges (KYTY_PRECISE_WRITE_RANGES, WriteRangeAnalysis.h): writable storage
	// bindings whose GPU-written range a shader proof narrowed below the binding, writable bindings
	// kept whole (unprovable store, or the switch is off), binding bytes not marked GPU-written,
	// and images overlapping a narrowed binding left valid because no written range overlaps them.
	WriteRangeNarrowed,
	WriteRangeWhole,
	WriteRangeBytesAvoided,
	WriteRangeImagesSpared,
	// Guest rendering instances ended (every cause; per-site Tracy plots
	// GpuOps.EndRendering.<site>) and post-draw shader-write barriers kept pending across a draw
	// continuing the same instance (KYTY_DRAW_WRITE_SINK).
	GpuRenderingEnds,
	GpuDrawWriteSinks,
	// Rendering instances with ZPASS counting enabled that began while no occlusion dump pair was
	// open, so no query was recorded (KYTY_OCCLUSION_GATE).
	OcclusionScopesGated,
	// DCC clear materialization of GPU-written metadata (TextureCache::MaterializeDccClear):
	// native GPU inspections recorded (per metadata slice), retained inspections reused, and CPU
	// readback fallbacks, each fallback also counted once under its first failing reason.
	DccGpuRecords,
	DccGpuReuses,
	DccCpuFallbacks,
	DccFallbackDisabled,       // KYTY_DCC_GPU is not 1, or the device lacks the helper features
	DccFallbackBinding,        // video-out binding (its clear key is not consumed)
	DccFallbackShape,          // volume, mip/aspect/view shape, or native extent mismatch
	DccFallbackFormat,         // no UINT storage alias of the view format's texel size
	DccFallbackImageState,     // image not registered/GPU-owned, CPU/buffer dirty, alias/depth
	DccFallbackMetadataAliased, // an image owns (overlaps) the metadata, or metadata overlaps data
	DccFallbackUnsupported,    // native image lacks storage/mutable usage, or size limits
	DccFallbackAlignment,      // canonical metadata buffer offset is not 4-byte aligned
	// Texture streaming (TextureCache, KYTY_TEXTURE_PARTIAL_UPLOAD): refreshes that uploaded only
	// the mip rows/levels overlapping dirty chunks, their tiled bytes, the bytes a full refresh
	// would have added, chunk-tracked refreshes that still had to upload the whole image
	// (ineligible, too dirty, or no partial validity), chunks newly dirtied by CPU writes
	// (~ write faults on tracked texture pages) and KYTY_TEXTURE_PARTIAL_VERIFY mismatches.
	TexturePartialUploads,
	TexturePartialUploadBytes,
	TexturePartialSkippedBytes,
	TexturePartialFallbacks,
	TextureChunkInvalidations,
	TexturePartialVerifyMismatches,
	// Overlapped texture images kept alive instead of freed (KYTY_TEXTURE_OVERLAP_KEEP_FRAMES).
	TextureOverlapKeeps,
	// TileManager scratch buffers reused from its pool or newly allocated (KYTY_TILER_SCRATCH_POOL).
	TilerScratchPoolHits,
	TilerScratchPoolMisses,
	// Texture refresh staging copies handed to the StagingCopier worker, and their bytes
	// (KYTY_TEXTURE_ASYNC_STAGING).
	TextureAsyncCopies,
	TextureAsyncCopyBytes,
	// Image refresh bytes by source route: an existing cache buffer or GPU-written bytes
	// (BufferCache), a synchronous staging copy on the GPU thread, or the StagingCopier worker.
	TextureUploadBytesBuffer,
	TextureUploadBytesStaging,
	TextureUploadBytesAsync,
	// Guest write-fault invalidations answered without the texture-cache lock because no
	// registered image covers the faulting 1 MiB page (KYTY_TEXTURE_FAULT_FAST_PATH).
	TextureInvalidateSkips,
	// Texel-buffer reads of image-backed memory (BufferCache::SynchronizeBufferFromImage):
	// image downloads recorded, and downloads skipped because neither the image nor the buffer
	// changed since the previous one (KYTY_TEXEL_SYNC_SKIP).
	TexelImageSyncDownloads,
	TexelImageSyncSkips,
	// Texel-read syncs of an image over stale GPU-dirty buffer bytes (KYTY_IMAGE_SUPERSEDES_GPU_DIRTY).
	TexelImageSyncOverGpuDirty,
	// Colour metadata draws (fast clear eliminate, FMASK/DCC decompress) consumed without drawing,
	// and targets of them resolved to materialize a DCC clear (KYTY_CB_METADATA_MATERIALIZE).
	MetadataColorOps,
	MetadataColorOpMaterializations,
	// Resident mip levels (KYTY_TEXTURE_RESIDENT_MIPS): images created holding only the levels
	// their views can sample and the levels left out, guest bytes their refreshes did not
	// upload, residency extensions (a finer MIN_LOD, or any non-sampling use), extensions to
	// the whole chain for non-sampling uses, partially resident images retired by an unmap of
	// their non-resident bytes, and GPU writes that reached a partially resident image (0).
	TextureResidentImages,
	TextureResidentLevelsSkipped,
	TextureResidentBytesSkipped,
	TextureResidencyExtensions,
	TextureResidencyFullFallbacks,
	TextureResidencyUnmapFrees,
	TextureResidencyViolations,
	// Residency extensions whose newly registered bytes another registered image already covers,
	// and those where that image is GPU-modified (the extended image's refresh then reads guest
	// memory the other image's native contents supersede; expected 0).
	TextureResidencyExtensionOverlaps,
	TextureResidencyExtensionGpuOverlaps,
	// Partially resident images retired after KYTY_TEXTURE_RESIDENT_IDLE_FRAMES unused frames.
	TextureResidentIdleFrees,
	// Headerless shader code hashed from a clean-backing copy (no fault possible) or in place
	// through the guest mapping (KYTY_SHADER_HASH_BACKING).
	ShaderCodeHashBacking,
	ShaderCodeHashDirect,
	// GPU-thread reads of GPU-owned bytes (KYTY_READBACK_SIDE_GPU_THREAD): served by a side copy
	// the GPU thread waited for, or by the drain path (fallback reason also counted above).
	ReadbackGpuThreadSideCopies,
	ReadbackGpuThreadDrains,
	// Side copies submitted to the second queue of the family (KYTY_SIDE_QUEUE).
	ReadbackSideQueueCopies,
	// Visibility-proxy end dumps (KYTY_OCCLUSION_PROXY_MODE); end-of-pipe labels deferred to their
	// tick's completion (all, armed by a proxy dump, kept behind an older deferred label to the
	// same address); WAIT_REG_MEM suspensions on a pending deferred label, and those that first
	// flushed the recording holding it.
	OcclusionProxyDumps,
	LabelWritesDeferred,
	LabelWritesDeferredProxy,
	LabelWritesDeferredOrdered,
	WaitRegMemDeferredLabel,
	WaitRegMemDeferredLabelFlushes,
	// Completion-runner (priority) operations run, and broadcasts to WaitPriorityOperations /
	// DrainPriorityOperations waiters (KYTY_PRIORITY_WAKE_BATCH: only at a tick boundary with waiters).
	PriorityOperationsRun,
	PriorityWaiterWakeups,
	// CP scheduler with every queue suspended (KYTY_CP_WAKEUPS): short retry spins, timed sleeps,
	// and completion/label/flip notifications that unblocked suspended queues.
	CpBlockedSpins,
	CpBlockedSleeps,
	CpProgressWakeups,
	// WAIT_FLIP_DONE packets that suspended their queue (KYTY_FLIP_WAIT_MODE=suspend) or blocked
	// the GPU thread (=block).
	FlipWaitSuspends,
	FlipWaitBlocking,
	// GDS end-of-pipe reads (RELEASE_MEM data_sel=5 / event source 1), and those snapshotted and
	// written at completion instead of draining the GPU (KYTY_GDS_EOP_MODE=defer).
	GdsEopReads,
	GdsEopReadsDeferred,
	// Early submits because the GPU had finished all submitted work (KYTY_IDLE_FLUSH_DRAWS), outside
	// and inside an active rendering instance.
	IdleFlushes,
	IdleFlushesInPass,
	// Graphics slices ended early so runnable async compute queues could run (KYTY_GFX_SLICE_DRAWS).
	GfxSliceYields,
	// End-of-pipe data writes Kyty used to drop (KYTY_EOP_DROPPED_LABELS): graphics-queue events
	// with INT_SEL=1, and RELEASE_MEM with INT_SEL=4 and DATA_SEL 1/2/3.
	EopLabelsIntSel1,
	ReleaseMemLabelsIntSel4,
	// CP WRITE_DATA packets recorded on the GPU timeline because their destination was owned by
	// recorded GPU work (KYTY_WRITE_DATA_GPU), and those written by the CPU at parse time.
	WriteDataGpu,
	WriteDataCpu,
	// AgcSuspendPoint calls in bounded mode (KYTY_AGC_DONE_MODE) that had to wait for the previous
	// frame's submissions.
	AgcDoneBoundedWaits,
	// Queue fronts the CP scheduler passed over because an earlier guest frame's submissions had
	// not completed yet (KYTY_FRAME_FENCE), counted once per scheduling pass.
	FrameFenceHolds,
	// Texture binding identity memo (KYTY_TEXTURE_BINDING_MEMO, pipeline/textureBindingMemo.h).
	// ResolveTexture answered from an entry (Hits), with no entry for the key (Misses), or with an
	// entry that a texture-cache structure change or the image's live state ruled out (Stale: new
	// page version, stencil association, pending rebind, residency extension needed, alias
	// partner not owned). Full resolutions recorded (Fills) or not memoizable (Rejects:
	// overlap/rebased answers, DCC, stencil redirect, non-first-page lookup). Sampled views
	// answered from an entry (ViewHits) or by FindTexture after the entry could not be used
	// (ViewMisses: first use, dirty/untracked/under-resident image). Hits whose binding already
	// held the description (DescCopiesAvoided). StructureChanges: registrations,
	// unregistrations and stencil associations (each bumps the versions of the pages it covers).
	TextureBindingMemoHits,
	TextureBindingMemoMisses,
	TextureBindingMemoStale,
	TextureBindingMemoFills,
	TextureBindingMemoRejects,
	TextureViewMemoHits,
	TextureViewMemoMisses,
	TextureBindingDescCopiesAvoided,
	TextureCacheStructureChanges,
	// Descriptor commit. Renderer pipeline layouts created, and pipelines that received an
	// existing interned layout (KYTY_LAYOUT_INTERN, pipeline/pipelineLayoutCache.h).
	PipelineLayoutsCreated,
	PipelineLayoutsShared,
	// vkCmdPushConstants recorded by descriptor commits, and identical updates skipped by the
	// per-command-buffer shadow (KYTY_PUSH_CONSTANT_SHADOW).
	PushConstantUpdates,
	PushConstantUpdatesAvoided,
	// Push-descriptor updates recorded (DescriptorPushesAvoided counts the skipped ones) and, for
	// renderer commits, why an update was needed: no comparable earlier update in this command
	// buffer for the bind point, or another layout (Layout); a different binding list (Shape); or
	// the kind of the first descriptor that differs: image, sampler, guest storage buffer, per-draw
	// upload (flattened SRT or shader data), other (GDS, BDA page table, fault buffer, mip stats).
	DescriptorPushes,
	DescriptorPushMissLayout,
	DescriptorPushMissShape,
	DescriptorPushMissImage,
	DescriptorPushMissSampler,
	DescriptorPushMissBuffer,
	DescriptorPushMissUpload,
	DescriptorPushMissOther,
	// Layouts beyond maxPushDescriptors (KYTY_DESCRIPTOR_SET_REUSE): sets allocated and written,
	// sets reused because this command buffer already wrote one with the same layout and
	// contents, and binds skipped because that set was still bound.
	DescriptorSetsWritten,
	DescriptorSetsReused,
	DescriptorSetBindsAvoided,
	// Per-draw shader-data/flattened-SRT uploads answered by the previous upload of the same stage
	// slot and kind before hashing (KYTY_UPLOAD_DEDUP); also counted in ShaderUploadReuseHits.
	ShaderUploadLastHits,
	// Guest memory tracking and buffer uploads (graphics/host_gpu/memoryStats.h, mirrored in the
	// hang-trace mem_* columns): guest write/read faults handled by RenderContext and the time
	// spent handling them (nanoseconds), host protection calls/pages removing write access and
	// restoring read-write access and their time, contended region tracking lock acquisitions,
	// tiler scratch buffer allocations (count, bytes, nanoseconds), SynchronizeBufferFromImage
	// downloads, and the copy commands, pipeline barriers and rendering-instance ends recorded by
	// CPU-dirty buffer uploads (BufferCache::SynchronizeBuffer).
	GuestWriteFaults,
	GuestReadFaults,
	GuestFaultNanoseconds,
	PageProtectCalls,
	PageProtectPages,
	PageUnprotectCalls,
	PageUnprotectPages,
	PageProtectNanoseconds,
	TrackerLockContended,
	TilerScratchAllocs,
	TilerScratchBytes,
	TilerScratchNanoseconds,
	BufferFromImageSyncs,
	BufferUploadCopies,
	BufferUploadBarriers,
	BufferUploadRenderSplits,
	// GPU-modified images moved into a buffer before a GPU write took their ownership
	// (BufferCache::PreserveImagesForGpuWrite): downloads, bytes, downloads covering only leading
	// mips, and overlapping images that could not be moved (their contents are lost as before).
	ImageWritebacks,
	ImageWritebackBytes,
	ImageWritebackPartial,
	ImageWritebackSkips,
	// Write-fault policy (KYTY_FAULT_AHEAD_KB, KYTY_HOT_PAGES, MemoryTracker): pages made CPU-dirty
	// ahead of use by a write fault, pages entering and leaving hot (sticky-dirty, unprotected)
	// tracking, hot pages visited by buffer uploads, and those skipped as unchanged.
	FaultAheadPages,
	HotPagePromotions,
	HotPageDemotions,
	HotPageUploads,
	HotPageUploadsSkipped,
	// Written-upload pages a racing guest write re-dirtied while they were copied outside the
	// tracker locks (KYTY_UPLOAD_COPY_OUTSIDE_LOCK), copied again under the locks.
	WrittenUploadLatePages,
	// Image::CopyImageWithBuffer rounds (barrier, image->buffer copy of all packed regions,
	// barrier, buffer->image copy); ImageCopyViaBuffer counts the copies themselves.
	ImageCopyViaBufferRounds,
	// Draw-prep S5/S6 (KYTY_DRAW_PREP=inline|parallel). Submitted: direct draws handed to the
	// engine. Published: entered the preparation window (parallel). Ready: prepared by a worker
	// before the command processor needed it. SelfPrepared: prepared on the command processor
	// (inline mode, or the head slot was still unclaimed). Committed: the prepared programs were
	// used after a valid certificate. Unused: the draw returned before its programs were needed.
	DrawPrepSubmitted,
	DrawPrepPublished,
	DrawPrepReady,
	DrawPrepSelfPrepared,
	DrawPrepCommitted,
	DrawPrepUnused,
	// Serial fallbacks per reason. Ineligible: tessellation, O15 reuse or a draw the engine does
	// not take. Unclean/Backing/Overflow/Inconsistent/Uncertified: preparation reads (readSet.h).
	// NotPublished: a source or permutation was missing. ShaderMap: the shader map changed before
	// commit. CertUnclean/CertChanged: a read range was not clean, or held other bytes, at commit.
	// CoherenceLog: the log check failed (KYTY_DRAW_PREP_CERT=log). Mismatch: the commit's own
	// pixel-activity or export-mapping decision differed from the snapshot's.
	DrawPrepFallbackIneligible,
	DrawPrepFallbackUnclean,
	DrawPrepFallbackBacking,
	DrawPrepFallbackOverflow,
	DrawPrepFallbackInconsistent,
	DrawPrepFallbackUncertified,
	DrawPrepFallbackNotPublished,
	DrawPrepFallbackShaderMap,
	DrawPrepFallbackCertUnclean,
	DrawPrepFallbackCertChanged,
	DrawPrepFallbackCoherenceLog,
	DrawPrepFallbackMismatch,
	// Certificate sizes (coalesced ranges and bytes of committed certificates).
	DrawPrepCertRanges,
	DrawPrepCertBytes,
	// KYTY_DRAW_PREP_VERIFY: prepared draws re-prepared serially, and differences.
	DrawPrepVerifyChecks,
	DrawPrepVerifyMismatches,
	// S0 window measurement: fences (packets that end a preparation window) and the number of
	// direct draws parsed since the previous fence, bucketed. Also counted with
	// KYTY_DRAW_PREP_HISTOGRAM=1 in off mode.
	DrawPrepFences,
	DrawPrepFenceDraws0,
	DrawPrepFenceDraws1,
	DrawPrepFenceDraws2To3,
	DrawPrepFenceDraws4To7,
	DrawPrepFenceDraws8To15,
	DrawPrepFenceDraws16To31,
	DrawPrepFenceDraws32To63,
	DrawPrepFenceDraws64Plus,
	// Parallel mode: drains (window committed because of a fence or a full window), the summed
	// window occupancy observed at each publish (mean = sum / Published), and head slots a worker
	// was still preparing when the command processor needed them (with KYTY_DRAW_PREP_STEAL it
	// prepares other slots meanwhile: DrawPrepSteals).
	DrawPrepDrains,
	DrawPrepWindowOccupancy,
	DrawPrepCommitWaits,
	// KYTY_DRAW_PREP_LOG_AUDIT=1 with value certificates: commits the log check would have
	// refused although the bytes were unchanged (LogWouldReject), and certificates whose ranges
	// were clean with no intersecting log entry but whose bytes had changed (LogMissed: the
	// log-mode certificate would have accepted stale bytes).
	DrawPrepLogWouldReject,
	DrawPrepLogMissed,
	// New compute pipelines (graphics ones are GraphicsPipelinesCreated). Programs and pipelines
	// are also recorded per compile in the hang trace (compiles.csv).
	ComputePipelinesCreated,
	// Program compiles run outside the exclusive programs lock: requests that waited for another
	// thread's compile of the same source or permutation, and finished compiles dropped because an
	// equal permutation had been published meanwhile.
	ProgramCompileWaits,
	ProgramCompileDuplicates,
	// Program permutations specialized from a source's kept translation instead of translating
	// the guest code again (KYTY_TRANSLATION_CACHE), and reused translations that
	// KYTY_TRANSLATION_CACHE_VERIFY found different from a fresh one.
	TranslationReuses,
	TranslationVerifyMismatches,
	// Persistent program cache (KYTY_PROGRAM_CACHE): permutations reloaded from disk instead of
	// translated and emitted, permutations emitted with the cache on (not stored), and
	// KYTY_PROGRAM_CACHE_VERIFY comparisons that found a stored record different.
	ProgramDiskHits,
	ProgramDiskMisses,
	ProgramDiskVerifyMismatches,
	// Graphics pipeline libraries (KYTY_PIPELINE_LIBRARY): pipelines fast-linked from libraries,
	// monolithic pipelines found in the driver cache instead, and libraries created.
	PipelineLibraryLinks,
	PipelineLibraryCacheHits,
	PipelineLibrariesCreated,
	// KYTY_BDA_HOT_SYNC (BufferCache::SynchronizeBdaBuffers): BDA passes that re-synchronized only
	// the hot page runs of the last full pass (the fault and structure epochs held), and the runs
	// they visited. BdaSyncPasses/BdaSyncScannedBuffers count the full passes only.
	BdaSyncHotPasses,
	BdaSyncHotRanges,
	// KYTY_UPLOAD_BATCH_SCOPED_FLUSH (BufferCache::UploadBatch): batch scopes that ended with
	// barriers pending but no queued upload, and so left them to the next flush point instead of
	// recording them (and ending the rendering instance) right away.
	UploadBatchFlushesDeferred,
	// KYTY_BDA_HOT_SYNC_VERIFY: hot passes re-checked by a full scan, and CPU-dirty non-hot pages
	// that scan found while the hot pass's epochs still held (pages the hot pass missed; 0).
	BdaSyncHotVerifyChecks,
	BdaSyncHotVerifyMismatches,
	// Draw-prep window fences by packet kind (DrawPrep::FenceKind, packetClass.h), counted with the
	// S0 histogram above in inline and parallel modes; in off mode only with aggregates and
	// KYTY_DRAW_PREP_FENCE_HISTOGRAM=1.
	DrawPrepFenceRegIndirect,
	DrawPrepFenceEventWrite,
	DrawPrepFenceEndOfPipe,
	DrawPrepFenceAcquireMem,
	DrawPrepFenceWait,
	DrawPrepFenceDataWrite,
	DrawPrepFenceConstantEngine,
	DrawPrepFenceMarker,
	DrawPrepFenceDispatch,
	DrawPrepFenceIndirectDraw,
	DrawPrepFenceContextControl,
	DrawPrepFenceOther,
	// Direct image transfers (KYTY_TILER_IMAGE_DIRECT, TileManager::DetileToImage/TileFromImage):
	// uploads that detiled straight into the image and downloads that tiled straight from it
	// (no buffer<->image copy), with the element bytes moved.
	TilerImageUploads,
	TilerImageUploadBytes,
	TilerImageDownloads,
	TilerImageDownloadBytes,
	// KYTY_TILER_IMAGE_DIRECT_VERIFY=1: direct transfers compared with the buffer path, and those
	// whose bytes differed.
	TilerImageVerifyChecks,
	TilerImageVerifyMismatches,
	// KYTY_DEPTH_FEEDBACK_KEEP: attachment <-> attachment+shader-read barriers of a sampled,
	// unwritten depth attachment left out inside one rendering instance.
	DepthFeedbackBarriersAvoided,
	// Why rendering instances end and barriers appear between draws (aggregate mode):
	// - barrier requests queued in the batcher, by origin (BarrierOrigin order, render.h);
	// - a second barrier of an image already pending, which records the pending batch first
	//   (ImageBarrierSameImageFlushes), and those of them that ended an active instance;
	// - draw bindings sampling an image that is also a color or depth attachment of the draw
	//   (a feedback read: its access toggles with the attachment's between such draws);
	// - sampled depth attachments that did not take KYTY_DEPTH_FEEDBACK_KEEP: the draw writes the
	//   attachment, no read-only proof for the active instance (first draw of an instance, or an
	//   earlier draw of it wrote), the contents changed since the proof, or the tracked
	//   layout/access/range differ.
	BarrierRequestsGuest,
	BarrierRequestsShaderAccess,
	BarrierRequestsShaderWrite,
	BarrierRequestsShaderWriteHazard,
	BarrierRequestsIndirectArgs,
	BarrierRequestsGds,
	BarrierRequestsImage,
	BarrierRequestsUpload,
	ImageBarrierSameImageFlushes,
	ImageBarrierSameImageRenderEnds,
	SampledColorAttachmentBindings,
	SampledDepthAttachmentBindings,
	DepthFeedbackKeepMissWrite,
	DepthFeedbackKeepMissInstance,
	DepthFeedbackKeepMissSerial,
	DepthFeedbackKeepMissState,
	// Draw-prep preparations that hashed headerless shader code through their recorder, making
	// the code part of the certificate (KYTY_DRAW_PREP_CODE_CERT, shader.cpp).
	ShaderCodeHashCertified,
	// SET_*_REG_INDIRECT packets that kept the draw-prep window open because their register pairs
	// were clean (KYTY_DRAW_PREP_REG_INDIRECT_WINDOW); the others count as DrawPrepFenceRegIndirect.
	DrawPrepRegIndirectKept,
	// KYTY_TILER_IMAGE_DIRECT_BC: direct uploads into block-compressed images (also counted in
	// TilerImageUploads/UploadBytes).
	TilerImageBlockUploads,
	TilerImageBlockUploadBytes,
	// Eager readback publication (BufferCache, KYTY_READBACK_EAGER): read-hot pages registered by
	// a readback; copies recorded at a submission (and their dirty bytes); candidates kept for a
	// later submission (writer in the current recording, pending readback or publication, no
	// slot); readers that waited for or finished a pending eager copy; pages unprotected at
	// completion or kept protected by a newer writer; early submissions after a recorded writer
	// of a page the GPU thread reads back.
	ReadbackEagerHotPages,
	ReadbackEagerCopies,
	ReadbackEagerCopyBytes,
	ReadbackEagerRetries,
	ReadbackEagerWaits,
	ReadbackEagerPagesUnmarked,
	ReadbackEagerPagesRetained,
	ReadbackEagerFlushes,
	// Why an indirect draw with GPU-owned arguments took the CPU-read path (the CPU read then
	// synchronizes with the arguments' writer): host feature/alignment/mapping limits, no usable
	// index buffer range, a target operation mode, legacy quads, a restart value only an index
	// scan can decide, or a mesh (merged NGG) vertex stage.
	DrawIndirectFallbackHost,
	DrawIndirectFallbackIndexBuffer,
	DrawIndirectFallbackTargetOp,
	DrawIndirectFallbackQuadList,
	DrawIndirectFallbackRestart,
	DrawIndirectFallbackMesh,
	// KYTY_NATIVE_INDIRECT_MESH (renderer/meshIndirect.h): indirect mesh draws recorded with GPU
	// arguments; those skipped without a read because INDEX_BUFFER_SIZE makes every record empty;
	// why an indirect mesh draw kept the CPU-read path (a multi-draw or count record, primitive
	// restart split by an index scan, a depth/stencil clear-enable draw, a program without the
	// indirect parameters); the completion checks (verify modes), their mismatches, and
	// conversions whose status word reported a difference from the CPU path.
	MeshIndirectDraws,
	MeshIndirectAlwaysEmpty,
	MeshIndirectDeclinedMulti,
	MeshIndirectDeclinedRestart,
	MeshIndirectDeclinedClear,
	MeshIndirectDeclinedProgram,
	MeshIndirectVerifyChecks,
	MeshIndirectVerifyMismatches,
	MeshIndirectStatus,
	// KYTY_SET_REUSE_FRESH: descriptor-set commits that skipped the reuse lookup because the set
	// refers to a shader-data or flattened-SRT upload made for this draw (no earlier set can hold it).
	DescriptorSetReuseSkippedFresh,
	// KYTY_BUFFER_RANGE_MEMO (bufferCache.h): read-only synchronizations skipped because the
	// range's tracker bits did not change since one that found nothing to upload; small read
	// bindings that reused their stream-copy decision; facts recorded; and the verify mode's
	// checks and mismatches (a skipped synchronization that would have uploaded, or a changed
	// decision).
	BufferRangeMemoCleanHits,
	BufferRangeMemoStreamHits,
	BufferRangeMemoRecords,
	BufferRangeMemoVerifyChecks,
	BufferRangeMemoVerifyMismatches,
	// KYTY_HOT_PAGE_CHECK_LIMIT: hot pages returned to normal tracking (settled clean) after that
	// many consecutive uploads found them unchanged.
	HotPageCheckSettles,
	// Bytes committed draw-prep certificates covered by digest (KYTY_DRAW_PREP_CODE_DIGEST: the
	// code of headerless shaders; DrawPrepCertBytes counts the byte-compared ranges).
	DrawPrepCertDigestBytes,
	// Draws binding what the draw before them bound (renderDraw.cpp CountBindingRepeats, opt-in
	// with KYTY_DRAW_BINDING_REPEAT_STATS=1): the same programs; and also the same images and
	// samplers; and also the same buffers; and also the same user data and flattened SRT words.
	DrawBindingRepeatPrograms,
	DrawBindingRepeatTextures,
	DrawBindingRepeatResources,
	DrawBindingRepeatAll,
	// KYTY_TEXTURE_MEMO_REVALIDATE: texture binding memo entries whose first page changed its owner
	// list and that the redone lookup confirmed (counted as hits too), and the verify mode's
	// disagreements with the full resolution.
	TextureBindingMemoRevalidated,
	TextureBindingMemoRevalidateMismatches,
	// KYTY_BUFFER_DIRTY_QUERY_COMBINED: small-read stream decisions from one locked query.
	BufferDirtyQueriesCombined,
	// KYTY_BACKING_INPLACE: shader-code hashes taken on the backing bytes in place (and of those,
	// under the mapping lock); DrawPrep::Validate ranges compared or hashed in place (and locked).
	BackingInPlaceHashes,
	BackingInPlaceHashesLocked,
	DrawPrepValidateInPlaceRanges,
	DrawPrepValidateInPlaceLocked,
	// KYTY_BACKING_INPLACE_VERIFY: in-place vs copied validations; Races = differed once, then
	// agreed on a second run (a guest write between them).
	DrawPrepValidateVerifyChecks,
	DrawPrepValidateVerifyMismatches,
	DrawPrepValidateVerifyRaces,
	// KYTY_DRAW_PREP_LOCKFREE_HINT_VERIFY: worker hints checked against the locked tracker bits.
	DrawPrepHintVerifyChecks,
	DrawPrepHintVerifyMismatches,
	// KYTY_DEPTH_LAYOUT_STABLE: draws whose unsampled depth target kept its current layout where
	// the per-draw layout would have transitioned it (and ended the rendering instance).
	DepthLayoutTransitionsAvoided,
	// KYTY_LOD_STATS_PLAIN_VARIANT: draws of GET_LOD_STATS-instrumented pixel programs that used the
	// plain (feedback-free) variant because no image had a counter, and those that needed the
	// instrumented one; verify mode: canary checks at GET_LOD_STATS and counters found changed.
	LodStatsPlainDraws,
	LodStatsInstrumentedDraws,
	LodStatsCanaryChecks,
	LodStatsCanaryMismatches,
	// KYTY_UPLOAD_DMA: buffer uploads whose host -> VRAM part went to the copy engine (count,
	// bytes), those kept on the graphics queue because the ring was full, transfer-queue
	// submissions, and guest submissions that had to wait for an unfinished transfer.
	UploadDmaCopies,
	UploadDmaBytes,
	UploadDmaRingFull,
	UploadDmaSubmits,
	UploadDmaSubmitWaits,
	// KYTY_UPLOAD_DMA_VERIFY=1: staged uploads compared, and those whose ring bytes differed.
	UploadDmaVerifyChecks,
	UploadDmaVerifyMismatches,
	// KYTY_CMASK_FAST_CLEAR (TextureCache::MaterializeCmaskClear), per bound render-target slice
	// with CMASK fast clears enabled: cleared to the CLEAR_WORD colour (all CMASK bytes 0, decided
	// on the CPU); nothing pending (all 0xFF); no proof (nonuniform or other bytes, or no native
	// inspection possible); CMASK bytes an image covers; clear colour format not decoded;
	// unsupported surface shape. Bindings whose GPU-owned CMASK bytes a native inspection
	// decides on the GPU (clearing when all bytes are 0), and inspections reused for unchanged
	// bytes.
	CmaskFastClears,
	CmaskFastClearExpanded,
	CmaskFastClearUnproven,
	CmaskFastClearAliased,
	CmaskFastClearFormat,
	CmaskFastClearShape,
	CmaskFastClearInspections,
	CmaskFastClearInspectionReuses,
	// Bindings that read the CMASK bytes back because no native inspection was possible
	// (device limits or image shape); the wait is counted in FrameWait::DccFallback.
	CmaskFastClearReadbacks,
	// Log-mode draw-prep certificates (the default) whose coherence-log check did not pass: a
	// logged transition intersected a certified range (Conflicts), an entry could not be read
	// (Unknown), or the interval was older than the ring (Overflows). Each is then decided by the
	// value check; ValueRescues counts those it accepted (the bytes were unchanged).
	DrawPrepLogConflicts,
	DrawPrepLogUnknown,
	DrawPrepLogOverflows,
	DrawPrepLogValueRescues,
	// KYTY_BUFFER_RANGE_MEMO_VERIFY: differences explained by a page turned CPU-dirty after the
	// memo hit (a guest write fault racing the re-evaluation), not counted as mismatches.
	BufferRangeMemoVerifyRaces,
	// KYTY_TRACKER_RELAXED_QUERIES: small-read decisions and read synchronizations skipped from
	// the lock-free dirty mirrors; KYTY_TRACKER_RELAXED_VERIFY checks (Races: another thread's
	// transition separated the two answers).
	TrackerRelaxedQueries,
	TrackerRelaxedSyncSkips,
	TrackerRelaxedVerifyChecks,
	TrackerRelaxedVerifyMismatches,
	TrackerRelaxedVerifyRaces,
	// KYTY_SYNC_EPOCH: epoch advances; KYTY_BDA_SYNC_EPOCH: BDA passes skipped within an epoch and
	// their verify mode (Mismatches: pages a skip missed that no guest write explains).
	SyncEpochAdvances,
	BdaSyncEpochSkips,
	BdaSyncEpochVerifyChecks,
	BdaSyncEpochVerifyMismatches,
	// KYTY_BDA_SYNC_PER_SUBMISSION: BDA passes skipped because the guest submission is unchanged.
	BdaSyncSubmissionSkips,
	// KYTY_BINDING_EPOCH_MEMO: read bindings reused within a sync epoch (a stream copy or a cache
	// buffer), results recorded, and the verify mode's checks, mismatches and races.
	BindingEpochMemoStreamHits,
	BindingEpochMemoCachedHits,
	BindingEpochMemoRecords,
	BindingEpochMemoVerifyChecks,
	BindingEpochMemoVerifyMismatches,
	BindingEpochMemoVerifyRaces,
	// Tracker-gap detectors (RenderContext::NoteHostBackingWrite, NoteGuestProtection): emulator
	// writes of guest bytes outside publications and the GPU-dirty and clean tracked pages they
	// land on; guest protection changes, the watched pages of GPU memory they cover and those
	// whose tracking protection they replace, and those that restrict access to GPU memory.
	HostBackingWrites,
	HostBackingWriteGpuDirtyPages,
	HostBackingWriteCleanPages,
	GuestProtectCalls,
	GuestProtectWatchedPages,
	GuestProtectOverriddenPages,
	GuestProtectRestrictsGpuMemory,
	// KYTY_CLAMP_RANGE_MEMO: lookups the per-thread committed-run cache could not answer, and its
	// verify mode's mismatches.
	ClampRangeMemoMisses,
	ClampRangeMemoVerifyMismatches,
	// KYTY_DRAW_SEQUENCE_FAST, targets (TextureCache::RepeatLookup): lookups answered by a proven
	// repeat of the slot's last one, recorded lookups whose proofs no longer held, and lookups
	// recorded as repeatable.
	DrawSequenceTargetRepeats,
	DrawSequenceTargetMisses,
	DrawSequenceTargetRecords,
	// KYTY_DRAW_SEQUENCE_FAST, textures: stages whose texture resolution repeated the last one
	// (TextureBindingMemo::TryRepeatResolve), stages with the same program and T# words whose
	// repeat was not proven, and stages whose views were all TryAcquireView hits (TryRepeatViews).
	DrawSequenceTextureRepeats,
	DrawSequenceTextureMisses,
	DrawSequenceViewRepeats,
	// KYTY_DRAW_SEQUENCE_VERIFY: reuses checked against the full path, disagreements, and
	// disagreements a guest write racing the check explains.
	DrawSequenceVerifyChecks,
	DrawSequenceVerifyMismatches,
	DrawSequenceVerifyRaces,
	// Draw-prep reads refused as not provably clean, by cause (DrawPrepFallbackUnclean; with the
	// hang trace also unclean.csv): a worker's GPU-dirty hint (tracker GPU-dirty pages) or pending
	// backing publication, the GPU thread's exact predicate, and an exact read without a backing.
	DrawPrepUncleanHintGpuDirty,
	DrawPrepUncleanHintPublication,
	DrawPrepUncleanExact,
	DrawPrepUncleanBacking,
	// KYTY_DRAW_SEQUENCE_FAST, textures: stages whose T# words matched one of their earlier sets
	// (PreparedBindings::texture_history), before that set's repeat was checked.
	DrawSequenceTextureHistoryHits,
	// KYTY_EQUEUE_COALESCE: kernel-event triggers merged into an event that was already pending
	// (instead of queued), and those of them that replaced a different data value.
	EqueueCoalescedTriggers,
	EqueueCoalescedDataChanges,
	// KYTY_DRAW_PREP_HOT: parked (cold) draw-prep workers woken because the unclaimed backlog grew.
	DrawPrepColdWakes,
	// KYTY_RENDER_STATE_FAST (colorRenderTarget.h): prepared vertex inputs copied as their used
	// prefixes, or in full (a count out of range); draws whose target entries were not
	// value-initialised first; stage preparations swapped member by member. VerifyChecks/Mismatches:
	// KYTY_RENDER_STATE_VERIFY comparisons with the full copy, swap or resolution.
	RenderStateVertexPartialCopies,
	RenderStateVertexFullCopies,
	RenderStateMinimalResets,
	RenderStateMemberSwaps,
	RenderStateVerifyChecks,
	RenderStateVerifyMismatches,
	// KYTY_DESCRIPTOR_SET_REUSE_AUDIT (render.h): descriptor-set commits, those whose exact
	// contents an earlier commit of the same command buffer had, and those a 64-slot cache indexed
	// by a full digest of the contents would have held (compare DescriptorSetsReused).
	DescriptorSetAuditCommits,
	DescriptorSetAuditRepeats,
	DescriptorSetAuditDigestSlotHits,
	// KYTY_UPLOAD_DMA_HOST_COPY (uploadDma.h): staged upload copies whose guest bytes the DMA
	// worker copied into the staging ring instead of the command processor, and those bytes.
	UploadDmaHostCopies,
	UploadDmaHostCopyBytes,
	// KYTY_DEFER_UNPROTECT: write-watcher releases whose host unprotect waited for the end of the
	// thread's scope (spans), the host calls those updates made, the spans that needed none
	// (another thread had applied them), spans applied at once because the thread's batch was full,
	// and write faults taken inside a scope that applied their page at once.
	DeferredUnprotectSpans,
	DeferredUnprotectCalls,
	DeferredUnprotectSettled,
	DeferredUnprotectOverflows,
	DeferredUnprotectNestedFaults,
	// KYTY_TRACKER_LOCK_PARK: tracking-lock waiters that parked after spinning their budget.
	TrackerLockParks,
	// KYTY_DEFER_UNPROTECT=verify: protection checks, and pages found looser than their watchers
	// ask or different from what was just set.
	ProtectVerifyChecks,
	ProtectVerifyMismatches,
	// KYTY_CP_RECORDER (commandRecorder.h): packets and bytes encoded, command buffers handed to
	// the recorder, drains (CP waits until the recorder caught up, before native recording in a
	// direct window), producer waits for ring space, wakes of a parked recorder, recorder parks,
	// verify checks and mismatches (argument hash, sequence, digest, ownership), and placement
	// samples (the recorder's processor against the CP's; SameCore: one physical core).
	CpRecorderPackets,
	CpRecorderBytes,
	CpRecorderSubmits,
	CpRecorderDrains,
	CpRecorderRingFullWaits,
	CpRecorderWakes,
	CpRecorderParks,
	CpRecorderVerifyChecks,
	CpRecorderVerifyMismatches,
	CpRecorderPlacementSamples,
	CpRecorderSameCoreSamples,
	// Drains that found the recorder idle (everything executed): no marker, no wake.
	CpRecorderIdleDrains,
	// KYTY_DRAW_PREP_STEAL: slots the command processor prepared, as a worker would, while a worker
	// held the head it had to commit (FrameWait DrawPrepSteal is their time).
	DrawPrepSteals,
	// KYTY_DRAW_PREP_BINDINGS (drawPrep/bindingPlan.h): binding plans computed by the preparing
	// threads; plans the command processor used (DrawPrep::Validate accepted the slot's
	// preparation) or dropped (a fallback, or a draw that returned before its programs); items a
	// preparing thread left to the command processor: a V# or vertex range its clamp would change,
	// a sampler not created yet, a pipeline not created yet (or a key the serial path refuses), the
	// pipeline cache busy (its lock taken); plan items not used at commit because the guest virtual
	// ranges changed or the pipeline's certificate failed (targets, topology, restart, generation);
	// pipelines taken from plans; verify mode's comparisons and differences.
	DrawPrepBindingPlans,
	DrawPrepBindingPlansUsed,
	DrawPrepBindingPlansDropped,
	DrawPrepBindingAbstainRanges,
	DrawPrepBindingAbstainSamplers,
	DrawPrepBindingAbstainPipeline,
	DrawPrepBindingAbstainPipelineBusy,
	DrawPrepBindingFallbackVm,
	DrawPrepBindingFallbackPipeline,
	DrawPrepBindingPipelinesUsed,
	DrawPrepBindingVerifyChecks,
	DrawPrepBindingVerifyMismatches,
	// KYTY_DRAW_PREP_BINDINGS hwcheck, dynamic (P4b-1b): draws whose uc_check/hw_check the plan
	// found quiet (skipped), and draws whose viewports and scissors came from their plan.
	DrawPrepBindingHwChecksSkipped,
	DrawPrepBindingViewportsUsed,
	// KYTY_DRAW_PREP_BINDINGS texturememo (P4b-2): texture bindings a preparing thread found a
	// memo entry for (hints), and bindings resolved and views acquired in runs of memo hits under
	// one texture-cache lock.
	DrawPrepBindingTextureHints,
	DrawPrepBindingTextureRunHits,
	DrawPrepBindingViewRunHits,
	// KYTY_DESCRIPTOR_OFFSET_AUDIT=1 (P4b-D0, descriptors.cpp): descriptor sets written and push
	// updates, how many repeat the previous one or any earlier one of the same command buffer
	// with the per-draw buffer descriptors' offsets ignored (the reuse potential of dynamic
	// offsets), and how many have more per-draw buffer descriptors than the dynamic limit.
	DescriptorOffsetAuditSets,
	DescriptorOffsetAuditSetRepeatsPrevious,
	DescriptorOffsetAuditSetRepeatsAny,
	DescriptorOffsetAuditPushes,
	DescriptorOffsetAuditPushRepeatsPrevious,
	DescriptorOffsetAuditPushRepeatsAny,
	DescriptorOffsetAuditOverLimit,
	// KYTY_EOP_TIMESTAMPS=gpu: guest clock writes rewritten with GPU times, left alone because the
	// guest had written the slot again, kept at record time (no query slot, result or calibration),
	// and given GPU times through deferred label writes; gpu-verify checks and mismatches.
	EopTimestampsRewritten,
	EopTimestampsSkipped,
	EopTimestampsUnavailable,
	EopTimestampsDeferred,
	EopTimestampsVerifyChecks,
	EopTimestampsVerifyMismatches,
	// KYTY_BDA_DIRTY_LOG (bufferCache.h): BDA passes that synchronized only the logged dirtied
	// ranges (and the recorded hot runs), those ranges, logs that overflowed (a full scan
	// followed); KYTY_BDA_DIRTY_LOG_VERIFY checks and the CPU-dirty pages a logged pass missed.
	BdaSyncLogPasses,
	BdaSyncLogRanges,
	BdaSyncLogOverflows,
	BdaSyncLogVerifyChecks,
	BdaSyncLogVerifyMismatches,
	// KYTY_HOST_WRITE_TRACKING (RenderContext::PrepareHostBackingWrite): emulator writes of guest
	// bytes that gave their clean tracked pages the transition of a guest write fault.
	HostBackingWritesTracked,
	// KYTY_BINDING_MEMO_CROSS_EPOCH (BufferCache::ObtainReadBinding): cache-buffer memos from an
	// earlier sync epoch reused (included in BindingEpochMemoCachedHits) or refused (CPU-dirty pages
	// or a racing transition), and why the other lookups missed: no memo of that range in the slot,
	// a tracker transition in the range's regions, a changed buffer structure or stream tick, or an
	// earlier epoch that could not be crossed (stream memos, or the switch off).
	BindingEpochMemoCrossHits,
	BindingEpochMemoCrossRejects,
	BindingEpochMemoMissSlot,
	BindingEpochMemoMissSignature,
	BindingEpochMemoMissGuard,
	BindingEpochMemoMissEpoch,
	// KYTY_WRITTEN_SYNC_SKIP (BufferCache::SynchronizeBuffer): written synchronizations of entirely
	// GPU-owned ranges skipped, and their verify mode.
	WrittenSyncSkips,
	WrittenSyncSkipVerifyChecks,
	WrittenSyncSkipVerifyMismatches,
	// KYTY_PENDING_OPS_NOWAIT: draw/dispatch pops that left completed operations queued because
	// the priority runner had not finished their tick, and the queue depth summed over them.
	PendingOpsDeferred,
	PendingOpsDeferredDepth,
	// KYTY_PRIORITY_WAIT_SPIN_US: WaitPriorityOperations spins, and spins that saw the wait end.
	PriorityWaitSpins,
	PriorityWaitSpinHits,
	// Placement samples (cpuPlacement.h): the CP, and off its reserved core; guest threads, and on
	// the physical core of the CP's latest sample; draw-prep workers and service threads, and on
	// that core; the recorder off its reserved core (cp+recorder). Threads found with a hard affinity
	// other than the process's, those confined to reserved cores, and those moved to the general
	// processors (KYTY_CPU_RESERVE_REPIN), each thread counted once.
	CpuPlacementCpSamples,
	CpuPlacementCpOffCore,
	CpuPlacementGuestSamples,
	CpuPlacementGuestOnCpCore,
	CpuPlacementHostSamples,
	CpuPlacementHostOnCpCore,
	CpuPlacementRecorderOffCore,
	CpuPlacementHardAffinity,
	CpuPlacementHardAffinityReserved,
	CpuPlacementRepinned,
	// KYTY_FALSE_SHARING_WRITES (BufferCache::TryFalseSharingWrite): write faults that released a
	// GPU-owned page without draining the GPU, the GPU-owned bytes left to their publication,
	// uploads that skipped such bytes, and the verify mode (CPU writes to them before publication).
	FalseSharingWrites,
	FalseSharingBytes,
	FalseSharingUploadSplits,
	FalseSharingVerifyChecks,
	FalseSharingVerifyConflicts,
	// KYTY_GPU_WRITE_IMAGE_SKIP (TextureCache::SkipGpuWriteImageWalk): GPU buffer writes whose
	// image checks found no registered image without the texture-cache lock, and the verify mode.
	GpuWriteImageSkips,
	GpuWriteImageSkipVerifyChecks,
	GpuWriteImageSkipVerifyRaces,
	GpuWriteImageSkipVerifyMismatches,
	// DccFallbackImageState by cause (TextureCache::TryMaterializeGpuMetadataClear refused the GPU
	// inspection of a DCC slice's image), checked in this order: the image is not registered, has a
	// stencil association, does not match the view's description, or is not safe to download
	// because it is buffer-modified, CPU-dirty, not GPU-modified, partially resident, or has
	// GPU-dirty buffer bytes. (Enumerated in declaration order below, which differs.)
	DccImageStateUnregistered,
	DccImageStateStencil,
	DccImageStateMismatch,
	DccImageStateNotGpuModified,
	DccImageStateBufferModified,
	DccImageStateCpuDirty,
	DccImageStatePartial,
	DccImageStateGpuDirtyBytes,
	// ReadbackSideFallbackOther by cause (BufferCache::TryIssueSideReadback): no registered owner,
	// unaligned window, read larger than the side-copy window, a pending backing publication, no
	// GPU-dirty bytes in the window, no free side-copy slot.
	ReadbackSideOtherOwner,
	ReadbackSideOtherAlignment,
	ReadbackSideOtherWindow,
	ReadbackSideOtherPublication,
	ReadbackSideOtherNoDirty,
	ReadbackSideOtherSlot,
	// KYTY_DCC_GPU_REFRESH (TextureCache::TryMaterializeGpuMetadataClear): DCC slices inspected on
	// the GPU for an image refreshed first instead of the CPU fallback, and the verify mode.
	DccGpuRefreshes,
	DccGpuRefreshVerifyChecks,
	DccGpuRefreshVerifyMismatches,
	// KYTY_DRAW_PREP_CERT_RANGES_VERIFY (drawPrep.cpp): certificate ranges built by the preparing
	// thread compared at commit with the commit-time list, and the differences.
	DrawPrepCertRangesVerifyChecks,
	DrawPrepCertRangesVerifyMismatches,
	// KYTY_IMAGE_LRU_SKIP (TextureCache::TouchImage): LRU touches skipped because the image was
	// touched in the current GC tick already, and the verify mode's mirror checks.
	ImageLruTouchSkips,
	ImageLruVerifyChecks,
	ImageLruVerifyMismatches,
	// KYTY_IMAGE_TRANSIT_SKIP (Image::Transit): transitions that return before GetBarriers
	// because nothing would change, and the verify mode's checks of that decision.
	ImageTransitSkips,
	ImageTransitVerifyChecks,
	ImageTransitVerifyMismatches,
	// KYTY_DRAW_SEQUENCE_FAST target records (TextureCache::FindImage) that end invalid, so the
	// next lookup of the target is a full FindImage, by the first reason (lookups of a null image,
	// which cost nothing, are not counted):
	//   NotFirstPage - the image did not come from the first-page lookup (overlap or new image);
	//   Changed      - the first-page answer changed during the lookup (residency, alias sync),
	//                  or the image is unregistered or a video-out surface;
	//   Dcc*         - the DCC decision was not a provable no-op: a recorded-fill clear was applied
	//                  (Clear), the native inspection (Native) or the readback (Fallback) decided
	//                  GPU-owned bytes, guest bytes decided (Guest), or the metadata pages could not
	//                  be captured (Pages);
	//   Cmask*       - the CMASK decision: GPU-owned bytes without a recorded fill (Native), or any
	//                  other non-provable decision (guest bytes, a clear, fills).
	TargetRecordNotFirstPage,
	TargetRecordChanged,
	TargetRecordDccClear,
	TargetRecordDccNative,
	TargetRecordDccFallback,
	TargetRecordDccGuest,
	TargetRecordDccPages,
	TargetRecordCmaskNative,
	TargetRecordCmaskOther,
	// KYTY_CP_SEQ (cpOps.h): ops handed from the front to the back through the op ring;
	// KYTY_CP_SEQ_VERIFY: ops compared with the reference front, ops or stream ends that
	// differed, and front reads whose bytes differed from the serial read.
	CpSeqOps,
	CpSeqVerifyChecks,
	CpSeqVerifyMismatches,
	CpSeqVerifyReadDivergences,
	// KYTY_CP_SEQ=1 (cpSequencer.h): the sequencer's lockstep waits (ordering points), guest reads
	// and command buffers it had the resolver read in order (GPU-touched pages), stops at pending
	// CP writes, register snapshots, publishes that found the draw-prep window full, and waits of
	// the front alone (REWIND); the resolver finding no op.
	CpSeqBarriers,
	CpSeqLockstepReads,
	CpSeqLockstepBuffers,
	CpSeqPendingWriteStops,
	CpSeqSnapshots,
	CpSeqWindowFull,
	CpSeqFrontWaits,
	CpSeqResolverStarved,
	// Guest reads the sequencer made directly (never GPU-touched pages): with CpSeqLockstepReads
	// the share the sticky bitmap lets through.
	CpSeqDirectReads,
	// Draw-prep log certificates checked at commit, and the log entries they walked (the cost of
	// longer preparation-to-commit intervals: entries per check).
	DrawPrepLogChecks,
	DrawPrepLogEntries,
	// KYTY_CP_SEQ=1: the sequencer's lockstep waits by kind. WAIT_REG_MEM on the destination of the
	// last end-of-pipe label the front emitted (the "wait for idle" idiom) or on another address;
	// WAIT_FLIP_DONE; predication, COND_EXEC and conditional branches. Lockstep reads are
	// CpSeqLockstepReads/Buffers. DrawBursts: barriers after which the sequencer published a draw
	// before its next barrier (the draw-prep window restarts empty there).
	CpSeqBarrierWaitSelfLabel,
	CpSeqBarrierWaitOther,
	CpSeqBarrierFlipWait,
	CpSeqBarrierCondition,
	CpSeqBarrierDrawBursts,
	// Draw-prep commit waits (DrawPrepCommitWaits) by what the command processor found when it
	// began to wait: the head was the first draw published after a sequencer barrier (Barrier) or
	// after another stop of the sequencer (Start: a submission start, the frame fence, a handoff,
	// a pending CP write), the window held at most two published draws (Shallow: the commit caught
	// up with the publication), or more (Deep: the preparing threads were behind). Unclaimed: waits
	// that began while another published slot was waiting for a preparing thread.
	DrawPrepCommitWaitsBarrier,
	DrawPrepCommitWaitsShallow,
	DrawPrepCommitWaitsDeep,
	DrawPrepCommitWaitsUnclaimed,
	DrawPrepCommitWaitsStart,
	// KYTY_CP_SEQ_PREFETCH (P3c, cpOps.h): speculative parses run at waits, the draws they
	// published, the ones the real parse adopted, the first differing draw of a parse (its slots
	// are dropped), and the slots the resolver retired unused. Stops: why speculative parses ended
	// before the answer came (a lockstep op, a read they could not prove clean, a pending CP write
	// over bytes they read, the window full or the draw cap, the stream end).
	CpSeqPrefetchRuns,
	CpSeqPrefetchDraws,
	CpSeqPrefetchAdopted,
	CpSeqPrefetchMismatches,
	CpSeqPrefetchSkipped,
	CpSeqPrefetchStopLockstep,
	CpSeqPrefetchStopRead,
	CpSeqPrefetchStopWindow,
	CpSeqPrefetchStopEnd,
	// Commit waits (DrawPrepCommitWaits) on an adopted slot: its speculative preparation had not
	// finished when the command processor needed it.
	CpSeqPrefetchAdoptedWaits,
	// KYTY_SUBMIT_MIN_INTERVAL_US: optional submits (idle flushes, end-of-pipe batches) put off
	// because the last submit was too recent (counted per check).
	SubmitIntervalDeferrals,
	// KYTY_CP_COMMIT=dccguest (renderer/cpCommit.h): target lookups on CPU-owned DCC metadata
	// recorded as provable repeats, and repeats refused because a key or the GPU ownership changed.
	CpCommitDccGuestRecords,
	CpCommitDccGuestRejects,
	// KYTY_CP_COMMIT=texdcc: DCC texture descriptions the texture binding memo recorded with their
	// decision's certificate, and memo lookups whose certificate no longer held.
	CpCommitTexDccRecords,
	CpCommitTexDccRejects,
	Count,
};
// Counted while aggregate diagnostics are on and a profiler was connected at the last guest flip
// (defined below).
inline void CountFrameEvent(FrameEvent kind, uint64_t amount = 1);

enum class FrameWait : uint32_t {
	ReadMemory,
	ShaderReadiness,
	DccFallback,
	ResourceMaterialization,
	DriverSubmit,
	ShaderProgramMiss,
	GraphicsPipelineCreate,
	ResourceReuseValidation,
	NativeImageCreate,
	NativeImageDestroy,
	// KYTY_GPU_TIMING (GpuTiming::OnGuestFlip), added once per guest flip. These are GPU
	// timeline spans, not CPU scopes. GpuBusy: calls = timed command buffers, time = union of
	// their [top-of-pipe, all-commands] spans. GpuIdle: calls = gaps between merged spans
	// (including the gap after the previous flip's last span), time = their sum. GpuStarved: the
	// part of those gaps before the next buffer's vkQueueSubmit returned. GpuRecordToStart /
	// GpuDispatchToStart / GpuEndToObserved: calls = calibrated samples, time = summed latency
	// from recording start / native submit to GPU start, and from GPU end to CPU collection.
	// Latencies and GpuStarved need VK_KHR/EXT_calibrated_timestamps.
	GpuBusy,
	GpuIdle,
	GpuStarved,
	GpuRecordToStart,
	GpuDispatchToStart,
	GpuEndToObserved,
	// Guest thread time waiting for a side-copy readback (its own or a duplicate) and publishing
	// it to the backing. Nested inside ReadMemory; the GPU thread does not wait for these.
	ReadbackSideWait,
	// Draw-prep S3: GPU-thread time spent spinning in the join after its own stage finished
	// (StagePrepJoin), and helper-thread job time (StagePrepHelper, not on the GPU thread).
	StagePrepJoin,
	StagePrepHelper,
	// GPU-thread CPU time of TextureCache guest-memory -> image refreshes (staging copies,
	// detile and copy recording), full or partial.
	TextureUpload,
	// StagingCopier worker time copying guest texture bytes into staging (not the GPU thread).
	TextureStagingCopy,
	// GPU (CP) thread blocked in MasterSemaphore::Wait (submission dispatch plus timeline wait;
	// waits that find the tick already complete are not counted), attributed to the caller that
	// set a ScopedGpuWaitReason: ReadMemory drains, occlusion publication waits (sync proxy,
	// predication on unpublished dumps, publish-slot reuse), stream-buffer wraps, LOD-stats slot
	// reuse, unmaps, predication/boolean waits, GDS end-of-pipe reads, side-copy waits of GPU
	// thread reads, and everything untagged (GpuWaitOther).
	GpuWaitDrain,
	GpuWaitOcclusion,
	GpuWaitStreamWrap,
	GpuWaitLodStats,
	GpuWaitUnmap,
	GpuWaitPredication,
	GpuWaitGds,
	GpuWaitSideCopy,
	GpuWaitOther,
	// Guest thread time in AgcSuspendPoint (GuestGpu::Done): the idle wait, or the bounded wait
	// for the previous frame's submissions (KYTY_AGC_DONE_MODE).
	AgcDoneWait,
	// Draw-prep: command-processor time spinning for a worker's head slot (CommitWaitNs; with
	// KYTY_DRAW_PREP_STEAL only once no other slot was left to prepare, see DrawPrepSteal),
	// preparation time on any thread, and certificate validation time at commit.
	DrawPrepCommitWait,
	DrawPrepPrepare,
	DrawPrepValidate,
	// Phases of a new program permutation, nested in ShaderProgramMiss: TranslateProgram,
	// CompileProgram (specialization + SPIR-V emission), spirv-val and vkCreateShaderModule.
	ShaderTranslate,
	ShaderEmit,
	ShaderValidate,
	ShaderModuleCreate,
	// Persistent program cache: key, lookups and decoding of stored plans and permutations.
	ShaderDiskLoad,
	// vkCreateGraphicsPipelines alone (nested in GraphicsPipelineCreate), and a whole new compute
	// pipeline (layouts and vkCreateComputePipelines).
	GraphicsPipelineDriver,
	ComputePipelineCreate,
	// Background optimized compiles replacing fast-linked pipelines (not on the GPU thread).
	PipelineOptimize,
	// KYTY_CP_RECORDER: CP time in drains and waiting for ring space; recorder-thread time
	// replaying packets (calls = batches).
	CpRecorderDrain,
	CpRecorderRingFull,
	CpRecorderExecute,
	// KYTY_DRAW_PREP_STEAL: command-processor time preparing stolen slots while a worker held the
	// head (DrawPrepCommitWait is then only the idle spin after nothing was left to claim).
	DrawPrepSteal,
	// KYTY_DRAW_PREP_BINDINGS: binding-plan time on the preparing threads (inside the worker times
	// below, or DrawPrepSteal on the command processor).
	DrawPrepBindingPlan,
	// Draw-prep worker load: the busy time of DrawPrep#k (preparation, repeat-trace hashes and
	// binding plan of each slot it claimed; calls = slots). Workers past the eighth add to
	// DrawPrepWorker8. Idle time is the flip interval minus this.
	DrawPrepWorker1,
	DrawPrepWorker2,
	DrawPrepWorker3,
	DrawPrepWorker4,
	DrawPrepWorker5,
	DrawPrepWorker6,
	DrawPrepWorker7,
	DrawPrepWorker8,
	// KYTY_EOP_TIMESTAMPS=gpu: command-processor time reading completed timestamp queries and
	// rewriting the guest slots (one call per publication).
	EopTimestampPublish,
	// KYTY_CP_SEQ=1: sequencer time waiting for the resolver (ordering points, pending CP writes,
	// a full op ring, window or snapshot ring), and resolver time waiting for the sequencer's next
	// op before it lets other queues run.
	CpSeqSequencerWait,
	CpSeqResolverStarved,
	// Draw-prep commits (Engine::Commit: the prepared draw recorded from its slot), per draw; with
	// KYTY_CP_SEQ=1 the resolver reads slots another core wrote (the snapshot cache-miss risk).
	DrawPrepCommit,
	// DrawPrepCommitWait split like the FrameEvent DrawPrepCommitWaits{Barrier,Shallow,Deep,Start}.
	DrawPrepCommitWaitBarrier,
	DrawPrepCommitWaitShallow,
	DrawPrepCommitWaitDeep,
	DrawPrepCommitWaitStart,
	// KYTY_CP_SEQ_PREFETCH: sequencer time in speculative parses (inside CpSeqSequencerWait).
	CpSeqPrefetch,
	Count,
};

namespace Detail {

// Where the aggregate counters go (CountFrameEvent, ScopedFrameWait, AddFrameWait):
//   Off    - nowhere: aggregate diagnostics are off, or no profiler was connected at the last
//            guest flip (PublishFrameWork refreshes it; Initialize sets it first);
//   Thread - the calling thread's counter block (ThreadCounters), summed at every guest flip;
//   Shared - KYTY_PROFILE_COUNTERS=shared, the previous implementation for comparisons: every
//            call checks the switches and the connection and adds to shared atomics.
// FrameWork counters follow frame-only mode instead: the calling thread's block, or the shared
// atomics with KYTY_PROFILE_COUNTERS=shared.
enum class CounterSink : uint8_t { Off, Thread, Shared };
extern std::atomic<CounterSink> g_event_sink;

constexpr size_t kFrameEventCount = static_cast<size_t>(FrameEvent::Count);
constexpr size_t kFrameWorkCount  = static_cast<size_t>(FrameWork::Count);
constexpr size_t kFrameWaitCount  = static_cast<size_t>(FrameWait::Count);

// One thread's counters, written only by that thread with a relaxed load and store (no locked
// instruction, no cache line shared with other writers) and summed by the flip publisher. Blocks
// are never freed: a thread's totals stay in the sums after it exits, and the next new thread
// reuses the released block and continues its totals (every published value is cumulative).
struct alignas(64) ThreadCounters {
	std::array<std::atomic<uint64_t>, kFrameEventCount> events {};
	std::array<std::atomic<uint64_t>, kFrameWorkCount>  work {};
	std::array<std::atomic<uint64_t>, kFrameWaitCount>  wait_calls {};
	std::array<std::atomic<uint64_t>, kFrameWaitCount>  wait_ns {};
	std::atomic<bool>                                    in_use {false};
	ThreadCounters*                                      next = nullptr; // registry, immutable
};
// constinit: other translation units read it directly, without the thread-local initialization
// guard an extern thread_local otherwise needs (which kept CurrentThreadCounters out of line).
extern constinit thread_local ThreadCounters* t_counters;
// Registers the calling thread's block (reusing a released one) and sets t_counters.
[[nodiscard]] ThreadCounters& AcquireThreadCounters() noexcept;
[[nodiscard]] inline ThreadCounters& CurrentThreadCounters() noexcept {
	auto* counters = t_counters;
	if (counters == nullptr) [[unlikely]] {
		counters = &AcquireThreadCounters();
	}
	return *counters;
}
// Owner-thread increment.
inline void Add(std::atomic<uint64_t>& counter, uint64_t amount) noexcept {
	counter.store(counter.load(std::memory_order_relaxed) + amount, std::memory_order_relaxed);
}
[[nodiscard]] inline bool SharedCounters() {
	const auto value = g_shared_counters.load(std::memory_order_relaxed);
	return value >= 0 ? value != 0 : ReadSharedCounters();
}
void CountFrameEventShared(FrameEvent kind, uint64_t amount) noexcept;
void CountFrameWorkShared(FrameWork kind) noexcept;
// Registered counter blocks (tests).
[[nodiscard]] size_t CounterBlockCount() noexcept;

} // namespace Detail

// The cumulative count of `kind` the next guest flip would publish (every thread's block plus the
// shared atomics). Diagnostics and tests.
[[nodiscard]] uint64_t FrameEventTotal(FrameEvent kind);

inline void CountFrameEvent(FrameEvent kind, uint64_t amount) {
	const auto sink = Detail::g_event_sink.load(std::memory_order_relaxed);
	if (sink == Detail::CounterSink::Off) [[likely]] {
		return;
	}
	if (sink == Detail::CounterSink::Thread) [[likely]] {
		Detail::Add(Detail::CurrentThreadCounters().events[static_cast<size_t>(kind)], amount);
		return;
	}
	Detail::CountFrameEventShared(kind, amount);
}

inline void CountFrameWork(FrameWork kind) {
	if (!FramesOnlyEnabled()) [[likely]] {
		return;
	}
	if (Detail::SharedCounters()) [[unlikely]] {
		Detail::CountFrameWorkShared(kind);
		return;
	}
	Detail::Add(Detail::CurrentThreadCounters().work[static_cast<size_t>(kind)], 1);
}

// Aggregate inclusive CPU work/wait durations, not GPU time or an additive frame budget.
// ReadMemory may nest within DCC fallback, shader readiness or resource materialization,
// and may itself be reentrant. DriverSubmit can overlap renderer work on another thread.
// ShaderProgramMiss includes translation and may nest resource materialization; it counts
// miss attempts/retries. ShaderProgramsCreated separately counts compiled permutations.
// GraphicsPipelineCreate excludes cache hits and includes native layout/pipeline setup.
// Completed scopes contribute only while connected with aggregate diagnostics enabled. Scopes
// spanning an on-demand connection change are omitted; no per-call zones are emitted.
class ScopedFrameWait {
public:
	explicit ScopedFrameWait(FrameWait kind): m_kind(kind) {
		if (Detail::g_event_sink.load(std::memory_order_relaxed) != Detail::CounterSink::Off)
		    [[unlikely]] {
			Begin();
		}
	}
	ScopedFrameWait(const ScopedFrameWait&) = delete;
	ScopedFrameWait& operator=(const ScopedFrameWait&) = delete;
	ScopedFrameWait(ScopedFrameWait&&) = delete;
	ScopedFrameWait& operator=(ScopedFrameWait&&) = delete;
	~ScopedFrameWait() {
		if (m_sink != Detail::CounterSink::Off) [[unlikely]] {
			Finish();
		}
	}

private:
	void Begin();
	void Finish();

	FrameWait           m_kind;
	uint64_t            m_start_ns   = 0;
	uint64_t            m_connection = 0;
	// Where the scope is counted (Off: not counted).
	Detail::CounterSink m_sink = Detail::CounterSink::Off;
};

// Adds externally measured totals (the GPU timeline entries above) with the same gating as
// ScopedFrameWait: aggregate diagnostics enabled and a connected profiler.
void AddFrameWait(FrameWait kind, uint64_t calls, uint64_t nanoseconds);

// Per-thread attribution of GPU waits (the GpuWait* categories): the innermost scope wins.
class ScopedGpuWaitReason {
public:
	explicit ScopedGpuWaitReason(FrameWait reason);
	ScopedGpuWaitReason(const ScopedGpuWaitReason&)            = delete;
	ScopedGpuWaitReason& operator=(const ScopedGpuWaitReason&) = delete;
	ScopedGpuWaitReason(ScopedGpuWaitReason&&)                 = delete;
	ScopedGpuWaitReason& operator=(ScopedGpuWaitReason&&)      = delete;
	~ScopedGpuWaitReason();

private:
	FrameWait m_previous;
};
[[nodiscard]] FrameWait CurrentGpuWaitReason() noexcept;

// Call immediately after the existing completed guest-flip marker. Snapshots
// include workload and wait totals, and are cumulative so an on-demand connection
// can discard its first sample. Independently atomic values can straddle a flip.
void PublishFrameWork();

// Opt-in loading diagnostics collected even before a Tracy client connects. A separate
// host thread publishes cumulative totals once a second, independently of guest flips.
[[nodiscard]] bool LoadingEnabled();
enum class LoadingEvent : uint32_t {
	AprRecordsCompleted,
	AprRequestedBytes,
	AprHostReadBytes,
	AprGuestCopiedBytes,
	AprErrors,
	AprShortReads,
	AmmErrors,
	PreadRequestedBytes,
	PreadReadBytes,
	PreadErrors,
	LodStatsPackets,
	LodStatsBufferBytes,
	LodStatsReportAndResetPackets,
	LodStatsForceResetPackets,
	LodStatsResetCountSum,
	LodStatsInterval100kSum,
	LodStatsCachePolicySum,
	DirectMemorySizeQueries,
	DirectMemoryAvailableQueries,
	DirectMemoryAvailableSuccesses,
	DirectMemoryAvailableBytesSum,
	PageTableStatsQueries,
	PageTableStatsSuccesses,
	PageTableCpuAvailableSum,
	PageTableGpuAvailableSum,
	PageTableCpuZeroAvailable,
	PageTableGpuZeroAvailable,
	FlexibleAvailableQueries,
	FlexibleAvailableSuccesses,
	FlexibleAvailableBytesSum,
	MemoryPoolStatsQueries,
	MemoryPoolAvailableBytesSum,
	AmmUsageStatsQueries,
	AmmOccupancyThresholdCalls,
	AmmOccupancyThresholdSum,
	AprReadCommandsAppended,
	AprCommandResets,
	AprCommandBufferSets,
	AprCommandBufferClears,
	AprSubmitAndGetResult,
	AprSubmitPlain,
	AprSubmitAndGetId,
	AprWaitCompleted,
	AprWaitUnknownId,
	AprUnsupportedWaitCommands,
	AprUnsupportedCounterCommands,
	AprDiagnosticRowsDropped,
	Count,
};
void CountLoadingEvent(LoadingEvent kind, uint64_t amount = 1);

// Request metadata only: no asset contents or faulting guest reads. The first 512
// submissions and every 257th thereafter are eligible, up to 8192 queued rows per
// process. The independent loading publisher drains the bounded queue to a CSV.
struct LoadingAprSubmission {
	uint64_t sequence = 0, timestamp_ns = 0, api_kind = 0;
	uint64_t command_buffer = 0, argument1 = 0, argument2 = 0;
	uint64_t result_address = 0, out_id_address = 0, submission_id = 0;
	uint64_t resolved_object = 0, generation = 0, append_serial = 0;
	uint64_t submits_in_generation = 0, unchanged_submits = 0;
	uint64_t cached_buffer = 0, cached_size = 0, cached_offset = 0;
	uint64_t header_valid = 0, header_buffer = 0, header_size = 0, header_offset = 0;
	uint64_t header_commands = 0, header_type = 0, header_mismatch = 0;
	uint64_t read_count = 0, event_count = 0, write_count = 0, map_count = 0;
	uint64_t commands_signature = 0, first_read_id = 0, first_read_destination = 0;
	uint64_t first_read_size = 0, first_read_offset = 0, read_bytes = 0;
	uint64_t execution_observed = 0, execution_result = 0, error_offset = 0;
	uint64_t submit_result = 0, api_kernel_result = 0;
};
[[nodiscard]] bool BeginLoadingAprSubmission(LoadingAprSubmission* record);
void RecordLoadingAprSubmission(const LoadingAprSubmission& record);

enum class LoadingOperation : uint32_t {
	AprSubmit,
	AprRead,
	AprHostRead,
	AprGuestCopy,
	AmmMap,
	AmmUnmap,
	Pread,
	PreadFileMutex,
	PreadCoherence,
	PreadNativeRead,
	Count,
};

// Completed means the scope returned, including errors. Durations are inclusive,
// may overlap across threads/nested scopes, and become visible only on completion.
class ScopedLoadingOperation {
public:
	explicit ScopedLoadingOperation(LoadingOperation kind);
	ScopedLoadingOperation(const ScopedLoadingOperation&) = delete;
	ScopedLoadingOperation& operator=(const ScopedLoadingOperation&) = delete;
	ScopedLoadingOperation(ScopedLoadingOperation&&) = delete;
	ScopedLoadingOperation& operator=(ScopedLoadingOperation&&) = delete;
	~ScopedLoadingOperation();
	void End();

private:
	LoadingOperation m_kind;
	uint64_t m_start_ns = 0;
	bool m_active = false;
};

void Initialize();
void Shutdown();

struct Lifecycle {
	static constexpr const char* name               = "Profiler";
	static constexpr auto        initialize         = Profiler::Initialize;
	static constexpr auto        shutdown           = Profiler::Shutdown;
	static constexpr auto        emergency_shutdown = Profiler::Shutdown;
};

} // namespace Profiler

#define KYTY_PROFILER_CONCAT_IMPL(a, b) a##b
#define KYTY_PROFILER_CONCAT(a, b)      KYTY_PROFILER_CONCAT_IMPL(a, b)
#define KYTY_PROFILER_COLOR_OR_DEFAULT(default_color, ...)                                         \
	KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color __VA_OPT__(, ) __VA_ARGS__, default_color)
#define KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color, color, ...) color

#define KYTY_PROFILER_BLOCK(name, ...)                                                             \
	KYTY_PROFILER_BLOCK_IMPL(__LINE__, true, name __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_DETAIL_BLOCK(name, ...)                                                      \
	KYTY_PROFILER_BLOCK_IMPL(__LINE__, Profiler::DetailedEnabled(), name __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_BLOCK_IMPL(line, active, name, ...)                                          \
	static constexpr tracy::SourceLocationData KYTY_PROFILER_CONCAT(                               \
	    kyty_profiler_source_location_, line) {name, TracyFunction, TracyFile,                     \
	                                           static_cast<uint32_t>(line),                        \
	                                           KYTY_PROFILER_COLOR_OR_DEFAULT(0, __VA_ARGS__)};    \
	Profiler::ScopedBlock KYTY_PROFILER_CONCAT(kyty_profiler_block_, line)(                        \
	    &KYTY_PROFILER_CONCAT(kyty_profiler_source_location_, line), active)

#define KYTY_PROFILER_FUNCTION(...)                                                               \
	KYTY_PROFILER_FUNCTION_IMPL(__LINE__, true __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_DETAIL_FUNCTION(...)                                                        \
	KYTY_PROFILER_FUNCTION_IMPL(__LINE__, Profiler::DetailedEnabled() __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_FUNCTION_IMPL(line, active, ...)                                             \
	static constexpr tracy::SourceLocationData KYTY_PROFILER_CONCAT(                               \
	    kyty_profiler_source_location_, line) {nullptr, TracyFunction, TracyFile,                  \
	                                           static_cast<uint32_t>(line),                        \
	                                           KYTY_PROFILER_COLOR_OR_DEFAULT(0, __VA_ARGS__)};    \
	Profiler::ScopedBlock KYTY_PROFILER_CONCAT(kyty_profiler_block_, line)(                        \
	    &KYTY_PROFILER_CONCAT(kyty_profiler_source_location_, line), active)

#define KYTY_PROFILER_END_BLOCK Profiler::EndBlock()

#define KYTY_PROFILER_THREAD(name) Profiler::SetThreadName(name)

#endif /* KYTY_COMMON_PROFILER_H_ */
