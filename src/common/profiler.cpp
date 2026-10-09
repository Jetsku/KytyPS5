#include "common/profiler.h"

#include "common/emulatorConfig.h"
#include "common/hangTrace.h"
#include "common/hangWatchdog.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <tracy/Tracy.hpp>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

namespace {

thread_local std::vector<Profiler::ScopedBlock*> g_block_stack;

constexpr size_t kFrameWorkCount = static_cast<size_t>(Profiler::FrameWork::Count);
std::array<std::atomic<uint64_t>, kFrameWorkCount> g_frame_work {};
std::atomic<uint64_t> g_completed_flips {0};
constexpr std::array<const char*, kFrameWorkCount> kFrameWorkNames {
    "FrameWork.DrawIndex.Cumulative",
    "FrameWork.DrawAuto.Cumulative",
    "FrameWork.DrawIndirectCommands.Cumulative",
    "FrameWork.DrawIndirectMultiCommands.Cumulative",
    "FrameWork.DispatchDirectCommands.Cumulative",
    "FrameWork.DispatchIndirectCommands.Cumulative",
};

constexpr size_t kFrameEventCount = static_cast<size_t>(Profiler::FrameEvent::Count);
std::array<std::atomic<uint64_t>, kFrameEventCount> g_frame_events {};
constexpr std::array<const char*, kFrameEventCount> kFrameEventNames {
    "FrameEvent.SubmitBoundaryUnprotected.Cumulative",
    "FrameEvent.SubmitBoundaryEopOnly.Cumulative",
    "FrameEvent.SubmitBoundaryEopMixed.Cumulative",
    "FrameEvent.SubmitBoundaryGeneric.Cumulative",
    "FrameEvent.MeshDraws.Cumulative",
    "FrameEvent.MeshRestartEnabledDraws.Cumulative",
    "FrameEvent.MeshInputIndices.Cumulative",
    "FrameEvent.MeshWorkgroups.Cumulative",
    "FrameEvent.ShaderProgramsCreated.Cumulative",
    "FrameEvent.GraphicsPipelinesCreated.Cumulative",
    "FrameEvent.SrtProbeHits.Cumulative",
    "FrameEvent.SrtProbeMisses.Cumulative",
    "FrameEvent.SrtProbeBytes.Cumulative",
    "FrameEvent.SrtProbeBatchHits.Cumulative",
    "FrameEvent.SrtUnmappedReads.Cumulative",
    "FrameEvent.ResourceReuseHits.Cumulative",
    "FrameEvent.ResourceReuseMisses.Cumulative",
    "FrameEvent.ResourceReuseValidationBytes.Cumulative",
    "FrameEvent.ResourceReuseRejectedCaptures.Cumulative",
    "FrameEvent.SrtRecipeSessions.Cumulative",
    "FrameEvent.SrtRecipeCompiledNodes.Cumulative",
    "FrameEvent.SrtRecipeFallbackNodes.Cumulative",
    "FrameEvent.SrtRecipeMemoHits.Cumulative",
    "FrameEvent.SrtTapeExecutions.Cumulative",
    "FrameEvent.SrtTapeOperations.Cumulative",
    "FrameEvent.SrtTapeBoundaryCalls.Cumulative",
    "FrameEvent.BdaSyncPasses.Cumulative",
    "FrameEvent.BdaSyncSkips.Cumulative",
    "FrameEvent.BdaSyncScannedBuffers.Cumulative",
    "FrameEvent.BdaSyncUploadBytes.Cumulative",
    "FrameEvent.BdaSyncUploadCopies.Cumulative",
    "FrameEvent.ShaderHeaderProbeHits.Cumulative",
    "FrameEvent.ShaderHeaderProbeMisses.Cumulative",
    "FrameEvent.VertexMetadataProbeHits.Cumulative",
    "FrameEvent.VertexMetadataProbeMisses.Cumulative",
    "FrameEvent.NativeImagePoolHits.Cumulative",
    "FrameEvent.NativeImagePoolMisses.Cumulative",
    "FrameEvent.NativeImagePoolRetires.Cumulative",
    "FrameEvent.NativeImagePoolAddedBytes.Cumulative",
    "FrameEvent.DccKnownFillClears.Cumulative",
    "FrameEvent.NativeImagePoolRemovedBytes.Cumulative",
    "FrameEvent.TextureCleanProofHits.Cumulative",
    "FrameEvent.TextureCleanProofMisses.Cumulative",
    "FrameEvent.TextureCleanProofStores.Cumulative",
    "FrameEvent.ResourceCacheKeyMatches.Cumulative",
    "FrameEvent.ResourceCacheBackingRejects.Cumulative",
    "FrameEvent.ResourceCacheHistoryHits.Cumulative",
    "FrameEvent.ResourceCacheBudgetRejects.Cumulative",
    "FrameEvent.ResourceCachePermutationHits.Cumulative",
    "FrameEvent.SrtSharedCleanMemoHits.Cumulative",
    "FrameEvent.UploadReservationsOutsideLocks.Cumulative",
    "FrameEvent.ShaderUploadReuseHits.Cumulative",
    "FrameEvent.ShaderUploadReuseMisses.Cumulative",
    "FrameEvent.ShaderUploadBytesAvoided.Cumulative",
    "FrameEvent.TextureDescriptionHits.Cumulative",
    "FrameEvent.TextureDescriptionMisses.Cumulative",
    "FrameEvent.PipelineBindsAvoided.Cumulative",
    "FrameEvent.DescriptorPushesAvoided.Cumulative",
    "FrameEvent.PredicatedPackets.Cumulative",
    "FrameEvent.PredicatedPacketsSkipped.Cumulative",
    "FrameEvent.OcclusionPredicates.Cumulative",
    "FrameEvent.OcclusionPredicatesPending.Cumulative",
    "FrameEvent.OcclusionCounterDumps.Cumulative",
    "FrameEvent.NativeOcclusionScopes.Cumulative",
    "FrameEvent.NativeOcclusionDumps.Cumulative",
    "FrameEvent.NativeOcclusionReductions.Cumulative",
    "FrameEvent.MeshRestartMarkers.Cumulative",
    "FrameEvent.MeshRestartSegments.Cumulative",
    "FrameEvent.DrawIndirectNative.Cumulative",
    "FrameEvent.DrawIndirectFallback.Cumulative",
    "FrameEvent.DrawIndirectInstanceReads.Cumulative",
    "FrameEvent.RtStubDispatches.Cumulative",
    "FrameEvent.RtStubDraws.Cumulative",
    "FrameEvent.GpuTimingDropped.Cumulative",
    "FrameEvent.CleanVerdictHits.Cumulative",
    "FrameEvent.CleanVerdictMisses.Cumulative",
    "FrameEvent.CleanVerdictStores.Cumulative",
    "FrameEvent.BackingMapCacheHits.Cumulative",
    "FrameEvent.BackingMapCacheMisses.Cumulative",
    "FrameEvent.ReadbackSideCopies.Cumulative",
    "FrameEvent.ReadbackSideCopyBytes.Cumulative",
    "FrameEvent.ReadbackSideDuplicateWaits.Cumulative",
    "FrameEvent.ReadbackSideFallbackCurrentWriter.Cumulative",
    "FrameEvent.ReadbackSideFallbackUnbounded.Cumulative",
    "FrameEvent.ReadbackSideFallbackOther.Cumulative",
    "FrameEvent.ReadbackSidePagesUnmarked.Cumulative",
    "FrameEvent.ReadbackSidePagesRetained.Cumulative",
    "FrameEvent.GpuRenderPassBegins.Cumulative",
    "FrameEvent.GpuPipelineBarriers.Cumulative",
    "FrameEvent.GpuImageLayoutTransitions.Cumulative",
    "FrameEvent.GpuGuestCommandBuffers.Cumulative",
    "FrameEvent.GpuBarrierRequests.Cumulative",
    "FrameEvent.GpuBarriersMerged.Cumulative",
    "FrameEvent.GpuBarriersElided.Cumulative",
    "FrameEvent.GpuBarriersSunk.Cumulative",
    "FrameEvent.GpuBarrierRenderSplits.Cumulative",
    "FrameEvent.ImageCopyDirect.Cumulative",
    "FrameEvent.ImageCopyMaintenance8.Cumulative",
    "FrameEvent.ImageCopyShaderDepthToColor.Cumulative",
    "FrameEvent.ImageCopyShaderColorToDepth.Cumulative",
    "FrameEvent.ImageCopyViaBuffer.Cumulative",
    "FrameEvent.ImageCopyD16.Cumulative",
    "FrameEvent.AliasSyncCopies.Cumulative",
    "FrameEvent.AliasSyncSkips.Cumulative",
    "FrameEvent.AliasBytesKept.Cumulative",
    "FrameEvent.AliasBytesBoundedClaims.Cumulative",
    "FrameEvent.AliasBytesMaterializations.Cumulative",
    "FrameEvent.AliasBytesMaterializedBytes.Cumulative",
    "FrameEvent.AliasBytesUnmaterialized.Cumulative",
    "FrameEvent.AliasBytesTexelReads.Cumulative",
    "FrameEvent.AliasBytesTexelScans.Cumulative",
    "FrameEvent.ImageUploads.Cumulative",
    "FrameEvent.ImageUploadBytes.Cumulative",
    "FrameEvent.BufferUploadBytes.Cumulative",
    "FrameEvent.StagePrepParallelDraws.Cumulative",
    "FrameEvent.StagePrepForkDeclined.Cumulative",
    "FrameEvent.StagePrepSpeculativeFailures.Cumulative",
    "FrameEvent.StagePrepLookupFallbacks.Cumulative",
    "FrameEvent.StagePrepVerifyMismatches.Cumulative",
    "FrameEvent.ProgramSourceMemoHits.Cumulative",
    "FrameEvent.ProgramSourceMemoMisses.Cumulative",
    "FrameEvent.PermutationMemoHits.Cumulative",
    "FrameEvent.PermutationMemoMisses.Cumulative",
    "FrameEvent.PipelineMemoHits.Cumulative",
    "FrameEvent.PipelineMemoMisses.Cumulative",
    "FrameEvent.SamplerMemoHits.Cumulative",
    "FrameEvent.SamplerMemoMisses.Cumulative",
    "FrameEvent.TargetDescMemoHits.Cumulative",
    "FrameEvent.TargetDescMemoMisses.Cumulative",
    "FrameEvent.DynamicStateCommandsEmitted.Cumulative",
    "FrameEvent.DynamicStateCommandsAvoided.Cumulative",
    "FrameEvent.VertexTableBatchHits.Cumulative",
    "FrameEvent.VertexTableBatchMisses.Cumulative",
    "FrameEvent.ShaderMapMemoHits.Cumulative",
    "FrameEvent.ShaderMapMemoMisses.Cumulative",
    "FrameEvent.WriteRangeNarrowed.Cumulative",
    "FrameEvent.WriteRangeWhole.Cumulative",
    "FrameEvent.WriteRangeBytesAvoided.Cumulative",
    "FrameEvent.WriteRangeImagesSpared.Cumulative",
    "FrameEvent.GpuRenderingEnds.Cumulative",
    "FrameEvent.GpuDrawWriteSinks.Cumulative",
    "FrameEvent.OcclusionScopesGated.Cumulative",
    "FrameEvent.DccGpuRecords.Cumulative",
    "FrameEvent.DccGpuReuses.Cumulative",
    "FrameEvent.DccCpuFallbacks.Cumulative",
    "FrameEvent.DccFallbackDisabled.Cumulative",
    "FrameEvent.DccFallbackBinding.Cumulative",
    "FrameEvent.DccFallbackShape.Cumulative",
    "FrameEvent.DccFallbackFormat.Cumulative",
    "FrameEvent.DccFallbackImageState.Cumulative",
    "FrameEvent.DccFallbackMetadataAliased.Cumulative",
    "FrameEvent.DccFallbackUnsupported.Cumulative",
    "FrameEvent.DccFallbackAlignment.Cumulative",
    "FrameEvent.TexturePartialUploads.Cumulative",
    "FrameEvent.TexturePartialUploadBytes.Cumulative",
    "FrameEvent.TexturePartialSkippedBytes.Cumulative",
    "FrameEvent.TexturePartialFallbacks.Cumulative",
    "FrameEvent.TextureChunkInvalidations.Cumulative",
    "FrameEvent.TexturePartialVerifyMismatches.Cumulative",
    "FrameEvent.TextureOverlapKeeps.Cumulative",
    "FrameEvent.TilerScratchPoolHits.Cumulative",
    "FrameEvent.TilerScratchPoolMisses.Cumulative",
    "FrameEvent.TextureAsyncCopies.Cumulative",
    "FrameEvent.TextureAsyncCopyBytes.Cumulative",
    "FrameEvent.TextureUploadBytesBuffer.Cumulative",
    "FrameEvent.TextureUploadBytesStaging.Cumulative",
    "FrameEvent.TextureUploadBytesAsync.Cumulative",
    "FrameEvent.TextureInvalidateSkips.Cumulative",
    "FrameEvent.TexelImageSyncDownloads.Cumulative",
    "FrameEvent.TexelImageSyncSkips.Cumulative",
    "FrameEvent.TexelImageSyncOverGpuDirty.Cumulative",
    "FrameEvent.MetadataColorOps.Cumulative",
    "FrameEvent.MetadataColorOpMaterializations.Cumulative",
    "FrameEvent.TextureResidentImages.Cumulative",
    "FrameEvent.TextureResidentLevelsSkipped.Cumulative",
    "FrameEvent.TextureResidentBytesSkipped.Cumulative",
    "FrameEvent.TextureResidencyExtensions.Cumulative",
    "FrameEvent.TextureResidencyFullFallbacks.Cumulative",
    "FrameEvent.TextureResidencyUnmapFrees.Cumulative",
    "FrameEvent.TextureResidencyViolations.Cumulative",
    "FrameEvent.TextureResidencyExtensionOverlaps.Cumulative",
    "FrameEvent.TextureResidencyExtensionGpuOverlaps.Cumulative",
    "FrameEvent.TextureResidentIdleFrees.Cumulative",
    "FrameEvent.ShaderCodeHashBacking.Cumulative",
    "FrameEvent.ShaderCodeHashDirect.Cumulative",
    "FrameEvent.ReadbackGpuThreadSideCopies.Cumulative",
    "FrameEvent.ReadbackGpuThreadDrains.Cumulative",
    "FrameEvent.ReadbackSideQueueCopies.Cumulative",
    "FrameEvent.OcclusionProxyDumps.Cumulative",
    "FrameEvent.LabelWritesDeferred.Cumulative",
    "FrameEvent.LabelWritesDeferredProxy.Cumulative",
    "FrameEvent.LabelWritesDeferredOrdered.Cumulative",
    "FrameEvent.WaitRegMemDeferredLabel.Cumulative",
    "FrameEvent.WaitRegMemDeferredLabelFlushes.Cumulative",
    "FrameEvent.PriorityOperationsRun.Cumulative",
    "FrameEvent.PriorityWaiterWakeups.Cumulative",
    "FrameEvent.CpBlockedSpins.Cumulative",
    "FrameEvent.CpBlockedSleeps.Cumulative",
    "FrameEvent.CpProgressWakeups.Cumulative",
    "FrameEvent.FlipWaitSuspends.Cumulative",
    "FrameEvent.FlipWaitBlocking.Cumulative",
    "FrameEvent.GdsEopReads.Cumulative",
    "FrameEvent.GdsEopReadsDeferred.Cumulative",
    "FrameEvent.IdleFlushes.Cumulative",
    "FrameEvent.IdleFlushesInPass.Cumulative",
    "FrameEvent.GfxSliceYields.Cumulative",
    "FrameEvent.EopLabelsIntSel1.Cumulative",
    "FrameEvent.ReleaseMemLabelsIntSel4.Cumulative",
    "FrameEvent.WriteDataGpu.Cumulative",
    "FrameEvent.WriteDataCpu.Cumulative",
    "FrameEvent.AgcDoneBoundedWaits.Cumulative",
    "FrameEvent.FrameFenceHolds.Cumulative",
    "FrameEvent.TextureBindingMemoHits.Cumulative",
    "FrameEvent.TextureBindingMemoMisses.Cumulative",
    "FrameEvent.TextureBindingMemoStale.Cumulative",
    "FrameEvent.TextureBindingMemoFills.Cumulative",
    "FrameEvent.TextureBindingMemoRejects.Cumulative",
    "FrameEvent.TextureViewMemoHits.Cumulative",
    "FrameEvent.TextureViewMemoMisses.Cumulative",
    "FrameEvent.TextureBindingDescCopiesAvoided.Cumulative",
    "FrameEvent.TextureCacheStructureChanges.Cumulative",
    "FrameEvent.PipelineLayoutsCreated.Cumulative",
    "FrameEvent.PipelineLayoutsShared.Cumulative",
    "FrameEvent.PushConstantUpdates.Cumulative",
    "FrameEvent.PushConstantUpdatesAvoided.Cumulative",
    "FrameEvent.DescriptorPushes.Cumulative",
    "FrameEvent.DescriptorPushMissLayout.Cumulative",
    "FrameEvent.DescriptorPushMissShape.Cumulative",
    "FrameEvent.DescriptorPushMissImage.Cumulative",
    "FrameEvent.DescriptorPushMissSampler.Cumulative",
    "FrameEvent.DescriptorPushMissBuffer.Cumulative",
    "FrameEvent.DescriptorPushMissUpload.Cumulative",
    "FrameEvent.DescriptorPushMissOther.Cumulative",
    "FrameEvent.DescriptorSetsWritten.Cumulative",
    "FrameEvent.DescriptorSetsReused.Cumulative",
    "FrameEvent.DescriptorSetBindsAvoided.Cumulative",
    "FrameEvent.ShaderUploadLastHits.Cumulative",
    "FrameEvent.GuestWriteFaults.Cumulative",
    "FrameEvent.GuestReadFaults.Cumulative",
    "FrameEvent.GuestFaultNanoseconds.Cumulative",
    "FrameEvent.PageProtectCalls.Cumulative",
    "FrameEvent.PageProtectPages.Cumulative",
    "FrameEvent.PageUnprotectCalls.Cumulative",
    "FrameEvent.PageUnprotectPages.Cumulative",
    "FrameEvent.PageProtectNanoseconds.Cumulative",
    "FrameEvent.TrackerLockContended.Cumulative",
    "FrameEvent.TilerScratchAllocs.Cumulative",
    "FrameEvent.TilerScratchBytes.Cumulative",
    "FrameEvent.TilerScratchNanoseconds.Cumulative",
    "FrameEvent.BufferFromImageSyncs.Cumulative",
    "FrameEvent.BufferUploadCopies.Cumulative",
    "FrameEvent.BufferUploadBarriers.Cumulative",
    "FrameEvent.BufferUploadRenderSplits.Cumulative",
    "FrameEvent.ImageWritebacks.Cumulative",
    "FrameEvent.ImageWritebackBytes.Cumulative",
    "FrameEvent.ImageWritebackPartial.Cumulative",
    "FrameEvent.ImageWritebackSkips.Cumulative",
    "FrameEvent.FaultAheadPages.Cumulative",
    "FrameEvent.HotPagePromotions.Cumulative",
    "FrameEvent.HotPageDemotions.Cumulative",
    "FrameEvent.HotPageUploads.Cumulative",
    "FrameEvent.HotPageUploadsSkipped.Cumulative",
    "FrameEvent.WrittenUploadLatePages.Cumulative",
    "FrameEvent.ImageCopyViaBufferRounds.Cumulative",
    "FrameEvent.DrawPrepSubmitted.Cumulative",
    "FrameEvent.DrawPrepPublished.Cumulative",
    "FrameEvent.DrawPrepReady.Cumulative",
    "FrameEvent.DrawPrepSelfPrepared.Cumulative",
    "FrameEvent.DrawPrepCommitted.Cumulative",
    "FrameEvent.DrawPrepUnused.Cumulative",
    "FrameEvent.DrawPrepFallbackIneligible.Cumulative",
    "FrameEvent.DrawPrepFallbackUnclean.Cumulative",
    "FrameEvent.DrawPrepFallbackBacking.Cumulative",
    "FrameEvent.DrawPrepFallbackOverflow.Cumulative",
    "FrameEvent.DrawPrepFallbackInconsistent.Cumulative",
    "FrameEvent.DrawPrepFallbackUncertified.Cumulative",
    "FrameEvent.DrawPrepFallbackNotPublished.Cumulative",
    "FrameEvent.DrawPrepFallbackShaderMap.Cumulative",
    "FrameEvent.DrawPrepFallbackCertUnclean.Cumulative",
    "FrameEvent.DrawPrepFallbackCertChanged.Cumulative",
    "FrameEvent.DrawPrepFallbackCoherenceLog.Cumulative",
    "FrameEvent.DrawPrepFallbackMismatch.Cumulative",
    "FrameEvent.DrawPrepCertRanges.Cumulative",
    "FrameEvent.DrawPrepCertBytes.Cumulative",
    "FrameEvent.DrawPrepVerifyChecks.Cumulative",
    "FrameEvent.DrawPrepVerifyMismatches.Cumulative",
    "FrameEvent.DrawPrepFences.Cumulative",
    "FrameEvent.DrawPrepFenceDraws0.Cumulative",
    "FrameEvent.DrawPrepFenceDraws1.Cumulative",
    "FrameEvent.DrawPrepFenceDraws2To3.Cumulative",
    "FrameEvent.DrawPrepFenceDraws4To7.Cumulative",
    "FrameEvent.DrawPrepFenceDraws8To15.Cumulative",
    "FrameEvent.DrawPrepFenceDraws16To31.Cumulative",
    "FrameEvent.DrawPrepFenceDraws32To63.Cumulative",
    "FrameEvent.DrawPrepFenceDraws64Plus.Cumulative",
    "FrameEvent.DrawPrepDrains.Cumulative",
    "FrameEvent.DrawPrepWindowOccupancy.Cumulative",
    "FrameEvent.DrawPrepCommitWaits.Cumulative",
    "FrameEvent.DrawPrepLogWouldReject.Cumulative",
    "FrameEvent.DrawPrepLogMissed.Cumulative",
    "FrameEvent.ComputePipelinesCreated.Cumulative",
    "FrameEvent.ProgramCompileWaits.Cumulative",
    "FrameEvent.ProgramCompileDuplicates.Cumulative",
    "FrameEvent.TranslationReuses.Cumulative",
    "FrameEvent.TranslationVerifyMismatches.Cumulative",
    "FrameEvent.ProgramDiskHits.Cumulative",
    "FrameEvent.ProgramDiskMisses.Cumulative",
    "FrameEvent.ProgramDiskVerifyMismatches.Cumulative",
    "FrameEvent.PipelineLibraryLinks.Cumulative",
    "FrameEvent.PipelineLibraryCacheHits.Cumulative",
    "FrameEvent.PipelineLibrariesCreated.Cumulative",
    "FrameEvent.BdaSyncHotPasses.Cumulative",
    "FrameEvent.BdaSyncHotRanges.Cumulative",
    "FrameEvent.UploadBatchFlushesDeferred.Cumulative",
    "FrameEvent.BdaSyncHotVerifyChecks.Cumulative",
    "FrameEvent.BdaSyncHotVerifyMismatches.Cumulative",
    "FrameEvent.DrawPrepFenceRegIndirect.Cumulative",
    "FrameEvent.DrawPrepFenceEventWrite.Cumulative",
    "FrameEvent.DrawPrepFenceEndOfPipe.Cumulative",
    "FrameEvent.DrawPrepFenceAcquireMem.Cumulative",
    "FrameEvent.DrawPrepFenceWait.Cumulative",
    "FrameEvent.DrawPrepFenceDataWrite.Cumulative",
    "FrameEvent.DrawPrepFenceConstantEngine.Cumulative",
    "FrameEvent.DrawPrepFenceMarker.Cumulative",
    "FrameEvent.DrawPrepFenceDispatch.Cumulative",
    "FrameEvent.DrawPrepFenceIndirectDraw.Cumulative",
    "FrameEvent.DrawPrepFenceContextControl.Cumulative",
    "FrameEvent.DrawPrepFenceOther.Cumulative",
    "FrameEvent.TilerImageUploads.Cumulative",
    "FrameEvent.TilerImageUploadBytes.Cumulative",
    "FrameEvent.TilerImageDownloads.Cumulative",
    "FrameEvent.TilerImageDownloadBytes.Cumulative",
    "FrameEvent.TilerImageVerifyChecks.Cumulative",
    "FrameEvent.TilerImageVerifyMismatches.Cumulative",
    "FrameEvent.DepthFeedbackBarriersAvoided.Cumulative",
    "FrameEvent.BarrierRequestsGuest.Cumulative",
    "FrameEvent.BarrierRequestsShaderAccess.Cumulative",
    "FrameEvent.BarrierRequestsShaderWrite.Cumulative",
    "FrameEvent.BarrierRequestsShaderWriteHazard.Cumulative",
    "FrameEvent.BarrierRequestsIndirectArgs.Cumulative",
    "FrameEvent.BarrierRequestsGds.Cumulative",
    "FrameEvent.BarrierRequestsImage.Cumulative",
    "FrameEvent.BarrierRequestsUpload.Cumulative",
    "FrameEvent.ImageBarrierSameImageFlushes.Cumulative",
    "FrameEvent.ImageBarrierSameImageRenderEnds.Cumulative",
    "FrameEvent.SampledColorAttachmentBindings.Cumulative",
    "FrameEvent.SampledDepthAttachmentBindings.Cumulative",
    "FrameEvent.DepthFeedbackKeepMissWrite.Cumulative",
    "FrameEvent.DepthFeedbackKeepMissInstance.Cumulative",
    "FrameEvent.DepthFeedbackKeepMissSerial.Cumulative",
    "FrameEvent.DepthFeedbackKeepMissState.Cumulative",
    "FrameEvent.ShaderCodeHashCertified.Cumulative",
    "FrameEvent.DrawPrepRegIndirectKept.Cumulative",
    "FrameEvent.TilerImageBlockUploads.Cumulative",
    "FrameEvent.TilerImageBlockUploadBytes.Cumulative",
    "FrameEvent.ReadbackEagerHotPages.Cumulative",
    "FrameEvent.ReadbackEagerCopies.Cumulative",
    "FrameEvent.ReadbackEagerCopyBytes.Cumulative",
    "FrameEvent.ReadbackEagerRetries.Cumulative",
    "FrameEvent.ReadbackEagerWaits.Cumulative",
    "FrameEvent.ReadbackEagerPagesUnmarked.Cumulative",
    "FrameEvent.ReadbackEagerPagesRetained.Cumulative",
    "FrameEvent.ReadbackEagerFlushes.Cumulative",
    "FrameEvent.DrawIndirectFallbackHost.Cumulative",
    "FrameEvent.DrawIndirectFallbackIndexBuffer.Cumulative",
    "FrameEvent.DrawIndirectFallbackTargetOp.Cumulative",
    "FrameEvent.DrawIndirectFallbackQuadList.Cumulative",
    "FrameEvent.DrawIndirectFallbackRestart.Cumulative",
    "FrameEvent.DrawIndirectFallbackMesh.Cumulative",
    "FrameEvent.MeshIndirectDraws.Cumulative",
    "FrameEvent.MeshIndirectAlwaysEmpty.Cumulative",
    "FrameEvent.MeshIndirectDeclinedMulti.Cumulative",
    "FrameEvent.MeshIndirectDeclinedRestart.Cumulative",
    "FrameEvent.MeshIndirectDeclinedClear.Cumulative",
    "FrameEvent.MeshIndirectDeclinedProgram.Cumulative",
    "FrameEvent.MeshIndirectVerifyChecks.Cumulative",
    "FrameEvent.MeshIndirectVerifyMismatches.Cumulative",
    "FrameEvent.MeshIndirectStatus.Cumulative",
    "FrameEvent.DescriptorSetReuseSkippedFresh.Cumulative",
    "FrameEvent.BufferRangeMemoCleanHits.Cumulative",
    "FrameEvent.BufferRangeMemoStreamHits.Cumulative",
    "FrameEvent.BufferRangeMemoRecords.Cumulative",
    "FrameEvent.BufferRangeMemoVerifyChecks.Cumulative",
    "FrameEvent.BufferRangeMemoVerifyMismatches.Cumulative",
    "FrameEvent.HotPageCheckSettles.Cumulative",
    "FrameEvent.DrawPrepCertDigestBytes.Cumulative",
    "FrameEvent.DrawBindingRepeatPrograms.Cumulative",
    "FrameEvent.DrawBindingRepeatTextures.Cumulative",
    "FrameEvent.DrawBindingRepeatResources.Cumulative",
    "FrameEvent.DrawBindingRepeatAll.Cumulative",
    "FrameEvent.TextureBindingMemoRevalidated.Cumulative",
    "FrameEvent.TextureBindingMemoRevalidateMismatches.Cumulative",
    "FrameEvent.BufferDirtyQueriesCombined.Cumulative",
    "FrameEvent.BackingInPlaceHashes.Cumulative",
    "FrameEvent.BackingInPlaceHashesLocked.Cumulative",
    "FrameEvent.DrawPrepValidateInPlaceRanges.Cumulative",
    "FrameEvent.DrawPrepValidateInPlaceLocked.Cumulative",
    "FrameEvent.DrawPrepValidateVerifyChecks.Cumulative",
    "FrameEvent.DrawPrepValidateVerifyMismatches.Cumulative",
    "FrameEvent.DrawPrepValidateVerifyRaces.Cumulative",
    "FrameEvent.DrawPrepHintVerifyChecks.Cumulative",
    "FrameEvent.DrawPrepHintVerifyMismatches.Cumulative",
    "FrameEvent.DepthLayoutTransitionsAvoided.Cumulative",
    "FrameEvent.LodStatsPlainDraws.Cumulative",
    "FrameEvent.LodStatsInstrumentedDraws.Cumulative",
    "FrameEvent.LodStatsCanaryChecks.Cumulative",
    "FrameEvent.LodStatsCanaryMismatches.Cumulative",
    "FrameEvent.UploadDmaCopies.Cumulative",
    "FrameEvent.UploadDmaBytes.Cumulative",
    "FrameEvent.UploadDmaRingFull.Cumulative",
    "FrameEvent.UploadDmaSubmits.Cumulative",
    "FrameEvent.UploadDmaSubmitWaits.Cumulative",
    "FrameEvent.UploadDmaVerifyChecks.Cumulative",
    "FrameEvent.UploadDmaVerifyMismatches.Cumulative",
    "FrameEvent.CmaskFastClears.Cumulative",
    "FrameEvent.CmaskFastClearExpanded.Cumulative",
    "FrameEvent.CmaskFastClearUnproven.Cumulative",
    "FrameEvent.CmaskFastClearAliased.Cumulative",
    "FrameEvent.CmaskFastClearFormat.Cumulative",
    "FrameEvent.CmaskFastClearShape.Cumulative",
    "FrameEvent.CmaskFastClearInspections.Cumulative",
    "FrameEvent.CmaskFastClearInspectionReuses.Cumulative",
    "FrameEvent.CmaskFastClearReadbacks.Cumulative",
    "FrameEvent.DrawPrepLogConflicts.Cumulative",
    "FrameEvent.DrawPrepLogUnknown.Cumulative",
    "FrameEvent.DrawPrepLogOverflows.Cumulative",
    "FrameEvent.DrawPrepLogValueRescues.Cumulative",
    "FrameEvent.BufferRangeMemoVerifyRaces.Cumulative",
    "FrameEvent.TrackerRelaxedQueries.Cumulative",
    "FrameEvent.TrackerRelaxedSyncSkips.Cumulative",
    "FrameEvent.TrackerRelaxedVerifyChecks.Cumulative",
    "FrameEvent.TrackerRelaxedVerifyMismatches.Cumulative",
    "FrameEvent.TrackerRelaxedVerifyRaces.Cumulative",
    "FrameEvent.SyncEpochAdvances.Cumulative",
    "FrameEvent.BdaSyncEpochSkips.Cumulative",
    "FrameEvent.BdaSyncEpochVerifyChecks.Cumulative",
    "FrameEvent.BdaSyncEpochVerifyMismatches.Cumulative",
    "FrameEvent.BdaSyncSubmissionSkips.Cumulative",
    "FrameEvent.BindingEpochMemoStreamHits.Cumulative",
    "FrameEvent.BindingEpochMemoCachedHits.Cumulative",
    "FrameEvent.BindingEpochMemoRecords.Cumulative",
    "FrameEvent.BindingEpochMemoVerifyChecks.Cumulative",
    "FrameEvent.BindingEpochMemoVerifyMismatches.Cumulative",
    "FrameEvent.BindingEpochMemoVerifyRaces.Cumulative",
    "FrameEvent.HostBackingWrites.Cumulative",
    "FrameEvent.HostBackingWriteGpuDirtyPages.Cumulative",
    "FrameEvent.HostBackingWriteCleanPages.Cumulative",
    "FrameEvent.GuestProtectCalls.Cumulative",
    "FrameEvent.GuestProtectWatchedPages.Cumulative",
    "FrameEvent.GuestProtectOverriddenPages.Cumulative",
    "FrameEvent.GuestProtectRestrictsGpuMemory.Cumulative",
    "FrameEvent.ClampRangeMemoMisses.Cumulative",
    "FrameEvent.ClampRangeMemoVerifyMismatches.Cumulative",
    "FrameEvent.DrawSequenceTargetRepeats.Cumulative",
    "FrameEvent.DrawSequenceTargetMisses.Cumulative",
    "FrameEvent.DrawSequenceTargetRecords.Cumulative",
    "FrameEvent.DrawSequenceTextureRepeats.Cumulative",
    "FrameEvent.DrawSequenceTextureMisses.Cumulative",
    "FrameEvent.DrawSequenceViewRepeats.Cumulative",
    "FrameEvent.DrawSequenceVerifyChecks.Cumulative",
    "FrameEvent.DrawSequenceVerifyMismatches.Cumulative",
    "FrameEvent.DrawSequenceVerifyRaces.Cumulative",
    "FrameEvent.DrawPrepUncleanHintGpuDirty.Cumulative",
    "FrameEvent.DrawPrepUncleanHintPublication.Cumulative",
    "FrameEvent.DrawPrepUncleanExact.Cumulative",
    "FrameEvent.DrawPrepUncleanBacking.Cumulative",
    "FrameEvent.DrawSequenceTextureHistoryHits.Cumulative",
    "FrameEvent.EqueueCoalescedTriggers.Cumulative",
    "FrameEvent.EqueueCoalescedDataChanges.Cumulative",
    "FrameEvent.DrawPrepColdWakes.Cumulative",
    "FrameEvent.RenderStateVertexPartialCopies.Cumulative",
    "FrameEvent.RenderStateVertexFullCopies.Cumulative",
    "FrameEvent.RenderStateMinimalResets.Cumulative",
    "FrameEvent.RenderStateMemberSwaps.Cumulative",
    "FrameEvent.RenderStateVerifyChecks.Cumulative",
    "FrameEvent.RenderStateVerifyMismatches.Cumulative",
    "FrameEvent.DescriptorSetAuditCommits.Cumulative",
    "FrameEvent.DescriptorSetAuditRepeats.Cumulative",
    "FrameEvent.DescriptorSetAuditDigestSlotHits.Cumulative",
    "FrameEvent.UploadDmaHostCopies.Cumulative",
    "FrameEvent.UploadDmaHostCopyBytes.Cumulative",
    "FrameEvent.DeferredUnprotectSpans.Cumulative",
    "FrameEvent.DeferredUnprotectCalls.Cumulative",
    "FrameEvent.DeferredUnprotectSettled.Cumulative",
    "FrameEvent.DeferredUnprotectOverflows.Cumulative",
    "FrameEvent.DeferredUnprotectNestedFaults.Cumulative",
    "FrameEvent.TrackerLockParks.Cumulative",
    "FrameEvent.ProtectVerifyChecks.Cumulative",
    "FrameEvent.ProtectVerifyMismatches.Cumulative",
    "FrameEvent.CpRecorderPackets.Cumulative",
    "FrameEvent.CpRecorderBytes.Cumulative",
    "FrameEvent.CpRecorderSubmits.Cumulative",
    "FrameEvent.CpRecorderDrains.Cumulative",
    "FrameEvent.CpRecorderRingFullWaits.Cumulative",
    "FrameEvent.CpRecorderWakes.Cumulative",
    "FrameEvent.CpRecorderParks.Cumulative",
    "FrameEvent.CpRecorderVerifyChecks.Cumulative",
    "FrameEvent.CpRecorderVerifyMismatches.Cumulative",
    "FrameEvent.CpRecorderPlacementSamples.Cumulative",
    "FrameEvent.CpRecorderSameCoreSamples.Cumulative",
    "FrameEvent.CpRecorderIdleDrains.Cumulative",
    "FrameEvent.DrawPrepSteals.Cumulative",
    "FrameEvent.DrawPrepBindingPlans.Cumulative",
    "FrameEvent.DrawPrepBindingPlansUsed.Cumulative",
    "FrameEvent.DrawPrepBindingPlansDropped.Cumulative",
    "FrameEvent.DrawPrepBindingAbstainRanges.Cumulative",
    "FrameEvent.DrawPrepBindingAbstainSamplers.Cumulative",
    "FrameEvent.DrawPrepBindingAbstainPipeline.Cumulative",
    "FrameEvent.DrawPrepBindingAbstainPipelineBusy.Cumulative",
    "FrameEvent.DrawPrepBindingFallbackVm.Cumulative",
    "FrameEvent.DrawPrepBindingFallbackPipeline.Cumulative",
    "FrameEvent.DrawPrepBindingPipelinesUsed.Cumulative",
    "FrameEvent.DrawPrepBindingVerifyChecks.Cumulative",
    "FrameEvent.DrawPrepBindingVerifyMismatches.Cumulative",
    "FrameEvent.DrawPrepBindingHwChecksSkipped.Cumulative",
    "FrameEvent.DrawPrepBindingViewportsUsed.Cumulative",
    "FrameEvent.DrawPrepBindingTextureHints.Cumulative",
    "FrameEvent.DrawPrepBindingTextureRunHits.Cumulative",
    "FrameEvent.DrawPrepBindingViewRunHits.Cumulative",
    "FrameEvent.DescriptorOffsetAuditSets.Cumulative",
    "FrameEvent.DescriptorOffsetAuditSetRepeatsPrevious.Cumulative",
    "FrameEvent.DescriptorOffsetAuditSetRepeatsAny.Cumulative",
    "FrameEvent.DescriptorOffsetAuditPushes.Cumulative",
    "FrameEvent.DescriptorOffsetAuditPushRepeatsPrevious.Cumulative",
    "FrameEvent.DescriptorOffsetAuditPushRepeatsAny.Cumulative",
    "FrameEvent.DescriptorOffsetAuditOverLimit.Cumulative",
    "FrameEvent.EopTimestampsRewritten.Cumulative",
    "FrameEvent.EopTimestampsSkipped.Cumulative",
    "FrameEvent.EopTimestampsUnavailable.Cumulative",
    "FrameEvent.EopTimestampsDeferred.Cumulative",
    "FrameEvent.EopTimestampsVerifyChecks.Cumulative",
    "FrameEvent.EopTimestampsVerifyMismatches.Cumulative",
    "FrameEvent.BdaSyncLogPasses.Cumulative",
    "FrameEvent.BdaSyncLogRanges.Cumulative",
    "FrameEvent.BdaSyncLogOverflows.Cumulative",
    "FrameEvent.BdaSyncLogVerifyChecks.Cumulative",
    "FrameEvent.BdaSyncLogVerifyMismatches.Cumulative",
    "FrameEvent.HostBackingWritesTracked.Cumulative",
    "FrameEvent.BindingEpochMemoCrossHits.Cumulative",
    "FrameEvent.BindingEpochMemoCrossRejects.Cumulative",
    "FrameEvent.BindingEpochMemoMissSlot.Cumulative",
    "FrameEvent.BindingEpochMemoMissSignature.Cumulative",
    "FrameEvent.BindingEpochMemoMissGuard.Cumulative",
    "FrameEvent.BindingEpochMemoMissEpoch.Cumulative",
    "FrameEvent.WrittenSyncSkips.Cumulative",
    "FrameEvent.WrittenSyncSkipVerifyChecks.Cumulative",
    "FrameEvent.WrittenSyncSkipVerifyMismatches.Cumulative",
    "FrameEvent.PendingOpsDeferred.Cumulative",
    "FrameEvent.PendingOpsDeferredDepth.Cumulative",
    "FrameEvent.PriorityWaitSpins.Cumulative",
    "FrameEvent.PriorityWaitSpinHits.Cumulative",
    "FrameEvent.CpuPlacementCpSamples.Cumulative",
    "FrameEvent.CpuPlacementCpOffCore.Cumulative",
    "FrameEvent.CpuPlacementGuestSamples.Cumulative",
    "FrameEvent.CpuPlacementGuestOnCpCore.Cumulative",
    "FrameEvent.CpuPlacementHostSamples.Cumulative",
    "FrameEvent.CpuPlacementHostOnCpCore.Cumulative",
    "FrameEvent.CpuPlacementRecorderOffCore.Cumulative",
    "FrameEvent.CpuPlacementHardAffinity.Cumulative",
    "FrameEvent.CpuPlacementHardAffinityReserved.Cumulative",
    "FrameEvent.CpuPlacementRepinned.Cumulative",
    "FrameEvent.FalseSharingWrites.Cumulative",
    "FrameEvent.FalseSharingBytes.Cumulative",
    "FrameEvent.FalseSharingUploadSplits.Cumulative",
    "FrameEvent.FalseSharingVerifyChecks.Cumulative",
    "FrameEvent.FalseSharingVerifyConflicts.Cumulative",
    "FrameEvent.GpuWriteImageSkips.Cumulative",
    "FrameEvent.GpuWriteImageSkipVerifyChecks.Cumulative",
    "FrameEvent.GpuWriteImageSkipVerifyRaces.Cumulative",
    "FrameEvent.GpuWriteImageSkipVerifyMismatches.Cumulative",
    "FrameEvent.DccImageStateUnregistered.Cumulative",
    "FrameEvent.DccImageStateStencil.Cumulative",
    "FrameEvent.DccImageStateMismatch.Cumulative",
    "FrameEvent.DccImageStateNotGpuModified.Cumulative",
    "FrameEvent.DccImageStateBufferModified.Cumulative",
    "FrameEvent.DccImageStateCpuDirty.Cumulative",
    "FrameEvent.DccImageStatePartial.Cumulative",
    "FrameEvent.DccImageStateGpuDirtyBytes.Cumulative",
    "FrameEvent.ReadbackSideOtherOwner.Cumulative",
    "FrameEvent.ReadbackSideOtherAlignment.Cumulative",
    "FrameEvent.ReadbackSideOtherWindow.Cumulative",
    "FrameEvent.ReadbackSideOtherPublication.Cumulative",
    "FrameEvent.ReadbackSideOtherNoDirty.Cumulative",
    "FrameEvent.ReadbackSideOtherSlot.Cumulative",
    "FrameEvent.DccGpuRefreshes.Cumulative",
    "FrameEvent.DccGpuRefreshVerifyChecks.Cumulative",
    "FrameEvent.DccGpuRefreshVerifyMismatches.Cumulative",
    "FrameEvent.DrawPrepCertRangesVerifyChecks.Cumulative",
    "FrameEvent.DrawPrepCertRangesVerifyMismatches.Cumulative",
    "FrameEvent.ImageLruTouchSkips.Cumulative",
    "FrameEvent.ImageLruVerifyChecks.Cumulative",
    "FrameEvent.ImageLruVerifyMismatches.Cumulative",
    "FrameEvent.ImageTransitSkips.Cumulative",
    "FrameEvent.ImageTransitVerifyChecks.Cumulative",
    "FrameEvent.ImageTransitVerifyMismatches.Cumulative",
    "FrameEvent.TargetRecordNotFirstPage.Cumulative",
    "FrameEvent.TargetRecordChanged.Cumulative",
    "FrameEvent.TargetRecordDccClear.Cumulative",
    "FrameEvent.TargetRecordDccNative.Cumulative",
    "FrameEvent.TargetRecordDccFallback.Cumulative",
    "FrameEvent.TargetRecordDccGuest.Cumulative",
    "FrameEvent.TargetRecordDccPages.Cumulative",
    "FrameEvent.TargetRecordCmaskNative.Cumulative",
    "FrameEvent.TargetRecordCmaskOther.Cumulative",
    "FrameEvent.CpSeqOps.Cumulative",
    "FrameEvent.CpSeqVerifyChecks.Cumulative",
    "FrameEvent.CpSeqVerifyMismatches.Cumulative",
    "FrameEvent.CpSeqVerifyReadDivergences.Cumulative",
    "FrameEvent.CpSeqBarriers.Cumulative",
    "FrameEvent.CpSeqLockstepReads.Cumulative",
    "FrameEvent.CpSeqLockstepBuffers.Cumulative",
    "FrameEvent.CpSeqPendingWriteStops.Cumulative",
    "FrameEvent.CpSeqSnapshots.Cumulative",
    "FrameEvent.CpSeqWindowFull.Cumulative",
    "FrameEvent.CpSeqFrontWaits.Cumulative",
    "FrameEvent.CpSeqResolverStarved.Cumulative",
    "FrameEvent.CpSeqDirectReads.Cumulative",
    "FrameEvent.DrawPrepLogChecks.Cumulative",
    "FrameEvent.DrawPrepLogEntries.Cumulative",
    "FrameEvent.CpSeqBarrierWaitSelfLabel.Cumulative",
    "FrameEvent.CpSeqBarrierWaitOther.Cumulative",
    "FrameEvent.CpSeqBarrierFlipWait.Cumulative",
    "FrameEvent.CpSeqBarrierCondition.Cumulative",
    "FrameEvent.CpSeqBarrierDrawBursts.Cumulative",
    "FrameEvent.DrawPrepCommitWaitsBarrier.Cumulative",
    "FrameEvent.DrawPrepCommitWaitsShallow.Cumulative",
    "FrameEvent.DrawPrepCommitWaitsDeep.Cumulative",
    "FrameEvent.DrawPrepCommitWaitsUnclaimed.Cumulative",
    "FrameEvent.DrawPrepCommitWaitsStart.Cumulative",
    "FrameEvent.CpSeqPrefetchRuns.Cumulative",
    "FrameEvent.CpSeqPrefetchDraws.Cumulative",
    "FrameEvent.CpSeqPrefetchAdopted.Cumulative",
    "FrameEvent.CpSeqPrefetchMismatches.Cumulative",
    "FrameEvent.CpSeqPrefetchSkipped.Cumulative",
    "FrameEvent.CpSeqPrefetchStopLockstep.Cumulative",
    "FrameEvent.CpSeqPrefetchStopRead.Cumulative",
    "FrameEvent.CpSeqPrefetchStopWindow.Cumulative",
    "FrameEvent.CpSeqPrefetchStopEnd.Cumulative",
    "FrameEvent.CpSeqPrefetchAdoptedWaits.Cumulative",
    "FrameEvent.SubmitIntervalDeferrals.Cumulative",
    "FrameEvent.CpCommitDccGuestRecords.Cumulative",
    "FrameEvent.CpCommitDccGuestRejects.Cumulative",
    "FrameEvent.CpCommitTexDccRecords.Cumulative",
    "FrameEvent.CpCommitTexDccRejects.Cumulative",
    "FrameEvent.BdaSettles.Cumulative",
    "FrameEvent.BdaSettlePages.Cumulative",
    "FrameEvent.BdaDroppedWrites.Cumulative",
    "FrameEvent.BdaAliasHits.Cumulative",
    "FrameEvent.BdaAliasedImages.Cumulative",
    "FrameEvent.BdaSettleCpuDirtyPages.Cumulative",
    "FrameEvent.VariantPlanSkips.Cumulative",
    "FrameEvent.BdaLoggedStores.Cumulative",
};
static_assert(kFrameEventNames.back() != nullptr, "FrameEvent names must match the enum");

constexpr size_t kFrameWaitCount = static_cast<size_t>(Profiler::FrameWait::Count);
struct FrameWaitTotals {
	std::atomic<uint64_t> calls {0};
	std::atomic<uint64_t> nanoseconds {0};
};
std::array<FrameWaitTotals, kFrameWaitCount> g_frame_waits {};
constexpr std::array<const char*, kFrameWaitCount> kFrameWaitCallNames {
    "FrameWait.ReadMemory.Calls.Cumulative",
    "FrameWait.ShaderReadiness.Calls.Cumulative",
    "FrameWait.DccFallback.Calls.Cumulative",
    "FrameWait.ResourceMaterialization.Calls.Cumulative",
    "FrameWait.DriverSubmit.Calls.Cumulative",
    "FrameWait.ShaderProgramMiss.Calls.Cumulative",
    "FrameWait.GraphicsPipelineCreate.Calls.Cumulative",
    "FrameWait.ResourceReuseValidation.Calls.Cumulative",
    "FrameWait.NativeImageCreate.Calls.Cumulative",
    "FrameWait.NativeImageDestroy.Calls.Cumulative",
    "FrameWait.GpuBusy.Calls.Cumulative",
    "FrameWait.GpuIdle.Calls.Cumulative",
    "FrameWait.GpuStarved.Calls.Cumulative",
    "FrameWait.GpuRecordToStart.Calls.Cumulative",
    "FrameWait.GpuDispatchToStart.Calls.Cumulative",
    "FrameWait.GpuEndToObserved.Calls.Cumulative",
    "FrameWait.ReadbackSideWait.Calls.Cumulative",
    "FrameWait.StagePrepJoin.Calls.Cumulative",
    "FrameWait.StagePrepHelper.Calls.Cumulative",
    "FrameWait.TextureUpload.Calls.Cumulative",
    "FrameWait.TextureStagingCopy.Calls.Cumulative",
    "FrameWait.GpuWaitDrain.Calls.Cumulative",
    "FrameWait.GpuWaitOcclusion.Calls.Cumulative",
    "FrameWait.GpuWaitStreamWrap.Calls.Cumulative",
    "FrameWait.GpuWaitLodStats.Calls.Cumulative",
    "FrameWait.GpuWaitUnmap.Calls.Cumulative",
    "FrameWait.GpuWaitPredication.Calls.Cumulative",
    "FrameWait.GpuWaitGds.Calls.Cumulative",
    "FrameWait.GpuWaitSideCopy.Calls.Cumulative",
    "FrameWait.GpuWaitOther.Calls.Cumulative",
    "FrameWait.AgcDoneWait.Calls.Cumulative",
    "FrameWait.DrawPrepCommitWait.Calls.Cumulative",
    "FrameWait.DrawPrepPrepare.Calls.Cumulative",
    "FrameWait.DrawPrepValidate.Calls.Cumulative",
    "FrameWait.ShaderTranslate.Calls.Cumulative",
    "FrameWait.ShaderEmit.Calls.Cumulative",
    "FrameWait.ShaderValidate.Calls.Cumulative",
    "FrameWait.ShaderModuleCreate.Calls.Cumulative",
    "FrameWait.ShaderDiskLoad.Calls.Cumulative",
    "FrameWait.GraphicsPipelineDriver.Calls.Cumulative",
    "FrameWait.ComputePipelineCreate.Calls.Cumulative",
    "FrameWait.PipelineOptimize.Calls.Cumulative",
    "FrameWait.CpRecorderDrain.Calls.Cumulative",
    "FrameWait.CpRecorderRingFull.Calls.Cumulative",
    "FrameWait.CpRecorderExecute.Calls.Cumulative",
    "FrameWait.DrawPrepSteal.Calls.Cumulative",
    "FrameWait.DrawPrepBindingPlan.Calls.Cumulative",
    "FrameWait.DrawPrepWorker1.Calls.Cumulative",
    "FrameWait.DrawPrepWorker2.Calls.Cumulative",
    "FrameWait.DrawPrepWorker3.Calls.Cumulative",
    "FrameWait.DrawPrepWorker4.Calls.Cumulative",
    "FrameWait.DrawPrepWorker5.Calls.Cumulative",
    "FrameWait.DrawPrepWorker6.Calls.Cumulative",
    "FrameWait.DrawPrepWorker7.Calls.Cumulative",
    "FrameWait.DrawPrepWorker8.Calls.Cumulative",
    "FrameWait.EopTimestampPublish.Calls.Cumulative",
    "FrameWait.CpSeqSequencerWait.Calls.Cumulative",
    "FrameWait.CpSeqResolverStarved.Calls.Cumulative",
    "FrameWait.DrawPrepCommit.Calls.Cumulative",
    "FrameWait.DrawPrepCommitWaitBarrier.Calls.Cumulative",
    "FrameWait.DrawPrepCommitWaitShallow.Calls.Cumulative",
    "FrameWait.DrawPrepCommitWaitDeep.Calls.Cumulative",
    "FrameWait.DrawPrepCommitWaitStart.Calls.Cumulative",
    "FrameWait.CpSeqPrefetch.Calls.Cumulative",
    "FrameWait.SubmitDependencyWait.Calls.Cumulative",
    "FrameWait.BdaSettle.Calls.Cumulative",
};
static_assert(kFrameWaitCallNames.back() != nullptr, "FrameWait names must match the enum");
constexpr std::array<const char*, kFrameWaitCount> kFrameWaitTimeNames {
    "FrameWait.ReadMemory.Nanoseconds.Cumulative",
    "FrameWait.ShaderReadiness.Nanoseconds.Cumulative",
    "FrameWait.DccFallback.Nanoseconds.Cumulative",
    "FrameWait.ResourceMaterialization.Nanoseconds.Cumulative",
    "FrameWait.DriverSubmit.Nanoseconds.Cumulative",
    "FrameWait.ShaderProgramMiss.Nanoseconds.Cumulative",
    "FrameWait.GraphicsPipelineCreate.Nanoseconds.Cumulative",
    "FrameWait.ResourceReuseValidation.Nanoseconds.Cumulative",
    "FrameWait.NativeImageCreate.Nanoseconds.Cumulative",
    "FrameWait.NativeImageDestroy.Nanoseconds.Cumulative",
    "FrameWait.GpuBusy.Nanoseconds.Cumulative",
    "FrameWait.GpuIdle.Nanoseconds.Cumulative",
    "FrameWait.GpuStarved.Nanoseconds.Cumulative",
    "FrameWait.GpuRecordToStart.Nanoseconds.Cumulative",
    "FrameWait.GpuDispatchToStart.Nanoseconds.Cumulative",
    "FrameWait.GpuEndToObserved.Nanoseconds.Cumulative",
    "FrameWait.ReadbackSideWait.Nanoseconds.Cumulative",
    "FrameWait.StagePrepJoin.Nanoseconds.Cumulative",
    "FrameWait.StagePrepHelper.Nanoseconds.Cumulative",
    "FrameWait.TextureUpload.Nanoseconds.Cumulative",
    "FrameWait.TextureStagingCopy.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitDrain.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitOcclusion.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitStreamWrap.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitLodStats.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitUnmap.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitPredication.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitGds.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitSideCopy.Nanoseconds.Cumulative",
    "FrameWait.GpuWaitOther.Nanoseconds.Cumulative",
    "FrameWait.AgcDoneWait.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommitWait.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepPrepare.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepValidate.Nanoseconds.Cumulative",
    "FrameWait.ShaderTranslate.Nanoseconds.Cumulative",
    "FrameWait.ShaderEmit.Nanoseconds.Cumulative",
    "FrameWait.ShaderValidate.Nanoseconds.Cumulative",
    "FrameWait.ShaderModuleCreate.Nanoseconds.Cumulative",
    "FrameWait.ShaderDiskLoad.Nanoseconds.Cumulative",
    "FrameWait.GraphicsPipelineDriver.Nanoseconds.Cumulative",
    "FrameWait.ComputePipelineCreate.Nanoseconds.Cumulative",
    "FrameWait.PipelineOptimize.Nanoseconds.Cumulative",
    "FrameWait.CpRecorderDrain.Nanoseconds.Cumulative",
    "FrameWait.CpRecorderRingFull.Nanoseconds.Cumulative",
    "FrameWait.CpRecorderExecute.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepSteal.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepBindingPlan.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker1.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker2.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker3.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker4.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker5.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker6.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker7.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepWorker8.Nanoseconds.Cumulative",
    "FrameWait.EopTimestampPublish.Nanoseconds.Cumulative",
    "FrameWait.CpSeqSequencerWait.Nanoseconds.Cumulative",
    "FrameWait.CpSeqResolverStarved.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommit.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommitWaitBarrier.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommitWaitShallow.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommitWaitDeep.Nanoseconds.Cumulative",
    "FrameWait.DrawPrepCommitWaitStart.Nanoseconds.Cumulative",
    "FrameWait.CpSeqPrefetch.Nanoseconds.Cumulative",
    "FrameWait.SubmitDependencyWait.Nanoseconds.Cumulative",
    "FrameWait.BdaSettle.Nanoseconds.Cumulative",
};
static_assert(kFrameWaitTimeNames.back() != nullptr, "FrameWait names must match the enum");

uint64_t FrameWaitClockNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	    std::chrono::steady_clock::now().time_since_epoch()).count());
}

constexpr size_t kLoadingEventCount = static_cast<size_t>(Profiler::LoadingEvent::Count);
std::array<std::atomic<uint64_t>, kLoadingEventCount> g_loading_events {};
constexpr std::array<const char*, kLoadingEventCount> kLoadingEventNames {
    "Loading.AprRecordsCompleted.Cumulative",
    "Loading.AprRequestedBytes.Cumulative",
    "Loading.AprHostReadBytes.Cumulative",
    "Loading.AprGuestCopiedBytes.Cumulative",
    "Loading.AprErrors.Cumulative",
    "Loading.AprShortReads.Cumulative",
    "Loading.AmmErrors.Cumulative",
    "Loading.PreadRequestedBytes.Cumulative",
    "Loading.PreadReadBytes.Cumulative",
    "Loading.PreadErrors.Cumulative",
    "Loading.LodStatsPackets.Cumulative",
    "Loading.LodStatsBufferBytes.Cumulative",
    "Loading.LodStatsReportAndResetPackets.Cumulative",
    "Loading.LodStatsForceResetPackets.Cumulative",
    "Loading.LodStatsResetCountSum.Cumulative",
    "Loading.LodStatsInterval100kSum.Cumulative",
    "Loading.LodStatsCachePolicySum.Cumulative",
    "Loading.DirectMemorySizeQueries.Cumulative",
    "Loading.DirectMemoryAvailableQueries.Cumulative",
    "Loading.DirectMemoryAvailableSuccesses.Cumulative",
    "Loading.DirectMemoryAvailableBytesSum.Cumulative",
    "Loading.PageTableStatsQueries.Cumulative",
    "Loading.PageTableStatsSuccesses.Cumulative",
    "Loading.PageTableCpuAvailableSum.Cumulative",
    "Loading.PageTableGpuAvailableSum.Cumulative",
    "Loading.PageTableCpuZeroAvailable.Cumulative",
    "Loading.PageTableGpuZeroAvailable.Cumulative",
    "Loading.FlexibleAvailableQueries.Cumulative",
    "Loading.FlexibleAvailableSuccesses.Cumulative",
    "Loading.FlexibleAvailableBytesSum.Cumulative",
    "Loading.MemoryPoolStatsQueries.Cumulative",
    "Loading.MemoryPoolAvailableBytesSum.Cumulative",
    "Loading.AmmUsageStatsQueries.Cumulative",
    "Loading.AmmOccupancyThresholdCalls.Cumulative",
    "Loading.AmmOccupancyThresholdSum.Cumulative",
    "Loading.AprReadCommandsAppended.Cumulative",
    "Loading.AprCommandResets.Cumulative",
    "Loading.AprCommandBufferSets.Cumulative",
    "Loading.AprCommandBufferClears.Cumulative",
    "Loading.AprSubmitAndGetResult.Cumulative",
    "Loading.AprSubmitPlain.Cumulative",
    "Loading.AprSubmitAndGetId.Cumulative",
    "Loading.AprWaitCompleted.Cumulative",
    "Loading.AprWaitUnknownId.Cumulative",
    "Loading.AprUnsupportedWaitCommands.Cumulative",
    "Loading.AprUnsupportedCounterCommands.Cumulative",
    "Loading.AprDiagnosticRowsDropped.Cumulative",
};
struct LoadingTotals {
	std::atomic<uint64_t> started {0};
	std::atomic<uint64_t> completed {0};
	std::atomic<uint64_t> in_flight {0};
	std::atomic<uint64_t> nanoseconds {0};
};
struct LoadingPlotNames {
	const char* started;
	const char* completed;
	const char* in_flight;
	const char* nanoseconds;
};
constexpr size_t kLoadingOperationCount = static_cast<size_t>(Profiler::LoadingOperation::Count);
std::array<LoadingTotals, kLoadingOperationCount> g_loading_operations {};
#define KYTY_LOADING_PLOT_NAMES(name) \
    LoadingPlotNames {"Loading." #name ".Started.Cumulative", \
                      "Loading." #name ".Completed.Cumulative", \
                      "Loading." #name ".InFlight", \
                      "Loading." #name ".Nanoseconds.Cumulative"}
constexpr std::array<LoadingPlotNames, kLoadingOperationCount> kLoadingOperationNames {
    KYTY_LOADING_PLOT_NAMES(AprSubmit),
    KYTY_LOADING_PLOT_NAMES(AprRead),
    KYTY_LOADING_PLOT_NAMES(AprHostRead),
    KYTY_LOADING_PLOT_NAMES(AprGuestCopy),
    KYTY_LOADING_PLOT_NAMES(AmmMap),
    KYTY_LOADING_PLOT_NAMES(AmmUnmap),
    KYTY_LOADING_PLOT_NAMES(Pread),
    KYTY_LOADING_PLOT_NAMES(PreadFileMutex),
    KYTY_LOADING_PLOT_NAMES(PreadCoherence),
    KYTY_LOADING_PLOT_NAMES(PreadNativeRead),
};
#undef KYTY_LOADING_PLOT_NAMES
std::mutex g_loading_publish_mutex;
std::condition_variable_any g_loading_publish_condition;

constexpr size_t kLoadingAprQueueSize = 1024;
constexpr uint64_t kLoadingAprMaxRows = 8192;
std::mutex g_loading_apr_mutex;
std::array<Profiler::LoadingAprSubmission, kLoadingAprQueueSize> g_loading_apr_queue {};
size_t g_loading_apr_head = 0, g_loading_apr_size = 0;
std::atomic<uint64_t> g_loading_apr_sequence {0}, g_loading_apr_retained {0};
constexpr const char* kLoadingAprHeader =
    "sequence,timestamp_ns,api_kind,command_buffer,argument1,argument2,result_address,out_id_address,"
    "submission_id,resolved_object,generation,append_serial,submits_in_generation,unchanged_submits,"
    "cached_buffer,cached_size,cached_offset,header_valid,header_buffer,header_size,header_offset,"
    "header_commands,header_type,header_mismatch,read_count,event_count,write_count,map_count,"
    "commands_signature,first_read_id,first_read_destination,first_read_size,first_read_offset,"
    "read_bytes,execution_observed,execution_result,error_offset,submit_result,api_kernel_result\n";
// Destroy the publisher before its queue, condition variable and mutexes.
std::jthread g_loading_publisher;

struct LoadingLogCloser {
	void operator()(std::FILE* file) const { std::fclose(file); }
};
using LoadingLog = std::unique_ptr<std::FILE, LoadingLogCloser>;

LoadingLog OpenLoadingLog(bool apr = false) {
	const auto* setting = std::getenv("KYTY_LOADING_LOG");
	if (setting == nullptr || *setting == '\0') return {};
	try {
		auto path = std::filesystem::u8path(setting);
		if (!path.is_absolute()) {
			::printf("Loading diagnostics CSV requires a new absolute path; log disabled\n");
			return {};
		}
		// A launcher can start more than one emulator with the same inherited
		// requested path. Give each process its own destination without replacing
		// an earlier capture; the timestamp also distinguishes reused process IDs.
		static const auto startup_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		    std::chrono::system_clock::now().time_since_epoch()).count();
#ifdef _WIN32
		const auto process_id = _getpid();
#else
		const auto process_id = ::getpid();
#endif
		auto filename = path.stem();
		if (apr) filename += ".apr-submissions";
		filename += ".pid-" + std::to_string(process_id) + ".start-" + std::to_string(startup_ns);
		filename += path.extension();
		path = path.parent_path() / filename;
		::printf("Loading %s CSV path: %s\n", apr ? "APR submissions" : "diagnostics",
		         Common::PathToString(path).c_str());
		std::FILE* handle = nullptr;
#ifdef _WIN32
		// Exclusive creation preserves earlier captures; allow readers while this
		// publisher writes, so loading progress can be inspected during a stall.
		int descriptor = -1;
		if (_wsopen_s(&descriptor, path.c_str(), _O_CREAT | _O_EXCL | _O_WRONLY | _O_TEXT,
		              _SH_DENYWR, _S_IREAD | _S_IWRITE) == 0) {
			handle = _wfdopen(descriptor, L"w");
			if (handle == nullptr) _close(descriptor);
		}
#else
		handle = std::fopen(path.c_str(), "wx");
#endif
		LoadingLog log(handle);
		if (!log) {
			::printf("Loading diagnostics CSV could not be opened; log disabled\n");
			return {};
		}
		if (apr) {
			std::fputs(kLoadingAprHeader, log.get());
			std::fflush(log.get());
			return log;
		}
		std::fprintf(log.get(), "elapsed_ns");
		for (const auto* name: kLoadingEventNames) std::fprintf(log.get(), ",%s", name);
		for (const auto& names: kLoadingOperationNames) {
			std::fprintf(log.get(), ",%s,%s,%s,%s", names.started, names.completed,
			             names.in_flight, names.nanoseconds);
		}
		std::fprintf(log.get(), ",Loading.CompletedFlips.Cumulative\n");
		std::fflush(log.get());
		return log;
	} catch (const std::exception&) {
		::printf("Loading diagnostics CSV path could not be used; log disabled\n");
		return {};
	}
}

void PublishLoadingAprSubmissions(std::FILE* log) {
	// Copy a bounded batch under the diagnostic mutex, then release it before I/O.
	// The guest path never takes the publisher/file lock and never writes a file.
	std::array<Profiler::LoadingAprSubmission, 128> batch {};
	for (;;) {
		size_t count = 0;
		{
			std::scoped_lock lock(g_loading_apr_mutex);
			count = std::min(batch.size(), g_loading_apr_size);
			for (size_t i = 0; i < count; ++i) {
				batch[i] = g_loading_apr_queue[g_loading_apr_head];
				g_loading_apr_head = (g_loading_apr_head + 1) % kLoadingAprQueueSize;
			}
			g_loading_apr_size -= count;
		}
		if (count == 0) break;
		for (size_t i = 0; i < count; ++i) {
			if (log == nullptr || std::ferror(log) != 0) {
				Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprDiagnosticRowsDropped);
				continue;
			}
			const auto& r = batch[i];
			const std::array values {
			    r.sequence, r.timestamp_ns, r.api_kind, r.command_buffer, r.argument1, r.argument2,
			    r.result_address, r.out_id_address, r.submission_id, r.resolved_object, r.generation,
			    r.append_serial, r.submits_in_generation, r.unchanged_submits, r.cached_buffer,
			    r.cached_size, r.cached_offset, r.header_valid, r.header_buffer, r.header_size,
			    r.header_offset, r.header_commands, r.header_type, r.header_mismatch, r.read_count,
			    r.event_count, r.write_count, r.map_count, r.commands_signature, r.first_read_id,
			    r.first_read_destination, r.first_read_size, r.first_read_offset, r.read_bytes,
			    r.execution_observed, r.execution_result, r.error_offset, r.submit_result, r.api_kernel_result};
			for (size_t column = 0; column < values.size(); ++column) {
				std::fprintf(log, column == 0 ? "%" PRIu64 : ",%" PRIu64, values[column]);
			}
			std::fputc('\n', log);
			if (std::ferror(log) != 0)
				Profiler::CountLoadingEvent(Profiler::LoadingEvent::AprDiagnosticRowsDropped);
		}
	}
	if (log != nullptr) std::fflush(log);
}

void PublishLoadingProgress(uint64_t elapsed_ns, std::FILE* log) {
	const bool plot = tracy::ProfilerAvailable() && TracyIsConnected;
	if (!plot && log == nullptr) return;
	if (log != nullptr) std::fprintf(log, "%" PRIu64, elapsed_ns);
	for (size_t i = 0; i < kLoadingEventCount; ++i) {
		const auto value = g_loading_events[i].load(std::memory_order_relaxed);
		if (plot) TracyPlot(kLoadingEventNames[i], static_cast<int64_t>(value));
		if (log != nullptr) std::fprintf(log, ",%" PRIu64, value);
	}
	for (size_t i = 0; i < kLoadingOperationCount; ++i) {
		const auto& totals = g_loading_operations[i];
		const auto& names = kLoadingOperationNames[i];
		const auto started = totals.started.load(std::memory_order_relaxed);
		const auto completed = totals.completed.load(std::memory_order_relaxed);
		const auto in_flight = totals.in_flight.load(std::memory_order_relaxed);
		const auto nanoseconds = totals.nanoseconds.load(std::memory_order_relaxed);
		if (plot) {
			TracyPlot(names.started, static_cast<int64_t>(started));
			TracyPlot(names.completed, static_cast<int64_t>(completed));
			TracyPlot(names.in_flight, static_cast<int64_t>(in_flight));
			TracyPlot(names.nanoseconds, static_cast<int64_t>(nanoseconds));
		}
		if (log != nullptr) {
			std::fprintf(log, ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
			             started, completed, in_flight, nanoseconds);
		}
	}
	// Independently atomic fields can straddle operations. The heartbeat terminates a
	// publication group; it does not turn that group into a transactional snapshot.
	const auto flips = g_completed_flips.load(std::memory_order_relaxed);
	if (plot) {
		TracyPlot("Loading.CompletedFlips.Cumulative", static_cast<int64_t>(flips));
		TracyPlot("Loading.HeartbeatMilliseconds", static_cast<int64_t>(elapsed_ns / 1000000));
	}
	if (log != nullptr) {
		std::fprintf(log, ",%" PRIu64 "\n", flips);
		std::fflush(log);
	}
}

void RemoveBlock(Profiler::ScopedBlock* block) {
	auto block_it = std::find(g_block_stack.rbegin(), g_block_stack.rend(), block);
	if (block_it != g_block_stack.rend()) {
		g_block_stack.erase(std::next(block_it).base());
	}
}

} // namespace

namespace Profiler {

namespace Detail {

std::atomic<int8_t>      g_frames_only {-1};
std::atomic<int8_t>      g_aggregate {-1};
std::atomic<int8_t>      g_detailed {-1};
std::atomic<int8_t>      g_shared_counters {-1};
std::atomic<bool>        g_zones {false};
std::atomic<CounterSink> g_event_sink {CounterSink::Off};
constinit thread_local ThreadCounters* t_counters = nullptr;

namespace {

bool EnvEquals(const char* name, const char* expected) {
	const auto* setting = std::getenv(name);
	return setting != nullptr && std::strcmp(setting, expected) == 0;
}

// Every thread's counter block (push-only list; blocks are never freed).
std::atomic<ThreadCounters*> g_counter_blocks {nullptr};

// Returns the thread's block at thread exit: a later thread continues its totals.
struct ThreadCountersRelease {
	ThreadCountersRelease()                                        = default;
	ThreadCountersRelease(const ThreadCountersRelease&)            = delete;
	ThreadCountersRelease& operator=(const ThreadCountersRelease&) = delete;
	~ThreadCountersRelease() {
		if (auto* block = t_counters; block != nullptr) {
			t_counters = nullptr;
			block->in_use.store(false, std::memory_order_release);
		}
	}
};

// The next guest flip's counting: see CounterSink.
CounterSink CurrentEventSink() {
	if (SharedCounters()) {
		return CounterSink::Shared;
	}
	return AggregateEnabled() && tracy::ProfilerAvailable() && TracyIsConnected ? CounterSink::Thread
	                                                                          : CounterSink::Off;
}

} // namespace

struct Totals {
	std::array<uint64_t, kFrameEventCount> events {};
	std::array<uint64_t, kFrameWorkCount>  work {};
	std::array<uint64_t, kFrameWaitCount>  wait_calls {};
	std::array<uint64_t, kFrameWaitCount>  wait_ns {};
};

// Every thread's block summed (block by block, each read in order); `aggregates`: also the event
// and wait counters.
void SumThreadCounters(Totals& totals, bool aggregates) {
	for (auto* block = g_counter_blocks.load(std::memory_order_acquire); block != nullptr;
	     block       = block->next) {
		for (size_t i = 0; i < kFrameWorkCount; ++i) {
			totals.work[i] += block->work[i].load(std::memory_order_relaxed);
		}
		if (!aggregates) {
			continue;
		}
		for (size_t i = 0; i < kFrameEventCount; ++i) {
			totals.events[i] += block->events[i].load(std::memory_order_relaxed);
		}
		for (size_t i = 0; i < kFrameWaitCount; ++i) {
			totals.wait_calls[i] += block->wait_calls[i].load(std::memory_order_relaxed);
			totals.wait_ns[i] += block->wait_ns[i].load(std::memory_order_relaxed);
		}
	}
}

bool ReadFramesOnly() noexcept {
	const bool enabled = EnvEquals("KYTY_PROFILE_FRAMES_ONLY", "1");
	g_frames_only.store(enabled ? 1 : 0, std::memory_order_relaxed);
	return enabled;
}

bool ReadAggregate() noexcept {
	const bool enabled = FramesOnlyEnabled() && EnvEquals("KYTY_PROFILE_AGGREGATES", "1");
	g_aggregate.store(enabled ? 1 : 0, std::memory_order_relaxed);
	return enabled;
}

bool ReadDetailed() noexcept {
	const bool enabled = !FramesOnlyEnabled() && EnvEquals("KYTY_PROFILE_DETAILS", "1");
	g_detailed.store(enabled ? 1 : 0, std::memory_order_relaxed);
	return enabled;
}

bool ReadSharedCounters() noexcept {
	const bool enabled = EnvEquals("KYTY_PROFILE_COUNTERS", "shared");
	g_shared_counters.store(enabled ? 1 : 0, std::memory_order_relaxed);
	return enabled;
}

size_t CounterBlockCount() noexcept {
	size_t count = 0;
	for (auto* block = g_counter_blocks.load(std::memory_order_acquire); block != nullptr;
	     block       = block->next) {
		count++;
	}
	return count;
}

ThreadCounters& AcquireThreadCounters() noexcept {
	thread_local ThreadCountersRelease release;
	(void)release;
	ThreadCounters* acquired = nullptr;
	for (auto* block = g_counter_blocks.load(std::memory_order_acquire); block != nullptr;
	     block       = block->next) {
		bool expected = false;
		if (!block->in_use.load(std::memory_order_relaxed) &&
		    block->in_use.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
			acquired = block;
			break;
		}
	}
	if (acquired == nullptr) {
		acquired = new ThreadCounters();
		acquired->in_use.store(true, std::memory_order_relaxed);
		auto* head = g_counter_blocks.load(std::memory_order_relaxed);
		do {
			acquired->next = head;
		} while (!g_counter_blocks.compare_exchange_weak(head, acquired, std::memory_order_release,
		                                                 std::memory_order_relaxed));
	}
	t_counters = acquired;
	return *acquired;
}

} // namespace Detail

void ScopedBlock::Begin(const tracy::SourceLocationData* source_location) {
	new (m_zone) tracy::ScopedZone(source_location, TRACY_CALLSTACK, true);
	m_active = true;
	g_block_stack.push_back(this);
}

void ScopedBlock::End() noexcept {
	if (m_active) {
		std::launder(reinterpret_cast<tracy::ScopedZone*>(m_zone))->~ScopedZone();
		m_active = false;
		RemoveBlock(this);
	}
}

void EndBlock() {
	if (!FramesOnlyEnabled() && !g_block_stack.empty()) {
		g_block_stack.back()->End();
	}
}

void SetThreadName(const char* name) {
	if (name != nullptr) HangWatchdog::SetThreadName(name);
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

// KYTY_PROFILE_COUNTERS=shared: the previous counting, every call checking the switches and the
// connection and adding to one shared array.
void Detail::CountFrameWorkShared(FrameWork kind) noexcept {
	if (FramesOnlyEnabled()) {
		g_frame_work[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
	}
}

void Detail::CountFrameEventShared(FrameEvent kind, uint64_t amount) noexcept {
	if (AggregateEnabled() && tracy::ProfilerAvailable() && TracyIsConnected) {
		g_frame_events[static_cast<size_t>(kind)].fetch_add(amount, std::memory_order_relaxed);
	}
}

#ifdef TRACY_ON_DEMAND
namespace {
// The on-demand connection a scope is counted against. The Thread sink can be on without a
// started profiler (tests set it directly, and a shutdown can follow the flip that chose it), and
// with TRACY_MANUAL_LIFETIME GetProfiler() then dereferences no profiler: 0 stands for "none", as
// for a started profiler that has not been connected yet.
uint64_t CurrentConnectionId() {
	return tracy::ProfilerAvailable() ? tracy::GetProfiler().ConnectionId() : 0;
}
} // namespace
#endif

void ScopedFrameWait::Begin() {
	const auto sink = Detail::g_event_sink.load(std::memory_order_relaxed);
	if (sink == Detail::CounterSink::Shared &&
	    (!AggregateEnabled() || !tracy::ProfilerAvailable() || !TracyIsConnected)) {
		return;
	}
#ifdef TRACY_ON_DEMAND
	m_connection = CurrentConnectionId();
#endif
	m_sink     = sink;
	m_start_ns = FrameWaitClockNs();
}

void ScopedFrameWait::Finish() {
	if (m_sink == Detail::CounterSink::Shared && (!tracy::ProfilerAvailable() || !TracyIsConnected)) {
		return;
	}
#ifdef TRACY_ON_DEMAND
	// Scopes spanning an on-demand connection change are omitted.
	if (m_connection != CurrentConnectionId()) {
		return;
	}
#endif
	const auto elapsed = FrameWaitClockNs() - m_start_ns;
	const auto index   = static_cast<size_t>(m_kind);
	if (m_sink == Detail::CounterSink::Thread) {
		auto& counters = Detail::CurrentThreadCounters();
		Detail::Add(counters.wait_ns[index], elapsed);
		Detail::Add(counters.wait_calls[index], 1);
		return;
	}
	auto& totals = g_frame_waits[index];
	totals.nanoseconds.fetch_add(elapsed, std::memory_order_relaxed);
	totals.calls.fetch_add(1, std::memory_order_relaxed);
}

uint64_t FrameEventTotal(FrameEvent kind) {
	const auto index = static_cast<size_t>(kind);
	uint64_t   total = g_frame_events[index].load(std::memory_order_relaxed);
	for (auto* block = Detail::g_counter_blocks.load(std::memory_order_acquire); block != nullptr;
	     block       = block->next) {
		total += block->events[index].load(std::memory_order_relaxed);
	}
	return total;
}

void AddFrameWait(FrameWait kind, uint64_t calls, uint64_t nanoseconds) {
	const auto sink  = Detail::g_event_sink.load(std::memory_order_relaxed);
	const auto index = static_cast<size_t>(kind);
	if (sink == Detail::CounterSink::Thread) {
		auto& counters = Detail::CurrentThreadCounters();
		Detail::Add(counters.wait_ns[index], nanoseconds);
		Detail::Add(counters.wait_calls[index], calls);
		return;
	}
	if (sink == Detail::CounterSink::Off || !AggregateEnabled() || !tracy::ProfilerAvailable() ||
	    !TracyIsConnected) {
		return;
	}
	auto& totals = g_frame_waits[index];
	totals.nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
	totals.calls.fetch_add(calls, std::memory_order_relaxed);
}

namespace {
thread_local FrameWait g_gpu_wait_reason = FrameWait::GpuWaitOther;
} // namespace

ScopedGpuWaitReason::ScopedGpuWaitReason(FrameWait reason): m_previous(g_gpu_wait_reason) {
	g_gpu_wait_reason = reason;
}

ScopedGpuWaitReason::~ScopedGpuWaitReason() {
	g_gpu_wait_reason = m_previous;
}

FrameWait CurrentGpuWaitReason() noexcept {
	return g_gpu_wait_reason;
}

void PublishFrameWork() {
	if (!FramesOnlyEnabled()) {
		return;
	}
	const auto flip = g_completed_flips.fetch_add(1, std::memory_order_relaxed) + 1;
	// The totals: the shared atomics (all of them with KYTY_PROFILE_COUNTERS=shared) plus every
	// thread's block. Each category is monotonic, but this is not an atomic cross-category
	// snapshot: producer work can straddle the marker and these loads.
	const bool aggregates = tracy::ProfilerAvailable() && AggregateEnabled() && TracyIsConnected;
	Detail::Totals totals;
	Detail::SumThreadCounters(totals, aggregates);
	if (tracy::ProfilerAvailable()) {
		for (size_t i = 0; i < kFrameWorkCount; ++i) {
			const auto count = g_frame_work[i].load(std::memory_order_relaxed) + totals.work[i];
			TracyPlot(kFrameWorkNames[i], static_cast<int64_t>(count));
		}
		if (aggregates) {
			for (size_t i = 0; i < kFrameEventCount; ++i) {
				const auto count = g_frame_events[i].load(std::memory_order_relaxed) + totals.events[i];
				TracyPlot(kFrameEventNames[i], static_cast<int64_t>(count));
			}
			// Calls/time are independently atomic, not a cross-thread transaction. A scope
			// completing at this boundary can split the pair across adjacent samples.
			// Durations belong to the completion interval, including time before its marker.
			for (size_t i = 0; i < kFrameWaitCount; ++i) {
				const auto calls =
				    g_frame_waits[i].calls.load(std::memory_order_relaxed) + totals.wait_calls[i];
				const auto ns =
				    g_frame_waits[i].nanoseconds.load(std::memory_order_relaxed) + totals.wait_ns[i];
				TracyPlot(kFrameWaitCallNames[i], static_cast<int64_t>(calls));
				TracyPlot(kFrameWaitTimeNames[i], static_cast<int64_t>(ns));
			}
		}
		// Written last so the exporter can identify a complete snapshot group.
		TracyPlot("FrameWork.CompletedFlips.Cumulative", static_cast<int64_t>(flip));
	}
	// Counting for the next flip follows the connection (CounterSink).
	Detail::g_event_sink.store(Detail::CurrentEventSink(), std::memory_order_relaxed);
}

bool LoadingEnabled() {
	static const bool enabled = [] {
		const auto* setting = std::getenv("KYTY_PROFILE_LOADING");
		return setting != nullptr && std::strcmp(setting, "1") == 0;
	}();
	return enabled;
}

void CountLoadingEvent(LoadingEvent kind, uint64_t amount) {
	if (LoadingEnabled()) {
		g_loading_events[static_cast<size_t>(kind)].fetch_add(amount, std::memory_order_relaxed);
	}
}

bool BeginLoadingAprSubmission(LoadingAprSubmission* record) {
	if (!LoadingEnabled() || record == nullptr) return false;
	const auto sequence = g_loading_apr_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	if ((sequence > 512 && (sequence - 512) % 257 != 0) ||
	    g_loading_apr_retained.load(std::memory_order_relaxed) >= kLoadingAprMaxRows) {
		return false;
	}
	record->sequence = sequence;
	record->timestamp_ns = FrameWaitClockNs();
	return true;
}

void RecordLoadingAprSubmission(const LoadingAprSubmission& record) {
	if (!LoadingEnabled()) return;
	std::scoped_lock lock(g_loading_apr_mutex);
	if (g_loading_apr_size == kLoadingAprQueueSize ||
	    g_loading_apr_retained.load(std::memory_order_relaxed) >= kLoadingAprMaxRows) {
		CountLoadingEvent(LoadingEvent::AprDiagnosticRowsDropped);
		return;
	}
	g_loading_apr_queue[(g_loading_apr_head + g_loading_apr_size) % kLoadingAprQueueSize] = record;
	++g_loading_apr_size;
	g_loading_apr_retained.fetch_add(1, std::memory_order_relaxed);
}

ScopedLoadingOperation::ScopedLoadingOperation(LoadingOperation kind): m_kind(kind) {
	if (!LoadingEnabled()) return;
	m_active = true;
	m_start_ns = FrameWaitClockNs();
	auto& totals = g_loading_operations[static_cast<size_t>(m_kind)];
	totals.started.fetch_add(1, std::memory_order_relaxed);
	totals.in_flight.fetch_add(1, std::memory_order_relaxed);
}

ScopedLoadingOperation::~ScopedLoadingOperation() {
	End();
}

void ScopedLoadingOperation::End() {
	if (!m_active) return;
	m_active = false;
	auto& totals = g_loading_operations[static_cast<size_t>(m_kind)];
	totals.nanoseconds.fetch_add(FrameWaitClockNs() - m_start_ns, std::memory_order_relaxed);
	totals.completed.fetch_add(1, std::memory_order_relaxed);
	totals.in_flight.fetch_sub(1, std::memory_order_relaxed);
}

void Initialize() {
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		const auto* cache_policy = std::getenv("KYTY_IMAGE_CACHE_POLICY");
		const bool pressure_cache = cache_policy != nullptr && std::strcmp(cache_policy, "pressure") == 0;
		TracySetProgramName(pressure_cache ? "KytyPS5 - B pressure cache" : "KytyPS5 - A exact lookup");
		const auto* lookup = std::getenv("KYTY_IMAGE_LOOKUP");
		if (lookup != nullptr && std::strcmp(lookup, "legacy") == 0) {
			TracySetProgramName(pressure_cache ? "KytyPS5 - B pressure cache, legacy lookup"
			                                : "KytyPS5 - diagnostic legacy lookup");
		} else if (lookup != nullptr && std::strcmp(lookup, "verify") == 0) {
			TracySetProgramName(pressure_cache ? "KytyPS5 - B pressure cache, lookup verification"
			                                : "KytyPS5 - lookup verification");
		}
		const auto* resources = std::getenv("KYTY_RESOURCE_MATERIALIZATION");
		const auto* submissions = std::getenv("KYTY_SUBMISSION_MODE");
		if (submissions != nullptr && std::strcmp(submissions, "queued") == 0) {
			const auto* coalesce = std::getenv("KYTY_SUBMISSION_COALESCE");
			TracySetProgramName(coalesce != nullptr && std::strcmp(coalesce, "1") == 0
			                        ? "KytyPS5 - C3 coalesced submissions"
			                        : "KytyPS5 - B2 queued submissions");
		} else if (resources != nullptr && std::strcmp(resources, "optimized") == 0) {
			TracySetProgramName("KytyPS5 - A2 resource preparation");
		}
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
		if (FramesOnlyEnabled()) {
			::printf("Tracy frame-only profiling enabled (KYTY_PROFILE_FRAMES_ONLY=1); "
			         "per-call zones and detailed plots disabled\n");
			if (AggregateEnabled()) {
				::printf("Tracy aggregate CPU work/wait diagnostics enabled "
				         "(KYTY_PROFILE_AGGREGATES=1); inclusive timings overlap\n");
			}
		} else if (DetailedEnabled()) {
			::printf("Tracy detailed zones enabled (KYTY_PROFILE_DETAILS=1)\n");
		}
		if (Detail::SharedCounters()) {
			::printf("Profiler counters: shared atomics (KYTY_PROFILE_COUNTERS=shared)\n");
		}
	}
	Detail::g_zones.store(!FramesOnlyEnabled() && tracy::ProfilerAvailable(),
	                      std::memory_order_relaxed);
	Detail::g_event_sink.store(Detail::CurrentEventSink(), std::memory_order_relaxed);
	HangTrace::Initialize();
	HangWatchdog::Initialize(HangTrace::OutputDirectory());
	if (LoadingEnabled() && tracy::ProfilerAvailable() && !g_loading_publisher.joinable()) {
		try {
			g_loading_publisher = std::jthread([](std::stop_token stop) {
				tracy::SetThreadName("Loading diagnostics");
				const auto start = FrameWaitClockNs();
				auto log = OpenLoadingLog();
				auto apr_log = OpenLoadingLog(true);
				uint32_t log_rows = 0;
				constexpr uint32_t max_log_rows = 7200;
				std::unique_lock lock(g_loading_publish_mutex);
				while (!stop.stop_requested()) {
					const bool write_log = log != nullptr && std::ferror(log.get()) == 0 &&
					                       log_rows < max_log_rows;
					PublishLoadingProgress(FrameWaitClockNs() - start, write_log ? log.get() : nullptr);
					PublishLoadingAprSubmissions(apr_log.get());
					if (write_log && ++log_rows == max_log_rows) log.reset();
					g_loading_publish_condition.wait_for(lock, stop, std::chrono::seconds(1),
					                                     [] { return false; });
				}
				PublishLoadingAprSubmissions(apr_log.get());
			});
			::printf("Tracy loading progress diagnostics enabled (KYTY_PROFILE_LOADING=1); "
			         "independent one-second cumulative snapshots\n");
		} catch (const std::exception&) {
			::printf("Loading diagnostics publisher could not start; game startup continues\n");
		}
	}
}

void Shutdown() {
	HangWatchdog::Shutdown();
	HangTrace::Shutdown();
	// The publisher must finish before Tracy's global state is torn down. Its wait
	// is interruptible and uses only a private host mutex, never guest/GPU locks.
	if (g_loading_publisher.joinable()) {
		g_loading_publisher.request_stop();
		g_loading_publisher.join();
	}
	Detail::g_zones.store(false, std::memory_order_relaxed);
	Detail::g_event_sink.store(Detail::CounterSink::Off, std::memory_order_relaxed);
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
